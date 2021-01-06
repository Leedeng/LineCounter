from keras import backend as BK
from keras.models import Model
from keras.layers import Layer, Wrapper, InputSpec, Input, Activation
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, AveragePooling2D, UpSampling2D,MaxPool2D,Add
from keras.layers import GRU, Dense, Concatenate, BatchNormalization, Flatten, GlobalAveragePooling2D,ZeroPadding2D,Conv2DTranspose
from keras.initializers import glorot_uniform
from keras.utils import conv_utils
from keras.constraints import NonNeg
from keras import activations
import numpy as np 
import tensorflow as tf

class SaltPepperNoise(Layer) :
    """Noise Layer to randomly add salt-and-peper noise
    """
    def __init__(self, rate=.05, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.rate = rate
    def call(self, inputs, training=None):
        def noised():
            noise = BK.random_uniform(shape=BK.shape(inputs),
                                     minval=0.,
                                     maxval=1. )
            thresh = BK.random_uniform(shape=(BK.shape(inputs)[0],1,1,1), 
                                      minval=-self.rate,
                                      maxval=self.rate )
            noise = BK.cast( noise-thresh < 0, 'float32' )
            return BK.abs( inputs - noise )
        return BK.in_train_phase(noised, inputs, training=training)
    def get_config(self):
        config = {'rate': self.rate}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def compute_output_shape(self, input_shape):
        return input_shape

class SymmetricPadding(Layer) :
    """Custom Layer to Apply Symmetric Padding
    """
    def __init__(self, kernel_size, **kwargs) :
        if ( isinstance( kernel_size, tuple ) ) :
            self.kh, self.kw = kernel_size
        else :
            self.kh = self.kw = kernel_size
        super().__init__(**kwargs)
    def call(self, inputs) :
        ph, pw = self.kh//2, self.kw//2
        inputs_pad = tf.pad( inputs, [[0,0],[ph,ph],[pw,pw],[0,0]], mode='symmetric' )
        return inputs_pad
    def compute_output_shape(self, input_shape) :
        bsize, nrows, ncols, nfeats = input_shape
        ph, pw = self.kh//2, self.kw//2
        new_nrows = nrows + ph * 2 if nrows is not None else None
        new_ncols = ncols + pw * 2 if ncols is not None else None
        return (bsize, new_nrows, new_ncols, nfeats)
    
class DirectionalProcessing(Wrapper) :
    """Wrapper Layer to support directional processing for 4D data
    NOTE: this layer will support to apply LSTM-like layer along 
    either row/column direction
    """
    def __init__(self, rnn_layer, time_axis=1, go_backward=False, **kwargs) :
        self.time_axis = time_axis
        self.go_backward = go_backward
        super().__init__(rnn_layer, **kwargs)
    def build(self, input_shape) :
        assert len(input_shape) >= 3
        self.input_spec = InputSpec(shape=input_shape)
        if self.time_axis != 1 :
            child_input_shape = (None,) + input_shape[2:]
        else :
            child_input_shape = (None, input_shape[1]) + input_shape[3:]
        if not self.layer.built:
            self.layer.build(child_input_shape)
            self.layer.built = True
        super().build()
    def compute_output_shape(self, input_shape):
        if self.time_axis != 1 :
            child_input_shape = (None,) + input_shape[2:]
        else :
            child_input_shape = (None, input_shape[1]) + input_shape[3:]
        child_output_shape = self.layer.compute_output_shape(child_input_shape)
        return input_shape[:3] + (child_output_shape[-1],)
    def call(self, x) :
        batch_size, nb_rows, nb_cols, nb_feats = [ tf.shape(x)[k] for k in range(4) ]
        if self.time_axis == 1:
            x4d = BK.permute_dimensions(x, [0, 2, 1, 3])
            x3d = tf.reshape(x4d, [batch_size * nb_cols, nb_rows, nb_feats])
        else :
            x3d = tf.reshape(x, [batch_size * nb_rows, nb_cols, nb_feats])
        if self.go_backward :
            x3d = BK.reverse(x3d, 1)
        y3d = self.layer.call(x3d)
        if self.go_backward :
            y3d = BK.reverse(y3d, 1)
        if self.time_axis == 1 :
            y4d = tf.reshape(y3d, [batch_size, nb_cols, nb_rows, -1])
            y4d = BK.permute_dimensions(y4d, [0, 2, 1, 3])
        else :
            y4d = tf.reshape(y3d, [batch_size, nb_rows, nb_cols, -1])
        return y4d

class SamplewiseProcessing(Wrapper) :
    """Wrapper Layer to support samplewise processing
    e.g. applying each sample a different convolutional kernel
    """
    def __init__(self, 
                 param_pred_layer,
                 filters, 
                 kernel_size,
                 use_bias=True,
                 data_format='channels_last',
                 padding='same',
                 dilation_rate=(1,1),
                 strides=(1,1),
                 activation=None,
                 **kwargs) :
        self.kernel_size = kernel_size
        self.filters = filters
        self.use_bias = use_bias
        self.data_format = data_format
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.strides = strides
        self.activation = activations.get(activation)
        super().__init__(param_pred_layer, **kwargs)
    def build(self, input_shapes) :
        input_shape = input_shapes[0]
        channel_axis = -1
        input_dim = input_shape[channel_axis]
        self.kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.depthwise_kernel_shape = self.kernel_size + (input_dim, 1)
        self.pointwise_kernel_shape = (1,) * 2 + (1 * input_dim, self.filters)
        param_indices = [np.product(self.depthwise_kernel_shape),
                         np.product(self.pointwise_kernel_shape)]        
        if self.use_bias:
            self.bias_shape = (self.filters,)
            param_indices.append(self.filters)
        else:
            self.bias_shape = None
        self.param_indices = np.cumsum(param_indices)
        print("need", param_indices)
        self.initializer = BK.placeholder(shape=Conv2D.compute_output_shape(self, input_shape))
        self.built = True
    def samplewise_call(self, acc, inputs) :
        # parse inputs
        if len(inputs) == 4 :
            x, depth_kernel, point_kernel, bias = inputs
        elif ( len(inputs) == 3 ) :
            x, depth_kernel, point_kernel = inputs
        elif ( len(inputs) == 2 ) :
            x, params = inputs
            d0, d1 = 0, self.param_indices[0]
            depth_kernel = params[d0:d1]
            p0, p1 = self.param_indices[:2]
            point_kernel = params[p0:p1]
            if self.use_bias :
                bias = params[p1:]
        else :
            raise NotImplementedError
        # reshape to conv kernels
        depth_kernel = BK.reshape(depth_kernel, self.depthwise_kernel_shape)
        point_kernel = BK.reshape(point_kernel, self.pointwise_kernel_shape)
        if (self.use_bias) :
            bias = BK.reshape(bias, self.bias_shape)
        outputs = BK.separable_conv2d(BK.expand_dims(x,axis=0),
                                     depth_kernel,
                                     point_kernel,
                                     data_format=self.data_format,
                                     strides=self.strides,
                                     padding=self.padding,
                                     dilation_rate=self.dilation_rate)
        if self.use_bias:
            outputs = BK.bias_add( outputs,
                                  bias,
                                  data_format=self.data_format)            
        if self.activation is not None:
            return self.activation(outputs)
        else :
            return outputs
    def call(self, inputs) :
        input_x_to_conv, input_y_to_pred_kernel = inputs
        params = self.layer(input_y_to_pred_kernel)
        outputs = tf.scan(fn=self.samplewise_call, 
                          elems=[input_x_to_conv,params],
                          initializer=self.initializer)
        outputs = BK.squeeze(outputs,1)
        return outputs
    def compute_output_shape(self, input_shapes):
        input_shape = input_shapes[0]
        output_shape = list(input_shape)
        c_axis, h_axis, w_axis = 3, 1, 2
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides
        output_shape[c_axis] = self.filters
        output_shape[h_axis] = conv_utils.deconv_length(
            output_shape[h_axis], stride_h, kernel_h, self.padding, None) 
        output_shape[w_axis] = conv_utils.deconv_length(
            output_shape[w_axis], stride_w, kernel_w, self.padding, None) 
        return tuple(output_shape)

class CounterWrapper(Wrapper) :
    """Counter Wrapper to support monotonic increasing output
    """
    def __init__(self, layer, counter_axis=1, activation=None, **kwargs) :
        self.counter_axis = counter_axis
        self.activation = activations.get(activation) if (activation is not None) else None
        super().__init__(layer, **kwargs)
    def build(self, input_shape) :
        if not self.layer.built :
            self.layer.build(input_shape)
            self.built = True
        super().build()
    def call(self, x) :
        y = self.layer.call(x)
        if self.activation is not None :
            y = self.activation(y)
        y64 = BK.cast(y, 'float64')
        y64 = BK.cumsum(y64, axis=self.counter_axis)
        y32 = BK.cast(y64, 'float32')
        return y32
    
def net_in_net_param_pred(input_shape, conv_input_dim, conv_filters, conv_kernel_size, conv_use_bias, num_blocks, num_filters=4) :
    """create a network in network to predict samplewise convolutional kernel
    
    #Arugments:
        input_shape: input 4D tensor to estimate the parameters
        num_blocks: number of convolutional blocks used in estimation
        conv_input_dim: Conv2D layer's input dimension (see def in Conv2D.build())
        conv_filters: Conv2D layer's filters argument
        conv_kernel_size: Conv2D layer's kernel_size argument
        conv_use_bias: Conv2D layer's use_bias arugment
    """
    x = Input(shape=input_shape, name='netNnet_in')
    f = x
    for block in range(num_blocks) :
        f = convbn(f, num_filters * (2**block), (3,3), (1,1), name='netNnet_c{}'.format(block+1))
        f = MaxPooling2D((2,2), name='netNnet_m{}'.format(block+1))(f)
    f = GlobalAveragePooling2D(name='netNnet_p')(f)
    # compute number of required nodes
    num_nodes_depth_kernel = np.product(conv_kernel_size) * conv_input_dim
    num_nodes_point_kernel = conv_input_dim * conv_filters
    num_nodes_bias = 0 if (not conv_use_bias) else conv_filters
    total_num_nodes = num_nodes_depth_kernel + num_nodes_point_kernel + num_nodes_bias
    f = Dense(total_num_nodes, name='netNnet_params')(f)
    mm = Model(inputs=x, outputs=f, name='netNnet')
    print(mm.summary())
    return mm


def convbn(x, filters, kernel_size=(3,3), strides=(1,1), padding='same', use_bn=True, name='module', **kwargs ) :
    """Shortcut for Conv2D+BN+ReLU
    """
    if padding == 'symmetric' :
        y = SymmetricPadding(kernel_size, name=name+'_spad')(x)
        y = Conv2D(filters,
               kernel_size,
               activation=None,
               padding='valid',
               strides=strides,
               name=name+'_conv',
               **kwargs)(y)
    else :
        y = Conv2D(filters,
                   kernel_size,
                   activation=None,
                   padding=padding,
                   strides=strides,
                   name=name+'_conv',
                   **kwargs)(x)
    if use_bn :
        y = BatchNormalization(axis=-1, name=name+'_bnorm')(y)
    y = Activation('relu',name=name+'_relu')(y)
    return y

def learning_to_count(x,
                      base,
                      kernel_size=(3,3), 
                      activation='hard_sigmoid',
                      use_sympadding=False,
                      name='count') :
    if use_sympadding :
        y = SymmetricPadding(kernel_size, name=name+'_spad')(x)
        y = CounterWrapper(Conv2D(base, 
                              kernel_size=kernel_size, 
                              padding='valid', 
                              activation=activation, 
                              name=name+'_core'), 
                       name=name+'_pred')(y)
    else :
        y = CounterWrapper(Conv2D(base, 
                              kernel_size=kernel_size, 
                              padding='same', 
                              activation=activation, 
                              name=name+'_core'), 
                       name=name+'_pred')(x)            
    return y

def encoder_pass(x, base, 
                 num_conv_blocks=5, 
                 use_bn=True, 
                 downsampling='drop',
                 kernel_size_a=(3,3),
                 kernel_size_b=(3,5),
                 use_sympadding=False) :
    """ResNet/VGG-backbone encoder
    """
    f = x
    padding = 'symmetric' if use_sympadding else 'same'
    # blockwise encoding process
    for block in range(num_conv_blocks) :
        f = convbn(f, 
                   base, 
                   kernel_size_a, 
                   (1,1), 
                   padding=padding, 
                   name='encoder_{}a'.format(block+1))
        if downsampling == 'drop' :
            f = convbn(f, 
                       base, 
                       kernel_size_b, 
                       (2,2), 
                       padding=padding, 
                       name='encoder_{}b'.format(block+1))
        else :
            f = convbn(f, 
                       base, 
                       kernel_size_b, 
                       (1,1), 
                       padding=padding,  
                       name='encoder_{}b'.format(block+1))
            if downsampling == 'average' :
                f = AveragePooling2D((2,2), name='encoder_{}p'.format(block+1))(f)
            elif downsampling == 'max' :
                f = MaxPooling2D((2,2), name='encoder_{}p'.format(block+1))(f)
        base *= 2
    return f



def decoder_pass(x, base, 
                 num_conv_blocks=5, 
                 use_bn=False, 
                 upsampling='bilinear',
                 kernel_size_a=(3,3),
                 kernel_size_b=(3,5),
                 use_sympadding=False) :
    """ResNet/VGG-backbone decoder
    """
    f = x
    padding = 'symmetric' if use_sympadding else 'same'
    base = base * (2 ** (num_conv_blocks-1))
    # blockwise decoding process
    for block in range(num_conv_blocks) :
        f = UpSampling2D(size=(2,2), 
                         interpolation=upsampling, 
                         name='decoder_{}p'.format(block+1))(f)
        f = convbn(f, 
                   base, 
                   kernel_size_b, 
                   (1,1), 
                   padding=padding, 
                   use_bn=use_bn, 
                   name='decoder_{}a'.format(block+1))
        f = convbn(f, 
                   base, 
                   kernel_size_a, 
                   (1,1), 
                   padding=padding,
                   use_bn=use_bn, 
                   name='decoder_{}b'.format(block+1))        
        base //= 2
    return f



def line_num_propagation(x, filters, 
                         activation='tanh', 
                         use_samplewise_conv=False, 
                         bidirectional=False, 
                         num_blocks=2) :
    """Line counter propagations with support to samplewise convolution
    """
    # propagate ${lineNum} along the horizontal direction
    colGRU = GRU(filters, activation='tanh', return_sequences=True, name='colGRU')
    f_lr = DirectionalProcessing(colGRU,
                                 time_axis=2,
                                 go_backward=False,
                                 name='prog_lr')(x)
    f_rl = DirectionalProcessing(colGRU,
                                 time_axis=2,
                                 go_backward=True,
                                 name='prog_rl')(x)
    f_col = Concatenate(axis=-1, name='prog_along_col')([f_lr, f_rl])
    # count ${lineNum}
    rowGRU = GRU(filters, activation=activation, return_sequences=True, name='rowGRU')
    if use_samplewise_conv :
        ParamPred = net_in_net_param_pred(input_shape=BK.int_shape(x)[1:], 
                                          conv_input_dim=BK.int_shape(f_col)[-1], 
                                          conv_filters=filters, 
                                          conv_kernel_size=(3,3), 
                                          conv_use_bias=True, 
                                          num_blocks=num_blocks, 
                                          num_filters=filters//2)
        # samplewise convolution
        f_row = SamplewiseProcessing(param_pred_layer=ParamPred,
                                 filters=filters, 
                                 kernel_size=(3,3),
                                 use_bias=True,
                                 data_format='channels_last',
                                 padding='same',
                                 dilation_rate=(1,1),
                                 strides=(1,1),
                                 activation=None,
                                 name='samplewise_prog_conv')([f_col, x])
    else :
        f_row = convbn(f_col, filters, kernel_size=(3,3), name='prog_conv')
    f_ud = DirectionalProcessing(rowGRU,
                                 time_axis=1,
                                 go_backward=False,
                                 name='prog_ud')(f_row)
    if bidirectional :
        f_du = DirectionalProcessing(rowGRU,
                                 time_axis=1,
                                 go_backward=True,
                                 name='prog_du')(f_row)
        # be careful in merge two passes
        f_row = Concatenate(axis=-1, name='prog_along_row')([f_ud, f_du])
        f_row = convbn(f_row, filters, kernel_size=(3,3))
        return f_row
    else :
        return f_ud
    
    


class Maximum(Layer):

    def __init__(self, **kwargs):
        #self.output_dim = output_dim
        super(Maximum, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        
        # A = ONLY TEXT RESULT
        # b = ALL IMAGE INCLUDING TEXT AND BACKGROUND
        #super(MyLayer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        assert isinstance(x, list)
        a, b = x
        max_value = BK.max(a) 
        T = BK.minimum(max_value,b)
        #T = BK.maximum(1.,b)
        return T

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return  shape_b
    
"""
Line counter LSTM training script

NOTE: you need to specify GPU node in commend line

i.e. CUDA_VISIBLE_DEVICES=0 python trainLineCounter.py arg1 arg2 ...

v2: 
- support symmetric padding, 
- add func learning_to_count
- fix bug "after decoder"

"""
import numpy as np 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' 
import sys
import keras
import argparse
import parse
from typing import Tuple, List, Iterator, Any, Dict
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from src import CLLayers
from src import CLLosses
from src import CLUtils
from keras.models import Model
from keras.optimizers import Adam
from keras.constraints import *
from keras.models import load_model
import random
import cv2
DataGenerator3 = CLUtils.DataGenerator3
DataGenerator2 = CLUtils.DataGenerator2
DataGenerator = CLUtils.DataGenerator
load_image_files = CLUtils.load_image_files
counter_loss = CLLosses.seg
counter_acc = CLLosses.acc
IOU_calc = CLLosses.IOU_calc
MatchScore = CLLosses.MatchScore
import tensorflow as tf
################################################################################
# Data Utils
################################################################################
def load_dataset(train_file_list: str,
                 valid_file_list: str, 
                 batch_size: int=4,
                 target_size: Tuple[int,int]=(768,768),
                 is_input_binary: bool=False) -> Tuple[Iterator[Tuple[np.ndarray, np.ndarray]],
                                                       Iterator[Tuple[np.ndarray, np.ndarray]]]:
    """Load training data from disk
    """
    if not (is_input_binary) :
        raise NotImplementedError("ERROR: handling inputs for grayscale/color images are NOT implemented")
        
    train_file_pairs = load_image_files(train_file_list)
    valid_file_pairs = load_image_files(valid_file_list)
    #print(train_file_pairs)
    train_datagen = DataGenerator3( train_file_pairs, 
                                   batch_size=batch_size, 
                                   nb_batches_per_epoch=None, 
                                   mode='training', 
                                   min_scale=.5, 
                                   seed = 123, 
                                   pad=32, 
                                   target_size = target_size,
                                   use_mirror=False)
    
    
    valid_datagen = DataGenerator3( valid_file_pairs, 
                                   batch_size=batch_size, 
                                   nb_batches_per_epoch=None, 
                                   mode='validation', 
                                   min_scale=.5, 
                                   seed = 123, 
                                   pad=32,
                                   target_size = target_size,
                                   use_mirror=False)
   
    return train_datagen, valid_datagen


################################################################################
# Name Utils
################################################################################
MODEL_NAME_FORMAT = ":".join(["BF{base:d}", 
                              "BLK{num_conv_blocks:d}",
                              "BN{use_encoder_bn:d},{use_decoder_bn:d}",
                              "M{counter_multiplier:d}",
                              "LA{activation}",
                              "LC{counter_location}",
                              "SC{use_samplewise_conv:d}",
                              "DS{downsampling_method}",
                              "US{upsampling_method}",
                              "BD{bidirectional:d}",
                              "N{noise_rate:.2f}",
                              "P{use_sympadding:d}"
                              ])

def spell_model_name(base: int=8,
                 counter_multiplier: int=8,
                 activation: str='tanh',
                 counter_location: str=None,
                 num_conv_blocks: int=5,
                 use_encoder_bn: bool=True,
                 use_decoder_bn: bool=True,
                 use_samplewise_conv: bool=False,
                 downsampling_method: str='drop',
                 upsampling_method: str='bilinear',
                 bidirectional: bool=False,
                 noise_rate: float=.05,
                 use_sympadding: bool=False) -> str :
    return MODEL_NAME_FORMAT.format(base=base,
                 counter_multiplier=counter_multiplier,
                 activation=activation,
                 counter_location=counter_location,
                 num_conv_blocks=num_conv_blocks,
                 use_encoder_bn=use_encoder_bn,
                 use_decoder_bn=use_decoder_bn,
                 use_samplewise_conv=use_samplewise_conv,
                 downsampling_method=downsampling_method,
                 upsampling_method=upsampling_method,
                 bidirectional=bidirectional,
                 noise_rate=noise_rate,
                 use_sympadding=use_sympadding)


def parse_model_name(model_name: str) -> Dict[str, Any]:
    ret = parse.parse(MODEL_NAME_FORMAT, model_name)
    return ret.named

################################################################################
# Callback Utils
################################################################################
def prepare_model_callbacks(model_name, expt_dir, patience=10) :
    os.system('mkdir -p {}'.format(expt_dir))
    os.system('mkdir -p {}/models/{}'.format(expt_dir, model_name))
    os.system('mkdir -p {}/logs/{}'.format(expt_dir, model_name))
    my_callbacks = [ModelCheckpoint(os.path.join(expt_dir, 'models', model_name) + "/" + model_name + '-{val_acc:.4f}.h5', 
                                    monitor='val_acc', 
                                    verbose=1, 
                                    save_best_only=True), 
                    ReduceLROnPlateau('val_acc', factor=0.5, patience=patience, verbose=1), 
                    TensorBoard(log_dir=os.path.join(expt_dir, 'logs', model_name), 
                                histogram_freq=0, 
                                write_grads=False, 
                                write_images=False ),
                    EarlyStopping(monitor='val_acc', patience=3*patience)]
    return my_callbacks

################################################################################
# Model Definition
################################################################################

def create_model_v2(base: int=8,
                    counter_multiplier: int=8,
                    activation: str='tanh',
                    counter_location: str=None,
                    num_conv_blocks: int=5,
                    use_encoder_bn: bool=True,
                    use_decoder_bn: bool=True,
                    use_samplewise_conv: bool=False,
                    downsampling_method: str='drop',
                    upsampling_method: str='bilinear',
                    bidirectional: bool=False,
                    noise_rate: float=.05,
                    kernel_size_a: Tuple[int, int]=(3,3),
                    kernel_size_b: Tuple[int, int]=(3,5),
                    use_sympadding: bool=False,
                    input_shape=(None, None, 1),
                    ) :
    # 1. input
    img = CLLayers.Input(shape=input_shape, name='doc_in')
    # 1.a Add noise if necesary
    if noise_rate > 1e-4 :
        f = CLLayers.SaltPepperNoise(rate=noise_rate, name='noise')(img)
    else :
        f = img    
    # 2. encoder pass
    f = CLLayers.encoder_pass(img, base, 
                     num_conv_blocks=num_conv_blocks, 
                     use_bn=use_encoder_bn, 
                     downsampling=downsampling_method,
                     kernel_size_a=kernel_size_a,
                     kernel_size_b=kernel_size_b,
                     use_sympadding=use_sympadding)
    # 3. line propagation
    f = CLLayers.line_num_propagation(f, 
                             base*counter_multiplier, 
                             activation=activation,
                             use_samplewise_conv=use_samplewise_conv,
                             bidirectional=bidirectional,
                             num_blocks=2)
    # 4. insert counter if necessary
    if (counter_location == 'before_decoder') :
        f = CLLayers.learning_to_count(f, 
                                       base*counter_multiplier, 
                                       kernel_size=(3,3), 
                                       use_sympadding=use_sympadding, 
                                       activation='hard_sigmoid', 
                                       name='count')
    # 5. decoder pass
    f = CLLayers.decoder_pass(f, base, 
                     num_conv_blocks=num_conv_blocks, 
                     use_bn=use_decoder_bn, 
                     upsampling=upsampling_method,
                     kernel_size_a=kernel_size_a,
                     kernel_size_b=kernel_size_b,
                     use_sympadding=use_sympadding)
    # 6. final prediction
    if (counter_location == 'after_decoder') :
        f = CLLayers.learning_to_count(f, 
                                       base, 
                                       kernel_size=(3,3), 
                                       use_sympadding=use_sympadding, 
                                       activation='hard_sigmoid', 
                                       name='count')
        out = CLLayers.convbn(f, 1, (3,3), 
                              padding='symmetric' if use_sympadding else 'same', 
                              use_bn=use_decoder_bn,
                              name='pred')        

    elif (counter_location == 'final') :
        f = CLLayers.learning_to_count(f, 
                                       base, 
                                       kernel_size=(3,3), 
                                       use_sympadding=use_sympadding, 
                                       activation='hard_sigmoid', 
                                       name='count')
        out = CLLayers.convbn(f, 1, (3,3), 
                              padding='symmetric' if use_sympadding else 'same', 
                              use_bn=False,
                              kernel_constraint=NonNeg(),
                              name='pred',)        
    else :
        # None
        f = CLLayers.convbn(f, 1, (3,3), 
                              padding='symmetric' if use_sympadding else 'same', 
                              use_bn=use_decoder_bn,
                              name='pred')
        max_value = keras.layers.Multiply()([f, img])
        out = CLLayers.Maximum()([max_value, f])
    model_name = spell_model_name(base,
                 counter_multiplier,
                 activation,
                 counter_location,
                 num_conv_blocks,
                 use_encoder_bn,
                 use_decoder_bn,
                 use_samplewise_conv,
                 downsampling_method,
                 upsampling_method,
                 bidirectional,
                 noise_rate,
                 use_sympadding, )
    model = Model(inputs=img, outputs=out, name=model_name)

    return model


#def create_loss() :
    # 1. input
    #img = CLLayers.Input(shape=(None, None, 1), name='doc_in')
    #gt = CLLayers.Input(shape=(None, None, 1), name='gt_in')

    #loss = CLLayers.Robustloss()([img, gt])
    
    #model = Model(inputs=[img,gt], outputs=loss, name="loss")
    #return model
################################################################################
# Main 

################################################################################
if __name__ == '__main__' :
    parser = argparse.ArgumentParser(prog='LineCounterTrainer',
                                     description='this is the training script for the line counter segmenation')
    parser.add_argument('train_list', type=str, help='the file list of (image,gt) for training')
    parser.add_argument('valid_list', type=str, help='the file list of (image,gt) for validation')        
    parser.add_argument('--baseFilter', dest='base', type=int, default=8,
                        help='the number of base filters used in encoder/decoder')
    parser.add_argument('--numConvBlock', dest='num_conv_blocks', type=int, default=5,
                        help='the number of conv blocks in encoder/decoder')
    parser.add_argument('--useEncoderBN', dest='use_encoder_bn', 
                        action='store_true', default=False,
                        help='whether or not use batch normalization in encoder')
    parser.add_argument('--useDecoderBN', dest='use_decoder_bn', 
                        action='store_true', default=False,
                        help='whether or not use batch normalization in decoder')
    parser.add_argument('--counterMultiplier', dest='counter_multiplier', type=int, 
                        default=8, help='${counterMultiplier} times of ${baseFilters} used in counter')
    parser.add_argument('--activation', dest='activation', type=str, 
                        default='tanh', help='activation used in counter')
    parser.add_argument('--counterLocation', dest='counter_location', type=str, 
                        default="None", help='where to apply the counter monotonic constraint')
    parser.add_argument('--useAdaptation', dest='use_samplewise_conv', action='store_true',
                        default=False, help='whether or not apply page adaptation')
    parser.add_argument('--downsampling', dest='downsampling_method', type=str,
                        default='drop', help='downsampling method {drop|average|max}')
    parser.add_argument('--upsampling', dest='upsampling_method', type=str,
                        default='bilinear', help='upsampling method {nearest|bilinear}')
    parser.add_argument('--useBidirectional', dest='bidirectional', action='store_true',
                        default=False, help='whether or not apply bidirectional processing')
    parser.add_argument('--useSympadding', dest='use_sympadding', action='store_true',
                        default=False, help='whether or not apply symmetric padding in convolutions')    
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--epoch', dest='epoch', type=int, default=500, help='number of epoches to train')
    parser.add_argument('--patience', dest='patience', type=int, default=10, help='half learning rate if no ${patience} improves')
    parser.add_argument('--exptDir', dest='expt_dir', type=str, default='./expts', help='where to save experiment dir')
    parser.add_argument('--trainSize', dest='target_size', type=int, default=768, help='network input size')
    parser.add_argument('--target_size_height', dest='target_size_height', type=int, default=768, help='network input size')
    parser.add_argument('--target_size_width', dest='target_size_width', type=int, default=768, help='network input size')
    parser.add_argument('--learningRate', dest='lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--noiseRate', dest='noise_rate', type=float, default=.05, help='additive salt & noise')
    parser.add_argument('--verbose', dest='verbose', default=False, action='store_true', help='print verbose info')
    # 1. parse arguments
    args = parser.parse_args()
    if args.verbose :
        
        print("INFO: use GPU=", os.environ["CUDA_VISIBLE_DEVICES"])
        print("INFO: Input arguments")
        for arg in vars(args) :
            print("\t{right_aligned:>20} = {left_aligned:<20}".format(right_aligned=arg, left_aligned=getattr(args, arg)))
            
    # 2. prepare training data
    train_datagen, valid_datagen = load_dataset( train_file_list=args.train_list,
                                                 valid_file_list=args.valid_list, 
                                                 batch_size=args.batch_size,
                                                 target_size=(args.target_size_height,args.target_size_width),
                                                 is_input_binary=True) 
    # 3. prepare training model
    model = create_model_v2(base=args.base,
                            counter_multiplier=args.counter_multiplier,
                            activation=args.activation,
                            counter_location=args.counter_location,
                            num_conv_blocks=args.num_conv_blocks,
                            use_encoder_bn=args.use_encoder_bn,
                            use_decoder_bn=args.use_decoder_bn,
                            use_samplewise_conv=args.use_samplewise_conv,
                            downsampling_method=args.downsampling_method,
                            upsampling_method=args.upsampling_method,
                            bidirectional=args.bidirectional,
                            noise_rate=args.noise_rate,
                            use_sympadding=args.use_sympadding
                            )  
    model.load_weights("expts/2022/baseline_ICDAR_1088x768xnewlayer/models/BF8:BLK5:BN0,0:M8:LArelu:LCbefore_decoder:SC0:DSdrop:USbilinear:BD0:N0.05:P1/BF8:BLK5:BN0,0:M8:LArelu:LCbefore_decoder:SC0:DSdrop:USbilinear:BD0:N0.05:P1-0.0000.h5")
    #model = load_model("expts/2022/baseline_384x544/models/BF8:BLK5:BN0,0:M8:LAtanh:LCNone:SC0:DSdrop:USbilinear:BD0:N0.05:P1/BF8:BLK5:BN0,0:M8:LAtanh:LCNone:SC0:DSdrop:USbilinear:BD0:N0.05:P1-0.8245.h5")
    #model.load_weights("expts/2023/ICDAR_1088x768xnewloss_origin/models/BF8:BLK5:BN0,0:M8:LArelu:LCbefore_decoder:SC0:DSdrop:USbilinear:BD0:N0.05:P1/BF8:BLK5:BN0,0:M8:LArelu:LCbefore_decoder:SC0:DSdrop:USbilinear:BD0:N0.05:P1-0.8892.h5")
    model_name = model.name


    
    if args.verbose :
        print(model.summary(line_length=120))
        print("-"*100)
        print("INFO: save model to", os.path.join(args.expt_dir, 'models', model_name))
        print("-"*100)
    # 4. prepare training utils
    my_callbacks = prepare_model_callbacks(model_name, args.expt_dir, patience=args.patience)
    #Rloss = CLLosses.RobustAdaptativeLoss()
    # 5. training
    #model.compile(loss=CLLosses.Robustloss, optimizer=Adam(args.lr), metrics=[counter_acc,CLLosses.alpha,CLLosses.scale])
    model.compile(loss=CLLosses.Robustloss, optimizer=Adam(args.lr), metrics=[counter_acc])

    #############
 
    model.fit_generator(train_datagen, train_datagen.nb_batches_per_epoch, 
                        epochs=args.epoch,
                        validation_data=valid_datagen, 
                        validation_steps=valid_datagen.nb_batches_per_epoch, 
                        callbacks=my_callbacks, 
                        initial_epoch=0,
                        max_queue_size=8, 
                        workers=4,
                        verbose=1)
    ############
    '''
    try :
        model.fit_generator(train_datagen, train_datagen.nb_batches_per_epoch, 
                        epochs=args.epoch,
                        validation_data=valid_datagen, 
                        validation_steps=valid_datagen.nb_batches_per_epoch, 
                        callbacks=my_callbacks, 
                        initial_epoch=0,
                        max_queue_size=8, 
                        workers=4 )
        
    except Exception as e:
        print("-"*100)
        print("ERROR: model name", model_name)
        print("INFO: Input arguments")
        for arg in vars(args) :
            print("\t{right_aligned:>20} = {left_aligned:<20}".format(right_aligned=arg, left_aligned=getattr(args, arg)))
        print("Reason:", e)
        print('-'*100)
   ''' 

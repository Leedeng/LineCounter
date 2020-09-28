from keras.metrics import mse, mae
from scipy.optimize import linear_sum_assignment
from keras import backend as K
import numpy as np 
import tensorflow as tf


import os
import sys

def lineClf_np( y_true, y_pred, th=1 ) :
    #y_true = assign_label_np( y_true, y_pred )
    zz = np.round(y_pred)
    yy = np.round(y_true)
    return np.mean( ( zz == yy )[yy >=th] ).astype('float32')

def acc( y_true, y_pred, th=1 ) :
    #y_true = assign_label( y_true, y_pred )
    #y_true = K.stop_gradient( y_true )    
    y_pred_int = K.round( y_pred )
    
    y_true_int = K.round( y_true )
    mask = K.cast( y_true>=th, 'float32' )
    matched = K.cast( K.abs(y_pred_int-y_true_int) < .1, 'float32' ) * mask
    return K.sum( matched, axis=(1,2,3) ) / K.sum( mask, axis=(1,2,3) )

def MatchScore( y_true, y_pred, th=1 ) :
    
    y_pred_int = K.round( y_pred )
    y_true_int = K.round( y_true )
    mask = K.cast( y_true>=th, 'float32' )
    matched = K.cast( K.abs(y_pred_int-y_true_int) ==0, 'float32' ) * mask
    return K.sum( matched, axis=(1,2,3) ) / K.sum( mask, axis=(1,2,3) )
def acc0( y_true, y_pred, th=0 ) :
    y_pred_int = K.round( y_pred )
    y_true_int = K.round( y_true )
    mask = K.cast( K.abs(y_true-th)<.1, 'float32' )
    matched = K.cast( K.abs(y_pred_int-y_true_int) < .1, 'float32' ) * mask
    return K.sum( matched, axis=(1,2,3) ) / K.sum( mask, axis=(1,2,3) )

def assign_label( y_true, y_pred ) :
    y_modi = tf.py_func( assign_label_np, [ y_true, y_pred ], 'float32', stateful=False )
    y_modi.set_shape( K.int_shape( y_true ) ) 
    return y_modi

def assign_label_np( y_true, y_pred ) :
    mask = ( y_true > 0 )
    y_pred = np.round( y_pred ).astype('int')
    y_true = np.round( y_true ).astype('int')
    y_modi = []
    for idx in range( len(y_pred) ):
        p = y_pred[idx]
        t = y_true[idx]
        #print "-" * 50
        #print "idx=", idx
        true_line_labels = filter( lambda v : v>0, np.unique(t) )
        pred_line_labels = filter( lambda v : v>0, np.unique(p) )
        print ("true_line_labels", true_line_labels)
        print ("pred_line_labels", pred_line_labels)
        mat = []
        for tl in true_line_labels :
            mm = t==tl
            row = []
            for pl in pred_line_labels :
                vv = -np.sum(p[mm] == pl)
                row.append(vv)
            mat.append( row )
        mat = np.row_stack(mat)
        row_ind, col_ind = linear_sum_assignment( mat )
        true_ind = [ true_line_labels[k] for k in row_ind ]
        pred_ind = [ pred_line_labels[k] for k in col_ind ]
        for tl, pl in zip( true_ind, pred_ind ) :
            t[ t==tl ] = pltf.compat.v1.disable_eager_execution()
            #print "assign", tl, "to", pltf.compat.v1.disable_eager_execution()
        y_modi.append( np.expand_dims(t,axis=0))
    return np.concatenate(y_modi).astype('float32')

def seg( y_true, y_pred, th=1 ) :
    loss = K.maximum( K.square( y_true - y_pred ), K.abs( y_true - y_pred ) ) 
    mask = K.cast( y_true>=th, 'float32' )
    return K.sum( mask * loss,axis=(1,2,3)) / K.sum( mask, axis=(1,2,3))






def prepare_group_loss_numpy(y_true, y_pred) :
    """Implement group loss in Sec3.3 of https://arxiv.org/pdf/1611.05424.pdf
    NOTE: the 2nd term of the loss has been simplified to the closest mu
    """
    within, between = [], []
    for a_true, a_pred in zip(y_true, y_pred) :
        N = int(a_true.max())
        
        within_true = np.zeros_like(a_true)
        #between_true = np.ones_like(a_true) * N
        between_true = np.zeros_like(a_true) 
        masks = []
        mu_list = []
        for line_idx in range(1, N+1) :
            #mask = np.abs(a_true - line_idx) < 0.1
            mask = (a_true==line_idx)
            vals = a_pred[mask]
            mu = vals.mean()
            within_true[mask] = mu
            mu_list.append(mu)
            masks.append(mask)
        mu_arr = np.array(mu_list)

       
        for mask, mu in zip(masks, mu_arr):
            #indices = np.argsort(np.abs(mu_arr - mu)) 
            ind_mu = np.where(mu_arr==mu)
            ind_mu = int(ind_mu[0])
            closest_mu = mu_arr[np.minimum(ind_mu+1,len(mu_arr)-1)]
    
            between_true[mask] = closest_mu
        # update output
        within.append(within_true)
        between.append(between_true)
  
    return np.stack(within, axis=0), np.stack(between, axis=0)

def grouping_loss(y_true, y_pred, sigma=0.025) :
    y_within, y_between = tf.py_func(prepare_group_loss_numpy,
                                     [y_true,y_pred],
                                     [tf.float32, tf.float32])
    y_within.set_shape(K.int_shape(y_true))
    y_between.set_shape(K.int_shape(y_true))
    y_within = K.stop_gradient(y_within)
    y_between = K.stop_gradient(y_between)
    mask = K.cast(y_true >=1, 'float32')
    diff = (y_within-y_pred)**2 + tf.exp(-(y_between-y_pred)**2/(2.*sigma))
    #diff =  tf.exp(-(y_between-y_pred)**2/(2.*sigma))
    loss = K.sum(diff * mask) / (K.sum(mask))
    
    origin = K.maximum( K.square( y_true - y_pred ), K.abs( y_true - y_pred ) ) 

    origin_loss =  K.sum( mask * origin,axis=(1,2,3)) / K.sum( mask, axis=(1,2,3))
    return origin_loss 
def MSE_loss(y_true, y_pred, sigma=0.025) :
    y_within, y_between = tf.py_func(prepare_group_loss_numpy,
                                     [y_true,y_pred],
                                     [tf.float32, tf.float32])
    y_within.set_shape(K.int_shape(y_true))
    y_between.set_shape(K.int_shape(y_true))
    y_within = K.stop_gradient(y_within)
    y_between = K.stop_gradient(y_between)
    mask = K.cast(y_true >=1, 'float32')
    diff = (y_within-y_pred)**2 + tf.exp(-(y_between-y_pred)**2/(2.*sigma))
    #diff =  tf.exp(-(y_between-y_pred)**2/(2.*sigma))
    loss = K.sum(diff * mask) / (K.sum(mask))
   
    
    origin =  K.square( y_true - y_pred )

    origin_loss =  K.sum( mask * origin,axis=(1,2,3)) / K.sum( mask, axis=(1,2,3))
    return origin_loss 

def MAE_loss(y_true, y_pred, sigma=0.025) :
    y_within, y_between = tf.py_func(prepare_group_loss_numpy,
                                     [y_true,y_pred],
                                     [tf.float32, tf.float32])
    y_within.set_shape(K.int_shape(y_true))
    y_between.set_shape(K.int_shape(y_true))
    y_within = K.stop_gradient(y_within)
    y_between = K.stop_gradient(y_between)
    mask = K.cast(y_true >=1, 'float32')
    diff = (y_within-y_pred)**2 + tf.exp(-(y_between-y_pred)**2/(2.*sigma))
    #diff =  tf.exp(-(y_between-y_pred)**2/(2.*sigma))
    loss = K.sum(diff * mask) / (K.sum(mask))
    origin =  K.abs( y_true - y_pred )
    origin_loss =  K.sum( mask * origin,axis=(1,2,3)) / K.sum( mask, axis=(1,2,3))
    return origin_loss 






class RobustAdaptativeLoss(object):
  def __init__(self):
    z = np.array([[0]])
    self.v_alpha = K.zeros(shape=(1088, 768, 1))
    self.v_scale = K.zeros(shape=(1088, 768, 1))

  def loss(self, y_true, y_pred, **kwargs):
    mask = K.cast(y_true >=1, 'float32')
    x = y_true - y_pred*mask
    #origin_loss =  K.sum( mask * x,axis=(1,2,3)) / K.sum( mask, axis=(1,2,3))
    
    #x = K.reshape(x, shape=(-1, 1))
    lossfun =  robust_loss.adaptive.AdaptiveImageLossFunction((1088, 768, 1), float_dtype='float32',color_space='RGB',representation='PIXEL')
    alpha = lossfun.alpha()
    scale = lossfun.scale()
    #loss, alpha, scale = robust_loss.adaptive.AdaptiveLossFunction(num_channels=1,float_dtype="float32")
    a = K.update(self.v_alpha, alpha)
    s = K.update(self.v_scale, scale)
    # The alpha update must be part of the graph but it should
    # not influence the result.
    
    #mask = K.cast(y_true >=1, 'float32')
    origin = lossfun(x)
    #origin_loss =  K.sum( origin,axis=(1,2,3)) / K.sum( mask, axis=(1,2,3))
    #origin_loss =  K.sum( mask * origin,axis=(1,2,3)) / K.sum( mask, axis=(1,2,3))
    return origin + 0 * a + 0 * s

  def alpha(self, y_true, y_pred):
    return self.v_alpha
  def scale(self, y_true, y_pred):
    return self.v_scale

#lossfun =  robust_loss.adaptive.AdaptiveImageLossFunction((1088, 768, 1), float_dtype='float32',color_space='RGB')
def Robustloss(y_true, y_pred):

    mask = K.cast(y_true >=1, 'float32')
    #lossfun =  robust_loss.adaptive.AdaptiveImageLossFunction((1088, 768, 1), float_dtype='float32',color_space='RGB')
    x = y_true-y_pred
    #Cauchy
    #origin =  K.log(0.5*x*x + 1)
    # Welsch
    #origin =  1-K.exp(-0.5*x*x)
    #Charbonnier
    origin =  K.sqrt(x*x+1) - 1 
    #Geman
    #origin =  -2*(1/(((x*x)/4)+1) - 1)
    #origin = lossfun(x)
    
    origin_loss =  K.sum( mask * origin,axis=(0,1,2)) / K.sum( mask, axis=(1,2,3))
    return origin_loss

def seg_2( y_true, y_pred, th=1 ) :
    loss = K.cast( y_true == y_pred, 'float32' )
    mask = K.cast( y_true>=th, 'float32' )
    return K.sum( mask * loss,axis=(1,2,3)) / K.sum( mask, axis=(1,2,3))

def IOU_calc(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 2*(intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


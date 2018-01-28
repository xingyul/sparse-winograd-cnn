#!/usr/bin/python

import tensorflow as tf
import numpy as np
import os
from tensorflow.python.framework import ops

# package_path = '/usr/local/lib/python2.7/dist-packages/tensorflow/core/user_ops/winograd2x2_transImage_cublas/winograd2x2_imTrans/'
package_path = os.path.dirname(os.path.realpath(__file__))
winograd2x2_imTrans_module = tf.load_op_library(os.path.join(package_path, 'winograd2x2_imTrans_op.so'))
winograd2x2_imTrans_grad_module = tf.load_op_library(os.path.join(package_path, 'winograd2x2_imTrans_grad_op.so'))

def winograd2x2_imTrans(I):
	return winograd2x2_imTrans_module.winograd2x2_im_trans(I)

def winograd2x2_imTrans_grad(grad):
  	return winograd2x2_imTrans_grad_module.winograd2x2_im_trans_grad(grad)


@ops.RegisterShape('Winograd2x2ImTrans')
def _my_winograd2x2_shape(op):
	shape = op.inputs[0].get_shape().with_rank(4)
	H = shape.dims[1]
	W = shape.dims[2]
	nH = (H+1)/2
	nW = (W+1)/2
	return [tf.TensorShape([16, shape.dims[0], nH, nW, shape.dims[3]])]

@ops.RegisterGradient('Winograd2x2ImTrans')
def _my_matmul_grad(op, grad_output):
 	# input = op.inputs[0]
	grad = winograd2x2_imTrans_grad_module.winograd2x2_im_trans_grad(grad_output)
 	return [grad]


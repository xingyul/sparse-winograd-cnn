#!/usr/bin/python

import tensorflow as tf
import numpy as np
import os
from tensorflow.python.framework import ops

# package_path = '/usr/local/lib/python2.7/dist-packages/tensorflow/core/user_ops/winograd2x2_transImage_cublas/winograd2x2_conv/'
package_path = os.path.dirname(os.path.realpath(__file__))
winograd2x2_conv_module = tf.load_op_library(os.path.join(package_path, 'winograd2x2_conv_op.so'))
winograd2x2_conv_grad_module = tf.load_op_library(os.path.join(package_path, 'winograd2x2_conv_grad_op.so'))

def winograd2x2_conv(I, W):
	return winograd2x2_conv_module.winograd2x2_conv(I, W)

def winograd2x2_conv_grad(i1, i2, grad):
 	return winograd2x2_conv_grad_module.winograd2x2_conv_grad(i1, i2, grad)

@ops.RegisterShape('Winograd2x2Conv')
def _my_winograd2x2_shape(op):
	shape1 = op.inputs[0].get_shape().with_rank(5)
	shape2 = op.inputs[1].get_shape().with_rank(3)
	B = shape1.dims[1]
	nH = shape1.dims[2]
	nW = shape1.dims[3]
	H = nH * 2
	W = nW * 2
	K = shape2.dims[2]
	return [tf.TensorShape([B, H, W, K])]

@ops.RegisterGradient('Winograd2x2Conv')
def _my_matmul_grad(op, grad_output):
 	input1 = op.inputs[0]
 	input2 = op.inputs[1]
 	grad1, grad2 = winograd2x2_conv_grad_module.winograd2x2_conv_grad(input1, input2, grad_output)
 	return [grad1, grad2]


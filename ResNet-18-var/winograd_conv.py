#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: winograd_conv.py
# Author: Xingyu Liu <liuxy610042@gmail.com>

import tensorflow as tf
from .common import layer_register
from ..utils.argtools import shape2d, shape4d

import winograd2x2_conv.winograd2x2_conv

__all__ = ['WinogradConv']


@layer_register()
def WinogradConv(x, in_channel, out_channel, mask=None, W_init=None):

    if W_init is None:
        W_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0 * 9.0 / 32.0)

    W = tf.get_variable('W', [16, in_channel, out_channel], initializer=W_init)
    ######
    if mask is not None:
        m = tf.constant(mask)
        W = W * m
    ######

    return winograd2x2_conv.winograd2x2_conv.winograd2x2_conv(x, W)


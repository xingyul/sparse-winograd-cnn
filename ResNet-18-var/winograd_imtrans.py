#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: conv2d.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from .common import layer_register
from ..utils.argtools import shape2d, shape4d

import winograd2x2_imTrans.winograd2x2_imTrans 

__all__ = ['WinogradImTrans']


@layer_register()
def WinogradImTrans(x, nl=tf.identity):
    return nl(winograd2x2_imTrans.winograd2x2_imTrans.winograd2x2_imTrans(x))


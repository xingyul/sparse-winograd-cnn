#!/usr/bin/python 

# -*- coding: UTF-8 -*- 
import numpy as np 
import tensorflow as tf 
import argparse 
import os 
import pickle
import glob
import sys

from tensorpack import * 
from tensorpack.tfutils.symbolic_functions import * 
from tensorpack.tfutils.summary import * 
from tensorflow.contrib.layers import variance_scaling_initializer 

import winograd2x2_conv.winograd2x2_conv
import winograd2x2_imTrans.winograd2x2_imTrans

def gen_prune_mask(mat, density):
	flatten_data = mat.flatten()
	mask = np.zeros_like(mat).flatten()
	mask[:] = 1
	rank = np.argsort(abs(flatten_data))
	flatten_data[rank[:-int(rank.size * density)]] = 0
	mask[rank[:-int(rank.size * density)]] = 0
	flatten_data = flatten_data.reshape(mat.shape)
	np.copyto(mat, flatten_data)
	return mask.reshape(mat.shape)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir', help='The directory to view.')
	parser.add_argument('--gpu', help='Which gpu to use.')
	args = parser.parse_args()

	view_dir = args.dir
	meta_file = glob.glob(view_dir + '/graph-*')[0]
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	# import pdb; pdb.set_trace()
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
		chpt = tf.train.latest_checkpoint(view_dir)
		saver.restore(sess, chpt)
		all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		prune_mask_dict = {}
		i = 0
		for v in all_vars:
			v_ = sess.run(v)
			s = v.name
			s = s[:s.find(':0')]
			if 'Winograd_W' in s and '/W' in s:
				print s, ' :', v_
				i += 1
	print 'finished'

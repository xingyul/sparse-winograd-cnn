#!/usr/bin/python 

# -*- coding: UTF-8 -*- 
import numpy as np 
import tensorflow as tf 
import argparse 
import os 

from tensorpack import * 
from tensorpack.tfutils.symbolic_functions import * 
from tensorpack.tfutils.summary import * 
from tensorflow.contrib.layers import variance_scaling_initializer 

# import winograd3x3  
import winograd2x2_conv.winograd2x2_conv
import winograd2x2_imTrans.winograd2x2_imTrans

import pickle
import glob
import sys

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
	parser.add_argument('--original_dir', help='The directory of the original model.')
	parser.add_argument('--output_dir', help='The directory of the output model.')
	parser.add_argument('--gpu', help='GPU that will be used.')
	parser.add_argument('--density', help='The density of each layer.')
	parser.add_argument('--existing_mask', help='The existing mask file.')
	args = parser.parse_args()

	original_dir = args.original_dir
	meta_file = glob.glob(original_dir + '/graph-*')[0]
	meta_file_name = os.path.basename(meta_file)
	out_dir = args.output_dir
	# density = args.density.split(',')
	density = float(args.density)
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	# import pdb; pdb.set_trace()
	os.system('mkdir %s/pruned_%s' % (out_dir, args.density))
	os.system('cp %s/%s %s/pruned_%s/%s' % (original_dir, meta_file_name, out_dir, args.density, meta_file_name))
	if args.existing_mask:
		existing_masks = pickle.load(open(args.existing_mask, 'rb'))
	# if args.apply_mask:
	# 	apply_masks = pickle.load(open(args.apply_mask, 'rb'))
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
		chpt = tf.train.latest_checkpoint(original_dir)
		saver.restore(sess, chpt)
		all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		prune_mask_dict = {}
		i = 0
		for v in all_vars:
			v_ = sess.run(v)
			s = v.name
			s = s[:s.find(':0')]
			print s
			if 'Winograd_W' in s and '/W' in s:  
				# if args.apply_mask is None:
				print 'inserting ', s, 'with density of', density
				if args.existing_mask:
					mask = gen_prune_mask(v_ * existing_masks[s], float(density))
				else:
					mask = gen_prune_mask(v_, float(density))
				prune_mask_dict[s] = mask
				# else:
				# 	mask = apply_masks[s]
				sess.run(v.assign(v_ * mask))
				i += 1
		saver.save(sess, args.output_dir + '/pruned_' + args.density + '/pruned_' + args.density)
	with open(out_dir + '/pruned_' + args.density + '/prune_mask_' + args.density + '.pkl', 'wb') as handle:
		pickle.dump(prune_mask_dict, handle)
		# os.system('cp %s/checkpoint.bak %s/checkpoint'%(original_dir, original_dir))
		# os.system('echo \'model_checkpoint_path: \"model-37830\"\' > %s/prune_chpt'%(exp_dir))
	print 'finished'

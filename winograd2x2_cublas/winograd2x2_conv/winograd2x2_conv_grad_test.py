#!/usr/bin/python

import tensorflow as tf
import numpy as np

import winograd2x2_conv

def kernel_transform_num(kernel):
    _, _, input_channel, output_channel = kernel.shape
    kernel_1_reshape = np.transpose(kernel, [2, 3, 0, 1])
    g_ = np.reshape(kernel_1_reshape, [input_channel * output_channel, 3 * 3])

    GgGT = np.stack([
        g_[:, 0], g_[:, 0]/2 + g_[:, 1]/2 + g_[:, 2]/2, g_[:, 0]/2 - g_[:, 1]/2 + g_[:, 2]/2, g_[:, 2],
        g_[:, 0]/2 + g_[:, 3]/2 + g_[:, 6]/2, g_[:, 0]/4 + g_[:, 1]/4 + g_[:, 2]/4 + g_[:, 3]/4 + g_[:, 4]/4 + g_[:, 5]/4 + g_[:, 6]/4 + g_[:, 7]/4 + g_[:, 8]/4,
        g_[:, 0]/4 - g_[:, 1]/4 + g_[:, 2]/4 + g_[:, 3]/4 - g_[:, 4]/4 + g_[:, 5]/4 + g_[:, 6]/4 - g_[:, 7]/4 + g_[:, 8]/4, g_[:, 2]/2 + g_[:, 5]/2 + g_[:, 8]/2,
        g_[:, 0]/2 - g_[:, 3]/2 + g_[:, 6]/2, g_[:, 0]/4 + g_[:, 1]/4 + g_[:, 2]/4 - g_[:, 3]/4 - g_[:, 4]/4 - g_[:, 5]/4 + g_[:, 6]/4 + g_[:, 7]/4 + g_[:, 8]/4,
        g_[:, 0]/4 - g_[:, 1]/4 + g_[:, 2]/4 - g_[:, 3]/4 + g_[:, 4]/4 - g_[:, 5]/4 + g_[:, 6]/4 - g_[:, 7]/4 + g_[:, 8]/4, g_[:, 2]/2 - g_[:, 5]/2 + g_[:, 8]/2,
        g_[:, 6], g_[:, 6]/2 + g_[:, 7]/2 + g_[:, 8]/2, g_[:, 6]/2 - g_[:, 7]/2 + g_[:, 8]/2, g_[:, 8]
    ])
    GgGT = np.transpose(GgGT)
    GgGT = np.reshape(GgGT, [input_channel, output_channel, 4 * 4])
    # GgGT = np.transpose(GgGT, [1, 2, 0])
    GgGT = np.transpose(GgGT, [2, 0, 1])
    return GgGT

def image_transform(image, nl=tf.identity):
        batch_size_d, image_width_d, image_height_d, input_channel_d = image.get_shape()
        batch_size = batch_size_d.value
        image_width = image_width_d.value
        image_height = image_height_d.value
        input_channel = input_channel_d.value
        n_patch_width = (image_width + 1 + 2 - 4) / 2 + 1
        n_patch_height = (image_height + 1 + 2 - 4) / 2 + 1
        if batch_size == None:
                batch_size = -1

        extract_image_patches_result = tf.extract_image_patches(image, [1, 4, 4, 1], [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')
        I_ = tf.reshape(extract_image_patches_result, [-1, 4 * 4, input_channel])

        I_0 = tf.reshape(tf.strided_slice(I_, [0, 0, 0], [0, 1, 0], [1, 1, 1], end_mask=5), [-1, input_channel])
        I_1 = tf.reshape(tf.strided_slice(I_, [0, 1, 0], [0, 2, 0], [1, 1, 1], end_mask=5), [-1, input_channel])
        I_2 = tf.reshape(tf.strided_slice(I_, [0, 2, 0], [0, 3, 0], [1, 1, 1], end_mask=5), [-1, input_channel])
        I_3 = tf.reshape(tf.strided_slice(I_, [0, 3, 0], [0, 4, 0], [1, 1, 1], end_mask=5), [-1, input_channel])
        I_4 = tf.reshape(tf.strided_slice(I_, [0, 4, 0], [0, 5, 0], [1, 1, 1], end_mask=5), [-1, input_channel])
        I_5 = tf.reshape(tf.strided_slice(I_, [0, 5, 0], [0, 6, 0], [1, 1, 1], end_mask=5), [-1, input_channel])
        I_6 = tf.reshape(tf.strided_slice(I_, [0, 6, 0], [0, 7, 0], [1, 1, 1], end_mask=5), [-1, input_channel])
        I_7 = tf.reshape(tf.strided_slice(I_, [0, 7, 0], [0, 8, 0], [1, 1, 1], end_mask=5), [-1, input_channel])
        I_8 = tf.reshape(tf.strided_slice(I_, [0, 8, 0], [0, 9, 0], [1, 1, 1], end_mask=5), [-1, input_channel])
        I_9 = tf.reshape(tf.strided_slice(I_, [0, 9, 0], [0, 10, 0], [1, 1, 1], end_mask=5), [-1, input_channel])
        I_10 = tf.reshape(tf.strided_slice(I_, [0, 10, 0], [0, 11, 0], [1, 1, 1], end_mask=5), [-1, input_channel])
        I_11 = tf.reshape(tf.strided_slice(I_, [0, 11, 0], [0, 12, 0], [1, 1, 1], end_mask=5), [-1, input_channel])
        I_12 = tf.reshape(tf.strided_slice(I_, [0, 12, 0], [0, 13, 0], [1, 1, 1], end_mask=5), [-1, input_channel])
        I_13 = tf.reshape(tf.strided_slice(I_, [0, 13, 0], [0, 14, 0], [1, 1, 1], end_mask=5), [-1, input_channel])
        I_14 = tf.reshape(tf.strided_slice(I_, [0, 14, 0], [0, 15, 0], [1, 1, 1], end_mask=5), [-1, input_channel])
        I_15 = tf.reshape(tf.strided_slice(I_, [0, 15, 0], [0, 16, 0], [1, 1, 1], end_mask=5), [-1, input_channel])

        CTdC_0 = (I_0 - I_2 - I_8 + I_10)
        CTdC_1 = (I_1 + I_2 - I_9 - I_10)
        CTdC_2 = (I_2 - I_1 + I_9 - I_10)
        CTdC_3 = (I_1 - I_3 - I_9 + I_11)
        CTdC_4 = (I_4 - I_6 + I_8 - I_10)
        CTdC_5 = (I_5 + I_6 + I_9 + I_10)
        CTdC_6 = (I_6 - I_5 - I_9 + I_10)
        CTdC_7 = (I_5 - I_7 + I_9 - I_11)
        CTdC_8 = (I_6 - I_4 + I_8 - I_10)
        CTdC_9 = (I_9 - I_6 - I_5 + I_10)
        CTdC_10 = (I_5 - I_6 - I_9 + I_10)
        CTdC_11 = (I_7 - I_5 + I_9 - I_11)
        CTdC_12 = (I_4 - I_6 - I_12 + I_14)
        CTdC_13 = (I_5 + I_6 - I_13 - I_14)
        CTdC_14 = (I_6 - I_5 + I_13 - I_14)
        CTdC_15 = (I_5 - I_7 - I_13 + I_15)

        CTdC = tf.concat(0, [CTdC_0, CTdC_1, CTdC_2, CTdC_3, CTdC_4, CTdC_5, CTdC_6, CTdC_7, CTdC_8, CTdC_9, CTdC_10, CTdC_11, CTdC_12, CTdC_13, CTdC_14, CTdC_15])
        CTdC = tf.reshape(CTdC, [16, -1, input_channel])
        CTdC = nl(CTdC)
        return CTdC # (16, batch_size * n_patch_width * n_patch_height, input_channel)

def winograd_conv(image, kernel, transformed_image, output_channel, mask=None):
        batch_size_d, image_width_d, image_height_d, input_channel_d = image.get_shape()
        batch_size = batch_size_d.value
        image_width = image_width_d.value
        image_height = image_height_d.value
        input_channel = input_channel_d.value
        n_patch_width = (image_width + 1 + 2 - 4) / 2 + 1
        n_patch_height = (image_height + 1 + 2 - 4) / 2 + 1
        if batch_size is None:
                batch_size = -1

        CTdC_0 = tf.reshape(tf.strided_slice(transformed_image, [0, 0, 0], [1, 0, 0], [1, 1, 1], end_mask=6), [-1, input_channel])
        CTdC_1 = tf.reshape(tf.strided_slice(transformed_image, [1, 0, 0], [2, 0, 0], [1, 1, 1], end_mask=6), [-1, input_channel])
        CTdC_2 = tf.reshape(tf.strided_slice(transformed_image, [2, 0, 0], [3, 0, 0], [1, 1, 1], end_mask=6), [-1, input_channel])
        CTdC_3 = tf.reshape(tf.strided_slice(transformed_image, [3, 0, 0], [4, 0, 0], [1, 1, 1], end_mask=6), [-1, input_channel])
        CTdC_4 = tf.reshape(tf.strided_slice(transformed_image, [4, 0, 0], [5, 0, 0], [1, 1, 1], end_mask=6), [-1, input_channel])
        CTdC_5 = tf.reshape(tf.strided_slice(transformed_image, [5, 0, 0], [6, 0, 0], [1, 1, 1], end_mask=6), [-1, input_channel])
        CTdC_6 = tf.reshape(tf.strided_slice(transformed_image, [6, 0, 0], [7, 0, 0], [1, 1, 1], end_mask=6), [-1, input_channel])
        CTdC_7 = tf.reshape(tf.strided_slice(transformed_image, [7, 0, 0], [8, 0, 0], [1, 1, 1], end_mask=6), [-1, input_channel])
        CTdC_8 = tf.reshape(tf.strided_slice(transformed_image, [8, 0, 0], [9, 0, 0], [1, 1, 1], end_mask=6), [-1, input_channel])
        CTdC_9 = tf.reshape(tf.strided_slice(transformed_image, [9, 0, 0], [10, 0, 0], [1, 1, 1], end_mask=6), [-1, input_channel])
        CTdC_10 = tf.reshape(tf.strided_slice(transformed_image, [10, 0, 0], [11, 0, 0], [1, 1, 1], end_mask=6), [-1, input_channel])
        CTdC_11 = tf.reshape(tf.strided_slice(transformed_image, [11, 0, 0], [12, 0, 0], [1, 1, 1], end_mask=6), [-1, input_channel])
        CTdC_12 = tf.reshape(tf.strided_slice(transformed_image, [12, 0, 0], [13, 0, 0], [1, 1, 1], end_mask=6), [-1, input_channel])
        CTdC_13 = tf.reshape(tf.strided_slice(transformed_image, [13, 0, 0], [14, 0, 0], [1, 1, 1], end_mask=6), [-1, input_channel])
        CTdC_14 = tf.reshape(tf.strided_slice(transformed_image, [14, 0, 0], [15, 0, 0], [1, 1, 1], end_mask=6), [-1, input_channel])
        CTdC_15 = tf.reshape(tf.strided_slice(transformed_image, [15, 0, 0], [16, 0, 0], [1, 1, 1], end_mask=6), [-1, input_channel])

        GgGT = kernel # (16, input_channel, output_channel)

        GgGT_0 = tf.reshape(tf.slice(GgGT, [0, 0, 0], [1, input_channel, output_channel]), [input_channel, output_channel])
        GgGT_1 = tf.reshape(tf.slice(GgGT, [1, 0, 0], [1, input_channel, output_channel]), [input_channel, output_channel])
        GgGT_2 = tf.reshape(tf.slice(GgGT, [2, 0, 0], [1, input_channel, output_channel]), [input_channel, output_channel])
        GgGT_3 = tf.reshape(tf.slice(GgGT, [3, 0, 0], [1, input_channel, output_channel]), [input_channel, output_channel])
        GgGT_4 = tf.reshape(tf.slice(GgGT, [4, 0, 0], [1, input_channel, output_channel]), [input_channel, output_channel])
        GgGT_5 = tf.reshape(tf.slice(GgGT, [5, 0, 0], [1, input_channel, output_channel]), [input_channel, output_channel])
        GgGT_6 = tf.reshape(tf.slice(GgGT, [6, 0, 0], [1, input_channel, output_channel]), [input_channel, output_channel])
        GgGT_7 = tf.reshape(tf.slice(GgGT, [7, 0, 0], [1, input_channel, output_channel]), [input_channel, output_channel])
        GgGT_8 = tf.reshape(tf.slice(GgGT, [8, 0, 0], [1, input_channel, output_channel]), [input_channel, output_channel])
        GgGT_9 = tf.reshape(tf.slice(GgGT, [9, 0, 0], [1, input_channel, output_channel]), [input_channel, output_channel])
        GgGT_10 = tf.reshape(tf.slice(GgGT, [10, 0, 0], [1, input_channel, output_channel]), [input_channel, output_channel])
        GgGT_11 = tf.reshape(tf.slice(GgGT, [11, 0, 0], [1, input_channel, output_channel]), [input_channel, output_channel])
        GgGT_12 = tf.reshape(tf.slice(GgGT, [12, 0, 0], [1, input_channel, output_channel]), [input_channel, output_channel])
        GgGT_13 = tf.reshape(tf.slice(GgGT, [13, 0, 0], [1, input_channel, output_channel]), [input_channel, output_channel])
        GgGT_14 = tf.reshape(tf.slice(GgGT, [14, 0, 0], [1, input_channel, output_channel]), [input_channel, output_channel])
        GgGT_15 = tf.reshape(tf.slice(GgGT, [15, 0, 0], [1, input_channel, output_channel]), [input_channel, output_channel])

        prod_0 = tf.matmul(CTdC_0, GgGT_0)
        prod_1 = tf.matmul(CTdC_1, GgGT_1)
        prod_2 = tf.matmul(CTdC_2, GgGT_2)
        prod_3 = tf.matmul(CTdC_3, GgGT_3)
        prod_4 = tf.matmul(CTdC_4, GgGT_4)
        prod_5 = tf.matmul(CTdC_5, GgGT_5)
        prod_6 = tf.matmul(CTdC_6, GgGT_6)
        prod_7 = tf.matmul(CTdC_7, GgGT_7)
        prod_8 = tf.matmul(CTdC_8, GgGT_8)
        prod_9 = tf.matmul(CTdC_9, GgGT_9)
        prod_10 = tf.matmul(CTdC_10, GgGT_10)
        prod_11 = tf.matmul(CTdC_11, GgGT_11)
        prod_12 = tf.matmul(CTdC_12, GgGT_12)
        prod_13 = tf.matmul(CTdC_13, GgGT_13)
        prod_14 = tf.matmul(CTdC_14, GgGT_14)
        prod_15 = tf.matmul(CTdC_15, GgGT_15)

        ATprodA_0 = prod_0 + prod_1 + prod_2 + prod_4 + prod_5 + prod_6 + prod_8 + prod_9 + prod_10
        ATprodA_1 = prod_1 - prod_2 - prod_3 + prod_5 - prod_6 - prod_7 + prod_9 - prod_10 - prod_11
        ATprodA_2 = prod_4 + prod_5 + prod_6 - prod_8 - prod_9 - prod_10 - prod_12 - prod_13 - prod_14
        ATprodA_3 = prod_5 - prod_6 - prod_7 - prod_9 + prod_10 + prod_11 - prod_13 + prod_14 + prod_15
        ATprodA = tf.concat(0, [ ATprodA_0, ATprodA_1, ATprodA_2, ATprodA_3 ])
        ATprodA = tf.reshape(ATprodA, [2, 2, batch_size, n_patch_width, n_patch_height, output_channel])
        ATprodA = tf.transpose(ATprodA, perm = [3, 0, 4, 1, 5, 2])
        Image = tf.reshape(ATprodA, [2 * n_patch_width, 2 * n_patch_height, -1])
        if image_width % 2 is not 0 or image_height % 2 is not 0:
                Image = tf.image.crop_to_bounding_box(Image, 0, 0, image_width, image_height)
        Image = tf.reshape(Image, [image_width, image_height, output_channel, batch_size])
        Image = tf.transpose(Image, perm=[3, 0, 1, 2])
        return Image



batch_size = 3
input_channel = 5
output_channel = 7
image_width = 6
n_patch_width = (image_width + 1 + 2 - 4) / 2 + 1

image = np.random.randn(batch_size, image_width, image_width, input_channel)
# image = np.ones([batch_size, image_width, image_width, input_channel])
# image = np.reshape(np.linspace(0.0, float(batch_size * image_width * image_width * input_channel - 1) * 0.1, batch_size * image_width * image_width * input_channel), [batch_size, image_width, image_width, input_channel])

image_holder_1 = tf.placeholder(
	tf.float32,
	shape=(batch_size, image_width, image_width, input_channel))

kernel = np.random.randn(3, 3, input_channel, output_channel)
# kernel = np.reshape(np.linspace(0.0, float(3 * 3 * input_channel * output_channel - 1) * 0.1, 3 * 3 * input_channel * output_channel), [3, 3, input_channel, output_channel])
kernel_holder = tf.placeholder(
	tf.float32,
	shape=(3, 3, input_channel, output_channel))

conv_result = tf.nn.conv2d(image_holder_1, kernel_holder, [1, 1, 1, 1], padding='SAME')

transformed_kernel_num = kernel_transform_num(kernel)
transformed_kernel_holder = tf.placeholder(
	tf.float32,
	shape=(16, input_channel, output_channel))

transformed_image = image_transform(image_holder_1, nl=tf.identity)
my_conv_result = winograd_conv(image_holder_1, transformed_kernel_holder, transformed_image, output_channel, mask=None)

with tf.Session() as sess_1:
	conv_result_num = sess_1.run(conv_result, {image_holder_1: image, kernel_holder: kernel})
	my_conv_result_num = sess_1.run(my_conv_result, {image_holder_1: image, transformed_kernel_holder: transformed_kernel_num})
	print np.sum(np.abs(conv_result_num - my_conv_result_num)) / (np.sum(np.abs(conv_result_num)) + (np.sum(np.abs(my_conv_result_num))))

#----------------------ensure the conv is correct------------------

output_grad = np.random.randn(batch_size, image_width, image_width, output_channel)
# output_grad = np.ones([batch_size, image_width, image_width, output_channel])
# output_grad = np.reshape(np.linspace(0.0, float(batch_size * image_width * image_width * output_channel - 1) * 0.1, batch_size * image_width * image_width * output_channel), [batch_size, image_width, image_width, output_channel])

output_grad_holder = tf.placeholder(
	tf.float32,
	shape=(batch_size, image_width, image_width, output_channel))

transformed_image_grad, transformed_kernel_grad = tf.gradients(my_conv_result, [transformed_image, transformed_kernel_holder], grad_ys=output_grad_holder)

with tf.Session() as sess_2:
	transformed_image_num = sess_2.run(transformed_image, {image_holder_1: image})
	transformed_image_grad_num = sess_2.run(transformed_image_grad, {image_holder_1: image, transformed_kernel_holder: transformed_kernel_num, output_grad_holder: output_grad})
	transformed_kernel_grad_num = sess_2.run(transformed_kernel_grad, {image_holder_1: image, transformed_kernel_holder: transformed_kernel_num, output_grad_holder: output_grad})

transformed_image_num = np.reshape(transformed_image_num, [16, batch_size, n_patch_width, n_patch_width, input_channel])
transformed_image_grad_num = np.reshape(transformed_image_grad_num, [16, batch_size, n_patch_width, n_patch_width, input_channel])

with tf.Session(config=tf.ConfigProto()) as sess_3:
	with tf.device('/gpu:0'):
		transformed_image_tf = tf.placeholder(dtype=tf.float32)
		transformed_kernel_tf = tf.placeholder(dtype=tf.float32)
		output_grad_tf = tf.placeholder(dtype=tf.float32)
		my_transformed_image_grad, my_transformed_kernel_grad = winograd2x2_conv.winograd2x2_conv_grad(transformed_image_tf, transformed_kernel_tf, output_grad_tf)

	sess_3.run(tf.initialize_all_variables())
	my_transformed_image_grad_num = sess_3.run(my_transformed_image_grad, {transformed_image_tf: transformed_image_num, transformed_kernel_tf: transformed_kernel_num, output_grad_tf: output_grad})
	my_transformed_kernel_grad_num = sess_3.run(my_transformed_kernel_grad, {transformed_image_tf: transformed_image_num, transformed_kernel_tf: transformed_kernel_num, output_grad_tf: output_grad})

print np.sum(np.abs(transformed_image_grad_num - my_transformed_image_grad_num)) / (np.sum(np.abs(transformed_image_grad_num)) + (np.sum(np.abs(my_transformed_image_grad_num))))
print np.sum(np.abs(transformed_kernel_grad_num - my_transformed_kernel_grad_num)) / (np.sum(np.abs(transformed_kernel_grad_num)) + (np.sum(np.abs(my_transformed_kernel_grad_num))))


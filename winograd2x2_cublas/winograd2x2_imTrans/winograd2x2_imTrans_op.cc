/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <stdio.h>
#include <assert.h> 

using namespace tensorflow;

REGISTER_OP("Winograd2x2ImTrans")
    .Input("input1: float")
    .Output("output: float")
    .Doc(R"doc(
)doc");

void Winograd2x2ImTransComputeLauncher(const float *Input, float *TransIm, int C, int B, int H, int W, int pad_h, int pad_w);

class Winograd2x2ImTransCudaOp : public OpKernel {
public:
  explicit Winograd2x2ImTransCudaOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& I_tensor = context->input(0);
    auto Input = I_tensor.flat<float>();
    // OP_REQUIRES(context, iA_tensor.dims()==2 && iB_tensor.dims()==2);

    int B = I_tensor.dim_size(0);
    int H = I_tensor.dim_size(1);
    int W = I_tensor.dim_size(2);
    int C = I_tensor.dim_size(3);
	int n_patch_width = (W + 1) / 2;
	int n_patch_height = (H + 1) / 2;
	
    // Create an output tensor
    Tensor* O_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{16, B, n_patch_height, n_patch_width, C}, &O_tensor));
    auto Output = O_tensor->template flat<float>();

    // Set all but the first element of the output tensor to 0.
	Winograd2x2ImTransComputeLauncher(Input.data(), Output.data(), C, B, H, W, 1, 1); 
  }
};

REGISTER_KERNEL_BUILDER(Name("Winograd2x2ImTrans").Device(DEVICE_GPU), Winograd2x2ImTransCudaOp);

class Winograd2x2ImTransOp : public OpKernel {
public:
  explicit Winograd2x2ImTransOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& I_tensor = context->input(0);

    int B = I_tensor.dim_size(0);
    int H = I_tensor.dim_size(1);
    int W = I_tensor.dim_size(2);
    int C = I_tensor.dim_size(3);
	int n_patch_width = (W + 1) / 2;
	int n_patch_height = (H + 1) / 2;
	TensorShape output_shape = I_tensor.shape();
	output_shape.set_dim(1, n_patch_width * n_patch_height);
	output_shape.set_dim(2, 16);

    // Create an output tensor
    Tensor* O_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{16, B, n_patch_height, n_patch_width, C}, &O_tensor));
	printf("This CPU code. We don't need this\n");
	exit(-1);
  }
};

REGISTER_KERNEL_BUILDER(Name("Winograd2x2ImTrans").Device(DEVICE_CPU), Winograd2x2ImTransOp);

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

using namespace tensorflow;

REGISTER_OP("Winograd2x2ImTransGrad")
    .Input("input: float")
    .Output("output: float")
    .Doc(R"doc(
)doc");

void Winograd2x2ImTransGradComputeLauncher(const float *Output_grad, float *Input_grad, int C, int B, int H, int W, int pad_h, int pad_w);

class Winograd2x2ImTransGradCudaOp : public OpKernel {
 public:
  explicit Winograd2x2ImTransGradCudaOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& O_grad_tensor = context->input(0);
    auto Output_grad = O_grad_tensor.flat<float>();
    // OP_REQUIRES(context, iA_tensor.dims()==2 && iB_tensor.dims()==2);

    int B = O_grad_tensor.dim_size(1);
    int nH = O_grad_tensor.dim_size(2);
    int nW = O_grad_tensor.dim_size(3);
    int C = O_grad_tensor.dim_size(4);
	int H = 2 * nH;
	int W = 2 * nW;
	
    // Create an output tensor
    Tensor* I_grad_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{B, H, W, C}, &I_grad_tensor));
    auto Input_grad = I_grad_tensor->template flat<float>();

    // Set all but the first element of the output tensor to 0.
	Winograd2x2ImTransGradComputeLauncher(Output_grad.data(), Input_grad.data(), C, B, H, W, 1, 1); 
  }
};

REGISTER_KERNEL_BUILDER(Name("Winograd2x2ImTransGrad").Device(DEVICE_GPU), Winograd2x2ImTransGradCudaOp);

class Winograd2x2ImTransGradOp : public OpKernel {
 public:
  explicit Winograd2x2ImTransGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
	// Grab the input tensor
    const Tensor& O_grad_tensor = context->input(0);
    auto Output_grad = O_grad_tensor.flat<float>();

    int B = O_grad_tensor.dim_size(1);
    int nH = O_grad_tensor.dim_size(2);
    int nW = O_grad_tensor.dim_size(3);
    int C = O_grad_tensor.dim_size(4);
	int H = 2 * nH;
	int W = 2 * nW;

    Tensor* I_grad_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{B, H, W, C}, &I_grad_tensor));
    auto Input_grad = I_grad_tensor->template flat<float>();

	printf("This CPU code. We don't need this\n");
    exit(-1);
  }
};

REGISTER_KERNEL_BUILDER(Name("Winograd2x2ImTransGrad").Device(DEVICE_CPU), Winograd2x2ImTransGradOp);

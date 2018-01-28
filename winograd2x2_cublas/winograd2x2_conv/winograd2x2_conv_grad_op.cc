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

REGISTER_OP("Winograd2x2ConvGrad")
    .Input("input1: float")
    .Input("input2: float")
    .Input("input3: float")
    .Output("output1: float")
    .Output("output2: float")
    .Doc(R"doc(
)doc");

void Winograd2x2ConvGradComputeLauncher(const float *TInput, const float *Weight, const float *Output_grad, float *TInput_grad, float *Weight_grad, float *tmp_data_buffer, long long *tmp_ptr_buffer, int C, int B, int H, int W, int K, int pad_h, int pad_w);

class Winograd2x2ConvGradCudaOp : public OpKernel {
 public:
  explicit Winograd2x2ConvGradCudaOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& TI_tensor = context->input(0);
    const Tensor& W_tensor = context->input(1);
    const Tensor& O_grad_tensor = context->input(2);
    auto TInput = TI_tensor.flat<float>();
    auto Weight = W_tensor.flat<float>();
    auto Output_grad = O_grad_tensor.flat<float>();
    // OP_REQUIRES(context, iA_tensor.dims()==2 && iB_tensor.dims()==2);

    int B = O_grad_tensor.dim_size(0);
    int H = O_grad_tensor.dim_size(1);
    int W = O_grad_tensor.dim_size(2);
    int K = O_grad_tensor.dim_size(3);
    int C = TI_tensor.dim_size(4);
	TensorShape transformed_input_grad_shape = TI_tensor.shape();
	TensorShape weight_grad_shape = W_tensor.shape();
	
    // Create an output tensor
    Tensor* TI_grad_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, transformed_input_grad_shape, &TI_grad_tensor));
    auto TInput_grad = TI_grad_tensor->template flat<float>();

    Tensor* W_grad_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, weight_grad_shape, &W_grad_tensor));
    auto Weight_grad = W_grad_tensor->template flat<float>();

	int nH = (H+1) / 2;
	int nW = (W+1) / 2;
	// Allocate temporary memory
    Tensor tmp_data_buffer_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape{16 * nH * nW * B * K}, &tmp_data_buffer_tensor));
    auto tmp_data_buffer = tmp_data_buffer_tensor.template flat<float>();

    Tensor tmp_ptr_buffer_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, TensorShape{5 * 16}, &tmp_ptr_buffer_tensor));
    auto tmp_ptr_buffer = tmp_ptr_buffer_tensor.template flat<long long>();

    // Set all but the first element of the output tensor to 0.
	Winograd2x2ConvGradComputeLauncher(TInput.data(), Weight.data(), Output_grad.data(), TInput_grad.data(), Weight_grad.data(), tmp_data_buffer.data(), tmp_ptr_buffer.data(), C, B, H, W, K, 1, 1); 
  }
};

REGISTER_KERNEL_BUILDER(Name("Winograd2x2ConvGrad").Device(DEVICE_GPU), Winograd2x2ConvGradCudaOp);

class Winograd2x2ConvGradOp : public OpKernel {
 public:
  explicit Winograd2x2ConvGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
	// Grab the input tensor
    const Tensor& TI_tensor = context->input(0);
    const Tensor& W_tensor = context->input(1);
    const Tensor& O_grad_tensor = context->input(2);

	TensorShape transformed_input_grad_shape = TI_tensor.shape();
	TensorShape weight_grad_shape = W_tensor.shape();

    Tensor* TI_grad_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, transformed_input_grad_shape, &TI_grad_tensor));

    Tensor* W_grad_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, weight_grad_shape, &W_grad_tensor));

	printf("This CPU code. We don't need this\n");
    exit(-1);
  }
};

REGISTER_KERNEL_BUILDER(Name("Winograd2x2ConvGrad").Device(DEVICE_CPU), Winograd2x2ConvGradOp);

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

REGISTER_OP("Winograd2x2Conv")
    .Input("input1: float")
    .Input("input2: float")
    .Output("output: float")
    .Doc(R"doc(
)doc");

void Winograd2x2ConvComputeLauncher(const float *Input, const float *Weight, float *Output, float *tmp_data_buffer, const long long *tmp_ptr_buffer, int C, int B, int nH, int nW, int K, int pad_h, int pad_w);

class Winograd2x2ConvCudaOp : public OpKernel {
public:
  explicit Winograd2x2ConvCudaOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& I_tensor = context->input(0);
    const Tensor& W_tensor = context->input(1);
    auto Input = I_tensor.flat<float>();
    auto Weight = W_tensor.flat<float>();
    // OP_REQUIRES(context, iA_tensor.dims()==2 && iB_tensor.dims()==2);

    int B = I_tensor.dim_size(1);
    int nH= I_tensor.dim_size(2);
    int nW= I_tensor.dim_size(3);
    int C = I_tensor.dim_size(4);
    int K = W_tensor.dim_size(2);
	
    // Create an output tensor
    Tensor* O_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{B, 2*nH, 2*nW, K}, &O_tensor));
    auto Output = O_tensor->template flat<float>();

	// Allocate temporary memory
    Tensor tmp_data_buffer_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape{16 * nH * nW * B * K}, &tmp_data_buffer_tensor));
    auto tmp_data_buffer = tmp_data_buffer_tensor.template flat<float>();

    Tensor tmp_ptr_buffer_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, TensorShape{3 * 16}, &tmp_ptr_buffer_tensor));
    auto tmp_ptr_buffer = tmp_ptr_buffer_tensor.template flat<long long>();

    // Set all but the first element of the output tensor to 0.
	Winograd2x2ConvComputeLauncher(Input.data(), Weight.data(), Output.data(), tmp_data_buffer.data(), tmp_ptr_buffer.data(), C, B, nH, nW, K, 1, 1); 
  }
};

REGISTER_KERNEL_BUILDER(Name("Winograd2x2Conv").Device(DEVICE_GPU), Winograd2x2ConvCudaOp);

class Winograd2x2ConvOp : public OpKernel {
public:
  explicit Winograd2x2ConvOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& I_tensor = context->input(0);
    const Tensor& W_tensor = context->input(1);

    int K = W_tensor.dim_size(0);
    TensorShape output_shape = I_tensor.shape();
    output_shape.set_dim(3, K);

    // Create an output tensor
    Tensor* O_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &O_tensor));
	printf("This CPU code. We don't need this\n");
	exit(-1);
  }
};

REGISTER_KERNEL_BUILDER(Name("Winograd2x2Conv").Device(DEVICE_CPU), Winograd2x2ConvOp);

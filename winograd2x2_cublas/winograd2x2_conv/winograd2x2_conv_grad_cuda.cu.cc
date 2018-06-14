#if GOOGLE_CUDA
#define EIGEN_USE_GPU
// #include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <cublas_v2.h>

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)

// O = (Batch, H, W, K)
// TI = (16, Batch, nH, nW, K)
template <typename T>
__global__ void Grad_inverse_transform(const T *Output_grad, T *Product_grad, int C, int B, int H, int W, int K, int pad_h, int pad_w) 
{ 
	int bx = blockIdx.x; // w
	int by = blockIdx.y; // h
	int bz = blockIdx.z; // b
	int tx = threadIdx.x; // k

	int nW = (W + 1 + 2 * pad_w - 4) / 2 + 1;
	int nH = (H + 1 + 2 * pad_h - 4) / 2 + 1;

	float output_grad_patch_0 = Output_grad [bz * H * W * K + (2 * by + 0) * W * K + (2 * bx + 0) * K + tx];
	float output_grad_patch_1 = Output_grad [bz * H * W * K + (2 * by + 0) * W * K + (2 * bx + 1) * K + tx];
	float output_grad_patch_2 = Output_grad [bz * H * W * K + (2 * by + 1) * W * K + (2 * bx + 0) * K + tx];
	float output_grad_patch_3 = Output_grad [bz * H * W * K + (2 * by + 1) * W * K + (2 * bx + 1) * K + tx];


	float product_grad_patch_0 = output_grad_patch_0;
	float product_grad_patch_1 = output_grad_patch_0 + output_grad_patch_1;
	float product_grad_patch_2 = output_grad_patch_0 - output_grad_patch_1;
	float product_grad_patch_3 =-output_grad_patch_1;
	float product_grad_patch_4 = output_grad_patch_0 + output_grad_patch_2;
	float product_grad_patch_5 = output_grad_patch_0 + output_grad_patch_1 + output_grad_patch_2 + output_grad_patch_3;
	float product_grad_patch_6 = output_grad_patch_0 - output_grad_patch_1 + output_grad_patch_2 - output_grad_patch_3;
	float product_grad_patch_7 =-output_grad_patch_1 - output_grad_patch_3;
	float product_grad_patch_8 = output_grad_patch_0 - output_grad_patch_2;
	float product_grad_patch_9 = output_grad_patch_0 + output_grad_patch_1 - output_grad_patch_2 - output_grad_patch_3;
	float product_grad_patch_10= output_grad_patch_0 - output_grad_patch_1 - output_grad_patch_2 + output_grad_patch_3;
	float product_grad_patch_11= output_grad_patch_3 - output_grad_patch_1;
	float product_grad_patch_12=-output_grad_patch_2;
	float product_grad_patch_13=-output_grad_patch_2 - output_grad_patch_3;
	float product_grad_patch_14= output_grad_patch_3 - output_grad_patch_2;
	float product_grad_patch_15= output_grad_patch_3;

	
	int base_1 = bz * nH * nW * K + by * nW * K + bx * K + tx;
	int interval_1 = B * nH * nW * K;
	Product_grad [0 * interval_1 + base_1] = product_grad_patch_0;
	Product_grad [1 * interval_1 + base_1] = product_grad_patch_1;
	Product_grad [2 * interval_1 + base_1] = product_grad_patch_2;
	Product_grad [3 * interval_1 + base_1] = product_grad_patch_3;
	Product_grad [4 * interval_1 + base_1] = product_grad_patch_4;
	Product_grad [5 * interval_1 + base_1] = product_grad_patch_5;
	Product_grad [6 * interval_1 + base_1] = product_grad_patch_6;
	Product_grad [7 * interval_1 + base_1] = product_grad_patch_7;
	Product_grad [8 * interval_1 + base_1] = product_grad_patch_8;
	Product_grad [9 * interval_1 + base_1] = product_grad_patch_9;
	Product_grad [10* interval_1 + base_1] = product_grad_patch_10;
	Product_grad [11* interval_1 + base_1] = product_grad_patch_11;
	Product_grad [12* interval_1 + base_1] = product_grad_patch_12;
	Product_grad [13* interval_1 + base_1] = product_grad_patch_13;
	Product_grad [14* interval_1 + base_1] = product_grad_patch_14;
	Product_grad [15* interval_1 + base_1] = product_grad_patch_15;


}

__global__ void assign(const float *TInput, const float *Weight, float *TInput_grad, float *Weight_grad, float *tmp_data_buffer, const float** TInput_ptrs_gpu, float** TInput_grad_ptrs_gpu, const float** Weight_ptrs_gpu, float** Weight_grad_ptrs_gpu, const float** tmp_product_grad_ptrs_gpu, int C, int B, int nH, int nW, int K) {
	int tx = threadIdx.x; // 16

	TInput_ptrs_gpu [tx] = TInput + tx * B * nH * nW * C;
	TInput_grad_ptrs_gpu [tx] = TInput_grad + tx * B * nH * nW * C;
	Weight_ptrs_gpu [tx] = Weight + tx * K * C;
	Weight_grad_ptrs_gpu [tx] = Weight_grad + tx * K * C;
	tmp_product_grad_ptrs_gpu [tx] = tmp_data_buffer + tx * nH * nW * B * K;
}
	
// W = (16, C, K)
void Winograd2x2ConvGradComputeLauncher(const float *TInput, const float *Weight, const float *Output_grad, float *TInput_grad, float *Weight_grad, float *tmp_data_buffer, long long *tmp_ptr_buffer, int C, int B, int H, int W, int K, int pad_h, int pad_w) {
	int nW = (W + 1 + 2 * pad_w - 4) / 2 + 1;
	int nH = (H + 1 + 2 * pad_h - 4) / 2 + 1;

	const float** TInput_ptrs_gpu_ = (const float **)(tmp_ptr_buffer);
	float** TInput_grad_ptrs_gpu_ = (float **)(tmp_ptr_buffer + 16);
	const float** Weight_ptrs_gpu_ = (const float **)(tmp_ptr_buffer + 2 * 16);
	float** Weight_grad_ptrs_gpu_ = (float **)(tmp_ptr_buffer + 3 * 16);
	const float** tmp_product_grad_ptrs_gpu_ = (const float **)(tmp_ptr_buffer + 4 * 16);

	dim3 bDim(16, 1, 1);
	dim3 gDim(1, 1, 1);
	assign <<<gDim, bDim>>> (TInput, Weight, TInput_grad, Weight_grad, tmp_data_buffer, TInput_ptrs_gpu_, TInput_grad_ptrs_gpu_, Weight_ptrs_gpu_, Weight_grad_ptrs_gpu_, tmp_product_grad_ptrs_gpu_, C, B, nH, nW, K);

	dim3 blockDim1(K, 1, 1);
	dim3 gridDim1(nW, nH, B);
	Grad_inverse_transform<float><<<gridDim1, blockDim1>>>(Output_grad, tmp_data_buffer, C, B, H, W, K, pad_h, pad_w);

	cublasHandle_t handle_1;
	cublasCreate(&handle_1);

	float one = 1;
	float zero = 0;

	cublasSgemmBatched(handle_1, CUBLAS_OP_T, CUBLAS_OP_N,
        C, B * nH * nW, K,
        &one,
        Weight_ptrs_gpu_, K,
        tmp_product_grad_ptrs_gpu_, K,
        &zero, TInput_grad_ptrs_gpu_, C, 16);

	cublasDestroy(handle_1);

	cublasHandle_t handle_2;
	cublasCreate(&handle_2);

	cublasSgemmBatched(handle_2, CUBLAS_OP_N, CUBLAS_OP_T,
        K, C, B * nH * nW,
        &one,
        tmp_product_grad_ptrs_gpu_, K,
        TInput_ptrs_gpu_, C,
        &zero, Weight_grad_ptrs_gpu_, K, 16);

	cublasDestroy(handle_2);

}

#endif

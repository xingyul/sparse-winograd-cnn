#if GOOGLE_CUDA
#define EIGEN_USE_GPU
// #include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <cublas_v2.h>


// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)

// Product = (16, Batch, nH, nW, K)
// Output = (Batch, H, W, K)
template <typename T>
__global__ void Output_transform(const T *Product, T *Output, int C, int B, int nH, int nW, int K, int pad_h, int pad_w)
{
	int bx = blockIdx.x; // w
	int by = blockIdx.y; // h
	int bz = blockIdx.z; // b 
	int tx = threadIdx.x; // K
	int H = 2 * nH;
	int W = 2 * nW;

	T product_patch_0 = Product [0 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx];
	T product_patch_1 = Product [1 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx];
	T product_patch_2 = Product [2 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx];
	T product_patch_3 = Product [3 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx];
	T product_patch_4 = Product [4 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx];
	T product_patch_5 = Product [5 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx];
	T product_patch_6 = Product [6 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx];
	T product_patch_7 = Product [7 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx];
	T product_patch_8 = Product [8 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx];
	T product_patch_9 = Product [9 * B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx];
	T product_patch_10= Product [10* B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx];
	T product_patch_11= Product [11* B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx];
	T product_patch_12= Product [12* B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx];
	T product_patch_13= Product [13* B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx];
	T product_patch_14= Product [14* B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx];
	T product_patch_15= Product [15* B * nH * nW * K + bz * nH * nW * K + by * nW * K + bx * K + tx];
	
	T output_patch_0 = 	product_patch_0 + product_patch_1 + product_patch_2 + product_patch_4 +
						product_patch_5 + product_patch_6 + product_patch_8 + product_patch_9 + product_patch_10;
	T output_patch_1 = product_patch_1 - product_patch_2 - product_patch_3 + product_patch_5 -
                       product_patch_6 - product_patch_7 + product_patch_9 - product_patch_10 - product_patch_11;
	T output_patch_2 = product_patch_4 + product_patch_5 + product_patch_6 - product_patch_8 -
                       product_patch_9 - product_patch_10 - product_patch_12 - product_patch_13 - product_patch_14;
	T output_patch_3 = product_patch_5 - product_patch_6 - product_patch_7 - product_patch_9 +
                       product_patch_10 + product_patch_11 - product_patch_13 + product_patch_14 + product_patch_15;
	
	Output[bz*H*W*K + (2*by+0)*W*K + (2*bx+0)*K + tx] = output_patch_0;
	Output[bz*H*W*K + (2*by+0)*W*K + (2*bx+1)*K + tx] = output_patch_1;
	Output[bz*H*W*K + (2*by+1)*W*K + (2*bx+0)*K + tx] = output_patch_2;
	Output[bz*H*W*K + (2*by+1)*W*K + (2*bx+1)*K + tx] = output_patch_3;
													
} 

__global__ void assign(const float *Input, const float *Weight, float *tmp_data_buffer, const float **Input_ptrs_gpu, const float **Weight_ptrs_gpu, float **tmp_product_ptrs_gpu, int C, int B, int nH, int nW, int K) {
	int tx = threadIdx.x; // 16
	
	Input_ptrs_gpu[tx] = Input + tx * B * nH * nW * C;
	Weight_ptrs_gpu[tx] = Weight + tx * K * C;
	tmp_product_ptrs_gpu[tx] = tmp_data_buffer + tx * nH * nW * B * K;
}

// Input = (16, B, nH, nW, C)
// Weight = (16, C, K)
void Winograd2x2ConvComputeLauncher(const float *Input, const float *Weight, float *Output, float *tmp_data_buffer, const long long *tmp_ptr_buffer, int C, int B, int nH, int nW, int K, int pad_h, int pad_w) {

	const float** Input_ptrs_gpu_ = (const float **)(tmp_ptr_buffer);
	const float** Weight_ptrs_gpu_ = (const float **)(tmp_ptr_buffer + 16);
	float** tmp_product_ptrs_gpu_ = (float **)(tmp_ptr_buffer + 16 * 2);

	dim3 bDim(16, 1, 1);
	dim3 gDim(1, 1, 1);
	assign <<<gDim, bDim>>> (Input, Weight, tmp_data_buffer, Input_ptrs_gpu_, Weight_ptrs_gpu_, tmp_product_ptrs_gpu_, C, B, nH, nW, K);
	
	float one = 1;
	float zero = 0;

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        K, B * nH * nW, C,
        &one,
        Weight_ptrs_gpu_, K,
        Input_ptrs_gpu_, C,
        &zero, tmp_product_ptrs_gpu_, K, 16);

	dim3 blockDim(K, 1, 1);
	dim3 gridDim(nW, nH, B);
	Output_transform <float> <<<gridDim, blockDim>>> (tmp_data_buffer, Output, C, B, nH, nW, K, pad_h, pad_w);

	cublasDestroy(handle);

}

#endif

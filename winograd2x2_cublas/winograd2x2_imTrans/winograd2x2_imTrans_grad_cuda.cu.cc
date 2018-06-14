#if GOOGLE_CUDA
#define EIGEN_USE_GPU
// #include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)

// O = (16, Batch, nH, nW, C)
template <typename T>
__global__ void OutputGradTransform(float *Output_grad, int C, int B, int H, int W, int pad_h, int pad_w) {
	int bx = blockIdx.x; // nw
	int by = blockIdx.y; // nh
	int bz = blockIdx.z; // b
	int tx = threadIdx.x; // c

	int nH = (H + 1) / 2;
	int nW = (W + 1) / 2;

	int offset_1 = bz * nH * nW * C + (by * nW + bx) * C + tx;
	int stride_1 = B * nH * nW * C;

	T trans_input_grad_patch_0 = Output_grad [ 0 * stride_1 + offset_1 ];
	T trans_input_grad_patch_1 = Output_grad [ 1 * stride_1 + offset_1 ];
	T trans_input_grad_patch_2 = Output_grad [ 2 * stride_1 + offset_1 ];
	T trans_input_grad_patch_3 = Output_grad [ 3 * stride_1 + offset_1 ];
	T trans_input_grad_patch_4 = Output_grad [ 4 * stride_1 + offset_1 ];
	T trans_input_grad_patch_5 = Output_grad [ 5 * stride_1 + offset_1 ];
	T trans_input_grad_patch_6 = Output_grad [ 6 * stride_1 + offset_1 ];
	T trans_input_grad_patch_7 = Output_grad [ 7 * stride_1 + offset_1 ];
	T trans_input_grad_patch_8 = Output_grad [ 8 * stride_1 + offset_1 ];
	T trans_input_grad_patch_9 = Output_grad [ 9 * stride_1 + offset_1 ];
	T trans_input_grad_patch_10= Output_grad [ 10* stride_1 + offset_1 ];
	T trans_input_grad_patch_11= Output_grad [ 11* stride_1 + offset_1 ];
	T trans_input_grad_patch_12= Output_grad [ 12* stride_1 + offset_1 ];
	T trans_input_grad_patch_13= Output_grad [ 13* stride_1 + offset_1 ];
	T trans_input_grad_patch_14= Output_grad [ 14* stride_1 + offset_1 ];
	T trans_input_grad_patch_15= Output_grad [ 15* stride_1 + offset_1 ];

	T input_grad_patch_0 = trans_input_grad_patch_0; 
	T input_grad_patch_1 = trans_input_grad_patch_1 - trans_input_grad_patch_2 + trans_input_grad_patch_3;
	T input_grad_patch_2 = trans_input_grad_patch_1 - trans_input_grad_patch_0 + trans_input_grad_patch_2;     
	T input_grad_patch_3 =-trans_input_grad_patch_3;
	T input_grad_patch_4 = trans_input_grad_patch_4 - trans_input_grad_patch_8 + trans_input_grad_patch_12; 
	T input_grad_patch_5 = trans_input_grad_patch_5 - trans_input_grad_patch_6 + trans_input_grad_patch_7 - 
									 trans_input_grad_patch_9 + trans_input_grad_patch_10 - trans_input_grad_patch_11 + 
									 trans_input_grad_patch_13 - trans_input_grad_patch_14 + trans_input_grad_patch_15; 
	T input_grad_patch_6 = trans_input_grad_patch_5 - trans_input_grad_patch_4 + trans_input_grad_patch_6 + 
									 trans_input_grad_patch_8 - trans_input_grad_patch_9 - trans_input_grad_patch_10 - 
									 trans_input_grad_patch_12 + trans_input_grad_patch_13 + trans_input_grad_patch_14; 
	T input_grad_patch_7 = trans_input_grad_patch_11 - trans_input_grad_patch_7 - trans_input_grad_patch_15;
	T input_grad_patch_8 = trans_input_grad_patch_4 - trans_input_grad_patch_0 + trans_input_grad_patch_8; 
	T input_grad_patch_9 = trans_input_grad_patch_2 - trans_input_grad_patch_1 - trans_input_grad_patch_3 + 
									 trans_input_grad_patch_5 - trans_input_grad_patch_6 + trans_input_grad_patch_7 + 
									 trans_input_grad_patch_9 - trans_input_grad_patch_10 + trans_input_grad_patch_11;    
	T input_grad_patch_10= trans_input_grad_patch_0 - trans_input_grad_patch_1 - trans_input_grad_patch_2 - 
									 trans_input_grad_patch_4 + trans_input_grad_patch_5 + trans_input_grad_patch_6 - 
									 trans_input_grad_patch_8 + trans_input_grad_patch_9 + trans_input_grad_patch_10;  
	T input_grad_patch_11= trans_input_grad_patch_3 - trans_input_grad_patch_7 - trans_input_grad_patch_11;
	T input_grad_patch_12=-trans_input_grad_patch_12;
	T input_grad_patch_13= trans_input_grad_patch_14 - trans_input_grad_patch_13 - trans_input_grad_patch_15;  
	T input_grad_patch_14= trans_input_grad_patch_12 - trans_input_grad_patch_13 - trans_input_grad_patch_14;
	T input_grad_patch_15= trans_input_grad_patch_15;

	__syncthreads();
	Output_grad [ 0 * stride_1 + offset_1 ] = input_grad_patch_0;
	Output_grad [ 1 * stride_1 + offset_1 ] = input_grad_patch_1;
	Output_grad [ 2 * stride_1 + offset_1 ] = input_grad_patch_2;
	Output_grad [ 3 * stride_1 + offset_1 ] = input_grad_patch_3;
	Output_grad [ 4 * stride_1 + offset_1 ] = input_grad_patch_4;
	Output_grad [ 5 * stride_1 + offset_1 ] = input_grad_patch_5;
	Output_grad [ 6 * stride_1 + offset_1 ] = input_grad_patch_6;
	Output_grad [ 7 * stride_1 + offset_1 ] = input_grad_patch_7;
	Output_grad [ 8 * stride_1 + offset_1 ] = input_grad_patch_8;
	Output_grad [ 9 * stride_1 + offset_1 ] = input_grad_patch_9;
	Output_grad [ 10* stride_1 + offset_1 ] = input_grad_patch_10;
	Output_grad [ 11* stride_1 + offset_1 ] = input_grad_patch_11;
	Output_grad [ 12* stride_1 + offset_1 ] = input_grad_patch_12;
	Output_grad [ 13* stride_1 + offset_1 ] = input_grad_patch_13;
	Output_grad [ 14* stride_1 + offset_1 ] = input_grad_patch_14;
	Output_grad [ 15* stride_1 + offset_1 ] = input_grad_patch_15;
}

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, H, W)

// I = (Batch, H, W, C)
// O = (16, Batch, nH, nW, C)
template <typename T>
__global__ void Winograd2x2ImTransGradCompute(const float *Output_grad, float *Input_grad, int C, int B, int H, int W, int pad_h, int pad_w) {
	int bx = blockIdx.x; // w
	int by = blockIdx.y; // h
	int bz = blockIdx.z; // b
	int tx = threadIdx.x; // c

	int nH = (H + 1) / 2;
	int nW = (W + 1) / 2;

	int w_eff = bx + pad_w;
    int h_eff = by + pad_h;
    int w_col_start = (w_eff < 4) ? 0 : (w_eff - 4) / 2 + 1;
    int w_col_end = min(w_eff / 2 + 1, nW);
    int h_col_start = (h_eff < 4) ? 0 : (h_eff - 4) / 2 + 1;
    int h_col_end = min(h_eff / 2 + 1, nH);

	T val = 0;
	int offset = bz * nH * nW * C + tx;
	int stride = B * nH * nW * C;
	for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
            int w_offset = w_eff - w_col * 2;   // within 16
            int h_offset = h_eff - h_col * 2;   // within 16
            val += Output_grad [offset + (h_offset * 4 + w_offset) * stride + (h_col * nW + w_col) * C];
        }
    }
	Input_grad[bz * H * W * C + by * W * C + bx * C + tx] = val;
} 

void Winograd2x2ImTransGradComputeLauncher(const float *Output_grad, float *Input_grad, int C, int B, int H, int W, int pad_h, int pad_w) {
	int n_patch_width = (W + 1 + 2 * pad_w - 4) / 2 + 1;
	int n_patch_height = (H + 1 + 2 * pad_h - 4) / 2 + 1;

	// cudaMemset(Input_grad, 0, sizeof(float) * B * C * H * W); 

	OutputGradTransform<float><<<dim3(n_patch_width, n_patch_height, B), dim3(C, 1, 1)>>>((float*)Output_grad, C, B, H, W, pad_h, pad_w);

	// dim3 blockDim1(C, 1, 1);
	// dim3 gridDim1(n_patch_height, n_patch_width, B);
	Winograd2x2ImTransGradCompute<float><<<dim3(W, H, B), dim3(C, 1, 1)>>>(Output_grad, Input_grad, C, B, H, W, pad_h, pad_w);
}

#endif

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
// #include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

// dim3 threadsPerBlock(C)
// dim3 numBlocks(Batch, nH, nW)

// I = (Batch, H, W, C)
// O = (16, Batch, nH, nW, C)
template <typename T>
__global__ void Winograd2x2ImTransCompute(const T *Input, T *Output, int C, int B, int H, int W, int pad_h, int pad_w)
{ 
	int bx = blockIdx.x; // w
	int by = blockIdx.y; // h
	int bz = blockIdx.z; // b 
	int t = threadIdx.x; // c

	int nW = (W + 1 + 2 * pad_w - 4) / 2 + 1;
    int nH = (H + 1 + 2 * pad_h - 4) / 2 + 1;

	int f_b = bz;
	int xBase = 2 * bx - pad_w;
	int yBase = 2 * by - pad_h;

	// T input_patch_1 [16] = {0};
	T input_patch_0;
	T input_patch_1;
	T input_patch_2;
	T input_patch_3;
	T input_patch_4;
	T input_patch_5;
	T input_patch_6;
	T input_patch_7;
	T input_patch_8;
	T input_patch_9;
	T input_patch_10;
	T input_patch_11;
	T input_patch_12;
	T input_patch_13;
	T input_patch_14;
	T input_patch_15;

	// load (4, 4, 1) patch of input from global memory
	int f_x, f_y;
	f_x = xBase + 0; f_y = yBase + 0;
	if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_0 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
	else input_patch_0 = 0;
	f_x = xBase + 1; f_y = yBase + 0;
	if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_1 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
	else input_patch_1 = 0;
	f_x = xBase + 2; f_y = yBase + 0;
	if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_2 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
	else input_patch_2 = 0;
	f_x = xBase + 3; f_y = yBase + 0;
	if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_3 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
	else input_patch_3 = 0;
	f_x = xBase + 0; f_y = yBase + 1;
	if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_4 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
	else input_patch_4 = 0;
	f_x = xBase + 1; f_y = yBase + 1;
	if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_5 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
	else input_patch_5 = 0;
	f_x = xBase + 2; f_y = yBase + 1;
	if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_6 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
	else input_patch_6 = 0;
	f_x = xBase + 3; f_y = yBase + 1;
	if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_7 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
	else input_patch_7 = 0;
	f_x = xBase + 0; f_y = yBase + 2;
	if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_8 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
	else input_patch_8 = 0;
	f_x = xBase + 1; f_y = yBase + 2;
	if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_9 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
	else input_patch_9 = 0;
	f_x = xBase + 2; f_y = yBase + 2;
	if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_10 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
	else input_patch_10 = 0;
	f_x = xBase + 3; f_y = yBase + 2;
	if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_11 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
	else input_patch_11 = 0;
	f_x = xBase + 0; f_y = yBase + 3;
	if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_12 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
	else input_patch_12 = 0;
	f_x = xBase + 1; f_y = yBase + 3;
	if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_13 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
	else input_patch_13 = 0;
	f_x = xBase + 2; f_y = yBase + 3;
	if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_14 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
	else input_patch_14 = 0;
	f_x = xBase + 3; f_y = yBase + 3;
	if((f_x > -1) && (f_x < W) && (f_y > -1) && (f_y < H)) input_patch_15 = Input [ f_b * H * W * C + f_y * W * C + f_x * C + t ]; 
	else input_patch_15 = 0;
	
	T trans_input_patch_0;
	T trans_input_patch_1;
	T trans_input_patch_2;
	T trans_input_patch_3;
	T trans_input_patch_4;
	T trans_input_patch_5;
	T trans_input_patch_6;
	T trans_input_patch_7;
	T trans_input_patch_8;
	T trans_input_patch_9;
	T trans_input_patch_10;
	T trans_input_patch_11;
	T trans_input_patch_12;
	T trans_input_patch_13;
	T trans_input_patch_14;
	T trans_input_patch_15;

	// Winograd Transform
	trans_input_patch_0 = input_patch_0 - input_patch_2 - input_patch_8 + input_patch_10;
	trans_input_patch_1 = input_patch_1 + input_patch_2 - input_patch_9 - input_patch_10;
	trans_input_patch_2 = input_patch_2 - input_patch_1 + input_patch_9 - input_patch_10;
	trans_input_patch_3 = input_patch_1 - input_patch_3 - input_patch_9 + input_patch_11;
	trans_input_patch_4 = input_patch_4 - input_patch_6 + input_patch_8 - input_patch_10;
	trans_input_patch_5 = input_patch_5 + input_patch_6 + input_patch_9 + input_patch_10;
	trans_input_patch_6 = input_patch_6 - input_patch_5 - input_patch_9 + input_patch_10;
	trans_input_patch_7 = input_patch_5 - input_patch_7 + input_patch_9 - input_patch_11;
	trans_input_patch_8 = input_patch_6 - input_patch_4 + input_patch_8 - input_patch_10;
	trans_input_patch_9 = input_patch_9 - input_patch_6 - input_patch_5 + input_patch_10;
	trans_input_patch_10 = input_patch_5 - input_patch_6 - input_patch_9 + input_patch_10;
	trans_input_patch_11 = input_patch_7 - input_patch_5 + input_patch_9 - input_patch_11;
	trans_input_patch_12 = input_patch_4 - input_patch_6 - input_patch_12 + input_patch_14;
	trans_input_patch_13 = input_patch_5 + input_patch_6 - input_patch_13 - input_patch_14;
	trans_input_patch_14 = input_patch_6 - input_patch_5 + input_patch_13 - input_patch_14;
	trans_input_patch_15 = input_patch_5 - input_patch_7 - input_patch_13 + input_patch_15;


	int offset = f_b * nH * nW * C + (by * nW + bx) * C + t;
	int stride = B * nH * nW * C;

	Output [ 0 * stride + offset ] = trans_input_patch_0;
	Output [ 1 * stride + offset ] = trans_input_patch_1;
	Output [ 2 * stride + offset ] = trans_input_patch_2;
	Output [ 3 * stride + offset ] = trans_input_patch_3;
	Output [ 4 * stride + offset ] = trans_input_patch_4;
	Output [ 5 * stride + offset ] = trans_input_patch_5;
	Output [ 6 * stride + offset ] = trans_input_patch_6;
	Output [ 7 * stride + offset ] = trans_input_patch_7;
	Output [ 8 * stride + offset ] = trans_input_patch_8;
	Output [ 9 * stride + offset ] = trans_input_patch_9;
	Output [ 10* stride + offset ] = trans_input_patch_10;
	Output [ 11* stride + offset ] = trans_input_patch_11;
	Output [ 12* stride + offset ] = trans_input_patch_12;
	Output [ 13* stride + offset ] = trans_input_patch_13;
	Output [ 14* stride + offset ] = trans_input_patch_14;
	Output [ 15* stride + offset ] = trans_input_patch_15;
} 

void Winograd2x2ImTransComputeLauncher(const float *Input, float *TransIm, int C, int B, int H, int W, int pad_h, int pad_w) {
	int n_patch_width = (W + 1 + 2 * pad_w - 4) / 2 + 1;
	int n_patch_height = (H + 1 + 2 * pad_h - 4) / 2 + 1;
	dim3 blockDim(C, 1, 1);
	dim3 gridDim(n_patch_width, n_patch_height, B);
	Winograd2x2ImTransCompute<float><<<gridDim, blockDim>>>(Input, TransIm, C, B, H, W, pad_h, pad_w);
}

#endif

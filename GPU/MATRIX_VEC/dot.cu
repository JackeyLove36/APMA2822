#include <cuda.h>
#include <stdio.h>

#define FULL_MASK 0xffffffff
#define __WARP_SIZE__ 32

__global__
void dot_product(double *data, int N, double* result) {
  size_t i = threadIdx.x + blockIdx.x*blockDim.x;
  double sum = 0.0;
  if (i < N)
    sum = data[i]*data[i];

  __syncwarp() ; //sync lanes within warp
  for (int offset = __WARP_SIZE__/2; offset > 0; offset /= 2)
    sum += __shfl_down_sync(FULL_MASK, sum, offset);

  __shared__ double s_mem[1024/__WARP_SIZE__];

  int nwarps = blockDim.x/__WARP_SIZE__;
  // int warpId = threadIdx.x/__WARP_SIZE__;

  if (threadIdx.x % __WARP_SIZE__ == 0) {
    s_mem[threadIdx.x/__WARP_SIZE__] = sum;
    printf("smem = %f\n", s_mem[threadIdx.x/__WARP_SIZE__]);
  }
  
  __syncthreads(); //sync threads within block
  if (threadIdx.x == 0) {
    printf("nwarps = %d\n", nwarps);
    for (int j = 0; j < nwarps; ++j) {
      printf("value = %f, smem = %f\n", result[0], s_mem[j]);
      result[0] += s_mem[j];
    }
      
  }
}

int main(){
  int N = 256;
  double *X;
  

  X = new double[N];
  for (auto i = 0; i < N; i ++ ) 
    X[i] = 1;

  double *x_d;
  double *result, *result_h;

  result_h = new double[1];

  cudaMalloc(&x_d, N*sizeof(double));
  cudaMalloc(&result, 1*sizeof(double));
  cudaMemcpy(x_d, X, N*sizeof(double), cudaMemcpyHostToDevice);
  
  dim3 nthreads(256,1,1);
  dim3 nblocks( (N+nthreads.x-1)/nthreads.x,1,1);
  dot_product<<<nblocks,nthreads,0,0>>>(x_d, N, result);

  cudaMemcpy(result_h, result, 1*sizeof(double), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  cudaFree(x_d);

  printf("value = %f\n", result_h[0]);

  delete[] X;
}
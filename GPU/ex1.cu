#include <cuda.h>
#include <stdio.h>


__global__
void my_fill_kernel(double *data, size_t N, double value){
  size_t i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < N)
     data[i] = value;
}

__global__
void my_daxpy_kernel(double *x, double *y, double alpha, size_t N){
  size_t i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < N)
     y[i] = y[i] + alpha*x[i];
}

int main() {

  size_t N = 1024;
  double *x_h, *y_h;
  double *x_d, *y_d;

  int ncuda_devices = 0;
  cudaGetDeviceCount(&ncuda_devices);
  printf("ncuda_devices = %d\n",ncuda_devices);

  if (ncuda_devices == 0) {
     fprintf(stderr,"NO CUDA DEVICES EXITING\n");
     return 0;
  }
  cudaSetDevice(0);

  x_h = new double[N];
  y_h = new double[N];

  for (size_t i = 0; i < N; ++i){
     x_h[i] = 1.1;
     y_h[i] = 1.3;
  }

  
  //allocate memory on device
  cudaMalloc( (void**) &x_d, sizeof(double)*N);
  cudaMalloc( (void**) &y_d, sizeof(double)*N); 

  //copy data from HOST to DEVICE
  cudaMemcpy(x_d,x_h,sizeof(double)*N,cudaMemcpyHostToDevice);
  cudaMemcpy(y_d,y_h,sizeof(double)*N,cudaMemcpyHostToDevice);

  //compute on DEVICE
  dim3 nthreads(256,1,1);
  dim3 nblocks( (N+nthreads.x-1)/nthreads.x,1,1 );

  my_daxpy_kernel<<<nblocks,nthreads,0,0>>>(x_d,y_d,2.0,N);
  cudaDeviceSynchronize();

  //copy data from DEVICE to HOST
  cudaMemcpy(y_h,y_d,sizeof(double)*N,cudaMemcpyDeviceToHost);


  //print some results
  for (size_t i = 0; i < 5; i+=1)
    printf("y_h[%d] = %g\n",i,y_h[i]);

  //free memory 
  delete[] x_h;
  delete[] y_h;

  cudaFree(x_d);
  cudaFree(y_d);
 
  return 0;
}


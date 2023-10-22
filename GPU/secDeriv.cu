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

__global__ 
void second_deriv_kernel(double* x, double* f, double* fSecondDeriv, double lEndPointX, double delta, int N) {
  size_t i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i == 0)
    fSecondDeriv[0] = 2;
  else if (i == N - 1) 
    fSecondDeriv[N - 1] = 2;
  else if (i < N)
    fSecondDeriv[i] = (f[i - 1] - 2*f[i] + f[i + 1])/(delta*delta);
}

double computeError(double* values, int N) {
  double error = 0;
  for (int i = 0; i < N; i ++ ) {
    printf("values[%d] = %f\n", i, values[i]);
    error += abs(values[i] - 2);
  }
  return error;
}

int main() {

  double *x, *f, *sec_deriv;
  double *x_d, *f_d, *sec_deriv_d;

  int N = 1024;
  double lEndPointX = 0;
  double rEndPointX = 1.0;
  double delta = (rEndPointX-lEndPointX)/(N-1);

  int ncuda_devices = 0;
  cudaGetDeviceCount(&ncuda_devices);
  printf("ncuda_devices = %d\n",ncuda_devices);

  if (ncuda_devices == 0) {
     fprintf(stderr,"NO CUDA DEVICES EXITING\n");
     return 0;
  }
  cudaSetDevice(0);

  x = new double[N];
  f = new double[N];
  sec_deriv = new double[N];

  for (int i = 0; i < N; i ++ ) {
    x[i] = lEndPointX + i*delta;
    f[i] = x[i] * x[i]; // x^2
  }

  //allocate memory on device
  cudaMalloc( (void**) &x_d, sizeof(double)*N);
  cudaMalloc( (void**) &f_d, sizeof(double)*N); 
  cudaMalloc( (void**) &sec_deriv_d, sizeof(double)*N); 

  //copy data from HOST to DEVICE
  cudaMemcpy(x_d,x,sizeof(double)*N,cudaMemcpyHostToDevice);
  cudaMemcpy(f_d,f,sizeof(double)*N,cudaMemcpyHostToDevice);

  //compute on DEVICE
  dim3 nthreads(256,1,1);
  dim3 nblocks( (N+nthreads.x-1)/nthreads.x,1,1);
  second_deriv_kernel<<<nblocks,nthreads,0,0>>>(x_d, f_d, sec_deriv_d, lEndPointX, delta, N);

  cudaDeviceSynchronize();
  //copy data from DEVICE to HOST
  cudaMemcpy(sec_deriv, sec_deriv_d, sizeof(double)*N,cudaMemcpyDeviceToHost);

  double error = computeError(sec_deriv, N);
  printf("\nerror_local = %f\n", error);
  //print some results
  // for (size_t i = 0; i < 5; i+=1)
  //   printf("y_h[%d] = %g\n",i,y_h[i]);


  //free memory 
  delete[] x;
  delete[] f;
  delete[] sec_deriv;

  cudaFree(x_d);
  cudaFree(sec_deriv_d);
 
  return 0;
}


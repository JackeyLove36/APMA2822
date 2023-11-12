#include <iostream>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <stdio.h>
#include <string>

#define FULL_MASK 0xffffffff
#define WARP_SIZE 32

__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__
void mvKernelNoWarp(double* matrix, double* vector, 
              int numRowsM, int vecLength,
              double* result) {
  int row = threadIdx.x + blockIdx.x * blockDim.x;
  if (row < numRowsM) {
    for (int col = 0; col < vecLength; col ++) {
      if (col < vecLength) {
        result[row] += matrix[row*vecLength + col] * vector[col];
      }
    }
  }
}

__global__
void mvKernelSingleWarp(double* matrix, double* vector, 
              int numRowsM, int vecLength,
              double* result) {

  int row = blockIdx.x;
  int lane = (threadIdx.x) % WARP_SIZE;
  double sum = 0.0;

  if (row < numRowsM) {
    for (int col = lane; col < vecLength; col += WARP_SIZE) { // modulus addition
      if (col < vecLength) {
        sum += matrix[row*vecLength + col] * vector[col];
      }
    }

    __syncwarp();
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }
    if (lane == 0) {
      result[row] = sum;
    }
  }
}

__global__
void mvKernelMultipleWarps(double* matrix, double* vector, 
              int numRowsM, int vecLength,
              double* result) {

  int row = blockIdx.x;
  int lane = threadIdx.x % WARP_SIZE;
  int warpid = threadIdx.x/WARP_SIZE;
  int nwarps = blockDim.x/WARP_SIZE;

  double sum = 0.0;

  if (row < numRowsM) {
    for (int col = lane + WARP_SIZE*warpid; col < vecLength; col += WARP_SIZE*nwarps) { // modulus addition
      if (col < vecLength) {
        sum += matrix[row*vecLength + col] * vector[col];
      }
    }
    __syncwarp();
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }

    __shared__ double s_mem[1024/WARP_SIZE]; // max 32 warps per block 
    if (lane == 0) {
      s_mem[warpid] = sum;
    }

    __syncthreads(); // sync threads within block
    if (threadIdx.x == 0) { // first lane in first warp
      for (int j = 0; j < nwarps; ++j) {
        result[row] += s_mem[j];
      }   
    }
  }
}

__global__
void mvKernelMultRowsSingleThreadBlock(double* matrix, double* vector, 
              int numRowsM, int vecLength,double* result) {
  int row = blockDim.x/WARP_SIZE * blockIdx.x + threadIdx.x/WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  double sum = 0.0;

  if (row < numRowsM) {
    for (int col = lane; col < vecLength; col += WARP_SIZE) { // modulus addition
      if (col < vecLength) {
        sum += matrix[row*vecLength + col] * vector[col];
      }
    }
    __syncwarp();
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }
    if (lane == 0) {
      result[row] = sum;
    }
  }
}

__global__
void mvKernelColStrat(double* matrix, double* vector, 
              int numRowsM, int vecLength,
              double* result) {
  int numBlocksPerRow = gridDim.x/numRowsM; 
  int row = blockIdx.x/numBlocksPerRow;
  int rowId = blockIdx.x % numBlocksPerRow;
  int lane = threadIdx.x % WARP_SIZE;
  int nwarpsPerBlock = blockDim.x/WARP_SIZE;
  int warpid = threadIdx.x/WARP_SIZE + rowId * nwarpsPerBlock;
  // two blocks per row here
  double sum = 0.0;
  if (row < numRowsM) {
    for (int col = lane + WARP_SIZE*warpid; col < vecLength; col += WARP_SIZE*nwarpsPerBlock) { // modulus addition
      if (col < vecLength) {
        sum += matrix[row*vecLength + col] * vector[col];
      }
    }
    __syncwarp();
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }

    __shared__ double s_mem[1024/WARP_SIZE]; // max 32 warps per block 
    if (lane == 0) {
      s_mem[warpid] = sum;
    }

    __syncthreads(); // sync threads within block
    if (threadIdx.x == 0) && ((blockIdx.x % numBlocksPerRow) == 0) { // first lane in first warp
      for (int j = 0; j < nwarpsPerBlock; ++j) {
        result[row] += s_mem[j];
      }   
    }
  }
}

void matVecMul(double* mat_h, double* vec_h, double* result_h, int numRowsM, int vecLength, int option) {
    double *mat_d, *vec_d, *result_d;

    cudaMalloc(&mat_d, numRowsM * vecLength * sizeof(double));
    cudaMalloc(&vec_d, vecLength * sizeof(double));
    cudaMalloc(&result_d, numRowsM * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(mat_d, mat_h, numRowsM * vecLength * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vec_d, vec_h, vecLength * sizeof(double), cudaMemcpyHostToDevice);

    if (option == 0) {
      dim3 nthreads(256,1,1);
      dim3 nblocks( (numRowsM+nthreads.x-1)/nthreads.x,1,1);
      mvKernelNoWarp<<<nblocks, nthreads>>>(mat_d, vec_d, numRowsM, vecLength, result_d);
    } else if (option == 1) {
      dim3 nthreads(WARP_SIZE,1,1);
      dim3 nblocks(numRowsM,1,1);
      mvKernelSingleWarp<<<nblocks, nthreads>>>(mat_d, vec_d, numRowsM, vecLength, result_d);
    } else if (option == 2){
      int nwarps = 32; // max 32 nwarps per block
      dim3 nthreads(WARP_SIZE * nwarps,1,1);
      dim3 nblocks(numRowsM,1,1);
      mvKernelMultipleWarps<<<nblocks, nthreads>>>(mat_d, vec_d, numRowsM, vecLength, result_d);
    } else if (option == 3){
      int nrows = 4;
      dim3 nthreads(WARP_SIZE * nrows,1,1);
      dim3 nblocks((numRowsM + nrows - 1)/nrows,1,1);
      mvKernelMultRowsSingleThreadBlock<<<nblocks, nthreads>>>(mat_d, vec_d, numRowsM, vecLength, result_d);
    } else if (option == 4) {
      int nwarps = 32;
      int nblocksPerRow = 4;
      dim3 nthreads(WARP_SIZE * nwarps,1,1);
      dim3 nblocks(nblocksPerRow * numRowsM,1,1);
      mvKernelColStrat<<<nblocks, nthreads>>>(mat_d, vec_d, numRowsM, vecLength, result_d);
    }
    cudaDeviceSynchronize();
    fflush(stdout);

    cudaMemcpy(result_h, result_d, numRowsM * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(mat_d);
    cudaFree(vec_d);
    cudaFree(result_d);
}

double** generateRandomMatrixContiguous(int numRows, int numCols) {
  double** matrix = new double*[numRows];
  matrix[0] = new double[numRows*numCols];

  for (int i = 1; i < numRows; i++)
    matrix[i] = matrix[0] + i*numCols;

  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < numCols; ++j) {
      matrix[i][j] = 2; //static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
    }
  }
  return matrix;
}

double* generateRandomVector(int length) {
  double* vector = new double[length];
  for (int i = 0; i < length; ++i) {
    vector[i] = 2;//static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
  }
  return vector;
}

void deleteMatrixContiguous(double** matrix, int numRows) {
  delete[] matrix[0];
}

void deleteMatrix(double** matrix, int numRows) {
  for (int i = 0; i < numRows; ++i) {
    delete[] matrix[i];
  }
  delete[] matrix;
}


void saveMatrix(const char* fileName, double** matrix, int numRows, int numCols) {
    std::ofstream outputFile(fileName);

    if (!outputFile.is_open()) {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return;
    }

    for (int r = 0; r < numRows; ++r) {
        for (int c = 0; c < numCols; ++c) {
            outputFile << matrix[r][c] << " ";
        }
        outputFile << std::endl;
    }

    outputFile.close();
    std::cout << "Matrix has been saved to " << fileName << std::endl;
}

void saveVector(const char* fileName, double* vector, int numRows) {
    std::ofstream outputFile(fileName);

    if (!outputFile.is_open()) {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return;
    }

    for (int r = 0; r < numRows; ++r) {
      outputFile << vector[r] << " ";
    }

    outputFile.close();
    std::cout << "Vector has been saved to " << fileName << std::endl;
}


unsigned long timedMVMult(int numRows, int numCols, int option) {

  srand(0);

  double** matrix = generateRandomMatrixContiguous(numRows, numCols);
  double* vector = generateRandomVector(numCols);
  double result_h[numRows];

  struct timeval start;
  struct timeval end;

  int numRowsM = numRows;
  int vecLength = numCols;

  gettimeofday(&start, 0);
  
  matVecMul(&matrix[0][0], vector, result_h, numRowsM, vecLength, option);
  for (int i = 0; i < numRowsM; i++) {
        std::cout << result_h[i] << " ";
    }
    std::cout << std::endl;
    printf("NumRowsM: %d\n", numRowsM);
    printf("VecLength: %d\n", vecLength);
  // saveVector("outputs/result.txt", result_h, numRowsM);
  gettimeofday(&end, 0);
  
  deleteMatrixContiguous(matrix, numRows);
  delete[] vector;

  return (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
}

double meanOfMatrix(double** matrix, int numRows, int numCols) {
  double sum = 0.0;
  int count = 0;
  for (int r = 0; r < numRows; ++r) {
    for (int c = 0; c < numCols; ++c) {
      if (std::isfinite(matrix[r][c])) {
          sum += matrix[r][c];
          count++;
      }
    }
  }
  std::cout << "Mean of matrix is: " << sum/(numRows*numCols) << std::endl;
  return sum/(numRows*numCols);
}

void saveLatex(const char* fileName, double** times, double** floprate, int numRows, int numCols, bool inclTime=true) {
    std::ofstream outputFile(fileName);

    if (!outputFile.is_open()) {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return;
    }

    for (int r = 0; r < numRows; ++r) {
      outputFile << "\\hline" << std::endl;
      outputFile << std::pow(10,(r + 1));
      for (int c = 0; c < numCols; ++c) {

        if (inclTime) {
          outputFile <<  " & " << std::pow(10,(c + 1)) << \
                      " & " << times[r][c] << " & " << floprate[r][c] << " \\\\" << std::endl;
        } else {
          outputFile <<  " & " << std::pow(10,(c + 1)) << \
                      " & " << floprate[r][c] << " \\\\" << std::endl;
        }

      }
    }
    outputFile << "\\hline" << std::endl;
    outputFile << "Mean floprate " << meanOfMatrix(floprate, numRows, numCols) << std::endl;
    outputFile.close();
    std::cout << "Matrix has been saved to " << fileName << std::endl;
}



int main(int argc, char *argv[] ) {
  // unsigned long elapsed_time = timedMVMult(100, 100, 0);
  double** times = new double*[4];
  double** floprate = new double*[4];
  int rowCount;
  int colCount;
  int option = 0;
  std::string fileName = "outputs/latex_many_rows_single_block.txt";
  if (argc >= 2) {
    option = atoi(argv[1]);
  }
   if (argc >= 3) {
    fileName = argv[2];
  }

  if (option < 5) {
    for (int numRows = 0; numRows < 4; numRows +=1) {
      times[numRows] = new double[4];
      floprate[numRows] = new double[4];

      rowCount = std::pow(10,(numRows + 1));

      for (int numCols = 0; numCols < 4; numCols += 1) {

        colCount = std::pow(10,(numCols + 1));
        unsigned long elapsed_time = timedMVMult(rowCount, colCount, option);
        if (elapsed_time == 0) {
          elapsed_time = 1;
        }
        times[numRows][numCols] = elapsed_time;
        floprate[numRows][numCols] = (2 * rowCount * colCount)/elapsed_time * std::pow(10, -6);
      }
    }

    meanOfMatrix(floprate, 4, 4);
    saveLatex(fileName.c_str(), times, floprate, 4, 4);

    deleteMatrix(times, 4);
    deleteMatrix(floprate, 4);

  } else if (option == 5) {
    unsigned long elapsed_time_fast = timedMVMult(10, 20000, 2);
    double floprate_fast = (2 * 10 * 20000)/elapsed_time_fast * std::pow(10, -6);
    std::ofstream outputFile(fileName);
    outputFile << "Time fast " << elapsed_time_fast << std::endl;
    outputFile << "Floprate fast " << floprate_fast << std::endl;
    
  } else {
    unsigned long elapsed_time_slow = timedMVMult(10, 20000, 0);
    double floprate_slow = (2 * 10 * 20000)/elapsed_time_slow * std::pow(10, -6);
    std::ofstream outputFile(fileName);
    outputFile << "Time slow " << elapsed_time_slow << std::endl;
    outputFile << "Floprate slow " << floprate_slow << std::endl;
  }
  return 0;
}
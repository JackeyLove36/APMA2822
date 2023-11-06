#include <iostream>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <stdio.h>

#define FULL_MASK 0xffffffff
#define WARP_SIZE 32
#define IDX2C(r,c,nr) (((c)*(nr))+(r))

__global__ void testKernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Test Kernel Running\n");
    }
}


__global__
void mvKernel(double* matrix, double* vector, 
              int numRowsM, int vecLength,
              double* result) {
  int row = threadIdx.x + blockIdx.x*blockDim.x;
  int lane = threadIdx.x % WARP_SIZE;
  printf("Kernel is running on row %d\n", row);

  double sum = 0.0;
  if (row < numRowsM) {
    for (int col = threadIdx.y; col < vecLength; col += WARP_SIZE) {
      if (col < vecLength) {
        sum += matrix[IDX2C(row, col, numRowsM)] * vector[col];
      }
    }
  }
  __syncwarp();
  // Use warp-level parallel reduction to sum the contributions from all threads in the warp
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(FULL_MASK, sum, offset);
  }
  __shared__ double s_mem[1024/WARP_SIZE];

  int nwarps = blockDim.x/WARP_SIZE;
  int warpId = threadIdx.x/WARP_SIZE;
  if (lane == 0) {
    s_mem[warpId] = sum;
    printf("smem = %f\n", s_mem[warpId]);
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    printf("nwarps = %d\n", nwarps);
    result[row] = sum;
  }
}

void matVecMul(double* mat_h, double* vec_h, double* result_h, int numRowsM, int vecLength) {
    double *mat_d, *vec_d, *result_d;

    cudaMalloc((void**)&mat_d, numRowsM * vecLength * sizeof(double));
    cudaMalloc((void**)&vec_d, vecLength * sizeof(double));
    cudaMalloc((void**)&result_d, numRowsM * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(mat_d, mat_h, numRowsM * vecLength * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vec_d, vec_h, vecLength * sizeof(double), cudaMemcpyHostToDevice);

    // Kernel launch - one warp per row
    dim3 nthreads(256,1,1);
    dim3 nblocks( (numRowsM+nthreads.x-1)/nthreads.x,1,1);
    testKernel<<<1, 1>>>();
    printf("KErn");
    //mvKernel<<<nthreads, nblocks>>>(mat_d, vec_d, numRowsM, vecLength, result_d);
    cudaDeviceSynchronize();
    fflush(stdout);

    cudaMemcpy(result_h, result_d, numRowsM * sizeof(double), cudaMemcpyDeviceToHost);
    printf("Result: %f\n", result_h[0]);
    printf("Result: %f\n", mat_h[0]);

    cudaFree(mat_d);
    cudaFree(vec_d);
    cudaFree(result_d);
}


double** generateRandomMatrix(int numRows, int numCols) {
  double** matrix = new double*[numRows];
  for (int i = 0; i < numRows; ++i) {
    matrix[i] = new double[numCols];
    for (int j = 0; j < numCols; ++j) {
      matrix[i][j] = 2; //static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
    }
  }
  return matrix;
}

double** generateRandomMatrixContiguous(int numRows, int numCols) {
  double** matrix = new double*[numRows];
  matrix[0] = new double[numRows*numCols];

  for (int i = 1; i < numRows; i++)
    matrix[i] = matrix[0] + i*numCols;

  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < numCols; ++j) {
      matrix[i][j] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
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


unsigned long timedMVMult(int numRows, int numCols) {

  srand(0);

  double** matrix = generateRandomMatrix(numRows, numCols);
  double* vector = generateRandomVector(numCols);
  double result_h[numRows];

  struct timeval start;
  struct timeval end;

  int numRowsM = numRows;
  int vecLength = numCols;

  gettimeofday(&start, 0);
  
  matVecMul(&matrix[0][0], vector, result_h, numRowsM, vecLength);
  saveVector("outputs/result.txt", result_h, numRowsM);
  gettimeofday(&end, 0);
  
  deleteMatrix(matrix, numRows);
  delete[] vector;

  return (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
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

    outputFile.close();
    std::cout << "Matrix has been saved to " << fileName << std::endl;
}

void meanOfMatrix(double** matrix, int numRows, int numCols) {
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
}

int main() {
  double** times = new double*[4];
  double** floprate = new double*[4];
  int rowCount;
  int colCount;

  for (int numRows = 0; numRows < 4; numRows +=1) {
    times[numRows] = new double[4];
    floprate[numRows] = new double[4];

    rowCount = std::pow(10,(numRows + 1));

    for (int numCols = 0; numCols < 4; numCols += 1) {

      colCount = std::pow(10,(numCols + 1));
      unsigned long elapsed_time = timedMVMult(rowCount, colCount);
      if (elapsed_time == 0) {
        elapsed_time = 1;
      }
      floprate[numRows][numCols] = (2 * rowCount * colCount)/elapsed_time * std::pow(10, -6);
    }
  }

  meanOfMatrix(floprate, 4, 4);
  saveLatex("outputs/latex.txt", times, floprate, 4, 4);

  deleteMatrix(times, 4);
  deleteMatrix(floprate, 4);

  return 0;
}
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <fstream>
#include <vector>
#include <cmath>

#define MIN(a,b) ((a) < (b) ? (a) : (b))

double** generateRandomMatrix(int numRows, int numCols) {
  double** matrix = new double*[numRows];
  for (int i = 0; i < numRows; ++i) {
    matrix[i] = new double[numCols];
    for (int j = 0; j < numCols; ++j) {
      matrix[i][j] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
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
    vector[i] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
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

double* performMatrixMatrixMultiplication(double** C, double** A, double** B, int N, int M, int K) {
  double** matrix = new double*[N];
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < K; ++k) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return C;
}

double* performMatrixMatrixMultiplicationR1(double** C, double** A, double** B, int N, int M, int K) {
  double** matrix = new double*[N];
  for (int j = 0; j < M; ++j) {
    for (int i = 0; i < N; ++i) {
      for (int k = 0; k < K; ++k) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return C;
}

double* performMatrixMatrixMultiplicationR2(double** C, double** A, double** B, int N, int M, int K) {
  double** matrix = new double*[N];
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j < M; ++j) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return C;
}

unsigned long timedMVMult(int N, int K, int M) {

  srand(0);

  double** C = generateRandomMatrix(N, M);
  double** A = generateRandomMatrix(N, K);
  double** B = generateRandomVector(K, M);

  struct timeval start;
  struct timeval end;
  unsigned long e_usec;

  gettimeofday(&start, 0);
  
  double** matrixOut = performMatrixMatrixMultiplication(C, A, B, N, K, M);
  
  gettimeofday(&end, 0);
  
  deleteMatrix(C, N);
  deleteMatrix(A, N);
  deleteMatrix(B, K);

  return (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
}

unsigned long timedMVMultContiguous(int numRows, int numCols) {

  srand(0);

  double** matrix = generateRandomMatrixContiguous(numRows, numCols);
  double* vector = generateRandomVector(numCols);

  struct timeval start;
  struct timeval end;
  unsigned long e_usec;

  int numRowsM = numRows;
  int vecLength = numCols;

  gettimeofday(&start, 0);
  
  double* vectorOut = performMatrixVectorMultiplication(matrix, vector, numRowsM, vecLength);
  
  gettimeofday(&end, 0);
  
  delete[] vectorOut;
  deleteMatrixContiguous(matrix, numRows);
  delete[] vector;

  return (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
}

unsigned long timedMVMultUnrolled2(int numRows, int numCols) {

  srand(0);

  double** matrix = generateRandomMatrixContiguous(numRows, numCols);
  double* vector = generateRandomVector(numCols);

  struct timeval start;
  struct timeval end;
  unsigned long e_usec;

  int numRowsM = numRows;
  int vecLength = numCols;

  gettimeofday(&start, 0);
  
  double* vectorOut = performMatrixVectorMultiplicationUnrolled2(matrix, vector, numRowsM, vecLength);
  
  gettimeofday(&end, 0);
  
  delete[] vectorOut;
  deleteMatrixContiguous(matrix, numRows);
  delete[] vector;

  return (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
}

unsigned long timedMVMultUnrolled4(int numRows, int numCols) {

  srand(0);

  double** matrix = generateRandomMatrixContiguous(numRows, numCols);
  double* vector = generateRandomVector(numCols);

  struct timeval start;
  struct timeval end;
  unsigned long e_usec;

  int numRowsM = numRows;
  int vecLength = numCols;

  gettimeofday(&start, 0);
  
  double* vectorOut = performMatrixVectorMultiplicationUnrolled4(matrix, vector, numRowsM, vecLength);
  
  gettimeofday(&end, 0);
  
  delete[] vectorOut;
  deleteMatrixContiguous(matrix, numRows);
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
        for (int c = 0; c < numCols; ++c) {
          if (inclTime) {
            outputFile << std::pow(10,(r + 1)) << " & " << std::pow(10,(c + 1)) << \
                        " & " << times[r][c] << " & " << floprate[r][c] << " \\\\" << std::endl;
          } else {
            outputFile << std::pow(10,(r + 1)) << " & " << std::pow(10,(c + 1)) << \
                        " & " << floprate[r][c] << " \\\\" << std::endl;
          }
          
        }
    }

    outputFile.close();
    std::cout << "Matrix has been saved to " << fileName << std::endl;
}

void saveLatexUnroll(const char* fileName, double** unroll2, double** unroll4, int numRows, int numCols) {
    std::ofstream outputFile(fileName);

    if (!outputFile.is_open()) {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return;
    }

    for (int r = 0; r < numRows; ++r) {
        for (int c = 0; c < numCols; ++c) {
          outputFile << std::pow(10,(r + 1)) << " & " << std::pow(10,(c + 1)) << \
                      " & " << unroll2[r][c] << " & " << unroll4[r][c] << " \\\\" << std::endl;
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
  // double** times = new double*[4];
  double** floprate2 = new double*[4];
  double** floprate4 = new double*[4];
  int rowCount;
  int colCount;

  for (int numRows = 0; numRows < 4; numRows +=1) {
    // times[numRows] = new double[4];
    floprate2[numRows] = new double[4];
    floprate4[numRows] = new double[4];

    rowCount = std::pow(10,(numRows + 1));

    for (int numCols = 0; numCols < 4; numCols += 1) {

      colCount = std::pow(10,(numCols + 1));
      unsigned long elapsed_time2 = timedMVMultUnrolled2(rowCount, colCount);
      unsigned long elapsed_time4 = timedMVMultUnrolled4(rowCount, colCount);
      if (elapsed_time2 == 0) {
        elapsed_time2 = 1;
      }
      if (elapsed_time4 == 0) {
        elapsed_time4 = 1;
      }
      floprate2[numRows][numCols] = (2 * rowCount * colCount)/elapsed_time2;
      floprate4[numRows][numCols] =  (2 * rowCount * colCount)/elapsed_time4;
    }
  }

  meanOfMatrix(floprate2, 4, 4);
  meanOfMatrix(floprate4, 4, 4);

  // saveMatrix("outputs/unrolled/times.txt", times, 4, 4);
  saveMatrix("outputs/unrolled/floprate2.txt", floprate2, 4, 4);
  saveMatrix("outputs/unrolled/floprate4.txt", floprate4, 4, 4);

  saveLatexUnroll("outputs/unrolled/latex.txt", floprate2, floprate4, 4, 4);

  // deleteMatrix(times, 4);
  deleteMatrix(floprate2, 4);
  deleteMatrix(floprate4, 4);

  return 0;
}
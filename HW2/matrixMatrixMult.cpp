#include <iostream>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <fstream>
#include <vector>
#include <cmath>
#include <omp.h>

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

void deleteMatrixContiguous(double** matrix, int numRows) {
  delete[] matrix[0];
}

void deleteMatrix(double** matrix, int numRows) {
  for (int i = 0; i < numRows; ++i) {
    delete[] matrix[i];
  }
  delete[] matrix;
}

void deleteTensor(double*** matrix, int numRows, int numCols) {
  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < numCols; ++j) {
      delete[] matrix[i][j];
    }
    delete[] matrix[i];
  }
  delete[] matrix;
}

double** loopBlockingMatrixMatrixMultiplication(double** C, double** A, double** B, int N, int M, int K, int blockSize) {
    for (int i = 0; i < N; i += blockSize) {
        for (int j = 0; j < M; j += blockSize) {
            for (int k = 0; k < K; k += blockSize) {
                for (int ii = i; ii < i + blockSize && ii < N; ++ii) {
                    for (int jj = j; jj < j + blockSize && jj < M; ++jj) {
                        for (int kk = k; kk < k + blockSize && kk < K; ++kk) {
                            C[ii][jj] += A[ii][kk] * B[kk][jj];
                        }
                    }
                }
            }
        }
    }
    return C;
}

double** parallelMatrixMatrixMultiplication(double** C, double** A, double** B, int N, int M, int K) {
  int i, j, k;
  #pragma omp parallel for private(i, j, k) shared(A, B, C) collapse(2)
  for (i = 0; i < N; ++i) {
    for (j = 0; j < M; ++j) {
      for (k = 0; k < K; ++k) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return C;
}

double** performMatrixMatrixMultiplication(double** C, double** A, double** B, int N, int M, int K) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < K; ++k) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return C;
}

double** performMatrixMatrixMultiplicationR1(double** C, double** A, double** B, int N, int M, int K) {
  for (int j = 0; j < M; ++j) {
    for (int i = 0; i < N; ++i) {
      for (int k = 0; k < K; ++k) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return C;
}

double** performMatrixMatrixMultiplicationR2(double** C, double** A, double** B, int N, int M, int K) {
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j < M; ++j) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return C;
}

double** performMatrixMatrixMultiplicationR3(double** C, double** A, double** B, int N, int M, int K) {
  for (int k = 0; k < K; ++k) {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return C;
}

unsigned long timedMMMult(int N, int K, int M, int loopUnrollParam) {

  srand(0);

  double** C = generateRandomMatrix(N, M);
  double** A = generateRandomMatrix(N, K);
  double** B = generateRandomMatrix(K, M);

  struct timeval start;
  struct timeval end;
  unsigned long e_usec;
  if (loopUnrollParam == -2) {
    gettimeofday(&start, 0);
    double** matrixOut = loopBlockingMatrixMatrixMultiplication(C, A, B, N, M, K, 8);
    gettimeofday(&end, 0);
  } 
  else if (loopUnrollParam == -1) {
    gettimeofday(&start, 0);
    double** matrixOut = parallelMatrixMatrixMultiplication(C, A, B, N, M, K);
    gettimeofday(&end, 0);
  } 
  else if (loopUnrollParam == 0) {
    gettimeofday(&start, 0);
    double** matrixOut = performMatrixMatrixMultiplication(C, A, B, N, M, K);
    gettimeofday(&end, 0);
  } else if (loopUnrollParam == 1) {
    gettimeofday(&start, 0);
    double** matrixOut = performMatrixMatrixMultiplicationR1(C, A, B, N, M, K);
    gettimeofday(&end, 0);
  } else if (loopUnrollParam == 2){
    gettimeofday(&start, 0);
    double** matrixOut = performMatrixMatrixMultiplicationR2(C, A, B, N, M, K);
    gettimeofday(&end, 0);
  } else {
    gettimeofday(&start, 0);
    double** matrixOut = performMatrixMatrixMultiplicationR3(C, A, B, N, M, K);
    gettimeofday(&end, 0);
  }
  
  deleteMatrix(C, N);
  deleteMatrix(A, N);
  deleteMatrix(B, K);

  return (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
}

void saveLatex(const char* fileName, double*** times, double*** floprate, int N, int M, int K, bool inclTime=true) {
    std::ofstream outputFile(fileName);

    if (!outputFile.is_open()) {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return;
    }
    bool first = true;
    bool second = false;
    for (int r = 0; r < N; ++r) {
      first = true;
      outputFile << "\\hline" << std::endl;
      outputFile << std::pow(10,(r + 1));

        for (int c = 0; c < M; ++c) {
          outputFile << " & " <<  std::pow(10,(c + 1)) << " & ";
          if (c > 0) {
            second = true;
          }
          for (int m = 0; m < K; ++m) {
            if (first) {
              outputFile <<  std::pow(10,(m + 1)) << " & "  << times[r][c][m] << " & " << floprate[r][c][m] << " \\\\" << std::endl;
              first = false;
            } else if (second) {
              outputFile <<  std::pow(10,(m + 1)) << " & "  << times[r][c][m] << " & " << floprate[r][c][m] << " \\\\" << std::endl;
              second = false;
            }
            else {
              outputFile << " & & " << std::pow(10,(m + 1)) << " & "  << times[r][c][m] << " & " << floprate[r][c][m] << " \\\\" << std::endl;
            }

          }
        }
      }

    outputFile.close();
    std::cout << "Matrix has been saved to " << fileName << std::endl;
}

void meanOfTensor(double*** matrix, int numRows, int numCols, int numMid) {
  double sum = 0.0;
  int count = 0;
  for (int r = 0; r < numRows; ++r) {
    for (int c = 0; c < numCols; ++c) {
      for (int k = 0; k < numMid; ++k) {
        if (std::isfinite(matrix[r][c][k])) {
          sum += matrix[r][c][k];
          count++;
        }
      }
    }
  }
  std::cout << "Mean of matrix is: " << sum/(numRows*numCols*numMid) << std::endl;
}

int main(int argc, char *argv[] ) {
  double*** times = new double**[3];
  double*** floprate = new double**[3];

  int N;
  int M;
  int K;

  for (int numRows = 0; numRows < 3; numRows +=1) {
    // times[numRows] = new double[3];
    times[numRows] = new double*[3];
    floprate[numRows] = new double*[3];

    N = std::pow(10,(numRows + 1));

    for (int numCols = 0; numCols < 3; numCols += 1) {

      times[numRows][numCols] = new double[3];
      floprate[numRows][numCols] = new double[3];

      M = std::pow(10,(numCols + 1));

      for (int numMid = 0; numMid < 3; numMid += 1) {

        K = std::pow(10,(numMid + 1));
        if (argc < 2) {
          std::cerr << "Usage: " << argv[0] << " <loop unroll param>" << std::endl;
          return 1;
        }
        int loopUnrollParam = atoi(argv[1]);
        unsigned long elapsed_time = timedMMMult(N, K, M, loopUnrollParam);
        printf("N: %d, M: %d, K: %d, time: %ld\n", N, M, K, elapsed_time);
        if (elapsed_time == 0) {
          elapsed_time = 1;
        }
        times[numRows][numCols][numMid] = elapsed_time;
        floprate[numRows][numCols][numMid] = (2 * N * M * K)/elapsed_time * std::pow(10, -6);
      }
    }
  }

  meanOfTensor(floprate, 3, 3, 3);
  // saveMatrix("outputs/unrolled/times.txt", times, 3, 3);
  saveLatex("outputs/time_floprate_u1.txt", times, floprate, 3, 3, 3);

  deleteTensor(times, 3, 3);
  deleteTensor(floprate, 3, 3);

  return 0;
}
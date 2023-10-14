#include <stdio.h> 
#include <math.h>
#include <algorithm>
#include <mpi.h>

using namespace std;

double f(double x) {
  return sin(2 * M_PI * x);
}

double f_sec_deriv(double x) {
  return -4*M_PI*M_PI*(sin(2 * M_PI * x));
}

void* printFunctionValues(double * functionValues, double* functionValuesTrue, int numberOfPoints) {
  for (int i = 0; i < numberOfPoints; i ++ ) {
    printf("%f %f \n", functionValues[i], functionValuesTrue[i]);
  }
  printf("\n");
}

void copy(double* values, double* valuesUpdated, int numberOfPoints) {
  for (int i = 0; i < numberOfPoints; i ++ ) {
    values[i] = valuesUpdated[i];
  }
}

double l2_error(double* functionValues, double* functionValuesTrue, int numberOfPoints) {
  double error = 0;
  double error_add = 0;
  for (int i = 0; i < numberOfPoints; i ++ ) {
    error_add = (functionValues[i] - functionValuesTrue[i]);
    error += error_add * error_add;
  }
  return error;
}

int main(int argc, char **argv){

  MPI_Init(&argc, &argv);

  int numberOfPoints = 100;

  double lEndPointX = 0;
  double rEndPointX = 1.0;
  
  double delta = (rEndPointX-lEndPointX)/(numberOfPoints-1);
  
  int comm_rank;
  int comm_size;

  MPI_Comm_rank(MPI_COMM_WORLD,&comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&comm_size);

  int chunk = (numberOfPoints+comm_size-1)/comm_size;

  int offset = chunk*comm_rank; 
  int next_rand_offset = chunk*(comm_rank+1);

  // COMPUTES LOCAL ENDPOINTS
  double lEndPoint = f(lEndPointX + offset * delta);
  double rEndPoint = f(lEndPointX + (offset + chunk - 1) * delta);

  double* functionValues = new double[chunk];
  double* functionValuesUpdated = new double[chunk];

  double* x = new double[chunk];
  double* functionSecDeriv = new double[chunk];
  double* functionValuesTrue = new double[chunk];

  for (int i = 0; i < chunk; i ++ ) {
    x[i] = lEndPointX + (i+offset)*delta;
    functionSecDeriv[i] = f_sec_deriv(lEndPointX + (i+offset) * delta);
    functionValuesTrue[i] = f(lEndPointX + (i+offset) * delta);
  }

  for (int i = 0; i < chunk; i ++ ) {
    functionValues[i] = 0;
  }
  if (comm_rank == 0) {
    functionValues[0] = lEndPoint;
  }
  if (comm_rank == comm_size - 1) {
    functionValues[chunk - 1] = rEndPoint;
  }

  copy(functionValuesUpdated, functionValues, chunk);

  MPI_Request sendBack;
  MPI_Request sendForward;
  MPI_Status receiveBack;
  MPI_Status receiveForward;

  double receiveFMinus1;
  double receiveFPlus1;

  for (int iter = 0; iter <= 10000; iter++) {
    for (int i = 1; i < chunk - 1; i++) {
      functionValuesUpdated[i] = (delta * delta * f_sec_deriv(lEndPoint + delta * i) -  functionValues[i + 1] - functionValues[i - 1])/(-2);
    }

    if ((comm_rank != 0) && (comm_rank != comm_size-1)) {
      // MIDDLE CHUNKS
      int sentCodeForward = MPI_Isend(&functionValues[chunk-1], 1, MPI_DOUBLE, comm_rank+1, 0, MPI_COMM_WORLD, &sendForward); // to forward
      int sentCodeBack = MPI_Isend(&functionValues[0], 1, MPI_DOUBLE, comm_rank-1, 0, MPI_COMM_WORLD, &sendBack); // to back
      //use structure to wait to deallocate
      int receivedCodeBack = MPI_Recv(&receiveFMinus1, 1, MPI_DOUBLE, comm_rank-1, 0, MPI_COMM_WORLD, &receiveBack); // from back
      int receivedCodeForward = MPI_Recv(&receiveFPlus1, 1, MPI_DOUBLE, comm_rank+1, 0, MPI_COMM_WORLD, &receiveForward); // from forward

      functionValuesUpdated[0] = (delta * delta * f_sec_deriv(lEndPoint + delta * (0)) -  functionValues[1] - receiveFMinus1)/(-2);
      functionValuesUpdated[chunk-1] = (delta * delta * f_sec_deriv(lEndPoint + delta * (chunk - 1)) - receiveFPlus1 - functionValues[chunk - 2])/(-2);

    } else if (comm_rank == 0){ 
      // FIRST CHUNK
      // printf("sending rank 0 %f\n", f[chunk-1]);
      int sentCodeForward = MPI_Isend(&functionValues[chunk-1], 1, MPI_DOUBLE, comm_rank+1, 0, MPI_COMM_WORLD, &sendForward);
      int receivedCodeForward = MPI_Recv(&receiveFPlus1, 1, MPI_DOUBLE, comm_rank+1, 0, MPI_COMM_WORLD, &receiveForward); // from forward
      // printf("receiving rank 0 %f\n", receiveFPlus1);

      functionValuesUpdated[chunk-1] = (delta * delta * f_sec_deriv(lEndPoint + delta * (chunk-1)) - receiveFPlus1 - functionValues[chunk - 2])/(-2);
      // printf("f second deriv rank 0 %f\n", fSecondDeriv[chunk-1]);
    } else {
      // LAST CHUNK
      // printf("sending rank 1 %f\n", f[0]);
      int sentCodeBack = MPI_Isend(&functionValues[0], 1, MPI_DOUBLE, comm_rank-1, 0, MPI_COMM_WORLD, &sendBack); // to back
      int receivedCodeBack = MPI_Recv(&receiveFMinus1, 1, MPI_DOUBLE, comm_rank-1, 0, MPI_COMM_WORLD, &receiveBack); // from back
      // printf("receiving rank 1 %f\n", receiveFMinus1);

      functionValuesUpdated[0] = (delta * delta * f_sec_deriv(lEndPoint + delta * 0) -  functionValues[1] - receiveFMinus1)/(-2);
      // printf("f second deriv rank 1 %f\n", fSecondDeriv[0]);

    }
    if (iter % 1000 == 0) {
      double error = l2_error(functionValues, functionValuesTrue, chunk);

      double reduce_error;
      int reduceCode = MPI_Reduce(&error, &reduce_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      if (comm_rank == 0) {
        printf("%d, error = %f\n", iter, reduce_error);
      }
    }
    copy(functionValues, functionValuesUpdated, chunk);
  }
  // printFunctionValues(functionValues, functionValuesTrue, 10);

  delete[] functionValues;
  delete[] functionValuesUpdated;
  delete[] functionSecDeriv;
  delete[] functionValuesTrue;
}


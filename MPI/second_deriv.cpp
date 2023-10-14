#include <mpi.h>
#include <stdio.h>
#include <sched.h>
#include <math.h>

double func(double x) {
  return x * x;
}

double f_sec_deriv(double x) {
  return 2;
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

double computeError(double* values, int numberOfPoints) {
  double error = 0;
  for (int i = 0; i < numberOfPoints; i ++ ) {
    printf("values[%d] = %f\n", i, values[i]);
    error += abs(values[i] - 2);
  }
  return error;
}

int main(int argc, char** argv){

  MPI_Init(&argc, &argv);

  int comm_rank;
  int comm_size;

  int numberOfPoints = 100;

  double lEndPointX = 0;
  double rEndPointX = 1.0;

  double delta = (rEndPointX-lEndPointX)/(numberOfPoints-1);

  MPI_Comm_rank(MPI_COMM_WORLD,&comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&comm_size);

  int chunk = (numberOfPoints+comm_size-1)/comm_size;

  // this is done to make sure no rounding error
  int offset = chunk*comm_rank; 
  int next_rand_offset = chunk*(comm_rank+1);

  double* x = (double*) malloc(chunk * sizeof(double));
  double* f = (double*) malloc(chunk * sizeof(double));
  double* fSecondDeriv = (double*) malloc(chunk * sizeof(double)); 

  for (int i = 0; i < chunk; i ++ ) {
    x[i] = lEndPointX + (i+offset)*delta;
    f[i] = func(x[i]);
  }

  for (int i = 1; i < chunk - 1; i ++ ) {
    fSecondDeriv[i] = (f[i - 1] - 2*f[i] + f[i + 1])/(delta*delta);
  }
  printf("Offset %d\n", offset);
  printf("X0 %f, XL %f\n", x[0], x[chunk - 1]);
  printf("F0 %f, FL %f\n", f[0], f[chunk - 1]);

  MPI_Request sendBack;
  MPI_Request sendForward;
  MPI_Status receiveBack;
  MPI_Status receiveForward;

  double receiveFMinus1;
  double receiveFPlus1;

  if ((comm_rank != 0) && (comm_rank != comm_size-1)) {
    int sentCodeForward = MPI_Isend(&f[chunk-1], 1, MPI_DOUBLE, comm_rank+1, 0, MPI_COMM_WORLD, &sendForward); // to forward
    int sentCodeBack = MPI_Isend(&f[0], 1, MPI_DOUBLE, comm_rank-1, 0, MPI_COMM_WORLD, &sendBack); // to back
    //use structure to wait to deallocate
    int receivedCodeBack = MPI_Recv(&receiveFMinus1, 1, MPI_DOUBLE, comm_rank-1, 0, MPI_COMM_WORLD, &receiveBack); // from back
    int receivedCodeForward = MPI_Recv(&receiveFPlus1, 1, MPI_DOUBLE, comm_rank+1, 0, MPI_COMM_WORLD, &receiveForward); // from forward

    fSecondDeriv[0] = (receiveFMinus1 - 2*f[0] + f[1])/(delta*delta);
    fSecondDeriv[chunk-1] = (f[chunk-2] - 2*f[chunk-1] + receiveFPlus1)/(delta*delta);

  } else if (comm_rank == 0){
    printf("sending rank 0 %f\n", f[chunk-1]);
    int sentCodeForward = MPI_Isend(&f[chunk-1], 1, MPI_DOUBLE, comm_rank+1, 0, MPI_COMM_WORLD, &sendForward);
    int receivedCodeForward = MPI_Recv(&receiveFPlus1, 1, MPI_DOUBLE, comm_rank+1, 0, MPI_COMM_WORLD, &receiveForward); // from forward
    printf("receiving rank 0 %f\n", receiveFPlus1);

    fSecondDeriv[chunk-1] = (f[chunk-2] - 2*f[chunk-1] + receiveFPlus1)/(delta*delta);
    printf("f second deriv rank 0 %f\n", fSecondDeriv[chunk-1]);

    fSecondDeriv[0] = 2; // hard code for the first point
  } else {
    printf("sending rank 1 %f\n", f[0]);
    int sentCodeBack = MPI_Isend(&f[0], 1, MPI_DOUBLE, comm_rank-1, 0, MPI_COMM_WORLD, &sendBack); // to back
    int receivedCodeBack = MPI_Recv(&receiveFMinus1, 1, MPI_DOUBLE, comm_rank-1, 0, MPI_COMM_WORLD, &receiveBack); // from back
    printf("receiving rank 1 %f\n", receiveFMinus1);

    fSecondDeriv[0] = (receiveFMinus1 - 2*f[0] + f[1])/(delta*delta);
    printf("f second deriv rank 1 %f\n", fSecondDeriv[0]);

    fSecondDeriv[chunk-1] = 2; // hard code for the last point
  }

  double error = computeError(fSecondDeriv, chunk);
  printf("\nerror_local = %f\n", error);

  double reduce_error;
  
  // if (comm_rank == 0) {
  int reduceCode = MPI_Reduce(&error, &reduce_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  printf("\nerror = %f\n", reduce_error);
  // }
  
  // printf("comm_rank = %d, comm_size = %d cpu_id = %d\n",comm_rank,comm_size, sched_getcpu());

  free(x);
  free(f);
  free(fSecondDeriv);

  MPI_Finalize();
  return 0;
}

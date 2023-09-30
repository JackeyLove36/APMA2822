#include <stdio.h> 
#include <cmath>
#include <algorithm>
#include <omp.h>

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
  #pragma omp parallel
  {
    #pragma omp for
    for (int i = 0; i < numberOfPoints; i ++ ) {
      values[i] = valuesUpdated[i];
    }
  }
}

double l2_error(double* functionValues, double* functionValuesTrue, int numberOfPoints) {
  double error = 0;
  double error_add = 0;
  for (int i = 0; i < numberOfPoints; i ++ ) {
    error_add = (functionValues[i] - functionValuesTrue[i]);
    error += error_add * error_add;
  }
  return sqrt(error);
}

int main(int argc, char **argv){

  int numberOfPoints = 100;

  double lEndPointX = 0;
  double rEndPointX = 1.0;
  
  double delta = (rEndPointX-lEndPointX)/(numberOfPoints-1);

  double lEndPoint = f(lEndPointX);
  double rEndPoint = f(rEndPointX);

  double* functionValues = new double[numberOfPoints];
  for (int i = 1; i < numberOfPoints - 1; i ++ ) {
    functionValues[i] = 0;
  }
  functionValues[0] = lEndPoint;
  functionValues[numberOfPoints - 1] = rEndPoint;

  double* functionValuesUpdated = new double[numberOfPoints];
  copy(functionValuesUpdated, functionValues, numberOfPoints);

  double* functionSecDeriv = new double[numberOfPoints];
  double* functionValuesTrue = new double[numberOfPoints];

  for (int i = 0; i < numberOfPoints; i ++ ) {
    functionSecDeriv[i] = f_sec_deriv(lEndPointX + i * delta);
  }
  for (int i = 0; i < numberOfPoints; i ++ ) {
    functionValuesTrue[i] = f(lEndPointX + i * delta);
  }


  for (int iter = 0; iter <= 10000; iter++) {
    for (int i = 1; i < numberOfPoints - 1; i++) {
      functionValuesUpdated[i] = (delta * delta * f_sec_deriv(lEndPoint + delta * i) -  functionValues[i + 1] - functionValues[i - 1])/(-2);
    }
    if (iter % 1000 == 0) {
      printf("%d, error: %f\n", iter, l2_error(functionValues, functionValuesTrue, numberOfPoints));
    }
    copy(functionValues, functionValuesUpdated, numberOfPoints);
  }
  printFunctionValues(functionValues, functionValuesTrue, 10);

  delete[] functionValues;
  delete[] functionValuesUpdated;
  delete[] functionSecDeriv;
  delete[] functionValuesTrue;
}


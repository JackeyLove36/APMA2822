#include <stdio.h> 
#include <math.h>

int main(int argc, char **argv){
  int i,N = 10;
  double dx = 0.1; 

  double x[10]; //request for statically allocating 10 elements of a type double
  double *z;
  double *y; //pointer to memory
  double a = 23;

  y = new double[10]; //request for dynamically allocating 10 elements of a type double
  z = new double[10];

	for (i = 0; i < N; i ++)
		y[i] = dx*i;

  for (i = 0; i < N; i ++) { 
		x[i] = sin(y[i]);
		z[i] = 2*x[i];
		y[i] = a * x[i] + y[i] + z[i];
	}
  //for (i = 0; i < N; i++) x[i] = dx*i; 

  //for (i = 0; i < N; i++) y[i] = sin( x[i] );


  //for (i = 0; i < N; ++i)
     //fprintf(stdout, "sin(%g) = %g\n", x[i], y[i]);

  delete[] y;  //free dynamically allocated memory
	delete[] z;
  return 0;
}


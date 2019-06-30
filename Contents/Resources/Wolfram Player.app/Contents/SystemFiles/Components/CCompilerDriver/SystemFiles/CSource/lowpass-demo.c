#include <stdio.h>
#include <stdlib.h>
#include "lopass.h"
#include "WolframRTL.h"
   
static WolframLibraryData libData = 0;
   
int main ()
{
	int err = 0;
	mint i, type, rank, nelems, *dims;
	double *data;
	MTensor x, y;
	double dt;
	double RC;
	libData = WolframLibraryData_new (WolframLibraryVersion);

	/* read x */
	type = MType_Real;
	rank = 1;
	dims = (mint*) malloc (rank * sizeof (mint));
	scanf (\" % d \", & nelems);
	dims[0] = nelems;

	err = (*(libData -> MTensor_new)) (type, rank, dims, & x);
	if (err) 
		return 1;
 
	free (dims);
 
	data = (*(libData -> MTensor_getRealData)) (x);
	for (i = 0; i < nelems; i++) {
		scanf (\" % lf \", & (data[i]));
	}
      
	/* read dt */
	scanf (\" % lf \", & dt);
	/* read RC */
	scanf (\" % lf \", & RC);
  
	err = Initialize_lopass (libData);
  
	y = 0;
	err = lopass (libData, x, dt, RC, & y);
	printf (\" % d \\ n \", err);
	if (0 == err) {
		dims = libData -> MTensor_getDimensions (y);
		nelems = dims[0];
		data = (*(libData -> MTensor_getRealData)) (y);
		printf (\" % d \\ n \", nelems);
		for (i = 0; i < nelems; i++)
			printf (\" % f \\ n \", data[i]);
	}

	Uninitialize_lopass (libData);
	return 0;
}

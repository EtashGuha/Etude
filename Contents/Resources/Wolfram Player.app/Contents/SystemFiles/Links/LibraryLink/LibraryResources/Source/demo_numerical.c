
/* Include required header */
#include "WolframLibrary.h"
#include "math.h"
#include "stdlib.h"

/* Return the version of Library Link */
DLLEXPORT mint WolframLibrary_getVersion() {
	return WolframLibraryVersion;
}

/* Initialize Library */
DLLEXPORT int WolframLibrary_initialize(WolframLibraryData libData) {
	return LIBRARY_NO_ERROR;
}

/* Uninitialize Library */
DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData libData) {
	return;
}

DLLEXPORT int parabola(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res)
{
	mreal x, a, f;
	if (Argc != 2) return LIBRARY_FUNCTION_ERROR;
	x = MArgument_getReal(Args[0]);
	a = MArgument_getReal(Args[1]);
	f = x*x - a;
	MArgument_setReal(Res, f);
	return 0;
}

static mint mandelbrot_max_iterations = 1000;

DLLEXPORT int mandelbrot(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res)
{
	mint n = 0;
	mcomplex z = {0.,0.};
	mcomplex c = MArgument_getComplex(Args[0]);
	mreal rr = mcreal(z)*mcreal(z);
	mreal ii = mcimag(z)*mcimag(z);
	while ((n < mandelbrot_max_iterations) && (rr + ii < 4)) {
		mcimag(z) = 2.*mcreal(z)*mcimag(z) + mcimag(c);
		mcreal(z) = rr - ii + mcreal(c);
		rr = mcreal(z)*mcreal(z);
		ii = mcimag(z)*mcimag(z);
		n++;
	}

	if (n == mandelbrot_max_iterations) 
		n = 0;

	MArgument_setInteger(Res, n);
	return LIBRARY_NO_ERROR;
}

DLLEXPORT int refine(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) 
{
	int err;
	mint i, j, k, m, n, nr;
	mint const * dims;
	mint rdims[2];
	mreal dist, distsq;
	mreal x1, x2, dx, y1, y2, dy, d, s, ds;
	mreal *x, *y, *rx, *ry;
	MTensor Targ, Tres;

	dist = MArgument_getReal(Args[0]);
	distsq = dist*dist*(1. + 1.e-8); /* Avoid adding extra points due to roundoff */

	Targ = MArgument_getMTensor(Args[1]);
	if (libData->MTensor_getType(Targ) != MType_Real) return LIBRARY_TYPE_ERROR;
	if (libData->MTensor_getRank(Targ) != 2) return LIBRARY_RANK_ERROR;
	dims = (libData->MTensor_getDimensions)(Targ);
	if (dims[0] != 2) return LIBRARY_DIMENSION_ERROR;
	n = dims[1];
	x = libData->MTensor_getRealData(Targ);
	y = x + n;

	/* Do a pass to determine the length of the result */
	x2 = x[0];
	y2 = y[0];
	nr = n;
	for (i = 1; i < n; i++) {
		x1 = x2;
		y1 = y2;
		x2 = x[i];
		y2 = y[i];
		dx = x2 - x1;
		dy = y2 - y1;
		d = dx*dx + dy*dy;
		if (d > distsq) {
			d = sqrt(d);
			k = ((mint) ceil(d/dist)) - 1;
			nr += k;
		}
	}

	rdims[0] = 2;
	rdims[1] = nr;
	err = libData->MTensor_new(MType_Real, 2, rdims, &Tres);
	if (err) return err;
	rx = libData->MTensor_getRealData(Tres);
	ry = rx + nr; 

	x2 = x[0];
	y2 = y[0];
	rx[0] = x2;
	ry[0] = y2;
	for (j = i = 1; i < n; i++, j++) {
		x1 = x2;
		y1 = y2;
		x2 = x[i];
		y2 = y[i];
		dx = x2 - x1;
		dy = y2 - y1;
		d = dx*dx + dy*dy;
		if (d > distsq) {
			d = sqrt(d);
			k = ((mint) ceil(d/dist)) - 1;
			ds = 1./((mreal) (k + 1));
			for (m = 1; m <= k; m++, j++) {
				s = m*ds;
				rx[j] = x1 + s*dx;
				ry[j] = y1 + s*dy;
			}
		}
		rx[j] = x2;
		ry[j] = y2;
	}

	MArgument_setMTensor(Res, Tres);
	return 0;
}

static mreal epsConstant = 0.3;
static mreal omegaConstant = 1.0;
static mreal gammaConstant = 0.15;


/* Overwrites x and y */
int duffing_rhs(mint n, mreal t, mreal *x, mreal *y)
{
	mint i;
	for (i = 0; i < n; i++) {
		mreal xi = x[i];
		x[i] = y[i];
		y[i] = epsConstant*cos(omegaConstant*t) + xi*(1 - xi*xi) - gammaConstant*y[i];
	}
	return 0;
}

DLLEXPORT int duffing_crk4(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) 
{
	int err = 0;
	mint i, n, nh, rank;
	mint const *dims;
	mreal t0, t1, h;
	mreal *x, *xr, *k, *k0, *k1, *k2, *k3;
	MTensor Targ, Tres;

	Targ = MArgument_getMTensor(Args[0]);
	t0 = MArgument_getReal(Args[1]);
	t1 = MArgument_getReal(Args[2]);
	h = t1 - t0;
	if (h < 0) return LIBRARY_FUNCTION_ERROR;

	n = (libData->MTensor_getFlattenedLength)(Targ);
	nh = n/2;
	if (n != nh*2) return LIBRARY_DIMENSION_ERROR;
	x = (libData->MTensor_getRealData)(Targ);

	k = (mreal *) malloc(4*n*sizeof(mreal));
	k0 = k; k1 = k0 + n; k2 = k1 + n; k3 = k2 + n;

	for (i = 0; i < n; i++) k0[i] = x[i];

	err = duffing_rhs(nh, t0, k0, k0 + nh);

	for (i = 0; i < n; i++) {
		k0[i] *= h;
		k1[i] = x[i] + 0.5*k0[i];
	}

	err = duffing_rhs(nh, t0 + h/2, k1, k1 + nh);
	if (err) goto clean_up;

	for (i = 0; i < n; i++) {
		k1[i] *= h;
		k2[i] = x[i] + 0.5*k1[i];
	}

	err = duffing_rhs(nh, t0 + h/2, k2, k2 + nh);
	if (err) goto clean_up;

	for (i = 0; i < n; i++) {
		k2[i] *= h;
		k3[i] = x[i] + k2[i];
	}
	
	err = duffing_rhs(nh, t1, k3, k3 + nh);
	if (err) goto clean_up;

	rank = (libData->MTensor_getRank)(Targ);
	dims = (libData->MTensor_getDimensions)(Targ);
	err = (libData->MTensor_new)(MType_Real, rank, dims, &Tres);
	if (err) goto clean_up;
	xr =(libData->MTensor_getRealData)(Tres);

	for (i = 0; i < n; i++) 
		xr[i] = x[i] + (k0[i] + 2.*(k1[i] + k2[i]) + h*k3[i])/6.;

	MArgument_setMTensor(Res, Tres);

clean_up:
	free(k);
	return err;
}


static mreal alpha = 0.02;

DLLEXPORT int brusselator_pde_rhs(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) 
{
	int err;
    mint i, j, n, nr;
    mreal dm, uuv;
    mreal *u, *v, *up, *vp;
    MTensor Tres, Targ;
    Targ = MArgument_getMTensor(Args[1]);
    if (libData->MTensor_getType(Targ) != MType_Real) return LIBRARY_TYPE_ERROR;
    if (libData->MTensor_getRank(Targ) != 1) return LIBRARY_RANK_ERROR;
    nr = libData->MTensor_getFlattenedLength(Targ);
    n = nr/2;
    u = libData->MTensor_getRealData(Targ);
    v = u + n;

    err = libData->MTensor_new(MType_Real, 1, &nr, &Tres);
	if (err) return err;
    up = libData->MTensor_getRealData(Tres);
    vp = up + n;

    /* Decrement n so loop excludes boundaries */
    n--;
    dm = (mreal) n;
    dm *= alpha*dm;

    /* Boundary conditions to converge to correct values */
    up[0] = 1. - u[0];
    up[n] = 1. - u[n];
    vp[0] = 3. - v[0];
    vp[n] = 3. - v[n];

    for (i = 1; i < n; i++) {
        mreal uuv = u[i];
        uuv *= uuv*v[i];
        up[i] = 1. + uuv - 4.*u[i] + dm*(u[i - 1] - 2.*u[i] + u[i + 1]);
        vp[i] = 3.*u[i] - uuv + dm*(v[i - 1] - 2.*v[i] + v[i + 1]);
    }

    MArgument_setMTensor(Res, Tres);
    return 0;
}

DLLEXPORT int brusselator_pde_jacobian_values(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) 
{
	int err;
    mint i, j, k, n, nr;
    mreal dm;
    mreal *u, *v, *jval;
    MTensor Tres, Targ;
    Targ = MArgument_getMTensor(Args[1]);
    if (libData->MTensor_getType(Targ) != MType_Real) return LIBRARY_TYPE_ERROR;
    if (libData->MTensor_getRank(Targ) != 1) return LIBRARY_RANK_ERROR;
    nr = libData->MTensor_getFlattenedLength(Targ);
    n = nr/2;
    u = libData->MTensor_getRealData(Targ);
    v = u + n;

	nr = 4 + 8*(n - 2);
    err = libData->MTensor_new(MType_Real, 1, &nr, &Tres);
	if (err) return err;
    jval = libData->MTensor_getRealData(Tres);

    /* Decrement n so loop excludes boundaries */
    n--;
    dm = (mreal) n;
    dm *= alpha*dm;

	k = 0;

	jval[k++] = -1.; /* u[0] bc */
    for (i = 1; i < n; i++) {	
		/* u equations */
		jval[k++] = dm;
		jval[k++] = 2.*u[i]*v[i] - 4. - 2.*dm;
		jval[k++] = dm;
		jval[k++] = u[i]*u[i]; 
	}
	jval[k++] = -1.; /* u[n] bc */
	jval[k++] = -1.; /* v[0] bc */
    for (i = 1; i < n; i++) {	
		/* v equations */
		jval[k++] = 3. - 2.*u[i]*v[i];
		jval[k++] = dm;
		jval[k++] = -u[i]*u[i] - 2.*dm;
		jval[k++] = dm;
	}
	jval[k++] = -1.; /* v[n] bc */

    MArgument_setMTensor(Res, Tres);
    return 0;
}

DLLEXPORT int brusselator_pde_jacobian_positions(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) 
{
	int err;
	mint dims[2];
    mint i, j, k, n, nr;
	mint *jpos;
    MTensor Tres, Targ;
    Targ = MArgument_getMTensor(Args[1]);
    if (libData->MTensor_getType(Targ) != MType_Real) return LIBRARY_TYPE_ERROR;
    if (libData->MTensor_getRank(Targ) != 1) return LIBRARY_RANK_ERROR;
    nr = libData->MTensor_getFlattenedLength(Targ);
    n = nr/2;

	dims[0] = 4 + 8*(n - 2);
	dims[1] = 2;
    err = libData->MTensor_new(MType_Integer, 2, dims, &Tres);
	if (err) return err;
    jpos = libData->MTensor_getIntegerData(Tres);

    /* Decrement n so loop excludes boundaries */
    n--;
	k = 0;

	jpos[k++] = 1; jpos[k++] = 1; /* u[0] bc */
    for (i = 1; i < n; i++) {	
		/* u equations */
		mint r = i + 1;
		jpos[k++] = r; jpos[k++] = i;
		jpos[k++] = r; jpos[k++] = i + 1;
		jpos[k++] = r; jpos[k++] = i + 2;
		jpos[k++] = r; jpos[k++] = i + n + 2;
	}
	jpos[k++] = n + 1; jpos[k++] = n + 1; /* u[n] bc */
	jpos[k++] = n + 2; jpos[k++] = n + 2; /* v[0] bc */
    for (i = 1; i < n; i++) {	
		/* v equations */
		mint r = i + n + 2;
		jpos[k++] = r; jpos[k++] = i + 1;
		jpos[k++] = r; jpos[k++] = i + n + 1;
		jpos[k++] = r; jpos[k++] = i + n + 2;
		jpos[k++] = r; jpos[k++] = i + n + 3;
	}
	jpos[k++] = 2*(n + 1); jpos[k++] = 2*(n + 1); /* v[n] bc */

    MArgument_setMTensor(Res, Tres);
    return 0;
}

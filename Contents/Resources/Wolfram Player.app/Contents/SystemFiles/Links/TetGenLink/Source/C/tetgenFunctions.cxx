/*************************************************************************

        Copyright 1986 through 2010 by Wolfram Research Inc.
        All rights reserved

*************************************************************************/

#include "tetgen.h"
#include "mathlink.h"
#include "WolframLibrary.h"
#include "tetgenWolframDLL.h"



extern "C" {
	DLLEXPORT int getFacets(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
	DLLEXPORT int setFacets(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
	DLLEXPORT int getFacetMarkers(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
	DLLEXPORT int setFacetMarkers(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
	DLLEXPORT int getFacetHoles(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
	DLLEXPORT int getFacetHoleLengths(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
	DLLEXPORT int setFacetHoles(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);

	DLLEXPORT int fileOperation(WolframLibraryData libData, MLINK mlp);
	DLLEXPORT int getPointList(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
	DLLEXPORT int setPointList(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
	DLLEXPORT int getPointAttributes(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
	DLLEXPORT int setPointAttributes(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
	DLLEXPORT int getPointMetricTensors(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
	DLLEXPORT int setPointMetricTensors(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
	DLLEXPORT int getPointMarkers(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
	DLLEXPORT int setPointMarkers(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);

	DLLEXPORT int tetrahedralizeFun(WolframLibraryData libData, MLINK mlp);

	DLLEXPORT int getElements(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
	DLLEXPORT int setElements(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
	DLLEXPORT int getElementAttributes(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
	DLLEXPORT int setElementAttributes(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
	DLLEXPORT int setMessages(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
	DLLEXPORT int getFaces(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
	DLLEXPORT int getFaceMarkers(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
	DLLEXPORT int setHoles_or_Regions(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
	DLLEXPORT int getHoles_or_Regions(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
	DLLEXPORT int getNeighbors(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);
	DLLEXPORT int getEdges(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);

	DLLEXPORT int setTetrahedraVolumes(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res);

	DLLEXPORT bool tetunsuitable(REAL* v1, REAL* v2, REAL* v3, REAL* v4, REAL* edgeLength, REAL area);
}



static int returnZeroLengthArray( WolframLibraryData libData, mint type, MArgument res)
{
	mint dims[1] = {0};
	MTensor resTen;
	int err = libData->MTensor_new( type, 1, dims, &resTen);
	MArgument_setMTensor(res, resTen);
	return err;
}

static int returnError( WolframLibraryData libData, MTensor ten, MArgument res)
{
	libData->MTensor_disown(ten);
	MArgument_setInteger(res, 0);
	return LIBRARY_FUNCTION_ERROR;
}


/*
 Get the vertex list.


*/
DLLEXPORT int getElements(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint id = MArgument_getInteger(Args[0]);
	MTensor tenOrder = MArgument_getMTensor(Args[1]);
	int type = libData->MTensor_getType(tenOrder);
	int rank = libData->MTensor_getRank(tenOrder);


	tetgenio *instance = getTetGenInstance( id);
	if ( instance == NULL|| type != MType_Integer || rank != 1) {
		libData->MTensor_disown(tenOrder);
		return LIBRARY_FUNCTION_ERROR;
	}

	mint numTetra = instance->numberoftetrahedra;
	mint numCorners = instance->numberofcorners;

	mint dims[2] = {numTetra, numCorners};
	MTensor tenElems;
	int err = libData->MTensor_new( MType_Integer, 2, dims, &tenElems);
	if (err) return err;
	mint* rawPts = libData->MTensor_getIntegerData( tenElems);

	mint pos = 0;
	if (dims[1] == 10) {
		/* quadratic elements */
		mint *o = libData->MTensor_getIntegerData( tenOrder);
		for (mint i = 0; i < numTetra; i++) {
			for (mint j = 0; j < numCorners; j++) {
				rawPts[pos++] = instance->tetrahedronlist[i * numCorners + o[j]];
			}
		}
	} else {
		/* linear elements */
		for (mint i = 0; i < numTetra; i++) {
			for (mint j = 0; j < numCorners; j++) {
				rawPts[pos++] = instance->tetrahedronlist[i * numCorners + j];
			}
		}
	}

	libData->MTensor_disown(tenOrder);
	MArgument_setMTensor(res, tenElems);
	return 0;
}

DLLEXPORT int setElements(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint id = MArgument_getInteger(Args[0]);

	MTensor tenElems = MArgument_getMTensor(Args[1]);
	int type = libData->MTensor_getType(tenElems);
	int rank = libData->MTensor_getRank(tenElems);

	const mint* dims = libData->MTensor_getDimensions(tenElems);
	tetgenio *instance = getTetGenInstance( id);
	if (instance == NULL || type != MType_Integer || rank != 2 || !(dims[1] == 4 || dims[1] == 10)) {
		libData->MTensor_disown(tenElems);
		MArgument_setInteger(res, 0);
		return LIBRARY_FUNCTION_ERROR;
	}

	mint numTetra = dims[0];
	mint numCorners = dims[1];
	instance->numberoftetrahedra = numTetra;
	instance->numberofcorners = numCorners;
	instance->tetrahedronlist = new int[numTetra * numCorners];

	mint* elementData = libData->MTensor_getIntegerData( tenElems);

	mint pos = 0;
	for (mint i = 0; i < numTetra; i++) {
		for (mint j = 0; j < numCorners; j++) {
			instance->tetrahedronlist[i * numCorners + j] = elementData[pos++];
		}
	}

	libData->MTensor_disown(tenElems);
	MArgument_setMTensor(res, 0);

	return 0;
}


DLLEXPORT int getElementAttributes(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint id = MArgument_getInteger(Args[0]);
	tetgenio *instance = getTetGenInstance( id);
	if ( instance == NULL) {
		return LIBRARY_FUNCTION_ERROR;
	}

	mint numTetra = instance->numberoftetrahedra;
	mint numAttrs = instance->numberoftetrahedronattributes;

	mint dims[2] = {numTetra, numAttrs};
	MTensor tenAttrs;
	int err = libData->MTensor_new( MType_Real, 2, dims, &tenAttrs);
	if (err) return err;
	double* rawPts = libData->MTensor_getRealData( tenAttrs);

	mint pos = 0;
	for (mint i = 0; i < numTetra; i++) {
		for (mint j = 0; j < numAttrs; j++) {
			rawPts[pos++] = instance->tetrahedronattributelist[i * numAttrs + j];
		}
	}
	MArgument_setMTensor(res, tenAttrs);
	return 0;
}


DLLEXPORT int setElementAttributes(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint id = MArgument_getInteger(Args[0]);
	MTensor  attrsTen = MArgument_getMTensor(Args[1]);
	int type = libData->MTensor_getType(attrsTen);
	int rank = libData->MTensor_getRank(attrsTen);

	const mint* dims = libData->MTensor_getDimensions(attrsTen);
	tetgenio *instance = getTetGenInstance( id);
	if (instance == NULL || type != MType_Real || rank != 2) {
		return returnError( libData, attrsTen, res);
	}

	mint numTetra = dims[0];
	if ( numTetra != instance->numberoftetrahedra) {
		return returnError( libData, attrsTen, res);
	}

	instance->numberoftetrahedronattributes = dims[1];
	mint len = dims[0]*dims[1];

	instance->tetrahedronattributelist = new REAL[len];

	double* attrsData = libData->MTensor_getRealData(attrsTen);
	for ( mint i = 0; i < len; i++) {
		instance->tetrahedronattributelist[i] = attrsData[i];
	}

	libData->MTensor_disown(attrsTen);
	MArgument_setInteger(res, 0);
	return 0;
}


DLLEXPORT int getFaces(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint id = MArgument_getInteger(Args[0]);
	tetgenio *instance = getTetGenInstance( id);
	if ( instance == NULL) {
		return LIBRARY_FUNCTION_ERROR;
	}

	mint numFaces = instance->numberoftrifaces;
	mint numTriCorners = 3;

	mint dims[2] = {numFaces, numTriCorners};
	MTensor tenElems;
	int err = libData->MTensor_new( MType_Integer, 2, dims, &tenElems);
	if (err) return err;
	mint* rawPts = libData->MTensor_getIntegerData( tenElems);

	mint pos = 0;
	for (mint i = 0; i < numFaces; i++) {
		for (mint j = 0; j < numTriCorners; j++) {
			rawPts[pos++] = instance->trifacelist[i * numTriCorners + j];
		}
	}
	MArgument_setMTensor(res, tenElems);
	return 0;
}

DLLEXPORT int getFaceMarkers(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint dims[2];

	mint id = MArgument_getInteger(Args[0]);
	tetgenio *instance = getTetGenInstance( id);
	if (instance == NULL) {
		return LIBRARY_FUNCTION_ERROR;
	}

	dims[0] = instance->trifacemarkerlist == NULL? 0: instance->numberoftrifaces;
	mint len = dims[0];
	MTensor tenPts;
	int err = libData->MTensor_new( MType_Integer, 1, dims, &tenPts);
	if (err) return err;
	mint* rawPts = libData->MTensor_getIntegerData( tenPts);
	for ( int i = 0; i < len; i++) {
		rawPts[i] = instance->trifacemarkerlist[i];
	}
	MArgument_setMTensor(res, tenPts);
	return 0;
}


/*
 Get the facet list.

 The result is an integer MTensor of the form:

 {pointBase, numFacet, numPoly, numVertex, f1Len, f2Len, ..., p1Len, p2Len, ...   p1ind1, p1ind2, ...}

 where p1ind1 is the index into the point list, starting from pointBase
*/
DLLEXPORT int getFacets(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint id = MArgument_getInteger(Args[0]);
	tetgenio *instance = getTetGenInstance( id);
	if ( instance == NULL) {
		return LIBRARY_FUNCTION_ERROR;
	}

	mint facetNum = instance->numberoffacets;
	mint polyNum = 0;
	mint vertexNum = 0;
	for ( int i = 0; i < facetNum; i++) {
		tetgenio::facet *f = &instance->facetlist[i];
		polyNum += f->numberofpolygons;
		for (int j = 0; j < f->numberofpolygons; j++) {
			tetgenio::polygon *p = &f->polygonlist[j];
			vertexNum += p->numberofvertices;
		}
	}
	mint dims[1] = {4 + polyNum + vertexNum + facetNum};

	MTensor tenPolys;
	int err = libData->MTensor_new( MType_Integer, 1, dims, &tenPolys);
	if (err) return err;
	mint* rawPts = libData->MTensor_getIntegerData( tenPolys);

	rawPts[0] = instance->firstnumber;
	rawPts[1] = facetNum;
	rawPts[2] = polyNum;
	rawPts[3] = vertexNum;
	mint facetPos = 4;
	mint polyPos = 4 + facetNum;
	mint vertexPos = 4 + facetNum + polyNum;

	for ( mint i = 0; i < instance->numberoffacets; i++) {
		tetgenio::facet *f = &instance->facetlist[i];
		rawPts[facetPos] = f->numberofpolygons;
		facetPos++;
		for (mint j = 0; j < f->numberofpolygons; j++) {
			tetgenio::polygon *p = &f->polygonlist[j];
			rawPts[polyPos] = p->numberofvertices;
			polyPos++;
			for ( mint k = 0; k < p->numberofvertices; k++) {
				rawPts[vertexPos] = p->vertexlist[k];
				vertexPos++;
			}
		}
	}
	MArgument_setMTensor(res, tenPolys);
	return 0;
}

DLLEXPORT int setFacets(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint id = MArgument_getInteger(Args[0]);
	MTensor vertTen = MArgument_getMTensor(Args[1]);

	mint* vertData = libData->MTensor_getIntegerData( vertTen);

	mint firstNum = vertData[0];
	mint facetNum = vertData[1];
	mint polyNum = vertData[2];

	tetgenio *instance = getTetGenInstance( id);
	if (instance == NULL) {
		libData->MTensor_disown(vertTen);
		MArgument_setInteger(res, 0);
		return LIBRARY_FUNCTION_ERROR;
	}
	instance->firstnumber = firstNum;
	instance->numberoffacets = vertData[1];
	instance->facetlist = new tetgenio::facet[facetNum];

	mint facetPos = 4;
	mint polyPos = 4 + facetNum;
	mint vertPos = 4 + facetNum + polyNum;

	for ( mint i = 0; i < facetNum; i++) {
		tetgenio::facet *f = &instance->facetlist[i];
		f->numberofpolygons = vertData[facetPos++];
		f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
		f->numberofholes = 0;
		f->holelist = NULL;
		for ( mint j = 0; j < f->numberofpolygons; j++) {
			tetgenio::polygon *p = &f->polygonlist[j];
			p->numberofvertices = vertData[polyPos++];
			p->vertexlist = new int[p->numberofvertices];
			for ( int k = 0; k < p->numberofvertices; k++) {
				p->vertexlist[k] = vertData[vertPos++];
			}
		}
	}
	libData->MTensor_disown(vertTen);
	MArgument_setInteger(res, 0);
	return 0;
}


DLLEXPORT int getFacetMarkers(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint id = MArgument_getInteger(Args[0]);
	tetgenio *instance = getTetGenInstance( id);
	if ( instance == NULL) {
		return LIBRARY_FUNCTION_ERROR;
	}

	mint markerNum = instance->facetmarkerlist == NULL? 0: instance->numberoffacets;
	mint dims[1] = {markerNum};
	MTensor tenPolys;
	int err = libData->MTensor_new( MType_Integer, 1, dims, &tenPolys);
	if (err) return err;
	mint* rawPts = libData->MTensor_getIntegerData( tenPolys);
	
	for ( mint i = 0; i < markerNum; i++) {
		rawPts[i] = instance->facetmarkerlist[i];
	}
	MArgument_setMTensor(res, tenPolys);
	return 0;
}


DLLEXPORT int setFacetMarkers(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint id = MArgument_getInteger(Args[0]);
	MTensor markerTen = MArgument_getMTensor(Args[1]);

	if ( libData->MTensor_getRank( markerTen) != 1 ||
			libData->MTensor_getType( markerTen) != MType_Integer) {
		return returnError( libData, markerTen, res);
	}

	mint numMarkers = libData->MTensor_getFlattenedLength( markerTen);	
	mint* markerData = libData->MTensor_getIntegerData( markerTen);

	tetgenio *instance = getTetGenInstance( id);
	if (instance == NULL) {
		return returnError( libData, markerTen, res);
	}
	instance->facetmarkerlist = new int[numMarkers];

	for ( mint i = 0; i < numMarkers; i++) {
		instance->facetmarkerlist[i] = markerData[i];
	}
	libData->MTensor_disown(markerTen);
	MArgument_setInteger(res, 0);
	return 0;
}

DLLEXPORT int getFacetHoleLengths(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint id = MArgument_getInteger(Args[0]);
	tetgenio *instance = getTetGenInstance( id);
	if ( instance == NULL) {
		return LIBRARY_FUNCTION_ERROR;
	}

	mint facetNum = instance->numberoffacets;
	mint dims[1] = {facetNum};
	MTensor tenLens;
	int err = libData->MTensor_new( MType_Integer, 1, dims, &tenLens);
	if (err) return err;
	mint* rawPts = libData->MTensor_getIntegerData( tenLens);
	
	for ( mint i = 0; i < facetNum; i++) {
		tetgenio::facet *f = &instance->facetlist[i];
		rawPts[i] = f->numberofholes;
	}
	MArgument_setMTensor(res, tenLens);
	return 0;
}

DLLEXPORT int getFacetHoles(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint id = MArgument_getInteger(Args[0]);
	tetgenio *instance = getTetGenInstance( id);
	if ( instance == NULL) {
		return LIBRARY_FUNCTION_ERROR;
	}

	mint facetNum = instance->numberoffacets;
	mint numHoles = 0;
	for ( mint i = 0; i < facetNum; i++) {
		tetgenio::facet *f = &instance->facetlist[i];
		numHoles += f->numberofholes;
	}

	if ( numHoles == 0) {
		return returnZeroLengthArray(libData, MType_Real, res);
	}

	mint dims[2] = {numHoles, 3};
	MTensor tenHoles;
	int err = libData->MTensor_new( MType_Real, 2, dims, &tenHoles);
	if (err) return err;
	double* rawPts = libData->MTensor_getRealData( tenHoles);
	
	mint pos = 0;
	for ( mint i = 0; i < facetNum; i++) {
		tetgenio::facet *f = &instance->facetlist[i];
		for ( mint j = 0; j < f->numberofholes; j++) {
			rawPts[pos++] = f->holelist[j*3];
			rawPts[pos++] = f->holelist[j*3+1];
			rawPts[pos++] = f->holelist[j*3+2];
		}
	}
	MArgument_setMTensor(res, tenHoles);
	return 0;
}


DLLEXPORT int setFacetHoles(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint id = MArgument_getInteger(Args[0]);
	MTensor holesTen = MArgument_getMTensor(Args[1]);
	MTensor lensTen = MArgument_getMTensor(Args[2]);
	tetgenio *instance = getTetGenInstance( id);

	if ( instance == NULL ||
			libData->MTensor_getRank( lensTen) != 1 ||
			libData->MTensor_getType( lensTen) != MType_Integer || 
			libData->MTensor_getRank( holesTen) != 2 ||
			libData->MTensor_getType( holesTen) != MType_Real) {
		libData->MTensor_disown(holesTen);
		libData->MTensor_disown(lensTen);
		MArgument_setInteger(res, 0);
		return LIBRARY_FUNCTION_ERROR;
	}

	mint numLens = libData->MTensor_getFlattenedLength( lensTen);
	if ( numLens != instance->numberoffacets) {
		libData->MTensor_disown(holesTen);
		libData->MTensor_disown(lensTen);
		MArgument_setInteger(res, 0);
		return LIBRARY_FUNCTION_ERROR;
	}

	mint* lens = libData->MTensor_getIntegerData( lensTen);
	double* pts = libData->MTensor_getRealData( holesTen);
	mint pos = 0;
	for ( mint i = 0; i < numLens; i++) {
		tetgenio::facet *f = &instance->facetlist[i];
		mint len = lens[i];
		f->numberofholes = len;
		f->holelist = new REAL[len*3];
		for ( mint j = 0; j < len; j++) {
			f->holelist[j*3] = pts[pos++];
			f->holelist[j*3+1] = pts[pos++];
			f->holelist[j*3+2] = pts[pos++];
		}
	}
	libData->MTensor_disown(holesTen);
	libData->MTensor_disown(lensTen);
	MArgument_setInteger(res, 0);
	return 0;
}


DLLEXPORT int setPointList(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint id = MArgument_getInteger(Args[0]);
	MTensor tenPts = MArgument_getMTensor(Args[1]);
	int type = libData->MTensor_getType(tenPts);
	int rank = libData->MTensor_getRank(tenPts);

	const mint* dims = libData->MTensor_getDimensions(tenPts);
	tetgenio *instance = getTetGenInstance( id);
	if (instance == NULL || type != MType_Real || rank != 2 || dims[1] != 3) {
		libData->MTensor_disown(tenPts);
		MArgument_setInteger(res, 0);
		return LIBRARY_FUNCTION_ERROR;
	}

	instance->numberofpoints = dims[0];
	mint len = instance->numberofpoints*3;
	instance->pointlist = new REAL[len];
	double* rawPts = libData->MTensor_getRealData(tenPts);
	for ( mint i = 0; i < len; i++) {
		instance->pointlist[i] = rawPts[i];
	}

	libData->MTensor_disown(tenPts);
	MArgument_setInteger(res, 0);
	return 0;
}


DLLEXPORT int getPointList(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	MTensor tenPts;
	mint dims[2];
	mint id, len;
	double* rawPts;

	id = MArgument_getInteger(Args[0]);
	tetgenio *instance = getTetGenInstance( id);
	if (instance == NULL) {
		return LIBRARY_FUNCTION_ERROR;
	}

	dims[0] = instance->numberofpoints;
	dims[1] = 3;   // Maybe should use mesh_dim

	if ( dims[0] == 0) {
		return returnZeroLengthArray( libData, MType_Real, res);
	}

	len = dims[0]*dims[1];
	int err = libData->MTensor_new( MType_Real, 2, dims, &tenPts);
	if (err) return err;
	rawPts = libData->MTensor_getRealData( tenPts);
	for ( int i = 0; i < len; i++) {
		rawPts[i] = instance->pointlist[i];
	}
	MArgument_setMTensor(res, tenPts);
	return 0;
}



DLLEXPORT int setPointAttributes(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint id = MArgument_getInteger(Args[0]);
	MTensor  tenPts = MArgument_getMTensor(Args[1]);
	int type = libData->MTensor_getType(tenPts);
	int rank = libData->MTensor_getRank(tenPts);

	const mint* dims = libData->MTensor_getDimensions(tenPts);
	tetgenio *instance = getTetGenInstance( id);
	if (instance == NULL || type != MType_Real || rank != 2) {
		return returnError( libData, tenPts, res);
	}

	mint numPoints = dims[0];
	if ( numPoints != instance->numberofpoints) {
		return returnError( libData, tenPts, res);
	}

	instance->numberofpointattributes = dims[1];
	mint len = dims[0]*dims[1];
	instance->pointattributelist = new REAL[len];
	double* rawPts = libData->MTensor_getRealData(tenPts);
	for ( mint i = 0; i < len; i++) {
		instance->pointattributelist[i] = rawPts[i];
	}

	libData->MTensor_disown(tenPts);
	MArgument_setInteger(res, 0);
	return 0;
}

DLLEXPORT int getPointAttributes(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	MTensor tenPts;
	mint dims[2];
	mint id, len;
	double* rawPts;

	id = MArgument_getInteger(Args[0]);
	tetgenio *instance = getTetGenInstance( id);
	if (instance == NULL) {
		return LIBRARY_FUNCTION_ERROR;
	}

	dims[0] = instance->numberofpoints;
	dims[1] = instance->numberofpointattributes;  
	len = dims[0]*dims[1];
	int err = libData->MTensor_new( MType_Real, 2, dims, &tenPts);
	if (err) return err;
	rawPts = libData->MTensor_getRealData( tenPts);
	for ( int i = 0; i < len; i++) {
		rawPts[i] = instance->pointattributelist[i];
	}
	MArgument_setMTensor(res, tenPts);
	return 0;
}

DLLEXPORT int setPointMetricTensors(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint id = MArgument_getInteger(Args[0]);
	MTensor  tenPts = MArgument_getMTensor(Args[1]);
	int type = libData->MTensor_getType(tenPts);
	int rank = libData->MTensor_getRank(tenPts);

	const mint* dims = libData->MTensor_getDimensions(tenPts);
	tetgenio *instance = getTetGenInstance( id);
	if (instance == NULL || type != MType_Real || rank != 2) {
		return returnError( libData, tenPts, res);
	}

	mint numPoints = dims[0];
	if ( numPoints != instance->numberofpoints) {
		return returnError( libData, tenPts, res);
	}

	instance->numberofpointmtrs = dims[1];
	mint len = dims[0]*dims[1];
	instance->pointmtrlist = new REAL[len];
	double* rawPts = libData->MTensor_getRealData(tenPts);
	for ( mint i = 0; i < len; i++) {
		instance->pointmtrlist[i] = rawPts[i];
	}

	libData->MTensor_disown(tenPts);
	MArgument_setInteger(res, 0);
	return 0;
}

DLLEXPORT int getPointMetricTensors(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	MTensor tenPts;
	mint dims[2];
	mint id, len;
	double* rawPts;

	id = MArgument_getInteger(Args[0]);
	tetgenio *instance = getTetGenInstance( id);
	if (instance == NULL) {
		return LIBRARY_FUNCTION_ERROR;
	}

	dims[0] = instance->numberofpoints;
	dims[1] = instance->numberofpointmtrs;  
	len = dims[0]*dims[1];
	int err = libData->MTensor_new( MType_Real, 2, dims, &tenPts);
	if (err) return err;
	rawPts = libData->MTensor_getRealData( tenPts);
	for ( int i = 0; i < len; i++) {
		rawPts[i] = instance->pointmtrlist[i];
	}
	MArgument_setMTensor(res, tenPts);
	return 0;
}


DLLEXPORT int setPointMarkers(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint id = MArgument_getInteger(Args[0]);
	MTensor  tenPts = MArgument_getMTensor(Args[1]);
	int type = libData->MTensor_getType(tenPts);
	int rank = libData->MTensor_getRank(tenPts);

	const mint* dims = libData->MTensor_getDimensions(tenPts);
	tetgenio *instance = getTetGenInstance( id);
	if (instance == NULL || type != MType_Integer || rank != 1) {
		return returnError( libData, tenPts, res);
	}

	mint numPoints = dims[0];
	if ( numPoints != instance->numberofpoints) {
		return returnError( libData, tenPts, res);
	}

	mint len = dims[0];
	instance->pointmarkerlist = new int[len];
	mint* rawPts = libData->MTensor_getIntegerData(tenPts);
	for ( mint i = 0; i < len; i++) {
		instance->pointmarkerlist[i] = rawPts[i];
	}

	libData->MTensor_disown(tenPts);
	MArgument_setInteger(res, 0);
	return 0;
}

DLLEXPORT int getPointMarkers(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint dims[1];

	mint id = MArgument_getInteger(Args[0]);
	tetgenio *instance = getTetGenInstance( id);
	if (instance == NULL) {
		return LIBRARY_FUNCTION_ERROR;
	}

	dims[0] = instance->pointmarkerlist == NULL? 0: instance->numberofpoints; 
	mint len = dims[0];
	MTensor tenPts;
	int err = libData->MTensor_new( MType_Integer, 1, dims, &tenPts);
	if (err) return err;
	mint* rawPts = libData->MTensor_getIntegerData( tenPts);
	for ( int i = 0; i < len; i++) {
		rawPts[i] = instance->pointmarkerlist[i];
	}
	MArgument_setMTensor(res, tenPts);
	return 0;
}



DLLEXPORT int setHoles_or_Regions(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint id = MArgument_getInteger(Args[0]);
	MTensor  tenPts = MArgument_getMTensor(Args[1]);
	mint isHoles = MArgument_getInteger(Args[2]);

	int type = libData->MTensor_getType(tenPts);
	int rank = libData->MTensor_getRank(tenPts);

	const mint* dims = libData->MTensor_getDimensions(tenPts);
	tetgenio *instance = getTetGenInstance( id);
	if (instance == NULL || type != MType_Real || rank != 2) {
		return returnError( libData, tenPts, res);
	}
	if ( dims[1] != (isHoles? 3: 5)) {
		return returnError( libData, tenPts, res);
	}

	mint len = dims[0]*dims[1];
	double* rawPts = libData->MTensor_getRealData(tenPts);

	if (isHoles) {
		instance->numberofholes = dims[0];
		instance->holelist = new REAL[len];
		for ( mint i = 0; i < len; i++) {
			instance->holelist[i] = rawPts[i];
		}
	}
	else {
		instance->numberofregions = dims[0];
		instance->regionlist = new REAL[len];
		for ( mint i = 0; i < len; i++) {
			instance->regionlist[i] = rawPts[i];
		}
	}

	libData->MTensor_disown(tenPts);
	MArgument_setInteger(res, 0);
	return 0;
}


DLLEXPORT int getHoles_or_Regions(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	MTensor tenPts;
	mint dims[2];
	mint id, len;
	double* rawPts;

	id = MArgument_getInteger(Args[0]);
	mint isHoles = MArgument_getInteger(Args[1]);
	tetgenio *instance = getTetGenInstance( id);
	if (instance == NULL) {
		return LIBRARY_FUNCTION_ERROR;
	}

	if ( isHoles) {
		dims[0] = instance->numberofholes;
		dims[1] = 3;   // Maybe should use mesh_dim
	}
	else {
		dims[0] = instance->numberofregions;
		dims[1] = 5;
	}

	if ( dims[0] == 0) {
		return returnZeroLengthArray( libData, MType_Real, res);
	}

	len = dims[0]*dims[1];
	int err = libData->MTensor_new( MType_Real, 2, dims, &tenPts);
	if (err) return err;
	rawPts = libData->MTensor_getRealData( tenPts);
	if ( isHoles) {
		for ( int i = 0; i < len; i++) {
			rawPts[i] = instance->holelist[i];
		}
	}
	else {
		for ( int i = 0; i < len; i++) {
			rawPts[i] = instance->regionlist[i];
		}
	}
	MArgument_setMTensor(res, tenPts);
	return 0;
}



/*
 Get the neighbor list.


*/
DLLEXPORT int getNeighbors(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint id = MArgument_getInteger(Args[0]);
	tetgenio *instance = getTetGenInstance( id);
	if ( instance == NULL) {
		return LIBRARY_FUNCTION_ERROR;
	}

	if( instance->neighborlist == (int *) NULL) {
		return LIBRARY_FUNCTION_ERROR;
	}
	mint numTetra = instance->numberoftetrahedra;
	mint numFaces = 4;

	mint dims[2] = {numTetra, numFaces};
	MTensor tenNeighbors;
	int err = libData->MTensor_new( MType_Integer, 2, dims, &tenNeighbors);
	if (err) return err;
	mint* rawPts = libData->MTensor_getIntegerData( tenNeighbors);

	mint pos = 0;
	for (mint i = 0; i < numTetra; i++) {
		for (mint j = 0; j < numFaces; j++) {
			rawPts[pos++] = instance->neighborlist[i * numFaces + j];
		}
	}
	MArgument_setMTensor(res, tenNeighbors);
	return 0;
}

/*
 Get the edges list.


*/
DLLEXPORT int getEdges(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint id = MArgument_getInteger(Args[0]);
	tetgenio *instance = getTetGenInstance( id);
	if ( instance == NULL) {
		return LIBRARY_FUNCTION_ERROR;
	}

	if( instance->edgelist == (int *) NULL) {
		return LIBRARY_FUNCTION_ERROR;
	}
	mint numEdges = instance->numberofedges;
	mint numEdgeIncidents = 2;

	mint dims[2] = {numEdges, numEdgeIncidents};
	MTensor tenEdges;
	int err = libData->MTensor_new( MType_Integer, 2, dims, &tenEdges);
	if (err) return err;
	mint* rawPts = libData->MTensor_getIntegerData( tenEdges);

	mint pos = 0;
	for (mint i = 0; i < numEdges; i++) {
		for (mint j = 0; j < numEdgeIncidents; j++) {
			rawPts[pos++] = instance->edgelist[i * numEdgeIncidents + j];
		}
	}
	MArgument_setMTensor(res, tenEdges);
	return 0;
}



static int returnRes( MLINK mlp, const char* str1, const char* str2, int res)
{
	if ( str1 != NULL)
		MLReleaseString(mlp, str1);
	if ( str2 != NULL)
		MLReleaseString(mlp, str2);
	return res;
}

static int returnRes( MLINK mlp, const char* str1, int res)
{
	if ( str1 != NULL)
		MLReleaseString(mlp, str1);
	return res;
}

DLLEXPORT int fileOperation(WolframLibraryData libData, MLINK mlp)
{
	int res = LIBRARY_FUNCTION_ERROR;
	int id;

#if MLINTERFACE >= 4
	int len;
#else
	long len;
#endif

	const char *fName = NULL;
	const char *fType = NULL;

#if MLINTERFACE >= 4
	if ( !MLTestHead( mlp, "List", &len))
#else
	if ( !MLCheckFunction( mlp, "List", &len))
#endif
		return returnRes(mlp, fName, fType, LIBRARY_FUNCTION_ERROR);

	if ( len != 3)
		return returnRes(mlp, fName, fType, LIBRARY_FUNCTION_ERROR);

	if (!MLGetInteger( mlp, &id))
		return returnRes(mlp, fName, fType, LIBRARY_FUNCTION_ERROR);

	if(! MLGetString(mlp, &fName))
		return returnRes(mlp, fName, fType, LIBRARY_FUNCTION_ERROR);

	if(! MLGetString(mlp, &fType))
		return returnRes(mlp, fName, fType, LIBRARY_FUNCTION_ERROR);

	if ( ! MLNewPacket(mlp) )
		return returnRes(mlp, fName, fType, LIBRARY_FUNCTION_ERROR);

	const char* resStr = "True";

	tetgenio* instance = getTetGenInstance(id);
	if (instance == NULL) {
		resStr = "False";
	}
	else if ( strcmp( fType, "load_node") == 0) {
		if ( !instance->load_node( (char*)fName)) {
			resStr = "False";
		}
	}
	else if ( strcmp( fType, "load_poly") == 0) {
		if ( !instance->load_poly( (char*)fName)) {
			resStr = "False";
		}
	}
#if 0
	else if ( strcmp( fType, "load_pbc") == 0) {
		if ( !instance->load_pbc( (char*)fName)) {
			resStr = "False";
		}
	}
#endif
	else if ( strcmp( fType, "load_var") == 0) {
		if ( !instance->load_var( (char*)fName)) {
			resStr = "False";
		}
	}
	else if ( strcmp( fType, "load_mtr") == 0) {
		if ( !instance->load_mtr( (char*)fName)) {
			resStr = "False";
		}
	}
	else if ( strcmp( fType, "load_off") == 0) {
		if ( !instance->load_off( (char*)fName)) {
			resStr = "False";
		}
	}
	else if ( strcmp( fType, "load_ply") == 0) {
		if ( !instance->load_ply( (char*)fName)) {
			resStr = "False";
		}
	}
	else if ( strcmp( fType, "load_stl") == 0) {
		if ( !instance->load_stl( (char*)fName)) {
			resStr = "False";
		}
	}
	else if ( strcmp( fType, "load_medit") == 0) {
		if ( !instance->load_medit( (char*)fName)) {
			resStr = "False";
		}
	}
	else if ( strcmp( fType, "load_tetmesh") == 0) {
		if ( !instance->load_tetmesh( (char*)fName)) {
			resStr = "False";
		}
	}
#if 0
	else if ( strcmp( fType, "load_voronoi") == 0) {
		if ( !instance->load_voronoi( (char*)fName)) {
			resStr = "False";
		}
	}
#endif
	else if ( strcmp( fType, "save_poly") == 0) {
		instance->save_poly( (char*)fName);
	}
	else if ( strcmp( fType, "save_nodes") == 0) {
		instance->save_nodes( (char*)fName);
	}
	else if ( strcmp( fType, "save_neighbors") == 0) {
		instance->save_neighbors( (char*)fName);
	}
	else if ( strcmp( fType, "save_faces") == 0) {
		instance->save_faces( (char*)fName);
	}
	else if ( strcmp( fType, "save_elements") == 0) {
		instance->save_elements( (char*)fName);
	}
	else if ( strcmp( fType, "save_edges") == 0) {
		instance->save_edges( (char*)fName);
	}
	if ( MLPutSymbol( mlp, resStr)) {
		res = 0;
	}

	return returnRes(mlp, fName, fType, res);
}







static WolframLibraryData gl_LibData = NULL;


/*
  Definitely not the right way to make the global WolframLibraryData available. 
  Really this should be stored/cleared at the start/end of every export function.
*/
static void sendMessage( char* mess)
{
//	gl_LibData->Message( mess);
	if ( gl_LibData == NULL) {
		return;
	}
	MLINK link = gl_LibData->getMathLink(gl_LibData);
	MLPutFunction( link, "EvaluatePacket", 1);
	MLPutFunction( link, "Print", 3);
	MLPutSymbol( link, "TetGenLink");
	MLPutString( link, " info ");
	MLPutString( link, mess);
	gl_LibData->processMathLink( link);
	int pkt = MLNextPacket( link);
	if ( pkt == RETURNPKT) {
		MLNewPacket(link);
	}
}


typedef void (*addMessage)( char*);


/*
 * setMessageFunction relies on modifications to TetGen
 * If you don't want to modify TetGen then define NOMESSAGEFUNCTION
 */

#ifdef NOMESSAGEFUNCTION
#define setMessageFunction(arg)
#else
extern void setMessageFunction(addMessage fun);
#endif

DLLEXPORT int setMessages(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint arg = MArgument_getInteger(Args[0]);

	if ( arg) {
		gl_LibData = libData;
		setMessageFunction( sendMessage);
	}
	else {
		gl_LibData = NULL;
		setMessageFunction( NULL);
	}
	MArgument_setInteger(res, 0);
	return 0;
}

WolframLibraryData savedLibData = NULL;

DLLEXPORT int tetrahedralizeFun(WolframLibraryData libData, MLINK mlp)
{
	int err = LIBRARY_FUNCTION_ERROR;
	int idIn, idOut;
#if MLINTERFACE >= 4
	int len;
#else
	long len;
#endif
	const char *fSwitches = NULL;
	int refine = 0;

	/* after an abort during tetrahedralize there may still be a "True" on the
	 * link. bug 268570 */
	if ( MLGetNext(mlp) == MLTKSYM) {
			MLNewPacket(mlp);
	}

#if MLINTERFACE >= 4
	if ( !MLTestHead( mlp, "List", &len))
#else
	if ( !MLCheckFunction( mlp, "List", &len))
#endif
		return returnRes(mlp, fSwitches, LIBRARY_FUNCTION_ERROR);

	if ( len != 4)
		return returnRes(mlp, fSwitches, LIBRARY_FUNCTION_ERROR);

	if ( !MLGetString(mlp, &fSwitches))
		return returnRes(mlp, fSwitches, LIBRARY_FUNCTION_ERROR);

	if ( !MLGetInteger(mlp, &idIn))
		return returnRes(mlp, fSwitches, LIBRARY_FUNCTION_ERROR);

	if ( !MLGetInteger(mlp, &idOut))
		return returnRes(mlp, fSwitches, LIBRARY_FUNCTION_ERROR);

	if ( !MLGetInteger(mlp, &refine))
		return returnRes(mlp, fSwitches, LIBRARY_FUNCTION_ERROR);

	if ( !MLNewPacket(mlp) )
		return returnRes(mlp, fSwitches, LIBRARY_FUNCTION_ERROR);

	tetgenio* in = getTetGenInstance(idIn);
	if ( in == NULL)
		return returnRes(mlp, fSwitches, LIBRARY_FUNCTION_ERROR);

	tetgenio* out = getTetGenInstance(idOut);
	if ( out == NULL)
		return returnRes(mlp, fSwitches, LIBRARY_FUNCTION_ERROR);

	if ( in != NULL && out != NULL) {
		try {
			/* the call back function needs libData */
			savedLibData = libData;
			if (refine) {
				in->tetunsuitable = &tetunsuitable;
			} else {
				in->tetunsuitable = NULL;
			}
			// in tetgen.cxx if (b->metric) branch on 28557
			// and associated else is commented out - background meshes may
			// not work as intended.
			tetrahedralize((char*)fSwitches, in, out, NULL, NULL);
			savedLibData = NULL;
			if ( MLPutSymbol( mlp, "True")) {
				err = 0;
			}
		}
		catch (int tetgetErr) {
			if ( MLPutInteger( mlp, tetgetErr)) {
				err = 0;
			}
		}
	}
	else {
		if ( MLPutSymbol( mlp, "False")) {
			err = 0;
		}
	}

	return returnRes(mlp, fSwitches, err);
}

DLLEXPORT int setTetrahedraVolumes(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{

	mint i;
	mint id;
	MTensor tenPts;
	int type;
	int rank;
	const mint* dims;
	mint numTetrahedraVolumes;
	mint len;
	double* rawPts;

	id = MArgument_getInteger(Args[0]);
	tenPts = MArgument_getMTensor(Args[1]);
	type = libData->MTensor_getType(tenPts);
	rank = libData->MTensor_getRank(tenPts);

	dims = libData->MTensor_getDimensions(tenPts);
	tetgenio *instance = getTetGenInstance( id);
	if (instance == NULL || type != MType_Real || rank != 1) {
		return returnError( libData, tenPts, res);
	}

	numTetrahedraVolumes = dims[0];
	if ( numTetrahedraVolumes != instance->numberoftetrahedra) {
		return returnError( libData, tenPts, res);
	}

	len = dims[0];

	instance->tetrahedronvolumelist = (REAL *) malloc(numTetrahedraVolumes * sizeof(REAL));
	rawPts = libData->MTensor_getRealData(tenPts);
	for ( i = 0; i < len; i++) {
		instance->tetrahedronvolumelist[i] = rawPts[i];
	}

	libData->MTensor_disown(tenPts);
	MArgument_setMTensor(res, 0);
	return 0;
}

/* el - edge length */
DLLEXPORT bool tetunsuitable(REAL* v1, REAL* v2, REAL* v3, REAL* v4, REAL* el, REAL area)
{
	mint Argc = 2;
	int err = 0;
	mbool refine = 0;
	mint dims[2];
	/*mint edims[1];*/
	mreal *r, *e;
	MArgument Args[3];
	/*MArgument Res;*/
	double darea = area;
	MTensor T0 = 0/*, T1 = 0*/;

	dims[0] = 4;
	dims[1] = 3;

	/*edims[0] = 6;*/

	/*assert(sizeof(REAL) == sizeof(double));*/
	/*assert(sizeof(mint) == sizeof(int));*/

	if (savedLibData) {

		err = (savedLibData->MTensor_new)(MType_Real, 2, dims, &T0);
		if (err) return err;
		r = (savedLibData->MTensor_getRealData)(T0);

		r[0] = v1[0]; r[1]  = v1[1]; r[2]  = v1[2];
		r[3] = v2[0]; r[4]  = v2[1]; r[5]  = v2[2];
		r[6] = v3[0]; r[7]  = v3[1]; r[8]  = v3[2];
		r[9] = v4[0]; r[10] = v4[1]; r[11] = v4[2];

		/*err = (savedLibData->MTensor_new)(MType_Real, 1, edims, &T1);
		if (err) return err;
		e = (savedLibData->MTensor_getRealData)(T1);

		e[0] = el[0]; e[1] = el[1]; e[2] = el[2];
		e[3] = el[3]; e[4] = el[4]; e[5] = el[5];*/

		MArgument_getMTensorAddress(Args[0]) = &T0;
		/*MArgument_getMTensorAddress(Args[1]) = &T1;*/
		MArgument_getRealAddress(Args[1]) = &darea;
		MArgument_getBooleanAddress(Args[2]) = &refine;

		err = tetUnsuitableCallback(savedLibData, Argc, Args, Args[2]);
		(*savedLibData->MTensor_free)(T0);
		/*(*savedLibData->MTensor_free)(T1);*/
		if (err) return 0;
	}
	return (refine == 0) ? false : true;
}


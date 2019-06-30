(* Mathematica Package *)

(* Created by the Wolfram Workbench 24-Feb-2009 *)

BeginPackage["TetGenLink`"]

$TetGenLibrary::usage = "$TetGenLibrary is the full path to the TetGen Library loaded by TetGenLink."

TetGenExpression::usage = "TetGenExpression[ id] represents an instance of a TetGen object."

TetGenExpressionQ::usage = "TetGenExpressionQ[ expr] returns True if expr represents an active instance of a TetGen object."

LoadTetGen::usage = "LoadTetGen[] loads the TetGen Library."

TetGenSetMessages::usage = "TetGenSetMessages[True|False] enables or disables the issuing of messages from TetGen."

TetGenCreate::usage = "TetGenCreate[] creates an instance of a TetGen expression."

TetGenDelete::usage = "TetGenDelete[ expr] removes an instance of a TetGen expression, freeing up memory."

TetGenExpressions::usage = "TetGenExpressions[] returns a list of active TetGen expressions."

TetGenGetFacets::usage = "TetGenGetFacets[ expr] returns the facets for a TetGen expression."

TetGenSetFacets::usage = "TetGenSetFacets[ expr, vertices] sets the facets for a TetGen expression.  "

TetGenGetFacetMarkers::usage = "TetGenGetFacetMarkers[ expr] returns the facet markers for a TetGen expression."

TetGenSetFacetMarkers::usage = "TetGenSetFacetMarkers[ expr, vertices] sets the facet markers for a TetGen expression.  "

TetGenSetFacetHoles::usage = "TetGenSetFacetHoles[ expr, holes] sets the holes in the facets."

TetGenGetFacetHoles::usage = "TetGenGetFacetHoles[ expr] gets the holes in the facets."

TetGenTetrahedralize::usage = "TetGenTetrahedralize[ expr, args] tetrahedralizes a TetGen expression using args, it returns the result in a new TetGen expression. "

TetGenGetElements::usage = "TetGenGetElements[expr] gets the elements in a TetGen expression."

TetGenSetElements::usage = "TetGenSetElements[expr, elements] sets the elements in a TetGen expression."

TetGenGetElementAttributes::usage = "TetGenGetElementAttributes[expr] gets the element attributes in a TetGen expression."

TetGenSetElementAttributes::usage = "TetGenSetElementAttributes[expr, attrs] sets the element attributes in a TetGen expression."

TetGenGetFaces::usage = "TetGenGetFaces[ expr] gets the faces in a TetGen expression."

TetGenGetFaceMarkers::usage = "TetGenGetFaceMarkers[ expr] returns the face markers for a TetGen expression."

TetGenImport::usage = "TetGenImport[ \"file.ext\", expr] imports data from a file into a TetGen expression. TetGenImport[ \"file\", expr, \"format\"] imports data in the specified format."

TetGenExport::usage = "TetGenExport[ \"file.ext\", expr] exports data from a TetGen expression into a file. TetGenExport[ \"file\", expr, \"format\"] exports data in the specified format."

TetGenGetPoints::usage = "TetGenGetPoints[ expr] returns the points in a TetGen expression."

TetGenSetPoints::usage = "TetGenSetPoints[ expr, points] sets the points in a TetGen expression."

TetGenGetPointAttributes::usage = "TetGenGetPointAttributes[ expr] returns the point attributes in a TetGen expression."

TetGenSetPointAttributes::usage = "TetGenSetPointAttributes[ expr, attrs] sets the point attributes in a TetGen expression."

TetGenGetPointMetricTensors::usage = "TetGenGetPointMetricTensors[ expr] returns the point metric tensors in a TetGen expression."

TetGenSetPointMetricTensors::usage = "TetGenSetPointMetricTensors[ expr, tens] sets the point metric tensors in a TetGen expression."

TetGenGetPointMarkers::usage = "TetGenGetPointMarkers[ expr] returns the point markers in a TetGen expression."

TetGenSetPointMarkers::usage = "TetGenSetPointMarkers[ expr, markers] sets the point markers in a TetGen expression."

TetGenGetHoles::usage = "TetGenGetHoles[ expr] returns the holes in a TetGen expression."

TetGenSetHoles::usage = "TetGenSetHoles[ expr, points] sets the holes in a TetGen expression."

TetGenGetRegions::usage = "TetGenGetRegions[ expr] returns the regions in a TetGen expression."

TetGenSetRegions::usage = "TetGenSetRegions[ expr, pts, index, attrs] sets the regions in a TetGen expression."

TetGenGetNeighbors::usage = "TetGenGetNeighbors[expr] gets the neighbors in a TetGen expression."

TetGenGetEdges::usage = "TetGenGetEdges[expr] gets the edges in a TetGen expression."

TetGenLink::info = "`1`"

TetGenDelaunay::usage = "TetGenDelaunay[ points] generates a Delaunay tetrahedralization for a 3D point set."

TetGenConvexHull::usage = "TetGenConvexHull[ points] generates a convex hull for a 3D point set."

TetGenDetectIntersectingFacets::usage = "TetGenDetectIntersectingFacets[ points, facets] returns a list of points and intersecting facets.";

TetGenIntersectingFacetsQ::usage = "TetGenIntersectingFacetsQ[ points, facets] returns True if the facetes intersect.";

$TetGenInstallationDirectory::usage = "$TetGenInstallationDirectory gives the top-level directory in which your TetGen installation resides."

TetGenSetTetrahedraVolumes::usage = "TetGenSetTetrahedraVolumes[ expr, volumes] sets the volumes for tetrahedra in a TetGen expression."

$TetGenVersion::usage = "$TetGenVersion gives the version number of the TetGen library."

Options[TetGenTetrahedralize] = {"TetrahedronRefinement" -> None};

Begin["`Private`"]
(* Implementation of the package *)

pack = Developer`ToPackedArray;

$TetGenVersion = "1.4.3.2"

$TetGenLibrary = FindLibrary[ "tetgenWolfram"]


needInitialization = True;

$TetGenInstallationDirectory = DirectoryName[ $InputFileName]


(*
 Load all the functions from the TetGen library.
*)
LoadTetGen[] :=
	Module[{},
		deleteFun	= LibraryFunctionLoad[$TetGenLibrary, "deleteTetGenInstance", {Integer}, Integer];
		instanceListFun	= LibraryFunctionLoad[$TetGenLibrary, "instanceList", {}, {Integer,1}];

		fileOperationFun = LibraryFunctionLoad[$TetGenLibrary, "fileOperation", LinkObject, LinkObject];

		getPointsFun = LibraryFunctionLoad[$TetGenLibrary, "getPointList", {Integer}, {Real,_}];
		setPointsFun = LibraryFunctionLoad[$TetGenLibrary, "setPointList", {Integer, {Real, 2, "Shared"}}, Integer];
		getPointAttributesFun = LibraryFunctionLoad[$TetGenLibrary, "getPointAttributes", {Integer}, {Real,2}];
		setPointAttributesFun = LibraryFunctionLoad[$TetGenLibrary, "setPointAttributes", {Integer, {Real, 2, "Shared"}}, Integer];
		getPointMetricTensorsFun = LibraryFunctionLoad[$TetGenLibrary, "getPointMetricTensors", {Integer}, {Real,2}];
		setPointMetricTensorsFun = LibraryFunctionLoad[$TetGenLibrary, "setPointMetricTensors", {Integer, {Real, 2, "Shared"}}, Integer];
		getPointMarkersFun = LibraryFunctionLoad[$TetGenLibrary, "getPointMarkers", {Integer}, {Integer,1}];
		setPointMarkersFun = LibraryFunctionLoad[$TetGenLibrary, "setPointMarkers", {Integer, {Integer, 1, "Shared"}}, Integer];

		getHolesOrRegionsFun = LibraryFunctionLoad[$TetGenLibrary, "getHoles_or_Regions", {Integer, Integer}, {Real,_}];
		setHolesOrRegionsFun = LibraryFunctionLoad[$TetGenLibrary, "setHoles_or_Regions", {Integer, {Real, 2, "Shared"}, Integer}, Integer];

		getFacetsFun = LibraryFunctionLoad[$TetGenLibrary, "getFacets", {Integer}, {Integer,1}];
		setFacetsFun = LibraryFunctionLoad[$TetGenLibrary, "setFacets", {Integer, {Integer, 1, "Shared"}}, Integer];
		getFacetMarkersFun = LibraryFunctionLoad[$TetGenLibrary, "getFacetMarkers", {Integer}, {Integer,1}];
		setFacetMarkersFun = LibraryFunctionLoad[$TetGenLibrary, "setFacetMarkers", {Integer, {Integer, 1, "Shared"}}, Integer];
		getFacetHolesFun = LibraryFunctionLoad[$TetGenLibrary, "getFacetHoles", {Integer}, {Real,_}];
		getFacetHoleLengthsFun = LibraryFunctionLoad[$TetGenLibrary, "getFacetHoleLengths", {Integer}, {Integer,1}];
		setFacetHolesFun = LibraryFunctionLoad[$TetGenLibrary, "setFacetHoles", {Integer, {Real, 2, "Shared"}, {Integer, 1, "Shared"}}, Integer];

		tetrahedralizeFun = LibraryFunctionLoad[$TetGenLibrary, "tetrahedralizeFun", LinkObject, LinkObject];
		getElementsFun = LibraryFunctionLoad[$TetGenLibrary, "getElements", {Integer, {Integer, 1, "Shared"}}, {Integer, 2}];
		setElementsFun = LibraryFunctionLoad[$TetGenLibrary, "setElements", {Integer, {Integer, 2, "Shared"}}, Integer];
		getElementAttrsFun = LibraryFunctionLoad[$TetGenLibrary, "getElementAttributes", {Integer}, {Real, 2}];
		setElementAttrsFun = LibraryFunctionLoad[$TetGenLibrary, "setElementAttributes", {Integer, {Real, 2, "Shared"}}, Integer];
		getFacesFun = LibraryFunctionLoad[$TetGenLibrary, "getFaces", {Integer}, {Integer,2}];
		getFaceMarkersFun = LibraryFunctionLoad[$TetGenLibrary, "getFaceMarkers", {Integer}, {Integer,1}];
		setMessagesFun = LibraryFunctionLoad[$TetGenLibrary, "setMessages", {Integer}, Integer];
		getNeighborsFun = LibraryFunctionLoad[$TetGenLibrary, "getNeighbors", {Integer}, {Integer,2}];
		getEdgesFun = LibraryFunctionLoad[$TetGenLibrary, "getEdges", {Integer}, {Integer,2}];

		setTetrahedraVolumesFun = LibraryFunctionLoad[$TetGenLibrary, "setTetrahedraVolumes", {Integer, {Real, 1, "Shared"}}, Integer];

		tetUnsuitableCallback = LibraryFunctionLoad[$TetGenLibrary, "tetUnsuitableCallback", {{Real, 2}, Real}, True|False];

		needInitialization = False;
	]


(*
 Functions for working with TetGenExpression
*)
getTetGenExpressionID[e_TetGenExpression] := ManagedLibraryExpressionID[e, "TetGenManager"];

TetGenExpressionQ[e_TetGenExpression] := ManagedLibraryExpressionQ[e, "TetGenManager"];

testTetGenExpression[][e_] := testTetGenExpression[TetGenExpression][TetGenExpression][e];

testTetGenExpression[mhead_Symbol][e_] := 
	If[
		TrueQ[TetGenExpressionQ[e]], 
		True,
		Message[MessageName[mhead,"tginst"], e]; False
	];
	
testTetGenExpression[_][e_] := 	TrueQ[TetGenExpressionQ[e]];
	
General::tginst = "`1` does not represent an active TetGen object.";

TetGenCreate[] :=
	Module[{},
		If[ needInitialization, LoadTetGen[]];
		CreateManagedLibraryExpression["TetGenManager", TetGenExpression]
	]

TetGenDelete[TetGenExpression[ id_]?(testTetGenExpression[TetGenDelete])] :=
	Module[{},
		deleteFun[ id]
	]

TetGenDelete[ l:{_TetGenExpression..}] := TetGenDelete /@ l

TetGenExpressions[] :=
	Module[ {list},
		If[ needInitialization, LoadTetGen[]];
		list = instanceListFun[];
		If[ !ListQ[ list], 
			$Failed,
			Map[ TetGenExpression, list]]
	]



(*
 Functions for with the file formats that TetGen supports
*)
TetGenImport::file = "An error was found loading `1` file, `2`. Try Import as an alternative."

TetGenImport[ file:(_String|_File), TetGenExpression[ id_]?(testTetGenExpression[TetGenImport]), form_String] :=
	Module[{res, formName, fileName, fileWithExtension,
		fns, newDir, validDirQ},
		formName = Switch[ form,
						"node", "load_node",
						"poly", "load_poly",
						"pbc", "load_pbc",
						"var", "load_var",
						"mtr", "load_mtr",
						"off", "load_off",
						"ply", "load_ply",
						"stl", "load_stl",
						"mesh", "load_medit",
						"tetmesh", "load_tetmesh",
						"voronoi", "load_voronoi",
						_, $Failed];
		If[ formName === $Failed, Return[ $Failed]];

		fileWithExtension = file;
		If[ FileExtension[file]=="",
			fileWithExtension = StringJoin[file, ".", form];
		];

		(* bug: 191880 *)
		fns = FileNameSplit[file];

		If[ Length[fns] == 0,
			Message[TetGenImport::file, form, fileWithExtension];
			Return[$Failed];
		];

		If[ Length[fns] >= 1,
			fileName = FileBaseName[ Last[fns]];
		,
			Message[TetGenImport::file, form, fileWithExtension];
			Return[$Failed];
		];

		If[ Length[fns] > 1,
			newDir = FileNameJoin[Most[fns]];
			validDirQ = DirectoryQ[newDir];
			If[ !validDirQ,
				Message[TetGenImport::file, form, fileWithExtension];
				Return[$Failed];
			,
				SetDirectory[newDir];
			];
		];

		If[ !FileExistsQ[StringJoin[{fileName, ".", form}]],
			If[ validDirQ, ResetDirectory[]];
			Message[TetGenImport::file, form, fileWithExtension];
			Return[$Failed];
		];

		res = fileOperationFun[ id, fileName, formName];

		If[ validDirQ, ResetDirectory[]; ];

		If[ res =!= True,
			Message[ TetGenImport::file, form, fileWithExtension];
			$Failed;
		,
			Null
		];
	]

TetGenExport::file = "An error was found saving `1` file, `2`."

TetGenExport[ file:(_String|_File), TetGenExpression[ id_]?(testTetGenExpression[TetGenExport]), form_String] :=
	Module[{res, formName, fileName, fileWithExtension,
		fns, newDir, validDirQ},
		formName = Switch[ form,
						"node", "save_nodes",
						"poly", "save_poly",
						"neigh", "save_neighbors",
						"face", "save_faces",
						"element", "save_elements",
						"ele", "save_elements",
						"edge", "save_edges",
						_, $Failed];
		If[ formName === $Failed, Return[ $Failed]];

		fileWithExtension = file;
		If[ FileExtension[file]=="",
			fileWithExtension = StringJoin[file, ".", form];
		];

		(* bug: 191880 *)
		fns = FileNameSplit[file];

		If[ Length[fns] == 0,
			Message[TetGenExport::file, form, fileWithExtension];
			Return[$Failed];
		];

		If[ Length[fns] >= 1,
			fileName = FileBaseName[ Last[fns]];
			,
			Message[TetGenExport::file, form, fileWithExtension];
			Return[$Failed];
		];

		If[ Length[fns] > 1,
			newDir = FileNameJoin[Most[fns]];
			validDirQ = DirectoryQ[newDir];
			If[ !validDirQ,
				Message[TetGenExport::file, form, fileWithExtension];
				Return[$Failed];
			,
			SetDirectory[newDir];
			];
		];

		res = fileOperationFun[ id, fileName, formName];

		If[ validDirQ,
			ResetDirectory[];
		];

		If[ res =!= True,
			Message[ TetGenExport::file, form, fileWithExtension];
			$Failed
			,
			Null
		];
	]

TetGenImport[ file:(_String|_File), TetGenExpression[ id_]?(testTetGenExpression[TetGenImport])] :=
	Module[{dir, fileName, ext},
		dir = FileNameDrop[ file, -1];
		fileName = FileBaseName[ file];
		ext = FileExtension[file];
		If[ dir =!= "", fileName = FileNameJoin[{dir, fileName}]];
		TetGenImport[ fileName, TetGenExpression[id], ext]
	]

TetGenExport[ file:(_String|_File), TetGenExpression[ id_]?(testTetGenExpression[TetGenExport])] :=
	Module[{dir, fileName, ext},
		dir = FileNameDrop[ file, -1];
		fileName = FileBaseName[ file];
		ext = FileExtension[file];
		If[ dir =!= "", fileName = FileNameJoin[{dir, fileName}]];
		TetGenExport[fileName,  TetGenExpression[id], ext]
	]


(*
 These are mostly calls to the corresponding TetGen function
*)
TetGenGetPoints[ TetGenExpression[ id_]?(testTetGenExpression[TetGenGetPoints])] :=
	Module[{},
		getPointsFun[ id]
	]

TetGenSetPoints[ TetGenExpression[ id_]?(testTetGenExpression[TetGenSetPoints]), pts_] :=
	Module[{},
		setPointsFun[ id, pack[pts]]
	]

TetGenGetPointAttributes[ TetGenExpression[ id_]?(testTetGenExpression[TetGenGetPointAttributes])] :=
	Module[{},
		getPointAttributesFun[ id]
	]

TetGenSetPointAttributes[ TetGenExpression[ id_]?(testTetGenExpression[TetGenSetPointAttributes]), pts_] :=
	Module[{},
		setPointAttributesFun[ id, pack[pts]]
	]

TetGenGetPointMetricTensors[ TetGenExpression[ id_]?(testTetGenExpression[TetGenGetPointMetricTensors])] :=
	Module[{},
		getPointMetricTensorsFun[ id]
	]

TetGenSetPointMetricTensors[ TetGenExpression[ id_]?(testTetGenExpression[TetGenSetPointMetricTensors]), pts_] :=
	Module[{},
		setPointMetricTensorsFun[ id, pack[pts]]
	]
	
TetGenGetPointMarkers[ TetGenExpression[ id_]?(testTetGenExpression[TetGenGetPointMarkers])] :=
	Module[{},
		getPointMarkersFun[ id]
	]

TetGenSetPointMarkers[ TetGenExpression[ id_]?(testTetGenExpression[TetGenSetPointMarkers]), pts_] :=
	Module[{},
		setPointMarkersFun[ id, pack[pts]]
	]


TetGenGetHoles[ TetGenExpression[ id_]?(testTetGenExpression[TetGenGetHoles])] :=
	Module[{},
		getHolesOrRegionsFun[ id, 1]
	]

TetGenSetHoles[ TetGenExpression[ id_]?(testTetGenExpression[TetGenSetHoles]), pts_ /; ArrayQ[pts,2]] :=
	Module[{},
		setHolesOrRegionsFun[ id, pack[pts], 1]
	]


TetGenGetRegions[ TetGenExpression[ id_]?(testTetGenExpression[TetGenGetRegions])] :=
	Module[{val},
		val = getHolesOrRegionsFun[ id, 0];
		If[ !MatchQ[ val, {{_,_,_,_,_}..}], Return[ val]];
		{ Part[ val, All, {1,2,3}], Part[ val, All, 4], Part[ val, All, 5]}
	]


TetGenSetRegions::indlen = "The number of region indices, `1`, does not match the number of points, `2`."
TetGenSetRegions::attrlen = "The number of region attributes, `1`, does not match the number of points, `2`."


TetGenSetRegions[ TetGenExpression[ id_]?(testTetGenExpression[TetGenSetRegions]), regPts: {{_,_,_}..}, regInd_ /; ArrayQ[regInd,1], regAttrs_/; ArrayQ[regAttrs,1]] :=
	Module[{len, arg},
		len = Length[ regPts];
		If[ Length[ regInd] =!= len, 
				Message[TetGenSetRegions::indlen, len, Length[ regInd]]; 
				Return[ $Failed]];
		If[ Length[ regAttrs] =!= len, 
				Message[TetGenSetRegions::attrlen, len, Length[ regAttrs]]; 
				Return[ $Failed]];
		arg = Join[regPts, Transpose[{regInd}], Transpose[{regAttrs}], 2];
		arg = pack[ N[arg]];
		setHolesOrRegionsFun[ id, arg, 0]
	]





(*
 Return the indices into the points of the vertices.  We do this keeping 
 track of the facet structure.   So the result is a list 
 { facet1, facet2, ...},  where faceti is the polys of the ith facet, 
 and is represented as { {p11, p12, p13,...}, {p21, p22, p23, ...}, ...}, 
 where p11 is an index into the points list.
 
 The result of the getFacetsFun is a list
 
 {pointBase, numFacet, numPoly, numVertex, f1Len, f2Len, ..., p1Len, p2Len, ...   p1ind1, p1ind2, ...}
 
 TODO apply the pointBase to the points in the poly list.
*)


TetGenGetFacets[ TetGenExpression[ id_]?(testTetGenExpression[TetGenGetFacets])] :=
	Module[{verts, pointBase, numFacet, numPoly, numVertex, facetLens, polyLens, polys},
		verts = getFacetsFun[ id];
		pointBase = Part[verts,1];
		numFacet = Part[verts,2];
		numPoly = Part[verts, 3];
		numVertex = Part[verts, 4];
		facetLens = Take[ verts, {5, 4+numFacet}];
		polyLens = Take[ verts, {5+numFacet, 4+numFacet+numPoly}];
		polys = takeLens[ verts, polyLens, 4+numFacet+numPoly];
		polys = takeLens[ polys, facetLens, 0]
	]

takeLens[ list_, lenList_, start_] :=
	Module[ {inds, t},
		inds = FoldList[Plus, start, lenList];
		t = Partition[inds, 2, 1];
		inds = Map[# + {1, 0} &, Partition[inds, 2, 1]];
		Map[ Take[ list, #]&, inds]
	]

getLength := 
	getLength = 
		Compile[{{in, _Integer, 2}}, Length[in], 
			RuntimeAttributes -> Listable]

getPoly := 
	getPoly = 
		Compile[{{in, _Integer, 1}}, Length[in], 
			RuntimeAttributes -> Listable]

TetGenSetFacets[ TetGenExpression[ id_]?(testTetGenExpression[TetGenSetFacets]), verts:{{{__Integer} ..} ..}] :=
	Module[{numFacet, facetLength, polys, numPoly, inds, numInds, arg},
		numFacet = Length[verts];
		facetLength = getLength[verts];
		polys = Flatten[ getPoly[verts]];
		numPoly = Length[ polys];
		inds = Flatten[ verts];
		numInds = Length[ inds];
		arg = pack[Join[ {1, numFacet, numPoly, numInds}, facetLength, polys, inds]];
		setFacetsFun[ id, arg]
	]

TetGenSetFacetsCheckCoplanarity[ TetGenExpression[ id_]?(testTetGenExpression[TetGenSetFacets]), ptsIn_, vertsIn:{{{__Integer} ..} ..}] :=
	Module[{pts, verts, pos, res, 
		numFacet, facetLength, polys, numPoly, inds, numInds, arg},
		pts = pack @ ptsIn;
		verts = pack @ vertsIn;
		numFacet = Length[verts];
		facetLength = getLength[verts];
		(*find facets that have more than one entry*)
		pos = SparseArray[Unitize[facetLength - 1]]["NonzeroPositions"];
		If[ Length[ pos] > 1,
			(*check coplanarity*)
			res = Union[Region`Mesh`SpannedDimension[
				Flatten[Region`Mesh`ToCoordinates[pts, #], 1]] & /@ Extract[verts, pos]];
			If[{2} =!= res, 
				Message[TetGenSetFacets::"tetnclf", Extract[verts, pos]];
				Return[-1, Module];
			];
		];
		polys = Flatten[ getPoly[verts]];
		numPoly = Length[ polys];
		inds = Flatten[ verts];
		numInds = Length[ inds];
		arg = pack[Join[ {1, numFacet, numPoly, numInds}, facetLength, polys, inds]];
		setFacetsFun[ id, arg]
	]


TetGenSetFacetHoles[TetGenExpression[ id_]?(testTetGenExpression[TetGenSetFacetHoles]), holes:{ {{_Real,_Real,_Real}...}..}] := 
	Module[{pts,lens}, 
		lens = Map[ Length, holes];
		pts = Flatten[ holes, 1];
		If[ Length[ pts] === 0,
			0,
			setFacetHolesFun[ id, pack[pts], pack[lens]]]
	]



TetGenGetFacetHoles[TetGenExpression[ id_]?(testTetGenExpression[TetGenGetFacetHoles])] := 
	Module[{pts,lens}, 
		lens = getFacetHoleLengthsFun[id];
		pts = getFacetHolesFun[id];
		takeLens[ pts, lens, 0]
	]




TetGenGetFacetMarkers[ TetGenExpression[ id_]?(testTetGenExpression[TetGenGetFacetMarkers])] :=
	Module[{},
		getFacetMarkersFun[ id]
	]


TetGenSetFacetMarkers[ TetGenExpression[ id_]?(testTetGenExpression[TetGenSetFacetMarkers]), markers:{__Integer}] :=
	Module[{},
		setFacetMarkersFun[ id, pack[ markers]]
	]



TetGenTetrahedralize::reterr = "Tetrahedralize returned an error, `1`."
TetGenTetrahedralize::iderr = "The tetgen expression, `1`, does not refrence a valid expression."
TetGenTetrahedralize::err = "Tetrahedralize returned an error."

TetGenTetrahedralize[TetGenExpression[ idIn_]?(testTetGenExpression[TetGenTetrahedralize]), 
	args_String, opts:OptionsPattern[TetGenTetrahedralize]] :=
	Module[{out, idOut, res, refine},
		out = TetGenCreate[];
		idOut = getTetGenExpressionID[ out];
		refine = OptionValue["TetrahedronRefinement"];
		If[ Head[refine] === CompiledFunction && 
				TrueQ[ConnectLibraryCallbackFunction["TetGenManager", refine]],
			res = tetrahedralizeFun[ args, idIn, idOut, 1];
		,
			res = tetrahedralizeFun[ args, idIn, idOut, 0];
		];

		Which[
			IntegerQ[ res], Message[ TetGenTetrahedralize::reterr, res]; Null,
			res === True, out,
			res === False, Message[ TetGenTetrahedralize::iderr, idIn]; Null,
			True, Message[ TetGenTetrahedralize::err, idIn]; Null]
	]


TetGenGetElements::order = "TetGenGetElements order `1` must contain all numbers 1 to 10."

TetGenGetElements[TetGenExpression[ id_]?(testTetGenExpression[TetGenGetElements])] :=
	Module[{elems, order},
		order = pack @ Range[10];
		(* C counts from 0 *)
		elems = getElementsFun[ id, order - 1];
		elems
	]

TetGenGetElements[TetGenExpression[ id_]?(testTetGenExpression[TetGenGetElements]),
	orderIn_/; (ArrayQ[orderIn,1,IntegerQ] && Length[orderIn] == 10)] :=
	Module[{elems, order},
		order = pack @ orderIn;
		If[ Union[ order] =!= pack @ {1,2,3,4,5,6,7,8,9,10},
			Message[ TetGenGetElements::order, orderIn]
			,
			(* C counts from 0 *)
			elems = getElementsFun[ id, order - 1];
			elems
		]
	]


TetGenSetElements[TetGenExpression[ id_]?(testTetGenExpression[TetGenSetElements]), elems_ /; ArrayQ[elems,2,IntegerQ]] :=
	Module[{},
		setElementsFun[ id, pack[ elems]]
	]

TetGenGetElementAttributes[TetGenExpression[ id_]?(testTetGenExpression[TetGenGetElementAttributes])] :=
	Module[{attrs},
		attrs = getElementAttrsFun[ id];
		attrs
	]

TetGenSetElementAttributes[TetGenExpression[ id_]?(TetGenSetElementAttributes), attrs_] :=
	Module[{},
		setElementAttrsFun[ id, pack[ N[ attrs]]]
	]


TetGenGetFaces[TetGenExpression[ id_]?(testTetGenExpression[TetGenGetFaces])] :=
	Module[{elems},
		elems = getFacesFun[ id];
		elems
	]

TetGenGetFaceMarkers[ TetGenExpression[ id_]?(testTetGenExpression[TetGenGetFaceMarkers])] :=
	Module[{},
		getFaceMarkersFun[ id]
	]

TetGenSetMessages[ arg:(True|False)] :=
	Module[{},
		setMessagesFun[If[ arg, 1, 0]]
	]

TetGenGetNeighbors[TetGenExpression[ id_]?(testTetGenExpression[TetGenGetNeighbors])] :=
	Module[{elems},
		elems = getNeighborsFun[ id];
		elems
	]

TetGenGetEdges[TetGenExpression[ id_]?(testTetGenExpression[TetGenGetEdges])] :=
	Module[{edges},
		edges = getEdgesFun[ id];
		edges
	]


validTetGenExprQ = TetGenExpressionQ;

ClearAll[ TetGenDelaunay];
TetGenDelaunay[ pts_] /; Last[ Dimensions[ pts]] == 3 :=
	With[{res = Catch[ iTetGenFun[ {TetGenDelaunay, "-QF"}, pts],
						"TetGenFail", $Failed]},
		res /; Head[ Unevaluated[ res]] =!= $Failed
	]

ClearAll[ TetGenConvexHull];
TetGenConvexHull[ pts_] /; Last[ Dimensions[ pts]] == 3 :=
	With[{res = Catch[ iTetGenFun[ {TetGenConvexHull, "-QE"}, pts],
						"TetGenFail", $Failed]},
		res /; Head[ Unevaluated[ res]] =!= $Failed
	]

ClearAll[ iTetGenFun];
iTetGenFun[ {fn_, cmdStr_}, pts_] :=
	Module[
		{error, inInst, outInst, newPts, res},

		inInst = TetGenCreate[];
		If[ !validTetGenExprQ[ inInst],
			Message[ fn::"tetinst", inInst];
			Throw[ $Failed, "TetGenFail"]
			];

		error = TetGenSetPoints[ inInst, pack[ N[ pts]]];
		If[ error =!= 0,
			Message[ fn::"tetpts", pts];
			Throw[ $Failed, "TetGenFail"]
			];

		outInst = TetGenTetrahedralize[ inInst, cmdStr];
		If[ !validTetGenExprQ[ outInst],
			Message[ fn::"tetinst", outInst];
			Throw[ $Failed, "TetGenFail"]
			];

		error = TetGenDelete[ inInst];
		If[ error =!= 0,
			Message[ fn::"tetdin", inInst];
			];

		If[ fn === TetGenDelaunay, res = TetGenGetElements[ outInst]; ];
		If[ fn === TetGenConvexHull, res = TetGenGetFaces[ outInst]; ];

		If[ res === {},
			(* failed computation *)
			Message[ fn::"tetfc", pts];
			Return[ $Failed, Module]
			];

		newPts = TetGenGetPoints[ outInst];
		If[ !MatchQ[ Dimensions[ newPts], {_, 3}],
			Message[ fn::"tetgpts"];
			Throw[ $Failed, "TetGenFail"];
			];

		error = TetGenDelete[ outInst];
		If[ error =!= 0,
			Message[ fn::"tetdin", outInst];
			];

		{newPts, res}
	]


ClearAll[ TetGenIntersectingFacetsQ];
TetGenIntersectingFacetsQ[ pts_, facets:{{{__Integer} ..} ..}] /; 
	Last[ Dimensions[ pts]] == 3 :=
With[{res = Catch[ iTetGenDetectIntersectingFacetsFun[ 
					TetGenIntersectingFacetsQ, pts, facets],
					"TetGenFail", $Failed]},
		If[ Head[ Unevaluated[ res]] =!= $Failed,
			If[ res[[2]] =!= {}, Return[ True], Return[ False]];
		];
		res /; Head[ Unevaluated[ res]] =!= $Failed
	]

ClearAll[ TetGenDetectIntersectingFacets];
TetGenDetectIntersectingFacets[ pts_, facets:{{{__Integer} ..} ..}] /; 
	Last[ Dimensions[ pts]] == 3 :=
With[{res = Catch[ iTetGenDetectIntersectingFacetsFun[
					TetGenDetectIntersectingFacets, pts, facets],
					"TetGenFail", $Failed]},
		res /; Head[ Unevaluated[ res]] =!= $Failed
	]

ClearAll[ iTetGenDetectIntersectingFacetsFun];
iTetGenDetectIntersectingFacetsFun[ fn_, pts_, facets_] :=
	Module[
		{error, inInst, outInst, newPts,
		intersectingSurfaceTriangles},

		inInst = TetGenCreate[];
		If[ !validTetGenExprQ[ inInst],
			Message[ fn::"tetinst", inInst];
			Throw[ $Failed, "TetGenFail"]
			];

		error = TetGenSetPoints[ inInst, pack@pts];
		If[ error =!= 0,
			Message[ fn::"tetpts", pts];
			Throw[ $Failed, "TetGenFail"]
			];

		error = TetGenSetFacetsCheckCoplanarity[ inInst, pts, pack@facets];
		If[ error =!= 0,
			Message[ fn::"tetfcs", facets];
			Throw[ $Failed, "TetGenFail"]
			];

		outInst = TetGenTetrahedralize[ inInst, "-d"];
		If[ !validTetGenExprQ[ outInst],
			Message[ fn::"tetinst", outInst];
			Throw[ $Failed, "TetGenFail"]
			];

		error = TetGenDelete[ inInst];
		If[ error =!= 0,
			Message[ fn::"tetdin", outInst];
			];

		intersectingSurfaceTriangles = TetGenGetFaces[ outInst];
		If[ !MatchQ[ Dimensions[ intersectingSurfaceTriangles], {_, 3}] &&
				intersectingSurfaceTriangles =!= {},
			Message[ fn::"tetgfcs"];
			Throw[ $Failed, "TetGenFail"]
			];

		newPts = {};
		If[ intersectingSurfaceTriangles =!= {}, 
			newPts = TetGenGetPoints[ outInst];
			If[ !MatchQ[ Dimensions[ newPts], {_, 3}],
				Message[ fn::"tetgpts"];
				Throw[ $Failed, "TetGenFail"]
				];
			];

		error = TetGenDelete[ outInst];
		If[ error =!= 0,
			Message[ fn::"tetdin", outInst];
			];

		{newPts, intersectingSurfaceTriangles}
	]


TetGenSetTetrahedraVolumes[ TetGenExpression[ id_]?(testTetGenExpression[TetGenSetTetrahedraVolumes]), pts_] :=
	Module[{},
		setTetrahedraVolumesFun[ id, pack[N[pts]]]
	]



TetGenDelaunay::"tetinst" =
TetGenConvexHull::"tetinst" =
TetGenDetectIntersectingFacets::"tetinst" =
TetGenIntersectingFacetsQ::"tetinst" =
	"A TetGen instance could not be created.";

TetGenDelaunay::"tetdin" =
TetGenConvexHull::"tetdin" =
TetGenDetectIntersectingFacets::"tetdin" =
TetGenIntersectingFacetsQ::"tetdin" =
	"A TetGen instance could not be deleted.";

TetGenDelaunay::"tetpts" =
TetGenConvexHull::"tetpts" =
TetGenDetectIntersectingFacets::"tetpts" =
TetGenIntersectingFacetsQ::"tetpts" =
	"The points `1` could not be set to the TetGen instance.";

TetGenDelaunay::"tetgpts" =
TetGenConvexHull::"tetgpts" =
TetGenDetectIntersectingFacets::"tetgpts" =
TetGenIntersectingFacetsQ::"tetgpts" =
	"The points could not be extracted from the TetGen instance.";

TetGenDelaunay::"tetfcs" =
TetGenConvexHull::"tetfcs" =
TetGenDetectIntersectingFacets::"tetfcs" =
TetGenIntersectingFacetsQ::"tetfcs" =
	"The facets `1` could not be set to the TetGen instance.";

TetGenDelaunay::"tetgfcs" =
TetGenConvexHull::"tetgfcs" =
TetGenDetectIntersectingFacets::"tetgfcs" =
TetGenIntersectingFacetsQ::"tetgfcs" =
	"The faces could not be extracted from the TetGen instance.";

TetGenDelaunay::"tetfc" = "A Delaunay tetrahedralization could not be found from the points `1`."
TetGenConvexHull::"tetfc" = "A convex hull could not be found from the points `1`.";

TetGenSetFacets::"tetnclf" = "Some of the facets `1` are not coplanar."


End[]

EndPackage[]


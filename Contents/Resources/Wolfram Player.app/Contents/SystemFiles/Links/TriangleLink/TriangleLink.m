(* Mathematica Package *)

BeginPackage["TriangleLink`"]


$TriangleLibrary::usage = "$TriangleLibrary is the full path to the Triangle Library loaded by TriangleLink."

TriangleExpression::usage = "TriangleExpression[ id] represents an instance of a Triangle expression."

TriangleExpressionQ::usage = "TriangleExpressionQ[ expr] returns True if expr represents an active instance of a Triangle object."

TriangleExpressions::usage = "TriangleExpressions[] returns a list of active Triangle expressions."

LoadTriangle::usage = "LoadTriangle[] loads the Triangle Library."

TriangleCreate::usage = "TriangleCreate[] creates an instance of a Triangle expression."

TriangleDelete::usage = "TriangleDelete[ expr] removes an instance of a Triangle expression, freeing up memory."

TriangleGetVertices::usage = "TriangleGetVertices[ expr] returns the vertices for a Triangle expression."

TriangleSetVertices::usage = "TriangleSetVertices[ expr, vertices] sets the vertices for a Triangle expression.  "

TriangleGetSegments::usage = "TriangleGetSegments[ expr] returns the segments for a Triangle expression."

TriangleSetSegments::usage = "TriangleSetSegments[ expr, segs] sets the segments for a Triangle expression.  "

TriangleTriangulate::usage = "TriangleTriangulate[ expr, args] Triangulates a Triangle expression using args, it returns the result in a new Triangle expression. "

TriangleGetElements::usage = "TriangleGetElements[expr] gets the elements in a Triangle expression."

TriangleSetElements::usage = "TriangleSetElements[expr, elements] sets the elements in a Triangle expression."

(*
TriangleSetMessages::usage = "TriangleSetMessages[True|False] enbables or disables the issuing of messages from Triangle."

TriangleLoadFile::usage = "TriangleLoadFile[ expr, file, type] loads file into a Triangle expression. Type can be any of \"node\", \"poly\", \"pbc\", \"var\", \"mtr\", \"off\", \"ply\", \"stl\",  \"medit\",  \"tetmesh\", or \"voronoi\"."

TriangleSaveFile::usage = "TriangleSaveFile[ expr, file, type] saves a Triangle expression into a file. Type can be any of \"node\", \"poly\", \"neighbor\", \"face\", \"element\", or \"edge\"."
*)


TriangleGetPoints::usage = "TriangleGetPoints[ expr] returns the points in a Triangle expression."

TriangleSetPoints::usage = "TriangleSetPoints[ expr, points] sets the points in a Triangle expression."

TriangleGetPointMarkers::usage = "TriangleGetPointMarkers[ expr] returns the point markers in a Triangle expression."

TriangleSetPointMarkers::usage = "TriangleSetPointMarkers[ expr, markers] sets the point markers in a Triangle expression."

TriangleSetTriangleAreas::usage = "TriangleSetTriangleAreas[ expr, areas] sets the area for triangles in a Triangle expression."

TriangleGetPointAttributes::usage = "TriangleGetPointAttributes[ expr] returns the point attributes in a Triangle expression."

TriangleSetPointAttributes::usage = "TriangleSetPointAttributes[ expr, attrs] sets the point attributes in a Triangle expression."

TriangleGetElementAttributes::usage = "TriangleGetElementAttributes[expr] gets the element attributes in a Triangle expression."

TriangleSetElementAttributes::usage = "TriangleSetElementAttributes[expr, attrs] sets the element attributes in a Triangle expression."

TriangleGetSegmentMarkers::usage = "TriangleGetSegmentMarkers[ expr] returns the facet markers for a Triangle expression."

TriangleSetSegmentMarkers::usage = "TriangleSetSegmentMarkers[ expr, vertices] sets the facet markers for a Triangle expression.  "

TriangleGetHoles::usage = "TriangleGetHoles[ expr] returns the holes in a Triangle expression."

TriangleSetHoles::usage = "TriangleSetHoles[ expr, points] sets the holes in a Triangle expression."

TriangleGetRegions::usage = "TriangleGetRegions[ expr] returns the regions in a Triangle expression."

TriangleSetRegions::usage = "TriangleSetRegions[ expr, pts, index, attrs] sets the regions in a Triangle expression."

TriangleGetNeighbors::usage = "TriangleGetNeighbors[expr] gets the neighbors in a Triangle expression."

TriangleDelaunay::usage = "TriangleDelaunay[ points] generates a Delaunay triangulation for a 2D point set."

TriangleConvexHull::usage = "TriangleConvexHull[ points] generates a convex hull for a 2D point set."

$TriangleInstallationDirectory::usage = "$TriangleInstallationDirectory gives the top-level directory in which your Triangle installation resides."


(*

TriangleGetPointMetricTensors::usage = "TriangleGetPointMetricTensors[ expr] returns the point metric tensors in a Triangle expression."

TriangleSetPointMetricTensors::usage = "TriangleSetPointMetricTensors[ expr, tens] sets the point metric tensors in a Triangle expression."


Graphics3DSplit::usage = "Graphics3DSplit  "

MakeElements::usage = "MakeElements  "
*)

Options[TriangleTriangulate] = {"TriangleRefinement" -> None};

Begin["`Private`"]
(* Implementation of the package *)

$TriangleLibrary = FindLibrary[ "triangleWolfram"]

pack = Developer`ToPackedArray;

needInitialization = True;

$TriangleInstallationDirectory = DirectoryName[ $InputFileName]



LoadTriangle[] :=
	Module[{},
		deleteFun = LibraryFunctionLoad[$TriangleLibrary, "deleteTriangleInstance", {Integer}, Integer];
		instanceListFun	= LibraryFunctionLoad[$TriangleLibrary, "instanceListQuery", {}, {Integer,1}];

		getPointsFun = LibraryFunctionLoad[$TriangleLibrary, "getPointList", {Integer}, {Real,2}];
		setPointsFun = LibraryFunctionLoad[$TriangleLibrary, "setPointList", {Integer, {Real, 2, "Shared"}}, Integer];

		getSegmentsFun = LibraryFunctionLoad[$TriangleLibrary, "getSegments", {Integer}, {Integer,2}];
		setSegmentsFun = LibraryFunctionLoad[$TriangleLibrary, "setSegments", {Integer, {Integer, 2, "Shared"}}, Integer];
		getSegmentMarkersFun = LibraryFunctionLoad[$TriangleLibrary, "getSegmentMarkers", {Integer}, {Integer,1}];
		setSegmentMarkersFun = LibraryFunctionLoad[$TriangleLibrary, "setSegmentMarkers", {Integer, {Integer, 1, "Shared"}}, Integer];

		triangulateFun = LibraryFunctionLoad[$TriangleLibrary, "mTriangulate", LinkObject, LinkObject];
		getElementsFun = LibraryFunctionLoad[$TriangleLibrary, "getElements", {Integer, {Integer, 1, "Shared"}}, {Integer, 2}];
		setElementsFun = LibraryFunctionLoad[$TriangleLibrary, "setElements", {Integer, {Integer, 2, "Shared"}}, Integer];
		getElementAttrsFun = LibraryFunctionLoad[$TriangleLibrary, "getElementAttributes", {Integer}, {Real, 2}];
		setElementAttrsFun = LibraryFunctionLoad[$TriangleLibrary, "setElementAttributes", {Integer, {Real, 2, "Shared"}}, Integer];
		setTriangleAreasFun = LibraryFunctionLoad[$TriangleLibrary, "setTriangleAreas", {Integer, {Real, 1, "Shared"}}, Integer];

		getPointMarkersFun = LibraryFunctionLoad[$TriangleLibrary, "getPointMarkers", {Integer}, {Integer,1}];
		setPointMarkersFun = LibraryFunctionLoad[$TriangleLibrary, "setPointMarkers", {Integer, {Integer, 1, "Shared"}}, Integer];
		getPointAttributesFun = LibraryFunctionLoad[$TriangleLibrary, "getPointAttributes", {Integer}, {Real,2}];
		setPointAttributesFun = LibraryFunctionLoad[$TriangleLibrary, "setPointAttributes", {Integer, {Real, 2, "Shared"}}, Integer];

		getHolesOrRegionsFun = LibraryFunctionLoad[$TriangleLibrary, "getHoles_or_Regions", {Integer, Integer}, {Real,_}];
		setHolesOrRegionsFun = LibraryFunctionLoad[$TriangleLibrary, "setHoles_or_Regions", {Integer, {Real, 2, "Shared"}, Integer}, Integer];

		getNeighborsFun = LibraryFunctionLoad[$TriangleLibrary, "getNeighbors", {Integer}, {Integer,2}];

		triangleUnsuitableCallback = LibraryFunctionLoad[$TriangleLibrary, "triangleUnsuitableCallback", {{Real, 2}, {Real, 0}}, {True|False, 0}];

(*
		setMessagesFun = LibraryFunctionLoad[$TriangleLibrary, "setMessages", {Integer}, Integer];
		fileOperationFun = LibraryFunctionLoad[$TriangleLibrary, "fileOperation", LinkObject, LinkObject];
*)
		needInitialization = False;
	]

(*
 Functions for working with TriangleExpression
*)
getTriangleExpressionID[e_TriangleExpression] := ManagedLibraryExpressionID[e, "TriangleManager"];

TriangleExpressionQ[e_TriangleExpression] := ManagedLibraryExpressionQ[e, "TriangleManager"];

testTriangleExpression[][e_] := testTriangleExpression[TriangleExpression][TriangleExpression][e];

testTriangleExpression[mhead_Symbol][e_] := 
	If[
		TrueQ[TriangleExpressionQ[e]], 
		True,
		Message[MessageName[mhead,"tginst"], e]; False
	];

testTriangleExpression[_][e_] := TrueQ[TriangleExpressionQ[e]];

General::tginst = "`1` does not represent an active Triangle object.";


TriangleCreate[] :=
	Module[{},
		If[ needInitialization, LoadTriangle[]];
		CreateManagedLibraryExpression["TriangleManager", TriangleExpression]
	]

TriangleDelete[ TriangleExpression[ id_]?(testTriangleExpression[TriangleDelete])] :=
	Module[{},
		deleteFun[ id]
	]

TriangleDelete[ l:{_TriangleExpression..}] := TriangleDelete /@ l


TriangleExpressions[] :=
	Module[ {list},
		If[ needInitialization, LoadTriangle[]];
		list = instanceListFun[];
		If[ !ListQ[ list], 
			$Failed,
			Map[ TriangleExpression, list]]
	]



(*
TriangleLoadFile::file = "An error was found loading `1` file, `2`."

TriangleLoadFile[ TriangleExpression[ id_]?(testTriangleExpression[TriangleLoad]), file_String, form_String] :=
	Module[{res, formName},
		formName = Switch[ form,
						"node", "load_node",
						"poly", "load_poly",
						"pbc", "load_pbc",
						"var", "load_var",
						"mtr", "load_mtr",
						"off", "load_off",
						"ply", "load_ply",
						"stl", "load_stl",
						"medit", "load_medit",
						"tetmesh", "load_tetmesh",
						"voronoi", "load_voronoi",
						_, $Failed];
		If[ formName === $Failed, Return[ $Failed]];
		res = fileOperationFun[ id, file, formName];
		If[ res =!= True, Message[ TriangleLoadFile::file, form, file]; $Failed, Null]
	]


TriangleSaveFile[ TriangleExpression[ id_]?(testTriangleExpression[TriangleSave]), file_String, form_String] :=
	Module[{res, formName},
		formName = Switch[ form,
						"node", "save_nodes",
						"poly", "save_poly",
						"neighbor", "save_neighbors",
						"segment", "save_segments",
						"element", "save_elements",
						"edge", "save_edges",
						_, $Failed];
		If[ formName === $Failed, Return[ $Failed]];
		res = fileOperationFun[ id, file, formName];
		If[ res =!= True, Message[ TriangleLoadFile::file, form, file]; $Failed, Null]
	]
*)

TriangleGetPoints[ TriangleExpression[ id_]?(testTriangleExpression[TriangleGetPoints])] :=
	Module[{},
		getPointsFun[ id]
	]

TriangleSetPoints[ TriangleExpression[ id_]?(testTriangleExpression[TriangleSetPoints]), pts_] :=
	Module[{},
		setPointsFun[ id, pack[ N[ pts]]]
	]

TriangleGetPointAttributes[ TriangleExpression[ id_]?(testTriangleExpression[TriangleGetPointAttributes])] :=
	Module[{},
		getPointAttributesFun[ id]
	]

TriangleSetPointAttributes[ TriangleExpression[ id_]?(testTriangleExpression[TriangleSetPointAttributes]), pts_] :=
	Module[{},
		setPointAttributesFun[ id, pack[ pts]]
	]

TriangleGetPointMarkers[ TriangleExpression[ id_]?(testTriangleExpression[TriangleGetPointMarkers])] :=
	Module[{},
		getPointMarkersFun[ id]
	]

TriangleSetPointMarkers[ TriangleExpression[ id_]?(testTriangleExpression[TriangleSetPointMarkers]), pts_] :=
	Module[{},
		setPointMarkersFun[ id, pack[pts]]
	]

TriangleSetTriangleAreas[ TriangleExpression[ id_]?(testTriangleExpression[TriangleSetTriangleAreas]), pts_] :=
	Module[{},
		setTriangleAreasFun[ id, pack[N[pts]]]
	]

TriangleGetHoles[ TriangleExpression[ id_]?(testTriangleExpression[TriangleGetHoles])] :=
	Module[{},
		getHolesOrRegionsFun[ id, 1]
	]

TriangleSetHoles[ TriangleExpression[ id_]?(testTriangleExpression[TriangleSetHoles]), pts_ /; ArrayQ[pts,2]] :=
	Module[{},
		setHolesOrRegionsFun[ id, pack[pts], 1]
	]


TriangleGetRegions[ TriangleExpression[ id_]?(testTriangleExpression[TriangleGetRegions])] :=
	Module[{val},
		val = getHolesOrRegionsFun[ id, 0];
		If[ !MatchQ[ val, {{_,_,_,_}..}], Return[ val]];
		{ Part[ val, All, {1,2}], Part[ val, All, 3], Part[ val, All, 4]}
	]


TriangleSetRegions::indlen = "The number of region indices, `1`, does not match the number of points, `2`."
TriangleSetRegions::attrlen = "The number of region attributes, `1`, does not match the number of points, `2`."


TriangleSetRegions[ TriangleExpression[ id_]?(testTriangleExpression[TriangleSetRegion]), regPts: {{_,_}..}, regInd_ /; ArrayQ[regInd,1], regAttrs_/; ArrayQ[regAttrs,1]] :=
	Module[{len, arg},
		len = Length[ regPts];
		If[ Length[ regInd] =!= len, 
				Message[TriangleSetRegions::indlen, len, Length[ regInd]]; 
				Return[ $Failed]];
		If[ Length[ regAttrs] =!= len, 
				Message[TriangleSetRegions::attrlen, len, Length[ regAttrs]]; 
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
 
 The result of the getSegmentsFun is a list
 
 {pointBase, numFacet, numPoly, numVertex, f1Len, f2Len, ..., p1Len, p2Len, ...   p1ind1, p1ind2, ...}
 
 TODO apply the pointBase to the points in the poly list.
*)

TriangleGetVertices[ TriangleExpression[ id_]?(testTriangleExpression[TriangleGetVertices])] :=
	TriangleGetSegments[ TriangleExpression[ id]]

TriangleGetSegments[ TriangleExpression[ id_]?(testTriangleExpression[TriangleGetSegments])] :=
	Module[{},
		getSegmentsFun[ id]
	]

(*
takeLens[ list_, lenList_, start_] :=
	Module[ {inds, t},
		inds = FoldList[Plus, start, lenList];
		t = Partition[inds, 2, 1];
		inds = Map[# + {1, 0} &, Partition[inds, 2, 1]];
		Map[ Take[ list, #]&, inds]
	]
*)

TriangleSetVertices[TriangleExpression[ id_]?(testTriangleExpression[TriangleSetVertices]), verts:{{_Integer, _Integer} ..}] :=
	TriangleSetSegments[ TriangleExpression[ id], pack[ verts]]

TriangleSetSegments[ TriangleExpression[ id_]?(testTriangleExpression[TriangleSetSegments]), verts:{{_Integer, _Integer} ..}] :=
	Module[{},
		setSegmentsFun[ id, pack[ verts]]
	]

TriangleGetSegmentMarkers[ TriangleExpression[ id_]?(testTriangleExpression[TriangleGetSegmentMarkers])] :=
	Module[{},
		getSegmentMarkersFun[ id]
	]


TriangleSetSegmentMarkers[ TriangleExpression[ id_]?(testTriangleExpression[TriangleSetSegmentMarkers]), markers:{__Integer}] :=
	Module[{},
		setSegmentMarkersFun[ id, pack[ markers]]
	]


TriangleTriangulate::reterr = "Triangulate returned an error, `1`."
TriangleTriangulate::iderr = "The Triangle expression, `1`, does not refrence a valid expression."
TriangleTriangulate::err = "Triangulate returned an error."


TriangleTriangulate[TriangleExpression[ idIn_]?(testTriangleExpression[TriangleTriangulate])
	, argsIn_String, opts:OptionsPattern[TriangleTriangulate]] :=
	Module[{out, idOut, res, refine, args},
		out = TriangleCreate[];
		idOut = getTriangleExpressionID[ out];
		refine = OptionValue["TriangleRefinement"];
		args = argsIn;
		If[ Head[refine] === CompiledFunction && 
				TrueQ[ConnectLibraryCallbackFunction["TriangleManager", refine]],
			args = StringJoin[argsIn, "u"];
		];
		res = triangulateFun[ args, idIn, idOut];
		Which[
			res === True, out,
			IntegerQ[ res], Message[ TriangleTriangulate::reterr, res]; Null,
			res === False, Message[ TriangleTriangulate::iderr, idIn]; Null,
			True, Message[ TriangleTriangulate::err, idIn]; Null]
	]


TriangleGetElements::order = "TriangleGetElements order `1` must contain all numbers 1 to 6."

TriangleGetElements[TriangleExpression[ id_]?(testTriangleExpression[TriangleGetElements])] :=
	Module[{elems, order},
		order = pack @ Range[6];
		(* C counts from 0 *)
		elems = getElementsFun[ id, order - 1];
		elems
	]

TriangleGetElements[TriangleExpression[ id_]?(testTriangleExpression[TriangleGetElements]), 
	orderIn_/; (ArrayQ[orderIn,1,IntegerQ] && Length[orderIn] == 6) ] :=
	Module[{elems, order},
		order = pack @ orderIn;
		If[ Union[ order] =!= pack @ {1,2,3,4,5,6},
			Message[ TriangleGetElements::order, orderIn]
			,
			(* C counts from 0 *)
			elems = getElementsFun[ id, order - 1];
			elems
		]
	]

TriangleSetElements[TriangleExpression[ id_]?(testTriangleExpression[TriangleSetElements]), elems_ /; ArrayQ[elems,2,IntegerQ]] :=
	Module[{},
		setElementsFun[ id, pack[ elems]]
	]

TriangleGetElementAttributes[TriangleExpression[ id_]?(testTriangleExpression[TriangleGetElementAttributes])] :=
	Module[{attrs},
		attrs = getElementAttrsFun[ id];
		attrs
	]

TriangleSetElementAttributes[TriangleExpression[ id_]?(testTriangleExpression[TriangleSetElementAttributes]), attrs_] :=
	Module[{},
		setElementAttrsFun[ id, pack[ N[attrs]]]
	]

(*
TriangleSetMessages[ arg:(True|False)] :=
	Module[{},
		setMessagesFun[If[ arg, 1, 0]]
	]

*)

TriangleGetNeighbors[TriangleExpression[ id_]?(testTriangleExpression[TriangleGetNeighbors])] :=
	Module[{elems},
		elems = getNeighborsFun[ id];
		elems
	]

validTriangleExprQ = TriangleExpressionQ;

ClearAll[ TriangleDelaunay];
TriangleDelaunay[ pts_] /; (Last[ Dimensions[ pts]] == 2) && (Length[ pts] > 2) :=
	With[{res = Catch[ iTriangleFun[ {TriangleDelaunay, "-Q "}, pts],
						"TriangleFail", $Failed]},
		res /; Head[ Unevaluated[ res]] =!= $Failed
	]

ClearAll[ TriangleConvexHull];
TriangleConvexHull[ pts_] /; (Last[ Dimensions[ pts]] == 2) && (Length[ pts] > 2) :=
	With[{res = Catch[ iTriangleFun[ {TriangleConvexHull, "-Qc"}, pts],
						"TriangleFail", $Failed]},
		res /; Head[ Unevaluated[ res]] =!= $Failed
	]

ClearAll[ iTriangleFun];
iTriangleFun[ {fn_, cmdStr_}, pts_] :=
	Module[
		{error, inInst, outInst, newPts, res},

		inInst = TriangleCreate[];
		If[ !validTriangleExprQ[ inInst],
			Message[ fn::"triinst", inInst];
			Throw[ $Failed, "TriangleFail"]
			];

		error = TriangleSetPoints[ inInst, pack[ N[ pts]]];
		If[ error =!= 0,
			Message[ fn::"tripts", pts];
			Throw[ $Failed, "TriangleFail"]
			];

		outInst = TriangleTriangulate[ inInst, cmdStr];
		If[ !validTriangleExprQ[ outInst],
			Message[ fn::"triinst", outInst];
			Throw[ $Failed, "TriangleFail"]
			];

		error = TriangleDelete[ inInst];
		If[ error =!= 0,
			Message[ fn::"tridin", inInst];
			];

		If[ fn === TriangleDelaunay, res = TriangleGetElements[ outInst]; ];
		If[ fn === TriangleConvexHull, res = TriangleGetSegments[ outInst]; ];

		If[ res === {},
			(* failed computation *)
			Message[ fn::"trifc", pts];
			Return[ $Failed, Module]
			];

		newPts = TriangleGetPoints[ outInst];
		If[ !MatchQ[ Dimensions[ newPts], {_, 2}],
			Message[ fn::"trigpts"];
			Throw[ $Failed, "TriangleFail"];
			];

		error = TriangleDelete[ outInst];
		If[ error =!= 0,
			Message[ fn::"tridin", outInst];
			];

		{newPts, res}
	]

TriangleDelaunay::"triinst" =
TriangleConvexHull::"triinst" =
	"A Tetrinagle instance could not be created.";

TriangleDelaunay::"tridin" =
TriangleConvexHull::"tridin" =
	"A Tetrinagle instance could not be deleted.";

TriangleDelaunay::"tripts" =
TriangleConvexHull::"tripts" =
	"The points `1` could not be set to the Tetrinagle instance.";

TriangleDelaunay::"trigpts" =
TriangleConvexHull::"trigpts" =
	"The points could not be extracted from the Triangle instance.";

TriangleDelaunay::"trifc" = "A Delaunay triangulation could not be found from the points `1`."
TriangleConvexHull::"trifc" = "A convex hull could not be found from the points `1`.";


End[]

EndPackage[]


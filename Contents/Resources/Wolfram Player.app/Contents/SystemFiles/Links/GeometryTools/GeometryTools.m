(* ::Package:: *)

(* :Name: GeometryTools.m *)

(* :Title: Wolfram Geometry Library *)

(* :Context: GeometryTols` *)

(* :Author: Charles Pooh *)

(* :Summary: This file contains interface functions to Wolfram Geometry Library *)

(* :Sources: *)

(* :Copyright: 2014 Wolfram Research, Inc. *)

(* :Package Version: 2.0 *)

(* :Mathematica Version: 11.0 *)

(* :History: Last updated on October 6, 2014 *)

(* :Keywords: None *)

(* :Warnings: *)

(* :Limitations: None *)

(* :Discussion: *)

(* :Requirements: None *)

(* :Examples: None *)


(*****************************************************************************)


BeginPackage["GeometryTools`"]

 Unprotect[

    (* 2D boolean operations *)

   WGLGetInstanceList,
   WGLCreateInstance,
   WGLCopyInstance,
   WGLDeleteInstance,
   WGLValidInstance,

   WGLSetCoordinates,
   WGLSetCells,

   WGLGetCoordinates,
   WGLGetCells,

   WGL2DPolygonIntersection,
   WGL2DPolygonUnion,
   WGL2DPolygonDifference,
   WGL2DPolygonSymmetricDifference

]

Unprotect[

	(* Import/Export *)

    WGTAiScene,
    WGTAiSceneInstanceList,
    WGTAssimpImport,
    WGTAssimpVertices,
    WGTAssimpVertexNormals,
    WGTAssimpVertexTextureCoordinates,
    WGTAssimpVertexColors,
    WGTAssimpMeshCellCount,
    WGTAssimpPolygonCells,
    WGTAssimpTriangleCells,
    WGTAssimpLineCells,
    WGTAssimpExport,
    WGTAssimpExportSTL
]

Unprotect[

	TESSCreateInstance,
	TESSDeleteInstance,
	TESSValidInstance,
	TESSGetInstanceList,
	TESSAddContour,
	TESSTesselate,
	TESSGetCoordinates,
	TESSGetCoordinateMap,
	TESSGetElementCount,
	TESSGetPrimitiveIndices,
	TESSGetCellIndices,
	TESSGetBoundaryIndices

]


Begin["`GeometryToolsDump`"]


(* find library *)

$LibraryResourcesPath =
	FileNameJoin[{
		DirectoryName[System`Private`$InputFileName],
		"LibraryResources",
		$SystemID
	}]

$WGTLibrary = Block[
	{$LibraryPath = $LibraryPath},
	PrependTo[$LibraryPath, $LibraryResourcesPath];
	FindLibrary["libGeometryTools"]
]

 $WGLLibrary = $WGTLibrary;

(* load function *)

If[FileExistsQ[$WGTLibrary],

	(* load library function  *)

Quiet[

      WGLCreateInstance = LibraryFunctionLoad[$WGTLibrary, "WGL_CreateInstance", {Integer}, Integer];
      WGLCopyInstance = LibraryFunctionLoad[$WGTLibrary, "WGL_CopyInstance", {Integer}, Integer];
      WGLDeleteInstance = LibraryFunctionLoad[$WGTLibrary, "WGL_DeleteInstance", {Integer}, Integer];
      WGLValidInstance = LibraryFunctionLoad[$WGTLibrary, "WGL_ValidInstance", {Integer}, "Boolean"];
      WGLGetInstanceList = LibraryFunctionLoad[$WGTLibrary, "WGL_GetInstanceList", {}, {Integer, 1}];

      WGLSetCoordinates = LibraryFunctionLoad[$WGTLibrary, "WGL_SetCoordinates", {Integer, {Real, _, "Shared"}}, Integer];
      WGLSetCells = LibraryFunctionLoad[$WGTLibrary, "WGL_SetCells", {Integer, Integer, {Integer, _, "Shared"}}, Integer];

      WGLGetCoordinates = LibraryFunctionLoad[$WGTLibrary, "WGL_GetCoordinates", {Integer}, {Real, _, "Shared"}];
      WGLGetCells = LibraryFunctionLoad[$WGTLibrary, "WGL_GetCells", {Integer, Integer}, {Integer, _, "Shared"}];

      WGL2DPolygonUnion = LibraryFunctionLoad[$WGTLibrary, "WGL_2DPolygon_Union", {{Integer, 1}}, Integer];
      WGL2DPolygonIntersection = LibraryFunctionLoad[$WGTLibrary, "WGL_2DPolygon_Intersection", {{Integer, 1}}, Integer];
      WGL2DPolygonDifference = LibraryFunctionLoad[$WGTLibrary, "WGL_2DPolygon_Difference", {{Integer, 1}}, Integer];
      WGL2DPolygonSymmetricDifference = LibraryFunctionLoad[$WGTLibrary, "WGL_2DPolygon_SymmetricDifference", {{Integer, 1}}, Integer];

    (* Import/Export modules *)

    (* iWGTAiSceneInstanceList = LibraryFunctionLoad[$WGTLibrary, "WGT_AiSceneInstanceList", {}, {Integer, 1}]; *)
    iWGTAssimpImport = LibraryFunctionLoad[$WGTLibrary, "WGT_AssimpImport", {Integer, "UTF8String", "UTF8String"}, Integer];
    iWGTAssimpVertices = LibraryFunctionLoad[$WGTLibrary, "WGT_AssimpVertices", {Integer}, {Real, _}];
    iWGTAssimpVertexNormals = LibraryFunctionLoad[$WGTLibrary, "WGT_AssimpVertexNormals", {Integer}, {Real, _}];
    iWGTAssimpVertexTextureCoordinates = LibraryFunctionLoad[$WGTLibrary, "WGT_AssimpVertexTextureCoordinates", {Integer}, {Real, _}];
    iWGTAssimpVertexColors = LibraryFunctionLoad[$WGTLibrary, "WGT_AssimpVertexColors", {Integer}, {Real, _}];
    iWGTAssimpMeshCellCount = LibraryFunctionLoad[$WGTLibrary, "WGT_AssimpMeshCellCount", {Integer, Integer}, Integer];

    iWGTAssimpPolygonCells = LibraryFunctionLoad[$WGTLibrary, "WGT_AssimpPolygonCells", {Integer}, {LibraryDataType[SparseArray]}];
    iWGTAssimpTriangleCells = LibraryFunctionLoad[$WGTLibrary, "WGT_AssimpTriangleCells", {Integer}, {Integer, _}];
    iWGTAssimpLineCells = LibraryFunctionLoad[$WGTLibrary, "WGT_AssimpLineCells", {Integer}, {Integer, _}];

    iWGTAssimpExport = LibraryFunctionLoad[$WGTLibrary, "WGT_AssimpExport",
    				    {"UTF8String", {Real, _}, {LibraryDataType[SparseArray]}, "UTF8String", Integer, "UTF8String"}, Integer];

    iWGTAssimpExportSTL = LibraryFunctionLoad[$WGTLibrary, "WGT_AssimpExportSTL",
    				    {"UTF8String", {Real, _}, {Integer, _}, "UTF8String", Integer, "UTF8String"}, Integer];


    (* Libtess2 modules *)

	TESSCreateInstance  = LibraryFunctionLoad[$WGTLibrary, "TESS_CreateInstance", {}, Integer];
	TESSDeleteInstance  = LibraryFunctionLoad[$WGTLibrary, "TESS_DeleteInstance", {Integer}, Integer];
	TESSValidInstance   = LibraryFunctionLoad[$WGTLibrary, "TESS_ValidInstance", {Integer}, "Boolean"];
	TESSGetInstanceList = LibraryFunctionLoad[$WGTLibrary, "TESS_GetInstanceList", {}, {Integer,1}];

	TESSAddContour = LibraryFunctionLoad[$WGTLibrary, "TESS_AddContour", {Integer,{Real,_,"Shared"}}, Integer];
	TESSTesselate  = LibraryFunctionLoad[$WGTLibrary, "TESS_Tesselate", {Integer, Integer, Integer}, Integer];

	TESSGetCoordinates      = LibraryFunctionLoad[$WGTLibrary, "TESS_GetCoordinates", {Integer}, {Real,2}];
	TESSGetCoordinateMap    = LibraryFunctionLoad[$WGTLibrary, "TESS_GetCoordinateMap", {Integer}, {Integer,1}];
	TESSGetElementCount     = LibraryFunctionLoad[$WGTLibrary, "TESS_GetElementCount", {Integer}, Integer];
	TESSGetPrimitiveIndices = LibraryFunctionLoad[$WGTLibrary, "TESS_GetPrimitiveIndices", {Integer}, {Integer,2}];
	TESSGetCellIndices      = LibraryFunctionLoad[$WGTLibrary, "TESS_GetCellIndices", {Integer, Integer}, {Integer,1}];
	TESSGetBoundaryIndices  = LibraryFunctionLoad[$WGTLibrary, "TESS_GetBoundaryIndices", {Integer, Integer} ,{Integer,2}];

	, LibraryFunction::overload];

]


(* ************************************************************************* **

                          WGT AiScene Import

   Comments:

   ToDo:

** ************************************************************************* *)


(* :WGTAiSceneInstanceList: *)

WGTAiSceneInstanceList[] :=
    Block[{res},
        res = iWGTAiSceneInstanceList[];
        res /; Developer`PackedArrayQ[res, Integer]
    ]

WGTAiSceneInstanceList[___] := $Failed


(* :WGTAssimpImport: *)

WGTAssimpImport[filename_?StringQ, type_?StringQ] :=
    Block[{res, id},
    	res = CreateManagedLibraryExpression["WGTAiScene", WGTAiScene];
    	(
    	  id = iWGTAssimpImport[ManagedLibraryExpressionID[res], filename, type];
          res /; (id === ManagedLibraryExpressionID[res])

        ) /; ManagedLibraryExpressionQ[res]
    ]

WGTAssimpImport[___] := $Failed


(* :WGTAssimpVertices: *)

WGTAssimpVertices[expr_WGTAiScene?ManagedLibraryExpressionQ] :=
    Block[{res},
    	res = iWGTAssimpVertices[ManagedLibraryExpressionID[expr]];
    	res /; Developer`PackedArrayQ[res, Real]
    ]

WGTAssimpVertices[___] := $Failed


(* :WGTAssimpVertexNormals: *)

WGTAssimpVertexNormals[expr_WGTAiScene?ManagedLibraryExpressionQ] :=
    Block[{res},
        res = iWGTAssimpVertexNormals[ManagedLibraryExpressionID[expr]];
        res /; Developer`PackedArrayQ[res, Real]
    ]

WGTAssimpVertexNormals[___] := $Failed


(* :WGTAssimpVertexTextureCoordinates: *)

WGTAssimpVertexTextureCoordinates[expr_WGTAiScene?ManagedLibraryExpressionQ] :=
    Block[{res},
        res = iWGTAssimpVertexTextureCoordinates[ManagedLibraryExpressionID[expr]];
        res /; Developer`PackedArrayQ[res, Real]
    ]

WGTAssimpVertexTextureCoordinates[___] := $Failed


(* :WGTAssimpVertexColors: *)

WGTAssimpVertexColors[expr_WGTAiScene?ManagedLibraryExpressionQ] :=
    Block[{res},
        res = iWGTAssimpVertexColors[ManagedLibraryExpressionID[expr]];
        res /; Developer`PackedArrayQ[res, Real]
    ]

WGTAssimpVertexColors[___] := $Failed


(* :WGTAssimpMeshCellCount: *)

WGTAssimpMeshCellCount[expr_WGTAiScene?ManagedLibraryExpressionQ, d_Integer] :=
    Block[{res},
        res = iWGTAssimpMeshCellCount[ManagedLibraryExpressionID[expr], d];
        res /; IntegerQ[res]
    ]

WGTAssimpMeshCellCount[___] := $Failed


(* :WGTAssimpPolygonCells: *)

WGTAssimpPolygonCells[expr_WGTAiScene?ManagedLibraryExpressionQ] :=
    Block[{spar, res},
    	spar = iWGTAssimpPolygonCells[ManagedLibraryExpressionID[expr]];
        (
        	res = spar["AdjacencyLists"];
        	res /; ListQ[res]

        ) /; (spar =!= $Failed)
    ]

WGTAssimpPolygonCells[___] := $Failed


(* :WGTAssimpTriangleCells: *)

WGTAssimpTriangleCells[expr_WGTAiScene?ManagedLibraryExpressionQ] :=
    Block[{res},
        res = iWGTAssimpTriangleCells[ManagedLibraryExpressionID[expr]];
        res  /; Developer`PackedArrayQ[res, Integer]
    ]

WGTAssimpTriangleCells[___] := $Failed


(* :WGTAssimpLineCells: *)

WGTAssimpLineCells[expr_WGTAiScene?ManagedLibraryExpressionQ] :=
    Block[{res},
        res = iWGTAssimpLineCells[ManagedLibraryExpressionID[expr]];
        res /; Developer`PackedArrayQ[res, Integer]
    ]

WGTAssimpLineCells[___] := $Failed


(* ************************************************************************* **
                          WGT AiScene Export

   Comments:

   ToDo:
** ************************************************************************* *)

(* :WGTAssimpExport: *)

WGTAssimpExport[filename_?StringQ, coords_List, faces_, type_?StringQ, form_Integer, name_?StringQ] :=
    Block[{res, id},

    	  res = iWGTAssimpExport[ filename, coords, faces, type, form, name];
          res /; (res === 1)


    ]

WGTAssimpExport[___] := $Failed


(* :WGTAssimpExportSTL: *)

WGTAssimpExportSTL[filename_?StringQ, coords_List, faces_List, type_?StringQ, form_Integer, name_?StringQ] :=
    Block[{res, id},

    	  res = iWGTAssimpExportSTL[ filename, coords, faces, type, form, name];
          res /; (res === 1)


    ]

WGTAssimpExportSTL[___] := $Failed


End[]

SetAttributes[
    {

        (* boolean *)

        WGLGetInstanceList,

        WGLCreateInstance,
        WGLCopyInstance,
        WGLDeleteInstance,
        WGLValidInstance,

        WGLSetCoordinates,
        WGLSetCells,

        WGLGetCoordinates,
        WGLGetCells,

        WGL2DPolygonIntersection,
        WGL2DPolygonUnion,
        WGL2DPolygonDifference,
        WGL2DPolygonSymmetricDifference,

	    (* Import/Export *)

	    WGTAiScene,
	    WGTAiSceneInstanceList,
	    WGTAssimpImport,
	    WGTAssimpVertices,
	    WGTAssimpVertexNormals,
	    WGTAssimpVertexTextureCoordinates,
	    WGTAssimpVertexColors,
	    WGTAssimpMeshCellCount,
	    WGTAssimpPolygonCells,
	    WGTAssimpTriangleCells,
	    WGTAssimpLineCells,

	    WGTAssimpExport,
	    WGTAssimpExportSTL,

	    (* Polygon Tesselation *)

	    	TESSCreateInstance,
		TESSDeleteInstance,
		TESSValidInstance,
		TESSGetInstanceList,
		TESSAddContour,
		TESSTesselate,
		TESSGetCoordinates,
		TESSGetCoordinateMap,
		TESSGetElementCount,
		TESSGetPrimitiveIndices,
		TESSGetCellIndices,
		TESSGetBoundaryIndices

    },
    {ReadProtected, Protected}
]

EndPackage[]


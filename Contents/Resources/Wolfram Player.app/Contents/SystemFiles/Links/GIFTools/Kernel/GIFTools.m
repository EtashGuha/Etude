(* ::Package:: *)

BeginPackage["GIFTools`"]


Begin["`Private`"]

$InitGIFTools = False;

$ThisDirectory = FileNameDrop[$InputFileName, -1]
$BaseLibraryDirectory = FileNameJoin[{$ThisDirectory, "LibraryResources", $SystemID}];
$GIFToolsLibrary = "GIFTools";
dlls["Linux"|"Linux-x86-64"|"Linux-ARM"|"MacOSX-x86-64"] = {"libgif"};
dlls["Windows"|"Windows-x86-64"] = {"giflib5"};
dlls[___] := $Failed;

safeLibraryLoad[debug_, lib_] :=
	Quiet[
		Check[
			LibraryLoad[lib],
			If[TrueQ[debug],
				Print["Failed to load ", lib]
			];
			Throw[$InitGIFTools = $Failed]
		]
	]
safeLibraryFunctionLoad[debug_, args___] :=
	Quiet[
		Check[
			LibraryFunctionLoad[$GIFToolsLibrary, args],
			If[TrueQ[debug],
				Print["Failed to load the function ", First[{args}], " from ", $GIFToolsLibrary]
			];
			Throw[$InitGIFTools = $Failed]
		]
	]
  
InitGIFTools[debug_:False] := If[TrueQ[$InitGIFTools],
	$InitGIFTools,
	$InitGIFTools = Catch[
	  If[dlls[$SystemID] === $Failed,
	  	Message[GIFTools::sys, "Incompatible SystemID"];
	  	Throw[$Failed]
	  ];
	  Block[{$LibraryPath = Prepend[$LibraryPath, $BaseLibraryDirectory]},
		  safeLibraryLoad[debug, #]& /@ Flatten[{dlls[$SystemID], $GIFToolsLibrary}];
		  
		  (* Import *)
		  $ReadOneFrame = safeLibraryFunctionLoad[debug,"ReadOneFrame",{{"UTF8String"},Integer}, "Image"];
		  $ReadAllFrames = safeLibraryFunctionLoad[debug,"ReadAllFrames",{"UTF8String"}, LibraryDataType[Image|Image3D]];
		  $ReadGlobalPalette = safeLibraryFunctionLoad[debug,"ReadGlobalPalette",{"UTF8String"}, {_Integer, _}];
		  $ReadPalettes = safeLibraryFunctionLoad[debug,"ReadPalettes", LinkObject, LinkObject];
		  $ReadRasterBits = safeLibraryFunctionLoad[debug,"ReadRasterBits", LinkObject, LinkObject];
		  $ReadFileMetadata = safeLibraryFunctionLoad[debug,"ReadFileMetadata", LinkObject, LinkObject];
		  $ReadFrameMetadata = safeLibraryFunctionLoad[debug,"ReadFrameMetadata", LinkObject, LinkObject];
		  $ClearCache = safeLibraryFunctionLoad[debug,"ClearCache", {"UTF8String"}, True|False];
		  
		  (* Export *)
		  $StringSeparator = safeLibraryFunctionLoad[debug, "StringSeparator", {}, "UTF8String"];
		  $WriteFrames = safeLibraryFunctionLoad[debug, "WriteFrames", {
		  	{LibraryDataType[Image|Image3D], "Constant"},	(* Image to be exported *)
		  	"UTF8String",									(* Output file name *)
		  	"UTF8String",									(* Comment *)
		  	{Real, 1, "Constant"}, 							(* Display durations *)
		  	"UTF8String",									(* Dithering method *)
		  	Integer,										(* Loop count *)
		  	"Boolean",										(* Interlacing *)
		  	{Integer, 1, "Constant"}, 						(* Background color *)
		  	{Integer, _, "Constant"}, 						(* Transparent color(s) *)
		  	{Integer, 1, "Constant"}						(* Disposal mode *)
		  }, "UTF8String"];									(* Return: output file name *)
		  
		  $ImageQuantize = safeLibraryFunctionLoad[debug, "ImageQuantize", {
		  	{LibraryDataType[Image|Image3D], "Constant"},	(* Image to be quantized *)
		  	Integer, 										(* Color map length *)
		  	"UTF8String",									(* Dithering method *)
		  	"Boolean",										(* Return color map? (otherwise return raw indices) *)
		  	Integer,										(* Quantization method (1 - MedianCut, 2 - Wu, 3 - Octree) *)
		  	Integer											(* Precision parameter *)
		  }, "RawArray"];
		  
		  $GetQuantizedImage = safeLibraryFunctionLoad[debug, "GetQuantizedImage", {
		  	{LibraryDataType[Image|Image3D], "Constant"},	(* Image to be quantized *)
		  	Integer, 										(* Color map length *)
		  	"UTF8String",									(* Dithering method *)
		  	Integer,										(* Quantization method (1 - MedianCut, 2 - Wu, 3 - Octree) *)
		  	Integer											(* Precision parameter *)
		  }, LibraryDataType[Image|Image3D]];
		  
		  $DitherImage = safeLibraryFunctionLoad[debug, "DitherImage", {
		  	{LibraryDataType[Image|Image3D], "Constant"},	(* Image to be quantized and dithered *)
		  	{"RawArray", "Constant"},						(* Color map *)
		  	"UTF8String",									(* Dithering method *)
		    "Boolean",										(* Use cache? *)
		    "Boolean"										(* Use fast method of pixel mapping? *)
		  }, LibraryDataType[Image|Image3D]];
		  
		  $WriteColormaps = safeLibraryFunctionLoad[debug, "WriteColormaps", {
		  	{"RawArray", "Constant"},						(* Colormap(s) *)
		  	{"RawArray", "Constant"}, 						(* Raw data (indices) *)
		  	"UTF8String",									(* Output file name *)
		  	"UTF8String",									(* Comment *)
		  	{Real, 1, "Constant"}, 							(* Display durations *)
		  	"UTF8String",									(* Dithering method *)
		  	Integer,										(* Loop count *)
		  	"Boolean",										(* Interlacing *)
		  	{Integer, 1, "Constant"}, 						(* Background color *)
		  	{Integer, _, "Constant"}, 						(* Transparent color(s) *)
		  	{Integer, 1, "Constant"}						(* Disposal mode *)
		  }, "UTF8String"];									(* Return: output file name *)
		  
		  $UniqueColors = safeLibraryFunctionLoad[debug, "UniqueColors", {
		  	{LibraryDataType[Image|Image3D], "Constant"},	(* Image to be color-counted *)
		  	"Boolean"	(* whether to use std::map (True) or std::unordered_map (False) *)
		  }, Integer];
	  ];
	  True
	]
]

End[]
EndPackage[]

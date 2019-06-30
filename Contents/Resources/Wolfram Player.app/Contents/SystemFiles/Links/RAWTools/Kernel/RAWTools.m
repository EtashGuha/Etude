BeginPackage["RAWTools`"]


Begin["`Private`"]

$InitRAWTools = False;

$ThisDirectory = FileNameDrop[$InputFileName, -1]
$BaseLibraryDirectory = FileNameJoin[{$ThisDirectory, "LibraryResources", $SystemID}];
$RawToolsLibrary = "RAWTools";
dlls["Windows"|"Windows-x86-64"] = {"zlib","jpeg","lcms2","libraw"};
dlls["Linux"|"Linux-x86-64"|"MacOSX-x86-64"] = {"libjpeg","liblcms2","libraw"};
dlls["Linux-ARM"] = {};
dlls[___] := $Failed;

safeLibraryLoad[debug_, lib_] :=
	Quiet[
		Check[
			LibraryLoad[lib],
			If[TrueQ[debug],
				Print["Failed to load ", lib]
			];
			Throw[$InitRAWTools = $Failed]
		]
	]
safeLibraryFunctionLoad[debug_, args___] :=
	Quiet[
		Check[
			LibraryFunctionLoad[$RawToolsLibrary, args],
			If[TrueQ[debug],
				Print["Failed to load the function ", First[{args}], " from ", $RawToolsLibrary]
			];
			Throw[$InitRAWTools = $Failed]
		]
	]
  
InitRAWTools[debug_:False] := If[TrueQ[$InitRAWTools],
	$InitRAWTools,
	$InitRAWTools = Catch[
	  If[dlls[$SystemID] === $Failed,
	  	Message[RAWTools::sys, "Incompatible SystemID"];
	  	Throw[$Failed]
	  ];
	  Block[{$LibraryPath = Prepend[$LibraryPath, $BaseLibraryDirectory]},
		  safeLibraryLoad[debug, #]& /@ Flatten[{dlls[$SystemID], $RawToolsLibrary}];
		  $ReadImageRAWInternal = safeLibraryFunctionLoad[debug, "ReadImageRAW",{{"UTF8String"},
		  	{_Integer, 1}, {_Integer, 1}, 
		  	{_Integer, 1}, {_Integer, 1}, 
		  	{_Real, 1}, {_Real, 1},
		  	{_Real, 1}, 
		  	_Integer, 
		  	_Real, 
		  	_Real,
		  	"Boolean", 
		  	"Boolean", 
		  	_Integer, 
		  	"Boolean", "Boolean", _Integer,
		  	{"UTF8String"}, {"UTF8String"}, {"UTF8String"}, {"UTF8String"}, 
		  	_Integer, 
		  	_Integer, 
		  	_Integer, {_Integer, 1},
		  	_Integer, 
		  	_Integer, 
		  	"Boolean", _Real, 
		  	_Real,
		  	"Boolean", 
		  	"Boolean", 
		  	_Integer, "Boolean", 
		  	_Integer, 
		  	"Boolean",
		  	_Integer, 
		  	"Boolean", {_Real, 1},
		  	"Boolean", _Real,
		  	"Boolean", {_Real, 1}, 
		  	"Boolean", _Real,
		  	"Boolean", {_Real, 1},
		  	"Boolean", {_Real, 1},
		  	"Boolean", 
		  	"Boolean", 
		  	"Boolean", 
		  	_Real,
		  	{"UTF8String"}}, "Image"];
		  $ReadImageRAW[file_, quality_, cameraWB_, gammaInv_, gammaSlope_] = $ReadImageRAWInternal[file,
		  	{0, 0}, {-1, -1}, 
		  	{0, 0}, {-1, -1}, 
		  	{-1, -1}, {gammaInv, gammaSlope}, 
		  	{1.0, 1.0, 1.0, 1.0}, 
		  	0, 
		  	1.0, 
		  	0.0, 
		  	False, 
		  	False, 
		  	0, 
		  	True, cameraWB, 1, 
		  	"", "", "", "",
		  	-1, 
		  	quality,
		  	-1, {-1, -1, -1, -1}, 
		  	-1, 
		  	0,
		  	True, -1, (* auto bright turned off *)
		  	-1, 
		  	True, 
		  	False, 
		  	-1, False, 
		  	-1, 
		  	False, 
		  	-1,
		  	False, {-1, -1},
		  	False, -1,
		  	False, {-1, -1},
		  	False, -1,
		  	False, {-1, -1}, 
		  	False, {0, 0, 0, 0}, 
		  	True, 
		  	False, 
		  	False, 
		  	-1,
		  	"3102"];
		  $ReadThumbnailRAW = safeLibraryFunctionLoad[debug, "ReadThumbnailRAW",{{"UTF8String"}}, "Image"];
		  $ReadDataRAWInternal = safeLibraryFunctionLoad[debug, "ReadDataRAW",{{"UTF8String"}, "Boolean"}, {_Integer, _}];
		  $ReadDataRAW[file_] = $ReadDataRAWInternal[file, True];
		  $ReadMetadataStringRAW = safeLibraryFunctionLoad[debug, "ReadMetadataStringRAW", {{"UTF8String"}, {"UTF8String"}}, "UTF8String"];
		  $ReadMetadataIntegerRAW = safeLibraryFunctionLoad[debug, "ReadMetadataIntegerRAW", {{"UTF8String"}, {"UTF8String"}}, _Integer];
		  $ReadMetadataRealRAW = safeLibraryFunctionLoad[debug, "ReadMetadataRealRAW", {{"UTF8String"}, {"UTF8String"}}, _Real];
		  $ReadGPSDataRAW = safeLibraryFunctionLoad[debug, "ReadGPSDataRAW", {"UTF8String"}, {_Integer, 1}];
		  $ReadICCRAW = safeLibraryFunctionLoad[debug, "ReadICCRAW", {"UTF8String"}, {_Integer, 1}];
		  $GetColorInformation = safeLibraryFunctionLoad[debug, "GetColorInformation", LinkObject, LinkObject];
		  $GetColorFilterPattern = safeLibraryFunctionLoad[debug, "GetColorFilterPattern", LinkObject, LinkObject];
		  $ReadExifRAW = safeLibraryFunctionLoad[debug, "ReadExifRAW", LinkObject, LinkObject];
	  ];
	  True
	]
]


End[]
EndPackage[]

BeginPackage["ImageFileTools`", {"ImageFileTools`RLE`"}]

Begin["`Private`"]

$InitImageFileTools = False;

$ImageFileToolsBaseDirectory = FileNameDrop[$InputFileName, -2];

Get[FileNameJoin[{$ImageFileToolsBaseDirectory, "LibraryResources", "LibraryLinkUtilities.wl"}]];

$BaseLibraryDirectory = FileNameJoin[{$ImageFileToolsBaseDirectory, "LibraryResources", $SystemID}];
$ImageFileToolsLibrary = "ImageFileTools";

InitImageFileTools[debug_ : False] :=
	If[TrueQ[$InitImageFileTools],
		$InitImageFileTools
		,
		$InitImageFileTools =
			Catch[
				Block[
					{
						$LibraryPath = Prepend[$LibraryPath, $BaseLibraryDirectory]
					},

					SafeLibraryLoad[debug, $ImageFileToolsLibrary];

					(* Initialize RLE *)
					InitImageFileToolsRLE[debug];
				];
				True
			]
	]

End[]

EndPackage[]
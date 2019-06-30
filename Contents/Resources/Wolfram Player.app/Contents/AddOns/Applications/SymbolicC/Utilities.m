
BeginPackage["SymbolicC`Utilities`"]

ClangFormatString
HasClangFormatQ

Begin["`Private`"]

$DefaultClangFormatSearchPath = {
	"/usr/local/opt/llvm@6/bin",
	"/usr/local/opt/llvm@5/bin",
	"/usr/local/bin",
	"/usr/bin",
	"/bin"
}

clangFormatSearchPath[opts_?AssociationQ] := 
	With[{
		path = Lookup[opts, "Path", $DefaultClangFormatSearchPath]
	},
		If[path === Automatic,
			$DefaultClangFormatSearchPath,
			path
		]
	]

ClearAll[HasClangFormatQ]
HasClangFormatQ[opts_?AssociationQ] := HasClangFormatQ[opts] =
	FileNames[
		"clang-format",
		clangFormatSearchPath[opts]
	] =!= {}

ClangFormatString[str_, opts:OptionsPattern[]] :=
	ClangFormatString[str, <| opts |>]
ClangFormatString[str_, opts_?AssociationQ] :=
	If[HasClangFormatQ[opts],
		Module[{
			res,
			tmpFile = makeTemporaryFileName["cpp"]
		},
			Export[tmpFile, str, "Text"];
			RunProcess[
				{"clang-format",  "-style={BasedOnStyle: llvm, ColumnLimit: 100}", "-i", tmpFile},
				All,
				ProcessEnvironment -> <|
					"PATH" -> StringRiffle[
						clangFormatSearchPath[opts],
						":"
					]
				|>
			];
			res = Import[tmpFile, "Text"];
			If[DirectoryQ[$TemporaryClangFormatDirectory],
	            DeleteDirectory[$TemporaryClangFormatDirectory, DeleteContents->True]
			];
			res
		],
		(* does not have clang-format *)
		str
	]
	
createDirIfDoesNotExist[dir_String, failOnExisting:(True|False):False] :=
    Quiet[
        Check[CreateDirectory[dir], If[failOnExisting, $Failed, dir], CreateDirectory::filex],
        {CreateDirectory::filex}
    ];

createDirIfDoesNotExist[parts__String, failOnExisting:(True|False):False] :=
    createDirIfDoesNotExist[FileNameJoin[{parts}], failOnExisting];

$TemporaryClangFormatDirectory := createDirIfDoesNotExist[$TemporaryDirectory, "ClangFormat"];

 
makeTemporaryFileName[extension_String, directory:_String?DirectoryQ|Automatic:Automatic]:=
    With[{dir = If[directory === Automatic, $TemporaryClangFormatDirectory, directory]},
        FileNameJoin[{dir, generateRandomString[10] <> "." <> extension}]
    ];

generateRandomString[n_Integer]:=
    StringJoin[
        ToString @ FromCharacterCode @ RandomInteger[{97,122},n],
        ToString @ AbsoluteTime[DateString[]]
    ];
    
End[]

EndPackage[]
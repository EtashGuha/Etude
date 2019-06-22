(* Mathematica Package *)

(* Created by the Wolfram Workbench Jan 31, 2010 *)

BeginPackage["LibraryLink`"]
(* Exported symbols added here with SymbolName::usage *) 

LibraryVersionInformation::usage = "LibraryVersionInformation[lib] returns a list of rules of library version information."
LibraryVersionInformation::invfile = "File `1` was not found in the library path."
LibraryVersionInformation::infsymloop = "Detected infinite symbolic loop in `1`."

LibraryVersionString::usage = "LibraryVersionString[lib] returns a string of library version information."

$LibraryError::usage = "$LibraryError returns the system dependent error message from loading a library or None if there was no error."

Begin["`Private`"]
(* Implementation of the package *)

Unprotect[
	LibraryVersionInformation
]

$LibraryLinkPath = DirectoryName[System`Private`$InputFileName];
$LibraryLinkLibraryResourcesPath = FileNameJoin[{$LibraryLinkPath, "LibraryResources", $SystemID}];

PrependTo[$LibraryPath, $LibraryLinkLibraryResourcesPath];

initializedQ = False
$LIBNAME := FindLibrary["LibraryVersionInformation"]


initialize[] :=
	Module[{Libraryfuns},
		If[initializedQ =!= False,
			Return[initializedQ]
		];
		initializedQ = If[$LIBNAME === $Failed,
			$Failed,
			Libraryfuns = Quiet /@ 
				Join[
					If[$OperatingSystem === "Windows",
						{
							cFileGetVersionInformation = LibraryFunctionLoad[$LIBNAME,
																		     "FileGetVersionInformation",
																		      {"UTF8String", "UTF8String"},
																			  "UTF8String"
							]
						},
						{
						}
					],
					{If[$OperatingSystem=!="Unix",
						cLibraryGetVersion = LibraryFunctionLoad[$LIBNAME,
																"LibraryGetVersion",
																{"UTF8String"},
																{_Integer, 1, Automatic}
						],Nothing],
						cWolframLibraryVersion = LibraryFunctionLoad[$LIBNAME,
															"getWolframLibraryVersion",
															{"UTF8String"},
															{_Integer}
						]
					}
				];
			If[MemberQ[Libraryfuns, $Failed],
				$Failed,
				True
			] 
		]
	]

LibraryPaths[] /; $OperatingSystem === "Windows" := Union[Join[
															$LibraryPath,
															StringSplit[Environment["PATH"], ";"]
													]]
LibraryPaths[] /; $OperatingSystem === "MacOSX" := Union[Join[
															$LibraryPath,
															StringReplace[#, "/bin" -> "/lib"]& /@ StringSplit[Environment["PATH"], ":"],
															{"/lib", "/usr/lib", "/usr/local/lib"}
													]]
LibraryPaths[] /; $OperatingSystem === "Unix" := Union[Join[
															$LibraryPath,
															StringSplit[Environment["LD_LIBRARY_PATH"], ":"],
															{"/lib", "/usr/lib", "/usr/local/lib"}
												 ]]
FindLibraryAbsolutePath[name_String] :=
	iFindLibraryAbsolutePath[name, ".so"] /; $OperatingSystem === "Unix"
FindLibraryAbsolutePath[name_String] :=
	iFindLibraryAbsolutePath[name, ".dylib"] /; $OperatingSystem === "MacOSX"
	
iFindLibraryAbsolutePath[name0_String, ext_String] :=
	Module[{name, candidates = {}, 
		ldpath = LibraryPaths[], libver},
		name = If[StringMatchQ[name0, ___ ~~ "/" ~~ ___] || StringMatchQ[name0, ___ ~~ ext ~~ ___],
			name0,
			name0 <> ext
		];
		If[StringMatchQ[name, ___ ~~ ("/") ~~ ___],
			Return[{name}]
		]; 
		libver = If[$SystemID === "Linux-x86-64", "64", ""];
		If[!MemberQ[ldpath, #],
			AppendTo[ldpath, #]
		] & /@ {"/lib" <> libver, "/usr/lib" <> libver};
		Function[{loc},
			If[FileExistsQ[FileNameJoin[{loc, name}]],
				AppendTo[candidates, FileNameJoin[{loc, name}]]
			]
		] /@ ldpath;
		candidates
	]

RealLibraryPath[pth_String] :=
	Module[{libPaths, loc},
		If[StringMatchQ[pth, ___ ~~ ("/" | "\\") ~~ ___],
			pth,
			libPaths = LibraryPaths[];
			loc = Select[libPaths, FileExistsQ[FileNameJoin[{#, pth}]]&, 1];
			If[loc === {},
				$Failed,
				FileNameJoin[{First[loc], pth}]
			]
		]
	] /; $OperatingSystem === "Windows"
RealLibraryPath[pth_String] := Quiet[AbsoluteFileName[pth]]

LibraryVersionString[name_String] :=
	LibraryVersionString[LibraryVersionInformation[name]]
LibraryVersionString[verInfo_Association] :=
	LibraryVersionString[Normal[verInfo]]
LibraryVersionString[verInfo_List] :=
	Module[{major = "MajorVersion" /. verInfo,
			minor = "MinorVersion" /. verInfo,
			revision = "RevisionNumber" /. verInfo},
		If[MemberQ[{major, minor, revision}, Except[_Real | _Integer]],
			"",
			StringJoin[Riffle[ToString /@ {major, minor, revision}, "."]]
		]
	]
LibraryVersionString[___] := ""

LibraryVersionInformation[name_String] :=(
	If[initializedQ === $Failed || initialize[] === $Failed,Return[$Failed]];
	iLibraryVersionInformation[name, ".so"]
	)
	
iLibraryVersionInformation[name_String, ext_String] :=
	Module[{pths, pth, strversion, splitversion, majorVersion = 0, minorVersion = 0, revionNumber = 0},
		pths = Catch[RealLibraryPath[#] & /@ FindLibraryAbsolutePath[name]];
		If[Length[pths] > 0 && pths =!= {$Failed},
			Quiet[Check[
				pth = First[pths];
				If[StringMatchQ[pth, {__ ~~ ext <> "." ~~ ___, 
									  __ ~~ "-" ~~  (DigitCharacter ~~ "." ...) .. ~~ ext,
									  ShortestMatch[__] ~~  ((DigitCharacter ~~ "." ...) ..) ~~ ext}],
					strversion = StringReplace[pth, {__ ~~ ext <> "." ~~ x___ -> x,
						   							 ShortestMatch[__] ~~ "-" ~~ x : ((DigitCharacter ~~ "." ...) ..) ~~ ext -> x,
						   							 ShortestMatch[__] ~~ x : ((DigitCharacter ~~ "." ...) ..) ~~ ext -> x}];
					splitversion = ToExpression /@ StringSplit[strversion, "."];
					Switch[Length[splitversion],
						1,
							majorVersion = splitversion[[1]];,
						2,
							majorVersion = splitversion[[1]];
							minorVersion = splitversion[[2]];, 
						_,
							majorVersion = splitversion[[1]];
							minorVersion = splitversion[[2]];
							revionNumber = ToExpression @ StringJoin[Riffle[ToString/@splitversion[[3;;If[Length[splitversion] > 4, 4, Length[splitversion]]]], "."]]
					];
					Association[{
						"Name" -> name,
						"Path" -> pth,
						"MajorVersion" -> majorVersion, 
						"MinorVersion" -> minorVersion,
						"RevisionNumber" -> revionNumber,
				 		"WolframLibraryVersion" -> getWolframLibraryVersion[pth]
					}],
					$Failed (* no version information *)
				], $Failed
			]],
			Message[LibraryVersionInformation::invfile, name];
			$Failed (* no path was found *)
		]
	] /; $OperatingSystem =!= "Windows"

LibraryVersionUsingOtool[pth_String] :=
	Module[{hasotoolQ, ln, verstr, verexpr},
		hasotoolQ = Import["!which otool", "Text"] =!= "";
		If[hasotoolQ === False,
			Return[$Failed]
		];
		ln = Import["!otool -L " <> pth <> "| grep current | head -1", "Text"];
 		If[StringQ[ln],
  			verstr = StringReplace[ln, ___ ~~ "current version " ~~ ver___ ~~ ")" ~~ ___ -> ver];
  			verexpr = ToExpression /@ StringSplit[verstr, "."];
  			If[FreeQ[verexpr, Except[_Integer], {1}, Heads -> False],
   				verexpr,
   				$Failed
   			],
   			$Failed
  		]
 	]


getWolframLibraryVersion[libPath_String] :=
	Module[{wolframLibraryVersion},
		wolframLibraryVersion = cWolframLibraryVersion[libPath];
		If[wolframLibraryVersion > 0,
			wolframLibraryVersion,
			$Failed
		]
	]
	
LibraryVersionInformation[name_String] :=
	Module[{pths, otoolversion, splitversion},
		If[initializedQ === $Failed || initialize[] === $Failed,
			Return[$Failed]
		];
		pths = Catch[RealLibraryPath[#] & /@ FindLibraryAbsolutePath[name]];
		If[Length[pths] > 0 && pths =!= {$Failed},
			otoolversion = Quiet[LibraryVersionUsingOtool[First@pths]];
			splitversion = If[ListQ[otoolversion] && otoolversion =!= {},
				otoolversion,
				cLibraryGetVersion[First@pths]
			];
			If[ListQ[splitversion] && splitversion =!= {},
				Association[{
				 "Name" -> name,
				 "Path" -> First[pths],
				 "MajorVersion" -> splitversion[[1]],
				 "MinorVersion" -> splitversion[[2]],
				 "RevisionNumber" -> splitversion[[3]],
				 "WolframLibraryVersion" -> getWolframLibraryVersion[First@pths]
				}],
				iLibraryVersionInformation[name, ".dylib"]
			]
			,
			Message[LibraryVersionInformation::invfile, name];
			$Failed (* no path was found *)
		]
	] /; $OperatingSystem === "MacOSX"
	
LibraryVersionInformation[name0_String] := 
	Module[{name, LibraryRes, pth},
		If[initializedQ === $Failed || initialize[] === $Failed,
			Return[$Failed]
		];
		name = If[StringMatchQ[name0, ___ ~~ ("/" | "\\") ~~ ___] || StringMatchQ[name0, ___ ~~ ".dll"],
			name0,
			name0 <> ".dll"
		];
		pth = RealLibraryPath[name];
		If[pth === $Failed,
			Message[LibraryVersionInformation::invfile, name];
			Return[$Failed]
		];
		Quiet[Check[
			LibraryRes = FileGetVersionInformation[pth, "ProductVersion"];
			If[StringQ[LibraryRes] && StringTrim[LibraryRes] =!= "",
				Return[formatVersionInformation[name, pth, ToExpression /@ StringSplit[DeleteNoneNumeric@LibraryRes, {".", ","}]]],
				LibraryRes = cLibraryVersionInformation[pth];
				If[StringQ[LibraryRes] && StringTrim[LibraryRes] =!= "",
					Return[formatVersionInformation[name, pth, ToExpression /@ StringSplit[DeleteNoneNumeric@LibraryRes, {".", ","}]]],
					LibraryRes = FileGetVersionInformation[pth, "FileVersion"];
					If[StringQ[LibraryRes] && StringTrim[LibraryRes] =!= "",
						Return[formatVersionInformation[name, pth, ToExpression /@ StringSplit[DeleteNoneNumeric@LibraryRes, {".", ","}]]],
						$Failed
					]
				]
			],
			$Failed
		]]
	] /; $OperatingSystem === "Windows"

DeleteNoneNumeric[str_String] := StringReplace[str, {x : DigitCharacter -> x, "." -> ".", _ -> ""}]

formatVersionInformation[name_, pth_, ver_List] /; $OperatingSystem === "Windows" :=
	Association[{
	 "Name" -> FileBaseName[name],
	 "Path" -> pth,
	 "MajorVersion" -> ver[[1]],
	 "MinorVersion" -> If[Length[ver] < 2, 0, ver[[2]]],
	 "RevisionNumber" -> If[Length[ver] < 3, 0, ToExpression @ StringJoin[Riffle[ToString/@ver[[3;;If[Length[ver] > 4, 4, Length[ver]]]], "."]]],
	 "WolframLibraryVersion" -> getWolframLibraryVersion[pth]
	}]

FileGetVersionInformation::invquery = "The query `1` is an invalid query."
FileGetVersionInformation[pth_String, query_String] :=
	Module[{},
		If[initializedQ === $Failed || initialize[] === $Failed,
			Return[$Failed]
		];
		If[!FileExistsQ[pth],
			Message[LibraryVersionInformation::invfile, pth];
			Return[$Failed]
		];
		If[ValidFileGetVersionInformationQueryQ[query],
			cFileGetVersionInformation[pth, query],
			Message[FileGetVersionInformation::invquery, query];
			$Failed
		]
	] /; $OperatingSystem === "Windows"

ValidFileGetVersionInformationQueryQ[query_String] :=
	Switch[query,
		"Comments" | "CompanyName" | "FileDescription" | "FileVersion" | "InternalName" |
		"LegalCopyright" | "LegalTrademarks" | "OriginalFilename" | "ProductName" |
		"ProductVersion" | "PrivateBuild" | "SpecialBuild", True,
		_, False
	] /; $OperatingSystem === "Windows"

SetAttributes[{
		LibraryVersionInformation
	}, {ReadProtected, Protected}
]

 
End[]

EndPackage[]


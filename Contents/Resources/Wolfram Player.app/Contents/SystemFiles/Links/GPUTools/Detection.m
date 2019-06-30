(* Mathematica Test File *)

BeginPackage["GPUTools`Detection`", {"GPUTools`Utilities`"}]

GPUTools`Internal`VideoCardInformation::usage = "GPUTools`Internal`VideoCardInformation[] queries video card information."
GPUTools`Internal`VideoCardName::usage = "GPUTools`Internal`VideoCardName[] returns the name of video card."
GPUTools`Internal`ValidSystemQ::usage = "GPUTools`Internal`ValidSystemQ[] True if system operating system supports CUDA."
GPUTools`Internal`$NVIDIADriverLibraryValidQ::usage = "GPUTools`Internal`$NVIDIADriverLibraryValidQ checks if the NVIDIA driver version is valid."
GPUTools`Internal`$NVIDIADriverLibraryVersion::usage "GPUTools`Internal`$NVIDIADriverLibraryVersion NVIDIA driver version"
GPUTools`Internal`$CUDALibraryValidQ::usage = "GPUTools`Internal`$CUDALibraryValidQ checks if the CUDA Library version is valid."
GPUTools`Internal`$CUDALibraryVersion::usage = "GPUTools`Internal`$CUDALibraryVersion CUDA library version." 
GPUTools`Internal`$NVIDIADriverLibraryPath::usage "GPUTools`Internal`$NVIDIADriverLibraryPath path to the NVIDIA driver Library"
GPUTools`Internal`$CUDALibraryPath::usage = "GPUTools`Internal`$CUDALibraryPath path to the CUDA system Library"
GPUTools`Internal`$GPUToolsPath::usage ="GPUTools`Internal`$GPUToolsPath path to the GPUTools application."
GPUTools`Internal`$GPUToolsLibraryResourcesPath = "GPUTools`Internal`$GPUToolsLibraryResourcesPath path to the GPUTools Library resources directory."
GPUTools`Internal`$GPUToolsSystemResourcesPath::usage = "GPUTools`Internal`$GPUToolsSystemResourcesPath path to the GPUTools system files."
GPUTools`Internal`$CUDALinkSystemLibaries::usage = "GPUTools`Internal`$CUDALinkSystemLibaries system libraries required by GPUTools."
GPUTools`Internal`FindCUDALinkSystemLibaries::usage = "GPUTools`Internal`$CUDALinkSystemLibaries system libraries required by GPUTools."
GPUTools`Internal`LoadCUDALibraries::usage = "GPUTools`Internal`LoadCUDALibraries loads CUDA libraries"
GPUTools`Internal`CUDAQueryFailReason::usage = "GPUTools`Internal`CUDAQueryFailReason[] returns messages as to why GPUTools failed to load."
GPUTools`Internal`$IsLegacyGPU::usage = "GPUTools`Internal`$IsLegacyGPU returns True|False depending on whether a legacy GPU is detected"
GPUTools`Internal`$NVIDIAOpenCLDriverValidQ::usage = "GPUTools`Internal`$NVIDIAOpenCLDriverValidQ checkes whether NVIDIA OpenCL driver is valid or not"
GPUTools`Internal`$NVIDIAOpenCLLibraryDirectory::usage = "GPUTools`Internal`$NVIDIAOpenCLLibraryDirectory directory housing the OpenCL library"
GPUTools`Internal`$NVIDIAOpenCLLibraryName::usage = "GPUTools`Internal`$NVIDIAOpenCLLibraryName name of the OpenCL library"
GPUTools`Internal`$NVIDIAOpenCLLibraryPath::usage = "GPUTools`Internal`$NVIDIAOpenCLLibraryPath path to the OpenCL library"

GPUTools`Internal`$ATIOpenCLLibraryDirectory::usage = "GPUTools`Internal`$ATIOpenCLLibraryDirectory directory housing the OpenCL library"
GPUTools`Internal`$ATIOpenCLLibraryName::usage = "GPUTools`Internal`$ATIOpenCLLibraryName name of the OpenCL library"
GPUTools`Internal`$ATIOpenCLLibraryPath::usage = "GPUTools`Internal`$ATIOpenCLLibraryPath path to the OpenCL library"

GPUTools`Internal`$ATIDriverPath::usage = "GPUTools`Internal`$ATIDriverPath path to the ATI driver library"
GPUTools`Internal`$ATIDriverVersion::usage = "GPUTools`Internal`$ATIDriverVersion ATI driver version"
GPUTools`Internal`$ATIDriverValidQ::usage = "GPUTools`Internal`$ATIDriverValidQ whether ATI driver is supported or not"

GPUTools`Internal`UsingNVIDIAQ::usage = "GPUTools`Internal`UsingNVIDIAQ whether NVIDIA OpenCL is used"
GPUTools`Internal`UsingATIQ::usage = "GPUTools`Internal`UsingNVIDIAQ whether ATI OpenCL is used"

GPUTools`Internal`$OpenCLDriverValidQ::usage = "GPUTools`Internal`$OpenCLDriverValidQ whether OpenCL driver is valid"
GPUTools`Internal`$OpenCLLibraryDirectory::usage = "GPUTools`Internal`$OpenCLLibraryDirectory OpenCL library directory"
GPUTools`Internal`$OpenCLLibraryPath::usage = "GPUTools`Internal`$OpenCLLibraryPath OpenCL library path"
GPUTools`Internal`$OpenCLLibraryName::usage = "GPUTools`Internal`$OpenCLLibraryName OpenCL library name"
GPUTools`Internal`$OpenCLDriverVersion::usage = "GPUTools`Internal`$OpenCLDriverVersion OpenCL driver version"

GPUTools`Internal`OpenCLQueryFailReason::usage = "GPUTools`Internal`OpenCLQueryFailReason[msgHd] returns reason for OpenCL query failure"
GPUTools`Internal`LoadOpenCLLibraries::usage = "GPUTools`Internal`LoadOpenCLLibraries loads OpenCL libraries"

GPUTools`Internal`$ATISTREAMSDKROOT::usage = "GPUTools`Internal`$ATISTREAMSDKROOT path to the ATISTREAMSDKROOT"

Begin["`Private`"]

Needs["LibraryLink`"]
Needs["PacletManager`"]
Needs["GPUTools`Utilities`"]


(* ::Section:: *)
(* Helper Methods *)

WindowsDriveLetter[] := Environment["SystemDrive"]
WindowsDirectory[] := Environment["SystemRoot"]

(* ::Section:: *)
(* OpenCL Hardware Tools *)


GPUTools`Internal`$NVIDIAOpenCLDriverValidQ /; $SystemID === "MacOSX-x86-64" :=
	DirectoryQ[FileNameJoin[{"/System", "Library", "Frameworks", "OpenCL.framework"}]]
GPUTools`Internal`$NVIDIAOpenCLDriverValidQ /; $SystemID === "MacOSX-x86" :=
	False
GPUTools`Internal`$NVIDIAOpenCLDriverValidQ /; $OperatingSystem === "Unix" :=
	With[{version = GPUTools`Internal`$NVIDIADriverLibraryVersion},
		If[version === $Failed || !NumberQ[GPUTools`Utilities`LibraryGetMajorVersion[version]],
			False,
			GPUTools`Utilities`LibraryGetMajorVersion[version] >= 12.0
		]
	]
GPUTools`Internal`$NVIDIAOpenCLDriverValidQ /; $OperatingSystem === "Windows" :=
	With[{version = GPUTools`Internal`$NVIDIADriverLibraryVersion},
		If[version === $Failed || !NumberQ[GPUTools`Utilities`LibraryGetRevisionVersion[version]],
			False,
			GPUTools`Utilities`LibraryGetRevisionVersion[version] >= 12.0
		]
	]

GPUTools`Internal`$NVIDIAOpenCLLibraryDirectory /; $SystemID === "Linux" :=
	FileNameJoin[{"/usr", "lib"}]
GPUTools`Internal`$NVIDIAOpenCLLibraryDirectory /; $SystemID === "Linux-x86-64" :=
	Module[{dirs},
		dirs =Select[
			{
				FileNameJoin[{"/usr", "lib64"}],
				FileNameJoin[{"/usr", "lib"}],
				FileNameJoin[{Lookup[PacletManager`PacletInformation["CUDAResources"],"Location",""],"CUDAToolkit","lib64"}],
				FileNameJoin[{"/usr/lib/x86_64-linux-gnu"}]
			},
			FileExistsQ[FileNameJoin[{#,"libOpenCL.so"}]]&
		];
		If[dirs === {},
			$Failed,
			First[dirs]
		]
	]
GPUTools`Internal`$NVIDIAOpenCLLibraryDirectory /; $OperatingSystem === "Windows" :=
	FileNameJoin[{WindowsDirectory[], "System32"}]
GPUTools`Internal`$NVIDIAOpenCLLibraryDirectory /; $OperatingSystem === "MacOSX" :=
	FileNameJoin[{"/System", "Library", "Frameworks", "OpenCL.framework"}]

GPUTools`Internal`$NVIDIAOpenCLLibraryName /; $OperatingSystem === "Unix" := "libOpenCL.so"
GPUTools`Internal`$NVIDIAOpenCLLibraryName /; $OperatingSystem === "Windows" := "OpenCL.dll"
GPUTools`Internal`$NVIDIAOpenCLLibraryName /; $OperatingSystem === "MacOSX" := "OpenCL"

GPUTools`Internal`$NVIDIAOpenCLLibraryPath = 
	If[GPUTools`Internal`$NVIDIAOpenCLLibraryDirectory === $Failed || GPUTools`Internal`$NVIDIAOpenCLLibraryName === $Failed,
		$Failed,
		FileNameJoin[{GPUTools`Internal`$NVIDIAOpenCLLibraryDirectory, GPUTools`Internal`$NVIDIAOpenCLLibraryName}]
	]

GPUTools`Internal`$ATISTREAMSDKROOT := GPUTools`Internal`$ATISTREAMSDKROOT =
	Module[{res},
		res = Quiet@Which[
			StringQ[Environment["AMDAPPSDKROOT"]],
				Environment["AMDAPPSDKROOT"],
			StringQ[Environment["ATISTREAMSDKROOT"]],
				Environment["ATISTREAMSDKROOT"],
			($OperatingSystem === "Windows") && DirectoryQ[FileNameJoin[{WindowsDriveLetter[], "Program Files" <> If[$SystemID === "Windows-x86-64", " (x86)", ""], "ATI Stream"}]],
				FileNameJoin[{WindowsDriveLetter[], "Program Files" <> If[$SystemID === "Windows-x86-64", " (x86)", ""], "ATI Stream"}],
			($OperatingSystem === "Windows") && DirectoryQ[FileNameJoin[{WindowsDriveLetter[], "Program Files" <> If[$SystemID === "Windows-x86-64", " (x86)", ""], "AMD APP"}]],
				FileNameJoin[{WindowsDriveLetter[], "Program Files" <> If[$SystemID === "Windows-x86-64", " (x86)", ""], "AMD APP"}],
			True,
				$Failed
		];
		GPUTools`Utilities`Logger["ATI Stream toolkit is located in ", res];
		res
	]
GPUTools`Internal`$ATIOpenCLLibraryDirectory /; $OperatingSystem === "Unix" :=
	With[{atisdkRoot = GPUTools`Internal`$ATISTREAMSDKROOT},
		If[StringQ[atisdkRoot],
			FileNameJoin[{atisdkRoot, "lib", If[$SystemID === "Linux", "x86", "x86_64"]}],
			$Failed
		]
	]
GPUTools`Internal`$ATIOpenCLLibraryDirectory /; $OperatingSystem === "Windows" :=
	If[GPUTools`Internal`$ATISTREAMSDKROOT === $Failed,
		$Failed,
		FileNameJoin[{GPUTools`Internal`$ATISTREAMSDKROOT, "bin", If[$SystemID === "Windows-x86-64", "x86_64", "x86"]}]
	]
GPUTools`Internal`$ATIOpenCLLibraryDirectory /; $OperatingSystem === "MacOSX" :=
	GPUTools`Internal`$NVIDIAOpenCLLibraryDirectory

selectOpenCLLibraryName[name_String] := String
selectOpenCLLibraryName[names_List, dir_String] :=
    Module[{libs = Select[names, TrueQ[Quiet[FileExistsQ[FileNameJoin[{dir, #}]]]]&]},
        If[libs === {},
            $Failed,
            First[libs]
        ]
    ]
selectOpenCLLibraryName[names_List, dirs_List] :=
	With[{libs = Select[selectOpenCLLibraryName[names, #]&/@dirs, #=!=$Failed&]},
		If[libs === {},
			$Failed,
			First[libs]
		]
	]
	
possibleATIPaths /; $OperatingSystem === "Windows" :=
	Flatten[{
		GPUTools`Internal`$ATIOpenCLLibraryDirectory,
		FileNameJoin[{WindowsDirectory[], "System32"}]
	}]
GPUTools`Internal`$ATIOpenCLLibraryName /; $SystemID === "Linux" := selectOpenCLLibraryName[{"libamdocl23.so", "libatiocl32.so"}, GPUTools`Internal`$ATIOpenCLLibraryDirectory]
GPUTools`Internal`$ATIOpenCLLibraryName /; $SystemID === "Linux-x86-64" := selectOpenCLLibraryName[{"libamdocl64.so", "libatiocl64.so"}, GPUTools`Internal`$ATIOpenCLLibraryDirectory]
GPUTools`Internal`$ATIOpenCLLibraryName /; $SystemID === "Windows" := selectOpenCLLibraryName[{"amdocl.dll", "atiocl.dll"}, possibleATIPaths]
GPUTools`Internal`$ATIOpenCLLibraryName /; $SystemID === "Windows-x86-64" := selectOpenCLLibraryName[{"amdocl64.dll", "atiocl64.dll"}, possibleATIPaths]
GPUTools`Internal`$ATIOpenCLLibraryName /; $OperatingSystem === "MacOSX" := GPUTools`Internal`$NVIDIAOpenCLLibraryName

GPUTools`Internal`$ATIOpenCLLibraryPath := 
	Module[{pth},
		If[GPUTools`Internal`$ATIOpenCLLibraryDirectory === $Failed || GPUTools`Internal`$ATIOpenCLLibraryName === $Failed,
			$Failed,
			pth = FileNameJoin[{GPUTools`Internal`$ATIOpenCLLibraryDirectory, GPUTools`Internal`$ATIOpenCLLibraryName}];
			If[TrueQ[Quiet[FileExistsQ[pth]]],
				pth,
				If[$OperatingSystem === "Windows",
					pth = FileNameJoin[{FileNameJoin[{WindowsDirectory[], "System32"}], GPUTools`Internal`$ATIOpenCLLibraryName}];
					If[TrueQ[Quiet[FileExistsQ[pth]]],
						pth,
						$Failed
					]
				]
			]
		]
	]

GPUTools`Internal`$ATIDriverPath /; $SystemID === "Windows" :=
	Which[
		FileExistsQ[FileNameJoin[{WindowsDirectory[], "System32", "atiadlxx.dll"}]],
			FileNameJoin[{WindowsDirectory[], "System32", "atiadlxx.dll"}],
		TrueQ[Quiet[StringMatchQ[GPUTools`Internal`VideoCardName[], ___ ~~ "FirePro" ~~ ___, IgnoreCase->True]]],
			FileNameJoin[{WindowsDirectory[], "System32", "atiu9p.dll"}],
		TrueQ[Quiet[StringMatchQ[GPUTools`Internal`VideoCardName[], ___ ~~ "Radeon" ~~ ___, IgnoreCase->True]]],
			FileNameJoin[{WindowsDirectory[], "System32", "atiu9p.dll"}],
		TrueQ[Quiet[StringMatchQ[GPUTools`Internal`VideoCardName[], ___ ~~ "FireStream" ~~ ___, IgnoreCase->True]]],
			FileNameJoin[{WindowsDirectory[], "System32", "ati2dvag.dll"}],
		True,
			FileNameJoin[{WindowsDirectory[], "System32", "atiu9p.dll"}]
	]
GPUTools`Internal`$ATIDriverPath /; $SystemID === "Windows-x86-64" :=
	Which[
		FileExistsQ[FileNameJoin[{WindowsDirectory[], "System32", "atiadlxx.dll"}]],
			FileNameJoin[{WindowsDirectory[], "System32", "atiadlxx.dll"}],
		TrueQ[Quiet[StringMatchQ[GPUTools`Internal`VideoCardName[], ___ ~~ "Radeon" ~~ ___, IgnoreCase->True]]],
			FileNameJoin[{WindowsDirectory[], "System32", "atiu9p64.dll"}],
		TrueQ[Quiet[StringMatchQ[GPUTools`Internal`VideoCardName[], ___ ~~ "FirePro" ~~ ___, IgnoreCase->True]]],
			FileNameJoin[{WindowsDirectory[], "System32", "atiu9p64.dll"}],
		TrueQ[Quiet[StringMatchQ[GPUTools`Internal`VideoCardName[], ___ ~~ "FireStream" ~~ ___, IgnoreCase->True]]],
			FileNameJoin[{WindowsDirectory[], "System32", "ati2dvag.dll"}],
		True,
			FileNameJoin[{WindowsDirectory[], "System32", "atiu9p.dll"}]
	]
GPUTools`Internal`$ATIDriverPath /; $SystemID === "Linux" :=
GPUTools`Internal`$ATIDriverPath /; $SystemID === "Linux-x86-64" :=
	Module[{files},
		files = Select[
			Table[
				FileNameJoin[{"/usr", "lib" <> arch, "libaticalcl.so"}],
				{arch, {"64", ""}}
			],
			FileType[#] === File&
		];
		If[files === {},
			$Failed,
			First[files]
		]
	]


GPUTools`Internal`$ATIDriverVersion /; $OperatingSystem === "Windows" :=
	If[FileExistsQ[GPUTools`Internal`$ATIDriverPath],
		LibraryVersionInformation[GPUTools`Internal`$ATIDriverPath],
		"0.0.0"
	]
	
GPUTools`Internal`$ATIDriverVersion /; $OperatingSystem === "Unix" :=
	0.0
	
GPUTools`Internal`$ATIDriverValidQ /; $OperatingSystem === "Unix" :=
	FileExistsQ[GPUTools`Internal`$ATIDriverPath]
GPUTools`Internal`$ATIDriverValidQ /; $OperatingSystem === "MacOSX" :=
	FileExistsQ[GPUTools`Internal`$ATIOpenCLLibraryPath]
GPUTools`Internal`$ATIDriverValidQ /; $OperatingSystem === "Windows" :=
	TrueQ@With[{ver = GPUTools`Internal`$ATIDriverVersion},
		If[StringQ[ver],
			Return[True]
		];
		If[ListQ[ver],
			TrueQ[Which[
				FileBaseName[GPUTools`Internal`$ATIDriverPath] === "atiadlxx",
					GPUTools`Utilities`LibraryGetMajorVersion[ver] >= 6 && GPUTools`Utilities`LibraryGetMinorVersion[ver] >= 14,
				TrueQ[Quiet[StringMatchQ[GPUTools`Internal`VideoCardName[], ___ ~~ "Radeon" ~~ ___, IgnoreCase->True]]],
					GPUTools`Utilities`LibraryGetMajorVersion[ver] >= 8 && GPUTools`Utilities`LibraryGetMinorVersion[ver] >= 10,
				TrueQ[Quiet[StringMatchQ[GPUTools`Internal`VideoCardName[], ___ ~~ "FirePro" ~~ ___, IgnoreCase->True]]],
					GPUTools`Utilities`LibraryGetMajorVersion[ver] >= 8 && GPUTools`Utilities`LibraryGetMinorVersion[ver] >= 10,
				TrueQ[Quiet[StringMatchQ[GPUTools`Internal`VideoCardName[], ___ ~~ "FireStream" ~~ ___, IgnoreCase->True]]],
					GPUTools`Utilities`LibraryGetMajorVersion[ver] >= 10 && GPUTools`Utilities`LibraryGetMinorVersion[ver] >= 2
			]],
			False
		]
	]
	
DeviceNamesInPCIInformation[] := DeviceNamesInPCIInformation[] =
	Select[Flatten[{("VideoCardPCIInformation" /. SystemInformation["Devices", "GraphicsDevices"])}], 
  		Head[#] === Rule && StringQ[First[#]] && StringMatchQ[First[#], "Device" ~~ (DigitCharacter ..)] &
  	] /. (HoldPattern[Rule[_, x_]] -> x)

GPUTools`Internal`UsingNVIDIAQ := GPUTools`Internal`UsingNVIDIAQ =
	If[Environment["OpenCL_USING_NVIDIAQ"] =!= $Failed,
		True,
		TrueQ@If[GPUTools`Internal`UsingATIQ,
			False,
			Quiet[Check[
				RegisteryQueryValidQ["nvcuda"] ||
				TrueQ[Quiet[FileExistsQ[GPUTools`Internal`$NVIDIADriverLibraryPath]]] ||
				Apply[Or, TrueQ[Quiet[StringMatchQ[#, ___ ~~ "NVIDIA" ~~ ___, IgnoreCase->True]]]& /@ DeviceNamesInPCIInformation[]] ||
				TrueQ[Quiet[StringMatchQ[GPUTools`Internal`VideoCardName[], ___ ~~ "NVIDIA" ~~ ___, IgnoreCase->True]]]
				, False
			]]
		]
	]

RegisteryQuery[] /; $OperatingSystem === "MacOSX" := RegisteryQuery[] = (* check if using OSX snow leopard *)
	With[{cmdout = Import["!sysctl kern.osrelease", "Text"]},
		If[StringTrim[cmdout] === "",
			$Failed,
			With[{
				  parsed = StringReplace[
				  	cmdout,
				  	"kern.osrelease: " ~~ (x : DigitCharacter ..) ~~ "." ~~ (y : DigitCharacter ..) ~~ ___ -> {x, y}
				  ]
				 },
				 If[Head[parsed] =!= StringExpression || Length[First[parsed]] != 2,
				 	$Failed,
				 	ToExpression[First[parsed][[1]] <> "." <> First[parsed][[2]]]
				 ]
			]
		]
	]
RegisteryQuery[] /; $OperatingSystem === "Unix" := RegisteryQuery[] =
	FileNames[FileNameJoin[{"/etc", "OpenCL", "vendors", "*"}]]
RegisteryQuery[] /; $OperatingSystem === "Windows" := RegisteryQuery[] = 
	Developer`ReadRegistryKeyValues["HKEY_LOCAL_MACHINE\\SOFTWARE\\Khronos\\OpenCL\\Vendors"]

RegisteryQueryValidQ[_] /; $OperatingSystem === "MacOSX" :=
	With[{reg = Quiet[RegisteryQuery[]]},
		NumericQ[reg] && reg >= 10.0			(* idealy we should be checking for 10.3 *)
	]
RegisteryQueryValidQ[vendor_] /; $OperatingSystem === "Unix" :=
	Apply[Or, StringMatchQ[#, ___ ~~ vendor ~~ ___]& /@ RegisteryQuery[]]
RegisteryQueryValidQ[vendor_] /; $OperatingSystem === "Windows" :=
	IntegerQ[vendor <> ".dll" /. RegisteryQuery[]]

GPUTools`Internal`UsingATIQ := GPUTools`Internal`UsingATIQ =
	If[Environment["OpenCL_USING_AMDQ"] =!= $Failed,
		True,
		Quiet[Check[
			(GPUTools`Internal`$ATISTREAMSDKROOT =!= $Failed &&
			 (RegisteryQueryValidQ["atiocl"] || RegisteryQueryValidQ["amdocl"] || RegisteryQueryValidQ["amdocl64"]) &&
			 DirectoryQ[GPUTools`Internal`$ATIOpenCLLibraryDirectory] &&
			 TrueQ[Quiet[FileExistsQ[GPUTools`Internal`$ATIOpenCLLibraryPath]]])
			, False
		]]
	]

GPUTools`Internal`$OpenCLLibraryDirectory :=
	If[Environment["OpenCL_LIBRARY_DIRECTORY"] =!= $Failed,
		Environment["OpenCL_LIBRARY_DIRECTORY"],
		If[GPUTools`Internal`UsingNVIDIAQ,
			GPUTools`Internal`$NVIDIAOpenCLLibraryDirectory,
			GPUTools`Internal`$ATIOpenCLLibraryDirectory
		]
	]

GPUTools`Internal`$OpenCLLibraryPath :=
	If[Environment["OpenCL_LIBRARY_LOCATION"] =!= $Failed,
		Environment["OpenCL_LIBRARY_LOCATION"],
		If[GPUTools`Internal`UsingNVIDIAQ,
			GPUTools`Internal`$NVIDIAOpenCLLibraryPath,
			GPUTools`Internal`$ATIOpenCLLibraryPath
		]
	]

GPUTools`Internal`$OpenCLLibraryName :=
	If[Environment["OpenCL_LIBRARY_NAME"] =!= $Failed,
		Environment["OpenCL_LIBRARY_NAME"],
		If[GPUTools`Internal`UsingNVIDIAQ,
			GPUTools`Internal`$NVIDIAOpenCLLibraryName,
			GPUTools`Internal`$ATIOpenCLLibraryName
		]
	]

GPUTools`Internal`LoadOpenCLLibraries[errHd_] := GPUTools`Internal`LoadOpenCLLibraries[errHd] =
	If[GPUTools`Utilities`IsSandboxedQ[errHd],
		$Failed,
		If[GPUTools`Internal`UsingNVIDIAQ || GPUTools`Internal`UsingATIQ,
			If[TrueQ[Quiet[DirectoryQ[GPUTools`Internal`$OpenCLLibraryDirectory]]] &&
	           TrueQ[Quiet[FileExistsQ[GPUTools`Internal`$OpenCLLibraryPath]]],
	            Catch[
	                Quiet[
	                    Switch[$SystemID,
	                        "Linux" | "Linux-x86-64",
	                            GPUTools`Internal`LibrariesSafeLoad /@ {GPUTools`Internal`$OpenCLLibraryPath, FileNameJoin[{GPUTools`Internal`$OpenCLLibraryDirectory, "libOpenCL.so"}]},
	                        _,
	                            True
	                    ]
	                ]
	            ],
				If[StringQ[GPUTools`Internal`$OpenCLLibraryName] && StringQ[GPUTools`Internal`$OpenCLLibraryDirectory],
					Message[errHd::invocllibp, GPUTools`Internal`$OpenCLLibraryName, GPUTools`Internal`$OpenCLLibraryDirectory],
					Message[errHd::invocllib]
				];
				$Failed
			],
			Message[errHd::invdrv];
			$Failed
		]
	]
	
(* ::Section:: *)
(* CUDA Hardware Tools *)

GPUTools`Internal`VideoCardInformation[prop_String] := GPUTools`Internal`VideoCardInformation[prop] =
	With[{sysinfo = GPUTools`Internal`VideoCardInformation[]},
		If[sysinfo === "",
			"",
			Replace[prop, sysinfo]
		]
	]

GPUTools`Internal`VideoCardInformation[] /; $OperatingSystem === "Windows" := GPUTools`Internal`VideoCardInformation[] =
	With[{sysinfo = Quiet[SystemInformation["Devices", "GraphicsDevices"]]},
		If[sysinfo === $Failed,
			"",
			Replace["DirectX",
 				sysinfo
			]
		]
 	]
GPUTools`Internal`VideoCardInformation[] /; $OperatingSystem === "Unix" := GPUTools`Internal`VideoCardInformation[] =
	With[{sysinfo = Quiet[SystemInformation["Devices", "GraphicsDevices"]]},
		If[sysinfo === $Failed,
			If[StringQ[DeviceNamesInPCIInformation[]],
				DeviceNamesInPCIInformation[],
				""
			],
			Replace["OnScreen", 
   				Replace["OpenGL", 
					sysinfo
				]
			]
		]
	]
GPUTools`Internal`VideoCardInformation[] /; $OperatingSystem === "MacOSX" := GPUTools`Internal`VideoCardInformation[] =
	With[{sysinfo = Quiet[SystemInformation["Devices", "GraphicsDevices"]]},
		If[sysinfo === $Failed,
			"",
			Replace["OnScreen",
				Replace["OpenGL",
					sysinfo
				]
			]
		]
	]

GPUTools`Internal`ValidSystemQ[] :=
	MemberQ[{"Windows", "Windows-x86-64", "Linux", "Linux-x86-64", "MacOSX-x86", "MacOSX-x86-64"}, $SystemID]

GPUTools`Internal`$NVIDIADriverLibraryPath := GPUTools`Internal`$NVIDIADriverLibraryPath =
	Module[{res},
		res = If[Environment["NVIDIA_DRIVER_LIBRARY_PATH"] =!= $Failed,
			Environment["NVIDIA_DRIVER_LIBRARY_PATH"],
			Switch[$SystemID,
				"Windows",
					FileNameJoin[{WindowsDirectory[], "System32", "nvapi.dll"}],
				"Windows-x86-64",
					FileNameJoin[{WindowsDirectory[], "System32", "nvapi64.dll"}],
				"Linux",
					With[{libs = FileNames[FileNameJoin[{"/usr", "lib", "libnvidia-tls.so.*"}]]},
						If[libs =!= {},
							First[libs],
							$Failed
						]
					],
				"Linux-x86-64", 
					Module[{libs},
						libs = Flatten[
							Table[
								FileNames[FileNameJoin[{"/usr", "lib" <> arch, "*", "libnvidia-tls.so.*"}]],
								{arch, {"64", ""}}
							]
						];
						If[libs =!= {},
							First[libs],
							$Failed
						]
					],
				"MacOSX-x86" | "MacOSX-x86-64",
					FileNameJoin[{"/Library", "Frameworks", "CUDA.framework", "Versions", "Current", "CUDA"}],
				_,
					$Failed
			]
		];
		GPUTools`Utilities`Logger["NVIDIA driver library is located in ", res];
		res
	]

GPUTools`Internal`$CUDALibraryPath :=
	If[Environment["CUDA_LIBRARY_PATH"] =!= $Failed,
		Environment["CUDA_LIBRARY_PATH"],
		Switch[$SystemID,
			"Windows" | "Windows-x86-64",
				FileNameJoin[{WindowsDirectory[], "System32", "nvcuda.dll"}],
			"Linux",
				FileNameJoin[{"/usr", "lib", "libcuda.so"}],
			"Linux-x86-64", 
				Module[{libs},
					libs = Quiet@Flatten[Table[
					FileNames["libcuda.so",
						dir,
						Infinity
					],
					{	dir,
						{
							FileNameJoin[{"/usr", "lib64"}], 
							FileNameJoin[{"/usr", "lib", "x86_64-linux-gnu"}], 
							FileNameJoin[{Lookup[PacletManager`PacletInformation["CUDAResources"],"Location",""], "CUDAToolkit", "lib64"}]
						}
					}
					]
					];
					If[libs === {},
						$Failed,
						First[libs]
					]
				],
			"MacOSX-x86" | "MacOSX-x86-64",
				FileNameJoin[{"/usr", "local", "cuda", "lib", "libcuda.dylib"}],
			_,
				$Failed
		]
	]
	
stringContainsQ[a_String] := TrueQ[StringMatchQ[#, ___ ~~ a ~~ ___]]&; 	
stringContainsQ[str_, a_String] := stringContainsQ[a][str]; 	

gpuNames[] /; $OperatingSystem === "MacOSX" :=
	Quiet[
		Module[{run, gpus},
			run = RunProcess[{"ioreg", "-p", "IOService", "-c", "IOPCIDevice", "-k", "NVCLASS", "-r", "-f", "-d", "1"}];
			If[run["ExitCode"] =!= 0,
				False,
				gpus = StringCases[run["StandardOutput"], "\"model\" = <\"" ~~ Shortest[nv__] ~~ "\">" -> nv];
				Select[gpus, stringContainsQ["NV"]]
			]
		]
	]
isLegacyOSXGPUs[] /; $OperatingSystem === "MacOSX" :=
	isLegacyOSXGPUs[gpuNames[]]
isLegacyOSXGPUs[gpus_?ListQ] /; $OperatingSystem === "MacOSX" :=
	AllTrue[gpus, isLegacyOSXGPUs]
isLegacyOSXGPUs[gpu_String] /; $OperatingSystem === "MacOSX" :=
	AnyTrue[
		{"9400M", "9600M", "330M"},
		TrueQ[stringContainsQ[gpu, #]]&
	]
	
GPUTools`Internal`$IsLegacyGPU :=
	Switch[$OperatingSystem,
		"MacOSX",
			isLegacyOSXGPUs[],
		_,
			False
	]

GPUTools`Internal`$NVIDIADriverLibraryValidQ :=
	If[Environment["NVIDIA_DRIVER_LIBRARY_VALIDQ"] =!= $Failed,
		True,
		iNVIDIADriverLibraryValidQ
	]

iNVIDIADriverLibraryValidQ /; $OperatingSystem === "MacOSX" :=
	TrueQ@Module[{version = GPUTools`Internal`$NVIDIADriverLibraryVersion, res},
		If[version === $Failed,
			GPUTools`Utilities`Logger["Cannot determine NVIDIA driver version."];
			False,
			res = NumberQ[GPUTools`Utilities`LibraryGetMajorVersion[version]] &&
				  GPUTools`Utilities`LibraryGetMajorVersion[version] >= 3;
			If[res,
				True,
				GPUTools`Utilities`Logger["NVIDIA driver version is not compatible with Mathematica. Version=", version]
			]
		]
	]

iNVIDIADriverLibraryValidQ /; $OperatingSystem === "Windows" :=
	TrueQ@Module[{version = GPUTools`Internal`$NVIDIADriverLibraryVersion, res},
		res = If[version === $Failed,
			GPUTools`Utilities`Logger["Cannot determine NVIDIA driver version."];
			False,
			res = NumberQ[GPUTools`Utilities`LibraryGetMajorVersion[version]] &&
				  GPUTools`Utilities`LibraryGetRevisionVersion[version] >= 12.0;
			If[!res,
				GPUTools`Utilities`Logger["NVIDIA driver version is not compatible with Mathematica. Version=", version];
			];
			res
		];
		res
	]
iNVIDIADriverLibraryValidQ /; $OperatingSystem === "Unix" :=
	TrueQ@Module[{version = GPUTools`Internal`$NVIDIADriverLibraryVersion, res},
		res = If[version === $Failed,
			GPUTools`Utilities`Logger["Cannot determine NVIDIA driver version."];
			False,
			res = NumberQ[GPUTools`Utilities`LibraryGetMajorVersion[version]] &&
				  GPUTools`Utilities`LibraryGetMajorVersion[version] >= 190;
			If[!res,
				GPUTools`Utilities`Logger["NVIDIA driver version is not compatible with Mathematica. Version=", version]
			];
			res
		];
		res
	]

GPUTools`Internal`$NVIDIADriverLibraryVersion := GPUTools`Internal`$NVIDIADriverLibraryVersion =
	With[{pth = GPUTools`Internal`$NVIDIADriverLibraryPath},
		If[pth =!= $Failed && FileExistsQ[pth],
			Quiet[LibraryVersionInformation[pth]],
			$Failed
		]
	]

GPUTools`Internal`$CUDALibraryValidQ :=
	If[Environment["CUDA_LIBRARY_VALIDQ"] =!= $Failed,
		True,
		iCUDALibraryValidQ
	]

iCUDALibraryValidQ /; $OperatingSystem === "Windows" :=
	TrueQ@Module[{version = GPUTools`Internal`$CUDALibraryVersion, res},
		res = If[version === $Failed,
			GPUTools`Utilities`Logger["Cannot determine CUDA library version."];
			False,
			res = NumberQ[GPUTools`Utilities`LibraryGetMajorVersion[version]] && GPUTools`Utilities`LibraryGetRevisionVersion[version] >= 12.0;
			If[!res,
				GPUTools`Utilities`Logger["CUDA library version is not compatible with Mathematica. Version=", version]
			];
			res
		];
		res
	]
	
iCUDALibraryValidQ /; $OperatingSystem === "Unix" :=
	TrueQ@Module[{version = GPUTools`Internal`$CUDALibraryVersion, res},
		res = If[version === $Failed,
			GPUTools`Utilities`Logger["Cannot determine CUDA library version."];
			False,
			res = NumberQ[GPUTools`Utilities`LibraryGetMajorVersion[version]] && GPUTools`Utilities`LibraryGetMajorVersion[version] >= 190;
			If[!res,
				GPUTools`Utilities`Logger["CUDA library version is not compatible with Mathematica. Version=", version]
			];
			res
		];
		res
	]

iCUDALibraryValidQ /; $OperatingSystem === "MacOSX" :=
	TrueQ@Module[{version = GPUTools`Internal`$NVIDIADriverLibraryVersion, res},
		res = If[version === $Failed,
			GPUTools`Utilities`Logger["Cannot determine CUDA library version."];
			False,
			res = NumberQ[GPUTools`Utilities`LibraryGetMajorVersion[version]] && GPUTools`Utilities`LibraryGetMajorVersion[version] >= 3;
			If[!res,
				GPUTools`Utilities`Logger["CUDA library version is not compatible with Mathematica. Version=", version]
			];
			res
		];
		res
	]

GPUTools`Internal`$CUDALibraryVersion := GPUTools`Internal`$CUDALibraryVersion = 
	With[{pth = GPUTools`Internal`$CUDALibraryPath},
		If[pth =!= $Failed && FileExistsQ[pth],
			Quiet[LibraryVersionInformation[pth]],
			$Failed
		]
	]

GPUTools`Internal`$GPUToolsPath := DirectoryName[System`Private`$InputFileName]
GPUTools`Internal`$GPUToolsSystemResourcesPath := 
	If[Environment["CUDALINK_RUNTIME_LIBRARY_PATH"] === $Failed,
		FileNameJoin[{GPUTools`Internal`$GPUToolsPath, "SystemFiles", $SystemID}],
		Environment["CUDALINK_RUNTIME_LIBRARY_PATH"]
	]
GPUTools`Internal`$GPUToolsLibraryResourcesPath := FileNameJoin[{GPUTools`Internal`$GPUToolsPath, "LibraryResources", $SystemID}]



GPUTools`Internal`FindCUDALinkSystemLibaries[pth_, sys:("Linux-x86-64" | "Linux")] := 
	GPUTools`Internal`FindCUDALinkSystemLibaries[pth, sys] =
    GPUTools`Internal`$CUDALinkSystemLibaries =
	Flatten[{
		GPUTools`Internal`$CUDALibraryPath,
		FileNames[FileNameJoin[{pth, "libcudart*so*"}]],
		FileNames[FileNameJoin[{pth, "libcufft*so*"}]],
		FileNames[FileNameJoin[{pth, "libcublas*so*"}]],
		FileNames[FileNameJoin[{pth, "libcurand*so*"}]]
	}]
	
GPUTools`Internal`FindCUDALinkSystemLibaries[pth_, "Windows-x86-64"] :=
	GPUTools`Internal`FindCUDALinkSystemLibaries[pth, "Windows-x86-64"] = 
    GPUTools`Internal`$CUDALinkSystemLibaries =
	Flatten[{
		GPUTools`Internal`$CUDALibraryPath,
		FileNames[FileNameJoin[{pth, "cudart*64*dll"}]],
		FileNames[FileNameJoin[{pth, "cufft*64*dll"}]],
		FileNames[FileNameJoin[{pth, "cublasLt*64*dll"}]],
		FileNames[FileNameJoin[{pth, "cublas*64*dll"}]],
		FileNames[FileNameJoin[{pth, "curand*64*dll"}]]
	}]

GPUTools`Internal`FindCUDALinkSystemLibaries[pth_, "Windows"] :=
	GPUTools`Internal`FindCUDALinkSystemLibaries[pth, "Windows"] =
    GPUTools`Internal`$CUDALinkSystemLibaries =
	Flatten[{
		GPUTools`Internal`$CUDALibraryPath,
		FileNames[FileNameJoin[{pth, "cudart*32*dll"}]],
		FileNames[FileNameJoin[{pth, "cufft*32*dll"}]],
		FileNames[FileNameJoin[{pth, "cublas*32*dll"}]],
		FileNames[FileNameJoin[{pth, "curand*32*dll"}]]
	}]
	
GPUTools`Internal`FindCUDALinkSystemLibaries[pth_, "MacOSX-x86-64" | "MacOSX-x86"] := 
	GPUTools`Internal`FindCUDALinkSystemLibaries[pth, "MacOSX-x86"] =
	GPUTools`Internal`FindCUDALinkSystemLibaries[pth, "MacOSX-x86-64"] =
    GPUTools`Internal`$CUDALinkSystemLibaries =
	Flatten[{
		GPUTools`Internal`$CUDALibraryPath,
		FileNames[FileNameJoin[{pth, "libtlshook*dylib"}]],
		FileNames[FileNameJoin[{pth, "libcuda*dylib"}]],
		FileNames[FileNameJoin[{pth, "libcufft*dylib"}]],
		FileNames[FileNameJoin[{pth, "libcublas*dylib"}]],
		FileNames[FileNameJoin[{pth, "libcurand*dylib"}]]
	}]

GPUTools`Internal`FindCUDALinkSystemLibaries[_] := 
    GPUTools`Internal`$CUDALinkSystemLibaries = None


GPUTools`Internal`LoadCUDALibraries[errHd_, pth_] := GPUTools`Internal`LoadCUDALibraries[errHd, pth] =
	Which[
		GPUTools`Utilities`IsSandboxedQ[errHd],
			$Failed,
	    GPUTools`Internal`$IsLegacyGPU,
			$Failed,
	    True,
			Module[{libs = GPUTools`Internal`FindCUDALinkSystemLibaries[pth, $SystemID]},
	        	GPUTools`Utilities`Logger["Loading CUDA Library Files: ", StringJoin@Riffle[libs, ", "]];
				If[libs === $Failed || Length[libs] < 4,
					$Failed,
					GPUTools`Utilities`Logger["NVIDIA Driver Library is ", If[GPUTools`Internal`$NVIDIADriverLibraryValidQ, "", "NOT "], "Valid"];
					GPUTools`Utilities`Logger["CUDA Library is ", If[GPUTools`Internal`$CUDALibraryValidQ, "", "NOT "], "Valid"];
					If[TrueQ[GPUTools`Internal`$NVIDIADriverLibraryValidQ] && TrueQ[GPUTools`Internal`$CUDALibraryValidQ],
						GPUTools`Internal`LibrariesSafeLoad[libs],
						$Failed
					]
				]
			]
	]

GPUTools`Internal`VideoCardName[] := GPUTools`Internal`VideoCardName[] =
    Switch[$OperatingSystem,
        "Unix",
               GPUTools`Internal`VideoCardInformation["Vendor"] <> "  " <> GPUTools`Internal`VideoCardInformation["Version"],
        "Windows",
               GPUTools`Internal`VideoCardInformation["Description"],
        "MacOSX",
               GPUTools`Internal`VideoCardInformation["Renderer"],
        _,
            $Failed
    ]


(* ::Section:: *)
(* Failure Reason *)


GPUTools`Internal`OpenCLQueryFailReason[errHd_, querySystemInformationQ_:True] :=
	Which[
		GPUTools`Utilities`IsSandboxedQ[errHd],
			Throw[$Failed],
		GPUTools`Internal`UsingATIQ,
			(* using an ATI card *)
			Which[
				!TrueQ[GPUTools`Internal`ValidSystemQ[]],
					Throw[Message[errHd::invsys, $SystemID]; $Failed],
				GPUTools`Internal`$ATIDriverPath === $Failed || !TrueQ[Quiet[DirectoryQ[DirectoryName[GPUTools`Internal`$ATIDriverPath]]]],
					Throw[Message[errHd::invdriv]; $Failed],
				TrueQ[Quiet[DirectoryQ[GPUTools`Internal`$ATIDriverPath]]],
					Throw[Message[errHd::invdrivp, DirectoryName[GPUTools`Internal`$ATIDriverPath]]; $Failed],
				!GPUTools`Internal`$ATIDriverValidQ,
					If[GPUTools`Internal`$ATIDriverVersion === $Failed,
						Message[errHd::invdrivver],
						If[TrueQ[Quiet[GPUTools`Utilities`LibraryGetDirectory[GPUTools`Internal`$ATIDriverVersion] =!= $Failed &&
						   DirectoryQ[GPUTools`Utilities`LibraryGetDirectory[GPUTools`Internal`$ATIDriverVersion]]]],
							Message[errHd::invdrivverv, GPUTools`Utilities`LibraryVersionInformationString[GPUTools`Internal`$ATIDriverVersion]],
							If[TrueQ[Quiet[DirectoryQ[GPUTools`Utilities`LibraryGetDirectory[GPUTools`Internal`$ATIDriverVersion]]]],
								Message[errHd::invdrivverd, GPUTools`Utilities`LibraryGetDirectory[GPUTools`Internal`$ATIDriverVersion]],
								Message[errHd::invdrivp, GPUTools`Internal`$ATIDriverPath]
							]
						]
					];
					Throw[$Failed],
				GPUTools`Internal`LoadOpenCLLibraries[] === $Failed,
					If[$MessageList === {},
						Message[errHd::syslibfld]
					];
					Throw[$Failed],
				True,
					True
			],
		True,
			If[querySystemInformationQ && TrueQ[Quiet[StringMatchQ[GPUTools`Internal`VideoCardName[], ___ ~~ "NVIDIA" ~~ ___, IgnoreCase->True]]],
				If[GPUTools`Internal`CUDAQueryFailReason[errHd] === True,
					Throw[Message[errHd::gpures]; $Failed]
				],
				If[querySystemInformationQ,
					Throw[Message[errHd::invdevnm, GPUTools`Internal`VideoCardName[]]; $Failed],
					Throw[Message[errHd::lnchunk]; $Failed]
				]
			]
	]
	
GPUTools`Internal`CUDAQueryFailReason[errHd_, querySystemInformationQ_:True] :=
	Which[
		GPUTools`Utilities`IsSandboxedQ[errHd],
			Throw[$Failed],
		!TrueQ[GPUTools`Internal`ValidSystemQ[]],
			Throw[Message[errHd::invsys, $SystemID]; $Failed],
		GPUTools`Internal`$IsLegacyGPU === True,
			Throw[Message[errHd::legcy, gpuNames[]]; $Failed],
		querySystemInformationQ && !TrueQ[Quiet[StringMatchQ[GPUTools`Internal`VideoCardName[], ___ ~~ "NVIDIA" ~~ ___, IgnoreCase->True]]] && 
		GPUTools`Internal`VideoCardInformation["Description"] =!= "Description",
			Throw[Message[errHd::invdevnm, GPUTools`Internal`VideoCardName[]]; $Failed],
		GPUTools`Internal`$NVIDIADriverLibraryPath === $Failed || !TrueQ[Quiet[DirectoryQ[DirectoryName[GPUTools`Internal`$NVIDIADriverLibraryPath]]]],
			Throw[Message[errHd::invdriv]; $Failed],
		TrueQ[Quiet[DirectoryQ[GPUTools`Internal`$NVIDIADriverLibraryPath]]],
			Message[errHd::invdrivp, DirectoryName[GPUTools`Internal`$NVIDIADriverLibraryPath]];
			Return[$Failed],
		!TrueQ[GPUTools`Internal`$NVIDIADriverLibraryValidQ],
			If[GPUTools`Internal`$NVIDIADriverLibraryVersion === $Failed,
				Message[errHd::invdrivver],
				If[GPUTools`Utilities`LibraryGetDirectory[GPUTools`Internal`$NVIDIADriverLibraryVersion] =!= $Failed &&
				   TrueQ[Quiet[DirectoryQ[GPUTools`Utilities`LibraryGetDirectory[GPUTools`Internal`$NVIDIADriverLibraryVersion]]]],
					Message[errHd::invdrivverv, GPUTools`Utilities`LibraryVersionInformationString[GPUTools`Internal`$NVIDIADriverLibraryVersion]],
					Message[errHd::invdrivverd, GPUTools`Utilities`LibraryGetDirectory[GPUTools`Internal`$NVIDIADriverLibraryVersion]]
				]
			];
			Throw[$Failed],
		GPUTools`Internal`LoadCUDALibraries[] === $Failed,
			Throw[Message[errHd::syslibfld]; $Failed],
		True,
			True
	]
	

End[]

EndPackage[]

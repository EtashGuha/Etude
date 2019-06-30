BeginPackage["CCompilerDriver`CCompilerDriverRegistry`"]

CCompilerRegister::usage = "CCompilerRegister[ name, targetSystemIDs] registers a C compiler with the system."

RegisteredCCompilerQ::usage = "RegisteredCCompilerQ[compiler] returns True if compiler has been registered with CCompilerRegister."

$CCompilers::usage = "$CCompilers is the list of all driver configurations that were detected automatically"
$CCompilersByDriver::usage = "$CCompilersbyDriver[driver] is the list of all driver configurations that driver can handle"

Begin["`Private`"] (* Begin Private Context *) 

Needs["CCompilerDriver`System`"]

If[!ListQ[$CCompilerList],
	$CCompilers = {};
];

CCompilerRegister[driver_] :=
	If[!Developer`$ProtectedMode && !TrueQ[RegisteredCCompilerQ[driver]],
		Module[{configs = FindInstallations[driver]},
			If[Head[configs] =!= List || configs === {},
				(* no installations found.  create a list with a single config *)
				configs = {{
					"Name" -> driver["Name"][],
					"Compiler" -> driver,
					"CompilerInstallation" -> None,
					"CompilerName" -> Automatic
				}};
			];
	
			Scan[CCompilerRegister, configs];
			RegisteredCCompilerQ[driver] = True;
		]
	]

FindInstallations[driver_] := 
	Module[{allInstallations, validInstallations},
		allInstallations = driver["Installations"][];
		validInstallations = Select[allInstallations, 
			(driver["ValidInstallationQ"][#])&];
		Table[
			{
				"Name" -> driver["Name"][],
				"Compiler" -> driver,
				"CompilerInstallation" -> installation,
				"CompilerName" -> Automatic
			},
			{installation, validInstallations}
		]
	]

CCompilerRegister[conf:{"Name" -> name_, "Compiler" -> driver_, 
	"CompilerInstallation" -> installation_, "CompilerName" -> _, ___}] := 
	(
		AppendTo[$CCompilers, conf];

		If[Head[$CCompilersByDriver[driver]] =!= List,
			$CCompilersByDriver[driver] = {}
		];
		AppendTo[$CCompilersByDriver[driver], conf];
	)

RegisteredCCompilerQ[{___, "Compiler" -> driver_, ___}] := 
	RegisteredCCompilerQ[driver]

RegisteredCCompilerQ[_] := False

Needs["CCompilerDriver`CCompilerDriverBase`"]
If[$OperatingSystem === "Windows",
	Needs["CCompilerDriver`VisualStudioCompiler`"];
]
If[$OperatingSystem === "MacOSX",
	Needs["CCompilerDriver`ClangCompiler`"];
]
If[$OperatingSystem =!= "Windows",
	Needs["CCompilerDriver`GCCCompiler`"];
]
If[$SystemID === "Windows",
	Needs["CCompilerDriver`MinGWCompiler`"];
	Needs["CCompilerDriver`CygwinGCC`"];
]
Needs["CCompilerDriver`IntelCompiler`"]
Needs["CCompilerDriver`GenericCCompiler`"]

End[] (* End Private Context *)

EndPackage[]

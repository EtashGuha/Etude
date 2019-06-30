BeginPackage["CCompilerDriver`System`"];
$PlatformSystemIDs::usage = "$PlatformSystemIDs is the list of values for $SystemID that correspond to the $OperatingSystem"

Begin["`Private`"];

$PlatformSystemIDs = 
	Switch[$OperatingSystem,
		"Windows", {"Windows", "Windows-x86-64"},
		"Unix", {"Linux", "Linux-x86-64", "Linux-ARM"},
		"MacOSX", {"MacOSX-x86-64"}
	]

End[];

EndPackage[];

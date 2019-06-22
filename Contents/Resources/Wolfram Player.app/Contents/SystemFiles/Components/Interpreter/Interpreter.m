(* 
	This file loaded by Get/Needs["Interpreter`"]. 
	It must load the package files and also ensure that Interpreter` context is on $ContextPath, which is not done by InterpreterLoader.
*)

BeginPackage["Interpreter`"]
EndPackage[]

(* 
	All loading of the paclet's Wolfram Language code should go through this file. 
	Developer maintains this list of symbols.
	autoloadSymbols must agree with the symbols listed in the Kernel extension in the PacletInfo.m file.
*)

Interpreter`Private`autoloadSymbols = {
	"System`$InterpreterTypes",
	"System`AnySubset",
	"System`CompoundElement",
	"System`DelimitedSequence", 
	"System`GeoLocation", 
	"System`Interpreter", 
	"System`RepeatingElement",
	"System`Restricted", 
	"Interpreter`InterpreterObject"
};

Map[
	(Unprotect[#];ClearAll[#]) &, Join[
		Interpreter`Private`autoloadSymbols, {
			"Interpreter`*",
			"Interpreter`PackageScope`*",
			"Interpreter`*`PackagePrivate`*"
		}
	]
];

PacletManager`Package`loadWolframLanguageCode[
	"Interpreter", 
	"Interpreter`", 
	DirectoryName[$InputFileName], 
	"Patterns.m",
	"AutoUpdate" -> True, 
	"ForceMX" -> False, 
	"Lock" -> False,
	"AutoloadSymbols"  -> Interpreter`Private`autoloadSymbols, 
	"SymbolsToProtect" -> Automatic,
	"HiddenImports"    :> If[
		$OperatingSystem =!= "iOS", { 
			"Security`", 
			"URLUtilities`", 
			"JLink`"
		}, {
			"Security`",
			"URLUtilities`"
		}
	]
]

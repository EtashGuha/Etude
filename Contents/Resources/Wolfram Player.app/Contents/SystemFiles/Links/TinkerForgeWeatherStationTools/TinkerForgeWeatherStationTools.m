(* Mathematica Package *)

(* Created by the Wolfram Workbench Nov 18, 2013 *)

BeginPackage["TinkerForgeWeatherStationTools`"]
(* Exported symbols added here with SymbolName::usage *) 

TinkerForgeWeatherStationTools`InstallMathLinkEXE::usage = "Installs the mathlink proxy for the TinkerForge WeatherStation.  Returns the LinkObject."

Begin["`Private`"]
(* Implementation of the package *)

$packageFile = $InputFileName;

$exeName = "TinkerForgeWeatherStationTools.exe";

$mathLinkExe = FileNameJoin[{FileNameTake[$packageFile, {1,-2}], "Binaries", $SystemID, $exeName}];

TinkerForgeWeatherStationTools`InstallMathLinkEXE[]:= Module[{ link},
	link = Install[$mathLinkExe];
	Return[ link];
]


End[]

EndPackage[]


(* ::Package:: *)

(* Mathematica Package *)

(* Created by the Wolfram Workbench Jan 9, 2014 *)

BeginPackage["UUID`"]

UUID::usage = "UUID[] generate a random UUID."

ResetUUIDGenerator::usage = "ResetUUIDGenerator[] resets a the random number generator for generating UUIDs."

Begin["`Private`"]

$LibraryResourcePath = 
	FileNameJoin[{
		DirectoryName[$InputFileName],
		"LibraryResources",
		$SystemID
	}];
	
If[FreeQ[$LibraryPath, $LibraryResourcePath], 
	PrependTo[$LibraryPath, $LibraryResourcePath]
];

If[$OperatingSystem === "iOS", (* on iOS, hyphens are not allowed in library names, because they're not valid C99 identifiers *)
	$UUIDLibrary=FindLibrary["uuid_link"],
	$UUIDLibrary=FindLibrary["uuid-link"]
];


initializeQ[] := initializeQ[] = (
	generateUUID = Quiet@LibraryFunctionLoad[$UUIDLibrary, "generateUUID", {}, "UTF8String"];
	restartRandom = Quiet@LibraryFunctionLoad[$UUIDLibrary, "restartRandom", {}, "Void"];
	generateUUID =!= $Failed && restartRandom =!= $Failed
)

UUID[] /; initializeQ[] := generateUUID[] 

ResetUUIDGenerator[] /; initializeQ[] := restartRandom[]

End[]

EndPackage[]

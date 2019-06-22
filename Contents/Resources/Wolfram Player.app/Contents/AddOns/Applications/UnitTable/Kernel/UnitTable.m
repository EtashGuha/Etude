(* Wolfram Language package *)

Begin["QuantityUnits`Private`"];

(*utility function used to load critical files; if anything goes awry it issues a message and ends the current context*)
getOrFail[file_String] := With[{
	r = If[
		FileExistsQ[file],
		Check[
			Get[file],
			$Failed,
			{
				DumpGet::bgabi,
				DumpGet::bgbf,
				DumpGet::bgcor,
				DumpGet::bgcom,
				DumpGet::bginc,
				DumpGet::bgmx,
				DumpGet::bgnew,
				DumpGet::bgsid,
				DumpGet::bgsys,
				DumpGet::bgver,
				Get::noopen
			}
		],
		$Failed
	]},
	
	If[SameQ[r,$Failed],
		Block[{Quantity},
			Quantity::noload = "Unable to load Units resources.  Please ensure that your $InstallationDirectory hasn't been corrupted.";
			Message[Quantity::noload];
		];
		End[];
	]
]

getOrFail[___] := getOrFail["NotAFile"]


$UnitDataFile = FileNameJoin[
	{
		FileNameJoin[
			FileNameSplit[$InputFileName][[;;-3]]],
			"UnitData",
			$SystemID,
			"Units.mx"
	}
];

$SystemArchitecture := Switch[$ProcessorType,
	"x86-64", "64Bit",
	"x86", "32Bit",
	"ARM-64", "64Bit",
	"ARM", "32Bit",
	_, "32Bit"]
	
If[
	Not[FileExistsQ[$UnitDataFile]],
	$UnitDataFile = FileNameJoin[
		{
			FileNameJoin[FileNameSplit[$InputFileName][[;;-3]]],
			"UnitData",
			$SystemArchitecture,
			"Units.mx"
		}
	]
];


getOrFail[$UnitDataFile];

standardNameRevisionsToRules[names_List] := Join @@ Map[
   Thread[Reverse[#]] &,
   names]
standardNameRevisionsToRules[___] := {}


aliases = Join[{
 "Celsius"->CalculateUnits`UnitCommonSymbols`DegreesCelsius,
 "Fahrenheit"->CalculateUnits`UnitCommonSymbols`DegreesFahrenheit,
 "Rankine"->CalculateUnits`UnitCommonSymbols`DegreesRankine, 
 "Reaumur"->CalculateUnits`UnitCommonSymbols`DegreesReaumur,
 "Roemer"->CalculateUnits`UnitCommonSymbols`DegreesRoemer,
 "DimensionlessUnit"->CalculateUnits`UnitCommonSymbols`PureUnities,
 "DimensionlessUnits"->CalculateUnits`UnitCommonSymbols`PureUnities,
(* per eww this was removed from W|A due to an Alpha-side issue; adding back here for WL side conversion *)
 "EarthEquatorialRadius"-> UnitCommonSymbols`NominalEarthEquatorialRadius},
 standardNameRevisionsToRules[UnitStandardNameRevisions]
];

Unprotect[QuantityUnits`$UnitList];
QuantityUnits`$UnitList = Join[Keys[QuantityUnits`$UnitTable], aliases[[All, 1]]];


End[];
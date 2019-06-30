(* JSON-LD 1.0 *)
(* https://www.w3.org/TR/json-ld/ *)

BeginPackage["GraphStore`Formats`JSONLD`", {"GraphStore`"}];

Needs["GraphStore`Formats`Utilities`"];
Needs["GraphStore`IRI`"];
Needs["GraphStore`RDF`"];

ExportJSONLD;
Options[ExportJSONLD] = {
	"CompactArrays" -> True,
	"Context" -> None,
	"ExpandContext" -> None,
	"MaxIndentationLevels" -> Infinity,
	"Profile" -> "Compacted",
	"UseNativeTypes" -> False,
	"UseRDFType" -> False
};

ImportJSONLD;
Options[ImportJSONLD] = {
	"Base" -> Automatic,
	"CompactArrays" -> True,
	"Context" -> None,
	"DocumentLoader" -> Automatic,
	"ExpandContext" -> None,
	"ProduceGeneralizedRDF" -> False,
	"Profile" -> None
};

ImportJSONLDRawData;
Options[ImportJSONLDRawData] = Options[ImportJSONLD];

$DocumentLoader = URLExecute[
	HTTPRequest[
		#,
		<|"Headers" -> {"Accept" -> "application/ld+json"}|>
	],
	"RawJSON"
] &;

Begin["`Private`"];

ExportJSONLD[args___] := Catch[iExportJSONLD[args], $failTag, (Message[Export::fmterr, "JSONLD"]; #) &];
ImportJSONLD[file_, opts : OptionsPattern[]] := Catch[iImportJSONLD[file, FilterRules[{opts}, Options[ImportJSONLD]]], $failTag, (Message[Import::fmterr, "JSONLD"]; #) &];
ImportJSONLDRawData[file_, opts : OptionsPattern[]] := Catch[iImportJSONLDRawData[file, FilterRules[{opts}, Options[ImportJSONLDRawData]]], $failTag, (Message[Import::fmterr, "JSONLD"]; #) &];


fail[___] := Throw[$Failed, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


(* -------------------------------------------------- *)
(* export *)

clear[iExportJSONLD];
Options[iExportJSONLD] = Options[ExportJSONLD];
iExportJSONLD[file_, data_, OptionsPattern[]] := Block[{
	$base = None
},
	Module[
		{res},
		res = Switch[data,
			(* raw data *)
			"RawData" :> _,
			Last[data],
			(* RDF *)
			_RDFStore,
			serializeRDFToJSONLD[
				data,
				"UseNativeTypes" -> OptionValue["UseNativeTypes"],
				"UseRDFType" -> OptionValue["UseRDFType"]
			],
			(* unknown *)
			_,
			fail[]
		];
		res = applyProfile[
			res,
			OptionValue["Profile"],
			"CompactArrays" -> OptionValue["CompactArrays"],
			"Context" -> OptionValue["Context"],
			"ExpandContext" -> OptionValue["ExpandContext"]
		];
		WriteString[file, StringReplace[ExportString[
			res,
			"RawJSON",
			"Compact" -> Replace[
				OptionValue["MaxIndentationLevels"],
				{
					Infinity -> False,
					0 -> True,
					Except[_Integer?Positive] :> fail[]
				}
			]
			(* undo escape of forward slash: https://bugs.wolfram.com/show?number=348378 *)
		], "\\/" -> "/"]];
	];
];

(* end export *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* import *)

clear[iImportJSONLD];
Options[iImportJSONLD] = Options[ImportJSONLD];
iImportJSONLD[file_, OptionsPattern[]] := {"Data" -> Block[{
	$base = ChooseBase[OptionValue["Base"], file],
	$documentLoader = OptionValue["DocumentLoader"]
},
	deserializeJSONLDToRDF[
		Quiet[Import[file, "RawJSON"]] // Replace[_?FailureQ :> fail[]],
		OptionValue["ProduceGeneralizedRDF"],
		"Context" -> OptionValue["ExpandContext"]
	]
]};

clear[iImportJSONLDRawData];
Options[iImportJSONLDRawData] = Options[ImportJSONLDRawData];
iImportJSONLDRawData[file_, OptionsPattern[]] := Block[{
	$base = ChooseBase[OptionValue["Base"], file],
	$documentLoader = OptionValue["DocumentLoader"]
},
	{"RawData" -> applyProfile[
		Quiet[Import[file, "RawJSON"]] // Replace[_?FailureQ :> fail[]],
		OptionValue["Profile"],
		"CompactArrays" -> OptionValue["CompactArrays"],
		"Context" -> OptionValue["Context"],
		"ExpandContext" -> OptionValue["ExpandContext"]
	]}
];

(* end import *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* common *)

clear[applyProfile];
Options[applyProfile] = {
	"CompactArrays" -> True,
	"Context" -> None,
	"ExpandContext" -> None
};
applyProfile[data_, profile_, OptionsPattern[]] := Switch[normalizeProfile[profile],
	None,
	data,
	"Expanded",
	expandElement[data, "Context" -> OptionValue["ExpandContext"]],
	"Compacted",
	compactElement[
		expandElement[data, "Context" -> OptionValue["ExpandContext"]],
		OptionValue["Context"],
		OptionValue["CompactArrays"]
	],
	"Flattened",
	flattenElement[
		expandElement[data, "Context" -> OptionValue["ExpandContext"]],
		OptionValue["Context"],
		"CompactArrays" -> OptionValue["CompactArrays"]
	]
];

clear[normalizeProfile];
normalizeProfile["Compacted" | "http://www.w3.org/ns/json-ld#compacted"] := "Compacted";
normalizeProfile["Expanded" | "http://www.w3.org/ns/json-ld#expanded"] := "Expanded";
normalizeProfile["Flattened" | "http://www.w3.org/ns/json-ld#flattened"] := "Flattened";
normalizeProfile[IRI[i_String]] := normalizeProfile[i];
normalizeProfile[None] := None;

clear[importContext];
importContext[file_] := Module[
	{importedContext},
	importedContext = Check[
		Replace[$documentLoader, Automatic :> $DocumentLoader][ExpandIRI[file, $base]],
		fail["loading remote context failed"]
	];
	If[! MatchQ[importedContext, <|___, "@context" -> _, ___|>],
		fail["invalid remote context"],
		importedContext = importedContext["@context"]
	];
	importedContext
];

(* end common *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* algorithms *)

clear[appendIfNotMember];
appendIfNotMember[expr_, elem_] := If[MemberQ[expr, elem],
	expr,
	Append[expr, elem]
];

clear[appendOrJoinTo];
SetAttributes[appendOrJoinTo, HoldFirst];
appendOrJoinTo[s_, elem_] := If[ListQ[elem],
	s = Join[s, elem],
	AppendTo[s, elem]
];

clear[keyInitialize];
SetAttributes[keyInitialize, HoldFirst];
keyInitialize[s_, key_, value_] := If[! KeyExistsQ[s, key], s[key] = value;];

clear[listify];
SetAttributes[listify, HoldAll];
listify[x_] := If[! ListQ[x], x = {x};];

clear[reapList];
SetAttributes[reapList, HoldFirst];
reapList[expr_] := First[Last[Reap[expr;]], {}];


(* JSON-LD 1.0 Processing Algorithms and API *)
(* https://www.w3.org/TR/json-ld-api/ *)


(* 4. General Terminology *)

clear[keywordQ];
keywordQ["@context" | "@id" | "@value" | "@language" | "@type" | "@container" | "@list" | "@set" | "@reverse" | "@index" | "@base" | "@vocab" | "@graph"] := True;
keywordQ[_] := False;

clear[iriQ];
iriQ[s_String] := ! StringStartsQ[s, "@" | "_:"];
iriQ[_] := False;

clear[relativeIRIQ];
relativeIRIQ[x_] := iriQ[x] && ! AbsoluteIRIQ[x];

clear[blankNodeIdentifierQ];
blankNodeIdentifierQ[x_String] := StringStartsQ[x, "_:"];
blankNodeIdentifierQ[_] := False;


(* 5. Algorithm Terms *)

clear[nodeObjectQ];
nodeObjectQ[x_] := AssociationQ[x] && KeyFreeQ[x, "@value" | "@list" | "@set"];

clear[valueObjectQ];
valueObjectQ[x_] := AssociationQ[x] && KeyExistsQ[x, "@value"];

clear[listObjectQ];
listObjectQ[x_] := AssociationQ[x] && KeyExistsQ[x, "@list"];

clear[scalarQ];
scalarQ[x_] := StringQ[x] || NumberQ[x] || BooleanQ[x];


(* 6. Context Processing Algorithms *)
Get[FileNameJoin[{DirectoryName[$InputFileName], "ContextProcessing.wl"}]];


(* 7. Expansion Algorithms *)
Get[FileNameJoin[{DirectoryName[$InputFileName], "Expansion.wl"}]];


(* 8. Compaction Algorithms *)
Get[FileNameJoin[{DirectoryName[$InputFileName], "Compaction.wl"}]];


(* 9. Flattening Algorithms *)
Get[FileNameJoin[{DirectoryName[$InputFileName], "Flattening.wl"}]];


(* 10. RDF Serialization/Deserialization Algorithms *)
Get[FileNameJoin[{DirectoryName[$InputFileName], "RDFSerializationDeserialization.wl"}]];

(* end algorithms *)
(* -------------------------------------------------- *)


End[];
EndPackage[];

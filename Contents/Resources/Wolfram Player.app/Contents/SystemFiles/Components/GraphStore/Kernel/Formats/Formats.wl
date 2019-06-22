BeginPackage["GraphStore`Formats`"];

ExportStringInternal;
ImportInternal;
ImportStringInternal;
MediaTypeToFormat;
$RDFMediaTypes;

Begin["`Private`"];

With[
	{path = DirectoryName[$InputFileName]},
	Get[FileNameJoin[{path, #}]] & /@ {
		"Utilities.wl",
		"SerdLink.wl",
		FileNameJoin[{"JSONLD", "Converter.wl"}],
		FileNameJoin[{"NQuads", "Converter.wl"}],
		FileNameJoin[{"NTriples", "Converter.wl"}],
		FileNameJoin[{"R2RML", "Converter.wl"}],
		FileNameJoin[{"RDFXML", "Converter.wl"}],
		FileNameJoin[{"SPARQLUpdate", "Converter.wl"}],
		FileNameJoin[{"SPARQLQuery", "Converter.wl"}],
		FileNameJoin[{"SPARQLResultsJSON", "Converter.wl"}],
		FileNameJoin[{"SPARQLResultsXML", "Converter.wl"}],
		FileNameJoin[{"SQL", "Converter.wl"}],
		FileNameJoin[{"TriG", "Converter.wl"}],
		FileNameJoin[{"Turtle", "Converter.wl"}]
	};
];

MediaTypeToFormat[type_String] := SelectFirst[
	$ImportFormats,
	StringStartsQ[
		type,
		Alternatives @@ Quiet[Check[ImportExport`GetMIMEType[#], {}, FileFormat::fmterr], FileFormat::fmterr],
		IgnoreCase -> True
	] &
];

$RDFMediaTypes = {
	"application/ld+json",
	"application/n-quads",
	"application/n-triples",
	"application/rdf+xml",
	"application/trig",
	"text/turtle"
};

ExportStringInternal[data_, format_String] := With[
	{file = CreateFile[]},
	With[
		{res = $exporters[format][file, data] // Replace[
			Except[_?FailureQ] :> ReadString[file]
		]},
		Quiet[
			DeleteFile[file],
			{DeleteFile::privv}
		];
		res
	]
];

ImportInternal[file_String | File[file_String], format_String] := Replace[
	$importers[format][file],
	l_List :> Lookup[l, "Data", $Failed]
];

ImportStringInternal[data_String, format_String] := With[
	{file = CreateFile[]},
	WriteString[file, data];
	With[
		{res = ImportInternal[file, format]},
		Quiet[
			DeleteFile[file],
			{DeleteFile::privv}
		];
		res
	]
];

$exporters = <|
	"R2RML" -> ExportR2RML,
	"SQL" -> ExportSQL
|>;

$importers = <|
	"R2RML" -> ImportR2RML,
	"SQL" -> ImportSQL
|>;

End[];
EndPackage[];

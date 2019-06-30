BeginPackage["GraphStore`RDF`"];

CompactRDFCollection;
DatatypeIRI;
ExpandRDFCollection;
FindRDFGraphIsomorphism;
FromRDFLiteral;
IsomorphicRDFStoreQ;
LexicalForm;
RDFMerge;
RDFPrefixData;
ToRDFLiteral;

Begin["`Private`"];

With[
	{path = DirectoryName[$InputFileName]},
	Get[FileNameJoin[{path, #}]] & /@ {
		"Collection.wl",
		"Isomorphism.wl",
		"Literal.wl",
		"Merge.wl",
		"PrefixData.wl",
		"RDFStore.wl"
	};
];

End[];
EndPackage[];

BeginPackage["GraphStore`Entity`"];

EntityEvaluateAlgebraExpression;
EntityFromIRI;
EntityRDFStore;
EntityToIRI;
RDFEntityStore;

Begin["`Private`"];

With[
	{path = DirectoryName[$InputFileName]},
	Get[FileNameJoin[{path, #}]] & /@ {
		"EntityRDFStore.wl",
		"Evaluation.wl",
		"RDFEntityStore.wl",
		"RDFEntityStoreFormatting.wl",
		"RDFEntityStoreNormal.wl",
		"ToFromIRI.wl"
	};
];

End[];
EndPackage[];

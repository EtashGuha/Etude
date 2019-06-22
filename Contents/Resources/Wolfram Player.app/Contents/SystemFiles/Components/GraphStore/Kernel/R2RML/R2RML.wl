BeginPackage["GraphStore`R2RML`"];

DatabaseRDFStore;
DatabaseRDFStoreEvaluateAlgebraExpression;
R2RMLDefaultMapping;

Begin["`Private`"];

With[
	{path = DirectoryName[$InputFileName]},
	Get[FileNameJoin[{path, #}]] & /@ {
		"DatabaseRDFStore.wl",
		"DatabaseRDFStoreFormatting.wl",
		"DatabaseRDFStoreNormal.wl",
		"DefaultMapping.wl"
	};
];

End[];
EndPackage[];

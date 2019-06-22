BeginPackage["GraphStore`"];

IRI;
RDFBlankNode;
RDFCollection;
RDFLiteral;
RDFStore;
RDFString;
RDFTriple;
SPARQLAdd;
SPARQLAggregate;
SPARQLAsk;
SPARQLClear;
SPARQLConstruct;
SPARQLCopy;
SPARQLCreate;
SPARQLDelete;
SPARQLDeleteData;
SPARQLDeleteInsert;
SPARQLDrop;
SPARQLEvaluation;
SPARQLExecute;
SPARQLGraph;
SPARQLInsert;
SPARQLInsertData;
SPARQLInverseProperty;
SPARQLLoad;
SPARQLMove;
SPARQLOptional;
SPARQLPropertyPath;
SPARQLQuery;
SPARQLSelect;
SPARQLService;
SPARQLUpdate;
SPARQLValues;
SPARQLVariable;

Begin["`Private`"];

Unprotect["GraphStore`*"];
ClearAll["GraphStore`*"];
ClearAll["GraphStore`*`*"];

With[
	{path = DirectoryName[$InputFileName]},
	Get[FileNameJoin[{path, #}]] & /@ {
		"ArrayAssociation.wl",
		"IRI.wl",
		"IRIValue.wl",
		"LanguageTag.wl",
		"Parsing.wl",
		"SQL.wl",
		"SubsetCases.wl",
		"VOWL.wl",
		FileNameJoin[{"Entity", "Entity.wl"}],
		FileNameJoin[{"Formats", "Formats.wl"}],
		FileNameJoin[{"R2RML", "R2RML.wl"}],
		FileNameJoin[{"RDF", "RDF.wl"}],
		FileNameJoin[{"SPARQL", "SPARQL.wl"}]
	}
];

ToExpression[
	Names["GraphStore`*"],
	InputForm,
	Function[Null,
		SetAttributes[#, {ReadProtected, Protected}],
		HoldAllComplete
	]
];

SPARQLAlgebraEvaluatorRegister[_DatabaseRDFStore, DatabaseRDFStoreEvaluateAlgebraExpression];
SPARQLAlgebraEvaluatorRegister[HoldPattern[Entity[_String] | {Entity[_String] ..}], EntityEvaluateAlgebraExpression];

End[];
EndPackage[];

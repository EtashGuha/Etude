BeginPackage["GraphStore`SPARQL`Entailment`", {"GraphStore`", "GraphStore`SPARQL`"}];

Needs["GraphStore`RDF`"];
Needs["GraphStore`SubsetCases`"];

Begin["`Private`"];

EvaluateBasicGraphPatternWithEntailment[args___] := With[{res = Catch[iEvaluateBasicGraphPatternWithEntailment[args], $failTag]}, res /; res =!= $failTag]


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


clear[iEvaluateBasicGraphPatternWithEntailment];

(* RDF *)
iEvaluateBasicGraphPatternWithEntailment[el_, patt_, IRI["http://www.w3.org/ns/entailment/RDF"]] := extendGraph[
	el,
	$rdfVocabulary,
	$rdfEntailmentRules
] // SPARQLQuery[SPARQLSelect[patt], SPARQLEntailmentRegime -> None] // Replace[_SPARQLQuery :> fail[]];

(* RDFS *)
iEvaluateBasicGraphPatternWithEntailment[el_, patt_, IRI["http://www.w3.org/ns/entailment/RDFS"]] := extendGraph[
	el,
	RDFMerge[{$rdfVocabulary, $rdfsVocabulary}],
	Join[$rdfEntailmentRules, $rdfsEntailmentRules]
] // SPARQLQuery[SPARQLSelect[patt], SPARQLEntailmentRegime -> None] // Replace[_SPARQLQuery :> fail[]];

iEvaluateBasicGraphPatternWithEntailment[el_, patt_, er_] := iEvaluateBasicGraphPatternWithEntailment[el, patt, normalizeEntailmentRegime[er]];


clear[normalizeEntailmentRegime];
normalizeEntailmentRegime["http://www.w3.org/ns/entailment/RDF" | "RDF"] := IRI["http://www.w3.org/ns/entailment/RDF"];
normalizeEntailmentRegime["http://www.w3.org/ns/entailment/RDFS" | "RDFS"] := IRI["http://www.w3.org/ns/entailment/RDFS"];
normalizeEntailmentRegime["http://www.w3.org/ns/entailment/D" | "D"] := IRI["http://www.w3.org/ns/entailment/D"];
normalizeEntailmentRegime["http://www.w3.org/ns/entailment/OWL-RDF-Based" | "OWLRDFBased"] := IRI["http://www.w3.org/ns/entailment/OWL-RDF-Based"];
normalizeEntailmentRegime["http://www.w3.org/ns/entailment/OWL-Direct" | "OWLDirect"] := IRI["http://www.w3.org/ns/entailment/OWL-Direct"];
normalizeEntailmentRegime["http://www.w3.org/ns/entailment/RIF" | "RIF"] := IRI["http://www.w3.org/ns/entailment/RIF"];
normalizeEntailmentRegime[URL[url_String]] := normalizeEntailmentRegime[url];
normalizeEntailmentRegime[x_] := (Message[SPARQLQuery::erropts, x, SPARQLEntailmentRegime]; fail[]);


(* vocabulary *)
clear /@ {rdf, rdfs};
rdf[s_String] := IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#" <> s];
rdfs[s_String] := IRI["http://www.w3.org/2000/01/rdf-schema#" <> s];


clear[resourceQ];
resourceQ[_IRI] := True;
resourceQ[_RDFBlankNode] := True;
resourceQ[_] := False;


(* http://www.w3.org/1999/02/22-rdf-syntax-ns# *)
$rdfVocabulary = RDFStore[{
	RDFTriple[IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#"], rdf["type"], IRI["http://www.w3.org/2002/07/owl#Ontology"]],
	RDFTriple[IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#"], IRI["http://purl.org/dc/elements/1.1/title"], "The RDF Concepts Vocabulary (RDF)"],
	RDFTriple[IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#"], IRI["http://purl.org/dc/elements/1.1/description"], "This is the RDF Schema for the RDF vocabulary terms in the RDF Namespace, defined in RDF 1.1 Concepts."],
	RDFTriple[rdf["HTML"], rdf["type"], rdfs["Datatype"]],
	RDFTriple[rdf["HTML"], rdfs["subClassOf"], rdfs["Literal"]],
	RDFTriple[rdf["HTML"], rdfs["isDefinedBy"], IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#"]],
	RDFTriple[rdf["HTML"], rdfs["seeAlso"], IRI["http://www.w3.org/TR/rdf11-concepts/#section-html"]],
	RDFTriple[rdf["HTML"], rdfs["label"], "HTML"],
	RDFTriple[rdf["HTML"], rdfs["comment"], "The datatype of RDF literals storing fragments of HTML content"],
	RDFTriple[rdf["langString"], rdf["type"], rdfs["Datatype"]],
	RDFTriple[rdf["langString"], rdfs["subClassOf"], rdfs["Literal"]],
	RDFTriple[rdf["langString"], rdfs["isDefinedBy"], IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#"]],
	RDFTriple[rdf["langString"], rdfs["seeAlso"], IRI["http://www.w3.org/TR/rdf11-concepts/#section-Graph-Literal"]],
	RDFTriple[rdf["langString"], rdfs["label"], "langString"],
	RDFTriple[rdf["langString"], rdfs["comment"], "The datatype of language-tagged string values"],
	RDFTriple[rdf["PlainLiteral"], rdf["type"], rdfs["Datatype"]],
	RDFTriple[rdf["PlainLiteral"], rdfs["isDefinedBy"], IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#"]],
	RDFTriple[rdf["PlainLiteral"], rdfs["subClassOf"], rdfs["Literal"]],
	RDFTriple[rdf["PlainLiteral"], rdfs["seeAlso"], IRI["http://www.w3.org/TR/rdf-plain-literal/"]],
	RDFTriple[rdf["PlainLiteral"], rdfs["label"], "PlainLiteral"],
	RDFTriple[rdf["PlainLiteral"], rdfs["comment"], "The class of plain (i.e. untyped) literal values, as used in RIF and OWL 2"],
	RDFTriple[rdf["type"], rdf["type"], rdf["Property"]],
	RDFTriple[rdf["type"], rdfs["isDefinedBy"], IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#"]],
	RDFTriple[rdf["type"], rdfs["label"], "type"],
	RDFTriple[rdf["type"], rdfs["comment"], "The subject is an instance of a class."],
	RDFTriple[rdf["type"], rdfs["range"], rdfs["Class"]],
	RDFTriple[rdf["type"], rdfs["domain"], rdfs["Resource"]],
	RDFTriple[rdf["Property"], rdf["type"], rdfs["Class"]],
	RDFTriple[rdf["Property"], rdfs["isDefinedBy"], IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#"]],
	RDFTriple[rdf["Property"], rdfs["label"], "Property"],
	RDFTriple[rdf["Property"], rdfs["comment"], "The class of RDF properties."],
	RDFTriple[rdf["Property"], rdfs["subClassOf"], rdfs["Resource"]],
	RDFTriple[rdf["Statement"], rdf["type"], rdfs["Class"]],
	RDFTriple[rdf["Statement"], rdfs["isDefinedBy"], IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#"]],
	RDFTriple[rdf["Statement"], rdfs["label"], "Statement"],
	RDFTriple[rdf["Statement"], rdfs["subClassOf"], rdfs["Resource"]],
	RDFTriple[rdf["Statement"], rdfs["comment"], "The class of RDF statements."],
	RDFTriple[rdf["subject"], rdf["type"], rdf["Property"]],
	RDFTriple[rdf["subject"], rdfs["isDefinedBy"], IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#"]],
	RDFTriple[rdf["subject"], rdfs["label"], "subject"],
	RDFTriple[rdf["subject"], rdfs["comment"], "The subject of the subject RDF statement."],
	RDFTriple[rdf["subject"], rdfs["domain"], rdf["Statement"]],
	RDFTriple[rdf["subject"], rdfs["range"], rdfs["Resource"]],
	RDFTriple[rdf["predicate"], rdf["type"], rdf["Property"]],
	RDFTriple[rdf["predicate"], rdfs["isDefinedBy"], IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#"]],
	RDFTriple[rdf["predicate"], rdfs["label"], "predicate"],
	RDFTriple[rdf["predicate"], rdfs["comment"], "The predicate of the subject RDF statement."],
	RDFTriple[rdf["predicate"], rdfs["domain"], rdf["Statement"]],
	RDFTriple[rdf["predicate"], rdfs["range"], rdfs["Resource"]],
	RDFTriple[rdf["object"], rdf["type"], rdf["Property"]],
	RDFTriple[rdf["object"], rdfs["isDefinedBy"], IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#"]],
	RDFTriple[rdf["object"], rdfs["label"], "object"],
	RDFTriple[rdf["object"], rdfs["comment"], "The object of the subject RDF statement."],
	RDFTriple[rdf["object"], rdfs["domain"], rdf["Statement"]],
	RDFTriple[rdf["object"], rdfs["range"], rdfs["Resource"]],
	RDFTriple[rdf["Bag"], rdf["type"], rdfs["Class"]],
	RDFTriple[rdf["Bag"], rdfs["isDefinedBy"], IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#"]],
	RDFTriple[rdf["Bag"], rdfs["label"], "Bag"],
	RDFTriple[rdf["Bag"], rdfs["comment"], "The class of unordered containers."],
	RDFTriple[rdf["Bag"], rdfs["subClassOf"], rdfs["Container"]],
	RDFTriple[rdf["Seq"], rdf["type"], rdfs["Class"]],
	RDFTriple[rdf["Seq"], rdfs["isDefinedBy"], IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#"]],
	RDFTriple[rdf["Seq"], rdfs["label"], "Seq"],
	RDFTriple[rdf["Seq"], rdfs["comment"], "The class of ordered containers."],
	RDFTriple[rdf["Seq"], rdfs["subClassOf"], rdfs["Container"]],
	RDFTriple[rdf["Alt"], rdf["type"], rdfs["Class"]],
	RDFTriple[rdf["Alt"], rdfs["isDefinedBy"], IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#"]],
	RDFTriple[rdf["Alt"], rdfs["label"], "Alt"],
	RDFTriple[rdf["Alt"], rdfs["comment"], "The class of containers of alternatives."],
	RDFTriple[rdf["Alt"], rdfs["subClassOf"], rdfs["Container"]],
	RDFTriple[rdf["value"], rdf["type"], rdf["Property"]],
	RDFTriple[rdf["value"], rdfs["isDefinedBy"], IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#"]],
	RDFTriple[rdf["value"], rdfs["label"], "value"],
	RDFTriple[rdf["value"], rdfs["comment"], "Idiomatic property used for structured values."],
	RDFTriple[rdf["value"], rdfs["domain"], rdfs["Resource"]],
	RDFTriple[rdf["value"], rdfs["range"], rdfs["Resource"]],
	RDFTriple[rdf["List"], rdf["type"], rdfs["Class"]],
	RDFTriple[rdf["List"], rdfs["isDefinedBy"], IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#"]],
	RDFTriple[rdf["List"], rdfs["label"], "List"],
	RDFTriple[rdf["List"], rdfs["comment"], "The class of RDF Lists."],
	RDFTriple[rdf["List"], rdfs["subClassOf"], rdfs["Resource"]],
	RDFTriple[rdf["nil"], rdf["type"], rdf["List"]],
	RDFTriple[rdf["nil"], rdfs["isDefinedBy"], IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#"]],
	RDFTriple[rdf["nil"], rdfs["label"], "nil"],
	RDFTriple[rdf["nil"], rdfs["comment"], "The empty list, with no items in it. If the rest of a list is nil then the list has no more items in it."],
	RDFTriple[rdf["first"], rdf["type"], rdf["Property"]],
	RDFTriple[rdf["first"], rdfs["isDefinedBy"], IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#"]],
	RDFTriple[rdf["first"], rdfs["label"], "first"],
	RDFTriple[rdf["first"], rdfs["comment"], "The first item in the subject RDF list."],
	RDFTriple[rdf["first"], rdfs["domain"], rdf["List"]],
	RDFTriple[rdf["first"], rdfs["range"], rdfs["Resource"]],
	RDFTriple[rdf["rest"], rdf["type"], rdf["Property"]],
	RDFTriple[rdf["rest"], rdfs["isDefinedBy"], IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#"]],
	RDFTriple[rdf["rest"], rdfs["label"], "rest"],
	RDFTriple[rdf["rest"], rdfs["comment"], "The rest of the subject RDF list after the first item."],
	RDFTriple[rdf["rest"], rdfs["domain"], rdf["List"]],
	RDFTriple[rdf["rest"], rdfs["range"], rdf["List"]],
	RDFTriple[rdf["XMLLiteral"], rdf["type"], rdfs["Datatype"]],
	RDFTriple[rdf["XMLLiteral"], rdfs["subClassOf"], rdfs["Literal"]],
	RDFTriple[rdf["XMLLiteral"], rdfs["isDefinedBy"], IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#"]],
	RDFTriple[rdf["XMLLiteral"], rdfs["label"], "XMLLiteral"],
	RDFTriple[rdf["XMLLiteral"], rdfs["comment"], "The datatype of XML literal values."]
}];

(* https://www.w3.org/TR/rdf11-mt/#patterns-of-rdf-entailment-informative *)
$rdfEntailmentRules = {
	(* rdfD2 *)
	{
		RDFTriple[xxx_, aaa_, yyy_]
	} :> RDFTriple[aaa, rdf["type"], rdf["Property"]]
};


(* http://www.w3.org/2000/01/rdf-schema# *)
$rdfsVocabulary = RDFStore[{
	RDFTriple[IRI["http://www.w3.org/2000/01/rdf-schema#"], rdf["type"], IRI["http://www.w3.org/2002/07/owl#Ontology"]],
	RDFTriple[IRI["http://www.w3.org/2000/01/rdf-schema#"], IRI["http://purl.org/dc/elements/1.1/title"], "The RDF Schema vocabulary (RDFS)"],
	RDFTriple[rdfs["Resource"], rdf["type"], rdfs["Class"]],
	RDFTriple[rdfs["Resource"], rdfs["isDefinedBy"], IRI["http://www.w3.org/2000/01/rdf-schema#"]],
	RDFTriple[rdfs["Resource"], rdfs["label"], "Resource"],
	RDFTriple[rdfs["Resource"], rdfs["comment"], "The class resource, everything."],
	RDFTriple[rdfs["Class"], rdf["type"], rdfs["Class"]],
	RDFTriple[rdfs["Class"], rdfs["isDefinedBy"], IRI["http://www.w3.org/2000/01/rdf-schema#"]],
	RDFTriple[rdfs["Class"], rdfs["label"], "Class"],
	RDFTriple[rdfs["Class"], rdfs["comment"], "The class of classes."],
	RDFTriple[rdfs["Class"], rdfs["subClassOf"], rdfs["Resource"]],
	RDFTriple[rdfs["subClassOf"], rdf["type"], rdf["Property"]],
	RDFTriple[rdfs["subClassOf"], rdfs["isDefinedBy"], IRI["http://www.w3.org/2000/01/rdf-schema#"]],
	RDFTriple[rdfs["subClassOf"], rdfs["label"], "subClassOf"],
	RDFTriple[rdfs["subClassOf"], rdfs["comment"], "The subject is a subclass of a class."],
	RDFTriple[rdfs["subClassOf"], rdfs["range"], rdfs["Class"]],
	RDFTriple[rdfs["subClassOf"], rdfs["domain"], rdfs["Class"]],
	RDFTriple[rdfs["subPropertyOf"], rdf["type"], rdf["Property"]],
	RDFTriple[rdfs["subPropertyOf"], rdfs["isDefinedBy"], IRI["http://www.w3.org/2000/01/rdf-schema#"]],
	RDFTriple[rdfs["subPropertyOf"], rdfs["label"], "subPropertyOf"],
	RDFTriple[rdfs["subPropertyOf"], rdfs["comment"], "The subject is a subproperty of a property."],
	RDFTriple[rdfs["subPropertyOf"], rdfs["range"], rdf["Property"]],
	RDFTriple[rdfs["subPropertyOf"], rdfs["domain"], rdf["Property"]],
	RDFTriple[rdfs["comment"], rdf["type"], rdf["Property"]],
	RDFTriple[rdfs["comment"], rdfs["isDefinedBy"], IRI["http://www.w3.org/2000/01/rdf-schema#"]],
	RDFTriple[rdfs["comment"], rdfs["label"], "comment"],
	RDFTriple[rdfs["comment"], rdfs["comment"], "A description of the subject resource."],
	RDFTriple[rdfs["comment"], rdfs["domain"], rdfs["Resource"]],
	RDFTriple[rdfs["comment"], rdfs["range"], rdfs["Literal"]],
	RDFTriple[rdfs["label"], rdf["type"], rdf["Property"]],
	RDFTriple[rdfs["label"], rdfs["isDefinedBy"], IRI["http://www.w3.org/2000/01/rdf-schema#"]],
	RDFTriple[rdfs["label"], rdfs["label"], "label"],
	RDFTriple[rdfs["label"], rdfs["comment"], "A human-readable name for the subject."],
	RDFTriple[rdfs["label"], rdfs["domain"], rdfs["Resource"]],
	RDFTriple[rdfs["label"], rdfs["range"], rdfs["Literal"]],
	RDFTriple[rdfs["domain"], rdf["type"], rdf["Property"]],
	RDFTriple[rdfs["domain"], rdfs["isDefinedBy"], IRI["http://www.w3.org/2000/01/rdf-schema#"]],
	RDFTriple[rdfs["domain"], rdfs["label"], "domain"],
	RDFTriple[rdfs["domain"], rdfs["comment"], "A domain of the subject property."],
	RDFTriple[rdfs["domain"], rdfs["range"], rdfs["Class"]],
	RDFTriple[rdfs["domain"], rdfs["domain"], rdf["Property"]],
	RDFTriple[rdfs["range"], rdf["type"], rdf["Property"]],
	RDFTriple[rdfs["range"], rdfs["isDefinedBy"], IRI["http://www.w3.org/2000/01/rdf-schema#"]],
	RDFTriple[rdfs["range"], rdfs["label"], "range"],
	RDFTriple[rdfs["range"], rdfs["comment"], "A range of the subject property."],
	RDFTriple[rdfs["range"], rdfs["range"], rdfs["Class"]],
	RDFTriple[rdfs["range"], rdfs["domain"], rdf["Property"]],
	RDFTriple[rdfs["seeAlso"], rdf["type"], rdf["Property"]],
	RDFTriple[rdfs["seeAlso"], rdfs["isDefinedBy"], IRI["http://www.w3.org/2000/01/rdf-schema#"]],
	RDFTriple[rdfs["seeAlso"], rdfs["label"], "seeAlso"],
	RDFTriple[rdfs["seeAlso"], rdfs["comment"], "Further information about the subject resource."],
	RDFTriple[rdfs["seeAlso"], rdfs["range"], rdfs["Resource"]],
	RDFTriple[rdfs["seeAlso"], rdfs["domain"], rdfs["Resource"]],
	RDFTriple[rdfs["isDefinedBy"], rdf["type"], rdf["Property"]],
	RDFTriple[rdfs["isDefinedBy"], rdfs["isDefinedBy"], IRI["http://www.w3.org/2000/01/rdf-schema#"]],
	RDFTriple[rdfs["isDefinedBy"], rdfs["subPropertyOf"], rdfs["seeAlso"]],
	RDFTriple[rdfs["isDefinedBy"], rdfs["label"], "isDefinedBy"],
	RDFTriple[rdfs["isDefinedBy"], rdfs["comment"], "The defininition of the subject resource."],
	RDFTriple[rdfs["isDefinedBy"], rdfs["range"], rdfs["Resource"]],
	RDFTriple[rdfs["isDefinedBy"], rdfs["domain"], rdfs["Resource"]],
	RDFTriple[rdfs["Literal"], rdf["type"], rdfs["Class"]],
	RDFTriple[rdfs["Literal"], rdfs["isDefinedBy"], IRI["http://www.w3.org/2000/01/rdf-schema#"]],
	RDFTriple[rdfs["Literal"], rdfs["label"], "Literal"],
	RDFTriple[rdfs["Literal"], rdfs["comment"], "The class of literal values, eg. textual strings and integers."],
	RDFTriple[rdfs["Literal"], rdfs["subClassOf"], rdfs["Resource"]],
	RDFTriple[rdfs["Container"], rdf["type"], rdfs["Class"]],
	RDFTriple[rdfs["Container"], rdfs["isDefinedBy"], IRI["http://www.w3.org/2000/01/rdf-schema#"]],
	RDFTriple[rdfs["Container"], rdfs["label"], "Container"],
	RDFTriple[rdfs["Container"], rdfs["subClassOf"], rdfs["Resource"]],
	RDFTriple[rdfs["Container"], rdfs["comment"], "The class of RDF containers."],
	RDFTriple[rdfs["ContainerMembershipProperty"], rdf["type"], rdfs["Class"]],
	RDFTriple[rdfs["ContainerMembershipProperty"], rdfs["isDefinedBy"], IRI["http://www.w3.org/2000/01/rdf-schema#"]],
	RDFTriple[rdfs["ContainerMembershipProperty"], rdfs["label"], "ContainerMembershipProperty"],
	RDFTriple[rdfs["ContainerMembershipProperty"], rdfs["comment"], "The class of container membership properties, rdf:_1, rdf:_2, ...,\n                    all of which are sub-properties of 'member'."],
	RDFTriple[rdfs["ContainerMembershipProperty"], rdfs["subClassOf"], rdf["Property"]],
	RDFTriple[rdfs["member"], rdf["type"], rdf["Property"]],
	RDFTriple[rdfs["member"], rdfs["isDefinedBy"], IRI["http://www.w3.org/2000/01/rdf-schema#"]],
	RDFTriple[rdfs["member"], rdfs["label"], "member"],
	RDFTriple[rdfs["member"], rdfs["comment"], "A member of the subject resource."],
	RDFTriple[rdfs["member"], rdfs["domain"], rdfs["Resource"]],
	RDFTriple[rdfs["member"], rdfs["range"], rdfs["Resource"]],
	RDFTriple[rdfs["Datatype"], rdf["type"], rdfs["Class"]],
	RDFTriple[rdfs["Datatype"], rdfs["isDefinedBy"], IRI["http://www.w3.org/2000/01/rdf-schema#"]],
	RDFTriple[rdfs["Datatype"], rdfs["label"], "Datatype"],
	RDFTriple[rdfs["Datatype"], rdfs["comment"], "The class of RDF datatypes."],
	RDFTriple[rdfs["Datatype"], rdfs["subClassOf"], rdfs["Class"]],
	RDFTriple[IRI["http://www.w3.org/2000/01/rdf-schema#"], rdfs["seeAlso"], IRI["http://www.w3.org/2000/01/rdf-schema-more"]]
}];

(* https://www.w3.org/TR/rdf11-mt/#patterns-of-rdfs-entailment-informative *)
$rdfsEntailmentRules = {
	(* rdfs2 *)
	{
		RDFTriple[aaa_, rdfs["domain"], xxx_],
		RDFTriple[yyy_?resourceQ, aaa_, zzz_]
	} :> RDFTriple[yyy, rdf["type"], xxx],
	(* rdfs3 *)
	{
		RDFTriple[aaa_, rdfs["range"], xxx_],
		RDFTriple[yyy_, aaa_, zzz_?resourceQ]
	} :> RDFTriple[zzz, rdf["type"], xxx],
	(* rdfs4a *)
	{
		RDFTriple[xxx_, aaa_, yyy_]
	} :> RDFTriple[xxx, rdf["type"], rdfs["Resource"]],
	(* rdfs4b *)
	{
		RDFTriple[xxx_, aaa_, yyy_]
	} :> RDFTriple[yyy, rdf["type"], rdfs["Resource"]],
	(* rdfs5 *)
	{
		RDFTriple[xxx_, rdfs["subPropertyOf"], yyy_],
		RDFTriple[yyy_, rdfs["subPropertyOf"], zzz_]
	} :> RDFTriple[xxx, rdfs["subPropertyOf"], zzz],
	(* rdfs6 *)
	{
		RDFTriple[xxx_, rdf["type"], rdf["Property"]]
	} :> RDFTriple[xxx, rdfs["subPropertyOf"], xxx],
	(* rdfs7 *)
	{
		RDFTriple[aaa_, rdfs["subPropertyOf"], bbb_],
		RDFTriple[xxx_, aaa_, yyy_]
	} :> RDFTriple[xxx, bbb, yyy],
	(* rdfs8 *)
	{
		RDFTriple[xxx_, rdf["type"], rdfs["Class"]]
	} :> RDFTriple[xxx, rdfs["subClassOf"], rdfs["Resource"]],
	(* rdfs9 *)
	{
		RDFTriple[xxx_, rdfs["subClassOf"], yyy_],
		RDFTriple[zzz_, rdf["type"], xxx_]
	} :> RDFTriple[zzz, rdf["type"], yyy],
	(* rdfs10 *)
	{
		RDFTriple[xxx_, rdf["type"], rdfs["Class"]]
	} :> RDFTriple[xxx, rdfs["subClassOf"], xxx],
	(* rdfs11 *)
	{
		RDFTriple[xxx_, rdfs["subClassOf"], yyy_],
		RDFTriple[yyy_, rdfs["subClassOf"], zzz_]
	} :> RDFTriple[xxx, rdfs["subClassOf"], zzz]
};


clear[extendGraph];
extendGraph[el_, vocab_, rules_] := RDFStore[
	FixedPoint[
		extendEdgeList[#, rules] &,
		First[RDFMerge[{RDFStore[el], vocab}]]
	]
];

clear[extendEdgeList];
extendEdgeList[el_, rules_] := Join[
	el,
	Complement[
		Join @@ Function[SubsetCases[el, #]] /@ rules,
		el
	]
];

End[];
EndPackage[];

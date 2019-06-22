BeginPackage["GraphStore`Entity`RDFEntityStore`", {"GraphStore`", "GraphStore`Entity`"}];

Needs["GraphStore`IRI`"];
Needs["GraphStore`RDF`"];
Needs["GraphStore`SPARQL`"];

Begin["`Private`"];

RDFEntityStore[meta : _String | _Rule | _RuleDelayed, rest___] := RDFEntityStore[{meta}, rest];
RDFEntityStore[meta : {(_String | _Rule | _RuleDelayed) ..}, rest___] := RDFEntityStore[
	<|Replace[
		meta,
		type_String :> type -> <||>,
		{1}
	]|>,
	rest
];

es_RDFEntityStore[args___] := With[{res = Catch[iRDFEntityStore[es, args], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


clear[iRDFEntityStore];


(* type list *)
iRDFEntityStore[es_] := Keys[getMetadata[es]];


(* type properties *)
iRDFEntityStore[es_, HoldPattern[type : Entity[_String]], "EntityClassCanonicalNames"] := With[
	{cvar = createVar["cvar"]},
	execute[
		es,
		SPARQLSelect[
			getMetadata[es, type, "EntityClassesQuery"][SPARQLVariable[cvar]] /; SPARQLEvaluation["isIRI"][SPARQLVariable[cvar]]
		]
	] // Query[All, cvar /* First]
];
iRDFEntityStore[es_, HoldPattern[type : Entity[_String]], "PropertyClassCanonicalNames"] := With[
	{pcvar = createVar["pcvar"]},
	execute[
		es,
		SPARQLSelect[
			getMetadata[es, type, "PropertyClassesQuery"][SPARQLVariable[pcvar]] /; SPARQLEvaluation["isIRI"][SPARQLVariable[pcvar]]
		]
	] // Query[All, pcvar /* First]
];
iRDFEntityStore[es_, HoldPattern[type : Entity[_String]], prop_String] := getMetadata[es, type, prop];


(* entity value *)
iRDFEntityStore[_, {}, _] := {};
iRDFEntityStore[_, ent_List, {}] := ConstantArray[{}, Length[ent]];
iRDFEntityStore[es_, HoldPattern[class : EntityClass[type_, _]], {}] := ConstantArray[
	{},
	Length[iRDFEntityStore[es, class, {EntityProperty[type, "CanonicalName"]}]]
];

iRDFEntityStore[es_, HoldPattern[ent : {Entity[type_, _] ..} | EntityClass[type_, _]], HoldPattern[prop : {__EntityProperty}?(MemberQ[EntityProperty[_, "Label"]])]] := Condition[
	If[ListQ[ent] && Length[prop] === 1,
		{labelFromIRI[#]} & /@ ent[[All, 2]],
		With[
			{pos = First[FirstPosition[prop, EntityProperty[_, "Label"], Null, {1}, Heads -> False]]},
			MapAt[
				labelFromIRI,
				iRDFEntityStore[
					es,
					ent,
					ReplacePart[
						prop,
						pos -> EntityProperty[type, "CanonicalName"]
					]
				],
				{All, pos}
			]
		]
	],
	! KeyExistsQ[getMetadata[es, Entity[type], "Properties"], "Label"]
];
iRDFEntityStore[es_, HoldPattern[ent : {Entity[type_, _] ..} | EntityClass[type_, _]], HoldPattern[prop : {__EntityProperty}]] := Condition[
	Module[{
		res,
		pos = First[FirstPosition[prop, p_ /; ! validPropertyQ[es, p], Null, {1}, Heads -> False]]
	},
		res = iRDFEntityStore[
			es,
			ent,
			Delete[prop, pos]
		];
		res = Insert[
			res,
			Missing["NotAvailable"],
			{#, pos} & /@ Range[Length[res]]
		];
		res
	],
	AnyTrue[
		prop,
		! validPropertyQ[es, #] &
	]
];

clear[validPropertyQ];
validPropertyQ[_, HoldPattern[EntityProperty[_, "CanonicalName"]]] := True;
validPropertyQ[es_, HoldPattern[EntityProperty[type_, pname_]]] := Or[
	KeyExistsQ[getMetadata[es, Entity[type], "Properties"], pname],
	IRIQ[pname]
];

iRDFEntityStore[es_, HoldPattern[ent : {Entity[type_, _] ..}], HoldPattern[prop : {__EntityProperty}]] := iRDFEntityStore[
	es,
	EntityClass[type, {EntityProperty[type, "CanonicalName"] -> Alternatives @@ ent[[All, 2]]}],
	Append[prop, EntityProperty[type, "CanonicalName"]]
] // GroupBy[#, Last -> Most, First] & // Lookup[ent[[All, 2]]];

iRDFEntityStore[es_, HoldPattern[class : EntityClass[type_, _]], HoldPattern[prop : {__EntityProperty}]] := With[{
	evar = createVar["evar"],
	pvar = createVar["pvar"],
	vvars = AssociationMap[createVar["vvar"] &, prop]
},
	execute[
		es,
		SPARQLSelect[{
			classToSubquery[es, SPARQLVariable[evar], class] /; SPARQLEvaluation["isIRI"][SPARQLVariable[evar]],
			SPARQLValues[pvar, CanonicalName[prop]],
			Sequence @@ Function[p,
				propertyToSubquery[es, SPARQLVariable[evar], SPARQLVariable[pvar], SPARQLVariable[vvars[p]], p]
			] /@ prop
		}]
	] //
	GroupBy[{Key[evar] -> KeyDrop[evar], Key[pvar] -> KeyDrop[pvar]}] //
	KeySort //
	Values //
	Map[Function[Lookup[#, CanonicalName[prop], {}]]] //
	Map[With[
		{pf = Function[p, With[{
			f = getMetadata[es, p, "ValueProcessingFunction"],
			v = vvars[p]
		},
			Select[KeyExistsQ[v]] /* Function[f[#, SPARQLVariable[v]]]
		]] /@ prop},
		Function[edata, MapThread[Compose, {pf, edata}]]
	]]
];


(* property value *)
iRDFEntityStore[es_, HoldPattern[prop : {__EntityProperty}], sub : {__String}] := Outer[getMetadata[es, ##] &, prop, sub];

iRDFEntityStore[es_, HoldPattern[class : EntityPropertyClass[type_, name_]], sub : {___String}] := iRDFEntityStore[
	es,
	EntityProperty[type, #] & /@ Switch[name,
		{},
		Union[
			Keys[getMetadata[es, Entity[type], "Properties"]],
			With[{pvar = createVar["pvar"]},
				execute[
					es,
					SPARQLSelect[
						getMetadata[es, Entity[type], "PropertiesQuery"][SPARQLVariable[pvar]] /; SPARQLEvaluation["isIRI"][SPARQLVariable[pvar]]
					]
				] // Query[All, pvar /* First]
			]
		],
		_String,
		With[{pvar = createVar["pvar"]},
			execute[
				es,
				SPARQLSelect[
					RDFTriple[SPARQLVariable[pvar], RDFPrefixData["rdf", "type"], IRI[name]]
				]
			] // Query[All, pvar /* First]
		],
		_,
		fail[]
	],
	sub
];


(* common name *)
iRDFEntityStore[_, HoldPattern[CommonName], {}] := {};
iRDFEntityStore[
	es_,
	HoldPattern[CommonName],
	HoldPattern[classes : {EntityClass[_, _String] ..} | {EntityPropertyClass[_, _String] ..}]
] := labelFromIRI /@ classes[[All, 2]];


clear[execute];
execute[RDFEntityStore[_, store_], query_] := SPARQLExecute[
	store,
	query
] // Replace[_SPARQLExecute :> fail[]];


clear[createVar];
createVar[prefix_String] := StringDelete[CreateUUID[prefix], "-"];


clear[getMetadata];
getMetadata[RDFEntityStore[meta_, ___]] := meta;

getMetadata[es_, HoldPattern[Entity[type_String]]] := Lookup[
	getMetadata[es],
	type,
	<||>
];
getMetadata[es_, HoldPattern[type : Entity[_String]], prop : "Properties"] := Lookup[
	getMetadata[es, type],
	prop,
	<||>
];
getMetadata[es_, HoldPattern[type : Entity[_String]], prop : "EntitiesQuery"] := Replace[
	Lookup[getMetadata[es, type], prop, Automatic],
	{
		Automatic :> Function[evar,
			SPARQLSelect[RDFTriple[evar, RDFBlankNode[], RDFBlankNode[]]] /*
			SPARQLDistinct[]
		],
		None :> Function[evar, {} /; False]
	}
];
getMetadata[es_, HoldPattern[type : Entity[_String]], prop : "EntityClassesQuery"] := Replace[
	Lookup[getMetadata[es, type], prop, Automatic],
	{
		Automatic :> Function[cvar,
			SPARQLSelect[RDFTriple[RDFBlankNode[], RDFPrefixData["rdf", "type"], cvar]] /*
			SPARQLDistinct[]
		],
		None :> Function[cvar, {} /; False]
	}
];
getMetadata[es_, HoldPattern[type : Entity[_String]], prop : "PropertiesQuery"] := Replace[
	Lookup[getMetadata[es, type], prop, Automatic],
	{
		Automatic :> Function[pvar,
			SPARQLSelect[RDFTriple[RDFBlankNode[], pvar, RDFBlankNode[]]] /*
			SPARQLDistinct[]
		],
		None :> Function[pvar, {} /; False]
	}
];
getMetadata[es_, HoldPattern[type : Entity[_String]], prop : "PropertyClassesQuery"] := Replace[
	Lookup[getMetadata[es, type], prop, Automatic],
	{
		Automatic :> Function[pcvar,
			With[{pvar = createVar["pvar"]},
				SPARQLSelect[{
					RDFTriple[RDFBlankNode[], SPARQLVariable[pvar], RDFBlankNode[]],
					RDFTriple[SPARQLVariable[pvar], RDFPrefixData["rdf", "type"], pcvar]
				}] /*
				SPARQLProject[First[pcvar]] /*
				SPARQLDistinct[]
			]
		],
		None :> Function[pcvar, {} /; False]
	}
];

getMetadata[es_, HoldPattern[EntityProperty[type_String, pname_String]]] := Lookup[
	getMetadata[es, Entity[type], "Properties"],
	pname,
	<||>
];
getMetadata[es_, HoldPattern[prop_EntityProperty], "CanonicalName"] := CanonicalName[prop];
getMetadata[es_, HoldPattern[prop_EntityProperty], sub : "Label"] := Lookup[
	getMetadata[es, prop],
	"Label",
	labelFromIRI[CanonicalName[prop]]
];
getMetadata[es_, HoldPattern[prop_EntityProperty], sub : "Query"] := Lookup[
	getMetadata[es, prop],
	sub,
	Replace[
		getMetadata[es, prop, "IRI"],
		{
			i_IRI :> (RDFTriple[#1, i, #2] &),
			_ :> If[CanonicalName[prop] === "CanonicalName",
				First[#2] -> #1 &,
				RDFTriple[#1, IRI[CanonicalName[prop]], #2] &
			]
		}
	]
];
getMetadata[es_, HoldPattern[prop_EntityProperty], sub : "ValueProcessingFunction"] := Lookup[
	getMetadata[es, prop],
	sub,
	If[CanonicalName[prop] === "CanonicalName",
		Function[{value, vvar}, value[[1, First[vvar], 1]]],
		Function[{value, vvar}, value[[All, First[vvar]]] // Replace[{} :> Missing["NotAvailable"]]]
	]
];

getMetadata[es_, obj_, entry_String] := Lookup[
	getMetadata[es, obj],
	entry,
	Missing["NotAvailable"]
];


$labelDelimiter = "#" | "/";

clear[labelFromIRI];
labelFromIRI[IRI[i_]] := labelFromIRI[i];
labelFromIRI[i_String] := If[StringEndsQ[i, $labelDelimiter],
	i,
	Last[StringSplit[i, $labelDelimiter]]
];


clear[classToSubquery];
classToSubquery[es_, evar_, HoldPattern[EntityClass[type_, {}]]] := getMetadata[es, Entity[type], "EntitiesQuery"][evar];
classToSubquery[es_, evar_, HoldPattern[EntityClass[_, rules : {__}]]] := Function[{p, pred},
	If[p[[2]] === "CanonicalName",
		pred // Replace[{
			Verbatim[Alternatives][sn__] :> SPARQLValues[First[evar], IRI /@ {sn}],
			_ :> fail[]
		}],
		fail[]
	]
] @@@ rules;
classToSubquery[es_, evar_, HoldPattern[EntityClass[_, name_String]]] := RDFTriple[evar, RDFPrefixData["rdf", "type"], IRI[name]];


clear[propertyToSubquery];
propertyToSubquery[es_, evar_, pvar_, vvar_, HoldPattern[prop_EntityProperty]] := SPARQLOptional[{
	First[pvar] -> CanonicalName[prop],
	getMetadata[es, prop, "Query"][evar, vvar]
}] // stripUnnecessaryOptional[evar, vvar, #] &;


clear[stripUnnecessaryOptional];
stripUnnecessaryOptional[evar_, SPARQLVariable[vvar_], SPARQLOptional[{_ -> _, binding : (Rule | RuleDelayed)[vvar_, evar_]}]] := binding;
stripUnnecessaryOptional[_, _, expr_] := expr;

End[];
EndPackage[];

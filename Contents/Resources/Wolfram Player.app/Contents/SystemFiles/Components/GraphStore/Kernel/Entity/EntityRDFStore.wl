BeginPackage["GraphStore`Entity`EntityRDFStore`", {"GraphStore`", "GraphStore`Entity`"}];
Begin["`Private`"];

EntityRDFStore[args___] := With[{res = Catch[iEntityRDFStore[args], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


clear[rdf];
rdf[s_String] := IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#" <> s];

clear[rdfs];
rdfs[s_String] := IRI["http://www.w3.org/2000/01/rdf-schema#" <> s];


clear[iEntityRDFStore];
iEntityRDFStore[{}, ___] := RDFStore[{}];
iEntityRDFStore[_, {}] := RDFStore[{}];
iEntityRDFStore[HoldPattern[ent : {Entity[_String] ..}]] := RDFStore[Join @@ Function[First[iEntityRDFStore[#]]] /@ ent];
iEntityRDFStore[HoldPattern[ent : {Entity[_String] ..}], prop : {___?Internal`PossibleEntityPropertyQ}] := RDFStore[Join @@ Function[t,
	First[iEntityRDFStore[
		t,
		Join[
			Cases[prop, EntityProperty[First[t], __]],
			Cases[prop, _String]
		]
	]]
] /@ ent];
iEntityRDFStore[ent_, prop_?Internal`PossibleEntityPropertyQ] := iEntityRDFStore[ent, {prop}];
iEntityRDFStore[ent_, prop : Repeated[_, {0, 1}]] := RDFStore[
	First[Last[Reap[
		valueTriples[ent, prop];
		If[{prop} === {},
			labelTriples[ent];
			classesTriples[ent];
		];
		,
		$tripleTag
	]], {}]
];

clear[toTriples];
toTriples[_, _, _?MissingQ] := Null;
toTriples[ent_, prop_, l : {Except[_List] ...}] := Scan[toTriples[ent, prop, #] &, l];
toTriples[ent_, prop_, value_] := (Sow[RDFTriple[ent, prop, value], $tripleTag];);

clear[valueTriples];
valueTriples[ent_, prop : Repeated[_, {0, 1}]] := EntityValue[entityRange[ent], prop, "Association"] // KeyValueMap[Function[{e, data},
	data // KeyValueMap[Function[{p, value},
		toTriples[e, p, value];
	]];
]];

clear[entityRange];
entityRange[HoldPattern[Entity[type_String]]] := EntityClass[type, All];
entityRange[HoldPattern[class_EntityClass]] := class;
entityRange[ent_List] := ent;
entityRange[Except[_List, ent_]] := {ent};

clear[labelTriples];
labelTriples[HoldPattern[ent : Entity[_String]]] := With[
	{objects = Join @@ ent /@ {"Entities", "EntityClasses", "Properties", "PropertyClasses"}},
	{objects, CommonName[objects]} // MapThread[Function[{o, l},
		toTriples[o, rdfs["label"], l];
	]];
];
labelTriples[HoldPattern[ent_Entity]] := toTriples[ent, rdfs["label"], CommonName[ent]];

clear[classesTriples];
classesTriples[HoldPattern[ent : Entity[_String]]] := (
	EntityValue[ent, "EntityClasses"] // Map[Function[c,
		EntityList[c] // Map[Function[e,
			toTriples[e, rdf["type"], c];
		]];
	]];
	EntityValue[ent, "PropertyClasses"] // Map[Function[c,
		EntityProperties[c] // Map[Function[p,
			toTriples[p, rdf["type"], c];
		]];
	]];
);
classesTriples[HoldPattern[ent_Entity]] := toTriples[ent, rdf["type"], ent[EntityProperty[EntityTypeName[ent], "EntityClasses"]]];

End[];
EndPackage[];

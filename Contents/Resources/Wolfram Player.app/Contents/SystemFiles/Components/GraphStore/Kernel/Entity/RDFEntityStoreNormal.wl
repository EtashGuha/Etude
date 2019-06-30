BeginPackage["GraphStore`Entity`RDFEntityStoreNormal`", {"GraphStore`", "GraphStore`Entity`"}];
Begin["`Private`"];

RDFEntityStore /: Normal[store_RDFEntityStore, Repeated[RDFEntityStore, {0, 1}]] := With[{res = Catch[iRDFEntityStoreNormal[store], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


clear[iRDFEntityStoreNormal];
iRDFEntityStoreNormal[store_RDFEntityStore] := EntityStore[
	Function[type,
		type -> DeleteCases[
			<|
				"Entities" -> Module[
					{ents, props, data},
					props = store[
						EntityPropertyClass[type, {}],
						{"CanonicalName"}
					][[All, 1]];
					data = store[
						EntityClass[type, {}],
						EntityProperty[type, #] & /@ Append[props, "CanonicalName"]
					];
					ents = data[[All, -1]];
					data = data[[All, 1 ;; -2]];
					AssociationThread[
						ents,
						AssociationThread[
							props,
							#
						] & /@ data
					]
				]
			|>,
			<||>
		]
	] /@ store[]
];

End[];
EndPackage[];

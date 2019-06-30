(* VOWL: Visual Notation for OWL Ontologies *)
(* http://purl.org/vowl/spec/ *)

BeginPackage["GraphStore`VOWL`", {"GraphStore`"}];

VOWLGraph;

Begin["`Private`"];

VOWLGraph[args___] := With[{res = Catch[iVOWLGraph[args], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


(* vocabulary *)
clear[rdf];
rdf[s_String] := IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#" <> s];

clear[rdfs];
rdfs[s_String] := IRI["http://www.w3.org/2000/01/rdf-schema#" <> s];

clear[owl];
owl[s_String] := IRI["http://www.w3.org/2002/07/owl#" <> s];


(* graphical primitives *)
clear[circle];
circle[owl["Thing"]] := Framed[text["Thing"], Background -> $neutralColor, RoundingRadius -> 10000, FrameStyle -> Dashed];
circle[rdfs["Resource"]] := Framed[text["Resource"], Background -> $rdfColor, RoundingRadius -> 10000, FrameStyle -> Dashed];
circle[s : owl["unionOf"] | owl["intersectionOf"] | owl["complementOf"]] := Framed[setGraphics[s], Background -> $generalColor, RoundingRadius -> 10000, FrameStyle -> Dashed];
circle[sol_, var_] := Framed[Replace[toLabel[sol, var], None :> ""], Background -> toColor[sol, var], RoundingRadius -> 10000];

clear[rectangle];
rectangle["PropertyLabel", sol_, var_] := Framed[toLabel[sol, var], Background -> toColor[sol, var], FrameStyle -> None];
rectangle["Datatype", sol_, var_] := If[KeyExistsQ[sol, var],
	Framed[toLabel[sol, var], Background -> $datatypeColor],
	Framed[text["Literal"], Background -> $datatypeColor, FrameStyle -> Dashed]
];

clear[text];
text[None] := None;
text[x_] := Style[x, $foregroundColor];


(* color scheme *)
$neutralColor := White;
$foregroundColor := Black;
$generalColor := LightBlue;
$rdfColor := LightPurple;
$deprecatedColor := LightGray;
$datatypeColor := Yellow;
$datatypePropertyColor := LightGreen;
$graphicsColor := Darker[$generalColor, .1];


(* set operators *)
clear[setGraphics];
setGraphics[owl["unionOf"]] := Graphics[{
	{$graphicsColor, Opacity[.7], Disk[{-0.05, 0}, 0.1], Disk[{0.05, 0}, 0.1]},
	Text[text["\[Union]"], {0, -0.01}]
}, ImageSize -> 20];
setGraphics[owl["intersectionOf"]] := Graphics[{
	{$graphicsColor, Opacity[.7], Disk[{-0.05, 0}, 0.1], Disk[{0.05, 0}, 0.1]},
	Text[text["\[Intersection]"], {0, -0.01}]
}, ImageSize -> 20];
setGraphics[owl["complementOf"]] := Graphics[{
	{$graphicsColor, Opacity[.7], Disk[{0, 0}, 0.1]},
	Text[text["\[Not]"], {0, -0.01}]
}, ImageSize -> 20];


clear[toLabel];
toLabel[sol_, name_String] := Lookup[
	sol,
	name <> "Label",
	iriToLabel[sol[name]]
] // Replace[RDFString[s_, _] :> s] // text;

clear[iriToLabel];
iriToLabel[IRI[i_String]] := StringSplit[i, "#", 2] // Replace[{
	{_, l_} :> l,
	_ :> Last[StringSplit[i, "/"]]
}];
iriToLabel[x_] := None;


clear[toColor];
toColor[sol_, name_String] := toColor[sol[name <> "Type"]];
toColor[owl["Class"]] := $generalColor;
toColor[rdfs["Class"]] := $rdfColor;
toColor[owl["ObjectProperty"]] := $generalColor;
toColor[owl["DatatypeProperty"]] := $datatypePropertyColor;
toColor[rdf["Property"]] := $rdfColor;
toColor[_] := $neutralColor;


clear[centered];
centered[x_] := Placed[x, Center];


(* class *)
clear[styleClass];
styleClass[sol_] := vertex[sol, "class"];

clear[vertex];
vertex[sol_, var_, rest___] := styleVertex[Lookup[sol, var, owl["Thing"]], sol, var, rest];

clear[styleVertex];
styleVertex[v : owl["Thing"] | rdfs["Resource"], sol_, _, origin_] := vertexProperty[{Lookup[sol, origin, CreateUUID[]]}, VertexLabels -> centered[circle[v]]];
styleVertex[v_, sol_, var_, ___] := vertexProperty[v, VertexLabels -> centered[circle[sol, var]]];

clear[vertexProperty];
vertexProperty[v_, props___] := (
	Sow[Property[v, {props}], $vtag];
	v
);


(* property *)
clear[styleProperty];
styleProperty[sol_] := Sow[Property[
	DirectedEdge[
		vertex[sol, "domain", "range"],
		If[sol["propType"] === owl["DatatypeProperty"],
			vertexProperty[CreateUUID[], VertexLabels -> centered[rectangle["Datatype", sol, "range"]]],
			vertex[sol, "range", "domain"]
		]
	],
	EdgeLabels -> rectangle["PropertyLabel", sol, "prop"]
], $etag];


clear[iVOWLGraph];
iVOWLGraph[store_RDFStore] := Module[
	{vertices, edges},
	{vertices, edges} = Curry[First][{}] /@ Last[Reap[
		store // SPARQLSelect[Alternatives[
			(* classes *)
			{
				classTypeQuery[SPARQLVariable["class"]],
				SPARQLOptional[labelQuery[SPARQLVariable["class"]]]
			},
			(* subclass relations *)
			{
				RDFTriple[SPARQLVariable["class1"], rdfs["subClassOf"], SPARQLVariable["class2"]],
				classInfoQueries[SPARQLVariable["class1"]],
				classInfoQueries[SPARQLVariable["class2"]]
			},
			(* properties *)
			{
				SPARQLSelect[Alternatives[
					propertyTypeQuery[SPARQLVariable["prop"]],
					RDFTriple[SPARQLVariable["prop"], rdfs["domain"], SPARQLVariable["domain"]],
					RDFTriple[SPARQLVariable["prop"], rdfs["range"], SPARQLVariable["range"]]
				] -> "prop", "Distinct" -> True],
				SPARQLOptional[{
					RDFTriple[SPARQLVariable["prop"], rdfs["domain"], SPARQLVariable["domain"]],
					classInfoQueries[SPARQLVariable["domain"]]
				}],
				SPARQLOptional[{
					RDFTriple[SPARQLVariable["prop"], rdfs["range"], SPARQLVariable["range"]],
					classInfoQueries[SPARQLVariable["range"]]
				}],
				SPARQLOptional[{
					propertyTypeQuery[SPARQLVariable["prop"]],
					classInfoQueries[SPARQLVariable["propType"]]
				}],
				SPARQLOptional[labelQuery[SPARQLVariable["prop"]]]
			},
			(* set relations *)
			{
				SPARQLValues["setProp", {owl["unionOf"], owl["intersectionOf"]}],
				RDFTriple[SPARQLVariable["setClass"], SPARQLVariable["setProp"], SPARQLVariable["setClassNode"]],
				SPARQLPropertyPath[SPARQLVariable["setClassNode"], {rdf["rest"] ..., rdf["first"]}, SPARQLVariable["setClassMember"]]
			},
			{
				SPARQLValues["setProp", {owl["complementOf"]}],
				RDFTriple[SPARQLVariable["setClass"], SPARQLVariable["setProp"], SPARQLVariable["setClassMember"]]
			}
		]] // Map[styleObject];
		,
		{$vtag, $etag}
	]];
	(* work around Graph[{Property[1 -> 2, EdgeLabels -> "p1"], Property[1 -> 2, EdgeLabels -> "p2"]}] no displaying the labels "p1" and "p2" *)
	edges = fixEdgeProperties[edges];
	Graph[vertices, edges, VertexShape -> None]
];


clear[fixEdgeProperties];
fixEdgeProperties[edges_List] := Module[
	{labels},
	labels = edges //
	GroupBy[Replace[Property[e_, ___] :> e] -> Replace[Property[_, props___] :> Flatten[{props}]]];
	Join @@ (labels // KeyValueMap[Function[{e, lList},
		ConstantArray[
			Property[
				e,
				EdgeShapeFunction -> Function[With[
					{props = pop[labels[e]]},
					{
						{
							Lookup[props, EdgeStyle, Nothing],
							If[Head[e] === DirectedEdge, Arrow, Identity][BSplineCurve[#]]
						},
						Lookup[props, EdgeLabels, Nothing] // Replace[
							Except[Nothing, l_] :> Text[l, middle[#]]
						]
					}
				]]
			],
			Length[lList]
		]
	]])
];

clear[pop];
SetAttributes[pop, HoldFirst];
pop[expr_] := With[
	{el = Last[expr]},
	expr = Most[expr];
	el
];

clear[middle];
middle[l_List] := Mean[l[[# ;; -#]] &[Ceiling[Length[l]/2]]];


clear[classInfoQueries];
classInfoQueries[var_] := Sequence @@ SPARQLOptional /@ Through[{classTypeQuery, labelQuery}[var]];

clear[classTypeQuery];
classTypeQuery[var : SPARQLVariable[name_String]] := With[
	{typeVar = SPARQLVariable[name <> "Type"]},
	Alternatives[
		{
			RDFTriple[var, rdf["type"], owl["Class"]],
			First[typeVar] -> owl["Class"]
		},
		{
			RDFTriple[var, rdf["type"], rdfs["Class"]] /; ! SPARQLEvaluation["EXISTS"][RDFTriple[var, rdf["type"], owl["Class"]]],
			First[typeVar] -> rdfs["Class"]
		}
	]
];

clear[propertyTypeQuery];
propertyTypeQuery[var : SPARQLVariable[name_String]] := With[
	{typeVar = SPARQLVariable[name <> "Type"]},
	Alternatives[
		{
			RDFTriple[var, rdf["type"], owl["ObjectProperty"]],
			First[typeVar] -> owl["ObjectProperty"]
		},
		{
			RDFTriple[var, rdf["type"], owl["DatatypeProperty"]],
			First[typeVar] -> owl["DatatypeProperty"]
		},
		{
			RDFTriple[var, rdf["type"], rdf["Property"]] /; ! SPARQLEvaluation["EXISTS"][RDFTriple[var, rdf["type"], owl["ObjectProperty"]] | RDFTriple[var, rdf["type"], owl["DatatypeProperty"]]],
			First[typeVar] -> rdf["Property"]
		}
	]
];

clear[labelQuery];
labelQuery[var : SPARQLVariable[name_String]] := With[
	{labelVar = SPARQLVariable[name <> "Label"]},
	SPARQLSelect[RDFTriple[var, rdfs["label"], labelVar]] /*
	SPARQLAggregate[{First[var], First[labelVar]}, var]
];


clear[styleObject];
styleObject[sol : KeyValuePattern["class" -> _]] := styleClass[sol];
styleObject[sol : KeyValuePattern["class1" -> _]] := styleSubclassRelation[sol];
styleObject[sol : KeyValuePattern["setProp" -> _]] := styleSetRelations[sol];
styleObject[sol : KeyValuePattern["prop" -> _]] := styleProperty[sol];


clear[styleSubclassRelation];
styleSubclassRelation[sol_] := Sow[Property[
	DirectedEdge[vertex[sol, "class1"], vertex[sol, "class2"]],
	{
		EdgeLabels -> Style[text["subclass of"], Background -> $neutralColor],
		EdgeStyle -> Dotted
	}
], $etag];

clear[styleSetRelations];
styleSetRelations[sol : KeyValuePattern["setProp" -> setProp_]] := Sow[Property[
	UndirectedEdge[
		vertexProperty[sol["setClass"], VertexLabels -> centered[circle[setProp]]],
		vertex[sol, "setClassMember"]
	],
	{
		EdgeStyle -> Dashed
	}
], $etag];

End[];
EndPackage[];

Package["NeuralNetworks`"]



PackageScope["NetMutationHandler"]

SetHoldAllComplete[NetMutationHandler];
NetMutationHandler[Set[var_Symbol[[parts___]], value_]] := Scope[
	result = NetReplacePart[var, {parts} -> value];
	If[FailureQ[result], 
		ReissueMessage @ result; $Failed,
		var ^= result
	]
];

NetMutationHandler[_] := Language`MutationFallthrough;



PackageExport["NetReplacePart"]
PackageScope["iNetReplacePart"]

Clear[NetReplacePart];

NetReplacePart[net_ ? ValidNetQ, specs_] := CatchFailureAsMessage[NetReplacePart, iNetReplacePart[net, specs]];

iNetReplacePart[NetP[head, assoc, meta], specs_] := Scope[
	$assoc = assoc;
	specs = If[AssociationQ[specs], 
		Normal[specs], 
		Flatten @ ToList @ specs
	];
	Block[{$preserveLVs = !FreeQ[StripCoders[specs], dynamicDimP|All]}, (* %HACK : All is here for "TargetLength" -> All *)
		(* if the spec contains a LengthVar, we want to *change* any integers in the original net to be
		lengthvars (normally ints would win). of course this isn't fully precise, because the spec
		might be a mixture of such cases with the normal situation. too bad. *)
		res = CatchFailure[NetReplacePart, 
			specs = Map[replaceNetPart, specs];
			ConstructWithInference[head, $assoc, meta]
		];
	];

	(* this detects if user is reshaping the input or output, which will require a wiping
	of the entire net's internal shapes so that the change can propagate *) 
	(* the Quiet is to prevent a double messaging with NetEncoders *)
	Quiet @ If[FailureQ[res] && SubsetQ[Union[Keys @ Inputs @ assoc, Keys @ Outputs @ assoc], Keys @ specs],
		res2 = CatchFailure[NetReplacePart,
			$assoc = iNetReshape[assoc, specs];
			ConstructWithInference[head, $assoc, meta]
		];
		If[!FailureQ[res2], res = res2];
	];
	If[FailureQ[res], ThrowRawFailure[res], res]
];
dimensions[asso_] := ReplaceAll[Map[TDimensions, asso], LengthVar[_] :> LengthVar];

NetReplacePart[p:CoderP, spec_] := CatchFailureAsMessage[NetReplacePart, Scope[
	$assoc = <|"Dummy" -> p|>;
	replaceCoderPart[NetPath["Dummy"], spec];
	$assoc["Dummy"]
]];

NetReplacePart::netarg1 = "First argument should be a valid net, NetEncoder, or NetDecoder."
NetReplacePart[_, _] := (Message[NetReplacePart::netarg1]; $Failed);

replaceNetPart[spec_ -> value_, doset_:True] := Scope[
	$spec = spec -> value; (* gives us a chance to rewrite this spec *)
	path = ToNetPath[$assoc, spec];
	(* next section tries to pick up changes to NetEncoders *)
	If[FailureQ[path] && ListQ[spec] && spec =!= {},
		path = ToNetPath[$assoc, First @ spec];
		(* e.g. {"Input", "TargetLength"} -> 5 *)
		If[!FailureQ[path] && MatchQ[$assoc @@ path, CoderP],
			Return[First[spec] -> replaceCoderPart[path, Last[spec] -> value]];
		];
		path = $Failed;
	];
	If[FailureQ[path], ThrowFailure["netnopart", spec]];
	If[RuleVectorQ[value] && MatchQ[$assoc @@ path, CoderP], 
		(* e.g. "Input" -> {"TargetLength" -> 5} *)
		Return[spec -> replaceCoderPart[path, value]]];
	{type, coercion} = extract @ path;
	Replace[path, NetPath[part___] :> Module[{val, val2, sarrays},
		val = checkValue[Take[path, -2], type, coercion @ value];
		If[AssociationQ[val],
			{val2, sarrays} = HoistSharedArrays[<|"Dummy" -> val|>];
			If[Length[sarrays] > 0,
				val = val2["Dummy"];
				Set[$assoc["SharedArrays"],
					JoinSharedArrays[Lookup[$assoc, "SharedArrays", <||>], sarrays]
				]
			];
		];
		If[doset,
			Set[$assoc[part], val],
			$spec = spec -> val;Null
		]
		(* ^ jeromel: doset = True is the regular path, without NetSharedArray.
			In the case of NetSharedArray, this path of code is used only to check the sizes
			and convert the given value (from list to NumericArray typically).
			The output "val" is used in the "SharredArray" property of the layer/net (where each shared arrays is referenced only once).
		*)
	]];
	(* XXXX: PostConstructionFunction
		jeromel @ taliesinb: what?
	*)
	$spec
];

replaceNetPart[NetSharedArray[name_] -> value_] := Scope[
	containerQ = True;
	sharedArrays = Lookup[$assoc, "SharedArrays", <||>];
	If[!KeyExistsQ[sharedArrays, name], ThrowFailure["netnopart", NetSharedArray[name]]];
	
	(* The following calls "replaceNetPart" 
		1) for checks and possible resize needs (see 2nd part of iNetReplacePart)
		2) to get the converted value (to NumericArray, typically) *)
	specs = Map[
		Function @ replaceNetPart[FromNetPath[#] -> value, (* doset = *) False],
		Position[$assoc, NetSharedArray[name], Infinity] /. Key[k_] :> k
	];
	$value = Quiet @ Last @ First[specs];
	Set[$assoc["SharedArrays"], ReplacePart[$assoc["SharedArrays"], name -> $value]];
	Sequence @@ specs
];

(* Change an array into a shared array *)
replaceNetPart[spec_ -> NetSharedArray[name_]] := Scope[
	path = ToNetPath[$assoc, spec];
	If[FailureQ[path], ThrowFailure["netnopart", spec]];
	value = $assoc @@ path;
	If[MatchQ[value, NetSharedArray[_]],
		value = Lookup[$assoc["SharedArrays"], First @ value]
	];
	Set[$assoc["SharedArrays"], JoinSharedArrays[Lookup[$assoc, "SharedArrays", <||>], <|name -> value|>]];
	Replace[path, NetPath[part___] :> 
		Set[$assoc[part], NetSharedArray[name]]
	];
	spec -> value;
];

NetReplacePart::nocoderkey = "`` does not have a parameter ``. Available parameters are ``."
NetReplacePart::immcoderkey = "The `` parameter of `` cannot be changed once created."

replaceCoderPart[path_, spec:(_Rule | {__Rule})] := Scope[
	old = $assoc @@ path; 
	Replace[old, (head:NetEncoder|NetDecoder)[kind_, assoc_, _] :> (
		data = If[head === NetEncoder, $EncoderData, $DecoderData][$kind = kind];
		$coderAssoc = assoc;
		UnpackAssociation[data, parameters, parameterCoercions];
		Scan[procReplaceCoderRule, ToList @ spec];
		new = ReinferCoder[
			head, kind, assoc,
			Complement[$coderAssoc, assoc], (* set of new options values *)
			data @ If[head === NetEncoder, "Output", "Input"]
		];
		Replace[path, NetPath[p__] :> ($assoc[p] ^= new)]
	)];
	new
]

replaceCoderPart[path_, pspec_] := 
	ThrowFailure["invrepspec", QuotedStringForm @ pspec];

procReplaceCoderRule[key_String -> value_] := Scope[
	paramtype = Lookup[parameters, key, ThrowFailure["nocoderkey", 
		CoderFormString @ Evaluate @ old, 
		QuotedStringForm @ key, 
		QuotedStringRow[DiscardStrings[Keys @ parameters, "$*"], " and "]]];
	coercion =  Lookup[parameterCoercions, key, Identity];
	If[key === "IgnoreCase" && MatchQ[$kind, "Characters"|"Tokens"] (* see use of PeekOption["IgnoreCase"] *), 
		(* jeromel: this is a %HACK *)
		(* TODO: create an ImmutableKeys field in coders to make this general *)
		ThrowFailure["immcoderkey", 
			QuotedStringForm @ key,
			CoderFormString @ Evaluate @ old]];
	$coderAssoc[key] = CoerceOption[key, coercion @ value, paramtype];
]

procReplaceCoderRule[pspec_] := 
	ThrowFailure["invrepspec", QuotedStringForm @ pspec];

Clear[checkValue];

checkValue[NetPath[io:"Inputs"|"Outputs", name_], p_, None] := Scope[
	prev = $assoc[io, name];
	If[MatchQ[prev, CoderP], 
		$spec[[2]] = CoderType[prev], 
		prev
	]
];

checkValue[NetPath["Inputs"|"Outputs", name_], type_, Automatic] := $Failed; 
(* ^ this forces the subsequent check to fail which causes iNetReshape to happen *)

checkValue[NetPath["Inputs", name_], type_, value_] :=
	ParseInputSpec[name, type, value];

checkValue[NetPath["Outputs", name_], type_, value_] := 
	ParseOutputSpec[name, type, value];

checkValue[NetPath["Parameters"|"Arrays"|"Nodes", name_], type_, value_] := 
	CoerceParam[name, value, type];

General::invsetpart = "Cannot update the specified subpart."
checkValue[_, _, _] := ThrowFailure["invsetpart"];

extract[NetPath[part___, field:"Inputs"|"Outputs"|"Parameters"|"Arrays", last_String]] := Scope[
	subnet = $assoc[part];
	extractLayerField[subnet["Type"], subnet, field, last]
];

General::netinvrep = "Cannot replace layer with ``, which is not a valid net."
extract[path:NetPath[___, "Nodes", _]] := ModuleScope[
	layer = $assoc @@ path;
	{
		NetT[Inputs[layer], Outputs[layer]],
		Function[
			If[!ValidNetQ[#], ThrowFailure["netinvrep", #]]; 
			TestLayersCompatible[layer, #]; 
			(* ^ we could just rely on CoerceParam, but this generates better
			error messages *)
			#
		]
	}
];

extract[e_] := ThrowFailure["invsetpart"];

extractLayerField["Graph"|"Chain", assoc_, field_, last_] := 
	{assoc[field, last], Identity};

NetReplacePart::immutable = "Cannot change elements of `` once it has been created.";

extractLayerField[type_, assoc_, field_, last_] := (
	If[$LayerData[type, "Immutable"] && !MatchQ[field, "Inputs"|"Outputs"], 
		ThrowFailure[NetReplacePart::immutable, $TypeToSymbol[type]]];
	{
		$LayerData[type, field, last], 
		If[field === "Parameters", Lookup[$LayerData[type, "ParameterCoercions"], last, Identity], Identity]
	}
);

NetReplacePart::invrepspec = "`` is not a valid part replacement spec."
replaceNetPart[pspec_] := ThrowFailure[NetReplacePart::invrepspec, QuotedStringForm @ pspec];

DeclareArgumentCount[NetReplacePart, 2, True];


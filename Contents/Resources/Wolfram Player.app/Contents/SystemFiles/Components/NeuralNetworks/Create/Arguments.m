Package["NeuralNetworks`"]



PackageScope["ParseArguments"]
PackageScope["$CurrentArgumentHead"]

$CurrentArgumentHead = General;
(* ^ various messages will get issued against this, e.g. FailValidation/FailConstruction messages *)


ParseArguments[head_, layerq_, definition_, args_] := Scope[

	$CurrentArgumentHead = head;

	UnpackAssociation[definition,
		$params:"Parameters", 
		$coercions:"ParameterCoercions",
		$defaults:"ParameterDefaults",
		$maxargs:"MaxArgCount", 
		$minargs:"MinArgCount",
		$posargs:"PosArgs"
	];

	If[layerq,
		UnpackAssociation[definition,  
			$arrays:"Arrays", 
			$inputs:"Inputs", 
			$outputs:"Outputs",
			$states:"States"
		],
		$arrays = $inputs = $outputs = $states = <||>;
	];

	argumentRewriter = Lookup[definition, "ArgumentRewriter", None];
	If[argumentRewriter =!= None,
		args = argumentRewriter[args]];	

	$userparams = $sarrays = Association[];
	$args = Keys[$params]; $rawargs = args;
	$i = 0; 
	
	If[$strictLayerArgs && badArgCountQ[head, Length @ Cases[args, Except[_Rule]], $minargs, $maxargs],
		ThrowRawFailure[$Failed]
	];

	FailValidation = FailVParam; 

	Scan[scanArg, args];

	$params = Association @ KeyValueMap[
		#1 -> Lookup[$userparams, #1,
			def = getParamDefault[#1, Null];
			If[def === Null, #2, parseParam[#1, def]]
		]&,
		$params
	];

	ret = If[layerq, {$arrays, $sarrays, $params, $inputs, $outputs, $states}, $params];

	RemapLengthVars[ret]
];

badArgCountQ[head_, c_, min_, max_] := !ArgumentCountQ[head, c, min, max];

badArgCountQ[sf_StringForm, c_, min_, max_] := 
	UnsafeQuietCheck[
		ArgumentCountQ[ZZZ, c, min, max]; False,
		With[{sym = sf[[2]]}, Message[MessageName[sym, "argcnterr"], EvaluateChecked[
			ArgumentCountQ[ZZZ, c, min, max],
			rewriteBadArgMsg[TextString @ sf]
		]]]; True
	];

rewriteBadArgMsg[name_String][failure_] := Scope[
	headstr = SymbolName[head];
	newmsg = StringReplace[
		TextString[failure],
		{"ZZZ" -> name, "called" -> "provided", "argument" -> "parameter"}
	];
	newmsg
];

General::argcnterr = "``";

scanArg[value_] := 
	If[++$i > $maxargs, Panic["TooManyArgs"],
		parseParam[$args[[$i]], value];
	];

General::netinvrule = "`` is not a valid option for ``."
scanArg[r_Rule] := ThrowFailure["netinvrule", r, $CurrentArgumentHead];

scanArg[Rule[sym_Symbol, value_]] /; Context[sym] === "System`" :=
	scanArg[Rule[SymbolName[sym], value]];

scanArg[Rule["Inputs", inputs_]] /; KeyExistsQ[$inputs, $Multiport] := Which[
	ListQ[inputs] && inputs =!= {},
		SetMultiport[$inputs, Length[inputs]];
		IMap[scanArg[IntegerString[#1] -> #2]&, inputs],
	IntegerQ[inputs] && inputs > 0,
		SetMultiport[$inputs, inputs];
		Do[scanArg[IntegerString[i] -> Automatic]&, {i, inputs}],
	True,
		ThrowFailure["netinvopt", "Inputs" -> inputs, "positive integer or non-empty list"]
];

PackageScope["SetCurrentLayerInputs"]
PackageScope["SetCurrentLayerOutputs"]
PackageScope["SetCurrentLayerStates"]
PackageScope["SetCurrentLayerParameters"]

SetCurrentLayerInputs[in_] := $inputs = in;
SetCurrentLayerOutputs[out_] := $outputs = out;
SetCurrentLayerStates[states_] := $states = states;
SetCurrentLayerParameters[params_] := AssociateTo[$params, params];

scanArg[Rule[key_String, value_]] := 
	Which[
		KeyExistsQ[$params, key],
			parseParam[key, value];,
		KeyExistsQ[$arrays, key],
			If[MatchQ[value, NetSharedArray[_String]],
				$arrays[key] = value;
				$sarrays[First[value]] = RealTensorT;
			,
				$arrays[key] = CoerceParam[key, value, $arrays[key]]
			],
		KeyExistsQ[$inputs, key],
			$inputs[key] = ParseInputSpec[key, $inputs[key], value];,
		KeyExistsQ[$outputs, key],
			$outputs[key] = ParseOutputSpec[key, $outputs[key], value];,
		key === "Input" && KeyExistsQ[$inputs, $Multiport],
			(* for backward compatibility of multiport layer construction code *)
			scanArg["Inputs" -> value],
		True,
			panicParamKey[key]
	];

getParamDefault[key_, else_] :=
	Replace[Lookup[$defaults, key, else], HeldDefault[d_] :> d];

parseParam[key_, value_] := Scope[
	If[MemberQ[$posargs, key], 
		index = IndexOf[$posargs, key];
		$label = StringForm["the `` (``)", ToLowerCase[deCamelCase[key]], PartForm[index, "argument"]];
		coercer = CoerceParam;
	,
		$label = key;
		coercer = CoerceOption;
	];
	type = $params[key];
	$value = value;
	If[$value === Automatic && FreeQ[type, Automatic], $value = getParamDefault[key, Automatic]];
	(* ^ if user provided Automatic, but type doesnt allow automatic, try find the default *)
	$value = Lookup[$coercions, key, Identity] @ $value;
	$userparams[key] = coercer[$label, $value, type]
];

(* this will give a better message when FailValidation is called during layer construction,
because the message will be attached to the actual parameter name. *)
General::netinvparam2 = "Value of `` given for `` was invalid: ``";
FailVParam[_Symbol, reason_String, args___] := 
	ThrowFailure["netinvparam2", $value, $label, fromStringForm @ StringForm[reason, args]];


PackageScope["PeekOption"]

PeekOption[key_] := Lookup[
	Cases[$rawargs, r_Rule :> MapAt[ToString, r, 1]], 
	key, Lookup[$defaults, key, $Failed]
];


PackageScope["ParseInputSpec"]

Clear[ParseInputSpec, ParseOutputSpec];

ParseInputSpec[param_, type_, spec_] := 
	Block[{$name = param, $type = type, $spec = spec, $preserveCoders = True}, 
		RemapCoderLengthVars @ checkType[StripCoders @ type, ToT[spec, NetEncoder]]
	];


PackageScope["ParseOutputSpec"]

ParseOutputSpec[param_, type_, spec_] := 
	Block[{$name = param, $type = type, $spec = spec, $preserveCoders = True},
		RemapCoderLengthVars @ checkType[StripCoders @ type, ToT[spec, NetDecoder]]
	]


Clear[checkType, checkType2];

checkType[t1_TensorT, p:CoderP /; !SequenceCoderQ[p]] := Scope[
	res = checkType2[t1, p];
	(* automatically raise type descriptions that aren't sequence-level to be sequence-level *)
	If[FailureQ[res], res = checkType2[t1, TensorT[{SizeT}, p]]];
	If[FailureQ[res], panicIOType[t1, p]];
	res
];

General::netunkname = "Cannot find `` in ``.";

checkType[_Missing, _] := ThrowFailure["netunkname",
	QuotedStringForm @ $name,
	$CurrentArgumentHead
];


checkType[t1_, t2_] := Scope[
	res = checkType2[t1, t2];
	If[FailureQ[res], panicIOType[t1, t2]];
	res
];

checkType2[type_, utype_] := Scope[
	(* %GENERALITY we work around no reasoning about variance of array inputs / outputs by just weakening the user type *)
	If[MatchQ[TType[type], _IndexIntegerT], utype = utype /. RealT -> AtomT];
	res = UnifyTypes[type, utype]; 
	res
];

General::invspenc = "`` producing ``, cannot be attached to port ``, which must be ``."
panicIOType[type_, enc_NetEncoder] :=
	ThrowFailure["invspenc", CoderForm[enc], TypeString[CoderType[enc]], $name, TypeString[type]];

General::invspdec = "`` taking ``, cannot be attached to port ``, which produces ``."
panicIOType[type_, dec_NetDecoder] :=
	ThrowFailure["invspdec", CoderForm[dec], TypeString[CoderType[dec]], $name, TypeString[type]];

General::invportshape = "Specification `` is not compatible with ``, which must be ``."
panicIOType[type_, u_] := ThrowFailure["invportshape", CoderForm[$spec], FmtPortName[$name], TypeString[type]];

General::netunkparam = "`` is not a known parameter for ``. Allowed parameters include: ``.";

panicParamKey[key_] :=
	ThrowFailure["netunkparam", 
		QuotedStringForm @ key,
		$CurrentArgumentHead, 
		QuotedStringRow[
			Select[
				Join[Keys @ $params, Keys @ $arrays, Keys @ $inputs],
				StringFreeQ["$"]
			],
			" and "
		]
	]
(* Mathematica package *)
Begin["Tools`MathematicalFunctionData`Private`"]

(*-------------- clear variables --------------------------------------------------------------------*)
(* Initialization *)
$ProtectedSymbols = {
	System`MathematicalFunctionData
};
Unprotect@@$ProtectedSymbols;
Clear@@$ProtectedSymbols;
$tag = "MathematicalFunctionDataCatchThrowTag";

ClearAll[MathematicalFunctionData, iMathematicalFunctionData]
ClearAll[MFDEntPattern, MFDEntListPattern, MFDEntClassPattern, MFDPropPattern, MFDPropListPattern, MFDPropClassPattern]
ClearAll[MFDEntGroupPattern, MFDPropGroupPattern, MFDStringReplace, MFDQualReplace, MFDArgReplace]
ClearAll[MFDEntities, MFDProperties, MFDEntityClasses, MFDPropertyClasses]
ClearAll[MFDSingleEntFormatStrings, MFDSinglePropFormatStrings, MFDSingleEntFormatPattern, MFDSinglePropFormatPattern]
ClearAll[MFDFormatStrings, MFDFormatStringPattern, MFDSubpropertyStrings, MFDSubpropertyStringPattern]
ClearAll[MFDGeneralInfoStrings, MFDAllowedStrings, MFDAllowedStringPattern]
ClearAll[MFDTagData, MFDLookup]
ClearAll[checkArgsAndIssueMessages, checkEntityAndIssueMessages, checkPropertyAndIssueMessages,
	checkThirdArgAndIssueMessages, checkFourthArgAndIssueMessages]

(*-------------- helper definitions ---------------------------------------------------*)
(* Define some useful lists *)
MFDEntities := MFDEntities = EntityValue["MathematicalFunction", "Entities"]
MFDProperties := MFDProperties = EntityValue["MathematicalFunction", "Properties"]
MFDEntityClasses := MFDEntityClasses = EntityValue["MathematicalFunction", "EntityClasses"]
MFDPropertyClasses := MFDPropertyClasses = EntityValue["MathematicalFunction", "PropertyClasses"]
MFDSingleEntFormatStrings= {"PropertyAssociation", "NonMissingProperties", "NonMissingPropertyAssociation"}
MFDSinglePropFormatStrings= {"EntityAssociation", "NonMissingEntities", "NonMissingEntityAssociation"}
MFDFormatStrings = Join[
	MFDSingleEntFormatStrings, MFDSinglePropFormatStrings,
	{"EntityPropertyAssociation", "PropertyEntityAssociation", "Dataset"}
]
MFDSingleEntFormatPattern = Alternatives @@ MFDSingleEntFormatStrings
MFDSinglePropFormatPattern = Alternatives @@ MFDSinglePropFormatStrings
MFDFormatStringPattern = Alternatives @@ MFDFormatStrings
MFDSubpropertyStrings = {"Description", "Label", "Qualifiers"}
MFDSubpropertyStringPattern = Alternatives @@ MFDSubpropertyStrings
MFDEntityInfoStrings = {"Entities", "EntityCount", "EntityCanonicalNames"}
MFDPropertyInfoStrings = {"Properties", "PropertyCount", "PropertyCanonicalNames"}
MFDGeneralInfoStrings = Join[
	MFDEntityInfoStrings, MFDPropertyInfoStrings,
	{
		"SampleEntities", "SampleEntityClasses", 
		"EntityClasses", "EntityClassCount", "EntityClassCanonicalNames",
		"PropertyClasses", "PropertyClassCount","PropertyClassCanonicalNames",
		"RandomEntity", "RandomEntities", "RandomEntityClass", "RandomEntityClasses"
	}
]
MFDAllowedStrings = Join[MFDGeneralInfoStrings, MFDSubpropertyStrings, MFDFormatStrings]
MFDAllowedStringPattern = Alternatives @@ MFDAllowedStrings
MFDTagData = EntityValue[Entity["MathematicalFunction", "Abs"], EntityProperty["MathematicalFunction", "AdditionFormulas", {"CrossReferences" -> "FullListing"}]]
MFDCleanedDLMFTags = Module[
	{res},
	res = ReplaceAll[MFDTagData, {{ent_String , prop_String}, rules : {_Rule ..}} :> ({ent, prop, ##} & @@@ MapAt[Select[#, MatchQ[#, "DLMF" -> _]&]&, rules, {All, 2}])] // Flatten[#, 1] &;
	res = res /. {ent_, prop_, num_, {"DLMF" -> linkList_List}} :> ({ent, prop, num, #}& /@ linkList) // Flatten[#, 1]&;
	res = ReplaceAll[res, {ent_, prop_, num_, link_} :> (
			link -> Inactive[Part][Inactive[EntityValue][Entity["MathematicalFunction", ent], EntityProperty[ "MathematicalFunction", prop]], num]
	)]
]

(* Set up some entity, property, and qualifier (and groupings thereof) patterns *)
MFDEntPattern = Entity["MathematicalFunction", _String] | EntityInstance[Entity["MathematicalFunction", _String], _]
MFDEntListPattern = {MFDEntPattern ..}
MFDEntClassPattern = EntityClass["MathematicalFunction", _String]
MFDPropPattern = EntityProperty["MathematicalFunction", _String] | EntityProperty["MathematicalFunction", _, {(_Rule | _RuleDelayed) ..}]
MFDPropListPattern = {MFDPropPattern ..}
MFDPropClassPattern = EntityPropertyClass["MathematicalFunction", _String]
MFDEntGroupPattern = MFDEntPattern | MFDEntListPattern | MFDEntClassPattern
MFDPropGroupPattern = MFDPropPattern | MFDPropListPattern | MFDPropClassPattern
MFDQualGroupPattern = {(_Rule | _RuleDelayed) ...} | (_Rule | _RuleDelayed) ..

(* Set up replacement rules to canonicalize Entity/Property groupings *)
MFDStringReplace[input_String, messageTag_String] := Module[
	{
		result
	},
	If[MatchQ[input, MFDAllowedStringPattern], Return[input, Module]];
	result = Join[
		Cases[MFDEntities, _?(ToLowerCase[#[[2]]] === ToLowerCase[input] &), {1}],
		Cases[MFDProperties, _?(ToLowerCase[#[[2]]] === ToLowerCase[input] &), {1}],
		Cases[MFDEntityClasses, _?(ToLowerCase[#[[2]]] === ToLowerCase[input] &), {1}],
		Cases[MFDPropertyClasses, _?(ToLowerCase[#[[2]]] === ToLowerCase[input] &), {1}],
		Cases[MFDAllowedStrings, _?(ToLowerCase[#] === ToLowerCase[input] &), {1}]
	];
	If[ result === {}, (* then give original input back *) input, (* else *) result[[1]]]
]
MFDStringReplace[input_List, messageTag_String] :=
	input //. {{first___, n_String, rest___} :> {first, MFDStringReplace[n, messageTag], rest}}
MFDStringReplace[input_String] := MFDStringReplace[input, "notent"]
MFDStringReplace[input_List] := MFDStringReplace[input, "notent"]
MFDStringReplace[input_, _] := input
MFDClassResolve[input_]:= 
	input //. {ec_EntityClass :> EntityValue[ec, "Entities"], pc_EntityPropertyClass :> EntityValue[pc, "Properties"]}

(* Code to combine properties with qualifiers *)
MFDQualReplace[prop : MFDPropPattern, quals : MFDQualGroupPattern] := Module[
	{ 
		validQuals = EntityValue[prop, "Qualifiers"],
		badQuals
	},
	badQuals = Complement[Flatten[{quals}][[All, 1]], validQuals];
	If[ badQuals =!= {}, Message[MathematicalFunctionData::badqual, quals, prop]; Return[prop]];
	Switch[prop,
		EntityProperty["MathematicalFunction", _String], Append[prop, {quals} // Flatten],
		EntityProperty["MathematicalFunction", _String, MFDQualGroupPattern], prop /. n : {(_Rule | _RuleDelayed) ...} -> Join[n, Flatten[{quals}]],
		_String, prop
	]
]
MFDQualReplace[pg : MFDPropListPattern | MFDPropClassPattern, quals : MFDQualGroupPattern] := MFDQualReplace[#, quals] & /@ MFDClassResolve[pg]

(* DLMFLookup *)
MFDLookup[input_List]:= Module[
	{
		lookupRule= First[input],
		lookupTag, lookupValue, res
	},
	Switch[First[lookupRule],
		"DigitalLibraryOfMathematicalFunctions"|"DLMF"|Entity["Source", "DigitalLibraryOfMathematicalFunctions"], lookupTag = "DLMF"; lookupValue = Last[lookupRule],
		_, Message[MessageName[MathematicalFunctionData, "notent"], input // FullForm, MathematicalFunctionData]; Throw[$Failed, $tag]
	];
	If[ lookupTag === "DLMF",
		(*then*)
		res = Select[MFDCleanedDLMFTags, StringMatchQ[#[[1]], lookupValue ~~ ___]&],
		(*else*)
		_, Message[MessageName[MathematicalFunctionData, "notent"], input // FullForm, MathematicalFunctionData]; Throw[$Failed, $tag]
	];
	If[res === {}, res, MapAt[Hyperlink["DLMF eq. " <> #, "http://dlmf.nist.gov/"<> StringReplace[#, LetterCharacter ~~ EndOfString :> ""]]&, res, {All, 1}]]
]

(* Canonicalize initial argument structure *)
MathematicalFunctionData::"unrecog" = "Unrecognized argument `1` in evaluation of `2`."
MFDArgReplace[args___]:= Module[
	{
		input, output, attemptToEntity
	},
	attemptToEntity[expr_] := Module[
		{toEntAttempt},
		toEntAttempt = Quiet[ToEntity[expr, "MathematicalFunction"]]; 
		If[Head[toEntAttempt] === Entity, toEntAttempt, expr]
	];
	attemptToEntity[exprList_List]:=attemptToEntity /@ exprList;
	If[ Length[{args}] === 0, input = {args}, input = {attemptToEntity[First[{args}]]}~Join~Rest[{args}] ];
	output = Switch[ input,
		{}, Return[{}],
		{_Rule, ___}, output = MFDLookup[input]; Throw[output, $tag],
		{_}, output = MFDStringReplace[input, "notent"],
		{_, _}, output = MapThread[MFDStringReplace, {input, {"notent", "notprop"}}],
		{_, _, __}, output = MapThread[MFDStringReplace, {input, {"notent", "notprop"}~Join~Table["unrecog", {Length[{args}]-2}]}]
	];
	output = Replace[output, {s: MFDAllowedStringPattern, qg: MFDQualGroupPattern} :> {MFDQualReplace[MFDProperties, qg], s}];
	output = Replace[output, {qg: MFDQualGroupPattern, s: MFDAllowedStringPattern} :> {MFDQualReplace[MFDProperties, qg], s}];
	output = Replace[output, {first_, qg: MFDQualGroupPattern} :> {first, MFDQualReplace[MFDProperties, qg]}];
	output = Replace[output, {first_, s: MFDAllowedStringPattern, qg: MFDQualGroupPattern} :> {first, MFDQualReplace[MFDProperties, qg], s}];
	output = Replace[output, {first_, qg: MFDQualGroupPattern, s: MFDAllowedStringPattern} :> {first, MFDQualReplace[MFDProperties, qg], s}];
	output = Replace[output, {first___, pg: MFDPropGroupPattern, qg: MFDQualGroupPattern} :> {first, MFDQualReplace[pg, qg]}];
	output = Replace[output, {first___, pg: MFDPropGroupPattern, s: MFDAllowedStringPattern, qg: MFDQualGroupPattern} :> {first, MFDQualReplace[pg, qg], s}];
	output = Replace[output, {first___, pg: MFDPropGroupPattern, qg: MFDQualGroupPattern, s: MFDAllowedStringPattern} :> {first, MFDQualReplace[pg, qg], s}];
	output
]

(*-------------- define main function downvalues ---------------------------------------------------*)
(* Error-handling *)
MathematicalFunctionData[args___]:= Module[
	{
		newArgList = Catch[MFDArgReplace[args], $tag]
	},
	If[MatchQ[newArgList, {(_Hyperlink -> _)..}], Return[newArgList]];
	MathematicalFunctionData@@newArgList /; newArgList =!= {args} && newArgList =!= $Failed
]
MathematicalFunctionData[args___] := Module[
	{
		res,
		newArgList = Catch[MFDArgReplace[args], $tag]
	},
	Switch[newArgList,
		$Failed, res = $Failed,
		_, res = Catch[iMathematicalFunctionData @@ newArgList, $tag]
	];
	res /; newArgList === {args} && res =!= $Failed
]

iMathematicalFunctionData[args___] := CompoundExpression[
	Check[System`Private`Arguments[MathematicalFunctionData[args], {0, 4}], Throw[$Failed, $tag]],
	checkArgsAndIssueMessages[args],
	Throw[$Failed, $tag]
]

checkArgsAndIssueMessages[ent_] := checkEntityAndIssueMessages[ent]
checkArgsAndIssueMessages[ent_, prop_] := CompoundExpression[
	checkEntityAndIssueMessages[ent],
	checkPropertyAndIssueMessages[prop]
]
checkArgsAndIssueMessages[ent_, prop_, thirdArg_] := CompoundExpression[
	checkEntityAndIssueMessages[ent],
	checkPropertyAndIssueMessages[prop],
	checkThirdArgAndIssueMessages[thirdArg]
]
checkArgsAndIssueMessages[ent_, prop_, thirdArg_, fourthArg_] := CompoundExpression[
	checkEntityAndIssueMessages[ent],
	checkPropertyAndIssueMessages[prop],
	checkThirdArgAndIssueMessages[thirdArg],
	checkFourthArgAndIssueMessages[fourthArg]
]

checkEntityAndIssueMessages[ent_] := If[
	!MatchQ[ent, Alternatives[MFDEntGroupPattern, MFDPropGroupPattern, All, Alternatives@@MFDGeneralInfoStrings, {"RandomEntities"|"RandomEntityClasses", _Integer}
		]],
	Message[MathematicalFunctionData::notent, ent, MathematicalFunctionData]
]

checkPropertyAndIssueMessages[prop_] := If[
	!MatchQ[prop, Alternatives @@ Union[
			MFDEntityInfoStrings, MFDPropertyInfoStrings,
			MFDSubpropertyStrings, MFDFormatStrings,
			{MFDPropGroupPattern, All}
		]],
	Message[MathematicalFunctionData::notprop, prop, MathematicalFunctionData]
]
checkThirdArgAndIssueMessages[arg_] := If[
		!MatchQ[arg, MFDFormatStringPattern],
		Message[MathematicalFunctionData::notform, arg, "third"]
]
checkFourthArgAndIssueMessages[arg_] := Message[MathematicalFunctionData::notform, arg, "fourth"]

MathematicalFunctionData::badqual = "`1` is not a valid qualifier specification for the property `2`.";
MathematicalFunctionData::notform = "`1` is not a valid `2` argument in MathematicalFunctionData.";

(* general domain info *)
iMathematicalFunctionData[] := MFDEntities
(iMathematicalFunctionData[#] := EntityValue["MathematicalFunction", #])& /@ MFDGeneralInfoStrings

(* single-argument *)
iMathematicalFunctionData[entOrList: MFDEntPattern | MFDEntListPattern]:= entOrList
iMathematicalFunctionData[ec: MFDEntClassPattern]:= EntityValue[ec, "Entities"]
iMathematicalFunctionData[propOrList: MFDPropPattern | MFDPropListPattern]:= propOrList
iMathematicalFunctionData[pc: MFDPropClassPattern]:= EntityValue[pc, "Properties"]

(* single-argument plus GeneralInfo string *)
iMathematicalFunctionData[eg : MFDEntGroupPattern, "Entities"] := eg // MFDClassResolve
iMathematicalFunctionData[eg : MFDEntGroupPattern, "EntityCount"] := MFDClassResolve[eg] // If[MatchQ[#, _List], Length[#], 1] &
iMathematicalFunctionData[eg : MFDEntGroupPattern, "EntityCanonicalNames"] := MFDClassResolve[eg] // If[MatchQ[#, _List], Part[#, All, 2], Part[#, 2]] &
iMathematicalFunctionData[pg : MFDPropGroupPattern, "Properties"] := pg // MFDClassResolve
iMathematicalFunctionData[pg : MFDPropGroupPattern, "PropertyCount"] := MFDClassResolve[pg] // If[MatchQ[#, _List], Length[#], 1] &
iMathematicalFunctionData[pg : MFDPropGroupPattern, "PropertyCanonicalNames"] := MFDClassResolve[pg] // If[MatchQ[#, _List], Part[#, All, 2], Part[#, 2]] &

(* single-argument plus format string *)
iMathematicalFunctionData[ent: MFDEntPattern, format: MFDSingleEntFormatPattern]:= EntityValue[ent, format]
iMathematicalFunctionData[eg: MFDEntListPattern | MFDEntClassPattern, format: MFDFormatStringPattern]:= EntityValue[MFDClassResolve[eg], format]
iMathematicalFunctionData[prop: MFDPropPattern, format: MFDSinglePropFormatPattern]:= EntityValue[MFDEntities, prop, format]
iMathematicalFunctionData[pg: MFDPropListPattern | MFDPropClassPattern, format: MFDFormatStringPattern]:= EntityValue[MFDEntities, MFDClassResolve[pg], format]

(* Entity/Property stuff *)
(* E-P pairs alone*)
iMathematicalFunctionData[eg: MFDEntGroupPattern, pg: MFDPropGroupPattern] := EntityValue[eg, pg]
(* explicit E-P's plus format string*)
iMathematicalFunctionData[ent: MFDEntPattern, props: MFDPropListPattern | MFDPropClassPattern, format: MFDSingleEntFormatPattern] := EntityValue[ent, props, format]
iMathematicalFunctionData[ents: MFDEntListPattern | MFDEntClassPattern, prop: MFDPropPattern, format: MFDSinglePropFormatPattern] := EntityValue[ents, prop, format]
iMathematicalFunctionData[ents: MFDEntListPattern | MFDEntClassPattern, props: MFDPropListPattern | MFDPropClassPattern, format: MFDFormatStringPattern] := EntityValue[ents, props, format]

(* Subproperty stuff *)
iMathematicalFunctionData[pg: MFDPropGroupPattern, subprop: MFDSubpropertyStringPattern]:= EntityValue[pg, subprop]

(* specified-number random entity/class stuff... unspecified case handled under GeneralInfo *)
iMathematicalFunctionData[input: {"RandomEntities", n_Integer}]:= EntityValue["MathematicalFunction", input]
iMathematicalFunctionData[input: {"RandomEntityClasses", n_Integer}]:= EntityValue["MathematicalFunction", input]

(* "All" stuff *)
iMathematicalFunctionData[All, args___]:= iMathematicalFunctionData[MFDEntities, args]
iMathematicalFunctionData[ents: MFDEntGroupPattern, All, rest___]:= iMathematicalFunctionData[ents, MFDProperties, rest]

(* Postscript *)
With[{symbols = $ProtectedSymbols},(*SetAttributes is HoldFirst*)
	SetAttributes[symbols, {ReadProtected}]
];
Protect@@$ProtectedSymbols;

End[];
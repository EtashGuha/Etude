Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["MXOperatorInvoke"]

SetUsage @ "
MXOperatorInvoke[name$, input$, output$, params$] applies the op name$ on a list of input NDArrays, \
and puts the output into a list of output NDArrays output$, and given an association of parameters params$
"

mxlDeclare[mxlMXImperativeInvoke, {"String", "IntegerVector", "String", "String", "IntegerVector"}];

MXOperatorInvoke[name_, inputs_, outputs_, params_] := 
	mxlCall[
		mxlMXImperativeInvoke, 
		name, 
		toIDs @ inputs, 
		mxlPackStringVector @ Keys @ params,
		mxlPackStringVector @ Map[mxParameterToString, Values @ params],
		toIDs @ outputs
	];

toIDs[input:{__NDArray}] := ManagedLibraryExpressionID /@ input;
toIDs[input:{___Integer}] := input;
toIDs[e_] := Panic["BadOperatorIDs", "`` is not a list of integers or a list of NDArrays.", e]

MXOperatorInvoke[___] := $Unreachable;

(******************************************************************************)

PackageExport["$MXOperatorNames"]

SetUsage @ "
$MXOperatorNames gives a list of all operators in the current version of MXNet."

mxlDeclare[mxlMXListAllOpNames, {}, "String"];

$MXOperatorNames := $MXOperatorNames = 
	mxlUnpackStringVector @ mxlCall[mxlMXListAllOpNames];

(******************************************************************************)

PackageExport["$MXOperatorData"]

SetUsage @ "
$MXOperatorData gives an association whose keys are the available MXNet ops and \
values are associations providing further information."

mxlDeclare[mxlMXSymbolGetAtomicSymbolInfo, {"String"}, "String"];

$MXOperatorData := $MXOperatorData = Discard[
	AssociationMap[procOpInfo, $MXOperatorNames],
	StringContainsQ[#Description, "`".. ~~ #Name ~~ "`".. ~~ " is deprecated"]&
]

procOpInfo[name_] := Scope[
	data = Developer`ReadRawJSONString @ mxlMXSymbolGetAtomicSymbolInfo[name];
	data["Arguments"] = Association @ Map[procArgInfo, data["Arguments"]];
	data["Name"] = name;
	data
];

procArgInfo[argInfo_Association] := Scope[	
	
	argInfo = argInfo;

	typeString = argInfo["Type"];

	optional = StringContainsQ[typeString, "optional"]; 
	(* check whether arg is optional *)

	typeString = StringSplit[typeString, {",required", ", required", ", optional", ",optional"}, 2];
	type = StringReplace[First[typeString, "missing"], "-or-" -> "/"];
	(* deal with certain ops (like _Native) that have no type *)

	argInfo["Default"] = If[optional, parseMXDefaults @ typeString, None];
	argInfo["Optional"] = optional;
	argInfo["Type"] = type;
	
	argInfo["Name"] -> argInfo
]

parseMXDefaults[defaultString_String] := Scope[
	(* Get only default part *)
	str = First @ StringCases[defaultString, "default=" ~~ x___ -> x, 1];
	str = StringDelete[str, "'"];
	str = StringReplace[str, {"("-> "{", ")"-> "}"}];
	(* Incase MXNet changes format, immediately fail *)
	If[!StringQ[str], Panic["UnableToParse"]];
	str
]


BeginPackage["Compile`API`Utilities`"]

checkFilePath

compileExportException

makeFailure

getCompilerOptions

compileArgumentError

checkFunctionForExport

Begin["`Private`"]



checkFilePath[ data_, pathIn_String /; StringLength[StringTrim[pathIn]] > 0] :=
	Module[ {path = ExpandFileName[pathIn], dir},
		dir = FileNameDrop[ path, -1];
		If[ !FileExistsQ[dir],
			With[ {h = data["head"]},
				Message[h::pathw, pathIn];
				Throw[Null, compileExportException[makeFailure[$Failed]]]]
		];
		path
	]
	
checkFilePath[ data_, path_] :=
	With[ {h = data["head"]},
		Message[h::path, path];
		Throw[Null, compileExportException[Null]]]

makeFailure[ $Failed] :=
	$Failed
	
makeFailure[ x_?FailureQ] :=
	x

makeFailure[ x_] :=
	$Failed	

General::compopts = "CompilerOptions setting `1` is not a rule or a list of rules."

getCompilerOptions[ head_, ex_, opts_] :=
	Module[{compOpts = OptionValue[head, opts,CompilerOptions]},
		If[ compOpts === Automatic,
			compOpts = {}];
		If[!OptionQ[compOpts],
			Message[head::compopts, compOpts];
			Throw[Null, ex[Null]]];
		compOpts
	]



argumentLength[{args___, opts:OptionsPattern[]}] :=
	Length[{args}]
	
compileArgumentError[ args_List, head_, expLen_] :=
	Module[ {len = argumentLength[args]},
		Which[
			len > expLen,
				Message[ head::nonopt, Part[args,len], expLen, HoldForm[head]@@args],
			expLen === 1 && len < expLen,
			  Message[ head::argx, head, len],
			len < expLen,
				Message[ head::argr, head, expLen],
			True,
				Null]		
	]


compileArgumentError[ args_List, head_, expLenMin_, expLenMax_] :=
	Module[ {len = argumentLength[args]},
		Which[
			expLenMin <= len && len <= expLenMax,
				Null,
			len === 1 && len < expLenMin,
				Message[ head::argbu, head, expLenMin, expLenMax],
			len < expLenMin,
				Message[ head::argb, head, len, expLenMin, expLenMax],
			len > expLenMax,
				Message[ head::nonopt, Part[args,len], expLenMax, HoldForm[head]@@args],
			True,
				Message[ head::argr, head, len]]		
	]


(*
  If it looks like a valid function then return else raise error.
*)
checkFunctionForExport[data_, func_Function] :=
	If[MatchQ[func, HoldPattern[Function][_,_]],
			func
			,
			raiseFunctionError[data, func]]

(*
  If it is a CompiledCodeFunction extract the input check it is valid and return else report an error and raise exception.
*)
checkFunctionForExport[data_, ccf:HoldPattern[CompiledCodeFunction][ccfData_?AssociationQ, __]] :=
	Module[{input = Lookup[ccfData, "Input", Null]},
		If[MatchQ[input, HoldPattern[Function][_,_]],
			input
			,
			With[ {h = data["head"]},
				Message[h::ccfinp, ccf];
				Throw[Null, compileExportException[Null]]
			]
		]
	]

(*
  Not recognized,  raise an error.
*)
checkFunctionForExport[data_, func_] :=
	raiseFunctionError[data, func]

(*
  Raise an error.  Issue a message and throw.  
  The payload of Null makes sure no more message comes out.
*)
raiseFunctionError[data_, func_] :=
	With[ {h = data["head"]},
		Message[h::fun, func];
		Throw[Null, compileExportException[Null]]
	]



End[]

EndPackage[]


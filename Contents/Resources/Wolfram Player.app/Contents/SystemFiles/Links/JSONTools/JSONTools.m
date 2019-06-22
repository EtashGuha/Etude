(* ::Package:: *)

(*
	A Mathematica Package that handles 
	JSON encoding and decoding.
	
	@author Marlon Chatman <marlonc@wolfram.com>
*)
BeginPackage["JSONTools`"]

ToJSON::usage = "ToJSON[expression]	Encodes a given expression into it's corresponding JSON representation";
FromJSON::usage = "FromJSON[JSON] Decodes a given JSON string into a expression \n
FromJSON[file] Decodes a given JSON file's contents into a expression";

Begin["`Private`"]



(* ::Section:: *)
(* Internal Variables *)
(******************************************************************************)


(* ::Subsection:: *)
(* The name of the dynamic library filename *)


If[$OperatingSystem === "iOS", (* on iOS, hyphens are not allowed in library names, because they're not valid C99 identifiers *)
	$DynamicLibrary = "json_link",
	$DynamicLibrary = "json-link"
];

$MessageHead = General;

(* Evaluate, in order to detirmine the path of the dynamic library  *)
$LibraryDirectory = FileNameJoin[{DirectoryName[$InputFileName], "LibraryResources", $SystemID}];

cSetOptions = Null;
cEncodeInt = Null;
cEncodeReal = Null;
cEncodeBool = Null;
cEncodeString = Null;
cEncodeBoolList = Null;
cEncodeIntList = Null;
cEncodeRealList = Null;
cEncodeExpr = Null;
cDecodeExpr = Null;

PRINTPRECISION = 1;
ALLOWSYMBOLS = 2;
INDENT = 3;
COMPACT = 4;
ENSUREASCII = 5;
SORTKEYS = 6;
PRESERVEORDER = 7;
ENCODEANY = 8;
ESCAPESLASH = 9;
REJECTDUPLICATES = 10;
DECODEANY = 11;
DISABLEEOFCHECK = 12;
DECODEINTASREAL = 13;

(*****************************************************************************************)



(* ::Section:: *)
(* Options/Attributes *)
(******************************************************************************)


(* ::Subsection:: *)
(**)


Options[FromJSON]={ 
	(*
		By default, the decoder expects an array or object as the input. 
		With this flag enabled, the decoder accepts any valid JSON value or type.
	*)
	"StrictDecoding"-> True,
	(*
		 By default, the decoder expects that its whole input constitutes
		 a valid JSON text, and issues an error if there is a extra data after
		 the otherwise valid JSON input. With this flag enabled, the decoder 
		 stops after decoding a valid JSON array or object, and thus allows
		 extra data after the JSON text.
	*)
	"DisableEOF" -> False,
	"AllowMessages"->False
};

Options[ToJSON]={ 
	(* 
		An expertimetail feature that handles more symbols 
		than Null, True, False 
	*)
	"AllowAllSymbols"->False,
	(*
		If True compact the JSON in a nice a minimal fashion, 
		or returns nicely formatted JSON with spacing and
		proper indention if False
	*)
	"Compact" -> False, 
	(*
		Ensures that only ASCII characters are brought back.
	*)
	"ASCIIOnly" -> False, 
	(*
		Sort the the keys in alpha numeric order 
	*)
	"SortKeys" -> False , 
	(*
		Preserves the order in which the input is given.
		ie Lists and Rule Lists
	*)
	"PreserveOrder" -> True,
	"AllowMessages"->False
};
(******************************************************************************)
(* End of Attributes/Options *)


(* ::Section:: *)
(* Internal Functions *)
(******************************************************************************)


(* ::Subsection:: *)
(**)


(* 
	loadLibrary:
	Loads the dynamic library into Mathematica at runtime.
*)
loadLibrary[name_String] :=
	If[$OperatingSystem =!= "iOS", (* on iOS, we can skip this step *)
	Check[
		Module[{lib = FindLibrary[name]},
			If[StringQ[lib] && FileExistsQ[lib],
				LibraryLoad[name],
				Throw[message["nolib"]; $Failed]
			]
		],
		Throw[message["libload"]; $Failed]
	]]
(*
	loadFunction:
	A wrapper function, that cleanly loads a dynamic library function
	into a global mathematica function. 
*)	
loadFunction[name_String, inputArgs_, outputArg_] :=
	Check[
		LibraryFunctionLoad[$DynamicLibrary, name, inputArgs, outputArg]
		,
		Throw[message["lfload", $DynamicLibrary, name]; $Failed]
	];
(*
	setMessageHead:
	Sets the $MessageHead variable.
*)
setMessageHead[head_] := ($MessageHead = head);
(*
	message:
	Cleanly call imessage, using only the tag and args.
*)
message[tag_, args___] := imessage[$MessageHead, tag, args];

(*
	imessage:
	Print out a Message onto Mathematica using it's given parameters.
*)
imessage[head_, tag_, args___] := Message[MessageName[head, tag], args];

(* 
	successQ:
	Checks all of the dynamic library function calls 
	have failed.
*)
successQ[_LibraryFunctionError | _LibraryFunction | $Failed] := False
successQ[___] := True

(* 
	failQ:
	Checks all of the dynamic library function calls 
	to detirmine if they were successfully called.
*)
failQ[_LibraryFunctionError | _LibraryFunction | $Failed] := True
failQ[___] := False

(*
	initialize:
	Is responsible for loading the exported functions from the dynamic
	library. This function only needs to become executed, at least once.s
*)
initialize[] := initialize[] = Module[{},
		(* Check if the dynamic library path is a member of the Global $LibraryPath *)
		If[!MemberQ[$LibraryPath, $LibraryDirectory],
			(* Prepend the $LibraryDirectory to the $LibrayPath, considering that it isn't apart. *)
			PrependTo[$LibraryPath, $LibraryDirectory];
		];
		loadLibrary[$DynamicLibrary];
		cSetOptions = LibraryFunctionLoad[$DynamicLibrary, "setOptions", {{_Integer, 1, "Constant"}}, _Integer];
		cEncodeInt = LibraryFunctionLoad[$DynamicLibrary, "encodeInt", {_Integer},"UTF8String"];
		cEncodeReal = LibraryFunctionLoad[$DynamicLibrary, "encodeReal", {_Real}, "UTF8String"];
		cEncodeBool = LibraryFunctionLoad[$DynamicLibrary, "encodeBool", {_Integer}, "UTF8String"];
		cEncodeString = LibraryFunctionLoad[$DynamicLibrary, "encodeString", {"UTF8String"}, "UTF8String"];
		cEncodeBoolList = LibraryFunctionLoad[$DynamicLibrary, "encodeBoolList", {{_Integer, 1, "Constant"}}, "UTF8String"];
		cEncodeIntList = LibraryFunctionLoad[$DynamicLibrary, "encodeIntList", {{_Integer, 1, "Constant"}}, "UTF8String"];
		cEncodeRealList = LibraryFunctionLoad[$DynamicLibrary, "encodeRealList", {{_Real, 1, "Constant"}}, "UTF8String"];
		cEncodeExpr = LibraryFunctionLoad[$DynamicLibrary, "encodeExpr", LinkObject, LinkObject];
		cDecodeExpr = LibraryFunctionLoad[$DynamicLibrary, "decode_json", LinkObject, LinkObject];
		
		True
	];
	
(*
	initializedQ:
	A wrapper around the initialize function,
	returning True or False, considering
	if the dynamic library has been correctly loaded.
*)
initializedQ[] :=
	Module[{res},
		Check[
			res = Catch[initialize[]];
			If[res === $Failed,
				message["init"];
				False,
				True
			],
			False
		]
	]


(* 
	InitOptions:
	Allows the C side of the code the chance to set the options.
	So that once the corresponding library call is made, it has the correct
	options in memory.
*)
InitOptions[options_List, FromJSON]:= Module[{mergeOptions,optionsList},
	optionsList = Map[(0) &, Range[12]];
	mergeOptions = DeleteDuplicates[ Flatten@Join[options,Options[FromJSON]],( #1[[1]] === #2[[1]] )&];
	optionsList[[DECODEANY]] = 1;
	cSetOptions[optionsList]
];

InitOptions[options_List, ToJSON]:= Module[{mergeOptions,optionsList},
	optionsList = Map[(0) &, Range[12]];
	mergeOptions = DeleteDuplicates[ Flatten@Join[options, Options[ToJSON]],( #1[[1]] === #2[[1]] )&];
	optionsList[[PRINTPRECISION]] = "MachineRealPrintPrecision" /. SystemOptions["MachineRealPrintPrecision"];
	optionsList[[ALLOWSYMBOLS]] = getOptionField[mergeOptions,"AllowAllSymbols"];
	optionsList[[INDENT]] = If[getOptionField[mergeOptions,"Compact"]===1,0,1];
	optionsList[[COMPACT]] = getOptionField[mergeOptions,"Compact"];
	optionsList[[ENSUREASCII]] =  getOptionField[mergeOptions,"ASCIIOnly"];
	optionsList[[SORTKEYS]] =  getOptionField[mergeOptions,"SortKeys"];
	optionsList[[PRESERVEORDER]] =  getOptionField[mergeOptions,"PreserveOrder"];
	optionsList[[ENCODEANY]] =  1;
	optionsList[[ESCAPESLASH]] = 0;
	optionsList[[REJECTDUPLICATES]] =0;
	cSetOptions[optionsList]
];

getOptionField[options_List, key_]:=Module[{value},
	value = (key /. Flatten[{options}] /. key-> 0);
	Which[SameQ[value, False], 0, SameQ[value, True],1, True,value]
];

allowMessages[options:OptionsPattern[],head_] := (
	("AllowMessages" /. DeleteDuplicates[ Flatten@Join[{options}, Options[head]],( #1[[1]] === #2[[1]] )&] /. "AllowMessages"-> False)
);

encodeJSON[i_Integer]:=(
	cEncodeInt[i]
);

encodeJSON[i_Real]:=(
	cEncodeReal[i]
);

(*
encodeJSON[str_String]:=(
	cEncodeString[str]
);
*)

encodeJSON[bool_(True|False)] :=(
	cEncodeString[If[bool, 1, 0]]
);

encodeJSON[intList:{__Integer}]:=(
	cEncodeIntList[intList]
);

encodeJSON[realList:{__Real}]:=(
	cEncodeRealList[realList]
);

encodeJSON[boolList:{__(True|False)}]:=(
	cEncodeBoolList[Map[(If[#, 1,0] )&,boolList]]
);

encodeJSON[Association[]]:= (
	"{}"
);

encodeJSON[expr_]:=(
	cEncodeExpr[expr]
);

decodeJSON[str_String] :=(
	cDecodeExpr[str]
);

(******************************************************************************)
(* End of the Internal Functions *)


(* ::Section:: *)
(* Exported Functions *)
(******************************************************************************)


(* ::Subsection:: *)
(**)


(*
	ToJSON:
	Encodes a given Mathematica expression into a corresponding JSON 
	representation

*)
ToJSON[expr_, opts:OptionsPattern[]] :=
	Module[{res},
		
		setMessageHead[ToJSON];
		(
			InitOptions[{opts},ToJSON];
			res = If[allowMessages[{opts}, ToJSON],encodeJSON[expr],Quiet[encodeJSON[expr]]];
			If[successQ[res], 
				res,
				$Failed
			]

		) /; initializedQ[]
	];
	
(*
	FromJSON:
	Decodes a given JSON string or file into the appropriate Mathematica expression
*)
FromJSON[""] = $Failed;
FromJSON[jsonstring_String, opts:OptionsPattern[]] := Module[{res},
		setMessageHead[FromJSON];
		(
				InitOptions[{opts},FromJSON];
				res = If[allowMessages[{opts}, FromJSON],decodeJSON[jsonstring],Quiet[decodeJSON[jsonstring]]];
				If[successQ[res], 
					res,
					$Failed
				]
			
		) /; initializedQ[]
	]

(******************************************************************************)
(* End of the Exported Functions *)

End[]

EndPackage[]

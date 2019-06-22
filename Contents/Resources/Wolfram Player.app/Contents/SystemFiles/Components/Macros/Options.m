Package["Macros`"]


PackageExport["BlockSetOptionValues"]
PackageExport["OptionValuePatterns"]

DeclareMacro[BlockSetOptionValues, blockSetOptionValues];

OptionValuePatterns[_] := <||>;

General::optg = "Option setting `` -> `` is invalid.";

blockSetOptionValues[context_String, body_] := Module[
	{patterns, global, assigns, name},
	
	patterns = OptionValuePatterns[$MacroHead];
	If[Length[patterns] === 0,
		Message[BlockSetOptionValues::nopatts, $MacroHead];
		Return[$Failed];
	];
	
	assigns = Function[{opt,pat},
		name = SymbolName[opt];
		name = ToLowerCase[StringTake[name, 1]] <> StringDrop[name, 1];
		global = InactiveSymbol[context <> "$" <> name];
		$Set$[global, optcode[$MacroHead, opt, pat] /. $OptionValue$[_, _, _] :> global]
	] @@@ Normal[patterns];
	
	assigns = DeleteCases[assigns, Null];

	$Block$[
		assigns,
		body
	]
];


BlockSetOptionValues::nosymbol = "No symbol `` was found in ``.";
BlockSetOptionValues::nopatts = "No OptionValuePatterns have been set for ``.";

OptMessage = None;

optfailcode[head_, name_, value_] := 
	$CompoundExpression$[
		If[OptMessage === None, 
			$Message$[$MessageName$[head, "optg"], name, value],
			$Message$[$MessageName$[head, "optvg"], name, value, OptMessage]
		],
		$OptionValue$[head, {}, name]
	];

optcode[head_, name_, patt_] := 
	$Replace$[
		$OptionValue$[head, name], 
		$RuleDelayed$[wrong:Except[patt], optfailcode[head, name, wrong]]]; 
	
optcode[head_, name_, list_List] := 
	$Replace$[
		$OptionValue$[head, name], 
		Append[optcode1[head, name, #]& /@ list, $RuleDelayed$[wrong_, optfailcode[head, name, wrong]]]];
	
optcode1[head_, name_, Rule[patt_, Inherited] | RuleDelayed[patt_, Inherited]] := 
	$RuleDelayed$[patt, $OptionValue$[head, {}, name]];
	
optcode1[_, _, patt_] := 
	p:patt :> p;
	
optcode1[_, _, rule_Rule | rule_RuleDelayed] := 
	rule;
	
optcode[head_, name_, Labeled[spec_, label_]] := 
	Block[{OptMessage = label}, optcode[head, name, spec]];
	
optrule[head_, name_Symbol -> rhs_] :=
	name -> optcode[head, name, rhs];
	
OptionValuePatterns::badspec = 
	"OptionValuePatterns[``] must be set to a list of rules mapping option symbols to allowed patterns.";
	
$toplevel = True;

OptionValuePatterns /: Set[OptionValuePatterns[head_Symbol], vals2_] /; $toplevel := Module[
	{code,vals},
	vals = Normal[vals2];
	If[!MatchQ[vals, {Repeated[Rule[_Symbol, _]]}],
		Message[OptionValuePatterns::badspec, head];
		Return[$Failed]];
	code = Map[optrule[head, #]&, vals];
	code = code /. $OptionValue$[h_, v_] :> $OptionValue$[h, $Slot$[1], v];
	code = Activate @ ParseInactives @ $Function$ @ $Association$ @ code;
	Block[{$toplevel = False},
		TagSet[head, OptionValuePatterns[head], vals];
		TagSet[head, OptionValueConstructor[head], code];
	];
];

OptionValueConstructor[head_Symbol] := 
	Function[opts,
		Association[
			Function[{key, val}, key -> Lookup[opts, key, val]] @@@
				Options[head]
		]
	];
	
PackageExport["OptionValues"]

OptionValues[head_, opts___] :=
	OptionValueConstructor[head][Flatten[{opts}]];
	

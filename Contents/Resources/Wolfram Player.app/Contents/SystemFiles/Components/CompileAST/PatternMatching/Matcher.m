
BeginPackage["CompileAST`PatternMatching`Matcher`"]

MExprMatcher
MExprMatcherQ
CreateMExprMatcher

MExprMatchQ::usage = "MExprMatchQ  "

MExprMatchGetBindings::usage = "MExprMatchGetBindings  "

MatcherSequence

Begin["`Private`"] 

Needs["CompileAST`Utilities`MExprVisitor`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for CatchException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileAST`PatternMatching`PatternObjects`ConditionalPattern`"]
Needs["CompileAST`Create`Construct`"]
Needs["CompileAST`PatternMatching`PatternObjects`ExceptPattern`"]
Needs["CompileAST`PatternMatching`PatternObjects`OptionalPattern`"]
Needs["CompileAST`PatternMatching`PatternObjects`SequencePattern`"]
Needs["CompileAST`Class`Normal`"]
Needs["CompileAST`PatternMatching`PatternObjects`NamedPattern`"]
Needs["CompileAST`PatternMatching`PatternObjects`PatternBindings`"]
Needs["CompileAST`PatternMatching`PatternObjects`SingleBlank`"]
Needs["CompileAST`PatternMatching`PatternObjects`Verbatim`"]
Needs["CompileUtilities`Callback`"]




RegisterCallback["DeclareCompileASTClass", Function[{st},
MExprMatcherClass = DeclareClass[
	MExprMatcher,
	<|
		"matchSingle" -> Function[{mexpr}, matchSingle[Self, mexpr]],
		"matchRecursive" -> Function[{mexpr, pattern}, matchRecursive[ Self, mexpr, pattern]],
		"processPattern" -> Function[ {obj}, processPattern[Self, obj]],
		"getPatternIndex" -> Function[{name}, getPatternIndex[Self, name]],
		"setBinding" -> Function[{index, mexpr}, setBinding[Self, index, mexpr]],
		"reset" -> Function[{}, reset[Self]],
		"toString" -> Function[{},
			toString[Self]
		],
		"toBoxes" -> Function[{fmt},
			toBoxes[Self, fmt]
		]
	|>,
	{
	   "pattern",
	   "visitor",
	   "bindings",
	   "namedPatternIndex",
	   "namedPatternMap",
	   "sequenceHead"
	},
    Predicate -> MExprMatcherQ
]
]]


toString[matcher_] := StringJoin[
	"MExprMatcher[",
	    matcher["pattern"]["toString"],
	"]"
]


icon := icon = Graphics[
	Text[
		Style[
			"Matcher", GrayLevel[0.7], Bold,
			0.9*CurrentValue["FontCapHeight"] / AbsoluteCurrentValue[Magnification]
		]
	],
	$FormatingGraphicsOptions
];   
     
toBoxes[matcher_, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"MExprMatcher",
		matcher,
  		icon,
		{
			With[{str = matcher["pattern"]["toString"]},
				BoxForm`SummaryItem[{"pattern: ", If[StringLength[str] > 100, "\n" <> str, str]}]
			]
		}
	  	,
		{}, 
  		fmt
  	]


CreateMExprMatcher[pattern_] :=
	CatchException[
		With[{self = CreateObject[MExprMatcher, <|"pattern" -> pattern|>]},
			initializeMatcher[self];
			self
		]
		,
		{{_, CreateFailure}}
	]

MatcherSequence := MatcherSequence = CreateMExprSymbol[InternalSequence];
		
initializeMatcher[ obj_] :=
	Module[ {pattern, bindings},
		obj["setVisitor", CreateObject[AddPatternsVisitor, <|"matcher" -> obj|>]];
		pattern = obj["pattern"];
		obj["setNamedPatternIndex", CreateReference[1]];
		obj["setNamedPatternMap", CreateReference[<||>]];
		obj["processPattern", pattern];
		bindings = CreateObject[ PatternBindings, <|
			"number" -> obj["namedPatternIndex"],
			"nameMap" -> obj["namedPatternMap"]["clone"]
		|>];
		obj["setBindings", bindings];
	]


RegisterCallback["DeclareCompileASTClass", Function[{st},
DeclareClass[
	AddPatternsVisitor,
	<|
		"visitNormal" -> Function[{obj}, addPattern[Self, obj]]
	|>,
	{
		"matcher"
	},
	Extends -> {MExprVisitorClass}]
]]


processPattern[self_, mexpr_] :=
	Module[ {vst},
		vst = self["visitor"];
		mexpr["accept", vst];
		mexpr
	]

patternClasses = <|
	"System`Blank" -> SingleBlank,
	"System`PatternTest" -> ConditionalPattern,
	"System`Pattern" -> NamedPattern,
	"System`Verbatim" -> VerbatimPattern,
	"System`Except" -> ExceptPattern
|>;

addPattern[self_, mexpr_] :=
	Module[ {hd, name, pattClass = Null, inst = Null},
		hd = mexpr["head"];
		If[ hd["symbolQ"],
			name = hd["fullName"];
			pattClass = Lookup[patternClasses, name, Null]
		];
		If[ pattClass =!= Null,
			inst = CreateObject[pattClass, <| "matcher" -> self["matcher"], "pattern" -> mexpr |>],
			inst = getOptionalSequence[self["matcher"], mexpr]
		];
		If[ inst === Null,
			True
			, (* Else *)
			mexpr["setProperty", "patternData" -> inst];
			False
		]
	]



getOptionalSequence[ matcher_, mexpr_] :=
	Module[ {args, optFound = False, seqFound = False},
		args = mexpr["arguments"];
		Scan[
			(
			optFound = optFound || IsOptionalPattern[ #];
			seqFound = seqFound || IsSequencePattern[ #];
			)&,
			args
		];
		Which[ 
			optFound,
				CreateObject[OptionalPattern, <| "matcher" -> matcher, "pattern" -> mexpr |>],
			seqFound,
				ProcessSequencePattern[ matcher, mexpr],
			True,
			    Null
		]
	]


patternName[mexpr_] :=
	If[MExprNormalQ[mexpr] && mexpr["hasHead", Pattern],
		mexpr["part", 1],
		None
	]

(*
 Pattern naming
*)

getPatternIndex[self_, name_] :=
	Module[ {index},
		index = self["namedPatternMap"]["lookup", name["fullName"], -1];
		If[ index < 0,
			index = self["namedPatternIndex"]["increment"];
			self["namedPatternMap"]["associateTo", name["fullName"] -> index];
		];
		index
	]

setBinding[self_, index_, mexpr_] :=
	Module[ {test},
		test = self["bindings"]["search", index];
		If[ test === Null,
			self["bindings"]["add", index, mexpr];
			mexpr,
			If[mexpr["sameQ", test],
				mexpr,
				Null
			]
		]
	]

reset[Self_] :=
	Self["bindings"]["reset"]

(* 
 Matcher Functions
*)
matchSingle[ self_, mexpr_] :=
	Module[ {res, pattern},
		pattern = self[ "pattern"];
		res = self["matchRecursive", mexpr, pattern];
		res
	]

matchRecursive[self_, mexpr_, pattern_] :=
	Module[ {data, res, arg1, arg2},
		Which[
			pattern["hasProperty", "patternData"],
				data = pattern["getProperty", "patternData"];
				data["matchPattern", mexpr],
			!mexpr["normalQ"],
				If[mexpr["sameQ", pattern],
					mexpr,
					Null
				],
			mexpr["length"] =!= pattern["length"],
				Null,
			matchRecursive[self, mexpr["head"], pattern["head"]] === Null,
				Null,
			True,
				res = mexpr;
				Do[ arg1 = mexpr["part",i]; 
					arg2 = pattern["part",i];
					If[matchRecursive[self, arg1, arg2] === Null,
						res = Null;
						Return[]
					],
					{i, mexpr["length"]}
				];
				res
		]
	]

(*
 Top Level matching functions
*)
MExprMatchQ[ mexpr_, pattern_] :=
	Module[ {matcher, ef},
		CatchException[
			matcher = CreateMExprMatcher[pattern];
			ef = matcher["matchSingle", mexpr];
			ef =!= Null
			,
			{{_, False&}}
		]
	]

MExprMatchQ[___] :=
	False


toMExprSymbol[s_String] :=
	With[{sym = Symbol[s]},
		CreateMExprSymbol[sym]
	]
	
MExprMatchGetBindings[ mexpr_, pattern_] :=
	Module[ {matcher, ef},
		CatchException[
			matcher = CreateMExprMatcher[pattern];
			ef = matcher["matchSingle", mexpr];
			If[ ef === Null, 
					<||>, 
					KeyMap[
						toMExprSymbol,
						matcher["bindings"]["getBindings"]
					]
			]
			,
			{{_, CreateFailure}}
		]
	]	


End[]

EndPackage[] 

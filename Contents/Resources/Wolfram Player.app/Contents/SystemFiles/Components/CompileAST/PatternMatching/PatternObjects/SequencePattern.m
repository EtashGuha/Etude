
BeginPackage["CompileAST`PatternMatching`PatternObjects`SequencePattern`"]

SequencePattern
IsSequencePattern
ProcessSequencePattern

Begin["`Private`"] 

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileAST`Create`Construct`"]
Needs["CompileAST`Export`FromMExpr`"]
Needs["CompileAST`PatternMatching`PatternObjects`ConditionalPattern`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileAST`PatternMatching`Matcher`"]
Needs["CompileUtilities`Callback`"]



ProcessSequencePattern[ matcher_, mexpr_] :=
	Module[{seqPos = CreateReference[{}]},
		MapIndexed[
			(
			If[IsSequencePattern[#1],
				seqPos["appendTo", First[#2]]];
			)&, mexpr["arguments"]];
		Which[
			seqPos["length"] === 1,
				CreateObject[UnarySequencePattern, <| "matcher" -> matcher, "pattern" -> mexpr, "seqPos" -> seqPos["getPart",1] |>],
			True,
				ThrowException[{"Cannot process sequence pattern: ", mexpr}]
		]
	]
	
	
	
	
	
	
RegisterCallback["DeclareCompileASTClass", Function[{st},
UnarySequencePatternClass = DeclareClass[
	UnarySequencePattern,
	<|
		"matchPattern" -> Function[{mexpr}, matchUnarySequenceMExpr[Self, mexpr]],
		"initialize" -> Function[{}, initializeUnarySequence[Self]]
	|>,
	{
	   "matcher",
	   "pattern",
	   "sequencePattern",
	   "seqPos",
	   "zeroLengthQ"
	},
    Predicate -> UnarySequencePatternQ
]
]]


(*
  There is only one sequence at the position seqPos.
*)
initializeUnarySequence[ self_] :=
	Module[ {matcher, pattern, seqPos, args, pos, monoSeq},
		matcher = self["matcher"];
		pattern = self["pattern"];
		seqPos = self["seqPos"];
		matcher["processPattern", pattern["head"]];
		args = pattern["arguments"];
		MapIndexed[
			(
			pos = First[#2];
			If[ pos === seqPos,
				self["setZeroLengthQ", IsZeroLengthSequence[#1]];
				monoSeq = CreateObject[MonoSequence, <| "matcher" -> matcher, "sequence" -> #1 |>];
				self["setSequencePattern", monoSeq]; 
				,
				matcher["processPattern", #1]];
			1;
			)&, args]; 
		
	]


matchUnarySequenceMExpr[ self_, mexpr_] :=
	Module[ {matcher, pattern, lenExpr, lenPatt, seqPos, ef, lenDiff, i},
		matcher = self["matcher"];
		pattern = self["pattern"];
		seqPos = self["seqPos"];
		lenExpr = mexpr["length"];
		lenPatt = pattern["length"];
		
		If[ lenExpr < lenPatt - If[ self["zeroLengthQ"], 1, 0],
			Return[Null]];
		ef = matcher["matchRecursive",  mexpr["head"], pattern["head"]];
		If[ ef === Null,
			Return[Null]];
		
		(*
		  Match up to the sequence
		*)
		Do[
			ef = matcher["matchRecursive",  mexpr["part", i], pattern["part", i]];
			If[ ef === Null,
				Return[Null]];
			,
			{i,seqPos-1}];
		If[ ef === Null,
			Return[ef]];
		
		(*
		  Match the sequence
		*)
		lenDiff = lenExpr - lenPatt;
		ef = self["sequencePattern"]["matchPattern", matcher, mexpr, seqPos, seqPos+lenDiff];
		If[ ef === Null,
			Return[ef]];
		
		(*
		  Match the rest
		*)
		Do[
			ef = matcher["matchRecursive",  mexpr["part", i+lenDiff], pattern["part", i]];
			If[ ef === Null,
				Return[Null]];
			,
			{i,seqPos+1,lenPatt}];
		If[ ef === Null,
			Return[ef]];
		mexpr
	]


RegisterCallback["DeclareCompileASTClass", Function[{st},
MonoSequenceClass = DeclareClass[
	MonoSequence,
	<|
		"matchPattern" -> Function[{matcher, mexpr, min, max}, matchMonoPattern[Self, matcher, mexpr, min, max]],
		"initialize" -> Function[{}, initializeMonoSequence[Self]]
	|>,
	{
	   "matcher",
	   "sequence",
	   "pattern",
	   "nameIndex",
	   "condition",
	   "hasCondition" -> False,
	   "hasName" -> False,
	   "isSimple" -> True
	},
    Predicate -> MonoSequenceQ
]
]]


initializeMonoSequence[ self_] :=
	Module[{matcher, sequence, name, pattName, altPatt, condition},
		matcher = self["matcher"];
		sequence = self["sequence"];
		name = getFullName[sequence];
		If[name === "System`PatternTest",
			self["setIsSimple", False];
			condition = sequence["part",2];
			condition = ReleaseHold[FromMExpr[condition]];
			self["setCondition", condition];
			self["setHasCondition", True];
			sequence = sequence["part",1];
			name = getFullName[sequence];
		];
		If[ name === "System`Pattern" && sequence["length"] === 2,
			self["setIsSimple", False];
			pattName = sequence["part",1];
			sequence = sequence["part",2];
			name = getFullName[sequence];
			self["setHasName", True];
			self["setNameIndex", matcher["getPatternIndex", pattName]];
		];
		If[ name === "System`BlankSequence" || name === "System`BlankNullSequence",
			altPatt = 
				If[ sequence["length"] === 0, 
						CreateMExprNormal[ Blank, {}],
						self["setIsSimple", False];
						CreateMExprNormal[ Blank, {sequence["part", 1]}]];
			matcher["processPattern", altPatt];
			self["setPattern", altPatt];
			,
			ThrowException[{"Cannot process sequence pattern mode 3: ", sequence}]
		];
	]

getFullName[mexpr_] :=
	Module[ {h},
		h = mexpr["head"];
		If[ !h["symbolQ"],
			ThrowException[{"Cannot process sequence pattern: ", mexpr}]
		];
		h["fullName"]
	]

matchMonoPattern[self_, matcher_, mexpr_, min_, max_] :=
	Module[ {pattern, ef, i, res, condition, hasCondition = False, mexprI, app},
		
		(*
		  If this is a simple mono,  ie no names,  no heads,  just __ or ___ then
		  we will have matched already due to the length computation.
		*)
		If[self["isSimple"],
			Return[ mexpr]];
		
		hasCondition = self["hasCondition"];		
		pattern = self["pattern"];		
		res = {};
		Scan[
			(
			If[ hasCondition,
				app = Apply[self["condition"], {#}];
				If[ !IsConditionTrue[app],
					ef = Null;
					Return[ef]]];
			ef = matcher["matchRecursive", #, pattern];
			If[ ef === Null,
				Return[ef]];
			res = {res, ef};
			)&
			,
			Take[ mexpr["arguments"], {min, max}]];
			
		If[ ef === Null,
			Return[ ef]];
		res = CreateMExprNormal @@ { MatcherSequence, Flatten[res]};
		If[ self["hasName"],
			matcher["setBinding", self["nameIndex"], res]
		];
		
		ef
	]
	
	

IsZeroLengthSequence[mexpr_] :=
	Module[ {h, fullName},
		h = mexpr["head"];
		If[ !h["symbolQ"],
			Return[ False]];
		fullName = h["fullName"];
		Which[ 
			fullName === "System`BlankSequence",
				False,
			fullName === "System`BlankNullSequence",
				True,
			fullName === "System`Pattern" && mexpr["length"] === 2,
				IsZeroLengthSequence[mexpr["part", 2]],
			True,
				False]
	]


IsSequencePattern[mexpr_] :=
	Module[ {h, fullName},
		h = mexpr["head"];
		If[ !h["symbolQ"],
			Return[ False]];
		fullName = h["fullName"];
		Which[ 
			fullName === "System`BlankSequence",
				True,
			fullName === "System`BlankNullSequence",
				True,
			fullName === "System`Pattern" && mexpr["length"] === 2,
				IsSequencePattern[mexpr["part", 2]],
			fullName === "System`PatternTest" && mexpr["length"] === 2,
				IsSequencePattern[mexpr["part", 1]],
			True,
				False
		]
	]


End[]

EndPackage[]

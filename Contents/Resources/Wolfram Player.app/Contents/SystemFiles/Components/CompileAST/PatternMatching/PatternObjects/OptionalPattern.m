
BeginPackage["CompileAST`PatternMatching`PatternObjects`OptionalPattern`"]

OptionalPattern
IsOptionalPattern

Begin["`Private`"] 

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileAST`Create`Construct`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Callback`"]




RegisterCallback["DeclareCompileASTClass", Function[{st},
OptionalPatternClass = DeclareClass[
	OptionalPattern,
	<|
		"matchPattern" -> Function[{mexpr}, matchMExpr[Self, mexpr]],
		"initialize" -> Function[{}, initialize[Self]]
	|>,
	{
	   "matcher",
	   "pattern",
	   "patternAlternates"
	},
    Predicate -> OptionalPatternQ
]
]]


(*
  There should be at least one Optional in this expr.
*)
initialize[ self_] :=
	Module[ {matcher, pattern, args, argsData = {{{},{}}}, l1, l2, patt, alt, patternAlternates, h},
		matcher = self["matcher"];
		pattern = self["pattern"];
		h = pattern["head"];
		args = pattern["arguments"];
		(*
		 argsData is a list of the form
		   {argsList, optsList},  where argsList is the arguments at each level and optsList is 
		   a list with elements {patt, alt}.
		*)
		Scan[
			(
			If[ IsOptionalPattern[#],
				(*
				This is an Optional,  we need to make two copies of argsData.  
				In the first we leave argsList alone and add the new {patt, alt} to each optsList.
				In the second we add the pattern to the end of each argsList structure and leave optsList alone
				*)
				patt = #["part",1];
				alt = #["part",2];
				l1 = Apply[Function[{argsList, optsList},{Append[argsList, patt], optsList}], argsData, {1}];
				l2 = Apply[Function[{argsList, optsList},{argsList, Append[optsList, {patt, alt}]}], argsData, {1}];
				argsData = Join[ l2, l1];
				, 
				 (*
				   Not optional,  so add # to the end of each element of argsList,  leave 
				   optsList alone
				 *)
				 argsData = Apply[Function[{argsList, optsList},{Append[argsList, #], optsList}], argsData, {1}]];
			)&, args]; 
		If[ Length[argsData] < 2, 
			ThrowException[{"Optional not found: ", pattern}]
		];
		argsData =
			Apply[ {CreateMExprNormal[h, #1], #2}&, argsData, {1}];
		patternAlternates = Apply[ {matcher["processPattern", #1], 
					Apply[ Function[ {opt, alt1}, {matcher["processPattern", opt], alt1}], #2, {1}]}&, argsData, {1}];
		self["setPatternAlternates", patternAlternates]
	]


matchMExpr[ self_, mexpr_] :=
	Module[ {matcher, pattAlternates, ef = Null},
		matcher = self["matcher"];
		If[ matcher["matchRecursive",  mexpr["head"], self["pattern"]["head"]] === Null,
			Return[Null]];
		pattAlternates = self["patternAlternates"];
		Scan[
			(
			ef = tryAlternate[ self["matcher"], mexpr, #];
			If[ ef =!= Null,
				Return[]]
			)&, pattAlternates];
		ef
	]

tryAlternate[ matcher_, mexpr_, {pattExpr_, pattAlts_}] :=
	Module[ {ef, patt, alt},
		ef = matcher["matchRecursive",  mexpr, pattExpr];
		If[ ef === Null,
			Return[ Null]];
		Scan[
			(
			patt = First[#];
			alt = Last[#];
			ef = matcher["matchRecursive",  alt, patt];
			If[ ef === Null,
				Return[ Null]];
			)&, pattAlts];
		ef
	]


IsOptionalPattern[mexpr_] :=
	Module[ {h, fullName},
		h = mexpr["head"];
		If[ !h["symbolQ"],
			Return[ False]];
		fullName = h["fullName"];
		fullName === "System`Optional" && mexpr["length"] === 2
	]


End[]

EndPackage[]

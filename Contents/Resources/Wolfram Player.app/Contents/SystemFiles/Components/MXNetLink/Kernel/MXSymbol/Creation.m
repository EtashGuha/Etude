Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["MXNetLink`Bootstrap`LoadOperators"]

LoadOperators::notloaded = "Cannot initialize symbols if libraries aren't loaded."

LoadOperators[] := If[
	TrueQ[$LibrariesLoaded],
	Scan[setupSymbol, Discard[StringStartsQ["_backward_"]] @ Keys @ $MXOperatorData],
	Message[LoadOperators::notloaded]; $Failed
]

(******************************************************************************)

mx2sym[mx_] := "MXNet`" <> StringReplace[mx, "_" -> "$"]

setupSymbol[name_String] := With[
	{sym = Symbol @ mx2sym @ name},
	sym::usage = generateSymbolUsage[sym, name];
	sym[args___] := CatchFailure @ createMXSymbol[name, {args}]
]


(******************************************************************************)

mxlDeclare[mxlMXSymbolCreateAtomicSymbol, {"Integer", "String", "String", "String"}]

createMXSymbol[opname_String, arguments_] := Scope[

	(* Symbol Info Key *)
	symbolInfo = $MXOperatorData[opname];
	varArgKey = symbolInfo["VariableArgumentKey"];
	
	(* variables *)
	paramKeys = {};
	paramVals = {};
	symbolKwargs = <||>;
	{args, kwargs} = SelectDiscard[arguments, RuleQ];
	kwargs = Association[kwargs];
	
	(* pop off name + attr kwargs *)
	name = Lookup[kwargs, "name", Automatic];
	If[name =!= Automatic, KeyDropFrom[kwargs, "name"]];

	attr = Lookup[kwargs, "attrs", None];
	If[attr =!= None, KeyDropFrom[kwargs, "attr"]];
	
	(* deal with varargs *)
	If[StringLength[varArgKey] > 0 && !KeyExistsQ[kwargs, varArgKey],
		AppendTo[paramKeys, varArgKey];
		AppendTo[paramVals, IntegerString @ Length @ args]
	];
	
	(* deal with kwargs *)
	KeyValueMap[
		If[MXSymbolQ[#2], 
			symbolKwargs[#1] = #2
		, 
			AppendTo[paramKeys, #1];
			AppendTo[paramVals, mxParameterToString @ #2];
		]&,
		kwargs
	];

	(* pack lists of strings for LL call *)
	keysString = mxlPackStringVector @ paramKeys;
	valsString = mxlPackStringVector @ paramVals;

	(* create output symbol *)
	outputSymbol = CreateManagedLibraryExpression["MXSymbol", MXSymbol];
	System`Private`SetNoEntry @ outputSymbol;

	mxlCall[mxlMXSymbolCreateAtomicSymbol, 
		MLEID @ outputSymbol, opname,
		keysString, valsString
	];
	
	(* error check *)
	If[Length[args] > 0 && Length[symbolKwargs] > 0,
		ThrowFailure["Can only accept input Symbols either as positional or keyword arguments, not both"]
	];
	
	If[name === Automatic,
		count = KeyIncrement[$SymbolNameCounter, name] - 1;
		name = ToLowerCase[opname] <> IntegerString[count];
	];

	MXSymbolCompose[outputSymbol, name, Join[args, Normal @ symbolKwargs]]
]

If[!AssociationQ[$SymbolNameCounter], $SymbolNameCounter = <||>];

(******************************************************************************)

PackageScope["$SymbolArgumentOrdering"]

$SymbolArgumentOrdering := $SymbolArgumentOrdering = Map[
	Keys[Select[#, StringStartsQ[#Type, "Symbol"]&]]&,
	$MXOperatorData[[All, "Arguments"]]
]

(******************************************************************************)

generateSymbolUsage[sym_, symbolName_String] := Scope[

	argInfo = $MXOperatorData[symbolName, "Arguments"];
	description = $MXOperatorData[symbolName, "Description"];

	argInfo = KeyValueMap[Prepend[#2, "Name" -> #1]&, argInfo];

	{inArgs, argInfo} = SelectDiscard[argInfo, StringContainsQ[#Type, "NDArray"]&];
	{optArgs, reqArgs} = SelectDiscard[argInfo, Key["Optional"]];

	argNames = toItalic /@ inArgs[[All, "Name"]];
	col = Flatten[{
		StringJoin[SymbolName[sym], "[ ", StringRiffle[argNames, ", "], " ] is an MXNet operator."],
		makeArgTable["inputs", inArgs],
		makeArgTable["required", reqArgs],
		makeArgTable["optional", optArgs],
		"\[Bullet] Description of operator:",
		prettyDescrip @ description
	}];

	StringRiffle[col, "\n"]
]

$gridLinearLeft = "\!\(\*TagBox[GridBox[";
$gridLinearRight = ", \
Rule[GridBoxAlignment, List[Rule[\"Columns\", List[List[Left]]]]], Rule[AutoDelete, \
False], Rule[GridBoxItemSize, List[Rule[\"Columns\", List[List[Automatic]]], \
Rule[BaseStyle, List[Rule[ShowStringCharacters, True]]], \
Rule[\"Rows\", List[List[Automatic]]]]], Rule[GridBoxSpacings, List[Rule[\"Columns\", \
List[List[2]]]]]], \"Grid\"]\)"

linearTable[table_] := Scope[
	elems = ToString[Map[ToBoxes, table, {2}], InputForm];
	$gridLinearLeft <> elems <> $gridLinearRight
]

toItalic[str_] := "\!\(\*StyleBox[\"" <> str <> "\",FontSlant->\"Italic\"]\)"
toBold[str_] := "\!\(\*StyleBox[\"" <> str <> "\",FontWeight->Bold]\)"

$headerLine = toBold /@ {
	StringPadRight["name", 20], 
	StringPadRight["type", 30],
	"description"
}

makeArgTable[_, {}] := {};
makeArgTable[adj_, args_] := {
	"\[Bullet] The following arguments are " <> adj <> ":",
	linearTable @ Prepend[
		Lookup[args, {"Name", "Type", "Description"}],
		$headerLine
	]
}

prettyDescrip[str_] := StringReplace[str, {
	"``" ~~ w:(WordCharacter|"_").. ~~ "``" :> toItalic[w],
	"**" ~~ w:(WordCharacter|"_").. ~~ "**" :> toBold[w],
	Repeated["\n", {3, Infinity}] -> "\n\n"
}]
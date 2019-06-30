(*
TODO
  This needs some thought and discussion.   It needs to hold an inert,  probably 
  stringified version of each symbol,  head,  compound head that can be lowered.
  
  Maybe a field called nameKey rather than fullName would be better.
*)

BeginPackage["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]

LanguagePrimitiveQ
CreateSystemPrimitive
CreateSystemPrimitiveAtom
CreateSystemPrimitiveLiteral

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["Compile`Utilities`Language`Attributes`"]
Needs["CompileUtilities`Format`"] (* for CreateBoxIcon *)
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Error`Exceptions`"]


toBoxes[sym_, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"LanguagePrimitive",
		sym,
  		CreateBoxIcon["LANG\nPRIM"],
  		{
  		    BoxForm`SummaryItem[{"name: ", sym["name"]}],
  		    BoxForm`SummaryItem[{"fullName: ", sym["fullName"]}],
  		    BoxForm`SummaryItem[{"context: ", sym["context"]}],
  		    BoxForm`SummaryItem[{"attributes: ", sym["attributes"]}]
  		},
  		{}, 
  		fmt
  	]

RegisterCallback["DeclareCompileClass", Function[{st},
DeclareClass[
	LanguagePrimitive,
	<|
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"sameQ" -> Function[{other}, sameQ[Self, other]]
	|>,
	{
		"symbol",
		"context",
		"name",
		"fullName",
		"attributes"
	}
	,
	Predicate -> LanguagePrimitiveQ
]
]]

sameQ[self_, other_] :=
	LanguagePrimitiveQ[other] &&
		self["context"] === other["context"] &&
		self["name"] === other["name"] &&
		self["fullName"] === other["fullName"] &&
		self["attributes"] === other["attributes"]






CreateSystemPrimitiveAtom[sym_String] :=
	CreateLanguagePrimitive[sym, "Compile`Internal`", sym, "Compile`Internal`"<>sym, {}]
	
CreateSystemPrimitiveAtom[sym_Symbol] :=
	CreateLanguagePrimitive[sym, Context[sym], SymbolName[sym], Context[sym]<>SymbolName[sym], attrs[sym]]

CreateSystemPrimitiveAtom[Native`Global[name_String]] :=
	CreateLanguagePrimitive[sym, "Compile`Global`", name, "Compile`Global`" <> name, {}]


CreateSystemPrimitive[sym_String] :=
	CreateLanguagePrimitive[sym, "Compile`Internal`", sym, ("Compile`Internal`"<>sym)[], {}]
	
CreateSystemPrimitive[sym_Symbol] :=
	CreateLanguagePrimitive[sym, Context[sym], SymbolName[sym], (Context[sym]<>SymbolName[sym])[], attrs[sym]]


(* subvalues *)
CreateSystemPrimitive[sym_String[___]] :=
	CreateLanguagePrimitive[sym, "Compile`Internal`", sym, ("Compile`Internal`"<>sym)[][], {}]

(* subvalues *)
CreateSystemPrimitive[sym_Symbol[___]] :=
	CreateLanguagePrimitive[sym, Context[sym], SymbolName[sym], (Context[sym]<>SymbolName[sym])[][], attrs[sym]]


CreateSystemPrimitive[args___] :=
	ThrowException[{"Unrecognized call to CreateSystemPrimitive", {args}}]

CreateSystemPrimitiveLiteral[] :=
	CreateObject[
		LanguagePrimitive,
		<|
			"fullName" -> "Literal"
		|>
	]


attrs[s_] :=
	Lookup[$SystemAttributes, ToString[s], Attributes[s]]


CreateLanguagePrimitive[sym_, context_, name_, fullName_, attributes_] :=
	CreateObject[
		LanguagePrimitive,
		<|
			"symbol" -> sym, 
			"context" -> context,
			"name" -> name,
			"fullName" -> fullName,
			"attributes" -> attributes
		|>
	]
	
End[]

EndPackage[]

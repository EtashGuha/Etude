
BeginPackage["CompileAST`Class`Base`"]

MExpr;
MExprClass;
MExprQ;
CoerceMExpr;
RegisterMExpr

Begin["`Private`"] 

Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileAST`Export`FromMExpr`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileAST`Create`Construct`"]
Needs["CompileAST`Export`ShowMExpr`"]
Needs["CompileAST`Export`Format`"] (* For MExprToFormattedString *)


clone[mexpr_, ___] :=
	With[{cln = mexpr["_clone"]},
		cln["setProperties", mexpr["clonedProperties"]];
		cln
	]

CoerceMExpr[a_] := (* TODO: Note, this will cause `a` to evaluate which is not good *)
	If[MExprQ[a], 
		a,
		CreateMExpr[a]
	]

RegisterCallback["DeclareCompileASTClass", Function[{st},
MExprClass = DeclareClass[
	MExpr,
	<|
		"initialize" -> Function[{},
			If[Self["properties"] === Undefined || Self["properties"] === Null,
				Self["setProperties", CreateReference[<||>]]
			]
		],
		"isOperator" -> Function[{}, False],
		"literalQ" -> Function[{}, False],
		"atomQ" -> Function[{}, False],
		"normalQ" -> Function[{}, False],
		"symbolQ" -> Function[{}, False],
		"head" -> Function[{}, CoerceMExpr[Self["_head"]]],
		"setHead" -> Function[{val}, SetData[Self["_head"], val]; Self],
		"getHead" -> Function[{},
			If[Self["normalQ"] && Self["head"]["symbolQ"],
				Self["_head"]["symbol"],
				Self["_head"]
			]
		],
		"unsameQ" -> Function[{val},
			!Self["sameQ", val]
		],
		"clone" -> (clone[Self, ##]&),
		"isScopingConstruct" -> Function[{},
			False
		],
		"addPropertyHolder" -> Function[{mexpr}, addPropertyHolder[Self, mexpr]],
		"serialize" -> Function[{}, FromMExpr[Self]],
		"accept" -> Function[{vst}, Null],
		"toExpression" -> Function[{}, FromMExpr[Self]],
		"dispose" -> Function[{}, dispose[Self]],
		"show" -> Function[{}, ShowMExpr[Self]],
		"prettyPrint" -> Function[{}, MExprToFormattedString[Self]]
	|>,
	{
		"id" -> 0,
		"_head",
		"span",
		"type" -> Undefined,
		"properties" -> Undefined			
	},
	Extends -> {
		ClassPropertiesTrait
	}, 
(*
 Set a predicate with an internal name that will not be called.
 This class is not designed to be instantiated.
*)
	Predicate -> nullMExprQ
]
]]

(*
  Add functionality for MExprQ
*)
$mexprObjects = <||>

RegisterMExpr[ name_] :=
	AssociateTo[$mexprObjects, name -> True]
	
MExprQ[ obj_] :=
	ObjectInstanceQ[obj] && KeyExistsQ[$mexprObjects, obj["_class"]]



(*
  addPropertyHolders is designed to hold any mexprs that have a property which 
  points to another mexpr.  This is stored in self, which is supposed to be the
  top mexpr.  The dispose method uses this to clear the properties which allows 
  the memory for the mexprs to be collected.
*)
addPropertyHolder[self_, mexpr_] :=
	Module[{propHolders = self["getProperty", "propertyHolders", Null]},
		If[ propHolders === Null,
			propHolders = CreateReference[<||>];
			self["setProperty", "propertyHolders" -> propHolders]];
		propHolders["associateTo", mexpr["id"] -> mexpr]
	]

dispose[self_] :=
	Module[{mexprHolders = self["getProperty", "propertyHolders", {}]},
		If[mexprHolders =!= {},
			mexprHolders = mexprHolders["values"]];
		Scan[#["setProperties", <||>]&, mexprHolders];
	]


End[]

EndPackage[]

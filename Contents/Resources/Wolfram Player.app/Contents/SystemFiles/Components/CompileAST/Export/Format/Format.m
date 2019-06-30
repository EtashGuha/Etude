BeginPackage["CompileAST`Export`Format`"]

MExprToFormattedString

Begin["`Private`"]

Needs["CompileAST`Export`Format`CodeFormatter`"];
Needs["CompileAST`Class`Base`"]


MExprToFormattedString[mexpr_?MExprQ] := Module[{
	(* Get mexpr as an expression and wrap it in HoldForm *)
	expr = Extract[mexpr["toExpression"], 1, HoldForm],
	boxes
},
	(* CodeFormatterMakeBoxes should return a box structure that doesn't contain any fancy box
	   elemensts like TagBox / FractionBox / etc. *)
	boxes = CodeFormatterMakeBoxes @@ {expr};
	(* Insert tabs/newlines to make it pretty *)
	(* This could be changed to FullCodeFormatCompact if more compact code is desired *)
	boxes = FullCodeFormat[boxes];
	If[$FrontEnd === Null,
		(* Flatten the RowBox structure into its constituent strings, and join them. *)
		StringJoin[boxes //. {RowBox[{args___}] :> args}],
		(* otherwise within the front end return the box data *)
		DisplayForm[boxes]
	]
]

End[]

EndPackage[]
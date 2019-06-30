
BeginPackage["Compile`API`CompiledCodeFunction`"]


Begin["`Private`"]

Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)	

(*
 Print form for CompiledCodeFunction
*)

(**************************************************)
(**************************************************)

icon := Graphics[Text[
  Style["Code", GrayLevel[0.7], Bold,
   CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];

MakeBoxes[ ccf:CompiledCodeFunction[data_?AssociationQ, ___] /; System`Private`ValidQ[Unevaluated[ccf]], fmt_] :=
	Module[ {sig},
	sig = Lookup[data, "Signature", Null];
	BoxForm`ArrangeSummaryBox[
		"CompiledCodeFunction",
		ccf,
		icon,
		{
			If[sig === Null,
				BoxForm`SummaryItem[{"Signature: ", ""}],
				BoxForm`SummaryItem[{"Signature: ", PrettyFormat[sig]}]
			]
		}
		,
		{}
		,
		fmt
	]
	]




(*
This exists mainly to strip the type from a TypeLiteral, e.g., 1:Integer64 is printed as 1
*)


PrettyFormat[Type[t_]] := PrettyFormat[t]
PrettyFormat[TypeSpecifier[t_]] := PrettyFormat[t]
PrettyFormat[Rule[params_, ret_]] := PrettyFormat[params] -> PrettyFormat[ret]
PrettyFormat[l_List] := PrettyFormat /@ l

PrettyFormat[ "Complex"["Real64"]] := "ComplexReal64"
PrettyFormat[ "Complex"["Real32"]] := "ComplexReal32"
 
PrettyFormat[app_String[args__]] := TypeSpecifier[ PrettyFormat[app]] @@ PrettyFormat[{args}]
PrettyFormat[TypeFramework`TypeLiteral[val_,_]] := PrettyFormat[val]
PrettyFormat[val_] := val



Information`AddRegistry[ CompiledCodeFunction, getInformation]

getInformation[HoldPattern[CompiledCodeFunction][ass_?AssociationQ,__]] :=
	Module[ {sig, res},
		sig = Lookup[ ass, "Signature", Null];
		res = <|"ObjectType" -> "CompiledCodeFunction"|>;
		If[ sig =!= Null && !MissingQ[sig],
			AssociateTo[res, "Signature" -> PrettyFormat[sig]]];
		res
	]





End[]

EndPackage[]


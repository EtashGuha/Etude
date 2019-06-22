
BeginPackage["CompileUtilities`Asserter`Assert`"]

AssertThat;
AssertionFailure;

Begin["`Private`"]


$Verb = "asserted"
$Dispatcher = AssertThat
$StrategyName = "AssertThat"
$FailureHandler = Function[{failure}, Assert[False, failure]]
$FailureSymbol = AssertionFailure


Attributes[AssertThat] = {HoldAllComplete}

AssertThat[___] := Function[Null, Function[Null, Null, {HoldAllComplete}], {HoldAllComplete}]


If[OwnValues[$AssertFunction] =!= {},
(* Asserts are turned on, so define AssertThat to do something *)
Get["CompileUtilities`Asserter`Internal`Internal`"]
]

End[]
EndPackage[]

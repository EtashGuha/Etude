
BeginPackage["CompileUtilities`Asserter`Expect`"]

ExpectThat;
ExpectationFailure;

Begin["`Private`"]

$Verb = "expected"
$Dispatcher = ExpectThat
$StrategyName = "ExpectThat"
$FailureHandler = Print
$FailureSymbol = ExpectationFailure


Attributes[ExpectThat] = {HoldAllComplete}

ExpectThat[___] := Function[Null, Function[Null, Null, {HoldAllComplete}], {HoldAllComplete}]


If[OwnValues[$AssertFunction] =!= {},
(* Asserts are turned on, so define ExpectThat to do something *)
Get["CompileUtilities`Asserter`Internal`Internal`"]
]


End[]
EndPackage[]

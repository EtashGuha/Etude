
BeginPackage["CompileUtilities`Asserter`Assume`"]

AssumeThat;
AssumptionFailure;

Begin["`Private`"]

$Verb = "assumed"
$Dispatcher = AssumeThat
$StrategyName = "AssumeThat"
$FailureHandler = Print
$FailureSymbol = AssumptionFailure


Attributes[AssumeThat] = {HoldAllComplete}

AssumeThat[___] := Function[Null, Function[Null, Null, {HoldAllComplete}], {HoldAllComplete}]


If[OwnValues[$AssertFunction] =!= {},
(* Asserts are turned on, so define AssumeThat to do something *)
Get["CompileUtilities`Asserter`Internal`Internal`"]
]


End[]
EndPackage[]

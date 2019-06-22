(* Mathematica Package *)

BeginPackage["IntegratedServices`"]

Begin["`Private`"]

$IntegratedServicesAPIBase := CloudObject["user:services-admin@wolfram.com/deployedservices/deployedserviceAPIs"];

IntegratedServices`$IntegratedServicesBase := $IntegratedServicesAPIBase; (*For use in GeoIntegratedServices*)

$ServiceCreditsRemainingEndpoint := URLBuild[{$IntegratedServicesAPIBase,"ServiceCreditsRemaining"}];

$CloudPRDBaseQ := TrueQ[StringMatchQ[$CloudBase,"https://www.wolframcloud.com/"|"https://www.wolframcloud.com"]];

$CloudEvaluateQ := TrueQ[$CloudEvaluation];

End[] (* End Private Context *)

EndPackage[]

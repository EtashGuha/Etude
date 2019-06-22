Package["NeuralNetworks`"]

PackageExport["NetReplace"]

Options[NetReplace] = {
	"FlattenInPlace" -> True
};

NetReplace[net_, rules_, OptionsPattern[]] := CatchFailureAsMessage @ 
	iNetReplace[TestValidNet[net]; NData[net], rules, OptionValue["FlattenInPlace"]];

DeclareArgumentCount[NetReplace, 2];

NetReplace::norep = "No pattern matches occured and hence no replacements were performed.";

iNetReplace[net_, rules_, fp_] := Scope[
	$disableFlatten = !TrueQ[fp];
	rules = procRARule /@ ToList[rules];
	tryReplace = unrawWrapper @ Replace @ rules /. ((a_ :> b_) :> (a :> FreshPath[b]));
	res = replaceDispatch[net];
	If[!AssociationQ[res] || !KeyExistsQ[res, "Type"], ReturnFailed["interr2"]];
	If[res === net, Message[NetReplace::norep]];
	ConstructWithInference[NSymbol[res], res]
];

procRARule[(head:Rule|RuleDelayed)[lhs_, rhs_]] := 
	head[procRALHS[lhs], rhs];

NetReplace::invrules = "Second argument should be a list of rules."
procRARule[other_] := ThrowFailure["invrules"];

procRALHS[lhs_] := ReplaceAll[lhs,
	net:(_Symbol[_Association, _Association] ? ValidNetQ) :> 
		RuleCondition @ LiteralNetToPattern[net]
];

DeclareMethod[replaceDispatch, replaceLayer, replaceContainerOrOperator, replaceContainerOrOperator]

replaceLayer[net_] := net // tryReplace;
replaceContainerOrOperator[net_] := RawNetMap[replaceDispatch, net] // tryReplace;




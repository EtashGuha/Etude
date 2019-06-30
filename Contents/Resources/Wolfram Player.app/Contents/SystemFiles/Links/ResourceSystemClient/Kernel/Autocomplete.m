(* Wolfram Language Package *)


BeginPackage["ResourceSystemClient`"]

Begin["`Private`"] (* Begin Private Context *) 

resourceTypeNames[rsbase_]:=publicResourceInformation["Names", rsbase,Automatic]

setResourceNameAutocomplete[]:=setResourceNameAutocomplete[Automatic]

setResourceNameAutocomplete[Automatic]:=setResourceNameAutocomplete[System`$ResourceSystemBase]

setResourceNameAutocomplete[rsbase_String]:=With[{names=resourceTypeNames[rsbase]},
	If[ListQ[names],
		setResourceNameAutocomplete[names]
	]
]
		
setResourceNameAutocomplete[names_Association]:=(
	setResourceNameAutocomplete[DeleteDuplicates[Flatten[Values[KeyTake[names,Join[$availableResourceTypes,$defaultAutocompleteResourceTypes]]]]]];
	)
	
setResourceNameAutocomplete[names_List]:=setResourceNameAutocomplete[#,names]&/@{
	"ResourceObject",
	"ResourceData"}
	
setResourceNameAutocomplete[symb_,names_List]:=
With[{cr = Rule[ToString[symb], {names}]},
	FE`Evaluate[FEPrivate`AddSpecialArgCompletion[cr]]];
	
$defaultAutocompleteResourceTypes={"DataResource","NeuralNet"};

publicResourceInformation["Names"]

End[] (* End Private Context *)

EndPackage[]

BeginPackage["Compile`Core`PassManager`PassInformation`"]

PassInformation;
PassInformationQ;
PassInformationClass;
CreatePassInformation;

Begin["`Private`"] 

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]



RegisterCallback["DeclareCompileClass", Function[{st},
PassInformationClass = DeclareClass[
	PassInformation,
	<|
		"toString" -> Function[{}, Self["name"] <> ":: " <> Self["information"]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
		"name",
		"information",
		"description",
		"metadata" -> <||>,
		"requires" -> {},
		"preserves" -> {}
	},
	Predicate -> PassInformationQ
]
]]

CreatePassInformation[name_String, information_String, opts:OptionsPattern[]] :=
	CreatePassInformation[name, information, Association[opts]]
CreatePassInformation[name_String, information_String, description_String, opts:OptionsPattern[]] :=
    CreatePassInformation[name, information, description, Association[opts]]
CreatePassInformation[name_String, information_String, opts_?AssociationQ] :=
    CreatePassInformation[name, information, information, opts]
CreatePassInformation[name_String, information_String, description_String, opts_?AssociationQ] :=
	CreateObject[
		PassInformation,
		<|
			"name" -> name,
			"information" -> information,
			"description" -> description,
			"metadata" -> opts
		|>
	]

(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)

icon := Graphics[Text[
  Style["Pass\nInfo", GrayLevel[0.7], Bold, 
   CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];  
       
toBoxes[var_?PassInformationQ, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		PassInformation,
		var,
  		icon,
  		{
  		    BoxForm`SummaryItem[{"name: ", var["name"]}],
  		    BoxForm`SummaryItem[{"information: ", var["information"]}]
  		},
  		{
  		    BoxForm`SummaryItem[{"description: ", var["description"]}],
            BoxForm`SummaryItem[{"metadata: ", var["metadata"]}],
  		    BoxForm`SummaryItem[{"requires: ", #["toString"]& /@ var["requires"]}],
  		    BoxForm`SummaryItem[{"preserves: ", #["toString"]& /@ var["preserves"]}]
  		}, 
  		fmt
  	]
End[]

EndPackage[]

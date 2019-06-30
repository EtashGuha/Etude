
BeginPackage["Compile`Core`IR`MetaData`"]


CreateMetaData
MetaDataQ


Begin["`Private`"] 

Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]

RegisterCallback["DeclareCompileClass", Function[{st},
DeclareClass[
	MetaDataClass,
	<|
		"getData" -> (getData[Self,##]&), 
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt},
			toBoxes[Self, fmt]
		]
	|>,
	{
		"data"
	},
	Predicate -> MetaDataQ
]
]]


CreateMetaData[ ] :=
	CreateMetaData[<||>]


(*
  Should run the validator on the MetaData
*)
CreateMetaData[data_?AssociationQ] :=
	Module[{obj},
		obj = CreateObject[
			MetaDataClass,
			<|
				"data" -> CreateReference[data]
			|>
		];
		obj
	]

CreateMetaData[args___] :=
    ThrowException[{"Invalid call to CreateMetaData.", {args}}]

getData[self_, key_] :=
	getData[self,key,Missing["KeyAbsent", key]]

getData[self_, key_, def_] :=
	self["data"]["lookup", key, def]

toString[obj_?MetaDataQ] :=
	Module[{data},
		data = Map[ {ToString[First[#]]," -> ",ToString[First[#]],"\n"}&, Normal[obj["data"]["get"]]];
		StringJoin[
			"MetaData[\n"
			,
				data
			,
			"]"
		]
	]


icon := Graphics[Text[
  Style["MD", GrayLevel[0.7], Bold, 
   CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];  
      
toBoxes[obj_?MetaDataQ, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"MetaData",
		"",
  		icon,
  		Prepend[	
  		Map[
  			BoxForm`SummaryItem[{First[#], " ", Last[#]}]&,
  			Normal[obj["data"]["get"]]
  		],
  		BoxForm`SummaryItem[{"", ""}]],
  		{}, 
  		fmt
  	]
 





End[]
EndPackage[]

BeginPackage["Compile`Core`IR`GetElementTrait`"]

GetElementTrait

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]

getElementIndex[ list_, All] :=
        list

getElementIndex[ list_, index_] :=
        Module[ {},
                If[ !IntegerQ[index] || index < 1 || index > Length[list],
                        Null,
                        Part[list,index] ]
        ]

getElement[self_, {i_}] :=
        getElementIndex[ self["getElements"], i]


getElement[self_, {i_, r__}] :=
        Module[ {elem},
                elem = getElementIndex[ self["getElements"], i];
                If[elem === Null, Null, getElement[elem, {r}]]
        ]
        
GetElementTrait = ClassTrait[<|
	"getElement" -> (getElement[Self, {##}]&)
|>]

End[]
EndPackage[]

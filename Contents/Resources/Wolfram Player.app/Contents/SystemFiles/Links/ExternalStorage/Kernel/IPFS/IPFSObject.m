Package["ExternalStorage`"]

PackageExport[IPFSObject]

IPFSObject[args___] := (ArgumentCountQ[IPFSObject, Length[{args}], {1}]; Null /; False)

IPFSObject /:
MakeBoxes[sak : IPFSObject[details_Association?AssociationQ], fmt : StandardForm | TraditionalForm] /; BoxForm`UseIcons :=
    Module[{icon, alwaysGrid, sometimesGrid},

        icon = BoxForm`GenericIcon[LinkObject] (* Placeholder *);
        alwaysGrid = {
            BoxForm`SummaryItem[{"FileName : ", Lookup[details, "FileName", Missing["NotAvailable"]]}],
            BoxForm`SummaryItem[{"Address: ", details["Address"]}]
        };
        sometimesGrid = {
            If[ KeyExistsQ[details,"FileHash"], BoxForm`SummaryItem[{"FileHash: ", details["FileHash"]}], Sequence@@{}]
            (*BoxForm`SummaryItem[{"File: ", details["ConsumerSecret"]}], here goes what always appears, none in this case  *)
        };
        
        BoxForm`ArrangeSummaryBox[IPFSObject, sak, icon, alwaysGrid, sometimesGrid, fmt, "Interpretable" -> True]
]
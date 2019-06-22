(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["NeuralNetResource`"]

Begin["`Private`"] (* Begin Private Context *) 

ResourceSystemClient`Private`repositoryBelowFoldItems[drt:$NeuralNetResourceType,id_, info_]:={
    BoxForm`SummaryItem[{"ByteCount: ", Lookup[info, "ByteCount", ""]}],
    BoxForm`SummaryItem[{"TrainingSetInformation: ", Lookup[info, "TrainingSetInformation", ""]}],
    BoxForm`SummaryItem[{"InputDomains: ", Short[Row[Lookup[info,"InputDomains",{}],","]]}],
    BoxForm`SummaryItem[{"TaskType: ", Lookup[info,"TaskType",""]}],
	BoxForm`SummaryItem[{"Keywords: ", Short[Row[Lookup[info,"Keywords",{}],","]]}],
    BoxForm`SummaryItem[{"Data Location: ", typesetElementStorageLocation[drt,id]}],
    BoxForm`SummaryItem[{"UUID: ", id}],
	BoxForm`SummaryItem[{"Version: ", Lookup[info,"Version",None]}],
	BoxForm`SummaryItem[{"Elements: ", Short[Row[Lookup[info,"ContentElements",{}],","],2]}]
    }


$nnDirectory=DirectoryName[System`Private`$InputFileName];

ResourceSystemClient`Private`resourceIcon[type:$NeuralNetResourceType] := (ResourceSystemClient`Private`resourceIcon[type]=
	ResourceSystemClient`Private`formatresourceicon[Import[FileNameJoin[{$nnDirectory,"Images","neuralNetResourceIcon.png"}], "Graphics"]])
	

End[] (* End Private Context *)

EndPackage[]



SetAttributes[{},
   {ReadProtected, Protected}
];
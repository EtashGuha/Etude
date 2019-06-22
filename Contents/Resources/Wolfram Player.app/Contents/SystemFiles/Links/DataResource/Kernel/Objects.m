(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["DataResource`"]

Begin["`Private`"] (* Begin Private Context *) 

ResourceSystemClient`Private`repositoryBelowFoldItems[drt:$DataResourceType,id_, info_]:={
    BoxForm`SummaryItem[{"Categories: ", Short[Row[Lookup[info,"Categories",{}],","]]}],
    BoxForm`SummaryItem[{"ContentTypes: ", Short[Row[Lookup[info,"ContentTypes",{}],","]]}],
	BoxForm`SummaryItem[{"Keywords: ", Short[Row[Lookup[info,"Keywords",{}],","]]}],
	ResourceSystemClient`Private`summaryResourceLink[info],	
    BoxForm`SummaryItem[{"Data Location: ", typesetElementStorageLocation[drt,id]}],
    BoxForm`SummaryItem[{"UUID: ", id}],
	BoxForm`SummaryItem[{"Version: ", Lookup[info,"Version",None]}],
	BoxForm`SummaryItem[{"Size: ", typesetSize[info]}],
	BoxForm`SummaryItem[{"Elements: ", Short[Row[Lookup[info,"ContentElements",{}],","],2]}]
    }

$drDirectory=DirectoryName[System`Private`$InputFileName];

ResourceSystemClient`Private`resourceIcon[$DataResourceType]:=ResourceSystemClient`Private`resourceicon[
	FileNameJoin[{$drDirectory,"Images","dataResourceIcon.pdf"}]]

End[] (* End Private Context *)

EndPackage[]



SetAttributes[{},
   {ReadProtected, Protected}
];
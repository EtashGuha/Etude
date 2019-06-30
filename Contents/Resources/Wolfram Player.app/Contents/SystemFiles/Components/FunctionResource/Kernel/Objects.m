(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["FunctionResource`"]

Begin["`Private`"] (* Begin Private Context *) 


$frDirectory=DirectoryName[System`Private`$InputFileName];
	
ResourceSystemClient`Private`resourceIcon[type:$FunctionResourceTypes] := (ResourceSystemClient`Private`resourceIcon[type]=
	ResourceSystemClient`Private`formatresourceicon[Import[FileNameJoin[{$frDirectory,"Images","FunctionResourceIcon.png"}], "Graphics"]])

End[] (* End Private Context *)

EndPackage[]



SetAttributes[{},
   {ReadProtected, Protected}
];
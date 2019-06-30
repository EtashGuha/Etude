(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["DataResource`"]

Begin["`Private`"] (* Begin Private Context *) 

$exampleNotebookTemplate:=($exampleNotebookTemplate=FileNameJoin[{$drDirectory,"Templates","DataResourceExampleNotebookTemplate.nb"}])

ResourceSystemClient`Private`repositoryCreateBlankExampleNotebook[$DataResourceType,id_,name_]:=Block[{nb},
	nb=GenerateDocument[$exampleNotebookTemplate,Association[
		"UUID"->id,
		"Resource"->ResourceObject[id]
	]
	];
	SetOptions[nb,"WindowTitle" -> "Examples for "<>ToString[name],DockedCells->{},Visible->True];
	nb
]  

End[] (* End Private Context *)

EndPackage[]



SetAttributes[{},
   {ReadProtected, Protected}
];
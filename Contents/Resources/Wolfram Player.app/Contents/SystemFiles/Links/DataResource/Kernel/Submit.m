(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["DataResource`"]

Begin["`Private`"] (* Begin Private Context *) 

ResourceSystemClient`Private`repositoryValidateSubmission[$DataResourceType,id_, info_]:=validateSubmission[id, info]

validateSubmission[id_, info_]:=info/;KeyExistsQ[info,"Content"]

validateSubmission[id_, info0_]:=Block[{location=info0["ContentElementLocations"], content, info=KeyDrop[info0,"ContentElementLocations"]},
	Switch[Head[location],
		System`File|LocalObject,
		content=Import[location];
		If[content=!=$Failed,
			Join[info,Association["Content"->importlocal[location],"Asynchronous"->False]]
			,
			Throw[$Failed]
		]
		,
		CloudObject,
		info0,
		String,
		If[FileExistsQ[location],
			Join[info,Association["Content"->importlocal[location],"Asynchronous"->False]]
			,
			If[!StringFreeQ[location,$CloudBase],
				Append[info,"ContentElementLocations"->CloudObject[location]]
				,
				Message[ResourceSubmit::invcon];Throw[$Failed]
			]
		],
		Association,
		multipartValidateSubmission["DataResource",id, info0]
	]
]/;KeyExistsQ[info0,"ContentElementLocations"]

validateSubmission[___]:=$Failed

multipartValidateSubmission[$DataResourceType,id_, info_]:=Block[{locations, content, local, cloud, values},
	If[Length[Lookup[info,"ContentElements",{}]]===0,
		Message[ResourceSubmit::noncont];Throw[$Failed]
	];
	If[Lookup[info,"Content",{}]=!={},Message[ResourceSubmit::invcon];Throw[$Failed]];
	locations=Lookup[info,"ContentElementLocations"];
	values=Lookup[info,"ContentValues",Association[]];
	If[!AssociationQ[locations],Message[ResourceSubmit::invcon];Throw[$Failed]];
	cloud=Select[locations,(MatchQ[Head[#], CloudObject] &)];
	local=Select[locations,((MatchQ[Head[#], System`File | LocalObject])||Quiet[stringFileExistsQ[#]])&];
	If[Complement[Keys[locations],Keys[cloud],Keys[local], Keys[values]]=!={},Message[ResourceSubmit::invcon];Throw[$Failed]];
	content=Join[importlocal/@local,values];
	locations=Join[cloud,Automatic&/@local,Automatic&/@values];
	Join[KeyDrop[info,"ContentValues"],Association["ContentElementLocations"->locations,"Content"->content],
		Association@If[Length[local]>0,"Asynchronous"->Automatic,{}]]
]

ResourceSystemClient`Private`repositorycompleteResourceSubmission[$DataResourceType, id_,as_]:=completeResourceSubmissionWithElements["DataResource",id, as]

ResourceSystemClient`Private`deployResourceSubmissionContentSizeLimit[$DataResourceType]=10^5;

ResourceSystemClient`Private`repositoryCreateSubmissionObject[$DataResourceType,res_Association, as_Association]:=dataResourceSubmissionSuccess[res,as]/;KeyExistsQ[as,"ContentElementLocations"]

dataResourceSubmissionSuccess[res_,as_]:=Success["ResourceSubmission", Association[
	"MessageTemplate" -> ResourceSubmit::subsuc,
	"MessageParameters" -> KeyTake[res,"SubmissionID"],
  	Join[res,KeyTake[as,"ContentElementLocations"]]]
]

End[] (* End Private Context *)

EndPackage[]

SetAttributes[{},
   {ReadProtected, Protected}
];
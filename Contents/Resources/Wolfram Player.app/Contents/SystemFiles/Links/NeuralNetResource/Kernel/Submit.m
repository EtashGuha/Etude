(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["NeuralNetResource`"]

Begin["`Private`"] (* Begin Private Context *) 

ResourceSystemClient`Private`repositoryValidateSubmission[$NeuralNetResourceType,id_, info_]:=validateSubmission[id, info]

validateSubmission[id_, info_]:=info/;KeyExistsQ[info,"Content"]

validateSubmission[id_, info0_]:=Block[{location=info0["ContentElementLocations"], content, info=KeyDrop[info0,"ContentElementLocations"]},
	Switch[Head[location],
		System`File|LocalObject,
		If[!FileExistsQ@location, Throw@$Failed];
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
		multipartValidateSubmission["NeuralNet",id, info0]
	]
]/;KeyExistsQ[info0,"ContentElementLocations"]

validateSubmission[___]:=$Failed

multipartValidateSubmission[$NeuralNetResourceType,id_, info_]:=Block[{locations, content, local, cloud, values},
	If[Length[Lookup[info,"ContentElements",{}]]===0,
		Message[ResourceSubmit::noncont];Throw[$Failed]
	];
	If[Lookup[info,"Content",{}]=!={},Message[ResourceSubmit::invcon];Throw[$Failed]];
	locations=Lookup[info,"ContentElementLocations"];
	If[!AssociationQ[locations],Message[ResourceSubmit::invcon];Throw[$Failed]];
	cloud=Select[locations,(MatchQ[Head[#], CloudObject] &)];
	If[KeyExistsQ[info,"ElementInformation"],
		ResourceSystemClient`Private`setSubmissionCloudObjectMetadata[cloud,info["ElementInformation"]]
	];
	local=Select[locations,((MatchQ[Head[#], System`File | LocalObject])||Quiet[stringFileExistsQ[#]])&];
	values=Lookup[info,"ContentValues",Association[]];
	If[Complement[Keys[locations],Keys[cloud],Keys[local],Keys[values]]=!={},Message[ResourceSubmit::invcon];Throw[$Failed]];
	locations=Join[cloud, local];
	Join[KeyDrop[info,{"ContentValues","ElementInformation"}],Association["ContentElementLocations"->locations,"Content"->values],
		Association@If[Length[local]>0,"Asynchronous"->Automatic,{}]]
]


ResourceSystemClient`Private`repositorycompleteResourceSubmission[$NeuralNetResourceType, id_,as_]:=completeResourceSubmissionWithElements["NeuralNet",id, as]

ResourceSystemClient`Private`deployResourceSubmissionContentSizeLimit[$NeuralNetResourceType]=0;

End[] (* End Private Context *)

EndPackage[]

SetAttributes[{},
   {ReadProtected, Protected}
];
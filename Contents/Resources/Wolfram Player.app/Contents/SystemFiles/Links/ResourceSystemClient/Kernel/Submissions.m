(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["ResourceSystemClient`"]
(* Exported symbols added here with SymbolName::usage *)  

ResourceSystemClient`SubmissionNotebook

Begin["`Private`"] (* Begin Private Context *) 

submissionInfo[_]:=Association[]

getSubmissionInfo[n_Integer]:=getSubmissionInfo[IntegerString[n, 10, 4]]

getSubmissionInfo[id_String]:=Block[{res, url},
	res=apifun["GetSubmission",{"SubmissionID"->id,"Element"->"Submission"},System`ResourceSubmissionObject];
	If[KeyExistsQ[res,"ResourceType"],
		loadResourceType[getResourceType[res]];
		System`ResourceSubmissionObject[res]
		,
		$Failed
	]
]

getSubmissionInfo[___]:=$Failed

submissionRequest[info_,"Status",args_]:=getSubmissionStatus[Lookup[info,"UUID",Lookup[info,"SubmissionID"]]]

submissionRequest[info_,key_,args_]:=info[key]/;KeyExistsQ[info, key]

submissionRequest[info_,req_,args_]:=(Message[ResourceSubmissionObject::unkreq,req];$Failed)

importSubmission[str_]:=getSubmissionInfo[str]

getSubmissionStatus[n_Integer]:=getSubmissionStatus[IntegerString[n, 10, 4]]

getSubmissionStatus[id_String]:=Block[{res},
	res=apifun["SubmissionStatus",{"SubmissionID"->id}, System`ResourceSubmissionObject];
	If[KeyExistsQ[res,"SubmissionStatus"],
		Lookup[res["SubmissionStatus"],"Submission"]
		,
		Missing["Not Found"]
	]
]

getSubmissionStatus[___]:=$Failed

ResourceSystemClient`SubmissionNotebook[args___]:=Catch[getSubmissionNotebook[args]]

getSubmissionNotebook[sub_ResourceSubmissionObject]:=getSubmissionNotebook[sub[[1,"UUID"]]]

getSubmissionNotebook[n_Integer]:=getSubmissionNotebook[IntegerString[n, 10, 4]]

getSubmissionNotebook[id_String]:=Block[{res, url, submissions},
	res=apifun["SubmissionNotebook",{"SubmissionID"->id},ResourceObject];
	If[Head[res]===Notebook,
		NotebookPut[res],
		$Failed		
	]
]



End[] (* End Private Context *)

EndPackage[]

SetAttributes[{},
   {ReadProtected, Protected}
];
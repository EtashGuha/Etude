(* Wolfram Language Package *)

BeginPackage["ResourceSystemClient`"]

ResourceSystemClient`ExampleNotebook
ResourceSystemClient`CreateExampleNotebook

Begin["`Private`"] (* Begin Private Context *) 

exampleNotebookLocation[id_]:=examplenotebookLocation[resourceDirectory[id]]

examplenotebookLocation[dir_]:=examplenotebooklocation[dir]/;DirectoryQ[dir]

examplenotebookLocation[_]:=None

examplenotebooklocation[dir_]:=FileNameJoin[{dir,"examples.nb"}]

blankExampleNotebook[_,id_]:=Notebook[
	{
	Cell[CellGroupData[{Cell["Basic Examples","Subsection"],
		Cell[CellGroupData[{Cell["First Example","Subsubsection"],
			Cell["Caption 1.","Text"],
			Cell[BoxData[RowBox[{"ResourceObject","[","...","]"}]],"Input"]},System`Open]],
		Cell[CellGroupData[{Cell["Second Example","Subsubsection"],
	Cell["Caption 2.","Text"],Cell[BoxData[RowBox[{"fun","[",RowBox[{RowBox[{"ResourceObject","[","...","]"}],",","arg"}],"]"}]],"Input"]},System`Open]]},System`Open]],
	Cell["Visualization","Subsection"],
	Cell["Analysis","Subsection"]
	}
	,StyleDefinitions->"Default.nb"]
  

saveExampleNotebook[id_, nb_]:=With[{file=exampleNotebookLocation[id]},
	NotebookSave[nb, file];
	file
]

createBlankExampleNotebook[rtype_,id_,name_]:=Block[{nb,file},
	loadResourceType[rtype];
	nb=repositoryCreateBlankExampleNotebook[rtype,id,name];
	If[Head[nb]===NotebookObject,
		If[MemberQ[$localResources,id],
			file=saveExampleNotebook[id, nb];
			setResourceInfo[id, Association["ExampleNotebook"->file]]
			,
			setResourceInfo[id, Association["ExampleNotebook"->nb]];
		];
		nb
		,
		$Failed
	]
]


repositoryCreateBlankExampleNotebook[rtype_,id_,name_]:=Block[{nb},
	nb=NotebookPut[blankExampleNotebook[rtype,id]];
	SetOptions[nb,"WindowTitle" -> "Examples for "<>ToString[name]];
	nb
]  

deployExampleNotebook[nbo_NotebookObject]:=deployExampleNotebook[NotebookGet[nbo]]

deployExampleNotebook[nb_Notebook]:=CloudDeploy[nb,Permissions->{$ResourceSystemAdminUser->"Read"}]

deployExampleNotebook[___]:=$Failed

ResourceSystemClient`CreateExampleNotebook[args___]:=Catch[createExampleNotebook[args]]

createExampleNotebook[args___]:=Block[{resourcesystemExampleNotebook},
	resourcesystemExampleNotebook[___]:=$Failed;
	exampleNotebook[args]
]


ResourceSystemClient`ExampleNotebook[args___]:=Catch[exampleNotebook[args]]

exampleNotebook[ro_System`ResourceObject]:=exampleNotebook[resourceObjectID[ro]]

exampleNotebook[id_String]:=getnotebook["Example",id]/;uuidQ[id]

exampleNotebook[___]:=$Failed

examplenotebook[args___]:=getnotebook["Example",args]

getnotebook[nbtype_,id_String]:=getnotebook[nbtype,{id, getResourceInfo[id]}]/;MemberQ[$localResources,id]

getnotebook[nbtype_,id_String]:=getnotebook[nbtype,{id, resourceInfo[id]}]/;MemberQ[$loadedResources,id]

getnotebook[nbtype_,id_String]:=With[{info=ResourceSystemClient`Private`loadResource[id]},
	If[AssociationQ[info],
		getnotebook[nbtype,{id, info}]
		,
		$Failed
	]	
]

getnotebook[nbtype_,info_Association]:=getnotebook[nbtype,{info["UUID"],info}]

getnotebook["Example",{id_String, info_Association}]:=(customExampleNotebook[id, info])/;userdefinedResourceQ[info]
getnotebook["Example",{id_String, info_Association}]:=resourcesystemExampleNotebook[id, info]/;marketplacebasedResourceQ[info]

getnotebook["Definition",{_String, KeyValuePattern[{"DefinitionNotebook"->as_Association}]}]:=With[
	{nb=ImportByteArray[as["Data"],as["Format"]]},
	If[Head[nb]===Notebook,
		NotebookPut[nb]
		,
		$Failed
	]
]/;KeyExistsQ[as,"Data"]&&KeyExistsQ[as,"Format"]
	
getnotebook["Definition",{_String, nb_Notebook}]:=NotebookPut[nb]
getnotebook["Definition",{_String, nbo_Notebook}]:=SetSelectedNotebook[nbo]

getnotebook["Definition",{id_String, info_Association}]:=resourcesystemDefinitionNotebook[id, info]/;marketplacebasedResourceQ[info]
	
getnotebook[___]:=$Failed

customExampleNotebook[id_, info_]:=Block[{nb=info["ExampleNotebook"]},
	Switch[nb,
		_String,NotebookOpen[nb],
		_NotebookObject,SetOptions[nb, Visible -> True];SetSelectedNotebook[nb],
		_,createBlankExampleNotebook[getResourceType[info],id,Lookup[info,"Name"]]
	]	
]/;KeyExistsQ[info,"ExampleNotebook"]

customExampleNotebook[id_, info_]:=With[{nb=openLocalExampleNotebook[id]},
	If[!FailureQ[nb],
		nb,
		createBlankExampleNotebook[getResourceType[info],id,Lookup[info,"Name"]]
	]	
]

openLocalExampleNotebook[id_,warn_:False]:=Block[{file=exampleNotebookLocation[id]},
	If[TrueQ[Quiet[fileExistsQ[file]]],
		NotebookOpen[file]
		,
		If[TrueQ[warn],
			Message[ResourceObject::depnbcl];
			$Failed
		]
	]	
]

resourcesystemExampleNotebook[id_, info_]:=With[{nb=info["ExampleNotebook"]},
	If[FileExistsQ[nb],
		NotebookOpen[nb]
		,
		resourcesystemExampleNotebook[id, KeyDrop[info,"ExampleNotebook"]]
	]	
]/;KeyExistsQ[info,"ExampleNotebook"]

resourcesystemExampleNotebook[id_, info_]:=Block[{res, nb},
	res=apifun["GetExampleNotebook",{"UUID"->id}, 
		ResourceObject, resourcerepositoryBase[info]];
	If[KeyExistsQ[res,"ExampleNotebook"],
		nb=NotebookOpen[res["ExampleNotebook"]];
		If[Head[nb]===NotebookObject,
			setResourceInfo[id, Association["ExampleNotebook"->res["ExampleNotebook"]]];
			SetOptions[nb,Visible->True];
			nb
			,
			$Failed
		]
	]
]

resourcesystemDefinitionNotebook[id_, info_]:=Block[{res, nb},
	res=Quiet[apifun["GetDefinitionNotebook",{"UUID"->id}, 
		ResourceObject, resourcerepositoryBase[info]],ToExpression::sntx];
	If[Quiet[KeyExistsQ[res,"DefinitionNotebook"]],
		nb=NotebookOpen[res["DefinitionNotebook"]];
		If[Head[nb]===NotebookObject,
			setResourceInfo[id, Association["DefinitionNotebook"->res["DefinitionNotebook"]]];
			SetOptions[nb,Visible->True];
			nb
			,
			$Failed
		],
		$Failed
	]
]

validateexamplenotebook[co:HoldPattern[_CloudObject]]:=verifyReviewerPermissions[co]

validateexamplenotebook[nb:(_NotebookObject|_Notebook)]:=With[{res=deployExampleNotebook[nb]},
    If[Head[res]=!=CloudObject,
        Message[ResourceSubmit::enbdf];
        Throw[$Failed]
    ];
    res
]

validateexamplenotebook[file_String]:=With[{nb=Get[file]},
    If[Head[nb]=!=Notebook,
        Message[ResourceSubmit::enbdf];
        Throw[$Failed]
    ];
    validateexamplenotebook[nb]
]/;FileExistsQ[file]

validateexamplenotebook[expr_]:=(Message[System`ResourceSubmit::invparam, "ExampleNotebook"];Throw[$Failed])



End[] (* End Private Context *)

EndPackage[]
(* Wolfram Language Package *)


BeginPackage["NeuralNetResource`"]

Begin["`Private`"] (* Begin Private Context *) 

$NotebookToolsDirectory=FileNameJoin[{$nnDirectory,"Templates"}];

$CreateNeuralNetResourceBlank=FileNameJoin[{$NotebookToolsDirectory,"NeuralNetResourceDefinition.nb"}];

ResourceSystemClient`Private`createResourceNotebook[$NeuralNetResourceType,rest___]:=newNeuralNetResourceDefinitionNotebook[]

newNeuralNetResourceDefinitionNotebook[]:=With[{nbo=newfunctionResourceDefinitionNotebook[]},
	If[Head[nbo]===NotebookObject,    
		SetOptions[nbo,{Visible->True}];  
		SetSelectedNotebook[nbo];
		nbo
		,
		$Failed
	]
]
newfunctionResourceDefinitionNotebook[]:=With[{nb=Get[$CreateNeuralNetResourceBlank]},
	NotebookPut[prepareNeuralNetResourceCreateNotebook[nb]]	
]

prepareNeuralNetResourceCreateNotebook[nb_]:=With[{uuid=CreateUUID[]},
	Replace[nb,{ResourceSystemClient`Private`$temporaryuuid:>uuid,
		"ResourceSystemClient`Private`$temporaryuuid":>uuid},Infinity]
	]


(******************************************************************************)


ResourceSystemClient`Private`scrapeDefinitonNotebookContent[$NeuralNetResourceType,id_, nb_]:=scrapeNeuralNetDefinitonNotebookContent[id, nb]

scrapeNeuralNetDefinitonNotebookContent[ id_, nb_ ] :=With[{enet=getNotebookEvaluationNet[nb]},
	With[{unet=untrainScrapedNet[enet],cnb=scrapeConstructionNotebook[nb]},
		With[{rules=
			{
				If[elementlocationQ[enet],
				"ContentElementLocations"->{"EvaluationNet"->enet},
				"ContentElements"->{"EvaluationNet"->enet}
				],
				If[elementlocationQ[cnb],
				"ContentElementLocations"->{"ConstructionNotebookExpression"->cnb},
				"ContentElements"->{"ConstructionNotebookExpression"->cnb}
				],
			"ContentElementFunctions"->openConstructionNBFunction[cnb]
			}},
			DeleteMissing@Association[Merge[rules,Association],
				"ResourceType"->"NeuralNet",
				"DefaultContentElement"->"EvaluationNet"]
		]
	]
]

untrainScrapedNet[___]:=Missing[] (* TODO *)

scrapeConstructionNotebook[nb_]:=With[{loc=scrapeConstructionNotebookLocation[nb]},
	If[MatchQ[loc,_CloudObject|_LocalObject|_NotebookObject],
		loc,
		If[StringQ[loc]&&FileExistsQ[loc],
			loc,
			scrapeConstructionNotebookCells[nb]
		]
	]	
]

scrapeConstructionNotebookLocation[nb_]:=With[{symb=Symbol[$Context <> "$$NBLocation"]},
	Block[{cells=findCellTags[nb, "NeuralNetResourceConstructionLink"], data},
	If[Length[cells]===1,
		ToExpression[cells[[1,1]]];
		symb
		,
		Missing[]
	]
]
]

scrapeConstructionNotebookCells[nb_]:=With[{groups=Cases[nb, 
	HoldPattern[CellGroupData][{_?(!FreeQ[#, "ResourceConstructionArea"] &), ___}, ___], 8]},
	If[Length[groups]>0,
    	scrapeconstructionNotebook[First[groups], nb]
    	,
    	Missing[]
    ]
]

scrapeconstructionNotebook[group_, nb_]:=
	Notebook[{
		ResourceSystemClient`Private`cleanScrapedNotebookExamples["NeuralNet",group]}, 
		DockedCells->{},
		Options[nb, StyleDefinitions], 
		Background->White]


openConstructionNB[m_MissingQ]:=Missing[]
openConstructionNBFunction[_]:=With[{func=openconstructionNBFunction[]},
	Association["ConstructionNotebook"->func]	
]

openconstructionNBFunction[]:=(With[{nb=#ConstructionNotebookExpression},
	Switch[Head[nb],
		Notebook,NotebookPut[nb],
		NotebookObject,SetSelectedNotebook[nb],
		CloudObject,NotebookOpen[nb],
		LocalObject,NotebookPut[Import[nb]],
		String,If[FileExistsQ[nb],NotebookPut[Import[nb]],$Failed],
		_,$Failed]]&
)


getNotebookEvaluationNet[nb_] := Quiet[With[{symb=Symbol[$Context <> "$$NetLocation"]},
	Block[{NeuralNetResource`$$ImportNet=Identity},
  		ToExpression[Cases[nb, Cell[cont_, "ResourceContentInput", ___] :> cont, 30]];
  		symb
	]
  ]]


NeuralNetResource`$$ImportNet=importNeuralNetResourceNotebookContent;

importNeuralNetResourceNotebookContent[obj:(_LocalObject)]:=Import[obj,"WLNet"]
importNeuralNetResourceNotebookContent[obj:(_File)]:=Import[First[obj],"WLNet"]
importNeuralNetResourceNotebookContent[co_CloudObject]:=Catch[
	Block[{tempfile=FileNameJoin[{$TemporaryDirectory,CreateUUID[]<>".wlnet"}]},
	ResourceSystemClient`Private`urlDownloadWithProgress[co,tempfile,CloudObjectInformation[co,"FileByteCount"]];
	With[{content=Import[tempfile]},
			DeleteFile[tempfile];
			If[FailureQ[content],
				$Failed,
				content
			]
		]	
]]




importNeuralNetResourceNotebookContent[expr_]:=expr

elementlocationQ[_CloudObject|_LocalObject|_File|_URL]:=True
elementlocationQ[str_String]:=FileExistsQ[str]
elementlocationQ[_]:=False

ResourceSystemClient`Private`scrapeResourceTypeProperties[$NeuralNetResourceType, id_, nb_]:=With[
	{wlVersion=scrapeNNByCellTag[nb,"ResourceWLVersion"],
	tsi=scrapeNNTSI[nb],
	tsd=scrapeDataLocation[nb,"ResourceTrainingSetData"], 
	testing=scrapeDataLocation[nb,"ResourceTestingData"]},
	DeleteMissing@Association[
		"WolframLanguageVersionRequired"->wlVersion,
		"TrainingSetInformation"->tsi[[1]],
		"TrainingSetInformationLinks"->tsi[[2]],
		"TrainingSetData"->tsd,
		"TestingData"->testing (* TODO: add support for "TestingData" in submission *)
	]
]

scrapeNNTSI[nb_]:=With[{cells=findCellTags[nb, "ResourceTrainingSetInformation"]},
	If[Length[cells]=!=1,{Missing[],Missing[]},
		splitTSILinks[cells[[1,1]]]
	]
]

splitTSILinks[str_String]:={str,Missing[]};
splitTSILinks[HoldPattern[TextData][{str_String}]]:={str,Missing[]};
splitTSILinks[HoldPattern[TextData][td_]]:=replaceLinkBoxes[td]

replaceLinkBoxes[td_]:=Block[{$links={}, new},
	new=Flatten[ReplaceRepeated[td,{HoldPattern[Cell[c_,___]]:>c,bd_BoxData:>With[{split=splitlinks[bd]},
		AppendTo[$links,split];split[[1]]]}]];
	If[MatchQ[new,{_String...}],
		{StringJoin[new],Normal@DeleteMissing[Association[$links]]},
		new
	]	
]

splitlinks[bd_]:=With[{rules=Cases[bd,hbox:HoldPattern[BoxData[TemplateBox[{label_,___},"HyperlinkURL"]]]:>(ToExpression[label]->ToExpression[hbox]),{0,Infinity}]},
	If[Length[rules]===1,
		First[rules],
		{""->Missing[]}
	]
]

scrapeNNByCellTag[nb_, tag_]:=scrapeonetextCell[findCellTags[nb, tag]]
scrapeDataLocation[nb_,tag_]:=Block[{cells=findCellTags[nb, tag], cell,res},
	If[Length[cells]===1,
		cell=First[cells];
		res=ResourceSystemClient`Private`scrapeRelatedResourceUUIDs[cell];
		If[Quiet[AllTrue[res,uuidQ]],
			res,
			DeleteMissing[Flatten[ResourceSystemClient`Private`scrapeExternalLinks[cell]]]/.{}->Missing[]
		]
		,
		Missing[]
	]
]

End[] (* End Private Context *)

EndPackage[]
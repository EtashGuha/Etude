(* Wolfram Language Package *)

BeginPackage["NeuralNetResource`"]
(* Exported symbols added here with SymbolName::usage *)  
Begin["`Private`"] (* Begin Private Context *) 

DataResource`resourceTypeData[$NeuralNetResourceType, {id_, info_},
	elem:("UninitializedEvaluationNet"|"EvaluationNet"), rest___]:=Block[
		{DataResource`Private`resourcedatauncached=wlnetimport},
		DataResource`resourceDataElement[{id, info},elem, rest]
	]


wlnetimport[lo:HoldPattern[_LocalObject]]:=Import[lo,"WLNet"]
wlnetimport[file:File[path_]]:=Import[file[[1]],"WLNet"]
wlnetimport[co:HoldPattern[_CloudObject]]:=CloudImport[co,"WLNet"]
wlnetimport[str_String]:=Import[str,"WLNet"]/;FileExistsQ[str]
wlnetimport[args___]:=NeuralNetworks`WLNetImport[args]

DataResource`resourceTypeData[$NeuralNetResourceType, {id_, info_},elem_, rest___]:=DataResource`resourceDataElement[{id, info},elem, rest]

DataResource`Private`cacheResourceContentInMemory["NeuralNet", _] := True /; $CloudEvaluation

ResourceSystemClient`Private`repositoryElementFormat["NeuralNet",_,_,_URL,Automatic]:=Automatic
ResourceSystemClient`Private`repositoryElementFormat["NeuralNet",__,Automatic]:="WLNet"


End[] (* End Private Context *)

EndPackage[]

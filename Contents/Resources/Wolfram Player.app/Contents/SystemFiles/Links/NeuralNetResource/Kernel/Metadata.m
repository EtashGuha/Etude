(* Wolfram Language Package *)

BeginPackage["NeuralNetResource`"]

Begin["`Private`"] (* Begin Private Context *) 


$ElementNameLengthLimit=200;
ResourceSystemClient`Private`respositoryMetadataSchema[$NeuralNetResourceType]:=(
	ResourceSystemClient`Private`respositoryMetadataSchema[$NeuralNetResourceType]=
	Association[
		"Content"->(With[{expr = #}, Interpreter[Evaluate, (SameQ[expr, #] &)][#]] &),
		"DefaultContentElement"->Restricted["String", RegularExpression[".{1,"<>ToString[$ElementNameLengthLimit]<>"}"]],
		"ContentElements"->RepeatingElement[Restricted["String", RegularExpression[".{1,"<>ToString[$ElementNameLengthLimit]<>"}"]]]
	]
	)


validateParameter[$NeuralNetResourceType, "ContentElementHashes", x_Association /; AllTrue[Values[x], IntegerQ]] := x
validateParameter[$NeuralNetResourceType, "ByteCount", x_ /; (IntegerQ[x] && (x > 0))] := x
validateParameter[$NeuralNetResourceType, "TrainingSetData", string_String] := string
validateParameter[$NeuralNetResourceType, "TrainingSetData", link:(_URL|_Hyperlink)] := link
validateParameter[$NeuralNetResourceType, "TrainingSetInformation", string_String] := string
validateParameter[$NeuralNetResourceType, "TrainingSetInformationLinks", rules:{HoldPattern[Rule][_String,_Hyperlink]...}] := rules
validateParameter[$NeuralNetResourceType, "TrainingSetInformationLinks", rule:HoldPattern[Rule][_String,_Hyperlink]] := {rule}
validateParameter[$NeuralNetResourceType,"TaskType",list_List]:=list
validateParameter[$NeuralNetResourceType,"InputDomains",list_List]:=list
validateParameter[$NeuralNetResourceType,"Performance",str_String]:=str/;StringLength[str]<10^4
validateParameter[$NeuralNetResourceType,"InformationElements",as_]:=as/;AssociationQ[as]
validateParameter[$NeuralNetResourceType,"ContentElementFunctions",as_]:=as/;AssociationQ[as]
validateParameter[$NeuralNetResourceType, "ParameterizationData", expr_] := expr
  
validateParameter[$NeuralNetResourceType,"ContentElementLocations",Automatic]=Automatic;
validateParameter[$NeuralNetResourceType,"ContentElementLocations",as_Association]:=validateParameter["NeuralNet","ContentElementLocations",#]&/@as
validateParameter[$NeuralNetResourceType,"ContentElementLocations",co:HoldPattern[_CloudObject]]:=With[{res=ResourceSystemClient`Private`verifyReviewerPermissions[co]},
	If[Head[res]=!=CloudObject,
		Message[ResourceSubmit::appperms,co];Throw[$Failed]];
	co]
validateParameter[$NeuralNetResourceType,"ContentElementLocations",local:HoldPattern[_File|_LocalObject|_String]] := local/;fileExistsQ[local]

ResourceSystemClient`Private`formatResourceMetadata[id_String, "TrainingSetData", ___] := trainingDataResource[id]/;uuidQ[id]
ResourceSystemClient`Private`formatResourceMetadata[url_String, "TrainingSetData", ___] := URL[url]/;StringContainsQ[url,"://"]
ResourceSystemClient`Private`formatResourceMetadata[url:(_URL|_Hyperlink), "TrainingSetData", ___] := url

ResourceSystemClient`Private`formatResourceMetadata[_, "TrainingSetData", ___] := Missing["NotAvailable"]

trainingDataResource[id_String] := With[{ro = ResourceObject[id]},
  If[ResourceSystemClient`Private`resourceObjectQ[ro],
   ro, Missing[]]]
trainingDataResource[_] := Missing[]

End[] (* End Private Context *)

EndPackage[]
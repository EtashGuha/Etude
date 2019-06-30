(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {System`ResourceSubmit,
"System`PublisherID",
"System`$PublisherID"}

BeginPackage["ResourceSystemClient`"]
(* Exported symbols added here with SymbolName::usage *)  

System`ResourceSubmit
System`PublisherID
System`$PublisherID

Begin["`Private`"] (* Begin Private Context *) 

$SubmissionSizeLimit=10^6;

$submissionNotes=None;
$localSubmissionNotes=None;
Options[System`ResourceSubmit]=Options[resourceSubmit]={System`PublisherID:>System`$PublisherID, 
	System`ResourceSystemBase:>System`$ResourceSystemBase,"SubmissionNotes"->None};

System`ResourceSubmit[args___]:=Catch[resourceSubmitCloudConnect[args]]

resourceSubmitCloudConnect[ro_,opts:OptionsPattern[System`ResourceSubmit]]:=resourceSubmitCloudConnect[ro,None,opts]

resourceSubmitCloudConnect[ro_,update_,opts:OptionsPattern[System`ResourceSubmit]]:=resourceSubmit[ro,update, opts]/;requestBaseConnected[
	OptionValue[System`ResourceSubmit, {opts}, System`ResourceSystemBase]]

resourceSubmitCloudConnect[ro_,update_,opts:OptionsPattern[System`ResourceSubmit]]:=(
	cloudConnect[System`ResourceSubmit, tocloudbase[OptionValue[System`ResourceSubmit, {opts}, System`ResourceSystemBase]]];
	resourceSubmit[ro,update, opts]
)

resourceSubmitCloudConnect[___]:=(Message[ResourceSubmit::iopts];$Failed)

$submissionPublisherID=Automatic;

resourceSubmit[expr_,opts:OptionsPattern[]]:=resourceSubmit[expr,None,opts]

resourceSubmit[nbo_NotebookObject,update_,OptionsPattern[]]:=Catch[submitResourceFromNotebook[nbo,update]]

resourceSubmit[ro_System`ResourceObject, update_,OptionsPattern[]]:=Block[{$submissionPublisherID=OptionValue[PublisherID], 
	resourcebase=OptionValue[System`ResourceSystemBase],$progressID=Unique[], res},
	$submitProgressContent="Checking resource \[Ellipsis]";
	res=resourceSubmitRO[ro, update,resourcebase];
	clearTempPrint[$progressID];
	res
]

resourceSubmit[___]:=(Message[ResourceSubmit::noro];$Failed)

$submissionResourceBase=Automatic;

resourcesubmit[as_Association,update_,resourcebase_]:=Block[{$submissionResourceBase=resourcebase},
	With[{fullparams=completeResourceSubmission[as]},
		submitresourceToSystem[fullparams,update,resourcebase]
	]
]

resourcesubmit[___]:=$Failed

resourceSubmitRO[ro:HoldPattern[System`ResourceObject][as_Association],update_, resourcebase_]:=With[{id=Lookup[as,"UUID",Throw[$Failed]]},
	If[MemberQ[$loadedResources,id],
		resourceSubmitRO[id,resourceInfo[id], update,resourcebase]
		,
		resourceSubmitRO[id,as, update,resourcebase]
	]
]

resourceSubmitRO[id_,info_,update_, _]:=(Message[System`ResourceSubmit::exists];$Failed)/;!userdefinedResourceQ[info]

resourceSubmitRO[id_, info_, update0_,resourcebase_]:=With[{rtype=getResourceType[info],
	update=standardizeSubmissionUpdate[update0,resourcebase]},	
	If[!StringQ[rtype],
		Message[ResourceSubmit::invrt,rtype];Throw[$Failed]
	];
	loadResourceType[rtype];
	$submissionPublisherID=checkPublisherID[rtype,$submissionPublisherID];
	
	resourcesubmit[
		$submitProgressContent="Checking resource \[Ellipsis]";
		repositoryValidateSubmission[rtype,id, info]
		,
		update
		,
		resourcebase
	]
	
]

repositoryValidateSubmission[_,id_, info_]:=info

resourceSubmitRO[___]:=$Failed

completeResourceSubmission[as_Association]:=With[{rtype=getResourceType[as]},
	loadResourceType[rtype];
	repositorycompleteResourceSubmission[rtype,Lookup[as,"UUID"],KeyDrop[as,$nonsubmittedParameters]]]/;KeyExistsQ[as,"ResourceType"]

	
completeResourceSubmission[as_Association]:=With[{rtype=promptForResourceType[]},
    If[!StringQ[rtype],Throw[$Failed]];
	completeResourceSubmission[Append[as,"ResourceType"->rtype]]]

$nonsubmittedParameters={"Version","ContentSize","RepositoryLocation","ResourceLocations","DownloadedVersion"}

repositorycompleteResourceSubmission[rtype_,_,as0_]:=Block[{missingKeys, as, values},
	loadResourceType[rtype];
	$submitProgressContent="Checking resource \[Ellipsis]";
	as=DeleteMissing[AssociationMap[validateParameter[rtype,#]&,as0]];
	If[!FreeQ[as,_Failure],Message[ResourceSubmit::invprop];Throw[$Failed]];
	missingKeys=Complement[Keys[requiredparameters[rtype]],Keys[as]];
	values=promptForMissingKeys[rtype, KeyTake[requiredparameters[rtype],missingKeys]];
	If[!AssociationQ[values],Throw[$Failed]];
	Join[as, values]
]

promptForResourceType[]:=FormFunction[{"ResourceType"->{"DataResource"}},#ResourceType&][]

promptForMissingKeys[rtype_, {}|Association[]]:=Association[]
promptForMissingKeys[rtype_, signature_]:=FormFunction[signature,#&][]

submitresourceToSystem[as_, update_,resourcebase_]:=Block[{res, params},
	params=	DeleteMissing@Association[as,
			"PublisherID"->$submissionPublisherID,
			"UpdatedResource"->If[update===None,Missing[],update],
			"SubmissionNotes"->If[StringQ[$localSubmissionNotes],
				 $localSubmissionNotes,Missing[]]
	];
	$submitProgressContent="Submitting resource \[Ellipsis]";
	res=apifun["SubmitResource",params, System`ResourceSubmit,resourcebase];
	If[Quiet[KeyExistsQ[res,"SubmissionID"]],
		createSubmissionObject[res,as]
		,
		$Failed
	]
]

checkPublisherID[id_]:=checkPublisherID[Automatic,id]
checkPublisherID[_,id_String]:=id
checkPublisherID[rtype_,Automatic|None|HoldPattern[System`$PublisherID]]:=If[StringQ[System`$PublisherID],System`$PublisherID,checkPublisherID[rtype,None]]

checkPublisherID[rtype_,Automatic|None]:=If[allowProvisionalSubmissionQ[rtype],
	None
	,
	With[{default=getDefaultPublisherID[]},
		If[StringQ[default],
			default,
			checkPublisherID[rtype,$Failed]
		]
	]
]

checkPublisherID[rtype_,_]:=(Message[ResourceSubmit::nopubidl,
	Hyperlink["https://datarepository.wolframcloud.com/publisheridrequest/"]];
	Throw[$Failed])

allowProvisionalSubmissionQ[_]:=True

checkSubmissionNotes[str_String]:=str/;StringLength[str]<10^6
checkSubmissionNotes[_]:=$submissionNotes

createSubmissionObject[res_Association,as_]:=repositoryCreateSubmissionObject[Lookup[res,"ResourceType"],res, as]

repositoryCreateSubmissionObject[_,res_Association,_]:=Success["ResourceSubmission", Association[
	"MessageTemplate" -> ResourceSubmit::subsuc,
	"MessageParameters" -> KeyTake[res,"SubmissionID"],
  	res]
]

standardizeSubmissionUpdate[ro_ResourceObject, resourcebase_]:=With[{uuid=ro["UUID"]},
	If[uuidQ[uuid],
		standardizesubmissionUpdate[uuid,resourcebase]
		,
		Message[ResourceSubmit::invupd,ro];Throw[$Failed]	
	]	
]

standardizeSubmissionUpdate[None, _]:=None
standardizeSubmissionUpdate[n_Integer, resourcebase_]:=standardizeSubmissionUpdate[IntegerString[n, 10, 4],resourcebase]

standardizeSubmissionUpdate[submissionid_String, resourcebase_]:=submissionid/;submissionidQ[submissionid]
standardizeSubmissionUpdate[uuid_String, resourcebase_]:=standardizesubmissionUpdate[uuid,resourcebase]/;uuidQ[uuid]
standardizeSubmissionUpdate[name_String, resourcebase_]:=With[{ro=ResourceObject[name]},
	If[resourceObjectQ[ro],
		standardizeSubmissionUpdate[ro,resourcebase]
	]
]

standardizesubmissionUpdate[id_String,resourcebase_]:=standardizesubmissionUpdate[id,getResourceInfo[id],resourcebase]

standardizesubmissionUpdate[id_,info_Association,resourcebase_]:=standardizesubmissionupdate[id, info,resourcebase]/;marketplacebasedResourceQ[info]

standardizesubmissionUpdate[expr_,___]:=(Message[ResourceSubmit::invupd,expr];Throw[$Failed])

standardizesubmissionupdate[id_, info_,resourcebase_]:=id/;First[info["RepositoryLocation"]]===resourcebase

standardizesubmissionupdate[id_,_,_]:=(Message[ResourceSubmit::invupdbase,id];Throw[$Failed])

submissionidQ[str_String]:=StringMatchQ[str, sid : (DigitCharacter ..) /; StringLength[sid] == 4]
submissionidQ[___]:=False

setSubmissionCloudObjectMetadata[cloud_,eleminfo_]:=setsubmissionCloudObjectMetadata[cloud[#],eleminfo[#]]&/@Intersection[Keys[cloud],Keys[eleminfo]]
setsubmissionCloudObjectMetadata[co:HoldPattern[_CloudObject],eleminfo_Association]:=SetOptions[co,MetaInformation->Normal[eleminfo]]

getDefaultPublisherID[]:=With[
	{res=apifun["DefaultPublisher", {}, ResourceSubmit]},
	If[KeyExistsQ[res,"DefaultPublisherID"],
		res["DefaultPublisherID"],
		$Failed
	]
]

End[] (* End Private Context *)

EndPackage[]

SetAttributes[{ResourceSubmit,"PublisherID"},
   {ReadProtected, Protected}
];
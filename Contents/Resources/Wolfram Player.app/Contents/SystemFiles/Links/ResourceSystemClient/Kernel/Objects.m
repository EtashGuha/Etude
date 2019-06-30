(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {System`ResourceObject,System`ResourceSubmissionObject}

BeginPackage["ResourceSystemClient`"]

System`ResourceObject
System`ResourceSubmissionObject (* Deprecated *)

Begin["`Private`"] (* Begin Private Context *) 

(* ResourceObject *)
Options[System`ResourceObject]={System`ResourceSystemBase:>System`$ResourceSystemBase,"Version"->Automatic,"WolframLanguageVersion"->Automatic}

System`ResourceObject[str_?uuidQ,opts:OptionsPattern[]]:=System`ResourceObject[usableResourceInfo[resourceInfo[str]],opts]/;MemberQ[$loadedResources, str]

System`ResourceObject[str_?uuidQ,opts:OptionsPattern[]]:=Block[{$importCloudResources=True},With[{res=Catch[loadResource[str]]},
	If[AssociationQ[res],
		System`ResourceObject[usableResourceInfo[res],opts]
		,
		$Failed
	]
]
]

System`ResourceObject[nbo_NotebookObject,OptionsPattern[]]:=Catch[createResourceFromNotebook[nbo]]
System`ResourceObject[nb_Notebook,OptionsPattern[]]:=Catch[createResourceFromNotebook[nb]]

$ResourceNameLookup=None;

System`ResourceObject[str_String,opts:OptionsPattern[]]:=
	System`ResourceObject[str,System`ResourceSystemBase->OptionValue[System`ResourceSystemBase],opts]/;FreeQ[{opts},System`ResourceSystemBase]

System`ResourceObject[name_String,opts:OptionsPattern[]]:=Block[{$ResourceNameLookup=name},
	Catch[findResourceObject[All, name,opts]]
]
System`ResourceObject[url_URL,opts:OptionsPattern[]]:=Catch[findResourceObjectByURL[First[url]]]
System`ResourceObject[co:HoldPattern[_CloudObject],opts:OptionsPattern[]]:=Catch[findResourceObjectByURL[co]]
System`ResourceObject[lo:HoldPattern[_LocalObject],opts:OptionsPattern[]]:=System`ResourceObject[ Import @ lo, opts ];
System`ResourceObject[Rule[type_, name_],opts:OptionsPattern[]]:=Block[{$ResourceNameLookup=name},
	Catch[findResourceObject[type, name,opts]]
]

(resource_System`ResourceObject)[str_String, rest___]:=ResourceSystemClient`ResourceInformation[resource, Association["Property"->str,"Parameters"->{rest}]]
(resource_System`ResourceObject)[All]:=With[{info=sortBasicInfo[getResourceInfo[resourceObjectID[resource]]]},
	If[AssociationQ[info],info,$Failed]]

System`ResourceObject[info_Association,opts:OptionsPattern[]]:=Catch[System`ResourceObject[usableResourceInfo[standardizecustomResourceInfo[info]]]]/;!filledResourceQ[info]
System`ResourceObject[info_Association,opts:OptionsPattern[]]:=Catch[autoloadResource[info]]/;autoloadResourceQ[info]

System`ResourceObject[info_Association,opts:OptionsPattern[]]:=If[TrueQ[$reacquiring],
	Message[ResourceObject::optunav,info["Name"],
		Association[
			"WolframLanguageVersion"->Lookup[ResourceSystemClient`Private`$ClientInformation,"WLVersion",$VersionNumber],opts]];$Failed,
	Block[{$reacquiring=True},
		System`ResourceAcquire[
		If[StringQ[$ResourceNameLookup],
			$ResourceNameLookup,info["UUID"]],opts]
	]
]/;optionMismatchQ[info, Association[opts]]

System`ResourceObject[ro:HoldPattern[System`ResourceObject][args___],OptionsPattern[]]:=ro
System`ResourceObject[expr:Except[_Association],OptionsPattern[]]:=(Message[ResourceObject::noas,expr];$Failed)

System`ResourceObject/:
MakeBoxes[resource:System`ResourceObject[_Association,___], form:StandardForm|TraditionalForm] := (
Catch[standardResourceObjectBoxes[resource, form]])

standardResourceObjectBoxes[resource_, form_]:=With[{id=Quiet[resourceObjectID[resource]]},
	If[StringQ[id],
        	With[{info=If[AssociationQ[#],#,First[resource]]&@resourceInfo[id]},
        		With[{rtype=getResourceType[info, None]},
        			loadResourceType[rtype];
		            BoxForm`ArrangeSummaryBox[
		                        (* Head *)System`ResourceObject, 
		                        (* Interpretation *)resource, 
		                        (* Icon *)resourceIcon[rtype], 
		                        (* Column or Grid *)
		                        {
		                        {BoxForm`SummaryItem[{"Name: ", summaryResourceName[info]}], 
		                        	summaryPurchaseBox[id]},
		                        	
		                        
		                        summaryTypeRow[rtype,info],
		                        {resourceSystemDescriptionSummaryItem[rtype,Lookup[info,"Description",Missing["NotAvailable"]]],SpanFromLeft}
		                        }
		                        ,
		                        (* Plus Box Column or Grid *)
		                        repositoryBelowFoldItems[rtype,id, info]
		                        ,
		            form]
        		]
        	]
        	,
        	ToBoxes[$Failed]
	]
]

summaryTypeRow[rtype_,info_Association]:=summaryTypeRowWithCost[rtype,info["PricingInformation"]]/;costPerUseResourceQ[info]
summaryTypeRowWithCost[rtype_,pi_Association]:=summaryTypeRowWithCost[rtype,pi["BaseUsagePrice"]]/;Quiet[pi["MarketplaceBilling"]==="PerUse"]
summaryTypeRowWithCost[rtype_,n_?NumberQ]:={Grid[{{BoxForm`SummaryItem[{"Type: ", rtype}],formatSummaryCost[rtype,n]}},
	Dividers -> {{False, True, False}, False}],SpanFromLeft}
summaryTypeRowWithCost[rtype_,_]:={BoxForm`SummaryItem[{"Type: ", rtype}], SpanFromLeft}
formatSummaryCost[_,n_]:=BoxForm`SummaryItem[{"Cost: ", ToString[n]<>" service credit/call"}]

summaryTypeRow[rtype_,_]:={BoxForm`SummaryItem[{"Type: ", rtype}],SpanFromLeft}

summaryPurchaseBox[id_]:=""/;summaryPurchaseStatus[id]===""
summaryPurchaseBox[id_]:=With[{stat=summaryPurchaseStatus[id]},
	If[Head[stat]===staticContent,
		stat[[1]],
		Dynamic[Replace[summaryPurchaseStatus[id],Except[_Style|_Hyperlink|_Button]->"",{0}]]
	]
]
summaryPurchaseStatus[_]:=""


summaryMustPurchase[info_Association]:=With[{url=Lookup[info,"AcquisitionURL",purchasinglocation[info]],
	str=FromCharacterCode[{85, 78, 65, 67, 81, 85, 73, 82, 69, 68, 32, 187}]},
	If[StringQ[url],
		Button[StatusArea[Style[str,10,Red], url],
			(SystemOpen[url];
			createResourceChannel[info["UUID"]]),
			Appearance -> None
		]
		,
		$summaryMustPurchase
	]
]
summaryMustPurchase[_]:=$summaryMustPurchase

$summaryMustPurchase:=($summaryMustPurchase=Style["UNACQUIRED",10,Red])
$summaryPurchased:=($summaryPurchased=Style["Acquired",10,Darker[Green]])

$ServiceCreditInformationURL="https://www.wolfram.com/service-credits/";

repositoryBelowFoldItems[_,id_, info_]:={
	summaryKeywords[Lookup[info,"Keywords",{}]],
	summaryResourceLink[info],
	BoxForm`SummaryItem[{"UUID: ", id}],
	BoxForm`SummaryItem[{"Version: ", Lookup[info,"Version",None]}]
	}
	
summaryKeywords[keywords_List]:=BoxForm`SummaryItem[{"Keywords: ", Short[Row[keywords,","]]}]/;Length[keywords]>0
summaryKeywords[_]:=Nothing

resourceSystemDescriptionSummaryItem[_,str_String]:=BoxForm`SummaryItem[{"Description: ", 
		str
 	}]/;Snippet[str,1]===str

resourceSystemDescriptionSummaryItem[_,str_String]:=DynamicModule[{len=1},
	BoxForm`SummaryItem[{"Description: ", 
		Button[Dynamic[
			Replace[snipDots[str, len],Except[_String]->Snippet[str, len],{0}]
			], len = Ceiling[len*1.5], Appearance -> None,BaseStyle -> {}]
 	}]
]

snipDots[str_, len_] := With[{snip = Snippet[str, len]},
	If[StringLength[StringTrim[str]] <= StringLength[snip],
		str,
		snip <>ToString["..."]
		]
	]
	
resourceSystemDescriptionSummaryItem[_,expr_]:=Nothing

fallbackResourceIcon=Null;

summaryResourceName[info_]:=summaryResourceName[Lookup[info,"Name",Missing["NotAvailable"]],resourceURL[info],costPerUseResourceQ[info]]

summaryResourceName[name_,url_URL,True]:=Hyperlink[Style[name<>" \[RightGuillemet]",Darker[Red]],url]
summaryResourceName[name_,url_URL,_]:=Row[{name," ",Hyperlink["\[RightGuillemet]",url]}]
summaryResourceName[name_,__]:=name

summaryResourceLink[info_Association]:=summaryResourceLink[resourceURL[info]]
summaryResourceLink[url_URL]:=BoxForm`SummaryItem[{"Documentation: ", Hyperlink[url]}]
summaryResourceLink[_]:=Nothing

resourceicon[file_]:=With[{img=Import[file]},
	Switch[Head[img],
		Graphics,
		formatresourceicon[img],
		List,
		formatresourceicon[img[[1]]],
		_,
		fallbackResourceIcon
	]		
]/;FileExistsQ[file]

resourceIcon[_]=fallbackResourceIcon;

formatresourceicon[gr_Graphics]:=Graphics[gr[[1]],
  AspectRatio -> 1, Axes -> False, Background -> None, Frame -> None, 
 FrameTicks -> None, 
  ImageSize -> Dynamic[{Automatic, 	
  	3.5*(CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification])},
  	ImageSizeCache -> {45., {0., 9.}}]]

filledResourceQ[info_Association]:=KeyExistsQ[info,"UUID"]
filledResourceQ[_]:=False

autoloadResourceQ[info_Association]:=TrueQ[Lookup[info,"Autoload"]]
autoloadResourceQ[_]:=False

System`ResourceSubmissionObject[suc:HoldPattern[_Success]]:=System`ResourceSubmissionObject[suc["SubmissionID"]]
System`ResourceSubmissionObject[rso_System`ResourceSubmissionObject]:=rso

System`ResourceSubmissionObject[id:(_String|_Integer)]:=Catch[importSubmission[id]]

System`ResourceSubmissionObject[info_]:=$Failed/;!Quiet[KeyExistsQ[info, "Name"]]

sub_System`ResourceSubmissionObject[req_,args___]:=Catch[submissionRequest[First[sub],req,{args}]]

System`ResourceSubmissionObject/:
MakeBoxes[resource_System`ResourceSubmissionObject, form:StandardForm|TraditionalForm] := (
Catch[With[{info=First[resource]},
            With[{id=Lookup[info,"UUID"],
            	rtype=getResourceType[info],
            	name=Lookup[info,"Name"]},
                BoxForm`ArrangeSummaryBox[
                            (* Head *)System`ResourceSubmissionObject, 
                            (* Interpretation *)resource, 
                            (* Icon *)formatSubmissionIcon[resourceIcon[rtype]], 
                            (* Column or Grid *)
                            {
                            BoxForm`SummaryItem[{"Name: ", name}],
                            BoxForm`SummaryItem[{"Type: ", rtype}],
                            BoxForm`SummaryItem[{"SubmissionID: ", Lookup[info,"SubmissionID",id]}]
                            }
                            ,
                            (* Plus Box Column or Grid *)
                            {
                            BoxForm`SummaryItem[{"UUID: ", id}],
                            BoxForm`SummaryItem[{"SubmissionDate: ", Lookup[info,"SubmissionDate"]}]
                            }, 
                form]
            ]
        ]])

formatSubmissionIcon[icon_Graphics]:=Graphics[Replace[icon[[1]], RGBColor[___] :> RGBColor[.4, .4, .4],  Infinity], 
	AspectRatio -> 1, Axes -> False, Background -> GrayLevel[0.9], Frame -> True,FrameStyle -> GrayLevel[0.6], FrameTicks -> None, 
 	ImageSize -> {Automatic, Dynamic[3.5*(CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]), ImageSizeCache -> {45., {0., 9.}}]}]

typesetSize[info_]:=bytecountQuantity[Lookup[info,"ContentSize",Missing["NotAvailable"]]]


End[] (* End Private Context *)

EndPackage[]

SetAttributes[{ResourceObject,ResourceSubmissionObject},
   {ReadProtected, Protected}
];
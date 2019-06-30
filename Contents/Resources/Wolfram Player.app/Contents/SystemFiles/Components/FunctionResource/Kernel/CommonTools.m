(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["FunctionResource`"]

Begin["`Private`"] (* Begin Private Context *) 

resourceSystemToolSet[str_]:=With[
	{symb=ToExpression[str], rscsymb=ToExpression["ResourceSystemClient`Private`"<>str]},
	symb=rscsymb]


SetAttributes[resourceSystemToolSetDelay,HoldAll]

resourceSystemToolSetDelay[str_String] := (
	Clear[str]; 
	With[{symb = ToExpression[str]}, resourceSystemToolSetDelay[symb]])
resourceSystemToolSetDelay[symb_Symbol] := 
	With[{name = "ResourceSystemClient`Private`" <> ToString[symb]}, 
  		symb := Symbol[name]]

resourceSystemToolSet/@{
	"resourceDirectory",
	"resourceInfoFile",
	"resourceInfo",
	"resourceObjectID",
	"fileExistsQ",
	"createDirectory",
	"bytecountQuantity",
	"fileByteCount",
	"cacheresourceinfo",
	"cloudpath",
	"userdefinedResourceQ",
	"marketplacebasedResourceQ",
	"importlocal",
	"uuidQ",
	"resourcesearchLocalName",
	"resourceObjectQ",
	"validateParameter",
	"resourceCacheDirectory",
	"setReviewerPermissions",
	"mapAt",
	"keyExistsQ",
	"stringFileExistsQ",
	"exportRawResourceContent",
	"cloudresourceDownload",
	"cacheResourceQ",
	"resourceDataPostProcessFunction",
	"getElementInfo",
	"resourceCopyDirectory",
	"resourceCopyInfoFile",
	"getElementCopyInfo",
	"resourceElementDirectory",
	"getElementFunction",
	"getAllElementFunction",
	"storeElementFunction",
	"multipartResourceQ",
	"cloudStoredQ",
	"localStoredQ",
	"contentElementQ",
	"storeContentFunctions",
	"storecontentFunctions",
	"resourcefiledownload",
	"resourcecopyfiledownload",
	"standardizeContentMetadataContentInfo",
	"saveresourceobjectwithelements",
	"bundleResourceObjectWithElementFunctions",
	"clouddeployResourceInfoWithElements",
	"cloudexportResourceContentElement",
	"cloudDeployResourceContentElements",
	"typesetElementStorageLocation",
	"produceResourceElementContent",
	"readResourceElementContent",
	"updateResourceInfoElements",
	"cacheResourceInfoWithElementFunctions",
	"completeResourceSubmissionWithElements",
	"standardizeContentMetadataWithElements",
	"contentElementSize"
}

resourceSystemToolSetDelay/@{
	"localObject",
	"$localResources",
	"$cloudResources",
	"$loadedResources",
	"localResourceNameMap"
	
}



dataResourceToolSet[str_]:=With[
	{symb=ToExpression[str], rscsymb=ToExpression["DataResource`Private`"<>str]},
	symb=rscsymb]
	
dataResourceToolSet/@{
	"resourcedataUncached",
	"resourcedataLocal",
	"resourcedataResourceSystem",
	"resourcedataCloud"
	
}	
	

End[] (* End Private Context *)

EndPackage[]



SetAttributes[{},
   {ReadProtected, Protected}
];
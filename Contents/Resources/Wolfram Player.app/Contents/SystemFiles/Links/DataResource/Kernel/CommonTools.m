(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["DataResource`"]

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
	"uRL",
	"fetchContent",
	"mapAt",
	"attributeCheck",
	"stringFileExistsQ",
	"keyExistsQ",
	"exportRawResourceContent",
	"cloudresourceDownload",
	"typesetSize",
	"cacheResourceQ",
	"resourceDataPostProcessFunction",
	"localResourceQ",
	"getElementInfo",
	"resourceCopyDirectory",
	"resourceCopyInfoFile",
	"resourcecopyInfoFile",
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
	"uncompressInformationElements",
	"contentElementSize",
	"printTempOnce",
	"clearTempPrint",
	"contentCached",
	"setElementCached"
}

resourceSystemToolSetDelay/@{
	"localObject",
	"$localResources",
	"$cloudResources",
	"$loadedResources",
	"localResourceNameMap",
	"$CacheResourceContent"
}

End[] (* End Private Context *)

EndPackage[]



SetAttributes[{},
   {ReadProtected, Protected}
];
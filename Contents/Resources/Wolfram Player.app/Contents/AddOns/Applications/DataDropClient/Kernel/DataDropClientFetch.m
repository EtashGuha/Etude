(* Mathematica Package *)

BeginPackage["DataDropClient`"]

Begin["`Private`"] (* Begin Private Context *) 

(*** API Function ***)
apifun[Except[_String],_]:=(Message[Databin::invreq];$Failed)


apifun["Add", as_] := Block[{request,callback},
	{request,callback}=databinAddRequestAndCallback[as];
	callback[
        If[TrueQ[System`$CloudEvaluation && javaCompatibleQ[as]],
           ddURLFetchInCloud @@ request,
           If[ddCloudConnectedQ[], 
               CloudObject`Private`authenticatedURLFetch, 
               URLFetch][#1, {"StatusCode", "ContentData"}, ##2
           ] & @@ request
       ]
   ]
]

asynchapifun["Add",as_,outercallback_]:=Block[{request,innercallback},
	{request,innercallback}=databinAddRequestAndCallback[as];
	With[{fun=Composition[outercallback,innercallback]},
		If[ddCloudConnectedQ[],CloudObject`Private`authenticatedURLFetchAsynchronous,URLFetchAsynchronous][#1,
			(callbackfun[fun, {##}]&),
			##2]&@@request
	]
]

apifun["Upload", as_] := Block[{raw, mpdata, strkeys, imgkeys},
    {strkeys, imgkeys, mpdata} = makeMPData[Normal[as]];
    raw = CloudObject`Private`authenticatedURLFetch[gatewayapi,{"StatusCode", "ContentData"},
        "MultipartData" -> {
            {"API", "text/plain", {85, 112, 108, 111, 97, 100}},
            {"InputStrings", "text/plain", ToCharacterCode[strkeys,"UTF8"]},
            {"CompressedWDFImages", "text/plain", ToCharacterCode[imgkeys,"UTF8"]},
            {"SourceType", "text/plain", {67, 111, 110, 110, 101, 99, 116, 101, 100,
                32, 87, 111, 108, 102, 114, 97, 109, 32, 76, 97, 110, 103, 117, 97, 103, 101}},
            {"ClientVersion", "text/plain",ToCharacterCode[$datadropclientversion,"UTF8"]},
            Sequence @@ mpdata
        },  "Method" -> "POST", "VerifyPeer" -> False, "CredentialsProvider" -> None]; 
    importResults["Add"][checkAvailable[raw, "Add"]]
    ]/;ddCloudConnectedQ[]&&KeyExistsQ[as,"DataDropReferences"]
    
apifun["Upload",as_]:=
With[{
    raw=CloudObject`Private`authenticatedURLFetch[gatewayapi,{"StatusCode","ContentData"},
        "Parameters"->Join[{"API"->"Upload","SourceType"->"\"WolframLanguage\"","InputStrings"->"True",
            "ClientVersion"->ToString[$datadropclientversion,InputForm]},
            preparedata[Normal[as]]],"Method"->"POST", 
            "VerifyPeer" -> False,"CredentialsProvider" -> None]},
    importResults["Upload"][checkAvailable[raw,"Add"]]
]/;ddCloudConnectedQ[]

apifun["GetUserBinsInfo",as_]:=
With[{
    raw=CloudObject`Private`authenticatedURLFetch[gatewayapi,{"StatusCode","ContentData"},
        "Parameters"->Join[{"API"->"GetUserBinsInfo","SourceType"->"\"WolframLanguage\"","InputStrings"->"True",
            "ClientVersion"->ToString[$datadropclientversion,InputForm]},
            preparedata[Normal[as]]], "Method"->"POST",
            "VerifyPeer" -> False,"CredentialsProvider" -> None]},
    importResults["GetUserBinsInfo"][checkAvailable[Quiet[ToExpression[raw]],"GetUserBinsInfo"]]
]/;ddCloudConnectedQ[]

apifun[name_,as_]:=
With[{
	raw=CloudObject`Private`authenticatedURLFetch[gatewayapi,{"StatusCode","ContentData"},
		"Parameters"->Join[{"API"->name,"SourceType"->"\"WolframLanguage\"","InputStrings"->"True",
            "ClientVersion"->ToString[$datadropclientversion,InputForm]},
			preparedata[Normal[as]]], 
			"VerifyPeer" -> False,"CredentialsProvider" -> None]},
	importResults[name][checkAvailable[raw,name]]
]/;ddCloudConnectedQ[]

apifun[name_,as_]:=
With[{
	raw=URLFetch[gatewayapi,{"StatusCode","ContentData"},
		"Parameters"->Join[{"API"->name,"SourceType"->"\"Unconnected Wolfram Language\"","InputStrings"->"True",
			"ClientVersion"->ToString[$datadropclientversion,InputForm]},
			preparedata[Normal[as]]],"VerifyPeer" -> False,"CredentialsProvider" -> None]},
	importResults[name][checkAvailable[raw,name]]
]/;MemberQ[Join[$nonauthenticatedrequests,$nonauthenticatedapinames],name]


apifun[name_,as_]:=(
	ddCloudConnect[];
	If[ddCloudConnectedQ[],
		apifun[name,as]
		,
		Message[Databin::cloudc];
		Throw[$Failed]
	]
)

ddURLFetchInCloud[url_, param_, other___] := Block[
   {method = Lookup[List[other], "Method"],
    paths = StringSplit[url, "/"],
    parameters = Lookup[List[param], "Parameters"],
    statuscode, message, callpacketresult},
    callpacketresult = CloudObject`Private`execute[$CloudBase, method, {paths[[-3]], paths[[-2]], paths[[-1]]}, 
                       CloudObject`Private`Parameters -> parameters];
    callpacketresult = Replace[
    	callpacketresult, 
        {
           result_List :> { 200, result[[2]] },
           err : CloudObject`Private`HTTPError[status_Integer?Positive, 
           content_, type_] :> { status, ToCharacterCode[content] }
        }
    ];
    
    callpacketresult
]

$DataDropHTTPCodes=_;
checkAvailable[{501,_},_]:=(Message[Databin::notav];Throw[$Failed])
checkAvailable[{504,_},"Read"|"Dashboard"]:=(Message[Databin::timeout1];Throw[$Failed])
checkAvailable[{504,_},_]:=(Message[Databin::timeout2];Throw[$Failed])
checkAvailable[$Failed,_]:=$Failed
checkAvailable[{code_,bytes_},_]:={code,Quiet[importBytes[bytes]]}
checkAvailable[___]:=$Failed

importBytes[bytes_]:=FromCharacterCode[bytes,"UTF8"]

readrequestpattern=("Read"|"Recent"|"Entries"|"Latest"|"Values");

importResults[readrequestpattern]:=(With[{data=datadropMXRead[Quiet[ToExpression[#[[2]]]]]},
     If[$ImportDataDropReferences,
         importDataDropReferences[checkWarnings[data]],
         checkWarnings[data]
     ]
]&)


importResults[name_]:=(
	If[MatchQ[#[[1]],$DataDropHTTPCodes],
		importresults[name][#]
		,
		errorcheck[importResults[name][{200, #[[2]]}], name]
	]&)

(*
HoldPattern[importResults][name_][{code_,raw_}]:=errorcheck[importResults[name][{200, 
raw}], name]/;!MatchQ[code,$DataDropHTTPCodes]
*)

importresults["JSON"]:=(ImportString[#[[2]],"RawJSON"]&)
importresults["Raw"]:=Last
importresults[name_]:=(Quiet[ToExpression[#[[2]]]]&)

makeMPData[rules_] := Block[{toexpkeys = {}, mpdata},
  mpdata = makempdata /@ rules;
  toexpkeys = Flatten[First /@ mpdata];
  {StringJoin[Riffle[toexpkeys, ","]], 
   StringJoin[Riffle[Complement[First /@ rules, toexpkeys], ","]], 
   Last /@ mpdata}
  ]
makempdata[_[key_, img_Image]] := {{}, {key, "image/png", 
   ToCharacterCode[ExportString[img, "PNG"]]}}
makempdata[_[key_, value_]] := {{key}, {key, "text/plain", 
   ToCharacterCode[ToString[value, InputForm]]}}


$JavaAddSizeLimit=25000; 
(* No images *)
javaCompatibleQ[as_]:=False/;!FreeQ[as,_Image] 

(* Put a size limit on the request *)
javaCompatibleQ[as_]:=False/;ByteCount[as]>$JavaAddSizeLimit

DataDropClient`$JavaCompatibleQ = True;
javaCompatibleQ[_Association]:=True/;DataDropClient`$JavaCompatibleQ
javaCompatibleQ[_]:=False


databinAddRequestAndCallback[as_]:=Block[{raw, mpdata, strkeys, imgkeys},
	{strkeys, imgkeys, mpdata} = makeMPData[Normal[as]];
  	{
  		{gatewayapi,"MultipartData" -> {
      		{"API", "text/plain", {65, 100, 100}},
      		{"InputStrings", "text/plain", ToCharacterCode[strkeys,"UTF8"]},
      		{"CompressedWDFImages", "text/plain", ToCharacterCode[imgkeys,"UTF8"]},
      		{"SourceType", "text/plain", {67, 111, 110, 110, 101, 99, 116, 101, 100,
      			32, 87, 111, 108, 102, 114, 97, 109, 32, 76, 97, 110, 103, 117, 97, 103, 101}},
            {"ClientVersion", "text/plain",ToCharacterCode[$datadropclientversion,"UTF8"]},
      		Sequence @@ mpdata
      	},  "Method" -> "POST", "VerifyPeer" -> False, "CredentialsProvider" -> None}
      	,
      	(importResults["Add"][checkAvailable[#,"Add"]]&)
  	}
    ]/;ddCloudConnectedQ[]&&!FreeQ[as,_Image]&&ByteCount[as]>$UncompressedImageLimit

databinAddRequestAndCallback[as_]:=(
	ddCloudConnect[];
	If[ddCloudConnectedQ[],
		databinAddRequestAndCallback[as]
		,
		Message[Databin::cloudc];
		Throw[$Failed]
	]
)/;!ddCloudConnectedQ[]&&!FreeQ[as,_Image]&&ByteCount[as]>$UncompressedImageLimit


databinAddRequestAndCallback[as_]:={
	{
		URLBuild[{$CloudBase, "databins", Lookup[as, "Bin"], "entries"}],
		"Parameters"->Join[{"SourceType"->"WolframLanguage","ClientVersion"->$datadropclientversion},
			preparedata[Normal[as]]],"Method"->"POST", 
			"VerifyPeer" -> False,"CredentialsProvider" -> None}
		,
		(With[{response=importResults["JSON"][checkAvailable[#,"Add"]]},
			If[AssociationQ[response] && First[#] === 200,
				Append[response,"Data"->Automatic],
				response
			]
		]&)
}/;javaCompatibleQ[as]


databinAddRequestAndCallback[as_]:={
	{gatewayapi,
		"Parameters"->Join[{"API"->"Add","SourceType"->"\"WolframLanguage\"","InputStrings"->"True",
			"ClientVersion"->ToString[$datadropclientversion,InputForm]},
			preparedata[Normal[as]]],"Method"->"POST", 
			"VerifyPeer" -> False,"CredentialsProvider" -> None}
		,
      	(importResults["Add"][checkAvailable[#,"Add"]]&)
}

callbackfun[fun_,{_,"data",{bytes_}}]:=(fun[{200,bytes}])
callbackfun[_,code:{_,"statuscode",Except[{200}|200]}]:=Message[Databin::asyncf]


CloudObject`Private`makeOAuthHeader["https://datadrop.wolframcloud.com/api/v1.0/Gateway", method_String, ctype_, body_] := 
  CloudObject`Private`makeOAuthHeader["https://datadrop.wolframcloud.com/objects/user-fa95220f-871c-4331-84ab-7951dd0666ca/Gateway", method, ctype, body] 
CloudObject`Private`makeOAuthHeader[ "https://datadrop.wolframcloud.com/api/v1.0/Gateway", method_String,  ctype_, body_, params_] := 
  CloudObject`Private`makeOAuthHeader[ "https://datadrop.wolframcloud.com/objects/user-fa95220f-871c-4331-84ab-7951dd0666ca/Gateway", method, ctype, body, params] 


End[] (* End Private Context *)

EndPackage[]
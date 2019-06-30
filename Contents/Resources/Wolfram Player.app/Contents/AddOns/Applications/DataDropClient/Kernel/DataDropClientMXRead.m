(* Mathematica Package *)


BeginPackage["DataDropClient`"]

Begin["`Private`"] (* Begin Private Context *) 

datadropMXRead[res_Association]:=Block[{format},
	format=Lookup[res,"ResponseFormat","Text"];
	getResponse[format, res]
]

datadropMXRead[res_]:=res

getResponse["MXFile", res_]:=mxRead[Lookup[res,"FileURL"]]

mxRead[url_String]:=With[{res=mxread[url]},
	If[res===$Failed,
		Throw[$Failed],
		cleanupMXFile[url];
		res
	]	
]


mxRead[_]:=$Failed

mxread[url_]:=With[{raw=URLFetch[CloudObject[url], {"StatusCode","ContentData"},"VerifyPeer" -> False,"CredentialsProvider" -> None]},
	If[raw[[1]]===200,
		ImportString[FromCharacterCode[raw[[2]]], "MX"]
		,
		Throw[$Failed]
	]
]

getResponse["Text", res_]:=res
getResponse[_, _]:=Throw[$Failed]


cleanupMXFile[url_]:=CloudObject`Private`authenticatedURLFetchAsynchronous[gatewayapi,
        "Parameters"->{"API"->"DeleteMXResponse","SourceType"->"WolframLanguage",
            "ClientVersion"->$datadropclientversion,"FileURL"->url}, 
            "VerifyPeer" -> False,"CredentialsProvider" -> None]

End[] (* End Private Context *)

EndPackage[]

(* Mathematica Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["DataDropClient`"]

Begin["`Private`"] (* Begin Private Context *) 

DataDropClient`$DataDropBase:=datadropbase[$CloudBase]
DataDropClient`$DataDropAPIBase:=datadropbaseapi[DataDropClient`$DataDropBase]

datadropbase["https://www.wolframcloud.com/"|"http://www.wolframcloud.com/"]="https://datadrop.wolframcloud.com/";
datadropbase[url_]:=StringReplace[url,"/www"->"/datadrop"]/;!StringFreeQ[url,"wolframcloud.com"]
datadropbase[_]:=If[checkDataDropBase[$CloudBase],
	$CloudBase,
	"https://datadrop.wolframcloud.com/"	
]

datadropbaseapi["https://datadrop.wolframcloud.com/"]:="https://datadrop.wolframcloud.com/api/v1.0/Gateway"
datadropbaseapi[ddbase_]:=URLBuild[{ddbase,"objects","datadrop-admin","Gateway"}]

binurlbase[]:=binurlbase[DataDropClient`$DataDropBase]
binurlbase[ddbase_]:=URLBuild[{ddbase,"databin",""}]

shorturlbase[]:=shorturlbase[DataDropClient`$DataDropBase]
shorturlbase[ddbase_]:="http://wolfr.am/"/;!StringFreeQ[ddbase,"wolframcloud.com"]
shorturlbase[ddbase_]:="http://wolfr.am/"

checkDataDropBase[base_]:=With[{code=Quiet[URLFetch[datadropbaseapi[base],"StatusCode","Parameters"->{"API"->"Test"},"VerifyPeer"->False]]},
	If[code===200,
		True,
		False
	]
]

shorturldomain[args___]:=With[{sub=shorturlbase[args]},
	If[StringQ[sub],
		URLParse[sub, "Domain"]
		,
		$Failed
	]
]

datadropbinform[]:=datadropbinform[DataDropClient`$DataDropBase]
datadropbinform[ddbase_]:="*://"<>Last[StringSplit[URLParse[binurlbase[], "AbsolutePath"], "://"]]~~("dd"|"DD")~~((WordCharacter | "-")..)~~("/"...);

gatewayapi:=datadropbaseapi[DataDropClient`$DataDropBase]

ddCloudConnectedQ[]:=($CloudConnected&&ddbaseMatchQ[$CloudBase,DataDropClient`$DataDropBase])

ddCloudConnect[]:=CloudConnect[]/;ddbaseMatchQ[$CloudBase,DataDropClient`$DataDropBase]

ddCloudConnect[]:=(Message[Databin::ddcb1];$Failed)

ddbaseMatchQ[cbase_,ddbase_]:=(ddbaseMatchQ[cbase,ddbase]=(DataDropClient`$DataDropBase===datadropbase[$CloudBase]))
End[] (* End Private Context *)

EndPackage[]

SetAttributes[{},
   {ReadProtected, Protected}
];

(* Mathematica Package *)

BeginPackage["IntegratedServices`"]

Begin["`Private`"]

(* Verify service API *)
$VerificationEndpoint = "Verify";

serviceEndpoint[service_]:= Block[{res, api},
	res = apirequest[$VerificationEndpoint, Association["ServiceName" -> service]];
	If[AssociationQ[res] && TrueQ[res["Connected"]],
		api = Lookup[res,"API"];
		If[StringQ[api] && StringContainsQ["wolframcloud"][api],
			api = CloudObject[api];
			serviceEndpoint[service] = api
			,
			$Failed
		]
		,
		$Failed
	]
]

(* Client Information *)

$clientinfo:=($clientinfo=Compress[
	Association[
		"WLVersion"-> $Version,
		"ISPacletVersion"-> Lookup[PacletManager`PacletInformation["IntegratedServices"], "Version"],
		"SystemID"-> $SystemID
	]
])

(* api request *)

apirequest[endpoint: _CloudObject | $VerificationEndpoint, params_]:= With[
	{resp = URLRead[HTTPRequest[buildCloudapi[endpoint], <|"Query"-> prepareparams[endpoint,params], Method->"POST"|>], {"StatusCode", "Headers", "BodyBytes"},VerifySecurityCertificates -> $CloudPRDBaseQ]},
	If[AssociationQ[resp],
		handleResponse[endpoint, resp]
		,
		Throw[$Failed, "apierror"]
	]
]


buildCloudapi[endpoint_CloudObject] = endpoint
buildCloudapi[$VerificationEndpoint] = CloudObject[URLBuild[{$IntegratedServicesAPIBase, $VerificationEndpoint}]]

prepareparams[$VerificationEndpoint, params_] = params
prepareparams[_, params_]:= <|
	"RequestData"	-> Compress[params],
	"ClientInfo"	-> $clientinfo,
	"ClientCredits"	-> Compress[<|"Quantity" -> serviceCredits[$WolframUUID], "Timestamp" -> serviceCreditTimestamp[$WolframUUID]|>]
	|>

handleResponse[endpoint_, response_?AssociationQ]:=
	importResponse[endpoint,
		importResponseBody[Lookup[response["Headers"], "content-type",ISReturnMessage[Symbol["IntegratedServices"], "unexp"];Throw[$Failed, "apierror"]], FromCharacterCode[response["BodyBytes"]]]] /; (response["StatusCode"] === 200)

handleResponse[endpoint_, response_?AssociationQ]:=
	handleError[endpoint, response["StatusCode"],
		importResponseBody[Lookup[response["Headers"], "content-type",ISReturnMessage[Symbol["IntegratedServices"], "unexp"];Throw[$Failed, "apierror"]], FromCharacterCode[response["BodyBytes"]]]]

importResponseBody[contenttype_,str_]:= ToExpression[str]/;textResponseQ[contenttype]
importResponseBody[contenttype_, str_]:= ImportString[str,"RawJSON"]/;jsonResponseQ[contenttype]
importResponseBody[_,_]:= $Failed

jsonResponseQ[contenttype_]:= StringContainsQ[contenttype, "application/json"]
textResponseQ[contenttype_]:= StringContainsQ[contenttype, "text/plain"]

importResponse[endpoint_,as_]:=Block[{res},
	updateAccountInfo[as];
	res=importResult[endpoint,as];
	res
]

importResult[_,as_Association]:=Switch[Lookup[as,"ResultFormat"],
	"Compressed",
		With[{held = Uncompress[as["Result"],Hold]},
			If[FreeQ[held,ServiceExecute],
				ReleaseHold[held],
				ISReturnMessage[Symbol["IntegratedServices"], "unexp"];Throw[$Failed,"unkwn"]
			]
		],
	"PlainText",
	as["Result"],
	"CloudObject",
	CloudGet[as["Result"]],
	_,
	as["Result"]
	]/;KeyExistsQ[as,"Result"]

importResult[_,expr_]:=expr

handleError[_,_,_]:= Throw[$Failed, "herror"] (* TODO: pass on messages from the response body *)


End[] (* End Private Context *)

EndPackage[]

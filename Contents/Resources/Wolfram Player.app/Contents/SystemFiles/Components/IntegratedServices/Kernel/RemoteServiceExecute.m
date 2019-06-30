(* Mathematica Package *)
BeginPackage["IntegratedServices`"]

IntegratedServices`RemoteServiceExecute

Begin["`Private`"]

$errortags = ("unkwn" | "apierror" | "herror" | "nosc" | "dialogerror" | "notos")

IntegratedServices`RemoteServiceExecute[args___]:=Catch[remoteServiceExecute[args], $errortags]

remoteServiceExecute[head_Symbol, service_String, request_String, params_Association, opts___]:= With[
	{sca = System`$ServiceCreditsAvailable,
		noscmsgname = MessageName[Evaluate@head, "nosc"], noscmsg = "Not enough Service Credits available.  Rerun when you have added more credits to your account.",
		notosmsgname = MessageName[Evaluate@head, "notos"], notosmsg = "To use this function, you must accept the terms of service."},
	Block[{api,resp},
		api = serviceEndpoint[service];
		If[FailureQ[api],
			Return[$Failed]
			,
			With[{callback = (apirequest[api, <|"Request"->request, "Parameters"->params|>])&, tos = tosApproved[service,$WolframUUID]},
			If[tos,
					If[sca>0,
						resp = callback[],
						(
							resp = IntegratedServices`CreatePurchasingDialogs[service, head];
							noscmsgname = noscmsg;
							Message[noscmsgname];
							noscmsgname =.;
							Throw[$Failed, "nosc"]
						)
					],
					resp = IntegratedServices`CreateTOSDialogs[service,head];
					If[TrueQ[resp],
						If[sca>0,
							resp = callback[],
							(
								resp = IntegratedServices`CreatePurchasingDialogs[service, head];
								noscmsgname = noscmsg;
								Message[noscmsgname];
								noscmsgname =.;
								Throw[$Failed, "nosc"]
							)
						],
						If[MatchQ[resp,$Canceled],
							notosmsgname = notosmsg;
							Message[notosmsgname];
							notosmsgname =.;
							Throw[resp, "notos"]
						,
							Throw[$Failed, "dialogerror"]
						]

					]
				]
			];
			resp
		]
	]
]

remoteServiceExecute[___]:= Throw[$Failed, "unkwn"]

End[] (* End Private Context *)

EndPackage[]

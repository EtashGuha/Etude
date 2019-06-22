(* :Title: The Channel Receiver Function  *)

(* :Context:  *)

(* :Author: Igor Bakshee *)

(* :Summary: *)

(* :Copyright: Copyright 2017 Wolfram Research, Inc. *)

(* :Package Version: 1.0 *)

(* :Mathematica Version: 10.0 *)

(* :History: *)

(* :Keywords: *)

(* :Sources: *)

(* :Warning: *)

(* :Discussion: *)

BeginPackage["ChannelReceiver`"];

ChannelReceiver`Private`Dump`ExtSymbols = {
	System`ChannelReceiverFunction
};

(* for reloading/debugging *)
Unprotect[Evaluate[ChannelReceiver`Private`Dump`ExtSymbols]];
ClearAll @@ ChannelReceiver`Private`Dump`ExtSymbols;
ClearAll @@ ({#<>"*",#<>"Private`*"}& @ Context[])

System`Private`NewContextPath[{"ChannelFramework`debug`", "ChannelFramework`", "ChannelReceiver`", "System`"}];

Begin["`Private`"];

(* some useful utilities *)
notOptionQ = ChannelFramework`Private`notOptionQ;
lazyMetaInformationAssoc = ChannelFramework`Private`lazyMetaInformationAssoc;
netaAssoc = ChannelFramework`Private`netaAssoc;
cloudPermissionsToChannelPermissions = ChannelFramework`Private`cloudPermissionsToChannelPermissions;
notOptionQ = ChannelFramework`Private`notOptionQ;

(* extra slash: CLOUD-8332, CLOUD-9799*)
$errorPage = "/channelRecieverError.html";

(* old meta can have the "Channel" data too *)
getCloudObjectMeta[co_CloudObject,___] := getCloudObjectMeta[lazyMetaInformationAssoc[co]]
getCloudObjectMeta[s_String,___] := getCloudObjectMeta[CloudObject[s]]
getCloudObjectMeta[assoc_Association,___] := assoc
getCloudObjectMeta[___] := <||>

cloudAppend[data_,co_] := CloudPut[
	Append[
		If[ListQ[#], #, {}]& @ Quiet[CloudGet[co]],
		data
	],
	co
]

(* same as for EvaluationData, but do we really need the *Complete part here? *)
SetAttributes[autoCRLogF,HoldAllComplete];
autoCRLogF[e_] := Module[
{
	result = EvaluationData[e]
},
	cloudAppend[
		Join[
			result,
			<|
				(* can be useful, but exposes too much internals? do it for "Log" -> Full ? could also add channel object, cloud object, etc.
				"HTTPRequestData" -> HTTPRequestData[],
				*)
				(*** should eventually be dropped, when EvaluationData will have "EvaluationTimestamp" of some kind, #328104 *)
				"EvaluationEndedTimestamp" -> Now
			|>
		],
		"Base/Logs/ChannelReceiverEvaluationLog"
	];
	result["Result"]
]

$logFunc = False;

(* maybe HTTPRequestData[] and whatnot should only be included in Full ? *)
crf[f_,s_,None,opts___] := crf[f,s,Identity,opts]
crf[f_,s_,Automatic,opts___ ]:= crf[f,s,autoCRLogF,opts]
crf[{args___},Hold[{s___}],logf_,opts___] := Block[{ $logFunc = True },
	(* only a call with this (documented) syntax will be recorded in the log *)
	logf[ ChannelReceiverFunction[args,opts][s] ]
]
crf[___] := $Failed

Options[ChannelReceiverFunction] = {
	"LoggingFunction" -> None
};

(* A slightly non-standard definition to let "LoggingFunction" fire both locally and in the cloud,
	and in both cases properly record all messages, even if ChannelReceiverFunction[...][...] is called with wrong arguments, opts, etc.
*)

ChannelReceiverFunction[args___?notOptionQ,opts:OptionsPattern[]][sargs___] := crf[{args},
	Hold[{sargs}],
	Quiet[OptionValue["LoggingFunction"]],
	opts
]/;!$logFunc

ChannelReceiverFunction[args___?notOptionQ,___?OptionQ][___] := $Failed/;
	!ArgumentCountQ[ChannelReceiverFunction,Length[{args}],1,1]

ChannelReceiverFunction[f_,opts:OptionsPattern[]][s_,___] := Catch[
	(* Re-executing to record OptionValue messages in the log. Slightly inefficient. Can be improved  
		by checking the first evaluation and passing the data here.
	*)
	OptionValue["LoggingFunction"];
	doCrf[f,s,{opts}]
]


doCrf[f_,s_String,opts_] := doCrf[f,
	ChannelFramework`ParseChannelMessageBody[s,ChannelReceiverFunction],
	opts	
]

doCrf[f_,assoc_Association,_] := f @ assoc

(* assume a message is already given *)
doCrf[f_,$Failed,_] := $Failed

doCrf[_,x_,_] := (
	Message[ChannelReceiverFunction::invastr,x];
	$Failed
)


ChannelReceiverFunction/:
CloudDeploy[ChannelReceiverFunction[fargs___],args___?notOptionQ,opts___?OptionQ] := Catch[ clouddeploy[{fargs},{Automatic},{args},{opts}] ]

(*** should probably limit to Rule|RuleDelayed, after the design is approved *)
cloudDeployDUMMY[_[ChannelReceiverFunction[fargs___],ch_],args___?notOptionQ,opts___?OptionQ] := Catch[ clouddeploy[{fargs},{ch},{args},{opts}] ]

With[
{
	protected = Unprotect[CloudDeploy]
},
	DownValues[CloudDeploy] = Join[
		DownValues[cloudDeployDUMMY] /. cloudDeployDUMMY -> CloudDeploy,
		DownValues[CloudDeploy]
	];
	ClearAll[cloudDeployDUMMY];
	Protect[protected];
]

chArgs2channel[Automatic] := chArgs2channel[]
chArgs2channel[args___] := ChannelObject[args] // If[ChannelObjectQ[#], #
		,
		Message[CloudDeploy::chobj, #];
		Throw[$Failed] 
	]&

(* assume a message is already given *)
clouddeploy[$Failed,__] := $Failed

clouddeploy[{fargs___},{ch0_},{args___},{opts___}] := Module[
{
	(* prevent CLOUD-8841 *)
	ometa = getCloudObjectMeta[args],
	meta = netaAssoc[opts,Options[CloudDeploy]],
	ch, url, co, metaopt
},
	If[meta === $Failed, Throw[$Failed]];
	
	(* ch0 is not defined, try getting from the cloud object *)
	If[ch0 === Automatic,
		ch = Lookup[ometa,"Channel",Automatic];
		If[ch =!= Automatic,
			ch = chArgs2channel[ch];
		];	
	];	
	If[Automatic === ch0 === ch,
		ch = ChannelObject[];
		,
		If[!ChannelObjectQ[ch],
			ch = chArgs2channel[ch0];
		]
	];
	url = ChannelURL[ch];
	If[!StringQ[url], Throw[$Failed]];
	
	(* wrapping in a List, CLOUD-8803 *)
	metaopt = MetaInformation -> Normal @ Join[ometa, <|"Channel" -> url|>];
	
	co = CloudDeploy[
		ChannelListen;
		With[{errorPage = $errorPage
		
		(***!!! temp--rm after the paclet is deployed to the cloud... at which point
			CloudEvaluate[ChannelListen; DownValues[ChannelFramework`Private`serializedDataAssocQ]] 
			will return a non-empty list
		*)
			,
			dv1 = DownValues[ChannelFramework`Private`unpackRawData],
			dv2 = DownValues[ChannelFramework`Private`toBinarySerializedData],
			dv3 = DownValues[ChannelFramework`Private`postOptions],
			dv4 = DownValues[ChannelFramework`Private`tCheckPackedHF],
			dv5 = DownValues[ChannelFramework`Private`parseMsgBody],
			dv6 = DownValues[ChannelFramework`Private`serializedDataAssocQ],
			dv7 = DownValues[ChannelFramework`Private`unpackRawDataAssoc],
			dv8 = DownValues[ChannelFramework`Private`extractRawData],
			dv9 = DownValues[ChannelFramework`Private`rawDataPrep]		
		},
			Delayed[ 
				If[HTTPRequestData[]["Body"] === "",
					HTTPRedirect[$CloudBase <> errorPage]
					,
					
					(***!!! temp--rm after #335750 / #334962 are verified as fixed *)
					ChannelListen;
					ChannelReceiverFunction;
					If[SubValues[ChannelReceiverFunction] === {}, 
						Get[FileNameJoin[{
							Lookup[PacletManager`PacletInformation["ChannelFramework"], "Location"], "Kernel", "ChannelReceiver.m"
						}] ]
					];
					(***!!! temp--rm after the paclet is deployed to the cloud *)
					ChannelListen;
					If[DownValues[ChannelFramework`Private`serializedDataAssocQ] === {},
						ChannelFramework`$BinarySerializeChannelMessages = True;
						ChannelFramework`Private`$serializedDataWrapper = "WL_SerializedData";
						ChannelFramework`Private`optionQ[e_] := OptionQ[e] || AssociationQ[e];
						ChannelFramework`Private`toSimplifiedJSON[a_ -> b_] := "{"<> wrapInQuotes[a] <> ":" <> wrapInQuotes[b] <> "}";
						DownValues[ChannelFramework`Private`toBinarySerializedData] = dv2;
						DownValues[ChannelFramework`Private`postOptions] = dv3;
						DownValues[ChannelFramework`Private`tCheckPackedHF] = dv4;
						DownValues[ChannelFramework`Private`parseMsgBody] = dv5;
						DownValues[ChannelFramework`Private`serializedDataAssocQ] = dv6;
						DownValues[ChannelFramework`Private`unpackRawData] = dv1;
						DownValues[ChannelFramework`Private`unpackRawDataAssoc] = dv7;
						DownValues[ChannelFramework`Private`extractRawData] = dv8;
						DownValues[ChannelFramework`Private`rawDataPrep] = dv9;
					];
					
					(* CloudGet[] on the deployed cloud object will show this part *)
					Module[
					{
						res = ChannelReceiverFunction[fargs] @ 
							(* Prevent immediate evaluation to let messages from ParseChannelMessageBody be recorded in the log *)
							Unevaluated @ ChannelFramework`ParseChannelMessageBody[HTTPRequestData[]["Body"],ChannelReceiverFunction],
						domain = URLParse[$ChannelBase, "Domain"],
						cookie
					},
						(* if the body uses a non-default ChannelBase, the user will have to disconnect manually, for the timer being *)
						cookie = FindCookies[domain];
						(* will bail quickly if not connected *)
						ChannelFramework`ChannelBrokerDisconnect[];
						If[#=!={} && #===cookie,
							(* ChannelBrokerDisconnect did not have a chance to fire, use brute force *)
							URLFetch[ URLBuild[ $ChannelBase, {"operation" -> "logout"}] ];
						]& @ FindCookies[domain];
						res
					]
				]
			]
		],
		args, opts, metaopt
	];
	If[Head[co] =!= CloudObject, Throw[$Failed]];
	
	clouddeploy[ CloudObject[co[[1]], metaopt], ch, {args}, {opts} ]
]

clouddeploy[co_CloudObject,ch_ChannelObject,_,opts_] := Module[
{
	oopts = Quiet[Options[ch],ChannelObject::nxst],
	nopts = {
		ChannelBrokerAction -> <|"URL" -> co[[1]]|>
	},
	perm = Permissions /. Flatten[{opts,Options[CloudDeploy]}],
	res
},
	If[oopts===$Failed, Throw[$Failed]];
	res = If[oopts === {},
		CreateChannel[ch, nopts, Permissions -> clPerm2chPerm[perm,Null]]
		,
		SetOptions[ch, nopts, Permissions -> clPerm2chPerm[perm, Permissions /. oopts]]
	];
	If[res===$Failed, Throw[$Failed]];
	co
]

(* new ChannelObject *)
clPerm2chPerm[Automatic,Null] := Automatic

(* Convention: Automatic in CloudDeploy means "respect ChannelObject's permissions."
	This is done to let the user set channelObject's permissions directly, w/o relying
	on the conversion cloudPermissionsToChannelPermissions rules.
*)
clPerm2chPerm[Automatic,chperm_] := chperm

clPerm2chPerm[perm_,_] := cloudPermissionsToChannelPermissions[perm]


(*------------------ Typesetting --------------------*)

(***!!! temp *)
ChannelReceiver`$Typesetting = True;

With[
{
	protected = Unprotect[CloudObject]
},
	(* apply only to our objects, narrowly defined *)

	CloudObject/: 
	MakeBoxes[in:CloudObject[url_,
		MetaInformation -> {"Channel" -> (_String|ChannelObject[_String])}
	], fmt:(StandardForm|TraditionalForm)] := Replace[
		MakeBoxes[CloudObject[url], fmt],
		InterpretationBox[boxes_, _, rest___] :> InterpretationBox[boxes, in, rest]
	] /; $Typesetting;

	Protect[protected];
]

(*--------------------------*)

End[];

System`Private`RestoreContextPath[];

(*** rm
(SetAttributes[#,ReadProtected]; Protect[#])& @ DeleteCases[
	ChannelReceiver`Private`Dump`ExtSymbols ,
	_String
];
*)

EndPackage[];

$ContextPath = DeleteCases[$ContextPath, "ChannelReceiver`"];

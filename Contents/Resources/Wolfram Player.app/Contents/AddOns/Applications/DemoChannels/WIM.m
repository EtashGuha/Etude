(* $Id$ *)

(* :Title: *)

(* :Context: *)

(* :Author: Igor Bakshee *)

(* :Summary: *)

(* :Copyright: Copyright 2014 Wolfram Research, Inc. *)

(* :Package Version: 1.0 ($Revision$/$Date$) *)

(* :Mathematica Version: 10.0 *)

(* :History: *)

(* :Keywords: *)

(* :Sources: *)

(* :Warning: *)

(* :Discussion: *)

BeginPackage["DemoChannels`WIM`"];

(* for reloading/debugging *)
Unprotect[Evaluate[Context[]<>"*"]];
ClearAll @@ ({#<>"*",#<>"Private`*"}& @ Context[])

CreateWIMChannel::usage = "CreateWIMChannel[] creates a demo WIM channel for the current authenticated user. CreateWIMChannel[\"Public\"] creates a public WIM channel.";
SetWIMAlias::usage = "SetWIMAlias[\"alias\"]";
WIMPreSend::usage = "WIMPreSend[\"msg\"] or WIMPreSend[\"msg\", attachment]";
WIMReceiver::usage = "WIMReceiver[assoc]";

Begin["`Private`"];

Needs["ChannelFramework`"];

CreateWIMChannel::connect = "Please connect to the Wolfram Cloud to create a channel.";
WIMPreSend::connect = "Please connect to the Wolfram Cloud to send a message.";

CreateWIMChannel[] := CreateWIMChannel[
	<|"All" -> "Write", "Owner" -> {"Read", "Execute"}|>
]/;StringQ[$WolframID]

CreateWIMChannel[perm_] := CreateChannel["WIM",
	HandlerFunctions -> <| "MessageReceived" -> WIMReceiver|>, 
	ChannelPreSendFunction -> WIMPreSend,
	NotificationFunction -> "TicTac",
	Initialization :> Needs["DemoChannels`WIM`"],
	Permissions -> perm
]/;StringQ[$WolframID]

CreateWIMChannel[args___] := (
	If[$WolframID===None,
		Message[CreateWIMChannel::connect];
		Return[$Failed]
	];
	Null /; False
)/;ArgumentCountQ[CreateWIMChannel,Length[{args}],0,1]


SetWIMAlias[s_] := $wimAlias = s

WIMPreSend[msg_String,opts___] := WIMPreSend[{msg},opts]
WIMPreSend[msg_,opts___] := WIMPreSend[{TextString[msg]},opts]

in:WIMPreSend[{msg_String,nbo_NotebookObject},opts___] := Module[
{
	nb = NotebookGet[nbo]
},
	If[nb===$Failed,
		Message[WIMPreSend::inv,HoldForm[in],nbo,NotebookObject];
		Return[$Failed]
		,
		WIMPreSend[{msg,nb},opts]
	]
]

WIMPreSend[{msg_String,body_:None},___] := Module[{},
	<|
		"From" -> $wimAlias, 
		"Message" -> msg, 
		"WL_Compressed" -> Compress[body],
		"ReceiptTo" -> $WolframID<>":WIM"
	|>
]/;StringQ[$WolframID]

WIMPreSend[___] := (
	Message[WIMPreSend::connect];
	$Failed
)/;$WolframID===None

WIMPreSend[___] := (
	Message[WIMPreSend::args, WIMPreSend];
	$Failed
)

$wimAlias := $WolframID

createDocument[HoldComplete[e_]] := CreateDocument[e]
createDocument[_] := $Failed

WIMReceiver[assoc0_Association] := Module[
{
	assoc = assoc0["Message"],
	replyUI = DefaultButton[],
	attachmentUI = Sequence[],
	from
},
	from = If[KeyExistsQ[assoc,"From"], assoc["From"], "Anonymous"];
	
	If[KeyExistsQ[assoc,"WL_Compressed"] && assoc["WL_Compressed"] =!= HoldComplete[None],
		attachmentUI = Button[Style["Open attachment", "Hyperlink"],
			With[{nb = createDocument[ #["WL_Compressed"] ]},
				If[Head[nb] === NotebookObject,
					SetSelectedNotebook[nb]
				]
			],
			Appearance -> None
		]&@assoc
	];
	If[KeyExistsQ[assoc,"ReceiptTo"],
		replyUI = DynamicModule[{reply = "", receiptTo = assoc["ReceiptTo"]},
			Column @ {
				TextCell["Any reply?"], 
				InputField[Dynamic[reply], String], 
				DefaultButton["Reply",
					DialogReturn[			    
						ChannelSend[receiptTo, reply];
					]
				]
			}
		]
	];
	CreateDialog[{
		TextCell["A message from " <> from <> ":"],
		TextCell[If[AssociationQ[assoc],assoc["Message"],assoc]],
		attachmentUI,
		replyUI
	}]
]

End[];

EndPackage[]

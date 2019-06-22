(* Wolfram Language Package *)


BeginPackage["ResourceSystemClient`"]

ResourceSystemClient`$progressID;

Begin["`Private`"] (* Begin Private Context *) 

ResourceSystemClient`$ShowResourceProgress=True;

$ProgressIndicatorContent="";

$DefaultResourceDownloadMessage="Downloading content from the repository \[Ellipsis]";
installPrintTemp[]:=installPrintTemp[$DefaultResourceDownloadMessage]
installPrintTemp[message_]:=PrintTemporary[downloadingMessageWindow[message]]
downloadingMessageWindow[]:=downloadingMessageWindow[$DefaultResourceDownloadMessage]
downloadingMessageWindow[message_String]:=progressPanel[message]

downloadingMessageWindow[message_String, progress_?NumberQ]:=progressPanel[message]

downloadingMessageWindow[message_String, _]:=downloadingMessageWindow[message]
downloadingMessageWindow[progress_?NumberQ]:=downloadingMessageWindow[$DefaultResourceDownloadMessage,progress]

downloadingMessageWindow[___]:=downloadingMessageWindow[$DefaultResourceDownloadMessage]

ResourceSystemClient`$progressID=None;

printTempOnce[None,___]:=None;
printTempOnce[pid_]:=printTempOnce[pid,Dynamic[$ProgressIndicatorContent]]
printTempOnce[pid_,args___]:=With[{cell=printTemporaryFrontEnd[args]},
		printTempOnce[pid,___]=printTempCell[pid]=cell;
		cell
]

clearPrintTemp=clearTempPrint;
clearTempPrint[None]:=Null;
clearTempPrint[pid_Symbol]:=clearTempPrint[printTempCell[pid]];
clearTempPrint[cell_CellObject]:=Quiet[NotebookDelete[cell]];

printTemporaryFrontEnd[args___]:=If[$Notebooks&&$ShowResourceProgress,PrintTemporary[args],None]

handleHeaders[as_, size_] := With[{
	new = Lookup[as, "ByteCountTotal", 
		Lookup[Lookup[as, "Headers", Association[]], "content-length", None]]},
  	If[IntegerQ[new], new, size]
  	]
  	
handleProgress[as_, size_] := handleprogress[
	Lookup[as, {"ByteCountTotal", "ByteCountDownloaded"}, None], size]

$ResourceDownloadProgressMinimumSize=10^6;

handleprogress[{total_Integer, _}, _] := {None, total}/;total<$ResourceDownloadProgressMinimumSize
handleprogress[{total_Integer, dl_Integer}, _] := {dl, total}
handleprogress[{_, _}, old_Integer] := {None, old}/;old<$ResourceDownloadProgressMinimumSize
handleprogress[{_, dl_Integer}, old_Integer] := {dl, old};
handleprogress[{_, _}, old_] := {None, old};

$progressBarCloudUpdatePeriod=20;

handleStatus[as_,status_]:=With[{sc=as["StatusCode"]},
	If[MissingQ[sc],status,handlestatus[sc]]]

handlestatus[200]="Success";
handlestatus[302|413|403]="Unauthorized";
handlestatus[expr_]=expr;

toStr[e_ /; e > 100] := TextString @ Round @ e;
toStr[e_] := TextString[NumberForm[e, 3]];

MemoryString[b_] := Block[
	{gb = N[b / 1*^9], mb = N[b / 1*^6], kb = N[b / 1*^3]},
	Which[
		gb > 1, toStr[gb] <> " GB", 
		mb > 1, toStr[mb] <> " MB",
		kb > 1, toStr[kb] <> " KB",
		True, toStr[Round[b]] <> " bytes"
	]
];

progressPanel:=(TextString;
	If[Length[DownValues[GeneralUtilities`ProgressPanel]]>0,
		progressPanel=GeneralUtilities`ProgressPanel
		,
		progressPanel=(Panel[Style[#, "Button",  GrayLevel[0.5]], 
			Appearance -> {"Default" ->  FrontEnd`FileName[{"Typeset", "PrintTemporary"}, "LightBlue.9.png"]}, 
			Alignment -> {Center, Center}, FrameMargins -> {{12, 12}, {8, 12}}]&)
	])

	



End[] (* End Private Context *)

EndPackage[]
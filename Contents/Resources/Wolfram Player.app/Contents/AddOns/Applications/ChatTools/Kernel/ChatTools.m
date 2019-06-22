(* ::Package:: *)

(* :Name: ChatTools` *)

(* :Title: Functions for ChatTools.  *)

(* :Author: Jay Warendorff *)

(* :Copyright: (c) 2017, Wolfram Research, Inc. All rights reserved. *)

(* :Mathematica Version: 11.3 *)

(* :Package Version: 0.01 *)

(* :Summary: Functions for ChatTools. *)



BeginPackage[ "ChatTools`"];


ChatTools`$ChatToolsDir; ChatTools`SaveChatNotebook; ChatTools`SendChat; ChatTools`$ScreenName; ChatTools`StartChatChannelListener; ChatTools`sendPrivateChatMessage;
ChatTools`sendPrivateChatMessage2; ChatTools`SendMessageToRoom; ChatTools`SetBannerDialog; ChatTools`InsertMoreRoomCellsFromCloud; ChatTools`InsertAllRoomCellsFromCloud; ChatTools`SetRoomBanner;
ChatTools`CreateChatRoom; ChatTools`DeleteChatRoom; ChatTools`ChatRoomModerators; ChatTools`SetChatRoomModerators; ChatTools`ChatRooms; ChatTools`NotebookCorrespondingToID;
ChatTools`cSendMessage; ChatTools`SendChatNotebookInformation; ChatTools`SendChatCells; ChatTools`InsertAllSentCellsFromCloud; ChatTools`InsertMoreSentCellsFromCloud;
ChatTools`MoveCursorAfterCellPosition; ChatTools`RemoveRoomListenerAndCloseRoom; ChatTools`ClearSentCells; ChatTools`$contactsToAdd; ChatTools`AcceptChatInvitation; ChatTools`SetAcceptRejectEmails;
ChatTools`SetAcceptRejectRegularExpressions; ChatTools`RemoveContacts; ChatTools`AuxiliaryUpdateParticipants; ChatTools`UpdateParticipants; ChatTools`auxiliaryCSendMessage;
ChatTools`auxiliaryRoomSendMessage; ChatTools`Chat;

Begin["`Private`"];

If[FrontEnd`Private`$KernelName === "ChatServices",

Off[ChannelSend::rcvr, ChannelListen::access, ChannelListen::rerr, ChannelListen::invldtd, CloudConnect::clver];

If[MathLink`CallFrontEnd[MLFS`FileType[FrontEnd`FileName[{FrontEnd`$ApplicationDocumentsDirectory}]]] === Directory, 
	MathLink`CallFrontEnd[MLFS`CreateDirectory[FrontEnd`FileName[{FrontEnd`$ApplicationDocumentsDirectory, "WolframChats"}]]]];

CurrentValue[$FrontEnd,{PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "SaveDirectory"}] =
	If[MathLink`CallFrontEnd[MLFS`FileType[FrontEnd`FileName[{FrontEnd`$ApplicationDocumentsDirectory, "WolframChats"}]]] === Directory, 
		FrontEndExecute@FrontEnd`ToFileName[FrontEnd`FileName[{FrontEnd`$ApplicationDocumentsDirectory, "WolframChats"}]], 
		FrontEndExecute@FrontEnd`ToFileName[FrontEnd`FileName[{FrontEnd`$ApplicationDocumentsDirectory}]]];

$AppDir = DirectoryName @ System`Private`$InputFileName;

$ChatToolsDir::notfound = 
"WARNING: The ChatTools installation directory was not found. \
Please set the parameter $ChatToolsDir to point to the installation \
directory; otherwise, certain features of ChatTools may not work as \
expected.";

$ChatToolsDir = 
	Quiet@Check[DirectoryName[FindFile["ChatTools`"], 2], 
	MessageDialog[$ChatToolsDir::notfound]; $Failed];
	
GetStillActiveInvitationsSentBeforeStartup[]:=
	Module[{ids, chatnotebooks, chatnotebookids, joinchatdialogIDs, idsToExclude, chatdata, recentchatdata}, 
		ids = TimeConstrained[URLExecute[CloudObject["https://www.wolframcloud.com/objects/7c85be03-b879-4cd1-ad79-7d34d8121053"], {"wid" -> $WolframID}, "WL", Method -> "POST"],3,$Failed];
		If[MatchQ[ids,{__String}],
			chatnotebooks = Select[Notebooks[], CurrentValue[#, {TaggingRules, "ChatNotebook"}] === "True" &];
			chatnotebookids = CurrentValue[#, {TaggingRules, "ChatNotebookID"}] & /@ chatnotebooks;
			joinchatdialogIDs = Select[CurrentValue[#, {TaggingRules, "JoinChat"}]&/@Notebooks[], StringQ];
			idsToExclude = Join[chatnotebookids, joinchatdialogIDs];
			chatdata = With[{idsToExclude1 = idsToExclude}, 
					Quiet[Cases[{#, CloudGet["https://www.wolframcloud.com/objects/" <> #]} & /@ids, {_, Association["ChatNotebookID" -> (i_ /; Not@MemberQ[idsToExclude1, i]), 
																	"wid" -> (a_ /; a =!= $WolframID), __]}],{CloudGet::cloudnf}]];
			If[chatdata =!= {},
				recentchatdata = Cases[chatdata, {_, Association[__, "ChatCreationDate" -> a_ /; Abs@DateDifference[Now, ToExpression@a] <= Quantity[1, "Days"], __]}];
				If[MatchQ[recentchatdata, {{_String, Association["ChatNotebookID" -> _, __]} ..}], 
					JoinChatDialog[#[[2]]["wid"], Uncompress[#[[2]]["ScreenName"]], "chatframework@wolfram.com" <> "/" <> #[[1]]] & /@ recentchatdata]]]];
				
InitializeChatSmall[] :=
	Module[{cico, ciOpts},
If[TrueQ@CurrentValue["InternetConnectionAvailable"] && TrueQ@CurrentValue["AllowDownloads"],
	Check[If[And[TrueQ@CurrentValue[$FrontEnd, "WolframCloudConnected"], $WolframID === None],
		(* Attempt to give $WolframID a value besides None in the ChatServices kernel. For example M- starts and user at first not logged in and then logs into local kernel. *)CloudConnect["Prompt" -> False]];
	If[And[TrueQ@CurrentValue[$FrontEnd, "WolframCloudConnected"], ($WolframID =!= None),
		TrueQ@CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "AvailableToReceiveChatInvitations"}]],
		$InitialChatSmallDone = True;
		ChannelListen; Once[BlacklistDynamics[]];
 		If[FindChannels["ChatInvitations"] === {},
 			CreateChannel["ChatInvitations", Permissions -> {"All" -> "Write", $WolframID -> {"Read", "Write", "Execute"}},
								HandlerFunctions -> Association["MessageReceived" -> (ChatTools`sendPrivateChatMessage[#RequesterWolframID, #Message]&)]],
			cico = ChannelObject["ChatInvitations"];
			ciOpts = Options[cico, Permissions];
			If[FreeQ[Normal[ciOpts], (All | "All") -> "Write"], 
				SetOptions[cico, Permissions -> Association[All -> "Write", $WolframID -> {"Write", "Read", "Execute"}]]]];
  		If[Not@StringQ@FirstCase[#["URL"] & /@ ChannelListeners[], a_String /; StringMatchQ[a, StringExpression[__, "/", $WolframID, "/", "ChatInvitations"]] :> a],
  			ChannelListen[ChannelObject[$WolframID <> ":ChatInvitations"], "TrustedChannel" -> True]];
  		CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "WolframID"}] = $WolframID;
  		GetStillActiveInvitationsSentBeforeStartup[]],
			
		MessageDialog["Set $CloudBase to \"https://www.wolframcloud.com/\", login and restart Mathematica.", WindowFrame -> "ModalDialog", WindowSize -> {560, All}],
		
		CloudConnect::fbdn]]];

RestartListeners[] :=
	Module[{wid = CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "WolframID"}], channellisteners, currentListenerIDs, chatRoomNotebooks, currentChatRoomIDs,
		currentChatRoomIDsWithoutCorrespondingListeners, sel, urls, ids, sel2, id, originator, screenname, removedparticipant, li, screenname1, chatnbs, nbids, iddata, idsRestart},
		
Quiet[If[TrueQ@CurrentValue["InternetConnectionAvailable"]&&TrueQ@CurrentValue["AllowDownloads"],
	
	If[And[TrueQ@CurrentValue[$FrontEnd, "WolframCloudConnected"], $WolframID === None],
		(* Attempt to give $WolframID a value besides None in the ChatServices kernel. For example M- starts and user at first not logged in and then logs into local kernel. *)CloudConnect[]];
		
	(* If invitations listener died either from computer sleep or changed WolframID restart it *)
	If[And[TrueQ@CurrentValue[$FrontEnd, "WolframCloudConnected"],
		$WolframID =!= None,
		Or[Not@StringQ@FirstCase[#["URL"] & /@ (channellisteners = ChannelListeners[]), a_String /; StringMatchQ[a, StringExpression[__, "/", $WolframID, "/", "ChatInvitations"]]], 
			wid =!= $WolframID],
		TrueQ@CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "AvailableToReceiveChatInvitations"}]],
		
		ChannelListen[ChannelObject[$WolframID <> ":ChatInvitations"], "TrustedChannel" -> True];
		
		(* If the listeners of some private chats have terminated - for example: user changed wolfram id - end use of those private chats. *)
		sel = Select[Notebooks[], And[Not@TrueQ@CurrentValue[#, {TaggingRules, "ChatRoom"}], Not@TrueQ@CurrentValue[#, {TaggingRules, "BeforeSend"}],
					Not@TrueQ@CurrentValue[#, {TaggingRules, "Terminated"}],
					CurrentValue[#, {TaggingRules, "ChatNotebook"}] === "True"] &];
		urls = #["URL"] & /@ channellisteners;
		ids = Select[(CurrentValue[#, {TaggingRules, "ChatNotebookID"}] & /@ sel),
			MemberQ[StringReplace[#, "https://channelbroker.wolframcloud.com/users/chatframework@wolfram.com/" -> ""] & /@ urls, #] &];
		sel2 = Select[Notebooks[], MemberQ[ids, CurrentValue[#, {TaggingRules, "ChatNotebookID"}]] &];
		If[sel2 =!= {},
			(CurrentValue[#, {TaggingRules, "Terminated"}] = True;
			CurrentValue[#, DockedCells] = {};
			st = Cells[#, CellStyle -> "Stem"];
			If[st =!= {},
				SetOptions[st[[-1]], Deletable -> True];
				NotebookWrite[st[[1]], Cell["The channel listener for this chat has been terminated.", "Text"], 
					AutoScroll -> (!CurrentValue[#, {TaggingRules, "ScrollLock"}, False])]];
			id = CurrentValue[#, {TaggingRules, "ChatNotebookID"}];
			originator = CurrentValue[#, {TaggingRules, "Originator"}];
			screenname = CurrentValue[#, {TaggingRules, "ScreenName"}];
			removedparticipant = CurrentValue[#, {TaggingRules, "OriginalWolframID"}];
			screenname1 = Compress@If[MemberQ[{"", Inherited}, screenname] || (StringQ@screenname && StringMatchQ[screenname, Whitespace]), "None", screenname];
			URLExecute[CloudObject["https://www.wolframcloud.com/objects/346cb1d3-8ba8-45a4-8baf-417a16f3b6fd"],
					{"id" -> id, "removedparticipant" -> removedparticipant, "screenname" -> screenname1}, "WL", Method -> "POST"])&/@sel2];
		If[wid =!= $WolframID, CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "WolframID"}] = $WolframID]];
		
	If[And[TrueQ@CurrentValue[$FrontEnd, "WolframCloudConnected"], ($WolframID =!= None),
		TrueQ@CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "AvailableToReceiveChatInvitations"}]],
		
		If[wid === $WolframID,
			(* Restart chat channel listeners for private chats. *)
			chatnbs = Select[Notebooks[], And[Not@TrueQ@CurrentValue[#, {TaggingRules, "ChatRoom"}], Not@TrueQ@CurrentValue[#, {TaggingRules, "BeforeSend"}], 
								Not@TrueQ@CurrentValue[#, {TaggingRules, "Terminated"}], CurrentValue[#, {TaggingRules, "ChatNotebook"}] === "True"] &];
			If[chatnbs =!= {},
				nbids = CurrentValue[#, {TaggingRules, "ChatNotebookID"}] & /@ chatnbs;
				iddata = StringReplace[#, "https://channelbroker.wolframcloud.com/users/chatframework@wolfram.com/" -> ""] &@(#["URL"] & /@ ChannelListeners[]);
				idsRestart = Select[nbids, Not@MemberQ[iddata, #] &];
				ChannelListen[ChannelObject["chatframework@wolfram.com" <> ":" <> #], "TrustedChannel" -> True] & /@ idsRestart]];
				
		currentListenerIDs = StringReplace[#["URL"], __ ~~ "/" ~~ (a_ /; StringFreeQ[a, $PathnameSeparator]) :> a] & /@ ChannelListeners[];
		chatRoomNotebooks = Select[Notebooks[], CurrentValue[#, {TaggingRules, "ChatRoom"}] === True &];
		currentChatRoomIDs = CurrentValue[#, {TaggingRules, "ChatNotebookID"}] & /@ chatRoomNotebooks;
		currentChatRoomIDsWithoutCorrespondingListeners = Complement[currentChatRoomIDs, currentListenerIDs];
		(* If there are open rooms without listeners restart them. *)
		ChannelListen[ChannelObject["chatframework@wolfram.com" <> ":" <> #], "TrustedChannel" -> True] & /@ currentChatRoomIDsWithoutCorrespondingListeners]],CloudConnect::fbdn]];

Chat::nosupp="`` is not supported in dynamic expressions in chat notebooks.";

blacklist={"CurrentImage","ImageCapture","AudioCapture","CurrentNotebookImage","CurrentScreenImage"};
BlacklistDynamics[]:=
	Map[(Unprotect[#]; Clear[#]; (* This will wipe out the autoload defs that trigger autoloading of RobotTools *) With[{sym = Symbol[#]},
		TagSetDelayed[sym, Dynamic[sym[___], ___], Message[Chat::nosupp, #]]; TagSetDelayed[sym, Manipulate[sym[___], ___], Message[Chat::nosupp, #]]; SetAttributes[sym, {Protected, Locked}]])&,
		blacklist];
    
SetAttributes[ButtonPairSequence, HoldAll];

ButtonPairSequence[button1_, button2_] := If[$OperatingSystem === "MacOSX", Unevaluated[Sequence @@ {button1, button2}], Unevaluated[Sequence @@ {button2, button1}]];
    
FirstLoginMessage[]:= 
	NotebookPut[Notebook[{Cell["", CellMargins -> {{Automatic, Automatic}, {1, 1}}],
				Cell["Connect to Cloud", "EnableChatTitleText"], 
				Cell["Chat functionality requires connecting to the Wolfram Cloud.", "EnableChatText"], 
				Cell[BoxData[GridBox[{{ButtonPairSequence[ButtonBox[StyleBox["Cancel", FontColor -> GrayLevel[0]], ButtonFunction :> DialogReturn[],
									Appearance -> {"ButtonType" -> "Cancel", "Cancel" -> None}, Background -> GrayLevel[.9], Evaluator -> Automatic, 
									Method -> "Preemptive", ImageSize -> {70, 25}], 
							ButtonBox[StyleBox["   Connect   ", FontColor -> GrayLevel[1]],
								ButtonFunction :> (CloudConnect[]; If[TrueQ[$CloudConnected], DialogReturn[]; ChatTools`Private`InitializeChatSmall[];]), 
								Appearance -> {"ButtonType" -> "Default", "Default" -> FrontEnd`ToFileName[{"Toolbars", "ChatTools"}, "SendButton-Default.9.png"], 
						"Hover" -> FrontEnd`ToFileName[{"Toolbars", "ChatTools"}, "SendButton-Hover.9.png"]}, FontColor -> GrayLevel[1], Background -> RGBColor[0., 0.5548332951857786, 1.], 
								Evaluator -> "ChatServices", Method -> "Queued"]]}}, 
							GridBoxAlignment -> {"Columns" -> {{Right}}}, GridBoxSpacings -> {"Columns" -> {{1}}}]], "EnableChatButtons"]}, WindowSize -> {520, 160}, 
				ShowCellBracket -> False,
				"CellInsertionPointCell" -> {}, 
				"BlinkingCellInsertionPoint" -> False, 
				"CellInsertionPointColor" -> GrayLevel[1], 
				WindowFrame -> "ModalDialog",
				WindowElements -> {}, 
				WindowFrameElements -> {"CloseBox"},
				ShowStringCharacters -> False,
				Background -> GrayLevel[1], 
				ScrollingOptions -> {"PagewiseScrolling" -> False, "PagewiseDisplay" -> True, "VerticalScrollRange" -> Fit}, 
				CellMargins -> {{0, 0}, {0, 0}}, 
				AutoMultiplicationSymbol -> False,
				Saveable -> False, 
				WindowTitle -> "Wolfram Chat",
				Editable -> False, 
				Selectable -> False, 
				StyleDefinitions -> Notebook[{Cell[StyleData["EnableChatTitleText"], FontSize -> 20, FontFamily -> "Source Sans Pro", FontColor -> RGBColor[.2, .2, .2],
									ShowCellBracket -> False, CellMargins -> {{30, 30}, {2, 14}}], 
								Cell[StyleData["EnableChatText"], FontSize -> 12, FontFamily -> "Source Sans Pro", FontColor -> RGBColor[.39215, .39215, .39215], 
									ShowCellBracket -> False, CellMargins -> {{30, 30}, {2, 14}}], 
								Cell[StyleData["EnableChatButtons"], TextAlignment -> Right, CellMargins -> {{30, 30}, {2, 15}}, 
										ButtonBoxOptions -> {ImageSize -> {80, 24}, BaseStyle -> {FontFamily -> "Source Sans Pro", FontSize -> 14}}]}], 
				NotebookEventActions -> {"ReturnKeyDown" :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
							{"MenuCommand", "EvaluateCells"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
							{"MenuCommand", "HandleShiftReturn"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
							{"MenuCommand", "EvaluateNextCell"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]], 
							"EscapeKeyDown" :> (FE`Evaluate[FEPrivate`FindAndClickCancelButton[]]; DialogReturn[$Failed]),
							"WindowClose" :> (FE`Evaluate[FEPrivate`FindAndClickCancelButton[]]; DialogReturn[$Failed])}, Evaluator -> "ChatServices"],
			Evaluator -> "ChatServices"];
	
chatIcon = Import[FileNameJoin[{$ChatToolsDir, "FrontEnd", "SystemResources", "ChatIcon.m"}]];

chatIconSmall = Import[FileNameJoin[{$ChatToolsDir, "FrontEnd", "SystemResources", "ChatIconSmall.m"}]];
   
activated = Import[FileNameJoin[{$ChatToolsDir, "FrontEnd", "SystemResources", "ActivatedIcon.m"}]];
   
deactivated = Import[FileNameJoin[{$ChatToolsDir, "FrontEnd", "SystemResources", "DeactivatedIcon.m"}]];

dialogopts = Sequence[ShowCellBracket -> False, "CellInsertionPointCell" -> {}, "BlinkingCellInsertionPoint" -> False, "CellInsertionPointColor" -> GrayLevel[1], WindowSize -> All,
			WindowFrame -> "ModelessDialog", WindowElements -> {}, WindowFrameElements -> {"CloseBox"}, ShowStringCharacters -> False, Background -> White,
			ScrollingOptions -> {"PagewiseScrolling" -> False, "PagewiseDisplay" -> True, "VerticalScrollRange" -> Fit}, ShowCellBracket -> False, CellMargins -> {{0, 0}, {0, 0}},
			AutoMultiplicationSymbol -> False, Saveable -> False, WindowTitle -> "Wolfram Chat", TaggingRules -> {"Dialog" -> "ChatSettings"}, (*Evaluator->"ChatServices",*) Editable -> False,
			Selectable -> False];

RunSettingsTask[nb_NotebookObject]:=RunScheduledTask[If[$CloudConnected,
           If[FindChannels["ChatInvitations"] === {},
             CreateChannel["ChatInvitations", Permissions -> {"All" -> "Write", $WolframID -> {"Read", "Write", "Execute"}},
               HandlerFunctions -> Association["MessageReceived" -> (ChatTools`sendPrivateChatMessage[#RequesterWolframID, #Message] &)]]];
             If[Not@StringQ@FirstCase[#["URL"] & /@ ChannelListeners[], a_String /; StringMatchQ[a, StringExpression[__, "/", $WolframID, "/", "ChatInvitations"]]],
		ChannelListen[ChannelObject[$WolframID <> ":ChatInvitations"], "TrustedChannel" -> True]]];
             CurrentValue[nb,{TaggingRules,"initDone"}]=True; RemoveScheduledTask[$ScheduledTask];,{0.5}];

settingsdialog[listenerq_] :=  
  With[{a=ChatTools`Private`activated,d=ChatTools`Private`deactivated},
    Notebook[{
      Cell[BoxData[GridBox[{{ChatTools`Private`chatIcon, Cell[TextData["Chat Settings"], "ChatSettingsTitleText"]}}, GridBoxSpacings->{"Columns"->{{1}}}]], "ChatSettingsTitle"],
      Cell["","ChatSettingsDelimiter"],
      Cell["Screen Name","ChatSettingsSection"],
      Cell[BoxData[InputFieldBox[Dynamic[CurrentValue[EvaluationNotebook[], {"TaggingRules", "ScreenName"}, ""]], String, BoxID -> "screenname"]],"ChatSettingsNameField"],
      Cell["Chat Services","ChatSettingsSection"],
      
      Cell[BoxData[DynamicBox[
       If[CurrentValue[EvaluationNotebook[], {TaggingRules, "initDone"}, listenerq] === True,
        PaneSelectorBox[{
          True -> TagBox[ButtonBox[a, ButtonFunction:>(CurrentValue[$FrontEnd, "AllowChatServices"] = False),Appearance -> {"Default" -> None},Evaluator->"Local",Background->GrayLevel[1],Method -> "Preemptive"],
          		MouseAppearanceTag["LinkHand"]],
          False->TagBox[ButtonBox[d,
            ButtonFunction:>(CurrentValue[$FrontEnd, "AllowChatServices"] = True),Appearance -> {"Default" -> None},Evaluator->"Local",Method -> "Preemptive"],MouseAppearanceTag["LinkHand"]]},
        Dynamic[CurrentValue[$FrontEnd, "AllowChatServices"]]],
        GridBox[{{StyleBox["Activating", FontFamily -> "Source Sans Pro"], 
          InterpretationBox[DynamicBox[FEPrivate`FrontEndResource["FEExpressions","PercolateAnimator"][Medium], ImageSizeCache->{50.,{2.,10.}}],
            ProgressIndicator[Appearance->"Percolate"],BaseStyle->{"Deploy"}]}},
            GridBoxSpacings->{"Columns"->{{2}}}]]]], "ChatSettingsSwitch"],
            
        Cell[BoxData[GridBox[{{CheckboxBox[Dynamic[CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "AvailableToReceiveChatInvitations"}, True],
          			(Set[CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "AvailableToReceiveChatInvitations"}],
          				If[TrueQ@CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "AvailableToReceiveChatInvitations"}], False, True]]; 
				If[TrueQ@CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "AvailableToReceiveChatInvitations"}],
					Needs["ChatTools`"]; ChatTools`Private`GetStillActiveInvitationsSentBeforeStartup[]]) &, Evaluator -> "ChatServices"],
          		Enabled -> Dynamic[TrueQ@CurrentValue[$FrontEnd, "AllowChatServices"]]],
          PaneSelectorBox[{True -> StyleBox["Accept new invitations", FontFamily -> "Source Sans Pro"], 
				False -> StyleBox["Accept new invitations", FontFamily -> "Source Sans Pro", FontColor -> GrayLevel[.5]]}, 
			Dynamic[TrueQ@CurrentValue[$FrontEnd, "AllowChatServices"]]]}}, 
           GridBoxSpacings->{"Columns"->{{1}}}]],"ChatSettingsNewInvites"],
           
        Cell["","ChatSettingsDelimiter"],
        Cell[BoxData[GridBox[{{ButtonPairSequence[ButtonBox[StyleBox["Cancel", FontColor -> GrayLevel[0]], ButtonFunction :> FrontEnd`NotebookClose[FrontEnd`ButtonNotebook[]], Evaluator -> None,
        							Appearance -> {"ButtonType" -> "Cancel", "Cancel" -> None}, Background -> GrayLevel[.9], ImageSize -> {70, 25}], 
				ButtonBox[StyleBox["   OK   ", FontColor -> GrayLevel[1]], ButtonFunction :> (Function[t, If[StringQ@t, 
	CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "ScreenName"}] = StringTrim@t]][CurrentValue[ButtonNotebook[], {"TaggingRules", "ScreenName"}]]; NotebookClose[]), 
						Appearance -> {"ButtonType" -> "Default", "Default" -> FrontEnd`ToFileName[{"Toolbars", "ChatTools"}, "SendButton-Default.9.png"], 
						"Hover" -> FrontEnd`ToFileName[{"Toolbars", "ChatTools"}, "SendButton-Hover.9.png"]}, Method -> "Preemptive", Evaluator -> "Local", 
						Background -> RGBColor[0., 0.5548332951857786, 1.]]]}}, 
   GridBoxAlignment -> {"Columns" -> {{Right}}}, 
   GridBoxSpacings -> {"Columns" -> {{1}}}]], "ChatSettingsButtons"]}, WindowSize->{480,330}, TaggingRules->{"initDone"->listenerq,
	"ScreenName" -> (Function[s, If[StringQ@s && Not@StringMatchQ[s, "" | Whitespace], s, ""]][CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "ScreenName"}]])},
	ChatTools`Private`dialogopts,
      StyleDefinitions->Notebook[{
        Cell[StyleData["ChatSettingsTitleText"],FontSize -> 20, FontFamily->"Source Sans Pro",FontColor -> GrayLevel[0.2],ShowCellBracket->False],
        Cell[StyleData["ChatSettingsTitle"],CellMargins->{{30,30},{2,14}},ShowCellBracket->False,
          GridBoxOptions->{GridBoxAlignment->{"Rows"->{{Center}},"Columns"->{{Left}}}}],
        Cell[StyleData["ChatSettingsDelimiter"],
          CellSize->{Automatic,1},ShowCellBracket->False,ShowStringCharacters->False,CellFrame->{{0,0},{.5,0}},CellMargins->{{0,0},{0,0}},CellFrameColor->GrayLevel[.75]],
        Cell[StyleData["ChatSettingsSection"],
          FontSize -> 16, FontColor->GrayLevel[0.2], FontFamily->"Source Sans Pro",ShowCellBracket->False,ShowAutoSpellCheck -> False,
          CellMargins->{{30,30},{5,20}}],
        Cell[StyleData["ChatSettingsNameField"],
          CellMargins->{{30,30},{0,5}},
          InputFieldBoxOptions->{BaseStyle->{FontFamily->"Source Sans Pro",FontSize->13},FieldSize->{28,1.3},Alignment->{Left,Center}}],
        Cell[StyleData["ChatSettingsSwitch"],
          CellMargins->{{30,30},{5,5}},ShowCellBracket->False],
        Cell[StyleData["ChatSettingsNewInvites"],
          CellMargins->{{30,30},{5,5}},ShowCellBracket->False],
        Cell[StyleData["ChatSettingsButtons"],TextAlignment->Right,
          CellMargins->{{30,30},{40,15}},
          ButtonBoxOptions->{ImageSize->{60,24}}]}],
		NotebookEventActions -> {"ReturnKeyDown" :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
					{"MenuCommand", "EvaluateCells"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
					{"MenuCommand", "HandleShiftReturn"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
					{"MenuCommand", "EvaluateNextCell"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]], 
					"EscapeKeyDown" :> (FE`Evaluate[FEPrivate`FindAndClickCancelButton[]]; DialogReturn[$Failed]), 
					"WindowClose" :> (FE`Evaluate[FEPrivate`FindAndClickCancelButton[]]; DialogReturn[$Failed])}]];

auxStartChatChannelListener[] := 
	Module[{sel = SelectFirst[Notebooks[], CurrentValue[#, {TaggingRules, "Dialog"}] === "ChatSettings" &], sn},
		If[Head@sel === NotebookObject,
			SetSelectedNotebook@sel, 
			FrontEnd`MoveCursorToInputField[#, "screenname"]&[
			  ChatTools`Private`dialog = NotebookPut[settingsdialog[StringQ@FirstCase[#["URL"] & /@ ChannelListeners[],
			  							a_String /; StringMatchQ[a, StringExpression[__, "/", $WolframID, "/", "ChatInvitations"]]]], Evaluator->"Local"]];
			If[Not@StringQ@FirstCase[#["URL"] & /@ ChannelListeners[], a_String /; StringMatchQ[a, StringExpression[__, "/", $WolframID, "/", "ChatInvitations"]]],
				RunSettingsTask[ChatTools`Private`dialog]]]];
			
ChatSettingsLoginMessage[]:= 
	NotebookPut[Notebook[{Cell["", CellMargins -> {{Automatic, Automatic}, {1, 1}}],
				Cell["Connect to Cloud", "EnableChatTitleText"], 
				Cell["Chat functionality requires connecting to the Wolfram Cloud.", "EnableChatText"], 
				Cell[BoxData[GridBox[{{ButtonPairSequence[ButtonBox[StyleBox["Cancel", FontColor -> GrayLevel[0]], ButtonFunction :> DialogReturn[],
									Appearance -> {"ButtonType" -> "Cancel", "Cancel" -> None}, Background -> GrayLevel[.9], Evaluator -> Automatic, 
									Method -> "Preemptive", ImageSize -> {70, 25}], 
							ButtonBox[StyleBox["   Connect   ", FontColor -> GrayLevel[1]],
								ButtonFunction :> (CloudConnect[]; If[TrueQ[$CloudConnected],
													DialogReturn[];
													auxStartChatChannelListener[],
													DialogReturn[]]), 
								Appearance -> {"ButtonType" -> "Default", "Default" -> FrontEnd`ToFileName[{"Toolbars", "ChatTools"}, "SendButton-Default.9.png"], 
						"Hover" -> FrontEnd`ToFileName[{"Toolbars", "ChatTools"}, "SendButton-Hover.9.png"]}, FontColor -> GrayLevel[1], Background -> RGBColor[0., 0.5548332951857786, 1.], 
								Evaluator -> "ChatServices", Method -> "Queued"]]}}, 
							GridBoxAlignment -> {"Columns" -> {{Right}}}, GridBoxSpacings -> {"Columns" -> {{1}}}]], "EnableChatButtons"]}, WindowSize -> {520, 160}, 
				ShowCellBracket -> False,
				"CellInsertionPointCell" -> {}, 
				"BlinkingCellInsertionPoint" -> False, 
				"CellInsertionPointColor" -> GrayLevel[1], 
				WindowFrame -> "ModalDialog",
				WindowElements -> {}, 
				WindowFrameElements -> {"CloseBox"},
				ShowStringCharacters -> False,
				Background -> GrayLevel[1], 
				ScrollingOptions -> {"PagewiseScrolling" -> False, "PagewiseDisplay" -> True, "VerticalScrollRange" -> Fit}, 
				CellMargins -> {{0, 0}, {0, 0}}, 
				AutoMultiplicationSymbol -> False,
				Saveable -> False, 
				WindowTitle -> "Wolfram Chat",
				Editable -> False, 
				Selectable -> False, 
				StyleDefinitions -> Notebook[{Cell[StyleData["EnableChatTitleText"], FontSize -> 20, FontFamily -> "Source Sans Pro", FontColor -> RGBColor[.2, .2, .2],
									ShowCellBracket -> False, CellMargins -> {{30, 30}, {2, 14}}], 
								Cell[StyleData["EnableChatText"], FontSize -> 12, FontFamily -> "Source Sans Pro", FontColor -> RGBColor[.39215, .39215, .39215], 
									ShowCellBracket -> False, CellMargins -> {{30, 30}, {2, 14}}], 
								Cell[StyleData["EnableChatButtons"], TextAlignment -> Right, CellMargins -> {{30, 30}, {2, 15}}, 
										ButtonBoxOptions -> {ImageSize -> {80, 24}, BaseStyle -> {FontFamily -> "Source Sans Pro", FontSize -> 14}}]}], 
				NotebookEventActions -> {"ReturnKeyDown" :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
							{"MenuCommand", "EvaluateCells"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
							{"MenuCommand", "HandleShiftReturn"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
							{"MenuCommand", "EvaluateNextCell"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]], 
							"EscapeKeyDown" :> (FE`Evaluate[FEPrivate`FindAndClickCancelButton[]]; DialogReturn[$Failed]),
							"WindowClose" :> (FE`Evaluate[FEPrivate`FindAndClickCancelButton[]]; DialogReturn[$Failed])}, Evaluator -> "ChatServices"],
			Evaluator -> "ChatServices"];
	
StartChatChannelListener[] := If[Not@$CloudConnected, ChatSettingsLoginMessage[], auxStartChatChannelListener[]];
	
(*************************** Utilities **************************************************************)

Options[CenteredCell] = {"ImageSize" -> 740, "CellMargins" -> {{0, 0}, {0, 0}}, "ImageMargins" -> 0, "Alignment" -> Center, "CacheGraphics" -> Automatic};

CenteredCell[expr_, func_, {funcopts___}, style_, {cellopts___}, opts___] := 
	Module[{isize = "ImageSize" /. {opts} /. Options[CenteredCell], cmargins = "CellMargins" /. {opts} /. Options[CenteredCell], 
		imargins = "ImageMargins" /. {opts} /. Options[CenteredCell], alignment1 = "Alignment" /. {opts} /. Options[CenteredCell], 
		cachegraphics1 = "CacheGraphics" /. {opts} /. Options[CenteredCell]}, 
		Cell[BoxData[ToBoxes[func[expr, funcopts, Alignment -> alignment1, BaseStyle -> {TextAlignment -> Center}, ImageMargins -> imargins, ImageSize -> isize]]], 
			If[StringQ[style], style, Unevaluated[Sequence[]]], cellopts, CellMargins -> cmargins, CacheGraphics -> cachegraphics1, ShowCellBracket -> False, TextAlignment -> Center, 
			Deployed -> True]];

FullWidthCenteredCell[expr_, func_, {funcopts___}, style_, {cellopts___}] := 
	Cell[BoxData[ToBoxes[func[Pane[expr, Alignment -> Left, BaseStyle -> {TextAlignment -> Left}, ImageSize -> 740], funcopts, Alignment -> Center, 
			BaseStyle -> {TextAlignment -> Left}, ImageSize -> Full]]], 
			If[StringQ[style], style, Unevaluated[Sequence[]]], cellopts, CellMargins -> {{0, -5}, {-5, -1}}, ShowCellBracket -> False, TextAlignment -> Center, Deployed -> True];

Options[CellToCenteredCell] = {"ImageSize" -> 740, "CellMargins" -> {{0, 0}, {0, 0}}, "ImageMargins" -> 0, "Alignment" -> Center};

CellToCenteredCell[Cell[BoxData[boxes_], style_String, cellopts___], opts___] := 
	Module[{isize = "ImageSize" /. {opts} /. Options[CenteredCell], cmargins = "CellMargins" /. {opts} /. Options[CenteredCell], imargins = "ImageMargins" /. {opts} /. Options[CellToCenteredCell],
		alignment1 = "Alignment" /. {opts} /. Options[CenteredCell]}, 
		CenteredCell[RawBoxes[boxes], Pane, {}, style, {cellopts, Background -> White}, "ImageSize" -> isize, "ImageMargins" -> imargins, "CellMargins" -> cmargins, "Alignment" -> alignment1]];

CellToCenteredCell[other_, opts___] := 
	Module[{isize = "ImageSize" /. {opts} /. Options[CenteredCell], cmargins = "CellMargins" /. {opts} /. Options[CenteredCell], imargins = "ImageMargins" /. {opts} /. Options[CellToCenteredCell],
		alignment1 = "Alignment" /. {opts} /. Options[CenteredCell]}, 
		CenteredCell[RawBoxes[other], Pane, {}, None, {}, "ImageSize" -> isize, "ImageMargins" -> imargins, "CellMargins" -> cmargins, "Alignment" -> alignment1]];

SetNBEvaluator[nbexpr_Notebook] :=
  Module[{nb = nbexpr},If[Quiet[Check[Options[nb, Evaluator], $Failed]]===$Failed,
		  nb = Append[ReplaceAll[nb, r_[Evaluator, _]:>r[Evaluator,"ChatServices"]], Evaluator->"ChatServices"],
		  nb = Notebook[nb[[1]],Sequence@@ReplaceAll[Rest[nb], r_[Evaluator, _] :> r[Evaluator, "ChatServices"]]]];
		  nb];
		  
SetNBEvaluator[expr_ /; (Head@expr =!= Notebook)] := expr /. Cell[a__] :> Append[DeleteCases[Cell[a], Evaluator -> _], Evaluator -> "ChatServices"];

Attributes[DynamicSendButtonCreator] = {HoldFirst};

DynamicSendButtonCreator[inputNotebook_] := 
	Dynamic[Refresh[Module[{(*inputNotebook = InputNotebook[], *)unsentCells, newcellobj}, 
				If[CurrentValue[inputNotebook, {TaggingRules, "AllowAttachSendButtons"}] === Inherited, 
					unsentCells = Select[Cells[inputNotebook], And[CurrentValue[#, {TaggingRules, "CellStatus"}] =!= "Sent",
				Not@MatchQ[Developer`CellInformation[#], {___, "Style" -> Alternatives[("Stem" | "CellLabelOptions" | "CellUserLabelOptions" | "More" | "HelpText"),
													{_, Alternatives@@Join[Join[$sentStyles, # <> "Top" & /@ $sentStyles],
																{"GrayLight"}, 
																"GrayLight" <> # <> "Top" & /@ $sentStyles]}], ___}]]&];
					If[Or[unsentCells === {}, 
						CurrentValue[inputNotebook, {TaggingRules, "ChatNotebook"}] =!= "True", 
						MatchQ[Developer`CellInformation[unsentCells], {___, {___, "Formatted" -> False, ___}, ___}]],
						
						If[ListQ[#], If[Head@# === CellObject, NotebookDelete[#, AutoScroll -> (!CurrentValue[inputNotebook, {TaggingRules, "ScrollLock"}, False])]] & /@ #] &[CurrentValue[inputNotebook, {TaggingRules, "SelectedCells"}]],
						
						With[{cell = Part[unsentCells, -1]}, 
							If[CurrentValue[cell, {TaggingRules, "CellStatus"}] =!= "Sent",
								If[ParentCell[#] =!= cell, NotebookDelete[#, AutoScroll -> (!CurrentValue[inputNotebook, {TaggingRules, "ScrollLock"}, False])]] & /@ CurrentValue[inputNotebook, {TaggingRules, "SelectedCells"}];
								If[Intersection[Map[ParentCell, CurrentValue[inputNotebook, {TaggingRules, "SelectedCells"}]], unsentCells] === {}, 
									newcellobj = FrontEndExecute[FrontEnd`AttachCell[cell, 
											Cell[BoxData[TooltipBox[TagBox[ButtonBox[
											  StyleBox["Send", FontFamily->"Source Sans Pro", FontWeight->"Bold", FontColor->GrayLevel[1],
											    FontSize->13], 
											      Appearance->{"Default"->FrontEnd`ToFileName[{"Toolbars","ChatTools"}, "SendButton-Default.9.png"],
											        "Hover" -> FrontEnd`ToFileName[{"Toolbars","ChatTools"}, "SendButton-Hover.9.png"]},
															ButtonFunction:>(SetOptions[cell, TaggingRules -> {"CellStatus" -> "Sent"}];
															NotebookDelete[EvaluationCell[], AutoScroll -> (!CurrentValue[inputNotebook, {TaggingRules, "ScrollLock"}, False])];
															Needs["ChatTools`"];
															ChatTools`SendChatNotebookInformation[]), 
															Method -> "Queued", ImageSize->{65, 22},
															Evaluator->"ChatServices"],
													  MouseAppearanceTag["LinkHand"]], 
													Cell[TextData[{"Use ",
													Cell[BoxData[TemplateBox[{"alt","enter"},"Key1",BaseStyle->{FontFamily:>CurrentValue["PanelFontFamily"]}]]],
													" to send message."}], FontFamily:>CurrentValue["PanelFontFamily"]], 
												  TooltipStyle->"SendTooltipStyle"]], 
												CellElementSpacings -> {"CellMinHeight" -> 1}],
															{Offset[{-24, 12}, 0], {"CellBracket", Bottom}}]];
									AppendTo[CurrentValue[inputNotebook, {TaggingRules, "SelectedCells"}], newcellobj]]]]]]], UpdateInterval -> 0.4],Evaluator->"ChatServices"];
									
SaveChatNotebook[nb_NotebookObject] := 
	Module[{savedir = CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "SaveDirectory"}], savepath, filename, ni,
		characterstoexclude = {"#", "%", "&", "{", "}", "\\", "<", ">", "*", "?", "/", "$", "!", "'", "\"", ":", "@"}}, 
		filename = StringReplace[StringReplace[If[MatchQ[ni = FirstCase[NotebookInformation@nb, _["FileName", FrontEnd`FileName[{__}, a_, ___]] :> a], Missing["NotFound"]], 
						StringReplace[StringReplace[AbsoluteCurrentValue[nb, WindowTitle], (a__ ~~ "'s Chat - " ~~ "(" ~~ b__ ~~ ")") :> (a ~~ "'s Chat - " ~~ b)],
						a__ ~~ ".nb" ~~ EndOfString :> a] <> ".nb", ni], # -> "" & /@ characterstoexclude], (" " ..) -> ""]; 
		savepath = FileNameJoin[{savedir, filename}]; 
		NotebookSave[nb, savepath, Interactive -> True]];
									
hourMinutes[stringDateObject_]:=
	Module[{stringDateObject1},
	StringJoin @@ Riffle[If[StringCases[stringDateObject, "TimeObject"] === {},
				stringDateObject1 = ToString[TimeZoneConvert[DateObject[#[[1]], TimeZone -> #[[-1]]] &[ToExpression@stringDateObject], $TimeZone]];
				StringCases[stringDateObject1, "{" ~~ __ ~~ ", " ~~ __ ~~ ", " ~~ __ ~~ ", " ~~ a__ ~~ ", " ~~ b__ ~~ ", " ~~ c__ ~~ "}" :> {If[StringLength@a === 1, "0" <> a, a], 
																If[StringLength@StringTrim@b === 1, "0" <> StringTrim@b, StringTrim@b]}], 
				StringCases[stringDateObject, "TimeObject[{" ~~ a__ ~~ "," ~~ b__ ~~ "," ~~ c__ ~~ "}" :> {If[StringLength@a === 1, "0" <> a, a], 
														If[StringLength@StringTrim@b === 1, "0" <> StringTrim@b, StringTrim@b]}]][[1]], ":"]];
														
hourMinutesSecondsHundredths[stringDateObject_] := 
	Module[{stringDateObject1 = ToString[TimeZoneConvert[DateObject[#[[1]], TimeZone -> #[[-1]]] &[ToExpression@stringDateObject], $TimeZone]]}, 
	StringJoin @@ Riffle[StringCases[stringDateObject1, "{" ~~ __ ~~ ", " ~~ __ ~~ ", " ~~ __ ~~ ", " ~~ a__ ~~ ", " ~~ b__ ~~ ", " ~~ c__ ~~ "}" :> {If[StringLength@a === 1, "0" <> a, a], 
			If[StringLength@StringTrim@b === 1, "0" <> StringTrim@b, StringTrim@b], 
			MapAt[If[StringLength@StringTrim@# === 2, StringTrim@# <> "0", StringTrim@#]&@StringReplace[ToString@#, "0." -> "."] &, 
				If[MatchQ[#, {_, 1.`}], {ToString[ToExpression[#[[1]]] + 1], 0.}, #] &@MapAt[Round[ToExpression["." <> #], .01] &, 
					MapAt[If[StringLength@StringTrim@# === 1, "0" <> StringTrim@#, StringTrim@#] &, 
						StringSplit[If[StringMatchQ[c, __ ~~ "."], c <> "001", c], "."], 1], 2], 2] /. {d_String, e_} :> d <> If[e === ".", ".00", e]}][[1]], ":"]];
			
NotebookCorrespondingToID[tag_, id_]:=SelectFirst[Notebooks[], CurrentValue[#, {TaggingRules, tag}] === id &];

WolframIDandTimeCell[rectangleColor_, wolframid_, time_] :=
	Module[{widsplit = StringSplit[wolframid, "@"]},
	Cell[BoxData[GridBox[{{ItemBox[GridBox[{{TemplateBox[{(rectangleColor/.Thread[Rule[$sentStyles, $mainColors]])}, "ChatColorSwatch"],
						If[StringFreeQ[wolframid, "@"],
								wolframid,
								RowBox[{widsplit[[1]], AdjustmentBox[StyleBox["@", FontSize -> 10], BoxBaselineShift -> 0], widsplit[[2]]}]]}},
						GridBoxAlignment -> Center, 
						GridBoxSpacings -> {"ColumnsIndexed" -> {2 -> .5}, "Rows" -> {{Automatic}}}], Alignment -> Left], 
				ItemBox[StyleBox[time, "TimestampText"], Alignment -> Right]}}, 
				GridBoxItemSize -> {"Columns" -> {{Scaled[0.5]}}}, BaseStyle -> "CellLabelText"]],
		"CellLabelOptions"]];

UserWolframIDandTimeCell[rectangleColor_, wolframid_, time_] := 
	Module[{widsplit = StringSplit[wolframid, "@"], colorRules = Thread[Rule[$sentStyles, $mainColors]]}, 
		Cell[BoxData[GridBox[{{ItemBox[FrameBox[If[StringFreeQ[wolframid, "@"],
								wolframid,
								RowBox[{widsplit[[1]], AdjustmentBox[StyleBox["@", FontSize -> 10], BoxBaselineShift -> 0], widsplit[[2]]}]],
						Background -> (rectangleColor/.colorRules), FrameMargins -> {{6, 6}, {6, 6}}, BoxFrame -> 0], 
				Alignment -> Left, BaseStyle -> "CellUserLabelText"], ItemBox[StyleBox[time, "TimestampText"], Alignment -> Right]}}, GridBoxItemSize -> {"Columns" -> {{Scaled[0.5]}}}, 
				BaseStyle -> "CellLabelText"]], "CellUserLabelOptions"]];
				
TimeZoneAdjust[stringDateObject_] := ToString[TimeZoneConvert[DateObject[#[[1]], TimeZone -> #[[-1]]] &[ToExpression@stringDateObject], $TimeZone]];

dateTime[stringDateObject_] := 
	Module[{stringDateObject1 = ToString[TimeZoneConvert[DateObject[#[[1]], TimeZone -> #[[-1]]] &[ToExpression@stringDateObject], $TimeZone]], s}, 
		s = StringCases[stringDateObject1, "{" ~~ a__ ~~ ", " ~~ b__ ~~ ", " ~~ c__ ~~ ", " ~~ d__ ~~ ", " ~~ e__ ~~ ", " ~~ f__ ~~ "}" :> {c, b, a, 
						If[StringLength@StringTrim@d === 1, "0" <> StringTrim@d, StringTrim@d], If[StringLength@StringTrim@e === 1, "0" <> StringTrim@e, StringTrim@e], 
						If[StringLength@# === 1, "0" <> #, #] &[ToString@Round@ToExpression@f]}][[1]]; 
		MapAt[# /. {"1" -> "Jan", "2" -> "Feb", "3" -> "Mar", "4" -> "Apr", "5" -> "May", "6" -> "Jun", "7" -> "Jul", "8" -> "Aug", "9" -> "Sep", "10" -> "Oct", "11" -> "Nov", "12" -> "Dec"} &, 
			s, 2] /. {a_, b_, c_, d_, e_, f_} :> a <> " " <> b <> " " <> c <> " " <> d <> ":" <> e <> ":" <> f];
			
RemoveContacts[list_ /; VectorQ[list, StringQ]] := (CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}] = Complement[CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}], list]);

(**************************************** Chat ******************************************************)

$stemCellTemplate = Cell[BoxData[GridBox[{
{
ItemBox["",ItemSize->1],
RowBox[{Cell["New cell: ", "StemText"],
ButtonBox[
          StyleBox[
           "Text", "StemButtonText"], BaseStyle -> 
          "StemButton", ButtonData->"Text"], "\[NegativeVeryThinSpace]",
ButtonBox[
          StyleBox[
           "Input", "StemButtonText"], BaseStyle -> 
          "StemButton2", ButtonData->"Input"]}],
        ItemBox["",ItemSize->Fit],
TooltipBox[
          TagBox[
           ButtonBox[DynamicBox[FEPrivate`Which[FEPrivate`And[FEPrivate`SameQ[CurrentValue[EvaluationNotebook[], {TaggingRules, "Percolate"}], True],
           							FEPrivate`SameQ[CurrentValue[$FrontEnd, "AllowChatServices"], True]], 
						InterpretationBox[DynamicBox[FEPrivate`FrontEndResource["FEExpressions", "PercolateAnimator"][Medium], ImageSizeCache -> {50., {2., 10.}}], 
									ProgressIndicator[Appearance -> "Percolate"], BaseStyle -> {"Deploy"}], 
						FEPrivate`Not[FEPrivate`SameQ[CurrentValue[$FrontEnd, "AllowChatServices"], True]], StyleBox["\[FilledSquare]", "SendButtonText"],
						True, 
						StyleBox["Send", "SendButtonText"]]], 
			ButtonFunction :> Module[{bn = ButtonNotebook[], sentStyles = {"Turquoise", "OffGreen", "Orange", "OffBlue", "Green", "Purple"}}, 
						If[Select[Cells[bn], 
					FreeQ["Style" /. Developer`CellInformation@#, Alternatives @@ Join[Join[sentStyles, # <> "Top" & /@ sentStyles], {"GrayLight", 
						"CellLabelOptions", "CellUserLabelOptions", "Stem", "More", "HelpText"}, "GrayLight" <> # <> "Top" & /@ sentStyles]] &] === {}, 
						CurrentValue[bn, {TaggingRules, "Percolate"}] = False;
						MessageDialog["There are no cells to send.", WindowFrame -> "ModalDialog", NotebookEventActions -> {{"MenuCommand", "SimilarCellBelow"} :> Null}],
						If[TrueQ@CurrentValue[bn, {TaggingRules, "BeforeSend"}], CurrentValue[bn, {TaggingRules, "Percolate"}] = True];
						ChatTools`SendChatNotebookInformation[bn];
						CurrentValue[bn, {TaggingRules, "Percolate"}] = False]], 
			BaseStyle -> "SendButton", 
			Enabled -> Dynamic[TrueQ@CurrentValue[$FrontEnd, "AllowChatServices"]]], 
           MouseAppearanceTag["LinkHand"]], 
          Cell[
           TextData[{"With your cursor in an unsent cell,\nuse ", 
             Cell[
              BoxData[
               
               TemplateBox[{"alt", "enter"}, "Key1", 
                BaseStyle -> {FontFamily :> CurrentValue["PanelFontFamily"]}]]], " to send message."}], 
           FontFamily :> CurrentValue["PanelFontFamily"]], TooltipStyle -> "SendTooltipStyle"]}}]], "Stem"];
		
Options[SendChatNotebookInformation] = {"Conclude" -> False, "ChatNotebookID" -> None};

SendChatNotebookInformation[opts___?OptionQ]:= SendChatNotebookInformation[ButtonNotebook[], opts];

SendChatNotebookInformation[bn_, opts___?OptionQ] := 
	Module[{conclude = ("Conclude" /. {opts} /. Options[SendChatNotebookInformation]), id2, beforesend, chatroom, originalWolframID, listenerterminated, viewparticipants, manageparticipants, st,
		id = ("ChatNotebookID" /. {opts} /. Options[SendChatNotebookInformation]), currentListenerIDs, originator, screenname, removedparticipant, screenname1, cells, sentStyles = $sentStyles,
		alias, li},
	If[Not@TrueQ@CurrentValue[$FrontEnd, "WolframCloudConnected"],
	
	CurrentValue[bn, {TaggingRules, "Percolate"}] = False;
	FirstLoginMessage[],
	
	If[TrueQ@conclude, CurrentValue[bn, {TaggingRules, "Terminated"}] = True];
	id2 = CurrentValue[bn, {TaggingRules, "ChatNotebookID"}];
	beforesend = CurrentValue[bn, {TaggingRules, "BeforeSend"}];
	chatroom = CurrentValue[bn, {TaggingRules, "ChatRoom"}];
	originalWolframID = CurrentValue[bn, {TaggingRules, "OriginalWolframID"}];
	If[(Not@TrueQ@chatroom && Not@TrueQ@beforesend) || TrueQ@chatroom,
		listenerterminated = Not@MemberQ[StringReplace[#["URL"], __ ~~ "/" ~~ (a_ /; StringFreeQ[a, $PathnameSeparator]) :> a] & /@ ChannelListeners[], id2]];
	If[And[(Not@TrueQ@chatroom && Not@TrueQ@beforesend) || TrueQ@chatroom,
		TrueQ@listenerterminated],
		CurrentValue[bn, {TaggingRules, "Terminated"}] = True];
	viewparticipants = CurrentValue[bn, {TaggingRules, "ViewParticipants"}];
	manageparticipants = CurrentValue[bn, {TaggingRules, "ManageParticipants"}];
		
	Which[TrueQ@viewparticipants,
	
		MessageDialog["You cannot send messages while viewing participants.", WindowFrame -> "ModalDialog", WindowSize -> {400, All}],
		
		TrueQ@manageparticipants,
		
		MessageDialog["You cannot send messages while managing participants.", WindowFrame -> "ModalDialog", WindowSize -> {400, All}],
		
		TrueQ@chatroom && (originalWolframID =!= $WolframID),
		
		MessageDialog["Your Wolfram ID has changed. This terminates the room's channel listener. Try clicking the Send button again.", WindowFrame -> "ModalDialog", WindowSize -> {700, All}];
		ChannelListen[ChannelObject["chatframework@wolfram.com" <> ":" <> id2], "TrustedChannel" -> True];
		CurrentValue[bn, {TaggingRules, "Terminated"}] = Inherited;
		CurrentValue[bn, {TaggingRules, "OriginalWolframID"}] = $WolframID,
		
		TrueQ@chatroom && TrueQ@listenerterminated,
		
		MessageDialog["The channel listener for this room has been terminated and is being restarted. Try clicking the Send button again.", WindowFrame -> "ModalDialog", WindowSize -> {680, All}];
		ChannelListen[ChannelObject["chatframework@wolfram.com" <> ":" <> id2], "TrustedChannel" -> True];
		CurrentValue[bn, {TaggingRules, "Terminated"}] = Inherited
		(* Restarting the listener and doing the send in one body of code did not work.
		Pause[.3];
		Internal`WithLocalSettings[elems = AbsoluteCurrentValue[bn, WindowFrameElements];
		
						CurrentValue[bn, WindowFrameElements] = DeleteCases[elems, "CloseBox"],
						
						SendChatCells[bn],
						
						CurrentValue[bn, WindowFrameElements] = elems]*),
						
		And[Not@TrueQ@chatroom, 
			Not@TrueQ@beforesend, 
			TrueQ@listenerterminated],
		
		MessageDialog["The channel listener for this chat has been terminated and is being restarted. Try clicking the Send button again.", WindowFrame -> "ModalDialog", WindowSize -> {680, All}];
		ChannelListen[ChannelObject["chatframework@wolfram.com" <> ":" <> id2], "TrustedChannel" -> True];
		CurrentValue[bn, {TaggingRules, "Terminated"}] = Inherited,
						
		(*And[Not@TrueQ@chatroom, 
			Not@TrueQ@beforesend, 
			TrueQ@listenerterminated],
			
		Internal`WithLocalSettings[elems = AbsoluteCurrentValue[bn, WindowFrameElements];
				
						CurrentValue[bn, WindowFrameElements] = DeleteCases[elems, "CloseBox"],
		
						CurrentValue[bn, DockedCells] = {};
						st = Cells[bn, CellStyle -> "Stem"];
						If[st =!= {},
							SetOptions[st[[-1]], Deletable -> True];
							NotebookWrite[st[[1]], Cell["The channel listener for this chat has been terminated.", "Text"],
									AutoScroll -> (!CurrentValue[bn, {TaggingRules, "ScrollLock"}, False])]];
						originator = CurrentValue[bn, {TaggingRules, "Originator"}];
						screenname = CurrentValue[bn, {TaggingRules, "ScreenName"}];
						removedparticipant = CurrentValue[bn, {TaggingRules, "OriginalWolframID"}];
						screenname1 = Compress@If[MemberQ[{"", Inherited}, screenname] || (StringQ@screenname && StringMatchQ[screenname, Whitespace]), "None", screenname];
						URLExecute[CloudObject["https://www.wolframcloud.com/objects/346cb1d3-8ba8-45a4-8baf-417a16f3b6fd"],
								{"id" -> id2, "removedparticipant" -> removedparticipant, "screenname" -> screenname1}, "WL", Method -> "POST"],
						
						CurrentValue[bn, WindowFrameElements] = elems],*)
		
		True,

		Which[TrueQ@CurrentValue[bn, {TaggingRules, "BeforeSend"}] && conclude === True,
		
			NotebookClose[bn],
		
			cells === {} && conclude === True,
		
			Internal`WithLocalSettings[elems = AbsoluteCurrentValue[bn, WindowFrameElements];
			
						CurrentValue[bn, WindowFrameElements] = DeleteCases[elems, "CloseBox"],
				
						id = CurrentValue[bn, {TaggingRules, "ChatNotebookID"}];
						originator = CurrentValue[bn, {TaggingRules, "Originator"}];
						alias = CurrentValue[bn, {TaggingRules, "Alias"}];
						screenname = CurrentValue[bn, {TaggingRules, "ScreenName"}];
						li = Association @@ List["id" -> id, "removedparticipant" -> $WolframID,
							"screenname" -> Compress@If[MemberQ[{"", Inherited}, screenname] || (StringQ@screenname && StringMatchQ[screenname, Whitespace]), "None", screenname],
							"celllist" -> Compress@{}];
						Quiet[ChannelSend[StringJoin["chatframework@wolfram.com", ":", id], li, ChannelPreSendFunction -> None], ChannelObject::uauth],
  							
  						NotebookClose[bn]],
  				
  			cells =!= {} && conclude === True,
  			
  			Internal`WithLocalSettings[elems = AbsoluteCurrentValue[bn, WindowFrameElements];
						
						CurrentValue[bn, WindowFrameElements] = DeleteCases[elems, "CloseBox"],
							
						id = CurrentValue[bn, {TaggingRules, "ChatNotebookID"}];
						originator = CurrentValue[bn, {TaggingRules, "Originator"}];
						alias = CurrentValue[bn, {TaggingRules, "Alias"}];
						screenname = CurrentValue[bn, {TaggingRules, "ScreenName"}];
						li = Association @@ List["id" -> id, "removedparticipant" -> $WolframID,
							"screenname" -> Compress@If[MemberQ[{"", Inherited}, screenname] || (StringQ@screenname && StringMatchQ[screenname, Whitespace]), "None", screenname],
							"celllist" -> Compress@{}];
						Quiet[ChannelSend[StringJoin["chatframework@wolfram.com", ":", id], li, ChannelPreSendFunction -> None], ChannelObject::uauth],
			  				
  						NotebookClose[bn]],
  							
  			True,
			
			Internal`WithLocalSettings[elems = AbsoluteCurrentValue[bn, WindowFrameElements];
		
						CurrentValue[bn, WindowFrameElements] = DeleteCases[elems, "CloseBox"],
						
						SendChatCells[bn],
						
						CurrentValue[bn, WindowFrameElements] = elems]]]]];
(*				
$backgroundColors = (Lighter@Lighter[Lighter[#]]&/@$mainColors)
*)

$mainColors = {RGBColor[0.6, 0.757, 0.99], RGBColor[0.6, 0.9059, 0.812], RGBColor[0.757, 0.671, 0.827], RGBColor[0.969, 0.89, 0.502], RGBColor[0.976, 0.74, 0.549], RGBColor[1., 0.549, 0.549]};

$backgroundColors = {RGBColor[0.881, 0.928, 0.997], RGBColor[0.88, 0.97, 0.944], RGBColor[0.928, 0.9, 0.9], RGBColor[0.99, 0.967, 0.85], RGBColor[0.993, 0.923, 0.866], RGBColor[1., 0.866, 0.866]};

$sentStyles = {"Turquoise", "OffGreen", "Orange", "OffBlue", "Green", "Purple"};

AddPasteButton[cell_Cell, backgroundcolor_] := cell;

Options[SendChatCells] = {"Conclude" -> False};

SendChatCells[opts___?OptionQ]:=SendChatCells[InputNotebook[], opts];
	
SendChatCells[bn_, opts___?OptionQ] := 
	Module[{id, originator, participants, date, windowtitle, allparticipants, pos, styleToUse, outputCellIDsNotSent, sentStyles = $sentStyles, cells, cellIDs, timestamp, type,
		screenname, alias, conclude = ("Conclude"/.{opts}/.Options[SendChatCells]), readCellList, st, fs, re, i, i2, i31, compressedCellList, chatVisibility, li, a, chatroom, banner,
		name, teachersadded, shortcut, chatCreationDate, CustomWindowTitle, sel, ModifyNotebookIndexAPIUUID, DeleteNotebookCellDataAndAPIsUUID},
		
	id = CurrentValue[bn, {TaggingRules, "ChatNotebookID"}];
	originator = CurrentValue[bn, {TaggingRules, "Originator"}];
			
	If[And[originator =!= $WolframID,
		Not@TrueQ@CurrentValue[bn, {TaggingRules, "BeforeSend"}],
		(* Need to be sure the following returns before going forward in the code. *)
		TrueQ@CurrentValue[bn, {TaggingRules, "ParticipantRemoved"}]],
			
		MessageDialog["You have been removed from this thread.", WindowFrame -> "ModalDialog"],
		
		If[Cases[Developer`CellInformation[Cells[bn]], {___, "Formatted" -> False, ___}] =!= {},
			FrontEndExecute[{FrontEndToken[bn, "SelectAll"]}];
			FrontEndExecute[{FrontEndToken[bn, "ToggleShowExpression"]}]];
		
		CurrentValue[bn, {TaggingRules, "AllowAttachSendButtons"}] = False;
		If[ListQ[#], If[Head@# === CellObject, NotebookDelete[#, AutoScroll -> (!CurrentValue[bn, {TaggingRules, "ScrollLock"}, False])]] & /@ #] &[CurrentValue[bn, {TaggingRules, "SelectedCells"}]];
		
		participants = CurrentValue[bn, {TaggingRules, "Participants"}];
		date = CurrentValue[bn, {TaggingRules, "ChatNotebookDate"}];
		windowtitle = CurrentValue[bn, {TaggingRules, "ChatNotebookWindowTitle"}];
		
		alias = CurrentValue[bn, {TaggingRules, "Alias"}];
		(* User has changed to another wolfram id. *)
		If[Not@MemberQ[#, $WolframID],
			CurrentValue[bn, {TaggingRules, "AllParticipants"}] = Append[#, $WolframID]]&[CurrentValue[bn, {TaggingRules, "AllParticipants"}]];
		allparticipants = CurrentValue[bn, {TaggingRules, "AllParticipants"}];
		pos = Position[allparticipants, If[alias === Inherited, $WolframID, alias]][[1, 1]];
		styleToUse = sentStyles[[If[# === 0, -1, #] &[Mod[pos, Length@sentStyles]]]];
		
		outputCellIDsNotSent = (CurrentValue[#, CellID] & /@ Select[Cells[bn, CellStyle -> "Output"], CurrentValue[#, Editable] === True &]);
		(NotebookFind[bn, #, All, CellID, AutoScroll -> False]; FrontEndExecute[{FrontEndToken[bn, "SelectionConvert", "BitmapConditional"]}]) & /@ outputCellIDsNotSent;
		
		Quiet[cells = Select[Cells[bn], FreeQ["Style" /. Developer`CellInformation@#, Alternatives@@Join[Join[sentStyles,# <> "Top" & /@ sentStyles],
														{"GrayLight", "CellLabelOptions", "CellUserLabelOptions", "Stem", "More", "HelpText"},
														"GrayLight" <> # <> "Top" & /@ sentStyles]] &], {ReplaceAll::reps}];
		If[(Length@cells > 1) && (NotebookRead[cells[[1]]] === $Failed), cells = Rest[cells]];

	If[Total[ByteCount /@ (NotebookRead /@ cells)] > 5000000,
	
		MessageDialog["The cells being sent cannot be more than five million bytes altogether.", WindowFrame -> "ModalDialog", WindowSize -> {450, All}],
		
		cellIDs = CurrentValue[#, CellID] &@cells;
		timestamp = ToString@Now;
		
		If[NotebookFind[bn, "HelpText", All, CellStyle, AutoScroll -> (!CurrentValue[bn, {TaggingRules, "ScrollLock"}, False])] =!= $Failed,
			CurrentValue[NotebookSelection[bn], Editable] = True;
			CurrentValue[NotebookSelection[bn], Deletable] = True;
			NotebookDelete[bn, AutoScroll -> (!CurrentValue[bn, {TaggingRules, "ScrollLock"}, False])]];
		
		type = CurrentValue[bn, {TaggingRules, "RoomType"}];
		screenname = CurrentValue[bn, {TaggingRules, "ScreenName"}];
		If[Not@StringQ@screenname, screenname = $WolframID];
		(*
		Do[NotebookFind[bn, cellIDs[[i]], All, CellID, AutoScroll -> False];
			
				If[MatchQ[Developer`CellInformation[cells[[i]]], {"Style" -> "Input", __}],

					NotebookWrite[bn, If[MatchQ[#, Cell[a_ /; Not@FreeQ[a, FrameBox[{__}, ___]], "Input", ___]],
								#/.FrameBox[{b__}, o___] :> FrameBox[RowBox[{b}], o],
								#]&[AddPasteButton[NotebookRead[bn], $backgroundColors[[If[# === 0, -1, #] &[Mod[pos, Length@$backgroundColors]]]]]], All,
								AutoScroll -> (!CurrentValue[bn, {TaggingRules, "ScrollLock"}, False])];

					cells = ReplacePart[cells, i -> (SelectedCells[bn][[1]])]];
				If[i === 1,
					SetOptions[cells[[1]], TaggingRules -> {"CellStatus" -> "Sent", "Sender" -> If[MemberQ[{"", "None"}, screenname],
															If[alias === Inherited, $WolframID, alias],
															screenname], "TimeStamp" -> timestamp}],
					SetOptions[cells[[i]], TaggingRules -> {"CellStatus" -> "Sent"}]];
				If[MatchQ[Developer`CellInformation[bn], {{___, "Style" -> {"Message", "MSG"}, __}}], 
					NotebookWrite[bn, ReplacePart[NotebookRead[bn], 3 -> Unevaluated[Sequence["MSG", styleToUse]]],
						All, AutoScroll -> (!CurrentValue[bn, {TaggingRules, "ScrollLock"}, False])];
					cells = ReplacePart[cells, i -> (SelectedCells[bn][[1]])],
					SetOptions[NotebookSelection[bn], If[i === 1, styleToUse<>"Top", styleToUse]]];
					
				If[i === 1,
					SelectionMove[bn, Before, Cell, AutoScroll -> (!CurrentValue[bn, {TaggingRules, "ScrollLock"}, False])];
					NotebookWrite[bn, UserWolframIDandTimeCell[styleToUse, If[MemberQ[{"", "None"}, screenname], If[alias === Inherited, $WolframID, alias], screenname],
								If[type === "PromptedResponse",
									hourMinutesSecondsHundredths@timestamp,
									StringReplace[StringReplace[DateString["ISODateTime"], __ ~~ "T" -> ""], a__ ~~ ":" ~~ b__ ~~ ":" ~~ __ :> a <> ":" <> b]]], All,
									AutoScroll -> (!CurrentValue[bn, {TaggingRules, "ScrollLock"}, False])]],
				
				{i, Length@cellIDs}];
		
		readCellList = NotebookRead[cells];
		readCellList = With[{screenname1 = screenname, wid1 = $WolframID, date1 = timestamp},
						Which[MatchQ[readCellList,{Cell[_,_,_,TaggingRules->{"CellStatus"->"Sent"},__],___}],
							MapAt[Function[t,t/.Cell[a_,b_,c_,TaggingRules->{"CellStatus"->"Sent"},d__]:>Cell[a,b,c<>"Top",TaggingRules->{"CellStatus"->"Sent","Sender"->If[MemberQ[{"","None"},screenname1],wid1,screenname1],"TimeStamp"->date1},d]],
									readCellList,
									{1}],
							MatchQ[readCellList,{Cell[_,_,_,d__/;Not@MemberQ[{d}, TaggingRules -> _]],___}],
							MapAt[Function[t,t/.Cell[a_,b_,c_,d__]:>Cell[a,b,c<>"Top",TaggingRules->{"CellStatus"->"Sent","Sender"->If[MemberQ[{"","None"},screenname1],wid1,screenname1],"TimeStamp"->date1},d]],
									readCellList,
									{1}],
							True,		
							readCellList]];
		
		st = Cells[bn, CellStyle -> "Stem"];
		If[st =!= {},
			fs = First[st];
			SetOptions[fs, Deletable -> True];
			NotebookDelete[fs, AutoScroll -> (!CurrentValue[bn, {TaggingRules, "ScrollLock"}, False])]];
		SelectionMove[bn, After, Notebook, AutoScroll -> (!CurrentValue[bn, {TaggingRules, "ScrollLock"}, False])];
		NotebookWrite[bn, $stemCellTemplate, AutoScroll -> (!CurrentValue[bn, {TaggingRules, "ScrollLock"}, False])];
		*)
		
		re = NotebookRead[cells];
		i = Insert[#, "Turquoise", If[MatchQ[#, Cell[_, "Message", "MSG", ___]], 4, 3]] & /@ re;
		i2 = MapAt[If[StringQ@#, # <> "Top", "TurquoiseTop"]&, i, If[MatchQ[i, Cell[_, "Message", "MSG", ___]], {1, 4}, {1, 3}]];
		i31 = Insert[i2[[1]], TaggingRules -> {"CellStatus" -> "Sent", 
							"Sender" -> If[MemberQ[{"", "None"}, screenname], If[alias === Inherited, $WolframID, alias], screenname],
							"TimeStamp" -> timestamp}, If[MatchQ[i2[[1]], Cell[_, "Message", "MSG", ___]], 5, 4]];
		readCellList = If[Length@i2 === 1, {i31}, Prepend[Insert[#, TaggingRules -> {"CellStatus" -> "Sent"}, If[MatchQ[#, Cell[_, "Message", "MSG", ___]], 5, 4]] & /@ Take[i2, {2, -1}], i31]];
		
		SelectionMove[bn, After, Notebook, AutoScroll -> (!CurrentValue[bn, {TaggingRules, "ScrollLock"}, False])];
		compressedCellList = Compress@readCellList;
		chatVisibility = "Private";
		chatroom = CurrentValue[bn, {TaggingRules, "ChatRoom"}];
		
	If[TrueQ@chatroom,
	
		banner = CurrentValue[bn, {TaggingRules, "Banner"}];
		name = CurrentValue[bn, {TaggingRules, "ChatNotebookWindowTitle"}];
		teachersadded = CurrentValue[bn, {TaggingRules, "Moderators"}];
		shortcut = CurrentValue[bn, {TaggingRules, "Shortcut"}];
		NotebookDelete[cells, AutoScroll -> (!CurrentValue[bn, {TaggingRules, "ScrollLock"}, False])];
		ChannelSend["https://channelbroker.wolframcloud.com/users/" <> "chatframework@wolfram.com" <> "/" <> id,
				Association["sender" -> $WolframID, "requesteraddress" -> "None", "allparticipants" -> Compress@allparticipants, "id" -> id, "compressedCellList" -> compressedCellList,
						"name" -> Compress@name, "alias" -> $WolframID, "originator" -> originator, "SpecialAction" -> "None", "Banner" -> banner, "RoomType" -> type,
						"Moderators" -> Compress@teachersadded, "Shortcut" -> Compress@shortcut,
						"ScreenName" -> Compress@If[MemberQ[{"", "None"}, screenname], "None", screenname]],
						ChannelPreSendFunction -> None];
		CurrentValue[bn, {TaggingRules, "AllowAttachSendButtons"}] = Inherited;
		If[CurrentValue[bn, {TaggingRules, "BeforeSend"}] === True, CurrentValue[bn, DockedCells] = ChatNotebookDockedCell[id, windowtitle, "CanSetVisibility" -> False,
											"Preferences" -> ChatClassRoomPreferencesMenu[id, shortcut,
											"Teacher" -> (($WolframID === originator) || (ListQ@teachersadded && MemberQ[teachersadded, $WolframID])),
										"CellLabelsDefault" -> If[type === "PromptedResponse", "On", "Off"]]]];
		CurrentValue[bn, {TaggingRules, "BeforeSend"}] = Inherited,
		
		Catch[If[TrueQ@conclude,
		
			NotebookDelete[cells, AutoScroll -> (!CurrentValue[bn, {TaggingRules, "ScrollLock"}, False])];
			URLExecute[CloudObject["https://www.wolframcloud.com/objects/558d160b-b9ad-408e-b425-46e71bd97401"], {"id" -> id, "removedparticipant" -> $WolframID}, "WL", Method -> "POST"]
			(* Do this instead through an api function - DeleteChannel[ChannelObject[StringJoin[originator, ":", id]]]*),
			
			chatCreationDate = CurrentValue[bn, {TaggingRules, "ChatCreationDate"}];
			CustomWindowTitle = CurrentValue[bn, {TaggingRules, "CustomWindowTitle"}];
			
			If[TrueQ@CurrentValue[bn, {TaggingRules, "BeforeSend"}],
			
				(* Create private chat channel and add entry to Chat index. *)
				
				URLExecute[CloudObject["https://www.wolframcloud.com/objects/1f92d1a6-30e8-40fb-9fe6-e0a29abaed6b"],
														{"ChatNotebookID" -> id, "wid" -> $WolframID,
															"ScreenName" -> Compress@If[MemberQ[{"", "None"}, screenname], "None", screenname],
															"date" -> timestamp, "ChatCreationDate" -> chatCreationDate,
															"cellList" -> compressedCellList, "participants" -> Compress@participants,
															"windowtitle" -> Compress@windowtitle,
															"CustomWindowTitle" -> Compress@CustomWindowTitle,
															"originator" -> originator, "allparticipants" -> Compress@allparticipants}, "WL",
														Method -> "POST"];
				{ModifyNotebookIndexAPIUUID, DeleteNotebookCellDataAndAPIsUUID} = {"9bdd1f54-92b3-4cc9-8692-1c0bf292d6a4", "f71b3c7e-597e-4c26-aad9-db602545d25a"};
				CurrentValue[bn, {TaggingRules, "ModifyNotebookIndexAPIUUID"}] = ModifyNotebookIndexAPIUUID;
				CurrentValue[bn, {TaggingRules, "DeleteNotebookCellDataAndAPIsUUID"}] = DeleteNotebookCellDataAndAPIsUUID,
								
				ModifyNotebookIndexAPIUUID = CurrentValue[bn, {TaggingRules, "ModifyNotebookIndexAPIUUID"}];
				DeleteNotebookCellDataAndAPIsUUID = CurrentValue[bn, {TaggingRules, "DeleteNotebookCellDataAndAPIsUUID"}]]];
											
			li = If[TrueQ@conclude,
					Association @@ List["id" -> id, "removedparticipant" -> $WolframID, "celllist" -> compressedCellList],
					Association @@ List["ChatNotebookID" -> id, "wid" -> $WolframID, "ScreenName" -> Compress@If[MemberQ[{"", "None"}, screenname], "None", screenname],
								"date" -> timestamp,
								"ChatCreationDate" -> chatCreationDate, "cellList" -> compressedCellList, "participants" -> Compress@participants,
								"windowtitle" -> Compress@windowtitle, "CustomWindowTitle" -> Compress@CustomWindowTitle, "originator" -> originator,
								"allparticipants" -> Compress@allparticipants]];
											
			If[chatVisibility === "Private",
				
				If[(* So this is the first send. *)TrueQ@CurrentValue[bn, {TaggingRules, "BeforeSend"}],
					
					ChannelListen[(*$WolframID*)"chatframework@wolfram.com" <> ":" <> id, "TrustedChannel" -> True];
						
					(* The private channel is sent to all of the participants besides the sender. The handler function for the invitation channel is sendPrivateChatMessage
			           	   which should start a listener on the private channel for all those wolfram ids sent to. *)
					a = Association["privatechannel" -> ("chatframework@wolfram.com" <> "/" <> id), "participants" -> Compress@participants,
							"ScreenName" -> Compress@If[MemberQ[{"", "None"}, screenname], "None", screenname]];
					NotebookDelete[cells, AutoScroll -> (!CurrentValue[bn, {TaggingRules, "ScrollLock"}, False])];
					Quiet[ChannelSend[#, a, ChannelPreSendFunction -> None]&/@(ChannelObject[# <> ":ChatInvitations"] & /@ participants(*DeleteCases[participants, $WolframID]*));
						ChannelObject::uauth],
							
					(* It is possible that a receiver has stopped their listener and then restarted between sends - so the receiver would need to know what private channel to listen on.
			                   This would need to be handled in the listener restart code. *)
			                NotebookDelete[cells, AutoScroll -> (!CurrentValue[bn, {TaggingRules, "ScrollLock"}, False])];
					Quiet[ChannelSend[StringJoin["chatframework@wolfram.com", ":", id], li, ChannelPreSendFunction -> None], ChannelObject::uauth]],
					
				NotebookDelete[cells, AutoScroll -> (!CurrentValue[bn, {TaggingRules, "ScrollLock"}, False])];
				ChannelSend["Wolfram:ChatTools", li]];
					
			sel = Select[Notebooks[], CurrentValue[#, {TaggingRules, "ChatNotebookID"}] === id &];
			If[Length@sel > 1, SetSelectedNotebook /@ Complement[sel, {bn}]];
					
			CurrentValue[bn, {TaggingRules, "AllowAttachSendButtons"}] = Inherited;
			If[CurrentValue[bn, {TaggingRules, "BeforeSend"}] === True, CurrentValue[bn, DockedCells] = ChatNotebookDockedCell[id, windowtitle, "CanSetVisibility" -> False,
							"Preferences" -> PrivateChatPreferencesMenu[id, "UpdateParticipants" -> SameQ[$WolframID, CurrentValue[bn, {TaggingRules, "Originator"}]]]]];
			CurrentValue[bn, {TaggingRules, "BeforeSend"}] = Inherited]];
			
	If[(Length[st = Cells[bn, CellStyle -> "Stem"]] > 1) && (CurrentValue[InputNotebook[], DockedCells] =!= {}),
		SetOptions[st[[1]], Deletable -> True];
		NotebookDelete[st[[1]], AutoScroll -> (!CurrentValue[bn, {TaggingRules, "ScrollLock"}, False])]];
	If[(CurrentValue[bn, DockedCells] === {}) && (Cells[bn, CellStyle -> "Stem"] =!= {}),
		(SetOptions[#, Deletable -> True]; NotebookDelete[#, AutoScroll -> (!CurrentValue[bn, {TaggingRules, "ScrollLock"}, False])])&/@Cells[bn, CellStyle -> "Stem"]]]]];

InputCellColorBackgroundRule := Cell[BoxData[TagBox[TooltipBox[ButtonBox[Cell[BoxData[PaneSelectorBox[{False -> FrameBox[a_, __], 
													True -> FrameBox[a_, __]}, _, Background -> _]], "Input", Background -> _],
										o1__], "Click to copy"], MouseAppearanceTag["LinkHand"]]], "Input", o2___] :> 
				Cell[BoxData[TagBox[TooltipBox[ButtonBox[Cell[BoxData[PaneSelectorBox[{False -> FrameBox[a, ContentPadding -> False, FrameStyle -> GrayLevel[0.93], BaseStyle->{ShowStringCharacters->True}], 
													True -> FrameBox[a, ContentPadding -> False, BaseStyle->{ShowStringCharacters->True}]}, Dynamic[CurrentValue["MouseOver"]], 
											Background -> GrayLevel[0.93]]], "Input", Background -> GrayLevel[0.93]], o1], "Click to copy"], 
																			MouseAppearanceTag["LinkHand"]]], "Input", o2];

Options[updateNotebook] = {"Private" -> False, "OpenFromInvitation" -> False};

updateNotebook[requester_, id_, timestamp_, date_, ChatCreationDate_, cellList_, participants_List, wid_, screenname_, windowtitle_, CustomWindowTitle_, originator_, allparticipants_, opts___?OptionQ] :=
	Module[{private = ("Private" /. {opts} /. Options[updateNotebook]), openFromInvitation = ("OpenFromInvitation" /. {opts} /. Options[updateNotebook]),
		sel = SelectFirst[Notebooks[], (CurrentValue[#, {TaggingRules, "ChatNotebookID"}] ===id) &], listeners, listenerdata,
		privatechannellistener, channellistener, wolframIDAndCellListPairs, allpartics, allparticipants1, allparticipants2, ParticipantStyleRules, celllist2, alias, pos, sentStyles, styleToUse,
		type, screenname1, celllist3, st, chatData, compressedCellListUUIDs, tk, Screenname, nb},
			Which[And[wid =!= $WolframID, sel =!= Missing["NotFound"]],
			
				If[Not@TrueQ@CurrentValue[sel, {TaggingRules, "ParticipantRemoved"}],
				
					listeners = ChannelListeners[];
					listenerdata = {#, #["URL"]} & /@ listeners;
					channellistener = With[{id1 = id}, FirstCase[listenerdata, {a_, b_String /; StringMatchQ[b, __ ~~ id1]} :> a]];
					With[{timestamp1 = ToString[channellistener["FullMessage"]["Timestamp"]]},
						wolframIDAndCellListPairs = If[StringQ@timestamp1 && StringMatchQ[timestamp1, "DateObject[" ~~ __],
								{{wid, cellList/. Cell[a__, TaggingRules -> {"CellStatus" -> "Sent", "Sender" -> s_, "TimeStamp" -> _}, b__] :> Cell[a, TaggingRules -> {"CellStatus" -> "Sent", "Sender" -> s, "TimeStamp" -> timestamp1}, b]}},
								{{wid, cellList}}]];
					allpartics = CurrentValue[sel, {TaggingRules, "AllParticipants"}];
					allparticipants1 = If[MemberQ[allpartics, $WolframID], allpartics, Append[allpartics, $WolframID]];
					If[Not@MemberQ[allpartics, $WolframID], CurrentValue[sel, {TaggingRules, "AllParticipants"}] = Append[allpartics, $WolframID]];
					allparticipants2 = If[MemberQ[allparticipants1, requester], allparticipants1, Append[allparticipants1, requester]];
					If[Not@MemberQ[allparticipants1, requester], CurrentValue[sel, {TaggingRules, "AllParticipants"}] = Append[allparticipants1, requester]];
					ParticipantStyleRules = ParticipantStyleRule[allparticipants2, #] & /@ allparticipants2;
					celllist2 = Flatten[transformCellList[#, allparticipants2, ParticipantStyleRules] & /@ (If[#[[1]] =!= $WolframID,
																# /. InputCellColorBackgroundRule,
																#] & /@ wolframIDAndCellListPairs)];
					st = Cells[sel, CellStyle -> "Stem"];
					If[st =!= {},
						SetOptions[st[[-1]], Deletable -> True];
						NotebookWrite[st[[-1]], Append[fixFrameBox/@celllist2, $stemCellTemplate],AutoScroll -> (!CurrentValue[sel, {TaggingRules, "ScrollLock"}, False])];
						If[Length[st = Cells[sel, CellStyle -> "Stem"]] > 1,
							SetOptions[st[[1]], Deletable -> True];
							NotebookDelete[st[[1]], AutoScroll -> (!CurrentValue[sel, {TaggingRules, "ScrollLock"}, False])]]]],
							
				And[wid === $WolframID, sel =!= Missing["NotFound"]],
				
				listeners = ChannelListeners[];
				listenerdata = {#, #["URL"]} & /@ listeners;
				privatechannellistener = With[{id1 = id}, FirstCase[listenerdata, {a_, b_String /; StringMatchQ[b, __ ~~ id1]} :> a]];
				channellistener = If[privatechannellistener["FullMessage"] === Missing["NotAvailable"],
							FirstCase[listenerdata, {a_, b_String /; StringMatchQ[b, __ ~~ "ChatInvitations"]} :> a],
							privatechannellistener];
				With[{timestamp1 = ToString[channellistener["FullMessage"]["Timestamp"]]},
					celllist2 = If[StringQ@timestamp1 && StringMatchQ[timestamp1, "DateObject[" ~~ __],
									cellList/. Cell[a__, TaggingRules -> {"CellStatus" -> "Sent", "Sender" -> s_, "TimeStamp" -> _}, b__] :> Cell[a, TaggingRules -> {"CellStatus" -> "Sent", "Sender" -> s, "TimeStamp" -> timestamp1}, b],
									cellList];
					alias = CurrentValue[sel, {TaggingRules, "Alias"}];
					allparticipants1 = CurrentValue[sel, {TaggingRules, "AllParticipants"}];
					pos = Position[allparticipants1, If[alias === Inherited, $WolframID, alias]][[1, 1]];
					sentStyles = $sentStyles;
					styleToUse = sentStyles[[If[# === 0, -1, #] &[Mod[pos, Length@sentStyles]]]];
					type = CurrentValue[sel, {TaggingRules, "RoomType"}];
					screenname1 = CurrentValue[sel, {TaggingRules, "ScreenName"}];
					If[Not@StringQ@screenname1, screenname1 = $WolframID];
					celllist3 = Prepend[celllist2, UserWolframIDandTimeCell[styleToUse, If[MemberQ[{"", "None"}, screenname1], If[alias === Inherited, $WolframID, alias], screenname1], 
												If[type === "PromptedResponse",
													hourMinutesSecondsHundredths@timestamp1, 
					    								hourMinutes@timestamp1]]];
					st = Cells[sel, CellStyle -> "Stem"];
					If[st =!= {},
						SetOptions[st[[-1]], Deletable -> True];					
						NotebookWrite[st[[-1]], Append[fixFrameBox/@celllist3, $stemCellTemplate],AutoScroll -> (!CurrentValue[sel, {TaggingRules, "ScrollLock"}, False])];
						If[Length[st = Cells[sel, CellStyle -> "Stem"]] > 1,
							SetOptions[st[[1]], Deletable -> True];
						NotebookDelete[st[[1]], AutoScroll -> (!CurrentValue[sel, {TaggingRules, "ScrollLock"}, False])]]]],
				
				And[wid =!= $WolframID, sel === Missing["NotFound"], MemberQ[participants, $WolframID], Select[Notebooks[], CurrentValue[#, {TaggingRules, "JoinChat"}] === id &] === {}],
				
				listeners = ChannelListeners[];
				listenerdata = {#, #["URL"]} & /@ listeners;
				channellistener = With[{id1 = id}, FirstCase[listenerdata, {a_, b_String /; StringMatchQ[b, __ ~~ id1]} :> a]];
				wolframIDAndCellListPairs = If[(Head@channellistener=== ChannelListener) && AssociationQ@channellistener["Message"] && Not@FreeQ[cellList, Cell],
								With[{timestamp1 = If[timestamp === "", ToString[channellistener["FullMessage"]["Timestamp"]], timestamp]},
									If[StringQ@timestamp1 && StringMatchQ[timestamp1, "DateObject[" ~~ __],
										{{wid, cellList/. Cell[a__, TaggingRules -> {"CellStatus" -> "Sent", "Sender" -> s_, "TimeStamp" -> _}, b__] :> Cell[a, TaggingRules -> {"CellStatus" -> "Sent", "Sender" -> s, "TimeStamp" -> timestamp1}, b]}},
										{{wid, cellList}}]],
								MapAt[Uncompress@CloudGet@CloudObject["https://www.wolframcloud.com/objects/" <> #] &, #, 2] & /@ cellList];
				allparticipants1 = If[MemberQ[allparticipants, $WolframID], allparticipants, Append[allparticipants, $WolframID]];
				allparticipants2 = If[MemberQ[allparticipants1, requester], allparticipants1, Append[allparticipants1, requester]];
				ParticipantStyleRules = ParticipantStyleRule[allparticipants2, #] & /@ allparticipants2;
				celllist3 = Flatten[transformCellList[#, allparticipants2, ParticipantStyleRules] & /@ (If[#[[1]] =!= $WolframID,
														# /. InputCellColorBackgroundRule,
														#] & /@ wolframIDAndCellListPairs)];
				chatData = CloudGet@CloudObject["https://www.wolframcloud.com/objects/" <> id];
				compressedCellListUUIDs = chatData["cellListUUIDs"];
				tk = If[Length@compressedCellListUUIDs <= 1, compressedCellListUUIDs, Take[compressedCellListUUIDs, -1]];
				Screenname = (Function[s, If[StringQ@s && Not@StringMatchQ[s, "" | Whitespace], s, ""]][CurrentValue[$FrontEnd,
															{PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "ScreenName"}]]);
				With[{id1 = id, date1 = date, participantsList = participants, windowtitle1 = windowtitle},
					nb = NotebookPut@Notebook[If[(Length@compressedCellListUUIDs >= 1) && (openFromInvitation === False), 
									f = With[{widMessageUUIDPairs = compressedCellListUUIDs}, 
										Button[Overlay[{Graphics[{EdgeForm[{Thin, GrayLevel[.60]}], #, Rectangle[{0, 0}, {3, 1}, RoundingRadius -> 0.2]}, 
													ImageSize -> {80, 30}], 
												Style["Show More...", FontFamily -> "Source Sans Pro", 13, FontColor -> RGBColor["#333333"]]}, Alignment -> Center], 
											ChatTools`InsertMoreRoomCellsFromCloud[widMessageUUIDPairs, "AddOldestInList" -> True], Method -> "Queued", Appearance -> "Frameless"] &]; 
									oldestTaken = tk[[1]];
									Prepend[#, Cell[BoxData[ToBoxes[Mouseover[f[RGBColor["#E5E5E5"]], f[RGBColor["#FFFFFF"]]]]], 
									"More", TaggingRules -> {"OldestTaken" -> oldestTaken}]], #]&[Append[fixFrameBox /@ celllist3, $stemCellTemplate]],
 								DockedCells -> ChatTools`Private`ChatNotebookDockedCell[id1, windowtitle, "CanSetVisibility" -> False,
												"Preferences" -> PrivateChatPreferencesMenu[id1]],
									WindowTitle -> windowtitle(*If[CustomWindowTitle === "None",
												StringJoin[If[MemberQ[{"", "None"}, Screenname],StringReplace[$WolframID, p__~~"@"~~__:>p], Screenname],
														"'s Chat - ",
														"("<>StringReplace[dateTime[TimeZoneAdjust@ChatCreationDate], a__ ~~ ":" ~~ __ :> a]<>")"],
												CustomWindowTitle]*),
									"TrackCellChangeTimes" -> False, 
									TaggingRules -> {"ChatNotebook" -> "True", "SelectedCells" -> {}, "ScrollLock" -> False, "Participants" -> participantsList,
												"ChatNotebookID" -> id1, "ChatNotebookDate" -> date1, "ChatCreationDate" -> ChatCreationDate, 
												"ChatNotebookWindowTitle" -> windowtitle1, "CustomWindowTitle" -> CustomWindowTitle,
												"Originator" -> originator, "ChatVisibility" -> "Private", "AllParticipants" -> allparticipants2,
												"ScreenName" -> (If[MemberQ[{"", "None"}, #], 
															"None",
															#]&[CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "ScreenName"}]]),
												"OriginalWolframID" -> $WolframID},
									CreateCellID -> True, 
									StyleDefinitions -> FrontEnd`FileName[{"Wolfram"}, "ChatTools.nb"], 
									CellLabelAutoDelete -> False, 
									WindowTitle -> windowtitle1, Evaluator->"ChatServices"]]; 
				NotebookFind[nb, "Stem", All, CellStyle, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])];
				SelectionMove[nb, After, Cell, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])];
				With[{nb1 = nb, id1 = id, screenname2 = Screenname},
					SetOptions[nb, NotebookEventActions -> {{"MenuCommand", "SimilarCellBelow"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
													{"MenuCommand", "SaveRename"} :> ChatTools`SaveChatNotebook[nb1],
								If[$VersionNumber < 12, Nothing, {"MenuCommand", "Copy"} :> Module[{in = InputNotebook[]}, 
									If[MatchQ[Developer`CellInformation[in], {___, {"Style" -> {_, __}, __, "CursorPosition" -> "CellBracket", __}, ___}], 
										CopyToClipboard[DeleteCases[ReplaceAll[NotebookRead@in,
						Cell[a_, b_String, _String, ___, TaggingRules -> _, ___] :> Cell[a, b]], Cell[__, "CellUserLabelOptions" | "CellLabelOptions" | "Stem", ___], Infinity]], 
										FrontEndExecute[{FrontEndToken[in, "Copy"]}]]]],
			"WindowClose" :> (CurrentValue[nb1, Saveable] = False; NotebookClose[nb1]; If[Not@TrueQ@CurrentValue[nb1, {TaggingRules, "BeforeSend"}],
				RunScheduledTask[Quiet[ChannelSend[StringJoin["chatframework@wolfram.com", ":", id1], Association @@ List["id" -> id1, "removedparticipant" -> $WolframID, 
		"screenname" -> Compress@If[MemberQ[{"", Inherited}, screenname2] || (StringQ@screenname2 && StringMatchQ[screenname2, Whitespace]), "None", screenname2], "celllist" -> Compress@{}], 
										ChannelPreSendFunction -> None], ChannelObject::uauth]; RemoveScheduledTask[$ScheduledTask];, {0.1}]]),
													PassEventsDown -> False}]]]];
													
Options[InsertMoreSentCellsFromCloud] = {"AddOldestInList" -> False};

InsertMoreSentCellsFromCloud[widMessageUUIDPairs_, opts___?OptionQ]:= InsertMoreSentCellsFromCloud[ButtonNotebook[], widMessageUUIDPairs, opts];

InsertMoreSentCellsFromCloud[nb_, widMessageUUIDPairs_, opts___?OptionQ] := 
	Module[{addOldestInList = ("AddOldestInList" /. {opts} /. Options[InsertMoreSentCellsFromCloud]), cellObject = Cells[nb, CellStyle -> "More"], oldestTaken, compressedCellListUUIDs, type,
		oldCompressedCellListUUIDsNotGotten, allparticipants, ParticipantStyleRules, wolframIDAndCellListPairs, f, cellsToAddInNotebook}, 
		
		oldestTaken = If[cellObject =!= {}, CurrentValue[cellObject[[1]], {TaggingRules, "OldestTaken"}], If[Length@widMessageUUIDPairs > 5, widMessageUUIDPairs[[-5]], Inherited]];
		
		compressedCellListUUIDs = widMessageUUIDPairs;
		type = CurrentValue[nb, {TaggingRules, "RoomType"}];
		
		oldCompressedCellListUUIDsNotGotten = If[cellObject === {},
								compressedCellListUUIDs,
								If[addOldestInList === True, compressedCellListUUIDs, compressedCellListUUIDs[[1;;Position[compressedCellListUUIDs, oldestTaken][[1, 1]] - 1]]]];
		allparticipants = CurrentValue[nb, {TaggingRules, "AllParticipants"}]; 
		ParticipantStyleRules = ParticipantStyleRule[allparticipants, #] & /@ allparticipants; 
		If[Length@oldCompressedCellListUUIDsNotGotten <= 5,
					
			(*If[cellObject =!= {}, NotebookDelete[cellObject[[1]]]];*)
			wolframIDAndCellListPairs = MapAt[Uncompress@CloudGet@CloudObject["https://www.wolframcloud.com/objects/" <> #] &, #, 2] & /@ oldCompressedCellListUUIDsNotGotten;
			cellsToAddInNotebook = Flatten[transformCellList[#, allparticipants, ParticipantStyleRules,
							"TimeSpecifier" -> If[type === "PromptedResponse", "PromptedResponse", Automatic]] & /@ (If[#[[1]] =!= $WolframID, # /. {InputCellColorBackgroundRule, InputCellColorBackgroundRule2}, #] & /@wolframIDAndCellListPairs)],
						
			oldestTaken = compressedCellListUUIDs[[If[oldestTaken === Inherited,
									0,
									If[addOldestInList === True, Length@compressedCellListUUIDs, Position[compressedCellListUUIDs, oldestTaken][[1, 1]] - 5]]]];
			wolframIDAndCellListPairs = MapAt[Uncompress@CloudGet@CloudObject["https://www.wolframcloud.com/objects/" <> #] &, #, 2] & /@ Take[oldCompressedCellListUUIDsNotGotten, -5];
			f = With[{compressedCellListUUIDs1 = compressedCellListUUIDs}, Button[Overlay[{Graphics[{EdgeForm[{Thin, GrayLevel[.60]}], #, Rectangle[{0, 0}, {3, 1}, RoundingRadius -> 0.2]}, ImageSize -> {80, 30}], 
										Style["Show More...", FontFamily -> "Source Sans Pro", 13, FontColor -> RGBColor["#333333"]]}, Alignment -> Center], ChatTools`InsertMoreSentCellsFromCloud[compressedCellListUUIDs1],
										Method -> "Queued", 
							Appearance -> "Frameless"]&];
			cellsToAddInNotebook = Prepend[Flatten[transformCellList[#, allparticipants, ParticipantStyleRules,
							"TimeSpecifier" -> If[type === "PromptedResponse", "PromptedResponse", Automatic]] & /@ (If[#[[1]] =!= $WolframID, # /. {InputCellColorBackgroundRule, InputCellColorBackgroundRule2}, #] & /@wolframIDAndCellListPairs)], 
							Cell[BoxData[ToBoxes[Mouseover[f[RGBColor["#E5E5E5"]], f[RGBColor["#FFFFFF"]]]]], "More", TaggingRules -> {"OldestTaken" -> oldestTaken}]]]; 
						
		If[cellObject === {},
					
			SelectionMove[nb, Before, Notebook, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])];
			NotebookWrite[nb, fixFrameBox/@cellsToAddInNotebook, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])];
			NotebookFind[nb, "Stem", All, CellStyle, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])];
			SelectionMove[nb, After, Cell, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])],

			SetOptions[cellObject[[1]], Deletable -> True];
			NotebookWrite[cellObject[[1]], fixFrameBox/@cellsToAddInNotebook, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])]];
						
		If[Length@oldCompressedCellListUUIDsNotGotten <= 5, ChatTools`Private`StopRestoringPrivateMessages = True]];
					
InsertAllSentCellsFromCloud[] := 
	Module[{nb = ButtonNotebook[], MoreCells, ChatData, compressedCellListUUIDs, allparticipants, ParticipantStyleRules, wolframIDAndCellListPairs, cellsInNotebook, tk, f, st, progressNB}, 
		If[TrueQ@CurrentValue[nb, {TaggingRules, "BeforeSend"}],
		
			MessageDialog["No messages have been sent.", WindowFrame -> "ModalDialog", WindowSize -> {230, All}],
			
			If[(MoreCells = Cells[nb, CellStyle -> "More"]) === {} && Cells[nb, CellStyle -> "CellLabelOptions"] =!= {}, ClearSentCells[]];
			
			If[Quiet[ChatData = CloudGet@CloudObject["https://www.wolframcloud.com/objects/" <> CurrentValue[nb, {TaggingRules, "ChatNotebookID"}]], CloudObject::notperm] === $Failed,
			
			MessageDialog["An error occurred while trying to get the chat room data. Try again later.", WindowFrame -> "ModalDialog", WindowSize -> {510, All}],
			
			compressedCellListUUIDs = ChatData["cellListUUIDs"];
			ChatTools`Private`StopRestoringPrivateMessages = False;
			ChatTools`Private`posp = 1;
			ChatTools`Private`lenp = Length@compressedCellListUUIDs;
			
			If[Cells[nb, CellStyle -> "More"] ==={} && compressedCellListUUIDs =!={}, 
							
				allparticipants = CurrentValue[nb, {TaggingRules, "AllParticipants"}]; 
				ParticipantStyleRules = ParticipantStyleRule[allparticipants, #] & /@ allparticipants;
				If[Length@compressedCellListUUIDs <= 1,
											
					wolframIDAndCellListPairs = MapAt[Uncompress@CloudGet@CloudObject["https://www.wolframcloud.com/objects/" <> #] &, #, 2] & /@ compressedCellListUUIDs;
					cellsInNotebook = Append[Flatten[transformCellList[#, allparticipants, ParticipantStyleRules,
											"TimeSpecifier" -> If[type === "PromptedResponse", "PromptedResponse", Automatic]] & /@ (If[#[[1]] =!= $WolframID, # /. {InputCellColorBackgroundRule, InputCellColorBackgroundRule2}, #] & /@wolframIDAndCellListPairs)],
								$stemCellTemplate],
												
					tk = Take[compressedCellListUUIDs, -1];
					wolframIDAndCellListPairs = MapAt[Uncompress@CloudGet@CloudObject["https://www.wolframcloud.com/objects/" <> #] &, #, 2] & /@ tk;
					f = With[{widMessageUUIDPairs = compressedCellListUUIDs}, Button[Overlay[{Graphics[{EdgeForm[{Thin, GrayLevel[.60]}], #, Rectangle[{0, 0}, {3, 1}, RoundingRadius -> 0.2]}, ImageSize -> {80, 30}], 
					Style["Show More...", FontFamily -> "Source Sans Pro", 13, FontColor -> RGBColor["#333333"]]}, Alignment -> Center], ChatTools`InsertMoreRoomCellsFromCloud[widMessageUUIDPairs],
						Method -> "Queued", 
						Appearance -> "Frameless"]&];
					cellsInNotebook = Prepend[Append[Flatten[transformCellList[#, allparticipants, ParticipantStyleRules,
													"TimeSpecifier" -> If[type === "PromptedResponse", "PromptedResponse", Automatic]] & /@ (If[#[[1]] =!= $WolframID, # /. {InputCellColorBackgroundRule, InputCellColorBackgroundRule2}, #] & /@wolframIDAndCellListPairs)], $stemCellTemplate],
												Cell[BoxData[ToBoxes[Mouseover[f[RGBColor["#E5E5E5"]], f[RGBColor["#FFFFFF"]]]]], "More", TaggingRules -> {"OldestTaken" -> tk[[1]]}]]];
																
					st = Cells[nb, CellStyle -> "Stem"];
					If[st =!= {},
						SetOptions[st[[-1]], Deletable -> True];
						NotebookWrite[st[[1]], cellsInNotebook, 
							AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])]]];
							
		Which[1 < Length@compressedCellListUUIDs < 7,
			
			ChatTools`InsertMoreSentCellsFromCloud[nb, compressedCellListUUIDs],
				
			7 <= Length@compressedCellListUUIDs,
				
			progressNB = NotebookPut@Notebook[{Cell[BoxData@ToBoxes@Grid[{{"", "", ""},
								{"", Style["Downloading all saved messages for\nthis chat from the Wolfram Cloud.", TextAlignment -> Center], ""},
								{"", Dynamic@Grid[{{Row[{ChatTools`Private`posp,"/",ChatTools`Private`lenp}],
											ProgressIndicator[ChatTools`Private`posp, {0, ChatTools`Private`lenp}]}}], ""},
								{"", "", ""},
								{"", Button["Cancel",ChatTools`Private`StopRestoringPrivateMessages = True; NotebookClose[], ImageSize -> Automatic], ""},
								{"", "", ""}}, BaseStyle -> {FontFamily -> "Source Sans Pro", FontSize -> 16}, 
							Spacings -> {{2 -> 1, 3 -> 1}, {2 -> 1, 3 -> 2, 4 -> 1}}, Alignment -> Center], "Text",
								CellMargins -> {{0, 0}, {0, 0}}, ShowCellBracket -> False, ShowStringCharacters -> False]},
							WindowMargins -> Automatic,
							WindowSize -> Fit,
							Evaluator -> "ChatServices",
							Background -> GrayLevel[1], 
							WindowTitle -> None,
							WindowFrameElements -> {},
							WindowElements -> {},
							"CellInsertionPointCell" -> {}, 
							"BlinkingCellInsertionPoint" -> False, 
							"CellInsertionPointColor" -> GrayLevel[1],
							Saveable -> False,
							Editable -> False, 
							Selectable -> False];
				While[ChatTools`Private`posp <= ChatTools`Private`lenp,
					If[Not@TrueQ@ChatTools`Private`StopRestoringPrivateMessages,
						ChatTools`InsertMoreSentCellsFromCloud[nb, compressedCellListUUIDs];
						If[ChatTools`Private`posp === ChatTools`Private`lenp, 
							NotebookClose[progressNB],
							ChatTools`Private`posp = If[ChatTools`Private`lenp - ChatTools`Private`posp < 5, 
											ChatTools`Private`lenp,
											ChatTools`Private`posp + 5]],
						NotebookClose[progressNB];
						ChatTools`Private`posp = ChatTools`Private`lenp + 1]]]]]];

(********** Begin code for working with chat rooms *******************)

SendMessageToRoom[msg_ /; And[KeyExistsQ[ReleaseHold@msg, "sender"], KeyExistsQ[ReleaseHold@msg, "requesteraddress"], KeyExistsQ[ReleaseHold@msg, "allparticipants"], KeyExistsQ[ReleaseHold@msg, "id"],
					KeyExistsQ[ReleaseHold@msg, "compressedCellList"], KeyExistsQ[ReleaseHold@msg, "name"], KeyExistsQ[ReleaseHold@msg, "alias"],
					KeyExistsQ[ReleaseHold@msg, "originator"], KeyExistsQ[ReleaseHold@msg,"SpecialAction"], KeyExistsQ[ReleaseHold@msg, "Banner"],
					KeyExistsQ[ReleaseHold@msg, "RoomType"], KeyExistsQ[ReleaseHold@msg, "Moderators"], KeyExistsQ[ReleaseHold@msg, "Shortcut"],
					KeyExistsQ[ReleaseHold@msg, "ScreenName"],
					Sort@Keys@ReleaseHold@msg === {"alias", "allparticipants", "Banner", "compressedCellList", "id", "Moderators", "name", "originator", "requesteraddress", "RoomType",
									"ScreenName", "sender", "Shortcut", "SpecialAction"}]] := 
	updateRoomChatNotebook[Lookup[ReleaseHold@msg, "sender"], Lookup[ReleaseHold@msg, "requesteraddress"], Uncompress@Lookup[ReleaseHold@msg, "allparticipants"],
				Lookup[ReleaseHold@msg, "id"], Uncompress@Lookup[ReleaseHold@msg, "compressedCellList"], Uncompress@Lookup[ReleaseHold@msg, "name"], Lookup[ReleaseHold@msg, "alias"],
				Lookup[ReleaseHold@msg, "originator"], Lookup[ReleaseHold@msg, "SpecialAction"], Lookup[ReleaseHold@msg, "Banner"], Lookup[ReleaseHold@msg, "RoomType"],
				Uncompress[Lookup[ReleaseHold@msg, "Moderators"]], Uncompress@Lookup[ReleaseHold@msg, "Shortcut"], Uncompress@Lookup[ReleaseHold@msg, "ScreenName"]];

AddPasteButtonForBanner[cell_Cell] := 
	If[MatchQ[cell, Cell[BoxData[_], "Input", ___]],
		Replace[cell, Cell[BoxData[a_], "Input", b___]:>Cell[BoxData[InputFieldBox[If[ListQ@a, RowBox@a, a],Expression, Appearance -> None]], "Input", b]],
		cell];

AddPasteButton[cell_Cell] := cell;

AddPasteButtonWithoutBannerStyle[cell_Cell] := cell;

UpdateBanner[nb_, compressedBannerCellList_String] := 
	Module[{dockedcell = CurrentValue[nb, DockedCells], uncompressed = If[compressedBannerCellList === "None", "None", Uncompress@compressedBannerCellList]}, 
		Which[MatchQ[dockedcell, Cell[_, "ChatDockedCell", a___ /; FreeQ[{a}, TaggingRules -> {"Banner" -> True}]]] && (uncompressed =!= "None"),
		
			If[Not@TrueQ@CurrentValue[nb, {TaggingRules, "ViewParticipants"}], CurrentValue[nb, DockedCells] = Prepend[Append[DeleteCases[#, CellChangeTimes->_], TaggingRules -> {"Banner" -> True}]&/@(Insert[#, "Banner",
																If[MatchQ[#, Cell[_, _String, ___]], 3, 2]] & /@(AddPasteButtonForBanner/@uncompressed)), dockedcell]];
			CurrentValue[nb, {TaggingRules, "Banner"}] = compressedBannerCellList,
			
			MatchQ[dockedcell, {_, Cell[__, TaggingRules -> {"Banner" -> True}]..}] && (compressedBannerCellList =!= "None"),
			
			CurrentValue[nb, DockedCells] = Prepend[Append[DeleteCases[#, CellChangeTimes->_], TaggingRules -> {"Banner" -> True}]&/@(Insert[#, "Banner",
															If[MatchQ[#, Cell[_, _String, ___]], 3, 2]] & /@(AddPasteButtonForBanner/@uncompressed)), dockedcell[[1]]];
			CurrentValue[nb, {TaggingRules, "Banner"}] = compressedBannerCellList,
			
			And[MatchQ[dockedcell, {Cell[_, "ChatDockedCell", a___ /; FreeQ[{a}, TaggingRules -> {"Banner" -> True}]], Cell[__, TaggingRules -> {"Banner" -> True}]..}],
				compressedBannerCellList === "None"],
				
			If[Not@TrueQ@CurrentValue[nb, {TaggingRules, "ViewParticipants"}], CurrentValue[nb, DockedCells] = dockedcell[[1]]];
			CurrentValue[nb, {TaggingRules, "Banner"}] = "None"]];
			
ParticipantStyleRule[participants_, participant_] := (participant -> $sentStyles[[If[# === 0, -1, #] &[Mod[Position[participants, participant][[1, 1]], Length@$sentStyles]]]]);

CellListStyle[celllist_, ParticipantStyleRules_] := (FirstCase[celllist[[1]], _[TaggingRules, {_, "Sender" -> s_, _}] :> s] /. ParticipantStyleRules);

Options[transformCellList] = {"TimeSpecifier" -> Automatic};

transformCellList[wolframIDAndCellListPair_, participants_, ParticipantStyleRules_, opts___?OptionQ] := 
	Module[{ts = ("TimeSpecifier" /. {opts} /. Options[transformCellList]), 
		cellData = FirstCase[wolframIDAndCellListPair[[2, 1]], _[TaggingRules, {_, "Sender" -> s_, "TimeStamp" -> t_}] :> {s, t}], cellListParticipant, time, style, topstyle}, 
		cellListParticipant = cellData[[1]];
		time = If[ts === "PromptedResponse", hourMinutesSecondsHundredths[cellData[[2]]], hourMinutes[cellData[[2]]]];
		style = (If[StringQ@#, #, "Turquoise"]&[CellListStyle[wolframIDAndCellListPair[[2]], ParticipantStyleRules /. wolframIDAndCellListPair[[1]] -> cellData[[1]]]]);
		If[wolframIDAndCellListPair[[1]] === $WolframID, 
			topstyle = style <> "Top";
			Prepend[If[Length[wolframIDAndCellListPair[[2]]] === 1,
					#,
					ReplacePart[#, {#, 3} -> style & /@ Rest@Range[Length[wolframIDAndCellListPair[[2]]]]]] &[ReplacePart[wolframIDAndCellListPair[[2]], {1, 3} -> topstyle]], 
				UserWolframIDandTimeCell[style, cellListParticipant, time]], 
			topstyle = "GrayLight" <> style <> "Top";
			Prepend[If[Length[wolframIDAndCellListPair[[2]]] === 1,
					#,
					ReplacePart[#, {#, 3} -> "GrayLight" & /@ Rest@Range[Length[wolframIDAndCellListPair[[2]]]]]] &[ReplacePart[wolframIDAndCellListPair[[2]], {1, 3} -> topstyle]], 
				WolframIDandTimeCell[style, cellListParticipant, time]]]];

InputCellColorBackgroundRule2 := Cell[BoxData[TagBox[TooltipBox[ButtonBox[Cell[BoxData[PaneSelectorBox[{False -> FrameBox[a_, __],
													True -> FrameBox[a_, __]},_, Background -> _]], "Input"], o1__],
								"Click to copy"], MouseAppearanceTag["LinkHand"]]], "Input", o2___] :> 
				Cell[BoxData[TagBox[TooltipBox[ButtonBox[Cell[BoxData[PaneSelectorBox[{False -> FrameBox[a, ContentPadding -> False, FrameStyle -> GrayLevel[0.93], BaseStyle->{ShowStringCharacters->True}], 
													True -> FrameBox[a, ContentPadding -> False, BaseStyle->{ShowStringCharacters->True}]}, Dynamic[CurrentValue["MouseOver"]], 
													Background -> GrayLevel[0.93]]], "Input", Background -> GrayLevel[0.93]], o1], "Click to copy"], 
													MouseAppearanceTag["LinkHand"]]], "Input", o2];

fixFrameBox:= If[MatchQ[#, Cell[a_ /; Not@FreeQ[a, FrameBox[{__}, ___]], "Input", ___]], # /. FrameBox[{b__}, o___] :> FrameBox[RowBox[{b}], o], #]&;

updateRoomChatNotebook[sender_, requesteraddress_, allparticipants_, id_, cellList_, name_, alias_, originator_, action_, compressedBannerCells_, type_, teachersadded1_, shortcut_, screenname_] := 
	Module[{moderatorq, teachersadded, moderators, sel = SelectFirst[Notebooks[], (CurrentValue[#, {TaggingRules, "ChatNotebookID"}] === id) &], presentModerators, st, allpartics, allparticipants1, allparticipants2, listeners,
		listenerdata, channellistener, RoomData, data, tk, compressedCellListUUIDs, wolframIDAndCellListPairs, ParticipantStyleRules, celllist2, celllist3, channellistener1, pos, sentStyles,
		styleToUse, screenname1, transformedCellsFromIndex, f, oldestTaken, cellList4, nb},
		
		moderatorq = URLExecute[CloudObject["https://www.wolframcloud.com/objects/1d3cacc5-7ec8-4e84-bc88-3c6a99ef78b8"], {"ShortName" -> Compress@shortcut, "wid" -> $WolframID}, "WL"];
		teachersadded = Which[MemberQ[{"None", None}, teachersadded1] && TrueQ@moderatorq && (originator =!= $WolframID),
					{$WolframID},
					MemberQ[{"None", None}, teachersadded1] && TrueQ@moderatorq,
					teachersadded1,
					ListQ@teachersadded1 && TrueQ@moderatorq && Not@MemberQ[teachersadded1, $WolframID],
					Append[teachersadded1, $WolframID],
					ListQ@teachersadded1 && TrueQ@moderatorq,
					teachersadded1,
					ListQ@teachersadded1 && Not@TrueQ@moderatorq && MemberQ[teachersadded1, $WolframID],
					DeleteCases[teachersadded1, $WolframID],
					True,
					teachersadded1];
		
		Which[sel =!= Missing["NotFound"],
		
			moderators = CurrentValue[sel, {TaggingRules, "Moderators"}];
			If[TrueQ@moderatorq && MemberQ[{"None", None}, moderators],
				CurrentValue[sel, {TaggingRules, "Moderators"}] = {$WolframID};
				moderators = {$WolframID}];
			If[TrueQ@moderatorq && ListQ@moderators && Not@MemberQ[moderators, $WolframID],
				CurrentValue[sel, {TaggingRules, "Moderators"}] = Append[moderators, $WolframID];
				moderators = Append[moderators, $WolframID]];
			If[Not@TrueQ@moderatorq && ListQ@moderators && MemberQ[moderators, $WolframID],
				CurrentValue[sel, {TaggingRules, "Moderators"}] = DeleteCases[moderators, $WolframID];
				moderators = DeleteCases[moderators, $WolframID]];
		
			Which[action === "UpdateBanner",
			
				UpdateBanner[sel, compressedBannerCells],
				
				action === "UpdateModerators",
				
				presentModerators = CurrentValue[sel, {TaggingRules, "Moderators"}];
				If[And[originator =!= $WolframID,
					presentModerators =!= teachersadded],
					
					If[And[Not@TrueQ@CurrentValue[sel, {TaggingRules, "ViewParticipants"}],
						Not@TrueQ@CurrentValue[sel, {TaggingRules, "ManageParticipants"}]],
					
						Which[And[(ListQ@teachersadded && MemberQ[teachersadded, $WolframID]),
							(# =!= {} && Cases[#[[1]], "\"Set Banner\"", Infinity] === {})&@Cases[CurrentValue[sel, DockedCells],
																	Cell[a__ /; Not@FreeQ[{a}, "\"Copy Channel UUID\""]], {0, 1}]],
																	
							CurrentValue[sel, DockedCells] = (If[(dockedcell = CurrentValue[nb, DockedCells]; MatchQ[dockedcell,
																		{_, Cell[__, TaggingRules -> {"Banner" -> True}]..}]),
											ReplacePart[dockedcell, 1 -> #],
											#]&[ChatNotebookDockedCell[id, name, "CanSetVisibility" -> False, 
															"Preferences" -> ChatClassRoomPreferencesMenu[id, shortcut, 
															"Teacher" -> True, 
															"CellLabelsDefault" -> If[type === "PromptedResponse", "On", "Off"]]]]);
							CurrentValue[sel, {TaggingRules, "Moderators"}] = teachersadded,
						
							And[(teachersadded === None) || (ListQ@teachersadded && Not@MemberQ[teachersadded, $WolframID]),
								(# =!= {} && Cases[#[[1]], "\"Set Banner\"", Infinity] =!= {})&@Cases[CurrentValue[sel, DockedCells],
																	Cell[a__ /; Not@FreeQ[{a}, "\"Copy Channel UUID\""]], {0, 1}]],
																	
							CurrentValue[sel, DockedCells] = (If[(dockedcell = CurrentValue[nb, DockedCells]; MatchQ[dockedcell, 
																		{_, Cell[__, TaggingRules -> {"Banner" -> True}]..}]),
											ReplacePart[dockedcell, 1 -> #],
											#]&[ChatNotebookDockedCell[id, name, "CanSetVisibility" -> False, 
															"Preferences" -> ChatClassRoomPreferencesMenu[id, shortcut, 
															"Teacher" -> False, 
															"CellLabelsDefault" -> If[type === "PromptedResponse", "On", "Off"]]]]);
							CurrentValue[sel, {TaggingRules, "Moderators"}] = teachersadded],
							
						CurrentValue[sel, {TaggingRules, "Moderators"}] = teachersadded]],
					
				True,
		
				Which[Or[(sender =!= $WolframID && type === "PromptedResponse" && ListQ@moderators && MemberQ[moderators, $WolframID]),
				
					(sender =!= $WolframID && type === "OpenGroup"),
				
					(* sender is different from receiver and sender is the originator ~ teacher sends to students *)
					(sender =!= $WolframID && type === "PromptedResponse" && sender === originator),
					
					(* sender is different from receiver and receiver is the originator (teacher) or any added teacher ~ student sends *)
					And[(sender =!= $WolframID), (type === "PromptedResponse"), Or[($WolframID === originator) && Not@ListQ@teachersadded,
													($WolframID === originator) || (ListQ@teachersadded && Or[MemberQ[teachersadded, $WolframID],
																				MemberQ[teachersadded, sender]])]],
					
					(* sender is sending from a form and teacher and any added teachers receive *)
					And[(action === "UpdateFromForm"), (type === "PromptedResponse"), Or[($WolframID === originator) && Not@ListQ@teachersadded,
													($WolframID === originator) || (ListQ@teachersadded && MemberQ[teachersadded, $WolframID])]]],
					
					st = Cells[sel, CellStyle -> "Stem"];
					If[st =!= {},
						If[Length@st > 1, (SetOptions[#, Deletable -> True]; NotebookDelete[#, AutoScroll -> (!CurrentValue[sel, {TaggingRules, "ScrollLock"}, False])])&/@Drop[st, -1]; st = Cells[sel, CellStyle -> "Stem"]];
						SetOptions[st[[1]], Deletable -> True];
						
						(*allparticipants1 = CurrentValue[sel, {TaggingRules, "AllParticipants"}]; 
						If[Not@MemberQ[allparticipants1, sender],
							allparticipants1 = Append[allparticipants1, sender]; 
							CurrentValue[sel, {TaggingRules, "AllParticipants"}] = allparticipants1];*)
							
						allpartics = CurrentValue[sel, {TaggingRules, "AllParticipants"}];
						allparticipants1 = If[MemberQ[allpartics, $WolframID], allpartics, Append[allpartics, $WolframID]];
						If[Not@MemberQ[allpartics, $WolframID], CurrentValue[sel, {TaggingRules, "AllParticipants"}] = Append[allpartics, $WolframID]];
						allparticipants2 = If[MemberQ[allparticipants1, sender], allparticipants1, Append[allparticipants1, sender]];
						If[Not@MemberQ[allparticipants1, sender], CurrentValue[sel, {TaggingRules, "AllParticipants"}] = Append[allparticipants1, sender]];
						listeners = ChannelListeners[];
						listenerdata = {#, #["URL"]} & /@ listeners;
						channellistener = With[{id1 = id}, FirstCase[listenerdata, {a_, b_String /; StringMatchQ[b, __ ~~ id1]} :> a]];
						With[{timestamp = ToString[channellistener["FullMessage"]["Timestamp"]]},
							wolframIDAndCellListPairs = If[StringQ@timestamp && StringMatchQ[timestamp, "DateObject[" ~~ __],
											{{sender, ReplaceAll[Replace[cellList, Cell[BoxData[a_String], "Input", o___] :> AddPasteButtonWithoutBannerStyle@Cell[BoxData[a], "Input", o], {1}],
								Cell[a__, TaggingRules -> {"CellStatus" -> "Sent", "Sender" -> s_, "TimeStamp" -> _}, b__] :> Cell[a, TaggingRules -> {"CellStatus" -> "Sent", "Sender" -> s, "TimeStamp" -> timestamp}, b]]}},
											{{sender, cellList}}]];
						ParticipantStyleRules = ParticipantStyleRule[allparticipants2, #] & /@ allparticipants2;
						celllist2 = Flatten[transformCellList[#, allparticipants2, ParticipantStyleRules,
											"TimeSpecifier" -> If[type === "PromptedResponse","PromptedResponse",Automatic]] & /@ (If[#[[1]] =!= $WolframID,
												# /. {InputCellColorBackgroundRule, InputCellColorBackgroundRule2},
										#] & /@ wolframIDAndCellListPairs)];
						celllist3 = fixFrameBox/@(If[MatchQ[celllist2, {_, _, Cell[_, _] ..}], MapAt[Append[#, "GrayLight"] &, celllist2, List /@ Drop[Range[Length@celllist2], 2]], celllist2]);
						NotebookWrite[st[[1]], Append[celllist3, $stemCellTemplate], AutoScroll -> (!CurrentValue[sel, {TaggingRules, "ScrollLock"}, False])]],
						
				sender === $WolframID,
				
				listeners = ChannelListeners[];
				listenerdata = {#, #["URL"]} & /@ listeners;
				channellistener1 = With[{id1 = id}, FirstCase[listenerdata, {a_, b_String /; StringMatchQ[b, __ ~~ id1]} :> a]];
				channellistener = If[channellistener1["FullMessage"] === Missing["NotAvailable"],
							FirstCase[listenerdata, {a_, b_String /; StringMatchQ[b, __ ~~ "ChatInvitations"]} :> a],
							channellistener1];
				With[{timestamp1 = ToString[channellistener["FullMessage"]["Timestamp"]]},
					celllist2 = If[StringQ@timestamp1 && StringMatchQ[timestamp1, "DateObject[" ~~ __],
									cellList/. Cell[a__, TaggingRules -> {"CellStatus" -> "Sent", "Sender" -> s_, "TimeStamp" -> _}, b__] :> Cell[a, TaggingRules -> {"CellStatus" -> "Sent", "Sender" -> s, "TimeStamp" -> timestamp1}, b],
									cellList];
					allpartics = CurrentValue[sel, {TaggingRules, "AllParticipants"}];
					allparticipants1 = If[MemberQ[allpartics, $WolframID], allpartics, Append[allpartics, $WolframID]];
					If[Not@MemberQ[allpartics, $WolframID], CurrentValue[sel, {TaggingRules, "AllParticipants"}] = Append[allpartics, $WolframID]];
					pos = Position[allparticipants1, If[alias === Inherited, $WolframID, alias]][[1, 1]];
					sentStyles = $sentStyles;
					styleToUse = sentStyles[[If[# === 0, -1, #] &[Mod[pos, Length@sentStyles]]]];
					screenname1 = CurrentValue[sel, {TaggingRules, "ScreenName"}];
					If[Not@StringQ@screenname1, screenname1 = $WolframID];
					celllist3 = Prepend[celllist2, UserWolframIDandTimeCell[styleToUse, If[MemberQ[{"", "None"}, screenname1], If[alias === Inherited, $WolframID, alias], screenname1], 
												If[type === "PromptedResponse",
													hourMinutesSecondsHundredths@timestamp1, 
					    								hourMinutes@timestamp1]]];
					st = Cells[sel, CellStyle -> "Stem"];
					If[st =!= {},
						SetOptions[st[[-1]], Deletable -> True];					
						NotebookWrite[st[[-1]], Append[fixFrameBox/@celllist3, $stemCellTemplate],AutoScroll -> (!CurrentValue[sel, {TaggingRules, "ScrollLock"}, False])];
						If[Length[st = Cells[sel, CellStyle -> "Stem"]] > 1,
							SetOptions[st[[1]], Deletable -> True];
						NotebookDelete[st[[1]], AutoScroll -> (!CurrentValue[sel, {TaggingRules, "ScrollLock"}, False])]]]],
						
				True,
								
				(* Still update all participants. *)
				allparticipants1 = CurrentValue[sel, {TaggingRules, "AllParticipants"}]; 
				If[Not@MemberQ[allparticipants1, sender],
					allparticipants1 = Append[allparticipants1, sender]; 
					CurrentValue[sel, {TaggingRules, "AllParticipants"}] = allparticipants1]]],
							
			And[(sender =!= $WolframID && MemberQ[{"OpenGroup", "PromptedResponse"}, type]), sel === Missing["NotFound"]], 
			
			RoomData = CloudGet@CloudObject["https://www.wolframcloud.com/objects/" <> id]; 
			If[AssociationQ@RoomData && Not@MatchQ[data = RoomData["cellListUUIDs"], Missing["KeyAbsent", _]], 
				compressedCellListUUIDs = Which[type === "OpenGroup",
								data,
								Or[($WolframID === originator) && Not@ListQ@teachersadded,
									($WolframID === originator) || (ListQ@teachersadded && MemberQ[teachersadded, $WolframID])],
								data,
								True,
								Cases[data, {b : If[ListQ@teachersadded,
											Alternatives @@ Join[teachersadded, {originator, $WolframID}],
											originator | $WolframID], c_} :> {b, c}]];
				tk = If[Length@compressedCellListUUIDs <= 1, compressedCellListUUIDs, Take[compressedCellListUUIDs, -1]];
				wolframIDAndCellListPairs = ReplaceAll[MapAt[Uncompress@CloudGet@CloudObject["https://www.wolframcloud.com/objects/" <> #] &, #, 2] & /@ tk,
									c : Cell[BoxData[_String], "Input", ___] :> AddPasteButtonWithoutBannerStyle@c];
				If[MatchQ[wolframIDAndCellListPairs, {{_String, {Cell[__] ..}} ..}],
				
					allparticipants1 = (If[MemberQ[#, sender], #, Append[#, sender]]&[DeleteDuplicates[First /@ compressedCellListUUIDs]]);
					ParticipantStyleRules = ParticipantStyleRule[allparticipants1, #] & /@ allparticipants1;
					transformedCellsFromIndex = Flatten[transformCellList[#, allparticipants1, ParticipantStyleRules,
											"TimeSpecifier" -> If[type === "PromptedResponse","PromptedResponse",Automatic]] & /@ (If[#[[1]] =!= $WolframID, # /. {InputCellColorBackgroundRule, InputCellColorBackgroundRule2}, #] & /@wolframIDAndCellListPairs)];
					transformedCellsFromIndex = If[MatchQ[#, Cell[_, "Input"]], Append[#, "GrayLight"], #] & /@ transformedCellsFromIndex;
					If[Length@compressedCellListUUIDs > 1,
						f = With[{widMessageUUIDPairs = compressedCellListUUIDs}, 
								Button[Overlay[{Graphics[{EdgeForm[{Thin, GrayLevel[.60]}], #, Rectangle[{0, 0}, {3, 1}, RoundingRadius -> 0.2]}, ImageSize -> {80, 30}], 
												Style["Show More...", FontFamily -> "Source Sans Pro", 13, FontColor -> RGBColor["#333333"]]}, Alignment -> Center], 
									ChatTools`InsertMoreRoomCellsFromCloud[widMessageUUIDPairs], Method -> "Queued", Appearance -> "Frameless"] &]; 
						transformedCellsFromIndex = Prepend[transformedCellsFromIndex, Cell[BoxData[ToBoxes[Mouseover[f[RGBColor["#E5E5E5"]], f[RGBColor["#FFFFFF"]]]]], "More",
															TaggingRules -> {"OldestTaken" -> tk[[1]]}]]];
					
					If[Or[type === "OpenGroup",
						(sender === originator) && Not@ListQ@teachersadded,
						(sender =!= originator) && ($WolframID === originator) && Not@ListQ@teachersadded,
						ListQ@teachersadded && MemberQ[teachersadded, $WolframID]],
					
						listeners = ChannelListeners[];
						listenerdata = {#, #["URL"]} & /@ listeners;
						channellistener = With[{id1 = id}, FirstCase[listenerdata, {a_, b_String /; StringMatchQ[b, __ ~~ id1]} :> a]];
						With[{timestamp = ToString[channellistener["FullMessage"]["Timestamp"]]},
							wolframIDAndCellListPairs2 = If[StringQ@timestamp && StringMatchQ[timestamp, "DateObject[" ~~ __],
											{{sender, cellList/.Cell[a__, TaggingRules -> {"CellStatus" -> "Sent", "Sender" -> s_, "TimeStamp" -> _}, b__] :> Cell[a, TaggingRules -> {"CellStatus" -> "Sent", "Sender" -> s, "TimeStamp" -> timestamp}, b]}},
											{{sender, cellList}}]];
						celllist2 = Flatten[transformCellList[#, allparticipants1, ParticipantStyleRules,
											"TimeSpecifier" -> If[type === "PromptedResponse","PromptedResponse",Automatic]] & /@ (If[#[[1]] =!= $WolframID,
																	# /. {InputCellColorBackgroundRule, InputCellColorBackgroundRule2},
															#] & /@ (wolframIDAndCellListPairs2 /. c : Cell[BoxData[_String], "Input", ___] :> AddPasteButtonWithoutBannerStyle@c))];
						celllist3 = If[MatchQ[celllist2, {_, _, Cell[_, _] ..}],
								MapAt[Append[#, "GrayLight"] &, celllist2, List /@ Drop[Range[Length@celllist2], 2]],
								celllist2],
								
						celllist3 = {}];
						
				If[celllist3 =!= {},
					
					cellList4 = fixFrameBox/@Join[transformedCellsFromIndex, celllist3];
					nb = NotebookPut@With[{id1 = id, shortcut1 = shortcut, type1 = type}, Notebook[Append[cellList4, $stemCellTemplate], 
									DockedCells -> (If[compressedBannerCells =!= "None",
												Prepend[Append[DeleteCases[#, CellChangeTimes->_],
														TaggingRules -> {"Banner" -> True}]&/@(Insert[#, "Banner",
													If[MatchQ[#, Cell[_, _String, ___]], 3, 2]] & /@(AddPasteButtonForBanner/@Uncompress@compressedBannerCells)),
													#],
												#]&[ChatNotebookDockedCell[id1, name, "CanSetVisibility" -> False,
															"Preferences" -> ChatClassRoomPreferencesMenu[id1, shortcut1, 
				"Teacher" -> (($WolframID === originator) || (ListQ@teachersadded && MemberQ[teachersadded, $WolframID])),
				"CellLabelsDefault" -> If[type === "PromptedResponse", "On", "Off"]]]]), 
									"TrackCellChangeTimes" -> False, 
									TaggingRules -> {"Originator" -> originator, "ChatNotebook" -> "True", "ScrollLock" -> False, "SelectedCells" -> {},
												"AllParticipants" -> allparticipants1,
												"ChatNotebookID" -> id, "ChatRoom" -> True, "RoomType" -> type, "Banner" -> compressedBannerCells,
											 "ChatNotebookWindowTitle" -> name, "Moderators" -> teachersadded,
											 "ScreenName" -> (If[MemberQ[{"", "None"}, #], 
															"None",
															#]&[CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "ScreenName"}]]),
											"Shortcut" -> shortcut1}, 
									CreateCellID -> True, 
									StyleDefinitions -> FrontEnd`FileName[{"Wolfram"}, "ChatTools.nb"], 
									CellLabelAutoDelete -> False, 
									WindowTitle -> name, If[type1 === "PromptedResponse", ShowCellLabel -> True, Nothing],
                                    Evaluator->"ChatServices"]];
					NotebookFind[nb, "Stem", All, CellStyle, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])];
					SelectionMove[nb, After, Cell, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])];
					With[{nb1 = nb}, SetOptions[nb, 
									NotebookEventActions -> {{"MenuCommand", "SimilarCellBelow"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
									{"MenuCommand", "SaveRename"} :> ChatTools`SaveChatNotebook[nb1],
								If[$VersionNumber < 12, Nothing, {"MenuCommand", "Copy"} :> Module[{in = InputNotebook[]}, 
									If[MatchQ[Developer`CellInformation[in], {___, {"Style" -> {_, __}, __, "CursorPosition" -> "CellBracket", __}, ___}], 
										CopyToClipboard[DeleteCases[ReplaceAll[NotebookRead@in,
						Cell[a_, b_String, _String, ___, TaggingRules -> _, ___] :> Cell[a, b]], Cell[__, "CellUserLabelOptions" | "CellLabelOptions" | "Stem", ___], Infinity]], 
										FrontEndExecute[{FrontEndToken[in, "Copy"]}]]]],
										PassEventsDown -> False}]]]]]]];
									
SetBannerDialog[shortcut_] := 
	Module[{nb = ButtonNotebook[], name, banner}, 
		name = CurrentValue[nb, WindowTitle];
		banner = CurrentValue[nb, {TaggingRules, "Banner"}];
		With[{shortcut1 = shortcut}, 
			NotebookPut[Notebook[If[banner === "None", {}, Uncompress@banner], 
						DockedCells -> {Cell[BoxData@ToBoxes@Grid[{{Spacer[51], 
											Grid[{{ButtonPairSequence[Button["Cancel", NotebookClose[]],
												Button["OK", Needs["ChatTools`"]; NotebookClose[]; ChatTools`SetRoomBanner[shortcut1]]]}}],
											Spacer[51],
											Grid[{{Item[Style["Enter banner cells below.", FontFamily -> "Source Sans Pro", FontSize -> 14], 
											Alignment -> Center]}}]}}, Alignment -> Baseline], "ChatBannerDockedCell", Background -> GrayLevel[.91]]},
						WindowSize -> {900, 350}, 
						ClosingAutoSave -> False,
						WindowTitle -> "Set Banner for " <> name, 
						Saveable -> False,
						WindowFrameElements -> {"CloseBox"}, 
						Background -> White,
						"CellInsertionPointCell" -> {}, 
						"BlinkingCellInsertionPoint" -> False,
						WindowFrame -> "ModalDialog", 
						WindowMargins -> {{Automatic, Automatic}, {Automatic, Automatic}}, 
						TaggingRules -> {"Dialog" -> "SetBanner"}, 
						WindowElements -> {"VerticalScrollBar"},
						Evaluator -> "ChatServices",
						WindowFrame -> "ModalDialog",
						StyleDefinitions -> Notebook[{Cell[StyleData[StyleDefinitions -> "Default.nb"]],
										Cell[StyleData["Notebook"], DefaultNewCellStyle -> "Text"]}]], Evaluator->"ChatServices"]]];
						
Options[SetRoomBanner] = {"Action" -> Automatic};
Default[SetRoomBanner] = {};

SetRoomBanner[shortcut_String, Optional[celllist_List], opts___?OptionQ] := 
	Module[{action = ("Action"/.{opts}/.Options[SetRoomBanner]), RoomData = URLExecute["https://www.wolframcloud.com/objects/5744ac03-463b-4692-ba9b-0f9f27c193af", {"Shortcut" -> Compress@shortcut},
												"WL", Method -> "POST"],
		originator, id, type, name, teachersadded, compressedBannerCells, nb, screenname},
	
	If[Not@TrueQ@$CloudConnected,
	
		FirstLoginMessage[],
	
		If[AssociationQ@RoomData,
		
			originator = RoomData["originator"]; 
			id = RoomData["ID"];
			type = RoomData["RoomType"];
			name = Uncompress@RoomData["Name"]; 
			teachersadded = Uncompress[RoomData["Moderators"]];
			If[If[Not@ListQ@teachersadded, originator === $WolframID, (originator === $WolframID) || MemberQ[teachersadded, $WolframID]],
					compressedBannerCells = Which[action === Automatic, 
									Compress@DeleteCases[NotebookRead[Cells[]], Cell[__, TaggingRules -> {"Insert" -> False}, ___]],
									action === "Manual",
									If[celllist =!= {},
									  Compress[celllist],
									  "None"
									],
									True,
									"None"];
					nb = SelectFirst[Notebooks[], (CurrentValue[#, {TaggingRules, "ChatNotebookID"}] === id) && (CurrentValue[#, {TaggingRules, "ChatRoom"}] === True) &];
					screenname = (If[MemberQ[{"", Inherited}, #] || (StringQ@# && StringMatchQ[#, Whitespace]), "None", #]&[CurrentValue[nb, {TaggingRules, "ScreenName"}]]);
					ChannelSend["https://channelbroker.wolframcloud.com/users/" <> "chatframework@wolfram.com" <> "/" <> id, 
							Association["sender" -> $WolframID, "requesteraddress" -> "None", "allparticipants" -> Compress@{}, "id" -> id, "compressedCellList" -> Compress@{},
									"name" -> Compress@name, "alias" -> $WolframID,
									"originator" -> originator, "SpecialAction" -> "UpdateBanner", "Banner" -> compressedBannerCells, "RoomType" -> type,
									"Moderators" -> Compress@teachersadded, "Shortcut" -> Compress@shortcut, "ScreenName" -> Compress@screenname],
								ChannelPreSendFunction -> None];
					CurrentValue[nb, {TaggingRules, "Banner"}] = compressedBannerCells, 
					Print["You do not have permissions to set the room's banner."]],
					
			MessageDialog["An error occurred while trying to get the chat room data. Try again later.", WindowFrame -> "ModalDialog", WindowSize -> {510, All}]]]];
									
auxiliaryRoomSendMessage["Chat" -> shortcut_]:=
	Module[{shortcutlowercase, roomdata, contacts, id, sel, idIndexPath, roomcontent, originator, compressedCellListUUIDs, type, teachersadded, allparticipants, ParticipantStyleRules,
		wolframIDAndCellListPairs, cellsInNotebook, tk, f, oldestTaken, compressedBannerCells, name, screenname, nb},
		
		shortcutlowercase = ToLowerCase@shortcut;
		Which[Not@TrueQ@$CloudConnected,
	
			FirstLoginMessage[];
			Abort[],
		
			(roomdata = URLExecute["https://www.wolframcloud.com/objects/5744ac03-463b-4692-ba9b-0f9f27c193af", {"Shortcut" -> Compress@shortcutlowercase}, "WL", Method -> "POST"]) === "Failure",
			
			MessageDialog["There is a problem accessing the room index.", WindowFrame -> "ModalDialog", WindowSize -> {310, All}],
			
			roomdata === "Not present",
			
			contacts = CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}];
			If[ListQ@contacts && MemberQ[contacts, shortcutlowercase],
				CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}] = DeleteCases[contacts, shortcutlowercase]];
			cSendMessage["Chat"];
			MessageDialog["That chat room does not exist.", WindowFrame -> "ModalDialog", WindowSize -> {210, All}],
			
			(contacts = CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}];
				 Which[ListQ@contacts && Not@MemberQ[contacts, shortcutlowercase],
				 	CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}] = Union[contacts, {shortcutlowercase}],
				 	Not@ListQ@contacts,
				 	CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}] = {shortcutlowercase}];
				id = roomdata["ID"]; sel = Select[Notebooks[], CurrentValue[#, {TaggingRules, "ChatNotebookID"}] === id &]) =!= {},
			
			SetSelectedNotebook[sel[[1]]],
			
			idIndexPath = "https://www.wolframcloud.com/objects/" <> id;
			roomcontent = CloudGet@CloudObject[idIndexPath];
			Not@AssociationQ@roomcontent,
					
			MessageDialog["An error occurred while trying to get the associated chat room data. Try again later.", WindowFrame -> "ModalDialog", 
								WindowSize -> {510, All}];
			Abort[],
			
			True,
					
			Which[And[KeyExistsQ[roomcontent, "originator"],
				KeyExistsQ[roomcontent, "cellListUUIDs"],
				KeyExistsQ[roomcontent, "RoomType"],
				originator = roomcontent["originator"];
				compressedCellListUUIDs = roomcontent["cellListUUIDs"];
				type = roomcontent["RoomType"];
				teachersadded = Uncompress[roomcontent["Moderators"]];
				MatchQ[compressedCellListUUIDs, {{_String, _String}..}]],
			
				allparticipants = If[MemberQ[#, $WolframID], #, Append[#, $WolframID]]&[Uncompress[roomcontent["allparticipants"]]];
				ParticipantStyleRules = ParticipantStyleRule[allparticipants, #] & /@ allparticipants;
				
				compressedCellListUUIDs = Which[type === "OpenGroup",
								compressedCellListUUIDs,
								Or[($WolframID === originator) && Not@ListQ@teachersadded,
									($WolframID === originator) || (ListQ@teachersadded && MemberQ[teachersadded, $WolframID])],
								(* A list of pairs of the form {wolfram id, cell list uuid} *)compressedCellListUUIDs,
								True,
								Cases[compressedCellListUUIDs, {b:If[ListQ@teachersadded,
													Alternatives @@ Join[teachersadded, {originator, $WolframID}],
													originator | $WolframID], c_} :> {b, c}]];
								
				If[Length@compressedCellListUUIDs <= 1,
				
					wolframIDAndCellListPairs = MapAt[Uncompress@CloudGet@CloudObject["https://www.wolframcloud.com/objects/" <> #] &, #, 2] & /@ compressedCellListUUIDs;
					cellsInNotebook = Append[Flatten[transformCellList[#, allparticipants, ParticipantStyleRules,
										"TimeSpecifier" -> If[type === "PromptedResponse", "PromptedResponse", Automatic]] & /@ (If[#[[1]] =!= $WolframID, # /. {InputCellColorBackgroundRule, InputCellColorBackgroundRule2}, #] & /@wolframIDAndCellListPairs)],
								$stemCellTemplate],
					
					tk = Take[compressedCellListUUIDs, -1];
					wolframIDAndCellListPairs = MapAt[Uncompress@CloudGet@CloudObject["https://www.wolframcloud.com/objects/" <> #] &, #, 2] & /@ tk;
					f = With[{widMessageUUIDPairs = compressedCellListUUIDs}, Button[Overlay[{Graphics[{EdgeForm[{Thin, GrayLevel[.60]}], #, Rectangle[{0, 0}, {3, 1}, RoundingRadius -> 0.2]}, ImageSize -> {80, 30}], 
							Style["Show More...", FontFamily -> "Source Sans Pro", 13, FontColor -> RGBColor["#333333"]]}, Alignment -> Center], ChatTools`InsertMoreRoomCellsFromCloud[widMessageUUIDPairs],
							Method -> "Queued", 
							Appearance -> "Frameless"]&];
					cellsInNotebook = Prepend[Append[Flatten[transformCellList[#, allparticipants, ParticipantStyleRules,
										"TimeSpecifier" -> If[type === "PromptedResponse", "PromptedResponse", Automatic]] & /@ (If[#[[1]] =!= $WolframID, # /. {InputCellColorBackgroundRule, InputCellColorBackgroundRule2}, #] & /@wolframIDAndCellListPairs)], $stemCellTemplate],
									Cell[BoxData[ToBoxes[Mouseover[f[RGBColor["#E5E5E5"]], f[RGBColor["#FFFFFF"]]]]], "More", TaggingRules -> {"OldestTaken" -> tk[[1]]}]];
					oldestTaken = tk[[1]]];
				compressedBannerCells = roomcontent["Banner"],
				
				And[AssociationQ@roomcontent,
					Keys@roomcontent =!= {},
					KeyExistsQ[roomcontent, "cellListUUIDs"],
					MatchQ[roomcontent["cellListUUIDs"], {}]],
				
				originator = roomdata["originator"];
				type = roomdata["RoomType"];
				teachersadded = Uncompress[roomdata["Moderators"]];
				cellsInNotebook = {$stemCellTemplate};
				compressedCellListUUIDs = {}; 
				allparticipants = {$WolframID};
				compressedBannerCells = roomcontent["Banner"],
				
				True(*AssociationQ@roomcontent && (Keys@roomcontent === {})*),
					
				originator = roomdata["originator"];
				type = roomdata["RoomType"];
				teachersadded = Uncompress[roomdata["Moderators"]];
				cellsInNotebook = {$stemCellTemplate};
				compressedCellListUUIDs = {}; 
				allparticipants = {$WolframID};
				compressedBannerCells = "None"];
				
			name = Uncompress[roomdata["Name"]];
			screenname = (If[StringQ@# && Not@StringMatchQ[#, "" | Whitespace], StringTrim@#, "None"] &[CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "ScreenName"}]]);
			
			With[{shortcut1 = shortcutlowercase, id1 = id, type1 = type}, nb = NotebookPut[Notebook[fixFrameBox/@cellsInNotebook, 
							DockedCells -> (If[compressedBannerCells === "None",
										#,
										Prepend[Append[DeleteCases[#, CellChangeTimes->_],
														TaggingRules -> {"Banner" -> True}]&/@(Insert[#, "Banner",
						If[MatchQ[#, Cell[_, _String, ___]], 3, 2]] & /@(AddPasteButtonForBanner/@Uncompress@compressedBannerCells)), #]]&[ChatNotebookDockedCell[id1, name, "CanSetVisibility" -> False,
						"Preferences" -> ChatClassRoomPreferencesMenu[id1, shortcut1, 
				"Teacher" -> (($WolframID === originator) || (ListQ@teachersadded && MemberQ[teachersadded, $WolframID])),
				"CellLabelsDefault" -> If[type === "PromptedResponse", "On", "Off"]]]]),
							"TrackCellChangeTimes" -> False,
							TaggingRules -> {"Originator" -> originator, "ChatNotebook" -> "True", "ScrollLock" -> False, "SelectedCells" -> {}, "AllParticipants" -> allparticipants,
									"ChatNotebookID" -> id, "ChatRoom" -> True, "RoomType" -> type, "ChatNotebookWindowTitle" -> name, "Banner" -> compressedBannerCells,
									"Moderators" -> teachersadded, "Shortcut" -> shortcut1, "ScreenName" -> screenname, "OriginalWolframID" -> $WolframID}, 
							CreateCellID -> True,
							StyleDefinitions -> FrontEnd`FileName[{"Wolfram"}, "ChatTools.nb"], 
							CellLabelAutoDelete -> False, If[type1 === "PromptedResponse", ShowCellLabel -> True, Nothing],
							WindowTitle -> name, Evaluator->"ChatServices"], Evaluator->"ChatServices"]];
			NotebookFind[nb, "Stem", All, CellStyle, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])];
			SelectionMove[nb, After, Cell, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])];
			With[{nb1 = nb}, SetOptions[nb, 
				NotebookEventActions -> {{"MenuCommand", "SimilarCellBelow"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
							{"MenuCommand", "SaveRename"} :> ChatTools`SaveChatNotebook[nb1],
								If[$VersionNumber < 12, Nothing, {"MenuCommand", "Copy"} :> Module[{in = InputNotebook[]}, 
									If[MatchQ[Developer`CellInformation[in], {___, {"Style" -> {_, __}, __, "CursorPosition" -> "CellBracket", __}, ___}], 
										CopyToClipboard[DeleteCases[ReplaceAll[NotebookRead@in,
						Cell[a_, b_String, _String, ___, TaggingRules -> _, ___] :> Cell[a, b]], Cell[__, "CellUserLabelOptions" | "CellLabelOptions" | "Stem", ___], Infinity]], 
										FrontEndExecute[{FrontEndToken[in, "Copy"]}]]]],
							PassEventsDown -> False}]];
			If[Not@MemberQ[#["ChannelObject"] & /@ ChannelListeners[], #],
				ChannelListen[#, "TrustedChannel" -> True]]&[ChannelObject["https://channelbroker.wolframcloud.com/users/" <> "chatframework@wolfram.com" <> "/" <> id]];
  			If[Not@StringQ@FirstCase[#["URL"] & /@ ChannelListeners[], a_String /; StringMatchQ[a, StringExpression[__, "/", $WolframID, "/", "ChatInvitations"]]],
				ChannelListen[ChannelObject[$WolframID <> ":ChatInvitations"], "TrustedChannel" -> True]]]];
				
Options[InsertMoreRoomCellsFromCloud] = {"AddOldestInList" -> False};

InsertMoreRoomCellsFromCloud[widMessageUUIDPairs_, opts___?OptionQ]:= InsertMoreRoomCellsFromCloud[ButtonNotebook[], widMessageUUIDPairs, opts];

InsertMoreRoomCellsFromCloud[nb_, widMessageUUIDPairs_, opts___?OptionQ] := 
	Module[{addOldestInList = ("AddOldestInList" /. {opts} /. Options[InsertMoreRoomCellsFromCloud]), cellObject = Cells[nb, CellStyle -> "More"], oldestTaken, compressedCellListUUIDs,
		oldCompressedCellListUUIDsNotGotten, allparticipants, ParticipantStyleRules, wolframIDAndCellListPairs, type = CurrentValue[nb, {TaggingRules, "RoomType"}], f, cellsToAddInNotebook}, 
		
		oldestTaken = If[cellObject =!= {}, CurrentValue[cellObject[[1]], {TaggingRules, "OldestTaken"}], If[Length@widMessageUUIDPairs > 5, widMessageUUIDPairs[[-5]], Inherited]];
		
		compressedCellListUUIDs = widMessageUUIDPairs;
									
		oldCompressedCellListUUIDsNotGotten = If[cellObject === {},
								compressedCellListUUIDs,
								If[addOldestInList === True, compressedCellListUUIDs, compressedCellListUUIDs[[1;;Position[compressedCellListUUIDs, oldestTaken][[1, 1]] - 1]]]];
		allparticipants = CurrentValue[nb, {TaggingRules, "AllParticipants"}]; 
		ParticipantStyleRules = ParticipantStyleRule[allparticipants, #] & /@ allparticipants; 
		If[Length@oldCompressedCellListUUIDsNotGotten <= 5(*1*),
					
			(*If[cellObject =!= {}, NotebookDelete[cellObject[[1]]]];*)
			wolframIDAndCellListPairs = MapAt[Uncompress@CloudGet@CloudObject["https://www.wolframcloud.com/objects/" <> #] &, #, 2] & /@ oldCompressedCellListUUIDsNotGotten;
			cellsToAddInNotebook = Flatten[transformCellList[#, allparticipants, ParticipantStyleRules,
							"TimeSpecifier" -> If[type === "PromptedResponse", "PromptedResponse", Automatic]] & /@ (If[#[[1]] =!= $WolframID, # /. {InputCellColorBackgroundRule, InputCellColorBackgroundRule2}, #] & /@wolframIDAndCellListPairs)],
						
			oldestTaken = compressedCellListUUIDs[[If[oldestTaken === Inherited,
									0,
									If[addOldestInList === True, Length@compressedCellListUUIDs, Position[compressedCellListUUIDs, oldestTaken][[1, 1]] - 5(*1*)]]]];
			wolframIDAndCellListPairs = MapAt[Uncompress@CloudGet@CloudObject["https://www.wolframcloud.com/objects/" <> #] &, #, 2] & /@ Take[oldCompressedCellListUUIDsNotGotten, -5(*1*)];
			f = With[{widMessageUUIDPairs1 = widMessageUUIDPairs}, Button[Overlay[{Graphics[{EdgeForm[{Thin, GrayLevel[.60]}], #, Rectangle[{0, 0}, {3, 1}, RoundingRadius -> 0.2]}, ImageSize -> {80, 30}], 
										Style["Show More...", FontFamily -> "Source Sans Pro", 13, FontColor -> RGBColor["#333333"]]}, Alignment -> Center], ChatTools`InsertMoreRoomCellsFromCloud[widMessageUUIDPairs1],
										Method -> "Queued", 
				Appearance -> "Frameless"]&];
			cellsToAddInNotebook = Prepend[Flatten[transformCellList[#, allparticipants, ParticipantStyleRules,
							"TimeSpecifier" -> If[type === "PromptedResponse", "PromptedResponse", Automatic]] & /@ (If[#[[1]] =!= $WolframID, # /. {InputCellColorBackgroundRule, InputCellColorBackgroundRule2}, #] & /@wolframIDAndCellListPairs)], 
							Cell[BoxData[ToBoxes[Mouseover[f[RGBColor["#E5E5E5"]], f[RGBColor["#FFFFFF"]]]]], "More", TaggingRules -> {"OldestTaken" -> oldestTaken}]]]; 
						
		If[cellObject === {},
					
			SelectionMove[nb, Before, Notebook, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])];
			NotebookWrite[nb, fixFrameBox/@cellsToAddInNotebook, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])];
			NotebookFind[nb, "Stem", All, CellStyle, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])];
			SelectionMove[nb, After, Cell, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])],

			SetOptions[cellObject[[1]], Deletable -> True];
			NotebookWrite[cellObject[[1]], fixFrameBox/@cellsToAddInNotebook, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])]];
						
		If[Length@oldCompressedCellListUUIDsNotGotten <= 5(*1*), ChatTools`Private`StopRestoringMessages = True]];

InsertAllRoomCellsFromCloud[] := 
	Module[{nb = ButtonNotebook[], RoomData, type, originator, teachersadded, compressedCellListUUIDs, allparticipants, ParticipantStyleRules, wolframIDAndCellListPairs, cellsInNotebook, tk, f, st,
		progressNB},
		
		If[Cells[nb, CellStyle -> "More"] === {} && Cells[nb, CellStyle -> "CellLabelOptions"] =!= {}, ClearSentCells[]];
	
		If[Quiet[RoomData = CloudGet@CloudObject["https://www.wolframcloud.com/objects/" <> CurrentValue[nb, {TaggingRules, "ChatNotebookID"}]], CloudObject::notperm] === $Failed,
			
			MessageDialog["An error occurred while trying to get the chat room data. Try again later.", WindowFrame -> "ModalDialog", WindowSize -> {510, All}],
			
			If[RoomData === Association[],
			
				MessageDialog["There are no messages to download.", WindowFrame -> "ModalDialog", WindowSize -> {260, All}],
     
				type = CurrentValue[nb, {TaggingRules, "RoomType"}];
				originator = CurrentValue[nb, {TaggingRules, "Originator"}];
				teachersadded = CurrentValue[nb, {TaggingRules, "Moderators"}]; 
				compressedCellListUUIDs = RoomData["cellListUUIDs"];
     
				compressedCellListUUIDs = Which[type === "OpenGroup",
									compressedCellListUUIDs, 
									Or[($WolframID === originator) && Not@ListQ@teachersadded,
										($WolframID === originator) || (ListQ@teachersadded && MemberQ[teachersadded, $WolframID])],
										(*A list of pairs of the form {wolfram id,cell list uuid}*)
									compressedCellListUUIDs,
									True, 
									Cases[compressedCellListUUIDs, {b : If[ListQ@teachersadded, Alternatives @@ Join[teachersadded, {originator, $WolframID}], 
																			originator | $WolframID], c_} :> {b, c}]];
     
				ChatTools`Private`StopRestoringMessages = False;
				ChatTools`Private`pos = 1;
				ChatTools`Private`len = Length@compressedCellListUUIDs;
				
				If[Cells[nb, CellStyle -> "More"] ==={} && compressedCellListUUIDs =!={}, 
				
					allparticipants = CurrentValue[nb, {TaggingRules, "AllParticipants"}]; 
					ParticipantStyleRules = ParticipantStyleRule[allparticipants, #] & /@ allparticipants;
					If[Length@compressedCellListUUIDs <= 1,
								
						wolframIDAndCellListPairs = MapAt[Uncompress@CloudGet@CloudObject["https://www.wolframcloud.com/objects/" <> #] &, #, 2] & /@ compressedCellListUUIDs;
						cellsInNotebook = Append[Flatten[transformCellList[#, allparticipants, ParticipantStyleRules,
												"TimeSpecifier" -> If[type === "PromptedResponse", "PromptedResponse", Automatic]] & /@ (If[#[[1]] =!= $WolframID, # /. {InputCellColorBackgroundRule, InputCellColorBackgroundRule2}, #] & /@wolframIDAndCellListPairs)],
									$stemCellTemplate],
									
						tk = Take[compressedCellListUUIDs, -1];
						wolframIDAndCellListPairs = MapAt[Uncompress@CloudGet@CloudObject["https://www.wolframcloud.com/objects/" <> #] &, #, 2] & /@ tk;
						f = With[{widMessageUUIDPairs = compressedCellListUUIDs}, Button[Overlay[{Graphics[{EdgeForm[{Thin, GrayLevel[.60]}], #, Rectangle[{0, 0}, {3, 1}, RoundingRadius -> 0.2]}, ImageSize -> {80, 30}], 
							Style["Show More...", FontFamily -> "Source Sans Pro", 13, FontColor -> RGBColor["#333333"]]}, Alignment -> Center], ChatTools`InsertMoreRoomCellsFromCloud[widMessageUUIDPairs],
							Method -> "Queued", 
							Appearance -> "Frameless"]&];
						cellsInNotebook = Prepend[Append[Flatten[transformCellList[#, allparticipants, ParticipantStyleRules,
														"TimeSpecifier" -> If[type === "PromptedResponse", "PromptedResponse", Automatic]] & /@ (If[#[[1]] =!= $WolframID, # /. {InputCellColorBackgroundRule, InputCellColorBackgroundRule2}, #] & /@wolframIDAndCellListPairs)], $stemCellTemplate],
													Cell[BoxData[ToBoxes[Mouseover[f[RGBColor["#E5E5E5"]], f[RGBColor["#FFFFFF"]]]]], "More", TaggingRules -> {"OldestTaken" -> tk[[1]]}]]];
													
					st = Cells[nb, CellStyle -> "Stem"];
					If[st =!= {},
						SetOptions[st[[-1]], Deletable -> True];
						NotebookWrite[st[[1]], cellsInNotebook, 
							AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])]]];
				
			Which[1 < Length@compressedCellListUUIDs < 7,
			
				ChatTools`InsertMoreRoomCellsFromCloud[nb, compressedCellListUUIDs],
				
				7 <= Length@compressedCellListUUIDs,
				
				progressNB = NotebookPut@Notebook[{Cell[BoxData@ToBoxes@Grid[{{"", "", ""},
								{"", Style["Downloading all saved messages for this\nchat room from the Wolfram Cloud.", TextAlignment -> Center], ""},
								{"", Dynamic@Grid[{{Row[{ChatTools`Private`pos,"/",ChatTools`Private`len}],
											ProgressIndicator[ChatTools`Private`pos, {0, ChatTools`Private`len}]}}], ""},
								{"", "", ""},
								{"", Button["Cancel",ChatTools`Private`StopRestoringMessages = True; NotebookClose[], ImageSize -> Automatic], ""},
								{"", "", ""}}, BaseStyle -> {FontFamily -> "Source Sans Pro", FontSize -> 16}, 
							Spacings -> {{2 -> 1, 3 -> 1}, {2 -> 1, 3 -> 2, 4 -> 1}}, Alignment -> Center], "Text",
								CellMargins -> {{0, 0}, {0, 0}}, ShowCellBracket -> False, ShowStringCharacters -> False]},
							WindowMargins -> Automatic,
							WindowSize -> Fit,
							Evaluator -> "ChatServices",
							Background -> GrayLevel[1], 
							WindowTitle -> None,
							WindowFrameElements -> {},
							WindowElements -> {},
							"CellInsertionPointCell" -> {}, 
							"BlinkingCellInsertionPoint" -> False, 
							"CellInsertionPointColor" -> GrayLevel[1],
							Saveable -> False,
							Editable -> False, 
							Selectable -> False];
				While[ChatTools`Private`pos <= ChatTools`Private`len,
					If[Not@TrueQ@ChatTools`Private`StopRestoringMessages,
						ChatTools`InsertMoreRoomCellsFromCloud[nb, compressedCellListUUIDs];
						If[ChatTools`Private`pos === ChatTools`Private`len, 
							NotebookClose[progressNB],
 							ChatTools`Private`pos = If[ChatTools`Private`len - ChatTools`Private`pos < 5, 
											ChatTools`Private`len,
											ChatTools`Private`pos + 5]],
						NotebookClose[progressNB];
						ChatTools`Private`pos = ChatTools`Private`len + 1]]]]]];
						
ChatTools`CreateChatRoom::incorcred = "The login credentials you typed are incorrect.";
ChatTools`CreateChatRoom::mustcc = "Chat functionality requires connecting to the Wolfram Cloud.";
ChatTools`CreateChatRoom::empt = "Both the short name and title must be nonempty strings.";
ChatTools`CreateChatRoom::inter = "An internet connection is required to use Chat Services.";
ChatTools`CreateChatRoom::indacc = "There is a problem accessing cloud data for this chat. Check your network and Wolfram Cloud connections and try again.";
ChatTools`CreateChatRoom::chan = "Could not create the channel for this chat. Check your network and Wolfram Cloud connections and try again.";
ChatTools`CreateChatRoom::chanbro = "Could not connect the channel broker to this chat. Check your network and Wolfram Cloud connections and try again.";
ChatTools`CreateChatRoom::api = "Could not deploy the API for sending or receiving messages in this chat. Check your network and Wolfram Cloud connections and try again.";
ChatTools`CreateChatRoom::rmshex = "A chat room with the same short name already exists.";
ChatTools`CreateChatRoom::maxcr = "Your login can only have at most 10 chat rooms associated with it.";
ChatTools`CreateChatRoom::notcr = "Unable to create the chat room at this time. Try again later.";
ChatTools`CreateChatRoom::usage = "CreateChatRoom[\"shortname\", \"title\"] creates a chat room with the specified short name and title.";

Options[ChatTools`CreateChatRoom] = {"RoomType" -> "OpenGroup", "Moderators" -> None};

CreateChatRoom[shortname_String, title_String, opts___?OptionQ] := 
	Module[{i = 0, s},
		If[Not@TrueQ@CurrentValue["InternetConnectionAvailable"] || Not@TrueQ@CurrentValue["AllowDownloads"],
		
			Message[CreateChatRoom::inter],
			
			If[Not@$CloudConnected,
			
				Quiet[Catch[Do[s = CloudConnect[];
						Switch[s, $Canceled, Throw["Close"], i++; $Failed, "Again", _String, Throw["Proceed"]], {3}]];
					If[s =!= $Canceled,
					
						If[i === 3 && s === $Failed,
						
							Message[CreateChatRoom::incorcred],
							
							auxiliaryCreateChatRoom[shortname, title, opts]],
							
					Message[CreateChatRoom::mustcc]],
					{CloudConnect::creds, CloudConnect::notauth, General::stop}],
					
				auxiliaryCreateChatRoom[shortname, title, opts]]]];
Attributes[ChatTools`CreateChatRoom] = {Protected, ReadProtected};
    
Options[auxiliaryCreateChatRoom] = {"RoomType" -> "OpenGroup", "Moderators" -> None};

auxiliaryCreateChatRoom[shortname_String, title_String, opts___?OptionQ] :=
	Module[{shortname1, title1, i = 0, s, type = ("RoomType" /. {opts} /. Options[auxiliaryCreateChatRoom]), teachersAdded = ("Moderators" /. {opts} /. Options[auxiliaryCreateChatRoom]),
		shortnameAvailable, contacts},
		shortname1 = StringTrim@ToLowerCase@shortname;
  		title1 = StringTrim@title;
  		If[StringMatchQ[shortname1, "" | Whitespace] || StringMatchQ[title1, "" | Whitespace],
			
			Message[CreateChatRoom::empt],

			shortnameAvailable = URLExecute[CloudObject["https://www.wolframcloud.com/objects/eeb3eb02-d951-46cd-a092-f801f43e782d"], {"Shortcut" -> Compress@shortname1}, "WL"];
			Which[shortnameAvailable === "Your login can only have at most 10 chat rooms associated with it.",
				Message[CreateChatRoom::maxcr],
				Not[(shortnameAvailable === True) || (Head@shortnameAvailable === CloudObject)], 
				Message[CreateChatRoom::indacc],
				Head@shortnameAvailable === CloudObject, 
				Message[CreateChatRoom::rmshex]; shortnameAvailable,
				True, 
				With[{wolframid = $WolframID, shortname2 = shortname1, title2 = title1, type1 = type, teachersAdded1 = Compress@teachersAdded}, 
					Which[# === "Could not create the channel for this room.",
						Message[CreateChatRoom::chan],
						# === "Could not make the channel broker action function for this room.",
						Message[CreateChatRoom::chanbro],
						# === "Could not cloud deploy the api function for modifying this room's index.",
						Message[CreateChatRoom::api],
						# === "There is a problem accessing the room index.",
						Message[CreateChatRoom::indacc],
						StringQ@#,
						Message[CreateChatRoom::notcr],
						True,
						contacts = CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}];
						Which[contacts === Inherited,
							CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}] = {shortname2},
							ListQ@contacts,
							CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}] = Append[contacts, shortname2]];
  						#]&[URLExecute[CloudObject["https://www.wolframcloud.com/objects/8b699812-11d1-4297-8469-e0050c2b1f00"],
							{"Creator" -> wolframid, "ShortCut" -> Compress@shortname2, "Name" -> Compress@title2, "RoomType" -> type1, "Moderators" -> teachersAdded1}, "WL",
								Method -> "POST"]]]]]];
Attributes[ChatTools`Private`auxiliaryCreateChatRoom] = {Protected, ReadProtected};

ChatTools`DeleteChatRoom::incorcred = "The login credentials you typed are incorrect.";
ChatTools`DeleteChatRoom::mustcc = "Chat functionality requires connecting to the Wolfram Cloud.";
ChatTools`DeleteChatRoom::inter = "An internet connection is required to use Chat Services.";
ChatTools`DeleteChatRoom::notex = "A chat room with that short name does not exist.";
ChatTools`DeleteChatRoom::notper = "You do not have permission to delete that chat room.";
ChatTools`DeleteChatRoom::usage = "DeleteChatRoom[\"shortname\"] deletes the chat room with the specified short name.";
DeleteChatRoom[shortname_String] := 
	Module[{i = 0, s},
		If[Not@TrueQ@CurrentValue["InternetConnectionAvailable"] || Not@TrueQ@CurrentValue["AllowDownloads"],
		
			Message[DeleteChatRoom::inter],
			
			If[Not@$CloudConnected,
			
				Quiet[Catch[Do[s = CloudConnect[];
						Switch[s, $Canceled, Throw["Close"], i++; $Failed, "Again", _String, Throw["Proceed"]], {3}]];
					If[s =!= $Canceled,
					
						If[i === 3 && s === $Failed,
						
							Message[DeleteChatRoom::incorcred],
							
							auxiliaryDeleteChatRoom[shortname]],
							
					Message[DeleteChatRoom::mustcc]],
					{CloudConnect::creds, CloudConnect::notauth, General::stop}],
					
				auxiliaryDeleteChatRoom[shortname]]]];
Attributes[ChatTools`DeleteChatRoom] = {Protected, ReadProtected};

auxiliaryDeleteChatRoom[shortname_String] := 
	Module[{shortnameAvailable, rem, contacts, shortname1 = StringTrim@ToLowerCase@shortname},
		shortnameAvailable = URLExecute[CloudObject["https://www.wolframcloud.com/objects/eeb3eb02-d951-46cd-a092-f801f43e782d"], {"Shortcut" -> Compress@shortname1}, "WL"];
			If[shortnameAvailable === True,
			
				Message[ChatTools`DeleteChatRoom::notex],
		
				rem = URLExecute[CloudObject["https://www.wolframcloud.com/objects/71caf01b-b555-4cd1-a450-99f34bb8b421"], {"Shortcut" -> Compress@shortname1}, "WL", Method -> "POST"];
				If[rem === "You do not have permissions to delete this chat room.",
				
					Message[ChatTools`DeleteChatRoom::notper],
					
					contacts = CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}];
					Which[contacts === {shortname1},
						CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}] = Inherited,
						ListQ@contacts,
						CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}] = DeleteCases[contacts, shortname1]]];]];
Attributes[ChatTools`Private`auxiliaryDeleteChatRoom] = {Protected, ReadProtected};


ChatTools`ChatRoomModerators::incorcred = "The login credentials you typed are incorrect.";
ChatTools`ChatRoomModerators::mustcc = "Chat functionality requires connecting to the Wolfram Cloud.";
ChatTools`ChatRoomModerators::inter = "An internet connection is required to use Chat Services.";
ChatTools`ChatRoomModerators::wcp = "There is a problem getting data from the Wolfram Cloud. Try again later.";
ChatTools`ChatRoomModerators::ndne = "A chat room with that short name does not exist.";
ChatTools`ChatRoomModerators::nmr = "Your Wolfram ID is not a moderator for this chat room.";
ChatTools`ChatRoomModerators::usage = "ChatRoomModerators[shortname] returns the list of moderators for the chat room having short name shortname. One must be a room moderator to get the moderator list.";
ChatRoomModerators[shortname_String /; Not@StringMatchQ[shortname, "" | Whitespace]]:=
	Module[{i = 0, s},
		If[Not@TrueQ@CurrentValue["InternetConnectionAvailable"] || Not@TrueQ@CurrentValue["AllowDownloads"],
		
			Message[ChatRoomModerators::inter],
			
			If[Not@$CloudConnected,
			
				Quiet[Catch[Do[s = CloudConnect[];
						Switch[s, $Canceled, Throw["Close"], i++; $Failed, "Again", _String, Throw["Proceed"]], {3}]];
					If[s =!= $Canceled,
					
						If[i === 3 && s === $Failed,
						
							Message[ChatRoomModerators::incorcred],
							
							auxiliaryChatRoomModerators[shortname]],
							
					Message[ChatRoomModerators::mustcc]],
					{CloudConnect::creds, CloudConnect::notauth, General::stop}],
					
				auxiliaryChatRoomModerators[shortname]]]];
Attributes[ChatTools`ChatRoomModerators] = {Protected, ReadProtected};

auxiliaryChatRoomModerators[shortname_] :=
	Module[{shortname1 = Compress@ToLowerCase@StringTrim@shortname, data, message = "There is a problem accessing the room index.", message1 = "A chat room with that short name does not exist.", 
		message3 = "Your Wolfram ID is not a moderator for this chat room."},
		data = URLExecute[CloudObject["https://www.wolframcloud.com/objects/c8262397-5f5c-4f32-9e27-672819b7605b"], {"ShortName" -> shortname1, "wid" -> $WolframID}, "WL"];
		Switch[data,
			message,
			Message[ChatRoomModerators::wcp],
			message1,
			Message[ChatRoomModerators::ndne],
			message3,
			Message[ChatRoomModerators::nmr],
			_,
			data]];
Attributes[ChatTools`Private`auxiliaryChatRoomModerators] = {Protected, ReadProtected};

ChatTools`SetChatRoomModerators::incorcred = "The login credentials you typed are incorrect.";
ChatTools`SetChatRoomModerators::mustcc = "Chat functionality requires connecting to the Wolfram Cloud.";
ChatTools`SetChatRoomModerators::inter = "An internet connection is required to use Chat Services.";
ChatTools`SetChatRoomModerators::wcp = "There is a problem getting data from the Wolfram Cloud. Try again later.";
ChatTools`SetChatRoomModerators::ndne = "A chat room with that short name does not exist.";
ChatTools`SetChatRoomModerators::nmr = "Your Wolfram ID is not a moderator for this chat room.";
ChatTools`SetChatRoomModerators::ndp = "The suggested change to the list of moderators is not different from what is already present.";
ChatTools`SetChatRoomModerators::usage = "SetChatRoomModerators[shortname, newmoderatorslist] updates the list of moderators for the chat room having short name shortname. One must be a room moderator to modify a chat room's moderators list.";
SetChatRoomModerators[shortname_String /; Not@StringMatchQ[shortname, "" | Whitespace],
		moderators_ /; (VectorQ[moderators, StringQ] && AllTrue[moderators, TextCases[StringTrim@#, "EmailAddress"] === {StringTrim@#} &]) || moderators === None] :=
	Module[{i = 0, s},
		If[Not@TrueQ@CurrentValue["InternetConnectionAvailable"] || Not@TrueQ@CurrentValue["AllowDownloads"],
		
			Message[SetChatRoomModerators::inter],
			
			If[Not@$CloudConnected,
			
				Quiet[Catch[Do[s = CloudConnect[];
						Switch[s, $Canceled, Throw["Close"], i++; $Failed, "Again", _String, Throw["Proceed"]], {3}]];
					If[s =!= $Canceled,
					
						If[i === 3 && s === $Failed,
						
							Message[SetChatRoomModerators::incorcred],
							
							auxiliarySetChatRoomModerators[shortname, moderators]],
							
					Message[SetChatRoomModerators::mustcc]],
					{CloudConnect::creds, CloudConnect::notauth, General::stop}],
					
				auxiliarySetChatRoomModerators[shortname, moderators]]]];
Attributes[ChatTools`SetChatRoomModerators] = {Protected, ReadProtected};		
		
auxiliarySetChatRoomModerators[shortname_, moderators_] :=		
	Module[{shortname1 = Compress@ToLowerCase@StringTrim@shortname, moderators1 = Compress@Which[moderators === {}, None, VectorQ[moderators, StringQ], StringTrim /@ moderators, True, moderators],
		data, originator, id, type, name, screenname}, 
			data = URLExecute["https://www.wolframcloud.com/objects/33bf668d-0d71-4c00-87c6-12093615690a", {"Shortcut" -> shortname1, "wid" -> $WolframID}, "WL", Method -> "POST"];
			Which[data === "Failure",
				Message[SetChatRoomModerators::wcp], 
				data === "Not present",
				Message[SetChatRoomModerators::ndne], 
				data === "Your Wolframid is not a moderator for this chat room.", 
				Message[SetChatRoomModerators::nmr], 
				Complement[Uncompress[data["Moderators"]] /. None -> {}, {data["originator"]}] === Complement[moderators /. None -> {}, {data["originator"]}], 
				Message[SetChatRoomModerators::ndp],
				True, 
				originator = data["originator"];
				id = data["ID"]; 
				type = data["RoomType"];
				name = Uncompress@data["Name"];
				screenname = CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "ScreenName"}]; 
				ChannelSend["https://channelbroker.wolframcloud.com/users/" <> "chatframework@wolfram.com" <> "/" <> id, 
						Association["sender" -> $WolframID, "requesteraddress" -> "None", "allparticipants" -> Compress@{},"id" -> id, "compressedCellList" -> Compress@{}, 
							"name" -> Compress@name, "alias" -> $WolframID, "originator" -> originator, "SpecialAction" -> "UpdateModerators", "Banner" -> Compress@{},
							"RoomType" -> type, "Moderators" -> moderators1, "Shortcut" -> shortname1, "ScreenName" -> Compress@screenname],
					ChannelPreSendFunction -> None]]];
Attributes[ChatTools`Private`auxiliarySetChatRoomModerators] = {Protected, ReadProtected};

ChatTools`ChatRooms::incorcred = "The login credentials you typed are incorrect.";
ChatTools`ChatRooms::mustcc = "Chat functionality requires connecting to the Wolfram Cloud.";
ChatTools`ChatRooms::inter = "An internet connection is required to use Chat Services.";
ChatTools`ChatRooms::wcp = "There is a problem getting data from the Wolfram Cloud. Try again later.";
ChatTools`ChatRooms::usage = "ChatRooms[] returns a list of pairs of the form {short name, room title}.";
ChatRooms[] := 
	Module[{i = 0, s},
		If[Not@TrueQ@CurrentValue["InternetConnectionAvailable"] || Not@TrueQ@CurrentValue["AllowDownloads"],
		
			Message[ChatRooms::inter],
			
			If[Not@$CloudConnected,
			
				Quiet[Catch[Do[s = CloudConnect[];
						Switch[s, $Canceled, Throw["Close"], i++; $Failed, "Again", _String, Throw["Proceed"]], {3}]];
					If[s =!= $Canceled,
					
						If[i === 3 && s === $Failed,
						
							Message[ChatRooms::incorcred],
							
							auxiliaryChatRooms[]],
							
					Message[ChatRooms::mustcc]],
					{CloudConnect::creds, CloudConnect::notauth, General::stop}],
					
				auxiliaryChatRooms[]]]];
Attributes[ChatTools`ChatRooms] = {Protected, ReadProtected};

auxiliaryChatRooms[] := 
	Module[{data = URLExecute[CloudObject["https://www.wolframcloud.com/objects/715951b3-1bbd-49ba-aa39-2c56aae50531"], {"wid" -> $WolframID}, "WL"]},
		If[StringQ@data,
			Message[ChatRooms::wcp],
			data]];
Attributes[ChatTools`Private`auxiliaryChatRooms] = {Protected, ReadProtected};
									
(********** End code for general chat rooms *******************)
		

sendPrivateChatMessage[requester_, msg_ /; And[KeyExistsQ[ReleaseHold@msg, "ChatNotebookID"], KeyExistsQ[ReleaseHold@msg,"addedParticipants"],
					KeyExistsQ[ReleaseHold@msg, "removedParticipants"], KeyExistsQ[ReleaseHold@msg, "wid"], KeyExistsQ[ReleaseHold@msg, "allparticipants"],
					KeyExistsQ[ReleaseHold@msg, "screenname"],
		Sort@Keys@ReleaseHold@msg === {"addedParticipants", "allparticipants", "ChatNotebookID", "removedParticipants", "screenname", "wid"}]] := ModifyParticipantsInThread[requester,
			Lookup[ReleaseHold@msg, "ChatNotebookID"], Uncompress@Lookup[ReleaseHold@msg, "addedParticipants"],
			Uncompress@Lookup[ReleaseHold@msg, "removedParticipants"], Lookup[ReleaseHold@msg, "wid"], Uncompress@Lookup[ReleaseHold@msg, "allparticipants"],
			Uncompress@Lookup[ReleaseHold@msg, "screenname"]];
			
ModifyParticipantsInThreadWrite[allparticipants_, wid_, co_, text_, scrolllockQ_:False]:=
	Module[{pos = (If[MatchQ[#, {{_}}], #[[1, 1]], 1]&[Position[allparticipants, wid]]), senderstyle, styleToUse},
		senderstyle = $sentStyles[[If[# === 0, -1, #] &[Mod[pos, Length@$sentStyles]]]];
		styleToUse = "GrayLight" <> senderstyle <> "Top";
		NotebookWrite[co, {WolframIDandTimeCell[senderstyle, wid, StringReplace[StringReplace[DateString["ISODateTime"], __ ~~ "T" -> ""], a__ ~~ ":" ~~ b__ ~~ ":" ~~ __ :> a <> ":" <> b]], 
					Cell[text, "Text", styleToUse], $stemCellTemplate}, AutoScroll -> (!TrueQ[scrolllockQ])]];
					
ModifyParticipantsInThreadWriteWithoutStemCell[allparticipants_, wid_, co_, text_, scrolllockQ_:False]:=
	Module[{pos = (If[MatchQ[#, {{_}}], #[[1, 1]], 1]&[Position[allparticipants, wid]]), senderstyle, styleToUse},
		senderstyle = $sentStyles[[If[# === 0, -1, #] &[Mod[pos, Length@$sentStyles]]]];
		styleToUse = "GrayLight" <> senderstyle <> "Top";
		NotebookWrite[co, {WolframIDandTimeCell[senderstyle, wid, StringReplace[StringReplace[DateString["ISODateTime"], __ ~~ "T" -> ""], a__ ~~ ":" ~~ b__ ~~ ":" ~~ __ :> a <> ":" <> b]], 
					Cell[text, "Text", styleToUse]}, AutoScroll -> (!TrueQ[scrolllockQ])]];

ModifyParticipantsInThread[requester_, id_, addedParticipants_List, removedParticipants_List, wid_, allparticipants_, screenname_] := 
	Module[{sel = SelectFirst[Notebooks[], (CurrentValue[#, {TaggingRules, "ChatNotebookID"}] === id) &], newparticipants, sentStyles = {"ChatLightBlue", "ChatLightOrange",
		"ChatLightRed", "ChatLightGreen", "ChatLightMagenta", "ChatLightBrown", "ChatLightCyan", "ChatLightPink", "ChatLightPurple", "ChatLightGray"}, pos, secondarystyle, st, rem, add}, 
		Which[And[MemberQ[removedParticipants, $WolframID], wid =!= $WolframID, sel =!= Missing["NotFound"]],
		
			CurrentValue[sel, {TaggingRules, "ParticipantRemoved"}] = True;
			CurrentValue[sel, {TaggingRules, "ViewParticipants"}] = Inherited;
			CurrentValue[sel, WindowTitle] = "Chat";
			CurrentValue[sel, NotebookDynamicExpression] = Inherited;
			CurrentValue[sel, DockedCells] = Inherited;
			CurrentValue[sel, {TaggingRules, "ChatNotebookID"}] = Inherited;
			CurrentValue[sel, {TaggingRules, "AllParticipants"}] = allparticipants;
			st = Cells[sel, CellStyle -> "Stem"];
			If[st =!= {},
				SetOptions[st[[1]], Deletable -> True];
				ModifyParticipantsInThreadWriteWithoutStemCell[allparticipants, If[screenname === "None", wid, screenname], st[[1]], "You have been removed from this notebook's chat thread.",
																		CurrentValue[sel, {TaggingRules, "ScrollLock"}, False]];
				CurrentValue[sel, {TaggingRules, "Terminated"}] = True];
			SetOptions[sel, NotebookDynamicExpression -> Inherited, NotebookEventActions -> Inherited, "BlinkingCellInsertionPoint" -> Inherited, "CellInsertionPointColor" -> Inherited, 
					"CellInsertionPointCell" -> Inherited];
			If[ListQ[#], If[Head@# === CellObject,
						NotebookDelete[#, AutoScroll -> (! CurrentValue[sel, {TaggingRules, "ScrollLock"}, False])]] & /@ #] &[CurrentValue[sel, {TaggingRules, "SelectedCells"}]],
							
			And[Not@MemberQ[removedParticipants, $WolframID], Or[addedParticipants =!= {}, removedParticipants =!= {}], (*wid =!= $WolframID,*) sel =!= Missing["NotFound"]],
			
			newparticipants = DeleteDuplicates@Join[Complement[CurrentValue[sel, {TaggingRules, "Participants"}], removedParticipants], addedParticipants];
			CurrentValue[sel, {TaggingRules, "Participants"}] = newparticipants;
			If[addedParticipants =!= {}, CurrentValue[sel, {TaggingRules, "AllParticipants"}] = Join[CurrentValue[sel, {TaggingRules, "AllParticipants"}], addedParticipants]];
			st = Cells[sel, CellStyle -> "Stem"];
			If[st =!= {},
				SetOptions[st[[1]], Deletable -> True];
				rem = If[Length@removedParticipants === 0,
						"", 
						StringJoin[Which[Length@removedParticipants === 1, 
									removedParticipants[[1]] <> " has", 
									Length@removedParticipants === 2, 
									removedParticipants[[1]] <> " and " <> removedParticipants[[2]] <> " have",
									True, 
									StringJoin[StringJoin @@ Riffle[Take[removedParticipants, Length@removedParticipants - 1], ", "],
											" and ", removedParticipants[[-1]], " have"]], 
								" been removed from this notebook's chat thread."]];
				add = If[Length@addedParticipants === 0,
						"", 
						StringJoin[Which[Length@addedParticipants === 1, 
									addedParticipants[[1]] <> " has", 
									Length@addedParticipants === 2, 
									addedParticipants[[1]] <> " and " <> addedParticipants[[2]] <> " have",
									True, 
									StringJoin[StringJoin @@ Riffle[Take[addedParticipants, Length@addedParticipants - 1], ", "],
											" and ", addedParticipants[[-1]], " have"]], 
								" been added to this notebook's chat thread."]];
				ModifyParticipantsInThreadWrite[allparticipants, If[screenname === "None", wid, screenname], st[[1]], Which[removedParticipants === {},
													add,
													addedParticipants === {},
													rem,
													True,
													StringJoin[add, " ", rem]], CurrentValue[sel, {TaggingRules, "ScrollLock"}, False]]],
													
			And[MemberQ[addedParticipants, $WolframID], wid =!= $WolframID, sel === Missing["NotFound"]],
			
			If[TrueQ@CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "AvailableToReceiveChatInvitations"}, True], 
				JoinChatDialog[requester, screenname, "chatframework@wolfram.com" <> "/" <> id]]]];
			
sendPrivateChatMessage2[requester_, msg_ /; And[KeyExistsQ[ReleaseHold@msg, "id"], KeyExistsQ[ReleaseHold@msg, "removedparticipant"], KeyExistsQ[ReleaseHold@msg, "screenname"],
						KeyExistsQ[ReleaseHold@msg, "celllist"],
							Sort@Keys@ReleaseHold@msg === {"celllist", "id", "removedparticipant", "screenname"}]] :=
			EndUserParticipationInChatThread[requester, Lookup[ReleaseHold@msg, "id"], Lookup[ReleaseHold@msg, "removedparticipant"], Uncompress@Lookup[ReleaseHold@msg, "screenname"],
		Quiet[Uncompress@Lookup[ReleaseHold@msg, "celllist"],{CloudObject::cloudnf, Uncompress::string}]];

EndUserParticipationInChatThreadWrite[allparticipants_, removedparticipant_, screenname_, celllist_, co_, text_, scrolllockQ_:False] :=
	Module[{pos = (If[MatchQ[#, {{_}}], #[[1, 1]], 1]&[Position[allparticipants, removedparticipant]]), senderstyle, styleToUse},
		senderstyle = $sentStyles[[If[# === 0, -1, #] &[Mod[pos, Length@$sentStyles]]]];
		styleToUse = "GrayLight" <> senderstyle <> "Top";
		time = StringReplace[StringReplace[DateString["ISODateTime"], __ ~~ "T" -> ""], a__ ~~ ":" ~~ b__ ~~ ":" ~~ __ :> a <> ":" <> b];
		NotebookWrite[co, 
				If[celllist === {},
					{WolframIDandTimeCell[senderstyle, If[screenname === "None", removedparticipant, screenname], time], Cell[text, "Text", styleToUse], $stemCellTemplate}, 
					Join[ReplaceAll[Prepend[celllist, WolframIDandTimeCell[senderstyle, If[screenname === "None", removedparticipant, screenname], time]],
							{Cell[a_, b_, c_String /; StringMatchQ[c, __ ~~ "Top"], d__] :> Cell[a, b, "GrayLight" <> c, d], 
							Cell[a_, b_, _String, d__] :> Cell[a, b, "GrayLight", d]}],
						{Cell[text, "Text", "GrayLight"], $stemCellTemplate}]], AutoScroll -> (!TrueQ[scrolllockQ])]];

RemoveCorrespondingListener[id_]:=
	Module[{listeners = ChannelListeners[], listenerdata, channellistener},
		If[listeners =!= {}, 
			listenerdata = {#, #["URL"]} & /@ listeners; 
			channellistener = With[{id1 = id}, FirstCase[listenerdata, {a_, b_String /; StringMatchQ[b, __ ~~ id1]} :> a]];
			If[Head@channellistener === ChannelListener, RemoveChannelListener@channellistener]]];

EndUserParticipationInChatThread[requester_, id_, removedparticipant_, screenname_, celllist_] := 
	Module[{sel = SelectFirst[Notebooks[], (CurrentValue[#, {TaggingRules, "ChatNotebookID"}] === id) &], participants, windowtitle, allparticipants, st}, 
		Which[And[(requester =!= $WolframID) && (sel =!= Missing["NotFound"])],
			participants = DeleteCases[CurrentValue[sel, {TaggingRules, "Participants"}], requester | removedparticipant];
			CurrentValue[sel, {TaggingRules, "Participants"}] = participants;
			windowtitle = CurrentValue[sel, {TaggingRules, "ChatNotebookWindowTitle"}];
			allparticipants = (If[MemberQ[#, requester], #, Append[#, requester]]&[CurrentValue[sel, {TaggingRules, "AllParticipants"}]]);
			st = Cells[sel, CellStyle -> "Stem"];
			If[Length@participants === 1,
				CurrentValue[sel, NotebookDynamicExpression] = Inherited;
				CurrentValue[sel, DockedCells] = Inherited;
				If[st =!= {},
					SetOptions[st[[1]], Deletable -> True];
					EndUserParticipationInChatThreadWrite[allparticipants, removedparticipant, screenname, celllist, st[[1]],
						StringJoin[If[screenname === "None", removedparticipant, screenname], " has left this chat. This chat has been discontinued."], 
						CurrentValue[sel, {TaggingRules, "ScrollLock"}, False]];
					st = Cells[sel, CellStyle -> "Stem"];
					fs = First[st];
					SetOptions[fs, Deletable -> True];
					NotebookDelete[fs, AutoScroll -> (!CurrentValue[sel, {TaggingRules, "ScrollLock"}, False])];
					CurrentValue[sel, {TaggingRules, "AllowAttachSendButtons"}] = False;
					If[ListQ[#], If[Head@# === CellObject, NotebookDelete[#, AutoScroll -> (!CurrentValue[sel, {TaggingRules, "ScrollLock"}, False])]] & /@ #] &[CurrentValue[sel, {TaggingRules, "SelectedCells"}]];
					RemoveCorrespondingListener[id];
					CurrentValue[sel, {TaggingRules, "Terminated"}] = True;
					CurrentValue[sel, NotebookEventActions] = DeleteCases[CurrentValue[sel, NotebookEventActions], _["WindowClose" | {"MenuCommand", "SimilarCellBelow"}, _]]],
				If[st =!= {},
					SetOptions[st[[1]], Deletable -> True];
					EndUserParticipationInChatThreadWrite[allparticipants, removedparticipant, screenname, celllist, st[[1]],
										StringJoin[If[screenname === "None", removedparticipant, screenname], " has left this chat."],
										CurrentValue[sel, {TaggingRules, "ScrollLock"}, False]]]],
			(requester === $WolframID),
			listeners = ChannelListeners[];
			If[listeners =!= {}, 
				listenerdata = {#, #["URL"]} & /@ listeners; 
				channellistener = With[{id1 = id}, FirstCase[listenerdata, {a_, b_String /; StringMatchQ[b, __ ~~ id1]} :> a]];
				If[Head@channellistener === ChannelListener, RemoveChannelListener@channellistener]];
			URLExecute[CloudObject["https://www.wolframcloud.com/objects/7d89b199-fae2-4af8-ae24-3c4b775ee5b2"], {"id" -> id, "removedparticipant" -> $WolframID}, "WL", Method -> "POST"]]];
			
sendPrivateChatMessage2[requester_, msg_ /; And[KeyExistsQ[ReleaseHold@msg, "id"], KeyExistsQ[ReleaseHold@msg, "participantnotacceptinvite"], KeyExistsQ[ReleaseHold@msg, "screenname"],
						KeyExistsQ[ReleaseHold@msg, "celllist"],
							Sort@Keys@ReleaseHold@msg === {"celllist", "id", "participantnotacceptinvite", "screenname"}]] :=
			UserNotAcceptParticipationInChatThread[requester, Lookup[ReleaseHold@msg, "id"], Lookup[ReleaseHold@msg, "participantnotacceptinvite"], Uncompress@Lookup[ReleaseHold@msg, "screenname"],
		Quiet[Uncompress@Lookup[ReleaseHold@msg, "celllist"],{CloudObject::cloudnf, Uncompress::string}]];
		
UserNotAcceptParticipationInChatThreadWrite[allparticipants_, participantnotacceptinvite_, screenname_, celllist_, co_, text_, scrolllockQ_:False] :=
	Module[{pos = (If[MatchQ[#, {{_}}], #[[1, 1]], 1]&[Position[allparticipants, participantnotacceptinvite]]), senderstyle, styleToUse},
		senderstyle = $sentStyles[[If[# === 0, -1, #] &[Mod[pos, Length@$sentStyles]]]];
		styleToUse = "GrayLight" <> senderstyle <> "Top";
		time = StringReplace[StringReplace[DateString["ISODateTime"], __ ~~ "T" -> ""], a__ ~~ ":" ~~ b__ ~~ ":" ~~ __ :> a <> ":" <> b];
		NotebookWrite[co, 
				If[celllist === {},
					{WolframIDandTimeCell[senderstyle, If[screenname === "None", participantnotacceptinvite, screenname], time], Cell[text, "Text", styleToUse], $stemCellTemplate}, 
					Join[ReplaceAll[Prepend[celllist, WolframIDandTimeCell[senderstyle, If[screenname === "None", participantnotacceptinvite, screenname], time]],
							{Cell[a_, b_, c_String /; StringMatchQ[c, __ ~~ "Top"], d__] :> Cell[a, b, "GrayLight" <> c, d], 
							Cell[a_, b_, _String, d__] :> Cell[a, b, "GrayLight", d]}],
						{Cell[text, "Text", "GrayLight"], $stemCellTemplate}]], AutoScroll -> (!TrueQ[scrolllockQ])]];
		
UserNotAcceptParticipationInChatThread[requester_, id_, participantnotacceptinvite_, screenname_, celllist_] := 
	Module[{sel = SelectFirst[Notebooks[], (CurrentValue[#, {TaggingRules, "ChatNotebookID"}] === id) &], participants, windowtitle, allparticipants, st}, 
		Which[And[(requester =!= $WolframID) && (sel =!= Missing["NotFound"])],
			participants = DeleteCases[CurrentValue[sel, {TaggingRules, "Participants"}], requester | participantnotacceptinvite];
			CurrentValue[sel, {TaggingRules, "Participants"}] = participants;
			windowtitle = CurrentValue[sel, {TaggingRules, "ChatNotebookWindowTitle"}];
			allparticipants = (If[MemberQ[#, requester], #, Append[#, requester]]&[CurrentValue[sel, {TaggingRules, "AllParticipants"}]]);
			st = Cells[sel, CellStyle -> "Stem"];
			If[Length@participants === 1,
				CurrentValue[sel, NotebookDynamicExpression] = Inherited;
				CurrentValue[sel, DockedCells] = Inherited;
				If[st =!= {},
					SetOptions[st[[1]], Deletable -> True];
					UserNotAcceptParticipationInChatThreadWrite[allparticipants, participantnotacceptinvite, screenname, celllist, st[[1]],
						StringJoin[If[screenname === "None", participantnotacceptinvite, screenname], " has not accepted the invitation to this chat. This chat has been discontinued."], 
						CurrentValue[sel, {TaggingRules, "ScrollLock"}, False]];
					st = Cells[sel, CellStyle -> "Stem"];
					fs = First[st];
					SetOptions[fs, Deletable -> True];
					NotebookDelete[fs, AutoScroll -> (!CurrentValue[sel, {TaggingRules, "ScrollLock"}, False])];
					CurrentValue[sel, {TaggingRules, "AllowAttachSendButtons"}] = False;
					If[ListQ[#], If[Head@# === CellObject, NotebookDelete[#, AutoScroll -> (!CurrentValue[sel, {TaggingRules, "ScrollLock"}, False])]] & /@ #] &[CurrentValue[sel, {TaggingRules, "SelectedCells"}]];
					RemoveCorrespondingListener[id];
					CurrentValue[sel, {TaggingRules, "Terminated"}] = True;
					CurrentValue[sel, NotebookEventActions] = DeleteCases[CurrentValue[sel, NotebookEventActions], _["WindowClose" | {"MenuCommand", "SimilarCellBelow"}, _]]],
				If[st =!= {},
					SetOptions[st[[1]], Deletable -> True];
					UserNotAcceptParticipationInChatThreadWrite[allparticipants, participantnotacceptinvite, screenname, celllist, st[[1]],
										StringJoin[If[screenname === "None", participantnotacceptinvite, screenname], " has not accepted the invitation to this chat."],
										CurrentValue[sel, {TaggingRules, "ScrollLock"}, False]]]],
			(requester === $WolframID),
			listeners = ChannelListeners[];
			If[listeners =!= {}, 
				listenerdata = {#, #["URL"]} & /@ listeners; 
				channellistener = With[{id1 = id}, FirstCase[listenerdata, {a_, b_String /; StringMatchQ[b, __ ~~ id1]} :> a]];
				If[Head@channellistener === ChannelListener, RemoveChannelListener@channellistener]];
			URLExecute[CloudObject["https://www.wolframcloud.com/objects/7d89b199-fae2-4af8-ae24-3c4b775ee5b2"], {"id" -> id, "removedparticipant" -> $WolframID}, "WL", Method -> "POST"]]];
									
JoinChatDialog[requester_, screenname_, privatechannel_] := 
	With[{requester1 = requester, privatechannel1 = privatechannel, screenname1 = CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "ScreenName"}],
		id = StringReplace[privatechannel, __ ~~ "/" ~~ (a__ /; StringFreeQ[a, "/"]) :> a], wid = $WolframID}, 
		Quiet[ChannelListen[ChannelObject[StringJoin["chatframework@wolfram.com", ":", id]], "TrustedChannel" -> True], ChannelObject::uauth];
		NotebookPut[Notebook[{Cell[BoxData[GridBox[{{ChatTools`Private`chatIconSmall,
								Cell[TextData["Chat Invitation"], "AcceptCancelTitleText"]}}, GridBoxSpacings -> {"Columns" -> {{.5}}}]], "AcceptCancelTitle"],
					Cell["", "ChatDialogDelimiterAbove"],
					Cell[TextData[{StyleBox[If[MemberQ[{"", "None"}, screenname], requester, screenname], FontColor -> GrayLevel[0]],
							" has invited you to chat."}], "AcceptCancelText"],
					Cell["", "ChatDialogDelimiterBelow"],
					Cell[BoxData[GridBox[{{ButtonPairSequence[ButtonBox[StyleBox["Cancel", FontColor -> GrayLevel[0]],
					ButtonFunction :> (NotebookClose[]; RunScheduledTask[Quiet[If[TimeConstrained[AssociationQ@CloudGet@CloudObject["https://www.wolframcloud.com/objects/" <> id], 2, False],
						ChannelSend[StringJoin["chatframework@wolfram.com", ":", id], 
							Association @@ List["id" -> id, "participantnotacceptinvite" -> wid, 
								"screenname" -> Compress@If[MemberQ[{"", Inherited}, screenname1] || (StringQ@screenname1 && StringMatchQ[screenname1, Whitespace]), "None", 
											screenname1], "celllist" -> Compress@{}], ChannelPreSendFunction -> None]], {ChannelObject::uauth, CloudGet::cloudnf}];
											RemoveScheduledTask[$ScheduledTask], {0.1}]), 
										Appearance -> {"ButtonType" -> "Cancel", "Cancel" -> None}, Background -> GrayLevel[.9], Evaluator -> Automatic,
										Method -> "Queued", ImageSize -> {70, 25}], 
								ButtonBox[StyleBox["   Accept   ", FontColor -> GrayLevel[1]],
									ButtonFunction :> (Needs["ChatTools`"]; NotebookClose[]; ChatTools`AcceptChatInvitation[requester1, privatechannel1]), 
									Appearance -> {"ButtonType" -> "Default", "Default" -> FrontEnd`ToFileName[{"Toolbars", "ChatTools"}, "SendButton-Default.9.png"], 
						"Hover" -> FrontEnd`ToFileName[{"Toolbars", "ChatTools"}, "SendButton-Hover.9.png"]}, FontColor -> GrayLevel[1], 
									Background -> RGBColor[0., 0.5548332951857786, 1.], Evaluator -> "ChatServices", Method -> "Queued"]]}}, 
								GridBoxAlignment -> {"Columns" -> {{Right}}}, GridBoxSpacings -> {"Columns" -> {{1}}}]], "AcceptCancelButtons"]},
					WindowSize -> {520, 180}, 
					ShowCellBracket -> False,
					"CellInsertionPointCell" -> {}, 
					"BlinkingCellInsertionPoint" -> False, 
					"CellInsertionPointColor" -> GrayLevel[1], 
					WindowFrame -> "ModelessDialog",
					WindowElements -> {}, 
					WindowFrameElements -> {}, 
					ShowStringCharacters -> False,
					Background -> GrayLevel[1], 
					ScrollingOptions -> {"PagewiseScrolling" -> False, "PagewiseDisplay" -> True, "VerticalScrollRange" -> Fit}, 
					CellMargins -> {{0, 0}, {0, 0}}, 
					AutoMultiplicationSymbol -> False,
					Saveable -> False, 
					WindowTitle -> "Wolfram Chat",
					Editable -> False, 
					Selectable -> False, 
					StyleDefinitions -> Notebook[{Cell[StyleData["AcceptCancelTitle"], FontColor -> GrayLevel[0], ShowCellBracket -> False, 
										CellMargins -> {{10, 30}, {2, 12}}], 
									Cell[StyleData["AcceptCancelTitleText"], FontSize -> 14, FontFamily -> "Source Sans Pro", FontColor -> GrayLevel[0], 
										ShowCellBracket -> False], 
									Cell[StyleData["AcceptCancelText"], FontSize -> 16, FontFamily -> "Source Sans Pro", FontColor -> GrayLevel[0.1], 
										ShowCellBracket -> False, CellMargins -> {{38, 30}, {2, 10}}, ShowAutoSpellCheck -> False], 
									Cell[StyleData["AcceptCancelButtons"], TextAlignment -> Right, CellMargins -> {{30, 30}, {15, 15}}, 
										ButtonBoxOptions -> {ImageSize -> {80, 24}, BaseStyle -> {FontFamily -> "Source Sans Pro", FontSize -> 14}}],
									Cell[StyleData["ChatDialogDelimiterAbove"], Editable -> False, Selectable -> False, CellFrame -> {{0, 0}, {0.5, 0}}, 
										ShowCellBracket -> False, CellMargins -> {{0, 0}, {10, 0}}, CellFrameColor -> GrayLevel[0.75], CellSize -> {Automatic, 1}, 
										CellFrameMargins -> 0, ShowStringCharacters -> False], 
									Cell[StyleData["ChatDialogDelimiterBelow"], Editable -> False, Selectable -> False, CellFrame -> {{0, 0}, {0.5, 0}}, 
										ShowCellBracket -> False, CellMargins -> {{0, 0}, {0, 10}}, CellFrameColor -> GrayLevel[0.75], CellSize -> {Automatic, 1}, 
										CellFrameMargins -> 0, ShowStringCharacters -> False]}], 
					NotebookEventActions -> {"ReturnKeyDown" :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
								{"MenuCommand", "EvaluateCells"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
								{"MenuCommand", "HandleShiftReturn"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
								{"MenuCommand", "EvaluateNextCell"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]], 
								"EscapeKeyDown" :> (FE`Evaluate[FEPrivate`FindAndClickCancelButton[]]; DialogReturn[$Failed]), 
								"WindowClose" :> (FE`Evaluate[FEPrivate`FindAndClickCancelButton[]]; DialogReturn[$Failed])},
					Evaluator -> "ChatServices",
					TaggingRules -> {"JoinChat" -> id}]]];

(* requester - who it is from - who sent the message *)

sendPrivateChatMessage[requester_, msg_ /; And[KeyExistsQ[ReleaseHold@msg, "participants"], KeyExistsQ[ReleaseHold@msg, "privatechannel"], KeyExistsQ[ReleaseHold@msg, "ScreenName"],
						Sort@Keys@ReleaseHold@msg === {"participants", "privatechannel", "ScreenName"}]] :=
	Module[{privatechannel = Lookup[ReleaseHold@msg, "privatechannel"], screenname = Uncompress@Lookup[ReleaseHold@msg, "ScreenName"], id, riid, timestamp},
		Which[And[requester =!= $WolframID, MemberQ[Quiet[Uncompress[Lookup[ReleaseHold@msg, "participants"]],{CloudObject::cloudnf, Uncompress::string}], $WolframID]],
		
			If[Not@MemberQ[StringMatchQ[#, __ ~~ "/" ~~ privatechannel] & /@ (#["Path"] & /@ (Cases[#["ChannelObject"] & /@ ChannelListeners[],
																					ChannelObject[a_] :> a])), True],
				If[TrueQ@CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "AvailableToReceiveChatInvitations"}, True],
			
					JoinChatDialog[requester, screenname, privatechannel]]],
					
			requester === $WolframID && (id = StringReplace[privatechannel, __ ~~ "/" ~~ (a__ /; StringFreeQ[a, "/"]) :> a];
							riid = Quiet[CloudGet@CloudObject["https://www.wolframcloud.com/objects/" <> id], CloudGet::cloudnf];
							riid =!= $Failed && AssociationQ@riid),
							
			timestamp = "";
			updateNotebook[requester, id, timestamp, riid["date"], riid["ChatCreationDate"],
						Uncompress@CloudGet["https://www.wolframcloud.com/objects/" <> riid["cellListUUIDs"][[1, 2]]],
						If[StringQ[#], Uncompress[#], #]&[riid["participants"]](*, "True"*), riid["wid"], Uncompress@riid["ScreenName"], Uncompress@riid["windowtitle"],
						Uncompress@riid["CustomWindowTitle"],
						riid["originator"], If[StringQ[#], Uncompress[#], #]&[riid["allparticipants"]]]]];
			
AcceptChatInvitation[requester_, privatechannel_]:= 
	Module[{id, riid, timestamp},
		id = StringReplace[privatechannel, __ ~~ "/" ~~ (a__ /; StringFreeQ[a, "/"]) :> a];
		riid = Quiet[CloudGet@CloudObject["https://www.wolframcloud.com/objects/" <> id], CloudGet::cloudnf];
		Which[riid === $Failed || Not@AssociationQ@riid || Not@MemberQ[Keys[riid], "participants"],
		
			MessageDialog["The chat has been discontinued.", WindowFrame -> "ModalDialog", WindowSize -> {250, All}, Evaluator -> "ChatServices"],
			
			Not@MemberQ[If[StringQ[#], Uncompress[#], #]&[riid["participants"]], $WolframID],
			
			MessageDialog["You have been removed from this chat in the time since you were invited.", WindowFrame -> "ModalDialog", WindowSize -> {500, All}],
			
			True,
			
			If[AssociationQ@riid,
				(*timestamp = ToString[CloudObjectInformation[riid["cellListUUIDs"], "Created"]];*)
				(*timestamp = (If[StringQ@# && StringMatchQ[#, "DateObject[" ~~ __], #, ""] &[ToString[CloudObjectInformation[CloudObject["https://www.wolframcloud.com/objects/" <> riid["cellListUUIDs"][[-1, -1]]], "Created"]]]);*)
				(*ChannelListen[ChannelObject[StringReplace[privatechannel,"/"->":"]], "TrustedChannel" -> True];*)
				timestamp = "";
				updateNotebook[requester, id, timestamp, riid["date"], riid["ChatCreationDate"],
						riid["cellListUUIDs"](*{Part[riid["cellListUUIDs"], -1]}*),
						If[StringQ[#], Uncompress[#], #]&[riid["participants"]](*, "True"*), riid["wid"], Uncompress@riid["ScreenName"], Uncompress@riid["windowtitle"],
						Uncompress@riid["CustomWindowTitle"],
						riid["originator"], If[StringQ[#], Uncompress[#], #]&[riid["allparticipants"]], "OpenFromInvitation" -> True], 
				MessageDialog["There was a problem accessing the chat index. Try reactivating the chat listener.", WindowFrame -> "ModalDialog", WindowSize -> {520, All}]]]];

sendPrivateChatMessage2[requester_, msg_ /; And[KeyExistsQ[ReleaseHold@msg, "ChatNotebookID"], KeyExistsQ[ReleaseHold@msg, "date"], KeyExistsQ[ReleaseHold@msg, "ChatCreationDate"],
					KeyExistsQ[ReleaseHold@msg, "cellList"], KeyExistsQ[ReleaseHold@msg, "participants"],
					KeyExistsQ[ReleaseHold@msg, "wid"], KeyExistsQ[ReleaseHold@msg, "ScreenName"], KeyExistsQ[ReleaseHold@msg, "windowtitle"],
					KeyExistsQ[ReleaseHold@msg, "CustomWindowTitle"],
					KeyExistsQ[ReleaseHold@msg, "originator"], KeyExistsQ[ReleaseHold@msg, "allparticipants"],
		Sort@Keys@ReleaseHold@msg === {"allparticipants", "cellList", "ChatCreationDate", "ChatNotebookID", "CustomWindowTitle", "date", "originator", "participants", "ScreenName", "wid",
						"windowtitle"}]] := updateNotebook[requester,
		Lookup[ReleaseHold@msg, "ChatNotebookID"], "", Lookup[ReleaseHold@msg, "date"], Lookup[ReleaseHold@msg, "ChatCreationDate"],
		Map[Function[t, Replace[t, Cell[a__] :> Append[DeleteCases[Cell[a], Evaluator -> _], Evaluator -> "ChatServices"]]], 
 			Quiet[Uncompress@Lookup[ReleaseHold@msg, "cellList"], {CloudObject::cloudnf, Uncompress::string}]],
		Uncompress@Lookup[ReleaseHold@msg, "participants"], Lookup[ReleaseHold@msg, "wid"], Uncompress@Lookup[ReleaseHold@msg, "ScreenName"], Uncompress@Lookup[ReleaseHold@msg, "windowtitle"],
		Uncompress@Lookup[ReleaseHold@msg, "CustomWindowTitle"], Lookup[ReleaseHold@msg, "originator"], Uncompress@Lookup[ReleaseHold@msg, "allparticipants"]];
					
SetAcceptRejectEmails[emailadddresses_?(VectorQ[#, StringQ] &), status : ("Accept" | "Reject"), permanence : ("Session" | "Permanent")] := 
	Module[{a = CurrentValue[If[permanence === "Session", $FrontEndSession, $FrontEnd], {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", status}]}, 
		If[Not[And @@ (TextCases[#, "EmailAddress"] === {#} & /@ emailadddresses)], 
			Print["Not all elements of the list are email addresses."], 
			CurrentValue[If[permanence === "Session", $FrontEndSession, $FrontEnd], {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", status}] = If[a === Inherited, emailadddresses, Join[a, emailadddresses]]]];

SetAcceptRejectRegularExpressions[regularexpressions_?(VectorQ[#, StringQ] &), status : ("Accept" | "Reject"), permanence : ("Session" | "Permanent")] := 
	Module[{a = CurrentValue[If[permanence === "Session", $FrontEndSession, $FrontEnd], {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", status}]}, 
		CurrentValue[If[permanence === "Session", $FrontEndSession, $FrontEnd], {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", status}] = If[a === Inherited,
																	RegularExpression /@ regularexpressions, 
																	Join[a, RegularExpression /@ regularexpressions]]];

(* Send can be done by Alt+Enter (Option+Return) via the notebook event action. *)

DisplayActiveParticipants[] := 
	Module[{nb = ButtonNotebook[], p1},
		If[TrueQ@CurrentValue[nb, {TaggingRules, "BeforeSend"}],
	
			MessageDialog["You must send something first.", WindowFrame -> "ModalDialog", Evaluator -> "ChatServices", WindowSize -> {230, All}],
	
			With[{id = CurrentValue[nb, {TaggingRules, "ChatNotebookID"}]},
				p1 = URLExecute[CloudObject["https://www.wolframcloud.com/objects/77ebfd95-ab31-44db-97a9-92ece4312e7b"], {"ChatNotebookID" -> id}, "WL", Method -> "POST"]];
				
			If[Head@p1 === Failure,
			
				MessageDialog["The channel is unavailable.", WindowFrame -> "ModalDialog", Evaluator -> "ChatServices", WindowSize -> {230, All}],
				
				If[ListQ@p1,
					CurrentValue[nb, {TaggingRules, "ViewParticipants"}] = True;
					SetOptions[nb, NotebookDynamicExpression -> Inherited(*, NotebookEventActions -> None*), "BlinkingCellInsertionPoint" -> False, "CellInsertionPointColor" -> GrayLevel[1],
						"CellInsertionPointCell" -> None];
					If[ListQ[#], If[Head@# === CellObject,
						NotebookDelete[#, AutoScroll -> (! CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])]] & /@ #] &[CurrentValue[nb, {TaggingRules, "SelectedCells"}]];
					CurrentValue[nb, DockedCells] = Cell[BoxData[ToBoxes[With[{p = p1}, 
						Grid[{{Style["PARTICIPANTS", FontFamily -> "Source Sans Pro", FontColor -> GrayLevel[.39]]},
							{Framed[Pane[Grid[Table[{Button[p[[j]], Null, BaseStyle -> {FontFamily -> "Source Sans Pro", FontColor -> GrayLevel[.39]}, Appearance -> None, 
											ContentPadding -> False, Evaluator -> None, Enabled -> False]}, {j, Length@p}], Alignment -> Left],
								AppearanceElements -> {}, Scrollbars -> {False, Automatic}, ImageSize -> {410, 200}], Background -> White, 
								FrameStyle -> GrayLevel[.7]]},
							{Grid[{{Button["Close",
								Module[{nb1 = ButtonNotebook[]},
									Needs["ChatTools`"];
									CurrentValue[nb1, DockedCells] = With[{id = CurrentValue[nb1, {TaggingRules, "ChatNotebookID"}],
														shortcut = CurrentValue[nb1, {TaggingRules, "Shortcut"}],
														type = CurrentValue[nb1, {TaggingRules, "RoomType"}],
														originator = CurrentValue[nb1, {TaggingRules, "Originator"}],
														teachersadded = CurrentValue[nb1, {TaggingRules, "Moderators"}]}, 
												If[TrueQ@CurrentValue[nb1, {TaggingRules, "ChatRoom"}],
													ChatNotebookDockedCell[id, CurrentValue[nb1, WindowTitle], "CanSetVisibility" -> False, 
												"Preferences" -> ChatClassRoomPreferencesMenu[id, shortcut, 
						"Teacher" -> (($WolframID === originator) || (ListQ@teachersadded && MemberQ[teachersadded, $WolframID])), 
																"CellLabelsDefault" -> If[type === "PromptedResponse", "On", "Off"]]],
													ChatNotebookDockedCell[id, CurrentValue[nb1, WindowTitle], "CanSetVisibility" -> False, 
													"Preferences" -> PrivateChatPreferencesMenu[id]]]];
									SetOptions[nb1, "BlinkingCellInsertionPoint" -> Inherited, "CellInsertionPointColor" -> Inherited, "CellInsertionPointCell" -> Inherited];
              								With[{nb2 = nb1}, 
										SetOptions[nb1,  
											NotebookEventActions -> {{"MenuCommand", "SimilarCellBelow"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
												{"MenuCommand", "SaveRename"} :> ChatTools`SaveChatNotebook[nb2],
								If[$VersionNumber < 12, Nothing, {"MenuCommand", "Copy"} :> Module[{in = InputNotebook[]}, 
									If[MatchQ[Developer`CellInformation[in], {___, {"Style" -> {_, __}, __, "CursorPosition" -> "CellBracket", __}, ___}], 
										CopyToClipboard[DeleteCases[ReplaceAll[NotebookRead@in,
						Cell[a_, b_String, _String, ___, TaggingRules -> _, ___] :> Cell[a, b]], Cell[__, "CellUserLabelOptions" | "CellLabelOptions" | "Stem", ___], Infinity]], 
										FrontEndExecute[{FrontEndToken[in, "Copy"]}]]]],
												PassEventsDown -> False}]];
									SelectionMove[nb1, After, Notebook, AutoScroll -> (!CurrentValue[nb1, {TaggingRules, "ScrollLock"}, False])];
									CurrentValue[nb1, {TaggingRules, "ViewParticipants"}] = Inherited;
									(Function[t, If[StringQ@t && t =!= "None", ChatTools`Private`UpdateBanner[nb1, t]]][CurrentValue[nb1, {TaggingRules, "Banner"}]])], 
								Evaluator -> "ChatServices", Active -> True, Method -> "Queued", FrameMargins -> 1]}}]}}, 
						Spacings -> {Automatic, {2 -> .5, 3 -> 1.5}}, Alignment -> Left]]]], "ChatDockedCell",
									TextAlignment -> Center, ShowCellBracket -> False, FontSize -> 13, Deployed -> True],
		
				With[{nb1 = nb}, 
					SetOptions[nb1, "BlinkingCellInsertionPoint" -> Inherited, "CellInsertionPointColor" -> Inherited, "CellInsertionPointCell" -> Inherited,
							NotebookEventActions -> {{"MenuCommand", "SimilarCellBelow"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
										{"MenuCommand", "SaveRename"} :> ChatTools`SaveChatNotebook[nb1],
								If[$VersionNumber < 12, Nothing, {"MenuCommand", "Copy"} :> Module[{in = InputNotebook[]}, 
									If[MatchQ[Developer`CellInformation[in], {___, {"Style" -> {_, __}, __, "CursorPosition" -> "CellBracket", __}, ___}], 
										CopyToClipboard[DeleteCases[ReplaceAll[NotebookRead@in,
						Cell[a_, b_String, _String, ___, TaggingRules -> _, ___] :> Cell[a, b]], Cell[__, "CellUserLabelOptions" | "CellLabelOptions" | "Stem", ___], Infinity]], 
										FrontEndExecute[{FrontEndToken[in, "Copy"]}]]]],
										PassEventsDown -> False}]]]]]];

MoveCursorAfterCellPosition[nb_] :=
	Which[MatchQ[Developer`CellInformation[nb], {{"Style" -> {_, "Banner"}, __}}],
		NotebookFind[nb, "Stem", All, CellStyle, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])];
		SelectionMove[nb, After, Cell, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])],
		MatchQ[Developer`CellInformation[nb], {{__, "CursorPosition" -> "CellBracket", __} ..}], 
		SelectionMove[nb, After, Cell, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])], 
		Developer`CellInformation[nb] =!= $Failed, 
		Module[{lnkre}, While[(LinkWrite[$ParentLink, FrontEnd`CellInformation[nb]]; lnkre = LinkRead[$ParentLink]);
					(lnkre =!= $Failed && Not[MemberQ["CursorPosition" /. lnkre, "CellBracket"]]), 
		FrontEndExecute[FrontEnd`SelectionMove[nb, All, Cell, AutoScroll -> False]]]];
		SelectionMove[nb, After, Cell, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])]];
		
RemoveRoomListenerAndCloseRoom[id_] := 
	Module[{listeners = ChannelListeners[], listenerdata, channellistener}, 
		If[listeners =!= {}, listenerdata = {#, #["URL"]} & /@ listeners;
			channellistener = FirstCase[listenerdata, {a_, b_String /; StringMatchQ[b, __ ~~ id]} :> a];
			If[Head@channellistener === ChannelListener, RemoveChannelListener@channellistener]];
		NotebookClose[ButtonNotebook[]]];
		
ClearSentCells[] := 
	Module[{nb = ButtonNotebook[], sel}, 
		sel = Select[Cells[nb], 
				Not@FreeQ["Style" /. Developer`CellInformation@#, 
						Alternatives @@ Join[Join[$sentStyles, # <> "Top" & /@ $sentStyles], {"GrayLight", "CellLabelOptions", "CellUserLabelOptions", "More"}, 
									"GrayLight" <> # <> "Top" & /@ $sentStyles]] &]; 
		SetOptions[#, Deletable -> True] & /@ sel;
		NotebookDelete[sel, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])]];
		
preferencesIcon = GraphicsBox[{Thickness[0.09090909090909091], 
	FaceForm[{RGBColor[0.392, 0.392, 0.392], Opacity[1.]}], 
	FilledCurveBox[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}}, {{{0.5, 7.5}, {9.5, 7.5}, {9.5, 8.5}, {0.5, 8.5}}, {{0.5, 4.5},
			{9.5, 4.5}, {9.5, 5.5}, {0.5, 5.5}}, {{0.5, 1.5}, {9.5, 1.5}, {9.5, 2.5}, {0.5, 2.5}}}]},
					AspectRatio -> Automatic, ImageSize -> {11., 9.}, PlotRange -> {{0., 11.}, {0., 9.}}];
					
Options[PrivateChatPreferencesMenu] = {"UpdateParticipants" -> False};

PrivateChatPreferencesMenu[id_, opts___?OptionQ] := 
	Module[{up = ("UpdateParticipants" /. {opts} /. Options[PrivateChatPreferencesMenu])},
	screenname = (Function[s, If[StringQ@s && Not@StringMatchQ[s, "" | Whitespace], s, ""]][CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "ScreenName"}]]);
	With[{id1 = "\""<>id<>"\"", id2 = id, screenname1 = screenname}, 
		ActionMenu[Button[RawBoxes@preferencesIcon,Appearance->{"ButtonType"->"Default",
				  "Default"->FrontEnd`ToFileName[{"Toolbars","ChatTools"}, "DockedCellButton-Default.9.png"],
				  "Hover"->FrontEnd`ToFileName[{"Toolbars","ChatTools"}, "DockedCellButton-Hover.9.png"],
				  "Pressed"->FrontEnd`ToFileName[{"Toolbars","ChatTools"}, "DockedCellButton-Pressed.9.png"]},
				  ImageSize->{All,22}],
			{"New Chat" :> (Needs["ChatTools`"]; ChatTools`Chat[]),
				Delimiter, 
			"Chat Settings" :> (Needs["ChatTools`"]; ChatTools`StartChatChannelListener[]), 
			"Copy Channel UUID" :> CopyToClipboard[id1],
			If[TrueQ@up,
				"Manage Participants" :> (Needs["ChatTools`"]; ChatTools`UpdateParticipants["Active" -> True]),
				"View Participants" :> (Needs["ChatTools`"]; ChatTools`Private`DisplayActiveParticipants[])],
			Delimiter,
			"Clear All Messages" :> (Needs["ChatTools`"]; ChatTools`ClearSentCells[]),
			"Restore All Messages" :> (Needs["ChatTools`"]; ChatTools`InsertAllSentCellsFromCloud[]),
			RawBoxes@GridBox[{{"Scroll on New Message", ToBoxes@Dynamic[FEPrivate`If[FEPrivate`SameQ[FrontEnd`CurrentValue[FrontEnd`EvaluationNotebook[], {TaggingRules, "ScrollLock"}], 
															True], 
													RawBoxes@GraphicsBox[{}, ImageSize -> {8., {1., 6.}}],
													RawBoxes@"\[Checkmark]"]]}}, 
						GridBoxSpacings -> {"Columns" -> {{1}}}] :>
			  (CurrentValue[ButtonNotebook[], {TaggingRules, "ScrollLock"}] = If[BooleanQ[Not[CurrentValue[ButtonNotebook[], {TaggingRules, "ScrollLock"}]]], 
			    Not[CurrentValue[ButtonNotebook[], {TaggingRules, "ScrollLock"}]], False]), 
			RawBoxes@GridBox[{{"Show In/Out Labels", ToBoxes@Dynamic[FEPrivate`If[FEPrivate`SameQ[FrontEnd`CurrentValue[FrontEnd`EvaluationNotebook[], ShowCellLabel],
														True],
												RawBoxes@"\[Checkmark]", 
												RawBoxes@GraphicsBox[{}, ImageSize -> {8.6`, 0}]]]}}, 
						GridBoxSpacings -> {"Columns" -> {{1}}}] :> Module[{bn = ButtonNotebook[]}, 
													CurrentValue[bn, ShowCellLabel] = If[TrueQ@CurrentValue[bn, ShowCellLabel], False, True]],
			Delimiter,
			"Help" :> (FEPrivate`FrontEndExecute[FrontEndToken["OpenHelpLink", {"paclet:workflowguide/UsingChat", 
					FEPrivate`If[CurrentValue["ShiftKey"], Null, FEPrivate`ButtonNotebook[]]}]]&),
			"Exit Chat" :> Module[{bn = ButtonNotebook[]}, CurrentValue[bn, Saveable] = False; NotebookClose[bn];
						If[Not@TrueQ@CurrentValue[bn, {TaggingRules, "BeforeSend"}], Quiet[RunScheduledTask[Quiet[ChannelSend[StringJoin["chatframework@wolfram.com", ":", id2],
												Association @@ List["id" -> id2, "removedparticipant" -> $WolframID, 
		"screenname" -> Compress@If[MemberQ[{"", Inherited}, screenname1] || (StringQ@screenname1 && StringMatchQ[screenname1, Whitespace]), "None", screenname1], "celllist" -> Compress@{}], 
										ChannelPreSendFunction -> None], ChannelObject::uauth]; RemoveScheduledTask[$ScheduledTask];, {0.1}],ChannelSend::uauth]]]}, 
				Appearance -> None, Method -> "Queued", Evaluator->"ChatServices"]]];
				
Options[ChatClassRoomPreferencesMenu] = {"Teacher" -> True, "CellLabelsDefault" -> "Off"};

ChatClassRoomPreferencesMenu[id_, shortcut_, opts___?OptionQ] :=
	Module[{teacher = ("Teacher" /. {opts} /. Options[ChatClassRoomPreferencesMenu]), celllabelsdefault = ("CellLabelsDefault" /. {opts} /. Options[ChatClassRoomPreferencesMenu])},
		With[{id1 = "\""<>id<>"\"", id2 = id, shortcut1 = shortcut},
			ActionMenu[Button[RawBoxes@preferencesIcon,Appearance->{"ButtonType"->"Default",
				  "Default"->FrontEnd`ToFileName[{"Toolbars","ChatTools"}, "DockedCellButton-Default.9.png"],
				  "Hover"->FrontEnd`ToFileName[{"Toolbars","ChatTools"}, "DockedCellButton-Hover.9.png"],
				  "Pressed"->FrontEnd`ToFileName[{"Toolbars","ChatTools"}, "DockedCellButton-Pressed.9.png"]},
				  ImageSize->{All,22}],
				{"New Chat" :> (Needs["ChatTools`"]; ChatTools`Chat[]),
				Delimiter,
				"Chat Settings" :> (Needs["ChatTools`"]; ChatTools`StartChatChannelListener[]), 
				"Copy Channel UUID" :> CopyToClipboard[id1],
				"View Participants" :> (Needs["ChatTools`"]; ChatTools`Private`DisplayActiveParticipants[]),
				Delimiter,
				"Clear All Messages" :> (Needs["ChatTools`"]; ChatTools`ClearSentCells[]),
				"Restore All Messages" :> (Needs["ChatTools`"]; ChatTools`InsertAllRoomCellsFromCloud[]), 
				RawBoxes@GridBox[{{"Scroll on New Message", ToBoxes@Dynamic[FEPrivate`If[FEPrivate`SameQ[FrontEnd`CurrentValue[FrontEnd`EvaluationNotebook[], {TaggingRules, "ScrollLock"}], 
																True], 
														RawBoxes@GraphicsBox[{}, ImageSize -> {8.6`, 0}],
														RawBoxes@"\[Checkmark]"]]}}, 
						GridBoxSpacings -> {"Columns" -> {{1}}}] :>
			  (CurrentValue[ButtonNotebook[], {TaggingRules, "ScrollLock"}] = If[BooleanQ[Not[CurrentValue[ButtonNotebook[], {TaggingRules, "ScrollLock"}]]], 
			    Not[CurrentValue[ButtonNotebook[], {TaggingRules, "ScrollLock"}]], False]),
				If[celllabelsdefault === "Off",
						RawBoxes@GridBox[{{"Show In/Out Labels", ToBoxes@Dynamic[FEPrivate`If[FEPrivate`SameQ[FrontEnd`CurrentValue[FrontEnd`EvaluationNotebook[],
																				ShowCellLabel],
																	True],
															RawBoxes@"\[Checkmark]", 
															RawBoxes@GraphicsBox[{}, ImageSize -> {8.6`, 0}]]]}}, 
							GridBoxSpacings -> {"Columns" -> {{1}}}] :> Module[{bn = ButtonNotebook[]}, 
													CurrentValue[bn, ShowCellLabel] = If[TrueQ@CurrentValue[bn, ShowCellLabel], False, True]],
						RawBoxes@GridBox[{{"Show In/Out Labels", ToBoxes@Dynamic[FEPrivate`If[FEPrivate`SameQ[FrontEnd`CurrentValue[FrontEnd`EvaluationNotebook[],
																					ShowCellLabel],
																			True],
															RawBoxes@"\[Checkmark]", 
															RawBoxes@GraphicsBox[{}, ImageSize -> {8.6`, 0}]]]}}, 
						GridBoxSpacings -> {"Columns" -> {{1}}}] :> Module[{bn = ButtonNotebook[]},
													CurrentValue[bn, ShowCellLabel] = If[TrueQ@CurrentValue[bn, ShowCellLabel], False, True]]],
				Delimiter,
				If[teacher === True,
					Unevaluated[Sequence["Set Banner" :> (Needs["ChatTools`"]; ChatTools`SetBannerDialog[shortcut1]), 
						"Remove Banner" :> (Needs["ChatTools`"]; ChatTools`SetRoomBanner[shortcut1, "Action" -> "Remove"]),
						Delimiter]],
					Nothing],
				"Help" :> (FEPrivate`FrontEndExecute[FrontEndToken["OpenHelpLink", {"paclet:workflowguide/UsingChat", 
					FEPrivate`If[CurrentValue["ShiftKey"], Null, FEPrivate`ButtonNotebook[]]}]]&),
				"Exit Chat" :> (Needs["ChatTools`"]; ChatTools`RemoveRoomListenerAndCloseRoom[id2])}, 
					Appearance -> None, Method -> "Queued", Evaluator -> "ChatServices"]]];

Options[ChatNotebookDockedCell] = {"CanSetVisibility" -> True, "Preferences" -> None};

ChatNotebookDockedCell[id_, windowtitle_, opts___?OptionQ] := 
	Module[{csv = ("CanSetVisibility" /. {opts} /. Options[ChatNotebookDockedCell]), pref = ("Preferences" /. {opts} /. Options[ChatNotebookDockedCell])}, 
		Cell[BoxData@ToBoxes@Grid[{{Spacer[1], RawBoxes@chatIcon,
			If[TrueQ@csv, DynamicModule[{$ChatVisibility = "Public"}, PopupMenu[Dynamic[$ChatVisibility, ($ChatVisibility = #; CurrentValue[ButtonNotebook[], {TaggingRules, 
									"ChatVisibility"}] = $ChatVisibility) &], {"Public", "Private"}, ContentPadding -> False,
									ImageSize -> {Automatic, If[$OperatingSystem === "Windows", 25, 20]}]], Unevaluated[Sequence[]]], 
			Style[windowtitle, "ChatDockedTitle"],
					Item["", ItemSize -> Fit],
					If[pref === None, Nothing, Unevaluated[Sequence[pref, Spacer[1]]]]}}, Alignment -> {Left, Center}], "ChatDockedCell"]];
										
AuxiliaryUpdateParticipants[] := 
	Module[{nb = ButtonNotebook[], contactsToAdd, oldparticipants, allparticipants, participantsWithoutUser, contacts, permissionGroupRules, PermissionsGroupsEmailListPairs,
		PermissionsGroupsFailedAccess, EmailAddressesToSendToFromPermissionsGroups, PermissionsGroupsToAdd, potentialEmailAddressesToAddToContacts, emailAddressesToAdd, emailListData,
		contactsToAddToSend, rules, newEmailListData, contactsToDeleteFromSend, id, newListToSendTo, newParticipantsList, c1, c2, compressedScreenName, un, banner},
		
		Catch[(* The following gives the string content from the NEW PARTICIPANTS input field which contains email addresses and/or permissions groups. *)
		contactsToAdd = CurrentValue[nb, {"TaggingRules", "ContactsToAdd"}];
		
		oldparticipants = CurrentValue[nb, {TaggingRules, "Participants"}];
		allparticipants = CurrentValue[nb, {TaggingRules, "AllParticipants"}];
		participantsWithoutUser = DeleteCases[oldparticipants, $WolframID];
		
		contacts = CurrentValue[nb, {TaggingRules, "Contacts"}];
		
		permissionGroupRules = {PermissionsGroup[a_String] :> PermissionsGroup[$WolframID, a],
					RuleDelayed[PermissionsGroup[a_String, b_String /; StringFreeQ[b, "https://www.wolframcloud.com/objects/"]],
							PermissionsGroup[a, "https://www.wolframcloud.com/objects/" <> StringReplace[a, "@wolfram.com" -> ""] <> "/PermissionsGroup/" <> b]]};
		(* Get the permissions groups that have members and form a list of pairs - {{pg1, members1}, {pg2, members2}, ...} *)					
		PermissionsGroupsEmailListPairs = Quiet[Map[{#, DeleteCases[PermissionsGroupEmailAddresses[If[MatchQ[#,
   				PermissionsGroup[a_String /; StringMatchQ[a, "https://www.wolframcloud.com/objects/" ~~ __]]], #, # /. permissionGroupRules] &[ToExpression[#]]], $WolframID]} &, 
					Union@StringCases[contactsToAdd, Shortest["PermissionsGroup[" ~~ __ ~~ "]"]]], {CloudObject::cloudnf, Set::shape, CloudObject::srverr}];
					
		If[(PermissionsGroupsFailedAccess = Cases[PermissionsGroupsEmailListPairs, {a_, $Failed} :> a]) === {},
			EmailAddressesToSendToFromPermissionsGroups = Union[Flatten[Last /@ PermissionsGroupsEmailListPairs]];
			PermissionsGroupsToAdd = First/@PermissionsGroupsEmailListPairs,
			Throw[ChatTools`Private`$perc = False;
				MessageDialog[StringJoin["One or more permissions groups including ", 
							PermissionsGroupsFailedAccess[[1]], 
							" that you are trying to send to either does not exist or you are not allowed to access the member list(s) to chat with."], 
						WindowFrame -> "ModalDialog", WindowSize -> {600, All}]]];
		
		(* Email addresses in the NEW PARTICIPANTS input field. *)
		potentialEmailAddressesToAddToContacts = Union@DeleteCases[TextCases[contactsToAdd, "EmailAddress"], $WolframID];
		(* Email addresses in the NEW PARTICIPANTS input field as well as those in permissions groups in that field. *)
		emailAddressesToAdd = Union[potentialEmailAddressesToAddToContacts, EmailAddressesToSendToFromPermissionsGroups];
		
		emailListData = CurrentValue[nb, {TaggingRules, "EmailListData"}];
		(* Add email addresses from NEW PARTICIPANTS input field content and those whose checkboxes are checked that are not in the current participants list. *)
		contactsToAddToSend = Complement[Union[emailAddressesToAdd, Pick[contacts, emailListData, 1]], participantsWithoutUser];
		
		rules = (# -> 1 & /@ Union[Pick[contacts, emailListData, 1], emailAddressesToAdd]);
		newEmailListData = (Union[contacts, emailAddressesToAdd] /. rules) /. _String -> 0;
		contactsToDeleteFromSend = Intersection[Pick[contacts, emailListData, 0], oldparticipants];
		id = CurrentValue[nb, {TaggingRules, "ChatNotebookID"}];
		newListToSendTo = Complement[Union[participantsWithoutUser, contactsToAddToSend], contactsToDeleteFromSend];
		If[newListToSendTo === {},
		
			ChatTools`Private`$perc = False;
			MessageDialog["The list of contacts chosen must include at least one apart from yourself.", WindowFrame -> "ModalDialog", WindowSize -> {460, All}],
			
			newParticipantsList = Union[newListToSendTo, {$WolframID}]; 
			If[Sort@DeleteCases[newParticipantsList, $WolframID] === Sort@participantsWithoutUser,
			
				ChatTools`Private`$perc = False;
				(*MessageDialog["The list of participants has not changed.", WindowFrame -> "ModalDialog", WindowSize -> {300, All}]*)
				Module[{nb1 = ButtonNotebook[]},
					Needs["ChatTools`"];
					CurrentValue[nb1, DockedCells] = With[{id = CurrentValue[nb1, {TaggingRules, "ChatNotebookID"}], windowtitle = CurrentValue[nb1, WindowTitle]}, 
								ChatNotebookDockedCell[id, windowtitle, "CanSetVisibility" -> False, 
											"Preferences" -> PrivateChatPreferencesMenu[id, "UpdateParticipants" -> True]]];
					CurrentValue[nb1, {TaggingRules, "EmailListData"}] = CurrentValue[nb1, {TaggingRules, "BackupEmailListData"}];
					SetOptions[nb1, "BlinkingCellInsertionPoint" -> Inherited, "CellInsertionPointColor" -> Inherited, "CellInsertionPointCell" -> Inherited];
              				With[{nb2 = nb1}, 
						SetOptions[nb1,  
							NotebookEventActions -> {{"MenuCommand", "SimilarCellBelow"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
												{"MenuCommand", "SaveRename"} :> ChatTools`SaveChatNotebook[nb2],
								If[$VersionNumber < 12, Nothing, {"MenuCommand", "Copy"} :> Module[{in = InputNotebook[]}, 
									If[MatchQ[Developer`CellInformation[in], {___, {"Style" -> {_, __}, __, "CursorPosition" -> "CellBracket", __}, ___}], 
										CopyToClipboard[DeleteCases[ReplaceAll[NotebookRead@in,
						Cell[a_, b_String, _String, ___, TaggingRules -> _, ___] :> Cell[a, b]], Cell[__, "CellUserLabelOptions" | "CellLabelOptions" | "Stem", ___], Infinity]], 
										FrontEndExecute[{FrontEndToken[in, "Copy"]}]]]],
												PassEventsDown -> False}]];
					SelectionMove[nb1, After, Notebook, AutoScroll -> (!CurrentValue[nb1, {TaggingRules, "ScrollLock"}, False])];
					CurrentValue[nb1, {TaggingRules, "ManageParticipants"}] = Inherited;
					CurrentValue[nb1, {"TaggingRules", "ContactsToAdd"}] = ""],
				
				CurrentValue[nb, {TaggingRules, "Participants"}] = newParticipantsList;
				(*If[Not@TrueQ@CurrentValue[nb, {TaggingRules, "BeforeSend"}] && StringQ@id,
					SetOptions[ChannelObject[$WolframID <> ":" <> id], 
							Permissions -> Association[DeleteCases[newParticipantsList, $WolframID] -> {"Read", "Write"}, $WolframID -> {"Read", "Write", "Execute"}]]];*)
				If[(c1 = Complement[contactsToAddToSend, allparticipants]) =!= {}, CurrentValue[nb, {TaggingRules, "AllParticipants"}] = Join[allparticipants, c1]];
				If[(c2 = Complement[Join[PermissionsGroupsToAdd, potentialEmailAddressesToAddToContacts], contacts]) =!= {},
					CurrentValue[nb, {TaggingRules, "Contacts"}] = Union[contactsToAddToSend, contacts];
					CurrentValue[nb, {TaggingRules, "EmailListData"}] = newEmailListData];
				If[(* Only need to send out the modified participants list if a send has already been done. *)
					Not@TrueQ@CurrentValue[nb, {TaggingRules, "BeforeSend"}],
					
					compressedScreenName = Compress@(If[MemberQ[{"", Inherited}, #] || (StringQ@# && StringMatchQ[#, Whitespace]), "None",
																		#] &[CurrentValue[nb, {TaggingRules, "ScreenName"}]]);
					(* This updates the main Chat index and the individual chat index with new participant data for the key id. Also sends chat invitations for added participants and
					   sends a removed message to removed participants. *)
					URLExecute[CloudObject["https://www.wolframcloud.com/objects/a57f48fd-3574-499a-ad9a-fae3a98b9fd0"], {"ChatNotebookID" -> id, 
						"participants" -> Compress@newParticipantsList, "allparticipants" -> Compress@Join[allparticipants, contactsToAddToSend], 
						"addedParticipants" -> Compress@contactsToAddToSend, "removedParticipants" -> Compress@contactsToDeleteFromSend, 
						"screenname" -> compressedScreenName}, "WL", Method -> "POST"];
				(* If we could assume that the Chat index is updated by the time the handler function on the channel StringJoin[originator, ":", id] is ready to fire this would work. If
				   not we need to delay it until the api function called by the the URLExecute above is done. *)
				(* While[URLExecute[CloudObject["https://www.wolframcloud.com/objects/437f747a-85ac-4b3c-8d0a-c7239dcc29bc"], {"id" -> id}, "WL", Method -> "POST"] =!= True, 1];
				a = Association["privatechannel" -> StringJoin[originator, ":", id], "participants" -> Compress@Complement[contactsToAddToSend, contactsToDeleteFromSend]];
				Quiet[ChannelSend[#, a, ChannelPreSendFunction -> None]&/@(ChannelObject[# <> ":ChatInvitations"] & /@ Complement[contactsToAddToSend, contactsToDeleteFromSend]),
					ChannelObject::uauth];*)
					(* Assumes the channel was not deleted. *)
					(*originator = CurrentValue[nb, {TaggingRules, "Originator"}];*)
					Quiet[ChannelSend[StringJoin[#, ":", "ChatInvitations"], Association @@ List["ChatNotebookID" -> id, "addedParticipants" -> Compress@contactsToAddToSend,
											"removedParticipants" -> Compress@contactsToDeleteFromSend, "wid" -> $WolframID,
											"allparticipants" -> Compress@Join[allparticipants, Complement[contactsToAddToSend, allparticipants]],
											"screenname" -> compressedScreenName],
											ChannelPreSendFunction -> None], ChannelObject::uauth]&/@Union[contactsToDeleteFromSend, newListToSendTo, {$WolframID}(*contactsToAddToSend*)]];
				un = Union[PermissionsGroupsToAdd, potentialEmailAddressesToAddToContacts];
				If[Complement[un, CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}]] =!= {},
					CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}] = (Function[t, If[MemberQ[{Inherited, {}}, t],
						Sort@emailAddressesToAdd, Union[t, un]]][CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}]])];
				With[{nb1 = nb},
					CurrentValue[nb1, Deployed] = Inherited;
					SetOptions[nb1, "BlinkingCellInsertionPoint" -> Inherited, "CellInsertionPointColor" -> Inherited, "CellInsertionPointCell" -> Inherited,
							NotebookEventActions -> {{"MenuCommand", "SimilarCellBelow"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
										{"MenuCommand", "SaveRename"} :> ChatTools`SaveChatNotebook[nb1],
								If[$VersionNumber < 12, Nothing, {"MenuCommand", "Copy"} :> Module[{in = InputNotebook[]}, 
									If[MatchQ[Developer`CellInformation[in], {___, {"Style" -> {_, __}, __, "CursorPosition" -> "CellBracket", __}, ___}], 
										CopyToClipboard[DeleteCases[ReplaceAll[NotebookRead@in,
						Cell[a_, b_String, _String, ___, TaggingRules -> _, ___] :> Cell[a, b]], Cell[__, "CellUserLabelOptions" | "CellLabelOptions" | "Stem", ___], Infinity]], 
										FrontEndExecute[{FrontEndToken[in, "Copy"]}]]]],
										PassEventsDown -> False}];
					CurrentValue[nb1, DockedCells] = With[{id1 = id, windowtitle = CurrentValue[nb1, WindowTitle]}, 
										ChatNotebookDockedCell[id1, windowtitle, "CanSetVisibility" -> False, 
											"Preferences" -> PrivateChatPreferencesMenu[id1, "UpdateParticipants" -> True]]];
					banner = CurrentValue[nb1, {TaggingRules, "Banner"}];
					If[StringQ[banner = CurrentValue[nb1, {TaggingRules, "Banner"}]] && banner =!= "None", UpdateBanner[nb1, banner]];
					SelectionMove[nb1, After, Notebook, AutoScroll -> (!CurrentValue[nb1, {TaggingRules, "ScrollLock"}, False])];
					CurrentValue[nb1, {TaggingRules, "ManageParticipants"}] = Inherited;
					CurrentValue[nb1, {"TaggingRules", "ContactsToAdd"}] = ""]]]]];

Options[UpdateParticipants] = {"Active" -> True};

UpdateParticipants[opts___?OptionQ] := 
	Module[{active = ("Active"/.{opts}/.Options[UpdateParticipants]), nb = ButtonNotebook[], contactAddresses1, eld1},
		CurrentValue[nb, {TaggingRules, "ManageParticipants"}] = True;
		If[ListQ[#], If[Head@# === CellObject, NotebookDelete[#, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])]] & /@ #] &[CurrentValue[nb, {TaggingRules, "SelectedCells"}]];
		contactAddresses1 = Sort@Cases[DeleteCases[CurrentValue[nb, {TaggingRules, "Contacts"}](*CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}]*),
								x_String /; StringMatchQ[x, "PermissionsGroup" ~~ __]], x_ /; StringMatchQ[x, __ ~~ "\\@" ~~ __]]; 
		SetOptions[nb, NotebookDynamicExpression -> Inherited(*, NotebookEventActions -> None*), "BlinkingCellInsertionPoint" -> False, "CellInsertionPointColor" -> GrayLevel[1],
								"CellInsertionPointCell" -> None];
		If[ListQ[#], If[Head@# === CellObject,
						NotebookDelete[#, AutoScroll -> (! CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])]] & /@ #] &[CurrentValue[nb, {TaggingRules, "SelectedCells"}]];
		CurrentValue[nb, {TaggingRules, "BackupEmailListData"}] = CurrentValue[nb, {TaggingRules, "EmailListData"}];
		eld1 = (CurrentValue[nb, {TaggingRules, "EmailListData"}] /. {1 -> True, 0 -> False});
		Unprotect@ChatTools`Private`$perc; ChatTools`Private`$perc = False;
		CurrentValue[nb, DockedCells] = If[TrueQ@active,
			Cell[BoxData[With[{contactAddresses = contactAddresses1, eld = eld1},
			With[{in1 = nb},
				GridBox[{{""},{GridBox[{{StyleBox["MANAGE PARTICIPANTS", FontFamily -> "Source Sans Pro", FontColor -> GrayLevel[.39]], 
							ToBoxes@TextCell["(check to include/uncheck to exclude)", FontFamily -> "Source Sans Pro", FontColor -> GrayLevel[.65]]}}]},
						{FrameBox[PaneBox[GridBox[List/@With[{tr = CurrentValue[in1, {TaggingRules, "EmailListData"}]},
														Table[ReplaceAll[DynamicModuleBox[{ChatTools`Private`u$$ = tv}, 
	GridBox[{{GridBox[{{CheckboxBox[Dynamic[ChatTools`Private`u$$, (ChatTools`Private`u$$ = #1; Module[{ChatTools`Private`in$ = InputNotebook[]}, 
		CurrentValue[ChatTools`Private`in$, {TaggingRules, "EmailListData"}] = ReplacePart[CurrentValue[ChatTools`Private`in$, {TaggingRules, "EmailListData"}], 
		i -> If[TrueQ[ChatTools`Private`u$$], 1, 0]]]) &]], ButtonBox[ca, ButtonFunction :> (If[ChatTools`Private`u$$ === True, ChatTools`Private`u$$ = False, 
			ChatTools`Private`u$$ = True]; 
	Module[{ChatTools`Private`in$ = InputNotebook[]},
		CurrentValue[ChatTools`Private`in$, {TaggingRules, "EmailListData"}] = ReplacePart[CurrentValue[ChatTools`Private`in$, {TaggingRules, "EmailListData"}], 
		i -> If[TrueQ[ChatTools`Private`u$$], 1, 0]]]), BaseStyle -> {FontFamily -> "Source Sans Pro", FontColor -> GrayLevel[0.39`]}, Appearance -> None, 
		ContentPadding -> False, Evaluator -> Automatic, Method -> "Preemptive"]}}, GridBoxAlignment -> {"Columns" -> {Center, Left}}, AutoDelete -> False, 
			GridBoxItemSize -> {"Columns" -> {{Automatic}}, "Rows" -> {{Automatic}}}, 
			GridBoxSpacings -> {"ColumnsIndexed" -> {2 -> 1}, "Rows" -> {{Automatic}}}]}}, GridBoxAlignment -> {"Columns" -> {{Left}}}, AutoDelete -> False, 
			GridBoxItemSize -> {"Columns" -> {{Automatic}}, "Rows" -> {{Automatic}}}], DynamicModuleValues :> {}], 
																{i -> j, ca -> contactAddresses[[j]],
																	tv -> eld[[j]]}], {j, Length@contactAddresses}]], 
					GridBoxAlignment->{"Columns" -> {{Left}}}], AppearanceElements -> {}, Scrollbars -> {False, Automatic}, ImageSize -> {410, 200}], Background -> GrayLevel[1], 
									FrameStyle -> GrayLevel[.7]]},
						{GridBox[{{StyleBox["NEW PARTICIPANTS", FontFamily -> "Source Sans Pro", FontColor -> GrayLevel[.39]], 
							StyleBox["(comma separated)", FontFamily -> "Source Sans Pro", FontColor -> GrayLevel[.65]]}}]},
						{PaneBox[InputFieldBox[Dynamic[CurrentValue[EvaluationNotebook[], {"TaggingRules", "ContactsToAdd"}, ""]], String, FieldHint -> ToBoxes@TextCell["Enter WolframIDs and/or permissions groups"], 
								ImageSize -> {420, {50, 100000}}, BaseStyle -> {FontFamily -> "Source Sans Pro"}], 
							ImageSize -> {422, 54}, Scrollbars -> {False, Automatic}, AppearanceElements -> {}]},
						{GridBox[{{ButtonPairSequence[ButtonBox[StyleBox["Cancel", FontColor -> GrayLevel[0]], ButtonFunction :> Module[{nb1 = ButtonNotebook[]},
									Needs["ChatTools`"];
									CurrentValue[nb1, DockedCells] = With[{id = CurrentValue[nb1, {TaggingRules, "ChatNotebookID"}],
														windowtitle = CurrentValue[nb1, WindowTitle]}, 
												ChatNotebookDockedCell[id, windowtitle, "CanSetVisibility" -> False, 
													"Preferences" -> PrivateChatPreferencesMenu[id, "UpdateParticipants" -> True]]];
									CurrentValue[nb1, {TaggingRules, "EmailListData"}] = CurrentValue[nb1, {TaggingRules, "BackupEmailListData"}];
									SetOptions[nb1, "BlinkingCellInsertionPoint" -> Inherited, "CellInsertionPointColor" -> Inherited, "CellInsertionPointCell" -> Inherited];
              								With[{nb2 = nb1}, 
										SetOptions[nb1,  
											NotebookEventActions -> {{"MenuCommand", "SimilarCellBelow"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
												{"MenuCommand", "SaveRename"} :> ChatTools`SaveChatNotebook[nb2],
								If[$VersionNumber < 12, Nothing, {"MenuCommand", "Copy"} :> Module[{in = InputNotebook[]}, 
									If[MatchQ[Developer`CellInformation[in], {___, {"Style" -> {_, __}, __, "CursorPosition" -> "CellBracket", __}, ___}], 
										CopyToClipboard[DeleteCases[ReplaceAll[NotebookRead@in,
						Cell[a_, b_String, _String, ___, TaggingRules -> _, ___] :> Cell[a, b]], Cell[__, "CellUserLabelOptions" | "CellLabelOptions" | "Stem", ___], Infinity]], 
										FrontEndExecute[{FrontEndToken[in, "Copy"]}]]]],
												PassEventsDown -> False}]];
									SelectionMove[nb1, After, Notebook, AutoScroll -> (!CurrentValue[nb1, {TaggingRules, "ScrollLock"}, False])];
									CurrentValue[nb1, {TaggingRules, "ManageParticipants"}] = Inherited;
									CurrentValue[nb1, {"TaggingRules", "ContactsToAdd"}] = ""], 
									Evaluator -> "ChatServices", Active -> True, Method -> "Queued",
									Appearance -> {"Default" -> FrontEnd`ToFileName[{"Toolbars", "ChatTools"}, "CancelButton-Default.9.png"], 
											"Hover" -> FrontEnd`ToFileName[{"Toolbars", "ChatTools"}, "CancelButton-Hover.9.png"]}, Background -> GrayLevel[1], ImageSize -> {70, 25}],
							ButtonBox[DynamicBox[FEPrivate`Which[FEPrivate`And[FEPrivate`SameQ[CurrentValue[EvaluationNotebook[], {TaggingRules, "OKPercolate"}], True], 
														FEPrivate`SameQ[CurrentValue[$FrontEnd, "AllowChatServices"], True]],
												InterpretationBox[DynamicBox[FEPrivate`FrontEndResource["FEExpressions", "PercolateAnimator"][Medium],
												ImageSizeCache -> {50., {2., 10.}}], ProgressIndicator[Appearance -> "Percolate"], BaseStyle -> {"Deploy"}], 
											FEPrivate`Not[FEPrivate`SameQ[CurrentValue[$FrontEnd, "AllowChatServices"], True]],
											StyleBox["\[FilledSquare]", FontColor -> GrayLevel[1]], 
											True,
											StyleBox["   OK   ", FontColor -> GrayLevel[1]]]], 
								ButtonFunction :> Module[{bn = ButtonNotebook[]}, 
												CurrentValue[bn, {TaggingRules, "OKPercolate"}] = True; 
												Needs["ChatTools`"]; ChatTools`AuxiliaryUpdateParticipants[]; 
												CurrentValue[bn, {TaggingRules, "OKPercolate"}] = False], 
								Method -> "Queued", Evaluator -> "ChatServices", ImageSize -> {70, 25}, 
								Appearance -> {"ButtonType" -> "Default", "Default" -> FrontEnd`ToFileName[{"Toolbars", "ChatTools"}, "SendButton-Default.9.png"], 
											"Hover" -> FrontEnd`ToFileName[{"Toolbars", "ChatTools"}, "SendButton-Hover.9.png"]}, 
								Enabled -> Dynamic[TrueQ@CurrentValue[$FrontEnd, "AllowChatServices"]]]]}}]}, {""}}, 
						GridBoxSpacings -> {"Columns" -> {{Automatic}}, "RowsIndexed" -> {3 -> 0.5, 4 -> 1.5}}, GridBoxAlignment->{"Columns" -> {{Left}}}]]]], 
							TextAlignment -> Center, ShowCellBracket -> False, FontSize -> 13, Deployed -> True],
			Cell[BoxData[ToBoxes[With[{p = DeleteCases[CurrentValue[nb, {TaggingRules, "Participants"}], $WolframID]}, 
						Grid[{{Style["PARTICIPANTS", FontFamily -> "Source Sans Pro", FontColor -> GrayLevel[.39]]},
							{Framed[Pane[Grid[Table[{Button[p[[j]], Null, BaseStyle -> {FontFamily -> "Source Sans Pro", FontColor -> GrayLevel[.39]}, Appearance -> None, 
											ContentPadding -> False, Evaluator -> None, Enabled -> False]}, {j, Length@p}], Alignment -> Left], 
									AppearanceElements -> {}, Scrollbars -> {False, Automatic}, ImageSize -> {410, 200}],
								Background -> White, FrameStyle -> GrayLevel[.7]]},
							{Grid[{{Button["Cancel", Module[{nb1 = ButtonNotebook[]}, Needs["ChatTools`"]; CurrentValue[nb1, Editable] = True;
								CurrentValue[nb1, DockedCells] = With[{id = CurrentValue[nb1, {TaggingRules, "ChatNotebookID"}],
													windowtitle = CurrentValue[nb1, WindowTitle]},
															ChatTools`Private`ChatNotebookDockedCell[id, windowtitle,
																"CanSetVisibility" -> False,
																"Preferences" -> ActionMenu[gearImage,
																{"Participants" :> ChatTools`UpdateParticipants["Active" -> False],
																"Chat UUID" :> CopyToClipboard[id]},
																				Appearance -> "PopupMenu", 
																				AutoAction -> True, 
																ImageSize -> {Automatic, If[$OperatingSystem === "Windows", 25, 16]}, 
																				ContentPadding -> False]]];
            							With[{nb2 = nb1}, 
									SetOptions[nb2, "BlinkingCellInsertionPoint" -> Inherited, "CellInsertionPointColor" -> Inherited,
											"CellInsertionPointCell" -> Inherited,
											NotebookEventActions -> {{"MenuCommand", "SimilarCellBelow"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
													{"MenuCommand", "SaveRename"} :> ChatTools`SaveChatNotebook[nb2],
								If[$VersionNumber < 12, Nothing, {"MenuCommand", "Copy"} :> Module[{in = InputNotebook[]}, 
									If[MatchQ[Developer`CellInformation[in], {___, {"Style" -> {_, __}, __, "CursorPosition" -> "CellBracket", __}, ___}], 
										CopyToClipboard[DeleteCases[ReplaceAll[NotebookRead@in,
						Cell[a_, b_String, _String, ___, TaggingRules -> _, ___] :> Cell[a, b]], Cell[__, "CellUserLabelOptions" | "CellLabelOptions" | "Stem", ___], Infinity]], 
										FrontEndExecute[{FrontEndToken[in, "Copy"]}]]]],
													PassEventsDown -> False}]];
            							CurrentValue[nb1, Deployed] = Inherited], 
										Evaluator -> Automatic, Active -> True, Method -> "Queued", FrameMargins -> 1]}}]}}, 
							Spacings -> {Automatic, {2 -> .5, 3 -> 1.5}}, Alignment -> Left]]]],
			TextAlignment -> Center, ShowCellBracket -> False, FontSize -> 13]]];

(* participantsList includes user's wolfram id. *)

Options[auxiliaryCSendMessage] = {"Alias" -> None};

auxiliaryCSendMessage["Chat" -> participantsList_, "Contacts" -> contacts_, "ScreenName" -> screenname_, "ChatTitle" -> chatTitle_, opts___?OptionQ] := 
	Module[{nb = ButtonNotebook[], alias = ("Alias"/.{opts}/.Options[auxiliaryCSendMessage]), id, chatCreationDate, windowtitle,
	        dialogstyles = {"ChatDialogTitle", "ChatDialogHeader", "ChatDialogDelimiter", "ChatDialogControls"}},
		id = CreateUUID[];
		CurrentValue[nb, {TaggingRules, "ChatNotebookID"}] = id;
		CurrentValue[nb, Editable] = Inherited;
		If[StringQ@alias && StringMatchQ[alias, "" | Whitespace], alias = None];
		If[alias =!= None, CurrentValue[nb, {TaggingRules, "Alias"}] = alias];
		CurrentValue[nb, {TaggingRules, "Originator"}] = $WolframID;
		CurrentValue[nb, {TaggingRules, "ScreenName"}] = screenname;
		chatCreationDate = ToString@Now;
		CurrentValue[nb, {TaggingRules, "ChatCreationDate"}] = chatCreationDate;
		windowtitle = (If[chatTitle === "",
					StringJoin[If[# === "None",
							StringReplace[$WolframID, p__ ~~ "@" ~~ __ :> p],
							#]&[If[MemberQ[{"", Inherited}, screenname] || (StringQ@screenname && StringMatchQ[screenname, Whitespace]), "None", screenname]],
						"'s Chat - ",
						"("<>StringReplace[dateTime[TimeZoneAdjust@chatCreationDate], a__ ~~ ":" ~~ __ :> a]<>")"],
					chatTitle]);
		CurrentValue[nb, {TaggingRules, "CustomWindowTitle"}] = If[chatTitle === "", "None", chatTitle];
		CurrentValue[nb, {TaggingRules, "ChatNotebookDate"}] = StringReplace[DateString[], (WordCharacter ..) ~~ " " ~~ a__ :> a];
		CurrentValue[nb, {TaggingRules, "ChatNotebookWindowTitle"}] = windowtitle; 
		CurrentValue[nb, {TaggingRules, "Participants"}] = participantsList;
		CurrentValue[nb, {TaggingRules, "AllParticipants"}] = If[alias === None, participantsList, participantsList/.$WolframID->alias];
		CurrentValue[nb, {TaggingRules, "Contacts"}] = DeleteCases[Cases[contacts, x_ /; StringMatchQ[x, __ ~~ "\\@" ~~ __]], $WolframID];
		With[{pl = participantsList},
			CurrentValue[nb, {TaggingRules, "EmailListData"}] = (contacts /. a_String :> If[MemberQ[pl, a], 1, 0])(*Table[0, Length@participantsList - 1]*)];
		CurrentValue[nb, {TaggingRules, "OriginalWolframID"}] = $WolframID;
		CurrentValue[nb, WindowTitle] = windowtitle;
		NotebookDelete[Cells[nb, CellStyle->dialogstyles], AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])];
		CurrentValue[nb, Background] = Inherited;
		NotebookWrite[nb, Cell["Participants will be invited to this chat after your first message.", "HelpText"], AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])];
		NotebookWrite[nb, ChatTools`Private`$stemCellTemplate, All, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])];
		SelectionMove[nb, After, Cell, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])];
		Needs["ChatTools`"];
		CurrentValue[nb, DockedCells] = ChatNotebookDockedCell[id, windowtitle, "CanSetVisibility" -> False, "Preferences" -> PrivateChatPreferencesMenu[id, "UpdateParticipants" -> True]];
		SetOptions[nb, "BlinkingCellInsertionPoint" -> Inherited, "CellInsertionPointColor" -> Inherited, "CellInsertionPointCell" -> Inherited];
	With[{nb1 = nb, id1 = id, screenname1 = screenname},
		SetOptions[nb,  
				NotebookEventActions -> {{"MenuCommand", "SimilarCellBelow"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
							{"MenuCommand", "SaveRename"} :> ChatTools`SaveChatNotebook[nb1],
								If[$VersionNumber < 12, Nothing, {"MenuCommand", "Copy"} :> Module[{in = InputNotebook[]}, 
									If[MatchQ[Developer`CellInformation[in], {___, {"Style" -> {_, __}, __, "CursorPosition" -> "CellBracket", __}, ___}], 
										CopyToClipboard[DeleteCases[ReplaceAll[NotebookRead@in,
						Cell[a_, b_String, _String, ___, TaggingRules -> _, ___] :> Cell[a, b]], Cell[__, "CellUserLabelOptions" | "CellLabelOptions" | "Stem", ___], Infinity]], 
										FrontEndExecute[{FrontEndToken[in, "Copy"]}]]]],
			"WindowClose" :> (CurrentValue[nb1, Saveable] = False;
						If[Not@TrueQ@CurrentValue[nb1, {TaggingRules, "BeforeSend"}], 
					RunScheduledTask[Quiet[ChannelSend[StringJoin["chatframework@wolfram.com", ":", id1], Association @@ List["id" -> id1, "removedparticipant" -> $WolframID, 
		"screenname" -> Compress@If[MemberQ[{"", Inherited}, screenname1] || (StringQ@screenname1 && StringMatchQ[screenname1, Whitespace]), "None", screenname1], "celllist" -> Compress@{}], 
										ChannelPreSendFunction -> None], ChannelObject::uauth]; RemoveScheduledTask[$ScheduledTask];, {0.1}]]),
							PassEventsDown -> False},
				Saveable -> Inherited]];
	If[Not@StringQ@FirstCase[#["URL"] & /@ ChannelListeners[], a_String /; StringMatchQ[a, StringExpression[__, "/", $WolframID, "/", "ChatInvitations"]]],
		ChannelListen[ChannelObject[$WolframID <> ":ChatInvitations"], "TrustedChannel" -> True]]];

(*

CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}] does not include user wolfram id.

CurrentValue[chat nb notebook object, {TaggingRules, "Participants"}] includes user wolfram id.

*)

ChatTools`Private`$chatDialogParticipants="";ChatTools`Private`$chatDialogSessionTitle="";
newChatDialog[contacts_List]:=If[contacts==={},
  Notebook[{
    Cell[BoxData[GridBox[{{Cell[BoxData[ChatTools`Private`chatIcon]],StyleBox["  Start New Chat Session  ", "ChatDialogTitle"],ItemBox["",ItemSize->Fit],
    			Cell[BoxData[TagBox[ButtonBox["",
    						ButtonFunction :> (FEPrivate`FrontEndExecute[FrontEndToken["OpenHelpLink", {"paclet:workflowguide/UsingChat", 
									FEPrivate`If[CurrentValue["ShiftKey"], Null, FEPrivate`ButtonNotebook[]]}]]&),
						Evaluator -> Automatic, Appearance -> {"ButtonType" -> "Default", "Default" -> FrontEnd`ToFileName[{"Toolbars", "ChatTools"}, "question.png"], 
											"Hover" -> FrontEnd`ToFileName[{"Toolbars", "ChatTools"}, "questionHover.png"]}, ImageSize -> Full,
						Method -> "Queued"], MouseAppearanceTag["LinkHand"]]],
				"Text"]}}, GridBoxAlignment->{"Rows"->{{Center}}}]],"ChatDialogTitle"],
    Cell["","ChatDialogDelimiter"],
    Cell[TextData[{"Contacts"," ",StyleBox["(comma separated)","ChatDialogParenthetical"]}],"ChatDialogHeader"],
    Cell[BoxData[PaneBox[InputFieldBox[Dynamic[ChatTools`Private`$chatDialogParticipants],String,FieldHint->"Enter Wolfram IDs, permissions groups or an existing chat room",ImageSize->{420,{50,100000}},BaseStyle->{FontFamily->"Source Sans Pro"}],ImageSize->{425,54},Scrollbars->{False,Automatic},AppearanceElements->{}]],"ChatDialogControls"],
    Cell[TextData[{"Chat Session Title"," ",StyleBox["(optional)","ChatDialogParenthetical"]}],"ChatDialogHeader"],
    Cell[BoxData[PaneBox[InputFieldBox[Dynamic[ChatTools`Private`$chatDialogSessionTitle],String,ImageSize->{420,30},BaseStyle->{FontFamily->"Source Sans Pro"}],ImageSize->{425,54},AppearanceElements->{}]],"ChatDialogControls"],
    Cell["","ChatDialogDelimiter"],
    Cell[BoxData[GridBox[{{TemplateBox[{355},"Spacer1"], 
				ButtonPairSequence[ButtonBox[StyleBox["Cancel", FontColor -> GrayLevel[0]], ButtonFunction :> DialogReturn[], Appearance -> {"ButtonType" -> "Cancel", "Cancel" -> None}, 
						Background -> GrayLevel[.9], ImageSize -> {70, 25}, Evaluator -> "ChatServices"], 
				ButtonBox[DynamicBox[FEPrivate`Which[FEPrivate`And[FEPrivate`SameQ[CurrentValue[EvaluationNotebook[], {TaggingRules, "Percolate"}], True], FEPrivate`SameQ[CurrentValue[$FrontEnd, "AllowChatServices"], True]],
							InterpretationBox[DynamicBox[FEPrivate`FrontEndResource["FEExpressions", "PercolateAnimator"][Medium], ImageSizeCache -> {50., {2., 10.}}], 
								ProgressIndicator[Appearance -> "Percolate"], BaseStyle -> {"Deploy"}], 
							FEPrivate`Not[FEPrivate`SameQ[CurrentValue[$FrontEnd, "AllowChatServices"], True]],
							StyleBox["\[FilledSquare]", FontColor -> GrayLevel[1]],
							True,
							StyleBox["   Start   ", FontColor -> GrayLevel[1]]]], 
						ButtonFunction :> (CurrentValue[EvaluationNotebook[], {TaggingRules, "Percolate"}] = True; ChatTools`Private`SetUpChat[ButtonNotebook[], ChatTools`Private`$chatDialogParticipants,
														ChatTools`Private`$chatDialogSessionTitle]; CurrentValue[ButtonNotebook[], {TaggingRules, "Percolate"}] = False),
						Method -> "Queued", Evaluator -> "ChatServices", Background -> RGBColor[0., 0.5548332951857786, 1.], 
						ImageSize -> {70, 25},
						Appearance -> {"ButtonType" -> "Default", "Default" -> FrontEnd`ToFileName[{"Toolbars", "ChatTools"}, "SendButton-Default.9.png"], 
								"Hover" -> FrontEnd`ToFileName[{"Toolbars", "ChatTools"}, "SendButton-Hover.9.png"]},
						Enabled -> Dynamic[TrueQ@CurrentValue[$FrontEnd, "AllowChatServices"]]]]}}, 
			GridBoxAlignment -> {"Columns" -> {{Right}}}, GridBoxSpacings -> {"Columns" -> {{1}}}]], "ChatDialogControls", CellMargins -> {{Inherited, Inherited}, {10, 10}}]},
  
      Evaluator->"ChatServices",StyleDefinitions -> FrontEnd`FileName[{"Wolfram"}, "ChatTools.nb"],"TrackCellChangeTimes" -> False,TaggingRules -> {"ChatNotebook" -> "True", "SelectedCells" -> {},
						"EmailListData" -> If[ChatTools`$contactAddresses ==={}, {}, Table[0, Length@ChatTools`$contactAddresses]],
						"BeforeSend" -> True, "Originator" -> $WolframID, "ScrollLock"->False},CreateCellID -> True,CellLabelAutoDelete -> False, WindowTitle -> "Wolfram Chat",ShowStringCharacters -> False,
						"BlinkingCellInsertionPoint" -> False, "CellInsertionPointColor" -> GrayLevel[1], "CellInsertionPointCell" -> None, Editable -> False, Saveable -> False,
		NotebookEventActions->{"ReturnKeyDown":>FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
		  {"MenuCommand","EvaluateCells"}:>FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
		  {"MenuCommand","HandleShiftReturn"}:>FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
		  {"MenuCommand","EvaluateNextCell"}:>FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
		  "EscapeKeyDown":>(FE`Evaluate[FEPrivate`FindAndClickCancelButton[]];DialogReturn[$Failed]),
		  "WindowClose":>(FE`Evaluate[FEPrivate`FindAndClickCancelButton[]];DialogReturn[$Failed])},WindowSize -> {600, 550},
		  ScrollingOptions -> {"HorizontalScrollRange" -> Inherited, "PagewiseDisplay" -> Inherited, "ScrollUndo" -> Inherited, "SpeedParameters" -> Inherited, "VerticalScrollRange" -> Fit}],

  Notebook[{
    Cell[BoxData[GridBox[{{Cell[BoxData[ChatTools`Private`chatIcon]],StyleBox["  Start New Chat Session  ", "ChatDialogTitle"],ItemBox["",ItemSize->Fit],
    			Cell[BoxData[TagBox[ButtonBox["",
    						ButtonFunction :> (FEPrivate`FrontEndExecute[FrontEndToken["OpenHelpLink", {"paclet:workflowguide/UsingChat", 
									FEPrivate`If[CurrentValue["ShiftKey"], Null, FEPrivate`ButtonNotebook[]]}]]&),
						Evaluator -> Automatic, Appearance -> {"ButtonType" -> "Default", "Default" -> FrontEnd`ToFileName[{"Toolbars", "ChatTools"}, "question.png"], 
											"Hover" -> FrontEnd`ToFileName[{"Toolbars", "ChatTools"}, "questionHover.png"]}, ImageSize -> Full,
						Method -> "Queued"], MouseAppearanceTag["LinkHand"]]],
				"Text"]}}, GridBoxAlignment->{"Rows"->{{Center}}}]],"ChatDialogTitle"],
    Cell["","ChatDialogDelimiter"],
    Cell[TextData[{"New Contacts"," ",StyleBox["(comma separated)","ChatDialogParenthetical"]}],"ChatDialogHeader"],
    Cell[BoxData[PaneBox[InputFieldBox[Dynamic[ChatTools`Private`$chatDialogParticipants],String,FieldHint->"Enter Wolfram IDs, permissions groups or an existing chat room",ImageSize->{420,{50,100000}},BaseStyle->{FontFamily->"Source Sans Pro"}],ImageSize->{425,54},Scrollbars->{False,Automatic},AppearanceElements->{}]],"ChatDialogControls"],
    Cell[TextData[{"Previous Contacts"," ",StyleBox["(check to include)","ChatDialogParenthetical"]}],"ChatDialogHeader"],
    Cell[BoxData[FrameBox[
    PaneBox[GridBox[Table[With[{j$$ = j}, {ToBoxes@DynamicModule[{u = False}, Grid[{{Grid[{{Checkbox[Dynamic[u, (u = #; Module[{in = InputNotebook[]}, 
		CurrentValue[in, {TaggingRules, "EmailListData"}] = ReplacePart[CurrentValue[in, {TaggingRules, "EmailListData"}], j$$ -> If[TrueQ@u, 1, 0]]]) &]], 
			Button[ChatTools`$contactAddresses[[j]], If[u === True, u = False, u = True]; Module[{in = InputNotebook[]}, 
		CurrentValue[in, {TaggingRules, "EmailListData"}] = ReplacePart[CurrentValue[in, {TaggingRules, "EmailListData"}], j$$ -> If[TrueQ@u, 1, 0]]], 
				BaseStyle -> {FontFamily -> "Source Sans Pro", FontColor -> GrayLevel[.39]}, Appearance -> None, ContentPadding -> False]}}, Spacings -> {{2 -> 1}, Automatic}, 
														Alignment -> {{Center, Left}}]}}, Alignment -> Left]]}], {j, Length@ChatTools`$contactAddresses}], 
					GridBoxAlignment -> {"Columns"->{{Left}}}], AppearanceElements -> {}, Scrollbars -> {False, Automatic}, ImageSize -> {410, 100}], Background -> White, FrameStyle -> GrayLevel[.7]]],"ChatDialogControls"],
    Cell[TextData[{"Chat Session Title"," ",StyleBox["(optional)","ChatDialogParenthetical"]}],"ChatDialogHeader"],
    Cell[BoxData[PaneBox[InputFieldBox[Dynamic[ChatTools`Private`$chatDialogSessionTitle],String,ImageSize->{420,30},BaseStyle->{FontFamily->"Source Sans Pro"}],ImageSize->{425,54},AppearanceElements->{}]],"ChatDialogControls"],
    Cell["","ChatDialogDelimiter"],
    Cell[BoxData[GridBox[{{TemplateBox[{355},"Spacer1"], 
				ButtonPairSequence[ButtonBox[StyleBox["Cancel", FontColor -> GrayLevel[0]], ButtonFunction :> DialogReturn[], Appearance -> {"ButtonType" -> "Cancel", "Cancel" -> None}, 
						Background -> GrayLevel[.9], ImageSize -> {70, 25}, Evaluator -> "ChatServices"], 
				ButtonBox[DynamicBox[FEPrivate`Which[FEPrivate`And[FEPrivate`SameQ[CurrentValue[EvaluationNotebook[], {TaggingRules, "Percolate"}], True], FEPrivate`SameQ[CurrentValue[$FrontEnd, "AllowChatServices"], True]],
							InterpretationBox[DynamicBox[FEPrivate`FrontEndResource["FEExpressions", "PercolateAnimator"][Medium], ImageSizeCache -> {50., {2., 10.}}], 
								ProgressIndicator[Appearance -> "Percolate"], BaseStyle -> {"Deploy"}], 
							FEPrivate`Not[FEPrivate`SameQ[CurrentValue[$FrontEnd, "AllowChatServices"], True]],
							StyleBox["\[FilledSquare]", FontColor -> GrayLevel[1]],
							True,
							StyleBox["   Start   ", FontColor -> GrayLevel[1]]]], 
						ButtonFunction :> (CurrentValue[EvaluationNotebook[], {TaggingRules, "Percolate"}] = True; ChatTools`Private`SetUpChat[ButtonNotebook[], ChatTools`Private`$chatDialogParticipants,
														ChatTools`Private`$chatDialogSessionTitle]; CurrentValue[ButtonNotebook[], {TaggingRules, "Percolate"}] = False),
						Method -> "Queued", Evaluator -> "ChatServices", Background -> RGBColor[0., 0.5548332951857786, 1.], 
						ImageSize -> {70, 25},
						Appearance -> {"ButtonType" -> "Default", "Default" -> FrontEnd`ToFileName[{"Toolbars", "ChatTools"}, "SendButton-Default.9.png"], 
								"Hover" -> FrontEnd`ToFileName[{"Toolbars", "ChatTools"}, "SendButton-Hover.9.png"]},
						Enabled -> Dynamic[TrueQ@CurrentValue[$FrontEnd, "AllowChatServices"]]]]}}, 
			GridBoxAlignment -> {"Columns" -> {{Right}}}, GridBoxSpacings -> {"Columns" -> {{1}}}]], "ChatDialogControls", CellMargins -> {{Inherited, Inherited}, {10, 10}}]},
  
      Evaluator->"ChatServices",StyleDefinitions -> FrontEnd`FileName[{"Wolfram"}, "ChatTools.nb"],"TrackCellChangeTimes" -> False,TaggingRules -> {"ChatNotebook" -> "True", "SelectedCells" -> {},
						"EmailListData" -> If[ChatTools`$contactAddresses ==={}, {}, Table[0, Length@ChatTools`$contactAddresses]],
						"BeforeSend" -> True, "Originator" -> $WolframID, "ScrollLock"->False},CreateCellID -> True,CellLabelAutoDelete -> False, WindowTitle -> "Wolfram Chat",ShowStringCharacters -> False,
						"BlinkingCellInsertionPoint" -> False, "CellInsertionPointColor" -> GrayLevel[1], "CellInsertionPointCell" -> None, Editable -> False, Saveable -> False,
		NotebookEventActions->{"ReturnKeyDown":>FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
		  {"MenuCommand","EvaluateCells"}:>FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
		  {"MenuCommand","HandleShiftReturn"}:>FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
		  {"MenuCommand","EvaluateNextCell"}:>FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
		  "EscapeKeyDown":>(FE`Evaluate[FEPrivate`FindAndClickCancelButton[]];DialogReturn[$Failed]),
		  "WindowClose":>(FE`Evaluate[FEPrivate`FindAndClickCancelButton[]];DialogReturn[$Failed])},WindowSize -> {600, If[$OperatingSystem === "Unix", 600, 550]},
		  ScrollingOptions -> {"HorizontalScrollRange" -> Inherited, "PagewiseDisplay" -> Inherited, "ScrollUndo" -> Inherited, "SpeedParameters" -> Inherited, "VerticalScrollRange" -> Fit}]];
		  
(* Determine absolute value for Automatic in default window height *)
ChatTools`Private`ysize[] := (ChatTools`Private`ysize[] = Module[{nbobj, size}, 
	nbobj = CreateWindow[Visible->False, WindowSize->Automatic, WindowTitle->"size" (*to keep from incrementing Untitled-n*)];
	size = AbsoluteCurrentValue[nbobj, WindowSize][[2]];
	NotebookClose[nbobj];
	size
]);

PermissionsGroupEmailAddresses[pergrp_] := 
	Internal`InheritedBlock[{CloudObject`Private`userUUIDToDisplay}, 
				DownValues[CloudObject`Private`userUUIDToDisplay] = DownValues[CloudObject`Private`userUUIDToDisplay] /. "displayName" -> "email";
				pergrp["Members"]];
		  
SetUpChat[nb2_NotebookObject, participants_String, title_String] := 
	Module[{addedContactAddresses, SavedContacts = DeleteCases[CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}], $WolframID], 
		checkedContacts, addressesInCheckedItems, premissionGroupRules, AddedPermissionsGroupPairs, UnableToAccessAddedPermissionsGroups, PermissionsGroupsInCheckedItemsPairs,
		UnableToAccessPermissionsGroupsInCheckedItemsPairs, PermissionsGroupsFailedAccess, contactAddressesToSendTo, roomsInparticipants, otherRooms},
  
		If[Not@TrueQ@CurrentValue[$FrontEnd, "WolframCloudConnected"],
   
			CurrentValue[EvaluationNotebook[], {TaggingRules, "Percolate"}] = False;
			FirstLoginMessage[];
			Abort[],
   
			addedContactAddresses = DeleteCases[TextCases[participants, "EmailAddress"], $WolframID];
			If[SavedContacts === Inherited, SavedContacts = {}];
   			checkedContacts = If[SavedContacts === {}, {}, Pick[SavedContacts, CurrentValue[nb2, {TaggingRules, "EmailListData"}], 1]];
   			addressesInCheckedItems = Cases[checkedContacts, x_String /; StringMatchQ[x, __ ~~ "\\@" ~~ __]];
   			
   			premissionGroupRules = {PermissionsGroup[a_String] :> PermissionsGroup[$WolframID, a], 
						PermissionsGroup[a_String, b_String /; StringFreeQ[b, "https://www.wolframcloud.com/objects/"]] :> PermissionsGroup[a, 
									"https://www.wolframcloud.com/objects/" <> StringReplace[a, "@wolfram.com" -> ""] <> "/PermissionsGroup/" <> b]};
   			AddedPermissionsGroupPairs = Quiet[Map[{#, DeleteCases[PermissionsGroupEmailAddresses[If[MatchQ[#,
   				PermissionsGroup[a_String /; StringMatchQ[a, "https://www.wolframcloud.com/objects/" ~~ __]]], #, # /. permissionGroupRules] &[ToExpression[#]]], $WolframID]} &, 
				Union@StringCases[participants, Shortest["PermissionsGroup[" ~~ __ ~~ "]"]]], {CloudObject::cloudnf, Set::shape, CloudObject::srverr}];
			UnableToAccessAddedPermissionsGroups = Cases[AddedPermissionsGroupPairs, {a_, $Failed} :> a];
				
   			PermissionsGroupsInCheckedItemsPairs = Quiet[Map[{#, DeleteCases[PermissionsGroupEmailAddresses[If[MatchQ[#,
   				PermissionsGroup[a_String /; StringMatchQ[a, "https://www.wolframcloud.com/objects/" ~~ __]]], #, # /. permissionGroupRules] &[ToExpression[#]]], $WolframID]} &, 
				Union@Cases[checkedContacts, x_String /; StringMatchQ[x, "PermissionsGroup[" ~~ __ ~~ "]"]]], {CloudObject::cloudnf, Set::shape, CloudObject::srverr}];
			UnableToAccessPermissionsGroupsInCheckedItemsPairs = Cases[PermissionsGroupsInCheckedItemsPairs, {a_, $Failed} :> a];
			
			PermissionsGroupsFailedAccess = Union[UnableToAccessAddedPermissionsGroups, UnableToAccessPermissionsGroupsInCheckedItemsPairs];
   			
   			Which[PermissionsGroupsFailedAccess =!= {},
   			
   				ChatTools`Private`$perc = False;
   				If[UnableToAccessPermissionsGroupsInCheckedItemsPairs =!= {},
   					CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}] = DeleteCases[CurrentValue[$FrontEnd,
   									{PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}], Alternatives@@UnableToAccessPermissionsGroupsInCheckedItemsPairs]];
   				MessageDialog[StringJoin["One or more permissions groups including ",
   								PermissionsGroupsFailedAccess[[1]],
   								" that you are trying to send to either does not exist or you are not allowed to access the member list(s) to chat with."], 
						WindowFrame -> "ModalDialog", WindowSize -> {600, All}],
   			
   				Or[addedContactAddresses =!= {}, addressesInCheckedItems =!= {}, AddedPermissionsGroupPairs =!= {}, PermissionsGroupsInCheckedItemsPairs =!= {}],
    
				contactAddressesToSendTo = Union[Flatten[{addedContactAddresses, addressesInCheckedItems, DeleteCases[Last/@AddedPermissionsGroupPairs, $WolframID],
										DeleteCases[Last/@PermissionsGroupsInCheckedItemsPairs, $WolframID]}]];
				If[Complement[Flatten[{addedContactAddresses, First/@AddedPermissionsGroupPairs}], SavedContacts] =!= {}, 
					CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}] = Union[SavedContacts,
																		Flatten[{addedContactAddresses,
																				First/@AddedPermissionsGroupPairs}]]];
    
				CurrentValue[EvaluationNotebook[], {TaggingRules, "Percolate"}] = False;
				ChatTools`auxiliaryCSendMessage["Chat" -> Prepend[DeleteCases[contactAddressesToSendTo, $WolframID], $WolframID], 
									"Contacts" -> Union[Cases[SavedContacts, _String?(StringMatchQ[#, __ ~~ "\\@" ~~ __] &)], contactAddressesToSendTo], 
									"ScreenName" -> ChatTools`$ScreenName,
									"ChatTitle" -> If[StringMatchQ[title, "" | Whitespace], "", title]]; 
				SetOptions[nb2, Editable -> Inherited, ScrollingOptions -> {"HorizontalScrollRange" -> Inherited, "PagewiseDisplay" -> Inherited, "ScrollUndo" -> Inherited, 
												"SpeedParameters" -> Inherited, "VerticalScrollRange" -> Inherited}];
				CurrentValue[nb2, ShowStringCharacters] = Inherited;
				
				SetOptions[nb2, WindowSize -> {600, ChatTools`Private`ysize[]}, WindowMargins -> Inherited];
				ChatTools`Private`$chatDialogParticipants = ""; ChatTools`Private`$chatDialogSessionTitle = "",
				
				True,
				
				Which[(roomsInparticipants = Cases[StringSplit[participants, ","], x_String /; Not@StringMatchQ[x, "" | Whitespace]]) =!= {},
     
					NotebookClose[nb2];
					ChatTools`auxiliaryRoomSendMessage["Chat" -> ToLowerCase[roomsInparticipants[[1]]]];
					ChatTools`Private`$chatDialogParticipants = ""; ChatTools`Private`$chatDialogSessionTitle = "",
     
					(otherRooms = Cases[checkedContacts, x_String /; Not@StringMatchQ[x, __ ~~ "\\@" ~~ __]]) =!= {},
     
					NotebookClose[nb2];
					ChatTools`auxiliaryRoomSendMessage["Chat" -> ToLowerCase[otherRooms[[1]]]];
					ChatTools`Private`$chatDialogParticipants = ""; ChatTools`Private`$chatDialogSessionTitle = "";,
     
					True,
     
					CurrentValue[EvaluationNotebook[], {TaggingRules, "Percolate"}] = False;
					MessageDialog["Enter a list of Wolfram IDs (which should not include your Wolfram ID) and/or one or more permissions groups that you are a member of. Alternatively, you may specify a chat room by its short name.", WindowFrame -> "ModalDialog", WindowSize -> {580, All}]]]]];

cSendMessage["Chat"] :=
	(ChatTools`$contactAddresses = (DeleteCases[If[MatchQ[#, {_String ..}], #, {}], $WolframID]&[DeleteCases[CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}], $WolframID]]);
	ChatTools`$ScreenName = (If[StringQ@# && Not@StringMatchQ[#, ""|Whitespace], StringTrim@#, ""]&[CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "ScreenName"}]]);
	NotebookPut[newChatDialog[ChatTools`$contactAddresses], Evaluator->"ChatServices"]);
	
chatinitDialogNotebook[] := 
	Notebook[{Cell["", "ChatInitDialogSpacer"], 
		Cell[BoxData[RowBox[{ToBoxes[ProgressIndicator[Appearance -> "Necklace"]], ToBoxes[Spacer[10]], Cell[TextData["Initializing chat. Please wait\[Ellipsis]"], "ChatInitDialogText"]}]], "ChatInitDialogBottomGrid"],
		Cell["", "ChatInitDialogSpacer"]}, 
		WindowTitle -> "Initializing\[Ellipsis]", 
		StyleDefinitions -> Notebook[{Cell[StyleData["ChatInitDialogText"], CellMargins -> {{30, 30}, {6, 8}}, FontFamily :> CurrentValue["PanelFontFamily"], FontColor -> GrayLevel[.2], FontSize -> 13, 
								ShowCellBracket -> False, ShowStringCharacters -> False], 
						Cell[StyleData["ChatInitDialogBottomGrid"], CellMargins -> {{30, 30}, {50, 8}}, GridBoxOptions -> {GridBoxAlignment -> {"Columns" -> {{Left}}}},
							ShowCellBracket -> False, ShowStringCharacters -> False], 
						Cell[StyleData["ChatInitDialogSpacer"], CellMargins -> {{0, 0}, {4, 4}}, ShowCellBracket -> False, ShowStringCharacters -> False, CellSize -> {Automatic, 1}]}, 
						StyleDefinitions -> "Default.nb"],
		WindowSize -> Fit, WindowElements -> None, WindowFrame -> "ModelessDialog", ShowSelection -> False, Editable -> False, Saveable -> False, Selectable -> False, Background -> GrayLevel[1], 
		WindowFrameElements -> {"CloseBox"}, ShowAutoSpellCheck -> False];

Chat[] := 
	Module[{m},
	If[Not@TrueQ@$InitialChatSmallDone, m = NotebookPut@chatinitDialogNotebook[]];
	Which[Not@TrueQ@CurrentValue["InternetConnectionAvailable"] || Not@TrueQ@CurrentValue["AllowDownloads"],
	
		NotebookClose@m;
		MessageDialog["An internet connection is required to use Chat Services. Check your network connection, and be sure to check \"Allow the Wolfram System to access the Internet\" in your preferences. To change your internet preferences choose \"Internet & Mail Settings...\" from the Help menu.", WindowFrame -> "ModalDialog", WindowSize -> {600, All}],
				
		Not@$CloudConnected,
		
		NotebookClose@m;
		Module[{i = 0, s},
			Quiet[Catch[Do[s = CloudConnect[];
					Switch[s, $Canceled, Throw["Close"], i++; $Failed, "Again", _String, Throw["Proceed"]], {3}]];
			If[s =!= $Canceled,
			
				If[i === 3 && s === $Failed,
				
					MessageDialog["The login credentials you typed are incorrect.", WindowFrame -> "ModalDialog", WindowSize -> {370, All}],
					
					TaskExecute@SessionSubmit[ScheduledTask[If[TrueQ@$InitialChatSmallDone, cSendMessage["Chat"]; TaskRemove[$CurrentTask]], Quantity[1, "Seconds"]]]], 
					
				MessageDialog["Chat functionality requires connecting to the Wolfram Cloud.", WindowFrame -> "ModalDialog", WindowSize -> {400, All}]],
				{CloudConnect::creds, CloudConnect::notauth, General::stop}]],
		True,
				
		TaskExecute@SessionSubmit[ScheduledTask[If[TrueQ@$InitialChatSmallDone, NotebookClose@m; cSendMessage["Chat"]; TaskRemove[$CurrentTask]], Quantity[1, "Seconds"]]]]];
	
Options[SendChat] = {"ChatTitle" -> ""};

SendChat[participants_ /; (StringQ@participants || VectorQ[participants, StringQ]), opts___?OptionQ] :=
	Module[{ct = ("ChatTitle" /. {opts} /. Options[SendChat])}, 
		If[StringQ@ct,
			SendChat[participants, "", "ChatTitle" -> ct],
			SendChat[participants, ""]]];

SendChat[participants_ /; (StringQ@participants || VectorQ[participants, StringQ]), message_String, opts___?OptionQ] := 
	Module[{recipientsList = If[StringQ@participants, {participants}, participants], alias = None, id, nb, st, fs, savedContacts, comp},
		If[Not@StringQ@$WolframID, $CloudConnected];
		If[(participants === $WolframID) || (participants === {$WolframID}),
		
			MessageDialog["Specify at least one participant apart from yourself.", WindowFrame -> "ModalDialog", WindowSize -> {330, All}],
			
			With[{participantsList = If[Not@MemberQ[recipientsList, $WolframID], Prepend[recipientsList, $WolframID], recipientsList], 
				screenname = (If[StringQ@# && Not@StringMatchQ[#, "" | Whitespace], StringTrim@#, ""] &[CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "ScreenName"}]]), 
				chatTitle = ("ChatTitle" /. {opts} /. Options[SendChat])},
				
				id = CreateUUID[];
				chatCreationDate = ToString@Now;
				windowtitle = (If[chatTitle === "", 
							StringJoin[If[# === "None",
									StringReplace[$WolframID, p__ ~~ "\\@" ~~ __ :> p],
									#] &[If[MemberQ[{"", Inherited}, screenname] || (StringQ@screenname && StringMatchQ[screenname, Whitespace]), "None", screenname]], 
										"'s Chat - ", 
										"("<>StringReplace[dateTime[TimeZoneAdjust@chatCreationDate], a__ ~~ ":" ~~ __ :> a]<>")"],
							chatTitle]);
				savedContacts = (If[MatchQ[#, {_String ..}], #, {}] &[DeleteCases[CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}], $WolframID]]);
				nb = NotebookPut[Notebook[{Cell["Participants will be invited to this chat after your first message.", "HelpText"], $stemCellTemplate},
							"TrackCellChangeTimes" -> False, 
							TaggingRules -> {"ChatNotebookID" -> id, "ScreenName" -> screenname, "ChatCreationDate" -> chatCreationDate, 
								"CustomWindowTitle" -> If[chatTitle === "", "None", chatTitle], 
								"ChatNotebookDate" -> StringReplace[DateString[], (WordCharacter ..) ~~ " " ~~ a__ :> a], "ChatNotebookWindowTitle" -> windowtitle, 
								"Participants" -> participantsList, "AllParticipants" -> participantsList, "Contacts" -> savedContacts, 
								"EmailListData" -> With[{pl = participantsList}, savedContacts /. a_String :> If[MemberQ[pl, a], 1, 0]], WindowTitle -> windowtitle,
								"ChatNotebook" -> "True", "SelectedCells" -> {}, "BeforeSend" -> True, "Originator" -> $WolframID, "ScrollLock" -> False},
							CreateCellID -> True, 
							StyleDefinitions -> FrontEnd`FileName[{"Wolfram"}, "ChatTools.nb"],
							CellLabelAutoDelete -> False,
							WindowTitle -> windowtitle, 
							ShowStringCharacters -> False,
							Evaluator -> "ChatServices"]];
				SelectionMove[nb, After, Notebook];
				CurrentValue[nb, DockedCells] = ChatNotebookDockedCell[id, windowtitle, "CanSetVisibility" -> False, "Preferences" -> PrivateChatPreferencesMenu[id]];
				With[{nb1 = nb}, SetOptions[nb,  
								NotebookEventActions -> {{"MenuCommand", "SimilarCellBelow"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
											{"MenuCommand", "SaveRename"} :> ChatTools`SaveChatNotebook[nb1],
								If[$VersionNumber < 12, Nothing, {"MenuCommand", "Copy"} :> Module[{in = InputNotebook[]}, 
									If[MatchQ[Developer`CellInformation[in], {___, {"Style" -> {_, __}, __, "CursorPosition" -> "CellBracket", __}, ___}], 
										CopyToClipboard[DeleteCases[ReplaceAll[NotebookRead@in,
						Cell[a_, b_String, _String, ___, TaggingRules -> _, ___] :> Cell[a, b]], Cell[__, "CellUserLabelOptions" | "CellLabelOptions" | "Stem", ___], Infinity]], 
										FrontEndExecute[{FrontEndToken[in, "Copy"]}]]]],
											PassEventsDown -> False}]];
				If[Not@StringQ@FirstCase[#["URL"] & /@ ChannelListeners[], a_String /; StringMatchQ[a, StringExpression[__, "/", $WolframID, "/", "ChatInvitations"]]],
					ChannelListen[ChannelObject[$WolframID <> ":ChatInvitations"], "TrustedChannel" -> True];
				
				If[message =!= "",
					st = Cells[nb, CellStyle -> "Stem"];
					If[st =!= {},
						fs = First[st];
						SetOptions[fs, Deletable -> True];
						NotebookWrite[nb, {Cell[message, "Text"], $stemCellTemplate}, AutoScroll -> (!CurrentValue[nb, {TaggingRules, "ScrollLock"}, False])]; 
						SendChatNotebookInformation[nb]]];
						
				savedContacts = (If[MatchQ[#, {_String ..}], #, {}] &[DeleteCases[CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}], $WolframID]]); 
				If[(comp = Complement[DeleteCases[participantsList, $WolframID], If[(savedContacts === Inherited) || Not@VectorQ[savedContacts, StringQ], {}, savedContacts]]) =!= {}, 
					CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}] = Union[comp, savedContacts]]]]]];
					
Quiet[TaskExecute@SessionSubmit[ScheduledTask[If[And[TrueQ@CurrentValue[$FrontEnd, "WolframCloudConnected"],
							TrueQ@CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "AvailableToReceiveChatInvitations"}]], InitializeChatSmall[];
     TaskRemove[$CurrentTask]], {Now - Quantity[12, "Seconds"], 15}]];
	SessionSubmit[ScheduledTask[RestartListeners[], Quantity[15, "Seconds"]]]],

(************ LOCAL (non-sandbox) KERNEL DEFINITIONS ************)
ChatTools`CreateChat::incorcred = "The login credentials you typed are incorrect.";
ChatTools`CreateChat::mustcc = "Chat functionality requires connecting to the Wolfram Cloud.";
ChatTools`CreateChat::inter = "An internet connection is required to use Chat Services.";
ChatTools`CreateChat::usage = "CreateChat[] creates a new chat notebook.";
ChatTools`CreateChat[] :=
	Module[{i = 0, s},
		If[Not@TrueQ@CurrentValue["InternetConnectionAvailable"] || Not@TrueQ@CurrentValue["AllowDownloads"],
		
			Message[CreateChat::inter],
			
			If[Not@$CloudConnected,
			
				Quiet[Catch[Do[s = CloudConnect[];
						Switch[s, $Canceled, Throw["Close"], i++; $Failed, "Again", _String, Throw["Proceed"]], {3}]];
					If[s =!= $Canceled,
					
						If[i === 3 && s === $Failed,
						
							Message[CreateChat::incorcred],
							
							FrontEndTokenExecute["NewChat"]],
							
					Message[CreateChat::mustcc]],
					{CloudConnect::creds, CloudConnect::notauth, General::stop}],
					
				FrontEndTokenExecute["NewChat"]]]];
Attributes[ChatTools`CreateChat] = {Protected, ReadProtected};

ChatTools`CreateChatRoom::incorcred = "The login credentials you typed are incorrect.";
ChatTools`CreateChatRoom::mustcc = "Chat functionality requires connecting to the Wolfram Cloud.";
ChatTools`CreateChatRoom::empt = "Both the short name and title must be nonempty strings.";
ChatTools`CreateChatRoom::inter = "An internet connection is required to use Chat Services.";
ChatTools`CreateChatRoom::indacc = "There is a problem accessing cloud data for this chat. Check your network and Wolfram Cloud connections and try again.";
ChatTools`CreateChatRoom::chan = "Could not create the channel for this chat. Check your network and Wolfram Cloud connections and try again.";
ChatTools`CreateChatRoom::chanbro = "Could not connect the channel broker to this chat. Check your network and Wolfram Cloud connections and try again.";
ChatTools`CreateChatRoom::api = "Could not deploy the API for sending or receiving messages in this chat. Check your network and Wolfram Cloud connections and try again.";
ChatTools`CreateChatRoom::rmshex = "A chat room with the same short name already exists.";
ChatTools`CreateChatRoom::maxcr = "Your login can only have at most 10 chat rooms associated with it.";
ChatTools`CreateChatRoom::notcr = "Unable to create the chat room at this time. Try again later.";
ChatTools`CreateChatRoom::usage = "CreateChatRoom[\"shortname\", \"title\"] creates a chat room with the specified short name and title.";

Options[ChatTools`CreateChatRoom] = {"RoomType" -> "OpenGroup", "Moderators" -> None};

CreateChatRoom[shortname_String, title_String, opts___?OptionQ] := 
	Module[{i = 0, s},
		If[Not@TrueQ@CurrentValue["InternetConnectionAvailable"] || Not@TrueQ@CurrentValue["AllowDownloads"],
		
			Message[CreateChatRoom::inter],
			
			If[Not@$CloudConnected,
			
				Quiet[Catch[Do[s = CloudConnect[];
						Switch[s, $Canceled, Throw["Close"], i++; $Failed, "Again", _String, Throw["Proceed"]], {3}]];
					If[s =!= $Canceled,
					
						If[i === 3 && s === $Failed,
						
							Message[CreateChatRoom::incorcred],
							
							auxiliaryCreateChatRoom[shortname, title, opts]],
							
					Message[CreateChatRoom::mustcc]],
					{CloudConnect::creds, CloudConnect::notauth, General::stop}],
					
				auxiliaryCreateChatRoom[shortname, title, opts]]]];
Attributes[ChatTools`CreateChatRoom] = {Protected, ReadProtected};
    
Options[auxiliaryCreateChatRoom] = {"RoomType" -> "OpenGroup", "Moderators" -> None};

auxiliaryCreateChatRoom[shortname_String, title_String, opts___?OptionQ] :=
	Module[{shortname1, title1, i = 0, s, type = ("RoomType" /. {opts} /. Options[auxiliaryCreateChatRoom]), teachersAdded = ("Moderators" /. {opts} /. Options[auxiliaryCreateChatRoom]),
		shortnameAvailable, contacts},
		shortname1 = StringTrim@ToLowerCase@shortname;
  		title1 = StringTrim@title;
  		If[StringMatchQ[shortname1, "" | Whitespace] || StringMatchQ[title1, "" | Whitespace],
			
			Message[CreateChatRoom::empt],

			shortnameAvailable = URLExecute[CloudObject["https://www.wolframcloud.com/objects/eeb3eb02-d951-46cd-a092-f801f43e782d"], {"Shortcut" -> Compress@shortname1}, "WL"];
			Which[shortnameAvailable === "Your login can only have at most 10 chat rooms associated with it.",
				Message[CreateChatRoom::maxcr],
				Not[(shortnameAvailable === True) || (Head@shortnameAvailable === CloudObject)], 
				Message[CreateChatRoom::indacc],
				Head@shortnameAvailable === CloudObject, 
				Message[CreateChatRoom::rmshex]; shortnameAvailable,
				True, 
				With[{wolframid = $WolframID, shortname2 = shortname1, title2 = title1, type1 = type, teachersAdded1 = Compress@teachersAdded}, 
					Which[# === "Could not create the channel for this room.",
						Message[CreateChatRoom::chan],
						# === "Could not make the channel broker action function for this room.",
						Message[CreateChatRoom::chanbro],
						# === "Could not cloud deploy the api function for modifying this room's index.",
						Message[CreateChatRoom::api],
						# === "There is a problem accessing the room index.",
						Message[CreateChatRoom::indacc],
						StringQ@#,
						Message[CreateChatRoom::notcr],
						True,
						contacts = CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}];
						Which[contacts === Inherited,
							CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}] = {shortname2},
							ListQ@contacts,
							CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}] = Append[contacts, shortname2]];
  						#]&[URLExecute[CloudObject["https://www.wolframcloud.com/objects/8b699812-11d1-4297-8469-e0050c2b1f00"],
							{"Creator" -> wolframid, "ShortCut" -> Compress@shortname2, "Name" -> Compress@title2, "RoomType" -> type1, "Moderators" -> teachersAdded1}, "WL",
								Method -> "POST"]]]]]];
Attributes[ChatTools`Private`auxiliaryCreateChatRoom] = {Protected, ReadProtected};

ChatTools`DeleteChatRoom::incorcred = "The login credentials you typed are incorrect.";
ChatTools`DeleteChatRoom::mustcc = "Chat functionality requires connecting to the Wolfram Cloud.";
ChatTools`DeleteChatRoom::inter = "An internet connection is required to use Chat Services.";
ChatTools`DeleteChatRoom::notex = "A chat room with that short name does not exist.";
ChatTools`DeleteChatRoom::notper = "You do not have permission to delete that chat room.";
ChatTools`DeleteChatRoom::usage = "DeleteChatRoom[\"shortname\"] deletes the chat room with the specified short name.";
DeleteChatRoom[shortname_String] := 
	Module[{i = 0, s},
		If[Not@TrueQ@CurrentValue["InternetConnectionAvailable"] || Not@TrueQ@CurrentValue["AllowDownloads"],
		
			Message[DeleteChatRoom::inter],
			
			If[Not@$CloudConnected,
			
				Quiet[Catch[Do[s = CloudConnect[];
						Switch[s, $Canceled, Throw["Close"], i++; $Failed, "Again", _String, Throw["Proceed"]], {3}]];
					If[s =!= $Canceled,
					
						If[i === 3 && s === $Failed,
						
							Message[DeleteChatRoom::incorcred],
							
							auxiliaryDeleteChatRoom[shortname]],
							
					Message[DeleteChatRoom::mustcc]],
					{CloudConnect::creds, CloudConnect::notauth, General::stop}],
					
				auxiliaryDeleteChatRoom[shortname]]]];
Attributes[ChatTools`DeleteChatRoom] = {Protected, ReadProtected};

auxiliaryDeleteChatRoom[shortname_String] := 
	Module[{shortnameAvailable, rem, contacts, shortname1 = StringTrim@ToLowerCase@shortname},
		shortnameAvailable = URLExecute[CloudObject["https://www.wolframcloud.com/objects/eeb3eb02-d951-46cd-a092-f801f43e782d"], {"Shortcut" -> Compress@shortname1}, "WL"];
			If[shortnameAvailable === True,
			
				Message[ChatTools`DeleteChatRoom::notex],
		
				rem = URLExecute[CloudObject["https://www.wolframcloud.com/objects/71caf01b-b555-4cd1-a450-99f34bb8b421"], {"Shortcut" -> Compress@shortname1}, "WL", Method -> "POST"];
				If[rem === "You do not have permissions to delete this chat room.",
				
					Message[ChatTools`DeleteChatRoom::notper],
					
					contacts = CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}];
					Which[contacts === {shortname1},
						CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}] = Inherited,
						ListQ@contacts,
						CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "Contacts"}] = DeleteCases[contacts, shortname1]]];]];
Attributes[ChatTools`Private`auxiliaryDeleteChatRoom] = {Protected, ReadProtected};


ChatTools`ChatRoomModerators::incorcred = "The login credentials you typed are incorrect.";
ChatTools`ChatRoomModerators::mustcc = "Chat functionality requires connecting to the Wolfram Cloud.";
ChatTools`ChatRoomModerators::inter = "An internet connection is required to use Chat Services.";
ChatTools`ChatRoomModerators::wcp = "There is a problem getting data from the Wolfram Cloud. Try again later.";
ChatTools`ChatRoomModerators::ndne = "A chat room with that short name does not exist.";
ChatTools`ChatRoomModerators::nmr = "Your Wolfram ID is not a moderator for this chat room.";
ChatTools`ChatRoomModerators::usage = "ChatRoomModerators[shortname] returns the list of moderators for the chat room having short name shortname. One must be a room moderator to get the moderator list.";
ChatRoomModerators[shortname_String /; Not@StringMatchQ[shortname, "" | Whitespace]]:=
	Module[{i = 0, s},
		If[Not@TrueQ@CurrentValue["InternetConnectionAvailable"] || Not@TrueQ@CurrentValue["AllowDownloads"],
		
			Message[ChatRoomModerators::inter],
			
			If[Not@$CloudConnected,
			
				Quiet[Catch[Do[s = CloudConnect[];
						Switch[s, $Canceled, Throw["Close"], i++; $Failed, "Again", _String, Throw["Proceed"]], {3}]];
					If[s =!= $Canceled,
					
						If[i === 3 && s === $Failed,
						
							Message[ChatRoomModerators::incorcred],
							
							auxiliaryChatRoomModerators[shortname]],
							
					Message[ChatRoomModerators::mustcc]],
					{CloudConnect::creds, CloudConnect::notauth, General::stop}],
					
				auxiliaryChatRoomModerators[shortname]]]];
Attributes[ChatTools`ChatRoomModerators] = {Protected, ReadProtected};

auxiliaryChatRoomModerators[shortname_] :=
	Module[{shortname1 = Compress@ToLowerCase@StringTrim@shortname, data, message = "There is a problem accessing the room index.", message1 = "A chat room with that short name does not exist.", 
		message3 = "Your Wolfram ID is not a moderator for this chat room."},
		data = URLExecute[CloudObject["https://www.wolframcloud.com/objects/c8262397-5f5c-4f32-9e27-672819b7605b"], {"ShortName" -> shortname1, "wid" -> $WolframID}, "WL"];
		Switch[data,
			message,
			Message[ChatRoomModerators::wcp],
			message1,
			Message[ChatRoomModerators::ndne],
			message3,
			Message[ChatRoomModerators::nmr],
			_,
			data]];
Attributes[ChatTools`Private`auxiliaryChatRoomModerators] = {Protected, ReadProtected};

ChatTools`SetChatRoomModerators::incorcred = "The login credentials you typed are incorrect.";
ChatTools`SetChatRoomModerators::mustcc = "Chat functionality requires connecting to the Wolfram Cloud.";
ChatTools`SetChatRoomModerators::inter = "An internet connection is required to use Chat Services.";
ChatTools`SetChatRoomModerators::wcp = "There is a problem getting data from the Wolfram Cloud. Try again later.";
ChatTools`SetChatRoomModerators::ndne = "A chat room with that short name does not exist.";
ChatTools`SetChatRoomModerators::nmr = "Your Wolfram ID is not a moderator for this chat room.";
ChatTools`SetChatRoomModerators::ndp = "The suggested change to the list of moderators is not different from what is already present.";
ChatTools`SetChatRoomModerators::usage = "SetChatRoomModerators[shortname, newmoderatorslist] updates the list of moderators for the chat room having short name shortname. One must be a room moderator to modify a chat room's moderators list.";
SetChatRoomModerators[shortname_String /; Not@StringMatchQ[shortname, "" | Whitespace],
		moderators_ /; (VectorQ[moderators, StringQ] && AllTrue[moderators, TextCases[StringTrim@#, "EmailAddress"] === {StringTrim@#} &]) || moderators === None] :=
	Module[{i = 0, s},
		If[Not@TrueQ@CurrentValue["InternetConnectionAvailable"] || Not@TrueQ@CurrentValue["AllowDownloads"],
		
			Message[SetChatRoomModerators::inter],
			
			If[Not@$CloudConnected,
			
				Quiet[Catch[Do[s = CloudConnect[];
						Switch[s, $Canceled, Throw["Close"], i++; $Failed, "Again", _String, Throw["Proceed"]], {3}]];
					If[s =!= $Canceled,
					
						If[i === 3 && s === $Failed,
						
							Message[SetChatRoomModerators::incorcred],
							
							auxiliarySetChatRoomModerators[shortname, moderators]],
							
					Message[SetChatRoomModerators::mustcc]],
					{CloudConnect::creds, CloudConnect::notauth, General::stop}],
					
				auxiliarySetChatRoomModerators[shortname, moderators]]]];
Attributes[ChatTools`SetChatRoomModerators] = {Protected, ReadProtected};		
		
auxiliarySetChatRoomModerators[shortname_, moderators_] :=		
	Module[{shortname1 = Compress@ToLowerCase@StringTrim@shortname, moderators1 = Compress@Which[moderators === {}, None, VectorQ[moderators, StringQ], StringTrim /@ moderators, True, moderators],
		data, originator, id, type, name, screenname}, 
			data = URLExecute["https://www.wolframcloud.com/objects/33bf668d-0d71-4c00-87c6-12093615690a", {"Shortcut" -> shortname1, "wid" -> $WolframID}, "WL", Method -> "POST"];
			Which[data === "Failure",
				Message[SetChatRoomModerators::wcp], 
				data === "Not present",
				Message[SetChatRoomModerators::ndne], 
				data === "Your Wolframid is not a moderator for this chat room.", 
				Message[SetChatRoomModerators::nmr], 
				Complement[Uncompress[data["Moderators"]] /. None -> {}, {data["originator"]}] === Complement[moderators /. None -> {}, {data["originator"]}], 
				Message[SetChatRoomModerators::ndp],
				True, 
				originator = data["originator"];
				id = data["ID"]; 
				type = data["RoomType"];
				name = Uncompress@data["Name"];
				screenname = CurrentValue[$FrontEnd, {PrivateFrontEndOptions, "InterfaceSettings", "ChatTools", "ScreenName"}]; 
				ChannelSend["https://channelbroker.wolframcloud.com/users/" <> "chatframework@wolfram.com" <> "/" <> id, 
						Association["sender" -> $WolframID, "requesteraddress" -> "None", "allparticipants" -> Compress@{},"id" -> id, "compressedCellList" -> Compress@{}, 
							"name" -> Compress@name, "alias" -> $WolframID, "originator" -> originator, "SpecialAction" -> "UpdateModerators", "Banner" -> Compress@{},
							"RoomType" -> type, "Moderators" -> moderators1, "Shortcut" -> shortname1, "ScreenName" -> Compress@screenname],
					ChannelPreSendFunction -> None]]];
Attributes[ChatTools`Private`auxiliarySetChatRoomModerators] = {Protected, ReadProtected};

ChatTools`ChatRooms::incorcred = "The login credentials you typed are incorrect.";
ChatTools`ChatRooms::mustcc = "Chat functionality requires connecting to the Wolfram Cloud.";
ChatTools`ChatRooms::inter = "An internet connection is required to use Chat Services.";
ChatTools`ChatRooms::wcp = "There is a problem getting data from the Wolfram Cloud. Try again later.";
ChatTools`ChatRooms::usage = "ChatRooms[] returns a list of pairs of the form {short name, room title}.";
ChatRooms[] := 
	Module[{i = 0, s},
		If[Not@TrueQ@CurrentValue["InternetConnectionAvailable"] || Not@TrueQ@CurrentValue["AllowDownloads"],
		
			Message[ChatRooms::inter],
			
			If[Not@$CloudConnected,
			
				Quiet[Catch[Do[s = CloudConnect[];
						Switch[s, $Canceled, Throw["Close"], i++; $Failed, "Again", _String, Throw["Proceed"]], {3}]];
					If[s =!= $Canceled,
					
						If[i === 3 && s === $Failed,
						
							Message[ChatRooms::incorcred],
							
							auxiliaryChatRooms[]],
							
					Message[ChatRooms::mustcc]],
					{CloudConnect::creds, CloudConnect::notauth, General::stop}],
					
				auxiliaryChatRooms[]]]];
Attributes[ChatTools`ChatRooms] = {Protected, ReadProtected};

auxiliaryChatRooms[] := 
	Module[{data = URLExecute[CloudObject["https://www.wolframcloud.com/objects/715951b3-1bbd-49ba-aa39-2c56aae50531"], {"wid" -> $WolframID}, "WL"]},
		If[StringQ@data,
			Message[ChatRooms::wcp],
			data]];
Attributes[ChatTools`Private`auxiliaryChatRooms] = {Protected, ReadProtected};

ChatTools`ChatServicesEnableDialog;

ChatServicesEnableDialog[type:("NewChat" | "ChatServicesMonitor")] := 
	NotebookPut@With[{button1 = ButtonBox[StyleBox["Cancel", FontColor -> GrayLevel[0]], ButtonFunction :> DialogReturn[],
								Appearance -> {"ButtonType" -> "Cancel", "Cancel" -> None}, Background -> GrayLevel[.9],
								Evaluator -> Automatic, Method -> "Preemptive", ImageSize -> {70, 25}],
			button2 = With[{type1 = type}, ButtonBox[DynamicBox[FEPrivate`If[FEPrivate`SameQ[CurrentValue[EvaluationNotebook[], {TaggingRules, "Percolate"}], True],
										InterpretationBox[DynamicBox[FEPrivate`FrontEndResource["FEExpressions", "PercolateAnimator"][Medium],
																					ImageSizeCache -> {50., {2., 10.}}], 
													ProgressIndicator[Appearance -> "Percolate"], BaseStyle -> {"Deploy"}], 
													StyleBox["   Enable   ", FontColor -> GrayLevel[1]]]],
						ButtonFunction :> Quiet[CurrentValue[ButtonNotebook[], {TaggingRules, "Percolate"}] = True;
									Catch[If[Not@TrueQ@CurrentValue["InternetConnectionAvailable"] || Not@TrueQ@CurrentValue["AllowDownloads"],
										Throw[DialogReturn[];
											MessageDialog["An internet connection is required to use Chat Services. Check your network connection, and be sure to check \"Allow the Wolfram System to access the Internet\" in your preferences. To change your internet preferences choose \"Internet & Mail Settings...\" from the Help menu.", WindowFrame -> "ModalDialog", WindowSize -> {600, All}]]];
									If[Not@TrueQ@CurrentValue[$FrontEnd, "WolframCloudConnected"], CloudConnect[]];
									If[TrueQ@CurrentValue[$FrontEnd, "WolframCloudConnected"],
										CurrentValue[ButtonNotebook[], {TaggingRules, "Percolate"}] = True;
										CurrentValue[$FrontEnd, "AllowChatServices"] = True; 
										FrontEndTokenExecute[type1],
										DialogReturn[]]],
									CloudConnect::clver], 
						Appearance -> {"ButtonType" -> "Default", "Default" -> FrontEnd`ToFileName[{"Toolbars", "ChatTools"}, "SendButton-Default.9.png"], 
						"Hover" -> FrontEnd`ToFileName[{"Toolbars", "ChatTools"}, "SendButton-Hover.9.png"]}, Method -> "Queued", Evaluator -> Automatic, 
										Background -> RGBColor[0., 0.5548332951857786, 1.]]]},
		Notebook[{Cell["", CellMargins -> {{Automatic, Automatic}, {1, 1}}], 
				Cell[TextData[{"Enable Chat Services",
						Cell[BoxData@ToBoxes@Spacer[40]]}], "EnableChatTitleText"], 
						Cell[BoxData[ToBoxes@TextCell[Row[{Mouseover @@ (Button[Style["Wolfram Chat", #], 
													(FEPrivate`FrontEndExecute[FrontEndToken["OpenHelpLink", {"paclet:workflowguide/UsingChat", 
														FEPrivate`If[CurrentValue["ShiftKey"], Null, FEPrivate`ButtonNotebook[]]}]]&),
													Method -> "Queued", Appearance -> None,
													BaseStyle -> "Text"] & /@ {RGBColor[0.11375, 0.48235, 0.749], RGBColor[0.8549, 0.3961, 0.1451]}), 
										Style[" is a notebook-based messaging interface for exchanging text, code, and other content in real time. Chat services must be enabled in your preferences to join chat rooms, send chats, or receive invitations.", "Text"]}]]], "EnableChatText"], 
						Cell[BoxData[GridBox[{If[$OperatingSystem === "MacOSX", {button1, button2}, {button2, button1}]}, 
									GridBoxAlignment -> {"Columns" -> {{Right}}}, GridBoxSpacings -> {"Columns" -> {{1}}}]], "EnableChatButtons"]},
				WindowSize -> {650, 220}, 
				ShowCellBracket -> False,
				"CellInsertionPointCell" -> {}, 
				"BlinkingCellInsertionPoint" -> False, 
				"CellInsertionPointColor" -> GrayLevel[1], 
				WindowFloating -> True,
				WindowElements -> {}, 
				WindowFrameElements -> {"CloseBox"},
				ShowStringCharacters -> False, 
				Background -> GrayLevel[1], 
				ScrollingOptions -> {"PagewiseScrolling" -> False, "PagewiseDisplay" -> True, "VerticalScrollRange" -> Fit}, 
				ShowCellBracket -> False,
				CellMargins -> {{0, 0}, {0, 0}}, 
				AutoMultiplicationSymbol -> False,
				Saveable -> False, 
				WindowTitle -> "Wolfram Chat",
				Editable -> False, 
				Selectable -> False, 
				StyleDefinitions -> Notebook[{Cell[StyleData["EnableChatTitleText"], FontSize -> 20, FontFamily -> "Source Sans Pro", FontColor -> RGBColor[.2, .2, .2],
									ShowCellBracket -> False, CellMargins -> {{30, 30}, {2, 14}}, ShowAutoSpellCheck -> False], 
								Cell[StyleData["EnableChatText"], FontSize -> 12, FontFamily -> "Source Sans Pro", FontColor -> RGBColor[.39215, .39215, .39215], 
									ShowCellBracket -> False, CellMargins -> {{30, 30}, {2, 14}}], 
								Cell[StyleData["EnableChatButtons"], TextAlignment -> Right, CellMargins -> {{30, 30}, {2, 15}}, 
									ButtonBoxOptions -> {ImageSize -> {80, 24}, BaseStyle -> {FontFamily -> "Source Sans Pro", FontSize -> 14}}]}], 
				NotebookEventActions -> {"ReturnKeyDown" :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
							{"MenuCommand", "EvaluateCells"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
							{"MenuCommand", "HandleShiftReturn"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
							{"MenuCommand", "EvaluateNextCell"} :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]], 
							"EscapeKeyDown" :> (FE`Evaluate[FEPrivate`FindAndClickCancelButton[]]; DialogReturn[$Failed]), 
							"WindowClose" :> (FE`Evaluate[FEPrivate`FindAndClickCancelButton[]]; DialogReturn[$Failed])},
				TaggingRules -> {"EnableChatServices" -> "True"}, 
				NotebookDynamicExpression -> Dynamic[Refresh[If[FE`Evaluate[FEPrivate`EvaluatorStatus["ChatServices"]] === "Running", 
										NotebookClose@SelectFirst[Notebooks[], CurrentValue[#, {TaggingRules, "EnableChatServices"}] === "True" &]],
										UpdateInterval -> 1]]]];
Attributes[ChatTools`ChatServicesEnableDialog] = {Protected, ReadProtected};
Attributes[ChatTools`Private`$perc] = {Protected, ReadProtected};

Quiet@Remove["ChatTools`*"];
Quiet@Remove["ChatTools`Private`*"];


];

End[];



EndPackage[];


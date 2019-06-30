(* ::Package:: *)

(* :Name: DemonstrationsTools` *)

(* :CVS Keywords:  $Id: DemonstrationsTools.m,v 1.23 2010/05/03 13:32:06 lou Exp $ *)

(* :Title: Utilities for Demonstrations Creation  *)

(* :Author: Andre Kuzniarek, Jay Warendorff, Buddy Ritchie, Jerry Walsh, Lou D'Andria *)

(* :Copyright: (c) 2006, Wolfram Research, Inc. All rights reserved. *)

(* :Mathematica Version: 6.0 *)

(* :Package Version: 0.01 *)

(* :Summary: Utilities for Demonstrations Creation Palettes. *)



BeginPackage[ "DemonstrationsTools`"]

$DemonstrationsToolsDir::usage =
"Installation directory for the DemonstrationsTools package.";

$DemonstrationsToolsDir::notfound =
"WARNING: The DemonstrationsTools installation directory was not found.
Please set the paramater $DemonstrationsToolsDir to point to the installation
directory; otherwise, certain features of DemonstrationsTools may not work as
expected.";

CellInf::usage = "Utility function for returning CellInformation packet data.";
MsgToConsole::usage = "Utility function for sending messages to the Messages notebook.";

SaveBrowseWithMemory::usage =
"Normal SaveBrowse remembers the path to previous saves, but shares this
memory with (and hence, can be trumped by an intervening) OpenBrowse.
This enhanced SaveBrowse maintains a path memory seperate from OpenBrowse
(see $LastSavePath).";

PreflightCheck::usage = ""
GetCurrentPackage::usage = ""

WebLink::usage = 
"Convert selected URL into button. In future will apply special 
formatting based on target.";

DemonstrationExampleOpen::usage = 
"Download and open example file from web, or open local file if there 
is no web connection or the file is missing online.";
DemonstrationTemplateOpen::usage = 
"Download and open template file from web, or open local file if there 
is no web connection or the file is missing online.";
DemonstrationTestMask::usage = 
"Opens a transparent notebook with a window for testing the size of demonstration
page objects.";

UpdateManipulateOutputs::usage = ""



Begin["`Private`"]

$DemonstrationsToolsDir =
  Quiet @ Check[
    DirectoryName[FindFile["DemonstrationsTools`"], 2],
    MessageToConsole[$DemonstrationsToolsDir::notfound]; $Failed
  ];


CellInf[ selNB_NotebookObject] := MathLink`CallFrontEnd[ FrontEnd`CellInformation[ selNB]]



SetAttributes[MsgToConsole, HoldFirst]

MsgToConsole[symbolWithValue_, values___] :=
 Module[{presetMessageOptionsValues, newMessageOptions, cs},
        (* Get current MessageOptions to restore after message is sent to console. *)
        presetMessageOptionsValues = MessageOptions /. Options[$FrontEnd, MessageOptions];
        
        (* New MessageOptions has "KernelMessageAction" with "PrintToConsole". *)
        newMessageOptionsValues = If[(cs = Cases[presetMessageOptionsValues, a : ("KernelMessageAction" -> _)]; cs) === {}, 
                                  Append[presetMessageOptionsValues, "KernelMessageAction" -> {"Beep", "PrintToNotebook"}],
        presetMessageOptionsValues /. ("KernelMessageAction" -> a_) :> ("KernelMessageAction" -> 
                  If[StringQ[a],"PrintToConsole", Append[DeleteCases[a,"PrintToNotebook"],"PrintToConsole"]])]; 
        SetOptions[$FrontEnd, MessageOptions -> newMessageOptionsValues]; 
        Message[symbolWithValue, values];
        SetSelectedNotebook[MessagesNotebook[]];
        (* Restore previous MessageOptions. *)
        SetOptions[$FrontEnd, MessageOptions -> presetMessageOptionsValues]]




Clear[$LastSavePath];

SaveBrowseWithMemory[] := SaveBrowseWithMemory[ButtonNotebook[]];

SaveBrowseWithMemory[nb_NotebookObject] :=
  Module[
    {name, file},
    name = GuessFileName[nb];
    file =
      ToFileName[{
        If[ValueQ[$LastSavePath], $LastSavePath, Directory[]]
      }, name];
    file = SystemDialogInput["FileSave", file];
    If[file =!= $Canceled,
      $LastSavePath = DirectoryName[file];
      NotebookSave[nb, file];
    ];
  ];

(*
  Derives a likely filename for an open notebook, trying (in order):
    - already-saved-under filename
    - window title
    - stock name ("filename.nb")
*)
GuessFileName[nb_NotebookObject] :=
  Module[
    {info, name},
    info = {"FileName", "WindowTitle"}
      /.  NotebookInformation[nb]
        /. {"FileName" -> None, "WindowTitle" -> None};
    name =
      Switch[info,
        {_String, _},
          info[[1]],
        {_FrontEnd`FileName, _},
          info[[1]] /. FrontEnd`FileName[_, n_, ___] :> n,
        {_, _String},
          info[[2]],
        _,
          "filename.nb"
      ];
    If[StringMatchQ[name, "*.nb"],
      name,
      name <> ".nb"
    ]
  ];


PreflightCheck::nosave = "Notebook must be saved before it can be uploaded.";

PreflightCheck[] := Module[{ nb = ButtonNotebook[]},
    
    If[ StringMatchQ[ "WindowTitle" /. NotebookInformation[ nb], "FileNameField.nb"],
      SetOptions[ ButtonNotebook[], ScreenStyleEnvironment -> "FileNameField"]; Abort[]];
      
    If[ StringMatchQ[ "WindowTitle" /. NotebookInformation[ nb], "*.nb"],
      NotebookSave[ nb]; NotebookLocate[{ URL["http://demonstrations.wolfram.com/upload.html"], None}],
   (* DemonstrationsTools`MsgToConsole[ PreflightCheck::nosave] *)
   (* CreateDocument[{ 
        Cell[ "Your demonstration notebook must be saved \nbefore it can be uploaded.", 
          FontFamily -> "Verdana", FontSize -> 12, 
          CellMargins -> {{20, 20}, {15, 30}}, 
          TextAlignment -> Center], 
        Cell[ BoxData[
          ButtonBox["OK", 
            ButtonFunction :> FrontEnd`NotebookClose[ FrontEnd`ButtonNotebook[]], 
            Appearance -> "DialogBox", 
            ButtonFrame -> "DialogBox", 
            Evaluator -> None]], 
          TextAlignment -> Center]}, 
        WindowSize -> {380, 130}, 
        WindowFrame -> "Palette", 
        Background -> White, 
        ScrollingOptions -> {"VerticalScrollRange" -> 1, "HorizontalScrollRange" -> 1},
        WindowFrameElements -> {"CloseBox"},
        WindowElements -> {}, 
        WindowTitle -> None, 
        Deployed -> True, 
        ShowCellBracket -> False] *)
      MessageDialog["Your Demonstration must be saved in your file system before it \
can be uploaded (please use the save button in the authoring notebook header).",
        WindowSize -> {330, 120}]];  (* Hack because linewrapping not working on Windows, bug 77649 *)
    ]


GetCurrentPackage[] := Null;


WebLink[] := Module[{ nb = ButtonNotebook[], info, URLstring},
    
    nb = ButtonNotebook[];
    info = DemonstrationsTools`CellInf[ nb];
    Which[ 
      info === $Failed,
        Abort[],
      {"CellBracket"} === ("CursorPosition" /. info),
        SelectionMove[ nb, All, CellContents],
      MatchQ["CursorPosition" /. info, {"CellBracket", "CellBracket" ..}],
        Abort[],
      SameQ @@ Flatten @ ("CursorPosition" /. info),
        FrontEndTokenExecute[ nb, "ExpandSelection"],
      True,
        Null];
      
    URLstring = NotebookRead[ nb];
    
    Which[
      StringQ[ URLstring] === False,
        SelectionMove[ nb, After, Selection];
        Abort[],
      StringMatchQ[ URLstring, "http:*"],
        NotebookWrite[nb, TextData[ButtonBox[URLstring, ButtonData -> {URL[URLstring], None}, DefaultContentStyle -> "Hyperlink"]]],
      StringMatchQ[ URLstring, "www.*"] || StringMatchQ[ URLstring, "*/*"] || StringMatchQ[ URLstring, "*.*"],
        URLstring = "http://" <> URLstring;
        NotebookWrite[nb, TextData[ButtonBox[URLstring, ButtonData -> {URL[URLstring], None}, DefaultContentStyle -> "Hyperlink"]]],
      StringQ[ URLstring] && Not[StringMatchQ[URLstring,""|(" "..)]],
      FrontEndTokenExecute[nb, "CreateHyperlinkDialog"],
      True,
        SelectionMove[ nb, After, Selection];
        Abort[]];
        
    ]

DemonstrationExampleOpen::nofile = "File not found.";

DemonstrationExampleOpen[]:= Module[{nb},
	Quiet[nb = Get[ToFileName[{DemonstrationsTools`$DemonstrationsToolsDir, "FrontEnd", "TextResources"},
						"DemonstrationExample.nb"]];
		If[Head@nb === Notebook, NotebookPut[nb]]
	]
]

DemonstrationExampleOpen[ url_String] := Module[{nb},
   Quiet[ 
     nb = Import[ url, "Text"]; (* Using text import to bypass dynamic warnings *)
     If[ nb === $Failed, 
       nb = Import[ ToFileName[{ $InstallationDirectory, "AddOns", "Applications", 
                      "DemonstrationsTools", "FrontEnd", "TextResources"},  "DemonstrationExample.nb"], "Text"]]
     ];
     If[ nb =!= $Failed,
       NotebookPut @ ReleaseHold @ ToExpression[ nb, StandardForm, Hold],
       DemonstrationsTools`MsgToConsole[ DemonstrationExampleOpen::nofile]]
   ]

(*** Alternate version which avoids use of Import[]:

DemonstrationExampleOpen[ url_String] := Module[{msgOpts, webFile, layoutFile, webNbExpr},
    msgOpts = Options[ $FrontEnd, MessageOptions];
    SetOptions[ $FrontEnd, MessageOptions -> {"ErrorAction" -> {}}];
    webFile = NotebookOpen[ url, Visible -> False];
    layoutFile = ToFileName[{ $InstallationDirectory, "AddOns", "Applications", 
                   "DemonstrationsTools", "FrontEnd", "TextResources"},  "DemonstrationExample.nb"];
    webNbExpr = NotebookGet @ webFile;
    If[ webFile =!= $Failed,
      If[ MatchQ[ webNbExpr, Notebook[{Cell[ cont_ /; StringQ[ cont] && StringMatchQ[ cont, "*Not Found*"], ___], ___}, ___]],
       (NotebookClose[ webFile];
        NotebookOpen @ layoutFile),
        SetOptions[ webFile, Visible -> Inherited]],
      NotebookOpen @ layoutFile];
    SetOptions[ $FrontEnd, First @ msgOpts]
    ]
***)

DemonstrationTemplateOpen::nofile = "File not found.";

DemonstrationTemplateOpen[ url_String] := Module[{nb},
   Quiet[ 
     nb = Import[ url <> "?mathid=" <> $MachineID <> "&license=" <> $LicenseID, "Text"]; (* Using text import to bypass dynamic warnings *)
     If[ nb === $Failed, 
       nb = Import[ ToFileName[{ $InstallationDirectory, "AddOns", "Applications", 
                      "DemonstrationsTools", "FrontEnd", "TextResources"}, "DemonstrationsTemplate.nb"], "Text"]]
     ];
     If[ nb =!= $Failed, 
       NotebookPut @ ReleaseHold @ ToExpression[ nb, StandardForm, Hold],
       DemonstrationsTools`MsgToConsole[ DemonstrationTemplateOpen::nofile]]
   ]

(*** Alternate version which avoids use of Import[]:

DemonstrationTemplateOpen[ url_String] := Module[{msgOpts, webTemplate, layoutTemplate, webNbExpr},
    msgOpts = Options[ $FrontEnd, MessageOptions];
    SetOptions[ $FrontEnd, MessageOptions -> {"ErrorAction" -> {}}];
    webTemplate = NotebookOpen[ url, Visible -> False];
    layoutTemplate =  ToFileName[{ $InstallationDirectory, "AddOns", "Applications", "DemonstrationsTools", "FrontEnd", "TextResources"}, "DemonstrationsTemplate.nb"];
    webNbExpr = NotebookGet @ webTemplate;
    If[ webTemplate =!= $Failed,
      If[ MatchQ[ webNbExpr, Notebook[{ Cell[ cont_ /; StringQ[ cont] && StringMatchQ[ cont, "*Not Found*"], ___], ___}, ___]],
       (NotebookClose[ webTemplate];
        NotebookPut @ Get @ layoutTemplate),
       (CreateDocument[ webNbExpr, Visible -> Inherited];
        NotebookClose[ webTemplate])],
      NotebookPut @ Get @ layoutTemplate];
    SetOptions[ $FrontEnd, First @ msgOpts]
    ]
***)

DemonstrationTestMask[] := 
    NotebookPut @ Notebook[{ 
      Cell[
        BoxData[ ToBoxes[
          Tooltip[
            ArrayPlot[ PadRight[{{0}}, {4, 5}, 0], 
              PixelConstrained -> True, 
              ImageSize -> {500, 400}, 
              Epilog -> Inset[ Cell[ TextData[{
                "\[LongLeftArrow] \[CenterDot] \[CenterDot] \[CenterDot]    The entire ",
                StyleBox["Manipulate", "MR", FontSize -> Inherited 1.2],
                " output, including controls,\t\t\t\n\t\t\tshould extend past this white area \
into the green area    \[CenterDot] \[CenterDot] \[CenterDot] \[LongRightArrow]"}],
                 FontFamily -> "Verdana"], {250, 350}]], 
            "Place this window over any Manipulate output, thumbnail, or snapshot and confirm that the \
Manipulate output being tested extends into, but does not exceed, the green area.", 
            TooltipDelay -> 0.4`]
          ]], "Output", 
        TextAlignment -> "Center", 
        CellMargins -> {{0, 0}, {Inherited, 65}}, 
        ShowCellBracket -> False],
      Cell[ BoxData[
        ButtonBox["Close", 
          ButtonFunction :> NotebookClose[ButtonNotebook[]], 
          Background -> Black, 
          BaseStyle -> { FontColor -> White, FontFamily -> "Verdana", FontWeight -> "Bold"}]], 
        ShowCellBracket -> False, 
        TextAlignment -> "Center"]}, 
      Deployed -> True, 
      Saveable -> False,
      Background -> RGBColor[0.8, 1, 0], 
      WindowTitle -> "Size Test", 
      WindowSize -> {650, 550}, WindowOpacity -> .5, 
      WindowFloating -> True, WindowFrame -> "Palette", 
      WindowElements -> {}, 
      WindowFrameElements -> {"CloseBox"}]
      
      

(************** ManipulatePalette code from Lou ***********************************************************)



(* ::Subsection::Closed:: *)
(*Bookmark Manager Lite, V3*)


(* ::Text:: *)
(* Lou D'Andria *)
(* May 11, 2006 *)
(**)
(*Simple-minded tools for managing Manipulate snapshots and bookmarks for the Demonstrations site.*)


(* ::Subsubsection::Closed:: *)
(*Basic workflow*)


(*
Someone will have a bunch of code in a notebook, some of which will be inside a
Manipulate. They'll evaluate the code and get the Manipulate output.

Whenever they find a place that looks interesting to them, they'll use copy and
paste to make a copy of the Manipualte output cell, and put it in one of several
possible locations (thumbnail, snapshots, bookmarks, etc)

If they then update their code, they can use the update mechanism defined in
this code to update all such Manipulate outputs based on the new code.

When the user wants to update their Manipulate outputs with new ones based on
the latest code, they'll click "Update Copies" on the palette.

Stephen was keen on having this Update actually paste an additional output
instead of overwriting the old one, but I'm not convinced that will be the most
common case, so "Update Copies" actually replaces the old images, while "Update
Copies Keeping Originals" pastes a new image immediately beneath each existing
one.

Both these buttons work by reading the current Manipulate input from the
notebook, and then walking the notebook, updating one output at a time. The
values for the parameters are taken to be the last values chosen in that output.
*)


(* ::Subsubsection::Closed:: *)
(*Issues*)


(*
Error checking could be a lot better.

I do not expect this to work with nested Manipulate expressions.
*)


(* ::Subsubsection::Closed:: *)
(*mergeManipulates[]*)


(* Load Manipulate.mx so we can use utilities from Manipulate`Dump` *)
Get[ToFileName[{$InstallationDirectory, "SystemFiles", "Kernel", 
   "SystemResources", $SystemID}, "Manipulate.mx"]] 


iOptQ = Manipulate`Dump`manipulateOptionQ;
iParamQ = Manipulate`Dump`validParameterOrOtherArgument;


ClearAll[mergeManipulates]


mergeManipulates[
	HoldPattern[Manipulate][m1_, params1___?iParamQ, opts11___?iOptQ, Initialization :> init1_, opts12___?iOptQ],
	HoldPattern[Manipulate][_, params2___?iParamQ, opts21___?iOptQ, Initialization :> _, opts22___?iOptQ]] := 
Manipulate[m1, Evaluate[mergeparams[{params1}, {params2}]], Initialization :> init1, opts11, opts12]


mergeManipulates[
	HoldPattern[Manipulate][m1_, params1___?iParamQ, opts11___?iOptQ, Initialization :> init1_, opts12___?iOptQ],
	HoldPattern[Manipulate][_, params2___?iParamQ, opts21___?iOptQ, Initialization :> _, opts22___?iOptQ]] := 
ReleaseHold[Manipulate @@@ 
	Thread[{Hold[m1], mergeparams[Hold /@ Unevaluated[{params1}], Hold /@ Unevaluated[{params2}]], Initialization :> init1, opts11, opts12}, Hold]]


mergeManipulates[
	m:HoldPattern[Manipulate][___, Initialization :> _, ___], 
	HoldPattern[Manipulate][args___]] := 
mergeManipulates[m, Manipulate[args, Initialization :> {}]]


mergeManipulates[
	HoldPattern[Manipulate][args___], 
	m:HoldPattern[Manipulate][___, Initialization :> _, ___]] := 
mergeManipulates[Manipulate[args, Initialization :> {}], m]


mergeManipulates[
	HoldPattern[Manipulate][args___], m:HoldPattern[Manipulate][args2___]] := 
mergeManipulates[Manipulate[args, Initialization :> {}], Manipulate[args2, Initialization :> {}]]


mergeparams[params1_, params2_] := Sequence @@ Map[mergeparam[#, If[MatchQ[#, Hold[_ -> {___}]], Extract[#, {1,2}, Hold], #]& /@ params2]&, params1]


(* Always use the new controller binding, if there is one: *)
mergeparam[Hold[x_String -> spec_], params2_] := Thread[Hold[x] -> mergeparam[Hold[spec], params2], Hold]


(* Match parameters which have the same variable name, the same domain spec, and the same options: (disabled) *)
(*
mergeparam[Hold @ {{var_, init_}, spec___}, {___, Hold @ {{var_, init2_}, spec___}, ___}] := Hold @ {{var, init2}, spec}

mergeparam[Hold @ {{var_, init_, lab_}, spec___}, {___, Hold @ {{var_, init2_, ___}, spec___}, ___}] := Hold @ {{var, init2, lab}, spec}

mergeparam[Hold @ {{var_, init_, lab_, reset_}, spec___}, {___, Hold @ {{var_, init2_, ___}, spec___}, ___}] := Hold @ {{var, init2, lab, reset}, spec}

mergeparam[Hold @ {var_Symbol, spec___}, {___, Hold @ {{var_, init2_, ___}, spec___}, ___}] := Hold @ {{var, init2}, spec}
*)


(* Match parameters which have the same variable name and the same domain spec: (disabled) *)
(*
mergeparam[Hold @ {{var_, init_}, spec___, newopts___Rule}, {___, Hold @ {{var_, init2_}, spec___, ___}, ___}] := Hold @ {{var, init2}, spec, newopts}

mergeparam[Hold @ {{var_, init_, lab_}, spec___, newopts___Rule}, {___, Hold @ {{var_, init2_, ___}, spec___, ___}, ___}] := Hold @ {{var, init2, lab}, spec, newopts}

mergeparam[Hold @ {{var_, init_, lab_, reset_}, spec___, newopts___Rule}, {___, Hold @ {{var_, init2_, ___}, spec___, ___}, ___}] := Hold @ {{var, init2, lab, reset}, spec, newopts}

mergeparam[Hold @ {var_Symbol, spec___, newopts___Rule}, {___, Hold @ {{var_, init2_, ___}, spec___, ___}, ___}] := Hold @ {{var, init2}, spec, newopts}
*)


(* Match parameters which have the same variable name: *)

mergeparam[Hold @ {{var_, init_}, spec___}, {___, Hold @ {{var_, init2_}, ___}, ___}] := Hold @ {{var, init2}, spec}

mergeparam[Hold @ {{var_, init_, lab_}, spec___}, {___, Hold @ {{var_, init2_, ___}, ___}, ___}] := Hold @ {{var, init2, lab}, spec}

mergeparam[Hold @ {{var_, init_, lab_, reset_}, spec___}, {___, Hold @ {{var_, init2_, ___}, ___}, ___}] := Hold @ {{var, init2, lab, reset}, spec}

mergeparam[Hold @ {var_Symbol, spec___}, {___, Hold @ {{var_, init2_, ___}, ___}, ___}] := Hold @ {{var, init2}, spec}


(* fall-through case *)
mergeparam[param_, params2_] := param 


(* ::Subsubsection::Closed:: *)
(*GetManipulateFromSelection[]*)


GetManipulateFromSelection[nbobj_] := 
Block[{cg = Cases[ NotebookRead[nbobj], Cell[_, "Input" | "Code", ___], Infinity], c},   (* Using Cases to address 80050 *)
	If[ ListQ @ cg, c = First @ cg, c = cg];
	c = MakeExpression[StripBoxes[First[c]]];
	If[FreeQ[c, Manipulate], Return[None]];
	First[Cases[c, m_Manipulate, Infinity]] ]


(* ::Subsubsection::Closed:: *)
(*GetManipulateFromNotebook[]*)


(* ::Text:: *)
(*The Manipulate input should be the first input after the "ManipulateSection" style cell.*)


GetManipulateFromNotebook[nbobj_] := 
Block[{},
	SelectionMove[nbobj, Before, Notebook];
	If[ $Failed =!= NotebookFind[nbobj, "ManipulateSection", Next, CellStyle],  (* Adding some error handling here *)
      SelectionMove[nbobj, All, CellGroup],
      {"No Manipulate Section"}];
	GetManipulateFromSelection[nbobj]]
	(* Tried the following to resolve 80050, but not robust enough -- gets misdirected if there Input cells elsewhere in notebook:
	If[ $Failed === NotebookFind[nbobj, "Input", Next, CellStyle], 
      NotebookFind[nbobj, "Code", Next, CellStyle]]; *)


(* ::Subsubsection::Closed:: *)
(*Global variables - obsolete*)


(*
$SnapshotCellTag = "SnapshotUpdateTag";
Manipulate`Dump`$PasteMode = "Demonstrations";
Manipulate`Dump`$PasteTag = $SnapshotCellTag;
Manipulate`Dump`$PasteFunction = PasteSnapshotImage;
*)


(* ::Subsubsection::Closed:: *)
(*ContextsPresent[]*)


ContextsPresent[expr_] := Union[Cases[expr, s_Symbol :> Context[s], Infinity, Heads -> True]]


(* ::Subsubsection::Closed:: *)
(*StripContexts[]*)


StripContexts[expr_] := StripContexts[expr, All]


StripContexts[expr_, All] := ToExpression[Block[{Internal`$ContextMarks = False}, ToString[expr, InputForm]]]


StripContexts[expr_, cxt_String] := expr //. stripContextRules[Cases[{expr}, s_Symbol :> Hold[s] /; Context[s] === cxt, Infinity, Heads -> True]]


StripContexts[expr_, cxts:{___String}] := Fold[StripContexts, expr, cxts]


stripContextRules[lis:{Hold[_Symbol]...}] := stripContextRule /@ Union[lis]


stripContextRule[Hold[sym_Symbol]] := stripContextRule[Hold[sym], ToExpression[SymbolName[Unevaluated[sym]], InputForm, Hold]]


stripContextRule[Hold[sym_Symbol], Hold[newsym_Symbol]] := HoldPattern[sym] :> newsym


(* ::Subsubsection::Closed:: *)
(*ChangeContexts[]*)


ChangeContexts[expr_, cxt_String -> newcxt_String] := expr //. changeContextRules[Cases[{expr}, s_Symbol :> Hold[s] /; Context[s] === cxt, Infinity, Heads -> True], newcxt]


ChangeContexts[expr_, cxts:{Rule[_String, _String]...}] := Fold[ChangeContexts, expr, cxts]


changeContextRules[lis:{Hold[_Symbol]...}, newcxt_String] := changeContextRule[#, newcxt]& /@ Union[lis]


changeContextRule[Hold[sym_Symbol], newcxt_String] := changeContextRule[Hold[sym], ToExpression[newcxt <> SymbolName[Unevaluated[sym]], InputForm, Hold]]


changeContextRule[Hold[sym_Symbol], Hold[newsym_Symbol]] := HoldPattern[sym] :> newsym


(* ::Subsubsection::Closed:: *)
(*FixNotebookContexts[]*)


FixNotebookContexts[expr_] := 
Block[{cxts}, 
	cxts = ContextsPresent[expr];
	cxts = Select[cxts, (StringMatchQ[#, "Notebook$$*`"] || StringMatchQ[#, "$CellContext`"])&];
	ChangeContexts[expr, Thread[cxts -> "Global`"]]]


(* ::Subsubsection::Closed:: *)
(*UpdateManipulateOutputs[]*)


(* ::Text:: *)
(*Find and read in the first Manipulate input.*)
(*Walk the notebok looking for Manipulate output.*)
(*When one is found, call UpdateManipulateOutput.*)


(* ::Text:: *)
(*Note that the CellProlog and CellEpilog lines below need to be kept in sync *manually* with the corresponding setting in the stylesheet. If you change either, make sure you change the other.*)


UpdateManipulateOutputs[nbobj_, overwriteQ_:False] := 
Block[{m, c = 0, dynamicupdating, saved}, 
    dynamicupdating = FE`Evaluate[FEPrivate`DynamicUpdating[nbobj]];
    FE`Evaluate[FEPrivate`SetDynamicUpdating[nbobj, False]];
    MathLink`CallFrontEnd[FrontEnd`NotebookSuspendScreenUpdates[nbobj]];
	
	(* CellProlog, 73436*)
	saved = ReplaceAll[ControllerLinking, Options[Manipulate]];
	SetOptions[Manipulate, ControllerLinking -> True];
	
	m = GetManipulateFromNotebook[nbobj];
	m = FixNotebookContexts[m];
	SelectionMove[nbobj, Before, Notebook];
	SelectionMove[nbobj, Next, Cell]; 
	While[
		Developer`CellInformation[nbobj] =!= $Failed, 
		UpdateManipulateOutput[nbobj, m, overwriteQ];
		SelectionMove[nbobj, Next, Cell]];
	
	(* CellEpilog, 73436*)
	SetOptions[Manipulate, ControllerLinking -> saved];
	
	SelectionMove[nbobj, Before, Notebook];
    MathLink`CallFrontEnd[FrontEnd`NotebookResumeScreenUpdates[nbobj]];
    With[{dynamicupdating = dynamicupdating},
    	FE`Evaluate[FEPrivate`SetDynamicUpdating[nbobj, dynamicupdating]]];
	MessageDialog[TextCell @ Row[{
		"Update Complete: ",
		Switch[c, 0, "no copies", 1, "one copy", _, ToString[c] <> " copies"], " updated."}]];
	c]


(* ::Subsubsection::Closed:: *)
(*UpdateManipulateOutput[]*)


UpdateManipulateOutput[nbobj_, m:((h:Manipulate)[___, _[Initialization, _], ___]), overwriteQ_:False] := 
Block[{cell, vars, m2}, 
	(* do nothing if the cell is not an output cell of the right structure *)
	If[!MatchQ[Developer`CellInformation[nbobj], {{___, "Style" -> "Output", ___}}], Return[Null]];

	cell = NotebookRead[nbobj];
	If[FreeQ[cell, Manipulate`InterpretManipulate], Return[Null]];

	m2 = Cases[MakeExpression[StripBoxes[First[cell]]], _Manipulate, Infinity];
	m2 = FixNotebookContexts[m2];
	If[m2 === {}, Return[Null]];

	m2 = First[m2];
	m2 = mergeManipulates[m, m2];
	If[Not @ TrueQ @ overwriteQ, SelectionMove[nbobj, After, Cell]];

	m2 = FixNotebookContexts[m2];
	m2 = ChangeContexts[m2, "Global`" -> ""];
	m2 = ToBoxes[m2];
	m2 = FixNotebookContexts[m2];
	m2 = ChangeContexts[m2, "Global`" -> "$CellContext`"];
	m2 = Cell[BoxData[m2], "Output"];
	NotebookWrite[nbobj, m2];
	++c
]


UpdateManipulateOutput[nb_, HoldPattern[Manipulate][body_, noinit___], overwriteQ_:False] := 
	UpdateManipulateOutput[nb, Manipulate[body, noinit, Initialization :> {}], overwriteQ]


(* ::Subsubsection::Closed:: *)
(*ManipulatePalette[]*)


SetAttributes[palettebutton, HoldRest]


palettebutton[label_, func_] := Button[label, FrontEnd`MessagesToConsole[func], Appearance ->"Palette", Method -> "Queued"]


ManipulatePalette[] := 
PalettePut[Column[{
	palettebutton["Update Copies", UpdateManipulateOutputs[InputNotebook[], True]], 
	palettebutton["Update Copies Keeping Originals", UpdateManipulateOutputs[InputNotebook[], False]]
}, RowSpacings -> 0]];


(* ::Subsubsection::Closed:: *)
(*Usage example*)


(* ::Input:: *)
(*ManipulatePalette[]*)


(* ::Subsection::Closed:: *)
(*Package footer*)





End[]

EndPackage[]


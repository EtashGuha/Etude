(* Mathematica Package *)

BeginPackage["NotebookTemplating`Authoring`",{"NotebookTemplating`"}]

evaluationTooltipLabels::usage = "evaluationTooltipLabels  "
(* Exported symbols added here with SymbolName::usage *)  


Begin["`Private`"] (* Begin Private Context *) 

Needs["NotebookTemplating`Utilities`"]

(*	Resources	*)

$NotebookTemplateVersion := 1.0

text[id_]:= Dynamic[RawBoxes[FEPrivate`FrontEndResource["NotebookTemplatingStrings", id]], BaseStyle -> "TextStyling", SingleEvaluation -> True]

tr[id_String, opts__] :=
    Style[text[id], "Text", opts, FontSize -> $DefaultFontSize, FontColor -> $DefaultFontColor, FontWeight -> $DefaultFontWeight]

tr[id_String] := tr[id, TextAlignment -> Center]

tr[id_String, "tooltip"] := Style[text[id], "Text", FontSize -> "Smaller"]

tr[id_String, "title", opts___]:= TextCell[Style[text[id], "Text", FontSize -> $DefaultFontSize], opts, StripOnInput -> True]

tr[id_String, "popupitem", opts___] := Style[text[id], "Text", opts, TextAlignment -> Left]

tr[id_?(Head[#] =!= String &), opts___] := tr[ToString[id], opts]


imageResource[id_] := Dynamic[RawBoxes[FEPrivate`FrontEndResource["NotebookTemplatingBitmaps", id]], SingleEvaluation -> True]

expressionResource[id_] := FEPrivate`FrontEndResource["NotebookTemplatingExpressions", id]
 
(* TODO: need to check this for AuthoringDockedCell in .tr, mostly for Cloud *)
If[ 
TrueQ[CloudSystem`$CloudNotebooks],
	
$ButtonDefaultAppearance := (Appearance -> expressionResource["ButtonDefaultAppearanceCloud"]);
$ButtonDropdownBothAppearance := (Appearance -> expressionResource["ButtonDropdownBothAppearanceCloud"]);
$ButtonDropdownLeftAppearance := (Appearance -> expressionResource["ButtonDropdownLeftAppearanceCloud"]);
$ButtonDropdownRightAppearance := (Appearance -> expressionResource["ButtonDropdownRightAppearanceCloud"]);
(*$ButtonOKAppearance := (Appearance -> expressionResource["ButtonOKAppearanceCloud"])*)
    
,
(* otherwise, use the desktop versions with different states*)    
$ButtonDefaultAppearance := (Appearance -> expressionResource["ButtonDefaultAppearance"]);
$ButtonDropdownBothAppearance := (Appearance -> expressionResource["ButtonDropdownBothAppearance"]);
$ButtonDropdownLeftAppearance := (Appearance -> expressionResource["ButtonDropdownLeftAppearance"]);
$ButtonDropdownRightAppearance := (Appearance -> expressionResource["ButtonDropdownRightAppearance"]);
(*$ButtonOKAppearance := (Appearance -> expressionResource["ButtonOKAppearance"])*)

] 

$DockedCellAppearance := (Appearance -> expressionResource["DockedCellBackground"]);
$DividerAppearance := (Appearance -> expressionResource["RepeatingBlockDivider"]);
(* unused? *)
$SlotDialogInputFieldAppearance := (Appearance -> expressionResource["SlotDialogInputField"]);
(* unused? *)
$ExpressionDialogInputFieldAppearance := (Appearance -> expressionResource["ExpressionDialogInputField"]);

iconImage[id_] := imageResource[StringJoin[{id, "Icon"}]];
evalImage[id_] := imageResource[StringJoin[{id, "Icon"}]];

evaluationIcons := {evalImage["evalDelete"], evalImage["evalHide"], 
  evalImage["uneval"], evalImage["exclude"]}

evaluationIconsMenu := {evalImage["evalDeleteMenu"], 
  evalImage["evalHideMenu"], evalImage["unevalMenu"], 
  evalImage["excludeMenu"], evalImage["removeMenu"]}

evaluationTooltipLabels := 
 Map[tr[#, "tooltip"] &, evaluationTooltipResource]

reportInputMenu := 
 Map[tr[#, "popupitem", Background -> None] &, reportInputMenuResource]


(*Utilities*)

$DefaultFontSize = 11;
$DefaultFontColor = RGBColor @@ Table[N[117/255], {3}];
$DefaultFontWeight = Bold;

$DefaultButtonSize = {Automatic, 30};
$HelpButtonFontSize = $DefaultFontSize - 1;
$EnabledFontColor = $DefaultFontColor;

$CancelFrameOpts = Sequence @@ {FrameStyle -> None, FrameMargins -> {{5, 5}, {Automatic, Automatic}}};
(*$OKFrameOpts = Sequence @@ {FrameStyle -> None, FrameMargins -> {If[CloudSystem`$CloudNotebooks === True, {5, 5}, {10, 10}], {Automatic, Automatic}}};*)
$OKFrameOpts = Sequence @@ {FrameStyle -> None, FrameMargins -> {{10, 10}, {Automatic, Automatic}}};
$HelpFrameOpts =  Sequence @@ {FrameStyle -> None, FrameMargins -> {{5, 5}, {Automatic, Automatic}}};
$DialogTextOpts = Sequence @@ {FontColor->RGBColor @@ Table[N[89/255], {3}], FontWeight -> Plain};

$DockDefaultLeftMarginWidth = 5;
$DockDefaultRightMarginWidth = 5;
$DockedCellPaneSettings = Sequence @@ {ImageMargins -> {{$DockDefaultLeftMarginWidth, $DockDefaultRightMarginWidth}, {Automatic, Automatic}}};
$DockedCellPanelSettings = Sequence @@ {FrameMargins -> {{8, 8}, {8, 12}}, ImageMargins -> -1, $DockedCellAppearance} ;
$DockedDivider = RGBColor @@ Table[N[183/255], {3}];

$BoxDialogTextOpts = Sequence @@ {FontColor -> RGBColor @@ N[{35, 89, 109}/255], FontSize -> 11};
$BoxDialogInputFieldHintStyle = {FontColor->RGBColor @@ Table[N[69/255], {3}]};

(* unused
$SlotDialogBackground =RGBColor@@N[{127,196,221}/255];
$SlotDialogTitle = Sequence @@ {FontWeight->Bold, FontColor->RGBColor@@N[{43,101,122}/255],FontSize->$DefaultFontSize};
$SlotDialogFrameStyle = RGBColor@@N[{80, 150,175}/255];
$SlotDialogInputFieldStyle = {FrameMargins -> {{3, 3}, {Automatic, Automatic}}, FieldSize -> {22, 1.2}, Appearance->None, BaseStyle -> {"ControlStyle"}};
*)

$RepeatingBlockFieldHintStyle = {FontColor->RGBColor @@ Table[N[114/255], {3}]};
$RepeatingBlockFieldSize = {52, 1.3};
(*$RepeatingBlockFieldSize = If[ CloudSystem`$CloudNotebooks === True, {30,1.3}, {52, 1.3}];*)
$RBInputFieldStyle = {FieldSize -> $RepeatingBlockFieldSize};
$RBWindowSize := Dynamic[FEPrivate`If[ FEPrivate`SameQ[FEPrivate`$OperatingSystem, "MacOSX"], {526, 348}, {620, 350}]];
$RBFramedTextOpts = Sequence @@ {FrameStyle -> None, FrameMargins -> {{0, 0}, {0, 1}}};
$RBcellFrameLabelsBackground = RGBColor@@N[{208,241,255}/255];
$RBcellFrameLabelsFontColor = RGBColor@@N[{75,129,152}/255];
$RBcellFrameLabelsFrameColor = RGBColor@@N[{126,195,224}/255];
$RBcellFrameLabelsTitColor = RGBColor@@N[{43,92,112}/255];

$ExpressionExpandedBackground = RGBColor[0.9372549019607843, 0.9803921568627451, 1.];
$ExpressionExpandedTitle = Sequence @@ {FontWeight->Bold, FontColor->RGBColor@@N[{76,124,142}/255]};
$ExpressionExpandedFrameStyle = RGBColor@@N[{126,195,221}/255];

(* Code font in expression box (cloud only) *)
$ExpressionInputBoxBaseStyle = {FontFamily -> "Source Sans Pro", Bold, 14};

(*DividerImage for RepeatingBlock*)
SetAttributes[dividerImage, HoldAll]

dividerImage[w_, h_] := Panel["", ImageSize -> {w, h}, $DividerAppearance]

dividerImage[w_] := 
 Panel[Style[Graphics[{}, ImageSize -> {Scaled[1], Automatic}], 
   CacheGraphics -> False], ImageSize -> {w, 2}, $DockedCellAppearance]

dividerImage[] := dividerImage[Full, 2]

(*createButton*)

SetAttributes[createButton, HoldRest]

createButton[label_, action_, opts___?OptionQ] := DynamicModule[{}, 
	Button[label, action, opts, Alignment -> Center],
	Initialization :> (Needs["NotebookTemplating`"]; 
        Needs["NotebookTemplating`Authoring`"];)
]

createButton[label_, action_, opts___?OptionQ, "default"] := 
 createButton[label, action, opts, $ButtonDefaultAppearance]

createButton[label_, action_, opts___?OptionQ, "TemplateSlotLeft"] := 
 createButton[label, action, opts, $ButtonDropdownLeftAppearance]

createButton[label_, action_, opts___?OptionQ, "TemplateSlotRight"] :=
  createButton[label, action, opts, $ButtonDropdownRightAppearance]

createButton[label_, action_, opts___?OptionQ, "EvaluatableCells"] := 
 createButton[label, action, opts, $ButtonDropdownBothAppearance]

(* createDefaultButton *)

SetAttributes[createDefaultButton, HoldRest]
(* unused *)
createDefaultButton[label_, action_, opts___?OptionQ] := DynamicModule[{},
	DefaultButton[label, action, opts, Alignment -> Center],
	Initialization :> (Needs["NotebookTemplating`"];
	    Needs["NotebookTemplating`Authoring`"];)
]

createDefaultButton[label_, action_, "RepeatingBlockDialog", opts___?OptionQ] := With[{
	appr = If[TrueQ[CloudSystem`$CloudNotebooks],
        expressionResource["ButtonOKAppearanceCloud"],
        System`FrontEndResource["NotebookTemplatingExpressions", "ButtonOKAppearance"]
    ]},
    (* Typesetting for DefaultButton contains magic that merges Appearance values ("ButtonType" -> "Default"
     * + user-supplied directives) if the values look valid. FEPrivate` values don't look valid,
     * so Appearance application based on client resources fails in cloud. Use a Button with
     * "ButtonType" -> "Default" instead.
     *)
    createButton[label, action, opts, Appearance -> appr]
]

(*buttonLabel*)
buttonLabel[icon_, label_, "side-by-side", layoutOpts___?OptionQ] := 
 Pane[Grid[{{icon, label}}, layoutOpts, Alignment -> {Automatic, Center},  Spacings -> {.5, Automatic}]]
  
framedCancel[label_, opts___] := Framed[label, opts, $CancelFrameOpts]
framedOK[label_, opts___] := Framed[label, opts, $OKFrameOpts]
framedHelp[label_, opts___] := Framed[label, opts, $HelpFrameOpts]
framedRBText[label_, opts___] := Framed[label, $RBFramedTextOpts]

createTemplate[] := 
 Module[{nb}, 
  ProcessWithFrontEnd[
   nb = CreateDocument[{}, CellContext -> Notebook, 
     ShowCellTags -> True, 
     TaggingRules -> {"NotebookTemplateVersion" -> $NotebookTemplateVersion, "NotebookTemplate" -> True}, 
     DockedCells -> FEPrivate`FrontEndResource["NotebookTemplatingExpressions", "AuthoringDockedCell"]];
   SelectionMove[nb, After, Notebook];
   (*Return Notebook Object*)nb]]


createHeader[nbObj_NotebookObject: InputNotebook[]] := Module[{dos, styles, tags},
 ProcessWithFrontEnd[
 dos = CurrentValue[nbObj,DockedCells];	
 If[dos==={},
  SetOptions[nbObj, CellContext -> Notebook, 
   TaggingRules -> {"NotebookTemplateVersion" -> $NotebookTemplateVersion, "NotebookTemplate" -> True}, 
   DockedCells -> FEPrivate`FrontEndResource["NotebookTemplatingExpressions", "AuthoringDockedCell"]];
  nbObj,
  If[!MatchQ[dos,{}],dos = {dos}]; 
  styles = Map[Select[Rest[#], StringQ][[1]]&,dos];
  tags =Map[CellTags /. Cases[#, _Rule]&,dos];
  handlingDockedCell[nbObj,dos, styles, tags]]
 ]
]

handlingDockedCell[nbObj_,dos_, styles_, tags_]:=
	Module[{title, dockNew},
		title = AbsoluteCurrentValue[nbObj, WindowTitle];
		Which[
			!FreeQ[styles,"NotebookTemplateDockedCell"],
			AuthoringMessageDialog[title <> ".nb is already a Template notebook."]; Return[],
			!FreeQ[tags,Alternatives["StaticMUnitToolbar","ConvertToTestingNotebook"]],
			(*TODO: Need to be updated *)
			AuthoringMessageDialog["You can not convert a Testing notebook to Template notebook."]; Return[],
			True,
			dockNew = Join[{FEPrivate`FrontEndResource["NotebookTemplatingExpressions", "AuthoringDockedCell"]}, dos];
			CurrentValue[nbObj, DockedCells->dockNew]
		]
		
	]

CreateTemplateNotebook[] := createTemplate[]
CreateTemplateNotebook[nb_NotebookObject] := createHeader[nb]
CreateTemplateNotebook[nb_Notebook] := 
CreateTemplateNotebook[CreateDocument[nb]]

ClearTemplateNotebook[
  nbObj_NotebookObject /; TemplateNotebookQ[nbObj]] := 
 Module[{tagRules, newTagRules}, 
  tagRules = TaggingRules /. Options[nbObj, TaggingRules];
  newTagRules = 
   DeleteCases[
    tagRules, ("NotebookTemplateVersion" | "NotebookTemplate") -> _];
  If[tagRules =!= newTagRules, 
   SetOptions[nbObj, TaggingRules -> newTagRules]];
  Scan[(CurrentValue[#, {CellFrameLabels, 1, 1}] = None; 
   CurrentValue[#, {CellFrameLabels, 2, 2}] = None; 
   CurrentValue[#, {CellBracketOptions, "Color"}] = 
    CurrentValue[nbObj, {CellBracketOptions, "Color"}]) &, Cells[nbObj]]; 
  SetOptions[nbObj, DockedCells -> {}]]

dialogText[text_String, opts___?OptionQ] := Text[Style[text, "Text", opts, $BoxDialogTextOpts]]

makeTemplateBox[box_, default_, mode_, opts___?OptionQ] := 
 TemplateBox[{box, default, mode, ToBoxes[box]}, "TemplateVariable", 
  opts]

checkString[x_String] := 
 Module[{val = x}, 
  If[StringTake[val, 1] =!= "\"", val = "\"" <> val];
  If[StringTake[val, -1] =!= "\"", val = val <> "\""];
  val]

checkString[x_] := x

uncheckString[x_String] := 
 Module[{val = x}, 
  If[StringTake[val, 1] === "\"", val = StringTrim[val, "\""]];
  If[StringTake[val, -1] === "\"", val = StringTrim[val, "\""]];
  val]

uncheckString[x_] := x

boxWrite[obj_, Sequence[], _, opts___?OptionQ] := 
 NotebookWrite[obj, Sequence[]]

boxWrite[obj_, box_, TextData, opts___?OptionQ] := 
 NotebookWrite[obj, 
  Cell[BoxData[FormBox[box, TextForm]], opts]]


(*
  NotebookWrite in the Desktop seems to strip out an extra layer of Cell[BoxData[]], which 
  is not the case in the Cloud.  Sometime we should understand this and either rewrite this
  or maybe make the Cloud NotebookWrite work in the same way.
*)
boxWrite[obj_, box_, _, opts___?OptionQ] := 
    If[ CloudSystem`$CloudNotebooks === True,
        NotebookWrite[obj, 
            Cell[BoxData[FormBox[box, TextForm]], opts]],
        NotebookWrite[obj, 
            Cell[BoxData[
                Cell[BoxData[FormBox[box, TextForm]], opts]]]]
    ]

returnFun[mode_String, contentData_] := 
 Module[{eval, cellData, value, default, box}, eval = EvaluationCell[];
  cellData = NotebookRead[eval];
  {value, default} = 
   Cases[cellData, InputFieldBox[dd_, ___] -> dd, Infinity];
  If[mode === "Named", value = checkString[value]];
  If[default === Null, default = value];
  box = createContractedBox[value, default, mode, contentData];
  boxWrite[eval, box, contentData]]


returnCancelFun[value_, default_, mode_String, contentData_] := 
 Module[{eval, box}, eval = EvaluationCell[];
  box = createContractedBox[value, default, mode, contentData];
  boxWrite[eval, box, contentData]]

(*Return whether the current selection is inside a BoxData or TextData.It needs to know this for making the TemplateSlot UI.
In the Desktop this is typically going to be a BoxData,but in the Cloud it will typically be TextData.TODO,rethink this.
It would be better not to need this.*)

getContentData[nbObj_] := Module[{ef},
	ef = cellInformation[nbObj, "ContentData"];
	If[ ef === TextData, TextData, BoxData]
]

makeTemplateSlot[nbObj_NotebookObject, mode_String] := 
 Module[{selection}, selection = NotebookRead[nbObj];
  makeTemplateSlot[nbObj, selection, mode]]

makeTemplateSlot[nbObj_NotebookObject, selection_String, 
  mode_String] := 
 Module[{variable, contentData, preview, box}, 
  variable = preprocessVar[selection, mode];
  contentData = getContentData[nbObj];
  preview = checkString[selection];
  box = createContractedBox[variable, preview, mode, contentData];
  boxWrite[nbObj, box, contentData]]

makeTemplateSlot[nbObj_NotebookObject, _, mode_String] := 
 Module[{contentData, box}, contentData = getContentData[nbObj];
  box = createExpandedBox[Null, Null, mode, contentData];
  boxWrite[nbObj, box, contentData]]

rewriteFun[a1_, a2_, mode_, contentData_] := 
 Module[{box}, box = createExpandedBox[a1, a2, mode, contentData];
  boxWrite[EvaluationCell[], box, contentData]]
  
  
templateSlotUIInputField[variable_,mode_] :=
    Module[{},
        If[ mode === "Named",
            InputFieldBox[variable, String, FieldHint -> "name"],
            InputFieldBox[variable, Number]
        ]
    ]
  
templateSlotUIValueLabel[mode_] :=
    If[mode === "Named", dialogText["Name:"], dialogText["    Position:"]]        
 
 
createExpandedBox[variable_, previewOld_, mode_, contentData_] := 
 Module[{inputField, defaultField,preview}, 
  inputField = 
   templateSlotUIInputField[variable,mode];
  preview = 
     ReplaceAll[previewOld, 
      Pattern[$CellContext`f, Blank[]][
        Pattern[$CellContext`x, Blank[]], BlankNullSequence[]] :> 
       Condition[$CellContext`x, 
        SymbolName[Unevaluated[$CellContext`f]] == 
         "TemplateArgBox"]]; 
  defaultField = InputFieldBox[preview, Boxes, FieldSize -> {22, 2.4}];
  TemplateBox[{variable, previewOld, mode, contentData, inputField, defaultField}, "NotebookTemplateSlotUI"]]

createContractedBox[variableIn_, preview_, mode_, contentData_] :=
    Module[ {variable = variableIn},
        If[ variable === Null, variable = "\"\""];
        TemplateBox[{variable, preview, mode, contentData},  "NotebookTemplateSlot" ]    
    ]  


(*Turn this switch off to restore old Expression mechanism*)
FF`NewEval = True

makeEvaluationExpression[nbObj_NotebookObject] := 
 Module[{selection}, 
  If[! FF`NewEval, makeEvaluationMarker[nbObj], 
   selection = NotebookRead[nbObj];
   makeEvaluationExpression[nbObj, selection]]]


makeEvaluationExpression[nbObj_NotebookObject, {}] := Module[
	{contentData, box},
	contentData = getContentData[nbObj];
	If[TrueQ[CloudSystem`$CloudNotebooks],
	    box = createExpressionBox[Null, contentData];
	    boxWrite[nbObj, box, contentData],
	    (* else *)
	    box = createContractedExpressionBox["", contentData];
	    boxWrite[nbObj, box, contentData];
	    FrontEndTokenExecute[nbObj, "MovePrevious"];
	    FrontEndTokenExecute[nbObj, "MovePrevious"];
	    FrontEndTokenExecute[nbObj, "MovePrevious"]
	]
]

makeEvaluationExpression[nbObj_NotebookObject, selection_] := 
 Module[{contentData, box}, contentData = getContentData[nbObj];
  box = createContractedExpressionBox[selection, contentData];
  boxWrite[nbObj, box, contentData]]

createExpressionBox[expr_, contentData_] :=
    Module[ {inputField,inputExpr},
        inputField = InputFieldBox[
        	If[StringQ[expr]||expr===Null (*So that nothing is displayed in the InputField*),
        		expr,
        		inputExpr=If[Head[expr]===Cell (*If you grab the whole Cell, BoxData is not required*),expr,BoxData[expr]];
        		UsingFrontEnd[MathLink`CallFrontEnd[FrontEnd`ExportPacket[inputExpr,"InputText"]]][[1]]
        	], Boxes
        	, FieldHint -> "DateString[]"
        	, BaseStyle -> $ExpressionInputBoxBaseStyle
            , FieldSize -> {22, 2.4}
        ];
        TemplateBox[{expr, contentData, inputField}, "NotebookTemplateExpressionUI"]
    ]
    
contractExpressionFun[contentData_] := 
 Module[{eval, cellData, expr, box}, eval = EvaluationCell[];
  cellData = NotebookRead[eval];
  {expr} = Cases[cellData, InputFieldBox[dd_, ___] -> dd, Infinity];
  box = createContractedExpressionBox[expr, contentData];
  boxWrite[eval, box, contentData]]
(*
 Come here from the Cancel button.  At one time if the 
 expr was Null, it would delete the cell.  But this is 
 hard to do in the Cloud and I'm not sure if it's the 
 right thing anyway.  Eg if you delete the contents and 
 hit OK, this doesn't delete.
*)
contractExpressionFun[exprIn_, contentData_] := 
    Module[{eval, box, expr = exprIn}, 
        If[ SymbolName[Head[expr]] === "TemplateArgBox" && Length[expr] > 0,
             expr = First[expr]];
    	eval = EvaluationCell[];
        box = createContractedExpressionBox[expr, contentData];
        boxWrite[eval, box, contentData]
    ]


rewriteExpressionFun[exprIn_, contentData_] := 
    Module[{box, expr = exprIn}, 
        If[ SymbolName[Head[expr]] === "TemplateArgBox" && Length[expr] > 0,
 		     expr = First[expr]];
        box = createExpressionBox[expr, contentData];
        boxWrite[EvaluationCell[], box, contentData]
    ]


createContractedExpressionBox[exprIn_, contentData_] := 
 Module[{exprArg = exprIn}, 
  If[StringQ[exprArg] && StringLength[exprArg] === 0 || 
    exprArg === Null, exprArg = " "];
  TemplateBox[{exprArg, "General", contentData}, "NotebookTemplateExpression"]]


defaultTipBox[value_, cutoff_] := Module[
	{str, length, text},
	str = If[MatchQ[value, _String],
		preprocessVar[ToString[value], "Named"], ToString[value]
	];
	length = StringLength[str];
	text = If[length > cutoff,
		StringJoin[{StringTake[str, cutoff], "..."}], str
	];
	Style[
		Column[{
		    Style["Default Value:", Small],
	        text
	    }, Alignment -> Left, Spacings -> Automatic, Frame -> None]
	    , "Text"
	    , FontColor -> Black (*$DefaultFontColor*)
	]
]

    Style[text[id], "Text", opts, FontSize -> $DefaultFontSize, FontColor -> $DefaultFontColor, FontWeight -> $DefaultFontWeight]

cellInformation[nbObj_NotebookObject, property_] := 
 Module[{cellInfo, info, newCellStyle}, 
  cellInfo = Developer`CellInformation[nbObj];
  info = If[MatchQ[cellInfo, $Failed], 
    newCellStyle = 
     OptionValue[Options[$FrontEndSession, DefaultNewCellStyle], 
      DefaultNewCellStyle];
    FrontEndExecute@FrontEnd`FrontEndToken[nbObj, "Style", newCellStyle];
    Developer`CellInformation[nbObj][[1]], cellInfo[[1]]];
  property /. info]

preprocessVar[selection_String, "Named"] := Module[{},
	If[StringMatchQ[selection, "\"" ~~ ___ ~~ "\""],
		selection,
		(* else *)
		StringJoin["\"", ToString[selection], "\""]
	]
]

preprocessVar[{}, "Named"] := Module[{}, ""]

preprocessVar[selection_String, "Positional"] := Module[{}, selection]

preprocessPos[pos_] := Module[{}, pos]

getPreview[var_, TextData] := var;

getPreview[var_, BoxData] := 
  If[StringMatchQ[var, "\"" ~~ ___ ~~ "\""], StringTrim[var, "\""], 
   ToExpression[var]];

makeEvaluationMarker[nbObj_NotebookObject] := 
 Module[{selection, contentData}, selection = NotebookRead[nbObj];
  contentData = getContentData[nbObj];
  (*TODO:selection is not string if in Input cell*)
  makeEvaluationMarker[nbObj, selection, contentData]]

makeEvaluationMarker[nbObj_NotebookObject, {}, contentData_] := 
 Module[{selection, box}, 
  selection = Cells[NotebookSelection[nbObj]];
  box = makeExpLabel["", "General", contentData];
  boxWrite[nbObj, box, contentData];
  If[selection === {}, FrontEndTokenExecute[nbObj, "MovePrevious"]];
  FrontEndExecute[FrontEndToken[nbObj, "MovePrevious"]];
  FrontEndExecute[FrontEndToken[nbObj, "MovePrevious"]]]

makeEvaluationMarker[nbObj_NotebookObject, selection_, contentData_] :=
  Module[{box, exp, expression, exp2, expression2, expNew}, 
  If[MatchQ[selection, {__}], selection = RowBox[selection]];
  If[FreeQ[selection, TemplateBox[{___}, "NotebookTemplateExpression", ___]], 
   box = makeExpLabel[selection, "General", contentData];
   boxWrite[nbObj, box, contentData], 
   If[Length[
      "CursorPosition" /. Developer`CellInformation[nbObj][[1]]] == 2,
     NotebookWrite[nbObj, 
      exp = selection /. 
        TemplateBox[{a_, ___}, "NotebookTemplateExpression", ___] :> a;
      expression = 
       If[MatchQ[exp, RowBox[{__String}]], 
        exp /. RowBox -> StringJoin, exp], All];, 
    SelectionMove[nbObj, All, Cell];
    NotebookWrite[nbObj, 
     exp2 = selection /. 
       TemplateBox[{a_, ___}, "NotebookTemplateExpression", ___] :> a;
     expression2 = 
      If[MatchQ[exp2, RowBox[{__String}]], 
       exp2 /. RowBox -> StringJoin, exp2], All];]];
  While[MatchQ[
    "CursorPosition" /. Developer`CellInformation[nbObj][[1]], _List],
    SelectionMove[nbObj, All, Cell]];
  expNew = NotebookRead[nbObj];
  NotebookWrite[nbObj, expNew]]

evalCellsMenu[i_] :=
    Grid[{{evaluationIconsMenu[[i]], reportInputMenu[[i]]}}, 
    Alignment -> Bottom]

evaluationTooltip[expr_, label_] := 
 Module[{}, ToBoxes[tooltip[expr, label]]]

tagEvaluationCell[index_] := 
 ProcessWithFrontEnd[
  Module[{nbObj, tagFromUI, tagResource}, 
   nbObj = SelectedNotebook[];
   tagFromUI = Part[evaluationTags, index];
   SelectionMove[nbObj, All, Cell];
   tagResource = StringJoin["CellBehaviorDisplayFunction",tagFromUI];
   manageCellBehaviorLabel[nbObj, tagResource];]]

untagEvaluationCell[nbObj_NotebookObject] := 
 ProcessWithFrontEnd[Module[{}, SelectionMove[nbObj, All, Cell];
   removeInputLabel[nbObj];]]


getInputTag[exp_Cell, tag_String] := 
 Module[{tagInfo}, 
  tagInfo = 
   Cases[exp, 
     tagRule_Rule /; MatchQ[tagRule[[1]], CellTags] :> 
      tagRule[[2]]] /. {{x___}} -> {x};
  If[MemberQ[tagInfo, Alternatives @@ evaluationTags], 
   Replace[tagInfo, 
    tagRule_String /; MemberQ[evaluationTags, tagRule] :> tag, {1}], 
   Join[tagInfo, {tag}]]]

manageCellBehaviorLabel[nbObj_NotebookObject,tagResource_] := 
 Scan[(CurrentValue[#, {CellFrameLabels, 1, 1}] = FrontEndResource["NotebookTemplatingExpressions", tagResource]) &, 
  Cells[NotebookSelection[nbObj]]]

removeInputLabel[nbObj_NotebookObject] := 
 Module[{}, 
  Scan[(CurrentValue[#, {CellFrameLabels, 1, 1}] = None) &, 
   Cells[NotebookSelection[nbObj]]]]

removeInputLabel[cell : {__Cell}] := removeInputLabel[#] & /@ cell

removeInputLabel[Cell[CellGroupData[{cells__Cell}, opts___]]] := 
 Cell[CellGroupData[removeInputLabel[#] & /@ {cells}, opts]]

removeInputLabel[cell_Cell] := 
 Module[{}, 
  If[Count[cell, Rule[CellFrameLabels, __]] == 0, cell, 
   Replace[cell, 
    Rule[CellFrameLabels, a__] :> 
     Rule[CellFrameLabels, {{None, None}, {None, None}}], {1}]]]

removeInputLabel[x_] := x

makeVarLabel[label_String] := 
 Module[{}, {preprocessVar[label, "Named"], "NotebookTemplateSlot"}]

makeExpLabel[label_String, mode_, contentData_] := 
 Module[{},  {label, "NotebookTemplateExpression"}]


manageTemplateNameBlock[nbObj_NotebookObject] := ProcessWithFrontEnd[Module[
	{msgChannel, nb, exp, label, var, inherit, firstC, xInit, listInit, varInit, exprInit},
	 
    msgChannel = CurrentValue[$FrontEndSession, {MessageOptions, "KernelMessageAction"}];
    CurrentValue[$FrontEndSession, {MessageOptions, "KernelMessageAction"}] = "PrintToConsole";
    
    SelectionMove[nbObj, All, Cell];
    nb = NotebookGet[nbObj];
    exp = NotebookRead[nbObj];
    
    If[exp === {},
        AuthoringMessageDialog["You need to select a cell or cell group to define a repeating block.", WindowTitle -> "Repeating Block"],
        (* else *)
        (* avoid CurrentValue to work around CLOUD-1324 *)
        (* label = CurrentValue[NotebookSelection[nbObj], {CellFrameLabels, 2, 2}]; *)
        label = Lookup[Options[exp], CellFrameLabels, {{None, None}, {None, None}}][[2, 2]];
        label = Replace[label, Cell[BoxData[TemplateBox[arg_, "NotebookRepeatingBlock", ___]]] :> arg];

        {var, inherit} = If[MatchQ[label, {_, _, _}], label[[{1, 3}]], {None, True}];
        firstC = If[MatchQ[var, None], {}, Characters[var][[1]]];

        xInit = 1; listInit = ""; varInit = ""; exprInit = "";
        Which[firstC === {}, xInit = 1, 
        	firstC === "{", xInit = 1; listInit = var, 
        	firstC === "\"", xInit = 2; varInit = var, 
        	firstC =!= "", xInit = 3; exprInit = var
        ];
        
        CreateRepeatingDialog[
        	createRepeatingBlockDialog[xInit, listInit, varInit, exprInit, nbObj, inherit]
        ]
    ];
    
    CurrentValue[$FrontEndSession, {MessageOptions, "KernelMessageAction"}] = msgChannel;
]]

AuthoringMessageDialog[x___, o:OptionsPattern[]] := 
    If[TrueQ[CloudSystem`$CloudNotebooks],
    	CloudSystem`CreateCloudDialog[ToBoxes[x], o],
    	(* else *)
    	MessageDialog[x, o]
    ]

CreateRepeatingDialog[expr_] := If[TrueQ[CloudSystem`$CloudNotebooks],
	CloudSystem`CreateCloudDialog[ToBoxes[expr], WindowTitle -> "Repeating Block"],
	(* else *)
	DialogInput[expr, Modal -> True, WindowSize -> $RBWindowSize, WindowTitle -> "Repeating Block"]
]

createRepeatingBlockDialog[xInit_, listInit_, varInit_, exprInit_,
	nbObj_, inheritance_] := DynamicModule[
	{x = xInit, list = listInit, var = varInit, expr = exprInit, 
		inherit = inheritance},
	Pane[Column[{
		
		framedRBText[tr["RBDdescription", "title", $DialogTextOpts]],
		(* pending SAAS-11829 *)
		(*dividerImage[First@$RBWindowSize , 2],*)
		dividerImage[],
		
		Grid[{
			{Grid[{{RadioButton[Dynamic[x], 1], framedRBText[tr["RBDlist", $DialogTextOpts]]}}]},
			{InputField[Dynamic[list], String, BoxID -> "inputfield", FieldHint -> "{1,2,3,4}",
				FieldHintStyle -> $RepeatingBlockFieldHintStyle,
				Sequence @@ $RBInputFieldStyle]},
			{},
		    {Grid[{{RadioButton[Dynamic[x], 2], framedRBText[tr["RBDvariable", $DialogTextOpts]]}}]},
		    {InputField[Dynamic[var], String, BoxID -> "inputfield", FieldHint -> "variable",
		    	FieldHintStyle -> $RepeatingBlockFieldHintStyle,
		    	Sequence @@ $RBInputFieldStyle]},
		    {},
		    {Grid[{{RadioButton[Dynamic[x], 3], framedRBText[tr["RBDexpression", $DialogTextOpts]]}}]},
		    {InputField[Dynamic[expr], String, BoxID -> "inputfield", FieldHint -> "Range[10]",
		    	FieldHintStyle -> $RepeatingBlockFieldHintStyle,
		    	Sequence @@ $RBInputFieldStyle]},
		    {}}, Alignment -> Left], 

		Grid[{
			{
				Checkbox[Dynamic[inherit]],
				tr["RBinherit", $DialogTextOpts]
			},
			{}
		}, Alignment -> Bottom],
      
        Grid[{{
        	Item[
        		createButton[framedHelp[tr["Help", FontSize -> 12, $DialogTextOpts]],
        			SystemOpen["paclet:guide/AutomatedReports"],
                    ImageSize -> $DefaultButtonSize,
        			"default"
        		],
        		Alignment -> Left
        	],
        	Item["", ItemSize -> Fit],
        	Item[Grid[{{
        		createButton[framedCancel[tr["Cancel", FontSize -> 12, $DialogTextOpts]],
        			dialogReturnFunction[],
                    ImageSize -> {60,30},
        			"default"
        		],
        		createDefaultButton[framedOK[tr["OK", FontSize -> 12, FontColor -> White, $DialogTextOpts]],
        			okFun[x, list, var, expr, nbObj, inherit],
        			"RepeatingBlockDialog"
        			, ImageSize -> {60,30}
        		]
        	}}], Alignment -> Right]
        }}, Alignment -> Right]
        
    }, Alignment -> Left]
        (* In cloud dialogs, there is an interaction between Pane content and ImageSize/Margins
         * that can result in extra whitespace at the bottom of the dialog. I haven't been able
         * to find the minimal case yet. | dillont *) 
		, ImageMargins-> If[TrueQ[CloudSystem`$CloudNotebooks], {{13, 13}, {7, 10}}, {{13, 13}, {Automatic, Automatic}}]
		, ImageSize -> If[TrueQ[CloudSystem`$CloudNotebooks], 680, Automatic]
    ],
    InheritScope -> True
]


SetAttributes[dialogReturnFunction, HoldAll]

dialogReturnFunction[] := dialogReturnFunction[1]

dialogReturnFunction[arg_] := (arg;
    If[TrueQ[CloudSystem`$CloudNotebooks],
    	NotebookClose[EvaluationNotebook[]],
    	(* else *)
    	DialogReturn[]
    ]
)

okFun[x_, list_, var_, expr_, nbObj_, inherit_] := 
 Module[{label}, 
  dialogReturnFunction[
   label = Which[x == 1, makeExpLabel[list, "General", TextData], 
     x == 2, makeVarLabel[var], x == 3, 
     makeExpLabel[expr, "General", TextData]];
   SelectionMove[nbObj, All, Cell];
   CurrentValue[Cells[NotebookSelection[nbObj]][[1]], 
   	{CellFrameLabels, 2, 2}] = Cell[ BoxData[ TemplateBox[ Join[label,{inherit}],"NotebookRepeatingBlock" ]]];
   CurrentValue[
   Cells[NotebookSelection[nbObj]][[1]], {CellBracketOptions, "Color"}] = RGBColor[0.1574, 0.8708, 1.]		  
 ]]

(*Functions for Preview Report*)

generatePreview[nb_NotebookObject] := 
 Module[{exp}, exp = NotebookGet[nb];
  generatePreview[exp]]

generatePreview[nb_Notebook] := Module[
	{msgChannel, vars, newNB, margins},
	msgChannel = CurrentValue[$FrontEndSession, {MessageOptions, "KernelMessageAction"}];
	CurrentValue[$FrontEndSession, {MessageOptions, "KernelMessageAction"}] = "PrintToConsole";
	vars = CollectTemplateVariable[nb];
	newNB = System`GenerateDocument[nb, vars, "ProgressIndicator" -> False];
	If[CloudSystem`$CloudNotebooks =!= True,
	    margins = AbsoluteOptions[newNB, WindowMargins];
	    margins = margins /. x_Integer :> x + 20;
	    SetOptions[newNB, margins]
	];
	CurrentValue[$FrontEndSession, {MessageOptions, "KernelMessageAction"}] = msgChannel;
]

boxToName[box_String] := 
 If[StringMatchQ[box, "\"" ~~ ___ ~~ "\""], StringTrim[box, "\""], 
  ToExpression[box]]


fixPreview[TemplateArgBox[arg_, ___]] := fixPreview[arg]

fixPreview[arg_] := Module[
	{res},
	Quiet[res = ToExpression[arg]];
	If[res === $Failed, res = RawBoxes[arg]];
	res
]

(*Copied from NotebookTemplating... combine TODO add error handling*)

createVariable[var_] := Module[{arg},
	arg = ToExpression[RowBox[{"{", var, "}"}]];
	If[MatchQ[arg, {_String, ___}], arg = Prepend[arg, 1]];
	arg
]

postVariable[varIn_, previewIn_, mode_] := Module[
	{variable, preview},
	variable = createVariable[varIn];
	preview = fixPreview[previewIn];
	Sow[{variable, preview, mode}]
]



CollectTemplateVariable[cell_Cell] := Module[
	{label, variableDataIn, variableData, variable},
	label = getRepeatingBlockLabel[cell];
	If[varLabelQ[label],
		variableDataIn = Cases[cell, 
			TemplateBox[{var_, preview_, "Named", contentData_}, "NotebookTemplateSlot", ___] :> 
                postVariable[var, preview, "Named"],
            {1, Infinity}
        ];
        variableData = createData[variableDataIn];
        variable = createVariable[label];
        Sow[{variable, variableData, "Named"}],
        (* else *) 
	    Cases[cell,
	        TemplateBox[{var_, preview_, "Named", contentData_}, "NotebookTemplateSlot", ___] :> 
	            postVariable[var, preview, "Named"],
	        {1, Infinity}
	    ]
    ]
]

CollectTemplateVariable[
  Cell[CellGroupData[{headCell_, otherCells___}, ___], ___]] := 
 Module[{label, newHeadCell, variableData, variable}, 
  label = getRepeatingBlockLabel[headCell];
  If[repeatingLabelQ[label], newHeadCell = removeLabelsRB[headCell];
   variableData = 
    Reap[CollectTemplateVariable /@ {newHeadCell, otherCells}];
   variableData = createData[variableData];
   (*If an expression label,
   then prune any positional settings and resow.If a name,
   then add sub-bindings from the defaults.*)
   If[varLabelQ[label], 
    If[! MatchQ[variableData, {_Association}], Return[{}]];
    label = varLabel[label];
    variable = createVariable[label];
    Sow[{variable, variableData, "Named"}];];, 
   CollectTemplateVariable /@ {headCell, otherCells}];]


CollectTemplateVariable[Notebook[a_List, c___]] := 
 Module[{vars}, vars = Reap[CollectTemplateVariable /@ a];
  createData[vars]]

CollectTemplateVariable[nb_NotebookObject] := 
 Module[{exp}, exp = NotebookGet[nb];
  CollectTemplateVariable[exp]]

createData[variableDataIn_] := 
 Module[{variableData = variableDataIn, argLen, res}, 
  If[! (Length[variableData] === 2 && 
      Length[Part[variableData, 2]] > 0), Return[{}]];
  variableData = variableData[[2, 1]];
  (*var is a list each element is {{pos},value,mode} or {{pos,name},
  value,mode}*)argLen = Max[variableData[[All, 1, 1]]];
  res = Table[Null, {argLen}];
  Fold[fixElement, res, variableData]]


fixElement[res_, {{pos_}, val_, _}] := ReplacePart[res, pos -> val]

fixElement[res_, {{pos_, name_}, value_, _}] := 
 Module[{elem}, elem = Part[res, pos];
  If[Head[elem] =!= Association, elem = Association[]];
  elem[name] = value;
  ReplacePart[res, pos -> elem]]
  	      
            
End[] (* End Private Context *)

EndPackage[]
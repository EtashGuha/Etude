(*

Copied from docutools palatte tools by Bobs on 07/19/2017

*)

BeginPackage[ "FunctionResource`DocuToolsTemplate`" ];

ClearAll @@ Names[ $Context ~~ ___ ];

(* Exported symbols added here with SymbolName::usage *)

FunctionTemplateToggle
FunctionTemplateLiteralInput
TableInsert
TableMerge
TableSort
DocDelimiter

Begin[ "`Private`" ];


FunctionTemplateToggle`DT`FunctionTemplateToggle = FunctionTemplateToggle;
FunctionTemplateToggle`DT`FunctionTemplateLiteralInput = FunctionTemplateLiteralInput;


messageDialog[ tag_String, args___ ] :=
  Module[ { string },
      string = FunctionResource`DefinitionNotebook`Private`getString[ "MessageDialogs", tag ];
      If[ StringQ @ string && string =!= "",
          MessageDialog[ string, args ],
          MessageDialog[ tag, args ]
      ]
  ];

messageDialog[ other___ ] :=
  MessageDialog @ other;


$LiteralInputButton=False;

FunctionTemplateToggle[] := FunctionTemplateToggle[InputNotebook[]]
 
FunctionTemplateToggle[nb_] :=
	Module[{ci, re},  
		Block[{$DefinitionNotebook=nb},
			Catch[If[nb === $Failed, Throw[messageDialog["NoResourceCreateNotebook", WindowFrame -> "ModalDialog", WindowSize -> {300, All}]]];
				ci = CellInfo[nb]; 
				If[ci === $Failed, Throw[messageDialog["CursorBetweenCells", WindowFrame -> "ModalDialog", WindowSize -> {220, All}]]];
				If[MatchQ[ci, {{__, "CursorPosition" ->"CellBracket", __}}], Throw[messageDialog["CellBracketIsSelected", WindowFrame -> "ModalDialog", WindowSize -> {220, All}]]];
				If[multipleCellBracketsSelected[ci], Throw[messageDialog["MultipleCellsHaveBeenSelected", WindowFrame -> "ModalDialog", WindowSize -> {240, All}]]];
				If[("ContentData" /. ci) === {TextData} && MatchQ["CursorPosition" /. ci, {{a_Integer, a_Integer}}],
					FrontEndExecute[{FrontEnd`FrontEndToken[nb, "MovePrevious"]}]; 
					ci2 = CellInfo[nb]; 
					If[(ci /. ("CursorPosition" -> _) -> Sequence[]) === (ci2 /. ("CursorPosition" -> _) -> Sequence[]), 
						FrontEndExecute[{FrontEnd`FrontEndToken[nb, "MoveNext"]}]; 
			       			FrontEndExecute[{FrontEnd`FrontEndToken[nb, "SelectPreviousWord"]}]; 
				   		re = OldNotebookRead[nb]; 
						If[FreeQ[re, "InlineFormula"], 
							FunctionTemplate[], 
							FunctionTemplate["RestoreText"]], 
						If[("Style" /. ci2) === {"InlineFormula"},
							FrontEndExecute[{FrontEnd`FrontEndToken[nb, "MoveNext"]}]; 
							FrontEndExecute[{FrontEnd`FrontEndToken[nb, "SelectPreviousWord"]}]; 
							FunctionTemplate["RestoreText"], 
							FrontEndExecute[{FrontEnd`FrontEndToken[nb, "MoveNext"]}]; 
							FunctionTemplate[]]], 
					FunctionTemplate[]]]]]
 
FunctionTemplateLiteralInput[args___] := Block[{$LiteralInputButton=True},
	FunctionTemplateToggle[args]
]
 
Attributes[FrontEnd`FileName] = {HoldAll, ReadProtected}
 
Attributes[MessageToConsole] = {HoldFirst}
 
MessageToConsole[symbolWithValue_, 
     values___] := MessageDialog[
     If[{values} === {}, symbolWithValue, 
      StringReplace[symbolWithValue, 
       MapIndexed[StringJoin["`", ToString[#2[[1]]], "`"] -> #1 & , 
        {values}]]], WindowFloating -> True]
 
MessageToConsole /: MessageToConsole::usage = 
     "Utility function for sending messages to the Messages notebook."
 
CellInfo[obj:_NotebookObject | _CellObject] := 
    MathLink`CallFrontEnd[FrontEnd`CellInformation[obj]]
 
CellInfo /: CellInfo::usage = 
     "Utility function for returning CellInformation packet data."
 
multipleCellBracketsSelected[x_] := 
    MatchQ[x, {{"Style" -> _, __}, {"Style" -> _, __}, ___}]
 
ci2 = {{"Style" -> "Text", "ContentData" -> TextData, 
      "ContentDataForm" -> TextForm, "Evaluating" -> False, 
      "Rendering" -> False, "NeedsRendering" -> False, 
      "CursorPosition" -> {24, 24}, "FirstCellInGroup" -> False, 
      "CellSerialNumber" -> 10080, "Formatted" -> True, 
      "ExpressionUUID" -> "9c4891f7-75dc-496e-8cf1-4394e5e1dafa"}}
 
OldNotebookRead[nb_] := 
    NotebookRead[nb, "WrapBoxesWithBoxData" -> True]
 
OldNotebookRead /: OldNotebookRead::usage = "Same as \
NotebookRead but uses the option \"WrapBoxesWithBoxData\" -> True to get the \
old behavior."
 
FunctionTemplate["Plain"] := 
    Module[{nb = InputNotebook[], ci}, ci = CellInfo[nb]; 
      If[("ContentData" /. CellInfo[nb]) === {BoxData}, Abort[]]; 
      Which[CursorInsideCellAndNonEmptySelection[ci], 
      	FrontEndExecute[
        {FrontEnd`FrontEndToken[nb, "CreateInlineCell"], 
         FrontEnd`FrontEndToken[nb, "MoveNext"], 
         FrontEnd`FrontEndToken[nb, "MoveNext"]}], 
       CursorInsideCellButEmptySelection[ci], 
       FrontEndExecute[{FrontEnd`FrontEndToken[nb, "CreateInlineCell"]}]]]
 
FunctionTemplate["RestoreText"] := 
	Module[{nb = $DefinitionNotebook, ci, re}, 
		Catch[With[{}, 
			If[nb === $Failed, Throw[messageDialog["NoResourceCreateNotebook", WindowFrame -> "ModalDialog", WindowSize -> {300, All}]]];
			ci = CellInfo[nb]; 
			If[ci === $Failed, Throw[messageDialog["CursorBetweenCells", WindowFrame -> "ModalDialog", WindowSize -> {220, All}]]];
			If[multipleCellBracketsSelected[ci], Throw[messageDialog["MultipleCellsHaveBeenSelected", WindowFrame -> "ModalDialog", WindowSize -> {240, All}]]];
			Which[("Style" /. ci) === {"InlineFormula"},
			
				While[("Style" /. CellInfo[nb]) === {"InlineFormula"}, 
					FrontEndExecute[{FrontEndToken[nb, "ExpandSelection"]}]]; 
				re = restoreText[OldNotebookRead[nb]]; 
				If[StringQ[re] || MatchQ[re, RowBox[{___String, SuperscriptBox[_String, _String], ___String}]],
					NotebookWrite[nb, re, All],
					messageDialog["SelectionHasUnhandledForm", WindowFrame -> "ModalDialog", WindowSize -> {530, All}]],
					
				("CursorPosition" /. ci) === {"CellBracket"},
				
				re = OldNotebookRead[nb] /. Cell[TextData[{a__}], b___] :> Cell[TextData[(If[MatchQ[#1, Cell[_, "InlineFormula", ___]], restoreText[#1[[1]]], #1] & ) /@ {a}], b]; 
				If[MatchQ[re, Cell[TextData[{__String}], __]], 
					NotebookWrite[nb, re, All], 
					messageDialog["SelectedUnhandled",
							WindowFrame -> "ModalDialog", WindowSize -> {680, All}]],
					
				re = OldNotebookRead[nb];
				MatchQ[re, BoxData[_]],
				
				re = restoreText[re];
				NotebookWrite[nb, re, All],
				
				re = OldNotebookRead[nb];
				MatchQ[re, {(Cell[BoxData[ButtonBox[_String, BaseStyle -> _]], _String] | _String)..}],
				
				re = restoreText[re]; 
				NotebookWrite[nb, re, All], 
				
				True,
				
				MessageToConsole[FunctionTemplate::inappsel]]]]]
 
FunctionTemplate[] := 
	Catch[Module[{nbo = $DefinitionNotebook, celinf,  objName = GuessObjectName[$DefinitionNotebook], threeparts, re}, 
			celinf = CellInfo[nbo]; 
			If[ !MatchQ[celinf, {{___}}], Throw[$Failed, BadSelection]]; 
			threeparts = {"ContentData", "CursorPosition", "InlineCellPosition"} /. First[celinf]; 
		With[{}, 
			Replace[threeparts, 
				{{TextData, cp_, icp_} :> (If[MatchQ[cp, {p_, p_}], FrontEndTokenExecute[nbo, "SelectPreviousWord"]]; 
								re = OldNotebookRead[nbo]; 
								textDataReplacement[re, nbo,objName]), 
				{BoxData, cp_, icp_} :> (If[MatchQ[cp, {p_, p_}], SelectionMove[nbo, All, If[icp === "InlineCellPosition", Cell, CellContents]]]; 
								re = OldNotebookRead[nbo]; 
								boxDataReplacement[re, nbo, objName,cp, icp]; 
								SelectionMove[nbo, After, Character]; 
								If[MatchQ[cp, {p_, p_}] && icp =!= "InlineCellPosition", FrontEndExecute[FrontEndToken[nbo, "MoveNext"]]])}]]], 
		BadSelection]
 
FunctionTemplate /: FunctionTemplate::betwcells = 
     "The cursor is between cells or not inside an input notebook."
 
FunctionTemplate /: FunctionTemplate::inappsel = "The \
cursor must be inside an \"InlineFormula\" cell, selecting part of such a \
cell or selecting a cell bracket."
 
FunctionTemplate /: FunctionTemplate::inappstruc = "The \
expression in the selection has a form which cannot be handled by this \
function."
 
FunctionTemplate /: FunctionTemplate::inappstruc2 = "One \
or more \"InlineFormula\" expressions in the selected cell have forms which \
cannot be handled by this function."
 
FunctionTemplate /: FunctionTemplate::mulcell = 
     "Multiple cells have been selected."
 
FunctionTemplate /: FunctionTemplate::noin = 
     "There is no input notebook."
 
FunctionTemplate /: FunctionTemplate::usage = "Initiates \
a function template, also formats a function template string selection into \
the proper format. With the argument \"RestoreText\" an \"InlineFormula\" \
function template is converted back into a string. This will occur if the \
cursor is inside or selecting part of an \"InlineFormula\" function template. \
With the cursor at the cell bracket, all \"InlineFormula\" function templates \
within the cell are converted into strings."
 
CursorInsideCellAndNonEmptySelection[x_] :=
	Module[{curpos}, 
		MatchQ[x, {{"Style" -> _, __}}] && (curpos = ("CursorPosition" /. x)[[1]]; ListQ[curpos] && !SameQ @@ curpos)]
 
CursorInsideCellButEmptySelection[x_] := 
	Module[{curpos}, MatchQ[x, {{"Style" -> _, __}}] && (curpos = ("CursorPosition" /. x)[[1]]; ListQ[curpos] && SameQ @@ curpos)]
 
restoreText[x_] := 
	Module[{expr}, 
		expr = ReplaceRepeated[ReplaceAll[ReplaceAll[ReplaceAll[ReplaceAll[ReplaceAll[ReplaceAll[x, Cell[BoxData[ButtonBox[a_, BaseStyle -> _]], _String] :> a], ButtonBox[a_, ___] :> a], 
											StyleBox["\[Ellipsis]", "TR"] :> "$$"], 
									SubscriptBox[StyleBox[a_String, _], b_String] :> a <> "$" <> b], 
								SubscriptBox[StyleBox[a_String, _], StyleBox[b_String, _]] :> a <> "$" <> b],
							StyleBox[a_String, _] :> a], 
					RowBox[{a__String}] :> StringJoin[a]]; 
		If[MatchQ[expr, {__String}], StringJoin @@ expr, expr[[1]]]]

$DefinitionNotebook=$Failed;
 
GuessObjectName[nbo_NotebookObject] := 
	Replace["FileName" /. NotebookInformation[nbo], 
		{FrontEnd`FileName[l_List, fn_String, ___] :> StringReplace[fn, {"-Definition.nb" -> "", ".nb" -> ""}], 
		_ :> Replace[Cases[NotebookGet[nbo], Cell[s_String, "Title", ___] :> s, -1, 2], {{} -> $Failed, {one_} :> StringReplace[one, " " ~~___ :> ""], _ -> $Failed}]}]
 
$DocumentationDirectory := FileNameJoin[{$InstallationDirectory,"Documentation","English"}];
 
$DocumentationDirectory /: 
    $DocumentationDirectory::usage = 
     "Directory variable for documentation system source files."

refPageLink[cont_]:=
  With[ { name = StringReplace[ReplaceAll[cont, {BoxData[a_] :> a}], {".nb" -> ""}] },
      SystemOpen[ "paclet:ref/" <> name ] /; StringQ @ name
  ];

eliminateTIStyleBoxFromLinearSyntaxStrings[
     expr_] := expr /. 
     x_String /; StringMatchQ[x, 
        "\"\!\(\*StyleBox["~~__~~",\"TI\"]\)\""] :> 
      StringReplace[x, "\"\!\(\*StyleBox["~~
         a__~~",\"TI\"]\)\"" :> a]
 
ParseTextTemplate[tpl_String,  thisObject_:""] := 
    If[True || SyntaxQ[tpl], 
     MathLink`CallFrontEnd[FrontEnd`ReparseBoxStructurePacket[
        tpl]] /. {thisObject -> thisObject, strg_String :> StylizeTemplatePart[strg], 
       RowBox[{objName_String, "::", msgTag:Except["tag", _String]}] :> 
        StylizeMessageName[objName, msgTag, thisObject]}
         ,

     With[{qzones = Interval @@ (#1 + {1, -1} & ) /@ 
          Select[Partition[First /@ StringPosition[tpl, 
              "\""], 2], #1 . {-1, 1} > 1 & ]}, 
      With[{wordData = ({First[#1], Length[#1]} & ) /@ 
          Split[MapIndexed[IntervalMemberQ[qzones, First[
                #2]] || LetterQ[#1] || DigitQ[#1] || #1 === "$" & , 
            Characters[tpl]]]}, 
       RowBox[Reap[Fold[(Sow[Replace[StringTake[#1, Last[#2]], {
                thisObject -> thisObject, 
                w_ /; First[#2] :> 
                 StylizeTemplatePart[w], 
                w_ :> StringReplace[w, 
                  " " -> ""]}], ParseTextTemplate]; 
             StringDrop[#1, Last[#2]]) & , tpl, 
           wordData], ParseTextTemplate][[
         -1,-1]]]]]]
 
StylizeTemplatePart["$$"] = StyleBox["\[Ellipsis]", "TR"]
 
StylizeTemplatePart[str_String, (opts___)?OptionQ]:=str/;$LiteralInputButton

StylizeTemplatePart[strg_String, (opts___)?OptionQ] :=
	If[StringLength[strg] > 2 && "\"\"" === StringDrop[strg, {2, -2}],
		Replace[If[MemberQ[$ArgLabelInitialLetters, StringTake[strg, {2, 2}]], StylizeArgumentLabel[StringTake[strg, {2, -2}], True], strg],
			{s_String :> NonSymbolReferenceLink[strg],
				StyleBox[s_String, sty_String] :> StringJoin["\"\!\(\*StyleBox[\"", s, "\",\"", sty, "\"]\)\""],
				SubscriptBox[StyleBox[s1_String, sty1_String], StyleBox[s2_String, sty2_String]] :> StringJoin["\"\!\(\*SubscriptBox[StyleBox[\"", s1, "\",\"", sty1, "\"],StyleBox[\"",
																s2, "\",\"", sty2, "\"]]\)\""],
				RowBox[pieces:{(StyleBox[_String, _String] | _String)..}] :> StringJoin["\"\<", StringEmbed /@ pieces, "\>\""],
				_ :> RowBox[{"\"", StylizeTemplatePart[StringTake[strg, {2, -2}], opts], "\""}]}],
      If[ MemberQ[$ArgLabelInitialLetters, StringTake[strg, 1]],
          StylizeArgumentLabel[strg],
          FunctionLinkButton[strg, Null]
      ]
	]

$GreekLowerCase = {"\[Alpha]", "\[Beta]", "\[Gamma]", 
     "\[Delta]", "\[Epsilon]", "\[CurlyEpsilon]", "\[Zeta]", "\[Xi]", 
     "\[Eta]", "\[Theta]", "\[Iota]", "\[Kappa]", "\[Lambda]", "\[Mu]", 
     "\[Nu]", "\[Omicron]", "\[Pi]", "\[Rho]", "\[Sigma]", "\[Tau]", 
     "\[Upsilon]", "\[Phi]", "\[CurlyPhi]", "\[Psi]", "\[Chi]", "\[Omega]"}
 
$GreekUpperCase = {"\[CapitalAlpha]", "\[CapitalBeta]", 
     "\[CapitalGamma]", "\[CapitalDelta]", "\[CapitalEpsilon]", 
     "\[CapitalZeta]", "\[CapitalXi]", "\[CapitalEta]", "\[CapitalTheta]", 
     "\[CapitalIota]", "\[CapitalKappa]", "\[CapitalLambda]", "\[CapitalMu]", 
     "\[CapitalNu]", "\[CapitalOmicron]", "\[CapitalPi]", "\[CapitalRho]", 
     "\[CapitalSigma]", "\[CapitalTau]", "\[CapitalUpsilon]", 
     "\[CapitalPhi]", "\[CapitalPsi]", "\[CapitalChi]", "\[CapitalOmega]"}
     
$ArgLabelInitialLetters = 
    Join[{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", 
      "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"}, 
     $GreekLowerCase, $GreekUpperCase]
 
StylizeArgumentLabel[s_String, stringContent_:False] := 
	Replace[StringSplit[s, "$", 2],
		{{one_} :> If[TrueQ[stringContent],
				Replace[StringSplit[one, "."], 
					{{fileBase_, fileExts__} :> RowBox@ReplaceRepeated[Riffle[Join[{Italicization[fileBase]}, 
													Replace[{fileExts}, "ext" -> Italicization["ext"], {1}]], "."],
											{a___, b1_String, b2__String, c___} :> {a, StringJoin[b1, b2], c}], 
					_ -> Italicization[one]}], 
				Italicization[one]], 
		{base_, sub_} :> StylizeSubscriptedArgument[base, sub]}]
 
Italicization[strg_String] := StyleBox[strg, If[GreekLetterQ[strg] || DigitQ[strg], "TR", "TI"]]
 
Italicization[other_] := other
 
GreekLetterQ[strg_String] := Complement[Characters[strg], $GreekUpperCase, $GreekLowerCase] === {}
 
StylizeSubscriptedArgument[base_, sub_] := SubscriptBox[Italicization[base], Italicization[sub]]
 
NonSymbolReferenceLink[strg_] := 
	With[{refType = ReferenceType[strg]},
		Replace[refType, {None -> strg, t_String :> ButtonBox[strg, BaseStyle -> "Link", ButtonData -> StringJoin["paclet:ref/", refType, "/", StringTake[strg, {2, -2}]]]}]]
 
ReferenceType[strg_ /; StringDrop[strg, {2, -2}] === "\"\""] := Replace[StringTake[strg, {2, -2}], Append[ReferenceNames[], _ -> None]]
 
ReferenceType[_] := None
 
ReferenceNames[] := ReferenceNames[] = 
	Map[Apply[Alternatives, StringReplace[#, {DirectoryName[#] -> "", ".nb" -> ""}] & /@ FileNames["*.nb", ToFileName[{$DocumentationDirectory, "System", "ReferencePages"},
		#]]] -> (# /. $ReferenceSubcategoryUriSubstring) &, {"Methods", "Formats", "C", "Entities", "Interpreters", "Programs", "NetFunctions", "Devices", "Services", "EmbeddingFormats",
									"FrontEndObjects", "MenuItems", "NetEncoders", "NetDecoders", "Classifiers", "Predictors", "ExternalEvaluationSystems"}]

$ReferenceSubcategoryUriSubstring = {s_String :> If[s === "Entities", "entity", ToLowerCase[StringDrop[s, -1]]]}
 
StringEmbed[s_String] := s
 
StringEmbed[StyleBox[s_String, 
      sty_String]] := StringJoin["\!\(\*StyleBox[\"", 
     s, "\",\"", sty, "\"]\)"]
 
FunctionLinkableQ[expr_] := And[$MakeLinks, AtomQ[expr], Head[expr] === String, StringMatchQ[expr, (WordCharacter | "$") ..], Names["System`" <> expr] =!= {}]
 
FunctionLinkableQ /: FunctionLinkableQ::usage = "Function\
LinkableQ[expr] returns True if\nexpr is a string that probably ought to be \
represented in documentation\nas a FunctionLink button."
 
$MakeLinks = True
 
$MakeLinks /: $MakeLinks::usage = "$MakeLinks is a global \
variable specifying whether\nformatting constructors like FunctionTemplate \
should make links\nfor symbol names that look like built-in function names."
 
$BuiltInSymbolInitialLetters = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "$"}


FunctionLinkButton[ name_String, _ ] :=
  With[ { button = documentationButton @ name },
      If[ MatchQ[ button, _TagBox ],
          button,
          name
      ]
  ];

docTypePriority["Symbols"]      = 1;
docTypePriority["Services"]     = 2;
docTypePriority["Formats"]      = 3;
docTypePriority["Interpreters"] = 4;

docTypePriority[_] := Infinity;


$documentationLocations :=
  $documentationLocations = SortBy[
      Select[
          FileNames[
              "*",
              FileNameJoin[
                  {
                      $InstallationDirectory,
                      "Documentation",
                      "English",
                      "System",
                      "ReferencePages"
                  }
              ]
          ],
          DirectoryQ
      ],
      docTypePriority @* FileBaseName
  ];


documentationButton // ClearAll;
documentationButton[name_] :=
  Module[
      {file},
      file = SelectFirst[
          Map[
              FileNameJoin[{#1, StringJoin[StringTrim[name, "\""], ".nb"]}] & ,
              If[
                  StringMatchQ[name, "\""~~__~~"\""],
                  Reverse[$documentationLocations],
                  $documentationLocations
              ]
          ],
          FileExistsQ
      ];
      If[
          StringQ[file],
          With[
              {type = FileNameTake[file, {-2}]},
              documentationButton[name] = TagBox[
                  ButtonBox[
                      StyleBox[
                          name,
                          StringJoin[type, "RefLink"],
                          ShowStringCharacters -> True,
                          FontFamily -> "Source Sans Pro"
                      ],
                      BaseStyle -> Dynamic[
                          FEPrivate`If[
                              CurrentValue["MouseOver"],
                              {"Link", FontColor -> RGBColor[0.854902, 0.396078, 0.145098]},
                              {"Link"}
                          ]
                      ],
                      ButtonData -> formatRefLinkURL @ StringRiffle[{"paclet:ref", type, FileBaseName[file]}, "/"]
                  ],
                  MouseAppearanceTag["LinkHand"]
              ]
          ],
          documentationButton[name] = None
      ]
  ];

documentationButton[name:"ResourceFunction"|"DefineResourceFunction"] :=
  TagBox[
      ButtonBox[
          StyleBox[
              name,
              "SymbolsRefLink",
              ShowStringCharacters -> True,
              FontFamily -> "Source Sans Pro"
          ],
          BaseStyle -> Dynamic[
              FEPrivate`If[
                  CurrentValue["MouseOver"],
                  {"Link", FontColor -> RGBColor[0.854902, 0.396078, 0.145098]},
                  {"Link"}
              ]
          ],
          ButtonData -> "paclet:ref/" <> name
      ],
      MouseAppearanceTag["LinkHand"]
  ];


formatRefLinkURL[ tgt_ ] :=
  Module[ { split, renamed },

      split = DeleteCases[ StringSplit[ tgt, "/" ], "Symbols" ];

      renamed = Replace[ split,
                         { a_, type_, name_ } :>
                           { a, StringDelete[ ToLowerCase @ type, "s"~~EndOfString ], name }
                ];

      StringRiffle[ renamed, "/" ]
  ];

 
StylizeMessageName[objName_, msgTag_, thisObject_:""] := RowBox[{Replace[objName, {thisObject -> thisObject, _ :> StylizeTemplatePart[objName]}], "::", msgTag}]


textDataReplacement[re_, nbo_, objName_]:=(
	Replace[re, 
		{s_String :> NotebookWrite[nbo, Cell[BoxData[ParseTextTemplate[s, objName]], 
	            "InlineFormula", ShowStringCharacters -> False, FontFamily -> "Source Sans Pro"], All],
	    b_BoxData :> 
	         NotebookWrite[nbo, Cell[b /. 
	             {btn_ButtonBox :>  btn, 
	              st_StyleBox :> st, 
	              objName -> objName, 
	              c_Cell :> c, 
	              strg_String :> 
	               StylizeTemplatePart[strg], 
	               RowBox[{msgObjName_String, "::", 
	                 msgTag:Except["tag", _String]}] :> 
	               StylizeMessageName[msgObjName, msgTag, objName]}, 
	            "InlineFormula", ShowStringCharacters -> False, FontFamily -> "Source Sans Pro"], All],
	      {umi:Cell[_, "ModInfo" | "UsageModInfo", ___], s_String} :> 
	      	(SetOptions[NotebookSelection[nbo], Deletable -> True]; 
	      	NotebookWrite[nbo,(#1& )[
	            TextData[{umi, Cell[BoxData[ParseTextTemplate[s, objName]], 
	               "InlineFormula", ShowStringCharacters -> 
	                False, FontFamily -> "Source Sans Pro"]}]], All])}];
     SetOptions[NotebookSelection[nbo], ShowStringCharacters -> Inherited]; 
     SelectionMove[nbo, After, Character]
     )




boxDataReplacement[re_, nbo_, objName_,cp_, icp_]:=Module[{func},
	With[{
	before = Replace[OldNotebookRead[nbo], 
		{BoxData[_GridBox] :> Throw[$Failed, BadSelection], 
			Cell[cont_, ___] :> cont}]}, 
		If[MatchQ[cp, {p_, p_}] && icp === "InlineCellPosition", 
			SelectionMove[nbo, All, CellContents]]; 
		NotebookWrite[nbo,func = (#1 /. 
			{btn_ButtonBox :>btn, 
             st_StyleBox :> st, 
             objName ->objName, 
             c_Cell :> c, 
             SubscriptBox[base_, sub_] :> 
             	StylizeSubscriptedArgument[base, sub], 
             strg_String :> StylizeTemplatePart[strg], 
             RowBox[{msgObjName_String, "::", msgTag:Except["tag", _String]}] :> 
             	StylizeMessageName[
                         msgObjName, 
                         msgTag, 
                         objName]} &); 
       If[MatchQ[before, BoxData[GridBox[__]]], 
         Cell[MapAt[func, before, {1, 1}], "InlineFormula", ShowStringCharacters -> False, FontFamily -> "Source Sans Pro"],
         Cell[
             If[Head@# === BoxData, #, BoxData@#]&[func@before],
             "InlineFormula",
             ShowStringCharacters -> False,
             FontFamily -> "Source Sans Pro"
         ]
       ] /. ButtonBox[s: "XXXX"|"\[Placeholder]", __] :> s, All]]
]

ExpandToCell[x_NotebookObject] := 
	Module[{lnkre}, 
		While[(LinkWrite[$ParentLink, FrontEnd`CellInformation[x]]; 
			lnkre = LinkRead[$ParentLink]);
			(lnkre =!= $Failed && Not[MemberQ["CursorPosition" /. lnkre, "CellBracket"]]), 
			FrontEndExecute[FrontEnd`SelectionMove[x, All, Cell, AutoScroll -> False]]]]
			
TableInsert /: TableInsert::usage = "TableInsert[n] inserts an n column table."

TableInsert[colnum_] :=
  Module[{nb = InputNotebook[]},
      If[ nb === $Failed,
          messageDialog[
              "NoResourceCreateNotebook",
              WindowFrame -> "ModalDialog",
              WindowSize -> {300, All}
          ],
          If[ Developer`CellInformation[nb] =!= $Failed,
              ExpandToCell[nb];
              SelectionMove[nb, After, Cell]
          ];
          NotebookWrite[
              nb,
              Cell[BoxData[GridBox[{#1, #1}]], "TableNotes"] &[
                  Table[Cell["\[Placeholder]", "TableText"], colnum]
              ],
              All
          ];
          SelectionMove[nb, All, CellContents];
          SelectionMove[nb, Before, CellContents];
          NotebookFind[nb, "\[Placeholder]"]
      ]
  ];

TableMerge /: TableMerge::usage = "TableMerge[] merges selected tables."

TableMerge[] :=
	Module[{nb = InputNotebook[], selcells, re, dims, content}, 
		Which[nb === $Failed, 
			messageDialog["NoResourceCreateNotebook", WindowFrame -> "ModalDialog", WindowSize -> {300, All}],
			(selcells = SelectedCells[nb]) === {}, 
			messageDialog["NoSelectedCells", WindowFrame -> "ModalDialog", WindowSize -> {220, All}],
			Union[("Style" /. Developer`CellInformation[#]) & /@ selcells] =!= {"TableNotes"}, 
			messageDialog["NotAllSelectedAreTableNotes", WindowFrame -> "ModalDialog", WindowSize -> {320, All}],
			Not@MatchQ[(re = NotebookRead /@ selcells), {Cell[BoxData[GridBox[{{__} ..}]], "TableNotes", ___] ..}], 
			messageDialog["NotAllSelectedContainTables", WindowFrame -> "ModalDialog", WindowSize -> {320, All}],
			Not[SameQ @@ (Last /@ (dims = Dimensions /@ (content = #[[1, 1, 1]] & /@ re)))], 
			messageDialog["SelectedIncompatibleColumnCounts", WindowFrame -> "ModalDialog", WindowSize -> {400, All}],
			True, 
			If[Length@selcells > 1,
				NotebookDelete /@ Rest@selcells; 
				NotebookWrite[selcells[[1]], Cell[BoxData[GridBox[Join @@ content]], "TableNotes"], All]]]]
				
TableSort /: TableSort::usage = "TableSort[n] sorts a table based on entries in its nth row."

TableSort[n_Integer?Positive] :=
	Module[{nb = InputNotebook[], ci, re, repart, keycolumn, forms, PlainStringForms, RowsWithNonQuotedFirstElement, RowsWithNonQuotedFirstElement2, RowsWithQuotedFirstElement, 
		RowsWithQuotedFirstElement2, newgrid, ce}, 
		Catch[If[nb === $Failed, 
			messageDialog["NoResourceCreateNotebook", WindowFrame -> "ModalDialog", WindowSize -> {300, All}]];
			ci = Developer`CellInformation[nb];
			If[(*The cursor was either not inside a notebook or not inside a cell or at a cell's bracket.*)ci === $Failed, 
				Throw[messageDialog["CursorOutsideTableCellOrBracket", WindowFrame -> "ModalDialog", WindowSize -> {520, All}]]];
			ExpandToCell[nb];
			ci = Developer`CellInformation[nb];
			If[(*The cursor was not initially positioned inside or at the cell bracket of a cell with a table style.*)
				Not@MatchQ[ci, {{"Style" -> "TableNotes", __}}], 
				Throw[messageDialog["CursorOutsideTableCell", WindowFrame -> "ModalDialog", WindowSize -> {300, All}]]];
			re = NotebookRead[nb];
			If[(*The cell does not have the correct structure.*)
				Not@MatchQ[re, Cell[BoxData[GridBox[_?MatrixQ, ___]], __]], 
				Throw[messageDialog["UnhandledSortStructure", WindowFrame -> "ModalDialog", WindowSize -> {320, All}]]];
			repart = re[[1, 1, 1]];
			If[(* n is > the number of columns in the table. *)
				n > Dimensions[repart][[2]], 
				Throw[messageDialog["NumberOfTableColumns", WindowFrame -> "ModalDialog", WindowSize -> {520, All}]]];
			keycolumn = repart[[All, n]];
			forms = {_String, (StyleBox | ButtonBox | Cell)[_String, __], RowBox[{ButtonBox[_String, __] | _String, "[", ___, "]"}], StyleBox[RowBox[{"\"", _String, "\""}], __], 
					RowBox[{"{", RowBox[{_String, ",", __}], "}"}]};
			repart = If[Position[repart, {_, _, "\[SpanFromLeft]"}] =!= {}, 
					repart //. {a___, PatternSequence[x : {_, _, "\[SpanFromLeft]"}, y_], b___} :> {a, Join[x, y], b}, repart];
			If[Not[And @@ (MatchQ[#, Alternatives @@ forms] & /@ keycolumn)], 
				Throw[messageDialog["UnhandledTableSortColumn",
							WindowFrame -> "ModalDialog", WindowSize -> {420, All}]]];
			PlainStringForms = {x_String /; Not@StringMatchQ[x, "\"" ~~ __ ~~ "\""], StyleBox[x_String /; Not@StringMatchQ[x, "\"" ~~ __ ~~ "\""], __], 
						ButtonBox[x_String /; Not@StringMatchQ[x, "\"" ~~ __ ~~ "\""], __],
						RowBox[{ButtonBox[x_String /; Not@StringMatchQ[x, "\"" ~~ __ ~~ "\""], __] | (x_String /; Not@StringMatchQ[x, "\"" ~~ __ ~~ "\""]), "[", ___, "]"}], 
						RowBox[{"{", RowBox[{x_String /; Not@StringMatchQ[x, "\"" ~~ __ ~~ "\""], ",", __}], "}"}], Cell[x_String /; Not@StringMatchQ[x, "\"" ~~ __ ~~ "\""], __]};
			RowsWithNonQuotedFirstElement = Cases[repart, x_ /; MatchQ[x[[n]], Alternatives @@ PlainStringForms]];
			RowsWithNonQuotedFirstElement2 = MapAt[Switch[#, _String, {#, 1}, (StyleBox | ButtonBox | Cell)[_String, __], {#[[1]], 1}, RowBox[{_String, "[", ___, "]"}], {#[[1, 1]], 2}, 
						RowBox[{ButtonBox[_String, __], "[", ___, "]"}], {#[[1, 1, 1]], 2},
						RowBox[{"{", RowBox[{_String, ",", __}], "}"}], {#[[1, 2, 1, 1]], 3}, _, {#, 1}] &, #, n] & /@ RowsWithNonQuotedFirstElement;
			RowsWithQuotedFirstElement = Complement[repart, RowsWithNonQuotedFirstElement];
			RowsWithQuotedFirstElement2 = MapAt[Switch[#, _?(StringQ[#] && StringMatchQ[#, "\"" ~~ __ ~~ "\""] &),
				{StringTake[#, {2, -2}], 1}, (StyleBox | ButtonBox | Cell)[_?(StringQ[#] && StringMatchQ[#, "\"" ~~ __ ~~ "\""] &), __], {StringTake[#[[1]], {2, -2}], 1}, 
				RowBox[{ButtonBox[_?(StringQ[#] && StringMatchQ[#, "\"" ~~ __ ~~ "\""] &), __], "[", ___, "]"}], {StringTake[#[[1, 1, 1]], {2, -2}], 2}, 
				RowBox[{"{", RowBox[{_?(StringQ[#] && StringMatchQ[#, "\"" ~~ __ ~~ "\""] &), ",", __}], "}"}],
				{StringTake[#[[1, 2, 1, 1]], {2, -2}], 3}, _, {#, 1}] &, #, n] & /@ RowsWithQuotedFirstElement;
			newgrid = Join[RowsWithNonQuotedFirstElement[[Ordering[RowsWithNonQuotedFirstElement2[[All, {n}]], All, 
					If[#1[[1]] === #2[[1]], OrderedQ[{#1[[1, 2]], #2[[1, 2]]}], OrderedQ[{#1[[1]], #2[[1]]}]] &]]], 
					RowsWithQuotedFirstElement[[Ordering[RowsWithQuotedFirstElement2[[All, {n}]], All, If[#1[[1]] === #2[[1]], OrderedQ[{#1[[1, 2]], #2[[1, 2]]}], 
					OrderedQ[{#1[[1]], #2[[1]]}]] &]]]] /. {a__, "\[SpanFromLeft]", b__} :> Unevaluated[Sequence[{a, "\[SpanFromLeft]"}, {b}]];
			ce = ReplacePart[re, newgrid, {1, 1, 1}];
			NotebookWrite[nb, ce, All]]]

TableSort[] := TableSort[1]

DocDelimiter /: DocDelimiter::usage = "DocDelimiter[] inserts a delimiter cell."

DocDelimiter[] :=
	Module[{nb = InputNotebook[], lineCell = Cell[BoxData[InterpretationBox[Cell["\t", "ExampleDelimiter"], ($Line = 0; Null)]], "ExampleDelimiter"]}, 
		If[nb === $Failed, 
			messageDialog["NoResourceCreateNotebook", WindowFrame -> "ModalDialog", WindowSize -> {300, All}],
			If[Developer`CellInformation[nb] === $Failed,
				NotebookWrite[nb, lineCell],
				ExpandToCell[nb]; 
				SelectionMove[nb, After, Cell];
				NotebookWrite[nb, lineCell]]]]



End[] 

EndPackage[]


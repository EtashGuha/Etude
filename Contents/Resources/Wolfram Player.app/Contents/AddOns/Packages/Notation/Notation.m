(* NotationSource.m :  Jason Harris,  October 24, 2012  at 23:14:31.       *)
(* This file is machine generated from NotationSource.nb*)
(* Please consult the source file NotationSource.nb. *)


(*
   Jason Harris, (c)1996-2008
*)

(*
   Notation Source Code
*)



(* ------------------------------------------------------------------------------------------------------------------ *)
(*   Package Beginnings  -------------------------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------------------------------------------------ *)



(*   Package Begin  ------------------------------------------------------------------------------------------------- *)
BeginPackage["Notation`"];

Unprotect["Notation`*"];
ClearAll @@ Complement[Names["Notation`*"], {"AutoLoadNotationPalette"}];
ClearAll["Notation`Private`*"];


(*   Force the symbol 'Notation' to be in the context Notation`.  - - - - - - - - - - - - - - - - - - - - - - - - -  *)
Notation;



(*   Package Post Begin  -------------------------------------------------------------------------------------------- *)
Begin["`Private`"];


(*   friendlyOff and friendlyOn  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   friendlyOff will turn off a message. friendlyOn will turn that message on only if it was on before the friendlyOff.
*)
SetAttributes[{friendlyOff, friendlyOn, messageStatus}, HoldAll];
messageStatus[MessageName[func_, mesg_String]] :=
   If[Head[MessageName[func, mesg] /. Messages[func]] =!= $Off, $On, $Off, $Off];
friendlyOff[mesg_MessageName] := (mesgWasOn[Hold[mesg]] = messageStatus[mesg] =!= $Off; Off[mesg]; );
friendlyOn[mesg_MessageName] := If[mesgWasOn[Hold[mesg]], On[mesg]; mesgWasOn[Hold[mesg]] = False; ];


(*   silentEvaluate  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   silentEvaluate will evaluate an expression, but report NO error messages, i.e. it will silently evaluate an 
   expression giving a result.
*)
SetAttributes[silentEvaluate, HoldAll];
silentEvaluate[expr_] := Block[{Message}, SetAttributes[Message, HoldFirst]; expr];


(*   Remove the Notation symbol from the system context.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   WRI still has not got around to removing the Notation symbol from the system context. It is not being used for 
   anything. Remove it from the system.
*)
silentEvaluate[Unprotect["System`Notation"]];
silentEvaluate[Remove["System`Notation"]];


(*   Handle overloaded symbols  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   If any of the vital symbols used by Notation are being used in the global context then warn the user and remove them 
   from the global context.
*)
notationsPublicFunctions =
   Map[
      StringJoin["Global`", #1] & ,
      {
         "Action",
         "ClearNotations",
         "CreateNotationRules",
         "InfixNotation",
         "ParsedBoxWrapper",
         "NotationPatternTag",
         "NotationBoxTag",
         "PrintNotationRules",
         "RemoveInfixNotation",
         "RemoveNotation",
         "RemoveNotationRules",
         "RemoveSymbolize",
         "Notation",
         "Symbolize",
         "SymbolizeRootName",
         "WorkingForm"
      }
   ];

Notation::gshadw =
   "The symbol '`1`' has been used in the global context. The Notation package needs the full use of the symbol '`1`' and has therefore removed this symbol from the global context.";

With[
   {overideNames = Intersection[Names["Global`*"], notationsPublicFunctions]},
   If[
      overideNames =!= {},
      (
         (Message[Notation::gshadw, #1] & ) /@ (StringDrop[#1, 7] & ) /@ overideNames;
         Unprotect /@ overideNames;
         ClearAll /@ overideNames;
         Remove /@ overideNames;
         Null
      )
   ]
];


(*   End "`Private`"  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
End[];



(*   Package Usage Statements  -------------------------------------------------------------------------------------- *)
If[
    !ValueQ[Symbolize::usage],
   Symbolize::usage =
      "Symbolize[boxes] forces any box structure matching boxes to be treated internally as a single symbol anywhere it appears in an input expression."
];

If[
    !ValueQ[RemoveSymbolize::usage],
   RemoveSymbolize::usage = "RemoveSymbolize[boxes] removes the symbolization of boxes."
];

If[
    !ValueQ[SymbolizeRootName::usage],
   SymbolizeRootName::usage =
      "SymbolizeRootName is an option for Symbolize specifying the name to be used internally for the symbolized boxes."
];

If[
    !ValueQ[InfixNotation::usage],
   InfixNotation::usage =
      "InfixNotation[infixOp, prefixHead] forces the box structure infixOp to be treated as an infix operator representing the function prefixHead in input and output."
];

If[
    !ValueQ[RemoveInfixNotation::usage],
   RemoveInfixNotation::usage = "RemoveInfixNotation[infixOp, prefixHead] removes the infix operator infixOp"
];

If[
    !ValueQ[Notation::usage],
   Notation::usage =
      "Notation[ExternalBoxes \[DoubleLongLeftRightArrow] InternalExpr] parses any input box structure ExternalBoxes internally as InternalExpr, and formats any expression matching InternalExpr as ExternalBoxes in output. To restrict Notation to only parsing, use Notation[ExternalBoxes \[DoubleLongRightArrow] InternalExpr], and to restrict Notation to only formatting, use Notation[ExternalBoxes \[DoubleLongLeftArrow] InternalExpr]."
];

If[
    !ValueQ[RemoveNotation::usage],
   RemoveNotation::usage =
      "RemoveNotation[ExternalBoxes \[DoubleLongLeftRightArrow] InternalExpr] removes the notation ExternalForm \[DoubleLongLeftRightArrow] InternalForm. To remove only the parsing, use RemoveNotation[ExternalBoxes \[DoubleLongRightArrow] InternalExpr], and to remove only the formatting, use RemoveNotation[ExternalBoxes \[DoubleLongLeftArrow] InternalExpr]."
];

If[
    !ValueQ[ParsedBoxWrapper::usage],
   ParsedBoxWrapper::usage =
      "ParsedBoxWrapper is a wrapper that wraps parsed boxes which come from the TemplateBoxes that are embedded in Notation, Symbolize and InfixNotation statements.  These embedded TemplateBoxes ensure correct parsing and retention of proper styling and grouping information."
];

If[
    !ValueQ[ClearNotations::usage],
   ClearNotations::usage =
      "ClearNotations[] will remove all \"notations\" , \"symbolizations\" and \"infix notations\". It does not destroy any rules for MakeExpression and MakeBoxes. This function will reset the notation handling to a pristine state."
];

If[
    !ValueQ[WorkingForm::usage],
   WorkingForm::usage =
      "WorkingForm is an option of Notation, Symbolize and InfixNotation, which specifies which form the notation will be defined in. Possible forms include StandardForm, TraditionalForm, user defined forms, and Automatic which defaults to the default output format type."
];

If[
    !ValueQ[Action::usage],
   Action::usage =
      "Action is an option of Notation, Symbolize and InfixNotation. It defines what action will be performed with the given notation statement. The possible values are CreateNotationRules, PrintNotationRules and RemoveNotationRules"
];

If[
    !ValueQ[CreateNotationRules::usage],
   CreateNotationRules::usage =
      "CreateNotationRules is a possible value for the option Action which is used in Notation, Symbolize and InfixNotation. If the option Action is set to CreateNotationRules, then a notation statement will enter the given notation into the system."
];

If[
    !ValueQ[RemoveNotationRules::usage],
   RemoveNotationRules::usage =
      "RemoveNotationRules is a possible value for the option Action which is used in Notation, Symbolize and InfixNotation. If the option Action is set to RemoveNotationRules, then a notation statement will remove the given notation from the system."
];

If[
    !ValueQ[PrintNotationRules::usage],
   PrintNotationRules::usage =
      "PrintNotationRules is a possible value for the option Action which is used in Notation, Symbolize and InfixNotation. If the option Action is set to PrintNotationRules, then a notation statement will print out a cell containing the rules defining the given notation."
];

If[
    !ValueQ[AutoLoadNotationPalette::usage],
   AutoLoadNotationPalette::usage =
      "AutoLoadNotationPalette is a boolean variable. If False then the Notation palette will not be loaded when the Notation package is loaded. If the value is undefined or True the Notation palette will be loaded when the Notation package loads. Other package designers can set this variable outside of the Notation package through a statement similar to Notation`AutoLoadNotationPalette = False."
];

If[
    !ValueQ[NotationMakeExpression::usage],
   NotationMakeExpression::usage =
      "NotationMakeExpression is a private version of MakeExpression. The Notation package uses NotationMakeExpression in an attempt to minimally interfere with other functions that use MakeExpression."
];

If[
    !ValueQ[NotationMakeBoxes::usage],
   NotationMakeBoxes::usage =
      "NotationMakeBoxes is a private version of MakeBoxes. The Notation package uses NotationMakeBoxes in an attempt to minimally interfere with other functions that use MakeBoxes."
];

If[
    !ValueQ[NotationBoxTag::usage],
   NotationBoxTag::usage =
      "This is provided for compatibility with previous versions of saved notation files. It has been superseded by the use of the string tag NotationTemplateTag and by ParsedBoxWrapper."
];

If[
    !ValueQ[NotationPatternTag::usage],
   NotationPatternTag::usage =
      "This is provided for compatibility with previous versions of saved notation files. It has been superseded by the use of the string tag NotationPatternTag."
];

If[
    !ValueQ[NotationMadeBoxesTag::usage],
   NotationMadeBoxesTag::usage =
      "This is provided for compatibility with previous versions of saved notation files. It has been superseded by the use of the string tag NotationMadeBoxesTag and by ParsedBoxWrapper"
];

If[
    !ValueQ[AddInputAlias::usage],
   AddInputAlias::usage =
      "AddInputAlias[\"shortForm\"->fullBoxes] adds the alias \[EscapeKey]shortForm\[EscapeKey] for fullBoxes to the aliases in the curret notebook. AddInputAlias[{\"sh1\"->boxes1, \"sh2\"->boxes2, ...},notebook] adds the aliases sh1, sh2,... to the notebook notebook."
];

If[
    !ValueQ[ActiveInputAliases::usage],
   ActiveInputAliases::usage =
      "ActiveInputAliases[] returns a list of all active aliases in the current notebook. ActiveInputAliases[notebook] returns a list of all active aliases in the notebook notebook."
];



(*   Error Message Declarations  ------------------------------------------------------------------------------------ *)


(*   Options for Notation, Symbolize and InfixNotation  ------------------------------------------------------------- *)

(*   Define the options for Notation.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
Options[Notation] = {WorkingForm -> Automatic, Action -> CreateNotationRules};


(*   Define the options for Symbolize.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
Options[Symbolize] = {WorkingForm -> Automatic, Action -> CreateNotationRules, SymbolizeRootName -> ""};


(*   Define the options for InfixNotation.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
Options[InfixNotation] = {WorkingForm -> Automatic, Action -> CreateNotationRules};



(*   Package Begin Private  ----------------------------------------------------------------------------------------- *)
Begin["`Private`"];



(*   adjustedOptions  ----------------------------------------------------------------------------------------------- *)

(*
   adjustedOptions returns the normal options for a function but substitutes on the fly WorkingForm -> Automatic for 
   WorkingForm -> Default Output FormatType. 
*)
adjustedOptions[form_] :=
   Options[form]  /.
      HoldPattern[(Rule | RuleDelayed)[WorkingForm, Automatic]]  :>
         (WorkingForm -> processWorkingForm[AbsoluteCurrentValue[InputNotebook[], {CommonDefaultFormatTypes, "Output"}]]);

processWorkingForm[InputForm] := StandardForm;
processWorkingForm[OutputForm] := StandardForm;
processWorkingForm[form_] := form;




(* ------------------------------------------------------------------------------------------------------------------ *)
(*   General Error Handling  ---------------------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------------------------------------------------ *)



(*   General error routines  ---------------------------------------------------------------------------------------- *)

(*
   Here are some basic functions which the error handling routines use. 
   heldLength just gives the length of an expression without evaluating anything.
   isNot will return True for anything that does not match the pattern.
   headIsNot will return True for anything whose head does not match the pattern.
*)
SetAttributes[{heldLength, headIsNot, isNot}, HoldAll];
heldLength[expr_] := Length[Unevaluated[expr]];
isNot[pattern_] := Function[expr,  !MatchQ[Unevaluated[expr], pattern], HoldAll];
headIsNot[pattern_] := Function[testHead,  !MatchQ[Head[Unevaluated[testHead]], pattern], HoldAll];

General::badarg = "`1` expected at position `2` in `3`";

characterQ[a_String] := StringLength[a] == 1;
characterQ[other___] := False;



(*   Frames shaded by depth  ---------------------------------------------------------------------------------------- *)

(*   structuralBoxes  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   structuralBoxes matches boxes which are structural in nature and affect the parsing of expressions.
*)
structuralBoxes =
   Alternatives[
      FractionBox,
      GridBox,
      InterpretationBox,
      RadicalBox,
      RowBox,
      SqrtBox,
      SubscriptBox,
      SuperscriptBox,
      SubsuperscriptBox,
      TagBox,
      TemplateBox,
      UnderscriptBox,
      OverscriptBox,
      UnderoverscriptBox
   ];


(*   colorizeStructuralBoxes  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   colorizeStructuralBoxes will shade the background of a box structure according to its structure. In this way the 
   user can visually observe the structure of an expression.
*)
colorizeStructuralBoxes[(a:structuralBoxes)[args__], depth_] :=
   ((StyleBox[#1, Background -> Hue[0, 1 - E^(-0.1579*depth), 1], FontColor -> Hue[0, 1 - E^(-0.1579*depth), 1]] & )[FrameBox[StyleBox[#1, FontColor -> RGBColor[0, 0, 0]]]] & )[
      a @@ (colorizeStructuralBoxes[#1, depth + 1] & ) /@ {args}
   ];
colorizeStructuralBoxes[(a_)[args__], depth_] := a @@ (colorizeStructuralBoxes[#1, depth] & ) /@ {args};
colorizeStructuralBoxes[a_, depth_] := a;



(*   parsableQ  ----------------------------------------------------------------------------------------------------- *)

(*
   parsableQ will determine whether a given box structure or string is parsable under the given working form. If it is 
   not, it will report the error message given by the first non-parsable object and return False. silentParsableQ 
   performs exactly the same function, but will not report ANY error messages.
*)
parsableQ[","] = True;
parsableQ[boxexpr_] :=
   Head[ToExpression[boxexpr, WorkingForm /. adjustedOptions[Notation], HoldComplete]] === HoldComplete;
parsableQ[boxexpr_, opts___] :=
   Head[ToExpression[boxexpr, WorkingForm /. {opts} /. adjustedOptions[Notation], HoldComplete]] === HoldComplete;

notParsableQ[boxexpr___] :=  !parsableQ[boxexpr];
silentNotParsableQ[boxexpr___] :=  !silentEvaluate[parsableQ[boxexpr]];
silentParsableQ[boxexpr___] := silentEvaluate[parsableQ[boxexpr]];



(*   SilentCheck  --------------------------------------------------------------------------------------------------- *)

(*
   Silently checking a message to see if it fails is not trivial. This should be built into Mathematica. The following 
   code, which circumvents this oversight, was originally written by Todd Gayley and modified by Robby Villegas.
*)
Attributes[SilentCheck] = HoldAll;
SilentCheck[expr_, failexpr_] :=
   Module[
      {returnValue},
      (
         Unprotect[Message];
         HoldPattern[mesg:Message[___]] := Block[{$Messages = {}}, mesg] /; $Messages =!= {};
         returnValue = Check[expr, failexpr];
         HoldPattern[mesg:Message[___]] =. ;
         Protect[Message];
         returnValue
      )
   ];




(* ------------------------------------------------------------------------------------------------------------------ *)
(*   Form Handling  ------------------------------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------------------------------------------------ *)



(*   Define notation's versions of MakeExpression and MakeBoxes  ---------------------------------------------------- *)

(*   NotationMakeExpression  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   We create NotationMakeExpression so the Notation package minimally interferes with MakeExpression.
*)
MakeExpression[boxes_, form_] := With[{expr = NotationMakeExpression[boxes, form]}, expr /; Head[expr] === HoldComplete];


(*   NotationMakeBoxes  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   We create NotationMakeBoxes so the Notation package minimally interferes with MakeBoxes.
*)
SetAttributes[NotationMakeBoxes, HoldAllComplete];

MakeBoxes[expr_, form_] := With[{boxes = NotationMakeBoxes[expr, form]}, boxes /; Head[boxes] =!= NotationMakeBoxes];



(*   ParsedBoxWrapper  ---------------------------------------------------------------------------------------------- *)

(*
   These define how ParsedBoxWrapper is handled both in input and output. Basically it is just the parsed form of a 
   TemplateBox wrapper with tag NotationTemplateTag. It is used to allow Notation and Symbolize to grab style information 
   etc.
*)
wasProteced = Unprotect[TemplateBox];
TagSetDelayed[
   TemplateBox,
   MakeExpression[TemplateBox[{boxes_}, "NotationTemplateTag", opts___], anyForm_],
   HoldComplete[ParsedBoxWrapper[boxes]]
];
Protect[Evaluate[wasProteced]];

NotationMakeBoxes[HoldPattern[ParsedBoxWrapper][boxes__], anyForm_] := TemplateBox[{boxes}, "NotationTemplateTag"];

SetAttributes[ParsedBoxWrapper, HoldAll];


(*   Compatibility with old Notation versions  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   For compatibility with all old Notations, recognize the tag NotationBoxTag. 
*)
NotationMakeExpression[TagBox[boxes_, NotationBoxTag, opts___], anyForm_] := HoldComplete[ParsedBoxWrapper[boxes]];
NotationMakeExpression[TagBox[boxes_, "NotationTemplateTag", opts___], anyForm_] :=
   HoldComplete[ParsedBoxWrapper[boxes]];

NotationBoxTag[args__] := ParsedBoxWrapper[args];

SetAttributes[NotationBoxTag, HoldAll];



(*   stripParsedBoxWrapper  ----------------------------------------------------------------------------------------- *)

(*
   stripParsedBoxWrapper will remove any ParsedBoxWrapper found in the box expression. ParsedBoxWrapper is used to 
   allow Notation to grab boxes before the style information and other things are stripped out.
*)
stripParsedBoxWrapper[any_] := any //. HoldPattern[ParsedBoxWrapper][stringPatternBoxes_] :> stringPatternBoxes;



(*   identityForm for output  --------------------------------------------------------------------------------------- *)

(*
   Boxes wrapped with this wrapper will appear as DisplayForm would show them, i.e. 
   identityForm[SuperscriptBox["x","2"]] would appear as -=< Ommitted Inline Cell >=-.
*)
identityForm /: NotationMakeBoxes[identityForm[any___], form_] := any;

SetAttributes[identityForm, HoldAll];



(*   TransformLegacySyntax  ----------------------------------------------------------------------------------------- *)

(*
   For compatibility with all old notations transform the old TagBoxes into the new TemplateBoxes.
*)
TransformLegacySyntax[contents__] :=
   Apply[
      Sequence,
      {contents}  //.
         {
            TagBox[stringPatternBoxes_, NotationPatternTag, opts___]  :>
               TemplateBox[{stringPatternBoxes}, "NotationPatternTag", opts],
            TagBox[stringPatternBoxes_, "NotationPatternTag", opts___]  :>
               TemplateBox[{stringPatternBoxes}, "NotationPatternTag", opts],
            TagBox[stringPatternBoxes_, "NotationTemplateTag", opts___]  :>
               TemplateBox[{stringPatternBoxes}, "NotationTemplateTag", opts],
            TagBox[stringPatternBoxes_, NotationMadeBoxesTag, opts___]  :>
               TemplateBox[{stringPatternBoxes}, "NotationMadeBoxesTag", opts],
            TagBox[stringPatternBoxes_, "NotationMadeBoxesTag", opts___]  :>
               TemplateBox[{stringPatternBoxes}, "NotationMadeBoxesTag", opts]
         }
   ];




(* ------------------------------------------------------------------------------------------------------------------ *)
(*   Character Utility Functions  ----------------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------------------------------------------------ *)



(*   whiteSpaceQ  --------------------------------------------------------------------------------------------------- *)

(*
   whiteSpaceQ tests to see if the given string can be considered white space.
*)
whiteSpaceQ[string_String] :=
   SameQ[
      DeleteCases[
         Characters[string],
         Alternatives[
            "\t",
            "\n",
            " ",
            "\[InvisibleSpace]",
            "\[VeryThinSpace]",
            "\[ThinSpace]",
            "\[MediumSpace]",
            "\[ThickSpace]",
            "\[NegativeVeryThinSpace]",
            "\[NegativeThinSpace]",
            "\[IndentingNewLine]",
            "\[NegativeMediumSpace]",
            "\[NegativeThickSpace]",
            "\r",
            "\[NoBreak]",
            "\[NonBreakingSpace]",
            "\[Continuation]",
            "\[SpaceIndicator]",
            "\[RoundSpaceIndicator]",
            "\[AlignmentMarker]",
            "",
            "\[LineSeparator]",
            "\[ParagraphSeparator]"
         ]
      ],
      {}
   ];

whiteSpaceQ[other___] := False;



(*   notWhiteSpaceQ  ------------------------------------------------------------------------------------------------ *)

(*
   notWhiteSpaceQ tests to see if the given string is not considered white space.
*)
notWhiteSpaceQ[any_] :=  !whiteSpaceQ[any];



(*   Define setInternalCharacterInformation  ------------------------------------------------------------------------ *)

(*   internalCharacterInformation  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   This creates the internalCharacterInformation function which gives information on the complete list of characters 
   from the UnicodeCharacters.tr file. These are needed since Mathematica does not currently have functions for testing 
   things like OperatorQ, PrefixQ, etc.
*)
setInternalCharacterInformation[
   {
      theCode_,
      theCharacter_,
      shortforms_,
      theFixity_,
      thePrecedence_,
      theGrouping_,
      theRightSpacing_,
      theLeftSpacing_,
      other___
   }
] :=
   (internalCharacterInformation[ToExpression[StringJoin["\"", theCharacter, "\""]]] =
      {
         theCode,
         theFixity,
         thePrecedence,
         theGrouping,
         theRightSpacing,
         theLeftSpacing,
         StringDrop[StringDrop[theCharacter, 2], -1]
      });

setInternalCharacterInformation[{theCode_, theCharacter_, shortforms_, theFixity_}] :=
   (internalCharacterInformation[ToExpression[StringJoin["\"", theCharacter, "\""]]] =
      {theCode, theFixity, StringDrop[StringDrop[theCharacter, 2], -1]});


(*
   We need to handle the cases when the characters are not actually characters at all.
*)
setInternalCharacterInformation[
   {
      theCode_,
      "\\[]",
      shortforms_,
      theFixity_,
      thePrecedence_,
      theGrouping_,
      theRightSpacing_,
      theLeftSpacing_,
      other___
   }
] :=
   $DoNothing;

setInternalCharacterInformation[{theCode_, "\\[]", shortforms_, theFixity_}] := $DoNothing;


(*   Error Checking for setInternalCharacterInformation.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   The first argument of setInternalCharacterInformation must be a list.
*)
error:setInternalCharacterInformation[(notList_)?(headIsNot[List]), ___] :=
   $Failed /; Message[setInternalCharacterInformation::list, HoldForm[error], 1];


(*
   setInternalCharacterInformation expects only one argument.
*)
error:setInternalCharacterInformation[___] :=
   With[
      {num = heldLength[error]},
      Condition[
         $Failed,
         num != 1 && Message[setInternalCharacterInformation::argx, HoldForm[setInternalCharacterInformation], num, 1]
      ]
   ];



(*   Load internalCharacterInformation  ----------------------------------------------------------------------------- *)

(*
   This creates the internalCharacterInformation table which gives information on the complete list of characters from 
   the UnicodeCharacters.tr file.
*)

(*   Missing file UnicodeCharacters.tr.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
$FEUnicodeCharactersFile =
   FrontEnd`FileName[
      {FrontEnd`$InstallationDirectory, "SystemFiles", "FrontEnd", "TextResources"},
      "UnicodeCharacters.tr"
   ];

UnicodeCharactersContentsAsString = MathLink`CallFrontEnd[MLFS`Get[$FEUnicodeCharactersFile]];


(*
   In some installed versions of Mathematica the file UnicodeCharacters.tr is missing. If so report this error.
*)
unicodeCharacters::missing =
   "It appears that the file 'UnicodeCharacters.tr' has not been included in your installation of Mathematica. The file 'UnicodeCharacters.tr' is necessary for Notation to determine the precedences of characters. Unfortunately Notation will not run without this file. This sometimes occurs in installations inside WRI.";

If[
    !StringQ[UnicodeCharactersContentsAsString] || UnicodeCharactersContentsAsString == "",
   Message[unicodeCharacters::missing]
];


(*   Missing characters in UnicodeCharacters.tr.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   Warning: 4.0 Beta 2 has many characters which are not properly incorporated into UnicodeCharacters.tr. Things like 
   Klingon characters, etc. (Who added these ?!?)
*)
UnicodeCharactersContentsAsStream = StringToStream[UnicodeCharactersContentsAsString];

silentEvaluate[
   Union[
      Map[
         setInternalCharacterInformation,
         ReadList[
            UnicodeCharactersContentsAsStream,
            Word,
            RecordLists -> True,
            WordSeparators -> {FromCharacterCode[9]},
            RecordSeparators -> {FromCharacterCode[13], FromCharacterCode[10]}
         ]
      ]
   ]
];

Close[UnicodeCharactersContentsAsStream];


(*   Add standard characters to internalCharacterInformation.  - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   This adds the uppercase letters, lowercase letters, and digits to internalCharacterInformation.
*)
Table[
   internalCharacterInformation[FromCharacterCode[i]] = {ToString[i], "Letter", StringJoin["Raw", FromCharacterCode[i]]},
   {i, 65, 90}
];
Table[
   internalCharacterInformation[FromCharacterCode[i]] = {ToString[i], "Letter", StringJoin["Raw", FromCharacterCode[i]]},
   {i, 97, 122}
];
Table[
   internalCharacterInformation[FromCharacterCode[i]] = {ToString[i], "Digit", StringJoin["Raw", FromCharacterCode[i]]},
   {i, 48, 57}
];



(*   Define characterInformation  ----------------------------------------------------------------------------------- *)

(*   characterInformation  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   characterInformation returns the information available on a character. 
*)
characterInformation[(char_String)?characterQ] := characterInformationAux[internalCharacterInformation[char]];
characterInformation[char_String, field_Symbol] := field /. characterInformation[char];
characterInformationAux[
   {theCode_, theFixity_, thePrecedence_, theGrouping_, theRightSpacing_, theLeftSpacing_, theCharacterFullName_}
] :=
   {
      CharacterCode -> theCode,
      CharacterFixity -> theFixity,
      CharacterPrecedence -> thePrecedence,
      CharacterGrouping -> theGrouping,
      CharacterRightSpacing -> theRightSpacing,
      CharacterLeftSpacing -> theLeftSpacing,
      CharacterFullName -> theCharacterFullName
   };
characterInformationAux[{theCode_, theFixity_, theCharacterFullName_}] :=
   {CharacterCode -> theCode, CharacterFixity -> theFixity, CharacterFullName -> theCharacterFullName};


(*   Error checking for characterInformation  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   The first argument of characterInformation must be a string.
*)
error:characterInformation[(notString_)?(headIsNot[String]), ___] :=
   $Failed /; Message[characterInformation::string, 1, HoldForm[error]];


(*
   The second argument of characterInformation must be a symbol.
*)
error:characterInformation[_, (notSymb_)?(headIsNot[Symbol]), ___] :=
   $Failed /; Message[characterInformation::sym, notSymb, 2];


(*
   characterInformation expects one or two arguments.
*)
error:characterInformation[___] :=
   With[
      {num = heldLength[error]},
      Condition[
         $Failed,
         And[
            num < 1 || num > 2,
            Message[characterInformation::argt, HoldForm[characterInformation], heldLength[error], 1, 2]
         ]
      ]
   ];




(* ------------------------------------------------------------------------------------------------------------------ *)
(*   Box and Character Querying Functions  -------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------------------------------------------------ *)



(*   Definition of isolatingBoxes and nonIsolatingBoxes and mutableIsolatingBoxes  ---------------------------------- *)

(*
   isolatingBoxes are the box structures that isolate the grouping nature and precedence nature of the internals of the 
   box.
*)
isolatingBoxes = ButtonBox | FormBox | FrameBox | GridBox | RowBox | RadicalBox | SqrtBox;


(*
   nonIsolatingBoxes are the box structures that do not isolate the grouping nature and precedence nature of the 
   internals of the box.
*)
nonIsolatingBoxes =
   Alternatives[
      AdjustmentBox,
      ErrorBox,
      FractionBox,
      StyleBox,
      SubscriptBox,
      SuperscriptBox,
      SubsuperscriptBox,
      UnderscriptBox,
      OverscriptBox,
      UnderoverscriptBox
   ];


(*
   mutableIsolatingBoxes are the box structures that possibly change the grouping nature and precedence nature of the 
   internals of the box.
*)
mutableIsolatingBoxes = TagBox | TemplateBox | InterpretationBox;

allTheBoxes = Flatten[nonIsolatingBoxes | isolatingBoxes | mutableIsolatingBoxes];



(*   boxStructureQ  ------------------------------------------------------------------------------------------------- *)

(*   boxStructureQ  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   boxStructureQ tests an expression to see if its head is a known box.
*)
boxStructureQ[nonIsolatingBoxes[args___]] := True;
boxStructureQ[isolatingBoxes[args___]] := True;
boxStructureQ[mutableIsolatingBoxes[args___]] := True;
boxStructureQ[_] := False;


(*   Error checking for boxStructureQ  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   boxStructureQ expects only one argument.
*)
error:boxStructureQ[___] :=
   With[{num = heldLength[error]}, $Failed /; num != 1 && Message[boxStructureQ::argx, HoldForm[boxStructureQ], num, 1]];



(*   nonIsolatingBoxStructureQ  ------------------------------------------------------------------------------------- *)

(*   nonIsolatingBoxStructureQ  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   nonIsolatingBoxStructureQ will give True if the box structure does not isolate the grouping nature and precedence 
   nature of the internals of the box.
*)
nonIsolatingBoxStructureQ[nonIsolatingBoxes[args___]] := True;
nonIsolatingBoxStructureQ[other_] := False;


(*   Error checking for nonIsolatingBoxStructureQ  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   nonIsolatingBoxStructureQ expects only one argument.
*)
error:nonIsolatingBoxStructureQ[___] :=
   With[
      {num = heldLength[error]},
      $Failed /; num != 1 && Message[nonIsolatingBoxStructureQ::argx, HoldForm[nonIsolatingBoxStructureQ], num, 1]
   ];



(*   mutableIsolatingBoxStructureQ  --------------------------------------------------------------------------------- *)

(*   mutableIsolatingBoxStructureQ  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   mutableIsolatingBoxStructureQ will give True if the box structure does not isolate the grouping nature and 
   precedence nature of the internals of the box.
*)
mutableIsolatingBoxStructureQ[mutableIsolatingBoxes[args___]] := True;
mutableIsolatingBoxStructureQ[other_] := False;


(*   Error checking for mutableIsolatingBoxStructureQ  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   mutableIsolatingBoxStructureQ expects only one argument.
*)
error:mutableIsolatingBoxStructureQ[___] :=
   With[
      {num = heldLength[error]},
      Condition[
         $Failed,
         num != 1 && Message[mutableIsolatingBoxStructureQ::argx, HoldForm[mutableIsolatingBoxStructureQ], num, 1]
      ]
   ];



(*   effectiveBoxes  ------------------------------------------------------------------------------------------------ *)

(*   effectiveBoxes  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   effectiveBoxes will give the string internal to the given box structure according to how the boxes "act". 
   E.g., SubscriptBox["+","R"] will "act" like "+".
*)
effectiveBoxes[string_String] := string;
effectiveBoxes[nonIsolatingBoxes[actsLike_, ___]] := effectiveBoxes[actsLike];
effectiveBoxes[mutableIsolatingBoxes[___, SyntaxForm -> actsLike_, ___]] := effectiveBoxes[actsLike];
effectiveBoxes[mutableIsolatingBoxes[actsLike_, ___]] := effectiveBoxes[actsLike];
effectiveBoxes[isolatingBoxes[___]] := "GenericSymbol";
effectiveBoxes[other_] := "GenericSymbol";


(*   Error checking for effectiveBoxes  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   The first argument of effectiveBoxes must be a box structure.
*)
error:effectiveBoxes[(badArg_)?(isNot[_?boxStructureQ]), ___] :=
   $Failed /; Message[effectiveBoxes::badarg, 1, HoldForm[error]];

effectiveBoxes::badarg = "Box structure or String expected at position `1` in `2`";


(*
   effectiveBoxes expects only one argument.
*)
error:effectiveBoxes[___] :=
   With[
      {num = heldLength[error]},
      $Failed /; num != 1 && Message[effectiveBoxes::argx, HoldForm[effectiveBoxes], num, 1]
   ];



(*   prefixOperatorQ  ----------------------------------------------------------------------------------------------- *)

(*
   This Boolean function determines if the given expression is normally treated as a PrefixOperator.
   If the expression is a box structure, then look inside the structure to see how it acts.
*)
prefixOperatorQ[(struct_)?nonIsolatingBoxStructureQ] := prefixOperatorQ[effectiveBoxes[struct]];
prefixOperatorQ[(struct_)?mutableIsolatingBoxStructureQ] := prefixOperatorQ[effectiveBoxes[struct]];


(*   prefixOperatorQ: exceptions  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
prefixOperatorQ["\[PartialD]"] = True;
prefixOperatorQ["\[Integral]"] = True;
prefixOperatorQ["\[ContourIntegral]"] = True;
prefixOperatorQ["\[CounterClockwiseContourIntegral]"] = True;
prefixOperatorQ["\[ClockwiseContourIntegral]"] = True;
prefixOperatorQ["\[DoubleContourIntegral]"] = True;
prefixOperatorQ["\[Sum]"] = True;
prefixOperatorQ["\[Product]"] = True;


(*   prefixOperatorQ: the general case  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
prefixOperatorQ["!"] = True;
prefixOperatorQ[(char_)?characterQ] := "Prefix" == characterInformation[char, CharacterFixity];
prefixOperatorQ[other_] := False;


(*   Error checking for prefixOperatorQ  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   prefixOperatorQ expects only one argument.
*)
error:prefixOperatorQ[___] :=
   With[
      {num = heldLength[error]},
      $Failed /; num != 1 && Message[prefixOperatorQ::argx, HoldForm[prefixOperatorQ], num, 1]
   ];



(*   infixOperatorQ  ------------------------------------------------------------------------------------------------ *)

(*
   This Boolean function determines if the given expression could normally be treated as an InfixOperator.
   If the expression is a box structure, then look inside the structure to see how it acts.
*)
infixOperatorQ[(struct_)?nonIsolatingBoxStructureQ] := infixOperatorQ[effectiveBoxes[struct]];
infixOperatorQ[(struct_)?mutableIsolatingBoxStructureQ] := infixOperatorQ[effectiveBoxes[struct]];


(*   infixOperatorQ: exceptions  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
infixOperatorQ["+"] = True;
infixOperatorQ["*"] = True;
infixOperatorQ["^"] = True;
infixOperatorQ["."] = True;
infixOperatorQ["-"] = True;
infixOperatorQ["->"] = True;
infixOperatorQ[":>"] = True;
infixOperatorQ["="] = True;
infixOperatorQ[":="] = True;
infixOperatorQ["^="] = True;
infixOperatorQ["^:="] = True;
infixOperatorQ["+="] = True;
infixOperatorQ["-="] = True;
infixOperatorQ["*="] = True;
infixOperatorQ["/="] = True;
infixOperatorQ["/."] = True;
infixOperatorQ["//."] = True;
infixOperatorQ["//"] = True;
infixOperatorQ["/;"] = True;
infixOperatorQ["/"] = True;
infixOperatorQ[":"] = True;
infixOperatorQ[";"] = True;
infixOperatorQ["<="] = True;
infixOperatorQ["<"] = True;
infixOperatorQ[">"] = True;
infixOperatorQ[">="] = True;
infixOperatorQ["=="] = True;
infixOperatorQ["==="] = True;
infixOperatorQ["!"] = True;
infixOperatorQ["!="] = True;
infixOperatorQ["=!="] = True;
infixOperatorQ["&&"] = True;
infixOperatorQ["||"] = True;
infixOperatorQ["?"] = True;
infixOperatorQ["@@"] = True;
infixOperatorQ["@"] = True;
infixOperatorQ["/@"] = True;
infixOperatorQ["||"] = True;
infixOperatorQ["|"] = True;


(*   infixOperatorQ: the general case  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
infixOperatorQ[(char_)?characterQ] := "Infix" == characterInformation[char, CharacterFixity];
infixOperatorQ[other_] := False;


(*   Error checking for infixOperatorQ  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   infixOperatorQ expects only one argument.
*)
error:infixOperatorQ[___] :=
   With[
      {num = heldLength[error]},
      $Failed /; num != 1 && Message[infixOperatorQ::argx, HoldForm[infixOperatorQ], num, 1]
   ];



(*   postfixOperatorQ  ---------------------------------------------------------------------------------------------- *)

(*
   This Boolean function determines if the given string could normally be treated as a PostfixOperator.
   If the expression is a box structure, then look inside the structure to see how it acts.
*)
postfixOperatorQ[(struct_)?nonIsolatingBoxStructureQ] := postfixOperatorQ[effectiveBoxes[struct]];
postfixOperatorQ[(struct_)?mutableIsolatingBoxStructureQ] := postfixOperatorQ[effectiveBoxes[struct]];


(*   postfixOperatorQ: exceptions  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
postfixOperatorQ[";"] = True;
postfixOperatorQ["&"] = True;
postfixOperatorQ["!!"] = True;
postfixOperatorQ["!"] = True;
postfixOperatorQ["'"] = True;
postfixOperatorQ["--"] = True;
postfixOperatorQ["++"] = True;
postfixOperatorQ["=."] = True;
postfixOperatorQ[".."] = True;
postfixOperatorQ["..."] = True;
postfixOperatorQ[other_] = False;


(*   postfixOperatorQ: the general case  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
postfixOperatorQ[(char_)?characterQ] := "Postfix" == characterInformation[char, CharacterFixity];
postfixOperatorQ[other_] := False;


(*   Error checking for postfixOperatorQ  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   postfixOperatorQ expects only one argument.
*)
error:postfixOperatorQ[___] :=
   With[
      {num = heldLength[error]},
      $Failed /; num != 1 && Message[postfixOperatorQ::argx, HoldForm[postfixOperatorQ], num, 1]
   ];



(*   operatorQ  ----------------------------------------------------------------------------------------------------- *)

(*
   This Boolean function determines if the given string is normally treated as an Operator.
*)
operatorQ[boxes_] := prefixOperatorQ[boxes] || infixOperatorQ[boxes] || postfixOperatorQ[boxes];


(*   Error checking for operatorQ  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   operatorQ expects only one argument.
*)
error:operatorQ[___] :=
   With[{num = heldLength[error]}, $Failed /; num != 1 && Message[operatorQ::argx, HoldForm[operatorQ], num, 1]];



(*   delimiterQ  ---------------------------------------------------------------------------------------------------- *)

(*
   This Boolean function determines if the given string could normally be treated as a Delimiter.
   If the expression is a box structure, then look inside the structure to see how it acts.
*)
delimiterQ[(struct_)?nonIsolatingBoxStructureQ] := delimiterQ[effectiveBoxes[struct]];
delimiterQ[(struct_)?mutableIsolatingBoxStructureQ] := delimiterQ[effectiveBoxes[struct]];


(*   delimiterQ: exceptions  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
delimiterQ["\[InvisibleComma]"] = True;
delimiterQ[","] = True;
delimiterQ["["] = True;
delimiterQ["]"] = True;
delimiterQ["("] = True;
delimiterQ[")"] = True;
delimiterQ["{"] = True;
delimiterQ["}"] = True;


(*   delimiterQ: the general case  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
delimiterQ[(char_)?characterQ] :=
   With[
      {theFixity = characterInformation[char, CharacterFixity]},
      theFixity == "Open" || theFixity == "InfixOpen" || theFixity == "Close"
   ];
delimiterQ[other_] := False;


(*   Error checking for delimiterQ  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   delimiterQ expects only one argument.
*)
error:delimiterQ[___] :=
   With[{num = heldLength[error]}, $Failed /; num != 1 && Message[delimiterQ::argx, HoldForm[delimiterQ], num, 1]];



(*   TokenToSymbol Functions  --------------------------------------------------------------------------------------- *)

(*
   These functions will take an operator string token that is not an exception and give the operator symbol that this 
   token parses to.
   The Check is in case the symbol is not recognized, in which case the precedence is reported as 'Times'.
*)
infixOperatorTokenToSymbol[infixop_String] :=
   silentEvaluate[
      Check[
         Module[
            {expr = ToHeldExpression[StringJoin["Notation`Private`a", infixop, "Notation`Private`b"]]},
            If[Depth[expr] == 3, expr[[1,0]], Times]
         ],
         Times
      ]
   ];

prefixOperatorTokenToSymbol["\[PartialD]"] = D;
prefixOperatorTokenToSymbol["\[Integral]"] = Integrate;
prefixOperatorTokenToSymbol["\[ContourIntegral]"] = Integrate;
prefixOperatorTokenToSymbol["\[CounterClockwiseContourIntegral]"] = Integrate;
prefixOperatorTokenToSymbol["\[ClockwiseContourIntegral]"] = Integrate;
prefixOperatorTokenToSymbol["\[DoubleContourIntegral]"] = Integrate;
prefixOperatorTokenToSymbol[prefixOp_String] :=
   silentEvaluate[
      Check[
         Module[
            {expr = ToHeldExpression[StringJoin[prefixOp, "Notation`Private`b"]]},
            If[Depth[expr] == 3, expr[[1,0]], Times]
         ],
         Times
      ]
   ];

postfixOperatorTokenToSymbol["'"] = Derivative;
postfixOperatorTokenToSymbol[postfixOp_String] :=
   silentEvaluate[
      Check[
         Module[
            {expr = ToHeldExpression[StringJoin["Notation`Private`b", postfixOp]]},
            If[Depth[expr] == 3, expr[[1,0]], Times]
         ],
         Times
      ]
   ];



(*   Error checking for Token to Symbol Functions  ------------------------------------------------------------------ *)

(*
   The first argument of infixOperatorTokenToSymbol must be a String.
*)
error:infixOperatorTokenToSymbol[(notString_)?(headIsNot[String]), ___] :=
   $Failed /; Message[infixOperatorTokenToSymbol::string, 1, HoldForm[error]];


(*
   The first argument of prefixOperatorTokenToSymbol must be a String.
*)
error:prefixOperatorTokenToSymbol[(notString_)?(headIsNot[String]), ___] :=
   $Failed /; Message[prefixOperatorTokenToSymbol::string, 1, HoldForm[error]];


(*
   The first argument of postfixOperatorTokenToSymbol must be a String.
*)
error:postfixOperatorTokenToSymbol[(notString_)?(headIsNot[String]), ___] :=
   $Failed /; Message[postfixOperatorTokenToSymbol::string, 1, HoldForm[error]];


(*
   infixOperatorTokenToSymbol expects only one argument.
*)
error:infixOperatorTokenToSymbol[___] :=
   With[
      {num = heldLength[error]},
      $Failed /; num != 1 && Message[infixOperatorTokenToSymbol::argx, HoldForm[infixOperatorTokenToSymbol], num, 1]
   ];


(*
   prefixOperatorTokenToSymbol expects only one argument.
*)
error:prefixOperatorTokenToSymbol[___] :=
   With[
      {num = heldLength[error]},
      $Failed /; num != 1 && Message[prefixOperatorTokenToSymbol::argx, HoldForm[prefixOperatorTokenToSymbol], num, 1]
   ];


(*
   postfixOperatorTokenToSymbol expects only one argument.
*)
error:postfixOperatorTokenToSymbol[___] :=
   With[
      {num = heldLength[error]},
      $Failed /; num != 1 && Message[postfixOperatorTokenToSymbol::argx, HoldForm[postfixOperatorTokenToSymbol], num, 1]
   ];



(*   effectiveOperator  --------------------------------------------------------------------------------------------- *)

(*
   effectiveOperator gives the effective operator by which output boxes need to be grouped.
*)

(*   effectiveOperator : exceptions  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
effectiveOperator[(struct_)?boxStructureQ] := effectiveOperator[effectiveBoxes[struct]];

effectiveOperator["\[PartialD]"] = D;

effectiveOperator["\[Integral]"] = Integrate;
effectiveOperator["\[ContourIntegral]"] = Integrate;
effectiveOperator["\[ClockwiseContourIntegral]"] = Integrate;
effectiveOperator["\[CounterClockwiseContourIntegral]"] = Integrate;
effectiveOperator["\[ClockwiseContourIntegral]"] = Integrate;


(*   effectiveOperator : the general case  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
effectiveOperator[(op_)?prefixOperatorQ] := prefixOperatorTokenToSymbol[op];
effectiveOperator[(op_)?infixOperatorQ] := infixOperatorTokenToSymbol[op];
effectiveOperator[(op_)?postfixOperatorQ] := postfixOperatorTokenToSymbol[op];
effectiveOperator[other_] := Times;


(*   Error checking for effectiveOperator  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   effectiveOperator expects only one argument.
*)
error:effectiveOperator[___] :=
   With[
      {num = heldLength[error]},
      $Failed /; num != 1 && Message[effectiveOperator::argx, HoldForm[effectiveOperator], num, 1]
   ];




(* ------------------------------------------------------------------------------------------------------------------ *)
(*   Utility Functions  --------------------------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------------------------------------------------ *)



(*   myHold, releaseMyHold, flattenAllMyHold & toMyHeldExpression  -------------------------------------------------- *)

(*
   myHold and releaseMyHold are exactly the same as the standard Hold and releaseHold, except they appear in a 
   different context so they will not stomp on other uses of Hold or held expressions.
*)
SetAttributes[{myHold, releaseMyHold}, HoldAllComplete];

releaseMyHold[expr___] := Evaluate @@ (HoldComplete[expr] //. myHold[term___] -> term);

flattenAllMyHold[expr_] := myHold @@ (HoldComplete[expr] //. myHold[term___] -> term);

toMyHeldExpression[args___] := myHold @@ MakeExpression[args];



(*   removePatternsAndBlanks  --------------------------------------------------------------------------------------- *)

(*
   This removes all pattern wrappers, optional wrappers, pattern tests and conditions, leaving just a pattern variable 
   if possible.
*)
removePatternsAndBlanks[expr_] :=
   Apply[
      Evaluate,
      myHold[expr]  //.
         {
            (holdHead:Optional)[a_, ___] -> a,
            (holdHead:Pattern)[a_, ___] -> a,
            (holdHead:PatternTest)[a_, ___] -> a,
            (holdHead:Condition)[a_, ___] -> a
         }
   ];


(*   Error Checking for removePatternsAndBlanks.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   removePatternsAndBlanks expects only one argument.
*)
error:removePatternsAndBlanks[___] :=
   With[
      {num = heldLength[error]},
      $Failed /; num != 1 && Message[removePatternsAndBlanks::argx, HoldForm[removePatternsAndBlanks], num, 1]
   ];



(*   convertPatterns  ----------------------------------------------------------------------------------------------- *)

(*   convertPatterns  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   convertPatterns will remove all PatternTests, Conditions and Optionals, as well as Heads, leaving just named 
   Patterns. It is useful for putting an expression into a form upon which further manipulations of the pattern variables may 
   be performed.
*)
convertPatterns[expr_] :=
   myHold[expr]  //.
      {
         (holdHead:Optional)[a_, ___] -> a,
         (holdHead:Pattern)[a_, _Blank] -> singleBlank[a],
         (holdHead:Pattern)[a_, _BlankSequence] -> doubleBlank[a],
         (holdHead:Pattern)[a_, _BlankNullSequence] -> tripleBlank[a],
         (holdHead:Pattern)[a_, _] -> complexPattern[a],
         (holdHead:PatternTest)[a_, ___] -> a,
         (holdHead:Condition)[a_, ___] -> a,
         _Blank -> singleBlank[],
         _BlankSequence -> doubleBlank[],
         _BlankNullSequence -> tripleBlank[]
      };

SetAttributes[{singleBlank, doubleBlank, tripleBlank, complexPattern}, HoldAll];


(*   Error Checking for convertPatterns.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   convertPatterns expects only one argument.
*)
error:convertPatterns[___] :=
   With[
      {num = heldLength[error]},
      $Failed /; num != 1 && Message[convertPatterns::argx, HoldForm[convertPatterns], num, 1]
   ];



(*   cleanBoxes & tidyBoxes  ---------------------------------------------------------------------------------------- *)

(*
   These two functions just clean up box structures by flattening single RowBoxes, removing nested RowBoxes and 
   removing white space, where applicable.
*)

(*   cleanBoxes  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
cleanBoxes[boxes_] :=
   Identity @@ StripBoxes[boxes //. {RowBox[{single_}] :> single, RowBox[{RowBox[{args___}]}] :> RowBox[{args}]}];


(*   tidyBoxes.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
tidyBoxes[boxes_] :=
   boxes  //.
      {RowBox[{RowBox[{args___}]}] :> RowBox[{args}], RowBox[{l___, RowBox[{single_}], r___}] :> RowBox[{l, single, r}]};


(*   Error Checking for cleanBoxes and tidyBoxes.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   cleanBoxes expects only one argument.
*)
error:cleanBoxes[___] :=
   With[{num = heldLength[error]}, $Failed /; num != 1 && Message[cleanBoxes::argx, HoldForm[cleanBoxes], num, 1]];


(*
   tidyBoxes expects only one argument.
*)
error:tidyBoxes[___] :=
   With[{num = heldLength[error]}, $Failed /; num != 1 && Message[tidyBoxes::argx, HoldForm[tidyBoxes], num, 1]];



(*   stripStylingBoxes  --------------------------------------------------------------------------------------------- *)

(*   StripStylingBoxes  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   StripStylingBoxes will remove any styleBoxes, AdjustmentBoxes or FrameBoxes from an expressions.
*)
SetAttributes[stripStylingBoxes, HoldAll];
stripStylingBoxes[StyleBox[boxes_, styles___]] := stripStylingBoxes[boxes];
stripStylingBoxes[AdjustmentBox[boxes_, styles___]] := stripStylingBoxes[boxes];
stripStylingBoxes[FrameBox[boxes_, styles___]] := stripStylingBoxes[boxes];
stripStylingBoxes[TagBox[boxes___]] := TagBox[boxes];
stripStylingBoxes[RowBox[boxes___]] := RowBox[stripStylingBoxes /@ boxes];
stripStylingBoxes[(a_)[args__]] /; boxStructureQ[a[args]] := a @@ stripStylingBoxes /@ {args};
stripStylingBoxes[{args__}] := stripStylingBoxes /@ {args};
stripStylingBoxes[a_] := a;


(*   Error Checking for stripStylingBoxes.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   stripStylingBoxes expects only one argument.
*)
error:stripStylingBoxes[___] :=
   With[
      {num = heldLength[error]},
      $Failed /; num != 1 && Message[stripStylingBoxes::argx, HoldForm[stripStylingBoxes], num, 1]
   ];




(* ------------------------------------------------------------------------------------------------------------------ *)
(*   Pattern Conversions, Handling, and Testing  -------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------------------------------------------------ *)



(*   boxedStringPatternsToPatterns  --------------------------------------------------------------------------------- *)

(*   boxedStringPatternsToPatterns  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   boxedStringPatternsToPatterns will take an expression consisting of boxes and convert all patterns present in the 
   boxes--patterns that are currently still unparsed strings and not yet expressions--to the corresponding patterned 
   expressions, leaving the other boxes alone.
*)
boxedStringPatternsToPatterns[patternBoxes_, (opts___)?OptionQ] :=
   Module[
      {WorkingFormOpt = WorkingForm /. {opts} /. adjustedOptions[Notation]},
      Apply[
         stripStylingBoxes,
         {
            patternBoxes  //.
               {
                  HoldPattern[ParsedBoxWrapper][stringPatternBoxes_] :> stringPatternBoxes,
                  TemplateBox[{stringPatternBoxes_}, "NotationPatternTag", ___]  :>
                     toMyHeldExpression[stringPatternBoxes, WorkingFormOpt],
                  TemplateBox[{stringPatternBoxes_}, "NotationMadeBoxesTag", ___]  :>
                     toMyHeldExpression[stringPatternBoxes, WorkingFormOpt],
                  string_String /; StringMatchQ[string, "*_"] :> ToExpression[string, WorkingFormOpt]
               }
         }
      ]
   ];


(*   Error Checking for boxedStringPatternsToPatterns.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   The second argument and beyond of boxedStringPatternsToPatterns must be Options.
*)
error:boxedStringPatternsToPatterns[_, ___, (notOption_)?(isNot[_?OptionQ]), ___] :=
   $Failed /; Message[boxedStringPatternsToPatterns::nonopt, HoldForm[notOption], 1, HoldForm[error]];



(*   patternToGeneralQ  --------------------------------------------------------------------------------------------- *)

(*
   patternToGeneralQ tests to see if a pattern is too general to be used in a Symbolization or Notation statement.
*)
SetAttributes[patternToGeneralQ, HoldAll];

patternToGeneralQ[(Blank | BlankNull | BlankSequence | BlankNullSequence)[___]] := True;
patternToGeneralQ[HoldPattern[Pattern][_, patternContent_]] := patternToGeneralQ[patternContent];
patternToGeneralQ[myHold[patternContent___]] := patternToGeneralQ[patternContent];
patternToGeneralQ[___] := False;



(*   makeHeldSequenceOfBoxes and makeHeldRowBoxOfBoxes  ------------------------------------------------------------- *)

(*   makeHeldSequenceOfBoxes  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   makeHeldSequenceOfBoxes inserts commas (i.e. "," ) between every boxed expression in the sequence of boxes given to 
   makeHeldSequenceOfBoxes. makeHeldSequenceOfBoxes holds its arguments.
*)
SetAttributes[{makeHeldSequenceOfBoxes, makeHeldRowBoxOfBoxes}, {HoldAllComplete}];

makeHeldSequenceOfBoxes[{}, form_Symbol, None] := Sequence[];
makeHeldSequenceOfBoxes[{expr_}, form_Symbol, None] := MakeBoxes[expr, form];
makeHeldSequenceOfBoxes[{expr_}, form_Symbol, parenthesizedBy_Symbol] := Parenthesize[expr, form, parenthesizedBy];
makeHeldSequenceOfBoxes[{expr___}, form_Symbol, None] :=
   (Sequence @@ Drop[#1, -1] & )[
      Flatten[Thread[{Function[term, MakeBoxes[term, form], {HoldAll}] /@ Unevaluated[{expr}], ","}]]
   ];
makeHeldSequenceOfBoxes[{expr___}, form_Symbol, parenthesizedBy_Symbol] :=
   (Sequence @@ Drop[#1, -1] & )[
      Flatten[
         Thread[{Function[term, Parenthesize[term, form, parenthesizedBy], {HoldAll}] /@ Unevaluated[{expr}], ","}]
      ]
   ];


(*   makeHeldRowBoxOfBoxes  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   makeHeldRowBoxOfBoxes inserts commas (i.e. "," ) between every boxed expression in the sequence of boxes given to 
   makeHeldRowBoxOfBoxes; in addition it returns the result wrapped in a RowBox. makeHeldRowBoxOfBoxes holds its arguments.
*)
makeHeldRowBoxOfBoxes[{}, form_Symbol, None] := RowBox[{}];
makeHeldRowBoxOfBoxes[{expr_}, form_Symbol, None] := MakeBoxes[expr, form];
makeHeldRowBoxOfBoxes[{expr_}, form_Symbol, parenthesizedBy_Symbol] := Parenthesize[expr, form, parenthesizedBy];
makeHeldRowBoxOfBoxes[{expr___}, form_Symbol, None] :=
   (RowBox[Drop[#1, -1]] & )[
      Flatten[Thread[{Function[term, MakeBoxes[term, form], {HoldAll}] /@ Unevaluated[{expr}], ","}]]
   ];
makeHeldRowBoxOfBoxes[{expr___}, form_Symbol, parenthesizedBy_Symbol] :=
   (RowBox[Drop[#1, -1]] & )[
      Flatten[
         Thread[{Function[term, Parenthesize[term, form, parenthesizedBy], {HoldAll}] /@ Unevaluated[{expr}], ","}]
      ]
   ];



(*   makeEvaluatedSequenceOfBoxes and makeEvaluatedRowBoxOfBoxes  --------------------------------------------------- *)

(*   makeEvaluatedSequenceOfBoxes  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   makeEvaluatedSequenceOfBoxes inserts commas (i.e. "," ) between every boxed expression in the sequence of boxes 
   given to makeEvaluatedSequenceOfBoxes. makeEvaluatedSequenceOfBoxes evaluates its arguments.
*)
makeEvaluatedSequenceOfBoxes[{}, form_Symbol, None] := Sequence[];
makeEvaluatedSequenceOfBoxes[{expr_}, form_Symbol, None] := MakeBoxes[expr, form];
makeEvaluatedSequenceOfBoxes[{expr_}, form_Symbol, parenthesizedBy_Symbol] := Parenthesize[expr, form, parenthesizedBy];
makeEvaluatedSequenceOfBoxes[{expr___}, form_Symbol, None] :=
   (Sequence @@ Drop[#1, -1] & )[
      Flatten[Thread[{Function[term, MakeBoxes[term, form], {HoldAll}] /@ Unevaluated[{expr}], ","}]]
   ];
makeEvaluatedSequenceOfBoxes[{expr___}, form_Symbol, parenthesizedBy_Symbol] :=
   (Sequence @@ Drop[#1, -1] & )[
      Flatten[
         Thread[{Function[term, Parenthesize[term, form, parenthesizedBy], {HoldAll}] /@ Unevaluated[{expr}], ","}]
      ]
   ];


(*   makeEvaluatedRowBoxOfBoxes  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   makeEvaluatedRowBoxOfBoxes inserts commas (i.e. "," ) between every boxed expression in the sequence of boxes given 
   to makeEvaluatedRowBoxOfBoxes; in addition it returns the result wrapped in a RowBox. makeEvaluatedRowBoxOfBoxes 
   evaluates its arguments.
*)
makeEvaluatedRowBoxOfBoxes[{}, form_Symbol, None] := RowBox[{}];
makeEvaluatedRowBoxOfBoxes[{expr_}, form_Symbol, None] := MakeBoxes[expr, form];
makeEvaluatedRowBoxOfBoxes[{expr_}, form_Symbol, parenthesizedBy_Symbol] := Parenthesize[expr, form, parenthesizedBy];
makeEvaluatedRowBoxOfBoxes[{expr___}, form_Symbol, None] :=
   (RowBox[Drop[#1, -1]] & )[
      Flatten[Thread[{Function[term, MakeBoxes[term, form], {HoldAll}] /@ Unevaluated[{expr}], ","}]]
   ];
makeEvaluatedRowBoxOfBoxes[{expr___}, form_Symbol, parenthesizedBy_Symbol] :=
   (RowBox[Drop[#1, -1]] & )[
      Flatten[
         Thread[{Function[term, Parenthesize[term, form, parenthesizedBy], {HoldAll}] /@ Unevaluated[{expr}], ","}]
      ]
   ];



(*   Error checking for makeHeldSequenceOfBoxes  -------------------------------------------------------------------- *)

(*
   makeHeldSequenceOfBoxes expects three arguments.
*)
error:makeHeldSequenceOfBoxes[___] :=
   With[
      {num = heldLength[error]},
      Condition[
         $Failed,
         Which[
            num == 1,
            Message[makeHeldSequenceOfBoxes::argr, HoldForm[makeHeldSequenceOfBoxes], 3],
            num != 3,
            Message[makeHeldSequenceOfBoxes::argrx, HoldForm[makeHeldSequenceOfBoxes], num, 3],
            True,
            False
         ]
      ]
   ];


(*
   The 1st argument of makeHeldSequenceOfBoxes must be a list.
*)
error:makeHeldSequenceOfBoxes[(notList_)?(headIsNot[List]), ___] :=
   $Failed /; Message[makeHeldSequenceOfBoxes::list, HoldForm[error], 1];


(*
   The 2nd argument of makeHeldSequenceOfBoxes must be a symbol.
*)
error:makeHeldSequenceOfBoxes[_, (notSymb_)?(headIsNot[Symbol]), ___] :=
   $Failed /; Message[makeHeldSequenceOfBoxes::sym, notSymb, 2];


(*
   The 3rd argument of makeHeldSequenceOfBoxes must be a symbol.
*)
error:makeHeldSequenceOfBoxes[_, _, (notSymb_)?(headIsNot[Symbol]), ___] :=
   $Failed /; Message[makeHeldSequenceOfBoxes::sym, notSymb, 3];



(*   Error checking for makeHeldRowBoxOfBoxes  ---------------------------------------------------------------------- *)

(*
   makeHeldRowBoxOfBoxes expects three arguments.
*)
error:makeHeldRowBoxOfBoxes[___] :=
   With[
      {num = heldLength[error]},
      Condition[
         $Failed,
         Which[
            num == 1,
            Message[makeHeldRowBoxOfBoxes::argr, HoldForm[makeHeldRowBoxOfBoxes], 3],
            num != 3,
            Message[makeHeldRowBoxOfBoxes::argrx, HoldForm[makeHeldRowBoxOfBoxes], num, 3],
            True,
            False
         ]
      ]
   ];


(*
   The 1st argument of makeHeldRowBoxOfBoxes must be a list.
*)
error:makeHeldRowBoxOfBoxes[(notList_)?(headIsNot[List]), ___] :=
   $Failed /; Message[makeHeldRowBoxOfBoxes::list, HoldForm[error], 1];


(*
   The 2nd argument of makeHeldRowBoxOfBoxes must be a symbol.
*)
error:makeHeldRowBoxOfBoxes[_, (notSymb_)?(headIsNot[Symbol]), ___] :=
   $Failed /; Message[makeHeldRowBoxOfBoxes::sym, notSymb, 2];


(*
   The 3rd argument of makeHeldRowBoxOfBoxes must be a symbol.
*)
error:makeHeldRowBoxOfBoxes[_, _, (notSymb_)?(headIsNot[Symbol]), ___] :=
   $Failed /; Message[makeHeldRowBoxOfBoxes::sym, notSymb, 3];



(*   Error checking for makeEvaluatedSequenceOfBoxes  --------------------------------------------------------------- *)

(*
   makeEvaluatedSequenceOfBoxes expects 3 arguments.
*)
error:makeEvaluatedSequenceOfBoxes[___] :=
   With[
      {num = heldLength[error]},
      Condition[
         $Failed,
         Which[
            num == 1,
            Message[makeEvaluatedSequenceOfBoxes::argr, HoldForm[makeEvaluatedSequenceOfBoxes], 3],
            num != 3,
            Message[makeEvaluatedSequenceOfBoxes::argrx, HoldForm[makeEvaluatedSequenceOfBoxes], num, 3],
            True,
            False
         ]
      ]
   ];


(*
   The 1st argument of makeEvaluatedSequenceOfBoxes must be a list.
*)
error:makeEvaluatedSequenceOfBoxes[(notList_)?(headIsNot[List]), ___] :=
   $Failed /; Message[makeEvaluatedSequenceOfBoxes::list, HoldForm[error], 1];


(*
   The 2nd argument of makeEvaluatedSequenceOfBoxes must be a symbol.
*)
error:makeEvaluatedSequenceOfBoxes[_, (notSymb_)?(headIsNot[Symbol]), ___] :=
   $Failed /; Message[makeEvaluatedSequenceOfBoxes::sym, notSymb, 2];


(*
   The 3rd argument of makeEvaluatedSequenceOfBoxes must be a symbol.
*)
error:makeEvaluatedSequenceOfBoxes[_, _, (notSymb_)?(headIsNot[Symbol]), ___] :=
   $Failed /; Message[makeEvaluatedSequenceOfBoxes::sym, notSymb, 3];



(*   Error checking for makeEvaluatedRowBoxOfBoxes  ----------------------------------------------------------------- *)

(*
   makeEvaluatedRowBoxOfBoxes expects three arguments.
*)
error:makeEvaluatedRowBoxOfBoxes[___] :=
   With[
      {num = heldLength[error]},
      Condition[
         $Failed,
         Which[
            num == 1,
            Message[makeEvaluatedRowBoxOfBoxes::argr, HoldForm[makeEvaluatedRowBoxOfBoxes], 3],
            num != 3,
            Message[makeEvaluatedRowBoxOfBoxes::argrx, HoldForm[makeEvaluatedRowBoxOfBoxes], num, 3],
            True,
            False
         ]
      ]
   ];


(*
   The 1st argument of makeEvaluatedRowBoxOfBoxes must be a list.
*)
error:makeEvaluatedRowBoxOfBoxes[(notList_)?(headIsNot[List]), ___] :=
   $Failed /; Message[makeEvaluatedRowBoxOfBoxes::list, HoldForm[error], 1];


(*
   The 2nd argument of makeEvaluatedRowBoxOfBoxes must be a symbol.
*)
error:makeEvaluatedRowBoxOfBoxes[_, (notSymb_)?(headIsNot[Symbol]), ___] :=
   $Failed /; Message[makeEvaluatedRowBoxOfBoxes::sym, notSymb, 2];


(*
   The 3rd argument of makeEvaluatedRowBoxOfBoxes must be a symbol.
*)
error:makeEvaluatedRowBoxOfBoxes[_, _, (notSymb_)?(headIsNot[Symbol]), ___] :=
   $Failed /; Message[makeEvaluatedRowBoxOfBoxes::sym, notSymb, 3];




(* ------------------------------------------------------------------------------------------------------------------ *)
(*   Symbolize Name Handling  --------------------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------------------------------------------------ *)



(*   operatorStringsToSymbolStrings  -------------------------------------------------------------------------------- *)
deleteMultipleUnderBrackets[exprString_String] :=
   StringReplace[exprString, "\[UnderBracket]\[UnderBracket]" -> "\[UnderBracket]"];

pruneEnclosingCharacters[str_String /; StringMatchQ[str, "\"\\[*]\""]] := StringDrop[StringDrop[str, -2], 3];
pruneEnclosingCharacters[other_] := "";

convertNonLettersToFullNames[exprString_String] :=
   StringJoin[
      Characters[exprString]  /.
         str_String /;  !LetterQ[str] &&  !DigitQ[str] && str != "\[UnderBracket]"  :>
            pruneEnclosingCharacters[ToString[FullForm[str]]]
   ];

deletePossibleLeadingAndTrailingUnderBrackets[exprString_String] :=
   Which[
      StringMatchQ[exprString, "\[UnderBracket]*\[UnderBracket]"],
      StringDrop[StringDrop[exprString, 1], -1],
      StringMatchQ[exprString, "\[UnderBracket]*"],
      StringDrop[exprString, 1],
      StringMatchQ[exprString, "*\[UnderBracket]"],
      StringDrop[exprString, -1],
      True,
      exprString
   ];

operatorStringsToSymbolStrings[exprString_String] :=
   deletePossibleLeadingAndTrailingUnderBrackets[
      deleteMultipleUnderBrackets[
         convertNonLettersToFullNames[
            StringReplace[
               exprString,
               {
                  " " -> "\[UnderBracket]Space\[UnderBracket]",
                  "->" -> "\[UnderBracket]Rule\[UnderBracket]",
                  ":>" -> "\[UnderBracket]RuleDelayed\[UnderBracket]",
                  ":=" -> "\[UnderBracket]SetDelayed\[UnderBracket]",
                  "!" -> "\[UnderBracket]Exclamation\[UnderBracket]",
                  "\"" -> "\[UnderBracket]DoubleQuote\[UnderBracket]",
                  "#" -> "\[UnderBracket]Hash\[UnderBracket]",
                  "$" -> "\[UnderBracket]Dollar\[UnderBracket]",
                  "%" -> "\[UnderBracket]Percent\[UnderBracket]",
                  "&&" -> "\[UnderBracket]And\[UnderBracket]",
                  "&" -> "\[UnderBracket]Ampersand\[UnderBracket]",
                  "'" -> "\[UnderBracket]Quote\[UnderBracket]",
                  "(" -> "\[UnderBracket]LeftParenthesis\[UnderBracket]",
                  ")" -> "\[UnderBracket]RightParenthesis\[UnderBracket]",
                  "*" -> "\[UnderBracket]Times\[UnderBracket]",
                  "+" -> "\[UnderBracket]Plus\[UnderBracket]",
                  "," -> "\[UnderBracket]Comma\[UnderBracket]",
                  "-" -> "\[UnderBracket]Dash\[UnderBracket]",
                  "." -> "\[UnderBracket]Dot\[UnderBracket]",
                  "/." -> "\[UnderBracket]Replace\[UnderBracket]",
                  "//." -> "\[UnderBracket]ReplaceRepeated\[UnderBracket]",
                  "//" -> "\[UnderBracket]BackAt\[UnderBracket]",
                  "/;" -> "\[UnderBracket]Condition\[UnderBracket]",
                  "/" -> "\[UnderBracket]Slash\[UnderBracket]",
                  ":" -> "\[UnderBracket]Colon\[UnderBracket]",
                  ";" -> "\[UnderBracket]Semicolon\[UnderBracket]",
                  "<=" -> "\[UnderBracket]LessEqual\[UnderBracket]",
                  "<" -> "\[UnderBracket]Less\[UnderBracket]",
                  "===" -> "\[UnderBracket]SameQ\[UnderBracket]",
                  "==" -> "\[UnderBracket]Equal\[UnderBracket]",
                  "=" -> "\[UnderBracket]Set\[UnderBracket]",
                  ">=" -> "\[UnderBracket]GreaterEqual\[UnderBracket]",
                  ">" -> "\[UnderBracket]Greater\[UnderBracket]",
                  "?" -> "\[UnderBracket]Question\[UnderBracket]",
                  "@@" -> "\[UnderBracket]Apply\[UnderBracket]",
                  "/@" -> "\[UnderBracket]Map\[UnderBracket]",
                  "@" -> "\[UnderBracket]At\[UnderBracket]",
                  "[" -> "\[UnderBracket]LeftBracket\[UnderBracket]",
                  "]" -> "\[UnderBracket]RightBracket\[UnderBracket]",
                  "\\" -> "\[UnderBracket]Backslash\[UnderBracket]",
                  "^" -> "\[UnderBracket]Wedge\[UnderBracket]",
                  "_" -> "\[UnderBracket]Underscore\[UnderBracket]",
                  "`" -> "\[UnderBracket]Backquote\[UnderBracket]",
                  "{" -> "\[UnderBracket]LeftBrace\[UnderBracket]",
                  "||" -> "\[UnderBracket]Or\[UnderBracket]",
                  "|" -> "\[UnderBracket]VerticalBar\[UnderBracket]",
                  "}" -> "\[UnderBracket]RightBrace\[UnderBracket]",
                  "~" -> "\[UnderBracket]Tilde\[UnderBracket]"
               }
            ]
         ]
      ]
   ];



(*   convertBoxesToStringRepresentation  ---------------------------------------------------------------------------- *)
convertBoxesToStringRepresentation[boxes_] :=
   operatorStringsToSymbolStrings[
      StringJoin[
         Flatten[
            {boxes}  //.
               {
                  AdjustmentBox[a_, ___] -> a,
                  ButtonBox[b_, ___] -> b,
                  ErrorBox[b_, ___] -> b,
                  FormBox[b_, ___] -> b,
                  FractionBox[a_, b_, ___] -> {a, "\[UnderBracket]Over\[UnderBracket]", b},
                  FrameBox[b_, ___] -> b,
                  GridBox[args_, ___] -> {args},
                  InterpretationBox[b_, ___] -> b,
                  RadicalBox[a_, b_, ___] -> {a, "\[UnderBracket]Root\[UnderBracket]", b},
                  RowBox[a_] -> a,
                  SqrtBox[a_, ___] -> {"Sqrt\[UnderBracket]", a},
                  StyleBox[a_, ___] -> a,
                  SubscriptBox[a_, b_, ___] -> {a, "\[UnderBracket]Subscript\[UnderBracket]", b},
                  SuperscriptBox[a_, b_, ___] -> {a, "\[UnderBracket]Superscript\[UnderBracket]", b},
                  SubsuperscriptBox[a_, b_, c_, ___]  ->
                     {a, "\[UnderBracket]Subsuperscript\[UnderBracket]", b, "\[UnderBracket]and\[UnderBracket]", c},
                  TagBox[b_, ___] -> b,
                  TemplateBox[{b__}, tag_]  :>
                     {
                        Riffle[{b}, "\[UnderBracket]TemplateArg\[UnderBracket]"],
                        "\[UnderBracket]Template\[UnderBracket]",
                        tag
                     },
                  UnderscriptBox[a_, b_] -> {a, "\[UnderBracket]Underscript\[UnderBracket]", b},
                  OverscriptBox[a_, b_] -> {a, "\[UnderBracket]Overscript\[UnderBracket]", b},
                  UnderoverscriptBox[a_, b_, c_]  ->
                     {a, "\[UnderBracket]Underoverscript\[UnderBracket]", b, "\[UnderBracket]and\[UnderBracket]", c}
               }
         ]
      ]
   ];




(* ------------------------------------------------------------------------------------------------------------------ *)
(*   Symbolize Implementation  -------------------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------------------------------------------------ *)



(*   Setups before the Symbolize  ----------------------------------------------------------------------------------- *)

(*   Transform from the legacy syntax for symbolizations to the new syntax.  - - - - - - - - - - - - - - - - - - - -  *)

(*
   We transform all of the contents of the Symbolize statement from the old legacy syntaxes if they are present to the 
   new syntax which allows us to use only the new syntax in the code which implements the Symbolize statements.
*)
$TransformedLegacySymbolizeSyntax = False;

Symbolize[anyContents___] /;  !$TransformedLegacySymbolizeSyntax :=
   Block[{$TransformedLegacySymbolizeSyntax = True}, Symbolize[TransformLegacySyntax[anyContents]]];



(*   Check Symbolize for parsing and valid options.  ---------------------------------------------------------------- *)

(*
   The condition /; True attached to some of the rules is to circumvent rule reordering bugs in Mathematica.
*)

(*   Handle RemoveSymbolize.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
RemoveSymbolize[symbolizeBoxes_, opts___] := Symbolize[symbolizeBoxes, Action -> RemoveNotationRules, opts];


(*   The Symbolize statement must be created from the Palette.  - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
badSymbolize:Symbolize[symbolizeBoxes_, ___] :=
   Condition[
      (Message[Symbolize::noboxtag, HoldForm[symbolizeBoxes], HoldForm[badSymbolize]]; $Failed),
      Head[symbolizeBoxes] =!= ParsedBoxWrapper
   ];

Symbolize::noboxtag =
   "The Symbolize boxes `1` do not have an embedded TemplateBox with tag NotationTemplateTag. The Symbolize statement `2` may not have been entered using the palette, or the embedded TemplateBox may have been deleted. The embedded TemplateBox ensures correct parsing and retention of proper styling and grouping information.";


(*   The other arguments of Symbolize must be options.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
badSymbolize:Symbolize[symbolizeBoxes_, ___, notOption_, ___] :=
   (Message[Symbolize::nonopt, HoldForm[notOption], 1, HoldForm[badSymbolize]]; $Failed) /;  !OptionQ[notOption];


(*   Check that the Action option is valid.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   The value of option Action should be valid.
*)
badSymbolize:Symbolize[symbolizeBoxes_, ___, (Rule | RuleDelayed)[Action, badValue_], ___] :=
   Condition[
      (Message[Symbolize::optcrp, Action, HoldForm[badValue], HoldForm[badSymbolize]]; $Failed),
       !MatchQ[badValue, CreateNotationRules | RemoveNotationRules | PrintNotationRules]
   ];

Symbolize::optcrp =
   "Value of option '`1` \[Rule] `2`\\' in `3` should be CreateNotationRules, RemoveNotationRules or PrintNotationRules.";


(*   Check not wrapped by RowBox.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   The boxes to be symbolized must also not have a head of RowBox.
*)
badSymbolize:Symbolize[ParsedBoxWrapper[RowBox[boxes___], ___], opts___] :=
   (Message[Symbolize::rowboxh, identityForm @@ {RowBox[boxes]}, HoldForm[badSymbolize]]; $Failed) /; True;
Symbolize::rowboxh =
   "The box structure '`1`' in `2` is not of the right form. Structures to be symbolized cannot have a RowBox as their head. Examine the full box structures for possible unintended groupings.";


(*   Options have all been checked and are o.k.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)


(*   Check form of the patterns in the Symbolize  ------------------------------------------------------------------- *)

(*   boxes too general  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   The boxes to be symbolized must not be too general.
*)
badSymbolize:Symbolize[ParsedBoxWrapper[boxes_, ___], opts___] :=
   With[
      {
         patternedBoxes =
            boxedStringPatternsToPatterns[
               cleanBoxes[boxes],
               WorkingForm -> WorkingForm /. {opts} /. adjustedOptions[Symbolize]
            ]
      },
      Condition[
         (Message[Symbolize::ptogen, patternedBoxes, identityForm[boxes], HoldForm[badSymbolize]]; $Failed),
         patternToGeneralQ[patternedBoxes]
      ]
   ];

Symbolize::ptogen = "The pattern `1` appearing in the symbolize statement Symbolize[`2`] is too general to be used.";


(*   The Symbolize is o.k.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   The Symbolize has been correctly parsed and checked. Now create the symbolization.
*)
Symbolize[symbolizeBoxes_, opts___] :=
   With[{externalBoxes = stripParsedBoxWrapper[symbolizeBoxes]}, createSymbolize[externalBoxes, opts]; ];


(*   Unknown error  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   Have encountered an unknown error in symbolization parsing... Report it!
*)
Symbolize[all___] :=
   ((
      Message[Symbolize::unknpars, identityForm @@ {{all}}];
      CellPrint[Cell[BoxData[colorizeStructuralBoxes[all, 1]], "Output"]];
      $Failed
   ));
Symbolize::unknpars =
   "Unknown error occurred in parsing the Symbolize definition `1`. Please report this to jasonh@wri.com, Please carefully examine the following box structures for spurious characters, weird groupings etc.";



(*   Definition of Symbolize containing patterns.  ------------------------------------------------------------------ *)

(*   createSymbolize  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   createSymbolize handles the symbolization of boxes containing a pattern.
*)
createSymbolize[boxes_, (opts___)?OptionQ] :=
   With[
      {
         WorkingFormOpt = WorkingForm /. {opts} /. adjustedOptions[Symbolize],
         ActionOpt = Action /. {opts} /. adjustedOptions[Symbolize],
         instantiationContext = Context[]
      },
      With[
         {patternedBoxes = boxedStringPatternsToPatterns[cleanBoxes[boxes], WorkingForm -> WorkingFormOpt]},
         (
            If[
               FreeQ[patternedBoxes, Pattern | Blank | BlankNull | BlankSequence | BlankNullSequence],
               createSymbolizeSingleInstance[boxes, instantiationContext, opts],
               executeSymbolizeAction[
                  myHold[NotationMakeExpression[matchedBoxes:patternedBoxes, WorkingFormOpt]],
                  myHold[
                     (
                        createSymbolizeSingleInstance[matchedBoxes, instantiationContext, opts];
                        MakeExpression[matchedBoxes, WorkingFormOpt]
                     )
                  ],
                  WorkingFormOpt,
                  instantiationContext,
                  ActionOpt
               ]
            ];
            Null
         )
      ]
   ];


(*   executeSymbolizeAction  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   executeSymbolizeAction enters, removes or prints the symbolization statement depending on action.
*)
executeSymbolizeAction[external_, internal_, WorkingFormOpt_, instantiationContext_, CreateNotationRules] :=
   releaseMyHold[external /; MemberQ[$ContextPath, instantiationContext] := internal];

executeSymbolizeAction[external_, internal_, WorkingFormOpt_, instantiationContext_, RemoveNotationRules] :=
   releaseMyHold[external /; MemberQ[$ContextPath, instantiationContext] =. ];

executeSymbolizeAction[external_, internal_, WorkingFormOpt_, instantiationContext_, PrintNotationRules] :=
   releaseMyHold[
      CellPrint[
         Cell[
            BoxData[MakeBoxes[external /; MemberQ[$ContextPath, instantiationContext] := internal, StandardForm]],
            "Output",
            ShowStringCharacters -> True
         ]
      ]
   ];



(*   Definition of Symbolize without patterns.  --------------------------------------------------------------------- *)

(*   createSymbolizeSingleInstance  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   createSymbolizeSingleInstance sets up MakeExpression & MakeBoxes rules for a given symbol without patterns in it.
*)
createSymbolizeSingleInstance[boxes_, instantiationContext_, (opts___)?OptionQ] :=
   With[
      {
         WorkingFormOpt = WorkingForm /. {opts} /. adjustedOptions[Symbolize],
         SymbolizeRootNameOpt = SymbolizeRootName /. {opts} /. adjustedOptions[Symbolize],
         ActionOpt = Action /. {opts} /. adjustedOptions[Symbolize]
      },
      With[
         {
            newSymbolString =
               If[
                  SymbolizeRootNameOpt === "",
                  convertBoxesToStringRepresentation[cleanBoxes[boxes]],
                  SymbolizeRootNameOpt
               ]
         },
         executeSymbolizeSingleInstanceAction[boxes, newSymbolString, WorkingFormOpt, instantiationContext, ActionOpt]
      ]
   ];


(*   executeSymbolizeSingleInstanceAction  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   executeSymbolizeSingleInstanceAction enters, removes or prints the single symbolization instance depending on action.
*)
executeSymbolizeSingleInstanceAction[
   boxes_,
   newSymbolString_,
   WorkingFormOpt_,
   instantiationContext_,
   CreateNotationRules
] :=
   ((
      If[Names[newSymbolString] != {}, Message[Symbolize::bsymbexs]];
      With[
         {newSymbol = Symbol[newSymbolString]},
         releaseMyHold[
            (
               Condition[
                  NotationMakeExpression[stripStylingBoxes[boxes], WorkingFormOpt],
                  MemberQ[$ContextPath, instantiationContext]
               ] :=
                  HoldComplete[newSymbol];
               NotationMakeBoxes[newSymbol, WorkingFormOpt] /; MemberQ[$ContextPath, instantiationContext] := boxes;
               HoldComplete[newSymbol]
            )
         ]
      ]
   ));

executeSymbolizeSingleInstanceAction[
   boxes_,
   newSymbolString_,
   WorkingFormOpt_,
   instantiationContext_,
   RemoveNotationRules
] :=
   silentEvaluate[
      With[
         {symbol = myHold @@ MakeExpression[boxes, WorkingFormOpt]},
         releaseMyHold[
            (
               Unset[
                  Condition[
                     NotationMakeExpression[stripStylingBoxes[boxes], WorkingFormOpt],
                     MemberQ[$ContextPath, instantiationContext]
                  ]
               ];
               NotationMakeBoxes[symbol, WorkingFormOpt] =. 
            )
         ]
      ]
   ];

executeSymbolizeSingleInstanceAction[
   boxes_,
   newSymbolString_,
   WorkingFormOpt_,
   instantiationContext_,
   PrintNotationRules
] :=
   With[
      {newSymbol = Symbol[newSymbolString], stripedBoxes = stripStylingBoxes[boxes]},
      releaseMyHold[
         (
            CellPrint[
               Cell[
                  BoxData[
                     MakeBoxes[
                        Condition[
                           NotationMakeExpression[stripedBoxes, WorkingFormOpt],
                           MemberQ[$ContextPath, instantiationContext]
                        ] :=
                           HoldComplete[newSymbol],
                        StandardForm
                     ]
                  ],
                  "Output",
                  ShowStringCharacters -> True
               ]
            ];
            CellPrint[
               Cell[
                  BoxData[
                     MakeBoxes[
                        NotationMakeBoxes[newSymbol, WorkingFormOpt] /; MemberQ[$ContextPath, instantiationContext] :=
                           boxes,
                        StandardForm
                     ]
                  ],
                  "Output",
                  ShowStringCharacters -> True
               ]
            ];
            Null
         )
      ]
   ];

Symbolize::bsymbexs =
   "Warning: The box structure attempting to be symbolized has a similar or identical symbol already defined, possibly overriding previously symbolized box structure.";


(*   Modify Remove for symbolized symbols.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   It was nice to be able to modify Remove to take into account the removal of MakeExpression & MakeBoxes rules for a 
   symbolized symbol, but in v 9.0.0 and later Remove is Locked so it is not possible to attach behaviors to Remove.
*)


(*   Symbolize Error Handling Catch All  ---------------------------------------------------------------------------- *)

(*
   Have encountered an unknown error in Symbolize checking... Report it!
*)
createSymbolize[all_, other___] :=
   ((
      Message[Symbolize::unknproc];
      CellPrint[Cell[BoxData[colorizeStructuralBoxes[all, 1]], "Output"]];
      HoldComplete[Symbol["$Failed"]]
   ));

Symbolize::unknproc =
   "Unknown error occurred in processing the Symbolize definition.  \t\tPlease report this to jasonh@wri.com, Please carefully examine the following box structures for spurious characters, weird groupings etc.";




(* ------------------------------------------------------------------------------------------------------------------ *)
(*   InfixNotation Implementation  ---------------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------------------------------------------------ *)



(*   Setups before the InfixNotation  ------------------------------------------------------------------------------- *)

(*   Transform from the legacy syntax for infix notations to the new syntax.  - - - - - - - - - - - - - - - - - - -  *)

(*
   We transform all of the contents of the InfixNotation statement from the old legacy syntaxes if they are present to 
   the new syntax which allows us to use only the new syntax in the code which implements the InfixNotation statements.
*)
$TransformedLegacyInfixNotationSyntax = False;

InfixNotation[anyContents___] /;  !$TransformedLegacyInfixNotationSyntax :=
   Block[{$TransformedLegacyInfixNotationSyntax = True}, InfixNotation[TransformLegacySyntax[anyContents]]];



(*   Check InfixNotation for parsing and valid options.  ------------------------------------------------------------ *)

(*   Handle RemoveInfixNotation.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
RemoveInfixNotation[infixopBoxes_, prefixHead_, opts___] :=
   InfixNotation[infixopBoxes, prefixHead, Action -> RemoveNotationRules, opts];


(*   The InfixNotation statement must be created from the palette.  - - - - - - - - - - - - - - - - - - - - - - - -  *)
badInfixNotation:InfixNotation[infixopBoxes_, ___] :=
   Condition[
      (Message[InfixNotation::noboxtag, HoldForm[infixopBoxes], HoldForm[badInfixNotation]]; $Failed),
      Head[infixopBoxes] =!= ParsedBoxWrapper
   ];

InfixNotation::noboxtag =
   "The InfixNotation boxes `1` do not have an embedded TemplateBox with tag NotationTemplateTag. The InfixNotation statement `2` may not have been entered using the palette, or the embedded TemplateBox may have been deleted. The embedded TemplateBox ensures correct parsing and retention of proper styling and grouping information.";


(*   The other arguments of InfixNotation must be options.  - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
badInfixNotation:InfixNotation[infixop_, prefixHead_, ___, notOption_, ___] :=
   (Message[InfixNotation::nonopt, HoldForm[notOption], 1, HoldForm[badInfixNotation]]; $Failed) /;  !OptionQ[notOption];


(*   Check that the Action option is valid.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   The value of option Action should be valid.
*)
badInfixNotation:InfixNotation[infixop_, prefixHead_, ___, (Rule | RuleDelayed)[Action, badValue_], ___] :=
   Condition[
      (Message[InfixNotation::optcrp, Action, HoldForm[badValue], HoldForm[badInfixNotation]]; $Failed),
       !MatchQ[badValue, CreateNotationRules | RemoveNotationRules | PrintNotationRules]
   ];

InfixNotation::optcrp =
   "Value of option '`1` \[Rule] `2`' in `3` should be CreateNotationRules, RemoveNotationRules or PrintNotationRules.";


(*   Check not wrapped by RowBox.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   The boxes to be InfixNotation must also not have a head of RowBox.
*)
badInfixNotation:InfixNotation[ParsedBoxWrapper[RowBox[boxes___], ___], prefixHead_, opts___] :=
   (Message[InfixNotation::rowboxh, identityForm @@ {RowBox[boxes]}, HoldForm[badInfixNotation]]; $Failed) /; True;
InfixNotation::rowboxh =
   "The InfixNotation box structure '`1`' is not of the right form. Structures to be used as an infix operator cannot have a RowBox as their head. Examine the full box structures of `2` for possible unintended groupings.";


(*   The prefixHead argument to InfixNotation must be a symbol.  - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
badInfixNotation:InfixNotation[ParsedBoxWrapper[infixopBoxes_, ___], badPrefixHead_, opts___] :=
   Condition[
      (
         Message[
            InfixNotation::bprfxh,
            HoldForm[badPrefixHead],
            identityForm @@ {infixopBoxes},
            HoldForm[badInfixNotation]
         ];
         $Failed
      ),
      Head[badPrefixHead] =!= Symbol
   ];
InfixNotation::bprfxh = "In `3`, the prefix head '`1`' corresponding to the infix operator '`2`' is not a symbol.";


(*   InfixNotation is ok. Create the InfixNotation.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   The InfixNotation has been correctly parsed and checked. Now create the InfixNotation.
*)
InfixNotation[infixop_, prefixHead_, opts___] :=
   With[{infixopBoxes = stripParsedBoxWrapper[infixop]}, createInfixNotation[infixopBoxes, prefixHead, opts]; ];


(*   Have encountered an unknown error in InfixNotation parsing... Report it!  - - - - - - - - - - - - - - - - - - -  *)
InfixNotation[all_, other___] :=
   ((
      Message[InfixNotation::unknpars, identityForm @@ {all}];
      CellPrint[Cell[BoxData[colorizeStructuralBoxes[all, 1]], "Output"]];
      $Failed
   ));
InfixNotation::unknpars =
   "Unknown error occurred in parsing the InfixNotation definition `1`. Please report this to jasonh@wri.com, Please carefully examine the following box structures for spurious characters weird groupings etc.";



(*   Definition of -=< Ommitted Inline Cell >=-  -------------------------------------------------------------------- *)

(*   createInfixNotation  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   createInfixNotation creates the MakeExpression and MakeBoxes rules that create the infix notation.
*)
createInfixNotation[infixop_, prefixHead_, (opts___)?OptionQ] :=
   With[
      {
         WorkingFormOpt = WorkingForm /. {opts} /. adjustedOptions[InfixNotation],
         ActionOpt = Action /. {opts} /. adjustedOptions[InfixNotation],
         instantiationContext = Context[]
      },
      executeInfixNotationAction[
         infixop,
         prefixHead,
         effectiveOperator[infixop],
         WorkingFormOpt,
         instantiationContext,
         ActionOpt
      ]
   ];


(*   executeInfixNotationAction  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   executeInfixNotationAction enters, removes or prints the InfixNotation statement depending on the action.
*)
executeInfixNotationAction[
   infixop_,
   prefixHead_,
   groupingPrecedence_,
   WorkingFormOpt_,
   instantiationContext_,
   CreateNotationRules
] :=
   ((
      Condition[
         NotationMakeExpression[RowBox[a_], WorkingFormOpt],
         MemberQ[a, infixop] && MemberQ[$ContextPath, instantiationContext]
      ] :=
         MakeExpression[parseFlatInfix[prefixHead, infixop, a], WorkingFormOpt] /; appearsInfix[infixop, a];
      NotationMakeBoxes[prefixHead[arg1_, args__], WorkingFormOpt] /; MemberQ[$ContextPath, instantiationContext] :=
         (RowBox[Riffle[#1, infixop]] & )[
            (Parenthesize[#1, WorkingFormOpt, groupingPrecedence] & ) /@ Unevaluated /@ Unevaluated[{arg1, args}]
         ];
      Null
   ));

executeInfixNotationAction[
   infixop_,
   prefixHead_,
   groupingPrecedence_,
   WorkingFormOpt_,
   instantiationContext_,
   RemoveNotationRules
] :=
   silentEvaluate[
      (
         Unset[
            Condition[
               NotationMakeExpression[RowBox[a_], WorkingFormOpt],
               MemberQ[a, infixop] && MemberQ[$ContextPath, instantiationContext]
            ]
         ];
         NotationMakeBoxes[prefixHead[arg1_, args__], WorkingFormOpt] /; MemberQ[$ContextPath, instantiationContext] =. ;
         Null
      )
   ];

executeInfixNotationAction[
   infixop_,
   prefixHead_,
   groupingPrecedence_,
   WorkingFormOpt_,
   instantiationContext_,
   PrintNotationRules
] :=
   ((
      CellPrint[
         Cell[
            BoxData[
               MakeBoxes[
                  Condition[
                     NotationMakeExpression[RowBox[a_], WorkingFormOpt],
                     MemberQ[a, infixop] && MemberQ[$ContextPath, instantiationContext]
                  ] :=
                     MakeExpression[parseFlatInfix[prefixHead, infixop, a], WorkingFormOpt] /; appearsInfix[infixop, a],
                  StandardForm
               ]
            ],
            "Output",
            ShowStringCharacters -> True
         ]
      ];
      CellPrint[
         Cell[
            BoxData[
               MakeBoxes[
                  MakeBoxes[prefixHead[arg1_, args__], WorkingFormOpt] /; MemberQ[$ContextPath, instantiationContext] :=
                     (RowBox[Riffle[#1, infixop]] & )[
                        Map[
                           Parenthesize[#1, WorkingFormOpt, groupingPrecedence] & ,
                           Unevaluated /@ Unevaluated[{arg1, args}]
                        ]
                     ],
                  StandardForm
               ]
            ],
            "Output",
            ShowStringCharacters -> True
         ]
      ];
      Null
   ));


(*   parseFlatInfix  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   parseFlatInfix parses a chain of operands separated by the infix operator
*)

(*
   Note the following coding semi breaks the style of the rest of the notebook somewhat, but the algorithm needs to be 
   this way due to speed.
*)
SetAttributes[{parseFlatInfix, parseFlatInfixAux}, HoldAll];

parseFlatInfix[
   prefixHead_,
   infixop_,
   {l___, (first_)?notWhiteSpaceQ, ___?whiteSpaceQ, infixop_, ___?whiteSpaceQ, (b_)?notWhiteSpaceQ, rest___}
] :=
   parseFlatInfixAux[prefixHead, infixop, first, {l}, {b, rest}];

parseFlatInfixAux[prefixHead_, infixop_, first_, {l___}, {rest___}] :=
   Block[
      {parseOperator, parseArgument, returnAnswer},
      (
         SetAttributes[{parseOperator, parseArgument}, HoldAll];
         parseOperator[_?whiteSpaceQ, r___] := parseOperator[r];
         parseOperator[infixop, r___] := parseArgument[r];
         parseArgument[_?whiteSpaceQ, r___] := parseArgument[r];
         parseArgument[a_, r___] := Sequence[a, parseOperator[r]];
         returnAnswer[{parsed__, parseOperator[r___]}] := returnFormatedOutput[prefixHead, {l}, {parsed}, {r}];
         returnAnswer[Block[{$RecursionLimit = Infinity}, {first, parseArgument[rest]}]]
      )
   ];

appearsInfix[op_, {___, _?notWhiteSpaceQ, ___?whiteSpaceQ, op_, ___?whiteSpaceQ, _?notWhiteSpaceQ, ___}] := True;
appearsInfix[_] := False;

returnFormatedOutput[prefixHead_, {l___}, {parsed__}, {r___}] :=
   RowBox[{l, RowBox[{ToString[prefixHead], "[", RowBox[Riffle[{parsed}, ","]], "]"}], r}] /; {l, r} =!= {};
returnFormatedOutput[prefixHead_, {l___}, {parsed__}, {r___}] :=
   RowBox[{ToString[prefixHead], "[", RowBox[Riffle[{parsed}, ","]], "]"}] /; {l, r} === {};




(* ------------------------------------------------------------------------------------------------------------------ *)
(*   AddInputAlias and ActiveInputAliases Implementation  ----------------------------------------------------------- *)
(* ------------------------------------------------------------------------------------------------------------------ *)



(*   Check AddInputAlias for parsing and valid options. (Legacy case of two-argument form.)  ------------------------ *)

(*   The AddInputAlias statement must be created from the palette.  - - - - - - - - - - - - - - - - - - - - - - - -  *)
badInputAlias:AddInputAlias[badBoxes_, ___] :=
   Condition[
      (Message[AddInputAlias::noboxtag, HoldForm[badBoxes], HoldForm[badInputAlias]]; $Failed),
       !MatchQ[Head[badBoxes], ParsedBoxWrapper | List | Rule]
   ];

AddInputAlias::noboxtag =
   "The InputAliasBoxes boxes `1` do not have an embedded TemplateBox with tag NotationTemplateTag. The AddInputAlias statement `2` may not have been entered using the palette, or the embedded TemplateBox may have been deleted. The embedded TemplateBox ensures correct parsing and retention of proper styling and grouping information.";


(*   The second argument of AddInputAlias must be a string.  - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
badInputAlias:AddInputAlias[fullBoxes_ParsedBoxWrapper, badShortForm_, ___] :=
   Condition[
      (Message[AddInputAlias::bshfrm, HoldForm[badShortForm], HoldForm[badInputAlias]]; $Failed),
      Head[badShortForm] =!= String
   ];

AddInputAlias::bshfrm = "The short form `1` in the AddInputAlias statement `2` is not a string.";


(*   The third argument of AddInputAlias must be a notebook object.  - - - - - - - - - - - - - - - - - - - - - - - -  *)
badInputAlias:AddInputAlias[fullBoxes_ParsedBoxWrapper, shortForm_String, badNotebook_] :=
   Condition[
      (Message[AddInputAlias::badrnb, HoldForm[badNotebook], HoldForm[badInputAlias]]; $Failed),
      Head[badNotebook] =!= NotebookObject
   ];

AddInputAlias::badrnb = "The third argument of `2` is not a notebook object.";



(*   Check AddInputAlias for parsing and valid options. (in rule form)  --------------------------------------------- *)

(*   validAliasRule  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   validAliasRule checks to see whether the given argument is a valid input alias rule before processing.
*)
validAliasRule[HoldPattern[Rule][_String, _ParsedBoxWrapper]] := True;
validAliasRule[_] := False;


(*   The first argument of AddInputAlias (in rule form) must be a string.  - - - - - - - - - - - - - - - - - - - - -  *)
badInputAlias:AddInputAlias[badShortForm_ -> fullBoxes_ParsedBoxWrapper, ___] :=
   Condition[
      (Message[AddInputAlias::bshfrm, HoldForm[badShortForm], HoldForm[badInputAlias]]; $Failed),
      Head[badShortForm] =!= String
   ];

AddInputAlias::bshfrm = "The short form `1` in the AddInputAlias statement `2` is not a string.";


(*   The AddInputAlias statement (in rule form) must be created from the palette.  - - - - - - - - - - - - - - - - -  *)
badInputAlias:AddInputAlias[_ -> badBoxes_, ___] :=
   Condition[
      (Message[AddInputAlias::noboxtag, HoldForm[badBoxes], HoldForm[badInputAlias]]; $Failed),
      Head[badBoxes] =!= ParsedBoxWrapper
   ];

AddInputAlias::noboxtag =
   "The InputAliasBoxes boxes `1` do not have an embedded TemplateBox with tag NotationTemplateTag. The AddInputAlias statement `2` may not have been entered using the palette, or the embedded TemplateBox may have been deleted. The embedded TemplateBox ensures correct parsing and retention of proper styling and grouping information.";


(*   The second argument of AddInputAlias must be a notebook object or $FrontEnd.  - - - - - - - - - - - - - - - - -  *)
badInputAlias:AddInputAlias[_?validAliasRule, badNotebook_] :=
   Condition[
      (Message[AddInputAlias::badnb, HoldForm[badNotebook], HoldForm[badInputAlias]]; $Failed),
      Head[badNotebook] =!= NotebookObject && badNotebook =!= $FrontEnd
   ];

AddInputAlias::badnb = "The second argument of `2` is not a notebook object.";



(*   Check AddInputAlias for parsing and valid options. (in lists-of-rules form)  ----------------------------------- *)

(*   The AddInputAlias statement with lists must have a non-empty list of aliases.  - - - - - - - - - - - - - - - -  *)
badInputAliases:AddInputAlias[{}, ___] := (Message[AddInputAlias::emptyals, HoldForm[badInputAliases]]; $Failed);

AddInputAlias::emptyals = "The input aliases list in `1` is empty. It must contain at least one valid input alias.";


(*   The AddInputAlias statement with lists must be created from the palette.  - - - - - - - - - - - - - - - - - - -  *)
badInputAliases:AddInputAlias[badBoxesList_List, ___] :=
   Condition[
      (Message[AddInputAlias::noboxtgl, HoldForm[badBoxesList], HoldForm[badInputAliases]]; $Failed),
      silentEvaluate[Union[Head /@ (#1[[2]] & ) /@ badBoxesList]] =!= {ParsedBoxWrapper}
   ];

AddInputAlias::noboxtgl =
   "The InputAliasBoxes boxes list `1` contains a rule which does not have an embedded TemplateBox with tag NotationTemplateTag. The AddInputAlias statement `2` may not have been entered using the palette, or the embedded TemplateBox may have been deleted. The embedded TemplateBox ensures correct parsing and retention of proper styling and grouping information.";


(*   The first arguments in the list of rules of AddInputAlias statement with lists must be a string.  - - - - - - -  *)
badInputAliases:AddInputAlias[badShortFormList_List, ___] :=
   Condition[
      (Message[AddInputAlias::bshfrml, HoldForm[badInputAliases]]; $Failed),
      silentEvaluate[Union[Head /@ First /@ badShortFormList]] =!= {String}
   ];

AddInputAlias::bshfrml = "One of the short forms in the AddInputAlias `1` statement is not a string.";


(*   The second argument of AddInputAlias must be a notebook object or $FrontEnd.  - - - - - - - - - - - - - - - - -  *)
badInputAlias:AddInputAlias[{__?validAliasRule}, badNotebook_] :=
   Condition[
      (Message[AddInputAlias::badnb, HoldForm[badNotebook], HoldForm[badInputAlias]]; $Failed),
      Head[badNotebook] =!= NotebookObject && badNotebook =!= $FrontEnd
   ];

AddInputAlias::badnb = "The second argument of `2` is not a notebook object.";


(*   AddInputAlias expects 1, 2, or 3 arguments.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
error:AddInputAlias[___] :=
   With[
      {num = heldLength[error]},
      Condition[
         $Failed,
         Which[1 > num || num > 3, Message[AddInputAlias::argb, HoldForm[AddInputAlias], num, 1, 3], True, False]
      ]
   ];



(*   Definition of -=< Ommitted Inline Cell >=-  -------------------------------------------------------------------- *)

(*
   The AddInputAlias has been correctly parsed and checked. Now add the alias(es) to the notebook. Basically just get 
   the current aliases and add new one(s) to this list.
*)

(*   Handles adding a list of rule aliases (much faster.)  - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   The following handles adding a list of rule aliases at one time (It is much faster since communication with the 
   front end is very slow.)
*)
AddInputAlias[{(aliases__Rule)?validAliasRule}, HoldPattern[notebook_:InputNotebook[]]] :=
   With[
      {stripedAliases = stripParsedBoxWrapper[{aliases}], oldAliases = Options[notebook, InputAliases][[1,2]]},
      With[
         {
            prunedAliases =
               DeleteCases[oldAliases, HoldPattern[Rule | RuleDelayed][Alternatives @@ First /@ stripedAliases, _]]
         },
         SetOptions[notebook, InputAliases -> Join[prunedAliases, stripedAliases]]
      ]
   ];


(*   The following handles adding a single rule alias.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
AddInputAlias[(alias_Rule)?validAliasRule, HoldPattern[notebook_:InputNotebook[]]] :=
   With[
      {stripedAlias = stripParsedBoxWrapper[{alias}], oldAliases = Options[notebook, InputAliases][[1,2]]},
      With[
         {
            prunedAliases =
               DeleteCases[oldAliases, HoldPattern[Rule | RuleDelayed][Alternatives @@ First /@ stripedAlias, _]]
         },
         SetOptions[notebook, InputAliases -> Join[prunedAliases, stripedAlias]]
      ]
   ];


(*   The original calling syntax for compatibility.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
AddInputAlias[fullBoxes_ParsedBoxWrapper, shortForm_String, HoldPattern[notebook_:InputNotebook[]]] :=
   AddInputAlias[shortForm -> fullBoxes, notebook];



(*   Check ActiveInputAliases for parsing and valid options.  ------------------------------------------------------- *)

(*   The first argument of ActiveInputAliases must be a notebook object.  - - - - - - - - - - - - - - - - - - - - -  *)
badActiveInputAliases:ActiveInputAliases[badNotebook_, ___] :=
   Condition[
      (Message[ActiveInputAliases::badnb, HoldForm[badNotebook], HoldForm[badActiveInputAliases]]; $Failed),
      Head[badNotebook] =!= NotebookObject
   ];

ActiveInputAliases::badnb = "The first argument of `2` is not a notebook object.";


(*   ActiveInputAliases expects zero arguments or one argument  - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
error:ActiveInputAliases[_, __] :=
   $Failed /; Message[ActiveInputAliases::argt, HoldForm[ActiveInputAliases], heldLength[error], 0, 1];



(*   Definition of -=< Ommitted Inline Cell >=-  -------------------------------------------------------------------- *)

(*   The ActiveInputAliases has been correctly parsed and checked. Now list the active Aliases.  - - - - - - - - - -  *)
ActiveInputAliases[HoldPattern[notebook_:InputNotebook[]]] :=
   DisplayForm[TableForm[Options[notebook, InputAliases][[1,2]]]];




(* ------------------------------------------------------------------------------------------------------------------ *)
(*   Notation Preprocessing  ---------------------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------------------------------------------------ *)



(*   Setups before the Notation  ------------------------------------------------------------------------------------ *)

(*   Transform from the legacy syntax for notations to the new syntax.  - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   We transform all of the contents of the Notation statement from the old legacy syntaxes if they are present to the 
   new syntax which allows us to use only the new syntax in the code which implements the Notation statements.
*)
$TransformedLegacyNotationSyntax = False;

Notation[anyContents___] /;  !$TransformedLegacyNotationSyntax :=
   Block[{$TransformedLegacyNotationSyntax = True}, Notation[TransformLegacySyntax[anyContents]]];



(*   Check Notation for parsing and valid options.  ----------------------------------------------------------------- *)

(*   Handle RemoveNotation.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
RemoveNotation[notation_, opts___] := Notation[notation, Action -> RemoveNotationRules, opts];


(*   Warn of legacy use of <=> vs. <==>.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
Notation[oldNotation_, rest___] :=
   Condition[
      (Message[Notation::oldnota]; Notation[DoubleLongLeftRightArrow @@ oldNotation, rest]),
      MatchQ[SymbolName[Head[oldNotation]], "DoubleLeftRightArrow"]
   ];

Notation::oldnota =
   "Future versions of the Notation package will no longer support \[DoubleLeftRightArrow], instead they will use \[DoubleLongLeftRightArrow]. Please make this change to all your Notations.";


(*   The main argument to Notation must be of the form (external notation) arrow (internal notation).  - - - - - - -  *)
validNotationHeads = "DoubleLongRightArrow" | "DoubleLongLeftArrow" | "DoubleLongLeftRightArrow";

Notation[badNotation_, rest___] :=
   (Message[Notation::badnota, badNotation]; $Failed) /;  !MatchQ[SymbolName[Head[badNotation]], validNotationHeads];

Notation::badnota =
   "The notation '`1`' is not of the form externalBoxes \[DoubleLongLeftRightArrow] internalExpression or externalBoxes \[DoubleLongRightArrow] internalExpression or externalBoxes \[DoubleLongLeftArrow] internalExpression. Examine the full box structures for possible unintended groupings.";


(*   The Notation statement must be created from the Palette.  - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
badNotation:Notation[(direction_)[badExternalNotation_, _], rest___] :=
   Condition[
      (Message[Notation::noexbtag, HoldForm[badExternalNotation], HoldForm[badNotation]]; $Failed),
      Head[badExternalNotation] =!= ParsedBoxWrapper
   ];

Notation::noexbtag =
   "The external representation `1` does not have an embedded TemplateBox with tag NotationTemplateTag. The Notation statement `2` may not have been entered using the palette, or the embedded TemplateBox may have been deleted. The embedded TemplateBox ensures correct parsing and retention of proper styling and grouping information.";

badNotation:Notation[(direction_)[_, badInternalNotation_], rest___] :=
   Condition[
      (Message[Notation::noinbtag, HoldForm[badInternalNotation], HoldForm[badNotation]]; $Failed),
      Head[badInternalNotation] =!= ParsedBoxWrapper
   ];

Notation::noinbtag =
   "The internal representation `1` does not have an embedded TemplateBox with tag NotationTemplateTag. The Notation statement `2` may not have been entered using the palette, or the embedded TemplateBox may have been deleted. The embedded TemplateBox ensures correct parsing and retention of proper styling and grouping information.";


(*   The other arguments of Notation must be options.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
badNotation:Notation[notation_, ___, notOption_, ___] :=
   (Message[Notation::nonopt, HoldForm[notOption], 1, HoldForm[badNotation]]; $Failed) /;  !OptionQ[notOption];


(*   The value of option Action should be valid.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
badNotation:Notation[notation_, ___, (Rule | RuleDelayed)[Action, badValue_], ___] :=
   Condition[
      (Message[Notation::optcrp, Action, HoldForm[badValue], HoldForm[badNotation]]; $Failed),
       !MatchQ[badValue, CreateNotationRules | RemoveNotationRules | PrintNotationRules]
   ];

Notation::optcrp =
   "Value of option '`1` \[Rule] `2`' in `3` should be CreateNotationRules, RemoveNotationRules or PrintNotationRules.";


(*   Checks that the internal expression argument of Notation is a parsable expression.  - - - - - - - - - - - - - -  *)
badNotation:Notation[(notationType_)[external_, internal_], opts___] :=
   Condition[
      (Message[Notation::brepbxs, HoldForm[internal], HoldForm[badNotation]]; $Failed),
      silentEvaluate[ !parsableQ[stripParsedBoxWrapper[internal], opts]]
   ];

Notation::brepbxs = "Cannot parse the internal representation '`1`' occurring in `2`";


(*   Options have all been checked and are o.k.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)


(*   Check form of complex NotationPatterns.  ----------------------------------------------------------------------- *)

(*   Check that the complex external patterns are of the right form.  - - - - - - - - - - - - - - - - - - - - - - -  *)

(*   Check that the complex internal patterns are of the right form.  - - - - - - - - - - - - - - - - - - - - - - -  *)

(*   complexPatternsInBoxes  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   Where complexPatternsInBoxes is defined by:
*)


(*   Check form of the patterns in the Notation  -------------------------------------------------------------------- *)

(*   Compare the external and internal patterns for consistency  - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   The external and internal patterns now have to be compared to see if they are consistent with the notation.
*)
fullArrowsToShortArrowsRules =
   {
      "DoubleLongRightArrow" -> "\[DoubleLongRightArrow]",
      "DoubleLongLeftArrow" -> "\[DoubleLongLeftArrow]",
      "DoubleLongLeftRightArrow" -> "\[DoubleLongLeftRightArrow]"
   };

Notation[(notationType_)[external_, internal_], opts___] :=
   checkNotationPatterns[
      external,
      patternsInBoxes[external, opts],
      SymbolName[notationType] /. fullArrowsToShortArrowsRules,
      internal,
      patternsInBoxes[internal, opts],
      opts
   ];


(*   patternsInBoxes  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   Where the patterns present in the boxes can be found from the following.
*)
patternsInBoxes[boxes_, opts___] :=
   Module[
      {WorkingFormOpt = WorkingForm /. {opts} /. adjustedOptions[Notation]},
      Union[
         Cases[
            boxes  //.
               {
                  TemplateBox[{stringPatternBoxes_}, "NotationPatternTag", ___]  :>
                     With[
                        {eval = convertPatterns[toMyHeldExpression[stringPatternBoxes, WorkingFormOpt]]},
                        eval /; True
                     ],
                  TemplateBox[{stringPatternBoxes_}, "NotationMadeBoxesTag", ___]  :>
                     With[
                        {eval = convertPatterns[toMyHeldExpression[stringPatternBoxes, WorkingFormOpt]]},
                        eval /; True
                     ],
                  string_String /; StringMatchQ[string, "*_"]  :>
                     With[{eval = convertPatterns[toMyHeldExpression[string, WorkingFormOpt]]}, eval /; True]
               },
            (a_singleBlank) | (a_doubleBlank) | (a_tripleBlank) | (a_complexPattern) -> a,
            {0, Infinity}
         ]
      ]
   ];



(*    checkNotationPatterns for free blanks.  ----------------------------------------------------------------------- *)

(*   Free blanks cannot occur in the output.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
checkNotationPatterns[
   externalBoxes_,
   externalPatterns_,
   "\[DoubleLongLeftArrow]" | "\[DoubleLongLeftRightArrow]",
   internalBoxes_,
   internalPatterns_,
   opts___
] :=
   (Message[Notation::frepatex, identityForm @@ externalBoxes]; $Failed) /; containsFreeBlanksQ[externalPatterns];

Notation::frepatex =
   "All patterns in the external representation must be able to be filled.  Free blank patterns found inside the external representation '`1`'.";


(*   Free blanks cannot occur in the input.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
checkNotationPatterns[
   externalBoxes_,
   externalPatterns_,
   "\[DoubleLongRightArrow]" | "\[DoubleLongLeftRightArrow]",
   internalBoxes_,
   internalPatterns_,
   opts___
] :=
   (Message[Notation::frepatin, identityForm @@ internalBoxes]; $Failed) /; containsFreeBlanksQ[internalPatterns];

Notation::frepatin =
   "All patterns in the internal representation must be able to be filled.  Free blank patterns found inside the internal representation '`1`'.";


(*   containsFreeBlanksQ  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   Where containsFreeBlanksQ is defined by:
*)
containsFreeBlanksQ[patterns_] :=  !FreeQ[patterns, singleBlank[] | doubleBlank[] | tripleBlank[]];



(*    checkNotationPatterns to catch patterns that are not fillable.  ----------------------------------------------- *)

(*   Check for patterns in the external representation which are not being filled.  - - - - - - - - - - - - - - - -  *)
checkNotationPatterns[
   externalBoxes_,
   externalPatterns_,
   "\[DoubleLongLeftArrow]" | "\[DoubleLongLeftRightArrow]",
   internalBoxes_,
   internalPatterns_,
   opts___
] :=
   With[
      {lhsPatternsNotInrhs = Complement[externalPatterns, internalPatterns]},
      Condition[
         (
            Message[
               Notation::expatnf,
               HoldForm @@ First[lhsPatternsNotInrhs],
               identityForm @@ externalBoxes,
               identityForm @@ internalBoxes
            ];
            $Failed
         ),
         lhsPatternsNotInrhs =!= {}
      ]
   ];

Notation::expatnf =
   "Pattern '`1`' appearing in the external representation '`2`' cannot be filled since '`1`' does not appear in the internal representation '`3`'.";


(*   Check for patterns in the internal representation which are not being filled.  - - - - - - - - - - - - - - - -  *)
checkNotationPatterns[
   externalBoxes_,
   externalPatterns_,
   "\[DoubleLongRightArrow]" | "\[DoubleLongLeftRightArrow]",
   internalBoxes_,
   internalPatterns_,
   opts___
] :=
   With[
      {rhsPatternsNotInlhs = Complement[internalPatterns, externalPatterns]},
      Condition[
         (
            Message[
               Notation::inpatnf,
               HoldForm @@ First[rhsPatternsNotInlhs],
               identityForm @@ internalBoxes,
               identityForm @@ externalBoxes
            ];
            $Failed
         ),
         rhsPatternsNotInlhs =!= {}
      ]
   ];

Notation::inpatnf =
   "Pattern '`1`' appearing in the internal representation '`2`' cannot be filled since '`1`' does not appear in the external representation '`3`'.";



(*   checkNotationPatterns for patterns that are not used.  --------------------------------------------------------- *)

(*   Check for named patterns in the external representation which are not being used.  - - - - - - - - - - - - - -  *)
checkNotationPatterns[
   externalBoxes_,
   externalPatterns_,
   "\[DoubleLongRightArrow]",
   internalBoxes_,
   internalPatterns_,
   opts___
] :=
   With[
      {
         lhsPatternsNotInrhs =
            Complement[externalPatterns, internalPatterns, {singleBlank[], doubleBlank[], tripleBlank[]}]
      },
      Condition[
         $Failed,
         And[
            lhsPatternsNotInrhs =!= {},
            Message[
               Notation::expatnu,
               (HoldForm @@ #1 & ) /@ lhsPatternsNotInrhs,
               identityForm @@ externalBoxes,
               identityForm @@ internalBoxes
            ]
         ]
      ]
   ];

Notation::expatnu =
   "Warning: The pattern(s) '`1`' appearing in the external representation '`2`' are not used in the internal representation '`3`'.";


(*   Check for named patterns in the internal representation which are not being used.   - - - - - - - - - - - - - -  *)
checkNotationPatterns[
   externalBoxes_,
   externalPatterns_,
   "\[DoubleLongLeftArrow]",
   internalBoxes_,
   internalPatterns_,
   opts___
] :=
   With[
      {
         rhsPatternsNotInlhs =
            Complement[internalPatterns, externalPatterns, {singleBlank[], doubleBlank[], tripleBlank[]}]
      },
      Condition[
         $Failed,
         And[
            rhsPatternsNotInlhs =!= {},
            Message[
               Notation::inpatnu,
               (HoldForm @@ #1 & ) /@ rhsPatternsNotInlhs,
               identityForm @@ externalBoxes,
               identityForm @@ internalBoxes
            ]
         ]
      ]
   ];

Notation::inpatnu =
   "Warning: The pattern(s) '`1`' appearing in the internal representation '`2`' are not used in the external representation '`3`'.";



(*   checkNotationPatterns for patterns that are too general.  ------------------------------------------------------ *)

(*   Check for patterns that are too general in the external representation.  - - - - - - - - - - - - - - - - - - -  *)
checkNotationPatterns[externalBoxes_, externalPatterns_, type_, internalBoxes_, internalPatterns_, opts___] :=
   With[
      {
         patternedBoxes =
            boxedStringPatternsToPatterns[
               cleanBoxes @@ externalBoxes,
               WorkingForm -> WorkingForm /. {opts} /. adjustedOptions[Notation]
            ]
      },
      Condition[
         (Message[Notation::expattg, patternedBoxes, externalBoxes, identityForm[type], internalBoxes]; $Failed),
         patternToGeneralQ[patternedBoxes]
      ]
   ];

Notation::expattg =
   "The external pattern `1` appearing in Notation[`2` `3` `4`] is too general to be used. Almost anything will match the external pattern `1`.";


(*   Check for patterns that are too general in the internal representation.  - - - - - - - - - - - - - - - - - - -  *)
checkNotationPatterns[externalBoxes_, externalPatterns_, type_, internalBoxes_, internalPatterns_, opts___] :=
   With[
      {
         patternedBoxes =
            boxedStringPatternsToPatterns[
               cleanBoxes @@ internalBoxes,
               WorkingForm -> WorkingForm /. {opts} /. adjustedOptions[Notation]
            ]
      },
      Condition[
         (Message[Notation::inpattg, patternedBoxes, externalBoxes, identityForm[type], internalBoxes]; $Failed),
         patternToGeneralQ[patternedBoxes]
      ]
   ];

Notation::inpattg =
   "The internal pattern `1` appearing in Notation[`2` `3` `4`] is too general to be used. Almost anything will match the internal pattern `1`.";



(*   Definition of Notation.  --------------------------------------------------------------------------------------- *)

(*   O.K. Now create the notation.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   The notation has been correctly parsed and checked. Now create the notation.
*)
checkNotationPatterns[externalBoxes_, externalPatterns_, type_, internalBoxes_, internalPatterns_, opts___] :=
   executeNotation[externalBoxes, type, internalBoxes, opts];

allok:executeNotation[external_, "\[DoubleLongLeftRightArrow]", internal_, opts___] :=
   With[
      {internalBoxes = stripParsedBoxWrapper[internal], externalBoxes = stripParsedBoxWrapper[external]},
      (
         createExternalToInternalRule[externalBoxes, internalBoxes, opts];
         createInternalToExternalRule[internalBoxes, externalBoxes, opts];
         Null
      )
   ];

allok:executeNotation[external_, "\[DoubleLongRightArrow]", internal_, opts___] :=
   With[
      {internalBoxes = stripParsedBoxWrapper[internal], externalBoxes = stripParsedBoxWrapper[external]},
      createExternalToInternalRule[externalBoxes, internalBoxes, opts]
   ];

allok:executeNotation[external_, "\[DoubleLongLeftArrow]", internal_, opts___] :=
   With[
      {internalBoxes = stripParsedBoxWrapper[internal], externalBoxes = stripParsedBoxWrapper[external]},
      createInternalToExternalRule[internalBoxes, externalBoxes, opts]
   ];



(*   Notation error handling catchalls.  ---------------------------------------------------------------------------- *)

(*   Have encountered an unknown error in notation parsing... Report it!  - - - - - - - - - - - - - - - - - - - - -  *)
Notation[other___] :=
   ((
      Message[Notation::unknpars];
      StylePrint[identityForm[colorizeStructuralBoxes[HoldForm[Notation][other], 1]], "Output"];
      $Failed
   ));

Notation::unknpars =
   "Unknown error occurred in parsing the notation statement. Please report this to jasonh@wri.com, Please carefully examine the following box structures for spurious characters weird groupings etc.";


(*   Have encountered an unknown error in notation pattern checking... Report it!  - - - - - - - - - - - - - - - - -  *)
checkNotationPatterns[other___] :=
   ((
      Message[Notation::unknpatu];
      StylePrint[identityForm[colorizeStructuralBoxes[HoldForm[Notation][other], 1]], "Output"];
      $Failed
   ));

Notation::unknpatu =
   "Unknown error occurred in checking the pattern used in the notation statement. It appears to have parsed ok. Please report this to jasonh@wri.com, Please carefully examine the following box structures for spurious characters weird groupings etc.";


(*   Have encountered an unknown error in notation execution... Report it!  - - - - - - - - - - - - - - - - - - - -  *)

(*
   executeNotation[all_,other___] :=
    (Message[Notation::unknexec];
     CellPrint[Cell[BoxData[ colorizeStructuralBoxes[ all , 1]],"Output"]];
     HoldComplete @ Symbol @ "$Failed");
*)
Notation::unknexec =
   "Unknown error occurred in executing the notation statement. It appears to have parsed and been processed ok. Please report this to jasonh@wri.com, Please carefully examine the following box structures for spurious characters weird groupings etc.";




(* ------------------------------------------------------------------------------------------------------------------ *)
(*   Notation InternalToExternal  ----------------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------------------------------------------------ *)



(*   convertInternalPatternsForInternalToExternal  ------------------------------------------------------------------ *)
convertInternalPatternsForInternalToExternal[patternBoxes_, (opts___)?OptionQ] :=
   With[
      {WorkingFormOpt = WorkingForm /. {opts} /. adjustedOptions[Notation]},
      cleanBoxes[patternBoxes]  //.
         {
            TemplateBox[{stringPatternBoxes_}, "NotationPatternTag", ___]  :>
               (MakeBoxes[#1, WorkingFormOpt] & ) @@ toInert[toMyHeldExpression[stringPatternBoxes, WorkingFormOpt]],
            TemplateBox[{stringPatternBoxes_}, "NotationMadeBoxesTag", ___]  :>
               (MakeBoxes[#1, WorkingFormOpt] & ) @@ toInert[toMyHeldExpression[stringPatternBoxes, WorkingFormOpt]]
         }
   ];



(*   convertInternalBoxesForInternalToExternal  --------------------------------------------------------------------- *)
convertInternalBoxesForInternalToExternal[internalBoxes_, (opts___)?OptionQ] :=
   With[
      {WorkingFormOpt = WorkingForm /. {opts} /. adjustedOptions[Notation]},
      With[
         {
            internal =
               toMyHeldExpression[convertInternalPatternsForInternalToExternal[internalBoxes, opts], WorkingFormOpt]
         },
         fromInert[transformNotationalPatterns[flattenAllMyHold[myHold[NotationMakeBoxes[internal, WorkingFormOpt]]]]]
      ]
   ];


(*   toInert & fromInert  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   toInert & fromInert transform all system patterns into inert patterns and back again.
*)
toInert[expr_] :=
   expr  /.
      {
         Pattern -> Symbol[StringJoin[Context[], "inert`Pattern"]],
         PatternTest -> Symbol[StringJoin[Context[], "inert`PatternTest"]],
         Condition -> Symbol[StringJoin[Context[], "inert`Condition"]],
         Alternatives -> Symbol[StringJoin[Context[], "inert`Alternatives"]],
         Optional -> Symbol[StringJoin[Context[], "inert`Optional"]],
         Repeated -> Symbol[StringJoin[Context[], "inert`Repeated"]],
         RepeatedNull -> Symbol[StringJoin[Context[], "inert`RepeatedNull"]]
      };
fromInert[expr_] :=
   expr  /.
      {
         Symbol[StringJoin[Context[], "inert`Pattern"]] -> Pattern,
         Symbol[StringJoin[Context[], "inert`PatternTest"]] -> PatternTest,
         Symbol[StringJoin[Context[], "inert`Condition"]] -> Condition,
         Symbol[StringJoin[Context[], "inert`Alternatives"]] -> Alternatives,
         Symbol[StringJoin[Context[], "inert`Optional"]] -> Optional,
         Symbol[StringJoin[Context[], "inert`Repeated"]] -> Repeated,
         Symbol[StringJoin[Context[], "inert`RepeatedNull"]] -> RepeatedNull
      };
SetAttributes[inertHold, HoldAll];


(*   transformNotationalPatterns  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   transformNotationalPatterns transforms all patterns that are not genuine patterns and converts them to match literal 
   expressions.
*)
transformNotationalPatterns[patterns_] :=
   (patterns //. HoldPattern[HoldPattern[HoldPattern][patt_]] -> Verbatim[HoldPattern][patt])  //.
      {
         Condition[
            Condition[
               HoldPattern[HoldPattern[Pattern][patternVariable_, patternContent_]],
                !MatchQ[patternContent, HoldPattern[Blank | BlankSequence | BlankNullSequence][]]
            ],
            (Message[Notation::notapatu, HoldForm[HoldForm[Pattern][patternVariable, patternContent]]]; True)
         ]  ->
            (patt:Pattern)[patternVariable, patternContent],
         Condition[
            HoldPattern[HoldPattern[PatternTest][patt_, patternTest_]],
            (Message[Notation::notapatu, HoldForm[patt?patternTest]]; True)
         ]  ->
            HoldPattern[PatternTest][patt, HoldPattern[patternTest]],
         Condition[
            HoldPattern[HoldPattern[Condition][patt_, cond_]],
            (Message[Notation::notapatu, HoldForm[patt /; cond]]; True)
         ]  ->
            HoldPattern[HoldPattern[Condition][patt, cond]],
         Condition[
            HoldPattern[HoldPattern[Alternatives][patt__]],
            (Message[Notation::notapatu, HoldForm[Alternatives[patt]]]; True)
         ]  ->
            HoldPattern[Alternatives][patt],
         HoldPattern[HoldPattern[Optional][patt__]] /; (Message[Notation::notapatu, HoldForm[Optional[patt]]]; True)  ->
            HoldPattern[Optional][patt],
         HoldPattern[HoldPattern[Repeated][patt__]] /; (Message[Notation::notapatu, HoldForm[patt..]]; True)  ->
            HoldPattern[Repeated][patt],
         HoldPattern[HoldPattern[RepeatedNull][patt__]] /; (Message[Notation::notapatu, HoldForm[patt...]]; True)  ->
            HoldPattern[RepeatedNull][patt]
      };

Notation::notapatu =
   "Warning: The pattern `1` is being interpreted as a notation and not a pattern. Use an embedded NotationPatternTag TemplateBox wrapper if you want this pattern to be treated as a genuine pattern.";



(*   convertExternalPatternsForInternalToExternal  ------------------------------------------------------------------ *)

(*   convertExternalPatternsForInternalToExternal  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   convertExternalPatternsForInternalToExternal will take an expression consisting of boxes and convert all patterns 
   present in the boxes--patterns that are currently still unparsed strings and not yet expressions--to the corresponding 
   patterned expressions, leaving the other boxes alone.
*)
convertExternalPatternsForInternalToExternal[patternBoxes_, (opts___)?OptionQ] :=
   Module[
      {WorkingFormOpt = WorkingForm /. {opts} /. adjustedOptions[Notation]},
      flattenAllMyHold[
         Apply[
            myHold,
            {
               patternBoxes  //.
                  {
                     TemplateBox[{stringPatternBoxes_}, "NotationMadeBoxesTag", ___]  :>
                        Apply[
                           myHold,
                           removePatternsAndBlanks[
                              complexPattern @@ toMyHeldExpression[stringPatternBoxes, WorkingFormOpt]
                           ]
                        ],
                     TemplateBox[{stringPatternBoxes_}, "NotationPatternTag", ___]  :>
                        removePatternsAndBlanks[
                           complexPattern @@ toMyHeldExpression[stringPatternBoxes, WorkingFormOpt]
                        ],
                     TemplateBox[{stringPatternBoxes_}, "NotationTemplateTag", ___] :> stringPatternBoxes,
                     string_String /; StringMatchQ[string, "*_"]  :>
                        convertPatterns[toMyHeldExpression[string, WorkingFormOpt]]
                  }
            }
         ]
      ]
   ];


(*   Error checking for convertExternalPatternsForInternalToExternal  - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   The second argument and beyond of convertExternalPatternsForInternalToExternal must be Options.
*)
error:convertExternalPatternsForInternalToExternal[_, ___, (notOption_)?(isNot[_?OptionQ]), ___] :=
   $Failed /; Message[convertExternalPatternsForInternalToExternal::nonopt, HoldForm[notOption], 1, HoldForm[error]];



(*   convertExternalBoxesForInternalToExternal  --------------------------------------------------------------------- *)

(*
   I might be able to get away with using myHold instead of myOtherHold but to be safe use myOtherHold since the 
   expression might already contain myHold and I am not imediatly familary enough with the my own code (which I haven\[CloseCurlyQuote]t 
   changed for quite a while...) so play it safe here.
*)
SetAttributes[{myOtherHold, releaseMyHold}, HoldAllComplete];
releaseMyOtherHold[expr___] := Evaluate @@ (HoldComplete[expr] //. myOtherHold[term___] -> term);

convertExternalBoxesForInternalToExternal[externalBoxes_, (opts___)?OptionQ] :=
   releaseMyOtherHold[
      Module[
         {CommaQ},
         (
            CommaQ["," | "\[InvisibleComma]"] := True;
            CommaQ[other_] := False;
            With[
               {
                  RB = RowBox,
                  B = singleBlank,
                  BB = doubleBlank | tripleBlank,
                  CP = complexPattern,
                  OpQ = operatorQ,
                  InQ = infixOperatorQ,
                  PreQ = prefixOperatorQ,
                  PostQ = postfixOperatorQ,
                  DelimQ = delimiterQ,
                  EffOp = effectiveOperator,
                  Paren = Parenthesize,
                  WhQ = whiteSpaceQ,
                  MH = myOtherHold,
                  form = WorkingForm /. {opts} /. adjustedOptions[Notation]
               },
               convertExternalPatternsForInternalToExternal[tidyBoxes[externalBoxes], opts]  //.
                  {
                     RB[{l___, (op_)?PreQ, (w___)?WhQ, B[symb_], r___}]  :>
                        With[{prec = EffOp[op]}, MH[RB[{l, op, w, Paren[symb, form, prec], r}]] /; True],
                     RB[{l___, B[symb_], (w___)?WhQ, (op_)?InQ, r___}]  :>
                        With[{prec = EffOp[op]}, MH[RB[{l, Paren[symb, form, prec], w, op, r}]] /; True],
                     RB[{l___, (op_)?InQ, (w___)?WhQ, B[symb_], r___}]  :>
                        With[{prec = EffOp[op]}, MH[RB[{l, op, w, Paren[symb, form, prec], r}]] /; True],
                     RB[{l___, B[symb_], (w___)?WhQ, (op_)?PostQ, r___}]  :>
                        With[{prec = EffOp[op]}, MH[RB[{l, Paren[symb, form, prec], w, op, r}]] /; True],
                     RB[{l___, (d_)?DelimQ, m___, B[symb_], r___}] :> RB[{l, d, m, MakeBoxes[symb, form], r}],
                     RB[{l___, B[symb_], m___, (d_)?DelimQ, r___}] :> RB[{l, MakeBoxes[symb, form], m, d, r}],
                     RB[{l___, B[symb_], r___}] :> RB[{l, Paren[symb, form, Times], r}] /;  !{l, r} === {},
                     B[symb_] :> MakeBoxes[symb, form],
                     RB[{l___, (d_)?CommaQ, (w___)?WhQ, BB[symb_], r___}]  :>
                        RB[{l, d, w, makeHeldSequenceOfBoxes[{symb}, form, None], r}],
                     RB[{l___, BB[symb_], (w___)?WhQ, (d_)?CommaQ, r___}]  :>
                        RB[{l, makeHeldSequenceOfBoxes[{symb}, form, None], w, d, r}],
                     RB[{l___, (d_)?DelimQ, (w___)?WhQ, BB[symb_], r___}]  :>
                        RB[{l, d, w, makeHeldRowBoxOfBoxes[{symb}, form, None], r}],
                     RB[{l___, BB[symb_], (w___)?WhQ, (d_)?DelimQ, r___}]  :>
                        RB[{l, makeHeldRowBoxOfBoxes[{symb}, form, None], w, d, r}],
                     BB[symb_] :> makeHeldRowBoxOfBoxes[{symb}, form, None],
                     RB[{l___, (d_)?CommaQ, (w___)?WhQ, CP[expr_], r___}]  :>
                        RB[{l, d, w, makeEvaluatedSequenceOfBoxes[{expr}, form, None], r}],
                     RB[{l___, CP[expr_], (w___)?WhQ, (d_)?CommaQ, r___}]  :>
                        RB[{l, makeEvaluatedSequenceOfBoxes[{expr}, form, None], w, d, r}],
                     RB[{l___, (d_)?DelimQ, (w___)?WhQ, CP[expr_], r___}]  :>
                        RB[{l, d, w, makeEvaluatedRowBoxOfBoxes[{expr}, form, None], r}],
                     RB[{l___, CP[expr_], (w___)?WhQ, (d_)?DelimQ, r___}]  :>
                        RB[{l, makeEvaluatedRowBoxOfBoxes[{expr}, form, None], w, d, r}],
                     CP[patt_] :> makeEvaluatedRowBoxOfBoxes[{patt}, form, None]
                  }
            ]
         )
      ]
   ];


(*   The 2nd argument and beyond of convertExternalBoxesForInternalToExternal must be Options.  - - - - - - - - - -  *)
error:convertExternalBoxesForInternalToExternal[_, ___, (notOption_)?(isNot[_?OptionQ]), ___] :=
   $Failed /; Message[convertExternalBoxesForInternalToExternal::nonopt, HoldForm[notOption], 1, HoldForm[error]];



(*   Definition of createInternalToExternalRule  -------------------------------------------------------------------- *)

(*   Creates appropriate MakeBoxes rule  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   This creates a MakeBoxes rule that will format an internal expression into an external box structure.
*)
createInternalToExternalRule[internalBoxes_, externalBoxes_, opts___] :=
   With[
      {
         external = convertExternalBoxesForInternalToExternal[externalBoxes, opts],
         internal = convertInternalBoxesForInternalToExternal[internalBoxes, opts],
         WorkingFormOpt = WorkingForm /. {opts} /. adjustedOptions[Notation],
         ActionOpt = Action /. {opts} /. adjustedOptions[Notation]
      },
      (executeInternalToExternalAction[internal, external, WorkingFormOpt, ActionOpt]; )
   ];


(*   Enter, remove or print rule  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   This determines if the rule should be entered, removed or printed.
*)
executeInternalToExternalAction[internal_, external_, WorkingFormOpt_, CreateNotationRules] :=
   releaseMyHold[internal := external];
executeInternalToExternalAction[internal_, external_, WorkingFormOpt_, RemoveNotationRules] :=
   releaseMyHold[silentEvaluate[internal =. ]];
executeInternalToExternalAction[internal_, external_, WorkingFormOpt_, PrintNotationRules] :=
   releaseMyHold[
      CellPrint[Cell[BoxData[MakeBoxes[internal := external, StandardForm]], "Output", ShowStringCharacters -> True]]
   ];




(* ------------------------------------------------------------------------------------------------------------------ *)
(*   Notation ExternalToInternal  ----------------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------------------------------------------------ *)



(*   convertInternalPatternsForExternalToInternal  ------------------------------------------------------------------ *)

(*   convertInternalPatternsForExternalToInternal  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   convertInternalPatternsForExternalToInternal will take an expression consisting of boxes and convert all patterns 
   present in the boxes--patterns that are currently still unparsed strings and not yet expressions--to the corresponding 
   naked patterned variables, stripping out the pattern content and leaving all other boxes alone.
*)
convertInternalPatternsForExternalToInternal[patternBoxes_, (opts___)?OptionQ] :=
   convertPatterns[flattenAllMyHold[boxedStringPatternsToPatterns[patternBoxes, opts]]]  //.
      {
         singleBlank[a_] -> a,
         doubleBlank[a_] :> stripSpuriousRowBox[a],
         tripleBlank[a_] :> stripSpuriousRowBox[a],
         complexPattern[a_] :> stripSpuriousRowBox[a]
      };


(*   Error checking for convertInternalPatternsForExternalToInternal  - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   The 2nd argument and beyond of convertInternalPatternsForExternalToInternal must be Options.
*)
error:convertInternalPatternsForExternalToInternal[_, ___, (notOption_)?(isNot[_?OptionQ]), ___] :=
   $Failed /; Message[convertInternalPatternsForExternalToInternal::nonopt, HoldForm[notOption], 1, HoldForm[error]];

stripSpuriousRowBox[RowBox[{args___}]] /; MemberQ[{args}, "," | "\[InvisibleComma]"] := args;
stripSpuriousRowBox[other___] := other;



(*   convertInternalBoxesForExternalToInternal  --------------------------------------------------------------------- *)
convertInternalBoxesForExternalToInternal[internalBoxes_, headIsRowBox_, (opts___)?OptionQ] :=
   With[
      {
         internal = convertInternalPatternsForExternalToInternal[cleanBoxes[internalBoxes], opts],
         WorkingFormOpt = WorkingForm /. {opts} /. adjustedOptions[Notation]
      },
      If[
         headIsRowBox,
         myHold[MakeExpression[RowBox[{lhs, internal, rhs}], WorkingFormOpt]],
         myHold[MakeExpression[internal, WorkingFormOpt]]
      ]
   ];



(*   convertExternalPatternsForExternalToInternal  ------------------------------------------------------------------ *)

(*   convertExternalPatternsForExternalToInternal  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   convertExternalPatternsForExternalToInternal will take an expression consisting of boxes and convert all patterns 
   present in the boxes--patterns that are currently still unparsed strings and not yet expressions--to the corresponding 
   patterned expressions, leaving the other boxes alone.
*)
convertExternalPatternsForExternalToInternal[patternBoxes_, (opts___)?OptionQ] :=
   boxedStringPatternsToPatterns[patternBoxes, opts];


(*   Error checking for convertInternalPatternsForExternalToInternal  - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   The 2nd argument and beyond of convertExternalPatternsForExternalToInternal must be Options.
*)
error:convertExternalPatternsForExternalToInternal[_, ___, (notOption_)?(isNot[_?OptionQ]), ___] :=
   $Failed /; Message[convertExternalPatternsForExternalToInternal::nonopt, HoldForm[notOption], 1, HoldForm[error]];



(*   convertExternalBoxesForExternalToInternal  --------------------------------------------------------------------- *)
convertExternalBoxesForExternalToInternal[externalBoxes_, headIsRowBox_, (opts___)?OptionQ] :=
   With[
      {
         external = convertExternalPatternsForExternalToInternal[cleanBoxes[externalBoxes], opts],
         WorkingFormOpt = WorkingForm /. {opts} /. adjustedOptions[Notation]
      },
      If[
         headIsRowBox,
         With[
            {newexternal = RowBox[{lhs___, Sequence @@ Sequence @@ external, rhs___}]},
            myHold[NotationMakeExpression[newexternal, WorkingFormOpt]]
         ],
         myHold[NotationMakeExpression[external, WorkingFormOpt]]
      ]
   ];



(*   Definition of createExternalToInternalRule  -------------------------------------------------------------------- *)

(*   Creates appropriate MakeExpression rule  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   This creates a MakeExpression rule that will parse an external box structure into an internal box structure.
*)
createExternalToInternalRule[externalBoxes_, internalBoxes_, opts___] :=
   With[
      {headIsRowBox = Head[convertExternalPatternsForExternalToInternal[cleanBoxes[externalBoxes]]] === RowBox},
      With[
         {
            internal = convertInternalBoxesForExternalToInternal[internalBoxes, headIsRowBox, opts],
            external = convertExternalBoxesForExternalToInternal[externalBoxes, headIsRowBox, opts],
            WorkingFormOpt = WorkingForm /. {opts} /. adjustedOptions[Notation],
            ActionOpt = Action /. {opts} /. adjustedOptions[Notation]
         },
         (executeExternalToInternalAction[external, internal, WorkingFormOpt, ActionOpt]; )
      ]
   ];


(*   Enter, remove or print rule  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   This determines if the rule should be entered, removed or printed.
*)
executeExternalToInternalAction[external_, internal_, WorkingFormOpt_, CreateNotationRules] :=
   releaseMyHold[external := internal];

executeExternalToInternalAction[external_, internal_, WorkingFormOpt_, RemoveNotationRules] :=
   releaseMyHold[external =. ];

executeExternalToInternalAction[external_, internal_, WorkingFormOpt_, PrintNotationRules] :=
   releaseMyHold[
      CellPrint[Cell[BoxData[MakeBoxes[external := internal, StandardForm]], "Output", ShowStringCharacters -> True]]
   ];




(* ------------------------------------------------------------------------------------------------------------------ *)
(*   Utilities for the Package and Cleanups  ------------------------------------------------------------------------ *)
(* ------------------------------------------------------------------------------------------------------------------ *)



(*   ClearNotations  ------------------------------------------------------------------------------------------------ *)

(*   ClearNotations[]  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   This function removes all definitions for MakeBoxes and MakeExpression, leaving only the definitions of this 
   package, Notation.m. Use this function to reset the notation handling to a pristine state.
*)
ClearNotations[] :=
   ((
      Clear[NotationMakeExpression, NotationMakeBoxes];
      NotationMakeExpression[TagBox[boxes_, "NotationTemplateTag", opts___], anyForm_] :=
         HoldComplete[ParsedBoxWrapper[boxes]];
      NotationMakeExpression[TagBox[boxes_, NotationBoxTag, opts___], anyForm_] := HoldComplete[ParsedBoxWrapper[boxes]];
      NotationMakeBoxes[ParsedBoxWrapper[boxes__], anyForm_] := TemplateBox[{boxes}, "NotationTemplateTag"];
      NotationMakeBoxes[identityForm[any___], anyForm_] := any;
      Null
   ));


(*   Error Checking for ClearNotations.  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   ClearNotations expects no arguments.
*)
error:ClearNotations[_] := $Failed /; Message[ClearNotations::argr, HoldForm[ClearNotations], 0];
error:ClearNotations[_, __] := $Failed /; Message[ClearNotations::argrx, HoldForm[ClearNotations], heldLength[error], 0];



(*   updateInputAliases  -------------------------------------------------------------------------------------------- *)

(*   updateInputAliases[]  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   UpdateInputAliases will ensure the Notation aliases are part of the system and of the given stylesheet. It will also 
   add aliases for Notation, Symbolize, InfixNotation, and AddInputAlias. 
*)
updateInputAliases[] :=
   AddInputAlias[
      {
         "notation"  ->
            Apply[
               ParsedBoxWrapper,
               {
                  RowBox[
                     {
                        "Notation",
                        "[",
                        RowBox[
                           {
                              TemplateBox[{"\[SelectionPlaceholder]"}, "NotationTemplateTag"],
                              " ",
                              "\[DoubleLongLeftRightArrow]",
                              " ",
                              TemplateBox[{"\[Placeholder]"}, "NotationTemplateTag"]
                           }
                        ],
                        "]"
                     }
                  ]
               }
            ],
         "notation>"  ->
            Apply[
               ParsedBoxWrapper,
               {
                  RowBox[
                     {
                        "Notation",
                        "[",
                        RowBox[
                           {
                              TemplateBox[{"\[SelectionPlaceholder]"}, "NotationTemplateTag"],
                              " ",
                              "\[DoubleLongRightArrow]",
                              " ",
                              TemplateBox[{"\[Placeholder]"}, "NotationTemplateTag"]
                           }
                        ],
                        "]"
                     }
                  ]
               }
            ],
         "notation<"  ->
            Apply[
               ParsedBoxWrapper,
               {
                  RowBox[
                     {
                        "Notation",
                        "[",
                        RowBox[
                           {
                              TemplateBox[{"\[SelectionPlaceholder]"}, "NotationTemplateTag"],
                              " ",
                              "\[DoubleLongLeftArrow]",
                              " ",
                              TemplateBox[{"\[Placeholder]"}, "NotationTemplateTag"]
                           }
                        ],
                        "]"
                     }
                  ]
               }
            ],
         "symb"  ->
            Apply[
               ParsedBoxWrapper,
               {RowBox[{"Symbolize", "[", TemplateBox[{"\[SelectionPlaceholder]"}, "NotationTemplateTag"], "]"}]}
            ],
         "infixnotation"  ->
            Apply[
               ParsedBoxWrapper,
               {
                  RowBox[
                     {
                        "InfixNotation",
                        "[",
                        RowBox[{TemplateBox[{"\[SelectionPlaceholder]"}, "NotationTemplateTag"], ",", "\[Placeholder]"}],
                        "]"
                     }
                  ]
               }
            ],
         "addia"  ->
            Apply[
               ParsedBoxWrapper,
               {
                  RowBox[
                     {
                        "AddInputAlias",
                        "[",
                        RowBox[
                           {
                              "\"\[SelectionPlaceholder]\"",
                              "\[Rule]",
                              TemplateBox[{"\[Placeholder]"}, "NotationTemplateTag"]
                           }
                        ],
                        "]"
                     }
                  ]
               }
            ],
         "pattwraper" -> ParsedBoxWrapper @@ {TemplateBox[{"\[SelectionPlaceholder]"}, "NotationPatternTag"]},
         "madeboxeswraper" -> ParsedBoxWrapper @@ {TemplateBox[{"\[SelectionPlaceholder]"}, "NotationMadeBoxesTag"]}
      },
      $FrontEnd
   ];




(* ------------------------------------------------------------------------------------------------------------------ *)
(*   Package Endings  ----------------------------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------------------------------------------------ *)



(*   Establish Autointeractive values  ------------------------------------------------------------------------------ *)

(*
   If any of these values are undefined they default to True.
*)
If[ !ValueQ[AutoLoadNotationPalette], AutoLoadNotationPalette = True];



(*   Open the Palettes  --------------------------------------------------------------------------------------------- *)

(*
   We try to obtain a localized version of the palettes if one is available, but otherwise we fall back to English 
   palettes.
*)
getLocalizedPalettesFilePath[paletteName_String] :=
   With[
      {dir = ToFileName[{$TopDirectory, "AddOns", "Packages", "Notation", "LocalPalettes", $Language}]},
      ToFileName[dir, paletteName] /; FileNames[paletteName, {dir}] =!= {}
   ];
getLocalizedPalettesFilePath[paletteName_String] :=
   With[
      {dir = ToFileName[{$TopDirectory, "AddOns", "Packages", "Notation", "LocalPalettes", "English"}]},
      ToFileName[dir, paletteName]
   ];


(*
   Open the Notation palette by default. Note with remote kernels NotebookOpen does not work properly. Make sure the 
   main notebook remains selected.
*)
If[
   AutoLoadNotationPalette === True && Notebooks["Notation Palette"] === {},
   Module[
      {nb = InputNotebook[], filePath = getLocalizedPalettesFilePath["NotationPalette.nb"]},
      (If[$Linked, NotebookPut[Get[filePath]], NotebookOpen[filePath]]; SetSelectedNotebook[nb]; )
   ]
];



(*   Update the input aliases if necessary  ------------------------------------------------------------------------- *)

(*   Update input aliases if necessary  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

(*
   Finally, ensure the Notation aliases are present in the system.
*)
updateInputAliases[];


(*   Warn if the current default input and output forms differ  - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
Notation::difform =
   "The defined current default input format `1` differs from the current default output format `2`. The WorkingForm option will default to the current default output format, but the Notations, Symbolizations, and InfixNotations may behave differently than expected.";

Notation::notintypesetform =
   "The defined current default input format `1` is not a typeset form. Notations, Symbolizations, and InfixNotations do not work with `1` since this form is not a typeset form.";

Notation::notouttypesetform =
   "The defined current default output format `1` is not a typeset form. Notations, Symbolizations, and InfixNotations do not work with `1` since this form is not a typeset form.";

With[
   {
      inForm = AbsoluteCurrentValue[InputNotebook[], {CommonDefaultFormatTypes, "Input"}],
      outForm = AbsoluteCurrentValue[InputNotebook[], {CommonDefaultFormatTypes, "Output"}]
   },
   (
      If[inForm =!= outForm, Message[Notation::difform, inForm, outForm]];
      If[inForm == InputForm || inForm == OutputForm, Message[Notation::notintypesetform, inForm]];
      If[outForm == InputForm || outForm == OutputForm, Message[Notation::notouttypesetform, outForm]];
      Null
   )
];



(*   End the package  ----------------------------------------------------------------------------------------------- *)

(*   End Private  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
End[];


(*   Protect Notation functions  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
SetAttributes[
   {
      ClearNotations,
      InfixNotation,
      Notation,
      ParsedBoxWrapper,
      NotationBoxTag,
      RemoveInfixNotation,
      RemoveNotation,
      RemoveSymbolize,
      Symbolize,
      AddInputAlias,
      ActiveInputAliases
   },
   {ReadProtected, Protected}
];

SetAttributes[
   {
      Action,
      CreateNotationRules,
      NotationPatternTag,
      NotationMadeBoxesTag,
      PrintNotationRules,
      RemoveNotationRules,
      SymbolizeRootName,
      WorkingForm
   },
   {Protected}
];


(*   End the Package  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
EndPackage[];


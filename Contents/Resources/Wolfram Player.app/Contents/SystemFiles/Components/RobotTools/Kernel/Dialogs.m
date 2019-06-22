(* ::Package:: *)

(* ::Title:: *)
(*Dialogs*)

(* ::Section:: *)
(*Information*)

(* :Title: Dialogs.m *)

(* :Author:
        Brenton Bostick
        brenton@wolfram.com
*)

(* :Package Version: 1.2 *)

(* :Mathematica Version: 7.0 *)

(* :Copyright: RobotTools source code (c) 2005-2008 Wolfram Research, Inc. All rights reserved. *)

(* :Discussion:
   Implementation of DialogAction.
*)

`Information`CVS`$DialogsRevision = "$Revision: 1.8 $"

(* ::Section:: *)
(*Public*)

DialogAction::usage =
"DialogAction[\"dialog\", action] executes a sequence of commands to perform the action for the dialog."

(* ::Section:: *)
(*Package*)

Begin["`Package`"]

End[] (*`Package`*)

(* ::Section:: *)
(*Private*)

(* ::Subsection:: *)
(*Begin*)

Begin["`Dialogs`"]


(*
DialogAction will support Methods:
Automatic - do simplest thing possible
Mnemonic - use Mnemonics
*)

Options[DialogAction] = {Method -> Automatic}

DialogAction[args___] :=
	(ArgumentCountQ[DialogAction, System`FEDump`NonOptionArgCount[{args}], 2, 3]; Null /; False)


DialogAction[nb:menuNotebookPat:MenuNotebook[], "BackgroundDialog", "OK"] /; $InterfaceEnvironment == "Windows" :=
	KeyType[nb, "\[EnterKey]"]

DialogAction[nb:menuNotebookPat:MenuNotebook[], "BackgroundDialog", "Cancel"] /; $InterfaceEnvironment == "Windows" :=
	KeyType[nb, "\[EscapeKey]"]

DialogAction[nb:menuNotebookPat:MenuNotebook[], "BackgroundDialog", "Close"] /; $InterfaceEnvironment == "Windows" :=
	KeyType[nb, "\[EscapeKey]"]


DialogAction[nb:menuNotebookPat:MenuNotebook[], "YesNoDialog", "Yes"] /; $InterfaceEnvironment == "Macintosh" :=
	KeyType[nb, "\[EnterKey]"]

DialogAction[nb:menuNotebookPat:MenuNotebook[], "YesNoDialog", "No"] /; $InterfaceEnvironment == "Macintosh" :=
	KeyType[nb, "\[TabKey]\[EnterKey]"]

DialogAction[nb:menuNotebookPat:MenuNotebook[], "YesNoDialog", "Yes"] /; $InterfaceEnvironment == "Windows" :=
	KeyType[nb, "y"]

DialogAction[nb:menuNotebookPat:MenuNotebook[], "YesNoDialog", "No"] /; $InterfaceEnvironment == "Windows" :=
	KeyType[nb, "n"]
	
DialogAction[nb:menuNotebookPat:MenuNotebook[], "YesNoDialog", "Yes"] /; $InterfaceEnvironment == "X" :=
	KeyType[nb, "\[EnterKey]"]

DialogAction[nb:menuNotebookPat:MenuNotebook[], "YesNoDialog", "No"] /; $InterfaceEnvironment == "X" :=
	KeyType[nb, "\[TabKey]\[EnterKey]"]


DialogAction[nb:menuNotebookPat:MenuNotebook[], "CellContextDialog", "Help"] /; $InterfaceEnvironment == "Windows" :=
	KeyType[nb, "h"]

DialogAction[nb:menuNotebookPat:MenuNotebook[], "CellContextDialog", "Cancel"] /; $InterfaceEnvironment == "Windows" :=
	KeyType[nb, "\[EscapeKey]"]

DialogAction[nb:menuNotebookPat:MenuNotebook[], "CellContextDialog", {context_String, "Set"}] /; $InterfaceEnvironment == "Windows" :=
	KeyType[nb, "\[AltKey]\[LeftModified]n\[RightModified]" <> context <> "\[AltKey]\[LeftModified]s\[RightModified]"]

(* ::Subsection:: *)
(*Arguments*)

(* ::Subsection:: *)
(*SyntaxInformation*)

SyntaxInformation[DialogAction] = {"ArgumentsPattern" -> {_., _, _, OptionsPattern[]}}

(* ::Subsection:: *)
(*End*)

End[] (*`Dialogs`*)

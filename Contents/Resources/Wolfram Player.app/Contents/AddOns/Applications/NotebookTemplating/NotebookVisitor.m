 (* Mathematica Package *)

(* :Title: NotebookVisitor.m *)

(* :Authors:
	Anthony Zupnik anthonyz@wolfram.com	
*)

(* :Mathematica Version: 10.3 *)
                     
(* :Copyright: (c) 2016, Wolfram Research, Inc. All rights reserved. *)

(* :Requirements: *)

(* :Discussion:
  
*)

BeginPackage["NotebookTemplating`NotebookVisitor`"]

NotebookVisit::usage = "Visit the Notebook down to the Cell level to apply a function on either the Group or Cell level"
IdentityFunctionGroup::usage = "Used as a groupfunction to return Identity of CellGroup and Data"
IdentityFunctionCell::usage = "Used as a cellFunction to return Identity of the Cell"

Begin["`Private`"]

(*
	Notes on Using the NotebookVisitor:
	Use NotebookVisit to apply a function at either the CellGroup or Cell level for a Notebook.
	Examples:
	NotebookVisit[nb, IdentityFunctionGroup, #1/.{1->2}&] will replace any 1 with a 2 at the Cell Level
*)

(*Notebook Level Operations*)
NotebookVisit[ nb_NotebookObject, groupFunction_, cellFunction_, data_] := 
	Module[{ nbExpr},
		nbExpr=NotebookGet[nb];	
		NotebookVisit[nbExpr,groupFunction,cellFunction,data]
	]

NotebookVisit[ nbExpr:Notebook[cellList_List, opts___], groupFunction_, cellFunction_, data_] :=
	Notebook[NotebookVisit[#,groupFunction,cellFunction,<|"UserData" -> data|>]&/@cellList, opts]

NotebookVisit[ cellList:{__Cell}, groupFunction_, cellFunction_, data_] := 
	NotebookVisit[#,groupFunction,cellFunction,Append[data, "ContentData" -> BoxData]]&/@ cellList

(*CellGroup Level Operations*)
NotebookVisit[ Cell[CellGroupData[cellList_List, opts___],opts2___], groupFunction_, cellFunction_, data_] :=
	Module[{newCellListAndData},
		(*Apply to the Cell List...*)
		newCellListAndData = groupFunction[cellList,data];
		Sequence@@Map[Cell[CellGroupData[NotebookVisit[#[[1]](*CellData*),groupFunction,cellFunction,
				Append[#[[2]](*Data*), "ContentData" -> BoxData]], opts],opts2]&,newCellListAndData]
	]

(*Cell Level Operations*)

(*BoxData Cells*)
NotebookVisit[ cell:Cell[BoxData[boxData_],opts___], groupFunction_, cellFunction_, data_] :=
	cellFunction[cell,Append[data, "ContentData" -> BoxData]]

(*TextData Cells*)	
NotebookVisit[ cell:Cell[TextData[contents_List],opts___], groupFunction_, cellFunction_, data_] := 
	cellFunction[cell,Append[data, "ContentData" -> TextData]]

NotebookVisit[ cell:Cell[TextData[contents_],opts___], groupFunction_, cellFunction_, data_]:= 
	cellFunction[cell,Append[data, "ContentData" -> TextData]]

NotebookVisit[ cell:Cell[str_String,opts___], groupFunction_, cellFunction_, data_]:=
	cellFunction[cell,Append[data, "ContentData" -> TextData]]

NotebookVisit[ x___, groupFunction_, cellFunction_, data_]:=(x)

IdentityFunctionGroup:={{#1,#2}}& (*Used in a groupFunction to return the CellGroup Identity*)
IdentityFunctionCell:#1& (*Used in a cellFunction to return Cell Identity*)

End[]

EndPackage[]
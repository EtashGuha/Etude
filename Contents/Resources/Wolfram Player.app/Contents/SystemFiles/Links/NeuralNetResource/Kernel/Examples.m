(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["NeuralNetResource`"]

Begin["`Private`"] (* Begin Private Context *) 
ResourceSystemClient`Private`repositoryCreateBlankExampleNotebook[$NeuralNetResourceType,id_,name_]:=nnNewExampleNotebook[resourceInfo[id],id,name]


nnExampleHeaderCellGroup[info_] := CellGroupData[nnExampleHeaderCells[info], True]

nnExampleHeaderCells[info_] := {
  Cell[info["Name"], "ResourceExampleTitle", 
   FontFamily -> "Source Sans Pro SemiBold", FontSize -> 32, 
   FontColor -> RGBColor[{31/85, 146/255, 11/85}], 
   ShowCellBracket -> False, ShowAutoSpellCheck -> False,
   CellMargins -> {{27, Inherited}, {10, 30}},
   CellGroupingRules -> {"TitleGrouping", 0},
   PageBreakBelow -> False, LineSpacing -> {1, 4},
   MenuSortingValue -> 1100, MenuCommandKey -> "1", 
   FontTracking -> -0.5],
  Cell[info["Description"], "ResourceExampleSubtitle", 
   FontFamily -> "Source Sans Pro Light", FontSize -> 26, 
   FontSlant -> Italic, FontColor -> RGBColor[{1/3, 1/3, 1/3}], 
   ShowCellBracket -> False, ShowAutoSpellCheck -> False, 
   CellMargins -> {{27, Inherited}, {0, 0}}, 
   CellGroupingRules -> {"TitleGrouping", 10}, PageBreakBelow -> False,
   CounterIncrements -> "Subtitle", MenuSortingValue -> 1550]
  }

nnNewExampleNotebook[info_Association,id_String,name_String]:=NotebookPut[
 Notebook[{nnExampleHeaderCellGroup[info], 
   Cell["Resource Retrieval", "Subsection"],
   Cell["Retrieve the resource object:", "Text"],
   Cell[BoxData[RowBox[{"ResourceObject", "[", ToBoxes[name], "]"}]], "Input"]
  }]]

nnNewExampleNotebook[___]:=$Failed


End[] (* End Private Context *)

EndPackage[]



SetAttributes[{},
   {ReadProtected, Protected}
];
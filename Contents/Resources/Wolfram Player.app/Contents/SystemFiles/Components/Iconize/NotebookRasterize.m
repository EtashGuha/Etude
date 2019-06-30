Package["Iconize`"]
PackageScope["makeIconNotebook"]

(*
	This is a package written by larrya to intelligently handle 
	rasterization of notebooks.  Given an arbitrary notebook,
	makeIconNotebook will return a new notebook expression that:
	* Is derived from an input Notebook expression and a specified WindowSize.
	* Includes only those cells that would display at the specified WindowSize. 
		All other cells are removed to reduce the size of the notebook.
	* Can be used to create a thumbnail bitmap/file icon.
	
	The motivation for this function was to handle cases where
	Rasterize will fail because of out of memory errors.  
	
*)

$BoxesWithImageSizes = {
  ActionMenuBox,
  AnimatorBox,
  ButtonBox,
  ColorSetterBox,
  FrameBox,
  Graphics3DBox,
  GraphicsBox,
  InputFieldBox,
  GraphicsComplex3DBox,
  GraphicsComplexBox,
  GraphicsGroup3DBox,
  GraphicsGroupBox,
  LocatorPaneBox,
  OverlayBox,
  PaneBox,
  PanelBox,
  PaneSelectorBox,
  PopupMenuBox,
  ProgressIndicatorBox,
  SetterBox,
  Slider2DBox,
  SliderBox};

(*getNotebook*)

getNotebook[filepath_] := Get[filepath]

(*

	Remove Invisible Cells (i.e., remove those cells that are Closed, hidden in a Closed CellGroup,\[Ellipsis])
	Remove the Cell wrapper that surrounds CellGroupData
*)


extractCellsFromCellGroupData[Cell[cgd_CellGroupData]] := extractCellsFromCellGroupData[cgd]
extractCellsFromCellGroupData[Cell[cgd_CellGroupData]] := extractCellsFromCellGroupData[cgd]


(*Extract Only Cells That Are Not Hidden Inside a Closed CellGroup*)


extractVisibleCell[cl_Cell] := cl
extractVisibleCell[Cell[cgd_CellGroupData]] := extractCellsFromGroup[cgd]


extractCellsFromGroup[CellGroupData[cellLst_List, _]] := Map[extractVisibleCell, cellLst]
extractCellsFromGroup[CellGroupData[cellLst_List, Closed]] := extractVisibleCell[First[cellLst]]
extractCellsFromGroup[CellGroupData[cellLst_List, {part_}]] := extractVisibleCell[Part[cellLst, part]]

(*getCellHeight*)

getCellHeight[cl_Cell] := getRasterizeBBHeight[cl]
getCellHeight[cl:Cell[_BoxData, ___]] := getBoxCellHeight[cl]


getRasterizeBBHeight[cl_Cell, rasterOpts:OptionsPattern[]] := Module[{rasterDimens = Quiet[Rasterize[cl, "BoundingBox", rasterOpts]]}, 
   Replace[rasterDimens, {{wd_, ht_, dh_} :> ht, _ :> $Failed}]];
getRasterizeBBHeight[cl_] := $Failed;

(*PreProcess BoxData Cell (to try to improve efficiency)*)

getBoxCellHeight[cl:Cell[BoxData[box_], ___]] := With[{imageSizeBoxes = $BoxesWithImageSizes}, 
  Module[{
      imageHeight = If[MemberQ[imageSizeBoxes, Head[box]], 
        getImageHeightFromBoxes[box], 
        $Failed
      ]
    },
    If[imageHeight === $Failed,
      imageHeight = getRasterizeBBHeight[cl]
    ];
    imageHeight
  ]
]

(*imgSz must return a numerical height as the second arg. of a 2-arg list.*)

(*imgSz expected return value of the form {{imgwidth_,imgheight_}}*)

getImageHeightFromBoxes[box_] := Module[{
    imgSz = Cases[box, _[ImageSize, size_] :> size, 1]
  }, 
  If[MatchQ[imgSz, {{_, _?NumberQ}..}], 
    Part[imgSz, 1, 2], 
    $Failed
  ]
]

(*Accumulate Cells until Max Icon Height is reached, or Cell List Is Exhausted*)

accumulateCells[cells_List, iconHeight_, maxDisplayHeight_] := 
 Module[
  	{
       trimmedClList = Flatten[extractVisibleCell[#] & /@ cells], 
       iconNBCells = {}, 
       htAccumulated = 0, 
       clHt
     },
    
    Catch[
       Map[
          (
              clHt = getCellHeight[#];
      
              If[clHt === $Failed, Throw[$Failed]];
      
              htAccumulated += clHt;
              AppendTo[iconNBCells, #];
      
              If[htAccumulated > maxDisplayHeight, Throw[$Failed]];
      
              If[htAccumulated > iconHeight, Throw[iconNBCells]]
            ) &,
          trimmedClList];
   
       Throw[iconNBCells]
     ]
  ]


setNBPageWidth[nb:Notebook[__, _[PageWidth, _?(NumberQ[#]||# === Infinity&)], ___], _] := nb
setNBPageWidth[nb:Notebook[__, _[PageWidth, _], ___], defaultPageWidth_] := Replace[nb, _[pw:PageWidth, _] :> (pw -> defaultPageWidth), {1}]
setNBPageWidth[nb:Notebook[__, _[WindowSize, Alternatives[windowWd_?(NumberQ[#]||# === Infinity&), {windowWd_?(NumberQ[#]||# === Infinity&), _}]], ___], _] := Append[nb, PageWidth -> windowWd]
setNBPageWidth[nb_Notebook, defaultPageWidth_] := Append[nb, PageWidth -> defaultPageWidth]

Options[makeIconNotebook] = {"LargeCellHeightLimit" -> 4000, "DefaultPageWidth" -> 800};

makeIconNotebook[nbX_Notebook, iconWidth_?(NumberQ[#] && (# > 0) &), iconHeight_?(NumberQ[#] && (# > 0) &), mINbOpts:OptionsPattern[]] := Module[{
    cellLst, 
    nbXWithPageWidth, 
    lgClHtLim = "LargeCellHeightLimit"/.{mINbOpts}/.Options[makeIconNotebook], 
    defaultPageWidth = "DefaultPageWidth"/.{mINbOpts}/.Options[makeIconNotebook]
  }, 

  cellLst = accumulateCells[nbX[[1]], iconHeight, lgClHtLim];

  Catch[
    If[cellLst === $Failed, Throw[$Failed]];

    nbXWithPageWidth = setNBPageWidth[nbX, defaultPageWidth];

    Replace[nbXWithPageWidth, Notebook[_, nbopts___] :> Append[DeleteCases[Notebook[cellLst, nbopts], _[WindowSize, _], {1}], WindowSize->{iconWidth, iconHeight}]]
  ]
]

(* ::Package:: *)

Package["Iconize`"]

PackageScope["dynamicObjectQ"]
SetAttributes[dynamicObjectQ, HoldFirst];
dynamicObjectQ[x_] := MatchQ[Head[Unevaluated[x]], Alternatives @@ $dynamicObjects]

(*
	gopals 6/19/20145:
	This function seems unnecessarily complicated, but I seem to remember
	having a thorny evaluation leak that prompted me to do it this way. 
	Please make absolutely certain that any simplifications do not
	result in evaluation leaks.  
*)
PackageScope["dynamicObjectListQ"]
SetAttributes[dynamicObjectListQ, HoldFirst];
dynamicObjectListQ[x__] := AllTrue[Map[MatchQ[#, Alternatives @@ $dynamicObjects]&, List@ReleaseHold[safeHead/@ Hold[x]]], TrueQ]

PackageScope["toicon"]

SetAttributes[toicon, HoldAll];

toicon[i_Image, sz_, bg_] := {"Image", makeThumbnail[i, sz, bg]};

toicon[{x__Image}, sz_, bg_] := {"ImageList", makeThumbnail[ImageCollage@{x}, sz, bg]}

toicon[g_Graphics | g_Graphics3D | g_Legended | g_Graph | g_GeoGraphics, sz_, bg_] := {"Graphics", makeThumbnailRaster[Show[Unevaluated[g], Ticks->None], sz, bg]}

toicon[{x__Graphics | x__Graphics3D | x__Legended | x__Graph | x__GeoGraphics}, sz_, bg_] := {"GraphicsList", makeThumbnail[ImageCollage[Show[#,Ticks->None, Background->White]&/@{x}],sz,bg]}

toicon[expr_NotebookObject | expr_Notebook, sz_, bg_] := Module[{pre, isz, pad, temp, img},
	temp = If[Head[expr] === NotebookObject, NotebookGet[expr], expr];
	simpNotebook = makeIconNotebook[temp, 800, 1000];
 
 	(*
 		makeIconNotebook will return $Failed if it cannot
 		handle the notebook.  See NotebookRasterize
 		for implementation details.    
 	*)
 	
	If[simpNotebook === $Failed, {"Notebook", $Failed}, 
		pre = ImageCrop[Rasterize[simpNotebook, "Graphics", ImageResolution -> $ImageResolution]];
		isz = ImageDimensions[pre];
 
		pad = (sz[[1]] - isz[[1]])/2;
		img = If[isz[[1]] >= sz[[1]], 
 		ImageResize[ImageTake[pre, isz[[1]]], sz[[1]]], 
 		ImagePad[ImageTake[pre, sz[[1]]], {{pad, pad}, {0, 0}}, White]];
 		{"Notebook", makeThumbnail[img, sz, bg]}
 	]
 ]

toicon[type_ExportForm, sz_, bg_]:= Module[{deployPad, documentIcon, format,imageSize, stringLength, text},
	format = Lookup[$formatFormats, type[[2]], "?"];
	stringLength = StringLength[format];
	imageSize = Which[1 <= stringLength <= 4, {Automatic, 94}, 4 <= stringLength <= 8, 250, stringLength > 8, 275];
	text = Rasterize[exportedFileStyle[format], "Graphics", Background -> None, ImageResolution -> 500, ImageSize -> imageSize, ImageResolution -> $ImageResolution];
   	documentIcon = ImageResize[ImageCompose[$notebookTemplate, text, {169.5, 100}], $documentHeight*sz[[1]]];
   	deployPad = ImageCompose[ImageResize[Graphics[{hexToRGB[bg], Rectangle[{0, 0}, sz]}, PlotRangePadding->None], sz], documentIcon];	
   {"ExportedFile", deployPad}
  ]
 
toicon[expr_, sz_, bg_] := {"DynamicObject", makeThumbnailRaster[expr, sz, bg]} /; dynamicObjectQ[expr]

toicon[{x__}, sz_, bg_] := ({"DynamicObjectList", makeThumbnail[ImageCollage[Rasterize[#, "Graphics", ImageResolution -> $ImageResolution]&/@{x}, Background -> hexToRGB[bg]], sz, bg]}) /; dynamicObjectListQ[x]

toicon[{x__}, sz_, bg_] := {"MixedGraphicsList", makeThumbnail[ImageCollage[{x}, Background -> White], sz, bg]} /; SubsetQ[$graphicsHeads, List@ReleaseHold[safeHead/@ Hold[x]]]

toicon[expr_, sz_, bg_] := {"GraphicsGrid", toicon[expr[[1,1,1]], sz, bg][[2]]} /; MatchQ[Unevaluated[expr], Grid[{{_Graphics | _Graphics3D | _Image | _Legended | _GeoGraphics}..}]]
 
toicon[expr_FormFunction, sz_, bg_] := {"FormFunction", With[{expr1=expr[[1]]}, formThumbnail[expr1, sz, bg]]}

toicon[expr_FormObject, sz_, bg_] := {"FormObject", formThumbnail[Unevaluated[expr], sz, bg]}

toicon[expr_, sz_, bg_] := {"Expression", makeThumbnail[expressionThumbnail[Unevaluated[expr], {(1 - $textSpacing)*sz[[1]], (1 - $upperTextSpacing)*sz[[2]]}, bg], sz, bg]}

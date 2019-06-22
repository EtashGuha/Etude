Package["Iconize`"]

(********************* Functions that do not depend on the input expression**************************************)

(*A function that lazily loads the icon and corresponding background color for a given deployment*)
PackageScope["deployInfo"]
deployInfo[x_] := deployInfo[x] = {Import[FileNameJoin[{PacletManager`PacletResource["Iconize","DeploymentIcons"], StringJoin[x]<>".png"}]],$deployColorInfo[x]}

(*TODO: This should probably be done with ImageMultiply*)
(*Function to fade the bottom of the icons when there is too much text*)
PackageScope["fade"]
fade[row_, pixel_, color_, arraySize_, numRow_] := 
 List @@ Blend[{RGBColor @@ pixel, color}, 
   If[row < (arraySize - numRow), 0, 
    N[(row - (arraySize - numRow))/numRow]]]

PackageScope["fadeArray"]
fadeArray[image_, color_, fraction_] := Module[{array, size, numRow},
  array = ImageData[image];
  size = Length@array;
  numRow = Round[fraction*size];
  Image@Partition[Reap[
      For[i = 1, i <= Length[array], i++,
       For[j = 1, j <= Length[array], j++,
        Sow[fade[i, array[[i, j]], color, size, numRow]]
        ]
       ]
      ][[2, 1]],
    Length[array]
    ]
  ]

PackageScope["fontsize"]
(*Grows the font size as a function of the image size*)
fontsize[n_] := Round[N[6/85]*n + 4]

(* Simply doing hexToRGB = Interpreter["StructuredColor"]; was giving an error message*)
PackageScope["hexToRGB"]
hexToRGB[x_] := Interpreter["StructuredColor"][x];

PackageScope["exportedFileStyle"]
(*CloudExport File Style*)
exportedFileStyle[format_]:=Style[format, 
   			FontSize -> 24, 
   			FontColor -> RGBColor[.5, .5, .5],
   			FontFamily -> "Helvetica", 
   			Bold
   			]
   			
PackageScope["makeThumbnail"]   			
(*Make thumbnail simply pads and resizes thumbnails*)
makeThumbnail[img_Image, {tw_,th_}, bg_] := Module[{w,h},
	{w,h} = ImageDimensions[img];
	If[w > tw || h > th,
		ImageResize[img, {{tw},{th}}],
		img] // ImageCrop[#, {tw, th}, Padding->hexToRGB[bg]]&
];

(********************* Functions that involve the input expressiopn**************************************)

(*Format expression*)
PackageScope["styledExpression"]
SetAttributes[styledExpression, HoldFirst];
styledExpression[expr_, fs_] := Style[HoldForm[expr], 
		LineIndentMaxFraction -> 0.05, 
		LineSpacing -> {1, 0}, 
		FontSize -> fs, 
		Italic,
		FontFamily -> "Helvetica", 
		FontColor -> Gray
		]		
		
(*Decrease font size if expression is larger than pane*)
(*TODO: expr is not used here anymore*)
PackageScope["findFontSize"]	
SetAttributes[findFontSize, HoldFirst];	
findFontSize[expr_, sz_] := Module[{paneSize, maxChar, fsTemp},
	paneSize = (1 - $textSpacing)*sz[[1]]*(1 - $textSpacing)*sz[[2]];
	fsTemp = fontsize[sz[[1]]];
	maxChar = Round[paneSize/((fsTemp^2)*$aspectRatio)];
	If[StringLength[ToString[Unevaluated[expr],StandardForm]]>maxChar, .9*fsTemp, fsTemp]
]		

(*Creates thumbnail icons for expressions.  This is the final catch-all category*)
PackageScope["expressionThumbnail"]
SetAttributes[expressionThumbnail, HoldFirst];
expressionThumbnail[expr_, sz_, bg_] := Module[{img,fs}, 
	fs = findFontSize[Unevaluated[expr], sz];
    img = Rasterize[Pane[styledExpression[Unevaluated[expr], fs], sz[[1]], ImageSizeAction -> "Clip"], "Image", Background->hexToRGB[bg], ImageResolution -> $ImageResolution];
    If[Last@ImageDimensions[img] > Last[sz], ImageTake[img, Last[sz]], img]
  ];

(*TODO: is this even used?*) 
(*Create thumbnail for graphics objects*)  
PackageScope["graphicsThumbnail"]
SetAttributes[graphicsThumbnail, HoldFirst];
graphicsThumbnail[expr_, sz_, bg_] := Module[{img},
  img = Rasterize[Pane[Show[Unevaluated[expr], Ticks->None], First@sz, ImageSizeAction -> "ResizeToFit"], "Image", Background -> hexToRGB[bg], ImageResolution -> $ImageResolution];
  If[Last@ImageDimensions[img] > Last[sz], ImageTake[img, Last[sz]], img]
  ]  

(*Commonly used patterns in toicon*)
PackageScope["makeThumbnailRaster"]
SetAttributes[makeThumbnailRaster, HoldFirst];
makeThumbnailRaster[expr_, sz_, bg_] := makeThumbnail[Rasterize[Unevaluated[expr],"Image", Background->hexToRGB[bg], ImageResolution -> $ImageResolution], sz, bg]

SetAttributes[formThumbnail, HoldFirst];
PackageScope["formThumbnail"]
formThumbnail[expr_, sz_, bg_] := makeThumbnail[
	Rasterize[Pane[styledExpression[Unevaluated[expr], fontsize[sz[[1]]]], 2*sz[[1]]], 
		"Image", 
		Background->hexToRGB[bg], 
		ImageSizeAction->"Clip",
		ImageResolution -> $ImageResolution], 
	sz, bg]

(*Used below to extract heads from lists of objects inside function conditions*)	
PackageScope["safeHead"]
SetAttributes[safeHead, HoldAllComplete];
safeHead[expr_] := Part[HoldComplete[expr], 1, 0]
Package["NeuralNetworks`"]


DefineAlias[copyArrayToArray, MXNetLink`PackageScope`mxlCopyArrayToArray]


PackageScope["toImageOutputDims"]

toImageOutputDims[interleaving_, channels_, size_] := 
	If[interleaving, Append, Prepend][Reverse @ size, channels];


PackageScope["validateVarianceImage"]

validateVarianceImage[None, ___] := Null;
validateVarianceImage[spec_, size_, space_, channels_, interleaving_] := Scope[
	data = Normal @ getConformedImageData[spec, size, space, channels, interleaving];
	(* check whether values are positive *)
	If[Min[data] > 0, Return[]];
	(* zeros are easily introduced when resizing or converting color spaces. warn user about this *)
	If[ImageQ[spec] && (ImageColorSpace[spec] =!= space || ImageDimensions[spec] =!= size),
		FailValidation[
			NetEncoder,
			"parameter \"VarianceImage\" contains non-positive values when it is converted to the NetEncoder color space `` using ConvertColor and resized to `` using ImageResize.",
			space, size
		]];
	FailValidation[NetEncoder, "parameter \"VarianceImage\" contains non-positive values."]
];


PackageScope["toImageChannelCount"]

toImageChannelCount["Grayscale"] = 1;
toImageChannelCount["CMYK"] = 4;
toImageChannelCount[Automatic] := SizeT;
toImageChannelCount[_] := 3;


PackageScope["toColorSpace"]

toColorSpace[space_String, _] := space;
toColorSpace[_, 1] = "Grayscale";
toColorSpace[_, 3] = "RGB";
toColorSpace[_, _Integer] = Automatic;
toColorSpace[_, SizeT] = "RGB";



PackageScope["makeWLImageEncoderFunction"]

(* this function builsd a pipeline out of various pieces that are defined below *)

makeWLImageEncoderFunction = Function @ RightComposition[
	loadImage[#ImageSize],
	conformImageColors[#ColorSpace, #ColorChannels],
	conformImageSize[#ImageSize],
	conformImageAlpha[#ColorChannels],
	toReal32Image,
	makeImageFinalizer[#MeanImage, #VarianceImage, #ImageSize, #ColorSpace, #ColorChannels, #Interleaving],
	If[#ColorChannels === 1, padArrayRankTo[1 + Length[#ImageSize], #Interleaving], Identity]
];

(* ------------------------------------------------------------------- *)
(* Step 1: normalize from a File or other graphics object to an Image  *)
loadImage[_][expr_] := EncodeFail["`` was expected to be an Image or File object", MsgForm @ expr];

loadImage[{_, _}][image_Image] := image;
loadImage[{_, _, _}][image_Image3D] := image;

(* load from path *)
loadImage[isize_][File[ipath_String]] := Scope[
	(* infer from filename *)
	path = ExpandFileName[ipath];
	If[!FileExistsQ[path],
		path = FindFile[ipath];
		If[FailureQ[path], EncodeFail["path `` does not exist", ipath]];
	];
	loadImageFile[isize, path]
];

(* 2d file case *)
loadImageFile[isize:{_Integer, _Integer}, path_] := Scope[
	image = Switch[
		ToLowerCase @ FileExtension[path],
		"jpg" | "jpeg",
			First @ Image`ImportExportDump`ImageReadJPEG[path],
		"png",
			tmp = First @ Image`ImportExportDump`ImageReadPNG[path];
			Image`ImportExportDump`DeleteCachePNG[];
			tmp,
		"tiff",
			First @ Image`ImportExportDump`ImageReadTIFF[path],
		_,
			Quiet @ Import[path]
	];
	If[!ImageQ[image], EncodeFail["couldn't load image"]];
	image
]

(* 3d file case *)
loadImageFile[isize:{_Integer, _Integer, _Integer}, path_] := Scope[
	image = Quiet @ Import[path, "Image3D"];
	If[!ImageQ[image], EncodeFail["couldn't load 3D image"]];
	image
];

(* 2d other case *)
loadImageFile[isize:{_Integer, _Integer}][other_] := If[
	TrueQ @ Internal`UnsafeQuietCheck[Image`PossibleImageQ[other], False], 
	Replace[
		Quiet @ Check[Image[other, "Real", ImageSize -> isize], $Failed],
		Except[_Image] :> EncodeFail["failed to rasterize expression with head ``", Head[other]]
	],
	EncodeFail["input is neither a 2D image or a File"]
];

(* 3d other case *)
loadImageFile[isize:{_Integer, _Integer, _Integer}][other_] := If[
	TrueQ @ Internal`UnsafeQuietCheck[Image`PossibleImage3DQ[other], False], 
	Replace[
		Quiet @ Check[Image3D[other, "Real", ImageSize -> isize], $Failed],
		Except[_Image3D] :> EncodeFail["failed to rasterize expression with head ``", Head[other]]
	],
	EncodeFail["input is neither a 3D image or a File"]
];

(* ------------------------------------------------------------------- *)
(* Step 2: ensure colorspace is correct                                *)

conformImageColors[colors_, count_][img_] := Scope[
	space = ImageColorSpace[img];
	Which[
		space === colors, img, 
		space === Automatic && count === ImageChannels[img], img,
		True, ColorConvert[img, colors]
	]
];

(* ------------------------------------------------------------------- *)
(* Step 3: ensure image size is correct                                *)

conformImageSize[size_][img_] := If[ImageDimensions[img] === size, img, ImageResize[img, size]];


(* ------------------------------------------------------------------- *)
(* Step 4: remove the alpha channel, if necessary                      *)

conformImageAlpha[channels_][img_] := Switch[
	ImageChannels[img],
	channels + 1, RemoveAlphaChannel[img, White], 
	channels, img,
	_, EncodeFail["image had wrong number of color channels (`` instead of ``)", ImageChannels[img], channels]
];


(* ------------------------------------------------------------------- *)
(* Step 5: ensure image uses single precision floats                   *)


toReal32Image[img:HoldPattern[Image[ra_NumericArray /; NumericArrayType[ra] =!= "Real32", ___]]] := Image[img, "Real32"];
toReal32Image[img:HoldPattern[Image3D[ra_NumericArray /; NumericArrayType[ra] =!= "Real32", ___]]] := Image3D[img, "Real32"];

toReal32Image[img_Image] := img;
toReal32Image[img_Image3D] := img;
toReal32Image[_] := EncodeFail["couldn't load image"];


(* ------------------------------------------------------------------- *)
(* Step 6: convert image to a NumericArray                             *)

makeImageFinalizer[None, None, _, _, _, interleaving_] := 
	Image`InternalImageData[#, Interleaving -> interleaving]&;

makeImageFinalizer[mean_, var_, size_, space_, channels_, interleaving_] := ModuleScope[
	meanArray = Normal @ normalizeMeanVarSpec[Identity, mean, size, channels, space, interleaving];
	invsdArray = Normal @ normalizeMeanVarSpec[invSqrt, var, size, channels, space, interleaving];
	rank = Length[size]+1;
	Function[input, Block[{tempIm},
		tempIm = padArrayRankTo[rank, interleaving] @ ImageData[input, Interleaving -> interleaving];
		If[meanArray =!= None, tempIm -= meanArray];
		If[invsdArray =!= None, tempIm *= invsdArray];
		toNumericArray[tempIm, "Real32"]
		(* TODO: switch to using NumericArrayUtilities ops when those have been merged *)
	]]
];

(* ------------------------------------------------------------------- *)
(* Step 7: if image is greyscale, we must add the channel dim manually *)

ClearAll[padArrayRankTo];
padArrayRankTo[rank_, interleaving_][array_] /; arrayDepth[array] < rank := 
	ArrayReshape[array, If[interleaving, Append[1], Prepend[1]] @ arrayDimensions[array]]

padArrayRankTo[_, _][array_] := array;


(******
makeOpenCVImageEncoderFunction: a parallel out-of-core image loader relying on OpenCV
	cv::imread (http://docs.opencv.org/3.1.0/d4/da8/group__imgcodecs.html).
	All the limitations of cv::imread apply:
	1) formats like GIF not supported. JPEG, PNG + TIFF supported
	2) Only subset of ColorSpace options supported: (RGB, XYZ, LAB, LUV, Grayscale)
	3) Interleaving is not currently supported
******)

PackageScope["makeOpenCVImageEncoderFunction"]

Clear[makeOpenCVImageEncoderFunction];

makeOpenCVImageEncoderFunction[params_Association, slowEnc_] /; Or[
	!MatchQ[params["ColorSpace"], "RGB" | "XYZ" | "LAB" | "LUV" | "Grayscale"],
	TrueQ[params["Interleaving"]]] := Map[slowEnc];
(* ^ if we can't support the requested encoding, we cannot use the fast-path encoder *)

invSqrt[val_] := 1.0 / Sqrt[val];
invSqrt[val_NumericArray] := invSqrt @ Normal @ val;

normalizeMeanVarSpec[func_, value_, dims_, channels_, space_, interleaving_] := Scope[
	If[MachineQ[value], value = CTable[value, channels]];
	Which[
		value === None,
			None,
		VectorQ[value, MachineQ] && Length[value] === channels,
			If[!interleaving, 
				ones = CTable[1.0, Reverse @ dims]; 
				Map[ones * #&, func @ N @ value]
			,
				CTable[func @ N @ value, Reverse @ dims]
			],
		ImageQ[value], 
			func @ getConformedImageData[value, dims, space, channels, interleaving],
		True, 
			Panic["BadMeanOrVarImage", "`` is not a valid value.", value]
	]
];
(* ^ turn a mean/var spec into a mean/invsd array, which may or may not be a numeric array *)

toOpenCVArray[None] := toNumericArray[{0.}]; (* dummy value *)
toOpenCVArray[a_] := toNumericArray[a];
toOpenCVArray[a_NumericArray] := If[NumericArrayType[a] === "Real32", a, toNumericArray[a, "Real32"]];
(* ^ the opencv librarylink function needs numeric arrays for the mean and stddev image *)

makeOpenCVImageEncoderFunction[params_Association, slowEnc_] := ModuleScope[
	UnpackAssociation[params, meanImage, varianceImage, colorSpace, imageSize, colorChannels, interleaving];
	meanArray = toOpenCVArray @ normalizeMeanVarSpec[Identity, meanImage, imageSize, colorChannels, colorSpace, interleaving];
	invsdArray = toOpenCVArray @ normalizeMeanVarSpec[invSqrt, varianceImage, imageSize, colorChannels, colorSpace, interleaving];
	threads = $ProcessorCount;
	useMean = If[meanImage === None, 0, 1];
	useStdDev = If[varianceImage === None, 0, 1];
	(* careful: dims of ImageData is reverse of ImageSize *)
	arraySize = toImageOutputDims[interleaving, colorChannels, imageSize];
	openCVArgs = {colorSpace, useMean, meanArray, useStdDev, invsdArray, threads, interleaving};
	OpenCVFileLoader[#, arraySize, openCVArgs, slowEnc]&
];

LoadOpenCV := Block[{$ContextPath = $ContextPath}, Needs["OpenCVLink`"]; Clear[LoadOpenCV]];

(* note space in message: prevents eval error during paclet loading!! *)
NetEncoder::imgimprt = "Cannot load ``. Using random image instead.";

OpenCVFileLoader[paths_List, arraySize_, openCVArgs_List, slowEncoder_] := Scope[

	If[!MatchQ[paths, {__File}], Return @ Map[slowEncoder, paths]];

	LoadOpenCV;
	
	outputDims = Prepend[arraySize, Length @ paths];
	output = Developer`AllocateNumericArray["Real32", outputDims];

	paths =  First /* ExpandFileName /@ paths;
	concatPath = StringJoin[paths]; (*  assume paths are list of files *)
	pathLengths = Length[ToCharacterCode[#, "UTF8"]]& /@ paths;
	
	PreemptProtect[
		failureIndices = OpenCVLink`Private`$LoadImagesFromPathInto[
			output, concatPath, pathLengths, 
			Sequence @@ openCVArgs
		];
	];

  	If[!VectorQ[failureIndices, IntegerQ], 
  		Panic["OpenCVImportFailed", "OpenCV function returned incorrect value ``.", failureIndices]];

  	(* if failures occur, replace the failed images with results from slow importor *)
  	If[Length[failureIndices] > 0,
   		indexedFailedPaths = AssociationThread[failureIndices, Part[paths, failureIndices]];
   		slowImports = Map[
   			Function[Quiet @ Catch[slowEncoder[#], EncodeFail]],
   			indexedFailedPaths
   		];
   		(* ^ import those paths that failed using slow, general encoder *)
   		Scan[
   			Message[NetEncoder::imgimprt, #]&, 
   			Pick[Values @ indexedFailedPaths, Values @ slowImports, $Failed]
   		];
   		(* ^ print messages for paths of any images that still failed to import *)
   		KeyValueScan[
   			copyArrayToArray[#2, output, 0, #1, False]&, 
   			Select[slowImports, NumericArrayQ]
   		];
   		(* ^ for successful imports, overwrite previous parts of output array *)
   	];

  	output
];

toReshaper[1, dims_, inter_] := With[
	{dims = If[inter, Prepend, Append][dims, 1], rank = Length[dims]}, 
	If[arrayDepth[#] < rank, ArrayReshape[#, dims], #]&
];

toReshaper[_, _, _] := Identity;

PackageScope["getConformedImageData"]

getConformedImageData[im_Image | im_Image3D, size_, space_, channels_, interleaving_] := Scope[
	conformer = conformImageColors[space, channels] /* conformImageSize[size] /* 
	conformImageAlpha[channels] /* toReal32Image;
	result = Image`InternalImageData[conformer[im], Interleaving -> interleaving];
	padArrayRankTo[1 + Length[size], interleaving] @ result
]

getConformedImageData[other_, _, _, _, _] := other;
(* ::Package:: *)

Begin["System`Convert`CommonGraphicsDump`"]

$GifDocumentedElements = {"Animation", "AnimationRepetitions", "Background", "BitDepth", "Channels", "ColorMap", "ColorSpace", 
	    "Comments", "Data", "DisplayDurations", "DisposalOperation", "GlobalColorMap", "Graphics", "GraphicsList","Image", "ImageCount", 
	    "ImageList", "ImageSize", "RawData", "Summary", "SummarySlideView", "Thumbnail", "ThumbnailList", "TransparentColor"}

$GifHiddenElements = {"DataType", "Frames", "GrayLevels", "Image3D", "ImageOffset", "RGBColorArray", "TransitionEffect", "UserInputFlag"};

$GifAvailableElements = Sort[Join[$GifDocumentedElements, $GifHiddenElements]];

getGifElements[___] := "Elements" -> Sort[Complement["ImportElements" /. System`ConvertersDump`FileFormatDataFull["GIF"], $GifHiddenElements]]


ImportExport`RegisterImport["GIF",
	{
		(*Documented elements*)
			"Animation"													        						:> GetAnimationElement["GIF"],
			"AnimationRepetitions" 																		:> GetImageMetaData["GIF","AnimationRepetitions", All],
			"Background" 																				:> GetImageMetaData["GIF","Background", All],
			"BitDepth" 																					:> GetImageMetaData["GIF","BitDepth", All],
			"Channels"|{"Channels", All|"All"}															:> GetImageMetaData["GIF","Channels", All],
			{"Channels", f:(_Integer|_List)}															:> GetImageMetaData["GIF","Channels", f],
			"ColorMap"|{"ColorMap", All|"All"}			        										:> GetRawDataAndColorMapElements["GIF", All],
			{"ColorMap", f:(_Integer|_List)}							    							:> GetRawDataAndColorMapElements["GIF", f],
			"ColorSpace" 																				:> GetImageMetaData["GIF","ColorSpace", All],
			"Comments"																					:> GetImageMetaData["GIF","Comments", All],
			"Data"|{"Data", All|"All"}								         							:> GetDataElement["GIF", All],
			{"Data", f:(_Integer|_List)}																:> GetDataElement["GIF", f],
			"DisplayDurations"|{"DisplayDurations", All|"All"}											:> GetImageMetaData["GIF","DisplayDurations", All],
			{"DisplayDurations", f:(_Integer|_List)}													:> GetImageMetaData["GIF","DisplayDurations", f],
			"DisposalOperation"|{"DisposalOperation", All|"All"}										:> GetImageMetaData["GIF","DisposalOperation", All],
			{"DisposalOperation", f:(_Integer|_List)}													:> GetImageMetaData["GIF","DisposalOperation", f],
			"GlobalColorMap"							        			    						:> GetGlobalColorMapElement["GIF"],
			"Graphics"													        						:> GetGraphicsElement["GIF"],
			"GraphicsList"|{"GraphicsList", All|"All"}				         							:> GetGraphicsListElement["GIF", All],
			{"GraphicsList", f:(_Integer|_List)}					         							:> GetGraphicsListElement["GIF", f],
			"Image"													            						:> GetImageElement["GIF"],
			"ImageCount"												        						:> GetImageCountElement["GIF"],
			"ImageList"|{"ImageList", All|"All"}				         								:> GetImageListElement["GIF", All],
			{"ImageList", f:(_Integer|_List)}					         								:> GetImageListElement["GIF", f],
			"ImageSize" 																				:> GetImageMetaData["GIF","ImageSize", All],
			"RawData"|{"RawData", All|"All"}						          	 						:> GetRawDataAndColorMapElements["GIF", All],
			{"RawData", f:(_Integer|_List)}																:> GetRawDataAndColorMapElements["GIF", f],
			"Summary"                                                           						:> CreateSummary["GIF", False],
			"SummarySlideView"                                                  						:> CreateSummary["GIF", True],
			"Thumbnail"                                                                                 :> GetThumbnailElement["GIF"],
			{"Thumbnail", s:(_Integer|_Symbol)}                                                         :> GetThumbnailElement["GIF", s],
			"ThumbnailList"|{"ThumbnailList", All|"All"}				         						:> GetThumbnailListElement["GIF", All],
			{"ThumbnailList", f:(_Integer|_List)}					         							:> GetThumbnailListElement["GIF", f],
			{"ThumbnailList", f:(_Integer|_List|All|"All"), s:(_Integer|_Symbol)} 						:> GetThumbnailListElement["GIF", f, s],  
			"TransparentColor"|{"TransparentColor", All|"All"}											:> GetImageMetaData["GIF","TransparentColor", All],
			{"TransparentColor", f:(_Integer|_List)}													:> GetImageMetaData["GIF","TransparentColor", f],
	
		(*Hidden elements*)
			elem_String /; StringMatchQ[elem, "DataType"]												:> GetImageMetaData["GIF","DataType", All],
			elem_String /; StringMatchQ[elem, "Frames"]													:> GetFramesElement["GIF", All],
			{"Frames", All|"All"}																		:> GetFramesElement["GIF", All],
			{"Frames", f:(_Integer|_List)} 																:> GetFramesElement["GIF", f],
			elem_String /; StringMatchQ[elem, "GrayLevels"]												:> GetGrayLevelsElement["GIF", All],
			{"GrayLevels", All|"All"}				         											:> GetGrayLevelsElement["GIF", All],
			{"GrayLevels", f:(_Integer|_List)}							         						:> GetGrayLevelsElement["GIF", f],
			elem_String /; StringMatchQ[elem, "Image3D"]			           							:> GetImage3DElement["GIF"],
			elem_String /; StringMatchQ[elem, "ImageOffset"]											:> GetImageMetaData["GIF","ImageOffset", All],
			{"ImageOffset", All|"All"}																	:> GetImageMetaData["GIF","ImageOffset", All],
			{"ImageOffset", f:(_Integer|_List)}															:> GetImageMetaData["GIF","ImageOffset", f],
			elem_String /; StringMatchQ[elem, "RGBColorArray"]											:> GetRGBColorArrayElement["GIF", All],
			{"RGBColorArray", All|"All"}				         										:> GetRGBColorArrayElement["GIF", All],
			{"RGBColorArray", f:(_Integer|_List)}						          						:> GetRGBColorArrayElement["GIF", f],
			elem_String /; StringMatchQ[elem, "TransitionEffect"]										:> GetImageMetaData["GIF","TransitionEffect", All],
			{"TransitionEffect", All|"All"}																:> GetImageMetaData["GIF","TransitionEffect", All],
			{"TransitionEffect", f:(_Integer|_List)}													:> GetImageMetaData["GIF","TransitionEffect", f],
			elem_String /; StringMatchQ[elem, "UserInputFlag"]											:> GetImageMetaData["GIF","UserInputFlag", All],	
			{"UserInputFlag", All|"All"}																:> GetImageMetaData["GIF","UserInputFlag", All],
			{"UserInputFlag", f:(_Integer|_List)}														:> GetImageMetaData["GIF","UserInputFlag", f],		
			
		(*"DefaultElement" intentionally left out, converter decides whether to return Image or ImageList*)
			Automatic 													          						:> GetDefaultImageElement["GIF"],
		(*All elements*)
			"Elements"					                                           						:> getGifElements,
			getGifElements
	},
	
	"AlphaChannel"      -> True,
    "BinaryFormat"      -> True,
    "Options" 			-> {"AnimationRepetitions", "ColorSpace", "DisplayDurations", "ImageSize", "DisposalOperation", "TransparentColor"},
    "Sources"           -> {"Convert`CommonGraphics`"},
	"AvailableElements" -> $GifAvailableElements
]

End[]

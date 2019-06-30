(* ::Package:: *)

Begin["System`Convert`MovieDump`"]


	(****************** QuickTime for Windows & OSX ******************)
If[ StringMatchQ[$SystemID, "Windows*" | "MacOSX*"],

	ImportExport`RegisterImport["QuickTime",
		{	"ImageList" 								:> (GetImageList["QuickTime"][##]&),
			"GraphicsList" 								:> (GetGraphicsList["QuickTime"][##]&),
			"Animation" 								:> (GetAnimation["QuickTime"][##]&),
			"Data" 										:> (GetData["QuickTime"][##]&),
			{"Frames", f:(_Integer|{__Integer})}		:> (GetFrames["QuickTime", f][##]&),
			{"Data", f:(_Integer|{__Integer})}	 		:> (GetFramesData["QuickTime", f][##]&),
			{"ImageList", f:(_Integer|{__Integer})} 	:> (GetFramesImage["QuickTime", f][##]&),
			{"GraphicsList", f:(_Integer|{__Integer})}	:> (GetFramesGraphics["QuickTime", f][##]&),
			(GetInfo["QuickTime"][##]&)
		},
		{},
		"FunctionChannels" -> {"FileNames"},
		"AvailableElements" -> {"Animation", "BitDepth", "ColorSpace", "Data", "Duration",
			"FrameCount", "FrameRate", "Frames", "GraphicsList", "ImageList", "ImageSize", "VideoEncoding"},
		"DefaultElement" -> "Frames",
		"SystemID" -> ("Windows*" | "Mac*"),
		"BinaryFormat" -> True,
		"Sources" -> {"Convert`CommonGraphics`", "Convert`QuickTime`"}
	]
	,
	(****************** QuickTime for all other platforms ******************)
	ImportExport`RegisterImport["QuickTime",
		{	(*Raw Importers*)
			"Animation" 					:> toAnimation,
		 	"GraphicsList" 					:> toGraphicsAll,
		 	"ImageList" 					:> toImageAll,
		 	"Data" 							:> toDataAll,
		 	{"Frames", a_Integer} 			:> toFrames[a],
		 	{"Frames", "Elements"} 			:> getFrameCountFrames,
		 	{"GraphicsList", "Elements"} 	:> noSubElements["GraphicsList"],
			{"ImageList", "Elements"}		:> noSubElements["ImageList"],
		 	{"Data", "Elements"} 			:> noSubElements["Data"],
		 	getMovieInfo
		},
       {
			(*Postimporters*)
		 	"Frames" 						:> getFrameList,
		 	{"Frames", a_Integer} 			:> StripGraphics[a]
		},
		"Sources" -> {"JLink`", "Convert`JMF`", "Convert`CommonGraphics`"},
		"AvailableElements" -> {"Animation", "BitDepth", "ColorSpace", "Data", "Duration",
			"FrameCount", "FrameRate", "Frames", "GraphicsList", "ImageList", "ImageSize", "VideoEncoding"},
		"DefaultElement" -> "Frames",
		"BinaryFormat" -> True
	]
]

End[]
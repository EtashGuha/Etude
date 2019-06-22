(* ::Package:: *)

Begin["System`Convert`MovieDump`"]


	(****************** AVI for Windows & OSX ******************)	
If[ StringMatchQ[$SystemID, "Windows*" | "MacOSX*"],
	
	ImportExport`RegisterImport["AVI",
		{	"ImageList" 								:> (GetImageList["AVI"][##]&),
			"GraphicsList" 								:> (GetGraphicsList["AVI"][##]&),
			"Animation" 								:> (GetAnimation["AVI"][##]&),
			"Data" 										:> (GetData["AVI"][##]&),
			{"Frames", f:(_Integer|{__Integer})}		:> (GetFrames["AVI", f][##]&),
			{"Data", f:(_Integer|{__Integer})}	 		:> (GetFramesData["AVI", f][##]&),
			{"ImageList", f:(_Integer|{__Integer})} 	:> (GetFramesImage["AVI", f][##]&),
			{"GraphicsList", f:(_Integer|{__Integer})}	:> (GetFramesGraphics["AVI", f][##]&),
			(GetInfo["AVI"][##]&)
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
	(****************** AVI for all other platforms ******************)
	ImportExport`RegisterImport["AVI",
		{	(*Raw Importers*)
			"Animation" 								:> toAnimation,
			"GraphicsList" 								:> toGraphicsAll,
			"ImageList" 								:> toImageAll,
			"Data" 										:> toDataAll,
			{"Frames", a_Integer} 						:> toFrames[a],
			{"Frames", "Elements"}						:> getFrameCountFrames,
			{"GraphicsList", "Elements"} 				:> noSubElements["GraphicsList"],
			{"ImageList", "Elements"} 					:> noSubElements["ImageList"],
			{"Data", "Elements"} 						:> noSubElements["Data"],
			getMovieInfo
		},
		{	(*Postimporters*)
			"Frames" 									:> getFrameList,
			{"Frames", a_Integer} 						:> StripGraphics[a]
		},
		"Sources" -> {"Convert`CommonGraphics`", "Convert`JMF`", "JLink`"},
		"AvailableElements" -> {"Animation", "BitDepth", "ColorSpace", "Data", "Duration",
				"FrameCount", "FrameRate", "Frames", "GraphicsList", "ImageList",
				"ImageSize", "VideoEncoding"},
		"DefaultElement" -> "Frames",
		"BinaryFormat" -> True
	];
]

End[]
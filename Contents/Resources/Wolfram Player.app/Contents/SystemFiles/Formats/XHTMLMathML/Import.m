(* ::Package:: *)

Begin["System`Convert`HTMLImportDump`"]


ImportExport`RegisterImport[
  "XHTMLMathML",
 {
		"Source" 	 :> ImportHTMLSource,
		"Elements"	:> getElements,
	    HTMLToXMLObject
  },
  {
		"Data"			:> HTMLTableImport,
		"FullData"		:> HTMLFullDataImport,
		"Hyperlinks"	:> HTMLHyperlinksImport,
		"ImageLinks"	:> HTMLImageLinksImport,
		"Images"		:> HTMLImagesImport,
		"Plaintext"		:> HTMLTextImport,
		"Title"			:> HTMLTitleImport
  },
  "Sources" -> {"JLink`", "Convert`HTMLImport`", "Convert`ConvertInit`"},
  "OriginalChannel" -> True,
  "AvailableElements" -> {"Data", "FullData", "Hyperlinks", "ImageLinks", "Images", "Plaintext", "Source", "Title", "XMLObject"},
  "DefaultElement" -> "Plaintext"
]


End[]

(* ::Package:: *)

Begin["System`Convert`XMLDump`"]


ImportExport`RegisterImport[
  "XML",
  ImportXML,
  {
	"Tags" :> getSymbolicXMLTags,
	{"Tags", i_String} :> getSymbolicXMLTagsElementS[i],
	{"Tags", i_String, "Plaintext"} :> getSymbolicXMLTagPlaintext[i],
	{"Tags", i_String, "CDATA"} :> getSymbolicXMLTagCDATA[i],
	"CDATA":> getSymbolicXMLCDATA,
	"XMLElement":> getXMLElement,
	"Comments" :> getSymbolicXMLComments,
	"Plaintext" :> getSymbolicXMLPtext,
	"EmbeddedDTD" :> getSymbolicXMLDTD
  },
  "Sources" -> {"XML.exe", "Convert`ConvertInit`"},
  "FunctionChannels" -> {"FileNames", "Streams"},
  "AvailableElements" -> {"CDATA", "Comments", "EmbeddedDTD", "Plaintext", "Tags", "XMLElement", "XMLObject"},
  "DefaultElement" -> "XMLObject"
]


End[]


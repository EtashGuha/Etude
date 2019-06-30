(* ::Package:: *)

Begin["System`Convert`NotebookMLDump`"]


ImportExport`RegisterImport[
  "ExpressionML",
  ExpressionMLToSymbolicXML,
  {
		"Boxes" -> ToBoxesElement,
		"HeldExpression" -> ToHeldExpressionElement,
		"Expression" -> ToExpressionElement,
		"XMLElement"->ToXMLElement
  },
  "Sources" -> {"XML.exe", "Convert`ConvertInit`"},
  "FunctionChannels" -> {"FileNames", "Streams"},
  "AvailableElements" -> {"Boxes", "Expression", "HeldExpression", "XMLElement", "XMLObject"},
  "DefaultElement" -> "Expression"
]


End[]

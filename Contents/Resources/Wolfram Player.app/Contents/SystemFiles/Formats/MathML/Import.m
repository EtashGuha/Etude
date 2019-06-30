(* ::Package:: *)

Begin["System`Convert`MathMLDump`"]


ImportExport`RegisterImport[
  "MathML",
  MathMLToSymbolicXML,
  {
	"Boxes" -> ToBoxesElement,
	"HeldExpression" -> ToHeldExpressionElement,
	"Expression" -> ToExpressionElement,
	"XMLElement"->ToXMLElement
  },
  "Sources" -> {"XML.exe", "Convert`ConvertInit`"},
  "FunctionChannels"-> {"FileNames", "Streams"},
  "AvailableElements" -> {"Boxes", "Expression", "HeldExpression", "XMLElement", "XMLObject"},
  "DefaultElement"->"Boxes"
]


End[]

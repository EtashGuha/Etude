Begin["System`ConvertersDump`WXF`"];


$AvailableElements = {"Expression", "HeldExpression"};

GetElements[___] := "Elements" -> $AvailableElements;


ImportExport`RegisterImport[
	"WXF",
	{
		"Expression" :> ImportWXF,
		"HeldExpression" :> ImportWXFHeld,
		"Elements" :> GetElements
	},
	"FunctionChannels" -> {"FileNames"},
	"BinaryFormat" -> True,
	"DefaultElement" -> "Expression",
	"AvailableElements" -> $AvailableElements
];

End[];
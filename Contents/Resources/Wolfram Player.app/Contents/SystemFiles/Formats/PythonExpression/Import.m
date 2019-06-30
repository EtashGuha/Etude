(* ::Package:: *)

Begin["ExternalEvaluate`Plugins`Python`ImportExport`"];

$PythonExpressionAvailableElements = {"Data"};

GetPythonExpressionElements[___] := "Elements" -> $PythonExpressionAvailableElements


ImportExport`RegisterImport[
	"PythonExpression",
	{
		"Data"->importPythonExpression,
		"Elements" -> GetPythonExpressionElements
	},
	"Sources"->{"ExternalEvaluatePython`"},
	"AvailableElements"->{"Data"},
	"DefaultElement"->"Data",
	"FunctionChannels"->{"Streams"},
	"BinaryFormat"->False
];

End[];
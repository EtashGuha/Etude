(* ::Package:: *)

Begin["ExternalEvaluate`Plugins`Python`ImportExport`"];

ImportExport`RegisterExport[
	"PythonExpression",
	exportPythonExpression,
	"Sources"->{"ExternalEvaluatePython`"},
	"DefaultElement"->"Data",
	"FunctionChannels"->{"Streams"},
	"BinaryFormat"->False
];

End[];

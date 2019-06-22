Begin["System`ConvertersDump`WXF`"];


$ValidExportWXFOptions = Keys[Options[Developer`WriteWXFFile]];

ExportWXF[filename_String, expr_, opts___] := 
	Developer`WriteWXFFile[
		filename, 
		Last[expr],
		Sequence @@ FilterRules[{opts}, $ValidExportWXFOptions],
		"IssueMessagesAs" -> Export
	] /; System`ConvertersDump`Utilities`SingleElementQ[expr, "Expression"];

ExportWXF[filename_String, expr_, opts___] :=
(
	Message[Export::noelem, System`ConvertersDump`Utilities`ElementNames@expr, "WXF"]; 
	$Failed
);

ImportWXF[filename_String, opts___] := 
	"Expression"->Developer`ReadWXFFile[filename, "IssueMessagesAs" -> Import]
ImportWXFHeld[filename_String, opts___] := 
	"HeldExpression"->Developer`ReadWXFFile[filename, HoldComplete, "IssueMessagesAs" -> Import]

End[];
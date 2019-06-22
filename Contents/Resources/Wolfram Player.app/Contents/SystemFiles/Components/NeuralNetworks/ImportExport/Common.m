Package["NeuralNetworks`"]


PackageScope["FailImport"]

General::impfail = "`` import failed for layer ``: ``";

FailImport[format_, layer_, reason_] := ThrowFailure["impfail", format, layer, fromStringForm @ reason];
FailImport[format_, layer_, reason_String, args__] := FailImport[format, layer, StringForm[reason, args]];
_FailImport := $Unreachable;
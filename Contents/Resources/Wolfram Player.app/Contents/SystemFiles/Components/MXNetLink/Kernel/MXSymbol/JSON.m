Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["MXSymbolFromJSON"]

SetUsage @ "
MXSymbolFromJSON[File['path$']] creates a symbol from a JSON file containing a definition.
MXSymbolFromJSON[assoc$] loads a file from an association containing a definition.
MXSymbolFromJSON['str$'] loads a file from a string containing a JSON-encoded definition."

MXSymbolFromJSON[File[path_]] := Scope[
	If[!FileQ[path], ReturnFailed[]];
	MXSymbolFromJSON @ FileString[path]
]

MXSymbolFromJSON[def_Assocation] := Scope[
	Lookup[def, {"nodes", "arg_nodes", "heads"}, ReturnFailed[]];
	json = WriteCompactJSON @ def;
	MXSymbolFromJSON[json]
]

mxlDeclare[mxlMXSymbolCreateFromJSON, {"Integer", "String"}]

MXSymbolFromJSON[def_String] := Scope[
	handle = CreateManagedLibraryExpression["MXSymbol", MXSymbol];
	mxlCall[mxlMXSymbolCreateFromJSON, MLEID @ handle, def];
	System`Private`SetNoEntry @ handle
]

MXSymbolFromJSON[json_Association] := 
	CatchFailure @ MXSymbolFromJSON @ WriteCompactJSON @ json

(******************************************************************************)

PackageExport["MXSymbolToJSON"]

SetUsage @ "
MXSymbolToJSON[MXSymbol[$$]] returns a JSON-like expression that describes the given MXSymbol."

mxlDeclare[mxlMXSymbolSaveToJSON, "Integer", "String"]

MXSymbolToJSON[symbol_MXSymbol] := CatchFailure[
	Developer`ReadRawJSONString @ mxlCall[mxlMXSymbolSaveToJSON, MLEID @ symbol]
]


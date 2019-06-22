Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["MXSymbolOutputs"]

SetUsage @ "
MXSymbolOutputs[MXSymbol[$$]] returns a list of output symbol names of an MXSymbol."

mxlDeclare[mxlMXSymbolStructureInfo, {"Integer", "Integer"}, "String"]

MXSymbolOutputs[symbol_MXSymbol] := CatchFailure[
	mxlUnpackStringVector @ mxlCall[mxlMXSymbolStructureInfo, MLEID @ symbol, 1]
]

(******************************************************************************)

PackageExport["MXSymbolAuxilliaryStates"]

SetUsage @ "
MXSymbolAuxilliaryStates[MXSymbol[$$]] returns the names of the auxilliary state symbols of an MXSymbol."

MXSymbolAuxilliaryStates[symbol_MXSymbol] := CatchFailure[
	mxlUnpackStringVector @ mxlCall[mxlMXSymbolStructureInfo, MLEID @ symbol, 2]
]

(******************************************************************************)

PackageExport["MXSymbolArguments"]

SetUsage @ "
MXSymbolArguments[MXSymbol[$$]] returns the argument names of an MXSymbol."

MXSymbolArguments[symbol_MXSymbol] := CatchFailure[
	mxlUnpackStringVector @ mxlCall[mxlMXSymbolStructureInfo, MLEID @ symbol, 0]
]

(******************************************************************************)

PackageExport["MXSymbolAttributes"]

SetUsage @ "
MXSymbolAttributes[MXSymbol[$$]] returns the attributes of a MXSymbol recursively. 
* Use option \"Recursive\" -> False to return a shallow set of attributes."

Options[MXSymbolAttributes] = {
	"Recursive" -> True
};

mxlDeclare[mxlMXSymbolListAttr, {"Integer", "Boolean"}, "String"]

MXSymbolAttributes[symbol_MXSymbol, opts:OptionsPattern[]] := CatchFailure[
	Developer`ReadRawJSONString @ mxlCall[
		mxlMXSymbolListAttr, MLEID @ symbol, !OptionValue["Recursive"]
	]
]

(******************************************************************************)

PackageExport["MXSymbolSetAttribute"]

SetUsage @ "
MXSymbolSetAttribute[MXSymbol[$$], att$, val$] sets the attribute att$ of symbol MXSymbol[$$]."

mxlDeclare[mxlMXSymbolSetAttr, {"Integer", "String", "String"}];

MXSymbolSetAttribute[symbol_MXSymbol, attr_String, value_] := CatchFailure @ 
	mxlCall[mxlMXSymbolSetAttr, MLEID @ symbol, attr, mxParameterToString @ value];	

(******************************************************************************)

PackageExport["MXSymbolGetAttribute"]

SetUsage @ "
MXSymbolGetAttribute[MXSymbol[$$], att$] gets the attribute with string name att$ 
of symbol MXSymbol[$$]."

mxlDeclare[mxlMXSymbolGetAttr, {"Integer", "String"}, "String"];

MXSymbolGetAttribute[symbol_MXSymbol, attribute_String] := 
	mxlMXSymbolGetAttr[MLEID @ symbol, attribute];

(******************************************************************************)

PackageExport["MXSymbolOperators"]

SetUsage @ "
MXSymbolOperators[MXSymbol[$$]] gives the operator names of the ops present in \
an MXSymbol."

MXSymbolOperators[symbol_MXSymbol] := CatchFailure @ Scope[
	json = MXSymbolToJSON @ symbol;
	If[FailureQ @ json, Return @ json];
	{nodes, argnodes, heads} = Lookup[json, {"nodes", "arg_nodes", "heads"}, Panic[]];
	argnodes += 1; (* change from zero indexed *)
	len = Length @ nodes;
	opsIndices = Complement[Range @ Length @ nodes, argnodes];
	Lookup[nodes[[opsIndices]], "name"]
]
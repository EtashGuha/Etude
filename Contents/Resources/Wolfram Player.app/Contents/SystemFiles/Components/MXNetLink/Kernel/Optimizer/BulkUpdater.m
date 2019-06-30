Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]	


(******************************************************************************)

PackageExport["ArrayUpdater"]

(******************************************************************************)

PackageScope["ArrayUpdaterCreate"]

mxlDeclare[mxlArrayUpdaterNew, {"Integer", "String", "IntegerMatrix", "String", "RealMatrix"}]

ArrayUpdaterCreate[opname_, arrays_, params_] := Scope[
	handle = CreateManagedLibraryExpression["ArrayUpdater", ArrayUpdater];
	mxlCall[mxlArrayUpdaterNew, 
		MLEID @ handle, opname, 
		Transpose @ Map[MLEID, arrays, {2}],
		mxlPackStringVector @ Keys @ params,
		Transpose @ Values @ params
	];
	handle
]

(***************************************************************************)

PackageScope["ArrayUpdaterBindKVStore"]

mxlDeclare[mxlArrayUpdaterBindKVStore, {"Integer", "Integer"}]

ArrayUpdaterBindKVStore[bulk_ArrayUpdater, kv_MXKVStore] :=
	mxlCall[mxlArrayUpdaterBindKVStore, MLEID @ bulk, MLEID @ kv]

(******************************************************************************)

PackageScope["ArrayUpdaterApply"]

mxlDeclare[mxlArrayUpdaterApply, "Integer"]

ArrayUpdaterApply[bulk_ArrayUpdater] := mxlArrayUpdaterApply[MLEID @ bulk]

(******************************************************************************)

PackageScope["ArrayUpdaterSetParamsColumn"]

mxlDeclare[mxlArrayUpdaterSetParamsColumn, {"Integer", "Integer", "RealVector"}]

ArrayUpdaterSetParamsColumn[bulk_ArrayUpdater, col_Integer, tensor_] := 
	mxlArrayUpdaterSetParamsColumn[MLEID @ bulk, col, tensor]


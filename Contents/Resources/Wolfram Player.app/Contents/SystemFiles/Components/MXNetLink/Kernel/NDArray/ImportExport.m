Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["NDArrayImport"]

SetUsage @ "
NDArrayImport[file$] imports an NDArray stored on disk using MXNet's binary format. Returns \
either an association or a list of RawArray objects.
NDArrayImport[file$, {NDArray[$$], $$}] imports the NDArray's in file file$ and copies them into \
existing NDArrays. 
NDArrayImport[file$, <|key1 -> NDArray[$$], $$|>] imports the NDArray's in file file$ and copies \
them into existing NDArrays."

(* %DESIGN Why doesn't this return NDArrays? The name 'NDArrayImport' suggests that it should! *)

NDArrayImport[file_String] := Scope[
	{names, arrays} = getImport @ file;
	rawArrays = mxlReadNDArray$numeric[#, 0, True]& /@ arrays;
	return = If[Length[names] > 0, 
		AssociationThread[names -> rawArrays],
		rawArrays
	];
	Scan[NDArrayFree, arrays];
	return
]

NDArrayImport[file_String, dst_List] := Scope[
	{names, arrays} = getImport @ file;
	If[Length[arrays] =!= Length[dst], Panic["IncompatibleLengths"]];
	ScanThread[NDArrayCopyTo, {dst, arrays}];
	Scan[NDArrayFree, arrays];
]

NDArrayImport[file_String, dst_Association] := Scope[
	{names, arrays} = getImport @ file;
	If[Length[arrays] =!= Length[dst], Panic["IncompatibleLengths"]];
	dstArrays = If[Length[names] === 0, 
		Values @ dst,
		Lookup[dst, names, Panic["MismatchedKeys"]]
	];
	ScanThread[NDArrayCopyTo, {dstArrays, arrays}];
	Scan[NDArrayFree, arrays];
]

_NDArrayImport := $Unreachable;

mxlDeclare[mxlNDArrayLoad, "String", "String"]

(* Utility function for getting arrays + names of ndarrays *)
getImport[file_] := Scope[
	file = ExpandFileName[file];
	If[Not @ FileExistsQ @ file, Panic["FileNotExist"]];
	json = mxlCall[mxlNDArrayLoad, file];
	model = Developer`ReadRawJSONString @ json;
	{model["out_names"], model["out_arr_handles"]}
]

(******************************************************************************)

PackageExport["NDArrayExport"]

SetUsage[NDArrayExport, "
NDArrayExport[file$, <|key1 -> NDArray[$$], $$|>] exports an association of NDArray[$$] objects \
to MXNet's binary export format with filename file$.  
NDArrayExport[file$, {NDArray[$$], $$}] exports a list of NDArray[$$] objects \
to MXNet's binary export format.  
"];

NDArrayExport[file_String, param_Association] := 
	ndExport[file, Keys@param, Values@param]

NDArrayExport[file_String, param_List] := ndExport[file, {}, param]

mxlDeclare[mxlNDArraySave, {"String", "String", "IntegerVector"}]

ndExport[file_, names_, arrays_] := (
	mxlCall[
		mxlNDArraySave, 
		ExpandFileName @ file, 
		mxlPackStringVector @ names, 
		MLEID /@ ReplaceAll[arrays, NDReplicaArray[{f_, ___}] :> f]
	];
	file
)

_NDArrayExport := $Unreachable


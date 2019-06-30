

BeginPackage["LLVMCompileTools`MTensor`"]

AddGetMTensorData::usage = "AddGetMTensorData  "

AddGetMTensorDimensions::usage = "AddGetMTensorData  "

AddSetMTensorPart::usage = "AddSetMTensorPart  "

AddMTensorLength::usage = "AddMTensorLength  "

AddMTensorGetDimension::usage = "AddMTensorGetDimension  "

AddSetArray::usage = "AddSetArray  "

AddGetArray::usage = "AddGetArray  "

AddGetMTensorRank

AddGetMTensorRefCount

AddMTensorRefCountIncrement

AddMTensorRefCountDecrement

AddMTensorRelease

Begin["`Private`"]

Needs["LLVMLink`"]
Needs["LLVMCompileTools`"]
Needs["LLVMCompileTools`Types`"]
Needs["LLVMCompileTools`Basic`"]
Needs["LLVMCompileTools`Comparisons`"]
Needs["CompileUtilities`Error`Exceptions`"]
Needs["LLVMTools`"]


(*
  Should combine these
*)

getBaseValue[base_] :=
	Switch[
		base,
		"Boolean", 1,
		"Integer", 2,
		"Real", 3,
		"Complex", 4,
		_, ThrowException[{"Cannot find MTensor base ", base}]
	]


getBaseId[ data_?AssociationQ, base_] :=
	Module[ {val},
		val = getBaseValue[base];
		AddConstantInteger[data, 32, val]
	]

$MTensorDataFuns :=
	<|
	   "Integer" -> GetMIntPointerType,
       "Integer[64]" -> GetMIntPointerType,
	   "Real" -> GetMRealPointerType
	|>


getTensorField[ data_?AssociationQ, src_, index_] :=
	Module[ {id},
		id = getTensorFieldAddress[data, src, index];
        (* Load this from memory *)
        id = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], id, ""];
        id
	]

getTensorFieldAddress[ data_?AssociationQ, src_, index_] :=
	Module[ {t1, t2, id},
		t1 = AddConstantInteger[data, 32, 0];
		t2 = AddConstantInteger[data, 32, index];
        id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildInBoundsGEP"][data["builderId"], src, #, 2, ""]&, {t1, t2}];
        id
	]


(*
  Core data operations on MTensors
*)

AddCodeFunction["GetMTensorIntegerData", getMTensorIntegerData]
AddCodeFunction["GetMTensorRealData", getMTensorRealData]

getMTensorIntegerData[ data_?AssociationQ, _, {src_}] :=
	getMTensorData[data, src, GetMIntPointerType]

getMTensorRealData[ data_?AssociationQ, _, {src_}] :=
	getMTensorData[data, src, GetMRealPointerType]

getMTensorData[ data_?AssociationQ, src_, baseFun_] :=
	Module[ {id},
		(* Get the MTensor void Data *)
		id = getTensorField[data, src, 7];
        (* cast to the appropriate type *)
        id = LLVMLibraryFunction["LLVMBuildBitCast"][data["builderId"], id, baseFun[data], ""]; 
        id
	]


(*
getMTensorData[ data_?AssociationQ, _, {src_, base_}] :=
	Module[ {id, baseFun},
		(* Get the MTensor void Data *)
		id = getTensorField[data, src, 7];
        baseFun = $MTensorDataFuns[base];
        If[ MissingQ[baseFun], 
        		ThrowException[{"Cannot find base function ", base}]];
        (* cast to the appropriate type *)
        id = LLVMLibraryFunction["LLVMBuildBitCast"][data["builderId"], id, baseFun[data], ""]; 
        id
	]
*)

AddGetMTensorData[ data_?AssociationQ, src_, base_] :=
	Module[ {id, baseFun},
		(* Get the MTensor void Data *)
		id = getTensorField[data, src, 7];
        baseFun = $MTensorDataFuns[base];
        If[ MissingQ[baseFun], 
        		ThrowException[{"Cannot find base function ", base}]];
        (* cast to the appropriate type *)
        id = LLVMLibraryFunction["LLVMBuildBitCast"][data["builderId"], id, baseFun[data], ""]; 
        id
	]
AddGetMTensorData[args___] :=
	ThrowException[{"Invalid arguments to AddGetMTensorData", {args}}]



(*
  Core data operations on MTensors
*)

AddCodeFunction["AddGetMTensorDimensions", AddGetMTensorDimensions]

AddGetMTensorDimensions[ data_?AssociationQ, _, {src_}] :=
	Module[ {id},
		(*  Get Dimensions *)
		id = getTensorField[data, src, 1];
		id
	]

AddCodeFunction["MTensorRank", AddGetMTensorRank]

AddGetMTensorRank[ data_?AssociationQ, _, {src_}] :=
	Module[ {id},
		id = getTensorField[data, src, 2];
		id
	]



AddCodeFunction["MTensorNumberOfElements", AddGetMTensorNumberOfElements]

AddGetMTensorNumberOfElements[ data_?AssociationQ, _, {src_}] :=
	Module[ {id},
		id = getTensorField[data, src, 6];
		id
	]


AddCodeFunction["GetMTensorData", AddGetMTensorRawData]

AddGetMTensorRawData[ data_?AssociationQ, _, {src_}] :=
	Module[ {id},
		id = getTensorField[data, src, 7];
        id
	]

AddCodeFunction["GetMTensorRefCount", AddGetMTensorRefCount]

AddGetMTensorRefCount[ data_?AssociationQ, _, {src_}] :=
	Module[ {id},
		id = getTensorField[data, src, 8];
        id
	]


AddCodeFunction["GetMTensorRefCountAddress", AddGetMTensorRefCountAddress]

AddGetMTensorRefCountAddress[ data_?AssociationQ, _, {src_}] :=
	Module[ {id},
		id = getTensorFieldAddress[data, src, 8];
        id
	]


AddCodeFunction["MTensorRefCountIncrement", AddMTensorRefCountIncrement]

AddMTensorRefCountIncrement[ data_?AssociationQ, _, {src_}] :=
    Module[ {refId, ref, id},
        refId = AddGetMTensorRefCountAddress[data, Null, {src}];
        ref = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], refId, "pre_refcnt"];
        id = LLVMLibraryFunction["LLVMBuildAdd"][ data["builderId"], ref, AddConstantInteger[data, 64, 1], "increment_refcnt"]; 
        LLVMLibraryFunction["LLVMBuildStore"][data["builderId"], id, refId];
        ref
    ]
    
AddCodeFunction["MTensorRefCountAtomicIncrement", AddMTensorRefCountAtomicIncrement]

AddMTensorRefCountAtomicIncrement[ data_?AssociationQ, _, {src_}] :=
	Module[ {id},
		id = AddGetMTensorRefCountAddress[data, Null, {src}];
		id = LLVMLibraryFunction["LLVMBuildAtomicRMW"][data["builderId"], LLVMEnumeration["LLVMAtomicRMWBinOp", "LLVMAtomicRMWBinOpAdd"], id, AddConstantInteger[data, 64, 1], LLVMEnumeration["LLVMAtomicOrdering", "LLVMAtomicOrderingMonotonic"], 1]; 
        id
	]

AddCodeFunction["MTensorRefCountDecrement", AddMTensorRefCountDecrement]

AddMTensorRefCountDecrement[ data_?AssociationQ, _, {src_}] :=
	Module[ {refId, ref, id},
        refId = AddGetMTensorRefCountAddress[data, Null, {src}];
        ref = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], refId, "pre_refcnt"];
        id = LLVMLibraryFunction["LLVMBuildSub"][ data["builderId"], ref, AddConstantInteger[data, 64, 1], "decrement_refcnt"]; 
        LLVMLibraryFunction["LLVMBuildStore"][data["builderId"], id, refId];
        ref
	]

AddCodeFunction["MTensorRefCountAtomicDecrement", AddMTensorRefCountAtomicDecrement]

AddMTensorRefCountAtomicDecrement[ data_?AssociationQ, _, {src_}] :=
    Module[ {id},
        id = AddGetMTensorRefCountAddress[data, Null, {src}];
        id = LLVMLibraryFunction["LLVMBuildAtomicRMW"][data["builderId"], LLVMEnumeration["LLVMAtomicRMWBinOp", "LLVMAtomicRMWBinOpSub"], id, AddConstantInteger[data, 64, 1], LLVMEnumeration["LLVMAtomicOrdering", "LLVMAtomicOrderingMonotonic"], 1];
        id
    ]



(*
 Structural operations on MTensors
*)


AddSetMTensorPart[ data_?AssociationQ, trgt_, base_, off_, val_] :=
	Module[ {id},
		(* Get the MTensor Data *)
		id = AddGetMTensorData[data, trgt, base];
		id = AddSetArray[data, id, off, val];
		id		
	]


AddMTensorGetDimension[ data_?AssociationQ, src_, ind_] :=
	Module[ {dimsId, id},
		dimsId = AddGetMTensorDimensions[data, Null, {src}];
		(* Get the desired dimension *)
        id = WrapIntegerArray[LLVMLibraryFunction["LLVMBuildInBoundsGEP"][data["builderId"], dimsId, #, 1, ""]&, {ind}];
        (*  Load from memory *)
        id = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], id, ""];
		id
	]

AddMTensorLength[ data_?AssociationQ, src_] :=
	Module[ {lenId, id},
		lenId = AddConstantInteger[data, 32, 0];
		id = AddMTensorGetDimension[ data, src, lenId];
		id
	]




AddSetArray[ data_?AssociationQ, trgt_, off_, val_] :=
	Module[ {id},
        (* Get the off-th element *)
        id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildInBoundsGEP"][data["builderId"], trgt, #, 1, ""]&, {off}];
        (* Store value into this memory *)
        LLVMLibraryFunction["LLVMBuildStore"][data["builderId"], val, id];
		val		
	]

AddGetArray[ data_?AssociationQ, trgt_, off_] :=
	Module[ {id},
        (* Get the off-th element *)
        id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildInBoundsGEP"][data["builderId"], trgt, #, 1, ""]&, {off}];
        (* Load value from memory *)
       	id = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], id, ""];
		id
	]

AddCodeFunction["GetArray_I", AddGetArrayI]

AddGetArrayI[ data_?AssociationQ, _, {trgt_, off_}] :=
	AddGetArray[ data, trgt, off]





(*
   Release an MTensor
   Decrement the refcount and if this is going to 0 call Free
*)
AddMTensorRelease[data_?AssociationQ, tyId_, srcId_] := 
	Module[{refId, compId, eTrue, bbTrue},
		refId = AddMTensorRefCountDecrement[ data, Null, {srcId}];
		compId = AddLLVMCompareCall[ data, "binary_equal_UnsignedInteger",{refId, AddConstantInteger[data, 64, 1]}];
		{eTrue, bbTrue} = 
			AddIfTrueBasicBlock[ data, compId,
				Function[{data1, bbFalse},
					AddRuntimeFunctionCall[ data1, "FreeMTensor", {srcId}];
					LLVMLibraryFunction["LLVMBuildBr"][data1["builderId"], bbFalse];
					]];
		compId
	]



End[]


EndPackage[]
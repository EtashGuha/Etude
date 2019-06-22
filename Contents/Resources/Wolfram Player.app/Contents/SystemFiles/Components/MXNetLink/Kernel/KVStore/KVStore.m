Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["MXKVStore"]

SetUsage @ "
MXKVStore[id$] represents a key-value store managed by MXNet."

(******************************************************************************)

PackageExport["MXKVStoreCreate"]

SetUsage @ "
MXKVStoreCreate[] creates a 'local' KVStore.
MXKVStoreCreate[type$] creates a KVStore of type type$."

mxlDeclare[mxlMXKVStoreCreate, {"Integer", "String"}]

MXKVStoreCreate[type_:"local"] := Scope[
	handle = CreateManagedLibraryExpression["MXKVStore", MXKVStore];
	mxlCall[mxlMXKVStoreCreate, MLEID[handle], type];
	System`Private`SetNoEntry[handle]
]

(******************************************************************************)

PackageExport["MXKVStoreGetType"]

SetUsage @ "
MXKVStoreGetType[KVStore[$$]] returns the string type of the KVStore."

mxlDeclare[mxlMXKVStoreGetType, "Integer", "String"]

MXKVStoreGetType[kv_MXKVStore] := 
	mxlCall[mxlMXKVStoreGetType, MLEID[kv]]

(******************************************************************************)

PackageExport["MXKVStoreInit"]

SetUsage @ "
MXKVStoreInit[MXKVStore[$$], key$ -> NDArray[$$]] initializes the KVStore \
with a integer or string key$ and NDArray[$$] value.
MXKVStoreInit[MXKVStore[$$], <|key$1 -> NDArray[$$], $$|>] uses an Association \
of key-values."

MXKVStoreInit[store_MXKVStore, Rule[k_, v_NDArray]] := 
	iMXKVStoreInit[store, {k}, {v}]

MXKVStoreInit[store_MXKVStore, Rule[k_List, v_List]] := 
	iMXKVStoreInit[store, k, v]

MXKVStoreInit[store_MXKVStore, a_Association] := 
	iMXKVStoreInit[store, Keys[a], Values[a]]

mxlDeclare[mxlMXKVStoreInit, {"Integer", "IntegerVector", "IntegerVector"}]

iMXKVStoreInit[store_, keys_, values_] := 
	mxlCall[mxlMXKVStoreInit, MLEID @ store, keys, MLEID /@ values]

mxlDeclare[mxlMXKVStoreInitEx, {"Integer", "String", "IntegerVector"}]

iMXKVStoreInit[store_, keys_ ? Developer`StringVectorQ, values_] := 
	mxlCall[mxlMXKVStoreInitEx, MLEID @ store, mxlPackStringVector @ keys, MLEID /@ values]

_MXKVStoreInit := $Unreachable

(******************************************************************************)

PackageExport["MXKVStorePush"]
PackageExport["MXKVStorePull"]

SetUsage @ "
MXKVStorePush[MXKVStore[$$], key$ -> NDArray[$$]] pushes the given array into \
the KV store, updating the existing value corresponding to the key.
MXKVStorePush[store$, key$ -> {array$1, $$}] pushes several arrays \
to a single key at once. 
MXKVStorePush[store$, {keys$1, $$} -> {array$1, $$}] pushes multiple arrays$ onto \
the corresponding keys.
MXKVStorePush[store$, <|key$1 -> array$1, $$|>] does the same.
MXKVStorePush[store$, spec$, priority$] uses the given integer priority.
* One more than one array is pushed to a key at once, they will be totalled \
before the update is performed.
* The default priority is 0."

SetUsage @ "
MXKVStorePull[MXKVStore[$$], key$ -> NDArray[$$]] pulls the value corresponding \
to the key from the KV store into the given array.
MXKVStorePull[store$, key$ -> {array$1, $$}] pulls the same value to \
multiple arrays at once.
MXKVStorePull[store$, {key$1, $$} -> {array$1, $$}|>] pulls the values of \
multiple keys into the corresponding arrays.
MXKVStorePull[store$, <|key$1 -> array$1, $$|>] does the same.
MXKVStorePull[store$, spec$, priority$] uses the given integer priority.
* Keys must be either all strings or all integers.
* The default priority is 0."

(* Push and pull share exactly the same argument processing and dispatch, they just
flip a flag to the underlying C function *)

MXKVStorePush[store_, spec_, priority_:0] := KVStorePushPull[store, spec, priority, True]
MXKVStorePull[store_, spec_, priority_:0] := KVStorePushPull[store, spec, priority, False]

_MXKVStorePush := $Unreachable
_MXKVStorePull := $Unreachable

(* KVStorePushPull normalizes argumnets to lists *)

KVStorePushPull[store_MXKVStore, Rule[k_, v_NDArray], prio_, dir_] :=
	iKVStorePushPull[store, {k}, {v}, prio, dir]

KVStorePushPull[store_MXKVStore, Rule[k_, v_List], prio_, dir_] := 
	iKVStorePushPull[store, If[ListQ[k], k, ConstantArray[k, Length[v]]], v, prio, dir]

(* fast path for single string key: avoids packing *)
KVStorePushPull[store_MXKVStore, Rule[k_String, v_NDArray], prio_, dir_] := 
	mxlCall[mxlMXKVStorePushPullEx, MLEID @ store, prio, k, List @ MLEID @ v, dir]

KVStorePushPull[store_MXKVStore, a_Association, prio_, dir_] :=
	iKVStorePushPull[store, Keys[a], Values[a], prio, dir]

_KVStorePushPull := $Unreachable

(* iKVStorePushPull chooses between string and int API calls *)

mxlDeclare[mxlMXKVStorePushPull, {"Integer", "Integer", "IntegerVector", "IntegerVector", "Boolean"}]

iKVStorePushPull[store_, keys_ /; VectorQ[keys, IntegerQ], values_, prio_, dir_] := 
	mxlCall[
		mxlMXKVStorePushPull, MLEID @ store, prio, 
		keys, MLEID /@ values,
		dir
	]

mxlDeclare[mxlMXKVStorePushPullEx, {"Integer", "Integer", "String", "IntegerVector", "Boolean"}]

iKVStorePushPull[store_, keys_ ? Developer`StringVectorQ, values_, prio_, dir_] := 
	mxlCall[
		mxlMXKVStorePushPullEx, MLEID @ store, prio, 
		mxlPackStringVector @ keys, MLEID /@ values, 
		dir
	]

_iKVStorePushPull := $Unreachable


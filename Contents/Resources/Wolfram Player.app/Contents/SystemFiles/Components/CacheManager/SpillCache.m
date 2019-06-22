Package["CacheManager`"]

PackageExport["SpillCache"]

(* SpillCache collects accesses and writes into an L1 cache. If enough writes accumulate,
it schedules a shared async task to flush the L1 cache into an L2 cache, which is limited 
to MaxSize elements.

This L2 cache is also written to disk.
*)

SetAttributes[SpillCache, HoldAllComplete];

Options[SpillCache] = {
};

iCreate["SpillCache", size_, OptionsPattern[SpillCache]] := With[
	{l1 = makeCacheSymbol["L1"], 
	 l2 = makeCacheSymbol["L2"], 
	 file = makeCacheFile["Spill.mx"]},
	l1 = Association[];
	l2 = Association[];
	If[FileExistsQ[file], Get[file]];
	SpillCache[l1, l2, file, size]
];

iFlush[self_, SpillCache[l1_, l2_, file_, len_]] := (
	l2 = LimitTo[Join[l2, l1], len];
	l1 = Association[];
	DumpSave[file, l2];
);

iClear[self_, SpillCache[l1_, l2_, _, _]] :=
	(
		l1 = Association[]; 
		l2 = Association[];
	)

iSet[self_, SpillCache[l1_, l2_, _, _], key_, value_] :=
	(
		l1[key] = value;
		If[CacheDirtiness[self]++ > 64, FlushCacheDeferred[self]];
	);

iUnset[self_, SpillCache[l1_, l2_, file_, _], key_] := 
	(
		l1[key] =.;
		l2[key] =.;
	);
	
iGet[self_, SpillCache[l1_, l2_, file_, _], key_] :=
	Lookup[l1, key, 
		With[{l = Lookup[l2, key]}, 
			If[Head[l] =!= Missing, l1[key] = l, l]
		]
	];
	
iKeys[self_, SpillCache[l1_, l2_, file_, _]] :=
	DeleteDuplicates @ Join[Keys[l1], Keys[l2]];
	
iInfo[SpillCache[l1_, l2_, file_, len_]] := {
	{
		"L1Count" -> Length[l1],
		"L2Count" -> Length[l2]
	}, 
	{
		"MaxSize" -> len,
		"File" -> file, 
		"FileByteCount" -> Dynamic[Quiet @ Check[FileByteCount[file], 0]]
	}
};

LimitTo[x_, n_] := If[Length[x] > n, Drop[x, Length[x] - n], x];


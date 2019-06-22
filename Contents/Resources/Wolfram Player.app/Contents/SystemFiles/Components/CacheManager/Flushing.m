Package["CacheManager`"]

PackageImport["GeneralUtilities`"]

PackageExport["$FlushInterval"]
PackageExport["$AllowAutoFlushing"]
PackageExport["CacheDirtyQ"]
PackageExport["CacheDirtiness"]

$AllowAutoFlushing = True;
$FlushInterval = 4.0;

$FlushTask := $FlushTask = CreateScheduledTask[FlushDirtyCaches[], {$FlushInterval}, "Context" -> "Internal`"];
$FlushPending = False;
CacheDirtiness[_] := 0;
CacheDirtyQ[sym_] := CacheDirtiness[sym] > 0;
FlushNotRequestedQ[_] := True;

PackageExport["FlushCacheDeferred"]

FlushCacheDeferred[sym_Symbol ? FlushNotRequestedQ] := (
	FlushNotRequestedQ[sym] = False;
	If[$FlushPending === False && $AllowAutoFlushing, 
		$FlushPending = True;
		StartScheduledTask[$FlushTask];
	]
);



PackageExport["FlushDirtyCaches"]

FlushDirtyCaches[] := With[
	{toflush = Select[$Caches, CacheDirtyQ]},
	$FlushPending = False;
	Scan[FlushCache, toflush];
];


PackageExport["FlushCache"]
PackageScope["iFlush"]

FlushCache::nexists = "No cache named \"``\"";
FlushCache[sym_String] := FlushCache[Lookup[$Caches, sym, Message[FlushCache::nexists, sym]; $Failed]];

FlushCache[$Failed] := $Failed;
FlushCache[sym_] := (
	CacheDirtiness[sym] = 0; 
	FlushNotRequestedQ[sym] = True; 
	With[{res = iFlush[sym, CacheData[sym]]}, res /; res =!= $Failed]
);

FlushCache::notcachesym = "`` is not a cache symbol"

iFlush[sym_, ___] := (Message[FlushCache::notcachesym, sym]; $Failed);


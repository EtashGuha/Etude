Package["CacheManager`"]



PackageExport["CachedExpression"]

CachedExpression::usage = "CachedExpression[cache, value] evaluates value unless it has prevously been evaluated.
CachedExpression[{cache, key}, value] caches the value under the key 'key'"

CachedExpression[___] := Missing["NotImplemented"];


PackageExport["CachedFunction"]

CachedFunction::usage = "CachedFunction[cache, function] represents a function whose outputs are cached in cache." 

CachedFunction[___] := Missing["NotImplemented"];


PackageExport["CachedMap"]

CachedMap::usage = "CachedMap[cache, func, keys] computes func /@ keys, relying on cached values where possible.
If the number of keys exceeds the size of the cache, no new cache entries will be created.";

CachedMap[___] := Missing["NotImplemented"];

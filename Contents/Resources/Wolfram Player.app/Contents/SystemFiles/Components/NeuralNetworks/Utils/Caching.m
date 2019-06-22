Package["NeuralNetworks`"]

clearExpressionStore[store_] := Scan[store["remove"[First[#]]]&, store["listTable"[]]];
listExpressionStore[store_] := pairsToAssoc /@ pairsToAssoc @ store["listTable"[]];
listExpressionStore[store_, obj_] := FirstCase[store["listTable"[]], {Verbatim[obj], vals_} :> pairsToAssoc[vals]];

pairsToAssoc[pairs_List] := Association[Rule @@@ pairs];


PackageScope["RecentNetCached"]
PackageExport["$MaxCachedNets"]

ClearAll[RecentNetCached];

SetUsage @ "
RecentNetCached[fund$, net$, args$] will clear
cached stuff for nets that were not used recently
($MaxCachedNets= 3 by default).

RecentNetCached is used to cache executors. A similar mechanism is used for the 
fast path cache (see NetApply), which is basically a special purpose cache that
bypasses all ordinary logic for simple nets to make them as low-overhead as possible
when doing plan evaluation (e.g. inside a Plot or Table). 

The point of using RecentNetCached is to reduce memory bloat from lots of different
nets. RecentNetCached will still result in an executor being reused in e.g. a Plot or
Table, but as soon as another net is used the first executor is thrown away. So there's
only ever one executor in the cache at any given time.

NetPlans are still cached permanently, via the old-fashioned Cached, whic has no 
occupancy limit. However, this doesnt result in much resource usage AFAIK -- MXSymbols
dont have much overhead until they are MXSymbolBind'd.
"

RunInitializationCode[$SingletonCache = Language`NewExpressionStore["SingletonNNCache"]]

$RecentNetUUIDs = {};
$MaxCachedNets = 3;
RecentNetCached[func_, net_, args__] := Scope[
	id = NetUUID[net];
	If[id =!= First[$RecentNetUUIDs, Null],
		If[!MemberQ[$RecentNetUUIDs, id] && Length[$RecentNetUUIDs] >= $MaxCachedNets,
			(* remove the oldest net of the cache *)
			idtoremove = Last[$RecentNetUUIDs];
			Scan[Function[If[NetUUID @ First[#] === idtoremove,
				$SingletonCache["remove"[First[#]]]]
				],
				$SingletonCache["listTable"[]]
			];
		];
		ClearCache["NDArrayCache"]; (* a bit unclean to couple this general-purpose caching
		mechanism to clearing ndarrays, but does the job, which is to fully wipe out
		any residual impact of the previous cached net *)

		(* Latest is comes first in the list *)
		$RecentNetUUIDs ^= Take[DeleteDuplicates[Prepend[$RecentNetUUIDs, id]], UpTo[$MaxCachedNets]];
	];
	Replace[$SingletonCache["get"[net, {func, args}]], Null :>
		Replace[func[net, args],
			res:Except[_ ? FailureQ] :> 
			($SingletonCache["put"[net, {func, args}, res]]; res)
		]
	]
]

PackageScope["Cached"]
PackageScope["NetUUID"]

RunInitializationCode[$GeneralCache = Language`NewExpressionStore["GeneralNNCache"]]

(* TODO: Replace this with a System`Private` function that just returns the raw pointer of an expression *)
NetUUID[net_] := Cached[getUUID, net];
$UUIDCount = 0;
getUUID[net_] := ++$UUIDCount;

ClearAll[Cached];

Cached[func_, net_, args__] := 
	Replace[$GeneralCache["get"[net, {func, args}]], Null :>
		Replace[func[net, args],
			res:Except[_ ? FailureQ] :> 
			($GeneralCache["put"[net, {func, args}, res]]; res)
		]
	];

Cached[func_, net_] := 
	Replace[$GeneralCache["get"[net, func]], Null :>
		Replace[func[net],
			res:Except[_ ? FailureQ] :> 
			($GeneralCache["put"[net, func, res]]; res)
		]
	];


PackageScope["CachedIf"]
	
CachedIf[True, args___] := Cached[args];
CachedIf[_, args___] := Construct[args];


PackageScope["ToCachedNDArray"]

RunInitializationCode[$NDArrayCache = Language`NewExpressionStore["NumericArrayNDArrayCache"]]

ToCachedNDArray[ra_, cont_, type_] := 
	Replace[$NDArrayCache["get"[ra, {cont, type}]], Null :> Block[
		{res = MXNetLink`NDArrayCreate[ra, cont, type]},
		$NDArrayCache["put"[ra, {cont, type}, res]]; res
	]];


PackageScope["$FastPathCache"]

RunInitializationCode[$FastPathCache = Language`NewExpressionStore["FastPathNNCache"]]



PackageScope["$AllCaches"]

$AllCaches = <|
	"NDArrayCache" :> $NDArrayCache,
	"GeneralCache" :> $GeneralCache,
	"SingletonCache" :> $SingletonCache,
	"FastPathCache" :> $FastPathCache
|>;


PackageScope["CacheContents"]

CacheContents[] := AssociationMap[CacheContents, Keys[$AllCaches]];
CacheContents["NetModel"] := $LocalNetModelCache;
CacheContents["NetMeasurements"] := $NetMeasurementsCache;
CacheContents[type_String] := listExpressionStore[$AllCaches[type]];
CacheContents[type_String, obj_] := listExpressionStore[$AllCaches[type], obj];


PackageExport["ClearCache"]

ClearCache[] := (Scan[ClearCache, Keys[$AllCaches]]; ClearNetModelCache[]; ClearPrecompCache[]; ClearNetMeasurementsCache[];);
ClearCache["Internal"] := Scan[ClearCache, Keys[$AllCaches]]; 
ClearCache[type_] := clearExpressionStore[$AllCaches[type]];
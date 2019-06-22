Package["CacheManager`"]

PackageImport["GeneralUtilities`"]
PackageImport["Macros`"]


PackageExport["CacheData"]
CacheData[_] := $Failed;

PackageExport["$DefaultCacheMethod"]
PackageExport["$DefaultCacheSize"]
PackageExport["CacheSize"]

$DefaultCacheSize = 4096;
$DefaultCacheMethod = "SpillCache";

PackageExport["CacheSymbolQ"]

CacheSymbolQ[sym_Symbol] := CacheData[sym] =!= $Failed;
CacheSymbolQ[_] := False;

PackageExport["$Caches"]

$Caches::usage = "$Caches maps a cache's name to the underlying CacheData";	
$Caches = Association[];



PackageExport["GetCache"]
PackageExport["CreateCache"]
PackageExport["RemoveCache"]

PackageScope["iCreate"]
iCreate[___] := $Failed;

Options[GetCache] = Options[CreateCache] = {
	Method :> $DefaultCacheMethod,
	CacheSize :> $DefaultCacheSize
};

GetCache::invld = "Invalid arguments `` to GetCache. Returning dummy symbol `` instead.";

GetCache[name_, opts:OptionsPattern[]] := Lookup[$Caches, name, CreateCache[name, opts]];
GetCache[args___] := With[{u = Block[{$Context = "Caches`Dummy`"}, Unique[]]}, Message[GetCache::invld, HoldForm[{args}], u]; u];

RemoveCache::nexists = "No cache named \"``\"";
RemoveCache::arg1 = "First argument `` must be the string name of an existing cache or a valid cache symbol.";


RemoveCache[name_Symbol] := ConditionalRHS[
	SymbolQ[name] && Context[name] === "Caches`", {"arg1", name}, 
	iRemoveCache[SymbolName[name]]
];

RemoveCache[name_String] := iRemoveCache[name];
	
iRemoveCache[name_] := 
Module[{dir, found = True},
	dir = FileNameJoin[{$CacheBaseDirectory, name}];
	Quiet[ CacheData[Symbol["Caches`" <> name]] =. ];
	Quiet[
		ClearAll[Evaluate["Caches`" <> name]];
		ClearAll[Evaluate["Caches`" <> name <> "`*"]];
		Check[
			Remove[Evaluate["Caches`" <> name]];
			Remove[Evaluate["Caches`" <> name <> "`*"]];
		,
			found = False;
		];
	];
	If[FileExistsQ[dir], 
		found = True;
		If[DeleteDirectory[dir, DeleteContents -> True] === $Failed,
			found = False
		];
	];
	If[KeyExistsQ[$Caches, name], 
		found = True;
		$Caches[name] =.;
	];
	If[!found, Message[RemoveCache::nexists, name]; $Failed]
];


CreateCache::arg1 = "Cache name `` should be a string.";
CreateCache::badsize = "Invalid value `` of CacheSize.";
CreateCache::badname = "Cache name \"``\" must be a valid symbol name.";
CreateCache::badmethod = "Invalid value `` for Method.";
CreateCache::failed = "Failed to create cache.";
CreateCache::nexists = "Cache named `` already exists";

(* cache creation is relatively expensive, anticipating there will be few named caches *)
CreateCache[name_, OptionsPattern[]] := ConditionalRHS[
	
	StringQ[name], {"arg1", name},
	
	StringMatchQ[name, LetterCharacter ~~ WordCharacter..], {"badname", name},
	
	Block[
	{$CurrentCacheName = name, type, opts, size, method},
	
	method = OptionValue[Method];
	{type, opts} = Match[
		method, 
			s_String :> {s, {}},
			{s_String, o___Rule} :> {s, {o}},
			Automatic -> {$DefaultCacheType, {}}
		,
			Message[CreateCache::badmethod, method]; 
			{$Failed, $Failed}
	];
	If[type === $Failed, Return[$Failed]];
	
	size = OptionValue[CacheSize];
	If[!Integer[size], 
		Message[CreateCache::badsize, name]; 
		Return[$Failed]
	];
	
	If[KeyExistsQ[$Caches, name], 
		Message[CreateCache::nexists, name];
		Return[$Failed]
	];
		
	ClearAll[Evaluate["Caches`" <> name]];
	ClearAll[Evaluate["Caches`" <> name <> "`*"]];

	With[{
		cacheSymbol = Symbol["Caches`" <> name],
		set = getMethod[type, "iSet"],
		get = getMethod[type, "iGet"],
		unset = getMethod[type, "iUnset"],
		
		newcache = iCreate[type, size, Sequence @@ opts]
	},
		
		If[newcache === $Failed, 
			Message[CreateCache::badmethod, method];
			Return[$Failed, Block]
		];
			
		(* patch the symbol so that it intercepts subvalue getting and setting *)
		PatchLValueOperations[cacheSymbol, 
			"SetSubValue" -> set,
			"GetSubValue" -> get,
			"UnsetSubValue" -> unset,
			"SelfFunction" -> CacheData
		];
		
		(* formats as its "pseudo own-value" *)
		cacheSymbol /: Format[cacheSymbol, StandardForm] := FormatCacheSymbol[cacheSymbol];
		cacheSymbol /: Normal[cacheSymbol] := EvalForm @ CacheData[cacheSymbol];
		
		cacheSymbol /: CacheSet[cacheSymbol, key_, value_] := set[cacheSymbol, CacheData[cacheSymbol], key, value];
		cacheSymbol /: CacheGet[cacheSymbol, key_] := get[cacheSymbol, CacheData[cacheSymbol], key];
		cacheSymbol /: CacheUnset[cacheSymbol, key_] := unset[cacheSymbol, CacheData[cacheSymbol], key];
		
		(* set the "pseudo own-value" *)
		CacheData[cacheSymbol] = newcache;
		
		(* store the symbol under its name *)
		$Caches[name] = cacheSymbol
	]
]];

EvalForm[head_[args___]] := Apply[HoldForm[head], {args}];

CreateCache[___] := $Failed;

(* each package fragment should implement package private versions of iCacheSet, etc. This saves a level of indirection. *)
getMethod[type_, name_] := Symbol["CacheManager`" <> type <> "`PackagePrivate`" <> name];


PackageExport["CacheGet"]
PackageExport["CacheSet"]
PackageExport["CacheUnset"]


CacheGet[name_String, key_] := CacheGet[Cache[name], key];
CacheSet[name_String, key_, value_] := CacheSet[Cache[name], key, value];
CacheUnset[name_String, key_, value_] := CacheUnset[Cache[name], key];


PackageExport["CacheKeys"]
PackageExport["iKeys"]

CacheKeys[name_String] := iKeys[Cache[name]];
CacheKeys[sym_Symbol] := iKeys[sym];
iKeys[sym_Symbol] := iKeys[sym, CacheData[sym]];


PackageExport["CacheInformation"]
PackageScope["iInfo"]

CacheInformation[symbol_] := toInfo @ iInfo @ CacheData @ symbol;
CacheInformation[symbol_, prop_] := Lookup[CacheInformation[symbol], prop];

toInfo[x_] := $Failed;
toInfo[{a_List, b_List}] := Association[a, b];

(* why am i doing this? *)
arrangeBox = ToExpression["BoxForm`ArrangeSummaryBox"];
makeItem = ToExpression["BoxForm`MakeSummaryItem"];

FormatCacheSymbol[sym_] := Dynamic[
	With[{co = CacheData[sym]},
		If[co === $Failed, Removed["Caches`" <> SymbolName @ sym],
		RawBoxes @ arrangeBox[Head[co], None, 
			Item[If[TrueQ[CacheDirtyQ[sym]],
				dirtyGraphic, cleanGraphic], 
				ItemSize -> {3,3},
				Alignment -> Center], 
			makeGrid @ iInfo[co], StandardForm]
		]
	]
];

iInfo[_] := {};

makeGrid[list_List] := Sequence @@ Map[
	makeItem[{ToString[#1] <> ": ", #2}, StandardForm]& @@@ #&,
	list
];

dirtyGraphic = Style["\[FilledCircle]", Orange, Large];
cleanGraphic = Style["\[FilledCircle]", Darker[Green], Large];


PackageExport["makeCacheSymbol"]
PackageExport["makeCacheFile"]

makeCacheSymbol[name_] := Symbol["Caches`" <> $CurrentCacheName <> "`" <> name];
makeCacheFile[name_] := ToFileName[CacheDirectory[$CurrentCacheName], name];
makeCacheDirectory[name_] := EnsureDirectory @ FileNameJoin[{CacheDirectory[$CurrentCacheName], name}];

PackageExport["CacheDirectory"]

CacheDirectory[name_] := EnsureDirectory[{$CacheBaseDirectory, name}];


PackageExport["$CacheBaseDirectory"]

$CacheBaseDirectory := $CacheBaseDirectory = EnsureDirectory[{$UserBaseDirectory, "Caches"}];


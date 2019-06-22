Package["NeuralNetworks`"]


(* 

SPECIAL VALUES

These undocumented content elements can be useful for debugging:

"LocalContentElements"
"LocalPath"
"Metadata"

These special 'models' are also useful:

"CachedModelNames"
"DownloadedModelNames"

*)

PackageExport["$ContentElementRedirect"]

SetUsage @ "
$ContentElementRedirect['elem$'] is a value returned by NetModel functions that implement \
EvaluationNet etc. that indicates that the real value is found in a separate content \
element named 'elem$'.
$ContentElementRedirect['elem$', f$] indicates that the function f$ should be applied to \
the net after it is loaded from 'elem$'."


PackageExport["NetModel"]
PackageExport["NetModelInternal"]

ClearAll[NetModel, NetModelInternal, LocalNetModel, iLocalNetModel];

LoadResourceSystemClient := Block[{$ContextPath = $ContextPath},
	If[Check[Needs["ResourceSystemClient`"]; True, False],
		Clear[LoadResourceSystemClient]
	]
];

$DefaultElement$ = "$DefaultElement$";

$ROCache = <||>;

General::invhttp = "Couldn't connect to server."

ValidROQ = ResourceSystemClient`Private`resourceObjectQ;

SetUsage @ "
NetModelInternal[symbol$, $$] does the same as NetModel[$$], without argument checking and displaying \
a message with the head symbol$ if the network cannot be loaded."

NetModelInternal[head_Symbol, model_String, elem_String] := Scope[
	If[head === NetModel && !MemberQ[$defaultContentElements, elem] && AssociationQ[index = NetModelIndex[]],
		(* if we already have the data via the index mechanism, return it immediately and
		skip resource system or local objects completely *)
		result = index[model, elem];
		If[!MissingQ[result], Return @ result];
	];

	If[elem === "Properties", Goto[SkipLocalFetch]];
	(* ^ the RO simulator is irrelevant, we want to report what Resource System thinks and then
	filter out some irrelevant stuff then add our own *)

	(* first we check if we have cached the result we need *)
	key = {model, elem};
	result = $LocalNetModelCache[key];
	If[!MissingQ[result], Goto @ SkipServerFetch];
	
	(* next we try service the request locally. this is a good simulation of how resource system
	works, if the required objects are present locally we won't ever need to load resource system client,
	and we can also avoid any network requests *)
	result = LocalNetModel[model, elem];

	(* if this succeeded, we don't have to load resource system and talk to the server *)
	If[!FailureQ[result], Goto @ SkipServerFetch];

	Label[SkipLocalFetch];
	If[!PacletManager`$AllowInternet, 
		Message[head::offline];
		Return[$Failed]];

	(* we will need to contact the server; load resource system *)
	LoadResourceSystemClient;
	Block[{PrintTemporary}, Quiet[
		ro = CacheTo[$ROCache, model, ResourceObject["NeuralNet" -> model]];
		If[FailureQ[ro], KeyDropFrom[$ROCache, model]];
		messages = $MessageList;
	]];
	If[ValidROQ[ro] =!= True,
		Which[
			MemberQ[messages, HoldForm[MessageName[_, "offline"]]],
				Message[head::offline],
			MemberQ[messages, HoldForm[URLFetch::invhttp]],
				Message[head::invhttp],
			MemberQ[messages, HoldForm[ResourceAcquire::cloudc]],
				Message[head::rscloudc],
			MemberQ[messages, HoldForm[ResourceAcquire::apierr]],
				noNetModel[head, model, NetModelNames[]],
			True,
				Message[head::interr,
					StringForm["An unexpected error occured on NetModel[\"``\", \"``\"]", model, elem]
				]
		];
		Return[$Failed]
	];

	(* Check $DynamicEvaluation here and switch to local-only mode when that happens, issuing a message as appropriate *)
	result = Which[
		elem === "ResourceObject",                  ro,
		elem === "LocalPath",                       None,
		elem === "Metadata",                        ro[All],
		elem === "Properties",                      niceROproperties @ ro,
		elem === $DefaultElement$,					quietResourceData[head, ro, Automatic],
		MemberQ[$NetModelInformationKeys, elem],    getComputedProp[quietResourceData[head, ro, "UninitializedEvaluationNet"], elem],
		MemberQ[$defaultContentElements, elem],     quietResourceData[head, ro, elem],
		MemberQ[$defaultProperties, elem],          tryROprop[ro, model, elem],
		MemberQ[ro["ContentElements"], elem],       quietResourceData[head, ro, elem],
		MemberQ[ro["Properties"], elem],            tryROprop[ro, model, elem],
		True,                                       invOrMissProp[model, elem]; $Failed
	];

	Label[SkipServerFetch];

	(* we might have got a function back from ResourceData, if so engage the parameterized model code path *)
	If[MatchQ[result, _Function] && MatchQ[elem, $netElemPattern | $DefaultElement$] && !$withinParameterizedLookup,
		result = ParameterizedNetModelInternal[head, {model}, elem]];

	(* cache the final result *)
	addToLocalNetModelCache[key, result];

	result
]

niceROproperties[ro_] := Scope[
	props = ro["Properties"];
	If[!ListQ[props], Return @ props];
	props = Join[props, Intersection[ro["ContentElements"], $defaultContentElements]];
	If[AssociationQ[pdata = ro[All]["ParameterizationData"]],
		props = Join[props, 
			Lookup[pdata, "ParameterizedContentElements", {}], 
			List @@ $ParameterizedModelPropertyP
		]
	];
	props = DeleteCases[props, $badROProps];
	Sort @ props
];

$badROProps = "ContentValues" | "RepositoryLocation" | "MyAccount" | "ResourceType" | "DefaultContentElement" | 
	"ContentElementLocations" | "ContentElements" | "Attributes" | "Format" | "ContentSize" | 
	"InformationElements" | "ResourceLocations" | "AutoUpdate" | "ParameterizedContentElements" |
	"ParameterizationData" | "ContributorInformation";
(* to strip off the wonky ResourceObject-related properties and internal propertiesthat we don't want users to see *)

tryROprop[ro_, model_, prop_] := Scope[
	val = Quiet @ ro[prop];
	If[FailureQ[val], invOrMissProp[model, prop], val]
];

NetModel::invprop = "The second argument `1` is not a known property for model ``. Typical properties include the following: ``."
NetModel::missprop = "The property `` is not available for model ``. The available properties for this model include: ``."
invOrMissProp[model_, prop_] := 
	If[MemberQ[$typicalProperties, prop],
		Message[NetModel::missprop, prop, model, Row[niceROproperties[ResourceObject[model]], ","]],
		Message[NetModel::invprop, prop, model, Row[$typicalProperties, ","]]
	];
General::dllock = "The requested neural network model is currently downloading in another \
process, try again when it is finished. If there is not a download in progress run \
DeleteObject[ResourceObject[``]] before trying again."

quietResourceData[head_, ro_, elem_] := Scope[
	$MessageList = {};
	Quiet[
		res = If[elem === Automatic, ResourceData[ro], ResourceData[ro, elem]];
		messages = $MessageList,
		{Import::wlnetcorr, ResourceData::invelem, ResourceData::dllock}
	];
	Which[
		MemberQ[messages, HoldForm[Import::wlnetcorr]],
			Message[head::wlnetcorr2],
		MemberQ[messages, HoldForm[ResourceData::invelem]],
			invOrMissProp[ro["Name"], elem],
		MemberQ[messages, HoldForm[ResourceData::dllock]],
			Message[head::dllock, ro["Name"]]
	];
	res
]

getComputedProp[net_ ? ValidNetQ, prop_] := NetInformation[net, prop];
getComputedProp[other_, _] := other;

$ParameterizedModelPropertyP = "Variants"|"DefaultVariant"|"ParameterNames"|"ParametersAllowedValues"|"ParametersInformation";

NetModelInternal[head_Symbol, model_String, prop:$ParameterizedModelPropertyP] :=
	ParameterizedNetModelInternal[head, {model}, prop]; 
(* force supermodel property lookups to go through supermodel code *)

NetModelInternal[head_Symbol, spec:{_String, ___Rule}, elem_] :=
	ParameterizedNetModelInternal[head, spec, elem];

$withinParameterizedLookup = False;

ParameterizedNetModelInternal[head_Symbol, arg:{model_String, params___Rule}, elem_String] := Scope[
	
	(* this prevents NetModelInternal from recursing back to us when we ask it for the function
	value itself *)
	$withinParameterizedLookup = True;

	metadata = Quiet @ NetModelInternal[head, model, "Metadata"];
	If[!AssociationQ[metadata], 
		TestNetModelName[model, head];
		ReturnFailed[]];

	If[elem === $DefaultElement$, elem = Lookup[metadata, "DefaultContentElement", "EvaluationNet"]];

	paramData = metadata["ParameterizationData"];

	Which[
		ByteArrayQ[paramData], paramData = BinaryDeserialize[paramData],
		MissingQ[paramData] || paramData === None, Null,
		True, Panic["BadParameterization", "ParameterizationData of `` should be a ByteArray.", model]
	];
	
	If[!AssociationQ[paramData], 
		If[MatchQ[elem, $ParameterizedModelPropertyP],
			Return @ paramNetModelProperty[<||>, elem, arg]];
		Return @ loadPlainModelWithHints[head, model, elem, {params}]
	];

	If[MatchQ[elem, $ParameterizedModelPropertyP],
		Return @ paramNetModelProperty[paramData["Parameters"], elem, arg]];

	UnpackAssociation[paramData, parameters, inputs, outputs];

	inputNames = Keys[inputs]; outputNames = Keys[outputs];
	paramTypes = Part[parameters, All, "Type"];
	defaults = DeleteMissing @ Part[parameters, All, "Default"];

	uparams = Association[params];
	params = Join[defaults, uparams];

	MapAssocAssoc[CoerceParam, KeyDrop[params, Union[inputNames, outputNames]], paramTypes, 
		ThrowFailure["pnmextra", model, #, QuotedStringRow[Keys[parameters], " and "]]&,
		ThrowFailure["pnmmiss", model, #]&
	];

	(* mods are things that modify param(s) of an existing coder, whereas hints entirely replace the input *)
	userInputHints = KeyTake[uparams, inputNames];
	userOutputHints = KeyTake[uparams, outputNames];
	portHints = If[userInputHints === <||> && userOutputHints === <||>, None,
		Join[
			KeyValueMap[#1 -> If[RuleVectorQ @ #2, #2, ParseInputSpec[#1, inputs[#1], #2]]&, userInputHints],
			KeyValueMap[#1 -> If[RuleVectorQ @ #2, #2, ParseOutputSpec[#1, outputs[#1], #2]]&, userOutputHints]
		]];

	contentElements = metadata["ContentElements"];

	If[elem === "ByteCount", elem = "ArraysTotalByteCount"]; (* <- close enough... *)
	If[MemberQ[$NetModelInformationKeys, elem],
		isComputed = True;
		computedElem = elem;
		elem = metadata["DefaultContentElement"]
	];

	If[!MemberQ[contentElements, elem], 
		invOrMissProp[model, elem];
		ThrowRawFailure[$Failed]];

	function = NetModelInternal[head, model, elem];
	If[Head[function] =!= Function, ReturnFailed[]];

	result = function[params];

	(* Parametrized NetModels can return Failure objects,
	   these are informative failures we need to return to top level *)
	If[Head[result] === Failure, Return @ result];

	If[Head[result] === $ContentElementRedirect,
		redirect = result;
		result = NetModelInternal[head, model, First @ redirect];
		If[Length[redirect] === 2 && ValidNetQ[result], result = Last[redirect] @ result]
		(* ^ apply finalizer function, if any *)
	];

	If[!ValidNetQ[result], ReturnFailed[]];

	If[portHints =!= None && ValidNetQ[result],
		result = NetReplacePart[result, portHints]];

	If[isComputed === True, 
		result = NetInformation[result, computedElem]];

	result
];

General::notpnetmodel = "Rules `` were supplied, but `` is not a parameterizable model, and the rules do not correspond to input or output ports.";

loadPlainModelWithHints[head_, model_, elem_, portHints_List] := Scope[
	(* e.g. NetModel[{"LeNet", "Input" -> {"ImageSize" -> 32}}] *)

	net = NetModelInternal[head, model, elem];

	If[!ValidNetQ[net], ReturnFailed[]];

	names = Join[InputNames @ net, OutputNames @ net];

	If[!SubsetQ[names, Keys @ portHints],
		ThrowFailure["notpnetmodel", CoderForm @ portHints, model]];

	NetReplacePart[net, portHints]
];

paramNetModelProperty[params_, "Variants", uargs_] := Scope[
	paramSettings = Map[
		Match[#Type, EnumT[l_] :> l, BooleanT :> {False, True}, _ :> None]&,
		KeyDrop[params, Keys @ Rest[uargs]]
	];
	{keys, vals} = KeysValues @ DeleteCases[None] @ paramSettings;
	name = First[uargs];
	If[vals === {}, Return @  {desingle @ uargs}];
	Prepend[Thread[keys -> #], name]& /@ Tuples[vals]
];

desingle[{e_String}] := e;
desingle[e_] := e;

paramNetModelProperty[params_, "DefaultVariant", uargs_] := 
	desingle @ Prepend[Normal @ Join[params[[All, "Default"]], Association @ Rest @ uargs], First @ uargs];

paramNetModelProperty[params_, "ParameterNames", _] := Keys @ params;

paramNetModelProperty[params_, "ParametersAllowedValues", _] := toTypeString /@ params[[All, "Type"]];

paramNetModelProperty[<||>, "ParametersInformation", _] := None;
paramNetModelProperty[params_, "ParametersInformation", _] := Scope[
	Map[
		Association[
			"Description" -> #Description, 
			"Default" -> Lookup[#, "Default", None], 
			"Allowed values" -> toTypeString[#Type]
		]&,
		params
	] // Dataset
];

toTypeString[EnumT[l_List]] := l;
toTypeString[e_] := TypeString[e];

General::wlnetcorr2 = "A required WLNet file is corrupted and could not be loaded."

$defaultContentElements = Sort @ {
	"EvaluationNet", "UninitializedEvaluationNet", 
	"ConstructionNotebook", "ConstructionNotebookExpression",
	"EvaluationExample", "TrainingNet"
};

(* Add to this list when a new property gets included. ResourceObjects have dozens of weird
non-user-relevant properties, plus we want to know if the user asked for something reasonable
that wasn't there but should have, versus if they asked for something bizarre, and tailor 
the message appropriately. would be nice if there was a database with a schema backing this
but hey! in his wisdom SW said no databases. *)

$defaultProperties = Sort @ {
	"Description", "DocumentationLink", "ByteCount", "InputDomains", "SourceMetadata", 
	"TaskType", "TrainingSetInformation", "ExampleNotebook", 
	"Performance", "Details", "ParameterizationData", "Properties", 
	"Originator", "ReleaseDate"
};

$typicalProperties = Sort @ Join[
	DeleteCases[$defaultContentElements, "ConstructionNotebookExpression"],
	DeleteCases[$defaultProperties, "ParameterizationData"]
];
	
NetModelInternal[head_Symbol, model_] := NetModelInternal[head, model, $DefaultElement$]

General::nonmind = "Could not obtain the NetModel index. Please ensure you have a working internet connection."
General::cached = "Could not update the NetModel index from Wolfram Research servers. Results may be out of date."

NetModel[All] := NetModel[];

NetModel[] := Scope[
	$NetModelIndexUpdateInterval = 60*60;
	failures = $NetModelIndexUpdateFailures;
	res = NetModelNames[];
	If[FailureQ[res], 
		Message[NetModel::nonmind],
		If[$NetModelIndexUpdateFailures > failures,
			Message[NetModel::cached]]];
	res
];

NetModel["CachedModelNames"] := NetModelNames[];

NetModel["DownloadedModelNames"] := LocalNetModel[];

NetModel[HoldPattern @ ResourceObject[assoc_Association, ___], elem_:$DefaultElement$] :=
	NetModel[assoc["Name"], elem];

NetModel[All, prop_?StringQ] := If[!MemberQ[$typicalProperties, prop],
	Message[NetModel::invprop, prop, $typicalProperties]; $Failed,
	Association @ Map[# -> NetModel[#, prop]&, DeleteCases["LeNet"] @ NetModelNames[]]
];

modelSpecP = _String ? StringQ | {_String ? StringQ, __Rule};

NetModel[models:{Repeated[modelSpecP]}] := 
	NetModel[models, $DefaultElement$];

NetModel[models:{Repeated[modelSpecP]}, prop_?StringQ] := 
	Map[NetModel[#, prop]&, models];

NetModel::arg1 = "First argument to NetModel, which was ``, should be either a model name, a model name with parameters, a list of models, or All.";
NetModel::arg2 = "Second argument to NetModel, which was ``, should be a valid property." 

NetModel::pnmextra = "Model `` does not have a parameter called ``. Available parameters are: ``.";
NetModel::pnmmiss = "Model `` requires a value for parameter ``, but one was not provided.";

NetModel[model_] := CatchFailureAsMessage @ Which[
	KeyExistsQ[$BuiltinNets, model],
		Return @ $BuiltinNets[model],
	!MatchQ[model, modelSpecP],
		Message[NetModel::arg1, model]; $Failed,
	True,
		NetModelInternal[NetModel, model]
];

(* we take port hints as options, but they are really part of the model spec *)
NetModel[model_String, prop__Rule] := 
	NetModel[{model, prop}];

NetModel[model_String, elem_String, prop__Rule] := 
	NetModel[{model, prop}, elem];

NetModel[model_, prop_] := CatchFailureAsMessage @ Which[
	!MatchQ[model, modelSpecP], 
		Message[NetModel::arg1, model]; $Failed,
	!StringQ[prop], 
		Message[NetModel::arg2, prop]; $Failed,
	True, 
		NetModelInternal[NetModel, model, prop]
];

DeclareArgumentCount[NetModel, {0, 2}];


PackageExport["LocalNetModel"]

SetUsage @ "
LocalNetModel[model$, elem$] is like NetModel[model$, elem$], but uses only local resources and will \
not connect to the internet or load expensive paclets like CloudObject.
* if LocalNetModel does not have the requested resources cached, it will return $Failed.
* LocalNetModel does not handle directly parameterized NetModels, whose net keys are functions. It just \
returns these functions to be dealt with by NetModelInternal."

LocalNetModel[model_String] := 
	LocalNetModel[model, $DefaultElement$];

LocalNetModel[model_String, elem_String] := Scope[
	indexCache = $NetModelIndexCached;
	If[AssociationQ[indexCache] && !MissingQ[result = indexCache[model, elem]], Return[result]];
	If[elem === $DefaultElement$ && AssociationQ[indexCache] && KeyExistsQ[indexCache, model], 
		elem = Lookup[indexCache[model], "DefaultContentElement", "EvaluationNet"]];
	key = {model, elem};
	cached = $LocalNetModelCache[key];
	If[!MissingQ[cached], Return[cached]];
	result = iLocalNetModel[model, elem];
	addToLocalNetModelCache[key, result];
	result
];

LocalNetModel[] := Block[{WLNetImport = Function[True], Import = Function[True]},
	If[TrueQ @ iLocalNetModel[#, $DefaultElement$], #, Nothing]& /@ NetModelNames[]
];


PackageExport["ClearNetModelCache"]

PackageScope["$MaxNetModelCacheByteCount"]
PackageScope["$LocalNetModelCache"]

SetUsage @ "
ClearNetModelCache[] clears all models cached in memory.
ClearNetModelCache['patt$'] clears all models whose name matches the string 'patt$'.
* The number of entries cleared is returned."

$LocalNetModelCache = <||>;
If[!FastValueQ[$MaxNetModelCacheByteCount], $MaxNetModelCacheByteCount = If[$CloudEvaluation, 100*^6, 1000*^6]];

ClearNetModelCache[] := Block[{count = Length[$LocalNetModelCache]}, $LocalNetModelCache = <||>; count]

ClearNetModelCache[model_String] := Block[{count = 0},
	$LocalNetModelCache = KeySelect[$LocalNetModelCache, If[StringMatchQ[First @ #, model], count++; False, True]&];
	count
];

(* if a result has already been cached, its entry gets put at the end *)
addToLocalNetModelCache[key_, result_] /; ValidNetQ[result] || Head[result] === Function := (
	AppendTo[$LocalNetModelCache, key -> result];
	While[
		(* drop items to try bring us down under the limit... but don't drop the thing we just cached *)
		Length[$LocalNetModelCache] > 1 && ByteCount[$LocalNetModelCache] > $MaxNetModelCacheByteCount, 
		$LocalNetModelCache = Rest[$LocalNetModelCache];
	];
);

iLocalNetModel[model_, elem_] := Scope[
	
	(* first try to resolve the name to a UUID *)
	obj = LocalObject["Persistence/ResourceNames/" <> Hash[model,"Expression","HexString"]];
	value = Quiet @ Check[Get[obj], $Failed];
	If[!AssociationQ[value], ReturnFailed[]];
	uuid = ReleaseHold @ Lookup[value, "HeldValue", $Failed];
	If[!StringQ[uuid], ReturnFailed[]];

	(* now try look up the dir corresponding to the UUID *)
	path = findResourceObjectPath[uuid];
	If[FailureQ[path], ReturnFailed[]];

	If[elem === "LocalPath", Return[path]];

	If[elem === "LocalContentElements", 
		dirs = Select[DirectoryQ] @ FileNames["*", FileNameJoin[{path, "download"}]];
		Return[URLDecode[FileBaseName[#]]& /@ dirs]
	];

	If[!FileExistsQ[path], ReturnFailed[]];

	modelMetadata = None;

	If[elem === $DefaultElement$,
		modelMetadata = getResourceObjectMetadata[path];
		If[FailureQ[modelMetadata], ReturnFailed[]];
		elem = Lookup[modelMetadata, "DefaultContentElement", "EvaluationNet"]];

	If[!MemberQ[$defaultContentElements, elem],
		If[!AssociationQ[modelMetadata],
			modelMetadata = getResourceObjectMetadata[path]];
		If[elem === "Metadata", Return @ modelMetadata];
		If[FailureQ[modelMetadata], ReturnFailed[]];
		result = Lookup[modelMetadata, elem, $Failed];
		If[!FailureQ[result], Return @ result];
		result = Lookup[Lookup[modelMetadata, "ContentValues", <||>], elem, $Failed];
		If[!FailureQ[result], Return @ result];
	];

	$modelName = model; $modelUUID = uuid; (* <- used to delete corrupt models *)
	result = If[MemberQ[$NetInformationProperties, elem], 
		getComputedProp[getContentElement[path, "UninitializedEvaluationNet"], elem],
		getContentElement[path, elem]
	];
	(* ^ these don't work for non-local models *)

	result
];

findResourceObjectPath[uuid_]:=StringJoin["Resources/", StringTake[uuid, 3], "/", uuid];

findResourceObjectPath[uuid_]:=findResourceObjectPath[$SystemWordLength,uuid]

findResourceObjectPath[bits_,uuid_] := Scope[
	path = LocalObjects`PathName[LocalObject @ resourceObjectPath[bits, uuid]];
	If[!FileExistsQ[path],
		If[bits===64,ReturnFailed[],resourceObjectPath[64,uuid]], 
		path
	]
];

resourceObjectPath[64, uuid_]:=StringJoin["Resources/", StringTake[uuid, 3], "/", uuid]
resourceObjectPath[_, uuid_]:=StringJoin["Resources32/", StringTake[uuid, 3], "/", uuid]

(* 
TODO: put this in GeneralUtilities.
This bypasses ResourceSystemClient, which is slow to load and will wantonly do network requests that by definition
we are trying to avoid in this fast path. This function will try to load a content element, using the same logic
as the current ResourceSystem. In particular it simulates ContentElementFunctions. *)

NetModel::modelcorrupt = "Cached version of model \"`1`\" appears to be corrupt. Remove the downloaded files using DeleteObject[ResourceObject[`1`]] and try again.";

getContentElement[roPath_, elem_] := Scope[
	elementPath = FileNameJoin[{roPath, "download", URLEncode @ URLEncode @ elem}];
	If[!FileExistsQ[elementPath], ReturnFailed[]];
	elementMetadata = getResourceObjectMetadata[elementPath];
	If[FailureQ[elementMetadata], ReturnFailed[]];
	{format, formats} = Lookup[elementMetadata, {"Format", "Formats"}, None];
	If[!StringQ[format] && MatchQ[formats, {_String, ___}], format = First[formats]];
	(* there are two possibilities: the content is provided in a specific file format on disk (more than one is 
	actually possible), or it is constructed by a function that has acess to the other content elements. *)
	(* first possibility: element is loaded from disk *)
	If[format === Automatic,	
		(* this exists for local testing, basically. because of bugs in resource system *)
		hash = Hash[Automatic, Automatic, "HexString"];
		elementFormatMetadataPath = FileNameJoin[{elementPath, hash, "metadata", "put.wl"}];
		If[!FileExistsQ[elementFormatMetadataPath], ReturnFailed[]];
		info = Quiet @ Check[Get[elementFormatMetadataPath], $Failed];
		If[!AssociationQ[info] || !KeyExistsQ[info, "Location"], ReturnFailed[]];
		{dataPath, format} = Lookup[info, {"Location", "Format"}]; 
		(* ^ Bobs' code does this for local ROs *)
		result = Quiet @ Check[Import[dataPath, format], $Failed];
		(* ^ directly import the file *)
		Return[result];
	];
	If[StringQ[format],
		hash = Hash[format, Automatic, "HexString"];
		elementFormatPath = FileNameJoin[{elementPath, hash}];
		Label[retryInfo];
		infoPath = FileNameJoin[{elementFormatPath, "object.wl"}];
		If[!FileExistsQ[infoPath], 
			elementFormatPath = FileNameJoin[{elementFormatPath, "data"}];
			If[!FileExistsQ[elementFormatPath], ReturnFailed[]];
			Goto[retryInfo];
		];
		(* ^ LocalObject, or at least V1 of it, provides an object.wl file that explains how to import
		the local object *)
		info = Quiet @ Check[Get[infoPath], $Failed];
		If[info["Version"] =!= 1, ReturnFailed[]];
		getter = info["Get"];
		If[!MissingQ[getter] && getter =!= ResourceSystemClient`Private`importRaw, ReturnFailed[]];
		(* ^ check that we're looking at a V1 LocalObject and that the resource system would just try to use 
		ordinary Import on the corresponding data file. we don't want to simulate any other behavior *)
		externalData = info["ExternalData"];
		(* ^ get the location of the actual data *)
		If[!StringQ[externalData], ReturnFailed[]];
		dataPath = FileNameJoin[{elementFormatPath, externalData}];
		If[!FileExistsQ[dataPath], ReturnFailed[]];
		result = Switch[format,
			"WLNet", WLNetImport[dataPath],
			(* ^ slightly faster than Import *)
			"Binary", Check[Developer`ReadWXFFile[dataPath], $Failed],
			(* ^ resource system uses Binary files as the carrier for WXF (we could also use Import with WXF format) *)
			_, Check[Import[dataPath, format], $Failed]
		];
		(* ^ directly import the file *)
		If[format === "WLNet" && FailureQ[result],
			Message[NetModel::modelcorrupt, $modelName];
			(* jeromel: Too dangerous to do this
			(* TODO: make this more selective. if you have a huge parameterized net model,
			we don't want to delete *all* of the content elements, just the corrupt one *)
			ResourceRemove[$modelUUID];
			*)
		];
		Return[result]
	];
	(* there was code to simulate ContentElementFunction here, but removed it since it was not
	being used *)
	$Failed
];

getResourceObjectMetadata[path_] := Scope[
	data = Quiet @ Check[Get @ FileNameJoin[{path, "metadata", "put.wl"}], $Failed];
	If[!AssociationQ[data], $Failed, data]
];


PackageScope["NetModelRemove"]

SetUsage @ "
NetModelRemove[] removes all locally cached NetModels."

NetModelRemove[] := Scope[
	LoadResourceSystemClient;
	res = Select[
		ResourceSystemClient`Private`$localResources, 
		(ResourceObject[#]["ResourceType"] === "NeuralNet")&
	];
	ResourceRemove /@ res
]


PackageScope["NetModelIndex"]

$NetModelIndexUpdateFailures = 0;

$NetModelIndexURL = StringJoin[
	"https://www.wolframcloud.com/objects/resourcesystem/published/NeuralNetRepository/",
	StringRiffle[Take[FromVersionString @ $NeuralNetworksVersionNumber, 2], "-"], (* <- e.g. 11-3 *)
	"/resourceinformation"
];
(* if this starts to fail, open a ticket like for https://jira.wolfram.com/jira/browse/NNETREPO-266 *)

$NetModelLocalObject := $NetModelLocalObject = LocalObject["Persistence/NetModelIndex"];

$NetModelIndexUpdateInterval = 7*24*60*60; (* seven days *)

$NetModelIndexLastUpdate := $NetModelIndexLastUpdate = 
	If[!FileExistsQ[$NetModelLocalObject], 0, 
		AbsoluteTime @ FileDate[$NetModelLocalObject]];

$NetModelIndexCached :=
	If[!FileExistsQ[$NetModelLocalObject], 
		$NetModelIndexCached = $Failed, 
		Get @ $NetModelLocalObject; $NetModelIndexCached];

toPath[HoldPattern @ lo_LocalObject] := LocalObjects`AuxPathName[lo];

NetModelIndex[] := Block[{file, data},
	If[$NetModelIndexLastUpdate > AbsoluteTime[] - $NetModelIndexUpdateInterval,
		Return @ $NetModelIndexCached];
	If[!TrueQ[PacletManager`$AllowInternet], 
		$NetModelIndexUpdateFailures++;
		Return @ $NetModelIndexCached];
	
	file = Quiet @ TimeConstrained[URLDownload[$NetModelIndexURL], 15]; (* had to work around 345989 *)

	If[!FailureQ[file], 
		data = Quiet @ Check[readIndex @ file, $Failed];
	];

	If[AssociationQ[data],
		$NetModelIndexLastUpdate = AbsoluteTime @ FileDate @ file;
		$NetModelIndexCached = data;
		DumpSave[$NetModelLocalObject, $NetModelIndexCached];
		UpdateNetModelAutocompleteData[];
	,
		$NetModelIndexUpdateFailures++;
		Quiet @ DeleteFile[file];
	];
	$NetModelIndexCached
];


PackageScope["UpdateNetModelAutocompleteData"]

$NetModelAutocompleteFile = FileNameJoin[{$NNCacheDir, "NetModelAutocomplete.mx"}];

UpdateNetModelAutocompleteData[] := If[$Notebooks,
	$NetModelAutocompleteData = "NetModel" -> {NetModelNames[], $typicalProperties};
	applyAutoData[];
	Quiet[
		EnsureDirectory[$NNCacheDir];
		DumpSave[$NetModelAutocompleteFile, $NetModelAutocompleteData];
	];
	$NetModelAutocompleteData
];

LoadNetModelAutocompleteData[] := If[FileExistsQ[$NetModelAutocompleteFile], 
	Get[$NetModelAutocompleteFile];
	applyAutoData[]
];

applyAutoData[] := Compose[FE`Evaluate, FEPrivate`AddSpecialArgCompletion[$NetModelAutocompleteData]];

RunInitializationCode[LoadNetModelAutocompleteData[]];

(* ^ this still only takes affect when the kernel loads. i should instead dump things in a .tr file in
the Paclet directory itself *)


PackageScope["NetModelNames"]

NetModelNames[] := Scope[
	list = Replace[NetModelIndex[], assoc_Association :> Keys[assoc]];
	If[!VectorQ[list, StringQ], Return[$Failed]];
	Sort @ Append[list, "LeNet"]
];

readIndex[file_] := Scope[
	assocs = BinaryDeserialize @ ReadByteArray @ file;
	If[!AssociationVectorQ[assocs], $Failed,
		Association[
			#Name -> KeyDrop[#, "Name"]& /@ assocs
		]
	]
];


PackageExport["UpdateNetModelIndex"]

UpdateNetModelIndex[] := (
	$NetModelIndexLastUpdate = 0;
	$NetModelIndexCached = $Failed;
	NetModelIndex[]
);


PackageScope["TestNetModelName"]

General::notmname = "`` is not the name of a net model."
General::notmname2 = "`` is not the name of a net model. Did you mean ``?"
General::invnmname = "Net model `` should be a string."

TestNetModelName[name_, head_] := Scope[
	names = NetModelNames[];
	Which[
		!StringVectorQ[names], Message[head::nonmind, name], 
		!StringQ[name], Message[head::invnmname, name],
		!MemberQ[names, name], noNetModel[head, name, names],
		True, Return[True]
	];
	False
];

If[$Notebooks,
	ssq[s_] := Style[s, ShowStringCharacters -> True],
	ssq[s_] := s
];

General::nonetmodel = "No model with name `` could be found."
General::nonetmodel2 = "No model with name `` could be found. Did you mean ``?"

noNetModel[head_, name_, names_] := Scope[
	If[StringVectorQ[names], 
		nearest = findPossibleModelName[name, names]];
	If[StringQ[nearest], 
		If[head === NetTrain && StringContainsQ[nearest, " Trained "], 
			nearest = First @ StringSplit[nearest, " Trained "]];
		Message[head::nonetmodel2, ssq[name], ssq[nearest]],
		Message[head::nonetmodel, ssq[name]]
	]
];

findPossibleModelName[name_, names_] := Scope[
	nearest = First @ Nearest[
		(ToLowerCase @ names) -> names, ToLowerCase @ name,
		DistanceFunction -> (-StringLength[LongestCommonSubsequence[#1, #2]]&)
	];
	min = Min[4, StringLength[name]];
	If[StringLength @ LongestCommonSubsequence[ToLowerCase @ nearest, ToLowerCase @ name] >= min,
		nearest, None
	]
];


PackageExport["CheckNetModelResourceDefinition"]

NetModelResourceDefinitionT = StructT[{
	"Name" -> StringT,
	"ResourceType" -> MatchT["NeuralNet"],
	"Description" -> StringT,
	"ParameterizationData" -> Defaulting @ MatchT[_ByteArray | None],
	"ContentElements" -> AssocT[StringT, ExpressionT],
	"DefaultContentElement" -> EnumT[{"EvaluationNet", "UninitializedEvaluationNet", "TrainingNet"}],
	"ContentElementLocations" -> Defaulting @ AssocT[StringT, MatchT[_String | File[_String]]],
	"ConstructionNotebookExpression" -> Defaulting @ ExpressionT,
	"InputDomains" -> Defaulting @ ListT[StringT],
	"TaskType" -> ListT[StringT],
	"TrainingSetData" -> Defaulting @ ExpressionT,
	"TrainingSetInformation" -> Defaulting @ ExpressionT,
	"WolframLanguageVersionRequired" -> Defaulting @ StringT
}]

ParameterizationDataT = StructT[{
	"Inputs" -> AssocT[StringT, TypeT],
	"Outputs" -> AssocT[StringT, TypeT],
	"Parameters" -> AssocT[StringT, StructT[{
		"Type" -> TypeT,
		"Default" -> ExpressionT,
		"Description" -> StringT
	}]]
}]

CheckNetModelResourceDefinition::notassoc = "Please provide the association that you would pass to ResourceObject.";
CheckNetModelResourceDefinition[_] := (
	Message[CheckNetModelResourceDefinition::notassoc];
	$Failed
);

CheckNetModelResourceDefinition::nouninit = "An \"InitializedEvaluationNet\" has been given, but there is no \"UninitializedEvaluationNet\".";
CheckNetModelResourceDefinition::nedefaultce = "The specified default content element `` doesn't exist.";
CheckNetModelResourceDefinition::noparams = "There should be at least one parameter in ParameterizationData.";
CheckNetModelResourceDefinition::badparamdefault = "The parameter default for `` doesn't actually match the type ``."
CheckNetModelResourceDefinition::nocontent = "Model appears not to contain any actual content in the form of construction functions or actual nets."
CheckNetModelResourceDefinition::coder = "ParameterizationData contains a NetEncoder or NetDecoder, which is forbidden."
CheckNetModelResourceDefinition::hasparamvers = "If ParameterizationData is present, WolframLanguageVersionRequired must also be present and be equal to 12.0 or above."
CheckNetModelResourceDefinition[assoc_Association] := CatchFailure @ Scope[
	
	CoerceParam["a neural net ResourceObject", assoc, NetModelResourceDefinitionT];

	UnpackAssociation[assoc, contentElements, defaultContentElement];

	parameterizationData = Lookup[assoc, "ParameterizationData", None];
	contentElementLocations = Lookup[assoc, "ContentElementLocations", <||>];
	wolframLanguageVersionRequired = Lookup[assoc, "WolframLanguageVersionRequired", None];

	If[wolframLanguageVersionRequired === None, 
		If[parameterizationData =!= None, ThrowFailure["hasparamvers"]];
		If[VersionOrder["12.0.0", wolframLanguageVersionRequired] === -1, 
			ThrowFailure["hasparamvers"]]];

	If[ByteArrayQ[parameterizationData],
		parameterizationData = BinaryDeserialize[parameterizationData];
		CoerceParam["ParameterizationData", parameterizationData, ParameterizationDataT];
		If[!FreeQ[parameterizationData, NetEncoder | NetDecoder], ThrowFailure["coder"]];
		params = parameterizationData["Parameters"];
		If[Length[params] === 0, ThrowFailure["noparams"]];
		paramDefaults = params[[All, "Default"]];
		paramTypes = params[[All, "Type"]];
		{paramInputs, paramOutputs} = Lookup[parameterizationData, {"Inputs", "Outputs"}];
		KeyValueScan[
			If[!TestType[#2, paramTypes[#1]], ThrowFailure["badparamdefault", #1, paramTypes[#1]]]&,  
			paramDefaults
		]
	];

	If[Count[Keys @ contentElements, $netElemPattern] === 0, ThrowFailure["nocontent"]];

	KeyValueScan[checkContentElementField, contentElements];

	If[!KeyExistsQ[contentElements, defaultContentElement] && !KeyExistsQ[contentElementLocations, defaultContentElement], 
		ThrowFailure["nedefaultce", defaultContentElement]];

	If[KeyExistsQ[contentElements, "EvaluationNet"] && !KeyExistsQ[contentElements, "UninitializedEvaluationNet"],
		ThrowFailure["nouninit"]];

	Success["DefinitionValid",
		<|"MessageTemplate" :> "The definition of model `Name` is valid.",
		 "MessageParameters" -> <|"Name" -> assoc["Name"]|>|>
	]
];

Clear[checkContentElementField];

CheckNetModelResourceDefinition::noparamdata = "You can't provide the key `` if your model has no ParameterizationData.";
CheckNetModelResourceDefinition::funcissuedmessages = "The function provided for `` issued messages when called with the default parameter values."
CheckNetModelResourceDefinition::funcbadnet = "The function provided for `` did not construct a valid net when called with the default parameter values. Instead it produced:\n``"
CheckNetModelResourceDefinition::badnetio = "The function provided for `` produced a net whose `` didn't match the types provided in ParameterizationData.\nSpecifically the net's type `` was not a subtype of ``."
CheckNetModelResourceDefinition::missingceredirect = "The function provided for `` redirected to the content element ``, but this element does not exist."
CheckNetModelResourceDefinition::badceredirect = "The function provided for `` redirected to the content element ``, but this does not follow the convention of XXX:YYY, where XXX is a normal net content element."

$netElemPattern = "EvaluationNet" | "UninitializedEvaluationNet" | "TrainingNet";

CheckNetModelResourceDefinition::cenet = "You shouldn't provide a net for the ContentElement key `` via the ContentElements association. Please provide the path to the WLNet file as a string in ContentElementLocations association instead."
checkContentElementField[key:$netElemPattern, value_] := Scope[
	If[Head[value] =!= Function, ThrowFailure["cenet", key]];
	If[parameterizationData === None, ThrowFailure["noparamdata", key]];
	result = Check[
		value @ paramDefaults,
		ThrowFailure["funcissuedmessages", key]];
	If[MatchQ[result, $ContentElementRedirect[_String] | $ContentElementRedirect[_String, _]],
		cekey = First[result]; finalFunc = If[Length[result] === 2, Last[result], Identity];
		If[!StringStartsQ[cekey, $netElemPattern ~~ ":"], ThrowFailure["badceredirect", key, cekey]];
		result = Lookup[contentElements, cekey, None];
		If[result === None,
			loc = Lookup[contentElementLocations, cekey, ThrowFailure["missingceredirect", key, cekey]];
			result = Import[loc, "WLNet"]
		];
		result = finalFunc @ result;
	];
	If[!ValidNetQ[result], 
		ThrowFailure["funcbadnet", key, result]];
	inputs = Inputs[result];
	outputs = Outputs[result];
	If[!SubTypeQ[inputs, paramInputs], ThrowFailure["badnetio", key, "inputs", TypeString /@ inputs, TypeString /@ paramInputs]];
	If[!SubTypeQ[outputs, paramOutputs], ThrowFailure["badnetio", key, "outputs", TypeString /@ outputs, TypeString /@ paramOutputs]];
]

checkContentElementField[key_String /; StringStartsQ[key, $netElemPattern ~~ ":"], value_] := 
	ThrowFailure["cenet", key];


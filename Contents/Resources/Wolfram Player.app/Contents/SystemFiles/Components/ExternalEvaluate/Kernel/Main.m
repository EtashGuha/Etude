(* Wolfram Language Package *)

(* Created by the Wolfram Workbench Apr 21, 2017 *)

BeginPackage["ExternalEvaluate`",{"ZeroMQLink`","PacletManager`"}]
(* Exported symbols added here with SymbolName::usage *)

eeStringTemplateQ::usage ="";
eeStringTemplateEvalFunc::usage = "";

ExternalEvaluate::error = "Exception Thrown : `Type`.";
ExternalEvaluate::invalidSession = "The session `1` is invalid and cannot be used.";
ExternalEvaluate::unknownSys="`1` is not a known system in the ExternalEvaluate Framework.";
ExternalEvaluate::noinstall="No valid installations for system `1` were found with the options specified.";
ExternalEvaluate::invalidInput="The input `1` is invalid.";
ExternalEvaluate::invalidFile="File `1` not found.";
ExternalEvaluate::interpFail="The result `1` failed to be interpreted as a WL expression. Use \"ReturnType\"->\"String\" instead."
ExternalEvaluate::invalidObject = "The `1` should be a string to evaluate with ExternalEvaluate."
ExternalEvaluate::assockeys = "The keys `1` are `2`.";
ExternalEvaluate::nofilefunc = "The System `1` doesn't support evaluating files.";

ExternalSessionObject;
ExternalSessionObject::invalidSession = ExternalEvaluate::invalidSession;

StartExternalSession::unknownSys = ExternalEvaluate::unknownSys;
StartExternalSession::unknownOpts = "The options `1` aren't valid options for StartExternalSession.";
StartExternalSession::nosys = "No \"System\" option specified.";
StartExternalSession::invalidExec = "The \"Executable\" `1` doesn't exist.";
StartExternalSession::invalidOpt = "The value for option `1`, `2` is invalid.";
StartExternalSession::noinstall = ExternalEvaluate::noinstall;
StartExternalSession::replFail = "The process for external system `1` (`2`) failed to start due to : `3`.";
StartExternalSession::sockFail = "Failed to connect to the external system `1` (`2`) running on `3`.";
StartExternalSession::unregistered="The evaluator for `1` located at `2` has been unregistered. Use RegisterExternalEvaluator to use this evaluator.";
StartExternalSession::invalidReturnType="The \"ReturnType\" option `1` is invalid - valid return types are \"Expression\" and \"String\".";
StartExternalSession::invalidSpec = "The session specification `1` is invalid.";
StartExternalSession::version = "The installation version does not match.";
StartExternalSession::depend = "The installation does not have the required dependencies.";
StartExternalSession::nodepend = "Unable to find Language dependency test files.";
StartExternalSession::langopts = "Invalid options provided for this language: `1`";

FindExternalEvaluators::unknownSys = ExternalEvaluate::unknownSys;

RegisterExternalEvaluator::location = "The executable `1` isn't usable for ExternalEvaluate.";
RegisterExternalEvaluator::unknownSys = StartExternalSession::unknownSys;

UnregisterExternalEvaluator::location = RegisterExternalEvaluator::location;
UnregisterExternalEvaluator::unknownSys = RegisterExternalEvaluator::unknownSys;
UnregisterExternalEvaluator::invalidUUID = "The evaluator `1` doesn't exist.";

ExternalSessions::unknownSys = StartExternalSession::unknownSys;

Begin["`Private`"]
(* Implementation of the package *)



ExternalSessionObject/:MakeBoxes[session:ExternalSessionObject[uuid_String/;AtomQ[Unevaluated@uuid] && KeyExistsQ[uuid]@$Links],StandardForm|TraditionalForm]:=
With[{lang=$Links[uuid,"System"]},
	BoxForm`ArrangeSummaryBox[
		(*first argument is the head to use*)
		ExternalSessionObject,
		(*second argument is the expression*)
		ExternalSessionObject[uuid],
		(*third argument is the icon to use*)
		If[MissingQ[#],None, Show[#,ImageSize -> {Automatic, Dynamic[3.5*CurrentValue["FontCapHeight"]]}]]& @ $LanguageHeuristics[lang,"Icon"],
		(*the next argument is the always visisble properties*)
		{
			{
				BoxForm`SummaryItem[{"System: ",If[StringQ[lang],lang,None]}],
				BoxForm`SummaryItem[
					{
						"EvaluationCount: ",
						With[
							{sess=$SessionID},
							Dynamic[
								If[$SessionID===sess,
									session["EvaluationCount"],
									None
								],
								TrackedSymbols :> {$SessionID,$Links}
							]
						]
					}
				]
			},
			{BoxForm`SummaryItem[{"UUID: ",uuid}],SpanFromLeft}
		},
		(*the next argument is the optional items that come down when the plus button is pressed*)
		KeyValueMap[
			BoxForm`SummaryItem[{ToString[#1]<>": ",Normal@#2}]&,
			Join[
				<|
					"System"->lang,
					"EvaluationCount"->Dynamic[If[ValueQ[$Links]&&KeyExistsQ[$Links,uuid],$Links[uuid,"EvaluationCount"],None]],
					"UUID"->uuid
				|>,
				KeyDrop[{"SessionTime","EvaluationCount","ProcessMemory","ProcessThreads"}]@If[ValueQ[$Links]&&KeyExistsQ[$Links,uuid],
					KeyDrop["StopTime"]@$Links[uuid],
					AssociationMap[None&,getSessionOpts[ExternalSessionObject[session],"Properties"]]
				],
				<|
					"ProcessMemory"->Dynamic[If[ValueQ[$Links]&&KeyExistsQ[$Links,uuid],$Links[uuid,"ProcessMemory"],None]],
					"ProcessThreads"->Dynamic[If[ValueQ[$Links]&&KeyExistsQ[$Links,uuid],$Links[uuid,"ProcessThreads"],None]],
					"SessionTime"->Dynamic[If[ValueQ[$Links]&&KeyExistsQ[$Links,uuid],session["SessionTime"],None]]
				|>
			]
		],
		(*lastly,the display form we want to display this with*)
		StandardForm,
		(*we want to completely replace the output to be a single column when the plus button is clicked*)
		"CompleteReplacement"->True
	]
];

(*this enforces a semantically pleasing ordering*)
(*of the keys in the summary box for ExternalObject*)
orderAssoc[assoc_] := Join[KeyTake[{"Name", "System"}]@assoc, KeyDrop[{"Name","System","Session"}]@assoc, KeyTake["Session"]@assoc]

(*this will make a nice monospaced, scrollable Panel type thing*)
(*that makes source code and stack traces display nicely*)
makeScrollableCodePanel[str_] := Framed[
	Pane[
		Style[
			str,
			"Program",
			LineBreakWithin->False,
			StripOnInput->True
		],
		(*we limit the max width to be 500, but only allow a short amount of vertical*)
		(*movement so it doesn't get gigantic*)
		ImageSize->{{1,500},Tiny},
		ContentPadding->False,
		FrameMargins->0,
		StripOnInput->True,
		BaselinePosition->Baseline
	],
	StripOnInput->True,
	Background->Nest[Lighter,Gray,4],
	RoundingRadius->5,
	FrameStyle->None,
	BaselinePosition->Baseline
]


ExternalObject/:MakeBoxes[ExternalObject[assoc_Association?AssociationQ], StandardForm|TraditionalForm]:=
	BoxForm`ArrangeSummaryBox[
		(*first argument is the head to use*)
		ExternalObject,
		(*second argument is the expression*)
		ExternalObject[assoc],
		(*third argument is the icon to use*)
		None,
		(*the next argument is the always visisble properties*)
		MapAt[
			(*this will make sure that we span on the top row*)
			Append[#,SpanFromLeft]&,
			(*this bit of code will make sure that we never include the Session or the Source keys*)
			(*and that we take up to of the first entries in the association*)
			(*and make them into a grid matrix*)
			Partition[
				KeyValueMap[
					Function[{key,val},BoxForm`SummaryItem[{key<>": ",val}]],
					Take[orderAssoc@KeyDrop[{"Session","Source"}]@assoc,UpTo[4]]
				],
				UpTo[2]
			],
			-1
		],
		(*the next argument is the optional items that come down when the plus button is pressed*)
		KeyValueMap[
			Function[{key,val},BoxForm`SummaryItem[{key<>": ",val}]],
			orderAssoc@If[KeyExistsQ["Source"]@#,
				MapAt[
					makeScrollableCodePanel,
					#,
					Key["Source"]
				],
				#
			]&@assoc
		],
		(*lastly,the display form we want to display this with*)
		StandardForm,
		(*we use complete replacement to completely ignore the first set of displayed values*)
		(*with the second one when the button is clicked*)
		"CompleteReplacement"->True
	];



(*$Links tracks the actual socket objects we use to communicate with - note this is in memory only*)
If[!ValueQ[$Links],
	$Links=<||>
];

(*reset cache will re-assign the cache to the default value*)
resetCache[]:=Block[{},
	(* Close objects before resetting the cache. *)
	Quiet[DeleteObject/@ExternalSessions[]];
	$versionCache = <||>;
	safeCacheAssign[
		<|
			"Installations"-><||>,
			(*we now cache the paclet information so that we can tell when the paclet version has been updated*)
			"CachedPacletInfo"-><|PacletManager`PacletInformation["ExternalEvaluate"]|>
		|>
	]
]

(*removing everything else from the contextpath like this ensures that we will get the fully qualified symbol names inside the PersistentObject, which is necessary to load properly*)
safeCacheAssign[assoc_] := Block[
	{$ContextPath = {}},
	PersistentValue["ExternalEvaluate`EvaluatorCache","Local"] = assoc
]

(*the cache is backed by a persistent value*)
$ExternalEvaluatorCache := PersistentValue["ExternalEvaluate`EvaluatorCache","Local"]

(*check cache makes sure that the cache exists and has the required key of Languages, if it doesn't then it's corrupted and needs to be reset*)
StartExternalSession::cache="Evaluator cache contains an invalid installation, it has been reset.";
checkCache[]:=(If[MissingQ[#] || !KeyExistsQ["Installations"]@#,Message[StartExternalSession::cache];resetCache[]]&@$ExternalEvaluatorCache)

(*setup the user preferences for language installations - this is a local thing*)
If[MissingQ[$ExternalEvaluatorCache],
	(*THEN*)
	(*we need to initialize it*)
	(
		resetCache[]
	)
];

(*simple function which does version comparison*)
needsUpdateQ[cache_, require_] :=
	Which[
	cache[[1]] > require[[1]],False,
	cache[[1]] < require[[1]], True,
	cache[[1]] == require[[1]],
		Which[
			cache[[2]] > require[[2]],False,
			cache[[2]] < require[[2]],True,
			cache[[2]] == require[[2]],cache[[3]] < require[[3]]
		]
	]

makeSequence[l_List] := Replace[l, List -> Sequence, {1}, Heads -> True];

(*now that we know the cache has at least been setup if it doesn't already exist, now we can go ahead with*)
(*checking the paclet version for any inconsistencies, etc. from the paclet update process*)
Block[{newcache,cache,cacheVers},
	If[$ExternalEvaluatorCache["CachedPacletInfo"] =!= <|PacletManager`PacletInformation["ExternalEvaluate"]|>,
		(*THEN*)
		(*this version at least has a paclet info cache, but it's different from the current one, so we just update that key*)
		(
			If[
				(*if any of the major/minor/bugfix versions are less than 1.3.0, when the cache layout was finalized, then we need to update it*)
				Or[
					!KeyExistsQ["CachedPacletInfo"]@$ExternalEvaluatorCache,
					needsUpdateQ[
						FromDigits[#,10]&/@StringSplit[$ExternalEvaluatorCache["CachedPacletInfo","Version"],"."],
						{1,3,0}
					]
				],
				(
					(*first update the cached paclet information to this version*)
					cache = $ExternalEvaluatorCache;
					newcache = <|"CachedPacletInfo"-><|PacletManager`PacletInformation["ExternalEvaluate"]|>,"Installations"-><||>|>;
					Function[{langAssoc},
						(*add all the registered / discovered installations first*)
						KeyValueMap[
							Function[{installUUID, installAssoc},
								newcache["Installations", installUUID] = Append[
									KeyTake[{"System", "Executable", "Version"}]@installAssoc,
									"Registered" -> If[MemberQ[installUUID]@Lookup[langAssoc, "DiscoveredInstallations", <||>],
										Automatic,
										True
									]
								]
							],
							langAssoc["Installations"]
						];
						(*now add the blacklisted or unregistered installations*)
						KeyValueMap[
							Function[{installUUID, installAssoc},
								newcache["Installations", installUUID] = Append[
									installAssoc,
									"Registered" -> False]
							],
							Lookup[langAssoc, "BlacklistedInstalls", <||>]]
					] /@ cache["Languages"];

					(*save back the cache*)
					safeCacheAssign[newcache]
				)
			];
		)
		(*ELSE*)
		(*the paclet info's the same we don't need to do anything*)
	]
];
(*we don't start out with any hard coded heuristics, we just add to them using RegisterSystem*)
If[!ValueQ[$LanguageHeuristics],
	$LanguageHeuristics=<||>
];

(*these are things that we will never be able to assume - i.e. we can't assume what files to run, nor can we assume where to look for the executable or what it will be called*)
$RequiredSystemOptions = {
	"ExecutablePatternFunction"
};

ExternalEvaluate`ExecuteScriptProcess = Function[{uuid,exec,file,opts},{exec, file}];
ExternalEvaluate`ExecuteVersionProcess = Function[{exec},{exec, "--version"}];
ExternalEvaluate`VersionSameQ = Function[{execVersion,userVersion},
		With[
			{
				userSplit = StringTrim/@StringSplit[userVersion,"."],
				execSplit = StringTrim/@StringSplit[execVersion,"."]
			},
			Take[execSplit,UpTo[Length@userSplit]] === userSplit
		]
	]
ExternalEvaluate`VersionValidationFunction = Function[{versionString,userQ},StringTrim@StringDelete[versionString, Except["." | DigitCharacter]]]
ExternalEvaluate`DefaultSessionIcon = Graphics[List[List[Thickness[0.038461538461538464`],List[List[FaceForm[List[GrayLevel[0.93`],Opacity[1.`]]],FilledCurve[List[List[List[1,4,3],List[0,1,0],List[1,3,3],List[0,1,0],List[1,3,3],List[0,1,0],List[1,3,3],List[0,1,0]]],List[List[List[25.499999999999996`,2.5`],List[25.499999999999996`,1.3953100000000003`],List[24.604699999999998`,0.49999999999999994`],List[23.5`,0.49999999999999994`],List[2.5`,0.49999999999999994`],List[1.3953100000000003`,0.49999999999999994`],List[0.49999999999999994`,1.3953100000000003`],List[0.49999999999999994`,2.5`],List[0.49999999999999994`,23.5`],List[0.49999999999999994`,24.604699999999998`],List[1.3953100000000003`,25.499999999999996`],List[2.5`,25.499999999999996`],List[23.5`,25.499999999999996`],List[24.604699999999998`,25.499999999999996`],List[25.499999999999996`,24.604699999999998`],List[25.499999999999996`,23.5`],List[25.499999999999996`,2.5`]]]]]],List[List[FaceForm[List[RGBColor[0.5`,0.5`,0.5`],Opacity[1.`]]],FilledCurve[List[List[List[0,2,0],List[0,1,0],List[0,1,0],List[0,1,0],List[0,1,0],List[0,1,0],List[0,1,0]]],List[List[List[20.492579663674487`,16.483437002807932`],List[16.333179663674485`,19.24283700280793`],List[16.333179663674485`,17.38813700280793`],List[6.880449663674485`,17.38813700280793`],List[6.880449663674485`,15.578737002807937`],List[16.333179663674485`,15.578737002807937`],List[16.333179663674485`,13.724037002807929`],List[20.492579663674487`,16.483437002807932`]]]],FilledCurve[List[List[List[0,2,0],List[0,1,0],List[0,1,0],List[0,1,0],List[0,1,0],List[0,1,0],List[0,1,0]]],List[List[List[5.225191405293065`,9.309262948550709`],List[9.384561405293063`,6.549862948550711`],List[9.384561405293063`,8.404162948550706`],List[18.83731140529306`,8.404162948550706`],List[18.83731140529306`,10.213962948550709`],List[9.384561405293063`,10.213962948550709`],List[9.384561405293063`,12.068162948550707`],List[5.225191405293065`,9.309262948550709`]]]]]]]],Rule[AspectRatio,1],Rule[Axes,False],Rule[Background,GrayLevel[0.93`]],Rule[Frame,True],Rule[FrameStyle,Directive[Thickness[Tiny],GrayLevel[0.7`]]],Rule[FrameTicks,None],Rule[ImagePadding,List[List[0.1`,1.1`],List[1.1`,0.1`]]],Rule[ImageSize,List[70.3359375`,Automatic]],Rule[PlotRange,List[List[-0.020833333333333315`,26.02083333333333`],List[0.`,25.999999999999996`]]],Rule[PlotRangePadding,Automatic]]

(*these options are optional - i.e. there's a reasonable default behavior for them*)
$OptionalSystemOptions = <|
	"Icon" -> ExternalEvaluate`DefaultSessionIcon,

	"ExecutablePathFunction" -> ({}&),

	"DependencyTestFile" -> None,

	"RunFileInSessionFunction" -> (None&),

	"ProgramFileFunction" -> (None&),

	"ExecutableSessionOptionsMatchQFunction"->(True&),

	(*the default version conform function simply trims the string and deletes all non number/period characters*)
	"VersionStringConformFunction"->ExternalEvaluate`VersionValidationFunction,

	(*to compare the versions, we compare the default conformed strings as a list by making them the same length*)
	"VersionSameQFunction"->ExternalEvaluate`VersionSameQ,

	(*the default version exec command just runs the executable with --version as the argument*)
	"VersionExecCommandFunction"->ExternalEvaluate`ExecuteVersionProcess,

	(*the default script exec function is just the executable and the file to execute*)
	"ScriptExecCommandFunction"->ExternalEvaluate`ExecuteScriptProcess,

	(*the default epilog and prolog are None*)
	"DefaultEpilog"->None,
	"DefaultProlog"->None,

	(*these two options are for starting the external evaluator if different options need to be provided for some reason*)
	"ProcessEnvironmentFunction"->(Inherited&),
	"ProcessDirectoryFunction"->(Inherited&),

	(*session prolog / epilogs are for the entire session, i.e. always evaluated at the start of the session*)
	(*and always at the end of the session, but each only once per session*)
	(*mainly used for setting up custom types and for providing default imports available immediately, etc.*)
	"SessionProlog"->None,
	"SessionEpilog"->None,

	(*what paclet to check for updates - by default use ExternalEvalute_system*)
	(*note that this is a placeholder, the actual value is inserted when RegisterSystem is called*)
	(*if it's not provided by the system register call*)
	"PacletName" -> None,

	(*nonzmq evaluation function is used for systems such as webunit or excel that don't interact through zmq*)
	(*if this is something other than None, then a socket is not created for a session, and evaluation happens through*)
	(*this function*)
	"NonZMQEvaluationFunction"->(None&),

	(*nonzmq initialize function is a function that is called to setup the system in addition to the process being started for a session*)
	"NonZMQInitializeFunction"->(None&),

	(*same as the init function, but for deinit when the session is closed*)
	"NonZMQDeinitializeFunction"->(None&),

	"SerializationFunction" -> Function[
		Developer`WriteRawJSONString[
			#,
			"ConversionRules" -> {_Missing | None -> Null},
			"Compact" -> True]
	],
	"DeserializationFunction" -> Composition[Developer`ReadRawJSONString, ByteArrayToString]
|>;


ExternalEvaluate`RegisterSystem::usage="ExternalEvaluate`RegisterSystem[lang,opts] adds heuristics/options for the specified system so that the system can be discovered and used with the ExternalEvaluate system."

(*we support the MergingFunction option for RegisterSystem when using parent / child options*)
Options[ExternalEvaluate`RegisterSystem] = {MergingFunction -> Last};

(*this will add options for the specified language so that external evaluate can find/use it*)
ExternalEvaluate`RegisterSystem[lang_?StringQ,sysopts_?AssociationQ,opts:OptionsPattern[]]:=Block[
	{
		defaults,
		full
	},
	(
		(*confirm that the required options are present and no more*)
		If[Sort[Intersection[Keys[sysopts],$RequiredSystemOptions]] === Sort[$RequiredSystemOptions] &&
			Complement[Keys[sysopts],Join[Keys[$OptionalSystemOptions],$RequiredSystemOptions]] == {},
			(*THEN*)
			(*it's valid and can be used*)
			(
				(*get the elements not specified that need to use the default versions*)
				defaults = KeyTake[Complement[Keys[$OptionalSystemOptions],Keys[sysopts]]] @ $OptionalSystemOptions;

				full = Join[defaults,sysopts];

				(*check the dependenytestfile for File[] vs String*)
				full["DependencyTestFile"] = Switch[#,
					File[a_?StringQ]/;a=!="",
					(*legitimate File[] wrapper, so just strip the File[] wrapper*)
					First[#],
					(*normal string*)
					a_?StringQ /; a=!="",#,
					(*anything else return $Failed*)
					_,Return[$Failed,Block]

				]& /@ (full["DependencyTestFile"] /. {a_String :> {a}, b_File :> {b}, c_List :> c, None -> None, _ :> {$Failed}});

				(*now store the association in the $LanguageHeuristics*)
				$LanguageHeuristics[lang] = full;

				(*check the "PacletName" to see if it was specified as a string, else if it's the default*)
				(*value, then we need to replace it with ExternalEvaluate_system*)
				If[$LanguageHeuristics[lang,"PacletName"] =!= None,
					(*THEN*)
					(*it's not None, so we need to check if it needs to be populated*)
					(
						(*check if it's meant to be automatically populated*)
						If[MemberQ[$LanguageHeuristics[lang,"PacletName"]]@{Default,Automatic,Inherited},
							(*THEN*)
							(*need to generate it as the default form*)
							$LanguageHeuristics[lang,"PacletName"] = "ExternalEvaluate_"<>lang,
							(*ELSE*)
							(*it's been populated, make sure the spec is a string and the paclet exists*)
							If[!StringQ[$LanguageHeuristics[lang,"PacletName"]]||
								PacletManager`PacletFind[$LanguageHeuristics[lang,"PacletName"]] === {},
								(*THEN*)
								(*doesn't exist or invalid spec*)
								(
									(*TODO : issue message about this - we can't use it so just ignore *)
									(*the paclet updating issue*)
									$LanguageHeuristics[lang,"PacletName"] = None
								)
								(*ELSE*)
								(*good to go*)
							]
						]
					)
					(*ELSE*)
					(*it's None, so we shouldn't check paclet updates for this driver*)
				]
			),
			(*ELSE*)
			(*invalid, either invalid options or not enough options present*)
			(
				$Failed
			)
		]
	)
]

(*update paclet plugin will perform an update for the specified system, using the specified paclet name*)
(*however, we don't accept Missing[], as when this is called at the top of functions like FindExternalEvaluators*)
(*etc. we could get called before knowing that the specified language doesn't exist*)
(*as such the second argument, which is either a direct string and should be used, None, which means*)
(*the system exists, but doesn't have an associated paclet, and Missing when the language doesn't exist*)
updatePacletPlugin[lang_?StringQ,langPacName_:Automatic]:=Block[
	{pacname = If[langPacName===Automatic,$LanguageHeuristics[lang,"PacletName"],langPacName]},
	(
		(*do the update for the paclet*)
		doPacletUpdate[pacname];
		(*mark this paclet as updated in this session - note we only do this when we know that*)
		(*the language's paclet name isn't Missing, as if it's missing, then this system doesn't exist*)
		If[!MissingQ[pacname],AppendTo[$UpdatedLanguages,lang]];
	)
]

(*function that actually performs the paclet update for a provided paclet*)
doPacletUpdate[langPacName_?(!MissingQ[#]&)]:=
	(*some systems might not have a paclet, i.e. ones that are just used dynamically in a kernel session*)
	(*when prototyping, these shouldn't cause a paclet check, etc.*)
	If[langPacName =!= None,
		(*THEN*)
		(*we proceed with checking for updates to the paclet*)
		(
			(*update the paclet*)
			PacletManager`Package`getPacletWithProgress[langPacName];
			(*now load the paclet's system*)
			Get[PacletManager`PacletResource[langPacName, "System"]];
		)
		(*ELSE*)
		(*this driver doesn't have an associated paclet, so don't do anything*)
	];

(*paclet system check looks for an external evaluate plugin paclet for a particular specified system*)
(*because this is an unknown language, we assume that the name of the plugin paclet is ExternalEvaluate_sys*)
pacletSystemCheck[lang_?StringQ]:=Block[{pacname = "ExternalEvaluate_"<>lang},
	(
		(*make sure at least one version of the paclet exists*)
		If[Length[PacletCheckUpdate[pacname, "UpdateSites" -> True]] > 0,
			(*THEN*)
			(*we were able to find a paclet for this language and should update it*)
			updatePacletPlugin[lang,pacname],
			(*ELSE*)
			(*didn't find one - return $Failed to indicate we didn't find one*)
			$Failed
		]
	)
]


(*form that implements inheritance*)
ExternalEvaluate`RegisterSystem[lang_?StringQ,parent_?StringQ,sysopts_?AssociationQ,opts:OptionsPattern[]] := Block[
	{
		parentOpts,
		fullChildOpts
	},
	(
		(*first check that the parent exists*)
		If[knownLanguageQ[parent],
			(*THEN*)
			(*parent class exists, now we can go handle it*)
			(
				If[Complement[Keys[sysopts],Join[Keys[$OptionalSystemOptions],$RequiredSystemOptions]] =!= {},
					(*THEN*)
					(*there are invalid keys specified for the child*)
					$Failed,
					(*ELSE*)
					(*good to go*)
					(
						(*to handle inheritance, we take all the keys of the child*)
						(*and the keys of the parent, merging them together, but preferring the child's*)
						parentOpts = $LanguageHeuristics[parent];
						fullChildOpts = Merge[{parentOpts,sysopts},OptionValue[MergingFunction]];
						ExternalEvaluate`RegisterSystem[lang,fullChildOpts]
					)
				]
			),
			(*ELSE*)
			(*parent class doesn't exist*)
			$Failed
		]
	)
]


Options[FindExternalEvaluators] = {"ResetCache"->False}

(*external evaluators returns a dataset for the given language of all the installations it knows about for that language*)
FindExternalEvaluators[lang_?StringQ, opts:OptionsPattern[]]:= Block[{reset},
	reset = OptionValue["ResetCache"];
	If[TrueQ[reset], resetCache[]];
	checkCache[];
	If[!MemberQ[$UpdatedLanguages,lang],
		updatePacletPlugin[lang];
	];
	If[!knownLanguageQ[lang],
		(*THEN*)
		(*we don't know anything about this language, so see if it's on the server, and install if that's the case*)
		(
			If[FailureQ[pacletSystemCheck[lang]],
				(*THEN*)
				(*we failed to find the paclet on the servers to install, so we just fail*)
				(
					Message[FindExternalEvaluators::unknownSys,lang];
					$Failed
				),
				(*ELSE*)
				(*we got the paclet, just recursively call this function again*)
				(
					FindExternalEvaluators[lang]
				)
			]
		),
		(*ELSE*)
		(*it's a known language, so update all evaluators then display the dataset*)
		(
			(*first try to find all external evaluators on the system for this language*)
			findAllInstalls[lang];
			(*now display the dataset of the preferences*)
			If[# === <||>,
				(*THEN*)
				(*just return an empty dataset, no evaluators found*)
				Dataset[#],
				(*ELSE*)
				(*build the dataset from the installations, then group them by the system, then take only the ones*)
				(*that are valid known languages*)
				If[KeyExistsQ[lang]@#,#[[lang]],#]&@
					KeyTake[Dataset[#][GroupBy[#System &]],lang]
			]& @ $ExternalEvaluatorCache["Installations"]
		)
	]
]

(*form for all languages*)
FindExternalEvaluators[opts:OptionsPattern[]]:=(
	If[TrueQ[OptionValue["ResetCache"]], resetCache[]];
	checkCache[];
	(*first try to find all external evaluators on the system*)
	findAllInstalls[];
	(*now display the dataset of the preferences*)
	If[# === <||>,
		(*THEN*)
		(*just return an empty dataset, no evaluators found*)
		Dataset[#],
		(*ELSE*)
		(*build the dataset from the installations, then group them by the system, then take only the ones*)
		(*that are valid known languages*)
		KeyTake[Dataset[#][GroupBy[#System &]],allKnownLanguages[]]
	]& @ $ExternalEvaluatorCache["Installations"]
)

ExternalSessions[] := (checkCache[]; ExternalSessionObject/@Keys@Select[#Active&]@$Links)

ExternalSessions[lang_?StringQ] := (
	checkCache[];
	If[!knownLanguageQ[lang],
		(*THEN*)
		(*we don't know anything about this language, so issue message and exit*)
		(
			Message[ExternalSessions::unknownSys,lang];
			$Failed
		),
		(*ELSE*)
		(*the language exists, so select all of those languages from $Links*)
		(
			ExternalSessionObject/@Keys@Select[#System === lang && TrueQ[#Active]&]@$Links
		)
	]
)


(*************************************)
(********                   **********)
(******** EXTERNAL SESSIONS **********)
(********                   **********)
(*************************************)


(*NOTE - ExternalSession System variant's aren't currently used, so that isn't really implemented yet*)
$SessionUserKeys = {
	"System", "Variant", "Version", "Executable", 
	"Prolog", "SessionProlog",
	"Epilog", "SessionEpilog",
	"ReturnType", "SessionOptions"
};

(*main form that takes all of the options for an external evaluator and returns an ExternalSessionObject*)
StartExternalSession[opts_?AssociationQ]:=Block[
	{
		langOpts = opts,
		cachedInstall,
		exec,
		prolog,
		epilog,
		linkUUID,
		zmqSocketAddr,
		assoc,
		version,
		langInstall
	},
	(
		checkCache[];
		If[!MemberQ[$UpdatedLanguages,langOpts["System"]],
			updatePacletPlugin[langOpts["System"]];
		];

		(*if there's not a "ReturnType" spec in the association, then add the default of "Expression"*)
		If[!KeyExistsQ["ReturnType"]@langOpts,
			langOpts["ReturnType"] = "Expression"
		];

		(*if there were no SessionOptions specified, then add an empty list*)
		If[!KeyExistsQ["SessionOptions"]@langOpts,
			langOpts["SessionOptions"] = {}
		];

		(*validate that there aren't any extra options in the association, and that we have the required System option*)
		Which[
			!KeyExistsQ["System"]@langOpts,
			(*don't know what system to use, so just issue message and fail*)
			(
				Message[StartExternalSession::nosys];
				$Failed
			),
			Complement[Keys[langOpts],$SessionUserKeys] =!= {},
			(*extra unknown options were specified*)
			(
				Message[StartExternalSession::unknownOpts,Complement[Keys[langOpts],$SessionUserKeys]];
				$Failed
			),
			!knownLanguageQ[langOpts["System"]],
			(*then the language specified isn't known to our system - it's not manually registered, and it's not in the heuristics*)
			(
				If[FailureQ[pacletSystemCheck[langOpts["System"]]],
					(*THEN*)
					(*we failed to find the paclet on the servers to install, so we just fail*)
					(
						Message[StartExternalSession::unknownSys,langOpts["System"]];
						$Failed
					),
					(*ELSE*)
					(*we got the paclet, just recursively call this function again*)
					(
						StartExternalSession[opts]
					)
				]
			),
			KeyExistsQ["ReturnType"]@langOpts && !MemberQ[langOpts["ReturnType"]]@{Automatic,Default,"Expression","String"},
			(*then the ReturnType is invalid*)
			(
				Message[StartExternalSession::invalidReturnType,langOpts["ReturnType"]];
				$Failed
			),
			KeyExistsQ["Executable"]@langOpts && KeyExistsQ["Executable"]=!=Automatic,
			(*user specified a language executable to use - so try to verify that location, caching it in the UserRegistered locations if it's valid*)
			(
				(*fix languageExecutable if it's specified as File[...]*)
				If[Head[langOpts["Executable"]] === File,
					(*THEN*)
					(*extract the actual path out from the inter File wrapper*)
					(
						langOpts["Executable"] = First[langOpts["Executable"]]
					)
				];
				Which[
					!MissingQ[(cachedInstall = SelectFirstKey[#Executable === langOpts["Executable"] &]@$ExternalEvaluatorCache["Installations"])],
					(*then the executable is an already known evaluator from the cache, now we confirm that any other details that were specified for this match what is known about this installation*)
					(
						(*get the options from the cache*)
						cachedInstall = $ExternalEvaluatorCache[["Installations",cachedInstall]];

						(*first check if this system has been unregistered*)
						If[cachedInstall["Registered"] === False,
							(*THEN*)
							(*this installation has been unregistered*)
							(
								Message[StartExternalSession::unregistered,langOpts["System"],langOpts["Executable"]];
								Return[$Failed]
							)
						];

						(*check if the user specified a version - if they did make sure it's the same as the cachedInstall*)
						Which[
							KeyExistsQ["Version"]@langOpts && langOpts["Version"] =!= Automatic && StringQ[langOpts["Version"]],
							(*the language version was specified and is a string, so make sure that the version specified by the option is the same as the version in the cache*)
							(
								If[$LanguageHeuristics[langOpts["System"],"VersionSameQFunction"][
										cachedInstall["Version"],
										$LanguageHeuristics[langOpts["System"],"VersionStringConformFunction"][langOpts["Version"],True]
									],
									(*THEN*)
									(*the versions are the same, so we're good to go*)
									(
										(*handle getting the epilog and prolog from the options and then create the link out of it*)
										epilogPrologLinkHandle[langOpts,cachedInstall,StartExternalSession]
									),
									(*ELSE*)
									(*differing versions - issue message and fail*)
									(
										Message[StartExternalSession::invalidOpt,"Version",langOpts["Version"]];
										Return[$Failed];
									)
								]
							),
							!KeyExistsQ["Version"]@langOpts || langOpts["Version"] === Automatic,
							(*didn't specify a version, so we don't need to check to make sure the version number is correct*)
							(
								(*handle getting the epilog and prolog from the options and then create the link out of it*)
								epilogPrologLinkHandle[langOpts,cachedInstall,StartExternalSession]
							),
							True,
							(*invalid option spec for version*)
							(
								Message[StartExternalSession::invalidOpt,"Version",langOpts["Version"]];
								Return[$Failed];
							)
						]
					),
					TrueQ@And[
						(*also remember to check the File[...] case*)
						StringQ[langOpts["Executable"]] || (Head[langOpts["Executable"]]===File && StringQ[First[langOpts["Executable"]]]),
						FileType[langOpts["Executable"]]===File,
						FileExistsQ[langOpts["Executable"]]
					],
					(*executable exists and is a file, isn't known to the system, so we need to verify the install and then add it to the cache if it's valid*)
					(
						(*first we need to make the required association of metadata*)
						assoc = makeInstallEntryAssoc[langOpts["System"],langOpts["Executable"]];

						(*now determine if there was a version number specified*)
						If[KeyExistsQ["Version"]@langOpts && langOpts["Version"] =!= Automatic && StringQ[langOpts["Version"]],
							(*THEN*)
							(*a non default version was specified, so need to confirm this install is that version*)
							(
								version = langOpts["Version"];
							),
							(*ELSE*)
							(*the version wasn't specified, so just check that it's a usable install with any version*)
							(
								version = "*";
							)
						];
            (* Register the executable *)
						langInstall = addInstallEntry[langOpts["System"], assoc, "UserRegistered", "Version"->version,
							 "Constraints"->langOpts["SessionOptions"], "IssueMessage"->True];
						If[StringQ[langInstall],
							(*THEN*)
							(*it's valid - add the entry and then proceed with checking the epilog and prolog*)
							(
								(*handle getting the epilog and prolog from the options and then create the link out of it*)
								epilogPrologLinkHandle[langOpts,assoc,StartExternalSession]
							),
							(*ELSE*)
							(*the install is invalid - fail*)
							(
								$Failed
							)
						]
					),
					True,
					(*user specified a path or value for Executable, but it's invalid*)
					(
						Message[StartExternalSession::invalidOpt,"Executable",langOpts["Executable"]];
						$Failed
					)
				]
			),
			!KeyExistsQ["Executable"]@langOpts || langOpts["Executable"] === Automatic,
			(*then the executable wasn't specified, so we have to try and find it*)
			(* check for specified version string *)
			If[KeyExistsQ["Version"]@langOpts && langOpts["Version"] =!= Automatic,
				If[StringQ[langOpts["Version"]],
					(* version specified *)
					version = langOpts["Version"];
					,
					(* version invalid opt *)
					Message[StartExternalSession::invalidOpt, "Version", langOpts["Version"]];
					Return[$Failed]
				]
				,
				(* Automatic, accept any version *)
				version = "*"
			];

			(*now attempt to resolve the language version*)
			langInstall = resolveLangInstall[langOpts["System"],version,langOpts["SessionOptions"]];
			If[
				FailureQ[langInstall],
				(*THEN*)
				(*didn't find a language - user will need to install/configure it*)
				Message[StartExternalSession::noinstall,langOpts["System"]];
				$Failed,
				(*ELSE*)
				(*found it correctly - now we can create the link*)
				(*now fish out the installation association from the cache - note that we need to use part at the end here cause it is returned as a*)
				(*Key[...] wrapper, which in association accessing is interpreted verbatim as the key Key[stuff]->blah, when we need stuff->blah *)
				assoc = $ExternalEvaluatorCache["Installations"][[langInstall]];
				(*handle getting the epilog and prolog from the options and then create the link out of it*)
				epilogPrologLinkHandle[langOpts,assoc,StartExternalSession]
			]
		]
	)
];

checkProcStats[item_String] := item;
checkProcStats[item_Quantity] := item;
checkProcStats[item_?NumericQ] := item;
checkProcStats[item___] := Missing["NotAvailable"];

(*this handles actually starting the process, setting up the zmq socket and such and registering the information with $Links*)
makeLink[entry_,lang_]:=
	Block[
		{
		linkUUID,
		zmqSocketAddr,
		session,
		res,
		process
		},
		(*now we have everything we need to create this entry in $Links*)
		linkUUID = CreateUUID[];

		(*to start the session, we use the script execution*)
		process = $LanguageHeuristics[lang,"ScriptExecCommandFunction"][
			linkUUID,
			entry["Executable"],
			(*the program file could depend on the version, so call the function*)
			$LanguageHeuristics[lang,"ProgramFileFunction"][entry["Version"]],
			entry["SessionOptions"]
		];
		If[ListQ[process] || StringQ[process],
			process = StartProcess[process,
				ProcessDirectory->$LanguageHeuristics[lang,"ProcessDirectoryFunction"][],
				ProcessEnvironment->$LanguageHeuristics[lang,"ProcessEnvironmentFunction"][]
			]
		];

		$Links[linkUUID] = <|
			"Process"->process,
			"Epilog"->entry["Epilog"],
			"Prolog"->entry["Prolog"],
			"SessionEpilog"->entry["SessionEpilog"],
			"SessionProlog"->entry["SessionProlog"],
			"Prolog"->entry["Prolog"],
			"System"->lang,
			(*we need to force this to be a closure around the uuid argument, otherwise it goes out of scope when makeLink exits*)
			With[{uuid=linkUUID},"ProcessMemory":>(Refresh[checkProcStats[$Links[uuid,"Process"]["Memory"]],UpdateInterval->5])],
			With[{uuid=linkUUID},"ProcessThreads":>(Refresh[checkProcStats[$Links[uuid,"Process"]["Threads"]],UpdateInterval->5])],
			"Version"->entry["Version"],
			"Executable"->entry["Executable"],
			"ReturnType"->entry["ReturnType"],
			"SessionTime"->AbsoluteTime[],
			"EvaluationCount"->0,
			"SessionOptions"->entry["SessionOptions"],
			"Active"->True
		|>;

		(*check the process to make sure it didn't fail to start for some reason*)
		If[
			ProcessStatus[$Links[linkUUID,"Process"]] =!= "Running",
			(*THEN*)
			(*failed, some error so raise a message with the stderr stream and return $Failed*)
			Message[
				StartExternalSession::replFail,
				entry["System"],
				entry["Executable"],
				ReadString[ProcessConnection[$Links[linkUUID,"Process"],"StandardError"]]
			];
			Return[$Failed];
			(*ELSE*)
			(*good to go, read from the process to get the relevant *)
		];

		res = $LanguageHeuristics[lang,"NonZMQInitializeFunction"][
			linkUUID,
			$Links[linkUUID,"Process"],
			entry["SessionOptions"]
		];

		(*check how we should initialize this function*)
		If[
			res =!= None,
			(*THEN*)
			(*we initialized the system with a WL function - don't connect the socket*)
			If[
				FailureQ[res],
				(*THEN*)
				(*failed to start*)
				Message[
					StartExternalSession::replFail,
					entry["System"],
					entry["Executable"],
					StringForm["the initialization function returning `1`",res]
				];
				$Failed,
				(*ELSE*)
				(*worked - update the *)
				$Links[linkUUID,"Socket"] = None;
				(*worked, check on the session prolog and return the object*)
				session = ExternalSessionObject[linkUUID];
				res = evalSessionLog[session,"Prolog"];
				If[FailureQ[res],
					(*THEN*)
					(*failed to evaluate one of the prologs, so tear down the session and return the failure*)
					(
						DeleteObject[session];
						res
					),
					(*ELSE*)
					(*worked return the session*)
					session
				]
			],
			(*ELSE*)
			(*we can just run the normal start routine - reading from the zmq socket, etc.*)
			(*now we read a line from the process to get the zmq socket to connect to*)
			zmqSocketAddr = ReadLine[$Links[linkUUID,"Process"]];

			(*check the address returned from ReadLine to ensure that it's a string and we didn't hit EndOfFile*)
			If[FailureQ[zmqSocketAddr] || zmqSocketAddr === EndOfFile,
				(*THEN*)
				(*we failed*)
				Message[
					StartExternalSession::replFail,
					entry["System"],
					entry["Executable"],
					"no input returned from process."
				];
				$Failed,
				(*ELSE*)
				(*worked, now just start up the socket and return*)
				(*check the host*)
				If[
					!StringQ[zmqSocketAddr],
					(*THEN*)
					(*coulnd't import the data it returned some nonsense*)
					Message[
						StartExternalSession::replFail,
						entry["System"],
						entry["Executable"],
						"invalid input returned from process."
					];
					$Failed,
					(*THEN*)
					(*worked, we have a valid host to attempt to connect to*)
					$Links[linkUUID,"Socket"] = SocketConnect[zmqSocketAddr,"ZMQ_Pair"];
					(*check the socket*)
					If[
						MatchQ[$Links[linkUUID,"Socket"], _SocketObject],
						(*THEN*)
						(*worked, check on the session prolog and return the object*)
						session = ExternalSessionObject[linkUUID];
						res = evalSessionLog[session,"Prolog"];
						If[FailureQ[res],
							(*THEN*)
							(*failed to evaluate one of the prologs, so tear down the session and return the failure*)
							DeleteObject[session];
							res,
							(*ELSE*)
							(*worked return the session*)
							session
						],
						(*ELSE*)
						(*something failed with connecting the socket*)
						Message[
							StartExternalSession::sockFail,
							entry["System"],
							entry["Executable"],
							zmqSocketAddr
						];
						$Failed
					]
				]
			]
		]
	];

(*this evaluates both the system setting for session prolog/epilog as well as the user setting for the session if specified*)
evalSessionLog[session_,logType:("Prolog"|"Epilog")]:=Block[
	{
		sysSessionlog = $LanguageHeuristics[session["System"],"Session"<>logType],
		sessionlog = session["Session"<>logType],
		res
	},

	(*evaluate the system session prolog*)
	If[sysSessionlog =!= None,
		res = externalEvaluateLinkSession[session,sysSessionlog];
		(* if failed, propogate the error *)
		If[FailureQ[res], Return[res]];
	];

	(*evaluate the session prolog*)
	If[sessionlog =!= None,
		res = externalEvaluateLinkSession[session,sessionlog];
		(* if failed, propogate the error*)
		If[FailureQ[res], res]
	]
]

(*this will check for what Epilog / Prolog options to use, defaulting to the ones in assoc if langOpts is Automatic|Inherited|Default, or if the keys are missing*)
(*entryAssoc is the association of information from the cache, while langOpts is what the user specified to StartExternalSession (or ExternalEvaluate, etc.)*)
updateEpilogProlog[langOpts_,entryAssoc_,symbol_]:=Block[
	(*we want to use as many options from the entryAssoc arg as that can is the pre-existing association of install details from the cache*)
	{assocCopy = entryAssoc},
	(
		Function[{opt},
			Which[
				MissingQ[langOpts[opt]]||MemberQ[langOpts[opt]]@{Inherited,Default,Automatic},
				(*we just use the default one in the cache / system registry*)

				(
					If[KeyExistsQ[opt]@entryAssoc,
						assocCopy[opt] = entryAssoc[opt],
						If[opt =!= "SessionProlog" && opt =!= "SessionEpilog",
							(*note we don't do this for SessionProlog / SessionEpilog*)
							(*for those settings we want both, i.e. to have the session prolog setup stuff*)
							(*but then to also have the option of a user setting up their own stuff too with those options*)
							(*in StartExternalSession*)
							assocCopy[opt] = $LanguageHeuristics[langOpts["System"],opt],
							assocCopy[opt] = None
						]
					]
				),
				StringQ[langOpts[opt]] || MemberQ[Head[langOpts[opt]]]@{File,CloudObject,LocalObject,URL} || MatchQ[langOpts[opt],$listEvalpat],
				(*a valid *log option was specified - use it*)
				(
					assocCopy[opt] = langOpts[opt]
				),
				True,
				(
					With[{s=symbol},Message[MessageName[s, "invalidOpt"],opt,langOpts[opt]]];
					Return[$Failed];
				)
			]
		] /@ {"Prolog","Epilog","SessionProlog","SessionEpilog"};

		(*also copy over the ReturnType and SessionOptions*)
		assocCopy["ReturnType"] = langOpts["ReturnType"];
		assocCopy["SessionOptions"] = Lookup[langOpts,"SessionOptions",{}];

		(*if we get here, we have valid options, so return the found ones*)
		assocCopy
	)
]

(*this helper function handles the two above helper functions into a single combined one*)
epilogPrologLinkHandle[langOpts_,assoc_,symbol_]:=Block[{res},(
		(*try and get the epilog and prolog using the options*)
		res = updateEpilogProlog[langOpts,assoc,symbol];
		If[FailureQ[res],
			(*THEN*)
			(*error, return $Failed, a message will already have been raised*)
			Return[$Failed]
		];

		(*now we're ready to actually setup the link*)
		makeLink[res,langOpts["System"]]
	)
]

(*for the default form of lang, we just create the corresponding association with just the language specified - defaulting to returning exprs*)
StartExternalSession[lang_?StringQ]:=StartExternalSession[<|"System"->lang|>]

StartExternalSession[lang_?StringQ->type_?StringQ]:=StartExternalSession[<|"System"->lang,"ReturnType"->type|>]

(*when the options are specified as the second element of a list, then combine the opts with the System option*)
StartExternalSession[{lang_?StringQ,opts_?AssociationQ}]:=StartExternalSession[Append[opts,"System"->lang]]

(*form where opts is a verbatim sequence of option patterns*)
StartExternalSession[{lang_?StringQ,opts:OptionsPattern[]}]:=StartExternalSession[Association[opts,"System"->lang]]

(*anything else fail*)
StartExternalSession[any___]:=(Message[StartExternalSession::invalidSpec,any]; $Failed)


(*DeleteObject stops the process, closes the socket, and then deletes the session from $Links*)
ExternalSessionObject /: HoldPattern[DeleteObject][session:ExternalSessionObject[sessionUUID_]] := Block[{res},
	(
		If[KeyExistsQ[sessionUUID]@$Links && TrueQ[$Links[sessionUUID,"Active"]],
			(*THEN*)
			(*the session exists, so we can close the socket, kill the process and remove it from the list of links*)
			(
				(*first check if there's a SessionEpilog we need to evaluate*)
				res = evalSessionLog[session,"Epilog"];
				If[# =!= None, Close[#]] & @ $Links[sessionUUID,"Socket"];
				res = $LanguageHeuristics[session["System"],"NonZMQDeinitializeFunction"][sessionUUID,$Links[sessionUUID,"Process"]];
				KillProcess[$Links[sessionUUID,"Process"]];
				(*mark this session as inactive*)
				$Links[sessionUUID,"StopTime"] = AbsoluteTime[];
				$Links[sessionUUID,"Active"] = False;
				(*finally if the epilog failed to evaluate and returned a Failure then return that to the user as well*)
				If[FailureQ[res],res]
			),
			(*ELSE*)
			(*doesn't exist, so issue message*)
			(
				Message[ExternalSessionObject::invalidSession,session];
				$Failed
			)
		]
	)
]

(*callable form for getting options directly by calling the object session[...]*)
ExternalSessionObject /: HoldPattern[session:ExternalSessionObject[_?StringQ]][rest___] := getSessionOpts[session,rest]

(*same thing, but when using Options[session,...]*)
ExternalSessionObject /: HoldPattern[Options][session:ExternalSessionObject[_?StringQ],rest___] := getSessionOpts[session,rest]

(*no specific options will map over all known options*)
getSessionOpts[session_]:=	getSessionOpts[session,getSessionOpts[session,"Properties"]]

(*a list of options just gets mapped over*)
getSessionOpts[session_,options:{_?StringQ...}]:=AssociationThread[options,getSessionOpts[session,#]&/@options]

(*uuid is handled specially, as it's not a key in $Links[sessionUUID], it is the UUID used for that*)
getSessionOpts[session_,"UUID"]:=First[session]

getSessionOpts[session_,"Properties"]:=
	{
		"UUID",
		"System",
		"Executable",
		"Version",
		"Socket",
		"Process",
		"Prolog",
		"Epilog",
		"SessionProlog",
		"SessionEpilog",
		"ReturnType",
		"SessionTime",
		"EvaluationCount",
		"Active",
		"ProcessMemory",
		"ProcessThreads"
	}

(*normal option getting function - looks up the specified option in $Links and returns it*)
getSessionOpts[session_,option_?StringQ]:=Block[{sessionUUID = First[session]},
	(
		If[KeyExistsQ[sessionUUID]@$Links,
			(*THEN*)
			(*the session exists, so we can query the properties from $Links*)
			(
				If[KeyExistsQ[option]@$Links[sessionUUID],
					(*THEN*)
					(*option exists, return that *)
					(
						If[option === "SessionTime",
							(*THEN*)
							(*return a refreshed object so it updates like a clock*)
							(
								If[TrueQ[$Links[sessionUUID,"Active"]],
									(*THEN*)
									(*session is running return a dynamically updating time*)
									Refresh[Round[AbsoluteTime[]-$Links[sessionUUID,"SessionTime"]],UpdateInterval->1],
									(*ELSE*)
									(*session is dead, return static time*)
									Round[$Links[sessionUUID,"StopTime"] - $Links[sessionUUID,"SessionTime"]]
								]
							),
							(*ELSE*)
							(*just return normal version*)
							$Links[sessionUUID,option]
						]
					),
					(*ELSE*)
					(*invalid option - raise message*)
					(
						Message[Options::optnf,option,session];
						$Failed
					)
				]
			),
			(*ELSE*)
			(*doesn't exist, so issue message*)
			(
				Message[ExternalSessionObject::invalidSession,session];
				$Failed
			)
		]
	)
]

(*form for RegisterExternalEvaluator that works with File[...] specification*)
RegisterExternalEvaluator[lang_?StringQ,File[exec_?StringQ]]:=RegisterExternalEvaluator[lang,exec]

(*RegisterExternalEvaluator is a function to add new evaluators that may not have been discovered automatically*)
(*the only relevant information the user provides is the location and the name of the language associated with the location*)
(*everything else can be determined from the process (assuming it exists and is valid)*)
RegisterExternalEvaluator[lang_?StringQ,exec_?StringQ]:=Block[
	{
		assoc,
		cachedInstallKey,
		cacheEntry
	},
	(
		checkCache[];
		If[!MemberQ[$UpdatedLanguages,lang],
			updatePacletPlugin[lang];
		];
		Which[
			!TrueQ@FileExistsQ[exec] || (FileExistsQ[exec] && FileType[exec] =!= File),
			(*then the location isn't usable because the binary doesn't exist*)
			(
				Message[RegisterExternalEvaluator::location,exec];
				$Failed
			),
			!knownLanguageQ[lang],
			(*then the language specified isn't known to our system - check if we can find it remotely*)
			(
				If[FailureQ[pacletSystemCheck[lang]],
					(*THEN*)
					(*we failed to find the paclet on the servers to install, so we just fail*)
					(
						Message[RegisterExternalEvaluator::unknownSys,lang];
						$Failed
					),
					(*ELSE*)
					(*we got the paclet, just recursively call this function again*)
					(
						RegisterExternalEvaluator[lang,exec]
					)
				]
			),
			FileExistsQ[exec] && FileType[exec] === File,
			(* Update existing, or validate and make a new entry *)
			(
				assoc = makeInstallEntryAssoc[lang,exec];
				addInstallEntry[lang,assoc,"UserRegistered", "TryBlacklisted"->True, "IssueMessage"->True]
			),
			True,
			(*some other kind of unknown error*)
			(
				$Failed
			)
		]
	)
]

(*form for UnregisterExternalEvaluator that works with the uuid's returned from FindExternalEvaluators[]*)
UnregisterExternalEvaluator[uuid_?StringQ]:=Block[
	{
		uuids,
		lang,
		exec,
		cache
	},
	(
		checkCache[];
		(*see if the uuid exists in the cache*)
		If[KeyExistsQ[uuid]@$ExternalEvaluatorCache["Installations"],
			(*THEN*)
			(*found the uuid in the installations language and can just mark the Registered key False*)
			(
				(*save a copy of the cache in memory*)
				cache = $ExternalEvaluatorCache;

				(*mark the key*)
				cache["Installations",uuid,"Registered"] = False;

				(*safely assign back to the on disk cache*)
				safeCacheAssign[cache];

				(*return the uuid*)
				uuid
			),
			(*ELSE*)
			(*didn't find this entry*)
			(
				Message[UnregisterExternalEvaluator::invalidUUID,uuid];
				$Failed
			)
		]
	)
]

(*File[...] wrapper form is also supported for UnregisterExternalEvaluator, just removing the File wrapper to call the main one*)
UnregisterExternalEvaluator[lang_?StringQ,File[exec_?StringQ]]:=UnregisterExternalEvaluator[lang,exec]

UnregisterExternalEvaluator[lang_?StringQ,exec_?StringQ]:=Block[
	{
		assoc,
		uuid
	},
	(
		checkCache[];
		If[!MemberQ[$UpdatedLanguages,lang],
			updatePacletPlugin[lang];
		];
		(*to be able to unregister the evaluator, it needs to be an existing language and an existing executable*)
		Which[
			!TrueQ@FileExistsQ[exec] || (FileExistsQ[exec] && FileType[exec] =!= File),
			(*executable doesn't exist*)
			(
				Message[UnregisterExternalEvaluator::location,exec];
				$Failed
			),
			!knownLanguageQ[lang],
			(*unknown language*)
			(
				If[FailureQ[pacletSystemCheck[lang]],
					(*THEN*)
					(*we failed to find the paclet on the servers to install, so we just fail*)
					(
						Message[UnregisterExternalEvaluator::unknownSys,lang];
						$Failed
					),
					(*ELSE*)
					(*we got the paclet, just recursively call this function again*)
					(
						UnregisterExternalEvaluator[lang,exec]
					)
				]
			),
			FileType[exec] === File,
			(*exists as a file, so ensure that it's a valid installation, then blacklist it*)
			(
				(*check to make sure this file / exec exists in the cache*)
				uuid = SelectFirstKey[#Executable === exec && #System === lang &]@$ExternalEvaluatorCache["Installations"];

				If[!MissingQ[uuid],
					(*THEN*)
					(*valid install - we can blacklist it*)
					(
						(*save a copy of the cache in memory*)
						cache = $ExternalEvaluatorCache;

						(*mark the key - note we have to unwrap the Key[...] wrapper around uuid*)
						cache["Installations",First[uuid],"Registered"] = False;

						(*safely assign back to the on disk cache*)
						safeCacheAssign[cache];

						(*return the uuid*)
						First[uuid]
					),
					(*ELSE*)
					(*the system doesn't exist, so we should confirm that it's usable, add it to the cache, then mark it as not usable*)
					(*this might happen for example when a user has never registered a particular instance that would be automatically found (or not)*)
					(*but hasn't been found yet, but they know it exists and want to blacklist it before it's ever used*)
					(
						(*validate that the language is a usable instance*)
						assoc = makeInstallEntryAssoc[lang,exec];
						addInstallEntry[lang,assoc,"UserBlacklisted", "TryBlacklisted"->True, "IssueMessage"->True]
					)
				]
			),
			True,
			(
				$Failed
			)
		]
	)
];

(*************************************)
(***********              ************)
(*********** CACHE SEARCH ************)
(***********              ************)
(*************************************)


(*this will attempt to use the cache as well as heuristics for the given language to find a suitable installation location on the current system*)
(*if it works, then it will return the uuid of this installation in the Installations key in the persistent value*)
(*it first checks the localobject cache for user defined values, then searches for previously found values stored, then finally will actually use the heuristics itself*)
(*NOTE - the special identifier "*" is used to represent any version, this may change to simply _ or some other WL symbolic construct*)
resolveLangInstall[lang_,version_:"*",opts_:{}]:=Block[
	{
		languuid = None,
		install,
		possibleUUIDs,
		possibleInstalls
	},
	(
		checkCache[];
		possibleUUIDs = Keys[Select[$ExternalEvaluatorCache["Installations"], #System === lang &]];
		(*first check the cache local object to see if there's anything stored here*)
		If[possibleUUIDs =!= {},
			(*THEN*)
			(*something at least exists in the cache for this language*)
			languuid = SelectFirstKey[validLangInstallQ[makeInstallEntryAssoc[lang,#Executable],"Version"->version,"Constraints"->opts]&] @ 
				KeyTake[possibleUUIDs] @ $ExternalEvaluatorCache["Installations"];
			If[!MissingQ[languuid], Return[First[languuid]]];
		];
		(* Find lang install *)
		findLangInstall[lang,version,"Constraints"->opts]
	)
]

(*addInstallEntry is the only function that writes install data to the persistentvalue and is used to add new language installs that the system has found to the cache*)
Options[addInstallEntry] = {"IssueMessage"->False, "Version"->"*", "Constraints"->{}, "TryBlacklisted"->False};
addInstallEntry[lang_,entry_,type:("DiscoveredInstallations"|"UserRegistered"|"UserBlacklisted"), opts:OptionsPattern[]]:=Block[
	{
		foundUUID,
		cache,
		entryAssoc,
		dependencies,
		existing,
		(* Build the entry details using the system version and executable keys from the entry assoc*)
		cachedEntry = KeyTake[entry, {"System","Version","Executable"}]
	},
	checkCache[];
	cache = $ExternalEvaluatorCache;

	(* Check for existing installation. *)
	existing = SelectFirstKey[$ExternalEvaluatorCache["Installations"], (#System === cachedEntry["System"] && #Executable === cachedEntry["Executable"])&];


	If[!MissingQ[existing],
		foundUUID = First[existing];
		(* No need to update cached if just discovering. *)
		If[type === "DiscoveredInstallations" && cache["Installations",foundUUID,"Registered"] =!= "MissingDependencies",
			Return[First[existing]]
		]
	];

	(* Determine registered key for this install. *)
	Which[
		type==="DiscoveredInstallations", cachedEntry["Registered"] = Automatic,
		type ==="UserRegistered",	cachedEntry["Registered"] = True,
		type==="UserBlacklisted",	cachedEntry["Registered"] = False
	];

	If[!validLangInstallQ[entry, "IgnoreDependencies"->True, makeSequence@FilterRules[{opts}, Options[validLangInstallQ]]],
		Return[$Failed]
	];

	If[cachedEntry["Registred"] =!= False,
		(* Make sure dependencies are not missing, if not unregistering *)
		dependencies = hasDependencies[entry, makeSequence@FilterRules[{opts}, Options[hasDependencies]]];
		If[!TrueQ[dependencies] && cachedEntry["Registered"] =!= False,
			cachedEntry["Registered"] = "MissingDependencies"
		];
	];

	If[!MissingQ[existing],
		(* Update the existing installation *)
		foundUUID = First[existing];
		,
		(* Create a new installation *)
		foundUUID = CreateUUID[];
	];

	If[cachedEntry["Registered"] === True,
		PrependTo[cache["Installations"], foundUUID->cachedEntry]
		,
		cache["Installations", foundUUID] = cachedEntry
	];

	(*update the persistent cache and return the uuid we generated*)
	safeCacheAssign[cache];

	(* don't return uuid as valid install if it's missing dependencies *)
	If[cachedEntry["Registered"] === "MissingDependencies",
		Return[$Failed]
	];


	(*return the uuid for this install*)
	foundUUID
];


(*helper function for selecting a key out of an association where they key's value matches a predicate*)
SelectFirstKey[assoc_,pred_,default_:Missing["NotFound"]]:=Block[
	{pos = LengthWhile[assoc,Not@*pred]},
	If[pos === Length[assoc],
		(*THEN*)
		(*we didn't find it, so return the default*)
		(default),
		(*ELSE*)
		(*we did find it, so return that key, incrementing the position, because LengthWhile counts the number of things before this element that are false for the predicate*)
		(Key[Keys[assoc][[pos+1]]])
	]
]

(*operator form*)
SelectFirstKey[pred_][assoc_]:=SelectFirstKey[assoc,pred]


(*validLangInstallQ will query the provided language installation for 4 things - *)
(*1 - if the files exist*)
(*2 - if the language is of the specified version - this test is ignored if version is "*" *)
(*3 - if the language is usable for zmq+json external evaluation*)
(*4 - if the executable location is blacklisted*)

(*langAssoc is an association with the following keys : *)
(* Executable - a file path to the executable to run files with *)
(* System - a string of what language this represents*)
(* VersionExecCommandFunction - a function of 1 arg which is the executable that returns a list suitable for RunProcess that can be run as a command in the terminal to print off the version text to stdout *)
(* VersionSameQFunction - a function of 2 args that compares the output of the command line version and a user specified version *)
(* ScriptExecCommandFunction - a function of 2 args, the executable and the script to run, that returns a list suitable for RunProcess that can be used to execute a script *)
(* DependencyTestFile - a file name path that points to a script which tests whether or not the required dependencies are available for use and prints off to stdout true/false *)
Options[validLangInstallQ] = {"Version"->"*", "Constraints"->{}, "IssueMessage"->False, "TryBlacklisted"->False, "IgnoreDependencies"->False}
validLangInstallQ[langAssoc_, opts:OptionsPattern[]] := Block[{installEntry, issueMessage, tryBlacklisted,
			constraints, version, exec, missingDependencies, registered, ignoreDependencies},
	issueMessage = TrueQ[OptionValue["IssueMessage"]];
	tryBlacklisted = TrueQ[OptionValue["TryBlacklisted"]];
	ignoreDependencies = TrueQ[OptionValue["IgnoreDependencies"]];
	constraints = OptionValue["Constraints"];
	version = OptionValue["Version"];
	exec = langAssoc["Executable"];

	(*make sure the executable exists*)
	If[!TrueQ@FileExistsQ[langAssoc["Executable"]] || FileType[langAssoc["Executable"]] =!= File,
		If[issueMessage,
			Message[StartExternalSession::invalidExec, langAssoc["Executable"]]
		];
		Return[False]
	];
	
	(* Check against blacklisted installs*)
	If[!tryBlacklisted,
		If[MemberQ[
				$ExternalEvaluatorCache["Languages",langAssoc["System"], "BlacklistedInstalls"]
				,
				<|___,"Executable"->exec,___|>
			],
			If[issueMessage,
				Message[StartExternalSession::unregistered];
			];
			Return[False]
		];

		installEntry = SelectFirst[$ExternalEvaluatorCache["Installations"], (#Executable === exec) &];
		registered = installEntry["Registered"];
		If[registered === False,
			If[issueMessage,
				Message[StartExternalSession::unregistered];
			];
		 	Return[False]
		];
	];

	If[!ignoreDependencies,
		If[registered === "MissingDependencies",
			If[issueMessage,
				Message[StartExternalSession::depend];
			];
		 	Return[False]
		];
	];

	(*to compare the executable specified version with the user specified string we use the VersionSameQFunction - but only if the version isn't "*", as if it's "*" we don't care about the version*)
	If[version =!= "*",
		If[
			!TrueQ[langAssoc["VersionSameQFunction"][
				(*conform the version strings before passing them to the same q function*)
				langAssoc["VersionStringConformFunction"][getVersionInstall[langAssoc["VersionExecCommandFunction"],langAssoc["Executable"]],False],
				langAssoc["VersionStringConformFunction"][version,True]
			]]
			,
			If[issueMessage,
				Message[StartExternalSession::version]
			];
			Return[False]
		]
	];

	(*cofirm the options*)
	If[constraints =!= {},
		(*options isn't an empty list, so we have to confirm that this executable matches the options*)
		If[!langAssoc["ExecutableSessionOptionsMatchQFunction"][langAssoc["Executable"],constraints],
			If[issueMessage,
				Message[StartExternalSession::langopts, constraints];
			];
			Return[False]
		]
	];

	True
]

Options[hasDependencies] = {"IssueMessage"->False, "Version"->"*", "Constraints"->{}};
hasDependencies[langAssoc_, opts:OptionsPattern[]] := Block[{issueMessage, testResult},
	issueMessage = TrueQ[OptionValue["IssueMessage"]];

	(*finally run the script file to test dependencies - ensuring the file exists and is a string, and that the output of running that command is true*) 
	Switch[langAssoc["DependencyTestFile"],
		(*no file specified*)
		None,
		True,
		(*list of files specified*)
		{(_?StringQ)..},
		(
			If[!AllTrue[
					langAssoc["DependencyTestFile"],
					(#=!=""  && FileExistsQ[#])&
				]
				,
				If[issueMessage,
					Message[StartExternalSession::nodepend];
				];
				Return[False]
			];
			If[!AllTrue[
					langAssoc["DependencyTestFile"]
					,
					(
					testResult = Quiet[RunProcess[
						langAssoc["ScriptExecCommandFunction"][None,langAssoc["Executable"],#,{}],
						"StandardOutput"
					]];
					(StringQ[testResult]&&ToUpperCase[StringTrim[testResult]] === "TRUE"))&
				]
				,
				If[issueMessage,
					Message[StartExternalSession::depend];
				];
				Return[False]
			]
		),
		(_?StringQ),
		(
			If[!FileExistsQ[langAssoc["DependencyTestFile"]],
				If[issueMessage,
					Message[StartExternalSession::nodepend];
				];
				Return[False]
			];
			testResult = Quiet[RunProcess[
				langAssoc["ScriptExecCommandFunction"][None,langAssoc["Executable"],langAssoc["DependencyTestFile"],{}],
				"StandardOutput"
			]];
			If[!StringQ[testResult] || ToUpperCase[StringTrim[testResult]] =!= "TRUE",
				If[issueMessage,
					Message[StartExternalSession::depend];
				];
				Return[False]
			]
		),
		(*if the dependency test file is anything other than a string or list of strings, then we can't execute it, and thus fail*)
		_,
		Return[False]
	];
	True
]


(*we should search the path for installs as well - note on os x we need to set the path variable up seperately*)
(*see https://mathematica.stackexchange.com/questions/99704/why-does-mathematica-use-a-different-path-than-terminal*)
getEnvironmentPathDirs[]:=
	(*in some cases the directories here might not exist, so just grab the ones that do exist*)
	Select[FileExistsQ]@(ExpandFileName/@StringSplit[
		If[$OperatingSystem === "MacOSX" && $FrontEnd =!= Null,
			(*THEN*)
			(*we can't necessarily trust the Environment["PATH"] value we get and should calculate it*)
			(*ourselves*)
			Import[
				(*first this loads the paths using the OSX leopard + utility of path_helper*)
				(*then loads bash_profile and bashrc and returns the $PATH*)
				"!eval `/usr/libexec/path_helper -s`; source ~/.bash_profile; source ~/.bashrc; echo $PATH",
				"Text"
			],
			(*ELSE*)
			(*standard install, just use environment*)
			Environment["PATH"]
		],
		":"
	]);

(*search for all installations that have an executable file matching the pattern in the directories*)
(*this will use the heuristics for the language and the specified version to attempt to find the install*)
Options[findLangInstall] = {"Constraints"->{}}
findLangInstall[lang_,version_,opts:OptionsPattern[]]:=Block[
	{
		(*the paths to search on*)
		paths = $LanguageHeuristics[lang,"ExecutablePathFunction"][version],
		(*the StringExpression or String that matches possible executables*)
		executablePattern = $LanguageHeuristics[lang,"ExecutablePatternFunction"][version],
		possibleInstalls,
		validInstall,
		entryAssocs,
		constraints
	},
	(
		constraints = OptionValue["Constraints"];

		(*search for the possible installs*)
		possibleInstalls = Select[FileExistsQ]@FileNames[executablePattern,Join[paths,getEnvironmentPathDirs[]]];

		If[possibleInstalls === {},
			(*none were found at all, return the Failure object*)
			Failure["NoLanguage",<||>]
			,
			(*possible installs found, test and add them*)
			(
				(*build entries*)
				entryAssocs = makeInstallEntryAssoc[lang,#]& /@ possibleInstalls;
				(*verify and add them*)
				possibleInstalls = addInstallEntry[lang, #, "DiscoveredInstallations", "Version"->version, "Constraints"->constraints]&/@entryAssocs;
				(*Use the first new valid entry*)
				validInstall = SelectFirst[possibleInstalls, StringQ];

				If[!MissingQ[validInstall],
					validInstall
					,
					Failure["NoValid",<||>]
				]
			)
		]
	)
]

(* Cache version info per session, to avoid excess runprocesses *)
$versionCache = <||>;
(*get version install will run the executable's version command, then combine stdout with stderr (this is necessary cause some programs like Python always output to stderr for some reason when used with RunProcess)*)
getVersionInstall[commandFunc_,File[executable_]]:=getVersionInstall[commandFunc,executable]
getVersionInstall[commandFunc_,executable_]:= Block[{exec, res, version},
	version = $versionCache[executable];
	If[StringQ[version], Return[version]];
	exec = commandFunc[executable];
	If[StringQ[exec], Return[exec]];
	If[!ListQ[exec], Return[""]];
	res = Quiet[RunProcess[exec]];
	If[!AssociationQ[res], $versionCache[executable]=""; Return[""]];
	version = StringJoin@Values@KeyTake[{"StandardOutput", "StandardError"}]@res;
	$versionCache[executable] = version;
	version
]

getVersionLangExec[lang_,File[exec_?StringQ]]:=getVersionLangExec[lang,exec];
getVersionLangExec[lang_,exec_]:=$LanguageHeuristics[lang,"VersionStringConformFunction"][getVersionInstall[$LanguageHeuristics[lang,"VersionExecCommandFunction"],exec],False]

(*a known language is one that either we have heuristics about, or is registered in the EvaluatorCache*)
knownLanguageQ[lang_?StringQ] := MemberQ[lang]@allKnownLanguages[]

(*all known languages are those known to the cache as well as ones we have heuristics about*)
allKnownLanguages[] := Keys@$LanguageHeuristics

(*makeInstallEntryAssoc is a convenience function to build up an association that validLangInstallQ expects from the language and executable*)
(*File[...] wrapper version*)
makeInstallEntryAssoc[lang_,File[executable_?StringQ]]:=makeInstallEntryAssoc[lang,executable];
(*normal version*)
makeInstallEntryAssoc[lang_,executable_]:=<|
		"Executable"->executable,
		"VersionExecCommandFunction"->$LanguageHeuristics[lang,"VersionExecCommandFunction"],
		"ExecutableSessionOptionsMatchQFunction"->$LanguageHeuristics[lang,"ExecutableSessionOptionsMatchQFunction"],
		"ExecutablePathFunction"->$LanguageHeuristics[lang,"ExecutablePathFunction"],
		"ExecutablePatternFunction"->$LanguageHeuristics[lang,"ExecutablePatternFunction"],
		"VersionSameQFunction"->$LanguageHeuristics[lang,"VersionSameQFunction"],
		"VersionStringConformFunction"->$LanguageHeuristics[lang,"VersionStringConformFunction"],
		"ScriptExecCommandFunction"->$LanguageHeuristics[lang,"ScriptExecCommandFunction"],
		"DependencyTestFile"->$LanguageHeuristics[lang,"DependencyTestFile"],
		"ProgramFileFunction"->$LanguageHeuristics[lang,"ProgramFileFunction"],
		"System"->lang,
		"Prolog"->$LanguageHeuristics[lang,"DefaultProlog"],
		"Epilog"->$LanguageHeuristics[lang,"DefaultEpilog"],
		"SessionEpilog"->$LanguageHeuristics[lang,"SessionEpilog"],
		"SessionProlog"->$LanguageHeuristics[lang,"SessionProlog"],
		"Version"->$LanguageHeuristics[lang,"VersionStringConformFunction"][
			getVersionInstall[
				$LanguageHeuristics[lang,"VersionExecCommandFunction"],
				executable
			],
				False
		]
	|>

(*findAllInstalls will search the system for all languages we have heuristics for, putting all of the valid ones into the cache*)
(*we just map the language version over all the language keys we know of in the $Heuristic association*)
findAllInstalls[]:=findAllInstalls/@allKnownLanguages[]

(*the version for a specific language uses the heuristics for the language with the "*" version to find possible installs*)
findAllInstalls[lang_?StringQ]:=Block[{
		(*the paths to search on*)
		paths = $LanguageHeuristics[lang,"ExecutablePathFunction"]["*"],
		executablePattern = $LanguageHeuristics[lang,"ExecutablePatternFunction"]["*"],
		possibleInstalls,
		entryAssocs,
		validInstalls,
		prevDiscoveredLocations = Values[(Select[#System === lang &]@$ExternalEvaluatorCache["Installations"])[[All, "Executable"]]]
	},
	(
		(*search for all installations that have an executable file matching the pattern in the directories*)
		possibleInstalls = Select[FileExistsQ]@FileNames[executablePattern,Join[paths,getEnvironmentPathDirs[]]];

		If[possibleInstalls =!= {},
			(*there are possible installs - select the valid ones and register them*)
			entryAssocs = makeInstallEntryAssoc[lang,#]& /@ possibleInstalls;
			addInstallEntry[lang, #, "DiscoveredInstallations"]& /@ entryAssocs
		]
	)
]
(*string template implementation*)
eeStringTemplateQ[str_?StringQ] := StringMatchQ[str, ___~~"<* " ~~ __ ~~ " *>" ~~ ___];
eeStringTemplateQ[_] = False;

eeStringTemplateEvalFunc[funcArg_?eeStringTemplateQ] := 	Module[{templateOutput},
	templateOutput = StringTemplate[funcArg][];
	If[StringQ[templateOutput],Return@templateOutput,Return[$Failed]]
];

(*************************************)
(***********            **************)
(*********** EVALUATION **************)
(***********            **************)
(*************************************)

(*short hand for a single arg*)
ExternalEvaluate[session_,func_String->arg_]:=
	ExternalEvaluate[session,<|"Command"->func,"Arguments"->{arg}|>]

(*"cooked" form of the assoc - we just turn it into the assoc*)
ExternalEvaluate[session_,func_String->args:{___}]:=
	ExternalEvaluate[session,<|"Command"->func,"Arguments"->args|>]

(*persistent form of ExternalEvaluate that uses a pre-existing session*)
ExternalEvaluate[session_ExternalSessionObject, input:_String?StringQ|_Association?AssociationQ]:=
	errorExit @ Block[
		{link, stringTemplateOutput, sessionUUID = First[session]},
		(*we have an external session to use, so just look up the link from $Links for the uuid*)
		If[
			KeyExistsQ[sessionUUID][$Links] && TrueQ[$Links[sessionUUID,"Active"]],
			(*THEN*)
			(*the session exists for us to use, evaluate the input*)
			If[
				eeStringTemplateQ[input] ,
				stringTemplateOutput = eeStringTemplateEvalFunc[input];
				If[
					stringTemplateOutput=!= $Failed,
					sessionEvaluateWithPrologEpilog[session, stringTemplateOutput],
					Return[$Failed]
				],
				sessionEvaluateWithPrologEpilog[session, input]
			],
			(*invalid session*)
			Message[ExternalEvaluate::invalidSession,session];
			$Failed
		]
	]

ExternalEvaluate[session_ExternalSessionObject, input:{__}]:=
	With[
		{sessionUUID = First[session]},
		checkCache[];
		If[
			KeyExistsQ[sessionUUID][$Links] && TrueQ[$Links[sessionUUID,"Active"]],
			Map[ExternalEvaluate[session, #] &, input],
			(*ELSE*)
			(*invalid session*)
			Message[ExternalEvaluate::invalidSession,session];
			$Failed
		]
	]

(*form with multiple statements to evaluate*)
ExternalEvaluate[spec_,input:{__}]:=
	Replace[
		checkCache[];
		StartExternalSession[spec],
		session_ExternalSessionObject :> With[
			(*evaluate all the input on the session*)
			{res = Map[ExternalEvaluate[session, #] &, input]},

			(* now close the link and return the result, the session might have been aborted *)
			Quiet[
				DeleteObject[session],
				ExternalSessionObject::invalidSession
			];
			res
		]
	]



(*for String form of lang, we should start up a new one-shot session for the code*)
ExternalEvaluate[spec_,input_?(StringQ[#]||AssociationQ[#]&)] :=
	errorExit @ Block[
		{res,session, stringTemplateOutput},
		checkCache[];
		(*make a session with these opts*)
		session = Check[StartExternalSession[spec],$Failed];
		If[
			FailureQ[session],
			(*THEN*)
			(*didn't work, just fail - StartExternalSession will have handled all the messages for us*)
			$Failed,
			If[
				eeStringTemplateQ[input] ,
				(
				stringTemplateOutput = eeStringTemplateEvalFunc[input];
				If[
					stringTemplateOutput =!= $Failed,
					res = sessionEvaluateWithPrologEpilog[session, stringTemplateOutput ],
					Return[$Failed]
				]
				),
				res = sessionEvaluateWithPrologEpilog[session, input];
			];
			DeleteObject[session];
			res
		]
	]

(*handlers for executing a file *)
fileToStringHandle[caller_,args:{___},File[file_?StringQ]]:=Block[
	{lang, res},
	(
		If[file =!= "" && (FileExistsQ[file] || !FailureQ[FindFile[file]]),
			(*THEN*)
			(*continue, the file exists*)
			(
				(*we need to determine the language from the spec so we can lookup the run file code to splice the *)
				(*file in*)
				lang = getLangFromSpec[First[args]];
				If[!FailureQ[lang],
					(*THEN*)
					(*we can now safely evaluate the string, as we know what language it is*)
					(
						res = $LanguageHeuristics[lang,"RunFileInSessionFunction"][
							(*this canonicalizes the file path so it can be pasted into a string easily, using only / over \\ and such*)
							(*it also uses WL convention/directory structure to resolve relative paths to absolute paths before passing it along*)
							StringJoin[Riffle[FileNameSplit[FindFile[file]], "/"]]
						];
						If[res === None,
							(*THEN*)
							(*system doesn't have a function to run a file in the session, so fail*)
							(
								Message[ExternalEvaluate::nofilefunc,lang];
								$Failed
							),
							(*ELSE*)
							(*the runfile in session function exists, so we can run it*)
							caller[
								##,
								res
							]&@@args
						]
					),
					(*ELSE*)
					(*failed to get the language from the spec, so fail*)
					$Failed
				]
			),
			(*ELSE*)
			(*file doesn't exist, issue message*)
			(
				Message[ExternalEvaluate::invalidFile,file];
				$Failed
			)
		]
	)
]



cloudObjectToFileHandle[caller_,args:{___},co:HoldPattern[CloudObject[___]]]:=Block[
	{
		fileStr,
		file ,
		strm,
		res
	},
	(
		(*see if we can figure anything out from the mimetype - because if a user did CopyFile["file.js",CloudObejct[]], then CloudGet'ing it will fail*)
		(*but for now the only recognized mime type is .js files, which show up as application/json and sometimes text/javascript*)
		If[MemberQ[{"application/json","text/javascript"},ToLowerCase[CloudObjectInformation[co, "MIMEType"]]] || ToLowerCase[FileExtension[Last[URLParse[First[co],"Path"]]]] === "py",
			(*THEN*)
			(*because we know that the mimetype is known type we can import as a raw URL, so just handle it as a URL of the CloudObject*)
			Return[caller[##,URL[First[co]]]&@@args],
			(*ELSE*)
			(*not a known mimetype, so CloudGet it normally, hoping it turns out to be a string*)
			fileStr = Check[CloudGet[co],$Failed]
		];
		(*check that the file downloaded successfully*)
		Which[
			StringQ[fileStr],
			(*successfully imported and it's a valid string, export it to a file and execute the file*)
			(
				(*make a new unique file with the specified name from taken from the cloud object*)
				file = CreateFile[uniqueLocalTempFile[co]];
				(*write out the string to the file and then execute it, deleting the file after we're done*)
				strm = WriteString[file,fileStr];
				Close[file];
				res = caller[##,File[file]]&@@args;
				DeleteFile[file];
				res
			),
			FailureQ[fileStr],
			(*failed to download the cloud object, but we should have gotten messages above should have issued a message for us*)
			(
				$Failed
			),
			True,
			(
				(*getting the file didn't fail, but it's not a failure, so it's the wrong type of expression and we can't evaluate it*)
				Message[ExternalEvaluate::invalidObject,"CloudObject",co];
				$Failed
			)
		]
	)
];

localobjectToFileHandle[caller_,args:{___},lo:HoldPattern[LocalObject[___]]]:=Block[
	{
		(*attempt to download the file - note that if the page returns 404, then we'll still get a valid http response and*)
		(*thus still a valid file, albeit likely that the next step will fail...*)
		fileStr = Check[Get[lo],$Failed],
		file,
		res,
		strm
	},
	(
		(*check that the file downloaded successfully*)
		Which[
			StringQ[fileStr],
			(*successfully imported the object as a string, export it to a file and execute the file*)
			(
				(*create a unique file name from the localobject*)
				file = CreateFile[uniqueLocalTempFile[lo]];
				strm = WriteString[file,fileStr];
				Close[file];
				res = caller[##,File[file]]&@@args;
				DeleteFile[file];
				res
			),
			FailureQ[fileStr],
			(*failed to get the LocalObject, but Get will have issued appropriate messages for us*)
			(
				$Failed
			),
			(*ELSE*)
			(*imported the local object, but it wasn't a string*)
			True,
			(
				Message[ExternalEvaluate::invalidObject,"LocalObject",lo];
				$Failed
			)
		]
	)
]

urlToFileHandle[caller_,args:{___},url_:URL[_]]:=Block[
	{
		(*get a unique filename to download the url into, note we don't make this file, as URLDownload will do this for us*)
		file = uniqueLocalTempFile[url],
		res
	},
	(
		(*attempt to download the file - note that if the page returns 404, then we'll still get a valid http response and*)
		(*thus still a valid file, albeit likely that the next step will fail...*)

		file = Check[URLDownload[url,file],$Failed];
		(*check that the file downloaded successfully*)
		If[
			(*make sure that it's a valid File spec, which could be either a File[...] object, or a string object*)
			((StringQ[file]) || (Head[file] === File && StringQ[First[file]])) &&
			(*and make sure the file actually exists*)
			FileExistsQ[file],
			(*THEN*)
			(*successfully downloaded, now just call it as you would normally as a file, note that URLDownload returns a File[...] wrapper already*)
			(
				res = caller[##,file]&@@args;
				DeleteFile[file];
				res
			),
			(*ELSE*)
			(*failed to download, but URLDownload should have issued a message for us*)
			(
				$Failed
			)
		]
	)
];


(*call the special handler for cloud objects*)
ExternalEvaluate[spec_,co:HoldPattern[CloudObject[___]]]:=
	cloudObjectToFileHandle[ExternalEvaluate,{spec},co]

(*call the special handler for local objects*)
ExternalEvaluate[spec_,lo:HoldPattern[LocalObject[___]]]:=
	localobjectToFileHandle[ExternalEvaluate,{spec},lo]

(*call the special handler for urls*)
ExternalEvaluate[spec_,url:URL[_]]:=
	urlToFileHandle[ExternalEvaluate,{spec},url]

(*the speical handler for files will turn the file into a string that loads it*)
ExternalEvaluate[spec_,File[file_?StringQ]]:=
	fileToStringHandle[ExternalEvaluate,{spec},File[file]]


(*URLDownload will by default the file as a uuid.tmp file, but a smarter thing to do is to take the actual name of the *)
(*file / site being downloaded, so we specify the file name path inside a new unique directory inside $TemporaryDirectory*)

(*for a local object, take the last element of the underlying file as the unique name*)
uniqueLocalTempFile[lo_LocalObject]:= uniqueLocalTempFile[FileNameTake[LocalObjects`PathName[lo]]]
(*for a cloud object, extract the url to use the URL form*)
uniqueLocalTempFile[co_CloudObject]:= uniqueLocalTempFile[URL[First[co]]]
(*for URL's, look up the last part of the url as the name to use for the temp file*)
uniqueLocalTempFile[URL[url_]]:= uniqueLocalTempFile[Last[URLParse[url, "Path"]]]
(*to make an always unique temp file with a specified name, we simply make a folder in the temp directory with a UUID, then save the file in there*)
(*this ensures if you call the function with the same file name twice, you'll get the same FileBaseName, but unique, different absolute paths*)
uniqueLocalTempFile[name_]:=Block[
	{
		(*make a unique directory we can download the file into - this ensures that executing the same file twice doesn't fail*)
		dir = CreateDirectory[FileNameJoin[{$TemporaryDirectory, CreateUUID[]}]]
	},
	(*simply join together the directory and the specified name*)
	FileNameJoin[{dir, name}]
]

(*operator form*)
ExternalEvaluate[spec_][input__]:=ExternalEvaluate[spec,input]

(*error form for any catch all expressions*)
ExternalEvaluate[spec_,any__]:=(Message[ExternalEvaluate::invalidInput,any];$Failed);

(*convenient wrapper for the various types of evaluation*)
(*these are necessary / useful for handling the Epilog/Prolog and SessionProlog and SessionEpilog*)
(*which can be a string, a File, a URL, CloudObject, or LocalObject*)
externalEvaluateLinkSession[session_,input_?StringQ]:=Block[
	{
		exprQ = session["ReturnType"] =!= "String",
		sys = session["System"],
		res
	},
	res = $LanguageHeuristics[sys,"NonZMQEvaluationFunction"][
		session,
		input,
		exprQ
	];
	If[
		res =!= None,
		(*THEN*)
		(*don't use normal evaluation mode, need to use the framework WL function to evaluate*)
		res,
		(*ELSE*)
		(*standard evaluation mode*)
		externalEvaluateLink[
			session,
			<|
				"input"->input<>"\n",
				"return_type"->If[exprQ,"expr","string"],
				"session_uuid"->First[session]
			|>,
			input,
			exprQ
		]
	]
]


externalEvaluateLinkSession[session_,input_?AssociationQ]:=Block[
	{
		exprQ = session["ReturnType"] =!= "String",
		sys = session["System"],
		res
	},
	res = $LanguageHeuristics[sys,"NonZMQEvaluationFunction"][
		session,
		input,
		exprQ
	];
	If[res =!= None,
		(*THEN*)
		(*don't use normal evaluation mode, need to use the framework WL function to evaluate*)
		(
			res
		),
		(*ELSE*)
		(*standard evaluation mode*)
		(
			externalEvaluateLink[
				session,
				(*fastest form of *)
				Join[
					input,
					<|
						"return_type"->If[exprQ,"expr","string"],
						"session_uuid"->First[session]
					|>
				],
				input,
				exprQ
			]
		)
	]
]

externalEvaluateLinkSession[session_,File[file_?StringQ]]:=
	fileToStringHandle[externalEvaluateLinkSession,{session},File[file]]

(*for the url case, recurse to a file form by downloading the file first*)
externalEvaluateLinkSession[session_,URL[url_]]:=
	urlToFileHandle[externalEvaluateLinkSession,{session},url]

(*call the special handler for cloud objects*)
externalEvaluateLinkSession[session_,co:HoldPattern[CloudObject[___]]]:=
	cloudObjectToFileHandle[externalEvaluateLinkSession,{session},co]

(*call the special handler for local objects*)
externalEvaluateLinkSession[session_,lo:HoldPattern[LocalObject[___]]]:=
	localobjectToFileHandle[externalEvaluateLinkSession,{session},co]

$listEvalpat = {(HoldPattern[File[___]] | HoldPattern[CloudObject[___]] | HoldPattern[LocalObject[___]] | HoldPattern[URL[___]] | _?StringQ | _?AssociationQ) ...};

externalEvaluateLinkSession[session_,evals:$listEvalpat]:=
	externalEvaluateLinkSession[session,#]&/@evals

(*anything else we fail*)
externalEvaluateLinkSession[session_,___]:= $Failed

sessionEvaluateWithPrologEpilog[session_,input_String]:=Block[
	{
		res
	},
	(
		(*check if we have to evaluate a prolog*)
		If[session["Prolog"] =!= None,
			externalEvaluateLinkSession[session,session["Prolog"]]
		];

		(*evaluate the input*)
		res = externalEvaluateLinkSession[session,input];

		(*increment the evaluation count by 1*)
		$Links[First[session],"EvaluationCount"]++;

		(*check if we have to evaluate a prolog*)
		If[session["Epilog"] =!= None,
			externalEvaluateLinkSession[session,session["Epilog"]]
		];

		res
	)
]

(*same function for Associations that will serialize arguments and such*)
sessionEvaluateWithPrologEpilog[session_,inputOrig_?AssociationQ]:=Block[
	{
		res,
		inKeys = Keys[inputOrig],
		input = inputOrig
	},
	(
		(*check if the Arguments key is missing, if it is then we can just add an argument of empty list*)
		(*for no arguments to the function*)
		If[!KeyExistsQ["Arguments"]@input,
			input["Arguments"] = {};
		];

		(*if the command argument doesn't exist then issue message and return $Failed*)
		If[!KeyExistsQ["Command"]@input,
			Message[ExternalEvaluate::assockeys,"Command","required"];
			Return[$Failed];
		];

		If[Complement[inKeys,{"Command","Arguments"}]=!={},
			Message[ExternalEvaluate::assockeys,Complement[{"Command","Arguments"},inKeys],"unknown"];
			Return[$Failed];
		];

		(*check if we have to evaluate a prolog*)
		If[session["Prolog"] =!= None,
			externalEvaluateLinkSession[session,session["Prolog"]]
		];

		(*now make the request and handle it*)
		res = externalEvaluateLinkSession[
			session,
			<|
				"function"->input["Command"],
				"args"->input["Arguments"]
			|>
		];

		(*increment the evaluation count by 1*)
		$Links[First[session],"EvaluationCount"]++;

		(*check if we have to evaluate a prolog*)
		If[session["Epilog"] =!= None,
			externalEvaluateLinkSession[session,session["Epilog"]]
		];

		res
	)
]

(*
	the loop writes the message and keeps waiting for messages until the response is NOT PythonKeepListening.
	python might be able to redirect stdout, to do that it will be sending a side effect in the form of
	PythonKeepListening[Print["foo"]], the side effect is evaluated immediately.
	The final result of the computation is not wrapped in PythonKeepListening and loop ends.
*)



SetAttributes[{errorHandler, errorExit}, HoldAllComplete]

errorExit[code_] := 
	Catch[code, internalTag]
errorHandler[session_, code_] :=
    Replace[
        CheckAll[code, HoldComplete], {
            (* everything was fine, we return the result*)
            _[res_, Hold[]] :> 
                res,

            (* Abort[] or Throw should just be propagated. in any case the error was 
               thrown in the middle of an IO operation, we cannot trust the process 
               to work on the next evaluation. we must 🔪 the process.
            *)
            _[res_, Hold[Abort[]]] :> (
            	DeleteObject @ session;
            	Throw[$Aborted, internalTag]
            ),
            _[res_, Hold[throw_]] :> (
            	DeleteObject @ session;
            	throw
            )
        }
    ]

externalEvaluateLoop[session_, message_String, rest___] :=
	externalEvaluateLoop[session, StringToByteArray[message], rest]

externalEvaluateLoop[session_, message_ByteArray?ByteArrayQ, write_, read_] :=
	With[
		{socket = session["Socket"]},
		errorHandler[
			session,
			NestWhile[
				(* blocking function that is importing the result of waiting for a message *)
				read[SocketReadMessage[socket]] &,
				(* we write the message and we prevent NestWhile from quitting the loop *)
				write[socket, message];
				PythonKeepListening[],
				(* the result should NOT be PythonKeepListening *)
				MatchQ[_PythonKeepListening]
			]
		]
	]

externalEvaluateLoop[_, message_?FailureQ, ___] := 
	message

externalEvaluateLoop[___] := 
	$Failed

externalEvaluateLink[session_, payload_, ___] := 
	With[{
		serializer   = $LanguageHeuristics[session["System"], "SerializationFunction"],
		deserializer = $LanguageHeuristics[session["System"], "DeserializationFunction"],
		write        = If[
			TrueQ[StringStartsQ[session["Socket"]["Protocol"], "ZMQ"]],
			ZeroMQLink`ZMQSocketWriteMessage,
			BinaryWrite
		]},
		Block[
			(* 
				all external functions are evaluated in the kernel 
				this global variable is used to set a default kernel session
			*)
			{$DefaultSession = session},
			externalEvaluateLoop[session, serializer[payload], write, deserializer]
		]
	]


getLangFromSpec[spec_]:=Which[
	Head[spec] === ExternalSessionObject && KeyExistsQ[First[spec]]@$Links,
	(*session object - we can get the language type by simply querying the object*)
	spec["System"],
	(*string spec of known language*)
	StringQ[spec] && knownLanguageQ[spec],
	spec,
	(*uuid of a session in $Links*)
	StringQ[spec] && KeyExistsQ[spec]@$Links,
	$Links[spec,"System"],
	AssociationQ[spec] && KeyExistsQ["System"]@spec && knownLanguageQ[spec["System"]],
	(*association form*)
	spec["System"],
	ListQ[spec] && StringQ[First[spec]] && knownLanguageQ[First[spec]],
	(*the language should be the first spec, this is the form of ExternalEvaluate[{"Python",...},input]*)
	First[spec],
	Head[spec] === Rule && StringQ[First[spec]] && knownLanguageQ[First[spec]],
	(*the language should be the first spec, this is the form of ExternalEvaluate[{"Python",...},input]*)
	First[spec],
	True,
	(*some other invalid form, let the other form handle this error for us*)
	ExternalEvaluate[spec,""]
]


(*now that we've set everything up, search for all locally installed external evaluate plugin paclets*)
(*note that we don't do any searching on the paclet server, just local ones at startup*)
(*remote ones are loaded lazily, i.e. only searched for when needed/requested*)
(*so to trigger loading of some new system, you have to actually use it with ExternalEvaluate*)
(*or manually download/install the paclet yourself*)
loadInstalledExternalEvaluatePlugins[] := With[
	{
		(*find all ExternalEvaluate_* plugins*)
		localPaclets = First /@ GatherBy[PacletManager`PacletFind["ExternalEvaluate_*"], #["Name"] &]
	},
	(
		(*get all the system files to register all the systems for the paclets we have locally*)
		Get/@Flatten[PacletManager`PacletResource[#, "System"]& /@ localPaclets];
		(*also set that we don't have any systems we've updated yet in this kernel session*)
		$UpdatedLanguages = {};
	)
]


(*function that the FrontEnd uses for evaluating inside ExternalEvaluate cells *)
ExternalEvaluate`FE`ExternalCellEvaluate[sys_,input_] := Module[
	{
		session
	},
	(*check if we have to initialize the cell sessions*)
	If[Not[ValueQ[ExternalEvaluate`FE`$CellSessions]],
		ExternalEvaluate`FE`$CellSessions = <||>
	];
	(*now check to see if we need to start up a session for this system*)
	session = If[And[
		KeyExistsQ[ExternalEvaluate`FE`$CellSessions, sys],
		MemberQ[ExternalSessions[], ExternalEvaluate`FE`$CellSessions[sys]]
	],
		(*THEN*)
		(*there is already a session for this system, so use that one*)
		ExternalEvaluate`FE`$CellSessions[sys],
		(*ELSE*)
		(*there isn't a session already, just start one with the default options*)
		ExternalEvaluate`FE`$CellSessions[sys] = StartExternalSession[sys]
	];
	(*finally if we have a valid session then evaluate the input, otherwise return $Failed*)
	If[session =!= $Failed,
		ExternalEvaluate[session,input],
		$Failed
	]
]

loadInstalledExternalEvaluatePlugins[]

End[]

(*Load other files from the paclet*)
Block[{dir = DirectoryName[$InputFileName]},
	Get[FileNameJoin[{dir, "ExternalFunction.wl"}]]
];

EndPackage[]

(* ::Package:: *)

Begin[ "System`"]        (* Everything in System context *)

General::writewarn = "Defining rule for `1`."

(*
General::spell1 =
    "Possible spelling error: new symbol name \"`1`\" is similar to existing symbol \"`2`\"."

General::spell =
    "Possible spelling error: new symbol name \"`1`\" is similar to existing symbols `2`."
*)

Off[General::newsym]

Off[DumpGet::valwarn]
Off[Compile::noinfo]
Off[General::sandbox]
Off[Part::keyw]
Off[BooleanRegion::drc]
Off[LibraryFunction::pversion]
Off[General::approx]

General::sysmain =
    "Error loading the main binary file `1`.
     Get[\"sysmake.m\"] must be run before continuing."

Begin[ "System`Private`"]

If[  Hold[ $InstallationDirectory] === Hold @@ { $InstallationDirectory},
    $InstallationDirectory = DirectoryName[ $Input, 5]]


If[ Hold[ $SystemFileDir] === Hold @@ { $SystemFileDir},
    $SystemFileDir =
            ToFileName[ {$InstallationDirectory, "SystemFiles", "Kernel",
                "SystemResources", $SystemID}]]

$MessagesDir =
    ToFileName[ {$InstallationDirectory, "SystemFiles", "Kernel", "TextResources"}]

(*
When the mainload.mx system can save a symbol with
a value these should be moved into mainload.mx.
*)

System`NotebookInformation = Developer`NotebookInformation

If[ $OperatingSystem === "MacOS" , SetDirectory[ $InstallationDirectory]]

If [ $OperatingSystem === "MacOSX" && Environment["LANG"] === "ja_JP",
    $SystemCharacterEncoding = "ShiftJIS" ]

(* Visualization-related global constants that should be user-settable *)
System`$PlotTheme = Automatic;
GraphicsGrid`$Autoalign = True;
DynamicGeoGraphics`$ResizeSpeedMultiplier = 2.5;
Charting`$LabelingMemorySizeLimit = 10^6;
Charting`$InteractiveGraphics = False;
Charting`$InteractiveGraphicsVersion = 1.2;
Charting`$DefaultGraphicsInteraction = {
    "Version" -> 1.2,
    "TrackMousePosition" -> {True, False},
    "Effects" -> {
        "Highlight" -> {"ratio" -> 2},
        "HighlightPoint" -> {"ratio" -> 2},
        "Droplines" -> {"freeformCursorMode" -> True, "placement" -> {"x" -> "All", "y" -> "None"}}
    }
};

Charting`$DefaultGraphicsInteractionV1 =
    <|
        "InteractiveGraphics" -> <|
            "version" -> "1.1",
            "TrackMousePosition" -> <|"x" -> True, "y" -> False|>,
            "effects" -> {
                <| "type" -> "HighlightPoint", "ratio" -> 2|>,
                <|
                    "type" -> "Droplines",
                    "freeformCursorMode" -> True,
                    "linePlacement" -> <|"x" -> "All", "y" -> "None"|>
                |>
            }
        |>
    |>;

(*
 Utility function for building MX files for an Application
*)

System`Private`BuildApplicationMXFunction[ {appName_, context_, outputMX_, path_}] :=
    Module[{outFile, app},
        Print[ "FF " , path, " ", context];
        app = appName <> "`";
        Get[app];
        CreateDirectory[ ToFileName[{path, appName, "Kernel","SystemResources"}, $SystemID],
                CreateIntermediateDirectories -> True];
        outFile = ToFileName[{path, appName, "Kernel","SystemResources", $SystemID}, outputMX];
        Print[ "Out file is ", outFile];
        DumpSave[outFile, context, HoldAllComplete];
    ]


Which[
    Names[ "System`$SystemInstall"] =!= {}
    ,
        System`Private`$SysmakeError = 0;
        AppendTo[ $Path,
            ToFileName[
                {$InstallationDirectory, "AddOns", "StandardPackages"},"StartUp"]
        ];
        SetOptions[$Output, PageWidth->Infinity];
        Get[ "sysmake.m"];
        Exit[System`Private`$SysmakeError]
    ,
    ListQ[ System`Private`BuildApplicationMX]
    ,
        System`Private`BuildApplicationMXFunction[System`Private`BuildApplicationMX];
        Exit[]
    ,
    True (* Normal start *)
    ,
        If[
            DumpGet[StringJoin[ $SystemFileDir, ContextToFileName[ "mainload`"], "x"] ] =!= Null,
            Message[ General::sysmain, "mainload`"]
        ];
    ]

Off[ Get::noopen]
Off[ General::initg]
Off[ General::initc]
Off[ General::spell]
Off[ General::spell1]
Off[ General::obspkgfn]


$CharacterEncoding = $SystemCharacterEncoding;

SetDirectory[ ToFileName[{$InstallationDirectory, "SystemFiles", "CharacterEncodings"}]];

System`$CharacterEncodings = StringTake[#, {1, -3}] & /@ FileNames[ "*.m"];

Protect[ System`$CharacterEncodings];

ResetDirectory[];


Internal`AddHandler["MessageTextFilter", Internal`MessageButtonHandler]


Unset[Developer`$InactivateExclusions];


(* Set $Path *)
$Path =
    Which[
         $LicenseType === "Player",
         {
        ToFileName[ {$InstallationDirectory, "SystemFiles"}, "Links"],
        ToFileName[ {$InstallationDirectory, "AddOns"}, "Packages"],
        ToFileName[ {$InstallationDirectory, "SystemFiles"}, "Autoload"],
        ToFileName[ {$InstallationDirectory, "AddOns"}, "Applications"],
        ToFileName[ {$InstallationDirectory, "AddOns"}, "ExtraPackages"],
        ToFileName[ {$InstallationDirectory, "SystemFiles"}, "Components"],
        ToFileName[ {$InstallationDirectory, "SystemFiles","Kernel"}, "Packages"],
        ToFileName[ {$InstallationDirectory, "Documentation",$Language}, "System"],
        "."
         },
         True,
        {
          ToFileName[ {$InstallationDirectory, "SystemFiles"}, "Links"],
        ToFileName[ $UserBaseDirectory, "Kernel"],
        ToFileName[ $UserBaseDirectory, "Autoload"],
        ToFileName[ $UserBaseDirectory, "Applications"],
        ToFileName[ $BaseDirectory, "Kernel"],
        ToFileName[ $BaseDirectory, "Autoload"],
        ToFileName[ $BaseDirectory, "Applications"],
        ".",
        HomeDirectory[],
        ToFileName[ {$InstallationDirectory, "AddOns"}, "Packages"],
        ToFileName[ {$InstallationDirectory, "SystemFiles"}, "Autoload"],
        ToFileName[ {$InstallationDirectory, "AddOns"}, "Autoload"],
        ToFileName[ {$InstallationDirectory, "AddOns"}, "Applications"],
        ToFileName[ {$InstallationDirectory, "AddOns"}, "ExtraPackages"],
        ToFileName[ {$InstallationDirectory, "SystemFiles","Kernel"}, "Packages"],
        ToFileName[ {$InstallationDirectory, "Documentation",$Language}, "System"],
        ToFileName[ {$InstallationDirectory, "SystemFiles","Data"}, "ICC"]
        }];


If[MathLink`NotebookFrontEndLinkQ[$ParentLink],
    System`Private`origFrontEnd = MathLink`SetFrontEnd[$ParentLink],
(* else *)
    System`Private`origFrontEnd = Null
]


(* Load all the non-autoload symbols defined in top-level. *)
Get[FileNameJoin[{$SystemFileDir,"NonAutoLoads.mx"}]]


(*
    Set up parallel computation features.
*)

If[ FindFile["Parallel`Kernel`sysload`"] =!= $Failed &&
        System`Parallel`$NoParallelLoad =!= True,
    Get["Parallel`Kernel`sysload`"]]


(* Run-time init for registration of annotations, see Annotations/Common.m *)

Annotation`AnnotationsDump`init[]

(* Run-time init for Audio`AudioGraph, see Sound/ProgrammaticPlayback/AudioGraph.m *)

Audio`AudioGraphDump`init[]

(* Run-time init for AudioStream, see Sound/ProgrammaticPlayback/AudioStream.m *)

Image`TransformsDump`init[]

(* Run-time init for ImagePyramid, see ImageProcessing/Transforms/Pyramid.m *)

Audio`AudioStreamDump`init[]

(* Run-time init for ShortTimeFourierData, see SignalProcessing/ShortTimeFourierData.m *)

Signal`ShortTimeFourierDataDump`init[]

(* PersistenceLocations pseudo package; this stuff is actually in StartUp, be we want a loadable context *)

If[ !MemberQ[$Packages, "PersistenceLocations`"],
	Block[{$ContextPath = $ContextPath},
		BeginPackage["PersistenceLocations`"]; EndPackage[]
]]

(* Run-time init for Persistence and Initialization, see StartUp/Persistence/ *)

StartUp`Persistence`init[]
StartUp`Initialization`init[]

(*
 Set up autoloading, using the new Package`DeclareLoad functionality.
*)

If[ $LicenseType =!= "Player Pro" && $LicenseType =!= "Player",
    Package`DeclareLoad[
        {System`InstallService},
        "WebServices`", Package`HiddenImport -> True]
]

Package`DeclareLoad[
        {JSONTools`ToJSON,JSONTools`FromJSON},
        "JSONTools`", Package`HiddenImport -> True]

(* dorianb: One day maybe we will completelly remove JSONTool, but that day is not today. *)

Options[Developer`ToJSON] = {"Compact" -> False, "ConversionFunction" -> None, "ConversionRules" -> {}};
Options[Developer`FromJSON] = {};

Developer`ToJSON[expr_, opts:OptionsPattern[]] :=
	Developer`WriteRawJSONString[
		expr,
		"JSONObjectAsList" -> True,
		"PrecisionHandling" -> "Coerce",
		"IssueMessagesAs" -> Developer`ToJSON,
		opts
];

Developer`FromJSON[json_String, opts:OptionsPattern[]] :=
	Developer`ReadRawJSONString[
		json,
		"JSONObjectAsList" -> True,
		"IssueMessagesAs" -> Developer`FromJSON,
		opts
];


Package`DeclareLoad[
        {System`URLFetch, System`URLSave,
          System`URLFetchAsynchronous, System`URLSaveAsynchronous,
          System`$HTTPCookies,System`$Cookies,System`$CookieStore,
         System`SetCookies,System`FindCookies,System`ClearCookies,
         System`CookieFunction,System`URLResponseTime
        },
        "CURLLink`", Package`HiddenImport -> True]

Developer`RegisterInputStream["HTTP",
    StringMatchQ[#, RegularExpression["https?://.*"]] &,
    Needs["CURLLink`"]; CURLLink`HTTP`Private`initialize[]];


Internal`DisableCloudObjectAutoloader[] := Block[{$Path},
    Quiet[System`CloudObject[]]
]

(* CloudSymbol and $CloudBase don't play nice with autoloading, so these defs need to be present at startup, in case a user
   assigns to either before CloudObject` has been loaded.
*)
Begin["CloudObject`Private`"]
System`$CloudBase = Automatic;
System`CloudSymbol /:
    Set[z_System`CloudSymbol, rhs_] /; (System`CloudSymbol; True) := Set[z, rhs] (* immediately delegate to the newly acquired upvalue *)
End[]

(* CloudExpression defines some UpValues on hold first symbols. This attribute prevents paclet from loading and triggers
wrong behaviours. It's particularly a problem for cloud evaluations (CloudEvaluate, APIFunction, ...) where the kernel
is usually fresh.
Using the same trick as for CloudSymbol. *)
Begin["CloudExpression`Main`PackagePrivate`"]
System`CloudExpression /: AppendTo[ce_System`CloudExpression, rhs_] /; (System`CloudExpression; True) := AppendTo[ce, rhs];
System`CloudExpression /: AssociateTo[ce_System`CloudExpression, rhs_] /; (System`CloudExpression; True) := AssociateTo[ce, rhs];
System`CloudExpression /: Set[ce_System`CloudExpression, rhs_] /; (System`CloudExpression; True) := Set[ce, rhs];
System`CloudExpression /: AddTo[ce_System`CloudExpression, rhs_] /; (System`CloudExpression; True) := AddTo[ce, rhs];
System`CloudExpression /: SubtractFrom[ce_System`CloudExpression, rhs_] /; (System`CloudExpression; True) := SubtractFrom[ce, rhs];
System`CloudExpression /: Increment[ce_System`CloudExpression] /; (System`CloudExpression; True) := Increment[ce];
System`CloudExpression /: Decrement[ce_System`CloudExpression] /; (System`CloudExpression; True) := Decrement[ce];
System`CloudExpression /: Unset[ce_System`CloudExpression] /; (System`CloudExpression; True) := Unset[ce];
End[]


{System`HTTPResponse, System`HTTPRedirect, System`HTTPRequestData,
    System`$ImageFormattingWidth, System`$EvaluationCloudObject = None}


QuantityUnits`Private`$QuantityUnitsAutoloads = Hold[
    System`MixedRadixQuantity, System`QuantityUnit,
    System`QuantityMagnitude, System`IndependentUnit,
    System`UnitConvert, System`CompatibleUnitQ,
    System`CommonUnits, System`UnitDimensions,
    System`UnitSimplify, System`Quantity,
    System`KnownUnitQ, Internal`DimensionToBaseUnit, Internal`QuantityToValue,
    QuantityUnits`Private`ToQuantityBox, QuantityUnits`Private`ToQuantityString,
    QuantityUnits`ToQuantityShortString,
    System`CurrencyConvert, System`UnityDimensions,
    System`QuantityVariable, System`QuantityVariableIdentifier, System`IndependentPhysicalQuantity,
    System`QuantityVariablePhysicalQuantity, System`QuantityVariableDimensions,
    System`QuantityVariableCanonicalUnit, System`DimensionalCombinations,
    System`IncludeQuantities,
    QuantityUnits`Private`EvaluateWithQuantityArithmetic, System`NondimensionalizationTransform,
    System`GeneratedQuantityMagnitudes
]

QuantityUnits`Private`$ExtraAutoLoadSymbols = Hold[
    System`HumanGrowthData, System`FetalGrowthData
]

Begin["QuantityUnits`Private`"];
Internal`DisableQuantityUnits[]:=CompoundExpression[
    Set[Internal`$DisableQuantityUnits,True],
    Set[$AlphaBlockFlag,True],
    (* Make the autoload OwnValues of Quantity and friends vanish completely; *)
    (* Now that the format values come from elsewhere, need to nuke those as well *)
    ReleaseHold @ Map[
        Function[
            s,
            CompoundExpression[
                Unprotect[s],
                ClearAttributes[s, {ReadProtected}],
                OwnValues[s]={},
                FormatValues[s] = Select[FormatValues[s], FreeQ[#, is_Symbol /; Context[is]==="QuantityUnits`"]&]
            ],
            HoldFirst
        ],
        Join[QuantityUnits`Private`$QuantityUnitsAutoloads, QuantityUnits`Private`$ExtraAutoLoadSymbols]
    ],
    ClearAttributes[Quantity,{HoldRest,NHoldRest}],
    System`QuantityMagnitude=Identity,
    True
];
Internal`DisablePredictiveAlphaUtilities[]:=CompoundExpression[
    Set[Internal`$DisablePredictiveAlphaUtilities,True],
    True
];
End[];


Package`DeclareLoad[
    List @@ QuantityUnits`Private`$QuantityUnitsAutoloads,
    "QuantityUnitsLoader`",
    Package`HiddenImport -> True
]

EntityFramework`Private`$EVDataPacletHeads = Hold[System`AdministrativeDivisionData, System`AircraftData,
System`AirportData, System`AnatomyData,
System`BridgeData, System`BroadcastStationData, System`BuildingData, System`CometData,
System`CompanyData, System`ConstellationData, System`DamData,
System`DeepSpaceProbeData, System`EarthImpactData,
System`ExoplanetData, System`GalaxyData,
System`GeologicalPeriodData, System`HistoricalPeriodData,
System`IslandData, System`LakeData, System`LanguageData,
System`LaminaData, System`MannedSpaceMissionData,
System`MedicalTestData, System`MeteorShowerData,
System`MineralData, System`MinorPlanetData, System`MountainData,
System`MovieData, System`NebulaData,
System`NeighborhoodData, System`NuclearExplosionData,
System`NuclearReactorData, System`OceanData, System`ParkData,
System`ParticleAcceleratorData,
System`PersonData, System`PhysicalSystemData, System`PlaneCurveData,
System`PlanetData, System`PlanetaryMoonData, System`PlantData,
System`PulsarData, System`SatelliteData,
System`SolarSystemFeatureData, System`SolidData, System`SpaceCurveData,
System`SpeciesData, System`StarData, System`StarClusterData, System`SupernovaData,
System`SurfaceData, System`TropicalStormData,
System`TunnelData, System`UnderseaFeatureData,
System`UniversityData, System`VolcanoData, System`WolframLanguageData,
System`ZIPCodeData];

(*EntityFramework now handles autoloading definitions in PacletInfo.m but we need to ensure any *.mc initialized symbols are cleared of attributes also*)
EntityFramework`Private`$EntityFrameworkSystemSymbols = Join[
	Hold[
		System`AggregatedEntityClass, System`CanonicalName, System`CommonName, System`Dated,
		System`Entity, System`EntityClass, System`EntityClassList, System`EntityCopies,
		System`EntityFunction, System`EntityGroup, System`EntityInstance, System`EntityList,
		System`EntityPrefetch, System`ExtendedEntityClass, System`EntityProperties,
		System`EntityProperty, System`EntityPropertyClass, System`EntityRegister, System`EntityStore,
		System`EntityStores, System`EntityTypeName, System`EntityUnregister, System`EntityValue,
		System`FilteredEntityClass, System`CombinedEntityClass, System`FromEntity, System`RandomEntity,
		System`SampledEntityClass, System`SortedEntityClass, System`ToEntity, System`$EntityStores,
		System`Utilities`$EntityPropertyRules, Internal`AddToEntityNameCache,
		Internal`AddToInterpreterCache, Internal`CacheEntityNames, Internal`ClearEntityValueCache,
		Internal`GetFromInterpreterCache, Internal`PossibleEntityListQ, Internal`PossibleEntityPropertyListQ,
		Internal`PossibleEntityPropertyQ, Internal`PossibleEntityQ, Internal`PreloadEntityNameCache,
		Internal`$DefaultEntityStores
	],
	EntityFramework`Private`$EVDataPacletHeads
];

(* Prevent loading of EntityFramework package *)
Begin["EntityFramework`Private`"];
Internal`DisableEntityFramework[]:=CompoundExpression[
    Internal`$DisableEntityFramework = True,
    ReleaseHold @ Map[
        Function[
            s,
            CompoundExpression[
                Unprotect[s],
                ClearAttributes[s, {ReadProtected}],
                OwnValues[s]={},
                FormatValues[s] = Select[FormatValues[s], FreeQ[#, is_Symbol /; Context[is]==="EntityFramework`Private`"]&]
            ],
            HoldFirst
        ],
        Append[EntityFramework`Private`$EntityFrameworkSystemSymbols, System`EarthquakeData]
    ],
    True
];
End[];


FormulaData`Private`$FormulaDataAutoloads = Hold[
    System`FormulaData, System`FormulaLookup,
    System`RequiredPhysicalQuantities, System`ExcludedPhysicalQuantities,
    System`PlanckRadiationLaw
]


Package`DeclareLoad[
        List @@ FormulaData`Private`$FormulaDataAutoloads,
        "FormulaDataLoader`",
        Package`HiddenImport->True
    ]

Begin["FormulaData`Private`"];
Internal`DisableFormulaData[]:=CompoundExpression[
    Internal`$DisableFormulaData = True;
    ReleaseHold @ Map[
        Function[
            s,
            CompoundExpression[
                Unprotect[s],
                ClearAttributes[s, {ReadProtected}],
                OwnValues[s]={},
                FormatValues[s] = {}
            ],
            HoldFirst
        ],
        FormulaData`Private`$FormulaDataAutoloads
    ],
    True
];
End[];

Package`DeclareLoad[
        {System`InflationAdjust, System`InflationMethod},
        "InflationAdjustLoader`",
        Package`HiddenImport -> True
    ]

Package`DeclareLoad[
        {System`SendMessage},
        "OAuthLoader`",
        Package`HiddenImport -> True
    ]

Package`DeclareLoad[
        {System`Databin, System`CreateDatabin,System`DatabinAdd, System`Databins,
        System`DatabinUpload,System`CopyDatabin,System`DatabinRemove},
        "DataDropClientLoader`",
        Package`HiddenImport -> True
    ]

Package`DeclareLoad[
        {
            (* PLI *)
            System`AllowLooseGrammar
            ,
            (* Common *)
            Semantic`PLIDump`$PLIFailed,
            Semantic`PLIDump`appendMessages,
            Semantic`PLIDump`callMethod,
            Semantic`PLIDump`doCall,
            Semantic`PLIDump`extractPliUUID,
            Semantic`PLIDump`PLICompress,
            Semantic`PLIDump`PLIUncompress,
            Semantic`PLIDump`returnError,
            Semantic`PLIDump`validateResult
            ,
            (* GrammarValidation *)
            Semantic`PLIDump`ValidateGrammar
            ,
            (* Grammar *)
            System`GrammarRules
            ,
            (* Parse *)
            System`GrammarApply
            ,
            (* CloudParse *)
            Semantic`PLIDump`receiveAlphaParse,
            Semantic`PLIDump`receiveWLParse
        },
        "PLILoader`",
        Package`HiddenImport -> True
    ]

Package`DeclareLoad[
        {System`GenerateDocument, System`NotebookTemplate, NotebookTemplating`CreateTemplateNotebook,
            NotebookTemplating`ClearTemplateNotebook, NotebookTemplating`TemplateNotebookQ},
        "NotebookTemplating`",
        Package`HiddenImport -> True
    ]

(* a ChannelFramework symbol that must be declared outside the paclet to prevent autoload *)
System`$AllowExternalChannelFunctions = False;
Protect[ $AllowExternalChannelFunctions];

(*
  Start the PacletManager.
*)

Quiet[
    Needs["PacletManager`"];
    PacletManager`Package`preparePacletManager[]
]

(*
  All contexts so far are to regarded as "system" contexts and should be excluded from parallel distribution.
  Contexts loaded from init.m files below should be *included*.
  This variable is used by Parallel Tools to derive a setting for the ExcludedContexts option for
  its internal uses of Language`ExtendedFullDefinition when Parallel Tools is loaded.
*)

Parallel`Static`$SystemContexts = Contexts[]

(* This setting of the "ExcludedContexts" option is maintained for b/w compatibility for any code,
   other than Parallel Tools, that uses Language`ExtendedFullDefinition and expect this setting.  *)

SetOptions[Language`ExtendedFullDefinition,
 "ExcludedContexts" -> Complement[
   Union[StringReplace[#, "`" ~~ ___ -> ""] & /@
       Parallel`Static`$SystemContexts], {"Global"}]]


If[System`Private`origFrontEnd =!= Null,
    MathLink`RestoreFrontEnd[System`Private`origFrontEnd]
]


On[ Get::noopen]
On[ General::initg]
On[ General::initc]
Off[ Series::esss]
Off[NIntegrate::levswitchosc]
Off[Integrate::gener]

Off[Area::nmet]
Off[ArcLength::nmet]
Off[Volume::nmet]
Off[Indexed::itv]

End[]

End[]

Null


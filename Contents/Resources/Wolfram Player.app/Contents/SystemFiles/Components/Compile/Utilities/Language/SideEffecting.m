(** Copied from GeneralUtilities/Code.m
  * pending https://stash.wolfram.com/projects/PAC/repos/generalutilities/pull-requests/13/overview
  * to use the GeneralUtilities code 
  *)
BeginPackage["Compile`Utilities`Language`SideEffecting`"]

$SystemSideEffectNames

Begin["`Private`"]

(* Much faster to store the names statically.
 * $SystemSideEffectNames := $SystemSideEffectNames = $SideEffectfulFunctions
 *)
$SystemSideEffectNames := $SystemSideEffectNames = 
{"AbortKernels", "AbortProtect", "AbortScheduledTask", \
"AbsoluteCurrentValue", "AbsoluteFileName", "AbsoluteOptions", \
"AddOnHelpPath", "AddTo", "AdministrativeDivisionData", \
"AircraftData", "AirportData", "AirPressureData", \
"AirTemperatureData", "AlgebraicNumberTrace", "AlgebraicRulesData", \
"AllowGroupClose", "AllowReverseGroupClose", "AnglePath", \
"AnimationRunning", "AnimationRunTime", "AppendTo", "ARCHProcess", \
"ARIMAProcess", "ARMAProcess", "ARProcess", "AssociateTo", \
"AssociationFormat", "AstronomicalData", "AsynchronousTaskObject", \
"AsynchronousTasks", "Attributes", "AutoloadPath", \
"AutoMultiplicationSymbol", "AutoNumberFormatting", \
"AutoOpenNotebooks", "AutoOpenPalettes", "AutoRemove", \
"BackgroundTasksSettings", "BarlowProschanImportance", "Begin", \
"BeginDialogPacket", "BeginFrontEndInteractionPacket", \
"BeginPackage", "BernoulliProcess", "BinaryFormat", "BinaryRead", \
"BinaryReadList", "BinaryWrite", "BinomialProcess", \
"BirnbaumImportance", "BitClear", "BitGet", "BitSet", "BoxData", \
"BoxFormFormatTypes", "BridgeData", "BroadcastStationData", \
"BrownianBridgeProcess", "BuildingData", "ButtonData", \
"ButtonNotebook", "CalendarData", "CatchStack", "CDFInformation", \
"CelestialSystem", "CellDynamicExpression", "CellGroupData", \
"CellOpen", "CellPrint", "CharacterEncodingsPath", \
"ChartElementData", "ChartElementDataFunction", "ChemicalData", \
"ChoiceDialog", "CityData", "ClassifierInformation", "Clear", \
"ClearAll", "ClearAttributes", "ClearCompiled", "ClearPermissions", \
"ClearSystemCache", "ClipboardNotebook", "Close", "Closed", \
"CloseKernels", "ClosenessCentrality", "ClosingAutoSave", \
"CloudAccountData", "CloudBase", "CloudConnect", "CloudDeploy", \
"CloudDirectory", "CloudDisconnect", "CloudEvaluate", "CloudExport", \
"CloudFunction", "CloudGet", "CloudImport", "CloudLoggingData", \
"CloudObject", "CloudObjectInformation", \
"CloudObjectInformationData", "CloudObjects", "CloudPublish", \
"CloudPut", "CloudSave", "CloudShare", "CloudSubmit", "CloudSymbol", \
"ColorData", "ColorDataFunction", "ColorProfileData", \
"ColorSelectorSettings", "ColorSetter", "ColorSetterBox", \
"ColorSetterBoxOptions", "ColorToneMapping", "CometData", \
"CommonDefaultFormatTypes", "CompanyData", "Compile", "Compiled", \
"CompiledFunction", "CompileEvaluateBody", "CompileExpr", \
"CompilerException", "CompilerFunctions", "CompoundPoissonProcess", \
"CompoundRenewalProcess", "Compress", "CompressedData", \
"ConfigurationPath", "ConnectLibraryCallbackFunction", \
"ConsolePrint", "ConstellationData", "ContextToFileName", \
"ContinuousMarkovProcess", "ContinuousTask", "ContinuousWaveletData", \
"ControllerInformation", "ControllerInformationData", \
"ControllerLinking", "ControllerPath", "ConvertToBitmapPacket", \
"ConvertToPostScript", "ConvertToPostScriptPacket", \
"CoordinateChartData", "CoordinatesToolOptions", \
"CoordinateTransformData", "CopyDatabin", "CopyDirectory", \
"CopyFile", "CopyToClipboard", "CountryData", \
"CoxIngersollRossProcess", "CreateArchive", "CreateBasicBlock", \
"CreateBasicBlockPass", "CreateBinaryInstruction", "CreateBitVector", \
"CreateBoxIcon", "CreateBranchInstruction", \
"CreateBuilderMExprVisitor", "CreateCallInstruction", "CreateTypeCast", \
"CreateTypeCastInstruction", "CreateCellID", "CreateCFunction", \
"CreateCompareInstruction", "CreateCompiledFunction", \
"CreateCompiledFunctionModulePass", "CreateConstantValue", \
"CreateCStringFunctionModulePass", "CreateDatabin", "CreateDialog", \
"CreateDirectory", "CreateDocument", "CreateFunctionKind", \
"CreateFunctionModule", "CreateFunctionModulePass", \
"CreateGenericTypeVariable", "CreateGetElementInstruction", \
"CreateGetFieldInstruction", "CreateGlobalValue", \
"CreateInertInstruction", "CreateInstructionInformation", \
"CreateInstructionVisitor", "CreateIntermediateDirectories", \
"CreateIR", "CreateLabelInstruction", "CreateLabelKind", \
"CreateLambdaInstruction", "CreateLinkedTypeVariable", \
"CreateLoadArgumentInstruction", "CreateLoadGlobalInstruction", \
"CreateLoadInstruction", "CreateCopyInstruction", "CreateLogger",
"CreateLoopInformation", "CreateManagedLibraryExpression", "CreateMetaInformation", \
"CreateMetaTypeInformation", "CreateMExpr", "CreateMExprLiteral", \
"CreateMExprNormal", "CreateMExprState", "CreateMExprSymbol", \
"CreateNewRecordInstruction", "CreateNotebook", "CreateNullaryKind", \
"CreateObject",  "CreatePaclet", \
"CreatePalette", "CreatePalettePacket", "CreatePassInformation", \
"CreatePermissionsGroup", "CreatePhiInstruction", \
"CreatePointerGraph", "CreateProgramModule", \
"CreateProgramModulePass", "CreateRecordExtendInstruction", \
"CreateRecordRestrictInstruction", "CreateRecordSelectInstruction", \
"CreateRecursiveMExprVisitor", "CreateReference", "CreateRegister", \
"CreateReturnInstruction", "CreateRowKind", "CreateScheduledTask", \
"CreateSearchIndex", "CreateSelectInstruction", \
"CreateSetElementInstruction", "CreateSetFieldInstruction", \
"CreateSourceLocation", "CreateSourceSpan", \
"CreateStackAllocateInstruction", "CreateStoreInstruction", \
"CreateSymbolicCFunctionModulePass", "CreateTemporary", "CreateType", \
"CreateTypeAlias", "CreateTypeApplication", "CreateTypeArrow", \
"CreateTypeConstructor", "CreateTypeDeclaration", \
"CreateTypeEnvironment", "CreateTypeName", "CreateTypePredicate", \
"CreateTypePredicated", "CreateTypeInferenceState", "CreateUnaryInstruction", \
"CreateUnboundTypeVariable", "CreateUnknownKind", \
"CreateUnreachableInstruction", "CreateUUID", "CreateVariable", \
"CreateWindow", "CriticalityFailureImportance", \
"CriticalitySuccessImportance", "CurrentImage", "CurrentValue", \
"CurveClosed", "DamData", "Databin", "DatabinAdd", "DatabinRemove", \
"Databins", "DatabinUpload", "DataCompression", "DataDistribution", \
"DataRange", "DataReversed", "Dataset", "DateFormat", \
"DateTicksFormat", "DeBrujinIndexWriter", "DeclareKnownSymbols", \
"DeepSpaceProbeData", "DefaultFormatType", \
"DefaultFormatTypeForStyle", "DefaultInlineFormatType", \
"DefaultInputFormatType", "DefaultNotebook", \
"DefaultOutputFormatType", "DefaultStyleDefinitions", \
"DefaultTextFormatType", "DefaultTextInlineFormatType", \
"DefaultTooltipStyle", "DefaultValues", "DefineInputStreamMethod", \
"DefineOutputStreamMethod", "Definition", "DeleteDirectory", \
"DeleteFile", "DeviceClose", "DeviceConfigure", "DeviceExecute", \
"DeviceExecuteAsynchronous", "DeviceObject", "DeviceOpen", \
"DeviceOpenQ", "DeviceRead", "DeviceReadBuffer", "DeviceReadLatest", \
"DeviceReadList", "DeviceReadTimeSeries", "Devices", "DeviceStreams", \
"DeviceWrite", "DeviceWriteBuffer", "Dialog", "DialogIndent", \
"DialogInput", "DialogLevel", "DialogNotebook", "DialogProlog", \
"DialogReturn", "DialogSymbols", "Directory", "DirectoryName", \
"DirectoryQ", "DirectoryStack", "DisableConsolePrintPacket", \
"DiscreteMarkovProcess", "DiscreteWaveletData", "DisplayEndPacket", \
"DisplaySetSizePacket", "DistributeDefinitions", \
"DocumentGeneratorInformation", "DocumentGeneratorInformationData", \
"DocumentNotebook", "DOSTextFormat", "DownValues", \
"DualSystemsModel", "DumpGet", "DumpSave", "Dynamic", "DynamicBox", \
"DynamicBoxOptions", "DynamicEvaluationTimeout", \
"DynamicGeoGraphics", "DynamicImage", "DynamicLocation", \
"DynamicModule", "DynamicModuleBox", "DynamicModuleBoxOptions", \
"DynamicModuleParent", "DynamicModuleValues", "DynamicName", \
"DynamicNamespace", "DynamicReference", "DynamicSetting", \
"DynamicUpdating", "DynamicWrapper", "DynamicWrapperBox", \
"DynamicWrapperBoxOptions", "EarthImpactData", "EarthquakeData", \
"EditButtonSettings", "EditCellTagsSettings", "ElementData", \
"EnableConsolePrintPacket", "Encode", "End", "EndAdd", \
"EndDialogPacket", "EndFrontEndInteractionPacket", "EndOfBuffer", \
"EndOfFile", "EndOfLine", "EndOfString", "EndPackage", "EntityValue", \
"EquatedTo", "ErrorsDialogSettings", "EstimatedProcess", \
"EvaluateScheduledTask", "EvaluationData", "EvaluationNotebook", \
"EventData", "ExampleData", "ExitDialog", "ExoplanetData", \
"ExpandFileName", "Export", "ExportAutoReplacements", "ExportForm", \
"ExportPacket", "ExportString", "ExpToTrig", \
"ExternalDataCharacterEncoding", "FARIMAProcess", \
"FEDisableConsolePrintPacket", "FEEnableConsolePrintPacket", \
"FetalGrowthData", "File", "FileBaseName", "FileByteCount", \
"FileDate", "FileExistsQ", "FileExtension", "FileFormat", "FileHash", \
"FileInformation", "FileName", "FileNameDepth", \
"FileNameDialogSettings", "FileNameDrop", "FileNameJoin", \
"FileNames", "FileNameSetter", "FileNameSplit", "FileNameTake", \
"FilePrint", "FileTemplate", "FileTemplateApply", "FileType", \
"FinancialData", "Find", "FindArgMax", "FindArgMin", "FindClique", \
"FindClusters", "FindCurvePath", "FindCycle", "FindDevices", \
"FindDistribution", "FindDistributionParameters", "FindDivisions", \
"FindEdgeCover", "FindEdgeCut", "FindEdgeIndependentPaths", \
"FindEulerianCycle", "FindFaces", "FindFile", "FindFit", \
"FindFormula", "FindFundamentalCycles", "FindGeneratingFunction", \
"FindGeoLocation", "FindGeometricTransform", "FindGraphCommunities", \
"FindGraphIsomorphism", "FindGraphPartition", "FindHamiltonianCycle", \
"FindHamiltonianPath", "FindHiddenMarkovStates", \
"FindIndependentEdgeSet", "FindIndependentVertexSet", "FindInstance", \
"FindIntegerNullVector", "FindKClan", "FindKClique", "FindKClub", \
"FindKPlex", "FindLibrary", "FindLinearRecurrence", "FindList", \
"FindMaximum", "FindMaximumFlow", "FindMaxValue", "FindMExpr", \
"FindMinimum", "FindMinimumCostFlow", "FindMinimumCut", \
"FindMinValue", "FindPath", "FindPeaks", "FindPermutation", \
"FindPostmanTour", "FindProcessParameters", "FindRoot", \
"FindSequenceFunction", "FindSettings", "FindShortestPath", \
"FindShortestTour", "FindSpanningTree", "FindThreshold", \
"FindVertexCover", "FindVertexCut", "FindVertexIndependentPaths", \
"FinishDynamic", "FiniteGroupData", "FlatTopWindow", \
"FlushPrintOutputPacket", "Format", "FormatName", "FormatRules", \
"FormatType", "FormatTypeAutoConvert", "FormatValues", "FormulaData", \
"FractionalBrownianMotionProcess", "FractionalGaussianNoiseProcess", \
"FrenetSerretSystem", "FrontEndDynamicExpression", \
"FrontEndEventActions", "FrontEndExecute", "FrontEndObject", \
"FrontEndResource", "FrontEndResourceString", "FrontEndStackSize", \
"FrontEndToken", "FrontEndTokenExecute", "FrontEndValueCache", \
"FrontEndVersion", "FullDefinition", \
"FullInformationOutputRegulator", "FullOptions", \
"FussellVeselyImportance", "GalaxyData", "GARCHProcess", \
"GenerateTestData", "GenomeData", "GeodesicOpening", "GeodesyData", \
"GeoElevationData", "GeogravityModelData", "GeologicalPeriodData", \
"GeomagneticModelData", "GeometricBrownianMotionProcess", "GeoPath", \
"GeoProjectionData", "Get", "GetBoundingBoxSizePacket", \
"GetCompilerMathFunction", "GetContext", \
"GetDefinitionFromDownValues", "GetElement", "GetElementInstruction", \
"GetElementInstructionClass", "GetElementInstructionQ", \
"GetEnvironment", "GetFieldInstruction", "GetFieldInstructionClass", \
"GetFieldInstructionQ", "GetFileName", \
"GetFrontEndOptionsDataPacket", "GetLinebreakInformationPacket", \
"GetMenusPacket", "GetPageBreakInformationPacket",  \
"GetSink", "GetSource", "GetVirtualCellGroup", "GrammarToken", \
"GraphData", "GraphElementData", "GraphicsData", \
"GraphLinkEfficiency", "GridCreationSettings", "GroupElementToWord", \
"GroupSetwiseStabilizer", "GroupTogetherGrouping", \
"GroupTogetherNestedGrouping", "HelpBrowserNotebook", \
"HelpBrowserSettings", "HiddenMarkovProcess", "HistoricalPeriodData", \
"HomeDirectory", "HTMLSave", "HTTPHandler", "HTTPRedirect", \
"HTTPRequestData", "HTTPResponse", "HumanGrowthData", \
"HyperlinkCreationSettings", "HypothesisTestData", "IconData", \
"ImageData", "ImageDataPacket", "ImageFileApply", "ImageFileFilter", \
"ImageFileScan", "ImageFormattingWidth", "ImagingDevice", "Import", \
"ImportAutoReplacements", "ImportOptions", "ImportString", \
"ImprovementImportance", "IncludeFileExtension", \
"IncludeGeneratorTasks", "IndependentEdgeSetQ", \
"IndependentVertexSetQ", "Information", \
"InhomogeneousPoissonProcess", "InputNotebook", "InputSettings", \
"InputStream", "InputToBoxFormPacket", "Install", "InstallNET", \
"InstallR", "InstallService", "InstructionInformation", \
"InstructionInformationClass", "InstructionInformationQ", \
"InterruptSettings", "IslandData", "IsotopeData", "ItoProcess", \
"JacobiSymbol", "JuliaSetBoettcher", "JuliaSetIterationCount", \
"JuliaSetPlot", "JuliaSetPoints", "Kernels", "KillProcess", \
"KnightTourGraph", "KnotData", "KroneckerSymbol", "LakeData", \
"LaminaData", "LanguageData", "LatticeData", "LaunchKernels", \
"LayoutInformation", "LibraryCallbackFunctionID", "LibraryDataType", \
"LibraryFunction", "LibraryFunctionError", \
"LibraryFunctionInformation", "LibraryFunctionLoad", \
"LibraryFunctionUnload", "LibraryLoad", "LibraryUnload", \
"LiftingFilterData", "LimitsPositioningTokens", \
"LinearizingTransformationData", "LinkActivate", "LinkClose", \
"LinkConnect", "LinkConnectedQ", "LinkCreate", "LinkedTypeVariableQ", \
"LinkError", "LinkFlush", "LinkFunction", "LinkHost", \
"LinkInterrupt", "LinkLaunch", "LinkMode", "LinkObject", "LinkOpen", \
"LinkOptions", "LinkPatterns", "LinkProtocol", "LinkRankCentrality", \
"LinkRead", "LinkReadHeld", "LinkReadyQ", "Links", "LinkService", \
"LinkWrite", "LinkWriteHeld", "ListCurvePathPlot", "ListFormat", \
"ListStreamDensityPlot", "ListStreamPlot", "LoadClass", \
"LoadNETAssembly", "LoadNETType", "LocalizeDefinitions", \
"LocalSymbol", "LocatorAutoCreate", "LoopInformation", \
"LoopInformationQ", "MacintoshSystemPageSetup", \
"MaintainDynamicCaches", "MakeBoxes", "MakeExpression", \
"ManagedLibraryExpressionID", "ManagedLibraryExpressionQ", \
"MandelbrotSetBoettcher", "MandelbrotSetDistance", \
"MandelbrotSetIterationCount", "MandelbrotSetMemberQ", \
"MandelbrotSetPlot", "MannedSpaceMissionData", "MAProcess", \
"MarkovProcessProperties", "MaxMixtureKernels", "MedicalTestData", \
"MessageDialog", "MessagesNotebook", "MetaInformation", \
"MetaInformationQ", "MetaTypeInformation", "MetaTypeInformationQ", \
"MeteorShowerData", "MExprSymbol", "MExprSymbolClass", \
"MExprSymbolQ", "MineralData", "MinorPlanetData", \
"MissingDataMethod", "MissingDataRules", "MountainData", "MovieData", \
"NebulaData", "NeedCurrentFrontEndPackagePacket", \
"NeedCurrentFrontEndSymbolsPacket", "Needs", "NeighborhoodData", \
"NETNew", "Notebook", "NotebookApply", "NotebookAutoSave", \
"NotebookClose", "NotebookConvertSettings", "NotebookCreate", \
"NotebookCreateReturnObject", "NotebookDefault", "NotebookDelete", \
"NotebookDirectory", "NotebookDynamicExpression", "NotebookEvaluate", \
"NotebookEventActions", "NotebookFileName", "NotebookFind", \
"NotebookFindReturnObject", "NotebookGet", \
"NotebookGetLayoutInformationPacket", \
"NotebookGetMisspellingsPacket", "NotebookImport", \
"NotebookInformation", "NotebookInterfaceObject", "NotebookLocate", \
"NotebookObject", "NotebookOpen", "NotebookOpenReturnObject", \
"NotebookPath", "NotebookPrint", "NotebookPut", \
"NotebookPutReturnObject", "NotebookRead", \
"NotebookResetGeneratedCells", "Notebooks", "NotebookSave", \
"NotebookSaveAs", "NotebookSelection", \
"NotebookSetupLayoutInformationPacket", "NotebooksMenu", \
"NotebookTemplate", "NotebookWrite", "NuclearExplosionData", \
"NuclearReactorData", "NumberFormat", "NValues", "OceanData", \
"OLEData", "Open", "OpenAppend", "Opener", "OpenerBox", \
"OpenerBoxOptions", "OpenerView", "OpenFunctionInspectorPacket", \
"Opening", "OpenRead", "OpenSpecialOptions", "OpenTemporary", \
"OpenWrite", "OperatingSystem", "OptimumFlowData", \
"OptionInspectorSettings", "Options", "OrnsteinUhlenbeckProcess", \
"OutputFormData", "OutputStream", "OwnValues", "PackPaclet", \
"Paclet", "PacletCheckUpdate", "PacletDirectoryAdd", \
"PacletDirectoryRemove", "PacletDisable", "PacletEnable", \
"PacletFind", "PacletFindRemote", "PacletInformation", \
"PacletInstall", "PacletInstallQueued", "PacletManager", \
"PacletManagerEnabled", "PacletNewerQ", "PacletResource", \
"PacletResources", "PacletSetLoading", "PacletSite", "PacletSiteAdd", \
"PacletSiteRemove", "PacletSites", "PacletSiteUpdate", \
"PacletUninstall", "PacletUpdate", "PaletteNotebook", "PalettePath", \
"ParallelArray", "ParallelCombine", "ParallelDo", "Parallelepiped", \
"ParallelEvaluate", "Parallelization", "Parallelize", "ParallelMap", \
"ParallelNeeds", "Parallelogram", "ParallelProduct", \
"ParallelSubmit", "ParallelSum", "ParallelTable", "ParallelTry", \
"ParentDirectory", "ParentNotebook", "ParkData", \
"ParticleAcceleratorData", "ParticleData", "PassInformation", \
"PassInformationClass", "PassInformationQ", "Paste", \
"PasteBoxFormInlineCells", "PasteButton", "Path", "PathGraph", \
"PathGraphQ", "PersonData", "PhysicalSystemData", "PIDData", \
"PillaiTrace", "PillaiTraceTest", "PlaneCurveData", \
"PlanetaryMoonData", "PlanetData", "PlantData", "PoissonProcess", \
"PolyhedronData", "PostTopologicalOrderRenumberPass", \
"PredictorInformation", "PreemptProtect", "PreferencesPath", \
"PrependCompileValue", "PrependTo", "Print", "PrintableASCIIQ", \
"PrintAction", "PrintForm", "PrintingCopies", "PrintingOptions", \
"PrintingPageRange", "PrintingStartingPageNumber", \
"PrintingStyleEnvironment", "PrintPrecision", "PrintTemporary", \
"PrivateFrontEndOptions", "PrivateNotebookOptions", "PrivatePaths", \
"ProcessConnection", "ProcessDirectory", "ProcessEnvironment", \
"Processes", "ProcessEstimator", "ProcessInformation", \
"ProcessObject", "ProcessParameterAssumptions", "ProcessParameterQ", \
"ProcessStateDomain", "ProcessStatus", "ProcessTimeDomain", \
"Protect", "Protected", "ProteinData", "PulsarData", "Put", \
"PutAppend", "QueueingNetworkProcess", "QueueingProcess", "Quit", \
"RandomWalkProcess", "RawBoxes", "RawData", "ReachingDefinitionPass", \
"Read", "ReadLine", "ReadList", "ReadMExpr", "ReadProtected", \
"ReadString", "RebuildPacletData", "RegisteredTestDataGeneratorQ", \
"RegisterTestDataGenerator", "Reinstall", "ReinstallNET", "Remove", \
"RemoveAlphaChannel", "RemoveAsynchronousTask", "RemoveBackground", \
"Removed", "RemoveDiacritics", "RemoveInputStreamMethod", \
"RemoveOutputStreamMethod", "RemoveProperty", \
"RemoveRedundantStackAllocatePass", "RemoveScheduledTask", \
"RemoveUsers", "RenameDirectory", "RenameFile", "RenewalProcess", \
"ResamplingAlgorithmData", "ResetDirectory", "ResetScheduledTask", \
"RestartPacletManager", "REvaluate", "RFunction", \
"RiskAchievementImportance", "RiskReductionImportance", \
"RLinkResourcesInstall", "Run", "RunPass", "RunPasses", "RunProcess", \
"RunScheduledTask", "RunThrough", "RuntimeAttributes", \
"RuntimeOptions", "SARIMAProcess", "SARMAProcess", "SatelliteData", \
"Save", "Saveable", "SaveAutoDelete", "SaveDefinitions", \
"ScheduledTask", "ScheduledTaskInformation", \
"ScheduledTaskInformationData", "ScheduledTaskObject", \
"ScheduledTasks", "SelectedNotebook", "SelectionCellCreateCell", \
"SelectionCreateCell", "SelectionEvaluate", \
"SelectionEvaluateCreateCell", "SelectionSetStyle", "SemanticImport", \
"SemanticImportString", "SendMail", "SendMessage", "SeriesData", \
"Set", "SetAccuracy", "SetAlphaChannel", "SetAttributes", "Setbacks", \
"SetBoxFormNamesPacket", "SetCloudDirectory", "SetCompiled", \
"SetData", "SetDelayed", "SetDirectory", "SetElementInstruction", \
"SetElementInstructionClass", "SetElementInstructionQ", \
"SetEnvironment", "SetEvaluationNotebook", "SetFieldInstruction", \
"SetFieldInstructionClass", "SetFieldInstructionQ", "SetFileDate", \
"SetFileLoadingContext", "SetInternetProxy", "SetIterate", \
"SetNotebookStatusLine", "SetOptions", "SetOptionsPacket", \
"SetPermissions", "SetPrecision", "SetProperty", \
"SetSelectedNotebook", "SetSharedFunction", "SetSharedVariable", \
"SetSpeechParametersPacket", "SetStreamPosition", "SetSystemOptions", \
"Setter", "SetterBar", "SetterBox", "SetterBoxOptions", "Setting", \
"SetUsers", "SetValue", "ShortestPathFunction", "ShowClosedCellArea", \
"ShowGroupOpenCloseIcon", "ShowGroupOpener", "SixJSymbol", "Skip", \
"SocialMediaData", "SolarSystemFeatureData", "SolidData", \
"SpaceCurveData", "SpeciesData", "SpellingDictionariesPath", \
"Splice", "SplicedDistribution", "SplineClosed", "Stack", \
"StackAllocateInstruction", "StackAllocateInstructionClass", \
"StackAllocateInstructionQ", "StackBegin", "StackCaught", \
"StackComplete", "StackInhibit", "StandardAtmosphereData", \
"StarClusterData", "StarData", "StartAsynchronousTask", \
"StartProcess", "StartScheduledTask", "StopAsynchronousTask", \
"StoppingPowerData", "StopScheduledTask", "StratonovichProcess", \
"StreamColorFunction", "StreamColorFunctionScaling", \
"StreamDensityPlot", "StreamPlot", "StreamPoints", "StreamPosition", \
"Streams", "StreamScale", "StreamStyle", "StringEndsQ", \
"StringFormat", "StringToStream", "StructuralImportance", \
"StyleData", "StyleDefinitions", "StyleNameDialogSettings", \
"StylePrint", "StyleSheetPath", "SubstitutionSystem", "SubValues", \
"SupernovaData", "SurfaceData", "Symbol", "SymbolName", "SymbolQ", \
"SymbolValueHead", "SyntaxInformation", "SystemDialogInput", \
"SystemException", "SystemGet", "SystemHelpPath", \
"SystemInformation", "SystemInformationData", "SystemOpen", \
"SystemOptions", "SystemsModelDelay", "SystemsModelDelayApproximate", \
"SystemsModelDelete", "SystemsModelDimensions", \
"SystemsModelExtract", "SystemsModelFeedbackConnect", \
"SystemsModelLabels", "SystemsModelLinearity", "SystemsModelMerge", \
"SystemsModelOrder", "SystemsModelParallelConnect", \
"SystemsModelSeriesConnect", "SystemsModelStateFeedbackConnect", \
"SystemsModelVectorRelativeOrders", "SystemStub", "TagSet", \
"TagSetDelayed", "TelegraphProcess", "TemporalData", "TestData", \
"TestDataQ", "TeXSave", "TextClipboardType", "TextData", \
"ThermodynamicData", "ThisLink", "ThreeJSymbol", "ThrowingTodo", \
"ThrowStackException", "ThrowWithStack", "TimeFormat", "ToBoxes", \
"ToCharacterCode", "ToColor", "ToContinuousTimeModel", "ToDate", \
"Today", "ToDiscreteTimeModel", "ToEntity", "ToeplitzMatrix", \
"ToExpression", "ToFileName", "Together", "Toggle", "ToggleFalse", \
"Toggler", "TogglerBar", "TogglerBox", "TogglerBoxOptions", \
"ToGraph", "ToHeldExpression", "ToInvertibleTimeSeries", \
"TokenWords", "Tolerance", "ToLowerCase", "Tomorrow", \
"ToNumberField", "TooBig", "Tooltip", "TooltipBox", \
"TooltipBoxOptions", "TooltipDelay", "TooltipStyle", "Top", \
"TopHatTransform", "ToPolarCoordinates", \
"TopologicalOrderRenumberPass", "TopologicalSort", "ToRadicals", \
"ToRules", "ToSphericalCoordinates", "ToString", "Total", \
"TotalHeight", "TotalVariationFilter", "TotalWidth", "TouchPosition", \
"TouchscreenAutoZoom", "TouchscreenControlPlacement", "ToUpperCase", \
"Trace", "TraceAbove", "TraceAction", "TraceBackward", "TraceDepth", \
"TraceDialog", "TraceForward", "TraceInternal", "TraceLevel", \
"TraceOff", "TraceOn", "TraceOriginal", "TracePrint", "TraceScan", \
"TrackedSymbols", "TransformedProcess", "TrigToExp", \
"TropicalStormData", "TunnelData", "TypeInformation", "Uncompress", \
"UnderseaFeatureData", "UnitSystem", "UniversityData", \
"UnpackPaclet", "Unprotect", "UpdateDynamicObjects", \
"UpdateDynamicObjectsSynchronous", "UpSet", "UpSetDelayed", \
"UpValues", "URL", "URLBuild", "URLDecode", "URLDispatcher", \
"URLEncode", "URLExecute", "URLExistsQ", "URLExpand", "URLFetch", \
"URLFetchAsynchronous", "URLParse", "URLQueryDecode", \
"URLQueryEncode", "URLSave", "URLSaveAsynchronous", "URLShorten", \
"UsingFrontEnd", "V2Get", "ValidationSet", "ValuesData", \
"VectorGlyphData", "VerboseConvertToPostScriptPacket", \
"VerifyPaclet", "VertexDataCoordinates", "ViewPointSelectorSettings", \
"VirtualGroupData", "VolcanoData", "WaitAsynchronousTask", \
"WeatherData", "WeightedData", "WhiteNoiseProcess", "WienerProcess", \
"WikipediaData", "WindDirectionData", "WindowToolbars", \
"WindSpeedData", "WindVectorData", "WolframLanguageData", \
"WordCloud", "WordData", "Write", "WriteLine", "WriteString", \
"ZIPCodeData", "ZipGetFile", "$AddOnsDirectory", "$AllowDataUpdates", \
"$AsynchronousTask", "$AtomicSymbols", "$BaseDirectory", \
"$BasePacletsDirectory", "$CloudBase", "$CloudConnected", \
"$CloudCreditsAvailable", "$CloudEvaluation", "$CloudRootDirectory", \
"$CloudSymbolBase", "$ConfiguredKernels", "$ContextPath", \
"$ControlActiveSetting", "$CurrentLink", "$DateStringFormat", \
"$DefaultFrontEnd", "$DefaultImagingDevice", "$DefaultPath", \
"$DynamicEvaluation", "$EvaluationCloudObject", "$ExportFormats", \
"$FinancialDataSource", "$FormatingGraphicsOptions", "$FormatType", \
"$FrontEnd", "$FrontEndSession", "$HomeDirectory", \
"$HTMLExportRules", "$HTTPCookies", "$HTTPRequestData", \
"$ImageFormattingWidth", "$ImagingDevice", "$ImagingDevices", \
"$ImportFormats", "$InitialDirectory", "$InputFileName", \
"$InputStreamMethods", "$InstallationDate", "$InstallationDirectory", \
"$LaunchDirectory", "$LibraryPath", "$LicenseProcesses", "$Linked", \
"$LinkSupported", "$LoadedFiles", "$LocalSymbolBase", \
"$MaxLicenseProcesses", "$MessagePrePrint", "$NewSymbol", \
"$Notebooks", "$NumPassesRun", "$OperatingSystem", \
"$OutputStreamMethods", "$PacletSite", "$ParentLink", \
"$ParentProcessID", "$PasswordFile", "$Path", "$PathnameSeparator", \
"$PreferencesDirectory", "$PrePrint", "$PreRead", "$PrintForms", \
"$PrintLiteral", "$ProcessID", "$ProcessorCount", "$ProcessorType", \
"$ProductInformation", "$RegisteredDeviceClasses", \
"$RegisteredTestDataGenerator", "$RootDirectory", "$ScheduledTask", \
"$SetParentLink", "$System", "$SystemAttributes", \
"$SystemCharacterEncoding", "$SystemID", "$SystemMemory", \
"$SystemReal", "$SystemRealBitLength", "$SystemShell", \
"$SystemShortNames", "$SystemSideEffectNames", "$SystemWordLength", \
"$TemplatePath", "$TemporaryDirectory", "$ThrowStackException", \
"$TopDirectory", "$TraceOff", "$TraceOn", "$TracePattern", \
"$TracePostAction", "$TracePreAction", "$UnitSystem", \
"$UserAddOnsDirectory", "$UserAgentOperatingSystem", \
"$UserBaseDirectory", "$UserBasePacletsDirectory", \
"$UserDocumentsDirectory"}

(*$MiscFunctions is compiled from StackExchage:
http://mathematica.stackexchange.com/questions/29364/list-of-dangerous-functions

The list is from:
FileNameJoin[{$InstallationDirectory, "SystemFiles", "FrontEnd", "TextResources", "MiscExpressions.tr"}]

as well as some collected from one of the user posts. 

The full list is larger, but overlaps with Tali's list below.  These are the remainder.  

*)

$MiscFunctions = {"InstallNET", "InstallR", "InstallService", "LoadClass", "LoadNETAssembly", "LoadNETType", "NETNew", "Options", "Reinstall", "ReinstallNET", "REvaluate", "RFunction", "RLinkResourcesInstall"};

allSymbols := allSymbols = Names["*"];
fastNames[pats0_] :=
  With[{pats = StringExpression[BlankNullSequence[], #, BlankNullSequence[]] & /@ pats0},
    Flatten[StringCases[allSymbols, pats]]
  ]

$SideEffectCategories := $SideEffectCategories = 
Map[fastNames, Association[
  "File" -> {"File", "Directory", "Path", "FindList"},
  "IO" -> {"Print", "Put", "Get", "Needs", "Dump", "Save", "Import", "Export", "Splice", "Encode"},
  "Stream" -> {"Stream", "Open", "Close", "Read", "Write", "Find", "Skip"},
  "Device" -> {"Device"},
  "Paclet" -> {"Paclet"},
  "Network" -> {"URL","HTTP"},
  "System" -> {"System"},
  "Dynamic" -> {"Dynamic"},
  "Data" -> {"Data", "EntityValue"},
  "Frontend" -> {"Notebook", "Frontend", "FrontEnd", "Clipboard", "CurrentImage", "CurrentValue", "AbsoluteOptions", "FullOptions", "SelectionEvaluate", "Paste"},
  "Mutate" -> {"Set", "Get", "To", "Protect", "Unprotect", "Clear", "Remove"},
  "Context" -> {"Begin", "BeginPackage", "End", "EndPackage"},
  "Symbol" -> {(* "Context", "$Context", "$ContextPath", "Name",*) "Symbol"},
  "Introspect" -> {
    "ToExpression", "MakeExpression", 
    "Attributes", "Stack", "Definition", "FullDefinition", "Information", 
    "DefaultValues", "DownValues", "DynamicModuleValues", "FormatValues", "NValues", "OwnValues", "SubValues", "UpValues", 
    "Stack", "Trace"},
  "Format" -> {"Format", "MakeBoxes", "ToBoxes", "RawBoxes", "Typeset`ToExpression"},
  "Compress" -> {"Compress", "Uncompress", "CompressedData"},
  "Compile" -> {"Compile"},
  "Print" -> {"Print"},
  "Process" -> {"Run", "RunThrough", "Process", "Install"},
  "Java" -> {"Java", "AddToClassPath"},
  "Task" -> {"Task"},
  "Create" -> {"Create"},
  "Link" -> {"Link", "Java"},
  "Send" -> {"Send"},
  "Dialog" -> {"Dialog"},
  "LibraryLink" -> {"Library"},
  "Parallel" -> {"Parallel", "Kernels"},
  "Cloud" -> {"Cloud"},
  "Quit" -> {"Quit"}
]]; 

$SideEffectfulFunctions := $SideEffectfulFunctions = Union[Flatten[{Values @ $SideEffectCategories, $MiscFunctions}]];

$SideEffectRules :=
	Reverse /@ Flatten[
		Thread /@ Normal[
			$SideEffectCategories]];


SetAttributes[SideEffectSymbolQ, HoldAllComplete]

SideEffectSymbolQ[s_Symbol] := 
	ValueQ[s] || MemberQ[$SideEffectfulFunctions, SymbolName[Unevaluated[s]]];

SideEffectSymbolQ[s_String] := 
	MemberQ[$SideEffectfulFunctions, s] || ToExpression[s, InputForm, ValueQ] (* TODO: FIX *);

SideEffectSymbolQ[_] := False;


End[]

EndPackage[]
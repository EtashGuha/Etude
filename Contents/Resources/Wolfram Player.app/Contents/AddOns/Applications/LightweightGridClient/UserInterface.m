(* :Copyright: 2010 by Wolfram Research, Inc. *)
BeginPackage["LightweightGridClient`UserInterface`"]

`configEditor;

Begin["`Private`"];

Needs["LightweightGridClient`"];
Needs["LightweightGridClient`ParallelConfiguration`"];


Needs["ResourceLocator`"]

$packageRoot = DirectoryName[System`Private`$InputFileName]
TR = TextResourceLoad["RemoteServices", $packageRoot]



(* Remote Services configuration tab content *)
configEditor[] := 
	agentRefresherView[configEditorView, 
		pruneAgents[discoveredAgents[]]; kernelAgents[], $UpdateFrequency];

(* Basic layout of Remote Services configuration tab content *)
configEditorView[urls_] := 
	Panel[
		Grid[{
			{buttonRow[], SpanFromLeft},
			{" ", SpanFromLeft},
			{remoteComputersTablePane[urls], detailInspector[urls]}
			},
			Spacings -> 0, Alignment -> {{Left, Right}, Top}],
		ImageSize -> {Scaled[1.0], All}, Appearance -> "Frameless"];

$buttonRowWidth = Scaled[0.99];
$remoteComputersTableWidth = Scaled[0.54];
$detailInspectorWidth = Scaled[0.45];
$computerSetterWidth = 150;
$launchSettingsInputFieldWidth = {100, Automatic};
$launchSettingsInputFieldSize = {70, 1};

(******************************************************************************)
(* Button row *)

buttonRow[] := Panel[buttonRowContent[], ImageSize -> {$buttonRowWidth, All}, 
	FrameMargins -> If[$OperatingSystem === "MacOSX", Automatic, 0]];

buttonRowContent[] := nowrap@Row[{
	bigButton[discoverMoreKernelsLabel, CreateNetworkStartingPointsDialog[]],
	bigButton[hideShowEnabledLabel, toggleHideShowEnabled[]],
	bigButton[setAllLabel, CreateSetAllDialog[]]
	}, Spacer[12]];

discoverMoreKernelsLabelText = TR["RS.DiscoverMoreKernels"];
discoverMoreKernelsLabel = discoverMoreKernelsLabelText;

hideShowEnabledLabel = 
  Dynamic[If[$ViewRemoved, TR["RS.Hide"], TR["RS.Show"]] <> TR["RS.Disabled"]];

setAllLabel = TR["RS.SetAll"];

SetAttributes[bigButton, HoldAll];

bigButton[label_, action_] := 
	bigButton[label, action, Evaluate[$OperatingSystem]];

bigButton[label_, action_, 
	os_ /; StringMatchQ[os, StartOfString ~~ "Win" ~~ ___]] := 
	Mouseover[
		Button[bigButtonLabel[label], Null, Appearance -> None, 
			FrameMargins -> 5], Button[bigButtonLabel[label], action]];

bigButton[label_, action_, _] := 
  Button[bigButtonLabel[label], action, Appearance -> "Palette"];

bigButtonLabel[{(*icon*)_, labelText_, extra___}] := 
	nowrap@Style[labelText, Bold];

bigButtonLabel[labelText_] :=
	nowrap@Style[labelText, Bold];
		
toggleHideShowEnabled[] := (
	$ViewRemoved = ! $ViewRemoved;
	With[{selected = getSelectedAgents[]},
		If[!ViewRemoved && Intersection[selected, $RemovedAgents] === selected,
			clearAgentSelection[]]];
);
(******************************************************************************)
(* Remote Computers table pane *)

remoteComputersTablePane[urls_] := 
	Panel[
		Column[{
			remoteComputersTable[urls]
		}],
		ImageSize -> {$remoteComputersTableWidth, Full}, 
		BaselinePosition -> Top];

getAgentSelected[url_] := $AgentSelection[url];

setAgentSelected[url_, value : True | False] := (
	Clear[$AgentSelection];
	$AgentSelection[url] = value;
	$AgentSelectionEmpty = getSelectedAgents[] === {};
);

clearAgentSelection[] := (Clear[$AgentSelection]; $AgentSelectionEmpty = True);

getSelectedAgents[] := 
  Cases[DownValues[$AgentSelection], _[_[_[x_]], True] :> x];

agentSelectionEmptyQ[] := $AgentSelectionEmpty;

(******************************************************************************)
(* Remote Computers table *)

remoteComputersTable[urls_] := 
	formatTable[filteredColumnLabels[], remoteComputersTableContent[urls],
		Dynamic[$RemoteComputersTableScrollPosition],
		Alignment -> {{Left, {Center}}, Top}];

$RemoteComputersTableScrollPosition;

remoteComputersTableContent[{}] := 
	{{" "}, {Style[`noComputersMessage, Italic], SpanFromLeft}, {" "}};

remoteComputersTableContent[urls_] := 
	With[{agents = obtainAgentObject /@ urls},
		agentTableRow /@ agents];
 
(******************************************************************************)
(* Appearance Configuration *)
tableBg = GrayLevel[0.85];

(******************************************************************************)
(* Text resources *)
`noComputersMessage = 
	TR["RS.Looking"];
`closeButtonTooltip = TR["RS.CloseTooltip"];
`networkStartingPointTitle = 
	TR["RS.DiscoverDialog.Text1"];
`networkStartingPointAddLabel = 
	TR["RS.DiscoverDialog.Text2"];

(******************************************************************************)
(* Table Row *)
agentTableRow[agent_Agent] := 
	#[[2]]& /@ agentTableRow["basic", agent];

agentTableRow["basic", agent:Agent[url_, ptr_Pointer]] := {
	TR["RS.Computer"] :> 
		Setter[Dynamic[getAgentSelected[url], 
			setAgentSelected[url, #] &], True,
			nowrap[LightweightGridClient`Private`AgentShortName[url]] ~tip~ url, 
			Appearance -> "Palette", ImageSize -> {$computerSetterWidth, 22}],
	TR["RS.Kernels"] :> 
		With[{countPtr = Address[ptr, {1, "KernelCount"}]},
			Spinner[getKernelCount[countPtr]&, 
				setKernelCount[countPtr, #]&]
		],
	TR["RS.Enable"] :>
		Checkbox[Dynamic[isAgentRemoved[url], 
			(setAgentRemoved[url, !#]; 
			 If[!$ViewRemoved, clearAgentSelection[]])&]]
};

filteredColumnLabels[] := $Columns;

(* INIT *)
$Columns = First /@ agentTableRow["basic", Agent["*", Pointer[]]];

(******************************************************************************)
(* Detail Inspector *)
detailInspector[urls_:agentUrls[]] := Panel[
	Column[detailInspectorContent[urls, getSelectedAgents[]]],
	ImageSize -> {$detailInspectorWidth, Full}, BaselinePosition -> Top];

detailInspectorContent[{}, {}] := 
	If[$ViewRemoved, 
		detailInspectorContentNoAgents[], 
		detailInspectorContentNoSelection[]];
detailInspectorContent[_, {}] := detailInspectorContentNoSelection[];

detailInspectorContentNoSelection[] := 
	{TR["RS.NothingSelected"], Pane[" ", ImageSize -> {Automatic, 200}]};

detailInspectorContentNoAgents[] := {Pane[Style[Column[{
	TR["RS.NoComputers"],
	" ",
	Grid[{
		{"\[Bullet]", 
		TR["RS.ClickDiscover1"] <> " \""<>discoverMoreKernelsLabelText<>
			"\" "<> TR["RS.ClickDiscover2"]},
		{" ", SpanFromLeft},
		{"\[Bullet]", 
		TR["RS.Install"]}}, 
		Alignment -> Left]
}], LineIndent -> 0]]};

detailInspectorContent[_, {url_, ___}] := {
	Row[{TR["RS.ComputerContent"], LightweightGridClient`Private`AgentShortName[url]}],
	TR["RS.ManagementInterface"],
	Hyperlink[truncate[url], url],
(*	" ",
	agentInfo[url],*)
	" ",
	kernelDetailEditor[url]
};

truncate[src_String, limit_:32] := 
	If[StringLength[src] > limit, 
		StringTake[src, limit]<>"... \[RightSkeleton]", 
		src];

(******************************************************************************)
(* Agent Info *)
agentInfo[url_String] := agentInfo[RemoteServicesAgentInformation[url]];

agentInfo[RemoteServicesAgents[rules_]] := Grid[{
	{"Operating System:", "OperatingSystem" /. rules},
	{"Kernel Load:", Length["KernelsRunning" /. rules]}},
	Alignment -> {Left, Top}];

(******************************************************************************)
(* Kernel Detail Editor *)

kernelDetailEditor[url_String] := kernelDetailEditor[obtainAgentObject[url]];

kernelDetailEditor[agent : Agent[url_, ___]] := 
	Column[{nowrap[TR["RS.LaunchSettings"]], kernelSettingEditor[agent]}];
(*
	OpenerView[
		{
			Style["Advanced Settings", Bold],
			Column[{nowrap["Launch Settings"], kernelSettingEditor[agent]}]
		},
		Dynamic[$KernelDetailOpen]];
*)

kernelSettingEditor[agent:Agent[url_, ptr_Pointer]] := 
	With[{
		ptrService = Address[ptr, {1, "Service"}],
		ptrLocalLinkMode = Address[ptr, {1, "LocalLinkMode"}],
		ptrTimeout = Address[ptr, {1, "Timeout"}]},

		Grid[{
			{nowrap[TR["RS.Service"]] ~tip~ 
				TR["RS.ServiceTooltip"], 
			InputField[Dynamic[deref[ptrService], 
				setService[ptrService, #]&], String, 
				ImageSize -> $launchSettingsInputFieldWidth,
				FieldSize -> $launchSettingsInputFieldSize]}
			, 
			{nowrap[TR["RS.Timeout"]] ~tip~ 
				TR["RS.TimeoutTooltip"], 
			InputField[Dynamic[deref[ptrTimeout], 
				setTimeout[ptrTimeout, #]&], Number, 
				ImageSize -> $launchSettingsInputFieldWidth,
				FieldSize -> $launchSettingsInputFieldSize]}
			, 
			{nowrap[TR["RS.LinkMode"]] ~tip~ 
				TR["RS.LinkModeTooltip"], 
			PopupMenu[Dynamic[deref[ptrLocalLinkMode],
				setLocalLinkMode[ptrLocalLinkMode, #]&], 
					{"Connect", "Create"}]} 
		}, Alignment -> {Left, Baseline}]
	];

getService[ptr_] := Part[$Kernels, partSequence[ptr]];
setService[ptr_, service_String] := 
	Part[$Kernels, partSequence[ptr]] = service;
getTimeout[ptr_] := Part[$Kernels, partSequence[ptr]];
setTimeout[ptr_, timeout_] := 
	Part[$Kernels, partSequence[ptr]] = timeout;
setLocalLinkMode[ptr_, localLinkMode: "Connect" | "Create"] := 
	Part[$Kernels, partSequence[ptr]] = localLinkMode;

(******************************************************************************)
(* Network Starting Points dialog *)
CreateNetworkStartingPointsDialog[] := CreateDialog[
	networkStartingPointsContent[],
	(* Dialog options *)
	Modal -> True,
	WindowTitle -> TR["RS.DiscoverDialog.Title"],
	WindowSize -> Fit
];

networkStartingPointsContent[] := Column[{
	Pane[Style[`networkStartingPointTitle, LineIndent -> 0],
		ImageSize -> {300, Automatic}],
	" ",
	`networkStartingPointAddLabel,
	addNetworkStartingPointRow[],
	Grid[{{networkStartingPointsTable[], networkStartingPointsTableButtons[]}},
		Alignment -> Top],
	Item[DefaultButton[], Alignment -> Right]
}];

$AddNetworkStartingPoint = "";
$StartingPointImageSize = {200, Automatic};
addNetworkStartingPointRow[] := 
	Row[{
		InputField[Dynamic[$AddNetworkStartingPoint], String, 
			ImageSize -> $StartingPointImageSize],
		Button[TR["RS.DiscoverDialog.Add"],
			$NetworkStartingPoints = 
				Union[$NetworkStartingPoints, {$AddNetworkStartingPoint}];
			Quiet@RemoteServicesAgents[$AddNetworkStartingPoint, 
				"TemporaryPrinting" -> False, "CacheTimeout" -> 0],
			ImageSize -> All, Method -> "Queued"]
	}];

networkStartingPointsTable[] := 
	Dynamic[formatFramedTable[{TR["RS.DiscoverDialog.StartingPoint"], TR["RS.DiscoverDialog.Status"]}, 
		networkStartingPointsTableContent[$NetworkStartingPoints]],
		UpdateInterval -> 3];

networkStartingPointsTableContent[{}] := 
	{{Invisible[startingPointToggler[" "]], 
		Invisible["Being Contacted"]}};

networkStartingPointsTableContent[urls_] := 
	startingPointRow /@ $NetworkStartingPoints;

startingPointRow[url_] := {
	startingPointToggler[url],
	startingPointStatus[url]
};

startingPointToggler[url_] := 
	setterToggler[nowrap[url ~tip~ url], 
		$startingPointSelection === url &, 
		($startingPointSelection = 
			If[$startingPointSelection === url, Null, url])&, 
		ImageSize -> $StartingPointImageSize];

(* INIT: *)
$startingPointSelection = Null;

startingPointStatus[url_String] := 
	startingPointStatus[LightweightGridClient`Private`$AgentsCache[url]];

startingPointStatus[{timestamp_, urls_List}] := 
	Row[{TR["RS.DiscoverDialog.AliveFound"], Length[urls]}];

startingPointStatus[{timestamp_, "Pending"}] := TR["RS.DiscoverDialog.BeingContacted"];

startingPointStatus[_] := TR["RS.DiscoverDialog.NotResponding"];

networkStartingPointsTableButtons[] := Column[{
    Button[TR["RS.DiscoverDialog.Edit"], CreateEditSelectedStartingPointDialog[],
    	Enabled -> Dynamic[StringQ[$startingPointSelection]]],
    Button[TR["RS.DiscoverDialog.Remove"], removeSelectedStartingPoint[], 
    	Enabled -> Dynamic[StringQ[$startingPointSelection]]]
}];

removeSelectedStartingPoint[] := (
	$NetworkStartingPoints = 
		Complement[$NetworkStartingPoints, {$startingPointSelection}];
	$startingPointSelection = Null;
);

(******************************************************************************)
(* Edit Starting Point dialog *)
CreateEditSelectedStartingPointDialog[] := (
	$EditSelectedStartingPoint = $startingPointSelection;
	CreateDialog[
		editSelectedStartingPointContent[],
		(* Dialog options *)
		Modal -> True,
		WindowTitle -> TR["RS.DiscoverDialog.Title"],
		WindowSize -> Fit
	]
);

editSelectedStartingPointContent[] := Column[{
	"Name:",
	InputField[Dynamic[$EditSelectedStartingPoint], String, 
		ImageSize -> $StartingPointImageSize],
	" ",
	ChoiceButtons[{
		(* OK: *)
		$NetworkStartingPoints = 
			$NetworkStartingPoints /. 
				{$startingPointSelection -> $EditSelectedStartingPoint};
		$startingPointSelection = $EditSelectedStartingPoint;
		Quiet@RemoteServicesAgents[$EditSelectedStartingPoint, 
				"TemporaryPrinting" -> False, "CacheTimeout" -> 0];
		DialogReturn[],
		(* Cancel: *)
		DialogReturn[]}]
}];

(******************************************************************************)
(* Remove computers *)
removeAgents[] :=
	With[{selected = getSelectedAgents[]},
		With[{removeable = Select[selected, 
				MatchQ[# /. $Kernels, 
					LightweightGrid[{___, "KernelCount" -> 0, ___}]] &]},
			removeAgents[removeable, Complement[selected, removeable]]]];
removeAgents[removeable_, {}] := (
	selectNoAgents[];
	$RemovedAgents = Union[$RemovedAgents, removeable]);
removeAgents[removeable_, unremoveable_] := 
	If[TrueQ@ChoiceDialog[
		"Some of the computers to be removed have kernels configured.\n"<>
		"Remove them anyway and remove their kernels from the configuration?"],
		selectNoAgents[];
		$RemovedAgents = Union[$RemovedAgents, removeable]];

isAgentRemoved[url_] := !MemberQ[$RemovedAgents, url];

setAgentRemoved[url_, True] := 
	$RemovedAgents = Union[$RemovedAgents, {url}]; 
setAgentRemoved[url_, False] := 
	$RemovedAgents = Complement[$RemovedAgents, {url}];

(******************************************************************************)
CreateSetAllDialog[] := CreateDialog[
	setAllContent[],
	(* Dialog options *)
	Modal -> True,
	WindowTitle -> TR["RS.SetAll.Title"],
	WindowSize -> Fit
];

$DoSetAllService = False;
$DoSetAllTimeout = False;
$DoSetAllLocalLinkMode = False;
$ShowSetAllAdvanced = False;

setAllContent[] := (
	$SetAllKernelCount = defaultSetAllValue["KernelCount", _Integer, 0];
	$SetAllService = defaultSetAllValue["Service", _String, ""];
	$SetAllTimeout = defaultSetAllValue["Timeout", _Integer?NonNegative, 
		"Timeout" /. Options[RemoteKernelOpen]];
	$SetAllLocalLinkMode = defaultSetAllValue["LocalLinkMode", 
		"Connect" | "Create", "Connect"];
	If[!MatchQ[$ShowSetAllAdvanced, True | False],
		$ShowSetAllAdvanced = False];
	If[!MatchQ[$DoSetAllService, True | False],
		$DoSetAllService = False];
	If[!MatchQ[$DoSetAllTimeout, True | False],
		$DoSetAllTimeout = False];
	If[!MatchQ[$DoSetAllLocalLinkMode, True | False],
		$DoSetAllLocalLinkMode = False];

	Column[{
		If[$ViewRemoved,
			TR["RS.SetAll.KernelCountAll"],
			TR["RS.SetAll.KernelCountEnabled"]],
		Spinner[$SetAllKernelCount&, ($SetAllKernelCount = #)&],
		" ",
		OpenerView[{TR["RS.SetAll.LaunchSettings"],
			Grid[{
				{RadioButton[Dynamic[$DoSetAllService], False], 
					TR["RS.SetAll.DontChangeService"]},
				{RadioButton[Dynamic[$DoSetAllService], True], 
					TR["RS.SetAll.SetService"]}, 
				{" ", InputField[Dynamic[$SetAllService], 
					String, ImageSize -> $launchSettingsInputFieldWidth,
					FieldSize -> $launchSettingsInputFieldSize,
					Enabled -> Dynamic[$DoSetAllService]]},
				{" ", SpanFromLeft},
				{RadioButton[Dynamic[$DoSetAllTimeout], False], 
					TR["RS.SetAll.DontChangeTimeout"]},
				{RadioButton[Dynamic[$DoSetAllTimeout], True], 
					TR["RS.SetAll.SetTimeout"]}, 
				{" ", InputField[Dynamic[$SetAllTimeout], 
					Number, ImageSize -> $launchSettingsInputFieldWidth,
					FieldSize -> $launchSettingsInputFieldSize, 
					Enabled -> Dynamic[$DoSetAllTimeout]]},
				{" ", SpanFromLeft},
				{RadioButton[Dynamic[$DoSetAllLocalLinkMode], False], 
					TR["RS.SetAll.DontChangeLocalLinkMode"]},
				{RadioButton[Dynamic[$DoSetAllLocalLinkMode], True], 
					TR["RS.SetAll.SetLocalLinkMode"]}, 
				{" ", PopupMenu[Dynamic[$SetAllLocalLinkMode], 
					{"Connect", "Create"}, 
					Enabled -> Dynamic[$DoSetAllLocalLinkMode]]}
			}, Alignment -> {Left}]
			},
			Dynamic[$ShowSetAllAdvanced]],
		" ",
		Item[ChoiceButtons[{commitSetAll[], DialogReturn[]}], 
			Alignment -> Right]
	}]
);

(* If all the agents have the same value for the property, use that, else 0.*)
defaultSetAllValue[property_String, pattern_, defaultValue_] := 
	Quiet@Union[
		Map[
			Function[agent, 
				Part[$Kernels,
					partSequence[Address[Evaluate@getKernelPointer[agent], 
						{1, property}]]]], 
			agentUrls[kernelAgents[]]]
	] /. {
		{x:pattern} :> x,
		_ :> defaultValue};

commitSetAll[] := (
	Map[Function[agent,
			With[{ptr = getKernelPointer[agent]},
				setKernelCount[Address[ptr, {1, "KernelCount"}], 
					$SetAllKernelCount];

				If[$DoSetAllService && StringQ[$SetAllService],
					Part[$Kernels, 
						partSequence[Address[ptr, {1, "Service"}]]] = 
						$SetAllService];
				If[$DoSetAllTimeout && NonNegative[$SetAllTimeout],
					Part[$Kernels, 
						partSequence[Address[ptr, {1, "Timeout"}]]] = 
						$SetAllTimeout];
				If[$DoSetAllLocalLinkMode && 
					MemberQ[{"Connect","Create"}, $SetAllLocalLinkMode],
					Part[$Kernels,
						partSequence[Address[ptr, {1, "LocalLinkMode"}]]] = 
						$SetAllLocalLinkMode];
			]
		],
		agentUrls[kernelAgents[]]];
	DialogReturn[]
);

(******************************************************************************)
(* Generic GUI utilities *)
nowrap[expr_, opts___] := Style[expr, LineBreakWithin -> False, opts];
tip[expr_, tiplabel_] := Tooltip[expr, tiplabel, TooltipDelay -> Automatic];
label[x_] := nowrap@Style[x, FontFamily -> "SansSerif"];

formatTable[headings_List, rows : {_List ...}, opts___] := 
	formatTable[headings, rows, ScrollPosition /. Options[Pane], opts];

formatTable[headings_List, rows : {_List ...}, scrollPos_, opts___] := 
	Pane[
		Grid[Join[{tableHeading /@ headings}, rows], opts, Alignment -> Left],
		{Automatic, 200}, 
		(* Pane options *)
		Scrollbars -> Automatic, ScrollPosition -> scrollPos,
		AppearanceElements -> None];

formatFramedTable[args___] := Framed[formatTable[args], FrameMargins -> 0];

tableHeading[expr_] := expr;
(*
tableHeading[expr_] := Item[Style[Pane[expr, ImageMargins -> {{4, 0}, {1, 0}}], 
	LineBreakWithin -> False], Alignment -> Left, 
	Frame -> {{True, True}, False}, FrameStyle -> White, Background -> tableBg];
*)

setterToggler[label_, getState_, setState_, opts___] := 
	Toggler[Dynamic[getState[], setState], {
		False -> Button[label, Null, Alignment -> Left, 
			Appearance -> "Palette", opts], 
		True -> Button[label, Null, Alignment -> Left, 
			Appearance -> {"Palette", "Pressed"}, opts]}];

(******************************************************************************)
(* Helpful abstractions *)

(* The set of URLs to display in the table *)
agentUrls[] := agentUrls[kernelAgents[]];
agentUrls[configuredAgents_] := 
	Complement[Union[Join[discoveredAgents[], configuredAgents]],
		If[$ViewRemoved, {}, $RemovedAgents]];

(* The URLs of agents discovered automatically. *)
discoveredAgents[] := With[{
	actives = Cases[#@"Agent" & /@ RemoteKernelInformation[], _String], 
	locals = If[$BrowseLocal === True, browseLocal[], {}], 
	remotes = Cases[Flatten[
			(Quiet@RemoteServicesAgents[#, "TemporaryPrinting" -> False, 
				"CacheTimeout" -> $UpdateFrequency])&
				/@ $NetworkStartingPoints], _String]
	},
	Union[Join[actives, locals, remotes]]
];

browseLocal[] := LightweightGridClient`Private`$RemoteServicesAgents;

(* Wrapper for a function that displays information about a list of WRS agents.
	This wrapper abstracts the Refresh expression.
*)
agentRefresherView[f_, configuredAgents_, updateInterval_:15] := 
	Dynamic[
		With[{urls = Refresh[agentUrls[configuredAgents], 
			UpdateInterval -> updateInterval]},
			f[urls]
		]];

(* obtainAgentObject[url] ensures that agent maps have an entry for URL.  
	It returns an Agent object for manipulating the $Kernels entry with Part.
	Signature: String -> Agent[url_, Pointer[...]] 
*)
obtainAgentObject[url_String] := (
	addKernel[url];            (* Ensure entry exists in $Kernels *)
	obtainSelectionState[url]; (* Ensure checkbox selection exists *)
	Agent[url, getKernelPointer[url]]
);

(* INIT *)
$KernelDetailOpen = False;

obtainSelectionState[url_String] := 
	With[{state = $AgentSelection[url]},
		If[MatchQ[state, True | False],
			state,
			$AgentSelection[url] = False]];

getKernelCount[ptr_Pointer] := Part[$Kernels, partSequence[ptr]];
setKernelCount[ptr_Pointer, kernelCount_Integer /; NonNegative[kernelCount]] := 
	Part[$Kernels, partSequence[ptr]] = kernelCount;

Spinner[get_Function, set_Function, inputFieldOptions___] := 
	SubKernels`Protected`Spinner[Dynamic[get[], set], {0, Infinity, 1}];

closeIcon::usage = "closeIcon is a Graphics expression in the shape of an X reminiscent of a window close icon";
closeIcon = Graphics[{Black, Thickness[0.14], Line[{{0.2, 0.2}, {0.8, 0.8}}], 
	Line[{{0.2, 0.8}, {0.8, 0.2}}]}, 
	PlotRange -> {{0, 1.1}, {-0.1, 1.1}}, ImageSize -> {12, 12}];

SetAttributes[closeButton, HoldFirst];
closeButton[action_, opts___] := 
	(*Button[closeIcon, action, opts, ImageSize -> {16, 16}, FrameMargins -> 0];*)
	Button[Style["Remove", Small], action, opts, ImageSize -> All];

(*****************************************************************************)

End[];

EndPackage[]

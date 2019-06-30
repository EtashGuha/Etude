Package["NeuralNetworks`"]

(* A NOTE ABOUT WRITING UPGRADE RULES:

Add them to layer definitions, in the field Upgraders:

Upgraders: {
	"11.3.0" -> Function[Block[{p = #}, p["Woot"] = 99; p]]
}

This will introduce a new field called "Woot" into the assoc of that
layer, when that layer, having been produced in a version prior to 11.3.0,
is loaded in a version 11.3.0 or a later version.

If there are several rules, then of course they will be applied in sequence
as appropriate (if people write MakeUpgraderFrom definitions properly, which
isn't hard).

Then, crucially, BUMP THE FINAL VERSION NUMBER IN THE PACLETINFO.M FILE.
This ensures that nets produced using the previous version will be properly
scrutinized when loaded in later versions. If you don't do this, then during
the pre-release period you can end up with various nets using layers in various
stages of flux that are all tagged with the same version, and therefore are
"zombies" that are half-alive and half-dead. 

Obviously this adds extra incenstive to get changes to layers right the first time.
Otherwise you'll have to write upgraders for your mistakes! 

*)


PackageScope["UpgradeAndSeal11V0Net"]
PackageScope["SealNet"]

SetHoldAll[UpgradeAndSeal11V0Net, SealNet];

(* this is called directly by single-assoc downvalues, because we know
those guys represent 11.0.0 nets and they have no metadata *)

UpgradeAndSeal11V0Net[head_Symbol[assoc_Association ? AssociationQ]] := 
	ReconstructNet[assoc, <|"Version" -> "11.0.0"|>, "interpret"];

(* this is called by two-assoc downvalues *)

SealNet[_[assoc_Association ? AssociationQ, metadata_Association]] :=
	ReconstructNet[assoc, metadata, "interpret"];

SealNet[___] := $Failed;


PackageScope["ReconstructNet"]

SetUsage @ "
ReconstructNet[assoc$, metadata$, 'action$'] takes a net's association and metadata \
and attempts to update it to the current version, if necessary.
* A 'sealed' net is produced, e.g. a no-entry expression that contains assoc$ and metadata$.
* Failure to do so uses a message that mentions 'action$'.
* The 'Version' field of metadata$ is set to the current version.
* This is the sister function to ConstructNet, which is how sealed nets are created from scratch."

General::nnincmpbdlfail = "The update could not be obtained. Please check your internet connection and try again."; 
General::nnincmpbdl = "The neural network framework must be updated to version `` or later to `` the network. The update will now be downloaded and applied; restarting the session may be required."; 
General::nnincmpb = "Cannot `` in version `` a network that was produced using version ``."

ReconstructNet[network_, metadata_, action_] := Scope[

	version = ToVersionString @ Lookup[metadata, "Version", Panic["CorruptMetadata"]];

	Which[
		(* if net is current *)
		version === $NeuralNetworksVersionNumber, 

			ConstructNet[network, metadata],

		(* if net is from prior version *)	
		VersionOrder[version, $NeuralNetworksVersionNumber] == 1, 
	
			(* we introduced coder versions in version 11.3.8. WLNetImport will
			already have upgraded these coders, but SealNet will not have. If ReconstructNet
			is being called by SealNet we must trigger re-evaluation of coders in the assoc *)
			If[action =!= "import" && VersionOrder[version, "11.3.9"] == 1,
				(* $AssumedCoderVersion will be used by the re-eval step *)
				network = network /. coder:(_NetEncoder | _NetDecoder) :> RuleCondition[upgradeLegacyCoder @ coder];
			];

			ConstructUpgradedNet[
				($LastUpgrader ^= MakeUpgraderFrom[version])[network],
				metadata
			],
	
		(* if net is from the future and is compatible with with this version *)
		compat = Lookup[metadata, "CompatibleVersions", None];
		ListQ[compat] && MemberQ[compat, $NeuralNetworksVersionNumber], 
			
			ConstructUpgradedNet[network, metadata],

		(* or if differs only in a point release ... *)
		thisMajorMinor = ToWLVersion @ $NeuralNetworksVersionNumber;
		otherMajorMinor = ToWLVersion @ version;
		thisMajorMinor === otherMajorMinor,

			InheritedMessage["nnincmpbdl", version, action];
			If[FailureQ @ UpdateNNPacletToAtLeast[version], ThrowFailure["nnincmpbdlfail", version, action]];
			(*ReloadNeuralNetworks[];*)
			If[ValueQ[$ReloadRecoveryFunction], $ReloadRecoveryFunction, ReconstructNet][network, metadata, action],

		(* otherwise there is no hope *)
		True,
			ThrowFailure["nnincmpb", action, thisMajorMinor, otherMajorMinor]
	]
];

(* inputform nets will have had their coders evaluate and acquire Indeterminate version.
we now fill that version in *)
upgradeLegacyCoder[head_[kind_, param_, type_]] /; !StringQ[param["$Version"]] := 
	head[kind, Append[param, "$Version" -> version], type];

upgradeLegacyCoder[e_] := e;

PackageScope["$AutoUpdateOnFutureLoad"]
PackageScope["UpdateNNPacletToAtLeast"]

If[!FastValueQ[$AutoUpdateOnFutureLoad], $AutoUpdateOnFutureLoad = True];

UpdateNNPacletToAtLeast[targetVersion_] := Scope[
	If[PacletManager`$AllowInternet =!= True || $AutoUpdateOnFutureLoad =!= True, ReturnFailed[]];
	result = Quiet @ Check[
		PacletManager`Package`getPacletWithProgress["NeuralNetworks", "Neural networks", "UpdateSites" -> False],
		$Failed
	];
	If[FailureQ[result] || result["Name"] =!= "NeuralNetworks", ReturnFailed[]];
	If[VersionOrder[targetVersion, result["Version"]] === -1, ReturnFailed[]];
];


PackageScope["$ReloadRecoveryFunction"]

(* if ReconstructNet is ever refactored in a future version, $ReloadRecoveryFunction should be a bridge *)


PackageScope["$LastUpgrader"] 
(* ^ this is purely for debugging *)

(*
NEW UPGRADE RULES:
simply add a line for each discrete version in which something changed and for which there
are corresponding rules in layer definitions of the form "VERSION" -> function. You should
chain these properly, so that an upgrader from 11.2.0 manually invokes the rules for 11.3.0
and then chains via /* onto MakeUpgraderFrom["11.3.0"] (once 11.3.0 comes around).

Automating this is overcomplicated and I suspect they'll be exceptions anyway. *)

Clear[MakeUpgraderFrom];

MakeUpgraderFrom[$NeuralNetworksVersionNumber] := Identity; 

(* There are no upgrade rules for this version. The number was bumped for a paclet update *)
MakeUpgraderFrom["12.0.7"] := 
	ReplaceAll[LayerUpgradeRules["12.0.8"]] /* MakeUpgraderFrom["12.0.8"];

MakeUpgraderFrom["12.0.6"] :=
	ReplaceAll[LayerUpgradeRules["12.0.7"]] /* MakeUpgraderFrom["12.0.7"];

MakeUpgraderFrom["12.0.5"] :=
	ReplaceAll[LayerUpgradeRules["12.0.6"]] /* MakeUpgraderFrom["12.0.6"];

MakeUpgraderFrom["12.0.4"] :=
	ReplaceAll[LayerUpgradeRules["12.0.5"]] /* MakeUpgraderFrom["12.0.5"];

MakeUpgraderFrom["12.0.3"] :=
	ReplaceAll[LayerUpgradeRules["12.0.4"]] /* MakeUpgraderFrom["12.0.4"];

MakeUpgraderFrom["12.0.2"] :=
	ReplaceAll[LayerUpgradeRules["12.0.3"]] /* MakeUpgraderFrom["12.0.3"];

MakeUpgraderFrom["12.0.1"] :=
	ReplaceAll[LayerUpgradeRules["12.0.2"]] /* MakeUpgraderFrom["12.0.2"];

MakeUpgraderFrom["12.0.0"] :=
	ReplaceAll[LayerUpgradeRules["12.0.1"]] /* MakeUpgraderFrom["12.0.1"];

MakeUpgraderFrom["11.3.9"] := 
	ReplaceAll[LayerUpgradeRules["12.0.0"]] /* MakeUpgraderFrom["12.0.0"];

Do[
	With[{prev = "11.3." <> IntegerString[i], next = "11.3." <> IntegerString[i+1]},
		MakeUpgraderFrom[prev] := ReplaceAll[LayerUpgradeRules[next]] /* MakeUpgraderFrom[next]
	]
,
	{i, 2, 8}  (* 11.3.2 to 11.3.8 only consisted of layer upgrades with the above simple chaining rule *)
];

MakeUpgraderFrom["11.3.1"] :=
	ReplaceAll[LayerUpgradeRules["11.3.2"]] /*
	ApplyGraphs[UpgradeGraphTupleEdges] /* 
    MakeUpgraderFrom["11.3.2"];

MakeUpgraderFrom["11.3.0"] :=
	ReplaceAll[LayerUpgradeRules["11.3.1"]] /*
	MakeUpgraderFrom["11.3.1"];

MakeUpgraderFrom["11.2.0"] := 
	ReplaceAll[LayerUpgradeRules["11.3.0"]] /* 
	MakeUpgraderFrom["11.3.0"];

(* 
OLD UPGRADE RULES:
It applies to nets from prior to 11.2 being upgraded. There was a lot of churn prior to 11.2, and also changes
in 11.0 -> 11.1 that didn't just involve layers being upgraded but entire internal representations being
replaced, so to avoid cluttering the layer definitions with all this legacy crap we leave it using these older,
centralized rules. *)

MakeUpgraderFrom["11.1.1"] := 
	ReplaceAll[LayerUpgradeRules["11.1.1 to 11.2.0"]] /* 
	MakeUpgraderFrom["11.2.0"];

MakeUpgraderFrom["11.1.0"] := 
	ReplaceAll[LayerUpgradeRules["11.1.0 to 11.1.1"]] /*
	ReplaceAll[LayerUpgradeRules["11.1.1 to 11.2.0"]] /* 
	MakeUpgraderFrom["11.2.0"];

RunInitializationCode[
	Import::netupgr = "Upgrading neural network to 11.1 format.";
	Off[Import::netupgr];
];

MakeUpgraderFrom["11.0.0"] := (
	Message[Import::netupgr];
	ReplaceAll[LayerUpgradeRules["11.0.0 to 11.1.1"]] /*
	(ReplaceRepeated[#, $TensorUpgradeRule]&) /*
	ReplaceAll[$CoderUpgradeRule] /*
	(ReplaceRepeated[#, HoldPattern[t_TensorT] :> RuleCondition[t]]&) /*
	ReplaceAll[NetPort -> NetPath] /*
	MakeUpgraderFrom["11.1.1"]
);

General::netupgrpr = "Networks saved during `` can not be loaded in ``.";

MakeUpgraderFrom[version_] := ThrowFailure["netupgrpr", version, $NeuralNetworksVersionNumber];


PackageScope["$TensorUpgradeRule"]

$CoderUpgradeRule = (EncodedType|DecodedType)[coder_, t_] :> RuleCondition @ UnifyCoderWith[coder, t //. $TensorUpgradeRule];
$TensorUpgradeRule = HoldPattern[t:TensorT[_Integer | SizeT | NaturalT, dims_]] :> TensorT[dims, RealT];


PackageScope["LayerUpgradeRules"]

(* 
LayerUpgradeRules is the processed form of $LayerUpgradeData, in which dispatch rules of the 
form type -> function are turned into actual patterns. 

It must be lazy because layers don't populate their upgrade rules until later. *)

Clear[LayerUpgradeRules];

LayerUpgradeRules[version_] := LayerUpgradeRules[version] = Block[
	{data = $LayerUpgradeData[version]},
	If[MissingQ[data], {}, Dispatch[makeUpgradePattern @@@ data]]
];

makeUpgradePattern[type_, func_] := 
	assoc:<|"Type" -> type, ___|> :> RuleCondition[Internal`UnsafeQuietCheck[func[assoc], unkerr[type]]];

General::netupgerr = "Could not upgrade the network created in version `` due to an incompatible ``.";

unkerr[type_] := ThrowFailure["netupgerr", version, $TypeToSymbol[type]];


PackageScope["$LayerUpgradeData"]

(* 
$LayerUpgradeData is populated both manually (for upgrades prior to 11.2)
and from  layer definitions. 

Manual rules use a different naming scheme "XXX to YYY" for clarity, automatic rules
just use "XXX".

Manual rules are defined only in LegacyUpgrade.m.
Automatic rules are attached by code in DefineLayer.m.

$LayerUpgradeData is actually initialized in LegacyUpgrade.m.
*)


PackageScope["UpgradeCoderParams"]
PackageScope["$AssumedCoderVersion"]
PackageScope["$currentlyUpgradingCoderType"]

(* this will have to be revisited if the structure of coders ever changes from
the current head[kind, params, type]
*)

UpgradeCoderParams[upgraders_, params_, coder_] := Scope[

	(* if there are no upgrade rules associated with this coder,
	there is nothing to do but bump the version to the current version *)
	If[upgraders === {}, Goto[NoUpgrade]];

	(* $AssumedCoderVersion is set by the ambient net being imported. if no net is
	being imported, this will be Indeterminate *)
	version = CacheTo[params, "$Version", $AssumedCoderVersion];

	If[version === Indeterminate, Return @ params];
	(* note: coders with indeterminate version can't be used, but they are at least
	marked as such *)

	version = FromVersionString @ version;
	
	(* select those funcs that came after the version of the coder *)
	funcs = Values @ Select[upgraders, Order[version, First[#]] == 1&];

	$currentlyUpgradingCoderType = coder[[1,3]]; (* <- gives access to type *)
	(* compose them in the right order *)
	params = Apply[RightComposition, funcs] @ params;

	Label[NoUpgrade];

	params["$Version"] = $NeuralNetworksVersionNumber;

	params
];


PackageScope["sortUpgradeRules"]

sortUpgradeRules[rules_] := Sort[FromVersionString[#1] -> #2& @@@ rules];


PackageScope["legacyCoderFail"]

General::nocoderupg = "The `` was created in a previous version of the Wolfram Language and cannot be used. Please recreate it in this version.";
legacyCoderFail[coder_] := ThrowFailure["nocoderupg", CoderFormString @ coder];


PackageScope["ApplyParams"]
PackageScope["MapAtParam"]
PackageScope["RenameParam"]
PackageScope["DropParam"]
PackageScope["AddParam"]
PackageScope["RenameArray"]

AddParam[f_Function][assoc_] := ApplyParams[Append[f @ assoc], assoc];
AddParam[newfields_][assoc_] := ApplyParams[Append[newfields], assoc];
ApplyParams[f_][assoc_] := ApplyParams[f, assoc];
ApplyParams[f_, assoc_] := MapAt[f, assoc, "Parameters"];
MapAtParam[f_, params_][assoc_] := MapAt[f, assoc, Thread[{"Parameters", params}]];
RenameParam[rules_][assoc_] := ApplyParams[KeyMap[Replace[rules]], assoc];
DropParam[key_][assoc_] := ApplyParams[KeyDrop[key], assoc];

ApplyArrays[f_][assoc_] := ApplyArrays[f, assoc];
ApplyArrays[f_, assoc_] := MapAt[f, assoc, "Arrays"];
RenameArray[rules_][assoc_] := ApplyArrays[KeyMap[Replace[rules]], assoc];

PackageScope["UpgradeAsymmetricPadding"]

(* this is a no-op for already-upgraded padding, which it must be because
the asymmetric padding upgrade was initially part of 11.3.7 but then got 
moved into 11.3.8 to prevent breaking nets already trained in 11.3.7  *)
UpgradeAsymmetricPadding = MapAtParam[tupleToMatrix, "PaddingSize"]
tupleToMatrix[ListT[rank_, NaturalT]] := ListT[rank, ListT[2, NaturalT]]
tupleToMatrix[list_List ? VectorQ] := Transpose[{list, list}]
tupleToMatrix[e_] := e

PackageScope["DropAllHiddenParams"]

DropAllHiddenParams[assoc_] := ApplyParams[KeySelect[StringFreeQ["$"]], assoc];
(* this strange beast exists because shape functions basically wiped out the
need for most hidden params *)


PackageScope["ApplyGraphs"]

ApplyGraphs[f_][assoc_] := ApplyGraphs[f, assoc];
ApplyGraphs[f_, assoc_] := ReplaceAll[assoc,
	container:<|"Type" -> "Graph", ___|> :>
		RuleCondition @ f @ MapAt[ApplyGraphs[f], container, "Nodes"]
];

(* :Copyright: 2010 by Wolfram Research, Inc. *)
BeginPackage["LightweightGridClient`ParallelConfiguration`"]

`$ConfigurationVersion;
`$BrowseLocal;
`$NetworkStartingPoints;
`$ViewRemoved;
`$RemovedAgents;
`$UpdateFrequency;
`$Kernels;
`$KernelsAtATime;
`getConfiguration;
`getKernels;
`loadConfiguration;
`resetSettings;
`kernelAgents;
`pruneAgents;
`getKernel;
`getKernelPointer;
`hasKernel;
`addKernel;
`createKernel;
`Pointer;
`partSequence;
`deref;
`Address;

Begin["`Private`"];

Needs["LightweightGridClient`SubKernels`"];

(* Called when SubKernels want to save the configuration *)
getConfiguration[] := {
	"Version" -> $ConfigurationVersion,
	"BrowseLocal" -> $BrowseLocal,
	"NetworkStartingPoints" -> $NetworkStartingPoints,
	"ViewRemoved" -> $ViewRemoved,
	"RemovedAgents" -> $RemovedAgents,
	"UpdateFrequency" -> $UpdateFrequency,
	"KernelsAtATime" -> $KernelsAtATime,
	"Kernels" -> $Kernels
};

(* Called when SubKernels wants to launch kernels *)
getKernels[] := #[[2]] & /@ Select[$Kernels, activeKernelQ];

activeKernelQ[(server_ /; ! MemberQ[$RemovedAgents, server]) -> 
    LightweightGrid[{___, "KernelCount" -> n_ /; n > 0, ___}]] := True;
activeKernelQ[_] := False;

loadConfiguration[cfg_] := (
	resetSettings[];
	
	SetIf[$BrowseLocal, "BrowseLocal" /. cfg, True | False];
	SetIf[$NetworkStartingPoints, "NetworkStartingPoints" /. cfg,
		List[_String ...]];
	SetIf[$ViewRemoved, "ViewRemoved" /. cfg, True | False];
	SetIf[$RemovedAgents, "RemovedAgents" /. cfg, List[_String ...]];
	SetIf[$UpdateFrequency, "UpdateFrequency" /. cfg, x_ /; NonNegative[x]];
	SetIf[$Kernels, "Kernels" /. cfg, 
		List[(_String -> _LightweightGrid) ...]];
	SetIf[$KernelsAtATime, "KernelsAtATime" /. cfg, 
		x_?IntegerQ /; NonNegative[x]];
);

resetSettings[] := (
	$ConfigurationVersion = "7.0.2";
	$BrowseLocal = True;
	$NetworkStartingPoints = {};
	$ViewRemoved = True;
	$RemovedAgents = {};
	$UpdateFrequency = 60.0;
	$Kernels = {};
	$KernelsAtATime = 2;
);

(* INIT: Ensure settings exist *)
resetSettings[];

SetAttributes[SetIf, HoldFirst];
SetIf[sym_, value_, pattern_, default_:Null] := 
	If[MatchQ[value, pattern],
		sym = value,
		default];

(*** Functions for manipulating the Kernels list ***)

(* Returns the list of URLs for agents in the $Kernels map *)
kernelAgents[] := First /@ $Kernels;

(* Prune kernel descriptions with 0 kernels or which do not appear in the given 
 list *)
pruneAgents[discovered_] := 
	With[{pruned = 
		Select[$Kernels, 
		MatchQ[#, _ -> LightweightGrid[{___, 
			"KernelCount" -> (n_ /; Positive[n]), ___}]] 
		|| 
		MemberQ[discovered, First[#]] &]},

		If[Length[pruned] < Length[$Kernels],
			$Kernels = pruned]];

getKernel[agent_String] := (agent /. $Kernels) /. {
	kernel_LightweightGrid :> kernel,
	_ :> Null
};

getKernelPointer[agent_String] := Address[$Kernels, {agent}];

hasKernel[agent_String] := MatchQ[getKernel[agent], _LightweightGrid];

addKernel[agent_String] := getKernel[agent] /. {
	kernel_LightweightGrid :> kernel,
	_ :> With[{kernel = createKernel[agent]},
		AppendTo[$Kernels, agent -> kernel];
		kernel]};

createKernel[agent_String] := 
	LightweightGrid[{"Agent" -> agent, "KernelCount" -> 0} ~Join~ 
		Options[RemoteKernelOpen]];

removeAgent[agent_String] := (
	$RemovedAgents = Union[Append[$RemovedAgents, agent]];
);

(*****************************************************************************)

(* Configuration Helper Functions *)

SetAttributes[Pointer, HoldFirst];
(* Given pointer to symbol A, you can write A[[partSequence[pointer]] *)
partSequence[Pointer[_, pspec_]] := Sequence @@ pspec;
(* deref[ptr] reads the expression pointed to *)
deref[ptr : Pointer[expr_, ___]] := Part[expr, partSequence[ptr]];
SetAttributes[Address, HoldFirst];
Clear[Address];

Address[sym_Symbol, fields_] := 
	Address[Evaluate[sym], fields] /. {
		Pointer[_, parts_] :> Pointer[sym, parts],
		_ :> $Failed};
Address[ptr : Pointer[expr_, pspec_], fields_] := 
	Address[Evaluate[deref[ptr]], fields] /. {
		Pointer[_, parts_] :> Pointer[expr, Join[pspec, parts]],
		_ :> $Failed};
Address[expr_, {}] := Pointer[expr, {}];
Address[rules : {_Rule ...}, {key_, rest___}] := 
	AddressWithKey[rules, {rulePosition[rules, key], rest}, key];
AddressWithKey[rules : {_Rule ...}, {index_Integer, rest___}, key_] := 
	Address[Evaluate[rules[[index]]], {rest}] /. {
		Pointer[_, parts_] :> Pointer[rules, Join[{index, 2}, parts]],
		_ :> $Failed};
AddressWithKey[rules : {_Rule ...}, {_, rest___}, key_] := (
	Message[Address::partw, key, rules]; 
	$Failed
);
Address[expr_, {index_Integer /; 1 >= index >= Length[expr], rest___}] := 
	Address[Evaluate[expr[[index]]], {rest}] /. {
		Pointer[_, parts_] :> Pointer[expr, Join[{index}, parts]],
		_ :> $Failed};
Address[expr_, {index_Integer, rest___}] := (
	Message[Address::partw, index, expr];
	$Failed
);

rulePosition[rules:{_Rule ...}, key_] := Position[rules, key -> _, 1] /. {
	{{index_Integer},___} :> index,
	_ :> $Failed};

End[];

EndPackage[]

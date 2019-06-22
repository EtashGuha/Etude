Package["NeuralNetworks`"]


PackageScope["Var"]

(* this is a labaratory for some ideas that i want to use to basically replace the complex
unification and setting inference engine i have at the moment. 

for now, it is grafted onto the current engine in the way of ShapeFunctions and RankFunctions,
which are opt-in mechanisms that layers can use to simplify their inference. 

The glue that is used to hook the this stuff into layers via inference rules is called
MakeRankAndShapeFunctionRules in Shapes.m 

the API for both shape and rank functions is to take a list of inputs and return a list of
outputs. shape functions take a list of input dim lists. rank functions take a list of ranks.
they must both produce the same. shape functions can produce $Failed for a given output dim
to 'give up'. both must accept and produce Indeterminate as a way of saying 'unknown'. This
is used instead of SizeT because it has nice chaining properties with * and + etc.

The ForwardInverseShape and ForwardInverseRank functions will call the shape and rank functions,
but will convert unknown SizeT values to Vars that will be used to track how dimensions interact.
therefore the functions must be able to handle Vars gracefully, and try to propogate them where
possible. It's fine to produce e.g. Var[..] + 5 as an output, because that can be solved when
unifying with an existing output!
*)

SetAttributes[VarSet, Orderless];
Clear[VarSet];
VarSet[v_Var, n_Integer] := v = n;
VarSet[v_Var, x_LengthVar] := v = x;
VarSet[v_Var, w_Var] := Switch[Order[v, w], -1, v = w, 0, v, 1, w = v];
VarSet[v_Var, t_$type] := v = t;
VarSet[m_Integer, n_Integer] := If[m =!= n, $Failed, m];
VarSet[x_LengthVar, y_LengthVar] := Switch[Order[x, y], -1, y = x, 0, x, 1, x = y];
VarSet[x_LengthVar, n_Integer] := If[$preserveLVs, x, x = n];
VarSet[x_$type, y_$type] := If[x === y, x, $type @ UnifyTypes[First[x], First[y]]];
VarSet[_Plus | _Min | _Max, b_Integer] := b;
VarSet[_Plus | _Min | _Max, b_] := Indeterminate;
VarSet[n_, Indeterminate] := n;
$vdelta = 1; (* 1 for dims, 0 for rank *)
VarSet[v_ + n_Integer, b_Integer] := If[n + $vdelta <= b, VarSet[v, b - n]; b, $Failed];
VarSet[Min[n_Integer, v_], b_Integer] := Which[n > b, VarSet[v, b], n == b, Indeterminate, n < b, $Failed];
VarSet[Max[n_Integer, v_], b_Integer] := Which[n < b, VarSet[v, b], n == b, Indeterminate, n > b, $Failed];
VarSet[n_Integer * v_, b_Integer] /; Divisible[b, n] := (VarSet[v, b / n]; b);
VarSet[a_, b_] := $Failed;

$type[$Failed] = $Failed;

PackageScope["InverseShape"]

(* for testing... *)
InverseShape[shapef_, in_, out_] := 
	First[ForwardInverseShape[shapef, in, out]];


PackageScope["ForwardInverseShape"]
PackageScope["ForwardInverseRank"]
PackageScope["ForwardInverseType"]

ForwardInverseType[f_, in_, out_] := 
	forwardInverse[f, $type /@ in, $type /@ out, unifyType, $type[AtomT]] /. $type[t_] :> t;

ForwardInverseShape[f_, in_, out_] := 
	forwardInverse[f, in, out, unifyShape, SizeT] /. $unknownShape :> SizeListT[];

ForwardInverseRank[f_, in_, out_] := Block[{$vdelta = 0}, 
	forwardInverse[f, in, out, unifyRank, NaturalT]
];

forwardInverse[f_, in_ /; FreeQ[in, NaturalT | SizeT | LengthVar | AtomT], out_ /; FreeQ[out, LengthVar], unifier_, base_] :=
	inversed[f, in, out, unifier, base];

forwardInverse[f_, in_, out_, unifier_, base_] := Block[
	{Var, LengthVar, i = 0},
	inversed[f,
		in /. base :> RuleCondition @ Var[i++],
		out, unifier, base
	] /. _Var -> base
];

inversed[f_, in_, out_, unifier_, base_] := Catch @ Block[
	{in2, out2},
	out2 = f[in];
	out2 = unifyOutputs[unifier, base, out /. base -> Indeterminate, out2];
	in2 = If[f===Identity, out2, in];
	(* ^ %HACK to automatically tight variables in simple cases
		(e.g. "TypeFunction: Identity" in structural layers)
		This permits to add more dependency and more checks (see e.g. last tests in Test/Layers/TypeInference.m) *)
	{in2, out2}
];

unifyOutputs[uf_, base_, old_List, new_List] /; Length[old] === Length[new] := 
	MapThread[uf, {old, new}] /. Indeterminate -> base;

unifyOutputs[_, _, a_, b_] := Panic["InvalidOutputUnification", "`` and ``", a, b];

(* The strategy here is to punt on explicit conflicts and allow them to be
reported by the general machinery. *)
unifyShape[$Failed, out2_] := out2;
unifyShape[out1_, out2_] := Which[
	!ListQ[out2], $unknownShape,
	Length[out1] === Length[out2], MapThread[unifyDim, {out1, out2}],
	True, out2 
];

unifyDim[a_, b_] := Replace[
	TrySet[a, b, b], 
	Except[_Integer | _LengthVar] :> SizeT
];

unifyRank[a_, b_] := Replace[
	TrySet[a, b, b], {
	v_Var + n_Integer :> ShapeException["rank of output conflicts with rank implied by input"],
	Except[_Integer] :> NaturalT
}];

unifyType[a_, b_] :=
	TrySet[a, b, b]; 


PackageScope["TrySet"]
PackageScope["TrySetAll"]

SetHoldRest[TrySet];
SetHoldRest[TrySetAll];
TrySet[a_, b_, f_] := Replace[VarSet[a, b], $Failed :> f];
TrySetAll[list_, f_] :=	Fold[TrySet[#1, #2, f]&, list];

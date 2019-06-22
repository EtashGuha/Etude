Package["NeuralNetworks`"]



PackageScope["UnifyTypes"]

SetUsage @ "
UnifyTypes[type$1, type$2] unifies two types into the narrowest type that satisfies both."

UnifyTypes[t1_, t2_] := Catch[unify[t1, t2], unify];


PackageScope["UnifyExternalTypes"]

UnifyExternalTypes[t1_, t2_] := Block[{$preserveCoders = True}, UnifyTypes[t1, t2]];


PackageScope["unify"]

Clear[unify];

SetAttributes[unify, Orderless];

unify[e_, e_] := e;

unify[e_, ExpressionT] := e;

unify[m_Missing, e_] := Throw[$Failed, unify];
unify[$Failed, e_] := Throw[$Failed, unify];

unify[TypeT, e_] := e;

unify[value_String | value_Symbol, EnumT[alts_List]] := If[MemberQ[alts, value], value, Throw[$Failed, unify]];
unify[StringT, s_String] := s;

unify[BooleanT, v:True|False] := v;

unify[AtomT, other_:RealT|_IndexIntegerT] := other;

unify[IntegerT, n_Integer] := n;
unify[PosIntegerT, n:PosIntP] := n;
unify[NaturalT, n:NatP] := n;
unify[IntegerT, PosIntegerT] := PosIntegerT;
unify[IntegerT, NaturalT] := NaturalT;
unify[NaturalT, PosIntegerT] := PosIntegerT;

unify[i_IndexIntegerT, TensorT[dims_, t_]] := TensorT[unify[dims, {}], unify[i, t]];
unify[i_IndexIntegerT, PosIntegerT|NaturalT|IntegerT] := i;
unify[IndexIntegerT[n_Integer], x_Integer] /; 1 <= x <= n := x;
unify[IndexIntegerT[n1_], IndexIntegerT[n2_]] := IndexIntegerT[unify[n1, n2]];
(* Priorities to fuse integers defined here (undefined <c unbounded <c positive <c positive with max index) *)
unify[IndexIntegerT[SizeT], IndexIntegerT[n_]] /; MatchQ[n, _?IntegerQ|Infinity|All] := IndexIntegerT[n];
unify[IndexIntegerT[All], IndexIntegerT[n_]] /; (IntegerQ[n] || n === Infinity) := IndexIntegerT[n];
unify[IndexIntegerT[Infinity], IndexIntegerT[n_Integer]] := IndexIntegerT[n];

unify[ScalarT, r_ ? NumberQ] := N[r];

unify[ImageT[sz1_, c1_], ImageT[sz2_, c2_]] := ImageT[unify[sz1,sz2], unify[c1,c2]];
unify[HoldPattern @ img_Image, t_ImageT] := Coerce[img, t];

unify[Image3DT[sz1_, c1_], Image3DT[sz2_, c2_]] := Image3DT[unify[sz1,sz2], unify[c1,c2]];
unify[HoldPattern @ img_Image3D, t_Image3DT] := Coerce[img, t];

unify[t:SizeP, IntegerT|PosIntegerT] := t;
unify[SizeT, n:PosIntP] := n;

PackageScope["$preserveCoders"]
$preserveCoders = False;

unify[HoldPattern[NetDecoder[type_, assoc_, tensor1_]], HoldPattern[NetDecoder[type_, assoc_, tensor2_]]] :=
	Block[{type3 = unify[tensor1, tensor2]}, 
		If[$preserveCoders, NetDecoder[type, assoc, type3], type3]
	];

unify[t1_, coder:CoderP] := 
	If[$preserveCoders, 
		unify[t1, CoderType[coder]]; coder,
		unify[t1, CoderType[coder]]
	];

unify[EitherT[a_List], EitherT[b_List]] /; Length[a] === Length[b] := 
	EitherT[MapThread[unify, {a, b}]];

unify[d_, EitherT[tlist_List]] := 
	Replace[
		DeleteDuplicates @ DeleteCases[$Failed] @ Map[UnifyTypes[d, #]&, tlist], {
		{} :> Throw[$Failed, unify],
		{e_} :> e,
		e_List :> EitherT[e]
	}];

unify[s_TensorT, t_TensorT] := tunify[s, t];

unify[ListT[n1_, t1_], ListT[n2_, t2_]] := ListT[unify[n1, n2], unify[t1, t2]];

unify[ListT[n_, t_], e_List] := (unify[n, Length[e]]; unify[t, #]& /@ e);

unify[t_List, StartsWithT[a_List]] := StartsWithT[unify[Take[t, UpTo[Length[a]]], a]];

unify[t_List, e_List] /; Length[t] == Length[e] := MapThread[unify, {t, e}];

unify[EnumT[t1_], EnumT[t2_]] := 
	If[ListQ[t1] && ListQ[t2],
		EnumT @ Intersection[t1, t2],
		EnumT @ unify[t1, t2]
	];

unify[Defaulting[t1_, d_], Defaulting[t2_, d_]] := Defaulting[unify[t1, t2], d];

unify[d_, Defaulting[t_, _]] := unify[d, t];

unify[Nullable[t1_], Nullable[t2_]] := Nullable @ unify[t1, t2];

unify[d_, Nullable[t_]] := If[d === None, None, unify[d, t]];

unify[t_, p:TypeAliasP] := unify[t, ResolveAlias[p]];

unify[a_, b_] := Throw[$Failed, unify]

PackageScope["$preserveLVs"]
$preserveLVs = False;

unify[s_LengthVar, p:PosIntP] := If[$preserveLVs, s, p];

unify[s_LengthVar, SizeT] := s;

unify[LengthVar[id1_], LengthVar[id2_]] := LengthVar[Max[id1, id2]];

unify[a1_Association, a2_Association] /; Keys[a1] === Keys[a2] :=
	IMap[unify[#2, a2[#1]]&, a1]; 

unify[NetT[inputs_, outputs_], assoc_Association] :=
	assoc; (* FOR NOW *)

unify[NetT[inputs1_, outputs2_], NetT[inputs2_, outputs2_]] :=
	NetT[unifyio[inputs1, input2], unifyio[outputs1, outputs2]];

unify[v_ValidatedParameter, _] := v;

unify[SizeT|PosIntegerT|NaturalT, RepeatedInteger[x_]] := x

unify[ListT[n_Integer, type_], r_RepeatedInteger] := CTable[unify[type, r], n];

unify[ListT[SizeT|NaturalT, type_], r_RepeatedInteger] := r;

unify[list_List, RepeatedInteger[r_]] := If[MatchQ[list, {r..}], list, Throw[$Failed, unify]];

unify[e_, MatchT[patt_]] /; MatchQ[e, patt] := e;

unifyio[assoc1_, assoc2_] := 
	Association @ MapAssocAssoc[#1 -> unify[#2, #3]&, assoc1, assoc2, badio, badio];

badio[_] := Throw[$Failed, unify];


PackageScope["UnifyTensors"]

(* %ROBUSTNESS we need to sit back and prove this is NP-complete or whatever. it probably is.
but lets then decide on a subset to solve properly and do that, and have loads of tests *)

Clear[tunify];

SetAttributes[{tunify, innerEq}, Orderless];

UnifyTensors[t1_, t2_] := Catch[tunify[t1, t2], unify];

(* coder stripping is required for operators to preserve their interior coders properly --
we should have tests though.
the rule is that whenever the inner array could be a coder, you must call
maybeStripCoders to make sure that it is stripped from the result 
(if $preserveCoders is false).
*)
maybeStripCoders[t_] := If[$preserveCoders, t, t /. p:CoderP :> CoderType[p]];

innerEq[RealT, AtomT] := True;
innerEq[_IndexIntegerT, AtomT] := True;
innerEq[_IndexIntegerT, _IndexIntegerT] := True;
innerEq[_, _] := False;

(* special case of the below, but slightly faster because its an equality match *)
tunify[l:RealTensorT|AnyTensorT, r_] /; MatchQ[TType[r], RealT | AtomT] := 
	maybeStripCoders[r] /. AtomT -> UnifyTypes[TType[l], TType[r]];

(* flat case: just two simple arrays, no funny stuff *)
tunify[TensorT[d1_, t1_], TensorT[d2_, t2_]] /; innerEq[t1, t2] :=
	TensorT[unify[d1, d2], unify[t1, t2]];

(* fast path: inner array of t1 matches rank of t2, the outer dims on t1 just go away *)
tunify[TensorT[SizeListT[], t1_], t2_TensorT] /; TRank[t1] == TRank[t2] :=
	unify[TensorT[{}, t1], t2];

(* general case, at least up to one arb-rank gap, e.g. dimensions of the form 
dim lists of the form {l1,l2,..,lm, ___, rn,..,r2,r1}
we put them into a normal form and then do the various cases.
*)

DefineMacro[L, L[e_] := Quoted @ Length[e]];
DefineMacro[Swap, Swap[a_, b_] := Quoted @ Block[{t = a}, a = b; b = t]];

(* %HACK this fill business is sucky, we shoudl deal with RepeatedInteger some other way *)	
SetHoldFirst[checkRepInt];
checkRepInt[var_] := Which[
	!MatchQ[var, _RepeatedInteger], Null,
	fill === SizeT, fill = First @ var; var = True, 
	fill === First[var], var = True,
	True, Throw[$Failed, unify]
];

(* %TEMP we will remove this when we make coders themselves map at different levels. *)
tunify[t1:TensorT[{d_}, c:CoderP], t2_TensorT] := Scope[
	res = tunify[StripCoders[t1], t2];
	If[$preserveCoders, 
		{first, rest} = splitOneDim[res];
		TensorT[{unify[d, first]}, UnifyCoderWith[c, rest]],
		res
	]
];

splitOneDim[TensorT[d_List, t_]] := {First[d], TensorT[Drop[d, 1], t]};
splitOneDim[_] := Panic[];

compactDims[d:{{l___, SizeT...}, {SizeT..., r___}}] := {{l}, {r}, Length @ Flatten @ d};
(* ^ this logic to make:
NeuralNetworks`Private`Unify`tunify[
 TensorT[ListT[NaturalT, SizeT], TensorT[{2}, RealT]], 
 TensorT[ListT[NaturalT, SizeT], TensorT[ListT[1, SizeT], RealT]]]
give the right answer *)

tunify[t1_TensorT, t2_TensorT] := Scope[
	{a1, g1, b1, i1} = TToNormalForm[t1]; 
	{a2, g2, b2, i2} = TToNormalForm[t2];
	i3 = unify[i1, i2];
	If[FailureQ[a1] || FailureQ[b1], Throw[$Failed, unify]];
	fill = SizeT;
	checkRepInt[g1]; checkRepInt[g2];
	Which[
		g1 && g2, 
			{a1, b1, n1} = compactDims[{a1, b1}];
			{a2, b2, n2} = compactDims[{a2, b2}];
			a3 = unifyLeft[a1, a2]; b3 = unifyRight[b1, b2];
			n3 = Length[a3] + Length[b3]; 
			n4 = Max[n1, n2];
			If[n3 < n4, a3 = Join[a3, Table[SizeT, n4 - n3]]];
			res = {a3, True, b3, i3},
		If[g1 && !g2, Swap[{a1, g1, b1}, {a2, g2, b2}]];
		g2 && !g1,
			excess = L[a1] - (L[a2] + L[b2]);
			If[excess < 0, Throw[$Failed, unify]];
			res = {unify[a1, Join[a2, Table[fill, excess], b2]], False, {}, i3},
		True,
			(* we shouldn't reach this point, the fast path would have happened a few defs up *)
			If[L[a1] =!= L[a2], Throw[$Failed, unify]];
			res = {unify[a1, a2], False, {}, i3};
	];
	If[fill =!= SizeT && res[[2]], res[[2]] = RepeatedInteger[fill]];
	TFromNormalForm[res]
];

unifyLeft[{}, {}] := {};
unifyLeft[a_, {}] := a;
unifyLeft[{}, b_] := b;
unifyLeft[a_List, b_List] := Scope[
	alen = Length[a]; blen = Length[b];
	Which[
		alen  > blen, Join[unify[Take[a, blen], b], Drop[a, blen]],
		alen == blen, unify[a, b],
		alen <  blen, Join[unify[Take[b, alen], a], Drop[b, alen]]
	]
];

unifyRight[a_, b_] := Reverse @ unifyLeft[Reverse[a], Reverse[b]];


(* 
for TensorT[{5, 3}, RealT] <=> TensorT[ListT[NaturalT, SizeT], TensorT[{3}, RealT] 
for TensorT[SizeListT[], IndexIntegerT[SizeT]] <=> TensorT[{}, IndexIntegerT[3]]
UnifyTensors[
 TensorT[ListT[NaturalT, SizeT], TensorT[{5}, RealT]],
 TensorT[{LengthVar[1]}, TensorT[ListT[NaturalT, SizeT], RealT]]] -> 
 	TensorT[{LengthVar[1]}, TensorT[ListT[NaturalT, SizeT], TensorT[{5}, RealT]]]
*)

(* for TensorT[{1, SizeT}, RealT] <=> TensorT[{}, NetEncoder["Scalar"]] 
the problem is that the r1 == 0
we could also solve this by having TensorT[{}, CoderP] evaluate to CodeP *)
(*tunify[TensorT[{}, c:CoderP], t_TensorT] := unify[c, t];*)


PackageScope["DebugUnifyTensors"]

DebugUnifyTensors[t1_, t2_] := (AttachDebugInfo[{unify, tunify, unifyLeft, unifyRight, TToNormalForm, TFromNormalForm}]; CallGrid @ UnifyTensors[t1, t2]);

PackageScope["$TensorTFormatting"]
$TensorTFormatting = False;

TensorT /: MakeBoxes[t_TensorT /; $TensorTFormatting, StandardForm] := Block[{f = TUnroll[t]}, ToBoxes @ Subscript[AngleBracket @@ Most[f], Last[f]]];
ListT /: MakeBoxes[l:ListT[n_, t_] /; $TensorTFormatting, StandardForm] := SubscriptBox[ToBoxes[t], ToBoxes[Replace[n, NaturalT -> "n"]]];


PackageScope["UnifyCoderWith"]

UnifyCoderWith[a_, b_] := cunify[a, b];

cunify[t1_TensorT, t2_TensorT] := Block[{$preserveCoders = True}, tunify[t1, t2]];
(* delegate to tunify, because it will do the right thing thanks to $preserveCoders *)

cunify[coder_, p:CoderP] := cunify[coder, CoderType[p]];

cunify[coder:head_[name_, assoc_, type_], type2_] := 
	Which[
		type === type2, 
			coder,
		SameQ @@ ({type, type2} /. _LengthVar -> 0), 
			ReconstructCoder[head, name, assoc, unify[type, type2]],
		True,
			ReinferCoder[head, name, assoc, unify[type, type2]]
	];

cunify[_, _] := $Unreachable;


PackageScope["UnifiableQ"]

UnifiableQ[t1_, t2_] := t1 === t2 || !FailureQ[UnifyTypes[t1, t2]];

Package["NeuralNetworks`"]


PackageScope["ToT"]
PackageScope["dynamicDimP"]

SetUsage @ "
ToT[spec$] converts a user-provided spec into an internal type.
ToT[spec$, mode$] will create encoders or decoders from strings.
* If mode$ is NetEncoder, encoders will be created, otherwise decoders."

$iomode = None;
ToT[t_, mode_:None] := Block[{$iomode = mode}, toT[t]];

singleLetterP = _String ? (StringMatchQ[LetterCharacter]);
tensorTypeP = Alternatives[
	"Real",
	"Integer",
	HoldPattern @ Restricted["Integer" | Integer, _Integer],
	Restricted["Integer" | Integer, Infinity]
];
dynamicDimP = "Varying" | singleLetterP | _LengthVar;
staticDimP = (_Integer ? Positive) | Automatic;

toT = MatchValues[
	n_Integer ? Positive := VectorT[n];
	n:{Repeated[dynamicDimP, {0,1}], staticDimP...} := toTensor[n, RealT];
	n:{Repeated[dynamicDimP, {0,1}], staticDimP..., t:tensorTypeP} := toTensor[Most[n], %[t]];
	{n:(dynamicDimP | staticDimP), p:CoderP} := toTensor[{n}, %[p]];
	"Varying" := TensorT[{NewLengthVar[]}, RealT];
	"Real" := ScalarT;
	HoldPattern[Restricted["Integer" | Integer, n_]] /; MatchQ[n, Infinity|(_Integer?Positive)] := IndexIntegerT[n];
	"Integer" := IndexIntegerT[All];
	"Vector" := TensorT[{SizeT}, AtomT];
	"Matrix" := TensorT[{SizeT, SizeT}, AtomT];
	Automatic := TypeT;
	enc_NetEncoder := If[$iomode === NetEncoder, enc, $Failed];
	dec_NetDecoder := If[$iomode === NetDecoder, dec, $Failed];
	Interval[{a_, b_}] := IntervalScalarT[a, b];
	HoldPattern[(RepeatingElement|SequenceOf)[el_]] := SequenceT[LengthVar[0], %[el]];
	$Raw[t_] := t;
	t_ ? ValidTypeQ := t;
	l:singleLetterP := toTensor[{l}, RealT];
	t:{singleLetterP, RepeatedNull[_Integer|Automatic]} := toTensor[t, RealT];
	spec_String := Which[
		$iomode === None, $Failed, 
		!KeyExistsQ[If[$iomode === NetEncoder, $EncoderData, $DecoderData], spec], $Failed,
		True, checkF @ $iomode @ spec];
	$Failed
];

toTensor[_, $Failed] := $Failed;
toTensor[spec_, inner_] := TensorT[toDims @ spec, inner];

toDims[dims_] := ReplaceAll[dims, {
	Automatic -> SizeT, 
	"Varying" :> NewLengthVar[], 
	l:singleLetterP :> NameToLengthVar[l]
}];

checkF[f_Failure] := ThrowRawFailure[f];
checkF[e_] := e;


PackageScope["ToDimsList"]

ToDimsList = MatchValues[
	dim_Integer ? Positive := {dim};
	dim:dynamicDimP := toDims[{dim}];
	dims:{Repeated[dynamicDimP, {0,1}], staticDimP...} := toDims[dims];
	$Failed
];


PackageScope["FromT"]

SetUsage @ "
FromT[spec$] converts an internal type back into its equivalent user-providable spec."

FromT = MatchValues[
	enc_NetEncoder := enc;
	dec_NetDecoder := dec;
	RealT := "Real";
	IndexIntegerT[n_Integer] := Restricted["Integer", n];
	IndexIntegerT[All] := "Integer";
	IndexIntegerT[SizeT|Infinity] := Restricted["Integer", Infinity];
	IntervalScalarT[a_, b_] := Interval[{a, b}];
	VectorT[n_Integer] := n;
	TensorT[{}, RealT] := "Real";
	TensorT[{}, i_IndexIntegerT] := FromT[i];
	TensorT[SizeListT[n_Integer], type_] := %[TensorT[Table[SizeT, n], type]];
	TensorT[dims_List, RealT|AtomT] := fromDims[dims];
	TensorT[dims_List, type:(_IndexIntegerT | CoderP)] := Append[fromDims[dims], %[type]];
	SymbolicRandomArray[dist_, dims_] := Distributed[dims, Replace[dist, NNConstantDist[v_] :> UniformDistribution[{v,v}]]];
	Automatic
]

fromDims[dims_List] := ReplaceAll[dims, {
	SizeT -> Automatic, 
	lv_LengthVar ? NamedLengthVarQ :> LengthVarToName[lv],
	_LengthVar -> "Varying"
}];

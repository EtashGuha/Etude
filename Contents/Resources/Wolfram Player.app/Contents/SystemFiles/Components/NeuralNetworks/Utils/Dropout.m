Package["NeuralNetworks`"]



PackageScope["RecurrentDropoutMethodT"]

SetUsage @ "
RecurrentDropoutMethodT is a shortcut for a fairly complex ValidatedParameterT that handles the \
various possible dropout techniques for input and recurrent states in GRU etc."

RecurrentDropoutMethodT = ValidatedParameterT[parseRDO, None];

$allowedDOMethods = {"VariationalWeights", "VariationalInput", "VariationalState", "StateUpdate"};
parseRDO[None] := None;
parseRDO[p_ ? NumericQ] := checkDP[p];
parseRDO[assoc_Association] := parseRDO[Normal[assoc]];
parseRDO[list_List] := Association @ Map[parseRDO1, list];
parseRDO[spec_] := invDropSpec[];

checkDP[p_] := Replace[N[p], {r_Real /; Between[r, {0,1}] :> r, _ :> ThrowFailure["invdrppval"]}];
General::invdrppval = "Dropout probabilities should be numeric values between 0 and 1.";

parseRDO1[method_String -> p_] /; MemberQ[$allowedDOMethods, method] := method -> checkDP[p];
parseRDO1[_] := invDropSpec[];

General::invdrpspc = "Specification for \"Dropout\" should be either None, a numeric probability, or a list of rules mapping methods to probabilities, where the allowed methods are ``."
invDropSpec[] := ThrowFailure["invdrpspc", QuotedStringRow[$allowedDOMethods, " and "]];


PackageScope["MakeRNNDropoutData"]

SetUsage @ "
MakeRNNDropoutData[method$, n$] produces functions that will perform dropout in an RNN layer, according to a user-provided method spec.
* n$ is the number of gates that require dropout functions to be created for them.
* The return value is {{inDrop$1, ..}, {stateDrop$1, ..}, stateUpdateDrop$, weightDrop$}, where:
	* inDrop$i is the pre-loop dropout function that drops out the contribution from the input to the i$th gate (variationally)
	* stateDrop$i is the within-loop dropout function that drops out the contribution from the state to the i'th gate (variationally)
	* stateUpdateDrop$ is the within-loop dropout function that drops out the previous state's value (non-variationally)
	* connectDrop$ is the pre-loop dropout function that does DropConnect on the weight matrix itself
Note that if you intend to use one of the variational dropout functions inside an SowForEach, you need to apply it beforehand to dummy
data so that you will get a random mask, as the function itself will only be called once if the MX foreach operator is used."

MakeRNNDropoutData[spec_, n_] := Scope[
	{ip, sp, sup, wp} = toDropoutProbs @ spec;
	{
		Table[preLoopVarDropF[ip], n],
		Table[inLoopVarDropF[sp], n], 
		nonVarDropF[sup], 
		dropConnectF[wp]
	}
]

toDropoutProbs[_] /; !$TMode := 
	{0., 0., 0., 0.};

toDropoutProbs[None] := 
	{0., 0., 0., 0.};

toDropoutProbs[p_Real] := 
	{0., 0., 0., p};

toDropoutProbs[assoc_Association] := 
	N @ Lookup[assoc, 
		{"VariationalInput", "VariationalState", "StateUpdate", "VariationalWeights"}, 
		0.
	]

(* preLoopVarDropF is variational, but uses the axes parameter to achieve 
the desired mask, on an input whose first dim should be the time dim. 
it expects to be called once. *)
preLoopVarDropF[0.] := Identity;
preLoopVarDropF[prob_] := Function[SowDropout[#, prob, {0}]]

(* inLoopVarDropF computes the mask once and then reuses it again and again.
it expects to be called once for each step. if it is used in foreach, it should
be 'primed' on a dummy input so that the mask is computed, as foreach will only
call this function once. *)
inLoopVarDropF[0.] := Identity;
inLoopVarDropF[prob_] := ModuleScope[ 
	mask = None;
	Function @ If[mask === None, 
		result = SowDropout[#, prob];
		mask = result; mask[[2]] = 1; 
		result
	,
		SowHad[mask, #]
	]
]

(* nonVarDropF just does a common or garden dropout, with no mask-reuse *)
nonVarDropF[0.] := Identity;
nonVarDropF[prob_] := Function[SowDropout[#, prob]]

dropConnectF[0.] := Identity;
dropConnectF[prob_] := Function[SowDropConnect[#, prob]]


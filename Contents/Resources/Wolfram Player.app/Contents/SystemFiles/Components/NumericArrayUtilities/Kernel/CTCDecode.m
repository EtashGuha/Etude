(**********************************************************************
Distance Matrix Functions
**********************************************************************)

Package["NumericArrayUtilities`"]
PackageImport["GeneralUtilities`"]

(*-------------------------------------------------------------------*)
DeclareLibraryFunction[ctcBeamDecode, 
	"WL_ctc_decoder_beam",
	{
		{"NumericArray", Automatic},	(* probs *)
		Integer, 					(* top paths *) 
		Integer,					(* blank index *)
		Integer, 					(* beam width *)
		True|False, 				(* logitQ *)
		{Real, 1, "Constant"}		(* scores *)
	},
	{Integer, 1, Automatic}
]

(* initialize decoder instance with blankIndex, beamWidth, numClasses, and mixLevel *)
DeclareLibraryFunction[lmctcDecoderInit,
	"lm_ctc_decoder_init",
	{Integer, Integer, Integer, Real},
	"Void"];
(* run one iteration of the beam search with acoustic and lm probs (logitQ) from a single timestep *)
DeclareLibraryFunction[lmctcDecoderStep,
	"lm_ctc_decoder_step",
	{{"NumericArray", "Constant"}, True|False, {"NumericArray", "Constant"}, True|False, Integer},
	"DataStore"];
(* retrieve the list of current beam strings *)
DeclareLibraryFunction[lmctcDecoderGetStrings,
	"lm_ctc_get_current_strings",
	{Integer},
	"DataStore"];
(* finish the beam search (topPaths) and retrieve the scores (overwritten) and list of paths *)
DeclareLibraryFunction[lmctcDecoderFinish,
	"lm_ctc_decoder_finish",
	{Integer, {Real, 1, "Constant"}},
	{Integer, 1, Automatic}];

(*-------------------------------------------------------------------*)
PackageExport["CTCBeamSearchDecode"]

SetUsage[CTCBeamSearchDecode, "
CTCBeamSearchDecode[input$, topPaths$, beamWidth$, logits$, blankIndex$] \
decodes a set of probability matrix input$ (a NumericArray of type 'Real32') \
of dimensions (sequence length, alphabet size + 1). If logits$ is False,
the input$ is expected to a normalized sequence of probability vectors, \
whilst input$ is an un-normalized sequences of logit vectors if logits$ is \
True. topPaths$ controls how many paths are returned, whilst beamWidth$ \
controls the beam size to search over. If beamWidth$ and blankIndex$ are 1, \
this is just greedy decoding. blankIndex$ gives the position of the \
blank label. Negative indexing is supported.
"
]

(* NOTE: currently bug https://github.com/tensorflow/tensorflow/issues/6034 
giving wrong log likelihoods!! *)

CTCBeamSearchDecode[
	input_NumericArray /; (NumericArrayType[input] == "Real32"), 
	topPaths_Integer, beamWidth_Integer, logits_ /; BooleanQ[logits],
	blankIndex_Integer] := 
Scope[
	(* tensorflow fails if topPaths is too large. Is there a formula? 
		Not obvious. For now: try lower paths till success! Not very efficient,
		but not the most important case. Could use bisection search in future.
	*)
	Do[
		out = iCTCBeamSearchDecode[input, (topPaths - i), beamWidth, logits, blankIndex];
		If[!FailureQ[out], Break[]]
	,
		{i, 0, topPaths - 1}
	];
	out
]

iCTCBeamSearchDecode[input_, topPaths_, beamWidth_, logits_, blankIndex_] := 
Scope[
	dims = Dimensions[input];
	If[Length[dims] =!= 2, Return[$Failed]];
	{seqlen, class} = dims;
	(* process blank index to conformed form *)
	blankIndex2 = blankIndex;
	If[blankIndex < 0, blankIndex2 = (class + 1 +  blankIndex)];
	If[class < blankIndex2 < 1, Return[$Failed]];
	blankIndex2 -= 1; (* zero indexing *)
	(* this will be completely overwritten in C *)
	loglikelihood = ConstantArray[0.0, topPaths];
	paths = ctcBeamDecode[input, topPaths, 
		blankIndex2, beamWidth, 
		logits, loglikelihood
	];
	If[!ListQ[paths], Return[$Failed]];
	paths = TakeList[paths[[topPaths + 1 ;; ]], paths[[;; topPaths]]];
	paths += 1; (* convert from 0-indexed *)
	loglikelihood *= -1; (* pytorch returns log likelihood, tf uses neg *)
	<|"Paths" -> paths, "LogLikelihood" -> loglikelihood|>
]

(*-------------------------------------------------------------------*)
PackageExport["CTCBeamSearchDecode2"]

SetUsage[CTCBeamSearchDecode2, ""]

Options[CTCBeamSearchDecode2] = {
	"Alphabet" -> Characters["' abcdefghijklmnopqrstuvwxyz"],
	"BeamSize" -> 50,
	"PostProcessOptions" -> None
};

CTCBeamSearchDecode2[
	input_?NumericArrayQ /; (NumericArrayType[input] == "Real32" && ArrayDepth[input] == 2),
	logitq_?BooleanQ,
	blankIndex_Integer,
	mixtureModel_?AssociationQ, opts:OptionsPattern[]] :=
Scope[
	{maxTime,numClasses} = Dimensions@input;
	{alphabet, beamWidth, postopts} =
		OptionValue[{"Alphabet", "BeamSize", "PostProcessOptions"}];
	If[!validAlphabetQ[alphabet, numClasses] ||
		!validBeamSizeQ[beamWidth] ||
		!validMixtureModelQ[mixtureModel] ||
		!validPostProcessOptsQ[postopts], Return[$Failed]];
	logitqLM = True;
	eps = mixtureModel["EmptyString"];
	lmbeams = newCharStringUpdate[initBeams[beamMaker, eps, Length@alphabet], mixtureModel["LanguageModelFun"]];
	lmLogProbs = Flatten[beamCharLogProbs /@ lmbeams, 1];
	ok = lmctcDecoderInit[processBlank[blankIndex,numClasses],
			beamWidth, numClasses, mixtureModel["MixtureLevel"]];
	If[ok =!= Null, Message[CTCBeamSearchDecode2::interr, CTCBeamSearchDecode2]; Return[$Failed]];
	Do[
		currentStrings = lmctcDecoderStep[NumericArray[input[[t]], "Real32"],
				logitq, NumericArray[lmLogProbs, "Real32"], logitqLM, eps];
		If[Head[currentStrings] =!= Developer`DataStore, ok = False; Break[]];
		currentStrings = List@@currentStrings;
		(* select unchanged beams *)
		uncbeams = KeyTake[lmbeams, currentStrings];
		(* create newChar beams *)
		newcharStrs = Keys@KeyDrop[AssociationMap[Identity, currentStrings], Keys[uncbeams]];
		newcharbeams = Map[
			createNewCharBeam[
				beamMaker,
				Lookup[lmbeams, Key[Most@#]],
				#,
				eps]&,
			newcharStrs
		];
		newcharbeams = newCharStringUpdate[newcharbeams, mixtureModel["LanguageModelFun"]];
		(* combine unchanged and newChar beams *)
		lmLogProbs = Flatten[beamCharLogProbs /@ Normal@KeyTake[Join[Normal@uncbeams, newcharbeams], currentStrings], 1];
		minLen = Min[Length/@currentStrings];
		lmbeams = Join[Select[lmbeams, Length@beamString@# >= minLen&], newcharbeams];
		,
		{t, maxTime}
	];
	If[ok =!= Null, Message[CTCBeamSearchDecode2::interr, CTCBeamSearchDecode2]; Return[$Failed]];
	postopts = Replace[postopts, None -> <||>];
	topPaths = "UpTo" /. postopts /. Except[_?Internal`PositiveIntegerQ] -> 1;
	scores = ConstantArray[0.0, topPaths * 2]; (* include lm probability *)
	paths = lmctcDecoderFinish[topPaths, scores];
	If[!ListQ[paths], Message[CTCBeamSearchDecode2::interr, CTCBeamSearchDecode2]; Return[$Failed]];
	topN = Length@DeleteCases[paths[[;; topPaths]], 0];
	paths = TakeList[paths[[topPaths + 1 ;; ]], paths[[;; topN]]];
	paths += 1; (* convert from 0-indexed *)
	MapThread[Association["Text" -> alphabet[[Rest@#1]],
				"CTCLogProbability" -> #2, "LanguageModelLogProbability" -> #3]&,
		{paths, scores[[;; topN]], scores[[topPaths + 1 ;; topN - topPaths - 1]]}]
]

createNewCharBeam[_, _?MissingQ, _] := Nothing;
createNewCharBeam[beamMaker_, beam_, newStr_, eps_] /; (Last@newStr == eps) :=
	beamMaker[beamState[beam]][newStr, {0.}];
createNewCharBeam[beamMaker_, beam_, newStr_, eps_] :=
	beamMaker[beamState[beam]][newStr, beamCharLogProbs[beam][[Last@newStr]]];

validAlphabetQ[a_, numClasses_] /;
	(ListQ[a] && Length[a] == (numClasses - 1)) := True;
validAlphabetQ[a_, numClasses_] /;
	(ListQ[a] && Length[a] != (numClasses - 1)) := (Message[CTCBeamSearchDecode2::alphsize]; False);
validAlphabetQ[a_, _] := (Message[CTCBeamSearchDecode2::alph]; False);

validBeamSizeQ[bw_] /;
	(Internal`PositiveIntegerQ[bw]) := True;
validBeamSizeQ[bw_] := (Message[CTCBeamSearchDecode2::beamsize]; False);

validMixtureModelQ[m_(* already checked ?AssociationQ *)] /;
	(MatchQ[Lookup[m, {"LanguageModelFun", "EmptyString", "MixtureLevel"}],
		{_, _?IntegerQ, _?Internal`RealValuedNumericQ}]) := True;
validMixtureModelQ[m_] := (Message[CTCBeamSearchDecode2::invmixturemodel]; False);

validPostProcessOptsQ[p:(None|_?AssociationQ)] := True;
validPostProcessOptsQ[p_] := (Message[CTCBeamSearchDecode2::invpostops]; False);

beamMaker[state_] := Function[{str, p},
	Rule[str, {state, p}]];
beamString[beam_] := First[beam];
beamState[beam_Rule] := First[Last[beam]];
beamState[beam_List] := First[beam];
beamCharLogProbs[beam_Rule]:=Last[Last[beam]];
beamCharLogProbs[beam_List]:=Last[beam];

initBeams[beamMaker_, emptyStringLabel_, alphabetSize_] :=
	List[beamMaker[None][{emptyStringLabel}, 0.]];

processBlank[blankIndex_,numClasses_] :=
Module[{blankIndex2 = blankIndex},
	If[blankIndex < 0, blankIndex2 = (numClasses + 1 +  blankIndex)];
	If[numClasses < blankIndex2 < 1, Return[$Failed]];
	blankIndex2 -= 1 (* zero indexing *)
];

newCharStringUpdate[{}, lmFun_] := {};
newCharStringUpdate[beams_, lmFun_] :=
Scope[
	beamsStringin = beamString /@ beams;
	beamsCharUpdate = Last /@ beamsStringin;
	{lmCharLogProbas, newstates} = Transpose@lmFun[beamState /@ beams, Transpose[{beamsCharUpdate}]];
	MapThread[
		beamMaker[#1][#2, #3]&,
		{
			newstates
			,beamsStringin
			,lmCharLogProbas
		}
	]
]




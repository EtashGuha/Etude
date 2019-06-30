Package["NeuralNetworks`"]


PackageScope["SowNetConstraints"]

DeclareMethod[SowNetConstraints, Null, sowContainerConstraints, sowOperatorConstraints];

sowContainerConstraints[net_] := 
	If[ContainsVarSequenceQ[net["Inputs"]],
		ScanNodes[SowNetConstraints, net]];

sowOperatorConstraints[net_] :=
	If[ContainsVarSequenceQ[net["Inputs"]],
		ScanSubNets[SowNetConstraints, net]];


PackageScope["SowConstraint"]

SowConstraint[True|False] := Null;
SowConstraint[other_] := If[FreeQ[other, $Failed], SowBag[other]];


PackageScope["ReapNetConstraints"]

ReapNetConstraints[net_NetP] := 
	ReapBag @ SowNetConstraints @ net;


PackageScope["NetConstraintData"]

NetConstraintData[net_NetP] := ModuleScope[
	
	constraints = ReapNetConstraints[net];
	{equalities, inequalities} = SelectDiscard[constraints, MatchQ[_Equal]];
	(* separate equalities, which are relevant for determining the dep and indep
	ivars, from inequalities, which just require checks *)

	allVars = UniqueLengthVars[net];
	inputs = StripCoders @ Inputs[net];
	inputVars = UniqueLengthVars[inputs];
	internalVars = Complement[allVars, inputVars];

	(* find solutions to some of the input vars in terms of other input vars *)
	reducedConstraints = reduceConstraints[constraints, inputVars, internalVars];
	{equalities, inequalities} = SelectDiscard[reducedConstraints, MatchQ[_Equal]];

	eqRules = Rule @@@ equalities;
	depVars = Keys[eqRules];
	indepVars = Complement[inputVars, depVars];

	(* construct function to take the full set of input vars and take the
	independent subset *)
	toIndep = If[depVars === {}, Identity,
		Extract @ Position[inputVars, Alternatives @@ indepVars]
	];

	(* turn these constraints into code *)
	i = 1; slots = AssociationMap[SlotPart[i++]&, inputVars];
	checker = CreateFunction @ Map[
		Function[const, Hold[If][
			Hold[anyFalse[const]] /. slots, 
			Hold @ failConst[const, varToInput, varNames, #]
		]],
		reducedConstraints
	];

	varToInput = AssociationMap[FirstPosition[inputs, #][[1,1]]&, inputVars];
	varNames = AssociationThread[inputVars, LengthVarScope[net, FormatLengthVar /@ inputVars]];
	
	(* create a function that reconstructs the ivar values from a tuple of the
	indep vars (i.e. the bucket key) *)
	fromIndep = If[depVars === {}, Identity,
		CreateFunction[Operate[Hold, inputVars] /. eqRules /. slots /. (a_Rational * b_) :> Floor[a * b]]
		(* ^ Floor because a constarint of the form a = 2 b will induce b = a/2, and bucketing might make
		a odd. Floor got removed by reduceConstraints to make things solvable, we are basically adding it 
		back in here *) 
	];

	If[!FreeQ[reducedConstraints, False], ThrowFailure["netimpconst"]];

	{reducedConstraints, checker, toIndep, fromIndep}
];

General::netimpconst = "The net contains a set of constraints among sequence lengths that is impossible to satisfy for any input.";

anyFalse[False] := True;
anyFalse[True] := False;
anyFalse[list_List] := MemberQ[list, False];
anyFalse[e_] := MemberQ[Thread[e], False]; 
(* ^ thread so that {1,2,3}>4 works *)

General::dyndimconst = "``, which had ``, failed to satisfy the constraint ``, where ``."
failConst[const_, varToInput_, varNames_, lengths_] := Scope[
	If[MatrixQ[lengths], 
		(* find the batch element where the violation occurs *)
		vars = Keys[varNames];
		indices = Replace[
			$LastGeneratorIndices,
			(_Integer|Automatic) :> Range @ Length @ First @ lengths
		];
		Do[
			If[anyFalse[const /. AssociationThread[vars, lengths[[All, i]]]],
				badIndex = i;
				lengths = lengths[[All, i]];
				Break[]
			],
			{i, 1, Length @ indices}
		];
		name = Replace[$LastGeneratorName, None -> "Input"];
		form = If[IntegerQ[$LastGeneratorIndices], 
			StringForm["`` #`` of generated batch #``", name, indices[[badIndex]], $LastGeneratorIndices],
			StringForm["`` #``", name, indices[[badIndex]]]
		];
	,
		form = "Input";
	];
	lengthString = TextString @ CommaForm[
		MapThread[StringForm["`` = ``", #1, #2]&, {Values[varNames] /. s_Subscript :> ToString[s, StandardForm], lengths}],
		" and "
	];
	ThrowFailure["dyndimconst", 
		form,
		lengthString,
		If[$Notebooks && !$NNTestingMode, TraditionalForm, Identity] @ Simplify[const /. varNames],
		CommaForm[
			Map[
				StringForm["`` is the length of the \"``\" array", # /. varNames, # /. varToInput]&,
				SortBy[UniqueLengthVars[const], Position[vars, #]&]
			],
			" and "
		]
	]
];

reduceConstraints[constraints_, inputVars_, internalVars_] := Scope[
	all = Union[inputVars, internalVars];
	Flatten @ ToList @ separateSolutions @ Quiet @ Reduce[
		Join[DeleteDuplicates[constraints /. Floor -> Identity], Element[#, Reals]& /@ all],
		inputVars, internalVars
	]
];
(* it appears not to be documented, but specifying the internalVars to Reduce, just like for Solve,
gets rid of them *)

separateSolutions[e_And] := List @@ e;
separateSolutions[e_] := e;


PackageScope["FindMinimalLengthVarSettings"]

FindMinimalLengthVarSettings[net_] := Scope[
	constraints = ReapNetConstraints[net];
	allVars = UniqueLengthVars[constraints];
	inputVars = UniqueLengthVars[Inputs[net]];
	internalVars = Complement[allVars, inputVars];

	reducedConstraints = reduceConstraints[constraints, inputVars, internalVars];
	{equalities, inequalities} = SelectDiscard[reducedConstraints, MatchQ[_Equal]];

	inequalities = Join[inequalities, Thread[inputVars > 0]];
	inputVars /. Last @ Minimize[{Total[inputVars], inequalities}, inputVars, Integers]
];


(******************************************************************************)
(* Helper functions for layers to use                                         *)
(******************************************************************************)

PackageScope["SpatialConstraintGenerator"]

(* this is used by Convolution and Pooling *)
SpatialConstraintGenerator[firstDimFunc_] := Function @ Block[
	{ilen, olen, olen2, offset, minCond, lhs, rhs},
	ilen = First[#Parameters["$InputSize"], $Failed];
	If[IntegerQ[ilen], Return[Null, Block]];
	olen = First[#Parameters["$OutputSize"], $Failed];
	If[First[#Parameters["Stride"]] === 1,
		offset = firstDimFunc[#Parameters, 0];
		SowConstraint[ilen > -offset];
		SowConstraint[olen == ilen + offset]
	,
		olen2 = firstDimFunc[#Parameters, ilen];
		rhs = 1; lhs = Replace[Floor[z_] :> z] @ Replace[1 + z_ :> (rhs--; z)] @ olen2;
		(* ^ the 1+z only happens with Pooling *)
		minCond = Last @ Reduce[lhs >= rhs, ilen, Integers];
		(* ^ Last will strip off useless ilen :elem: Z *)
		SowConstraint[minCond];
		SowConstraint[olen == olen2]
	];
]


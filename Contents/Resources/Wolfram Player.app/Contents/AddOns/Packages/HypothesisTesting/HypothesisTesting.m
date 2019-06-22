(* ::Package:: *)

(*:Mathematica Version: 8.0 *)

(*:Package Version: 2.0 *)

(*:Name: Hypothesis Testing *)

(*:Context: HypothesisTesting` *)

(*:Title: Hypothesis Testing Related to the Normal Distribution *)

(*:Author:
  David Withoff (Wolfram Research), February 1990
*)

(*:History:
  Merged Statistics`HypothesisTests` and Statistics`ConfidenceIntervals`
  	into HypothesisTesting`, 2006, Darren Glosemeyer.
  Version 2.0 updated for move of hypothesis test functionality to the kernel,
  	2010, Darren Glosemeyer
*)

(*:Copyright: Copyright 1990-2010, Wolfram Research, Inc. *)

(*:Reference: Usage messages only. *)

(*:Summary:
This package provides hypothesis tests and confidence intervals based on 
elementary distributions derived from the normal distribution.  
Distributions represented are NormalDistribution, StudentTDistribution, 
ChiSquareDistribution, and FRatioDistribution.
*)

(*:Keywords: hypothesis test, significance level, confidence interval,
			confidence level *)

(*:Requirements: No special system requirements. *)

(*:Warning: None. *)

(*:Sources: Basic statistics texts. *)

BeginPackage["HypothesisTesting`"]

Unprotect[MeanTest,MeanDifferenceTest, 
	VarianceRatioTest, NormalPValue,StudentTPValue, ChiSquarePValue,
	FRatioPValue,OneSidedPValue, TwoSidedPValue,
	TwoSided, MeanCI, VarianceCI, MeanDifferenceCI, VarianceRatioCI, 
	StudentTCI, NormalCI, ChiSquareCI, FRatioCI, 
	KnownVariance, EqualVariances]

If[ Not@ValueQ[MeanTest::usage],
MeanTest::usage =
"MeanTest[list, mu0, options] returns a probability estimate (p-value) \
and other hypothesis test information for the relationship between the \
hypothesized population mean mu0 and Mean[list]."]

If[ Not@ValueQ[MeanDifferenceTest::usage],
MeanDifferenceTest::usage =
"MeanDifferenceTest[list1, list2, diff0, options] returns a probability \
estimate (p-value) and other hypothesis test information for the \
relationship between the hypothesized population mean difference diff0 \
and Mean[list1] - Mean[list2]."]

If[ Not@ValueQ[VarianceRatioTest::usage],
VarianceRatioTest::usage =
"VarianceRatioTest[numlist, denlist, ratio0, options] returns a probability \
estimate (p-value) and other hypothesis test information for the relationship \
between the hypothesized population variance ratio ratio0 and the the ratio \
Variance[numlist]/Variance[denlist]."]

If[ Not@ValueQ[NormalPValue::usage],
NormalPValue::usage =
"NormalPValue[teststat] returns the cumulative density beyond teststat \
for NormalDistribution[0,1]."]

If[ Not@ValueQ[StudentTPValue::usage],
StudentTPValue::usage =
"StudentTPValue[teststat, dof] returns the cumulative density beyond \
teststat for the StudentTDistribution, with dof degrees of freedom."]

If[ Not@ValueQ[ChiSquarePValue::usage],
ChiSquarePValue::usage =
"ChiSquarePValue[teststat, dof] returns the cumulative density beyond \
teststat for the ChiSquareDistribution with dof degrees of freedom."]

If[ Not@ValueQ[FRatioPValue::usage],
FRatioPValue::usage =
"FRatioPValue[teststat, numdof, dendof] returns the cumulative \
density beyond teststat for the FRatioDistribution with numdof numerator \
degrees of freedom and dendof denominator degrees of freedom."]

	(* Output names *)

If[ Not@ValueQ[OneSidedPValue::usage],
OneSidedPValue::usage =
"OneSidedPValue is used in the output of statistical hypothesis tests to \
identify the probability of observing a value further from the population \
parameter than is the sample parameter, and on the same side of the sampling \
distribution."]

If[ Not@ValueQ[TwoSidedPValue::usage],
TwoSidedPValue::usage =
"TwoSidedPValue is used in the output of statistical hypothesis tests to \
identify the probability of observing a value further from the population \
parameter than is the sample parameter, on either side of the sampling \
distribution."]

	(*  Options *)

If[ Not@ValueQ[TwoSided::usage],
TwoSided::usage =
"TwoSided is an option to statistical hypothesis tests and is used \
to request a two-sided test."]

If[ Not@ValueQ[FullReport::usage],
FullReport::usage = 
"FullReport is an option to statistical hypothesis tests and is used \
to indicate whether such information as the estimator, test statistic, and \
number of degrees of freedom should be included in the output."]

If[ Not@ValueQ[KnownVariance::usage],
KnownVariance::usage =
"KnownVariance is an option to statistical confidence interval and \
hypothesis test functions, and is used to specify the population variance \
whenever formulas corresponding to known population variance are to \
be used. The variance specification should be a list of variances if \
more than one different variance is to be specified. If only one \
variance is specified, it is used for all appropriate populations."]

If[ Not@ValueQ[EqualVariances::usage],
EqualVariances::usage =
"EqualVariances is an option to certain statistical confidence interval \
and hypothesis test functions and is used to specify that the appropriate \
unknown population variances are equal."]

If[ Not@ValueQ[MeanCI::usage],
MeanCI::usage =
"MeanCI[list, options] returns a list {lower, upper} \
representing a confidence interval for the population mean, using \
the entries in list as a sample drawn from the population."]

Options[MeanCI] = {ConfidenceLevel -> .95, KnownVariance -> None}

If[ Not@ValueQ[VarianceCI::usage],
VarianceCI::usage =
"VarianceCI[list] returns a list {lower, upper} representing \
a confidence interval for the population variance, using the entries in \
list as a sample drawn from the population."]

Options[VarianceCI] = {ConfidenceLevel -> .95}

If[ Not@ValueQ[MeanDifferenceCI::usage],
MeanDifferenceCI::usage =
"MeanDifferenceCI[list1, list2, options] returns a list {lower, upper} \
representing a confidence interval for the difference Mean[list1] \
- Mean[list2], using the entries in list1 as a sample from the \
first population and the entries in list2 as a sample from the \
second population."]

Options[MeanDifferenceCI] = {ConfidenceLevel -> .95, KnownVariance -> None,
    EqualVariances -> False}

If[ Not@ValueQ[VarianceRatioCI::usage],
VarianceRatioCI::usage =
"VarianceRatioCI[list1, list2, options] returns a list representing \
a confidence interval for the ratio of estimated population variances, \
Variance[list1]/Variance[list2], using the entries in list1 as a \
sample from the first population and the entries in list2 as a \
sample from the second population."]
 
Options[VarianceRatioCI] := {ConfidenceLevel -> .95}

If[ Not@ValueQ[StudentTCI::usage],
StudentTCI::usage =
"StudentTCI[mean, se, dof, ConfidenceLevel -> c] returns \
a list {lower, upper} representing a confidence interval at \
confidence level c for the population mean, based on a sample mean, \
its standard error se and dof degrees of freedom. This function is used \
by MeanCI when the population variance is estimated from \
the sample."]

Options[StudentTCI] = {ConfidenceLevel -> .95}

If[ Not@ValueQ[NormalCI::usage],
NormalCI::usage =
"NormalCI[mean, sd, ConfidenceLevel -> c] returns a list \
{lower, upper} representing a confidence interval at confidence level \
c for the population mean, based on the sample mean and its standard \
deviation. This function is used by MeanCI when the population \
variance is specified."]

Options[NormalCI] = {ConfidenceLevel -> .95}

If[ Not@ValueQ[ChiSquareCI::usage],
ChiSquareCI::usage =
"ChiSquareCI[var, dof, ConfidenceLevel -> c] returns a list \
{lower, upper} representing a confidence interval at confidence level c \
for the population variance, based on a sample with dof degrees of \
freedom and unbiased variance estimate var. This function is used \
by VarianceCI."]

Options[ChiSquareCI] = {ConfidenceLevel -> .95}

If[ Not@ValueQ[FRatioCI::usage],
FRatioCI::usage =
"FRatioCI[ratio, numdof, dendof, ConfidenceLevel -> c] returns \
a list {lower, upper} representing a confidence interval at confidence \
level c for ratio, where ratio is a ratio of sample variances, numdof is the \
number of numerator degrees of freedom, and dendof is the number of \
denominator degrees of freedom. This function is used by \
VarianceRatioCI."]

Options[FRatioCI] = {ConfidenceLevel -> .95}


	(*  Options  *)


Begin["`Private`"]

baseopts = {SignificanceLevel -> None, TwoSided -> False}

Options[MeanTest] = {FullReport -> False, KnownVariance -> None} ~Join~ baseopts

Options[MeanDifferenceTest] = {FullReport->False, KnownVariance -> None, 
    EqualVariances -> False} ~Join~  baseopts

Options[VarianceRatioTest] = {FullReport->False} ~Join~ baseopts

Options[NormalPValue] = {TwoSided -> False}

Options[StudentTPValue] = {TwoSided -> False}

Options[ChiSquarePValue] = {TwoSided -> False}

Options[FRatioPValue] = {TwoSided -> False}


VariancePair[{var0_}] := {var0, var0}

VariancePair[var0_List] := {var0[[1]], var0[[2]]} /; Length[var0] >= 2

VariancePair[var0_] := {var0, var0} /; Head[var0] =!= List

(* ==== Hypothesis tests for one mean ================================= *)

obsoleteMeanTestMessageFlag = True;

MeanTest[args___] :=
	(If[TrueQ[obsoleteMeanTestMessageFlag],
        	Message[General::obsfun, MeanTest, LocationTest];
        	obsoleteMeanTestMessageFlag = False];
    Block[{answer = iMeanTest[OptionExtract[{args}, MeanTest]]},
        answer /; answer =!= Fail
    ])


iMeanTest[{list_List?(Length[#] > 0 &), mu0_, optionlist_}] :=
    Block[{mean = Mean[list], delta, n, var0, rep, test, subopts,
		 out, nmean, ntest, newopts},
        delta = mean - mu0;
        n = Length[list];
        newopts=optionlist;
        {var0, rep} = ReplaceAll[{KnownVariance, 
            FullReport}, optionlist] /. Options[MeanTest];
	nmean = If[Precision[mean] === Infinity, N[mean], mean];
        If[var0 =!= None,
            test = delta/Sqrt[var0/n];
	    ntest = If[Precision[test] === Infinity, N[test], test];  
            subopts = FilterRules[Join[optionlist,Options[MeanTest]], 
            	{SignificanceLevel,TwoSided}];
            out = iNormalPValue[ntest, SignificanceLevel/.subopts,TwoSided/.subopts];
	    If[TrueQ[rep],
	     (* NOTE: 8/94: changed full report to give a valid list of
		 replacement rules and the distribution in symbolic form *)	
	     rep = FullReport ->
		 TableForm[{{nmean, ntest, NormalDistribution[0,1]}},
		TableHeadings -> {None, {"Mean","TestStat", "Distribution"}}];
	     out = Flatten[{rep, out}]
	    ];
	    out,	
        (* else, estimate the variance from the sample. *)
            var0 = Variance[list]/Length[list];
            If[Not[FreeQ[var0,Variance]],
                Message[MeanTest::novar]; Return[Fail] ];
            If[var0 == 0, Message[MeanTest::zerovar]];
            test = delta/Sqrt[var0];
	    ntest = If[Precision[test] === Infinity, N[test], test];
            subopts = FilterRules[Join[optionlist,Options[MeanTest]], 
				{SignificanceLevel,TwoSided}];
            out = iStudentTPValue[ntest, n-1, SignificanceLevel/.subopts,
            	TwoSided/.subopts];
	    If[TrueQ[rep],
	        (* NOTE: 8/94: changed full report to give a valid list of
		   replacement rules and the distribution in symbolic form *)	
		rep = FullReport ->
		 TableForm[{{nmean, ntest, StudentTDistribution[n-1]}},
		 TableHeadings -> {None, {"Mean", "TestStat", "Distribution"}}];
		out = Flatten[{rep, out}]
	    ];
	    out
	]
    ]
    

iMeanTest[badargs_] :=
    (If[badargs =!= Fail, Message[MeanTest::badargs]]; Fail)

MeanTest::badargs = "Incorrect number or type of arguments."

MeanTest::novar = "Unable to estimate variance from the sample."

MeanTest::zerovar = "Warning: Estimated variance is zero; subsequent \
results may be misleading."

(* ==== Hypothesis test for a difference of means ===================== *)

obsoleteMeanDifferenceMessageFlag = True;

MeanDifferenceTest[args___] := (
    If[TrueQ[obsoleteMeanDifferenceMessageFlag],
        	Message[General::obsfun, MeanDifferenceTest, LocationTest];
        	obsoleteMeanDifferenceMessageFlag = False];
    Block[{result = vMeanDifferenceTest[
                OptionExtract[{args}, MeanDifferenceTest]]},
        result /; result =!= Fail])

vMeanDifferenceTest[{list1_List, list2_List, diff1minus2_, optionlist_}] :=
    Block[{meandiff = Mean[list1] - Mean[list2], diff, var0, pval,newopts,delta},
        delta = meandiff - diff1minus2;
        newopts=optionlist;
        {var0} = ReplaceAll[{KnownVariance},
                newopts] /. Options[MeanDifferenceTest];
        iMeanDifferenceTest[list1, list2, delta, var0, Join[newopts,Options[MeanDifferenceTest]]]

    ]

vMeanDifferenceTest[badargs_] :=
    (If[badargs =!= Fail, Message[MeanDifferenceTest::badargs]]; Fail)

MeanDifferenceTest::badargs = "Incorrect number or type of arguments."

iMeanDifferenceTest[list1_, list2_, delta_, None, optionlist_] :=
    Block[{equalvar, rep, var1, var2, n1, n2, dof, pooledvar, test, subopts,
		 out, meandiff = Mean[list1]-Mean[list2], nmeandiff, ntest},
        {equalvar, rep} = {EqualVariances, FullReport} /. optionlist;
        {var1, var2} = {Variance[list1], Variance[list2]};
        {n1, n2} = {Length[list1], Length[list2]};
        If[TrueQ[equalvar],
            dof = n1 + n2 - 2;
            pooledvar = ((n1-1) var1 + (n2-1) var2) / dof;
            test = delta/Sqrt[pooledvar (1/n1 + 1/n2)],
        (* else *)
            pooledvar = var1/n1 + var2/n2;
            dof = pooledvar^2 /
                      ((var1/n1)^2/(n1-1) + (var2/n2)^2/(n2-1));
            test = delta/Sqrt[pooledvar]
        ];
	ntest = If[Precision[test] === Infinity, N[test], test];
        subopts = FilterRules[Join[optionlist, Options[MeanDifferenceTest]],
        	{SignificanceLevel,TwoSided}];
        out = iStudentTPValue[ntest, dof, SignificanceLevel/.subopts,TwoSided/.subopts];
        If[TrueQ[rep],
	   (* NOTE: 8/94: changed full report to give a valid list of
		 replacement rules and the distribution in symbolic form *)	
	   nmeandiff =
                 If[Precision[meandiff] === Infinity, N[meandiff], meandiff];
           rep = FullReport ->
		 TableForm[{{nmeandiff, ntest, StudentTDistribution[dof]}},
		 TableHeadings -> {None,
			{"MeanDiff", "TestStat", "Distribution"}}];
	   out = Flatten[{rep, out}]
	];
	out   
    ]

iMeanDifferenceTest[list1_, list2_, delta_, var0_, options___] :=
    Block[{rep, var1, var2, n1, n2, pooledvar, test, subopts, out,
		meandiff = Mean[list1]-Mean[list2], nmeandiff, ntest},
	rep = FullReport /. options;
        {var1, var2} = VariancePair[var0];
        {n1, n2} = {Length[list1], Length[list2]};
        pooledvar = var1/n1 + var2/n2;
        test = delta/Sqrt[pooledvar];
	ntest = If[Precision[test] === Infinity, N[test], test]; 
        subopts = FilterRules[Join[{options},Options[MeanDifferenceTest]],
         	{SignificanceLevel,TwoSided}];
        out = iNormalPValue[ntest, SignificanceLevel/.subopts,TwoSided/.subopts];
	If[TrueQ[rep],
	  (* NOTE: 8/94: changed full report to give a valid list of
		 replacement rules and the distribution in symbolic form *)	
	  nmeandiff =
		 If[Precision[meandiff] === Infinity, N[meandiff], meandiff];
	  rep = FullReport ->
		 TableForm[{{nmeandiff, ntest, NormalDistribution[0,1]}},
	  	 TableHeadings -> {None,
			{"MeanDiff","TestStat", "Distribution"}}];
	  out = Flatten[{rep, out}]
        ];
	out   
    ] /; var0 =!= None


(* ==== Hypothesis test for a variance ratio ========================== *)

obsoleteVarianceRatioMessageFlag = True;

VarianceRatioTest[args___] := (
    If[TrueQ[obsoleteVarianceRatioMessageFlag],
        	Message[General::obsfun, VarianceRatioTest, VarianceEquivalenceTest];
        	obsoleteVarianceRatioMessageFlag = False];
    Block[{result = iVarianceRatioTest[
                OptionExtract[{args}, VarianceRatioTest]]},
        result /; result =!= Fail])

iVarianceRatioTest[
        {numlist_List, denlist_List, ratio0_, optionlist_}] :=
    Block[{rep, test, numdof, dendof, subopts, out,
	   varRat = Variance[numlist]/Variance[denlist], nvarRat, ntest},
	rep = FullReport /.optionlist /. Options[VarianceRatioTest];
        test = varRat/ratio0;
	ntest = If[Precision[test] === Infinity, N[test], test];
        numdof = Length[numlist] - 1;
        dendof = Length[denlist] - 1;
        subopts = FilterRules[Join[optionlist,Options[VarianceRatioTest]], 
        	{SignificanceLevel,TwoSided}];
        out = iFRatioPValue[ntest, numdof, dendof, SignificanceLevel/.subopts,
        	TwoSided/.subopts];
     If[TrueQ[rep],
	(* NOTE: 8/94: changed full report to give a valid list of
		 replacement rules and the distribution in symbolic form *)	
	nvarRat = If[Precision[varRat] === Infinity, N[varRat], varRat];
        rep = FullReport ->
	  TableForm[{{nvarRat, ntest, FRatioDistribution[numdof, dendof]}},
          TableHeadings -> {None, {"Ratio", "TestStat", "Distribution"}}];
        out = Flatten[{rep, out}]
     ];
     out
    ]

iVarianceRatioTest[badargs_] :=
    CompoundExpression[ 
        If[badargs =!= Fail,
            Message[VarianceRatioTest::badargs, badargs]];
        Fail
    ]

VarianceRatioTest::badargs = "Incorrect arguments `1`."

(* ==== Basic hypothesis test functions ========================== *)

NormalPValue[test_, options___] :=
    Block[{twosided},
	  twosided = TwoSided /.
            Options[{options}] /. Options[NormalPValue];
      iNormalPValue[test, None, twosided]
    ]

iNormalPValue[test_, sig_, twosided_]:=
	Block[{pval = CDF[NormalDistribution[0,1], -Abs[test] ]},
	If[Precision[pval] === Infinity && NumberQ[N[pval]],
	   pval = N[pval]];
	resultOfTest[pval, 
                SignificanceLevel -> sig, TwoSided -> twosided]	
    ]

StudentTPValue[test_, dof_, options___] :=
    Block[{twosided},
	  twosided = TwoSided /.
            Options[{options}] /. Options[StudentTPValue];
      iStudentTPValue[test, dof, None, twosided]
    ]

iStudentTPValue[test_, dof_, sig_, twosided_]:=
	Block[{pval = CDF[StudentTDistribution[dof], -Abs[test] ]},
	If[Precision[pval] === Infinity && NumberQ[N[pval]],
	   pval = N[pval]];
	(* assume that a symbolic test value is real *)
	pval = pval /. {Sign[Abs[_]] -> 1, Power[Abs[x_], 2] -> x^2};
	resultOfTest[pval, 
                SignificanceLevel -> sig, TwoSided -> twosided]	
    ]

ChiSquarePValue[test_, dof_, options___] :=
    Block[{twosided},
	  twosided = TwoSided /.
            Options[{options}] /. Options[ChiSquarePValue];
      iChiSquarePValue[test, dof, None, twosided]
    ]/;DistributionParameterQ[ChiSquareDistribution[dof]]&&
    	If[NumericQ[test],Element[test,Reals],True]


iChiSquarePValue[test_, dof_, sig_, twosided_]:=
	Block[{pval},
	pval=If[TrueQ[test<=0], 
		N[0,Precision[{test,dof}]], 
		(* use direct GammaRegularized computations to avoid possible loss 
		   of precision in 1-CDF[...] *)
		Min[GammaRegularized[dof/2, test/2], GammaRegularized[dof/2, 0, test/2]]
		]/.Overflow[]->Indeterminate;
	If[Precision[pval] === Infinity && NumberQ[N[pval]],
	   pval = N[pval]];
	resultOfTest[pval, 
                SignificanceLevel -> sig, TwoSided -> twosided]	
    ]


FRatioPValue[test_, numdof_, dendof_, options___] := 
    Block[{twosided},
      	twosided = TwoSided /. Options[{options}] /. Options[FRatioPValue];
      	iFRatioPValue[test, numdof, dendof, None, twosided]
      	]/; DistributionParameterQ[FRatioDistribution[numdof, dendof]] &&
      		If[NumericQ[test],Element[test,Reals],True]


iFRatioPValue[test_, numdof_, dendof_, sig_, twosided_]:=
	Block[{pval,cdfval,n1, n2,testRational,
		prec = Precision[{test, numdof, dendof}]},
	pval = If[TrueQ[test <= 0], N[0,Precision[{test,numdof,dendof}]],
      		cdfval=CDF[FRatioDistribution[numdof,dendof],test];
      		Min[cdfval,1-cdfval]];
      	(* if pval is a small machine number, there may be a significant loss of precision;
      	   so compute a higher precision result and numericize appropriately *)
    If[MachineNumberQ[cdfval]&&pval<10^-10.,
      		{n1, n2, testRational} = Rationalize[{numdof, dendof, test}, 0];
          	pval=If[n2/(n2 + n1*testRational) < 1/2,
            		Min[BetaRegularized[n2/(n2 + n1*testRational), 1, n2/2, n1/2], 
            		1 - BetaRegularized[n2/(n2 + n1*testRational), 1, n2/2, n1/2]]
            		,
            		Min[BetaRegularized[n1*testRational/(n2 + n1*testRational), n1/2, n2/2], 
            		1 - BetaRegularized[n1*testRational/(n2 + n1*testRational), n1/2, n2/2]]]];
    pval = If[MemberQ[{MachinePrecision, Infinity}, prec] && NumberQ[N[pval]], 
      		N[N[pval, 20]], 
      		N[pval, prec]]/.Overflow[]->Indeterminate;
	resultOfTest[pval, 
                SignificanceLevel -> sig, TwoSided -> twosided]	
    ]                    

(*  Hypothesis test utilities  *)

resultOfTest[pval_, options___] :=
    Block[{sig, twosided, report, pval1 = pval},
        {sig, twosided} = {SignificanceLevel, TwoSided} /. Options[{options}];
        If[TrueQ[twosided],
		pval1 = 2 pval1;
		report = TwoSidedPValue -> pval1,
		report = OneSidedPValue -> pval1
	];
        If[sig =!= None,
            report = {report, SignificanceMessage[pval1, sig]}
        ];
        report
    ]
 
SignificanceMessage[pval_, level_] :=
    If[N[pval > level],
        "Fail to reject null hypothesis at significance level" -> level,
        "Reject null hypothesis at significance level" -> level,
         If[pval > level,
        	"Fail to reject null hypothesis at significance level" -> level,
        	"Reject null hypothesis at significance level" -> level
	 ]
    ]

OptionExtract[input_List, f_] :=
    Module[{n, opts, answer, known},
        For[n = Length[input], n > 0, n--,
            If[!OptionQ[input[[n]]], Break[]]
        ];
        answer = Take[input, n];
        opts = Options[input];
        known = Map[First, Options[f]];
        opts = Select[opts,
            If[MemberQ[known,First[#]], True,
                Message[f::optx, #, f]; False] &];
        AppendTo[answer, opts]
    ]


(* code previously contained in Statistics`ConfidenceIntervals` *)

(* Confidence Intervals for one Mean *)

MeanCI[list_List, options___] :=
    Block[{mean = Mean[list], n = Length[list], c, var0,newopts},
    	newopts=Flatten[{options}];
    	{c, var0} =
            {ConfidenceLevel, KnownVariance} /.
                newopts /. Options[MeanCI];
        If[!(0 <= N[c] <= 1), Message[MeanCI::clev, c]];
        If[var0 === None,
            Return[StudentTCI[mean, StandardDeviation[list]/Sqrt[Length[list]],
                                    n-1, ConfidenceLevel->c]],
            Return[NormalCI[mean, Sqrt[var0/n], ConfidenceLevel->c]]
        ]
    ]

MeanCI::clev = "Warning: ConfidenceLevel specification `1` is not \
in the expected range 0 <= ConfidenceLevel <= 1."

StudentTCI[mean_, se_, dof_, options___] :=
    Block[{c = ConfidenceLevel /. {options} /. Options[StudentTCI], h},
        If[N[se] <= 0, Message[StudentTCI::badse, se]];
        h = se Quantile[StudentTDistribution[dof], (1+c)/2];
        {mean-h, mean+h}]

StudentTCI::badse = "Warning: Standard error `1` is normally \
a positive quantity."

NormalCI[mean_, sd_, options___] :=
    Block[{c = ConfidenceLevel /. {options} /. Options[NormalCI], h},
        h = sd Quantile[NormalDistribution[0, 1], (1+c)/2];
        {mean-h, mean+h}]

(* Confidence Intervals for one Variance *) 

VarianceCI[list_List, options___] :=
    Block[{dof = Length[list] - 1, var = Variance[list],
	   c = ConfidenceLevel /. {options} /. Options[VarianceCI]},
        ChiSquareCI[var, dof, ConfidenceLevel->c]
    ]

ChiSquareCI[var_, dof_, options___] :=
    Block[{c = ConfidenceLevel /. {options} /. Options[ChiSquareCI],
           sumsq = dof var, lower, upper},
        upper = sumsq/Quantile[ChiSquareDistribution[dof], (1-c)/2];
        lower = sumsq/Quantile[ChiSquareDistribution[dof], (1+c)/2];
        {lower, upper}
    ]

(* Comparison of two Means *)

MeanDifferenceCI[list1_List, list2_List, options___] :=
    Block[{diff = Mean[list1] -  Mean[list2], var0, equalvar, c,newopts},
    	newopts=Flatten[{options}];
    	{var0, equalvar, c} = {KnownVariance, 
	 	EqualVariances, ConfidenceLevel} /.
            newopts /. Options[MeanDifferenceCI];
        mdInterval[list1, list2, diff, var0, equalvar, c]
    ]


mdInterval[list1_, list2_, diff_, None, equalvar_, c_] :=
    Block[{var1 = Variance[list1], var2 = Variance[list2],
	   n1 = Length[list1], n2 = Length[list2], dof, stderr},
        If[TrueQ[equalvar],
            dof = n1 + n2 - 2;
            stderr = Sqrt[(((n1-1) var1 + (n2-1) var2) / dof)*
                              (1/n1 + 1/n2)],
        (* else *)
            stderr = Sqrt[var1/n1 + var2/n2];
            dof = stderr^4 /
                  ((var1/n1)^2/(n1-1) + (var2/n2)^2/(n2-1))
        ];
        StudentTCI[diff, stderr, dof, ConfidenceLevel->c]
    ]

mdInterval[list1_, list2_, diff_, var0_, equalvar_, c_] :=
    Block[{var1, var2, stderr},
        {var1, var2} = VariancePair[var0];
        stderr = Sqrt[var1/Length[list1] + var2/Length[list2]];
        NormalCI[diff, stderr, ConfidenceLevel->c]
    ] /; var0 =!= None

VariancePair[{var0_}] := {var0, var0}

VariancePair[var0_List] := {var0[[1]], var0[[2]]} /; Length[var0] >= 2

VariancePair[var0_] := {var0, var0} /; Head[var0] =!= List

(* Ratio of two variances *)

VarianceRatioCI[numlist_List, denlist_List, options___] :=
    Block[{ratio = Variance[numlist]/Variance[denlist],
	   numdof = Length[numlist] - 1, dendof = Length[denlist] - 1,
	   c = (ConfidenceLevel /. {options} /. Options[VarianceRatioCI])},
        FRatioCI[ratio, numdof, dendof, ConfidenceLevel -> c]
    ]

FRatioCI[ratio_, numdof_, dendof_, options___] :=
    Block[{c = ConfidenceLevel /. {options} /. Options[FRatioCI],
	 lower, upper},
        lower = 1/Quantile[FRatioDistribution[numdof, dendof], (1+c)/2];
        upper = 1/Quantile[FRatioDistribution[numdof, dendof], (1-c)/2];
        {lower, upper} ratio
    ]


 
End[]
	
SetAttributes[
    {MeanTest,MeanDifferenceTest,
	VarianceRatioTest, NormalPValue,StudentTPValue, ChiSquarePValue,
	FRatioPValue,OneSidedPValue, TwoSidedPValue,
	TwoSided,MeanCI, VarianceCI, MeanDifferenceCI, VarianceRatioCI, 
	StudentTCI, NormalCI, ChiSquareCI, FRatioCI, 
	KnownVariance, EqualVariances},
    {Protected, ReadProtected}
];

EndPackage[]

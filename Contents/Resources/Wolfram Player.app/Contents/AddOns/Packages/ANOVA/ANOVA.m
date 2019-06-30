(* :Title: Analysis of Variance Package *)

(* :Author: Wolfram Research *)

(* :Copyright: Copyright 2000-2007 Wolfram Research, Inc *)

(* :Mathematica Version: 6.0 *)

(* :Package Version: 2.0 *)

(* :History: 
  Version 1.0, Ian Brooks and Erwann Rogard, Wolfram Research, 2000. 
  Version 1.1, Darren Glosemeyer, Wolfram Research, December 2004: 
     	Basically rewritten:
     	Model fitting is now always done with full rank design matrices and DesignedRegress.  
     	   This is more efficient and numerically stable than Regress with directly coded 
     	   variables, which results in a rank-deficient and potentially large design matrix.  
     	   This also allows for better handling of models with missing intermediate interaction terms.  
     	CellMeans and PostTests are now indexed by factor levels rather than sort order of 
     	   factor levels.  
     	Efficiency of quantiles for PostTests is improved by using a change of variable 
     	   and interpolation.
     	Messages and handling of degenerate cases are also improved.
  Version 1.2, Darren Glosemeyer, Wolfram Research, December 2005:
  		Improved efficiency of Quantile computations using new NIntegrate 
  		   Method option values.
  Version 2.0, Darren Glosemeyer, Wolfram Research, 2006:
  	 	moved from Statistics`ANOVA` to ANOVA`.
  Version 2.1, Darren Glosemeyer, Wolfram Research, 2008:
  	changed code to obtain ANOVA table results directly from LinearModelFit.
*)

(* :Summary:

This package provides least squares N-Way (any number of factors) Analysis of Variance,
allowing only for crossing designs.  Type 1 errors are used.

*)

(* :Context: ANOVA`*)

(* :Keywords: ANOVA, Analysis of Variance, Factors, Levels *)

(* :Requirements: No special system requirements *)

(* :Warnings:

Means in Infinite Precision but ANOVA not - inconsistency
Handle wrong results - as shown by NIST
Clean up unnecessary handling of non-interaction problems
Need to exit gracefully from bad regress call

Mainly Fixed effects does not supply estimates for Random or Mixed effect models or
give correct answers for n-way with interactions.

Calculation of Critical values for Dunnett's test have been optimized for speed at normal p values.
It may not give accurate values for p<0.001 due to numericising constants.
*)

(* :Discussion:

 Structure of the programme

=== Functions          ===
=== Function to export ===
	= Options      	=
	= Design        =
	= Regression  	=
	= ANOVA         =
	= Report	=
	= Diagnostics	=

Comments:
-Functions: perform temporary calculations called by the Function to Export
-Function to export: includes all the steps leading to the report
-Options: reads user options and converts them into options readable by the
	programme
-Design: converts input for model and factor into notation readable by the
	programme.
-Regression: creates all the input required for Regress i.e. data, model,
	vars  and performs the regression. Means are
	calculated this part if this option is activated by the user
-Anova: uses the regression ouput (specifically the sequential sum of squares
	and ANOVATable) to partition the SS to conform
	to the model specified by the user and generates related statistics (MS, F,
	PValue)
- Report: arranges the data in table format
- Diagnostics: checks for valid input by the user

*)

(* :Sources:
1 - J . D Jobson, Applied Multivariate Data Analysis, Volume I : Regression
and Experimental Design, 1991, Springer Verlag
2 - Shayle r . Searle, Linear Models For Unbalanced Data, 1987, Wiley & Sons
*)


BeginPackage["ANOVA`",
	{"HypothesisTesting`"}];

If[FileType[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"]]===File, 
   Select[FindList[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"],"ANOVA`"], 
   StringMatchQ[#,StartOfString~~"ANOVA`*"]&]//ToExpression; 
   ]; 


If[ Not@ValueQ[ANOVA::usage],
	ANOVA::usage =
"ANOVA[data, model, factors, Options] provides the partitioning of the variability of the \
response according to the combination of factors (the model) by which the sources of data \
can be categorized. This is performed using Least Square regression."]

If[ Not@ValueQ[CellMeans::usage],
CellMeans::usage=
"CellsMeans->True provides the means of the data for various classifications that are based \
on the factors specified in model, and their levels. CellsMeans are weighted averages, \
they are not necessarily identical to the expected cell means of the regression."]

If[ Not@ValueQ[Bonferroni::usage],
Bonferroni::usage=
"Bonferroni performs a Bonferroni post-test on the group means with Significance Level given by the \
Significance Level option."]

If[ Not@ValueQ[Tukey::usage],
Tukey::usage=
"Tukey performs a Tukey post-test on the group means with Significance Level given by the \
Significance Level option."]

If[ Not@ValueQ[Duncan::usage],
Duncan::usage=
"Duncan performs a Duncan post-test on the group means with Significance Level given by the \
Significance Level option."]

If[ Not@ValueQ[StudentNewmanKeuls::usage],
StudentNewmanKeuls::usage=
"StudentNewmanKeuls performs a StudentNewmanKeuls post-test on the group means with Significance Level given by the \
Significance Level option."]

If[ Not@ValueQ[Dunnett::usage],
Dunnett::usage=
"Dunnett performs a two-sided Dunnett post-test on the group means with Significance Level given by the \
Significance Level option. The first group is taken to be the reference group. The test is \
only valid when the number of samples in the test groups are approximately the same as the number \
of samples in the reference group."]

If[ Not@ValueQ[PostTests::usage],
PostTests::usage=
"PostTests->{name} provides the post-tests specified by name. These are Bonferroni, Tukey, StudentNewmanKeuls, and Duncan."]

(* Error Messages  *)

ANOVA::arg1="The 1st argument has unequal columns or rows."

ANOVA::arg11="The last column of the 1st argument (the response data ) \
contains non numeric elements."

ANOVA::arg2="The factor(s) `1` in the 2nd argument is(are) not listed in the \
3rd argument."

ANOVA::arg3="The 3rd argument has `1` elements; `2` elements are expected."

ANOVA::arg31="Times is Head to one or several elements in the 3rd argument. \
Times should be used only for expressing interactions in the second \
argument."

ANOVA::oneway="The number of columns `1` in the argument is not valid. The number of columns must be 2 for a one-way ANOVA."

(* since the variables are categorical, only the precision of the last column of the data matters,
   so use a modified precw message *)
ANOVA::precw="The precision of the last column of the data (`1`) is less than WorkingPrecision (`2`)."

ANOVA::dunnettprecision="Calculation of critical values for Dunnett's test are optimized for \
speed at p values >0.001 and may suffer from loss of accuracy below this level."

ANOVA::zerodf="The model contains `1` `2` zero degrees of freedom. `3` will be omitted from the ANOVA table."

ANOVA::errdf="The model contains no degrees of freedom for Error. The data are insufficient to obtain a meaningful analysis of variance." 

ANOVA::nottest="`1` is not a valid PostTests entry, and will be ignored. Valid PostTests entries are Bonferroni, Tukey, StudentNewmanKeuls, Duncan, and Dunnett."

ANOVA::testdf="Unable to compute PostTests. The degrees of freedom due to Error must be greater than 0 to compute PostTests."

Unprotect[ANOVA];
Unprotect[CellMeans];
Unprotect[PostTests];
Unprotect[Bonferroni];
Unprotect[Tukey];
Unprotect[Duncan];
Unprotect[StudentNewmanKeuls];
Unprotect[Dunnett];

Begin["`Private`"]

Options[ANOVA]=
{CellMeans->True,WorkingPrecision->MachinePrecision,PostTests->{},SignificanceLevel->0.05}

(* === Functions ===*)

RowJoin[data:{___List}..] := Apply[Join, Transpose[{data}], {1}] /;
				Equal @@ Map[Length, {data}]

(* Functions for computing cell means *)

CategoryMap[fun_,mat_?MatrixQ,col1_Integer?Positive,crit_Integer?Positive]:= 
  CategoryMap[fun,mat,col1,{crit}]
    
CategoryMap[fun_, mat_?MatrixQ, col1_Integer?Positive, totcrit : {_Integer?Positive ..}] := 
 Module[{cols, spltset}, 
 	cols = mat[[All, Join[totcrit, {col1}]]];
  	spltset = Split[Sort[cols], (Most[#1] == Most[#2] &)];
  	Map[{Most[#[[1]]],fun[#[[All,-1]]]}&,spltset]
  	]

meantable[data_, model_, factors_,prec_] := Block[{modelterms, newdata,conversionrule, val, xx, yy, zz, catmap},
    newdata=data;
    modelterms = (model /. Times -> List) /. MapThread[Rule[#1, #2] &, 
    	{factors, Range[Length[factors]]}];
    catmap[yy_] := CategoryMap[N[Mean[#],prec]&, newdata, Length[newdata[[1]]], yy];
    conversionrule[val_] := (List[List[xx__], zz_?NumericQ] :> 
    	{Apply[Times, MapThread[((factors[[#1]])[#2]) &, {Flatten[{val}], {xx}}]], zz});
    Join[{{"All", N[Mean[data[[All, -1]]],prec]}}, Apply[Join, Map[With[{res = catmap[#]},
              	res /. conversionrule[#]] &, modelterms]]]]

celltotal[data_, model_, factors_] := Block[{modelterms}, 
    modelterms = (model /. Times -> List) /. MapThread[Rule[#1, #2] &, 
    	{factors, Range[Length[factors]]}];
    Apply[Join,Map[CategoryMap[Length, data, Length[data[[1]]], #] &, modelterms]]]

               	
(*=== Function to export === *)

(* one-way ANOVA given by single argument *)
ANOVA[dataInput_List,opts___?OptionQ] :=
   Module[{result = iANOVA[dataInput,opts]},
       result/;(result =!= $Failed)
   ]

(* error checking for one-way, single argument case *)
iANOVA[dataInput_List,opts___?OptionQ]:=Block[{datamatQ,len1,matQ},
   len1=If[Length[dataInput]>0,Length[dataInput[[1]]],0];
   Which[
	(datamatQ=MatrixQ[dataInput]&&len1>0&&VectorQ[dataInput[[All,-1]],NumericQ])&&
		len1===2
	,
	iANOVA[dataInput,{"Model"},{"Model"},opts]
	,
	!(matQ=MatrixQ[dataInput])
	,
	Message[ANOVA::arg1];$Failed
	,
	matQ&&len1=!=2
	,
	Message[ANOVA::oneway,len1];$Failed
	,
	matQ&&!datamatQ
	,
	Message[ANOVA::arg11];$Failed
	,
	True
	,
	$Failed]]
	

(* general three argument ANOVA *)
ANOVA[dataInput_List,model_List,factors_List,opts___?OptionQ] :=
   Module[{result = iANOVA[dataInput,model,factors,opts]},
       result/;(result =!= $Failed)
   ]


(* main function for computing ANOVA results *)
iANOVA[dataInput_List,model_List,factors_List,opts___?OptionQ]:=
  Block[{oneway, meanvals, newdata=dataInput, newmodel,  regressterms, regressvars, 
  	matrank, desmat, factortype, df, regoutput, seqSOS, regANOVA, modelsssq, 
  	errorrow, totalrow, lANOVAMS, lANOVAF, lANOVAP, regeffects, bMean, iPrecision, 
  	pTests, siglevel, lANOVAReportFormatted,lMeansReportFormatted, lDesignModelLevels, 
  	cellCounts, maineffects, countVals, pTestResults, partitionedTestInfo, tablemodelterms, temporarymodelvar},

(*=Options=*)

	{bMean,iPrecision,pTests,siglevel}={CellMeans,WorkingPrecision,PostTests,SignificanceLevel}
		/.Flatten[{opts,Options[ANOVA]}];
	(* allow for single post test not given in list *)
	pTests=Flatten[{pTests}];

(* issue a message if the WorkingPrecision is greater than the response precision;
   since the variables are categorical, only the precision of the response is relevant *)
	Apply[If[#<iPrecision||(#===MachinePrecision&&iPrecision=!=#),Message[ANOVA::precw,#,iPrecision]]&,
		{Precision[dataInput[[All,-1]]]}];

(* Convert from alpha to 1-alpha *)

	siglevel=1-siglevel;

	oneway=(Length[factors]===1);

	newdata[[All,-1]]=If[iPrecision===MachinePrecision,N[dataInput[[All,-1]]],
		N[Rationalize[dataInput[[All,-1]],0],iPrecision]];
		
(* include all interactions if All is in model *)
	newmodel=If[!FreeQ[model,All]
		,
		Map[Apply[Times, #] &, Rest[Subsets[Select[model, (# =!= All &)]]]]
		, 
		model
		];

(* compute cell mean table; this is also quickly determines the factor cells and finds 
   empty cells, so do this regardless of whether or not CellMeans->True *)
   
   	meanvals = meantable[newdata, newmodel, factors, iPrecision];


	(* if the variation in the response is very small compared to the magnitude of the response 
	   (e.g. NIST example smls09) and machine precision is used, subtract out the overall mean;
	   rationalize, subtract, and then numericize to avoid introducing additional noise;
	   do not use this approach with variable precision WorkingPrecision because it will 
	   artifically increase the precision of the data and potentially give "significant" 
	   digits that are not significant *)
	
	If[iPrecision===MachinePrecision,
	   With[{resp = newdata[[All, -1]]},
		If[!PossibleZeroQ[Mean[resp]]&&
		      Max[Abs[(resp - Mean[resp])/Mean[resp]]] < 10^-10
		   , 
		   With[{newresp = Rationalize[resp, 0]}, 
		     newdata[[All, -1]] = N[newresp - Mean[newresp]]]
		   ]]];

	
	regANOVA=Quiet[LinearModelFit[newdata, newmodel/.{"Model"->temporarymodelvar}, 
		factors/.{"Model"->temporarymodelvar}, IncludeConstantBasis->True,
		WorkingPrecision->iPrecision, NominalVariables->All]
		,
		(* message about reduced rank is always to be expected from LinearModelFit for ANOVA models, 
	   	   but should not be issued from ANOVA *)
		LinearModelFit::rank];
	If[!FreeQ[regANOVA,LinearModelFit],
		Return[$Failed]];
	(* errdf message should come from ANOVA, so quiet it here *)
	{tablemodelterms,regANOVA,df}=Quiet[regANOVA[{"ANOVATable","ANOVATableEntries","ANOVATableDegreesOfFreedom"}],
			{FittedModel::precw, FittedModel::errdf}];
	tablemodelterms=Rest[tablemodelterms[[1, 1]]][[All, 1]];
	errorrow=regANOVA[[-2]];
	
	If[errorrow[[1]] === 0
	    ,
	    (*catch and issue warning for degenerate case of 0 DF for Error*)
	    Message[ANOVA::errdf]];
	(* issue message when effects or interactions are dropped;
	   tablemodelterms contains "Error" and "Total", so reduce count by 2 *)
	With[{diff=Length[newmodel]-(Length[tablemodelterms]-2)},
		Which[diff===1, 
			Message[ANOVA::zerodf, 1, "effect or interaction that has", "This term"],   
        	diff>1, 
        	Message[ANOVA::zerodf, diff, "effects or interactions that have", "These terms"],   
            True, 
            Null]   
        ]; 
    
(*=Report=*)
	lANOVAReportFormatted=Rule[ANOVA,regANOVA];

lANOVAReportFormatted=Rule[ANOVA,
		TableForm[regANOVA,
		    TableHeadings -> {tablemodelterms/.{temporarymodelvar->"Model"},
		    {"DF", "SumOfSq", "MeanSq", "FRatio", "PValue"}}]];
		
(*=Means=*)

	If[bMean,
		lMeansReportFormatted = Rule[CellMeans,TableForm[meanvals]];

		lANOVAReportFormatted={lANOVAReportFormatted,lMeansReportFormatted}];


(*=Post Tests=*)
	With[{invalidtests=UnsortedComplement[pTests,{Bonferroni,Tukey,Duncan,StudentNewmanKeuls,Dunnett}]},
		If[Length[invalidtests]=!=0
		  , 
		  Map[Message[ANOVA::nottest,#]&,invalidtests];
		  pTests=UnsortedComplement[pTests,invalidtests]]];
	
	If[pTests =!= {}&&errorrow[[1]]<=0, Message[ANOVA::testdf]];

	If[pTests =!= {}&&errorrow[[1]]>0
	  ,
	  (* each test relies on the mean values for each main effect level, the number of levels for each main effect,
	     and an indexed mapping back to the original factor level values *)
	  cellCounts=celltotal[dataInput, maineffects=Select[newmodel,(Head[#]===Symbol||#==="Model"&)], factors];
	  
	  (* the number of levels per effect is one more than the number of degrees of freedom *)
	  lDesignModelLevels=Take[df,Length[maineffects]]+1;
	  countVals = Join[{0}, Rest[FoldList[Plus, 0, lDesignModelLevels]]];
	  
	  (* group the means, cell counts and indexed factor level mappings by factor *)
	  partitionedTestInfo=Transpose[Table[{
	  	Take[Rest[meanvals[[All, -1]]], {1 + countVals[[i]], countVals[[i + 1]]}], 
   		Take[cellCounts[[All, -1]], {1 + countVals[[i]], countVals[[i + 1]]}],
   		Thread[Rule[Range[countVals[[i + 1]] - countVals[[i]]], Take[cellCounts[[All, 1, 1]]
   		, {1 + countVals[[i]], countVals[[i + 1]]}]]]}, {i, Length[countVals] - 1}]];
   	  
   	  pTests=Flatten[{pTests}];
   	  pTestResults={};
   	  
   	  If[MemberQ[pTests, Bonferroni]
   	  	,
		pTestResults=Append[pTestResults,Thread[Rule[Bonferroni,
		MapThread[(Bonferroni[#1, errorrow[[1]], #2,errorrow[[3]], siglevel]/.#3)&, partitionedTestInfo]]]];
		];

	  If[MemberQ[pTests, Tukey]
		,
		pTestResults=Append[pTestResults,Thread[Rule[Tukey,
		MapThread[(Tukey[#1, errorrow[[1]], #2,errorrow[[3]], siglevel]/.#3)&,partitionedTestInfo]]]];
		];

	  If[MemberQ[pTests, Duncan]
		,
		pTestResults=Append[pTestResults,Thread[Rule[Duncan,
		MapThread[(Duncan[#1, errorrow[[1]], #2,errorrow[[3]], siglevel]/.#3)&,partitionedTestInfo]]]];
		];

	  If[MemberQ[pTests, StudentNewmanKeuls]
		,
		pTestResults=Append[pTestResults,Thread[Rule[StudentNewmanKeuls,
		MapThread[(StudentNewmanKeuls[#1, errorrow[[1]], #2,errorrow[[3]], siglevel]/.#3)&,partitionedTestInfo]]]];
		];

	  If[MemberQ[pTests, Dunnett]
		,
		pTestResults=Append[pTestResults,Thread[Rule[Dunnett,
		MapThread[(Dunnett[#1, errorrow[[1]], #2,errorrow[[3]], siglevel]/.#3)&,partitionedTestInfo]]]];
		];

	(* Replace {} by {""} so that an empty list shows up in the TableForm   *)

	lANOVAReportFormatted={lANOVAReportFormatted,
		Rule[PostTests, Thread[Rule[If[oneway,factors,maineffects],
    		Thread[TableForm[Transpose[pTestResults] /. {Rule -> List, {} -> {""}},
    		TableSpacing -> {3, 1},TableDepth -> 2]]]]]};
	];
	Flatten[lANOVAReportFormatted]
	]

(*=Diagnostics=*)


iANOVA[dataInput_List,model_List,factors_List,opts___?OptionQ]:=
    (Message[ANOVA::arg1];Return[$Failed])/;(MatrixQ[dataInput]===False)

iANOVA[dataInput_List,model_List,factors_List,opts___?OptionQ]:=
    (Message[ANOVA::arg2,Complement[DeleteCases[Flatten[model /. Times -> List], All], factors]];
    Return[$Failed])/;Complement[DeleteCases[Flatten[model /. Times -> List], All], factors] =!= {}

iANOVA[dataInput_List,model_List,factors_List,opts___?OptionQ]:=
    (Message[ANOVA::arg3,Length[factors],Length[dataInput[[1]]]-1];Return[$Failed])/;
      Length[factors]!=(Length[dataInput[[1]]]-1)

iANOVA[dataInput_List,model_List,factors_List,opts___?OptionQ]:=
    (Message[ANOVA::arg31];Return[$Failed])/; MemberQ[Map[Head, factors], Times]

iANOVA[dataInput_List,model_List,factors_List,opts___?OptionQRule]:=
    (Message[ANOVA::arg11];Return[$Failed])/; !VectorQ[dataInput[[All, -1]], NumericQ]


UnsortedUnion[x_] := Module[{f,g}, f[y_] := (f[y] = Sequence[]; y); g= Map[f,x];Clear[f];g]

UnsortedComplement[ee_, ff_] := Block[
  {origlist = DeleteCases[ee, Apply[Alternatives, ff]], finallist = {}, i = 1, len}
  ,
  len = Length[origlist];
  While[i <= len, If[Intersection[{origlist[[i]]}, finallist] === {}, 
        finallist = Join[finallist, {origlist[[i]]}]]; i++];
  finallist]
  
UnsortedIntersection[ee_, ff_] := Block[
  {origlist = Cases[ee, Apply[Alternatives, ff]], finallist = {}, i = 1, len}
  , 
  len = Length[origlist];
  While[i <= len, If[Intersection[{origlist[[i]]}, finallist] === {}, 
        finallist = Join[finallist, {origlist[[i]]}]]; i++];
  finallist]
  
  
(* Handle unbalanced post-tests *)

Bonferroni[means_List, df_Integer, n_List, mse_, alpha_] :=
  Module[{nGroups = Length[means], alpha0, criticalValue,
      criticalDifference, sigGroups},

    alpha0 = (1-alpha)/(nGroups(nGroups - 1)/2);
    criticalValue = Quantile[StudentTDistribution[df], 1 - (alpha0/2)];
    criticalDifference = criticalValue Sqrt[mse];
	sigGroups =
    	Partition[Flatten[Table[If[Abs[means[[i]] - means[[j]]] >
          criticalDifference Sqrt[1/n[[j]] + 1/n[[i]]], {j, i}, {}], {i,1, Length[means]},
          {j, 1, i}]], 2];

	If[Length[sigGroups] > 1, sigGroups,Flatten[sigGroups]]
    ]

Tukey[means_List, df_Integer, n_List, mse_, alpha_] :=
  Module[{ criticalValue,criticalDifference, sigGroups},

    Internal`DeactivateMessages[
    	criticalValue = Quantile[StudentizedRangeDistribution[df,Length[n]], alpha],
   NIntegrate::slwcon];

      criticalDifference = criticalValue Sqrt[mse];

	sigGroups =
    	Partition[Flatten[Table[If[Abs[means[[i]] - means[[j]]] >
          criticalDifference Sqrt[.5 (1/n[[j]] + 1/n[[i]])], {j, i}, {}], {i,1, Length[means]},
          {j, 1, i}]], 2];

          If[Length[sigGroups] > 1, sigGroups,Flatten[sigGroups]]
    ]


Duncan[means_List, df_Integer, n_List, mse_, alpha_] :=
  Module[{criticalDifference},

   	sigGroups={};
	criticalDifference =  Sqrt[mse];

    Internal`DeactivateMessages[

		result=RangeLevel[{Sort[means]}, df, n, criticalDifference, alpha, Duncan],

    NIntegrate::slwcon];

   If[Length[sigGroups] > 1,
  	Union[Map[Sort,Partition[Flatten[Map[Position[means, #] &, Flatten[sigGroups]]], 2]]],
  	Sort[Flatten[Map[Position[means, #] &, Flatten[sigGroups]]]]]
]


StudentNewmanKeuls[means_List, df_Integer, n_List, mse_, alpha_] :=
  Module[{criticalDifference},

	sigGroups={};
   	criticalDifference =  Sqrt[mse];

    	Internal`DeactivateMessages[
		result=RangeLevel[{Sort[means]}, df, n, criticalDifference, alpha, StudentNewmanKeuls],
	NIntegrate::slwcon];

   If[Length[sigGroups] > 1,
  	Union[Map[Sort,Partition[Flatten[Map[Position[means, #] &, Flatten[sigGroups]]], 2]]],
  	Sort[Flatten[Map[Position[means, #] &, Flatten[sigGroups]]]]]
]

Dunnett[means_List, df_Integer, n_List, mse_, alpha_] :=
  Module[{nGroups = Length[means]-1, criticalDifference, sigGroups},

If[alpha <= 0.999,
    criticalDifference = Quantile[DunnettDistribution[df, nGroups],alpha] * Sqrt[mse],
    Message[ANOVA::dunnettprecision];
    criticalDifference = Quantile[DunnettDistribution[df, nGroups],alpha] * Sqrt[mse]
    ];

	sigGroups =
    	Flatten[Table[If[Abs[means[[i]] - means[[1]]] >
          criticalDifference Sqrt[1/n[[1]] + 1/n[[i]]], {i}, {}], {i,2, Length[means]}
          ]];

	If[Length[sigGroups] > 1, sigGroups,Flatten[sigGroups]]
    ]

RangeTest[means_List,criticalValue_] :=
  	If[Last[means] - First[means] > criticalValue, sigGroups = Append[sigGroups,
        {First[means],Last[means]}];
    	Partition[means, Length[means] - 1, 1], Dead[means]]

RangeLevel[means_Dead, df_Integer, n_List, criticalDifference_, alpha_, method_] = means

RangeLevel[means_List, df_Integer, n_List, criticalDifference_, alpha_, method_] :=
  Module[{nmeans = Length[First[means]],pvalue,res},
  If[method===StudentNewmanKeuls,pvalue=alpha, pvalue = (1 - (1 - alpha)^(nmeans - 1))];
  
	  criticalValue = criticalDifference Sqrt[.5 (1/First[n] + 1/Last[n])] *
	   Quantile[StudentizedRangeDistribution[ df, nmeans],pvalue];


	  res = Flatten[Map[RangeTest[#, criticalValue] &, means],1];


		If[MemberQ[Head /@ res, List] && Length[First[Cases[res, _List]]]>1,
    		RangeLevel[DeleteCases[res, _Dead],df,n,criticalDifference,alpha,method], Sort[sigGroups]
		]
	]

(* interpolation for quantile of NormalDistribution[0,1] used by Quantile[StudentizedRangeDistribution[]] 
   and Quantile[DunnettDistribution[]] for efficient evaluation with sufficient precision and accuracy 
*)
phiInverse = Interpolation[
  Map[With[{inverf = InverseErf[0, -1 + 2#]}, 
  	{{#}, 
  	Sqrt[2]*inverf, E^inverf^2*Sqrt[2*Pi], 2*Sqrt[2]*E^(2*inverf^2)*inverf*Pi,
  	 Sqrt[2]*(2*E^(3*inverf^2)*Pi^(3/2) + 8*E^(3*inverf^2)*inverf^2*Pi^(3/2)), 
  	 Sqrt[2]*(28*E^(4*inverf^2)*inverf*Pi^2 + 48*E^(4*inverf^2)*inverf^3*Pi^2),
  	 Sqrt[2]*(28*E^(5*inverf^2)*Pi^(5/2) + 368*E^(5*inverf^2)*inverf^2*Pi^(5/2) + 
  	 	384*E^(5*inverf^2)*inverf^4*Pi^(5/2)), 
  	 Sqrt[2]*(1016*E^(6*inverf^2)*inverf*Pi^3 + 5216*E^(6*inverf^2)*inverf^3*Pi^3 + 
  	 	3840*E^(6*inverf^2)*inverf^5*Pi^3)}] &,
    (* Sqrt[2]*InverseErf[0, -1 + 2x] changes rapidly near x == 0 and x == 1 so more 
       points are needed in the tails to get a good interpolation*)
  Join[Flatten[Table[Range[9]*10.^zz, {zz, -10, -3}]], Table[zz, {zz, .01, .99, .01}], 
  	Flatten[Table[1 - Reverse[Range[9]]*10.^zz, {zz, -3, -10, -1}]]]]];


(* the following Quantile[]s take advantage of the change of variable 
   u->phiInverse[z] to transform the integration limits {u,-Infinity,Infinity} to {z,0,1},
   taking advantage of some knowledge of the integrand *)
   
StudentizedRangeDistribution/: Quantile[StudentizedRangeDistribution[n_, r_], p_]:=
Module[{nstuff = n^(n/2)/(Gamma[n/2]*2^(n/2 - 1)), nfunc,z,x,q},
	(*This function is somewhat slow, but should be accurate.  Further speed up might 
	  be possible using the change of variable x -> Quantile[ChiSquareDistribution[n], y]
	  if numeric evaluation of InverseGammaRegularized[n/2, 0, y] was faster.
	*)
  nfunc[q_?NumberQ] := nstuff*NIntegrate[x^(n - 1)*Exp[((-n)*x^2)/2]*r*(z - Erfc[((q*x) - 
	phiInverse[z])/Sqrt[2]]/2)^(-1 + r), {z, 0, 1}, {x, 0, Infinity}, 
	Method -> {"MultiDimensionalRule", "Generators" -> 5, "SymbolicProcessing" -> 0},
	SingularityDepth->Infinity,	PrecisionGoal->4];
  Abs[q /. FindRoot[nfunc[q] == p, {q, 3, 6}, MaxIterations -> 500,
  	PrecisionGoal->3,AccuracyGoal->4]]]


DunnettDistribution/: Quantile[DunnettDistribution[n_, r_], p_, opts___]:=
Module[
  {width = 10, peak = Sqrt[n - 1], sn = Sqrt[n], nfunc, const = (2^(1 - r))/Gamma[n/2],iAccGoal, iWp,xx,z,q}, 
  {iAccGoal, iWp} = {AccuracyGoal, WorkingPrecision} /. Flatten[{opts, Options[ANOVA], AccuracyGoal -> 3}];
  nfunc[q_?NumberQ] := NIntegrate[
  	const*E^(-0.34657359027997264*n - 0.5*xx^2)*xx^(-1. + n)*(Erf[(q*xx - phiInverse[z])/Sqrt[2]] + 
  	Erf[(q*xx + phiInverse[z])/Sqrt[2]])^r, {z, 0, 1}, {xx, Max[0, peak - width], peak + width}, 
      	Method -> {"GlobalAdaptive", "SingularityHandler" -> {"IMT", "TuningParameters" -> 1}, 
      		"SymbolicProcessing" -> 0},
      	AccuracyGoal -> iAccGoal, PrecisionGoal -> Infinity] - p;
  (q /. FindRoot[nfunc[q], {q, 1.5/sn, 6.5/sn}, MaxIterations -> 500, 
       	WorkingPrecision -> iWp, AccuracyGoal -> iAccGoal + 1])*0.7071067811865475*Sqrt[n]]


SetAttributes[ANOVA,ReadProtected];

End[]
Protect[PostTests];
Protect[Bonferroni];
Protect[Tukey];
Protect[Duncan];
Protect[StudentNewmanKeuls];
Protect[Dunnett];
Protect[CellMeans];
Protect[ANOVA];
EndPackage[]

(* ::Package:: *)

Begin["Tools`MathematicaFormulas`FormulaData`Private`"]

(* eliminate? *)
MathematicaFormula;
System`Entity;

(* Load the downvalues.m file. *)
$ProtectedSymbols = {
	MathematicaFormula, FormulaData, FormulaLookup, 
	RequiredPhysicalQuantities, ExcludedPhysicalQuantities,
	PlanckRadiationLaw
};
	
Unprotect@@$ProtectedSymbols;
$QueryTimeout = Automatic;
$tag = "FormulaDataCatchThrowTag";

downvalues = Get["FormulaData`downvalues`"]
DownValues[MathematicaFormula] = downvalues[[1]]
	
WolframAlpha;(*load WolframAlphaClient`*)

Attributes[APICompute] = {HoldAll};
APICompute[type_, args_] := Internal`MWACompute[type,args,"ContextPath" -> {"Internal`MWASymbols`", "System`", 
  "Tools`MathematicaFormulas`FormulaData`Private`"},"MessageHead"->FormulaLookup]

decamel[input_String] := 
 StringTrim[
  StringReplace[input, 
   MapThread[
    Function[{u, l}, 
     Rule[u, StringJoin[" ", l]]], {CharacterRange["A", "Z"], 
     CharacterRange["a", "z"]}]]]

localParse[input_String] := 
 Module[{data, n = Floor[StringLength[input]/2]}, 
  data = If[MatchQ[#, _List], StringJoin[Riffle[#, ":"]], #] & /@ 
    FormulaData[];
  data = Nearest[data, input, {All, 1}, 
    DistanceFunction -> (1/(SmithWatermanSimilarity[#1, #2, 
           GapPenalty -> 1,IgnoreCase->True] + $MachineEpsilon) &)];
  data = Select[data, 
    SmithWatermanSimilarity[input, #, GapPenalty -> 1,IgnoreCase->True] >= 
      Min[Max[n, Floor[StringLength[#]/2]],8] &];
  If[StringFreeQ[#, ":"], #, StringSplit[#, ":"]] & /@ data]


MWANameLookup[input_String] := 
  Module[{data = 
     APICompute["MWAFormulaNameLookup", 
      Internal`MWASymbols`MWAFormulaNameLookup[input]], st}, 
   If[MatchQ[data, (HoldComplete|Hold)[{__}]], DeleteCases[ReleaseHold[data], {}],
    st = decamel[input];
    data = 
     APICompute["MWAFormulaNameLookup", 
      Internal`MWASymbols`MWAFormulaNameLookup[st]]; 
    If[MatchQ[data, (HoldComplete|Hold)[{__}]], DeleteCases[ReleaseHold[data], {}],
     If[StringLength[input] > 15||MatchQ[data, (HoldComplete|Hold)[{}]], localParse[input], {}]]]
   ];



(* to sort between popular and rarely used formulas for FormulaDataLookup *)
validOptions = {RequiredPhysicalQuantities, ExcludedPhysicalQuantities, 
  "RequiredPhysicalQuantities", "ExcludedPhysicalQuantities"}

Options[iFormulaLookup]={ExcludedPhysicalQuantities->{},RequiredPhysicalQuantities->{}};

iFormulaLookup[input_,n_,o:OptionsPattern[]]:=Catch[Module[{
	list, limit=n, hasOptions, rpqlist, xpqlist},
	Quiet[(*check for bad options*)
		Check[
			OptionValue[RequiredPhysicalQuantities],
			Block[{FormulaLookup},
				Message[
					FormulaLookup::optx,
					First[First[DeleteCases[{o}, Rule[x_, _] /; MemberQ[validOptions, x]]]],
					FormulaLookup[input,n,o]
				];
				Throw[$Failed,$tag]
			],
			{OptionValue::nodef}],
		{OptionValue::nodef}
	];
	rpqlist=OptionValue[RequiredPhysicalQuantities]/.Automatic->{};
	xpqlist=OptionValue[ExcludedPhysicalQuantities]/.Automatic->{};
	If[Not[MatchQ[rpqlist,List[___String]]],Message[FormulaLookup::pqlst,RequiredPhysicalQuantities,rpqlist];Throw[$Failed,$tag]];
	If[Not[MatchQ[xpqlist,List[___String]]],Message[FormulaLookup::pqlst,ExcludedPhysicalQuantities,xpqlist];Throw[$Failed,$tag]];
	If[Or[UnsameQ[rpqlist,{}],UnsameQ[xpqlist,{}]],hasOptions=True];
	Which[
		limit===All,
			limit=Infinity,
		And[Not[IntegerQ[limit]],limit=!=Infinity,Not[TrueQ[limit>=0]]],
			Block[{FormulaLookup},Message[FormulaLookup::innf,FormulaLookup[input,n,o],2]];
			Throw[$Failed,$tag]
	];
	Which[
		input===All,
			If[TrueQ[hasOptions],
					list=ReleaseHold[DownValues[MathematicaFormula]/.MathematicaFormula -> Identity],
					list=FormulaData[]
			],
		StringQ[input],
			list=MWANameLookup[input];
			If[list==={},
				Return[Missing["NotAvailable"]],
				list=DeleteCases[If[MathematicaFormula[#]===Missing["NotAvailable"],{},#]&/@list,{}];
				If[list==={},
					Return[Missing["NotAvailable"]]
				]
			];
			If[TrueQ[hasOptions],list={#,MathematicaFormula[#]}&/@list];,
		True,
			Throw[$Failed,$tag]
	];
	If[Length[rpqlist]>0,
		list = Function[{pq}, 
			Cases[list, _?(Not[FreeQ[#, pq]] &)]] /@ rpqlist;
		list=Intersection @@ list
	];
	If[Length[xpqlist]>0,
		list = Function[{pq}, 
			Cases[list, _?(Not[FreeQ[#, pq]] &)]] /@ xpqlist;
		list=Intersection @@ list;
		list = list[[All,1]];
		If[Length[list]<limit,list,list[[;;limit]]],
		If[TrueQ[hasOptions],list=list[[All,1]]];
		If[Length[list]<limit,list,list[[;;limit]]]
	]
],
	$tag]

		
Options[FormulaLookup]={ExcludedPhysicalQuantities->{},RequiredPhysicalQuantities->{}};

FormulaLookup[]:=FormulaData[]
FormulaLookup[All]:=FormulaData[]
FormulaLookup[string_String,o:OptionsPattern[]]:=With[{res=iFormulaLookup[string,All,o]},res/;res=!=$Failed]
FormulaLookup[All,o:OptionsPattern[]]:=With[{res=iFormulaLookup[All,All,o]},res/;res=!=$Failed]
(* with limit argument *)
FormulaLookup[string_String,n_,o:OptionsPattern[]]:=With[{res=iFormulaLookup[string,n,o]},res/;res=!=$Failed]
FormulaLookup[All,n_,o:OptionsPattern[]]:=With[{res=iFormulaLookup[All,n,o]},res/;res=!=$Failed]
FormulaLookup[arg:Except[_String],OptionsPattern[]]:=(Message[FormulaLookup::notent,arg,FormulaLookup];Null/;False)
FormulaLookup[arg:Except[_String],_,OptionsPattern[]]:=(Message[FormulaLookup::notent,arg,FormulaLookup];Null/;False)
FormulaLookup[args__] := (ArgumentCountQ[FormulaLookup,Length[DeleteCases[{args},_Rule,Infinity]],1,2];Null/;False)

(* verfiying that input is a PQ *)
verifyPQ[data_,inputrules_]:=Module[{rules=Cases[inputrules,HoldPattern[_ -> _Quantity]],
	vars=DeleteDuplicates[Cases[data,QuantityVariable[x_,y_String,___]:>Rule[x,y],Infinity]]},
	If[Length[Cases[vars[[All,1]],#]]>1||Length[Cases[vars[[All,2]],#/.Entity["PhysicalQuantity", x_] :> x]]>1,
		Message[FormulaData::indet,#]]&/@inputrules[[All,1]];
	rules=rules/.Entity["PhysicalQuantity", x_] :> x;
	If[Length[rules]>0,
		(* quantities included are correct *)
		verifyPQVariable[vars,#]&/@rules
	]
]; 

verifyPQVariable[data_,inputvar_->value_Quantity]:=Module[
{vars=data,pq,temperatureQ,unit},
  If[MemberQ[vars[[All,1]],inputvar]||MemberQ[vars[[All, 2]],inputvar],
    pq=inputvar/.vars;
    unit=QuantityVariableCanonicalUnit[pq];
    temperatureQ=ReleaseHold@(UnitDimensions[unit]/."TemperatureDifferenceUnit"->"TemperatureUnit")===
    	(UnitDimensions[QuantityUnit[value]]/."TemperatureDifferenceUnit"->"TemperatureUnit");
    If[CompatibleUnitQ[QuantityUnit[value], unit]||temperatureQ,
        Null,
        Message[FormulaData::pq,inputvar,pq]
    ],
    Message[QuantityVariable::unkpq,inputvar]
  ]
];

$cleanDVRules={
	QuantityVariable[x_,y_String,___]:>QuantityVariable[x,y]};
	
Clear[System`FormulaData]
System`FormulaData[args___]:=Module[{res=iFormulaData[args]},
	res/;res=!=$Failed
]

$formulaProperties=Sort@{"Formula", "QuantityVariableDimensions", "QuantityVariableNames", 
	"QuantityVariablePhysicalQuantities", "QuantityVariables", "QuantityVariableTable"};
$extraProperties={"FormulaSummary"};

iFormulaData[]=Cases[ReleaseHold[DownValues[MathematicaFormula] /. MathematicaFormula -> Identity][[All, 1]], _String | {_String ..}];
iFormulaData[All]:=iFormulaData[];
iFormulaData["Properties"] := $formulaProperties;
iFormulaData[string_,"Properties"] := Module[{data=MathematicaFormula[string]},
	If[data===Missing["NotAvailable"],
		Message[FormulaData::nvfn,string];
		$formulaProperties,
		$formulaProperties
	]
];
iFormulaData[string:(_String|{__String}), args___]:=Module[{data=MathematicaFormula[string]},(*check if is standard name *)
	If[data===Missing["NotAvailable"],(*eventually add Resourceformula code here*)
		Message[FormulaData::nvfn,string];
		Missing["NotAvailable"],
		iFormulaCompute[data,args]
	]
]
iFormulaData[arg:Except[_String|{__String}],args___]:=(Message[FormulaData::notent, arg, FormulaData];$Failed)
iFormulaData[_,arg_,___]:=((Message[FormulaData::notprop,arg,FormulaData];$Failed)/;Not[MemberQ[Append[Join[$formulaProperties,$extraProperties],"Properties"],args]])
iFormulaData[___]:=$Failed

iFormulaCompute[data_]:=iFormulaCompute[data, "Formula"];
iFormulaCompute[data_, props_List?(Complement[#,$formulaProperties,$extraProperties]==={}&)]:=iFormulaCompute[data, #]&/@props;
iFormulaCompute[data_, "Formula"]:=data/.$cleanDVRules;
iFormulaCompute[data_, "Association"]:=iFormulaCompute[data, "Formula", "Association"]
iFormulaCompute[data_, "QuantityVariableTable"]:=Module[{vars},
	vars=DeleteDuplicates[Cases[data,QuantityVariable[x_,y_String,___,"Name"->name_,___,
			"UnitDimensions"->z_,___]:>{x,name,y,z},Infinity]/.
		{x_,n_,{y_->z_},a_}:>{x,n,z,a}];
	vars=Prepend[vars,{"symbol","description","physical quantity","dimensions"}];
	Style[Grid[vars,
		Alignment->{Left,Baseline},
		Dividers->{{2->GrayLevel[0.7]},{2->GrayLevel[0.7]}}], "DialogStyle", StripOnInput->False]
]
iFormulaCompute[data_, "FormulaSummary"]:=Module[{vars,
	formula=(data/.$cleanDVRules)/.s_String/;StringLength[s] === 1 && 65 <= ToCharacterCode[s][[1]] <= 128 :> Style[s, Italic]},
	formula=If[MatchQ[formula,_List],Column[formula],formula];
	vars=DeleteDuplicates[Cases[data,QuantityVariable[x_,y_String,___,"Name"->name_,___]:>{x/.
				s_String/;StringLength[s] === 1 && 65 <= ToCharacterCode[s][[1]] <= 128 :> Style[s, Italic],name,y},Infinity]/.
		{x_,n_,{y_->z_},a_}:>{x,n,z,a}];
	vars=Prepend[vars,{"symbol","description","physical quantity"}];
	Column[{TraditionalForm[formula],
		Style[Grid[vars,
		Alignment->{Left,Baseline},
		Dividers->{{2->GrayLevel[0.7]},{2->GrayLevel[0.7]}}], "DialogStyle", StripOnInput->False]}
	]
]
iFormulaCompute[data_,"QuantityVariables"]:=DeleteDuplicates[Cases[data,
	QuantityVariable[x_,y_String,___]:>QuantityVariable[x,y],Infinity]]
iFormulaCompute[data_,"QuantityVariablePhysicalQuantities"]:=DeleteDuplicates[Cases[data,
		QuantityVariable[x_,y_String,___]:>QuantityVariable[x,y]->y,Infinity]/.
		HoldPattern[x_->{y_->_}]:>x->y]
iFormulaCompute[data_,"QuantityVariableDimensions"]:=DeleteDuplicates[Cases[data,
	QuantityVariable[x_,y_String,___,"UnitDimensions"->z_,___]:>QuantityVariable[x,y]->z,Infinity]]
iFormulaCompute[data_,"QuantityVariableNames"]:=DeleteDuplicates[Cases[data,
	QuantityVariable[x_,y_String,___,"Name"->z_,___]:>QuantityVariable[x,y]->z,Infinity]]
iFormulaCompute[data_,v_Association,a___]:=iFormulaCompute[data,Normal[v],a]
iFormulaCompute[rawdata_,v_?(MatchQ[#,{(Rule | RuleDelayed)[_,_]..}]&)]:=Module[{vars,final,rules,PQtoIDrules,result,data,
	AIDtoIDrules=DeleteDuplicates[Cases[rawdata,QuantityVariable[x_,___,"AlternateIdentifier"->y_,___]:>Rule[y,x],Infinity]]},
	rules=parameterConvert[v];
	If[AIDtoIDrules=!={},rules=rules/.AIDtoIDrules];
	PQtoIDrules=Cases[rawdata,QuantityVariable[x_,y_String,___]:>Rule[y,x],Infinity];
	verifyPQ[rawdata,rules]; (* check that variables are valid *)
	rules=rules/.PQtoIDrules/.Entity["PhysicalQuantity", x_] :> x;
	vars=Cases[rawdata,QuantityVariable[x_,___]:>x,Infinity];
	{rules,data}=temperatureCheck[{rules,rawdata}];
	If[Length[Complement[vars,rules[[All,1]]]]===Length[Flatten[{data}]]&&VectorQ[rules[[All, 2]], MatchQ[#, _Quantity] || NumericQ[#] &],
		final=Complement[vars,rules[[All,1]]];
		final=Flatten[{Cases[data,QuantityVariable[#,___],Infinity][[1]]}&/@final];
		result=(data/.Map[Rule[QuantityVariable[#[[1]],__],#[[2]]]&,rules]);
		final=Quiet[Reduce[result/.{x_Real:>N[x],
			TildeEqual|Greater|GreaterEqual|LessEqual|Less|TildeTilde|Greater->Equal},final,Reals]/.$cleanDVRules];
		Which[Quiet[MatchQ[final,_Reduce]],
			result/.$cleanDVRules,
			MatchQ[final,False],
			Message[FormulaData::nosol];result/.$cleanDVRules,
			True,
			final/.x_Quantity:>UnitSimplify[x]
		],
		If[Length[Complement[vars,rules[[All,1]]]]<Length[Flatten[{data}]],
			Message[FormulaData::overdet];$Failed,
			result=Quiet[data/.Map[Rule[QuantityVariable[#[[1]],__],#[[2]]]&,rules]/.$cleanDVRules];
			result
		]
	]
]
iFormulaCompute[data_,v_?(MatchQ[#,{(Rule | RuleDelayed)[_,_]..}]&),{"Plot",opts___}]:=Module[
	{sol=iFormulaCompute[data,v,"Association"],allvars=Cases[data,QuantityVariable[x_,___]:>x,Infinity],
	yvars,xvars,svars,xs,ys,xrange,yunit,xunits,x,ylabel,xlabel,factorrules,point,
	test,formula=data/.QuantityVariable[x_,y_String,___]:>QuantityVariable[x,y],
	PQtoIDrules=Cases[data,QuantityVariable[x_,y_String,___]:>Rule[y,x],Infinity],
	IDtoDefaultrules=Cases[data,QuantityVariable[x_,___,"DefaultValue"->y_,___]:>Rule[x,y],Infinity],
	IDtoUniquerules},
	IDtoUniquerules=Cases[data,QuantityVariable[id_,___]:>Rule[id,Unique[x]],Infinity];
	(*TODO: add handling for multiple formulas*)
	(*test inputs*)
	If[sol===$Failed,Print["failed solution"];(*TODO: add error message*)Return[$Failed]];
	If[MatchQ[{opts},{_Rule..}|{}],
		yvars=parameterConvert[Flatten[{"PlotQuantities"/.{opts}}]];
		xvars=parameterConvert[Flatten[{"VariableQuantities"/.{opts}}]],
		Print["bad options"];Return[$Failed](*TODO: add error message*)
	];
	If[Intersection[yvars,xvars]=!={},Print["overlap"];(*TODO: add error message*)Return[$Failed]];
	If[Length[yvars]=!=1,Print["wrong length for yvars"];(*TODO: add error message*)Return[$Failed]];
	If[Not[1<=xvars<=2],Print["wrong length for xvars"];(*TODO: add error message*)Return[$Failed]];
	(* pick x and y variables if undefined *)
	If[MatchQ[yvars,{"PlotQuantities"}|{}],yvars={allvars[[1]]}];
	If[MatchQ[xvars,{"VariableQuantities"}|{}],If[Length[allvars]>2,xvars=allvars[[2;;3]],xvars={allvars[[2]]}]];
	(*gather all solution values and fill in anything missing *)
	sol=Join[parameterConvert[Normal[sol]],parameterConvert[v]]/.PQtoIDrules;
	sol=#->(#/.sol)&/@allvars; (*to check we have values for everything*)
	test=DeleteCases[sol,HoldPattern[_->(_?NumericQ|_Quantity)]];
	If[test=!={},
		test=#[[1]]->(#[[1]]/.IDtoDefaultrules)&/@test;
		sol=DeleteDuplicates[Join[test,sol],First[#1] === First[#2] &]
	];
	svars=Complement[allvars,xvars,yvars];(*get static (for plot) variables, TODO: remove Length[formulas]-1 variables *)
	(*extract formulas for results, yvars need to be symbols? *)
	ys=yvars/.IDtoUniquerules;
	xs=xvars/.IDtoUniquerules;
	formula=parameterConvert[formula/.IDtoUniquerules];
	formula=Quiet[Reduce[formula,ys]];
	formula=Cases[{formula},Equal[First[ys],_?(!FreeQ[#,Alternatives@@xs]&)],Infinity];
	If[FreeQ[formula,$Failed]&&Length[formula]>0,formula=formula[[1,2]],Print["failure to solve"];Return[$Failed]];
	xrange=generateRange[#/.IDtoUniquerules,#/.(Reverse/@PQtoIDrules),
		With[{res=#/.sol},If[MatchQ[res,Quantity[_?NumericQ,__]]||NumericQ[res],res,#/.IDtoDefaultrules]]]&/@xvars;
	(*convert units over*)
	If[svars=!={},
		svars=MapThread[Rule,{svars/.IDtoUniquerules,svars/.sol}],
		svars={}
	];
	yunit=QuantityUnit[yvars[[1]]/.sol];
	xunits=QuantityUnit[#]&/@(xvars/.sol);
	ylabel=Row[{yvars[[1]]," (",If[yunit==="DimensionlessUnit","dimensionless",Quantity[None,yunit]],")"}];
	xlabel=MapThread[Row[{#1," (",If[#2==="DimensionlessUnit","dimensionless",Quantity[None,#2]],")"}]&,{xvars,xunits}];
	factorrules=Append[MapThread[Rule[#1,Quantity[#1,#2]]&,{xs,xunits}], ys->Quantity[ys,yunit]];
	formula=Thread[QuantityMagnitude[formula/.Join[svars,factorrules],yunit],Plus];
	point=QuantityMagnitude[(Flatten[{xvars,yvars}])/.sol];
	Which[Length[xvars]>2||Length[xvars]<1,Return[$Failed],
		Length[xvars]===2,
		With[{v1=First[xrange],v2=Last[xrange]},
			Show[Plot3D[Evaluate[formula],v1,v2,AxesLabel->Flatten[{xlabel,ylabel}]],
				Graphics3D[{PointSize[Large], Red, If[MatchQ[point,{_?NumericQ,_?NumericQ,_?NumericQ}],Point[point],Sequence@@{}]}]]
		],
		Length[xvars]===1,
		With[{v1=First[xrange]},
			Plot[Evaluate[formula],v1,
				Epilog->{PointSize[Large], Red, If[MatchQ[point,{_?NumericQ,_?NumericQ,_?NumericQ}],Point[point],Sequence@@{}]},
				Frame->True,FrameLabel->Flatten[{xlabel,ylabel}]]
		]	
	]
]
iFormulaCompute[data_,arg_,"List"]:=iFormulaCompute[data,arg]
iFormulaCompute[data_,arg_,"Association"]:=Which[
	MatchQ[arg,_List?(Complement[#,$formulaProperties,$extraProperties]==={}&)],
		Association[#->iFormulaCompute[data,#,"Association"]&/@arg],
	arg==="Formula",
		Association[Flatten[{data}/.$cleanDVRules]/.Equal->Rule],
	MemberQ[Join[$formulaProperties,$extraProperties],arg],
		If[MatchQ[#,{_Rule..}],Association[#],#]&@iFormulaCompute[data,arg],
	True,
		toAssociation[iFormulaCompute[data,arg]]
]
iFormulaCompute[data_,arg_String,___]:=(Message[FormulaData::notprop,arg,FormulaData];$Failed)
iFormulaCompute[___]:=$Failed

temperatureCheck[expr_]:=Module[{},
	If[FreeQ[expr, "DegreesCelsius" | "Kelvins" | "DegreesFahrenheit"],
		expr,
		If[FreeQ[expr,"KelvinsDifference"],
			expr,
			expr/.{"DegreesCelsius" ->"DegreesCelsiusDifference", "Kelvins" ->"KelvinsDifference",
				 "DegreesFahrenheit"->"DegreesFahrenheitDifference"}
		]
	]
]

(*convert parameters to standard form*)
parameterConvert[p_]:=((p/.QuantityVariable[x_,___]:>x)/.x_String?(StringMatchQ[#, ___ ~~ "Box" ~~ __] &) :> ToExpression[x, StandardForm])/.
		Subscript[l_?AtomQ, s_] :> Subscript[ToString[l], s];
(*create parameter ranges*)
generateRange[symbol_,"SoundSpeed"|"Speed",value_]:=Module[{s=0.5*value,e=Min[2*value,Quantity["SpeedOfLight"]]},
	{symbol,QuantityMagnitude[s],QuantityMagnitude[e]}
]
generateRange[symbol_,"Albedo",value_]:={symbol,0,1}
generateRange[symbol_,"VolumeFraction",value_]:={symbol,Quantity[0,"VolumePercent"],Quantity[100,"VolumePercent"]}
generateRange[symbol_,"Age",value_]:=Module[{s=0.5*value,e=1.5*value},
	{symbol,QuantityMagnitude[s],QuantityMagnitude[e]}
]
generateRange[symbol_,"Angle",value_]:=Module[{s=0.5*value,e=Min[2*value,Quantity[360,"AngularDegrees"]]},
	{symbol,QuantityMagnitude[s],QuantityMagnitude[e]}
]
generateRange[symbol_,pq_,value_]:=Module[{s=0.5*value,e=2*value},
	{symbol,QuantityMagnitude[s],QuantityMagnitude[e]}
]
(*turn Reduce results to Associations*)
toAssociation[arg_]:=Module[{res},
	res=If[MatchQ[arg,_List],
		Flatten[itoAssociation/@arg],
		itoAssociation@arg,
		$Failed
	];
	If[FreeQ[res,$Failed],Association[res],$Failed]
]

itoAssociation[arg_And]:=If[FreeQ[arg,Or],(List@@arg)/.Equal->Rule,$Failed]
itoAssociation[arg_Or]:=$Failed
itoAssociation[arg_Equal]:=If[FreeQ[arg,Or],arg/.Equal->Rule,$Failed]
itoAssociation[arg_]:=$Failed


If[FileExistsQ[#],Get[#]]&[FileNameJoin[{DirectoryName[$InputFileName],"PlancksLaw.m"}]]

With[{s=$ProtectedSymbols},SetAttributes[s,{ReadProtected}]];
Protect@@$ProtectedSymbols;

End[]; (* End Private Context *)
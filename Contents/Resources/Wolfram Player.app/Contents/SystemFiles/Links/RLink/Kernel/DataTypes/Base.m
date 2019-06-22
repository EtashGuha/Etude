(* :Title: Base.m *)

(* :Author:
        Leonid Shifrin
        leonids@wolfram.com
*)

(* :Package Version: 1.0 *)

(* :Mathematica Version: 8.0 *)

(* :Copyright: RLink source code (c) 2011-2012, Wolfram Research, Inc. All rights reserved. *)

(* :Discussion:


	This package implements some of the important R data types, such as factors and data 
	frames - that is, higher-level convenient Mathematica representation for those. These
	data types, while important, technically are special cases of core R data types such 
	as lists and vectors, and therefore they are not included in the core RLink.
	
	The implementations below can also serve as examples of how one implements extended 
	data types and registers them with RLink 
	
*)

BeginPackage["RLink`DataTypes`Base`",{"RLink`","RLink`DataTypes`Common`"}]
(* Exported symbols added here with SymbolName::usage *)  

RFactor::usage = "RFactor[_List, RFactorLevels[__],  a : (_RAttributes | None) : None]] is an \
inert head representing R Factor data type. The pattern shows the generic form of Mathematica \
expression interpreted by RLink as a representation of R factor ";

RFactorLevels::usage = "RFactorLevels[levels__] is an inert head (container) for a sequence of \
factor levels for R factor object";

RFactorQ::usage = "RFactorQ[expr_] returns True is expr represents R factor for RLink's purposes,\
and False otherwise";

RFactorToVector::usage = "RFactorToVector[expr_] converts R factor (its RLink Mathematica \
representation) into R vector (its RLink Mathematica representation). If expr does not \
represent an R factor (for RLink's purposes), $Failed is returned.";

RDataFrame::usage = 
"RDataFrame[RNames[names___String],RData[data__], rn : (_RRowNames | Automatic) : Automatic,  \
 a : (_RAttributes | None) : None ]  is an inert head representing R data frame data type. The \
pattern shows the generic form of Mathematica expression interpreted by RLink as a representation \
of R data frame";

RDataFrameQ::usage = "RFactorQ[expr_] returns True is expr represents R data frame for RLink's \
purposes, and False otherwise";

RGetFactorLevels::usage = "RGetFactorLevels[factor_] returns a list of factor levels for an R
factor represented by \"factor\"";

RFactorsToVectorsInDataFrame::usage = "RFactorsToVectorsInDataFrame[df_] converts all top-level 
factors possibly contained in the data of a given data frame representation, to vectors";



Begin["`Private`"] (* Begin Private Context *) 

Needs["RLink`RDataTypeTools`"]


(***********************************************************************)
(****************				FACTORS					****************)
(***********************************************************************)

partWithMissing[expr_, inds_List] :=
    With[{posNA = Position[inds, Missing[]]},
     MapAt[Missing[] &, Part[expr, MapAt[1 &, inds, posNA]], posNA]];
     

     

ClearAll[RFactor];
RFactorQ[_RFactor]:=True;
RFactorQ[r_RObject]:= RInstanceOf["factor"][r];
RFactorQ[_]:=False;

RFactor /: 
	RGetFactorLevels[ RFactor[_List, RFactorLevels[levs__],  a : _ : None]]:= {levs};

RFactor /: RGetData[ RFactor[p_List,__]]:= p;

RFactor /: 
	RGetAttributes[RFactor[_List, RFactorLevels[__],  a : (_RAttributes | None) : None]]:=
		RGetAllAttributes[a];
		

Clear[RFactorToVector];
RFactorToVector[f_RFactor] :=  
   With[{data = partWithMissing[RGetFactorLevels[f], RGetData[f]]},
    FromRForm @ ToRForm @ 
      RObject[data, RRemoveAttributes[RAttributes@@RGetAttributes[f], {"class" , "levels"}]]
   ];
   
RFactorToVector[_] = $Failed;


(*  			Register the type 			*)

RDataTypeRegister["factor",
	
 RFactor[_List, RFactorLevels[__],  a : (_RAttributes | None) : None],
 
 RFactor[p_List, RFactorLevels[levs__],   a : (_RAttributes | None) : None] :>
  RObject[p, RAddAttributes[a,{"levels" :> {levs}, "class" :> "factor"}]],
    
 _RObject ? RFactorQ,
 
 RObject[p_List, a_RAttributes] ? RFactorQ :>    
    RFactor[p, 
     	RFactorLevels @@ RExtractAttribute[a,"levels"], 
     	RRemoveAttributesComplete[a, {"levels", "class"}]    
    ]   
 ]
 
 
(***********************************************************************)
(****************				DATA FRAMES				****************)
(***********************************************************************) 
 
 
 
ClearAll[RDataFrameQ];
RDataFrameQ[_RDataFrame]:=True;
RDataFrameQ[r_RObject]:= RInstanceOf["data.frame"][r];
RDataFrameQ[_]:=False;


(*			 	Accessor functions 			*)


RDataFrame /: RGetNames[ RDataFrame[RNames[names___],__]] := {names};

RDataFrame /: RGetAttributes[ RDataFrame[__,a : (_RAttributes | None) : None]]:=
	RGetAllAttributes[a];
	
	
RDataFrame /: 
	RGetRowNames[
		RDataFrame[
				_,
				RData[data__],
				rnames : (RRowNames[rn__] | Automatic) : Automatic,
				a : (RAttributes[__] | None) : None
		]]:=
	If[rnames === Automatic, Range[Length[{data}]],{rn}];
	
RDataFrame /: RGetData[RDataFrame[_, RData[data__],___]]:= {data};




ClearAll[RFactorsToVectorsInDataFrame]
RFactorsToVectorsInDataFrame[df_RDataFrame]:=
	Replace[
		df,
		RData[data__]:> RData@@Replace[{data},f_?RFactorQ :> RFactorToVector[f],{1}],
		{1}
	];


(* 						Display function					*)



ClearAll[displayDataFrame]; 
displayDataFrame[ df_RDataFrame/;MemberQ[RGetData[df],_?RFactorQ]]:=
	displayDataFrame[ RFactorsToVectorsInDataFrame[df] ];
	
displayDataFrame[df_RDataFrame]:=
	TableForm[
		Transpose@RGetData[df] /. Missing[] -> "<NA>", 
    	TableHeadings -> {RGetRowNames[df], RGetNames[df]}
    ];


RDataFrame /: TableForm[df_RDataFrame] := displayDataFrame[df];

		

(*  			Register the type 			*)


RDataTypeRegister["data.frame",
 	
 RDataFrame[
 	RNames[names___String], 
 	RData[data__], 
  	rn : (_RRowNames | Automatic) : Automatic,  
  	a : (_RAttributes | None) : None
 ],
  
 df:RDataFrame[RNames[names___], RData[data__], rn: (_RRowNames | Automatic) : Automatic, a : (_RAttributes | None) : None] :>
  With[{rnn = RGetRowNames[df]},
     RObject[
      	{data}, 
      	RAddAttributes[a,
       		{
       			"names" :> {names},
       			"class" :> {"data.frame"}, 
        		"row.names" :> rnn
       		}
       ]
     ]
  ],
  
 _RObject ? RDataFrameQ,  
 
 RObject[d_, a_RAttributes] ? RDataFrameQ :>
  	With[ {get = RExtractAttribute[a, #] &},
      DeleteCases[
       	RDataFrame[
       		RNames @@ get["names"], 
       		RData @@ d, 
       		RRowNames @@ get["row.names"]
       	], 
       	_[$Failed]
      ]
  	]
]
 
 


End[] (* End Private Context *)

EndPackage[]
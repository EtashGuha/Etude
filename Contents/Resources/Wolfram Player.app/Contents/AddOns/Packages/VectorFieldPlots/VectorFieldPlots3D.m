(* :Title: 3D Vector Fields *)

(* :Context: VectorFieldPlots` *)

(* :Author: John M. Novak *)

(* :Summary:
This package does plots of vector fields in three dimensions.
VectorFieldPlot3D allows one to specify the functions describing the
three components of the field.  GradientFieldPlot3D plots the 
gradient vector field associated with a scalar function. 
ListVectorFieldPlot3D plots a three-dimensional array of vectors.
*)

(* :Mathematica Version: 2.0 *)
(* :Copyright: Copyright 1990-2007, Wolfram Research, Inc.*)

(* :Package Version: 1.0 *)

(* :History:
	V 1.0 April 1991 by John M. Novak, based extensively on
		PlotField.m by Kevin McIsaac, Mike Chan, ECM, and John Novak
		VectorField3D.m by Wolfram Research 1990 and ECM.
*)

(* :Keywords:
	vector fields, gradient field, 3D graphics
*)

(* :Limitations: *)

BeginPackage["VectorFieldPlots`"]

Begin["`Private`"]

cross3[{a1_, a2_, a3_}, {b1_, b2_, b3_}] := 
	{-(a3 b2) + a2 b3, a3 b1 - a1 b3,  -(a2 b1) + a1 b2}

mag[a_] := Sqrt[Apply[Plus, a^2]]

automatic[x_, value_] :=
	If[x === Automatic, value, x]

vector3D[point:{x_, y_, z_}, grad:{dx_, dy_, dz_},False] :=
	Line[{point, point + grad}]

vector3D[point:{x_,y_,z_}, grad:{dx_,dy_,dz_},True] :=
	Point[{x,y,z}]/;grad == {0,0,0}

vector3D[point:{x_, y_, z_}, grad:{dx_, dy_, dz_},True] :=
	Module[{endpoint, perp, perpm, offsetPoint,
		   arrowA, arrowB, arrowC, arrowD},
	  endpoint = point + grad;

	  perp = cross3[grad, {0,0,1}];
	  perpm = mag[perp];
	  If[perpm == 0,
		perp = cross3[grad, {0,1,0}];
		perpm = mag[perp]
	  ];
	  perp = perp mag[grad]/(7 perpm);
	  
	  offsetPoint = point + 4/5 grad;
	  arrowA = offsetPoint + perp;
	  
	  perp = cross3[grad, perp];
	  perp = perp mag[grad]/(7 mag[perp]);
	  arrowB = offsetPoint + perp;
	  
	  perp = cross3[grad, perp];
	  perp = perp mag[grad]/(7 mag[perp]);
	  arrowC = offsetPoint + perp;
	  
	  perp = cross3[grad, perp];
	  perp = perp mag[grad]/(7 mag[perp]);
	  arrowD = offsetPoint + perp;
	  
	  {Line[{point, endpoint}], 			(* 3D arrow shaft *)
	   Line[{arrowA, endpoint, arrowC}], 		(* point of arrow *)
	   Line[{arrowB, endpoint, arrowD}], 		(* point of arrow *)
	   Line[{arrowA, arrowB, arrowC, arrowD, arrowA}] (* base of point *)
	  }
	]

Options[ListVectorFieldPlot3D] = 
	SortBy[Join[{ScaleFactor->Automatic, 
	 ScaleFunction->None,
	 MaxArrowLength->None,
	 ColorFunction->None,
	 VectorHeads->False},Options[Graphics3D]], 
	 First];


ListVectorFieldPlot3D[vects:{{_?VectorQ,_?VectorQ}..},opts:OptionsPattern[]] :=
	Module[{maxsize,scale,scalefunct,colorfunct,heads,points,
			vectors,mags,colors,scaledmag,allvecs,vecs=N[vects]},
		{maxsize,scale,scalefunct,colorfunct,heads} = OptionValue[
			{MaxArrowLength,ScaleFactor,ScaleFunction,
			ColorFunction,VectorHeads}];
		
		(* option checking *)
		If[Not[NumberQ[maxsize]] && maxsize != None,
			maxsize = None,
			maxsize = N[maxsize]];
		If[Not[NumberQ[scale]] && scale =!= Automatic,
			scale = Automatic,
			scale = N[scale]];
		heads = TrueQ[heads];
		
		vecs = Cases[vecs,{_,_?(VectorQ[#,NumberQ]&)}];
		{points, vectors} = Transpose[vecs];
		mags = Map[mag,vectors];
		If[colorfunct == None, colorfunct = {}&];
		If[Max[mags - Min[mags]] == 0,
			colors = Map[colorfunct,Table[0,{Length[mags]}]],
			colors = Map[colorfunct,
				(mags - Min[mags])/Max[mags - Min[mags]]]
		];

		If[scalefunct =!= None,
		 	scaledmag = (If[# == 0, 0, scalefunct[#]]&) /@ mags;
		 	vectors = MapThread[If[#2 == 0, {0,0,0}, #1 #2/#3]&,
				{vectors,scaledmag,mags}];
		 	mags = scaledmag
		   ];

		allvecs = Transpose[{colors,points,vectors,mags}];  

		If[maxsize =!= None,
		 	allvecs = Select[allvecs, (#[[4]]<=maxsize)&]
		   ];
		
		If[Max[mags] != 0,
			scale = automatic[scale,Max[mags]]/Max[mags];
 			allvecs = Map[{#[[1]],#[[2]],scale #[[3]]}&,
 				allvecs]
 		];

		(* alternate method of vector generation requires pr.
		pr = PlotRange[ Graphics3D[
				Flatten[Apply[Line[{#2,#2+#3}]&,allvecs,{1}]]]];
		*)
		
		Show[Graphics3D[
		 		Flatten[Apply[{#1,vector3D[#2,#3,heads]}&,
		 			allvecs,{1}]],
		 		FilterRules[Flatten[{opts}], Options[Graphics3D]]]]
	]/; Last[Dimensions[vects]] === 3

Options[VectorFieldPlot3D] =  Options[GradientFieldPlot3D] =
	Join[Options[ListVectorFieldPlot3D],{PlotPoints->7}]

SetAttributes[VectorFieldPlot3D, HoldFirst]

VectorFieldPlot3D[f_, {u_, u0_, u1_, du_:Automatic},
			 {v_, v0_, v1_, dv_:Automatic},
			 {w_,w0_,w1_,dw_:Automatic},opts___] :=
	Module[{plotpoints,dua,dva,dwa,vecs, sf},
		{plotpoints, sf} = {PlotPoints, ScaleFactor}/.{opts}/.
			Options[VectorFieldPlot3D];
		dua = automatic[du,(u1 - u0)/(plotpoints-1)];
		dva = automatic[dv,(v1 - v0)/(plotpoints-1)];
		dwa = automatic[dw,(w1 - w0)/(plotpoints-1)];
		If[sf =!= None && !NumberQ[N[sf]],
		   sf = N[Min[dua, dva, dwa]]
        ];
		vecs = Flatten[Table[{N[{u,v,w}],N[f]},
			Evaluate[{u,u0,u1,dua}],Evaluate[{v,v0,v1,dva}],
			Evaluate[{w,w0,w1,dwa}]],2];
		ListVectorFieldPlot3D[vecs,
			Flatten[{FilterRules[Flatten[{opts}], Options[ListVectorFieldPlot3D]],
			FilterRules[Flatten[{opts}], Options[Graphics3D]],
			ScaleFactor->sf}] ]
	]

GradientFieldPlot3D[function_, 
		{u_, u0__}, 
		{v_, v0__},
		{w_, w0__},
		options___] :=
	VectorFieldPlot3D[Evaluate[{D[function, u],
					D[function, v],D[function,w]}],
			{u, u0},
			{v, v0},
			{w, w0},
			options]

End[]

EndPackage[]

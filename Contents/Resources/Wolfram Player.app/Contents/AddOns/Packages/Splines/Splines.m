
(* :Title: Spline *)

(* :Author: John M. Novak *)

(* :Summary:
This package introduces a Spline graphics primitive
and provides utilities for rendering splines.
*)
	
(* :Context: Graphics`Spline` *)

(* :Copyright: Copyright 1990-2007, Wolfram Research, Inc.*)

(* :Package Version: 2.0.2 *)

(* :History:
	V1.0 by John M. Novak, December 1990.
	V2.0 by John M. Novak, July 1992.
	V2.0.1 by John M. Novak, bug fix to handle DisplayString, February 1997.
	V2.0.2 by John M. Novak, remove default use of RenderSpline, February 1998.
*)

(* :Keywords: splines, curve fitting, graphics *)

(* :Sources:
	Bartels, Beatty, and Barsky: An Introduction to
		Splines for Use in Computer Graphics and
		Geometric Modelling, Morgan Kaufmann, 1987.
	de Boor, Carl: A Practical Guide to Splines, Springer-Verlag, 1978.
	Levy, Silvio: Bezier.m: A Mathematica Package for 
		Bezier Splines, December 1990.
*)

(* :Warning: Adds definitions to the function Display. *)

(* :Mathematica Version: 2.2 *)

(* :Limitation: Does not currently handle 3D splines, although
	some spline primitives may produce a curve in space. *)

BeginPackage["Splines`"]

Spline::sptt = "Warning: Value of option SplinePoints -> `1` is not an integer >= 3, setting to 25.";

Spline::args = "The spline object `` is not a valid 2D spline of the form Spline[pts, type, (SplineFunction[...]), (opts)], and cannot be rendered.";

SplineFit::cbezlen = "Points are needed to generate a spline.";

SplineFunction::dmval = "Input value `1` lies outside the domain of the spline function.";

Begin["`Private`"]

Format[Spline[p_,t_,b__]] :=
	Row[{"Spline[",p,",",t,",","<>","]"}]

Options[Spline] =
	{SplinePoints -> 25,
	SplineDots-> None,
	MaxBend -> 10.,
	SplineDivision -> 20.};

Spline[pts_List,type_Symbol,opts:(_Rule | _RuleDelayed)...] :=
	Spline[pts,type,
		SplineFit[pts,type],
		opts]
		
Spline[pts_List,type_Symbol,
		SplineFunction[t_,r_,cpts_,rest___],
		opts:(_Rule | _RuleDelayed)...] :=
	Spline[pts,type,SplineFit[pts,type],
		opts]/; pts != cpts || type != t


(* The following routines handle rendering of splines *)

splinetoline[Spline[pts:{{_?NumericQ,_?NumericQ}..},type_Symbol,
		fn_SplineFunction,
		opts:(_Rule | _RuleDelayed)...]] :=
	Module[{res,dots,spts,line},
		{dots} = {SplineDots}/.{opts}/.
			Options[Spline];
		If[MemberQ[{Automatic, True}, dots],
			dots = {PointSize[.03], Hue[0]}];
		line = RenderSpline[pts, type, fn];
		If[Head[line] === RenderSpline,
			line = Line[splinepoints[pts,
				type, fn, opts]]
		];
		If[dots === None || dots === False,
			line,
			{Flatten[{dots,Map[Point,pts]}],
				line}
		]
	]

splinetoline[sp_] := (Message[Spline::args, sp]; {})

splinepoints[genpts_, type_,
		fn:SplineFunction[_,{min_, max_},___],
		 opts___] :=
	Module[{pts, ipts, rng, spts, sdiv, maxb},
		{spts, sdiv, maxb} = {SplinePoints,
			SplineDivision, MaxBend}/.{opts}/.
			Options[Spline];
		If[!IntegerQ[spts] || spts < 3,
			Message[Spline::sptt, spts];
			spts = 25];
		rng = Range[min, max,(max - min)/(spts-1)];
		ipts = Transpose[{rng, pts = Map[fn,rng]}];
		If[MemberQ[{0,0., Infinity, None, False}, sdiv],
			pts,
			Last[Transpose[sampleall[ipts,fn,maxb,
				(max - min)/(spts sdiv)]]]
		]
	]

splinepoints[args___] := (Message[Spline::args, Spline[args]]; {})

(* following adaptively samples the spline *)

bend[{u1_,pt1_}, {u2_,pt2_}, {u3_, pt3_}, min_] :=
	Module[{v1 = pt1 - pt2, v2 = pt3 - pt2,n},
		If[N[(n = norm[v1] norm[v2]) == 0 ||
				u2 - u1 <= min ||
				u3 - u2 <= min],
			0,
			Re[Pi - ArcCos[(v1 . v2)/n]]
		]
	]

norm[{x_, y_}] := Sqrt[x^2 + y^2]

sampleall[pts_, fun_, maxb_, mini_] :=
	Fold[sample[#1,#2,fun,maxb,mini]&,
		 Take[pts, 2], Drop[pts,2]]

sample[pts_, next_, function_, maxbend_, min_] :=
	With[{first = Drop[pts, -2], last = Sequence @@ Take[pts, -2]},
		If[N[bend[last, next, min] > maxbend Degree],
			Join[first,
				sampleall[interp[last, next, function],
							function, maxbend, min]
			],
		(* else *)
			Append[pts, next]
		]
	]

interp[pt1:{u1_,_},pt2:{u2_,_},pt3:{u3_,_},fun_] :=
	Module[{i1 = (u2 + u1)/2, i2 = (u3 +u2)/2,out},
		{pt1, {i1, fun[i1]}, pt2, {i2, fun[i2]}, pt3}
	]

(* following combines a spline into a polygon or line *)
splinesubsume[shape_] :=
	shape/.sp_Spline :> (Sequence @@ (splinepoints @@ sp))

Typeset`MakeBoxes[sp:Spline[___], fmt_, Graphics] :=
    Typeset`MakeBoxes[#,fmt, Graphics]& @@ {splinetoline[sp]}

(* these rules need to be placed before the regular rules for Line/Polygon,
   or they'll never get hit; thus, we need a by-hand ordering... *)
DownValues[Typeset`MakeBoxes] =
    {HoldPattern[Typeset`MakeBoxes[
     Graphics`Spline`Private`p:Polygon[{___, _Spline, ___}], 
     Graphics`Spline`Private`fmt_, Graphics]] :> 
   (Typeset`MakeBoxes[#1, Graphics`Spline`Private`fmt, 
      Graphics] & ) @@ {Graphics`Spline`Private`splinesubsume[
      Graphics`Spline`Private`p]}, 
  HoldPattern[Typeset`MakeBoxes[Graphics`Spline`Private`p:
      Line[{___, _Spline, ___}], Graphics`Spline`Private`fmt_, 
     Graphics]] :> (Typeset`MakeBoxes[#1, 
      Graphics`Spline`Private`fmt, Graphics] & ) @@ 
    {Graphics`Spline`Private`splinesubsume[Graphics`Spline`Private`p]}} ~Join~
  DownValues[Typeset`MakeBoxes];

Unprotect[Display];

Display[f_,gr_?(!FreeQ[#,Spline[___]]&),opts___] :=
	(Display[f,gr/.{p:(Polygon[{___,_Spline,___}] | 
					Line[{___,_Spline,___}]) :>
					splinesubsume[p],
				v_Spline:>splinetoline[v]},
			opts];
			gr)

Protect[Display];

Unprotect[DisplayString];

DisplayString[gr_?(!FreeQ[#,Spline[___]]&),opts___] :=
	DisplayString[gr/.{p:(Polygon[{___,_Spline,___}] | 
					Line[{___,_Spline,___}]) :>
					splinesubsume[p],
				v_Spline:>splinetoline[v]},
			opts]

Protect[DisplayString];

(**** SplineFit ****)

SplineFit[pts_List?(MatrixQ[#, NumberQ[N[#]]&]&),
	type_Symbol?(MemberQ[{Cubic, Bezier, CompositeBezier}, #]&)] :=
	SplineFunction[type, {0., N[Length[pts] - 1]},
		pts,
		splineinternal[pts,type]]

Format[SplineFunction[t_,r_, b__]] :=
	Row[{"SplineFunction[",t,", ", r,", ","<>","]"}]


SplineFunction[type_, {min_, max_}, pts_,
		internal_][in_?(NumberQ[N[#]]&)] :=
	Module[{out},
		If[in < min || in > max,
			Message[SplineFunction::dmval, in];
			out = $Failed,
		(* else *)
			out = evalspline[
				 Which[in == max, Min[max, in],
					   in == min, Max[min, in],
					   True, in], type, pts, internal]
		];
		out/;out =!= $Failed
	]

(* the spline internal routines.  This is where the internal
	forms for various spline types are defined. *)

splineinternal[pts_List,Cubic] :=
		Transpose[Map[splinecoord,Transpose[pts]]]

splineinternal[pts_List,CompositeBezier] :=
	Module[{eqns, gpts = pts,ln = Length[pts],end},
		If[ln < 3 || OddQ[ln],
			Which[ln == 1, gpts = Flatten[Table[gpts,{4}]],
				ln == 2, gpts = {gpts[[1]],gpts[[1]],gpts[[2]],gpts[[2]]},
				OddQ[ln], AppendTo[gpts,Last[gpts]],
				True, Message[SplineFit::cbezlen];
					Return[InString[$Line]]]];
		end = Take[gpts,-4];
		gpts = Partition[Drop[gpts,-2],4,2];
		gpts = Apply[{#1,#2,#3 - (#4 - #3),#3}&,gpts,{1}];
		AppendTo[gpts,end];	
		Apply[Transpose[{#1,3(#2 - #1),
				3(#3 - 2 #2 + #1),#4 - 3 #3 + 3 #2 - #1}]&,
			gpts,{1}]
	]

splineinternal[pts_List,Bezier] :=
	Module[{n, eq, deg = Length[pts] - 1},
		eq = Table[#^n (1 - #)^(deg - n),{n,0,deg}];
		Function[Evaluate[Plus @@ (pts Table[Binomial[deg,n],
			{n,0,deg}] eq)]]
	]

(* some functions to assist the Cubic splineinternal routine *)
trisolve[lst_, ln_] :=
	Module[{},
		LinearSolve[
		 SparseArray[{Band[{1, 1}] -> Join[{2}, Table[4, {ln - 2}], {2}], Band[{1, 2}] -> 1, 
		   Band[{2, 1}] -> 1}], lst]
	]

splinecoord[vals_] := 
	Module[{lst,ln = Length[vals],d,n},
		lst = Join[{3 (vals[[2]] - vals[[1]])},
			Table[3 (vals[[n + 2]] - vals[[n]]),
					{n,ln - 2}],
			{3 (vals[[ln]] - vals[[ln - 1]])}];
		d = trisolve[lst, ln];
		Table[{vals[[n]],d[[n]],
			3(vals[[n+1]]-vals[[n]])-2 d[[n]]-d[[n+1]],
			2(vals[[n]]-vals[[n+1]])+d[[n]]+d[[n+1]]},
				{n,1,ln - 1}]]

(* routines to evaluate the spline function at particular
	values of the parameter *)

evalspline[pt_?(# == 0 &), Cubic, pts_, internal_] :=
	internal[[1,All,1]]

evalspline[pt_, Cubic, pts_, internal_] :=
	Module[{tmp},
		({1, #, #^2, #^3}& @
			If[(tmp = Mod[pt,1]) == 0, 1, tmp]) .
		Transpose[internal[[Ceiling[pt]]]]
	]

evalspline[pt_, CompositeBezier, pts_, internal_] :=
	Module[{ln = Length[pts] - 1},
		evalspline[If[ln <= 3,
			pt/ln,
			pt (1/2 - If[OddQ[ln], 1/(2 ln) ,0])],
		 Cubic, pts, internal]
	]

evalspline[pt_, Bezier, pts_, internal_] :=
	internal @ (pt/(Length[pts] - 1))

End[]

EndPackage[]

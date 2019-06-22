(* :Package Version: 2.5 *)

(* :Mathematica Version: 6.0 *)

(* :Copyright: Copyright 1990-2007, Wolfram Research, Inc. *)

(* :Name: VectorFieldPlots` *)

(* :Title: Vector Field Plots of 2D Vector Functions *)

(* :Author:
    Kevin McIsaac, Wolfram Research, Inc.
    Updated by Mike Chan and Bruce Sawhill, Wolfram Research, Inc.,
    September 1990.
    Modified April 1991, by John M. Novak.
    V2.3, July/October 1992, by John M. Novak--use Arrow.m package.
    V2.4 January 1994 by John M. Novak. Various revisions and improvements.
    V2.5 November 2004 by John M. Novak. Change to use Arrow primitives.
*)

(* :Keywords:
    vector field plot, 2D Vector Functions, Polya representation
*)

(* :Requirements: None. *)

(* :Warnings: None. *)

(* :Sources: *)

(*:Summary:
This package does plots of vector fields in the plane.
VectorFieldPlot allows one to specify the functions describing the
two components of the field.  GradientFieldPlot and HamiltonianFieldPlot
plot the respective vector fields associated with a scalar function.
PolyaFieldPlot plots the field associated with a complex-valued
function.  ListVectorFieldPlot plots a rectangular array of vectors.
*)

Message[General::obspkg, "VectorFieldPlots`"]

BeginPackage["VectorFieldPlots`"]

Begin["`Private`"]

(* --- utilities --- *)
(* local numberQ - supersede by NumericQ after V3.0. *)
numberQ[n_] := NumberQ[N[n]]

(* little utility that is really a holdover from an early version of
   the package; returns the second arg if the first arg is Automatic,
   otherwise return the first arg. *)
automatic[x_, value_] :=
    If[x === Automatic, value, x]

(* utility to compute magnitude of vector *)
magnitude[v_List] := Sqrt[v . v]

(* compatibility hack utility to allow old-style arrows; if an old-style
   Arrow.m Arrow option is supplied to the plotting function, the plotting
   function will invoke Arrow.m and use that syntax. Arrow.m is not loaded
   unless one of these options is used. *)
getoldarrowopts[opts_] :=
    Module[{selopts},
      (* since package no longer loaded by default, user may not have option in right
         context, so we'll detect in all contexts. *)
        selopts = Select[opts, MemberQ[{"HeadScaling", "HeadLength", "HeadCenter",
                                        "HeadWidth", "HeadShape", "ZeroShape"},
                                        SymbolName[First[#]]]&
        ];
        Map[((SymbolName[First[#]]/.{"HeadScaling" -> Graphics`Arrow`HeadScaling,
                                    "HeadLength" -> Graphics`Arrow`HeadLength,
                                    "HeadCenter" -> Graphics`Arrow`HeadCenter,
                                    "HeadWidth" -> Graphics`Arrow`HeadWidth,
                                    "HeadShape" -> Graphics`Arrow`HeadShape,
                                    "ZeroShape" -> Graphics`Arrow`ZeroShape}) ->
                                    #[[2]])&, selopts]
    ]

(* --- plotting functions --- *)
(* Plot a list of {base, vector} pairs or a matrix of vectors (assumed
   to be placed on an integer base grid) *)
   
ListVectorFieldPlot::lpvf =
"ListVectorFieldPlot requires a rectangular array of vectors or a list \
of {base, vector} pairs.";

Options[ListVectorFieldPlot] = 
    Sort[Join[
        {ScaleFactor->Automatic, 
         ScaleFunction->None,
         MaxArrowLength->None,
         ColorFunction->None},
        Developer`GraphicsOptions[]
    ]];

SetOptions[ListVectorFieldPlot,
            PlotRange -> All,
            AspectRatio -> Automatic];

ListVectorFieldPlot[ vects:{{{_,_},{_,_}}..}, opts___?OptionQ] :=
    Module[{maxsize,scale,scalefunct,colorfunct,points,
            vectors,colors,mags,scaledmag,allvecs,
            vecs = N[vects], arropts},
      (* -- get option values -- *)
        {maxsize,scale,scalefunct,colorfunct} =
            {MaxArrowLength,ScaleFactor,ScaleFunction,
            ColorFunction}/.Flatten[{opts, Options[ListVectorFieldPlot]}];
      (* select things that can only be vectors from the input *)
        vecs = Cases[vecs,
               {{_?numberQ, _?numberQ}, {_?numberQ, _?numberQ}},
               Infinity];
        {points, vectors} = Transpose[vecs];
        mags = Map[magnitude,vectors];
      (* -- determine the colors -- *)
      (* if the colorfunction is None, cause it to generate empty lists *)
        If[colorfunct == None, colorfunct = {}&];
      (* if all vectors are the same size, make list of colorfunct[0],
          else map the color function across the magnitudes *)
        If[Equal @@ mags,
            colors = Table[Evaluate[colorfunct[0]],{Length[mags]}],
            colors = Map[colorfunct,
                (mags - Min[mags])/Max[mags - Min[mags]]]
        ];
      (* -- scale vectors by scale function -- *)
        If[scalefunct =!= None,
             scaledmag = Map[If[# == 0, 0, scalefunct[#]]&, mags];
             {vectors, mags} = Transpose[MapThread[
                  If[#3 == 0 || !numberQ[#2], {{0,0}, 0}, {#1 #2/#3, #2}]&,
                  {vectors, scaledmag, mags}
              ]]
        ];
      (* regroup colors, points, and mags with the associated vectors *)
        allvecs = Transpose[{colors, points, vectors, mags}];  
      (* pull all vectors with magnitude greater than MaxArrowLength *)
        If[numberQ[maxsize],
             allvecs = Select[allvecs, (#[[4]] <= N[maxsize])&]
        ];
      (* calculate scale factor *)
        If[numberQ[scale],
            scale = scale/Max[mags],
            scale = 1
        ];
      (* compatability hack: see if user supplied old-style arrowoptions *)
        arropts = getoldarrowopts[Flatten[{opts, Options[ListVectorFieldPlot]}]];
      (* turn the vectors into Arrow objects *)
        If[arropts =!= {},
            Needs["Graphics`Arrow`"];
            allvecs = Apply[
                Flatten[{#1, Arrow[#2, #2 + scale #3, 
                                   arropts,
                                   Graphics`Arrow`HeadScaling -> Automatic,
                                   Graphics`Arrow`HeadLength -> 0.02]
                                   }]&,
                allvecs, {1}],
          (* else V6-style arrows *)
            allvecs = Apply[
                Flatten[{#1, If[scale #3 == {0., 0.}, Point[#2],
                                Arrow[{#2, #2 + scale #3}]]}]&,
                allvecs, {1}]
        ];
      (* -- show the vector field plot -- *)
      (* note that line thickness is forced to 0.0001 (thin lines);
         this can be overridden by use of ColorFunction option *)
        Graphics[
             {Thickness[Small], Arrowheads[0.02], allvecs},
             FilterRules[Flatten[{opts, Options[ListVectorFieldPlot]}], Options[Graphics]]
        ]
    ]

(* given a matrix of vectors with no specified bases, generate base points
   on an integer grid, and pass back to ListVectorFieldPlot. *)    
ListVectorFieldPlot[ vects_List?(ArrayDepth[#] === 3 &),opts___] :=
    ListVectorFieldPlot[
        Flatten[MapIndexed[{Reverse[#2], #1}&, Reverse[vects], {2}], 1],
        opts
    ]

ListVectorFieldPlot[v_, ___] := Null/;(
    Message[ListVectorFieldPlot::lpvf]; False
    )
    
(* VectorFieldPlot takes a function that generates vectors *)
Options[VectorFieldPlot] =
    Sort[Join[Options[ListVectorFieldPlot], {PlotPoints -> Automatic}]];

SetAttributes[VectorFieldPlot, HoldFirst];

(* Note: the slightly odd specification of range increment is for backwards
  compatibility. Users should preferentially use PlotPoints, and not the
  range increment arguments; however, so that it will usually still work
  the way it used to, we allow PlotPoints to take the value Automatic;
  when set, the range argument is used. (If the range argument is also
  Automatic, the increment is treated as if PlotPoints were set to 15.)
  Otherwise, the value of PlotPoints will override whatever is specified
  in the range argument. Eventually, we should phase out the range
  argument completely. --JMN Jan. 94 *)
VectorFieldPlot[f_, {u_, u0_?numberQ, u1_?numberQ, du_:Automatic},
             {v_, v0_?numberQ, v1_?numberQ, dv_:Automatic}, opts___?OptionQ] :=
    Module[{plotpoints, dua, dva, vecs, xpp, ypp, sf},
      (* -- grab options -- *)
        {plotpoints, sf} = {PlotPoints, ScaleFactor}/.Flatten[{opts}]/.
             Options[VectorFieldPlot];
        If[Head[plotpoints] === List,
            xpp = First[plotpoints];
            ypp = Last[plotpoints],
          (* else *)
            xpp = ypp = plotpoints
        ];
      (* determine interval between bases of vectors *)
        If[!IntegerQ[xpp],
            dua = automatic[du,(u1 - u0)/14],
            dua = (u1 - u0)/(xpp - 1)
        ];
        If[!IntegerQ[ypp],
            dva = automatic[dv,(v1 - v0)/14],
            dva = (v1 - v0)/(ypp - 1)
        ];
      (* set the scaling factor based on the intervals if it is not
            explicitly None or a number *)
        If[ sf =!= None && !numberQ[sf],
            sf = N[Min[dua, dva]]
        ];
      (* -- determine the vectors -- *)
        vecs = Flatten[Table[{N[{u,v}],N[f]},
            Evaluate[{u,u0,u1,dua}],Evaluate[{v,v0,v1,dva}]],1];
      (* call ListVectorFieldPlot *)
        ListVectorFieldPlot[vecs,
            (* note dependency on LPVF filtering its own opts quietly *)
            Flatten[{ScaleFactor -> sf, opts, Options[VectorFieldPlot]}]
        ]/;MatchQ[vecs, {{_?VectorQ, _?VectorQ}..}]
    ]

(* GradientFieldPlot - computes the gradient of a scalar function and
   calls VectorFieldPlot on the result *)
Options[GradientFieldPlot] = Options[VectorFieldPlot];

GradientFieldPlot[function_, 
        {u_, u0__}, 
        {v_, v0__},
        options___] :=
    VectorFieldPlot[Evaluate[{D[function, u], D[function, v]}],
                {u, u0},
                {v, v0},
                options, Options[GradientFieldPlot]]

(* HamiltonianFieldPlot - computes the hamiltonian field from a scalar
   function and passes it on to VectorFieldPlot *)
Options[HamiltonianFieldPlot] = Options[VectorFieldPlot];

HamiltonianFieldPlot[function_, 
        {u_, u0__}, 
        {v_, v0__},
        options___] :=
    VectorFieldPlot[Evaluate[{D[function, v], -D[function, u]}],
                {u, u0},
                {v, v0},
                options, Options[HamiltonianFieldPlot]]


(* PolyaFieldPlot takes a complex scalar function and produces a vector
   field in the complex plane, with vector magnitudes scaled as indicated. *) 
Options[PolyaFieldPlot] = Options[VectorFieldPlot];

SetOptions[PolyaFieldPlot, ScaleFunction -> (Log[# + 1]&) ]

SetAttributes[PolyaFieldPlot, HoldFirst]

PolyaFieldPlot[f_, x_List, y_List, opts___] :=
    VectorFieldPlot[{Re[#], -Im[#]} & @ f, x, y, opts,
      Options[PolyaFieldPlot] ]
                
End[]   (* `Private` *)

EndPackage[]    (* VectorFieldPlots` *)

(*:Limitations: None known. *)


(*:Examples:

VectorFieldPlot[ {Sin[x],Cos[y]},{x,0,Pi},{y,0,Pi}]

VectorFieldPlot[ { Sin[x y], Cos[x y] },{x,0,Pi},{y,0,Pi}]

GradientFieldPlot[ x^3 + y^4,{x,0,10},{y,0,10}]

PolyaFieldPlot[ (x+I y)^4,{x,5,10},{y,5,10}]


*)








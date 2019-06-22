(* ::Package:: *)

(* :Title: Computational Geometry *)

(* :Context: ComputationalGeometry` *)

(* :Author:
    Damrong Guoy, John M. Novak, E.C. Martin
*)

(* :Copyright: Copyright 1990-2007, Wolfram Research, Inc.  *)

(* :Package Version: 2.0 *)

(* :Mathematica Version: 4.0 *)

(* :Summary:
This package implements selected functions from computational geometry
relevent to nearest neighbor point location, including triangulation.
*)

(* :Keywords: Delaunay triangulation, Voronoi diagram, nearest neighbor,
		convex hull, computational geometry
*)

(* :History: 
    Original Version by E.C. Martin (Wolfram Research), 1990-1991.
    Improved handling of high precision data by E.C. Martin, 1992.
    Version 1.2, added Bounded Diagram and TileAreas, ECM, 1996-1998.
    Version 1.2.1, added ability to handle set of collinear points to
            VoronoiDiagram, John M. Novak, September 1998.
    Version 2.0 by Damrong Guoy, August 2000. Completely revised
         ConvexHull to be much more efficient.
*)

(* :Requirements: No special system requirements. *)

(* :Warnings:  None. *)

(* :Limitations: Computational efficiency is maximized for non-collinear points.
BoundedDiagram fails if the boundary vertices do not fall into unique
Voronoi polygons (open or closed) in the unbounded Voronoi diagram.  This
limitation can be avoided with a more careful implementation.
*)
	

(* :Sources:
[Prep 85]  F. R. Preparata & M. I. Shamos, Computational Geometry: 
           An Introduction, Springer-Verlag, 1985.
[Lee 80]   D. T. Lee & B. J. Schachter, "Two Algorithms for constructing 
           a Delaunay triangulation," Int. J. of Computer & Information 
           Sciences, Vol. 9, No. 3, pp. 219-242, 1980.
[Guib 85]  L. Guibas & J. Stolfi, "Primitives for the Manipulation of General
	       Subdivisions and the Computations of Voronoi Diagrams," ACM Trans.
	       on Graphics, Vol. 4, No. 2, pp. 74-123, 1985.	
*)

(* :Discussion:

   The Delaunay triangulation is represented as a vertex adjacency list
    {{v1,{v11,v12,...}},...,{vn,{vn1,vn2,...}} where vi is the label of 
    the ith vertex of the triangulation and indicates the position of the
    point in the original coordinate list (this may not be position i if 
    there are duplicates in the original list).  The list {vi1,vi2,...}
    are the vertices adjacent to vertex vi, listed in counterclockwise order.
    If vi is on the convex hull, then vi1 is the next counterclockwise point
    on the convex hull.

   The Voronoi diagram is represented as (1) a list of Voronoi polygon vertex
    coordinates and (2) a vertex adjacency list: {{q1,...,qm}, {{v1,{u11,u12,
    ...}},...,{vn,{un1,un2,...}}}.  The Voronoi polygons at the periphery of
    the diagram are unbounded and this means that the qj may be a true Voronoi
    vertex {rj,sj} or a "quasi" vertex Ray[{rj,sj},{r1j,s1j}].  A closed 
    Voronoi polygon will be defined by a list of true vertices, while an open
    Voronoi polygon will be defined by a list containing two or more true 
    vertices and precisely two Rays located adjacently in the list.
    Ray[{rj,sj},{r1j,s1j}] indicates a ray having origin {rj,sj} (a true
    vertex) and containing point {r1j,s1j}.
    In the vertex adjacency list vi is the label of the ith vertex of the
    dual Delaunay triangulation which is also the defining  point of the ith
    Voronoi polygon.  The list {ui1,ui2,...} contains the Voronoi vertices
    that compose the ith polygon, listed in counterclockwise order.

    The function BoundedDiagram is designed to use the results of the
    function VoronoiDiagram.

    When data is collected at points in the plane within a polygonal region, it
    is useful to bound the unbounded Voronoi diagram of the points in the plane 
    with the bounding polygon.  BoundedDiagram begins by finding the unbounded 
    Voronoi diagram. It then works counterclockwise around the boundary, 
    integrating bounding polygon vertices into the diagram, and deleting Voronoi
    diagram vertices falling outside of the boundary.

    Bounding an open tile of the Voronoi diagram allows one to approximate
    the "true" underlying closed tile one would have if the data collection
    had not been limited to a portion of the plane.
*)

BeginPackage["ComputationalGeometry`"]

If[FileType[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"]]===File,
Select[FindList[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"],"ComputationalGeometry`"],
StringMatchQ[#,StartOfString~~"ComputationalGeometry`*"]&]//ToExpression;
];

If[Not@ValueQ[DelaunayTriangulation::usage],DelaunayTriangulation::usage =
"DelaunayTriangulation[{{x1,y1},{x2,y2},...,{xn,yn}}] yields the (planar) \
Delaunay triangulation of the points. The triangulation is represented as a \
vertex adjacency list, one entry for each unique point in the original \
coordinate list indicating the adjacent vertices in counterclockwise order."];
(* A duplicate point is indexed according to the location of the first instance
	of the point in the original input list. *)
(* O(n log(n)) time *)

If[Not@ValueQ[VoronoiDiagram::usage],VoronoiDiagram::usage =
"VoronoiDiagram[{{x1,y1},{x2,y2},...,{xn,yn}}] yields the (planar) Voronoi \
diagram of the points. The diagram is represented as two lists: (1) a Voronoi \
vertex coordinate list, and (2) a vertex adjacency list, one entry for each \
unique point in the original coordinate list indicating the associated Voronoi \
polygon vertices in counterclockwise order. VoronoiDiagram[{{x1,y1},{x2,y2}, \
...,{xn,yn}},val] takes val to be the vertex adjacency list of the dual \
Delaunay triangulation. VoronoiDiagram[{{x1,y1},{x2,y2},...,{xn,yn}},val,hull] \
takes hull to be the convex hull of the unique points."];
(* Supplying more arguments to VoronoiDiagram reduces computation time. *)
(* A duplicate point is indexed according to the location of the first instance
	of the point in the original input list. *)
(* O(n log(n)) time *)

If[Not@ValueQ[ConvexHull::usage],ConvexHull::usage =
"ConvexHull[{{x1,y1},{x2,y2},...,{xn,yn}}] yields the (planar) convex hull \
of the n points, represented as a list of point indices arranged in \
counterclockwise order."];
(* A duplicate point is indexed according to the location of the first instance
	of the point in the original input list. *)
(* O(n log(n)) time *)

If[Not@ValueQ[ConvexHullMedian::usage],ConvexHullMedian::usage =	
"ConvexHullMedian[{{x11, ..., x1p}, ..., {xn1, ..., xnp}}] estimates the \
p-dimensional median to be the mean of the p-dimensional vectors lying on the \
innermost layer of the convex layers of the n p-dimensional points."];

If[Not@ValueQ[ConvexHullArea::usage],ConvexHullArea::usage =
"ConvexHullArea[{{x1, y1}, ..., {xn, yn}}] gives the area \
of the convex hull of the bivariate data."];
(* This should be extended to dimensions > 2 as soon as convex hull for
dimensions > 2 is implemented. *)

If[Not@ValueQ[NearestNeighbor::usage],NearestNeighbor::usage =
"NearestNeighbor[pt, pts] is an obsolete function. Use
Nearest[pts -> Automatic, pt] instead."];
(* old usage:
"NearestNeighbor[{a,b},{{x1,y1},{x2,y2},...,{xn,yn}}] yields the label of the \
nearest neighbor of {a,b} out of the points {{x1,y1},{x2,y2},...,{xn,yn}}. \
NearestNeighbor[{a,b},{q1,q2,...,qm},val] yields the label of the nearest \
neighbor of {a,b} using the Voronoi vertices qj and the Voronoi vertex \
adjacency list val. NearestNeighbor[{{a1,b1},...,{ap,bp}},{{x1,x2},..., \
{xn,yn}}] or NearestNeighbor[{{a1,b1},...,{ap,bp}},{q1,q2,...,qm},val] yields \
the nearest neighbor labels for the list {{a1,b1},...,{ap,bp}}."
*)

If[Not@ValueQ[TriangularSurfacePlot::usage],TriangularSurfacePlot::usage =
"TriangularSurfacePlot[{{x1,y1,z1},...,{xn,yn,zn}}] plots the zi according \
to the planar Delaunay triangulation established by the {xi,yi}. \
TriangularSurfacePlot[{{x1,y1,z1},...,{xn,yn,zn}},val] plots the zi according \
to the planar triangulation of the {xi,yi} stipulated by the vertex adjacency \
list val."];

If[Not@ValueQ[PlanarGraphPlot::usage],PlanarGraphPlot::usage =
"PlanarGraphPlot[{{x1,y1},{x2,y2},...,{xn,yn}}] plots the planar graph \
corresponding to the Delaunay triangulation established by the {xi,yi}. \
PlanarGraphPlot[{{x1,y1},{x2,y2},...,{xn,yn}},list] plots the planar graph of \
the {xi,yi} as stipulated by list; list may have the form {{l1,{l11,l12,...}, \
...,{ln,{ln1,ln2,...}}} (vertex adjacency list) or {l1,...,ln} (representing a \
single circuit) where li is the position of the ith unique point in the input \
list."];

If[Not@ValueQ[DiagramPlot::usage],DiagramPlot::usage =
"DiagramPlot[{{x1,y1},{x2,y2},...,{xn,yn}}] plots the {xi,yi} and the polygons \
corresponding to the Voronoi diagram established by the {xi,yi}. \
DiagramPlot[{{x1,y1},{x2,y2},...,{xn,yn}},{q1,q2,...,qm},val] plots the polygon \
'centers' {xi,yi} and polygon vertices qj as stipulated by the vertex adjacency \
list val."];

If[Not@ValueQ[DelaunayTriangulationQ::usage],DelaunayTriangulationQ::usage =
"DelaunayTriangulationQ[{{x1,y1},{x2,y2},...,{xn,yn}},val] returns True if the \
triangulation of the {xi,yi} represented by the vertex adjacency list val is a \
Delaunay triangulation, and False otherwise. DelaunayTriangulationQ[{{x1,y1}, \
{x2,y2},...,{xn,yn}},val,hull] takes hull to be the convex hull of the unique \
points. The val must be such that a point on the hull lists first the hull \
neighbor that follows the point on a counterclockwise traversal of the hull."];

If[Not@ValueQ[LabelPoints::usage],LabelPoints::usage =
"LabelPoints is an option of the plotting functions within \
ComputationalGeometry.m. LabelPoints -> True means that the points will be \
labeled according to their position in the input list.  Repeated instances of \
a point will be plotted once and labeled according to the position of the \
first instance in the list."];

If[Not@ValueQ[Ray::usage],Ray::usage =
"Ray[{x1,y1},{x2,y2}] is a means of representing the infinite rays found in \
diagrams containing open polygons. Here {x1,y1} indicates the head of the ray \
(a polygon vertex), and {x2,y2} indicates a point along the ray."];

If[Not@ValueQ[AllPoints::usage],AllPoints::usage =
"Allpoints is an option of ConvexHull indicating whether all distinct points \
on the hull or only the minimum set of points needed to define the hull are \
returned."];

If[Not@ValueQ[Hull::usage],Hull::usage =
"Hull is an option of DelaunayTriangulation indicating whether the convex hull \
is to be returned in addition to the vertex adjacency list val describing the \
triangulation. Hull -> False implies that val is returned, and Hull -> True \
implies that {val,hull} is returned."];

If[Not@ValueQ[TrimPoints::usage],TrimPoints::usage =
"TrimPoints is an option of DiagramPlot indicating the order of the diagram \
outlier vertex that lies on the PlotRange limit. The default TrimPoints -> 0 \
implies that all diagram vertices lie within PlotRange. TrimPoints -> n \
implies that PlotRange is such that the nth largest outlier vertex lies on the \
PlotRange limit."];
(* Regardless of the value of TrimPoints, PlotRange is never reduced below
what is needed to plot the original set of points (i.e., polygon "centers"). *)

(* using a syntax like the VoronoiDiagram syntax *)
If[Not@ValueQ[BoundedDiagram::usage],BoundedDiagram::usage =
"BoundedDiagram[{{a1,b1},{a2,b2},...,{ap,bp}},{{x1,y1},{x2,y2},...,{xn,yn}}] \
yields the bounded diagram formed by bounding the infinite Voronoi diagram of \
the points {{x1,y1},{x2,y2},...,{xn,yn}} by the polygon defined by the vertices \
{{a1,b1},{a2,b2},...,{ap,bp}}. The new bounded diagram is represented as \
two lists: (1) a vertex coordinate list, and (2) a a vertex adjacency list, one \
entry for each point in the original unbounded diagram indicating the \
associated bounded polygon vertices in counterclockwise order. \
BoundedDiagram[{{a1,b1},{a2,b2},...,{ap,bp}},{{x1,y1},{x2,y2},...,{xn,yn}},val] \
takes val to be the vertex adjacency list of the dual Delaunay triangulation. \
BoundedDiagram[{{a1,b1},{a2,b2},...,{ap,bp}},{{x1,y1},{x2,y2},...,{xn,yn}}, \
val,hull] takes hull to be the convex hull of the unique points."];
(* Supplying more arguments to BoundedDiagram reduces computation time. *)
(*  An unbounded
diagram will have vertices with head Ray, but the resulting bounded diagram
will have no Ray vertices. *)

If[Not@ValueQ[TileAreas::usage],TileAreas::usage =
"TileAreas[{{x1,y1},{x2,y2},...,{xn,yn}},{q1,q2,...,qm},val] finds the \
areas of the tiles centered on {xi,yi} and having vertices qj as stipulated by \
the vertex adjacency list val."];


Begin["`Private`"]

(* value to numericalize exact input to; this is in the form of a
   global so that it can be changed externally to the package. Possibly
   this should be controlled by an option, the design is unclear... *)
$ExactInputPrecision = MachinePrecision;

numericalQ[x_] := Apply[And,Map[NumberQ,Flatten[x]]]

(* for compatability *)
numericalize[x_] := First[numericalizeAndPrec[x]]

(* numericalize input to all the same precision; return numericalized
   input and precision; note assumption that input is a matrix
  (list of n-tuples) *)
numericalizeAndPrecision[x_] :=
Module [{prec, nx},
  prec = Internal`EffectivePrecision[x];
  If[prec === Infinity, prec = $ExactInputPrecision];
  nx = N[x, prec];
  If[MatrixQ[nx, NumberQ], {nx, prec}, {$Failed, prec}]
]

(*====================================== ConvexHull =========================================*)
(*
/* compute 2D convex hull using Graham's scan algorithm ([Prep 85] p.106)
*)
Options [ConvexHull] = {AllPoints->True}
ConvexHull::collinear  = "All points are collinear."
Off[ConvexHull::collinear] (* off by default *)
ConvexHull::duplicated = "`1` duplicated point(s) ignored."

ConvexHull [set:{{_,_}..}, opts___Rule]:=
Module[{coordinates, polarOrder, returnHull, computingPrecision, 
        allPointOption = TrueQ[AllPoints /.Flatten[{opts, Options[ConvexHull]}]]},
   
  {coordinates, computingPrecision} = numericalizeAndPrecision[set];
    
  ( (* block conditional on dataset being numericized *)       
    (* polar sort (angle, distance), polarOrder is an Integer-list of
       point indexes *)
      polarOrder = If [computingPrecision===MachinePrecision,
                   polarSortCompile   [coordinates, computingPrecision],
                   polarSortUnCompile [coordinates, computingPrecision]]; 
 
    (* check how many duplicated points *)
      If [Length [polarOrder] =!= Length [coordinates],
          Message [ConvexHull::duplicated,
                   Length [coordinates] - Length [polarOrder] ] ];
  

    (* polarSort has removed duplicated points for us, now check trivial case *)
      If [Length [polarOrder] <= 2, Return [polarOrder] ];
    (* check collinear points; if so, return non-duplicated points sorted
       by (x,y) *)
      If [collinearQ [coordinates [[polarOrder]], computingPrecision],

          Message [ConvexHull::collinear];
          Sort [Thread [{coordinates [[polarOrder]], polarOrder}] ] [[All, 2]],
        (* else Graham Scan *)

          GrahamScan [polarOrder, coordinates, computingPrecision,
                      allPointOption]
      ])/;
    coordinates =!= $Failed (* conditional checking that dataset numericized *)
]

(*============================== collinearQ ================================*)

collinearQ [coordinates:{{_,_}..}, precision_] :=
If [precision===MachinePrecision,
    collinearQCompile   [coordinates, precision],
    collinearQUnCompile [coordinates, precision]]

collinearQModule := (* args: coordinates, precision *)
Module [{x1,x2, y1,y2, a,b,c, x,y, 
         flat, chopSize = 10.^(1. - precision), i},
  {{x1,y1},{x2,y2}} = coordinates [[{1,2}]];
  a = y1-y2;
  b = x2-x1;
  c = x1*y2 - x2*y1;
  flat = True;
  Do[ {x,y} = coordinates [[i]]; 
      If [0 != Chop [a*x + b*y + c], flat = False; Break[]],
      {i,3,Length[coordinates]} ];
  flat
] // Hold
        
collinearQCompile = 
Function [compileBody,
          Compile [{{coordinates,_Real,2}, {precision, _Real}}, compileBody],
          {HoldAll}] @@ collinearQModule

collinearQUnCompile =
Function [module,
          Function @@ Hold [{coordinates, precision}, module],
          {HoldAll}] @@ collinearQModule
        
(*================================ GrahamScan ===============================*)

GrahamScan [polarOrder_, coordinates:{{_,_}..}, precision_, allPointOption_]:=
(* Perform Graham's scan. *)
Module[{xVals, nextPrev, next, prev, convexhull},
   
  (* Create circular-doubly-linked-list represented by two integer-lists : next and prev *)
  (* Assignment {listA, listB} = aCompileFunction[] results in unpacked array, do element-wise assignment instead *)
  nextPrev = createNextPrev [polarOrder, Length[coordinates]];
  next = nextPrev [[1]];  prev = nextPrev [[2]];
  
  (* Find the first point in (-Pi,Pi] polar-sorted list with maxX which automatically implies minY *)
  xVals = coordinates [[polarOrder, 1]];
  start = polarOrder [[ Position [xVals, Max [xVals], 1,1][[1,1]] ]];

      
  convexhull =  
  If [precision === MachinePrecision,
      GrahamScanMainLoopCompile   [start, next, prev, coordinates, precision, allPointOption],
      GrahamScanMainLoopUnCompile [start, next, prev, coordinates, precision, allPointOption]]
]

(*==================================== createNextPrev =======================================*)

createNextPrev = Compile [ {{polarOrder, _Integer, 1}, {numPoints, _Integer}}, 
                           (* duplicated points make numPoints > Length [polarOrder] *)
Module [{next, prev},
  next = prev = Table [0, {numPoints}];
  next [[polarOrder]] = RotateLeft  [polarOrder];
  prev [[polarOrder]] = RotateRight [polarOrder];
 {next, prev}
]]

(*========================== GrahamScanMainLoop{Compile/UnCompile} ==========================*)

GrahamScanMainLoopModule := (* args: start, inNext, inPrev, coordinates, precision, allPointOption *)
Module [{next = inNext, prev = inPrev,
         scanComplete, v, last, nextOfV, nextNextOfV,
         chopSize, 
         x1,x2,x3, y1,y2,y3, theSignOfArea,
         turn, left=1, right=0, n, out},         

  chopSize = 10.^(1.-precision);
         
  (* use the flag scanComplete to distinguish the two cases of reaching start point :
  /* completely scanning the whole ring, and backtracking all the way back.
  *)
  scanComplete = 0;  last = prev [[v = start]];
  While [ (start =!= next [[v]]) || (scanComplete === 0),
      If [ last === next [[v]], scanComplete = 1];
      nextNextOfV = next [[ nextOfV = next [[v]] ]];
      x1 = coordinates [[          v, 1]];
      x2 = coordinates [[    nextOfV, 1]];
      x3 = coordinates [[nextNextOfV, 1]];
      y1 = coordinates [[          v, 2]];
      y2 = coordinates [[    nextOfV, 2]];
      y3 = coordinates [[nextNextOfV, 2]];
      theSignOfArea = Sign [Chop[ (x2 - x3) (y2 - y1) + (x2 - x1) (y3 - y2), chopSize]];          
     
      Which[ theSignOfArea == 1,  turn = left,
             theSignOfArea == -1, turn = right,
            ({x1,y1}-{x2,y2}).({x2,y2}-{x3,y3}) < 0, (* check 180-degree, treat as right-turn except at start *)
                                  turn = If [nextOfV =!= start, right, left],
            allPointOption,       turn = left,
            True,                 turn = right];
         
      If [ turn == left, v = nextOfV,  (* forward scan *)
           prev [[ next [[v]] = nextNextOfV ]] = v; v = prev [[v]]   (* del next and backtrack *)
      ];
  ];
  
  (* extract the convex hull *)
  out = Table[0, {Length [next]} ];
  n = 0;  v  = start;  out [[++n]] = v; v = next [[v]];
  While [ v != start,  out [[++n]] = v; v = next [[v]] ];
  
  Take [out, n]
  
]  // Hold

(*
  Compile with CompileOptimizations->False for now. Please see:
  http://hypermail.wolfram.com/archives/wri-release/2001/May/msg00045.html
  for the reason why.  When this is fixed, we can revert to normal
  behavior.
  
  -- 3/2010: This has been reverted, nine years later, because CompileOptimizations is
    no longer an option to Compile.
*)

GrahamScanMainLoopCompile =
Function [compileBody,
          Compile [ {{start,_Integer}, {inNext, _Integer, 1}, {inPrev, _Integer, 1}, 
                    {coordinates, _Real, 2}, {precision, _Real}, {allPointOption, True|False }},
                    compileBody ],
          {HoldAll}] @@ GrahamScanMainLoopModule
          
GrahamScanMainLoopUnCompile =
Function [module,
          Function @@ Hold[{start, inNext, inPrev, coordinates, precision, allPointOption}, module],
          {HoldAll}] @@ GrahamScanMainLoopModule

(*================================ polarSort ===============================*)
(* Preprocessing of Graham's scan algorithm for Convex Hull construction.
/* Return a list of point indexes corresponding to points in polar-sorted order.
/* Explicitly store the angle and length in order to use Sort with no second argument.
*)
polarSortModule := (* args: coordinates, precision *)
Module[{disp, iPoint, zero, angleLengthIndex, 
        n, list, previousPoint},
  
  (* compute {{angle,length,index},...} *)      
  iPoint = Apply [Plus, coordinates]/ Length [coordinates];
  disp   = Map [(#-iPoint)&, coordinates];
  zero   = If [precision===MachinePrecision, N[0], SetAccuracy[0, precision]];
  angleLengthIndex =
      Transpose[{
          Map [If [#=={0,0}, zero, ArcTan[#[[1]],#[[2]] ]]&,disp],
          Map [(#.#)&, disp],
          N [Range [Length [disp] ] ] }];      (* machine-precision for packed-array *)

  angleLengthIndex = Sort [angleLengthIndex];

  (* collect non-duplicated points *)
  list = Table [0, {Length [angleLengthIndex]}];
          list [[ n=1 ]] =  angleLengthIndex [[1,  3  ]] // Round;
          previousPoint  =  angleLengthIndex [[1,{1,2}]];
  Do[ If [previousPoint =!= angleLengthIndex [[i,{1,2}]],
          previousPoint  =  angleLengthIndex [[i,{1,2}]];
          list [[ ++n ]] =  angleLengthIndex [[i,  3  ]] // Round ],
      {i, 2, Length [angleLengthIndex] } ];
      
  Take [list, n]
  
] // Hold

polarSortCompile = 
Function [compileBody,
          Compile [{{coordinates,_Real,2}, {precision, _Real}}, compileBody],
          {HoldAll}] @@ polarSortModule

polarSortUnCompile =
Function [module,
          Function @@ Hold[{coordinates, precision}, module],
          {HoldAll}] @@ polarSortModule

(*=============================== SignOfArea =================================*)
(* This one is still here because DelaunTriangulation and others need it.
/* --Damrong
*)

(* left turn if positive, right turn if negative *)

SignOfArea[{x1_,y1_},{x2_,y2_},{x3_,y3_}]:=
  Module[{area = (x2 - x3) (y2 - y1) + (x2 - x1) (y3 - y2), prec},
    prec = (Internal`EffectivePrecision[area]/. Infinity -> MachinePrecision);
    Sign[Chop[area, 10^(1-prec)]]
  ] (* assume that inputs are numerical
		/; numericalQ[ N[{x1,y1,x2,y2,x3,y3}] ] *)

SignOfArea[{x1_,y1_},{x2_,y2_},{x3_,y3_}]:= 0 /; ((x1-x2 == x1-x3 ==
				x2-x3 == 0) || (y1-y2 == y1-y3 == y2-y3 == 0))


(*=============================== CollinearQ =================================*)

CollinearQ[set:{{_,_}..}] :=
  Module[{collinearq, length = Length[set], nset = N[set]},
      collinearq = Scan[Module[{next = succ[#,length]},
				If[ Apply[SignOfArea,
				  nset[[ {#,next,succ[next,length]} ]] ] != 0,
				  Return[False]
                                ]
                       ]&,
		       Range[length]
             ];
      collinearq =!= False
  ]   (* assume that inputs are numerical
		/; numericalQ[N[set]] *)



(*=============================== Distinct =================================*)
(* This one is still here just because DelaunayTriangulation needs it.
/* I don't like the way Distinct iteratively call AppendTo to create the
/* list of distinct points.  It takes O(n^2). 
/* --Damrong
*)
(* returns an object of the form {{l1,{x1,y1}},{l2,{x2,y2}},...,{ln,{xn,yn}}} *)
Distinct[orig:{{_,_}..}]:=
Module[{union,distinct={}},
    (* {0,0}, {0,0.}, and {0.,0.} must be specifically mapped to {0.,0.}
	/*  before Union recognises duplicates 
	/*)
	union=Union[Map[If[#=={0,0},{0.,0.},#]&,orig]];
	
	(* Find positions of duplicates & generate unsorted list of unique
	/* coordinates labeled according to their position in the original list. 
	/*)
	Scan[
 	    (If[#=={0.,0.},
	         (* {0,0}, {0,0.}, {0.,0} & {0.,0.} represents a special case *)
	         position=Flatten[Join[
		                  Position[orig,{0,0} ],Position[orig,{0,0.} ],
	                      Position[orig,{0.,0}],Position[orig,{0.,0.}]]],
	         position=Flatten[Position[orig,#]]];
	      AppendTo[distinct,{position[[1]],orig[[position[[1]]]]}]
	     )&,
	     union
	];
	distinct
]



(*================================ PolarSort ===============================*)
(* This one is still here just because DelaunayTriangulation needs it.
/* I don't like the way it uses non-canonical Sort with pure Function and
/* non-homogenious non-packed array.
/* It costs a huge hidden constant to the time complexity.
/* --Damrong
*)

PolarSort[l:{{_,{_,_}}..}, cent_:Automatic]:=
Module[{n=Length[l],origin,p1,p2,in,sorted},
	(* The centroid of the points is interior to the convex hull, if not
       otherwise specified. *)
    If[MatchQ[cent, {_?NumberQ, _?NumberQ}],
        origin = cent,
	    origin=Apply[Plus,Map[#[[2]]&,l]]/n
    ];
	(* 1st component of elements of 'in' is label,
	   2nd component of elements of 'in' is original coordinate,
	   3rd component of elements of 'in' is centered coordinate,
	   4th component of elements of 'in' is polar angle *)
	in = Map[Join[#, {#[[2]]-origin, PolarAngle[#[[2]]-origin]}]&, l];
	sorted = Sort[in,
	              Function[{p1,p2},
		                p1[[4]]<p2[[4]] ||
				(* Changed the test p1[[4]]==p2[[4]] to
				p1[[4]]-p2[[4]]==0 for numerical precision. *)
		               (p1[[4]]-p2[[4]]==0 && 
		               (p1[[3,1]]^2 + p1[[3,2]]^2 <  
		                p2[[3,1]]^2 + p2[[3,2]]^2))
		      ] (* end Function *)
		 ]; (* end Sort *)
	Map[Drop[#, -2]&, sorted]
	] 	(* assume that inputs are numerical
			/;      numericalQ[N[l]] *)

(*============================== PolarAngle ===============================*)

(* if point is on origin, define angle as zero *)
PolarAngle[{x_,y_}]:= If[x == 0 && y == 0, 0,
     If[ y>=0,
	Arg[x + I y],
	Arg[x + I y] + N[2 Pi, Precision[x + I y]]
		      ] ](* assume numerical inputs
						/; numericalQ[N[{x, y}]] *)

(*============================== CommonTangents ===========================*)

(* CommonTangents[set,hullL,hullR,rmL,lmR] *)
(* Returns {{l1,r1},{l2,r2}}.  l1 indicates the position in hullL of the
   leftmost point of the lower common tangent (of hullL & hullR).  r1 indicates
   the position in hullR of the rightmost point of the lower common tangent.
   l2 indicates the position in hullL of the leftmost point of the upper common
   tangent.  r2 indicates the position in hullR of the rightmost point of the
   upper common tangent. *)
(* Below 'rightmostLeft' is a pointer to the rightmost point in hullLeft,
   while 'leftmostRight' is a pointer to the leftmost point in hullRight. *)


CommonTangents[set:{{_,_}..}, hullLeft:{_Integer..}, hullRight:{_Integer..},
	       rightmostLeft_Integer, leftmostRight_Integer] :=
  Module[{x, y, predx, succy, succx, predy, lowertan={}, uppertan={}, 
	  lengthLeft = Length[hullLeft], lengthRight = Length[hullRight],
	  soaR, soaL},
    
    (*  LOWER TAN *)
    {x,y} = {rightmostLeft,leftmostRight};
    {predx,succy} = {pred[x,lengthLeft],succ[y,lengthRight]};
    While[ SameQ[lowertan,{}],
      soaR = SignOfArea[ set[[hullLeft[[x]]]], set[[hullRight[[y]]]],
			set[[hullRight[[succy]]]] ];
      If[ soaR <= 0,
	  If[ soaR == 0, Return[comtan[set,hullLeft,hullRight]] ];
          (* update y (by moving ccw on hullRight to succy) *)
          {succy,y} = {succ[succy,lengthRight],succy}, 
	  soaL = SignOfArea[ set[[hullLeft[[predx]]]], set[[hullLeft[[x]]]],
			    set[[hullRight[[y]]]] ];
          If[ soaL <= 0,
		If[ soaL == 0, Return[comtan[set,hullLeft,hullRight]] ];
                (* update x (by moving cw on hullLeft to predx) *)
                {predx,x} = {pred[predx,lengthLeft],predx},
		(* otherwise {x,y} points to the lower common tangent *)
		lowertan = {x,y}
          ]
      ]
    ];

    (* UPPER TAN *)
    {x,y} = {rightmostLeft,leftmostRight};
    {succx,predy} = {succ[x,lengthLeft],pred[y,lengthRight]};
    While[ SameQ[uppertan,{}], 
      soaR = SignOfArea[ set[[hullLeft[[x]]]], set[[hullRight[[y]]]],
			 set[[hullRight[[predy]]]] ];
      If[ soaR >= 0,
	  If[ soaR == 0, Return[comtan[set,hullLeft,hullRight]] ];
          (* update y (by moving cw on hullRight to predy) *)
          {predy,y} = {pred[predy,lengthRight],predy},
	  soaL = SignOfArea[ set[[hullLeft[[succx]]]], set[[hullLeft[[x]]]],
			     set[[hullRight[[y]]]] ];
          If[ soaL >= 0,
		If[ soaL == 0, Return[comtan[set,hullLeft,hullRight]] ];
                (* update x (by moving ccw on hullLeft to succx) *)
                {succx,x} = {succ[succx,lengthLeft],succx},
		(* otherwise {x,y} points to the upper common tangent *)
		uppertan = {x,y}
          ]
      ]
    ];

    {lowertan,uppertan}
  ] 


  (*** comtan is inefficient, but it will not be called unless points are  
     	collinear.  This inefficiency is not necessary. *)
comtan[set:{{_,_}..}, hullLeft:{_Integer..}, hullRight:{_Integer..}] :=
   Module[{hull = ConvexHull[ Join[ set[[hullLeft]], set[[hullRight]] ] ],
	   leng1 = Length[hullLeft], leng2 = Length[hullRight],
	   leng, uppertan, lowertan},
      leng = Length[hull];
      While[ !(hull[[1]] < leng1+1 && hull[[leng]] > leng1),
		hull = RotateLeft[hull]];
      (* Now the first points in hull belong to hullLeft. *)
      uppertan = { hull[[1]], hull[[leng]] - leng1 }; 
      While[ !(hull[[1]] > leng1 && hull[[leng]] < leng1+1),
		hull = RotateLeft[hull]];
      (* Now the first points in hull belong to hullRight. *)
      lowertan = { hull[[leng]], hull[[1]] - leng1 };
      {lowertan,uppertan}
   ]
	
	


(*======================== DelaunayTriangulation ===========================*)

Options[DelaunayTriangulation] = {Hull->False}

DelaunayTriangulation::collin = "Points `` are collinear."

DelaunayTriangulation[set:{{_,_}..},opts___Rule]:=
   Module[{orig, distinct, sorted, label, unlabeled,
	   delaunay, delval, hull, leftmost, rightmost,
	   returnhull = Hull /. {opts} /. Options[DelaunayTriangulation]},
     (
	 (* Make sure numerical coordinates are distinct and labeled
	    according to position in original list.  Also, order 
	    according to x coordinate before recursive triangulation. *)
	 sorted = Sort[distinct,
          If[Chop[#1[[2,1]] - #2[[2,1]], 10^(1 - Internal`EffectivePrecision[{#1,#2}])] == 0, #1[[2,2]] > #2[[2,2]],
                 #2[[2,1]] == #2[[2,1]] ] &
		(*	 (#1[[2,1]]<#2[[2,1]] ||
			  (* Changed the test #1[[2,1]]==#2[[2,1]] to
			#1[[2,1]]-#2[[2,1]]==0 for numerical precision. *)
			  (#1[[2,1]]-#2[[2,1]]==0 && #1[[2,2]]>#2[[2,2]]))& *)
                  ];
         (* Save labels, but use points sans labels in calculating
	      triangulation. *)
         {label,unlabeled} = Transpose[sorted];
	 (* Recursive triangulation. *)
	 delaunay = Del[unlabeled];
	 If[ SameQ[delaunay,$Failed],
		Return[$Failed],
	        {delval,hull,leftmost,rightmost} = delaunay
	 ];
	 (* Add vertex label field to delval and relabel delval with positions
	    of points in original set. *)
         delval = Map[Module[{v=#[[1]],adjlist=#[[2]]},
	                	{label[[v]],
		                 Map[label[[#]]&,adjlist]}
			    ]&,
                      Transpose[{Range[Length[delval]],delval}]
                  ];
	 (* Sort vertex adjacency list according to vertex label. *)
	 delval = Sort[delval,(#1[[1]] < #2[[1]])&];
         If[returnhull,
	    (* Relabel convex hull with positions of points in original set. *)
	    hull = Map[label[[#]]&,hull];
	    {delval,hull},
	    delval
         ]
     ) /; ( ((orig = numericalize[set]) =!= $Failed) &&
	  (distinct = Distinct[orig]; Length[distinct] >= 3) )
   ] 






(************* 2 point triangulation *************)
(***** returns {delval,convexhull,leftmost,rightmost} *****)
Del[s:{{x1_,y1_},{x2_,y2_}}] :=
  Module[{hull},
     (* The point that has the smallest ordinate out of all rightmost points
	is listed first. *)
     (* Changed the test x1==x2 to x1-x2==0 for numerical precision. *)
     hull = If[ x1 > x2 || (x1-x2==0 && y1 < y2),
		{1,2},
		{2,1}];
     {{{2},{1}},hull,2,1}
  ]



(************* 3 point triangulation (non-collinear) *************)
(***** returns {delval,convexhull,leftmost,rightmost} *****)
Del[s:{{_,_},{_,_},{_,_}}] :=
  Module[{sorted,hull,maxx,miny,maxxlist,val,leftmost},
     sorted = PolarSort[Transpose[{Range[3],s}]];
     {hull,sorted} = Transpose[sorted];
     maxx = Apply[Max, Map[#[[1]]&,sorted]];
     (* Changed the test (#[[1]]==maxx)& to (#[[1]]-maxx==0)& for
		numerical precision. *)
     maxxlist = Select[sorted, (#[[1]]-maxx==0)&];
     miny = Apply[Min, Map[#[[2]]&,maxxlist]];
     (* Rotate hull so that rightmost point having the smallest ordinate
		is listed first. *)
     While[ s[[ First[hull] ]] != {maxx,miny} ,
		hull = RotateLeft[hull]
     ];
     val = Map[Module[{p = Position[hull,#][[1,1]]},
		    Drop[ RotateLeft[hull,p], -1]
                  ]&,
	          Range[3]];
     leftmost = If[ s[[hull[[2]],1]] < s[[hull[[3]],1]] ||
		(* Changed the test s[[hull[[2]],1]] == s[[hull[[3]],1]] to
			s[[hull[[2]],1]] - s[[hull[[3]],1]] == 0 for
			numerical precision. *)
		    	(s[[hull[[2]],1]] - s[[hull[[3]],1]] == 0 &&
			 s[[hull[[2]],2]] > s[[hull[[3]],2]]),
			2,
			3];
     {val,hull,leftmost,1}
  ]



(**************** >=6 point triangulation *****************)
(***** returns {delval,convexhull,leftmost,rightmost} *****)
Del[s:{{_,_}..}]:=
   Module[{leng0 = Length[s], leng1, leng2, val, hull, lenghull, order,
	leftptr, rightptr, xcoord, minx, maxx, lowertan, uppertan,
	s1, val1, hull1, l1, r1, lenghull1, num1, part1, lpart1,
	s2, val2, hull2, l2, r2, lenghull2, num2, part2, lpart2,
	LTl, LTr, UTl, UTr, collinearq1 = False, collinearq2 = False},
       If[ CollinearQ[s],
		Message[DelaunayTriangulation::collin,s];
		Return[$Failed]
       ];
       leng1 = Ceiling[leng0/2];
     (** s[[i]] is mapped into s1[[i]] for i=1,...,leng1 *)
       s1 = Take[s,leng1]; 
       leng2 = leng0 - leng1;
     (** s[[i]] is mapped into s2[[i-leng1]] for i=leng1+1,...,leng0 *)
       s2 = Drop[s,leng1];
     (** Delaunay triangulation of the subsets s1 & s2. *)
     (** If either subset is collinear, need information about overall hull,
		in order to compute a suitable triangulation for the collinear
		subset. *)
       collinearq1 = leng1 > 2 && CollinearQ[s1];
       collinearq2 = leng2 > 2 && CollinearQ[s2];
     (** Computing ConvexHull is time consuming; presumably there will be few
		cases where (collinearq1 || collinearq2) == True. *)
       If[ collinearq1 || collinearq2,
	(************ "Time-consuming" procedure for collinear points. ******)
		hull = ConvexHull[s];
		lenghull = Length[hull];
		While[ !(hull[[1]] < leng1+1 && hull[[lenghull]] > leng1),
			hull = RotateLeft[hull]
		];
	      (** Now points in s1 listed first in hull. *)
		lpart1 = Length[Select[hull,(# < leng1+1)&]];
		uppertan = {hull[[1]],hull[[lenghull]]};
		While[ !(hull[[1]] > leng1 && hull[[lenghull]] < leng1+1),
			hull = RotateLeft[hull]
		];
	      (** Now points in s2 listed first in hull. *)
		lpart2 = Length[Select[hull,(# > leng1)&]];
		lowertan = {hull[[lenghull]],hull[[1]]};
	      (** Calculate ptrs to leftmost & rightmost points in hull. *)
		xcoord = Map[#[[1]]&, s[[hull]] ];
		{minx,maxx} = {Min[xcoord],Max[xcoord]};
		leftptr = Position[xcoord,minx][[1,1]];
		rightptr = Position[xcoord,maxx][[1,1]];
	      (** Calculate subset adjacency lists such that merge will
		  work properly as it adds edges from lowertan to uppertan. *)
		val1 = If[ collinearq1,
			  If[ lpart1 == 1 ||
				(lpart1 > 1 && lowertan[[1]] != 1),
				  Join[{{2}},
				      Map[{#+1,#-1}&,Range[2,leng1-1]],
				       {{leng1-1}}],
				  Join[{{2}},
				      Map[{#-1,#+1}&,Range[2,leng1-1]],
				       {{leng1-1}}]
			  ],
			  Del[s1][[1]]
		       ];
                val2 = If[ collinearq2,
			  If[ lpart2 == 1 ||
				(lpart2 > 1 && lowertan[[2]] == leng1+1),
				  Join[{{leng1+2}},
				       Map[{#+1,#-1}&,
					 Range[leng1+2,leng1+leng2-1]],
				       {{leng1+leng2-1}}],
				  Join[{{leng1+2}},
				       Map[{#-1,#+1}&,
					 Range[leng1+2,leng1+leng2-1]],
				       {{leng1+leng2-1}}]
			  ],
			  Del[s2][[1]] + leng1
		      ],
        (************ "Normal" procedure for non-collinear points. ******)
       		{val1,hull1,l1,r1} = Del[s1];
       		{val2,hull2,l2,r2} = Del[s2];
     	      (** Offset labels in val2 & hull2 by number of points in s1. *)
       		val2+=leng1; hull2+=leng1;  
     	      (** Compute ptrs to lower & upper tangents of the two hulls. *)
       		lenghull1 = Length[hull1];
		lenghull2 = Length[hull2];
       		{{LTl,LTr},{UTl,UTr}} = CommonTangents[s,hull1,hull2,r1,l2];
       		lowertan = {hull1[[LTl]],hull2[[LTr]]};
       		uppertan = {hull1[[UTl]],hull2[[UTr]]};
	      (** num1 is # of points in hull1 also in hull *)
		num1 = Mod[LTl-UTl,lenghull1]+1;
	      (** part1 is the part of hull1 also in hull *)
		part1 = Take[RotateLeft[hull1,UTl-1],num1];
		leftptr = Mod[l1-UTl,lenghull1]+1;
	      (** num2 is # of points in hull2 also in hull *)
		num2 = Mod[UTr-LTr,lenghull2]+1;
	      (** part2 is the part of hull2 also in hull *)
		part2 = Take[RotateLeft[hull2,LTr-1],num2];
		rightptr = Mod[r2-LTr,lenghull2]+1+num1;
		hull = Join[part1,part2]
       ];
     (** Merge vertex adjacency lists val1 & val2 into val.  *)
       val = MergeDel[s,
		      Join[val1,val2],
		      lowertan, 		(* lowertan points *)
		      uppertan  		(* uppertan points *)
             ];
     (** Return new vertex adjacency list, convex hull, and pointers to 
	 leftmost & rightmost points in hull. *)
       {val,hull,leftptr,rightptr}
   ]


MergeDel[s:{{_,_}..},val:{{_Integer..}..},
	 {LTl_Integer,LTr_Integer},{UTl_Integer,UTr_Integer}] :=
   Module[{edge = {LTl,LTr}, uppertan = {UTl,UTr}, l0 = LTl, r0 = LTr,
           merge = val, l0r0r1LeftTurn, r0l0l1RightTurn,
	   l1, l2, r1, r2, l1val, r1val,  
	   r0ptr, l0ptr, r1ptr, l1ptr, r2ptr, l2ptr, lengr0, lengl0},
       
       (* Lengths of r0's val & l0's val AFTER first edge is added. *)
       lengr0 = Length[merge[[r0]]] + 1;
       lengl0 = Length[merge[[l0]]] + 1;
       (* Insert lower tangent edge {l0,r0}. *)
       r0ptr = 1;                   (* r0ptr will point to r0 in l0's val *)
       l0ptr = lengr0;              (* l0ptr will point to l0 in r0's val *)
       (* insert r0 in l0's val at position r0ptr;
	  insert l0 in r0's val at position l0ptr *)
       {merge[[l0]],merge[[r0]]} = insert[merge[[{l0,r0}]],l0,r0,l0ptr,r0ptr];
     
     (* Merge the vertex adjacency lists by adding & deleting edges in val;
	work from lower tangent edge to upper tangent edge. *)
     While[ edge != uppertan,
       (* Initially {l0,r0,r1} form a left turn & {r0,l0,l1} form a right. *)
       l0r0r1LeftTurn = r0l0l1RightTurn = True;

       (********************* Derive points r1 and r2. ********************)
       r1ptr = pred[l0ptr,lengr0];     (* r1ptr will point to r1 in r0's val *)
       r1 = merge[[r0,r1ptr]];
       If[ SignOfArea[s[[l0]],s[[r0]],s[[r1]]] == 1,  (* left turn *)
	  r2ptr = pred[r1ptr,lengr0]; (* r2ptr will point to r2 in r0's val *)
	  r2 = merge[[r0,r2ptr]];
	  (* Delete {r0,r1} only if edge {r1,r2} still exists. *)
	  While[ MemberQ[merge[[r1]],r2] &&
	       ((* {r0, r2, r1, l0} forms a CCW quadrilateral by construction *)
		 CircleMemberQ[ s[[{r1, r0, r2, l0}]] ]),
            (* Delete edge {r0,r1}. *)
	    (* r1ptr is position of r1 in r0's val. *)
	    {merge[[r0]],merge[[r1]]} = delete[merge[[{r0,r1}]],
					Position[merge[[r1]],r0][[1,1]],r1ptr];
	    If[ l0ptr > r1ptr, l0ptr--];	(* update l0ptr *)
	    If[ r2ptr > r1ptr, r2ptr--];	(* update r2ptr *)
	    lengr0--;				(* update lengr0 *)
	    r1ptr = r2ptr; r2ptr = pred[r1ptr,lengr0]; 
	    r1 = merge[[r0,r1ptr]]; r2 = merge[[r0,r2ptr]]
          ],
	  l0r0r1LeftTurn = False
       ];

       (********************* Derive points l1 & l2. **********************)
       l1ptr = succ[r0ptr,lengl0]; 	(* l1ptr will point to l1 in l0's val *)
       l1 = merge[[l0,l1ptr]];
       If[ SignOfArea[s[[r0]],s[[l0]],s[[l1]]] == -1,  (* right turn *)
	  l2ptr = succ[l1ptr,lengl0]; (* l2ptr will point to l2 in l0's val *)
	  l2 = merge[[l0,l2ptr]];
	  (* Delete {l0,l1} only if edge {l1,l2} still exists. *)
	  While[ MemberQ[merge[[l1]],l2] &&
	       ((* {l1, l2, l0, r0} forms a CCW quadrilateral by construction *)
		 CircleMemberQ[ s[[{l0, l1, l2, r0}]] ]),
	    (* Delete edge {l0,l1}. *)
	    {merge[[l0]],merge[[l1]]} = delete[merge[[{l0,l1}]],
					Position[merge[[l1]],l0][[1,1]],l1ptr];
	    If[ r0ptr > l1ptr, r0ptr--];	(* update r0ptr *)
	    If[ l2ptr > l1ptr, l2ptr--];	(* update l2ptr *)
	    lengl0--;				(* update lengl0 *)
	    l1ptr = l2ptr; l2ptr = succ[l1ptr,lengl0];
	    l1 = merge[[l0,l1ptr]]; l2 = merge[[l0,l2ptr]]
          ],
	  r0l0l1RightTurn = False
       ];



       (*********** Derive new l0 or new r0, & update l0ptr & r0ptr. **********)
       If[ !l0r0r1LeftTurn,
	  (* Connect r0 to l1. *)
	  l1val = merge[[l1]]; lengr0++; lengl0 = Length[l1val]+1;
	  {l0,r0ptr} = {merge[[l0,l1ptr]],
		        succ[Position[l1val,l0][[1,1]],lengl0]},
	  If[ !r0l0l1RightTurn,
	     (* Connect l0 to r1. *)
	     r1val = merge[[r1]]; lengl0++; lengr0 = Length[r1val]+1;
	     {r0,l0ptr,r0ptr} = {merge[[r0,r1ptr]],
				 Position[r1val,r0][[1,1]],
				 succ[r0ptr,lengl0]}, 
	     (* Neither {l0,r0,r1} nor {r0,l0,l1} are collinear. *)
	     If[ CircleMemberQ[ s[[{r1, l0, r0, l1}]] ],
		(* Connect r0 to l1. *) 
		l1val = merge[[l1]]; lengr0++; lengl0 = Length[l1val]+1;
		{l0,r0ptr} = {merge[[l0,l1ptr]],
			      succ[Position[l1val,l0][[1,1]],lengl0]},
		(* Connect l0 to r1. *)
		r1val = merge[[r1]]; lengl0++; lengr0 = Length[r1val]+1;
		{r0,l0ptr,r0ptr} = {merge[[r0,r1ptr]],
				    Position[r1val,r0][[1,1]],
				    succ[r0ptr,lengl0]}
             ]
          ]
       ];



       (***************** Update edge & insert into merge. ******************)
       edge = {l0,r0};
       (* Insert edge {l0,r0} by inserting r0 in l0's val at position r0ptr
	  and inserting l0 in r0's val at position l0ptr. *)
       {merge[[l0]],merge[[r0]]} =
	  insert[ merge[[{l0,r0}]],l0,r0,l0ptr,r0ptr ]
     ];
     merge
   ]


        
(*================= Auxiliary Triangulation Functions ======================*)

succ[i_Integer,mod_Integer] := Mod[i,mod]+1

succ[{{i_Integer}},mod_Integer] := Mod[i,mod]+1

pred[i_Integer,mod_Integer] := Mod[i-2,mod]+1

pred[{{i_Integer}},mod_Integer] := Mod[i-2,mod]+1


   
(**** Insert v2 in position v1's adjacency list at position v2ptr. 
      Insert v1 in v2's adjacency list at position v1ptr. *)
(**** Returns object of the form {val1,val2}. *)
insert[{val1:{_Integer..},val2:{_Integer..}},
	v1_Integer, v2_Integer, v1ptr_Integer, v2ptr_Integer] :=
    {Insert[val1, v2, v2ptr ],
     Insert[val2, v1, v1ptr ]
    } 


(**** Delete v2 (pointed to by v2ptr) from v1's adjacency list val1.
      Delete v1 (pointed to by v1ptr) from v2's adjacency list val2. *)   
delete[{val1:{_Integer..},val2:{_Integer..}},
	v1ptr_Integer, v2ptr_Integer] :=
   {Delete[val1,v2ptr],
    Delete[val2,v1ptr]
   } 
   
(**** The default is to not include membership in circle boundary. *)

(* Assume that {a,b,c} form a CCW circle.  This returns True if any
	point in q is a member of circle {a, b, c}. *)
CircleMemberQ[{a:{_,_},b:{_,_},c:{_,_}}, q:{{_,_}..}, boundary_:False] :=
   Module[{scan},
     scan = Scan[If[(* Input points to CMQ must be placed in correct order. *)
		    If[SignOfArea[#, a, c] > 0,
		       CircleMemberQ[{c, a, b, #}, boundary],
		       If[SignOfArea[#, b, a] > 0,
			  CircleMemberQ[{a, b, c, #}, boundary],
			  If[SignOfArea[#, c, b] > 0,
			     CircleMemberQ[{b, c, a, #}, boundary],
			     True (* Point falls inside or on boundary of
					triangle {a, b, c}. *)
     			  ]
		       ]
		    ],
		    Return[True]]&, q];
     scan === True
   ]

CircleMemberQ[{a:{_,_},b:{_,_},c:{_,_},d:{_,_}}, boundary_:False] :=
  Module[{det, det1, xa, ya, xb, yb, xc, yc, xd, yd},
	(* Assume that c is to the left of the line from a to b and d (the
		point whose circle membership in {a,b,c} is of interest)
		is to the right. *)
	(* The following expression is
		Det[Map[{1, #[[1]], #[[2]], #[[1]]^2 + #[[2]]^2}&,
			{{xa, ya}, {xb, yb}, {xc, yc}, {xd, yd}}]].
	   The expanded expression is used instead of Det for numerical
	   precision. *)
	det =
	   (-(xb^2*xc*ya) + xb*xc^2*ya + xb^2*xd*ya - xc^2*xd*ya - xb*xd^2*ya +
	   xc*xd^2*ya + xa^2*xc*yb - xa*xc^2*yb - xa^2*xd*yb + xc^2*xd*yb +
	   xa*xd^2*yb - xc*xd^2*yb + xc*ya^2*yb - xd*ya^2*yb - xc*ya*yb^2 +
	   xd*ya*yb^2 - xa^2*xb*yc + xa*xb^2*yc + xa^2*xd*yc - xb^2*xd*yc -
	   xa*xd^2*yc + xb*xd^2*yc - xb*ya^2*yc + xd*ya^2*yc + xa*yb^2*yc -
	   xd*yb^2*yc + xb*ya*yc^2 - xd*ya*yc^2 - xa*yb*yc^2 + xd*yb*yc^2 +
	   xa^2*xb*yd - xa*xb^2*yd - xa^2*xc*yd + xb^2*xc*yd + xa*xc^2*yd -
	   xb*xc^2*yd + xb*ya^2*yd - xc*ya^2*yd - xa*yb^2*yd + xc*yb^2*yd +
	   xa*yc^2*yd - xb*yc^2*yd - xb*ya*yd^2 + xc*ya*yd^2 + xa*yb*yd^2 -
	   xc*yb*yd^2 - xa*yc*yd^2 + xb*yc*yd^2) /.
	   Thread[Rule[{xa, ya, xb, yb, xc, yc, xd, yd}, Flatten[{a,b,c,d}]]];
	det1 = If[det == 0 || Precision[det] == 0, 0,
		Chop[det, 10^(-Accuracy[det])]];
	If[boundary, det1 <= 0, det1 < 0]
  ]


(*======================= DelaunayTriangulationQ ===========================*)

(*** It is assumed that the val is such that a vertex on the hull lists first
     the neighbor that follows it on a counterclockwise traversal of the convex
     hull. *)

DelaunayTriangulationQ::inval = "Triangle `` is not a valid Delaunay triangle."
DelaunayTriangulationQ::hull =
"Computation of convex hull triangulation vertex adjacency list failed."
DelaunayTriangulationQ::col = "Triangle `` is collinear."

(*** Only point set and Delaunay vertex adjacency list available. ***)
DelaunayTriangulationQ[set:{{_,_}..},val:{{_Integer,{_Integer..}}..}] :=
  Module[{orig, maxx, maxxlist, miny, start, hull},
   (
     hull = TriangulationToHull[orig, val];
     If[ SameQ[hull,$Failed],
		Message[DelaunayTriangulationQ::hull];
		False,
     		iDelaunayTriangulationQ[set,val,hull]
     ]
    ) /; ( (orig = numericalize[set]) =!= $Failed &&
	   !CollinearQ[Map[#[[2]]&, Distinct[orig]]] )
   ] /; (Max[Flatten[val]] <= Length[set]) &&  (* val refers to valid pts *) 
	Length[val] >= 3             (* val must have at least 1 triangle *)


(*** Point set, vertex adjacency list (val), AND convex hull available. ***)
DelaunayTriangulationQ[set:{{_,_}..},val:{{_Integer,{_Integer..}}..},
	               hull:{_Integer..}]:=
  Module[{orig},
   (
    iDelaunayTriangulationQ[set,val,hull]
   ) /; ( (orig = numericalize[set]) =!= $Failed &&
	  !CollinearQ[Map[#[[2]]&, Distinct[orig]]] )
  ] /; (Max[Flatten[val]] <= Length[set]) &&  (* val refers to valid pts *)
	(Max[hull] <= Length[set])        (* hull refers to valid pts *)
	Length[val] >= 3             (* val must have at least 1 triangle *)



iDelaunayTriangulationQ[set_, val_, hull_] :=
  Module[{orig = numericalize[set], circle,
          vertices = Map[First, val], delaunayq},
     (* Compute list of triples defining Delaunay triangles. *)
     circle = Map[
		 (Module[{vert1=#[[1]],vertlist=#[[2]],
			      vertNum=Length[#[[2]]],circle, hullpoly},
       (* Create list of triples representing triangles adjacent to vert1. *)
             circle = Map[{vert1, vertlist[[#]],
			           vertlist[[Mod[#,vertNum]+1]]}&,
		                   Range[vertNum]];
             If[MemberQ[hull,vert1],
                hullpoly = hull/.{{vert1, b_,___,a_} :> {vert1,a,b},
                                 {___,a_, vert1, b_, ___} :> {vert1, a, b},
                                 {b_, ___, a_, vert1} :> {vert1, a, b}} ;
			    circle = DeleteCases[circle, hullpoly]
             ];
		     Map[Switch[Position[#, Min[#]],
				{{1}}, #,
				{{2}}, RotateLeft[#],
				{{3}}, RotateRight[#]]&, circle]
                 ])&,
		 val];
     (* Eliminate duplicates. *)
     circle = Union[Flatten[circle,1]];
     delaunayq = If[Scan[(If[SignOfArea @@ orig[[#]] == 0,
			     Message[DelaunayTriangulationQ::col, #];
			     Return[$Failed]])&,
			 circle] === $Failed,
		    (* One or more of the triangles are collinear! *)
		    False,	
		    (* Check that the circumcircles do not include
		       other vertices *)
                    Scan[(Module[{tri = #, compare, x},
		  	      compare = Complement[
					 Apply[Union,
					  Map[Module[{x = #},
					       Select[val,
					         MatchQ[#,{x,_List}]&,
					         1][[1,2]]
					      ]&,
					      tri]],
					 tri];	
       			      If[ CircleMemberQ[orig[[tri]],orig[[compare]]], 
				   Message[DelaunayTriangulationQ::inval,tri];
				   Return[False]]
          	         ])&,
	  	         circle]
                 ];
     delaunayq === Null
  ]

(*========================= TriangulationToHull ============================*)

(* if start point hasn't been specified yet... *)
TriangulationToHull[pts:{{_?NumberQ, _?NumberQ}...},
                    val:{{_Integer,{_Integer..}}..}] :=
   Module[{maxx, maxxlist, miny, start},
       maxx = Max[First[Transpose[pts]]];
       miny = Min[Last[Transpose[ Cases[pts, {maxx, _}] ]]];
       start = First[First[Position[pts, {maxx, miny}, {1}, 1]]];
       lpts = Transpose[{Range[Length[#]], #}]&[
                 pts[[ Cases[val, {start, _}][[1,2]] ]]
              ];
       nextpt = PolarSort[lpts, {maxx, miny}][[1,2]];
       next = First[First[Position[pts, nextpt, {1}, 1]]];
       TriangulationToHull[val, {start, next}]
   ]

(* Given two of the points on the hull, construct the rest. (An older
   version of this function required only one point, but had the constraint
   that the next point on the hull must be in a certain position.) *)
TriangulationToHull[val:{{_Integer,{_Integer..}}..}, {start_, next_}]:=
  Module[{label1, relabel1, start1, val1, hull, next1, vlen = Length[val],
          nextconnects, nextpos, ostart},
    (* Relabel vertex labels to make tracing hull easier.  This wouldn't
       be necessary if all the points in the original set were unique. *)
     label1 = Map[First,val];
     relabel1 = MapIndexed[(#1 -> First[#2])&, label1];
     start1 = start /. relabel1;
     next1 = next /. relabel1;
     val1 = Map[Last, val]/.relabel1;
   (* set the initial hull, and save the first hull point for reference *)
     hull = {start1};
     ostart = start1;
   (* Trace points on hull using relabeled adjacency list. *)  
     While[next1 != ostart,
       (* Guard against infinite loops. *)
       If[ Length[hull] > vlen, Return[$Failed]];
       AppendTo[hull, next1];
       nextconnects = val1[[next1]];
     (* make sure the element exists... not a problem if the triangulation
        is valid *)
       If[(nextpos = Position[nextconnects, start1]) === {}, Return[$Failed]];
       nextpos = If[# > Length[nextconnects], 1, #]&[First[First[nextpos]] + 1];
       start1 = next1; next1 = nextconnects[[nextpos]]
     ];
    (* Label convexhull with original labels. *)
     label1[[hull]]
  ]

(*=========================== VoronoiDiagram ================================*)

(*** special cases: 1 point, or set of collinear points, in set ***)
(* if points are collinear, then it really doesn't help to know the
   Delaunay adjacency or convex hull, so we ignore them *)
VoronoiDiagram[pt:{{_?NumericQ,_?NumericQ}}, ___] :=
   {pt, {{1,{1}}}}

VoronoiDiagram[set:{{_?NumericQ,_?NumericQ}..}?CollinearQ, ___] :=
  (* union eliminates coincident points *)
    Module[{minset = Sort[Union[set, SameTest -> ((#1 - #2 == {0, 0}) &)]], posns,
            pairs},
      (* if only one unique point *)
        If[Length[minset] === 1, Return[{minset, {{1,{1}}}}]];
      (* positions of unique points : use position of first encountered
         coincident point -- note that Union also sorts, which puts
         points in left-to-right/bottom-to-top ordering *)
        posns = Map[
            Function[tmp,
                First[First[Position[set, _?((tmp - # == {0,0})&),{1}]]]
            ],
            minset
        ];
      (* construct rays from adjacent pairs; resulting adjacency set is
          constructed from positions in which corresponding rays get placed *)
        {Flatten[Map[constructraypair[set[[#]]]&, Partition[posns, 2, 1]],1],
         Join[{{First[posns], {1,2}}},
            MapIndexed[{#1, 2*First[#2] + {-1,0,1,2}}&, Take[posns,{2,-2}]],
            {{Last[posns], 2 (Length[minset]-1) + {-1,0}}}]
        }
    ]

(* I'm not quite sure of this technique for constructing the magnitudes;
   it will look odd if points close together are surrounded by points
   far apart. On the other hand, the inverse situation looks better... *)
constructraypair[{pt1_, pt2_}] :=
    Block[{mid = (pt1 + pt2)/2, dir = Reverse[pt2 - pt1]},
        {Ray[mid, mid + 1.5 {1,-1} dir],
         Ray[mid, mid + 1.5 {-1,1} dir]}
    ]

(*** Only point set available. ***)
VoronoiDiagram[set:{{_,_}..}]:=
   Module[{delaunay=DelaunayTriangulation[set,Hull->True]},
     If[ SameQ[delaunay,$Failed],
	$Failed,
        VoronoiDiagram[set,delaunay[[1]],delaunay[[2]]]
     ]		/; FreeQ[delaunay,DelaunayTriangulation]
   ] /; numericalQ[N[set]]  (* numerical points *)

(*** Both point set and Delaunay vertex adjacency list available. ***)
VoronoiDiagram[set:{{_,_}..},delval:{{_Integer,{_Integer..}}..}]:=
   Module[{hull = TriangulationToHull[numericalize[set], delval]},
     If[ SameQ[hull,$Failed] || !FreeQ[hull,TriangulationToHull] ,
		$Failed,
     		VoronoiDiagram[set,delval,hull]
     ]
   ] /; numericalQ[N[set]] &&  (* numerical points *)
	(Max[Flatten[delval]] <= Length[set]) &&  (* val refers to valid pts *)
	Length[delval] >= 3		(* val must have at least 1 triangle *)


(*** Point set, Delaunay vertex adjacency list, and convexhull available. ***)
VoronoiDiagram[set:{{_,_}..},delval:{{_Integer,{_Integer..}}..},
	       hull:{_Integer..}]:=
   Module[{orig, lhull = Length[hull], ltrue, relabel,
	   vorval, vorvertices, truevertices, quasivertices,
	   truecoordinates, quasicoordinates},
    (
    (* Compute Voronoi vertex adjacency list. *)
     vorval = Map[
		 (Module[{delvert1=#[[1]],delvertlist=#[[2]],
			  delvertNum=Length[#[[2]]],vorpoly, pos, hullpoly},
       (*  Each vertex of a Voronoi polygon corresponds to a Delaunay vertex
	   triple.  Exceptions: a "quasi" Voronoi vertex corresponds to a
	   Delaunay vertex pair; a "true"  Voronoi vertex can correspond to
	   two Delaunay vertex triples. *)
       (* Create list of triples representing triangles adjacent to delvert1.
	  These determine Voronoi vertices. *)
                vorpoly = Apply[{delvert1, ##}&,
                    Partition[delvertlist, 2,1,{1,1}],
                    {1}];
       (* If delvert1 is on hull, add information needed for determining
	  infinite rays of associated Voronoi polygon. *)
                If[MemberQ[hull, delvert1],
                  (* select the poly formed from the adjacent hull points *)
                    hullpoly = hull/.{{delvert1, b_,___,a_} :> {delvert1,a,b},
                                 {___,a_, delvert1, b_, ___} :> {delvert1, a, b},
                                 {b_, ___, a_, delvert1} :> {delvert1, a, b}} ;
                    pos = First[First[
                        Position[vorpoly, hullpoly, {1}, 1, Heads -> False]
                     ]]; (* i.e., the poly with all points on the hull *)
                     vorpoly[[pos]] = Map[Sort, {{delvert1, hullpoly[[2]]},
                          vorpoly[[(pos - 1)/. 0 -> delvertNum]]}];
                     vorpoly = Insert[vorpoly,
                         Map[Sort, {{delvert1, hullpoly[[3]]}, vorpoly[[(pos + 1)/. delvertNum + 1 -> 1]]}],
                         pos + 1]
                  ];
                  Map[Sort, vorpoly]
                 ])&,
		 delval];

    (* We ~can~ use Union here because we are not dealing with numeric
	quantities  The Union leaves the lists representing quasi Voronoi
	vertices at the beginning of the vorvertices list. *)
    vorvertices = Union[Flatten[vorval,1]];
    quasivertices = Take[vorvertices,lhull];
    truevertices = Drop[vorvertices,lhull];
    coordinates = Map[Apply[circlecenter,orig[[#]]]&,truevertices];

    (* True Voronoi vertices.  Can't use Union[coordinates] because Union 
	distinguishes between SetAccuracy[.5, 15] and SetAccuracy[.5, 16]
	and we don't want to.  To preserve results prior to V3.0,
        use temp = Union[coordinates], rather than simple
        temp = coordinates. *)
    Module[{temp = Union[coordinates], first},
      truecoordinates = {};
      While[Length[temp] > 0,
	first = First[temp]; rest = Rest[temp];
	AppendTo[truecoordinates, first];
        temp = Delete[rest,
		Position[rest, zz_ /; TrueQ[zz-first=={0, 0}], 1] ]
      ] ];
    ltrue = Length[truecoordinates];
    (* lables for true Voronoi vertices *)

    relabel = Flatten[
	MapIndexed[Module[{pos = Flatten[
			   	 Position[coordinates,
					 zz_ /; TrueQ[zz-#1=={0, 0}], 1], 1],
			   label = #2[[1]]},
		     Map[(truevertices[[#]] -> label)&,pos]
                   ]&,
		   truecoordinates], 1];
    (* lables for quasi Voronoi vertices *)
    relabel = Join[relabel,
		   MapIndexed[(#1 -> #2[[1]]+ltrue)&,
			      quasivertices
                   ]
              ];
    (* Label Voronoi vertices in Voronoi val uniquely, and eliminate
	duplicates. *)
    vorval = Map[Module[{val = # /. relabel, unique, current},
		   current = val[[1]];
		   unique = {current};
	           Scan[(If[ # != current,
			     current = #;
			     AppendTo[unique,#]
                         ])&,
			Drop[val,1]];
                   unique
                 ]&,
		 vorval];
    (* Add a field indicating Delaunay vertex label to each Voronoi val. *)
    vorval = MapIndexed[{delval[[#2[[1]],1]],#1}&,vorval];

    (* Quasi Voronoi vertices. *)
    (* Re-establish the ccw direction of the hull edges that define quasi
	Voronoi vertices.  (Direction was lost when unique vertices were
	found using "Union".)  Then determine ray "tail" from ray "head"
	and the associated Delaunay hull edge. *)
    quasicoordinates = 
         Map[Module[{edge = #[[1]],
                     truevertex = #[[2]] /. relabel,
                     head, ptr1},
                 ptr1 = Position[hull,edge[[1]]][[1,1]];
                 If[edge[[2]] != hull[[ succ[ptr1,lhull] ]],
                    edge = edge[[{2,1}]] ];
                 head = truecoordinates[[ truevertex ]];
                 Ray[head, raytail[ head, orig[[edge]] ]]
             ]&,
             quasivertices
         ];

    {Join[truecoordinates,quasicoordinates],vorval}   
   ) /;  (orig = numericalize[set]) =!= $Failed 
   ] /; (Max[Flatten[delval]] <= Length[set]) &&  (* val refers to valid pts *)
	(Max[hull] <= Length[set]) &&	  (* hull refers to valid pts *)
	Length[delval] >= 3		(* val includes at least 1 triangle *)


(*** Compute the center of the circumcircle of triangle {p1,p2,p3}. *)
(* old code
circlecenter[p1:{_,py1_}, p2:{_,py2_}, p3:{_,py3_}] :=
   Module[{u1, u2, u3, m2, m3, bx2, by2, bx3, by3},
     (* Insure that there is no "/0" problem. *)
     {u1,u2,u3} = If[SameQ[py1,py2],
			{p3,p2,p1},
			If[SameQ[py1,py3],
				{p2,p1,p3},
				{p1,p2,p3}]];
     m2 = -Apply[Divide,u1-u2]; {bx2,by2} = (u1+u2)/2;
     m3 = -Apply[Divide,u1-u3]; {bx3,by3} = (u1+u3)/2;
     { (by3-by2) + m2 bx2 - m3 bx3,
       m2 m3 (bx2-bx3) + m2 by3 - m3 by2} / (m2 - m3)
   ] /; ( (* Changed the test !(py1 == py2 == py3) to
	   !(py1-py2 == py1-py3 == py2-py3 == {0,0}) for numerical precision *)
	  !(py1-py2 == py1-py3 == py2-py3 == {0,0}) )
*)

circlecenter[p1_, p2_, p3_] :=
   Block[{d1, d2, d3, c1, c2, c3, ca}, (* Block for speed *)
      d1 = (p3 - p1) . (p2 - p1);
      d2 = (p3 - p2) . (p1 - p2);
      d3 = (p1 - p3) . (p2 - p3);
      c1 = d2 d3; c2 = d3 d1; c3 = d1 d2;
      ca = c1 + c2 + c3;
      ((c2 + c3) p1 + (c3 + c1) p2 + (c1 + c2) p3)/(2 ca)
   ]


(*** Compute the tail of the ray having head "head" and bisecting segment
     {e1,e2}.  The tail is defined by a point a distance dist(e1,e2) from
     the ray head in a direction such that {e1,head,tail} is a right turn
     OR a distance dist(e1,e2) from the edge midpoint {xm,ym} in a direction
     such that {e1,{xm,ym},tail} is a right turn.  The distance is computed
     from the head or from {xm,ym} depending on which will look better in
     DiagramPlot. *)
raytail[head:{xh_,yh_},{e1:{_,_},e2:{_,_}}] :=
   Module[{m, out, xm, ym, distfromhead, soa, 
	   d0 = Sqrt[ Apply[Plus,(e1-e2)^2] ]},
      {xm,ym} = (e1+e2)/2; (* edge midpoint *)
      (* Determine whether d0 will be measured from head or {xm,ym}; 
	 this is an aesthetic consideration. *)
      If[ SignOfArea[e1,{xm,ym},head] == -1,
		distfromhead = True,
		distfromhead = False;
		d0 = d0 + Sqrt[ Apply[Plus,(head - {xm,ym})^2] ]
      ];
      out = If[ Chop[#, 10^(1-Internal`EffectivePrecision[#])]&[e1[[2]] - e2[[2]]] === 0,
	        (* ray is vertical *)
	        {{xh,yh-d0},{xh,yh+d0}},
	        (* ray has slope m *)
		m = -Apply[(#1/#2) &,e1-e2];
		Map[{#,m (#-xm)+ym}&,
	            Module[{b = (m (ym-m xm-yh)-xh)/(1+m^2),sqrt},
		      sqrt = Sqrt[b^2 - (xh^2 + (ym-m xm-yh)^2 - d0^2)/(1+m^2)];
		      -b + {sqrt,-sqrt}
                    ]
		] 
      ];
      (* Choose the correct direction of the ray.
	 {e1,head,tail} or {e1,{xm,ym},tail} should form a right turn. *)
      soa = If[ distfromhead,
		SignOfArea[e1,head,out[[1]]],
		SignOfArea[e1,{xm,ym},out[[1]]]
            ];
      If[ soa == -1,
	 	out[[1]], (* right turn *)
	 	out[[2]]] (* left or no turn *)
   ]




VoronoiDiagram::notimp =
  "VoronoiDiagram is not yet implemented for dimension ``."

VoronoiDiagram[set_List]:=
   Module[{out=Module[{d=Length[set[[1]]]},
		  Message[VoronoiDiagram::notimp,d];
		  $Failed]},
	out /; !SameQ[out,$Failed]
   ] /; (Apply[And,Map[NumberQ,N[set]]] ||
	(Apply[And,Map[NumberQ,N[Flatten[set]]]] &&
	 Apply[Equal,Map[Length[#]&,set]] &&
	 Apply[And,Map[SameQ[Head[#],List]&,set]])) &&
        (Length[First[set]] =!= 2)


(*=========================== NearestNeighbor ==============================*)

iNearestNeighbor[input_, vorpts_, vorval_]:=
  Module[{ninput = numericalize[input], closed = {}, open = {}},
    (* If VoronoiDiagram produced vorval, the open polygons are guarranteed to
       be listed after the closed polygons.  The following general approach
       to identifying open and closed polygons is useful if some other method
       generated the diagram of convex polygons represented by vorval. *)
    Scan[(If[FreeQ[vorpts[[#[[2]]]],Ray],
	     AppendTo[closed,#],
	     AppendTo[open,#]
	  ])&,
	  vorval];
    (* If VoronoiDiagram produced vorval, each open polygon is guarranteed to
       list the two Ray "vertices" last.  The following scheme for putting
       open polygons in this format is useful if some other method generated
       the diagram of convex polygons represented by vorval.  A valid open
       polygon will list the two rays adjacently. *)
    open = Apply[(vl = vorpts[[#2]];
                If[MatchQ[vl, {_Ray, ___, _Ray}],
                    {#1, RotateLeft[#2]},
                    {#1, RotateLeft[#2,
                            First[Flatten[Position[vl, Ray[___]]]] + 1]}
                ])&,
	       open, {1}];
    (* First compute an internal point for each closed convex polygon. *)
    closed = Map[Append[#,
			Apply[Plus, vorpts[[ Take[#[[2]],3] ]] ]/3
                 ]&,
                 closed];
    (* For each input point, determine the label of the nearest neighbor. *)
    Map[Module[{query = #, class},
	  (* Scan closed polygons for location of query. *)
	  class = Scan[Module[{internal = #[[3]]},
			 If[ClosedPolygonMemberQ[query-internal,
					   Map[(#-internal)&,
					   vorpts[[ #[[2]] ]] ]],
			    Return[#[[1]]]
                         ]
                       ]&,
                       closed];
          (* If query not located yet, scan the open polygons. *)
          If[SameQ[class,Null],
	    class = Scan[If[OpenPolygonMemberQ[query, vorpts[[ #[[2]] ]] ],
				Return[#[[1]]]
                         ]&,
			 open]
          ];
	  If[class === Null, $Failed, class]
	]&, 
	ninput]
  ]

NearestNeighbor[input:{{_,_}..},vorpts_List,
		vorval:{{_Integer,{_Integer..}}..}]:=
  Module[{nn = iNearestNeighbor[input,vorpts,vorval]},
	If[ nn === $Failed,
		$Failed,
		nn
	]
  ] /; (numericalQ[N[input]] && ValidDiagramQ[vorpts,vorval])

NearestNeighbor[{a_,b_},vorpts_List,vorval:{{_Integer,{_Integer..}}..}]:=
  Module[{nn = iNearestNeighbor[{{a,b}},vorpts,vorval]},
	If[ nn === $Failed,
		$Failed,
		nn[[1]]]
  ] /; (numericalQ[N[{a, b}]] && ValidDiagramQ[vorpts,vorval])

NearestNeighbor::obslt =
"NearestNeighbor is obsolete. Use Nearest instead. (See the \
usage for the new syntax.)";
$$obsltmsgflag = False;

NearestNeighbor[input:{{_,_}..},pts:{{_,_}..}]:=
    Module[{nf},
        If[!$$obsltmsgflag,
            Message[NearestNeighbor::obslt];
            $$obsltmsgflag = True;
        ];
        nf = Nearest[pts -> Automatic];
        Flatten[nf /@ input]/;Head[nf] === NearestFunction
    ]

NearestNeighbor[{a_,b_},pts:{{_,_}..}]:=
    Module[{n},
        If[!$$obsltmsgflag,
            Message[NearestNeighbor::obslt];
            $$obsltmsgflag = True;
        ];
        n = Nearest[pts -> Automatic, {a,b}];
        First[n]/; ListQ[n] && Length[n] >= 1
    ]

(*========================== ClosedPolygonMemberQ ============================*)

(* Binary search of wedges that comprise a closed convex polygon containing
   the origin.  Returns True if the polygon contains the query. Polygon must
   contain origin because this makes use of PolarAngle. *)
ClosedPolygonMemberQ[query:{_,_},polygon:{{_,_}..}] :=
    Module[{median,a1,am,aq,p = Append[polygon,First[polygon]]},
      While[ Length[p]>2,
        median = Floor[Length[p]/2]+1;
	{a1,am,aq} = Map[PolarAngle,{p[[1]],p[[median]],query}];
	If[ (a1 <= aq < am) || ((a1 > am) && !(aq < a1 && aq >= am)),
	   (* query lies in wedge between p[[1]] & p[[median]] *)
	   p = Take[p,median],
	   (* query lies in wedge between p[[median]] & p[[1]] *)
	   p = Drop[p,median-1]
        ]
      ]; (* here p is of the form {{x1,y1},{x2,y2}} *)
      (* Include points on the boundary... this means that lower-labled
	 polygons will be favored over higher-labled polygons when it
	 comes to points on the boundary. *)
      (* test for left turn or no turn *)
      SignOfArea[query,p[[1]],p[[2]]] >= 0 
    ]


(*=========================== OpenPolygonMemberQ =============================*)

(* Linear search of two rays and segments comprising an open convex polygon. *)
OpenPolygonMemberQ[query:{_,_},polygon_List] :=
   Module[{rays = Take[polygon, -2] /. Ray ->List,
	   sides = Drop[polygon, -2]},
      (* Include points on the boundary... this means that lower-labled
	 polygons will be favored over higher-labled polygons when it
	 comes to points on the boundary. *)
     rayq = SignOfArea[query,
	      Sequence @@ rays[[1]]] >= 0 (* left or no turn *) &&
     	    SignOfArea[query,
	      Sequence @@ Reverse[rays[[2]]]] >= 0 (* left or no turn *);	
     If[sides === {}, 
	rayq,
	scan = Scan[If[SignOfArea[query,
				  Sequence @@ #] < 0 (* right turn *),
			Return[False]]&,
		    Transpose[{Drop[#, -1], Drop[#, 1]}]&[sides]];
        rayq && (scan === Null)
     ]
   ]

(*============================= ValidDiagramQ ===============================*)

ValidDiagramQ[vert_List,val:{{_Integer,{_Integer..}}..}] :=
  (
	(* vertices must have head List or Ray *)
	Apply[And,Map[(SameQ[Head[#],List] || SameQ[Head[#],Ray])&,vert]] &&
	(* vertices must have length 2 *)
	Apply[And,Map[(Length[#]==2)&,vert]] &&
	(* each polygon must have 0 rays (closed) or 2 rays (open) or 4 rays
            (open collinear) *)
     Apply[And,Map[Module[{raynum=Length[Position[ vert[[ #[[2]] ]], Ray]]},
		     raynum == 0 || raynum == 2 || raynum == 4
		   ]&,
		   val]] &&
        (* adjacency list should not refer to vertices not in vertex list *)
	(Length[vert] == Length[Union[Flatten[Map[#[[2]]&,val]]]])
  )


(*=================== Auxiliary Plotting Functions ===========================*)

(* In computational geometry, it's important to plot things without any
	scaling or stretching. *)
absolutePlotRange[set:{{_,_}..}] :=
  Module[{xcoord, ycoord, minx, maxx, miny, maxy,
	  xhalfrange, yhalfrange, xmid, ymid},
      {xcoord,ycoord} = Transpose[set]; 
      {minx,maxx} = {Min[xcoord],Max[xcoord]};
      {miny,maxy} = {Min[ycoord],Max[ycoord]};
      {xhalfrange,yhalfrange} = {maxx-minx,maxy-miny}/2;
      If[ xhalfrange > yhalfrange,
	 ymid = miny+yhalfrange;
	 {{minx,maxx},{ymid-xhalfrange,ymid+xhalfrange}},
	 xmid = minx+xhalfrange;
	 {{xmid-yhalfrange,xmid+yhalfrange},{miny,maxy}}
      ]
 ]

(*============================= DiagramPlot ================================*)

Options[DiagramPlot] =  Join[{LabelPoints->True, TrimPoints->0},
			     Options[Graphics]]

DiagramPlot::trim =
"Warning: TrimPoints -> `` is not a non-negative integer. \
TrimPoints -> 0 is used instead."
			


(* Polygon diagram is already computed and polygon center coordinates are
   available. *)
DiagramPlot[set:{{_,_}..}, vorvert_List, val:{{_Integer,{_Integer..}}..},
	opts___] :=
  Module[{labelpoints, trimpoints, orig, distinct},
   (
    {labelpoints,trimpoints} = {LabelPoints,TrimPoints} /. {opts} /.
					Options[DiagramPlot];
    If[ !(IntegerQ[trimpoints] && !Negative[trimpoints]),
		Message[DiagramPlot::trim,trimpoints];
		trimpoints = 0
    ];
    distinct = Map[{#[[1]],orig[[#[[1]]]]}&,val];
    diagramplot[distinct,vorvert,val,labelpoints,trimpoints,opts] 
   ) /;  ((orig = numericalize[set]) =!= $Failed) 
  ] /; ( ValidDiagramQ[vorvert,val] &&
	 (*  adjacency list should not refer to points not in set *)
         (Max[Map[#[[1]]&,val]] <= Length[set]) )


(* Compute Voronoi diagram before plotting. *)
DiagramPlot[set:{{_,_}..},opts___Rule] :=
  Module[{voronoi = VoronoiDiagram[set], distinct, vorvert, val},
    If[ SameQ[voronoi,$Failed], Return[$Failed]];
    {vorvert,val} = voronoi;
    DiagramPlot[set,vorvert,val,opts]
  ] /; numericalQ[N[set]] 


(* diagramplot *)
diagramplot[distinct:{{_,{_,_}}..},vorvert_List,
	val:{{_Integer,{_Integer..}}..},
	labelpoints_Symbol,trimpoints_Integer,opts___] :=
  Module[{centerlist,vertexlist={PointSize[.012]},
	  linelist={Thickness[.003]}, leng = Length[distinct],
	  trim = trimpoints, original, absolutepoints, centroid,
	  max2, dist2, pos, xmin, xmax, ymin, ymax, delx, dely},
    (* diagram polygon "centers" *)
    centerlist = If[labelpoints,
                    Map[Apply[Text,#]&,distinct],
     	            Prepend[Map[Point,Map[#[[2]]&,distinct]],
		      PointSize[.012]]
		 ];
    (* diagram vertices and infinite rays *)
    Scan[(If[FreeQ[#,Ray],
	    AppendTo[vertexlist,Point[#]],
	    AppendTo[linelist,Line[Apply[List,#]]]
	  ])&,
	 vorvert];
    (* closed diagram polygons *)
    Scan[(Module[{list = vorvert[[#[[2]]]]},
         If[FreeQ[list, Ray],
           (* if no rays, make a closed polygon *)
             AppendTo[linelist, Line[Append[list, First[list]]] ],
           (* if rays, reorder the list so the rays are on the ends, then
              delete the rays and make an open line *)
             list = DeleteCases[RotateLeft[list,
                        First[Flatten[Position[list, Ray[___]]]]],
                    Ray[___]];
             AppendTo[linelist, Line[list]]
         ]])&,
	 val];
    (* Compute points that will determine PlotRange. *)
    original = Map[#[[2]]&,distinct];
    If[ trim == 0,
	(* include all diagram vertices AND ray tails *)
	absolutepoints = Join[ original,
			       Map[(If[ SameQ[Head[#],Ray],
					#[[2]],
					#])&,
    				   vorvert]
                             ],
        (* exclude ray tails and (trim-1) of the furthest outlying diagram
	   vertices *)
        absolutepoints = Join[ original,
			       Select[vorvert, !SameQ[Head[#],Ray]&] ];
        If[ trim > 1,
        	centroid = Apply[Plus,original]/Length[original];
		max2 = Max[Map[Apply[Plus,(#-centroid)^2]&,
		               original
                          ]];
        	dist2 = Map[Apply[Plus,(#-centroid)^2]&,
			        Drop[absolutepoints,leng]
                        ];
        	If[dist2 =!= {}, pos = Position[dist2,Max[dist2]][[1,1]]];
                (* Delete diagram vertices from 'absolutepoints' as long as 
		   they lie further from the centroid then does the furthest
		   polygon "center". *)
                 While[ trim!=1 && dist2 =!= {} && dist2[[pos]] > max2,
			dist2 = Delete[dist2,pos];
			absolutepoints = Delete[absolutepoints, pos+leng];
			If[dist2 =!= {}, pos = Position[dist2,Max[dist2]][[1,1]]];
			trim--
                 ]
	]
    ];
    {{xmin,xmax},{ymin,ymax}} = absolutePlotRange[absolutepoints];
    {delx,dely} = .05/1.9 {xmax-xmin,ymax-ymin};
    optslist = Select[{opts},!(SameQ[#[[1]],LabelPoints] ||
			       SameQ[#[[1]],TrimPoints])&];
    Show[Graphics[
        {centerlist,
         vertexlist,
         linelist},
        PlotRange -> {{xmin-delx,xmax+delx},{ymin-dely,ymax+dely}},
	AspectRatio->1
    ],optslist]
  ]


(*=========================== PlanarGraphPlot ===============================*)

Options[PlanarGraphPlot] = Join[{LabelPoints->True},
				Options[Graphics]]


(* This is the form for plotting convex hull. *)
PlanarGraphPlot[set:{{_,_}..}, circuit:{_Integer..}, opts___]:=
   Module[{orig, pointlist, linelist, distinct,
	   xmax, xmin, ymax, ymin, delx, dely, optslist,
	   labelpoints = LabelPoints /. {opts} /. Options[PlanarGraphPlot]},
     (
      If[labelpoints,
	 distinct = Distinct[orig];
	 pointlist = Map[Text[#,orig[[#]]]&,Transpose[distinct][[1]]],
	 pointlist = Prepend[Map[Point,orig],PointSize[.012]] ];
      linelist = {Thickness[.003],
		  Line[Append[ orig[[circuit]],orig[[circuit[[1]]]] ]]};
      {{xmin,xmax},{ymin,ymax}} = absolutePlotRange[orig];
      {delx,dely} = .05/1.9 {xmax-xmin,ymax-ymin};
      optslist = Select[{opts},!SameQ[#[[1]],LabelPoints]&];
      Show[Graphics[
	  {pointlist,
	   linelist},
	  PlotRange -> {{xmin-delx,xmax+delx},{ymin-dely,ymax+dely}},
	  AspectRatio->1
      ],optslist]
     ) /; (orig = numericalize[set]) =!= $Failed
   ]	/; Max[circuit] <= Length[set] && Length[set] > 2


(* Vertex adjacency list is already computed. *)
PlanarGraphPlot[set:{{_,_}..},val:{{_Integer,{_Integer..}}..},opts___] :=
  Module[{orig, pointlist, linelist, distinct, pairs, 
	  xmax, xmin, ymax, ymin, delx, dely, optslist,
	  labelpoints = LabelPoints /. {opts} /. Options[PlanarGraphPlot]},
     (
      If[labelpoints,
	 distinct = Distinct[orig];
	 pointlist = Map[Text[#,orig[[#]]]&,Transpose[distinct][[1]]],
	 pointlist = Prepend[Map[Point,orig],PointSize[.012]] ];
      pairs = Map[(Outer[List,{#[[1]]},#[[2]]][[1]])&,
	          val];
      pairs = Union[Map[Sort,Flatten[pairs,1]]];
      linelist = Prepend[Map[Line[orig[[#]]]&,pairs],Thickness[.003]];
      {{xmin,xmax},{ymin,ymax}} = absolutePlotRange[orig];
      {delx,dely} = .05/1.9 {xmax-xmin,ymax-ymin};
      optslist = Select[{opts},!SameQ[#[[1]],LabelPoints]&];
      Show[Graphics[
	  {pointlist,
	   linelist},
	  PlotRange -> {{xmin-delx,xmax+delx},{ymin-dely,ymax+dely}},
	  AspectRatio->1
      ],optslist]
     ) /; (orig = numericalize[set]) =!= $Failed
  ]   /; Max[Flatten[val]] <= Length[set] && Length[set] > 2


(* Compute Delaunay triangulation before plotting. *)
PlanarGraphPlot[set:{{_,_}..},opts___] :=
  Module[{delaunay = DelaunayTriangulation[set]},
    If[ SameQ[delaunay,$Failed],
	$Failed,
        PlanarGraphPlot[set,delaunay,opts]
    ]
  ] /; numericalQ[N[set]] && Length[set] > 2





(*======================= TriangularSurfacePlot =============================*)

    (* If two or more 3d points have the same {x,y} coordinates, but different
       z coordinates, only the first instance of the {x,y} coordinates in the
       original set is considered.  Other values for z are ignored. *)

Options[TriangularSurfacePlot] = Options[Graphics3D]

(* Vertex adjacency list is already computed. *)
TriangularSurfacePlot[set:{{_,_,_}..},val:{{_Integer,{_Integer..}}..},
        opts___] := TriangularSurfacePlot[set, {val, None}, opts]

TriangularSurfacePlot[set:{{_,_,_}..},{val:{{_Integer,{_Integer..}}..}, hull_},
	opts___] :=
  Module[{orig, distinct3, distinct2, sorted, label, polygonlist},
   ( 
    (* determine triangles from adjacency list *)
    polygonlist = Map[Module[{vertex=#[[1]],adjvert=#[[2]],
			      vertNum=Length[#[[2]]]},
			Map[{vertex,adjvert[[#]],
			     adjvert[[ Mod[#,vertNum]+1 ]]}&,
			     Range[vertNum]]
                      ]&,
		      val
		  ];
    (* unique triangles determined by creating all
       triangles; only valid ones are those created by
       the adjacency lists for all 3 corner vertices,
       so select from the groups of identical triangles
       those groups of length at least 3 *)
    polygonlist = Map[First,
                      Select[Split[Sort[Map[Sort,Flatten[polygonlist,1]]]],
                                 Length[#] >= 3 &]
                  ];
    (* if there is more than one triangle generated, and the convex hull
       is in the list of triangles, then it is overlapping some other
       triangles and needs to be removed. *)
    If[Length[polygonlist] > 1,
         polygonlist = DeleteCases[polygonlist, Sort[Flatten[{hull}]]]
    ];
    (* Map Polygon onto coordinates of unique triangles  *)
    polygonlist = Map[Polygon[Module[{triangle = #},
			        Map[orig[[#]]&, triangle]
                              ]]&,
                      polygonlist
                  ];
    Show[Graphics3D[polygonlist],opts]
   ) /; (orig = numericalize[set]) =!= $Failed
  ] /; Max[Flatten[val]] <= Length[set]

(* Compute Delaunay triangulation before plotting. *)
TriangularSurfacePlot[set:{{_,_,_}..},opts___] :=
  Module[{delaunay = DelaunayTriangulation[Map[Drop[#,-1]&,set], Hull -> True]},
    If[ SameQ[delaunay,$Failed],
	$Failed,
        TriangularSurfacePlot[set,delaunay,opts]
    ]
  ] /; numericalQ[N[set]] && Length[set] > 2


(* ================================= TileAreas ============================== *)

(* triangular areas computed using "Heron's formula" *)
TileAreas[set:{{_,_}..},vorvert_List,val:{{_Integer,{_Integer..}}..}] :=
  Map[Module[{center, vertices, segments, a, b, c, s},
          center = set[[#[[1]]]];
          vertices = vorvert[[#[[2]]]];
          If[FreeQ[vertices, Ray],
             segments = Transpose[{vertices, RotateLeft[vertices]}];
             Apply[Plus,
                Map[(a = distanceD[center, #[[1]]];
                     b = distanceD[center, #[[2]]];
                     c = distanceD[#[[1]], #[[2]]];
                     s = (a + b + c)/2;
                     Sqrt[s (s-a)(s-b)(s-c)])&, segments]],
             Infinity
          ]
      ]&, (* end Module *)
      val]

distanceD[p_, q_] := Sqrt[Apply[Plus, (p-q)^2]]

(*===============================  delta  ===================================*)
(* Utility to routine to estimate amount of fuzz to use in comparison *)
(* this is drawn from a routine in the Statistics`MultinormalDistribution`
   package, though cut down a bit for simplicity *)
delta[num_] := 
  Max[10^(-(Max[Accuracy[num], MachinePrecision] - 1)), 
    Abs[num] 10^(-(Max[Precision[num], MachinePrecision] - 1))]

(*============================= intersection ================================*)

(* Find intersection of bounding segment {{x1, y1}, {x2, y2}} with
        voronoiTile, such that intersected edge is a new edge.
   Return intersection coordinates and edge that is intersected. *)
intersection[input:{{x1_, y1_}, {x2_, y2_}}, voronoiTile_, prevEdge_] :=
  Module[{x, y, boundingSegmentEqn, tileSegmentsRays, scan},
    boundingSegmentEqn = ( (y-y1)(x2-x1) == (x-x1)(y2-y1) );
    tileSegmentsRays = Transpose[{voronoiTile, RotateLeft[voronoiTile, 1]}];
    tileSegmentsRays = Map[If[Head[#[[1, 1]]] === Ray,
                              #[[1]],
                              If[Head[#[[2, 1]]] === Ray,
                                 {},
                                 #]]&, tileSegmentsRays];
    tileSegmentsRays = tileSegmentsRays /. {a___, {}, b___} -> {a, b};
    scan = Scan[Module[{u1, v1, u2, v2, l1, l2, l, x0, y0, xs, ys},
        If[Head[#[[1]]] === Ray,
           { {{u1, v1}, {u2, v2}}, l } = (# /. Ray -> List);
           If[l =!= prevEdge,
              solve = Solve[{boundingSegmentEqn,
                                (y-v1)(u2-u1) == (x-u1)(v2-v1)}, {x, y}];
              If[FreeQ[solve, Solve] && solve =!= {},
                 (* check that intersection lies within the endpoints of
                    the bounding segment and within the single endpoint
                    of the Voronoi tile ray *)
                 {x0, y0} = ({x, y} /. solve)[[1]];
                 xs = delta[x0]; ys = delta[y0];
                 If[ (Min[{x1, x2}] - xs) <= x0 <= (Max[{x1, x2}] + xs) &&
                     (Min[{y1, y2}] - ys) <= y0 <= (Max[{y1, y2}] + ys) &&
                     ((u1 < u2 && u1 - xs < x0) || (u1 > u2 && u1 + xs > x0) ||
                     (u1 == u2 == x0)) &&
                     ((v1 < v2 && v1 - ys < y0) || (v1 > v2 && v1 + ys > y0) ||
                     (v1 == v2 == y0)),
                     (* valid intersection, return the intersection
                        coordinates and the label of the
                        Voronoi tile ray being intersected *)
                     Return[{ {x0, y0}, l }]
                 ]
              ] (* end If FreeQ[solve, Solve] *)
           ], (* end If l =!= prevEdge *)
           { {{u1, v1}, l1}, {{u2, v2}, l2} } = #;
           If[{l1, l2} =!= prevEdge,
              solve = Solve[{boundingSegmentEqn,
                                (y-v1)(u2-u1) == (x-u1)(v2-v1)}, {x, y}];
              If[FreeQ[solve, Solve] && solve =!= {},
                 (* check that intersection lies within endpoints of
                          both segments *)
                 {x0, y0} = ({x, y} /. solve)[[1]];
                 xs = delta[x0]; ys = delta[y0];
                 If[ (Max[Min[{x1, x2}], Min[{u1, u2}]] - xs) <= x0 <=
                     (Min[Max[{x1, x2}], Max[{u1, u2}]] + xs) &&
                     (Max[Min[{y1, y2}], Min[{v1, v2}]] - ys) <= y0 <=
                     (Min[Max[{y1, y2}], Max[{v1, v2}]] + ys),
                     (* valid intersection, return the intersection
                        coordinates and the labels of the
                        endpoints of the Voronoi tile segment
                        being intersected *)
                     Return[{ {x0, y0}, {l1, l2} }]
                 ]
              ] (* end If FreeQ[solve, Solve] *)
           ] (* end If {l1, l2} =!= prevEdge *)
        ] (* end If Head[#[[1]]] === Ray *)
    ]&, tileSegmentsRays]; (* end Scan *)
    If[scan =!= Null,
         Return[scan]];
    $Failed
  ]



(*============================== iBoundedDiagram =============================*)

iBoundedDiagram[input_, vorpts_, vorval_] :=
  Module[{newpts = vorpts, newval = vorval,
          inputNN, inputInfo, inputSegments, firstVorLabel,
          usedLabels, unusedLabels, rules},
    inputNN = NearestNeighbor[input, vorpts, vorval];
    If[Length[Union[inputNN]] != Length[inputNN],
       (* NOTE: it should not be necessary to require that all boundary
                points lie in unique Voronoi polygons.  This requirement
                is made to simplify writing the code. *)
       Message[BoundedDiagram::notuniq];
       Return[$Failed]];
    (* add the bounding polygon vertices to the new list of diagram vertices *)
    (* make labels for bounding polygon vertices *)
    inputLabels = Length[newpts] + Range[Length[input]];
    (* add bounding polygon vertices to list of diagram vertices *)
    newpts = Join[newpts, input];
    inputInfo = Transpose[{input, inputNN, inputLabels}];
    inputSegments = Transpose[{inputInfo, RotateLeft[inputInfo, 1]}];
    firstVorLabel = First[inputNN];

    (* compute new diagram vertices for EACH SEGMENT OF BOUNDING POLYGON *)
    scan = Scan[Module[{begBoundVert, begVorLabel, begVertLabel,
                        endBoundVert, endVorLabel, endVertLabel,
                        currentVertLabel, currentVorLabel, currentEdge,
                        nextVertLabel, nextVorLabel,  nextEdge,
                        currrentVorVertexLabels, int, intVert, beg, end, pos},
           {{begBoundVert, begVorLabel, begVertLabel},
            {endBoundVert, endVorLabel, endVertLabel}} = #;

           currentVertLabel = begVertLabel;
           currentVorLabel = begVorLabel;
           currentEdge = Null; (* the initial vertex is not an intersection
                                        with an edge of the diagram... so it
                                        does not lie on an edge *)

           (* go from ONE END OF the bounding polynomial SEGMENT TO THE
                OTHER (from beginning Voronoi tile to ending Voronoi tile) *)
           While[ currentVorLabel =!= endVorLabel,
                 currentVorVertexLabels = Select[newval,
                        (#[[1]] === currentVorLabel)&][[1, 2]];
                 (* Note that if an edge is a segment, then the ends of
                        the edge are traversed in the opposite order on
                        adjoining polygons... hence, the Reverse.
                    A Ray edge, on the other hand, is designated by a single
                        integer, not a list of two integers, so there is
                        no need for Reverse. *)
                 int = intersection[{begBoundVert, endBoundVert},
                        Transpose[{newpts[[currentVorVertexLabels]],
                                 currentVorVertexLabels}],
                        If[VectorQ[currentEdge], Reverse[currentEdge],
                                currentEdge] ];

                 (* does this mean that iBoundedDiagram
                        returns $Failed or that Scan does? *)
                 If[ int === $Failed, Return[$Failed] ];

                 {intVert, nextEdge} = int;
                 (* make a label for this new diagram vertex, formed by
                    intersecting a boundary segment with the original
                    diagram *)
                 AppendTo[newpts, intVert];
                 nextVertLabel = Length[newpts];

                 nextVorLabel = nextTile[currentVorLabel, nextEdge, newval];



                 (* ================================================== *)
                 If[ currentVorLabel === begVorLabel,
                    (* START of boundary segment *)
                    (* rotate val so that vertex label to be trimmed is at
                        beginning of list *)
                    beg = If[VectorQ[nextEdge], nextEdge[[1]], nextEdge];
                    While[First[currentVorVertexLabels] =!= beg,
                         currentVorVertexLabels =
                                 RotateLeft[currentVorVertexLabels]];

                    If[ begVorLabel === firstVorLabel,

                       (* trim vertex label first in list *)
                       currentVorVertexLabels =
                                 Drop[currentVorVertexLabels, 1],

                       (* trim more *)
                       While[Last[currentVorVertexLabels] =!= currentVertLabel,
                             currentVorVertexLabels =
                                 RotateLeft[currentVorVertexLabels]];
                       (* trim vertex labels from beginning of list *)
                       end = If[VectorQ[nextEdge], nextEdge[[1]], nextEdge];
                       currentVorVertexLabels =
                                 Drop[currentVorVertexLabels, -1];
                       currentVorVertexLabels = Drop[currentVorVertexLabels,
                                Position[currentVorVertexLabels, end][[1, 1]]]
                    ],

                    (* INTERMEDIATE portion of boundary segment *)
                    (* rotate val so that vertex labels to be trimmed are at
                        beginning of list *)
                    beg = If[VectorQ[currentEdge],
                         currentEdge[[1]], currentEdge];
                    While[First[currentVorVertexLabels] =!= beg,
                         currentVorVertexLabels =
                                 RotateLeft[currentVorVertexLabels]];
                    (* trim vertex labels from beginning of list *)
                    end = If[VectorQ[nextEdge], nextEdge[[1]], nextEdge];
                    currentVorVertexLabels = Drop[currentVorVertexLabels,
                         Position[currentVorVertexLabels, end][[1, 1]]]
                 ]; (* end If currentVorLabel === begVorLabel *)
                 (* ================================================== *)


                 currentVorVertexLabels =
                         Join[{currentVertLabel, nextVertLabel},
                        currentVorVertexLabels];

                 (* update val for tile designated by currentVorLabel *)
                 newval = (newval /.
                        {{currentVorLabel, _} -> {currentVorLabel,
                         currentVorVertexLabels}});
                 currentVorLabel = nextVorLabel;
                 currentEdge = nextEdge;
                 currentVertLabel = nextVertLabel
           ]; (* end While currentVorLabel =!= endVorLabel *)


           (* merge last piece in this segment of the boundary into val for
                tile *)
           currentVorVertexLabels = Select[newval,
                        (#[[1]] === currentVorLabel)&][[1, 2]];

           currentVorVertexLabels = If[VectorQ[currentEdge],
                Flatten[currentVorVertexLabels /.
                 {currentEdge[[1]] -> {currentVertLabel, endVertLabel}}],
                Flatten[currentVorVertexLabels /.
                 {currentEdge -> {currentVertLabel, endVertLabel}}]
           ];
           pos = Position[currentVorVertexLabels, endVertLabel];
           If[Length[pos] == 2,
                (* last segment of boundary *)
                currentVorVertexLabels = Take[currentVorVertexLabels,
                        pos[[2, 1]]-1]
           ];

           (* rotate vertex adjacency list so non-intersected sides of
                Voronoi polygon are listed first *)
           While[Last[currentVorVertexLabels] =!= endVertLabel,
                currentVorVertexLabels = RotateLeft[currentVorVertexLabels] ];


          (* update val for tile designated by currentVorLabel *)
          newval = (newval /.
             {{currentVorLabel, _} -> {currentVorLabel,
                         currentVorVertexLabels}});


         ]&, (* end Module *)
        inputSegments]; (* end Scan *)
    If[scan === $Failed, Return[$Failed]];

    (* clean up newpts and newval by eliminating old vertices that are no
        longer used in the val and relabeling *)
    usedLabels = Union[Flatten[Map[#[[2]]&, newval]]];
    unusedLabels = Complement[Range[Max[usedLabels]], usedLabels];
    newpts = Delete[newpts, Map[{#}&, unusedLabels]];
    rules = Thread[usedLabels -> -Range[Length[usedLabels]]];
    newval = Map[{#[[1]], (#[[2]] /. rules)}&, newval];
    newval = Map[{#[[1]], -#[[2]]}&, newval];
    {newpts, newval}

  ] (* end of iBoundedDiagram *)


(* ============================== nextTile ================================ *)

nextTile[currentVorLabel_, nextEdge_, newval_] :=
 (
   If[VectorQ[nextEdge],
      Select[newval,
                            ( !FreeQ[#[[2]], nextEdge[[1]]] &&
                              !FreeQ[#[[2]], nextEdge[[2]]] &&
                              #[[1]] =!= currentVorLabel )&][[1, 1]],
      Select[newval,
                             ( !FreeQ[#[[2]], nextEdge] &&
                               #[[1]] =!= currentVorLabel )&][[1, 1]]
   ]
 )




(*============================== BoundedDiagram ==============================*)

BoundedDiagram::ncon =
"Polygon boundary is not convex.  Boundary points must form a convex hull."

BoundedDiagram::notuniq =
"BoundedDiagram requires that boundary vertices lie in unique Voronoi polygons."

BoundedDiagram::notbd = "Not all points lie within the boundary."

BoundedDiagram::nodel = "Delaunay triangulation failed."

BoundedDiagram::novor = "Voronoi diagram failed."

BoundedDiagram::nobd = "Bounded diagram failed."

BoundedDiagram::nohull = "Convex hull failed."


(*** Only point set available. ***)
BoundedDiagram[ibound:{{_,_}..},set:{{_,_}..}]:=
  Module[{boundhull, internal, bound0, set0, scan, delaunay, delval, hull,
          voronoi, vorpts, vorval, bd},
   (
        bd
   ) /; ( boundhull = ConvexHull[ibound];
          bound = ibound[[boundhull]];
          If[Length[boundhull] != Length[ibound],
             Message[BoundedDiagram::ncon]; False,
             (* check that set is within boundhull *)
             internal = Apply[Plus, Take[bound, 3]]/3;
             bound0 = Map[(#-internal)&, bound];
             set0 = Map[(#-internal)&, set];
             scan = Scan[If[!ClosedPolygonMemberQ[N[#], N[bound0]],
                                Return[$Failed]]&, set0];
             If[scan === $Failed,
                Message[BoundedDiagram::notbd];  False,
                True]
          ] ) &&

        (* find triangulation and convex hull of set *)
        FreeQ[(delaunay=DelaunayTriangulation[set,Hull->True]),
                DelaunayTriangulation] &&
        If[ delaunay === $Failed,
           Message[BoundedDiagram::nodel];  False,
           True] &&

        (* find Voronoi diagram *)
        (
          {delval, hull} = delaunay;
          voronoi = VoronoiDiagram[set,delval,hull];
          If[ voronoi === $Failed,
             Message[BoundedDiagram::novor];  False,
             True] ) &&

        (* find bounded diagram *)
        (
          {vorpts, vorval} = voronoi;
          bd = iBoundedDiagram[bound, vorpts, vorval];
          If[ bd === $Failed,
             Message[BoundedDiagram::nobd];  False,
             True] )

  ] /; numericalQ[N[{ibound, set}]] && Length[ibound] >= 3


(*** Both point set and Delaunay vertex adjacency list available. ***)
BoundedDiagram[ibound:{{_,_}..},set:{{_,_}..},
        delval:{{_Integer,{_Integer..}}..}]:=
  Module[{boundhull, internal, bound0, set0, scan, hull, voronoi,
          vorpts, vorval, bd},
   (
        bd
   ) /; ( boundhull = ConvexHull[ibound];
          bound = ibound[[boundhull]];
          (* check convex hull of boundary *)
          If[Length[boundhull] != Length[ibound],
             Message[BoundedDiagram::ncon];
             False,
             (* check that set is within boundhull *)
             internal = Apply[Plus, Take[bound, 3]]/3;
             bound0 = Map[(#-internal)&, bound];
             set0 = Map[(#-internal)&, set];
             scan = Scan[If[!ClosedPolygonMemberQ[N[#], N[bound0]],
                                Return[$Failed]]&, set0];
             If[scan === $Failed,
                Message[BoundedDiagram::notbd];  False,
                True]
          ] ) &&

          (* find convex hull of set *)
          FreeQ[(hull = TriangulationToHull[numericalize[set], delval]),
                        TriangulationToHull]  &&

          If[hull === $Failed,
             Message[BoundedDiagram::nohull];  False,
             True] &&

          (* find Voronoi diagram *)
          ( voronoi = VoronoiDiagram[set,delval,hull];
            If[ voronoi === $Failed,
             Message[BoundedDiagram::novor];  False,
             True] ) &&

          (* find bounded diagram *)
          ( {vorpts, vorval} = voronoi;
            bd = iBoundedDiagram[bound, vorpts, vorval];
            If[ bd === $Failed,
               Message[BoundedDiagram::nobd];  False,
               True] )

  ] /; numericalQ[N[{ibound, set}]] && Length[ibound] >= 3 &&
       (Max[Flatten[delval]] <= Length[set]) &&  (* val refers to valid pts *)
        Length[delval] >= 3             (* val must have at least 1 triangle *)



(*** Point set, Delaunay vertex adjacency list, and convexhull available. ***)
BoundedDiagram[ibound:{{_,_}..},set:{{_,_}..},
        delval:{{_Integer,{_Integer..}}..},hull:{_Integer..}]:=
  Module[{boundhull, internal, bound0, set0, scan, voronoi,
          vorpts, vorval, bd},
   (
        bd
   ) /; ( boundhull = ConvexHull[ibound];
          bound = ibound[[boundhull]];
          If[Length[boundhull] != Length[ibound],
             Message[BoundedDiagram::ncon]; False,
             (* check that set is within boundhull *)
             internal = Apply[Plus, Take[bound, 3]]/3;
             bound0 = Map[(#-internal)&, bound];
             set0 = Map[(#-internal)&, set];
             scan = Scan[If[!ClosedPolygonMemberQ[N[#], N[bound0]],
                                Return[$Failed]]&, set0];
             If[scan === $Failed,
                Message[BoundedDiagram::notbd]; False,
                True]
          ] ) &&

        (* find Voronoi diagram *)
        ( voronoi = VoronoiDiagram[set,delval,hull];
          If[ voronoi === $Failed,
             Message[BoundedDiagram::novor];  False,
             True] ) &&

        (* find bounded diagram *)
        ( {vorpts, vorval} = voronoi;
          bd = iBoundedDiagram[bound, vorpts, vorval];
          If[ bd === $Failed,
             Message[BoundedDiagram::nobd];  False,
             True] )

  ] /; Length[ibound] >= 3 &&
       (Max[Flatten[delval]] <= Length[set]) &&  (* val refers to valid pts *)
       (Max[hull] <= Length[set]) &&     (* hull refers to valid pts *)
       Length[delval] >= 3             (* val includes at least 1 triangle *)



(* ======== ConvexHullMedian multivariate robust measure of location ======= *)

(* signed volume of p-dimensional simplex is actually
	signedvolume[m]/Factorial[p] *)

signedvolume[m_] := Det[Prepend[
	Transpose[m], Table[1, {Length[m]}] ]] (* matrix is ((p+1) x p) *)

ConvexHullMedian::notimp =
"ConvexHullMedian[matrix] not implemented for matrices n x p, where p > 2."

ConvexHullMedian[coord_?MatrixQ] :=
   Module[{layer = Layer[coord], innermost, n, median},
     (
      innermost = Last[layer];
      n = Length[innermost];
      median = Mean[coord[[innermost]]]
     ) /; FreeQ[layer, Layer]
   ] /; Length[coord[[1]]] == 2

ConvexHullMedian[m_?MatrixQ] :=
   Module[{},
	Null /; (Message[ConvexHullMedian::notimp];
		 False)
   ] /; Length[m[[1]]] > 2

(* ================================ Layer ============================== *)
(* 
Layer[{{x11, ... , x1p}, ... , {xn1, ... , xnp}}] gives the indices of
those points lying on the convex layers of the p-dimensional n-point input
set ordered from outermost to innermost.  Layer[{{x11, ... , x1p}, ... ,
{xn1, ... , xnp}}, m] gives the outermost m layers.
*)

Layer[coord_] :=
	Module[{result = iLayer[coord, Infinity]},
	       result /; result =!= $Failed
	]

Layer[coord_, m_Integer?Positive] :=
	Module[{result = iLayer[coord, m]},
			result /; result =!= $Failed
	]

iLayer[coord_, m_] :=
	Module[{n, p, output = {}, i, singlelayer, vertices},
	   {n, p} = Dimensions[coord];
	   vertices = Range[n];
	   If[n <= p+1, Return[{vertices}]];
	   For[i = 1, (i <= m) && (Length[vertices] > p+1), i++,
	      singlelayer = ConvexHull[coord[[vertices]]];
	      If[!FreeQ[singlelayer, ConvexHull],
	         (* do not issue message when Layer is used internally... 
		 Message[Layer::conv, i]; *)
		 Return[$Failed]
	      ];
	      singlelayer = vertices[[singlelayer]];
	      output = Join[output, {singlelayer}];
	      vertices = Complement[vertices, singlelayer];
	   ];
	   If[Length[vertices] > 0 && m === Infinity,
		(* Anywhere from 1 to p+1 points in the interior. *)
		output = Join[output, {vertices}]];
	   output
        ]

Layer::conv = "ConvexHull failed at layer ``."

(* ======================= ConvexHullArea ======================= *)
(* ====== scalar-valued multivariate robust measure of dispersion ====== *)

ConvexHullArea::notimp =
"ConvexHullArea[matrix] not implemented for matrices n x p, \
where p > 2."

ConvexHullArea[coord_?MatrixQ] :=
   Module[{result},
     (
	result
     ) /; FreeQ[hull = ConvexHull[coord], ConvexHull] &&
	   FreeQ[result = iConvexHullArea[coord, hull], $Failed]
   ] /; Length[coord[[1]]] == 2

iConvexHullArea[coord_, hull_] := 
   Module[{mean = Mean[coord], pairs},
     pairs = Partition[Append[hull, First[hull]], 2, 1];
     Apply[Plus,
	   Map[Abs[signedvolume[Append[coord[[#]], mean]]]&,
	       pairs]]/2
   ]



(* ========================================================================== *)
End[]

EndPackage[] 


(* :Examples:

input = {{4.4,14},{6.7,15.25},{6.9,12.8},{2.1,11.1},{9.5,14.9},{13.2,11.9},
	{10.3,12.3},{6.8,9.5},{3.3,7.7},{.6,5.1},{5.3,2.4},{8.45,4.7},
	{11.5,9.6},{13.8,7.3},{12.9,3.1},{11,1.1}};
query = Map[(#+{.001,.001})&,input]
input3D = Map[{#[[1]],#[[2]],Sqrt[64-(#[[1]]-8)^2-(#[[2]]-8)^2]}&,input]
b1 = {{0, 1}, {14, 1}, {14, 16}, {0, 16}}

*****************************************************************************
  convex hull of "input":
		ConvexHull[input]

  plot convex hull of "input":
		PlanarGraphPlot[input,ConvexHull[input]]

*****************************************************************************
  Delaunay triangulation of "input":
		val = DelaunayTriangulation[input]

  plot Delaunay triangulation of "input" where the Delaunay vertex adjacency
	list is not available:
		PlanarGraphPlot[input]

  plot triangulation of "input" where the triangulation vertex adjacency list
	is "val" ("val" need not represent a Delaunay triangulation):
		PlanarGraphPlot[input,val]

  plot triangular surface where the vertex adjacency list representing the
	Delaunay triangulation of the projected input is not available:
		TriangularSurfacePlot[input3D]

  plot triangular surface where the vertex adjacency list representing the
	triangulation of the projected input is "val" ("val" need not
	represent a Delaunay triangulation):
		val = DelaunayTriangulation[Map[Drop[#,-1]&,input3D]];
		TriangularSurfacePlot[input3D,val]

*****************************************************************************
  Voronoi diagram of "input":
		{diagvert,diagval} = VoronoiDiagram[input]

  Voronoi diagram of "input" where the vertex adjacency list representing the
	dual Delaunay triangulation is "val":
		VoronoiDiagram[input,val]

  Voronoi diagram of "input" where the vertex adjacency list representing the
	dual Delaunay triangulation is "val" and the convex hull of "input"
	is "hull":
		{val,hull} = DelaunayTriangulation[input,Hull->True];
		VoronoiDiagram[input,val,hull]

  plot Voronoi diagram of "input" where the Voronoi vertices and vertex
	adjacency list are not available:
		DiagramPlot[input]

  plot diagram of "input" where the diagram vertices are "diagvert" and the
	diagram vertex adjacency list is "diagval" ("diagvert" and "diagval"
	need not represent a Voronoi diagram):
		DiagramPlot[input,diagvert,diagval]

  nearest neighbors of "query" where the Voronoi diagram vertices and vertex
	adjacency list associated with "input" are not available:
		NearestNeighbor[query,input]

  nearest neighbors of "query" where the Voronoi diagram of "input" is
	represented by the vertices "diagvert" and the vertex adjacency list
	"diagval":
		NearestNeighbor[query,diagvert,diagval]

*****************************************************************************
  Delaunay triangulation query:
	DelaunayTriangulationQ[input,val]

	notdelaunayval = Join[{{1,{4,2}},{2,{1,4,3,5}},{3,{2,4,8,7,5}},
				{4,{10,9,3,2,1}}}, Drop[val,4]];
	DelaunayTriangulationQ[input,notdelaunayval]

  Plotting a non-Delaunay triangulation:
	PlanarGraphPlot[input,notdelaunayval]

*****************************************************************************
   areas of tiles (some bounded, some unbounded):
        TileAreas[input,diagvert,diagval]

   Bounded diagram of "input" where bound is the rectangle "b1":
                {diagvert1,diagval1} = BoundedDiagram[b1, input]

   Bounded diagram of "input" where the vertex adjacency list representing the
        dual Delaunay triangulation is "val":
                val = DelaunayTriangulation[input];
                {diagvert1,diagval1} = BoundedDiagram[b1,input,val]

   Bounded diagram of "input" where the vertex adjacency list representing the
        dual Delaunay triangulation is "val" and the convex hull of "input"
        is "hull":
                {val,hull} = DelaunayTriangulation[input,Hull->True];
                {diagvert1,diagval1} = BoundedDiagram[b1,input,val,hull]

   plot diagram of "input" where the diagram vertices are "diagvert1" and the
        diagram vertex adjacency list is "diagval1" ("diagvert1" and "diagval1"
        need not represent a Voronoi diagram):
                DiagramPlot[input,diagvert1,diagval1]

   areas of (bounded) tiles:
        TileAreas[input,diagvert1,diagval1]

*)



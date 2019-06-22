(* ::Package:: *)

(* :Title: Polytopes *)

(* :Author: Stephen Wolfram and Roman Maeder *)

(* :Summary:
This package contains functions that give geometrical 
characteristics of regular polygons and polyhedra.
*)

(* :Context: Polytopes` *)

(* :Package Version: 1.4 *)

(* :Copyright: Copyright 1987-2007, Wolfram Research, Inc. *)

(* :History:
	Version 1.1 by Stephen Wolfram (Wolfram Research), February 1987.
	Modified by E.C. Martin (Wolfram Research), December 1990, May 1996.
	  Moved definitions for Tetrahedron, Cube, Octahedron,
	  Dodecahedron, Icosahedron, and Hexahedron from
	  Graphics`Polyhedra` to Geometry`Polytopes`.
	Version 1.4 by John M. Novak, February 1998 -- modified definitions
	  for Vertices to use exact values. This involved precomputing vertices
	  for some polyhedra, see notes in code.
    Updated to fit new documentation paradigm, December 2006.
*)

(* :Keywords: *)

(* :Source:
	H. S. M. Coxeter, Regular Polytopes, (Dover, 1973) *)

(* :Warning: None. *)

(* :Mathematica Version: 2.0 *)

(* :Limitation: None. *)

(* :Discussion: *)
		
		
BeginPackage["Polytopes`"]


If[Not@ValueQ[NumberOfVertices::usage],NumberOfVertices::usage =
"NumberOfVertices[polytope] gives the number of vertices of polytope."];
If[Not@ValueQ[NumberOfEdges::usage],NumberOfEdges::usage =
"NumberOfEdges[polytope] gives the number of edges of polytope."];
If[Not@ValueQ[NumberOfFaces::usage],NumberOfFaces::usage=
"NumberOfFaces[polytope] gives the number of faces of polytope."];
If[Not@ValueQ[Vertices::usage],Vertices::usage =
"Vertices[polytope] gives a list of the vertex coordinates of \
polytope.  To get the number of vertices of polytope use \
NumberOfVertices[polytope].  Edges are not necessarily \
normalized to unit length."];
If[Not@ValueQ[Faces::usage],Faces::usage =
"Faces[polytope] gives a list of the faces of polytope.  Each face \
is a list of the numbers of the vertices that comprise that face. \
To get the number of faces of polytope use NumberOfFaces[polytope]."];
If[Not@ValueQ[Area::usage],Area::usage =
"Area[polygon] gives the area of polygon, when the edges of \
polygon have unit length.  Area[polyhedron] gives \
the area of a face of polyhedron, when the edges of polyhedron \
have unit length."];
If[Not@ValueQ[InscribedRadius::usage],InscribedRadius::usage =
"InscribedRadius[polytope] gives the radius of an inscribed \
circle/sphere of polytope, when the edges of polytope have unit length."];
If[Not@ValueQ[CircumscribedRadius::usage],CircumscribedRadius::usage =
"CircumscribedRadius[polytope] gives the radius of a circumscribed \
circle/sphere of polytope, when the edges of polytope have unit length."];

(* No longer exported because these are only relevant for polyhedra (see below)

Volume::usage =
"Volume[polytope] gives the volume of polytope, when the edges of \
polytope have unit length."
Dual::usage =
"Dual[p] gives the dual of polytope p (if it exists)."
Schlafli::usage =
"Schlafli[p] gives the Schlafli symbol for polytope p."*)

MapThread[(Evaluate[#]::"usage" =
	StringJoin[ToString[#],
	   " is a regular polygon with ", #2,
	   " edges, for use with polytope functions."])&,
	{{Digon,Triangle,Square,Pentagon,Hexagon,Heptagon,Octagon,Nonagon,
	Decagon,Undecagon,Dodecagon},
	{"two", "three", "four", "five", "six", "seven", "eight", "nine",
	 "ten", "eleven", "twelve"}}]

(*Removed for version 6--superseded by PolyhedronData[].

MapThread[(Evaluate[#]::"usage" =
        StringJoin[ToString[#], " is a polyhedron with ", #2, " faces and ",
          #3, " vertices, for use with polytope functions."])&,
        {{Tetrahedron, Cube, Hexahedron, Octahedron, Dodecahedron, Icosahedron},
        {"four", "six", "six", "eight", "twelve", "twenty"},
        {"four", "eight", "eight", "six", "twenty", "twelve"}}]*)


Begin["`Private`"]

(* =========================== 2 dimensions =============================== *)

num[Digon] = 		2
num[Triangle] = 	3
num[Square] =		4
num[Pentagon] =		5
num[Hexagon] =		6
num[Heptagon] =		7
num[Octagon] =		8
num[Nonagon] =		9
num[Decagon] =		10
num[Undecagon] =	11
num[Dodecagon] =	12

NumberOfVertices[polygon_] :=   
(
Module[{nn},
			 (
			  nn
			 ) /; (nn = num[polygon]; FreeQ[nn, num])
			])
NumberOfEdges[polygon_] :=	
(
Module[{nn},
                         (
			  nn
			 ) /; (nn = num[polygon]; FreeQ[nn, num])
                        ])
NumberOfFaces[polygon_] := 	
(
Module[{nn},
			 (
			   1
		         ) /; (nn = num[polygon]; FreeQ[nn, num])
			])
Vertices[polygon_] :=	
(
Module[{nn},
			  (
			  Table[{Cos[2Pi i/nn], Sin[2Pi i/nn]}, {i,1,nn}]
			  ) /; (nn = num[polygon]; FreeQ[nn, num])
			])
Faces[polygon_] := 	
(
Module[{nn},
			  (
			  Range[nn]
			  ) /; (nn = num[polygon]; FreeQ[nn, num])
			])
Area[polygon_] :=	(
Module[{nn},
			 (
			  nn/(4 Tan[Pi/nn])
			 ) /; (nn = num[polygon]; FreeQ[nn, num])
		        ])
InscribedRadius[polygon_] :=  (
Module[{nn},
			 (
		    	  1/(2 Tan[Pi/nn])
			 ) /; (nn = num[polygon]; FreeQ[nn, num])
		        ])
CircumscribedRadius[polygon_] :=  (
Module[{nn},
			     (
			 	1/(2 Sin[Pi/nn])
			     ) /; (nn = num[polygon]; FreeQ[nn, num])
			    ])



(* ============================= 3 dimensions ============================ *)

PT$AllCyc[list_] := Array[RotateLeft[list,#]&, Length[list], 0]

PT$AllSign = Flatten[Array[(-1)^List[##]&, {2,2,2}, 0], 2]


(* Tetrahedron *)
Unprotect[Tetrahedron]; (* System symbol in V9 *)
NumberOfVertices[Tetrahedron] ^=	4
NumberOfEdges[Tetrahedron] ^=		6
NumberOfFaces[Tetrahedron] ^=		4
(* Coords[Tetrahedron] ^= 	{{1,1,1}} ~Join~ PT$AllCyc[{1,-1,-1}] *)
Tetrahedron /: Vertices[Tetrahedron] = 
     {{0, 0, 3^(1/2)}, {0, (2*2^(1/2)*3^(1/2))/3, -3^(1/2)/3},
      {-2^(1/2), -(2^(1/2)*3^(1/2))/3, -3^(1/2)/3},
      {2^(1/2), -(2^(1/2)*3^(1/2))/3, -3^(1/2)/3}}
Tetrahedron /: Faces[Tetrahedron] =
     {{1, 2, 3}, {1, 3, 4}, {1, 4, 2}, {2, 4, 3}}
Area[Tetrahedron] ^= 		Sqrt[3]/4
InscribedRadius[Tetrahedron] ^= 	Sqrt[6]/12
CircumscribedRadius[Tetrahedron] ^= 	Sqrt[6]/4
Volume[Tetrahedron] ^= 		Sqrt[2]/12
Dual[Tetrahedron] ^= 		Tetrahedron
Schlafli[Tetrahedron] ^= 	{3,3}
Protect[Tetrahedron]; (* System symbol in V9 *)

(* Hexahedron/Cube *)

Unprotect[Hexahedron]; (* System symbol in V9 *)
NumberOfVertices[Hexahedron] ^= 		8
NumberOfEdges[Hexahedron] ^= 			12
NumberOfFaces[Hexahedron] ^= 			6
(* Coords[Hexahedron] ^= 		PT$AllSign *)
Hexahedron /: Vertices[Hexahedron] = Sqrt[2]/2 *
     {{1, 1, 1}, {-1, 1, 1}, {-1, -1, 1}, {1, -1, 1},
      {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1}}
Hexahedron /: Faces[Hexahedron] =
     {{1, 2, 3, 4}, {1, 4, 6, 7}, {1, 7, 8, 2},
      {2, 8, 5, 3}, {5, 8, 7, 6}, {3, 5, 6, 4}}
Area[Hexahedron] ^= 			1
InscribedRadius[Hexahedron] ^= 		1/2
CircumscribedRadius[Hexahedron] ^= 		Sqrt[3]/2
Volume[Hexahedron] ^= 		1
Dual[Hexahedron] ^= 			Octahedron
Schlafli[Hexahedron] ^= 		{4,3}
Protect[Hexhedron];

Cube/: NumberOfVertices[Cube] := NumberOfVertices[Hexahedron]
Cube/: NumberOfEdges[Cube] := NumberOfEdges[Hexahedron]
Cube/: NumberOfFaces[Cube] := NumberOfFaces[Hexahedron]
(* Cube/: Coords[Cube] := Coords[Hexahedron] *)
Cube/: Vertices[Cube] := Vertices[Hexahedron]
Cube/: Faces[Cube] := Faces[Hexahedron]
Cube/: Area[Cube] := Area[Hexahedron]
Cube/: InscribedRadius[Cube] := InscribedRadius[Hexahedron]
Cube/: CircumscribedRadius[Cube] := CircumscribedRadius[Hexahedron]
Cube/: Volume[Cube] := Volume[Hexahedron]
Cube/: Dual[Cube] := Dual[Hexahedron]
Cube/: Schlafli[Cube] := Schlafli[Hexahedron]


(* Octahedron *)

NumberOfVertices[Octahedron] ^= 	6
NumberOfEdges[Octahedron] ^= 		12
NumberOfFaces[Octahedron] ^= 		8
(* Coords[Octahedron] ^=	PT$AllCyc[{1,0,0}] ~Join~ PT$AllCyc[{-1,0,0}] *)
Octahedron /: Vertices[Octahedron] = Sqrt[2] *
     {{0, 0, 1}, {1, 0, 0}, {0, 1, 0},
      {0, 0, -1}, {-1, 0, 0}, {0, -1, 0}}
Octahedron /: Faces[Octahedron] =
     {{1, 2, 3}, {1, 3, 5}, {1, 5, 6}, {1, 6, 2},
      {2, 6, 4}, {2, 4, 3}, {4, 6, 5}, {3, 4, 5}}
Area[Octahedron] ^= 		Sqrt[3]/4
InscribedRadius[Octahedron] ^= 	Sqrt[6]/6
CircumscribedRadius[Octahedron] ^= 	Sqrt[2]/2
Volume[Octahedron] ^= 		Sqrt[2]/3
Dual[Octahedron] ^= 		Cube
Schlafli[Octahedron] ^= 	{3,4}

(* Dodecahedron *)

NumberOfVertices[Dodecahedron] ^= 	20
NumberOfEdges[Dodecahedron] ^= 		30
NumberOfFaces[Dodecahedron] ^= 		12
(* Vertices of the Dodecahedron were originally computed by the following
   combinatoric operation:
   Coords[Dodecahedron] ^=
	PT$AllSign ~Join~ 
	   Flatten[
	      Array[PT$AllCyc[{0,(-1)^#1 GoldenRatio^-1,(-1)^#2 GoldenRatio}]&,
	         {2,2}],
	          2]
    When the package was combined with Polyhedra.m, this switched to
    simply computing DualVertices[Icosahedron]. However, this forced the
    values to be numeric. M. Trott claimed (rightly, I think) that
    Polytopes.m needs exact values. Unfortunately, computing these on the
    fly leads to too much overhead. So, these will be set to precomputed
    exact values. (Couldn't regress to the combinatoric definition above,
    because orientation and radius of the polyhedron needed to remain the
    same as Polyhedra.m for compatability with people's graphics.)
*)
Dodecahedron/: Vertices[Dodecahedron] =
 {{Sqrt[1/2 - 1/(2*Sqrt[5])], (3 - Sqrt[5])/2, 
  Sqrt[1/2 + 1/(2*Sqrt[5])]}, {-Sqrt[5/2 - 11/(2*Sqrt[5])], 
  (-1 + Sqrt[5])/2, Sqrt[1/2 + 1/(2*Sqrt[5])]}, 
 {-2*Sqrt[(5 - 2*Sqrt[5])/5], 0, Sqrt[1/2 + 1/(2*Sqrt[5])]}, 
 {-Sqrt[5/2 - 11/(2*Sqrt[5])], (1 - Sqrt[5])/2, 
  Sqrt[1/2 + 1/(2*Sqrt[5])]}, {Sqrt[1/2 - 1/(2*Sqrt[5])], 
  (-3 + Sqrt[5])/2, Sqrt[1/2 + 1/(2*Sqrt[5])]}, 
 {Sqrt[1/2 + 1/(2*Sqrt[5])], (-1 + Sqrt[5])/2, 
  Sqrt[5/2 - 11/(2*Sqrt[5])]}, {-Sqrt[(5 - 2*Sqrt[5])/5], 1, 
  Sqrt[5/2 - 11/(2*Sqrt[5])]}, {-Sqrt[(2*(5 - Sqrt[5]))/5], 
  0, Sqrt[5/2 - 11/(2*Sqrt[5])]}, {-Sqrt[(5 - 2*Sqrt[5])/5], 
  -1, Sqrt[5/2 - 11/(2*Sqrt[5])]}, 
 {Sqrt[1/2 + 1/(2*Sqrt[5])], (1 - Sqrt[5])/2, 
  Sqrt[5/2 - 11/(2*Sqrt[5])]}, {Sqrt[(5 - 2*Sqrt[5])/5], 1, 
  -Sqrt[5/2 - 11/(2*Sqrt[5])]}, {-Sqrt[1/2 + 1/(2*Sqrt[5])], 
  (-1 + Sqrt[5])/2, -Sqrt[5/2 - 11/(2*Sqrt[5])]}, 
 {-Sqrt[1/2 + 1/(2*Sqrt[5])], (1 - Sqrt[5])/2, 
  -Sqrt[5/2 - 11/(2*Sqrt[5])]}, {Sqrt[(5 - 2*Sqrt[5])/5], 
  -1, -Sqrt[5/2 - 11/(2*Sqrt[5])]}, 
 {Sqrt[(2*(5 - Sqrt[5]))/5], 0, 
  -Sqrt[5/2 - 11/(2*Sqrt[5])]}, {Sqrt[5/2 - 11/(2*Sqrt[5])], 
  (-1 + Sqrt[5])/2, -Sqrt[1/2 + 1/(2*Sqrt[5])]}, 
 {-Sqrt[1/2 - 1/(2*Sqrt[5])], (3 - Sqrt[5])/2, 
  -Sqrt[1/2 + 1/(2*Sqrt[5])]}, {-Sqrt[1/2 - 1/(2*Sqrt[5])], 
  (-3 + Sqrt[5])/2, -Sqrt[1/2 + 1/(2*Sqrt[5])]}, 
 {Sqrt[5/2 - 11/(2*Sqrt[5])], (1 - Sqrt[5])/2, 
  -Sqrt[1/2 + 1/(2*Sqrt[5])]}, {2*Sqrt[(5 - 2*Sqrt[5])/5], 
  0, -Sqrt[1/2 + 1/(2*Sqrt[5])]}};
Dodecahedron/: Faces[Dodecahedron] =
    {{1, 2, 3, 4, 5}, {1, 5, 10, 15, 6}, {1, 6, 11, 7, 2},
     {2, 7, 12, 8, 3}, {3, 8, 13, 9, 4}, {9, 14, 10, 5, 4},
     {6, 15, 20, 16, 11}, {7, 11, 16, 17, 12}, {8, 12, 17, 18, 13},
     {9, 13, 18, 19, 14}, {10, 14, 19, 20, 15}, {16, 20, 19, 18, 17}};
Area[Dodecahedron] ^= 		Sqrt[25+10 Sqrt[5]]/4
InscribedRadius[Dodecahedron] ^= 	Sqrt[250+110 Sqrt[5]]/20
CircumscribedRadius[Dodecahedron] ^= 	(Sqrt[15]+Sqrt[3])/4
Volume[Dodecahedron] ^= 	(15 + 7 Sqrt[5])/4
Dual[Dodecahedron] ^= 		Icosahedron
Schlafli[Dodecahedron] ^= 	{5,3}

(* Icosahedron *)

NumberOfVertices[Icosahedron] ^= 	12
NumberOfEdges[Icosahedron] ^= 		30
NumberOfFaces[Icosahedron] ^= 		20
(* Vertices of Icosahedron suffer similar issues to Dodecahedron
   above. The computational overhead isn't as great, but the
   expression could still be considerably simplified. First, the
   old (V2.2) definition:
   Coords[Icosahedron] ^=
	Flatten[ Array[PT$AllCyc[{0,(-1)^#1 GoldenRatio,(-1)^#2}]&,
		{2,2}],
		2]
    Next, the precomputed exact values based on Polyhedra.m. The
    original formula is:
    SphericalToCartesian /@
        {{0,0},
         {ArcTan[2], 0}, {ArcTan[2], 2Pi/5}, {ArcTan[2], 4Pi/5},
           {ArcTan[2], 6Pi/5}, {ArcTan[2], 8Pi/5},
         {Pi - ArcTan[2], Pi/5}, {Pi - ArcTan[2], 3Pi/5},
           {Pi - ArcTan[2], 5Pi/5}, {Pi - ArcTan[2], 7Pi/5},
           {Pi - ArcTan[2], 9Pi/5},
         {Pi,0}
        })/(1/2 + Cos[ArcTan[2]]/2)^(1/2)
    The cached values are:
*)
Icosahedron /: Vertices[Icosahedron] =
 {{0, 0, Sqrt[5/2 - Sqrt[5]/2]}, {Sqrt[(2*(5 - Sqrt[5]))/5], 
  0, Sqrt[1/2 - 1/(2*Sqrt[5])]}, {Sqrt[(5 - 2*Sqrt[5])/5], 
  1, Sqrt[1/2 - 1/(2*Sqrt[5])]}, 
 {-Sqrt[1/2 + 1/(2*Sqrt[5])], (-1 + Sqrt[5])/2, 
  Sqrt[1/2 - 1/(2*Sqrt[5])]}, {-Sqrt[1/2 + 1/(2*Sqrt[5])], 
  (1 - Sqrt[5])/2, Sqrt[1/2 - 1/(2*Sqrt[5])]}, 
 {Sqrt[(5 - 2*Sqrt[5])/5], -1, Sqrt[1/2 - 1/(2*Sqrt[5])]}, 
 {Sqrt[1/2 + 1/(2*Sqrt[5])], (-1 + Sqrt[5])/2, 
  -Sqrt[1/2 - 1/(2*Sqrt[5])]}, {-Sqrt[(5 - 2*Sqrt[5])/5], 1, 
  -Sqrt[1/2 - 1/(2*Sqrt[5])]}, {-Sqrt[(2*(5 - Sqrt[5]))/5], 
  0, -Sqrt[1/2 - 1/(2*Sqrt[5])]}, {-Sqrt[(5 - 2*Sqrt[5])/5], 
  -1, -Sqrt[1/2 - 1/(2*Sqrt[5])]}, 
 {Sqrt[1/2 + 1/(2*Sqrt[5])], (1 - Sqrt[5])/2, 
  -Sqrt[1/2 - 1/(2*Sqrt[5])]}, 
 {0, 0, -Sqrt[5/2 - Sqrt[5]/2]}};

Icosahedron /: Faces[Icosahedron] =
  {{1, 2, 3}, {1, 3, 4}, {1, 4, 5}, {1, 5, 6}, {1, 6, 2}, {2, 7, 3},
   {3, 8, 4}, {4, 9, 5}, {5, 10, 6}, {6, 11, 2}, {7, 8, 3}, {8, 9, 4},
   {9, 10, 5}, {10, 11, 6}, {11, 7, 2}, {7, 12, 8}, {8, 12, 9},
   {9, 12, 10}, {10, 12, 11}, {11, 12, 7}};
Area[Icosahedron] ^= 		Sqrt[3]/4
InscribedRadius[Icosahedron] ^= 	Sqrt[42+18 Sqrt[5]]/12
CircumscribedRadius[Icosahedron] ^= 	Sqrt[10+2 Sqrt[5]]/4
Volume[Icosahedron] ^= 		5 (3 + Sqrt[5])/12
Dual[Icosahedron] ^= 		Dodecahedron
Schlafli[Icosahedron] ^= 	{3,5}

(* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *)
(* DualFaces, DualVertices, and SphericalToCartesian taken from
	 Graphics`Polyhedra` by Roman Maeder *)

DualFace[ vertex_, faces_ ] :=
    Block[{incident, current, newfaces={}, newface},
        incident = Select[ faces, MemberQ[#, vertex]& ];
        incident = RotateLeft[#, Position[#, vertex][[1,1]]-2]& /@ incident;
        incident = Take[#, 3]& /@ incident;
        current = incident[[1]];
        While[incident =!= {},
            newface = Select[ incident,
              Length[Intersection[#, current]] > 1& ] [[1]];
            AppendTo[ newfaces,
              Position[faces, _List?(Length[Intersection[#, newface]]==3 &)] [[1
, 1]] ];
            current = newface;
            incident = Complement[ incident, {current} ];
        ];
        newfaces
    ]

DualFaces[name_] :=
        Block[{i, faces = Faces[name], vertices = Vertices[name]},
                Table[ DualFace[i, faces], {i, Length[vertices]} ]
        ]

DualVertices[name_] :=
        Block[{faces = Faces[name], vertices = Vertices[name],
               dvertices, length1, length2},
                dvertices = apex /@ (vertices[[#]]&) /@ faces;
                length1 = norm[ (vertices[[faces[[1,1]]]] +
                                vertices[[faces[[1,2]]]])/2 ];
                dvertices = dvertices / length1;
                dvertices = 1/norm[#]^2 # & /@ dvertices;
                dvertices
        ]

norm[ v_ ] := Sqrt[Plus @@ (v^2)]
apex[ v_ ] := Plus @@ v / Length[v]

SphericalToCartesian[{theta_, phi_}] :=
        {Sin[theta] Cos[phi], Sin[theta] Sin[phi], Cos[theta]}


End[]

EndPackage[]

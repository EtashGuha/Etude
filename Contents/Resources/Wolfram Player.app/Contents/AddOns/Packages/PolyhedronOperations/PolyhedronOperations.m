(*:Copyright: Copyright 1989-2007, Wolfram Research, Inc. *)

(*:Mathematica Version: 4.0 *)

(*:Package Version: 1.8.5 *)

(*:Name: Graphics`Polyhedra` *)

(*:Title: Graphics with Platonic Polyhedra  *)

(*:Context: Graphics`Polyhedra` *)

(*:Author: Roman Maeder *)

(*:Keywords:
	Polyhedron, vertices, faces, Stellate, Geodesate,
	Truncate, OpenTruncate,
	Tetrahedron, Cube, Octahedron,
	Dodecahedron, Icosahedron, Hexahedron,
	GreatDodecahedron, SmallStellatedDodecahedron,
	GreatStellatedDodecahedron, GreatIcosahedron
*)

(*:Requirements: None. *)

(*:Warnings:
	Faces and Vertices fail on
	GreatDodecahedron, SmallStellatedDodecahedron, and
	GreatStellatedDodecahedron.
*)

(*:Limitations:
	Truncate sometimes creates non-convex 3D polygons which are
	not displayed correctly.
*)


(*:Source:
	Roman E. Maeder: Programming in Mathematica, 2nd Ed.,
	Addison-Wesley, 1991.
*)

(*:History:
  Version 1.5: 
        Added the functions Geodesate, Truncate, and 
           OpenTruncate, Jeff Adams,  August 1992.
  Version 1.6: 
        Fixed Truncate so that it works with v1.1 Polyhedra.m
           polyhedra specifications.
        Set Options[Polyhedron] = Options[Graphics3D] and made PlotRange->All
	   the default setting.
        Allow the ratio value for Truncate and OpenTruncate to include 0.5.
  Version 1.7:
	Uses standard FilterOptions.
  Version 1.8:
	Moved definitions for Tetrahedron, Cube, Octahedron,
		Dodecahedron, Icosahedron, and Hexahedron
		to Geometry`Polytopes` and added Geometry`Polytopes`
		to BeginPackage.  Now both packages can be used together
		without symbol shadowing.
  Version 1.8.5 by John M. Novak, February 1998 -- Geometry`Polytopes`
        modified to return exact values for the vertices, so a numeric
        cache of the values is now being created. Polyhedron[] is redefined
        to use the numeric cache. Also fixed so that the standard polyhedra
        defined as stellations of other polyhedra scale and translate right.
*)


(*:Summary:
This package provides graphics primitives for rendering various
Platonic polyhedra and functions for accessing the coordinates
of the vertices and the vertex numbers of the faces.  The package
also includes functions for operating on polyhedra, such as
Stellate, Geodesate, Truncate, and OpenTruncate.
*)

(*:Discussion: *)

BeginPackage["PolyhedronOperations`"]

If[!ValueQ[Geodesate::usage], Geodesate::usage = "\!\(\*RowBox[{\"Geodesate\", \"[\", RowBox[{StyleBox[\"expr\", \"TI\"], \",\", StyleBox[\"n\", \"TI\"]}], \"]\"}]\) replaces each polygon in graphics expression \!\(\*StyleBox[\"expr\", \"TI\"]\) by the projection onto the circumscribed sphere of the order \!\(\*StyleBox[\"n\", \"TI\"]\) regular tessellation of that polygon.\n\!\(\*RowBox[{\"Geodesate\", \"[\", RowBox[{StyleBox[\"expr\", \"TI\"], \",\", StyleBox[\"n\", \"TI\"], \",\", RowBox[{\"{\", RowBox[{StyleBox[\"x\", \"TI\"], \",\", StyleBox[\"y\", \"TI\"], \",\", StyleBox[\"z\", \"TI\"]}], \"}\"}], \",\", StyleBox[\"radius\", \"TI\"]}], \"]\"}]\) does the projection onto the sphere of radius \!\(\*StyleBox[\"radius\", \"TI\"]\) centered at \!\(\*RowBox[{\"{\", RowBox[{StyleBox[\"x\", \"TI\"], \",\", StyleBox[\"y\", \"TI\"], \",\", StyleBox[\"z\", \"TI\"]}], \"}\"}]\)."];
If[!ValueQ[OpenTruncate::usage], OpenTruncate::usage = "\!\(\*RowBox[{\"OpenTruncate\", \"[\", StyleBox[\"expr\", \"TI\"], \"]\"}]\) truncates each edge of each polygon in graphics expression \!\(\*StyleBox[\"expr\", \"TI\"]\) without filling in with a polygon.\n\!\(\*RowBox[{\"OpenTruncate\", \"[\", RowBox[{StyleBox[\"expr\", \"TI\"], \",\", StyleBox[\"ratio\", \"TI\"]}], \"]\"}]\) truncates to the specified \!\(\*StyleBox[\"ratio\", \"TI\"]\) of the edge length."];
If[!ValueQ[Stellate::usage], Stellate::usage = "\!\(\*RowBox[{\"Stellate\", \"[\", StyleBox[\"expr\", \"TI\"], \"]\"}]\) replaces each polygon in graphics expression \!\(\*StyleBox[\"expr\", \"TI\"]\) by a pyramid with the polygon as its base.\n\!\(\*RowBox[{\"Stellate\", \"[\", RowBox[{StyleBox[\"expr\", \"TI\"], \",\", StyleBox[\"ratio\", \"TI\"]}], \"]\"}]\) uses a stellation ratio \!\(\*StyleBox[\"ratio\", \"TI\"]\)."];
If[!ValueQ[Truncate::usage], Truncate::usage = "\!\(\*RowBox[{\"Truncate\", \"[\", StyleBox[\"expr\", \"TI\"], \"]\"}]\) truncates each edge of each polygon in graphics expression \!\(\*StyleBox[\"expr\", \"TI\"]\).\n\!\(\*RowBox[{\"Truncate\", \"[\", RowBox[{StyleBox[\"expr\", \"TI\"], \",\", StyleBox[\"ratio\", \"TI\"]}], \"]\"}]\) truncates to the specified \!\(\*StyleBox[\"ratio\", \"TI\"]\) of the edge length."];


{Stellate,Geodesate,Truncate,OpenTruncate}

Begin["`Private`"]


(* Stellate *)

Stellate[gfx_, k_:2] := 
	Flatten[ gfx /. {
		g_GraphicsComplex :> StellateFace[g, k],
		Polygon[x_] :> StellateFace[x, k] 
		}
	]/; NumberQ[N[k]]

StellateFace[face_List, k_] :=
	Block[ { apex,  n = Length[face], i } ,
		apex = N [ k Apply[Plus, face] / n ] ;
		Table[ Polygon[ {apex, face[[i]], face[[ Mod[i, n] + 1 ]] }],
		    {i, n} ]
	]

StellateFace[GraphicsComplex[ipts_, iprims_, opts:OptionsPattern[]], k_:2] :=
	Block[{pts = ipts, prims = iprims, index, count = 0, newpts},
		index[pt_] := index[pt] = (newpts[++count] = pt;count);
		index /@ pts;
		prims = prims /. { 
			Polygon[a:{{__Integer}..}] :> Polygon[Flatten[stellatePolygon[pts[[#]], #, k]& /@ a, 1]],
			Polygon[a:{__Integer}] :> Polygon[stellatePolygon[pts[[a]], a, k]]
			};
		GraphicsComplex[newpts /@ Range[count], prims]
	
	]

stellatePolygon[facepts_List, faceind_List, k_] :=
	Block[ { apex,  n = Length[facepts], i , new} ,
		apex = k Mean[facepts] ;
		new = index[apex];		
		Table[ {new, faceind[[i]], faceind[[ Mod[i, n] + 1 ]] }, {i, n} ]
	]



(* Geodesate *) 	

$DefaultGeodesateN = 2;
	
Geodesate[poly_, n_, center_List:{0,0,0}, radius_:1] :=
	(Message[Geodesate::tessv, n, $DefaultGeodesateN];
	Geodesate[poly, $DefaultGeodesateN, center, radius]) /; !(IntegerQ[n] && Positive[n])
	
Geodesate[gfx_, n_:$DefaultGeodesateN, center_List:{0,0,0}, radius_:1] := 
	Block[{},
		gfx /. {
			g_GraphicsComplex :> GeodesateFace[g, n, center, radius],
			Polygon[{a_, mid___, b_}/;TrueQ[a==b]] :> GeodesateFace[{a, mid}, n, center, radius],
			Polygon[x_] :> GeodesateFace[x, n, center, radius]
		}
	] /; NumberQ[N[radius]]



(* triangular polygons*)
GeodesateFace[face:{_,_,_}, n_, center_, size_] := Block[{i, j, vtab},
    
        vtab = Table[size Normalize[{n-i-j,i,j}.face/n -center]+center, 
			{i, 0, n}, {j, 0, n-i}];
        Flatten[Table[Polygon[{vtab[[i,j]], vtab[[i+1,j]], vtab[[i,j+1]]}],
                {i, n}, {j, n+1-i}]] ~Join~
        Flatten[Table[Polygon[{vtab[[i,j]], vtab[[i-1,j]], vtab[[i,j-1]]}],
                {i, 2, n+1}, {j, 2, n+2-i}]]
        ]

(* rectangular polygons *)
GeodesateFace[face:{a_, b_, c_, d_}, n_, center_, size_] :=
    Block[{mid = (Plus @@ face)/4, ord = If[n > 1, n - 1, 1]},
        Map[GeodesateFace[#, ord, center, size]&,
            {{a, b, mid}, {b, c, mid}, {c, d, mid}, {d, a, mid}}
        ]
    ]

(* 5-sided and higher polygons *)		
GeodesateFace[face_List, n_, center_, size_] := 
	Block[{tripoint = (Plus @@ face)/Length[face]},

	GeodesateFace[#,n,center,size]& /@ Map[ Join[#,{tripoint}]&,
		Join[ Partition[face,2,1], {{First[face],Last[face]}} ] ]
	]

GeodesateFace[GraphicsComplex[ipts_, iprims_], n_, center_, radius_] := 
	Block[{pts = ipts, prims = iprims, newpts, index, count = 0},
		index[pt_] := index[pt] = (newpts[++count] = pt;count);
		index /@ pts;
		index[center];
		prims = prims /. { 
			Polygon[a:{{__Integer}..}] :> 
				Polygon[Flatten[geodesatePolygon[pts[[#]], #, n, center, radius]& /@ a, 1]],
			Polygon[a:{__Integer}] :> 
				Polygon[geodesatePolygon[pts[[a]], a, n, center, radius]]
			};
		GraphicsComplex[newpts /@ Range[count], prims]
	
	]

geodesatePolygon[points_/;Length[points] == 3, indices_, n_, center_, radius_] := 
	Block[{vtab},
		vtab = Table[radius Normalize[{n-i-j,i,j}.points/n -center]+center, 
			{i, 0, n}, {j, 0, n-i}];
		vtab = Map[index, vtab, {2}];
        Flatten[Table[
			{vtab[[i,j]], vtab[[i+1,j]], vtab[[i,j+1]]}, {i, n}, {j, n+1-i}],1] ~Join~
        Flatten[Table[
        	{vtab[[i,j]], vtab[[i-1,j]], vtab[[i,j-1]]}, {i, 2, n+1}, {j, 2, n+2-i}],1]			
	]

geodesatePolygon[points_, indices_, n_, center_, radius_] := 
	Block[{plist, ilist, midc, midi},
		midc = Mean[points];
		midi = index[midc];

		plist = {##, midc} & @@@ Partition[points, 2, 1, 1];
		ilist = {##, midi} & @@@ Partition[indices, 2, 1, 1];

		Flatten[MapThread[
			geodesatePolygon[## ,n,center,radius]&,
			{plist, ilist}
			], 1]

	]
	
Geodesate::tessv =
"Warning: `1` must be a positive integer.  The default Geodesate tessellation of `2` will be used."

(* Truncate *)

$DefaultTruncateRatio = 3/10;


$TruncList = {}
$TruncHalfList = {}
	
truncList[face_, 0.5] :=  (0.5 (Plus @@ #))& /@ 
      Join[ Partition[face,2,1], {{Last[face],First[face]}} ]
			
truncList[face_, ratio_] := Flatten[{ratio(#[[2]]-#[[1]])+#[[1]], 
	(1-ratio)(#[[2]]-#[[1]])+#[[1]]}& /@ 
		Join[ Partition[face,2,1], {{Last[face],First[face]}} ] ,1];
		 
OpenTruncateFace[face_List,ratio_] := Polygon[ truncList[ face,ratio] ]

OpenTruncate[poly_, ratio_?(#==0&)] := poly

OpenTruncate[poly_, ratio_:$DefaultTruncateRatio] := 
	Flatten[ poly /. {
		g_GraphicsComplex :> TruncateFace[g, ratio, True],
		Polygon[x_] :> OpenTruncateFace[x, N[ratio]]} ] /; 
		NumberQ[N[ratio]]
		
TruncateFace[face_List, 0.5] :=  Block[{array, array2},
	array = truncList[face, 0.49]; 
	AppendTo[ $TruncList, Partition[RotateLeft[array],2]];
	array2 = Flatten[(0.5 {Plus @@ #,Plus @@ #})& /@ Partition[array,2],1];
	AppendTo[ $TruncHalfList, Partition[RotateLeft[array2],2]];
	Polygon[array]]
	
TruncateFace[face_List, ratio_] :=  Block[{array},
	array = truncList[face, ratio]; 
	AppendTo[ $TruncList, Partition[RotateLeft[array],2]];
	Polygon[array]]


combineSame[ x_List] := Block[{set},
	 set = If[ Norm[x[[2]]-x[[1]]] > Norm[x[[3]]-x[[2]]],
	      RotateLeft[x], x];
	 Polygon[(0.5  Plus @@ #)& /@ Partition[set,2]] ]

Truncate[poly_, ratio_?(# == 0 &)] := poly

Truncate[ipoly_, ratio_:$DefaultTruncateRatio] := Block[{pol, res, reslist = {}, nratio = N[ratio],
	search, begin, val, $TruncList, $TruncHalfList, polygon},
	
	poly = ipoly /. {g_GraphicsComplex :> 
		(TruncateFace[g, ratio, False] /. Polygon -> polygon)};

	$TruncList = {};
	$TruncHalfList = {};
	pol = Flatten[ poly /. Polygon[x_] :> TruncateFace[x, nratio] ];

	
	If[Length[res] > 0,
		
		res = Round[$TruncList*100000000]/100000000.;
		res = Sort[ Flatten[MapIndexed[{#1,#2}&,res,{3}],2], 
				OrderedQ[{#1[[1]], #2[[1]]}]& ];
		res = Join[{res[[{2,1,2}]]}, Partition[res,3,1], 
				{res[[Length[res]-{1,0,1}]]}];
		res = Select[res, (#[[2,1]] == #[[1,1]] || #[[2,1]] == #[[3,1]])&];
		res =  Partition[Transpose[Transpose[res][[2]]][[2]] ,2];
		While[res =!= {},
			search = {First[res][[2]]} /. {a_,b_,c_} -> {a,b,If[c==2,1,2]};
			begin = search;
			res = Rest[res];
			While[(val = Select[res, MemberQ[#,search[[1]]]&, 1]) =!= {},
				search = Complement[val[[1]],search] /. 
					{a_,b_,c_} -> {a,b,If[c==2,1,2]};
				begin = Join[begin,search];
				res = DeleteCases[res,val[[1]]];
				];
			AppendTo[reslist,begin];
			];
		If[ nratio === 0.5,
		    reslist = Apply[($TruncHalfList[[##]])&, reslist,{2}];
			  pol = pol /. Polygon[x_] :> combineSame[x],
		    reslist = Apply[($TruncList[[##]])&, reslist,{2}] ];
		If[ Head[pol] === Graphics3D,
		     Join[Graphics3D[Join[First[pol], Polygon /@ reslist]], Rest[pol]],
			 Join[pol, Polygon /@ reslist] ]
	    ,
		pol]  /. {polygon -> Polygon}
    ] /; NumberQ[N[ratio]]

TruncateFace[GraphicsComplex[ipts_, iprims_, opts___], ratio_, open_] := 
	Block[{index, newpts, count=0, pts=ipts, prims=iprims, faces},
		index[pt_] := index[pt] = (newpts[++count] = pt;count);
		index /@ pts;
		prims = prims /. { 
			Polygon[a:{{__Integer}..}] :> 
				Polygon[truncatePolygon[pts[[#]], #, ratio]& /@ a],
			Polygon[a:{__Integer}] :> 
				Polygon[truncatePolygon[pts[[a]], a, ratio]]
			};
		If[!open, 
			faces = Join[
				Flatten[Cases[{iprims}, Polygon[a:{{__Integer}..}] :> a, Infinity],1],
				Cases[{iprims}, Polygon[a:{__Integer}] :> a, Infinity]
			];
			prims = Append[{prims}, Polygon[newfaces[ipts, faces, ratio]]]
			];
		GraphicsComplex[newpts /@ Range[count], prims]
		
	]	

truncatePolygon[ptc_, pti_, ratio_] := Block[{new},
	new = Partition[ptc, 2, 1, 1];
	If[ratio == 0.5, 
		new = Mean /@ new,
		new = Flatten[{ratio #2 + (1-ratio) #1, ratio #1 + (1-ratio) #2}& @@@ new, 1]
	];
	index /@ new
	]

newfaces[points_, faces_, ratio_] :=
    Block[ {edges, polys, corners},
        corners = Flatten[Partition[#, 3, 1, 1] & /@ faces, 1];
        polys = {#, Cases[corners, {a_, #, b_} :> a -> b, Infinity]} & /@ 
          Range[Length[points]];
		polys = DeleteCases[polys, {_,{}}];
        polys = organize /@ polys;
        Apply[Function[{center,face},
          Map[index[(1 - ratio) points[[center]] + ratio points[[#1]]] &, face]
          ], polys,1]
    ]

organize[{i_,pairs_}] :=
    {i,Reverse@NestList[# /. pairs &, First[First[pairs]], Length[pairs] - 1]}

End[]   (* Graphics`Polyhedra`Private` *)

EndPackage[]   (* Graphics`Polyhedra` *)

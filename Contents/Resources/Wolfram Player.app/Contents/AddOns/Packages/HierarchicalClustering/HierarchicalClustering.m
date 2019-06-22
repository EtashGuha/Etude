(* :Copyright: Copyright 2004-2007, Wolfram Research, Inc.  *)

(* :Name: Hierarchical Clustering *)

(* :Title: ClusterAnalysis *)

(* :Author: Darren Glosemeyer *)

(* :Package Version: 1.0 *)

(* :Mathematica Version: 6.0 *)

(*:Summary:
	Provides functionality for cluster creation, plotting,
	and verification. This package contains the functions 
	from the Statistics`ClusterAnalysis` standard add-on 
	that were not added to the kernel in version 6.0.
*)

(* :History:
	Created April 2004 by Andrew A. de Laix
	Modified per design changes September 2004 by Darren Glosemeyer
	Moved to new Hierarchical Clustering package, August 2006 by Darren Glosemeyer
*)

BeginPackage["HierarchicalClustering`"]; 


If[ Not@ValueQ[Agglomerate::usage],
Agglomerate::usage = "Agglomerate[list] collects the elements of list \
into a hierarchy of clusters."]

If[ Not@ValueQ[Cluster::usage],
Cluster::usage = "Cluster[c1,c2,d,n1,n2] represents a merger in the \
cluster hierarchy where the elements c1 and c2 are the subclusters \
merged with distance or dissimilarity value d and the subclusters contain n1 and \
n2 data elements respectively."]

If[ Not@ValueQ[DirectAgglomerate::usage],
DirectAgglomerate::usage = "DirectAgglomerate[m] constructs a cluster \
hierarchy based on the square distance or dissimilarity matrix m.  \
DirectAgglomerate[m,list] associates the elements of list with the rows \
of the distance or dissimilarity matrix m inserting them into the cluster \
hierarchy."]

If[ Not@ValueQ[DendrogramPlot::usage],
DendrogramPlot::usage = "DendrogramPlot[{y1, y2, ... }] plots the \
clustering dendrogram derived from the list of input values. \
DendrogramPlot[c] plots the clustering dendrogram for the Cluster object c."]

If[ Not@ValueQ[LeafLabels::usage],
LeafLabels::usage="LeafLabels is an option to DendrogramPlot that specifies labels for the dendogram leaves."]

If[ Not@ValueQ[TruncateDendrogram::usage],
TruncateDendrogram::usage="TruncateDendrogram is an option to DendrogramPlot that specifies the fusion \
levels at which to truncate the dendrogram.  With the default setting of All, all levels of the \
dendrogram are shown."]

If[ Not@ValueQ[HighlightLevel::usage],
HighlightLevel::usage="HighlightLevel is an option to DendrogramPlot that specifies the level at which \
to highlight the dendrogram."]

If[ Not@ValueQ[HighlightStyle::usage],
HighlightStyle::usage="HighlightStyle is n option to DendrogramPlot that specifies the style for \
highlighted clusters."]

If[ Not@ValueQ[Orientation::usage],
Orientation::usage="Orientation is an option to DendrogramPlot that specifies the orientation of the \
dendrogram. Possible settings are Top, Bottom, Left, and Right."]

If[ Not@ValueQ[Linkage::usage],
Linkage::usage="Linkage is an option to Agglomerate and DendrogramPlot that specifies the linkage \
method to be used for agglomerative clustering."]

If[ Not@ValueQ[DistanceMatrix::usage],
DistanceMatrix::usage = "DistanceMatrix[list] computes the \
symmetric matrix of distance or dissimilarity coefficients between the elements of list."] 

If[ Not@ValueQ[ClusterFlatten::usage],
ClusterFlatten::usage = "ClusterFlatten[c] flattens the cluster c \
returning a list of the original input data."]

If[ Not@ValueQ[ClusterSplit::usage],
ClusterSplit::usage = "ClusterSplit[c,n] splits the cluster c into n \
clusters by repeatedly dividing at the largest remaining fusion level."]

Agglomerate::bdlink = "Value of option Linkage->`1` is not \
\"Single\", \"Complete\", \"Average\", \"WeightedAverage\", \"Centroid\", \"Median\", \
\"Ward\" or a pure function.";

Agglomerate::amtd = "Agglomerate is unable to automatically select an \
appropriate distance or dissimilarity function for the input data `1`."

Agglomerate::xnum = "A nonnumeric, negative or complex distance or dissimilarity \
value was computed; distances and dissimilarities must be nonnegative and real valued.";

Agglomerate::ties = "`1` ties have been detected; reordering input may \
produce a different result.";

Agglomerate::nelist = "Argument `1` at position `2` is not a nonempty list.";

DirectAgglomerate::bdlink = "Value of option Linkage->`1` is not \
\"Single\", \"Complete\", \"Average\", \"WeightedAverage\", \"Centroid\", \"Median\", \
\"Ward\" or a pure function.";

DirectAgglomerate::ties = "`1` ties have been detected; reordering \
input may produce a different result.";

DirectAgglomerate::dcbd = "The input distance or dissimilarity matrix `1` contains \
non-numeric, negative, or complex values; the distance or dissimilarity matrix must \
be non-negative and real valued.";

DirectAgglomerate::xnum = "A nonnumeric, negative or complex \
distance or dissimilarity value was computed; distances and dissimilarities must be nonnegative and \
real valued.";

DirectAgglomerate::bdat = "The length of the data list `1` at position \
2 does not match the dimensions of the distance or dissimilarity matrix `2` at \
position 1.";

ClusterSplit::dep = 
   "Cannot split a cluster of `1` elements into `2` clusters."; 
   
ClusterSplit::notcl = ClusterFlatten::notcl = "`1` is not a valid cluster.";

DendrogramPlot::labl = "The number of labels (`1`) does not match the \
number of data points (`2`)."; 

DendrogramPlot::labo = "`1` is not a valid LeafLabels option."; 

DendrogramPlot::labm = "Cannot map the labels in `1` to the data \
elements of the cluster `2`; the mapping is ambiguous."; 

DendrogramPlot::trnc = "`1` is not a valid truncation range."; 

DendrogramPlot::hltr = 
   "The highlight value `1` should be between 1 and `2`."; 

DendrogramPlot::hlt = "`1` is not a valid highlight specification."; 

DendrogramPlot::orn = "The orientation value `1` should be either Top, \
Bottom, Left, or Right."; 

DendrogramPlot::arg1 =" `1` is not a list of elements, a Cluster object, \
or a rule mapping data elements to labels.";

DendrogramPlot::cldist = "Distance information is contained within the input Cluster object. \
The DistanceFunction option will be ignored.";

DistanceMatrix::amtd = "DistanceMatrix is unable to automatically select an appropriate dissimilarity function for the input data `1`.";

DistanceMatrix::xnum = "A nonnumeric, negative or complex distance or dissimilarity \
value was computed; distances and dissimilarities must be nonnegative and real valued.";

Begin["`Private`"]; 

ClusterSplit[args___] := Module[{res = iClusterSplit[args]}, 
    res /; res =!= $Failed]; 

(*
   iClusterSplit
   
   Description
   	seprarates a Cluster expression into multiple clusters by
   	repeatedly splitting at the largest fusion value
   	
   Arguments
   	clust_ : a cluster object
   	n_ : the number of clusterst to be produced
   	
   Returns
   	a list of cluster objects
*)

iClusterSplit[clust_Cluster, n_] := Module[{mx, res},
    If[Length[Apply[List,clust]] =!= 5
    	, 
    	Message[ClusterSplit::notcl, clust]; Return[$Failed]
    	,
      	mx = clust[[4]] + clust[[5]];
      	Which[! IntegerQ[n] || n <= 0,
        	Message[ClusterSplit::intpm, HoldForm[ClusterSplit[clust, n]], 2]; 
        	Return[$Failed],
        	n > mx,
        	Message[ClusterSplit::dep, mx, n]; Return[$Failed],
        	True,
        	Null];
      	Catch[Nest[(res = With[{p = Replace[#1, cl_ :> Switch[cl, _Cluster, If[Length[Apply[List, cl]] =!= 5, 
      			Message[ClusterSplit::notcl, clust]; $Failed, cl[[3]]], _, -1.], {1}]}, 
      			If[! FreeQ[p, $Failed], $Failed, 
      		  	MapAt[Sequence @@ Take[#1, 2] &, #1, Ordering[p, -1]]]]; 
      		  	If[res === $Failed, Throw[$Failed], res]) &
      		  , {clust}, n - 1]]]]
      		  

iClusterSplit[args___] := If[Length[{args}] != 2, 
   Message[ClusterSplit::argrx, ClusterSplit, Length[{args}], 2]; 
    Return[$Failed], Return[$Failed]]


ClusterFlatten[args___] := Module[{res = iClusterFlatten[args]}, 
   res /; res =!= $Failed]

(*
   iClusterFlatten[cl_Cluster]
   
   Description
   	Flattens out a Cluster expression returning the data
   	elements in the cluster
   	
   Arguments
   	cl_ : a Cluster expression
   
   Returns
   	a list of data elements
*)

iClusterFlatten[cl_Cluster] := Module[{foo, res},
   If[FreeQ[res = cl //. Cluster[aa_, bb_, cc_, dd_, ee_] -> foo[aa, bb], Cluster]
   	,
      	List @@ Flatten[res, Infinity, foo]
      	,
      	Message[ClusterFlatten::notcl, cl]; $Failed]]

iClusterFlatten[args___] := If[Length[{args}] != 1, 
   Message[ClusterFlatten::argx, ClusterFlatten, Length[{args}]]; 
    Return[$Failed], Return[$Failed]]
    

(* ProcessOptionNames allows for option names to be strings or symbols *)    
SetAttributes[ProcessOptionNames, Listable];

ProcessOptionNames[(r : (Rule | RuleDelayed))[name_Symbol, val_]] :=
    r[SymbolName[name], val];

ProcessOptionNames[opt_] := opt;


Options[DendrogramPlot] = {DistanceFunction->Automatic, LeafLabels -> None, 
    TruncateDendrogram -> All, HighlightLevel -> None, 
    HighlightStyle -> Automatic, Orientation -> Top, 
    PlotStyle -> Automatic, Linkage -> Automatic};  

DendrogramPlot[args___] := Module[{res = iDendrogramPlot[args]}, 
   res /; res =!= $Failed]

(*
   iDendrogramPlot (v1)
   	This is the main argument parsing function for plotting
   	dendrograms with data (as compared to Cluster expression) input.
   	
   Arguments
   	data_List : a list of data elements
   	opts___ : options for this function
   	
   Returns
   	a Graphics object
*)

(*Handle a list of rules {e1->v1, e2->v2,...}*)
iDendrogramPlot[v:{__Rule}, (opts___)?OptionQ]:= iDendrogramPlot[v[[All,1]]->v[[All,2]], opts]
	
iDendrogramPlot[data:(_List|_Rule), (opts___)?OptionQ]:= 
  Block[{i = 0, $SowLabel, $lotruncate, $hitruncate, 
    	$hi, $Orientation, $LabelFunction = First}
    	, 
   	Module[{c, lab, mydata, link, distfun, processedoptions}
     	  , 
     	  (* use ProcessOptionNames to convert all option names to strings;
     	     also include opts so Graphics options are included *)
     	  processedoptions = Join[ProcessOptionNames[Flatten[{opts, Options[DendrogramPlot]}]],{opts}];
     	  lab = "LeafLabels"/.processedoptions;

    	  Switch[lab
    	    ,
    	    None | False
    	    ,
    	    mydata = data; $SowLabel = False;
    	    ,
    	    _Function
    	    ,
    	    $LabelFunction = lab;
    	    mydata = data;
    	    $SowLabel = True;
    	    ,
    	    Automatic | _List
    	    ,

    	    If[Head[data]===List, 
    	    	If[lab === Automatic,lab = Range[Length[data]]];
    	    	If[Length[lab] != Length[data]
       		  		, 
       		  		Message[DendrogramPlot::labl, Length[lab], Length[data]]; 
       		  		$SowLabel = False; 
         	  		mydata = data; 
         	  		, 
         	  		$SowLabel = True; 
         	  		If[False && ArrayDepth[data] < 2&&Head[data[[1]]]=!=String (*False fixes bug 218745*)
         				,
         				(*Why is this Partition needed?  It lead to bug 218745*)
          				mydata = Partition[data, 1]
          				, 
          				mydata = data
          				
          			]; 
    
         	  	mydata = MapThread[Rule[#1, #2] &, {mydata, lab}];
         	  	$LabelFunction = (#&);
         	  	]];
    	    If[Head[data]===Rule, 
    	    	If[lab===Automatic, lab = Range[Length[data[[2]]]]];
    	    	If[Length[lab] != Length[data[[2]]]
		  , 
		  Message[DendrogramPlot::labl, Length[lab], Length[data]];
		  $SowLabel = False;
		  mydata = data;
		  , 
		  $SowLabel = True;
		  mydata = MapThread[Rule[#1, #2] &, {data[[1]], lab}];
		  $LabelFunction = (#&);
         	  ]];
           ,
           _
           ,
           Message[DendrogramPlot::labo, lab];
           mydata = data; $SowLabel = False; ];
         link = "Linkage" /. processedoptions;
         distfun = "DistanceFunction" /. processedoptions;

         Check[c = Agglomerate[mydata, "Linkage" -> link, "DistanceFunction" -> distfun]
     		, 
     		Return[$Failed]
     		,  
      		Agglomerate::bmtd, Agglomerate::amtd, Agglomerate::xnum];
      	 iDendrogramFromCluster[c, processedoptions]
      ]]
   
   

(*
   iDendrogramPlot (v2)
   	This is the main argument parsing function for plotting
   	dendrograms with a Cluster expression as input.
   	
   Arguments
   	c_Cluster : a Cluster expression
   	opts___ : options for this function
   	
   Returns
   	a Graphics object
*)

iDendrogramPlot[c_Cluster, (opts___)?OptionQ] := 
  Block[{i = 0, $SowLabel, $lotruncate, $hitruncate, $hi, $Orientation, 
    $LabelFunction}
    , 
    Module[{lab, distfun, 
    	processedoptions}
    	,
    	(* use ProcessOptionNames to convert all option names to strings;
     	     also include opts so Graphics options are included *)
    	processedoptions = Join[ProcessOptionNames[Flatten[{opts, Options[DendrogramPlot]}]],{opts}];
    	(* check against opts so a message is only issued for user-defined "DistanceFunction" *)
    	If[(distfun="DistanceFunction"/.ProcessOptionNames[{opts}])=!="DistanceFunction"
    		,
    		Message[DendrogramPlot::cldist, distfun]];
    	lab = "LeafLabels" /. processedoptions; 
     	Switch[lab
     		, 
     		None | False
     		, 
     		$SowLabel = False
     		, 
     		Automatic
     		, 
      		$SowLabel = True; $LabelFunction = Short; 
      		, 
      		_Function
      		, 
      		$SowLabel = True; $LabelFunction = lab; 
      		, 
      		_List
      		, 
      		Message[DendrogramPlot::labm, lab, c]; $SowLabel = False
      		, 
      		_
      		, 
      		Message[DendrogramPlot::labo, lab]; $SowLabel = False; 
      		]; 
    
     	iDendrogramFromCluster[c, processedoptions]
     ]]

(*
   iDendrogramFromCluster
   
   Description
   	This builds up the dendrogram graphics from a Cluster expression.
   
   Arguments
   	c_ : a Cluster expression
   	opts___ : options for this function
*)

iDendrogramFromCluster[c_, opts___] := 
  Module[{lines, lablist, dcs, trunc, hilite, rects, txtfxn, hs, ps}
  	, 

   	dcs = Reverse[Sort[Cases[{c}, cc_Cluster :> cc[[3]], Infinity]]]; 
    	trunc = "TruncateDendrogram" /. opts; 
    	Switch[trunc
    	  , 
    	  All | Infinity
    	  , 
    	  {$lotruncate, $hitruncate} = {-Infinity, Infinity}
    	  , 
    	  n_Integer /; Inequality[0, Less, n, LessEqual, Length[dcs]]
    	  , 
    	  {$lotruncate, $hitruncate} = {dcs[[trunc]], First[dcs]}
    	  , 
    	  {n_Integer, m_Integer} /; m > n && n > 0 && m < Length[dcs]
    	  , 
    	  {$lotruncate, $hitruncate} = {dcs[[Last[trunc]]], dcs[[First[trunc]]]}
    	  , 
    	  {n_Integer /; Inequality[0, Less, n, LessEqual, Length[dcs]], Infinity}
    	  , 
    	  {$lotruncate, $hitruncate} = {-Infinity, dcs[[First[trunc]]]}
    	  , 
    	  _
    	  , 
    	  Message[DendrogramPlot::trnc, trunc]; Return[$Failed]
    	  ];
    	hilite = "HighlightLevel" /. opts;
    	$hi = Switch[hilite
    	  , 
    	  False | None
    	  , 
    	  -Infinity
    	  , 
    	  n_Integer /; Inequality[0, Less, n, LessEqual, Length[dcs]]
    	  , 
    	  dcs[[hilite]]
    	  , 
    	  _Integer
    	  , 
    	  Message[DendrogramPlot::hltr, hilite, Length[dcs]]; 
    	  Return[$Failed]
    	  , 
    	  _
    	  , 
    	  Message[DendrogramPlot::hlt, hilite]; 
    	  Return[$Failed]
    	  ]; 
    	$Orientation = "Orientation" /. opts; 
    	txtfxn = Switch[$Orientation
    	  , 
    	  Left
    	  , 
    	  Text[#2, Offset[{4, 0}, {0, #1}], {-1, 0}] & 
    	  , 
    	  Right
    	  , 
    	  Text[#2, Offset[{-4, 0}, {0, #1}], {1, 0}] & 
    	  , 
    	  Bottom
    	  , 
    	  Text[#2, Offset[{0, 4}, {#1, 0}], {0, -1}] & 
    	  , 
    	  Top
    	  , 
    	  Text[#2, Offset[{0, -4}, {#1, 0}], {0, 1}] & 
    	  , 
    	  _
    	  , 
    	  Message[DendrogramPlot::orn, $Orientation]; 
    	  Return[$Failed]
    	  ];
    	{lines, lablist, rects} = Last[Reap[iGenerateDendrogram[c], 
       		{"lines", "labels", "rectangles"}]]; 
       	ps = "PlotStyle" /. opts; 
       	hs = "HighlightStyle" /. opts; 
       	Show[
       	  Graphics[{
       	    	{Switch[hs,Automatic,RGBColor[0, 1, 0],_List,Sequence @@ hs,_,hs], rects}
       	    	, 
       	    	{Switch[ps,Automatic,{},_List,Sequence @@ ps,_,ps], lines}
       	    	, 
       	    	If[$SowLabel, Apply[txtfxn, First[lablist], {1}], {}]}
      	   , 
      	   FilterRules[Join[opts, {PlotRange -> All, AspectRatio -> 1/GoldenRatio}],
                 	Options[Graphics]]]
     	]]

iDendrogramPlot[arg_,___?OptionQ] := (Message[DendrogramPlot::arg1, arg]; 
   $Failed)/;!MemberQ[{List,Cluster,Rule}, Head[arg]]

iDendrogramPlot[args___] := (Message[DendrogramPlot::argx, 
    DendrogramPlot, Length[Select[{args},  !OptionQ[#1] & ]]]; 
   $Failed)
   
(*
   iGenerateDendrogram
   
   Description
   	This function calls itself recursively sowing the graphics elements
   	that make up the dendrogram until the recursion terminates on a
   	leaf (a data element)
*)
   

iGenerateDendrogram[v_] := 
  (i++; If[$SowLabel, Sow[{i, $LabelFunction[v]}, "labels"]; i, i, i])

iGenerateDendrogram[c_Cluster] := Block[
	{x0, xl, xr, yl, yr, $RecursionLimit=Infinity}
	, 
   	If[c[[3]] > $hitruncate
   	  , 
   	  iGenerateDendrogram[c[[1]]]; 
   	  iGenerateDendrogram[c[[2]]]; 
   	  Return[]]; 
	If[c[[3]] <= $lotruncate
	  , 
	  i++; 
	  Return[If[$SowLabel
	  	, 
        	Sow[{i, DisplayForm[FrameBox[ToBoxes[c[[-2]] + c[[-1]]]]]},"labels"]; 
        	i
        	,
        	i, i]]
          ];
    	xl = -If[c[[-2]] == 1 || c[[1,3]] <= $lotruncate, 0, c[[1,3]]]; 
    	xr = -If[c[[-1]] == 1 || c[[2,3]] <= $lotruncate, 0, c[[2,3]]]; 
    	x0 = -c[[-3]]; 
    	yl = iGenerateDendrogram[c[[1]]]; 
    	yr = iGenerateDendrogram[c[[2]]]; 
    	Switch[$Orientation
    		, 
    		Left
    		, 
     		Sow[Line[{{xl, yl}, {x0, yl}, {x0, yr}, {xr, yr}}], "lines"]; 
      		If[c[[3]] <= $hi
      		  , 
      		  Sow[Rectangle[{x0, yl}, {0, yr}], "rectangles"]
      		  ]
      		, 
      		Right
      		, 
     		Sow[Line[{{-xl, yl}, {-x0, yl}, {-x0, yr}, {-xr, yr}}], "lines"]; 
      		If[c[[3]] <= $hi
      		  , 
      		  Sow[Rectangle[{0, yl}, {-x0, yr}], "rectangles"]
      		  ]
        	, 
        	Bottom
        	, 
     		Sow[Line[{{yl, xl}, {yl, x0}, {yr, x0}, {yr, xr}}], "lines"]; 
      		If[c[[3]] <= $hi
      		  , 
      		  Sow[Rectangle[{yl, x0}, {yr, 0}], "rectangles"]
      		  ]
        	, 
        	Top
        	, 
     		Sow[Line[{{yl, -xl}, {yl, -x0}, {yr, -x0}, {yr, -xr}}], "lines"]; 
      		If[c[[3]] <= $hi
      		  , 
      		  Sow[Rectangle[{yl, 0}, {yr, -x0}], "rectangles"]
      		  ]
      		]; 
      	(yl + yr)/2.]

End[]; 

EndPackage[]; 

(* :Context: AuthorTools`DiffReport` *)

(* :Author: Dale R. Horton *)

(* :Summary:
    This package defines low-level list differencing functions.
*)

(* :Copyright: *)

(* :Package Version: $Revision: 1.10 $, $Date: 2017/03/08 16:27:33 $ *)

(* :Mathematica Version: 5.0 *)

(* :History:
    
*)

(* :Keywords:
     
*)

(* :Discussion:
    
*)

(* :Warning:
    
*)


BeginPackage["AuthorTools`DiffReport`", "AuthorTools`Common`"]


(* Usage messages *)

DiffReport::usage = "DiffReport[list1, list2] returns a set of rules that indicate the differences between list1 and list2."

Linear::usage = "Linear is a value for the Method option of DiffReport and NotebookDiff. With Method->Linear, a linear difference routine finds the differences."

ShowDiffProgress::usage = "ShowDiffProgress is an option of DiffReport and NotebookDiff that specifies whether to show incremental progress."

(* Error messages *)

General::same = "You must compare two different objects but both are `1`."



Begin["`Private`"]

(*DIFF ENGINE*)

(*LCS*)
(*J. W. Hunt, T. G. Szymanski. A Fast Algorithm for Computing Longest Common
Subsequences, Communications of the ACM, 20:5, 350-353 (May 1977).*)
LCS[lis1_, lis2_, opts___?OptionQ]:=
Block[{n, (*a, b,*) k, matchlist, thresh, link, f, count, countmax},
  f = ShowDiffProgress/.{opts}/.Options[DiffReport];
  If[f===True, f = $DiffReportTrackingFunction];
  If[f===False, f = Identity];
  n = Max[Length@lis1,Length@lis2];
  matchlist = Reverse[Flatten[Position[lis2,#,{1}]]]&/@lis1;
  thresh[_] = n+1;
  link[0] = {};
  count=1;
  countmax=Length@Flatten@matchlist;
  MapIndexed[BuildLinks, matchlist, {2}];
  k = n; While[thresh[k]===n+1&&k>0, k--];
  If[k===0, {}, Transpose@link[k]]
]

$DiffReportTrackingFunction[{1, total_}] :=
    Print[$Resource["Diff", "Starting"]]
    
$DiffReportTrackingFunction[{count_, total_}] :=
  If[Mod[count, Ceiling[total/10]] === 0,
    Print[$Resource["Diff", "Processing"], " ", Floor[100*count/total], "%"]
  ]
 
$DiffReportTrackingFunction[{total_, total_}] :=
    Print[$Resource["Diff", "Finsihed"]]

BuildLinks[j_, {i_, jj_}] :=
Module[{k},
  k=1; While[j>thresh[k], k++];
  thresh[k]=j;
  link[k]=Join[link[k-1],{{i,j}}];
(* apply ShowDiffProgress *)
  f[{count++, countmax}]; 
]


(* Experimentally tie into a built-in LongestCommonSubsequence function *)


LCS[lis1_,lis2_,opts___?OptionQ] :=
  StringPattern`LongestCommonSubsequence[lis1, lis2] /; (TrueQ[$ExperimentalLCS])

$ExperimentalLCS = True;


        
(*DiffReport*)
Options[DiffReport] = {Method -> Linear, ShowDiffProgress -> False}

(* Identical *)
DiffReport[lis1_, lis1_, opts___?OptionQ] := 
  {"Delete" -> {}, "Insert" -> {}, "Update" -> {}}

(* One list empty *)
DiffReport[lis1_, {}, opts___?OptionQ] := 
  {"Delete" -> {{Range@Length@lis1, 1, lis1}}, "Insert" -> {}, "Update" -> {}}

DiffReport[{}, lis2_, opts___?OptionQ] := 
  {"Delete" -> {}, "Insert" -> {{1 , Range@Length@lis2, lis2}}, "Update" -> {}}

DiffReport[lis1_, lis2_, opts___?OptionQ] := 
  Module[{meth},
    meth = Method /. {opts} /. Options[DiffReport];
    diff[meth][lis1, lis2, opts]
  ]



(* get the symbols in the right context *)
NotebookDiff = AuthorTools`NotebookDiff`NotebookDiff;
IgnoreCellStyleDiffs = AuthorTools`NotebookDiff`IgnoreCellStyleDiffs;
IgnoreOptionDiffs = AuthorTools`NotebookDiff`IgnoreOptionDiffs;
IgnoreContentStructure = AuthorTools`NotebookDiff`IgnoreContentStructure;


preprocessCells[lis1:{__Cell}, lis2:{__Cell}, opts___] := 
Block[{cells1 = lis1, cells2 = lis2, igsty, igopt, igstruct},
  {igsty, igopt, igstruct} =
    {IgnoreCellStyleDiffs, IgnoreOptionDiffs, IgnoreContentStructure} /.
    Flatten[{opts, Options[NotebookDiff]}];

  (* discard cell options *)
  igopt = Switch[igopt,
    All, All,
    None, {ImageCache, ImageRangeCache},
    _List, Join[igopt, {ImageCache, ImageRangeCache}],
    _, {igopt, ImageCache, ImageRangeCache}
  ];
  (* use ReplaceRepeated to remove options from inline cells and top-level cells *)
  {cells1, cells2} = If[igopt===All,
    {cells1, cells2} //. Cell[a__, _Rule|_RuleDelayed] :> Cell[a],
    {cells1, cells2} //. Cell[a__, (Rule|RuleDelayed)[Alternatives@@igopt,_], b___] :> Cell[a,b]
  ];
  
  (* discard styles. *)
  If[TrueQ@igsty,
    cells1=cells1 /. Cell[cont_, sty_, o___] :> Cell[cont, o]; 
    cells2=cells2 /. Cell[cont_, sty_, o___] :> Cell[cont, o]
  ];
  
  (* discard all structure and compare only content strings *)
  If[TrueQ@igstruct,
    cells1=cells1 /. Cell[x_, y___] :> Cell[Cases[{x},_String,Infinity]//StringJoin,y];
    cells2=cells2 /. Cell[x_, y___] :> Cell[Cases[{x},_String,Infinity]//StringJoin,y];
  ];
  
  {cells1, cells2}
]

preprocessCells[lis1_, lis2_, opts___] := {lis1, lis2}



(* linear diff *)
diff[Linear][lis1_, lis2_, opts___?OptionQ] := 
  Module[{cells1, cells2, l1, l2, lcs, d1, d2, extra1, extra2, nonmatching},    
    
    {cells1, cells2} = preprocessCells[lis1, lis2, opts];

    lcs = LCS[cells1, cells2, opts];
    
    If[lcs === {}, (* everything is an update *)
      Return[{"Delete" -> {}, "Insert" -> {}, 
          "Update" -> {{Range@Length@lis1, Range@Length@lis2, List @@ lis1, 
                List @@ lis2}}}]];
    {l1, l2} = lcs;
    l1 = Join[{0}, l1, {Length[lis1] + 1}];
    l2 = Join[{0}, l2, {Length[lis2] + 1}];
    extra1 = extra2 = nonmatching = {};
    Do[
      d1 = l1[[i + 1]] - l1[[i]]; 
      d2 = l2[[i + 1]] - l2[[i]];
      Which[
      (* "Delete" *)
      d2 === 1 && d1 > 1, 
        AppendTo[
          extra1, {Range[l1[[i]] + 1, l1[[i + 1]] - 1], l2[[i + 1]], 
            Take[lis1, {l1[[i]] + 1, l1[[i + 1]] - 1}]}],
      (* "Insert" *)
      d1 === 1 && d2 > 1, 
        AppendTo[
          extra2, {l1[[i + 1]], Range[l2[[i]] + 1, l2[[i + 1]] - 1], 
            Take[lis2, {l2[[i]] + 1, l2[[i + 1]] - 1}]}], 
      (* "Update" *)
      d1 > 1 && d2 > 1,
        AppendTo[
          nonmatching, {Range[l1[[i]] + 1, l1[[i + 1]] - 1], 
            Range[l2[[i]] + 1, l2[[i + 1]] - 1], 
            Take[lis1, {l1[[i]] + 1, l1[[i + 1]] - 1}], 
            Take[lis2, {l2[[i]] + 1, l2[[i + 1]] - 1}]}]
      ], {i, Length[l1] - 1}
    ];
    {"Delete" -> extra1, "Insert" -> extra2, "Update" -> nonmatching}
  ]
    
            
End[]

EndPackage[]

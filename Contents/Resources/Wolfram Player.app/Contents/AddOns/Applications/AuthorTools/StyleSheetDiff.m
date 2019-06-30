(* :Context: AuthorTools`StyleSheetDiff` *)

(* :Author: Dale R. Horton *)

(* :Summary:
    Package to compare style sheets.
*)

(* :Copyright: *)

(* :Package Version: $Revision: 1.4 $, $Date: 2003/03/27 17:23:40 $ *)

(* :Mathematica Version: 5.0 *)

(* :History:
    
*)

(* :Keywords:
     
*)

(* :Discussion:
    
*)

(* :Warning:
    
*)



BeginPackage["AuthorTools`StyleSheetDiff`",
  {"AuthorTools`DiffReport`", "AuthorTools`Common`"}]


(* Usage messages *)

StyleSheetDiff::usage = "StyleSheetDiff[nb1, nb2] creates a report of the differences between the style sheets nb1 and nb2."
  
(* Error messages *)

StyleSheetDiff::nbtype = "The file `1` is not a notebook."

StyleSheetDiff::nbval = "The value `1` for the notebook is not a string or a notebookobject."

(* Other messages *)

StyleSheetDiff::diff = "`1` that differ";

StyleSheetDiff::only = "`1` only in `2`";



Begin["`Private`"]
    
(* cannot diff the same nb *)
StyleSheetDiff[nb1_, nb1_] := (Message[StyleSheetDiff::same, nb1]; $Failed)
    
StyleSheetDiff[nb1_, nb2_] := 
    Module[{NB1, NB2, f1, f2},
      (* test nbs *)
      {{NB1, f1}, {NB2, f2}} = Map[testnb, {nb1, nb2}];
      If[NB1===$Failed||NB2===$Failed, Return[$Failed]];
      ssdiff[NB1, NB2, f1, f2]
    ]

(* test nbs *)
testnb[nb_] :=
Switch[nb,
        _String, Which[FileType@nb===None,
                    Message[StyleSheetDiff::noopen, nb]; {$Failed, $Failed},
                        !StringMatchQ[nb, "*.nb"],
                    Message[StyleSheetDiff::nbtype, nb]; {$Failed, $Failed},
                        True,
                    {Get@nb, nb}
                ],
        _NotebookObject, {NotebookGet@nb, FullOptions[nb, WindowTitle]},
        _Notebook, {nb, $Resource["Diff", "No name"]},
        _, Message[StyleSheetDiff::nbval, nb]; {$Failed, $Failed}
]

(* core function *)
ssdiff[nb1_, nb2_, f1_, f2_] :=
  Module[{sty1, sty2, env1, env2, onlyin1, onlyin2, env, up, ins, del, sty, 
      num, begin},
    begin = {Cell[$Resource["Diff", "Style title"], "Title", ShowCellBracket->False], 
        Cell[BoxData[
            GridBox[{{$Resource["Diff", "New"], $Resource["Diff", "Old"]},
                  {f1, f2}}, 
          ColumnWidths -> {.5}, GridFrame->True, ColumnAlignments -> Center, 
          RowLines->True, ColumnLines->True]], "Text", ShowCellBracket->False]};
    {env1, sty1} = stylelist@nb1;
    {env2, sty2} = stylelist@nb2;
    (* strip out environments that only occur in one ss *)
    onlyin1 = Complement[env1, env2];
    onlyin2 = Complement[env2, env1];
    env = {onlyCell[onlyin1, $Resource["Diff", "New"], $Resource["Diff", "Environments"]], 
        onlyCell[onlyin2, $Resource["Diff", "Old"], $Resource["Diff", "Environments"] ]};
    sty1 = DeleteCases[sty1, 
        Cell[StyleData[_, env_ /; MemberQ[onlyin1, env]], ___]];
    sty2 = DeleteCases[sty2, 
        Cell[StyleData[_, env_ /; MemberQ[onlyin2, env]], ___]];
    (* strip out styles that only occur in one ss *)
    onlyin1 = Complement[First /@ sty1, First /@ sty2];
    onlyin2 = Complement[First /@ sty2, First /@ sty1];
    del = Cases[sty1, Cell[sd_ /; MemberQ[onlyin1, sd], ___] :> sd];
    del = onlyCell[del, $Resource["Diff", "New"], $Resource["Diff", "Styles"]];
    ins = Cases[sty2, Cell[sd_ /; MemberQ[onlyin2, sd], ___] :> sd];
    ins = onlyCell[ins, $Resource["Diff", "Old"], $Resource["Diff", "Styles"]];
    sty1 = DeleteCases[sty1, Cell[sd_ /; MemberQ[onlyin1, sd], ___]];
    sty2 = DeleteCases[sty2, Cell[sd_ /; MemberQ[onlyin2, sd], ___]];
    (* strip out common environments *)
    num = Length@Intersection[env1, env2];
    sty = Transpose@{sty1, sty2};
    envdiff = diffCell[Take[sty, num], $Resource["Diff", "Environments"]];
    sty = Drop[sty, num];
    (* strip out nb options *)
    nbdiff = diffCell[{First@sty}, $Resource["Diff", "Notebook Options"]];
    sty = Rest@sty;
    (* diff remaining styles *)
    up = diffCell[sty, $Resource["Diff", "Styles"]];
    Notebook[Join[begin, env, {envdiff, nbdiff, del, ins, up}], 
        GridBoxOptions -> {ColumnAlignments -> Left},
        WindowTitle->$Resource["Diff", "Style title"]
    ]
  ]
    



(* cellgroup of differences *)    
diffCell[stys_?MatrixQ, type_String] := 
  Module[{diffs = diffCell /@ stys},
    If[diffs === {}, Sequence @@ {}, 
      Cell[CellGroupData[
          Prepend[diffs, Cell[ToString @ StringForm[StyleSheetDiff::diff, type], "Section"]], 
          Closed]]
    ]
  ]
    
(* compare two cells *)    
diffCell[{cell1_, cell2_}] := 
  If[cell1 === cell2, Sequence @@ {}, styDiff[cell1, cell2], "not"]

styDiff[Cell[sty_, opt1___], Cell[sty_, opt2___]] := 
  Module[{cellopts, cell},
    cellopts = sortopts /@ {{opt1}, {opt2}}; 
    cellopts = Map[ToString[#, StandardForm] &, cellopts, {2}];
    cell = 
      Cell[BoxData@
          GridBox[Transpose[elementDiff @@ cellopts], ColumnWidths -> .5], 
        "Text"];
    If[sty === StyleData["Notebook"], cell,
      Cell[
        CellGroupData[{Cell[First@styname@sty, "Subsection"], cell}, 
          Closed]]
      ]
    ]

(* sort opts so that there can be no "move" operations *)
sortopts[opts_List] := Sort[opts, (OrderedQ[First /@ {#1, #2}]) &]

(* find differences in options *)
elementDiff[el1_, el2_] := 
  Module[{del, ins, up, mark1, mark2, len}, 
    {del, ins, up} = {"Delete", "Insert", "Update"} /. DiffReport[el1, el2];
    new1 = If[el1 === {}, {},
        mark1 = Flatten@{Map[First, del], Map[First, up]};
        Extract[el1, List /@ mark1]];
    new2 = If[el2 === {}, {},
        mark2 = Flatten@{Map[#[[2]] &, ins], Map[#[[2]] &, up]};
        Extract[el2, List /@ mark2]];
    len = Max[Length /@ {new1, new2}];
    Map[PadRight[#, len, ""] &, {new1, new2}]
  ]


(* list of occurrences in only 1 ss *)
onlyCell[{}, __] := Sequence[]

onlyCell[stys_List, nb_, type_String] := 
  Cell[CellGroupData[{
    Cell[ToString @ StringForm[StyleSheetDiff::only, type, nb], "Section"], 
    onlyCell@stys}, Closed]]

onlyCell[sty_List] := Cell[BoxData@FormBox[GridBox[styname /@ sty], TextForm], "Text"]

(* printing style names *)
styname[StyleData[All, env_]] := {env}

styname[StyleData[sty_, env_]] := {sty <> "/" <> env}

styname[StyleData[sty_]] := {sty}

styname[env_String] := {env}

(* generate a list of environments and styles *)
stylelist[nb1_] :=
  Module[{sty},
    sty = Cases[nb1, Cell[_StyleData, ___], Infinity];
    env = Cases[sty, Cell[StyleData[All, env_], ___]];
    env = Sort[env, sortstys];
    env = Map[#[[1, 2]] &, env];
    sty = Sort[sty, sortstys];
    (* remove duplicate styles *)
    sty = sty//.{x___, c1:Cell[sty_,___], c2:Cell[sty_,___], y___} ->
      {x, c1, y};
    {env, sty}
  ]

(*
Sort the styles and environments of a style. Working environment floats to top.
Then the other environments. Then the Notebook style. Then the rest of the styles.
*)
sortstys[Cell[StyleData[sty1_, env1___], ___], 
    Cell[StyleData[sty2_, env2___], ___]] :=
  Which[{env1} === {"Working"} || sty1 === All || sty1 === "Notebook", True,
    {env2} === {"Working"} || sty2 === All || sty2 === "Notebook", False,
    sty1 === sty2, OrderedQ[{{env1}, {env2}}],
    True, OrderedQ[{sty1, sty2}]
  ]
    
End[]

EndPackage[]
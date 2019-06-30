

Begin["`Private`"]
(* XMLTransform functions *)
P
DIV
Ol
Li
H1
H2
H3
H4
TD
Span
A
UL
Br
Img
Image
IfCommentSSI


$BoxTypes = {
  ActionMenuBox -> {"Children" -> {}},
  AdjustmentBox -> {},
  AnimatorBox -> {"Children" -> {}},
  ArrowBox -> {"Children" -> {}},
  BoxData -> {},
  ButtonBox -> {"Styles" -> (Cases[#, _[BaseStyle, s_] :> s]&)},
  Cell -> {
    "Styles" -> (Cases[Rest[#], _String] &), 
    "NodeID" -> True},
  CellBoundingBox -> {"Children" -> {}},
  CellElementsBoundingBox -> {"Children" -> {}},
  CellGroupData -> {"Styles" -> (Cases[Rest[#[[1, 1]]], _String] &)},
  CheckboxBox -> {"Children" -> {}},
  CircleBox -> {"Children" -> {}},
  ColorSetterBox -> {"Children" -> {}},
  ContentsBoundingBox -> {"Children" -> {}},
  CounterBox -> {"Children" -> {}},
  CuboidBox -> {"Children" -> {}},
  CylinderBox -> {"Children" -> {}},
  DiskBox -> {"Children" -> {}},
  DynamicBox -> {"Children" -> {}},
  DynamicModuleBox -> {"Children" -> {}},
  DynamicWrapperBox -> {"Children" -> {}},
  ErrorBox -> {},
  FormBox -> {"Styles" -> ({#[[2]]}&)},
  FractionBox -> {"Children" -> {{1}, {2}}},
  FrameBox -> {},
  GeometricTransformationbox -> {"Children" -> {}},
  Graphics3DBox -> {"Children" -> {}},
  GraphicsBox -> {"Children" -> {}},
  GraphicsComplex3DBox -> {"Children" -> {}},
  GraphicsComplexBox -> {"Children" -> {}},
  GraphicsData -> {"Children" -> {}},
  GraphicsGridBox -> {"Children" -> {}},
  GraphicsGroupBox -> {"Children" -> {}},
  GridBox -> {},
  GridColumn -> {"Children" -> All},
  GridRow -> {"Children" -> All},
  InputFieldBox -> {"Children" -> {}},
  Inset3DBox -> {"Children" -> {}},
  InsetBox -> {"Children" -> {}},
  InterpretationBox -> {},
  ItemBox -> {"Children" -> {}},
  Line3DBox -> {"Children" -> {}},
  LineBox -> {"Children" -> {}},
  List -> {"Children" -> All},
  LocatorBox -> {"Children" -> {}},
  LocatorPaneBox -> {"Children" -> {}},
  Notebook -> {"NodeID" -> True},
  OpenerBox -> {"Children" -> {}},
  OptionValueBox -> {"Children" -> {}},
  OverscriptBox -> {"Children" -> {{1}, {2}}},
  PaneBox -> {"Children" -> {}},
  PanelBox -> {"Children" -> {}},
  PaneSelectorBox -> {"Children" -> {}},
  Point3DBox -> {"Children" -> {}},
  PointBox -> {"Children" -> {}},
  Polygon3DBox -> {"Children" -> {}},
  PolygonBox -> {"Children" -> {}},
  PopupMenuBox -> {"Children" -> {}},
  ProgressIndicatorBox -> {"Children" -> {}},
  RadicalBox -> {"Children" -> {{1}, {2}}},
  RadioButtonBox -> {"Children" -> {}},
  RasterBox -> {"Children" -> {}},
  RawData -> {},
  RectangleBox -> {"Children" -> {}},
  RotationBox -> {"Children" -> {}},
  RowBox -> {},
  SetterBox -> {"Children" -> {}},
  ShrinkWrapBoundingBox -> {"Children" -> {}},
  Slider2DBox -> {"Children" -> {}},
  SliderBox -> {"Children" -> {}},
  SphereBox -> {"Children" -> {}},
  SqrtBox -> {},
  StyleBox -> {
    "Styles" -> (Cases[Rest[#], _String] &), 
    "NodeID" -> True},
  SubscriptBox -> {"Children" -> {{1}, {2}}},
  SubsuperscriptBox -> {"Children" -> {{1}, {2}, {3}}},
  SuperscriptBox -> {"Children" -> {{1}, {2}}},
  TabViewBox -> {"Children" -> {}},
  TagBox -> {},
  TemplateBox -> {"Children" -> {}},
  Text3DBox -> {"Children" -> {}},
  TextBoundingBox -> {"Children" -> {}},
  TextBox -> {"Children" -> {}},
  TextData -> {},
  TogglerBox -> {"Children" -> {}},
  TooltipBox -> {"Children" -> {{1}, {2}}},
  UnderoverscriptBox -> {"Children" -> {{1}, {2}, {3}}},
  UnderscriptBox -> {"Children" -> {{1}, {2}}},
  ValueBox -> {}
};

(* munge the context path to add some useful symbols.  Make sure
   System`ConvertersDump` is in the path by calling Import; *)
Import;

(* some Debug helper shortcuts *)
`trace = ResourceShingleTransmogrify`Debug`Private`trace
`traceOpts = ResourceShingleTransmogrify`Debug`Private`traceOpts

`bag = Internal`Bag
`stuffBag = Internal`StuffBag
`bagLength = Internal`BagLength
`bagPart = Internal`BagPart

(* --------------------- 
         s e t u p       
   --------------------- *)

(* setup the XMLTransform search path.  Searched from first to last element *)
(* a la GUIKit, find any installed package with an XMLTransforms subdir *)  

$XMLTransformPath = {$ResourceShingleTransmogrifyDirectory}

$XMLTransforms := Select[ FileNames["*.m", $XMLTransformPath, Infinity], XMLTransformQ[#]& ]

$DefaultXMLTransforms := Select[ FileNames["*.m", ToFileName[{$ResourceShingleTransmogrifyDirectory}, "XMLTransforms"], Infinity], XMLTransformQ[#]& ]

$ExportImageFormat = "GIF"

(* decide what package import/export format to use *)
$PackageFormat = "Package";


XMLTransformQ[{__String},_String]=True
XMLTransformQ[expr_]:= MatchQ[expr,
  (
    _String|
    {__String}|
    {{__String},_String}
  )]
XMLTransformQ[___]=False

TransformQ[_?XMLTransformQ|_XMLTransform]=True
TransformQ[___]=False

Options[ResourceShingleTransmogrify] = {
    AbortOnError -> True,
    AtomsReturnSelf -> True,
    AutoRecurse -> True,
    CreateImages -> True,
    ExportOptions -> {"AttributeQuoting"->"\""},
    ExportFormat -> "XML",
    (*IgnoreBoxes -> {},*)
    InputDirDepth -> 1,
    MaxExportErrors -> Infinity,
    OutputExtension -> "html"
}

(** Messages **)
ResourceShingleTransmogrify::childindex="SelectChildren[`1`] doesn't exist for `2`"
ResourceShingleTransmogrify::cantselect = "Can't select the `1` of node `2`.";
ResourceShingleTransmogrify::ambiguous = "The box `1` has an ambiguous position.";
Off[ResourceShingleTransmogrify::ambiguous];
ResourceShingleTransmogrify::noboxinfo = "No information about box type `1`.";
ResourceShingleTransmogrify::noparam = "Could not resolve the parameter `1`.";
ResourceShingleTransmogrify::nofileop = "When called without a filename argument, ResourceShingleTransmogrify\
 cannot use the operation `1`.";
ResourceShingleTransmogrify::badxml = "Export as XML failed due to bad markup.  Please fix your transforms.\
 The generated output was saved as $XMLOutput, with error positions as $XMLErrors for your\
 debugging pleasure."
ResourceShingleTransmogrify::baddir = "The output file specified is in a directory that couldn't be\
 written to."



(** GetChildPositions **)
GetChildPositions[expr_] := Module[{boxinfo, cpos},
  boxinfo = Head[expr] /. $BoxTypes;
  If[boxinfo === Head[expr],
    Message[ResourceShingleTransmogrify::noboxinfo, ToString[Head[expr]]];
    cpos = All,
    cpos = "Children" /. boxinfo /. "Children" -> {{1}}
    ];
  Which[
    cpos === All,
    cpos = DeleteCases[Position[expr, arg_ /; FreeQ[{Rule, RuleDelayed}, Head[arg]], 1], {0}],
    cpos === {{1}} && Head[First[expr]] === List,
    cpos = {1, #} & /@ Range[Length[First[expr]]]
  ];
  cpos
];


(** ExtractOne **)
ExtractOne[expr_, pos_] := If[Length[pos] == 0, expr, Extract[expr, pos]];


(** GetNodeID **)
GetNodeID[_String] := None;
GetNodeID[expr_] := If[MatchQ[expr, _[___, "NodeID" -> _, ___]],
  First[Cases[expr, ("NodeID" -> id_) :> id]],
  expr];


(** GetNodeSourcePosition *)
GetNodeSourcePosition[expr_, opts___?OptionQ]  := Module[{pos, parents, position, ppos},
  pos = $NodePosition[GetNodeID[expr]];
  If[ListQ[pos], Return[{$Notebook, pos}]];

  parents = Parents /. {opts} /. {Parents -> $Parents};
  position = Position /. {opts} /. {Position -> $Position};
  If[Length[parents] == 0,
    Return[{expr, {{}}}] ];

  ppos = GetNodeSourcePosition[First[parents], Parents -> Rest[parents], Position -> Rest[position]];
  {First[ppos], Append[Last[ppos], First[position]]}
];


(** GetNodePosition **)
GetNodePosition[expr_, opts___?OptionQ] :=
  Module[{spos = GetNodeSourcePosition[expr, opts]},
    If[Head[First[spos]] === Notebook, Last[spos], None]];


(** ParseNotebook **)
ParseNotebook[nb_] :=
  Block[{$NodeIDCounter = 0, $RecursionLimit = 1024}, ParseExpr[nb, {{}}]];


(** ParseExpr **)
ParseExpr[s_String, ___] := s;
ParseExpr[s_Integer, ___] := s;
ParseExpr[expr_, pos_] := Module[{boxinfo, cpos, children, out, nid},
  boxinfo = Head[expr] /. $BoxTypes;
  If[boxinfo === Head[expr],
    Message[ResourceShingleTransmogrify::noboxinfo, ToString[Head[expr]]];
    cpos = All,
    cpos = "Children" /. boxinfo /. "Children" -> {{1}}
    ];
  Which[
    cpos === All,
      cpos = DeleteCases[Position[expr, arg_ /; FreeQ[{Rule, RuleDelayed}, Head[arg]], 1], {0}],
    cpos === {{1}} && Head[First[expr]] === List,
      cpos = {1, #} & /@ Range[Length[First[expr]]]
  ];
  If[Length[cpos] > 0,
    children = Map[ParseExpr[Extract[expr, #], Append[pos, #]] &, cpos];
    out = ReplacePart[expr, MapThread[(#1 -> #2) &, {cpos, children}]],
    out = expr
  ];
  If[("NodeID" /. boxinfo) === True && !MatchQ[expr, Cell[_CellGroupData, ___]],
    out = Append[out, "NodeID" -> ++$NodeIDCounter]];

  nid = GetNodeID[out];
  Switch[$NodePosition[nid],
    None,
      $NodePosition[nid] = pos,
    "Ambiguous",
      None,
    _,
      Message[ResourceShingleTransmogrify::ambiguous, out]; $NodePosition[nid] = "Ambiguous"
  ];
    
  out
];

ResourceShingleTransmogrify[SelectChildren[args___], opts___?OptionQ] :=
Module[{OldPosition = $Position, OldParents, NewParents, NewParentsSet = False},
  OldPosition = $Position;
  OldParents = $Parents;
  Sequence @@ Map[
    Function[{pos}, Module[{expr = ExtractOne[$Self, pos]},
      If[MatchQ[expr, _[___, "NodeID" -> _, ___]],
        Block[{$Parents = {}, $Position = {}},
          MakeItSo[expr, opts]
        ],
        Block[{$Parents, $Position},
          If[!NewParentsSet, NewParents = Prepend[OldParents, $Self]; NewParentsSet = True];
          $Parents = NewParents;
          $Position = Prepend[OldPosition, pos];
          MakeItSo[expr, opts]
        ]
    ]]],
    SelectPositions[SelectChildren, args]
  ]
] /; TrueQ[$Walking];

ResourceShingleTransmogrify[expr_, opts___?OptionQ] :=
Block[{$Parents = {}, $Position = {}},
  MakeItSo[Evaluate[expr], opts]
] /; TrueQ[$Walking];

ResourceShingleTransmogrify[expr_ /; Head[expr]=!=String, transform_?TransformQ, opts___?OptionQ] :=
	ResourceShingleTransmogrify[None, expr, XMLTransformInit[transform,opts], opts]

ResourceShingleTransmogrify[exprfile_String, transform_?TransformQ, opts___?OptionQ] := 
	ResourceShingleTransmogrify[None, Import[exprfile, "Notebook"], XMLTransformInit[transform,opts], opts]

ResourceShingleTransmogrify[output_String, expr_/;!StringQ[expr], transformFile_?XMLTransformQ, opts___?OptionQ]:=
    ResourceShingleTransmogrify[output, expr, XMLTransformInit[transformFile, opts], opts]

ResourceShingleTransmogrify[filename_String, exprfile_String, transform_?TransformQ, opts___?OptionQ]:= 
    ResourceShingleTransmogrify[filename, Import[exprfile, "Notebook"],
      XMLTransformInit[transform,opts],opts
    ]/;FileType[exprfile]===File

ResourceShingleTransmogrify[outputdir_String, inputdir_String, transform_?TransformQ, opts___?OptionQ]:= 
Module[
  {nbs, lev},
  lev = InputDirDepth /. {opts} /. Options[ResourceShingleTransmogrify];

  If[!ListQ[
    nbs=FileNames["*.nb",inputdir, lev]],Return[$Failed]];
  
  If[nbs === {},
    Message[ResourceShingleTransmogrify::badinput,inputdir];
    Return[$Failed]
    ,
    ResourceShingleTransmogrify[outputdir, nbs, XMLTransformInit[transform, opts], opts ]
  ]
]/;FileType[inputdir]===Directory

ResourceShingleTransmogrify::badinput = "`1` is not a valid input filename or directory."

ResourceShingleTransmogrify[outputdir_String, infiles:{__String}, transform_?XMLTransformQ, opts___?OptionQ]:= 
    ResourceShingleTransmogrify[outputdir, infiles, XMLTransformInit[transformFile,opts],opts]

ResourceShingleTransmogrify[_String,in_String,___]:=(Message[ResourceShingleTransmogrify::badinput,in];$Failed)
ResourceShingleTransmogrify[_,$Failed,__]=$Failed


ResourceShingleTransmogrify[outdir_String, infiles:{__String}, trans:(_XMLTransform|$CachedTransform), opts___?OptionQ]:=
Module[
  {transform,ext},
  ext = OutputExtension /. {opts} /. Options[ResourceShingleTransmogrify];
  If[!StringQ[ext],
    Message[ResourceShingleTransmogrify::outext,ext];
    ext = ""
  ];
  (* parse transform once *)
  ResourceShingleTransmogrifyInit[trans, opts];
  (* then map ResourceShingleTransmogrify over everything *)
  ResourceShingleTransmogrify[ToFileName[{outdir}, StringReplace[#,
    {
      StartOfString~~WordCharacter__~~$PathnameSeparator->"",
      "."~~WordCharacter__~~EndOfString->"."
    }]<>ext], Get[#],
    $CachedTransform, opts]&/@infiles
]
 

ResourceShingleTransmogrify[outfile:(_String|None),
	expr_ /;!AtomQ[expr], transform:(_XMLTransform|$Failed|$CachedTransform), 
    opts___?OptionQ] :=
Block[
  { 
    output, ret, $Counters,
    $Notebook, $NodePosition, $NotebookContext,
    $Ancestors=#, $Walking=True,
    $OutputFile = outfile,
    format, exportopts, maxerrs, dir, 
    saveProfile, profileFile
  },

  AppendTo[Attributes[ResourceShingleTransmogrify], HoldFirst];

  GetNodeFunctionInit;
  Clear[$NodePosition];
  $NodePosition[_] := None;
  $Notebook = ParseNotebook[expr];
  $NotebookContext = ("c" <> ToString[Hash[$Notebook]] <> "`");
  
  If[ StringQ[$OutputFile], $OutputFile = StringReplace[$OutputFile,"~"->$HomeDirectory] ];

  (* get the directory to which we're going to save files *)
  dir = If[outfile === None || StringFreeQ[ outfile, RegularExpression["^(?:/|[A-Z]:\\\\)"]],
    Directory[],
    ResourceShingleTransmogrify`Private`createDirectory[DirectoryName[outfile]]
  ];

  (* die if we can't write to a new directory *)
  If[dir===$Failed,
    Message[ResourceShingleTransmogrify::baddir,dir];
    Return[$Failed],
    (* otherwise change directories *)
    SetDirectory[dir];
  ];

  (* first initialize the ResourceShingleTransmogrify environment *)
  ResourceShingleTransmogrifyInit[transform,opts];

  (* now set the user's DefaultParameters, which will override
     anything set in the Transform *)
  If[ Length[params=DefaultParameters /. {opts}]>0,
    Scan[ ($Parameters[#[[1]]] = {#[[2]]})&, params ] ];

  (* get some user opts *)
  {ResourceShingleTransmogrify`Internal`$AutoRecurse, atomval, format, exportopts,maxerrs} = 
    {AutoRecurse, AtomsReturnSelf, ExportFormat, ExportOptions, MaxExportErrors} /. 
      {opts} /. Options[ResourceShingleTransmogrify];

  (* set up return value for atomic recursion.  don't use Sequence[]! *)
  If[TrueQ[atomval],
    ResourceShingleTransmogrify`Internal`$AtomReturnValue := $Self,
    ResourceShingleTransmogrify`Internal`$AtomReturnValue = Null
  ];

  (* do the job *)
  output = ResourceShingleTransmogrify[$Notebook];
  
  (* remove Nulls.  Evaluate[] to remove single Sequence[] *)
  output = Evaluate[Sequence @@ DeleteCases[{output},Null,Infinity]];

  (* remove any previous definitions to $XML(Output|Errors) just to save space *)
  Clear[$XMLOutput,$XMLErrors];

  (* set up return value *)
  ret = Which[
    (* return SymbolicXML *)
    !StringQ[outfile], output,
    (* export Text for stuff that's already Strings *)
    StringQ[output] || format==="Text",
      Export[ outfile, output, "Text", Sequence @@ exportopts ],
    (* save SymbolicXML to file if valid *)
    format === "XML",
      (* BUG #61406 *)
      Check[ $XMLErrors = XML`SymbolicXMLErrors[output], 
        If[$XMLErrors === {}, $XMLErrors = {{0}}] ];

      If[ Length@$XMLErrors > maxerrs,
        Message[ResourceShingleTransmogrify::badxml]; $XMLOutput=output; $Failed,
        Export[ outfile, output, "XML", Sequence @@ exportopts ]
      ],
    (* save all other formats *)
    True,
      Export[ outfile, output, format, Sequence @@ exportopts ]
  ];

  ResetDirectory[];
  Attributes[ResourceShingleTransmogrify] = DeleteCases[Attributes[ResourceShingleTransmogrify], HoldFirst];

  ret
]

defaultRules = {};

ResourceShingleTransmogrifyInit[trans:(_XMLTransform|$Failed|$CachedTransform), opts___?OptionQ]:=
Module[
  {transform=trans, transdefs},

  `trace[TransformParsing, "Setting up ResourceShingleTransmogrify Variables"];

  (* initialize counters *)
  ResetCounters[];
  $Counters[_] = 0;

  Which[$CachedTransform === trans,
    `trace[TransformParsing, "XMLTransform Cached!  Not reparsing."];
    Return[],
    
    Head[transform]=!=XMLTransform,
      Message[ResourceShingleTransmogrify::notrans];
      Abort[]
  ];
  If[$ScreenStyleEnvironment =!= None,
    transform = DeleteCases[transform,
      _[{_, _, env_/;!MatchQ[env,$ScreenStyleEnvironment|All]},_],
	  {2}]
  ];

  $Rules = transform[[1]];
  $Rules[[All,1]] = Release[mkPattern/@$Rules[[All,1]]];

  `trace[TransformParsing,"Parsed "<>ToString@Length[$Rules]<>" rules."];
  `trace[{TransformParsing,2},"List of $Rules: ",$Rules];
 
  (* clear any variables lying around to save memory *)
  Clear[$Parameters,$TagsToCells];
  $Parameters[_] = {};

  If[Length[
    transdefs = Cases[Options[transform], DefaultParameters ~(Rule|RuleDelayed)~
      _]]>0,
    transdefs = Flatten@Reverse@transdefs[[All,2]];
    Scan[ ($Parameters[#[[1]]] = {#[[2]]})&, transdefs ] 
  ];
]
Attributes[parse] = {HoldAllComplete}

SetAttributes[{HeldOptionQ, HeldNonOptionQ}, HoldAllComplete]

HeldOptionQ[expr_] := OptionQ @ Unevaluated[expr]
HeldOptionQ[exprs__] := Thread[Unevaluated @ HeldOptionQ[And[exprs]], And]

HeldNonOptionQ[expr_] := Not @ HeldOptionQ[expr]
HeldNonOptionQ[exprs__] := Thread[Unevaluated @ HeldNonOptionQ[And[exprs]], And]

(* exprs with the standard idea of Cell-type "Styles" *)
parse[ h:(Cell|StyleBox)[stuff_, styles__String, opts___?OptionQ] ] := 
  ($Children = {stuff}; $Styles = {styles}; $Options = {opts};)

(* pseudo-style for CellGroupData[] ... *)
parse[ CellGroupData[stuff:{Cell[_,styles__String,___],___},open___] ]:=
  ($Children = stuff; $Styles = {styles}; $GroupOpen = open)

(* handle CellGroupData[nonstyled-cells].  otherwise Open|Closed might show
   up in $Children *)
parse[ CellGroupData[l_, open___] ] :=
  ($Children = l; $Styles={}; $GroupOpen = open;)

parse[ TagBox[stuff_, tag_, opts___?OptionQ] ] :=
  ($Children = {stuff}; $Styles = {ToString[tag]}; $Options = {opts};)

(* ... and FormBox[] *)
parse[ FormBox[stuff_, style_, opts___?OptionQ] ] :=
  ($Children = {stuff}; $Styles = {style}; $Options = {opts};)

(* all other normal expressions, including things that _should_ have styles
   but don't *)
parse[ h_[stuff_List, opts___?HeldOptionQ] ] := 
  ($Children = stuff; $Styles = {}; $Options = {opts};)

parse[ h_[stuff_, opts___?HeldOptionQ] ] := 
  ($Children = {stuff}; $Styles = {}; $Options = {opts};)

parse[ h_[stuff__, opts___?HeldOptionQ] ] := 
  ($Children = {stuff}; $Styles = {}; $Options = {opts};)

(* TODO - figure out what to do with Hold expressions *)
(* parse[ h_?holdQ[stuff__, opts___?OptionQ] ] := 
    ($Children = {HoldPattern@stuff}; $Styles = {}; $Options= {opts};)
*)

(* anything else that matches should be an Atom *)
parse[_] := ($Children = $Styles = $Options = {};)

(** holdQ[] 
    simple function to find out if a function head has a Hold*
    attribute
**)
holdQ[head_AtomQ] := MemberQ[Attributes[head], 
  HoldFirst|HoldRest|HoldAll|HoldAllComplete]

(** getKids[]
    this is a helper function to recurse through expressions in 
    certain selector functions.  See SelectNearestDescendants[]
    for example.  TODO - obey evaluation rules for held 
    expressions?
**)
getKids[(Cell | StyleBox | FormBox)[stuff_, ___]] := If[ListQ[stuff], stuff, {stuff}];
getKids[CellGroupData[stuff_List,___]] := stuff;
getKids[h_[stuff__, ___?OptionQ]] := {stuff};
getKids[sibs__] := getKids /@ {sibs};
getKids[_] := {};

MakeItSo[lst:{___}, opts___?OptionQ] := Block[{ret, tmp=$Ancestors},
  `trace[Recursion,"mapping over " <> ToString@Length@lst <> " nodes"];
  `trace[{Recursion,2}, "Mapping over ", lst];

  (* Apply Sequence to remove the ExprSequence[] wrapper *)
  ret = Sequence @@ Flatten[Function[e, MakeItSo[e, opts]] /@ lst];
  `trace[{Recursion,2}, "Map returned: ", ret];
  ret
]


MakeItSo[expr_, opts___?OptionQ]:= Block[
  {
    h=Head@expr, out, a, afunc,
    $Self=expr, $Options, $Parent, $Children, $Styles, $Siblings,
    branchAncestors=$Ancestors, $UserWantsSelf=False,
    oldShowStringCharacters=$ShowStringCharacters
  },

If[MatchQ[expr, Cell[___, "Output", ___]], OUTPUT=True, OUTPUT=False];
  {a,$Siblings} = {Ancestor,Sibs} /. {opts} /. {Ancestor->None, Sibs->None};

  Which[
    MatchQ[$Self, _[___, _[ShowStringCharacters, _], ___]],
    $ShowStringCharacters = First@Cases[$Self, _[ShowStringCharacters, ssc_] :> ssc],
    MatchQ[$Self, _Notebook],
    $ShowStringCharacters = True
  ];

  (* first set up the easy environment stuff *)
  parse[expr];

  (* If expr ain't an Atom, set up $Ancestors for this iteration.  Note 
     that this is treated as an anonymous function to make life easier.
     # can be replaced by the current value of $Children when and if 
     SelectAncestors is called.  See the Selector functions below for
     more. *)
  If[!AtomQ[expr],
    afunc=Function[f,f&][If[a===None,$Ancestors,a]];

    $Ancestors = afunc[Which[
      h===CellGroupData, h[#,$GroupOpen],
      h===TagBox, h[#, Last[expr]],
      HasStyle[], h[#, Sequence@@$Styles, $Options],
      True, h[#, $Options]
    ]]
  ];

  (* do the actual replacement.  We're using {0} so as to
     not interfere with other replacements as we're walking. *)
  out = Replace[ expr, $Rules, {0} ];

  If[ ResourceShingleTransmogrify`Internal`$AutoRecurse && 
        out === expr && !deletedQ[out] && !$UserWantsSelf,
    `trace[Recursion, "No matches for "<>ToString@Head@expr<>" expr.  Recursing!"];
    out = ResourceShingleTransmogrify[SelectChildren[]];
    ,
    (* otherwise we matched something; save if we're profiling *)
    `trace[TransformRules, ToString@Head@expr<>" matched rule"]
  ];
  (* reset $Ancestors to last branch point *)
  $Ancestors=branchAncestors;

  $ShowStringCharacters = oldShowStringCharacters;

  out
]
deletedQ = Function[x, MatchQ[Unevaluated@x, HoldPattern[Sequence][]|{}|Null|""], SequenceHold]


mkPattern[String] = _String;
mkPattern[{head_?AtomQ}|head_?AtomQ] := _head

mkPattern[{Cell, CellGroupData}] := Cell[_CellGroupData, ___?OptionQ];
mkPattern[{Cell, None}] := Cell[c_/;(Head[c]=!=CellGroupData), ___?OptionQ]
mkPattern[{head:(StyleBox|FormBox), None} ] := head[_, ___?OptionQ]
mkPattern[{head:(Cell|StyleBox|FormBox), style_}] := (
  head[__, style, ___?StringQ, ___?OptionQ]/.{All->_(*,None->Sequence[]*)})

(* this three arg form will interfere with the Screenenvironment three arg
   form below, but just added it to see if it would be used. *)
mkPattern[{CellGroupData, None, open_:None}] := CellGroupData[{Cell[_,___?OptionQ],___},open/.None->___]
mkPattern[{CellGroupData, open:(Open|Closed)}] := mkPattern[{CellGroupData, All, open}]
mkPattern[{CellGroupData, style_, open_:None}] := (
  CellGroupData[{Cell[_, style,  ___?OptionQ],___},open] /. {All->_,None->___})

(* three argument selectors. TODO - change screen to options? *)
mkPattern[{head:(Cell|StyleBox), style_, screen_}] := 
  Hold[head[_, Sequence @@ style, ___?OptionQ] /; $ScreenEnvironment === screen]

(* FIXME *)
mkPattern[{CellGroupData, style_, screen_}] = _

(* thread over Alternatives *)
x:mkPattern[_Alternatives] := Thread[Unevaluated@x,Alternatives]

(* thread over Alternatives for list-based shortcuts *)
mkPattern[{x:HoldPattern[Alternatives][__?AtomQ]}] := mkPattern[x]
mkPattern[{HoldPattern[Alternatives][a__?AtomQ],style_}] := 
  Alternatives@@(mkPattern[{#,style}]&/@{a})

(* assume that anything else is a valid pattern
   given by the user *)
mkPattern[x_]:= x



(* style helpers *)
GetStyle[] := GetStyle[$Self] /; TrueQ[$Walking];
GetStyle[expr_] := Module[{lst},
  lst = GetStyleList[expr];
  If[!ListQ[lst] || lst === {}, None, First[lst]]
] /; TrueQ[$Walking];

GetStyleList[] := GetStyleList[$Self] /; TrueQ[$Walking];
GetStyleList[expr_] := Module[{boxinfo, func},
  boxinfo = Head[expr] /. $BoxTypes;
  If[boxinfo === Head[expr],
    Message[ResourceShingleTransmogrify::noboxinfo, ToString[Head[expr]]];
    Return[{}]];
  func = "Styles" /. boxinfo;
  If[func === "Styles", Return[{}]];
  If[ListQ[#],#,{#}]&[func[expr]]
] /; TrueQ[$Walking];

HasStyle[] := HasStyle[$Self] /; TrueQ[$Walking];
HasStyle[expr_] := Module[{lst},
  lst = GetStyleList[expr];
  If[ListQ[lst] && Length[lst] > 0, True, False]
] /; TrueQ[$Walking]

(* fast *)
GetOption[x_] := x /. Options[$Self] /. x->None

(* slow *)
GetOption[s__] := (
  (*Message[ResourceShingleTransmogrify::obs,GetOption," See Options[] instead."];*)
  If[MatchQ[$Self,
		_@@Fold[{___, (Rule|RuleDelayed)[#2, #1], ___}&, _, Reverse[Flatten[{s}]]] ],
	Fold[(#2 /. Cases[#1, _~(Rule|RuleDelayed)~_])&, $Self, {s}],
    None]
) /; TrueQ[$Walking]

(* fast *)
HasOption[x_] := MemberQ[$Self, x ~ (Rule|RuleDelayed) ~ _]

(* slow *)
HasOption[s__] := MatchQ[$Self,
  _@@Fold[{___, (Rule|RuleDelayed)[#2, #1], ___}&, _, Reverse[Flatten[{s}]]] 
] /; TrueQ[$Walking]

(* -------------- Selector Functions -------------- *)


(** Recurse **)
Recurse[] := ResourceShingleTransmogrify[SelectChildren[]] /; TrueQ[$Walking]
Recurse[expr_,opts___?OptionQ] := ResourceShingleTransmogrify[expr,opts] /; TrueQ[$Walking]


(** SelectorQ **)
Attributes[SelectorQ] = {HoldFirst};
SelectorQ[sel_] := MemberQ[{SelectChildren}, Head[sel]];


(** SelectSelf **)
SelectSelf[] := ($UserWantsSelf=True;$Self) /; TrueQ[$Walking]

(** SelectLiteral **)
SelectLiteral[x___] := x;

(** SelectChildren **)
SelectChildren[args___] := Extract[$Self, SelectPositions[SelectChildren, args]] /; TrueQ[$Walking];

SelectPositions[SelectChildren] := GetChildPositions[$Self];
SelectPositions[SelectChildren, i_Integer] :=
  Module[{ch},
    ch = GetChildPositions[$Self];
    If[0 < Abs[i] <= Length[ch],
      {ch[[i]]},
      Message[ResourceShingleTransmogrify::childindex,i,k]; {}
  ]];
SelectPositions[SelectChildren, head_, style_] :=
  SelectPositions[SelectChildren, {head, style}];
SelectPositions[SelectChildren, selector_] :=
  Module[{pos},
    pos = GetChildPositions[$Self];
    Cases[Map[{#, ExtractOne[$Self, #]}&, pos], {i_, mkPattern[selector]} :> i]
  ];


SelectSiblings[] := Module[{selfpos, parent, sibpos},
  selfpos = $NodePosition[GetNodeID[$Self]];
  If[MemberQ[{"Ambiguous", None}, selfpos],
    Message[ResourceShingleTransmogrify::cantselect, "siblings", $Self]; Return[{}]];
  If[selfpos === {{}}, Return[{}]];
  parent = ExtractOne[$Notebook, Flatten[Most[selfpos]]];
  sibpos = GetChildPositions[parent];
  sibpos = DeleteCases[sibpos, Last[selfpos]];
  Extract[parent, sibpos]
] /; TrueQ[$Walking];

SelectSiblings[selector_] :=
  Cases[SelectSiblings[], mkPattern[selector]] /; TrueQ[$Walking];

SelectAncestors[] := Module[{selfpos},
  selfpos = GetNodePosition[$Self];
  If[MemberQ[{"Ambiguous", None}, selfpos],
    Message[ResourceShingleTransmogrify::cantselect, "ancestors", $Self]; Return[{}]];
  If[selfpos === {{}}, Return[{}]];
  Extract[$Notebook,
    Map[
      Flatten[Drop[selfpos, -#]]&,
      Range[Length[selfpos] - 1]] ]
] /; TrueQ[$Walking];

SelectAncestorsAndSelf[] := Module[{selfpos},
  selfpos = GetNodePosition[$Self];
  If[MemberQ[{"Ambiguous", None}, selfpos],
    Message[ResourceShingleTransmogrify::cantselect, "ancestors", $Self]; Return[{}]];
  If[selfpos === {{}}, Return[{}]];
  Extract[$Notebook,
    Map[
      Flatten[Drop[selfpos, -#]]&,
      Range[0, Length[selfpos] - 1]] ]
] /; TrueQ[$Walking];

SelectAncestors[selector_] :=
  Cases[SelectAncestors[], mkPattern[selector]] /; TrueQ[$Walking];

SelectAncestorsAndSelf[selector_] :=
  Cases[SelectAncestorsAndSelf[], mkPattern[selector]] /; TrueQ[$Walking];

SelectNearestAncestor[selector_] :=
  If[Length[#] > 0, First[#], None]&[
    Cases[SelectAncestors[], mkPattern[selector], 1, 1]
  ] /; TrueQ[$Walking];

SelectNearestAncestorAndSelf[selector_] :=
  If[Length[#] > 0, First[#], None]&[
    Cases[SelectAncestorsAndSelf[], mkPattern[selector], 1, 1]
  ] /; TrueQ[$Walking];

(** SelectParent[] **)
SelectParent[i_Integer:0] := Module[{selfpos},
  selfpos = GetNodePosition[$Self];
  If[selfpos === {{}}, Return[None]];
  If[MemberQ[{"Ambiguous", None}, selfpos],
    (* Try to use the $Parents Block-scoped variable instead *)
    If[Length[$Parents] > 0,
      If[i >= Length[$Parents],
        Message[SelectParent::badindex,i]; Return[None]];
      Return[Part[$Parents, i]]];
    (* Otherwise there's nothing we can do *)
    Message[ResourceShingleTransmogrify::cantselect, "parent", $Self]; Return[None]];
  If[i > Length[selfpos],
    Message[SelectParent::badindex,i]; Return[None]];

  ExtractOne[$Notebook, Flatten[Drop[selfpos, -i]]]
] /; TrueQ[$Walking];


SelectParent::badindex = "Parent index `1` must be a positive integer."

getDescendants[_String, ___] := {};
getDescendants[expr_, pattern_, nearest_] := Module[{cpos},
  cpos = GetChildPositions[expr];
  Map[
    Function[{pos}, Module[{child, cmatch},
      child = ExtractOne[expr, Flatten[pos]];
      cmatch = MatchQ[child, pattern];
      (* FIXME: put in some recursion optimization on known box types *)
      Apply[Sequence, Join[
        If[cmatch, {child}, {}],
        If[!(cmatch && TrueQ[nearest]),
          getDescendants[child, pattern, nearest],
          {}]
      ]]
    ]],
    cpos]
];

(** SelectDescendants[] **)
SelectDescendants[] :=
  getDescendants[$Self, _, False] /; TrueQ[$Walking];
SelectDescendants[selector_] :=
  getDescendants[$Self, mkPattern[selector], False] /; TrueQ[$Walking];

(** SelectDescendantsAndSelf[] **)
SelectDescendantsAndSelf[] :=
  Join[{$Self}, getDescendants[$Self, _, False]] /; TrueQ[$Walking];
SelectDescendantsAndSelf[selector_] := 
  Join[
    If[MatchQ[$Self, mkPattern[selector]], {$Self}, {}],
    getDescendants[$Self, _, False]
] /; TrueQ[$Walking];
  
SelectNearestDescendants[] :=
  getDescendants[$Self, _, True] /; TrueQ[$Walking];
SelectNearestDescendants[selector_] :=
  getDescendants[$Self, mkPattern[selector], True] /; TrueQ[$Walking];

StringValue[] := StringValue[$Self] /; TrueQ[$Walking]
StringValue[ExprSequence[e_]] := StringValue[e]
StringValue[s_String] := s
StringValue[expr_] := Block[{$Children, $Options, $Styles},
  parse[expr];
  StringJoin @@ (SelectDescendantsAndSelf[String] /. None->"")
] /; TrueQ[$Walking]

Options[BoxToImage] = {
    Inline->False,
    ImageFormat->Automatic,
    ImageSize->Automatic,
    Magnification->Automatic,
    MaxImageSize->635,
    CropImage->False,
    StringContent->None,
    ConversionOptions->{}, 
    TransparentBackground->False
}

BoxToImage[opts___?OptionQ] := Module[ { format },
  format = Format /. {opts} /. Options[ResourceShingleTransmogrify];
  BoxToImage[StringJoin[
      GetFileBaseName[],"_img_",ToString[IncrementCounter["image"]],".",format
    ], opts ]
] /; TrueQ[$Walking]

BoxToImage[filename_String, opts___?OptionQ]:= BoxToImage[filename,$Self,opts] /; TrueQ[$Walking]

BoxToImage[filename_String,self_,opts___?OptionQ]:=BoxToInlineImage[self,opts]/;$InlineImages

BoxToInlineImage[self_, opts___?OptionQ] :=
Module[
  { expr, format, failexpr,
    width, height, newwidth, newheight, maxsize, convopts, mag,
    takex, takey, strcontent, inline, transparentbackground, isManipulate, baseline = Null, needexport, data,dims},
  (* don't create images if user requested not to *)
  If[ !TrueQ[ CreateImages /. {opts} /. Options[ResourceShingleTransmogrify] ] ,
    Return[]
  ];
  
  failexpr = {"Data" -> "","Width" -> "0", "Height" -> "0"};

  Catch[
    `trace[ImageCreation, "Creating image for a "<>ToString[Head@expr]<>" Expression"]; 
  
    (** process user options **)
    {format, mag, maxsize, convopts, strcontent, inline, transparentbackground} = 
      {
        ImageFormat, Magnification, MaxImageSize,
        ConversionOptions, StringContent, Inline, TransparentBackground
      } /. {opts} /. Options[BoxToImage];

	If[StringQ[strcontent], self = strcontent];
  
    Which[ 
      MatchQ[maxsize,None|Infinity], maxsize={Infinity,Infinity},
      MatchQ[maxsize,x_?Positive], maxsize={maxsize,Infinity},
      !MatchQ[maxsize,{x_?Positive,y_?Positive}], maxsize={Infinity,Infinity};
        Message[BoxToImage::badsz,maxsize]
    ];
      
    If[ format === Automatic,
      format=$ExportImageFormat];

    isManipulate = (Count[self, Manipulate`InterpretManipulate[_], Infinity] > 0);
    Which[
      (Head[self] === Cell) && isManipulate,
        expr = mangleCell[self, inline, transparentbackground];
      ,
      mag===Automatic,
        `trace[ImageCreation,"leaving magnification alone."];
        expr = parentify[self, opts];
      ,
      MatchQ[mag,x_?Positive],
        `trace[ImageCreation,"Setting magnification of expr to ",mag];
        
        If[ MatchQ[expr,_Cell|_StyleBox],
          `trace[ImageCreation, "Applying Magnification inline."];
          expr = parentify[ResourceShingleTransmogrify`Private`PrependOptions[$Self, Magnification->N@mag]]
          ,
          `trace[ImageCreation, "Applying magnification at parent level"];
          expr = ResourceShingleTransmogrify`Private`PrependOptions[parentify[$Self, opts], Magnification->N@mag]
        ]
      ,
      True,
        Message[BoxToImage::badmag,mag];
        expr = parentify[self, opts];
    ];
    If[Head[expr] =!= Notebook && !isManipulate,
      If[Head[expr] =!= Cell,
      	If[Head[expr] =!= List, expr = Cell[expr]] ];
      expr = Notebook[Flatten@{expr}, Sequence@@Rest[$Notebook]];
    ];
    
    expr = DeleteCases[expr, Rule[DockedCells, _]];

   With[{img = Rasterize[expr, "Image"]},
	 data = ExportString[
	   ImportString[ExportString[ img, "GIF"], "String"], "Base64"];
	 {width, height} = ImageDimensions[img];
	 ];
    Which[ 
      Inner[Greater, {width,height}, maxsize, Or],
        `trace[ImageCreation,"Image larger than MaxImageSize"];
        {newwidth,newheight} = Min@@@Transpose[{{width,height},maxsize}];    
        `trace[ImageCreation,"shrinking with HTML"]
      ,
      True,
        `trace[ImageCreation,"leaving image size alone."];
        {newwidth, newheight} = {width,height}
    ];  
      
    If[data === $Failed,
      `trace[ImageCreation,"Call to Export Failed!"];
  	  Throw[failexpr]
    ];
  
  	data="data:image/gif;base64,"<>data;
    (* baseline defined? *)
    If[baseline === Null, 0, baseline];

    (* return a list of rules *)
    `trace[ImageCreation, "Leaving BoxToImage"];

    { "Data" -> data, "Width" -> ToString@newwidth, "Height" -> ToString@newheight }
  ]
] /; TrueQ[$Walking]

































BoxToImage[filename_String, self_, opts___?OptionQ] :=
Module[
  { expr, format, failexpr, fullname = FileNameJoin[{DeployedResourceShingle`$webresourcepath,filename}],
    width, height, newwidth, newheight, size, maxsize, crop, convopts, mag,
    takex, takey, strcontent, inline, transparentbackground, isManipulate, baseline = Null, needexport},
  (* don't create images if user requested not to *)
  If[ !TrueQ[ CreateImages /. {opts} /. Options[ResourceShingleTransmogrify] ] ,
    Return[]
  ];
  
  failexpr = {"Filename" -> filename, "URL" -> StringReplace[filename,"\\"->"/"],
	    "Width" -> "0", "Height" -> "0", "Directory"->Directory[]};

  Catch[
    `trace[ImageCreation, "Creating image for a "<>ToString[Head@expr]<>" Expression"]; 
    `trace[{ImageCreation,2}, "Expr = "<>ToString@expr];
    `trace[ImageCreation,"FileName: "<>filename];
  
    (* create directory relative to current dir or complain *)
    If[ ResourceShingleTransmogrify`Private`createDirectory[DirectoryName[fullname]]===$Failed,
      `trace[ImageCreation, "Couldn't create directory "<>fullname];
      Throw[failexpr]
    ];
  
    (** process user options **)
    {format, size, mag, maxsize, convopts, strcontent, inline, transparentbackground} = 
      {
        ImageFormat, ImageSize, Magnification, MaxImageSize,
        ConversionOptions, StringContent, Inline, TransparentBackground
      } /. {opts} /. Options[BoxToImage];

	If[StringQ[strcontent], self = strcontent];
  
    Which[ 
      MatchQ[maxsize,None|Infinity], maxsize={Infinity,Infinity},
      MatchQ[maxsize,x_?Positive], maxsize={maxsize,Infinity},
      !MatchQ[maxsize,{x_?Positive,y_?Positive}], maxsize={Infinity,Infinity};
        Message[BoxToImage::badsz,maxsize]
    ];
      
    If[ format === Automatic,
      format=$ExportImageFormat];

    isManipulate = (Count[self, Manipulate`InterpretManipulate[_], Infinity] > 0);
    Which[
      (Head[self] === Cell) && isManipulate,
        expr = mangleCell[self, inline, transparentbackground];
      ,
      mag===Automatic,
        `trace[ImageCreation,"leaving magnification alone."];
        expr = parentify[self, opts];
      ,
      MatchQ[mag,x_?Positive],
        `trace[ImageCreation,"Setting magnification of expr to ",mag];
        
        If[ MatchQ[expr,_Cell|_StyleBox],
          `trace[ImageCreation, "Applying Magnification inline."];
          expr = parentify[ResourceShingleTransmogrify`Private`PrependOptions[$Self, Magnification->N@mag]]
          ,
          `trace[ImageCreation, "Applying magnification at parent level"];
          expr = ResourceShingleTransmogrify`Private`PrependOptions[parentify[$Self, opts], Magnification->N@mag]
        ]
      ,
      True,
        Message[BoxToImage::badmag,mag];
        expr = parentify[self, opts];
    ];
    If[Head[expr] =!= Notebook && !isManipulate,
      If[Head[expr] =!= Cell,
      	If[Head[expr] =!= List, expr = Cell[expr]] ];
      expr = Notebook[Flatten@{expr}, Sequence@@Rest[$Notebook]];
    ];
    
    expr = DeleteCases[expr, Rule[DockedCells, _]];

    If[ (MatchQ[format,"GIF"|"PPM"] && convopts==={})&&!$CloudEvaluation,
      `trace[ImageCreation,"Exporting with ExportPacket"];

        ret = MathLink`CallFrontEnd[ ExportPacket[expr, format, fullname, opts]];
      
      If[MatchQ[ret, $Failed|"$Failed"],
        `trace[ImageCreation,"Call to ExportPacket Failed!"];
        Message[BoxToImage::failed, Short[self]];
        Throw[failexpr]
      ];
      
      baseline = Last@ret;
      {width, height} = #[[2]]-#[[1]]& /@ Transpose@Round[ret[[2]]]
    ,
      `trace[ImageCreation,"Creating raster for format: "<>format];
      rdp = System`ConvertersDump`ToRasterDataPacket[Evaluate@expr,format];
      {height,width} = Switch[Length[dims=Dimensions[rdp[[3]]]],
        2, dims,
        3, Most[dims]
      ];
      needexport=True
    ];
  
    Which[ 
      Inner[Greater, {width,height}, maxsize, Or],
        `trace[ImageCreation,"Image larger than MaxImageSize"];
  
        {newwidth,newheight} = Min@@@Transpose[{{width,height},maxsize}];    
    
        ,
      size=!=Automatic,
        `trace[ImageCreation,"Using FromRDP to Export image with ImageSize->", size];
        If[MatchQ[size, _?Positive],
          {newwidth, newheight} = {size, ""}
          ,
          {newwidth, newheight} = size
        ]
      ,
      True,
        `trace[ImageCreation,"leaving image size alone."];
        {newwidth, newheight} = {width,height}
    ];  
    (* export image if we didn't already *)
        `trace[ImageCreation,"needexport"->needexport];
    If[ needexport,
    	ret=If[$CloudEvaluation,
        `trace[ImageCreation,"CloudExport"];
    		CloudExport[Evaluate@expr,"GIF",fullname,IconRules->{},Permissions->DeployedResourceShingle`$webImagePermissions],
        `trace[ImageCreation,"Export ***"];
    		Export[filename, Evaluate@expr, format, Sequence@@convopts]
    	]
    ];
      
    If[ret === $Failed,
      `trace[ImageCreation,"Call to ExportPacket Failed!"];
  	  Throw[failexpr]
    ];
  
    (* baseline defined? *)
    If[baseline === Null, 0, baseline];

    (* return a list of rules *)
    `trace[ImageCreation, "Leaving BoxToImage"];

    { "Filename" -> filename, "Width" -> ToString@newwidth, "Height" -> ToString@newheight, 
      "Directory"->Directory[], "URL" -> StringReplace[filename,"\\"->"/"], 
      "Baseline"-> ToString@baseline }
  ]
] /; TrueQ[$Walking]

BoxToImage::badsz = "ImageSize specifications should be either a positive real number or a list of two positive real numbers.  `1` is neither"
BoxToImage::badmag = "Magnification->`1` should be a positive real number."
BoxToImage::failed = "BoxToImage cannot create an image of `1`"


GetNodeFunctionInit := (
ClearAll[GetNodeFunction];
GetNodeFunction[pos_] := GetNodeFunction[pos] = Module[{ex, ch},
  ex = ExtractOne[$Notebook, pos];
  ch = GetChildPositions[ex];
  ex = Delete[ReplacePart[ex, First[ch] -> #], Rest[ch]];
  ex
];
GetNodeFunction[src_, pos_] := GetNodeFunction[src, pos] = Module[{ex, ch},
  ex = ExtractOne[src, pos];
  ch = GetChildPositions[ex];
  ex = Delete[ReplacePart[ex, First[ch] -> #], Rest[ch]];
  ex
];
);

parentify[expr_, opts___?OptionQ] := Module[
  {
    inline = TrueQ[Inline /. {opts} /. Options[BoxToImage]],
    transparentbackground = TrueQ[TransparentBackground /. {opts} /. Options[BoxToImage]],
    new, src, pos, isn, funs
  },

  (* new = If[Head[expr] === Cell, expr, Cell[expr]]; Notebook[{new}] *)

  {src, pos} = GetNodeSourcePosition[$Self];

  If[Head[expr] === Cell && Head[First[expr]] =!= CellGroupData,
    new = mangleCell[expr, inline, transparentbackground],
    new = expr];
  isn = (src === $Notebook);
  funcs = Map[
    Function[{p}, Module[{po, ex, ch},
      po = Flatten[p];
      ex = If[isn, GetNodeFunction[po], GetNodeFunction[src, po]];
      ex = ex /.{SuperscriptBox[a_]:>a};
      If[ExtractOne[src, Append[po, 0]] === Cell &&
         ExtractOne[src, Join[po, {1, 0}]] =!= CellGroupData,
        ex = mangleCell[ex, inline, transparentbackground] ];
      Function[Evaluate@ex]
    ]],
    Rest@Most@FoldList[Append[#1, #2]&, {}, pos]
  ];
  new = Apply[Composition, funcs][new];
  new /. Notebook[c_Cell, o___] :> Notebook[{c}, o]
];

mangleCell[expr_, inline_, transparentbackground_] := Module[{new, optColumnWidths = Automatic},

  If[
    Or[
      Positive@Length@Cases[expr, Rule[CellLabel, "*TableForm*"], Infinity],
      (* a pattern that happens to be present in Dataset outputs: *)
      Positive@Length@Cases[expr, BoxData[TagBox[TagBox[StyleBox[GridBox[{__}, __], __], Deploy], False]], Infinity],
      And[
        Positive@Length@Cases[expr, Cell[_, "Output", ___], Infinity],
        Positive@Length@Cases[expr, TemplateBox[_, "Dataset", ___], Infinity]
      ]
    ],
    optColumnWidths = All];

  new = ResourceShingleTransmogrify`Private`PrependOptions[expr, {
    (** always **)
    CellMargins->{{0,0},{0,1}},
    ShowCellBracket->False,
    CellOpen->True,
    If[transparentbackground,
      (** only transparentbackground **)
      Unevaluated@Sequence[
        Background->GrayLevel[1,0]
      ],
      Unevaluated@Sequence[]
      ],
    If[inline,
      (** only inline **)
      Unevaluated@Sequence[
        TextAlignment->Left,
        CellDingbat->None,
        ShowCellLabel->False,
        CellFrameLabels->{{None, None}, {None, None}},
        GridBoxOptions->{ ColumnWidths -> optColumnWidths, GridBoxDividers->{} },
        CellFrame->{{0, 0}, {0, 0}},
        CellFrameMargins->{{0, 0}, {0, 0}}
      ],
      Unevaluated@Sequence[]
  ]}];
  If[inline,
    new = DeleteCases[new, _[CellLabel, _]] ];
  new
];



GetParameter[s_String] := (
  If[ 
    Length[#] > 0, First[#], Message[ResourceShingleTransmogrify::noparam, s]; Null
  ]&[$Parameters[s]]
) /; TrueQ[$Walking] 

IncrementCounter[counter_String] := ToString[++$Counters[counter]] /; TrueQ[$Walking]
DecrementCounter[counter_String] := ToString[--$Counters[counter]] /; TrueQ[$Walking]
GetCounter[counter_String] := ToString[$Counters[counter]] /; TrueQ[$Walking]

SetCounter[counter_String, val_]:= ToString[$Counters[counter]=val] /; TrueQ[$Walking]

ResetCounters[] := ($Counters=.;$CellLabelMap=<||>;) /; TrueQ[$Walking]
ResetCounters[counter_String] := ($Counters[counter]=0)  /; TrueQ[$Walking] 

$CellLabelMap=<||>;

getInputLabel[HoldPattern[Cell][___,HoldPattern[Rule][CellLabel,val_],___]]:=useInputCellLabel[val]
getInputLabel[_]:=(IncrementCounter["iOutput"];IncrementCounter["iInput"])

useInputCellLabel[str_String]:=With[{n=StringCases[str,"In["~~d:(DigitCharacter..)~~"]":>d]},
	If[Length[n]==1,
		useInputCellLabel[FromDigits[First[n]]],
		getInputLabel[Automatic]
	]
]

useInputCellLabel[n_Integer]:=With[{c=IncrementCounter["iInput"]},
	SetCounter["iOutput",FromDigits[c]];
	$CellLabelMap[n]=c
]

useInputCellLabel[_]:=getInputLabel[Automatic]

getOutputLabel[HoldPattern[Cell][___,HoldPattern[Rule][CellLabel,val_],___]]:=useOutputCellLabel[val]
getOutputLabel[_]:=GetCounter["iOutput"]

useOutputCellLabel[str_String]:=With[{n=StringCases[str,"Out["~~d:(DigitCharacter..)~~"]":>d]},
	If[Length[n]==1,
		useOutputCellLabel[FromDigits[First[n]]],
		getOutputLabel[Automatic]
	]
]

useOutputCellLabel[n_Integer]:=With[{c=$CellLabelMap[n]},
	If[StringQ[c],c,smartIncrementOutput[n]]
]

smartIncrementOutput[n_]:=Block[{maxKey=Max[Keys[$CellLabelMap]], max},
	max=Max[{
			Replace[FromDigits[$CellLabelMap[maxKey]],Except[_Integer]->0,{0}]+n-maxKey,$Counters["iOutput"]+1}];
	SetCounter["iInput",max];
	SetCounter["iOutput",max]
]/;Length[$CellLabelMap]>0

smartIncrementOutput[n_]:=With[{new=$Counters["iOutput"]+1},
	SetCounter["iInput",new];
	SetCounter["iOutput",new]
]

useOutputCellLabel[_]:=getOutputLabel[Automatic]

XMLTransformInit[x_XMLTransform,___?OptionQ]:=Module[
  {t = x},
  Which[ 
    MatchQ[t, XMLTransform[]], 
      t = XMLTransform[{}],
    !MatchQ[t, XMLTransform[l_List,___?OptionQ]],
      Return[$Failed]
  ];
  $XMLTransformList=`bag[];
  If[ $LastRequestedTransform === t,
    `trace[TransformParsing, "Same raw transform requested. Using cached transform!"];
    $CachedTransform,
    `trace[TransformParsing, "New raw transform requested. Saving."];
    $LastRequestedTransform = t
  ]
]

(* see if a file is either a full file name or a URI *)
uriQ[f_String]:=!StringFreeQ[f,RegularExpression["^(?:/|[A-Z]:\\\\|(?:(?i)http|f(?:tp|ile)|paclet)://)"]]



(* XMLTransform related error messages *)
ResourceShingleTransmogrify::badtrans = "The XMLTransform file `1` did not contain an XMLTransform."
ResourceShingleTransmogrify::imptrans= "Could not import the file `1` for some reason.  Make sure that it is a file containing valid Mathematica code."
ResourceShingleTransmogrify::obs = "The option or function `1` for ResourceShingleTransmogrify is now obsolete. `2` Please\
 see the documentation."

ResourceShingleTransmogrify::obstrans = "The Template[] syntax for ResourceShingleTransmogrify is now obsolete.\
 Attempting to convert to XMLTransform[].\n\
 Please see the documentation."

ResourceShingleTransmogrify::notrans = "You specified an XMLTransform but ResourceShingleTransmogrify\
 was not able to correctly process it or one of its included transforms."
ResourceShingleTransmogrify::notransfile = "ResourceShingleTransmogrify could not find the transform file \"`1`\"."


XMLTransform::obsfn = "A transform file for `1` was not found, but one for `1`.m was.
 The filename format of {\"dir\",\"format\"}, where format does not have a full extension\
 is deprecated.  Please use the entire filename extension."

XMLTransform::imgattrs = "Warning: missing `1` attribute within Image tag."; 


DieIf[ bool_ ]:= If[ bool === True,
  Message[ResourceShingleTransmogrify::abort]; 
  Abort[ ]
  ,
  Message[ResourceShingleTransmogrify::continue];
]
ResourceShingleTransmogrify::abort = "ResourceShingleTransmogrify encountered an error and is Aborting (see\
 AbortOnError)"
ResourceShingleTransmogrify::continue = "ResourceShingleTransmogrify encountered an error but is attempting\
 to continue anyway.  Results may be unpredictable. (see AbortOnError)"

TagElement[element_String, attr_List, cont_]:=  
XMLElement[element, attr, Flatten@{
  		Which[HasOption[CellTags] && HasOption[CellID],
			Sequence@{cont  },
		HasOption[CellID], Sequence@{cont}, 
		True, cont]
		}];

DIV[ attr_List:{}, cont_ ] := TagElement["div", attr, cont];
P[ attr_List:{}, cont_ ] := TagElement["p", attr, cont];
Ol[ attr_List:{}, cont_ ] := TagElement["ol", attr, cont];
Li[ attr_List:{}, cont_ ] := TagElement["li", attr, cont];
H1[ attr_List:{}, cont_ ] := TagElement["h1", attr, cont];
H2[ attr_List:{}, cont_ ] := TagElement["h2", attr, cont];
H3[ attr_List:{}, cont_ ] := TagElement["h3", attr, cont];
H4[ attr_List:{}, cont_ ] := TagElement["h4", attr, cont];
TD[ attr_List:{}, cont_ ] := TagElement["td", attr, cont];

Unprotect[Span];
Span[ attr_List:{}, cont_ ] := XMLElement["span", attr, Flatten@{cont} ];
A[ attr_List:{}, cont_ ] := XMLElement["a", attr, Flatten@{cont} ];
UL[ attr_List:{}, cont_ ] := XMLElement["ul", attr, Flatten@{cont} ];
(* Li[ attr_List:{}, cont_ ] := XMLElement["li", attr, Flatten@{cont} ]; *)
Br[ attr_List:{} ] := XMLElement["br", attr, {}];
Br[] := XMLElement["br", {}, {}];
Img[ attr_List:{} ] := 
(
  XMLElement["img", attr, {}]
);

$InlineImages=True;

(** Image **)
Unprotect[Image];
Image[ filename_String, attr_List:{}, expr___, opts___?OptionQ]:=(
  XMLElement["img",{
    "src"->("URL"/.#),"height"->("Height"/.#),"width"->("Width"/.#),Sequence@@attr},{}
  ]&[BoxToImage[filename, expr, Sequence@@Flatten[{opts}]]]
)/;!$InlineImages

Image[ filename_String, attr_List:{}, expr___, opts___?OptionQ]:=(
  XMLElement["img",{
    "src"->("Data"/.#),"height"->("Height"/.#),"width"->("Width"/.#),Sequence@@attr},{}
  ]&[BoxToImage[filename, expr, Sequence@@Flatten[{opts}]]]
)/;$InlineImages


(* If Comment for SSI  *)
Options[IfCommentSSI] = {ExprCommand -> "is_development", Recurse->True};
IfCommentSSI[t_, f___, opts___?OptionQ] := 
DIV[{
  "\n", 
  XMLObject["Comment"]["#if expr='" <> 
    "${is_development} = 1" <> 
    "' "], 
  Recurse @ t, 
  XMLObject["Comment"]["#else"], 
  Recurse @ f, 
  XMLObject["Comment"]["#endif"], 
  "\n"
}];


End[] (* "`Private`" *)
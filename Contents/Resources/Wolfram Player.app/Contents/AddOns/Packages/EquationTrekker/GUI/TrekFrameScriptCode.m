
(* The 'Script global' variables used across various functions in the
   EquationTrekker Scripts
   
   NOTE, as with all GUIKit` scripts, these are not global in the kernel Global`
   sense but global within one instance of an EquationTrekker so there is no danger of
   these symbols overlapping other instances of EquationTrekker that might also be running
*)
 
(* We store independent and dependent symbols during init to recompute in later functions *)
$IndepVar;

(* We include support for letting the user adjust the min and max independent variable
   values which would recompute a trek, just as changing a parameter would.
   Display this with min/max textfields and possibly a slider for displaying
   a highlighted point on each trek at the selected indep var value
 *)
$DefaultMinIndepValue;
$DefaultMaxIndepValue;

$DefaultOriginIndependent;

$DependVars;

(* This stores a list of {"param key" -> currentValue, ...} rules *)
$ParameterKeyRules = {};

(* At some point in the future we may want to store the state of the above parameters
  in the user interface in case editing and manipulation of the parameters that exist
  would be done in the UI. At least in a model that could initiate from the ui 
  if treks with equations were optionally saved as documents (ie ExpresionML etc... )
*)

(* During init these are filtered and stored for later use in other functions *)
$FilteredEquationTrekkerOptions;

(* This is the function which will be called to generate points from
   SetTrekPoints *)
$TrekGenerator;

$IsFirstOrder = False;

(* These keep some of the original input data used to make a state *)
$InputData;
$InputOptions;

(*******************************
   This initializes the particular equation trekker, 
     evaluating the equation,
     finding all parameters and building up
     corresponding user interface elements 
     representing each parameter

    If initialization fails, $IsFailing is set so we do not
    make any attempt to return results.

 *******************************)

$IsFailing; 

InitEquationTrekker[args___] := 
Module[{iet},
    iet = Catch[Apply[InitEquationTrekkerImpl, {args} /. 
      ((s_Symbol /; (Context[s] === "Global`")):> Symbol[SymbolName[s]])]];
    $IsFailing = TrueQ[iet === $Failed];
    If[$IsFailing,
        CloseGUIObject[]];
    iet
];

InitEquationTrekkerImpl[eqns_, dvarsin_, {ivar_, begin_, end_}, opts___?OptionQ] :=
Block[{xpts, ypts, values, xmax, xmin, ymax, ymin, dim, sze, tvars, 
  liv, lab, generator, gopts = {}, state, trekData = {}, canvasSize, dmode},

    $InputData = {eqns, dvarsin, {ivar, begin, end}};
    $InputOptions = Flatten[{opts}];

    state = "State" /. $InputOptions;
    If[Head[state] === EquationTrekkerState, 
      Return[
        InitEquationTrekker[Sequence @@ state[[1]], "TrekData" -> state[[3]], Sequence @@ state[[4]]]
        ]
      ];

    (* Once the GUI is created we need to prepopulate with the trek data below *)
    trekData = "TrekData" /. $InputOptions /. {"TrekData" -> {}};
    $InputOptions = DeleteCases[$InputOptions, "TrekData" -> _];

    $FilteredEquationTrekkerOptions = Flatten@
      {FilterRules[{opts}, Options[EquationTrekker]],
       Options[EquationTrekker]};


    $DefaultMinIndepValue = begin; $DefaultMaxIndepValue = end;
    $DefaultOriginIndependent = begin;

    (* Set up the trek generator *)
    generator = TrekGenerator /. $FilteredEquationTrekkerOptions;
    If[ListQ[generator],
        If[(Length[generator] == 0) || Not[OptionQ[gopts = Rest[generator]]],
            Message[EquationTrekker::tgen, generator];
            Throw[$Failed]];
        generator = First[generator]
    ];
    If[Not[MatchQ[generator, _Symbol]],
        Message[EquationTrekker::gsym, TrekGenerator /. $FilteredEquationTrekkerOptions];
        Throw[$Failed]
    ];
    If[ivar === None, 
        $TrekGenerator = InitializeGenerator[generator, eqns, dvarsin, gopts],
        $TrekGenerator = InitializeGenerator[generator, eqns, dvarsin, {ivar, begin, end}, gopts]
    ];
    tvars = $TrekGenerator@"Variables"[];
    If[ivar === None,
        $IndepVar = None;
        $DependVars = tvars,        
        {$IndepVar, $DependVars} = tvars
    ];
    $IsFirstOrder = (Length[$DependVars] == 1);
    If[$IsFirstOrder && ($IndepVar === None),
        (* First order mode doesn't make unless there is an independent variable *)
        Throw[$Failed]];
   
    (* Processes the PlotRange option *)
    pr = PlotRange /. $FilteredEquationTrekkerOptions;
    If[ListQ[pr] && Length[pr] == 2 && First[pr] === Automatic,
      If[ TrueQ[$IsFirstOrder], 
        pr[[1]] = {$DefaultMinIndepValue, $DefaultMaxIndepValue},
        pr[[1]] = {-1,1} ];
      $FilteredEquationTrekkerOptions = Flatten[{PlotRange -> pr, $FilteredEquationTrekkerOptions}]
      ];
    pr = N[pr];
    If[Not[TensorDimensionQ[pr, {2, 2}, MRealQ]],
      Message[ MessageName[EquationTrekker, "prange"], pr];
    (* Use unit square as default *)
    pr = {{0,1},{0,1}}];
    {{xmin, xmax}, {ymin, ymax}} = pr;
    
    (* This would exist if we are processing an EquationTrekkerState set *)
    canvasSize = "CanvasSize" /. Flatten[{opts}] /. {"CanvasSize" -> Automatic};
    (* Processes the ImageSize option. *)
    If[ canvasSize === Automatic,
      canvasSize = Round[ImageSize /. $FilteredEquationTrekkerOptions]
      ];
    (* Do we really want to throw here or instead come up with a default *)
    If[PositiveIntegerQ[canvasSize], canvasSize = canvasSize*{1,1}];
    If[Not[TensorDimensionQ[canvasSize, {2}, PositiveIntegerQ]], 
      Message[ MessageName[EquationTrekker, "isize"], canvasSize];
      Throw[$Failed]];
      
    {xpts, ypts} = canvasSize;

    (* Process the parameters option *)
    parameters = Flatten[{TrekParameters /. $FilteredEquationTrekkerOptions}];
    If[Not[ParameterRuleQ[parameters]],
      Message[EquationTrekker::prules, parameters];
      Throw[$Failed]];

    (* This spacing needs to get called before creating parameter ui *)
    InvokeMethod[{"parameterPanel", "addSpace"}, 5];
    
    parameters = Union[parameters, SameTest->Function[SameQ[#1[[1]], #2[[1]]]]];
    $ParameterKeyRules = (# -> PropertyValue[{#, "defaultValue"}])& /@ (CreateParameter /@ parameters);

    InvokeMethod[{"parameterPanel", "addGlue"}];
        
    sze = Widget["Dimension", {"width" -> xpts, "height" -> ypts}];
    SetPropertyValue[{"canvas", "preferredSize"}, sze];
    InvokeMethod[{"canvas", "reshape"}, 0, 0, xpts, ypts];

    typesetEquation = ColumnForm[ Flatten[{ Decontext[$TrekGenerator@"Display"[]] }]];
    
    SetPropertyValue[{"typesetEquationsLabel", "data"},
      ExportString[ TraditionalForm[typesetEquation], "GIF",
        "TransparentColor" -> GrayLevel[1]]
      ];
   
    PropertyValue[{"trekInspectorPanel", "xInitialConditionImageLabel"}, 
      Name -> "xInitialConditionImageLabel"];
    PropertyValue[{"trekInspectorPanel", "yInitialConditionImageLabel"}, 
      Name -> "yInitialConditionImageLabel"];
    PropertyValue[{"trekInspectorPanel", "minIndependentVariableImageLabel"}, 
      Name -> "minIndependentVariableImageLabel"];
    PropertyValue[{"trekInspectorPanel", "maxIndependentVariableImageLabel"}, 
      Name -> "maxIndependentVariableImageLabel"];
    
    liv = Subscript[Decontext[$IndepVar], 0];
    If[$IsFirstOrder, 
        lab = liv,
        lab = Decontext[First[$DependVars]];
        If[$IndepVar =!= None, lab = lab[liv]]
    ];
    SetPropertyValue[{"xInitialConditionImageLabel", "data"},
        ExportString[ 
            TraditionalForm[lab], 
            "GIF",
            "TransparentColor" -> GrayLevel[1]]
    ];

    lab = Decontext[Last[$DependVars]];
    If[$IndepVar =!= None, lab = lab[liv]];
    SetPropertyValue[{"yInitialConditionImageLabel", "data"},
        ExportString[ 
            TraditionalForm[lab], 
            "GIF",
            "TransparentColor" -> GrayLevel[1]]
    ];

    If[$IndepVar =!= None, 
        SetPropertyValue[{"minIndependentVariableImageLabel", "data"},
            ExportString[ TraditionalForm[
                Subscript[Decontext[$IndepVar], "min"]], "GIF",
                "TransparentColor" -> GrayLevel[1]]
        ];
        SetPropertyValue[{"maxIndependentVariableImageLabel", "data"},
            ExportString[ TraditionalForm[Subscript[Decontext[$IndepVar], "max"]], "GIF",
            "TransparentColor" -> GrayLevel[1]]
        ];
      ,
      InvokeMethod[{"trekInspectorPanel", "setupForNoIndependentVariable"}];
      
    ];

    InvokeMethod[{"trekPane", "setTransform"}, 
       xmin, xmax, ymin, ymax, 
       N[Abs[xpts/(xmax-xmin)]], N[Abs[ypts/(ymax-ymin)]] ];
    
    InvokeMethod[{"trekPane", "setDefaultIndependentRange"}, 
      N[{$DefaultMinIndepValue, $DefaultMaxIndepValue}]];
    
    If[ !TrueQ[$IsFirstOrder] && ($IndepVar =!= None),
        PropertyValue[{"trekInspectorPanel", "originIndependentImageLabel"}, 
            Name -> "originIndependentImageLabel"];
        SetPropertyValue[{"originIndependentImageLabel", "data"},
            ExportString[ TraditionalForm[liv], 
            "GIF",
            "TransparentColor" -> GrayLevel[1]]
        ];
        InvokeMethod[{"trekPane", "setDefaultOriginIndependent"}, N[$DefaultOriginIndependent]];
        ,
        InvokeMethod[{"trekInspectorPanel", "setupForFirstOrder"}];
    ];
      
    InvokeMethod[{"trekInspectorPanel", "updateFromSelection"}];
    
    (* Setup the compute equations replacing parameter placeholders with their
       current values *)
    $ParameterRules = Thread[Rule[parameters[[All,1]], $ParameterKeyRules[[All,2]]]];
    (* Make sure that parameters are set in the generator *)
    $TrekGenerator = $TrekGenerator@"ChangeParameters"[$ParameterRules];

    (* Set the default display mode *)
    dmode = $TrekGenerator@"DisplayMode"[];
    If[Not[SameQ[dmode, "Points"]],
        dmode = "Line"];
    SetPropertyValue[{"trekPane", "defaultDisplayMode"}, PropertyValue[{"trekPane", dmode}]];

    If[ trekData =!= {} && Length[trekData] > 0,
      CreateTrekFromState /@ trekData;
      ];
       
    ]; (* end Block *)



(*******************************
   CreateTrekFromState
   
   Takes EquationTrekkerState data
   and produces an equivalent
   trek in the user interface
 *******************************)
 
CreateTrekFromState[TrekData[form_, cond_, ivdata_, {color_, style_}]] :=
  Module[{trekID, fig},
    trekID = InvokeMethod[{"trekPane", "createTrekKey"}];
    If[$IsFirstOrder,
        SetTrekPoints[trekID, Prepend[cond, First[ivdata]], Sequence @@ ivdata],
        SetTrekPoints[trekID, cond, Sequence @@ ivdata]];
    
    (* Now set trek figure color and display type *)
    fig = InvokeMethod[{"trekPane", "getTrekFigure"}, trekID];
    SetPropertyValue[{fig, "colorExpr"}, color];
    If[ style === "Points", 
      SetPropertyValue[{fig, "displayMode"}, PropertyValue[{fig, "Points"}]]
      ];
      
    ];
    
(*******************************
   SetTrekPoints
   
   This is where the actual calculation of
   trek points occurs
 *******************************)
   
SetTrekPoints[trekID_, x0_, t0_, {independMin_, independMax_}] :=
  Module[{gent, t = t0, points = x0},
    If[$IsFirstOrder, points = Drop[x0, 1]; t = First[x0]];
    gent = $TrekGenerator["GenerateTrek"[points, {t, independMin, independMax}]];
     If[gent =!= $Failed && ListQ[gent] && (Length[gent] > 0),
        If[(Length[gent] == 2) && Not[ListQ[gent[[2]]]],
            points = gent[[1]],
            points = gent
        ]
    ];
    If[MatrixQ[points],
        If[Dimensions[points][[2]] > 2,
            (* Has {t, x, y}: drop t for now *)
            points = Take[points, All, -2]];
        points = Transpose[points];
        (* Relax type checking for performance speed when sending large arrays *)
        Block[{$RelaxedTypeChecking = True},
            InvokeMethod[{"trekPane", "setTrekPoints"}, trekID, x0, t0, {independMin, independMax}, points];
        ],
    (* else *)
        If[gent =!= {},Message[EquationTrekker::ncsol, sol]];
  ];
];


(*******************************
   ParameterChanged
   
   This binding is setup when creating each 
   user interface for parameters so that
   when the user changes a ui parameter value
   we update the values in the kernel and
   request all treks to recalculate their points
 *******************************)

ParameterChanged[ paramKey_, newValue_] := (
   $ParameterKeyRules = $ParameterKeyRules /. HoldPattern[paramKey -> _] -> (paramKey -> newValue);
   $ParameterRules = Thread[Rule[$ParameterRules[[All,1]], $ParameterKeyRules[[All,2]]]];
   (* Setup the compute equations replacing parameter placeholders with their
       current values *)
   $TrekGenerator = $TrekGenerator["ChangeParameters"[$ParameterRules]];
   InvokeMethod[{"trekPane", "updateTrekPoints"}];
   );
   
(*******************************
   Event bindings from user 
   interface into Mathematica
 *******************************)

(* Whenever the user interface requires updated trek points calculated
   it fires an event so we listen for any moved treks and compute
   the new points given the event's new proposed origin
   As with all BindEvents, "#" is the vended event object, in this case, a TrekEvent
 *)
BindEvent[{"trekPane", "trekOriginDidChange"}, 
  Script[
    SetTrekPoints[ PropertyValue[{"#", "key"}], PropertyValue[{"#", "origin"}], 
      PropertyValue[{"#", "originIndependent"}], PropertyValue[{"#", "independentRange"}] ];
    ] ];
BindEvent[{"trekPane", "trekIndependentRangeDidChange"}, 
  Script[
    SetTrekPoints[ PropertyValue[{"#", "key"}], PropertyValue[{"#", "origin"}], 
      PropertyValue[{"#", "originIndependent"}], PropertyValue[{"#", "independentRange"}] ];
    ] ];
    

PropertyValue[{"trekInspectorPanel", "xInitialConditionTextField"}, Name -> "xInitialConditionTextField"];
PropertyValue[{"trekInspectorPanel", "yInitialConditionTextField"}, Name -> "yInitialConditionTextField"];
BindEvent[{"xInitialConditionTextField", "action"}, 
  Script[ UpdateFromConditionFields[] ]];
BindEvent[{"yInitialConditionTextField", "action"}, 
  Script[ UpdateFromConditionFields[] ]];
  
UpdateFromConditionFields[] := Module[{newX, newY},
  newX = ToExpression[PropertyValue[{"xInitialConditionTextField","text"}]];
  newY = ToExpression[PropertyValue[{"yInitialConditionTextField","text"}]];
  If[ NumberQ[ N[newX]] && NumberQ[ N[newY]],
    InvokeMethod[{"trekPane", "setSelectionInitialConditions"}, {N[newX], N[newY]}];
    InvokeMethod[{"canvas", "requestFocus"}]
    ];
  ];
  
PropertyValue[{"trekInspectorPanel", "originIndependentTextField"}, 
  Name -> "originIndependentTextField"];
BindEvent[{"originIndependentTextField", "action"}, 
  Script[ UpdateFromOriginIndependentField[] ]];

UpdateFromOriginIndependentField[] := Module[{newVal},
  newVal = ToExpression[PropertyValue[{"originIndependentTextField","text"}]];
  If[ NumberQ[ N[newVal]],
    InvokeMethod[{"trekPane", "setSelectionOriginIndependent"}, N[newVal]];
    InvokeMethod[{"canvas", "requestFocus"}]
    ];
  ];
  
PropertyValue[{"trekInspectorPanel", "minIndependentVariableTextField"}, 
  Name -> "minIndependentVariableTextField"];
PropertyValue[{"trekInspectorPanel", "maxIndependentVariableTextField"}, 
  Name -> "maxIndependentVariableTextField"];
BindEvent[{"minIndependentVariableTextField", "action"}, 
  Script[ UpdateFromIndependentVariableFields[] ]];
BindEvent[{"maxIndependentVariableTextField", "action"}, 
  Script[ UpdateFromIndependentVariableFields[] ]];
  
UpdateFromIndependentVariableFields[] := Module[{newX, newY},
  newX = ToExpression[PropertyValue[{"minIndependentVariableTextField","text"}]];
  newY = ToExpression[PropertyValue[{"maxIndependentVariableTextField","text"}]];
  If[ NumberQ[ N[newX]] && NumberQ[ N[newY]],
    InvokeMethod[{"trekPane", "setSelectionIndependentRange"}, {N[newX], N[newY]}];
    InvokeMethod[{"canvas", "requestFocus"}]
    ];
  ];
  
PropertyValue[{"trekInspectorPanel", "defaultIndependentRangeMinButton"}, 
  Name -> "defaultIndependentRangeMinButton"];
BindEvent[{"defaultIndependentRangeMinButton", "action"}, 
  Script[ UpdateDefaultIndependentVariableMinField[] ]];
  
UpdateDefaultIndependentVariableMinField[] := Module[{newX},
  newX = ToExpression[PropertyValue[{"minIndependentVariableTextField","text"}]];
  If[ NumberQ[ N[newX]],
    InvokeMethod[{"trekPane", "setDefaultIndependentRangeMin"}, N[newX]];
    InvokeMethod[{"canvas", "requestFocus"}]
    ];
  ];
  
PropertyValue[{"trekInspectorPanel", "defaultIndependentRangeMaxButton"}, 
  Name -> "defaultIndependentRangeMaxButton"];
BindEvent[{"defaultIndependentRangeMaxButton", "action"}, 
  Script[ UpdateDefaultIndependentVariableMaxField[] ]];
  
UpdateDefaultIndependentVariableMaxField[] := Module[{newX},
  newX = ToExpression[PropertyValue[{"maxIndependentVariableTextField","text"}]];
  If[ NumberQ[ N[newX]],
    InvokeMethod[{"trekPane", "setDefaultIndependentRangeMax"}, N[newX]];
    InvokeMethod[{"canvas", "requestFocus"}]
    ];
  ];
  
PropertyValue[{"trekInspectorPanel", "defaultOriginIndependentButton"}, 
  Name -> "defaultOriginIndependentButton"];
BindEvent[{"defaultOriginIndependentButton", "action"}, 
  Script[ UpdateDefaultOriginIndependentField[] ]];
  
UpdateDefaultOriginIndependentField[] := Module[{newVal},
  newVal = ToExpression[PropertyValue[{"originIndependentTextField","text"}]];
  If[ NumberQ[ N[newVal]],
    InvokeMethod[{"trekPane", "setDefaultOriginIndependent"}, N[newVal]];
    InvokeMethod[{"canvas", "requestFocus"}]
    ];
  ];
  

(*******************************
   CreateParameter
   
   Takes a defined Parameter expression and generates
   an appropriate user interface object with
   bindings back into Mathematica when the
   user interface requests a change in the
   parameter value so we can recalculate all
   the trek points
 *******************************)
 
CreateParameter[name_->value_?NumericQ] := 
    If[Positive[value], CreateParameter[name->{value, {0., 2. value}}],
        If[Negative[value], CreateParameter[name->{value, {-2. value, 0.}}],
            If[Developer`ZeroQ[value], CreateParameter[name->{0., {-1.,1.}}],
                (* Need message *)
                Throw[Message[EquationTrekker::param, value]; $Failed]
            ]
        ]
    ];

CreateParameter[name_->{value_}] := CreateParameter[name->value];

(* Currently we are only building slider based user interfaces but
   we can and should extend this to various convenient controls given
   the type of value ranges for a parameter
*)
CreateParameter[Rule[name_, p:{valuein_, {vminin_, vmaxin_}}]] := 
  Block[{value = N[valuein], vmin = N[vminin], vmax = N[vmaxin],paramKey, param, paramUI, paramUITextKey},
  
    If[Not[MRealQ[value]], 
      Throw[ Message[EquationTrekker::param, p]]];
    If[Not[MRealQ[vmin] && MRealQ[vmax] && Positive[Abs[vmax - vmin]]], 
      Throw[ Message[EquationTrekker::param, p]]];
      
    paramKey = ToString[Unique["ParameterStore"]];
    
    param = Widget["class:com.wolfram.guikit.trek.Parameter", {
       "key" -> paramKey,
       "description" -> ToString[name],
       "defaultValue" -> value,
       "minValue" -> vmin,
       "maxValue" -> vmax
       }, Name -> paramKey];
       
    SetPropertyValue[{param, "name"}, MakeJavaExpr[name]];
    
    BindEvent[{paramKey, "didChange"},
      Script[ParameterChanged[#, PropertyValue[{"#", "newValue"}] ]] ]&[paramKey];
          
    paramUI = Widget["class:com.wolfram.guikit.trek.ParameterUI"];
    label = Widget["ImageLabel"];
    SetPropertyValue[{label, "data"},
      ExportString[ TraditionalForm[Decontext[name]], "GIF",
        "TransparentColor" -> GrayLevel[1]]
      ];
    SetPropertyValue[{paramUI, "label"}, label];
    SetPropertyValue[{paramUI, "parameter"}, param];
    SetPropertyValue[{paramUI, "trekPane"}, WidgetReference["trekPane"]];
    paramUITextKey = paramKey <> "TextField";
    PropertyValue[{paramUI, "textField"}, Name -> paramUITextKey];
    BindEvent[{paramUITextKey, "action"},
      Script[
        val = ToExpression[PropertyValue[{PropertyValue[{"#", "source"}], "text"}]];
        If[ NumberQ[N[val]],
          SetPropertyValue[{#, "value"}, N[val]];
          InvokeMethod[{"canvas", "requestFocus"}];
          ];
        ]
       ]&[paramKey];
    
    InvokeMethod[{"parameterPanel", "add"}, paramUI];
    
    paramKey
    ];

CreateParameter[p_] := 
  Throw[ Message[EquationTrekker::param, p]];


(*******************************
    CreateTrekkerState

    This creates an EquationTrekkerState object which can be
    used to recreate the settings and display at the time
    which it was created.

    It is called by CreateEndModalResults
    and could be called non-modally given a user interface
    hook for doing this and putting the results somewhere
    appropriate (e.g. a notebook)
*)
 
CreateTrekkerState[] :=
  Module[{treks, trekFigures, paramValues, conditions, pathColors, trekDisplayType, trekPointInfo, plotGraphics,
     xmax, xmin, ymax, ymin, rnges, originIndep, xscale, yscale, form, state, parms, canvasSize, useOpts, oldGrs},

   (* use getInitialPlotRange to get the original plotRange sent in instead of current visible rect *)
   {xmin, xmax, ymin, ymax} = InvokeMethod[{"trekPane", "getPlotRange"}];
   {xscale, yscale} = InvokeMethod[{"trekPane", "getScale"}];

   treks = PropertyValue[{"trekPane", "treks"}];
   trekFigures = PropertyValue[{"trekPane", "trekFigures"}];
   
   trekDisplayType = PropertyValue[{#, "displayMode"}]& /@ trekFigures;
   conditions = PropertyValue[{#, "origin"}]& /@ treks;
   originIndep = PropertyValue[{#, "originIndependent"}]& /@ treks;
   rnges = PropertyValue[{#, "independentRange"}]& /@ treks;
   
   pathColors = PropertyValue[{#, "colorExpr"}]& /@ trekFigures;
   Block[{$RelaxedTypeChecking = True},
     trekPointInfo = Transpose[PropertyValue[{#, "points"}]]& /@ treks;
     ];

   If[ Length[conditions] > 0,
    oldGrs = TrueQ[$VersionNumber < 6.0];
    plotGraphics = 
       Block[{$DisplayFunction = Identity}, 
         Show[ 
          MapThread[
           {ListPlot[#1, Joined-> If[ #4 === 1, True, False], PlotStyle->{#3} ], 
           Graphics[{ PointSize[0.02], #3, Point[#2], 
             If[Length[#1] > 1 && #4 === 1,
							 If[ oldGrs,
							   Arrow[#1[[-2]], #1[[-1]], HeadCenter -> 0.7],
							   Arrow[{#1[[-2]], #1[[-1]]}] ],
  	           {}] }] }&,
          {trekPointInfo, conditions, pathColors, trekDisplayType}], 
        AspectRatio -> (ymax-ymin)/(xmax-xmin) (yscale/xscale),
        PlotRange->{{xmin, xmax},{ymin, ymax}}, 
        Sequence @@ FilterRules[$FilteredEquationTrekkerOptions, Options[ListPlot]] ]];
        If[oldGrs, Show[plotGraphics]]
    ,
    plotGraphics = {}];

    If[$IsFirstOrder,
        originIndep = conditions[[All,1]];
        conditions = Take[conditions, All, -1]
    ];

    form = MapThread[Function[$TrekGenerator@"FormatTrek"[##]], {originIndep, conditions, rnges}];

    state = MapThread[
        TrekData[#1, #2, {#3, #4}, {#5, If[#6 == 1, "Line", "Points"]}]&,
        {form, conditions, originIndep, rnges, pathColors, trekDisplayType}];
      
    parms = TrekParameters /. $InputOptions /. TrekParameters -> {};
    parms = MapThread[resetPopt, {Sort[$ParameterRules], Sort[parms]}];
    useOpts = DeleteCases[$InputOptions, PlotRange -> _];
    useOpts = DeleteCases[useOpts, TrekParameters -> _];
    useOpts = DeleteCases[useOpts, "CanvasSize" -> _];
    canvasSize = {PropertyValue[{"canvas", "width"}], PropertyValue[{"canvas", "height"}]};
      
    state = EquationTrekkerState[ $InputData, $ParameterRules, state, Flatten[{TrekParameters -> parms, PlotRange -> {{xmin, xmax}, {ymin, ymax}}, 
         "CanvasSize" -> canvasSize, useOpts}] ];

   {state, plotGraphics}
   ];

 (*******************************
   CreateEndModalResults
   
   produces the Mathematica content
   returned to the kernel session when
   the EquationTrekker user interface 
   ends
   
   TODO need to include each trek's 
   independentRange
 *******************************)
 
CreateEndModalResults[] := If[$IsFailing, Return[$Failed], CreateTrekkerState[]];
 
(*******************************
   Utility functions
 *******************************)
 
(* Structure/type checking functions used for processing options *)

PositiveIntegerQ[x_] := And[IntegerQ[x], x > 0];
MRealQ[x_] := And[MachineNumberQ[x], Not[Head[x] === Complex]];

TensorDimensionQ[x_, {dim_}, pred_] := 
  And[VectorQ[x, pred], Length[x] == dim];
TensorDimensionQ[x_, {dim1_, dim2_}, pred_] := 
  And[Dimensions[x] === {dim1, dim2}, MatrixQ[x, pred]];
TensorDimensionQ[___] := False;

(* For displayed things to show without context being shown explicitly *)
Decontext[x_] := (x /. s_Symbol:>If[Context[s] === "System`",s,Symbol[SymbolName[s]]]);

(* Checking form of Parameter rules *)
ParameterRuleQ[prules_List] := 
  Catch[Scan[
      If[Not[TrueQ[ParameterRuleQ[#]]], Throw[False, ParameterRuleQ]] &, 
      prules]; True, ParameterRuleQ];

ParameterRuleQ[Rule[lhs_, rhs_]] := Not[NumericQ[lhs]];
    
(* Resets a Parameter Rule *)
resetPopt[p_->v_, p_->Parameter[spec__]] := Module[{new = Parameter[spec]},
    new[[1]] = v;
    p->new];

resetPopt[p_->v_, p_->{spec__}] := Module[{new = {spec}},
    new[[1]] = v;
    p->new];

resetPopt[p_->v_, p_->_] := p->v;

(* We have passed in the arguments to the top-level EquationTrekker function
   as a J/Link Expr list of Mathematica expressions *)
   
(*******************************
  Initializaton call to build up 
  runtime components based on arguments
 *******************************)

InitEquationTrekker @@ WidgetReference["##"];

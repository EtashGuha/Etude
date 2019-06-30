updateRules[] :=
    Module[ {},
        SetPropertyValue[{"methodRule", "items"}, 
               
              ToExpression[
                PropertyValue[{"methodStrategy", 
                  "selectedItem"}]] /. {Automatic -> {"Automatic", 
                  "ClenshawCurtisOscillatoryRule", 
                              "ClenshawCurtisRule", "GaussKronrodRule", 
                  "LobattoKronrodRule", "MonteCarloRule", "NewtonCotesRule", 
                  "TrapezoidalRule"}, 
                   
                GlobalAdaptive -> {"Automatic", "ClenshawCurtisOscillatoryRule", 
                  "ClenshawCurtisRule", "GaussKronrodRule", "LobattoKronrodRule", 
                              "NewtonCotesRule", "TrapezoidalRule"}, 
                LocalAdaptive -> {"Automatic", "ClenshawCurtisOscillatoryRule", 
                  "ClenshawCurtisRule", 
                              "GaussKronrodRule", "LobattoKronrodRule", 
                  "NewtonCotesRule", "TrapezoidalRule"}, 
                DoubleExponential -> {"Automatic"}, 
                          Trapezoidal -> {"Automatic"}, MonteCarlo -> {"Automatic"}, 
                AdaptiveMonteCarlo -> {"Automatic"}, 
                QuasiMonteCarlo -> {"Automatic"}, 
                          AdaptiveQuasiMonteCarlo -> {"Automatic"}, 
                DuffyCoordinates -> {"Automatic", "ClenshawCurtisOscillatoryRule", 
                  "ClenshawCurtisRule", 
                              "GaussKronrodRule", "LobattoKronrodRule", 
                  "NewtonCotesRule", "TrapezoidalRule"}, 
                   
                Oscillatory -> {"Automatic", "ClenshawCurtisOscillatoryRule", 
                  "ClenshawCurtisRule", "GaussKronrodRule", "LobattoKronrodRule", 
                              "MonteCarloRule", "NewtonCotesRule", 
                  "TrapezoidalRule"}}];
    ]


updateMonitor[] :=
    Module[ {inputcode, method, selstrategy, symbolic, selrule, methodcode, accgoal, precgoal, workprec, optioncode, maxpts, minrec, maxrec},
        selstrategy = 
         ToExpression[PropertyValue[{"methodStrategy", "selectedItem"}]];
        selrule = ToExpression[PropertyValue[{"methodRule", "selectedItem"}]];
        method = If[ MatchQ[selstrategy, Automatic],
                     selrule,
                     If[ MatchQ[selrule, Automatic],
                         selstrategy,
                         {selstrategy, Method -> selrule}
                     ]
                 ];
        Which[PropertyValue[{"symbnone", "selected"}] || 
          PropertyValue[{"symbauto", "selected"}], 
                 (SetPropertyValue[{#1, "Enabled"}, 
              "False"] & ) /@ {"symbevenodd", "symbpiece", "symboscildet", 
            "ucauto", "uctrue", "ucfalse"}; , 
              
      PropertyValue[{"symbcustom", 
        "selected"}], (SetPropertyValue[{#1, "Enabled"}, 
           "True"] & ) /@ {"symbevenodd", "symbpiece", "symboscildet", 
                    "ucauto", "uctrue", "ucfalse"}; ];
        Which[
             PropertyValue[{"symbnone", "selected"}], 
            methodcode = StringJoin["Method->", ToString[If[Head[method] === List, Append[method,SymbolicProcessing->0],{method,SymbolicProcessing->0}]]]
            , 
            PropertyValue[{"symbauto", "selected"}], 
            methodcode = StringJoin["Method->", ToString[method]]
            , 
            PropertyValue[{"symbcustom", "selected"}], 
            methodcode = StringJoin["Method->", "{", "SymbolicPreprocessing", ",", "EvenOddSubdivision->", 
                              ToString[PropertyValue[{"symbevenodd", "selected"}]], ",", "OscillatorySelection->", 
                               ToString[PropertyValue[{"symboscildet", "selected"}]], ",", "SymbolicPiecewiseSubdivision->", 
                            ToString[PropertyValue[{"symbpiece", "selected"}]], ",", "UnitCubeRescaling->", 
                                  Which[PropertyValue[{"ucauto", "selected"}], "Automatic", 
                                      PropertyValue[{"uctrue", "selected"}], "True", 
                                      PropertyValue[{"ucfalse", "selected"}], "False"], ",", 
                            "Method->", ToString[method], "}"]
            ];
        accgoal = ToExpression[PropertyValue[{"accGoal", "text"}]];
        precgoal = ToExpression[PropertyValue[{"precGoal", "text"}]];
        workprec = ToExpression[PropertyValue[{"workPrec", "text"}]];
        maxpts = ToExpression[PropertyValue[{"maxPts", "text"}]];
        minrec = ToExpression[PropertyValue[{"minRec", "text"}]];
        maxrec = ToExpression[PropertyValue[{"maxRec", "text"}]];
        If[ Or[ And[NumericQ[maxrec], maxrec >= 0], maxrec === Automatic],
        If[ Or[ And[NumericQ[minrec], minrec >= 0], minrec === Automatic],
        If[ Or[ And[NumericQ[maxpts], maxpts >= 0], maxpts === Automatic], 
        		If[ Or[NumericQ[accgoal],accgoal===Infinity,accgoal===Automatic],
    If[ Or[NumericQ[precgoal],precgoal===Infinity,precgoal===Automatic],
        If[ NumericQ[workprec],
            optioncode = StringJoin["AccuracyGoal->",ToString[accgoal],",","PrecisionGoal->",ToString[precgoal],",","WorkingPrecision->",ToString[workprec],
            						",","MaxPoints->",ToString[maxpts],",","MinRecursion->",ToString[minrec],",","MaxRecursion->",ToString[maxrec]];
            inputcode = 
      StringJoin["NIntegrate[",PropertyValue[{"integrandField", "text"}], ",", 
       PropertyValue[{"regionField", "text"}], ",", methodcode, ",", optioncode,"]"];
            SetPropertyValue[{"inputcodeField", "text"}, ToString[inputcode]];,
            SetPropertyValue[{"canvas", "mathCommand"}, 
         Show[Graphics[Text["Check WorkingPrecision and try again"], ImageSize -> {300, 300},AspectRatio->Full]]];
         SetPropertyValue[{"inputcodeField", "text"}, ToString[Null]]
        ],
        SetPropertyValue[{"canvas", "mathCommand"}, 
         Show[Graphics[Text["Check Precision and try again"], ImageSize -> {300, 300},AspectRatio->Full]]];
         SetPropertyValue[{"inputcodeField", "text"}, ToString[Null]]
    ],
    SetPropertyValue[{"canvas", "mathCommand"}, 
         Show[Graphics[Text["Check Accuracy and try again"], ImageSize -> {300, 300},AspectRatio->Full]]];
         SetPropertyValue[{"inputcodeField", "text"}, ToString[Null]]
]
         ,
         SetPropertyValue[{"canvas", "mathCommand"}, 
         Show[Graphics[Text["Check MaxPoints and try again"], ImageSize -> {300, 300},AspectRatio->Full]]];
         SetPropertyValue[{"inputcodeField", "text"}, ToString[Null]]]
        ,
        SetPropertyValue[{"canvas", "mathCommand"}, 
         Show[Graphics[Text["Check MinRecursion and try again"], ImageSize -> {300, 300},AspectRatio->Full]]];
         SetPropertyValue[{"inputcodeField", "text"}, ToString[Null]]
        ]
        ,
        SetPropertyValue[{"canvas", "mathCommand"}, 
         Show[Graphics[Text["Check MaxRecursion and try again"], ImageSize -> {300, 300},AspectRatio->Full]]];
         SetPropertyValue[{"inputcodeField", "text"}, ToString[Null]]]
]

numericIntegrate[] :=
    Module[ {code, int, ranges, numres, vars, res, t, timelim},
        If[PropertyValue[{"inputcodeField","text"}] != "Null",
        SetPropertyValue[{"canvas", "mathCommand"}, 
         Show[Graphics[Text["Calculating..."], ImageSize -> {300, 300},AspectRatio->Full]]];
        InvokeMethod[{"canvas", "repaintNow"}];
        int = ToExpression[PropertyValue[{"integrandField", "text"}]];
        ranges = ToExpression[
          ToString[{PropertyValue[{"regionField", "text"}]}]];
        vars = First /@ ranges;
        timelim = ToExpression[PropertyValue[{"timeConstraint", "text"}]];
        res = Insert[ToExpression[
                     PropertyValue[{"inputcodeField", "text"}],InputForm,Hold],EvaluationMonitor :> Sow[vars],{1,-1}];
        If[NumericQ[timelim] && timelim > 0,
		res = TimeConstrained[Reap[res[[1]]], timelim];
        Check[numres = res[[1]];
        t = res[[2,1]];
        ptstoplot = N[t];
        SetPropertyValue[{"resultField", "text"}, ToString[numres, InputForm]];
        samplePlotting[],SetPropertyValue[{"canvas", "mathCommand"}, 
         Show[Graphics[Text["Please check integrand and region are valid and try again"], ImageSize -> {300, 300},AspectRatio->Full]]];
         SetPropertyValue[{"resultField", "text"}, ToString[Null]]], 
		SetPropertyValue[{"canvas", "mathCommand"}, 
         Show[Graphics[Text["Time limit must be numeric greater than 0"], ImageSize -> {300, 300},AspectRatio->Full]]];
	],
	SetPropertyValue[{"resultField", "text"}, ToString[Null]]]
]

samplePlotting[] :=
    Module[ {plotter, t},
    If[ Length[ptstoplot] == 0,
        plotter = "Please edit the input and then press Evaluate",
        SetPropertyValue[{"canvas", "mathCommand"}, 
             ToString[Show[Graphics[Text["Rendering..."], ImageSize -> {300, 300}, AspectRatio->Full]], InputForm]];
        InvokeMethod[{"canvas", "repaintNow"}];
        Which[Last[Dimensions[ptstoplot]] == 1, t = Flatten[ptstoplot];
                                                plotter = ListPlot[Transpose[{N[t], Range[Length[t]]}],AspectRatio->1.5], 
             Last[Dimensions[ptstoplot]] == 2, 
                 
             plotter = 
              ListPlot[ptstoplot,AspectRatio->1.5], 
             Last[Dimensions[ptstoplot]] == 3, 
                 
             plotter = 
              ListPointPlot3D[ptstoplot],
             Last[Dimensions[ptstoplot]] > 3,
             plotter = SetPropertyValue[{"canvas", "mathCommand"}, 
         Show[Graphics[Text["Integration over more than three dimensions, graphic not available"], ImageSize -> {300, 300},AspectRatio->Full]]]
    ];
    SetPropertyValue[{"canvas", "mathCommand"}, 
                     ToString[plotter, InputForm]];
    InvokeMethod[{"canvas", "repaintNow"}];
]
]

notebookCreate[] := 
	Module[{guinb, nbfind}, 
		nbfind = Select[Notebooks[], (CurrentValue[#, WindowTitle] == "NIntegrate Explorer Graphic") &];
		If[nbfind == {},
		guinb = NotebookCreate[WindowTitle -> "NIntegrate Explorer Graphic", WindowMargins -> {{50, Automatic}, {Automatic, 50}}], 
		guinb = nbfind[[1]]
	];
	SelectionMove[guinb,After,Notebook];
	NotebookWrite[guinb, ToBoxes[ToExpression[PropertyValue[{"canvas", "mathcommand"}]]]];
	SelectionMove[guinb,Before,Cell];
	]
	     
resetMethod[] := 
	Module[{}, 
		SetPropertyValue[{"methodStrategy", "selectedItem"},"Automatic"];
		SetPropertyValue[{"symbauto", "selected"},True]
	]      
            
(*This section contains all of the BindEvents*)

BindEvent[{"integrandField","Action"}, Script[updateMonitor[]]]
BindEvent[{"regionField","Action"}, Script[updateMonitor[]]]
BindEvent[{"accGoal","Action"}, Script[updateMonitor[]]]
BindEvent[{"precGoal","Action"}, Script[updateMonitor[]]]
BindEvent[{"workPrec","Action"}, Script[updateMonitor[]]]
BindEvent[{"Evaluate","Action"}, Script[updateMonitor[];numericIntegrate[]]]
BindEvent[{"graphNotebook","Action"}, Script[notebookCreate[]]]
BindEvent[{"methodStrategy","Action"}, Script[updateRules[]]]
BindEvent[{"methodRule","Action"}, Script[updateMonitor[]]]
BindEvent[{"symbauto","Action"}, Script[updateMonitor[]]]
BindEvent[{"symbnone","Action"}, Script[updateMonitor[]]]
BindEvent[{"symbcustom","Action"}, Script[updateMonitor[]]]
BindEvent[{"symbevenodd","Change"}, Script[updateMonitor[]]]
BindEvent[{"symbpiece","Change"}, Script[updateMonitor[]]]
BindEvent[{"symboscildet","Change"}, Script[updateMonitor[]]]
BindEvent[{"ucauto","Action"}, Script[updateMonitor[]]]
BindEvent[{"uctrue","Action"}, Script[updateMonitor[]]]
BindEvent[{"ucfalse","Action"}, Script[updateMonitor[]]]
BindEvent[{"Reset","Action"}, Script[resetMethod[]]]
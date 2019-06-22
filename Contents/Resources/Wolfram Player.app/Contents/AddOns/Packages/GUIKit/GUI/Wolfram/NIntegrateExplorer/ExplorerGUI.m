

(* Basic Mathematica Code File *)

Widget["Frame", 
  {"title" -> "NIntegrate Explorer", 
     WidgetGroup[
      {
        WidgetGroup[
         {
             WidgetGroup[
              {
               WidgetGroup[
                   {Widget["Label", 
                           {"Text" -> "Integrand"}], 
                    WidgetAlign[], 
                    Widget["TextField", 
                            {"text" -> "1/Sqrt[x]", "Columns" -> "15"}, 
                           Name -> "integrandField"]
                   }, WidgetLayout -> Row], 
              
               WidgetGroup[
                   {Widget["Label", 
                           {"text" -> "Region"}], 
                    WidgetAlign[], 
                   Widget["TextField", {"text" -> "{x,0,1}", "Columns" -> "15"}, 
                           Name -> "regionField"]
                   }, WidgetLayout -> Row]
                
                }, WidgetLayout -> {"Grouping" -> Column, "Border" -> "Input"}], 
               
            WidgetGroup[
             {
              WidgetGroup[
                  {Widget["Label", 
                          {"text" -> "Accuracy"}], 
                    Widget["TextField", 
                            {"text" -> "Infinity", "Columns" -> "4"}, 
                            Name -> "accGoal"],
                    Widget["Label", 
                           {"text" -> "Precision"}],
                    Widget["TextField", 
                            {"text" -> "Automatic", "Columns" -> "6"}, 
                            Name -> "precGoal"]
               }, WidgetLayout -> Row], 
               
              WidgetGroup[
                  {Widget["Label", 
                          {"text" -> "Working Precision"}],
                    Widget["TextField", 
                            {"text" -> "MachinePrecision", "Columns" -> "8"}, 
                            Name -> "workPrec"]
                            }, WidgetLayout -> Row], 
               
               WidgetGroup[
               		{Widget["Label", 
                          {"text" -> "MaxPoints"}],
                     Widget["TextField", 
                            {"text" -> "Automatic", "Columns" -> "4"}, 
                            Name -> "maxPts"],
                     Widget["Label", 
                           {"Text" -> "Time Limit"}], 
                     Widget["TextField", {"text" -> "20"}, Name -> "timeConstraint"]
               		}, WidgetLayout -> Row]
                  }, 
                  WidgetLayout -> {"Grouping" -> Column, "Border" -> "Precision"}], 
               
              WidgetGroup[
              		{Widget["Label", 
                          {"text" -> "Min"}],
                     Widget["TextField", 
                            {"text" -> "0", "Columns" -> "5"}, 
                            Name -> "minRec"],
                     Widget["Label", 
                           {"Text" -> "Max"}], 
                     Widget["TextField", {"text" -> "Automatic", "Columns" -> "5"}, Name -> "maxRec"]
               		}, WidgetLayout -> {"Grouping" -> Row, "Border" -> "Recursion" }
              ],
             
              WidgetGroup[
                  {Widget["Button", 
                          {"text" -> "Evaluate"}, 
                         Name -> "Evaluate"], 
                  Widget["Button", 
                          {"text" -> "Graphics Notebook"},
                          Name -> "graphNotebook"]
                  }, WidgetLayout -> {"Grouping" -> Row, "Border" -> ""}],
              WidgetGroup[
              {
              }
              ]
          },  WidgetLayout -> Column], 
           
        WidgetGroup[
         {
          WidgetGroup[
           {
            WidgetGroup[
                {Widget["Label", 
                        {"text" -> "Strategy"}], 
                   WidgetAlign[], 
                    Widget["ComboBox", 
                            {"items" -> {"Automatic", "GlobalAdaptive", "LocalAdaptive", "DoubleExponential", "Trapezoidal", 
                             "MonteCarlo", "AdaptiveMonteCarlo", "QuasiMonteCarlo", "AdaptiveQuasiMonteCarlo", 
                             "DuffyCoordinates", "Oscillatory"}}, 
                           Name -> "methodStrategy"]
                    }, WidgetLayout -> Row], 
                    
               WidgetGroup[
                   {Widget["Label", 
                           {"text" -> "Rule"}], 
                    WidgetAlign[], 
                    Widget["ComboBox", 
                            {"items" -> Script[{"Automatic", "ClenshawCurtisOscillatoryRule", "ClenshawCurtisRule", "GaussKronrodRule", 
                          "LobattoKronrodRule", "MonteCarloRule", "NewtonCotesRule", "TrapezoidalRule"}]}, 
                          Name -> "methodRule"]
                  }, WidgetLayout -> Row]
                
                }, WidgetLayout -> {"Border" -> "", "Grouping" -> Column}], 
               
              WidgetGroup[
               {
                WidgetGroup[
                    {Widget["RadioButton", 
                           {"text" -> "Automatic", "selected" -> True}, 
                           Name -> "symbauto"], 
                   Widget["RadioButton", 
                           {"text" -> "None", "selected" -> False}, 
                           Name -> "symbnone"], 
                   Widget["RadioButton", 
                           {"text" -> "Custom", "selected" -> False}, 
                           Name -> "symbcustom"], 
                    Widget["ButtonGroup", 
                         {WidgetReference["symbauto"], WidgetReference["symbnone"], WidgetReference["symbcustom"]}], 
                    Widget["CheckBox", 
                            {"text" -> "EvenOddSubdivision", "Enabled" -> "False", "selected" -> "True"}, 
                        Name -> "symbevenodd"], 
                   Widget["CheckBox", 
                           {"text" -> "SymbolicPiecewiseSubdivision", "Enabled" -> "False", "selected" -> "True"}, 
                          Name -> "symbpiece"], 
                   Widget["CheckBox", 
                           {"text" -> "OscillatorySelection", "Enabled" -> "False", "selected" -> "True"}, 
                        Name -> "symboscildet"]
                }, WidgetLayout -> {"Grouping" -> Column, "Border" -> "SymbolicProcessing"}], 
                  
                WidgetGroup[
                	{WidgetGroup[
                    {Widget["RadioButton", 
                           {"text" -> "Automatic", "selected" -> True, "Enabled" -> "False"},
                           Name -> "ucauto"], 
                    Widget["RadioButton", 
                            {"text" -> "True", "selected" -> False, "Enabled" -> "False"}, 
                         Name -> "uctrue"], 
                   Widget["RadioButton", 
                           {"text" -> "False", "selected" -> False, "Enabled" -> "False"}, 
                           Name -> "ucfalse"], 
                   Widget["ButtonGroup", 
                           {WidgetReference["ucauto"], WidgetReference["uctrue"], WidgetReference["ucfalse"]}]
                   }, WidgetLayout -> {"Grouping" -> Column, "Border" -> "UnitCube"}],
                   Widget["Button", 
                          {"text" -> "Reset"}, 
                         Name -> "Reset"]
                      },WidgetLayout->{"Grouping" -> Column}
                   ]
                  
                }, WidgetLayout -> Row]
          }, WidgetLayout -> {"Border" -> "Method Options", "Grouping" -> Column}]
         }
        ], 
     WidgetGroup[
              {Widget["TextPanel", 
                        {"editable" -> "False", "preferredSize" -> Widget["Dimension", {"width" -> 50, "height" -> 60}]}, 
                      Name -> "inputcodeField"]
              }, WidgetLayout -> {"Grouping"->Row, "Border"->"Input Code"}],
     WidgetGroup[
         {WidgetGroup[
              {Widget["Label", 
                      {"text" -> "Result"}], 
               WidgetAlign[], 
               Widget["TextField", 
                       {"Columns" -> "10", "editable" -> "false"}, 
                     Name -> "resultField"]
             }, WidgetLayout -> Row], 
          
          Widget["MathPanel", 
                  {"preferredSize" ->  
                      Widget["Dimension", 
                                {"width" -> 300, "height" -> 300}], "usesFE" -> True, 
                                BindEvent["ComponentResized", Script[samplePlotting[]]]
                    }, WidgetLayout -> {"Stretching" -> {Maximize, Maximize}}, 
                   Name -> "canvas"]
          
          }, WidgetLayout -> {"Grouping" -> Column, "Border" -> ""}], 
          
  Script[ {}, ScriptSource -> "ExplorerScript.m" ]}]

     

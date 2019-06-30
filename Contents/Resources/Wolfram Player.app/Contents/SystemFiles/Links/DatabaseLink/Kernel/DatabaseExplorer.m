(* Mathematica package *)

DatabaseExplorer::usage = "DatabaseExplorer[ ] launches a graphical user interface to DatabaseLink.";

Begin["`DatabaseExplorer`Private`"]

(*
 * Defaults for new query.
 *)
newQuerySkeleton[o:OptionsPattern[]] := Module[{thing},
	thing[queryIndex] = 0;
	thing[conn] = Null;
	newQuerySkeleton[thing, o]
];
newQuerySkeleton[last_, o:OptionsPattern[]] := Module[
	{
		qi = last[queryIndex] + 1,
		q,
		condCols, cond2Cols, cond2ColsDisp, condOpers,
		sortCols, sortOrders,
		
		myConn, myTables, myColumnList
	},
	myConn = last[conn];
	myTables = If[!SQLConnectionOpenQ@myConn, {}, With[{tabs = SQLTableNames[myConn]}, Take[tabs, Min[Length@tabs, 1]]]];
	myColumnList = If[myConn === Null, {}, With[
		{cols = Flatten[SQLColumnNames[myConn, #] & /@ myTables, 1]},
		Take[cols, Min[Length@cols, 1]]]
	];
	
	q[queryIndex] = qi;
	q[limit] = True;
	q[limitValue] = OptionValue["MaxRows"];
	q[timeout] = True;
	q[timeoutValue] = OptionValue["Timeout"];
	q[distinct] = OptionValue["Distinct"];
	q[sch] = OptionValue["ShowColumnHeadings"];
	q[queryTitle] = "Untitled-" <> ToString[qi];
	q[conn] = myConn;
	q[tables] = myTables;
	q[columnList] = myColumnList;
	q[condOp] = And;
	(* Enable join conditions *)
	q[condBool] =  False;

	q[sortBool] = False;

	q[gas] = OptionValue["GetAsStrings"];
	(* Don't try any queries on initialization *)
	q[resultSet] = {};
	
	(* If this is Null, the query has no disk presence. *)
	q[filename] = OptionValue["Location"];
	(* If True, query needs saving. *)
	q[stale] = True;
	q[scanQuery] = OptionValue["ScanQueryFiles"];
	
	(*
	 * M-- can't do part assignments to downvalues (like a[foo][[1]] = bar)
	 * so for cases where we need to mutate elements in a list of stuff associated with the query
	 * we need a separate var for it.
	 *)
    condCols = {Null};
    cond2Cols = {Null};
    cond2ColsDisp = {Null};
    condOpers = {"=="};

    sortCols = {Null};
    sortOrders = {"Ascending"};

	{q, condCols, cond2Cols, cond2ColsDisp, condOpers, sortCols, sortOrders}
];


(*
 * De-serialize an SQLSelect statement into something we can populate the gui with.
 *)
SetAttributes[canonicalOptions, {Listable}];
canonicalOptions[name_Symbol -> val_] := SymbolName[name] -> val;
canonicalOptions[expr___] := expr;

deconstructQuery[SQLSelect[connName_String,
                          table:(_SQLTable | {___SQLTable} | _String | {___String}),
                          columns:(_SQLColumn | {___SQLColumn}),
                          condition_,
                          rawOpts___Rule], 
    con_SQLConnection,
    i_Integer,
    o:OptionsPattern[]] := Module[
    {
        opts = List@canonicalOptions[rawOpts],
        q,
        condCols, cond2Cols, cond2ColsDisp, condOpers,
        sortCols, sortOrders,
        
        sortRaw, wc
    },
    q[queryIndex] = i;
    q[filename] = "Location" /. opts /. o;
    q[limit] = True;
    q[limitValue] = "MaxRows" /. opts /. o;
    q[timeout] = True;
    q[timeoutValue] = "Timeout" /. opts /. o;
    q[distinct] = "Distinct" /. opts /. o;
    q[sch] = "ShowColumnHeadings" /. opts /. o;
    q[gas] = "GetAsStrings" /. opts /. o;
    
    q[queryTitle] = "Name" /. opts /. "Name" -> FileBaseName[q[filename]];
    q[conn] = con;
    q[tables] = Flatten[{table /. SQLTable[x_] :> x}];
    q[columnList] = Flatten[{columns /. SQLColumn[x_] :> x}, 1];

    q[condOp] = And;
    (* Enable join conditions *)
    q[condBool] =  False;

    wc = inferWildcard[con];
    If[MatchQ[condition, None | Null],
        q[condOp] = And;
        q[condBool] = False;
	    condCols = {Null};
	    cond2Cols = {Null};
	    cond2ColsDisp = {Null};
	    condOpers = {"=="};,
	    
	    (* else *)
	    q[condOp] = Head[condition];
	    q[condBool] = True;
        condCols = (List @@ condition)[[All, 1]] /. SQLColumn[x_] :> x;
        cond2Cols = (List @@ condition)[[All, 2]] /. x_String :> 
            First@StringCases[x, RegularExpression["^" <> wc <> "?(.*?)" <> wc <> "?$"] -> "$1"];
        cond2ColsDisp = cond2Cols /. SQLColumn[x_] :> ToString@Row[x, ","];
	    condOpers = Function[{cond}, Switch[cond,
	        SQLStringMatchQ[_, s_String] /; StringMatchQ[s, wc ~~ ___ ~~ wc],
	            SQLStringMatchQ[#1, wildcard <> #2 <> wildcard] &,
	        !SQLStringMatchQ[_, s_String] /; StringMatchQ[s, wc ~~ ___ ~~ wc],
	            !SQLStringMatchQ[#1, wildcard <> #2 <> wildcard] &,
	        SQLStringMatchQ[_, s_String] /; StringMatchQ[s, ___ ~~ wc],
	            SQLStringMatchQ[#1, #2 <> wildcard] &,
	        SQLStringMatchQ[_, s_String] /; StringMatchQ[s, wc ~~ ___],
	            SQLStringMatchQ[#1, wildcard <> #2] &,
	        _, cond[[0]]
	    ]] /@ (List @@ condition);
    ];
    
    sortRaw = "SortingColumns" /. opts /. o;
    If[MatchQ[sortRaw, {_Rule ..}],
    	q[sortBool] = True;
        sortCols = sortRaw[[All, 1, 1]];
        sortOrders = sortRaw[[All, 2]];,
        (* else *)
        q[sortBool] = False;
	    sortCols = {Null};
	    sortOrders = {"Ascending"};
    ];

    (* Don't try any queries on initialization *)
    q[resultSet] = {};
    q[queryArgs] = constructQuery[q, condOpers, condCols, cond2Cols, sortOrders, sortCols];
    q[stale] = False;
    
    {q, condCols, cond2Cols, cond2ColsDisp, condOpers, sortCols, sortOrders}
];


(*
 * Construct query
 *)
SetAttributes[constructQuery, HoldFirst];
constructQuery[q_, cO_, cC_, c2C_, sO_, sC_] := Module[
    {args, theWildcard = inferWildcard[q[conn]],
        condArgs = DeleteCases[Transpose[{cO, cC, c2C}], {Null, _, _}|{_, Null, _}],
        sortArgs = DeleteCases[Transpose[{sC, sO}], {Null, _}]
    },
    args = {
        q[conn],
        q[tables],
        q[columnList],
        Sequence @@ If[q[condBool] === True && Length[condArgs] > 0, 
            List[q[condOp] @@ Map[
                (#[[1]] /. wildcard -> theWildcard)[SQLColumn[#[[2]]], #[[3]]] &,
                condArgs
            ]],
            (* else *)
            {None}
        ],
        "SortingColumns" -> If[q[sortBool] && Length[sortArgs] > 0,
            Map[
                Rule[SQLColumn[#[[1]]], #[[2]]] &,
                sortArgs
            ],
            (* else *)
            "SortingColumns" /. Options[SQLSelect]
        ], 
        "MaxRows" -> If[q[limit], q[limitValue], "MaxRows" /. Options[SQLSelect]],
        "Timeout" -> If[q[timeout], q[timeoutValue], "Timeout" /. Options[SQLSelect]], 
        "Distinct" -> q[distinct],
        "GetAsStrings" -> q[gas],
        "ShowColumnHeadings" -> q[sch],
        "ScanQueryFiles" -> q[scanQuery]
    };
    If[q[queryArgs] =!= args, q[stale] = True];
    (*
    Print[{q[queryArgs], args}];
    Print@q[condOp];
    Print@sC;
    Print@sO;*)
    (*Print@args;*)
    args
];

executeQuery[queryArgs_] := Check[
    SQLSelect @@ queryArgs
    ,
    $Failed
];

constructAndExecute[args__] := With[{quargs = constructQuery[args]},
    (*Print@StringForm["executing, args = ``", quargs];*)
    {quargs, executeQuery[quargs]}
];



(* For Tables and Columns pane heights *)
$showAdvPaneHeights = {True -> 140, _ -> 365};

Options[DatabaseExplorer] = Options[newQuerySkeleton] = Options[deconstructQuery] = {
    "Location" -> Null,
    "MaxRows" -> 100,
    "Timeout" -> 10,
    "Distinct" -> False,
    "ShowColumnHeadings" -> True,
    "GetAsStrings" -> False,
    "SortingColumns" -> None, 
    "ScanQueryFiles" -> True
};

DatabaseExplorer[o:OptionsPattern[]] := CreateDialog[DynamicModule[
	{
	    (* Lists of query properties *)
	    qq, condCols, cond2Cols, cond2ColsDisp, condOpers, sortCols, sortOrders,
	    
	    (* Current query index *)
	    cqi = -1,
	    
	    saveQuery, sdiFn, validSelectionsQ,
	    
	    showAdvancedText = "Show Advanced Options",
	    showAdvancedToggle = False,
	    
	    existing,
		(* Connections opened in this explorer *)
		connectionsOpened = {},
        (* Save as ... *)
		i, qt
	},
    existing = MapIndexed[
        	With[{conn = OpenSQLConnection[First@#1]},
        		If[SQLConnectionOpenQ[conn],
        			AppendTo[connectionsOpened, conn];
        			deconstructQuery[#1, conn, First@#2, Options[SQLSelect]],
        			$Failed
        		]
        	] &,
        	SQLQueries[]
        ] /. $Failed -> Sequence[];

    {qq, condCols, cond2Cols, cond2ColsDisp, condOpers, sortCols, sortOrders} = 
	    If[Length@existing == 0,
	        List /@ newQuerySkeleton[Join[Flatten@{o}, Options[DatabaseExplorer]]],
	        (* else *)
	        Transpose[existing]
	    ];
    cqi = Length@qq;

    (*
     * Saveout mechanics ...
     *)
    saveQuery[fpath_, q:SQLSelect[args__, op:OptionsPattern[]]] := Module[
        {err, revOpts},
        (* Previous version of DatabaseLink had a description field etc.
         * Option rewriting machinery retained.
         *)
        revOpts = DeleteDuplicates[
        	Join[{
        		"Version" -> DatabaseLink`Information`$VersionNumber
        	   },
        	   Flatten[{op}]
        	],
        	First@#1 === First@#2 &
        ];
        If[Quiet@Check[
            Put[SQLSelect[args, Sequence @@ revOpts], fpath];
            err = $MessageList;
            False,
            
            True],
            
            CreateDialog[{
                Column[{
                    StringForm["Error saving query to"],
                    Style[fpath <> ":"],
                    ToString[Style[First@err, {Red, Italic}], StandardForm],
                    DefaultButton[NotebookClose[EvaluationNotebook[]];]
                }, Center]
            }, WindowFloating -> True, Modal -> True];
            Null,
            (* else *)
            True
        ]
    ];


    (*
     * Macro used for enabling a few operations and UI elements
     *)
    validSelectionsQ[] := Quiet@SQLConnectionOpenQ[qq[[cqi]][conn]] &&
        (*!FreeQ[SQLTableNames[qq[[cqi]][conn]], Alternatives @@ qq[[cqi]][tables]] &&
        !FreeQ[SQLColumnNames[qq[[cqi]][conn]], Alternatives @@ qq[[cqi]][columnList]] &&*)
        Intersection[qq[[cqi]][columnList][[All, 1]], qq[[cqi]][tables]] =!= {};

    Style[Column[{
    	Row[{
        (* Menu bar for function buttons *)
        Dynamic@Row[{
            (* Menu button for creating a new database connection. *)

            Button[
                Tooltip[Import["DatabaseLink/GUI/DatabaseLink/Images/New16.gif"], "Connect to a data source"],
                (* Opens the dialog window to connect to a database. *)
                With[{c = OpenSQLConnection[]},
                	If[c =!= $Canceled && SQLConnectionOpenQ[c],
                        connectionsOpened = DeleteDuplicates[Append[connectionsOpened, c]];
                        qq[[cqi]][conn] = c;
	                    With[{tabs = SQLTableNames[c]},
	                    	qq[[cqi]][tables] = Take[tabs, Min[Length@tabs, 1]];
	                    	With[{cols = Flatten[SQLColumnNames[c, #] & /@ tabs, 1]},
	                            qq[[cqi]][columnList] = Take[cols, Min[Length@cols, 1]];
	                    	]
	                    ]
                	]
                ],
                Appearance -> "Palette", ImageSize -> {35, 25}, Method -> "Queued"
            ],

            (* Menu button for creating a new query. *)
            Button[
                Tooltip[Import["DatabaseLink/GUI/DatabaseLink/Images/Add16.gif"], "Create a new query"],
                With[{qEtc = newQuerySkeleton[Last@qq]},
                    AppendTo[qq, qEtc[[1]]];
                    AppendTo[condCols, qEtc[[2]]];
                    AppendTo[cond2Cols, qEtc[[3]]];
                    AppendTo[cond2ColsDisp, qEtc[[4]]];
                    AppendTo[condOpers, qEtc[[5]]];
                    AppendTo[sortCols, qEtc[[6]]];
                    AppendTo[sortOrders, qEtc[[7]]];
                    cqi = Length@qq;
                ],
                Appearance -> "Palette", ImageSize -> {35, 25}
            ],

            (* Menu button for saving a query. *)
            Button[
                Tooltip[Import["DatabaseLink/GUI/DatabaseLink/Images/Save16.gif"], "Save query"],
                qq[[cqi]][queryArgs] = constructQuery @@ Map[#[[cqi]] &, {qq, condOpers, condCols, cond2Cols, sortOrders, sortCols}];
                sdiFn = qq[[cqi]][filename] /. Null :>
                    SystemDialogInput["FileSave", FileNameJoin[{$UserBaseDirectory, "DatabaseResources", qq[[cqi]][queryTitle] <> ".m"}]];
                If[sdiFn =!= $Canceled,
	                If[TrueQ@saveQuery[sdiFn, SQLSelect @@ ReplacePart[qq[[cqi]][queryArgs], 1 -> ("Name" /. Options@qq[[cqi]][conn])]],
	                    qq[[cqi]][filename] = sdiFn;
	                    qq[[cqi]][queryTitle] = FileBaseName[sdiFn];
	                    qq[[cqi]][stale] = False;
	                    (* else *)
	                ]
                ],
                Appearance -> "Palette", ImageSize -> {35, 25}, Method-> "Queued"
            ],
            
            (* Menu button for save as. *)
            Button[
                Tooltip[Import["DatabaseLink/GUI/DatabaseLink/Images/SaveAs16.gif"], "Save query as..."],

                qt = qq[[cqi]][queryTitle];
                x = cqi;
                AppendTo[qq, dvClone[qq[[cqi]]]];
                AppendTo[condCols, condCols[[cqi]]];
                AppendTo[cond2Cols, cond2Cols[[cqi]]];
                AppendTo[cond2ColsDisp, cond2ColsDisp[[cqi]]];
                AppendTo[condOpers, condOpers[[cqi]]];
                AppendTo[sortCols, sortCols[[cqi]]];
                AppendTo[sortOrders, sortOrders[[cqi]]];
                cqi = Length@qq;

                qq[[cqi]][queryIndex] = cqi;
                qq[[cqi]][stale] = True;
                qq[[cqi]][filename] = Null;

		        i = 1;
		        While[FileNames[qq[[cqi]][queryTitle] <> ".m", FileNameJoin[{$UserBaseDirectory, "DatabaseResources"}]] =!= {},
		            qq[[cqi]][queryTitle] = qt <> " (" <> ToString[++i] <> ")"
		        ];
		        
                qq[[cqi]][queryArgs] = constructQuery @@ Map[#[[cqi]] &, {qq, condOpers, condCols, cond2Cols, sortOrders, sortCols}];

                With[{fpath = qq[[cqi]][filename] /. Null -> FileNameJoin[{$UserBaseDirectory, "DatabaseResources", qq[[cqi]][queryTitle] <> ".m"}]},
                    sdiFn = SystemDialogInput["FileSave", fpath];
                    If[sdiFn =!= $Canceled,
                    	If[TrueQ@saveQuery[sdiFn, SQLSelect @@ ReplacePart[qq[[cqi]][queryArgs], 1 -> ("Name" /. Options@qq[[cqi]][conn])]],
                    		qq[[cqi]][filename] = sdiFn;
                    		qq[[cqi]][queryTitle] = FileBaseName[sdiFn];
                    		qq[[cqi]][stale] = False;
                    		(* else *)
                    	]
                    ]
                ],
                Appearance -> "Palette", ImageSize -> {35, 25}, Method-> "Queued"
            ],

            (* Menu button for deleting a query. *)
            Button[
                Tooltip[Import["DatabaseLink/GUI/DatabaseLink/Images/Delete16.gif"], "Delete query"], 

                (* Confirmation ... *)
                CreateDialog[
		            Grid[{
		                {"Are you sure you want to delete this query?"},
		                {ChoiceButtons[{
		                	Which[
		                        qq[[cqi]][filename] =!= Null && Quiet@DeleteFile[qq[[cqi]][filename]] === $Failed,
		                        NotebookClose[EvaluationNotebook[]];
					            CreateDialog[{
					                Column[{
					                    StringForm["Error deleting query at"],
					                    Style[qq[[cqi]][filename]],
					                    DefaultButton[NotebookClose[EvaluationNotebook[]];]
					                }, Center]
					            }, WindowFloating -> True, Modal -> True],
						        
						        True,
	                            qq = Drop[qq, {cqi}];
	                            condCols = Drop[condCols, {cqi}];
	                            cond2Cols = Drop[cond2Cols, {cqi}];
	                            cond2ColsDisp = Drop[cond2ColsDisp, {cqi}];
	                            condOpers = Drop[condOpers, {cqi}];
	                            sortCols = Drop[sortCols, {cqi}];
	                            sortOrders = Drop[sortOrders, {cqi}];
	                            cqi--;
	                            If[Length[qq] == 0,
	                            	{qq, condCols, cond2Cols, cond2ColsDisp, condOpers, sortCols, sortOrders} = List /@ newQuerySkeleton[];
	                            	cqi = 1;
	                            ];
	                            NotebookClose[EvaluationNotebook[]]
                            ],
	                        NotebookClose[EvaluationNotebook[]]
		                }]}
		            }],
                    WindowFloating -> True, Modal -> True
                ],
                Appearance -> "Palette", ImageSize -> {35, 25}(*, Enabled -> Dynamic@TrueQ[Length[qq] > 1]*)
            ],

            (* Menu button for re-runninng the query (doesn't revert to disk). *)
            Button[
                Tooltip[Import["DatabaseLink/GUI/DatabaseLink/Images/Refresh16.gif"], "Reload query"],

                (* Re-evaluate the query *)
                {qq[[cqi]][queryArgs], qq[[cqi]][resultSet]} = 
                    constructAndExecute @@ Map[#[[cqi]] &, {qq, condOpers, condCols, cond2Cols, sortOrders, sortCols}],
                Appearance -> "Palette", ImageSize -> {35, 25}
            ],
                        
            (* Menu button used to create a notebook containing both code required to execute the query and the result of the query *)
            Button[
                Tooltip[Import["DatabaseLink/GUI/DatabaseLink/Images/Edit16.gif"], "Create a notebook with this query"],
                {qq[[cqi]][queryArgs], qq[[cqi]][resultSet]} =
                    constructAndExecute @@ Map[#[[cqi]] &, {qq, condOpers, condCols, cond2Cols, sortOrders, sortCols}];
                NotebookPut[
                    Notebook[{
                    	Cell[qq[[cqi]][queryTitle], "Section"],
                        Cell[BoxData[RowBox[{"SQLExecute", "[", RowBox[{ToBoxes[
                        	SQLSelect @@ ReplacePart[qq[[cqi]][queryArgs], 1 -> ("Name" /. Options@qq[[cqi]][conn])]
                        ]}], "]", "//", "TableForm"}]], "Input"],
                        Cell[BoxData[ToBoxes[TableForm[qq[[cqi]][resultSet] /. _?(MatchQ[#, SQLSelect[___]] &) -> {}]]], "Output"]
                    }]
                ],
                Appearance -> "Palette", ImageSize -> {35, 25}
            ]
        }],

        Spacer[430],
        Hyperlink["Help", "paclet:DatabaseLink/tutorial/TheDatabaseExplorer"]
        }],

        Spacer[{1, 1}],

        Dynamic@Grid[{{
        (* Body of the DatabaseExplorer UI. *)
            Labeled[
            	Framed[
            		Pane[
		        		(* List the queries that the user has created so far and which is selected. *)
		                ListPicker[
		                	{Dynamic@cqi},
		                    MapThread[#1 -> #2 &,
		                    	{Range@Length@qq, (#[stale] /. {True -> SuperStar, _ -> Identity}) @ #[queryTitle] & /@ qq}
		                    ], 
		                    ImageSize -> {175, 500},
		                    Multiselection -> False,
		                    Appearance -> "Frameless",
		                    Background -> White
		                ],
		                Scrollbars -> True
	            	],
	            	FrameMargins -> 0, FrameStyle -> Lighter@Gray
            	],
                Style["Queries", 14],
                Top
            ],
            
            (* This tab is where the constraints and information about the query can be entered by the user. *)
            TabView[{"Query" -> Grid[{
                (* Selecting the connection to use *)
                List[
                	Framed[
	                    Labeled[
	                    	PopupMenu[
	                    		Dynamic@qq[[cqi]][conn],
                            	Map[(Rule[#, "Name" /. Options[#]]) &, connectionsOpened],
                            	Style["No open connections", {Italic}],
	                            ImageSize -> 396
	                        ],
	                        Style["Connection:", 14],
	                        Left
	                    ],
	                    FrameStyle -> Lighter@Gray,
	                    FrameMargins -> {{Automatic, Automatic}, {12, 12}}
                    ],
                    SpanFromLeft
                ],

                List[
                    (* Selecting the table or tables to be used *)
                    Labeled[
                        Dynamic@ListPicker[
                            Dynamic[qq[[cqi]][tables]], 
                            SQLTableNames[qq[[cqi]][conn]] /. $Failed | SQLTableNames[_] :> {},
                            ImageSize -> {235, (showAdvancedToggle /. $showAdvPaneHeights) - 33},
                            AppearanceElements -> All
                        ] /. _?(MatchQ[#, ListPicker[_, {}, ___]] &) -> Invisible[""],
                        Style["Tables", 14],
                        Top
                    ],

                    (* Selecting the columns to be used *)
                    Labeled[
                        Dynamic@With[{cn = Flatten[SQLColumnNames[qq[[cqi]][conn], #] & /@ qq[[cqi]][tables], 1]
                        	/. {($Failed | SQLColumnNames[__])..} :> {}},
                        	Column[{
                                ListPicker[
                            	    Dynamic[qq[[cqi]][columnList]],
                                    Rule @@@ Transpose[{cn, Row[#, ","] & /@ cn}],
                                    ImageSize -> {235, (showAdvancedToggle /. $showAdvPaneHeights) - 33},
                                    AppearanceElements -> All
                                ] /. _?(MatchQ[#, ListPicker[_, {}, ___]] &) -> Invisible[""],
                                Button["Select All", qq[[cqi]][columnList] = cn, Enabled -> Length@cn > 0]
                            }]
                        ],
	                    Style["Columns", FontSize -> 14],
                        Top
                    ]
                ],
    
                List[
                    Button[Dynamic[showAdvancedText], showAdvancedToggle = !showAdvancedToggle; 
                        showAdvancedText = showAdvancedToggle /. {True -> "Hide Advanced Options", _ -> "Show Advanced Options"},
                        ImageSize -> 500
                    ],
                    SpanFromLeft
                ],

                Sequence @@ If[showAdvancedToggle, {{Panel[Grid[{

                (* This pane is used to set the conditions for the query. *)

                List[
                	Row[{
                        Checkbox[Dynamic[qq[[cqi]][condBool]]],
                        Style["Match"],

                        (* Select how multiple conditions should be joined *)

                        PopupMenu[Dynamic@qq[[cqi]][condOp], {And -> "All", Or -> "Any"}, 
                            Enabled -> Dynamic@qq[[cqi]][condBool]
                        ],
                        Style["of the following conditions:"]
                    }, Spacer[1]],
                    SpanFromLeft
                ],
                List[
                	Framed[
                    	Dynamic@Grid[
                        	List /@ Table[With[{i = i},
	                            Row[{
                                    (* Select the first column in the condition *)
                                    PopupMenu[
                                        Dynamic@condCols[[cqi, i]],
                                        If[validSelectionsQ[],
                                        	With[{cl = Flatten[SQLColumnNames[qq[[cqi]][conn], #] & /@ qq[[cqi]][tables], 1]},
                                        		Rule @@@ Transpose[{cl, ToString@Row[#, ", "] & /@ cl}]
                                        	],
                                            (* else *)
                                            {}
                                        ],
                                        Enabled -> Dynamic@qq[[cqi]][condBool], 
                                        ImageSize -> {170, Automatic}
                                    ],
                                    
                                    (* Select the operation in the condition *)
                                    (* N.B. no logic here to distinguish columns from text slugs *)
                                    PopupMenu[
                                        Dynamic@condOpers[[cqi, i]], {
                                        	Equal -> "is equal to",
                                            Unequal -> "is not equal to", 
                                            Greater -> "is greater than", 
                                            Less -> "is less than", 
                                            GreaterEqual -> "is greater than or equal to", 
                                            LessEqual -> "is less than or equal to",
                                            SQLStringMatchQ[#1, wildcard <> #2 <> wildcard] & -> "contains",
                                            !SQLStringMatchQ[#1, wildcard <> #2 <> wildcard] & -> "does not contain",
                                            SQLStringMatchQ[#1, #2 <> wildcard] & -> "starts with",
                                            SQLStringMatchQ[#1, wildcard <> #2] & -> "ends with"
                                            (* , Null & -> "is contained in"
                                            , Null & -> "is not contained in" *)
                                        },
                                        Enabled -> Dynamic@qq[[cqi]][condBool],
                                        ImageSize -> 110
                                    ],

                                    (* Select the second column of the condition or inputting some other value *)
									Developer`Combobox[Dynamic[cond2ColsDisp[[cqi, i]], (cond2ColsDisp[[cqi, i]] = ToString@#;
										cond2Cols[[cqi, i]] = # /. Row[x_, ___] :> SQLColumn[x]) &], 
									    String,
									    If[validSelectionsQ[], 
									    	With[{cl2 = Flatten[SQLColumnNames[qq[[cqi]][conn], #] & /@ qq[[cqi]][tables], 1]},
									            Row[#, ","] & /@ cl2
									    	],
									    	(*else*)
									    	{}
									    ],
									    ImageSize -> {134, 18},
									    Method -> "PopupMenu",
									    Enabled -> Dynamic@qq[[cqi]][condBool]
									],

									(* Button used to add another condition *)
									Button[Style["+", Bold],
										AppendTo[condCols[[cqi]], condCols[[cqi, -1]]];
										AppendTo[condOpers[[cqi]], "=="];
										AppendTo[cond2ColsDisp[[cqi]], Null];
										AppendTo[cond2Cols[[cqi]], Null];,
										Appearance -> "Palette"
									],
	                        
                                    (* Button used to remove a condition *)
		                            Button[Style["-", Bold],
                                        condCols[[cqi]] = Drop[condCols[[cqi]], {i}]; 
                                        condOpers[[cqi]] = Drop[condOpers[[cqi]], {i}];
                                        cond2ColsDisp[[cqi]] = Drop[cond2ColsDisp[[cqi]], {i}];
                                        cond2Cols[[cqi]] = Drop[cond2Cols[[cqi]], {i}];,
		                                Appearance -> "Palette",
		                                ImageSize -> 14,
                                        Enabled -> TrueQ[Length@condCols[[cqi]] > 1]
		                            ]
	                            }, Spacer[1]]
			                ], {i, Length@condCols[[cqi]]}]
			                (* Grid opts here *)
                    	],
                    	FrameStyle -> Lighter@Gray
                    ],
                    SpanFromLeft
			    ],

                (* This section is used to sort the query. *)
                 
                List[
                	Row[{
                        Checkbox[Dynamic@qq[[cqi]][sortBool]], 
                        Style["Sort:"]
                    }, Spacer[1]],
                    SpanFromLeft
                ],
                
                List[
                    Framed[
                        Dynamic@Grid[
                            List /@ Table[With[{i = i},
                                Row[{

                                    (* Select the column to be sorted on *)
		                            PopupMenu[
		                                Dynamic@sortCols[[cqi, i]],
		                                If[validSelectionsQ[],
		                                	With[{cl = qq[[cqi]][columnList]}, Rule @@@ Transpose[{cl, Row[#, ","] & /@ cl}]],
		                                    (* else *)
		                                    {}
		                                ],
		                                Enabled -> Dynamic@qq[[cqi]][sortBool],
		                                ImageSize -> {315, Automatic}
		                            ],

		                            (* Select how the column should be sorted *)
		                            PopupMenu[
		                                Dynamic@sortOrders[[cqi, i]], 
		                                {"Ascending" -> "Ascending", "Descending" -> "Descending"}, 
		                                Enabled -> Dynamic@qq[[cqi]][sortBool],
		                                ImageSize -> 110
		                            ],

		                            (* Button used to add another sort *)
		                            Button[Style["+", Bold],
		                                AppendTo[sortCols[[cqi]], sortCols[[cqi, -1]]];
		                                AppendTo[sortOrders[[cqi]], "Ascending"];,
		                                Appearance -> "Palette"
		                            ],

		                            (* Button used to remove a sort *)
		                            Button[Style["-", Bold],
	                                    sortCols[[cqi]] = Drop[sortCols[[cqi]], {i}]; 
	                                    sortOrders[[cqi]] = Drop[sortOrders[[cqi]], {i}];,
		                                Appearance -> "Palette",
		                                ImageSize -> 14,
		                                Enabled -> TrueQ[Length@sortCols[[cqi]] > 1]
		                            ]

                                }, Spacer[1]]
                            ], {i, Length@sortCols[[cqi]]}
                            (* Grid opts *)
                        ]
                    ], FrameStyle -> Lighter@Gray],
                    SpanFromLeft
                ],

                List[
                	Grid[{
                        (* Setting the maximum number of rows to display and when the connection should timeout. *)
                		{
                    		Row[{
	                    		Checkbox[Dynamic@qq[[cqi]][limit]],
	                    		Style[" Limit to "], 
	                            InputField[Dynamic@qq[[cqi]][limitValue], Number, Enabled -> qq[[cqi]][limit], FieldSize -> 5],
	                            Style[" rows"];
	                        }],
	                        Row[{
	                        	Checkbox[Dynamic@qq[[cqi]][timeout]],
	                            Style[" Timeout after "], 
	                            InputField[Dynamic@qq[[cqi]][timeoutValue], Number, Enabled -> qq[[cqi]][timeout], FieldSize -> 3], 
	                            Style[" seconds"]
	                        }],
	                        SpanFromLeft
                		},
                        (* Select whether the rows should be distinct, if column headings should be shown, and if the results should be returned as a string. *)
                		{
	                        Row[{Checkbox[Dynamic@qq[[cqi]][distinct]], Style["Select Distinct Rows"]}, Spacer[1]],
	                        Row[{Checkbox[Dynamic@qq[[cqi]][sch]], Style["Show Column Headings"]}, Spacer[1]], 
	                        Row[{Checkbox[Dynamic@qq[[cqi]][gas]], Style["Get as Strings"]}, Spacer[1]]
                		}
                	}, ItemSize -> All, Dividers -> None, Background -> Automatic, Spacings -> {0, Automatic}],
                    SpanFromLeft
                ]}, Dividers -> None, Alignment -> {Left}, Spacings -> {2, 1}], Background -> Lighter@Lighter@Gray], SpanFromLeft}},
                (* else Advanced toggle *)
                {}
                ]},
                Dividers -> None, 
                Alignment -> {Left, 2 -> Top},
                Spacings -> {2, 1}
            ],

            (* This tab is where the results of the query are shown *)

            "Result" -> 
                Dynamic@Pane[
                	If[validSelectionsQ[],
                        $error = Null;
                        If[Quiet@Check[
                            {qq[[cqi]][queryArgs], qq[[cqi]][resultSet]} = 
                                constructAndExecute @@ Map[#[[cqi]] &, {qq, condOpers, condCols, cond2Cols, sortOrders, sortCols}];
                            $error = $MessageList;
                            False,
                            
                            True
                        ] || qq[[cqi]][resultSet] === $Failed,
                            Style["Error: " <> ToString[First@$error /. Null -> "(no message)"], FontSlant -> Italic, FontColor -> Red],
                            (* else *)
	                        Grid[
	                            With[{base = Replace[
	                                    qq[[cqi]][resultSet], {x_List, y___} :> 
	                                	   {Item[Style[#, {Bold}], Alignment -> Center] & /@ x, y}
	                                	   /; TrueQ@qq[[cqi]][sch]
	                                ],
	                                rowLabels = If[qq[[cqi]][sch],
	                                		Style[#, {Smaller, Italic}] & /@ Prepend[Range@Max[0, Length@qq[[cqi]][resultSet] - 1], Invisible[""]],
	                                		(* else *)
	                                        Range@Length@qq[[cqi]][resultSet]
	                                	]
	                                },
	                                Join[
	                                    Transpose[{rowLabels}],
	                                    base /. SQLDateTime[x_] :> x,
	                                    2
	                                ]
	                            ],
	                            Dividers -> LightGray,
	                            Spacings -> {Automatic, 1},
	                            ItemSize -> Full,
	                            Background -> White,
	                            Alignment -> List@With[{al = alignmentsByColumnType[qq[[cqi]][conn], qq[[cqi]][columnList]]}, Prepend[al, Right]]
	                        ]
                        ]
                    ], 
                    ImageSize -> {508, 500},
                    Scrollbars -> True
                ]},
                ControlPlacement -> Top,
                FrameMargins -> 0
            ]}},
            Alignment -> Top
        ],
        

        (* Status bar at bottom *)
        Dynamic@Framed[
        	Pane[
                Which[
                	!SQLConnectionOpenQ[qq[[cqi]][conn]],
                	Style["Please open a connection.", FontSlant -> Italic],
                	
                    (*!MemberQ[connectionsOpened, qq[[cqi]][conn]],
              		Style["Connection not found. Please open a connection.", FontSlant -> Italic],*)
              		
                    !validSelectionsQ[],
                    Style["Invalid query. Please choose valid tables and columns.", FontSlant -> Italic], 

                    True,
                    Style[ToString@Max[0, Length[qq[[cqi]][resultSet]] - (qq[[cqi]][sch] /. {True -> 1, _ -> 0})] <> " records.",
                    	FontSlant -> Italic
                    ]
                ],
                ImageSize -> {707, 15}
            ],
            FrameStyle -> Lighter@Gray
        ]
    }], FontSize -> 12]],
    WindowTitle -> "Database Explorer",
    NotebookEventActions -> {"WindowClose" :> (CloseSQLConnection /@ connectionsOpened)}
]

(*
 * For formatting result set Grid.
 *)
alignmentsByColumnType[conn_, cols_List] := Module[
	{ci = SQLColumnInformation[conn, #][[1]] & /@ cols},
	Switch[#,
		"CHAR" | "TEXT" | "NCHAR" | "NVARCHAR" | "NTEXT" | "VARCHAR", Left,
        "BOOLEAN", Center,
		_, Right
	] & /@ ci[[All, 6]]
];


(*
 * MS has the only currently supported dbs with non-standard wildcards.
 *)
inferWildcard[conn_] := With[{j = conn[[2]]},
	Switch[j@getClass[]@getName[],
		(* Access, Excel *)
		_?(StringMatchQ[#, "sun.jdbc.odbc.*"] &), "*",
		_, "%"
	]
] /; SQLConnectionOpenQ[conn];
inferWildcard[_] := "%";


(*
 * Clones DownValues, for SaveAs etc.
 *)
dvClone[obj_] := Module[
	{c, dv = DownValues[obj]},
    MapThread[c[#1] = #2; &,
    	{ReleaseHold@dv[[All, 1, All, 1]], dv[[All, 2]]}
    ];
    c
]


End[] (* "`DatabaseExplorer`Private`" *)
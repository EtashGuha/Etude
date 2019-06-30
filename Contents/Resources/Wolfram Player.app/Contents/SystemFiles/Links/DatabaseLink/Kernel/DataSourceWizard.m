(* Author:          Dillon Tracy, Dylan Boliske *)
(* Copyright:       Copyright 2013, Wolfram Research, Inc. *)

Begin["`DataSourceWizard`Private`"]

OpenSQLConnection[] := openSQLDialog[];

(*
 * This UI is used to connect to a database. 
 * The user may either connect to one that already exists, 
 * create a new one, edit an existing one, or delete a database
 * connection.
 *)
openSQLDialog[] := Module[{openedConn}, DynamicModule[
    {
        ds,
        confirm, inform,
        newPicker,
        preflightCheck, preflightFlag, conn, u, p, dso,
        confirmed,
        nbc, nbi,
        file,
        picker
    },
    
    (* Custom ChoiceDialog etc. is necessary when nesting inside DialogInput *)
    confirm[label_] := (confirmed = Null;
        nbc = CreateDialog[{Column[{
            label,
            ChoiceButtons[{confirmed = True; NotebookClose[nbc], confirmed = False; NotebookClose[nbc]}]
        }, Center]}, WindowFloating -> True, Modal -> True]
    );

    inform[label_] := (
        nbi = CreateDialog[{Column[{
            label,
            DefaultButton[NotebookClose[nbi]]
        }, Center]}, WindowFloating -> True, Modal -> True]
    );
    
    newPicker[] := With[{dss = DataSources[]},
        ds = Take[dss, Min[1, Length@dss]];
    	ListPicker[Dynamic[ds],
            Rule @@@ Transpose[{dss, DataSourceNames[]}],
            Appearance -> "Frameless",
            Spacings -> 3,
            Scrollbars -> False,
            ContentPadding -> False,
            FieldSize -> {Automatic, Length@dss}
						(* , Background -> "Gray", FieldSize -> {25, 20}, Spacings -> 3, Multiselection -> False *)
    	]
    ];
    
    (* Perform the $Prompt check here, to avoid hanging the FE with a nested DialogInput
     * when calling OpenSQLConnection.
     *)
    preflightCheck[n_] := Module[
    	{list},
    	preflightFlag = Null;
        list = Cases[DataSources[], c_ /; ("Name" /. canonicalOptions[Options[c]] /. 
        	Options[SQLConnection]) === n];
        	
        If[Length[list] > 0,
        	conn = First@list;
        	dso = Options[conn];
        	u = "Username" /. dso;
        	p = "Password" /. dso;
        	If[p === "$Prompt",
        		p = None;
        		DatabaseLink`UI`Private`NestablePasswordDialog[{u, p}, preflightFlag];,
        	    (* else *)
        	    preflightFlag = True
        	];,
        	(* else *)
        	Message[OpenSQLConnection::notfound, n];
        	preflightFlag = $Failed
        ]
    ];
    
    picker = newPicker[];

    openedConn = DialogInput[
        Dynamic[
            If[TrueQ[confirmed],
                (* Delete the file associated with the given database connection *)
                confirmed = Null; 
                file = "Location" /. Options[First@ds] /. Options[SQLConnection];
                If[DeleteFile[file] === Null,
                    (* inform[StringForm["Deleted data source \"``\".", "Name" /. Options[First@ds]]] *)Null,
                    inform[StringForm["Failed to delete data source \"``\" at ``.", "Name" /. Options[First@ds], file]]
                ];
                picker = newPicker[];
            ];
            If[TrueQ@cdUpdate,
            	cdUpdate = Null;
            	picker = newPicker[];
            ];
            If[TrueQ@preflightFlag,
            	preflightFlag = Null;
            	DialogReturn[
            		With[{c = OpenSQLConnection[conn[[1]], "Username" -> u, "Password" -> p, Sequence @@ dso]},
            			c /. Rule["Password", _] -> Rule["Password", "$Prompt"]
            		]
            	]
            ];

            Column[{
                Row[{
                    Labeled[
                    	Pane[
	                        Dynamic@picker,
	                        Scrollbars -> True, ImageSize -> {250, 375}
                    	],
	                    Style["Connections", FontSize -> 14],
	                    Top
                    ],
                    " ", 
                    Column[{
                        (* Will open the Database Connection Wizard to create a new database connection. *)
                        Button[Style["New..."],connectionDialog[]],
                        (* Will open the Database Connection Wizard to edit the selected database connection. *)
                        Button[Style["Edit..."], connectionDialog[First@ds]], 
                        (* Will delete the selected database connection if the user agrees. *)
                        Button[Style["Delete..."], With[{name = "Name" /. Options[First@ds] /. Options[SQLConnection]},
                            If[StringQ[name], confirm[StringForm["Delete data source \"``\"?", name]]]
                        ]],
                        Button[Style["Refresh"], picker = newPicker[];]
                    }]
                }], 
                ChoiceButtons[
                	{"Connect", "Cancel"}, 
                	{
                		With[{name = "Name" /. Options[First@ds] /. Options[SQLConnection]},
                            preflightCheck[name]
                        ],
                		DialogReturn@$Canceled
                	}
                ]
            }, Alignment -> Center]
        ]
        , WindowTitle -> "Connection Tool"
    ];
]; openedConn];


(* The toc used by the Database Connection Wizard. *)
connectionMenu[page_Integer] := Panel[
    Grid[{
        List@Item[Style["Steps", FontSize -> 16], Alignment -> Left],
        Sequence @@ List /@ {
            "\n1. Overview",
            "2. Specify Name and Description",
            "3. Specify Visibility",
            "4. Select Type of Database",
            "5. Connect to Database",
            "6. Specify Connection Properties"
        }}, 
        Alignment -> Left,
        ItemStyle -> {Automatic, Automatic, {{page + 1, 1} -> Bold}},
        Spacings -> {Automatic, 2},
        Dividers -> {None, {2 -> LightGray}},
        FrameStyle -> Automatic
    ],
    ImageSize -> {195, 350},
    Background -> White
]


connectionContentsTemplate[title_, stuff__, o:OptionsPattern[]] := Pane[
    Panel[Grid[{
    	    {
                Item[Style[title, FontSize -> 16], Alignment -> Left],
            	Item[Hyperlink["Help", "paclet:DatabaseLink/tutorial/TheDatabaseExplorer#9010"], Alignment -> Right]
            },
            Sequence @@ (List[#, SpanFromLeft] & /@ {stuff})
        },
        o,
        Alignment -> {Left, Baseline},
        Spacings -> {Automatic, 2},
        Dividers -> {None, {2 -> LightGray}},
        FrameStyle -> Automatic
    ], ImageSize -> {410, 400}]
];

$linebreakAdjustments = {0.95, 10, 100, 500, 500};
$defaultErrorMsg = Style["No results", FontSlant -> Italic];
(* "?" (e.g. mysql) and ";" (derby, sql server) are the only two setoff characters I can find
 * for connection attributes. However the jdbc spec is loose on this point so this may need flexibility in the future.
 *)
$connectionAttributeSetoffs = {";", "?"};

staticConnectionContents[1] := connectionContentsTemplate["Overview",
    Pane[Style[
        "\nThis wizard helps create a reusable data source for use with DatabaseLink.\n\n" <>
        "Some information about the data is required to use this wizard:\n\n\t" <> 
        "\[Bullet]" <> " Type of database\n\t" <> 
        "\[Bullet]" <> " Location of database\n\t" <> 
        "\[Bullet]" <> " Username to use to connect to database\n\n" <>
        "Many types of databases are preconfigured within this wizard to contain custom wizard pages to make it easier to configure a data source. " <>
        "However, if a type of database is not listed, one may be added using this wizard.",
        LinebreakAdjustments -> $linebreakAdjustments,
        LineIndent -> 0
    ]]
];


(* Opens the Database Connection Wizard and allows the user to either edit or create a database connection *)
Options[connectionDialog] = {
    Sequence @@ Options[SQLConnection]
};

(* New source ... *)
connectionDialog[o:OptionsPattern[]] := Module[
    {type = "HSQL(Standalone)", rules},
	rules = FilterRules[type /. $RDBMSSpecificProperties, Options[connectionDialog]];
	connectionDialog[1, type, None, Sequence @@ Flatten[{{o}, rules}]]
]

(* Edit source ... *)
connectionDialog[SQLConnection[JDBC[type_, url_], ___, oConn:OptionsPattern[]], o:OptionsPattern[]] :=
    connectionDialog[1, type, url, Sequence @@ Flatten[{{o}, {oConn}}]];

connectionDialog[pageIn_Integer, type_String, url_, o:OptionsPattern[]] := CreateDialog[DynamicModule[
    {
        page = pageIn,
        errorMsg = $defaultErrorMsg,
        $error, conn,

        connectionContents,
        catalogs, prereadCatalogs, catU, catP, readCatsIsAGo,
        pretest, testU, testP, testIsAGo,
        finish,

        storePassword,
        visibility,
        jdbcProperties, propertiesEditor,

        jdbcNames,
        jdbcDrivers,
        jdbcNameToDesc,
        refreshJDBCInfo,

        (* For all connection properties *)
        p
    },
    (* Flag for wizard closure *)
    cdUpdate = Null;

    p["Type"] = type;
    p["URL"] = url;
    (* Set baseline properties and properties not available through options.
     * Options have the last word now but subsequent calls to setSourceDefaults will override.
     *)
    setSourceDefaults[p, p["Type"]];
    (* Read properties from options *)
    Scan[
        (p[ToString@First@#] = Last@#;) &,
        Reverse[Flatten[{{o}, Options[connectionDialog]}]]
    ];
    
    jdbcProperties = If[MatchQ[p["Properties"], {Rule[_String, _String] ...}], p["Properties"], {}];
    storePassword = !TrueQ[p["Password"] === "$Prompt"];
    visibility = !TrueQ[StringQ[p["Location"]] && StringMatchQ[p["Location"], $BaseDirectory <> "*"]];
    (* At least one RDBMS (Excel) flips out if it has any TIL setting at all, so we can't map Automatic
     * onto a particular setting.
     *)
    (*If[p["TransactionIsolationLevel"] === Automatic, p["TransactionIsolationLevel"] = "ReadUncommitted"];*)
    
    (* Take apart URL *)
    If[StringQ[p["URL"]],
    	If[p["hostedQ"],
	        {{p["Server"], p["Port"], p["Database"], p["Attributes"]}} = StringCases[p["URL"],
	            w:(Except[":"]).. ~~ (":"~~x:NumberString...)... ~~ ("/"~~y__)... ~~ 
	               (Alternatives @@ $connectionAttributeSetoffs~~z__)... :> {w, x, y, z}
	        ],
	        
	        (* else: server and port are meaningless *)
	        {p["Server"], p["Port"]} = {"", ""};
	        {{p["Database"], p["Attributes"]}} = StringCases[p["URL"],
                y__.. ~~ (Alternatives @@ $connectionAttributeSetoffs~~z__)... :> {y, z}
            ];
    	]
        (* Else new source (or corrupt config) *)
    ];

    (* Prepare driver descriptions for lookup etc. *)
    refreshJDBCInfo[] := (
        jdbcNames = DeleteDuplicates@Sort@JDBCDriverNames[];
        jdbcDrivers = JDBCDrivers[];
        jdbcNameToDesc = With[{nameToDriver = List["Name" /. #, #] & /@ List @@@ jdbcDrivers},
            Rule[First@#, "Description" /. Last@# /. "Description" -> ""] & /@ nameToDriver
        ];
    );
    refreshJDBCInfo[];

    (* For dynamically reading databases available at a given source 
     * (not all RDBMSs can do this) 
     *)
    prereadCatalogs[] := Module[
    	{},
        catU = p["Username"];
        catP = p["Password"];
        readCatsIsAGo = Null;
        If[catP === "$Prompt",
            catP = None;
            DatabaseLink`UI`Private`NestablePasswordDialog[{catU, catP}, readCatsIsAGo];,
            (* else *)
            readCatsIsAGo = True
        ]
    ];
    catalogs = {};

    (*
     * For Test button on pp. 5, 6.
     * Some options not tracked in p are set manually here.
     *
     * Perform the $Prompt check here, to avoid hanging the FE with a nested DialogInput
     * when calling OpenSQLConnection.
     *)
    pretest[] := Module[
    	{},
    	testU = p["Username"];
    	testP = p["Password"];
    	testIsAGo = Null;
        If[testP === "$Prompt",
            testP = None;
            DatabaseLink`UI`Private`NestablePasswordDialog[{testU, testP}, testIsAGo];,
            (* else *)
            testIsAGo = True
        ]
    ];
    	
    (*
     * For saving out connection
     *)
    finish[] := Module[
        {name, fn, nb = EvaluationNotebook[], err},
        If[Quiet@Check[
            {name, fn} = finishConn[p, storePassword, visibility, jdbcProperties];
            err = $MessageList;
            False,
            
            True],
            
            CreateDialog[{
                Column[{
                    StringForm["Error saving data source '``' to", name],
                    Style[fn <> ":"],
                    ToString[Style[First@err, {Red, Italic}], StandardForm],
                    DefaultButton[NotebookClose[EvaluationNotebook[]];]
                }, Center]
            }, WindowFloating -> True, Modal -> True],
            
            CreateDialog[{
                Column[{
                    StringForm["Saved data source '``' to", name],
                    Style[fn],
                    DefaultButton[NotebookClose[EvaluationNotebook[]];]
                }, Center]
            }, WindowFloating -> True, Modal -> True];
            cdUpdate = True;
            NotebookClose[nb]
        ]
    ];
 
    (*
     * Contents for each page of the wizard.
     *)
    connectionContents[1] := staticConnectionContents[1];
    connectionContents[2] := With[{dsns = DataSourceNames[]}, connectionContentsTemplate["Name and Description",
        Style[
            Apply[StringJoin, ToString[#, StandardForm] & /@ {
                Style["\nSpecify the name.", FontColor -> Black],
                If[MemberQ[dsns, p["Name"]], Style[" A connection with this name already exists.", FontSlant -> Italic, FontColor -> Black], ""]
            }],
            LineIndent -> 0
        ],
        InputField[Dynamic[p["Name"]], String, ImageSize -> {375, 20}],
        "Specify the description.",
        InputField[Dynamic[p["Description"]], String, ImageSize -> {375, 150}], 
        Style[
            Apply[StringJoin, ToString[#, StandardForm] & /@ {
                Style["After specifying the name and description, choose 'Next' to select the visibility of the configuration.", FontColor -> Black],
                If[MemberQ[dsns, p["Name"]], Style[" Continuing will eventually overwrite existing settings.", FontSlant -> Italic, FontColor -> Black], ""]
            }],
            LineIndent -> 0, FontColor -> Black, LinebreakAdjustments -> $linebreakAdjustments
        ],
        Spacings -> {Automatic, {2, {3 -> Automatic, 5 -> Automatic}}}
    ]];

    connectionContents[3] := connectionContentsTemplate["Specify Visibility",
        "\nPlease specify the visibility of the data source configuration.",
        Row[{
            RadioButton[Dynamic[visibility], True, BaselinePosition -> Scaled[0.2]], 
            Style["User Level", FontSize -> 12],
            Spacer[10], 
            RadioButton[Dynamic[visibility], False, BaselinePosition -> Scaled[0.2]], 
            Style["System Level", FontSize -> 12]
        }, Spacer[1.0]], 
        Style["After specifying the visibility of the data source configuration, choose 'Next' to select the type of database.", LineIndent -> 0,
            LinebreakAdjustments -> $linebreakAdjustments
        ],
        Spacings -> {Automatic, {2, {3 -> 1.5, 4 -> 1.5}}}
    ];
    
    connectionContents[4] := connectionContentsTemplate["Type of Database",
        "\nSelect the type of database:",
        Dynamic@PopupMenu[Dynamic[p["Type"], 
        	(p["Type"] = #; setSourceDefaults[p, p["Type"]]) &],
        	jdbcNames
        ], 
        Dynamic@Style[InputField[Dynamic[p["Type"] /. jdbcNameToDesc], String, Enabled -> False, ImageSize -> {375, 150}]],
        Row[{
            (*Opens the JDBC driver dialog to allow the user to create a new JDBC driver description.*)
            Button["New", DatabaseLink`JDBCWizard`Private`JDBCDialog["New"]],
            (*Opens the JDBC driver dialog to allow the user to edit the selected JDBC driver description.*)
            Button["Edit", DatabaseLink`JDBCWizard`Private`JDBCDialog["Edit", p["Type"]]],
            Button["Refresh", refreshJDBCInfo[]]
        }],
        Style["After choosing the type of database, choose 'Next' to specify the location.", LineIndent -> 0],

        (* Row[{Style["Test", Larger], PopupMenu[Dynamic[p["Database"]], catalogs]}], *)

        Spacings -> {Automatic, {2, {3 -> Automatic, 4 -> Automatic, 5 -> 1.0}}}
    ];

    connectionContents[5] := connectionContentsTemplate[p["Type"] <> " Connection",
        "\nSpecify how to connect to the database:",
        
        Switch[p["GUIVariant"],
        	"File, base name",
            Grid[{
                {Style["Database File", Larger], 
                    Row[{InputField[Dynamic[p["Database"]], String, FieldSize -> 17],
                    (* HSQL (for example) wants a basename slug with which to construct the .script, .properties, etc.,
                     * but FileSetter will come back with a specific path. Instead of creating foo.ext.script,
                     * foo.ext.properties, etc., remove the extension of the selected file and use that as the
                     * slug.
                     *)
                     
                    FileNameSetter[
                    	Dynamic[p["Database"],
                    		(p["Database"] = FileNameJoin[With[{path = If[TrueQ@p["RelativePath"], absToRel[#, p["Location"]], #]},
                    			{DirectoryName@path, altFileBaseName@path}]]) &
                    	],
                    	p["FileNameSetter spec"],
                    	p["FileNameSetter patterns"],
                    	WindowTitle -> "Select Database", Method -> "Preemptive"
                    ]}]},
                   
                {Style["Relative Path", Larger],
                	DynamicModule[{$enableCheckbox1,location1},If[p["Database"] == "", $enableCheckbox1 = False, $enableCheckbox1=True];
                	Checkbox[
                		Dynamic[p["RelativePath"], (
                			p["RelativePath"] = #;
                            p["Database"] = If[TrueQ@p["RelativePath"],
      					        location1 = DirectoryName[p["Database"]];
      					        If[location1 == "",location1 = Nest[First,DatabaseResourcesPath[],2]];
    					        FileNameJoin[
                                 With[{path = absToRel[p["Database"], location1]}, {DirectoryName@path, altFileBaseName@path}]]
                           ,(*else*)
                                ExpandFileName[FileNameJoin[{location1 /. Null -> "", p["Database"]}]]
                            ]
                ) &], BaselinePosition -> Scaled[0.2], Enabled -> Dynamic[$enableCheckbox1]]]},
                {Style["Attributes", Larger], InputField[Dynamic[p["Attributes"]], String]},
                {Style["Username", Larger], InputField[Dynamic[p["Username"]], String]},
                {Style["Password", Larger], InputField[Dynamic[p["Password"]], String, FieldMasked -> True]}
            }, Alignment -> Left], 
            
            "File",
            Grid[{
                {Style["Database File", Larger], 
                    Row[{InputField[Dynamic[p["Database"]], String, FieldSize -> 17],
                    FileNameSetter[
                        Dynamic[p["Database"],
                            (p["Database"] = With[{path = If[TrueQ@p["RelativePath"], absToRel[#, p["Location"]], #]}, path]) &
                        ],
                        p["FileNameSetter spec"],
                        p["FileNameSetter patterns"],
                        WindowTitle -> "Select Database", Method -> "Preemptive"
                    ]}]},
                {Style["Relative Path", Larger],
                	DynamicModule[{$enableCheckbox2,location2},If[p["Database"] == "", $enableCheckbox2 = False, $enableCheckbox2=True];
                    Checkbox[
                        Dynamic[p["RelativePath"], (
                            p["RelativePath"] = #;
                            p["Database"] = If[TrueQ@p["RelativePath"],
                            	location2 = DirectoryName[p["Database"]];
      					        If[location2 == "",location2 = Nest[First,DatabaseResourcesPath[],2]];
                                With[{path = absToRel[p["Database"], location2]}, path],
                                (* else *)
                                ExpandFileName[FileNameJoin[{location2 /. Null -> "", p["Database"]}]]   
                            ]
                        ) &], BaselinePosition -> Scaled[0.2],Enabled->Dynamic[$enableCheckbox2]]]},
                {Style["Attributes", Larger], InputField[Dynamic[p["Attributes"]], String]},
                {Style["Username", Larger], InputField[Dynamic[p["Username"]], String]},
                {Style["Password", Larger], InputField[Dynamic[p["Password"]], String, FieldMasked -> True]}
            }, Alignment -> Left], 

            "Memory",
            Grid[{
                {Style["Database", Larger], 
                    InputField[Dynamic[p["Database"]], String]}
            }, Alignment -> Left], 
            
            "Memory, no database",
            Grid[{
                {Style["(Nothing to configure for this connection type)", {Italic, Larger}]}
            }, Alignment -> Left], 
            
            "Hosted, choose file",
            (* Firebird & co. don't support connections that are not tied to a specific database.
             * So you can't establish a connection to list the databases. Use input/browse.
             *)
            Grid[{
                {Style["Hostname", Larger], InputField[Dynamic[p["Server"]], String, FieldSize -> 25]},
                {Style["Port", Larger], InputField[Dynamic[p["Port"]], String]},
                {Style["Database File", Larger], Row[{
                    InputField[Dynamic[p["Database"]], String, FieldSize -> 17],
                    FileNameSetter[
                        Dynamic[p["Database"],
                            (p["Database"] = With[{path = If[TrueQ@p["RelativePath"], absToRel[#, p["Location"]], #]}, path]) &
                        ],
                        p["FileNameSetter spec"],
                        p["FileNameSetter patterns"],
                        WindowTitle -> "Select Database", Method -> "Preemptive"
                    ]
                }]},
                {Style["Relative Path", Larger],
                	DynamicModule[{$enableCheckbox3,location3},If[p["Database"] == "", $enableCheckbox3 = False, $enableCheckbox3 = True];
                    Checkbox[
                        Dynamic[p["RelativePath"], (
                            p["RelativePath"] = #;
                            p["Database"] = If[TrueQ@p["RelativePath"],
                            	location3 = DirectoryName[p["Database"]];
      					        If[location3 == "",location3 = Nest[First,DatabaseResourcesPath[],2]];
                                With[{path = absToRel[p["Database"],location3]}, path],
                                (* else *)
                                ExpandFileName[FileNameJoin[{location3 /. Null -> "", p["Database"]}]]
                            ]
                        ) &], BaselinePosition -> Scaled[0.2],Enabled->Dynamic[$enableCheckbox3]]]},
                {Style["Attributes", Larger], InputField[Dynamic[p["Attributes"]], String]},
                {Style["Username", Larger], InputField[Dynamic[p["Username"]], String]},
                {Style["Password", Larger], InputField[Dynamic[p["Password"]], String, FieldMasked -> True]}

            }, Alignment -> Left],

            "Hosted, choose file base name",
            Grid[{
                {Style["Hostname", Larger], InputField[Dynamic[p["Server"]], String, FieldSize -> 25]},
                {Style["Port", Larger], InputField[Dynamic[p["Port"]], String]},
                {Style["Database File", Larger], Row[{
                    InputField[Dynamic[p["Database"]], String, FieldSize -> 17],
                    FileNameSetter[
                        Dynamic[p["Database"],
                            (p["Database"] = FileNameJoin[With[{path = If[TrueQ@p["RelativePath"], absToRel[#, p["Location"]], #]},
                                {DirectoryName@path, altFileBaseName@path}]]) &
                        ],
                        p["FileNameSetter spec"],
                        p["FileNameSetter patterns"],
                        WindowTitle -> "Select Database", Method -> "Preemptive"
                    ]
                }]},
                {Style["Relative Path", Larger],
                	DynamicModule[{$enableCheckbox4,location4},If[p["Database"] == "", $enableCheckbox4 = False, $enableCheckbox4 = True];
                    Checkbox[
                        Dynamic[p["RelativePath"], (
                            p["RelativePath"] = #;
                            p["Database"] = If[TrueQ@p["RelativePath"],
                            	location4 = DirectoryName[p["Database"]];
      					        If[location4 == "",location4 = Nest[First,DatabaseResourcesPath[],2]];
                                With[{path = absToRel[p["Database"], location4]}, path],
                                (* else *)
                                ExpandFileName[FileNameJoin[{location4 /. Null -> "", p["Database"]}]]
                            ]
                        ) &], BaselinePosition -> Scaled[0.2],Enabled->Dynamic[$enableCheckbox4]]]},
                {Style["Attributes", Larger], InputField[Dynamic[p["Attributes"]], String]},
                {Style["Username", Larger], InputField[Dynamic[p["Username"]], String]},
                {Style["Password", Larger], InputField[Dynamic[p["Password"]], String, FieldMasked -> True]}

            }, Alignment -> Left],

            "Hosted, unlistable databases",
            Grid[{
                {Style["Hostname", Larger], InputField[Dynamic[p["Server"]], String, FieldSize -> 25]},
                {Style["Port", Larger], InputField[Dynamic[p["Port"]], String]},
                {Style["Database", Larger], InputField[Dynamic[p["Database"]], String]},
                {Style["Attributes", Larger], InputField[Dynamic[p["Attributes"]], String]},
                {Style["Username", Larger], InputField[Dynamic[p["Username"]], String]},
                {Style["Password", Larger], InputField[Dynamic[p["Password"]], String, FieldMasked -> True]}

            }, Alignment -> Left],

            "Hosted" | _,
            Grid[{
                {Style["Hostname", Larger], InputField[Dynamic[p["Server"], (p["Server"] = #;) &], String, FieldSize -> 25]},
                {Style["Port", Larger], InputField[Dynamic[p["Port"], (p["Port"] = #;) &], String]},
                {
                	Style["Database", Larger],
                	Row[{
                		Developer`Combobox[Dynamic[p["Database"]], 
		                    String,
		                    catalogs,
		                    ImageSize -> {198, 18},
		                    Method -> "PopupMenu"
                		],
                		Button["Refresh", prereadCatalogs[]]
                	}]
                },
                {Style["Attributes", Larger], InputField[Dynamic[p["Attributes"]], String]},
                {Style["Username", Larger], InputField[Dynamic[p["Username"], (p["Username"] = #;) &], String]},
                {Style["Password", Larger], InputField[Dynamic[p["Password"], (p["Password"] = #;) &], String, FieldMasked -> True]}
            }, Alignment -> Left]
        ], 
        
        Row[{
            Button["Test", pretest[], ImageSize -> 50, Enabled -> StringQ[propToUrl@p]],
            Panel[Dynamic[errorMsg]]
        }, Spacer[10]], 
        Style["After specifying the location of the database, choose 'Next' to specify connection properties.", 
            LinebreakAdjustments -> $linebreakAdjustments,
            LineIndent -> 0
        ],
        Spacings -> {Automatic, {2, {3 -> Automatic, 4 -> 1.5}}}
    ];

    propertiesEditor[] := Framed[
        Pane[
            Dynamic@Grid[
                Join[
                    List@{
                        Item["Property", ItemSize -> {17, 1.}, Alignment -> Center, Background -> Lighter[LightGray, 0.75], 
                            Frame -> {{False, True}, {False, False}}, FrameStyle -> LightGray],
                        Item["Value", ItemSize -> {17, 1.}, Alignment -> Center, Background -> Lighter[LightGray, 0.75]],
                        Item["", Alignment -> Center, Background -> Lighter[LightGray, 0.75]]
                    },
                    Join[
                        List @@@ MapIndexed[
                            Item[
                                InputField[Dynamic[jdbcProperties[[Sequence @@ #2]]], FieldSize -> 15.5, ImageSize -> {Automatic, 16}], 
                                ItemSize -> {17, 2.15}
                            ] &,
                            jdbcProperties,
                            {2}
                        ] /. {} -> {{}},
                        Table[List@With[{idx = i},
                            Item[
                                Tooltip[
                                    Button[
                                        Style["\[Times]", {Bold, Red}],
                                        jdbcProperties = Delete[jdbcProperties, idx],
                                        Appearance -> None
                                    ],
                                    "Delete this property"
                                ],
                                ItemSize -> 1.5
                            ]
                        ], {i, Length@jdbcProperties}] /. {} -> {{}},
                        2
                    ],
                    {{Item[Button["Add Property", AppendTo[jdbcProperties, "" -> ""], ImageSize -> 120]], SpanFromLeft}}
                ]
                , Alignment -> {Center, Center}
                , Spacings -> {0, 0.3}
                , Dividers -> None
            ]
            , ImageSize -> {Automatic, 75}
            , FrameMargins -> 0
            , Scrollbars -> Automatic
        ]
        , FrameStyle -> LightGray
        , FrameMargins -> 0
        (*, BaseStyle -> {FontFamily -> "Dialog"}*)
    ];
 
    connectionContents[6] := connectionContentsTemplate["Connection Properties",
    	
        Grid[{
            {Checkbox[Dynamic[storePassword], BaselinePosition -> Scaled[0.2]], 
            	Style["Store password (in plain text).", Larger]},
            {Row[{Checkbox[Dynamic[p["UseConnectionPool"]], {False, True, Automatic}, BaselinePosition -> Scaled[0.2]], 
            	Style[Dynamic[p["UseConnectionPool"] /. Except[Automatic] -> ""], {Bold}]}, Spacer[1]],
            	Style["Use a connection pool.", Larger]}, 
            {Row[{Checkbox[Dynamic[p["ReadOnly"]], {False, True, Automatic}, BaselinePosition -> Scaled[0.2]], 
                Style[Dynamic[p["ReadOnly"] /. Except[Automatic] -> ""], {Bold}]}, Spacer[1]],
                Style["Use read only connection.", Larger]}
        }, Alignment -> Left],
        Row[{
            PopupMenu[
                Dynamic[p["TransactionIsolationLevel"]],
                {Automatic, "ReadUncommitted", "ReadCommitted", "RepeatableRead", "Serializable"}
            ],
            "Select a transaction isolation level."
        }, Spacer[5]], 
        Row[{
            Developer`Combobox[Dynamic[p["Catalog"]], 
                String,
                Prepend[catalogs, Automatic],
                ImageSize -> {198, 18},
                Method -> "PopupMenu"
            ],
            "Provide a catalog."
        }, Spacer[5]],
        propertiesEditor[],
        Row[{
            Button["Test", pretest[], ImageSize -> 50, Enabled -> StringQ[propToUrl@p]],
            Panel[Dynamic[errorMsg]]
        }, Spacer[10]], 
        Style["After specifying the connection properties, choose 'Finish' to save the data source configuration", 
            LinebreakAdjustments -> $linebreakAdjustments,
            LineIndent -> 0
        ],
        Spacings -> {Automatic, {2, {3 -> 1., 4 -> 1., 5 -> 1., 6 -> 1.5, 7 -> 1.5}}}
    ];

    Dynamic[
    	If[TrueQ[testIsAGo],
    		testIsAGo = Null;
	        $error = Null;
	        If[Quiet@Check[
	            conn = OpenSQLConnection[JDBC[p["Type"], propToUrl[p]], 
	                "Username" -> testU, "Password" -> testP,
	                Sequence @@ propertiesToOptions[p]
	            ];
	            $error = $MessageList;
	            False,
	
	            True
	        ] || Head[conn] =!= SQLConnection,
	            errorMsg = Style["Error: " <> ToString[First@$error /. Null -> "(no message)"], FontSlant -> Italic, FontColor -> Red],
	            errorMsg = Style["Test successful", FontSlant -> Italic, FontColor -> Darker@Green]
	        ];
	        CloseSQLConnection[conn];
    	];

        If[TrueQ[readCatsIsAGo],
            readCatsIsAGo = Null;
            Off[JDBC::error];
            conn = OpenSQLConnection[JDBC[p["Type"], propToUrl[p]],
            	    "Username" -> catU, "Password" -> catP,
            	    Sequence @@ propertiesToOptions[p]
            ];
            catalogs = SQLCatalogNames[conn];
            CloseSQLConnection[conn];
            On[JDBC::error];
            If[!ListQ[catalogs], catalogs = {}];
        ];

        Column[{
            Pane[
                Row[{
                    connectionMenu[page], 
                    Spacer[10],
                    connectionContents[page]
                }, Alignment -> {Center, Top}, BaselinePosition -> Top]
            ], 
            Pane[Row[{
                Button["Back", page--, Enabled -> !TrueQ[page == 1]],
                If[page != 6, 
                    DefaultButton["Next", page++, Enabled -> !TrueQ[page == 2 && !StringQ@p["Name"]]], 
                    DefaultButton["Finish", finish[]]
                ],
                Spacer[150],
                CancelButton[cdUpdate = False; NotebookClose[EvaluationNotebook[]]]
            }, Spacer[5]]]
        }, Alignment -> Right, Dividers -> {None, {2 -> Gray}}]
    ]

], Modal -> True, WindowTitle -> "Data Source Wizard"];


(*
 * Convert property downvalues to SQLConnection-suitable Options list.
 *)
propertiesToOptions[p_] := With[
    {d = DownValues[p]},
    FilterRules[
        Rule @@@ Transpose[{ReleaseHold@d[[All, 1, All, 1]], d[[All, 2]]}],
        Options[SQLConnection]
    ]
];


(*
 * Reconstructs url from server, port, db.  One or more may be blank.
 *)
propToUrl[p_] := Module[
    {separator, sp},
    separator = Switch[p["Type"],
    	"SQLite(Memory)", ":",
    	_, "/"
    ];
    sp = StringJoin @@ Riffle[{p["Server"], p["Port"]} /. "" -> Sequence[], ":"];
    StringJoin @@ Riffle[{sp, p["Database"] <> p["Attributes"]} /. "" -> Sequence[], separator]
];


(*  
 * Assigns a default port number etc. to a given source type.
 * The hostedQ property distinguishes hosted client-server RDBMSs (like MySQL) 
 * from in-process ones which do not require a hostname, port, etc. (like standalone HSQL).
 * Note that some hosted types (like Firebird) still require file specification.
 *)
SetAttributes[setSourceDefaults, HoldFirst];
setSourceDefaults[p_, type_] := With[{rules = type /. $RDBMSSpecificProperties},
    Scan[(p[#] = "";) &, {"Server", "Port", "Username", "Password", "Database", "Attributes"}];
    p["FileNameSetter spec"] = "Open";
    p["FileNameSetter patterns"] = {"All Files" -> {"*"}};
    Scan[(p[#] = (# /. rules);) &, rules[[All, 1]]];
];


(* Dialect-specific information (by driver name) is localized here and in WriteDataSource[].
 * There is some logic based on driver class in Connections.m; see SQLConnectionUsableQ.
 *)
$RDBMSSpecificProperties = {

    "Derby(Embedded)" -> {
        "GUIVariant" -> "File",
        "FileNameSetter spec" -> "Directory",
        "Attributes" -> ";create=true",
        "hostedQ" -> False
    },
        
    "Derby(Server)" -> {
    	"Username" -> None,
    	"Password" -> None,
        "Port" -> "1527",
        "GUIVariant" -> "Hosted, choose file",
        "FileNameSetter spec" -> "Directory",
        "Attributes" -> ";create=true",
        "hostedQ" -> True
    },

    "Firebird" -> {
        "Port" -> "3050",
        "GUIVariant" -> "Hosted, choose file",
        "FileNameSetter spec" -> "Open",
        "FileNameSetter patterns" -> {"Firebird Databases (*.fdb)" -> {"*.fdb"}, "All Files" -> {"*"}},
        "hostedQ" -> True
    },

    "H2(Embedded)" -> {
        "Username" -> "sa",
        "GUIVariant" -> "File, base name",
        "FileNameSetter spec" -> "Open",
        (* N.B. "*.h2.db" not supported *)
        "FileNameSetter patterns" -> {"H2 Databases (*.h2.db)" -> {"*.db"}, "All Files" -> {"*"}},
        "hostedQ" -> False
    },
        
    "H2(Memory)" -> {
        "Username" -> "sa",
        "GUIVariant" -> "Memory",
        "hostedQ" -> False
    },
        
    "H2(Server)" -> {
        "Port" -> "9092", (* 8082 is supposed to work for remote connections *)
        "GUIVariant" -> "Hosted, choose file base name",
        "FileNameSetter spec" -> "Open",
        "FileNameSetter patterns" -> {"H2 Databases (*.h2.db)" -> {"*.db"}, "All Files" -> {"*"}},
        "hostedQ" -> True
    },

    "hsqldb" | "HSQL(Standalone)" -> {
        "Username" -> "sa",
        "GUIVariant" -> "File, base name",
        "FileNameSetter spec" -> "Open",
        "FileNameSetter patterns" -> {"HSQL Databases (*.script, *.properties)" -> {"*.script", "*.properties"}, "All Files" -> {"*"}},
        "hostedQ" -> False
    },

    "HSQL(Memory)" -> {
        "Username" -> "sa",
        "GUIVariant" -> "Memory",
        "hostedQ" -> False
    },

    "HSQL(Server)" | "HSQL(Server+TLS)" | "HSQL(Webserver)" | "HSQL(Webserver+TLS)" -> {
        "Port" -> "9001",
        "Username" -> "sa",
        "GUIVariant" -> "Hosted, unlistable databases",
        (*"FileNameSetter spec" -> "Open",*)
        (*"FileNameSetter patterns" -> {"HSQL Databases ( *.script, *.properties)" -> {"*.script", "*.properties"}, "All Files" -> {"*"}},*)
        "hostedQ" -> True
    },

    "jtds_sqlserver" | "Microsoft SQL Server(jTDS)" -> {
        "Port" -> "1433",
        "GUIVariant" -> "Hosted",
        "hostedQ" -> True
    },

    "jtds_sybase" | "Sybase(jTDS)" -> {
        "Port" -> "5000",
        "GUIVariant" -> "Hosted",
        "hostedQ" -> True
    },

    "Microsoft Access(ODBC)" -> {
        "GUIVariant" -> "File",
        "FileNameSetter spec" -> "Open",
        "FileNameSetter patterns" -> {"Access Databases (*.accdb, *.mdb)" -> {"*.accdb", "*.mdb"}, "All Files" -> {"*"}},
        "hostedQ" -> False
    },
        
    "Microsoft Excel" -> {
        "GUIVariant" -> "File",
        "FileNameSetter spec" -> "Open",
        "FileNameSetter patterns" -> {"Excel Files (*.xls, *.xlsx, *.xlsm, *.xlsb)" -> {"*.xls", "*.xlsx", "*.xlsm", "*.xlsb"}, "All Files" -> {"*"}},
        "hostedQ" -> False
    },

    "mysql" | "MySQL(Drizzle)" | "MySQL(MariaDB)" -> {
        "Port" -> "3306",
        "GUIVariant" -> "Hosted",
        "hostedQ" -> True
    },

    "PostgreSQL" -> {
        "Port" -> "5432",
        "GUIVariant" -> "Hosted",
        "hostedQ" -> True
    },

    "odbc" | "ODBC(DSN)" -> {
        "GUIVariant" -> "Memory",
        "hostedQ" -> False
    },

    "Oracle(thin)" -> {
        "Port" -> "1521",
        "GUIVariant" -> "Hosted",
        "hostedQ" -> True
    },

    "SQLite" -> {
        "GUIVariant" -> "File",
        "FileNameSetter spec" -> "Open",
        "FileNameSetter patterns" -> {"SQLite Databases (*.db)" -> {"*.db"}, "All Files" -> {"*"}},
        "hostedQ" -> False
    },

    "SQLite(Memory)" -> {
        "GUIVariant" -> "Memory",
        "hostedQ" -> False
    },

    (* Note SAP MaxDB runs on 7200 by default, but it doesn't work in the connection string. *)
    _ -> {
    	"hostedQ" -> True,
    	"GUIVariant" -> "Hosted"
    }
};



(*
 * Write the new or edited connection
 *)
finishConn[p_, storePassword_, visibility_, jdbcProperties_] := Module[
    {dbrdir, dir, loc, name, conn, i, o},
    
    If[!TrueQ[storePassword], p["Password"] = "$Prompt"];

    p["Properties"] = DeleteCases[jdbcProperties, Rule["", ""]];
        
    If[!StringQ[p["Username"]] && p["Password"] === "", p["Password"] = None];
    If[p["Catalog"] === "", p["Catalog"] = Automatic];
    
    dbrdir = If[TrueQ@visibility, $UserBaseDirectory, $BaseDirectory];
    dir = FileNameJoin[{dbrdir, "DatabaseResources"}];
    
    (* This saves to $BaseDirectory or $UserBaseDirectory regardless of existing location. *)
    name = p["Name"];
    If[!StringQ[p["Location"]] || StringDrop[DirectoryName[p["Location"]], -1] =!= dir,
        (* Make a unique name *)
        i = 1;
        While[FileNames[name <> ".m", dir] =!= {},
            name = p["Name"] <> " (" <> ToString[++i] <> ")"
        ];
        loc = FileNameJoin[{dir, name <> ".m"}],

        (* else accommodate name changes *)
        loc = FileNameJoin[Append[Drop[FileNameSplit[p["Location"]], -1], name <> ".m"]]
    ];
    
    If[TrueQ@p["RelativePath"] && !p["hostedQ"] && p["Database"] === ExpandFileName@p["Database"],
        p["Database"] = absToRel[p["Database"], p["Location"]]
    ];
    
    o = DeleteDuplicates[
        Join[
            {
                "Version" -> DatabaseLink`Information`$VersionNumber,
                "Location" -> loc,
                "Name" -> name
            },
            propertiesToOptions[p]
        ],
        First@#1 === First@#2 &
    ];

    conn = SQLConnection[JDBC[p["Type"], propToUrl[p]], Sequence @@ o];
    
    If[!DirectoryQ[dir], CreateDirectory[dir];];

    Put[conn, loc];
    {name, loc}
];


(*
 * Some of the databases use chained extensions, which FileBaseName cant' handle.
 *)
altFileBaseName[""] := "";
altFileBaseName[fn_String] := With[
    {splat = Last@FileNameSplit[fn]},
    First@StringCases[splat, RegularExpression["^([^\\.]+).*$"] -> "$1"]
];


(*
 * For converting absolute paths to database files to relative ones (to accommodate
 * the relative path option.)
 *)
absToRel[of_, Null] := of;
absToRel[of_, wrt_] := absToRel[of, DirectoryName@wrt];
absToRel[of_, wrt_?DirectoryQ] := Module[
	{eOf, eWrt, len, common},

	{eOf,eWrt} = FileNameSplit[ExpandFileName[#]] & /@ {of, wrt};

    len = Min[Length /@ {eOf, eWrt}];
    common = LengthWhile[
        Transpose[Take[#, len] & /@ {eOf, eWrt}],
        First@# === Last@# &
    ];

    FileNameJoin[
    	Join[
    		If[Length@eWrt > common,
    			Table["..", {Length@eWrt - common}],
    			Drop[eWrt, common]
    		],
    		List@".",
    		Drop[eOf,common]
    	]
    ]
]

End[] (* DataSourceWizard`Private` *)

(* Author:          Dylan Boliske, dillont *)
(* Copyright:       Copyright 2013, Wolfram Research, Inc. *)
(*
 * Wizard for creating and editing jdbc connections.
 *)

Begin["`JDBCWizard`Private`"]


(*The toc used by the JDBC Connection Wizard.*)
JDBCMenu[page_Integer] := Panel[
    Grid[{
        List@Item[Style["Steps", FontSize -> 16], Alignment -> Left],
        Sequence @@ List /@ {
            "\n1. Overview",
            "2. Specify Name and Description",
            "3. Specify Visibility",
            "4. Specify Driver Class",
            "5. Specify Protocol"
        }}, 
        Alignment -> Left,
        ItemStyle -> {Automatic, Automatic, {{page + 1, 1} -> Bold}},
        Spacings -> {Automatic, 2},
        Dividers -> {None, {2 -> LightGray}},
        FrameStyle -> Automatic
    ],
    ImageSize -> {205, 350},
    Background -> White
]

$linebreakAdjustments = {0.95, 10, 100, 500, 500};
JDBCContentsTemplate[title_, stuff__, o:OptionsPattern[]] := Pane[
    Panel[Grid[{
            List@Item[Style[title, FontSize -> 16], Alignment -> Left],
            Sequence @@ (List /@ {stuff})
        },
        o,
        Alignment -> {Left, Baseline},
        Spacings -> {Automatic, 2},
        Dividers -> {None, {2 -> LightGray}},
        FrameStyle -> Automatic
    ], ImageSize -> {400, 400}]
];


staticJDBCContents[1] := JDBCContentsTemplate["Overview",
    Pane[Style[
        "This wizard will help you create a JDBCDriver configuration used to connect to a certain database with DatabaseLink." <>
        "\n\nSome information about the drivers is required to use this wizard." <>
        "\n\n\t" <> "\[Bullet]" <> " Driver class" <>
        "\n\t" <> "\[Bullet]" <> " URL protocol" <>
        "\n\nMany types of drivers are preconfigured within this wizard to contain custom wizard pages to make it easier to configure a data source. " <>
        "However, if a type of driver is not listed, one may be added using this wizard.", 
        LinebreakAdjustments -> $linebreakAdjustments,
        LineIndent -> 0
    ]]
];


(* Opens the Database Connection Wizard and allows the user to either edit or create a database connection *)
Options[JDBCDialog] = {
    "Name" -> Null,
    "Description" -> Null,
    "Driver" -> Null,
    "Protocol" -> Null,
    "Location" -> Null
};
JDBCDialog["New", o:OptionsPattern[]] := JDBCDialog[1, o, Sequence @@ Options[JDBCDialog]];
JDBCDialog["Edit", name_, o:OptionsPattern[]] := With[
	{listing = First@Cases[JDBCDrivers[], _?(("Name" /. List @@ #) === name &), 1, 1]},
    JDBCDialog[1, o, Sequence @@ Options[listing]]
];
JDBCDialog[pageIn_Integer, o:OptionsPattern[]] := CreateDialog[DynamicModule[
    {
        page = pageIn,
            	
    	visibility = True,
    	
    	finish,
    	JDBCContents,
    	
    	(* For all driver properties *)
        p
    },

    (* Read in properties from options *)
    Scan[
        (p[First@#] = Last@#;) &,
        Reverse[{o}]
    ];
    visibility = If[StringQ[p["Location"]] && StringMatchQ[p["Location"], $BaseDirectory <> "*"], False, True];


    (*
     * For saving out driver
     *)
    finish[] := Module[
        {name, fn, nb = EvaluationNotebook[], err},
        If[Quiet@Check[
            {name, fn} = finishJDBC[p, visibility];
            err = $MessageList;
            False,

            True],

            CreateDialog[{
                Column[{
                    StringForm["Error saving driver configuration '``' to", name],
                    Style[fn <> ":"],
                    ToString[Style[First@err, {Red, Italic}], StandardForm],
                    DefaultButton[NotebookClose[EvaluationNotebook[]];]
                }, Center]
            }, WindowFloating -> True, Modal -> True],
            
            CreateDialog[{
                Column[{
                    StringForm["Saved driver configuration '``' to", name],
                    Style[fn],
                    DefaultButton[NotebookClose[EvaluationNotebook[]];]
                }, Center]
            }, WindowFloating -> True, Modal -> True];
            NotebookClose[nb]
        ]
    ];

    (*
     * Contents for each page of the wizard.
     *)
    JDBCContents[1] := staticJDBCContents[1];
    JDBCContents[2] := With[{jdbcns = JDBCDriverNames[]}, JDBCContentsTemplate["Name and Description",
        Style[
            Apply[StringJoin, ToString[#, StandardForm] & /@ {
                Style["\nSpecify the name.", FontColor -> Black],
                If[MemberQ[jdbcns, p["Name"]], Style[" A driver configuration with this name already exists.", FontSlant -> Italic, FontColor -> Black], ""]
            }],
            LineIndent -> 0
        ],
        InputField[Dynamic[p["Name"]], String, ImageSize -> {375, 20}],
        "Specify the description.",
        InputField[Dynamic[p["Description"]], String, ImageSize -> {375, 150}], 
        Style[
            Apply[StringJoin, ToString[#, StandardForm] & /@ {
                Style["After specifying the name and description, choose 'Next' to select the visibility of the configuration.", FontColor -> Black],
                If[MemberQ[jdbcns, p["Name"]], Style[" Continuing will eventually overwrite existing settings.", FontSlant -> Italic, FontColor -> Black], ""]
            }],
            LineIndent -> 0, FontColor -> Black, LinebreakAdjustments -> $linebreakAdjustments
        ],
        Spacings -> {Automatic, {2, {3 -> Automatic, 5 -> Automatic}}}
    ]];
    
    JDBCContents[3] := JDBCContentsTemplate["Specify Visibility",
        "\nPlease specify the visibility of the driver configuration.",
        Row[{
            RadioButton[Dynamic[visibility], True, BaselinePosition -> Scaled[0.2]], 
            Style["User Level", FontSize -> 12],
            Spacer[10], 
            RadioButton[Dynamic[visibility], False, BaselinePosition -> Scaled[0.2]], 
            Style["System Level", FontSize -> 12]
        }, Spacer[1.0]], 
        Style["After specifying the visibility of the driver configuration, choose 'Next' to specify the driver class.", LineIndent -> 0,
            LinebreakAdjustments -> $linebreakAdjustments
        ],
        Spacings -> {Automatic, {2, {3 -> 1.5, 4 -> 1.5}}}
    ];
    
    JDBCContents[4] := JDBCContentsTemplate["Driver Class",
        Style["\nSpecify the JDBC driver class. This class should be located on the Java classpath as directed by JLink. " <>
            "It can be automatically found in the Java directory of this application or any other application." <>
            "\n\nExamples:" <>
            "\n\n\t\[Bullet] org.hsqldb.jdbcDriver" <>
            "\n\t\[Bullet] com.mysql.jdbc.Driver",
            LinebreakAdjustments -> $linebreakAdjustments,
            LineIndent -> 0
        ],
        InputField[Dynamic[p["Driver"]], String],
        "After choosing the driver class, choose 'Next' to specify a protocol.",
        Spacings -> {Automatic, {2, {3 -> 1.5}}}
    ];
    
    JDBCContents[5] := JDBCContentsTemplate["Protocol",
    	Style["Specify the protocol used to prefix the URL which connects to the database. " <>
        	"By specifying the protocol users can omit the protocol when specifying a URL in a SQLConnection. " <>
        	"The protocol specified by the JDBCDriver is automatically prepended to the URL." <>
    	    "\n\nExamples:" <>
        	"\n\n\t\[Bullet] jdbc:hsqldb:" <>
    	    "\n\t\[Bullet] jdbc:mysql://",
            LinebreakAdjustments -> $linebreakAdjustments,
            LineIndent -> 0
        ],
        InputField[Dynamic[p["Protocol"]], String],
        "After choosing the protocol, choose 'Finish' to save the driver configuration.",
        Spacings -> {Automatic, {2, {3 -> 1.5}}}
    ];

    Dynamic[
        Column[{
            Pane[
                Row[{
                    JDBCMenu[page], 
                    Spacer[15],
                    JDBCContents[page]
                }, Alignment -> {Center, Top}, BaselinePosition -> Top]
            ], 
            Pane[Row[{
                Button["Back", page--, Enabled -> If[page == 1, False, True]],
                If[page != 5, 
                    DefaultButton["Next", page++, Enabled -> If[(page == 2 && !StringQ@p["Name"]) || (page == 4 && !StringQ@p["Driver"]), False, True]], 
                    DefaultButton["Finish", finish[]]
                ],
                Spacer[215],
                CancelButton[NotebookClose[EvaluationNotebook[]]]
            }, Spacer[5]]]
        }, Alignment -> Right, Dividers -> {None, {2 -> Gray}}]
    ]

], Modal -> True, WindowTitle -> "JDBC Driver Wizard"];


(*
 * Convert property downvalues to Options list.
 *)
propertiesToOptions[p_] := With[
    {d = DownValues[p]},
    DeleteCases[
        Rule @@@ Transpose[{ReleaseHold@d[[All, 1, All, 1]], d[[All, 2]]}],
        Rule[Null, _]
    ]
];


finishJDBC[p_, visibility_] := Module[
	{basedir, dir, basebasename, basename, loc, i, jdbc, o},

    basedir = If[TrueQ@visibility, $UserBaseDirectory, $BaseDirectory];
    dir = FileNameJoin[{basedir, "DatabaseResources"}];

    (*
     * This saves to $BaseDirectory or $UserBaseDirectory regardless of existing location.
     * For filename, prioritize the basename of Location over Name.
     *)
    basebasename = basename = If[StringQ[p["Location"]], FileBaseName[p["Location"]], p["Name"]];
    (* If Location looks invalid, or directory is changing ... *)
    If[!StringQ[p["Location"]] || StringDrop[DirectoryName[p["Location"]], -1] =!= dir,
        (* Make a unique name *)
        i = 1;
        While[FileNames[basename <> ".m", dir] =!= {},
            basename = basebasename <> " (" <> ToString[++i] <> ")"
        ];
        loc = FileNameJoin[{dir, basename <> ".m"}],

        (* else *)
        loc = p["Location"]
    ];

    o = DeleteDuplicates[
        Join[
            {
                "Version" -> DatabaseLink`Information`$VersionNumber,
                "Location" -> loc
            },
            propertiesToOptions[p]
        ],
        First@#1 === First@#2 &
    ];

    jdbc = JDBCDriver @@ o;

    If[!DirectoryQ[dir], CreateDirectory[dir];];

    Put[jdbc, loc];
    {p["Name"], loc}
]


End[]
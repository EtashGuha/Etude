BeginPackage["Compile`Core`Utilities`Visualization`DotGraphics`"]

ToDotGraphics

ToDotGraphicsString

DotGraphicsStringImport

Begin["`Private`"]

Needs["Compile`Core`IR`ProgramModule`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`IR`Instruction`InstructionQ`"]
Needs["Compile`Core`IR`Instruction`LabelInstruction`"]
Needs["Compile`Core`IR`Instruction`BranchInstruction`"]
Needs["CompileUtilities`Markup`"] (* For $UseANSI *)

nodeTemplate := nodeTemplate = StringTemplate["`From` -> `To` [ `Attrs` ]\n"]
edgeTemplate := edgeTemplate = StringTemplate["`ID` [ `Attrs` ]\n"]

addDotHeader[bag_, name_] :=
    (
        Internal`StuffBag[bag, "digraph " <> name <> " {"];
        Internal`StuffBag[bag, "graph [compound=true layout=dot splines=spline nodesep=1 ranksep=\"1.5 equally\" overlap=vpsc];"];
        Internal`StuffBag[bag, "node [shape=record fontsize=10 fontname=\"Verdana\"];"];
    )


ToDotGraphics[pm_?ProgramModuleQ] :=
    GraphicsRow[
        Map[
		   Function[{fm},
		       With[ {str = ToDotGraphicsString[fm]},
					DotGraphicsStringImport[
						str
					]
				]
		   ],
		   pm["getFunctionModules"]
	    ],
	    Frame -> All,
	    ImageSize -> {Scaled[0.5], All}
    ]
	
ToDotGraphicsString[fm_?FunctionModuleQ] :=
	Block[{$FrontEnd = Null, $UseANSI = False},
	With[{bag = Internal`Bag[]},
        addDotHeader[bag, escapeGraphName[fm["name"]]];
		toDot[bag, fm];
        fm["scanBasicBlocks",
             Function[{bb},
                bb["scanInstructions", makeCallEdges[bag, bb, #]&]
             ]
        ];
		Internal`StuffBag[bag, "}"];
		Quiet[
			StringRiffle[
				Internal`BagPart[bag, All],
				"\n"
			]
		]
	]]
	
ToDotGraphics[fm_?FunctionModuleQ] :=
	With[ {str = ToDotGraphicsString[fm]},
		DotGraphicsStringImport[
			str
		]
	]

DotGraphicsStringImport[str_?StringQ] :=
	Module[{tf, pdf, cmd, st, res},
		tf = makeTemporaryFileName["dot"];
		WriteString[tf, str];
		pdf = makeTemporaryFileName["pdf"];
		cmd = {"dot", "-o", pdf, "-T", "pdf", tf};
		st = RunProcess[
			cmd,
			All,
			ProcessEnvironment -> <|
				"PATH" -> "/usr/local/bin:/usr/bin:/bin"
			|>
		];
		res = If[st["ExitCode"] =!= 0,
		    CopyToClipboard[str];
			Failure["DotGraphics", <|
				"MessageTemplate" :> "The Dot command failed to run with the following error `StandardError`", 
				"MessageParameters" -> Join[
					st
				],
				"DotString" -> str
			|>],
			ReplacePart[
				cleanupGraphics[
					First[
					   Import[pdf, "PDF"]
				    ]
				],
				2 -> (ImageSize -> {Scaled[0.5], All})
			]
		];
		If[DirectoryQ[$TemporaryDotGraphicsDirectory],
            DeleteDirectory[$TemporaryDotGraphicsDirectory, DeleteContents->True]
		];
        
        res
	]
DotGraphicsStringImport[_] :=
	Failure["DotGraphics", <|
		"MessageTemplate" :> "Failed to serialize the IR into a Dot format.", 
		"MessageParameters" -> <||>
	|>]
	
	
weirdSpace = FilledCurve[
    List[List[List[0, 2, 0], List[0, 1, 0], List[0, 1, 0]], List[List[0, 2, 0], List[0, 1, 0]]], _
]
cleanupGraphics[g_] :=
    g /. weirdSpace -> {}

toDot[bag_, fm_?FunctionModuleQ] := 
	With[{label = fm["name"] <> " function module"},
        Internal`StuffBag[bag, makePortName[fm["name"]] <> " [style = invis];"];
		fm["scanBasicBlocks", toDot[bag, #]&];
		Internal`StuffBag[bag, "{ rank = same; "];
        fm["scanBasicBlocks",
           Function[{bb},
        	       Internal`StuffBag[bag, makePortName[bb] <> " "]
           ];
        ];
        Internal`StuffBag[bag, "};\n"];
        fm["scanBasicBlocks",
             Function[{bb},
                bb["scanInstructions", makeBranchEdges[bag, bb, #]&]
             ]
        ];
        Internal`StuffBag[bag,
            "\tlabel=<" <>
                "<FONT POINT-SIZE=\"26.0\"><B>" <> label <> "</B></FONT>>;"];
	]

makePortName[str_?StringQ] :=
    "\"" <> str <> "\""
makePortName[s_Symbol] :=
    "\"" <> StringTrim[makePortName[Context[s]], "\""] <> "`" <> StringTrim[makePortName[SymbolName[s]], "\""] <> "\"" 
makePortName[h_[args__]] :=
    makePortName[h] <> "args" <> StringRiffle[makePortName /@ {args}, "arg"] 
makePortName[fm_?FunctionModuleQ] :=
    "f" <> ToString[fm["id"]]
makePortName[bb_?BasicBlockQ] :=
    "b" <> ToString[bb["id"]]
makePortName[inst_?InstructionQ] :=
    "i" <> ToString[inst["id"]]

ClearAll[makeBranchEdges]
ClearAll[makeCallEdges]
(*makeCallEdges[bag_, bb_, term_?CallInstructionQ] :=
    If[ConstantValueQ[term["function"]],
        Internal`StuffBag[bag, StringRiffle[{"\t", makePortName[term], " -> ", makePortName[term["function"]["value"]], ":start;"}, ""]]
    ]*)
makeBranchEdges[bag_, bb_, term_?BranchInstructionQ] :=
    If[term["isConditional"],
        Internal`StuffBag[bag, StringRiffle[{"\t", makePortName[term["getBasicBlock"]], ":left", " -> ", makePortName[term["getOperand", 1]], ":start", "[penwidth=4.0,weight=1.0,splines=\"spline\",color=\"#68BC36\",headport=_, tailport=sw];"}, ""]];
        Internal`StuffBag[bag, StringRiffle[{"\t", makePortName[term["getBasicBlock"]], ":right", " -> ", makePortName[term["getOperand", 2]], ":start", "[penwidth=4.0,weight=0.5,splines=\"spline\",color=\"#D73027\",headport=_, tailport=se];"}, ""]];
        , (* Else *)
        Internal`StuffBag[bag, StringRiffle[{"\t", makePortName[term["getBasicBlock"]], ":end", " -> ", makePortName[term["getOperand", 1]], ":start[penwidth=4.0,weight=0.5,splines=\"spline\",headport=c,tailport=s];"}, ""]]
    ]
toDot[bag_, bb_?BasicBlockQ] := 
	With[{insts = bb["getInstructions"]},
		With[{label = StringRiffle[Flatten[dotForm /@ insts], "\n"]}, 
			Internal`StuffBag[bag, StringJoin[{
				"\t",
				makePortName[bb],
				" [shape=Mrecord,fontname=Courier,label=<",
				"\n",
				"<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"2\" PORT=\"start\">",
                "\n",
				"<TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\" ><FONT POINT-SIZE=\"18.0\"><B>",
                "\n",
				escape[bb["fullName"]],
				"</B></FONT></TD></TR>",
                "\n",
				If[Length[insts] <= 2,
					"",
					"<HR/>"
				],
				label,
                "\n",
				"</TABLE>",
                "\n",
				">];"
			}]];
		]
	]

dotForm[inst_] := (
	Which[
		LabelInstructionQ[inst],
			Nothing,
		BranchInstructionQ[inst],
		  If[inst["isConditional"],
			{
				"<TR><TD ALIGN=\"LEFT\" COLSPAN=\"2\" PORT=\"end\">" <> escapeHTML[inst["toHTML"]] <> "</TD></TR>",
				"<HR/>\n",
				"<TR><TD ALIGN=\"CENTER\" PORT=\"left\" SIDES=\"R\" bgcolor=\"#68BC36\"><B>True</B></TD>",
				"<VR/>",
				"<TD ALIGN=\"CENTER\" PORT=\"right\" bgcolor=\"#D73027\"><B>False</B></TD> </TR>"
			},
			{
                "<TR><TD ALIGN=\"LEFT\" PORT=\"end\">" <> escapeHTML[inst["toHTML"]] <> "</TD></TR>"
			}
		  ],
		True,
			"<TR><TD ALIGN=\"LEFT\" COLSPAN=\"2\" >" <> escapeHTML[inst["toHTML"]] <> "</TD></TR>"
	]
)
	
escapeGraphName[str_] :=
    StringReplace[str, {
        "\[Lambda]" -> "&lambda;",
        "$" -> "_",
        "`" -> "_"
    }]
    
escape[str_] :=
	StringTrim@StringReplace[str, {
		"->" -> "&rarr;",
		"<-" -> "&larr;",
        "\[Rule]" -> "&rarr;",
		"\"" -> "",
		"{" -> "\\{",
		"}" -> "\\}",
		">" -> "&gt;",
		"<" -> "&lt;",
		"\[Lambda]" -> "&lambda;",
        "\[DottedSquare]" -> "&#9633;", 
		"$" -> "_"
	}]
	
escapeHTML[str_] :=
    StringTrim@StringReplace[str, {
        "->" -> "&rarr;",
        "<-" -> "&larr;",
        "[" -> "&#91;",
        "]" -> "&#93;"
        (*,
        "&#9633;" -> "_"*)
    }]
	
(********************************************)
(********************************************)
(********************************************)
createDirIfDoesNotExist[dir_String, failOnExisting:(True|False):False] :=
    Quiet[
        Check[CreateDirectory[dir], If[failOnExisting, $Failed, dir], CreateDirectory::filex],
        {CreateDirectory::filex}
    ];

createDirIfDoesNotExist[parts__String, failOnExisting:(True|False):False] :=
    createDirIfDoesNotExist[FileNameJoin[{parts}], failOnExisting];

generateRandomString[n_Integer]:=
    StringJoin[
        ToString @ FromCharacterCode @ RandomInteger[{97,122},n],
        ToString @ AbsoluteTime[DateString[]]
    ];
        
makeTemporarySubdirectory[root_String?DirectoryQ, maxAttempts_:10000]:=
    Module[{dir, ctr = 0 },
        While[
        	FailureQ[(dir = createDirIfDoesNotExist[root, generateRandomString[10]])]
        	&& ++ctr < maxAttempts
        ];
        dir
    ];
   
$TemporaryDotGraphicsDirectory := createDirIfDoesNotExist[$TemporaryDirectory, "DotGraphics"];

 
makeTemporaryFileName[extension_String, directory:_String?DirectoryQ|Automatic:Automatic]:=
    With[{dir = If[directory === Automatic, $TemporaryDotGraphicsDirectory, directory]},
        FileNameJoin[{dir, generateRandomString[10] <> "." <> extension}]
    ];


End[] 
  
    
EndPackage[]

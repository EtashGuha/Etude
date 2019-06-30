
BeginPackage["Compile`Core`IR`ProgramModule`"]

ProgramModule;
ProgramModuleQ;
ProgramModuleClass;
CreateProgramModule

DeserializeProgramModule

Begin["`Private`"] 

Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`IR`GetElementTrait`"]
Needs["Compile`Core`Utilities`Visualization`DotGraphics`"]
Needs["Compile`Core`IR`ExternalDeclaration`"]
Needs["Compile`Utilities`Serialization`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`PassManager`PassRunner`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Markup`"]



RegisterCallback["DeclareCompileClass", Function[{st},
ProgramModuleClass = DeclareClass[
	ProgramModule,
	<|
		"clone" -> (clone[Self, ##]&),
		"scanGlobalValues" -> Function[{fun},
			Scan[fun, Self["getGlobalValues"]]
		],
		"scanExternalDeclarations" -> Function[{fun},
			Scan[fun, Self["getExternalDeclarations"]]
		],
		"scanFunctionModules" -> Function[{fun},
			Scan[fun, Self["getFunctionModules"]]
		],
		"scanMetaInformation" -> Function[{fun},
			Scan[fun, Self["getMetaInformation"]]
		],
		"getFunctionModule" -> Function[{name},
			SelectFirst[Self["getFunctionModules"], #["name"] === name&]
		],
		"addFunctionModule" -> Function[{fm},  addFunctionModule[Self, fm]]
		,
		"addFunctionModuleIfNotPresent" -> Function[{fm}, addFunctionModuleIfNotPresent[Self, fm]]
		,
		"getFunctionModules" -> Function[{},
			Self["functionModules"]["get"]
		],
		"getTypeDeclarations" -> Function[{},
			Self["typeDeclarations"]["get"]
		],
		"getFunctionDeclarations" -> Function[{},
			Self["functionDeclarations"]["get"]
		],
		(* TODO: This won't make any sense if we ever have a module/privacy system *)
		"topLevelFunctions" -> Function[{},
			Module[{topLevelFuns = {}},
				Scan[Function[fm,
						If[!fm["getProperty", "localFunction", False],
							AppendTo[topLevelFuns, fm]
						]
					],
					Self["getFunctionModules"]
				];
				topLevelFuns
			]
		],
		"exportedFunctions" -> Function[{},
			Module[{funs = {}},
				Scan[Function[fm,
						If[TrueQ[fm["getProperty", "exported", False]],
							AppendTo[funs, fm]
						]
					],
					Self["getFunctionModules"]
				];
				funs
			]
		],
		"exprTypeQ" ->  Function[{},
			Lookup[ Self["getProperty", "environmentOptions"], "TypeSystem", Null] === "Expr"
		],
		"runPass" -> (runPass[Self, ##]&),
		"dispose" -> Function[{}, dispose[Self]],
		"getElements" -> Function[ {},  Self["getFunctionModules"]],
		"serialize" -> (serialize[ Self, #]&),
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"toDot" -> Function[{}, ToDotGraphics[Self]],
        "toString" -> Function[{},
            toString[Self]
        ],
        "toHTML" -> Function[{},
            toHTML[Self]
        ],
		"prettyPrint" -> (prettyPrint[Self]&),
		"makePrettyPrintBoxes" -> Function[{}, makePrettyPrintBoxes[Self]],
		"initialize" -> Function[{},
							Self["setPassStore", CreateReference[<||>]];
							Self["setProperties", CreateReference[<||>]]]
	|>,
	{
		"mexpr",
		"globalValues",
		"typeDeclarations",
		"externalDeclarations",
		"functionDeclarations",
		"functionModules",
		"metaInformation",
		"typeEnvironment",
		"passStore",
		"entryFunctionName",
		"properties"
	},
	Predicate -> ProgramModuleQ,
	Extends -> {
		ClassPropertiesTrait,
		GetElementTrait
	}
]
]]

runPass[pm_, passes_?ListQ, opts:OptionsPattern[]] :=
    RunPasses[passes, pm, opts]
runPass[pm_, pass_, opts:OptionsPattern[]] :=
    RunPass[pass, pm, opts]
runPass[pm_, args___] :=
    ThrowException[{"Invalid call to runPass. Expecting a pass or a list of passes, but not ", {args}}]

addFunctionModuleIfNotPresent[ self_, fm_] :=
	Module[ {name = fm["name"], missing},
		missing = MissingQ[self["getFunctionModule", name]];
		If[missing,
			addFunctionModule[self, fm]];
		missing
	]		

addFunctionModule[self_, fm_] :=
	(
	fm["setProgramModule", self];
	fm["setTypeEnvironment", self["typeEnvironment"]];
	self["functionModules"]["appendTo", fm]
	)



prettyPrint[self_] :=
	CellPrint[
	    Cell[
	        BoxData[
	            self["makePrettyPrintBoxes"]
			],
   			"Text"
	    ]
	]

CreateProgramModule[mexpr_, globalVals_, typeDecls_, externalDecls_, functionModules_, metaInformation_] :=
	Module[{pm},
		pm = CreateObject[
			ProgramModule,
			<|
				"mexpr" -> mexpr,
				"globalValues" -> globalVals,
				"typeDeclarations" -> typeDecls,
				"functionDeclarations" -> CreateReference[{}],
				"externalDeclarations" -> CreateExternalDeclaration[externalDecls],
				"functionModules" -> functionModules,
				"metaInformation" -> metaInformation
			|>
		];
		pm
	]


DeserializeProgramModule[ env_, "ProgramModule"[ data_]] :=
	deserialize[env, data, 
		env["getElementNoDeserialize", data, "functionModules"], 
		env["getElementNoDeserialize", data, "globalValues"], 
		env["getElementNoDeserialize", data, "typeDeclarations"],
		env["getElementNoDeserialize", data, "externalDeclarations"],
		env["getElementNoDeserialize", data, "metaInformation"]
	]

deserialize[ env_, data_, fmList_, globalsValuesList_, typeDeclarationsList_, externalDeclarationsIn_, metaInformationList_] :=
	Module[ {pm, fms, globalValues, typeDeclarations, externalDeclarations, metaInformation},
		fms = CreateReference[Map[ env["deserialize", #]&, fmList]];
		globalValues = CreateReference[globalsValuesList];
		typeDeclarations = CreateReference[typeDeclarationsList];
		externalDeclarations = DeserializeExternalDeclaration[env, externalDeclarationsIn];
		metaInformation = CreateReference[metaInformationList];
		pm = CreateProgramModule[ None, globalValues, typeDeclarations, externalDeclarations, fms, metaInformation];
		pm["setTypeEnvironment", env["typeEnvironment"]];
		Scan[ (
				#["setProgramModule", pm];
				#["setTypeEnvironment", env["typeEnvironment"]];
			   )&, fms["get"]];
		pm
	]


serialize[ self_, env_] :=
	Module[ {data},
		data = <| "functionModules" -> Map[ #["serialize", env]&, self["getFunctionModules"]],
				  "globalValues" -> self["globalValues"]["get"],
				  "typeDeclarations" -> self["typeDeclarations"]["get"],
				  "externalDeclarations" -> self["externalDeclarations"]["serialize", env],
				  "metaInformation" -> self["metaInformation"]["get"]
				  |>;
		"ProgramModule"[ data]
	]


clone[self_, env_] := 
	deserialize[env, self,
		self["getFunctionModules"],
		self["globalValues"]["get"],
		self["typeDeclarations"]["get"],
		self["externalDeclarations"]["get"],
		self["metaInformation"]["get"]
	]

clone[self_] :=
	clone[self, CreateCloneEnvironment[]] 

dispose[self_] :=
	Module[{},
		self["scanFunctionModules", #["dispose"]&];
		self["properties"]["set", <||>];
		self["mexpr"]["dispose"]
	]


(**************************************************)
(**************************************************)
	
dividerStringStart   = "(**********************************************************";
dividerStringEnd     = " *********************************************************)\n";
dividerStringRow     = "(**********************************************************)\n";

functionModuleHeader = " * Function Modules -------------------------------------";
typeDeclsHeader      = " * Type Declarations ------------------------------------";
functionDeclsHeader  = " * Function Declarations --------------------------------";

declsString[decls_, header_] :=
	With[{},
		If[decls === {},
			Return[""]
		];
		StringJoin[
            GrayText[dividerStringStart],
            If[header === "",
            	"\n",
            	{
		            "\n",
		            	GrayText[header],
		            "\n",
		            	GrayText[dividerStringEnd],
					"\n"
            	}
            ],
			StringRiffle[
				Map[
					Function[{decl},
						decl["toString"]
					],
					decls
				],
				"\n"
			],
			"\n"
		]
	]
	
fmsString[self_] :=
	With[{
		decls = self["getTypeDeclarations"],
		fms = self["getFunctionModules"],
		fmDivider = StringJoin[
    		"\n",
			 GrayText[dividerStringRow <> dividerStringRow],
    		"\n"
		]
    },
	With[{
		base = StringJoin[
			Riffle[
				Map[
					Function[{fm},
						fm["toString"]
					],
					fms
				],
				fmDivider
			]
		]
	},
	    If[decls === {},
	    	  base,
	    	  StringJoin[
	            GrayText[dividerStringStart],
	            "\n",
	            	GrayText[functionModuleHeader],
	            "\n",
	            	GrayText[dividerStringEnd],
				"\n",
	            base
	        ]
	    ]
	]]
toString[self_] :=
	StringTrim[
		StringJoin[
			declsString[self["getTypeDeclarations"], typeDeclsHeader],
			 "\n", 
			 declsString[self["getFunctionDeclarations"], functionDeclsHeader],
			 "\n",
			 fmsString[self]
		]
	]

toHTML[self_] :=
    Block[{$FrontEnd = Null, $UseANSI = False, $UseHTML = True},
        StringReplace[
            self["toString"],
            "\n" -> "<br/>"
        ]
    ]
	
(**************************************************)
(**************************************************)

icon := Graphics[Text[
  Style["Prog\nMod", GrayLevel[0.7], Bold, 
   CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];  


toBoxes[pm_?ProgramModuleQ, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"ProgramModule",
		pm,
  		icon,
  		{
  		    (*If[pm["getTypeDeclarations"] === Undefined,
  		    		Nothing,
  		    		BoxForm`SummaryItem[{"typeDeclarations: ", Map[#["name"]&, pm["getTypeDeclarations"]]}]
  		    ],*)
  		    BoxForm`SummaryItem[{"functionModules: ", Map[#["name"]&, pm["getFunctionModules"]]}]
  		},
  		Map[BoxForm`SummaryItem[{#["name"] <> ": ", LazyFormat[FunctionModuleQ, #]}] &, Apply[List, pm["getFunctionModules"]]], 
  		fmt,
		"Interpretable" -> False
  	]

makePrettyPrintBoxes[self_] :=
	With[{fms = self["getFunctionModules"]},
		Table[
			FrameBox[
				RowBox[{
					StyleBox[fm["name"] <> ":", Bold, "Section"],
					"\[NewLine]",
				    fm["makePrettyPrintBoxes"]
				}]
			],
			{fm ,fms}
		]
	]
End[]
EndPackage[]

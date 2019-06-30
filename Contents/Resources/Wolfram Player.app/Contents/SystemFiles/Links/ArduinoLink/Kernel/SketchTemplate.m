(* ::Package:: *)

(* Wolfram Language Package *)

(*==========================================================================================================
			
					SKETCH TEMPLATE
			
Author: Ian Johnson
			
Copyright (c) 2015 Wolfram Research. All rights reserved.			

Sketch Template is a package to setup an Arduino Uno sketch with the specified options.

CURRENT SUPPORTED BOARDS:
~Arduino Uno

USER ACCESSIBLE FUNCTIONS:
sketchSetup

==========================================================================================================*)


BeginPackage["SketchTemplate`"]
(* Exported symbols added here with SymbolName::usage *)  

sketchSetup::usage="sketchSetup wil setup the wolfram sketch with the options specified";

Begin["`Private`"] (* Begin Private Context *) 

(*=========================================================================================================
============================ SKETCH SETUP ================================================================
===========================================================================================================

sketchSetup will setup the wolfram library for the Arduino with the user's source code if necessary, and
will also add in any libraries the user requests, etc.



===========================================================================================================
=====================ALGORITHM=============================================================================
===========================================================================================================

The general algorithm for this function is as follows:
===========================================================================================================
Step 1.		First import the sketch file from the paclet
===========================================================================================================
Step 2. 	Place in the references to the libraries requested
===========================================================================================================
Step 3. 	Place in the source code for their functions
===========================================================================================================
Step 4. 	Generate the function code for callFunctionWithArguments, etc.
===========================================================================================================


===========================================================================================================
=============================PARAMETERS====================================================================
===========================================================================================================

	fileLocation - the location of the sketch to use as the template
	
===========================================================================================================


===========================================================================================================
================================RETURN=====================================================================
===========================================================================================================
Return Value - 
	$Failed - if the file wasn't found
	The full sketch text - successful creation of the program file 

===========================================================================================================


===========================================================================================================
==================================OPTIONS==================================================================
===========================================================================================================

	"FileLocation" - the location of the sketch file to use
	"UserFuncSource" - the verbatim source code of the user's function
	"UserFuncTypeInfo" - the information on the argument structure, the function name's, etc.
			(Note: should be of the form)
	"Libraries" - all of the libraries to include in this sketch
	"Debug" - whether or not debugging protocols should be followed
	Initialization - initialization code for anything later in the program.

===========================================================================================================


=================================FUNCTION CODE FOLLOWS=====================================================
===========================================================================================================
=========================================================================================================*)

sketchSetup::fileNotFound="The file `1` was not found";
Options[sketchSetup]=
{
	"UserFuncSource"->None,
	"UserFuncTypeInfo"->None,
	"Libraries"->{},
	"Debug"->False,
	Initialization->None,
	"BootFunctionOptions"->None
};
sketchSetup[fileLocation_,OptionsPattern[]]:=Module[{programRawText},
	(
		(*get the program text*)
		(*Check and see if what the user passed actually exists, if it does, then we can use that, if not return $Failed*)
		programRawText = If[FileExistsQ[fileLocation],
			(*the file exists*)
			Import[fileLocation,"Text"],
			(*doesn't exist, return $Failed*)
			(
				Message[sketchSetup::fileNotFound,fileLocation];
				Return[$Failed];
			)
		];
		templateOptions = <||>;
		(*add the libraries requested by the user*)
		AppendTo[templateOptions,addLibraries[OptionValue["Libraries"]]];
		(*add the source code for the functions*)
		AppendTo[templateOptions,addFunctionSourceCode[OptionValue["UserFuncSource"]]];
		(*add the functions*)
		AppendTo[templateOptions,#]&/@generateFunctions[OptionValue["UserFuncTypeInfo"]];
		(*add the initialization*)
		AppendTo[templateOptions,addInitialization[OptionValue[Initialization]]];
		(*add the setup function*)
		AppendTo[templateOptions,addBootFunction[OptionValue["BootFunctionOptions"]]];
		(*for debugging purposes. timestamp the sketch and put some information about the system information in the sketch*)
		AppendTo[templateOptions,"time"->DateString[]];
		AppendTo[templateOptions,"sysInfo"->systemInformationString[]];
		(*return the raw text replaced with the templateOptions association*)
		StringTemplate[programRawText][Association@templateOptions]
	)
];

(*this produces a nicely formatted string for the sketch*)
systemInformationString[]:=Module[{},
	StringJoin[
		"\nKernel: \n",
		Prepend[
			Riffle[
				(*this parses out from SystemInformation the kernel information and gets the date from it *)
				ToString/@("Kernel" /. SystemInformation["Small"]/.(date_DateObject :> DateString[date])),
				"\n\t"
			],
			"\t"
		],
		"\nFrontEnd: \n",
		Prepend[
			(*only parse out SystemInformation for FrontEnd info if we have a front end*)
			If[$FrontEnd=!=Null,
				(*THEN*)
				(*front end should exist, so we can parse out SystemInformation*)
				Riffle[
					ToString/@("FrontEnd" /. SystemInformation["Small"]/.(date_DateObject :>DateString[date])),
					"\n\t"
				],
				(*ELSE*)
				(*the front end doesn't exist, so we should just put in a $Failed string*)
				{"$Failed"}
			],
			"\t"
		]
		,
		"\n"
	]
];


addBootFunction[setupOpts_]:=Module[{},
	(
		If[setupOpts===None,
			(*THEN*)
			(*we don't have to add a task, so just put in an empty string*)
			(
				Return["preTaskSetup"->""];
			),
			(*ELSE*)
			(*we need to get the Scheduling options out of the options association*)
			(
				Return[
					(*this is the default setup function function in C code*)
					(*all we need to do with it is set the scheduling options*)
					"preTaskSetup"->"
	//create a task
    task * preTask = (task *)calloc(1,sizeof(task));
    //check to make sure that the allocation worked
    if(preTask)
    {
        //then make void functions and pass that in
        arguments * voidArgs = (arguments *)calloc(1,sizeof(arguments));
        setLongArgNum(voidArgs,0);
        setFloatArgNum(voidArgs,0);
        setStringArgNum(voidArgs,0);
        setFloatArrayArgNum(voidArgs,0);
        setArgs(preTask,voidArgs);
        safeDeleteArguments(voidArgs);
        //we know deterministically that the ID is always zero, as if there is a boot function
        //we prepend it to the list of functions
        setID(preTask,0);
        setType(preTask,FUNCTION_CALL);
        //timing stuff
        "<>
        (*note that all of these are validated in the calling ArduinoConfigureDriver function, so we don't have to confirm anything*)
        Switch[setupOpts["Scheduling"],
        	_Missing,(*no scheduling option specified, so just default to running it once*)
        	(
        		"
		setIterationCount(preTask,1);
        		"
        	),
        	_Integer|_Real, (*run infinitely every x seconds*)
        	(
        		"
		setIterationCount(preTask,0); //0 times corresponds to no limit on the number of times to run
		setSyncTime(preTask,"<>ToString[CForm[Round[setupOpts["Scheduling"]*1000]]]<>"L);
        		"
        	),
        	{_Integer|_Real,_Integer}, (*run every x seconds for a maximum of y times*)
        	(
        		"
		setIterationCount(preTask,"<>ToString[CForm[Round[Last[setupOpts["Scheduling"]]]]]<>"L);
		setSyncTime(preTask,"<>ToString[CForm[Round[First[setupOpts["Scheduling"]]*1000]]]<>"L);
        		"
        	),
        	{_Integer|_Real}, (*run once in x seconds*)
        	(
        		"
        setIterationCount(preTask,1);
		setInitialWaitTime(preTask,"<>ToString[CForm[Round[First[setupOpts["Scheduling"]]*1000]]]<>"L);
        		"
        	),
        	_,
        	"EPIC FAILURE"
        ]<>
        "
        if(getInitialWaitTime(preTask))
        {
            /*the wait time is more than 0 seconds, so */
            /*add it to the scheduling queue*/
            taskScheduledQueueAdd(scheduledQueue,
                preTask);
            /*now delete the task*/
            safeDeleteTask(nextTask);
        }
        else
        {
            /*the function should be handled */
            /*immediately, so put it in the queue*/
            taskQueueEnqueue(immediateTaskQueue,preTask);
            /*now that the nextTask has been stored,*/
            /*we can safely free that memory*/
            safeDeleteTask(preTask);
            //note we don't run the task now, as we can just let that happen in the main while loop
        }
    }
					"
				];
			)
		]
		
	)
];


addInitialization[init_String]:=Module[{},
	(
		"initializations"->init
	)
]



CleanArgReturns[args_]:=Module[{},
	If[Head[#] === Rule,
		(*THEN*)
		(*it's a return, so take the first part to get just the arguments*)
		#[[1]],
		(*ELSE*)
		(*it's a void, so just take it normally*)
		#
	]&/@args
];


produceFree[args_] := Module[{},
	StringJoin[
		Flatten[
			MapIndexed[
				individualFuncFree[#1, First[#2]-1] &,
				args
			]
		]
	]
]


individualFuncFree[args_, functionNum_]:=Module[
	{
		argNum, 
		frees = {},
		funcNum = functionNum
	},
	(
		For[argNum = 0, argNum < Length[args], argNum++,
			If[args[[argNum + 1]]==={Real}||args[[argNum + 1]]==={Integer}||args[[argNum + 1]]===String,
				(*THEN*)
				(*the argument has to be freed, so generate that code and append it to the list*)
				AppendTo[
					frees,
					individualFree[funcNum, argNum]
				]
			]
		];
		Return[frees];
	)
]


individualFree[funcNum_, argNum_] := Module[{},
	(
		"\tfree(func" <> ToString@funcNum <> "arg" <> ToString@argNum <> ");\n"
	)
]


generateMemType[argTypes_]:=Module[{justArgs},
	(
		justArgs=CleanArgReturns[argTypes];
		produceFree[justArgs]
	)
];



addFunctionSourceCode[funcSource_]:=Module[{},
	(
		(*make the text passed a string template and replace userFunctionSources with what the user passed*)
		"userFunctionSources"->If[funcSource===None,"",funcSource]
	)
]


addLibraries[libraries_]:=Module[{},
	If[!TrueQ[libraries==={}],
	(
		"libraries"->StringJoin[Table["#include \""<>ToString@lib<>"\"\n",{lib,libraries}]]
	),
	"libraries"->""
	]
];
	

(*===================================================================*)
(*===================================================================*)
(*===================FUNCTION CALL GENERATION========================*)
(*===================================================================*)
(*===================================================================*)

(*example input: {{{String},{String}->Integer},{lcdWrite,stringReturnFunc}} *)
generateFunctions[funcTypeInfo_]:=Module[{},
	(
		(*make the text passed a string template and replace callFunctionWithArguments with the result of functionTemplate*)
		{
			"callFunctionWithArguments"->If[funcTypeInfo===None,functionTemplate[{},{}],functionTemplate@@funcTypeInfo],
			"functionID"->functionIDCreate[If[funcTypeInfo===None,0,Length[funcTypeInfo[[2]]]]]
		}
	)
];


(*==============================================================*)
(*===============Valid Function ID in the sketch================*)
(*==============================================================*)
(*==============================================================*)


functionIDCreate[numFunctions_] := Module[{},
	StringTemplate[
		StringJoin[
			"byte validFunctionID(byte idToCheck)\n",
			"{\n\tswitch(idToCheck)\n\t{\n\t\t`cases`",
			"\n\t\t//this switch case will be populated\n",
			"\t\t// with the possible ID's at compile time\n",
			"\t\tdefault:\n\t\t\t//not found, so invalid\n\t\t\treturn 0;\n\t}\n}"
		]
	][<|"cases" -> idCheckCase[numFunctions]|>]
];

(*these are for generating the validFunctionID function in the sketch*)
idCheckCase[totalFunctions_] := Module[{},
	(
	StringJoin[
		Table["\t\tcase "<>ToString@funcNum<>":\n\treturn 1;\n",{funcNum,0,totalFunctions - 1}]]
	)
];



(*TEMPLATING FUNCTIONS FOLLOW*)

(*this is just the straight string template that we use for the callFunctionWithArguments function*)
(*it has the following members that have to be replaced: cases, argInitializations, and memFree*)
callFunctionTemplate = 
  StringTemplate["void callFunctionWithArguments(task * runTask)
   {
   \t/*this function will largely be populated */
   \t/*at compile time with the given functions necessary*/
   \targuments * functionArgs = getArgs(runTask);
   \t/*for tasks that return values*/
   \tlong returnLong;
   \tfloat returnFloat;
   \t/*returnString is initialized to zero so that when we free it*/
   \t/*nothing will happen if it doesn't end up being used*/
   \tchar * returnString = 0;
   \t/*variables and pointers for storing the arguments*/
   \tlong * longArgs = 0;
   \tfloat * floatArgs = 0;
   \tfloat2DArray * floatArrayArgs =0;
   \tlong2DArray * longArrayArgs = 0;
   \tchar2DArray * stringArgs = 0;
   \t/*get structs and arrays out of arguments adt*/
   \tif(getLongArgNum(functionArgs))
   \t{
   \t\tlongArgs = getLongArgArray(functionArgs);
   \t}
   \tif(getFloatArgNum(functionArgs))
   \t{
   \t\tfloatArgs = getFloatArgArray(functionArgs);
   \t}
   \tif(getStringArgNum(functionArgs))
   \t{
   \t\tstringArgs = getStringArgArray(functionArgs);
   \t}
   \tif(getLongArrayArgNum(functionArgs))
   \t{
   \t\tlongArrayArgs = getLongArrayArgArray(functionArgs);
   \t}
   \tif(getFloatArrayArgNum(functionArgs))
   \t{
   \t\tfloatArrayArgs = getFloatArrayArgArray(functionArgs);
   \t}
   \t/*ARGUMENT INITIALIZATIONS*/
   \t`argInitializations`
   \t
   \tswitch(getID(runTask))
   \t{
   \t\t`cases`
   \t}
   `memFree`
   \t/*finally free all the data we just allocated for the arguments*/
   \tsafeDeleteChar(stringArgs);
   \tsafeDeleteLong(longArrayArgs);
   \tsafeDeleteFloat(floatArrayArgs);
   \tsafeDeleteArguments(functionArgs);
   \tfree(longArgs);
   \tfree(floatArgs);
   \t\n}"];
   
   
functionTemplate[args_, names_]:=Module[{},
	(
		If[args=!={}&&names=!={},
			(*THEN*)
			(*check if the func has a return type*)
			(
				Return[fullTemplate[args,names]]
			),
			(*ELSE*)
			(*there are no custom functions, so we can just intitalize it to a dummy function with a default break statement and no argument initializations*)
			Return[callFunctionTemplate[<|"cases"->"default: \n\t\tbreak;\n","argInitializations"->"","memFree"->""|>]];
		]
	)
];


(*this is called by functionTemplate*)
fullTemplate[args_,names_]:=Module[{},
	(
		StringTemplate[firstReplace[Length[args],args]][secondReplaceValues[args, names]]
	)
]



(*this is called by fullTemplate*)
firstReplace[totalFuncs_,args_]:=Module[{},
	(
		callFunctionTemplate[
			<|
				"cases"->caseBlocks[totalFuncs],
				"argInitializations"->If[totalFuncs>0,
					(*THEN*)
					(*create the initializations*)
					argumentInitializes[totalFuncs],
					(*ELSE*)
					(*there are no functions, so just return an empty string*)
					""
				],
				"memFree"->generateMemType[args]
			|>
		]
	)
];


(*this is called by firstReplace*)
(*this will create the case's inside the switch block that switches on the function id*)
caseBlocks[totalFuncs_] :=Module[{},
	(
		StringJoin[
			Riffle[
				Table[
					"\t\tcase "<>ToString@funcNum<>":
					`setArguments"<>ToString@funcNum<>"`
					`verbatimFunctionCall"<>ToString@funcNum<>"`
					break;",
					{funcNum,0,totalFuncs-1}
				],
				"\n"
			]
		]
	)
];


(*this is called by firstReplace*)
(*this creates keys for the first template, to be replaced before the second template pass*)
argumentInitializes[totalFuncs_] :=Module[{},
	(
		If[totalFuncs<=0,
			(*THEN*)
			(*there are no functions, so just pass an empty string back*)
			"",
			(*ELSE*)
			(*there are functions, so create the initialization keys for the template*)
			(
				(*each initialization key is basically just `argInitializationX`, where X is the function number*)
				StringJoin[
					Riffle[
						Table[
							"`argInitialization"<>ToString@funcNum<>"`",
							{funcNum,0,totalFuncs-1}
						],
						"\n"
					]
				]
			)
		]
	)
];


(*this will be called by fullTemplate*)
(*this will create the association for the second replace, making something similar to:*)
(*<|"argInitialization0->"code","setArguments0"->"code","verbatimFunctionCall"->"code"|>*)
secondReplaceValues[allArgumentTypes_, functionNames_]:=Module[
	{
		numberOfFunctions = Length[functionNames]
	},
	(
		Association[
			MapThread[
				Rule,
				{
					secondReplaceKeys[numberOfFunctions],
					Flatten[
						{
							(*first values are the argument intializers*)
							allArgumentInitializer[allArgumentTypes],
							(*next is the assignment of the value to the arguments*)
							If[Flatten[allArgumentTypes]=!={},
								(*THEN*)
								(*the functions have arguments, so assign them to values with argTypeSetter*)
								MapIndexed[argTypeSetter[If[Head[#1]===Rule,#1[[1]],#1], #2[[1]] - 1] &, allArgumentTypes],
								(*ELSE*)
								(*the functions don't have any arguments, so return an empty string*)
								Table["",{numberOfFunctions}]
							],
							(*finally is the actual function call*)
							newAllFunctionCalls[allArgumentTypes, functionNames]
						}
					]
				}
			]
		]
	)
];


(*this is called by secondReplaceValues*)
argTypeSetter[argTypes_, funcNum_] :=Module[
	{
		argSlots = {},
		longArgNum = 0,
		floatArgNum = 0,
		stringArgNum = 0,
		floatArrayArgNum = 0,
		longArrayArgNum = 0
	},
	(
		If[Flatten[argTypes]=!={},
			(*THEN*)
			(*the function has arguments, so assign them values*)
			For[argNum = 0, argNum < Length[argTypes], argNum++,
				AppendTo[argSlots,
					"func" <> ToString@funcNum <> "arg" <> ToString@argNum <> "=" <>
					singleArgType[argTypes[[argNum + 1]],
						Which[
							argTypes[[argNum + 1]] === Integer, longArgNum++,
							argTypes[[argNum + 1]] === Real, floatArgNum++,
							argTypes[[argNum + 1]] === String, stringArgNum++,
							argTypes[[argNum + 1]] === {Real}, floatArrayArgNum++,
							argTypes[[argNum + 1]] === {Integer}, longArrayArgNum++
						]
					]
				]
			],
			(*ELSE*)
			(*the function doesn't have any arguments, so don't inititalize anything, and just return an empty string*)
			(
				Return[""];
			)
		];
		Return[StringJoin[Riffle[argSlots, "\n"]]];
	)
];

(*this is called by argTypeSetter*)
singleArgType[type_, argNum_]:=Module[{},
	(
		Which[
			type === Integer, "longArgs[" <> ToString@argNum <> "];",
			type === Real, "floatArgs[" <> ToString@argNum <> "];",
			type === String, "getGivenArrayChar(stringArgs," <> ToString@argNum <> ");",
			type === {Real}, "getGivenArrayFloat(floatArrayArgs," <> ToString@argNum <> ");",
			type === {Integer}, "getGivenArrayLong(longArrayArgs," <> ToString@argNum <> ");"
		]
	)
];


(*this is called by secondReplaceValues*)
newAllFunctionCalls[allArgTypes_, funcNames_]:=Module[{},
	(
		MapIndexed[
			If[returnQ[#1],
				(*THEN*)
				(*the function has a return type, so use returnFunctionCall*)
				(
					functionReturnCall[
						#1[[2]],
						If[Flatten[#1[[1,1]]]==={},
							(*THEN*)
							(*the function doesn't have any arguments, so pass in 0*)
							0,
							(*ELSE*)
							(*the function has arguments, pass in how many*)
							Length[#1[[1,1]]]
						],
						#2[[1]]-1,
						#1[[1,2]]
					]
				),
				(*ELSE*)
				(*the function doesn't have a return type, so use functionVoidCall*)
				(
					functionVoidCall[
						#1[[2]],
						If[Flatten[#1[[1]]]==={},
							(*THEN*)
							(*the function doesn't have any arguments, so pass in 0*)
							0,
							(*ELSE*)
							(*the function has arguments, pass in how many*)
							Length[#1[[1]]]
						],
						#2[[1]]-1
					]
				)
			]&,
			Transpose[{allArgTypes,funcNames}]]
	)
];


(*this will be called by functionCall*)
(*this will produce code of the form*)
(*
returnLong = someUserFunction(func0arg0,func0arg1,func0arg3, ... );
firmataTaskLongSend(returnLong);
*)
functionReturnCall[funcName_String,totalArgs_,funcNum_,returnType_] := Module[{},
	(
		(*TODO: implement more rudimentary checking for these Which statements*)
		StringJoin[
			Which[
				returnType === Integer, "returnLong",
				returnType === Real,"returnFloat",
				returnType === String,"returnString"
			],
			"=",
			funcName,
			"(",
			(*this creates the arguments inside the function of the form funcXargX, where X is the function number*)
			Riffle[
				Table[
					"func"<>ToString@funcNum<>"arg"<>ToString@argNum,
					{argNum,0,totalArgs-1}
				],
				","
			],
			");\n",
			Which[
				returnType===Integer,"firmataTaskLongSend(returnLong);\n",
				returnType===Real,"firmataTaskFloatSend(returnFloat);\n",
				(*for String return types, add a check that frees the variable if it is necessary*)
				returnType===String,StringJoin[{"firmataTaskStringSend(returnString);\n",
					"if(isPointerDynamic(returnString))",
					"{",
					"\t\tfree(returnString);",
					"}"}]
			]
		]
	)		
];


(*this will be called by functionCall*)
functionVoidCall[funcName_String, totalArgs_, funcNum_]:=Module[{},
	(
		StringJoin[
			funcName,
			"(",
			If[totalArgs === 0,
				(*THEN*)
				(*there are no arguments, so just enter in a empty string where the arguments should go*)
				(
					""
				),
				(*ELSE*)
				(*the function has arguments, so generate those*)
				Riffle[
					Table[
						"func"<>ToString@funcNum<>"arg"<>ToString@argNum,
						{argNum,0,totalArgs-1}
					],
					","
				]
			],
			");"
		]
	)
];


(*this is called by secondReplaceValues*)
(*this will apply the function initializeArgumentFunctions to all the possible argument types with the correct argument structure*)
allArgumentInitializer[allArgTypes_]:=Module[{},
	(
		If[Flatten[allArgTypes]=!={},
			(*THEN*)
			(*there are arguments to create, so initialize them*)
			(
				MapIndexed[
					initializeArgumentFunctions[
						(*first argument is the raw argument types for each function*)
						If[Head[#1]===Rule,#1[[1]],#1],
						(*second argument is the number of raw argument types for each function*)
						Length[If[Head[#1]===Rule,#1[[1]],#1]],
						(*decrement the index by 1 to use zero indexing*)
						#2[[1]]-1
					]&,
					allArgTypes
				]
			),
			(*ELSE*)
			(*the function doesnt' have any arguments, so don't initialize anything and just return an empty string*)
			(
				Return[Table["",{Length[allArgTypes]}]];
			)
		]
	)
];

(*this is called by allArgumentInitializer*)
(*this function will return a string with the argument initializations, so something like*)
(*
long func0arg0;
float func0arg1;
...
*)
(*it needs to know the argument types, the total number of arguments, and which function number this is*)
(*funcNum could be *)
initializeArgumentFunctions[argTypes_, totalArgs_, funcNum_]:=Module[{},
	(
		(*top part makes the template*)
		StringTemplate[
			If[Flatten[argTypes]=!={},
				(*THEN*)
				(*the function has arguments, so generate those*)
				StringJoin[
					Riffle[
						Table[
							"`arg" <> ToString@argNum <> "Type` func" <> ToString@funcNum <>"arg" <> ToString@argNum <> "=0;",
							{argNum, 0, totalArgs - 1}
						],
						"\n"
					]
				],
				(*ELSE*)
				(*the function doesn't have arguments, leave this blank*)
				(
					""
				)
			]
		(*bottom part creates the values to put in the template*)
		][
			Association[
				Table[
					Rule[
						"arg" <> ToString@(type - 1) <> "Type",
						(*replaces the high level Wolfram types, Integer, Real, String, etc. with char *, float, long, etc. for C/C++*)
						argTypes[[type]] /. {Integer -> "long", Real -> "float", String -> "char *", {Integer} -> "long *", {Real} -> "float *"}
					],
					{type, Length[argTypes]}
				]
			]
		]
	)
];


(*this is called by secondReplaceValues*)
(*this function basically creates the keys that will need to be replaced in the next template*)
secondReplaceKeys[totalFuncs_]:=Module[{},
	(
		Sort[
			Flatten[
				Table[
				{
					"argInitialization" <> ToString@funcNum,
					"setArguments" <> ToString@funcNum,
					"verbatimFunctionCall"<>ToString@funcNum
				},
				(*start from 0 instead of 1*)
				{funcNum,0,totalFuncs - 1}
				]
			]
		]
	)
];



(*helper function for the function generating functions*)
returnQ[function_]:=Module[{},
	(
		Return[Head[function[[1]]]===Rule]
	)
]




End[] (* End Private Context *)

EndPackage[]

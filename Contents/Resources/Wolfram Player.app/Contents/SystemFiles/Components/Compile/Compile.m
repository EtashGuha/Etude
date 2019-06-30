BeginPackage["Compile`"]

$VerboseDebugging
$Initialize

CompileExpr
CompileExprRecurse

$DefaultTypeEnvironment
$DefaultMacroEnvironment

SelectTypeEnvironment
CreateCoreTypeEnvironment
CreateCoreMacroEnvironment

LanguageException

CompileToCodeFunction
CompileToCodeFunctionList
CompileToExternalFunction
CompileToLLVMString
CompileToLLVMIR
CompileToWIRSerialization
CompileToLLVMModule
CompileValues
CompileSymbols
CompileToLibrary
CompileToAST
CompileToIR

InitializeCompiler
ShutdownCompiler

Program
DeclareType
DeclareFunction
ConstantValue
New
Field
Unknown
SymbolQ
LLVMString

$CompilerSandbox

Begin["`Private`"]

Compile`Utilities`$CompileRootDirectory = FileNameDrop[$InputFileName, -1]

$CompilerSandbox = False


Needs["Compile`Utilities`"]
Needs["Compile`AST`"]
Needs["Compile`API`"]
Needs["Compile`Core`"]
Needs["Compile`TypeSystem`"]
Needs["Compile`Driver`"]
Needs["Compile`CompileToLibrary`"]
Needs["Compile`Values`CompileValues`"]
Needs["Compile`ResourceLoader`"]
Needs["Compile`TypeSystem`Environment`TypeEnvironment`"]

Needs["CompileUtilities`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for CatchException *)

Needs["TypeFramework`"]

Needs["CompileAST`"]
Needs["LLVMLink`"] (* For InstallLLVM[] *)
Needs["CompileUtilities`Platform`Platform`"]


LanguageException::usage = "An exception that represents an error in the code being compiled."

$DefaultTypeEnvironment = Function[{method}, ThrowException["$DefaultTypeEnvironment is uninitialized. Please evaluate InitializeCompiler[] first."]]
$DefaultMacroEnvironment = Function[{method}, ThrowException["$DefaultMacroEnvironment is uninitialized. Please evaluate InitializeCompiler[] first."]]


$VerboseDebugging = False
$initialized = False

Compiler`Internal`$Version = 1.0

CreateCoreMacroEnvironment[] :=
	Module[ {env, state},
			CatchException[
				env = CreateMacroEnvironment[];
				state = <|"macroEnvironment" -> env|>;
				RunCallback[ "SetupMacros", state];
				env
				,
				{{_, CreateFailure}}
			]
	]

(*
  Select the type environment to use. If newQ is set we just create a new one.
  Otherwise use the systemID to choose the machine integer size and then if things 
  match use the default else create a new one.
*)

SelectTypeEnvironment[ systemID_, newQ_] :=
	Module[ {size = getMachineIntegerSize[systemID], test},
		Which[
			TrueQ[newQ],
				CreateCoreTypeEnvironment["TargetSystemID" -> systemID]
			,
			$DefaultTypeEnvironment["getProperty", "MachineIntegerSize"] === size,
		test = $DefaultTypeEnvironment["getProperty", "MachineIntegerSize"];
			$DefaultTypeEnvironment,
		True,
			CreateCoreTypeEnvironment["TargetSystemID" -> systemID, "LoadResources" -> False]
		]
	]
	
	
Options[CreateCoreTypeEnvironment] = 
	{"LoadResources" -> True, "TargetSystemID" -> Automatic, "IntegerIntrinsics" -> False}

CreateCoreTypeEnvironment[opts:OptionsPattern[]] :=
	(
	InitializeCompiler[];
	CreateCoreTypeEnvironmentInternal[OptionValue["LoadResources"], OptionValue["TargetSystemID"], OptionValue["IntegerIntrinsics"]]
	)


getMachineIntegerSize[ Automatic] :=
	$SystemWordLength


	
getMachineIntegerSize[id_] :=
	MachineIntegerSizeFromSystemID[id]

getTargetSystemID[ Automatic] :=
	$SystemID
	
getTargetSystemID[ targetSysID_] :=
	targetSysID

CreateCoreTypeEnvironmentInternal[load_, targetSysIDIn_, integerIntrinsics_] :=
	CatchException[
		Module[ {env, state, 
				targetSysID = getTargetSystemID[targetSysIDIn],
				mIntSize = getMachineIntegerSize[targetSysIDIn]},
			env = CreateCompileTypeEnvironment[];
			env["setProperty", "MachineIntegerSize" -> mIntSize];
			env["setProperty", "TargetSystemID" -> targetSysID];
			state = <|"typeEnvironment" -> env, "integerIntrinsics" -> integerIntrinsics|>;
			RunCallback[ "SetupTypeSystem", state];
			RunCallback[ "FinalizeTypeSystem", state];
			If[ TrueQ[load],
				loadCompilerState[env]];
			env
		]
		,
		{{_, CreateFailure}}
	]



Options[InitializeCompiler] = {"LoadResources" -> True, "InstallLLVM" -> False, "TargetSystemID" -> Automatic, "IntegerIntrinsics" -> False}
InitializeCompiler[opts:OptionsPattern[]] :=
	Module[ {envTy, envMacro},
		If[
			$initialized,
			Null
			,
			InitializeCompileUtilitiesClasses[];
			InitializeTypeFrameworkClasses[];
			InitializeCompileASTClasses[];

			RunCallback["InitializeCompileFormat", {}];

			SortCallbacks["DeclareCompileClass", SortClassesFunction];
			RunCallback["DeclareCompileClass", {}];

			SortCallbacks["RegisterPass", sortRegisterPasses];
			RunCallback["RegisterPass", {}];

			RunCallback["RegisterPrimitive", {}];

			(* depends on DeclareClass *)
			RunCallback["RegisterInstruction", {}];
			
			RunCallback["RegisterTestDataGenerator", {}];

			(* depends on RegisterInstruction *)
			RunCallback["InstructionDispatchTrait", {}];

			(* depends on RegisterInstruction and InstructionDispatchTrait *)
			RunCallback["DeclareCompileClassPostRegisterInstruction", {}];

			RunCallback["FinalizeExpressionInitialize", {}];

			envTy = CreateCoreTypeEnvironmentInternal[OptionValue["LoadResources"], OptionValue["TargetSystemID"], OptionValue["IntegerIntrinsics"]];

			envMacro = CreateCoreMacroEnvironment[];

			FinalizeCallbacks[];

			Which[
				!TypeEnvironmentQ[envTy],
					ThrowException[{"An error was encountered initializing the core type environment.", envTy}]
					,
				!MacroEnvironmentQ[envMacro],
					ThrowException[{"An error was encountered initializing the core macro environment.", envMacro}]
					,
				True,
					$DefaultTypeEnvironment = envTy;
					$DefaultMacroEnvironment = envMacro;
					$initialized = True
			]
		];

		Switch[OptionValue["InstallLLVM"],
			True,
				InstallLLVM[],
			False,
				Null,
			_,
				ThrowException[{"Bad value for option \"InstallLLVM\"", OptionValue["InstallLLVM"]}]
		];
	]


(*
sort the passes depending on dependency information from "requires" and "preserves"
*)
sortRegisterPasses[funs0_] :=
	Module[{names, requires, preserves, g, sort, headPattern, newFuns, edges, vertices, funs},
		funs = Select[funs0, # =!= Null&];
		headPattern = CreateFunctionModulePass | CreateProgramModulePass | CreateBasicBlockPass | CreateMExprPass | CreateCompiledProgramPass;
		names = Association[Flatten[Cases[#, HoldPattern[Set][name_, headPattern[_]] :> (name -> #), Infinity] & /@ funs]];
		requires = Flatten[Cases[#, headPattern[Association[___, "requires" -> r_, ___]] :> r, Infinity]]& /@ names;
		preserves = Flatten[Cases[#, headPattern[Association[___, "preserves" -> p_, ___]] :> p, Infinity]]& /@ names;
		edges = Flatten[Map[Thread, Normal[requires] ~Union~ Normal[preserves]]];
		vertices = Keys[names];
		g = Graph[vertices, edges];
		sort = TopologicalSort[g];
		(* The original order was Pass1 -> Pass2, with Pass1 requires|preserves Pass2, but we want the reverse *)
		sort = Reverse[sort];
		newFuns = Lookup[names, sort];
		(*Print[{Length[newFuns], Length[funs], Complement[newFuns, funs], Complement[funs, newFuns]}];*)
		If[Length[newFuns] =!= Length[funs],
			Module[{
				 newNames = Sort[SymbolName /@ sort],
				 oldNames = Sort[SymbolName /@ Keys[names]]
			},
			     (* there is a bad context which will be hidden by the use of symbol name *)
			     If[Union[newNames] === Union[oldNames],
			     	  newNames = Sort[sort];
                      oldNames = Sort[Keys[names]];
			     ];
			With[{
				 maxLen = Max[Length[newFuns], Length[funs]]
			},
			With[{
				 diff = If[Length[newNames] >= Length[oldNames],
				 	 Complement[newNames, oldNames],
                     Complement[oldNames, newNames]
				 ]
			},
			     ThrowException[{"Cannot sort RegisterPasses. A registered pass in the following table is unknown ",
			     	"Lengths" -> <|
			     	   "new" -> Length[newFuns],
                       "old" -> Length[funs]
			     	|>,
			     	If[diff === {},
			     		Nothing,
			     		{"maybe ", diff}
			     	],
			     	Grid[
			     		Prepend[
			     			MapThread[
			     				Function[{n, new, old},
			     					If[new =!= old,
			     						{n, Style[new, Background -> Red], Style[old, Background -> Red]},
			     						{n, new, old}
			     					]
			     				],
			     				{Range[maxLen], PadRight[newNames, maxLen, "-"], PadRight[oldNames, maxLen, "-"]}
			     			],
			     			{"#", "New Passes", "Old Passes"}
			     		]
			     		,
			     		Background -> {None, {Lighter[Yellow, .9], {White, White, White}}},
			     		Dividers -> Darker[Gray, .6],
			     		Frame -> Darker[Gray, .6]
			     	]
			     }];
			]]]
		];
		newFuns
	]



ShutdownCompiler[] :=
	Module[{},

		RunCallback["ShutdownLLVMCompileTools", {}];

		RunCallback["ShutdownLLVMToObjectFile", {}];

		(*
		Do not call until state of LLVM is better understood
		ShutdownLLVM[]
		*)
	]

RegisterCallback["LLVMMemoryLimitExceeded", Function[{st},
ShutdownCompiler[]
]]






Compile::nocache = "TypeEnvironmentCache \"`1`\" could not be loaded."


loadCompilerState[env_?TypeEnvironmentQ] :=
	Module[ {data, cacheName},
		cacheName = FileNameJoin[{Compile`Utilities`$CompileRootDirectory, "CompileResources", $SystemID, "TypeEnvironmentCache.mx"}];
		Get[cacheName];
		data = Compile`$TypeEnvironmentCache;
		If[!AssociationQ[data],
			Message[Compile::nocache, cacheName]
			,
			SetTypeEnvironmentCache[env, data]];
	]

loadCompilerState[args___] :=
	ThrowException[{"Unrecognized call to loadCompilerState", {args}}]


Attributes[SymbolQ] = {}
SymbolQ = Developer`SymbolQ
Attributes[SymbolQ] = Attributes[Developer`SymbolQ]


End[]


EndPackage[]


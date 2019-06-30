
BeginPackage["Compile`Core`PassManager`PassLogger`"]


CreateDefaultPassLogger
CreateRecordingPassLogger
RecordingPassLoggerQ

CreatePrintingPassLogger
CreateFunctionRecurseProfiler

Begin["`Private`"] 

Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Debug`Logger`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`"]



(*

*)

RegisterCallback["DeclareCompileClass", Function[{st},
DefaultPassLoggerClass = DeclareClass[
	DefaultPassLogger,
	<|
		"prePass" -> (prePassDefault[Self, ##]&),
		"postPass" -> (postPassDefault[Self, ##]&),
		"prePassEnd" -> (Null&),
		"postPassEnd" -> (Null&),
		"startRunPass" -> (startRunPassDefault[Self, ##]&),
		"endRunPass" -> (endRunPassDefault[Self, ##]&),
		"startProcess" -> (Null&),
		"endProcess" -> (Null&)
	|>,
	{
		"logger"
	}
]
]]

CreateDefaultPassLogger[level_:"WARN"] :=
	Module[ {obj},
		InitializeCompiler[];
		obj = CreateObject[
			DefaultPassLogger,
			<|
				"logger" -> CreateLogger["PassRunner", level]
			|>
		];
		obj
	] 
	

prePassDefault[ self_, pass_, args_List] :=
	self["logger"][ "trace", "Running required passes = " <> ToString[#["name"]& /@ args]]
	
postPassDefault[ self_, pass_, args_List] :=
	self["logger"][ "trace", "Running post passes = " <> ToString[#["name"]& /@ args]]


startRunPassDefault[ self_, pass_, obj_] :=
	AbsoluteTime[]

endRunPassDefault[ self_, pass_, t1_, obj_] :=
	Module[ {level, t2 = AbsoluteTime[]},
		level = If[t2-t1 < 0.5,
				"trace",
				"info"];
		self["logger"][level, "Running " <> pass["name"] <> " took " <> ToString[t2 - t1]]
	]


(*
  Recording Pass Logger
*)

RegisterCallback["DeclareCompileClass", Function[{st},
RecordingPassLoggerClass = DeclareClass[
	RecordingPassLogger,
	<|
		"prePass" -> (depPassStart[Self, "pre", ##]&),
		"postPass" -> (depPassStart[Self, "post", ##]&),
		"prePassEnd" -> (depPassEnd[Self, ##]&),
		"postPassEnd" -> (depPassEnd[Self, ##]&),
		"startRunPass" -> (startRunPass[Self, ##]&),
		"endRunPass" -> (endRunPass[Self, ##]&),
		"getData" -> (getData[Self, ##]&),
		"startProcess" -> (startProcess[Self]&),
		"endProcess" -> (endProcess[Self]&)
	|>,
	{
		"dependencyPath",
		"data",
		"startTime",
		"endTime"
	},
	Predicate -> RecordingPassLoggerQ
]
]]


startProcess[self_] :=
	self["setStartTime", AbsoluteTime[]]

endProcess[self_] :=
	self["setEndTime", AbsoluteTime[]]


depPassStart[self_, type_, pass_, deps_] :=
	self["dependencyPath"]["pushBack", <|"type" -> type, "passName" -> pass["name"]|>]
	
depPassEnd[self_, arg_] :=
	self["dependencyPath"]["popBack"]

startRunPass[ self_, pass_, obj_] :=
	AbsoluteTime[]

endRunPass[ self_, pass_, timeStart_, obj_] :=
	Module[ {time, tNow = AbsoluteTime[], elem},
		time = tNow - timeStart;
		elem = <|"type" -> "endRun", "passName" -> pass["name"], "runtime" -> time, "time" -> tNow, "path" -> self["dependencyPath"]["get"] |>;
		self["data"]["appendTo", elem]
	]



CreateRecordingPassLogger[] :=
	Module[ {obj},
		InitializeCompiler[];
		obj = CreateObject[
			RecordingPassLogger,
			<|
				"data" -> CreateReference[{}],
				"dependencyPath" -> CreateReference[{}]
			|>
		];
		obj
	] 
	

getData[self_] :=
	self["data"]["get"]



(*
  Printing Pass Logger
*)

RegisterCallback["DeclareCompileClass", Function[{st},
PrintingPassLoggerClass = DeclareClass[
	PrintingPassLogger,
	<|
		"prePass" -> (null[Self, "pre", ##]&),
		"postPass" -> (null[Self, "post", ##]&),
		"prePassEnd" -> (null[Self, ##]&),
		"postPassEnd" -> (null[Self, ##]&),
		"startRunPass" -> Function[{pass, obj}, processPrintPass[Self, "StartRunPass", pass, obj]],
		"endRunPass" -> Function[{pass, time, obj}, processPrintPass[Self, "EndRunPass", pass, obj]],
		"startProcess" -> Function[{obj}, processPrintPass[Self, "StartProcess", Null, obj]],
		"endProcess" -> Function[{obj}, processPrintPass[Self, "EndProcess", Null, obj]]
	|>,
	{
		"printMode",
		"function",
		"shouldRecurse",
		"data"
	},
	Predicate -> PrintingPassLoggerQ
]]]

null[___] :=
	Null

CreatePrintingPassLogger[ fun_, recurseQ_:False] :=
	Module[ {obj},
		InitializeCompiler[];
		obj = CreateObject[
			PrintingPassLogger,
			<|
				"function" -> fun,
				"shouldRecurse" -> recurseQ
			|>
		];
		obj
	] 


processPrintPass[self_, selector_, pass_, obj_] :=
	self["function"][selector, pass, obj]


RegisterCallback["DeclareCompileClass", Function[{st},
FunctionRecurseProfilerClass = DeclareClass[
	FunctionRecurseProfiler,
	<|
		"start" -> Function[{obj}, start[Self, obj]],
		"end" -> Function[{obj}, end[Self, obj]]
	|>,
	{
		"startTime",
		"childTime",
		"result"
	}
]]]

CreateFunctionRecurseProfiler[] :=
	Module[ {logger, profiler},
		InitializeCompiler[];
		profiler = CreateObject[FunctionRecurseProfiler, 
						<|"result" -> CreateReference[<||>],
						  "startTime" -> CreateReference[{0}],
						  "childTime" -> CreateReference[{0}]
						|>];
		logger = CreatePrintingPassLogger[ recurseProfiler[profiler], True];
		logger["setData", profiler];
		logger
	]

	
recurseProfiler[ profiler_]["StartProcess", _, obj_] :=
	profiler["start", obj]

recurseProfiler[ profiler_]["EndProcess", _, obj_] :=
	profiler["end", obj]

start[self_, obj_] :=
	Module[ {},
		self["childTime"]["appendTo",0];
		self["startTime"]["appendTo",AbsoluteTime[]];
	]
	
end[self_, obj_] :=
	Module[ {endTime = AbsoluteTime[], name, childTime, startTime, timeDiff, len},
		childTime = self["childTime"]["popBack"];
		startTime = self["startTime"]["popBack"];
		len = self["childTime"]["length"];
		name = getName[len, obj];
		timeDiff = endTime - startTime;
		self["childTime"]["setPart", len, self["childTime"]["last"] + timeDiff];
		timeDiff = timeDiff - childTime;
		self["result"]["associateTo", name -> {timeDiff, obj}]
	]

getName[1, _] :=
	Top
	
getName[_, TypeFramework`MetaData[args_][___]] :=
 	Lookup[args, "Name", ""]

getName[_, _] := None
	


End[] 

EndPackage[] 

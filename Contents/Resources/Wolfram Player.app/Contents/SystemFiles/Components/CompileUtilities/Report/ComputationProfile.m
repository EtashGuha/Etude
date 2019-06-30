
BeginPackage["CompileUtilities`Report`ComputationProfile`"]

ProfileReportOperation

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)

SetAttributes[ProfileReportOperation, HoldAll]

Options[ ProfileReportOperation] = 
	{"Handler" -> CreateReport, "ReportType" -> "TotalSelf", "LimitLength" -> 50, "SortKey" -> "self"}

ProfileReportOperation[ expr_, desc_, opts:OptionsPattern[]] :=
	Module[{handler = OptionValue["Handler"], res, trace, reportType = OptionValue["ReportType"]},
		{res, trace} = RuntimeTools`ProfileDataEvaluate[expr];
		trace = processData[trace];
		handler[trace, Hold[expr], desc, reportType, opts];
		res
	]

processData[ rawData_] :=
	Module[ {data = rawData, secBase, usecBase, totalTime, processor, processedData},
		initialize[];
		secBase = Part[data,1,3];
		usecBase = Part[data,1,4];
		data = Apply[ {#1, #2, #3-secBase + (#4-usecBase)/1000000.}&, data, {1}];
   		totalTime = Part[data, -1,3]-Part[data, 1,3];
   		processor = createProfileProcessor[];
   		Map[ processor["add", #]&, data];
   		processedData = processor["extractOutput"];
   		processedData = AssociationMap[processFunction[totalTime, #]&, processedData];
		<|"totalTime" -> totalTime, "data" -> processedData|>
	]


processFunction[ totalTime_, {fun_, file_, line_} -> data_List] :=
	Module[{calls = Length[data], depthZero, total, totalSelf},
		totalSelf = Total[ Part[data,All,"selfTime"]];
		depthZero = Select[data, #["depth"] === 0&];
		total = Total[ Part[depthZero,All,"methodTime"]];
		{fun, file, line} -> <|
		  "calls" -> calls,
		  "self" -> totalSelf,
		  "total" -> total,
		  "selfPerCall" -> totalSelf/calls,
		  "totalPerCall" -> total/calls,
		  "selfPercent" -> 100 totalSelf/totalTime,
		  "totalPercent" -> 100 total/totalTime,
		  "function" -> fun, "file" -> file, "line" -> line
		|>
	]

add[ self_, {key_, True, time_}] :=
	Module[ {stack, current, aggregateTime},
		aggregateTime = self["aggregateTime"];
		stack = self["stack"];
		current = stack["lookup", key, Null];
		If[ current === Null,
			current = CreateReference[{}];
			stack["associateTo", key -> current]];
		current[ "pushBack", {time, current["length"], aggregateTime}];
		self["setAggregateTime", 0];
	]


add[ self_, {key_, False, time_}] :=
	Module[ {stack, current, startTime, methodTime, selfTime, childTime, depth, oldAggregateTime, outputList},
		stack = self["stack"];
		current = stack["lookup", key, Null];
		If[ current === Null || current["length"] < 1,
			ThrowException[{"Cannot find start for", {key, current}}]
		];
		{startTime, depth, oldAggregateTime} = current["popBack"];
		methodTime = time-startTime;
		outputList = self["output"]["lookup", key, Null];
		If[ outputList === Null,
			outputList = CreateReference[{}];
			self["output"]["associateTo", key -> outputList]];
		childTime = self["aggregateTime"];
		selfTime = methodTime - childTime;
		outputList["appendTo", <|"methodTime" -> methodTime, "depth" -> depth, 
								"selfTime" -> selfTime, "childTime" -> childTime|>];
		self["setAggregateTime", oldAggregateTime+methodTime]
	]

extractOutput[self_] :=
	Map[ #["get"]&, self["output"]["get"]]

If[!ValueQ[$needsInitialization],
	$needsInitialization = True]

initialize[] :=
If[$needsInitialization,
	DeclareClass[
		ProfileProcessor, 
		<|
		"add" -> Function[{data}, add[Self,data]],
		"extractOutput" -> Function[{}, extractOutput[Self]]
	   	|>, 
	   {
	   	"stack",
	   	"output",
	   	"aggregateTime"
	   }
	];
	$needsInitialization = False;
]

 createProfileProcessor[] :=
 	CreateObject[ProfileProcessor, 
 		<|"aggregateTime" -> 0, "stack" -> CreateReference[<||>], "output" -> CreateReference[<||>]|>
 	]


ReportData[ rawData_, Hold[input_], desc_, "TotalSelf", opts:OptionsPattern[]] :=
	Module[{data,  totalTime, tempData,
			lim = OptionValue[ProfileReportOperation,{opts}, "LimitLength"], 
			sortKey = OptionValue[ProfileReportOperation,{opts}, "SortKey"],keys},
		totalTime = rawData["totalTime"];
		data = Values[rawData["data"]];
		data = SortBy[data, Lookup[#, sortKey]&];
		data = Reverse[data];
		data = Take[data, UpTo[lim]];
		keys = {"function", "file", "line", "calls", "self", "total", "selfPerCall", "totalPerCall", "selfPercent", "totalPercent"};
		data = Map[ Lookup[#, keys]&, data];
		data = Prepend[data, keys];
		tempData = <|
			"totalTime" -> totalTime,
			"sortKey" -> sortKey,
			"computation" -> RawBoxes[ MakeBoxes[input]],
			"description" -> desc,
			"data" -> Grid[data, Frame->All]
		|>;
		tempData
	]

CreateReport[ data_, input_, desc_, type_, opts:OptionsPattern[]] :=
	Module[{reportData = ReportData[data, input, desc, type, opts], nbTemp, nb},
		nbTemp = getTemplate[type];
		nb = GenerateDocument[nbTemp, reportData];
		SetOptions[ nb, WindowMargins->Automatic];
		nb
	]



$baseDirectory = DirectoryName[ $InputFileName]

getTemplate["TotalSelf"] :=
	FileNameJoin[ {$baseDirectory, "Templates", "TotalSelfReport.nb"}]


End[]
	
EndPackage[]
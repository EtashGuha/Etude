
BeginPackage["CompileUtilities`Report`ClassProfile`"]

ProfileClassOperation
ProfileClassReport
ProfileClassReportOperation

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Callback`"]

SetAttributes[ProfileClassReportOperation, HoldAll]

Options[ ProfileClassReportOperation] = {"IncludeRawData" -> False}

ProfileClassReportOperation[ expr_, desc_, opts:OptionsPattern[]] :=
	ProfileClassOperation[expr, ProfileClassReport[#, HoldForm[expr], desc,OptionValue["IncludeRawData"]]&]


SetAttributes[ProfileClassOperation, HoldAll]

ProfileClassOperation[ expr_, callback_] :=
	Module[ {ef},
		initialize[];
		Compile`Utilities`Class`Impl`SetProfile[True];
		ef = expr;
		callback[getProfileData[]];
		Compile`Utilities`Class`Impl`SetProfile[False];
		ef
	]



$baseDirectory = DirectoryName[ $InputFileName]

ProfileClassReport[data_, expr_: "", desc_:"", includeRawData_:False] :=
	Module[ {reportData, nbTemp, nb},
		reportData = Join[ data, 
					<| "description" -> desc, 
						"computation" -> expr |>];
		If[ !TrueQ[includeRawData], reportData["rawData"] = Null];
		nbTemp = FileNameJoin[ {$baseDirectory, "Templates", "ClassProfileReport.nb"}];
		nb = GenerateDocument[nbTemp, reportData];
		SetOptions[ nb, WindowMargins->Automatic];
		nb
	]




$elemType = <|0 -> "General", 1 -> "SetData", 2 -> "GetData", 3 -> "ClassMethod"|>

getProfileData[] :=
	Module[{data, rawData, secBase,usecBase, processor, processedData, reportData, totalTime},
		data = Compile`Utilities`Class`Impl`GetProfileData[];
		data = Reverse[data];
		secBase = Part[data,1,5];
		usecBase = Part[data,1,6];
		data = Map[ ReplacePart[#, {4 -> $elemType[Part[#,4]], 5 -> Part[#,5]-secBase, 6 -> Part[#,6]-usecBase}]&, data];
		data = Apply[ {#1, #2, #3,  #4, #5 + (#6/1000000.)} &, data, {1}];
		rawData = Apply[<| "className" -> #1, "methodName" -> #2, "start" -> #3, 
   					"elementType" -> #4, 
   					"timestamp" -> #5|> &, data, {1}];
   		totalTime = Part[rawData, -1,5]-Part[rawData, 1,5];
   		processor = createProfileProcessor[];
   		Map[ processor["add", #]&, data];
   		processedData = processor["extractOutput"];
   		reportData = getReportData[ processedData];
   		Join[ <|"rawData" -> data, "processedData" -> processedData, "totalTime" -> totalTime|>, reportData]
	]



(*
 data has the form <| {"className", "methodName"} -> { <| "methodTime" -> t1, "childTime" -> t2|>, ...}, ...|>
 
 return data of the form
 <|
 "classData" ->
 {
    <|  "className" -> name, "data" -> 
       { <|"methodName",  "callNumber", "fullTotal", "fullTotalPerCall", "soleTotal", 
            "soleTotalPerCall",  "fullCalls", "soleCalls"|>...}|>, 
    ...
 }
 ,
 "totalBarChart" -> bar,
 "soleBarChart" -> bar
 |>
 
*)


getBarChartPlots[ data_, fun_] :=
	Module[ {sort, plotData, lim = 50},
		sort = SortBy[ data, fun];
		sort = If[ Length[sort] < lim, sort, Take[ sort, {-lim, -1}]];
		plotData = BarChart[Map[ Labeled[fun[#], Rotate[ Text[ #["methodName"] <> "   " <> #["className"]], Pi/2]]&, sort],  ImageSize -> 900];
		plotData
	]

splitFun[ className_ -> data_] :=
	className -> <|"className" -> className, "methodData" -> SortBy[data, #["methodName"]&]|>

getReportData[ dataIn_] :=
	Module[ {data, plots},
		data = Map[ processReportElement, Normal[dataIn]];
		plots = <| "totalBarChart" -> getBarChartPlots[ data, #["fullTotal"]&], "soleBarChart" -> getBarChartPlots[ data, #["soleTotal"]&] |>;
		data = GroupBy[data, #["className"]&];
		data = AssociationMap[ splitFun, data];
		data = Values[data];
		Append[ plots,   "classData" -> SortBy[data, #["className"]&] ]
	]

processReportElement[ {className_, methodName_} -> dataList_] :=
	Module[ {len, fullCalls,  childCalls, soleCalls, reportData, hist1, hist2},
		len = Length[dataList];
		fullCalls = Part[dataList, All, "methodTime"];
		childCalls = Part[dataList, All, "childTime"];
		soleCalls = fullCalls - childCalls;
		reportData =
			<|
			"className" -> className,
			"methodName" -> methodName,
			"callNumber" -> len, 
			"fullTotal" -> Total[fullCalls], "fullTotalPerCall" -> Total[fullCalls]/len, 
			"soleTotal" -> Total[soleCalls], "soleTotalPerCall" -> Total[soleCalls]/len, 
			"fullCalls" -> fullCalls,  "soleCalls" -> soleCalls
			|>;
			{hist1, hist2} =
				If[ reportData["soleTotal"] < 2.*^-4,
				{
				None, None
				},
				{
				Histogram[reportData["fullCalls"], 20],
				Histogram[reportData["soleCalls"], 20]
				}];
			Append[reportData,
				{
			   "grid" ->
			   	Grid[{
			   		{"Calls", reportData["callNumber"]},
			   		{"Total", reportData["fullTotal"]},
			   		{"Total/Call", reportData["fullTotalPerCall"]},
			   		{"Total (sole)", reportData["soleTotal"]},
			   		{"Total/Call (sole)", reportData["soleTotalPerCall"]}
			   	}, Frame -> All, Alignment -> {{Left, Decimal}, {Bottom}}],
			   	"fullCallsPlot" -> hist1,
			   	"soleCallsPlot" -> hist2
			   	}
			   	]
	]




add[ self_, {className_, methodName_, True, elemType_, time_}] :=
	Module[ {stack, current, aggregateTime},
		aggregateTime = self["aggregateTime"];
		stack = self["stack"];
		current = stack["lookup", {className, methodName}, Null];
		If[ current === Null,
			current = CreateReference[{}];
			stack["associateTo", {className, methodName} -> current]];
		current[ "pushBack", {time, aggregateTime}];
		self["setAggregateTime", 0];
	]


	
	
add[ self_, {className_, methodName_, False, elemType_, time_}] :=
	Module[ {stack, current, startTime, methodTime, oldAggregateTime, outputList},
		stack = self["stack"];
		current = stack["lookup", {className, methodName}, Null];
		If[ current === Null || current["length"] < 1,
			ThrowException[{"Cannot find start for", {className, methodName, current}}]
		];
		{startTime, oldAggregateTime} = current["popBack"];
		methodTime = time-startTime;
		outputList = self["output"]["lookup", {className, methodName}, Null];
		If[ outputList === Null,
			outputList = CreateReference[{}];
			self["output"]["associateTo", {className, methodName} -> outputList]];
		outputList["appendTo", <|"methodTime" -> methodTime, "childTime" -> self["aggregateTime"]|>];
		self["setAggregateTime", oldAggregateTime+methodTime]
	]
	
	
	

extractOutput[self_] :=
	Map[ #["get"]&, self["output"]["get"]]


$needsInitialization = True

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

RegisterCallback["DeclareCompileUtilitiesClassProfile", Function[{st},
	initialize[];
]]

   
 createProfileProcessor[] :=
 	CreateObject[ProfileProcessor, <|"aggregateTime" -> 0, "stack" -> CreateReference[<||>], "output" -> CreateReference[<||>]|>]
 	
 	
 	
   
   
   
   



End[]

EndPackage[]



BeginPackage["Compile`Utilities`Report`PassReport`"]

PassReport 
PassReportData

Begin["`Private`"]

Needs["Compile`Core`PassManager`PassLogger`"]
Needs["Compile`Core`PassManager`PassRegistry`"]


(*
 Drop the CreateIR pass since this accumulates other passes.  
 Maybe there could be some setting that caused a pass to be dropped.
*)
PassReportData[logger_, desc_] :=
	Module[ {ds, data, passFlatProfile, passAggregateData, passAggregatePlot, passData, total},
		total = logger["endTime"]-logger["startTime"];
		ds = Dataset[logger["getData"]];
		ds = ds[Select[#passName =!= "CreateIR" &]];
		passFlatProfile = getPassTimeData[ds];
		passAggregateData = makeGrid[ collectPassByTime[ds]];
		passAggregatePlot = plotPassByTime[ds];
		passData = getPassData[ds];
		data = <|"rawData" -> ds, "totalTime" -> total, "passRuns" -> ds[Length], 
					"passNumber" -> ds[GroupBy["passName"]][Length],
					"passFlatProfile" -> passFlatProfile,
					"passAggregateTimeData" -> passAggregateData,
					"passAggregateTimePlot" -> passAggregatePlot,
					"passData" -> passData,
					"description" -> desc|>
		
	]


$baseDirectory = DirectoryName[ $InputFileName]

PassReport[logger_?RecordingPassLoggerQ, desc_:""] :=
	Module[ {data, nbTemp, nb},
		data = PassReportData[logger, desc];
		nbTemp = FileNameJoin[ {$baseDirectory, "Templates", "PassReport.nb"}];
		nb = GenerateDocument[nbTemp, data];
		SetOptions[ nb, WindowMargins->Automatic];
		nb
	]


makeGrid[ ds_] :=
	Module[ {ef},
		ef = Normal[ ds];
		ef = Map[ Values, ef];
		Grid[ef, Frame -> All]
	]

getPassData[ds_] :=
	Module[ {used, rest},
		used = DeleteDuplicates[Normal[ds[All, "passName"]]];
		rest = Complement[Values[ $Passes], used];
		Map[ getPassData[ds, #]&, Join[ used, rest]] 
	]

getPassData[ds_, passName_] :=
	Module[ {data, path, graph, time, sub, len},
		path = Normal[
  					ds[Select[#passName === passName &]][All, "path", All,  "passName"]];
  		path = If[ Flatten[path] === {}, None, Grid[ path, Frame -> All]];
  		graph = DependencyGraph[passName];
		sub = ds[Select[#passName === passName &]];
		time = sub[Total, "runtime"];
		len = sub[Length];
		data = <|"passName" -> passName, "dependencyPath" -> path, "graph" -> graph,
		       "passTime" -> time, "calls" -> len|>;
		data
	]


getPassTimeData[ ds_] :=
	Module[ {len, divs, passTimePlots},
		len = ds[Length];
		divs = Append[ getDivs[ len, 75], -75;;-1];
		passTimePlots = 
			Map[ Function[{span},
					<| "plot" ->
						ds[Select[#type === "endRun" &]][span][
 							BarChart[##, AspectRatio -> 0.2, ImageSize -> 900] &, 
 							Labeled[#runtime, Rotate[Text[#passName], Pi/2]] &]|>], divs];
 		passTimePlots
	]

getDivs[len_, div_] :=
	Module[ {num},
		num = Ceiling[len/div];
		Table[Span[1 + (i - 1) div, If[i === num, -2, i div]], {i, num}]
	]



timeForPass[ds_, passName_] :=
	ds[Select[#passName === passName &]][Total, "runtime"]


collectPassByTime[ds_] :=
	ds[GroupBy["passName"]][
   		KeyValueMap[  <|   "name" -> #1, "calls" -> Length[#2], 
      "value" -> Total[Part[#2, All, "runtime"]] |> &]][SortBy["value"]]


plotPassByTime[ds_] :=
	collectPassByTime[ds][
  		BarChart[##, AspectRatio -> 0.2, ImageSize -> 900] &, 
  			Labeled[#value, Rotate[Text[#name <> ":" <> ToString[#calls]], Pi/2]] &]


getDependencyRules[ pass_String] :=
    Module[ {g, passObj, deps},
        passObj = Lookup[$Passes, pass, Null];
        deps = Map[#["name"] &, 
          Join[ passObj["requires"], passObj["preserves"]]];
        g = Map[pass -> # &, deps];
        DeleteDuplicates[Flatten[{g, Map[getDependencyRules, deps]}]]
    ]

DependencyGraph[pass_String] :=
    Module[ {edges, verts},
        edges = getDependencyRules[pass];
        verts = Flatten[{pass, Apply[List, edges, {1}]}];
        verts = DeleteDuplicates[verts];
        Graph[verts, edges, VertexLabels -> "Name"]
    ]

End[]


EndPackage[]
BeginPackage["Compile`Utilities`Report`TypeReport`"]

PresentTypes

TypeGraph

TypeGraphPlot

TypeReportData

TypeReport 


Begin["`Private`"]

Needs["TypeFramework`"]


getLabel[t1_ -> t2_] :=
    {t1 -> t1, t2 -> t2}

getTypeGraph[env_, ty_] :=
    Module[ {edges = getTypeGraphData[env, ty], labels},
    	If[ edges === {},
    		None,
        	labels = Flatten[Map[getLabel, edges]];
        	Graph[edges, VertexLabels -> labels]]
    ]

getAbstractTypeGraph[env_, ty_] :=
    Module[ {edges = getAbstractGraphData[env, ty["typename"]], labels},
    	If[ edges === {},
    		None,
        	labels = Flatten[Map[getLabel, edges]];
        	Graph[edges, VertexLabels -> labels, GraphLayout -> {"LayeredDigraphEmbedding", "RootVertex" -> ty["typename"]}]]
    ]



getTypeGraphData[env_, name_String] :=
    getTypeGraphData[env, env["typeconstructors"]["lookup", name]]


getTypeGraphData[env_, ty_] :=
    Module[ {implements},
        implements = ty["implements"];
        Flatten[
         Map[Prepend[getAbstractGraphData[env, #], ty["name"] -> #] &, implements]]
    ]

getAbstractGraphData[env_, name_] :=
    Module[ {obj, supers},
        obj = env["abstracttypes"]["getClass", name];
        supers = obj["supers"]["get"];
        DeleteDuplicates[
         Flatten[Map[
           Prepend[getAbstractGraphData[env, #], obj["typename"] -> #] &, 
           supers]]]
    ]



formatKind[k_] :=
	StringReplace[ k["toString"], {"*" -> "\[Star]", "->" -> "\[Rule]"}]
	


getType[ env_, fun_ -> Type[ty_]] :=
	{fun, ty}
	
getType[ env_, fun_ -> TypeSpecifier[ty_]] :=
	{fun, ty}
	
getType[env_, _] :=
	Null


privateSymbolQ[ sym_Symbol] :=
	!MemberQ[ {"System`", "Global`"}, Context[sym]]
	
privateSymbolQ[_] :=
	False
	

fixSymbol[ sym_?privateSymbolQ] :=
	Symbol[ "Global`" <> SymbolName[sym]]
	
fixSymbol[arg_] :=
	arg

formatImpl[ impl_] :=
	Module[ {ee, pos, conts},
		ee = Map[Context, Hold[impl], {2, -1}, Heads -> True];
		pos = Position[ee, HoldPattern[Context[_Symbol]], {-2}, Heads -> True];
		conts = DeleteDuplicates[ Join[ {"System`"}, Apply[Part[ee, ##] &, pos, {1}]]];
		
		RawBoxes[
			Block[ {$ContextPath = conts},
				ToBoxes[impl]
		]]
	]


getMethod[env_, fun_ -> MetaData[d_]@Typed[ty_][impl_]] :=
	getMethod[env, fun -> Typed[impl, ty]]

getMethod[env_, fun_ -> MetaData[d_]@Typed[impl_, ty_]] :=
	getMethod[env, fun -> Typed[impl, ty]]


getMethod[env_, fun_ -> Typed[ty_]@impl_] :=
	getMethod[env, fun -> Typed[impl, ty]]

getMethod[env_, fun_ -> Typed[impl_, ty_]] :=
	{fun, ty, formatImpl[impl]}

getMethod[env_, _] :=
	Null



getMemberData[env_, data_, fun_] :=
	Module[{members = data["members"]},
		members = Map[ fun[env, #]&, members];
		members = DeleteCases[members, Null];
		If[ Length[ members] === 0,
			None,
			members]
	]


getAbstractMemberData[ env_, ty_, implsData_, fun_] :=
	Module[{tyName = ty["typename"], impls, members},
		impls = Lookup[ implsData, tyName, {}];
		members = Flatten[Map[env["abstracttypes"]["getClass",#]["members"]&,impls]];
		members = Map[ fun[env, #]&, members];
		members = DeleteCases[members, Null];
		If[ Length[ members] === 0,
			None,
			members]
	] 

getAbstractFullImplements[env_, name_] :=
    Module[ {obj, supers},
        obj = env["abstracttypes"]["getClass", name];
        supers = obj["supers"]["get"];
        DeleteDuplicates[ Flatten[Join[supers, Map[ getAbstractFullImplements[env,#]&, supers]]]]
    ]

getFullImplements[env_, ty_] :=
	Module[{base, full},
		base = ty["implements"];
		full = DeleteDuplicates[ Flatten[Join[base, Map[ getAbstractFullImplements[env,#]&, base]]]];
		ty["typename"] -> full
	]
	

getInstances[ env_, data_, inst_ -> absList_] :=
	Module[ {newData},
		newData = Map[
					# -> DeleteDuplicates[ Append[ Lookup[data, #, {}], inst]]&,
					absList];
		Append[data, newData]
	]

	
TypeReportData[env_] :=
	Module[ {typeObjs, typeData, absObjs, absData,impls, implData, reverse,data = <||>},
		typeObjs = env["typeconstructors"]["types"]["values"];
		
		impls = Map[ getFullImplements[env,#]&, typeObjs];
		implData = Replace[ impls, (n_ -> {}) -> (n -> {None}), {1}];

		implData = Replace[ implData, (n_ -> l_) :> (n -> Map[List,l]), {1}];

		reverse = Fold[ getInstances[env, #1, #2]&, <||>, impls];
		reverse = Map[ Sort, reverse];
		reverse = Map[ List, reverse, {2}];
		
		typeData = Map[ <| "name" -> #["typename"], 
					"kind" -> formatKind[ #["kind"]],
					"types" -> getAbstractMemberData[env, #, impls, getType],
					"methods" -> getAbstractMemberData[env, #, impls, getMethod],
					"implements" -> Lookup[implData, #["typename"], None],
					"graph" -> getTypeGraph[env, #]|>&, typeObjs];
		typeData = SortBy[typeData, #["name"]];
		data["concrete"] =  typeData;
		
		absObjs = env["abstracttypes"]["classes"]["values"];
		
		absData = Map[ <| "name" -> #["typename"],
			"types" -> getMemberData[env, #, getType],
			"methods" -> getMemberData[env, #, getMethod],
			"instances" -> Lookup[reverse, #["typename"], None],
			"graph" -> getAbstractTypeGraph[env, #]|>&, absObjs];
		absData = SortBy[absData, #["name"]];
		data["abstract"] = absData;
				
(*		types = $BuiltinTypeEnvironment["types"]["values"];
   		types = SortBy[Cases[types, _? TypeConstructorQ], #["name"] &];
   		types = DeleteDuplicatesBy[ types,  #["name"] &];
   		types = Map[ getTypeData, types];
   		data["concrete"] = types;
   		abstractTypes = $BuiltinTypeEnvironment["abstractTypes"]["values"];
   		abstractTypes = DeleteDuplicatesBy[ abstractTypes,  #["name"] &];
   		abstractTypes = Map[ getAbstractTypeData, abstractTypes];
		data["abstract"] = abstractTypes; *)
   		data 
	]


$baseDirectory = DirectoryName[ $InputFileName]

TypeReport[env_,saveName_:"TypeReport.nb"] :=
	Module[ {data, nbTemp, nb, searchList},
		
		data = TypeReportData[env];
		
		(* Add styling to data *)
		
		data["concrete"] = Map[
			<|
				#, 
				"types" -> styleData[#["types"]],
				"methods" -> styleData[#["methods"]],
				"implements" -> generateButtonData[#["implements"]],
				"graph" -> styleData[#["graph"]]
			|> &, data["concrete"]];
		
		data["abstract"] = Map[
			<|
				#, 
				"types" -> styleData[#["types"]],
				"methods" -> styleData[#["methods"]],
				"instances" -> generateButtonData[#["instances"]],
				"graph" -> styleData[#["graph"]]
			|> &, data["abstract"]];
		
		searchList = Sort[Flatten[Values[data[[All, All, "name"]]]]];
		nbTemp = FileNameJoin[ {$baseDirectory, "Templates", "TypeReport.nb"}];
		nb = CreateDocument[GenerateDocument[nbTemp, data, "HeadlessMode" -> True]];
		
		(* Create an empty history using TaggingRules *)
		setHistoryList[{},"Notebook"->nb];
		setHistoryPosition[0,"Notebook"->nb];
		
		SetOptions[ nb, 
			{
				DockedCells -> Cell[BoxData[ToBoxes[makeSearchBar[searchList]]], "DockedCell"],
				WindowMargins->Automatic, 
				ShowCellTags -> False, 
				ShowCellLabel -> False,
				WindowTitle -> "Type Report"
			}];
		
		makeSearchCellTags[nb,searchList];
		
		nb
	]


(* Looks at all Cells with CellStyle "Subsection" (CellGroup parent of Type)
and add CellTags so that NotebookLocate/SelectionMove works.*)

makeSearchCellTags[nb_NotebookObject,searchList_List]:=
	Scan[
		With[{cellContent=NotebookRead[#][[1]]},
			If[
				(* Match if the cell's contents are a value in the List of types (searchList)  *)
				matchString[cellContent,searchList],
				SetOptions[#, CellTags -> cellContent]
			]
		]&,
		Cells[nb, CellStyle -> "Subsection"]
	]
	
(* Return True is x_String is contained in y_List *)
matchString[x_String,y_List/;AllTrue[y, Head[#] == String&]]:= StringMatchQ[x,y, IgnoreCase -> True]

(* It's possible that the BoxData isn't a String *)
matchString[___]:=False

(* Styles *)

(* 
	Data must be in form {{"x"},{"y"}} 
	Note: ButtonData -> {"x"} will do SelectionMove to Cells with CellTags->"x"
*)

hyperlinkColumn[data_List/;AllTrue[data, Head[#] == List &]]:=
	Multicolumn[
		Map[
			  Button[
				  #[[1]], 
				  Module[{linkDestination=Cells[CellTags -> #][[1]]},
					  (* Move to destination, remove all "forward" values, and add both the link and the link destination *)
					  SelectionMove[linkDestination, All, Cells];
					  resetPosition[];
					  addToHistory[EvaluationCell[]];
					  addToHistory[linkDestination];
				  ], 
				BaseStyle -> {"Hyperlink", "Text"}, 
				Appearance -> "Frameless"] &, 
		  data],
		  If[Length[data] < 4, Length[data], 4],
		  Spacings -> {1, 1}, Background -> {Automatic, {{LightOrange, White}}},
		  Frame -> All, FrameStyle -> White
	  ]
	  
(* CONSIDER: is this really necessary? *)
hyperlinkColumn[___]:= noneStyle[]

noneStyle[]:= Text["None"]


(* For Implements and Instances *)
generateButtonData[data_]:=
	If[
		data === {{None}},
		noneStyle[],
		hyperlinkColumn[data]
	]

styleData[data_]:=
	If[
		data=== None,
		noneStyle[],
		TextGrid[data, Spacings -> {1, 1}, Background -> {Automatic, {{LightOrange, 
			White}}}, Frame -> All, FrameStyle -> White]
	]

styleData[data_Graph]:=
	If[
		data=== None,
		noneStyle[],
		data
	]

(* Search Bar *)

makeSearchBar[searchList_]:=
	DynamicModule[
		{searchValue},
		Grid[{{	
				backButton[],
				forwardButton[],
				EventHandler[
					InputField[
						Dynamic[searchValue],
						String,
						FieldCompletionFunction->(Select[searchList,StringStartsQ[#,IgnoreCase -> True]]&),
						BaseStyle -> {"Text"},
						ImageSize -> {200, 25}
					]
				, {"ReturnKeyDown" :> (NotebookLocate[searchValue])}],
				Button[
					"Search",
					NotebookLocate[searchValue],
					Appearance -> "Palette", 
					ImageSize -> {50,27}, 
					ImageMargins -> {{0, 0}, {0, 0}}
				]
			}}],
		Initialization :> 
		{
		Needs["Compile`Utilities`Report`TypeReport`"];
		}
	]

(* Button behaviour *)

(* A list of Cells which are used to SelectionMove to when using the forward and back buttons. *)
$HistoryList := CurrentValue[EvaluationNotebook[], {TaggingRules, "HistoryList"}]

(* Option exists so that its possible to use setHistoryList and setHistoryPosition with the GenerateDocument code. *)
Options[setHistoryList] := {"Notebook" -> Unevaluated[EvaluationNotebook[]]}

(* Sets a List of CellObjects *)
setHistoryList[historyList_List,OptionsPattern[]]:= CurrentValue[OptionValue["Notebook"], {TaggingRules, "HistoryList"}] = historyList;

$HistoryPosition := CurrentValue[EvaluationNotebook[], {TaggingRules, "HistoryPostion"}]

Options[setHistoryPosition] := {"Notebook" -> Unevaluated[EvaluationNotebook[]]}

setHistoryPosition[historyPosition_Integer,OptionsPattern[]] := CurrentValue[OptionValue["Notebook"], {TaggingRules, "HistoryPostion"}] = historyPosition;

(* 
	NotebookObject has form NotebookObject[FrontEndObject[LinkObject["demmq_shm", 3, 1]], 78]
	2nd argument is the id, which we use for our typereports association.
*)

(* Adds a Cell link to the $HistoryList *)
addToHistory[cell_CellObject] := 
	If[Quiet[$HistoryList[[-1]]===cell],Null,setHistoryList[Append[$HistoryList,cell]]]


(* Removes Cell links from the $HistoryList *)
removeFromHistory[] := removeFromHistory[1]

removeFromHistory[x_Integer/;x > 0] := setHistoryList[$HistoryList[[;;-x]]]

removeFromHistory[___] := Null


resetHistoryList[] := ($HistoryList = {})

resetHistoryPosition[] := (setHistoryPosition[1])

(* removes all Cell links up to where the current history position is, and start from the end. *)
resetPosition[] := (removeFromHistory[$HistoryPosition];resetHistoryPosition[])

movePosition[x_Integer] := SelectionMove[$HistoryList[[-x]], All, Cells];

(* Go back in link history *)
goBack[] := 
(
	setHistoryPosition[$HistoryPosition + 1];
	movePosition[$HistoryPosition];
)

(* Go forward in link history *)
goForward[]:= 
(
	setHistoryPosition[$HistoryPosition - 1];
	movePosition[$HistoryPosition];
)

backButton[]:=
	Dynamic[
		If[
			(* Enable Button if there is a history *)
			Length[$HistoryList] <= $HistoryPosition,
				Button[
				Dynamic[RawBoxes[FEPrivate`ImportImage[FrontEnd`FileName[
				  	{"Toolbars", "DocCenter"}, "DisabledBackIcon.png"]]], 
				   	ImageSizeCache -> {21., {10., 15.}}], Inherited, Appearance -> None, 
					Enabled -> False, FrameMargins -> 0],
			 	Button[
				  PaneSelector[{
					True -> Dynamic[RawBoxes[FEPrivate`ImportImage[FrontEnd`FileName[
					{"Toolbars", "DocCenter"}, "BackIconHot.png"]]], 
					ImageSizeCache -> {21., {10., 15.}}], 
					False -> Dynamic[RawBoxes[FEPrivate`ImportImage[FrontEnd`FileName[
					{"Toolbars", "DocCenter"}, "BackIcon.png"]]], ImageSizeCache -> 
					{21., {10., 15.}}]}, Dynamic[CurrentValue["MouseOver"]]
				   ], 
				  (goBack[]), 
				  Appearance -> None, FrameMargins -> 0
			  ]
		]
	
	]

	
  	  
forwardButton[]:=
	Dynamic[
		If[
			(* If the HistoryPostion is >1, we should be able to move forward in the HistoryList *)
			$HistoryPosition <= 1,
			Button[
				Dynamic[RawBoxes[FEPrivate`ImportImage[FrontEnd`FileName[
				  	{"Toolbars", "DocCenter"}, "DisabledForwardIcon.png"]]], 
				  	ImageSizeCache -> {21., {10., 15.}}], Inherited, Appearance -> None, 
				  	Enabled -> False, FrameMargins -> 0],
			  Button[
				  PaneSelector[{
					True -> Dynamic[RawBoxes[FEPrivate`ImportImage[FrontEnd`FileName[
					{"Toolbars", "DocCenter"}, "ForwardIconHot.png"]]], 
					ImageSizeCache -> {21., {10., 15.}}], 
					False -> Dynamic[RawBoxes[FEPrivate`ImportImage[FrontEnd`FileName[
					{"Toolbars", "DocCenter"}, "ForwardIcon.png"]]], ImageSizeCache -> 
					{21., {10., 15.}}]}, Dynamic[CurrentValue["MouseOver"]]
				   ], 
				  (goForward[]), 
				  Appearance -> None, FrameMargins -> 0
			  ]
		]
	
	]


End[]


EndPackage[]
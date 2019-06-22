 (* Mathematica Package *)

(* :Title: NotebookTemplating.m *)

(* :Authors:
	Anthony Zupnik, anthonyz@wolfram.com
    Andrew Hunt, andy@wolfram.com
    Adam Berry, adamb@wolfram.com
    Fang Liu, fangl@wolfram.com
    Nick Lariviere, nickl@wolfram.com
*)

(* :Package Version: 0.50 *)

(* :Mathematica Version: 10.2 *)
                     
(* :Copyright: (c) 2010, Wolfram Research, Inc. All rights reserved. *)

(* :Requirements: *)

(* :Discussion:
  
*)


BeginPackage["NotebookTemplating`"]

(* Main public functions *)

CreateTemplateNotebook::usage = "CreateTemplateNotebook[] creates a new notebook template, CreateNotebookTemplate[nbobj] converts nbobj into a notebook template."
TemplateNotebookQ::usage = "TemplateNotebookQ[nb] returns True if nb is a notebook template."
ClearTemplateNotebook::usage = "ClearTemplateNotebook[nb] clears the data from a notebook template."

NotebookTemplateSlot::usage = "NotebookTemplateSlot"
NotebookTemplateExpression::usage = "NotebookTemplateExpression"

(*HeadlessNotebookEvaluate::usage = "Evaluates the passed notebook expression, returning a new notebook expression";*)

(*ReapedHeadlessNotebookEvaluate::usage = "Evaluates the passed notebook expression, returning a) a new notebook expression, and b)\
the list of evaluation results." *)

(*TemplateSubstituteAndExpand::usage = "TemplateSubstituteAndExpand  "*)

Begin["`Private`"]

Needs["NotebookTemplating`Utilities`" ]
Needs["NotebookTemplating`Authoring`" ]
Needs["NotebookTemplating`NotebookVisitor`" ]

SetAttributes[NotebookTemplating`NotebookTemplateExpression, HoldFirst];

prot=Unprotect[ System`NotebookTemplate]

NotebookTemplate[ string_String] :=
    Module[ {file},
        file = FindFile[ string];
        NotebookTemplate[ NotebookOpen[file, Visible -> False]] /; StringQ[file] && FileType[file] === File
    ]
    

NotebookTemplate[ nb_Notebook] :=
    NotebookTemplate[ CreateDocument[nb]]
    
NotebookTemplate[ nbObj_NotebookObject] :=
    TemplateObject[ nbObj]
    
Map[ Protect, prot]


    
System`GenerateDocument[ template : (_?(StringQ[#] && # != "" &) | _Notebook | _NotebookObject), args_Association, opts:OptionsPattern[]] :=
	System`GenerateDocument[template, {args}, "", opts]
	
System`GenerateDocument[ template : (_?(StringQ[#] && # != "" &) | _Notebook | _NotebookObject), args_List, opts:OptionsPattern[]] :=
	System`GenerateDocument[template, args, "", opts]
 
System`GenerateDocument[ template : (_?(StringQ[#] && # != "" &) | _Notebook | _NotebookObject), opts:OptionsPattern[]] :=
	System`GenerateDocument[template, {}, "", opts]
	
System`GenerateDocument[ template : (_?(StringQ[#] && # != "" &) | _Notebook | _NotebookObject), outFile_String, opts:OptionsPattern[]] :=
	System`GenerateDocument[template, {}, outFile, opts]	

System`GenerateDocument[ template_NotebookObject, rules_Association, outFile_, opts:OptionsPattern[]]:=
	System`GenerateDocument[ template, {rules}, outFile, opts]
		
System`GenerateDocument[ template_NotebookObject, rules_List, outFile_, opts:OptionsPattern[]]:=
    Module[ {nbExpr},
    	nbExpr = NotebookGet[ template];
    	If[ Head[nbExpr] === Notebook,
    		System`GenerateDocument[nbExpr, rules, outFile, opts],
    		Message[ GenerateDocument::tlvalid, template];
    		$Failed]
    ]

System`GenerateDocument[ template_String, rules:(_List|_Association), outFile_, opts:OptionsPattern[]] :=
    If[ Or[FileExistsQ[template], !MatchQ[FindFile[template], $Failed]],
        System`GenerateDocument[ Import[ template], rules, outFile, opts],
        Message[ GenerateDocument::tlnotavail, template];
        Return[$Failed]
    ]

(* 
    Run report from file 
    in: ReportGenerate[outfile, template, {list of rules}]
    out: file
*)
System`GenerateDocument[templateFile_Notebook, data_Association, outFile_String, opts:OptionsPattern[]]:=
	GenerateDocumentDriver[templateFile, data, outFile, opts]

(* https://bugs.wolfram.com/show?number=332485 *)

System`GenerateDocument[template_, ___]:=
	Message[System`GenerateDocument::tlnotavail,template]

System`GenerateDocument[template_Notebook, data_List, outFile_String, opts:OptionsPattern[]] :=
    GenerateDocumentDriver[template, Association@@data, outFile, opts]

(*File Support*)

System`GenerateDocument[File[template_String],otherParameters___]:=
	System`GenerateDocument[template,otherParameters]

System`GenerateDocument[x___ /; (Message[System`GenerateDocument::argb,System`GenerateDocument,Length[{x}],1,3]/;False)] :=
   {}

System`GenerateDocument::ftmp= "A problem was encountered in saving a temporary file."
System`GenerateDocument::unavail= "The template slot `1` cannot be filled from the arguments. "
System`GenerateDocument::tlnotavail = "The template `1` is not available."
System`GenerateDocument::tlvalid = "The template `1` is not a valid NotebookObject."


Options[GenerateDocumentDriver] :={ 
    "PostProcess" -> True,
    "HeadlessMode" -> False, 
    "ProgressIndicator"->True,
    "RetainAsTemplate" -> False
}


GenerateDocumentDriver[templateFile_Notebook, data_Association, outFile_String, opts:OptionsPattern[]] :=
    Module[ {nb,postProcess, replacedNotebook, evaluatedNotebook, feOpen, nbRes,dir,headless, progressIndicator, rat},        	
    Catch[
    	Quiet[
            postProcess =  OptionValue["PostProcess"];
            headless = OptionValue["HeadlessMode"];
            rat = OptionValue["RetainAsTemplate"];
            progressIndicator = OptionValue["ProgressIndicator"], OptionValue::nodef];
                
        feOpen = SameQ[ Head[ $FrontEnd], FrontEndObject];
        If[ (!feOpen) && (outFile === ""), Message[ FrontEndObject::notavail];Return[$Failed] ];
        nb = If[ MatchQ[templateFile,_Notebook],
                 templateFile,
                 Import[templateFile]
             ];
        
        
        (* Expand Templates (Repeating Blocks) and Substitute in values for NotebookTemplateSlots *)
        replacedNotebook = TemplateSubstituteAndExpand[nb, data, "RetainAsTemplate" -> rat];
        (* Evaluate NotebookTemplateExpressions by applying EvaluateTemplateExpressions at the Cell level with the NotebookVisitor*)
       	replacedNotebook = NotebookVisit[replacedNotebook,IdentityFunctionGroup,EvaluateTemplateExpressions,data];
       
        evaluatedNotebook = If [postProcess,
        	postProcessReplacedNotebook[replacedNotebook,headless, rat, progressIndicator, nb](*replacedNotebook*),
        	replacedNotebook] ;  
        If[ outFile === ""
        	
        	, 
        	If[!headless,
        	nbRes = ProcessWithFrontEnd[ NotebookPut[ evaluatedNotebook] ],
        	nbRes = evaluatedNotebook
        	]
        	,   	
        	ProcessWithFrontEnd[
        		dir=DirectoryName[ExpandFileName[outFile]];
        		If[!DirectoryQ[dir],CreateDirectory[dir]]
        		];
        	
        	Export[ outFile, evaluatedNotebook, If[ToLowerCase@FileExtension[outFile]==="cdf", "CDF", "NB"]];
        	If[!headless,
        	nbRes = FindFile[ outFile];	
        	nbRes = ProcessWithFrontEnd[ NotebookOpen[ nbRes] ]
        	,
        	nbRes = FindFile[ outFile];
        	]
        ];
        nbRes
    	]
    ]

(* 
    Run report from notebook expression
    In: TemplateSubstituteAndExpand[template, {Association of rules}]
	Processing: Apply NotebookVisit where:
		RepeatingBlockGroupFunction applies at the CellGroup level to expand out RepeatingBlocks and
		TemplateSubstitute applies at the Cell level to fill NotebookTemplateSlots with values
    Out: result expression
*)

TemplateSubstituteAndExpand[template_Notebook, data_Association, opts___?OptionQ] :=
    Module[ {temp, rat},
        rat = TrueQ["RetainAsTemplate" /. {opts}];
        temp = updateTemplateOptions[template, "NotebookTemplate" -> rat];
      	NotebookVisit[temp,RepeatingBlockGroupFunction,TemplateSubstitute,data]      
    ]

(*TemplateSubstitute is called by the Notebook Visitor to process the contents of cells, replacing NotebookTemplateSlots with values. *)
TemplateSubstitute[cellData_,fullData_]:=
Module[{data=fullData["UserData"], repeatData=fullData["RepeatingData"], newBoxData},
	(*For example, if we have in UserData <|"a" -> 1000, "b" -> 1000|> and in RepeatingData <|"a" -> 1, "b" -> 2|>, we want RepeatingData to Override UserData currently
	TODO not sure if this is currently the right place to have this code run, related to this line:
	newCellDataList={{newHeadCell,Sequence@@otherCells},newData=Append[data,"RepeatingData"->{#}]}&/@repeaterData
	*)
	(*Check to see if this should be repeated but we didn't come from a cellGroup *)
	If[
		repeatingLabelQ[getRepeatingBlockLabel[cellData]],
		NotebookVisit[Cell[CellGroupData[{cellData}, Open]],RepeatingBlockGroupFunction,TemplateSubstitute,fullData]
		,
		If[AssociationQ[repeatData],data=Association[data,repeatData](*Replaces Global Assocations to local ones, but is non destructive eg. if there is a Repeating Block within a Repeating Block*)];
	
		newBoxData=Replace[cellData,
			Cell[BoxData[FormBox[TemplateBox[{var_, default_, mode_, contentData_}, "NotebookTemplateSlot", ___], TextForm]], ___] :>
		            If[mode==="Named",
		            	
		            	formatTemplateVariable[makeValue[(*TODO Remove the need for the list wraparound*){data}, "", var, default], None, contentData],
		            	(*Means this is positional*)
		            	(*TODO if RepeatingData has no key and if its outside of a RepeatingBlock, what do we expect?*)
		            	
		            	ToBoxes[ListOrAssocation[repeatData][[ToExpression[var]]]]
		            ]
		            	,{0,Infinity}];
		newBoxData
	]
	
]

(* Turns to List if it is not a List Or Association *)
ListOrAssocation[data_]:=(If[ListQ[data] || AssociationQ[data], data, (*TODO: If the repeatingData basically isn't a list or Association, what do we do?*)<|1->data|>])

(* 
	Expand out a RepeatingBlock CellGroup and Data

	At the CellGroup level we look for a Repeating Block
	
	If True:
		Map across the NotebookTemplateSlot/NotebookTemplate expression
			strip out the CellFrameLabel information
			create a new list of Cell data and the original data
			appending to the data which is local to the Repeating Block to be passed down the Notebook Visitor
			
	If False:
		Return the Identity	
*)

RepeatingBlockGroupFunction[cellList_,data_]:=(
	Module[{ label, inherit, exp, newHeadCell, headCell=cellList[[1]], otherCells=cellList[[2;;]], repeaterData, newCellDataList},
			(*Get CellFrameLabel from the head cell*)
	        label = getRepeatingBlockLabel[headCell];
	       
			exp = If[
	        		(*If the label is a repeatingLabelQ*)
	        		repeatingLabelQ[label],
	        		(*Replace NotebookTemplateExpressions and NotebookTemplateSlots*)
	        		inherit = inheritQ[label];
	        		(*TODO: Should probably look at standardising this with the way that we handle TempalteExpressions and Slots*)
	        		repeaterData=Replace[label,{
	        			Cell[BoxData[TemplateBox[{expression_,"NotebookTemplateExpression",True},"NotebookRepeatingBlock"]]]:>
	        				(ToExpression[expression]),
	        			Cell[BoxData[TemplateBox[{slotVariable_,"NotebookTemplateSlot",True},"NotebookRepeatingBlock"]]]:>
	        				(With[{expressionAttempt=data["RepeatingData", ToExpression[slotVariable]]}
	        					,
	        					If[MissingQ[expressionAttempt],
	        						data["UserData", ToExpression[slotVariable]],
	        						expressionAttempt
	        					]
	        				]
	        				)},{0,Infinity}];
	        		newHeadCell = removeLabelsRB[headCell];
	        			        		newCellDataList=
	        			{{newHeadCell,Sequence@@otherCells},
	        				Module[
	        					{newData=data},
	        					(*TODO:RepeatingData is basically useless here, I've hacked this together and it should be changed
	        					It may not even be required, code does need to be reworked though.
	        					*)
	        					(*newData["UserData"]=Append[newData["UserData"], #];*)
	        					newData=Append[newData,"RepeatingData"->ListOrAssocation[#]];
	        					newData
	        				]
	        			}&/@ListOrAssocation[repeaterData];

	        		,
	        		(*If False then return identity*)
	        		newCellDataList={{cellList,data}};
	             	];
	    newCellDataList
	]
		
		)


(*Evaluate any NotebookTemplateExpression Cells within a Cell*)
EvaluateTemplateExpressions[cellData_,data_]:=
	Replace[cellData,
		Cell[BoxData[FormBox[TemplateBox[{var_,"General",contentData_},"NotebookTemplateExpression",___],TextForm]],___]:>
			formatExpr[ToExpression[var],contentData],{0,Infinity}]

postProcessReplacedNotebook[replacedNotebook_Notebook, headless_, rat_, progressIndicator_, nb_Notebook] := 
    Module[ {res, tmp, tempFile, nbMenuOpts, oldOutputFormat},
        If[!headless && CloudSystem`$CloudNotebooks =!= True, 
            UsingFrontEnd[
                If[progressIndicator,tmp = PrintTemporary[ProgressIndicator[Dynamic[Clock[Infinity]], Indeterminate]]];
                nbMenuOpts = NotebooksMenu /. Options[$FrontEnd, NotebooksMenu];
 	              res = taggingInputCells[replacedNotebook, rat]; 	              
 	              tempFile = FileNameJoin[{$TemporaryDirectory, ToString[AbsoluteTime[DateString[]]] <> "_MathematicaReport.nb"}];
 	              Export[tempFile, res];
 	              If[Not[FileExistsQ[tempFile]], Message[GenerateDocument::ftmp];Throw[$Failed, "reportTag"]];
 	              oldOutputFormat = Options[$Output, FormatType];
 	              (*Added so that GenerateDocument works with standalone kernel. Bug 298488*)
 	              SetOptions[$Output, FormatType -> StandardForm];
 	              NotebookEvaluate[tempFile, InsertResults -> True];
 	              SetOptions[$Output, oldOutputFormat];
 	              res = Import[tempFile];
 	              DeleteFile[tempFile];
 	            SetOptions[$FrontEnd, NotebooksMenu->nbMenuOpts];  
 	              If[progressIndicator,Quiet[NotebookDelete[tmp]]]];
 	              ,  (* else headless *)
            res = taggingInputCells[replacedNotebook, rat];
            res = ReleaseHold[HeadlessNotebookEvaluate[res]]
        ];
        displayInputCells[res]
    ]

System`GenerateDocument::multivar="Template variable `1` appears more than once in the same group. Only the first will be used."     

formatExpr[ value_String, TextData] := value
formatExpr[ value_String, BoxData] := ToBoxes[value]


formatExpr[ value_, TextData] := Cell[ BoxData[ FormBox[ToBoxes[value],TextForm]]]
formatExpr[ value_, BoxData] := ToBoxes[value]     
        
ConvertValue[ value_, exp_TemplateBox] :=
    If[ Head[value] === String,
        EscapeString[ value],
        ToBoxes[value]
    ]
    
ConvertValue[value_, exp_Cell] :=
    If[ Head[value] === String,
        value,
        ToString[value]
    ]

EscapeString[string_String] :=
    StringJoin["\"" <> string <> "\""]
    
taggingInputCells[Notebook[a_List, c___], rat_] := 
 Notebook[Map[taggingInputCells[#, rat]&, a], c]
 
taggingInputCells[gp:Cell[CellGroupData[{first_Cell, other___}, c___]], rat_] := 
 Module[{cells,label, tag, res},
 	cells= gp;
 	label = GetEvaCellLabel[ first];
	tag = label /. MapThread[Rule, {evaluationTooltipLabels, evaluationTags}];
	If[!MatchQ[tag,{}],
    res = 
	If[MatchQ [tag, evaluationTags[[4]]],
	cells ->Sequence[],
	Verbatim[cells]->Cell[CellGroupData[Map[taggingSubCells[#, tag, rat]&, {first, other}], c]]
	];
	cells/.res
	,
	Cell[CellGroupData[Map[taggingInputCells[#, rat]&, {first, other}], c]]]
 ] 

taggingInputCells[CellGroupData[a_List, c___], rat_] := 
 Cell[CellGroupData[Map[taggingInputCells[#, rat]&, a], c]]

taggingInputCells[expr_Cell, rat_] := 
 taggingFromLabel[expr, rat]

taggingInputCells[x___, rat_] := x 

taggingSubCells[a_List, tag_, rat_] := 
 Map[taggingSubCells[#, tag, rat]&, a]
 
taggingSubCells[Cell[CellGroupData[lis_List, c___]], tag_, rat_] := 
 Cell[CellGroupData[Map[taggingSubCells[#, tag, rat]&, lis], c]]
 
taggingSubCells[expr_Cell, tag_, rat_] := 
 taggingFromLabel[expr, tag, rat] 
 
taggingSubCells[x_,tag_, rat_]:=x 
    	
taggingFromLabel[cell_Cell, rat_]:=
	Module[{label, tag },
		label = GetEvaCellLabel[cell];
	    tag = label /. MapThread[Rule, {evaluationTooltipLabels, evaluationTags}];
        taggingFromLabel[cell, tag, rat]
    ]
 
taggingFromLabel[cell_Cell, tag_, rat_]:=
	Module[{cellNew },
		cellNew= cell;
        If[MatchQ[tag,evaluationTags[[3]]], (*Unevaluated*)
        	cellNew = Append[cell, Evaluatable -> False]];
        cellNew = If[TrueQ@rat, cellNew, removeLabelsEvaCell[cellNew]];	
        If[!MatchQ[tag,{}],
        	cellNew = addTaggingRls[cellNew, {tag -> True}]];
        If[
        	MatchQ[tag,evaluationTags[[4]]],(*Exclude*)
        	cellNew = Sequence[]];
        cellNew
    ] 
         
displayInputCells[ notebook_Notebook] :=
    getEvaluationGroups[notebook]
    
addTaggingRls[cell_Cell, rls_List]:=
	Module[{taggingRules},
		taggingRules = TaggingRules /. Options[cell, TaggingRules];
		If[MatchQ[taggingRules, TaggingRules],
			Append[cell, TaggingRules -> rls],
			Replace[cell,Rule[TaggingRules, _]:>(TaggingRules->Union[Join[taggingRules,rls]]),{1}]]
	]
	
removeTaggingRls[cell_Cell, rl_Rule]:= 
	Module[{taggingRules, taggingRulesNew},
		taggingRules = TaggingRules /. Options[cell, TaggingRules];
		taggingRulesNew = DeleteCases[taggingRules,rl];
		If[taggingRulesNew==={},
			DeleteCases[cell, TaggingRules -> _],
			Replace[cell,Rule[TaggingRules, _]:>taggingRulesNew,{1}] ]
	] 	    	

getEvaluationGroups[ Cell[CellGroupData[{first_Cell /; GetEvaCellTaggingRls[first] =!= -1, other___}, args___]]] := 
	fixInput[Cell[CellGroupData[{first, other}, args]]]
 
getEvaluationGroups[Cell[CellGroupData[a_List, c___]]] := 
	Cell[CellGroupData[getEvaluationGroups /@ a,c]]

getEvaluationGroups[x_Cell] := fixInput[x] 

getEvaluationGroups[Notebook[a_List, c___]] := Notebook[getEvaluationGroups /@ a,c]

getEvaluationGroups[x___] := x
 
getOutputIndex[ cells_List] :=
    Module[ {pos},
        pos = Position[cells, Cell[ _,"Output",___]];
        If[ MatchQ[pos, {{_Integer},___}],Part[pos,1,1], -1]
    ]

(*
fixInput happens after the evaluation of Template Notebook:
	Applies Cell Behaviour
	Removes TaggingRules
*) 

fixInput[ group:Cell[CellGroupData[{inputCell:Cell[a_, style_ , b___], other__}, ext___]]] :=
    Module[ {tag, res, index, len,inputQ,firstCell},
    	inputQ = isEvaluationStyle[style];
    	len = Length[{other}];
        tag = GetEvaCellTaggingRls[inputCell];
        Switch[tag,
        	evaluationTags[[1]] (*EvaluateDeleteInput*)
            ,
            If[inputQ,res = First[{other}];
            	If[ 
            	len === 1,
            	   res,
            	   res = If[FreeQ[res, Rule[CellGroupingRules, _]], 
            	   	           Append[res, CellGroupingRules -> "InputGrouping"],res];
            	   Cell[ CellGroupData[ {res, Rest[{other}]}]]
           		]
            	,res = Sequence[] (*If not an evaluatable Cell, then we essentially show nothing*)
            ];
            
            ,       	
            evaluationTags[[2]] (*EvaluateHideInput*)
            ,
            firstCell = removeTaggingRls[inputCell,tag->True];
            (*If this isEvaluationStyle (Code|Input cell types), simply add to 1 to the index as Output should be one after the Input*)
            If[inputQ, 	
            	index = getOutputIndex[{other}];
	            index = {If[ index === -1, 2 (*Default to 2 if index goes wrong*), index+1]};
	            (*res = removeTaggingRls[inputCell,tag->True];*)
	            res = Cell[ CellGroupData[ {firstCell, other}, index]];
	            ,
                (*If first cell is not inputQ, then seperate first cell and the rest and apply fixInput again*)
	            res = Cell[ CellGroupData[ fixInput/@{firstCell, other}, Closed]]];
            ,            
            evaluationTags[[3]] (*Unevaluated*)
            ,
            
            res = inputCell;
            res = removeTaggingRls[res,tag->True]; 
            res = DeleteCases [res, Rule[Evaluatable, False]];
            res = Cell[ CellGroupData[ fixInput/@{res, other}, ext]]
            ,
            True
            ,
            res = group
            ];
            res
]
      
fixInput[ expr_Cell ] :=
    Module[ {tag,res},
        tag = GetEvaCellTaggingRls[expr];
        res = expr;
        If[ 
        	MatchQ[ tag, evaluationTags[[3]]](*Unevaluated*)
            ,
            res = removeTaggingRls[expr,tag->True]; 
            res = DeleteCases [res, Rule[Evaluatable, False]]];
        If[ 
        	MatchQ[ tag, evaluationTags[[1]]] (*EvaluateDeleteInput*)
            ,
            res = Sequence[]; 
            ];    
         res
        ] 
        
fixInput[ group_List ]:=fixInput/@group

fixInput[x_]:=x	                    

getNotebookFileName[notebook_NotebookObject]:=With[{title=Quiet[NotebookFileName[notebook]]},
	If[StringQ[title],
		StringReplace[Last[FileNameSplit[title]],".nb"->""],
		"Untitled",
		""]]
getNotebookFileName[___]:="Untitled"

makeValue[data_, global_, var_, default_] := Module[
	{value},
	value = varValue[data, global, var];
	If[ MatchQ[value, $Failed],
		NotebookTemplating`Authoring`Private`fixPreview[default],
		value
    ]
]

varValue[data_, global_, var_] := Module[
	{arg, res},
	arg = ToExpression[ RowBox[ {"{", var, "}"}]];
	If[ MatchQ[ arg, {_String, ___}],
        arg = Prepend[ arg, 1]
    ];
    res = extractVariableValue[ data, arg];
    If[ res === $Failed,
    	res = extractVariableValue[ global, arg]
    ];
    res
]

extractVariableValue[data_, arg_] := Module[
	{res},
	res = With[{ex = Quiet[Extract[ data, arg]]},
		(* In cases where the value is a slot, this will strip out the ExpressionCell mucky-muck. *)
		ex /. ExpressionCell[_[n : _NotebookTemplating`NotebookTemplateSlot | _NotebookTemplating`NotebookTemplateExpression]] :> n
	];
	Which[
		Head[res] === Missing,
		    $Failed,
		Head[res] === Extract,
            $Failed,
        True,
            res
    ]
]


NotebookTemplating`NotebookTemplateSlot /: MakeBoxes[NotebookTemplating`NotebookTemplateSlot[name_], fmt_] := 
    MakeBoxes[NotebookTemplating`NotebookTemplateSlot[name, Null, "Named", BoxData], fmt]

NotebookTemplating`NotebookTemplateSlot /: MakeBoxes[NotebookTemplating`NotebookTemplateSlot[name_, def_], fmt_] := 
    MakeBoxes[NotebookTemplating`NotebookTemplateSlot[name, def, "Named", BoxData], fmt]

NotebookTemplating`NotebookTemplateSlot /: MakeBoxes[NotebookTemplating`NotebookTemplateSlot[name_, def_, type_], fmt_] := 
    MakeBoxes[NotebookTemplating`NotebookTemplateSlot[name, def, type, BoxData], fmt]

NotebookTemplating`NotebookTemplateSlot /: MakeBoxes[NotebookTemplating`NotebookTemplateSlot[name_, def_, type_, format_], fmt_] := 
    Cell[BoxData[FormBox[TemplateBox[{MakeBoxes[name, fmt], MakeBoxes[def, fmt], MakeBoxes[type, fmt], format}, "NotebookTemplateSlot"], TextForm]]];

NotebookTemplating`NotebookTemplateExpression /: MakeBoxes[NotebookTemplating`NotebookTemplateExpression[v_], fmt_] := 
    MakeBoxes[NotebookTemplating`NotebookTemplateExpression[v, General, BoxData], fmt]

NotebookTemplating`NotebookTemplateExpression /: MakeBoxes[NotebookTemplating`NotebookTemplateExpression[v_, format_], fmt_] := 
    MakeBoxes[NotebookTemplating`NotebookTemplateExpression[v, General, format], fmt]

NotebookTemplating`NotebookTemplateExpression /: MakeBoxes[NotebookTemplating`NotebookTemplateExpression[v_, type_, format_], fmt_] := 
    Cell[BoxData[FormBox[TemplateBox[{MakeBoxes[v, fmt], MakeBoxes[type, fmt], format}, "NotebookTemplateExpression"], TextForm]]];

NotebookTemplateVerbatim[templateExpression_NotebookTemplating`NotebookTemplateExpression]:=Inactivate[templateExpression]

formatTemplateVariable[ value_String, None, TextData] := value
formatTemplateVariable[ value_String, None, BoxData] := ToBoxes[value]

formatTemplateVariable[ value_String, TextData, opt___] := StyleBox[ value, opt]
formatTemplateVariable[ value_String, BoxData, opt___] := StyleBox[ ToBoxes[value], opt]

otherValueQ[value_] := MatchQ[value, 
	Except[Alternatives[
		_String
	]]
];

formatTemplateVariable[ value_?otherValueQ, None, TextData] := Cell[ BoxData[ FormBox[ToBoxes[value], TextForm]]]
formatTemplateVariable[ value_?otherValueQ, None, BoxData] := ToBoxes[value]

formatTemplateVariable[ value_, opt_, contentData_] := Cell[ BoxData[ StyleBox[ ToBoxes[value], opt]]]

(***Start: Functions used for HeadlessNotebook Evaluation ***)
evaluatableCellQ[ style_, opts_List] :=
    Quiet[
        Module[ {eval},
            eval = Evaluatable /. opts;
            Which[
            	eval === True,
            	   True,
            	eval === False,
            	   False,
            	True,
            	   isEvaluationStyle[style]
            ]
        ]
    ]


createInput[ a_String] := ToExpression[a]

cnt = 0;

getFragment[inp_String] :=
    (
    Sow[ "base.notebooktemplating.temp" :> ""];
    inp
    )
    
getFragment[inp_] := 
    Module[ {res, name, rule},
    	cnt++;
    	name = "base.notebooktemplating.temp" <> ToString[cnt];
    	res = MakeExpression[inp, StandardForm];
    	res = Replace[ res, {HoldComplete[ ExpressionCell[TextForm[x_]]] -> HoldComplete[x], 
    		                   HoldComplete[ ExpressionCell[x_]] -> HoldComplete[x]}];
    	rule = 
    	   With[ {name1 = name},
        	 Apply[Function[val, (name1 :> val), {HoldAll}], res]];
    	Sow[rule];
    	ToString[ name, InputForm]
    ]

createInput[ TextData[a_List]] := 
    Module[ {tmp, rules},
    	{tmp, rules} = Reap[ Map[ getFragment, a]];
        If[ !MatchQ[ rules, {_}], Return[ $Failed]];        
        If[ !MatchQ[ tmp, {__String}], Return[ $Failed]];
    	tmp = StringJoin @@ tmp;
    	tmp = ToExpression[ tmp, InputForm, Hold];
    	tmp = tmp /. First[ rules];
    	ReleaseHold[ tmp]
    ]

makeOutputGroup[ inp_, style_, opts_] :=
    Module[ {resultExp, resultBox, displayForm, content},
    	resultExp = createInput[inp];
    	If[ resultExp === Null,
    	   Cell[inp, style, Apply[ Sequence, opts]],
    	   displayForm = Head[resultExp];
    	   resultBox = 
                With[ {res = resultExp},
                    If[ MemberQ[displayForms, displayForm] && Length[resultExp] === 1,
                        content = res[[1]];
                        FormBox[MakeBoxes[content, StandardForm], displayForm],
                        MakeBoxes[res, StandardForm]
                    ]
                ];
            Cell[CellGroupData[{Cell[inp, style, Apply[ Sequence, opts]], 
            Cell[BoxData[resultBox], "Output"]}]]
    	]
    ]


(*Input/Code cells: multiple evaluation within one cell (not compound expression) *)
HeadlessNotebookEvaluate[
  Cell[BoxData[a_List], style_ /; isEvaluationStyle[style], 
   c___]] := 
 Module[{resultExp, resultBox, displayForm, content}, 
  resultExp = DeleteCases[ToExpression /@ a, Null];
  Sow[resultExp, "Unused?"];
  resultBox = 
   With[{res = #}, 
      If[MemberQ[Head[#], displayForm], content = res[[1]];
       FormBox[MakeBoxes[content, StandardForm], Head[#]], 
       MakeBoxes[res, StandardForm]]] & /@ resultExp;
  Cell[CellGroupData[{Cell[BoxData[a], style, c], 
     Sequence @@ (Cell[BoxData[#], "Output"] & /@ resultBox)}, Open]]]

(*Input/Code cells, with Evaluatable can be False/True*)
HeadlessNotebookEvaluate[
  Cell[BoxData[a_], style_ /; isEvaluationStyle[style], c___, 
   Evaluatable -> False, d___]] :=
    Cell[BoxData[a], style, c, Evaluatable -> False, d]

HeadlessNotebookEvaluate[
  Cell[BoxData[a_], style_ /; isEvaluationStyle[style], 
   c___]] :=
    Module[ {resultExp, resultBox, displayForm, content},
        resultExp = ToExpression[a];
		Sow[resultExp, "Interesting"];
        displayForm = Head[resultExp];
        resultBox = 
         With[ {res = resultExp},
             If[ MemberQ[displayForms, displayForm],
                 content = res[[1]];
                 FormBox[MakeBoxes[content, StandardForm], displayForm],
                 MakeBoxes[res, StandardForm]
             ]
         ];
        If[MatchQ[resultBox, "Null"], 
        	Cell[BoxData[a], style, c],
 			Cell[CellGroupData[{Cell[BoxData[a], style, c], 
    			Cell[BoxData[resultBox], "Output"]}, Open]]]
    ]

HeadlessNotebookEvaluate[
  Cell[CellGroupData[{Cell[BoxData[a_], 
      style_ /; isEvaluationStyle[style], b___, 
      Evaluatable -> False, c___], Cell[BoxData[d_], "Output", f___]},
     g___]]] :=
    Cell[CellGroupData[{Cell[BoxData[a], style, b, Evaluatable -> False, 
        c], Cell[BoxData[d], "Output", f]}, g]]

HeadlessNotebookEvaluate[
  Cell[CellGroupData[{Cell[BoxData[a_], 
      style_ /; isEvaluationStyle[style], c___], 
     Cell[BoxData[d_], "Output", f___]}, g___]]] :=
    Module[ {resultExp, resultBox, displayForm},
        resultExp = ToExpression[a];
		Sow[resultExp, "Interesting"];
        displayForm = Head[resultExp];
        resultBox = 
         With[ {res = resultExp},
             If[ MemberQ[displayForms, displayForm],
                 FormBox[MakeBoxes[res, StandardForm], displayForm],
                 MakeBoxes[res, StandardForm]
             ]
         ];
        Cell[CellGroupData[{Cell[BoxData[a], style, c], 
           Cell[BoxData[resultBox], "Output", f]}, g]]
    ]

HeadlessNotebookEvaluate[
  Cell[CellGroupData[{Cell[BoxData[a_], 
      style_ /; isEvaluationStyle[style], b___, 
      Evaluatable -> False, c___], Cell[d_, "Output", f___]}, 
    g___]]] :=
    Cell[CellGroupData[{Cell[BoxData[a], style, b, Evaluatable -> False, 
        c], Cell[d, "Output", f]}, g]]

HeadlessNotebookEvaluate[
  Cell[CellGroupData[{Cell[BoxData[a_], 
      style_ /; isEvaluationStyle[style], c___], 
     Cell[d_, "Output", f___]}, g___]]] :=
    Module[ {resultExp, resultBox, displayForm},
        resultExp = ToExpression[a];
		Sow[resultExp, "Unused?"];
        displayForm = Head[resultExp];
        resultBox = 
         With[ {res = resultExp},
             If[ MemberQ[displayForms, displayForm],
                 FormBox[MakeBoxes[res, StandardForm], displayForm],
                 MakeBoxes[res, StandardForm]
             ]
         ];
        Cell[CellGroupData[{Cell[BoxData[a], style, c], 
           Cell[BoxData[resultBox], "Output", f]}, g]]
    ]

(*Text like cells, with Evaluatable can be False/True*)
HeadlessNotebookEvaluate[
    Cell[a_String, style_, opts___] /; evaluatableCellQ[style, {opts}]] :=
        makeOutputGroup[a, style, {opts}]

HeadlessNotebookEvaluate[
    Cell[TextData[a_], style_, opts___] /; evaluatableCellQ[style, {opts}]] :=
        makeOutputGroup[TextData[a], style, {opts}]


HeadlessNotebookEvaluate[Cell[CellGroupData[a_List, c___]]] :=
    Cell[CellGroupData[HeadlessNotebookEvaluate /@ a, c]]

HeadlessNotebookEvaluate[Notebook[a_List, c___]] := Notebook[HeadlessNotebookEvaluate /@ a, c]

HeadlessNotebookEvaluate[x___] := x

(* Include evaluation trail. *)
ReapedHeadlessNotebookEvaluate[Notebook[a_List, c___]] :=
    Reap[Notebook[HeadlessNotebookEvaluate /@ a, c], "Interesting"] /. {nb_Notebook, {res_List}} :> {nb, res}

ReapedHeadlessNotebookEvaluate[x___] := x
(***End: Functions used for HeadlessNotebook Evaluation ***)

End[]

EndPackage[]

(* GUIFromXML subpackage *)

(* :Context: GUIKit` *)

(* :Copyright: Copyright 2004, Wolfram Research, Inc. *)

(* Each subpackage is called within `Private` *)

(*******************************
   Options
 *******************************)

(*******************************
   Messages
 *******************************)

(*******************************
   Register $GUIKitXMLFormat
   as a module to Import[]
 *******************************)
ImportExport`RegisterFormat[$GUIKitXMLFormat];


If[ $VersionNumber >= 6.0,
	ImportExport`RegisterImport[$GUIKitXMLFormat,
		GUIKitXMLGet,
		ImportExport`BinaryFormat -> False,
		ImportExport`FunctionChannels -> {"FileNames", "Streams"}
	],
	ImportExport`RegisterImport[$GUIKitXMLFormat,
		GUIKitXMLGet,
		ImportExport`BinaryFormat -> False,
		ImportExport`FunctionChannels -> {FileNames, Streams}
	]	
]
  
GUIKitXMLGet[source_, opts___] := Module[{result},
  result = Import[source, "XML", opts];
  If[ result === $Failed, Return[$Failed]];
  SymbolicGUIKitXMLToWidget[ result]
  ]
   
(*******************************
   SymbolicGUIKitXMLToWidget   

 *******************************)
 
SymbolicGUIKitXMLToWidget[ o:XMLObject[___][___], opts___] :=
  xmlToGUI[o]

SymbolicGUIKitXMLToWidget[ o:XMLElement[___], opts___] :=
  xmlToGUI[o]


xmlToGUI[d:XMLObject["Document"][{prolog___}, root_XMLElement, {epilog___}, opts___]] :=
  Module[{prologResult, rootResult, epilogResult},
  
  prologResult = Flatten[{xmlToGUI /@ {prolog}}];
  epilogResult = Flatten[{xmlToGUI /@ {epilog}}];
  
  rootResult = xmlToGUI[ root];
  
  If[ Length[prologResult] > 0 && MemberQ[{Widget, WidgetReference, InvokeMethod}, Head[rootResult]],
    rootResult = ReplacePart[rootResult, Join[ prologResult, rootResult[[2]]], 2],
    (* we should issue a warning here as prolog content is lost in result *)
    {}
    ];
    
  rootResult
  ]
  
xmlToGUI[m:XMLElement[("widget" | {_, "widget"}), {atts___}, {children___}]] :=
  Module[{src, ref, cl, id, args = None, exposeList = None, layoutAtt = None, layout = Automatic,
     useChildren = {children}, childContent},
  
    (* Currently src and class are identical but we may phase out class or change its use *)
    {id, ref, src, cl, layoutAtt} = extractAttributes[ {"id", "ref", "src", "class", "layout"}, {atts}];
      
    If[ layoutAtt =!= None,
      layout = Switch[layoutAtt,
        "None", None,
        "Column", Column,
        "Row", Row,
        "Grid", Grid,
         _, Automatic];
      ];
      
    (* need to do a pass for InitialArguments option and for ExposeWidgetReferences option *)
    If[ MemberQ[ useChildren, XMLElement[("args" | {_, "args"}), __] ],
       args = xmlToGUI[ First[ Cases[ useChildren, XMLElement[("args" | {_, "args"}), __]]] ];
       ];
    If[ MemberQ[ useChildren, XMLElement[("expose" | {_, "expose"}), __] ],
       exposeList = xmlToGUI /@ Cases[ useChildren, XMLElement[("expose" | {_, "expose"}), __]];
       ];
    If[ MemberQ[ useChildren, XMLElement[("layout" | {_, "layout"}), __] ],
       layout = First[ Cases[ useChildren, XMLElement[("layout" | {_, "layout"}), __]]];
       layout = Flatten[{xmlLayoutElementsToGUI /@ Last[layout]}];
       If[ layout === {}, layout = Automatic];
       If[ MatchQ[layout, {"Grouping" -> val_Symbol}], layout = layout[[1,2]] ];
       ];
       
    useChildren = DeleteCases[useChildren, 
      XMLElement[("args" | {_, "args"}), __]  | XMLElement[("expose" | {_, "expose"}), __] | XMLElement[("layout" | {_, "layout"}), __]
      ];
      
    childContent = xmlToGUI /@ useChildren;
    If[ MatchQ[childContent, {WidgetGroup[___]}] || MatchQ[childContent, {{___}}], childContent = First[childContent]];
    
    If[ ref =!= None,
      (* TODO support a class specification for WidgetReference as supported in XML version *)
      WidgetReference[ If[id =!= None, ref -> id, ref], 
         If[ childContent =!= {}, childContent, Unevaluated[Sequence[]] ],
         If[ layout =!= Automatic, WidgetLayout -> layout, Unevaluated[Sequence[]] ]
         ],
         
      If[ src === None, 
        If[ cl =!= None && StringQ[cl],
          src = If[ StringMatchQ[cl, "class:*"], cl, "class:" <> cl];
          ];
        ];   
      (* This should generate an error? *)
      If[ src === None, src = ""];
      Widget[src, 
         If[ childContent === {}, Unevaluated[Sequence[]], childContent], 
         If[id =!= None, Name -> id, Unevaluated[Sequence[]]  ],
         If[args =!= None, InitialArguments -> args, Unevaluated[Sequence[]] ],
         If[exposeList =!= None, ExposeWidgetReferences -> exposeList, Unevaluated[Sequence[]]  ],
         If[ layout =!= Automatic, WidgetLayout -> layout, Unevaluated[Sequence[]] ]
        ]
      ]
    ]
    
    
xmlToGUI[m:XMLElement[("group" | {_, "group"}), {atts___}, {children___}]] :=
  Module[{id, layoutAtt = None, layout = Automatic, useChildren = {children}, childContent},
  
    {id, layoutAtt} = extractAttributes[ {"id", "layout"}, {atts}];
      
    If[ layoutAtt =!= None,
      layout = Switch[layoutAtt,
        "None", None,
        "Column", Column,
        "Row", Row,
        "Grid", Grid,
        "Split", Split,
         _, Automatic];
      ];
      
    (* need to do a pass for InitialArguments option and for ExposeWidgetReferences option *)
    If[ MemberQ[ useChildren, XMLElement[("layout" | {_, "layout"}), __] ],
       layout = First[ Cases[ useChildren, XMLElement[("layout" | {_, "layout"}), __]]];
       layout = Flatten[{xmlLayoutElementsToGUI /@ Last[layout]}];
       If[ layout === {}, layout = Automatic];
       If[ MatchQ[layout, {"Grouping" -> val_Symbol}], layout = layout[[1,2]] ];
       ];

    useChildren = DeleteCases[useChildren, 
      XMLElement[("layout" | {_, "layout"}), __]
      ];
    
    childContent = xmlToGUI /@ useChildren;
    
    If[ layout =!= Automatic || id =!= None,
       WidgetGroup[childContent, 
        If[id =!= None, Name -> id, Unevaluated[Sequence[]]  ],
        WidgetLayout -> layout],
       childContent
       ]
    ]
    

xmlLayoutElementsToGUI[ m:XMLElement[("grouping" | {_, "grouping"}), {"type" -> "Tabs"}, {
    XMLElement[("tabs" | {_, "tabs"}), {"orient" -> or_}, {childStrings___}]}] ] :=
  "Grouping" -> {Tabs, Switch[or, "Left", Left, "Right", Right, "Bottom", Bottom, _, Top], 
    xmlToGUI /@ {childStrings}}
  
xmlLayoutElementsToGUI[ m:XMLElement[("grouping" | {_, "grouping"}), {"type" -> "Split"}, {
   XMLElement[("split" | {_, "split"}), {"orient" -> or_}, {}]}] ] :=
  "Grouping" -> {Split, Switch[or, "Horizontal", Horizontal, _, Vertical]}
  
(* TODO instead of ToExpression this should match strings for valid values *)

xmlLayoutElementsToGUI[ m:XMLElement[("grouping" | {_, "grouping"}), {"type" -> t_}, {}] ] :=
  "Grouping" -> ToExpression[t]
xmlLayoutElementsToGUI[ m:XMLElement[("grouping" | {_, "grouping"}), {}, {child_}] ] :=
  "Grouping" -> xmlToGUI[child]
  
xmlLayoutElementsToGUI[ m:XMLElement[("alignment" | {_, "alignment"}), {"type" -> t_}, {}] ] :=
  "Alignment" -> (If[ Length[#] > 1, #, First[#]])& [ToExpression["{" <> t <> "}"]]

xmlLayoutElementsToGUI[ m:XMLElement[("stretching" | {_, "stretching"}), {"type" -> t_}, {}] ] :=
  "Stretching" -> (If[ Length[#] > 1, #, {First[#], First[#]}])& [ToExpression["{" <> t <> "}"]]
  
xmlLayoutElementsToGUI[ m:XMLElement[("spacing" | {_, "spacing"}), {"value" -> t_}, {}] ] :=
  "Spacing" -> ToExpression[t]
  
xmlLayoutElementsToGUI[ m:XMLElement[("border" | {_, "border"}), {"title" -> t_}, {}] ] :=
  "Border" -> t
xmlLayoutElementsToGUI[ m:XMLElement[("border" | {_, "border"}), {"left" -> l_, "right" -> r_, "top" -> t_, "bottom" -> b_}, {}] ] :=
  Block[{lv=ToExpression[l],rv=ToExpression[r],tv=ToExpression[t],bv=ToExpression[b]},
  If[ TrueQ[lv === rv === tv === bv], 
    "Border" -> lv,
    "Border" -> {{lv, rv},{tv, bv}} ]
  ]
  
(* Single child assumed Widget *)
xmlLayoutElementsToGUI[ m:XMLElement[("border" | {_, "border"}), {}, {child_}] ] :=
  "Border" -> xmlToGUI[child]
(* Multiple children assumed compound Border *)
xmlLayoutElementsToGUI[ m:XMLElement[("border" | {_, "border"}), {}, {children__}] ] :=
  "Border" -> Last /@ (Flatten[{xmlLayoutElementsToGUI /@ {children}}])
  
xmlLayoutElementsToGUI[___] := {}

xmlToGUI[m:XMLElement[("fill" | {_, "fill"}), {}, {}]] :=
  WidgetFill[]

xmlToGUI[m:XMLElement[("align" | {_, "align"}), {}, {}]] :=
  WidgetAlign[]
  
xmlToGUI[m:XMLElement[("align" | {_, "align"}), {atts__}, {}]] :=
  Module[{ref, from, to},
   {ref, from, to} = extractAttributes[ {"ref", "from", "to"}, {atts}];
   If[ ref =!= None && from =!= None && to =!= None,
     from = If[ StringMatchQ[from, "Before"], Before, After];
     to = If[ StringMatchQ[to, "After"], After, Before];
     WidgetAlign[{ref, from}, to],
     (* TODO issue warning message and $Failed *)
     {}]
   ]
   
xmlToGUI[m:XMLElement[("space" | {_, "space"}), {atts__}, {}]] :=
  Module[{val},
   {val} = extractAttributes[ {"value"}, {atts}];
   If[ val =!= None, val = ToExpression[val], val = 0];
   If[ !IntegerQ[val], val = 0];
   WidgetSpace[val]
   ]
   
xmlToGUI[m:XMLElement[("true" | {_, "true"}), {atts___}, {children___}]] := True

xmlToGUI[m:XMLElement[("false" | {_, "false"}), {atts___}, {children___}]] := False
  
xmlToGUI[m:XMLElement[("property" | {_, "property"}), {atts___}, {children___}]] :=
  Module[{tgt,nm,inx,val,id, childContent, invokeThread, invokeWait},
  
    {tgt, nm, inx, val, id, invokeThread, invokeWait} = 
        extractAttributes[ {"target", "name", "index", "value", "id", "invokeThread", "invokeWait"}, {atts}];
   
    childContent = xmlToGUI /@ {children};
    
    nm = {
      If[tgt =!= None, tgt, Unevaluated[Sequence[]]  ],
      nm,
      If[inx =!= None, ToExpression[inx], Unevaluated[Sequence[]]  ]
      };
    If[ Length[nm] === 1, nm = First[nm]];
    
    If[ (val =!= None) || (Length[{children}] > 0),
       If[ id === None && invokeThread === None && invokeWait === None,
          nm -> If[ val =!= None, val, 
             If[ Length[childContent] === 1, First[childContent], childContent]],
          SetPropertyValue[nm, If[ val =!= None, val, 
             If[ Length[childContent] === 1, First[childContent], childContent]], 
              If[id =!= None, Name -> id, Unevaluated[Sequence[]]  ],
              If[invokeThread =!= None, InvokeThread -> invokeThread, Unevaluated[Sequence[]]  ],
              If[invokeWait =!= None, InvokeWait -> Which[
                  StringMatchQ[invokeWait, "true", IgnoreCase -> True] ||
                  StringMatchQ[invokeWait, "yes", IgnoreCase -> True], True,
                  StringMatchQ[invokeWait, "false", IgnoreCase -> True] ||
                  StringMatchQ[invokeWait, "no", IgnoreCase -> True], False,
                  True, Automatic ], Unevaluated[Sequence[]]  ]
              ] ],
       PropertyValue[nm, 
         If[id =!= None, Name -> id, Unevaluated[Sequence[]]  ],
         If[invokeThread =!= None, InvokeThread -> invokeThread, Unevaluated[Sequence[]]  ],
              If[invokeWait =!= None, InvokeWait -> Which[
                  StringMatchQ[invokeWait, "true", IgnoreCase -> True] ||
                  StringMatchQ[invokeWait, "yes", IgnoreCase -> True], True,
                  StringMatchQ[invokeWait, "false", IgnoreCase -> True] ||
                  StringMatchQ[invokeWait, "no", IgnoreCase -> True], False,
                  True, Automatic ], Unevaluated[Sequence[]]  ]
         ]
       ]
    ]
    
xmlToGUI[m:XMLElement[("bindevent" | {_, "bindevent"}), {atts___}, {children___}]] :=
  Module[{tgt, nm, flt, id, childContent, invokeThread, invokeWait, useArgs},
  
    {tgt,nm,flt, id, invokeThread, invokeWait} = 
      extractAttributes[ {"target", "name", "filter", "id", "invokeThread", "invokeWait"}, {atts}];
      
    childContent = xmlToGUI /@ {children};
       
    (* This should actually issue a message if child is not valid *)
    useArgs = {
     If[ tgt =!= None, 
       {tgt, nm, flt},
       If[ flt =!= None, {Automatic, nm, flt}, nm]
       ], 
      If[ Length[childContent] === 1, First[childContent], childContent],
      If[id =!= None, Name -> id, Unevaluated[Sequence[]]  ],
      If[invokeThread =!= None, InvokeThread -> invokeThread, Unevaluated[Sequence[]]  ],
      If[invokeWait =!= None, InvokeWait -> Which[
        StringMatchQ[invokeWait, "true", IgnoreCase -> True] ||
        StringMatchQ[invokeWait, "yes", IgnoreCase -> True], True,
        StringMatchQ[invokeWait, "false", IgnoreCase -> True] ||
        StringMatchQ[invokeWait, "no", IgnoreCase -> True], False,
           True, Automatic ], Unevaluated[Sequence[]]  ]
      };
      
    BindEvent @@ useArgs
    ]
    
xmlToGUI[m:XMLElement[("script" | {_, "script"}), {atts___}, {children___}]] :=
  Module[{lang, src, opts, childContent, isMathematicaScript = True},
    {lang, src} = extractAttributes[ {"language", "src"}, {atts}];
    
    childContent = xmlToGUI /@ {children};
    
    isMathematicaScript = If[ lang === None || 
      (StringQ[lang] && StringMatchQ[lang, "mathematica", IgnoreCase->True]),
         True, False];
    
    childContent = If[ MatchQ[childContent, {__String}],  
        (* Should we make a special wrap case for doing a SyntaxQ check or no? 
           Also will some people want this to stay as a String in Mathematica or always
           do a SyntaxQ on it if language is Mathematica?
        *)
        If[ TrueQ[isMathematicaScript],
          ToExpression[ StringJoin[childContent], InputForm, HoldComplete],
          {StringJoin[childContent]} ]
        , 
        If[ StringQ[lang] && (
        StringMatchQ[lang, "xml", IgnoreCase -> True] || StringMatchQ[lang, $GUIKitXMLFormat, IgnoreCase -> True]),
          childContent,
          If[ Length[childContent] > 0, {First[childContent]}, {Null}]
          ]
        ];
    opts = Script @@ {If[ TrueQ[isMathematicaScript],
              Unevaluated[Sequence[]], Language -> lang ],
            If[ src === None,  Unevaluated[Sequence[]], ScriptSource -> src]};
    Join[ Function[x, Script[x], {HoldAllComplete}] @@ childContent, opts]
    ]
    
xmlToGUI[m:XMLElement[("invokemethod" | {_, "invokemethod"}), {atts___}, {children___}]] :=
  Module[{tgt, nm, id, invokeThread, invokeWait},
    {tgt, nm, id, invokeThread, invokeWait} =
      extractAttributes[ {"target", "name", "id", "invokeThread", "invokeWait"}, {atts}];
    
    (InvokeMethod[ If[ tgt =!= None, {tgt, nm}, nm], ##,
      If[ id =!= None, Name -> id, Unevaluated[Sequence[]]  ],
      If[invokeThread =!= None, InvokeThread -> invokeThread, Unevaluated[Sequence[]]  ],
      If[invokeWait =!= None, InvokeWait -> Which[
        StringMatchQ[invokeWait, "true", IgnoreCase -> True] ||
        StringMatchQ[invokeWait, "yes", IgnoreCase -> True], True,
        StringMatchQ[invokeWait, "false", IgnoreCase -> True] ||
        StringMatchQ[invokeWait, "no", IgnoreCase -> True], False,
        True, Automatic ], Unevaluated[Sequence[]]  ]
      ]&) @@ (xmlToGUI /@ {children})
    ]
    
    
(* This rule is here only to handle "args" as child of Widget,
   since this returns a plain WidgetReference we may need
   to change how this is handled in general *)
xmlToGUI[m:XMLElement[("args" | {_, "args"}), {atts___}, {children___}]] :=
  Module[{ref, result = {}},
    {ref} = extractAttributes[ {"ref"}, {atts}];
    
    result = xmlToGUI /@ {children};
    
    If[ ref =!= None, 
      If[ Length[result] > 0,
        result = Join[{WidgetReference[ref]},result],
        result = WidgetReference[ref] ];
      ];
    result
  ]
  
xmlToGUI[m:XMLElement[("expose" | {_, "expose"}), {atts___}, {children___}]] :=
  Module[{as, ref},
  
    {ref, as} = extractAttributes[ {"ref", "as"}, {atts}];
   
    (* TODO issue warnings or errors if id att not found *)
    If[ as =!= None, ref -> as, ref]
    ]
    
    
xmlToGUI[m:XMLElement[("string" | {_, "string"}), {atts___}, {children___}]] :=
  Module[{content, val, id},
    {id, val} = extractAttributes[ {"id", "value"}, {atts}];
    (* Check if value attribute exists otherwise children combined as strings *)
    If[ val =!= None,
      content = val,
      (* TODO we may need to remove the Flatten call if children can be nested Widget groups *)
      content = StringJoin[ ToString /@ Flatten[{xmlToGUI /@ {children}}] ] 
      ];
    If[ id =!= None, 
      Widget["class:java.lang.String", InitialArguments -> {content}, Name -> id],
      content]
    ]

xmlToGUI[m:XMLElement[("integer" | {_, "integer"}), {atts___}, {children___}]] :=
  Module[{content, val, id},
    {id, val} = extractAttributes[ {"id", "value"}, {atts}];
    (* Check if value attribute exists otherwise children combined as strings *)
    If[ val =!= None,
      content = val,
      (* TODO we may need to remove the Flatten call if children can be nested Widget groups *)
      content = StringJoin[ ToString /@ Flatten[{xmlToGUI /@ {children}}] ]
      ];
    If[ id =!= None, 
      Widget["class:int", InitialArguments -> {content}, Name -> id],
      ToExpression[content]]
    ]

xmlToGUI[m:XMLElement[("double" | {_, "double"}), {atts___}, {children___}]] :=
  Module[{content, val, id},
    {id, val} = extractAttributes[ {"id", "value"}, {atts}];
    (* Check if value attribute exists otherwise children combined as strings *)
    If[ val =!= None,
      content = val,
      (* TODO we may need to remove the Flatten call if children can be nested Widget groups *)
      content = StringJoin[ ToString /@ Flatten[{xmlToGUI /@ {children}}] ] 
      ];
    If[ id =!= None, 
      Widget["class:double", InitialArguments -> {content}, Name -> id],
      ToExpression[content] ]
    ]
    
xmlToGUI[m:XMLElement[("null" | {_, "null"}), {atts___}, {children___}]] := Null

xmlToGUI[ XMLObject["Doctype"][name_, atts___] ] :=
  {}

xmlToGUI[ XMLObject["ProcessingInstruction"][target_, data_] ] :=
  {}

xmlToGUI[ XMLObject["Declaration"][atts___] ] :=
  {}
  
xmlToGUI[ XMLObject["CDATASection"][str_String] ] :=
  str
  
xmlToGUI[str_String] := str

xmlToGUI[x___] := {}
  
extractAttributes[l_List, atts_] := 
  Flatten[extractOne[#, atts]& /@ l]

extractOne[ findLHS_String, atts_] := 
  Module[{res},
  res = Cases[atts, HoldPattern[Rule[findLHS | {_, findLHS}, _]]];
  If[ Length[res] > 0,  res[[1,2]], None]
  ]
  
  
(* End not called in subpackages *)
(* EndPackage not called in subpackages *)

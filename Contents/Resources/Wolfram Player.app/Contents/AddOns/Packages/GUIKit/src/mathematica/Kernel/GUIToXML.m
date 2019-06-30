
(* GUIToXML subpackage *)

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
   as a module to Export[]
 *******************************)
ImportExport`RegisterFormat[$GUIKitXMLFormat];

If[ $VersionNumber >= 6.0,
	ImportExport`RegisterExport[$GUIKitXMLFormat,
		GUIKitXMLSave,
		ImportExport`BinaryFormat -> False,
		ImportExport`FunctionChannels -> {"FileNames", "Streams"}
	],
	ImportExport`RegisterExport[$GUIKitXMLFormat,
		GUIKitXMLSave,
		ImportExport`BinaryFormat -> False,
		ImportExport`FunctionChannels -> {FileNames, Streams}
	]	
];
  
GUIKitXMLSave[source_, obj_, opts___] := Module[{result},
  result = WidgetToSymbolicGUIKitXML[obj];
  If[ result === $Failed, Return[$Failed]];
  Export[ source, result, "XML", opts]
  ]

   
(*******************************
   WidgetToSymbolicGUIKitXML   

 *******************************)
 
WidgetToSymbolicGUIKitXML[ expr_?guiExpressionQ, opts___] :=
  Module[{rootElement},
  rootElement = guiToXMLContent[expr];
  If[ rootElement === $Failed, Return[$Failed]];
  XMLObject["Document"][{}, rootElement, {}]
  ]

(** Widget **)

(* Supports XML <widget>
 
 Forms: (new instances)
 
 Widget["class:className" | "file.*" | URL[]]
 Widget["class:className" | "file.*" | URL[], {content__}]
 
 Supports XML attribute src="" (or class="")
 
 Widget[
    "class:a.b.C" | "file" (implied extention) | "file.xml" | "file.m" | URL["http://..."],
    {content__}, 
    Name -> None | "id",
    InitialArguments -> None | WidgetReference["##"] | {} | {content__},
    ExposeWidgetReferences -> Automatic |
       {"id1", "id2", ...} | {"id1" -> "newid1", "id2" -> "newid2", ...}
    ]

 *)
 
guiToXMLContent[ Widget[source_?guiResourceQ, opts___?extendedOptionQ]] :=
  guiToXMLContent[ Widget[source, {}, opts]]
  
(* Special case check for string with id *)
guiToXMLContent[ w:Widget["class:java.lang.String", {}, opts___?extendedOptionQ] ] := 
  Module[{args, id, atts = {}, contentResults = {}},

  (* Needs to be done separate from others because of larger content *)
 args = Cases[Flatten[{opts}], HoldPattern[InitialArguments->_], 1, 1];
  If[ Length[args] < 1,
      args = Cases[Options[Widget], HoldPattern[InitialArguments->_], 1, 1];
      ];
  If[ Length[args] > 0, args = Last[First[args]], args = None];

  useOpts = canonicalOptions[Flatten[{opts}]];
  
  (Message[Widget::optx, #, HoldForm[w]]&) /@ Select[First /@ useOpts, 
    (!MemberQ[{"Name", ExposeWidgetReferences, WidgetLayout, InitialArguments}, #]&)];
    
  id = "Name" /. useOpts /. canonicalOptions[Options[Widget]];
  
  If[ id =!= None, atts = Join[atts, {"id" -> ToString[id]}] ];
  If[ args =!= None && MatchQ[args, {_String}], contentResults = args ];
    
  XMLElement["string", atts, contentResults]
  ]
  
guiToXMLContent[ Widget[src_, {contents___}, opts___?extendedOptionQ] ] :=
  guiToXMLContent[ Widget[src, WidgetGroup[{contents}, WidgetLayout -> Automatic], opts] ]
  
guiToXMLContent[ w:Widget[source_String | URL[source_String], contents_WidgetGroup, opts___?extendedOptionQ] ] := 
  Module[{useSource = ToString[source], args, id, exp, layout, layresult, atts = {},  contentResults = {},
    useOpts},
  (* ToString on source is used to remove any possible embedded stylebox linear syntax *)
  If[ StringMatchQ[useSource, "class:*"],
    atts = {"class" -> StringDrop[useSource, 6]},
    atts = {"src" -> useSource}
    ];
    
  (* Needs to be done separate from others because of larger content *)
  args = Cases[Flatten[{opts}], HoldPattern[InitialArguments->_], 1, 1];
  If[ Length[args] < 1,
      args = Cases[Options[Widget], HoldPattern[InitialArguments->_], 1, 1];
      ];
  If[ Length[args] > 0, args = Last[First[args]], args = None];
  
  useOpts = canonicalOptions[Flatten[{opts}]];
  
  (Message[Widget::optx, #, HoldForm[w]]&) /@ Select[First /@ useOpts, 
    (!MemberQ[{"Name", ExposeWidgetReferences, WidgetLayout, InitialArguments}, #]&)];
  
  {id, exp, layout} = {"Name", ExposeWidgetReferences, WidgetLayout} /. 
    useOpts /. canonicalOptions[Options[Widget]];
  
  If[ id =!= None, 
    atts = Join[atts, {"id" -> ToString[id]}] ];

  If[ Head[contents] === List,
    contentResults = Flatten[{guiToXMLContent /@ contents}],
    (* Root WidgetGroup needs attribute set *)
    contentResults = guiToXMLContent[contents];
    If[ contentResults === $Failed, Return[$Failed]];
    contentResults = ReplacePart[contentResults,  Join[contentResults[[2]], {"root" -> "true"}], 2];
    contentResults = {contentResults}
    ];
  
  If[ MemberQ[contentResults, $Failed], Return[ $Failed]];
  
  If[exp =!= Automatic,
    contentResults = Join[ Flatten[createExposeElement /@ 
      If[ Head[exp] === List, exp, {exp}]], contentResults] 
    ];
    
  If[ args =!= None,
    attContent = {};
    If[ MatchQ[args, WidgetReference[_String]],
      attContent = {"ref" -> ToString[First[args]]};
      args = {};
      ];
    argContent = Flatten[createArgsElement /@ 
      If[ TrueQ[Head[args] === List], args, {args} ] ];
    If[ MemberQ[argContent, $Failed], Return[$Failed]];
    contentResults = Join[ {XMLElement["args", attContent, argContent]}, contentResults] 
    ];

  layresult = createSimpleLayoutAttribute[layout];
  If[ layresult === $Failed,
     (* Not as a simple attribute so layout needs to be built as first element *)
     contentResults = Flatten[Prepend[contentResults, createLayoutElements[layout]]],
     atts = Join[atts, layresult];
     ];
     
  XMLElement["widget", atts, contentResults]
  ]
  
(* ToStrings on strings called to stip possible embedded linear syntax inside strings *)
createExposeElement[ (Rule | RuleDelayed)[str_String, id_String] ] := 
  XMLElement["expose", {"ref" -> ToString[str], "as" -> ToString[id]}, {}]
createExposeElement[ str_String] := 
  XMLElement["expose", {"ref" -> ToString[str]}, {}]
createExposeElement[x___] := 
  ( Message[Widget::optv, ExposeWidgetReferences, HoldForm[x]];
    {} )

createArgsElement[ expr_?guiExpressionQ] := guiToXMLContent[expr]
createArgsElement[x___] := 
  ( Message[Widget::optv, InitialArguments, HoldForm[x]];
    {} )
  
(** WidgetReference **)

(* lookup/call existing instance with ref="id" 

 WidgetReference["ref id"]]
 WidgetReference["class:name"]
 
 WidgetReference["ref id" | "ref id" -> "new id", {InvokeMethod[]...}]
 
 *)
 
guiToXMLContent[ WidgetReference[ id:(_String | (Rule | RuleDelayed)[_String, _String]), opts___?extendedOptionQ ] ] :=
  guiToXMLContent[ WidgetReference[id, {}, opts]]

guiToXMLContent[ w:WidgetReference[str_String, contents_List, opts___?extendedOptionQ] ] :=
  Block[{contentResult = Flatten[{guiToXMLContent /@  contents}], layout, layresult, 
      atts = {"ref" -> ToString[str]}, useOpts},
    If[ MemberQ[contentResult, $Failed], Return[$Failed]];
    
    useOpts = canonicalOptions[Flatten[{opts}]];
    (Message[WidgetReference::optx, #, HoldForm[w]]&) /@ Select[First /@ useOpts, 
      (!MemberQ[{WidgetLayout}, #]&)];
    
    layout = WidgetLayout /. useOpts /. Options[WidgetReference];
    layresult = createSimpleLayoutAttribute[layout];
    If[ layresult === $Failed,
       (* Not as a simple attribute so layout needs to be built as first element *)
       contentResult = Flatten[Prepend[contentResult, createLayoutElements[layout]]],
       atts = Join[atts, layresult];
       ];
    XMLElement["widget", atts, contentResult ]
   ]
   
guiToXMLContent[ w:WidgetReference[(Rule | RuleDelayed)[str_String, newid_String], contents_List, opts___?extendedOptionQ] ] :=
  Block[{contentResult = Flatten[{guiToXMLContent /@  contents}], layout, layresult, 
      atts ={"ref" -> ToString[str], "id" -> ToString[newid]}, useOpts},  
    If[ MemberQ[contentResult, $Failed], Return[$Failed]];
    
    useOpts = canonicalOptions[Flatten[{opts}]];
    (Message[WidgetReference::optx, #, HoldForm[w]]&) /@ Select[First /@ useOpts, 
      (!MemberQ[{WidgetLayout}, #]&)];
      
    layout = WidgetLayout /. useOpts /. Options[WidgetReference];
    layresult = createSimpleLayoutAttribute[layout];
    If[ layresult === $Failed,
      (* Not as a simple attribute so layout needs to be built as first element *)
      contentResult = Flatten[Prepend[contentResult, createLayoutElements[layout]]],
     atts = Join[atts, layresult];
     ];
    XMLElement["widget", atts, contentResult ]
   ]

      
(** WidgetGroup **)

(*
 NOTE: Lists are implied WidgetGroups in expression version
 
 WidgetGroup[{___}, WidgetLayout -> _]
 *)
 
 guiToXMLContent[ contents_List] :=
   guiToXMLContent[ WidgetGroup[contents]]
  
 guiToXMLContent[ w:WidgetGroup[ contents_List, opts___?extendedOptionQ] ] :=
   Block[{id, contentResult = Flatten[{guiToXMLContent /@  contents}], atts = {}, 
        layout, layresult, useOpts},
     If[ MemberQ[contentResult, $Failed], Return[$Failed]];
     
     useOpts = canonicalOptions[Flatten[{opts}]];
     (Message[WidgetGroup::optx, #, HoldForm[w]]&) /@ Select[First /@ useOpts, 
      (!MemberQ[{"Name", WidgetLayout}, #]&)];
      
     id = "Name" /. useOpts /. canonicalOptions[Options[WidgetGroup]];
     If[ id =!= None, atts = Join[atts, {"id" -> ToString[id]}] ];
     layout = WidgetLayout /. useOpts /. Options[WidgetGroup];
     layresult = createSimpleLayoutAttribute[layout];
     If[ layresult === $Failed,
       (* Not as a simple attribute so layout needs to be built as first element *)
       contentResult = Flatten[Prepend[contentResult, createLayoutElements[layout]]],
        atts = Join[atts, layresult];
       ];
     XMLElement["group", atts, contentResult]
   ]
 
 

(** WidgetLayout tools **)
 
guiToXMLContent[ WidgetSpace[] ] := guiToXMLContent[ WidgetSpace[0] ]
guiToXMLContent[ WidgetSpace[ n_Integer] ] :=
  XMLElement["space", {"value" -> ToString[n]}, {}]

guiToXMLContent[ WidgetFill[___] ] :=
  XMLElement["fill", {}, {}]
  
guiToXMLContent[ WidgetAlign[] ] :=
  XMLElement["align", {}, {}]
guiToXMLContent[ WidgetAlign[{ref_String, loc:(Before | After)}, to:(Before | After)] ] :=
  XMLElement["align", {"ref" -> ToString[ref], "from" -> ToString[loc], "to" -> ToString[to]}, {}]

  
 (* Only if these forms pattern match can a WidgetLayout rule become just
    an XML attribute and not a required XML element set
  *)
createSimpleLayoutAttribute[Automatic] := {}
createSimpleLayoutAttribute[n:(None | Column | Row | Grid | Split)] := {"layout" -> ToString[n]}
createSimpleLayoutAttribute[{"Grouping" -> val_}] := createSimpleLayoutAttribute[val]
createSimpleLayoutAttribute[___] := $Failed

(* TODO this needs to be extended to any guiExpressionQ that could possibly
   return a layout Java object *)
createLayoutElements[ c:(_Widget | _WidgetReference)] := 
  XMLElement["layout", {},  Flatten[{createSingleLayoutElement["Grouping" -> c]}] ]

createLayoutElements[l_List] :=
  XMLElement["layout", {}, Flatten[{createSingleLayoutElement /@ l}] ]
     
(* This should generate a message and error *)
createLayoutElements[x___] := 
  ( Message[Widget::optv, WidgetLayout, HoldForm[x]];
    {} )


createSingleLayoutElement["Grouping" -> Automatic] := {}

createSingleLayoutElement["Grouping" -> n:(None | Column | Row | Grid | Split)] :=
  XMLElement["grouping", {"type" -> ToString[n]}, {}]
 
(* Here are the patterns that are accepted for using tabbed panes *)

createSingleLayoutElement["Grouping" -> {Tabs, {s__String}}] :=
  createSingleLayoutElement["Grouping" -> {Tabs, Automatic, {s}}]
  
createSingleLayoutElement["Grouping" -> {Tabs, n_, {s__String}}] :=
  XMLElement["grouping", {"type" -> "Tabs"}, {
    XMLElement["tabs", {"orient" -> 
      Switch[n,
        Left, "Left",
        Right, "Right",
        Bottom, "Bottom",
        _, "Top"]}, 
      guiToXMLContent /@ {s}
      ]
    }]


(* Here are the patterns that are accepted for Split panes *)
  
createSingleLayoutElement["Grouping" -> {Split, orient_}] :=
  XMLElement["grouping", {"type" -> "Split"}, {
    XMLElement["split", {"orient" -> 
      Switch[orient,
        Horizontal, "Horizontal",
        _, "Vertical"]}, {}]
    }]


(* Here one is hopefully specifying a Java layout class *)
createSingleLayoutElement["Grouping" -> c:(_Widget | _WidgetReference)] :=
  XMLElement["grouping", {}, Flatten[{guiToXMLContent[c]}] ]

createSingleLayoutElement["Alignment" -> Automatic] := {}
createSingleLayoutElement["Alignment" -> {Automatic, Automatic}] := {}
createSingleLayoutElement["Alignment" -> n:(Left | Center | Right | Top | Bottom)] := 
  XMLElement["alignment", {"type" -> ToString[n]}, {}]
createSingleLayoutElement["Alignment" -> {n:(Automatic | Left | Center | Right), m:(Automatic | Center | Top | Bottom)}] :=
  XMLElement["alignment", {"type" -> ToString[n] <> "," <> ToString[m]}, {}]


createSingleLayoutElement["Stretching" -> Automatic] := {}
createSingleLayoutElement["Stretching" -> {Automatic, Automatic}] := {}
(* If both are supplied one cannot mix with Automatic *)
createSingleLayoutElement["Stretching" -> {n:(None | False | WidgetAlign | True | Maximize ), 
    m:(None | False | WidgetAlign | True | Maximize )}] :=
  XMLElement["stretching", {"type" -> ToString[n] <> "," <> ToString[m]}, {}]
  
createSingleLayoutElement["Spacing" -> Automatic] := {}
createSingleLayoutElement["Spacing" -> n_Integer] := 
  XMLElement["spacing", {"value" -> ToString[n]}, {}]


createSingleLayoutElement["Border" -> (Automatic | None)] := {}

createSingleLayoutElement["Border" -> title_String] := 
  XMLElement["border", {"title" -> ToString[title]}, {}]
  
createSingleLayoutElement["Border" -> n_Integer] := 
  XMLElement["border", {"left" -> ToString[n],"right" -> ToString[n],"top" -> ToString[n],"bottom" -> ToString[n]}, {}]
  
createSingleLayoutElement["Border" -> {{l_Integer, r_Integer},{t_Integer,b_Integer}}] := 
  XMLElement["border", {"left" -> ToString[l],"right" -> ToString[r],"top" -> ToString[t],"bottom" -> ToString[b]}, {}]
  
createSingleLayoutElement["Border" ->c:(_Widget | _WidgetReference)] :=
  XMLElement["border", {}, Flatten[{ guiToXMLContent[c]}] ]
  
createSingleLayoutElement["Border" -> {c:(_RGBColor | _Hue | _GrayLevel | _CMYKColor), n_Integer}] := 
  XMLElement["border", {}, 
   Flatten[{ guiToXMLContent[ 
      InvokeMethod[{"class:javax.swing.BorderFactory", "createLineBorder"}, 
        Widget["Color", {}, InitialArguments -> ((Floor[255 #]&) /@ (List @@ ToColor[c,RGBColor])) ], n ]
      ] }] ]
  
createSingleLayoutElement["Border" -> {c:(_RGBColor | _Hue | _GrayLevel | _CMYKColor), 
    {{l_Integer, r_Integer},{t_Integer,b_Integer}}}] := 
  XMLElement["border", {}, 
   Flatten[{ guiToXMLContent[ 
      InvokeMethod[{"class:javax.swing.BorderFactory", "createMatteBorder"}, t, l, b, r,
        Widget["Color", {}, InitialArguments -> ((Floor[255 #]&) /@ (List @@ ToColor[c,RGBColor])) ] ]
      ] }] ]
      
createSingleLayoutElement["Border" -> l_List] := 
  XMLElement["border", {}, Flatten[{ createSingleLayoutElement["Border" -> #]& /@ l }]]
  
  
createSingleLayoutElement[x___] :=   
  ( Message[Widget::optv, WidgetLayout, HoldForm[x]];
    {} )


(** PropertyValue **)

(* Supports XML <property> 

 PropertyValue["name", 
    Name -> "id" | None
   ]
   
 *)
Off[PropertyValue::pvobj,PropertyValue::poi];
 
guiToXMLContent[ PropertyValue[name_String, opts___?extendedOptionQ] ] :=
  guiToXMLContent[ PropertyValue[{Automatic, name, All}, opts]]

guiToXMLContent[ PropertyValue[{name_String}, opts___?extendedOptionQ] ] :=
  guiToXMLContent[ PropertyValue[{Automatic, name, All}, opts]]
  
guiToXMLContent[ PropertyValue[{tgt_String, name_String}, opts___?extendedOptionQ] ] :=
  guiToXMLContent[ PropertyValue[{tgt, name, All}, opts]]
  
guiToXMLContent[ PropertyValue[{name_String, pt_Integer}, opts___?extendedOptionQ] ] :=
  guiToXMLContent[ PropertyValue[{Automatic, name, pt}, opts]]
  
guiToXMLContent[ w:PropertyValue[{tgt_, name_String, pt_}, opts___?extendedOptionQ] ] :=
  Module[{id, atts = {"name" -> ToString[name]}, contentResults={}, invokeThread, invokeWait,
    useOpts},
  
  useOpts = canonicalOptions[Flatten[{opts}]];
  (Message[PropertyValue::optx, #, HoldForm[w]]&) /@ Select[First /@ useOpts, 
      (!MemberQ[{"Name", InvokeThread, InvokeWait}, #]&)];
      
  {id, invokeThread, invokeWait} = {"Name", InvokeThread, InvokeWait} /. 
    useOpts /. canonicalOptions[Options[PropertyValue]];
  
  If[ pt =!= All && IntegerQ[pt], atts = Join[atts, {"index" -> ToString[pt, InputForm]}] ];
  If[ id =!= None, atts = Join[atts, {"id" -> ToString[id]}] ];
  If[ tgt =!= Automatic && StringQ[tgt], atts = Join[atts, {"target" -> ToString[tgt]}] ];
  If[ invokeThread =!= "Current", atts = Join[atts, {"invokeThread" -> ToString[invokeThread]}] ];
  If[ invokeWait =!= Automatic, atts = Join[atts, {"invokeWait" -> ToString[invokeWait]}] ];
  
  XMLElement["property", atts, contentResults]
  ]

On[PropertyValue::pvobj,PropertyValue::poi]
(** SetPropertyValue **)
  
(* Supports XML <property> setter 
  Shorthand notation for just name and value is a Rule
 
  "name" -> "value" | valueContent_
  
  SetPropertyValue[
     "name", 
     "value" | valueContent_,
     Name -> "id" | None
   ]
   
*)

guiToXMLContent[ (Rule | RuleDelayed)[name_String, val_] ] :=
  guiToXMLContent[ SetPropertyValue[name, val]]
  
guiToXMLContent[ (Rule | RuleDelayed)[{name_String}, val_] ] :=
  guiToXMLContent[ SetPropertyValue[name, val]]
  
guiToXMLContent[ (Rule | RuleDelayed)[{tgt_String, name_String}, val_] ] :=
  guiToXMLContent[ SetPropertyValue[{tgt, name}, val]]
  
guiToXMLContent[ (Rule | RuleDelayed)[{name_String, pt_Integer}, val_] ] :=
  guiToXMLContent[ SetPropertyValue[{name, pt}, val]]
  
guiToXMLContent[ (Rule | RuleDelayed)[{tgt_String, name_String, pt_Integer}, val_] ] :=
  guiToXMLContent[ SetPropertyValue[{tgt, name, pt}, val]]
  
guiToXMLContent[ SetPropertyValue[name_String, val_, opts___?extendedOptionQ] ] :=
  guiToXMLContent[ SetPropertyValue[{Automatic, name, All}, val, opts] ]
  
guiToXMLContent[ SetPropertyValue[{name_String}, val_, opts___?extendedOptionQ] ] :=
  guiToXMLContent[ SetPropertyValue[{Automatic, name, All}, val, opts] ]
  
guiToXMLContent[ SetPropertyValue[{tgt_String, name_String}, val_, opts___?extendedOptionQ] ] :=
  guiToXMLContent[ SetPropertyValue[{tgt, name, All}, val, opts] ]
  
guiToXMLContent[ SetPropertyValue[{name_String, pt_Integer}, val_, opts___?extendedOptionQ] ] :=
  guiToXMLContent[ SetPropertyValue[{Automatic, name, pt}, val, opts] ]
  
guiToXMLContent[ w:SetPropertyValue[{tgt_, name_String, pt_}, val_, opts___?extendedOptionQ] ] :=
  Module[{id, atts = {"name" -> ToString[name]}, contentResults={}, invokeThread, invokeWait,
    useOpts},
  
  useOpts = canonicalOptions[Flatten[{opts}]];
  (Message[SetPropertyValue::optx, #, HoldForm[w]]&) /@ Select[First /@ useOpts, 
      (!MemberQ[{"Name", InvokeThread, InvokeWait}, #]&)];
      
  {id, invokeThread, invokeWait} = {"Name", InvokeThread, InvokeWait} /. 
    useOpts /. canonicalOptions[Options[SetPropertyValue]];
  
  If[ pt =!= All && IntegerQ[pt], atts = Join[atts, {"index" -> ToString[pt, InputForm]}] ];
  If[ id =!= None, atts = Join[atts, {"id" -> ToString[id]}] ];
  If[ tgt =!= Automatic && StringQ[tgt], atts = Join[atts, {"target" -> ToString[tgt]}] ];
  If[ invokeThread =!= "Current", atts = Join[atts, {"invokeThread" -> ToString[invokeThread]}] ];
  If[ invokeWait =!= Automatic, atts = Join[atts, {"invokeWait" -> ToString[invokeWait]}] ];
  
  If[ StringQ[val],
    atts = Join[atts, {"value" -> val}],
    If[ val =!= None (* Off as we want to see errors *)(* && guiExpressionQ[val] *), 
      contentResults = Flatten[{guiToXMLContent[val]}];
      If[ MemberQ[contentResults, $Failed], Return[$Failed]];
      ];
    ];
  XMLElement["property", atts, contentResults]
  ]
 

(** BindEvent **)

(* Supports XML <bindevent> 

 In order for us to move "target" to part of first
 argument we would have to assume two arg list is name,filter
 and require at least "filter" -> None for use with a target
 
 BindEvent[
    "name" | {"target", "name"} | {"target", "name", "filter"},
    content__
    ]
    
 *)
 
guiToXMLContent[ BindEvent[ name_String, others___]] :=
  guiToXMLContent[ BindEvent[ {Automatic, name, None}, others]]
  
guiToXMLContent[ BindEvent[ {tgt_, name_String}, others___]] :=
  guiToXMLContent[ BindEvent[ {tgt, name, None}, others]]
  
guiToXMLContent[ w:BindEvent[ {tgt_, name_String, filter_}, {content__?(!invokeOrBindOptionQ[#]&)}, opts___?invokeOrBindOptionQ] ] :=
  Module[{id, atts = {"name" -> ToString[name]}, contentResults={}, invokeThread, invokeWait,
    useOpts},
    
  useOpts = canonicalOptions[Flatten[{opts}]];
  (Message[BindEvent::optx, #, HoldForm[w]]&) /@ Select[First /@ useOpts, 
      (!MemberQ[{"Name", InvokeThread, InvokeWait}, #]&)];
      
  {id, invokeThread, invokeWait} = {"Name", InvokeThread, InvokeWait} /. 
    useOpts /. canonicalOptions[Options[BindEvent]];
    
  If[ id =!= None, atts = Join[atts, {"id" -> ToString[id]}] ];
  If[ StringQ[filter] && !StringMatchQ[filter, ""], atts = Join[atts, {"filter" -> ToString[filter]}] ];
  If[ tgt =!= Automatic && StringQ[tgt], atts = Join[atts, {"target" -> ToString[tgt]}] ];
  If[ invokeThread =!= "Current", atts = Join[atts, {"invokeThread" -> ToString[invokeThread]}] ];
  If[ invokeWait =!= Automatic, atts = Join[atts, {"invokeWait" -> ToString[invokeWait]}] ];
  
  contentResults = Flatten[{guiToXMLContent /@ {content}}];
  If[ MemberQ[contentResults, $Failed], Return[$Failed]];
  XMLElement["bindevent", atts, contentResults]
  ]
  
guiToXMLContent[ BindEvent[ {tgt_, name_String, filter_}, content__?(!invokeOrBindOptionQ[#]&), opts___?invokeOrBindOptionQ] ] :=
  guiToXMLContent[ BindEvent[ {tgt, name, filter}, {content}, opts]]
  
  
(** Script **)

 (* Supports XML <script> 
 
 Script[
    HoldAllComplete[code] | "script string contents" | InvokeMethod[]... | XMLDocument[],
    Language -> Automatic |  "mathematica" | "JavaScript",
    ScriptSource -> None | "file.m" | "file.xml"
    ]
    
 *)
 
guiToXMLContent[ Script[]] :=
   guiToXMLContent[Script[Null]]
   
guiToXMLContent[ w:Script[ content_, opts___?extendedOptionQ]] :=
  Module[{atts = {}, src = None, useLang, contentResults = {}, useOpts},
  
  useOpts = canonicalOptions[Flatten[{opts}]];
  (Message[Script::optx, #, HoldForm[w]]&) /@ Select[First /@ useOpts, 
      (!MemberQ[{Language, ScriptSource}, #]&)];
      
  {useLang, src} = {Language, ScriptSource} /. useOpts /. Options[Script];
  
  If[ useLang === Automatic || !StringQ[useLang], useLang = "mathematica"];
  
  If[ StringMatchQ[useLang, "xml", IgnoreCase -> True] ||
      StringMatchQ[useLang, $GUIKitXMLFormat, IgnoreCase -> True],
    If[ !StringQ[content]  && (guiExpressionQ[content] || MatchQ[content, {__?guiExpressionQ}]), 
        contentResults = Flatten[{guiToXMLContent[content]}];
        If[ MemberQ[contentResults, $Failed], Return[$Failed]];
        ,
        If[ MatchQ[content, XMLElement[__]],
          contentResults = {content},
          contentResults = {XMLObject["CDATASection"][convertToScriptString[content]]}
          ];
      ],
    contentResults = {XMLObject["CDATASection"][convertToScriptString[content]]}
    ];
    
  If[ !StringMatchQ[ useLang, "mathematica", IgnoreCase -> True],
    atts = Join[atts, {"language" -> ToString[useLang]}] ];
   
  If[ src =!= None, atts = Join[atts, {"src" -> ToString[src]}] ];
  
  If[ MemberQ[contentResults, $Failed], Return[$Failed]];
  XMLElement["script", atts, contentResults]
  ]
  


(** InvokeMethod **)

(* Supports XML <invokemethod>
 
 InvokeMethod[
    "name", 
    {} | content__,
    Name -> "id" | None
   ]
   
 *)
  
guiToXMLContent[ InvokeMethod[name_String, args___] ] :=
   guiToXMLContent[ InvokeMethod[{Automatic, name}, args]]
   
guiToXMLContent[ w:InvokeMethod[{tgt_, name_String}, contents___?(!invokeOrBindOptionQ[#]&), opts___?invokeOrBindOptionQ] ] :=
  Module[{id, atts = {"name" -> ToString[name]}, contentResults={}, invokeThread, invokeWait,
    useOpts},
  
  useOpts = canonicalOptions[Flatten[{opts}]];
  (Message[InvokeMethod::optx, #, HoldForm[w]]&) /@ Select[First /@ useOpts, 
      (!MemberQ[{"Name", InvokeThread, InvokeWait}, #]&)];
      
  {id, invokeThread, invokeWait} = {"Name", InvokeThread, InvokeWait} /. 
    useOpts /. canonicalOptions[Options[InvokeMethod]];
  
  If[ id =!= None, atts = Join[atts, {"id" -> ToString[id]}] ];
  If[ tgt =!= Automatic && StringQ[tgt], atts = Join[atts, {"target" -> ToString[tgt]}] ];
  If[ invokeThread =!= "Current", atts = Join[atts, {"invokeThread" -> ToString[invokeThread]}] ];
  If[ invokeWait =!= Automatic, atts = Join[atts, {"invokeWait" -> ToString[invokeWait]}] ];
  
  contentResults = Flatten[{guiToXMLContent /@ {contents} }];

  If[ MemberQ[contentResults, $Failed], Return[$Failed]];
  XMLElement["invokemethod", atts, contentResults]
  ]
  

(** String **)

(* Supports XML <string>
 
 If id= attribute is present we need to use a form of Widget[] wrapped
    to preserve the id setting
    
 "string"
  
 *)
(* We do not call ToString[] on this string as elsewhere since linear syntax embedded may need to stay
   since this would not be called from name attributes *)
guiToXMLContent[ str_String] := XMLElement["string", {}, {str}]

(** True | False **)

guiToXMLContent[ True] := 
  XMLElement["true", {}, {}]

guiToXMLContent[ False] := 
  XMLElement["false", {}, {}]
  
guiToXMLContent[ Null ] :=
   XMLElement["null", {}, {}]
   
(** Symbol **)
(* Decide whether this should be here, as it will not roundtrip back as a
   symbol but only as a string. Perhaps still useful as a one way trip?
 *)
(* No this is not useful in general as it causes JavaObjects to turn into
  Strings and until we handle references expicitly they don't work right 
  in general
*)
(*
guiToXMLContent[ s_Symbol] := 
  guiToXMLContent[ ToString[s, InputForm]]
*)

 (* 
 We choose to associate simple form with "int" and "double" primitive
 types instead of the Widgets because they are more common in UI widgets.
 Although there are areas where Integer and int are synonymous,
 for some InvokeMethod methods, the signature of argument Class types currently
   must be unique and cannot interchange Integer and int
 *)
    
(** int **)

guiToXMLContent[ i_Integer] := 
   Module[{},
   If[ Abs[i] <= 2147483647, 
      XMLElement["integer", {"value" -> ToString[i, InputForm]}, {}]
      ,
      XMLElement["widget", {"class"->"java.math.BigInteger"}, {
        XMLElement["args", {}, {XMLElement["string", {}, {ToString[i, InputForm]}] }] }]
      ]
   ]
   
(** double **)

guiToXMLContent[ r_Real] := 
   Module[{},
   If[ Abs[r] <= 1.7976931348623157*^308,
      XMLElement["double", {"value" -> ToString[r, CForm]}, {}]
      ,
      XMLElement["widget", {"class"->"java.math.BigDecimal"}, {
        XMLElement["args", {}, {XMLElement["string", {}, {ToString[r, CForm]}] }] }]
      ]
   ]
   
(* Last catchall pattern *)

(* Need to consider if we issue a top-level message here? 
   Perhaps one associated with $GUIKitXMLFormat and runtime calls would
   turn this off with their use of the XML
*)
guiToXMLContent[x___] := 
  ( Message[Widget::deferr, HoldForm[x]];
    $Failed )


(* End not called in subpackages *)
(* EndPackage not called in subpackages *)

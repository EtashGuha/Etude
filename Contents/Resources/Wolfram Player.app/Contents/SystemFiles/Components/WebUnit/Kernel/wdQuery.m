
(*valid arg for LocateElements
 args = {XPath->//*[@id="_nav-search"]/a/i[2]}*)

locatorToStringAssoc = <|
  "XPath"->XPath,
  "CSSSelector"->CssSelector,
  "Id"-> Id,
  "HyperlinkText" -> LinkText,
  "PartialHyperlinkText" -> PartialLinkText,
  "Tag" -> TagName,
  "ElementClassName" -> ElementClassName,
  "Name" -> Name
|>;


queryMethodGenerator[arg_?ListQ]:= Block[{key,val,len,query, webElems},
  (* If we were passed a list of WebElementObjects, bypass them. *)
  webElems = Select[arg, MatchQ[#, _WebElementObject]&];
  If[webElems =!= {}, Return[webElems]];

  len = Length[arg];
  Which[
    (len === 1 && MatchQ[arg[[1]], _Rule]),
      key = arg[[1,1]];
      val = arg[[1,2]],
    len === 2,
      key = arg[[1]];
      val = arg[[2]],
    True,
      Return[$Failed]
  ];
  query = locatorToStringAssoc[key];
  If[MissingQ[query],
    Message[WebExecute::locate, key];
    $Failed
    ,
    query[val]
  ]
];

queryMethodGenerator [arg_Rule]:= 
Block[{key,val,locatorToStringAssoc},
  key = arg[[1]];
  val = arg[[2]];
  query = locatorToStringAssoc[key];
  If[MissingQ[query],
    Message[WebExecute::locate, key];
    $Failed
    ,
    query[val]
  ]
];

(* Let WebElementObjects bypass query *)
queryMethodGenerator[arg_WebElementObject]:= arg;
queryMethodGenerator[arg_String]:= arg;

queryMethodGenerator [arg___]:= $Failed;


QueryMethod[sessionInfo_, ElementClassName[_String] ] ^:= "class name";
QueryMethod[sessionInfo_, CssSelector[_String] ] ^:= "css selector";
QueryMethod[sessionInfo_, Id[_String] ] ^:= "id";
QueryMethod[sessionInfo_, Name[_String] ] ^:= "name";
QueryMethod[sessionInfo_, LinkText[_String] ] ^:= "link text";
QueryMethod[sessionInfo_, PartialLinkText[_String] ] ^:= "partial link text";
QueryMethod[sessionInfo_, TagName[_String] ] ^:= "tag name";
QueryMethod[sessionInfo_, XPath[_String] ] ^:= "xpath";
QueryMethod[args___] := $Failed;


QueryValue[sessionInfo_, ElementClassName[s_String] ] ^:= s;
QueryValue[sessionInfo_, CssSelector[s_String] ] ^:= s;
QueryValue[sessionInfo_, XPath[s_String] ] ^:= s;
QueryValue[sessionInfo_, Name[s_String] ] ^:= s;
QueryValue[sessionInfo_, LinkText[s_String] ] ^:= s;
QueryValue[sessionInfo_, PartialLinkText[s_String] ] ^:= s;
QueryValue[sessionInfo_, TagName[s_String] ]  ^:= s;
(*comments*)
QueryValue[sessionInfo_, Id[s_String] ] /; sessionInfo[[2]] != "Firefox" ^:= s;
QueryValue[sessionInfo_, Id[s_String] ] /; sessionInfo[[2]] == "Firefox" := (Message[Id::nnarg, s];  $Failed);
QueryValue[args___] := $Failed;
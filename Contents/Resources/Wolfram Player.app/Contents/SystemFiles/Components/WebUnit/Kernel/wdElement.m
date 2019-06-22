(* ::Section:: *)
(*element state functions*)


LocateElement[valueId_] := LocateElement[$CurrentWebSession, valueId];
LocateElement[sessionInfo_, valueId_] := Module[{result, method, value, elems},
(* Bypass webelementobjects *)
  If[ListQ[valueId],
    elems = Select[valueId, MatchQ[#,_WebElementObject]&];
    If[elems =!= {},
      elems = elems[[1,1]]["ElementId"];
      Return[elems];
    ]
  ];
  If[MatchQ[valueId,WebElementObject[_Association]], Return[valueId[[1]]["ElementId"]]];

  method = QueryMethod[sessionI, valudId];
  value = QueryValue[sessionI, valudId];
  If[!StringQ[method] || !StringQ[value], Return[$Failed]];
  result = Catch[element[sessionInfo, {"using" -> method, "value" -> value}]];
  Which[
    result === $Failed, Null,
    result === "ELEMENT", result = {},
    sessionInfo[[2]] == "Firefox" && result =!= $Failed, result = result[[1]][[2]],
    sessionInfo[[2]] == "Chrome", result = "ELEMENT" /. result[[1]],
    sessionInfo[[2]] == "InternetExplorer", result = "ELEMENT" /. result[[1]],
    sessionInfo[[2]] == "Edge", result = "ELEMENT" /. result[[2]]
  ];
  result
];


LocateElements[valueId_] := LocateElements[$CurrentWebSession, valueId];
LocateElements[sessionInfo_, valueId_] := Module[{result, method, value, elems},
  (* Bypass webelementobjects *)
  If[ListQ[valueId],
    elems = Select[valueId, MatchQ[#,WebElementObject[_Association]]&];
    If[elems =!= {},
      elems = (#[[1]]["ElementId"])&/@elems;
      Return[elems]
    ]
  ];
  If[MatchQ[valueId,WebElementObject[_Association]],Return[valueId[[1]]["ElementId"]]];

  method = QueryMethod[sessionInfo, valueId];
  value = QueryValue[sessionInfo, valueId];
  If[!StringQ[method] || !StringQ[value], Return[$Failed]];
  result = Catch[elements[sessionInfo, {"using" -> method, "value" -> value}]];
  Which[
    result === $Failed, Null,
    result === "ELEMENT", result = {},
    sessionInfo[[2]] == "Firefox" && result =!= $Failed, result = result[[#]][[1]][[2]] & /@ Range[Length[result]],
    sessionInfo[[2]] == "Chrome", result = result[[#]][[1]][[2]] & /@ Range[Length[result]],
    sessionInfo[[2]] == "InternetExplorer", result = "ELEMENT" /. result[[1]],
    sessionInfo[[2]] == "Edge" && result =!= $Failed, result = result[[#]][[1]][[2]] & /@ Range[Length[result]]
  ];
  result
];


ClickElement[valueId_] := ClickElement[$CurrentWebSession, valueId];
ClickElement[sessionInfo_, valueId_] := Module[ {elementId},
  elementId = If[ StringQ[valueId], valueId, LocateElements[sessionInfo, valueId]];
  Which[
    elementId === {}, Message[WebExecute::noelem],
    StringQ[elementId], click[sessionInfo, elementId],
    ListQ[elementId], click[sessionInfo, #]&/@elementId
  ]
];


TypeElement[valueId_, text_] := TypeElement[$CurrentWebSession, valueId, text];
TypeElement[sessionInfo_, valueId_, text_] := Module[ {elementId},
  elementId = If[ StringQ[valueId], valueId, LocateElements[sessionInfo, valueId]];
  Which[
    elementId === {}, Message[WebExecute::noelem],
    StringQ[elementId],
      clear[sessionInfo, elementId];
      value[sessionInfo, elementId, text],
    ListQ[elementId], (
        clear[sessionInfo, elementId];
        value[sessionInfo, elementId, text];
      )&/@elementId
  ]
];


HoverElement[valueId_] := HoverElement[$CurrentWebSession, valueId];
HoverElement[sessionInfo_, valueId_] /; sessionInfo[[2]] == "Firefox" := (Message[WebExecute::ffhover, s];  Return[$Failed]);

HoverElement[sessionInfo_, valueId_] /; sessionInfo[[2]] != "Firefox" := Module[ {elementId},
  elementId = If[StringQ[valueId], valueId, LocateElements[sessionInfo, valueId]];
  Which[
    elementId === {}, Message[WebExecute::noelem],
    StringQ[elementId], moveto[sessionInfo, elementId],
    ListQ[elementId], moveto[sessionInfo, #]&/@elementId
  ]
];

HideElement[valueId_] := HideElement[$CurrentWebSession, valueId];
HideElement::nnarg = "Not supported in this Web-browser.";
HideElement[sessionInfo_, valueId_] /; sessionInfo[[2]] == "Firefox" := (Message[HideElement::nnarg, s];  Return[$Failed]);

HideElement[sessionInfo_, valueId_] /; sessionInfo[[2]] != "Firefox" := Module[ {elementId},
  elementId = If[StringQ[valueId], valueId, LocateElements[sessionInfo, valueId]];
  Which[
    elementId === {}, Message[WebExecute::noelem],
    StringQ[elementId], execute[sessionInfo, "arguments[0].style.visibility='hidden'", {{"ELEMENT" -> elementId}}],
    ListQ[elementId], execute[sessionInfo, "arguments[0].style.visibility='hidden'", {{"ELEMENT" -> #}}]&/@elementId
  ]
];

ShowElement[valueId_] := ShowElement[$CurrentWebSession, valueId]
ShowElement::nnarg = "Not supported in this Web-browser.";
ShowElement[sessionInfo_, valueId_] /; sessionInfo[[2]] == "Firefox" := (Message[ShowElement::nnarg, s];  Return[$Failed]);

ShowElement[sessionInfo_, valueId_] /; sessionInfo[[2]] != "Firefox" := Module[ {elementId},
  elementId = If[StringQ[valueId], valueId, LocateElements[sessionInfo, valueId]];
  Which[
    elementId === {}, Message[WebExecute::noelem],
    StringQ[elementId], execute[sessionInfo, "arguments[0].style.visibility='visible'", {{"ELEMENT" -> elementId}}],
    ListQ[elementId], execute[sessionInfo, "arguments[0].style.visibility='visible'", {{"ELEMENT" -> #}}]&/@elementId
  ]
];

SubmitElement[valueId_] := SubmitElement[$CurrentWebSession, valueId];
SubmitElement[sessionInfo_, valueId_] := Module[ {elementId},
  elementId = If[StringQ[valueId], valueId, LocateElements[sessionInfo, valueId]];
  Which[
    elementId === {}, Message[WebExecute::noelem],
    StringQ[elementId], submit[sessionInfo, elementId],
    ListQ[elementId], submit[sessionInfo, #]&/@elementId
  ]
];

ElementText[valueId_] := ElementText[$CurrentWebSession, valueId];
ElementText[sessionInfo_, valueId_] := Module[ {elementId},
  elementId = If[StringQ[valueId], valueId, LocateElements[sessionInfo, valueId]];
  Which[
    StringQ[elementId], text[sessionInfo, elementId],
    ListQ[elementId], text[sessionInfo, #]&/@elementId,
    True, $Failed
  ]
];

ElementName[valueId_] := ElementName[$CurrentWebSession, valueId];
ElementName[sessionInfo_, valueId_] := Module[ {elementId},
  elementId = If[StringQ[valueId], valueId, LocateElements[sessionInfo, valueId]];
  Which[
    StringQ[elementId], name[sessionInfo, elementId],
    ListQ[elementId], name[sessionInfo, #]&/@elementId,
    True, $Failed
  ]
];


ElementSelected[valueId_] := ElementSelected[$CurrentWebSession, valueId];
ElementSelected[sessionInfo_, valueId_] := Module[ {elementId},
  elementId = If[StringQ[valueId], valueId, LocateElements[sessionInfo, valueId]];
  Which[
    StringQ[elementId], selected[sessionInfo, elementId],
    ListQ[elementId], selected[sessionInfo, #]&/@elementId,
    True, $Failed
  ]
];

ElementEnabled[valueId_] := ElementEnabled[$CurrentWebSession, valueId];
ElementEnabled[sessionInfo_, valueId_] := Module[ {elementId},
  elementId = If[StringQ[valueId], valueId, LocateElements[sessionInfo, valueId]];
  Which[
    StringQ[elementId], enabled[sessionInfo, elementId],
    ListQ[elementId], enabled[sessionInfo, #]&/@elementId,
    True, $Failed
  ]
];
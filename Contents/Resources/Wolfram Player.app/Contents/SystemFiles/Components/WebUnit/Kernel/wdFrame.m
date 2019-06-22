(* ::Section:: *)
(* Less Used*)

FocusFrame[valueId_] := FocusFrame[$CurrentWebSession, valueId];
FocusFrame[sessionInfo_, valueId_] := Module[ {elementId},
  elementId = If[StringQ[valueId], valueId, LocateElements[sessionInfo, valueId]];
  Which[
    elementId === {}, Message[WebExecute::noelem],
    StringQ[elementId], frame[sessionInfo, elementId],
    ListQ[elementId], frame[sessionInfo, #]&/@elementId
  ]

];

(* ::Section:: *)
(* Javascript based functions *)


JavascriptExecute[javascript_] := JavascriptExecute[$CurrentWebSession, javascript];
JavascriptExecute[sessionInfo_, javascript_] /; sessionInfo[[2]] != "Firefox" := execute[sessionInfo, javascript, {}];
JavascriptExecute[sessionInfo_, javascript_] /; sessionInfo[[2]] == "Firefox" := executesync[sessionInfo, javascript, {}];
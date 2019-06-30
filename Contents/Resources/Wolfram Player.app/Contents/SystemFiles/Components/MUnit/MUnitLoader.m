(* Try to update MUnit if newer paclet is available *)
PacletManager`Package`getPacletWithProgress["MUnit", "MUnit", 
	"IsDataPaclet" -> True, "AllowUpdate" -> TrueQ[PacletManager`$AllowDataUpdates], "UpdateSites" -> False];

(* Load the code *)
Get["MUnit`"];

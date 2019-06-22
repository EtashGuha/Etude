(* This FormulaDataLoader file only exists to separate the updating of the FormulaData paclet from its use. 
   This allows any update to the paclet, triggered below, to take effect in the same session. In other words,
   we load the FormulaDataLoader.m file from the currently-installed version of the paclet, it calls PacletUpdate
   (via getPacletWithProgress[] below), and then the Get["FormulaData`"] will load the "real" code from the new paclet.
*)

PacletManager`Package`getPacletWithProgress["InflationAdjust"];

Get["InflationAdjust`"]
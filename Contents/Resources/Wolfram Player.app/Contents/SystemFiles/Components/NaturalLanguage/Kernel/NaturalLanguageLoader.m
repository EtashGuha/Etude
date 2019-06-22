(*
This is an example "Paclet Loader" file. 
It's primary purpose is to check for any paclet-server updates for the paclet, and then load the paclet proper.
This file should be the first one listed under "Contexts" in PacletInfo.m, 
which means its what will be loaded if an AutoLoad symbol is evaluated.
*)

(*this function checks with the Paclet Server(if available) for any updates of the paclet*)
PacletManager`Package`getPacletWithProgress["NaturalLanguage"];

(*this loads "NaturalLanguage.m" which is the source file for this project*)
Get["NaturalLanguage`"];
(*if there were other files to load they could also be loaded via Get[...] here*)
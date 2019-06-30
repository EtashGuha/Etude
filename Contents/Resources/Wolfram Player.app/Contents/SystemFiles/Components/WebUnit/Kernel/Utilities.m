
$Debug = False; DebugPrint[e_] := If[$Debug, Print, Identity];

NonTextKeys[] = {"NULL" -> "\\uE000", "Cancel" -> "\\uE001", "Help" -> "\\uE002", "Backspace" -> "\\uE003", "Tab" -> "\\uE004", 
   "Clear" -> "\\uE005", "Return" -> "\\uE006", "Enter" -> "\\uE007", "Shift" -> "\\uE008", "Control" -> "\\uE009", "Alt" -> "\\uE00A", 
   "Pause" -> "\\uE00B", "Escape" -> "\\uE00C", "Space" -> "\\uE00D", "Pageup" -> "\\uE00E", "Pagedown" -> "\\uE00F", "End" -> "\\uE010",
    "Home" -> "\\uE011", "Leftarrow" -> "\\uE012", "Uparrow" -> "\\uE013", "Rightarrow" -> "\\uE014", 
   "DownArrow" -> "\\uE015", "Insert" -> "\\uE016", "Delete" -> "\\uE017", "Semicolon" -> "\\uE018", 
   "Equals" -> "\\uE019", "Numpad0" -> "\\uE01A", "Numpad1" -> "\\uE01B", "Numpad2" -> "\\uE01C", 
   "Numpad3" -> "\\uE01D", "Numpad4" -> "\\uE01E", "Numpad5" -> "\\uE01F", "Numpad6" -> "\\uE020", 
   "Numpad7" -> "\\uE021", "Numpad8" -> "\\uE022", "Numpad9" -> "\\uE023", "Undefined" -> "\\uE024", 
   "Undefined" -> "\\uE025", "Undefined" -> "\\uE026", "Undefined" -> "\\uE027", "Undefined" -> "\\uE028", 
   "Undefined" -> "\\uE029", "Undefined" -> "\\uE02A",  "Multiply" -> "\\uE02B", "Add" -> "\\uE02C", 
   "Separator" -> "\\uE02D", "Subtract" -> "\\uE02E", "Decimal" -> "\\uE02F", "Divide" -> "\\uE030", "F1" -> "\\uE031", 
   "F2" -> "\\uE032", "F3" -> "\\uE033", "F4" -> "\\uE034", "F5" -> "\\uE035", "F6" -> "\\uE036", "F7" -> "\\uE037", 
   "F8" -> "\\uE038", "F9" -> "\\uE039", "F10" -> "\\uE03A",  "F11" -> "\\uE03B", "F12" -> "\\uE03C", "Command" -> "\\uE03D",
   "\[ShiftKey]" -> "\\uE008", "\[EnterKey]" -> "\\uE007", "\[ControlKey]" -> "\\uE009", "\[TabKey]" -> "\\uE004","\n"->"\\ue007"};

(* ::Section:: *)
(* json *)

format[jsonObject_] := 
 Panel[Grid[
   Map[{First[#], 
      If[MatchQ[Last[#], List[_Rule ...]], format[Last[#]], 
       Short@Last[#]]} &, jsonObject], Alignment -> Left]];


json[Null] := "";
json[expr_] := (DebugPrint[expr]; ImportString[ StringReplace[ExportString[expr, "JSON"], "\\\\u" -> "\\u"], "Text"])

fetch[sessioninfo_, type_, path_, data_, key_] := Module[{res, fetchResult},
	fetchResult = URLFetch[ sessioninfo[[3]] <> path, "Method" -> type, "BodyData" -> json[data]];
	If[!StringQ[fetchResult], Return[$Failed]];
	res = ImportString[fetchResult, "JSON"];
	DebugPrint[format[res]];
	key /. res
];

(* ::Section:: *)
(* get *)
get[sessioninfo_, path_]                := get      [sessioninfo, path, "value"];
get[sessioninfo_, path_, key_]          := fetch    [sessioninfo, "Get", path, Null, key];

(* ::Section:: *)
(* post *)
post[sessioninfo_, path_]               := post     [sessioninfo, path, {"xxx" -> "xxx"}]; (* hack, not all systems like empty post commands *)
post[sessioninfo_, path_, data_]        := post     [sessioninfo, path, data, "value"];
post[sessioninfo_, path_, data_, key_]  := fetch    [sessioninfo, "Post", path ,data, key];

(* ::Section:: *)
(* delete *)
delete[sessioninfo_, path_]             := delete   [sessioninfo, path, "value"];
delete[sessioninfo_, path_, key_]       := fetch    [sessioninfo, "DELETE", path, Null, key];



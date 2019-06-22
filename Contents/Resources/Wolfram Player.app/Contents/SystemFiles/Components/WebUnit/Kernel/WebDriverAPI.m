
(* webDriver api *)

status[sessionInfo_]    := get[sessionInfo, "/status"];

sessions[sessionInfo_]  := get[sessionInfo, "/sessions"];

broswerNames = <|"Chrome"->"chrome","InternetExplorer"->"internet explorer","Firefox"->"firefox"|>;



setsession[sessionInfo_, "Chrome", {Visible -> True}]  :=  post[sessionInfo, "/session", {"desiredCapabilities" -> {"browserName" -> "chrome"}}, {"sessionId", "value"}];
setsession[sessionInfo_, "Chrome",{Visible -> False}]  :=  post[sessionInfo, "/session", {"desiredCapabilities" ->  {"browserName" -> "chrome",
                                                                  "chromeOptions" -> {"args" -> {"--headless", "--disable-gpu"}}}}, {"sessionId", "value"}];
setsession[sessionInfo_, "Chrome",{Visible -> "DisableImage"}]  :=  post[sessionInfo, "/session", {"desiredCapabilities" ->  {"browserName" -> "chrome",
                                                                  "chromeOptions" -> {"prefs" -> {"profile" -> {"default_content_setting_values" -> {"images" -> 2}}}}}},
                                                                  {"sessionId", "value"}];

setsession[sessionInfo_, "Firefox",{Visible -> True}]:=  ("sessionId"/.fetch[sessionInfo, "Post", "/session",
  {"desiredCapabilities" -> {"browserName" -> "firefox"}}, "value"]);

setsession[sessionInfo_, "Firefox",{Visible -> False}]:=  ("sessionId"/.fetch[sessionInfo, "Post", "/session",
  {"desiredCapabilities" ->{"browserName" -> "firefox", "moz:firefoxOptions" -> {"args" -> {"--headless"}}}}, "value"]);

setsession[sessionInfo_, "Firefox",{Visible -> "DisableImage"}]:=  Message[StartWebSession::implem, Return[$Failed]];


getsession[]                            := getsession[$CurrentWebSession];
getsession[sessionInfo_]                := get[sessionInfo, "/session/" <> sessionInfo[[1]]];

forward[]                               := forward[$CurrentWebSession];
forward[sessionInfo_]                   := post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/forward"];

back[]                                  := back[$CurrentWebSession];
back[sessionInfo_]                      := post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/back"];

refresh[]                               := refresh[$CurrentWebSession];
refresh[sessionInfo_]                   := post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/refresh"];

execute[script_, args_]                 := execute[$CurrentWebSession, script, args];
execute[sessionInfo_, script_, args_]   := post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/execute", {"script" -> script, "args" -> args}];

executesync[script_, args_]                 := executesync[$CurrentWebSession, script, args];
executesync[sessionInfo_, script_, args_]   := post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/execute/sync", {"script" -> script, "args" -> args}];

executeasync[script_, args_]                := executeasync[$CurrentWebSession, script, args];
executeasync[sessionInfo_, script_, args_]  := post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/execute/async", {"script" -> script, "args" -> args}];

availableengines[]                      := availableengines[$CurrentWebSession];
availableengines[sessionInfo_]          := get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/available_engines"];

activeengine[]                          := activeengine[$CurrentWebSession];
activeengine[sessionInfo_]              := get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/active_engine"];

activated[]                             := activated[$CurrentWebSession];
activated[sessionInfo_]                 := get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/activated"];

deactivate[]                            := deactivate[$CurrentWebSession];
deactivate[sessionInfo_]                := post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/deactivate"];

activate[]                              := activate[$CurrentWebSession];
activate[sessionInfo_, engine_]         := post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/activate", {"engine" -> engine}];

frame[]                                 := frame[Null];
frame[elementId_]                       := frame[$CurrentWebSession,elementId];
frame[sessionInfo_, elementId_]         := post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/frame", {"id" -> If[elementId===Null,Null,{"ELEMENT"->elementId}]}];

geturl[]                                := geturl[$CurrentWebSession];
geturl[sessionInfo_]                    := get[sessionInfo, "/session/" <> sessionInfo [[1]] <> "/url"];

seturl[url_]                            := seturl[$CurrentWebSession,url];
seturl[sessionInfo_, url_String]        := post[sessionInfo, "/session/" <> sessionInfo [[1]]<> "/url", {"url" -> url}];

screenshot[]                            := screenshot[$CurrentWebSession];
screenshot[sessionInfo_]                := ImportString[get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/screenshot"], "Base64"];

elementscreenshot [elementId_]                  := elementscreenshot[$CurrentWebSession,  elementId];
elementscreenshot [sessionInfo_, elementId_]    := ImportString[ get[sessionInfo, "/session/" <> $CurrentWebSession <> "/element/" <> elementId <> "/screenshot"], "Base64"];

getcookie[]                             := cookie[$CurrentWebSession];
getcookie[sessionInfo_]                 := get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/cookie"];

setcookie[cookie_]                      := setcookie[$CurrentWebSession,cookie];
setcookie[sessionInfo_, cookie_]        := post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/cookie", {"cookie" -> cookie}];

deletecookie[]                          := deletecookie[$CurrentWebSession];
deletecookie[sessionInfo_]              := delete[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/cookie"];

title[]                                 := title[$CurrentWebSession];
title[sessionInfo_]                     := get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/title"];

source[]                                := source[$CurrentWebSession];
source[sessionInfo_]                    := get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/source"];

getorientation[]                        := orientation[$CurrentWebSession];
getorientation[sessionInfo_]            := get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/orientation"];

setorientation[orientation_]            := setorientation[$CurrentWebSession,orientation];
setorientation[sessionInfo_, orientation_] := post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/orientation", {"orientation" -> orientation}];

getalerttext[]                      := getalerttext[$CurrentWebSession];
getalerttext[sessionInfo_]          := get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/alert_text"];
setalerttext[text_]                 := setalerttext[$CurrentWebSession,text];
setalerttext[sessionInfo_, text_]   := post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/alert_text", {"text" -> Characters[text]}];

acceptalert[]                       := acceptalert[$CurrentWebSession];
acceptalert[sessionInfo_]           := post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/accept_alert"];

dismissalert[]                      := dismissalert[$CurrentWebSession];
dismissalert[sessionInfo_]          := post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/dismiss_alert"];

localstorage[]                      := localstorage[$CurrentWebSession];
localstorage[sessionInfo_]          := get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/local_storage"];

moveto[elementId_]                  := moveto[$CurrentWebSession, elementId];
moveto[sessionInfo_, elementId_]    := post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/moveto", {"element" -> elementId}];

getwindow[]                         := getwindow[$CurrentWebSession];
getwindow[sessionInfo_]             := get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/window"];

setwindow[window_]                  := setwindow[$CurrentWebSession,window];
setwindow[sessionInfo_, window_]    := Switch [sessionInfo [[2]],
    "Chrome"|"InternetExplorer"|"Edge", post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/window", {"name" -> window}],
    "Firefox",                          post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/window", {"handle" -> window}]
];

deletewindow[]                      := deletewindow[$CurrentWebSession];
deletewindow[sessionInfo_]          := delete[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/window"];

deletesession[]                     := deletesession[$CurrentWebSession];
deletesession[sessionInfo_]         := delete[sessionInfo, "/session/" <> sessionInfo[[1]] ];

windowhandle[]                      := windowhandle[$CurrentWebSession];
windowhandle[sessionInfo_]          := Switch[sessionInfo [[2]],
    "Chrome",                               get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/window_handle"],
    "Firefox"|"InternetExplorer"|"Edge",    get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/window/handle"]
]


windowhandles[]                     := windowhandles[$CurrentWebSession];
windowhandles[sessionInfo_]         := Switch[sessionInfo [[2]],
    "Chrome",                               get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/window_handles"],
    "Firefox"|"InternetExplorer"|"Edge",    get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/window/handles"]
];

getwindowsize[windowHandle_]                := getwindowsize[$CurrentWebSession, windowHandle];
getwindowsize[sessionInfo_, windowHandle_]  := Switch[sessionInfo [[2]],
    "Chrome"|"InternetExplorer"|"Edge", get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/window/" <> windowHandle <> "/size"],
    "Firefox",                          get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/window" <> "/size"]
]

setwindowsize[windowHandle_, {width_, height_}]                 := setwindowsize[$CurrentWebSession, windowHandle, {width, height}];
setwindowsize[sessionInfo_, windowHandle_, {width_, height_}]   := Switch[sessionInfo [[2]],
    "Chrome"|"InternetExplorer"|"Edge", post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/window/" <> windowHandle <> "/size", {"width" -> width, "height" -> height}],
    "Firefox",                          post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/window"  <> "/rect",                 {"width" -> width, "height" -> height}]
];

getwindowposition[windowHandle_]                := getwindowposition[$CurrentWebSession, windowHandle];
getwindowposition[sessionInfo_, windowHandle_]  :=  Switch [sessionInfo [[2]],
    "Chrome"|"InternetExplorer"|"Edge", get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/window/"  <>  windowHandle    <>  "/position"],
    "Firefox",                          get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/window"   <>  "/position"]
];


setwindowposition[windowHandle_, {x_, y_}]                  := setwindowposition[$CurrentWebSession,windowHandle, {x, y}];
setwindowposition[sessionInfo_, windowHandle_, {x_, y_}]    :=  Switch [sessionInfo [[2]],
    "Chrome"|"InternetExplorer"|"Edge", post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/window/" <> windowHandle <> "/position", {"x" -> x, "y" -> y}],
    "Firefox",                          post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/window" <> "/position", {"x" -> x, "y" -> y}]
]


windowmaximize[windowHandle_]               := windowmaximize[$CurrentWebSession, windowHandle];
windowmaximize[sessionInfo_, windowHandle_] := Switch [sessionInfo [[2]],
    "Chrome"|"Edge",                post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/window/" <> windowHandle <> "/maximize"],
    "Firefox"|"InternetExplorer",   post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/window"<>"/maximize"]
];

windowminimize[windowHandle_]               := windowminimize[$CurrentWebSession, windowHandle];
windowminimize[sessionInfo_, windowHandle_] := Switch [sessionInfo [[2]],
    "Chrome",                               post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/window/" <> windowHandle <> "/minimize"],
    "Firefox"|"InternetExplorer"|"Edge",    post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/window"<>"/minimize"]
];


windowfullscreen::nnarg             = "Only supported for Chrome webbrowser.";
windowfullscreen[]                  := windowfullscreen[$CurrentWebSession];
windowfullscreen[sessionInfo_]/; sessionInfo[[2]] == "Chrome"   := post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/window" <>  "/fullscreen"];
windowfullscreen[sessionInfo_]/; sessionInfo[[2]] != "Chrome"   :=	(Message[windowfullscreen::nnarg, sessionInfo];   Return[$Failed]);

element[data_]                      :=  element[$CurrentWebSession, data];
element[sessionInfo_, data_]        :=  post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/element", data];

activeelement[]                     := activeelement[$CurrentWebSession];
activeelement[sessionInfo_]         := "ELEMENT" /. post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/element/active"];

elements[data_]                     := elements[$CurrentWebSession, data];
elements[sessionInfo_, data_]       :=  post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/elements", data];

describe[elementId_]                := describe[$CurrentWebSession, elementId];
describe[sessionInfo_, elementId_]  := get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/element" <> elementId];

click[elementId_]                   := click[$CurrentWebSession, elementId];
click[sessionInfo_, elementId_]     := post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/element/" <> elementId <> "/click"];

submit[elementId_]                  := submit[$CurrentWebSession,elementId];
submit[sessionInfo_, elementId_]    := post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/element/" <> elementId <> "/submit"];

text[elementId_]                    := text[$CurrentWebSession,elementId];
text[sessionInfo_, elementId_]      := get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/element/" <> elementId <> "/text"];

name[elementId_]                    := name[$CurrentWebSession,elementId];
name[sessionInfo_, elementId_]      := get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/element/" <> elementId <> "/name"];

clear[elementId_]                   := clear[$CurrentWebSession,elementId];
clear[sessionInfo_, elementId_]     := post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/element/" <> elementId <> "/clear"];

selected[elementId_]                := selected[$CurrentWebSession,elementId];
selected[sessionInfo_, elementId_]  := get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/element/" <> elementId <> "/selected"];

enabled[elementId_]                 := enabled[$CurrentWebSession,elementId];
enabled[sessionInfo_, elementId_]   := get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/element/" <> elementId <> "/enabled"];

displayed[elementId_]               := displayed[$CurrentWebSession,elementId];
displayed[sessionInfo_, elementId_] := get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/element/" <> elementId <> "/displayed"];

attribute[elementId_, attributeName_]                   := attribute[$CurrentWebSession, elementId, attributeName];
attribute[sessionInfo_, elementId_, attributeName_]     :=  get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/element/" <> elementId <> "/attribute/" <> attributeName];

equals[elementId1_,elementId2_]                         := equals[$CurrentWebSession,elementId1,elementId2];
equals[sessionInfo_, elementId1_, elementId2_]          := get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/element/" <> elementId1 <> "/equals/" <> elementId2];

size[elementId_]                    := size[$CurrentWebSession,elementId];
size[sessionInfo_, elementId_]      := {"width", "height"} /. get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/element/" <> elementId <> "/size"];

location[elementId_]                := location[$CurrentWebSession,elementId];
location[sessionInfo_, elementId_]  := {"x", "y"} /. get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/element/" <> elementId <> "/location"];

locationinview[elementId_]                              := locationinview[$CurrentWebSession,elementId];
locationinview[sessionInfo_, elementId_]                := {"x", "y"} /. get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/element/" <> elementId <> "/location_in_view"];

elementcssproperty[elementId_, propertyName_]                   := elementcssproperty[$CurrentWebSession,elementId,propertyName];
elementcssproperty[sessionInfo_, elementId_, propertyName_]     := get[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/element/" <> elementId <> "/css/" <> propertyName];

value[elementId_, keyStrokeSequence_String]                 := value[$CurrentWebSession,elementId, keyStrokeSequence];
value[sessionInfo_, elementId_, keyStrokeSequence_String]   := value[sessionInfo, elementId, Characters[keyStrokeSequence]];
value[sessionInfo_, elementId_, keyStrokeSequence_List]     := 	 Switch[sessionInfo [[2]],
    "Firefox",                          post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/element/" <> elementId <> "/value", {"text" -> StringJoin[keyStrokeSequence /. NonTextKeys[]]}],
    "Chrome"|"InternetExplorer"|"Edge", post[sessionInfo, "/session/" <> sessionInfo[[1]] <> "/element/" <> elementId <> "/value", {"value" -> (keyStrokeSequence /. NonTextKeys[])}]
		];

keys[keyStrokeSequence_String]                  := keys[$CurrentWebSession, keyStrokeSequence];
keys[sessionInfo_, keyStrokeSequence_String]    := keys[sessionInfo, Characters[keyStrokeSequence]];
keys[sessionInfo_, keyStrokeSequence_List]      := post["/session/" <> sessionInfo[[1]] <> "/keys", {"value" -> (keyStrokeSequence /. NonTextKeys[])}];



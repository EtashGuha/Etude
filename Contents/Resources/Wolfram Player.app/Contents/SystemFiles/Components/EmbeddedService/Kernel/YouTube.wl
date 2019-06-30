Options[YouTube]={
 "Method"->"InvisibleFrame", (* only supported option at the moment *)
 "AutoHide"->Automatic, (* other options: "FadeProgressBar"  (default), "FadeAllControls", "Off" *)
 "AutoPlay" ->Automatic, (* other options: True and False (default) *)
 "ClosedCaptioning"->Automatic, (* other options: True (can not be False, automatic default based on user settings) *)
 "ProgressColor"->Automatic, (* other options: "Red" (default) and "White" *)
 "PlayerControls"->Automatic, (* other options: True (default) and False  *)
 "KeyboardControls"->Automatic, (* other options: "Enable" (default) and "Disable" *)
 "JavascriptAPI"->Automatic, (* other options: True and False (default) *)
 "StopTime"->None, (* positive integer (seconds from start of video) *)
 "FullScreenButton"->Automatic, (* other options: True (default) and False *)
 "Language"->Automatic, (* ISO 639-1 language codes *)
 "ShowVideoAnnotations"->Automatic, (* other options: True (default) and False *)
 "VideoList"->None, (* other options: {"Search", queryString}, {"UserUploads",userChannel}, {"PlayList", playListID} *)
 "LoopVideo"->Automatic, (* other options: True and False (default) *)
 "ModestBranding"->Automatic, (* other options: True and False (default) *)
 "OriginSecurity"->Automatic, (* should be set to your domain *)
 "PlayerAPIIdentifier"->None, (* any alphanumeric string (used for Javascript API ) *)
 "PlayList"->None, (* list of video ids to play (after the initial video id specified already) *)
 "ShowRelatedVideos"->Automatic, (* other options: True (default) and False *)
 "ShowMetaInformation"->Automatic, (* other options: True (default) and False *)
 "StartTime"->None, (* a positive integer (seconds from start of video) *)
 "PlayerTheme"->Automatic, (* other options: "Dark" (default) and "Light" *)
 (* iframe related options *)
 "Width"->560,
 "Height"->315,
 "FrameBorder"->0,
 "AllowFullScreen"->True
};

options[name_String,option_String]:=TemplateApply[StringTemplate["`name`='`option`'"],<|"name"->name,"option"->option|>];

urloptions[name_String,option_String]:=TemplateApply[StringTemplate["`name`=`option`"],<|"name"->name,"option"->URLEncode[option]|>];

YouTube[id_String,OptionsPattern[]]:=Module[
 {iframe,iframeTemplate,url,urlTemplate,method,autohide,autoplay,ccloadpolicy,color,controls,disablekb,enablejsapi,end,fs,hl,ivloadpolicy,list,listType,loop,modestbranding,origin,playerapiid,playlist,playsinline,rel,showinfo,start,theme,width,height,frameborder,allowfullscreen,src},
 method=OptionValue["Method"];
 If[method=!="InvisibleFrame",Message[YouTube::notsupported];Return[$Failed]];
 (* *)
 (* url handling *)
 (* *)
 urlTemplate="https://www.youtube.com/embed/`videoid`?`autohide`&`autoplay`&`ccloadpolicy`&`color`&`controls`&`disablekb`&`enablejsapi`&`end`&`fs`&`hl`&`ivloadpolicy`&`list`&`listType`&`loop`&`modestbranding`&`origin`&`playerapiid`&`playlist`&`playsinline`&`rel`&`showinfo`&`start`&`theme`";
 autohide=urloptions["autohide",OptionValue["AutoHide"]/.{Automatic->"2","FadeProgressBar"->"2","FadeAllControls"->"1","Off"->"0"}];
 autoplay=urloptions["autoplay",OptionValue["AutoPlay"]/.{Automatic->"0",True->"1",False->"0"}];
 ccloadpolicy=urloptions["cc_load_policy",OptionValue["ClosedCaptioning"]/.{Automatic->"0",True->"1",False->"0"}];
 color=urloptions["color",OptionValue["ProgressColor"]/.{Automatic->"red","Red"->"red","White"->"white"}];
 controls=urloptions["controls",OptionValue["PlayerControls"]/.{Automatic->"1",True->"1",False->"0"}];
 disablekb=urloptions["disablekb",OptionValue["KeyboardControls"]/.{Automatic->"0",True->"0",False->"1"}];
 enablejsapi=urloptions["enablejsapi",OptionValue["JavascriptAPI"]/.{Automatic->"0",True->"1",False->"0"}];
 end=urloptions["end",OptionValue["StopTime"]/.{None->"",n_Integer:>ToString[n]}];
 fs=urloptions["fs",OptionValue["FullScreenButton"]/.{Automatic->"1",True->"1",False->"0"}];
 hl=urloptions["hl",OptionValue["Language"]/.{Automatic->""}];
 ivloadpolicy=urloptions["iv_load_policy",OptionValue["ShowVideoAnnotations"]/.{Automatic->"1",True->"1",False->"3"}];
 list=urloptions["list",OptionValue["VideoList"]/.{None->"",{_,e_}:>e}];
 listType=urloptions["listType",OptionValue["VideoList"]/.{None->"",{"Search",_}:>"search",{"UserUploads",_}:>"user_uploads",{"PlayList",_}:>"playlist"}];
 loop=urloptions["loop",OptionValue["LoopVideo"]/.{Automatic->"0",True->"1",False->"0"}];
 modestbranding=urloptions["modestbranding",OptionValue["ModestBranding"]/.{Automatic->"",True->"1",False->""}];
 origin=urloptions["origin",OptionValue["OriginSecurity"]/.{Automatic->"wolframcloud.com"}];
 playerapiid=urloptions["playerapiid",OptionValue["PlayerAPIIdentifier"]/.{None->""}];
 playlist=urloptions["playlist",OptionValue["PlayList"]/.{None->"",n_List:>StringJoin[Riffle[n,","]]}];
 rel=urloptions["rel",OptionValue["ShowRelatedVideos"]/.{Automatic->"1",True->"1",False->"0"}];
 showinfo=urloptions["showinfo",OptionValue["ShowMetaInformation"]/.{Automatic->"1",True->"1",False->"0"}];
 start=urloptions["start",OptionValue["StartTime"]/.{None->"",n_Integer:>ToString[n]}];
 theme=urloptions["theme",OptionValue["PlayerTheme"]/.{Automatic->"dark","Dark"->"dark","Light"->"light"}];
 url=TemplateApply[StringTemplate[urlTemplate],<|"videoid"->id,"autohide"->autohide,"autoplay"->autoplay,"ccloadpolicy"->ccloadpolicy,"color"->color,"controls"->controls,"disablekb"->disablekb,"enablejsapi"->enablejsapi,"end"->end,"fs"->fs,"hl"->hl,"ivloadpolicy"->ivloadpolicy,"list"->list,"listType"->listType,"loop"->loop,"modestbranding"->modestbranding,"origin"->origin,"playerapiid"->playerapiid,"playlist"->playlist,"rel"->rel,"showinfo"->showinfo,"start"->start,"theme"->theme|>];
 (* *)
 (* iframe handling *)
 (* *)
 width=options["width",OptionValue["Width"]/.{n_Integer:>ToString[n]}];
 height=options["height",OptionValue["Height"]/.{n_Integer:>ToString[n]}];
 frameborder=options["frameborder",OptionValue["FrameBorder"]/.{n_Integer:>ToString[n]}];
 allowfullscreen=options["allowfullscreen",OptionValue["AllowFullScreen"]/.{True->"true",False->"false"}];
 iframeTemplate=FileTemplate[ FileNameJoin[ { $TemplatesDirectory, "youtube.template" } ] ];
 src=options["src",url];
 iframe=TemplateApply[iframeTemplate,<|"id"->"id='ytplayer'","src"->src,"width"->width,"height"->height,"frameborder"->frameborder,"allowfullscreen"->allowfullscreen|>];
 xPrint[iframe];
 EmbeddedHTML[iframe, ImageSize -> 1.1*{OptionValue["Width"], OptionValue["Height"]} ]
];

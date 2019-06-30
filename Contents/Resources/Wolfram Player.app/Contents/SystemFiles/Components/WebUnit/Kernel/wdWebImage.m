Options[WebImage]= {Method -> Automatic, Visible -> False} ;

WebImage[urls_?ListQ,opts:OptionsPattern[]]:= WebImage[#,opts]&/@urls;

WebImage[url_?StringQ,opts:OptionsPattern[]] := Block[{s,img,checkOpts},

  (*checkOpts is a list & checkOpts is not an empty list*)
  checkOpts = {opts};

  (*If opts == {___} passed by user, then the following step makes checkOpts a single list
  but not a {{___}}*)
  If[MatchQ[checkOpts, {{___}}], checkOpts = First[checkOpts]];
(*To make sure there are no invalid options*)
  If[FilterRules[checkOpts, Except[Options[WebImage]]] =!= {},
    Message[WebImage::invalidmethod, checkOpts];Return[$Failed]
  ];

  s = StartWebSession[OptionValue[Method], Visible->OptionValue[Visible]];

  If[!MatchQ[s,_WebSessionObject],
    Message[WebImage::start];
    Return[$Failed]
  ];
  WebExecute[s, "OpenWebPage" -> url];
  img = WebExecute[s, "CapturePage"];
  DeleteObject[s];
  Return[img]
]
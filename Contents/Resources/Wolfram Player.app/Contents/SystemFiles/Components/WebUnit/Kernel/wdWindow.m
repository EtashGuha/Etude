(* ::Section:: *)
(*Window commands*)


DeleteWindow[x___] := deletewindow[x];
GetWindow[x___] := getwindow[x];
GetWindowSize[x___] := getwindowsize[x];
SetWindowSize[x___] := setwindowsize[x];
GetWindowPosition[x___] := getwindowposition[x];
SetWindowPosition[x___] := setwindowposition[x];
(*SwitchWindow[x___]          :=	setwindow[x];*)
SetBrowserWindow[x___] := setwindow[x];

WindowFullscreen[x___] := windowfullscreen[x];
WindowMaximize[x___] := windowmaximize[x];
WindowMinimize[x___] := windowminimize[x];


(*
WindowMinimize[x___] := Message[WindowMinimize::nnarg, x]; Return[$Failed] ;
WindowMinimize::nnarg = "Currently not supported. Running WindowMaximize command will make \
the browser window size back to the size before maximization.";
*)


BrowserWindows[x___] := windowhandles[x];
CurrentWindowHandle = windowhandle
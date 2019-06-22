(* :Title: InstallNET.m *)

(* :Author:
        Todd Gayley
        tgayley@wolfram.com
*)

(* :Package Version: 1.7 *)

(* :Mathematica Version: 5.0 *)
             
(* :Copyright: .NET/Link source code (c) 2003-2019, Wolfram Research, Inc. All rights reserved.

   Use is governed by the terms of the .NET/Link license agreement, which can be found at
   www.wolfram.com/solutions/mathlink/netlink.
*)

(* :Discussion:
   InstallNET, UninstallNET and related.
    
   This file is a component of the .NET/Link Mathematica source code.
   It is not a public API, and should never be loaded directly by users or programmers.

   .NET/Link uses a special system wherein one package context (NETLink`) has its implementation
   split among a number of .m files. Each component file has its own private context, and also
   potentially introduces public symbols (in the NETLink` context) and so-called "package" symbols,
   where the term "package" comes from Java terminology, referring to symbols that are visible
   everywhere within the implementation of .NET/Link, but not to clients. The NETLink.m file itself
   is produced by an automated tool from the component files and contains only declarations.
   
   Do not modify the special comment markers that delimit Public- and Package-level exports.
*)


(*<!--Public From InstallNET.m

InstallNET::usage =
"InstallNET[] launches the .NET runtime and prepares it to be used from the Wolfram Language. Only one .NET runtime is ever launched; \
subsequent calls to InstallNET after the first have no effect."

UninstallNET::usage =
"UninstallNET[] shuts down the .NET runtime that was started by InstallNET. It is provided mainly for developers who are \
actively recompiling .NET types for use in the Wolfram Language and therefore need to shut down and restart the .NET runtime to reload \
the modified types. Users generally have no reason to call UninstallNET. The .NET runtime is a shared resource used by \
potentially many Wolfram Language programs. You should leave it running unless you are absolutely sure you need to shut it down."

ReinstallNET::usage =
"ReinstallNET[] is a convenience function that calls UninstallNET[] followed by InstallNET[]. See the usage messages for \
InstallNET and UninstallNET for more information."

NETLink::usage =
"NETLink[] returns the MathLink LinkObject that is used to communicate with the .NET/Link .NET runtime. It will return \
Null if .NET is not running."

NETUILink::usage =
"NETUILink[] returns the MathLink LinkObject used by calls to the Wolfram Language that originate from .NET user-interface actions, or Null if no such link is present."

-->*)


(*<!--Package From InstallNET.m

$inPreemptiveCallFromNET

-->*)


(* Current context will be NETLink`. *)

Begin["`InstallNET`Private`"]


InstallNET::fail = "A link to the .NET runtime could not be established."

InstallNET::uifail = "The separate .NET user-interface link could not be established."

NET::init = "The .NET runtime is not running. You must call InstallNET[] to start the .NET runtime."

General::netlink = "`1` cannot operate because InstallNET[] failed to launch the .NET runtime."

NETLink::net =
"The .NET Framework is not installed on this machine. It must be installed before .NET/Link can be used. \
You can install the .NET Framework at Microsoft's Windows Update web site: http://windowsupdate.microsoft.com. \
If you do not see the .NET Framework listed among the available updates you probably need to update your version \
of Internet Explorer first."

NETLink::netvers =
".NET/Link requires at least version 2.0 of the .NET Framework. \
You can install the latest .NET Framework at Microsoft's Windows Update web site: http://windowsupdate.microsoft.com."



If[!ValueQ[$nlink], $nlink = Null]
If[!ValueQ[$uilink], $uilink = Null]

Internal`SetValueNoTrack[$nlink, True]
Internal`SetValueNoTrack[$uilink, True]


NETLink[] := $nlink

NETUILink[] := $uilink


Options[InstallNET] = {"Force32Bit" -> False, "MonoPath"->"mono"}


InstallNET[opts:OptionsPattern[]] := InstallNET[Null, opts]

(* This is called from .NET code, as part of EnableObjectReferences(). Not really intended for use by users. *)
InstallNET[linkName_String, opts:OptionsPattern[]] := 
    Module[{links},
        links = Select[Links[], First[#] === linkName &];
        If[Length[links] > 0,
            InstallNET[First[links], opts],
        (* else *)
            $Failed
        ]
    ]

InstallNET[link:(_LinkObject | Null), OptionsPattern[]] :=
	preemptProtect[
		Module[{netLinkExeName, netlinkPath, useUILink, force32Bit, monoPath},
			(* Don't launch if running in protected mode, for security reasons. Possibly enable
			   security-restricted launch in the future, like we do with J/Link.
			*)
			If[Developer`CheckProtectedMode[InstallNET],
				(* CheckProtectedMode will have already issued its own message. *)
				Message[InstallNET::fail];
				Return[$Failed]
			];
			(* Bail out right away if link is already open and OK. *)
			If[MemberQ[Links[], $nlink],
				(* Only do the check on the health of the link if not already in a preemptive
				   transaction with .NET. It's obviously not necessary in such cases, but it also
				   can cause the link to die if we call LinkReadyQ on a link when we happen
				   to already be blocking in its yield function in the main computation.
				*)
				If[$inExternalCall,
					Return[$nlink]
				];
				LinkReadyQ[$nlink]; (* Hit link to force LinkError to give current value. *)
				If[First[LinkError[$nlink]] === 0,
					Return[$nlink],
				(* else *)
					UninstallNET[]
				],
			(* else *)
				(* This extra test (nlink has a value, but it is not in Links[]) is to catch cases where
				   user has improperly shut down .NET (e.g., by calling LinkClose).
				*)
				If[Head[$nlink] === LinkObject,
					resetMathematica[];
					$nlink = Null
				]
			];
			(* Check for the existence of the .NET Framework on this machine. *)
			If[StringQ[Environment["WINDIR"]] && !DirectoryQ[ToFileName[{Environment["WINDIR"], "assembly"}]],
			    Message[NETLink::net];
				Return[$Failed]
			];
			If[Head[link] === LinkObject,
				$nlink = Install[link],
			(* else *)
				{force32Bit, monoPath} = OptionValue[{"Force32Bit", "MonoPath"}];
				netLinkExeName = If[TrueQ[force32Bit], "InstallableNET32.exe", "InstallableNET.exe"];
				netlinkPath = ToFileName[$netlinkDir, netLinkExeName];
				If[!osIsWindows[],
					netlinkPath = monoPath <> " \"" <> netlinkPath <> "\""
				];
				(* On Windows and Unix, LinkQuote is not needed, and can interfere with finding the program to launch.
				   Because LinkQuote is wrapped around another function we don't want called (FindFile), we replace it
				   with Hold. 
				*)
				Block[{System`Dump`LinkQuote = Hold},
					(* On OSX, we need to make sure that Mono can find the mathlink shared lib. The best/only way I know how
					   to do this is by making its dir the current dir during the launch. Remember that the kernel process doesn't
					   inherit environment variables like LD_LIBRARY_PATH because it is typically not launched from a shell.
					*)
					If[osIsMacOSX[],
					    (* Accommodate both pre-10.0.2 and post-10.0.2 layouts. *)
					    Scan[
					        If[DirectoryQ[#], SetDirectory[#]]&, 
					        {ToFileName[{$InstallationDirectory, "Frameworks", "mathlink.framework"}],
					         ToFileName[{$InstallationDirectory, "Contents", "Frameworks", "mathlink.framework"}]}
					    ]				       
					];
					$nlink = Install[netlinkPath];
					If[osIsMacOSX[], ResetDirectory[]]
				]
			];
	        
			(* Don't want to set up uiLink if we are calling InstallNET[$ParentLink] from NET (this happens
			   during IKernelLink.EnableObjectReferences()). In that case there won't be a Reader thread, so
			   we don't want the uiLink.
			*)
			useUILink = useUILinkQ[] && link =!= $ParentLink;
			
			If[Head[$nlink] === LinkObject,
				MathLink`LinkAddInterruptMessageHandler[$nlink];
				$uilink = initNET[useUILink]
			];
			If[Head[$nlink] === LinkObject && (Head[$uilink] === LinkObject || !useUILink),
				$nlink,
			(* else *)
				Message[InstallNET::fail];
				If[Head[$nlink] === LinkObject, LinkClose[$nlink]];
				$nlink = Null;
				If[Head[$uilink] === LinkObject, LinkClose[$uilink]];
				$uilink = Null;
				$Failed
			]
		]
	]


UninstallNET[] :=
	preemptProtect[
		Module[{res = Null, wasOn},
			If[Head[$nlink] === LinkObject,
				(* To avoid potentially many errors, only call onUnloadType methods if nlink is alive and well. *)
				If[MemberQ[Links[], $nlink],
					LinkReadyQ[$nlink]; (* Hit link to force LinkError to give current value. *)
					If[First[LinkError[$nlink]] === 0,
						(* TODO: Not wired up yet. *)
						callAllUnloadTypeMethods[]
					]
				];
				UnshareFrontEnd[$nlink];
				If[MemberQ[SharingLinks[], $nlink],
					UnshareKernel[$nlink]
				];
				resetMathematica[];
				(* Before calling Uninstall, turn off the linkn message. This prevents seeing messages warning that
				   the link is dead while we are closing it. They would otherwise show up when re-installing after a
				   crash or forced quit of .NET.
				*)
				wasOn = Head[LinkObject::linkn] =!= $Off;
				Off[LinkObject::linkn];
				res = Uninstall[$nlink];
				If[Head[$uilink] === LinkObject,
					MathLink`RemoveSharingLink[$uilink];
					LinkClose[$uilink]
				];
				If[wasOn, On[LinkObject::linkn]];
				$nlink = Null;
				$uilink = Null
			];
			res
		]
	]


Options[ReinstallNET] = Options[InstallNET]

ReinstallNET[args___] := preemptProtect[UninstallNET[]; InstallNET[args]]


initNET[setupUILink:(True | False)] :=
    Module[{link, prot},
        (* Load all assemblies in the special WRI SystemFiles/assembly dir. *)
        LoadNETAssembly[FileNames[{"*.dll", "*.exe"}, ToFileName[{$TopDirectory, "SystemFiles", "assembly"}]]];
        If[isPreemptiveKernel[] && setupUILink,
	        (* Set up the UI link. *)
	        prot = "SharedMemory";
	        link = LinkCreate[LinkProtocol->prot];
	        If[TrueQ[nUILink[First[link], prot]],
	            LinkConnect[link];
				MathLink`AddSharingLink[link,
						MathLink`LinkSwitchPre -> linkSwitchPreFunc,
						MathLink`LinkSwitchPost -> linkSwitchPostFunc,
						MathLink`AllowPreemptive -> True,
						MathLink`ImmediateStart -> True
				];
				MathLink`LinkAddInterruptMessageHandler[link];
				(* UI link is a daemon link, meaning that kernel should not stay alive just for it. *)
				MathLink`SetDaemon[link, True];
	            link,
	        (* else *)
	            Message[InstallNET::uifail];
	            LinkClose[link];
	            Null
	        ],
	    (* else *)
	    	(* Version 5.0 or earlier--no UI Link *)
	    	Null
	    ]
	]


If[!ValueQ[$inPreemptiveCallFromNET], $inPreemptiveCallFromNET = False]


linkSwitchPreFunc[___] :=
	Block[{uiLink, oldFrontEnd},
	    uiLink = NETUILink[];
		If[MathLink`IsPreemptive[], $inPreemptiveCallFromNET = True];
		oldFrontEnd = 
			If[isServiceFrontEnd[],
				MathLink`SetServiceFrontEnd[],
			(* else *)
				If[FrontEndSharedQ[NETLink[]],
					(* Note that for legacy reasons, users call ShareFrontEnd[NETLink[]], but really
					   it is the NETUILink[] ($activeNETLink at the moment) that the FE-specific traffic
					   goes out on.
					*)
					MathLink`SetFrontEnd[uiLink],
				(* else *)
					MathLink`SetFrontEnd[Null];
					MathLink`SetMessageLink[uiLink]
				]
			];
		If[First[oldFrontEnd] === False,
			(* There was no ServiceLink, and the MessageLink was set to Null.
			   FE services won't work, but at least we can set the MessageLink
			   to the active link (uiLink), so that side-effect output will come to
			   .NET and not get completely lost.
			*)
			MathLink`SetMessageLink[uiLink]
		];
		oldFrontEnd
	]
	
linkSwitchPostFunc[oldFrontEnd_] :=
	(
		MathLink`RestoreFrontEnd[oldFrontEnd];
		$inPreemptiveCallFromNET = False;
	)
	


(* The Mathematica-side things that must be done when starting a fresh .NET session. After this func is run, it should
    be true that the kernel is in the same state it was in before any .NET sessions were started. The possible exception
    to this is that certain contexts may exist that weren't present before, but their contents should be empty. This function
    is only used during InstallNET; it is not user-visible, and no attempt is made to provide users with a way to
    "reset" their Mathematica-.NET session.
*)
resetMathematica[] :=
    AbortProtect[
        clearNETDefs[];
        resetNETBlock[];
        clearComplexClass[];
    ]


(* Tells whether to set up the separate UI link. Don't want to setup uiLink if not a preemptive kernel, or a standalone kernel. *)
useUILinkQ[] :=
	isPreemptiveKernel[] && ValueQ[$ParentLink] && $ParentLink =!= Null;


End[]

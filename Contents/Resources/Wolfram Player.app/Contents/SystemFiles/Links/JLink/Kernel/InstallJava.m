(*! ProblemDetectorDisable[ classlocator ]  !*)
(* :Title: InstallJava.m *)

(* :Author:
        Todd Gayley
        tgayley@wolfram.com
*)

(* :Package Version: 4.9 *)

(* :Mathematica Version: 4.0 *)

(* :Copyright: J/Link source code (c) 1999-2019, Wolfram Research, Inc. All rights reserved.

   Use is governed by the terms of the J/Link license agreement, which can be found at
   www.wolfram.com/solutions/mathlink/jlink.
*)

(* :Discussion:
   InstallJava, UninstallJava and related.

   This file is a component of the J/Link Mathematica source code.
   It is not a public API, and should never be loaded directly by users or programmers.

   J/Link uses a special system wherein one package context (JLink`) has its implementation
   split among a number of .m files. Each component file has its own private context, and also
   potentially introduces public symbols (in the JLink` context) and so-called "package" symbols,
   where the term "package" comes from Java terminology, referring to symbols that are visible
   everywhere within the implementation of J/Link, but not to clients.
*)


InstallJava::usage =
"InstallJava[] launches the Java runtime and prepares it to be used from the Wolfram Language. Only one Java runtime is ever launched; subsequent calls to InstallJava after the first have no effect."

StartJava::usage =
"StartJava is deprecated. Use InstallJava instead."

UninstallJava::usage =
"UninstallJava[] shuts down the Java runtime that was started by InstallJava. It is provided mainly for developers who are actively recompiling Java classes for use in the Wolfram Language and therefore need to shut down and restart the Java runtime to reload the modified classes. Users generally have no reason to call UninstallJava. The Java runtime is a shared resource used by potentially many Wolfram Language programs. You should leave it running unless you are absolutely sure you need to shut it down."

QuitJava::usage =
"QuitJava is deprecated. Use UninstallJava instead."

ReinstallJava::usage =
"ReinstallJava[] is a convenience function that calls UninstallJava followed by InstallJava. It takes the same arguments as InstallJava. See the usage messages for InstallJava and UninstallJava for more information."

RestartJava::usage =
"RestartJava is deprecated. Use ReinstallJava instead."

JavaLink::usage =
"JavaLink[] returns the MathLink LinkObject that is used to communicate with the J/Link Java runtime. It will return Null if Java is not running."

JavaUILink::usage =
"JavaUILink[] returns the MathLink LinkObject used by calls to the Wolfram Language that originate from Java user-interface actions, or Null if no such link is present."

ClassPath::usage =
"ClassPath is an option to InstallJava that controls whether the Java runtime should include the contents of the CLASSPATH environment variable in its class search path. The default is ClassPath->Automatic, which means to include CLASSPATH. If you specify ClassPath->None, CLASSPATH will be ignored. You can also specify a string giving a classpath specification in the standard platform-specific notation. Users considering specifying a string as the value for the ClassPath option should probably use the more flexible AddToClassPath function instead. The main use for the ClassPath option is to set it to None, in case you want to specifically prevent J/Link from including the contents of the CLASSPATH variable in its search path."

CommandLine::usage =
"CommandLine is an option to InstallJava that specifies the first part of the command line that should be used to launch the Java runtime. You can use this option to specify a name other than \"java\" for the Java runtime, or if you have more than one version of Java installed and you need to specify the full path to the runtime you want launched. The value of this option does not specify the entire command line, however, as InstallJava may add some arguments to the end of the line, depending on other options."

JVMArguments::usage =
"JVMArguments is an option to InstallJava that allows you to specify additional command-line arguments passed to the Java virtual machine at startup. The string you specify is added to the command line used to launch Java. You can use this option to specify properties with the standard -D syntax, such as \"-Dsome.property=true\". This option is not supported on Mac OSX."

CreateExtraLinks::usage =
"CreateExtraLinks is an option to InstallJava that allows you to specify whether J/Link should establish the special extra links it uses internally. Only very advanced programmers will be concerned with this option. The default value is Automatic."

(* This is no longer supported (starting in M10. *)
RegisterJavaInitialization::usage =
"RegisterJavaInitialization is an internal symbol."


Begin["`Package`"]

autoClassPath

inPreemptiveCallFromJava

finishInstall

javaPreemptiveLink  (* Not public like JavaLink, JavaUILink *)

(* Experimental features for 6.0 *)
InstallJavaInternal
AsyncInstall

$jlinkAppDir (* Workbench can set this to control where JLink.app is located on OSX. *)

(* Experimental feature to allow users to overide default behavior and force synchronous launch behavior. *)
$ForceSynchronousLaunch = False

(* Expression to be evaluated during InstallJava; allows clients to hook into Java startup to exec M code. Used by MOnline. *)
$jlinkInit

(* temporary *)
$InternalLink

End[]   (* `Package` *)


(* Current context will be JLink`. *)

Begin["`InstallJava`Private`"]


(* The options that deal with classpath are intended for users, not developers. In other words, you probably
    won't distribute code that uses these options. For one thing, you can't currently specify paths in a
    cross-platform way. Developers will have to tell their users to put their Java classes in a blessed place,
    or have them tweak their CLASSPATH variable.
*)

InstallJava::launch = "The Java runtime could not be launched."
InstallJava::fail = "A link to the Java runtime could not be established."
InstallJava::uifail = "The separate Java user-interface link could not be established."

InstallJava::path =
"The J/Link package appears to be improperly installed. The package file is not in an appropriate location in relation to $Path."

InstallJava::opt = "Warning: unrecognized option in InstallJava."
InstallJava::reinst = "Java is already running. If you need to specify non-standard options to control how Java is launched, use ReinstallJava instead."

Java::init = "Java is not running. You must call InstallJava[] to start the Java runtime."


(* StartJava/QuitJava/RestartJava are synonyms. *)
StartJava = InstallJava
QuitJava = UninstallJava
RestartJava = ReinstallJava

(* I generally don't protect options from redefinition if the package is read in twice, but the options
   to InstallJava are important, and users are encouraged to set them in their init.m if necessary. We
   don't want them to get redefined accidentally.
*)
If[Options[InstallJava] === {},
	Options[InstallJava] = {ClassPath->Automatic, CommandLine->Automatic,
				JVMArguments->None, ForceLaunch->False, Default->Automatic,
                    CreateExtraLinks->Automatic, "Asynchronous"->Automatic}
]

(*
   The InstallJava procedure is split into two parts so as to hide the Java startup
   time as much as possible. To make InstallJava very fast, it now only launches
   Java--it does not try to connect the link. That is done in finishInstall, which also
   performs the other initialization (like set up the extra links). finishInstall is
   called by all "j" functions, so it is guaranteed to have happened before any
   calls into Java are made. It is also done in a one-time periodical task so that the
   link will be connected even if no code explicitly tries to use Java. This is done because
   Java will exit if the link has not been connected before a timeout period elapses (this
   is to prevent Java processes from being left behind if the kernel is launched and
   killed before finishInstall[] is called).

   Because InstallJava is fast, it can be done at kernel startup time. If no code tries to
   use Java before it finishes starting (typically less than a second), then there is
   no perceived delay due to Java startup.

   InstallJava also has the ability to start Java operations running before the link
   is connected, so that other types of Java-side initialization can be performed.
   Use RegisterJavaInitialization to register a Java class to be run while Java is
   launching. See the comments for that function for info.
*)

InstallJava[opts___?OptionQ] :=
	PreemptProtect[
		Module[{existingLink, err, CP, cmdLine, jvmArgs, forceLaunch, default, useExtraLinks,
					link, jvm, initFile, launchTime, installFinished, async},
			{cmdLine, CP, jvmArgs, forceLaunch, default, useExtraLinks, async} =
				{CommandLine, ClassPath, JVMArguments, ForceLaunch, Default, CreateExtraLinks, "Asynchronous"}
                     /. Flatten[{opts}] /. Options[InstallJava];
			If[!forceLaunch,
				(* Bail out right away if link is already open and OK. *)
				jvm = getDefaultJVM[];
				existingLink = javaLinkFromJVM[jvm];
				If[MemberQ[Links[], existingLink],
					(* Only do the check on the health of the link if not already in a call to Java, or
					   in a preemptive transaction arriving from Java. It's obviously not necessary in such
					   cases, but it also can cause the link to die if we call LinkReadyQ on a link when we happen
					   to already be blocking in its yield function in the main computation.
					*)
					If[inPreemptiveCallFromJava[jvm] || Length[$externalCallLinks] > 0,
						Return[existingLink]
					];
					LinkReadyQ[existingLink]; (* Hit link to force LinkError to give current value. *)
					err = First[LinkError[existingLink]];
					(* Error 10 is "deferred connection still unconnected", and we allow it because we allow calls
					   into Java to hang until the link becomes connected by an asynchronous means.
					*)
					If[err === 0 || err === 10,
					    (* Issue a message to users who are calling InstallJava with options to warn them that
					       Java is already running and they must use ReinstallJava instead if they want their
					       options to take effect. But don't count the Asynchronous option, because that is used
					       by the PacletManager, and it isn't a reason to issue this message anyway.
					    *)
					    If[Length[DeleteCases[Flatten[{opts}], HoldPattern["Asynchronous" -> _]]] > 0,
					        Message[InstallJava::reinst]
					    ];
						Return[existingLink],
					(* else *)
						UninstallJava[existingLink]
					],
				(* else *)
					(* This extra test (existingLink has a value, but it is not in Links[]) is to catch cases where
					   user has improperly shut down Java (e.g., by calling LinkClose).
					*)
					If[Head[existingLink] === LinkObject,
						(* Used to issue a warning message here saying that the previous session was improperly shut down.
						   Decided that was more irritating than useful, as it only shows up when you force-quit the old session.
						*)
						resetMathematica[GetJVM[existingLink]]
					]
				]
			];
			If[Length[{filterOptions[InstallJava, Flatten[{opts}]], filterOptions[LinkOpen, Flatten[{opts}]]}] != Length[Flatten[{opts}]],
				Message[InstallJava::opt]
			];
			If[$DebugCommandLine, Print["cmdline = ", createCommandLine[cmdLine, jvmArgs]]];
			cmdLine = createCommandLine[cmdLine, jvmArgs];
			initFile = writeInitFile[CP];
			If[initFile =!= $Failed, cmdLine = cmdLine <> " -init " <> "\"" <> initFile <> "\""];
			link = LinkLaunch[cmdLine, filterOptions[LinkLaunch, opts]];
			launchTime = SessionTime[];
			installFinished = False;
			If[Head[link] === LinkObject,
				jvm = addJVM[createJVMName[], link, Null, Null, initFile, launchTime, installFinished, useExtraLinks];
				If[TrueQ[default] || !forceLaunch && default === Automatic, setDefaultJVM[jvm]];
				If[(TrueQ[async] || (async === Automatic && Head[$ParentLink] === LinkObject)) && !TrueQ[$ForceSynchronousLaunch],
					(* In a linked kernel, arrange for finishInstall[] to be called asynchronously. In a standalone kernel
					   finishInstall will be called when the first call to Java is made.
					*)
					With[{jvmName = nameFromJVM[jvm]}, RunScheduledTask[finishInstallFromTask[jvmName], {1,$connectTimeout}]],
			    (* else *)
					(* Finish synchronously in standalone kernel unless options say otherwise. *)
                    If[Head[finishInstall[jvm]] =!= JVM, link = $Failed]
				];
				link,
			(* else *)
				Message[InstallJava::launch];
				If[TrueQ[default] || !forceLaunch && default === Automatic, setDefaultJVM[Null]];
				$Failed
			]
		]
	]

InstallJava[link_LinkObject, opts___?OptionQ] :=
	PreemptProtect[
		Module[{jvm, default, useExtraLinks},
			{default, useExtraLinks} = {Default, CreateExtraLinks} /. Flatten[{opts}] /. Options[InstallJava];
			(* Infinity because we never want to timeout connecting the link in this version of InstallJava. *)
			jvm = finishInstall[addJVM[createJVMName[], link, Null, Null, $Failed, Infinity, False, useExtraLinks]];
			If[Head[jvm] === JVM,
				If[TrueQ[default] || default === Automatic && getDefaultJVM[] === Null,
					setDefaultJVM[jvm]
				];
				link,
			(* else *)
				$Failed
			]
		]
	]


(* Number of seconds to allow finishInstall to wait for the link to be ready
   to be connected. This interval is the time it takes for Java to start up
   (the kernel and/or front end might be simultaneously busy).
   Defined here as a private variable to allow users in special cases
   to control its value.
*)
$connectTimeout = 20

(* Finishes the InstallJava procedure. Designed to be called separately as part of
   a two-part InstallJava process that hides the Java startup time. Can be called with a timeout
   to support a periodic background testing to see if the link is ready. Returns a JVM expr to 
   indicate that the link is successfully connected, $Failed to indicate that there was a known failure
   and it should not be called again for this link, or Null to indicate that the timeout passed with
   no detected problems, and the function can be called again.
*)
finishInstall[jvm_JVM] := finishInstall[jvm, Infinity]

finishInstall[jvm_JVM, maxWait_] :=
	executionProtect[
		If[installFinishedFromJVM[jvm],
			jvm,
		(* else *)
			Module[{jlink, launchTime, wasAborted = False, old, useExtraLinks, uilink, prelink},
				jlink = javaLinkFromJVM[jvm];
                launchTime = launchTimeFromJVM[jvm];
				(* Here we implement a timeout mechanism for the connect. Enable preemption
				   during this loop.
				*)
				If[!MathLink`IsPreemptive[],
				    old = MathLink`EnablePreemptiveFunctions[True]
				];
				CheckAbort[
					While[!TrueQ[LinkReadyQ[jlink]] && !TrueQ[LinkConnectedQ[jlink]] &&
										SessionTime[] < launchTime + Min[maxWait, $connectTimeout],
						Pause[.05]
					],
					wasAborted = True
				];
                If[!MathLink`IsPreemptive[],
				    MathLink`EnablePreemptiveFunctions[old]
                ];
 				If[wasAborted,
					Abort[],
				(* else *)
					(* This test catches the case where a preemptive Java call during the loop above
					   called finishInstall[]. If that happened, then when we get here the install has
					   been completed already and installFinished will have been set to True.
					*)
					If[installFinishedFromJVM[jvm],
						Return[jvm]
					];
					If[!TrueQ[LinkReadyQ[jlink]] && !TrueQ[LinkConnectedQ[jlink]],
                        (* Fail if we have waited past the timeout and the link is still not ready to connect. *)
					    If[SessionTime[] >= launchTime + $connectTimeout,
						    quietLinkClose[jlink];
						    Message[InstallJava::fail];
						    removeJVM[jvm];
						    Return[$Failed],
						(* else *)
						    (* The link is not ready yet, but the global connect timeout has not passed. Safe to call again. *)
						    Return[Null]
					    ]
					];
					(* This will not block, since LinkReadyQ or LinkConnectedQ is True. *)
					jlink = LinkConnect[jlink];
					(* Need to set this here, as we make calls into Java during initJava[]. *)
					setJVMInstallFinished[jvm];

					(* If no initFile was used, manually set the starting classpath. *)
                    If[!StringQ[initFileFromJVM[jvm]],
                        Function[{cpEntry, searchForJars},
                                jAddToClassPath[jvm, {cpEntry}, searchForJars, False];
                        ] @@@ buildStartingClassPath[Automatic]
                    ];

					useExtraLinks = useExtraLinksFromJVM[jvm];
					If[useExtraLinks === Automatic,
						(* Don't want to set up uiLink and preLink if we are calling InstallJava[$ParentLink] from Java (this happens
						   during KernelLink.enableObjectReferences()). In that case there won't be a Reader thread, so
						   we don't want the extra links. Also, don't set up extra links if we are connecting to a remote
						   JVM, which we detect by looking for an "@" char in the name.
						*)
						useExtraLinks = isPreemptiveKernel[] && ValueQ[$ParentLink] && $ParentLink =!= Null && jlink =!= $ParentLink &&
						                     Head[jlink] === LinkObject && !StringMatchQ[First[jlink], "*\\@*"];
					];
					If[Head[jlink] === LinkObject,
						MathLink`LinkAddInterruptMessageHandler[jlink];
						{uilink, prelink} = initJava[jvm, useExtraLinks]
			        ];
					If[Head[jlink] === LinkObject && ((Head[uilink] === LinkObject && Head[prelink] === LinkObject) || !useExtraLinks),
						setJVMExtraLinks[jvm, uilink, prelink];
						jvm,
					(* else *)
						Message[InstallJava::fail];
						If[Head[jlink] === LinkObject, quietLinkClose[jlink]];
						If[Head[uilink] === LinkObject, quietLinkClose[uilink]];
						If[Head[prelink] === LinkObject, quietLinkClose[prelink]];
						removeJVM[jvm];
						Null
					]
				]
			]
		]
	]

(* Called as a one-time periodical to call finishInstall if no direct user call does it.
   We need to arrange for this to happen because otherwise Java will eventually quit
   (so as to prevent Java processes from being left around if kernel exits before link is connected).
*)
finishInstallFromTask[jvmName_String] :=
	Module[{jvm = GetJVM[jvmName]},
		Quiet @ If[Head[finishInstall[jvm, .1]] === JVM, StopScheduledTask[$ScheduledTask]; RemoveScheduledTask[$ScheduledTask]];
	]


UninstallJava[] :=
	Module[{jvm = getDefaultJVM[]},
		If[Head[jvm] === JVM,
			UninstallJava[jvm],
		(* else *)
			Null
		]
	]

UninstallJava[jlink_LinkObject] := UninstallJava[GetJVM[jlink]]

UninstallJava[jvm_JVM] :=
	PreemptProtect[
		Module[{jlink, uilink, prelink},
			finishInstall[jvm];
			jlink = javaLinkFromJVM[jvm];
			(* To avoid potentially many errors, only call onUnloadClass methods if jlink is alive and well. *)
			If[MemberQ[Links[], jlink],
				LinkReadyQ[jlink]; (* Hit link to force LinkError to give current value. *)
				If[First[LinkError[jlink]] === 0,
					callAllUnloadClassMethods[jvm]
				]
			];
			UnshareFrontEnd[jlink];
			If[MemberQ[SharingLinks[], jlink],
				UnshareKernel[jink]
			];
			resetMathematica[jvm];
			quietLinkClose[jlink];
			uilink = javaUILinkFromJVM[jvm];
            If[Head[uilink] === LinkObject,
				MathLink`RemoveSharingLink[uilink];
                quietLinkClose[uilink]
            ];
			prelink = javaPreemptiveLinkFromJVM[jvm];
            If[Head[prelink] === LinkObject,
                quietLinkClose[prelink]
            ];
            If[jvm === getDefaultJVM[],
            	setDefaultJVM[Null]
            ];
            removeJVM[jvm]
		];
	]


Options[ReinstallJava] = Options[InstallJava]

ReinstallJava[args___] := PreemptProtect[UninstallJava[]; InstallJava[args]]


(* RegisterJavaInitialization is no longer supported (starting in M10. Using it has no effect. *)
(* Register some Java initialization to occur while Java is launching. Public, but not
   documented for users. Will be used by internal code (such as the PacletManager).
   Argument strings must fit a precise form, starting with s single keyword, followed
   by a single space char and space-separated arguments:
        cp some/dir/or/jar/file
        cpf some/dir           (cpf means don't search for jars in the dir)
        run ClassNameHavingAMainMethod arg1ToMain arg2ToMain ...
   In a "run" line, the args must not have spaces in them. If you need to have spaces
   (such as in file paths), convert them to %20.
   You can also pass a held expression to RegisterJavaInitialization, in which case it will
   be evaluated when it is used and should return a string in the above form.
*)
RegisterJavaInitialization[s_String] :=
    PreemptProtect[
        (* Hold is used only because we need a wrapper that will be stripped later, and expressions
           come wrapped in Hold, so might as well use the same wrapper for strings.
        *)
        If[!MemberQ[$javaInit, Hold[s]], AppendTo[$javaInit, Hold[s]]];
    ]

RegisterJavaInitialization[expr:(_Hold | _HoldForm | _HoldComplete)] :=
    PreemptProtect[
        If[!MemberQ[$javaInit, expr], AppendTo[$javaInit, expr]];
    ]

If[!ValueQ[$javaInit], $javaInit = {}]



JavaLink[] := JavaLink[getDefaultJVM[]]
JavaLink[jvm_JVM] := javaLinkFromJVM[jvm]
JavaLink[Null] = Null

JavaUILink[] := JavaUILink[getDefaultJVM[]]
JavaUILink[jvm_JVM] := javaUILinkFromJVM[jvm]
JavaUILink[Null] = Null

javaPreemptiveLink[] := javaPreemptiveLink[getDefaultJVM[]]
javaPreemptiveLink[jvm_JVM] := javaPreemptiveLinkFromJVM[jvm]
javaPreemptiveLink[Null] = Null

(* Future feature. *)
$useSandboxSecurity = False

(* $jlinkExtraReadDirs and $jlinkExtraWriteDirs are lists of dirs that are made accessible in the sandbox, beyond the standard ones
   that are based on the user (e.g., $UserBaseDirectory, etc.) We can add to this list as we discover bits of Java
   functionality that we want to allow that requires read or write access to an assorted set of extra locations. The ones below
   are used by JDBC on Linux.
*)
If[$useSandboxSecurity && !MemberQ[Attributes[$jlinkExtraReadDirs], Locked],
    $jlinkExtraReadDirs = {"/dev/random", "/dev/urandom"};
    $jlinkExtraWriteDirs = {"/dev/random", "/dev/urandom"};
    SetAttributes[$jlinkExtraReadDirs, {Protected, ReadProtected, Locked}];
    SetAttributes[$jlinkExtraWriteDirs, {Protected, ReadProtected, Locked}]
]

createCommandLine[cmdLine_, jvmArgs_] :=
	Module[{jlinkPath, cpSpec, prefsSpec, sysLoaderSpec, javaCmd, extraArgs, quoteChar},
		jlinkPath = ToFileName[$jlinkDir, "JLink.jar"];
		javaCmd =
			Which[
				StringQ[cmdLine],
					cmdLine,
				StringQ[Environment["WRI_JAVA_HOME"]] && DirectoryQ[Environment["WRI_JAVA_HOME"]],
					ToFileName[{Environment["WRI_JAVA_HOME"], "bin"}, If[osIsWindows[], "javaw.exe", "java"]],
				$SystemID == "Windows" || $SystemID == "Windows-x86-64",
					If[FileExistsQ[#], #, "javaw.exe"]& @
						ToFileName[{$InstallationDirectory, "SystemFiles", "Java", $SystemID, "bin"}, "javaw.exe"],
				osIsMacOSX[],
					ToFileName[{getJLinkAppDir[], "JLink.app", "Contents", "MacOS"}, "Launcher"],
				True,
					(* Fallthrough for UNIX. *)
					If[FileExistsQ[#], #, "java"]& @
						ToFileName[{$InstallationDirectory, "SystemFiles", "Java", $SystemID, "bin"}, "java"]
			];
	    (* If user did not manually specify cmdLine, quote it to accommodate spaces. *)
	    quoteChar = If[$OperatingSystem === "Windows", "\"", "'"];
		If[!StringQ[cmdLine], javaCmd = quoteChar <> javaCmd <> quoteChar];
		(* The command-line classpath spec points only at JLink.jar. All other class locations (including from CLASSPATH)
		   are specified later. Don't include these two args in the Mac command line, because they are supplied via
		   the Info.plist file.
		*)
		If[osIsMacOSX[] && !StringQ[cmdLine],
			cpSpec = "";
			sysLoaderSpec = "",
		(* else *)
			cpSpec = " -classpath \"" <> jlinkPath <> "\"";
			sysLoaderSpec = " -Djava.system.class.loader=com.wolfram.jlink.JLinkSystemClassLoader"
		];
		(* Disabling the Java prefs subsystem on Unix/Linux is a hack to work around a very annoying problem with
		   that subsystem. Maybe Sun will fix this in JDK 1.5. Because this is an experimental fix, we'll put in
		   the $disablePrefs flag as a backdoor that could be set from top level before launching Java.
		*)
		prefsSpec =
			If[!osIsWindows[] && !osIsMacOSX[] && $disablePrefs =!= False,
				" -Djava.util.prefs.PreferencesFactory=com.wolfram.jlink.DisabledPreferencesFactory",
			(* else *)
				""
			];
		extraArgs = If[StringQ[jvmArgs], " " <> jvmArgs, ""];
		If[$useSandboxSecurity && Developer`$ProtectedMode && !$CloudEvaluation, (* cloud handles its own security *)
		    extraArgs = " -Dcom.wolfram.jlink.security=com.wolfram.jlink.JLinkSandboxSecurityManager" <> extraArgs
		];
		If[osIsMacOSX[] && StringQ[cmdLine], extraArgs = extraArgs <> " -Xdock:name=J/Link"];
		(* Increase default max heap from 64 Mb to 256 Mb, but only if caller has not specified another value. *)
		If[!StringMatchQ[extraArgs, "*-xmx*", IgnoreCase->True] && !StringMatchQ[extraArgs, "*AggressiveHeap*", IgnoreCase->True],
			extraArgs = extraArgs <> " -Xmx512m "
		];
		javaCmd <> cpSpec <> extraArgs <> sysLoaderSpec <> prefsSpec <> " com.wolfram.jlink.Install"
	]


(* Creates the init file read by Java at startup. This is a means to pass information
   to Java before the link is connected.
*)
writeInitFile[cpOpt_] :=
	Module[{initFile},
		Quiet[
		    initFile = OpenTemporary[CharacterEncoding->"UTF8"]
		];
		If[Head[initFile] === OutputStream,
			Function[{cpEntry, searchForJars},
				WriteString[initFile, If[searchForJars, "cp ", "cpf "] <> cpEntry <> "\n"]
			] @@@ buildStartingClassPath[cpOpt];
			WriteString[initFile, # <> "\n"]& @@@ $javaInit;
			Close[initFile],
		(* else *)
		    (* One scenario where OpenTemporary can fail is when running in low-privilege mode on Windows as plug-in.
		       No problem--just skip the initfile-based startup optimization.
		    *)
		    $Failed
		]
	]


initJava[jvm_JVM, setupExtraLinks:(True | False)] :=
    Module[{uilink, prelink, prot, uiLinkName, preLinkName, linkSnooperCmdLine, allowedReadDirs, allowedWriteDirs},
    	jSetVMName[jvm, nameFromJVM[jvm]];
    	jSetUserDir[jvm, $HomeDirectory];
    	(* This is a hook that allows clients to execute M code during InstallJava. Used by Cloud Platform. *)
    	ReleaseHold[$jlinkInit];
        If[$useSandboxSecurity && Developer`$ProtectedMode && !$CloudEvaluation, (* cloud handles its own security *)
            allowedReadDirs = Flatten[{$BaseDirectory, $UserBaseDirectory, PacletManager`$UserBasePacletsDirectory, $TemporaryDirectory} ~Join~ $jlinkExtraReadDirs];
            allowedWriteDirs = Flatten[{$TemporaryDirectory} ~Join~ $jlinkExtraWriteDirs];
            LoadJavaClass["com.wolfram.jlink.JLinkSandboxSecurityManager"];
            JLinkSandboxSecurityManager`setAllowedDirectories[allowedReadDirs, allowedWriteDirs]
        ];
        If[isPreemptiveKernel[] && setupExtraLinks,
	        (* Set up the UI link. $UILinkProtocol exists as a backdoor for users who need to
	           force a particular protocol (e.g., TCP to avoid problems with TCPIP).
			*)
	        prot =
				Which[
	        		StringQ[$UILinkProtocol], $UILinkProtocol,
	        		osIsWindows[] || $VersionNumber >= 6.0, "SharedMemory",
					True, "TCPIP"
				];
	        uilink = LinkCreate[LinkProtocol->prot];
	        prelink = LinkCreate[LinkProtocol->prot];
	        (* On OS/X can have problems if name includes "@localhost", so remove it. See bug 58268. *)
	        uiLinkName = StringReplace[First[uilink], "@localhost" -> ""];
	        preLinkName = StringReplace[First[prelink], "@localhost" -> ""];
	        If[TrueQ[$UseLinkSnooper],
	        	linkSnooperCmdLine = StringReplace[createCommandLine[Null, Null],
						"com.wolfram.jlink.Install" -> "com.wolfram.jlink.util.LinkSnooper"],
	        (* else *)
	        	linkSnooperCmdLine = ""
	        ];
	        If[TrueQ[jExtraLinks[jvm, uiLinkName, preLinkName, prot, linkSnooperCmdLine]],
	            LinkConnect[uilink];
	            LinkConnect[prelink];
				MathLink`AddSharingLink[uilink,
						MathLink`LinkSwitchPre -> linkSwitchPreFunc,
						MathLink`LinkSwitchPost -> linkSwitchPostFunc,
						MathLink`AllowPreemptive -> True,
						MathLink`ImmediateStart -> True
				];
				MathLink`LinkAddInterruptMessageHandler[uilink];
				(* UI link is a daemon link, meaning that kernel should not stay alive just for it. *)
				MathLink`SetDaemon[uilink, True];
	            {uilink, prelink},
	        (* else *)
	            Message[InstallJava::uifail];
	            quietLinkClose[uilink];
	            quietLinkClose[prelink];
	            {Null, Null}
	        ],
	    (* else *)
	    	(* Version 5.0 or earlier--no UI Link *)
	    	{Null, Null}
	    ]
	]


buildStartingClassPath[cpOpt_] :=
	Module[{cp = {}},
	    (* The first thing we add is the previous classpath, if this is a relaunch of Java within a session.
	       This is what allows the entries added with AddToClassPath[] to be sticky across restarts of Java.
	       Note that we set the "searchForJars" parameter to be False, as any dirs have already had their
	       component jar files added, if appropriate, when the dirs were first added.
	    *)
	    If[MatchQ[$currentClassPath, {__String}], cp = Join[cp, {#, False}& /@ $currentClassPath]];
		cp = Join[cp, {#, True}& /@ autoClassPath[]];
		(* Add CLASSPATH variable _after_ auto class path. We must add CLASSPATH manually since we are using the
		   -classpath command-line option to point solely at JLink.jar.
		*)
		If[cpOpt === Automatic && StringQ[Environment["CLASSPATH"]],
			(* False for "don't search for jars in dirs". *)
			cp = Join[cp, {#, False}& /@ splitClasspath[Environment["CLASSPATH"]]]
		];
		(* Here we add the contents of the user-specified ClassPath option. *)
		If[StringQ[cpOpt],
			cp = Join[cp, {#, True}& /@ splitClasspath[cpOpt]]
		];
		cp = Join[cp, {#, True}& /@ $ExtraClassPath];
		(* Now we add our bundled tools.jar. We put it at the end on the off chance that a user might want to
		   have another source for these classes loaded earlier. Tools.jar contains the Java compiler, among other things.
		*)
		AppendTo[cp, {ToFileName[{$InstallationDirectory, "SystemFiles", "Java", $SystemID, "lib"}, "tools.jar"], False}];
		deleteDuplicates[DeleteCases[cp, {}]]
	]


inPreemptiveCallFromJava[_] = False

(* In M 6.0 and later, the 2nd arg to this func tells whether this call is preemptive or not,
   but for 5.x compatibility I won't use it, and instead call MathLink`IsPreemptive[] instead.
*)
linkSwitchPreFunc[link_, ___] :=
	Block[{jvm, uiLink, oldFrontEnd, oldFormatType, oldEndDlgPktLink, res},
		jvm = GetJVM[link];
		uiLink = JavaUILink[jvm];
		If[MathLink`IsPreemptive[], inPreemptiveCallFromJava[jvm] = True];
		oldFormatType = FormatType /. Options["stdout"];
		If[!hasServiceFrontEnd[], SetOptions["stdout", FormatType->OutputForm]];
		oldFrontEnd =
			If[hasServiceFrontEnd[],
				MathLink`SetServiceFrontEnd[],
			(* else *)
				If[FrontEndSharedQ[JavaLink[jvm]],
					(* Note that for legacy reasons, users call ShareFrontEnd[JavaLink[]], but really
					   it is the JavaUILink[] that the FE-specific traffic goes out on.
					*)
					MathLink`SetFrontEnd[uiLink],
				(* else *)
					res = MathLink`SetFrontEnd[Null];
					MathLink`SetMessageLink[uiLink];
					res
				]
			];
		If[First[oldFrontEnd] === False,
			(* There was no ServiceLink, and the MessageLink was set to Null.
			   FE services won't work, but at least we can set the MessageLink
			   to the activeJavaLink, so that side-effect output will come to
			   Java and not get completely lost.
			*)
			MathLink`SetMessageLink[uiLink]
		];
		oldEndDlgPktLink = MathLink`$EndDialogPacketLink;
		MathLink`$EndDialogPacketLink = Null;
		{jvm, oldFrontEnd, oldFormatType, oldEndDlgPktLink}
	]

linkSwitchPostFunc[{jvm_, oldFrontEnd_, oldFormatType_, oldEndDlgPktLink_}] :=
	(
		MathLink`RestoreFrontEnd[oldFrontEnd];
		SetOptions["stdout", FormatType->oldFormatType];
		MathLink`$EndDialogPacketLink = oldEndDlgPktLink;
		inPreemptiveCallFromJava[jvm] = False;
	)

(* Determines the default automatic set of extra directories to search for classes. Looks for Java subdirectories of
   any of several standard application directories. Also in paclets.
*)
autoClassPath[] :=
	Module[{appPaths, appDirs, javaDirs, pacletJavaResources, appJavaResources, pacletNames},
		appPaths = {ToFileName[{$InstallationDirectory, "SystemFiles", "Links"}],
					ToFileName[{$InstallationDirectory, "AddOns", "Applications"}],
					ToFileName[{$InstallationDirectory, "AddOns", "ExtraPackages"}],
					ToFileName[{$InstallationDirectory, "AddOns", "Autoload"}]};
		If[!Developer`$ProtectedMode || $CloudEvaluation,
			appPaths =
				{ToFileName[{$UserAddOnsDirectory, "Applications"}],
				 ToFileName[{$UserAddOnsDirectory, "AutoLoad"}],
				 ToFileName[{$AddOnsDirectory, "Applications"}],
				 ToFileName[{$AddOnsDirectory, "Autoload"}]} ~Join~ appPaths
		];
		appDirs = Select[Flatten[FileNames["*", #]& /@ appPaths], DirectoryQ];
		(* Now append $AddOns and $UserAddOns, to allow Java subdirs to be found (i.e., allow
		   $UserAddOnsDirectory/Java, not just $UserAddOnsDirectory/Applications/SomeApp/Java).
		   Append instead of prepend, to give applications primacy.
		*)
		If[!Developer`$ProtectedMode || $CloudEvaluation,
			If[StringQ[$AddOnsDirectory],
				appDirs = appDirs ~Join~ {$AddOnsDirectory, $UserAddOnsDirectory}
			]
		];
		(* This gets {{"/path/to/Java", "AppName"}...} *)
		appJavaResources = Select[{ToFileName[{#, "Java"}], FileNameTake[#]}& /@ appDirs, DirectoryQ[First[#]]&];
		(* This gets: {{"/path/to/Java", "PacletName"}, {"/path/to/Java", "PacletName"}, ...} where
           PacletName can be repeated (a paclet can provide more than one resource item). The Cases call just
           allows the result to be ignored if PacletManager has not been loaded (to support -nopaclet operation).
           This should be switched over to the new public PacletResources[] function, but for now we want to support newer JLink in an older PM. 
        *)
        pacletJavaResources = Cases[PacletManager`Package`resourcesLocate["Java"], {_String, _String}];
        (* Merge these lists by dropping all resources found by the "app" method that match the paclet name of a
           resource found by the paclet method.
        *)
        pacletNames = DeleteDuplicates[Last /@ pacletJavaResources];
        javaDirs = First /@ Join[Select[appJavaResources, !MemberQ[pacletNames, Last[#]]&], pacletJavaResources];

		(* Add some special dirs in the layout. *)
		PrependTo[javaDirs, ToFileName[{$InstallationDirectory, "AddOns", "Packages", "GUIKit", "Java"}]];
		PrependTo[javaDirs, ToFileName[{$InstallationDirectory, "SystemFiles", "Converters", "Java"}]];
		PrependTo[javaDirs, ToFileName[{$InstallationDirectory, "SystemFiles", "Autoload", "PacletManager", "Java"}]];
		(* Here we put the special WRI SystemFiles/Java dir first. The thinking for putting it absolutely first
		   in the search path is that it becomes a convenient place to put classes and be sure they override any
		   provided by applications. It can be used to resolve application conflicts in this way.
		*)
		PrependTo[javaDirs, ToFileName[{$InstallationDirectory, "SystemFiles", "Java"}]];
		javaDirs
	]


(* The Mathematica-side things that must be done when starting a fresh Java session. After this func is run, it should
    be true that the kernel is in the same state it was in before any Java sessions were started. The possible exception
    to this is that certain contexts may exist that weren't present before, but their contents should be empty. This function
    is only used during InstallJava; it is not user-visible, and no attempt is made to provide users with a way to
    "reset" their Mathematica-Java session.
*)
resetMathematica[jvm_JVM] :=
	(
		If[jvm === getDefaultJVM[],
			setDefaultJVM[Null]
		];
		clearObjectDefs[jvm];
		unregisterAllWindows["Java", jvm];
	)


(* Splits a platform-specific classpath specification into a list of strings, one for each component. *)
splitClasspath[cp_String] :=
	Module[{cpSep, str, result},
		cpSep = If[osIsWindows[], ";", ":"];
		strm = StringToStream[cp];
		result = ReadList[strm, Word, WordSeparators->cpSep, NullWords->False];
		Close[strm];
		result
	]


(* Users (e.g., the Workbench) can set $jlinkAppDir to be a special location for JLink.app on OS/X. *)
getJLinkAppDir[] := If[ValueQ[$jlinkAppDir], $jlinkAppDir, $jlinkDir]


(****************************)

AsyncInstall[] :=
	With[{loop = LinkOpen[LinkMode->Loopback]},
		(* Needs["JLink`"]; *)
		LinkWriteHeld[loop, Hold[$InternalLink = LinkCreate[]]];
		LinkWriteHeld[loop, Hold[InstallToInternalLink[$InternalLink]]];
		LinkWriteHeld[loop, Hold[InstallJava[$InternalLink]]];
		LinkWriteHeld[loop, Hold[LoadJavaClass["java.lang.Object"]]];
		LinkWriteHeld[loop, Hold[LoadJavaClass["java.awt.Component"]]];
		LinkWriteHeld[loop, Hold[LoadJavaClass["com.wolfram.guikit.GUIKitDriver"]]];
		LinkWriteHeld[loop, Hold[LoadJavaClass["com.wolfram.bsf.engines.MathematicaBSFEngine"]]];
		LinkWriteHeld[loop, Hold[LoadJavaClass["org.apache.bsf.util.BSFEngineImpl"]]];
		LinkWriteHeld[loop, Hold[LoadJavaClass["com.wolfram.bsf.util.MathematicaBSFFunctions"]]];
		LinkWriteHeld[loop, Hold[LoadJavaClass["java.io.FileInputStream"]]];
		LinkWriteHeld[loop, Hold[LoadJavaClass["java.util.jar.Manifest"]]];
		LinkWriteHeld[loop, Hold[LoadJavaClass["java.util.jar.Attributes"]]];
		LinkWriteHeld[loop, Hold[LoadJavaClass["com.wolfram.guikit.swing.GUIKitJFrame"]]];
		(* TODO: Move setlookandfeel in here. *)
		LinkWriteHeld[loop, Hold[LinkClose[loop]]];
		MathLink`AddSharingLink[loop];
	]

InstallJavaInternalDEVICE[] :=
	Module[{},
		$InternalLink = LinkCreate[];
		Java`InstallToInternalLink[$InternalLink];
		InstallJava[$InternalLink]
	]

InstallJavaInternalLOOPBACK[] :=
	Module[{res},
		$InternalLink = LinkOpen[LinkMode->Loopback];
		res = Java`InstallToInternalLink[$InternalLink];
		If[res === $Failed,
			Message[InstallJava::fail];
			Return[$Failed]
		];
		Block[{jlinkDefineExternal = jlinkDefineInternal},
			Install[$InternalLink]
		];
		JLink`InstallJava`Private`$jlink = $InternalLink
	]

(* This one gets used--swap to eiher LOOPBACK or DEVICE version. *)
InstallJavaInternal[] :=
	Module[{loop, res},
		loop = LinkOpen[LinkMode->Loopback];
		res = Java`InstallToInternalLink[loop];
		If[res === $Failed,
			Message[InstallJava::fail];
			Return[$Failed]
		];
		(**********
		Block[{jlinkDefineExternal = jlinkDefineInternal},
			Install[$InternalLink]
		];
		JLink`InstallJava`Private`$jlink = $InternalLink
		***********)
		$InternalLink = loop;
		If[True,  (* Full treatment *)
			InstallJava[loop],
		(* else *)  (* leaves out uilink, etc., for debugging. *)
			Install[loop];
			JLink`InstallJava`Private`$jlink = $InternalLink
		]
	]

jlinkDefineInternal[p_String, a_, n_] :=
	Module[{e, pat = ToHeldExpression[p], args = ToHeldExpression[a]},
		e = Hold[_ := jlinkInternalCall[$InternalLink, CallPacket[_, _]]];
		e = ReplaceHeldPart[e, pat, {1, 1}];
		e = ReplacePart[e, n, {1, 2, 2, 1}];
		e = ReplaceHeldPart[e, args, {1, 2, 2, 2}];
		ReleaseHold[e];
	]


(* TODO: Not sure about using AbortProtect here as in jlinkDefineExternal. That's a big issue... *)
jlinkInternalCall[link_LinkObject, packet_CallPacket] :=
	Block[{ThisLink = link, $CurrentLink = link, pkt = packet, res},
		While[True,
			If[LinkWrite[link, pkt] === $Failed, Return[$Failed]];
			Java`DispatchToJava[link];
			res = LinkReadHeld[link];
			Switch[res,
				Hold[EvaluatePacket[_]],
					(* Re-enable aborts during the computation in Mathematica of EvaluatePacket contents, but have
					   them just cause $Aborted to be returned to Java, not call Abort[].
					*)
					pkt = ReturnPacket[CheckAbort[res[[1,1]], $Aborted]],
				Hold[ReturnPacket[_]],
					Return[res[[1,1]]],
				Hold[_],
					Return[res[[1]]],
				_,
					Return[res]
			]
		]
	]

(********************************)
(* Include private defs here to avoid reliance on the Utilities`FilterOptions` standard package (which is
   not included in a Minimal Install.
*)
filterOptions[command_Symbol, options___] := filterOptions[First /@ Options[command], options]
filterOptions[opts_List, options___] := Sequence @@ Select[Flatten[{options}], MemberQ[opts, First[#]]&]


(* Because the PacletManager's def of Documentation`CreateMessageLink calls (and thus launches) Java, we
   need to avoid having messages issued at certain sensitive times during Java startup, during which reentrancy
   is not safe. The only message I know of that can be issued in these sensitive blocks is from LinkClose.
   Therefore we quiet it with this utility function. This also avoids users seeing an ugly message when they
   call ReinstallJava[] after Java crashes or is killed. This is a fix for 142578. *)
quietLinkClose[link_] := Quiet[LinkClose[link]]


End[]

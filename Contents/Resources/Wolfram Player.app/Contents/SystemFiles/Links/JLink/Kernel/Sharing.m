(* :Title: Sharing.m *)

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
   ShareKernel, ShareFrontEnd, and Periodicals.

   This file is a component of the J/Link Mathematica source code.
   It is not a public API, and should never be loaded directly by users or programmers.

   J/Link uses a special system wherein one package context (JLink`) has its implementation
   split among a number of .m files. Each component file has its own private context, and also
   potentially introduces public symbols (in the JLink` context) and so-called "package" symbols,
   where the term "package" comes from Java terminology, referring to symbols that are visible
   everywhere within the implementation of J/Link, but not to clients.
*)

(*
   Note that to avoid triggering SystemStub-based loading of FrontEnd-specific code when this package
   is loaded, we replace direct refs to FE symbols with Symbol["Name"]. This makes the package load
   much faster, which is desired in a standalone kernel.
*)

ShareKernel::usage =
"ShareKernel[] initiates sharing of the kernel between the front end and a link to another program such as Java or .NET. You typically call ShareKernel to allow the kernel to respond to Java- or .NET-based user interfaces. For backwards compatibility, calling ShareKernel[] with no arguments initiates sharing with the Java link, but new code should specify the intended link, either by calling ShareKernel[JavaLink[]] or ShareKernel[NETLink[]]. If the specified link is already sharing the kernel, ShareKernel does nothing. ShareKernel returns a token that you should save and later pass to UnshareKernel to unregister the request for kernel sharing. ShareKernel is unnecessary in Mathematica 5.1 and later because the kernel is always ready for computations that originate in Java or .NET, even when busy with another computation."

UnshareKernel::usage =
"UnshareKernel[token] is used to \"unregister\" an earlier call to ShareKernel that returned the given token. When all calls to ShareKernel have had a corresponding call to UnshareKernel, kernel sharing is turned off and the front end returns to its normal state of having full control of the kernel. Turning off kernel sharing also stops execution of any periodical tasks established with AddPeriodical. UnshareKernel[linkobject] turns off all sharing of the kernel with the given link, no matter how many times ShareKernel has been called for that link (this form should only be used during development, not in production code). For backwards compatibility, UnshareKernel[] is equivalent to UnshareKernel[JavaLink[]], but new code should always include the link argument."

KernelSharedQ::usage =
"KernelSharedQ[] returns True if the kernel is currently being shared among two or more links, False otherwise. KernelSharedQ[link] returns True if the kernel is currently being shared with the specified link, False otherwise."

SharingLinks::usage =
"SharingLinks[] returns a list of the links currently sharing the kernel."

SharingPrompt::usage =
"SharingPrompt is a deprecated option to ShareKernel."

AddPeriodical::usage =
"AddPeriodical[expr, interval] adds the computation expr to the set of operations that are periodically performed automatically. It returns an integer ID that can be used in RemovePeriodical. The time interval is specified in seconds, and can be less than one second, although the timing is not expected to be very precise."

RemovePeriodical::usage =
"RemovePeriodical[id] removes the computation corresponding to the integer id from the set of operations that are periodically performed automatically. The id number must be one returned by AddPeriodical or Periodicals."

Periodical::usage =
"Periodical[id] returns information about the periodical task corresponding to the specified integer id. It returns a list with the form {id, code, seconds}. The id number must be one returned by AddPeriodical or Periodicals. "

Periodicals::usage =
"Periodicals[] returns a list of integer id numbers corresponding to the set of operations that are periodically performed automatically when the kernel is not busy with another computation."

SetPeriodicalInterval::usage =
"SetPeriodicalInterval[id, interval] resets the time interval for the periodical task with the given id."

$ThisPeriodical::usage =
"$ThisPeriodical is a variable that holds the id of the currently-executing periodical task. You can use $ThisPeriodical from within your periodical task to alter its time interval for future calls (by calling SetPeriodicalInterval) or to remove it entirely (with RemovePeriodical)."

ShareFrontEnd::usage =
"ShareFrontEnd[link] initiates sharing of the notebook front end with an external program such as Java or .NET. For backwards compatibility, calling ShareFrontEnd[] with no arguments initiates sharing with the Java link, but new code should specify the intended link, either by calling ShareFrontEnd[JavaLink[]] or ShareFrontEnd[NETLink[]]. If the front end is already being shared by that link, ShareFrontEnd does nothing. ShareFrontEnd returns a token that you should save and later pass to UnshareFrontEnd to unregister the request for front end sharing."

UnshareFrontEnd::usage =
"UnshareFrontEnd[token] is used to \"unregister\" an earlier call to ShareFrontEnd that returned the given token. When all calls to ShareFrontEnd have had a corresponding call to UnshareFrontEnd, front end sharing is turned off. UnshareFrontEnd[linkobject] turns off sharing of the front end with the given link. For backwards compatibility, calling UnshareFrontEnd[] with no arguments turns off sharing with the Java link (thus it is equivalent to UnshareFrontEnd[JavaLink[]]), but programmers should always use UnshareFrontEnd[token], passing the token returned from a previous call to ShareFrontEnd."

FrontEndSharedQ::usage =
"FrontEndSharedQ[link] returns True if the front end is being shared with specified link, False otherwise. For backwards compatibility, calling FrontEndSharedQ[] with no arguments defaults to the Java link (thus it is equivalent to FrontEndSharedQ[JavaLink[]])."


Begin["`Package`"]

myLinkReadyQ

End[]


(* Current context will be JLink`. *)

Begin["`Sharing`Private`"]


(********************************************  ShareKernel  *************************************************)

(* Note that ShareKernel can probably only be correctly executed inside an EnterTextPacket or EnterExpressionPacket,
   not an EvaluatePacket. This is normally the case, as it is typically run from the notebook front end. If another program
   wants to start up sharing, it should wrap the call in EnterTextPacket, not EvaluatePacket.
*)

ShareKernel::nofe = "ShareKernel cannot be called unless the kernel is linked to a front end program of some sort."
ShareKernel::java = "ShareKernel cannot be called with zero arguments unless Java has been launched by InstallJava."

(* Support for the deprecated Prompt option. *)
Prompt = SharingPrompt

Options[ShareKernel] = {SharingPrompt->"(sharing) "}

(* If you give ShareKernel 0 args, use JavaLink[].
   1 arg, it is the non-3.0 FE, non $ParentLink
   2+ args, first is 3.0 FE, $ParentLink
   Or, go manual with a list of pairs.
*)

ShareKernel[opts___?OptionQ] :=
	If[Head[JavaLink[]] === LinkObject,
		(* Note the call to InstallJava. We don't want ShareKernel to launch Java (it is a general design decision to
		   force users to call InstallJava[] manually), but we want to guard against cases where there is a JavaLink[],
		   but Java is not running (e.g., the user killed Java but has not yet called UninstallJava[]) because the kernel
		   can hang if it goes into sharing mode and Java is not cooperating. Once we have passed the test above that
		   JavaLink[] is defined, we make sure that Java is indeed operating normally by calling InstallJava[].
		*)
		ShareKernel[InstallJava[], opts],
	(* else *)
		Message[ShareKernel::java];
		$Failed
	]

ShareKernel[inst_LinkObject, opts___?OptionQ] := ShareKernel[{$ParentLink, inst}, opts]

ShareKernel[links__LinkObject, opts___?OptionQ] := ShareKernel[{links}, opts]

ShareKernel[links:{__LinkObject}, opts___?OptionQ] := ShareKernel[Join[{{First[links], True}}, {#, False}& /@ Rest[links]], opts]

ShareKernel[pairs:{{_LinkObject, True|False}..}, opts___?OptionQ] :=
	Module[{prompt, theseFes, pos, tok, matchingRecPosition, netLink},
		If[isPreemptiveKernel[],
			Return[preemptiveShareKernel[pairs, opts]]
		];
		tok = $nextShareKernelToken++;
		prompt = contextIndependentOptions[SharingPrompt, {opts}, Options[ShareKernel]];
		If[StringQ[prompt], $prompt = prompt];
		theseFes = Apply[createLinkInfo, Append[#, {tok}]& /@ pairs, {1}];
		netLink = NETLink`NETLink[];
		If[Head[netLink] === LinkObject && !FreeQ[pairs, netLink],
		    If[$VersionNumber < 5.0,
		        (* We need Java for yielding in M 4.2 and earlier. *)
		        InstallJava[]
		    ];
			(* We need to let .NET/Link know if sharing is being turned on for that link. *)
			NETLink`Package`nShareKernel[True]
		];
		If[KernelSharedQ[],
			(* Already sharing. Append any new fes to current list. If a link is already there, replace it with the new
			   linkInfo--this is a means of changing the isFE state of a link (but only allow notFE --> isFE, not "dumbing down"
			   of the isFE state. That would allow someone to turn off front end sharing behavior just by calling ShareKernel[]
			   again for that link). Also, lump together token lists.
			*)
			Function[{newLinkInfoRec},
				matchingRecPosition = Flatten @ Position[$fes, _?(getLink[#] === getLink[newLinkInfoRec]&)];
				If[matchingRecPosition === {},
					AppendTo[$fes, newLinkInfoRec],
				(* else *)
					$fes = ReplacePart[$fes, mergeLinkInfo[$fes[[First @ matchingRecPosition]], newLinkInfoRec], First @ matchingRecPosition]
				]
			] /@ theseFes,
		(* else *)
			$loop = LinkOpen[LinkMode->Loopback];
			$fes = theseFes;
			goToLoopback = goToLoopbackImplementation;
			(* We want to execute goToLoopback as late as possible in the evaluation of the current cell because after it is called,
			   all Print output, messages, etc. generated until the end of the cell's computation are sent to the loopback
			   link. They can be sent to the FE later (this happens in the 'else' branch below), but things would break if the
			   kernel called the FE and expected an answer (like calling Notebooks[]). If the call to ShareKernel occuured within
			   an EnterXXXPacket, we can hook the write of the InputNamePacket at the end of the current computation and switch to the
			   loopback there. This is very convenient, as it is truly the absolute last thing that will happen in the computation.
			   Unfortunately, this doesn't work if the call to ShareKernel was made from within an EvaluatePacket, as there appears
			   to be no way to hook from top-level code the LinkWrite of the ReturnPacket. For those circumstances, we must switch
			   to the loopback link at the moment ShareKernel is called. Any further packets will pile up on the loopback link and
			   we will send them later. Things will hang, though, if the kernel writes to $ParentLink expecting an answer. This
			   is unlikely since FE functions should write to $FrontEnd, not $ParentLink. Moral: if you are going to call
			   ShareKernel from a palette, you are better off doing it late in the execution of the button function, just to be safe.
			*)
			If[TrueQ[$Notebooks] && Head[Symbol["ButtonNotebook"][]] =!= Symbol["NotebookObject"],
				(* $Notebooks and NOT from a palette --> EnterXXXPacket. *)
				Unprotect[LinkWrite];
				LinkWrite[_, _InputNamePacket, ___] :=
					(Unprotect[LinkWrite];
					Unset[LinkWrite[_, _InputNamePacket, ___]];
					Protect[LinkWrite];
					$Line++;
					goToLoopback[True, None, $Line - 1]);
				Protect[LinkWrite],
			(* else *)
				(* Call to ShareKernel occurred within an EvaluatePacket, or perhaps in an EnterXXXPacket (e.g., from a
				   specialty Java program). This branch works for EnterXXXPacket, but it is not ideal, as explained above.
				*)
				goToLoopbackFirstTime[]
			]
		];
		tok
	]

(* With no $ParentLink, all calls to ShareKernel will fall through to here. *)
ShareKernel[___] := Message[ShareKernel::nofe] /; $ParentLink === Null


UnshareKernel::idx = "UnshareKernel called with invalid index. There was no call to ShareKernel that returned index `1`."
UnshareKernel::lnk = "Link `1` is not currently being shared."
UnshareKernel::jlnk = "There is no link to Java in use. ShareKernel called with zero arguments assumes that you mean the Java link."

(* UnshareKernel with no args kills all sharing, regardless of extant tokens.
   UnshareKernel with a link arg kills sharing of that link, regardless of extant tokens.
   UnshareKernel[tok] removes that token from token list, much like decrementing a refcount.
*)

UnshareKernel[tok_Integer] :=
	Module[{fesThatWillRemain},
		If[isPreemptiveKernel[],
			(* No support for tokens in the modern UnshareKernel implementation. If you have your own link to a
			   MathLink program and you want to use the J/Link Share/UnshareKernel API, you must call UnshareKernel[link].
			*)
			Return[Null]
		];
		If[KernelSharedQ[],
			If[MemberQ[Flatten[getTokenList /@ $fes], tok],
				$fes = removeToken[#, tok]& /@ $fes;
				fesThatWillRemain = Select[$fes, getTokenList[#] =!= {} &];
				(* Turn off FE sharing if JavaLink/NETLink will be removed from sharing list. *)
				If[FrontEndSharedQ[JavaLink[]] && !MemberQ[getLink /@ fesThatWillRemain, JavaLink[]], UnshareFrontEnd[JavaLink[]]];
				If[FrontEndSharedQ[NETLink`NETLink[]] && !MemberQ[getLink /@ fesThatWillRemain, NETLink`NETLink[]], UnshareFrontEnd[NETLink`NETLink[]]];
				(* Now sweep to see if any links have no tokens remaining and thus should be removed from sharing list.
				   If no links have any tokens remaining then we are going to shut down sharing. In that case, however,
				   we want to leave in the list of fes the one link that we want to become the $ParentLink. If we remove
				   them all from $fes, then unshKernel[] will not be able to guess correctly which link should become
				   $ParentLink. We apply the logic that the link that should become $ParentLink is the first one we find
				   that satisfies isFE.
				*)
				If[fesThatWillRemain === {} && Select[$fes, isFE] =!= {},
					$fes = Take[Select[$fes, isFE], 1],
				(* else *)
					$fes = Select[$fes, getTokenList[#] =!= {} &]
				];
				If[Length[$fes] <= 1, unshKernel[]],
			(* else *)
				Message[UnshareKernel::idx, tok]
			]
		];
	]

UnshareKernel[link_LinkObject] :=
	If[KernelSharedQ[],
		If[isPreemptiveKernel[],
			preemptiveUnshareKernel[link],
		(* else *)
			If[MemberQ[getLink /@ $fes, link],
				$fes = Select[$fes, getLink[#] =!= link &];
				Which[
					link === JavaLink[] && FrontEndSharedQ[JavaLink[]],
						UnshareFrontEnd[JavaLink[]],
					link === NETLink`NETLink[] && FrontEndSharedQ[NETLink`NETLink[]],
						UnshareFrontEnd[NETLink`NETLink[]]
				];
				If[Length[$fes] <= 1, unshKernel[]],
			(* else *)
				Message[UnshareKernel::lnk, link]
			]
		]
	]

(* Zero-argument form for backwards compatibility with older uses in J/Link. *)
UnshareKernel[] :=
	(
		If[isPreemptiveKernel[],
			(* Do nothing for JavaLink[], as it handles its own sharing. *)
			Return[Null]
		];
		If[KernelSharedQ[],
			(* We only want to issue a warning if the kernel was shared by some link other than Java, so the user
			   might have inadvertantly left out the link spec. Still should be silent if sharing is not on at all.
			*)
			If[Head[JavaLink[]] === LinkObject && KernelSharedQ[JavaLink[]],
				UnshareKernel[JavaLink[]],
			(* else *)
				Message[UnshareKernel::jlnk]
			]
		]
	)


(* Note that this function must not be specific to the Java link, but honestly report whether any sharing at all is occurring. *)
KernelSharedQ[] := Length[SharingLinks[]] > 0

KernelSharedQ[link_LinkObject] := MemberQ[SharingLinks[], link]


SharingLinks[] :=
	If[isPreemptiveKernel[],
		MathLink`SharingLinks[],
	(* else *)
		getLink /@ $fes
	]


unshKernel[] :=
	Module[{linkToPointAt, loopIsParentLink},
		If[FrontEndSharedQ[], UnshareFrontEnd[]];
		If[Head[NETLink`NETLink[]] === LinkObject, NETLink`Package`nShareKernel[False]];
		If[ValueQ[$periodicalLink], LinkClose[$periodicalLink]; $periodicalLink =.];
		$periodicals = {};
		loopIsParentLink = $ParentLink === $loop;
		If[ValueQ[$loop], LinkClose[$loop]; $loop =.];
		(* Leave $ParentLink pointing at first link in $fes that is an FE.
		   If no links are FEs, then leave pointing at current $ParentLink.
		*)
		linkToPointAt = If[Length[#] > 0, First[#], Null]& @ Select[$fes, isFE];
		$fes = {};
		Which[
			linkToPointAt =!= Null && getLink[linkToPointAt] =!= $ParentLink,
				(* This branch handles the case where UnshareKernel is called from a program other than the front
				  end, typically from Java. More precisely, it is for when the link that we want to leave the kernel
				  pointing at after sharing is over is not the one on which UnshareKernel is being called.

				  This code assumes that the call to UnshareKernel occurred within an EnterTextPacket or
				  EnterExpressionPacket. I could not get it to work any other way, so I will live with
				  that limitation. The problem is that we must not let the front end get any packets from
				  any of this (receiving any packets after an InputNamePacket puts the OSX/UNIX FE into a weird state where
				  it eats up the CPU). If the other side (say, Java) uses EvaluatePacket to wrap the call to UnshareKernel
				  then the FE will get a ReturnPacket. There seems to be no way to hook LinkWrite from top level
				  to prevent that ReturnPacket from being written. Therefore if you call UnshareKernel from Java or some
				  other program, you need to do it in EnterTextPacket or EnterExpressionPacket, not the more
				  typical EvaluatePacket.
				*)
				(* This satisfies the other program's need to get some sort of answer back: *)
				LinkWrite[$ParentLink, InputNamePacket["In[" <> ToString[$Line] <> "]:= "]];
				(* We have to prevent the front end from getting any packets at all as we switch to it. We suppress
				   writes of the ResumePacket and InputNamePacket with this kludge.
				*)
				suppressPacket[InputNamePacket];
				suppressPacket[ResumePacket];
				goToLink[linkToPointAt];
				$Line--;,
			linkToPointAt === Null && loopIsParentLink,
				(* Here we have the case where the FE was quit from its menu while sharing was on. In that case,
				   $ParentLink is the loopback link (or at least it was before we closed the loopback link above).
				   We choose to kill the kernel even though Java and/or .NET programs may be using it.
				*)
				Quit[]
		];
		goToLoopback = (Out[--$Line]; Null)&;
		destroySyntaxRule[];
	]


(* Private` and undocumented, but available to users if necessary. *)
$SleepInterval = 20

(*******************  Implementation  ********************)

(* Safe initializations for multiple reads of this file. *)
If[!ValueQ[$nextShareKernelToken], $nextShareKernelToken = 0]
If[!ValueQ[$fes], $fes = {}]

(* These six funcs are the only ones that know the internals of a linkInfo object. *)

createLinkInfo[link_LinkObject, isFE:(True | False), tokList:{__Integer}] := linkInfo[link, isFE, tokList]
(* MergeLinkInfo lumps the token lists from the two linkInfos, and also changes the isFE state iff the new
   state is FE and the old one is not (that is, only goes _to_ isFE state, never backwards).
*)
mergeLinkInfo[old_linkInfo, new_linkInfo] :=
	linkInfo[First[old], If[!isFE[old] && isFE[new], True, isFE[old]], Join[Last[old], Last[new]]]
removeToken[li_linkInfo, tok_Integer] := ReplacePart[li, DeleteCases[Last[li], tok], -1]
getLink[li_linkInfo] := li[[1]]
isFE[li_linkInfo] := li[[2]]
getTokenList[li_linkInfo] := li[[3]]

(* "Dumbs down" the link from the isFE state to not (called from UnshareFrontEnd). *)
endFEMode[link_LinkObject] := $fes = $fes /. {link, _, toks_} -> {link, False, toks}

(* This is the function used to switch to the loopback link for the first time when the call occurred in an
   EvaluatePacket (it also works for EnterXXXPacket--see ShareKernel for comments).
*)
goToLoopbackFirstTime[] :=
	Module[{curLink},
		curLink = $ParentLink;
		$ParentLink = $loop;
		purge[$loop];  (* Get rid of the ResumePacket from above line. *)
		With[{curLink = curLink}, LinkWrite[$loop, Unevaluated @ sendLoopContents[curLink]]];
		$Line++;
	]


goToLoopbackImplementation[sendInputNamePkt_:True, syntaxErrorType_:None, lineNumber_:0] :=
	Module[{savedOut, lineNum, curLink},
		If[syntaxErrorType =!= None,
			savedOut = $Line - 1,
			savedOut = $Line - 2
		];
		If[ValueQ[tmpOut], Unprotect[Out]; Out[savedOut] = tmpOut; Unset[tmpOut]; Protect[Out]];
		(* The parameter lineNumber is 0 if the default of $Line is to be used, otherwise, it will be a specified number. *)
		lineNum = If[lineNumber == 0, $Line, lineNumber];
		(* First[$fes] is always the previous link, the one we are leaving. *)
		curLink = First[$fes];
		purge[$loop];
		(* If we are finishing servicing a link other than NETLink, we will have earlier set nAllowUIComputations
		   to be false. Reenable it.
		*)
		If[Head[NETLink`NETLink[]] === LinkObject && getLink[curLink] =!= NETLink`NETLink[],
		    NETLink`Package`nAllowUIComputations[True]
		];
		Switch[syntaxErrorType,
			enterText,
				LinkWrite[$loop, Unevaluated @ cleanupEnterTextSyntaxError[]],
			enterExpr,
				(* Odd that this works. What I need to do here is get the ExpressionPacket following the Syntax::sntxi message
				   and send it to FE, followed by a "(sharing)" InputNamePacket. But the ExpressionPacket is not on the loopback
				   link, despite that it is written after the SuspendPacket from the $ParentLink switch away from the fe.
				   For some reason, it gets sent to the FE anyway, so all I need to do here is send an InputNamePacket and get
				   back into sharing loop.
				*)
				LinkWrite[$loop, Unevaluated @ cleanupEnterExprSyntaxError[]],
			_,
				(* no syntax error *)
				$fes = RotateRight[$fes];
				LinkWrite[$loop, Unevaluated @ mainLoop[]]
		];
		$ParentLink = $loop;
		If[sendInputNamePkt,
			LinkWrite[getLink[curLink], InputNamePacket[$prompt <> "In[" <> ToString[lineNum] <> "]:= "]]
		];
	]


(* During the entire runtime of mainLoop, loop is $ParentLink. *)
mainLoop[] :=
	Module[{e, activeLink, dead, javaAvailable, netAvailable, box, inStr, linkStr, result},
		purge[$loop];
		While[!TrueQ[Or @@ (myLinkReadyQ[getLink[#]]& /@ Select[$fes, getLink[#] =!= $periodicalLink &])],
			{$fes, dead} = removeDeadLinks[$fes];
			If[Select[dead, isFE] =!= {} && Select[$fes, isFE] === {},
				(* Kill the kernel if an FE link died and no FE links remain. *)
				Quit[]
			];
			If[Length[$fes] == 1,
				(* The kernel should not stay alive just to service periodicals. *)
				If[getLink[First[$fes]] === $periodicalLink,
					Quit[]
				];
				(* If there is just one link alive, turn off sharing. It will become the parent link. *)
				LinkWrite[$loop, Unevaluated @ EnterExpressionPacket[$Line--; unshKernel[]]];
				Return[Null]
			];
			(* Would be nice if Pause[] took milliseconds instead of integer seconds only. As it is we need to call into
			   Java just to get kernel to yield some CPU in this never-ending computation called mainLoop[].
			*)
			javaAvailable = Head[JavaLink[]] === LinkObject;
			netAvailable = Head[NETLink`NETLink[]] === LinkObject;
			If[MemberQ[SharingLinks[], $periodicalLink] && LinkReadyQ[$periodicalLink],
				(* Don't want to burn up too much CPU checking if a periodical needs servicing, so we sleep double
				   the normal amount before breaking out.
				*)
				Which[
				    $VersionNumber >= 5.0,
				        Pause[.02],
				    javaAvailable,
				        jYieldTime[getDefaultJVM[], 2 $SleepInterval]
				];
				Break[],
			(* else *)
				Which[
				    $VersionNumber >= 5.0,
				        Pause[.01],
				    javaAvailable,
				        jYieldTime[getDefaultJVM[], $SleepInterval]
				]
			]
		];
		(* activeLink is a linkInfo, not a LinkObject. *)
		activeLink = Scan[If[myLinkReadyQ[getLink[#]], Return[#]]&, $fes];
		If[Length[$fes] === 1 && isFE[activeLink],
			(* If we're down to one link, warn user to turn off sharing. *)
			LinkWrite[getLink[activeLink], MessagePacket[ShareKernel, "onlyone"]];
			LinkWrite[getLink[activeLink], TextPacket["ShareKernel::onlyone:  Only one link is using the kernel. You should use UnshareKernel[] to turn off sharing."]]
		];
		If[getLink[activeLink] === JavaLink[],
		    jAllowUIComputations[getDefaultJVM[], True, False]
		];
		If[getLink[activeLink] =!= NETLink`NETLink[],
		    NETLink`Package`nAllowUIComputations[False]
		];
		(* Bring activeLink to front of $fes. *)
		$fes = Cases[$fes, activeLink] ~Join~ DeleteCases[$fes, activeLink];
		e = LinkRead[getLink[activeLink], Hold]; (* Is Hold (vs. HoldComplete) perfectly safe? *)
		If[e === $Failed,
			(* An FE has closed its side of the link. *)
			goToLoopback[];
			Return[]
		];
		Switch[e[[1,0]],
			EnterExpressionPacket,
				If[MatchQ[e, Hold[EnterExpressionPacket[_MakeExpression]]],
				    (* Typical EnterExpressionPacket from the front end. *)
					(* $Line is 1 greater at start than normal eval... *)
					$Line -= 2;
					box = e[[1, 1, 1, 1]];  (* box is expr inside BoxData *)
					box = {RowBox[{"Last", "[", "{", "JLink`Sharing`Private`goToLink", "[", ToString[activeLink, InputForm], "]", ",",
									"JLink`Sharing`Private`setupSyntaxRule", "[", "]", ",",
									"JLink`Sharing`Private`tmpOut", "=", "Out", "[", "$Line", "]", "}", "]", ";"}
							],
							box,
							RowBox[{"JLink`Sharing`Private`goToLoopback", "[", "]"}]
						  } // Flatten; (* The Flatten will have no effect if box is not multi-expr cell. *)
					e = ReplacePart[e, box, {1, 1, 1, 1}],
				(* else *)
				    (* Here we have an EnterExpressionPacket that was probably not from the front end (which uses EnterExprPacket
				       for box input and thus wraps it in MakeExpression). We build the same expression as above.
				    *)
					e = MapAt[CompoundExpression, e, {1, 1}];
					With[{activeLink = activeLink},
					    e = Insert[e, Unevaluated @ Last[{JLink`Sharing`Private`goToLink[activeLink],
									JLink`Sharing`Private`setupSyntaxRule[], JLink`Sharing`Private`tmpOut=Out[$Line]}], {1,1,1}]
					];
					e = Insert[e, Unevaluated @ JLink`Sharing`Private`goToLoopback[], {1,1,-1}]
				],
			EnterTextPacket,
				(* $Line is 1 greater at start than normal eval... *)
				$Line -= 2;
				linkStr = ToString[activeLink, InputForm];
				If[!SyntaxQ[e[[1,1]]],
					(* Note differences to non-syntax error input: Can't put gotoloopback
					   at end because that might fix a syntax error. Rest of magic here is in $SyntaxHandler.
					*)
					inStr = "JLink`Sharing`Private`goToLinkSyntaxError[" <> linkStr <> "];JLink`Sharing`Private`tmpOut=Out[$Line];\n" <> e[[1,1]],
				 			(* spacing in above line cannot be changed; chars are counted in syntax error. *)
				(* else *)
					If[Out[$Line] === Null,
						(* Different handling for when last Out value was Null. I need my first line below to cause
						   Null to be assigned to Out, but Mathematica aggressively seeks to find some non-Null expr
						   generated during the eval of the line to assign to Out. If I use a CompoundExpression anywhere
						   during the evaluation, Mathematica finds its last non-Null expr, no matter how deviously hidden
						   the compound expression is (e.g., in an assignment to a local variable in a Block or Module).
   						   This behavior is not seen in normal operation of the main loop.
						   I must avoid all CEs, including in the body of goToLink and the functions it calls, etc.
						   Same applies to EnterExprPacket handling above.
						*)
						inStr = "Last[{JLink`Sharing`Private`goToLink[" <> linkStr <> "], JLink`Sharing`Private`tmpOut=Out[$Line]}]\n" <>
									e[[1,1]] <> "\n JLink`Sharing`Private`goToLoopback[];",
					(* else *)
						inStr = "JLink`Sharing`Private`goToLink[" <> linkStr <> "];JLink`Sharing`Private`tmpOut=Out[$Line];\n" <>
									e[[1,1]] <> "\n JLink`Sharing`Private`goToLoopback[];"
					]
				];
				e = Hold @@ {EnterTextPacket[inStr]}
				,
			EvaluatePacket,
				With[{fe = activeLink},
					e = e /. EvaluatePacket[expr_] :>
								EvaluatePacket[
									goToLink[fe];
									result = CheckAbort[expr, $Aborted];
									If[KernelSharedQ[],
										LinkWrite[getLink[fe], ReturnPacket[result]];
										(* Prevent the FE from getting anything after the ReturnPacket. *)
										suppressPacket[SuspendPacket];
										goToLoopback[False];,
									(* else *)
										(* Here we accommodate the case where UnshareKernel was called from this EvaluatePacket. *)
										suppressPacket[InputNamePacket];
										--$Line;
										result
									]
								]
				],
			_,
				Null (* do nothing *)
		];
		LinkWriteHeld[$loop, e];
	]


(* Used only by goToLoopbackFirstTime. *)
sendLoopContents[link_LinkObject] :=
	Module[{e},
		(* What is on the loop is a ResumePacket followed by all packets generated from the computation
		   that called ShareKernel (at least all those generated after ShareKernel was called. This includes
		   the ReturnPacket/ReturnTextPacket with the result. There is also always an InputNamePacket last, even if
		   the call to ShareKernel happened in an EvaluatePacket (don't know why this is). These packets should
		   be dumped to the link that called ShareKernel, except that we don't want to send the trailing InputNamePacket
		   after a ReturnPacket (this is the same problem we wrestle with throughout this package--the FE can go into
		   a state where it consumes the CPU if it gets a packet after it thinks the evaluation is done).
		*)
		While[LinkReadyQ[$loop],
			e = LinkReadHeld[$loop];
			LinkWriteHeld[link, e];
			If[MatchQ[e, Hold[ReturnPacket[__]]],
				(* Throw away the InputNamePacket. *)
				purge[$loop]
			]
		];
		LinkWrite[$loop, Unevaluated @ mainLoop[]]
	]


purge[l_LinkObject] := While[LinkReadyQ[l], LinkReadHeld[l]]

(* Should be a ResumePacket, followed by syntax-related pkts, followed by InputNamePacket, followed by ReturnPacket from
   mainLoop finishing. The last pkt I'm interested in is the SyntaxPacket, which marks the close of syntax-related stuff. *)
cleanupEnterTextSyntaxError[] :=
	Module[{pkt, fe = First[$fes]},
		While[True,
			pkt = LinkReadHeld[$loop];
			If[pkt[[1,0]] === SyntaxPacket,
				(* Subtraction accounts for the extra chars I added to input. *)
				LinkWrite[getLink[fe], SyntaxPacket[pkt[[1,1]] - (85 + StringLength[ToString[fe, InputForm]])]];
				goToLoopback[];
				$Line++;
				Return[],
			(* else *)
				LinkWriteHeld[getLink[fe], pkt]
			]
		]
	]

cleanupEnterExprSyntaxError[] := (goToLoopback[]; $Line++)

goToLink[fe_linkInfo] := If[isFE[fe], goToFE[getLink[fe]], goToOther[getLink[fe]]]

goToLinkSyntaxError[fe_linkInfo] :=
	(
		$SyntaxHandler = (goToLoopback[False, enterText]; $Failed)&;
		goToLink[fe];
	)

(* Very deliberate that the implementations of goToFE and goToOther have no CompoundExpressions. See the comment in
   the EnterTextPacket branch of mainLoop.

   Some machinations in here are extremely specific to current implementation details of FE-kernel interaction.
   In particular, System`FEDump`FrontEndQ will probably be changed or go away in future versions of Mathematica.
   This code will need to be revisited every time a new version of Mathematica is released, until documented
   conventions exist about FE-Kernel interaction (Notebook.m, SystemFiles.m, perhaps others).
*)

goToFE[fe_LinkObject] :=
	AbortProtect[
		{
			If[!TrueQ[System`Private`$prevWasFE] && ValueQ[System`Private`$nbOpts],
				{
				SetOptions["stdout", System`Private`$nbOpts],
				Unprotect[$Notebooks],
				$Notebooks = True,
				Protect[$Notebooks],
				OwnValues[System`Private`$SystemPrint] = System`Private`$nbSystemPrint,
				OwnValues[$FormatType] = System`Private`$nbFormatType,
				OwnValues[System`FEDump`FrontEndQ] = System`Private`$nbFrontEndQ
				}
			],
			$FrontEnd = FrontEndObject[fe],
			System`Private`$prevWasFE = True,
			$ParentLink = fe
		}
	]

goToOther[inst_LinkObject] :=
	AbortProtect[
		{
			If[TrueQ[System`Private`$prevWasFE],
				{
				System`Private`$nbOpts = Options["stdout"],
				SetOptions["stdout", FormatType->OutputForm, PageWidth->Infinity, NumberMarks->False],
				Unprotect[$Notebooks],
				$Notebooks = False,
				Protect[$Notebooks],
				Unset[$FrontEnd],
				System`Private`$nbSystemPrint = OwnValues[System`Private`$SystemPrint],
				Unset[System`Private`$SystemPrint],
				System`Private`$nbFormatType = OwnValues[$FormatType],
				$FormatType = OutputForm,
				System`Private`$nbFrontEndQ = OwnValues[System`FEDump`FrontEndQ],
				System`FEDump`FrontEndQ = False,
				System`Private`$prevWasFE = False
				}
			],
			$ParentLink = inst
		}
	]

setupSyntaxRule[] :=
	AbortProtect[
		{
			Unprotect[LinkWrite],
			LinkWrite[a_, b:MessagePacket[Syntax, "sntxi"], c___] :=
				(destroySyntaxRule[]; LinkWrite[a, b, c]; goToLoopback[False, enterExpr];),
			Protect[LinkWrite]
		}
	]

destroySyntaxRule[] :=
	AbortProtect[
		Module[{wasOn},
			Unprotect[LinkWrite];
			(* This function can get called in circumstances when the syntax rule was never set up.
			   Rather than keeping track of when it has been set or not, we'll just turn the warning
			   message off and clear it every time.
			*)
			wasOn = Head[Unset::norep] =!= $Off;
			Off[Unset::norep];
			Unset[LinkWrite[a_, b:MessagePacket[Syntax, "sntxi"], c___]];
			If[wasOn, On[Unset::norep]];
			Protect[LinkWrite]
		];
	]

myLinkReadyQ[link_] :=
	Which[
		link === JavaLink[],
			JAssert[!LinkReadyQ[link]];
			(* Despite its name, jUIThreadWaitingQ isn't guaranteed to return T/F (e.g., a MathLink error). Thus, TrueQ. *)
			TrueQ[jUIThreadWaitingQ[getDefaultJVM[]]],
		LinkConnectedQ[link],
			LinkReadyQ[link],
		True,
			If[LinkReadyQ[link],
				LinkConnect[link];
				LinkWrite[link, InputNamePacket["In[" <> ToString[$Line] <> "]:="]]
			];
			LinkReadyQ[link]
	]

removeDeadLinks[fes_List] :=
	Module[{deadPos},
		deadPos =
			Position[Check[ If[!MatchQ[First @ LinkError @ getLink @ #, 0 | 10], LinkClose[getLink[#]]; Null, #], Null ]& /@ fes, Null];
		{Delete[fes, deadPos], Extract[fes, deadPos]}
	]

suppressPacket[pkt_Symbol] :=
	(
		Unprotect[LinkWrite];
		LinkWrite[_, _pkt, ___] :=
			(
				Unprotect[LinkWrite];
				Unset[LinkWrite[_, _pkt, ___]];
				Protect[LinkWrite];
			);
		Protect[LinkWrite];
	)


(**************************************  preemptiveShareKernel  *******************************************)

(* Modern, 5.1+ implementation that uses kernel internals. *)


preemptiveShareKernel[pairs:{{_LinkObject, True|False}..}, opts___?OptionQ] :=
	Module[{links},
		links = First /@ pairs;
		(* JavaUILink[] and NETLink[] manually manage their own sharing setup, so remove
           them from the set if someone tries to manipulate them via ShareKernel. Also, prevent
           anyone from trying to share JavaLink[].
		*)
		links = DeleteCases[links, JavaLink[] | JavaUILink[] | NETLink`NETLink[]];
		(* Also delete any links currently being shared. If you want to re-share a link, say to change its
		   sharing options, you must of course use MathLink`AddSharingLink, not ShareKernel.
		*)
		links = links ~Complement~ MathLink`SharingLinks[];
		MathLink`AddSharingLink /@ links;
		(* Return an integer ID to conform to legacy version, although it is essentially meaningless. *)
		0
	]


preemptiveUnshareKernel[link_] :=
	(
		(* JavaUILink[] and NETLink[] manually manage their own sharing setup, so prevent anyone from
		   trying to manipulate them directly via UnshareKernel. Also, prevent anyone from trying to use JavaLink[].
		*)
		If[link =!= JavaLink[] && link =!= JavaUILink[] && link =!= NETLink`NETLink[],
			MathLink`RemoveSharingLink[link]
		];
	)


(********************************************  Periodicals  ***************************************************)

(* In M 5.1 and later, periodicals use Internal` functions. The comments below apply only to the legacy implementation.

   Periodicals are not concerned with tokens in their use of ShareKernel. When the last periodical is deleted, we call
   UnshareKernel[$periodicalLink] to wipe the link off the sharing list. We know that no one will accidentally
   turn off sharing while periodicals are alive because $periodicalLink gets a unique token when we call ShareKernel[$periodicalLink].
   No one will ever get that token again, so they cannot shut us off. Unless, of course, they call UnshareKernel[]
   with no args, which kills all sharing, but that it deliberate. Note also that we clutter up the front end link with a token
   when we call ShareKernel[$periodicalLink], but tokens are in fact irrelevant for the front end link. Only non-FE links
   need to be concerned with tokens.
*)

AddPeriodical::nofe = "Periodical tasks cannot be used unless the kernel is linked to a front end program of some sort."
RemovePeriodical::notfound = "There is no periodical task registered with an id of `1`."
Periodical::notfound = "There is no periodical task registered with an id of `1`."
SetPeriodicalInterval::notfound = "There is no periodical task registered with an id of `1`."


If[!ValueQ[$periodicals], $periodicals = {}]

SetAttributes[AddPeriodical, HoldFirst]

AddPeriodical[expr_, secs_?NumberQ] :=
	Module[{id, parentLinkIsFE, e},
		If[$ParentLink === Null,
			Message[AddPeriodical::nofe];
			Return[$Failed]
		];
        id = If[Length[Periodicals[]] > 0, Max[Periodicals[]] + 1, 1];
        AppendTo[$periodicals, {id, e, secs, SessionTime[]}];
		(* Internal`AddPeriodical automatically strips the wrapping Hold. *)
		If[MatchQ[Hold[expr], Hold[(Hold | HoldForm | HoldComplete)[___]]],
			e = With[{pid = id}, Hold[Block[{$ThisPeriodical = pid}, ReleaseHold[expr]]]],
		(* else *)
			e = With[{pid = id}, Hold[Block[{$ThisPeriodical = pid}, expr]]]
		];
		If[isPreemptiveKernel[],
			Internal`AddPeriodical[Evaluate[e], secs],
		(* else *)
			(* Old kernel. Use legacy method. *)
			If[!ValueQ[$periodicalLink],
				$periodicalLink = LinkOpen[LinkMode->Loopback];
				LinkWrite[$periodicalLink, Unevaluated[EvaluatePacket[servicePeriodicals[]]]]
			];
			parentLinkIsFE = hasFrontEnd[] && $ParentLink === First[$FrontEnd];
			ShareKernel[{{$ParentLink, parentLinkIsFE}, {$periodicalLink, False}}]
		];
		id
	]

RemovePeriodical[ids:{___Integer}] := (RemovePeriodical /@ ids;)

RemovePeriodical[id_Integer] :=
	Module[{p},
		p = Cases[$periodicals, {id, __}];
		If[Length[p] > 0,
			$periodicals = DeleteCases[$periodicals, {id, __}];
			If[isPreemptiveKernel[],
				Internal`RemovePeriodical[Evaluate[First[p][[2]]]],
			(* else *)
				If[Length[Periodicals[]] == 0,
					UnshareKernel[$periodicalLink];
					(* UnshareKernel might have closed and unset $periodicalLink, so we check here before doing it. *)
					If[ValueQ[$periodicalLink],
						LinkClose[$periodicalLink];
						$periodicalLink =.
					]
				]
			],
		(* else *)
			Message[RemovePeriodical::notfound, id]
		];
	]

Periodical[id_Integer] :=
	Module[{per},
		per = Cases[$periodicals, {id, __}];
		If[Length[per] > 0,
			(* Take out the "last time" field for display. *)
			Drop[per[[1]], -1],
		(* else *)
			Message[Periodical::notfound, id]
		]
	]

Periodicals[] := First /@ $periodicals


SetPeriodicalInterval[id_Integer, secs_?NumberQ] :=
	Module[{pos},
		pos = Position[$periodicals, {id, __}];
		If[pos === {},
			Message[SetPeriodicalInterval::notfound, id],
		(* else *)
			pos = First[Flatten[pos]];
			If[isPreemptiveKernel[],
				(* No API in kernel for modifying interval, so just re-add it. *)
				Internal`RemovePeriodical[Evaluate[$periodicals[[pos, 2]]]];
				Internal`AddPeriodical[Evaluate[$periodicals[[pos, 2]]], secs]
			];
			$periodicals[[pos, 3]] = secs
		];
	]


servicePeriodicals[] :=
	Module[{recs},
		recs = doPeriodical /@ $periodicals;
		(* recs looks like {{id, lastTime}...}. Next we update the lastTime elements based on id, without interfering
		   with other changes to $periodicals. This allows periodicals to reset their intervals or delete themselves
		   while they are running.
		*)
		Scan[($periodicals = Replace[$periodicals, {#[[1]], code_, secs_, _} :> {#[[1]], code, secs, #[[2]]}, {1}])&, recs];
		purge[$periodicalLink];
		LinkWrite[$periodicalLink, Unevaluated[EvaluatePacket[servicePeriodicals[]]]]
	]

doPeriodical[{id_, code_, secs_, last_}] :=
	If[SessionTime[] - last >= secs,
		Block[{$ThisPeriodical = id},
			ReleaseHold[code]
		];
		{id, SessionTime[]},
	(* else *)
		{id, last}
	]


(*******************************************  ShareFrontEnd  *****************************************************)

(* Note that ShareFrontEnd calls ShareKernel. It is not meaningful to share the frontend and not the kernel.

   ShareFrontEnd, like ShareKernel, returns a token that can be passed to UnshareFrontEnd. In this way, a
   programmer can unregister their code's request for sharing without blindly turning off sharing when some other
   code is using it. Every ShareFrontEnd request is also a ShareKernel request, so ShareFrontEnd just returns
   the token from its internal ShareKernel call. We maintain a simple list of tokens, $shareFrontEndTokens, that
   holds the tokens from every call to ShareFrontEnd. In other words, it works fine for ShareKernel and ShareFrontEnd
   to share the same token space. The only thing ShareFrontEnd needs to do is to manage its own separate list of tokens,
   so it knows when all requests for FE sharing have been unregistered.
*)

ShareFrontEnd::nofe = "ShareFrontEnd can only be executed from the notebook front end."
ShareFrontEnd::fail = "An unknown error occurred while attempting to establish sharing."
ShareFrontEnd::openx = "A connection to the front end could not be established."
ShareFrontEnd::java = "You must use InstallJava to start the Java runtime before calling ShareFrontEnd with no arguments."
UnshareFrontEnd::nofe = "UnshareFrontEnd can only be executed from the notebook front end."
UnshareFrontEnd::idx = "UnshareFrontEnd called with invalid index. There is no currently active call to ShareFrontEnd that returned index `1`."


ShareFrontEnd[opts___?OptionQ] :=
	If[Head[JavaLink[]] =!= LinkObject,
		Message[ShareFrontEnd::java];
		Return[$Failed],
	(* else *)
		ShareFrontEnd[JavaLink[], opts]
	]

ShareFrontEnd[link_LinkObject, opts___?OptionQ] :=
	Module[{linkName, protocol, result, fe, linkResult},
		fe = getFrontEndObject[];
		If[Head[fe] =!= FrontEndObject,
			Message[ShareFrontEnd::nofe];
			Return[$Failed]
		];
		If[hasServiceFrontEnd[],
			Return[serviceShareFrontEnd[link]]
		];
		If[FrontEndSharedQ[link],
			(* Get another ShareKernelToken for this combo. *)
			result = ShareKernel[{{First[fe], True}, {link, True}}, opts],
		(* else *)
			result = $Failed;
			{linkName, protocol} = setupServerNb[link, fe];
			(* setupServerNb is responsible for issuing messages if it fails. *)
			If[StringQ[linkName],
				linkResult =
					Switch[link,
						JavaLink[],
							jConnectToFEServer[getDefaultJVM[], linkName, protocol],
						NETLink`NETLink[],
							NETLink`Package`nConnectToFEServer[linkName]
					];
				Switch[linkResult,
					True,
						result = ShareKernel[{{First[fe], True}, {link, True}}, opts],
					False,
						Message[ShareFrontEnd::openx],  (* Link to FE failed to open or initialize properly. *)
					_,
						Message[ShareFrontEnd::fail]  (* Weird exception. *)
				]
			]
		];
		If[IntegerQ[result],
			AppendTo[$shareFrontEndTokens, {link, result}]
		];
		result
	]


UnshareFrontEnd[] := If[Head[JavaLink[]] === LinkObject, UnshareFrontEnd[JavaLink[]], Null]

UnshareFrontEnd[tok_Integer] :=
	Module[{linkForThisTok},
		If[hasServiceFrontEnd[],
			Return[serviceUnshareFrontEnd[tok]]
		];
		linkForThisTok = Cases[$shareFrontEndTokens, {lnk_LinkObject, tok} -> lnk];
		If[Length[linkForThisTok] > 0,
			linkForThisTok = First[linkForThisTok],
		(* else *)
			Message[UnshareFrontEnd::idx, tok];
			Return[]
		];
		If[$ParentLink =!= Null && Head[getFrontEndObject[]] =!= FrontEndObject,
			Message[UnshareFrontEnd::nofe];
			Return[]
		];
		If[Length[Cases[$shareFrontEndTokens, {linkForThisTok, _}]] == 1,
			(* If this is our last token on this link, turn off sharing for that link. *)
			UnshareFrontEnd[linkForThisTok],
		(* else *)
			UnshareKernel[tok];
			$shareFrontEndTokens = Drop[$shareFrontEndTokens, First @ Flatten @ Position[$shareFrontEndTokens, {linkForThisTok, tok}]]
		];
	]

UnshareFrontEnd[link_LinkObject] :=
	Module[{feToksForThisLink},
		If[hasServiceFrontEnd[],
			Return[serviceUnshareFrontEnd[link]]
		];
		If[FrontEndSharedQ[link],
			If[$ParentLink =!= Null && Head[getFrontEndObject[]] =!= Symbol["FrontEndObject"],
				Message[UnshareFrontEnd::nofe];
				Return[]
			];
			destroyServerNb[link, getFrontEndObject[]];
			Switch[link,
				JavaLink[],
					jDisconnectToFEServer[getDefaultJVM[]],
				NETLink`NETLink[],
					NETLink`Package`nDisconnectToFEServer[]
			];
			(* Before calling UnshareKernel we must clear $shareFrontEndTokens, which will make FrontEndSharedQ give False.
			   Otherwise, ShareKernel will try to call UnshareFrontEnd reentrantly, and we get problems.
			*)
			feToksForThisLink = Cases[$shareFrontEndTokens, {link, tok_} -> tok];
			$shareFrontEndTokens = DeleteCases[$shareFrontEndTokens, {link, _}];
			(* Tell ShareKernel that JavaLink/NETLink is no longer an FE ("dumb" it back down to the normal
			   ShareKernel state). This changes the isFE state of the the link without changing its token count.
			*)
			endFEMode[link];
			UnshareKernel /@ feToksForThisLink;
		]
	]


FrontEndSharedQ[] :=
	Which[
		hasServiceFrontEnd[],
			True,
		Head[JavaLink[]] === LinkObject,
			FrontEndSharedQ[JavaLink[]],
		True,
			False
	]

FrontEndSharedQ[link_LinkObject] :=
	hasServiceFrontEnd[] || Length[Cases[$shareFrontEndTokens, {link, _}]] > 0


(* Safe initialization for multiple reads of this file. *)
If[!ValueQ[$shareFrontEndTokens], $shareFrontEndTokens = {}]

(* Global needed because of Mac FE bug: Notebooks[] doesn't include ones with Visible->False.
   I only use this variable on the Mac, because the dynamic method of finding the server nb
   by looking up the nb's Evaluator is nicer.
*)
$serverNb

setupServerNb[link_LinkObject, fe_] :=
	Module[{origEvalNames, newEvalNames, serverNb, nbOpts, newLinkName, evaluatorName, evaluator, protocol},
		evaluatorName =
			Switch[link,
				JavaLink[],
					"JavaServer",
				NETLink`NETLink[],
					"NETLinkServer"
			];
		(* Set up special evaluator. Must force TCP protocol on Unix-like OS's (avoid using newer TCPIP protocol
		   for compatibility with older versions of the front end).
		*)
		protocol =
			Which[
				osIsWindows[],
					Automatic,
				TrueQ[$forceTCP],
					(* TCPIP is preferred, but not used in the past, thus the $forceTCP flag is provided as
					   a backdoor to allow top-level programmers to force use of the older TCP protocol.
					*)
					"TCP",
				$VersionNumber >= 5.0,
					"TCPIP",
				True,
					"TCP"
			];
		(newLinkName = First[#]; LinkClose[#])& [ LinkCreate[LinkProtocol -> protocol] ];
		(* Link names from TCP or TCPIP protocol won't be adequate (missing domain). Thus we strip
		   down to just the port number. *)
		If[StringMatchQ[newLinkName, "*\\@*"],
			newLinkName = StringTake[newLinkName, First @ Flatten @ StringPosition[newLinkName, "@"] - 1]
		];
		evaluator = evaluatorName -> {"EvaluatorMenuListing"->evaluatorName, "MLOpenArguments"->
					("-LinkMode listen -LinkName " <> newLinkName <> If[protocol === Automatic, "", " -LinkProtocol " <> protocol])};
		(* Hacks below for $Context and $ContextPath required for M 3.0. *)
		origEvalNames =
			Block[{$Context = "System`", $ContextPath = {"System`"}},
				System`EvaluatorNames /. Options[fe, "EvaluatorNames"]
			];
		If[(evaluatorName /. origEvalNames) =!= evaluatorName,
			(* If for some reason a JavaServer evaluator name exists already, get rid of it, so that it is gone even after
			   we restore the original evaluator names.
			*)
			origEvalNames = DeleteCases[origEvalNames, evaluatorName -> _]
		];
		(* Add special evaluator. *)
		newEvalNames = Append[origEvalNames, evaluator];
		(* There was once a bug in J/Link that caused the call to Options[fe, "EvaluatorNames"] above to get
		   a packet from some other computation (this was when ShareKernel and ShareFrontEnd were called in the same cell).
		   This bug was fixed, but I can imagine circumstances when something like it could happen again, and the
		   effect was catastrophic--all your evaluator settings would get wiped out. As a fail-safe, I'll test to make
		   sure that newEvalNames is at least a list of rules before I overwrite the user's evaluator settings with it. I have
		   no idea where else things might fail if such a bug cropped up again, but at least the SetOptions line below
		   won't cause a problem.
		*)
		If[MatchQ[newEvalNames, {__Rule}],
			SetOptions[fe, System`EvaluatorNames->newEvalNames]
		];
		(* Create invisible notebook. *)
		nbOpts = {Visible->False, Evaluator->evaluatorName, NotebookAutoSave->False};
		If[$VersionNumber >= 4.0, AppendTo[nbOpts, ClosingAutoSave->False]];
		serverNb = Symbol["NotebookCreate"] @@ nbOpts;
		If[osIsMacOSX[],
			(* See comment on declaration of $serverNb above about this Mac FE bug workaround. Also, NotebookObjects store the
			   current $FrontEnd when they are created, and many operations on the object use that link. Because we will
			   be calling NotebookClose on the NotebookObject from Java, we need to replace the stored link with the link
			   to Java.
			*)
			$serverNb = ReplacePart[serverNb, Symbol["FrontEndObject"][JavaLink[]], 1];
		];
		Symbol["NotebookWrite"][serverNb,
			{
				Symbol["Cell"]["(* This notebook is created, used, and destroyed by the ShareFrontEnd[] function. It is normally hidden. This cell will remain in the calculating state. If you close this notebook manually, you will break the connection between Java or .NET and the Front End required by ShareFrontEnd. *)", "Input"],
				Symbol["Cell"][BoxData[ButtonBox["Hide This Window Again", ButtonFunction:>(Symbol["FrontEndExecute"][FrontEnd`SetOptions[Symbol["FrontEnd`InputNotebook[]"], Visible->False]]&), Active->True, ButtonFrame->"DialogBox"]], "Output"]
			}
		];
		Symbol["SelectionMove"][serverNb, Previous, Cell, 2];
		Symbol["SelectionEvaluate"][serverNb];
		Pause[1];  (* May be necessary to avoid race condition with Java. *)
		(* Restore the original set of evaluator names, so that the JavaServer evaluator does not appear in the menu. See the
		   SetOptions call above for explanation of the If test here.
		*)
		If[MatchQ[origEvalNames, {__Rule}],
  			SetOptions[fe, System`EvaluatorNames->origEvalNames]
  		];
		{newLinkName, If[protocol === Automatic, "", protocol]}
	]

destroyServerNb[link_LinkObject, fe_] :=
	Module[{evaluatorName, serverNbs},
		(* Destroy invisible notebook. *)
		evaluatorName =
			Switch[link,
				JavaLink[],
					"JavaServer",
				NETLink`NETLink[],
					"NETLinkServer"
			];
		serverNbs = Select[Symbol["Notebooks"][fe], ((Evaluator /. Options[#]) === evaluatorName)&];
		If[osIsMacOSX[] && ValueQ[$serverNb],
			(* See comment on declaration of $serverNb above about this Mac FE bug workaround. *)
			serverNbs = Union[{$serverNb}, serverNbs];
			$serverNb =.
		];
		DebugPrint[2, "serverNBs = ", serverNbs];
		(* Should only be 1, but might as well just code it to acommodate > 1. *)
		Symbol["NotebookClose"] /@ serverNbs;
	]

(* Create a function for this, even though its current implementation is trivial. *)
getFrontEndObject[] := $FrontEnd


(************************************  serviceShareFrontEnd  **************************************)

(* Modern, 6.0+ implementation that uses FE service. *)

serviceShareFrontEnd[___] = 0   (* Return int for compatibility *)

serviceUnshareFrontEnd[___] = Null



End[]

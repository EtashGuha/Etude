
BeginPackage["StreamLink`"]

StreamLinkLoad
StreamLinkBegin
SetupPreemptiveLink

$MainLinkSwitchPre
$MainLinkSwitchPost

$PreemptiveLinkSwitchPre
$PreemptiveLinkSwitchPost


Begin["`Private`"]

(* Clients of this package, like CloudSystem.m, can override these definitions of the LinkSwitchPre/Post functions, for example to
   call SetFrontEnd as appropriate.
*)
$MainLinkSwitchPre = Function[{link}, MathLink`SetFrontEnd[Null]]
$MainLinkSwitchPost = Function[{prev}, MathLink`RestoreFrontEnd[prev]]

$PreemptiveLinkSwitchPre = Function[{link}, MathLink`SetFrontEnd[Null]; MathLink`SetMessageLink[Null] (* preemptive links should set this to Null if they SetFrontEnd[Null] *)]
$PreemptiveLinkSwitchPost = Function[{prev}, MathLink`RestoreFrontEnd[prev]]


StreamLinkLoad[] :=
    If[!TrueQ[$streamLinkLibraryLoaded],
        (* These must be global, otherwise the library will be unloaded when the StreamLinkBegin Module exits. *)
        createLink = LibraryFunctionLoad["StreamLink", "createLink",  {"UTF8String", Integer, Integer}, "UTF8String"];
        (* Load this function now, as later loading could fail when cloud sandbox has been set up (the path to the lib is no longer visible). *) 
        logDepthFunction = LibraryFunctionLoad["StreamLink", "setLogDepth", {Integer}, Integer];
        $streamLinkLibraryLoaded = True;
    ]


(* Sets up the kernel's $ParentLink to be a StreamLink using either a socket or stdin/stdout for the streams. This mimics a traditional non-preemptive "main link" between a front end and kernel. *)

StreamLinkBegin[] := StreamLinkBegin[Null]

StreamLinkBegin[portNumber:(Null | _Integer)] :=
	Module[{name, link},
		StreamLinkLoad[];
		name = createLink["StreamLinkMain", 1 (* Main link type *), If[portNumber === Null, 0, portNumber]];
		link = LinkConnect[name, LinkProtocol->"IntraProcess"];
		$ParentLink = link;
		(* By setting $ParentLink, we have entered our link into the kernel's link-sharing system, but now we need to modify some of the default behavior.
		   This is needed to keep $FrontEnd setup correctly in the presence of switching back and forth between the main and preemptive links.
		*)
		MathLink`AddSharingLink[link, MathLink`LinkSwitchPre :> $MainLinkSwitchPre, MathLink`LinkSwitchPost :> $MainLinkSwitchPost, MathLink`Terminating -> True]
	]


SetupPreemptiveLink[portNumber_Integer] :=
    Module[{preemptiveName, preemptiveLink},
        StreamLinkLoad[];
        preemptiveName = createLink["StreamLinkPreemptive", 2 (* Preemptive link type *), portNumber];
        If[StringQ[preemptiveName],
            preemptiveLink = LinkConnect[preemptiveName, LinkProtocol->"IntraProcess"];
            MathLink`AddSharingLink[preemptiveLink, MathLink`AllowPreemptive -> True,
                    MathLink`SendInputNamePacket -> False, MathLink`Daemon -> True, MathLink`ImmediateStart -> False,
                    MathLink`LinkSwitchPre :> $PreemptiveLinkSwitchPre, MathLink`LinkSwitchPost :> $PreemptiveLinkSwitchPost];
            preemptiveLink,
        (* else *)
            $Failed
        ]
    ]


(* Private function to control logging. Can only be called after StreamLinkLoad[]. Use -1 for no logging at all, 0 for absolutely minimal, 1 to only see the heads of expressions, etc. 
   Returns the previous log depth.
*)
setLogDepth[logDepth_Integer] := logDepthFunction[logDepth]
 

End[]

EndPackage[]

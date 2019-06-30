(* :Title: Utils.m *)

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


(*<!--Public From Utils.m

FixCRLF::usage =
"FixCRLF[\"str\"] changes the linefeeds in the given string to the CR/LF Windows convention. Use this function on strings that \
are generated in the Wolfram Language and need to be placed into text boxes or other .NET GUI elements. Wolfram Language strings use just the \\n \
character (ASCII 10) for newlines, and these characters generally show up as rectangles in Windows text-based controls."

-->*)

(*<!--Package From Utils.m

osIsWindows
osIsMacOSX
isPreemptiveKernel
isServiceFrontEnd
contextIndependentOptions
filterOptions
preemptProtect

-->*)


(* Current context will be NETLink`. *)

Begin["`Utils`Private`"]


osIsWindows[] = StringMatchQ[$System, "*Windows*"]

osIsMacOSX[] = StringMatchQ[$System, "*Mac*X*"]

isPreemptiveKernel[] = $VersionNumber >= 5.1

isServiceFrontEnd[] = $VersionNumber >= 6.0   (* TODO: Really should be a call to the FE for version info. BUT this call needs to be fast... *)


FixCRLF[s_String] := StringReplace[s, {"\r\n" -> "\r\n", "\n" -> "\r\n"}]


(* This processes the names of options in a context-independent way.
*)
contextIndependentOptions[optName_Symbol, opts_List, defaults_List] :=
    First[ contextIndependentOptions[{optName}, opts, defaults] ]

contextIndependentOptions[optNames_List, opts_List, defaults_List] :=
    Module[{optNameStrings, stringifiedOptionSettings, stringifiedOptionDefaults},
        optNameStrings = (# /. x_Symbol :> SymbolName[x])& /@ optNames;
        stringifiedOptionSettings = MapAt[(# /. x_Symbol :> SymbolName[x])&, #, {1}]& /@ Flatten[{opts}];
        stringifiedOptionDefaults = MapAt[(# /. x_Symbol :> SymbolName[x])&, #, {1}]& /@ Flatten[{defaults}];
        optNameStrings /. stringifiedOptionSettings /. stringifiedOptionDefaults
    ]


(* Include package-level defs here to avoid reliance on the Utilities`FilterOptions` standard package (which is
   not included in a Minimal Install).
*)
filterOptions[command_Symbol, options___] := filterOptions[First /@ Options[command], options]
filterOptions[opts_List, options___] := Sequence @@ Select[Flatten[{options}], MemberQ[opts, First[#]]&]


(* Define a version of MathLink`PreemptProtect that is a no-op in 5.2 and earlier. *)

SetAttributes[preemptProtect, {HoldFirst}]

If[$VersionNumber >= 6,
	preemptProtect = MathLink`PreemptProtect,
(* else *)
	preemptProtect = Identity
]


End[]

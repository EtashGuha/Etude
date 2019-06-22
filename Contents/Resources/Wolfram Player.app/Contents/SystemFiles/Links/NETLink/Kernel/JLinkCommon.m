(* :Title: JLinkCommon.m *)

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


(*<!--Public From JLinkCommon.m

Off[General::shdw]

NETLink`InstanceOf = JLink`InstanceOf
JLink`InstanceOf::usage = JLink`InstanceOf::usage <>
"\n\nInstanceOf[netobject, nettype] gives True if netobject is an instance of the type nettype, or a subtype. \
Otherwise, it returns False. It mimics the behavior of the C# language's 'is' operator. The nettype argument can \
be either the fully-qualified class or interface name as a string, or a NETType expression."
NETLink`InstanceOf::usage = JLink`InstanceOf::usage

NETLink`SameObjectQ = JLink`SameObjectQ
JLink`SameObjectQ::usage = JLink`SameObjectQ::usage <>
"\n\nSameObjectQ[netobject1, netobject1] returns True if and only if the NETObject expressions netobject1 and netobject2 \
refer to the same .NET object. It is a shortcut to calling Object`ReferenceEquals[netobject1, netobject2]."
NETLink`SameObjectQ::usage = JLink`SameObjectQ::usage

On[General::shdw]

-->*)

(*<!--Package From JLinkCommon.m

-->*)


(* Current context will be NETLink`. *)

Begin["`JLinkCommon`Private`"]

Unprotect[JLink`SameObjectQ, JLink`InstanceOf]

JLink`SameObjectQ[obj1_?NETObjectQ, obj2_?NETObjectQ] := TrueQ[nSameQ[obj1, obj2]]


InstanceOf::badobj = "`1` is not a valid Java object reference."
InstanceOf::badcls = "Invalid class specification `1`."

JLink`InstanceOf[Null, _NETType] = False

JLink`InstanceOf[obj_?NETObjectQ, type_NETType] := TrueQ[nInstanceOf[obj, getAQTypeName[type]]]

JLink`InstanceOf[obj_?NETObjectQ, typeName_String] :=
    Module[{type},
        type = LoadNETType[typeName];
        If[Head[type] === NETType,
            TrueQ[nInstanceOf[obj, getAQTypeName[type]]],
        (* else *)
            (* Message will have already been issued by LoadNETType. *)
            False
        ]
    ]

Protect[JLink`SameObjectQ, JLink`InstanceOf]

End[]

(* :Title: ComplexType.m *)

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


(*<!--Public From ComplexType.m

GetComplexType::usage =
"GetComplexType[] returns the .NET type that is currently mapped to Wolfram Language Complex numbers. This is the \
type that will be used when Complex numbers are sent to .NET, and objects of this type will be converted to Complex \
when sent to the Wolfram Language. It returns Null when no type has yet been designated via SetComplexType."

SetComplexType::usage =
"SetComplexType[type] tells .NET/Link to map the specified type to Wolfram Language Complex numbers. This is the \
type that will be used when Complex numbers are sent to .NET, and objects of this type will be converted to Complex \
when sent to the Wolfram Language. The type argument can be specified as a string or as a NETType expression obtained \
from LoadNETType."

-->*)

(*<!--Package From ComplexType.m

-->*)


(* Current context will be NETLink`. *)

Begin["`ComplexType`Private`"]


SetComplexType::typeerr =
"The .NET Type represented by `1` does not have the required members to act as a type for complex numbers in .NET/Link. \
For example, it might not have the required Re or Im public accessor methods or properties."


SetComplexType[typeName_String] := 
    Module[{type = LoadNETType[typeName]},
        If[Head[type] === NETType,
            SetComplexType[type],
        (* else *)
            $Failed
        ]
    ]

SetComplexType[type_NETType] :=
    Switch[nSetComplex[getAQTypeName[type]],
        True,
            $complexClass = type;
            Null,
        False,
            Message[SetComplexType::typeerr, type];
            $Failed,
        _,
            (* Will be $Failed, and an exception message has already been issued. *)
            $Failed
    ]

(* It is too expensive to call into .NET to get the complex class, so we store it in Mathematica
   as well. This means that user must not let the values stored in .NET and Mathematica get out
   of sync. The only way to do that would be to manually set the ComplexType property in a .NET language.
*)

GetComplexType[] := If[ValueQ[$complexClass], $complexClass, Null]


End[]

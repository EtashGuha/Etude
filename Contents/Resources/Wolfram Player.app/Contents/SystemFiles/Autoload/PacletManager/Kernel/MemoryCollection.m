(* :Title: MemoryCollection.m *)

(* :Author:
        Todd Gayley
        tgayley@wolfram.com
*)

(* :Package Version: 3.0 *)

(* :Mathematica Version: 8.1 *)

(* :Copyright: Mathematica source code (c) 1999-2019, Wolfram Research, Inc. All rights reserved. *)

(* :Discussion: This file is a component of the PacletManager Mathematica source code. *)


(* This "class" will contain code for working with memory-based collections. These are collections that are
   represented in moemory as lists of Paclet expressions, and read either from a serialized file of the component
   PacletInfo.m files.
   All its funcs will need to take a collection as the first argument, as the several collections that use
   this code are held elsewhere.

*)

Begin["`Package`"]


MemoryCollection


MCfindForDocResource


End[]  (* `Package` *)


(* Current context will be PacletManager`. *)

Begin["`MemoryCollection`Private`"]



MCfindForDocResource[collection_, linkBase:(_String | All), context:(_String | All),
                        expandedResourceName:(_String | All), language_String] :=
    Module[{},
             
        (* Single-word lookups are converted into a linkbase and empty resName. For user paclets, this supports
           resolving a paclet name to its main page.
        *)

        Select[
            Cases[{#, PgetDocResourcePath[#, linkBase, context, expandedResourceName, language]}& /@ collection,
                {_Paclet, _String}
            ],
            (systemIDMatches[getPIValue[First[#], "SystemID"]] && kernelVersionMatches[getPIValue[First[#], "WolframVersion"]] && productNameMatches[getPIValue[First[#], "ProductName"]])&
        ]
    ]
    

End[]


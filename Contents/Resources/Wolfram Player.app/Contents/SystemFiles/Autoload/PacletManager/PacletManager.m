(* :Title: PacletManager *)

(* :Context: PacletManager` *)

(* :Author:
        Todd Gayley
        tgayley@wolfram.com
*)

(* :Package Version: 3.0 *)

(* :Mathematica Version: 8.1 *)

(* :Copyright: Mathematica source code (c) 1999-2019, Wolfram Research, Inc. All rights reserved. *)

(* :Discussion:

   PacletManager uses a special system wherein one package context (PacletManager`) has its
   implementation split among a number of .m files. Each component file has its own private
   context, and also potentially introduces public symbols (in the PacletManager` context) and
   so-called "package" symbols, where the term "package" comes from Java terminology,
   referring to symbols that are visible everywhere within the implementation of PacletManager,
   but not to clients.
*)

(* :Keywords:  *)


BeginPackage["PacletManager`"]

(*********************  Try to locate the PacletManager.mx file  *********************)

`Package`$pmDir = DirectoryName[$InputFileName]
(* $pmDir is captured in mx file so we need to restore its value after loading the mx. Store a copy. *)
System`Private`savedPMDir = `Package`$pmDir

(* the PacletManager includes its Mathematica code in both a platform- and version-specific .mx file as well as .m files.
   For speed, we want to load the .mx file if appropriate, otherwise we fall back to reading the .m files.
   The test to load the .mx file is whether the version of Mathematica that built the .mx file is the same
   as the one the user is running.
*)
If[SyntaxQ["@mathematicaVersionNumber@"] && ToExpression["@mathematicaVersionNumber@"] == $VersionNumber,
    Quiet[
        Check[
            `Private`foundMX =
                (Get[ToFileName[{`Package`$pmDir, "Kernel", "SystemResources", $SystemID}, "PacletManager.mx"]] === Null),
        (* If messages generated (e.g. DumpGet::bgcorr), leave foundMX=False. *)
            `Private`foundMX = False
        ]
    ],
(* else *)
    `Private`foundMX = False
]

`Package`$pmDir = System`Private`savedPMDir

(***************************  Information Context  ****************************)

(* Programmers can use these values (using their full context, as in
   PacletManager`Information`$ReleaseNumber) to test version information about a user's
   PacletManager installation.
*)

(* The SyntaxQ checks allow this file to be used (albeit without meaningful
   values for these constants) even if the Ant preprocessing step that replaces
   the @name@ parts is not performed. This is for debugging in the Workbench.
*)
`Information`$VersionNumber = If[SyntaxQ["3.0"], ToExpression["3.0"], 0.0]
`Information`$ReleaseNumber = If[SyntaxQ["0"], ToExpression["0"], 0]
`Information`$CreationID = If[SyntaxQ["20190519204327"], ToExpression["20190519204327"], 0]
`Information`$CreationDate = If[SyntaxQ["{2019,05,19,20,43,27}"], ToExpression["{2019,05,19,20,43,27}"], {0,0,0,0,0,0}]
`Information`$Version = "PacletManager Version 3.0.0"


(*******************  Read in the implementation files.  **********************)

Begin["`Private`"]

processDecls[file_] :=
    Module[{strm, e, moreLines = True},
        strm = OpenRead[file];
        If[Head[strm] =!= InputStream,
            Return[$Failed]
        ];
        While[moreLines,
            e = Read[strm, Hold[Expression]];
            ReleaseHold[e];
            If[e === $Failed || MatchQ[e, Hold[_End]],
                moreLines = False
            ]
        ];
        Close[file]
    ]

End[]  (* `Private` *)


If[!`Private`foundMX,
    (* If we don't turn this message off, user will get shadowing warnings if a Global` symbol
       has the same name as any Package` symbols. It's not a valid warning, because Package`
       is never on ContextPath at the same time as Global`.
    *)
    `Private`wasOn = (Head[General::shdw] =!= $Off);
    Off[General::shdw];

    (* Make the Package` symbols visible to all implementation files. *)
    AppendTo[$ContextPath, "PacletManager`Package`"];

    `Private`implementationFiles =
        {
            ToFileName[{PacletManager`Package`$pmDir, "Kernel"}, "Utils.m"],
            ToFileName[{PacletManager`Package`$pmDir, "Kernel"}, "Paclet.m"],
            ToFileName[{PacletManager`Package`$pmDir, "Kernel"}, "Collection.m"],
            ToFileName[{PacletManager`Package`$pmDir, "Kernel"}, "LayoutDocsCollection.m"],
            ToFileName[{PacletManager`Package`$pmDir, "Kernel"}, "MemoryCollection.m"],
            ToFileName[{PacletManager`Package`$pmDir, "Kernel"}, "Extension.m"],
            ToFileName[{PacletManager`Package`$pmDir, "Kernel"}, "Documentation.m"],
            ToFileName[{PacletManager`Package`$pmDir, "Kernel"}, "Services.m"],
            ToFileName[{PacletManager`Package`$pmDir, "Kernel"}, "Packer.m"],
            ToFileName[{PacletManager`Package`$pmDir, "Kernel"}, "Zip.m"],
            ToFileName[{PacletManager`Package`$pmDir, "Kernel"}, "Manager.m"]
        };

    (* Read the public and package-visibility exports from the implementation files. *)
    `Private`processDecls /@ `Private`implementationFiles;

    (* Read in the code. *)
    Get /@ `Private`implementationFiles;

    If[`Private`wasOn, On[General::shdw]]
]

EndPackage[]

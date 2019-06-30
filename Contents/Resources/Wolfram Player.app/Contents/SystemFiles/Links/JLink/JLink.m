(* :Title: JLink *)

(* :Context: JLink` *)

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
   J/Link is a Mathematica enhancement that integrates Java and Mathematica. You can use J/Link to call
   Java from Mathematica or call Mathematica from Java. Find out more at www.wolfram.com/solutions/mathlink/jlink.

   J/Link uses a special system wherein one package context (JLink`) has its implementation
   split among a number of .m files. Each component file has its own private context, and also
   potentially introduces public symbols (in the JLink` context) and so-called "package" symbols,
   where the term "package" comes from Java terminology, referring to symbols that are visible
   everywhere within the implementation of J/Link, but not to clients.
*)

(* :Keywords: Java MathLink *)



BeginPackage["JLink`"]


Java::usage = "Java is only used as a generic symbol for some messages."


(* If we don't turn this message off, user will get shadowing warnings if a Global` symbol
   has the same name as any Package` symbols. It's not a valid warning, because Package`
   is never on ContextPath at the same time as Global`.
*)
`Private`shdwWasOn = (Head[General::shdw] =!= $Off)
Off[General::shdw]


Unprotect[{AddPeriodical, AddToClassPath, AllowRaggedArrays, 
   AppletViewer, BeginJavaBlock, ClassName, CloseFrontEnd, 
   ConnectToFrontEnd, Constructors, ConvertToCellExpression, DoModal, 
   EndJavaBlock, EndModal, EvaluateToImage, EvaluateToTypeset, Fields,
   FrontEndLink, FrontEndSharedQ, GetClass, GetComplexClass, 
   GetJavaException, GetJVM, ImplementJavaInterface, InstallJava, 
   InstanceOf, JavaBlock, JavaClassPath, JavaLink, JavaNew, 
   JavaObjectQ, JavaObjectToExpression, JavaShow, JavaThrow, 
   JavaUILink, JavaWindowToFront, KeepJavaObject, KernelSharedQ, 
   LoadedJavaClasses, LoadedJavaObjects, LoadJavaClass, MakeJavaExpr, 
   MakeJavaObject, Methods, ParentClass, Periodical, Periodicals, 
   RegisterJavaInitialization, ReinstallJava, ReleaseJavaObject, 
   RemovePeriodical, ReturnAsJavaObject, SameObjectQ, ServiceJava, 
   SetComplexClass, SetField, SetPeriodicalInterval, ShareFrontEnd, 
   ShareKernel, SharingLinks, ShowJavaConsole, UninstallJava, 
   UnshareFrontEnd, UnshareKernel, UseFrontEnd, UseJVM}]


(*********************  Try to locate the JLink .mx file  *********************)

`Package`$jlinkDir = DirectoryName[$InputFileName]
(* $jlinkDir is captured in mx file so we need to restore its value after loading the mx. Store a copy. *)
System`Private`savedJLinkDir = `Package`$jlinkDir

(* J/Link includes its Mathematica code in both a platform- and version-specific .mx file as well as .m files.
   For speed, we want to load the .mx file if appropriate, otherwise we fall back to reading the .m files.
   The test to load the .mx file is whether the version of Mathematica that built the .mx file is the same
   as the one the user is running. The !ValueQ[`Private`foundMX] part tests that this
   is not a re-load of the file within a session, which must use the .m files instead of .mx, otherwise the
   handful of globals that J/Link uses that are protected by If[!ValueQ[foo], foo = defaultvalue] would
   get reset to their defaults.
*)
If[StringMatchQ["12.0", DigitCharacter ~~ ___] && ToExpression["12.0"] == $VersionNumber && !ValueQ[`Private`foundMX],
    Quiet[
        Check[
            `Private`foundMX =
                (Get[ToFileName[{`Package`$jlinkDir, "Kernel", "SystemResources", $SystemID}, "JLink.mx"]] === Null),
        (* If messages generated (e.g. DumpGet::bgcorr), leave foundMX=False. *)
            `Private`foundMX = False
        ]
    ],
(* else *)
    `Private`foundMX = False
]

`Package`$jlinkDir = System`Private`savedJLinkDir
    
(***************************  Information Context  ***************************)

(* Programmers can use these values (using their full context, as in
   JLink`Information`$ReleaseNumber) to test version information about a user's
   J/Link installation.
*)

(* The SyntaxQ checks allow this file to be used (albeit without meaningful
   values for these constants) even if the Ant preprocessing step that replaces
   the @name@ parts is not performed. This is for debugging in the Workbench.
*)
JLink`Information`$VersionNumber = If[SyntaxQ["4.9"], ToExpression["4.9"], 0.0]
JLink`Information`$ReleaseNumber = If[SyntaxQ["1"], ToExpression["1"], 0]
JLink`Information`$BuildNumber = If[SyntaxQ["524"], ToExpression["524"], 0]
JLink`Information`$CreationID = If[SyntaxQ["20190519184916"], ToExpression["20190519184916"], 0]
JLink`Information`$CreationDate = If[SyntaxQ["{2019,05,19,18,49,16}"], ToExpression["{2019,05,19,18,49,16}"], {0,0,0,0,0,0}]
JLink`Information`$Version = "J/Link Version 4.9.1"


(*************************  obj@field = val syntax  **************************)

(* This allows natural syntax for setting Java fields: obj@field = val.
   We register a VetoableValueChange handler. The jlinkVetoFunction will be called whenever obj@field = val is called.
   We evaluate setField, which does the Java work, as a side-effect and then return False to prevent the
   Mathematica-side assignment from doing anything.
   This must be done outside the .mx file because it wouldn't be captured in the .mx file.
*)
Internal`AddHandler["VetoableValueChange", `Private`jlinkVetoFunction];
`Private`jlinkVetoFunction[HoldComplete[`Private`sym_Symbol, `Private`lhs_, `Private`rhs_, _, DownValues]] :=
    (JLink`CallJava`Private`setField[`Private`lhs, `Private`rhs]; False) /; JavaObjectQ[Unevaluated[`Private`sym]]


(* Read .m files if no appropriate JLink.mx was found. *)
If[!`Private`foundMX,

    (*******************  Read in the implementation files.  **********************)

    `Private`implementationFiles =
        {
            ToFileName[{`Package`$jlinkDir, "Kernel"}, "Debug.m"],
            ToFileName[{`Package`$jlinkDir, "Kernel"}, "InstallJava.m"],
            ToFileName[{`Package`$jlinkDir, "Kernel"}, "CallJava.m"],
            ToFileName[{`Package`$jlinkDir, "Kernel"}, "Java.m"],
            ToFileName[{`Package`$jlinkDir, "Kernel"}, "Exceptions.m"],
            ToFileName[{`Package`$jlinkDir, "Kernel"}, "Reflection.m"],
            ToFileName[{`Package`$jlinkDir, "Kernel"}, "JavaBlock.m"],
            ToFileName[{`Package`$jlinkDir, "Kernel"}, "MakeJavaObject.m"],
            ToFileName[{`Package`$jlinkDir, "Kernel"}, "Sharing.m"],
            ToFileName[{`Package`$jlinkDir, "Kernel"}, "Misc.m"],
            ToFileName[{`Package`$jlinkDir, "Kernel"}, "ArgumentTests.m"],
            ToFileName[{`Package`$jlinkDir, "Kernel"}, "EvaluateTo.m"],
            ToFileName[{`Package`$jlinkDir, "Kernel"}, "FrontEndServer.m"],
            ToFileName[{`Package`$jlinkDir, "Kernel"}, "JVMs.m"]
        };

    `Private`processDecls[`Private`file_] :=
        Module[{`Private`strm, `Private`e, `Private`moreLines = True},
            `Private`strm = OpenRead[`Private`file];
            If[Head[`Private`strm] =!= InputStream,
                Return[$Failed]
            ];
            While[`Private`moreLines,
                `Private`e = Read[`Private`strm, Hold[Expression]];
                ReleaseHold[`Private`e];
                If[`Private`e === $Failed || MatchQ[`Private`e, Hold[_End]],
                    `Private`moreLines = False
                ]
            ];
            Close[`Private`file]
        ];

    (* Make the Package` symbols visible to all implementation files. *)
    AppendTo[$ContextPath, "JLink`Package`"];

    (* Read the public and package-visibility exports from the implementation files. *)
    `Private`processDecls /@ `Private`implementationFiles;

    (* Read in the code. *)
    Get /@ `Private`implementationFiles;

]

If[`Private`shdwWasOn, On[General::shdw]]


Protect[{AddPeriodical, AddToClassPath, AllowRaggedArrays, 
  AppletViewer, BeginJavaBlock, ClassName, CloseFrontEnd, 
  ConnectToFrontEnd, Constructors, ConvertToCellExpression, DoModal, 
  EndJavaBlock, EndModal, EvaluateToImage, EvaluateToTypeset, Fields, 
  FrontEndLink, FrontEndSharedQ, GetClass, GetComplexClass, 
  GetJavaException, GetJVM, ImplementJavaInterface, InstallJava, 
  InstanceOf, JavaBlock, JavaClassPath, JavaLink, JavaNew, 
  JavaObjectQ, JavaObjectToExpression, JavaShow, JavaThrow, 
  JavaUILink, JavaWindowToFront, KeepJavaObject, KernelSharedQ, 
  LoadedJavaClasses, LoadedJavaObjects, LoadJavaClass, MakeJavaExpr, 
  MakeJavaObject, Methods, ParentClass, Periodical, Periodicals, 
  RegisterJavaInitialization, ReinstallJava, ReleaseJavaObject, 
  RemovePeriodical, ReturnAsJavaObject, SameObjectQ, ServiceJava, 
  SetComplexClass, SetField, SetPeriodicalInterval, ShareFrontEnd, 
  ShareKernel, SharingLinks, ShowJavaConsole, UninstallJava, 
  UnshareFrontEnd, UnshareKernel, UseFrontEnd, UseJVM}]


EndPackage[]

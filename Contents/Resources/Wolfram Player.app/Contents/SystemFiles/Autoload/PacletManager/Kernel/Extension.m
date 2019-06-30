(* :Title: Extension.m *)

(* :Author:
        Todd Gayley
        tgayley@wolfram.com
*)

(* :Package Version: 3.0 *)

(* :Mathematica Version: 8.1 *)

(* :Copyright: Mathematica source code (c) 1999-2019, Wolfram Research, Inc. All rights reserved. *)

(* :Discussion: This file is a component of the PacletManager Mathematica source code. *)


(* This "class" contains code for working with Extensions. *)


(**************************************************************************

    {"Documentation",
        LinkBase -> _String,                  [opt: defaults to paclet name]
        Language -> _String | {__String},     [opt: defaults to All]
        ProductName -> _String | {__String} | patt | All,   [opt: defaults to All]
        MainPage -> _String,                  [opt: defaults to first-named Guide page]
        Root -> _String,                      [opt: defaults to "Documentation" (also check "Documentation/Language")]
        Context -> _String,                   [opt: rarely used]
        Resources -> {(_String | {_String, _String})..}  [opt]
    }

    {"Kernel",
        Context -> _String | {(_String | {_String, _String})..},  [opt, but useless without]
        Root -> _String,                                          [opt: defaults to ".". I like Kernel better, but that
                                                                   has compatibility issues with legacy ($Path-based) apps]
        WolframVersion -> _String,                                [opt: defaults to All]
        ProductName -> _String | {__String} | patt | All,         [opt: defaults to All]
        SystemID -> _String | {__String},                         [opt: defaults to All]
        Symbols -> _String | Symbol | {(_String | Symbol)..}      [opt, but required for autoloading]
        HiddenImport -> True | False | "context`" | None          [opt; defaults to False; rarely used]
    }

    {"FrontEnd",
        Root -> _String                      [opt: defaults to FrontEnd]
        WolframVersion -> _String,           [opt: defaults to All]
        ProductName -> _String | {__String} | patt | All,   [opt: defaults to All]
        SystemID -> _String | {__String}     [opt: defaults to All]
    }

    {"Resource",
        Root -> _String,                                          [opt: defaults to "."]
        WolframVersion -> _String,                                [opt: defaults to All]
        ProductName -> _String | {__String} | patt | All,         [opt: defaults to All]
        SystemID -> _String | {__String},                         [opt: defaults to All]
        Resources -> {(_String | {_String, _String})...}          [opt]
    }

    {"LibraryLink",
        Root -> _String                           [opt: defaults to LibraryResources]
          .. I don't think I will require, or even support, the listing of library names.
             Seems sensible to not require it, but then why support it at all? I still need to
             search FileExistsQ for everything.
        WolframVersion -> _String,            [opt: defaults to All]
        ProductName -> _String | {__String} | patt | All,   [opt: defaults to All]
        SystemID -> _String | {__String}          [opt: defaults to All]
    }

    {"JLink",
        Root -> _String                      [opt: defaults to Java]
        WolframVersion -> _String,       [opt: defaults to All]
        SystemID -> _String | {__String}     [opt: defaults to All]
    }

    {"Path",
        Base -> _String,                   [opt: defaults to paclet name]
        Root -> _String                      [opt: defaults to "."]
        WolframVersion -> _String,       [opt: defaults to All]
        ProductName -> _String | {__String} | patt | All,   [opt: defaults to All]
        SystemID -> _String | {__String}     [opt: defaults to All]
    }

    TODO: NETLink, Demonstration

**************************************************************************)


Begin["`Package`"]

EXTgetType
EXTgetProperty
EXTgetResourcePath
EXTgetNamedResourcePath


$defaultDocumentationRoot = "Documentation"
$defaultKernelRoot = "."  (* Not "Kernel", although that makes more sense. For compatibility with old path-style apps. *)
$defaultJLinkRoot = "Java"
$defaultLibraryRoot = "LibraryResources"
$defaultFrontEndRoot = "FrontEnd"


End[]  (* `Package` *)


Begin["`Extension`Private`"]


EXTgetType[ext_] := First[ext]

(* Rest is needed because the first element of an extension is the type, not a rule. *)
EXTgetProperty[ext_, property_String] :=  EXTgetProperty[ext, property, Null]
EXTgetProperty[ext_, property_String, default_] := property /. Rest[ext] /. property -> default

(* Overloads that provide the default Root. This is a bit like having special "classes" for each ext type
   that encapsulate their special behaviors, such as the default root, which callers shouldn't have to know.
   Other extension properties might benefit from this type of special coding.
*)
EXTgetProperty[ext:{"Documentation", ___}, "Root"] := "Root" /. Rest[ext] /. "Root" -> $defaultDocumentationRoot
EXTgetProperty[ext:{"Kernel", ___}, "Root"] := "Root" /. Rest[ext] /. "Root" -> $defaultKernelRoot
EXTgetProperty[ext:{"Application", ___}, "Root"] := "Root" /. Rest[ext] /. "Root" -> $defaultKernelRoot
EXTgetProperty[ext:{"JLink", ___}, "Root"] := "Root" /. Rest[ext] /. "Root" -> $defaultJLinkRoot
EXTgetProperty[ext:{"LibraryLink", ___}, "Root"] := "Root" /. Rest[ext] /. "Root" -> $defaultLibraryRoot
EXTgetProperty[ext:{"FrontEnd", ___}, "Root"] := "Root" /. Rest[ext] /. "Root" -> $defaultFrontEndRoot
EXTgetProperty[ext_, "Root"] := "Root" /. Rest[ext] /. "Root" -> "."
(* Special rule to support older paclets that use "MathematicaVersion" instead of "WolframVersion". *)
EXTgetProperty[ext_, "WolframVersion", default_] := 
    (
        If[# =!= "MathematicaVersion", Return[#]]& ["MathematicaVersion" /. Rest[ext]];
        If[# =!= "WolframVersion", Return[#]]& ["WolframVersion" /. Rest[ext]];
        default
    )

(****  WORKING (to my knowledge), but UNUSED
EXThasNamedDocResource[ext:{"Documentation", ___}, requestedResourceName_String, ignoreCase:(True | False)] :=
    Module[{result, resName},
        result =
            Scan[
                Function[{resource},
                    resName = If[ListQ[resource], First[resource], resource];
                    (* Every Doc ext that names a main page or at least one guide page by default has "" as a named resource. *)
                    If[
                        requestedResourceName == "" && (EXTgetProperty[ext, "MainPage"] =!= Null || StringMatchQ[resName, "Guides/*"])
                            || ignoreCase && ToLowerCase[resName] == ToLowerCase[requestedResourceName]
                                || resName == requestedResourceName,
                       Return[True]
                    ]
                ],
                EXTgetProperty[ext, "Resources"]
            ];
        (* result will be Null if not found during the Scan. *)
        TrueQ[result]
    ]

(* False for extensions other than doc extensions. *)
EXThasNamedDocResource[__] = False
****)


(* Always returns a string. Answers the question "If this extension supplied the given resource, what would be its path
   from the extension root?" It doesn't tell whether the extension explicitly names the resource, or even if it has
   an explicit Resources section at all. It's for looking up docs, where we have already decided that the paclet
   has a good chance of providing the requested file, and we later decide if the path points to an actual file.
   This also handles resolution of MainPage (or "first Guide page") for Doc extenstions and resName of "".
   TODO: At one time I thought it was important to case-insensitive matching of resName here (in the Cases call); now I don't see why.
*)
EXTgetResourcePath[ext_, resName_String] :=
    Module[{path, mainPage},
        If[resName == "" && EXTgetType[ext] == "Documentation",
            (* Find MainPage or first Guide page. *)
            mainPage = EXTgetProperty[ext, "MainPage"];
            If[!StringQ[mainPage],
                mainPage =
                    Scan[Which[StringQ[#] && StringMatchQ[#, "Guides/*"], Return[#],
                               MatchQ[#, {_String, _String}] && StringMatchQ[First[#], "Guides/*"], Return[First[#]]
                         ]&,
                         EXTgetProperty[ext, "Resources", {}]
                    ];
            ];
            If[StringQ[mainPage],
                (* We found a mapping for the main page to a new resource name (e.g., "Guides/MyPaclet". Call again
                   with this new name to get it resolved to a path.
                *)
                EXTgetResourcePath[ext, mainPage],
            (* else *)
                (* We always want to return a string from this function, regardless of whether the file exists or not.
                   When resName == "" we have no basis on which to build a potential filename, so we just make up a bogus one.
                *)
                "nosuchfile.nb"
            ],
        (* else *)
            path = Cases[EXTgetProperty[ext, "Resources", {}], {resName, p_} :> p];
            If[Length[path] > 0,
                (* Expand the # placeholder, if present. *)
                StringReplace[First[path], "#" -> resName],
            (* else *)
                (* The resource isn't listed with a path, or isn't listed at all. Use the default. *)
                resName <> Switch[EXTgetType[ext], "Documentation", ".nb", "Demonstration", ".nbp", "CDF", ".cdf", _, ""]
            ]
        ]
    ]

(* This is like EXTgerResourcePath, except it requires that the resource be named explicitly in a Resource field.
   This function is used by PacletResource, which wants to find only explicitly-named resources.
   Allows use of # placeholder notation.
   Returns a string or Null.
*)
EXTgetNamedResourcePath[ext_, resName_String] :=
    Module[{paths},
        paths = Cases[EXTgetProperty[ext, "Resources", {}], p:({resName, _} | resName) :> Last[Flatten[{p}]]];
        If[Length[paths] > 0,
            (* Expand the # placeholder, if present. *)
            StringReplace[First[paths], "#" -> resName],
        (* else *)
            Null
        ]
    ]


End[]


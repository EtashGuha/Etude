(* :Title: COM.m *)

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


(*<!--Public From COM.m

CreateCOMObject::usage =
"CreateCOMObject[str] creates a COM object specified by the string str, which can be either a ProgID (such as \"Excel.Application\") \
or a CLSID (such as \"{8E27C92B-1264-101C-8A2F-040224009C02}\"). CreateCOMObject is analogous to the COM API function CoCreateInstance."

GetActiveCOMObject::usage =
"GetActiveCOMObject[str] acquires an already-running COM object specified by the string str, which can be either a ProgID \
(such as \"Excel.Application\") or a CLSID (such as \"{8E27C92B-1264-101C-8A2F-040224009C02}\"). GetActiveCOMObject is analogous \
to the COM API function GetActiveObject."

ReleaseCOMObject::usage =
"ReleaseCOMObject[obj] releases COM resources held by the specified .NET object. Although any COM resources will be \
released when the .NET object is garbage-collected, it is often desirable to force their release explicitly. Each call to \
ReleaseCOMObject decrements the reference count on the COM resources held by the object. The resources will be freed when \
the reference count goes to 0 (or the .NET object is garbage-collected). ReleaseCOMObject returns the new reference count on the \
COM resources, or a list of these counts if it was passed a list of objects. ReleaseCOMObject should not be \
confused with ReleaseNETObject. ReleaseNETObject allows the .NET object to be garbage-collected, but does not force this to \
happen in a timely manner. ReleaseCOMObject can be used to force the immediate release of the COM resources held by the object."

CastCOMObject::usage = 
"CastCOMObject is deprecated. Use the more general CastNETObject instead."

LoadCOMTypeLibrary::usage =
"LoadCOMTypeLibrary[typeLibPath] creates a so-called \"interop\" assembly from the named type library and loads that assembly. \
Once a type library has been loaded in this way, all its types will have managed equivalents created for them, so you can program \
with these types as if they were native .NET types. LoadCOMTypeLibrary is the programmatic equivalent of running the tlbimp.exe \
tool that is part of the .NET Framework SDK. The assembly can optionally be saved to disk (using the SaveAssemblyAs option) so that \
you do not have to call LoadCOMTypeLibrary in the future. If you plan to do serious work with COM objects described in a given type \
library, it is recommended that you use LoadCOMTypeLibrary or the tlbimp.exe tool to create an interop assembly and then use that \
assembly."

SafeArrayAsArray::usage =
"SafeArrayAsArray is an option to LoadCOMTypeLibrary that specifies whether to import all SAFEARRAY's as System.Array rather than \
a typed, single dimensional managed array. The default is False. See the .NET Framework documentation for the \
System.Runtime.InteropServices.TypeLibImporterFlags enumeration for more details on this advanced option."

SaveAssemblyAs::usage =
"SaveAssemblyAs is an option to LoadCOMTypeLibrary that allows you to specify a file name into which to write the interop assembly \
that gets generated. LoadCOMTypeLibrary can be time-consuming for large type libraries, so it is useful to save the generated \
assembly in a file. It can then be loaded directly, bypassing future calls to LoadCOMTypeLibrary. You can specify a directory name only \
and get a default name for the assembly."

-->*)

(*<!--Package From COM.m

isCOMNonPrimitiveFieldOrSimpleProp

-->*)


(* Current context will be NETLink`. *)

Begin["`COM`Private`"]


(*******************************  CreateCOMObject/GetActiveCOMObject  ***********************************)

(* TODO: Allow the "server" arg (to Type.GetTypeFromCLSID() to be specified by user. *)

(* Creates the specified COM object from a ProgID or CLSID. Like the VB function CreateObject.
   If the clsIDOrProgID arg is a CLSID, it can be in any format that the .NET Guid class constructor accepts:
        "dddddddd-dddd-dddd-dddd-dddddddddddd"
        "{dddddddd-dddd-dddd-dddd-dddddddddddd}"
        "{0xdddddddd,0xdddd, 0xdddd,{0xdd},{0xdd},{0xdd},{0xdd},{0xdd},{0xdd},{0xdd},{0xdd}}"
*)

CreateCOMObject[clsIDOrProgID_String] :=
    Block[{$internalNETExceptionHandler = associateMessageWithSymbol[CreateCOMObject]},
        If[Head[InstallNET[]] =!= LinkObject,
            Message[CreateCOMObject::netlink, "CreateCOMObject"];
            Return[$Failed]
        ];
        nCreateCOM[clsIDOrProgID]
    ]

CreateCOMObject[clsIDOrProgID_?NETObjectQ] :=
    CreateCOMObject[clsIDOrProgID@ToString[]] /; InstanceOf[clsIDOrProgID, "System.Guid"]


GetActiveCOMObject[clsIDOrProgID_String] :=
    Block[{$internalNETExceptionHandler = associateMessageWithSymbol[GetActiveCOMObject]},
        If[Head[InstallNET[]] =!= LinkObject,
            Message[CreateCOMObject::netlink, "CreateCOMObject"];
            Return[$Failed]
        ];
        nGetActiveCOM[clsIDOrProgID]
    ]

GetActiveCOMObject[clsIDOrProgID_?NETObjectQ] :=
    GetActiveCOMObject[clsIDOrProgID@ToString[]] /; InstanceOf[clsIDOrProgID, "System.Guid"]


(*****************************************  ReleaseCOMObject  *****************************************)

(* Releases the COM object. Although this happens when the .NET object is garbage-collected, it is useful to have
   a way to force the release.
*)

ReleaseCOMObject[obj_?NETObjectQ] :=
    Block[{$internalNETExceptionHandler = associateMessageWithSymbol[ReleaseCOMObject]},
        nReleaseCOM[obj]
    ]

ReleaseCOMObject[objs__?NETObjectQ] :=
    ReleaseCOMObject[{objs}]
    
ReleaseCOMObject[objs:{___?NETObjectQ}] :=
    Block[{$internalNETExceptionHandler = associateMessageWithSymbol[ReleaseCOMObject]},
        nReleaseCOM /@ objs
    ]

(* Message here is defined for General. *)
ReleaseCOMObject[obj_] := (Message[ReleaseCOMObject::netobj1, obj]; $Failed)


(*****************************************  CastCOMObject  *****************************************)

(* Deprecated. *)
CastCOMObject = CastNETObject


(*****************************************  LoadCOMTypeLibrary  *****************************************)

(* Loads the COM type library by programmatically generating an interop assembly, and then loading that assembly.
   The assembly can optionally be saved to disk, so it can later be loaded directly without re-importing the type library.
*)

LoadCOMTypeLibrary::asmfile = "Invalid value for SaveAssemblyAsFile option: `1`. Must be a string specifying the full path to the assembly file to be written."
LoadCOMTypeLibrary::pia = "A Primary Interop Assembly (PIA) has already been installed on your system by the vendor of this type library. The PIA will be used instead of generating and writing out a new assembly."

Options[LoadCOMTypeLibrary] = {SafeArrayAsArray -> False, SaveAssemblyAs -> None}

LoadCOMTypeLibrary[path_String, opts___?OptionQ] :=
    Module[{safeArrayAsArray, assemblyFile, result, asm, foundPIAInstead},
        If[Head[InstallNET[]] =!= LinkObject,
            Message[LoadCOMTypeLibrary::netlink, "LoadCOMTypeLibrary"];
            Return[$Failed]
        ];
        {safeArrayAsArray, assemblyFile} = contextIndependentOptions[{SafeArrayAsArray, SaveAssemblyAs}, Flatten[{opts}], Options[LoadCOMTypeLibrary]];
        If[assemblyFile === None, assemblyFile = ""];
        If[!StringQ[assemblyFile],
            Message[LoadCOMTypeLibrary::asmfile, assemblyFile];
            assemblyFile = ""
        ];
        (* Ensure that if assemblyFile is only a directory spec, it ends with a dir separator (this is required in the C# code). *)
        If[assemblyFile != "" && FileType[assemblyFile] === Directory && StringTake[assemblyFile, -1] != $PathnameSeparator,
            assemblyFile = assemblyFile <> $PathnameSeparator
        ];
        result = nLoadTypeLibrary[path, TrueQ[safeArrayAsArray], assemblyFile];
        If[result =!= $Failed,
            {asm, foundPIAInstead} = result;
            (* If the user specified a value for the SaveAssemblyAs option and we found a PIA instead, warn them that we did
               not write out an assembly as requested.
            *)
            If[foundPIAInstead && assemblyFile != "",
                Message[LoadCOMTypeLibrary::pia]
            ];
            LoadNETAssembly[asm],
        (* else *)
            $Failed
        ]
    ]


(***********************************  isCOMNonPrimitiveFieldOrSimpleProp  *************************************)

(* This is used to decide whether or not the given member is a COM prop on the object. This is needed to work
   out precedence in chained calls: obj@prop@foo  ---> (obj@prop)@foo.
*)

isCOMNonPrimitiveFieldOrSimpleProp[obj_, memberName_String] :=
    nIsCOMProp[obj, memberName]


End[]

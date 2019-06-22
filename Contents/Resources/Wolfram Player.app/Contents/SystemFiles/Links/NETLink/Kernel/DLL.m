(* :Title: DLL.m *)

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


(*<!--Public From DLL.m

DefineDLLFunction::usage =
"DefineDLLFunction[\"funcName\", \"dllName\", returnType, argTypes] returns a Wolfram Language function that calls the \
specified function in the specified unmanaged DLL. The argsTypes argument is a list of type specifications for the arguments, and \
can be omitted if the function takes no arguments. The type specifications for argTypes and returnType are strings or, \
less commonly, NETType expressions. Strings can be given in C-style syntax (such as \"char*\"), C# syntax \
(\"string\"), Visual Basic .NET syntax (\"ByVal As String\") or by using many Windows API types (such as \"HWND\", \"DWORD\", \
\"BOOL\", and so on.) Priority is given to the C interpretation of type names, so char and long have their meanings in C \
(1 and 4 bytes, respectively), not C#. You need to give the full pathname to the DLL if it is not located in a standard \
location (standard locations are a directory on your system PATH or a DLL subdirectory in a Wolfram System application directory, \
such as $InstallationDirectory\\AddOns\\Applications\\SomeApp\\DLL). DefineDLLFunction[\"declaration\"] lets you write a full \
C#-syntax 'extern' function declaration. Use this form when you need to write a complex function declaration that requires \
features not available using options to DefineDLLFunction, such as specific \"MarshalAs\" atributes on each of the parameters."

ReferencedAssemblies::usage =
"ReferencedAssemblies is an option to DefineDLLFunction that specifies assemblies needed in your function declaration. \
For example, if your DLL function involves a type from another assembly, such as System.Drawing.Rectangle from the \
System.Drawing assembly, you would specify ReferencedAssemblies->{\"System.Drawing.dll\"}. Note that you should use the \
actual filename of the assembly, not its display name (which would be just \"System.Drawing\" in this example)."

MarshalStringsAs::usage =
"MarshalStringsAs is an option to DefineDLLFunction that specifies how string arguments should be marshaled into the DLL function. \
This applies to any arguments that are mapped to the System.String class, which includes types specified in your declaration \
as \"char*\", \"string\", or \"ByVal As String\". The possible values are \"ANSI\", \"Unicode\", and Automatic. The default is \
\"ANSI\", meaning that strings will be sent as single-byte C-style strings. This is appropriate for most DLL functions, which \
generally expect C-style strings. \"Unicode\" means to send strings as 2-byte Unicode strings. Use this if you know the function \
expects 2-byte strings (e.g., if the type name in the C prototype is wchar_t* ). The Automatic setting picks the platform \
default (\"Unicode\" on Windows NT/2000/XP, \"ANSI\" on 98/ME). Automatic should rarely be used, as it is intended mainly for \
certain Windows API functions that automatically switch behaviors on different versions of Windows."

CallingConvention::usage =
"CallingConvention is an option to DefineDLLFunction that specifies what calling convention the DLL function uses. The possible \
values are \"StdCall\", \"CDecl\", \"ThisCall\", \"WinApi\", and Automatic. The string values for this option are not case \
sensitive. The default is Automatic, which means use the platform default (\"StdCall\" on all platforms except Windows CE, \
which is not supported by .NET/Link). \
Most DLL funtions use the \"StdCall\" convention. For more information on these values, see the .NET Framework documentation \
for the System.Runtime.InteropServices.CallingConvention enumeration."

-->*)

(*<!--Package From DLL.m

fixType
$csharpToNETTypeRules
$vbToNETTypeRules
$netTypeToCSharpRules
$netTypeToVBRules

-->*)


(* Current context will be NETLink`. *)

Begin["`DLL`Private`"]


(*****************************************  DefineDLLFunction  ****************************************)

DefineDLLFunction::compileerr =
"The compiler reported errors in your DLL function declaration: `1`"

DefineDLLFunction::strformat =
"The value given for the MarshalStringsAs option was not valid and will be ignored. The only valid values \
are \"ANSI\", \"Unicode\" and Automatic."

DefineDLLFunction::callconv =
"The value given for the CallingConvention option was not valid and will be ignored. It must be Automatic or \
one of the following strings (case is ignored): \"WinApi\", \"CDecl\", \"StdCall\", \"ThisCall\"."

DefineDLLFunction::vb =
"You must use C# syntax instead of VB syntax when writing a DLL function declaration as a complete string of code."


Options[DefineDLLFunction] =
    {ReferencedAssemblies -> Automatic, MarshalStringsAs -> "ANSI", CallingConvention -> Automatic}


DefineDLLFunction[funcName_String, dllName_String, returnType_String:"System.Void", opts___?OptionQ] :=
    defDLL["", funcName, dllName, returnType, {}, opts]

DefineDLLFunction[funcName_String, dllName_String, returnType_String:"System.Void", argTypes:{(_String | _NETType)...}, opts___?OptionQ] :=
    defDLL["", funcName, dllName, returnType, argTypes, opts]
    
DefineDLLFunction[declaration_String, opts___?OptionQ] :=
    defDLL[declaration, "", "", "", {}, opts]


(* This worker function will be called with either declaration meaningful and everything else ignored, or vice versa. *)

defDLL[declaration_String, fName_String, dllName_String, returnType_String, argTypes:{(_String | _NETType)...}, opts___?OptionQ] :=
    Module[{result, newTypeName, typeContext, appDLLs, appDLLNames, fullDLLPath, libName, funcName, argCount,
                fixedReturnType, fixedArgTypes, areOutParams, refAsm, strFormat, callConv, lang},
        If[Head[InstallNET[]] =!= LinkObject,
            Message[DefineDLLFunction::netlink, "DefineDLLFunction"];
            Return[$Failed]
        ];
        refAsm = contextIndependentOptions[ReferencedAssemblies, Flatten[{opts}], Options[DefineDLLFunction]];
        If[!MatchQ[refAsm, {___String}], refAsm = {}];
        If[declaration != "",
            (* Definition to be based on full declaration provided by user. *)
            If[StringMatchQ[ToUpperCase[declaration], "*<DLLIMPORT*"] || StringMatchQ[ToUpperCase[declaration], "*DECLARE *"],
                lang = "vb",
            (* else *)
                lang = "csharp"
            ];
            (* TODO: Get VB syntax working and remove this If function. *)
            If[lang == "vb",
                Message[DefineDLLFunction::vb];
                Return[$Failed]
            ];
            (* Modify the dll name to include the full path if the DLL is found in a special app directory. Note that
               the logic used for extracting the dll name assumes that it is the only part of the declaration that
               would be enclosed in quotes.
            *)
            libName = ToLowerCase[StringJoin @@ (Characters[declaration] /. {__, "\"", dll__, "\"", __} -> {dll})];
            If[StringMatchQ[libName, "*.dll"] && !StringMatchQ[libName, "*\\*"],
                appDLLs = findAppDLLs[];
                (* Get the filename part of the paths in app DLLs. *)
                appDLLNames = ToLowerCase /@ (StringDrop[#, Last[Flatten[StringPosition[#, $PathnameSeparator]]]]&) /@ appDLLs;
                If[MemberQ[appDLLNames, libName],
                    fullDLLPath = First[Cases[Transpose[{appDLLNames, appDLLs}], {libName, path_} -> path]];
                    
                ]
            ];
            result = nCreateDLL2[declaration, refAsm, lang];
            If[!MatchQ[result, {_String, _String, _String}],
                (* There was a compiler error. result will be {"report"}. *)
                Message[DefineDLLFunction::compileerr, First[result]];
                Return[$Failed]
            ];
            {newTypeName, funcName, argCount} = result;
            (* argCount comes back as a string. *)
            argCount = ToExpression[argCount],
        (* else *)
            LoadNETAssembly /@ refAsm;
            {strFormat, callConv} =
                contextIndependentOptions[{MarshalStringsAs, CallingConvention}, Flatten[{opts}], Options[DefineDLLFunction]];
            If[strFormat === Automatic, strFormat = "auto"];
            If[!StringQ[strFormat] || !MemberQ[{"ansi", "unicode", "auto"}, ToLowerCase[strFormat]],
                Message[DefineDLLFunction::strformat, strFormat];
                strFormat = "ansi"
            ];
            If[callConv === Automatic, callConv = "winapi"];
            If[StringQ[callConv] && MemberQ[{"stdcall", "thiscall", "winapi", "cdecl"}, ToLowerCase[callConv]],
                callConv = ToLowerCase[callConv],
            (* else *)
                Message[DefineDLLFunction::callconv, callConv];
                callConv = "winapi"
            ];
            fixedReturnType = fixType[returnType];
            fixedArgTypes = fixType /@ argTypes;
            areOutParams = (StringQ[#] && StringMatchQ[#, "out *"])& /@ argTypes;
            (* If the user specified a DLL name without path info, look for it among app\DLL dirs. Not an error if it is
               not found, as it could also be on PATH.
            *)
            fullDLLPath = dllName;
            If[!StringMatchQ[dllName, "*\\*"],
                appDLLs = findAppDLLs[];
                (* Get the filename part of the paths in app DLLs. *)
                appDLLNames = ToLowerCase /@ (StringDrop[#, Last[Flatten[StringPosition[#, $PathnameSeparator]]]]&) /@ appDLLs;
                If[MemberQ[appDLLNames, ToLowerCase[dllName]],
                    fullDLLPath = First[Cases[Transpose[{appDLLNames, appDLLs}], {ToLowerCase[dllName], path_} -> path]]
                ]
            ];
            newTypeName = nCreateDLL1[fName, fullDLLPath, callConv, fixedReturnType, fixedArgTypes, areOutParams, ToLowerCase[strFormat]];
            funcName = fName;
            argCount = Length[argTypes]
        ];
        If[StringQ[newTypeName],
            LoadNETType[newTypeName, AllowShortContext->False, StaticsVisible->False];
            typeContext = StringReplace[newTypeName, "." -> "`"] <> "`";
            ToExpression["Function[Null, If[" <> ToString[checkArgCount] <> "[\"" <> funcName <> "\",{##1}," <>
                            ToString[argCount] <> "]," <> typeContext <> toLegalName[funcName] <> "[##1]," <>
                            "$Failed], {HoldAll}]"],
        (* else *)
            (* Rely on message emitted by CreateDLLCall. *)
            $Failed
        ]
    ]


checkArgCount[fname_String, args_List, expectedArgc_Integer] :=
    If[Length[args] == expectedArgc,
        True,
    (* else *)
        Which[
            Length[args] == 1,
                Message[NET::argr, "DLL function " <> fname, expectedArgc],
            expectedArgc == 1,
                Message[NET::argx, "DLL function " <> fname, Length[args]],
            True,
                Message[NET::argrx, "DLL function " <> fname, Length[args], expectedArgc]
        ];
        False
    ]


(* Returns a list of all DLLs in the Libraries/$SystemID subdirectory of any app directory in any of the standard app locations. *)
findAppDLLs[] :=
    Module[{dllDirs},
        dllDirs = ToFileName[{#, "Libraries", $SystemID}]& /@ appDirs[];
        (* Here we add the special WRI SystemFiles/Libraries/$SystemID dir. *)
        PrependTo[dllDirs, ToFileName[{$TopDirectory, "SystemFiles", "Libraries", $SystemID}]];
        FileNames["*.dll", dllDirs]
    ]


(* Finds all top-level dirs for all apps in standard locations. This function is generic and reusable, although
   it currently is not needed anywhere else in .NET/Link.
*)
appDirs[] :=
    Module[{appPaths},
        appPaths = {ToFileName[{$TopDirectory, "AddOns", "Applications"}], ToFileName[{$TopDirectory, "AddOns", "ExtraPackages"}]};
        If[StringQ[$AddOnsDirectory],
            (* This branch is for 4.2 and later. *)
            PrependTo[appPaths, ToFileName[{$AddOnsDirectory, "Applications"}]]
        ];
        If[StringQ[$UserAddOnsDirectory],
            (* 4.2 and later *)
            PrependTo[appPaths, ToFileName[{$UserAddOnsDirectory, "Applications"}]],
        (* else *)
            PrependTo[appPaths, ToFileName[{$PreferencesDirectory, "AddOns", "Applications"}]]
        ];
        Select[FileNames["*", appPaths], (FileType[#] === Directory)&]
    ]


(****************************  fixType  ****************************)

(* fixType converts type names in C, C#, or VB syntax into .NET Framework names.
   It is used by DefineDLLFunction and also by DefieNETDelegate in another source file.
   
   fixType also takes NETType expressions and returns their assembly-qualified type names.
   This is just so that it can be conveniently used in places where the user could supply a
   string or NETType.
*)

fixType[type_String] :=
    Module[{isByRef = False, fixedType = type, asPos, isVBArray},
        If[StringMatchQ[fixedType, "const *"],
            fixedType = StringDrop[fixedType, 6]
        ];
        Which[
            StringMatchQ[fixedType, "out *"] || StringMatchQ[fixedType, "ref *"],
                isByRef = True;
                fixedType = StringDrop[fixedType, 4],
            StringMatchQ[fixedType, "ByRef *", IgnoreCase->True],
                isByRef = True;
                fixedType = StringDrop[fixedType, 6],
            StringMatchQ[fixedType, "ByVal *", IgnoreCase->True],
                fixedType = StringDrop[fixedType, 6],
            StringMatchQ[fixedType, "*&"],
                (* Always strip off any trailing & and add it again at the end. This allows us to handle
                   mistaken attempts to use a & in C++-style syntax (int&).
                *)
                isByRef = True;
                fixedType = StringDrop[fixedType, -1]
        ];
        (* This for VB-style type decls (x As Integer). *)
        If[StringMatchQ[fixedType, "* As *", IgnoreCase->True],
            asPos = First[Flatten[StringPosition[fixedType, " As ", IgnoreCase->True]]];
            (* Convert VB array notation "x() As Integer" to "Integer[]" .*)
            isVBArray = StringMatchQ[fixedType, "*) As *", IgnoreCase->True];
            fixedType = StringDrop[fixedType, asPos + 3] <> If[isVBArray, "[]" , ""]
        ];
        fixedType = fixedType /. $winToNETTypeRules /. $cToNETTypeRules /. $csharpToNETTypeRules /. $vbToNETTypeRules;
        If[isByRef,
            fixedType <> "&",
        (* else *)
            fixedType
        ]
    ]
    
fixType[type_NETType] := getAQTypeName[type]


$cToNETTypeRules = {
    "void" -> "System.Void",
    "int" -> "System.Int32",
    "unsigned" -> "System.UInt32",
    "unsigned int" -> "System.UInt32",
    "short" -> "System.Int16",
    "unsigned short" -> "System.UInt16",
    "char" -> "System.SByte",
    "unsigned char" -> "System.Byte",
    "long" -> "System.Int32",
    "unsigned long" -> "System.UInt32",
    "float" -> "System.Single",
    "double" -> "System.Double",
    "bool" -> "System.Boolean",
    "int[]" -> "System.Int32[]",
    "unsigned int[]" -> "System.UInt32[]",
    "unsigned[]" -> "System.UInt32[]",
    "char[]" -> "System.SByte[]",
    "unsigned char[]" -> "System.Byte[]",
    "short[]" -> "System.Int16[]",
    "unsigned short[]" -> "System.UInt16[]",
    "long[]" -> "System.Int32[]",
    "unsigned long[]" -> "System.UInt32[]",
    "float[]" -> "System.Single[]",
    "double[]" -> "System.Double[]",
    "bool[]" -> "System.Boolean[]",
    "int*" -> "System.Int32&",
    "unsigned int*" -> "System.UInt32&",
    "unsigned*" -> "System.UInt32&",
    "char*" -> "System.String",
    "unsigned char*" -> "System.Byte&",
    "short*" -> "System.Int16&",
    "unsigned short*" -> "System.UInt16&",
    "long*" -> "System.Int32&",
    "unsigned long*" -> "System.UInt32&",
    "float*" -> "System.Single&",
    "double*" -> "System.Double&",
    "bool*" -> "System.Boolean*",
    "void*" -> "System.IntPtr"
}

$winToNETTypeRules = {
    "BOOL" -> "System.Int32",
    "HWND" -> "System.IntPtr",
    "INT" -> "System.Int32",
    "UINT" -> "System.UInt32",
    "LONG" -> "System.Int32",
    "DWORD" -> "System.Int32",
    "LPARAM" -> "System.Int32",
    "WPARAM" -> "System.Int32",
    "WORD" -> "System.Int16",
    "SHORT" -> "System.Int16",
    "BYTE" -> "System.Byte",
    "LPSTR" -> "System.String",
    "LPCSTR" -> "System.String",
    "LPTSTR" -> "System.String",
    "LPCTSTR" -> "System.String",
    "HMENU" -> "System.IntPtr",
    "INT_PTR" -> "System.IntPtr",
    "UINT_PTR" -> "System.UIntPtr",
    "PTR" -> "System.IntPtr"
}

$csharpToNETTypeRules = {
    "void" -> "System.Void",
    "int" -> "System.Int32",
    "unsigned" -> "System.UInt32",
    "unsigned int" -> "System.UInt32",
    "byte" -> "System.Byte",
    "sbyte" -> "System.SByte",
    "short" -> "System.Int16",
    "char" -> "System.Char",
    "unsigned short" -> "System.UInt16",
    "long" -> "System.Int64",
    "unsigned long" -> "System.UInt64",
    "float" -> "System.Single",
    "double" -> "System.Double",
    "bool" -> "System.Boolean",
    "string" -> "System.String",
    "object" -> "System.Object",
    "int[]" -> "System.Int32[]",
    "unsigned int[]" -> "System.UInt32[]",
    "unsigned[]" -> "System.UInt32[]",
    "byte[]" -> "System.Byte[]",
    "sbyte[]" -> "System.SByte[]",
    "short[]" -> "System.Int16[]",
    "unsigned short[]" -> "System.UInt16[]",
    "long[]" -> "System.Int64[]",
    "unsigned long[]" -> "System.UInt64[]",
    "float[]" -> "System.Single[]",
    "double[]" -> "System.Double[]",
    "bool[]" -> "System.Boolean[]",
    "string[]" -> "System.String[]",
    "object[]" -> "System.Object[]",
    "int*" -> "System.Int32&",
    "unsigned int*" -> "System.UInt32&",
    "unsigned*" -> "System.UInt32&",
    "byte*" -> "System.Byte&",
    "sbyte*" -> "System.SByte&",
    "short*" -> "System.Int16&",
    "unsigned short*" -> "System.UInt16&",
    "long*" -> "System.Int64&",
    "unsigned long*" -> "System.UInt64&",
    "float*" -> "System.Single&",
    "double*" -> "System.Double&",
    "bool*" -> "System.Boolean&",
    "string*" -> "System.String&"
}

$vbToNETTypeRules = {
    (* Other VB types agree with .NET type names. *)
    "Short" -> "System.Int16",
    "Integer" -> "System.Int32",
    "Long" -> "System.Int64",
    (* Users would never enter array notation with [] for VB, but fixType constructs this form. *)
    "Short[]" -> "System.Int16[]",
    "Integer[]" -> "System.Int32[]",
    "Long[]" -> "System.Int64[]"
}


(* These lists aren't used in this file, but since their counterparts are, we'll define them here anyway. *)

$netTypeToCSharpRules = Reverse /@ $csharpToNETTypeRules
$netTypeToVBRules = Reverse /@ $vbToNETTypeRules
(* No need for $netTypeToCRules or $netTypeToWinRules. *)


$cToNETTypeRules = Dispatch[$cToNETTypeRules]
$winToNETTypeRules = Dispatch[$winToNETTypeRules]
$csharpToNETTypeRules = Dispatch[$csharpToNETTypeRules]
$vbToNETTypeRules = Dispatch[$vbToNETTypeRules]

$netTypeToCSharpRules = Dispatch[$netTypeToCSharpRules]
$netTypeToVBRules = Dispatch[$netTypeToVBRules]


End[]

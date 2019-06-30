(* :Title: DefineNETClass.m *)

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


(*<!--Public From DefineNETClass.m

BeginNETClass::usage =
""

EndNETClass::usage =
""

DefineNETClass::usage =
""

NETMethod::usage =
""

MethodAttributes::usage =
""

DefineNETMethod::usage =
""

ImportNamespaces::usage =
""

-->*)

(*<!--Package From DefineNETClass.m

methodCallbackWrapper

-->*)


(* Current context will be NETLink`. *)

Begin["`DefineNETClass`Private`"]

(***

TODO: Don't like the name NETMethod. DefineNETMethod is used for aomething else.

In DefineNETMethod, get rid of need to put in "public static". ACtually, I left this out in my tempversion,
so the real issue is "put back the ability to include it without having things break".

***)


BeginNETClass::netlink = "BeginNETClass cannot operate because InstallNET[] failed to launch the .NET runtime."
EndNETClass::begin = "EndNETClass was called without a matching BeginNETClass."
EndNETClass::fail = "The .NET class could not be created."

$classBuilder

BeginNETClass[name_String, parentClass_String:"", interfaces___String] :=
    Module[{assemblyName = ""},
        (* TODOL: support assemblyName as an arg. *)
        If[Head[InstallNET[]] =!= LinkObject,
            Message[BeginNETClass::netlink];
            Return[$Failed]
        ];
        $classBuilder = NETNew["Wolfram.NETLink.Internal.DefineClassHelper", assemblyName];
        $classBuilder@BeginClass[name, parentClass, {interfaces}]
    ]

EndNETClass[] :=
    JavaBlock[
        Module[{type},
            If[NETObjectQ[$classBuilder],
                type = $classBuilder@EndClass[];
                If[NETObjectQ[type],
                    LoadNETType[type],
                (* else *)
                    Message[EndNETClass::fail];
                    $Failed
                ],
            (* else *)
                Message[EndNETClass::begin];
                $Failed
            ]
        ]
    ]

Attributes[DefineNETClass] = {HoldRest}

DefineNETClass[name_String, parentClass_String:"", interfaces___String, body_] :=
    Module[{},
        If[BeginNETClass[name, parentClass, interfaces] =!= $Failed,
            body;
            EndNETClass[],
        (* else *)
            $Failed
        ]
    ]
    
    
NETMethod::class = "NETMethod must be called within DefineNETClass, or between BeginNETClass and EndNETClass."
NETMethod::methattrs = "Bad value supplied for the MethodAttributes option: `1`. It must be a list of zero or more strings."

Options[NETMethod] = {MethodAttributes -> {"public"}}

NETMethod[name_String, retType_String:"System.Void", paramTypes:{___String}, mFunc_Symbol, opts___?Option] :=
    NETMethod[name, retType, paramTypes, Context[mFunc] <> SymbolName[mFunc], opts]

NETMethod[name_String, retType_String:"System.Void", paramTypes:{___String}, mFunc_String, opts___?OptionQ] :=
    Module[{methAttrs, isStatic, isVirtual, isOverride},
        If[!NETObjectQ[$classBuilder],
            Message[NETMethod::class];
            Return[$Failed]
        ];
        (* LoadNETType["System.Reflection.MethodAttributes"]; *)
        methAttrs = MethodAttributes /. Flatten[{opts}] /. Options[NETMethod];
        If[!MatchQ[methAttrs, {___String}],
            Message[NETMethod::methattrs, methAttrs];
            methAttrs = {"public"}
        ];
        methAttrs = ToLowerCase /@ methAttrs;
        (* In VB, "Shadows" == "new", "Overrides" == "override", "Overridable" == "virtual". *)
        isStatic = MemberQ[methAttrs, "static"] || MemberQ[methAttrs, "shared"];
        isVirtual = MemberQ[methAttrs, "virtual"] || MemberQ[methAttrs, "overridable"];
        isOverride = MemberQ[methAttrs, "override"] || MemberQ[methAttrs, "overrides"];
        (* TODO: Allow C# or VB syntax for types, like in DefineDLLFunction. *)
        $classBuilder@DefineCallbackMethod[name, isStatic, isVirtual, isOverride, fixType[retType], fixType /@ paramTypes, mFunc, -1 (* ignored for now *)]
    ]

(* This version for pure functions. Must come after the def above. *)
NETMethod[name_String, retType_String:"System.Void", paramTypes:{___String}, mFunc_, opts___?OptionQ] :=
    NETMethod[name, retType, paramTypes, ToString[mFunc, InputForm], opts]


(*************************************  DefineNETMethod  ****************************************)

(**

TODO: Resolve how to supply "using" declarations. This is much more likely to be necessary here than when defining DLL functions.
Perhaps ImportNamespaces->{...}

**)

DefineNETMethod::compileerr = "The compiler reported errors in your method declaration: `1`"

Options[DefineNETMethod] = {ReferencedAssemblies -> {}, ImportNamespaces -> {}}

DefineNETMethod[code_String, opts___?OptionQ] :=
    Module[{result, newTypeName, funcName, argCount, argList, typeContext, refAss, ns},
        If[Head[InstallNET[]] =!= LinkObject,
            Message[DefineNETMethod::netlink, "DefineNETMethod"];
            Return[$Failed]
        ];
        LoadNETType["Wolfram.NETLink.Internal.DefineClassHelper", AllowShortContext->False, StaticsVisible->False];
        (* Definition to be based on full declaration provided by user. *)
        {ns, refAss} = {ImportNamespaces, ReferencedAssemblies} /. Flatten[{opts}] /. Options[DefineNETMethod];
        (* This just adds brackets if user left them off. *)
        {ns, refAss} = {Flatten[{ns}], Flatten[{refAss}]};
        If[!MatchQ[refAss, {___String}], refAss = {}];
        If[!MatchQ[ns, {___String}], ns = {}];
        AppendTo[refAss, "System.dll"];
        result = Wolfram`NETLink`Internal`DefineClassHelper`defineMethod[code, ns, refAss];
        If[!MatchQ[result, {_String, _String, _String}],
            (* There was a compiler error. result will be {"report"}. *)
            Message[DefineNETMethod::compileerr, First[result]];
            Return[$Failed]
        ];
        {newTypeName, funcName, argCount} = result;
        (* argCount comes back as a string. *)
        argCount = ToExpression[argCount];
        If[StringQ[newTypeName],
            LoadNETType[newTypeName, AllowShortContext->False, StaticsVisible->False];
            typeContext = StringReplace[newTypeName, "." -> "`"] <> "`";
            argList = ToString[Table["arg" <> ToString[i], {i, argCount}], FormatType->OutputForm];
            ToExpression["Function[" <> argList <> "," <>
                    typeContext <> funcName <> "[" <> StringDrop[StringDrop[argList, 1], -1] <> "], {HoldAll}]"],
        (* else *)
            (* Rely on message emitted by DefineMethod(). *)
            $Failed
        ]
    ]


(******************************  Callback Wrappers  *******************************)

methodCallbackWrapper[this_, func_String, args_List] :=
    Module[{result},
        Block[{$This = this}, result = ToExpression[func] @@ args];
        (* This is the value returned to .NET from the delegate callback: *)
        {argTypeToInteger[result], result}
    ]



End[]

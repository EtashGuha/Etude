(* :Title: Reflection.m *)

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


(*<!--Public From Reflection.m

NETTypeInfo::usage =
"NETTypeInfo[type] prints information about the specified type, including its inheritance hierarchy, assembly name, \
and its public members (constructors, methods, properties, and so on.) The type argument can be a fully-qualified type name \
given as a string, or a NETType expression. NETTypeInfo[obj] prints information about the object's type. NETTypeInfo[assembly] \
prints information about the types in the assembly specified by the given NETAssembly expression. NETTypeInfo[type, members] \
prints information about only the specified members, which can be any of the following strings (or a list of them): \
\"Type\", \"Constructors\", \"Methods\", \"Fields\", \"Properties\", or \"Events\". When calling NETTypeInfo on a NETAssembly \
expression, the members argument must be any of the following strings (or a list of them): \"Classes\", \"Interfaces\", \
\"Structures\", \"Delegates\", or \"Enums\". NETTypeInfo[type, members, pattern] prints only the members whose names \
match the given string pattern. For example, \"Set*\" shows all members with names that start with Set."

LanguageSyntax::usage =
"LanguageSyntax is an option to NETTypeInfo that specifies which language syntax will be used to display the type information. \
The possible values are the strings \"CSharp\" (or just \"C#\") and \"VisualBasic\" (or just \"VB\"). The default is C#."

-->*)

(*<!--Package From Reflection.m

-->*)


(* Current context will be NETLink`. *)

Begin["`Reflection`Private`"]


NETTypeInfo::null = "Object is null."
NETTypeInfo::lang = "Unrecognized value for LanguageSyntax option: `1`. Defaulting to C# syntax."
NETTypeInfo::members1 = "Invalid specification for member types to display: `1`. It should be All or a list of one or more of the following strings: \"Type\", \"Constructors\", \"Fields\", \"Properties\", \"Methods\", \"Events\"."
NETTypeInfo::members2 = "Invalid specification for member types to display: `1`. It should be All or a list of one or more of the following strings: \"Classes\", \"Interfaces\", \"Structures\", \"Delegates\", \"Enums\"."
(* TODO: improve this message (e.g., by saying that the type library has to be tlbimp'ed) when the importing of type libraries is either automated or better documented in .NET/Link. *)
NETTypeInfo::com = "Type information is not currently available for \"raw\" COM objects."
NETTypeInfo::arg1 = "A valid object reference, .NET type specification, or .NET assembly specification is required as the first argument."

Options[NETTypeInfo] = {Inherited -> True, LanguageSyntax -> "CSharp", IgnoreCase -> False}


(*****  Null as first arg  *****)

NETTypeInfo[Null, ___] := Message[NETTypeInfo::null]


(*****  Member type and pattern not specified  *****)

NETTypeInfo[x_, opts___?OptionQ] := NETTypeInfo[x, All, opts]


(*****  Pattern not specified  *****)

NETTypeInfo[x_, members:(All | {___String} | "Type" | "Constructors" | "Methods" | "Fields" | "Properties" | "Events" |
                            "Classes" | "Interfaces" | "Structures" | "Delegates" | "Enums"), opts___?OptionQ] :=
    NETTypeInfo[x, members, "*", opts]


(*****  Members not specified  *****)

NETTypeInfo[x_, pat_String, opts___?OptionQ] := NETTypeInfo[x, All, pat, opts]


(*****  First arg is anything but a string  *****)

NETTypeInfo[obj_?NETObjectQ, members_, pat_String, opts___?OptionQ] :=
    NETTypeInfo[getAQTypeName[obj], members, pat, opts]

NETTypeInfo[type_NETType, members_, pat_String, opts___?OptionQ] :=
    NETTypeInfo[getAQTypeName[type], members, pat, opts]


(*****  First arg is a string (all others are routed through here)  *****)

NETTypeInfo[typeName_String, members_, pat_String, opts___?OptionQ] := 
    Module[{type, inherited, lang, ignoreCase, useTypesetOutput, propNames, eventNames, membersList, typeInfo,
            ctors, fields, props, methods, events, typeRows, ctorRows, fieldRows, propRows, methodRows, eventRows},
        type = LoadNETType[typeName];
        If[Head[type] =!= NETType,
            Return[Null]
        ];
        {inherited, lang, ignoreCase} = contextIndependentOptions[{Inherited, LanguageSyntax, IgnoreCase}, {opts}, Options[NETTypeInfo]];
        Which[
            StringQ[lang] && MatchQ[ToUpperCase[lang], "VB" | "VISUALBASIC"],
                lang = $vb,
            StringQ[lang] && MatchQ[ToUpperCase[lang], "C#" | "CSHARP"],
                lang = $csharp,
            True,
                Message[NETTypeInfo::lang, lang];
                lang = $csharp                
        ];
        Switch[members,
            All,
                membersList = {"type", "constructors", "fields", "properties", "methods", "events"},
            _String,
                membersList = {ToLowerCase[members]},
            {___String},
                membersList = ToLowerCase /@ members,
            _,
                Message[NETTypeInfo::members1, members];
                membersList = {"type", "constructors", "fields", "properties", "methods", "events"}
        ];
        If[Intersection[membersList, {"type", "constructors", "fields", "properties", "methods", "events"}] =!= Sort[membersList],
            Message[NETTypeInfo::members1, Flatten[{members}]];
            membersList = {"type", "constructors", "fields", "properties", "methods", "events"}
        ];
        useTypesetOutput = (FormatType /. Options["stdout"]) =!= OutputForm && Head[$FrontEnd] === FrontEndObject;
        
        {typeInfo, ctors, fields, props, methods, events} = nReflectType[getAQTypeName[type]];

        (* Reject attempts to get type info for a "raw" COM object. This information currently will just be the base methods
           for MarshalByRefObject, which is almost undoubtedly not what the user wants to see. Perhaps in the future we
           will do the hard work of mining the COM type library for this information.
        *)
        If[typeInfo[[2]] == "System.__ComObject",
            Message[NETTypeInfo::com];
            Return[TableForm[{}]]
        ];
        
        (****************  General type info  ****************)
        (* typeInfo looks like this:
            {name, fullName, {parentNames...}, {intfs...}, isValueType, isEnum, isDelegate, isInterface, aqName, location}
        *)
        If[MemberQ[membersList, "type"],
            typeRows = makeTypeRows[typeInfo, lang, useTypesetOutput],
        (* else *)
            typeRows = {}
        ];
        
        (****************  Constructors  ****************)
        (* ctors looks like this (just a list of params):
            {{isOptional, defaultValue, isOut, isByRef, paramType, name}...}
        *)
        If[MemberQ[membersList, "constructors" && pat == "*"],
            (* typeInfo[[1]] is the short name of the class. *)
            ctorRows = makeCtorRow[typeInfo[[1]], #, lang, useTypesetOutput]& /@ ctors,
        (* else *)
            ctorRows = {}
        ];
        
        (****************  Fields  ****************)
        (* fields looks like this:
            {{isInherited, isStatic, isLiteral, isInitOnly, fieldType, name}...}
        *)
        If[MemberQ[membersList, "fields"],
            If[!TrueQ[inherited],
                fields = DeleteCases[fields, {True, __}]
            ];
            fields = Select[fields, StringMatchQ[#[[6]], pat, IgnoreCase -> ignoreCase]&];
            fields = Sort[fields, OrderedQ[{#1[[6]], #2[[6]]}]&];
            fieldRows = makeFieldRow[#, lang, useTypesetOutput]& /@ fields,
        (* else *)
            fieldRows = {}
        ];
        
        (****************  Properties  ****************)
        (* props looks like this:
            {{isInherited, isStatic, isVirtual, isOverride, isAbstract, canRead, canWrite, propType, name, {{isOptional, defaultValue, isOut, isByRef, paramType, name}...}}...}
        *)
        propNames = #[[9]]& /@ props;
        If[MemberQ[membersList, "properties"],
            If[!TrueQ[inherited],
                props = DeleteCases[props, {True, __}]
            ];
            props = Sort[props, OrderedQ[{#1[[9]], #2[[9]]}]&];
            props = Select[props, StringMatchQ[#[[9]], pat, IgnoreCase -> ignoreCase]&];
            propRows = makePropRow[#, lang, useTypesetOutput]& /@ props,
        (* else *)
            propRows = {}
        ];
        
        (****************  Events  ****************)
        (* events looks like this:
            {{isInherited, isStatic, isVirtual, isOverride, isAbstract, delegateType, name, dlgRetType, {dlgParams}}...}
        *)
        eventNames = #[[7]]& /@ events;
        If[MemberQ[membersList, "events"],
            If[!TrueQ[inherited],
                events = DeleteCases[events, {True, __}]
            ];
            events = Sort[events, OrderedQ[{#1[[7]], #2[[7]]}]&];
            events = Select[events, StringMatchQ[#[[7]], pat, IgnoreCase -> ignoreCase]&];
            eventRows = makeEventRow[#, lang, useTypesetOutput]& /@ events,
        (* else *)
            eventRows = {}
        ];
        
        (****************  Methods  ****************)
        (* methods looks like this:
            {{isInherited, isStatic, isVirtual, isOverride, isAbstract, retType, name, {{isOptional, defaultValue, isOut, isByRef, paramType, name}...}}...}
        *)
        If[MemberQ[membersList, "methods"],
            If[!TrueQ[inherited],
                methods = DeleteCases[methods, {True, __}]
            ];
            methods = Sort[methods, OrderedQ[{#1[[7]], #2[[7]]}]&];
            (* Remove the set_XXX and get_XXX methods that are for properties XXX. *) 
            methods = DeleteCases[methods, methRec_ /; (StringMatchQ[methRec[[7]], "get_*"] || StringMatchQ[methRec[[7]], "set_*"]) &&
                                    MemberQ[propNames, StringDrop[methRec[[7]], 4]]];
            (* Remove the add_XXX and remove_XXX methods that are for events XXX. *) 
            methods = DeleteCases[methods, methRec_ /; (StringMatchQ[methRec[[7]], "add_*"] && MemberQ[eventNames, StringDrop[methRec[[7]], 4]] ||
                                                        StringMatchQ[methRec[[7]], "remove_*"] && MemberQ[eventNames, StringDrop[methRec[[7]], 7]])];                                   
            methods = Select[methods, StringMatchQ[#[[7]], pat, IgnoreCase -> ignoreCase]&];
            methodRows = makeMethodRow[#, lang, useTypesetOutput]& /@ methods,
        (* else *)
            methodRows = {}
        ];
        
        formatOutput[{typeRows, ctorRows, fieldRows, propRows, methodRows, eventRows}, useTypesetOutput, pat]
    ]
    

(************  This version for Assembly info  **************)

NETTypeInfo[asm_NETAssembly, classTypes_, pat_String, opts___?OptionQ] := 
    Module[{asmInfo, rows, lang, ignoreCase, useTypesetOutput, classTypesList, asmName, asmFullName, asmLoc,
                types, classes, interfaces, structs, delegates, enums, members, memberType, fullName, ns},
        {lang, ignoreCase} = contextIndependentOptions[{LanguageSyntax, IgnoreCase}, {opts}, Options[NETTypeInfo]];
        InstallNET[];
        Which[
            StringQ[lang] && MatchQ[ToUpperCase[lang], "VB" | "VISUALBASIC"],
                lang = $vb,
            StringQ[lang] && MatchQ[ToUpperCase[lang], "C#" | "CSHARP"],
                lang = $csharp,
            True,
                Message[NETTypeInfo::lang, lang];
                lang = $csharp                
        ];
        Switch[classTypes,
            All,
                classTypesList = {"classes", "interfaces", "structures", "delegates", "enums"},
            _String,
                classTypesList = {ToLowerCase[classTypes]},
            {___String},
                classTypesList = ToLowerCase /@ classTypes,
            _,
                Message[NETTypeInfo::members2, classTypes];
                classTypesList = {"classes", "interfaces", "structures", "delegates", "enums"}
        ];
        If[Intersection[classTypesList, {"classes", "interfaces", "structures", "delegates", "enums"}] =!= Sort[classTypesList],
            Message[NETTypeInfo::members2, Flatten[{classTypes}]];
            classTypesList = {"classes", "interfaces", "structures", "delegates", "enums"}
        ];
        useTypesetOutput = (FormatType /. Options["stdout"]) =!= OutputForm && Head[$FrontEnd] === FrontEndObject;
        
        asmInfo = nReflectAsm[getFullAsmName[asm]];
        asmName = asmInfo[[1]];
        asmFullName = asmInfo[[2]];
        asmLoc = asmInfo[[3]];
        types = Drop[asmInfo, 3];
        
        rows = {};
        
        If[useTypesetOutput,
            AppendTo[rows, RowBox[{"Assembly: ", StyleBox[asmName, FontWeight->"Bold"]}]],
        (* else *)
            AppendTo[rows, "Assembly: " <> asmName]
        ];
        AppendTo[rows, "Full Name: " <> asmFullName];
        AppendTo[rows, "Location: " <> asmLoc];
        If[useTypesetOutput,
            (* To improve formatting of type info strings that have . or \ in them, force them to be treated as
               raw strings instead of broken into RowBoxes. The StringReplace here doubles up backslashes if they
               are present (e.g., in the assembly location path). Don't know why that is needed, but it is.
            *)
            rows = rows /. s_String :> ("\<\"" <> StringReplace[s, "\\" -> "\\\\"] <> "\"\>") /;
                            !StringMatchQ[s, "*\"*"] && (StringMatchQ[s, "*.*"] || StringMatchQ[s, "*\\*"])
        ];
        
        (* types looks like this:
            {{fullName, namespace, isValueType, isEnum, isDelegate, isInterface}...}
        *)
        
        (****************  Classes  ****************)
        
        If[MemberQ[classTypesList, "classes"],
            members = Cases[types, {fullName_, ns_, False, False, False, False} -> {ns, fullName}];
            memberType = If[lang === $vb, "Class ", "class "];
            rows = Join[rows, makeAssemblyMemberRows[members, memberType, pat, "Classes", useTypesetOutput, ignoreCase]]
        ];
        
        (****************  Interfaces  ****************)
        
        If[MemberQ[classTypesList, "interfaces"],
            members = Cases[types, {fullName_, ns_, _, _, _, True} -> {ns, fullName}];
            memberType = If[lang === $vb, "Interface ", "interface "];
            rows = Join[rows, makeAssemblyMemberRows[members, memberType, pat, "Interfaces", useTypesetOutput, ignoreCase]]
        ];

        (****************  Structures  ****************)
        
        If[MemberQ[classTypesList, "structures"],
            members = Cases[types, {fullName_, ns_, True, _, _, _} -> {ns, fullName}];
            memberType = If[lang === $vb, "Structure ", "struct "];
            rows = Join[rows, makeAssemblyMemberRows[members, memberType, pat, "Structures", useTypesetOutput, ignoreCase]]
        ];

        (****************  Delegates  ****************)
        
        If[MemberQ[classTypesList, "delegates"],
            members = Cases[types, {fullName_, ns_, _, _, True, _} -> {ns, fullName}];
            memberType = If[lang === $vb, "Delegate ", "delegate "];
            rows = Join[rows, makeAssemblyMemberRows[members, memberType, pat, "Delegates", useTypesetOutput, ignoreCase]]
        ];

        (****************  Enums  ****************)
        
        If[MemberQ[classTypesList, "enums"],
            members = Cases[types, {fullName_, ns_, _, True, _, _} -> {ns, fullName}];
            memberType = If[lang === $vb, "Enum ", "enum "];
            rows = Join[rows, makeAssemblyMemberRows[members, memberType, pat, "Enums", useTypesetOutput, ignoreCase]]
        ];

        rows = List /@ rows;
        
        If[useTypesetOutput,
            CellPrint[Cell[BoxData[GridBox[rows, ColumnAlignments->Left, RowMinHeight->1.2]],
                            "Output", FormatType->StandardForm, AutoSpacing->False]],
        (* else *)
            TableForm[rows]
        ]
    ]
    

(* Must come last. *)
NETTypeInfo[bad_, members_, pat_String, opts___?OptionQ] :=
    Message[NETTypeInfo::arg1]

NETTypeInfo[bad_, members_, opts___?OptionQ] :=
    Message[NETTypeInfo::arg1]


(************************************  Implementation  **************************************)

makeTypeRows[typeInfo_List, lang_Symbol, useTypesetOutput_] :=
    Module[{name, fullName, parentNames, intfNames, isValueType, isEnum, isDelegate, isInterface,
            inheritanceRow, intfRow, aqName, location, classType, rows},
        {name, fullName, parentNames, intfNames, isValueType, isEnum, isDelegate, isInterface, aqName, location} = typeInfo;
        (* If the location comes back as "", this means that the assembly was generated in memory. *)
        If[location == "", location = "Dynamically generated"];
        Which[
            isEnum,
                classType = If[lang === $vb, "Enum", "enum "],
            isValueType,
                classType = If[lang === $vb, "Structure ", "struct "],
            isDelegate,
                classType = If[lang === $vb, "Delegate ", "delegate "],
            isInterface,
                classType = If[lang === $vb, "Interface ", "interface "],
            True,
                classType = If[lang === $vb, "Class ", "class "]
        ];
        If[useTypesetOutput,
            rows = {RowBox[{classType, StyleBox[fullName, FontWeight->"Bold"]}]};
            If[parentNames =!= {},
                (* I think parentNames is only empty for an interface. *)
                inheritanceRow = GridBox[{{"Inheritance:"}}, ColumnAlignments->{Left}];
                MapIndexed[(inheritanceRow = Insert[inheritanceRow, {RowBox[Join[{"   "}, Table["   ", #2], {#1}]]}, {1, -1}])&, Reverse[parentNames]];
                inheritanceRow = Insert[inheritanceRow, {RowBox[Table["   ", {Length[parentNames] + 2}] ~Join~ {StyleBox[fullName, FontWeight->"Bold"]}]}, {1, -1}];
                AppendTo[rows, inheritanceRow];
            ],
        (* else *)
            rows = {classType <> fullName};
            If[parentNames =!= {},
                (* I think parentNames is only empty for an interface. *)
                inheritanceRow = {"Inheritance:"};
                MapIndexed[AppendTo[inheritanceRow, "   " <> (StringJoin @@ Table["   ", #2]) <> #1 ]&, Reverse[parentNames]];
                AppendTo[inheritanceRow, (StringJoin @@ Table["   ", {Length[parentNames] + 2}]) <> fullName];
                rows = Join[rows, inheritanceRow]
            ]
        ];
        If[intfNames === {},
            intfRow = "Interfaces Implemented: None",
        (* else *)
            intfRow = "Interfaces Implemented: ";
            (intfRow = intfRow <> convertTypeName[#, lang, useTypesetOutput] <> ", ")& /@ intfNames;
            (* Drop last ", ". *)
            intfRow = StringDrop[intfRow, -2]
        ];
        AppendTo[rows, intfRow];
        AppendTo[rows, "Assembly-Qualified Name: " <> aqName];
        AppendTo[rows, "Assembly Location: " <> location];
        If[useTypesetOutput,
            (* To improve formatting of type info strings that have . or \ in them, force them to be treated as
               raw strings instead of broken into RowBoxes. The StringReplace here doubles up backslashes if they
               are present (e.g., in the assembly location path). Don't know why that is needed, but it is.
            *)
            rows = rows /. s_String :> ("\<\"" <> StringReplace[s, "\\" -> "\\\\"] <> "\"\>") /;
                            !StringMatchQ[s, "*\"*"] && (StringMatchQ[s, "*.*"] || StringMatchQ[s, "*\\*"])
        ];
        rows
    ]


makeCtorRow[className_String, params_List, lang_Symbol, useTypesetOutput_] :=
    Module[{row},
        If[useTypesetOutput,
            row = RowBox[{StyleBox[className, FontWeight->"Bold"], makeParams[params, lang, useTypesetOutput]}],
        (* else *)
            row = className <> makeParams[params, lang, useTypesetOutput]
        ];
        row
    ]
    
    
makeFieldRow[fieldRec_List, lang_Symbol, useTypesetOutput_] :=
    Module[{isInherited, isStatic, isLiteral, isInitOnly, fieldType, name, row},
        {isInherited, isStatic, isLiteral, isInitOnly, fieldType, name} = fieldRec;
        Switch[lang,
            $vb,
                If[useTypesetOutput,
                    row = RowBox[{}];
                    Which[
                        (* Const is always Shared, and "Shared" is left off in declarations. *)
                        isLiteral, row = Insert[row, {"Const", " "}, {1, -1}],
                        isStatic, row = Insert[row, {"Shared", " "}, {1, -1}]
                    ];
                    If[isInitOnly, row = Insert[row, {"ReadOnly", " "}, {1, -1}]];
                    row = Insert[row, {StyleBox[name, FontWeight->"Bold"], " ", "As", " "}, {1, -1}];
                    row = Insert[row, {convertTypeName[fieldType, lang, useTypesetOutput]}, {1, -1}];
                    row = MapAt[Flatten, row, {1}],
                (* else *)
                    row = 
                        Which[
                            (* Const is always Shared, and "Shared" is left off in declarations. *)
                            isLiteral, "Const ",
                            isStatic, "Shared ",
                            True, ""
                        ];
                    If[isInitOnly, row = row <> "ReadOnly "];
                    row = row <> name <> " As ";
                    row = row <> convertTypeName[fieldType, lang, useTypesetOutput]
                ],
            $csharp,                
                If[useTypesetOutput,
                    row = RowBox[{}];
                    Which[
                        (* Const is always static, and "static" is left off in declarations. *)
                        isLiteral, row = Insert[row, {"const", " "}, {1, -1}],
                        isStatic, row = Insert[row, {"static", " "}, {1, -1}]
                    ];
                    If[isInitOnly, row = Insert[row, {"readonly", " "}, {1, -1}]];
                    row = Insert[row, {convertTypeName[fieldType, lang, useTypesetOutput], " "}, {1, -1}];
                    row = Insert[row, StyleBox[name, FontWeight->"Bold"], {1, -1}];
                    row = MapAt[Flatten, row, {1}],
                (* else *)
                    row = 
                        Which[
                            (* Const is always static, and "static" is left off in declarations. *)
                            isLiteral, "const ",
                            isStatic, "static ",
                            True, ""
                        ];
                    If[isInitOnly, row = row <> "readonly "];
                    row = row <> convertTypeName[fieldType, lang, useTypesetOutput] <> " ";
                    row = row <> name
                ]
        ];
        row
    ]
    

makePropRow[propRec_List, lang_Symbol, useTypesetOutput_] :=
    Module[{isInherited, isStatic, isVirtual, isOverride, isAbstract, canRead, canWrite, propType, name, params, row},
        {isInherited, isStatic, isVirtual, isOverride, isAbstract, canRead, canWrite, propType, name, params} = propRec;
        (* params looks like:
            {{isOptional, defaultValue, isOut, isByRef, paramType, name}...}
        *)
        Switch[lang,
            $vb,
                If[useTypesetOutput,
                    row = RowBox[{}];
                    If[isStatic, row = Insert[row, {"Shared", " "}, {1, -1}]];
                    Which[
                        !canRead,
                            row = Insert[row, {"WriteOnly", " "}, {1, -1}],                       
                        !canWrite,
                            row = Insert[row, {"ReadOnly", " "}, {1, -1}]         
                    ];  
                    Which[
                        isAbstract,
                            row = Insert[row, {"MustOverride", " "}, {1, -1}],
                        isOverride,
                            row = Insert[row, {"Overrides", " "}, {1, -1}],
                        isVirtual,
                            row = Insert[row, {"Overridable", " "}, {1, -1}]
                    ];
                    row = Insert[row, {"Property", " ", StyleBox[name, FontWeight->"Bold"]}, {1, -1}];
                    If[params =!= {},
                        row = Insert[row, makeParams[params, lang, useTypesetOutput], {1, -1}]
                    ];
                    row = Insert[row, {" ", "As", " ", convertTypeName[propType, lang, useTypesetOutput]}, {1, -1}];
                    row = MapAt[Flatten, row, {1}],
                (* else *)
                    row = If[isStatic, "Shared ", ""];
                    Which[
                        !canRead,
                            row = row <> "WriteOnly ",                       
                        !canWrite,
                            row = row <> "ReadOnly "         
                    ];  
                    Which[
                        isAbstract,
                            row = row <> "MustOverride ",
                        isOverride,
                            row = row <> "Overrides ",
                        isVirtual,
                            row = row <> "Overridable "
                    ];
                    row = row <> "Property " <> name;
                    If[params =!= {},
                        row = row <> makeParams[params, lang, useTypesetOutput]
                    ];
                    row = row <> " As " <> convertTypeName[propType, lang, useTypesetOutput]
                 ],
            $csharp,
                If[useTypesetOutput,
                    row = RowBox[{}];
                    If[isStatic, row = Insert[row, {"static", " "}, {1, -1}]];
                    Which[
                        isAbstract,
                            row = Insert[row, {"abstract", " "}, {1, -1}],
                        isOverride,
                            row = Insert[row, {"override", " "}, {1, -1}],
                        isVirtual,
                            row = Insert[row, {"virtual", " "}, {1, -1}]
                    ];
                    row = Insert[row, {convertTypeName[propType, lang, useTypesetOutput], " "}, {1, -1}];
                    row = Insert[row, StyleBox[name, FontWeight->"Bold"], {1, -1}];
                    If[params =!= {},
                        row = Insert[row, makeParams[params, lang, useTypesetOutput], {1, -1}]
                    ];
                    Which[
                        !canRead,
                            row = Insert[row, "  [write only]", {1, -1}],                       
                        !canWrite,
                            row = Insert[row, "  [read only]", {1, -1}]         
                    ];  
                    row = MapAt[Flatten, row, {1}],
                (* else *)
                    row = If[isStatic, "static ", ""];
                    Which[
                        isAbstract,
                            row = row <> "abstract ",
                        isOverride,
                            row = row <> "override ",
                        isVirtual,
                            row = row <> "virtual "
                    ];
                    row = row <> convertTypeName[propType, lang, useTypesetOutput] <> " ";
                    row = row <> name;
                    If[params =!= {},
                        row = row <> makeParams[params, lang, useTypesetOutput]
                    ];
                    Which[
                        !canRead,
                            row = row <> "  [write only]",                        
                        !canWrite,
                            row = row <> "  [read only]"
                    ]                   
                ]
        ];
        row
    ]
    
    
makeMethodRow[methodRec_List, lang_Symbol, useTypesetOutput_] :=
    Module[{isInherited, isStatic, isVirtual, isOverride, isAbstract, retType, name, params, row, funcType},
        {isInherited, isStatic, isVirtual, isOverride, isAbstract, retType, name, params} = methodRec;
        (* params looks like:
            {{isOptional, defaultValue, isOut, isByRef, paramType, name}...}
        *)
        Switch[lang,
            $vb,
                funcType = If[retType == "System.Void", "Sub", "Function"];
                If[useTypesetOutput,
                    row = RowBox[{}];
                    If[isStatic, row = Insert[row, {"Shared", " "}, {1, -1}]];
                    Which[
                        isAbstract,
                            row = Insert[row, {"MustOverride", " "}, {1, -1}],
                        isOverride,
                            row = Insert[row, {"Overrides", " "}, {1, -1}],
                        isVirtual,
                            row = Insert[row, {"Overridable", " "}, {1, -1}]
                    ];
                    row = Insert[row, {funcType, " ", StyleBox[name, FontWeight->"Bold"]}, {1, -1}];
                    row = Insert[row, makeParams[params, lang, useTypesetOutput], {1, -1}];
                    If[funcType == "Function",
                        row = Insert[row, {" ", "As", " ", convertTypeName[retType, lang, useTypesetOutput]}, {1, -1}]
                    ];
                    row = MapAt[Flatten, row, {1}],
                (* else *)
                    row = If[isStatic, "Shared ", ""];
                    Which[
                        isAbstract,
                            row = row <> "MustOverride ",
                        isOverride,
                            row = row <> "Overrides ",
                        isVirtual,
                            row = row <> "Overridable "
                    ];
                    row = row <> funcType <> name;
                    row = row <> makeParams[params, lang, useTypesetOutput];
                    If[funcType == "Function",
                        row = row <> " As " <> convertTypeName[retType, lang, useTypesetOutput]
                    ]
                ],
            $csharp,                
                If[useTypesetOutput,
                    row = RowBox[{}];
                    If[isStatic, row = Insert[row, {"static", " "}, {1, -1}]];
                    Which[
                        isAbstract,
                            row = Insert[row, {"abstract", " "}, {1, -1}],
                        isOverride,
                            row = Insert[row, {"override", " "}, {1, -1}],
                        isVirtual,
                            row = Insert[row, {"virtual", " "}, {1, -1}]
                    ];
                    row = Insert[row, {convertTypeName[retType, lang, useTypesetOutput], " "}, {1, -1}];
                    row = Insert[row, StyleBox[name, FontWeight->"Bold"], {1, -1}];
                    row = Insert[row, makeParams[params, lang, useTypesetOutput], {1, -1}];
                    row = MapAt[Flatten, row, {1}],
                (* else *)
                    row = If[isStatic, "static ", ""];
                    Which[
                        isAbstract,
                            row = row <> "abstract ",
                        isOverride,
                            row = row <> "override ",
                        isVirtual,
                            row = row <> "virtual "
                    ];
                    row = row <> convertTypeName[retType, lang, useTypesetOutput] <> " ";
                    row = row <> name;
                    row = row <> makeParams[params, lang, useTypesetOutput];
                ]
        ];
        row
    ]
    

makeEventRow[eventRec_List, lang_Symbol, useTypesetOutput_] :=
    Module[{isInherited, isStatic, isVirtual, isOverride, isAbstract, dlgType, name, dlgRetType, params, row},
        {isInherited, isStatic, isVirtual, isOverride, isAbstract, dlgType, name, dlgRetType, params} = eventRec;
        Switch[lang,
            $vb,        
                If[useTypesetOutput,
                    row = RowBox[{}];
                    If[isStatic, row = Insert[row, {"Shared", " "}, {1, -1}]];
                    Which[
                        isAbstract,
                            row = Insert[row, {"MustOverride", " "}, {1, -1}],
                        isOverride,
                            row = Insert[row, {"Overrides", " "}, {1, -1}],
                        isVirtual,
                            row = Insert[row, {"Overridable", " "}, {1, -1}]
                    ];
                    row = Insert[row, {"Event", " ", StyleBox[name, FontWeight->"Bold"], " ", "As", " ",
                                        convertTypeName[dlgType, lang, useTypesetOutput]}, {1, -1}];
                    row = Insert[row, {"\<\"  [arguments to delegate: \"\>", makeParams[params, lang, useTypesetOutput], "\<\"]\"\>"}, {1, -1}];
                    row = MapAt[Flatten, row, {1}],
                (* else *)
                    row = If[isStatic, "Shared ", ""];
                    Which[
                        isAbstract,
                            row = row <> "MustOverride ",
                        isOverride,
                            row = row <> "Overrides ",
                        isVirtual,
                            row = row <> "Overridable "
                    ];
                    row = row <> "Event " <> name <> " As " <> convertTypeName[dlgType, lang, useTypesetOutput] <>
                                    "  [signature of delegate: " <> makeParams[params, lang, useTypesetOutput] <> "]"  
                ],
            $csharp,                
                If[useTypesetOutput,
                    row = RowBox[{}];
                    If[isStatic, row = Insert[row, {"static", " "}, {1, -1}]];
                    Which[
                        isAbstract,
                            row = Insert[row, {"abstract", " "}, {1, -1}],
                        isOverride,
                            row = Insert[row, {"override", " "}, {1, -1}],
                        isVirtual,
                            row = Insert[row, {"virtual", " "}, {1, -1}]
                    ];
                    row = Insert[row, {"event", " ", convertTypeName[dlgType, lang, useTypesetOutput], " ", StyleBox[name, FontWeight->"Bold"]}, {1, -1}];
                    (* Wrap the sig part in \<\" to force it to be treated as a raw string. Otherwise the [ ] get printed funny. *)
                    row = Insert[row, {"\<\"  [arguments to delegate: \"\>", makeParams[params, lang, useTypesetOutput], "\<\"]\"\>"}, {1, -1}];
                    row = MapAt[Flatten, row, {1}],
                (* else *)
                    row = If[isStatic, "static ", ""];
                    Which[
                        isAbstract,
                            row = row <> "abstract ",
                        isOverride,
                            row = row <> "override ",
                        isVirtual,
                            row = row <> "virtual "
                    ];
                    row = row <> "event " <> convertTypeName[dlgType, lang, useTypesetOutput] <> " " <> name <>
                                "  [signature of delegate: " <> makeParams[params, lang, useTypesetOutput] <> "]"
                ]
        ];
        row
    ]
    

makeParams[params_List, lang_, useTypesetOutput_] :=
    Module[{str, paramtype},
        (* Note that because we must call convertTypeName with its useTypesetOutput argument set to False to keep it from
           wrapping the type strings in "\<\"". We will do that at the end on the final string if we are producing typeset output.
        *)
        str = "(";
        If[Length[params] > 0,
            Function[{isOptional, defaultValue, isOut, isByRef, type, name},
                paramType = type;
                Switch[lang,
                    $vb,
                        If[isOptional,
                            str = str <> "Optional "
                        ];
                        If[isByRef,
                            (* Because we will use "ByRef" to indicate byref, drop the trailing & in the type name. I test
                                for its presence first, but I think it must always be there.
                            *)
                            If[StringTake[paramType, -1] == "&", paramType = StringDrop[paramType, -1]];
                            str = str <> "ByRef "
                        ];
                        str = str <> name <> " As " <> convertTypeName[paramType, lang, False] <>
                                    If[defaultValue =!= Default, " = " <> defaultValue, ""] <> ", ",
                    $csharp,
                        If[isOptional,
                            str = str <> "[optional" <> If[defaultValue =!= Default, ", default = " <> defaultValue, ""] <> "] "
                        ];
                        If[isByRef,
                            (* Because we will use "out" or "ref" to indicate byref, drop the trailing & in the type name. I test
                                for its presence first, but I think it must always be there.
                            *)
                            If[StringTake[paramType, -1] == "&", paramType = StringDrop[paramType, -1]];
                            str = str <> If[isOut, "out ", "ref "]
                        ];
                        str = str <> convertTypeName[paramType, lang, False] <> " " <> name <> ", ";
                ]
            ] @@@ params;
            (* Drop the last ", ". *)
            str = StringDrop[str, -2]
        ];
        str = str <> ")";
        If[useTypesetOutput,
            str = "\<\"" <> str <> "\"\>"
        ];
        str
    ]
    

convertTypeName[type_, lang_, useTypesetOutput_] :=
    Block[{isArray, isPointer, baseType, firstBracketPos, result},  (* Block for speed only. *)
        isArray = StringMatchQ[type, "*[]"] || StringMatchQ[type, "*[,*"];
        isPointer = StringMatchQ[type, "*\\*"];
        Which[
            isArray,
                firstBracketPos = First[Flatten[StringPosition[type, "["]]];
                baseType = StringTake[type, firstBracketPos - 1],
            isPointer,
                baseType = StringDrop[type, -1],
            True,
                baseType = type
        ];
        Switch[lang,
            $vb,
                result = baseType /. $netTypeToVBRules;
                Which[
                    isArray,
                        result = StringReplace[result <> StringDrop[type, firstBracketPos - 1], {"[" -> "(", "]" -> ")"}],
                    isPointer,
                        (* VB has no syntax for pointers, so convert all ptrs to IntPtr. *)
                        result = "IntPtr"
                ],
            $csharp,
                result = baseType /. $netTypeToCSharpRules;
                Which[
                    isArray,
                        result = result <> StringDrop[type, firstBracketPos - 1],
                    isPointer,
                        result = result <> "*"
                ]
        ];
        (* If the type is of the form System.XXX, where there are no further namespace divisions in XXX,
           strip off the XXX (e.g., System.Type ---> Type).
        *)
        If[StringMatchQ[result, "System.*"] && StringPosition[result, "."] === {{7,7}},
            result = StringDrop[result, 7]
        ];
        If[useTypesetOutput,
            result = "\<\"" <> result <> "\"\>"
        ];
        result
    ]


(* The arg should always be a full 5-part list, in the correct order (ctors, fields, props, meths, events}. If any elements
   have no members or were not included in the search, they will be represented as an empty list.
*)
formatOutput[{typeInfo_List, ctors_List, fields_List, props_List, methods_List, events_List}, useTypesetOutput_, pat_String] :=
    Module[{result, suffix},
        If[pat != "*",
            suffix = " (matching string pattern " <> pat <> ")",
        (* else *)
            suffix = ""
        ];
        If[useTypesetOutput,
            result = List /@ Join[
                        If[typeInfo =!= {}, {makeHeading["Type", "", useTypesetOutput]}, {}],
                        typeInfo,
                        If[ctors =!= {}, {"", makeHeading["Constructors", "", useTypesetOutput]}, {}],
                        ctors,
                        If[fields =!= {}, {"", makeHeading["Fields", suffix, useTypesetOutput]}, {}],
                        fields,
                        If[props =!= {}, {"", makeHeading["Properties", suffix, useTypesetOutput]}, {}],
                        props,
                        If[methods =!= {}, {"", makeHeading["Methods", suffix, useTypesetOutput]}, {}],
                        methods,
                        If[events =!= {}, {"", makeHeading["Events", suffix, useTypesetOutput]}, {}],
                        events
                    ];
            If[result =!= {},
                CellPrint[Cell[BoxData[GridBox[result, ColumnAlignments->Left, RowMinHeight->1.2]],
                                "Output", FormatType->StandardForm, AutoSpacing->False]]
            ],
        (* else *)
            result = Join[
                        If[typeInfo =!= {}, {makeHeading["Type", "", useTypesetOutput]}, {}],
                        typeInfo,
                        If[ctors =!= {}, {"", makeHeading["Constructors", "", useTypesetOutput]}, {}],
                        ctors,
                        If[fields =!= {}, {"", makeHeading["Fields", suffix, useTypesetOutput]}, {}],
                        fields,
                        If[props =!= {}, {"", makeHeading["Properties", suffix, useTypesetOutput]}, {}],
                        props,
                        If[methods =!= {}, {"", makeHeading["Methods", suffix, useTypesetOutput]}, {}],
                        methods,
                        If[events =!= {}, {"", makeHeading["Events", suffix, useTypesetOutput]}, {}],
                        events
                    ];
            TableForm[result]
        ]
    ]


makeHeading[label_String, sfx_String, useTypesetOutput_] :=
    Module[{suffix = sfx},
        If[suffix != "" && useTypesetOutput && !StringMatchQ[suffix, "*\"*"],
            suffix = "\<\"" <> suffix <> "\"\>"
        ];
        If[useTypesetOutput,
            Cell[{RowBox[{"\[FilledCircle] ", StyleBox[label, FontWeight->"Bold", FontSlant->"Italic", FontSize->18], suffix}]}],
        (* else *)
            "---- " <> label <> " ----" <> suffix
        ]
    ]
    

makeAssemblyMemberRows[members_List, classType_String, pat_String, typeHeader_String, useTypesetOutput_, ignoreCase_] :=
    Module[{mems = members, rows = {}, nsPrefixLength},
        mems =
            If[StringMatchQ[pat, "*.*"],
                (* If pattern contains a period, match against full name with namespace. If not, only
                    match against type name.
                *)
                Select[members, StringMatchQ[#[[2]], pat, IgnoreCase -> ignoreCase]&],
            (* else *)
                Select[members, StringMatchQ[StringDrop[#[[2]], If[#[[1]] == "", 0, StringLength[#[[1]]] + 1]], pat, IgnoreCase -> ignoreCase]&]
            ] // Sort;
        If[Length[mems] > 0,
            AppendTo[rows, ""];  (* Blank line. *)
            AppendTo[rows, makeHeading[typeHeader, "", useTypesetOutput]];
            AppendTo[rows,
                nsPrefixLength = If[#1 == "", 0, StringLength[#1] + 1];
                If[useTypesetOutput,
                    RowBox[{classType, If[#1 == "", "", "\<\"" <> #1 <> ".\"\>"], StyleBox[StringDrop[#2, nsPrefixLength], FontWeight->"Bold"]}],
                (* else *)
                    classType <> #1 <> If[#1 == "", "", "."] <> StringDrop[#2, nsPrefixLength]
                ]
            ]& @@@ mems
        ];
        rows
    ]


End[]

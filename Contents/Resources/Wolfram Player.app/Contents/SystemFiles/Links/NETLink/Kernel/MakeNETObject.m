(* :Title: MakeNETObject.m *)

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


(*<!--Public From MakeNETObject.m

MakeNETObject::usage =
"MakeNETObject[expr] constructs a .NET object that represents the given Wolfram Language value. It operates on numbers, \
True/False, strings, and lists of these, up to 3-deep. It is a shortcut to calling NETNew, and it is especially useful \
for arrays, as the array object can be created and initialized with values in a single call. MakeNETObject[expr, type] \
creates an object of the specified type from expr. Use the type argument to force a non-standard type. For example, \
MakeNETObject[{1,2,3}] will create an array of int (type System.Int32[]). If you want an array of Int16, you would use \
MakeNETObject[{1,2,3}, \"System.Int16[]\"]."

-->*)


(*<!--Package From MakeNETObject.m

-->*)


(* Current context will be NETLink`. *)

Begin["`MakeNETObject`Private`"]


MakeNETObject::arg =
"MakeNETObject cannot convert `1` to a .NET object. It does not operate on arguments of that type."

MakeNETObject::type =
"Type `1` is not recognized or cannot be created from argument `2`."

MakeNETObject::arraytype =
"Type `1` is not recognized or cannot be created from this argument."

MakeNETObject::complex =
"You cannot call MakeNETObject on a complex number until you have set the .NET type to use for representing complex numbers by calling SetComplexType."

MakeNETObject::cmplxmatch =
"The type specified does not match the type assigned to represent complex numbers using SetComplexType[]. You can leave out a type specification when calling MakeNETObject on a complex number, as the type assigned via SetComplexType[] is always used."

MakeNETObject::array =
"When sending a list, you must specify an array type using [] notation, for example Int32[]."

MakeNETObject::array2 =
"When sending a matrix, you must specify a two-dimensional array type using [,] notation, for example Int32[,]."

MakeNETObject::array3 =
"When sending a 3-deep array, you must specify a three-dimensional array type using [,,] notation, for example Int32[,,]."

MakeNETObject::array4 =
"MakeNETObject does not operate on arrays with more than three dimensions."

MakeNETObject::empty =
"To create an empty array you need to supply a second argument that gives type information, for example \"Int32[]\"."


(************************  Definitions for Expr  **************************)

(* Anything can go as an Expr. *)

MakeNETObject[e_, "Wolfram.NETLink.Expr"] :=
    (
        InstallNET[];
        nMakeObject["Wolfram.NETLink.Expr", argTypeToInteger[e], e]
    )

MakeNETObject[e_List, "Wolfram.NETLink.Expr[]"] :=
    (
        InstallNET[];
        nMakeObject["Wolfram.NETLink.Expr[]", argTypeToInteger[e], e]
    )

MakeNETObject[e:{__List}, "Wolfram.NETLink.Expr[,]"] :=
    (
        InstallNET[];
        nMakeObject["Wolfram.NETLink.Expr[,]", argTypeToInteger[e], e]
    )

MakeNETObject[e:{__List}, "Wolfram.NETLink.Expr[][]"] :=
    (
        InstallNET[];
        nMakeObject["Wolfram.NETLink.Expr[][]", argTypeToInteger[e], e]
    )


(*****************  Definitions for atomic expressions  *******************)

MakeNETObject[obj_?NETObjectQ] := obj
MakeNETObject[obj_?NETObjectQ, _] := obj

MakeNETObject[e_Integer] := MakeNETObject[e, "System.Int32"]
MakeNETObject[e_Real] := MakeNETObject[e, "System.Double"]
MakeNETObject[e:(True | False)] := MakeNETObject[e, "System.Boolean"]
MakeNETObject[e_String] := MakeNETObject[e, "System.String"]
MakeNETObject[e_Complex] :=
    Module[{complexType = GetComplexType[]},
        If[Head[complexType] =!= NETType,
            Message[MakeNETObject::complex];
            $Failed,
        (* else *)
            MakeNETObject[e, First[complexType]]
        ]
    ]

MakeNETObject[e:(_Integer | _Real | True | False | _String | _Complex), type_String] :=
    (
        InstallNET[];
        Which[
            (* IntegerQ test is to allow ints to be passed for Enum types. *)
            IntegerQ[e] || MemberQ[allowedTypes[e], type],
                nMakeObject[type, argTypeToInteger[e], e],
            Head[e] === Complex,
                If[Head[GetComplexType[]] =!= NETType,
                    Message[MakeNETObject::complex],
                (* else *)
                    (* type does not match GetComplexType[]. *)
                    Message[MakeNETObject::cmplxmatch]
                ];
                $Failed,
            True,
                Message[MakeNETObject::type, type, e];
                $Failed
        ]
    )
    
MakeNETObject[e:(_Integer | _Real | True | False | _String | _Complex), type_NETType] :=
    (
        InstallNET[];
        If[Head[e] === Complex && type =!= GetComplexType[],
            Message[MakeNETObject::cmplxmatch];
            $Failed,
        (* else *)
            nMakeObject[getAQTypeName[type], argTypeToInteger[e], e]
        ]
    )
    

(*****************  Definitions for 1-deep arrays  ******************)

MakeNETObject[{}] := (Message[MakeNETObject::empty]; $Failed)

MakeNETObject[e_?VectorQ] :=
    Module[{type},
        InstallNET[];
        type = defaultTypeFor[First[e]];
        If[type === $Failed,
            If[Head[First[e]] === Complex,
                Message[MakeNETObject::complex],
            (* else *)
                Message[MakeNETObject::arg, e]
            ];
            $Failed,
        (* else *)
            type = type <> "[]";
            nMakeObject[type, argTypeToInteger[e], e]
        ]
    ]

MakeNETObject[e_?VectorQ, type_String] :=
    Module[{elementType, firstElement},
        InstallNET[];
        If[StringMatchQ[type, "*[]"],
            elementType = StringDrop[type, -2];
            If[e =!= {}, firstElement = First[e]];
            Which[
                (* IntegerQ test is to allow ints to be passed for Enum types. *)
                e === {} || IntegerQ[firstElement] || NETObjectQ[firstElement] || MemberQ[allowedTypes[firstElement], elementType],
                    nMakeObject[type, argTypeToInteger[e], e],
                Head[firstElement] === Complex,
                    If[Head[GetComplexType[]] =!= NETType,
                        Message[MakeNETObject::complex],
                    (* else *)
                        (* type does not match GetComplexType[]. *)
                        Message[MakeNETObject::cmplxmatch]
                    ];
                    $Failed,
                True,
                    Message[MakeNETObject::arraytype, type];
                    $Failed
            ],
        (* else *)
            Message[MakeNETObject::array];
            $Failed
        ]
    ]

(*****************  Definitions for 2-deep arrays  ******************)

MakeNETObject[{{}..}] := (Message[MakeNETObject::empty]; $Failed)

MakeNETObject[e_?MatrixQ] :=
    Module[{type},
        InstallNET[];
        type = defaultTypeFor[e[[1,1]]];
        If[type === $Failed,
            If[Head[e[[1,1]]] === Complex,
                Message[MakeNETObject::complex],
            (* else *)
                Message[MakeNETObject::arg, e]
            ];
            $Failed,
        (* else *)
            type = type <> "[,]";
            nMakeObject[type, argTypeToInteger[e], e]
        ]
    ]

MakeNETObject[e_?MatrixQ, type_String] :=
    Module[{elementType, firstElement, isEmpty},
        InstallNET[];
        Which[
            StringMatchQ[type, "*[,]"],
                elementType = StringDrop[type, -3],
            StringMatchQ[type, "*[][]"],
                elementType = StringDrop[type, -4],
            True,
                Message[MakeNETObject::array2];
                Return[$Failed]
        ];
        isEmpty = First[e] === {};
        If[!isEmpty, firstElement = e[[1,1]]];
        Which[
            (* IntegerQ test is to allow ints to be passed for Enum types. *)
            isEmpty || IntegerQ[firstElement] || NETObjectQ[firstElement] || MemberQ[allowedTypes[firstElement], elementType],
                nMakeObject[type, argTypeToInteger[e], e],
            Head[firstElement] === Complex,
                If[Head[GetComplexType[]] =!= NETType,
                    Message[MakeNETObject::complex],
                (* else *)
                    (* type does not match GetComplexType[]. *)
                    Message[MakeNETObject::cmplxmatch]
                ];
                $Failed,
            True,
                Message[MakeNETObject::arraytype, type];
                $Failed
        ]
    ]


(*****************  Definitions for 3-deep arrays  ******************)

MakeNETObject[e_List /; ArrayDepth[e] == 3] :=
    Module[{type},
        InstallNET[];
        type = defaultTypeFor[e[[1,1,1]]];
        If[type === $Failed,
            If[Head[e[[1,1,1]]] === Complex,
                Message[MakeNETObject::complex],
            (* else *)
                Message[MakeNETObject::arg, e]
            ];
            $Failed,
        (* else *)
            type = type <> "[,,]";
            nMakeObject[type, argTypeToInteger[e], e]
        ]
    ]

MakeNETObject[e_List /; ArrayDepth[e] == 3, type_String] :=
    Module[{elementType, firstElement, isEmpty},
        InstallNET[];
        Which[
            StringMatchQ[type, "*[,,]"],
                elementType = StringDrop[type, -4],
            StringMatchQ[type, "*[][][]"],
                elementType = StringDrop[type, -6],
            StringMatchQ[type, "*[,][]"],
                elementType = StringDrop[type, -5],
            True,
                Message[MakeNETObject::array3];
                Return[$Failed]
        ];
        isEmpty = e[[1,1]] === {};
        If[!isEmpty, firstElement = e[[1,1,1]]];
        Which[
            (* IntegerQ test is to allow ints to be passed for Enum types. *)
            isEmpty || IntegerQ[firstElement] || NETObjectQ[firstElement] || MemberQ[allowedTypes[firstElement], elementType],
                nMakeObject[type, argTypeToInteger[e], e],
            Head[firstElement] === Complex,
                If[Head[GetComplexType[]] =!= NETType,
                    Message[MakeNETObject::complex],
                (* else *)
                    (* type does not match GetComplexType[]. *)
                    Message[MakeNETObject::cmplxmatch]
                ];
                $Failed,
            True,
                Message[MakeNETObject::arraytype, type];
                $Failed
        ]
    ]
    
    
(*****************  Definitions for 2-D and 3-D ragged arrays  ******************)
    
MakeNETObject[e_List] := 
    Module[{firstLeaf, depth, x, type},
        (* We know VectorQ failed, so there must be a List somewhere at the first level.
           If they are not all lists or Null we know we have a bad expression.
        *)
        If[!MatchQ[e, {(_List | Null)..}],
            Message[MakeNETObject::arg, e];
            Return[$Failed]
        ];
        depth = 2;
        x = e[[1, 1]];
        While[ListQ[x], depth++; x = First[x]];
        If[depth > 3,
            Message[MakeNETObject::array4];
            Return[$Failed]
        ];
        firstLeaf = x;
        type = defaultTypeFor[firstLeaf];
        If[type === $Failed,
            If[Head[firstLeaf] === Complex,
                Message[MakeNETObject::complex],
            (* else *)
                Message[MakeNETObject::arg, e]
            ];
            $Failed,
        (* else *)
            Switch[ArrayDepth[e],
                1,
                    (* Ragged in 2nd dimension. *)
                    If[depth == 2,
                        MakeNETObject[e, type <> "[][]"],
                    (* else *)
                        MakeNETObject[e, type <> "[][][]"]
                    ],
                2,
                    (* Ragged in 3rd dimension. *)
                    MakeNETObject[e, type <> "[][][]"],
                _,
                    Message[MakeNETObject::arg, e];
                    $Failed
            ]
        ]
    ]

MakeNETObject[e_List, type_String] :=
    (
        InstallNET[];
        (* Don't bother to try to detect problems with types on the M side. Just send it all to .NET and
           let whatever error might occur get reported.
        *)
        nMakeObject[type, argTypeToInteger[e], e]
    )


(***************  Fallthroughs to issue a message  *****************)

MakeNETObject[a_] := (Message[MakeNETObject::arg, a]; $Failed)
MakeNETObject[a_, type_String] := (Message[MakeNETObject::arg, a]; $Failed)


(**********************  Helper functions  ************************)

defaultTypeFor[e_] :=
    Switch[e,
        _Integer,
            "System.Int32",
        _Real,
            "System.Double",
        _String,
            "System.String",
        True | False,
            "System.Boolean",
        _?NETObjectQ,
            "System.Object",
        _Complex,
            If[Head[GetComplexType[]] =!= NETType,
                $Failed,
            (* else *)
                First[GetComplexType[]]
            ],
        _,
            $Failed
    ]


(* Note some defs use := because they have a call to GetComplexType[] on the rhs. *)

(* This rule for _Integer is no longer used, as we want to allow them to be passed for Enum types. *)
allowedTypes[_Integer] := {"Byte", "SByte", "Char", "Int16", "UInt16", "Int32", "UInt32", "Int64", "UInt64",
                          "Single", "Double", "Decimal", "Object", "System.Byte", "System.SByte", "System.Char", "System.Int16",
                          "System.UInt16", "System.Int32", "System.UInt32", "System.Int64", "System.UInt64",
                          "System.Single", "System.Double", "System.Decimal", "System.Object", getComplexTypeName[]}

allowedTypes[_Real] := {"Single", "Double", "Decimal", "Object", "System.Single", "System.Double",
                        "System.Decimal", "System.Object", getComplexTypeName[]}

allowedTypes[True | False] = {"Boolean", "System.Boolean", "Object", "System.Object"}

allowedTypes[_String] = {"String", "System.String", "Object", "System.Object"}

allowedTypes[_Complex] := {"Object", "System.Object", getComplexTypeName[]}


getComplexTypeName[] := If[# === Null, Null, First[#]]& @ GetComplexType[]


End[]

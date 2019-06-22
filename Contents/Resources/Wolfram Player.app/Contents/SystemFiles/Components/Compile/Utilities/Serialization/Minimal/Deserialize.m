BeginPackage["Compile`Utilities`Serialization`Minimal`Deserialize`"]

deserialize

Begin["`Private`"]

Needs["CompileUtilities`Reference`"] (* For CreateReference *)
Needs["CompileUtilities`Error`Exceptions`"] (* For ThrowException *)

(* private imports *)
Needs["Compile`Utilities`Serialization`Minimal`"]



(* deserialize takes a serialization form of the kind produced by MinimalSerialize and
   produces a serialization of the kind produced by WIRSerialize. This is the inverse of
   serialize. The WIRSerialize kind can then be made into a ProgramModule or
   FunctionModule inside MinimalDeserialize. *)

deserialize[fmIdCounter_?ReferenceQ, "ProgramModule"[fms_?ListQ]] := Module[{
    deserFms, assoc = <||>
},
    deserFms = Map[deserialize[fmIdCounter, #]&, fms];

    AssociateTo[assoc, "functionModules" -> deserFms];
    AssociateTo[assoc, "externalDeclarations" -> "ExternalDeclarations"[<||>]];
    AssociateTo[assoc, "globalValues" -> {}];
    AssociateTo[assoc, "metaInformation" -> {}];
    AssociateTo[assoc, "typeDeclarations" -> {}];

    Assert[Sort@Keys@assoc === Compile`Utilities`Serialization`Minimal`Private`$pmKeys];
    "ProgramModule"[assoc]
]

extractBBName["BasicBlock"[name_String, insts_?ListQ]] := name
extractBBName[args___] := ThrowException["bad args to extractBBName: " <> ToString[{args}]]

deserialize[fmIdCounter_?ReferenceQ, "FunctionModule"[name_String, type_, bbs_?ListQ]] := Module[{
    (* A mapping of String -> Integer *)
    bbNameIds = <||>,
    (* A mapping of String -> Integer
       Each variable name should match the regular expression: var[0-9]+ *)
    varNameIds = CreateReference[<| "idCounter" -> 1 |>],
    (* Each are a mapping of bbId_Integer -> ListReference[_Integer]. These are filled in
       as we visit Branch / Jump instructions using `addRelation`, which also guarntees
       there are no repeated elements. *)
    parentsMap = CreateReference[<||>],
    childrenMap = CreateReference[<||>],
    (* A counter for instruction ids *)
    nextInstId = CreateReference[1],
    arguments = CreateReference[{}],
    (* A reference set when visiting the Return instruction *)
    result = CreateReference[],
    basicBlocks,
    assoc = <||>
},
    MapIndexed[Function[{bb, pos}, Module[{bbName},
        bbName = extractBBName[bb];
        If[KeyExistsQ[bbNameIds, bbName],
            ThrowException["more than one basic block with name '" <> bbName <> "'"];
        ];
        AssociateTo[bbNameIds, bbName -> Length[bbNameIds] + 1];
    ]], bbs];

    basicBlocks = Map[deserialize[varNameIds, bbNameIds, nextInstId, arguments, result,
                      parentsMap, childrenMap, #] &,
                      bbs];
    
    (* We can't add the "children" and "parents" fields until every block has been
       visited, so do that now *)
    basicBlocks = Map[
        Function[bbData0, Module[{bbData = bbData0[[1]], children, parents},
            Assert[MatchQ[bbData0, "BasicBlock"[_?AssociationQ]]];
            Assert[AssociationQ[bbData]];

            children = Replace[childrenMap["lookup", bbData["id"]], {
                ref_?ReferenceQ :> ref["get"],
                Missing[___] :> {}, (* This block has no children, so no entry was made *)
                _ :> ThrowException["Lookup in childrenMap returned unexpected result"]
            }];
            children = Map["BasicBlockID"[#] &, children];

            parents = Replace[parentsMap["lookup", bbData["id"]], {
                ref_?ReferenceQ :> ref["get"],
                Missing[___] :> {}, (* This block has no parents, so no entry was made *)
                _ :> ThrowException["Lookup in parentsMap returned unexpected result"]
            }];
            parents = Map["BasicBlockID"[#] &, parents];

            Assert[MatchQ[children, {"BasicBlockID"[_Integer] ...}]];
            Assert[MatchQ[parents, {"BasicBlockID"[_Integer] ...}]];

            AssociateTo[bbData, "children" -> children];
            AssociateTo[bbData, "parents" -> parents];

            "BasicBlock"[bbData]
        ]],
        basicBlocks
    ];

    If[result["get"] === Null,
        ThrowException["Function " <> name <> " has no Return instructions"]
    ];

    (* Check that the MinimalSerialization function has an explicit type *)
    Replace[type, {
            Undefined | Type[Undefined] | TypeSpecifier[Undefined] :>
                ThrowException["Function has explicitly Undefined type"],
            ((TypeSpecifier|Type)[{args___} -> _]) | ({args___} -> _) :> (
                If[Length[{args}] =!= arguments["length"],
                    ThrowException["Number of LoadArgument instructions differs from number" <>
                                " of parameter given in the type: " <> ToString[{args}]]
                ];
                Null
            ),
            _ :>
                ThrowException[{ "Bad function type expression: ", type }]
        }
    ];

    AssociateTo[assoc, "arguments" -> arguments["get"]];
    AssociateTo[assoc, "basicBlocks" -> basicBlocks];
    AssociateTo[assoc, "id" -> fmIdCounter["increment"]];
    AssociateTo[assoc, "name" -> name];
    AssociateTo[assoc, "result" -> result["get"]]; (* TODO: This should be set to value *)
    AssociateTo[assoc, "type" -> type];

    Assert[Sort@Keys@assoc === Compile`Utilities`Serialization`Minimal`Private`$fmKeys];
    "FunctionModule"[assoc]
]

Compile`Utilities`Serialization`Minimal`Private`$fmKeys = {"arguments", "basicBlocks", "id", "name", "result", "type"};
Compile`Utilities`Serialization`Minimal`Private`$bbKeys = {"children", "id", "instructions", "name", "parents"};

deserialize[varNameIds_?ReferenceQ, bbNameIds_?AssociationQ, nextInstId_?ReferenceQ,
            arguments_?ReferenceQ,
            result_?ReferenceQ,
            parents_?ReferenceQ, children_?ReferenceQ,
            "BasicBlock"[name_String, insts_?ListQ]] := Module[{
    id,
    serInsts,
    assoc = <||>
},
    id = getBBId[bbNameIds, name];
    Assert[IntegerQ[id]];

    serInsts = Map[deserializeInst[varNameIds, bbNameIds, nextInstId, parents, children,
                   id, arguments, result, #] &,
                   insts];

    (* Assert[MatchQ[children["get"], {___Integer}]];
    Assert[MatchQ[parents["get"], {___Integer}]]; *)

    (* AssociateTo[assoc, "children" -> Map["BasicBlockID"[#] &, children["get"]]]; *)
    AssociateTo[assoc, "id" -> id];
    AssociateTo[assoc, "name" -> name];
    AssociateTo[assoc, "instructions" -> serInsts];
    (* AssociateTo[assoc, "parents" -> Map["BasicBlockID"[#] &, parents["get"]]]; *)

    (* Assert[Sort@Keys@assoc === Compile`Utilities`Serialization`Minimal`Private`$bbKeys]; *)
    "BasicBlock"[assoc]
]

deserialize[args___] :=
    ThrowException["bad args to deserialize: " <> ToString[{args}]]


deserializeInst[varNameIds_?ReferenceQ, bbNameIds_?AssociationQ, nextInstId_?ReferenceQ,
                parents_?ReferenceQ, children_?ReferenceQ, currentBB_Integer,
                funcArguments_?ReferenceQ, result_?ReferenceQ,
                instShortName_[contents___]] := Module[{
    assoc, toValue, toVar
},
    toValue[value_] := toValueSerialization[varNameIds, value];
    toVar[var_] := toVarSerialization[varNameIds, var];

    (* Print["instruction: ", instShortName , Join[Map[ToString, {contents}]] ]; *)

    assoc = Replace[instShortName[contents], {
	  "Label"[name_] :>
            <| "name" -> name |>,
       (* TODO: This always sets "compileQ" -> False; is this always valid? *)
        "LoadArgument"[target_, index_] :> (
            funcArguments["appendTo", toVar[target]];
            <| "index" -> index, "target" -> toVar[target], "compileQ" -> False |>
        ),
        "Copy"[target_, source_] :>
            <| "target" -> toVar[target], "source" -> toValue[source] |>,
        "Call"[target_, functionValue_, args_?ListQ] :>
            <| "target" -> toVar[target], "function" -> toValue[functionValue],
               "arguments" -> Map[toValue, args] |>,
        (* Unconditional branch *)
        "Branch"[basicBlockTargets_?ListQ] :> Module[{childBBId},
            Assert[Length[basicBlockTargets] === 1];

            childBBId = getBBId[bbNameIds, basicBlockTargets[[1]]];

            addRelation[children, currentBB, childBBId];
            addRelation[parents, childBBId, currentBB];

            <| "basicBlockTargets" -> {"BasicBlockID"[childBBId]}, "condition" -> None |>
        ],
        "Branch"[s_String] :> Throw[StringForm[
            "Bad MinimalSerialization form of Branch. Perhaps you meant \"Branch\"[{`1`}]", s]],
        (* Conditional branch *)
        "Branch"[condition_, basicBlockTargets_?ListQ] :> Module[{childBBIds},
            childBBIds = Map[getBBId[bbNameIds, #] &, basicBlockTargets];

            (* Add each element of `childBBIds` as a child of the current basic block *)
            Scan[addRelation[children, currentBB, #] &, childBBIds];
            (* Add the current basic block as a parent of each block in `childBBIds` *)
            Scan[addRelation[parents, #, currentBB] &, childBBIds];

            childBBIds = Map["BasicBlockID"[#] &, childBBIds];

            <| "basicBlockTargets" -> childBBIds, "condition" -> toValue[condition] |>
        ],
        "Phi"[target_, sources_?ListQ] :>
            <| "source" -> Map[{"BasicBlockID"@getBBId[bbNameIds, Part[#, 1]], toValue[Part[#, 2]]} &,
                               sources],
               "target" -> toVar[target] |>,
        "StackAllocate"[target_, operator_, size_] :>
            <| "target" -> toVar[target], "operator" -> toValue[operator],
               "size" -> toValue[size] |>,
        "Load"[target_, source_, operator_, operands_?ListQ] :>
            <| "target" -> toVar[target], "source" -> toValue[source],
               "operator" -> toValue[operator], "operands" -> Map[toValue, operands] |>,
        "Return"[value_] :> (
            If[result["get"] === Null,
                result["set", toValue[value]]
                ,
                (* `result` was already set, so we must have already encountered a Return
                   instruction. *)
                ThrowException["Could not set function result because of multiple Return instructions"]
            ];
            <| "value" -> toValue[value] |>
        ),
        "Compare"[target_, operator_, operands_?ListQ] :>
            <| "target" -> toVar[target], "operator" -> toValue[operator],
               "operands" -> Map[toValue, operands] |>,
        "Binary"[target_, operator_, operands_?ListQ] :>
            <| "target" -> toVar[target], "operator" -> toValue[operator],
               "operands" -> Map[toValue, operands] |>,
        "Lambda"[target_, source_] :>
            <| "target" -> toVar[target], "source" -> toValue[source] |>,
        "Inert"[target_, head_, arguments_?ListQ] :>
            <| "target" -> toVar[target], "head" -> toValue[head],
               "arguments" -> Map[toValue, arguments] |>,
        _ :> ThrowException["unimplemented deserialization handler for: " <>
                            ToString[InputForm@instShortName[contents]]]
    }];

    Assert[AssociationQ[assoc]];

    PrependTo[assoc, "id" -> nextInstId["increment"]];
    PrependTo[assoc, "instructionName" -> instShortName <> "Instruction"];

    Assert[MemberQ[Keys@assoc, "instructionName"]];
    Assert[MemberQ[Keys@assoc, "id"]];

    "Instruction"[assoc]
]

deserializeInst[args___] :=
    ThrowException["bad args to deserializeInst: " <> ToString[{args}]]

addRelation[map_?ReferenceQ, bbA_Integer, bbB_Integer] := Module[{list},
    If[ ! map["keyExistsQ", bbA],
        map["associateTo", bbA -> CreateReference[{}]];
    ];
    list = map["lookup", bbA];
    If[ ! list["memberQ", bbB],
        list["appendTo", bbB];
    ];
]
addRelation[args___] :=
    ThrowException["bad args to addRelation: " <> ToString[{args}]]

getBBId[bbNameIds_?AssociationQ, name_?StringQ] := Module[{},
    If[ ! KeyExistsQ[bbNameIds, name],
        ThrowException["could not get basic block id for unknown name: '" <> name <> "'"];
    ];
    Lookup[bbNameIds, name]
]

toValueSerialization[varNameIds_?ReferenceQ, value_] := Module[{},
    If[StringQ[value] && StringMatchQ[value, "var" ~~ ___],
        toVarSerialization[varNameIds, value]
        ,
        "ConstantValue"[<| "value" -> value, "type" -> Undefined |>]
    ]
]

toVarSerialization[varNameIds_?ReferenceQ, value_] := Module[{},
    If[!StringQ[value] || !StringMatchQ[value, "var" ~~ ___],
        ThrowException[{"MinimalSerialization variables should be a string matching 'var.*', got: ", InputForm[value]}]
    ];
    If[ ! varNameIds["keyExistsQ", value],
        varNameIds["associateTo", value -> varNameIds["lookup", "idCounter"]];
        varNameIds["associateTo", "idCounter" -> varNameIds["lookup", "idCounter"] + 1];
    ];
    "Variable"[<| "id" -> varNameIds["lookup", value], "name" -> value,
                  "type" -> Undefined |>]
]

toValueSerialization[args___] :=
    ThrowException["bad args to toValueSerialization: " <> ToString[{args}]]

End[]

EndPackage[]
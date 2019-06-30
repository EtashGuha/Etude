BeginPackage["Compile`Utilities`Serialization`Minimal`Serialize`"]

serialize

Begin["`Private`"]

Needs["Compile`Utilities`Serialization`Minimal`"] (* For MinimalSerialize /
                                                         MinimalSerialization *)
Needs["CompileUtilities`Reference`"] (* For CreateReference *)
Needs["CompileUtilities`Error`Exceptions`"] (* For ThrowException *)



(* `serialize` takes a serialization form of the kind produced by `WIRSerialize` and
   produces the MinimalSerialization kind. *)

(*======================================*)
(* serialize Workers *)
(*======================================*)

serialize["ProgramModule"[assoc_?AssociationQ]] := Module[{
    functionModules, lookup, lowFunctionModules
},
    lookup[key_] := Lookup[assoc, key, Message[MinimalSerialize::nokey, key]];

    functionModules = lookup["functionModules"];
    lowFunctionModules = Map[serialize, functionModules];

    Assert[Sort@Keys@assoc === Compile`Utilities`Serialization`Minimal`Private`$pmKeys];

    (* Assert that every field but "functionModules" is empty. The MinimalSerialization
       form currently has no way of representing these items, so restrict ourselves for
       now to programs which contain none of these.
       ProgramModule's passed as arguments to MinimalSerialize should be constructed with
       "OptimizationLevel" -> None.
    *)
    Assert[lookup["externalDeclarations"] === "ExternalDeclarations"[<||>]];
    Assert[lookup["globalValues"] === {}];
    (* We know it's safe to ommit "TargetSystemID". Each time a new meta data item reaches
       this point and the Assert fails, we'll have to decide whether it's appropriate to
       quitely let that item fall away, or whether it's an item which we need to represent
       some how in the MinimalSerialization *)
    Assert[MatchQ[Normal[lookup["metaInformation"]], {}|{"TargetSystemID" -> _String}]];
    Assert[lookup["typeDeclarations"] === {}];

   "ProgramModule"[lowFunctionModules]
]

serialize["FunctionModule"[assoc_?AssociationQ]] := Module[{
    basicBlocks, name, type,
    lookup,
    lowBBs,
    (* A mapping of variable ids to sequential variable ids starting over at 0.
       This makes the MinimalSerialize form the same regardless of the id's of the
       Variables. "idCounter" is incremented for every new variable encountered. *)
    varIdMap = CreateReference[<|"idCounter" -> 0|>],
    bbIdNames = <||>
},
    Assert[Sort@Keys@assoc === Compile`Utilities`Serialization`Minimal`Private`$fmKeys];

    lookup[key_String] := Lookup[assoc, key, Message[MinimalSerialize::nokey, key]];

    basicBlocks = lookup["basicBlocks"];
    name = lookup["name"];
    type = lookup["type"];

    Assert[StringQ[name]];

    (* Fill in `bbIdNames` with the mapping of BBId_Integer -> name_String *)
    Scan[
        Function[Module[{data = getBasicBlockIdAndName[#]},
            If[MemberQ[Values[bbIdNames], data[[2]]],
                ThrowException["MinimalSerialize: a basic block named '" <> data[[2]] <>
                               "' appears more than once"];
            ];
            AssociateTo[bbIdNames, data[[1]] -> data[[2]] ]
        ]],
        basicBlocks
    ];
   
    lowBBs = Map[serialize[varIdMap, bbIdNames, #]&, basicBlocks];
   
    Assert[ListQ[lowBBs]];
   
    "FunctionModule"[name, type, lowBBs]
]

serialize[varIdMap_?ReferenceQ, bbIdNames_?AssociationQ, "BasicBlock"[assoc_?AssociationQ]] := Module[{
    instructions, name,
    lookup,
    lowInsts
},
    Assert[Sort@Keys@assoc === Compile`Utilities`Serialization`Minimal`Private`$bbKeys];

    lookup[key_String] := Lookup[assoc, key, Message[MinimalSerialize::nokey, key]];
   
    instructions = lookup["instructions"];
    name = lookup["name"];
   
    Assert[StringQ[name]];
    (* Sanity check *)
    Assert[bbIdNames[lookup["id"]] === name];
   
    lowInsts = Map[serialize[varIdMap, bbIdNames, #]&, instructions];
   
    "BasicBlock"[name, lowInsts]
]

serialize[varIdMap_?ReferenceQ, bbIdNames_?AssociationQ, "Instruction"[assoc0_?AssociationQ]] := Module[{
    assoc, instName, unwrap
},
    unwrap[val_] := unwrapValue[varIdMap, val];

    instName = Lookup[assoc0, "instructionName", Message[MinimalSerialize::nokey, "instructionName"]];
    Assert[StringQ[instName]];

    instName = StringReplace[instName, "Instruction" -> ""];

    (* Sort the keys before dispatching; we don't care about order and want to be less
       dependant on the WIR serialize implementation *)
    assoc = KeySort[KeyDrop[assoc0, {"id", "instructionName"}]];

    Replace[instName[assoc], {
        "Label"[<|"name" -> name_String|>] :>
            (* These are just extra information, not necessary *)
            Nothing,
        "LoadArgument"[<| "compileQ" -> _, "index" -> index_Integer, "target" -> target_ |>] :>
            "LoadArgument"[unwrap@target, index],
        "Copy"[<| "source" -> source_, "target" -> target_|>] :>
            "Copy"[unwrap[target], unwrap[source]],
        "Call"[<| "arguments" -> args_?ListQ, "function" -> functionValue_, "target" -> target_ |>] :>
            "Call"[unwrap@target, unwrap@functionValue, Map[unwrap, args]],
        "Branch"[<| "basicBlockTargets" -> targets_, "condition" -> condition_ |>] :>
            "Branch"[Replace[condition, {
                Blank["ConstantValue"] | Blank["Variable"] :> unwrap[condition],
                None -> Sequence[],
                _ :> ThrowException["bad BranchInstruction condition: " <> ToString[condition]]
            }], Map[getBBName[bbIdNames, #]&, targets]],
        "Phi"[<| "source" -> sources_?ListQ, "target" -> target_ |>] :>
            "Phi"[unwrap[target], Map[{getBBName[bbIdNames, #[[1]] ], unwrap@#[[2]]} &, sources]],
        "StackAllocate"[<| "operator" -> operator_, "size" -> size_, "target" -> target_ |>] :>
            "StackAllocate"[unwrap@target, unwrap@operator, unwrap@size],
        "Load"[<| "operands" -> operands_?ListQ, "operator" -> operator_, "source" -> source_, "target" -> target_ |>] :>
            "Load"[unwrap@target, unwrap@source, unwrap@operator, Map[unwrap, operands]],
        "Return"[<| "value" -> value_ |>] :>
            "Return"[unwrap[value]],
        "Compare"[<| "operands" -> operands_List, "operator" -> operator_, "target" -> target_ |>] :>
            "Compare"[unwrap[target], unwrap[operator], Map[unwrap, operands]],
        "Binary"[<| "operands" -> operands_List, "operator" -> operator_, "target" -> target_ |>] :>
            "Binary"[unwrap[target], unwrap[operator], Map[unwrap, operands]],
        "Lambda"[<| "source" -> source_, "target" -> target_ |>] :>
            "Lambda"[unwrap[target], unwrap[source]],
        "Inert"[<| "arguments" -> arguments_List, "head" -> head_, "target" -> target_ |>] :>
            "Inert"[unwrap[target], unwrap[head], Map[unwrap, arguments]],
        _ :> ThrowException["unimplemented serialization handler for: " <> ToString[InputForm@instName[assoc]]]
    }]
]

serialize[args___] := ThrowException["bad args to serialize: " <> ToString[{args}]];

(*======================================*)
(* Utilities *)
(*======================================*)

getBBName[bbIdNames_?AssociationQ, "BasicBlockID"[id_Integer]] :=
    Lookup[bbIdNames, id, ThrowException["no value for key BasicBlockID: " <> ToString[id]]]
getBBName[args___] := ThrowException["Bad args to getBBName: " <> ToString[{args}]]

getBasicBlockIdAndName["BasicBlock"[assoc_?AssociationQ]] := Module[{
    id, name
},
    id = assoc["id"];
    name = assoc["name"];
    Assert[IntegerQ[id]];
    Assert[StringQ[name]];
    {id, name}
] 
getBasicBlockIdAndName[args___] := ThrowException["Bad args to getBasicBlockIdAndName: " <> ToString[{args}]]

unwrapValue[varIdMap_?ReferenceQ, value_] := Module[{const, varId},
    Replace[value, {
        "ConstantValue"[assoc_?AssociationQ] :> (
            const = Lookup[assoc, "value", ThrowException["no key 'value'"]];
            Assert[Sort@Keys@assoc === {"type", "value"}];
            const
        ),
        "Variable"[assoc_?AssociationQ] :> (
            Assert[Sort@Keys@assoc === {"id", "name", "type"}];

            varId = Lookup[assoc, "id", ThrowException["no key 'id'"]];
            Assert[IntegerQ[varId]];
            (* Assert[Lookup[assoc, "type"] === Undefined]; *)
            
            If[ ! varIdMap["keyExistsQ", varId],
                varIdMap["associateTo", varId -> varIdMap["lookup", "idCounter"]];
                varIdMap["associateTo", "idCounter" -> varIdMap["lookup", "idCounter"] + 1];
            ];
            "var" <> ToString[varIdMap["lookup", varId]] 
        ),
        _ :> ThrowException["bad argument to unwrapValue: " <> ToString[value]]
    }]
];
unwrapValue[args___] := ThrowException["bad args to unwrapValue: " <> ToString[{args}]]

End[]

EndPackage[]
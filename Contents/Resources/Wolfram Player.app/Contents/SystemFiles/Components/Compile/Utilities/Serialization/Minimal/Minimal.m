BeginPackage["Compile`Utilities`Serialization`Minimal`"]

(* The serialization produced by this code should not be persisted -- it's unversioned and
   will almost certainly change as the compiler does. *)

(* Take output from WIRSerialize and produces a serialized form of the WIR which does not
   contain instruction or basic block ids. Also renumbers variables to always begin
   counting at one. *)

MinimalSerialize
MinimalSerialize::nokey = "A required key \"`1`\" was not found in the Association"
MinimalSerialize::tyenv = "Object has no associated TypeEnvironment: `1`"

MinimalDeserialize

(* The head of the form returned by MinimalSerialize *)
MinimalSerialization

Begin["`Private`"]

$pmKeys
$fmKeys
$bbKeys

Needs["Compile`"] (* For InitializeCompiler *)
Needs["Compile`Core`PassManager`PassRunner`"] (* For RunPass *)
Needs["Compile`Core`IR`ProgramModule`"] (* For ProgramModuleQ *)
Needs["Compile`Core`IR`FunctionModule`"] (* For FunctionModuleQ *)
Needs["Compile`Utilities`Serialization`"] (* For WIRSerialize *)
Needs["Compile`Utilities`Serialization`Minimal`Serialize`"];
Needs["Compile`Utilities`Serialization`Minimal`Deserialize`"];

Needs["TypeFramework`"] (* For TypeEnvironmentQ *)

Needs["CompileUtilities`Reference`"] (* For CreateReference *)
Needs["CompileUtilities`Error`Exceptions`"] (* For ThrowException *)
Needs["Compile`Core`Lint`IR`"]



(* The keys needed to fully deconstruct / reconstruct a WIRSerialization for an item.
   These should be sorted alphabetically. These appear in matching Assert's in serialize /
   deserialize. *)
$pmKeys = {"externalDeclarations", "functionModules", "globalValues", "metaInformation",
           "typeDeclarations"};
$fmKeys = {"arguments", "basicBlocks", "id", "name", "result", "type"};
$bbKeys = {"children", "id", "instructions", "name", "parents"};

(*======================================*)
(* Serialize *)
(*======================================*)

MinimalSerialize[pm_?ProgramModuleQ] := Module[{
    tyEnv = pm["typeEnvironment"],
    wirSerialization
},
    If[!TypeEnvironmentQ[tyEnv],
        Message[MinimalSerialize::tyenv, tyEnv];
        Return[$Failed];
    ];

    wirSerialization = WIRSerialize[tyEnv, pm];

    Assert[MatchQ[wirSerialization,
                  WIRSerialization["ProgramModule"[_?AssociationQ], <| "Version" -> _Integer |>]]
    ];
    Assert[wirSerialization[[2]]["Version"] === 1];

    MinimalSerialization[serialize[ wirSerialization[[1]] ]]
]

MinimalSerialize[fm_?FunctionModuleQ] := Module[{
    tyEnv = fm["typeEnvironment"], wirSerialization
},
    If[!TypeEnvironmentQ[tyEnv],
        If[!TypeEnvironmentQ[$DefaultTypeEnvironment],
            Message[MinimalSerialize::tyenv, tyEnv];
            Return[$Failed];
        ];
        tyEnv = $DefaultTypeEnvironment;
    ];
    
    wirSerialization = WIRSerialize[tyEnv, fm];

    Assert[MatchQ[wirSerialization,
                  WIRSerialization["FunctionModule"[_?AssociationQ], <| "Version" -> _Integer |>]]
    ];
    Assert[wirSerialization[[2]]["Version"] === 1];

    MinimalSerialization[serialize[ wirSerialization[[1]] ]]
]

MinimalSerialize[args___] := ThrowException["bad args to MinimalSerialize: " <> ToString[{args}]]

(*======================================*)
(* Deserialize *)
(*======================================*)

MinimalDeserialize[MinimalSerialization[data_]] := Module[{
    deser, fmIdCounter = CreateReference[1], obj
},
    deser = deserialize[fmIdCounter, data];
    InitializeCompiler[]; (* We need to do this before using $DefaultTypeEnvironment *)
    (* Wrap the WIRSerialization head around the deserialized form. This is not a call. *)
    deser = WIRSerialization[deser, <| "Version" -> 1 |>];
    obj = WIRDeserialize[Compile`$DefaultTypeEnvironment, deser, "UniqueID" -> True];
    (* Run the lint pass, but intentionally ignore the result. MinimalSerialization is
       used to write tests, so we need to be able to construct intentionally illegal WIR. *)
    Which[
        ProgramModuleQ[obj],
            RunPass[LintIRPass, obj],
        FunctionModuleQ[obj],
            RunPass[LintIRPass, obj],
        True,
            ThrowException[{"Unknown form deserialized in MinimalDeserialize: ", obj}]
    ];
    obj
]

MinimalDeserialize[args___] := ThrowException["bad args to MinimalDeserialize: " <> ToString[{args}]]

End[]

EndPackage[]
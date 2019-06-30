(* Paclet Info File *)

(* created 2016/11/07*)

Paclet[
    Name -> "FunctionResource",
    Version -> "1.2.13",
    MathematicaVersion -> "11.3+",
    Loading -> Automatic,
    Extensions -> 
        {
            {"Kernel", Symbols -> {
                "System`ResourceFunction",
                "System`DefineResourceFunction"
            }
            , Root -> "Kernel", Context -> 
                {"FunctionResourceLoader`","FunctionResource`"}
            }, 
            {"FrontEnd", Prepend -> True}
        }
]



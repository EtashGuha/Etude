(* Paclet Info File *)

(* created 2017/01/30*)

Paclet[
    Name -> "WebSearch",
    Version -> "12.0.5",
    MathematicaVersion -> "11.1+",
    Loading -> Automatic,
    Extensions ->
        {
           {"Kernel", Symbols ->
                {"System`WebSearch", "System`WebImageSearch", "System`AllowAdultContent"},
                Root -> "Kernel", Context -> {"WebSearchLoader`","WebSearch`"}, WolframVersion -> "11.1.0,11.1.1"
            }
            ,
            {"Kernel", Symbols ->
                {"System`WebSearch", "System`WebImageSearch"},
                Root -> "Kernel", Context -> {"WebSearchLoader`","WebSearch`"}, WolframVersion -> "11.2+"
            }
        }
]

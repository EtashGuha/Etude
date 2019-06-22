(* Paclet Info File *)


Paclet[
    Name -> "NeuralNetResource",
    Version -> "1.10.0",
    WolframVersion -> "12.0+",
    Loading -> Automatic,
    Extensions -> 
        {
            {"Kernel", Root -> "Kernel", Context -> 
                {"NeuralNetResourceLoader`","NeuralNetResource`"}},
        	{"FrontEnd", Prepend -> True}
        }
]



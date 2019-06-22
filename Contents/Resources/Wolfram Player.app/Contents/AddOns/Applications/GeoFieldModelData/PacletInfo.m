(* ::Package:: *)

(* Paclet Info File *)

(* created 2014/08/13*)

Paclet[
    Name -> "GeoFieldModelData",
    Version -> "1.3.0",
    MathematicaVersion -> "12+",
    Loading -> Automatic,
    Extensions -> 
        {
            {"Kernel", Symbols -> 
                {"System`GeogravityModelData", "System`GeomagneticModelData"}
            , Root -> "Kernel", Context -> 
                {"GeoFieldModelDataLoader`", "GeoFieldModelData`"}
            }
        }
]



(* ::Package:: *)

(* Paclet Info File *)

(* created Jan. 31 2017*)

Paclet[
    Name -> "SystemTools",
    Version -> "0.0.5",
    MathematicaVersion -> "11.2+",
    Loading -> Automatic,
    Extensions -> {
        {"LibraryLink",SystemID->$SystemID},
        {"Kernel",
            Root -> "Kernel",
            Context -> {
                "SystemTools`"
            },
            HiddenImport->"SystemTools`",
            Symbols-> {
                "System`MemoryAvailable",
                "System`$NetworkConnected",
                "System`DomainRegistrationInformation",
                "SystemTools`Private`$MemoryNames",
                "SystemTools`Private`systemInformation",
                "SystemTools`FileJoin",
                "SystemTools`FilePartition"
            }
        }
    }
]



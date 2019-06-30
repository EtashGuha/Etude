(* ::Package:: *)

(* Paclet Info File *)

(* created 2013/10/02*)

Paclet[
    Name -> "ProcessLink",
    Version -> "0.0.4",
    MathematicaVersion -> "11.3+",
    Loading->Automatic,
    Extensions -> {
        {"LibraryLink"},
        {
            "Kernel",
            Root->"Kernel",
            Context->{"ProcessLinkLoader`","ProcessLink`"},
            Symbols-> 
                {
                    "System`StartProcess",
                    "System`RunProcess",
                    "System`KillProcess",
                    "System`SystemProcesses",
                    "System`SystemProcessData",
                    "System`ProcessConnection",
                    "System`ProcessInformation",
                    "System`ProcessStatus",
                    "System`ProcessObject",
                    "System`Processes",
                    "System`$SystemShell",
                    "System`EndOfBuffer",
                    "System`ReadString",
                    "System`ReadLine",
                    "System`WriteLine",
                    "System`ProcessDirectory",
                    "System`ProcessEnvironment"
                }
        }
    }
]
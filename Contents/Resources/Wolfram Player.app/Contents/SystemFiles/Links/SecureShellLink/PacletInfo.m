(* ::Package:: *)

(* Paclet Info File *)

(* created Jan. 31 2017*)

Paclet[
    Name -> "SecureShellLink",
    Version -> "0.0.8",
    MathematicaVersion -> "11.2+",
    Loading -> Automatic,
    Extensions -> {
        {"LibraryLink",SystemID->$SystemID},
        {"Kernel",
            Root -> "Kernel",
            Context -> {
                "SecureShellLink`"
            },
            HiddenImport->"SecureShellLink`",
            Symbols-> {
                "System`RemoteRun",
                "System`RemoteRunProcess",
                "System`RemoteConnect",
                "System`RemoteConnectionObject",
                "System`RemoteAuthorizationCaching",
                "System`$SSHAuthentication",
                "System`RemoteFile",
                "SecureShellLink`RemoteCopyFile"
            }
        }
    }
]

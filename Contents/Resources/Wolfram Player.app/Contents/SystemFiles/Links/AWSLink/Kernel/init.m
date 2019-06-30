
(*
The following line make sure ErrorHandling.m is the first file to be loaded.
(because macros such as FailInOtherCases are located there.)
Then other, then Package["AWSLink`"] is found and loading happens in alphabetical order
*)
Get @ FileNameJoin[{FileNameDrop[$InputFileName], "ErrorHandling.m"}]

(*
Package["AWSLink`"]
PackageImport["JLink`"];
*)
(*
SetOptions[ReinstallJava, JVMArguments -> "-Xmx" <> ToString@Max[512, Floor[1/4*$SystemMemory/10^6]] <> "m"];
ReinstallJava[JVMArguments->"-Xmx"<>ToString@Max[512, Floor[1/4 *$SystemMemory/10^6]]<>"m"];
*)

If[TrueQ[Databases`Private`$DevelopmentMode],
    Databases`Private`ClearPackage["Database"];
]

(* Working around bug 352499. Needed here since Database depends on EF *)
Needs["EntityFramework`"] 

Get["Databases`Database`Operation`"]

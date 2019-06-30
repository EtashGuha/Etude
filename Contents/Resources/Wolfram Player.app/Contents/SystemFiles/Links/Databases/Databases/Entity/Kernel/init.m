
If[TrueQ[Databases`Private`$DevelopmentMode],
    Databases`Private`ClearPackage["Entity"];
]

Needs["EntityFramework`"] (* Working around bug 352499 *)

Get["Databases`Entity`EntityFunction`"]

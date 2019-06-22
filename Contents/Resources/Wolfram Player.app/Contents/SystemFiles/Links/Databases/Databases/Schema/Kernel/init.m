
If[TrueQ[Databases`Private`$DevelopmentMode],
    Databases`Private`ClearPackage["Schema"];
]


Get["Databases`Schema`Patterns`"]

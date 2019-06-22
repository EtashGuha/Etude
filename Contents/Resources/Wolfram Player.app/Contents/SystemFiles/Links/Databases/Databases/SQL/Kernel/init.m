
If[TrueQ[Databases`Private`$DevelopmentMode],
    Databases`Private`ClearPackage["SQL"];
]


Get["Databases`SQL`Init`"] (* NOTE: calling this file first is important *)

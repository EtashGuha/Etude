
If[
    TrueQ[Databases`Private`$DevelopmentMode],
        Databases`Private`ClearPackage["Python"];
        Get["Databases`Python`Python`"]
    ,
    Get["Databases`Python`Python`"]
]




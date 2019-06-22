BeginPackage["TypeFramework`Utilities`TypeEnvironmentUtilities`"]

$TypeEnvironment
ClearTypeEnvironment
InitializeTypeEnvironment

Begin["`Private`"]

Needs["TypeFramework`"]
Needs["TypeFramework`Environments`TypeEnvironment`"]
Needs["TypeFramework`Utilities`Error`"]

$TypeEnvironmentInitialized

If[!System`ValueQ[$TypeEnvironmentInitialized],
    $TypeEnvironmentInitialized = False
]

ClearTypeEnvironment[] := (
    $TypeEnvironment = Function[{method}, TypeFailure["TypeEnvironment", "$TypeEnvironment is uninitialized. Please evaluate InitializeTypeEnvironment[] first."]];
    $TypeEnvironmentInitialized = False
);


If[!TrueQ[$TypeEnvironmentInitialized],
    ClearTypeEnvironment[]
]

InitializeTypeEnvironment[] :=
(
    If[!$TypeEnvironmentInitialized,
        InitializeTypeFrameworkClasses[];
        $TypeEnvironment = CreateTypeEnvironment[];
        $TypeEnvironmentInitialized = True
    ]
)

End[]

EndPackage[]


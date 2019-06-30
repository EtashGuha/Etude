BeginPackage["Compile`Core`IR`Lower`Primitives`General`"]

Begin["`Private`"]


Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Callback`"]

(* private imports *)
Needs["Compile`Core`IR`Lower`Primitives`Inert`"]



lower := lower = Compile`Core`IR`Lower`Primitives`Inert`Private`lower

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive["General"], lower]
]]

End[]

EndPackage[]

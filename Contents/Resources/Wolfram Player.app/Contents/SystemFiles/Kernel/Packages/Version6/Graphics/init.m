
(** set graphics outputs to settings suitable for a
    6.0 notebook front end **) 

Begin["System`"] (* currently unnecessary, since no new symbols defined *)

(* Display streams *)

$Display = {"stdout"}
$SoundDisplay = {"stdout"}

(* Display functions *)
$DisplayFunction = Identity
$SoundDisplayFunction = Identity

SetSystemOptions[ "GraphicsBoxes" -> True]
Developer`SymbolicGraphics[]
$ContextPath = DeleteCases[$ContextPath, "Graphics`Legacy`"]

End[]

Null


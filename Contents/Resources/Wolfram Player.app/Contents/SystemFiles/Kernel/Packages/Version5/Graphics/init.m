
(** send graphics to stdout in PostScript format **)

Begin["System`"] 

$Display := {$PSDirectDisplay}
$SoundDisplay := {$PSDirectDisplay}
$PSDirectDisplay = "stdout"

$DisplayFunction = Display[$Display, #, "MPS"] &
$SoundDisplayFunction = Display[$SoundDisplay, #, "MPS"]&

SetSystemOptions[ "GraphicsBoxes" -> False]
Developer`LegacyGraphics[]
$ContextPath = DeleteCases[$ContextPath, "Graphics`Legacy`"]
$ContextPath = Prepend[$ContextPath, "Graphics`Legacy`"]

End[]

If[ !($BatchOutput || $Linked || $ParentLink =!= Null),
	Print[" -- PostScript to stdout graphics initialized -- "] ]

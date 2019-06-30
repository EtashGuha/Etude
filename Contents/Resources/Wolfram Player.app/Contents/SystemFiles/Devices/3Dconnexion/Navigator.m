
(* Windows *)

"3Dconnexion Navigator" ->
{
"X" -> "X Axis",
"Y" -> "Z Axis",
"Z" -> "Y Axis",
"X1" -> "X Axis",
"Y1" -> "Z Axis",
"Z1" -> "Y Axis",
"X2" -> "X Rotation",
"Y2" -> "Z Rotation",
"Z2" -> "Y Rotation",
"X3" -> Switch[{"Button 1", "Button 2"}, {False, False}, 0, {True, False}, -1, {False, True}, 1, {True, True}, 0.00001],
"CommonKnownAxes"
}

(* Other *)

"SpaceNavigator" ->
{
"X" -> "X Axis",
"Y" -> (-"Y Axis"),
"Z" -> (-"Z Axis"),
"X1" -> "X Axis",
"Y1" -> (-"Y Axis"),
"Z1" -> (-"Z Axis"),
"X2" -> "X Rotation",
"Y2" -> (-"Y Rotation"),
"Z2" -> (-"Z Rotation"),
"X3" -> Switch[{"Button 1", "Button 2"}, {False, False}, 0, {True, False}, -1, {False, True}, 1, {True, True}, 0.00001],
"CommonKnownAxes"
}


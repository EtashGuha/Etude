
(* Older Generic case *)

"XBOX 360 For Windows (Controller)" ->
{
"X2" -> "X Rotation",
"Y2" -> (-"Y Rotation"),
"Z" -> (-"Z Axis"),
"Z1" -> (-"Z Axis"),
"X4" -> Switch[{"Button 3", "Button 2"}, {False, False}, 0, {True, False}, -1, {False, True}, 1, {True, True}, 0.00001],
"Y4" -> Switch[{"Button 1", "Button 4"}, {False, False}, 0, {True, False}, -1, {False, True}, 1, {True, True}, 0.00001],
"JB" -> "Button 9",
"JB1" -> "Button 9",
"JB2" -> "Button 10",
"A Button" -> "Button 1",
"B Button" -> "Button 2",
"X Button" -> "Button 3",
"Y Button" -> "Button 4",
"TLB" -> "Button 5",
"TRB" -> "Button 6",
"Select Button" -> "Button 7",
"Back Button" -> "Button 7",
"Start Button" -> "Button 8",
"CommonKnownAxes"
}

(* Generic Case *)

"Controller (XBOX 360 For Windows)" ->
{
"X2" -> "X Rotation",
"Y2" -> (-"Y Rotation"),
"Z" -> (-"Z Axis"),
"Z1" -> (-"Z Axis"),
"X4" -> Switch[{"Button 3", "Button 2"}, {False, False}, 0, {True, False}, -1, {False, True}, 1, {True, True}, 0.00001],
"Y4" -> Switch[{"Button 1", "Button 4"}, {False, False}, 0, {True, False}, -1, {False, True}, 1, {True, True}, 0.00001],
"JB" -> "Button 9",
"JB1" -> "Button 9",
"JB2" -> "Button 10",
"A Button" -> "Button 1",
"B Button" -> "Button 2",
"X Button" -> "Button 3",
"Y Button" -> "Button 4",
"TLB" -> "Button 5",
"TRB" -> "Button 6",
"Select Button" -> "Button 7",
"Back Button" -> "Button 7",
"Start Button" -> "Button 8",
"CommonKnownAxes"
}

(* XInput installed *)

"Xbox 360 Controller for Windows" ->
{
"X2" -> "X Rotation",
"Y2" -> (-"Y Rotation"),
"X4" -> Switch[{"X Button", "B Button"}, {False, False}, 0, {True, False}, -1, {False, True}, 1, {True, True}, 0.00001],
"Y4" -> Switch[{"A Button", "Y Button"}, {False, False}, 0, {True, False}, -1, {False, True}, 1, {True, True}, 0.00001],
"Z" -> (-"Slider" + "Slider 2"),
"Z1" -> (-"Slider"),
"Z2" -> "Slider 2",
"B" -> "A Button",
"B1" -> "A Button",
"Button 1" -> "A Button",
"B2" -> "B Button",
"Button 2" -> "B Button",
"B3" -> "X Button",
"Button 3" -> "X Button",
"B4" -> "Y Button",
"Button 4" -> "Y Button",
"B5" -> "Left Bumper Button",
"Button 5" -> "Left Bumper Button",
"TLB" -> "Left Bumper Button",
"B6" -> "Right Bumper Button",
"Button 6" -> "Right Bumper Button",
"TRB" -> "Right Bumper Button",
"B7" -> "Back Button",
"Button 7" -> "Back Button",
"Select Button" -> "Back Button",
"B8" -> "Start Button",
"Button 8" -> "Start Button",
"B9" -> "Left Thumbstick Button",
"Button 9" -> "Left Thumbstick Button",
"JB" -> "Left Thumbstick Button",
"JB1" -> "Left Thumbstick Button",
"B10" -> "Right Thumbstick Button",
"Button 10" -> "Right Thumbstick Button",
"JB2" -> "Right Thumbstick Button",
"CommonKnownAxes"
}


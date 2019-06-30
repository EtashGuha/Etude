If[FE`Evaluate[$OperatingSystem] === "Linux",
_ ->
{
"X2" -> "Unknown Axis 3",
"Y2" -> (-"Unknown Axis 4"),
"JB" -> "Button 11",
"JB1" -> "Button 11",
"JB2" -> "Button 12",
"JB3" -> None,
"TLB" -> "Button 5",
"TRB" -> "Button 6",
"BLB" -> "Button 7",
"BRB" -> "Button 8",
"CommonUnknownAxes"
}
,
_ ->
{
"X2" -> "Z Axis",
"Y2" -> (-"Z Rotation"),
"JB" -> "Button 11",
"JB1" -> "Button 11",
"JB2" -> "Button 12",
"JB3" -> None,
"TLB" -> "Button 5",
"TRB" -> "Button 6",
"BLB" -> "Button 7",
"BRB" -> "Button 8",
"CommonKnownAxes"
}
]


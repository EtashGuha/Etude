

BeginPackage["Compile`Core`IR`Internal`Show`"]




$OperatorColor := $OperatorColor = RGBColor[0.6, 0.4, 0.4];
$SourceVariableColor := $SourceVariableColor = RGBColor[0.5, 0.5, 0.5];
$GlobalVariableColor := $GlobalVariableColor = Blue;
$TypeColor := $TypeColor = Darker[Red];
$LabelColor := $LabelColor = RGBColor[0.9, 0.5, 0.2];
$ConstantValueColor := $ConstantValueColor = RGBColor[0.3, 0.3, 0.3]; 


$TargetVariableColor := $TargetVariableColor = RGBColor[0.269, 0.538, 0.356]; 
$VariableColor := $VariableColor = $TargetVariableColor

(*
getColor[arg_] :=
	Which[
	    ConstantVariableQ[arg],
			$ConstantValueColor,
		True,
			$TargetVariableColor
	]
*)

EndPackage[]

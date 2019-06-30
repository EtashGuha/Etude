Package["NeuralNetworks`"]

PackageScope["toNumericArray"]

toNumericArray[a_, type_:"Real32"] := NumericArray[a, type, "ClipAndCoerce"];
(* ^ To avoid a crash with extreme values
     In[*]:= $MaxMachineNumber -> First @ Normal @ NumericArray[{$MaxMachineNumber}, "Real32", "ClipAndCoerce"]
     Out[*]= 1.79769*10^308 -> 3.40282*10^38
*)

PackageScope["arrayMin"]
PackageScope["arrayMax"]
PackageScope["arrayMinMax"]

(* these exist because Min/Max don't work properly on NAs yet *)
arrayMinMax[e_NumericArray] := {Min[e], Max[e]};
arrayMinMax[e_ ? PackedArrayQ] := MinMax[e];
arrayMinMax[e_] := MinMax[e /. na_NumericArray :> arrayMinMax[na]];

arrayMin[e_NumericArray] := Min[e];
arrayMax[e_NumericArray] := Max[e];
arrayMin[e_ ? PackedArrayQ] := Min[e];
arrayMax[e_ ? PackedArrayQ] := Max[e];
arrayMin[e_] := Min[e /. na_NumericArray :> Min[na]];
arrayMax[e_] := Max[e /. na_NumericArray :> Max[na]];


PackageScope["arrayMean"]

(* this is a flat mean *)
arrayMean[e_NumericArray] := arrayMean @ Normal @ e;
arrayMean[e_List] := If[VectorQ[e], Mean[e], Mean @ Flatten @ e];
arrayMean[e_] := e;

(* jeromel: The following was introduced to safely cope with change of behaviour (bug 368999) *)
PackageScope["arrayDimensions"]
PackageScope["arrayDepth"]
PackageScope["machineArrayDimensions"]
PackageScope["arrayFlatten"]

arrayDimensions[e_, args___] := Dimensions[e, args, AllowedHeads -> {List, NumericArray}];
arrayDepth[e_] := ArrayDepth[e, AllowedHeads -> {List, NumericArray}];

(* The behaviour of GeneralUtilities`MachineArrayDimensions also changed.
	For now fixing everything here 
*)
machineArrayDimensions[e_] := Which[
	MachineArrayQ[e], arrayDimensions[e],
	MachineQ[e], {}, 
	True, $Failed
];

arrayFlatten[e:{__NumericArray}, args___] := Flatten[Normal /@ e, args];
arrayFlatten[e_, args___] := Flatten[e, args];
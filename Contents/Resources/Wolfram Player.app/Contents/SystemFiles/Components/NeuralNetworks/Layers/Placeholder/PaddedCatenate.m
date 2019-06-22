(* 
This is a dummy layer needed to import nn.DepthConcat from Torch, a concatenation
layer featuring automatic zero-padding to match the spatial map size of its own inputs
(smaller inputs are zero-padded so as to match the size of the largest).
It has no evaluation code, and it's only used to trigger shape inference for its output.
It is then replaced by the correct combination of PaddingLayer(s) and CatenateLayer 
(see replaceDummyLayers in ImportTorchFormat.m).
*)

Inputs: 
	$Multiport: RealTensorT

Output: RealTensorT

ShapeFunction: List[PaddedCatenateShape[#]]&

RankFunction: List[CatenateRank[#, 1]]&

PaddedCatenateShape[dimLists_] := Prepend[Rest@MapThread[Max, dimLists], Total@dimLists[[All, 1]]];
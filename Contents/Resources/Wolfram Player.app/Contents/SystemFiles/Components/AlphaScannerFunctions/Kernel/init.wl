(* ::Package:: *)

(* ::Chapter:: *)
(*package header*)


(* :Summary: Support for some Wolfram|Alpha scanner functionality in the Wolfram Language. *)

(* :Mathematica Version: Mathematica 12 *)

(* :Author: Paco Jain (pacoj@wolfram.com *)

(* :Keywords: *)

(* :Discussion: *)

(* :Warning: *)

(* :Sources: *)




BeginPackage["AlphaScannerFunctions`"]


(* ::Chapter:: *)
(*Get individual code files*)


With[
	{path = DirectoryName[$InputFileName]},
	DeleteCases[FileNames["*.wl",{path}], FileNameJoin[{path,"init.wl"}]]
];


(* ::Chapter:: *)
(*package footer*)


EndPackage[]

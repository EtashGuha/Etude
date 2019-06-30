(* This file loaded by Get/Needs["TypeSystem`"]. It must load the package files by calling TypeSystemLoader.m,
   and also ensure that TypeSystem` context is on $ContextPath, which is not done by TypeSystemLoader.
*)

Off[Atom::shdw]; (* because of TypeSystem`TypeSystem`Atom, see 359829 *)

BeginPackage["TypeSystem`"];
EndPackage[];

Get["TypeSystemLoader`"];

On[Atom::shdw];
(* Master Loader of Parallel Computing Toolkit *)

(* :Package Version: 3.0  *)

(* :Summary:
   Get["Parallel`Kernel`subinit`"]
*)

(* :Discussion:
   this initializes subkernels; it is called from sysload.m,
   if System`Parallel`$SubKernel is True
*)

(* create all the symbols *)

Symbol/@Parallel`Private`mainNames;
Symbol/@Parallel`Private`nonLoadNames;
(* Symbol/@Parallel`Private`devNames; *)

(* important early settings *)

Begin["Parallel`Private`"]
`masterLink[link_LinkObject] := (
	Unprotect[Parallel`Client`Private`$link];
	Parallel`Client`Private`$link=link;
	Protect[Parallel`Client`Private`$link];
	MathLink`AddSharingLink[link, MathLink`Terminating -> True];
	MathLink`LinkSetPrintFullSymbols[link, True];
)

End[]

If[!ValueQ[Parallel`Client`Private`$link], Parallel`Private`masterLink[$ParentLink]]; (* default is $ParentLink *)
Parallel`Client`Private`$cp = $ContextPath; Protect[Parallel`Client`Private`$cp];

(* set $EvaluationEnvironment *)

Unprotect[$EvaluationEnvironment];
$EvaluationEnvironment = "Subkernel";
Protect[$EvaluationEnvironment];


BeginPackage["Parallel`"] (* empty context *)

(* load remaining stuff,  but should not show up on context path *)
Get["Parallel`NullMaster`"]

(* system-specific initializations; guard against unsupported features *)


(* change messages for subkernels *)

General::subpar = "Parallel computations cannot be nested; proceeding with sequential evaluation."
General::subnopar = "Parallel programming is not available in subkernels of another parallel computation."

EndPackage[]
$ContextPath=DeleteCases[$ContextPath, "Parallel`", 1, 1]; (* no longer on context path *)

(* support code *)
Get["Parallel`Client`"]
$ContextPath=DeleteCases[$ContextPath, "Parallel`Client`", 1, 1]; (* no longer on context path *)

(* e o f *)

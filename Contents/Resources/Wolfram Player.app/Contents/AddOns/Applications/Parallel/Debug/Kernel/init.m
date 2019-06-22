(* :Title: Parallel`Debug` -- User-Level PCT debugging compatibility package *)

(* :Context: Parallel`Debug` *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   support old-style Needs["Parallel`Debug`"] under Parallel Tools
 *)

(* :Package Version: 4.0  *)

(* :Mathematica Version: 9 *)

(* :Note: this package must be read *before* the Parallel Computing Toolkit itself.
	Otherwise, if debugging was enabled during load, just put context back on path if it got lost. *)

Which[

	TrueQ[Parallel`Debug`$Debug], (* alread loaded; just make sure context is on path *)
		If[ !MemberQ[$ContextPath, "Parallel`Debug`"], PrependTo[$ContextPath, "Parallel`Debug`"]];

	, NameQ["Parallel`Private`$PackageVersion"], (* too late, Parallel Tools already loaded without debugging *)
		Parallel`Debug`$Debug::toolate = "The debugging package cannot be read after the Parallel Tools.";
		Message[Parallel`Debug`$Debug::toolate];
		Abort[];

	, True, (* set up loading *)
		Parallel`Static`$loadDebug=True;
		$KernelID;
]

(* autoload helper for Parallel Tools *)

(* :Summary:
   this is read when a symbol declared with DeclareLoad is used for the first time
*)

(* :Discussion:
   makes sure the right version of PCT is first on $Path, then read Kernel`autoinit`
*)

AbortProtect[ (* make sure we are not interrupted at a bad time *)
	
Parallel`Static`$autoload=True;

(* remember path for diagnostic purposes *)

Parallel`Private`autodir = ParentDirectory[ParentDirectory[DirectoryName[$InputFileName]]];

(* set up $Path to avoid loading old versions of PCT installed *)

If[StringQ[Parallel`Private`tooldir], PrependTo[$Path, Parallel`Private`tooldir]];


(* temporarily turn off shadowing warning *)

Parallel`Private`shdw = Head[General::shdw] =!= $Off; Off[General::shdw];


(* taken from Parallel`Palette` *)
(* debugging must be read before PCT, so we need to look at the preferences to set the variable early *)
Parallel`Palette`Private`remctx = !MemberQ[$ContextPath, "Parallel`Preferences`"];
Needs["Parallel`Preferences`"];
If[ !ValueQ[Parallel`Static`$loadDebug] && Parallel`Preferences`debugQ[], Parallel`Static`$loadDebug=True ];
If[ Parallel`Palette`Private`remctx, $ContextPath=DeleteCases[$ContextPath, "Parallel`Preferences`"] ];

Get["Parallel`Kernel`autoinit`"];

If[StringQ[Parallel`Private`tooldir], $Path = DeleteCases[$Path, Parallel`Private`tooldir, 1, 1]]; (* restore *)


If[Parallel`Private`shdw, On[General::shdw]]; (* condrestore shadowing warning *)

] (* AbortProtect *)

(* for sysload/autolaunch also launch kernels or prepare to load them on demand *)
(* note that symbols with values cannot be used here! *)

If[Parallel`Static`$sysload, Which[
	Parallel`Static`$autolaunch===True,
		Parallel`Protected`doAutolaunch[False],
	Parallel`Static`$autolaunch===Automatic,
		Parallel`Protected`declareAutolaunch[
			Parallelize, ParallelTry, ParallelCombine, ParallelEvaluate,
			ParallelMap, ParallelTable, ParallelSum, ParallelProduct, ParallelDo, ParallelArray,
			WaitAll, WaitNext,
			Parallel`Developer`ParallelDispatch, Parallel`Developer`QueueRun
		]
]]

Null

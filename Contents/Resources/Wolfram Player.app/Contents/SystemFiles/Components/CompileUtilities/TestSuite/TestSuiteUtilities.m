
BeginPackage["CompileUtilities`TestSuite`TestSuiteUtilities`"]

FailurePassthrough::usage = "FailurePassthrough[expr] returns the first occurence of Failure[] that appears inside of expr, or expr itself."

MemoryLeakCheck::usage = "MemoryLeakCheck[ expr] returns True if repeated evaluation of expr does not leak memory and False otherwise."





EnableAllRuntimeChecks::usage = "EnableAllRuntimeChecks[] enables all runtime checks"



Begin["`Private`"]


Needs["CompileUtilities`RuntimeChecks`"]
Needs["LLVMLink`Unsafe`"]
Needs["LLVMLink`RuntimeChecks`"]
Needs["CompileUtilities`Error`Exceptions`"]


Attributes[FailurePassthrough] = {HoldAll}

FailurePassthrough[expr0_] :=
	Module[{expr, failures, failure},
		Block[{Failure},
			expr = expr0;
			(* first look for Failure[] *)
			failures = Cases[expr, Failure[___], Infinity, Heads -> True];
			If[failures =!= {},
				failure = failures[[1]];
				failure
				,
				(* now just look for $Failed *)
				failures = Cases[expr, $Failed, Infinity, Heads -> True];
				If[failures =!= {},
					failure = failures[[1]];
					failure
					,
					(* neither Failure[] nor $Failed appeared in result *)
					expr
				]
			]
		]
	];

(*
  Memory management,  run computations twice, just to avoid any warmup memory cost, eg for caches.
*)

SetAttributes[ MemoryLeakCheck, HoldAll]

MemoryLeakCheck[ expr_, lims_:{100, 100000}, tol_:0] :=
	Module[ {mem, val},
		mem = MemoryInUse[]; 
		Do[expr, {First[lims]}]; 
		5.6; 
		MemoryInUse[] - mem;
		mem = MemoryInUse[]; 
		Do[expr, {Last[lims]}]; 
		5.6; 
		val = Abs[(MemoryInUse[] - mem)];
		If[val <= tol, True, Echo[val]; False]
	]




EnableAllRuntimeChecks[] := (

	On[Assert];
	$AssertFunction = Function[{assert}, Throw[{"Assert Failure", assert}, "Assert"]];

	EnableRuntimeChecks[];

	DisableUnsafeLLVMLinkFunctions[];

	EnableLLVMRuntimeChecks[];
)



End[]

EndPackage[]

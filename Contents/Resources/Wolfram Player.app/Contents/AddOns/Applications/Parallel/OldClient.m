(* :Title: OldClient.m -- set up older subkernels  *)

(* :Context: Parallel`OldClient` *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   runtime initialization of a new subkernel with a lower Mathematica version to
   bring it at the same level as new kernels that read those initializations
   from Parallel`Client`.
   This file is read only if an old subkernel needs to be initialized.
   The lowest parallel language version supported is 7.0, which corresponds to Mathematica 7
 *)

(* :Package Version: 2.0  *)

(* :Mathematica Version: 11.1 *)


BeginPackage["Parallel`OldClient`"]

initOldKernel::usage = "initOldKernel[kernel, version] sets up compatibility definitions in kernel."

(* needs symbols from other parallel contexts *)
BeginPackage["Parallel`Developer`"]
EndPackage[]

BeginPackage["Parallel`Protected`"]
EndPackage[]

BeginPackage["Parallel`Client`"] (* make sure symbols are found *)

`HoldCompound

`CallBackPacket
`ReplyPacket
`CallBack
`remoteIdentity

`setSharedVariable
`setSharedFunction
`unsetShared

`makeDefinitions

EndPackage[]

(* master side of HoldCompound *)
holdCompound = Parallel`Client`HoldCompound

Begin["`Private`"]

`$PackageVersion = 1.0;
`$thisFile = $InputFileName

(* definitions previously done at init time through mathlink; note the comma in place of ; *)

(* from 8.0 to 9, need to amend setSharedVariable *)

$initDefsPre9 = holdCompound[

  Clear[setSharedVariable],

  setSharedVariable[s_, attrs_] := Module[{},
	Unprotect[s]; ClearAll[s];
	(* for all variables read access *)
	s := CallBack[s];
    s/: c:HoldPattern[Part[s,__]] := CallBack[c]; (* Part[Unevaluated[s], ...] *)
    s/: c:HoldPattern[Extract[s,__]] := CallBack[c]; (* Extract[Unevaluated[s], ...] *)
    s/: c:HoldPattern[Lookup[s,__]] := CallBack[c]; (* Lookup[s, ...] *)
    s/: c:HoldPattern[KeyExistsQ[s,__]] := CallBack[c]; (* KeyExistsQ[Unevaluated[s], ...] *)
    s/: c:HoldPattern[Keys[s]] := CallBack[c]; (* Keys[Unevaluated[s], ...] *)
    s/: c:HoldPattern[Values[s]] := CallBack[c]; (* Values[Unevaluated[s], ...] *)

    (* for mutable variables *)
    If[ !MemberQ[attrs, Protected], With[{pp = Unprotect[Part]},
        s/: c:HoldPattern[s =rhs_] := (CallBack[c;];rhs); (* can we return the local copy of rhs? *)
        s/: c:HoldPattern[s:=rhs_] := CallBack[s:=Parallel`Developer`SendBack[rhs]];
        s/: c:(s++) := CallBack[c];
        s/: c:(s--) := CallBack[c];
        s/: c:(++s) := CallBack[c];
        s/: c:(--s) := CallBack[c];
        s/: c:AppendTo[s,rhs_]  := CallBack[c;]; (* don't waste bandwidth *)
        s/: c:PrependTo[s,rhs_] := CallBack[c;]; (* don't waste bandwidth *)
        s/: c:(s+=v_) := CallBack[c];
        s/: c:(s-=v_) := CallBack[c];
        s/: c:(s*=v_) := CallBack[c];
        s/: c:(s/=v_) := CallBack[c];
        (* associations *)
        s/: c:AssociateTo[s,__]  := CallBack[c;]; (* don't waste bandwidth *)
        s/: c:KeyDropFrom[s,__]  := CallBack[c;]; (* don't waste bandwidth *)
        (* part assignments *)
        Part/: c:HoldPattern[Part[s,args__]=rhs_] :=
        	Replace[{args}, {brgs___} :> CallBack[Part[s,brgs]=rhs]];
        Part/: c:HoldPattern[AppendTo[Part[s,args__],rhs_]] :=
        	Replace[{args}, {brgs___} :> CallBack[AppendTo[Part[s,brgs],rhs]]];
        Part/: c:HoldPattern[PrependTo[Part[s,args__],rhs_]] :=
        	Replace[{args}, {brgs___} :> CallBack[PrependTo[Part[s,brgs],rhs]]];

      Protect[pp]]
    ];
    Attributes[s] = Union[attrs,{Protected}];
  ]

]

(* case by case for all supported older language versions *)

initOldKernel[kernels_, subLanguageVersion_] := Module[{},
	With[{masterVersion = Parallel`Private`$ParallelLanguageVersion},
		kernelInitialize[Parallel`Client`$masterVersion=masterVersion;, kernels]]; (* record our setting *)
	If[subLanguageVersion < 9, (* 8.0 *)
		With[{clientCode=$initDefsPre9}, kernelInitialize[ clientCode, kernels ]];
	];
]


End[]

EndPackage[]

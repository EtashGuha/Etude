(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["FunctionResource`"]

Begin["`Private`"] (* Begin Private Context *) 


ResourceSystemClient`Private`repositoryresourceaccess[$FunctionResourceTypes,args___]:=functionresourceaccess[args]
	
functionresourceaccess[fun0_,id_,info_,rest___]:=Block[{funcObj, fun=prepareAccessFunction[fun0]},
    funcObj=ResourceFunction[id];
    fun[funcObj, rest]
]/;MemberQ[$FunctionResourceAccessors,fun0]

neuralnetresourceaccess[___]:=$Failed

prepareAccessFunction[Get]=Identity
prepareAccessFunction[f_]:=f

$FunctionResourceAccessors={Get};

End[] (* End Private Context *)

EndPackage[]

SetAttributes[{},
   {ReadProtected, Protected}
];
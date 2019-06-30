(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["NeuralNetResource`"]

Begin["`Private`"] (* Begin Private Context *) 


ResourceSystemClient`Private`repositoryresourceaccess[$NeuralNetResourceType,args___]:=neuralnetresourceaccess[args]
	
neuralnetresourceaccess[fun0_,id_,info_,rest___]:=Block[{data, fun=prepareAccessFunction[fun0]},
    data=DataResource`resourceDataElement[{id, info},"EvaluationNet"];
    fun[data, rest]
]/;MemberQ[$NeuralNetResourceAccessors,fun0]

neuralnetresourceaccess[___]:=$Failed

prepareAccessFunction[Get]=Identity
prepareAccessFunction[f_]:=f

$NeuralNetResourceAccessors={Get};

End[] (* End Private Context *)

EndPackage[]

SetAttributes[{},
   {ReadProtected, Protected}
];
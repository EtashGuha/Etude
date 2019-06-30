(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["DataResource`"]

Begin["`Private`"] (* Begin Private Context *) 


ResourceSystemClient`Private`repositoryresourceaccess[$DataResourceType,args___]:=dataresourceaccess[args]
	
dataresourceaccess[fun0_,id_,info_,rest___]:=Block[{data, fun=prepareAccessFunction[fun0]},
    checkDataAccessFunction[fun,getContentElementAccessType[info], info];
    data=resourcedata[id];
    fun[data, rest]
]/;MemberQ[$DataResourceAccessors,fun0]

dataresourceaccess[fun_,id_,info_,___]:=readDataResourceSystem[id, info,Automatic,fun]/;attributeCheck[info,"Computable"]

prepareAccessFunction[Get]=Identity
prepareAccessFunction[f_]:=f


checkDataAccessFunction[fun:HoldPattern[Keys|Values|Select|Dataset],"Dataset", info_]:=If[
	!attributeCheck[info,"Computable"],
	Message[ResourceObject::accfun,fun];
	Throw[$Failed]
]
checkDataAccessFunction[fun:HoldPattern[TimeSeries|EventSeries|Keys|Values],"EventSeries", info_]:=If[
    !attributeCheck[info,"Computable"],
    Message[ResourceObject::accfun,fun];
    Throw[$Failed]
]

$DataResourceAccessors:=$DataResourceAccessors={Get,Keys,Values,Select,Dataset,TimeSeries,EventSeries};

End[] (* End Private Context *)

EndPackage[]

SetAttributes[{},
   {ReadProtected, Protected}
];
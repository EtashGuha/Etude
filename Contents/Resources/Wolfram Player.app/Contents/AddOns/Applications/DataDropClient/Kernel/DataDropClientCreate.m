(* Mathematica Package *)

(Unprotect[#]; Clear[#])& /@ {System`CreateDatabin,System`CopyDatabin}

BeginPackage["DataDropClient`"]
(* Exported symbols added here with SymbolName::usage *)  
System`CreateDatabin
System`CopyDatabin
DataDropClient`CreateClass

Begin["`Private`"] (* Begin Private Context *) 

(********************** Create Databins ************************)
System`CreateDatabin[args___]:=Catch[createDatabin[args]]

createDatabin[]:=createDatabin[Association[{}]]

createDatabin[name_String]:=CreateDatabin[Association["Name"->name]]

createDatabin[as_Association]:=If[KeyExistsQ[as,"Administrator"]||!ddCloudConnectedQ[],
	createdatabin[as],
	createdatabin[Join[Association[{"Administrator"->$WolframID}],as]]
]

createDatabin[opts___?OptionQ]:=createDatabin[Association[{opts}]]

createdatabin[args___]:=(
	ddCloudConnect[];
	If[ddCloudConnectedQ[],
		createdatabin[args]
		,
		Message[CreateDatabin::cloudc];
		$Failed
	]
)/;!ddCloudConnectedQ[]

createdatabin[as_Association]:=With[{res=apifun["Create",validateDatabinParams[as]]},
	createdatabin0[res,as]
]

validateDatabinParams[as0_]:=Block[{as=as0},
	If[KeyExistsQ[as,Permissions],
		as["Permissions"] = as[Permissions];
		as[Permissions] =.;
	];
	If[KeyExistsQ[as,"Interpretation"],
		as["Interpretation"]=checkInterpretation[as["Interpretation"]]
	];
	as
]

checkInterpretation[interp_]:=Replace[FormObject[interp], {FormObject[a_?AssociationQ] :> a, _ :> Throw[$Failed]}]


createdatabin0[res_, as_, err_:"Create"]:=Block[{id,name,shortid, tokens},
	If[KeyExistsQ[res,"UUID"],
		shortid="ShortID"/.res;
		id="UUID"/.res;
		If[shortid==="ShortID",shortid=id];
		name=Lookup[res,"Name",Lookup[as,"Name",None]];
		tokens="Tokens"/.res;
		If[tokens=!="Tokens",
			readauth[id]=First@Replace[Lookup[tokens,"ReadTokens",{None}], {Except[_List] :> {None},{}->{None}}, {0}];
			writeauth[id]=First@Replace[Lookup[tokens,"WriteTokens",{None}], {Except[_List] :> {None},{}->{None}}, {0}];
		];
		creationDate[id]=Quiet[Replace[
			timezoneconvert[Lookup[res,"CreationDate",DateObject[]]], Except[_DateObject] :> Missing[], {0}]
			];
		
		expirationDate[id]=Quiet[Replace[
			timezoneconvertCheck[Lookup[res,"ExpirationDate",None]], Except[_DateObject] :> None, {0}]
			];
		
		binroles[id,"Creator"]=Lookup[res,"Creator",None];
		binroles[id,"Owner"]=Lookup[res,"Owner",None];
		datadropclientcache[{"DatabinLatest", id}]=datadropclientcache[{"DatabinLatest", shortid}]={};
		$loadeddatabins=Join[$loadeddatabins,{id,shortid}];
		
		If[StringQ[id]&&id=!="ShortID",
			makedatabin[shortid, id, name,Lookup[res,"ShortURL",None] ],
			errorcheck[res,err]
		],
		errorcheck[res,err]
	]
]

createdatabin[class_String,rest___]:=createdatabin["Class"->class,rest]
createdatabin[class_,email_String,rest___]:=createdatabin[class,"Administrator"->email]
createdatabin[r:(_Rule...)]:=createdatabin[Association[{r}]]
createdatabin[{r:(_Rule...)}]:=createdatabin[Association[{r}]]

createDatabin[___]:=$Failed
createdatabin[___]:=$Failed
createdatabin0[___]:=$Failed

(********************** Copy Databins ************************)

System`CopyDatabin[args___]:=Catch[copyDatabin[args]]

copyDatabin[bin_Databin,___]:=(Message[CopyDatabin::limit];Throw[$Failed])/;limitedbinQ[bin]
copyDatabin[bin_Databin,rest___]:=copydatabin[getBinID[bin],rest]
copyDatabin[str_String,rest___]:=Block[{bin},
    Check[bin=Databin[str],
        (Message[CopyDatabin::nobin,str];Return[$Failed])
    ];
    copyDatabin[bin,rest]
]

copyDatabin[first_,___]:=(Message[CopyDatabin::nobin,first];$Failed)

copydatabin[id_String,rest___]:=Block[{opts, res, bin, newid, timestamp},
	opts=Switch[{rest},
		{},Association[],
		{_Rule...}|{{_Rule...}},Association[rest],
		{_Association},Association[rest],
		_,Throw[$Failed]
	];
	res=copydatabin0[id, opts];
	bin=createdatabin0[res, opts, "Copy"];
    newid=getBinID[bin];
    timestamp=Lookup[Lookup[res,"Information",Association[]],"LatestTimestamp",Missing[]];
    res["Timestamp"]=timestamp;
	cacheuploadresults[newid,res];
	bin
]

copydatabin0[id_, opts_Association]:=apifun["CopyDatabin",
	Join[KeyMap[# /. Permissions -> "Permissions" &, opts],Association["Bin"->id]]]

copyDatabin[___]:=$Failed
copydatabin[___]:=$Failed
copydatabin0[___]:=$Failed


(********************** Create Classes ************************)
DataDropClient`CreateClass[args___]:=createClass[args]

createClass[as_Association]:=createclass[as]

createClass[opts___?OptionQ]:=createClass[Association[opts]]

createClass[___]:=$Failed

createclass[as_]:=Block[{name, res},
	res=apifun["CreateClass",as];
	If[Quiet[KeyExistsQ[res,"ClassName"]],
		Lookup[res,"ClassName",$Failed],
		If[Quiet[KeyExistsQ[res,"Message"]],
			Message[DataDropClient`CreateClass::apierr,Lookup[res,"Message",""]];
			$Failed
		]
	]
	
]/;KeyExistsQ[as,"ClassName"]


End[] (* End Private Context *)

EndPackage[]

SetAttributes[{CreateDatabin, CopyDatabin},
   {ReadProtected, Protected}
];
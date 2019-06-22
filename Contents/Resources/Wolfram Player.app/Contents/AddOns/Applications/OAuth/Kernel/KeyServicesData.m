
Begin["KeyClient`"] 

KeyClient`$predefinedKeyservicelist;
KeyClient`KeyServicesData;
KeyClient`keycookeddata;
KeyClient`keysendmessage;
KeyClient`addKeyservice;

Begin["`Private`"]

(Unprotect[#]; Clear[#])& /@ {KeyClient`KeyServicesData,KeyClient`keycookeddata,KeyClient`keysendmessage,KeyClient`addKeyservice}
Unprotect[KeyClient`$predefinedKeyservicelist];

defaultKeyParams={
					(* defaults *)
					"ServiceName"       -> Null,
				    "Information"		-> "",
				    "URLFetchFun"		-> URLFetch
				    };

defaultKeyLabels=First/@defaultKeyParams;		    
(*************************** KeyServices *************************************)

KeyClient`$predefinedKeyservicelist={}

KeyClient`KeyServicesData[args___]:=With[{res=keyservices[args]},
	res/;!FailureQ[res]&&Head[res]=!=keyservicedata]

keyservices[name_,prop___]:=Module[{data=Once[keyservicedata[name]],availableproperties},
	availableproperties=First/@data;
	Switch[{prop},
		{},	data,
		{"Requests"},availableproperties,
		{"Authentication"},
			Thread[defaultKeyLabels->(defaultKeyLabels/.Join[data,defaultKeyParams])]
		,
		{Alternatives@@availableproperties},
		prop/.data,
		_,
		keyservicedata[name,prop]		
	]
]

keyservices[___]:=$Failed
KeyClient`KeyServicesData[___]:=$Failed

KeyClient`addKeyservice[name_, dir_: DirectoryName[System`Private`$InputFileName]]:=Module[{funs, file},
	Unprotect[KeyClient`$predefinedKeyservicelist,keyservicedata,KeyClient`keycookeddata,KeyClient`keysendmessage];
	KeyClient`$predefinedKeyservicelist=Union[Append[KeyClient`$predefinedKeyservicelist,name]];
	ServiceConnections`Private`appendservicelist[name,"APIKey"];
	file=FileNameJoin[{dir,name<>".m"}];
	If[!FileExistsQ[file],Return[$Failed]];
	funs=Get[file];
	keyservicedata[name,args___]:=funs[[1]][args];
	KeyClient`keycookeddata[name,args___]:=funs[[2]][args];
	KeyClient`keysendmessage[name,args___]:=funs[[3]][args];
	Protect[KeyClient`$predefinedKeyservicelist,keyservicedata,KeyClient`keycookeddata,KeyClient`keysendmessage];
]

Unprotect[KeyClient`keycookeddata,KeyClient`keysendmessage,keyservicedata];

(**** error handling ***)
keyservicedata[___]:=$Failed
KeyClient`keycookeddata[___]:=Throw[$Failed]
KeyClient`keysendmessage[___]:=Throw[$Failed]

SetAttributes[{KeyClient`$predefinedKeyservicelist,KeyClient`KeyServicesData,KeyClient`keycookeddata,KeyClient`keysendmessage,KeyClient`addKeyservice},{ReadProtected, Protected}];

End[];
End[];

{}
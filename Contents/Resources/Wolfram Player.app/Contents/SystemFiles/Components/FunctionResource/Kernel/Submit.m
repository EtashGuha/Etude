(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["FunctionResource`"]

Begin["`Private`"] (* Begin Private Context *) 

ResourceSystemClient`Private`repositoryValidateSubmission[$FunctionResourceTypes,id_, info_]:=validateSubmission[id, info]

validateSubmission[id_, info0_]:=Block[{location=info0["FunctionLocation"], content, info=KeyDrop[info0,"Function"]},
	Switch[location,
		_CloudObject,
		info,
		_String|_System`File|_LocalObject,
		If[FileExistsQ[location],
			info["FunctionLocation"]=location;
			info
			,
			If[!StringFreeQ[location,$CloudBase],
				info["FunctionLocation"]=CloudObject[location];
				info
				,
				Message[ResourceSubmit::invcon];Throw[$Failed]
			]
		],
		None,
		info
	]
]/;KeyExistsQ[info0,"FunctionLocation"]&&!MatchQ[info0["FunctionLocation"],None|"Inline"]

validateSubmission[id_, info_]:=info/;KeyExistsQ[info,"Function"]
validateSubmission[id_, info_]:=info/;KeyExistsQ[info,"DefinitionData"]
validateSubmission[id_, info_]:=
  With[ { newInfo = Quiet @ ResourceFunctionInformation @ inlineDefinitions @ info },
      newInfo /; ByteArrayQ @ newInfo[ "DefinitionData" ]
  ];

validateSubmission[___]:=$Failed

ResourceSystemClient`Private`repositorycompleteResourceSubmission[$FunctionResourceTypes, id_,as0_]:=Block[{as, funcs,deployed},
	as=DeleteMissing[AssociationMap[validateParameter["Function",#]&,as0]];
	If[TrueQ[ResourceSystemClient`$DeployResourceSubmissionContent],
		If[!StringQ[as["SymbolName"]]&&KeyExistsQ[as["Symbol"]],
			as["SymbolName"]=createSymbolNameString[as["Symbol"]];
		];
		as["Symbol"]=.;
		If[KeyExistsQ[as,"FunctionLocation"]&&!MatchQ[as["FunctionLocation"],None|"Inline"],
			as["FunctionLocation"]=ResourceSystemClient`Private`deployLargeDataSubmissionFile["Function",id,Automatic,as["FunctionLocation"]]
		];
		If[KeyExistsQ[as,"DefinitionData"],
			deployed=deployFunctionSubmissionContent[id, Automatic,as["DefinitionData"]];
			If[deployed=!={},
				as["Function"]=.;
				as["FunctionLocation"]=deployed[[2]]
			]
			,
			If[KeyExistsQ[as,"Function"],
				deployed=ResourceSystemClient`Private`deployLargeDataSubmissionContent["Function",id, Automatic,as["Function"]];
				If[deployed=!={},
					as["Function"]=.;
					as["FunctionLocation"]=deployed[[2]]
				]
			]
		]
	];	
	If[KeyExistsQ[as,"Documentation"],
		as["Documentation"] = withContext @ Compress[as["Documentation"]]
	];
	If[KeyExistsQ[as,"VerificationTests"],
		as["VerificationTests"] = withContext @ Compress[as["VerificationTests"]]
	];
	If[!(KeyExistsQ[as,"Function"]||KeyExistsQ[as,"FunctionLocation"]),Message[ResourceSubmit::noncont];Throw[$Failed]];
	as
]

ResourceSystemClient`Private`deployResourceSubmissionContentSizeLimit[$FunctionResourceTypes]=0;

importLocalBytes[ obj_LocalObject ] :=
  ByteArray @ Import @ obj;

importLocalBytes[ file_String ] :=
  ByteArray @ Import[ file, "Binary" ];

importLocalBytes[ File[ file_String ] ] :=
  importLocalBytes @ file;


ResourceSystemClient`Private`deployLargeDataSubmissionFile["Function",id_,_,local_]:=With[{bytes=importLocalBytes @ local},
	CloudExport[bytes, {"Byte", "Binary"}, ResourceSystemClient`Private`submissionContentLocation[id, Automatic]]
]


ResourceSystemClient`Private`deployLargeDataSubmissionContent["Function",id_, key_, s_]:=
	deployFunctionSubmissionContent[id,key,serializeWithDefinitions @ Unevaluated @ s]


deployFunctionSubmissionContent[id_, key_, ba_]:= (
    If[ ! StringQ @ ResourceSystemClient`Private`$ResourceSystemAdminUser,
        ResourceSystemClient`Private`$ResourceSystemAdminUser = "marketplace-admin@wolfram.com"
    ];
    key -> CloudExport[ Normal @ ba,
                        { "Byte", "Binary" },
                        ResourceSystemClient`Private`submissionContentLocation[ id, key ],
                        Permissions -> { ResourceSystemClient`Private`$ResourceSystemAdminUser -> "Read" }
           ]
)


createSymbolNameString[symb_]:=Replace[symb,HoldComplete[s_]:>ToString@Unevaluated@s]

ResourceSystemClient`Private`allowProvisionalSubmissionQ["Function"]=True;
End[] (* End Private Context *)

EndPackage[]

SetAttributes[{},
   {ReadProtected, Protected}
];

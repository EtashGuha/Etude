Package["ExternalStorage`"]

PackageExport[IPFSDownload]

IPFSDownload::inv="`1` is not a valid IPFSObject."
IPFSDownload::out="`1` is not a valid element nor filepath."
IPFSDownload::hash="FileHash of the downloaded file doesn't match the one in `1`"

$outputs={Automatic(*,"Body","BodyBytes","BodyByteArray"*)}

Options[IPFSDownload]= {"VerifyFileHash" -> False, "ImportContent"-> False}

IPFSDownload[eso_, opts:OptionsPattern[]]:= IPFSDownload[eso, Automatic, opts]

IPFSDownload[eso : HoldPattern[IPFSObject[_]], output_, opts:OptionsPattern[]]:= ipfsDisplay[eso, output, opts] /; MemberQ[$outputs, output]

IPFSDownload[eso : HoldPattern[IPFSObject[_]], output_, opts:OptionsPattern[]]:= ipfsWrite[eso, output, opts] /; Quiet[MatchQ[output,_?StringQ|_File]]

IPFSDownload[eso : HoldPattern[IPFSObject[_]], output_, opts:OptionsPattern[]]:= (Message[IPFSDownload::out, output]; Null /; False)

IPFSDownload[eso_, output_, opts:OptionsPattern[]]:= (Message[IPFSDownload::inv, HoldForm[eso]]; $Failed)

IPFSDownload[args___, OptionsPattern[]] := (ArgumentCountQ[IPFSDownload, Length[{args}], 1, 2]; Null /; False)

ipfsDisplay[eso: IPFSObject[details_?AssociationQ], output_ , opts:OptionsPattern[IPFSDownload]]:=  ipfsDisplay0[details,output,opts]

ipfsWrite[eso: IPFSObject[details_?AssociationQ], output_ , opts:OptionsPattern[IPFSDownload]]:= ipfsWrite0[details,output,opts]

ipfsDisplay0[details_, Automatic, OptionsPattern[IPFSDownload]] := Module[{request, res, hash},
	If[ KeyExistsQ[details, "Address"],
		request = HTTPRequest[$IPFSAddress <> "/api/v0/cat", <|"Query" -> {"arg" -> details["Address"]}|>];
		res = URLRead[request, "BodyByteArray"];
		If[ TrueQ[OptionValue["VerifyFileHash"]],
			If[ !KeyExistsQ[details, "FileHash"], Message[IPFSDownload::inv,HoldForm[details]]; Return[Missing["FileHash"]]];
			hash = Hash[res, "MD5", "HexString"];
			If[ !SameQ[details["FileHash"], hash],
				Message[IPFSDownload::hash,HoldForm[IPFSObject[details]]];
				Return[Failure["InvalidHash",<|details|>]]
			]
		];
		If[ TrueQ[OptionValue["ImportContent"]],
			ImportByteArray[res],
			res
		]
		,
        Message[IPFSDownload::inv,HoldForm[IPFSObject[details]]];
        Missing["Address"]
	]
]

ipfsWrite0[details_, output_, OptionsPattern[IPFSDownload]] := Module[{request, response, hash},
	If[ KeyExistsQ[details, "Address"],
		request = HTTPRequest[$IPFSAddress <> "/api/v0/cat", <|"Query" -> {"arg" -> details["Address"]}|>];
		response = URLRead[request, "BodyByteArray"];
		If[ TrueQ[OptionValue["VerifyFileHash"]],
			If[ !KeyExistsQ[details, "FileHash"], Message[IPFSDownload::inv,HoldForm[details]]; Return[Missing["FileHash"]]];
			hash = Hash[response, "MD5", "HexString"];
			If[ SameQ[details["FileHash"], hash],
				Export[output, response, "Byte"],
				Message[IPFSDownload::hash,HoldForm[IPFSObject[details]]];
				Failure["InvalidHash",<|details|>]
			],
			Export[output, response, "Byte"]
		]
		,
        Message[IPFSDownload::inv,HoldForm[IPFSObject[details]]];
        Missing["Address"]
	]
]

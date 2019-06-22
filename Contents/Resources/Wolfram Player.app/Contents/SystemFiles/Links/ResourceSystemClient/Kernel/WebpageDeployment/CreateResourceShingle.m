
Begin["DeployedResourceShingle`"]

DeployedResourceShingle`CreateResourceShingle

Begin["`Private`"]

DeployedResourceShingle`CreateResourceShingle[args___]:=Catch[createResourceObjectShingle[args]]

createResourceObjectShingle[info_Association,target_]:=With[{html=createResourceShingleHTML[info,target]},
	If[StringQ[html],
		html,
		$Failed
	]		
]

createResourceObjectShingle[str_String, target_]:=With[{ro=ResourceObject[str]},
	If[AssociationQ[ro[All]],
		createResourceObjectShingle[ro[All],target],
		HTTPResponse[ByteArray[ExportString[ro, "Base64"]], 
			Association["Headers" ->Association["Content-Disposition" -> "inline"], 
				"ContentType" -> "text/plain;charset=utf-8"], CharacterEncoding -> "UTF-8"]		
	]
]

createResourceObjectShingle[one_]:=createResourceObjectShingle[one, None]
createResourceObjectShingle[___]:=$Failed

createResourceShingleHTML[info_Association,target_]:=With[{rtype=info["ResourceType"]},
	createResourceShingleHTML[rtype,templateFile[rtype],info, target]
]

createResourceShingleHTML[rtype_,template_,info_,target_]:=createresourceShingleHTML[rtype,template, processShingleData[rtype,info, target]]/;FileExistsQ[template]

createResourceShingleHTML[___]:=$Failed

createresourceShingleHTML[rtype_,template_, info_]:=With[{as=Join[info,getWebResourceInfo[rtype]]},
	TemplateApply[FileTemplate[template], as]]

End[]

End[]


Begin["DeployedResourceShingle`"]

DeployedResourceShingle`CloudDeployResourceWithShingle

Begin["`Private`"]

DeployedResourceShingle`CloudDeployResourceWithShingle[args___]:=Catch[resourceObjectDeployWithShingle[args]]


resourceObjectDeployWithShingle[ ro_, rest___ ] :=
  With[ { id = ro[ "UUID" ] },

    (* trigger autoload of CloudObject paclet if it's not already loaded *)
    CloudObject;

    Block[ { deployargs = cloudDeployArgs[ro,rest], shingleresponse },

        shingleresponse =
          DeployedResourceShingle`CreateResourceShingle[ ro[ All ],
                                                         targetURL @ First @ deployargs
          ];

        Block[
            {
                System`ResourceObject,
                ResourceSystemClient`Private`$resourceObjectShingleDeploy = False,
                ResourceSystemClient`Private`createResourceShingle,
                CloudObject`Private`$IncludedContexts = { "ResourceSystemClient" },
                CloudObject`Private`normalizePermissionsSpec
            },

            CloudObject`Private`normalizePermissionsSpec[ ___ ] := { "Read",  "Execute" };


			ResourceSystemClient`Private`createResourceShingle[ _ ] := Sequence[ shingleresponse, "ContentType" -> "text/html" ];
            CloudDeploy[
                Unevaluated[
                	ResourceSystemClient`Private`createResourceShingle;
                    HoldPattern[ CloudObject`CloudDeployActiveQ @ ResourceObject @ KeyValuePattern[ "UUID" -> id ] ] = True;
                    If[ $CloudEvaluation,
                        HoldPattern[ GenerateHTTPResponse @ ResourceObject @ KeyValuePattern[ "UUID" -> id ] ] := HTTPResponse[ shingleresponse, "ContentType" -> "text/html;charset=utf-8" ];
                    ];
                    ro
                ],
                Sequence @@ deployargs
            ]
        ]
    ]
];


cloudDeployArgs[ro_,Automatic,opts___?OptionQ]:={defaultCloudObjectLocation[ro],opts}
cloudDeployArgs[ro_,opts___?OptionQ]:={CloudObject[],opts}
cloudDeployArgs[_,args___]:={args}

defaultCloudObjectLocation[HoldPattern[ResourceObject][info:KeyValuePattern[{"ResourceType"->rtype_String}]]]:=(
	ResourceSystemClient`Private`loadResourceType[rtype];
	defaultCloudObjectLocation[rtype,info]
	)

defaultCloudObjectLocation[rtype_,KeyValuePattern["ShortName"->sn_String]]:=defaultCloudObjectLocation[rtype,sn]
defaultCloudObjectLocation[rtype_,KeyValuePattern["Name"->name_String]]:=defaultCloudObjectLocation[rtype,name]

defaultCloudObjectLocation[rtype_,name_String]:=With[{safename=safeCloudName[name]},
	If[StringLength[safename]>0,
		CloudObject[FileNameJoin[{cloudDeployDefaultPath[rtype],safename},OperatingSystem->"Unix"]],
		CloudObject[]
	]
]

safeCloudName[name_]:=StringReplace[URLEncode[StringReplace[name, {Whitespace -> "-"}]],{("-"..)->"-",":"->""}]

defaultCloudObjectLocation[___]:=CloudObject[]

cloudDeployDefaultPath[rtype_]:=FileNameJoin[{"DeployedResources",rtype},OperatingSystem->"Unix"]

targetURL[str_String]:=CloudObject[str][[1]]
targetURL[obj:(_CloudObject|_URL)]:=First[obj]

resourceObjectDeployWithShingle[___]:=$Failed

End[]
End[]

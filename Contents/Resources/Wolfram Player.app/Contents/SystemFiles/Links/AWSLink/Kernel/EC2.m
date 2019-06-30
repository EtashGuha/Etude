(*******************************************************************************

EC2,functions

*******************************************************************************)

Package["AWSLink`"]

PackageImport["JLink`"]
PackageImport["GeneralUtilities`"]

(*----------------------------------------------------------------------------*)

jClassQ[obj_, refName_String]:= 
	If[
		JavaObjectQ[obj],
			classNameFromInstance[obj] === "com.amazonaws.services.ec2." <> refName
		,
		False,
		False
	];

validEC2Q[obj_] :=
	Switch[obj,
		_EC2Image, validEC2ImageQ[obj],
		_EC2Snapshot, validEC2SnapshotQ[obj],
		_EC2Volume, validEC2VolumeQ[obj],
		_EC2Instance, validEC2InstanceQ[obj],
		_EC2KeyPairInfo, validEC2KeyPairInfoQ[obj],
		_EC2SecurityGroup, validEC2SecurityGroupQ[obj],
		_EC2SpotInstanceRequest, validEC2SpotInstanceRequestQ[obj],
		_, False
	];

getValidEC2Client[obj_] :=
	If[ validEC2Q[obj],
		obj["EC2Client"]
		,
		ThrowFailure["awsunvalid", Head@obj, "EC2"]
	];

pickEC2Client[client_] := Switch[
	client
	,
	(cl_EC2Client /; validEC2ClientQ[cl]),
		client,
	Automatic,
		If[
			validEC2ClientQ[$AWSDefaultClients["ec2"]]
			,
			$AWSDefaultClients["ec2"]
			,
			ThrowFailure["awsnoclient", "EC2", "ec2"]
		],
	_,
		ThrowFailure["wrgclient", client]
];

(*----------------------------------------------------------------------------*)
PackageExport["EC2InstanceQ"]
PackageExport["GetEC2InstanceMetadata"]

(* According to: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/identify_ec2_instances.html*)
SetUsage[EC2InstanceQ,
"EC2InstanceQ[] inspects the System UUID / dmi to assert if your are running on an EC2 instance (it could be fooled)"
]

EC2InstanceQ[]:= Scope[
	Switch[ $OperatingSystem
	,"Windows"
		, res = StreamExecute["!wmic path win32_computersystemproduct get uuid"];
		res = MatchQ[res, str_?StringQ /; StringStartsQ[str, "ec2", IgnoreCase -> True]]
	,"Linux" | "Unix"
		, Which[
			FileExistsQ["/sys/hypervisor/uuid"]
			, res = ReadString["/sys/hypervisor/uuid"];
			 res = MatchQ[res, str_?StringQ /; StringStartsQ[str, "ec2", IgnoreCase -> True]]

			, FileExistsQ["/sys/class/dmi/id/board_vendor"]
			, res = ReadString["/sys/class/dmi/id/board_vendor"];
			res = MatchQ[res, str_String /; StringMatchQ[str, "Amazon EC2"~~__]]
			
			, True
			, False
		]
	,_
		, False
	]
];

(*----------------------------------------------------------------------------*)
(* According to: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-metadata.html#instancedata-data-categories *)
SetUsage[GetEC2InstanceMetadata,
"GetEC2InstanceMetadata[] returns the hierarchy of available metadata at root level <|\"Categories\" -> {cat$1/, $$}, \"Properties\" -> {prop$1, $$}|>.
GetEC2InstanceMetadata[prop$] returns the value associated to prop$.
GetEC2InstanceMetadata[All, path$] with path$ = \"cat$i/$$/cat$ijk\" returns hierarchy of available metadata at path$ level."
]

GetEC2InstanceMetadata::"ec2noresponding" = "Unable to connect. Either you are not on a Linux EC2 instance, or the operation timed out for other reasons.";
GetEC2InstanceMetadata::"ec2wrgmeta" = "`1` is not an available metadata. Run GetEC2InstanceMetadata[] to check available categories and properties.";
GetEC2InstanceMetadata::"ec2wrgstatus" = "Metadata query returned status code `1`";
GetEC2InstanceMetadata::"ec2wrgmetadata" = "metadata categorie  endding \".../\" detected. Calling GetEC2InstanceMetadata[All, `1`] instead."

GetEC2InstanceMetadata[metadata_String]:=
	CatchFailureAndMessage@If[StringEndsQ[metadata, "/"]
		, Message[GetEC2InstanceMetadata::"ec2wrgmetadata", metadata];
		GetEC2InstanceMetadata[All, metadata]
		, readEC2URL[metadata]
	];

GetEC2InstanceMetadata[] := CatchFailureAndMessage@GetEC2InstanceMetadata[All, ""];
GetEC2InstanceMetadata[All, path_String:""]:= Scope@CatchFailureAndMessage[
	res = readEC2URL[path];
	AssociationThread[
		{"Categories", "Properties"} ->
		Lookup[GroupBy[StringSplit[res, "\n"], StringEndsQ["/"]], {True, False}]
		]
]
readEC2URL[suffix_String] := Scope[
	(* sad to need EC2InstanceQ but first URLRead call outside EC2 takes up to 5 seconds *)
	url = URLBuild[{"http://169.254.169.254/latest/meta-data/", suffix}];
	res = If[EC2InstanceQ[]
		, URLRead[url, {"Body", "StatusCode"}, TimeConstraint->0.05]
		, ThrowFailure["ec2noresponding"]
	];
	Switch[res["StatusCode"]
		, 404
			, ThrowFailure["ec2wrgmeta", suffix]
		, 200
			, res["Body"]
		, _
			, ThrowFailure["ec2wrgstatus", res["StatusCode"]]
	]
]

(*----------------------------------------------------------------------------*)
PackageExport["EC2Connect"]
SetUsage[EC2Connect,
"EC2Connect[obj$, EC2Client[$$]] builds a new object obj$ with the provided EC2Client[$$]."
]
Options[EC2Connect] = {"Client" -> Automatic};
EC2Connect[ec2Client_EC2Client /; validEC2ClientQ[ec2Client], obj_, query:OptionsPattern[]] :=
	EC2Connect[obj, "Client" -> Replace[OptionValue["Client"], Automatic -> ec2Client], query];

Unprotect[EC2Client]
Unprotect[EC2Image]
Unprotect[EC2Snapshot]
Unprotect[EC2Volume]
Unprotect[EC2Instance]
Unprotect[EC2KeyPairInfo]
Unprotect[EC2SecurityGroup]
Unprotect[EC2SpotInstanceRequest]

(*----------------------------------------------------------------------------*)
(*
TODO: A Mechanism to save this accros sessions and create profiles for exchange
=> see http://docs.aws.amazon.com/autoscaling/latest/userguide/create-lc-with-instanceID.html
*)

(*----------------------------------------------------------------------------*)
(* ::Section:: *)
(* EC2Client *)
(*----------------------------------------------------------------------------*)

PackageExport["EC2Client"]
SetUsage[EC2Client,
	"EC2Client[$$] represents a connection to Elastic Cloud Computing services.
Clients are created using AWSClientConnect[\"ec2\", $$].
related: AWSClientConnect."]

(* Makes a nicely formatted display box *)
DefineCustomBoxes[EC2Client, 
	e:EC2Client[EC2ClientContent_Association] :> Block[{},
	BoxForm`ArrangeSummaryBox[
		EC2Client, e, None, 
		{
			BoxForm`SummaryItem[{"ID: ", e["Id"]}],
			BoxForm`SummaryItem[{"Region: ", e["Region"]}]
		},
		{
			(*
			BoxForm`SummaryItem[{"State: ", jec2ClientGetState[EC2ClientContent["JClient"]]}],
			*)
		},
		StandardForm
	]
]];

EC2Client /: Format[HoldPattern[e:EC2Client[EC2ClientContent_Association]], OutputForm] :=
	SequenceForm[EC2Client][BoxForm`SurroundWithAngleBrackets[e["Id"]]];

validEC2ClientQ[EC2Client[jHandles_Association]] :=
	jClassQ[jHandles["JClient"], "AmazonEC2Client"];
validEC2ClientQ[anythingElse___] := False;

ValidEC2Q[ec2client_EC2Client] :=
	validEC2ClientQ[ec2client];

jEC2ClientQ[obj_] := jClassQ[obj, "AmazonEC2Client"];

jec2ClientGetId[jclient_/;jEC2ClientQ[jclient]]:=
	StringDelete[Last@StringSplit[safeLibraryInvoke @ jclient@toString[], "."], "AmazonEC2Client@"];
jec2ClientGetId[anythingelse___] := $Failed;

jec2ClientGetRegion[jclient_ /; jEC2ClientQ[jclient]]:=
	JavaBlock[safeLibraryInvoke @ jclient@describeAvailabilityZones[]@getAvailabilityZones[]@get[0]@getRegionName[]];
jec2ClientGetRegion[anythingelse___] := $Failed;

jec2ClientGetState[jclient_]:=
	Deploy @ Graphics[{If[jEC2ClientQ[jclient], Green,Red, Red], Disk[]}, ImageSize -> 10];

EC2Client[Content_Association][value_String] := 
	CatchFailureAsMessage @ If[ validEC2ClientQ[EC2Client[Content]]
		,
		Switch[ value,
			"Properties",
				{"Id", "Region"},
			"JClient",
				Content[value],
			"Id",
				jec2ClientGetId[Content["JClient"]],
			"Region",
				jec2ClientGetRegion[Content["JClient"]],
			_,
				ThrowFailure["awsnoprop", value, EC2Client]
		]
		,
		ThrowFailure["awsbrokenclient", EC2Client, "EC2"]
	];

Protect[EC2Client]

(*----------------------------------------------------------------------------*)

PackageScope["EC2ClientConnect"]
SetUsage[EC2ClientConnect,
"EC2ClientConnect[] create to EC2 using the default credential provider chain."
]
Options[EC2ClientConnect] = Options[AWSClientConnect];
EC2ClientConnect[OptionsPattern[]] :=
	Module[
		{client},
		JavaBlock[
				safeLibraryInvoke[
					LoadJavaClass["com.amazonaws.services.ec2.AmazonEC2ClientBuilder"];
				];

				client = safeLibraryInvoke[
					com`amazonaws`services`ec2`AmazonEC2ClientBuilder`standard[]
				];
				
				If[ OptionValue["Region"] =!= Automatic,
					client = safeLibraryInvoke[client@withRegion[OptionValue["Region"]]]
				];
				client = safeLibraryInvoke[client@build[]];
				
				KeepJavaObject[client];
		];
		
		EC2Client[<|"JClient" -> client|>]
	]
FailInOtherCases[EC2ClientConnect];


(*----------------------------------------------------------------------------*)
(* ::Section:: *)
(* EC2Image *)
(*----------------------------------------------------------------------------*)

PackageExport["EC2Image"]
SetUsage[EC2Image,
	"EC2Image[$$] represent an EC2 image
EC2Image[EC2Client[$$], imageId] builds the corresponding EC2Image[$$]
Dataset[EC2Image[$$]] queries all the properties of EC2Image[$$]
EC2Image[$$][\"property\"] queries the value associated to property (listable)"
]
(* Makes a nicely formatted display box *)
DefineCustomBoxes[EC2Image,
	e:EC2Image[EC2ImageContent_Association] :> Block[{},
	BoxForm`ArrangeSummaryBox[
		EC2Image, e, None, 
		{
			BoxForm`SummaryItem[{"Id: ", e["ImageId"]}]
		},
		{
			(*BoxForm`SummaryItem[{"Name: ", e["Name"]}],
			BoxForm`SummaryItem[{"Description: ", e["Description"]}]*)
		},
		StandardForm
	]
]];

EC2Image /: Format[HoldPattern[e:EC2Image[EC2ImageContent_Association]], OutputForm] := 
	SequenceForm[EC2Image][BoxForm`SurroundWithAngleBrackets[e["ImageId"]]];

EC2Image[EC2ImageContent_Association]["ImageId"] := EC2ImageContent["ImageId"];
EC2Image[EC2ImageContent_Association]["EC2Client"] := EC2ImageContent["EC2Client"];
(*ToDo -> cache for unmutable values...*)

validEC2ImageQ[EC2Image[jHandles_Association]] :=
	validEC2ClientQ[jHandles["EC2Client"]];
validEC2ImageQ[anythingElse___] := False;

ValidEC2Q[obj_EC2Image] :=
	validEC2ImageQ[obj];

EC2Image /: AWSGetInformation[obj:EC2Image[Content_Association]] := 
	CatchFailureAsMessage[AWSGetInformation, ec2ImageMetadata[obj]];
EC2Image /: Dataset[img:EC2Image[Content_Association]] :=
	CatchFailureAsMessage[EC2Image, Dataset[ec2ImageMetadata[img]]];

EC2Image[EC2ImageContent_Association][props : (_String | {__String})] :=
	CatchFailureAsMessage[
		EC2Image,
		getCurrentProperties[
			"EC2",
			EC2Image[EC2ImageContent],
			validEC2ImageQ,
			ec2ImageMetadata,
			Keys@EC2ImageTemplate,
			props
		]
	];

buildEC2Image[client_EC2Client /; validEC2ClientQ[client], jobj_ /; jClassQ[jobj, "model.Image"]] :=
	EC2Image[
		<|
		"EC2Client" -> client,
		"ImageId" -> #ImageId,
		"ImageName" -> #ImageName
		|>
	]& @
	getPropertiesFromTemplate[
		EC2ImageTemplate,
		jobj
	];

EC2Image[ec2Client_EC2Client /; validEC2ClientQ[ec2Client], imageId_String]:=
	EC2Image[<|"EC2Client" -> ec2Client, "ImageName" -> Missing[], "ImageId" -> imageId|>];

EC2Connect[obj_EC2Image, opt : OptionsPattern[]] := 
	CatchFailureAsMessage @ EC2Image[pickEC2Client[OptionValue["Client"]], obj["ImageId"]];

Protect[EC2Image]
(*----------------------------------------------------------------------------*)
ec2ImageMetadata[img_EC2Image] :=
	ec2ImageMetadata[getValidEC2Client[img], img["ImageId"]];
ec2ImageMetadata[client_EC2Client /; validEC2ClientQ[client], imageId_String] := JavaBlock[
	getPropertiesFromTemplate[
		EC2ImageTemplate,
		safeLibraryInvoke[
			client["JClient"] @
				describeImages[
					JavaNew["com.amazonaws.services.ec2.model.DescribeImagesRequest"]
						@withImageIds[{imageId}]
				]@getImages[]@get[0]
		]
	]
];

(*----------------------------------------------------------------------------*)
PackageExport["EC2DescribeImages"]
SetUsage[EC2DescribeImages,
"EC2DescribeImages[EC2Client] returns an Association of all available images and their attributes.
EC2DescribeImages[$$, \"ExecutableUsers\"-> {user$1,$$}] filters on image users with executable priviledges (default None).
EC2DescribeImages[EC2Client, \"Filters\" -> <|filterName$1 ->{acceptedValue$1, $$}|>] perform a filtered request.
see: http://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeImages.html for all filters names and descriptions.
EC2DescribeImages[$$, MaxItems-> num] returns up to num results per batch (default 1000).
EC2DescribeImages[$$, \"Owners\"-> {owner$1,$$}] filters on image owner:\
 self (sender of the request), or an AWS owner alias (valid values are\
 amazon | aws-marketplace | microsoft) (default {\"self\"})."
]

Options[EC2DescribeImages] = 
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic,
		"ExecutableUsers" -> Automatic,
		"Filters" -> Automatic,
		"ImageId" -> Automatic,
		MaxItems -> 1000,
		"Owners" -> {"self"}
	};
EC2DescribeImages[ec2Client_/;validEC2ClientQ[ec2Client], query:OptionsPattern[]] :=
	EC2DescribeImages["Client" -> Replace[OptionValue["Client"], Automatic -> ec2Client], query];
EC2DescribeImages[query:OptionsPattern[]] :=
	CatchFailureAsMessage @ JavaBlock @ Module[
		{
			ec2Client = pickEC2Client[OptionValue["Client"]],
			imageList, imageReqResult, request, i},
		
		request =
			iEC2ImageDescribeSetRequest[
				OptionValue["ExecutableUsers"],
				OptionValue["Filters"],
				OptionValue["ImageId"],
				OptionValue["Owners"]
			];
		
		If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[ec2Client["JClient"], request],
			imageReqResult = safeLibraryInvoke @ ec2Client["JClient"]@describeImages[request];
			imageList = safeLibraryInvoke @ imageReqResult@getImages[];
			Table[getPropertiesFromTemplate[EC2ImageTemplate, imageList@get[i]], {i, 0, Min[imageList@size[], Replace[OptionValue[MaxItems], Automatic -> Infinity]]-1}]
		]
	];
FailInOtherCases[EC2DescribeImages];

(*----------------------------------------------------------------------------*)
PackageExport["EC2ImageCollection"]
SetUsage[EC2ImageCollection, "EC2ImageCollection[$$] is an iterator over EC2Image[$$] objects.
Get one object:
ec2image = AWSCollectionRead[EC2ImageCollection[$$]].
Get up to num objects:
{ec2image$1, ec2image$2, $$} = AWSCollectionReadList[EC2ImageCollection[$$], num]
related: EC2Images"];

DefineCustomBoxes[EC2ImageCollection, 
	e:EC2ImageCollection[iterator_] :> Block[{},
	BoxForm`ArrangeSummaryBox[
		EC2ImageCollection, e, None, 
		{
			BoxForm`SummaryItem[{"EC2ImageCollection:", ""}]
		},
		{
		},
		StandardForm
	]
]];

EC2ImageCollection /: Format[HoldPattern[EC2ImageCollection[___]], OutputForm] :=
	SequenceForm[EC2ImageCollection][BoxForm`SurroundWithAngleBrackets[""]]

PackageExport["EC2Images"]
SetUsage[EC2Images,
"EC2Images[EC2Client[$$]] returns a EC2ImageCollection[$$] iterating over the\
 images EC2Image[$$] available to you.
EC2Images[EC2Client, \"Filters\" -> <|\"filterName\"->{acceptedValue$1, $$}|>]\
 performs a filtered request.
EC2Images[EC2Client, \"Filters\" -> <|\"tag\[Colon]key\"-> { value$1, $$}, $$|>]\
 returns all the images tagged with the pair key -> value$i.
EC2Images[EC2Client, \"Owners\" -> {\"owner$1\", $$}] returns only images\
 owned by owners listed (use \"self\" for yourself).
see: http://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeImages.html\
 for descriptions and all filters available"
]

Options[EC2Images] = 
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic,
		"Filters" -> Automatic,
		"Owners" -> {"self"} (*{"self"}*)
	};

EC2Images[client_EC2Client /; validEC2ClientQ[client], query:OptionsPattern[]] :=
	EC2Images["Client" -> Replace[OptionValue["Client"], Automatic -> client], query];
EC2Images[query:OptionsPattern[]] :=
	CatchFailureAsMessage @ Module[
	{
		filters = OptionValue["Filters"],
		owners = OptionValue["Owners"],
		client = pickEC2Client[OptionValue["Client"]]
	},
	If[ Replace[OptionValue["DryRun"], Automatic -> False],
		AWSDryRun[client["JClient"], iEC2ImageDescribeSetRequest[Automatic, filters, Automatic, owners]],
		EC2ImageCollection[
			NewIterator[
				EC2ImageCollection,
				{
					nextToken = ""
				},
				CatchFailureAsMessage @ If[nextToken === Null,
					IteratorExhausted,
					JavaBlock @ Block[
						{req, res, objListToStrm, cachedvalues},
						req = iEC2ImageDescribeSetRequest[Automatic, filters, Automatic, owners];
						res = safeLibraryInvoke @ client["JClient"]@describeImages[req];
						ReleaseJavaObject[req];
						nextToken = Null;
						objListToStrm = safeLibraryInvoke @ res@getImages[];
						cachedvalues =
							Map[
								EC2Image[<|"EC2Client" -> client, "ImageId" -> #|>]&,
								MapJavaMethod[
									objListToStrm,
									"com.amazonaws.services.ec2.model.Image",
									"getImageId",
									"java.lang.String"
								]
							];
						DelegateIterator[ListIterator[cachedvalues]]
					]
				]
			]
		]
	]
];
FailInOtherCases[EC2Images];

iEC2ImageDescribeSetRequest[executableUsers_, filters_, imageIds_, owners_]:=
	JavaBlock @ Module[
		{
			request,
			ownermod = Replace[owners, All -> Automatic]
		},
		
		request =
			safeLibraryInvoke[JavaNew, "com.amazonaws.services.ec2.model.DescribeImagesRequest"];
		
		If[ executableUsers =!= Automatic,
			request = safeLibraryInvoke @
				request
					@withExecutableUsers[executableUsers];
		];
		
		If[ filters =!= Automatic,
			request = safeLibraryInvoke @
				request
					@withFilters[KeyValueMap[createFilter, filters]]
		];
		
		If[ imageIds =!= Automatic,
			request = safeLibraryInvoke @
				request
					@withImagesIds[imageIds];
		];
		
		If[ ownermod =!= Automatic,
			request = safeLibraryInvoke @
				request
					@withOwners[ownermod];
		];
		
		request@toString[];
		
		request
	];

(*----------------------------------------------------------------------------*)
PackageExport["EC2CreateImage"]
SetUsage[EC2CreateImage,
"EC2CreateImage[EC2Instance[$$], imageName] creates a EC2Image[$$] of the EC2Instance[$$] named imageName.
EC2CreateImage[$$, \"Description\" -> description] sets the description field of the newly created image - default: \"(generated by AWSLink)\".
EC2CreateImage[$$, \"Reboot\" -> False] forbids instance shut-down.
Warning: by default the method attempts to shut down and reboot the instance before creating the image.
"
]

Options[EC2CreateImage] = 
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic,
		"Description" -> "(generated by AWSLink)",
		"Reboot" -> True
		(*
		BlockDeviceMapping -> 
		*)
	};
EC2CreateImage[ec2Instance_/;validEC2InstanceQ[ec2Instance], imgName_String, query:OptionsPattern[]] :=
	EC2CreateImage[ec2Instance["InstanceId"], imgName, "Client" -> Replace[OptionValue["Client"], Automatic -> ec2Instance["EC2Client"]], query];
EC2CreateImage[instanceId_String, imgName_String, query:OptionsPattern[]] :=
	CatchFailureAsMessage @ JavaBlock @ Module[
		{
			ec2client = pickEC2Client[OptionValue["Client"]],
			req, res
		},
		req = safeLibraryInvoke @
			JavaNew["com.amazonaws.services.ec2.model.CreateImageRequest"]
								@withName[imgName]
								@withInstanceId[instanceId]
								@withDescription[OptionValue["Description"]]
								@withNoReboot[MakeJavaObject[Not[OptionValue["Reboot"]]]];
		If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[ec2client["JClient"], req],
			res = 
				safeLibraryInvoke @ 
					ec2client["JClient"]
						@createImage[req]@getImageId[];
			EC2Image[<|"EC2Client"-> ec2client, "ImageId"->res, "ImageName" -> imgName|>]
		]
	]

FailInOtherCases[EC2CreateImage];

(*----------------------------------------------------------------------------*)
(* ::Section:: *)
(* EC2Snapshot *)
(*----------------------------------------------------------------------------*)
PackageExport["EC2Snapshot"]
SetUsage[EC2Snapshot,
	"EC2Snapshot[$$] represent an EC2 snapshot.
EC2Snapshot[EC2Client[$$], snapshotId] builds the corresponding EC2Snapshot[$$].
Dataset[EC2Snapshot[$$]] queries all the properties of EC2Snapshot[$$].
EC2Snapshot[$$][\"property\"] queries the value associated to property (listable)."
]
DefineCustomBoxes[EC2Snapshot, 
	e:EC2Snapshot[EC2SnapshotContent_Association] :> Block[{},
		BoxForm`ArrangeSummaryBox[
			EC2Snapshot, e, None, 
			{
				BoxForm`SummaryItem[{"Id: ", EC2SnapshotContent["SnapshotId"]}]
			},
			{
				(*BoxForm`SummaryItem[{"State: ", content$[["Properties", "State"]]}],
				BoxForm`SummaryItem[{"VolumeSize: ", content$[["Properties", "VolumeSize"]]}],
				BoxForm`SummaryItem[{"Tags: ", Grid[Normal /@ content$[["Properties", "Tags"]] ]}],
				BoxForm`SummaryItem[{"Owner: ", content$[["Properties", "OwnerAlias"]]}]*)
			},
			StandardForm
		]
	]
];

EC2Snapshot /: Format[HoldPattern[EC2Snapshot[EC2SnapshotContent_Association]], OutputForm] :=
	SequenceForm[EC2Snapshot][BoxForm`SurroundWithAngleBrackets[EC2SnapshotContent["SnapshotId"]]]

EC2Snapshot[Content_Association]["EC2Client"] := Content["EC2Client"];
EC2Snapshot[Content_Association]["SnapshotId"] := Content["SnapshotId"];

validEC2SnapshotQ[EC2Snapshot[Content_Association]] :=
	validEC2ClientQ[Content["EC2Client"]];
validEC2SnapshotQ[anythingElse___] := False;

ValidEC2Q[obj_EC2Snapshot] :=
	validEC2SnapshotQ[obj];
	
EC2Snapshot /: AWSGetInformation[obj:EC2Snapshot[Content_Association]] :=
	CatchFailureAsMessage[AWSGetInformation, ec2SnapshotMetadata[obj]];
EC2Snapshot /: Dataset[obj:EC2Snapshot[Content_Association]] :=
	CatchFailureAsMessage[
			EC2Snapshot,
			Dataset[ec2SnapshotMetadata[obj]]
	];

EC2Snapshot[Content_Association][props : (_String | {__String})] :=
	CatchFailureAsMessage[
		EC2Snapshot,
		getCurrentProperties[
			"EC2",
			EC2Snapshot[Content],
			validEC2SnapshotQ,
			ec2SnapshotMetadata,
			Keys@EC2SnapshotTemplate,
			props
		]
	];

buildEC2Snapshot[client_EC2Client /; validEC2ClientQ[client], jobj_ /; jClassQ[jobj, "model.Snapshot"]] :=
	EC2Snapshot[
		<|
		"EC2Client" -> client,
		"SnapshotId" -> #SnapshotId
		|>
	]& @
	getPropertiesFromTemplate[
		EC2SnapshotTemplate,
		jobj
	];

EC2Snapshot[ec2Client_EC2Client /; validEC2ClientQ[ec2Client], id_String]:=
	EC2Snapshot[<|"EC2Client" -> ec2Client, "SnapshotId" -> id|>];
	
EC2Connect[obj_EC2Snapshot, opt : OptionsPattern[]] := 
	CatchFailureAsMessage @ EC2Snapshot[pickEC2Client[OptionValue["Client"]], obj["SnapshotId"]];

Protect[EC2Snapshot]
(*----------------------------------------------------------------------------*)
ec2SnapshotMetadata[snapshot_EC2Snapshot] :=
	ec2SnapshotMetadata[getValidEC2Client[snapshot], snapshot["SnapshotId"]];
ec2SnapshotMetadata[client_EC2Client /; validEC2ClientQ[client], id_String] := JavaBlock[
	getPropertiesFromTemplate[
		EC2SnapshotTemplate,
		safeLibraryInvoke @ 
			client["JClient"]@describeSnapshots[
				JavaNew["com.amazonaws.services.ec2.model.DescribeSnapshotsRequest"]
					@withSnapshotIds[{id}]
			]@getSnapshots[]@get[0]
	]
];

(*----------------------------------------------------------------------------*)
PackageExport["EC2DescribeSnapshots"]
SetUsage[EC2DescribeSnapshots,
"EC2DescribeSnapshots[EC2Client] returns an Association with keys:
 - \"Batch\": a list of available snapshots and their attributes.
 - \"NextToken\": a token to chains calls and query the next batch of results.
EC2DescribeSnapshots[EC2Client, \"Filters\" -> <|filterName$1 ->{acceptedValue$1, $$}|>] perform a filtered request.
see: http://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeSnapshots.html for all filters names and descriptions.
EC2DescribeSnapshots[$$, \"SnapshotIds\"-> {id$1,$$}] filters on instanceIds (default All).
EC2DescribeSnapshots[$$, MaxItems-> num] returns up to num results per batch (default 1000).
EC2DescribeSnapshots[$$, \"NextToken\"-> token] chains call to returns the next batch of results."
]

Options[EC2DescribeSnapshots] = 
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic,
		"Filters" -> Automatic,
		"OwnerIds" -> Automatic,
		"RestorableByUserIds" -> Automatic,
		"SnapshotIds" -> Automatic,
		MaxItems -> 1000,
		"NextToken" -> ""
	};
EC2DescribeSnapshots[ec2Client_/;validEC2ClientQ[ec2Client], query:OptionsPattern[]] :=
	EC2DescribeSnapshots["Client" -> Replace[OptionValue["Client"], Automatic -> ec2Client], query];
EC2DescribeSnapshots[query:OptionsPattern[]] :=
	CatchFailureAsMessage @ JavaBlock @ Module[
		{
			ec2Client = pickEC2Client[OptionValue["Client"]],
			snapshotList, requestResult, request, i
		},
		
		request =
			iEC2DescribeSnapshotsSetRequest[
				OptionValue["Filters"],
				OptionValue[MaxItems],
				OptionValue["NextToken"],
				OptionValue["OwnerIds"],
				OptionValue["RestorableByUserIds"],
				OptionValue["SnapshotIds"]
			];
		If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[ec2Client["JClient"], request],
		requestResult = safeLibraryInvoke @ ec2Client["JClient"]@describeSnapshots[request];
		snapshotList = requestResult@getSnapshots[];
		<|
		"Batch" -> Table[getPropertiesFromTemplate[EC2SnapshotTemplate, snapshotList@get[i]], {i, 0, snapshotList@size[]-1}],
		"NextToken" -> requestResult@getNextToken[]
		|>
		]
	];
FailInOtherCases[EC2DescribeSnapshots];

(*----------------------------------------------------------------------------*)
PackageExport["EC2SnapshotCollection"]
SetUsage[EC2SnapshotCollection, "EC2SnapshotCollection[$$] is an iterator over EC2Snapshot[$$] objects.
Get one object:
ec2snapshot = AWSCollectionRead[EC2SnapshotCollection[$$]].
Get up to num objects:
{ec2snapshot$1, ec2snapshot$2, $$} = AWSCollectionReadList[EC2SnapshotCollection[$$], num]
related: EC2Snapshots"];

DefineCustomBoxes[EC2SnapshotCollection, 
	e:EC2SnapshotCollection[iterator_] :> Block[{},
	BoxForm`ArrangeSummaryBox[
		EC2SnapshotCollection, e, None, 
		{
			BoxForm`SummaryItem[{"EC2SnapshotCollection:", ""}]
		},
		{
		},
		StandardForm
	]
]];

EC2SnapshotCollection /: Format[HoldPattern[EC2SnapshotCollection[___]], OutputForm] :=
	SequenceForm[EC2SnapshotCollection][BoxForm`SurroundWithAngleBrackets[""]];

PackageExport["EC2Snapshots"]
SetUsage[EC2Snapshots,
"EC2Snapshots[EC2Client[$$]] returns a EC2SnapshotCollection[$$] iterating over the\
 snapshots EC2Snapshot[$$] available to you.
EC2Snapshots[EC2Client, \"Filters\" -> <|\"filterName\"->{acceptedValue$1, $$}|>]\
 performs a filtered request.
EC2Snapshots[EC2Client, \"Filters\" -> <|\"tag\[Colon]key\"-> { value$1, $$}, $$|>]\
 returns all the snapshots tagged with the pair key -> value$i.
EC2Snapshots[EC2Client, \"Owners\" -> {\"owner$1\", $$}] returns only snapshots\
 owned by owners listed.
see: http://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeSnapshots.html\
 for descriptions and all filters available"
]

Options[EC2Snapshots] = 
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic,
		"Filters" -> Automatic,
		"Owners" -> Automatic,
		"RestorableByUserIds" -> Automatic,
		"SnapshotIds" -> Automatic
		(*MaxItems -> Automatic (*not an interesting option for an iterator*)*)
	};
EC2Snapshots[client_EC2Client /; validEC2ClientQ[client], query:OptionsPattern[]] :=
	EC2Snapshots["Client" -> Replace[OptionValue["Client"], Automatic -> client], query];
EC2Snapshots[query:OptionsPattern[]] :=
	CatchFailureAsMessage @ Module[
	{
		client = pickEC2Client[OptionValue["Client"]],
		filters = OptionValue["Filters"],
		ownerIds = OptionValue["Owners"],
		restorableByUserIds = OptionValue["RestorableByUserIds"]
	},
	
	If[ Replace[OptionValue["DryRun"], Automatic -> False],
		AWSDryRun[client["JClient"], iEC2DescribeSnapshotsSetRequest[filters, Automatic, nextToken, ownerIds, restorableByUserIds, Automatic]],
		EC2SnapshotCollection[
			NewIterator[
				EC2SnapshotCollection,
				{
					nextToken = ""
				},
				CatchFailureAsMessage @ If[nextToken === Null,
					IteratorExhausted,
					JavaBlock @ Block[
						{req, res, objListToStrm, cachedvalues},
						req = iEC2DescribeSnapshotsSetRequest[filters, Automatic, nextToken, ownerIds, restorableByUserIds, Automatic];
						res = safeLibraryInvoke @ client["JClient"]@describeSnapshots[req];
						ReleaseJavaObject[req];
						nextToken = safeLibraryInvoke @ res@getNextToken[];
						objListToStrm = safeLibraryInvoke @ res@getSnapshots[];
						cachedvalues =
							Map[
								EC2Snapshot[<|"EC2Client" -> client, "SnapshotId" -> #|>]&,
								MapJavaMethod[
									objListToStrm,
									"com.amazonaws.services.ec2.model.Snapshot",
									"getSnapshotId",
									"java.lang.String"
								]
							];
						DelegateIterator[ListIterator[cachedvalues]]
					]
				]
			]
		]
	]
];
FailInOtherCases[EC2Snapshots];

iEC2DescribeSnapshotsSetRequest[filters_, maxItems_, nextToken_, ownerIds_, restorableByUserIds_, snapshotIds_]:=
Module[
	{
		request
	},
	
	request = safeLibraryInvoke @
		JavaNew["com.amazonaws.services.ec2.model.DescribeSnapshotsRequest"]
			@withNextToken[nextToken];
	
	If[ filters =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withFilters[KeyValueMap[createFilter, filters]]
	];
	
	If[ maxItems =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withMaxResults[MakeJavaObject[maxItems]]
	];
	
	If[ ownerIds =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withOwnerIds[ownerIds]
	];
	
	If[ restorableByUserIds =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withRestorableByUserIds[restorableByUserIds]
	];
	
	If[ snapshotIds =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withSnapshotIds[snapshotIds]
	];
	
	request
];

(*----------------------------------------------------------------------------*)
PackageExport["EC2CreateSnapshot"]
SetUsage[EC2CreateSnapshot,
"EC2CreateSnapshot[EC2Volume[$$]] creates a EC2Snapshot[$$] of the EC2Volume[$$].
EC2CreateSnapshot[$$, \"Description\" -> desc] sets the description field of the\
 created snapshot to desc (default: \"(generated by AWSLink)\").
"
]

Options[EC2CreateSnapshot] = 
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic,
		"Description" -> "(generated by AWSLink)"
	};
	

EC2CreateSnapshot[ec2Volume_/;validEC2VolumeQ[ec2Volume], query:OptionsPattern[]] :=
	EC2CreateSnapshot[ec2Volume["VolumeId"], "Client" -> Replace[OptionValue["Client"], Automatic -> ec2Volume["EC2Client"]], query];

EC2CreateSnapshot[volumeId_String, query:OptionsPattern[]] :=
	CatchFailureAsMessage @ JavaBlock @ Module[
		{
			client = pickEC2Client[OptionValue["Client"]],
			req, jres
		},
		req = safeLibraryInvoque @
			JavaNew["com.amazonaws.services.ec2.model.CreateSnapshotRequest"]
						@withVolumeId[volumeId]
						@withDescription[OptionValue["Description"]];
		If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[client["JClient"], req],
			jres = safeLibraryInvoke @
				client["JClient"]@createSnapshot[req]@getSnapshot[];
			buildEC2Snapshot[client, jres]
		]
	];

FailInOtherCases[EC2CreateSnapshot];


(*----------------------------------------------------------------------------*)
PackageExport["EC2SnapshotDelete"]
SetUsage[EC2SnapshotDelete,
"EC2SnapshotDelete[EC2Snapshot[$$]] deletes an EC2Snapshot[$$]."
]

Options[EC2SnapshotDelete] =
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic
	};

EC2SnapshotDelete[ec2Snapshot_/;validEC2SnapshotQ[ec2Snapshot], query:OptionsPattern[]] :=
	EC2SnapshotDelete[ec2Snapshot["SnapshotId"], "Client" -> Replace[OptionValue["Client"], Automatic -> ec2Snapshot["EC2Client"]], query];

EC2SnapshotDelete[snapshotId_, query:OptionsPattern[]] :=
	JavaBlock @ Module[
		{
			client = pickEC2Client[OptionValue["Client"]],
			req, jres
		},
		
		req = safeLibraryInvoke @
			JavaNew["com.amazonaws.services.ec2.model.DeleteSnapshotRequest"]
				@withSnapshotId[snapshotId];

		If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[client["JClient"], req],
			jres = safeLibraryInvoke @
				client["JClient"]@deleteSnapshot[req];
			<||>
		]
	];

FailInOtherCases[EC2SnapshotDelete]

(*----------------------------------------------------------------------------*)
PackageExport["EC2SnapshotCopy"]
SetUsage[EC2SnapshotCopy,
"EC2SnapshotCopy[EC2Snapshot[$$], destination] copies a point-in-time EC2Snapshot[$$]\
 and stores it in Amazon S3 in region destination.\
 You can copy the snapshot within the same region or from one region to another.\
 You can use the snapshot to create EBS volumes or Amazon Machine Images (AMIs).
"
]

Options[EC2SnapshotCopy] = 
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic,
		"Description" -> "(generated by AWSLink)"
	};
EC2SnapshotCopy[ec2Snapshot_/;validEC2SnapshotQ[ec2Snapshot], destination_String, query:OptionsPattern[]] :=
	EC2SnapshotCopy[
		ec2Snapshot["SnapshotId"],
		destination,
		"Client" -> Replace[OptionValue["Client"], Automatic -> ec2Snapshot["EC2Client"]],
		query
	];

EC2SnapshotCopy[snapshotId_, dest_, query:OptionsPattern[]] :=
	CatchFailureAsMessage @ JavaBlock @ Module[
		{
			client = pickEC2Client[OptionValue["Client"]],
			req, jres
		},
		
		req = safeLibraryInvoke @
			JavaNew["com.amazonaws.services.ec2.model.CopySnapshotRequest"]
				@withSourceSnapshotId[snapshotId]
				@withDestinationRegion[dest]
				@withDescription[OptionValue["Description"]];
		
		If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[client["JClient"], req],
			jres = safeLibraryInvoke @
				client["JClient"]@createSnapshot[req]@getSnapshot[];
			buildEC2Snapshot[client, jres]
		]
	];

FailInOtherCases[EC2SnapshotCopy]

(*----------------------------------------------------------------------------*)
(* ::Section:: *)
(* EC2Volume *)
(*----------------------------------------------------------------------------*)
PackageExport["EC2Volume"]
SetUsage[EC2Volume,
	"EC2Volume[$$] represent an EC2 volume.
EC2Volume[EC2Client[$$], volumeId] builds the corresponding EC2Volume[$$].
Dataset[EC2Volume[$$]] queries all the properties of EC2Volume[$$].
EC2Volume[$$][\"property\"] queries the value associated to property (listable)."
];

(* Makes a nicely formatted display box *)
DefineCustomBoxes[EC2Volume, 
	e:EC2Volume[content_Association] :> Block[{},
		BoxForm`ArrangeSummaryBox[
			EC2Volume, e, None, 
			{
				BoxForm`SummaryItem[{"Id: ", e["VolumeId"]}]
			},
			{
			},
			StandardForm
		]
	]
];

EC2Volume /: Format[HoldPattern[e:EC2Volume[content_Association]], OutputForm] :=
	SequenceForm[EC2Volume][BoxForm`SurroundWithAngleBrackets[e["VolumeId"]]]

EC2Volume[content_Association]["VolumeId"] := content["VolumeId"];
EC2Volume[content_Association]["EC2Client"] := content["EC2Client"];
(*ToDo -> cache for unmutable values...*)

validEC2VolumeQ[EC2Volume[jHandles_Association]] :=
	validEC2ClientQ[jHandles["EC2Client"]];
validEC2VolumeQ[anythingElse___] := False;

ValidEC2Q[obj_EC2Volume] :=
	validEC2VolumeQ[obj];

EC2Volume /: AWSGetInformation[obj:EC2Volume[Content_Association]] :=
	CatchFailureAsMessage[AWSGetInformation, ec2VolumeMetadata[obj]];
EC2Volume /: Dataset[obj:EC2Volume[Content_Association]] :=
	CatchFailureAsMessage[
			EC2Volume,
			Dataset[ec2VolumeMetadata[obj]]
	];

EC2Volume[content_Association][props : (_String | {__String})] :=
	CatchFailureAsMessage[
		EC2Volume,
		getCurrentProperties[
			"EC2",
			EC2Volume[content],
			validEC2VolumeQ,
			ec2VolumeMetadata,
			Keys@EC2VolumeTemplate,
			props
		]
	];

buildEC2Volume[client_EC2Client /; validEC2ClientQ[client], jobj_ /; jClassQ[jobj, "model.Volume"]] :=
	EC2Volume[
		<|
		"EC2Client" -> client,
		"VolumeId" -> #VolumeId
		|>
	]& @
	getPropertiesFromTemplate[
		EC2VolumeTemplate,
		jobj
	];

EC2Volume[ec2Client_EC2Client /; validEC2ClientQ[ec2Client], volumeId_String]:=
	EC2Volume[<|"EC2Client" -> ec2Client, "VolumeId" -> volumeId|>];
	
EC2Connect[obj_EC2Volume, opt : OptionsPattern[]] := 
	CatchFailureAsMessage @ EC2Volume[pickEC2Client[OptionValue["Client"]], obj["VolumeId"]]

Protect[EC2Volume]
(*----------------------------------------------------------------------------*)
ec2VolumeMetadata[img_EC2Volume] :=
	ec2VolumeMetadata[getValidEC2Client[img], img["VolumeId"]];
ec2VolumeMetadata[client_EC2Client /; validEC2ClientQ[client], volumeId_String] := JavaBlock[
	getPropertiesFromTemplate[
		EC2VolumeTemplate,
		safeLibraryInvoke[
			client["JClient"]
				@describeVolumes[
					JavaNew["com.amazonaws.services.ec2.model.DescribeVolumesRequest"]
						@withVolumeIds[{volumeId}]
				]@getVolumes[]@get[0]
		]
	]
];

(*----------------------------------------------------------------------------*)
PackageExport["EC2DescribeVolumes"]
SetUsage[EC2DescribeVolumes,
"EC2DescribeVolumes[EC2Client] returns an Association with keys:
 - \"Batch\": a list of available volumes and their attributes.
 - \"NextToken\": a token to chains calls and query the next batch of results.
EC2DescribeVolumes[EC2Client, \"Filters\" -> <|filterName$1 ->{acceptedValue$1, $$}|>] perform a filtered request.
see: http://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeVolumes.html for all filters names and descriptions.
EC2DescribeVolumes[$$, \"VolumeIds\"-> {id$1,$$}] filters on instanceIds (default All).
EC2DescribeVolumes[$$, MaxItems-> num] returns up to num results per batch per batch (default 1000).
EC2DescribeVolumes[$$, \"NextToken\"-> token] chains call to returns the next batch of results."
]

Options[EC2DescribeVolumes] = 
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic,
		"Filters" -> Automatic,
		"VolumeIds" -> Automatic,
		MaxItems -> Automatic,
		"NextToken" -> ""
	};
EC2DescribeVolumes[ec2Client_/;validEC2ClientQ[ec2Client], query:OptionsPattern[]] :=
	EC2DescribeVolumes[
		"Client" -> Replace[OptionValue["Client"], Automatic -> ec2Client],
		query
	];
EC2DescribeVolumes[query:OptionsPattern[]] :=
	CatchFailureAsMessage @ JavaBlock @ Module[
		{
			ec2Client = pickEC2Client[OptionValue["Client"]],
			volumeList, requestResult, request, i},
		
		request =
			iEC2DescribeVolumesSetRequest[
				OptionValue["Filters"],
				OptionValue[MaxItems],
				OptionValue["NextToken"],
				OptionValue["VolumeIds"]
			];
		
		If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[ec2Client["JClient"], request],
			requestResult = safeLibraryInvoke @ ec2Client["JClient"]@describeVolumes[request];
			volumeList = requestResult@getVolumes[];
			<|
				"Batch" -> Table[getPropertiesFromTemplate[EC2VolumeTemplate, volumeList@get[i]], {i, 0, volumeList@size[]-1}],
				"NextToken" -> requestResult@getNextToken[]
			|>
		]
	];
FailInOtherCases[EC2DescribeVolumes];

(*----------------------------------------------------------------------------*)
PackageExport["EC2VolumeCollection"]
SetUsage[EC2VolumeCollection, "EC2VolumeCollection[$$] is an iterator over EC2Volume[$$] objects.
Get one object:
ec2volume = AWSCollectionRead[EC2VolumeCollection[$$]].
Get up to num objects:
{ec2volume$1, ec2volume$2, $$} = AWSCollectionReadList[EC2VolumeCollection[$$], num]
related: EC2Volumes"];

DefineCustomBoxes[EC2VolumeCollection, 
	e:EC2VolumeCollection[iterator_] :> Block[{},
	BoxForm`ArrangeSummaryBox[
		EC2VolumeCollection, e, None, 
		{
			BoxForm`SummaryItem[{"EC2VolumeCollection:", ""}]
		},
		{
		},
		StandardForm
	]
]];

EC2VolumeCollection /: Format[HoldPattern[EC2VolumeCollection[___]], OutputForm] :=
	SequenceForm[EC2VolumeCollection][BoxForm`SurroundWithAngleBrackets[""]]

(*----------------------------------------------------------------------------*)
PackageExport["EC2Volumes"]
SetUsage[EC2Volumes,
"EC2Volumes[EC2Client[$$]] returns a EC2VolumeCollection[$$] iterating over the\
 volumes EC2Volume[$$] available to you.
EC2Volumes[EC2Client, \"Filters\" -> <|\"filterName\"->{acceptedValue$1, $$}|>]\
 performs a filtered request.
EC2Volumes[EC2Client, \"Filters\" -> <|\"tag\[Colon]key\"-> { value$1, $$}, $$|>]\
 returns all the solumes tagged with the pair key -> value$i.
see: http://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeVolumes.html\
 for descriptions and all filters available"
]

Options[EC2Volumes] = 
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic,
		"Filters" -> Automatic,
		"VolumeIds" -> Automatic
		(*MaxItems -> 1000*)
	};
EC2Volumes[client_EC2Client /; validEC2ClientQ[client], query:OptionsPattern[]] :=
	EC2Volumes["Client" -> Replace[OptionValue["Client"], Automatic -> client], query];
EC2Volumes[query:OptionsPattern[]] :=
	CatchFailureAsMessage @ Module[
	{
		client = pickEC2Client[OptionValue["Client"]],
		filters = OptionValue["Filters"],
		volumeIds = OptionValue["VolumeIds"]
	},
	
	If[ Replace[OptionValue["DryRun"], Automatic -> False],
		AWSDryRun[client["JClient"], iEC2DescribeVolumesSetRequest[filters, Automatic, "", volumeIds]],
		EC2VolumeCollection[
			NewIterator[
				EC2VolumeCollection,
				{
					nextToken = ""
				},
				CatchFailureAsMessage @ If[nextToken === Null,
					IteratorExhausted,
					JavaBlock @ Block[
						{req, res, objListToStrm, cachedvalues},
						req = iEC2DescribeVolumesSetRequest[filters, Automatic, nextToken, volumeIds];
						res = safeLibraryInvoke @ client["JClient"]@describeVolumes[req];
						ReleaseJavaObject[req];
						nextToken = safeLibraryInvoke @ res@getNextToken[];
						objListToStrm = safeLibraryInvoke @ res@getVolumes[];
						cachedvalues =
							Map[
								EC2Volume[<|"EC2Client" -> client, "VolumeId" -> #|>]&,
								MapJavaMethod[
									objListToStrm,
									"com.amazonaws.services.ec2.model.Volume",
									"getVolumeId",
									"java.lang.String"
								]
							];
						DelegateIterator[ListIterator[cachedvalues]]
					]
				]
			]
		]
	]
];
FailInOtherCases[EC2Volumes];

iEC2DescribeVolumesSetRequest[filters_, maxResults_, nextToken_, volumeIds_]:=
Module[
	{
		request
	},
	
	request = safeLibraryInvoke @
		JavaNew["com.amazonaws.services.ec2.model.DescribeVolumesRequest"]
			@withNextToken[nextToken];
	
	If[ filters =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withFilters[KeyValueMap[createFilter, filters]]
	];
	
	If[ maxResults =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withMaxResults[MakeJavaObject[maxResults]]
	];
	
	If[ volumeIds =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withVolumeIds[volumeIds]
	];
	
	request
];

(*----------------------------------------------------------------------------*)
PackageExport["EC2VolumeDelete"]
SetUsage[EC2VolumeDelete,
"EC2VolumeDelete[EC2Volume[$$]] request the EC2Volume[$$] deletion."
]

EC2VolumeDelete[volume_EC2Volume/;validEC2VolumeQ[volume], opt:OptionsPattern[]] :=
	CatchFailureAsMessage @
		EC2VolumeDelete[
			volume["VolumeId"],
			"Client" -> Replace[OptionValue["Client"], Automatic -> volume["EC2Client"]],
			opt
		]
Options[EC2VolumeDelete] =
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic
	}
EC2VolumeDelete[volumeId_String, op:OptionsPattern[]] :=
	CatchFailureAsMessage @ JavaBlock @ Module[
		{req, res, client = pickEC2Client[OptionValue["Client"]]},
		
		req = JavaNew[
			"com.amazonaws.services.ec2.model.DeleteVolumeRequest",
			volumeId
			];

		 If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[client["JClient"], req],
			res = safeLibraryInvoke @
				client["JClient"]@deleteVolume[req];
			<||>
		]
	];

(*----------------------------------------------------------------------------*)
PackageExport["EC2VolumeAttach"]
SetUsage[EC2VolumeAttach,
"EC2VolumeAttach[EC2Volume[$$], EC2Instance[$$], deviceName] attaches the EC2Volume[$$]\
 to the EC2Instance[$$] using the device name deviceName.
deviceName: Linux EBS recommanded /dev/sd[f-p] or /dev/sdf[1-6]
see:
 - https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/device_naming.html
 - https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html
"
]

Options[EC2VolumeAttach] =
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic
	};
EC2VolumeAttach[ec2Volume_/;validEC2VolumeQ[ec2Volume], ec2Instance_/;validEC2InstanceQ[ec2Instance], deviceName_String, query:OptionsPattern[]] :=
	CatchFailureAsMessage @
		EC2VolumeAttach[
			ec2Volume["VolumeId"],
			ec2Instance["InstanceId"],
			deviceName,
			"Client" -> Replace[OptionValue["Client"], Automatic -> ec2Volume["EC2Client"]],
			query
		];

EC2VolumeAttach[volumeId_String, instanceId_String, deviceName_String, op:OptionsPattern[]] :=
	CatchFailureAsMessage @ JavaBlock @ Module[
		{
			client = pickEC2Client[OptionValue["Client"]],
			request, res
		},
		
		request =
			JavaNew[
				"com.amazonaws.services.ec2.model.AttachVolumeRequest",
				volumeId,
				instanceId,
				deviceName
			];

		If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[client["JClient"], request],
		res = safeLibraryInvoke @ 
			client["JClient"]@attachVolume[request];
		EC2Volume[<|"EC2Client" -> client, "VolumeId" -> volumeId|>]
		]
	];

(* ::Section:: *)
(* EC2Instance *)
(*----------------------------------------------------------------------------*)
PackageExport["EC2Instance"]
SetUsage[EC2Instance,
	"EC2Instance[$$] represent an EC2 instance.
EC2Instance[EC2Client[$$], instanceId] builds the corresponding EC2Instance[$$].
Dataset[EC2Instance[$$]] queries all the properties of EC2Instance[$$].
EC2Instance[$$][\"property\"] queries the value associated to property (listable)."
];

(* Makes a nicely formatted display box *)
DefineCustomBoxes[EC2Instance,
	e:EC2Instance[content_Association] :> Block[{},
		BoxForm`ArrangeSummaryBox[
			EC2Instance, e, None, 
			{
				BoxForm`SummaryItem[{"Id: ", e["InstanceId"]}]
			},
			{
			},
			StandardForm
		]
	]
];

EC2Instance /: Format[HoldPattern[e:EC2Instance[content_Association]], OutputForm] :=
	SequenceForm[EC2Instance][BoxForm`SurroundWithAngleBrackets[e["InstanceId"]]];

EC2Instance[content_Association]["InstanceId"] := content["InstanceId"];
EC2Instance[content_Association]["EC2Client"] := content["EC2Client"];
EC2Instance[content_Association]["Volumes"] :=
	Association @@
	Map[
		#DeviceName -> EC2Volume[<|"EC2Client"-> getValidEC2Client[EC2Instance[content]], "VolumeId" -> #Ebs[["VolumeId"]]|>]&,
		EC2Instance[content]["BlockDeviceMappings"]
	];
(*ToDo -> cache for unmutable values...*)

validEC2InstanceQ[EC2Instance[jHandles_Association]] :=
	validEC2ClientQ[jHandles["EC2Client"]];
validEC2InstanceQ[anythingElse___] := False;

ValidEC2Q[obj_EC2Instance] :=
	validEC2InstanceQ[obj];

EC2Instance /: AWSGetInformation[obj:EC2Instance[Content_Association]] :=
	CatchFailureAsMessage[AWSGetInformation, ec2InstanceMetadata[obj]];
EC2Instance /: Dataset[obj:EC2Instance[Content_Association]] :=
	CatchFailureAsMessage[
			EC2Instance,
			Dataset[ec2InstanceMetadata[obj]]
	];

EC2Instance[content_Association][props : (_String | {__String})] :=
	CatchFailureAsMessage[
		EC2Instance,
		getCurrentProperties[
			"EC2",
			EC2Instance[content],
			validEC2InstanceQ,
			ec2InstanceMetadata,
			Keys@EC2InstanceTemplate,
			props
		]
	];

buildEC2Instance[client_EC2Client /; validEC2ClientQ[client], jobj_ /; jClassQ[jobj, "model.Instance"]] :=
	EC2Instance[
		<|
		"EC2Client" -> client,
		"Instanced" -> #InstanceId
		|>
	]& @
	getPropertiesFromTemplate[
		EC2InstanceTemplate,
		jobj
	];

EC2Instance[ec2Client_EC2Client /; validEC2ClientQ[ec2Client], instanceId_String]:=
	EC2Instance[<|"EC2Client" -> ec2Client, "InstanceId" -> instanceId|>];

ec2InstanceMetadata[img_EC2Instance] :=
	ec2InstanceMetadata[getValidEC2Client[img], img["InstanceId"]];
ec2InstanceMetadata[client_EC2Client /; validEC2ClientQ[client], instanceId_String] := JavaBlock[
	getPropertiesFromTemplate[
		EC2InstanceTemplate,
		safeLibraryInvoke[
			client["JClient"]
				@describeInstances[
					JavaNew["com.amazonaws.services.ec2.model.DescribeInstancesRequest"]
						@withInstanceIds[{instanceId}]
				]@getReservations[]@get[0]@getInstances[]@get[0]
		]
	]
];
	
EC2Connect[obj_EC2Instance, opt : OptionsPattern[]] :=
	CatchFailureAsMessage @ EC2Instance[pickEC2Client[OptionValue["Client"]], obj["InstanceId"]]

Protect[EC2Instance]
(*----------------------------------------------------------------------------*)
PackageExport["EC2DescribeInstances"]
SetUsage[EC2DescribeInstances,
"EC2DescribeInstances[EC2Client] returns an Association with keys:
 - \"Batch\": a list of available instances and their attributes grouped by reservation.
 - \"NextToken\": a token to chains calls and query the next batch of results.
EC2DescribeInstances[EC2Client, \"Filters\" -> <|filterName$1 ->{acceptedValue$1, $$}|>] perform a filtered request.
see: http://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeInstances.html for all filters names and descriptions.
EC2DescribeInstances[$$, \"InstanceIds\"-> {id$1,$$}] filters on instanceIds (default All).
EC2DescribeInstances[$$, MaxItems-> num] returns up to num results per batch (default 1000).
EC2DescribeInstances[$$, \"NextToken\"-> token] chains call to returns the next batch of results."
]

Options[EC2DescribeInstances] = 
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic,
		"Filters" -> Automatic,
		"InstanceIds" -> Automatic,
		"NextToken" -> "",
		MaxItems -> Automatic
	};
EC2DescribeInstances[ec2Client_/;validEC2ClientQ[ec2Client], query:OptionsPattern[]] :=
	EC2DescribeInstances[
		"Client" -> Replace[OptionValue["Client"], Automatic -> ec2Client],
		query
	];
EC2DescribeInstances[query:OptionsPattern[]] :=
	CatchFailureAsMessage @ JavaBlock @ Module[
		{
			ec2Client = pickEC2Client[OptionValue["Client"]],
			reservationList, requestResult, request, i},
		
		request =
			iEC2DescribeInstancesSetRequest[
				OptionValue["Filters"],
				OptionValue["InstanceIds"],
				OptionValue[MaxItems],
				OptionValue["NextToken"]
			];
		If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[ec2Client["JClient"], request],
			requestResult = safeLibraryInvoke @ ec2Client["JClient"]@describeInstances[request];
			reservationList = requestResult@getReservations[];
			<|
			"Batch" -> Table[getPropertiesFromTemplate[EC2ReservationTemplate, reservationList@get[i]], {i, 0, reservationList@size[]-1}],
			"NextToken" -> requestResult@getNextToken[]
			|>
		]
	];
FailInOtherCases[EC2DescribeInstances];

(*----------------------------------------------------------------------------*)
PackageExport["EC2InstanceCollection"]
SetUsage[EC2InstanceCollection, "EC2InstanceCollection[$$] is an iterator over EC2Instance[$$] objects.
Get one object:
ec2instance = AWSCollectionRead[EC2InstanceCollection[$$]].
Get up to num objects:
{ec2instance$1, ec2instance$2, $$} = AWSCollectionReadList[EC2InstanceCollection[$$], num]
related: EC2Instances"];

DefineCustomBoxes[EC2InstanceCollection, 
	e:EC2InstanceCollection[iterator_] :> Block[{},
	BoxForm`ArrangeSummaryBox[
		EC2InstanceCollection, e, None, 
		{
			BoxForm`SummaryItem[{"EC2InstanceCollection:", ""}]
		},
		{
		},
		StandardForm
	]
]];

EC2InstanceCollection /: Format[HoldPattern[EC2InstanceCollection[___]], OutputForm] :=
	SequenceForm[EC2InstanceCollection][BoxForm`SurroundWithAngleBrackets[""]];

(*----------------------------------------------------------------------------*)
PackageExport["EC2Instances"]
SetUsage[EC2Instances,
"EC2Instances[EC2Client[$$]] returns a EC2InstanceCollection[$$] iterating over the\
 instances EC2Instance[$$] available to you.
EC2Instances[EC2Client, \"Filters\" -> <|\"filterName\"->{acceptedValue$1, $$}|>]\
 performs a filtered request.
EC2Instances[EC2Client, \"Filters\" -> <|\"tag\[Colon]key\"-> { value$1, $$}, $$|>]\
 returns all the solumes tagged with the pair key -> value$i.
see: http://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeInstances.html\
 for descriptions and all filters available"
]

Options[EC2Instances] = 
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic,
		"Filters" -> Automatic
		(*MaxItems -> Automatic (*not an interesting option for an iterator*)*)
	};
EC2Instances[client_EC2Client /; validEC2ClientQ[client], query:OptionsPattern[]] :=
	EC2Instances["Client" -> Replace[OptionValue["Client"], Automatic -> client], query];
EC2Instances[query:OptionsPattern[]] :=
	CatchFailureAsMessage @ Module[
	{
		client = pickEC2Client[OptionValue["Client"]],
		filters = OptionValue["Filters"]
	},
	
	If[ Replace[OptionValue["DryRun"], Automatic -> False],
		AWSDryRun[client["JClient"], iEC2DescribeInstancesSetRequest[filters, Automatic, Automatic, ""]],
		EC2InstanceCollection[
			NewIterator[
				EC2InstanceCollection,
				{
					nextToken = ""
				},
				CatchFailureAsMessage @ If[nextToken === Null,
					IteratorExhausted,
					JavaBlock @ Block[
						{req, res, reservationList, instanceList, cachedvalues, i},
						req = iEC2DescribeInstancesSetRequest[filters, Automatic, Automatic, nextToken];
						res = safeLibraryInvoke @ client["JClient"]@describeInstances[req];
						ReleaseJavaObject[req];
						nextToken = safeLibraryInvoke @ res@getNextToken[];
						reservationList = safeLibraryInvoke @ res@getReservations[];
						instanceList = JavaNew["java.util.ArrayList"];
						Do[
							safeLibraryInvoke[instanceList@addAll[reservationList@get[i]@getInstances[]]],
							{i, 0, reservationList@size[]-1}
						];
						cachedvalues =
							Map[
								EC2Instance[<|"EC2Client" -> client, "InstanceId" -> #|>]&,
								MapJavaMethod[
									instanceList,
									"com.amazonaws.services.ec2.model.Instance",
									"getInstanceId",
									"java.lang.String"
								]
							];
						DelegateIterator[ListIterator[cachedvalues]]
					]
				]
			]
		]
	]
];
FailInOtherCases[EC2Instances];

iEC2DescribeInstancesSetRequest[filters_, instanceIds_, maxResults_, nextToken_]:=
Module[
	{
		request
	},
	
	request = safeLibraryInvoke @
		JavaNew["com.amazonaws.services.ec2.model.DescribeInstancesRequest"]
			@withNextToken[nextToken];
	
	If[ filters =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withFilters[KeyValueMap[createFilter, filters]]
	];
	
	If[ instanceIds =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withInstanceIds[instanceIds]
	];
	
	If[ maxResults =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withMaxResults[MakeJavaObject[maxResults]]
	];
	
	request
];

(*----------------------------------------------------------------------------*)
PackageExport["EC2InstanceTerminate"]
SetUsage[EC2InstanceTerminate,
"EC2InstanceTerminate[EC2Instance[$$]] terminates the instance EC2Instance[$$]."
]

Options[EC2InstanceTerminate]=
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic
	};
EC2InstanceTerminate[instance_EC2Instance /; validEC2InstanceQ[instance], opt:OptionsPattern[]] :=
	CatchFailureAsMessage @
		EC2InstanceTerminate[
			instance["InstanceId"],
			"Client" -> Replace[OptionValue["Client"], Automatic -> instance["EC2Client"]],
			opt
		];

EC2InstanceTerminate[instanceId_String, opt:OptionsPattern[]] :=
	CatchFailureAsMessage @ JavaBlock @ Module[
		{
			req, jres,
			client = pickEC2Client[OptionValue["Client"]]
		},
		
		req =
			JavaNew["com.amazonaws.services.ec2.model.TerminateInstancesRequest"]
				@withInstanceIds[{instanceId}];
		
		If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[client["JClient"], req],
			jres = safeLibraryInvoke @ client["JClient"]@terminateInstances[req];
			<||>
		]
	];
FailInOtherCases[EC2InstanceTerminate]


(*----------------------------------------------------------------------------*)
PackageExport["EC2InstanceConsoleOutput"]
SetUsage[EC2InstanceConsoleOutput,
"EC2InstanceConsoleOutput[EC2Instance] returns the last consol output saved by\
 EC2 with the corresponding timestamp for debuggig."
]
Options[EC2InstanceConsoleOutput] =
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic
	};
EC2InstanceConsoleOutput[instance_EC2Instance/;validEC2InstanceQ[instance]] :=
	CatchFailureAsMessage @
		EC2InstanceConsoleOutput[
			instance["InstanceId"],
			"Client" -> Replace[OptionValue["Client"], Automatic -> instance["EC2Client"]]
		];

EC2InstanceConsoleOutput[instanceId_, op:OptionsPattern[]] :=
	JavaBlock @ Module[
		{req, res, client = pickEC2Client[OptionValue["Client"]]},
		
		req = JavaNew["com.amazonaws.services.ec2.model.GetConsoleOutputRequest", instanceId];
		
		If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[client["JClient"], req],
			res = safeLibraryInvoke @ client["JClient"]@getConsoleOutput[req];
			<|"Timestamp"-> #Timestamp, "ConsolOutput"-> #DecodedOutput|>& @ getPropertiesFromTemplate[EC2GetConsoleOutputResultTemplate, res]
		]
	];

FailInOtherCases[EC2InstanceConsoleOutput]

(*----------------------------------------------------------------------------*)
PackageExport["EC2InstanceNames"]
SetUsage[EC2InstanceNames,
"EC2InstanceNames[] returns the list of string names of all EC2 Instances.
Note: as AWS do not yet provides a solution to query those, this returns the\
 values available at build time."
]
EC2InstanceNames[] :=
	JavaBlock@Module[
		{instancesObjects},
		LoadJavaClass["com.amazonaws.services.ec2.model.InstanceType"];
		instancesObjects = com`amazonaws`services`ec2`model`InstanceType`values[];
		(#@toString[]) & /@ instancesObjects
	]

(*----------------------------------------------------------------------------*)
PackageExport["EC2ResourceTypes"]
SetUsage[EC2ResourceTypes,
"EC2ResourceTypes[] returns the list of string names of all EC2 Ressources Type."
]
Ec2RessourcesTypes[] :=
JavaBlock[
	Module[
		{RessourcesObjects},
		LoadJavaClass["com.amazonaws.services.ec2.model.ResourceType"];
		RessourcesObjects = com`amazonaws`services`ec2`model`ResourceType`values[];
		(#@toString[]) & /@ RessourcesObjects
	]
];


(*----------------------------------------------------------------------------*)
(* ::Section:: *)
(* EC2KeyPairInfo *)
(*----------------------------------------------------------------------------*)
PackageExport["EC2KeyPairInfo"]
SetUsage[EC2KeyPairInfo,
	"EC2KeyPairInfo[$$] represents available information on an EC2 keys pair.
EC2KeyPairInfo[EC2Client[$$], keypairname] builds the corresponding EC2KeyPairInfo[$$].
Dataset[EC2KeyPairInfo[$$]] queries all the properties of EC2KeyPairInfo[$$].
EC2KeyPairInfo[$$][\"property\"] queries the value associated to property (listable)."
]

DefineCustomBoxes[EC2KeyPairInfo, 
	e:EC2KeyPairInfo[content_Association] :> Block[{},
	BoxForm`ArrangeSummaryBox[
		EC2KeyPairInfo, e, None, 
		{
			BoxForm`SummaryItem[{"Name: ", e["KeyName"]}]
		},
		{
			(*BoxForm`SummaryItem[{"Fingerprint: ", EC2KeyPairInfoContent[["Properties", "KeyFingerprint"]]}]*)
		},
		StandardForm
	]
]];

EC2KeyPairInfo /: Format[HoldPattern[e:EC2KeyPairInfo[content_Association]], OutputForm] :=
	SequenceForm[EC2KeyPairInfo][BoxForm`SurroundWithAngleBrackets[e["KeyName"]]];

EC2KeyPairInfo[content_Association]["KeyName"] := content["KeyName"];
EC2KeyPairInfo[content_Association]["EC2Client"] := content["EC2Client"];
(*ToDo -> cache for unmutable values...*)

validEC2KeyPairInfoQ[EC2KeyPairInfo[jHandles_Association]] :=
	validEC2ClientQ[jHandles["EC2Client"]];
validEC2KeyPairInfoQ[anythingElse___] := False;

ValidEC2Q[obj_EC2KeyPairInfo] :=
	validEC2KeyPairInfoQ[obj];

EC2KeyPairInfo /: AWSGetInformation[obj:EC2KeyPairInfo[Content_Association]] :=
	CatchFailureAsMessage[AWSGetInformation, ec2KeyPairInfoMetadata[obj]];
EC2KeyPairInfo /: Dataset[obj:EC2KeyPairInfo[Content_Association]] :=
	CatchFailureAsMessage[
			EC2KeyPairInfo,
			Dataset[ec2KeyPairInfoMetadata[obj]]
	];

EC2KeyPairInfo[content_Association][props : (_String | {__String})] :=
	CatchFailureAsMessage[
		EC2KeyPairInfo,
		getCurrentProperties[
			"EC2",
			EC2KeyPairInfo[content],
			validEC2KeyPairInfoQ,
			ec2KeyPairInfoMetadata,
			Keys@EC2KeyPairInfoTemplate,
			props
		]
	];

buildEC2KeyPairInfo[client_EC2Client /; validEC2ClientQ[client], jobj_ /; jClassQ[jobj, "model.KeyPairInfo"]] :=
	EC2KeyPairInfo[
		<|
		"EC2Client" -> client,
		"KeyName" -> #KeyName
		|>
	]& @
	getPropertiesFromTemplate[
		EC2KeyPairInfoTemplate,
		jobj
	];

EC2KeyPairInfo[ec2Client_EC2Client /; validEC2ClientQ[ec2Client], keyPairName_String]:=
	EC2KeyPairInfo[<|"EC2Client" -> ec2Client, "KeyName" -> keyPairName|>];
	
EC2Connect[obj_EC2KeyPairInfo, opt : OptionsPattern[]] := 
	CatchFailureAsMessage @EC2KeyPairInfo[pickEC2Client[OptionValue["Client"]], obj["KeyName"]];

Protect[EC2KeyPairInfo]

(*----------------------------------------------------------------------------*)
ec2KeyPairInfoMetadata[img_EC2KeyPairInfo] :=
	ec2KeyPairInfoMetadata[getValidEC2Client[img], img["KeyName"]];
ec2KeyPairInfoMetadata[client_EC2Client /; validEC2ClientQ[client], keyPairName_String] := JavaBlock[
	getPropertiesFromTemplate[
		EC2KeyPairInfoTemplate,
		safeLibraryInvoke[
			client["JClient"]
				@describeKeyPairs[
					JavaNew["com.amazonaws.services.ec2.model.DescribeKeyPairsRequest"]
						@withKeyNames[{keyPairName}]
				]@getKeyPairs[]@get[0]
		]
	]
];

(*----------------------------------------------------------------------------*)
PackageExport["EC2DescribeKeyPairs"]
SetUsage[EC2DescribeKeyPairs,
"EC2DescribeKeyPairs[EC2Client] returns a list of Associations describing all\
 available key pairs and their attributes.
EC2DescribeKeyPairs[EC2Client, \"Filters\" -> <|filterName$1 ->{acceptedValue$1, $$}|>] perform a filtered request.
see: http://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeKeyPairs.html for all filters names and descriptions.
EC2DescribeKeyPairs[$$, \"KeyNames\"-> {name$1,$$}] filters on key names (default All).
EC2DescribeKeyPairs[$$, MaxItems-> num] returns up to num results per batch (default 1000)."
]

Options[EC2DescribeKeyPairs] = 
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic,
		"Filters" -> Automatic,
		"KeyNames" -> Automatic,
		MaxItems -> 1000
	};
EC2DescribeKeyPairs[ec2Client_/;validEC2ClientQ[ec2Client], query:OptionsPattern[]] :=
	EC2DescribeKeyPairs[
		"Client" -> Replace[OptionValue["Client"], Automatic -> ec2Client],
		query
	];
EC2DescribeKeyPairs[query:OptionsPattern[]] := 
	CatchFailureAsMessage @JavaBlock @ Module[
		{
			ec2Client = pickEC2Client[OptionValue["Client"]],
			keyPairList, requestResult, request, i
		},
		
		request =
			iEC2DescribeKeyPairsSetRequest[
				OptionValue["Filters"],
				OptionValue["KeyNames"]
			];
		If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[ec2Client["JClient"], request],
			requestResult = safeLibraryInvoke @ ec2Client["JClient"]@describeKeyPairs[request];
			keyPairList = requestResult@getKeyPairs[];
			Table[getPropertiesFromTemplate[EC2KeyPairInfoTemplate, keyPairList@get[i]], {i, 0, Min[keyPairList@size[], Replace[OptionValue[MaxItems], Automatic -> Infinity]]-1}]
		]
	];
FailInOtherCases[EC2DescribeKeyPairs];

(*----------------------------------------------------------------------------*)
PackageExport["EC2KeyPairInfoCollection"]
SetUsage[EC2KeyPairInfoCollection, "EC2KeyPairInfoCollection[$$] is an iterator over EC2KeyPairInfo[$$] objects.
Get one object:
ec2keyPair = AWSCollectionRead[EC2KeyPairInfoCollection[$$]].
Get up to num objects:
{ec2keyPair$1, ec2keyPair$2, $$} = AWSCollectionReadList[EC2KeyPairInfoCollection[$$], num]
related: EC2KeyPairs"];

DefineCustomBoxes[EC2KeyPairInfoCollection, 
	e:EC2KeyPairInfoCollection[iterator_] :> Block[{},
	BoxForm`ArrangeSummaryBox[
		EC2KeyPairInfoCollection, e, None, 
		{
			BoxForm`SummaryItem[{"EC2KeyPairInfoCollection:", ""}]
		},
		{
		},
		StandardForm
	]
]];

EC2KeyPairInfoCollection /: Format[HoldPattern[EC2KeyPairInfoCollection[___]], OutputForm] :=
	SequenceForm[EC2KeyPairInfoCollection][BoxForm`SurroundWithAngleBrackets[""]];

(*----------------------------------------------------------------------------*)
PackageExport["EC2KeyPairs"]
SetUsage[EC2KeyPairs,
"EC2KeyPairs[EC2Client[$$]] returns a EC2KeyPairInfoCollection[$$] iterating over the\
 key pairs EC2KeyPairInfo[$$] available to you.
EC2KeyPairs[EC2Client, \"Filters\" -> <|\"filterName\"->{acceptedValue$1, $$}|>]\
 performs a filtered request.
EC2KeyPairs[EC2Client, \"Filters\" -> <|\"tag\[Colon]key\"-> { value$1, $$}, $$|>]\
 returns all the solumes tagged with the pair key -> value$i.
see: http://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeKeyPairs.html\
 for descriptions and all filters available"
]

Options[EC2KeyPairs] = 
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic,
		"Filters" -> Automatic,
		"KeyNames" -> Automatic
		(*MaxItems -> Automatic (*not an interesting option for an iterator*)*)
	};
EC2KeyPairs[client_EC2Client /; validEC2ClientQ[client], query:OptionsPattern[]] :=
	EC2KeyPairs["Client" -> Replace[OptionValue["Client"], Automatic -> client], query];
EC2KeyPairs[query:OptionsPattern[]] := 
	CatchFailureAsMessage @Module[
	{
		client = pickEC2Client[OptionValue["Client"]],
		filters = OptionValue["Filters"],
		keyNames = OptionValue["KeyNames"]
	},
	
	If[ Replace[OptionValue["DryRun"], Automatic -> False],
		AWSDryRun[client["JClient"], iEC2DescribeKeyPairsSetRequest[filters, keyNames]],
		EC2KeyPairInfoCollection[
			NewIterator[
				EC2KeyPairInfoCollection,
				{
					nextToken = ""
				},
				CatchFailureAsMessage @ If[nextToken === Null,
					IteratorExhausted,
					JavaBlock @ Block[
						{req, res, objListToStrm, cachedvalues},
						req = iEC2DescribeKeyPairsSetRequest[filters, keyNames];
						res = safeLibraryInvoke @ client["JClient"]@describeKeyPairs[req];
						ReleaseJavaObject[req];
						nextToken = Null;
						objListToStrm = safeLibraryInvoke @ res@getKeyPairs[];
						cachedvalues =
							Map[
								EC2KeyPairInfo[<|"EC2Client" -> client, "KeyName" -> #|>]&,
								MapJavaMethod[
									objListToStrm,
									"com.amazonaws.services.ec2.model.KeyPairInfo",
									"getKeyName",
									"java.lang.String"
								]
							];
						DelegateIterator[ListIterator[cachedvalues]]
					]
				]
			]
		]
	]
];
FailInOtherCases[EC2KeyPairs];

iEC2DescribeKeyPairsSetRequest[filters_, keyNames_]:=
Module[
	{
		request
	},
	
	request = safeLibraryInvoke @
		JavaNew["com.amazonaws.services.ec2.model.DescribeKeyPairsRequest"];
	
	If[ filters =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withFilters[KeyValueMap[createFilter, filters]]
	];
	
	If[ keyNames =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withKeyNames[keyNames]
	];
	
	request
];


(*----------------------------------------------------------------------------*)
PackageExport["EC2CreateKeyPair"]
SetUsage[EC2CreateKeyPair,
"EC2CreateKeyPair[EC2Client[$$], keyname] creates a key pair named keyname and\
 returns an association with the following fields:
 - \"EC2KeyPairInfo\": the EC2KeyPairInfo[$$] representing the key pair.
 - \"KeyFingerprint\": the SHA-1 digest of the DER encoded private key..
 - \"KeyMaterial\": an unencrypted PEM encoded RSA private key.
"
]
Options[EC2CreateKeyPair]=
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic
	};
EC2CreateKeyPair[client_/;validEC2ClientQ[client], keyname_String, query:OptionsPattern[]] :=
	EC2CreateKeyPair[
		keyname,
		"Client" -> Replace[OptionValue["Client"], Automatic -> client],
		query
	];
EC2CreateKeyPair[keyname_String, query:OptionsPattern[]] :=
	CatchFailureAsMessage @JavaBlock @
		Module[
			{
				client = pickEC2Client[OptionValue["Client"]],
				req, values
			},
			
			req =
				JavaNew["com.amazonaws.services.ec2.model.CreateKeyPairRequest"]
					@withKeyName[keyname];
			
			If[ Replace[OptionValue["DryRun"], Automatic -> False],
				AWSDryRun[client["JClient"], req],
			
				values =
					getPropertiesFromTemplate[
						EC2KeyPairTemplate,
						safeLibraryInvoke @ client["JClient"]@createKeyPair[req]@getKeyPair[]
					];
				<|
				"EC2KeyPairInfo" -> EC2KeyPairInfo[<|"EC2Client" -> client, "KeyName" -> #KeyName|>],
				"KeyFingerprint" -> #KeyFingerprint,
				"KeyMaterial" -> #KeyMaterial
				|>& @ values
			]
		];
FailInOtherCases[EC2CreateKeyPair]

(*----------------------------------------------------------------------------*)
PackageExport["EC2ImportKeyPair"]
SetUsage[EC2ImportKeyPair,
"EC2ImportKeyPair[EC2Client[$$], keyname, publickey] import the publickey from\
 the RSA key pair that you created as key pair named keyname and returns the\
 EC2KeyPairInfo[$$] representing the key pair.
"
]

Options[EC2ImportKeyPair]=
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic
	};
EC2ImportKeyPair[client_/;validEC2ClientQ[client], keyname_String, publicKey_String, query:OptionsPattern[]] :=
	EC2ImportKeyPair[
		"Client" -> Replace[OptionValue["Client"], Automatic -> client],
		keyname,
		query
	];
EC2ImportKeyPair[keyName_, publicKey_, query:OptionsPattern[]] :=
	CatchFailureAsMessage @JavaBlock @ Module[
		{
			client = pickEC2Client[OptionValue["Client"]],
			req, keyout
		},
		
		req =
			JavaNew["com.amazonaws.services.ec2.model.ImportKeyPairRequest"]
				@withKeyName[keyName]
				@withPublicKeyMaterial[ExportString[publicKey, "Base64"]];
		
		If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[client["JClient"], req],
			keyout = safeLibraryInvoke @ client["JClient"]@importKeyPair[req]@getKeyName[];
			EC2KeyPairInfo[<|"EC2Client" -> client, "KeyName" -> keyout|>]
		]
	];
FailInOtherCases[EC2ImportKeyPair]

(*----------------------------------------------------------------------------*)
PackageExport["EC2KeyPairDelete"]
SetUsage[EC2KeyPairDelete,
"EC2KeyPairDelete[EC2KeyPairInfo[$$]] deletes the corresponding key pair."
]

Options[EC2KeyPairDelete]=
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic
	};

EC2KeyPairDelete[ec2KeyPairInfo_/;validEC2KeyPairInfoQ[ec2KeyPairInfo], query:OptionsPattern[]] :=
	EC2KeyPairDelete[
		ec2KeyPairInfo["KeyName"],
		"Client" -> Replace[OptionValue["Client"], Automatic -> ec2KeyPairInfo["EC2Client"]],
		query
	];

EC2KeyPairDelete[keyName_String, query:OptionsPattern[]] := 
	CatchFailureAsMessage @JavaBlock @ Module[
		{
			client = pickEC2Client[OptionValue["Client"]],
			req, jres
		},
		
		req =
			JavaNew["com.amazonaws.services.ec2.model.DeleteKeyPairRequest"]
				@withKeyName[keyName];
		If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[client["JClient"], req],
			jres = safeLibraryInvoke @ client["JClient"]@deleteKeyPair[req];
			<||>
		]
	];

FailInOtherCases[EC2KeyPairDelete];
	
(*----------------------------------------------------------------------------*)
(* ::Section:: *)
(* EC2SecurityGroup *)
(*----------------------------------------------------------------------------*)
PackageExport["EC2SecurityGroup"]
SetUsage[EC2SecurityGroup,
	"EC2SecurityGroup[$$] represent an EC2 securityGroup.
EC2SecurityGroup[EC2Client[$$], securityGroupId] builds the corresponding EC2SecurityGroup[$$].
Dataset[EC2SecurityGroup[$$]] queries all the properties of EC2SecurityGroup[$$].
EC2SecurityGroup[$$][\"property\"] queries the value associated to property (listable)."
];

(* Makes a nicely formatted display box *)
DefineCustomBoxes[EC2SecurityGroup, 
	e:EC2SecurityGroup[content_Association] :> Block[{},
		BoxForm`ArrangeSummaryBox[
			EC2SecurityGroup, e, None, 
			{
				BoxForm`SummaryItem[{"Id: ", e["SecurityGroupId"]}]
			},
			{
			},
			StandardForm
		]
	]
];

EC2SecurityGroup /: Format[HoldPattern[e:EC2SecurityGroup[content_Association]], OutputForm] :=
	SequenceForm[EC2SecurityGroup][BoxForm`SurroundWithAngleBrackets[e["SecurityGroupId"]]];

EC2SecurityGroup[content_Association]["SecurityGroupId"] := content["SecurityGroupId"];
EC2SecurityGroup[content_Association]["EC2Client"] := content["EC2Client"];
(*ToDo -> cache for unmutable values...*)

validEC2SecurityGroupQ[EC2SecurityGroup[jHandles_Association]] :=
	validEC2ClientQ[jHandles["EC2Client"]];
validEC2SecurityGroupQ[anythingElse___] := False;

ValidEC2Q[obj_EC2SecurityGroup] :=
	validEC2SecurityGroupQ[obj];

EC2SecurityGroup /: AWSGetInformation[obj:EC2SecurityGroup[Content_Association]] :=
	CatchFailureAsMessage[AWSGetInformation, ec2SecurityGroupMetadata[obj]];
EC2SecurityGroup /: Dataset[obj:EC2SecurityGroup[Content_Association]] :=
	CatchFailureAsMessage[
			EC2SecurityGroup,
			Dataset[ec2SecurityGroupMetadata[obj]]
	];

EC2SecurityGroup[content_Association][props : (_String | {__String})] :=
	CatchFailureAsMessage[
		EC2SecurityGroup,
		getCurrentProperties[
			"EC2",
			EC2SecurityGroup[content],
			validEC2SecurityGroupQ,
			ec2SecurityGroupMetadata,
			Keys@EC2SecurityGroupTemplate,
			props
		]
	];

buildEC2SecurityGroup[client_EC2Client /; validEC2ClientQ[client], jobj_ /; jClassQ[jobj, "model.SecurityGroup"]] :=
	EC2SecurityGroup[
		<|
		"EC2Client" -> client,
		"SecurityGroupId" -> #SecurityGroupId
		|>
	]& @
	getPropertiesFromTemplate[
		EC2SecurityGroupTemplate,
		jobj
	];

EC2SecurityGroup[ec2Client_EC2Client /; validEC2ClientQ[ec2Client], securityGroupId_String]:=
	EC2SecurityGroup[<|"EC2Client" -> ec2Client, "SecurityGroupId" -> securityGroupId|>];

EC2Connect[obj_EC2SecurityGroup, opt : OptionsPattern[]] := 
	CatchFailureAsMessage @ EC2SecurityGroup[pickEC2Client[OptionValue["Client"]], obj["SecurityGroupId"]];

Protect[EC2SecurityGroup]

(*----------------------------------------------------------------------------*)
ec2SecurityGroupMetadata[img_EC2SecurityGroup] :=
	ec2SecurityGroupMetadata[getValidEC2Client[img], img["SecurityGroupId"]];
ec2SecurityGroupMetadata[client_EC2Client /; validEC2ClientQ[client], securityGroupId_String] := JavaBlock[
	getPropertiesFromTemplate[
		EC2SecurityGroupTemplate,
		safeLibraryInvoke[
			client["JClient"]
				@describeSecurityGroups[
					JavaNew["com.amazonaws.services.ec2.model.DescribeSecurityGroupsRequest"]
						@withGroupIds[{securityGroupId}]
				]@getSecurityGroups[]@get[0]
		]
	]
];
	
(*----------------------------------------------------------------------------*)
PackageExport["EC2DescribeSecurityGroups"]
SetUsage[EC2DescribeSecurityGroups,
"EC2DescribeSecurityGroups[EC2Client] returns an Association with keys:
 - \"Batch\": a list of available security groups and their attributes.
 - \"NextToken\": a token to chains calls and query the next batch of results.
EC2DescribeSecurityGroups[EC2Client, \"Filters\" -> <|filterName$1 ->{acceptedValue$1, $$}|>] perform a filtered request.
see: http://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeSecurityGroups.html for all filters names and descriptions.
EC2DescribeSecurityGroups[$$, \"SecurityGroupIds\"-> {id$1,$$}] filters on instanceIds (default All).
EC2DescribeSecurityGroups[$$, MaxItems-> num] returns up to num results per batch (default 1000).
EC2DescribeSecurityGroups[$$, \"NextToken\"-> token] chains call to returns the next batch of results."
]

Options[EC2DescribeSecurityGroups] = 
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic,
		"Filters" -> Automatic,
		MaxItems -> Automatic,
		"NextToken" -> "",
		"SecurityGroupIds" -> Automatic
	};
EC2DescribeSecurityGroups[ec2Client_/;validEC2ClientQ[ec2Client], query:OptionsPattern[]] :=
	EC2DescribeSecurityGroups[
		"Client" -> Replace[OptionValue["Client"], Automatic -> ec2Client],
		query
	];
EC2DescribeSecurityGroups[query:OptionsPattern[]] := 
	CatchFailureAsMessage @JavaBlock @ Module[
		{
			ec2Client = pickEC2Client[OptionValue["Client"]],
			securityGroupList, requestResult, request, i
		},
		
		request =
			iEC2DescribeSecurityGroupsSetRequest[
				OptionValue["Filters"],
				OptionValue[MaxItems],
				OptionValue["NextToken"],
				OptionValue["SecurityGroupIds"]
			];
			
		If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[ec2Client["JClient"], request],
			requestResult = safeLibraryInvoke @ ec2Client["JClient"]@describeSecurityGroups[request];
			securityGroupList = requestResult@getSecurityGroups[];
			<|
			"Batch" -> Table[getPropertiesFromTemplate[EC2SecurityGroupTemplate, securityGroupList@get[i]], {i, 0, Min[securityGroupList@size[], Replace[OptionValue[MaxItems], Automatic -> Infinity]]-1}],
			"NextToken" -> requestResult@getNextToken[]
			|>
		]
	];
FailInOtherCases[EC2DescribeSecurityGroups];

(*----------------------------------------------------------------------------*)
PackageExport["EC2SecurityGroupCollection"]
SetUsage[EC2SecurityGroupCollection, "EC2SecurityGroupCollection[$$] is an iterator over EC2SecurityGroup[$$] objects.
Get one object:
ec2securityGroup = AWSCollectionRead[EC2SecurityGroupCollection[$$]].
Get up to num objects:
{ec2securityGroup$1, ec2securityGroup$2, $$} = AWSCollectionReadList[EC2SecurityGroupCollection[$$], num]
related: EC2SecurityGroups"];

DefineCustomBoxes[EC2SecurityGroupCollection, 
	e:EC2SecurityGroupCollection[iterator_] :> Block[{},
	BoxForm`ArrangeSummaryBox[
		EC2SecurityGroupCollection, e, None, 
		{
			BoxForm`SummaryItem[{"EC2SecurityGroupCollection:", ""}]
		},
		{
		},
		StandardForm
	]
]];

EC2SecurityGroupCollection /: Format[HoldPattern[EC2SecurityGroupCollection[___]], OutputForm] :=
	SequenceForm[EC2SecurityGroupCollection][BoxForm`SurroundWithAngleBrackets[""]];

(*----------------------------------------------------------------------------*)
PackageExport["EC2SecurityGroups"]
SetUsage[EC2SecurityGroups,
"EC2SecurityGroups[EC2Client[$$]] returns a EC2SecurityGroupCollection[$$] iterating over the\
 securityGroups EC2SecurityGroup[$$] available to you.
EC2SecurityGroups[EC2Client, \"Filters\" -> <|\"filterName\"->{acceptedValue$1, $$}|>]\
 performs a filtered request.
EC2SecurityGroups[EC2Client, \"Filters\" -> <|\"tag\[Colon]key\"-> { value$1, $$}, $$|>]\
 returns all the solumes tagged with the pair key -> value$i.
see: http://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeSecurityGroups.html\
 for descriptions and all filters available"
]

Options[EC2SecurityGroups] = 
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic,
		"Filters" -> Automatic
		(*MaxItems -> Automatic (*not an interesting option for an iterator*)*)
	};
EC2SecurityGroups[client_EC2Client /; validEC2ClientQ[client], query:OptionsPattern[]] :=
	EC2SecurityGroups["Client" -> Replace[OptionValue["Client"], Automatic -> client], query];
EC2SecurityGroups[query:OptionsPattern[]] := 
	CatchFailureAsMessage @Module[
	{
		client = pickEC2Client[OptionValue["Client"]],
		filters = OptionValue["Filters"]
	},
	
	If[ Replace[OptionValue["DryRun"], Automatic -> False],
		AWSDryRun[ec2Client["JClient"], iEC2DescribeSecurityGroupsSetRequest[filters, Automatic, "", Automatic]],
		EC2SecurityGroupCollection[
			NewIterator[
				EC2SecurityGroupCollection,
				{
					nextToken = ""
				},
				CatchFailureAsMessage @ If[nextToken === Null,
					IteratorExhausted,
					JavaBlock @ Block[
						{req, res, objListToStrm, cachedvalues},
						req = iEC2DescribeSecurityGroupsSetRequest[filters, Automatic, nextToken, Automatic];
						res = safeLibraryInvoke @ client["JClient"]@describeSecurityGroups[req];
						ReleaseJavaObject[req];
						nextToken = safeLibraryInvoke @ res@getNextToken[];
						objListToStrm = safeLibraryInvoke @ res@getSecurityGroups[];
						cachedvalues =
							Map[
								EC2SecurityGroup[<|"EC2Client" -> client, "SecurityGroupId" -> #|>]&,
								MapJavaMethod[
									objListToStrm,
									"com.amazonaws.services.ec2.model.SecurityGroup",
									"getGroupId",
									"java.lang.String"
								]
							];
						DelegateIterator[ListIterator[cachedvalues]]
					]
				]
			]
		]
	]
];
FailInOtherCases[EC2SecurityGroups];

iEC2DescribeSecurityGroupsSetRequest[filters_, maxResults_, nextToken_, securityGroupIds_]:=
Module[
	{
		request
	},
	
	request = safeLibraryInvoke @
		JavaNew["com.amazonaws.services.ec2.model.DescribeSecurityGroupsRequest"]
			@withNextToken[nextToken];
	
	If[ filters =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withFilters[KeyValueMap[createFilter, filters]]
	];
	
	If[ maxResults =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withMaxResults[MakeJavaObject[maxResults]]
	];
	
	If[ securityGroupIds =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withInstanceIds[securityGroupIds]
	];
	
	request
];

(*----------------------------------------------------------------------------*)
PackageExport["EC2CreateSecurityGroup"]
SetUsage[EC2CreateSecurityGroup,
"EC2CreateSecurityGroup[EC2Client[$$], groupName] creates a security group named\
 groupName and returns the corresponding EC2SecurityGroup[$$].
EC2CreateSecurityGroup[$$, \"Description\" -> desc] sets the description field\
 of the newly created EC2SecurityGroup[$$] (default: \"(generated by AWSLink)\")."
]

Options[EC2CreateSecurityGroup] =
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic,
		"Description" -> "(generated by AWSLink)"
		(*withVpcId*)
	};

EC2CreateSecurityGroup[client_/;validEC2ClientQ[client], groupName_String, query:OptionsPattern[]] :=
	EC2CreateSecurityGroup[
		groupName,
		"Client" -> Replace[OptionValue["Client"], Automatic -> client],
		query
	];

EC2CreateSecurityGroup[groupName_, query:OptionsPattern[]] :=
	CatchFailureAsMessage @JavaBlock @ Module[
		{
			client = pickEC2Client[OptionValue["Client"]],
			req, gpId
		},
		
		req =
			JavaNew["com.amazonaws.services.ec2.model.CreateSecurityGroupRequest"]
				@withGroupName[groupName]
				@withDescription[OptionValue["Description"]];
		
		If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[client["JClient"], req],
			gpId = safeLibraryInvoke @
				client["JClient"]@createSecurityGroup[req]@getGroupId[];
			EC2SecurityGroup[<|"EC2Client" -> client, "SecurityGroupId" -> gpId|>]
		]
	];
FailInOtherCases[EC2CreateSecurityGroup]

(*----------------------------------------------------------------------------*)
PackageExport["EC2SecurityGroupAuthorizeIngress"]
SetUsage[EC2SecurityGroupAuthorizeIngress,
"EC2SecurityGroupAuthorizeIngress[
	CidrIp: \"string\",
	FromPort: 123,
	GroupName: \"string\",
	IpPermissions: {
		<|
			\"FromPort\"->123,
			\"IpProtocol\"->\"string\",
			\"IpRanges\"->{
				<|
					\"CidrIp\"->\"string\",
					\"Description\"->\"string\"
				|>,
			},
			\"Ipv6Ranges\"->{
				<|
					\"CidrIpv6\"->\"string\",
					\"Description\"->\"string\"
				|>,
			},
			\"PrefixListIds\"->{
				<|
					\"Description\"->\"string\",
					\"PrefixListId\"->\"string\"
				|>,
			},
			\"ToPort\"->123,
			\"UserIdGroupPairs\"->{
				<|
					\"Description\"->\"string\",
					\"GroupId\"->\"string\",
					\"GroupName\"->\"string\",
					\"PeeringStatus\"->\"string\",
					\"UserId\"->\"string\",
					\"VpcId\"->\"string\",
					\"VpcPeeringConnectionId\"->\"string\"
				|>,
			}
		|>,
	},
	IpProtocol: \"string\",
	SourceSecurityGroupName: \"string\",
	SourceSecurityGroupOwnerId: \"string\",
	ToPort: 123,
	DryRun: True|False
]"
]

Options[EC2SecurityGroupAuthorizeIngress] = 
	{
		"CidrIp" -> Automatic,
		"Client" -> Automatic,
		"DryRun" -> Automatic,
		"GroupName" -> Automatic,
		"FromPort" -> Automatic,
		"IpPermissions" -> Automatic,
		"IpProtocol" -> Automatic,
		"SourceSecurityGroupName" -> Automatic,
		"SourceSecurityGroupOwnerId" -> Automatic,
		"ToPort" -> Automatic
	};
EC2SecurityGroupAuthorizeIngress[ec2SecurityGroup_EC2SecurityGroup, query:OptionsPattern[]] :=
	EC2SecurityGroupAuthorizeIngress[
		ec2SecurityGroup["GroupId"],
		"Client" -> Replace[OptionValue["Client"], Automatic -> getValidEC2Client[ec2SecurityGroup]],
		query
	];
EC2SecurityGroupAuthorizeIngress[groupId_String, query:OptionsPattern[]] := 
	CatchFailureAsMessage @JavaBlock @ Module[
		{
			ec2Client = pickEC2Client[OptionValue["Client"]],
			requestResult, request
		},
		
		request =
			iEC2SecurityGroupAuthorizeIngressSetRequest[
				OptionValue["CidrIp"],
				groupId,
				OptionValue["FromPort"],
				OptionValue["IpPermissions"],
				OptionValue["IpProtocol"],
				OptionValue["SourceSecurityGroupName"],
				OptionValue["SourceSecurityGroupOwnerId"],
				OptionValue["ToPort"]
			];
		If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[ec2Client["JClient"], request],
			requestResult = safeLibraryInvoke @ ec2Client["JClient"]@authorizeSecurityGroupIngress[request];
			<||>
		]
	];
FailInOtherCases[EC2SecurityGroupAuthorizeIngress];

iEC2SecurityGroupAuthorizeIngressSetRequest[
	cidrIp_, groupId_, fromPort_,
	ipPermissions_, ipProtocol_, sourceSecurityGroupName_,
	sourceSecurityGroupOwnerId_, toPort_]:=
Module[
	{
		request, ipPermissionval
	},
	
	request = safeLibraryInvoke @
		JavaNew["com.amazonaws.services.ec2.model.AuthorizeSecurityGroupIngressRequest"]
			@withGroupId[groupId];
	
	If[ cidrIp =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withCidrIp[cidrIp]
	];
	
	If[ fromPort =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withFromPort[fromPort]
	];
	
	If[ ipPermissions =!= Automatic,
		ipPermissionval = iIpPermissionsSet[ipPermissions];
		request = safeLibraryInvoke @
			request
				@withIpPermissions[ipPermissionval]
	];
	
	If[ ipProtocol =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withIpProtocol[ipProtocol]
	];
	
	If[ sourceSecurityGroupName =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withSourceSecurityGroupName[sourceSecurityGroupName]
	];
	
	If[ sourceSecurityGroupOwnerId =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withSourceSecurityGroupOwnerId[sourceSecurityGroupOwnerId]
	];
	
	If[ toPort =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withToPort[toPort]
	];
	
	request
];

iIpPermissionsSet[ipPermissions_List] := Block[
	{ipPermissionCol},
	safeLibraryInvoke[LoadJavaClass, "java.util.Arrays"];
	ipPermissionCol =
		Map[
			makeIpPermission,
			ipPermissions
		];
	safeLibraryInvoke @ java`util`Arrays`asList[ipPermissionCol]
]

makeIpPermission[ipPermission_Association]  := Block[
	{ipPermissionsVal},
	ipPermissionsVal = safeLibraryInvoke @
		JavaNew["com.amazonaws.services.ec2.model.IpPermission"];
	KeyValueMap[
		ipPermissionSetFunction[ipPermissionsVal, ##]&,
		ipPermission
	];
	ipPermissionsVal
]

SetAttribute[ipPermissionSetFunction, HoldFirst];
ipPermissionSetFunction[ipPermissionsVal_, ipPermissionField_, ipPermissionSetting_] :=
	Switch[ ipPermissionField,
		"FromPort", safeLibraryInvoke @ ipPermissionsVal@setFromPort[MakeJavaObject@ipPermissionSetting],
		"IpProtocol", safeLibraryInvoke @ ipPermissionsVal@setIpProtocol[ipPermissionSetting],
		"IpRanges", safeLibraryInvoke @ ipPermissionsVal@setIpv4Ranges[iIpv4RangesSet[ipPermissionSetting]],
		"Ipv6Ranges", safeLibraryInvoke @ ipPermissionsVal@setIpv6Ranges[iIpv6RangesSet[ipPermissionSetting]],
		"PrefixListIds", safeLibraryInvoke @ ipPermissionsVal@setPrefixListIds[iPrefixListIdsSet[ipPermissionSetting]],
		"ToPort", safeLibraryInvoke @ ipPermissionsVal@setToPort[MakeJavaObject@ipPermissionSetting],
		"UserIdGroupPairs", safeLibraryInvoke @ ipPermissionsVal@setUserIdGroupPairs[iUserIdGroupPairsSet[ipPermissionSetting]],
		_, ThrowFailure["awsnofield", "IpPermission", ipPermissionField]
	];

iIpv4RangesSet[ipRanges_List] := Block[
	{ipRangeCol},
	safeLibraryInvoke[LoadJavaClass, "java.util.Arrays"];
	ipRangeCol =
		Map[
			makeIpv4Range,
			ipRanges
		];
	safeLibraryInvoke @ java`util`Arrays`asList[ipRangeCol]
];

makeIpv4Range[ipRange_Association] := Block[
	{ipRangeVal},
	ipRangeVal = safeLibraryInvoke @
		JavaNew["com.amazonaws.services.ec2.model.IpRange"];
	KeyValueMap[
		ipv4RangeSetFunction[ipRangeVal, ##]&,
		ipRange
	];
	ipRangeVal
]

SetAttribute[ipv4RangeSetFunction, HoldFirst];
ipv4RangeSetFunction[ipRangeVal_, ipRangeField_, ipRangeSetting_]:=
	Switch[ ipRangeField,
		"CidrIp", safeLibraryInvoke @ ipRangeVal@setCidrIp[ipRangeSetting],
		"Description", safeLibraryInvoke @ ipRangeVal@setDescription[ipRangeSetting],
		_, ThrowFailure["awsnofield", "IpRanges", ipRangeField]
	]

iIpv6RangesSet[ipv6Ranges_List]:= Block[
	{ipv6RangesCol},
	safeLibraryInvoke[LoadJavaClass, "java.util.Arrays"];
	ipv6RangesCol = 
		Map[
			makeIpv6Range,
			ipv6Ranges
		];
	safeLibraryInvoke @ java`util`Arrays`asList[ipv6RangesCol]
];

makeIpv6Range[ipv6Range_Association] := Block[
	{ipv6RangeVal},
	ipv6RangeVal = safeLibraryInvoke @
		JavaNew["com.amazonaws.services.ec2.model.Ipv6Range"];
	KeyValueMap[
		ipv6RangeSetFunction[ipv6RangeVal, ##]&,
		ipv6Range
	];
	ipv6RangeVal
]

SetAttribute[ipv6RangeSetFunction, HoldFirst];
ipv6RangeSetFunction[ipv6RangeVal_, ipv6RangeField_, ipv6RangeSetting_]:=
	Switch[ ipv6RangeField,
		"CidrIpv6", safeLibraryInvoke @ ipv6RangeVal@setCidrIpv6[ipv6RangeSetting],
		"Description", safeLibraryInvoke @ ipv6RangeVal@setDescription[ipv6RangeSetting],
		_, ThrowFailure["awsnofield", "Ipv6Ranges", ipv6RangeField]
	]

iPrefixListIdsSet[prefixListIds_List]:= Block[
	{prefixListIdsCol},
	safeLibraryInvoke[LoadJavaClass, "java.util.Arrays"];
	prefixListIdsCol = 
		Map[
			makePrefixListId,
			prefixListIds
		];
	safeLibraryInvoke @ java`util`Arrays`asList[prefixListIdsCol]
];

makePrefixListId[prefixListId_Association] := Block[
	{prefixListIdVal},
	prefixListIdVal = safeLibraryInvoke @
		JavaNew["com.amazonaws.services.ec2.model.PrefixListId"];
	KeyValueMap[
		prefixListIdSetFunction[prefixListIdVal, ##]&,
		prefixListId
	];
	prefixListIdVal
]

SetAttribute[prefixListIdSetFunction, HoldFirst];
prefixListIdSetFunction[prefixListIdVal_, prefixListIdField_, prefixListIdSetting_]:=
	Switch[ prefixListIdField,
		"Description" -> safeLibraryInvoke @ prefixListIdVal@setDescription[prefixListIdSetting],
		"PrefixListId" -> safeLibraryInvoke @ prefixListIdVal@setPrefixListId[prefixListIdSetting],
		_, ThrowFailure["awsnofield", "PrefixListIds", prefixListIdField]
	]

iUserIdGroupPairsSet[userIdGroupPairs_List]:= Block[
	{userIdGroupPairsCol},
	safeLibraryInvoke[LoadJavaClass, "java.util.Arrays"];
	userIdGroupPairsCol =
		Map[
			makeUserIdGroupPairs,
			userIdGroupPairs
		];
	safeLibraryInvoke @ java`util`Arrays`asList[userIdGroupPairsCol]
];

makeUserIdGroupPairs[userIdGroupPair_Association] := Block[
	{userIdGroupPairsVal},
	userIdGroupPairsVal = safeLibraryInvoke @
		JavaNew["com.amazonaws.services.ec2.model.IdGroupPair"];
	KeyValueMap[
		userIdGroupPairSetFunction[userIdGroupPairsVal, ##]&,
		userIdGroupPair
	];
	userIdGroupPairsVal
]

SetAttribute[userIdGroupPairSetFunction, HoldFirst];
userIdGroupPairSetFunction[userIdGroupPairVal_, userIdGroupPairField_, userIdGroupPairSetting_]:=
	Switch[ userIdGroupPairField,
		"Description" -> safeLibraryInvoke @ userIdGroupPairVal@setDescription[userIdGroupPairSetting],
		"GroupId" -> safeLibraryInvoke @ userIdGroupPairVal@setGroupId[userIdGroupPairSetting],
		"GroupName" -> safeLibraryInvoke @ userIdGroupPairVal@setGroupName[userIdGroupPairSetting],
		"PeeringStatus" -> safeLibraryInvoke @ userIdGroupPairVal@setPeeringStatus[userIdGroupPairSetting],
		"UserId" -> safeLibraryInvoke @ userIdGroupPairVal@setUserId[userIdGroupPairSetting],
		"VpcId" -> safeLibraryInvoke @ userIdGroupPairVal@setVpcId[userIdGroupPairSetting],
		"VpcPeeringConnectionId" -> safeLibraryInvoke @ userIdGroupPairVal@setVpcPeeringConnectionId[userIdGroupPairSetting],
		_, ThrowFailure["awsnofield", "UserIdGroupPairs", userIdGroupPairField]
	]

(*----------------------------------------------------------------------------*)
PackageExport["EC2SecurityGroupDelete"]
SetUsage[EC2SecurityGroupDelete,
"EC2SecurityGroupDelete[EC2SecurityGroup[$$]] deletes the security group."
]
Options[EC2SecurityGroupDelete] =
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic
	};
EC2SecurityGroupDelete[ec2SecurityGroup_ /; validEC2SecurityGroupQ[ec2SecurityGroup], query:OptionsPattern[]] :=
	CatchFailureAsMessage @
		EC2SecurityGroupDelete[
			ec2SecurityGroup["SecurityGroupId"],
			"Client" -> Replace[OptionValue["Client"], Automatic -> ec2SecurityGroup["EC2Client"]],
			query
		];

EC2SecurityGroupDelete[securityGroupId_, op:OptionsPattern[]] := 
	CatchFailureAsMessage @JavaBlock @ Module[
		{req, jres, client = pickEC2Client[OptionValue["Client"]]},
		
		req =
			JavaNew["com.amazonaws.services.ec2.model.DeleteSecurityGroupRequest"]
				@withGroupId[securityGroupId];
			
		If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[client["JClient"], req],
			jres = safeLibraryInvoke @ client["JClient"]@deleteSecurityGroup[req];
			<||>
		]
	];

FailInOtherCases[EC2SecurityGroupDelete]
	

(*----------------------------------------------------------------------------*)

(* ::Section:: *)
(* Tags *)

(*----------------------------------------------------------------------------*)
PackageExport["EC2SetTag"]
SetUsage[EC2SetTag,
"EC2SetTag[obj$, <| key$1-> value$1, $$|>] creates/overwrites the tags\
 \"key$i\" with the values \"value$i\" on the object obj. The object obj\
 can be: EC2Image[$$], EC2Instance[$$], EC2Snapshot[$$], EC2Volume[$$]."
]

EC2Tag::ignored = "`1` object provided is not supported and has been ignored";
getRessourcesId[ec2Object_ /; ValidEC2Q[ec2Object]]:=
	Switch[ec2Object,
		(*_EC2Reservation, ec2Object["ReservationId"],*)
		_EC2Image,
			ec2Object["ImageId"],
		_EC2Instance,
			ec2Object["InstanceId"],
		_EC2Snapshot,
			ec2Object["SnapshotId"],
		_EC2SpotInstanceRequest,
			ec2Object["SpotInstanceRequestId"],
		_EC2Volume,
			ec2Object["VolumeId"],
		_, Message[EC2Tag::ignored, Head[ec2Object]]; Nothing
	];

EC2SetTag[ec2Object_ /; ValidEC2Q[ec2Object], tags_Association] := 
	CatchFailureAsMessage @JavaBlock @ Module[
		{request, tags$, ressourceId},
		tags$ = KeyValueMap[JavaNew["com.amazonaws.services.ec2.model.Tag", ##]&, tags];
		ressourceId = getRessourcesId[ec2Object];
		If[ ressourceId =!= Nothing,
		request = iEC2SetTagRequest[{ressourceId}, tags$, "Set"];
		getValidEC2Client[ec2Object]["JClient"]@createTags[request]
		];
		ec2Object
	];
FailInOtherCases[EC2SetTag];

PackageExport["EC2DeleteTag"]
SetUsage[EC2DeleteTag,
"EC2DeleteTag[obj$, {key$1, $$}] deletes the tags \"key$i\" from the object obj.
 The object obj$ can be: EC2Image[$$], EC2Instance[$$], EC2Snapshot[$$], EC2Volume[$$].
EC2DeleteTag[obj$, <|key$1 -> value$1, $$|>] deletes the tags\
 \"key$i\" only if its value is \"value$i\" from the object obj."
]
EC2DeleteTag[ec2Object_/;ValidEC2Q[ec2Object], tags:(_Association|{__String})] := 
	CatchFailureAsMessage @JavaBlock @ Module[
		{request, tags$, ressourceId},
		tags$ = 
			Switch[tags,
				_Association,
					KeyValueMap[safeLibraryInvoke[JavaNew, "com.amazonaws.services.ec2.model.Tag",##]&, tags],
				_List,
					Map[safeLibraryInvoke[JavaNew, "com.amazonaws.services.ec2.model.Tag", #]&, tags]
			];
		ressourceId = getRessourcesId[ec2Object];
		If[ ressourceId =!= Nothing,
		request = iEC2SetTagRequest[{ressourceId}, tags$, "Delete"];
		getValidEC2Client[ec2Object]["JClient"]@deleteTags[request]
		];
		
		ec2Object
	];
FailInOtherCases[EC2DeleteTag];

Options[iEC2SetTagRequest] = 
	{
		"GeneralProgressListener"-> Automatic,
		"RequestMetricCollector" -> Automatic,
		"SdkClientExecutionTimeout" -> Automatic,
		"SdkRequestTimeout" -> Automatic
	};
iEC2SetTagRequest[ressourceIds:{__String}, tags_, action_]:=
Module[
	{
		request
	},
	
	request = safeLibraryInvoke @
		Switch[ action,
			"Set",
				JavaNew["com.amazonaws.services.ec2.model.CreateTagsRequest"],
			"Delete",
				JavaNew["com.amazonaws.services.ec2.model.DeleteTagsRequest"]
		]
			@withResources[ressourceIds]
			@withTags[tags];
	
	request
];

(* ::Section:: *)
(* EC2SpotInstanceRequest *)
(*----------------------------------------------------------------------------*)
PackageExport["EC2SpotInstanceRequest"]
SetUsage[EC2SpotInstanceRequest,
	"EC2SpotInstanceRequest[$$] represent an EC2 spot-instance request.
EC2SpotInstanceRequest[EC2Client[$$], spotInstanceRequestId] builds the corresponding EC2SpotInstanceRequest[$$].
Dataset[EC2SpotInstanceRequest[$$]] queries all the properties of EC2SpotInstanceRequest[$$].
EC2SpotInstanceRequest[$$][\"property\"] queries the value associated to property (listable)."
];

(* Makes a nicely formatted display box *)
DefineCustomBoxes[EC2SpotInstanceRequest, 
	e:EC2SpotInstanceRequest[content_Association] :> Block[{},
		BoxForm`ArrangeSummaryBox[
			EC2SpotInstanceRequest, e, None, 
			{
				BoxForm`SummaryItem[{"Id: ", e["SpotInstanceRequestId"]}]
			},
			{
			},
			StandardForm
		]
	]
];

EC2SpotInstanceRequest /: Format[HoldPattern[e:EC2SpotInstanceRequest[content_Association]], OutputForm] :=
	SequenceForm[EC2SpotInstanceRequest][BoxForm`SurroundWithAngleBrackets[e["SpotInstanceRequestId"]]];

EC2SpotInstanceRequest[content_Association]["SpotInstanceRequestId"] := content["SpotInstanceRequestId"];
EC2SpotInstanceRequest[content_Association]["EC2Client"] := content["EC2Client"];
(*ToDo -> cache for unmutable values...*)

validEC2SpotInstanceRequestQ[EC2SpotInstanceRequest[jHandles_Association]] :=
	validEC2ClientQ[jHandles["EC2Client"]];
validEC2SpotInstanceRequestQ[anythingElse___] := False;

ValidEC2Q[obj_EC2SpotInstanceRequest] :=
	validEC2SpotInstanceRequestQ[obj];

EC2SpotInstanceRequest /: AWSGetInformation[obj:EC2SpotInstanceRequest[Content_Association]] :=
	CatchFailureAsMessage[AWSGetInformation, ec2SpotInstanceRequestMetadata[obj]];
EC2SpotInstanceRequest /: Dataset[obj:EC2SpotInstanceRequest[Content_Association]] :=
	CatchFailureAsMessage[
			EC2SpotInstanceRequest,
			Dataset[ec2SpotInstanceRequestMetadata[obj]]
	];

EC2SpotInstanceRequest[content_Association][props : (_String | {__String})] :=
	CatchFailureAsMessage[
		EC2SpotInstanceRequest,
		getCurrentProperties[
			"EC2",
			EC2SpotInstanceRequest[content],
			validEC2SpotInstanceRequestQ,
			ec2SpotInstanceRequestMetadata,
			Keys@EC2SpotInstanceRequestTemplate,
			props
		]
	];

buildEC2SpotInstanceRequest[client_EC2Client /; validEC2ClientQ[client], jobj_ /; jClassQ[jobj, "model.SpotInstanceRequest"]] :=
	EC2SpotInstanceRequest[
		<|
		"EC2Client" -> client,
		"SpotInstanceRequestId" -> #SpotInstanceRequestId
		|>
	]& @
	getPropertiesFromTemplate[
		EC2SpotInstanceRequestTemplate,
		jobj
	];

EC2SpotInstanceRequest[ec2Client_EC2Client /; validEC2ClientQ[ec2Client], spotInstanceRequestId_String]:=
	EC2SpotInstanceRequest[<|"EC2Client" -> ec2Client, "SpotInstanceRequestId" -> spotInstanceRequestId|>];
	
EC2Connect[obj_EC2SpotInstanceRequest, opt : OptionsPattern[]] := 
	CatchFailureAsMessage @ EC2SpotInstanceRequest[pickEC2Client[OptionValue["Client"]], obj["SpotInstanceRequestId"]];

Protect[EC2SpotInstanceRequest]
(*----------------------------------------------------------------------------*)
ec2SpotInstanceRequestMetadata[obj_EC2SpotInstanceRequest] :=
	ec2SpotInstanceRequestMetadata[getValidEC2Client[obj], obj["SpotInstanceRequestId"]];
ec2SpotInstanceRequestMetadata[client_EC2Client /; validEC2ClientQ[client], spotInstanceRequestId_String] := JavaBlock[
	getPropertiesFromTemplate[
		EC2SpotInstanceRequestTemplate,
		safeLibraryInvoke[
			client["JClient"]
				@describeSpotInstanceRequests[
					JavaNew["com.amazonaws.services.ec2.model.DescribeSpotInstanceRequestsRequest"]
						@withSpotInstanceRequestIds[{spotInstanceRequestId}]
				]@getSpotInstanceRequests[]@get[0]
		]
	]
];

(*----------------------------------------------------------------------------*)
PackageExport["EC2DescribeSpotInstanceRequests"]
SetUsage[EC2DescribeSpotInstanceRequests,
"EC2DescribeSpotInstanceRequests[EC2Client] returns an Association with keys:
 - \"Batch\": a list of available spotInstanceRequests and their attributes.
 - \"NextToken\": a token to chains calls and query the next batch of results.
EC2DescribeSpotInstanceRequests[EC2Client, \"Filters\" -> <|filterName$1 ->{acceptedValue$1, $$}|>] perform a filtered request.
see: http://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeSpotInstanceRequests.html for all filters names and descriptions.
EC2DescribeSpotInstanceRequests[$$, MaxItems-> num] returns up to num results per batch per batch (default 1000).
EC2DescribeSpotInstanceRequests[$$, \"SpotInstanceRequestIds\"-> {id$1,$$}] filters on instanceIds (default All)."
]

Options[EC2DescribeSpotInstanceRequests] = 
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic,
		"Filters" -> Automatic,
		MaxItems -> 1000,
		"SpotInstanceRequestIds" -> Automatic
	};
EC2DescribeSpotInstanceRequests[ec2Client_/;validEC2ClientQ[ec2Client], query:OptionsPattern[]] :=
	EC2DescribeSpotInstanceRequests[
		"Client" -> Replace[OptionValue["Client"], Automatic -> ec2Client],
		query
	];
EC2DescribeSpotInstanceRequests[query:OptionsPattern[]] := 
	CatchFailureAsMessage @JavaBlock @ Module[
		{
			ec2Client = pickEC2Client[OptionValue["Client"]],
			spotInstanceRequestList, spotInstanceRequestReqResult, request, i
		},
		
		request =
			iEC2SpotInstanceRequestDescribeSetRequest[
				OptionValue["Filters"],
				OptionValue["SpotInstanceRequestIds"]
			];
		If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[ec2Client["JClient"], request],
			spotInstanceRequestReqResult = safeLibraryInvoke @ ec2Client["JClient"]@describeSpotInstanceRequests[request];
			spotInstanceRequestList = safeLibraryInvoke @ spotInstanceRequestReqResult@getSpotInstanceRequests[];
			<|
				"Batch" ->
					Table[
						getPropertiesFromTemplate[EC2SpotInstanceRequestTemplate, spotInstanceRequestList@get[i]],
						{i, 0, Min[spotInstanceRequestList@size[], Replace[OptionValue[MaxItems], Automatic -> Infinity]]-1}
					],
				"NextToken" -> ""
			|>
		]
	];

FailInOtherCases[EC2DescribeSpotInstanceRequests];

iEC2SpotInstanceRequestDescribeSetRequest[filters_, spotInstanceRequestIds_]:=
Module[
	{
		request
	},
	
	request = safeLibraryInvoke @
		JavaNew["com.amazonaws.services.ec2.model.DescribeSpotInstanceRequestsRequest"];
	
	If[ filters =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withFilters[KeyValueMap[createFilter, filters]]
	];
	
	If[ spotInstanceRequestIds =!= Automatic,
		request = safeLibraryInvoke @
			request
				@Automatic[spotInstanceRequestIds]
	];
	
	request
];

(*----------------------------------------------------------------------------*)
PackageExport["EC2SpotInstanceRequestCollection"]
SetUsage[EC2SpotInstanceRequestCollection, "EC2SpotInstanceRequestCollection[$$] is an iterator over EC2SpotInstanceRequest[$$] objects.
Get one object:
ec2spotInstanceRequest = AWSCollectionRead[EC2SpotInstanceRequestCollection[$$]].
Get up to num objects:
{ec2spotInstanceRequest$1, ec2spotInstanceRequest$2, $$} = AWSCollectionReadList[EC2SpotInstanceRequestCollection[$$], num]
related: EC2SpotInstanceRequests"];

DefineCustomBoxes[EC2SpotInstanceRequestCollection, 
	e:EC2SpotInstanceRequestCollection[iterator_] :> Block[{},
	BoxForm`ArrangeSummaryBox[
		EC2SpotInstanceRequestCollection, e, None, 
		{
			BoxForm`SummaryItem[{"EC2SpotInstanceRequestCollection:", ""}]
		},
		{
		},
		StandardForm
	]
]];

EC2SpotInstanceRequestCollection /: Format[HoldPattern[EC2SpotInstanceRequestCollection[___]], OutputForm] :=
	SequenceForm[EC2SpotInstanceRequestCollection][BoxForm`SurroundWithAngleBrackets[""]];

(*----------------------------------------------------------------------------*)
PackageExport["EC2SpotInstanceRequests"]
SetUsage[EC2SpotInstanceRequests,
"EC2SpotInstanceRequests[EC2Client[$$]] returns a EC2SpotInstanceRequestCollection[$$] iterating over the\
 spotInstanceRequests EC2SpotInstanceRequest[$$] available to you.
EC2SpotInstanceRequests[EC2Client, \"Filters\" -> <|\"filterName\"->{acceptedValue$1, $$}|>]\
 performs a filtered request.
EC2SpotInstanceRequests[EC2Client, \"Filters\" -> <|\"tag\[Colon]key\"-> { value$1, $$}, $$|>]\
 returns all the solumes tagged with the pair key -> value$i.
see: http://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeSpotInstanceRequests.html\
 for descriptions and all filters available"
]

Options[EC2SpotInstanceRequests] = 
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic,
		"Filters" -> Automatic,
		(*MaxItems -> 1000*)
		"SpotInstanceRequestIds" -> Automatic
	};
EC2SpotInstanceRequests[client_EC2Client /; validEC2ClientQ[client], query:OptionsPattern[]] :=
	EC2SpotInstanceRequests["Client" -> Replace[OptionValue["Client"], Automatic -> client], query];
EC2SpotInstanceRequests[query:OptionsPattern[]] := 
	CatchFailureAsMessage @Module[
	{
		client = pickEC2Client[OptionValue["Client"]],
		filters = OptionValue["Filters"],
		spotInstanceRequestIds = OptionValue["SpotInstanceRequestIds"]
	},
	
	If[ Replace[OptionValue["DryRun"], Automatic -> False],
		AWSDryRun[client["JClient"], iEC2DescribeSpotInstanceRequestsSetRequest[filters, Automatic, spotInstanceRequestIds]],
		EC2SpotInstanceRequestCollection[
			NewIterator[
				EC2SpotInstanceRequestCollection,
				{
					nextToken = ""
				},
				CatchFailureAsMessage @ If[nextToken === Null,
					IteratorExhausted,
					JavaBlock @ Block[
						{req, res, objListToStrm, cachedvalues},
						req = iEC2DescribeSpotInstanceRequestsSetRequest[filters, Automatic, spotInstanceRequestIds];
						res = safeLibraryInvoke @ client["JClient"]@describeSpotInstanceRequests[req];
						ReleaseJavaObject[req];
						nextToken = Null;
						objListToStrm = safeLibraryInvoke @ res@getSpotInstanceRequests[];
						cachedvalues =
							Map[
								EC2SpotInstanceRequest[<|"EC2Client" -> client, "SpotInstanceRequestId" -> #|>]&,
								MapJavaMethod[
									objListToStrm,
									"com.amazonaws.services.ec2.model.SpotInstanceRequest",
									"getSpotInstanceRequestId",
									"java.lang.String"
								]
							];
						DelegateIterator[ListIterator[cachedvalues]]
					]
				]
			]
		]
	]
];
FailInOtherCases[EC2SpotInstanceRequests];

iEC2DescribeSpotInstanceRequestsSetRequest[filters_, maxResults_, spotInstanceRequestIds_]:=
Module[
	{
		request
	},
	
	request = safeLibraryInvoke @
		JavaNew["com.amazonaws.services.ec2.model.DescribeSpotInstanceRequestsRequest"];
	
	If[ filters =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withFilters[KeyValueMap[createFilter, filters]]
	];
	
	If[ spotInstanceRequestIds =!= Automatic,
		request = safeLibraryInvoke @
			request
				@withSpotInstanceRequestIds[spotInstanceRequestIds]
	];
	
	request
];

(*----------------------------------------------------------------------------*)
PackageExport["EC2SpotInstanceRequestCancel"]
SetUsage[EC2SpotInstanceRequestCancel,
"EC2SpotInstanceRequestCancel[EC2SpotInstanceRequest[$$]] request the\
 EC2SpotInstanceRequest[$$] deletion."
]
Options[EC2SpotInstanceRequestCancel] =
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic
	};
EC2SpotInstanceRequestCancel[spotInstanceRequest_EC2SpotInstanceRequest/;validEC2SpotInstanceRequestQ[spotInstanceRequest], query:OptionsPattern[]] :=
	CatchFailureAsMessage @
		EC2SpotInstanceRequestCancel[
			spotInstanceRequest["SpotInstanceRequestId"],
			"Client" -> Replace[OptionValue["Client"], Automatic -> spotInstanceRequest["EC2Client"]],
			query
		];

EC2SpotInstanceRequestCancel[spotInstanceRequestId_String, op:OptionsPattern[]]:= 
	CatchFailureAsMessage @JavaBlock @ Module[
		{req, res, client = pickEC2Client[OptionValue["Client"]]},
		
		req =
			JavaNew["com.amazonaws.services.ec2.model.CancelSpotInstanceRequestsRequest"]
				@withSpotInstanceRequestIds[{spotInstanceRequestId}];
		
		If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[client["JClient"], req],
			res = safeLibraryInvoke @ client["JClient"]@cancelSpotInstanceRequests[req];
			<||>
		]
	];


(*----------------------------------------------------------------------------*)
(* ::Section:: *)
(* EC2SpotInstancePriceHistory *)
(*----------------------------------------------------------------------------*)
PackageExport["EC2SpotInstancePriceHistory"]
SetUsage[EC2SpotInstancePriceHistory,
"EC2SpotInstancePriceHistory[EC2Client] use the EC2 client to return the history of the spot instances prices in the last hour
EC2SpotInstancePriceHistory[$$, \"From\"-> StartDate, \"To\"-> EndDate] returns the history of the spot instances prices between startDate to endDate
EC2SpotInstancePriceHistory[$$,  \"InstanceTypes\"-> {instanceName1, $$}] query only for the specified spot instances types
EC2SpotInstancePriceHistory[$$,  \"ProductDescriptions\"-> {ProductDescriptions1, $$}] query only for the specified spot instances product (\"Windows\"|\"Linux\\UNIX\"|\"SUZE Linux\")
see http://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeSpotPriceHistory.html  for all filters names and descriptions"
]

Options[EC2SpotInstancePriceHistory] = 
	{
		"Client" -> Automatic,
		"DryRun" -> Automatic,
		"AvailabilityZone" -> Automatic,(*NOT WORKING: To specify multiple Availability Zones, separate them using commas; for example, "us-west-2a, us-west-2b". *)
		"Filters" -> Automatic,
		"From" -> Hold@DatePlus[Now,-1/24],
		"InstanceTypes" -> Automatic,
		"ProductDescriptions" -> Automatic, (* The product description for the Spot price (Linux/UNIX | SUSE Linux | Windows | Linux/UNIX (Amazon VPC) | SUSE Linux (Amazon VPC) | Windows (Amazon VPC)) *)
		"To" -> Hold@Now,
		"Token" -> ""(*,
		"GeneralProgressListener"-> Automatic,
		"MaxResults" -> Automatic,
		"RequestMetricCollector" -> Automatic,
		"SdkClientExecutionTimeout" -> Automatic,
		"SdkRequestTimeout" -> Automatic,
		*)
	};
EC2SpotInstancePriceHistory[ec2Client_EC2Client, query:OptionsPattern[]] :=
	EC2SpotInstancePriceHistory[
		"Client" -> Replace[OptionValue["Client"], Automatic -> ec2Client],
		query
	];

EC2SpotInstancePriceHistory[query:OptionsPattern[]] := 
	CatchFailureAsMessage @JavaBlock @ Module[
		{
			ec2Client = pickEC2Client[OptionValue["Client"]],
			request
		},

		request =
			iEC2SpotInstancePriceHistorySetRequest[
				OptionValue["AvailabilityZone"], OptionValue["Token"],
				ReleaseHold@OptionValue["To"], OptionValue["Filters"],
				OptionValue["InstanceTypes"], OptionValue["ProductDescriptions"],
				ReleaseHold@OptionValue["From"]
			];
		If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[ec2Client["JClient"], request],
			formatSpotInstancePriceHistory @ iEC2SpotInstancePriceHistory[ec2Client["JClient"], request]
		]
	];

iEC2SpotInstancePriceHistorySetRequest[availabilityZone_, currentToken_, endDate_, filters_, instanceTypes_, productDescriptions_, startDate_]:=
Module[
	{request, instanceTypesMod},
	
	request =
		safeLibraryInvoke @ 
		JavaNew["com.amazonaws.services.ec2.model.DescribeSpotPriceHistoryRequest"]
		@withNextToken[currentToken]
		@withStartTime[JavaNew["java.util.Date",(UnixTime[startDate]*1000)]]
		@withEndTime[JavaNew["java.util.Date",(UnixTime[endDate]*1000)]];
		
	If[availabilityZone =!= Automatic,
		request = 
			safeLibraryInvoke @ request
			@withAvailabilityZone[availabilityZone]
	];
	
	If[productDescriptions =!= Automatic,
		request = 
			safeLibraryInvoke @ request
			@withProductDescriptions[Flatten[{productDescriptions}]]
	];
	
	If[filters =!= Automatic,
		request = 
			safeLibraryInvoke @ request
				@withFilters[KeyValueMap[createFilter, filters]]
	];
	
	If[instanceTypes =!= Automatic,
		safeLibraryInvoke @ LoadJavaClass["com.amazonaws.services.ec2.model.InstanceType"];
		instanceTypesMod = safeLibraryInvoke @ com`amazonaws`services`ec2`model`InstanceType`fromValue[#]& /@ Flatten[{instanceTypes}];
		request = 
			safeLibraryInvoke @ request
			@withInstanceTypes[instanceTypesMod]
	];
	
	request
];
	
iEC2SpotInstancePriceHistory[clientJavaj_/;jClassQ[clientJavaj, "AmazonEC2Client"], request_]:=
	Module[
		{spotPriceHistoryList, priceBatch},
		
		spotPriceHistoryList = safeLibraryInvoke[JavaNew,"com.wolfram.awslink.SpotPriceHistoryList", clientJavaj, request];
		If[
			FailureQ[spotPriceHistoryList]
			,
			spotPriceHistoryList
			,
			priceBatch =
				Map[
					Association,
					Transpose[
						{
							Thread["Time" -> (FromUnixTime /@ (#@getTimesList[]/1000))],
						 	Thread["InstanceType" -> #@getInstanceTypesList[]],
						 	Thread["ProductDescription" -> #@getProductDescriptionsList[]],
						 	Thread["SpotPrice" -> Quantity[ToExpression[#@getSpotPricesList[]], Times[Power["Hours", -1], "USDollars"]]],
						 	Thread["AvailabilityZone" -> #@getAvailabilityZonesList[]]
						 }
					]
				]& @ spotPriceHistoryList
		]
	]

formatSpotInstancePriceHistory[pricesHist_]:=
	Dataset[
		Map[
			TimeSeries[Map[Values, #]] &,
			Normal[Dataset[pricesHist][GroupBy["ProductDescription"], GroupBy["InstanceType"], GroupBy["AvailabilityZone"], KeyDrop[{"ProductDescription", "InstanceType", "AvailabilityZone"}]]],
			{-6}]
	]


(*----------------------------------------------------------------------------*)
(* ::Section:: *)
(* Launch Instances *)
(*----------------------------------------------------------------------------*)
$CommonRunAndRequestOptions =
	{
		"BlockDeviceMappings" -> Automatic,
		"ClientToken" -> Automatic,
		(*"DryRun" -> Automatic,*)
		"EbsOptimized" -> Automatic,
		"IamInstanceProfile" -> Automatic,
		"InstanceType" -> Automatic,
		"KernelId" -> Automatic,
		(*KeyName*) "KeyPair"-> Automatic,
		"Monitoring" -> Automatic,
		"Placement" -> Automatic,
		"RamdiskId" -> Automatic,
		"SecurityGroups"-> Automatic,
		"UserData" -> Automatic
		(*
		"InstanceInterruption(ShutDown)Behavior" -> Automatic,
		"NetworkInterfaces",
		*)
	}

$iCommonRunAndRequestSettings =
	<|
	"ClientToken" -> Missing["Optional"],
	(*"DryRun" -> Missing["Optional"],*)
	"LaunchSpecification" -> 
		<|
		"BlockDeviceMappings" -> Missing["Optional"],
		"EbsOptimized" -> Missing["Optional"],
		"IamInstanceProfile" -> Missing["Optional"],
		"ImageId" ->  Missing["Optional"],
		"InstanceType" -> Missing["Optional"],
		"KernelId" -> Missing["Optional"],
		"KeyPair" (*KeyName*)-> Missing["Optional"],
		"Monitoring" -> Missing["Optional"],
		"Placement" -> Missing["Optional"],
		"RamdiskId" -> Missing["Optional"],
		"SecurityGroupIds"-> Missing["Optional"],
		"UserData" -> Missing["Optional"]
		|>
	|>;

$OnDemandOnlyOption =
	(*conflict $SpotOnlyOption*)
	{
		{"Placement", "Affinity"} -> Automatic,
		{"Placement", "HostId"} -> Automatic,
		{"Placement", "SpreadDomain"} -> Automatic,
		"TagSpecifications" -> Automatic
		
	};
$SpotOnlyOption =
	(*conflict $OnDemandOnlyOption*)
	{
		{"InstanceMarketOptions", "MarketType"} -> Automatic,
		{"InstanceMarketOptions", "SpotOptions", "MaxPrice"} -> Automatic,
		{"InstanceMarketOptions", "SpotOptions", "SpotInstanceType"} -> Automatic,
		{"InstanceMarketOptions", "SpotOptions", "BlockDurationMinutes"} -> Automatic,
		{"InstanceMarketOptions", "SpotOptions", "ValidUntil"} -> Automatic
		(*{"InstanceMarketOptions", "SpotOptions", "InstanceInterruptionBehavior"} -> Automatic*)
	};

Options[iCommonRunAndRequestSettings] = $CommonRunAndRequestOptions;
iCommonRunAndRequestSettings[requestSettings_, img_, opts:OptionsPattern[]] := Module[
	{requestSettings$ = requestSettings},
	(*check options*)
		(*"InstanceInterruptionBehavior" (spot) / "InstanceInitiatedShutdownBehavior" (run)*)
		(*"LaunchSpecification"*)
			(*"BlockDeviceMappings"*)
			requestSettings$ = requestSettings;
			requestSettings$[["LaunchSpecification", "BlockDeviceMappings"]] =
				CheckInput[
					{
						Automatic -> requestSettings$[["LaunchSpecification", "BlockDeviceMappings"]],
						{_Association..} -> OptionValue["BlockDeviceMappings"]
					}
					,
					OptionValue["BlockDeviceMappings"]
					,
					"BlockDeviceMappings"
				];
			(*"EbsOptimized"*)
			requestSettings$[["LaunchSpecification", "EbsOptimized"]] =
				CheckInput[
					{
						Automatic -> requestSettings$[["LaunchSpecification", "EbsOptimized"]],
						(True | False) -> OptionValue["EbsOptimized"]
					}
					,
					OptionValue["EbsOptimized"]
					,
					"EbsOptimized"
				];
			(*"IamInstanceProfile"*)
			requestSettings$[["LaunchSpecification", "IamInstanceProfile"]] =
				CheckInput[
					{
						Automatic -> requestSettings$[["LaunchSpecification", "IamInstanceProfile"]],
						_Association -> OptionValue["IamInstanceProfile"]
					}
					,
					OptionValue["IamInstanceProfile"]
					,
					"IamInstanceProfile"
				];
			(*"ImageId"*)
			requestSettings$[["LaunchSpecification", "ImageId"]] =
				CheckInput[
					{
						Automatic -> requestSettings$[["LaunchSpecification", "ImageId"]],
						_String -> img,
						_EC2Image -> img["ImageId"]
					}
					,
					img
					,
					"AmazonMachineImage"
				];
			(*"InstanceType"*)
			requestSettings$[["LaunchSpecification", "InstanceType"]] =
				CheckInput[
					{
						Automatic -> requestSettings$[["LaunchSpecification", "InstanceType"]],
						_String -> OptionValue["InstanceType"]
					}
					,
					OptionValue["InstanceType"]
					,
					"InstanceType"
				];
			(*"KernelId"*)
			requestSettings$[["LaunchSpecification", "KernelId"]] =
				CheckInput[
					{
						Automatic -> requestSettings$[["LaunchSpecification", "KernelId"]],
						_String -> OptionValue["KernelId"]
					}
					,
					OptionValue["KernelId"]
					,
					"KernelId"
				];
			(*"KeyName"*)
			requestSettings$[["LaunchSpecification", "KeyName"]] =
				CheckInput[
					{
						Automatic ->
							Hold @ If[
								Not@ChoiceDialog[
									"You have chosen not to attach any key pair\
 to the instance(s) request\nYou will not be able to connect to them !\nProceed anyway ?"],
								Abort[],
								requestSettings$[["LaunchSpecification", "KeyName"]]
							],
						_String -> OptionValue["KeyPair"],
						_EC2KeyPairInfo -> OptionValue["KeyPair"]["KeyName"]
					}
					,
					OptionValue["KeyPair"]
					,
					"KeyPair"
				];
			(*"Monitoring"/"Enabled" (spot) -- Monioring (run)*)
			requestSettings$[["LaunchSpecification", "Monitoring"]] =
				CheckInput[
					{
						Automatic -> requestSettings$[["LaunchSpecification", "Monitoring"]],
						_Association -> OptionValue["Monitoring"],
						(True | False) -> <|"Enabled" -> OptionValue["Monitoring"]|>
					}
					,
					OptionValue["Monitoring"]
					,
					"Monitoring"
				];
			(*"Placement"*)
			requestSettings$[["LaunchSpecification", "Placement"]] =
					CheckInput[
						{
							Automatic -> requestSettings$[["LaunchSpecification", "Placement"]],
							_Association -> OptionValue["Placement"]
						}
						,
						OptionValue["Placement"]
						,
						"Placement"
					];
			requestSettings$[["LaunchSpecification", "RamdiskId"]] =
				CheckInput[
					{
						Automatic -> requestSettings$[["LaunchSpecification", "RamdiskId"]],
						_String -> OptionValue["RamdiskId"]
					}
					,
					OptionValue["RamdiskId"]
					,
					"RamdiskId"
				];
			(*"SecurityGroups"*)
			requestSettings$[["LaunchSpecification", "SecurityGroupIds"]] =
				CheckInput[
					{
						Automatic -> requestSettings$[["LaunchSpecification", "SecurityGroups"]],
						{__String} -> OptionValue["SecurityGroups"],
						{__EC2SecurityGroup} -> Map[(#["SecurityGroupId"]&), OptionValue["SecurityGroups"]]
					},
					OptionValue["SecurityGroups"],
					"SecurityGroups"
				];
			(*"NetworkInterfaces"*)
			(*"SubnetId" -> deprecated*)
			(*"UserData"*)
			requestSettings$[["LaunchSpecification", "UserData"]] =
				CheckInput[
					{
						Automatic -> requestSettings$[["LaunchSpecification", "UserData"]],
						_String -> ExportString[OptionValue["UserData"], "Base64"]
					}
					,
					OptionValue["UserData"]
					,
					"UserData"
				];
	requestSettings$
]

(*----------------------------------------------------------------------------*)
(* ::Subsection:: *)
(* On-demand instances : EC2RunInstance *)
(*----------------------------------------------------------------------------*)
PackageExport["EC2CreateInstance"]
SetUsage[EC2CreateInstance,
	"EC2CreateInstance[EC2Image[$$]] launch an on-demand EC2 instance using\
 the machine image EC2Image[$$].
EC2CreateInstance[EC2Image[$$], countspec$] launch countspec% on-demand EC2\
 instances using the machine image EC2Image[$$] and return them in a List.\
 Count specififications can be of the form: exactcount, or {mincount, maxcount}.
EC2CreateInstance[Automatic, $$] will use the default machine image specified]\
 in the launch template specified in \"LaunchTemplate\" option.
EC2CreateInstance accepts the following options:
\"BlockDeviceMappings\" ->
	{
		<|
		\"DeviceName\" -> _String,
		\"VirtualName\" -> _String,
		\"Ebs\"->
			<|
			\"Encrypted\" -> True|False,
			\"DeleteOnTermination\"->True|False,
			\"Iops\" -> _Integer,
			\"SnapshotId\" -> _String,
			\"VolumeSize\"->_Integer,
			\"VolumeType\" -> \"standard\"|\"io1\"|\"gp2\"|\"sc1\"|\"st1\"
			|>,
		\"NoDevice\" -> _String
		|>, $$
	},
\"ClientToken\" -> _String,
\"EbsOptimized\" -> True|False,
\"IamInstanceProfile\" ->
	<|
	\"Arn\" -> _String,
	\"Name\" -> _String
	|>,
\"InstanceMarketOptions\" -> <|
	\"MarketType\" ->\"spot\",
	\"SpotOptions\" -> <|
		\"MaxPrice\" -> _String,
		\"SpotInstanceType\" ->\"one-time\"|\"persistent\",
		\"BlockDurationMinutes\" -> _Integer,
		\"ValidUntil\" -> _DateObject,
		\"InstanceInterruptionBehavior\" ->\"hibernate\"|\"stop\"|\"terminate\"
		|>
	|>
\"InstanceType\" -> _String,
\"KernelId\" -> _String,
\"KeyPair\" -> _String,
\"LaunchTemplate\" ->
	<|
	\"LaunchTemplateId\" -> _String,
	\"LaunchTemplateName\" -> _String,
	\"Version\" -> _String
	|>,
\"Monitoring\" -> <|\"Enabled\"->True|False|>,
\"Placement\" ->
	<|
	\"AvailabilityZone\" -> _String,
	\"GroupName\" -> _String,
	\"Tenancy\" -> \"default\"|\"dedicated\"|\"host\"
	|>,
\"RamdiskId\" -> _String,
\"SecurityGroups\" -> { _String, $$},
\"TagSpecifications\" ->
	{
	<|
	\"ResourceType\" -> \"customer-gateway\"|\"dhcp-options\"|\"image\"|\"instance\"|$$,
	\"Tags\" -> <|_String -> _String, $$|>
	|>, $$
	},
\"UserData\" -> _String
"
]

Options[EC2CreateInstance] =
	Join[
		$CommonRunAndRequestOptions,
		{
			"Client" -> Automatic,
			"DryRun" -> Automatic,
			"InstanceMarketOptions" -> Automatic,
			"LaunchTemplate" -> Automatic,
			"TagSpecifications" -> Automatic(*, "InstanceMarketOptions" -> Automatic*)
		}
	];
EC2CreateInstance[img_, query:OptionsPattern[]] :=
		First[EC2CreateInstance[img, 1, query], $Failed];
EC2CreateInstance[img_EC2Image, countspec_, query:OptionsPattern[]] :=
	EC2CreateInstance[img["ImageId"], countspec, "Client" -> Replace[OptionValue["Client"], Automatic -> img["EC2Client"]], query];
EC2CreateInstance[img_, count_Integer, query:OptionsPattern[]]:=
	EC2CreateInstance[img, {count, count}, query];
EC2CreateInstance[ImageId_ /; MatchQ[ImageId, (_String| Automatic)], countspec:{min_Integer, max_Integer}, query:OptionsPattern[]] := 
	CatchFailureAsMessage @JavaBlock @ Module[
		{
			ec2Client = pickEC2Client[OptionValue["Client"]], requestSettings,
			query$, request, javaMethod, results, instanceList
		},

		javaMethod = "runInstance";
		requestSettings =
			<|
			$iCommonRunAndRequestSettings,
			"InstanceMarketOptions" -> Missing["Optional"],
			"LaunchTemplate" -> Missing["Optional"],
			"TagSpecifications" -> Missing["Optional"]
			|>;

		query$ = Thread[# -> OptionValue[EC2CreateInstance, #]]& @ Map[First, Options[EC2CreateInstance]];
		requestSettings = iCommonRunAndRequestSettings[requestSettings, ImageId, FilterOptions[iCommonRunAndRequestSettings, query$]];
		
		(*MinCount*)
		requestSettings[["MinCount"]] = min;
		(*MaxCount*)
		requestSettings[["MaxCount"]] = max;
		(*LaunchTemplate*)
		requestSettings[["LaunchTemplate"]] =
			CheckInput[
				{
					Automatic -> requestSettings[["LaunchTemplate"]],
					_Association -> OptionValue["LaunchTemplate"]
				}
				,
				OptionValue["LaunchTemplate"]
				,
				"LaunchTemplate"
			];
		requestSettings[["InstanceMarketOptions"]] =
			CheckInput[
				{
					Automatic -> requestSettings[["InstanceMarketOptions"]],
					_Association -> OptionValue["InstanceMarketOptions"]
				}
				,
				OptionValue["InstanceMarketOptions"]
				,
				"InstanceMarketOptions"
			];
		(*TagSpecifications*)
		requestSettings[["TagSpecifications"]] =
			CheckInput[
				{
					Automatic -> requestSettings[["TagSpecifications"]],
					{__Association} -> OptionValue["TagSpecifications"]
				}, 
				OptionValue["TagSpecifications"],
				"TagSpecifications"
			];
		
		(*we do not provide any user custom default option values*)
		requestSettings = optionshelper[{All}, requestSettings, Association@query$, <||>];
		
		request = iEC2CreateInstanceSetOnDemandRequest[requestSettings];
		
		If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[ec2Client["JClient"], request],
			results = safeLibraryInvoke @ ec2Client["JClient"]@runInstances[request];
			instanceList = results@getReservation[]@getInstances[];
			Map[
				EC2Instance[<|"EC2Client" -> ec2Client, "InstanceId" -> #|>]&,
				MapJavaMethod[
					instanceList,
					"com.amazonaws.services.ec2.model.Instance",
					"getInstanceId",
					"java.lang.String"
				]
			]
		]
	];
FailInOtherCases[EC2CreateInstance];

iEC2CreateInstanceSetOnDemandRequest[requestSettings_]:= JavaBlock[
	Module[
		{
			request, LaunchSpecificationSettings, blockDeviceMappings,
			iamInstanceProfil, instanceMarketOptions, launchTemplate,
			securityGroup, placement, tagSpecifications
		},
		
		request = safeLibraryInvoke[JavaNew, "com.amazonaws.services.ec2.model.RunInstancesRequest"];
		
		LaunchSpecificationSettings = requestSettings["LaunchSpecification"];
		
		(* BlockDeviceMappings *)
		blockDeviceMappings = makeBlockDeviceMappings[LaunchSpecificationSettings["BlockDeviceMappings"]];
		If[ nonDefaultQ@blockDeviceMappings,
			safeLibraryInvoke @ request@setBlockDeviceMappings[blockDeviceMappings];
		];
		(* DisableApiTermination *)
		If[ nonDefaultQ@LaunchSpecificationSettings["DisableApiTermination"],
			ThrowFailure["awsunsupported", "DisableApiTermination"]
		];
		(* EbsOptimized *)
		If[ nonDefaultQ@LaunchSpecificationSettings["EbsOptimized"],
			safeLibraryInvoke @ request@setEbsOptimized[LaunchSpecificationSettings["EbsOptimized"]]
		];
		(* ElasticGpuSpecification *)
		If[ nonDefaultQ@LaunchSpecificationSettings["ElasticGpuSpecification"],
			ThrowFailure["awsunsupported", "ElasticGpuSpecification"]
		];
		(* IamInstanceProfile *)
		If[ nonDefaultQ@LaunchSpecificationSettings["IamInstanceProfile"],
			iamInstanceProfil = makeIamInstanceProfile[LaunchSpecificationSettings["IamInstanceProfile"]];
			safeLibraryInvoke @ request@setIamInstanceProfile[iamInstanceProfil]
		];
		(* ImageId *)
		If[ nonDefaultQ@LaunchSpecificationSettings["ImageId"],
		safeLibraryInvoke @ request@setImageId[LaunchSpecificationSettings["ImageId"]];
		];
		(* InstanceInitiatedShutdownBehavior*)
		If[ nonDefaultQ@requestSettings["InstanceInterruptionBehavior"],
			ThrowFailure["awsunsupported", "InstanceInterruptionBehavior"]
		];
		(*InstanceMarketOptions*)
		If[ nonDefaultQ@requestSettings["InstanceMarketOptions"],
			instanceMarketOptions = makeInstanceMarketOption[requestSettings["InstanceMarketOptions"]];
			safeLibraryInvoke @ request@setInstanceMarketOptions[instanceMarketOptions]
		];
		(* InstanceType *)
		If[ nonDefaultQ@LaunchSpecificationSettings["InstanceType"],
			safeLibraryInvoke @ request@setInstanceType[LaunchSpecificationSettings["InstanceType"]];
		];
		(*Ipv6AddressCount*)
		If[ nonDefaultQ@LaunchSpecificationSettings["Ipv6AddressCount"],
			ThrowFailure["awsunsupported", "Ipv6AddressCount"]
		];
		(*Ipv6Addresses*)
		If[ nonDefaultQ@LaunchSpecificationSettings["Ipv6Addresses"],
			ThrowFailure["awsunsupported", "Ipv6Addresses"]
		];
		(*KernelId*)
		If[ nonDefaultQ@LaunchSpecificationSettings["KernelId"],
			safeLibraryInvoke @ request@setKernelId[LaunchSpecificationSettings["KernelId"]];
		];
		(*KeyName*)
		If[ nonDefaultQ@LaunchSpecificationSettings["KeyName"],
			safeLibraryInvoke @ request@setKeyName[LaunchSpecificationSettings["KeyName"]]
		];
		(*LaunchTemplate*)
		If[ nonDefaultQ@requestSettings["LaunchTemplate"],
			launchTemplate = makeLaunchTemplate[requestSettings["LaunchTemplate"]];
			safeLibraryInvoke @ request@setLaunchTemplate[launchTemplate]
		];
		(*MaxCount*)
		safeLibraryInvoke @ request@setMaxCount[MakeJavaObject[requestSettings["MaxCount"]]];
		(*MinCount*)
		safeLibraryInvoke @ request@setMinCount[MakeJavaObject[requestSettings["MinCount"]]];
		(*Monitoring*)
		If[ nonDefaultQ@LaunchSpecificationSettings[["Monitoring"]],
			safeLibraryInvoke @ request@setMonitoring[MakeJavaObject[LaunchSpecificationSettings[["Monitoring", "Enabled"]]]]
		];
		(*NetworkInterfaces*)
		If[ nonDefaultQ@LaunchSpecificationSettings["NetworkInterfaces"],
			ThrowFailure["awsunsupported", "NetworkInterfaces"]
		];
		(*Placement*)
		placement = makePlacement[LaunchSpecificationSettings[["Placement"]], "Instance"];
		safeLibraryInvoke @ request@setPlacement[placement];
		(*PrivateIpAddress*)
		If[ nonDefaultQ@LaunchSpecificationSettings["PrivateIpAddress"],
			ThrowFailure["awsunsupported", "PrivateIpAddress"]
		];
		(*RamdiskId*)
		If[ nonDefaultQ@LaunchSpecificationSettings["RamdiskId"],
			safeLibraryInvoke @ request@setRamdiskId[RamdiskId];
		];
		(*ReadLimit*)
		If[ nonDefaultQ@LaunchSpecificationSettings["ReadLimit"],
			ThrowFailure["awsunsupported", "ReadLimit"]
		];
		(*SecurityGroupIds*)
		securityGroup = makeSecurityGroup[LaunchSpecificationSettings, "GroupId"];
		If[ nonDefaultQ@securityGroup,
			safeLibraryInvoke @ request@setSecurityGroupIds[securityGroup];
		];
		(*SecurityGroups*)
		If[ nonDefaultQ@LaunchSpecificationSettings["SecurityGroups"],
			ThrowFailure["awsunsupported", "SecurityGroups"]
		];
		(*SubnetId*)
		If[ nonDefaultQ@LaunchSpecificationSettings["SubnetId"],
			ThrowFailure["awsunsupported", "SubnetId"]
		];
		(*TagSpecifications*)
		tagSpecifications = makeTagSpecifications[requestSettings["TagSpecifications"]];
		If[ nonDefaultQ@tagSpecifications,
			safeLibraryInvoke @ request@setTagSpecifications[tagSpecifications];
		];
		(*UserData*)
		If[ nonDefaultQ@LaunchSpecificationSettings["UserData"],
			safeLibraryInvoke @ request@setUserData[LaunchSpecificationSettings["UserData"]];
		];
		
		(*returns*)
		request
	]
	];

nonDefaultQ[exp_] := Not@MatchQ[exp, (Automatic | _Missing)];

makeTagSpecifications[None] := None;
makeTagSpecifications[Automatic] := Automatic;
makeTagSpecifications[tagSpecs:{__Association}] := JavaBlock[
	LoadJavaClass["java.util.Arrays"];
	java`util`Arrays`asList @ Map[makeTagSpecifications, tagSpecs]
];
makeTagSpecifications[<|"ResourceType" -> res_, "Tags" -> tags_|>]:=
	JavaBlock @ Module[
		{jTagSpec},
		safeLibraryInvoke[LoadJavaClass, "java.util.Arrays"];
		jTagSpec = safeLibraryInvoke[JavaNew, "com.amazonaws.services.ec2.model.TagSpecification"];
		safeLibraryInvoke @ jTagSpec @ setResourceType[res];
		safeLibraryInvoke @
			jTagSpec @ 
				setTags[
					java`util`Arrays`asList[
						KeyValueMap[
							JavaNew["com.amazonaws.services.ec2.model.Tag", ##]&, 
							tags
						]
					]
				];
		jTagSpec
	];
	
makeLaunchTemplate[specs_] := JavaBlock @ Block[
	{launchTemplate},
	launchTemplate = safeLibraryInvoke[JavaNew, "com.amazonaws.services.ec2.model.LaunchTemplateSpecification"];
	If[nonDefaultQ @ Echo@specs["LaunchTemplateId"],
		safeLibraryInvoke @ launchTemplate@setLaunchTemplateId[specs["LaunchTemplateId"]]
	];
	If[nonDefaultQ @ Echo@specs["LaunchTemplateName"],
		safeLibraryInvoke @ launchTemplate@setLaunchTemplateName[specs["LaunchTemplateName"]]
	];
	If[nonDefaultQ @ Echo@specs["Version"],
		safeLibraryInvoke @ launchTemplate@setVersion[specs["Version"]]
	];
	
	(*returns*)
	launchTemplate
	];
	
makeInstanceMarketOption[specs_] := JavaBlock @ Block[
	{instanceMarketOptions, spotOptions},
	instanceMarketOptions = safeLibraryInvoke[JavaNew, "com.amazonaws.services.ec2.model.InstanceMarketOptionsRequest"];
	If[ nonDefaultQ @ specs["MarketType"],
		safeLibraryInvoke @ instanceMarketOptions@setMarketType[specs["MarketType"]]
	];
	If[ nonDefaultQ @ specs[["SpotOptions"]],
		spotOptions = safeLibraryInvoke[JavaNew, "com.amazonaws.services.ec2.model.SpotMarketOptions"];
		
		If[ nonDefaultQ @ specs[["SpotOptions", "BlockDurationMinutes"]],
		safeLibraryInvoke @ spotOptions@setBlockDurationMinutes[specs[["SpotOptions", "BlockDurationMinutes"]]]
		];
		If[ nonDefaultQ @ specs[["SpotOptions", "InstanceInterruptionBehavior"]],
		safeLibraryInvoke @ spotOptions@setInstanceInterruptionBehavior[specs[["SpotOptions", "InstanceInterruptionBehavior"]]]
		];
		If[ nonDefaultQ @ specs[["SpotOptions", "MaxPrice"]],
		safeLibraryInvoke @ spotOptions@setMaxPrice[specs[["SpotOptions", "MaxPrice"]]]
		];
		If[ nonDefaultQ @ specs[["SpotOptions", "SpotInstanceType"]],
		safeLibraryInvoke @ spotOptions@setSpotInstanceType[specs[["SpotOptions", "SpotInstanceType"]]]
		];
		If[ nonDefaultQ @ specs[["SpotOptions", "ValidUntil"]],
		safeLibraryInvoke @ spotOptions@setValidUntil[specs[["SpotOptions", "ValidUntil"]]]
		];
		
		safeLibraryInvoke @ instanceMarketOptions@setSpotOptions[spotOptions]
	];
	
	(*returns*)
	instanceMarketOptions
]

(*----------------------------------------------------------------------------*)
(* ::Subsection:: *)
(* Spot instances : EC2RequestSpotInstance *)
(*----------------------------------------------------------------------------*)
PackageExport["EC2RequestSpotInstance"]
SetUsage[EC2RequestSpotInstance,
	"EC2RequestSpotInstance[EC2Image[$$]] submits a spot\
 instance request using the machine image EC2Image[$$] and returns the\
  corresponding EC2SpotInstanceRequest[$$].
EC2RequestSpotInstance[EC2Image[$$], count] submits count\
 spot instance requests and return them in a List.
EC2RequestSpotInstance accepts the following options:
\"AvailabilityZoneGroup\" -> _String,
\"BlockDuration\" -> Quantity[_, \"Minutes\"],
\"LaunchGroup\" -> Automatic,
\"ValidFrom\" -> _DateObject,
\"ValidUntil\" -> _DateObject,
\"Persistent\" -> (True | False),
\"RequestTags\" -> Automatic,
\"SpotPrice\" -> Quantity[_, \"$\"],
\"BlockDeviceMappings\" ->
	{
		<|
		\"DeviceName\" -> _String,
		\"VirtualName\" -> _String,
		\"Ebs\"->
			<|
			\"Encrypted\" -> True|False,
			\"DeleteOnTermination\"->True|False,
			\"Iops\" -> _Integer,
			\"SnapshotId\" -> _String,
			\"VolumeSize\"->_Integer,
			\"VolumeType\" -> \"standard\"|\"io1\"|\"gp2\"|\"sc1\"|\"st1\"
			|>,
		\"NoDevice\" -> _String
		|>, $$
	},
\"ClientToken\" -> _String,
\"EbsOptimized\" -> True|False,
\"IamInstanceProfile\" ->
	<|
	\"Arn\" -> _String,
	\"Name\" -> _String
	|>,
\"InstanceType\" -> _String,
\"KernelId\" -> _String,
\"KeyPair\" -> _String,
\"Monitoring\" -> <|\"Enabled\"->True|False|>,
\"Placement\" ->
	<|
	\"AvailabilityZone\" -> _String,
	\"GroupName\" -> _String,
	\"Tenancy\" -> \"default\"|\"dedicated\"|\"host\"
	|>,
\"RamdiskId\" -> _String,
\"SecurityGroups\" -> { _String, $$},
\"UserData\" -> _String,
"
];

Options[EC2RequestSpotInstance]=
	Join[
		$CommonRunAndRequestOptions /. {HoldPattern["InstanceType" -> Automatic] -> "InstanceType" -> "m1.small"}
		,
		{
			"Client" -> Automatic,
			"DryRun" -> Automatic,
			"AvailabilityZoneGroup" -> Automatic, (*user specified name -> same region*)
			"SpotDuration" -> Automatic,
			"LaunchGroup" -> Automatic, (*user specified name -> terminated together*)
			(*Type*)"Persistent" -> Automatic, (*"Type"*)
			"SpotPrice" -> Automatic,
			"ValidFrom" -> Automatic,
			"ValidUntil" -> Automatic
		}
	];
EC2RequestSpotInstance[img_, query:OptionsPattern[]] :=
	First @ EC2RequestSpotInstance[img, 1, query];
EC2RequestSpotInstance[img_EC2Image /; validEC2ImageQ[img], count_Integer, query:OptionsPattern[]]:=
	EC2RequestSpotInstance[img["ImageId"], count, "Client" -> Replace[OptionValue["Client"], Automatic -> img["EC2Client"]], query];
EC2RequestSpotInstance[ImageId_String, count_Integer, query:OptionsPattern[]]:= 
CatchFailureAsMessage @JavaBlock @ Module[
	{
		ec2Client = pickEC2Client[OptionValue["Client"]], requestSettings,
		request, spotInstanceRequestList, query$
	},
	
	requestSettings =
		<|
			$iCommonRunAndRequestSettings,
			"AvailabilityZoneGroup" -> Missing["Optional"],
			"BlockDurationMinutes" -> Missing["Optional"],
			"InstanceCount" -> count,
			"LaunchGroup" -> Missing["Optional"],
			"SpotPrice" -> Missing["Optional"],
			"Type" -> Missing["Optional"],
			"ValidFrom" -> Missing["Optional"],
			"ValidUntil" -> Missing["Optional"]
		|>;
		
		(*check and set common options*)
		query$ = Thread[# -> OptionValue[EC2RequestSpotInstance, #]]& @ Map[First, Options[EC2RequestSpotInstance]];
		requestSettings = iCommonRunAndRequestSettings[requestSettings, ImageId, FilterOptions[iCommonRunAndRequestSettings, query$]];
		
		(*check remaining options*)
		
		(*"AvailabilityZoneGroup"*)
		requestSettings[["AvailabilityZoneGroup"]] =
			CheckInput[
				{
					Automatic -> requestSettings[["AvailabilityZoneGroup"]],
					_String -> OptionValue["AvailabilityZoneGroup"]
				}, 
				OptionValue["AvailabilityZoneGroup"],
				"AvailabilityZoneGroup"
			];
		(*"BlockDurationMinutes"*)
		requestSettings[["BlockDurationMinutes"]] =
			CheckInput[
				{
					Automatic -> requestSettings[["BlockDurationMinutes"]],
					_Quantity -> Ceiling[ UnitConvert[OptionValue["SpotDuration"], "Minutes"] / 60] * 60
				}, 
				OptionValue["SpotDuration"],
				"SpotDuration"
			];
		(*"LaunchGroup"*)
		requestSettings[["LaunchGroup"]] =
			CheckInput[
				{
					Automatic -> requestSettings[["LaunchGroup"]],
					_String -> OptionValue["LaunchGroup"]
				}, 
				OptionValue["LaunchGroup"],
				"LaunchGroup"
			];
		(*"SpotPrice"*)
		requestSettings[["SpotPrice"]] =
			CheckInput[
				{
					Automatic -> requestSettings[["SpotPrice"]],
					_String -> OptionValue["SpotPrice"],
					_Quantity -> (ToString[QuantityMagnitude[UnitConvert[#, "$"]]]&)
				}
				,
				OptionValue["SpotPrice"]
				,
				"SpotPrice"
			];
		(*"Type"*)
		requestSettings[["Type"]] =
			CheckInput[
				{
					Automatic -> Automatic,
					False -> "persistent",
					True -> "one-time"
				}
				,
				OptionValue["Persistent"]
				,
				"Persistent"
			];
		(*"ValidFrom"*)
		requestSettings[["ValidFrom"]] =
			CheckInput[
				{
					Automatic -> requestSettings[["ValidFrom"]],
					_DateObject -> OptionValue["ValidFrom"]
				}
				,
				OptionValue["ValidFrom"]
				,
				"ValidFrom"
			];
		(*"ValidUntil"*)
		requestSettings[["ValidUntil"]] =
			CheckInput[
				{
					Automatic -> requestSettings[["ValidUntil"]],
					_DateObject -> OptionValue["ValidUntil"]
				}
				,
				OptionValue["ValidUntil"]
				,
				"ValidUntil"
			];
		
		
		requestSettings =
			optionshelper[{All}, requestSettings, Association@query$, <||>];
		
		request = safeLibraryInvoke @
			iEC2RequestSpotInstanceSetRequest[requestSettings];

		If[ Replace[OptionValue["DryRun"], Automatic -> False],
			AWSDryRun[ec2Client["JClient"], request],
			
			spotInstanceRequestList = safeLibraryInvoke @
				ec2Client["JClient"]@requestSpotInstances[request]@getSpotInstanceRequests[];
			Map[
				EC2SpotInstanceRequest[<|"EC2Client" -> ec2Client, "SpotInstanceRequestId" -> #|>]&,
				MapJavaMethod[
					spotInstanceRequestList,
					"com.amazonaws.services.ec2.model.SpotInstanceRequest",
					"getSpotInstanceRequestId",
					"java.lang.String"
				]
			]
		]
];
FailInOtherCases[EC2RequestSpotInstance];

optionshelper[pos_, opt_, query_, defaults_] := Association[
	KeyValueMap[optionsetter[pos, ##, query, defaults]&
		,
		opt
	]
]

optionsetter[pos_, key_, value_, query_, defaults_] := key -> Switch[
	value,
	_Association,
		optionshelper[Append[pos, key], value, query, defaults],
	_Missing,
		setValue[pos, First@value, key, query, defaults],
	_, value
]

setValue::required = "Mandatory value for `key` neither found in user default nor provided."
setValue[pos_, behavior_, key_, query_, defaults_] :=
	Module[
		{opts, value},
		
		value = Lookup[query, key];
		
		If[ MatchQ[value, _Missing],
			opts = Fold[nextLevel, defaults, pos];
			value = Lookup[opts, key];
			
			If[ MatchQ[value, _Missing],
				If[ behavior === "Required",
				ThrowFailure["awsoptrequired", ToString[key], pos]
				,
				value = Automatic
				]
			]
	];
	
	value
];

nextLevel[asso_Association, partspec_] := If[ partspec === All, asso, Lookup[asso, partspec, <||>]];


iEC2RequestSpotInstanceSetRequest[requestSettings_]:=
	JavaBlock @ Module[{request, launchSpecification},
			LoadJavaClass["java.lang.Integer"];
			
			request = JavaNew["com.amazonaws.services.ec2.model.RequestSpotInstancesRequest"];
			
			If[ nonDefaultQ @ requestSettings["AvailabilityZoneGroup"],
				request	@ setAvailabilityZoneGroup[requestSettings["AvailabilityZoneGroup"]];
			];
			
			If[ nonDefaultQ @ requestSettings["BlockDurationMinutes"],
				request	@ setBlockDurationMinutes[requestSettings["BlockDurationMinutes"]];
			];
			
			request @ setInstanceCount[ReturnAsJavaObject[Integer`valueOf[requestSettings["InstanceCount"]]]];
			
			If[ nonDefaultQ @ requestSettings["LaunchGroup"],
				request	@ setLaunchGroup[requestSettings["LaunchGroup"]];
			];
			
			launchSpecification = makeLaunchSpecification[requestSettings["LaunchSpecification"]];
			request	@ setLaunchSpecification[launchSpecification];
			
			If[ nonDefaultQ @ requestSettings["SpotPrice"],
				request @ setSpotPrice[requestSettings[["SpotPrice"]]];
			];
			
			If[ nonDefaultQ @ requestSettings["Type"],
				request @ setType[requestSettings[["Type"]]];
			];
			
			If[ nonDefaultQ @ requestSettings["ValidFrom"],
				request @ setValidFrom[ toJavaDate[requestSettings["ValidFrom"]] ];
			];
			
			If[ nonDefaultQ @ requestSettings["ValidUntil"],
				request @ setValidUntil[ toJavaDate[requestSettings["ValidUntil"]] ]
			];
			
			(*returns*)
			request
	];


makeLaunchSpecification[LaunchSpecificationSettings_] :=
	JavaBlock[
		Module[
			{launchSpecification, securityGroup, iamInstanceProfile, blockDeviceMappings, placement},
			launchSpecification = JavaNew["com.amazonaws.services.ec2.model.LaunchSpecification"];
			launchSpecification @ setImageId[LaunchSpecificationSettings["ImageId"]];
			
			If[ nonDefaultQ @ LaunchSpecificationSettings["InstanceType"],
				launchSpecification @ setInstanceType[LaunchSpecificationSettings["InstanceType"]];
			];
			
			If[ nonDefaultQ @ LaunchSpecificationSettings["KeyName"],
				launchSpecification @ setKeyName[LaunchSpecificationSettings["KeyName"]]
			];
			
			If[ nonDefaultQ @ LaunchSpecificationSettings["SecurityGroups"],
				ThrowFailure["awsunsupported", "SecurityGroups"];
			];
			
			If[ nonDefaultQ @ LaunchSpecificationSettings["AddressingType"],
				ThrowFailure["awsunsupported", "AddressingType"];
			];
			
			If[ nonDefaultQ @ LaunchSpecificationSettings["IamInstanceProfile"],
				iamInstanceProfile = makeIamInstanceProfile[LaunchSpecificationSettings["IamInstanceProfile"]];
				launchSpecification @ setIamInstanceProfile[iamInstanceProfile]
			];
			
			If[ nonDefaultQ @ LaunchSpecificationSettings["Monitoring"],
				ThrowFailure["awsunsupported", "Monitoring"];
			];
			
			If[ nonDefaultQ @ LaunchSpecificationSettings["NetworkInterfaces"],
				ThrowFailure["awsunsupported", "NetworkInterfaces"];
			];
			
			If[ nonDefaultQ @ LaunchSpecificationSettings["RamdiskId"],
				ThrowFailure["awsunsupported", "RamdiskId"];
			];
			
			If[ nonDefaultQ @ LaunchSpecificationSettings["SubnetId"],
				ThrowFailure["awsunsupported", "SubnetId"];
			];
			
			If[ nonDefaultQ @ LaunchSpecificationSettings["EbsOptimized"],
				ThrowFailure["awsunsupported", "EbsOptimized"];
			];
			
			placement = makePlacement[LaunchSpecificationSettings[["Placement"]], "Spot"];
			launchSpecification @ setPlacement[placement];
			
			securityGroup = makeSecurityGroup[LaunchSpecificationSettings, "GroupIdentifier"];
			If[ nonDefaultQ @ securityGroup,
				launchSpecification @ setAllSecurityGroups[securityGroup];
			];
			
			blockDeviceMappings = makeBlockDeviceMappings[LaunchSpecificationSettings["BlockDeviceMappings"]];
			If[ nonDefaultQ @ blockDeviceMappings,
				launchSpecification @ setBlockDeviceMappings[blockDeviceMappings];
			];
			
			If[ nonDefaultQ @ LaunchSpecificationSettings["UserData"],
				launchSpecification @ setUserData[LaunchSpecificationSettings["UserData"]];
			];
			
			(*returns*)
			launchSpecification
		]
	]


makeIamInstanceProfile[iamInstanceProfile_] := JavaBlock @ Block[
	{iamInstanceProfileSpecification},
	iamInstanceProfileSpecification =
		safeLibraryInvoke[
			JavaNew,
			"com.amazonaws.services.ec2.model.IamInstanceProfileSpecification"
		];
	
	If[ nonDefaultQ @ iamInstanceProfile["Arn"],
		iamInstanceProfileSpecification@setArn[iamInstanceProfile["Arn"]]
	];
	
	If[ nonDefaultQ @ iamInstanceProfile["Name"],
		iamInstanceProfileSpecification@setName[iamInstanceProfile["Name"]]
	];
	
	Return[iamInstanceProfileSpecification]
]


makePlacement[PlacementSettings_, type_/;MatchQ[type, ("Spot"|"Instance")]] :=
JavaBlock[
		Module[
			{placement},
			
			If[ type === "Spot"
				,
				placement = JavaNew["com.amazonaws.services.ec2.model.SpotPlacement"];
				,
				placement = JavaNew["com.amazonaws.services.ec2.model.Placement"];
			];
			
			If[ nonDefaultQ @ PlacementSettings,
				If[ nonDefaultQ @ PlacementSettings["AvailabilityZone"],
					placement@ setAvailabilityZone[PlacementSettings["AvailabilityZone"]];
				];
				
				If[ nonDefaultQ @ PlacementSettings["GroupName"],
					placement@ setGroupName[PlacementSettings["GroupName"]];
				];
				
				If[ nonDefaultQ @ PlacementSettings["Tenancy"],
					placement@ setTenancy[PlacementSettings["Tenancy"]];
				];
			];
			
			placement
		]
];


makeBlockDeviceMappings[BlockDeviceMappingsSettings_] :=
	JavaBlock[
		Module[
			{blockDeviceMappings, targeti, mappings, i},
			(*
				************************** Add the block device mappings ************************
				ArrayList<BlockDeviceMapping> blockList = new ArrayList<BlockDeviceMapping>();
				blockList.add(blockDeviceMapping);
				
				Constraints (size GiB): 
					1-16384 for General Purpose SSD (gp2 ),
					4-16384 for Provisioned IOPS SSD (io1 ),
					500-16384 for Throughput Optimized HDD (st1 ),
					500-16384 for Cold HDD (sc1 ), and 1-1024 for Magnetic (standard ) volumes.
					If you specify a snapshot, the volume size must be equal to or larger than the snapshot size.
					
				target=
					{
						<|
						"DeviceName"->"string", (*The device name exposed to the instance "/dev/sda0" , "/dev/sda1" - "/dev/xvdb", "/dev/xvdc" ...*)
						"VirtualName"->"string", (*ephemeralN. Instance store volumes are numbered starting from 0. An instance type with 2 available instance store volumes can specify mappings for ephemeral0 and ephemeral1 .The number of available instance store volumes depends on the instance type.*)
						"Ebs"->
							{
								"Encrypted"->True|False, (req -> userDefault)
								"DeleteOnTermination"->True|False, (req -> userDefault)
								"InputOutputPerSecond"->123, (*io1(req: 100-20000) only*)
								"SnapshotId"->"string",
								"VolumeType"->"standard"|"io1"|"gp2"|"sc1"|"st1" (req -> userDefault)
								"VolumeSize"->123, (* GiB *) 1-1024|4-16384|1-16384|500-16384|500-16384 >= "SnapshotId" size (default) or (req)
							},
						"NoDevice"->"string" (* Suppresses the specified device included in the block device mappings of the AMI *)
						|>
			*)
			If[ Length[BlockDeviceMappingsSettings] != 0 && BlockDeviceMappingsSettings =!= Automatic,
				
				blockDeviceMappings = JavaNew["java.util.ArrayList"];
				mappings = Table[{JavaNew["com.amazonaws.services.ec2.model.BlockDeviceMapping"], JavaNew["com.amazonaws.services.ec2.model.EbsBlockDevice"]}, Length[BlockDeviceMappingsSettings]];
				
				Do[ 
					targeti = BlockDeviceMappingsSettings[[i]];
					
					(* ebs *)
					mappings[[i, 2]] @ setDeleteOnTermination[ MakeJavaObject[targeti[["Ebs", "DeleteOnTermination"]]] ];
					mappings[[i, 2]] @ setVolumeType[ targeti[["Ebs", "VolumeType"]] ];
					
					If[ nonDefaultQ @ targeti[["Ebs", "Encrypted"]],
						mappings[[i, 2]] @ setEncrypted[ targeti[["Ebs", "Encrypted"]] ]
					];
					
					If[ nonDefaultQ @ targeti[["Ebs","SnapshotId"]],
						mappings[[i, 2]] @ setSnapshotId[ targeti[["Ebs","SnapshotId"]] ];
					];
					
					If[ nonDefaultQ @ targeti[["Ebs","VolumeSize"]],
						mappings[[i, 2]] @ setVolumeSize[ MakeJavaObject[targeti[["Ebs","VolumeSize"]]] ];
					];
					
					If[ nonDefaultQ @ targeti[["Ebs","InputOutputPerSecond"]],
						mappings[[i, 2]] @ setIops[ targeti[["Ebs","InputOutputPerSecond"]] ];
					];
					
					mappings[[i, 1]] @ setEbs[ mappings[[i, 2]] ];
					
					(* virtual *)
					
					mappings[[i, 1]] @ setDeviceName[ targeti["DeviceName"] ];
					If[ nonDefaultQ @ targeti[["VirtualName"]],
						mappings[[i, 1]] @ setVirtualName[ targeti[["VirtualName"]] ];
					];
					If[ nonDefaultQ @ targeti[["NoDevice"]],
						mappings[[i, 1]] @ setNoDevice[ targeti[["NoDevice"]] ];
					];
					
					blockDeviceMappings @ add[ mappings[[i, 1]] ]
					,
					{i, Length[BlockDeviceMappingsSettings]}
				]
				,
				blockDeviceMappings = Automatic
			];
			(*returns*)
			blockDeviceMappings
		]
	]
	
makeSecurityGroup[query_, type_] :=
	JavaBlock @ Module[
		{securityGroup, userQuery = query[["SecurityGroupIds"]]},
		
		If[ userQuery === Automatic,
			securityGroup = Automatic;
			,
			If[ MatchQ[ userQuery, {__String}]
				, 
				securityGroup = JavaNew["java.util.ArrayList"];
				LoadJavaClass["java.util.Arrays"];
				Switch[ type,
					"GroupId",
						securityGroup@addAll@java`util`Arrays`asList@
							MakeJavaObject[userQuery],
					"GroupIdentifier",
						securityGroup@addAll@java`util`Arrays`asList@
							Map[
								JavaNew["com.amazonaws.services.ec2.model.GroupIdentifier"]
									@withGroupId[#] &,
								userQuery
							]
				]
				,
				ThrowFailure["awswrongopt", "SecurityGroupIds", userQuery]
			]
		];
		(*returns*)
		securityGroup
	];


(*----------------------------------------------------------------------------*)
(* ::Section:: *)
(* Filtering *)
(*----------------------------------------------------------------------------*)
(*http://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeVolumes.html*)
(*http://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeImages.html*)
(*http://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeInstances.html*)
(*http://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeSnapshots.html*)
(*http://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeSpotInstanceRequests.html*)
 
createFilter[filterName_String, filteredvalues_/;AtomQ[filteredvalues]] :=
	createFilter[filterName, {Replace[filteredvalues, {True -> "true", False -> "false"}]}];

createFilter[filterName_String, filteredvalues_List]:=
JavaBlock[
		LoadJavaClass["java.util.Arrays"];
		safeLibraryInvoke[JavaNew, "com.amazonaws.services.ec2.model.Filter", filterName, java`util`Arrays`asList[MakeJavaObject[filteredvalues]]]
];


(*----------------------------------------------------------------------------*)
(*ToDo: this is currently a hack !!*)
SetUsage[
	ec2Describe,
	"(! meant for debugging only !)
	ec2Describe[ec2Client, \"object\"] list all availables \"object\" and their properties"
];

$validec2Describe = {
	"AccountAttributes", "Addresses", "AvailabilityZones", "BundleTasks",
	"ClassicLinkInstances", "ConversionTasks", "CustomerGateways", "DhcpOptions",
	"ExportTasks", "FlowLogs", "Hosts", "IdFormat",
	"ImportImageTasks", "ImportSnapshotTasks", "Instances",
	"InstanceStatus", "InstanceStatus", "InternetGateways", "KeyPairs",
	"MovingAddresses", "MovingAddresses", "NetworkAcls", "NetworkInterfaces",
	"PlacementGroups", "PrefixLists", "Regions", "ReservedInstances",
	"RouteTables", "SecurityGroups", "SpotFleetRequests", "SpotInstanceRequests",
	"Subnets", "Tags", "Volumes", "VolumeStatus", "VpcClassicLink", "VpcEndpoints",
	"VpcEndpointServices", "VpcPeeringConnections", "Vpcs", "VpnConnections",
	"VpnGateways"
	};
	
$argec2Describe = {
	"EgressOnlyInternetGateways", "HostReservationOfferings", "HostReservations", "IamInstanceProfileAssociations",
	"IdentityIdFormat", "ImageAttribute", "InstanceAttribute", "NatGateways",
	"NetworkInterfaceAttribute", "ScheduledInstanceAvailability", "ScheduledInstances", "SecurityGroupReferences",
	"SnapshotAttribute", "SpotFleetInstances", "SpotFleetRequestHistory", "StaleSecurityGroups",
	"VolumeAttribute", "VolumesModifications", "VolumeStatus", "VpcAttribute",
	"VpcClassicLinkDnsSupport"
};

$tolongec2Describe = {"Images", "Snapshots", "SpotDatafeedSubscription"};
$forbiddenec2Describe = {"ReservedInstancesListings", "ReservedInstancesModifications", "ReservedInstancesOfferings"};

$changingec2Describe = <|
	"Instances" -> "Reservations", "ClassicLinkInstances" -> "Instances", "IdFormat" -> "Statuses",
	"InstanceStatus" -> "InstanceStatuses", "VpcClassicLinkDnsSupport" -> "Vpcs", "VpcEndpointServices" -> "ServiceNames",
	"MovingAddresses" -> "MovingAddressStatuses", "SpotFleetRequests" -> "SpotFleetRequestConfigs","VpcClassicLink" -> "Vpcs",
	"VolumeStatus" -> "VolumeStatuses"
|>;

ec2Describe::notimpl = "The following method is not yet available try \"MyImages\", \"MySnapshots\"... instead";
ec2Describe::wrongObj = "The object cannot be described";
ec2Describe[EC2Client[<|"JClient"->obj_|>], object_String]:= 
JavaBlock[
	Module[
		{object$},
		If[ MemberQ[$validec2Describe, object],
				
				object$ = Lookup[$changingec2Describe, object, object];
				Dataset@
					iCollectionParser[
						ReleaseHold[
							Hold[obj][ToExpression["describe"<>object<>"[]@get"<>object$<>"[]@toString[]"]]
						]
					]
			,
				If[ MemberQ[Join[$argec2Describe,$tolongec2Describe], object],
					Failure["Not supported", <|"MessageTemplate" -> ec2Describe::notimpl|>]
					,
					Failure["Wrong Object", <|"MessageTemplate" -> ec2Describe::wrongObj|>]
				]
		]
	]
]

ec2Describe[EC2Client[<|"JClient"->obj_|>], "MyImages"] := JavaBlock[
	Dataset[
		iCollectionParser[
		obj @ describeImages[
				JavaNew["com.amazonaws.services.ec2.model.DescribeImagesRequest"]@withOwners[{"self"}]
			]@toString[]
		]["Images"]
	]
]

ec2Describe[EC2Client[<|"JClient"->obj_|>], "MySnapshots"] := JavaBlock[
	Dataset[
		iCollectionParser[
		obj @ describeSnapshots[
				JavaNew["com.amazonaws.services.ec2.model.DescribeSnapshotsRequest"]@withOwnerIds[{"self"}]
			] @ toString[]
		]["Snapshots"]
	]
]


iCollectionParser[str_String]:=
	ReplaceRepeated[#, {r__Rule} :> <|r|>]& @
	ReplaceAll[{""} :> {}]@
	ToExpression@
	StringReplace[",|>" -> "|>"]@
	StringReplace[",}" -> "}"]@
	StringReplace["->," -> "->Null,"]@
	StringReplace["->," -> "->Null,"]@
	StringReplace["<|" ~~ e : Except[("|" | "<" | "}")] :> "<|\"" <> e]@
	StringReplace["{" ~~ e : Except[("{" | "<")] :> "{\"" <> e]@
	StringReplace[e : Except[("|" | ">" | "}" | ",")] ~~ "|>" :> e <> "\"|>"]@
	StringReplace[e : Except[("|" | ">" | "}" | ",")] ~~ "}" :> e <> "\"}"]@
	StringReplace["," ~~ e : Except[("<" | "{" | "}" | "|")] :> ",\"" <> e]@
	StringReplace[e : Except[("}" | ">")] ~~ "," :> e <> "\","]@
	StringReplace["->" ~~ e : Except[("{" | "<" | ",")] :> "->\"" <> e]@
	StringReplace["->" -> "\"->"]@
	StringReplace[{"[" -> "{", "]" -> "}"}]@
	StringReplace[{"[]" -> "{}", "[{" -> "{<|", "}]" -> "|>}", "},{" -> "|>,<|"}]@
	StringDelete[" "]@
	StringReplace["\"" -> "\\\""]@
	StringReplace[": " ~~ p : Except[("," | "[" | "{")] ... ~~t : ("," | "[" | "{"|"}") :> "->"<>p<>t] @ str;
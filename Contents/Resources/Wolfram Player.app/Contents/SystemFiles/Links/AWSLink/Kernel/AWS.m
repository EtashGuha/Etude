(*******************************************************************************

Region functions

*******************************************************************************)

Package["AWSLink`"]

PackageImport["JLink`"]
PackageImport["GeneralUtilities`"]

(*----------------------------------------------------------------------------*)
PackageExport["$AWSDefaultClients"]
$AWSDefaultClients = <|"ec2" -> None|>;

(*----------------------------------------------------------------------------*)
PackageExport["AWSClientConnect"]
SetUsage[AWSClientConnect,
"AWSClientConnect[\"service\"] initiate a connexion to an AWS service (\"ec2\"...)\
 represented by a client object using the default credentials."
]
AWSClientConnect::"unk" = "Unsupported service `service`. Available service is \"ec2\".";
Options[AWSClientConnect] =
	{
		(*if Region is incorrect, will return US_Standard - glups!*)
		"Region" -> Automatic
	};
AWSClientConnect[service_String, opt:OptionsPattern[]] := CatchFailureAsMessage @ (
	(*JVMArguments -> "-Xmx" <> ToString@Max[512, Floor[1/4*$SystemMemory/10^6]] <> "m"*)
	InstallJava[];
	Switch[ToLowerCase[service],
		"ec2",
			EC2ClientConnect[opt],
		_,
			Failure[
				"arginv", 
				<|
				"MessageTemplate" :> AWSClientConnect::"unk",
				"MessageParameters" ->
					<|
					"service" -> service
					|>
				|>
			]
	]
)
FailInOtherCases[AWSClientConnect];

(*----------------------------------------------------------------------------*)
PackageExport["AWSGetInformation"]
SetUsage[AWSGetInformation, "AWSGetInformation[obj$] returns an Association containing information on obj$"];

(*----------------------------------------------------------------------------*)
PackageExport["AWSRegionNames"]
SetUsage[AWSRegionNames,
"AWSRegionNames[] returns the list containing the string names of all AWS regions."
]

AWSRegionNames[] := CatchFailureAsMessage @ JavaBlock[
	Module[
		{regionObjects},
		LoadJavaClass["com.amazonaws.regions.Regions"];
		regionObjects = com`amazonaws`regions`Regions`values[];
		(#@getName[]) & /@ regionObjects
	]
];
FailInOtherCases[AWSRegionNames];
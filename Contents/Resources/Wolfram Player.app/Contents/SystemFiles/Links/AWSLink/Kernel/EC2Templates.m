(* Wolfram Language package *)
Package["AWSLink`"]

(*******************************************************
The following give automatic acces to the listed properties
of the object throug given method to build AWSLink Object.

Template syntax:
<|
"propName" -> "accessMethod[]" -> postprocessingFunction
|>

you  should nest templates to avoid duplicates.
<|
...
"propName" ->
	Missing[
		"NextClassAccessMethod[]",
		NextClassTemplate
	]
...
|>

Map over a List<NextClass> to obtain a mathematia counterpart using:
<|
...
"propName" ->
	Missing[
		"NextCollectionAccessMethod[]",
		{ NextClassTemplate }
	]
...

translate dictionnaris of a Map<NextClass1, NextClass2> to obtain a mathematia counterpart using:
<|
...
"propName" ->
	Missing[
		"NextMapAccessMethod[]",
		"NextClass1AccessMethod[]" ->  "NextClass2AccessMethod[]"
	]
...
or

 ******************************************************)

(*******************************************************
 *  
 * EC2:
 *  
 ******************************************************)

(*******************************************************

*******************************************************)
ec2StateTemplate =
<|
"Code" -> "getCode[]",
"Name" -> "getName[]"
|>;


(*******************************************************

*******************************************************)
(*
"com.amazonaws.services.ec2.model.IamInstanceProfileSpecification "
*)
ec2IamInstanceProfileSpecificationTemplate = 
<|
"Arn" -> "getArn[]",
"Name" -> "getName[]"
|>;


(*******************************************************

*******************************************************)
ec2StatusTemplate =
<|
"Code" -> "getCode[]",
"UpdateTime" -> "getUpdateTime[]",
"Message" -> "getMessage[]"
|>


(*******************************************************

*******************************************************)
ec2MonitoringTemplate =
<|
"State" -> "getState[]"
|>;

(*
"com.amazonaws.services.ec2.model.ElasticGpuSpecification "
*)
ec2ElasticGpuSpecificationTemplate = 
<|
"Type" -> "getType[]"
|>;

(*******************************************************

*******************************************************)
PackageScope["EC2AvailabilityZonesTemplate"]
EC2AvailabilityZonesTemplate =
<|
"ZoneName" -> "getZoneName[]",
"State" -> "getState[]",
"RegionName" -> "getRegionName[]",
"Messages" -> "getMessages[]"
|>;


(*******************************************************

*******************************************************)
PackageScope["EC2RegionTemplate"]
EC2RegionTemplate =
<|
"RegionName" -> "getRegionName[]",
"Endpoint" -> "getEndpoint[]"
|>;


(*******************************************************

*******************************************************)
ec2NetworkInterfaceAttachmentTemplate =
<|
"AttachmentId"->"getAttachmentId[]",
"AttachTime"->"getAttachTime[]",
"DeleteOnTermination"->"getDeleteOnTermination[]",
"DeviceIndex"->"getDeviceIndex[]",
"Status"->"getStatus[]"
(*
"InstanceId"->"getInstanceId[]",
"InstanceOwnerId"->"getInstanceOwnerId[]",
*)
|>;


(*******************************************************

*******************************************************)
ec2NetworkInterfaceAssociationTemplate =
<|
(*
"AllocationId"->"getAllocationId[]",
"AssociationId"->"getAssociationId[]",
*)
"IpOwnerId"->"getIpOwnerId[]",
"PublicDnsName"->"getPublicDnsName[]",
"PublicIp"->"getPublicIp[]"
|>

(*
"com.amazonaws.services.ec2.model.Placement "
*)
ec2PlacementTemplate = 
<|
"Affinity" -> "getAffinity[]",
"AvailabilityZone" -> "getAvailabilityZone[]",
"GroupName" -> "getGroupName[]",
"HostId" -> "getHostId[]",
"SpreadDomain" -> "getSpreadDomain[]",
"Tenancy" -> "getTenancy[]"
|>


(*******************************************************

*******************************************************)
ec2SpotPlacementTemplate =
<|
"AvailabilityZone" -> "getAvailabilityZone[]",
"GroupName" -> "getGroupName[]",
"Tenancy" -> "getTenancy[]"
|>;


(*******************************************************
com.amazonaws.services.ec2.model.GetConsoleOutputResult
*******************************************************)
PackageScope["EC2GetConsoleOutputResultTemplate"]
EC2GetConsoleOutputResultTemplate =
<|
"DecodedOutput" -> "getDecodedOutput[]", 
"InstanceId" -> "getInstanceId[]", 
"Output" -> "getOutput[]", 
"Timestamp" -> "getTimestamp[]"
|>;

(*******************************************************
com.amazonaws.services.ec2.model.GetConsoleScreenshotRequest
*******************************************************)
PackageScope["EC2GetConsoleScreenshotResultTemplate"]
EC2GetConsoleScreenshotResultTemplate =
<|
"ImageData" -> "getImageData[]",
"InstanceId" -> "getInstanceId[]"
|>;


(*
"com.amazonaws.services.ec2.model.Tag "
*)
EC2TagTemplate =
<|
"Key" -> "getKey[]",
"Value" -> "getValue[]"
|> -> Function[{asso}, <|asso["Key"] -> asso["Value"]|>];

(*
"com.amazonaws.services.ec2.model.TagSpecification "
*)
ec2TagSpecificationTemplate = 
<|
"ResourceType" -> "getResourceType[]",
"Tags" ->
	Missing[
		"getTags[]",
		{ EC2TagTemplate }
	]
|>;


(*******************************************************
   com.amazonaws.services.ec2.model.CreateTagsRequest
*******************************************************)
ec2CreateTagsRequestTemplate =
<|
"CustomQueryParameters" -> 
	Missing[
		"getCustomQueryParameters[]",
		"keySet[] @ toArray[]" ->
			Missing[
				"values[]", 
				{ "toArray[]" }
			]
	],
"CustomRequestHeaders" ->
	Missing[
		"getCustomRequestHeaders[]",
		"keySet[] @ toArray[]" -> "values[] @ toArray[]"
	], 
"ReadLimit" -> "getReadLimit[]", 
"Resources" -> "getResources[]@toArray[]",
"SdkClientExecutionTimeout" -> "getSdkClientExecutionTimeout[]", 
"SdkRequestTimeout" -> "getSdkRequestTimeout[]", 
"Tags" -> 
	Missing[
		"getTags[]",
		{
			EC2TagTemplate
		}
	]
|>;

(*******************************************************
com.amazonaws.services.ec2.model.GroupIdentifier
*******************************************************)
ec2GroupIdentifierTemplate =
<|
"GroupName" -> "getGroupName[]",
"GroupId" -> "getGroupId[]"
|>;

(*******************************************************
com.amazonaws.services.ec2.model.Snapshot
*******************************************************)
PackageScope["EC2SnapshotTemplate"]
EC2SnapshotTemplate =
<|
"DataEncryptionKeyId" -> "getDataEncryptionKeyId[]",
"Description" -> "getDescription[]",
"Encrypted" -> "getEncrypted[]",
"KmsKeyId" -> "getKmsKeyId[]",
"OwnerAlias" -> "getOwnerAlias[]",
"OwnerId" -> "getOwnerId[]",
"Progress" -> "getProgress[]",
"SnapshotId" -> "getSnapshotId[]",
"StartTime" -> "getStartTime[]",
"State" -> "getState[]",
"StateMessage" -> "getStateMessage[]",
"Tags" -> Missing[
		"getTags[]",
		{
			EC2TagTemplate
		}
	],
"VolumeId" -> "getVolumeId[]",
"VolumeSize" -> "getVolumeSize[]"
|>;

(*
"com.amazonaws.services.ec2.model.InstanceIpv6Address "
*)
ec2InstanceIpv6AddressTemplate = 
<|
"Ipv6Address" -> "getIpv6Address[]"
|>

(*******************************************************
com.amazonaws.services.ec2.model.Ipv6Range
com.amazonaws.services.ec2.model.IpRange
com.amazonaws.services.ec2.model.PrefixListId
com.amazonaws.services.ec2.model.UserIdGroupPair
com.amazonaws.services.ec2.model.IpPermission
*******************************************************)
ec2Ipv6RangeTemplate=
<|"CidrIpv6"->"getCidrIpv6[]"|>;

ec2Ipv4RangeTemplate=
<|"CidrIp"->"getCidrIp[]"|>;

ec2PrefixListIdTemplate=
<|"PrefixListId"->"getPrefixListId[]"|>;

ec2UserIdGroupPairTemplate=
<|
"GroupId"->"getGroupId[]",
"GroupName"->"getGroupName[]",
"PeeringStatus"->"getPeeringStatus[]",
"UserId"->"getUserId[]",
"VpcId"->"getVpcId[]",
"VpcPeeringConnectionId"->"getVpcPeeringConnectionId[]"
|>;

ec2IpPermissionTemplate=
<|
"FromPort"->"getFromPort[]",
"IpProtocol"->"getIpProtocol[]",
"Ipv4Ranges"->
	Missing[
		"getIpv4Ranges[]",
		{
			ec2Ipv4RangeTemplate
		}
	],
"Ipv6Ranges"->
	Missing[
		"getIpv6Ranges[]",
		{
			ec2Ipv6RangeTemplate
		}
	],
"PrefixListIds"->
	Missing[
		"getPrefixListIds[]",
		{
			ec2PrefixListIdTemplate
		}
	],
"ToPort"->"getToPort[]",
"UserIdGroupPairs"->
	Missing[
		"getUserIdGroupPairs[]",
		{
			ec2UserIdGroupPairTemplate
		}
	]
|>;

(*******************************************************

*******************************************************)
ec2IpPermissionsEgressTemplate =
<|
"IpProtocol"->"getIpProtocol[]",
"UserIdGroupPairs"->"getUserIdGroupPairs[]",
"Ipv6Ranges"->"getIpv6Ranges[]",
"PrefixListIds"->"getPrefixListIds[]",
"Ipv4Ranges"->
	Missing[ "getIpv4Ranges[]",
		{
			ec2Ipv4RangeTemplate
		}
	]
|>;

(*******************************************************
com.amazonaws.services.ec2.model.SecurityGroup
*******************************************************)
PackageScope["EC2SecurityGroupTemplate"]
EC2SecurityGroupTemplate =
<|
"OwnerId" -> "getOwnerId[]",
"GroupName" -> "getGroupName[]",
"GroupId" -> "getGroupId[]",
"Description" -> "getDescription[]",
"IpPermissions" -> 
	Missing[
		"getIpPermissions[]"
		,
		{
		 	ec2IpPermissionTemplate
		}
	],
"IpPermissionsEgress" ->
	Missing[
		"getIpPermissionsEgress[]"
		,
		{
			ec2IpPermissionsEgressTemplate
		}
	],
"VpcId" -> "getVpcId[]",
"Tags"->
	Missing[
		"getTags[]",
		{
		EC2TagTemplate
		}
	]
|>

(*******************************************************

*******************************************************)
ec2GroupTemplate =
<|
"GroupName" -> "getGroupName[]",
"GroupId" -> "getGroupId[]"
|>;


(*******************************************************

*******************************************************)
(*
"com.amazonaws.services.ec2.model.KeyPair "
*)
PackageScope["EC2KeyPairTemplate"]
EC2KeyPairTemplate = 
<|
"KeyFingerprint" -> "getKeyFingerprint[]",
"KeyMaterial" -> "getKeyMaterial[]",
"KeyName" -> "getKeyName[]"
|>;

(*******************************************************

*******************************************************)
PackageScope["EC2KeyPairInfoTemplate"]
EC2KeyPairInfoTemplate =
<|
"KeyName" -> "getKeyName[]",
"KeyFingerprint" -> "getKeyFingerprint[]"
|>;


(*
"com.amazonaws.services.ec2.model.PrivateIpAddressSpecification "
*)
ec2PrivateIpAddressSpecificationTemplate = 
<|
"Primary" -> "getPrimary[]",
"PrivateIpAddress" -> "getPrivateIpAddress[]"
|>

(*******************************************************

*******************************************************)
ec2PrivateIpAddressesTemplate =
<|
"PrivateIpAddress" -> "getPrivateIpAddress[]",
"PrivateDnsName" -> "getPrivateDnsName[]",
"Primary" -> "getPrimary[]",
"Association" -> Missing["getAssociation[]" , ec2NetworkInterfaceAssociationTemplate]
|>;

(*******************************************************

*******************************************************)
ec2InstanceNetworkInterfacesTemplate =
<|
	"Association" -> 
		Missing["getAssociation[]", ec2NetworkInterfaceAssociationTemplate],
	"Attachment" -> 
		Missing["getAttachment[]", ec2NetworkInterfaceAttachmentTemplate],
	"Description"->"getDescription[]",
	"Groups"-> 
		Missing[
			"getGroups[]",
			{ ec2GroupTemplate }
		],
	"Ipv6Addresses"->
		Missing[
			"getIpv6Addresses[]",
			{
			<|"Ipv6Address"->"getIpv6Address[]"|>
			}
		],
	"MacAddress"->"getMacAddress[]",
	"NetworkInterfaceId"->"getNetworkInterfaceId[]",
	"OwnerId"->"getOwnerId[]",
	"PrivateDnsName"->"getPrivateDnsName[]",
	"PrivateIpAddress"->"getPrivateIpAddress[]",
	"PrivateIpAddresses"->
		Missing[
			"getPrivateIpAddresses[]",
			{ ec2PrivateIpAddressesTemplate }
		],
	"SourceDestCheck"->"getSourceDestCheck[]",
	"Status"->"getStatus[]",
	"SubnetId"->"getSubnetId[]",
	"VpcId"->"getVpcId[]"
|>;

(*
"com.amazonaws.services.ec2.model.InstanceNetworkInterfaceSpecification "
*)
ec2InstanceNetworkInterfaceSpecificationTemplate = 
<|
"AssociatePublicIpAddress" -> "getAssociatePublicIpAddress[]",
"DeleteOnTermination" -> "getDeleteOnTermination[]",
"Description" -> "getDescription[]",
"DeviceIndex" -> "getDeviceIndex[]",
"Groups" -> "getGroups[]" -> "@toArray[]",
"Ipv6AddressCount" -> "getIpv6AddressCount[]",
"Ipv6Addresses" ->
	Missing[
		"getIpv6Addresses[]",
		{ ec2InstanceIpv6AddressTemplate }
	],
"NetworkInterfaceId" -> "getNetworkInterfaceId[]",
"PrivateIpAddress" -> "getPrivateIpAddress[]",
"PrivateIpAddresses" ->
	Missing[
		"getPrivateIpAddresses[]",
		{ ec2PrivateIpAddressSpecificationTemplate }
	],
"SecondaryPrivateIpAddressCount" -> "getSecondaryPrivateIpAddressCount[]",
"SubnetId" -> "getSubnetId[]"
|>

ec2NetworkInterfacesTemplate =
<|
	ec2InstanceNetworkInterfacesTemplate,
	"AvailabilityZone"->"getAvailabilityZone[]",
	"InterfaceType"->"getInterfaceType[]",
	"RequesterId"->"getRequesterId[]",
	"RequesterManaged"->"getRequesterManaged[]",
	"Tags"->
		Missing[
			"getTagSet[]",
			{ EC2TagTemplate }
		]
|>;

(*
"com.amazonaws.services.ec2.model.EbsBlockDevice "
*)
ec2EbsBlockDeviceTemplate = 
<|
"DeleteOnTermination" -> "getDeleteOnTermination[]",
"Encrypted" -> "getEncrypted[]",
"Iops" -> "getIops[]",
"SnapshotId" -> "getSnapshotId[]",
"VolumeSize" -> "getVolumeSize[]",
"VolumeType" -> "getVolumeType[]"
|>


(*******************************************************

*******************************************************)
ec2EbsTemplate =
<|
"DeleteOnTermination" -> "getDeleteOnTermination[]",
"VolumeSize" -> "getVolumeSize[]",
"Iops" -> "getIops[]",
"SnapshotId" -> "getSnapshotId[]",
"VolumeType" -> "getVolumeType[]",
"Encrypted" -> "getEncrypted[]"
|>;

(*******************************************************

*******************************************************)
ec2EbsInstanceBlockDeviceTemplate =
<|
"AttachTime" -> "getAttachTime[]",
"DeleteOnTermination" -> "getDeleteOnTermination[]",
"Status" -> "getStatus[]",
"VolumeId" -> "getVolumeId[]"
|>;


(*
"com.amazonaws.services.ec2.model.BlockDeviceMapping "
*)
ec2BlockDeviceMappingTemplate = 
<|
"DeviceName" -> "getDeviceName[]",
"Ebs" -> Missing["getEbs[]", ec2EbsBlockDeviceTemplate],
"NoDevice" -> "getNoDevice[]",
"VirtualName" -> "getVirtualName[]"
|>

(*******************************************************

*******************************************************)
ec2InstanceBlockDeviceMappingTemplate =
<|
"DeviceName" -> "getDeviceName[]",
"Ebs" -> Missing["getEbs[]", ec2EbsInstanceBlockDeviceTemplate]
|>;


(*******************************************************

*******************************************************)
(*
"com.amazonaws.services.ec2.model.LaunchSpecification "
*)

ec2LaunchSpecificationTemplate =
<|
"AddressingType"->"getAddressingType[]",
"AllSecurityGroups"-> 
	Missing[
		"getAllSecurityGroups[]",
		{
			<|
			"GroupName" -> "getGroupName[]",
			"GroupId" -> "getGroupId[]"
			|>
		}
	],
"BlockDeviceMappings"->
	Missing[
		"getBlockDeviceMappings[]"
		,
		{ ec2BlockDeviceMappingTemplate }
	],
"EbsOptimized"->"getEbsOptimized[]",
"IamInstanceProfile"->
	Missing[
		"getIamInstanceProfile[]", 
		ec2IamInstanceProfileSpecificationTemplate
	],
"ImageId"->"getImageId[]",
"InstanceType"->"getInstanceType[]",
"KernelId"->"getKernelId[]",
"KeyName"->"getKeyName[]",
"MonitoringEnabled"->"getMonitoringEnabled[]",
"NetworkInterfaces"->
	Missing[
		"getNetworkInterfaces[]",
		{ ec2InstanceNetworkInterfacesTemplate }
	],
"Placement"->
	Missing[
		"getPlacement[]"
		,
		ec2SpotPlacementTemplate
	],
"RamdiskId"->"getRamdiskId[]",
"SecurityGroups" -> "getSecurityGroups[]@toArray[]";
"SubnetId"->"getSubnetId[]",
"UserData"->"getUserData[]"
|>;

(*******************************************************

*******************************************************)
(*
"com.amazonaws.services.ec2.model.StateReason "
*)
ec2StateReasonTemplate = 
<|
"Code" -> "getCode[]",
"Message" -> "getMessage[]"
|>

(*
"com.amazonaws.services.ec2.model.ProductCode "
*)
ec2ProductCodeTemplate = 
<|
"ProductCodeId" -> "getProductCodeId[]",
"ProductCodeType" -> "getProductCodeType[]"
|>

(*******************************************************

*******************************************************)
(*
"com.amazonaws.services.ec2.model.Image "
*)
PackageScope["EC2ImageTemplate"]
EC2ImageTemplate = 
<|
"Architecture" -> "getArchitecture[]",
"BlockDeviceMappings" -> Missing[
	"getBlockDeviceMappings[]"
	,
	{
		ec2BlockDeviceMappingTemplate
	}
	],
"CreationDate" -> "getCreationDate[]",
"Description" -> "getDescription[]",
"EnaSupport" -> "getEnaSupport[]",
"Hypervisor" -> "getHypervisor[]",
"ImageId" -> "getImageId[]",
"ImageLocation" -> "getImageLocation[]",
"ImageOwnerAlias" -> "getImageOwnerAlias[]",
"ImageType" -> "getImageType[]",
"KernelId" -> "getKernelId[]",
"Name" -> "getName[]",
"OwnerId" -> "getOwnerId[]",
"Platform" -> "getPlatform[]",
"ProductCodes" -> Missing[
	"getProductCodes[]"
	,
	{
		ec2ProductCodeTemplate
	}
	],
"Public" -> "getPublic[]",
"RamdiskId" -> "getRamdiskId[]",
"RootDeviceName" -> "getRootDeviceName[]",
"RootDeviceType" -> "getRootDeviceType[]",
"SriovNetSupport" -> "getSriovNetSupport[]",
"State" -> "getState[]",
"StateReason" -> Missing[
	"getStateReason[]"
	,
	ec2StateReasonTemplate
	],
"Tags"->
	Missing[
	"getTags[]"
	,
	{
		EC2TagTemplate
	}
	],
"VirtualizationType" -> "getVirtualizationType[]"
|>

(*******************************************************

*******************************************************)
ec2VolumeAttachmentTemplate =
<|
"DeleteOnTermination" -> "getDeleteOnTermination[]",
"Device" -> "getDevice[]",
"InstanceId" -> "getInstanceId[]",
"State" -> "getState[]",
"VolumeId" -> "getVolumeId[]"
|>;


(*******************************************************

*******************************************************)
PackageScope["EC2VolumeTemplate"]
EC2VolumeTemplate =
<|
"SnapshotId" -> "getSnapshotId[]",
"AvailabilityZone" -> "getAvailabilityZone[]",
"State" -> "getState[]",
"CreateTime" -> "getCreateTime[]",
"Attachments" ->
	Missing[
		"getAttachments[]"
		,
		{
			ec2VolumeAttachmentTemplate
		}
		],
"VolumeId" -> "getVolumeId[]",
"VolumeType" -> "getVolumeType[]",
"Size" -> "getSize[]",
"Iops" -> "getIops[]",
"Encrypted" -> "getEncrypted[]",
"KmsKeyId" -> "getKmsKeyId[]",
"Tags"->
	Missing[
		"getTags[]"
		,
		{
		EC2TagTemplate
		}
	]
|>;


(*******************************************************

*******************************************************)
PackageScope["EC2InstanceTemplate"]
EC2InstanceTemplate =
<|
"InstanceId" -> "getInstanceId[]",
"ImageId" -> "getImageId[]",
"State" -> Missing["getState[]", ec2StateTemplate],
"PrivateDnsName" -> "getPrivateDnsName[]",
"PublicDnsName" -> "getPublicDnsName[]",
"StateTransitionReason" -> "getStateTransitionReason[]",
"KeyName" -> "getKeyName[]",
"AmiLaunchIndex" -> "getAmiLaunchIndex[]",
"ProductCodes" -> "getProductCodes[]@toArray[]",
"InstanceType" -> "getInstanceType[]",
"LaunchTime" -> "getLaunchTime[]",
"Placement" -> Missing["getPlacement[]", ec2SpotPlacementTemplate],
"Monitoring" -> Missing["getMonitoring[]", ec2MonitoringTemplate],
"SubnetId" -> "getSubnetId[]",
"VpcId" -> "getVpcId[]",
"PrivateIpAddress" -> "getPrivateIpAddress[]",
"PublicIpAddress" -> "getPublicIpAddress[]",
"Architecture" -> "getArchitecture[]",
"RootDeviceType" -> "getRootDeviceType[]",
"RootDeviceName" -> "getRootDeviceName[]",
"BlockDeviceMappings" -> 
	Missing[
		"getBlockDeviceMappings[]"
		,
		{
			ec2InstanceBlockDeviceMappingTemplate
		}
	],
"VirtualizationType" -> "getVirtualizationType[]",
"InstanceLifecycle" -> "getInstanceLifecycle[]",
"SpotInstanceRequestId" -> "getSpotInstanceRequestId[]",
"ClientToken" -> "getClientToken[]",
"Tags" ->
	Missing[
		"getTags[]"
		,
	{
		EC2TagTemplate
	}],
"SecurityGroups" ->
	Missing[
		"getSecurityGroups[]",
		{
			ec2GroupIdentifierTemplate
		}
	],
"SourceDestCheck" -> "getSourceDestCheck[]",
"Hypervisor" -> "getHypervisor[]",
"NetworkInterfaces" ->
	Missing[
		"getNetworkInterfaces[]"
		,
		{
			ec2InstanceNetworkInterfacesTemplate
		}
	],
"EbsOptimized" -> "getEbsOptimized[]",
"EnaSupport" -> "getEnaSupport[]"
|>;

(*******************************************************

*******************************************************)
PackageScope["EC2ReservationTemplate"]
EC2ReservationTemplate =
<|
"ReservationId" -> "getReservationId[]",
"OwnerId" -> "getOwnerId[]",
"RequesterId" -> "getRequesterId[]",
"Groups" -> 
	Missing[
		"getGroups[]"
		,
		{ec2GroupIdentifierTemplate}
	],
"Instances" ->
	Missing[
		"getInstances[]"
		,
		{
			EC2InstanceTemplate
		}
	],
"GroupNames" -> "getGroupNames[]@toArray[]"
|>;


(*******************************************************

*******************************************************)
PackageScope["EC2DescribeInstanceResultTemplate"]
EC2DescribeInstanceResultTemplate =
<|
"Reservations" ->
	Missing[
		"getReservations[]"
		,
		{
			EC2ReservationTemplate
		}
	]
|>;
(*******************************************************

*******************************************************)
ec2SpotInstanceStateFaultTemplates =
<|
"Code" -> "getCode[]",
"Message" -> "getMessage[]"
|>;

(*******************************************************
*******************************************************)
dollarPerHourFun =
Function[
		{price}, 
		If[
			MatchQ[price, _String],
			Quantity[ToExpression[price], Times[Power["Hours", -1], "USDollars"]],
			price
		]
	];
(*******************************************************

*******************************************************)
PackageScope["EC2SpotInstanceRequestTemplate"]
EC2SpotInstanceRequestTemplate =
<|
"ActualBlockHourlyPrice" -> "getActualBlockHourlyPrice[]" -> dollarPerHourFun,
"AvailabilityZoneGroup" -> "getAvailabilityZoneGroup[]",
"BlockDurationMinutes" -> "getBlockDurationMinutes[]",
"CreateTime" -> "getCreateTime[]",
"Fault" ->
	Missing[
		"getFault[]",
		ec2SpotInstanceStateFaultTemplates
	],
"InstanceId"->"getInstanceId[]",
"LaunchedAvailabilityZone"->"getLaunchedAvailabilityZone[]",
"LaunchGroup"->"getLaunchGroup[]",
"LaunchSpecification"->
	Missing[
		"getLaunchSpecification[]",
		ec2LaunchSpecificationTemplate
	],
"ProductDescription"->"getProductDescription[]",
"SpotInstanceRequestId"->"getSpotInstanceRequestId[]",
"SpotPrice"-> "getSpotPrice[]" -> dollarPerHourFun,
"State"->"getState[]",
"Status" -> 
	Missing[
		"getStatus[]"
		,
		ec2StatusTemplate
	],
"Tags"->
	Missing[
		"getTags[]",
		{
			EC2TagTemplate
		}
	],
"Type"->"getType[]",
"ValidFrom"->"getValidFrom[]",
"ValidUntil"->"getValidUntil[]"
|>;


(*
"com.amazonaws.services.ec2.model.RunInstancesRequest "
*)
ec2RunInstancesRequestTemplate = 
<|
"AdditionalInfo" -> "getAdditionalInfo[]",
"BlockDeviceMappings" -> 
	Missing[
		"getBlockDeviceMappings[]",
		{ec2BlockDeviceMappingTemplate}
	],
"ClientToken" -> "getClientToken[]",
"CustomQueryParameters" -> 
	Missing[
		"getCustomQueryParameters[]",
		"keySet[] @ toArray[]" ->
			Missing[
				"values[]", 
				{ "toArray[]" }
			]
	],
"CustomRequestHeaders" ->
	Missing[
		"getCustomRequestHeaders[]",
		"keySet[] @ toArray[]" -> "values[] @ toArray[]"
	],
"DisableApiTermination" -> "getDisableApiTermination[]",
"EbsOptimized" -> "getEbsOptimized[]",
"ElasticGpuSpecification" ->
	Missing[
		"getElasticGpuSpecification[]",
		{ ec2ElasticGpuSpecificationTemplate }
	],
"IamInstanceProfile" -> Missing["getIamInstanceProfile[]", ec2IamInstanceProfileSpecificationTemplate],
"ImageId" -> "getImageId[]",
"InstanceInitiatedShutdownBehavior" -> "getInstanceInitiatedShutdownBehavior[]",
"InstanceType" -> "getInstanceType[]",
"Ipv6AddressCount" -> "getIpv6AddressCount[]",
"Ipv6Addresses" -> 
	Missing[
		"getIpv6Addresses[]",
		{ec2InstanceIpv6AddressTemplate}
	],
"KernelId" -> "getKernelId[]",
"KeyName" -> "getKeyName[]",
"MaxCount" -> "getMaxCount[]",
"MinCount" -> "getMinCount[]",
"Monitoring" -> "getMonitoring[]",
"NetworkInterfaces" ->
	Missing[
		"getNetworkInterfaces[]",
		{ ec2InstanceNetworkInterfaceSpecificationTemplate }
	],
"Placement" ->
	Missing[
		"getPlacement[]",
		ec2PlacementTemplate
	],
"PrivateIpAddress" -> "getPrivateIpAddress[]",
"RamdiskId" -> "getRamdiskId[]",
"ReadLimit" -> "getReadLimit[]",
"SdkClientExecutionTimeout" -> "getSdkClientExecutionTimeout[]",
"SdkRequestTimeout" -> "getSdkRequestTimeout[]",
"SecurityGroupIds" -> Missing["getSecurityGroupIds[]", "@toArray[]"],
"SecurityGroups" -> Missing["getSecurityGroups[]", "@toArray[]"],
"SubnetId" -> "getSubnetId[]",
"TagSpecifications" -> Missing["getTagSpecifications[]", {""}],
"UserData" -> "getUserData[]"
|>

(*
"com.amazonaws.services.ec2.model.RequestSpotInstancesRequest "
*)
ec2RequestSpotInstancesRequestTemplate = 
<|
"AvailabilityZoneGroup" -> "getAvailabilityZoneGroup[]",
"BlockDurationMinutes" -> "getBlockDurationMinutes[]",
"ClientToken" -> "getClientToken[]",
"CustomQueryParameters" -> 
	Missing[
		"getCustomQueryParameters[]",
		"keySet[] @ toArray[]" ->
			Missing[
				"values[]", 
				{ "toArray[]" }
			]
	],
"CustomRequestHeaders" ->
	Missing[
		"getCustomRequestHeaders[]",
		"keySet[] @ toArray[]" -> "values[] @ toArray[]"
	],
"InstanceCount" -> "getInstanceCount[]",
"InstanceInterruptionBehavior" -> "getInstanceInterruptionBehavior[]",
"LaunchGroup" -> "getLaunchGroup[]",
"LaunchSpecification" -> Missing["getLaunchSpecification[]", ec2LaunchSpecificationTemplate],
"ReadLimit" -> "getReadLimit[]",
"SdkClientExecutionTimeout" -> "getSdkClientExecutionTimeout[]",
"SdkRequestTimeout" -> "getSdkRequestTimeout[]",
"SpotPrice" -> "getSpotPrice[]",
"Type" -> "getType[]",
"ValidFrom" -> "getValidFrom[]",
"ValidUntil" -> "getValidUntil[]"
|>


(*******************************************************
 *  
 * ERRORS:
 *  
 ******************************************************)

(*******************************************************
"com.amazonaws.services.ec2.model.AmazonEC2Exception"
*******************************************************)
PackageScope["EC2ExceptionTemplate"]
EC2ExceptionTemplate =
<|
"Name"-> "getClass[]@getCanonicalName[]",
"ErrorCode" -> "getErrorCode[]", 
"ErrorMessage" -> "getErrorMessage[]",
"HttpHeaders" ->
	Missing[
		"getHttpHeaders[]"
		,
		"keySet[] @ toArray[]" -> "values[] @ toArray[]"
	], 
"LocalizedMessage" -> "getLocalizedMessage[]", 
"Message" -> "getMessage[]", 
"RawResponseContent" -> "getRawResponseContent[]", 
"RequestId" -> "getRequestId[]",
"ServiceName" -> "getServiceName[]",
"StatusCode" -> "getStatusCode[]"
|>;


(*******************************************************
    com.amazonaws.SdkClientException 
*******************************************************)
PackageScope["SdkClientExceptionTemplate"]
SdkClientExceptionTemplate =
<|
"Name"-> "getClass[]@getCanonicalName[]",
"LocalizedMessage" -> "getLocalizedMessage[]", 
"Message" -> "getMessage[]"
 |>

 
(*******************************************************
    java exception 
*******************************************************)
PackageScope["JavaExceptionTemplate"]
JavaExceptionTemplate =
 <|
"Name"-> "getClass[]@getCanonicalName[]",
"ErrorCode" -> "getErrorCode[]",
"Message" -> "getErrorMessage[]"
|>
(*******************************************************************************

JSON Tools

*******************************************************************************)

Package["ProtobufLink`"]

PackageImport["GeneralUtilities`"]


(*----------------------------------------------------------------------------*)
PackageExport["ProtobufMessage"]
PackageScope["messageMLE"]

(*----------------------------------------------------------------------------*)

messageReadJSON = LibraryFunctionLoad[$ProtoLinkLib, "WL_MessageReadJSON", 
	{
		Integer,
		"UTF8String"
	},
	"Void"		
]

messageReadBinary = LibraryFunctionLoad[$ProtoLinkLib, "WL_MessageReadBinary", 
	{
		Integer,
		"UTF8String"
	},
	"Void"		
]

dscpoolToMutableMessage = LibraryFunctionLoad[$ProtoLinkLib, "WL_DescriptorPoolToMutableMessage", 
	{
		Integer,
		Integer,
		"UTF8String"
	},
	"Void"		
]


messageGetFields = LibraryFunctionLoad[$ProtoLinkLib, "WL_MessageFieldNames", 
		LinkObject,
		LinkObject		
]

messageGetFieldType = LibraryFunctionLoad[$ProtoLinkLib, "WL_messageGetFieldType", 
		{Integer, Integer},
		"UTF8String"	
]

messageName = LibraryFunctionLoad[$ProtoLinkLib, "WL_messageName", 
		{Integer},
		"UTF8String"	
]

messageGetFieldCount = LibraryFunctionLoad[$ProtoLinkLib, "WL_messageGetFieldCount", 
		{Integer},
		Integer	
]

messageInformation = LibraryFunctionLoad[$ProtoLinkLib, "WL_MessageDescriptor", 
		LinkObject,
		LinkObject		
]

(*----------------------------------------------------------------------------*)
(* This is a utility function defined in GeneralUtilities, which makes a nicely
formatted display box *)
DefineCustomBoxes[ProtobufMessage, 
	e:ProtobufMessage[mle_, desc_, name_] :> Block[{},
	BoxForm`ArrangeSummaryBox[
		ProtobufMessage, e, None, 
		{
			BoxForm`SummaryItem[{"Name: ", name}],
			BoxForm`SummaryItem[{"ID: ", getMLEID[mle]}]
		},
		{},
		StandardForm
	]
]];

(* Upvalues *)
ProtobufMessage /: Keys[x_ProtobufMessage] := 
	ProtobufMessageFields[x]

ProtobufMessage /: Length[x_ProtobufMessage] := 
	ProtobufMessageFieldCount[x]

getMLE[ProtobufMessage[mle_, ___]] := mle;
getMLEID[ProtobufMessage[mle_, ___]] := ManagedLibraryExpressionID[mle];


(*----------------------------------------------------------------------------*)
PackageExport["ProtobufDescriptorToMessage"]

ProtobufDescriptorToMessage[dsc_ProtobufDescriptor, message_String] := CatchFailure @ Module[
	{mle}
	,
	mle = CreateManagedLibraryExpression["ProtoMessage", messageMLE];
	safeLibraryInvoke[dscpoolToMutableMessage,
		getMLEID[dsc],
		getMLEID[mle],
		message
	];
	System`Private`SetNoEntry @ ProtobufMessage[mle, dsc, message]
]

(*----------------------------------------------------------------------------*)
PackageExport["ProtobufMessageSet"]

SetUsage[
"ProtobufMessageSet[ProtobufMessage[$$], expr$] sets the ProtobufMessage[$$] message \
fields with values given in the Association expr$.
ProtobufMessageSet[ProtobufMessage[$$], File[f$]] sets the ProtobufMessage[$$] message \
fields with values given in the binary file File[f$].
"
]

(* This should be implemented more efficiently later *)
ProtobufMessageSet[msg_ProtobufMessage, expr_Association] := CatchFailure @ Module[
	{json = Developer`WriteRawJSONString[expr]},
	safeLibraryInvoke[messageReadJSON, getMLEID[msg], json];
]

ProtobufMessageSet[msg_ProtobufMessage, File[bin_]] := CatchFailure @ Module[
	{mle}
	,
	safeLibraryInvoke[messageReadBinary,
		getMLEID[msg],
		fileConform[bin]
	];
]

(*----------------------------------------------------------------------------*)
(* Message Information Functions *)
(*----------------------------------------------------------------------------*)

PackageExport["ProtobufMessageName"]

ProtobufMessageName[m_ProtobufMessage] := CatchFailure @ 
	safeLibraryInvoke[messageName, getMLEID[m]]

(*----------------------------------------------------------------------------*)
PackageExport["ProtobufMessageFieldType"]

ProtobufMessageFieldType[m_ProtobufMessage, field_Integer] := CatchFailure @ 
	safeLibraryInvoke[messageGetFieldType, getMLEID[m], field]

(*----------------------------------------------------------------------------*)
PackageExport["ProtobufMessageFieldCount"]

ProtobufMessageFieldCount[m_ProtobufMessage] := CatchFailure @ 
	safeLibraryInvoke[messageGetFieldCount, getMLEID[m]]

(*----------------------------------------------------------------------------*)
PackageExport["ProtobufMessageFieldTypes"]

ProtobufMessageFieldTypes[m_ProtobufMessage] := CatchFailure @ Module[
	{len = ProtobufMessageFieldCount[m]}
	,
	Table[
		ProtobufMessageFieldType[m, i],
		{i, len}
	]
]

(*----------------------------------------------------------------------------*)
PackageExport["ProtobufMessageFields"]

ProtobufMessageFields[m_ProtobufMessage] := CatchFailure @ 
	safeLibraryInvoke[messageGetFields, getMLEID[m]]

(*----------------------------------------------------------------------------*)
PackageExport["ProtobufMessageDescription"]

ProtobufMessageDescription[m_ProtobufMessage] := CatchFailure @ Dataset @
	safeLibraryInvoke[messageInformation, getMLEID[m]]


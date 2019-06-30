(* ::Package:: *)

Begin["System`Convert`MBoxDump`"];

(* elements1x : extracted with MBoxImport[_, element] *)
(* elements11 : compatible with a specific message (number or id),
have a custom processMessage definition *)
(* Implemented by processMessage *)
elements11 = {
	(* "RawHeaders", "RawData", *) "Plaintext"
};
(* elements12 : incompatible with a specific message (number or id), should
always be available for any MBox *)
(* Implemented by MBoxImport *)
elements12 = {
	"MessageCount"
};

(* elements2 : extracted with MIMEMessageGetElement[msg, element] *)
elements2 = {
	"From", "FromAddress", "FromName",
	"Subject", "MIMEVersion", "MessageID",
	"ToList", "ToAddressList", "ToNameList",
	"CcList", "CcAddressList", "CcNameList",
	"BccList", "BccAddressList", "BccNameList",
	"ReplyToList", "ReplyToAddressList", "ReplyToNameList",

	"HeaderRules", "HeaderString",
	"ReturnReceiptRequested", "DeliveryChainHostnames",
	"DeliveryChainRecords", "ContentType", "OriginatingIPAddress",
	"OriginatingHostname", "OriginatingCountry",
	"OriginatingDate", "OriginatingTimezone",
	"ServerOriginatingDate", "ServerOriginatingTimezone",
	"ReplyToMessageID", "ReturnPath", "Precedence",
	"CharacterEncoding", "OriginatingMailClient",

	"Body", "BodyPreview",
	"Attachments", "AttachmentData", "AttachmentSummaries",
	"AttachmentAssociations", "AttachmentNames", "HasAttachments",

	"NewBodyContent", "QuotedContent", "ReferenceMessageIDList",
	(*"ContentAssociationList", "ContentList", *)

	"MessageSummaries", "MessageElements"
};

(* elements3 : an association containing all elements from elements11 and elements 2 *)
elements3 = {
	"FullMessageElements"
};

deprecatedElements1 = Association[
	"RawData" -> "Plaintext"
]
deprecatedElements2 = Association[
	"ReplyTo" -> "ReplyToList",
	"To" -> "ToList",
	"cc" -> "CcList",
	"Cc" -> "CcList",
	"CC" -> "CcList",
	"ccList" -> "CcList",
	"CCList" -> "CcList",
	"bcc" -> "BccList",
	"Bcc" -> "BccList",
	"BCc" -> "BccList",
	"BCC" -> "BccList",
	"bccList" -> "BccList",
	"BCcList" -> "BccList",
	"BCCList" -> "BccList",
	"Date" -> "OriginatingDate",
	"Data" -> "Body",
	"RawAttachments" -> "AttachmentData",
	"EmailClient" -> "OriginatingMailClient",
	"MessageSummary" -> "MessageSummaries",
	"AttachmentSummary" -> "AttachmentSummaries"
]



ImportExport`RegisterImport[
	"MBOX",
	{
		"Elements" -> (
			(# -> {} &) /@
			Sort[{
				Sequence @@ elements11,
				Sequence @@ elements12,
				Sequence @@ elements2,
				Sequence @@ elements3
			}]
		&),
		
		messageNumber_Integer :> (MBoxImport[#1, messageNumber, ##2] &),

		(* elements1 *)

		element_String /; MemberQ[Join[elements11, elements12, elements3], element]
		:> (MBoxImport[#1, element, element, ##2] &),

		element_String /; MemberQ[Keys[deprecatedElements1], element]
		:> (MBoxImport[#1, element, Replace[element, deprecatedElements1], ##2] &),

		{element_String, messageId:({__Integer} | _String | _Integer)}
		/; MemberQ[Join[elements11, elements3], element]
		:> (MBoxImport[#1, element, element, messageId, ##2] &),

		{element_String, messageId:({__Integer} | _String | _Integer)}
		/; MemberQ[Keys[deprecatedElements1], element]
		:> (MBoxImport[#1, element, Replace[element, deprecatedElements1], messageId, ##2] &),

		(* elements2 *)

		element_String /; MemberQ[elements2, element]
		:> (MBoxImport[#1, element, {"Element", element}, ##2] &),

		element_String /; MemberQ[Keys[deprecatedElements2], element]
		:> (MBoxImport[#1, element, {"Element", Replace[element, deprecatedElements2]}, ##2] &),

		{element_String, messageId:({__Integer} | _String | _Integer)}
		/; MemberQ[elements2, element]
		:> (MBoxImport[#1, element, {"Element", element}, messageId, ##2] &),

		{element_String, messageId:({__Integer} | _String | _Integer)}
		/; MemberQ[Keys[deprecatedElements2], element]
		:> (MBoxImport[#1, element, {"Element", Replace[element, deprecatedElements2]}, messageId, ##2] &),

		MBoxImport
	},
	"AvailableElements" -> {
		Sequence @@ elements11,
		Sequence @@ elements12,
		Sequence @@ elements2,
		Sequence @@ elements3,
		Sequence @@ Keys[deprecatedElements1],
		Sequence @@ Keys[deprecatedElements2]
	},
	"Sources" -> ImportExport`DefaultSources["MBox"]
];

End[];

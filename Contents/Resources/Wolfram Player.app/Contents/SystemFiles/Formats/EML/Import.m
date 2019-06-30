(* ::Package:: *)

Begin["System`Convert`EMLDump`"];

(* elements1x : extracted with EMLImport[_, element] *)
(* Implemented by processMessage *)
elements1 = {
	(* "RawHeaders", "RawData", *) "Plaintext"
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

	"MessageSummary", "MessageElements"
};

(* elements3 : an association containing all elements from elements1 and elements 2 *)
elements3 = {
	"FullMessageElements"
};

deprecatedElements1 = Association[
	"RawData" -> "Plaintext"
]
deprecatedElements2 = Association[
	"ReplyTo" -> "ReplyToList",
	"To" -> "ToList",
	"Cc" -> "CcList",
	"CC" -> "CcList",
	"Bcc" -> "BccList",
	"BCC" -> "BccList",
	"Date" -> "OriginatingDate",
	"Data" -> "Body",
	"RawAttachments" -> "AttachmentData",
	"EmailClient" -> "OriginatingMailClient",
	"MessageSummaries" -> "MessageSummary",
	"AttachmentSummaries" -> "AttachmentSummary"
]


ImportExport`RegisterImport[
	"EML",
	{
		"Elements" -> (
			(# -> {} &) /@
			Sort[{
				Sequence @@ elements1,
				Sequence @@ elements2,
				Sequence @@ elements3
			}]
		&),
		
		messageNumber_Integer :> (EMLImport[#1, messageNumber, ##2] &),

		(* elements1 *)

		element_String /; MemberQ[Join[elements1, elements3], element]
		:> (element -> EMLImport[#1, element, element, ##2] &),

		element_String /; MemberQ[Keys[deprecatedElements1], element]
		:> (element -> EMLImport[#1, element, Replace[element, deprecatedElements1], ##2] &),

		(* elements2 *)

		element_String /; MemberQ[elements2, element]
		:> (element -> EMLImport[#1, element, {"Element", element}, ##2] &),

		element_String /; MemberQ[Keys[deprecatedElements2], element]
		:> (element -> EMLImport[#1, element, {"Element", Replace[element, deprecatedElements2]}, ##2] &),

		EMLImport
	},
	"AvailableElements" -> {
		Sequence @@ elements1,
		Sequence @@ elements2,
		Sequence @@ elements3,
		Sequence @@ Keys[deprecatedElements1],
		Sequence @@ Keys[deprecatedElements2]
	},
	"Sources" -> ImportExport`DefaultSources["EML"]
];

End[];

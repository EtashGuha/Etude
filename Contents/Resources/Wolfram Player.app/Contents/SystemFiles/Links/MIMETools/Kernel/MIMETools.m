(* ::Package:: *)

(* Mathematica Package *)

(* Created by the Wolfram Workbench Jan 30, 2012 *)

BeginPackage["MIMETools`"]
(* Exported symbols added here with SymbolName::usage *)

CreateMIMEToolsException::usage = "MIMEToolsException[type_String, errorCode_Integer, message_String, param_List, cause_] creates a MIMEToolsException."
MIMEToolsException::usage = "MIMEToolsException is an exception object returned by CreateMIMEToolsException. MIMEToolsException[field] queries for the following properties: \"Type\", \"ErrorCode\", \"Message\", \"Parameters\", \"Cause\", \"ErrorID\""
MIMEMessageCreate::usage = "MIMEMessageCreate[rules] creates and returns a properly formatted MIME message as a string from the fields specified in rules.";
MIMEMessageOpen::usage = "MIMEMessageOpen[message] returns a MIMEMessage object.  \"message\" can be a String, and it also can be a File.";
MIMEMessageRead::usage = "MIMEMessageRead[obj, fields] returns data from the fields queried.";
MIMEMessageClose::usage = "MIMEMessageClose[obj] closes the MIMEMessage specified.";
MIMEMessageParse::usage = "MIMEMessageParse[message] parses the message and returns a list of Associations containing all the parts found in the message. This will return the content-type and disposition parameters in a flat list.";
MIMEMessageRawParse::usage = "MIMEMessageParse[message]parses the message and returns a list of Associations containing all the parts found in the message. This will return the raw data of the contents with content-type and disposition associations.";
MIMEMessageParse::usage = "MIMEMessageParse[message, field] parses the message or message File and returns a list of Associations containing the given field.";
MIMEAttachmentSave::usage = "MIMEAttachmentSave[obj, attachmentDirectory] saves the attachments in obj to attachmentDirectory.";
(*MIMEMessageUnquotedBody::usage = "MIMEMessageUnquotedBody[obj,Options] parses the unquoted part Body using OptionValue[Bound] as the boundary.";*)
getOpenMIMEMessage::usage = "getMIMEMessageOpen returns open MIMEMessage.";
MIMEMessageGetElement::usage = "MIMEMessageGetElement[ obj, elem] gets the value for a mail element.  MIMEMessageGetElement[] returns a list of possible elements."

MIMEMessage::usage = "MIMEMessage[id] is an object representing an open MIMEMessage that can be queried for its parts.";
MIMEBodyParse::usage = "MIMEBodyParse[body_string, \"NewBodyContent\"|\"QuotedContent\"|\"AllQuotedContent\"|\"Attribution\"|All]"
MailElementLookup::usage = "MailElementLookup[elem_MailElement, propKey_String]"
(*MIMEBodyParseThread::usage = "MIMEBodyParse[oldbody_String, newbody_String]"*)
MailElement::usage = "MailElement[content, attribute_association] represents a mail element object."

Begin["`Private`"]
(* Implementation of the package *)

$packageFile = $InputFileName;
$libraryFileName = Switch[$OperatingSystem, "Windows", "MIMETools.dll", "MacOSX", "MIMETools.dylib", "Unix", "MIMETools.so", _, $Failed];

$library = FileNameJoin[{FileNameTake[$packageFile, {1,-3}], "LibraryResources", $SystemID, $libraryFileName}];
$libraryGMime = FileNameJoin[{FileNameTake[$packageFile, {1,-3}], "LibraryResources", $SystemID, Switch[$OperatingSystem, "Windows", "gmime.dll", "MacOSX", "TODO", "Unix", "libgmime-2.6.so.0"]}];

$asyncObj = Null;
$adapterLoaded = False;

loadAdapter[] :=
If[$libraryFileName === $Failed,
	Return@CreateMIMEToolsException[47];
	,
	Module[{},
		If[!$adapterLoaded,
			If[$SystemID =!= "Linux-ARM" && ($OperatingSystem === "Windows" || $OperatingSystem === "Unix"),
				LibraryLoad[$libraryGMime];
			];
			If[LibraryLoad[$library] =!= $Failed,
				lfMIMEFileOpen = LibraryFunctionLoad[MIMETools`Private`$library, "MIMEFileOpen", LinkObject, LinkObject];
				lfMIMEStringOpen = LibraryFunctionLoad[MIMETools`Private`$library, "MIMEStringOpen", LinkObject, LinkObject];
				lfMIMEStringCreate = LibraryFunctionLoad[MIMETools`Private`$library, "MIMEStringCreate", LinkObject,  LinkObject];
				lfMIMEMessageClose = LibraryFunctionLoad[MIMETools`Private`$library, "MIMEMessageClose", LinkObject, LinkObject];
				lfGetOpenMIMEStream = LibraryFunctionLoad[MIMETools`Private`$library, "GetOpenMIMEStream", LinkObject,  LinkObject];
				lfMIMEMessageLookup = LibraryFunctionLoad[MIMETools`Private`$library, "MIMEMessageLookup", LinkObject, LinkObject];
				(*lfMIMEFileLookup = LibraryFunctionLoad[MIMETools`Private`$library, "MIMEFileLookup", LinkObject, LinkObject];*)
				$adapterLoaded = True;
				,
				Return@CreateMIMEToolsException[47];
			]
			,
			$adapterLoaded = False;
			Return@CreateMIMEToolsException[47];
		]
	]
]

(* ::Section:: *)
(* Exceptions *)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)

$errID = 0;

$exceptionCoreLUT = Association[
	32 -> <|"Type" -> "MalformedMessageError", "ErrorCode" -> 32, "Message" -> "Malformed message passed to MIMEMessageOpen."|>,
	33 -> <|"Type" -> "ReadUserInputError", "ErrorCode" -> 33, "Message" -> "Unknown field passed to MIMEMessageRead. Please see the cause for the unknown field."|>,
	34 -> <|"Type" -> "MessageParseError", "ErrorCode" -> 34, "Message" -> "A MIMETools object was passed to MIMEMessageParse. Please use MIMEMessageRead, or pass in a String/File."|>,
	35 -> <|"Type" -> "ReadInputTypeError", "ErrorCode" -> 35, "Message" -> "An invalid object passed to MIMEMessageRead. Please provide an open MIMEMessage."|>,
	36 -> <|"Type" -> "OpenInputTypeError", "ErrorCode" -> 36, "Message" -> "An incompatible data format was passed to MIMEMessageOpen. Please provide a File or a String."|>,
	37 -> <|"Type" -> "EmailInterpretFailure", "ErrorCode" -> 37, "Message" -> "An attempt to interpret the email address has failed. See cause for more details."|>,
	38 -> <|"Type" -> "IPInterpretFailure", "ErrorCode" -> 38, "Message" -> "An attempt to interpret the IPAddress has failed. See cause for more details."|>,
	39 -> <|"Type" -> "ImportFailure", "ErrorCode" -> 39, "Message" -> "An attempt to import the data has failed. Please see the exception parameters for the raw bytestring."|>,
	40 -> <|"Type" -> "ElementInputTypeError", "ErrorCode" -> 40, "Message" -> "An incompatible data format was passed to MIMEMessageGetElement. Please provide a MIMEMessage object."|>,
	41 -> <|"Type" -> "ElementUserInputError", "ErrorCode" -> 41, "Message" -> "Unknown field passed to GetElement. Please see the cause for the unknown field."|>,
	42 -> <|"Type" -> "GeneralReadError", "ErrorCode" -> 42, "Message" -> "An error was encountered reading the field specified." |>,
	43 -> <|"Type" -> "UnsupportedFormat", "ErrorCode" -> 43, "Message" -> "A MIMEType with an unsupported format has been passed to MIMEMessageRead. Please see the exception parameters for more details."|>,
	44 -> <|"Type" -> "CloseInputError", "ErrorCode" -> 44, "Message" -> "An invalid object was passed to MIMEMessageClose."|>,
	45 -> <|"Type" -> "MIMEBodyParseInputTypeError", "ErrorCode" -> 45, "Message" -> "An incompatible data format was passed to MIMEBodyParse. Please provide a String, or open MIMEMessage object."|>,
	46 -> <|"Type" -> "GetOpenStreamError", "ErrorCode" -> 46, "Message" -> "An error occured opening the stream."|>,
	47 -> <|"Type" -> "LoadAdapterFailure", "ErrorCode" -> 47, "Message" -> "An error occured loading the adapter."|>,
	48 -> <|"Type" -> "MIMEMessageCreateError", "ErrorCode" -> 48, "Message" -> "An error occured creating the message."|>,
	49 -> <|"Type" -> "MIMEMessageOpenError", "ErrorCode" -> 49, "Message" -> "An error occured opening the message."|>,
	50 -> <|"Type" -> "MIMEMessageCloseError", "ErrorCode" -> 50, "Message" -> "An error occured closing the message."|>,
	51 -> <|"Type" -> "OSNotSupported", "ErrorCode" -> 51, "Message" -> "MIMETools is not supported on this operating system."|>,
	52 -> <|"Type" -> "FileNotFound", "ErrorCode" -> 52, "Message" -> "File not found."|>
];

(* MIMEToolsException[
	type_String,
	errorCode_Integer,
	message_String,
	param_List,
	cause_(*MIMEToolsException*)] :=
Association[
	"Type" -> type,
	"ErrorCode" -> errorCode,
	"Message" -> message,
	"Parameters" -> param,
	"Cause" -> cause]; *)

MIMEToolsException /:
	MakeBoxes[exc_MIMEToolsException, form : StandardForm | TraditionalForm] :=
		RowBox[{"MIMEToolsException", "[<", exc["ErrorID"], ">, ", exc["Type"], "]"}]

MIMEToolsException[assoc_Association][key_] := assoc[key]

CreateMIMEToolsException[errorCode_, cause_:None, params_:{}] :=
Module[{res},
	res = Lookup[$exceptionCoreLUT, errorCode, Association["Type" -> "UnknownErrorType", "ErrorCode" -> errorCode, "Message" -> "Unknown error."]];
	AssociateTo[res, {"Parameters" -> params, "Cause" -> cause, "ErrorID" -> ++$errID}];
	Return[MIMEToolsException[res]]
]

(* ::Section:: *)
(* MIMEMessageGetElement *)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)

(* ::Subsection:: *)
(* Elements *)
(* List of available elements, not including header elements *)

MIMEMessageGetElement[] :=
Flatten@Join[
	{
		"Attachments",
		"AttachmentSummary",
		"AttachmentSummaries",
		"AttachmentAssociations",
		"AttachmentData",
		"AttachmentNames",
		"HasAttachments",
		"HeaderRules",
		"HeaderString",
		"Body",
		"BodyPreview",
		"MessageSummary",
		"MessageSummaries",
		"MessageElements",
		"NewBodyContent",
		"QuotedContent"

	}
	,
	MIMEMessageGetElement["Header"]
]

(* List of header elements. THE ORDER THE ELEMENTS WILL BE SORTED IN FOR CALLS TO MIMEMessageRead[.., "Header"] *)
MIMEMessageGetElement["Header"] := {
	"ReturnPath",
	"DeliveryChainRecords",
	"ContentType",
	"Type",
	"Subtype",
	"Boundary",
	"CharacterEncoding",
	"Subject",
	"From",
	"FromAddress",
	"FromName",
	"ToList",
	"ToAddressList",
	"ToNameList",
	"CcList",
	"CcAddressList",
	"CcNameList",
	"BccList",
	"BccAddressList",
	"BccNameList",
	"Date",
	"MessageID",
	"MIMEVersion",
	"ReferenceMessageIDList",
	"ReplyToList",
	"ReplyToAddressList",
	"ReplyToNameList",
	"ReplyToMessageID",
	"ListSubscribe",
	"ListUnsubscribe",
	"Precedence",
	"ReturnReceiptRequested",
	"DeliveryChainHostnames",
	"OriginatingMailClient",
	"OriginatingHostname",
	"OriginatingIPAddress",
	"OriginatingCountry",
	"OriginatingDate",
	"OriginatingTimezone",
	"ServerOriginatingDate",
	"ServerOriginatingTimezone"
}
elementQ[elem_String] := MemberQ[MIMEMessageGetElement[], elem]
headerElementQ[elem_String] := MemberQ[MIMEMessageGetElement["Header"], elem]

(* ::Subsection:: *)
(* General *)

MIMEMessageGetElement[obj_?openMessageQ, "MessageSummary" | "MessageSummaries"] :=
Module[{headers},
	headers = MIMEMessageRead[obj, "Header", {"From", "ToList", "CcList", "BccList", "OriginatingDate", "Subject", "MessageID"}];
	Association[
		"From" -> headers["From"],
		"ToList" -> headers["ToList"],
		"CcList" -> headers["CcList"],
		"BccList" -> headers["BccList"],
		"OriginatingDate" -> headers["OriginatingDate"],
		"Subject" -> headers["Subject"],
		"BodyPreview" -> MIMEMessageGetElement[obj, "BodyPreview"],
		"HasAttachments" -> MIMEMessageGetElement[obj, "HasAttachments"],
		"MessageID" -> headers["MessageID"]
	]
]

MIMEMessageGetElement[obj_?openMessageQ, "MessageElements"] :=
Module[{a},
	a = MIMEMessageRead[obj, "Header", $messagesElements];
	KeyDropFrom[a, "MIMEMessageClass"];
	KeyDropFrom[a, Complement[MIMEMessageGetElement["Header"], $messagesElements]];
	AssociateTo[a, "Body" -> MIMEMessageGetElement[obj, "Body"]];
	AssociateTo[a, "Attachments" -> MIMEMessageGetElement[obj, "Attachments"]];
	KeySortBy[a, sortMessages]
]

$messagesElements = {
	"From",
	"FromAddress",
	"FromName",
	"ToList",
	"ToAddressList",
	"ToNameList",
	"CcList",
	"CcAddressList",
	"CcNameList",
	"BccList",
	"BccAddressList",
	"BccNameList",
	"ReplyToList",
	"ReplyToAddressList",
	"ReplyToNameList",
	"OriginatingDate",
	"Subject",
	"Body",
	"Attachments",
	"MessageID"
}

sortMessages := (Position[$messagesElements, #] /. ({} :> {{Length@$messagesElements + 1}})) &

(* ::Subsection:: *)
(* Header *)

MIMEMessageGetElement[obj_?openMessageQ, elem_?headerElementQ] := MIMEMessageRead[obj, "Header", {elem}][elem]
MIMEMessageGetElement[obj_?openMessageQ, "HeaderRules"] := Normal[MIMEMessageRead[obj, "Header"]]
MIMEMessageGetElement[obj_?openMessageQ, "MessageId"] := MIMEMessageRead[obj, "Header", {"MessageID"}]["MessageID"]
MIMEMessageGetElement[obj_?openMessageQ, "HeaderString"] := StringReplace[MIMEMessageRead[obj, "HeaderString"], "\r\n" -> "\n"]

(* ::Subsection:: *)
(* Body *)

MIMEMessageGetElement[obj_?openMessageQ, "BodyPreview"] :=
Module[{nbc},
	nbc = MIMEMessageGetElement[obj, "NewBodyContent"];
	If[Length[nbc] > 0,
		Short[First[nbc]]
		,
		None
	]
]

MIMEMessageGetElement[obj_?openMessageQ, "Body"] :=
Module[{rawbod, cTypes, pos},
	rawbod = MIMEMessageRead[obj, "DecodedRawBody"];
	cTypes = ToLowerCase[Lookup[#, "ContentType", ""]] & /@ rawbod;
	If[FreeQ[cTypes, "text/plain"],
		pos = FirstPosition[cTypes, "text/html"];
		If[MissingQ[pos],
			None
			,
			Replace[
				Lookup[rawbod[[First@pos]], "Contents", None]
				,
				x : Except[None] :> ImportString[x, {"HTML", "Plaintext"}]
			]
		]
		,
		pos = FirstPosition[cTypes, "text/plain"];
		Lookup[
			rawbod[[First@pos]]
			,
			"Contents"
			,
			None
		]
	]
]

(* ::Subsection:: *)
(* Attachments *)

$WolframMBoxEnvelope := "From WolframMBoxEnvelope " <> DateString[{"DayNameShort", " ", "MonthName", " ", "Day", " ", "Time", " ", "Year"}] <> "\n"

MIMEMessageGetElement[obj_?openMessageQ, "HasAttachments"] := MIMEMessageRead[obj, "HasAttachments"]
MIMEMessageGetElement[obj_?openMessageQ, "AttachmentNames"] := Lookup[Lookup[#, "Content-Disposition", <||>], "filename", Missing["NotAvailable"]] & /@ MIMEMessageRead[obj, "RawAttachments"]
MIMEMessageGetElement[obj_?openMessageQ, "AttachmentData"] :=
Module[{rawAtt, attAssoc, Name, MIMEType, RawContent, ContentTransferEncoding, dispo, modDate, byteCount, mat},
	rawAtt = MIMEMessageRead[obj, "RawAttachments"];
	attAssoc = {"Name", "MIMEType", "RawContent", "ContentTransferEncoding", "ContentDisposition", "ModificationDate", "ByteCount"};
	Name = Lookup[Lookup[#, "Content-Disposition", <||>], "filename", Missing["NotAvailable"]] & /@ rawAtt;
	MIMEType = ToLowerCase[#["Content-Type"]["Type"] <> "/" <> #["Content-Type"]["Subtype"] & /@ rawAtt];
	RawContent = #["Contents"] & /@ rawAtt;
	ContentTransferEncoding = Lookup[#, "Content-Transfer-Encoding", Missing["NotAvailable"]] & /@ wolframizeValues[rawAtt];
	dispo = Replace[Lookup[Lookup[#, "Content-Disposition", <||>], "Content-Disposition", Missing["NotAvailable"]] & /@ rawAtt, wolframDispositions, {1}];
	modDate = If[KeyExistsQ[Lookup[#, "Content-Disposition", <||>],"modification-date"], getProcessedDate[#["Content-Disposition"]["modification-date"]], Missing["NotAvailable"]] & /@ rawAtt;
	byteCount = ByteCount[#["Contents"]] & /@ rawAtt;
	mat = Transpose@{Name, MIMEType, RawContent, ContentTransferEncoding, dispo, modDate, byteCount};
	AssociationThread[attAssoc, #] & /@ mat
]

MIMEMessageGetElement[obj_?openMessageQ, "Attachments"] :=
Module[{att},
	att = MIMEMessageRead[obj, "Attachments"];
	att = Lookup[#, "FileName", Missing["NotAvailable"]] -> #["Contents"] & /@ att;
	att = Replace[att, {msg_MIMEMessage :> MIMEMessageGetElement[msg, "MessageElements"]}, {2}];
	If[Length@att > 0,
		att
		,
		{}
	]
]

MIMEMessageGetElement[obj_?openMessageQ, "AttachmentAssociations"] :=
Module[{rawAtt, att, attAssoc, Content, Name, MIMEType, RawContent, ContentTransferEncoding, dispo, modDate, byteCount, mat},
	rawAtt = MIMEMessageRead[obj, "RawAttachments"];
	att = MIMEMessageRead[obj, "Attachments"];
	attAssoc = {"Content", "Name", "MIMEType", "RawContent", "ContentTransferEncoding", "ContentDisposition", "ModificationDate", "ByteCount"};
	Content = Replace[#["Contents"], {msg_MIMEMessage :> MIMEMessageGetElement[msg, "MessageElements"]}, {0}] & /@ att;
	Name = Lookup[Lookup[#, "Content-Disposition", <||>], "filename", Missing["NotAvailable"]] & /@ rawAtt;
	MIMEType = #["ContentType"] & /@ att;
	RawContent = #["Contents"] & /@ rawAtt;
	ContentTransferEncoding = Lookup[#, "ContentTransferEncoding", Missing["NotAvailable"]] & /@ att;
	dispo = Lookup[#, "ContentDisposition", Missing["NotAvailable"]] & /@ att;
	modDate = If[KeyExistsQ[#,"modification-date"], getProcessedDate[#["modification-date"]], Missing["NotAvailable"]] & /@ att;
	byteCount = ByteCount[#["Contents"]] & /@ rawAtt;
	mat = Transpose@{Content, Name, MIMEType, RawContent, ContentTransferEncoding, dispo, modDate, byteCount};
	AssociationThread[attAssoc, #] & /@ mat
]

MIMEMessageGetElement[obj_?openMessageQ, "AttachmentSummary" | "AttachmentSummaries"] :=
Module[{rawAtt, attAssoc, Name, MIMEType, byteCount, mat},
	rawAtt = MIMEMessageRead[obj, "RawAttachments"];
	attAssoc = {"Name", "MIMEType", "ByteCount"};
	Name = Lookup[Lookup[#, "Content-Disposition", <||>], "filename", Missing["NotAvailable"]] & /@ rawAtt;
	MIMEType = ToLowerCase[#["Content-Type"]["Type"] <> "/" <> #["Content-Type"]["Subtype"] & /@ rawAtt];
	byteCount = ByteCount[#["Contents"]] & /@ rawAtt;
	mat = Transpose@{Name, MIMEType, byteCount};
	AssociationThread[attAssoc, #] & /@ mat
]

(* ::Subsection:: *)
(* QuotedMail *)

MIMEMessageGetElement[obj_?openMessageQ, "NewBodyContent"] :=
Module[{nbc},
	nbc = MIMEBodyParse[obj, "NewBodyContent"];
	If[Head@nbc === MIMEToolsException,
		None
		,
		#["Content"] & /@ nbc
	]
]

MIMEMessageGetElement[obj_?openMessageQ, "QuotedContent"] :=
Module[{qc},
	qc = MIMEBodyParse[obj, "QuotedContent"];
	If[Head@qc === MIMEToolsException,
		None
		,
		#["Content"] & /@ qc
	]
]

MIMEMessageGetElement[obj_?openMessageQ, "AllQuotedContent"] :=
Module[{aqc},
	aqc = MIMEBodyParse[obj, "AllQuotedContent"];
	If[Head@aqc === MIMEToolsException,
		None
		,
		#["Content"] & /@ aqc
	]
]

(* ::Subsection:: *)
(* Arg-checking *)

MIMEMessageGetElement[obj_, _] := CreateMIMEToolsException[40] (* If not a mimemessage, no need to check elem *)
MIMEMessageGetElement[obj_?openMessageQ, elem_] := CreateMIMEToolsException[41, elem]

(* ::Section:: *)
(* MIMEMessageCreate *)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)

Options[MIMEMessageCreate] = {"From" :> $WolframID, "ReplyTo" -> None, "To" -> None,
   "CC" -> None, "BCC" -> None, "Subject" -> None, "Body" -> None,
   "Attachments" -> None, "Inline" -> None};

$fOptionNames = {"From", "ReplyTo", "To", "CC", "BCC", "Subject", "Body", "Attachments", "Inline"};

MIMEMessageCreate[opts:OptionsPattern[]] :=
If[$libraryFileName === $Failed,
	Return@CreateMIMEToolsException[51];
	,
	Module[{optstemp, inputlist, fromArg, res},
		If[$adapterLoaded == True|| !MatchQ[loadAdapter[], _MIMEToolsException],
			fromArg = OptionValue["From"];
			optstemp = {#, Length@OptionValue@#, If[# === "From" && (OptionValue@#) === None, " ", (OptionValue@#)]
			} & /@ $fOptionNames;
			optstemp = DeleteCases[optstemp, {field_, n_, None}];
			 inputlist =ReplaceAll[optstemp,
			{{field_, 0, data_} :> {field, 1, data}, (* 0 -> 1 For number of options passed in *)
			{"Attachments", n_, v_String} :> {"Attachments", n, ImportExport`FileUtilities`GetFilePath@v},
			{"Attachments", n_, v_List} :> {"Attachments", n, ImportExport`FileUtilities`GetFilePath /@ v},
			{"Inline", n_, v_String} :> {"Inline", n, ImportExport`FileUtilities`GetFilePath@v},
			{"Inline", n_, v_List} :> {"Inline", n, ImportExport`FileUtilities`GetFilePath /@v}}
			];
			res = Quiet@lfMIMEStringCreate[Sequence@@Flatten[inputlist]];
			If[MatchQ[res, _LibraryFunctionError],
				CreateMIMEToolsException[48]
				,
				res
			]
			,
			Return@CreateMIMEToolsException[47];
		]
	]
];

(* ::Section:: *)
(* MIMEMessageOpen *)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)

MIMEMessageOpen[""] := CreateMIMEToolsException[49]

MIMEMessageOpen[message_String] :=
If[$libraryFileName === $Failed,
	Return@CreateMIMEToolsException[51];
	,
	If[$adapterLoaded == True||!MatchQ[loadAdapter[], _MIMEToolsException],
		Module[{res},
			res = Quiet@lfMIMEStringOpen[StringReplace[message, RegularExpression["^\\s*"] -> ""]];
			If[MatchQ[res, _LibraryFunctionError],
				CreateMIMEToolsException[49]
				,
				MIMEMessage[res]
			]
		]
		,
		CreateMIMEToolsException[47]
	]
]

MIMEMessageOpen[filePath_File] :=
If[$libraryFileName === $Failed,
	Return@CreateMIMEToolsException[51];
	,
	Module[{filename,rawFilePath},
		If[$adapterLoaded == True||!MatchQ[loadAdapter[], _MIMEToolsException]
			,
			rawFilePath = filePath[[1]];
			If[StringQ[rawFilePath] && FileType[rawFilePath] === File
				,
				filename = ImportExport`FileUtilities`GetFilePath[rawFilePath];
				Module[{res},
					res = Quiet@lfMIMEFileOpen[filename];
					If[MatchQ[res, _LibraryFunctionError],
						CreateMIMEToolsException[49]
						,
						MIMEMessage[res]
					]
				]
				,
				Return@CreateMIMEToolsException[52];
			]
			,
			CreateMIMEToolsException[47]
		]
	]
]

MIMEMessageOpen[_] := CreateMIMEToolsException[36]


(* ::Section:: *)
(* MIMEMessageRead *)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)

(* The output for the LHS should come from passing the RHS into lfMIMEMessageLookup, but should still be processed by the LHS field *)
fieldSource = {
	"BodyContentType" -> "BodyHeaders",
	"DecodedRawBody" -> "Body"
}

$readFields = (
	"MessageContentType" | "BodyContentType" | "Header" | "HeaderString" |
	"BodyHeaders" | "Body" | "Attachments" | "RawHeader" | "RawBodyHeaders" |
	"RawBody" | "RawAttachments" | "HasAttachments" | "DecodedRawBody"
)

Options[MIMEMessageRead] = {"MessageContentType" -> None}
(* Read will call processOutput which determines based on the field how to process and return the data *)
MIMEMessageRead[obj_?openMessageQ, field : $readFields, headers_List : {}] :=
If[$libraryFileName === $Failed,
	Return@CreateMIMEToolsException[51];
	,
	Module[{streamID},
		If[$adapterLoaded == True||!MatchQ[loadAdapter[], _MIMEToolsException],
			streamID = obj[[1]];
			res = Quiet@lfMIMEMessageLookup[streamID,Replace[field, fieldSource]];
			If[MatchQ[res, _LibraryFunctionError],
				CreateMIMEToolsException[42]
				,
				processOutput[field, res, Replace[headers, {{} -> Sequence[]}]]
			]
			,
			CreateMIMEToolsException[47]
		]
	]
]

MIMEMessageRead[obj_?openMessageQ, field_String] := CreateMIMEToolsException[33, field]
MIMEMessageRead[obj_, field_String] := CreateMIMEToolsException[35]

(* ::Subsection:: *)
(* processOutput *)
(* ------------------------------------------------------------------------- *)

(* Processing order, steps performed in this order, some steps may be omitted by certain fields*)
(*
	Group together header fields (parseHeaderFields | GroupBy) ->
	Import any byte strings/ process dates/ split email fields into lists (importObject) ->
	Rename raw header fields to Wolframized names, then introduce Wolfram speced header fields (wolframizeHeaders)
*)

processOutput["BodyHeaders" | "DecodedRawBody", lookupRet_List] := wolframizeHeaders[wolframizeValues[parseEncodings@lookupRet]]
processOutput["BodyContentType", lookupRet_List] := ToLowerCase[#["Content-Type"] & /@ lookupRet]
processOutput["Header", lookupRet_List] := wolframizeHeaders["Header", importObject[parseHeaderFields[lookupRet]]]
processOutput["Header", lookupRet_List, headers_List] := wolframizeHeaders["Header", importObject[parseHeaderFields[lookupRet]], headers];
processOutput["RawHeader", lookupRet_List] := GroupBy[lookupRet, First -> Last, Replace[#, {x_} :> x] &]
processOutput["Body", lookupRet_List] := wolframizeHeaders[importObject[#] & /@ wolframizeValues[parseEncodings@lookupRet]]
processOutput["Attachments", lookupRet_List] := wolframizeHeaders[importObject[#] & /@ wolframizeValues[lookupRet]]
processOutput["RawBodyHeaders" | "RawBody" | "RawAttachments" | "HeaderString" | "HasAttachments" , lookupRet : (_String | _List | _Symbol)] := lookupRet
processOutput["MessageContentType", lookupRet_String] := ToLowerCase[lookupRet]
processOutput[_, err_MIMEToolsException] := err

parseEncodings[l_List] :=
Block[{},
	If[AssociationQ[#],
		val = Lookup[#, "Contents"];
		If[!MissingQ[val],
			Association[#,
				"Contents" -> interpretBytes[val]
			]
			,
			#
		]
		,
		#
	]& /@ l

]

parseEncodings[expr_] := expr

interpretBytes[str_?StringQ] :=
	Quiet[
		Check[
			FromCharacterCode[ToCharacterCode[str], "UTF8"]
			,
			FromCharacterCode[ToCharacterCode[str], "ISOLatin1"]
			,
			{$CharacterEncoding::utf8}
		]
		,
		{$CharacterEncoding::utf8}
	]
interpretBytes[___] := None

(* ::Subsection:: *)
(* parseHeaderFields *)
(* ------------------------------------------------------------------------- *)


(* These fields will get pulled into an association with their corresponding Received header *)
receivedGroupMembers = {
	"DKIM-Filter",
	"DKIM-Signature",
	"DomainKey-Signature",
	"DomainKey-Filter"
};

receivedGroupQ[field_String] :=
	MemberQ[StringMatchQ[receivedGroupMembers, field, IgnoreCase -> True], True];

parseHeaderFields[header_List] :=
Module[{receivedGroup, otherGroup},
	receivedGroup = {};
	otherGroup = {};
	Scan[
		Which[
			StringMatchQ[#[[1]], "Received"], 	AppendTo[receivedGroup, {{#[[1]], #[[2]]}}],
			receivedGroupQ[#[[1]]], 			AppendTo[receivedGroup[[-1]], {#[[1]], #[[2]]}],
			True, 								AppendTo[otherGroup, {#[[1]], #[[2]]}]
		] &
		,
		header
	];
	otherGroup = GroupBy[otherGroup, First -> Last, Replace[#, {x_} :> x] &];
	receivedGroup = "DeliveryChainRecords" -> Reverse@(GroupBy[#, First -> Last, Replace[#, {x_} :> x] & ] & /@ receivedGroup);
	AssociateTo[otherGroup, receivedGroup]
]

(* ::Subsection:: *)
(* importObject *)
(* ------------------------------------------------------------------------- *)

(*
	These functions are used process the data of specific header fields.
	Turns To/Cc/Reply to into a list, processes some image attachments,
	and turns date strings into date objects.
*)

(* List of formats MIMETools will attempt to import *)
mimeToFormat = {
	"octet" -> "Octet", (* This is the only circumstance where we attempt to use ImportString without a format specifier *)
	"octet-stream" -> "Octet",
	"basic" -> "AU",
	"png" -> "PNG",
	"x-wav" -> "WAV",
	"html" -> "HTML",
	"jpeg" -> {"JPEG", "ImageNoExif"},
	"jpg" -> {"JPEG", "ImageNoExif"},
	"gif"-> "GIF",
	"pdf" -> "PDF",
	"x-aiff" -> "AIFF",
	"x-msvideo" -> "AVI",
	"x-font-bdf" -> "BDF",
	"bmp" -> "BMP",
	"x-bzip2" -> "BZIP2",
	"csv" -> "CSV",
	"vnd.collada+xml" -> "DAE",
	"vnd.dxf" -> "DXF",
	"x-hdf" -> "HDF",
	"vnd.iccprofile" -> "ICC",
	"x-icon" -> "ICO",
	"calendar" -> "ICS",
	"jpm" -> "JPEG2000",
	"vnd.google-earth.kml+xml" -> "KML",
	"x-latex" -> "LaTeX",
	"mathml+xml" -> "MathML",
	"mbox" -> "MBOX",
	"x-msaccess" -> "MDB",
	"mesh" -> "MESH",
	"midi" -> "MIDI",
	"mpeg" -> "MP3",
	"x-netcdf" -> "NetCDF",
	"x-cdf" -> "NASACDF",
	"vnd.oasis.opendocument.spreadsheet" -> "ODS",
	"ogg" -> "OGG",
	"x-portable-bitmap" -> "PBM",
	"x-pcx" -> "PCX",
	"vnd.palm" -> "PDB",
	"x-portable-graymap" -> "PGM",
	"x-portable-anymap" -> "PNM",
	"x-portable-pixmap" -> "PPM",
	"quicktime" -> "QuickTime",
	"rss+xml" -> "RSS",
	"rtf" -> "RTF",
	"vnd.ms-pki.stl" -> "STL",
	"vnd.sun.xml.calc" -> "SXC",
	"x-tar" -> "TAR",
	"tiff" -> "TIFF",
	"tab-separated-values" -> "TSV",
	"x-uuencode" -> "UUE",
	"x-vcard" -> "VCF",
	"x-vcalendar" -> "VCS",
	"webp" -> "WebP",
	"x-xbitmap" -> "XBM",
	"xhtml+xml" -> "XHTML",
	"vnd.ms-excel" -> "XLS",
	"vnd.openxmlformats-officedocument.spreadsheetml.sheet" -> "XLSX",
	"xml" -> "XML",
	"x-xyz" -> "XYZ",
	"zip" -> "ZIP",
	_ -> Sequence[]
}

importObject[assoc_Association] :=
Module[{retVal},
	retVal = Association[KeyValueMap[
		Switch[#1,
			"Return-Path", #1 -> StringDelete[#2, RegularExpression["[<>]"]],
			"To", #1 -> MIMETools`Developer`commaSplitIgnoreQuotes[assoc["To"]], (* Regex to split on comma ignoring commas inside quotes *)
			"Cc", #1 -> MIMETools`Developer`commaSplitIgnoreQuotes[assoc["Cc"]],
			"CC", #1 -> MIMETools`Developer`commaSplitIgnoreQuotes[assoc["CC"]],
			"Bcc", #1 -> MIMETools`Developer`commaSplitIgnoreQuotes[assoc["Bcc"]],
			"BCC", #1 -> MIMETools`Developer`commaSplitIgnoreQuotes[assoc["BCC"]],
			"Reply-To", #1 -> MIMETools`Developer`commaSplitIgnoreQuotes[assoc["Reply-To"]],
			"References", #1 -> StringSplit[#2, RegularExpression["\\s+"]],
			"X-Originating-IP", #1 -> IPAddress[StringDelete[#2, RegularExpression["[^0-9.]"]]],
			_, #1 -> #2] &, assoc
		]
	];
	If[!((ToLowerCase[assoc["Type"]] === "text" && ToLowerCase[assoc["Subtype"]] === "plain") || ToLowerCase[assoc["Type"]] === "multipart"),
		AssociateTo[retVal, "Contents" -> importAttachment[retVal, ToLowerCase[retVal["Subtype"]] /. mimeToFormat]];
	];
	retVal
]

importObject[expr_] := expr;

(* ImportFailure exception for when ImportString fails *)
(* These calls assume the encoding is on our white list *)
importAttachment[assoc_Association, format: (_String | _List)] :=
	Replace[Quiet@ImportString[Lookup[assoc, "Contents", ""], format], $Failed :> CreateMIMEToolsException[39, None, {wolframizeHeaders[wolframizeValues[assoc]], format}], {0}]

importAttachment[assoc_Association, "Octet"] :=
	Replace[Lookup[assoc, "Contents", ""], $Failed :> CreateMIMEToolsException[39, None, {wolframizeHeaders[wolframizeValues[assoc]], "Binary"}], {0}]

(* Without a mimetype -> format, return a UnsupportedFormat exception. *)
importAttachment[assoc_Association] := CreateMIMEToolsException[43, None, {wolframizeHeaders[wolframizeValues[assoc]]}]
(* MIMEMessage objects have already been imported at the C level, and should be passed back untouched. *)
importAttachment[msg_?openMessageQ] := msg
(* Exceptions also untouched *)
importAttachment[msg_MIMEToolsException, format: (_String | _List)] := msg
importAttachment[msg_MIMEToolsException] := msg

MIMETools`Developer`commaSplitIgnoreQuotes[str_String] :=
If[StringContainsQ[str, "\""],
	StringTrim /@ StringSplit[str, RegularExpression[",(?=([^\\\"]*\"[^\\\"]*\\\")*[^\\\"]*$)"]]
	,
	StringTrim /@ StringSplit[str, ","]
]

(* ::Subsection:: *)
(* wolframizeContentTransferEncoding *)
(* ------------------------------------------------------------------------- *)


wolframEncodings = {
	"7bit" -> "7Bit",
	"8bit" -> "8Bit",
	"binary" -> "Binary",
	"quoted-printable" -> "QuotedPrintable",
	"base64" -> "Base64"
}

wolframDispositions = {
	"inline" -> "Inline",
	"attachment" -> "Attachment"
}

wolframizeValues[l_List] :=
Association[KeyValueMap[
	Switch[#1,
		"Content-Transfer-Encoding",
			#1 -> Replace[ToLowerCase[#2], wolframEncodings]
		,
		"Content-Disposition",
			#1 -> Replace[ToLowerCase[#2], wolframDispositions]
		,
		"Content-Type"
		,
			#1 -> ToLowerCase[#2]
		,
		"Type",
			#1 -> ToLowerCase[#2]
		,
		"Subtype",
			#1 -> ToLowerCase[#2]
		,
		_,
			#1 -> #2
	] &
	,
	#
]] & /@ l;

wolframizeValues[expr_] := expr

(* ::Subsection:: *)
(* wolframizeHeaders *)
(* ------------------------------------------------------------------------- *)
(* First rename header fields, then introduce Wolfram specific headers *)

(* Renamed headers *)
wolframHeaders = {
	"name" -> "Name",
	"charset" -> "CharacterEncoding",
	"boundary" -> "Boundary",
	"filename" -> "FileName",
	"Content-Transfer-Encoding" -> "ContentTransferEncoding",
	"Content-Disposition" -> "ContentDisposition",
	"To" -> "ToList",
	"Cc" -> "CcList",
	"CC" -> "CcList",
	"Bcc" -> "BccList",
	"BCC" -> "BccList",
	"Return-Path" -> "ReturnPath",
	"Reply-To" -> "ReplyToList",
	"In-Reply-To" -> "ReplyToMessageID",
	"Message-ID" -> "MessageID",
	"Content-Type" -> "ContentType",
	"MIME-Version" -> "MIMEVersion",
	"X-Mailer" -> "OriginatingMailClient",
	"User-Agent" -> "OriginatingMailClient",
	"X-Originating-IP" -> "OriginatingIPAddress",
	"Message-Id" -> "MessageID",
	"DispositionNotificationTo" -> "ReturnReceiptRequested",
	"Disposition-Notification-To" -> "ReturnReceiptRequested",
	"List-Subscribe" -> "ListSubscribe",
	"List-Unsubscribe" -> "ListUnsubscribe",
	"References" -> "ReferenceMessageIDList"
}

wolframizeHeaders[assoc_Association] :=
KeyMap[Replace[#, wolframHeaders, {0}] &, assoc]

wolframizeHeaders[l_List] :=
KeyMap[Replace[#, wolframHeaders, {0}] &, #] & /@ l

wolframizeHeaders["Header", assoc_Association, headers_List:{}] :=
Module[{a, includedFields},
	a = KeyMap[Replace[#, wolframHeaders, {0}] &, assoc];
	If[Length[headers] > 0,
		includedFields = Select[$nonSpecFields, StringMatchQ[#[[1]], headers] &];
		If[KeyExistsQ[a, "OriginatingIPAddress"], includedFields = DeleteCases[includedFields, {_?originatingIPTest, _}]];
		,
		includedFields = $nonSpecFields;
	];
	(AssociateTo[a, #[[1]] -> processHeaderField[#[[1]], a[#[[2]]]]]) & /@ includedFields;
	a = KeyTake[a, MIMEMessageGetElement["Header"]];
	AssociateTo[a, # -> Lookup[$emptyReturnValues, #, Missing["NotAvailable"]]] & /@ Complement[MIMEMessageGetElement["Header"], Keys[a]];
	AssociateTo[a, "MIMEMessageClass" -> "Header"];
	AssociateTo[a, "ContentType" -> ToLowerCase[a["ContentType"]]];
	AssociateTo[a, "Type" -> ToLowerCase[a["Type"]]];
	AssociateTo[a, "Subtype" -> ToLowerCase[a["Subtype"]]];
	(* a = KeyValueMap[If[StringContainsQ[#1, "List"], #1 -> Replace[#2, Missing["NotAvailable"] -> {}], #1 -> #2] &, a]; *)
	KeySortBy[a, sortFxn]
]

wolframizeHeaders[obj_] := obj;

$emptyReturnValues =
Association[
	"Subject" -> None,
	"ReplyToMessageID" -> None,
	"ReturnReceiptRequested" -> None,
	"ToList" -> {},
	"CcList" -> {},
	"BccList" -> {},
	"ReplyToList" -> {},
	"ReferenceMessageIDList" -> {}
]


(* Wolfram specific headers *)
(* These parameters specify the wolframized field we're processing (target), and the source field from the MIME message, for processHeaderField *)
$nonSpecFields = {
	(*{target, source} *)
	{"FromAddress", "From"},
	{"FromName", "From"},
	{"ToAddressList", "ToList"},
	{"ToNameList", "ToList"},
	{"CcAddressList", "CcList"},
	{"CcNameList", "CcList"},
	{"BccAddressList", "BccList"},
	{"BccNameList", "BccList"},
	{"ReplyToAddressList", "ReplyToList"},
	{"ReplyToNameList", "ReplyToList"},
	{"DeliveryChainHostnames", "DeliveryChainRecords"},
	{"OriginatingHostname", "DeliveryChainRecords"},
	{"OriginatingIPAddress", "DeliveryChainRecords"},
	{"OriginatingCountry", "DeliveryChainRecords"},
	{"OriginatingDate", "Date"},
	{"OriginatingTimezone", "Date"},
	{"ServerOriginatingDate", "DeliveryChainRecords"},
	{"ServerOriginatingTimezone", "DeliveryChainRecords"},
	{"Precedence", "Precedence"}
}

originatingIPTest = StringMatchQ[#, "OriginatingIPAddress"] &

(* processHeaderField will generate the field specified by the target, using the source data *)
(* Precedence *)
processHeaderField["Precedence", p_String] := p
processHeaderField["Precedence", _Missing] := None
(* Email address/name lists *)
processHeaderField["FromAddress" | "ToAddressList" | "CcAddressList" | "BccAddressList" | "ReplyToAddressList", addresses : (_List | _String)] := Replace[Interpreter["EmailAddress"][addresses], x_Failure :> CreateMIMEToolsException[37, x], {0,1}]
processHeaderField[param : ("FromName" | "ToNameList" | "CcNameList" | "BccNameList" | "ReplyToNameList"), str_String] := First[processHeaderField[param, {str}]]
processHeaderField[param : ("FromName" | "ToNameList" | "CcNameList" | "BccNameList" | "ReplyToNameList"), addresses_List] := With[{split = StringSplit[addresses, "<"]}, If[Length[#] > 1, StringTrim[First[#]], Missing["NotAvailable"]] & /@ split]
(* Processed from the last received field, IFF there is a from part, with a valid domain *)
processHeaderField["OriginatingHostname", l_List] :=
If[Length@l > 0,
	Module[{lastReceived},
		lastReceived = First[#["Received"] & /@ l];
		If[StringContainsQ[StringTake[lastReceived, 10], "from"] && StringContainsQ[First[StringSplit[lastReceived, {"by", "via", "with", "id", "for"}]], RegularExpression[domainRegEx]],
			First[processHeaderField["DeliveryChainHostnames", l]]
			,
			Missing["NotAvailable"]
		]
	]
	,
	Missing["NotAvailable"]
]

(* Processed from the last received field, IFF there is a from part*)
processHeaderField["OriginatingIPAddress", l_List] :=
If[Length@l > 0,
	Module[{lastReceived},
		lastReceived = First[#["Received"] & /@ l];
		If[StringContainsQ[StringTake[lastReceived, 10], "from"],
			IPObject[Replace[processReceivedField["IPAddress", #["Received"] & /@ l], x_Failure :> CreateMIMEToolsException[38, x], {0,1}]]
			,
			Missing["NotAvailable"]
		]
	]
	,
	Missing["NotAvailable"]
]
(* Processed off the result of OriginatingIPAddress *)
processHeaderField["OriginatingCountry", l_List] := geoCountry[processHeaderField["OriginatingIPAddress", l]]
(* Processed from the last received field, there is guaranteed to always be a date at the end in a received field, so no need to check *)
processHeaderField["OriginatingDate", date_String] := getProcessedDate[date]
processHeaderField["OriginatingTimezone", date_String] := Replace[processHeaderField["OriginatingDate", date], {res_ /; !DateObjectQ[res] -> Missing["NotAvailable"], res_ :> res["TimeZone"]}]
processHeaderField["OriginatingDate", date_Missing] := Missing["NotAvailable"]
processHeaderField["OriginatingTimezone", date_Missing] := Missing["NotAvailable"]
(* Pulled from the Date field *)
processHeaderField["ServerOriginatingDate", l_List] :=
If[Length@l > 0,
	processReceivedField["DateTime", #["Received"] & /@ l]
	,
	Missing["NotAvailable"]
]
processHeaderField["ServerOriginatingTimezone", l_List] :=
If[Length@l > 0,
	Replace[processHeaderField["ServerOriginatingDate", l], {res_ /; !DateObjectQ[res] -> Missing["NotAvailable"], res_ :> res["TimeZone"]}]
	,
	Missing["NotAvailable"]
]
(* Process the received fields in chronological order, getting the trace of host names *)
processHeaderField["DeliveryChainHostnames", l_List] :=
If[Length@l > 0,
	Flatten[
		Map[First[#, Sequence[]] &,
				Replace[Replace[Map[StringCases[#, RegularExpression[domainRegEx]] &,
									StringSplit[
										StringSplit[selectFromBy[#]
										, {"from", "by"}]
									, " "] & /@ (#["Received"] & /@ l)
								, {2}]
						, {} -> Sequence[], {3}]
				, {} -> Sequence[], {2}], {2}
		]
	]
	,
	Missing["NotAvailable"]
]


(* List of domains MIMETools recognizes as possible host names *)
domainSuffixes = {
  "com",
  "edu",
  "org",
  "net",
  "gov",
  "info",
  "biz",
  "me",
  "info",
  "us",
  "eu",
  "uk",
  "ca"
  }

domainRegEx =
"[A-Za-z0-9\\.\\-]+\\.(" <> StringJoin[Riffle[domainSuffixes, "|"]] <> ")"

selectFromBy[str_String] := StringReplace[str, {
	RegularExpression["\\svia\\s.*$"] -> "",
	RegularExpression["\\swith\\s.*$"] -> "",
	RegularExpression["\\sid\\s.*$"] -> "",
	RegularExpression["\\sfor\\s.*$"] -> ""}]

$headerFields1 = (
	"DeliveryChainHostnames" | "OriginatingHostname" | "OriginatingIPAddress" |
	"OriginatingCountry" | "ServerOriginatingDate" | "ServerOriginatingTimezone"
)

processHeaderField[type : $headerFields1, str_String] :=
	processHeaderField[type, StringSplit[str, "from"]]

$headerFields2 = (
	"FromAddress" | "ToAddressList" | "CcAddressList" | "ReplyToAddressList" | "FromName" |
	"ToNameList" | "CcNameList" | "DeliveryChainHostnames" | "DeliveryChainRecords" |
	"OriginatingHostname" | "OriginatingIPAddress" | "OriginatingCountry" |
	"ServerOriginatingDate" | "ServerOriginatingTimezone" | "BccNameList" | "ReplyToNameList" |
	"BccAddressList" | "ReplyToAddressList"| "Precedence"
)
processHeaderField[$hederFields2, expr_] := expr

$headerFields3 = (
	"DeliveryChainHostnames" | "DeliveryChainRecords" | "OriginatingHostname" |
	"OriginatingIPAddress" | "OriginatingCountry" | "ServerOriginatingDate" |
	"ServerOriginatingTimezone" | "FromAddress" | "FromName"
)

processHeaderField[$headerFields3, expr_Missing] :=
	Missing["NotAvailable"]

$headerFields4 = (
	"ToAddressList" | "CcAddressList" | "ToNameList" | "CcNameList" |
	"BccNameList" | "ReplyToNameList" | "BccAddressList" | "BccNameList" |
	"ReplyToAddressList" | "ReplyToAddressList"
)

processHeaderField[$headerFields4, expr_Missing] :=
	{}

(* processHeaderField helper functions *)

IPObject[ip_String] := IPAddress[ip]
IPObject[excp_] := excp

processReceivedField[type_, l_List] := processReceivedField1[type, First[l]]
processReceivedField[_, expr_] := expr

processReceivedField1[type_, received_] :=
 With[{res =
	processReceivedField2[type,
	 Switch[type, "IPAddress", First, _, Last][
	  StringSplit[received, {"\n", "by", "for"}]]]},
  If[Length[res] > 0, First[res], Missing["NotAvailable"]]]

processReceivedField2["IPAddress", line_] :=
 Interpreter["IPAddress"][
  StringCases[
   line, (ip : ((HexadecimalCharacter | "." |
		   ":") ...) /; (stringLength[ip] > 5)) :> ip]]

processReceivedField2["DateTime", line_] :=
 {Replace[getProcessedDate[
   StringCases[line,
	";" ~~ date : (__) ~~ ("(" | EndOfString) :> date]], _Failure ->
   Missing["NotAvailable"]]}

stringLength[str_String] := StringLength[str]
stringLength[_] := 0

getProcessedDate[expr_List] :=
	Module[{input,processedDate,date1,r},
		input = StringTrim[expr[[1]]];
		If[StringCases[input,{RegularExpression["\\D*\\,\\ +\\d*\\ +\\D*\\d*\\ +\\d*\\:\\d*\\:\\d*\\ +\\-\\d*"]|RegularExpression["\\D*\\,\\ +\\d*\\ +\\D*\\d*\\ +\\d*\\:\\d*\\:\\d*\\ +\\+\\d*"]}]=!={},
			date1 = StringReplace[input, {RegularExpression["\\s+"] -> " "}];
			r=Flatten@StringSplit[date1,{" "|":"}];
			processedDate=ToExpression[Flatten[{r[[4]],r[[3]],r[[2]],r[[{5,6,7}]]}]/.{"Jan"->1,"Feb"->2,"Mar"->3,"Apr"->4,"May"->5,"Jun"->6,"Jul"->7,"Aug"->8,"Sep"->9,"Oct"->10,"Nov"->11,"Dec"->12}]
			,
			Return[Missing["NotAvailable"]]
		];
		getDate[processedDate,Internal`StringToDouble[StringReplace[r[[8]],"0" -> ""]<>"."]]
	]

getProcessedDate[expr_String] :=
	Module[{input, processedDate,date1,r},
		input = StringTrim[expr];
		If[StringCases[input,{RegularExpression["\\D*\\,\\ +\\d*\\ +\\D*\\d*\\ +\\d*\\:\\d*\\:\\d*\\ +\\-\\d*"]|RegularExpression["\\D*\\,\\ +\\d*\\ +\\D*\\d*\\ +\\d*\\:\\d*\\:\\d*\\ +\\+\\d*"]}]=!={},
			date1 = StringReplace[expr, {RegularExpression["\\s+"] -> " "}];
			r=Flatten@StringSplit[date1,{" "|":"}];
			processedDate=ToExpression[Flatten[{r[[4]],r[[3]],r[[2]],r[[{5,6,7}]]}]/.{"Jan"->1,"Feb"->2,"Mar"->3,"Apr"->4,"May"->5,"Jun"->6,"Jul"->7,"Aug"->8,"Sep"->9,"Oct"->10,"Nov"->11,"Dec"->12}]
			,
			Return[Missing["NotAvailable"]]
		];
		getDate[processedDate,Internal`StringToDouble[StringReplace[r[[8]],"0" -> ""]<>"."]]
	]

getDate[a_, tz_] := Quiet[Check[DateObject[a, TimeZone -> tz], Missing["NotAvailable"]]]

(* This requires FindGeoLocation["ip"] to work. Once FindGeoLocation is fixed, this will be turned on *)
(* geoCountry[ip_String] := First[GeoNearest[Entity["Country"],GeoPosition[{40.11`, -88.24`}]]]  *)
(* Above demonstrates the functionality. Below we replace GeoPosition[{40.11`, -88.24`}] with FindGeoPosition[ip] for the final implmenetation*)
(* geoCountry[ip_String] := First[GeoNearest[Entity["Country"],FindGeoPosition[ip]]] *)
geoCountry[_] := Missing["NotAvailable"]

(* Sort based on the order of MIMEMessageGetElement["Header"] *)
sortFxn := (Position[MIMEMessageGetElement["Header"], #] /. ({} :> {{Length@MIMEMessageGetElement["Header"] + 1}})) &

(* End wolframize functions *)

(* Scan through each requested header field, accumulating the outputs,
	allowing the whole function to escape/return error for each individual field *)
MIMEMessageRead[obj_?openMessageQ, fields_List] :=
If[$libraryFileName === $Failed,
	Return@CreateMIMEToolsException[51];
	,
	Module[{streamID, errorCond = False, error},
		If[$adapterLoaded == True||!MatchQ[loadAdapter[], _MIMEToolsException],
			streamID = obj[[1]];
			res = (Reap[Scan[
				((partRes = MIMEMessageRead[obj, #];
				If[Head@partRes === MIMEToolsException,
					errorCond = True; error = partRes; Return[];
					,
					Sow[partRes]
				]) &), fields]]);

			If[errorCond,
				Return[error]
				,
				Return[Flatten[res[[2, 1]]]]
			]
			,
			CreateMIMEToolsException[47]
		]
	]
]
MIMEMessageRead[obj_, fields_List] := CreateMIMEToolsException[35]


(* ::Section:: *)
(* MIMEMessageClose *)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)


MIMEMessageClose[obj_?openMessageQ] :=
If[$libraryFileName === $Failed,
	Return@CreateMIMEToolsException[51];
	,
	Module[{streamID},
		If[$adapterLoaded == True||!MatchQ[loadAdapter[], _MIMEToolsException],
			streamID = obj[[1]];
			res = Quiet@lfMIMEMessageClose[streamID];
			If[MatchQ[res, _LibraryFunctionError],
				CreateMIMEToolsException[50]
				,
				Null
			]
			,
			CreateMIMEToolsException[47]
		]
	]
]

MIMEMessageClose[expr_] := CreateMIMEToolsException[44]


(* ::Section:: *)
(* MIMEMessageParse *)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)


MIMEMessageParse[message_] :=
If[$libraryFileName === $Failed,
	Return@CreateMIMEToolsException[51];
	,
	Module[{obj, retVal},
		If[$adapterLoaded == True||!MatchQ[loadAdapter[], _MIMEToolsException],
			obj = MIMEMessageOpen[message];
			If[Head@obj === MIMEToolsException,
				Return[obj]
				,
				retVal = Flatten[MIMEMessageRead[obj, #] & /@ {"Header", "Body"}]
			];
			MIMEMessageClose[obj];
			retVal
			,
			CreateMIMEToolsException[47]
		]
	]
]

MIMEMessageParse[message_, field_String] :=
If[$libraryFileName === $Failed,
	Return@CreateMIMEToolsException[51];
	,
	Module[{obj,retVal},
		If[$adapterLoaded == True||!MatchQ[loadAdapter[], _MIMEToolsException],
			obj = MIMEMessageOpen[message];
			If[Head@obj === MIMEToolsException,
				Return[obj]
				,
				retVal = MIMEMessageRead[obj,field]
			];
			MIMEMessageClose[obj];
			retVal
			,
			CreateMIMEToolsException[47]
		]
	]
]

MIMEMessageParse[msg_MIMEMessage] := CreateMIMEToolsException[34]
MIMEMessageParse[msg_MIMEMessage,field_String] := CreateMIMEToolsException[34]

MIMEMessageRawParse[message_] :=
If[$libraryFileName === $Failed,
	Return@CreateMIMEToolsException[51];
	,
	Module[{obj, retVal},
		If[$adapterLoaded == True||!MatchQ[loadAdapter[], _MIMEToolsException],
			obj = MIMEMessageOpen[message];
			If[Head@obj === MIMEToolsException,
				Return[obj]
				,
				retVal = Flatten[MIMEMessageRead[obj, #] & /@ {"RawHeader", "RawBody"}]
			];
			MIMEMessageClose[obj];
			retVal
			,
			CreateMIMEToolsException[47]
		]
	]
]

MIMEMessageRawParse[msg_MIMEMessage] := CreateMIMEToolsException[34]

(* ::Section:: *)
(* getOpenMIMEMessage *)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)

getOpenMIMEMessage:=
Module[{res},
	If[$libraryFileName === $Failed,
		Return@CreateMIMEToolsException[51];
		,
		If[$adapterLoaded == True||!MatchQ[loadAdapter[], _MIMEToolsException],
			res = Quiet@lfGetOpenMIMEStream["MIME"];
			If[MatchQ[res, _LibraryFunctionError],
				Return@CreateMIMEToolsException[46];
			];
			Map[MIMEMessage,res]
			,
			Return@CreateMIMEToolsException[47];
		]
	]
]

openMessageQ[obj_MIMEMessage] := MemberQ[getOpenMIMEMessage, obj]
openMessageQ[obj_] := False


(* ::Section:: *)
(* MIMEBodyParse*)
(* ------------------------------------------------------------------------- *)
(* ------------------------------------------------------------------------- *)


(* ::Subsection:: *)
(* AttributionQ *)
(* ------------------------------------------------------------------------- *)


AttributionQ[message_String] := Or[
	(* some mail client use the format
		------ On date *** wrote:
	so need to capture the - *)
	StringMatchQ[message, RegularExpression["^" ~~ "-*\ *" ~~ "On.*wrote:$"]],
	StringMatchQ[message, RegularExpression["^Il giorno.*ha scritto:$"]]
]



(* ::Subsection:: *)
(* ExtractAttributionInfo *)
(* ------------------------------------------------------------------------- *)


(* I use the assumption that, the format is usually something like
On XXX, YYY wrote:
so first take out YYY by , and wrote:
then seperate YYY into author name and address *)

(* currently, I can only handle a line. but will need to expand to cover blocks, that's where message_List *)

ExtractAttributionInfo[message_List] := If[Length@message > 0, ExtractAttributionInfo[message[[1]]], {}]

ExtractAttributionInfo[message_String] := Module[{author, date, authorAddress, authorName},

	author = StringCases[message, __ ~~ "," ~~ name__ ~~ "wrote:" -> name];
	date = StringCases[message, "On " ~~ time__ ~~ author ~~ __ -> time];
	authorAddress = Flatten[StringCases[author, address:RegularExpression["\\w*@\\w*"] -> address], 1];
	authorAddress = If[Length[authorAddress]>0, authorAddress[[1]], ""];
	authorName = Flatten[StringCases[author, name__  ~~ authorAddress  ~~ __ -> name], 1];
	authorName = If[Length[authorName] > 0, authorName[[1]], ""];
	Association[{"Author" -> If[Length[author] > 0, StringTrim@author[[1]], Missing["NotAvailable"]],
	"Date" -> If[Length[date] > 0, StringTrim[date[[1]], ","], Missing["NotAvailable"]],
	"AuthorAddress" -> If[StringLength[authorAddress] > 0, authorAddress, Missing["NotAvailable"]],
	"AuthorName" -> If[StringLength[authorName] > 0, StringTrim[authorName, RegularExpression["[<\ ]*"]], Missing["NotAvailable"]]}]
]

(* ::Subsection:: *)
(* ReplyIndicatorLevel *)
(* ------------------------------------------------------------------------- *)

Options[ReplyIndicatorLevel] = {
	"ReplyIndicator" ->Automatic
};

ReplyIndicatorLevel[data_String, opts : OptionsPattern[]] := Module[{$ReplyIndicators},
	$ReplyIndicators = ConstructReplyIndicatorRex@OptionValue["ReplyIndicator"];
	With[{trimmed = StringReplace[data, "\ " -> ""]},
		If[Length[StringCases[trimmed, RegularExpression["^[" ~~ StringJoin[$ReplyIndicators] ~~ "]+$"]]] > 0,
			StringLength[trimmed]
			,
			0
		]
	]
]



(* ::Subsection:: *)
(* ReplyIndicatorTrim *)
(* ------------------------------------------------------------------------- *)


Options[ReplyIndicatorTrim] = {
	"ReplyIndicator" -> Automatic
};

ReplyIndicatorTrim[data_List, opts : OptionsPattern[]] :=
	Map[ReplyIndicatorTrim[#, "ReplyIndicator" -> OptionValue["ReplyIndicator"]] &, data]

ReplyIndicatorTrim[data_String, opts : OptionsPattern[]] :=
	StringTrim[StringReplace[data, RegularExpression["^[" ~~ ConstructReplyIndicatorRex@OptionValue["ReplyIndicator"] ~~ "]*"] -> ""]]



(* ::Subsection:: *)
(* GroupIntoParagraph *)
(* ------------------------------------------------------------------------- *)


GroupIntoParagraph[index_List] := Split[index, #2 - #1 == 1 &]


(* ::Subsection:: *)
(* CreateMailElement *)
(* ------------------------------------------------------------------------- *)


Options[CreateMailElement] =
{
	"UUID" -> Automatic,
	"Level" -> Automatic,
	"MessageID" -> Automatic,
	"AttributionLineReference" -> Automatic,
	"AuthorName" -> Automatic,
	"AuthorAddress" -> Automatic,
	"Author" -> Automatic,
	"Date" -> Automatic,
	"InReplyToReference" -> Automatic,
	"AnswerReference" -> Automatic,
	"OriginalMailReference" -> Automatic,
	"Type" -> Automatic,
	"StartEndLineNumbers" -> Automatic
}

CreateMailElement[msg_String, opts : OptionsPattern[]] := MailElement[msg,
	Association[
		"UUID" -> Replace[OptionValue@"UUID", Automatic -> Missing["NotAvailable"]],
		"Level" -> Replace[OptionValue@"Level", Automatic -> Missing["NotAvailable"]],
		"MessageID" -> Replace[OptionValue@"MessageID", Automatic -> Missing["NotAvailable"]],
		"AttributionLineReference" -> Replace[OptionValue@"AttributionLineReference", Automatic -> Missing["NotAvailable"]],
		"AuthorName" -> Replace[OptionValue@"AuthorName", Automatic -> Missing["NotAvailable"]],
		"AuthorAddress" -> Replace[OptionValue@"AuthorAddress", Automatic -> Missing["NotAvailable"]],
		"Author" -> Replace[OptionValue@"Author", Automatic -> Missing["NotAvailable"]],
		"Date" -> Replace[OptionValue@"Date", Automatic -> Missing["NotAvailable"]],
		"InReplyToReference" -> Replace[OptionValue@"InReplyToReference", Automatic -> Missing["NotApplicable"]],
		"AnswerReference" -> Replace[OptionValue@"AnswerReference", Automatic -> Missing["NotApplicable"]],
		"OriginalMailReference" -> Replace[OptionValue@"OriginalMailReference", Automatic -> Missing["NotAvailable"]],
		"Type" -> Replace[OptionValue@"Type", Automatic -> Missing["NotAvailable"]],
		"StartEndLineNumbers" -> Replace[OptionValue@"StartEndLineNumbers", Automatic -> Missing["NotAvailable"]]
	]
]

(* ::Subsection:: *)
(* ConstructReplyIndicatorRex *)
(* ------------------------------------------------------------------------- *)

(* this function make a list of characters into a string, making it ready for Rex*)

ConstructReplyIndicatorRex[indicators_] := With[{tmp = indicators /. {Automatic -> {">", "|", "\ "}}}, StringJoin[tmp]]

(* ::Subsection:: *)
(* ExtractReplyIndicatorToken *)
(* ------------------------------------------------------------------------- *)

Options[ExtractReplyIndicatorToken] = {
	"ReplyIndicator" -> Automatic
};

ExtractReplyIndicatorToken[data_String, opts : OptionsPattern[]] :=
Module[{position},
	(* position is the first character that's not a reply indicator *)
	position = StringPosition[data, RegularExpression["[^" ~~ ConstructReplyIndicatorRex[OptionValue["ReplyIndicator"]] ~~ "]"]];
	(* if found a non replyIndicator character, take anything before that.  If everything is replyIndicator, return the whole thing *)
	Switch[position, {__}, StringTake[data, position[[1]][[1]] - 1], {___},StringTake[data, {1,-1}]]
]

(* ::Subsection:: *)
(* MessageParsingHelper*)
(* ------------------------------------------------------------------------- *)
(* The returned structure is going to be two list.  The  first is a list of message blocks, where a block is {{line# in this block}, block_level, UUID, InReplyToReference, AnswerReference}, and second list is list of attribution blocks, with the same format without the last 2 field. *)

Options[MessageParsingHelper] = {
	"ReplyIndicator" -> Automatic
};

MessageParsingHelper[data_List, opts : OptionsPattern[]] :=
Module[
	{ paddedData, attributionTemp, currentLineNumber = 0,
	blockCharLength = 0, currentBlockStartNumber = 1, currentBlockEndNumber = 1,
	previousBlockLevel = 0, previousBlockContentType = 0, currentLineContentType = 0, finalMessageBlock = {},
	finalAttributionBlock = {}, pairedBlockRule},

	(* previousContentType used to denote what type the previous line/block is.  0 is content, 1 is attribution
		this is needed because we not only have level change, we also consider attribution as a block itself *)
	(* we also need to pad the data since we want to group the message blocks in 1 pass *)

	paddedData = Append[data, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> end"];
	Scan[Function[arg,
		With[{currentLineLevel = ReplyIndicatorLevel[ExtractReplyIndicatorToken[arg, "ReplyIndicator" -> Automatic]],
			currentLineContent = ReplyIndicatorTrim[arg]
			},
			currentLineNumber++;
			currentLineContentType = If[AttributionQ[currentLineContent], 1, 0];
			If[currentLineLevel == previousBlockLevel && currentLineContentType == previousBlockContentType,
				(* test to see if we are starting a block with content *)
				(* if length total length is 0. means leading blank space. we advance the beginning
				of the block to the next line *)
				If[blockCharLength + (StringLength@currentLineContent) == 0,
					currentBlockStartNumber = currentLineNumber + 1
					,
					blockCharLength = blockCharLength + StringLength@currentLineContent
				];
				(* only advance the end of blcok if encounter a line with characters.  This takes
				care of trailing blank lines in a block *)
				If[StringLength@currentLineContent > 0, currentBlockEndNumber = currentLineNumber]
				,
				If[blockCharLength > 0,
					(* we create the attribution block with UUID, but not the message block for now. because we'll be combining blocks in later steps *)
					Switch[previousBlockContentType,
						1,
						AppendTo[finalAttributionBlock, {Range[currentBlockStartNumber, currentBlockEndNumber], previousBlockLevel, CreateUUID[]}],
						0,
						AppendTo[finalMessageBlock, {Range[currentBlockStartNumber, currentBlockEndNumber], previousBlockLevel}]
					]
				];
				blockCharLength = StringLength@currentLineContent;
				currentBlockStartNumber = If[blockCharLength > 0, currentLineNumber, currentLineNumber + 1];
				currentBlockEndNumber = If[blockCharLength > 0, currentLineNumber, currentLineNumber + 1];
				previousBlockContentType = currentLineContentType;
				previousBlockLevel = currentLineLevel
			];

		]],
		paddedData
	];

	(* different mail client use different standard.
	---------------------
	Unquoted....

	[>] Attribution line
	> quoted
	---------------------
	on level 0, some client has > and some doesn't.  we need to align it to level 1
	this process needs to propogate through the list of attribution lines through out the whole message.  Idea is, first, in each reply, there is only 1 attribution
	   line.  second, the attribution line always appear before the quoted message.  I use a temp list that holds rules for line# \[Rule] level#.
	now scan through the attributions lines' level, the level has to be in strickly increasing order and > 0.  once we're done
	apply the level change back to the result *)

	attributionTemp = {0 -> 0};
	(*	attribution = Select[final, TrueQ[#[[3]]]&][[All, {1,2}]]; *)

	Scan[If[MemberQ[attributionTemp[[All, 2]], #[[2]]],
		AppendTo[attributionTemp, #[[1]][[1]] -> #[[2]] + 1], AppendTo[attributionTemp, #[[1]][[1]] -> #[[2]]]]&, finalAttributionBlock];
	attributionTemp = Drop[attributionTemp,1];
	finalAttributionBlock = If[MemberQ[attributionTemp[[All, 1]], #[[1]][[1]]], ReplacePart[#, {2 -> (#[[1]][[1]] /. attributionTemp)}], #] & /@ finalAttributionBlock;

	(* now need to combine blocks back if they are blocks seperated by blank lines.  The test is to see if they are adjacent blocks
		with the same level *)
	finalMessageBlock = Split[finalMessageBlock, #1[[2]] == #2[[2]] &];
	finalMessageBlock = {Union[Flatten[#[[All, 1]]]], #[[1]][[2]], CreateUUID[],
		Missing["NotApplicable"], Missing["NotApplicable"]} & /@ finalMessageBlock;

	(* we do the pairing here. we set up a set of rules on how to do the reference, then use ReplacePart *)
	pairedBlockRule = MapIndexed[
		With[{currentBlockIndex = First@#2},
			(* we only care about every block except the first one, and if the block level is smaller than the
			previous block, then we do the pairing *)
			If[currentBlockIndex > 1 && #1[[2]] < finalMessageBlock[[currentBlockIndex - 1]][[2]],
				{{currentBlockIndex, 4} -> finalMessageBlock[[currentBlockIndex - 1]][[3]],
				{currentBlockIndex - 1, 5} -> #1[[3]]}
				,
				Unevaluated[Sequence[]]
			]
		]&
		,
		finalMessageBlock];

	pairedBlockRule = Flatten[pairedBlockRule, 1];

	{ReplacePart[finalMessageBlock, pairedBlockRule], finalAttributionBlock}
]

(* ::Subsection:: *)
(* SimpleDiffHashedLine *)
(* ------------------------------------------------------------------------- *)

SimpleDiffHashedLine[newMessage_List, oldMessage_List] := Module[
	{newhash, oldhash, pos, quotedIndex,  unQuotedIndex, temp = {0}},

	(* hash every line in both new and old message, remove the indicator at the beginning, and that'll trim off any space at the beginning and end as well. *)
	newhash = Composition[Hash, ReplyIndicatorTrim] /@ newMessage;
	oldhash = Composition[Hash, ReplyIndicatorTrim] /@ oldMessage;

	(* for each oldhash, collect the position in new hash where it was found *)
	pos = 1 - Unitize[newhash-#] & /@ oldhash;
	pos = Flatten[SparseArray[#]["NonzeroPositions"]] & /@ pos;

	(*some string from old hash will be found at many places in the newhash, for example, an empty line consist of
	 \n is going to show up at many place in the new hash.  The idea is, the old hash lines, if they
	are found in the new hash, it has to be an increasing sequence.  for example, suppose old has
	lines 5,7,8  found in the new hash, the corresponding new hash line must be an increasing
	sequence.  so pos={{pos. where old hash line 1 is found}, {pos where oldhash 2 is found}...}
	the idea is to pick at most 1 element from elements of pos st the elements picked
	form an increasing sequence.  we also want this to be the longest sequence possible.  so for
	each of the list in pos, we choose the miminum that'll satisfy the increasing sequence
	critirium, so that's when I use Min[Select...] *)

	Scan[Function[arg, AppendTo[temp, Min[Select[arg, # > Max[temp] &]]]], pos];

	(* I used {0} as a starting point for the temp. list that holds the increasing sequence. drop
	 the 0 now *)
	quotedIndex = Drop[temp, 1];
	unQuotedIndex = Complement[Range[Length[newhash]], quotedIndex];
	(* return type is the line index for the unQuoted and Quoted parts *)
	{unQuotedIndex,quotedIndex}
]

(* ::Subsection:: *)
(* MIMEBodyParse *)
(* ------------------------------------------------------------------------- *)

Options[MIMEBodyParse] =
{
	"MessageID" -> Automatic,
	"AuthorName" -> Automatic,
	"AuthorAddress" -> Automatic,
	"Author" -> Automatic,
	"Date" -> Automatic,
	"ReplyIndicator" -> Automatic
}

MIMEBodyParse[msgObj_?openMessageQ, args___, opts:OptionsPattern[]] := Module[{body, header, messageID, authorName, authorAddress, author, date},
	body = MIMEMessageGetElement[msgObj, "Body"];

	header = MIMEMessageRead[msgObj, "Header", {"MessageID", "FromName", "FromAddress", "From", "OriginatingDate"}];

	messageID = Replace[header["MessageID"], exc_MIMEToolsException -> Missing["NotAvailable"]];
	authorName = Replace[header["FromName"], exc_MIMEToolsException -> Missing["NotAvailable"]];
	authorAddress = Replace[header["FromAddress"], exc_MIMEToolsException -> Missing["NotAvailable"]];
	author = Replace[header["From"], exc_MIMEToolsException -> Missing["NotAvailable"]];
	date = Replace[header["OriginatingDate"], exc_MIMEToolsException -> Missing["NotAvailable"]];

	If[ Head@body === MIMEToolsException,
		Return[CreateMIMEToolsException[ 42, body]];
		,
		Return[ MIMEBodyParse[ body, args, "MessageID" -> messageID, "AuthorName" -> authorName, "AuthorAddress" -> authorAddress, "Author" -> author, "Date" -> date]];
	];
]

MIMEBodyParse[body_String, opts:OptionsPattern[]] := MIMEBodyParse[body, All, opts]
MIMEBodyParse[body_String, type : "NewBodyContent" | "QuotedContent" | "Attribution" | "PairedContent" | "AllQuotedContent", opts : OptionsPattern[]] := MIMEBodyParseInternal[body, 0, type, opts]
MIMEBodyParse[body_String, type : All, opts:OptionsPattern[]] := MIMEBodyParseInternal[body, {0,All}, type, opts]
MIMEBodyParse[body_String, level : _Integer|{_Integer, _Integer|All}, type : "NewBodyContent" | "QuotedContent" | "Attribution" | "PairedContent" | "AllQuotedContent", opts:OptionsPattern[]] := MIMEBodyParseInternal[body, level, type, opts]
MIMEBodyParse[_, args___] := CreateMIMEToolsException[45]


ValidSecondArgQ[x_] := (IntegerQ[x] || x === All)

Options[MIMEBodyParseInternal] =
{
	"MessageID" -> Automatic,
	"AuthorName" -> Automatic,
	"AuthorAddress" -> Automatic,
	"Author" -> Automatic,
	"Date" -> Automatic,
	"ReplyIndicator" -> Automatic
}

MIMEBodyParseInternal[body_String,
	levelSpec : _Integer | {_Integer, _?ValidSecondArgQ},
	type : "NewBodyContent" | "QuotedContent" | "PairedContent" | "AllQuotedContent" | "Attribution" | All,
	opts : OptionsPattern[]] := Module[
	{dataLines, messageBlock, attributionBlock, selectedBlock, quotedBlock, allQuotedBlock, low, high, outputElement},

	dataLines = If[StringQ[body], ImportString[body, "Lines"], body];
	{low, high} = If[IntegerQ[levelSpec], {levelSpec, levelSpec}, levelSpec];


	{messageBlock, attributionBlock}=MessageParsingHelper[dataLines];

	high = high /. {All -> Max[messageBlock[[All, 2]]]};

	selectedBlock = Select[messageBlock, MemberQ[Range[low, high], #[[2]]] &];
	allQuotedBlock = Select[messageBlock, #[[2]]>high&];

	(* this is the special case for top posting where the remainder of the text is considered the quoted block. *)

	quotedBlock =
	If[
		Length[selectedBlock] == 1 &&
		(Min[selectedBlock[[1]][[1]]] < Min[Flatten[Select[messageBlock, selectedBlock[[1]][[2]] < #[[2]]&][[All, 1]], 1]] ||
		Min[selectedBlock[[1]][[1]]] > Max[Flatten[Select[messageBlock, selectedBlock[[1]][[2]] < #[[2]]&][[All, 1]], 1]])
		,
		Flatten[Map[Function[arg,
		Select[messageBlock, arg[[2]] < #[[2]] &]], selectedBlock], 1]
		,
		Flatten[Map[Function[arg,
		Select[messageBlock, arg[[4]] == #[[3]] &]], selectedBlock], 1]
	];

	outputElement =
	If[type === "Attribution",
		CreateMailElement[StringJoin[Riffle[dataLines[[#[[1]]]], "\n"]], "Level" -> #[[2]], "UUID" -> #[[3]], "MessageID" -> OptionValue@"MessageID",
			"Type" -> "Attribution", "StartEndLineNumbers" -> {Min[#[[1]]], Max[#[[1]]]}, "AttributionLineReference" -> Missing["NotApplicable"]
		] & /@ attributionBlock
		,
		Map[
			Function[arg,
				Module[{tmp, assoc},
					tmp = Select[attributionBlock, arg[[2]] == #[[2]] &];
					assoc = If[Length[tmp] > 0, ExtractAttributionInfo[ReplyIndicatorTrim[dataLines[[tmp[[1]][[1]]]]]]];
					CreateMailElement[StringJoin[Riffle[ReplyIndicatorTrim[dataLines[[arg[[1]]]]], "\n"]],
						"Level" -> arg[[2]], "UUID" -> arg[[3]], "MessageID" -> OptionValue@"MessageID",
						"Type" -> "Content",
						"StartEndLineNumbers" -> {Min[arg[[1]]], Max[arg[[1]]]}, "InReplyToReference" -> arg[[4]],
						"AnswerReference" -> arg[[5]], "OriginalMailReference" -> Missing["NotAvailable"],
						"AuthorName" -> If[Length[tmp] > 0, assoc["AuthorName"], OptionValue@"AuthorName"],
						"AuthorAddress" -> If[Length[tmp] > 0, assoc["AuthorAddress"], OptionValue@"AuthorAddress"],
						"Author" -> If[Length[tmp] > 0, assoc["Author"], OptionValue@"Author"],
						"Date" -> If[Length[tmp] > 0, assoc["Date"], OptionValue@"Date"],
						"AttributionLineReference" -> If[Length[tmp] > 0, tmp[[1]][[3]], Missing["NotApplicable"]]
					]
				]
			]
			,
			Switch[type,
				"NewBodyContent",
					selectedBlock
				,
				"QuotedContent",
					quotedBlock
				,
				"AllQuotedContent",
					allQuotedBlock
				,
				"PairedContent",
					Sort[Union[selectedBlock, quotedBlock], #1[[1]][[1]] < #2[[1]][[1]] &]
				,
				All,
					messageBlock
			]
		]
	];

	If[type === All,
		With[{retVal = Union[outputElement,
			CreateMailElement[StringJoin[Riffle[dataLines[[#[[1]]]], "\n"]], "Level" -> #[[2]], "UUID" -> #[[3]],
				"MessageID" -> OptionValue@"MessageID", "Type" -> "Attribution",
				"StartEndLineNumbers" -> {Min[#[[1]]], Max[#[[1]]]},
				"AttributionLineReference" -> Missing["NotApplicable"]
			] & /@ attributionBlock]}
			,
			Sort[retVal, #1[[2]]["StartEndLineNumbers"][[1]] < #2[[2]]["StartEndLineNumbers"][[1]] &]]
		,
		outputElement
	]
]

$bodyElementProperties = {
	"Type", "Level", "MessageID", "AttributionLineReference", "AuthorName", "AuthorAddress", "Author", "Date",
	"OriginalMailReference", "StartEndLineNumbers",
	"UUID", "InReplyToReference", "AnswerReference"
}

bodyElementQ[ elem_] := MemberQ[ $bodyElementProperties, elem]

MailElement[ content_, prop_Association][ "Properties"] := prop
MailElement[ content_, prop_Association][ "Content"] := content
MailElement[ content_, prop_Association][] := Keys[ prop]
MailElement[ content_, prop_Association][ key_] := prop[key]

MailElement /: MakeBoxes[mElem : MailElement[ content_String, prop_Association], form : StandardForm | TraditionalForm] :=
Module[{ level, type},
	level = mElem[ "Level"];
	type = mElem[ "Type"];
	Switch[ type,
		"Content",
			ToBoxes[ Text[ Style[ content, ColorData[ 97, "ColorList"][[level + 1]] ]]],
		"Attribution",
			ToBoxes[ Text[ Style[ content, Italic, Gray]]],
		_,
			ToBoxes[ Text[ content]]
	]
]

MailElementLookup[elem_MailElement, propKey_String] := Lookup[elem["Properties"],propKey, Missing["NotAvailable"]]

(* ::Subsection:: *)
(* MIMEBodyParseThread *)
(* ------------------------------------------------------------------------- *)

MIMEBodyParse[{newMsg_?openMessageQ, oldMsg_?openMessageQ}, type: "NewBodyContent" | "QuotedContent" | "Attribution" | "PairedContent" | All, opts : OptionsPattern[]] :=
Module[{ newBody, oldBody},
	newBody = MIMEMessageGetElement[ newMsg, "Body"];
	oldBody = MIMEMessageGetElement[ oldMsg, "Body"];
	Return[ MIMEBodyParse[{newBody, oldBody}, type, opts]];
]

MIMEBodyParse[{newbody_String, oldbody_String}, type: "NewBodyContent" | "QuotedContent" | "Attribution" | "PairedContent" | All, opts : OptionsPattern[]] :=
Module[
	{oldDataLine, newDataLine, unquotedParagraph, quotedParagraph, pairedParagraph, attributionLines, data, attributionElement, pairedLine,
	unquotedElement, quotedElement, pairedElement},
	oldDataLine = If[StringQ[oldbody], ReplyIndicatorTrim[ImportString[oldbody, "Lines"]], oldbody];
	newDataLine = If[StringQ[newbody], ReplyIndicatorTrim[ImportString[newbody, "Lines"]], newbody];
	data = MapIndexed[{First[#2], #1} &, newDataLine];
	pairedLine = SimpleDiffHashedLine[newDataLine, oldDataLine];

	attributionLines = If[AttributionQ[#[[2]]], #[[1]], Unevaluated[Sequence[]]] & /@ data;

	unquotedParagraph = GroupIntoParagraph[Complement[pairedLine[[1]], attributionLines]];
	quotedParagraph = GroupIntoParagraph[pairedLine[[2]]];

	(* for every unquotedParagraph, find the corresponding quotedParagraph by checking if the first line of unquoted is a member of the quoted *)
	pairedParagraph = Map[Function[arg, If[MemberQ[#, Min[arg] - 1], {arg, #}, {arg, {}}] & /@ quotedParagraph], unquotedParagraph];
	pairedParagraph = Flatten[Gather[pairedParagraph, #1[[1]] == #2[[1]] &], 1];
	pairedParagraph = {Union[Flatten[#[[All, 1]]]], Union[Flatten[#[[All, 2]]]]} & /@ pairedParagraph;

	unquotedElement = CreateMailElement[StringJoin[Riffle[newDataLine[[#]], "\n"]], "Level" -> 0, "StartEndLineNumbers" -> {Min[#], Max[#]}, "Type" -> "Content"] & /@ unquotedParagraph;
	quotedElement = CreateMailElement[StringJoin[Riffle[newDataLine[[#]], "\n"]], "Level" -> 1, "StartEndLineNumbers" -> {Min[#], Max[#]}, "Type" -> "Content"] & /@ quotedParagraph;
	attributionElement = CreateMailElement[newDataLine[[#]], "Level" -> 1, "StartEndLineNumbers" -> {Min[#], Max[#]}, "Type" -> "Attribution"] & /@ attributionLines;

	pairedElement = {If[Length@#[[1]] > 0, CreateMailElement[StringJoin[Riffle[newDataLine[[#[[1]]]], "\n"]], "Level" -> 0, "StartEndLineNumbers" -> {Min[#[[1]]], Max[#[[1]]]}, "Type" -> "Content"], Unevaluated[Sequence[]]],
	 If[Length@#[[2]] > 0, CreateMailElement[StringJoin[Riffle[newDataLine[[#[[2]]]], "\n"]], "Level" -> 1, "StartEndLineNumbers" -> {Min[#[[2]]], Max[#[[2]]]}, "Type" -> "Content"], Unevaluated[Sequence[]]]} & /@ pairedParagraph;

	Switch[type,
		"NewBodyContent",
			unquotedElement
		,
		"QuotedContent",
			quotedElement
		,
		"Attribution",
			attributionElement
		,
		"PairedContent",
			Sort[Flatten[Union[pairedElement], 1], #1[[2]]["StartEndLineNumbers"][[1]] < #2[[2]]["StartEndLineNumbers"][[1]] &]
			,
		All,
			Sort[Union[unquotedElement, quotedElement, attributionElement], #1[[2]]["StartEndLineNumbers"][[1]] < #2[[2]]["StartEndLineNumbers"][[1]] &]
	]
]

End[]
EndPackage[]

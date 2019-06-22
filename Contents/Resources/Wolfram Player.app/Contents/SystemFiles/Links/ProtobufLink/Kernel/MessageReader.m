(*******************************************************************************

Message Reader 

*******************************************************************************)

Package["ProtobufLink`"]

PackageImport["GeneralUtilities`"]

(*----------------------------------------------------------------------------*)

messageToJSON = LibraryFunctionLoad[$ProtoLinkLib, "WL_MessageToJSON", 
	{
		Integer,
		True|False,
		True|False,
		True|False,
		True|False
	},
	"UTF8String"	
]

messageToExpression = LibraryFunctionLoad[$ProtoLinkLib, "WL_MessageToExpression", 
		LinkObject,
		LinkObject		
]

(*----------------------------------------------------------------------------*)
PackageExport["ProtobufImport"]

SetUsage[ProtobufImport,
"ProtobufImport[proto$, bin$, message$] imports a message message$ from a \
binary file bin$ from the given schema proto$.

The following options are available:

|'Paths' | None | A director or list of directories to use as a search path to resolve \
import dependencies in the protobuf definition proto$. |
"
]

Options[ProtobufImport] = {
	"Paths" -> None
};

ProtobufImport[proto_, bin_, msgName_String, opts:OptionsPattern[]] := CatchFailure @ Module[
	{paths, dsc, msg},
	paths = Replace[OptionValue["Paths"], None -> {}];

	dsc = ProtobufDescriptorCreate[proto, paths];
	If[FailureQ[dsc], Return[dsc]];

	msg = ProtobufDescriptorToMessage[dsc, msgName];
	If[FailureQ[msg], Return[msg]];
	ProtobufMessageSet[msg, File@fileConform[bin]];
	ProtobufMessageToExpression[msg]
]

(*----------------------------------------------------------------------------*)
PackageExport["ProtobufMessageToExpression"]

SetUsage[ProtobufMessageToExpression,
"ProtobufMessageToExpression[ProtobufMessage[$$]] converts a ProtobufMessage[$$] \
to a WL expression.
"
]

ProtobufMessageToExpression[msg_ProtobufMessage] := CatchFailure @ Module[
	{expr},
	expr = safeLibraryInvoke[messageToExpression, getMLEID[msg]];
	expr
]


(*----------------------------------------------------------------------------*)
PackageExport["ProtobufMessageToJSONExpression"]

SetUsage[ProtobufMessageToJSONExpression,
"ProtobufMessageToJSONExpression[ProtobufMessage[$$]] converts a ProtobufMessage[$$] \
to JSON, which is represented as a Wolfram Langauge expression via \
Developer`ReadRawJSONString. The following options for printing the JSON are available:

| 'AddWhitespace' | False | Whether to add spaces, line breaks and indentation \
to make the JSON output easy to read. |
| 'AlwaysPrintPrimitiveFields' | False | Whether to always print primitive fields. \
By default primitive fields with default values will be omitted in JSON joutput. For example, an int32 \
field set to 0 will be omitted. Set this flag to true will override the default \
behavior and print primitive fields regardless of their values. |
| 'AlwaysPrintEnumsAsInts' | False | Whether to always print enums as ints. By \
default they are rendered as strings. |
| 'PreserveProtoFieldNames' | True | Whether to preserve proto field names. By \
default protobuf will generate JSON field names using the json_name option, \
or lower camel case, in that order. Setting this flag will preserve the original \
field names. |
"
]


Options[ProtobufMessageToJSONExpression] = {
	"AddWhitespace" -> False,
	"AlwaysPrintPrimitiveFields" -> False,
	"AlwaysPrintEnumsAsInts" -> False,
	"PreserveProtoFieldNames" -> True
};

ProtobufMessageToJSONExpression[msg_ProtobufMessage, opts:OptionsPattern[]] := 
CatchFailure @ Module[
	{json},

	{white, primitive, enums, fields} = OptionValue[{
		"AddWhitespace", "AlwaysPrintPrimitiveFields", 
		"AlwaysPrintEnumsAsInts", "PreserveProtoFieldNames"
		}
	];
	json = safeLibraryInvoke[messageToJSON, 
		getMLEID[msg],
		white, 
		primitive, 
		enums,
		fields
	];
	(* return string will keep massive JSON alive unless cleared *)
	clearReturnString[];
	Developer`ReadRawJSONString[json]
]


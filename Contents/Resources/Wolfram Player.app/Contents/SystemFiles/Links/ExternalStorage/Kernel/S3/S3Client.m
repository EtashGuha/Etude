(*******************************************************************************

Client level functions

*******************************************************************************)

Package["ExternalStorage`"]

PackageImport["GeneralUtilities`"]

(*----------------------------------------------------------------------------*)
(****** Load Library Functions ******)

s3Declare[s3ClientHandleCreate, {Integer, Integer}]

(*----------------------------------------------------------------------------*)
PackageExport["S3Client"]

SetUsage[
"S3Client[id$] is an object that represents a S3 client connection."
]

(* This is a utility function defined in GeneralUtilities, which makes a nicely
formatted display box *)
DefineCustomBoxes[S3Client, 
	e:S3Client[mle_ /; ManagedLibraryExpressionQ[mle]] :> Block[{},
	BoxForm`ArrangeSummaryBox[
		S3Client, e, None,
		{BoxForm`SummaryItem[{"ID: ", getMLEID[mle]}]},
		{},
		StandardForm
	]
]];

getMLE[S3Client[mle_]] := mle;
getMLEID[S3Client[mle_ /; ManagedLibraryExpressionQ[mle]]] := 
	ManagedLibraryExpressionID[mle];

(*----------------------------------------------------------------------------*)
PackageExport["$DefaultS3Client"]
PackageScope["getDefaultS3Client"]

General::s3defclient = "No default client available. Run S3Connect to set the default S3 client.";

getDefaultS3Client[] := If[Head[$DefaultS3Client] === S3Client,
	$DefaultS3Client,
	ThrowFailure["s3defclient"]
]

PackageScope["parseS3Client"]

General::s3invclient = "Value for 'Client' must be either Automatic or an S3Client object, but `` was given."

parseS3Client[Automatic] := getDefaultS3Client[]
parseS3Client[x_S3Client] := x
parseS3Client[x___] := ThrowFailure["s3invclient", x]

(*----------------------------------------------------------------------------*)
PackageExport["S3Connect"]

SetUsage[
"S3Connect[] connects to S3 authenticating via the default credential provider chain. Returns \
a S3Client[$$] object and sets $DefaultS3Client to S3Client[].

The following options are available:
|'ThreadNumber' | Automatic | Number of threads for thread pool. If Automatic, use $ProcessorCount. |
"
]

Options[S3Connect] = {
	"ThreadNumber" -> Automatic
}

DeclareArgumentCount[S3Connect, 0];
S3Connect[opts:OptionsPattern[]] := CatchFailureAndMessage @ Module[
	{
		clientHandle,
		threadNumber = OptionValue["ThreadNumber"]
	},
	threadNumber = Replace[threadNumber, Automatic -> $ProcessorCount];

	clientHandle = CreateManagedLibraryExpression["S3client", clientMLE];
	(* return client object *)
	s3Call[s3ClientHandleCreate, getMLEID[clientHandle], threadNumber];
	$DefaultS3Client = System`Private`SetNoEntry @ S3Client[clientHandle];
	$DefaultS3Client
]


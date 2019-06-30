(*******************************************************************************

Bucket level functions

*******************************************************************************)

Package["ExternalStorage`"]

PackageImport["GeneralUtilities`"]

(*----------------------------------------------------------------------------*)
(****** Load Library Functions ******)

s3Declare[s3GetObjectNames, LinkObject, LinkObject]
s3Declare[s3DeleteObjects, LinkObject, LinkObject]
s3Declare[s3CopyObject, {Integer, "UTF8String", "UTF8String", "UTF8String", "UTF8String"}]

(*----------------------------------------------------------------------------*)

PackageExport["S3Keys"]

SetUsage[
"S3Keys[bucket$] returns a list of object names in the bucket bucket$.

The following options are available:
|'Client' | Automatic | If Automatic, use the $DefaultS3Client. Otherwise, use supplied S3Client.|
| 'Delimiter' | None | emulates a virtual file hierachy and returns its content \
at root level in the bucket considering keys as full path and using \"/\" as the path delimiter|
| MaxItems | 1000 | maximum keys to return. |
| 'Prefix' | None | returns only the objects which keys starts by prefix.|
"
]

Options[S3Keys] = {
	"Client" -> Automatic,
	MaxItems -> 1000,
	"Prefix" -> None,
	"Delimiter" -> None
}

DeclareArgumentCount[S3Keys, 1];
S3Keys[bucket_String, opts:OptionsPattern[]] := CatchFailureAndMessage @ Scope[
	UnpackOptions[client, maxItems, prefix, delimiter];
	client = parseS3Client[client];

	If[!IntegerQ[maxItems] && (maxItems =!= Infinity), ThrowFailure["invmax1", maxItems]];

	keysRes = {};
	contChar = None;
	remainingKeys = maxItems;
	moreQ = True;

	Join @@ First[Last[Reap[
		While[moreQ,
			num = Min[1000, remainingKeys];
			{moreQ, contChar, keys} = iS3Keys[client, bucket, num, prefix, delimiter, contChar];
			Sow[keys, "keys"];
			remainingKeys -= num;
			If[remainingKeys === 0, moreQ = False];
		],
		"keys"
	]], {}]
]

iS3Keys[client_, bucket_, maxItems_, prefix_, delimiter_, continueChar_] := Scope[
	delimiterQ = (delimiter =!= None);
	delimiter2 = Replace[delimiter, None -> ""];

	prefixQ = (prefix =!= None);
	prefix2 = Replace[prefix, None -> ""];

	contQ = (continueChar =!= None);
	cont2 = Replace[continueChar, None -> ""];

	s3Call[s3GetObjectNames, 
		getMLEID[client], 
		maxItems,
		bucket,
		delimiterQ, delimiter2,
		prefixQ, prefix2,
		contQ, cont2
	]
]

S3Keys::invmax1 = "MaxItems was expected to be a positive integer or Infinity, but got ``."
S3Keys::invarg = "Expected a string, but got ``."
S3Keys[x_] := CatchFailureAndMessage @ ThrowFailure["invarg", x]

(*----------------------------------------------------------------------------*)
PackageExport["S3ObjectCopy"]

SetUsage[
"S3ObjectCopy[bucket$src -> bucket$dest, key$] copies the object key$ \
form bucket$src to bucket$dest.
S3ObjectCopy[bucket$src -> bucket$dest, key$src -> key$dest] copies the object key$src \
form bucket$src to the object key$dest in bucket$dest. 

The following options are available:
|'Client' | Automatic | If Automatic, use the $DefaultS3Client. Otherwise, use supplied S3Client.|
"
]

Options[S3ObjectCopy] = {
	"Client" -> Automatic
}

DeclareArgumentCount[S3ObjectCopy, 2];
S3ObjectCopy::invarg1 = "Expected a string or rule for the second argument, but got ``.";
S3ObjectCopy[Rule[buckSrc_String, buckTar_String], keys_, opts:OptionsPattern[]] := CatchFailureAndMessage @ Module[
	{
		client = OptionValue["Client"],
		srcKey, targKey
	},
	client = parseS3Client[client];
	Which[
		StringQ[keys],
			{srcKey, targKey} = {keys, keys},
		MatchQ[keys, Rule[_String, _String]],
			srcKey = First[keys];
			targKey = Last[keys],
		True,
			ThrowFailure["invarg1"]
	];
	s3Call[s3CopyObject, getMLEID[client], buckSrc, buckTar, srcKey, targKey];
	S3Success[S3ObjectCopy]
]

S3ObjectCopy::invarg2 = "Expected a rule string1 -> string2 for the first argument, but got ``."
S3ObjectCopy[x_, _] := CatchFailureAndMessage @ ThrowFailure["invarg2", x]

(*----------------------------------------------------------------------------*)
PackageExport["S3ObjectDelete"]

SetUsage[
"S3ObjectDelete[bucket$, key$] deletes the object key$ in bucket bucket$.
S3ObjectDelete[bucket$, {key$1, $$, key$n}] deletes the objects key$i in bucket bucket$.

The following options are available:
|'Client' | Automatic | If Automatic, use the $DefaultS3Client. Otherwise, use supplied S3Client.|
"
]

Options[S3ObjectDelete] = {
	"Client" -> Automatic
}

DeclareArgumentCount[S3ObjectDelete, 2];
S3ObjectDelete::invarg1 = "Expected a string or list of strings for the key names, but got ``.";
S3ObjectDelete[bucket_String, keys_, opts:OptionsPattern[]] := CatchFailureAndMessage @ Module[
	{
		client = OptionValue["Client"],
		keys2
	},
	client = parseS3Client[client];
	Which[
		StringQ[keys],
			keys2 = {keys},
		VectorQ[keys, StringQ],
			keys2 = keys,
		True,
			ThrowFailure["invarg1", keys]
	];
	s3Call[s3DeleteObjects, getMLEID[client], keys2, bucket];
	S3Success[S3ObjectDelete]
]

S3ObjectDelete::invarg2 = "Expected a string as the bucket name, but got ``."
S3ObjectDelete[x_, _] := CatchFailureAndMessage @ ThrowFailure["invarg2", x]

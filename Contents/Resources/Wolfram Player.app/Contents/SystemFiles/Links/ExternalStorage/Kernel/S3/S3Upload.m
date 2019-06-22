(*******************************************************************************

Upload level functions

*******************************************************************************)

Package["ExternalStorage`"]

PackageImport["GeneralUtilities`"]

(*----------------------------------------------------------------------------*)
PackageExport["S3UploadSubmit"]

Options[S3UploadSubmit] = {
	"Client" -> Automatic
}

DeclareArgumentCount[S3DownloadSubmit, {2, 3}];
S3UploadSubmit[bucket_, files_, keys_:None, opts:OptionsPattern[]] := CatchFailureAndMessage @ Module[
	{
		client = parseS3Client[OptionValue["Client"]]
	},
	iS3UploadSubmit[bucket, files, keys, client]
]

General::s3invupsubspec1 = "The bucket name should be a String, but had head of ``."
General::s3invupsubspec2 = "The number of keys is `` whilst the number of files is ``."
General::s3invupsubspec3 = "The keys are not all strings."

iS3UploadSubmit[bucket_, files_, keys_, client_] := Module[
	{
		files2,
		keys2
	},

	If[!StringQ[bucket], 
		ThrowFailure["s3invupsubspec1", Head[bucket]]
	];

	(* fileConform checks that file exists etc *)
	files2 = fileConform /@ If[!ListQ[files], {files}, files];

	keys2 = If[keys === None, 
		FileNameTake /@ files2,
		If[!ListQ[keys], {keys}, keys]
	];

	If[Length[keys2] =!= Length[files2], 
		ThrowFailure["s3invupsubspec2", Length[keys2], Length[files2]]
	];
	If[!VectorQ[keys2, StringQ], 
		ThrowFailure["s3invupsubspec3"]
	];
	createTask[client, bucket, keys2, files2, False]
]

(*----------------------------------------------------------------------------*)
PackageExport["S3Upload"]

SetUsage[
"S3Upload[bucket$, file$] uploads the content of the file$ into the bucket$ using a key name FileNameTake[file$]. 
S3Upload[bucket$, {file$1, $$, file$n}] uploads the files file$i into the bucket$ using key names FileNameTake[file$i].
S3Upload[bucket$, file$, key$] uploads file$ into bucket$ with keyname key$.
S3Upload[bucket$, {file$1, $$, file$n}, {key$1, $$, key$n}] uploads files file$i into bucket$ with keynames key$i.

S3Upload returns either string keynames or Failure[$$] objects.

The following options are available:
|'Client' | Automatic | If Automatic, use the $DefaultS3Client. Otherwise, use supplied S3Client.|
"
]

Options[S3Upload] = {
	"Client" -> Automatic
}

DeclareArgumentCount[S3Upload, {2, 3}];
S3Upload[bucket_, files_, keys_:None, opts:OptionsPattern[]] := CatchFailureAndMessage @ Module[
	{
		client = parseS3Client[OptionValue["Client"]],
		filesOut, keys2
	},
	task = iS3UploadSubmit[bucket, files, keys, client];
	CheckAbort[
		iS3TaskWait[task];

		keys2 = task["EvaluationResult"];
		S3TaskRemove[task];
		If[!ListQ[files],
			First[keys2],
			keys2
		]
		,
		S3TaskRemove[task];
		$Aborted
	]
]

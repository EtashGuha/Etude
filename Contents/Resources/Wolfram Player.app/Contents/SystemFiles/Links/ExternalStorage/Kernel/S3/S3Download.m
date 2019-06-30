(*******************************************************************************

Download level functions

*******************************************************************************)

Package["ExternalStorage`"]

PackageImport["GeneralUtilities`"]

(*----------------------------------------------------------------------------*)
PackageExport["S3DownloadSubmit"]

Options[S3DownloadSubmit] = {
	"Client" -> Automatic,
	OverwriteTarget -> False
}

DeclareArgumentCount[S3DownloadSubmit, {2, 3}];
S3DownloadSubmit[bucket_, keys_, opts:OptionsPattern[]] := S3DownloadSubmit[bucket, keys, Automatic, opts]
S3DownloadSubmit[bucket_, keys_, files_, opts:OptionsPattern[]] := CatchFailureAndMessage @ Module[
	{
		client = parseS3Client[OptionValue["Client"]],
		overwrite = OptionValue[OverwriteTarget]
	},
	iS3DownloadSubmit[bucket, keys, files, client, overwrite]
]

General::s3invdownsubspec1 = "The bucket name should be a String, but had head of ``."
General::s3invdownsubspec2 = "The files must be a list of file specifications a valid directory path or Automatic."
General::s3invdownsubspec3 = "The keys can only be strings."
General::s3invdownsubspec4 = "The list of keys is of length `` and the list of files is of length ``."
General::s3copyfile = "Cannot overwrite existing file ``. Use option OverwriteTarget to control this behaviour."


iS3DownloadSubmit[bucket_, keys_, fileSpec_, client_, overwrite_] :=
Module[
	{
		files
	},

	(* Validate bucket *)
	If[!StringQ[bucket], 
		ThrowFailure["s3invdownsubspec1", Head[bucket]]
	];

	(* Validate keys *)
	If[!StringQ[keys] && !VectorQ[keys, StringQ],
		ThrowFailure["s3invdownsubspec3"]
	];

	(* Validate files *)
	If[
		MatchQ[fileSpec, None|Automatic]
		,
		files = CreateDirectory[];
		If[!Quiet[DirectoryQ[files]],
			ThrowFailure["interr", "error creating temporary directroy"]
		]
		,
		files = Replace[fileSpec, File[s_String] :> s, Infinity]
	];
	Which[
		(* {remotes} -> dir *)
		VectorQ[keys] && StringQ[files],
		files = verifyDirectory[files, overwrite];
		files = FileNameJoin[{files, #}]& /@ keys;
		,
		(* remote -> local *)
		StringQ[keys] && StringQ[files],
		files = {files};
		,
		(* {remotes} -> {locals} *)
		VectorQ[keys] && VectorQ[files, StringQ],
		If[Length[keys] =!= Length[files], 
			ThrowFailure["s3invdownsubspec4", Length[keys], Length[files]]
		];
		,
		(* bad file specification *)
		True,
		ThrowFailure["s3invdownsubspec2"]
	]

	(* check whether files already exist *)
	checkFileExists[files, overwrite];

	(* task evaluate *)
	createTask[client, bucket, Developer`ToList[keys], files, True]
]

checkFileExists[file_String, overwrite_] := checkFileExists[{file}, overwrite]
checkFileExists[files_List, overwrite_] := 
Module[
	{conflict, directories},

	(* check for existing files with the same name *)
	conflict = SelectFirst[files, FileExistsQ, None];
	If[
		!TrueQ[overwrite] && conflict =!= None,
		ThrowFailure["s3copyfile", conflict]
	];

	(* not using DirectoryName to avoid the trailing delimiter *)
	directories = DeleteCases[DeleteDuplicates[FileNameTake[#, {1, -2}]& /@ files], ""];
	(* validate/create intermediate directories *)
	Scan[verifyDirectory[#, overwrite]&, directories];

]

verifyDirectory[name_, overwrite_] :=
Which[
	DirectoryQ[name],
	ExpandFileName[name],

	And[
		FileExistsQ[name],
		!DirectoryQ[name],
		TrueQ[overwrite]
	],
	DeleteFile[name];
	CreateDirectory[name],

	!FileExistsQ[name],
	CreateDirectory[name],

	True,
	ThrowFailure["s3copyfile", name]
]

(*----------------------------------------------------------------------------*)
(* Dev Note: a major feature missing for S3Download is a progress bar *)

PackageExport["S3Download"]

SetUsage[
"S3Download[bucket$, key$] downloads the content of the object key$ in bucket$ to a local temporary file. 
S3Download[bucket$, {key$1, $$, key$n}] downloads the objects key$i to local temporary files 
S3Download[bucket$, key$, file$] downloads the object key$ to file$, where file$ is either a \
string path or File[$$] object.
S3Download[bucket$, key$, dir$] downloads the object key$ to directory dir$
S3Download[bucket$, {key$1, $$, key$n}, {file$1, $$, file$n}]  downloads keys key$i to files file$i.
S3Download[bucket$, {key$1, $$, key$n}, dir$]  downloads keys key$i to directory dir$.

S3Download returns either File[$$] objects or Failure[$$] objects.

The following options are available:
|'Client' | Automatic | If Automatic, use the $DefaultS3Client. Otherwise, use supplied S3Client.|
|OverwriteTarget | False | whether to overwrite if the target file already exists. |
"
]

Options[S3Download] = {
	"Client" -> Automatic,
	OverwriteTarget -> False
}

DeclareArgumentCount[S3Download, {2, 3}];
S3Download[bucket_, keys_, opts:OptionsPattern[]] := S3Download[bucket, keys, None, opts]
S3Download[bucket_, keys_, files_, opts:OptionsPattern[]] := CatchFailureAndMessage @ Module[
	{
		client = parseS3Client[OptionValue["Client"]],
		overwrite = OptionValue[OverwriteTarget],
		filesOut, keys2
	},
	task = iS3DownloadSubmit[bucket, keys, files, client, overwrite];

	CheckAbort[
		iS3TaskWait[task];

		filesOut = task["EvaluationResult"];
		S3TaskRemove[task];
		If[!ListQ[keys],
			First[filesOut],
			filesOut
		]
		,
		S3TaskRemove[task];
		$Aborted
	]
]

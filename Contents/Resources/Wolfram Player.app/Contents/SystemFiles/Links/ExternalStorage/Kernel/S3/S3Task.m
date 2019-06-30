(*******************************************************************************

Transfer manager level functions

*******************************************************************************)
Package["ExternalStorage`"]

PackageImport["GeneralUtilities`"]

(*----------------------------------------------------------------------------*)
(****** Load Library Functions ******)

s3Declare[s3CreateTransfer, {Integer, Integer}]
s3Declare[s3CleanTransfer, {Integer}]

s3Declare[s3TransferGroupAddUploadDownloadFile, 
	{Integer, "UTF8String", "UTF8String", "UTF8String", True|False}
]
s3Declare[s3TransferGroupBytesTransferred, {Integer}, Integer]
s3Declare[s3TransferGroupBytesTotalSize, {Integer}, Integer]
s3Declare[s3TransferGroupFinishedCount, {Integer}, Integer]
s3Declare[s3TransferGroupGetSize, {Integer}, Integer]

s3Declare[s3TransferGroupSuccessQ, {Integer}, True|False]
s3Declare[s3TransferGroupTaskStatus, {Integer}, True|False]

s3Declare[s3TransferGroupCancel, {Integer, True|False}]
s3Declare[s3TransferGroupWait, {Integer}]
s3Declare[s3TransferGroupFileKeyStatus, LinkObject, LinkObject]

(*----------------------------------------------------------------------------*)
PackageScope["$S3TaskCount"]
If[!IntegerQ[$S3TaskCount], $S3TaskCount = 1];

PackageScope["$S3Tasks"]
If[!AssociationQ[$S3Tasks], $S3Tasks = <||>];

PackageExport["S3Tasks"]
SetUsage[
"S3Tasks[] gives a list of S3TaskObject expressions that represent currently submitted tasks."
]

S3Tasks[] := Values[$S3Tasks]

General::s3taskrmv = "Task `` is removed." 
removedTaskQ[id_] := If[MissingQ[$S3Tasks[id]], True, False]
checkedTaskID[id_] := If[removedTaskQ[id], ThrowFailure["s3taskrmv", id], id]

(*----------------------------------------------------------------------------*)
PackageExport["S3TaskObject"]

DefineCustomBoxes[S3TaskObject, 
	e:S3TaskObject[taskID_Integer, client_S3Client, direction_String] :> Block[{},
	BoxForm`ArrangeSummaryBox[
		S3TaskObject, e, None,
		{
			BoxForm`SummaryItem[{"ID: ", taskID}],
			BoxForm`SummaryItem[{"Client ID: ", getMLEID[client]}],
			BoxForm`SummaryItem[{"Type: ", direction}]
		},
		{},
		StandardForm
	]
]];

$TaskProperties = <||>;
$TaskProperties["BytesTotalSize"] := 
	s3Call[s3TransferGroupBytesTotalSize, checkedTaskID[#id]]&

$TaskProperties["BytesTransferred"] := 
	s3Call[s3TransferGroupBytesTransferred, checkedTaskID[#id]]&

$TaskProperties["CompletedCount"] := 
	s3Call[s3TransferGroupFinishedCount, checkedTaskID[#id]]&

$TaskProperties["TotalCount"] := 
	s3Call[s3TransferGroupGetSize, checkedTaskID[#id]]&

$TaskProperties["ProgressFraction"] = progressFraction[#id]&;

$TaskProperties["EvaluationResult"] = 
	s3Call[s3TransferGroupFileKeyStatus, checkedTaskID[#id], #downloadQ, True]&;
	
$TaskProperties["Keys"] = 
	s3Call[s3TransferGroupFileKeyStatus, checkedTaskID[#id], False, False]&;

$TaskProperties["Files"] = 
	s3Call[s3TransferGroupFileKeyStatus, checkedTaskID[#id], True, False]&;

$TaskProperties["MultiPartQ"] = s3Call[s3TransferGroupIsMultiPart, checkedTaskID[#id]]&;

$TaskProperties["TaskStatus"] = Which[
	removedTaskQ[#id], 
		"Removed",
	s3Call[s3TransferGroupTaskStatus, #id],
		"Running",
	True,
		"Finished"
]&

$TaskProperties["SuccessQ"] = s3Call[s3TransferGroupSuccessQ, checkedTaskID[#id]]&;

S3TaskObject::invprop = "The property `` does not exist."

S3TaskObject[id_, __]["Properties"] := Keys[$TaskProperties]

S3TaskObject[taskID_, _, dir_][prop_] := CatchFailureAndMessage @ Module[
	{f, downloadQ},
	f = Lookup[$TaskProperties, prop, ThrowFailure["invprop", prop]];
	downloadQ = (dir === "Download");
	f[<|"id" -> taskID, "downloadQ" -> downloadQ|>]
]

(*-----*)
(* utils *)
progressFraction[id_] := CatchFailureAndMessage @ Module[
	{total, finished},
	checkedTaskID[id];
	total = s3Call[s3TransferGroupGetSize, id];
	finished = s3Call[s3TransferGroupFinishedCount, id];
	N[finished / total]
]

(*----------------------------------------------------------------------------*)
PackageExport["S3TaskWait"]

SetUsage[
"S3TaskWait[S3TaskObject[$$]] blocks until all tasks in S3TaskObject[$$] are completed.
"
]

DeclareArgumentCount[S3TaskWait, 1];
S3TaskWait[task_S3TaskObject] := 
	CatchFailureAndMessage @ iS3TaskWait[task]

PackageScope["iS3TaskWait"]
iS3TaskWait[S3TaskObject[id_, __]] := s3Call[s3TransferGroupWait, checkedTaskID[id]]

(*----------------------------------------------------------------------------*)
PackageExport["S3TaskRemove"]

SetUsage[
"S3TaskRemove[S3TaskObject[$$]] cancels all tasks in S3TaskObject[$$] and removes all internal \
task objects. Will block until all subtasks are cancelled.
"
]

DeclareArgumentCount[S3TaskRemove, 1];

SetAttributes[S3TaskRemove, Listable];
S3TaskRemove[S3TaskObject[id_, __]] := CatchFailureAndMessage[
	checkedTaskID[id];
	s3Call[s3TransferGroupCancel, id, False];
	s3Call[s3TransferGroupWait, id];
	s3Call[s3CleanTransfer, id];
	KeyDropFrom[$S3Tasks, id];
]

(*----------------------------------------------------------------------------*)
PackageExport["S3TaskAbort"]

SetUsage[
"S3TaskAbort[S3TaskObject[$$]] sends an abort signal to cancel all tasks in S3TaskObject[$$]. \
Returns immediately. 
S3TaskAbort[S3TaskObject[$$], blocking$] blocks until all tasks are cancelled if blocking$ \
is True, otherwise returns immediately.
"
]

DeclareArgumentCount[S3TaskAbort, {1, 2}];
S3TaskAbort[S3TaskObject[id_, __], blocking_:False] := CatchFailureAndMessage[
	s3Call[s3TransferGroupCancel, checkedTaskID[id], False];
	If[blocking,
		s3Call[s3TransferGroupWait, id];
	]
]

(*----------------------------------------------------------------------------*)
PackageScope["createTask"]
createTask[client_S3Client, bucket_, keys_, files_, downloadQ_] := Module[
	{id = $S3TaskCount, type},
	(* create transfer manager *)
	s3Call[s3CreateTransfer, getMLEID[client], id];

	MapThread[
		s3Call[s3TransferGroupAddUploadDownloadFile, 
			id,
			bucket,
			#1,
			#2,
			downloadQ
		]&,
		{keys, files}
	];
	type = If[downloadQ, "Download", "Upload"];
	(* add to task registry *)
	$S3TaskCount += 1;
	$S3Tasks[id] = S3TaskObject[id, client, type];
	$S3Tasks[id]
]


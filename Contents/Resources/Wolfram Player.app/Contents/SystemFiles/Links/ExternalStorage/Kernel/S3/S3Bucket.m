(*******************************************************************************

Bucket level functions

*******************************************************************************)

Package["ExternalStorage`"]

PackageImport["GeneralUtilities`"]

(*----------------------------------------------------------------------------*)
(****** Load Library Functions ******)

s3Declare[s3CreateBucket, {Integer, "UTF8String"}]
s3Declare[s3DeleteBucket, {Integer, "UTF8String"}]
s3Declare[s3GetBucketNames, LinkObject, LinkObject]

(*----------------------------------------------------------------------------*)
PackageExport["S3Buckets"]

SetUsage[
"S3Buckets[] returns a list of bucket names.

The following options are available:
|'Client' | Automatic | If Automatic, use the $DefaultS3Client. Otherwise, use supplied S3Client.|
"
]

Options[S3Buckets] = {
	"Client" -> Automatic
}

DeclareArgumentCount[S3Buckets, 0];
S3Buckets[opts:OptionsPattern[]] := CatchFailureAndMessage @ Module[
	{client = OptionValue["Client"]},
	client = parseS3Client[client];
	s3Call[s3GetBucketNames, getMLEID[client]]
]

(*----------------------------------------------------------------------------*)
PackageExport["S3BucketCreate"]

SetUsage[
"S3BucketCreate[bucket$] creates the specified S3 bucket bucket$.

The following options are available:
|'Client' | Automatic | If Automatic, use the $DefaultS3Client. Otherwise, use supplied S3Client.|
"
]

Options[S3BucketCreate] = {
	"Client" -> Automatic
}

DeclareArgumentCount[S3BucketCreate, 1];
S3BucketCreate[bucket_String, opts:OptionsPattern[]] := CatchFailureAndMessage @ Module[
	{client = OptionValue["Client"]},
	client = parseS3Client[client];
	s3Call[s3CreateBucket, getMLEID[client], bucket];
	S3Success[S3BucketCreate]
]

(*----------------------------------------------------------------------------*)
PackageExport["S3BucketDelete"]

SetUsage[
"S3BucketDelete[bucket$] deletes the specified S3 bucket bucket$.

The following options are available:
|'Client' | Automatic | If Automatic, use the $DefaultS3Client. Otherwise, use supplied S3Client.|
"
]

Options[S3BucketDelete] = {
	"Client" -> Automatic
}

DeclareArgumentCount[S3BucketDelete, 1];
S3BucketDelete[bucket_, opts:OptionsPattern[]] := CatchFailureAndMessage @ Module[
	{client = OptionValue["Client"]},
	client = parseS3Client[client];
	s3Call[s3DeleteBucket, getMLEID[client], bucket];
	S3Success[S3BucketDelete]
]

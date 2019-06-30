Package["ExternalStorage`"]

PackageExport[IPFSUpload]

IPFSUpload::nobj="`1` is not a valid ExternalServiceObject."
IPFSUpload::file="The specified argument `1` should be a valid string or File."
IPFSUpload::wbase="The specified argument `1` should be a valid ExternalStorage or Automatic."
IPFSUpload::bprms="`1` should be an Association with parameters valid for base `2`."

PackageScope[$IPFSAddress]

$IPFSAddress = "http://10.10.142.28:5001";

Options[IPFSUpload]= {"FileHash"->False}

IPFSUpload[file_, opts:OptionsPattern[]]:= (Message[IPFSUpload::file, file];$Failed) /; Quiet[FailureQ[FindFile[file]]]

IPFSUpload[file_, opts:OptionsPattern[]]:= ipfsUpload[file, opts]

IPFSUpload[args___, OptionsPattern[]] := (ArgumentCountQ[IPFSUpload, Length[{args}], 1, 3]; Null /; False)

ipfsUpload[file0_, OptionsPattern[IPFSUpload]]:= 
	Module[{file,request, addresshash = 0, filehash, filename},
		file = FindFile[file0];
		filename = FileNameTake[file];
		request = HTTPRequest[$IPFSAddress <> "/api/v0/add", <|"Body" -> {"file" -> File[file]}, Method -> "POST"|>];
		addresshash = ImportByteArray[URLRead[request, "BodyByteArray"], "RawJSON"]["Hash"];
		If[ TrueQ[OptionValue["FileHash"]],
			filehash = FileHash[file, "MD5", All, "HexString"];
			IPFSObject[<|"FileName"->filename,"Address"->addresshash,"FileHash"->filehash|>],
			IPFSObject[<|"FileName"->filename,"Address"->addresshash|>]
		]
]
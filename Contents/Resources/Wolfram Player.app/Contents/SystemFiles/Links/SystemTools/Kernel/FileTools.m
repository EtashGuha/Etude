(* ::Package:: *)

Begin["SystemTools`"];

FilePartition;
FileJoin;

Begin["Private`"];

Options[FilePartition] = {"TargetDirectory"->Automatic}

FilePartition[file_String, opts:OptionsPattern[]]/;FileExistsQ[file] := Block[{size},
	size = Round[UnitConvert[FileSize[file], "Bytes"]/5];
	FilePartition[file, size]
]

FilePartition[file_String, size_, opts:OptionsPattern[]] := Module[
	{n, dir, filesize, count, in, uuid, partdir, out, bytes, done},

	count = If[IntegerQ[size], size, QuantityMagnitude[UnitConvert[size, "Bytes"]]];
	dir = DirectoryName[file];
	in = OpenRead[file, BinaryFormat -> True];
	uuid = CreateUUID[];
	partdir = OptionValue["TargetDirectory"];
	Which[
		MatchQ[partdir, _String|File[_String,___]],
			If[!DirectoryQ[partdir],
				partdir = CreateDirectory[partdir]
			],
		True,
			partdir = CreateDirectory[]
	];
	If[!StringQ[partdir], Return[$Failed]];
	i = 0;
	While[Length[bytes = BinaryReadList[in, "Byte", count]] > 0,
		out = OpenWrite[
			FileNameJoin[{partdir,
				"part-" <> IntegerString[(i++), 10, 16] <> ".data"}],
			BinaryFormat -> True
		];
		BinaryWrite[out, bytes, "Byte"];
		Close[out];
	];
	Close[in];
	partdir
]

absolutePathQ[path_] := ExpandFileName[path] === FileNameJoin@FileNameSplit[path];

FileJoin[ parts_List, file_String, opts:OptionsPattern[] ] :=  Module[{out, in, bytes},
	out = OpenWrite[file, BinaryFormat -> True];
	If[FailureQ[out], Return[]];
	Map[
		Function[{f},
			in = OpenRead[f, BinaryFormat -> True];
			bytes = BinaryReadList[in, "Byte"];
			BinaryWrite[out, bytes, "Byte"];
			Close[in];
		]
		,parts
	];
	Close[out];
	file
]



FileJoin[partdir_String, name_String, OptionsPattern[] ] := Module[{parts, file},
	parts = Sort[FileNames["part-*.data", partdir]];
	If[ parts === {},
		parts = Sort[FileNames["*", partdir]]
	];
	If[absolutePathQ[name],
		file = name,
		file = FileNameJoin[{partdir, name}];
	];
	FileJoin[parts,file]
]

End[];
End[];
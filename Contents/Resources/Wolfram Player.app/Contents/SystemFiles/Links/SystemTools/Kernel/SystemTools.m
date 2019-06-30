(* ::Package:: *)

Block[{SystemTools`Private`dir = DirectoryName[$InputFileName]},
	Get[FileNameJoin[{SystemTools`Private`dir, "SystemMemory.m"}]];
	Get[FileNameJoin[{SystemTools`Private`dir, "SysInfo.m"}]];
	Get[FileNameJoin[{SystemTools`Private`dir, "NetworkInfo.m"}]];
	Get[FileNameJoin[{SystemTools`Private`dir, "FileTools.m"}]];
]

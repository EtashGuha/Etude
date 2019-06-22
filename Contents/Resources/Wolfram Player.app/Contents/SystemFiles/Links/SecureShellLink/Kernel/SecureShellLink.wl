(* ::Package:: *)

Block[{SecureShellLink`Private`dir = DirectoryName[$InputFileName]},
	Get[FileNameJoin[{SecureShellLink`Private`dir, "Remote.wl"}]];
]

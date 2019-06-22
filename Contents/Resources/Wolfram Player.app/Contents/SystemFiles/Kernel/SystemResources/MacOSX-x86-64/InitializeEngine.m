(*
  InitializeEngine
*)

(** user-level initialization: init files and persistent initialization **)

Begin["System`Private`"]

`$initFiles = {}; (* collect evidence *)

System`Private`GetFileInsist[ file_ ] := (
	AppendTo[$initFiles, file];
	Catch[Catch[ Get[file] ], _];
)

(* read a file, if it exists *)

System`Private`GetFile[ file_ ] := If[ FileType[file]===File, GetFileInsist[file] ]

(* read an "init.m" file in a given directory, if it exists *)

System`Private`GetInitFile[ dir_ ] := GetFile[ FileNameJoin[{dir, "init.m"}] ]

End[]

(* enable minimal FE functionality for init.m files; same as in sysinit.m *)

If[MathLink`NotebookFrontEndLinkQ[$ParentLink],
    System`Private`origFrontEnd = MathLink`SetFrontEnd[$ParentLink],
(* else *)
    System`Private`origFrontEnd = Null
]

(* Load code from startup-loaded paclets *)

PacletManager`Package`loadStartupPaclets[]

(* Set the context to Global` so that user variables are created there *)

Begin[ "Global`"];

(*
 Load init.m from $InstallationDirectory/Configuration/Kernel.
*)

System`Private`GetInitFile[ FileNameJoin[{$InstallationDirectory, "Configuration", "Kernel"}]];
	
(*
 Load init.m files from installation Autoload directories, "Kernel" subdirectory only
*)

Scan[
	System`Private`GetInitFile[FileNameJoin[{#, "Kernel"}]]&,
	Select[
		FileNames["*", FileNameJoin[{$InstallationDirectory, "AddOns", "Autoload"}]],	
		(FileType[#] === Directory)&]
]

(* $Epilog is really an initialization *)

$Epilog := If[FindFile["end`"] =!= $Failed, Get[ "end`"] ];


(*
 If -noinit is not set and we are not a player kernel, then load
 
 	{$BaseDirectory,$UserBaseDirectory}/Autoload/*/{,Kernel}
	init.m from $BaseDirectory/Kernel.
	init.m from $UserBaseDirectory/Kernel.
*)

If[ !MemberQ[ $CommandLine, "-noinit"] && !($LicenseType === "Player Pro" || $LicenseType === "Player"),
	
	Scan[
		(System`Private`GetInitFile[FileNameJoin[{#, "Kernel"}]];
		 System`Private`GetInitFile[#];)&,
		Select[ Join[
			FileNames["*", FileNameJoin[{$BaseDirectory, "Autoload"}]],
			FileNames["*", FileNameJoin[{$UserBaseDirectory, "Autoload"}]]
			],
			(FileType[#] === Directory)&]
	];

	System`Private`GetInitFile[ FileNameJoin[{$BaseDirectory, "Kernel"}]];
	System`Private`GetInitFile[ FileNameJoin[{$UserBaseDirectory, "Kernel"}]]
		
]

(* read any -initfile arguments *)

System`Private`GetFileInsist /@
	Cases[Partition[$CommandLine, 2, 1], {"-initfile", System`Private`file_String} :> System`Private`file]

(* Persistent kernel initialization *)

System`Private`$InitialEnvironment = $EvaluationEnvironment; (* preserve the evidence *)

If[ FileType[FileNameJoin[{$InstallationDirectory, "SystemFiles", "Components"}]] === Directory &&
    	StartUp`Initialization`$PersistentInitialization,
	(* only cause loading if needed *)
	StartUp`Initialization`KernelInitialization[];
	System`Private`$Initialization = True;
]

System`Private`$InitsLoaded = True;

End[]  (* Global`*)

(* restore FE to what it was before *)

If[System`Private`origFrontEnd =!= Null,
    MathLink`RestoreFrontEnd[System`Private`origFrontEnd]
]

Null

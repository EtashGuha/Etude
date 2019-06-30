(* ::Package:: *)

BeginPackage["SystemInstallWinpCap`"];

Begin["`Private`"];



SystemInstall`Private`RegisterSystemInstallation["WinPcap",
	<|
		"URLCommand"->(
			Switch[$SystemID,
				"Windows-x86-64"|"Windows",
				{"https://www.winpcap.org/install/bin/WinPcap_4_1_3.exe", "WinPcap_4_1_3.exe"},
				(*no support for other platforms*)
				_,$Failed
			]&
		),
		"Message"->
		"The WinPcap software package is under a separate license."<>
		" By proceeding you understand and agree that WinPcap is a separate software package "<>
		"with separate licensing. ",
		"DependencyCheckCommand":>(
			If[!StringMatchQ[$SystemID,"Windows"~~___],
				{False,"WinPcap is only available to be installed on Windows."},
				{True,"Success"}
			]&
		),
		"FileHashCommand"->Function[{},
			Switch[$SystemID,
				"Windows"|"Windows-x86-64",
				214141661342423701618286573725407207629
			]
		],
		"PostInstallCommand"->None,
		"InteractiveInstallCommand"->Function[{executable,desiredLocation},
			(
				(*to run the installer, we have different commands to run per OS*)
				Switch[$SystemID,
					(*on windows, run the executable directly*)
					"Windows"|"Windows-x86-64",(
						Run[executable];
						(*on windows because we specifically told the installer what folder to use, check the folder we were provided*)
						If[AllTrue[FileExistsQ[FileNameJoin[{Environment["windir"],"System32",#}]]&]@{"Packet.dll","wpcap.dll"},
							desiredLocation,
							$Failed
						]
					),
					_,
					$Failed
				]
			)
		],
		(*no silent option for mac*)
		"SilentInstallSupportedQ"->(
			False&
		),
		"SilentInstallCommand"->None,
		"InstalledQCommand"->Function[{installedLoc},
			Switch[$SystemID,
				(*on windows, run the executable directly*)
				"Windows"|"Windows-x86-64",(
					(*on windows because we specifically told the installer what folder to use, check the folder we were provided*)
					AllTrue[FileExistsQ[FileNameJoin[{Environment["windir"],"System32",#}]]&]@{"Packet.dll","wpcap.dll"}
				),
				_,
				False
			]
		],
		(*also support Python3 as a name for this system*)
		"Aliases"->{"NetworkPacketCaptureLibraries","Wpcap","WindowsPCAP","WindowsPacketCaptureLibraries"},
		"PacletName"->Automatic
	|>
]

End[];

EndPackage[];

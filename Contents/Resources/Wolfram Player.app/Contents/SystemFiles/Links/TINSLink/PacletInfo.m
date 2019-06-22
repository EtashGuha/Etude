(* Paclet Info File *)

(* created 2017.08.04*)

Paclet[
	Name -> "TINSLink",
	Version -> "0.9.10",
	MathematicaVersion -> "11.2+",
	Creator ->"Alessio Sarti <alessios@wolfram.com>, Ian Johnson <ijohnson@wolfram.com>",
	Loading -> Automatic,
	Extensions -> {
		{
			"Kernel",
			Root -> "Kernel",
			Context ->
			{
				"TINSLink`"
			},
			Symbols-> {
				"System`NetworkPacketRecording",
				"System`NetworkPacketTrace",
				"System`NetworkPacketRecordingDuring",
				"System`NetworkPacketCapture",
				"System`$NetworkInterfaces",
				"System`$DefaultNetworkInterface"
			}
		},
		{"LibraryLink",SystemID->{"MacOSX-x86-64", "Linux", "Linux-ARM", "Linux-x86-64", "Windows", "Windows-x86-64"}}
	}
]
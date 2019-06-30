(* Paclet Info File *)

Paclet[
	Name -> "ZeroMQLink",
	Version -> "1.1.20",
	MathematicaVersion -> "11.2+",
	Creator ->"Ian Johnson <ijohnson@wolfram.com>",
	Loading->Automatic,
	Extensions -> {
		{
			"Kernel",
			Root -> "Kernel",
			Context -> 
			{
				"ZeroMQLinkLoader`","ZeroMQLink`"
			},
			Symbols-> 
			{
				"System`SocketConnect",
				"System`SocketObject",
				"System`SocketReadMessage",
				"System`SocketOpen",
				"System`SocketListen",
				"System`SocketListener",
				"System`SocketWaitAll",
				"System`SocketWaitNext",
				"System`SocketReadyQ",
				"System`Sockets"
			}
		},
		{"LibraryLink",SystemID->{"MacOSX-x86-64", "Linux", "Linux-ARM", "Linux-x86-64", "Windows", "Windows-x86-64"}}

		(* Uncomment the following block for Paclet Documentation *)
		(*
		{
			"Documentation",
			MainPage -> "Guides/ZeroMQLink",
			Language -> "English"
		},
		*)

		(* Uncomment the following block for Paclet Resources *)
		(*
		{
			"Resource",
			Root -> "Resources",
			Resources -> {}
		}
		*)
	}

]
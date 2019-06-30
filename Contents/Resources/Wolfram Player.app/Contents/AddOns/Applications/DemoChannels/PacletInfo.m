(* Paclet Info File *)

Paclet[
	Name -> "DemoChannels",
	Description -> "Demo channels for the Channel Framework",
    Creator -> "Igor Bakshee <bakshee@wolfram.com>",
	MathematicaVersion -> "10+",
	Version -> "0.5.10",
	Extensions -> {
		{"Kernel", Root->"Kernel",
			Context -> "DemoChannels`Oneliner`"
        },
		{"Kernel", Root->"Kernel",
			Context -> "DemoChannels`WIM`"
        },
        {"ChannelFramework"}
	}
]

If[!ValueQ[ChannelFramework`Channels`Demos`Dump`updated],
	If[Internal`$PrototypeBuild,
		PacletUpdate["DemoChannels", 
			"Site" -> "http://paclet-int.wolfram.com:8080/PacletServerInternal"]
		,
		PacletManager`Package`getPacletWithProgress["DemoChannels"]
	];
	ChannelFramework`Channels`Demos`Dump`updated = True;
]

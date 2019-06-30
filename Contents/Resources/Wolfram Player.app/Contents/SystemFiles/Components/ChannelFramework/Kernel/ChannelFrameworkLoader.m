If[False && Internal`$PrototypeBuild,
	ChannelFramework`debug`$Server = "dev";
	PacletManager`PacletUpdate["ChannelFramework", 
		"Site" -> "http://paclet-int.wolfram.com:8080/PacletServerInternal"]
	,
	PacletManager`Package`getPacletWithProgress["ChannelFramework"]
];

Get["ChannelFramework`"];
Get["ChannelFramework`ChannelReceiver`"];

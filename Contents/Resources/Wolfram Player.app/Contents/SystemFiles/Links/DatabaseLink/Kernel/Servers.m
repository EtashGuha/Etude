(*===================================================================*) 
(*====================== SQL Server =================================*) 
(*===================================================================*) 

Begin["`SQL`Private`"] 


SQLServerLaunch::address = "Invalid value for address: `1`"

SQLServerLaunch::port = "Invalid value for port: `1`"

Options[SQLServer] = {
    "Name" -> "", 
    "Description" -> "", 
    "Address" -> Automatic,
    "Port" -> Automatic,
    "SecureSockets" -> False,
    "Version" -> ""
}

Options[SQLServerLaunch] = Options[SQLServer]


$serverIndex = 0;

SQLServers[] := $sqlServers;

If[!ListQ[$sqlServers], 
    $sqlServers = {};
];

SQLServerLaunch[databases:{Rule[_String,_String] .. }, opts:OptionsPattern[]] := JavaBlock[Module[
	{server, useOpts, name, address, port, ss, id, s},
    Block[{$JavaExceptionHandler = ThrowException},
        Catch[
        
	        (* Process Options *)
	        useOpts  = Join[canonicalOptions[Flatten[{opts}]], Options[SQLServerLaunch]];
	        name     = Lookup[useOpts, "Name"];
	        address  = Lookup[useOpts, "Address"];
	        port     = Lookup[useOpts, "Port"];
	        ss       = Lookup[useOpts, "SecureSockets"];
	          
	        If[address =!= Automatic && !StringQ[address], 
	            Message[SQLServerLaunch::address, address];
	            Return[$Failed];
	        ];
	        
	        If[port =!= Automatic && !IntegerQ[port], 
	            Message[SQLServerLaunch::port, port];
	            Return[$Failed];
	        ];
	          
	        (* Create Server *)        
	        InstallJava[];
	        server = JavaNew["org.hsqldb.Server"];
	        KeepJavaObject[server];
	      
	        (* Configure server *)
	        server@setNoSystemExit[True];
	        If[address =!= Automatic, server@setAddress[address]];
	        If[port =!= Automatic, server@setPort[port]];
	        server@setTls[TrueQ[ss]];
	      
	        (* Add databases *)  
	        Do[
	            server@setDatabaseName[i - 1, First[databases[[i]]]];
	            server@setDatabasePath[i - 1, Last[databases[[i]]]];,
	            {i, Length[databases]}
	        ];
	        
	        (* Start server *)          
	        server@start[];
	          
	        id = ++$serverIndex;
	        s = SQLServer[server, id, opts];
	        AppendTo[$sqlServers, s];
	        s
        ] (* Catch *)
    ] (* Block *)
]]

SQLServerShutdown[SQLServer[server_, id_Integer, ___OptionQ]] := If[JavaObjectQ[server], 
    server@shutdown[];
    ReleaseJavaObject[server];
    $sqlServers = Drop[$sqlServers, 
        First@Position[$sqlServers, SQLServer[_, id, ___?OptionQ]]];
]  
  
SQLServerInformation[SQLServer[server_?JavaObjectQ, id_Integer, ___?OptionQ]] := 
    {{"ADDRESS", "PORT", "PRODUCT_NAME", "PRODUCT_VERSION", "PROTOCOL", "SECURE_SOCKETS", "STATE"}, 
     {server@getAddress[], server@getPort[], server@getProductName[], server@getProductVersion[], server@getProtocol[], server@isTls[], server@getStateDescriptor[]}}

SQLServer /: MakeBoxes[
    SQLServer[
        server_Symbol,
        id_Integer,
        opts___?BoxForm`HeldOptionQ
    ],
    StandardForm] := Module[
    
    {name, addr = "", port = "", icon = summaryBoxIcon, status = Style["Unavailable", {Italic, GrayLevel[0.55]}],
        sometimesOpts, o = canonicalOptions[Join[Flatten[{opts}], Options[SQLServer]]], oPrime},
    
    name = Lookup[o, "Name"];
    
    If[JavaObjectQ[server] && server =!= Null,
        status = Replace[server@getStateDescriptor[], {
            "ONLINE" -> Style["Online", {Black, Bold}],
            "SHUTDOWN" -> Style["Shutdown", {Black, Bold}]
        }];
        addr = server@getAddress[];
        port = server@getPort[],
        (* else *)
        Null
    ];

    sometimesOpts = Sort[
        DeleteCases[Options[SQLServer][[All, 1]], Alternatives @@ {
            "Name"
        }]
    ];

    oPrime = FilterRules[Join[{"Address" -> addr, "Port" -> port}, o], Options[SQLServer]];
    
    BoxForm`ArrangeSummaryBox[SQLServer, 
        SQLServer[server, id, opts],
        icon,
        (* Always *)
        {
            {BoxForm`SummaryItem[{"Name: ", name}], BoxForm`SummaryItem[{"ID: ", id}]},
            {BoxForm`SummaryItem[{"Status: ", status}], ""}
        },
        (* Sometimes *)
        BoxForm`SummaryItem[{# <> ": ", Replace[Lookup[oPrime, #], {Null -> "", None -> ""}]}] & /@ sometimesOpts,

        StandardForm
    ]
]

End[] (* `SQL`Private` *)

(* Author:          Christopher Williamson *)
(* Copyright:       Copyright 2004-2013, Wolfram Research, Inc. *)
Begin[ "`DataSources`Private`" ] 

Needs["ResourceLocator`"];
Needs["JLink`"];

Options[WriteDataSource] = 
    JoinOptions[
      {
        "Description" -> "", 
        "URL" -> Automatic, 
        "Username" -> Automatic, 
        "Password" -> Automatic,
        "Properties"-> Automatic,
        "Location" -> "User",
        "RelativePath" -> Automatic,
        "UseConnectionPool" -> Automatic
      },
      {
        "Catalog" -> Automatic,
        "ReadOnly" -> Automatic,
        "TransactionIsolationLevel" -> Automatic
      }
    ]

DatabaseResourcesPath::install = "Multiple installations of Database Package exist \
at `1`. This may lead to unpredictable results when running Database Package.";
WriteDataSource::location = "Illegal value for Location option: `1`"; 
WriteDataSource::url = "`1` does not support generating a URL. A URL must be supplied.";
WriteDataSource::exists = "DataSource `1` already exists.";

AddDatabaseResources[ x_String] := ResourceAdd[x, "DatabaseResources"];

DatabaseResourcesPath[] := ResourcesLocate["DatabaseResources"];

If[!MemberQ[DatabaseResourcesPath[], 
            {ToFileName[$UserBaseDirectory, "DatabaseResources"], None}], 
  AddDatabaseResources[ToFileName[$UserBaseDirectory, "DatabaseResources"]]
];
    
If[!MemberQ[DatabaseResourcesPath[], 
            {ToFileName[$BaseDirectory, "DatabaseResources"], None}], 
  AddDatabaseResources[ToFileName[$BaseDirectory, "DatabaseResources"]]
];

dataSourceQ[file_String] := 
  Module[{is, word},
    is = OpenRead[file];
    word = Read[is, Word , WordSeparators -> {" ", "\n", "\r", "\t", "["}];
    Close[is];
    word === "SQLConnection"
  ]
  
DataSources[] := Cases[ Flatten[FileNames["*.m", First[#]] & /@ DatabaseResourcesPath[]], 
    file_String /; (FileType[file] =!= Directory && dataSourceQ[file]) :> Append[Get[file], "Location" -> file]];
    
DataSources[dataSourceName_String] := FirstCase[DataSources[], SQLConnection[___, "Name" -> dataSourceName, ___], Null]

DataSourceNames[] := With[{opts = Join[canonicalOptions[Options[#]], Options[SQLConnection]]},
    Lookup[opts, "Name"]
] & /@ DataSources[];

queryQ[file_String] := Module[{is, word},
    is = OpenRead[file];
    word = Read[is, Word , WordSeparators -> {" ", "\n", "\r", "\t", "["}];
    Close[is];
    word === "SQLSelect"
]

SQLQueries[] := Cases[ Flatten[FileNames["*.m", First[#]]& /@ DatabaseResourcesPath[]], 
    file_String/;(FileType[file] =!= Directory && queryQ[file]):>Append[Get[file], "Location" -> file]]; 

SQLQueryNames[] := With[{opts = Join[canonicalOptions[Options[#]], Options[SQLSelect]]},
	Lookup[opts, "Name", None]
] & /@ SQLQueries[];

WriteDataSource[name_String, opts:OptionsPattern[]] := WriteDataSource[name, "HSQL(Standalone)", opts]

WriteDataSource[name_String, driver_String, opts:OptionsPattern[]] := Module[
	{desc, url, user, passwd, ucp, ro, til, cat, props, loc, i, rel, conn, 
         dbrdir, dir, useOpts},
          
     useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[WriteDataSource]];
     desc = Lookup[useOpts, "Description"]; 
     url = Lookup[useOpts, "URL"];
     user = Lookup[useOpts, "Username"]; 
     passwd = Lookup[useOpts, "Password"]; 
     props = Lookup[useOpts, "Properties"]; 
     loc = Lookup[useOpts, "Location"];
     rel = Lookup[useOpts, "RelativePath"]; 
     ucp = Lookup[useOpts, "UseConnectionPool"]; 
     cat = Lookup[useOpts, "Catalog"]; 
     ro = Lookup[useOpts, "ReadOnly"]; 
     til = Lookup[useOpts, "TransactionIsolationLevel"]; 
     
     Switch[loc,
       "User", 
         dbrdir = $UserBaseDirectory;
         dir = ToFileName[{$UserBaseDirectory, "DatabaseResources"}],
       "System", 
         dbrdir = $BaseDirectory;
         dir = ToFileName[{$BaseDirectory, "DatabaseResources"}], 
       _, 
         Message[WriteDataSource::location, loc];
         Return[$Failed]
     ]; 

     If[url === Automatic, 
       Switch[driver, 
         "HSQL(Standalone)" | "H2(Embedded)",
           url = name;
           If[user === Automatic, user = None];
           If[rel === Automatic, rel = True], 
         "Derby(Embedded)",
           url = name <> ";create=true",
         "HSQL(Memory)", 
           url = name;
           If[user === Automatic, user = None], 
         "SQLite(Memory)" | "H2(Memory)", 
           url = "";
           If[user === Automatic, user = None], 
         _, 
           Message[WriteDataSource::url, driver];
           Return[$Failed] 
       ];
     ];
     If[user === Automatic, user = None];
     If[passwd === Automatic, passwd = None];
     If[props === Automatic, props = {}];
     
     (* Make a unique name *) 
     loc = name <> ".m";
     i = 0;
     While[FileNames[loc, dir] =!= {},
       loc = name <> "(" <> ToString[++i] <> ").m"
     ];
     loc = ToFileName[dir, loc];
          
     conn = SQLConnection[JDBC[driver, url],
                             "Name" -> name,
                             "Description" -> desc,
                             "Username" -> user, 
                             "Password" -> passwd,
                             "Properties" -> props,
                             "RelativePath" -> TrueQ[rel],
                             "UseConnectionPool" -> ucp,
                             "Catalog" -> cat,
                             "ReadOnly" -> ro,
                             "TransactionIsolationLevel" -> til,
                             "Version" -> DatabaseLink`Information`$VersionNumber];
     
     If[FileNames["DatabaseResources", {dbrdir}] === {},  
       CreateDirectory[dir];
     ];
     
     If[MemberQ[DataSourceNames[], name], 
       Message[WriteDataSource::exists, name];
       $Failed, 
       Put[conn, loc];
       conn
     ]
]

(* Basic driver download framework. Users may run installJDBCDriver[name] to download and install the driver
        to their $UserBaseDirectory. Supported:
        o --
 *)

(* RDBMS-specific driver info captured in this hash.
 *      DownloadHandler: returns path to jar
 *)
 
(*
$installables = Association[
	"MySQL(Connector/J)" -> With[{version = "5.1.38"}, Association[
		"Version" -> version,
		"JDBCConfig" -> JDBCDriver[
            "Name" -> "MySQL(Connector/J)",
            "Driver" -> "com.mysql.jdbc.Driver",
            "Protocol" -> "jdbc:mysql://",
            "Version" -> DatabaseLink`Information`$VersionNumber, 
            "Description" -> StringTemplate["MySQL using Connector/J version `1`"][version]
        ],
        "URL" -> StringTemplate["http://dev.mysql.com/get/Downloads/Connector-J/mysql-connector-java-`1`.zip"][version],
        "ClassSlug" -> "mysql-connector-java",
        "JDBCConfigSlug" -> "MySQL-Connector-J",
        "DownloadHandler" -> Function[{path}, 
        	With[{files = ExtractArchive[path, mark@CreateDirectory[]]},
        		If[files === $Failed,
        			$Failed,
			        Replace[
			            Flatten@StringCases[files, __ ~~ $PathnameSeparator ~~ "mysql-connector-java-" ~~ version ~~ "-bin.jar"],
			            s : {__String} :> First[s]
			        ]
        		]
        	]
        ]
	]]
];
*)

(* N.B. File tracking not reentrant *)
(*
$outstanding = <||>;
mark[arg_] := (AppendTo[$outstanding, arg -> True]; arg)
clean[] := With[{files = Keys[$outstanding]}, Scan[
	(
	    If[
	    	Quiet@DirectoryQ[#],
	    	DeleteDirectory[#, DeleteContents -> True],
	    	Quiet@DeleteFile[#]
	    ];
	    KeyDropFrom[$outstanding, #]
    ) &
   , files
]]

statusUpdate[status_] := Print[Style[status, {Italic}]];
existingDrivers[slug_String] := Flatten@StringCases[JavaClassPath[], ___ ~~ $PathnameSeparator ~~ slug ~~ ___]
*)
(*
downloadDriver[driverName_String, url_] := Module[{path, archiveName},
    archiveName = FileNameTake[url];
    path = mark@URLSave[url, FileNameJoin[{$TemporaryDirectory, archiveName}]];
    If[path === $Failed,
        $Failed,
   
        $installables[driverName]["DownloadHandler"][path]
    ]
]

installDownloadedDriver[toDir_String, from_String] := Module[
    {to = toDir},
    If[!DirectoryQ[to],
        to = CreateDirectory[toDir, CreateIntermediateDirectories -> True]
    ];
    If[to === $Failed,
    	$Failed,
    
        CopyFile[from, FileNameJoin[{toDir, FileNameTake[from]}]]
    ]
];

installJar[driverName_String, to_String, url_String] := Module[
    {from, rval},
    statusUpdate["Downloading " <> url];
    from = downloadDriver[driverName, url];
     
    If[from === $Failed,
        statusUpdate["Download error"];
        Return[$Failed]
    ];
     
    rval = With[{cand = FileNameJoin[{to, FileNameTake[from]}]},
        If[Quiet@FileExistsQ[cand],
	        statusUpdate[FileNameTake[cand] <> " already installed"];
	        cand,
	        
	        If[FileExistsQ[from],
	            statusUpdate["Installing " <> FileNameTake[from]];
	            With[{installedFile = installDownloadedDriver[to, from]},
	                If[
	                    And[
                            installedFile =!= $Failed,
	            
				            ReinstallJava[];
				            MemberQ[existingDrivers[$installables[driverName]["ClassSlug"]], installedFile]
	                    ],
	           
	                    statusUpdate[StringTemplate["Installed `1` to `2`"][FileNameTake[from], to]];
	                    installedFile,
	                    (* else *)
	                    statusUpdate["Installation error"];
	                    $Failed
	                ]
	            ],
	            (* else *)
	            statusUpdate["Download error"];
	            $Failed
	        ]
        ]
    ];

    clean[];
    rval
];

installJDBCConfig[toDir_String, driverName_String, configFileBase_String] := Module[
    {to = toDir, config, loc, toPath},
    If[!DirectoryQ[to],
    	to = CreateDirectory[toDir, CreateIntermediateDirectories -> True]
    ];
    If[to === $Failed,
        $Failed,
      
        config = JDBCDrivers[driverName];
        loc = Lookup[Options[config], "Location"];
        toPath = FileNameJoin[{toDir, configFileBase <> ".m"}];
        If[config === Null,
        	statusUpdate["Writing JDBC config to " <> toPath];
            Check[
                Put[$installables[driverName]["JDBCConfig"], toPath];
                (* statusUpdate[StringTemplate["Named JDBC driver \"`1`\" now available."][driverName]]; *)
                to,
        
                statusUpdate["Error writing JDBC config to " <> toPath];
                $Failed
            ],
            (* else *)
            statusUpdate[StringTemplate["`1` JDBC configuration already installed at `2`"][driverName, loc]];
            loc
        ]
    ]
];
*)
(*
InstallJDBCDriver::notsupp = "`1` is not available for installation. Evaluate InstallJDBCDriver[] for a list \
of available drivers.";

InstallJDBCDriver[] := Sort[Keys[$installables]];

InstallJDBCDriver[name:"MySQL(Connector/J)"] := Module[
    {where = FileNameJoin[{$UserBaseDirectory, "Applications", "MySQL", "Java"}],
     whereJDBC = FileNameJoin[{$UserBaseDirectory, "Applications", "MySQL", "DatabaseResources"}] 
    },
  
    If[installJar[name, where, $installables[name]["URL"]] =!= $Failed,
        installJDBCConfig[whereJDBC, name, $installables[name]["JDBCConfigSlug"]]
    ];
    JDBCDrivers[name]
]

InstallJDBCDriver[args___] := (Message[InstallJDBCDriver::notsupp, args]; $Failed);
*)

End[]

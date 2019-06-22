(*===================================================================*)
(*=================== SQLConnection Functionality ===================*)
(*===================================================================*)

Begin["`SQL`Private`"] 


$SQLTimeout = Automatic;


SQLConnection::optreset = "Option `1` cannot be reset in `2`"
SQLConnection::conn = "Connection is not opened properly."
OpenSQLConnection::driver = "Illegal Driver value specified by the JDBCDriver: `1`"
OpenSQLConnection::location = "When the RelativePath option is set to True, the Location option must be a string: `1`"
OpenSQLConnection::notfound = "DataSource not found: `1`"
OpenSQLConnection::password = "Illegal value for Password option: `1`"
OpenSQLConnection::properties = "Illegal value for Properties option: `1`"
OpenSQLConnection::protocol = "Illegal Protocol value specified by the JDBCDriver: `1` (ignoring value)"
OpenSQLConnection::relativepath = "Illegal value for RelativePath option: `1`"
OpenSQLConnection::sqltimeout = "Illegal value for $SQLTimeout: `1`"
OpenSQLConnection::sqluseconnectionpool = "Illegal value for $SQLUseConnectionPool: `1`"
OpenSQLConnection::timeout = "Illegal value for Timeout option: `1` (continuing with option default value)"
OpenSQLConnection::useconnectionpool = "Illegal value for UseConnectionPool option: `1` (continuing with default value)"
OpenSQLConnection::username = "Illegal value for Username option: `1`"
OpenSQLConnection::noconnectorjdbc = "The specified JDBC configuration `1` refers to a driver not present \
in this installation of the Wolfram System. Evaluate `2` to download and install this driver.";
OpenSQLConnection::noconnectorjdbcalt = "The specified JDBC configuration `1` refers to a driver not present \
in this installation of the Wolfram System. Evaluate `2` to download and install this driver; alternatively, \
use one of the following drivers:\n\t`3`";
OpenSQLConnection::noconnector = "The specified driver class `1` is not present \
in this installation of the Wolfram System. Evaluate `2` to download and install this driver.";
OpenSQLConnection::noconnectoralt = "The specified driver class `1` is not present \
in this installation of the Wolfram System. Evaluate `2` to download and install this driver; alternatively, \
use one of the following drivers:\n\t`3`";
SQLConnectionUsableQ::notest = "No test query available for driver `1`; use two-argument form"
SQLConnection::til = "Illegal value for TransactionIsolationLevel option: `1`"
SQLConnection::readonly = "Illegal value for ReadOnly option: `1`"
SQLConnection::catalog = "Illegal value for Catalog option: `1`"

Options[ SetSQLConnectionOptions ] = {
    "Catalog" -> Automatic,
    "ReadOnly" -> Automatic,
    "TransactionIsolationLevel" -> Automatic
}

Options[ SQLConnection ] = JoinOptions[
    {
        "Name" -> None, 
        "Description" -> None, 
        "Username" -> None, 
        "Password" -> None,
        "Properties" -> {},
        "Location" -> None,
        "RelativePath" -> False,
        "UseConnectionPool" -> Automatic,
        "Version" -> None
    },
    Options[SetSQLConnectionOptions] 
]

Options[OpenSQLConnection] = JoinOptions[
    Options[SQLConnection],
    {
        "Timeout" :> $SQLTimeout
    }
]

Options[ SQLConnectionWarnings ] = {
    "ShowColumnHeadings" -> True
}

$connectionIndex = 0;
$poolIndex = 0;

If[!ListQ[$sqlConnections], 
    $sqlConnections = {}
];

(*This function modifies java.library.path after JVM has been started. This property becomes read only once JVM has initialized hence
 modifying it by using java`lang`System`setProperty has no effect on it. At JVM initialization if ClassLoader.sys_path=null then JVM takes notice of any new values in java.library.path.
This function uses reflection to explicitly set the value of internal field ClassLoader.sys_path to null, which forces the JVM to take notice of the new value (here path to ntlmauth.dll)*)
modifyJavaLibPath[] := Quiet@Module[{oldPath, newPath, separatorChar, class, field},
	InstallJava[];
    LoadJavaClass["java.lang.System"];
    oldPath = java`lang`System`getProperty["java.library.path"];
    separatorChar = java`lang`System`getProperty["path.separator"];
    newPath = StringJoin[Riffle[Append[StringSplit[oldPath, separatorChar], FileNameJoin[{$DatabaseLinkDirectory, "Java", "Libraries", If[$SystemWordLength == 64, "Windows-x86-64", "Windows"]}]], separatorChar]];
    java`lang`System`setProperty["java.library.path", newPath];
    LoadJavaClass["com.wolfram.jlink.JLinkClassLoader"];
    class = JLinkClassLoader`classFromName["java.lang.ClassLoader"];
    field = class@getDeclaredField["sys_paths"];
    java`lang`reflect`Field`setAccessible[{field}, True];
    field@set[Null, Null]
]

SQLConnections[] := $sqlConnections;

(* Overrides for specific drivers *)
OpenSQLConnection[
    JDBC[driver:"org.apache.derby.jdbc.EmbeddedDriver"|"Derby(Embedded)", url_String], opts:OptionsPattern[]] := 
    Module[{},
    DatabaseLink`SQL`Private`derbySecurityFix[];
    openSQLConnectionImplementation[JDBC[driver, url], opts]
    ]

OpenSQLConnection[JDBC[driver:"net.sourceforge.jtds.jdbc.Driver"|"Microsoft SQL Server(jTDS)",url_String],opts:OptionsPattern[]] := 
    Module[{},
	DatabaseLink`SQL`Private`modifyJavaLibPath[];
	openSQLConnectionImplementation[JDBC[driver,url],opts]
    ]

OpenSQLConnection[JDBC[driver_String, url_String], opts:OptionsPattern[]] := openSQLConnectionImplementation[JDBC[driver, url], opts]

openSQLConnectionImplementation[JDBC[driver_String, url_String], opts:OptionsPattern[]] := JavaBlock[Module[
    {result, useOpts, name, username, password, useConnectionPool, location, relativePath, timeout,
         properties, readOnly, transactionIsolationLevel, catalog,
         u = url, d = driver, jdbc, protocol, props,
         connectionPool = Null, basicDataSource, to, id, connection, conn, fromPool = False},

    Block[{$JavaExceptionHandler = ThrowException},
        result = Catch[

            (* Process options 
             * Description and Version are informational options that are not used here. *)
            useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[OpenSQLConnection]];
            name = Lookup[useOpts, "Name"];
            location = Lookup[useOpts, "Location"];
            relativePath = Lookup[useOpts, "RelativePath"];
            username = Lookup[useOpts, "Username"];
            password = Lookup[useOpts, "Password"];
            timeout = Lookup[useOpts, "Timeout"];
            useConnectionPool = Lookup[useOpts, "UseConnectionPool"];
            properties = Lookup[useOpts, "Properties"];
            readOnly = Lookup[useOpts, "ReadOnly"];
            transactionIsolationLevel = Lookup[useOpts, "TransactionIsolationLevel"];
            catalog = Lookup[useOpts, "Catalog"];

            (* Location specifies the filename that is used to store the connection.  
               RelativePath uses the Location directory as the base directory for the URL parameter. 
               For file URLs this will allow the URL to find the database relative to the Location 
               directory.  However, this is not applicable to other URLs, so RelativePath should 
               be set to False when using URLs that are not file based. If RelativePath is set to 
               True, the URL will be treated as a file URL and the base directory will be set. *)
            Switch[relativePath, 
                False,
                Null,
                 
                True, 
                If[StringQ[location],
                    u = FileNameJoin[{If[DirectoryQ[location], location, DirectoryName[location]], u}],
                    Message[OpenSQLConnection::location, location];
                    Return[$Failed]
                ],
                  
                _, 
                (* Since the user is attempting to use relativePath it appears the default location is not 
                   acceptable to the user. Since continuing using a default would result in the database 
                   being created in a place the user does not wish, $Failed is returned when 
                   RelativePath is invalid. *)
                Message[OpenSQLConnection::relativepath, relativePath];
                Return[$Failed]
            ];
      
            (* The driver parameter may be used to specify a JDBCDriver configuration.
               If the driver value is found among the names of JDBCDriver configurations, 
               then the driver value will be set to the value of the driver option specified 
               in the JDBCDriver configuration.  This must be a Java class to work correctly.  
               Also the protocol specified within the JDBCDriver configuration is prepended 
               to the URL, if it is not already there. This saves users from having to 
               remember complicated protocols. *)
            jdbc = SelectFirst[JDBCDrivers[], Lookup[canonicalOptions[Options[#]], "Name"] === driver &];
            If[!MatchQ[jdbc, _Missing], 
                {d, protocol} = Lookup[Join[canonicalOptions[Options[jdbc]], Options[JDBCDriver]], {"Driver", "Protocol"}];
                Which[
                    !StringQ[d], 
                    (* If the driver is not a String, $Failed is returned.  This cannot be fixed at this point. *)
                    Message[OpenSQLConnection::driver, d];
                    Return[$Failed],
                  
                    !StringQ[protocol], 
                    (* If the protocol is not a String, a Message is returned, but the function 
                       ignores the protocol and continues on. It could be that the user has 
                       already specified the correct protocol. *)
                    Message[OpenSQLConnection::protocol, protocol],
                  
                    !StringMatchQ[u, protocol <> "*"],
                    u = protocol <> u
                ]
            ];

            (* Initialize Java and Java classes *)
            InstallJava[];
            LoadJavaClass["com.wolfram.jlink.JLinkClassLoader"];
            LoadJavaClass["com.wolfram.databaselink.JDBCConnectionManager"];

            (* Initialize the JDBC driver.  This is required per JDBC. *)
            JLinkClassLoader`classFromName[d]@newInstance[];

            (* Set useConnectionPool to the global default if invalid. *)
            If[useConnectionPool === Automatic, useConnectionPool = $SQLUseConnectionPool];
            If[useConnectionPool =!= True && useConnectionPool =!= False,
                Message[OpenSQLConnection::useconnectionpool, useConnectionPool];
                useConnectionPool = $SQLUseConnectionPool
            ];
        
            (* Process the useConnectionPool option *)
            Switch[useConnectionPool, 
                True | False,
                Null, 
                (* A Message is returned if the value is not True or False.  However, TrueQ 
                   is called on the value, so if the value is not True then it is False. 
                   So this does not fail for invalid values.  Rather it tries to continue by 
                   not using connection pools. *)
                _, Message[OpenSQLConnection::sqluseconnectionpool, useConnectionPool];
            ];

            (* If specified by Password, prompt for the password.*)
            If[(StringQ[password] && StringMatchQ[password, "$Prompt"]) && 
                (* Prompt only if the connection pool has not started or if a connection pool is not used *)
                ((TrueQ[useConnectionPool] && SQLConnectionPools[SQLConnection[JDBC[driver, url], opts]] === Null) || 
                !TrueQ[useConnectionPool]), 
                (* 
                 * PasswordDialog uses DialogInput and will hang the FE if OpenSQLConnection has been called from within
                 * another DialogInput. If calling from another DialogInput, intercept the password check
                 * and use NestablePasswordDialog. See DataSourceWizard.m for an example of this technique.
                 *)
                {username, password} = DatabaseLink`UI`Private`PasswordDialog[{username, None}];
            ];

            (*
             * Set the timeout to the global default if invalid.
             * setLoginTimeout(int seconds) only
             *)
            If[(!IntegerQ[timeout] && timeout =!= None && timeout =!= Automatic) || timeout <= 0, 
                Message[OpenSQLConnection::timeout, timeout];
                timeout = $SQLTimeout;
            ];

            (* Process the timeout option *)
            Switch[timeout,
                (* These will effectively fall back on driver defaults in JDBCConnectionManager. *)
                None|Automatic,
                timeout = 0,
                
                _Integer?NonNegative,
                Null,
            
                _,
                Message[OpenSQLConnection::sqltimeout, timeout];
                Return[$Failed];
            ];

            (* Process the properties including Username and Password *)
            If[MatchQ[properties, {(_String -> _String) ...}], 
                (*
                 * Bug 249675: HSQL in standalone mode won't delete lock files on connection close, unless
                 * a SHUTDOWN command is issued first or the shutdown property is true. Properties are processed in order,
                 * so this can be overridden by user settings.
                 *)
                If[d === "HSQL(Standalone)", PrependTo[properties, "shutdown" -> "true"]];
                props = JavaNew["java.util.Properties"];
                props@setProperty[First[#], Last[#]] & /@ properties;
                Switch[username, 
                    None, 
                    Null,
                     
                    _String, 
                    Switch[password, 
                        None, 
                        Null,
                         
                        _String, 
                        (* Only set username and password if they are both strings *)
                        props@setProperty["user", username];
                        props@setProperty["password", password],
                         
                        _, 
                        Message[OpenSQLConnection::password, password]
                    ],
                
                    _, 
                    Message[OpenSQLConnection::username, username]
                ];,
                (* else *)
                (* Since the properties may be very important to how a connection is made, 
                   $Failed is returned when invalid Properties are received. *)
                Message[OpenSQLConnection::properties, properties];
                Return[$Failed]
            ];
          
            If[TrueQ[useConnectionPool],
                (* Make connection using connection pool *)
                connectionPool = SQLConnectionPools[SQLConnection[JDBC[driver, url], opts]];
                If[MatchQ[connectionPool, Null|_Missing],
                    basicDataSource = JDBCConnectionManager`getPool[d, u, props];
                    connectionPool = SQLConnectionPool[basicDataSource, JDBC[driver, url], ++$poolIndex, 
                        Sequence @@ DeleteCases[useOpts, "Timeout" -> _]
                    ];
                    AppendTo[$connectionPools, connectionPool];
                    AppendTo[$poolToConnections, $poolIndex -> {}];
                    KeepJavaObject[basicDataSource];
                    , 
                    basicDataSource = First[connectionPool];
                ];
            
                (* Set the pool options here so when a user specifies a dynamic 
                   property, it will be updated in an existing pool. *)
                SetSQLConnectionPoolOptions[connectionPool, Sequence @@ FilterRules[useOpts, Options[SetSQLConnectionPoolOptions]]];

                to = basicDataSource@getMaxWait[];
                basicDataSource@setMaxWait[timeout*1000];
                connection = basicDataSource@getConnection[];
                fromPool = True;
                basicDataSource@setMaxWait[to];
                ,
                (* else *)
                (* Make connection without using connection pool. *)
                connection = JDBCConnectionManager`getConnection[u, props, timeout];
            ];

            (* Setup SQLConnection expression *)
            id = ++$connectionIndex;
            conn = SQLConnection[JDBC[driver, url], connection, id, opts];
          
            (* Set options that may be configured dynamically. *)  
            conn = SetSQLConnectionOptions[conn, 
                "ReadOnly" -> readOnly,
                "TransactionIsolationLevel" -> transactionIsolationLevel,
                "Catalog" -> catalog
            ];
          
            (* Protect the connection from cleanup in Java *)
            KeepJavaObject[connection];
          
            (* Add SQLConnection to the list of open connections. *)
            AppendTo[$sqlConnections, conn];
            If[TrueQ@fromPool, 
                AppendTo[$poolToConnections[$poolIndex], conn];
            ];
            conn
        ]; (* Catch *)
        If[result === $Failed && TrueQ[useConnectionPool] && connectionPool =!= Null, 
            SQLConnectionPoolClose[connectionPool];
        ];
        result
    ] (* Block *)
]]
    
OpenSQLConnection[SQLConnection[jdbc_JDBC, opts:OptionsPattern[]], opts2:OptionsPattern[]] := 
    OpenSQLConnection[SQLConnection[jdbc, Null, -1, opts], opts2]

OpenSQLConnection[SQLConnection[
                    jdbc_JDBC,
                    _,
                    _Integer,
                    opts:OptionsPattern[]], 
                  opts2:OptionsPattern[]] := Module[
    {cat, desc, location, name, pw, ro, relativePath, to, transactionIsolationLevel, un, v, ucp, properties},

    (* The options are processed so that options specified in the connection may be overriden 
       by options specified in the function. *)
    {cat, desc, location, name, pw, properties, ro, relativePath, to, transactionIsolationLevel, ucp, un, v} = Lookup[
        Join[canonicalOptions[Flatten[{opts2}]], canonicalOptions[Flatten[{opts}]], Options[OpenSQLConnection]],
        {"Catalog", "Description", "Location", "Name", "Password", 
            "Properties", "ReadOnly", "RelativePath", "Timeout", 
            "TransactionIsolationLevel", "UseConnectionPool", "Username", "Version"}
    ];

    OpenSQLConnection[jdbc, "Catalog" -> cat, "Description" -> desc, "Location" -> location, "Name" -> name, "Password" -> pw,
                            "Properties" -> properties, "ReadOnly" -> ro, "RelativePath" -> relativePath, "Timeout" -> to,
                            "TransactionIsolationLevel" -> transactionIsolationLevel, "UseConnectionPool" -> ucp, "Username" -> un, 
                            "Version" -> v]
]

OpenSQLConnection[name_String, opts:OptionsPattern[]] := Module[{src},
    src = FirstCase[DataSources[], SQLConnection[___, "Name" -> name, ___]];

    If[MatchQ[src, _Missing],
        Message[OpenSQLConnection::notfound, name];
        $Failed,
        (* else *)
        OpenSQLConnection[src, opts]
    ]
]

SQLConnection /: SetOptions[ SQLConnection[jdbc_JDBC,
                                connection_,
                                id_Integer,
                                opts:OptionsPattern[]], opts2___] := 
    SetSQLConnectionOptions[ SQLConnection[jdbc, connection, id, opts], opts2]


SetSQLConnectionOptions[SQLConnection[
                          jdbc_JDBC,
                          connection_,
                          id_Integer,
                          opts:OptionsPattern[]], 
                        opts2:OptionsPattern[]] := Module[
    {cat, desc, location, name, pw, properties, ro, relativePath, til, un, ucp, v, conn, optTest},
    Block[
        {$JavaExceptionHandler = ThrowException},
      Catch[
        {desc, location, name, pw, properties, relativePath, ucp, un, v} = Lookup[
            Join[canonicalOptions[Flatten[{opts}]], Options[OpenSQLConnection]],
            {"Description", "Location", "Name", "Password", "Properties", 
                "RelativePath", "UseConnectionPool", "Username", "Version"} 
        ];
    
        optTest = FilterRules[ {opts2}, Except[Options[SetSQLConnectionOptions]]];
        If[optTest =!= {}, optionsErrorMessage[optTest, SQLConnection, SQLConnection]; Return[$Failed]];
        
        {cat, ro, til} = Lookup[
            Join[canonicalOptions[Flatten[{opts2}]], canonicalOptions[Flatten[{opts}]], Options[OpenSQLConnection]],
            {"Catalog", "ReadOnly", "TransactionIsolationLevel"}
        ];
    
        If[!JavaObjectQ[connection], 
          Message[SQLConnection::conn];
          Return[$Failed]
        ]; 
         
        (* Catalog *)
        Switch[cat, 
          _?StringQ, 
            connection@setCatalog[cat],
          Automatic, 
            Null,
          _, 
            Message[SQLConnection::catalog, cat];
            cat = Lookup[Join[canonicalOptions[Flatten[{opts}]], Options[OpenSQLConnection]], "Catalog"];
        ];
    
        (* Transaction Isolation Level *)
        Switch[til, 
          "ReadUncommitted",
            connection@setTransactionIsolation[1],
          "ReadCommitted",
            connection@setTransactionIsolation[2],
          "RepeatableRead",
            connection@setTransactionIsolation[4],
          "Serializable",
            connection@setTransactionIsolation[8],
          Automatic, 
            Null,
          _, 
            Message[SQLConnection::til, til];
            til = Lookup[Join[canonicalOptions[Flatten[{opts}]], Options[OpenSQLConnection]], "TransactionIsolationLevel"];
        ];
    
        (* Read Only *)
        Switch[ro, 
          (True | False), 
            connection@setReadOnly[ro],
          Automatic, 
            Null,
          _, 
            Message[SQLConnection::readonly, ro];
            ro = Lookup[Join[canonicalOptions[Flatten[{opts}]], Options[OpenSQLConnection]], "ReadOnly"];
        ];
    
        conn = SQLConnection[jdbc, connection, id, "Catalog" -> cat, "Description" -> desc, 
             "Location" -> location, "Name" -> name, "Password" -> pw,
             "Properties" -> properties, "ReadOnly" -> ro, "RelativePath" -> relativePath,
             "TransactionIsolationLevel" -> til, "UseConnectionPool" -> ucp, "Username" -> un, "Version" -> v];
        $sqlConnections = Replace[$sqlConnections, SQLConnection[_, _, id, ___] -> conn, 1];
        conn
      ]
    ]
  ];


CloseSQLConnection[ SQLConnection[_JDBC, connection_, id_Integer, ___?OptionQ]] := Block[
    {$JavaExceptionHandler = ThrowException},
    Catch[
        If[JavaObjectQ[connection],
           If[!connection@isClosed[],
               connection@close[]
           ];
           ReleaseJavaObject[connection];
           $inTransaction = False;
           $sqlConnections = Drop[$sqlConnections, 
               First@Position[$sqlConnections, SQLConnection[_, _, id, ___?OptionQ]]
           ]
        ]
    ]
]

(*
 * http://download.oracle.com/javase/1.4.2/docs/api/java/sql/Connection.html#isClosed()
 * This method itself is not reliable enough to test for usable connections.
 *)
SQLConnectionOpenQ[SQLConnection[_JDBC, conn_, id_Integer, opts:OptionsPattern[]]] := 
    (JavaObjectQ[conn] && !conn@isClosed[]);
    
SQLConnectionOpenQ[_] = False;

(*
 * This executes a query on the passed connection and may raise an exception in some
 * cases, like an open streaming result set on the connection. Note there isn't a 1:1 mapping
 * between the JDBC driver and the RDBMS, but there isn't a clean way to infer the RDBMS from 
 * the connection object.
 *)
$usabilityTests = {
    StartOfString ~~ "Oracle" ~~ ___ ~~ EndOfString -> {"SELECT 1 FROM DUAL", {{1.}}}
    , StartOfString ~~ "HSQL Database Engine" ~~ ___ ~~ EndOfString -> {"SELECT 1 FROM INFORMATION_SCHEMA.SYSTEM_USERS", {{1}}}
    , StartOfString ~~ "Firebird" ~~ ___ ~~ EndOfString -> {"SELECT 1 FROM RDB$RELATIONS WHERE 1=0", {}}
    , StartOfString ~~ "ACCESS" ~~ ___ ~~ EndOfString -> {"SELECT 1", {{1}}}
    , StartOfString ~~ "EXCEL" ~~ ___ ~~ EndOfString -> {"SELECT 1", {{1}}}
    , StartOfString ~~ "Microsoft SQL Server" ~~ ___ ~~ EndOfString -> {"SELECT 1", {{1}}}
    , StartOfString ~~ "MySQL" ~~ ___ ~~ EndOfString -> {"SELECT 1", {{1}}}
    , StartOfString ~~ "PostgreSQL" ~~ ___ ~~ EndOfString -> {"SELECT 1", {{1}}}
    , StartOfString ~~ "SQLite" ~~ ___ ~~ EndOfString -> {"SELECT 1",  {{1}}}
    , StartOfString ~~ "H2" ~~ ___ ~~ EndOfString -> {"SELECT 1", {{1}}}
    , StartOfString ~~ "Apache Derby" ~~ ___ ~~ EndOfString -> {"VALUES 1", {{1}}}
    (*, StartOfString ~~ __ ~~ EndOfString -> {"SELECT 1", {{1}}}*)
};

SQLConnectionUsableQ[conn_SQLConnection] := 
    SQLConnectionUsableQ[conn, StringCases[getRDBMS[conn], $usabilityTests]] /;
    SQLConnectionOpenQ[conn];
    
SQLConnectionUsableQ[conn_SQLConnection, {}] := (
    Message[SQLConnectionUsableQ::notest, getRDBMS[conn]];
    (* Fall through to SELECT 1 *)
    SQLConnectionUsableQ[conn, {"SELECT 1", {{1}}}]
)
    
SQLConnectionUsableQ[conn_SQLConnection, {{testSQL_String, res_}}] := 
    SQLConnectionUsableQ[conn, {testSQL, res}];
    
SQLConnectionUsableQ[conn_SQLConnection, {testSQL_String, res_}] := (
    MatchQ[Quiet@SQLExecute[conn, testSQL], res]
)

SQLConnectionUsableQ[_] = False;

(*
 * Given an SQLConnection returns the RDBMS as a verbose-ish string.
 * This is the basic unit of reliability in doing dialect switches.
 * This works for connections pulled from a pool, or not.
 *)
getRDBMS[SQLConnection[_JDBC, conn_?JavaObjectQ, _Integer, ___?OptionQ]] := 
    getRDBMS[conn];

getRDBMS[conn_?JavaObjectQ] := 
    conn@getMetaData[]@getDatabaseProductName[];

(* A couple MySQL drivers support weird settings for streaming result sets;
 * cataloged here. *)
$MySQLStreamingClasses = {
    "org.mariadb.jdbc.MariaDbConnection" (* from org.mariadb.jdbc.Driver *),
    "com.mysql.jdbc.JDBC4Connection" (* from com.mysql.jdbc.Driver *)
};
supportsMySQLStreamingQ[conn_?JavaObjectQ] := With[
    {driver = conn@getClass[]@getName[]},
    MemberQ[$MySQLStreamingClasses, driver]
]
supportsMySQLStreamingQ[___] := False;


SQLConnectionInformation[SQLConnection[_JDBC, connection_, _Integer, ___?OptionQ]] := Block[
    {$JavaExceptionHandler = ThrowException},
    Catch[
        If[!JavaObjectQ[connection], 
            Message[SQLConnection::conn];
            Return[$Failed]
        ];
      
        LoadJavaClass["com.wolfram.databaselink.SQLStatementProcessor"];
        SQLStatementProcessor`getConnectionMetaData[ connection ]
    ]
]

SQLConnectionInformation[
    SQLConnection[_JDBC, connection_, _Integer, ___?OptionQ],
    (metaDataItem_String | metaDataItem_List)
] := Block[
    {$JavaExceptionHandler = ThrowException},
    Catch[
        If[!JavaObjectQ[connection], 
            Message[SQLConnection::conn];
            Return[$Failed]
        ];
      
        LoadJavaClass["com.wolfram.databaselink.SQLStatementProcessor"];
        SQLStatementProcessor`getConnectionMetaData[ connection, metaDataItem ]
    ]
]

SQLConnectionWarnings[
    SQLConnection[_JDBC, connection_, _Integer, ___?OptionsQ],
    opts:OptionsPattern[]
] := JavaBlock[Module[
    {useOpts, sch, warn, warnList},
    Block[
        {$JavaExceptionHandler = ThrowException},
        Catch[
            useOpts = Join[canonicalOptions[Flatten[{opts}]], Options[SQLConnectionWarnings]];
            warnList = {};
                  
            sch = Lookup[useOpts, "ShowColumnHeadings"];
            If[TrueQ[sch],
                AppendTo[warnList, {"Message", "SQLState", "ErrorCode"}]
            ];

            If[!JavaObjectQ[connection],
                Message[SQLConnection::conn];
                Return[$Failed]
            ];
                  
            warn = connection@getWarnings[];
                  
            (* During testing the below warning messages were created/forced to test
               the While loop and front-end display, since a connection warning
               could not be caused.
                       
               warn=JavaNew["java.sql.SQLWarning","HELP"];
               warn2=JavaNew["java.sql.SQLWarning","HELP2"];
               warn@setNextWarning[warn2];
             *)
                  
            While[warn =!= Null,
                AppendTo[warnList, {warn@getMessage[], warn@getSQLState[], warn@getErrorCode[]}];
                warn = warn@getNextWarning[];
            ];
            warnList
        ] (* Catch *)
    ] (* Block *)
]]


(* Recent versions of Derby require that the JVM be launched with a security policy *)
derbySecurityFix[] := Module[
	{cls, derbyPath, policyURL},
	InstallJava[];
	LoadJavaClass["com.wolfram.jlink.JLinkClassLoader"];
    cls = JLinkClassLoader`classFromName["org.apache.derby.jdbc.EmbeddedDriver"];
    derbyPath = cls@getProtectionDomain[]@getCodeSource[]@getLocation[]@toString[];
    policyURL = StringReplace[derbyPath, "derby.jar" -> "derby.policy"];
    ReinstallJava[JVMArguments -> "-Djava.security.policy=" <> policyURL]
]


SQLConnection /: MakeBoxes[
    SQLConnection[
        j:JDBC[_String, _String],
        conn_Symbol,
        id_Integer,
        opts___?BoxForm`HeldOptionQ
    ],
    StandardForm] := Module[
    
    {name, icon = summaryBoxIcon, til = "", status = Style["Closed", {Italic, GrayLevel[0.55]}], 
        catalog = "", ro = "", sometimesOpts, o = Join[canonicalOptions[Flatten[{opts}]], Options[SQLConnection]], oPrime},

    name = Lookup[o, "Name"]; 

    If[JavaObjectQ[conn] && conn =!= Null && !conn@isClosed[],
        status = Style["Open", {Black, Bold}];
        catalog = With[{c = conn@getCatalog[]}, If[StringQ[c], c, ""]];
        ro = conn@isReadOnly[];
        (* Not all drivers implement this method, so Quiet here *)
        til = Quiet@Switch[conn@getTransactionIsolation[],
            2, "ReadCommitted",
            4, "RepeatableRead",
            8, "Serializable",
            _, Null
        ],
        (* else *)
        Null
    ];

    sometimesOpts = Sort[
        DeleteCases[Options[SQLConnection][[All, 1]], Alternatives @@ {
            "Catalog",
            "Name",
            "Username",
            "Password",
            "Properties"
        }]
    ];

    oPrime = FilterRules[Join[{"ReadOnly" -> ro, "TransactionIsolationLevel" -> til}, o], Options[SQLConnection]];
    
    BoxForm`ArrangeSummaryBox[SQLConnection, 
        SQLConnection[j, conn, id, opts],
        icon,
        (* Always *)
        {
            {BoxForm`SummaryItem[{"Name: ", name}], BoxForm`SummaryItem[{"ID: ", id}]},
            {BoxForm`SummaryItem[{"Status: ", status}], BoxForm`SummaryItem[{"Catalog: ", catalog}]}
        },
        (* Sometimes *)
        BoxForm`SummaryItem[{# <> ": ", Replace[Lookup[oPrime, #], {Null -> "", None -> ""}]}] & /@ sometimesOpts,

        StandardForm
    ]
]

End[] (* `SQL`Private` *)

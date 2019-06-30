
If[TrueQ[Databases`Private`$DevelopmentMode],
    With[
        {
            debugMode = Databases`Common`$DBDebugMode,
            showStack = Databases`Common`$DBShowStackTraceInDebug,
            defaultFormattingRules = Databases`Common`StackTraces`PackagePrivate`$defaultFormattingRules,
            executableStackFrames = Databases`Common`$DBExecutableStackFrames
        },
        Databases`Private`ClearPackage["Common"];
        Databases`Common`$DBDebugMode = debugMode;
        Databases`Common`$DBShowStackTraceInDebug = showStack;
        Databases`Common`StackTraces`PackagePrivate`$defaultFormattingRules = defaultFormattingRules;
        Databases`Common`$DBExecutableStackFrames = executableStackFrames;
    ]
]


Get["Databases`Common`Common`"]

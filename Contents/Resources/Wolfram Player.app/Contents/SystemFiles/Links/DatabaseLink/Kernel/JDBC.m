(*===================================================================*)
(*======================= JDBC Functionality ========================*)
(*===================================================================*)

Begin["`SQL`Private`"] 

JDBC::error = "`1`"

JDBC::classnotfound = "`1`"


Options[ JDBCDriver ] = {
    "Name" -> "" , 
    "Description" -> "" , 
    "Driver" -> "", 
    "Protocol" -> "",
    "Location" -> "",
    "Version" -> ""
}

JDBCDrivers[] := Cases[ Flatten[FileNames["*.m", First[#]]& /@ DatabaseResourcesPath[]], 
    file_String /; (FileType[file] =!= Directory && jdbcDriverQ[file]) :> Append[Get[file], "Location" -> file]];

JDBCDrivers[driverName_String] := FirstCase[JDBCDrivers[], JDBCDriver[___, "Name" -> driverName, ___], Null]
    
JDBCDriverNames[] := With[{opts = Join[canonicalOptions[Options[#]], Options[JDBCDriver]]},
    Lookup[opts, "Name"]
] & /@ JDBCDrivers[];

jdbcDriverQ[file_String] := Module[{is, word},
    is = OpenRead[file];
    word = Read[is, Word , WordSeparators -> {" ", "\n", "\r", "\t", "["}];
    Close[is];
    word === "JDBCDriver"
]

End[] (* `SQL`Private` *)

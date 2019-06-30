(*===================================================================*) 
(*====================== Stored Procedures ==========================*) 
(*===================================================================*) 

Begin["`SQL`Private`"] 


(*

This prototype was in the repo prior to version 3. I don't know how far along it is. --dillont

InstallStoredProcedures[SQLConnection[ _JDBC, connection_, _Integer, ___?OptionQ],
                        context_String] :=
  Module[{meta, procedures},

    If[!StringMatchQ[context, "*`"], 
      Message[
        InstallService::"context", 
        context, 
        "contexts must end with a '`'"];
      Return[$Failed];
    ];
  
    If[!MatchQ[
         StringPosition[
           context, 
           {".", "_", "~", "!", "@", "#", "$", "%", "^", 
            "&", "*", "(", ")", "-", "+", "=", "{", "[", 
            "}", "]", "|", "\\", ":", ";", "\"", "\'", 
            "<", ",", ">", "?", "/", " "}], 
         {}], 
      Message[
        InstallService::"context", 
        context, 
        "contexts must be alpha-numeric."];
      Return[$Failed];
    ];
 
    positions = 
      Drop[Prepend[(First[#] + 1) & 
        /@ StringPosition[context, "`"], 1], -1];
    test = (If[DigitQ[StringTake[context, {#,#}]], $Failed] & /@ positions);
    If[Length[Cases[test, $Failed]] > 0, 
      Message[
        InstallService::"context", 
        context, 
        "contexts must not begin with a digit."];
      Return[$Failed];
    ];
    
    meta = connection@getMetaData[];
    procedures = meta @getProcedures[Null, Null, Null];
    SQLStatementProcessor`getResultData[procedures, False, False]
  ]
*)


End[] (* `SQL`Private` *)

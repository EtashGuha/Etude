(* Support for Google Apps Script (scipritng language for Google Docs, Google Sheets, Google Forms, etc.)

   Return types supported (e.g., EmbedCode[apiFunc, "GoogleDocs", ExternalFunctionSignature->{Automatic, "return type"}])
   
        Number
        String (default)
        Boolean
        Date
        Array
        Blob
*)



(* Pick a unique private context for each implementation file. *)
Begin["EmbedCode`GoogleDocs`Private`"]


(* JavaScript-Common.m isn't automatically loaded for GoogleDocs because its name doesn't match. Load it manually. *)
Get[FileNameJoin[{EmbedCode`Private`$environmentsDir, "JavaScript-Common.m"}]]


EmbedCode`Common`iEmbedCode["googledocs", apiFunc_APIFunction, url_, opts:OptionsPattern[]] := 
    Module[{paramInfo, returnType, finalArgSpec, code, sig, funcName, argTypes, argsStr, payloadSection, js2wlDef},
        {sig, funcName} = OptionValue[{ExternalTypeSignature, ExternalFunctionName}];
        If[sig === Automatic, sig = {Automatic, Automatic}];
        If[!StringQ[funcName], funcName = $defaultFunctionName];
        argTypes = sig[[1]];
        returnType = sig[[2]];
        (* paramInfo will be a list of triples: {{"name", type, default (symbol None if no default)}, ...} *)
        (* For now, the default values are being ignored *)
        paramInfo = EmbedCode`Common`getParamInfo[apiFunc];
        finalArgSpec = EmbedCode`Common`createFullArgumentSpec[paramInfo, argTypes, interpreterTypeToGoogleDocsType];
        (* finalArgSpec looks like {{"name", "JavaScript type"}...}.  *)
        argsStr = StringJoin[Riffle[finalArgSpec[[All, 1]] , ", "]];
        payloadSection = StringJoin[createPayloadLine /@ finalArgSpec];
        (* If any of the arg slots are taking Expression types, add the definition of the JStoWL function. *)
        js2wlDef = If[MemberQ[finalArgSpec[[All, 2]], "Expression"], EmbedCode`JavaScript`Common`JStoWLFunction, ""];
        code = 
        TemplateApply[
            gaTemplate,
            Association[
                "funcName" -> funcName,
                "args" -> argsStr,
                "url" -> url,
                "js2wlDef" -> js2wlDef,
                "payloadSection" -> payloadSection,
                "output" -> gaResult[returnType]
            ]
        ];

        Association[{
            "EnvironmentName" -> "GoogleDocs",
            "CodeSection" -> Association["Content" -> code, 
                             (* Leave out Description field to indicate that there should be no description text (redundant when no lib files are present) *)
                             "Title" -> Automatic]
        }]
    ]
    

gaResult["Number" | "number" | "Integer"] = 
"return Number(_result.getContentText());\n"
gaResult["Boolean" | "boolean" | "bool"] =
"return Boolean(_result.getContentText());\n"
gaResult["Date"] =
"return new Date(_result.getContentText());\n"
gaResult["Array"] =
"var _text = _result.getContentText();
    return JSON.parse(_text.replace(/{/g, \"[\").replace(/}/g, \"]\").replace(/True/g, \"true\").replace(/False/g, \"false\"));\n"
gaResult["Blob"] =
"return _result.getBlob();\n" 
gaResult[_] =
"return _result.getContentText();\n"


$defaultFunctionName = "wolframCloudCall"


(* Mappings from types recognized by the Interpreter[] function (these are the native WL-side types for APIFunction calls)
  to JavaScript native types. At the moment, these are unused (except for Expression), since JavaScript doesn't have
  type info on arguments, and we have no run-time type checking in place. Expression is the one type that needs special
  handling when sending.
*)
interpreterTypeToGoogleDocsType["Integer" | "Number"] = "Number"
interpreterTypeToGoogleDocsType["Boolean"] = "Boolean"
interpreterTypeToGoogleDocsType["DateTime"] = "Date"
interpreterTypeToGoogleDocsType[DelimitedSequence[__]] = "DelimitedSequence"
interpreterTypeToGoogleDocsType["Expression"] = "Expression"
interpreterTypeToGoogleDocsType[_] = "String"


createPayloadLine[argSpec_] :=
    Module[{name, type},
        name = argSpec[[1]];
        type = argSpec[[2]];
        Switch[type,
            "Expression",
                "_payload[\"" <> name <> "\"] = _JStoWL(" <> name <> ");\n    ",
            "DelimitedSequence",
                "if (" <> name <> " instanceof Array) {\n    " <>
                "    _payload[\"" <> name <> "\"] = " <> name <> ".toString();\n    " <>
                "} else {\n    " <>
                "    _payload[\"" <> name <> "\"] = " <> name <> ";\n    " <>
                "}",
            _,
                "_payload[\"" <> name <> "\"] = " <> name <> ";\n    "
        ]
    ]


gaTemplate = 
"function `funcName`(`args`) {\n"                                         <>
"\n"                                                                      <>
"    var _url = \"`url`\";\n"                                             <>
"    `js2wlDef`\n"                                                        <>
"    var _payload = {};\n"                                                <>   
"    `payloadSection`\n"                                                  <>
"    var _options =\n"                                                    <>
"    {\n"                                                                 <>
"      \"method\" : \"post\",\n"                                          <>
"      \"payload\" : _payload,\n"                                         <>
"      \"headers\" : {\"User-Agent\" : \"EmbedCode-GoogleDocs/1.0\"}\n"   <>
"    };\n"                                                                <>
"\n"                                                                      <>
"    var _result = UrlFetchApp.fetch(_url, _options);\n"                  <>
"    `output`\n"                                                          <>
"}"


End[]
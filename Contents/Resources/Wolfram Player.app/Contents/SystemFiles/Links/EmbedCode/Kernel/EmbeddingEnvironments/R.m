(* Embedding for R language *)

Begin["EmbedCode`R`Private`"]

EmbedCode`Common`interpreterTypeToRType["Integer" | "Integer8" | "Integer16" | "Integer32" | Integer | "Integer64" | "Number"] = "numeric";
EmbedCode`Common`interpreterTypeToRType[_] = "character";

EmbedCode`Common`iEmbedCode["r", apiFunc_APIFunction, url_, opts:OptionsPattern[]] := 
	Module[
		{paramInfo, returnType, finalArgSpec, code,
		sig, argTypes,
		strArgs, strUrl, strArgsObj},
		sig = OptionValue[ExternalTypeSignature];
		If[sig === Automatic, sig = {Automatic, Automatic}];
		argTypes = sig[[1]];
		returnType = sig[[2]];
        (* paramInfo will be a list of triples: {{"name", type, default (symbol None if no default)}, ...} *)
        (* For now, the default values are being ignored *)
        paramInfo = EmbedCode`Common`getParamInfo[apiFunc];
        finalArgSpec = EmbedCode`Common`createFullArgumentSpec[paramInfo, argTypes, EmbedCode`R-Common`interpreterTypeToRType];
        (* finalArgSpec looks like {{"name", "R type"}...}.  *)
        
        strArgs = StringJoin[Riffle[finalArgSpec[[All, 1]], ", "]];
        (* strUrl = StringReplace[url, "https" -> "http"]; *)
        strUrl = url;
        strArgsObj = "list(" <> StringJoin[Riffle[( # <> " = " <> # &) /@ finalArgSpec[[All, 1]] , ", "]] <> ")";
                
        code = 
        TemplateApply[
        	StringJoin[
        		strHeader <> "\n\n" <>
        		strAuxiliarFunctions <> "\n\n" <>
	            strWolframCloudCallFunction <> "\n\n" <>
	            strFooter
	        ],
			Association[
				"args" -> strArgs,
			 	"url" -> strUrl,
			 	"argsObj" -> strArgsObj,
				"output" -> strResult[returnType]
			]
        ];

        Association[{
            "EnvironmentName" -> "R",
            "CodeSection" -> Association["Content" -> code, 
                             (* Leave out Description field to indicate that there should be no description text (redundant when no lib files are present) *)
                             "Title" -> Automatic]
        }]
    ];

strResult["numeric"] := "as.numeric(result)";
strResult["character"] := "substr(result, 2, nchar(result) - 1)";
strResult[Automatic] := "result";
strResult[_] := "result";

strHeader = 
"# R EmbedCode usage example
# wcc = WolframCloudCall$new()
# result = wcc$call(`args`)
 
library(RCurl)

WolframCloudCall <- setRefClass(
	\"WolframCloudCall\",
	fields = list(),
	methods = list(
		initialize = function() { },";

strAuxiliarFunctions = 
"		auxCall = function(url, args) {
			h <- basicTextGatherer()
			do.call(postForm, c(list(uri = url, .opts = curlOptions(writefunction = h$update, useragent=\"EmbedCode-R/1.0\")), args))
			h$value()
		},";
  
strWolframCloudCallFunction =
"		call = function(`args`) {
			url <- \"`url`\"
			args <- `argsObj`
			result <- auxCall(url, args)
			`output`
		}";
	
strFooter =
"	)
)";

End[];
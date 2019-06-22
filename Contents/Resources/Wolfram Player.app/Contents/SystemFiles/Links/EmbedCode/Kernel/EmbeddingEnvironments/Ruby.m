(* Embedding for Ruby language *)

Begin["EmbedCode`Ruby`Private`"]

EmbedCode`Common`interpreterTypeToRubyType["Integer" | "Integer8" | "Integer16" | "Integer32" | Integer | "Integer64" | "Number"] = "Fixnum";
EmbedCode`Common`interpreterTypeToRubyType["Number"] = "Float";
EmbedCode`Common`interpreterTypeToRubyType[_] = "String";

EmbedCode`Common`iEmbedCode["ruby", apiFunc_APIFunction, url_, opts:OptionsPattern[]] := 
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
        finalArgSpec = EmbedCode`Common`createFullArgumentSpec[paramInfo, argTypes, EmbedCode`Ruby-Common`interpreterTypeToRubyType];
        (* finalArgSpec looks like {{"name", "Ruby type"}...}.  *)
        
        strArgs = StringJoin[Riffle[finalArgSpec[[All, 1]], ", "]];
        (* strUrl = StringReplace[url, "https" -> "http"]; *)
        strUrl = url;
        strArgsObj = "{" <> StringJoin[Riffle[(":" <> # <> " => " <> # &) /@ finalArgSpec[[All, 1]] , ", "]] <> "}";
        
        code = 
        TemplateApply[
        	StringJoin[
        		strHeader <> "\n\n" <>
	            strWolframCloudCallFunction <> "\n\n" <>
	            strAuxiliarFunctions <> "\n\n" <>
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
            "EnvironmentName" -> "Ruby",
            "CodeSection" -> Association["Content" -> code, 
                             (* Leave out Description field to indicate that there should be no description text (redundant when no lib files are present) *)
                             "Title" -> Automatic]
        }]
    ];

strResult["Fixnum"] := "result.to_i()";
strResult["Float"] := "result.to_f()";
strResult["String"] := "result[1, result.length() - 2]";
strResult[Automatic] := "result";
strResult[_] := "result";

strHeader = 
"# Ruby EmbedCode usage:
# wcc = WolframCloudCall.new()
# result = wcc.call(`args`)
 
require \"net/https\"

class WolframCloudCall";

strAuxiliarFunctions = 
"	private
	
	def auxCall(url, args)
		uri = URI.parse(url)

		https = Net::HTTP.new(uri.host, uri.port)
		https.use_ssl = true
		https.verify_mode = OpenSSL::SSL::VERIFY_NONE

		request = Net::HTTP::Post.new(uri.path)
		request.add_field(\"User-Agent\", \"EmbedCode-Ruby/1.0\")
		request.add_field(\"Content-Type\", \"application/x-www-form-urlencoded; charset=utf-8\")
		request.set_form_data(args)

		response = https.request(request)

		if !response.is_a?(Net::HTTPOK)
			raise \"Response code: \" + response.code
		else
			return response.body
		end
	end";
  
strWolframCloudCallFunction =
"	def call(`args`)
		url = \"`url`\"
		args = `argsObj`
		result = auxCall(url, args)
		return `output`
	end";
	
strFooter = "end"

End[];
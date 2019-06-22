(* Embedding for Scala language *)

Begin["EmbedCode`Scala`Private`"]

interpreterTypeToScalaType["Integer" | "Integer8" | "Integer16" | "Integer32" | Integer | "Integer64"] = "Int";
interpreterTypeToScalaType["Number"] = "Double";
interpreterTypeToScalaType[_] = "String";

EmbedCode`Common`iEmbedCode["scala", apiFunc_APIFunction, url_, opts:OptionsPattern[]] := 
	Module[
		{paramInfo, returnType, finalArgSpec, code,
		sig, argTypes, strTypedArgs,
		strArgs, strUrl, strWriteBytes},
		sig = OptionValue[ExternalTypeSignature];
		If[sig === Automatic, sig = {Automatic, Automatic}];
		argTypes = sig[[1]];
		returnType = sig[[2]];
		(* If[returnType === Automatic, returnType = "String"]; *)
        (* paramInfo will be a list of triples: {{"name", type, default (symbol None if no default)}, ...} *)
        (* For now, the default values are being ignored *)
        paramInfo = EmbedCode`Common`getParamInfo[apiFunc];
        finalArgSpec = EmbedCode`Common`createFullArgumentSpec[paramInfo, argTypes, interpreterTypeToScalaType];
        (* finalArgSpec looks like {{"name", "Scala type"}...}.  *)
        
        strArgs = StringJoin[Riffle[finalArgSpec[[All, 1]], ", "]];
        (* strUrl = url; *)
        strUrl = StringReplace[url, "https" -> "http"];
        strTypedArgs = StringJoin[Riffle[(#[[1]] <> ": " <> #[[2]] &) /@ finalArgSpec, ", "]];
        
        (* arguments in String format *)
        strWriteBytes = StringJoin[
        	Riffle[
		        (TemplateApply[
		        	"\"`varName`=\" + URLEncoder.encode(\"\" + `varName`, \"UTF-8\") + \"&\"",
		        	Association["varName" -> #[[1]]]
		        ]&) /@ finalArgSpec,
		        " + "
        	]
        ];
        
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
				"typedArgs" -> strTypedArgs,
			 	"url" -> strUrl,
			 	"writeBytes" -> strWriteBytes,
			 	"returnType" -> returnType,
				"output" -> strResult[returnType]
			]
        ];

        Association[{
            "EnvironmentName" -> "Scala",
            "CodeSection" -> Association["Content" -> code, 
                             (* Leave out Description field to indicate that there should be no description text (redundant when no lib files are present) *)
                             "Title" -> Automatic]
        }]
    ];

strResult["Int"] := "result.toInt";
strResult["Float"] := "result.toFloat";
strResult["Double"] := "result.toDouble";
strResult["String"] := "result.substring(1, result.length() - 1)";
strResult[Automatic] := "result";
strResult[_] := "result";

strHeader = 
"// Scala EmbedCode usage:
// wcc = new WolframCloudCall
// result = wcc.call(`args`)
 
import java.net.URL
import java.net.HttpURLConnection
import java.net.URLEncoder
import java.io.InputStream
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.DataOutputStream
import java.io.IOException

class WolframCloudCall {";

strAuxiliarFunctions = 
"	private def auxCall(strUrl:String, strArgs:String): String = {
		var url:URL = new URL(strUrl)
		var conn:HttpURLConnection = url.openConnection.asInstanceOf[HttpURLConnection]
		conn.setRequestMethod(\"POST\")
		conn.setDoOutput(true)
		conn.setUseCaches(false)
		conn.setAllowUserInteraction(false)
		conn.setRequestProperty(\"User-Agent\", \"EmbedCode-Scala/1.0\")

		var out = new DataOutputStream(conn.getOutputStream)
		out.writeBytes(strArgs)
		out.close

		if (conn.getResponseCode != 200) {
			throw new IOException(conn.getResponseMessage)
		}

		var rdr = new BufferedReader(new InputStreamReader(conn.getInputStream))
		var sb = new StringBuilder

		var line = rdr.readLine
		while (line != null) {
			sb.append(line)
			line = rdr.readLine
		}
		rdr.close
		conn.disconnect

		return sb.toString
	}";
  
strWolframCloudCallFunction =
"	def call(`typedArgs`): `returnType` = {
		var url:String = \"`url`\"
		var args:String = `writeBytes`;
		var result:String = auxCall(url, args)
		return `output`
	}";
	
strFooter = "}";

End[];


(* Pick a unique private context for each implementation file. *)
Begin["EmbedCode`CSharp`Private`"]


EmbedCode`Common`iEmbedCode["c#", apiFunc_APIFunction, url_, opts:OptionsPattern[]] :=
    Module[{sig, argTypes, retType, paramInfo, returnType, finalArgSpec, argTypeString, code},
        sig = OptionValue[ExternalTypeSignature];
        If[sig === Automatic, sig = {Automatic, Automatic}];
        {argTypes, retType} = sig;
        (* paramInfo will be a list of triples: {{"name", type, default (symbol None if no default)}, ...} *)
        paramInfo = EmbedCode`Common`getParamInfo[apiFunc];
        returnType = If[retType === Automatic, "String", retType];
        finalArgSpec = EmbedCode`Common`createFullArgumentSpec[paramInfo, argTypes, interpreterTypeToCSharpType];
        (* finalArgSpec looks like {{"name", "csharp type"}...}.  *)
        argTypeString = StringJoin @@ Riffle[StringJoin @@@ (Riffle[#, " "]& /@ Reverse /@ finalArgSpec), ", "];
        addParams = StringJoin @@ (StringTemplate["postData +=  \"`paramName`\" + \"=\" + `paramName` + \"&\";"][<|"paramName" -> #|>]& /@ First /@ finalArgSpec);
        code = TemplateApply[template, <|"returnType" -> returnType, "argTypes" -> argTypeString, "addParams" -> addParams,
        	"url" -> StringReplace[url, "https" -> "http"], "retType" -> retType|>];

        Association[{
            "EnvironmentName" -> "C#",
            "CodeSection" -> <|"Content" -> code, 
                             (* Leave out Description field to indicate that there should be no description text (redundant when no lib files are present) *)
                             "Title" -> Automatic|>
        }]
    ]

interpreterTypeToCSharpType["Integer" | "Integer8" | "Integer16" | "Integer32" | Integer] = "int";
interpreterTypeToCSharpType["Integer64"] = "long";
interpreterTypeToCSharpType["Number"] = "double";
interpreterTypeToCSharpType[_] = "String";

template = StringTemplate[
"using System;
using System.Net;
using System.Text;
using System.IO;
using System.Collections.Specialized;

namespace RequestTest
{
	class MainClass
	{

		public static `returnType` call(`argTypes`)
		{
			using (var client = new WebClient())
			{
    			client.Headers.Add(\"Content-Type\",\"application/x-www-form-urlencoded\");
				client.Headers.Add(\"User-Agent\",\"EmbedCode-CSharp/1.0\");
				
				string postData=\"\";	
				`addParams`
				
				byte[] byteArray = Encoding.ASCII.GetBytes(postData);
				string url = \"`url`\";
				var response = client.UploadData(url, byteArray);

   				var responseString = Encoding.Default.GetString(response);
				client.Dispose();
				
<* result[If[retType === Automatic, \"string\", #retType]] *>
			}
		}
	}
}
"]

result["int"] =
"        		return Int32.Parse(responseString);
"

result["double"] =
"        		return Double.Parse(responseString);
"

result[Automatic] =
"        		return responseString;
"

result["String"] =
"        		return responseString.Substring(1,responseString.Length - 2);
"

result["string"] =
"        		return responseString.Substring(1,responseString.Length - 2);
"


(* EmbedCode for Data Drop *)

EmbedCode`Common`iEmbedCode["c#", databin_Databin, url_, opts:OptionsPattern[]] :=
	Module[
		{code},
		code = 
        TemplateApply[
        	templateDataDrop,
			Association[
				"binId" -> databin[[1]]
			]
        ];
        
		Association[{
            "EnvironmentName" -> "C#",
            "CodeSection" -> Association["Content" -> code, 
                             (* Leave out Description field to indicate that there should be no description text (redundant when no lib files are present) *)
                             "Description" -> "This code uses the Json.NET library (http://www.newtonsoft.com/json). Adding a reference to System.Web is also needed.",
                             "Title" -> Automatic]
        }]
	];

templateDataDrop =
"/*
Usage:

WolframDatabin databin = new WolframDatabin(\"`binId`\");

dynamic result = databin.Recent();
Console.WriteLine(result);

Dictionary<string, string> data = new Dictionary<string, string>() {
    {\"x\", \"x_value\"},
    {\"y\", \"y_value\"}
};
result = databin.Add(data);
Console.WriteLine(result);
*/

using System;
using System.Net;
using System.Text;
using System.IO;
using System.Web;
using System.Collections.Specialized;
using System.Collections.Generic;
using Newtonsoft.Json;

class WolframDatabin
{
    private static string baseUrl = \"http://datadrop.wolframcloud.com/api/v1.0\";
    private string databinId;

    private static string UrlFetch(string url, string data)
    {
        WebClient client = new WebClient();
        client.Headers.Add(\"Content-Type\", \"application/x-www-form-urlencoded\");
        client.Headers.Add(\"User-Agent\", \"EmbedCode-CSharp/1.0\");
        client.Headers.Add(\"From\", \"FromHeaderValue\");

        byte[] byteArray = Encoding.ASCII.GetBytes(data);
        byte[] response = client.UploadData(url, byteArray);
        string responseString = Encoding.Default.GetString(response);

        client.Dispose();
        return responseString;
    }

    private static string EncodeArgs(Dictionary<string, string> args)
    {
        string encodedArgs = \"\";
        foreach(KeyValuePair<string, string> entry in args) {
            if (encodedArgs != \"\") encodedArgs += \"&\";
            string key = entry.Key;
            string value = entry.Value;
            key = HttpUtility.UrlEncode(key);
            value = HttpUtility.UrlEncode(value);
            encodedArgs += key + \"=\" + value;
        }
        return encodedArgs;
    }

    public dynamic Recent()
    {
        Dictionary<string, string> args = new Dictionary<string, string> {
            {\"Bin\", databinId},
            {\"_exportform\", \"JSON\"}
        };
        string responseString = UrlFetch(baseUrl + \"/Recent\", EncodeArgs(args));
        return JsonConvert.DeserializeObject(responseString);
    }

    public dynamic Add(Dictionary<string, string> data)
    {
        Dictionary<string, string> args = new Dictionary<string, string> {
            {\"Bin\", databinId},
            {\"Data\", EncodeArgs(data)},
            {\"_exportform\", \"JSON\"}
        };
        string responseString = UrlFetch(baseUrl + \"/Add\", EncodeArgs(args));
        return JsonConvert.DeserializeObject(responseString);
    }

    public WolframDatabin(string databinId) {
        this.databinId = databinId;
    }
}";

End[]


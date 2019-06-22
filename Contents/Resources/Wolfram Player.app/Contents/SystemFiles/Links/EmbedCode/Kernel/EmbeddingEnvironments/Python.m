Begin["EmbedCode`Python`Private`"]

EmbedCode`Common`iEmbedCode["python", apiFunc_APIFunction, url_, opts:OptionsPattern[]] :=
    Module[{sig, argTypes, retType, returnType, paramNames, paramList, paramDict, code},
        sig = OptionValue[ExternalTypeSignature];
        If[sig === Automatic, sig = {Automatic, Automatic}];
        {argTypes, retType} = sig;
        paramInfo = EmbedCode`Common`getParamInfo[apiFunc];
        returnType = retType;
        finalArgSpec = EmbedCode`Common`createFullArgumentSpec[paramInfo, argTypes, Null&];
        paramNames = First /@ finalArgSpec;
        paramList = StringJoin[Riffle[paramNames, ", "]];
        paramDict = StringJoin[Riffle[# <> "=" <> # & /@ paramNames, ", "]];
        code = StringJoin[StringTemplate[pythonCode][<|"url" -> StringReplace[url, "https" -> "http"], "paramList" -> paramList, "paramDict" -> paramDict|>],
            pythonCodeResult[returnType]];
        Association[{
            "EnvironmentName" -> "Python",
            "CodeSection" -> <|"Content" -> code, "Title" -> Automatic, "Filename" -> "WolframCloud.py"|>
        }]
    ]

pythonCode =
"from urllib import urlencode
from urllib2 import urlopen

class WolframCloud:

    def wolfram_cloud_call(self, **args):
        arguments = dict([(key, arg) for key, arg in args.iteritems()])
        result = urlopen(\"`url`\", urlencode(arguments))
        return result.read()

    def call(self, `paramList`):
        textresult =  self.wolfram_cloud_call(`paramDict`)
        ";

pythonCodeResult["int"]=
        "return int(textresult)";

pythonCodeResult["long"]=
        "return long(textresult)";

pythonCodeResult["float"]=
        "return float(textresult)";

pythonCodeResult["str"]=
        "return str(textresult)";

pythonCodeResult[Automatic]=
        "return textresult";


(* EmbedCode for Data Drop *)

EmbedCode`Common`iEmbedCode["python", databin_Databin, url_, opts:OptionsPattern[]] :=
	Module[
		{code},
		code = 
        TemplateApply[
        	pythonDataDropCode,
			Association[
				"binId" -> databin[[1]]
			]
        ];
        
		Association[{
            "EnvironmentName" -> "JavaScript",
            "CodeSection" -> Association["Content" -> code, 
                             (* Leave out Description field to indicate that there should be no description text (redundant when no lib files are present) *)
                             (* "Description" -> "x", *)
                             "Title" -> Automatic]
        }]
	];

pythonDataDropCode = 
"# Usage:
# databin = WolframDatabin(\"`binId`\")
# result = databin.recent()
# print result
# result = databin.add(x = 1, y = 2)
# print result

from urllib import urlencode
from urllib2 import urlopen
import json

class WolframDatabin:
    base_url = \"http://datadrop.wolframcloud.com/api/v1.0\"
    
    def call(self, url, **args):
        arguments = dict([(key, arg) for key, arg in args.iteritems()])
        result = urlopen(url, urlencode(arguments))
        return result.read()

    def recent(self):
        result =  self.call(WolframDatabin.base_url + \"/Recent\", Bin = self.databinId, _exportform = \"JSON\")
        return json.loads(result)
    
    def add(self, **args):
        call_args = {\"Bin\": self.databinId, \"_exportform\": \"JSON\"}
        for key, val in args.iteritems():
            call_args[key] = val
        result = self.call(WolframDatabin.base_url + \"/Add\", **call_args)
        return json.loads(result)
    
    def __init__(self, databinId):
        self.databinId = databinId
";

End[]
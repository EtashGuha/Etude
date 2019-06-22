(* Embedding for PHP language using no extra libs. *)

(* Pick a unique private context for each implementation file. *)
Begin["EmbedCode`PHP`Private`"]

EmbedCode`Common`iEmbedCode["php", apiFunc_APIFunction, url_, opts:OptionsPattern[]] :=
    Module[{sig, argTypes, retType, paramInfo, returnType, finalArgSpec, argTypeString, code},
    	sig = OptionValue[ExternalTypeSignature];
    	 If[sig === Automatic, sig = {Automatic, Automatic}];
        {argTypes, retType} = sig;
        (* paramInfo will be a list of triples: {{"name", type, default (symbol None if no default)}, ...} *)
        paramInfo = EmbedCode`Common`getParamInfo[apiFunc];
        returnType = If[retType === Automatic, "String", retType];     
        finalArgSpec = EmbedCode`Common`createFullArgumentSpec[paramInfo, argTypes, EmbedCode`PHPCommon`interpreterTypeToPHPType];
        (* finalArgSpec looks like {{"name", "Php type"}...}.  *)
        countParameters = (First/@finalArgSpec)//Length;
        phpParameters = StringJoin[Riffle[{"'" <> # <> "='.$" <> #} & /@ First /@ finalArgSpec,".'&'."]];
        code = StringTemplate["<?php
class httpPost {
	public $url = '`url`';
	public function WolframCloudCall(`functionParameters`){`body`
		$opts = array(
				'http'=> array(
					'header' => \"Content-Type: application/x-www-form-urlencoded\\r\\n\".
								\"User-Agent:EmbedCode-PHP/1.0\\r\\n\",
					'method' => 'POST'`httpParameters`));
		$context = stream_context_create($opts);
		$page = file_get_contents($this->url, false, $context);
		$result = (`phpType`)$page;
		<* If[#show,\"return substr($result,1,-1);\",\"return $result;\"]*>
	}	
}
?>"]
			[<|"url" -> StringReplace[url, "https" -> "http"],
			   "functionParameters" -> 
			   		If[countParameters>1,
			   			StringJoin[Riffle["$" <> # & /@ First /@ finalArgSpec, {","}]],
			   			StringJoin["$" <> # & /@ First /@ finalArgSpec]
			   		],
			   "httpParameters"->If[countParameters >= 1,",'content' =>"<>phpParameters,""],
			   "body"->If[countParameters >= 1,"\n\t$body ="<>phpParameters<>";",""],
			   "phpType"->
			   		If[MatchQ[returnType,"Integer"|"Float"|"String"],returnType,String],
			   "show"->If[retType === "String", True, False]
			      |>];
        Association[{
            "EnvironmentName" -> "PHP",
            "CodeSection" -> <|"Content" -> code, 
                             (* Leave out Description field to indicate that there should be no description text (redundant when no lib files are present) *)
                             "Title" -> Automatic|>
        }]
    ]
End[]
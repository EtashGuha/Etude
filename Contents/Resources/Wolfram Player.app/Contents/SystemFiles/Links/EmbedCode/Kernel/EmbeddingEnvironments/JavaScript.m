(* Embedding for JavaScript language for web-client using cross-compatible asynchronous request XMLHttpRequest / XDomainRequest *)

Begin["EmbedCode`JavaScript`Private`"]

EmbedCode`Common`iEmbedCode["javascript", apiFunc_APIFunction, url_, opts:OptionsPattern[]] := 
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
        finalArgSpec = EmbedCode`Common`createFullArgumentSpec[paramInfo, argTypes, EmbedCode`JavaScript-Common`interpreterTypeToJavaScriptType];
        (* finalArgSpec looks like {{"name", "JavaScript type"}...}.  *)
        
        If[Length[finalArgSpec] =!= 0,
        	strArgs = StringJoin[Riffle[finalArgSpec[[All, 1]] , ", "]] <> ", ",
        	strArgs = ""];
        strUrl = StringReplace[url, "https" -> "http"];
        strArgsObj = "{" <> StringJoin[Riffle[(# <> ": " <> # &) /@ finalArgSpec[[All, 1]] , ", "]] <> "}";
        
        code = 
        TemplateApply[
        	StringJoin[
        		jsHeader <> "\n\n" <>
	            jsAuxiliarFunctions <> "\n\n" <>
	            jsWolframCloudCallFunction <> "\n" <>
	            jsFooter
	        ],
			Association[
				"args" -> strArgs,
			 	"url" -> strUrl,
			 	"argsObj" -> strArgsObj,
				"output" -> jsResult[returnType]
			]
        ];

        Association[{
            "EnvironmentName" -> "JavaScript",
            "CodeSection" -> Association["Content" -> code, 
                             (* Leave out Description field to indicate that there should be no description text (redundant when no lib files are present) *)
                             "Title" -> Automatic]
        }]
    ];

jsResult["Number"] := "Number(result)";
jsResult["String"] := "result.substr(1, result.length - 2)";
jsResult[Automatic] := "result";
jsResult[_] := "result";

jsHeader = 
"/* 
JavaScript EmbedCode usage:

var wcc = new WolframCloudCall();
wcc.call(`args`function(result) { console.log(result); });
*/
 
var WolframCloudCall;

(function() {
WolframCloudCall = function() {	this.init(); };

var p = WolframCloudCall.prototype;

p.init = function() {};";

jsAuxiliarFunctions = 
"p._createCORSRequest = function(method, url) {
	var xhr = new XMLHttpRequest();
	if (\"withCredentials\" in xhr) {
		xhr.open(method, url, true);
	} else if (typeof XDomainRequest != \"undefined\") {
		xhr = new XDomainRequest();
		xhr.open(method, url);
	} else {
		xhr = null;
	}
	return xhr;
};

p._encodeArgs = function(args) {
	var argName;
	var params = \"\";
	for (argName in args) {
		params += (params == \"\" ? \"\" : \"&\");
		params += encodeURIComponent(argName) + \"=\" + encodeURIComponent(args[argName]);
	}
	return params;
};

p._auxCall = function(url, args, callback) {
	var params = this._encodeArgs(args);
	var xhr = this._createCORSRequest(\"post\", url);
	if (xhr) {
		xhr.setRequestHeader(\"Content-Type\", \"application/x-www-form-urlencoded\");
		xhr.setRequestHeader(\"EmbedCode-User-Agent\", \"EmbedCode-JavaScript/1.0\");
		xhr.onload = function() {
			if ((xhr.status >= 200 && xhr.status < 300) || xhr.status == 304) {
				callback(xhr.responseText);
			} else {
				callback(null);
			}
		};
		xhr.send(params);
	} else {
		throw new Error(\"Could not create request object.\");
	}
};";
  
jsWolframCloudCallFunction =
"p.call = function(`args`callback) {
	var url = \"`url`\";
	var args = `argsObj`;
	var callbackWrapper = function(result) {
		if (result === null) callback(null);
		else callback(`output`);
	};
	this._auxCall(url, args, callbackWrapper);
};";

jsFooter = "})();"


(* EmbedCode for Data Drop *)

EmbedCode`Common`iEmbedCode["javascript", databin_Databin, url_, opts:OptionsPattern[]] :=
	Module[
		{code},
		code = 
        TemplateApply[
        	StringJoin[
        		jsDataDropHeader <> "\n\n" <>
	            jsAuxiliarFunctions <> "\n\n" <>
	            jsDataDropRecentFunction <> "\n\n" <>
	            jsDataDropAddFunction <> "\n\n" <>
	            jsFooter
	        ],
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


jsDataDropHeader =
"/*
Usage:
var databinId = \"`binId`\";
var databin = new WolframDatabin(databinId);
databin.recent(function(result) { console.log(result); });
databin.add({x: 1, y: 2}, function(result) { console.log(result); });
*/

var WolframDatabin;

(function() {
WolframDatabin = function(databinId) {
	this.init(databinId);
};

var p = WolframDatabin.prototype;

WolframDatabin.baseUrl = \"http://datadrop.wolframcloud.com/api/v1.0\";

p.init = function(databinId) {
	this.databinId = databinId;
};";

jsDataDropAddFunction =
"p.add = function(data, callback) {
	var args = {\"Bin\": this.databinId, \"_exportform\": \"JSON\"};
	for (var i in data) args[i] = data[i];
	var callbackWrapper = function(result) {
		if (result === null) callback(null);
		else callback(JSON.parse(result));
	};
	this._auxCall(WolframDatabin.baseUrl + \"/Add\", args, callbackWrapper);
};";

jsDataDropRecentFunction = 
"p.recent = function(callback) {
	var args = {\"Bin\": this.databinId, \"_exportform\": \"JSON\"};
	var callbackWrapper = function(result) {
		if (result === null) callback(null);
		else callback(JSON.parse(result));
	};
	this._auxCall(WolframDatabin.baseUrl + \"/Recent\", args, callbackWrapper);
};";


End[];
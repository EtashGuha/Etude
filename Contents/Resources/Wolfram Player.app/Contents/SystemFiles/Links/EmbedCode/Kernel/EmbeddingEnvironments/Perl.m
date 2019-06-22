(* Embedding for Perl language *)

Begin["EmbedCode`Perl`Private`"]

EmbedCode`Common`interpreterTypeToPerlType["Integer" | "Integer8" | "Integer16" | "Integer32" | Integer | "Integer64" | "Number"] = "Number";
EmbedCode`Common`interpreterTypeToPerlType[_] = "String";

EmbedCode`Common`iEmbedCode["perl", apiFunc_APIFunction, url_, opts:OptionsPattern[]] := 
	Module[
		{paramInfo, returnType, finalArgSpec, code,
		sig, argTypes,
		strArgs, strUrl, strArgsObj, strArgsWithLeadingComma},
		sig = OptionValue[ExternalTypeSignature];
		If[sig === Automatic, sig = {Automatic, Automatic}];
		argTypes = sig[[1]];
		returnType = sig[[2]];
        (* paramInfo will be a list of triples: {{"name", type, default (symbol None if no default)}, ...} *)
        (* For now, the default values are being ignored *)
        paramInfo = EmbedCode`Common`getParamInfo[apiFunc];
        finalArgSpec = EmbedCode`Common`createFullArgumentSpec[paramInfo, argTypes, EmbedCode`Perl-Common`interpreterTypeToPerlType];
        (* finalArgSpec looks like {{"name", "Perl type"}...}.  *)
        
        strArgs = StringJoin[Riffle[("$" <> #&) /@ finalArgSpec[[All, 1]], ", "]];
        strArgsWithLeadingComma = If[Length[finalArgSpec] === 0, "", ", " <> strArgs];
        (* strUrl = StringReplace[url, "https" -> "http"]; *)
        strUrl = url;
        strArgsObj = "[" <> StringJoin[Riffle[(# <> " => $" <> # &) /@ finalArgSpec[[All, 1]] , ", "]] <> "]";
        
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
			 	"argsWithLeadingComma" -> strArgsWithLeadingComma,
				"output" -> strResult[returnType]
			]
        ];

        Association[{
            "EnvironmentName" -> "Perl",
            "CodeSection" -> Association["Content" -> code, 
                             (* Leave out Description field to indicate that there should be no description text (redundant when no lib files are present) *)
                             "Title" -> Automatic]
        }]
    ];

strResult["Number"] := "$result";
strResult["String"] := "substr($result, 1, length($result) - 2)";
strResult[Automatic] := "$result";
strResult[_] := "$result";

strHeader = 
"# Perl EmbedCode usage:
# my $wcc = WolframCloudCall->new();
# my $result = $wcc->call(`args`);
 
package WolframCloudCall;

use HTTP::Request::Common;
use LWP::UserAgent;

sub new {
	my $class = @_[0];
	return bless { }, $class;
}";

strAuxiliarFunctions = 
"sub auxCall {
	my ($url, $args) = @_;
	my $ua = LWP::UserAgent->new();
	$ua->agent('EmbedCode-Perl/1.0');
	my $request = POST($url, $args);
	my $result = $ua->request($request)->decoded_content();
	return $result;
}";
  
strWolframCloudCallFunction =
"sub call {
	my ($self`argsWithLeadingComma`) = @_;
	my $url = \"`url`\";
	my $args = `argsObj`;
	my $result = auxCall($url, $args);
	return `output`;
}";
	
strFooter = "";

End[];
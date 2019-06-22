(* Embedding for Java language using the Jersey toolkit for RESTful web services.

     http://jersey.java.net/

*)

(* Pick a unique private context for each implementation file. *)
Begin["EmbedCode`JavaJersey`Private`"]


EmbedCode`Common`iEmbedCode["java-jersey", apiFunc_APIFunction, url_, opts:OptionsPattern[]] :=
    Module[{sig, argTypes, retType, paramInfo, returnType, finalArgSpec, argTypeString, code},
        sig = OptionValue[ExternalTypeSignature];
        If[sig === Automatic, sig = {Automatic, Automatic}];
        {argTypes, retType} = sig;
        (* paramInfo will be a list of triples: {{"name", type, default (symbol None if no default)}, ...} *)
        paramInfo = EmbedCode`Common`getParamInfo[apiFunc];
        returnType = If[retType === Automatic, "byte[]", retType];
        finalArgSpec = EmbedCode`Common`createFullArgumentSpec[paramInfo, argTypes, EmbedCode`JavaCommon`interpreterTypeToJavaType];
        (* finalArgSpec looks like {{"name", "java type"}...}. *)
        argTypeString = StringJoin @@ Riffle[StringJoin @@@ (Riffle[#, " "]& /@ Reverse /@ finalArgSpec), ", "];
        code = StringJoin[
            StringReplace[javaHeader, {"${returnType}" -> returnType, "${argTypes}" -> argTypeString, "${url}" -> StringReplace[url, "https" -> "http"]}],
            StringJoin @@ (StringReplace[javaAddParams, {"${paramName}" -> #}]& /@ First /@ finalArgSpec),
            StringReplace[javaInvocation, "${resultMimeType}" -> "MediaType.TEXT_PLAIN_TYPE"],
            javaResult[If[retType === Automatic, "String", retType]],
            javaFooter
        ];

        Association[{
            "EnvironmentName" -> "Java, using the Jersey toolkit",
            "CodeSection" -> <|"Content" -> code, "Description" -> Automatic, "Title" -> Automatic, "Filename" -> "WolframCloudCall.java"|>,
            "FilesSection" -> <|
                "FileList" -> {
                    "jersey-client.jar" -> "http://www.wolframcloud.com/res/FCI/Java/Jersey/jersey-client.jar",
                    "jersey-common.jar" -> "http://www.wolframcloud.com/res/FCI/Java/Jersey/jersey-common.jar",
                    "javax.ws.rs-api-2.0.jar" -> "http://www.wolframcloud.com/res/FCI/Java/Jersey/javax.ws.rs-api-2.0.jar"
                },
                "ZipDownloadURL" -> "http://www.wolframcloud.com/res/FCI/Java/Jersey/JerseyLibs.zip",
                "TarDownloadURL" -> "http://www.wolframcloud.com/res/FCI/Java/Jersey/JerseyLibs.tar",
                "Description" -> Automatic,
                "Title" -> Automatic
            |>
        }]
    ]



javaHeader =
"import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.client.Invocation;
import javax.ws.rs.client.WebTarget;
import javax.ws.rs.core.Form;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import org.glassfish.jersey.client.*;

public class WolframCloudCall {

    public static ${returnType} call(${argTypes}) {

        ClientConfig _clientConfig = new ClientConfig();
        Client _client = ClientBuilder.newClient(_clientConfig);
        WebTarget _webTarget = _client.target(\"${url}\");
"

javaAddParams =
"        _webTarget = _webTarget.queryParam(\"${paramName}\", ${paramName});
"

javaInvocation =
"        Invocation.Builder _invocationBuilder = _webTarget.request(${resultMimeType}).header(\"User-Agent\", \"EmbedCode-Java-Jersey/1.0\");
        Response _response = _invocationBuilder.get();
"

javaResult["byte"] =
"
        return _response.readEntity(byte.class);
"

javaResult["short"] =
"
        return _response.readEntity(short.class);
"

javaResult["int"] =
"
        return _response.readEntity(int.class);
"

javaResult["long"] =
"
        return _response.readEntity(long.class);
"

javaResult["char"] =
"
        return _response.readEntity(char.class);
"

javaResult["float"] =
"
        return _response.readEntity(float.class);
"

javaResult["double"] =
"
        return _response.readEntity(double.class);
"

javaResult["boolean"] =
"        return \"True\".equals(_response.readEntity(String.class));
"

javaResult["Date"] = javaResult["java.util.Date"] =
"
        return _response.readEntity(java.util.Date.class);
"

javaResult["String"] =
"
        return _response.readEntity(String.class);
"

(* Without a fallthrough, EmbedCode fails with terrible StringJoin messages. *)
javaResult[_] = javaResult["String"]

javaFooter =
"    }
}
"

End[]
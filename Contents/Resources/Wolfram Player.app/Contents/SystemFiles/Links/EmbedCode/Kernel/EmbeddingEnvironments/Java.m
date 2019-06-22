(* Embedding for Java language using no extra libs. *)

(* Pick a unique private context for each implementation file. *)
Begin["EmbedCode`Java`Private`"]


EmbedCode`Common`iEmbedCode["java", apiFunc_APIFunction, url_, opts:OptionsPattern[]] :=
    Module[{sig, funcName, argTypes, retType, paramInfo, returnType, finalArgSpec, argTypeString, code, needsMultipart},
        {sig, funcName} = OptionValue[{ExternalTypeSignature, ExternalFunctionName}];
        If[sig === Automatic, sig = {Automatic, Automatic}];
        If[!StringQ[funcName], funcName = EmbedCode`JavaCommon`$defaultFunctionName];
        {argTypes, retType} = sig;
        (* paramInfo will be a list of triples: {{"name", type, default (symbol None if no default)}, ...} *)
        paramInfo = EmbedCode`Common`getParamInfo[apiFunc];
        returnType = If[retType === Automatic, "String", retType];
        finalArgSpec = EmbedCode`Common`createFullArgumentSpec[paramInfo, argTypes, EmbedCode`JavaCommon`interpreterTypeToJavaType];
        (* finalArgSpec looks like {({"name", "java type"} | {"name", "java type", delimitedSequenceSeparatorSpec})...}.  *)
        argTypeString = StringJoin @@ Riffle[StringJoin @@@ (Riffle[#, " "]& /@ Reverse /@ finalArgSpec[[All, 1;;2]]), ", "];
        (* If any args are binary data, we will use multipart request; otherwise POST with data form-urlencoded in the body. *)
        needsMultipart = Or @@ (EmbedCode`JavaCommon`isBinaryType /@ finalArgSpec[[All, 2]]);
        code = header[url, argTypeString, returnType, funcName,
                       If[needsMultipart, "multipart/form-data; boundary=" <> EmbedCode`Common`$multipartBoundary, "application/x-www-form-urlencoded; charset=utf-8"]];
        (code = code <> addParam[#, needsMultipart])& /@ finalArgSpec;
        If[needsMultipart,
            code = code <> StringTemplate["        _out.writeBytes(\"--`boundary`--\\r\\n\");\n\n"] [<|"boundary" -> EmbedCode`Common`$multipartBoundary|>]
        ];
        code = code <>
            Which[
                returnType == "byte[]",
                    invocationBinary,
                EmbedCode`JavaCommon`isImage[returnType],
                    invocationImage,
                True,
                    invocationString
            ];
        code = code <> javaResult[If[retType === Automatic, "String", retType]];
        code = code <> footer;
        
        Association[{
            "EnvironmentName" -> "Java",
            "CodeSection" -> <|"Content" -> code, 
                             (* Leave out Description field to indicate that there should be no description text (redundant when no lib files are present) *)
                             "Title" -> Automatic,
                             "Filename" -> "WolframCloudCall.java"|>
        }]
    ]

(* Fallthrough def for "java" embedding of a non-APIFunction argument. We want to give a slightly better message than the generic EmbedCode::noembed. 
   Note that EmbedCode will be protected at the time this file is loaded, so we cannot define the message directly for EmbedCode.
*)
General::noembedj = "EmbedCode for Java only works on objects of type APIFunction. The given object, `1` at `2`, is not of this type."

EmbedCode`Common`iEmbedCode["java", expr_, uri_, OptionsPattern[]] := (Message[EmbedCode::noembedj, expr, uri]; $Failed)



header[url_String, argSequence_String, returnType_String, funcName_String, contentType_String] :=
    TemplateApply[
        StringTemplate[
            "import java.net.URL;\n"                                                           <>
            "import java.net.HttpURLConnection;\n"                                             <>
            "import java.net.URLEncoder;\n"                                                    <>
            "import java.io.InputStream;\n"                                                    <>
            "import java.io.BufferedReader;\n"                                                 <>
            "import java.io.InputStreamReader;\n"                                              <>
            "import java.io.DataOutputStream;\n"                                               <>
            "import java.io.IOException;\n"                                                    <>
            "\n"                                                                               <>
            "public class WolframCloudCall {\n"                                                <>
            "\n"                                                                               <>
            "    public static `returnType` `funcName`(`argSequence`) throws `exceptions` {\n" <>
            "\n"                                                                               <>
            "        URL _url = new URL(\"`url`\");\n"                                         <>
            "        HttpURLConnection _conn = (HttpURLConnection) _url.openConnection();\n"   <>
            "        _conn.setRequestMethod(\"POST\");\n"                                      <>
            "        _conn.setDoOutput(true);\n"                                               <>
            "        _conn.setDoInput(true);\n"                                                <>
            "        _conn.setUseCaches(false);\n"                                             <>
            "        _conn.setAllowUserInteraction(false);\n"                                  <>
            "        _conn.setRequestProperty(\"Content-Type\", \"`contentType`\");\n"         <>
            "        _conn.setRequestProperty(\"User-Agent\", \"EmbedCode-Java/1.0\");\n"      <>
            "        DataOutputStream _out = new DataOutputStream(_conn.getOutputStream());\n" <>
            "\n"                                                                             
        ],
        <|"url" -> url, "argSequence" -> argSequence, "returnType" -> returnType, "funcName" -> funcName, "contentType" -> contentType,
            "exceptions" -> If[returnType === "java.util.Date", "IOException, java.text.ParseException", "IOException"]|>
    ]


footer =
"    }
}
"

addParam[{paramName_String, paramJavaType_String, delimiterSpec:_:None}, useMultipart:(True|False)] :=
    Module[{multipartHeader, multipartData, multipartEnd, contentType, delimStart, delimMiddle, delimEnd},
        Switch[delimiterSpec,
            _String,
                {delimStart, delimMiddle, delimEnd} = {"", delimiterSpec, ""},
            {_String, _String, _String},
                {delimStart, delimMiddle, delimEnd} = delimiterSpec,
            _,
                {delimStart, delimMiddle, delimEnd} = {"", "", ""}
        ];
        If[useMultipart,
            contentType = If[EmbedCode`JavaCommon`isBinaryType[paramJavaType], "application/octet-stream", "text/plain; charset=utf-8"];
            multipartHeader =
                StringTemplate[
                    "        _out.writeBytes(\"--`boundary`\\r\\n\");\n"                                              <>
                    "        _out.writeBytes(\"Content-Disposition: form-data; name=\\\"`paramName`\\\"\\r\\n\");\n"  <>
                    "        _out.writeBytes(\"Content-Type: `contentType`\\r\\n\\r\\n\");\n"
                ] [<|"paramName" -> paramName, "contentType" -> contentType, "boundary" -> EmbedCode`Common`$multipartBoundary|>];
            multipartData =
                Which[
                    EmbedCode`JavaCommon`isImage[paramJavaType],
                        StringTemplate[
                            "        javax.imageio.ImageIO.write(`paramName`, \"png\", _out);\n"
                        ],
                    EmbedCode`JavaCommon`is1DArrayType[paramJavaType] && delimiterSpec =!= None,
                        (* 1D arrays sent to DelimitedSequence slots. The test for delimiterSpec =!= None is to avoid catching cases
                           where it is raw binary data sent as a byte[], which should go into the next branch.
                        *)
                        StringTemplate[
                            If[StringLength[delimStart] > 0,
                                "        _out.write(\"`delimStart`\".getBytes(\"UTF-8\"));\n",
                            (* else *)
                                ""
                            ]                                                                               <>
                            "        if (`paramName`.length > 0) {\n"                                       <>
                            "            byte[] _delim = \"`delimMiddle`\".getBytes(\"UTF-8\");\n"                    <>
                            "            _out.write((\"\" + `paramName`[0]).getBytes(\"UTF-8\"));\n"        <>
                            "            for (int _i = 1; _i < `paramName`.length; _i++) {\n"               <>
                            "                _out.write(_delim);\n"                                    <>
                            "                _out.write((\"\" + `paramName`[_i]).getBytes(\"UTF-8\"));\n"   <>
                            "            }\n"                                                               <>
                            "        }\n"                                                                   <>
                            If[StringLength[delimEnd] > 0,
                                "        _out.write(\"`delimEnd`\".getBytes(\"UTF-8\"));\n",
                            (* else *)
                                ""
                            ] 
                        ],
                    EmbedCode`JavaCommon`isBinaryType[paramJavaType],
                        (* This will only work if param is a byte[]. *)
                        StringTemplate[
                            "        _out.write(`paramName`);\n"
                        ],
                    True,
                        StringTemplate[
                            "        _out.write((\"\" + `paramName`).getBytes(\"UTF-8\"));\n"
                        ]
                ] [<|"paramName" -> paramName, "delimStart" -> delimStart, "delimMiddle" -> delimMiddle, "delimEnd" -> delimEnd|>];
            multipartEnd = "        _out.writeBytes(\"\\r\\n\");\n";
            multipartHeader <> multipartData <> multipartEnd <> "\n",
        (* else *)
            (* Not multipart; use POST body *)
            StringTemplate[
                "        _out.writeBytes(\"`paramName`\");\n"                                   <>
                "        _out.writeBytes(\"=\");\n"                                             <>
                If[EmbedCode`JavaCommon`is1DArrayType[paramJavaType],
                    If[StringLength[delimStart] > 0,
                        "        _out.write(\"`delimStart`\".getBytes(\"UTF-8\"));\n",
                    (* else *)
                        ""
                    ]                                                                               <>
                    "        if (`paramName`.length > 0) {\n"                                       <>
                    "            byte[] _delim = \"`delimMiddle`\".getBytes(\"UTF-8\");\n"                    <>
                    "            _out.write((\"\" + `paramName`[0]).getBytes(\"UTF-8\"));\n"        <>
                    "            for (int _i = 1; _i < `paramName`.length; _i++) {\n"               <>
                    "                _out.write(_delim);\n"                                    <>
                    "                _out.write((\"\" + `paramName`[_i]).getBytes(\"UTF-8\"));\n"   <>
                    "            }\n"                                                               <>
                    "        }\n"                                                                   <>
                    If[StringLength[delimEnd] > 0,
                        "        _out.write(\"`delimEnd`\".getBytes(\"UTF-8\"));\n",
                    (* else *)
                        ""
                    ],
                (* else *)
                "        _out.writeBytes(URLEncoder.encode(\"\" + `paramName`, \"UTF-8\"));\n" 
                ]                                                                               <>
                "        _out.writeBytes(\"&\");\n"
            ] [<|"paramName" -> paramName, "delimStart" -> delimStart, "delimMiddle" -> delimMiddle, "delimEnd" -> delimEnd|>]
        ]
    ]
    

invocationString =
"
        _out.close();

        if (_conn.getResponseCode() != 200) {
            throw new IOException(_conn.getResponseMessage());
        }
        
        BufferedReader _rdr = new BufferedReader(new InputStreamReader(_conn.getInputStream()));
        StringBuilder _sb = new StringBuilder();
        String _line;
        while ((_line = _rdr.readLine()) != null) {
            _sb.append(_line);
        }
        _rdr.close();
        _conn.disconnect();
"

invocationBinary =
"
        _out.close();

        if (_conn.getResponseCode() != 200) {
            throw new IOException(_conn.getResponseMessage());
        }
        
        InputStream _strm = _conn.getInputStream();
        int _len = _conn.getContentLength();
        byte[] _buf = new byte[_len];
        int _bytesRead = 0;
        while (_bytesRead < _len) {
            int _num = _strm.read(_buf);
            if (_num >= 0)
                _bytesRead += _num;
            else
                break;
        }
        _strm.close();
        _conn.disconnect();
"

invocationImage =
"
        _out.close();

        if (_conn.getResponseCode() != 200) {
            throw new IOException(_conn.getResponseMessage());
        }
        
        InputStream _strm = _conn.getInputStream();
        java.awt.image.BufferedImage _im = javax.imageio.ImageIO.read(_strm);
        _strm.close();
        _conn.disconnect();
"

javaResult["byte"] =
"        return Byte.valueOf(_sb.toString()).byteValue();
"

javaResult["short"] =
"        return Short.valueOf(_sb.toString()).shortValue();
"

javaResult["int"] =
"        return Integer.valueOf(_sb.toString()).intValue();
"

javaResult["long"] =
"        return Long.valueOf(_sb.toString()).longValue();
"

javaResult["char"] =
"        return _sb.toString().charAt(0);
"

javaResult["float"] =
"        return Float.valueOf(_sb.toString()).floatValue();
"

javaResult["double"] =
"        return Double.valueOf(_sb.toString()).doubleValue();
"

javaResult["boolean"] =
"        return \"True\".equals(_sb.toString());
"

javaResult["String"] =
"        return _sb.toString();
"

(* Java's date/time parsing is weak. You have to create the date/time in a very specific format for this parse to succeed.
   Use getDateTimeInstance() to get parsing of date _and_ time. Stupidly, however, that fails if no time spec is present.
 *)
javaResult["Date"] = javaResult["java.util.Date"] =
"        java.text.DateFormat _df = java.text.DateFormat.getDateInstance();
        return _df.parse(_sb.toString());
"

javaResult["Image"] = javaResult["java.awt.Image"] = javaResult["BufferedImage"] = javaResult["java.awt.image.BufferedImage"] =
"        return _im;
"

(***************************  Array results  ****************************)

(* Ugly. byte[] is special and means the raw stream of bytes in the result. *)
javaResult["byte[]"] =
"        return _buf;
"

javaResult["char[]"] =
"        return _buf.toString().toCharArray();
"

javaResult[type:("short[]" | "int[]" | "long[]" | "float[]" | "double[]")] :=
    Module[{arrayType, parseFunction},
        arrayType = StringDrop[type, -2];
        parseFunction = Switch[type,
            "short[]",
                "Short.parseShort",
            "int[]",
                "Integer.parseInt",
            "long[]",
                "Long.parseLong",
            "float[]",
                "Float.parseFloat",
            "double[]",
                "Double.parseDouble"
        ];
        StringTemplate[
"        String _withoutBraces = _sb.substring(1, _sb.length() - 1);
        String[] _elements = _withoutBraces.split(\", \");
        `arrayType`[] _result = new `arrayType`[_elements.length];
        int _i = 0;
        try {
            for ( ; _i < _elements.length; _i++) {
                _result[_i] = `parseFunction`(_elements[_i]);
            }
        } catch (NumberFormatException _e) {
            throw new IOException(\"Error parsing a Java `arrayType` from the string \" + _elements[_i], _e);
        }
        return _result;
"
        ][<|"arrayType" -> arrayType, "parseFunction" -> parseFunction|>]
    ]


(* Any types not handled specifically above will be returned as strings. *)
javaResult[_] =
"        return _sb.toString();
"

End[]

(* EmbedCode for Data Drop *)
EmbedCode`Common`iEmbedCode["java", databin_Databin, url_, opts:OptionsPattern[]] :=
	Module[
		{code},
		code = 
        TemplateApply[
        	StringJoin[
        		javaDataDropUsage <> "\n\n" <>
        		javaDataDropImport <> "\n\n" <>
        		javaClassHeader <> "\n\n" <>
	            javaDataDropRecentFunction <> "\n\n" <>
	            javaDataDropAddFunction <> "\n\n" <>
	            javaAuxiliarFunctions  <> "\n\n" <>
	            javaFooter
	        ],
			Association[
				"binId" -> databin[[1]]
			]
        ];
        
		Association[{
            "EnvironmentName" -> "java",
            "CodeSection" -> Association["Content" -> code, 
                             (* Leave out Description field to indicate that there should be no description text (redundant when no lib files are present) *)
                              "Description" -> "Requirement: \nPut JSON.simple jar in your CLASSPATH", 
                             "Title" -> Automatic]
        }]
	];

javaDataDropUsage= 
"/*
Usage:

WolframDataDrop datadrop = new WolframDataDrop(\"`binId`\");

// Add
HashMap<String,String> newData = new HashMap<String,String>();
newData.put(\"x\",5);
HashMap<String,Object> addConfirmation = datadrop.addData(newData);
System.out.println(\"Message: \" + addConfirmation.get(\"Message\"));      
System.out.println(\"Data: \" + addConfirmation.get(\"Data\"));

// Recent
ArrayList< HashMap<String,Object> > res = datadrop.getRecent();
for(int i=0; i<res.size(); i++){
         HashMap<String,Object> hash = res.get(i);
         Iterator iterator = hash.keySet().iterator();
         for(;iterator.hasNext();){
            String key   = (String)iterator.next();
            String value = hash.get(key).toString();
            System.out.println(key+\" \"+value);
         }
      }
*/
"

javaDataDropImport = "
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;

import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLDecoder;
import java.net.URLEncoder;

import org.json.simple.JSONObject;
import org.json.simple.JSONArray;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
"

javaClassHeader = "
public class WolframDataDrop{

   String idDataBin;
   String baseURL = \"https://datadrop.wolframcloud.com/api/v1.0/\";
   
   public WolframDataDrop(){
      idDataBin = null;
   }

   public WolframDataDrop(String _idBin){
      idDataBin = _idBin;
   }
"

javaDataDropRecentFunction = "
public ArrayList< HashMap<String,Object> > getRecent(){
     if(idDataBin==null) return null;
     ArrayList< HashMap<String,Object> > arrayMap = null;
     String getUrl = baseURL + \"Recent?bin=\" + idDataBin + \"&_exportform=JSON\";
     JSONParser parser = new JSONParser();
     try{
        String s = sendGet(getUrl);
        JSONArray jsonarray = (JSONArray)parser.parse(s);
        arrayMap = new ArrayList< HashMap<String,Object> >();
        for(int i=0; i<jsonarray.size(); i++){
           HashMap<String,Object> hash = new HashMap<String,Object>();
           JSONObject jsonObj = (JSONObject)jsonarray.get(i);
           Iterator iterator = jsonObj.keySet().iterator();
           for(;iterator.hasNext();){
              String key   = (String)iterator.next();
              Object value = jsonObj.get(key);
              hash.put(key,value);
           }
           arrayMap.add(hash);
        }
     }catch(Exception e){
     }
     return arrayMap;
   }
"

javaDataDropAddFunction = "
   public HashMap<String,Object> addData(HashMap<String,String> newData){
     if(idDataBin==null) return null;

     HashMap<String,Object> res = null;
     String encodedData = \"\";
     Iterator iterator;

     iterator = newData.keySet().iterator();
     boolean first = false;
     for(;iterator.hasNext();){
        String key   = (String)iterator.next();
        String value = (String)newData.get(key);
        if(first){
           first = false;
        }else{
           encodedData += \"&\";
        }
        encodedData += key + \"=\" + value;
     }

     try{
        encodedData = URLEncoder.encode(encodedData, \"UTF-8\");
     }catch(Exception e){
     }

     String addUrl = baseURL + \"Add?bin=\" + idDataBin + \"&Data=\" + encodedData + \"&_exportform=JSON\";

     JSONParser parser = new JSONParser();
     try{
        String response = sendGet(addUrl);
        JSONObject jsonResponse = (JSONObject)parser.parse(response);
        res = new HashMap<String,Object>();
        iterator = jsonResponse.keySet().iterator();
        for(;iterator.hasNext();){
           String key   = (String)iterator.next();
           Object value = jsonResponse.get(key);
           res.put(key,value);
        }
     }catch(Exception e){
     }

     return res;
   }
"

javaAuxiliarFunctions = "
private String sendGet(String url) throws Exception{
      URL obj = new URL(url);
         HttpURLConnection con = (HttpURLConnection) obj.openConnection();
      con.setRequestMethod(\"GET\");
      int responseCode = con.getResponseCode();

      BufferedReader in = new BufferedReader(
                 new InputStreamReader(con.getInputStream()));
         String inputLine;
         StringBuffer response = new StringBuffer();

         while ((inputLine = in.readLine()) != null) {
            response.append(inputLine);
         }
         in.close();
      return response.toString();
   }
"

javaFooter = "}
"

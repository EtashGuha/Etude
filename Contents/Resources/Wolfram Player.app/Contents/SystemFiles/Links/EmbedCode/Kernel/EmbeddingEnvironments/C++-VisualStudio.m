(* ::Package:: *)

(* Embedding for C++ Visual Studio language using no extra libs. *)

(* Pick a unique private context for each implementation file. *)
Begin["EmbedCode`CppVisualStudio`Private`"]


EmbedCode`Common`iEmbedCode["c++-visualstudio", apiFunc_APIFunction, url_, opts:OptionsPattern[]] :=
    Module[{paramInfo, returnType, finalArgSpec, paramTypeList, code, paramList, returnTypeParser, sig, funcName, argTypes, retType, paramListString},
        (* paramInfo will be a list of triples: {{"name", type, default (symbol None if no default)}, ...} *)
        {sig, funcName} = OptionValue[{ExternalTypeSignature, ExternalFunctionName}];
        If[sig === Automatic, sig = {Automatic, Automatic}];
        If[!StringQ[funcName], funcName = "call"];
        {argTypes, retType} = sig;
        (* paramInfo will be a list of triples: {{"name", type, default (symbol None if no default)}, ...} *)
        paramInfo = EmbedCode`Common`getParamInfo[apiFunc];
        returnType = If[retType === Automatic, "String^", retType]/."String"->"String^";
        
        finalArgSpec = EmbedCode`Common`createFullArgumentSpec[paramInfo, argTypes, interpreterTypeToCppVisualStudioType]/."String"->"String^";
        (* finalArgSpec looks like {{"name", "c++ type"}...}.  *)
        paramTypeList = StringJoin @@ Riffle[StringJoin @@@ (Riffle[#, " "]& /@ Reverse /@ finalArgSpec), ", "];
        paramList = First /@ finalArgSpec;
        returnTypeParser = If[retType === Automatic, "httpresult", If[returnType==="String^", "httpresult->Substring(1, httpresult->Length-2)", returnType<>"::Parse(httpresult)"]];
        paramListString = If[#[[2]]==="String^", #[[1]], #[[1]]<>".ToString()"]& /@ finalArgSpec;
        code = visualStudioPostTemplate[<|
                 "returnType" -> returnType,
                 "paramTypeList" -> paramTypeList,
                 "url" -> url,
                 "paramList" -> paramList,
                 "paramListString" -> paramListString,
                 "returnTypeParser" -> returnTypeParser
            |>];

        Association[{
            "EnvironmentName" -> "C++ using the .NET Framework",
            "CodeSection" -> <|"Content" -> code, 
                             (* Leave out Description field to indicate that there should be no description text (redundant when no lib files are present) *)
                             "Title" -> Automatic,
                             "Filename" -> "WolframCloudCall.cpp"|>
        }]
    ]


visualStudioPostTemplate = StringTemplate@"// Visual C++ WolframCloudCall using .NET 3.5

using namespace System;
using namespace System::Net;
using namespace System::IO;
using namespace System::Text;
using namespace System::Web;

public ref class WolframCloudCall
{
public:
\tstatic `returnType` call(`paramTypeList`) {
\t\tString^ url = \"`url`\";
\t\tString^ data;
\t\tString^ httpresult;
\t\t`returnType` result;
\t\tdata = \"<*#paramList[[1]]*>=\"+HttpUtility::UrlEncode(<*#paramListString[[1]]*>);
<*StringJoin[Table[{\"\t\tdata += \\\"&\",#paramList[[i]],\"=\\\"+HttpUtility::UrlEncode(\",#paramListString[[i]],\");\n\"}, {i,2,Length[#paramListString]}]]*>
\t\thttpresult = httpPostRequest(url,data);
\t\tresult = `returnTypeParser`;
\t\treturn result;
\t};

private:
\tstatic String^ httpPostRequest(String^ postUrl, String^ postData) {
        HttpWebRequest^ httpRequest = nullptr;
        HttpWebResponse^ httpResponse = nullptr;
        Stream^ httpPostStream = nullptr;
        BinaryReader^ httpResponseStream = nullptr;
\t\t//BinaryWriter^ result = nullptr;
        StringWriter^ result = nullptr;

        try
        {
            array<Byte>^ postBytes = nullptr;
 
            // Create HTTP web request
            httpRequest = (HttpWebRequest^)WebRequest::Create(postUrl);
            // Change method to \"POST\"
            httpRequest->Method = \"POST\";
\t\t\t// UserAgent to get accepted by WolframCloud API
\t\t\thttpRequest->UserAgent = \"EmbedCode-C++-VisualStudio/1.0\";
            // Posted forms need to be encoded so change the content type
            httpRequest->ContentType = \"application/x-www-form-urlencoded\";
            // Retrieve a byte array representation of the data
            postBytes = Encoding::UTF8->GetBytes(postData->ToString());
            // Set the content length (the number of bytes in the POST request)
            httpRequest->ContentLength = postBytes->Length;
            // Retrieve the request stream so we can write the POST data
            httpPostStream = httpRequest->GetRequestStream();
            // Write the POST request
            httpPostStream->Write(postBytes, 0, postBytes->Length);
            httpPostStream->Close();
            httpPostStream = nullptr;
            // Retrieve the response
            httpResponse = (HttpWebResponse^)httpRequest->GetResponse();
            // Retrieve the response stream
            httpResponseStream = gcnew BinaryReader(httpResponse->GetResponseStream(), Encoding::UTF8);
 
            array<Char>^ readData;
            
\t\t\tresult = gcnew StringWriter;
 
            while (true)
            {
                readData = httpResponseStream->ReadChars(4096);
                if (readData->Length == 0)
                    break;
                result->Write(readData, 0, readData->Length);
            }
        }
        catch (WebException^ wex)
        {
            Console::WriteLine(\"HttpMethodPost() - Exception occurred: {0}\", wex->Message);
            httpResponse = (HttpWebResponse^)wex->Response;
        }
        finally
        {
            // Close any remaining resources
            if (httpResponse != nullptr)
            {
                httpResponse->Close();
            }
        }
\t\treturn result->ToString();
\t};

};
";


interpreterTypeToCppVisualStudioType["Integer" | "Integer8" | "Integer16" | "Integer32" | Integer] = "int"
interpreterTypeToCppVisualStudioType["Integer64"] = "long"
interpreterTypeToCppVisualStudioType["Number"] = "double"
interpreterTypeToCppVisualStudioType[_] = "String^"


End[]


(* EmbedCode for Data Drop *)
EmbedCode`Common`iEmbedCode["c++-visualstudio", databin_Databin, url_, opts:OptionsPattern[]] :=
	Module[
		{code},
		code = 
        TemplateApply[
        	StringJoin[
        		cppvisualDataDropUsage <> "\n\n" <>
        		cppvisualDataDropImport <> "\n\n" <>
        		cppvisualDataDropNamespaces <> "\n\n" <>
        		cppvisualClassHeader <> "\n\n" <>
        		cppvisualGlobalVariablesAndConstructors <> "\n\n" <>
	            cppvisualDataDropRecentFunction <> "\n\n" <>
	            cppvisualDataDropAddFunction <> "\n\n" <>
	            cppvisualAuxiliarFunctions  <> "\n\n" <>
	            cppvisualFooter
	            
	        ],
			Association[
				"binId" -> databin[[1]]
			]
        ];
        
		Association[{
            "EnvironmentName" -> "c++-visualstudio",
            "CodeSection" -> Association["Content" -> code, 
                             (* Leave out Description field to indicate that there should be no description text (redundant when no lib files are present) *)
                              "Description" -> "Requirement:\n - Include RapidJSON library on your project's path.\n - Add a reference to System.Web.\n", 
                             "Title" -> Automatic]
        }]
	];
	
cppvisualDataDropUsage = "
/*
Usage:

WolframDatabin databin(\"`binId`\");

// Add
rapidjson::Document addResponse;
std::map<std::string, std::string> newData;
newData[\"x\"] = \"10\";
newData[\"y\"] = \"20\";
addResponse = databin.addData(newData);
std::cout << addResponse[\"Message\"].GetString() << std::endl;

// Recent
rapidjson::Document recentResponse;
recentResponse = databin.getRecent();
for (rapidjson::SizeType i = 0; i < recentResponse.Size(); i++){
		const rapidjson::Value& aDoc = recentResponse[i];
		const rapidjson::Value& aData = aDoc[\"Data\"];
		printf(\"Data of recentResponse[%d]:\n\", i);
	
		for (rapidjson::Value::ConstMemberIterator itr = aData.MemberBegin() ;
			itr != aData.MemberEnd(); ++itr)
		{
			printf(\" Member %s has value %s\n\",
				itr->name.GetString(), itr->value.GetString());
		}
}
*/
"

cppvisualDataDropImport = "
#include <vector>
#include <map>

#include <msclr/marshal.h>
#include <msclr/marshal_cppstd.h>


#include \"rapidjson/document.h\"
#include \"rapidjson/writer.h\"
"

cppvisualDataDropNamespaces = "
using namespace System;
using namespace System::Net;
using namespace System::IO;
using namespace System::Text;
using namespace System::Web;
"

cppvisualClassHeader = "
public ref class WolframDatabin
{
"

cppvisualGlobalVariablesAndConstructors =  "
String^ idDatabin;
String^ baseURL = \"https://datadrop.wolframcloud.com/api/v1.0/\";

WolframDatabin(){
	idDatabin = nullptr;
}

WolframDatabin(String^ _idBin){
	idDatabin = _idBin;
}
"


cppvisualDataDropRecentFunction = "
rapidjson::Document getRecent(){
rapidjson::Document res = nullptr;

String^ getResponse = nullptr;
String^ getUrl = baseURL + \"Recent?bin=\" + idDatabin + \"&_exportform=JSON\";

getResponse = httpGetRequest(getUrl);

rapidjson::Document d;
std::string getResponseString = msclr::interop::marshal_as<std::string>(getResponse);
const char *getResponsejson = getResponseString.c_str();
res.Parse(getResponsejson);

return res;
}
"
cppvisualDataDropAddFunction = "
rapidjson::Document addData(std::map<std::string, std::string> newData){
	bool first = true;
	std::string data = \"\";
	rapidjson::Document res = nullptr;
	if (idDatabin == nullptr){
		return res;
	}

	std::map<std::string, std::string>::iterator iterator;
	for (iterator = newData.begin(); iterator != newData.end(); iterator++){
		std::string key = iterator->first;
		std::string value = iterator->second;
		if (first){
			first = false;
		}
		else{
			data += \"&\";
		}
		data += key + \"=\" + value;
	}
	String^ encodedData = HttpUtility::UrlEncode(gcnew String(data.c_str()));

	String^ addUrl = baseURL + \"Add?bin=\" + idDatabin + \"&Data=\" + encodedData + \"&_exportform=JSON\";
	String^ addResponse = httpGetRequest(addUrl);
	std::string addResponseString = msclr::interop::marshal_as<std::string>(addResponse);
	const char *addResponsejson = addResponseString.c_str();
	res.Parse(addResponsejson);
	return res;
}
"

cppvisualAuxiliarFunctions = "
private:
	String^ httpGetRequest(String^ getUrl){
		HttpWebRequest^ httpRequest = nullptr;
		HttpWebResponse^ httpResponse = nullptr;
		BinaryReader^ httpResponseStream = nullptr;
		StringWriter^ result = nullptr;

		try{
			array<Byte>^ getBytes = nullptr;

			// Create HTTP web request
			httpRequest = (HttpWebRequest^)WebRequest::Create(getUrl);
			// Change method to \"Get\"
			httpRequest->Method = \"GET\";
			// UserAgent to get accepted by WolframCloud API	
			httpRequest->UserAgent = \"EmbedCode-C++-VisualStudio/1.0\";
			// Posted forms need to be encoded so change the content type
			httpRequest->ContentType = \"application/x-www-form-urlencoded\";

			// Retrieve the response
			httpResponse = (HttpWebResponse^)httpRequest->GetResponse();

			// Retrieve the response stream
			httpResponseStream = gcnew BinaryReader(httpResponse->GetResponseStream(), Encoding::UTF8);
			array<Char>^ readData;
			result = gcnew StringWriter;
			while (true)
			{
				readData = httpResponseStream->ReadChars(4096);
				if (readData->Length == 0)
					break;
				result->Write(readData, 0, readData->Length);
			}
		}
		catch (WebException^ wex){
			Console::WriteLine(\"HttpMethodPost() - Exception occurred: {0}\", wex->Message);
			httpResponse = (HttpWebResponse^)wex->Response;
		}
		finally{
			// Close any remaining resources
			if (httpResponse != nullptr)
			{
				httpResponse->Close();
			}
		}
		return result->ToString();
	}
"

cppvisualFooter = "
};
"

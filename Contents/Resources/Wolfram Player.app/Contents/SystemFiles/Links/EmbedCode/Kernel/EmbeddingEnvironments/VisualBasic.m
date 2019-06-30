Begin["EmbedCode`VisualBasic`Private`"]

EmbedCode`Common`iEmbedCode["visualbasic", apiFunc_APIFunction, url_, opts:OptionsPattern[]] :=
    Module[{sig, argTypes, retType, paramInfo, returnType, finalArgSpec, argTypeString, code},
        sig = OptionValue[ExternalTypeSignature];
        If[sig === Automatic, sig = {Automatic, Automatic}];
        {argTypes, retType} = sig;
        paramInfo = EmbedCode`Common`getParamInfo[apiFunc];
        returnType = If[retType === Automatic, "String", retType];
        finalArgSpec = EmbedCode`Common`createFullArgumentSpec[paramInfo, argTypes, interpreterTypeToVisualBasicType];
        argTypeString = StringJoin @@ Riffle[StringJoin @@@ (Riffle[Insert[Insert[#, "ByVal", 1],"As", 3]," "]& /@ finalArgSpec), ", "];
        paramNames = First /@ finalArgSpec;
        paramDict = StringJoin[Riffle["\"" <> # <> "=\" + CStr(" <> # <>")" & /@ paramNames, " + \"&\" + "]];
        code = StringJoin[StringTemplate[VBCode]
            [<|"argTypeString"->argTypeString,"returnType"->returnType,"paramDict"->paramDict,"url"->StringReplace[url, "https" -> "http"]|>]
            ,VBResult[returnType],VBEnd];
        Association[{
            "EnvironmentName" -> "Visual Basic",
            "CodeSection" -> <|"Content" -> code, "Title" -> Automatic|>
        }]
    ]

interpreterTypeToVisualBasicType["Int" | "Integer8" | "Integer16" | "Integer32" | Integer | "Integer"] = "Integer";
interpreterTypeToVisualBasicType["Integer64"|"Long"] = "Long";
interpreterTypeToVisualBasicType["Float32"|"Single"] = "Single";
interpreterTypeToVisualBasicType["Float"|"Float64"|"Double"] = "Double";
interpreterTypeToVisualBasicType[_] = "String";

VBCode =
"Imports System.Net
Imports System.IO
Imports System.Text

Public Class Wolfram_Cloud

    Public Function Wolfram_Cloud_Call(`argTypeString`) As `returnType`

        Dim data As String = `paramDict`
        Dim encoding As New UTF8Encoding
        Dim byteData As Byte() = encoding.GetBytes(data)

        Dim request As HttpWebRequest = DirectCast(WebRequest.Create(\"`url`\"), HttpWebRequest)
        request.Method = \"POST\"
        request.ContentType = \"application/x-www-form-urlencoded\"
        request.Referer = \"`url`\"
        request.UserAgent = \"Mozilla/5.0 (Windows NT 6.1; WOW64; rv:26.0 Gecko/20100101 Firefox/26.0\"
        request.KeepAlive = True
        request.ContentLength = byteData.Length

        Dim requestStream As Stream = request.GetRequestStream()
        requestStream.Write(byteData, 0, byteData.Length)
        requestStream.Close()

        Dim postResponse As HttpWebResponse
        postResponse = DirectCast(request.GetResponse(), HttpWebResponse)
        Dim requestReader As New StreamReader(postResponse.GetResponseStream())

";

VBResult["Integer"] =
"        Dim reader As String = requestReader.ReadToEnd
        Dim result As Integer = CInt(reader)"

VBResult["Long"] =
"        Dim reader As String = requestReader.ReadToEnd
        Dim result As Long = CLng(reader)"

VBResult["Single"] =
"        Dim reader As String = requestReader.ReadToEnd
        Dim result As Single = CSng(reader)"

VBResult["Double"] =
"        Dim reader As String = requestReader.ReadToEnd
        Dim result As Double = CDbl(reader)"

VBResult["String"] =
"        Dim result As String = requestReader.ReadToEnd"

VBResult[Automatic] =
"        Dim result As String = requestReader.ReadToEnd"

VBEnd =
"       
        Return result

    End Function

End Class"

End[]

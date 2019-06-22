(* :Author: Chris Williamson *)

Needs[ "JLink`" ]


invoke[endPoint_String, envelope_String, options___?OptionQ] :=
  Module[{method, timeout, proxyHost, proxyPort, statusCode, response, transportURI,
          httpClient, hostConfig, soapActionURI, locationHeader, redirectLocation,
          opts = canonicalOptions[Flatten[{options}]]},
  
    If[TrueQ[$PrintServiceRequest], Print[envelope]];

    (* Process options *)
    {transportURI, soapActionURI} = 
      {"TransportStyleURI", "SOAPActionURI"} /. 
         opts  /. Options[InvokeServiceOperation];

    (* Transport URI *)
    If[!StringMatchQ[transportURI, "http://schemas.xmlsoap.org/soap/http"], 
      Message[InvokeServiceOperation::transport, transportURI];
      Return[$Failed];
    ];    

    (* End Point *)
    method = JavaNew["org.apache.commons.httpclient.methods.PostMethod", endPoint];
    If[method === $Failed, Return[$Failed]];
    
    timeout = "Timeout" /. opts  /. Options[InvokeServiceOperation];
    Switch[timeout,
        Automatic,
            Null,  (* do nothing; use PM default *)
        None, 
            method@getParams[]@setSoTimeout[0],
        _Integer,
            method@getParams[]@setSoTimeout[timeout],
        _,
            Message[InvokeServiceOperation::opttype, timeout, "Timeout"];
            Return[$Failed]
    ];
    method@setDoAuthentication[True];

    If[StringQ[soapActionURI], 
      method@setRequestHeader["SOAPAction", ToString[soapActionURI, InputForm]],
      Message[InvokeServiceOperation::opttype, soapActionURI, "SOAPActionURI"];       
    ];

    (* initialize http client *)
    httpClient = getHttpClient[method, options];
    
    (* configure proxy. *)
    {proxyHost, proxyPort} = getProxyHostAndPort[endPoint];
    proxyPort = ToExpression[proxyPort];
    hostConfig = Null;
    If[proxyHost =!= Null && proxyPort =!= 0,
        LoadJavaClass["org.apache.commons.httpclient.HostConfiguration"];
        hostConfig = JavaNew["org.apache.commons.httpclient.HostConfiguration",
                            HostConfiguration`ANYUHOSTUCONFIGURATION];
        hostConfig@setProxy[proxyHost, proxyPort]
    ];
        
    (* Set content *)
    method@setRequestHeader["Content-Type", "text/xml;charset=UTF-8"];
    method@setRequestBody[envelope];
    
    (* Invoke method *)
    If[TrueQ[$PrintPerformanceNumbers], Print["Invoking: ",Date[]]];
    statusCode =
        If[hostConfig =!= Null,
            (* Execute with proxy. *)
            httpClient@executeMethod[hostConfig, method],
        (* else *)
            (* No proxy. *)
            httpClient@executeMethod[method]
        ];
    If[TrueQ[$PrintPerformanceNumbers], Print["Finished Invoking: ",Date[]]];
    
    (* Process response *)
    Switch[statusCode, 
      x_/; x >= 200 && x < 300, 
        response = method@getResponseBodyAsString[],
      x_/; x >= 300 && x < 400, 
        locationHeader = method@getResponseHeader["location"];
        If[locationHeader =!= Null,
          redirectLocation = locationHeader@getValue[];
          If[MemberQ[$redirects, redirectLocation], 
            Message[InvokeServiceOperation::circularredirect, redirectLocation];
            Return[$Failed]
            ,
            AppendTo[$redirects, endPoint];
            Return[invoke[redirectLocation, envelope, options]]
          ];
          ,
          Message[InvokeServiceOperation::redirectlocation, endPoint];
          Return[$Failed]
        ],        
      x_/; x >= 500 && x < 600, 
        Message[InvokeServiceOperation::httpcode, statusCode, method@getStatusLine[]@toString[]];
        response = method@getResponseBodyAsString[],
      $Failed,
        Return[$Failed],
      _, 
        Message[InvokeServiceOperation::httpcode, statusCode, method@getStatusLine[]@toString[]];
        Return[$Failed];
    ];
    If[response =!= $Failed && TrueQ[$PrintServiceResponse], Print[response]];
    
    (* Clean up method *)
    method@releaseConnection[];
            
    (* convert response *)
    If[response =!= $Failed && response =!= Null, 
      response = XML`ToVerboseXML[XML`Parser`XMLGetString[response]],
      Message[InvokeServiceOperation::rspns];
      Return[$Failed];
    ];    
    response
  ]

getHttpClient[options___?OptionQ] := getHttpClient[Null, options]

getHttpClient[method_, options___?OptionQ] := 
  Module[{username, password, timeout, creds,
          opts = canonicalOptions[Flatten[{options}]]},
  
    (* Process options *)
    {username, password, timeout} = 
      {"Username", "Password", "Timeout"} /. 
         opts  /. Options[InvokeServiceOperation];
    (* Setup the client *)
    
    If[!JavaObjectQ[httpClient],
        InstallJava[];
        LoadJavaClass["java.lang.System"];
        java`lang`System`setProperty["java.net.useSystemProxies", "true"];
        httpClient = JavaNew["org.apache.commons.httpclient.HttpClient",
                             JavaNew["org.apache.commons.httpclient.MultiThreadedHttpConnectionManager"]
                     ]
    ];
    
    If[method =!= Null && username =!= None && password =!= None, 
      Which[
        StringQ[username] && StringQ[password],
          LoadJavaClass["org.apache.commons.httpclient.auth.AuthScope"];
          creds = JavaNew["org.apache.commons.httpclient.UsernamePasswordCredentials", username, password];
          httpClient@getState[]@setCredentials[
            JavaNew["org.apache.commons.httpclient.auth.AuthScope", 
              method@getURI[]@getHost[], 
              method@getURI[]@getPort[], 
              AuthScope`ANYUREALM], 
            creds],
        StringQ[username],
          Message[InvokeServiceOperation::opttype, password, "Password"];
          Return[$Failed],
        StringQ[password],
          Message[InvokeServiceOperation::opttype, username, "Username"];
          Return[$Failed];
      ]
    ];
        
    httpClient
  ]
  
  
getProxyHostAndPort[endPoint_String] :=
    Module[{useProxy, httpProxy, httpsProxy, proxySelector, uri, result, proxyList},
        {useProxy, httpProxy, httpsProxy} = {"UseProxy", "HTTP", "HTTPS"} /. PacletManager`$InternetProxyRules;
        Switch[useProxy,
            Automatic,
                JavaBlock[
                    LoadJavaClass["java.net.ProxySelector"];
                    proxySelector = ProxySelector`getDefault[];
                    uri = JavaNew["java.net.URI", endPoint];
                    proxyList = JavaObjectToExpression[proxySelector@select[uri]];
                    result = Scan[
                        If[#@address[] =!= Null, Return[{#@address[]@getHostName[], #@address[]@getPort[]}]]&,
                        proxyList
                    ]
                ],
            True,
                If[StringMatchQ[endPoint, "https:*", IgnoreCase->True],
                    result = httpsProxy,
                (* else *)
                    result = httpProxy
                ]
        ];
        If[MatchQ[result, {_String, _Integer}],
            result,
        (* else *)
            {Null, 0}
        ]
    ]


FromServiceResponse[
  response:XMLObject["Document"][{___}, envelope_XMLElement, ___],
  options___?OptionQ] :=
  Module[{body, return, 
          returnType, resp, opts = canonicalOptions[Flatten[{options}]], 
          envelopeNamespaces, envelopeDefaultNamespace, 
          bodyNamespaces, bodyDefaultNamespace, 
          returnNamespaces, returnDefaultNamespace, 
          fault, faultstring = "", references},
    
    {returnType} = {"ReturnType"} /. opts  /. Options[FromServiceResponse]; 
            
    (* Process SOAP Envelope *)
    If[!MatchQ[
         envelope, 
         XMLElement[
           ("Envelope" | {"http://schemas.xmlsoap.org/soap/envelope/", "Envelope"}), 
           {___}, 
           {__XMLElement}]],
      Message[InvokeServiceOperation::rspnsenv, envelope];
      Return[$Failed]
    ];
    envelopeNamespaces = mapNamespaces[envelope[[2]], {}];
    envelopeDefaultNamespace = getDefaultNamespace[envelope[[2]], ""];
    
    
    (* Process SOAP Body *)
    body = First[envelope[[3]]];
    If[!MatchQ[
         body, 
         XMLElement[
           ("Body" | {"http://schemas.xmlsoap.org/soap/envelope/", "Body"}), 
           {___}, 
           {__XMLElement}]],
      Message[InvokeServiceOperation::rspnsbdy, body];
      Return[$Failed]
    ];
    bodyNamespaces = mapNamespaces[body[[2]], envelopeNamespaces];
    bodyDefaultNamespace = getDefaultNamespace[body[[2]], envelopeDefaultNamespace];

    (* Process return *)
    return = First[body[[3]]];
    If[!MatchQ[
         body, 
         XMLElement[(_String | {_String, _String}), {___}, {___XMLElement}]],
      Message[InvokeServiceOperation::rspnsrtn, 
              ExportString[body[[3]], "XML"]];
      Return[$Failed]
    ];
    returnNamespaces = mapNamespaces[return[[2]], bodyNamespaces];
    returnDefaultNamespace = getDefaultNamespace[return[[2]], bodyDefaultNamespace];
    
    (* Process SOAPFault *)
    If[MatchQ[
        return, 
        XMLElement[
          { "http://schemas.xmlsoap.org/soap/envelope/", "Fault"}, 
          {___}, 
          {__XMLElement}]], 
      fault = return;
      faultstring = 
        Cases[fault[[3]], XMLElement[{"", "faultstring"}, {___}, {str_String}] :> str];
      If[Length[faultstring] > 0, 
        faultstring = faultstring[[1]],
        faultstring = ""
      ];
      Message[InvokeServiceOperation::rspnsflt, faultstring];
      Return[$Failed]
    ];
    
    If[!MatchQ[return[[3]], {___XMLElement}],
      Message[InvokeServiceOperation::rspnsinv, return[[3]]]; 
      Return[$Failed]
    ];
    
    If[Length[return[[3]]] === 0, 
      Return[] 
    ]; 

    (* Begin Deserialize *)
    If[TrueQ[$PrintPerformanceNumbers], Print["Deserializing: ", Date[]]];

    (* Set references *)
    references = setSOAPReference[#, bodyNamespaces, bodyDefaultNamespace] & /@ body[[3]];
    
    (* Deserialize *)
    resp = Catch[DeserializeSchemaInstance[return, returnType, bodyNamespaces, bodyDefaultNamespace]];
    
    (* Clear references *)
    If[# =!= Null, 
      Unset[SOAPReference[Evaluate[#]]]
    ] & /@ references;
    
    (* Process exceptions *)
    If[Length[TypeElements[returnType]] === 1 && !MatchQ[resp, _SchemaException],
      resp = First[resp]
    ];
    
    (* Return value *)
    If[MatchQ[resp, _->_], 
      resp = Last[resp]
    ];

    If[TrueQ[$PrintPerformanceNumbers], Print["Finished Deserializing: ", Date[]]];

    resp
  ]

setSOAPReference[
  e:XMLElement[
     (_String | {_String, _String}), 
     {___, {"", "id"}->id_String, ___}, 
     {___}],
  namespaces_List:{},
  defaultNamespace_String:""] := (SOAPReference["#" <> id] = {e, namespaces, defaultNamespace};"#" <> id)

setSOAPReference[___] := Null
  
ToServiceRequest[ parameters_List, 
                  headerParameters_List, 
                  options___?OptionQ] :=
  Module[{encodingStyleURI, namespace = Null, name = Null, ns = Null,
          params, children, envelope, headerParams, bodyAttributes = {}, operationName, 
          headerAttributes = {}, namespaces},
          
    (* Process options *)
    {encodingStyleURI, operationName} = 
      {"EncodingStyleURI", "OperationName"} /. 
         canonicalOptions[Flatten[{options}]]  /. Options[ToServiceRequest];

    If[TrueQ[$PrintPerformanceNumbers], Print["Building SOAPEnvelope: ", Date[]]];

    If[operationName =!= None,
      If[MatchQ[operationName, {_String, _String}],
        namespace = First[operationName];
        name = Last[operationName]
        ,
        Message[InvokeServiceOperation::opttype, operationName, "OperationName"];
      ];
    ];
  
    namespaces = {"soapenv"->"http://schemas.xmlsoap.org/soap/envelope/",
                  "soapenc"->"http://schemas.xmlsoap.org/soap/encoding/",
                  "xsd"->"http://www.w3.org/2001/XMLSchema",
                  "xsi"->"http://www.w3.org/2001/XMLSchema-instance",
                  "xmlns"->"http://www.w3.org/1999/xmlns/",
                  "xmlns"->"http://www.w3.org/2000/xmlns/",
                  "xmlns"->"http://www.w3.org/2001/xmlns/"};

    (* Add SOAP encoding to headers *)
    If[MemberQ[encodingStyleURI, "http://schemas.xmlsoap.org/soap/encoding/"], 
      AppendTo[headerAttributes, {"soapenv", "encodingStyle"}->"http://schemas.xmlsoap.org/soap/encoding/"];
    ];
    headerParams = Catch[SerializeSchemaInstance[#, namespaces, encodingStyleURI] & /@ headerParameters];
    If[MatchQ[headerParams, _SchemaException], Return[$Failed]];

    If[namespace =!= "" && namespace =!= Null, 
      ns = "ns0";
      AppendTo[namespaces, "ns0"->namespace],
      ns = ""
    ];

    (* serialize parameters *)  
    params = Catch[SerializeSchemaInstance[#, namespaces, encodingStyleURI] & /@ parameters];
    If[MatchQ[params, _SchemaException], Return[$Failed]];
    
    (* Add SOAP encoding *)
    If[MemberQ[encodingStyleURI, "http://schemas.xmlsoap.org/soap/encoding/"], 
      AppendTo[bodyAttributes, {"soapenv", "encodingStyle"}->"http://schemas.xmlsoap.org/soap/encoding/"];
    ];

    If[name =!= Null, 
      children = {XMLElement[{"ns0", name}, {{"xmlns", "ns0"}->namespace}, params]},
      children = params;
    ];    
    
    children = {XMLElement[{"soapenv", "Body"}, bodyAttributes, children]};
    
    (* Add headers *)
    If[headerParams =!= {}, 
      PrependTo[children, XMLElement[{"soapenv", "Header"}, headerAttributes, headerParams]]
    ];
    
    (* Create envelope *)
    envelope = 
      XMLObject["Document"][
        {}, 
        XMLElement[
          {"soapenv", "Envelope"}, 
          {{"xmlns", "soapenv"}->"http://schemas.xmlsoap.org/soap/envelope/",
           {"xmlns", "xsd"}->"http://www.w3.org/2001/XMLSchema",
           {"xmlns", "xsi"}->"http://www.w3.org/2001/XMLSchema-instance", 
           {"xmlns", "soapenc"}->"http://schemas.xmlsoap.org/soap/encoding/"},
          children],
        {}];
    
    If[TrueQ[$PrintPerformanceNumbers], Print["Finished Building SOAPEnvelope: ", Date[]]];
    
    envelope
  ]

InvokeServiceOperation[endPoint_String, 
                       message:XMLObject["Document"][___], 
                       options___?OptionQ] :=
  JavaBlock[
    Module[{envelope, result},

      InstallJava[];      
      Block[{$JavaExceptionHandler},      
        If[TrueQ[$PrintShortErrorMessages], 
          $JavaExceptionHandler = reportShortError,
          $JavaExceptionHandler = JLink`Exceptions`Private`$internalJavaExceptionHandler
        ];        
        envelope = ExportString[message, "XML"];
        $redirects = {};
        result = invoke[endPoint, envelope, options];
        $redirects = {};
      ];      
      result
    ]
  ]
  

(* InstallServiceOperation *)
InstallServiceOperation[exprName_Symbol, 
                        endPoint_String,
                        args_List,
                        headers_List,
                        options___?OptionQ] :=
    JavaBlock[
      Module[{allowShortContext, context, argPats = {}, argSyms, argPatsString,
              operationName, operationStyle, soapActionURI, transportStyleURI,
              encodingStyle, encodingStyleURI, returnType, headerPats, headerSyms,
              headerPatsString},
        InstallJava[];

        {allowShortContext, operationName, operationStyle, soapActionURI, 
         transportStyleURI, encodingStyle, encodingStyleURI, returnType} = 
           {"AllowShortContext", "OperationName", "OperationStyle", "SOAPActionURI",
            "TransportStyleURI", "EncodingStyle", "EncodingStyleURI", "ReturnType"} /. 
          canonicalOptions[Flatten[{options}]]  /. Options[InstallServiceOperation];
        
        If[TrueQ[allowShortContext],
          context = Context[exprName];
          If[!MemberQ[$ContextPath, context], AppendTo[$ContextPath, context]];
        ];
        
        argSyms = Unique /@ args;  
        argPats = MapThread[Pattern[#1, getParamType[ElementType[#2]]| Null]&, {argSyms, args}];
        argPatsString = 
          StringJoin[
            ElementLocalName[#], 
            Which[
              MatchQ[ElementType[#], Verbatim[(True | False)]], ":(True | False)", 
              TypeArrayQ[ElementType[#]], ":" <> ToString[getArrayType[ElementType[#]]], 
              True, ToString[getParamType[ElementType[#]]]
            ]
          ]& /@ args;
          
        headerSyms = Unique /@ headers;  
        headerPats = MapThread[Pattern[#1, getParamType[ElementType[#2]]| Null]&, {headerSyms, headers}];
        headerPatsString = 
          StringJoin[
            ElementLocalName[#], 
            Which[
              MatchQ[ElementType[#], Verbatim[(True | False)]], ":(True | False)", 
              TypeArrayQ[ElementType[#]], ":" <> ToString[getArrayType[ElementType[#]]], 
              True, ToString[getParamType[ElementType[#]]]
            ]
          ]& /@ headers;

        With[{expr = exprName, o = options, arguments = args, headerArguments = headers,              
              argSyms = argSyms, argPats = Sequence@@argPats, argPatsString = argPatsString, 
              headerSyms = headerSyms, headerPats = Sequence@@headerPats, headerPatsString = headerPatsString, 
              optSymbol = ToExpression[Context[exprName] <> "Private`" <> "opts"],
              optPat = PatternTest[Pattern@@{ToExpression[Context[exprName] <> "Private`" <> "opts"], BlankNullSequence[]}, OptionQ]},

          OperationName[expr] ^= operationName;
          OperationStyle[expr] ^= operationStyle;
          OperationElements[expr] ^= arguments;
          OperationHeaders[expr] ^= headerArguments;
          SOAPActionURI[expr] ^= soapActionURI;
          TransportStyleURI[expr] ^= transportStyleURI;
          EncodingStyle[expr] ^= encodingStyle;
          EncodingStyleURI[expr] ^= encodingStyleURI;
          ReturnType[expr] ^= returnType;
          
          expr[envelope:XMLObject["Document"][{___}, 
                     XMLElement[
                       ("Envelope" | 
                        {_String, "Envelope"}), 
                       {___}, 
                       {__XMLElement}], 
                     {___}], 
                   optPat] := 
            Module[{opts},
              opts = Union[Flatten[{optSymbol}], Flatten[{o}]];                                                        
              InvokeServiceOperation[endPoint, envelope, opts]                  
            ];

          expr[argPats, headerPats, optPat] := 
            Module[{envelope, result, opts, params, headerParams},
                                                        
              opts = Union[Flatten[{optSymbol}], Flatten[{o}]];
              params = MapThread[#1->#2&, {arguments, argSyms}];
              headerParams = MapThread[#1->#2&, {headerArguments, headerSyms}];
              envelope = ToServiceRequest[params, headerParams, opts];
              If[envelope === $Failed, Return[$Failed]];
              result = InvokeServiceOperation[endPoint, envelope, opts];
              If[result === $Failed, Return[$Failed]];
              FromServiceResponse[result, opts]
                
            ];
                
          ToServiceRequest[expr, argPats, headerPats, optPat] := 
            Module[{opts = Union[Flatten[{optSymbol}], Flatten[{o}]], 
                    params, headerParams},
              params = MapThread[#1->#2&, {arguments, argSyms}];
              headerParams = MapThread[#1->#2&, {headerArguments, headerSyms}];
              ToServiceRequest[params, headerParams, opts]
            ];
            
          InvokeServiceOperation[
                  expr,
                  envelope:XMLObject["Document"][{___}, 
                     XMLElement[
                       ("Envelope" | 
                        {_String, "Envelope"}), 
                       {___}, 
                       {__XMLElement}], 
                     {___}], 
                   optPat] := 
            Module[{opts = Union[Flatten[{optSymbol}], Flatten[{o}]]},
              InvokeServiceOperation[endPoint, envelope, opts]                  
            ];

          InvokeServiceOperation[expr, argPats, headerPats, optPat] := 
            Module[{opts, envelope, params, headerParams}, 
              opts = Union[Flatten[{optSymbol}], Flatten[{o}]];
              params = MapThread[#1->#2&, {arguments, argSyms}];
              headerParams = MapThread[#1->#2&, {headerArguments, headerSyms}];
              envelope = ToServiceRequest[params, headerParams, opts];
              If[envelope === $Failed, Return[$Failed]];  

              InvokeServiceOperation[endPoint, envelope, opts]
            ];
            
          FromServiceResponse[
                  expr,
                  envelope:XMLObject["Document"][{___}, 
                     XMLElement[
                       ("Envelope" | 
                        {_String, "Envelope"}), 
                       {___}, 
                       {__XMLElement}], 
                     {___}], 
                   optPat] := 
            Module[{opts = Union[Flatten[{optSymbol}], Flatten[{o}]]},
              FromServiceResponse[envelope, opts]                  
            ];

          MessageName[expr, "usage"] = StringJoin[
            If[Length[TypeElements[returnType]] === 1,
              returnType = ElementType[First[TypeElements[returnType]]]
            ];
            Which[
              returnType === Null, "void",
              ListQ[returnType], ToString[getListType[returnType]],
              TypeArrayQ[returnType], ToString[getArrayType[returnType]], 
              MatchQ[returnType, Verbatim[(True | False)]], "(True | False)", 
              True, SymbolName[returnType]], 
            " ", 
            SymbolName[expr], "[", ExportString[{Join[argPatsString, headerPatsString]}, "CSV"], "]" 
          ];
              
        ];
        
        $InstalledServices = Union[$InstalledServices, {exprName}];
        
        exprName
      ]
    ]
 
getParamType[type_] :=
  Which[
    MatchQ[type, Verbatim[(True | False)]], 
      type, 
    type === SchemaAnyType,
      Blank[],
    ListQ[type], 
      getListType[type],
    TypeArrayQ[type], 
      getArrayType[type],
    True, 
      Blank[type]
  ]
   
getListType[{type_}] := List[BlankNullSequence[type]]
  
getArrayType[type_] := 
  Module[{element, elementType},
    If[Length[TypeElements[type]] === 1,
      element = First[TypeElements[type]];
      Switch[ElementMinOccurs[element],
        0,  
          elementType = ElementType[element];
          If[elementType =!= Null,                 
            List[BlankNullSequence[elementType]],
            Message[InstallService::type, element]
          ],
        1, 
          elementType = ElementType[element];
          If[elementType =!= Null,                 
            List[BlankSequence[elementType]],
            Message[InstallService::type, element]
          ]
      ]
      ,
      Message[InstallService::array, type];
      Return[$Failed]
    ]
  ];
 
InstallService[wsdlURL_String,
               context_String:"",
               options___?OptionQ] :=
  Module[{result},

    InstallJava[];        
    If[TrueQ[$PrintPerformanceNumbers], Print["Installing Service: ", Date[]]];          
    result = InstallWSDL[wsdlURL, context, options];          
    If[TrueQ[$PrintPerformanceNumbers], Print["Finished Installing Service: ", Date[]]];        
    result
  ]

reportShortError[symbol_Symbol, tagname_String, message_String] :=
  Module[{msg = GetJavaException[]@getMessage[]}, 
    Switch[msg, 
      Null,       
        JLink`Exceptions`Private`$internalJavaExceptionHandler[symbol, tagname, message],
      _,
        Message[InvokeServiceOperation::native, msg]
    ];
  ]

 

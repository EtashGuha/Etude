 (* :Author: Chris Williamson *)

InstallWSDL::usage = 
  "InstallWSDL is a utility function used to map a WSDL file into Mathematica functions."

(* Retrieve definitions and store conveniently in memory *)
getDefinitions[
  XMLObject["Document"][
    {___}, 
    def:XMLElement[
      {"http://schemas.xmlsoap.org/wsdl/", "definitions"},
      {___}, 
      {___}], 
    {___}]] := def

getDefinitions[___] := Null

(* Retrieve targetNamespace and store conveniently in memory *)
getDefinitionsTargetNamespace[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "definitions"},
    {___, {"", "targetNamespace"}->targetNamespace_String, ___}, 
    {___}]] := 
  targetNamespace
                     
getDefinitionsTargetNamespace[___] := ""

getServices[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "definitions"}, 
    {___}, 
    children:{___}]] :=
  Cases[children, XMLElement[{"http://schemas.xmlsoap.org/wsdl/", "service"}, {___, {"", "name"}->name_String,___}, {___}]]
  
getServices[___] := {}

getImports[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "definitions"}, 
    {___}, 
    children:{___}]] :=
  Cases[children, XMLElement[{"http://schemas.xmlsoap.org/wsdl/", "import"}, {___}, {___}]]

getImports[___] := {}

getImportNamespace[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "import"}, 
    {___,{"", "namespace"}->namespace_String,___}, 
    {___}]] := namespace
  
getImportNamespace[___] := Null

getImportLocation[
  XMLElement[
  {"http://schemas.xmlsoap.org/wsdl/", "import"}, 
  {___,{"", "location"}->location_String,___}, 
  {___}]] := location
  
getImportLocation[___] := Null

getTypes[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "definitions"}, 
    {___}, 
    {___, 
     types:XMLElement[
       {"http://schemas.xmlsoap.org/wsdl/", "types"}, 
       {___}, 
       {___}],
     ___}]] := types
     
getTypes[___] := Null

getTypesSchemas[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "types"}, 
    {___}, 
    children:{___}]] :=
  Cases[children, XMLElement[{_?schemaNamespaceQ, "schema"}, {___}, {___}]]

getTypesSchemas[___] := Null

setImport[
  import:XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "import"}, 
    {___}, 
    {___}]] :=
       
  Module[{namespace, location, symbolicXML},
    namespace = getImportNamespace[import];
    If[namespace === Null, 
      Message[InstallServiceOperation::import, Null];
      Return[$Failed]
    ];
    
    location = getImportLocation[import];
    If[location === Null, 
      Message[InstallServiceOperation::import, namespace];
      Return[$Failed]
    ];
      
    symbolicXML = XML`Parser`XMLGet[location];
    If[symbolicXML === $Failed, 
      Message[InstallServiceOperation::import, namespace];
      Return[$Failed]
    ];
      
    symbolicXML = XML`ToVerboseXML[symbolicXML];
    If[symbolicXML === $Failed, 
      Message[InstallServiceOperation::import, namespace];
      Return[$Failed]
    ];
    
    If[getDefinitions[symbolicXML] =!= Null, 
      getWSDLDefinitions[namespace] = getDefinitions[symbolicXML],
      If[symbolicXML === $Failed, 
        Message[InstallServiceOperation::import, namespace];
        Return[$Failed]
      ];
    ];
  ]

InstallWSDL[
  wsdlURL_String, 
  context_String:"",
  options___?OptionQ] :=

  Module[{positions, test, symbolicXML, targetNamespace, 
          services, definitionsNamespaces, 
          definitionsDefaultNamespace, results},
      (* Check to make sure the user-specified context is legal *)
      If[context =!= "", 
 
        (* Context must end with a '`' *)
        If[!StringMatchQ[context, "*`"], 
          Message[
            InstallService::context, 
            context, 
            "contexts must end with a '`'"];
          Return[{}];
        ];
  
        (* Context must not contain illegal characters *)
        If[!MatchQ[
             StringPosition[
               context, 
               {".", "_", "~", "!", "@", "#", "$", "%", "^", 
                "&", "*", "(", ")", "-", "+", "=", "{", "[", 
                "}", "]", "|", "\\", ":", ";", "\"", "\'", 
                "<", ",", ">", "?", "/", " "}], 
             {}], 
          Message[
            InstallService::context, 
            context, 
            "contexts must be alpha-numeric."];
          Return[{}];
        ];
 
        (* Contexts must not begin with a digit *)
        positions = 
          Drop[Prepend[(First[#] + 1) & 
            /@ StringPosition[context, "`"], 1], -1];
        test = (If[DigitQ[StringTake[context, {#,#}]], $Failed] & /@ positions);
        If[Length[Cases[test, $Failed]] > 0, 
          Message[
            InstallService::context, 
            context, 
            "contexts must not begin with a digit."];
          Return[{}];
        ];
      ];
      getUserContext[] = context;
      If[$PrintWSDLDebug === True, Print["User Context: ", getUserContext[]]];
      
      getWsdlUrl[] = wsdlURL;
      If[$PrintWSDLDebug === True, Print["WSDL URL: ", getWsdlUrl[]]];

      getOptions[] = canonicalOptions[Flatten[{options}]];
      If[$PrintWSDLDebug === True, Print["Options: ", getOptions[]]];
        
      symbolicXML = getWSDL[wsdlURL, options];
      If[symbolicXML === $Failed, Return[$Failed]];
                
      getDefinition[] = getDefinitions[symbolicXML];
      If[getDefinition[] === Null, 
        Message[InstallService::definition]; Return[{}]
      ];
  
      targetNamespace = getDefinitionsTargetNamespace[getDefinition[]];
      If[$PrintWSDLDebug === True, Print["Target Namespace: ", targetNamespace]];
           
      getWSDLDefinitions[targetNamespace] = getDefinition[];

      definitionsNamespaces = mapNamespaces[getDefinition[][[2]], {}];
      definitionsDefaultNamespace = getDefaultNamespace[getDefinition[][[2]], ""];
       
      (* This should get services from imports as well *)
      services = getServices[getDefinition[]];
      If[services === {}, 
        Message[InstallService::services];
      ];

      (* Flattens the list because there may be multiple services. 
         Should try to get mapService to return a list all the time. 
         This would enable us to remove the Select expression. *)
      results = 
        Select[
          Flatten[(
            mapService[#, definitionsNamespaces, definitionsDefaultNamespace
          ]) & /@ services], 
          # =!= Null &];
      
      Unset[getDefinition[]];
      Unset[getUserContext[]];
      Unset[getWsdlUrl[]];
      Unset[getOptions[]];
      Unset[getWSDLDefinitions];
      
      results
    ]

getWSDL[url_String, options___?OptionQ] :=
  JavaBlock[
    Module[{method, statusCode, timeout, response, httpClient, proxyHost, proxyPort,
            hostConfig, opts = canonicalOptions[Flatten[{options}]], 
            locationHeader, redirectLocation},
          
      {username, password, timeout} = 
        {"Username", "Password", "Timeout"} /. 
           opts  /. Options[InstallService];
    
      method = JavaNew["org.apache.commons.httpclient.methods.GetMethod", url];
      If[method === $Failed, Return[$Failed]];
        
      (* Configure method. *)
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
      
      (* Setup the client *)
      httpClient = getHttpClient[method, options];
    
      (* configure proxy. *)
      {proxyHost, proxyPort} = getProxyHostAndPort[url];
      proxyPort = ToExpression[proxyPort];
      hostConfig = Null;
      If[proxyHost =!= Null && proxyPort =!= 0,
          LoadJavaClass["org.apache.commons.httpclient.HostConfiguration"];
          hostConfig = JavaNew["org.apache.commons.httpclient.HostConfiguration",
                            HostConfiguration`ANYUHOSTUCONFIGURATION];
          hostConfig@setProxy[proxyHost, proxyPort]
      ];
      
      (* Invoke method *)
      statusCode =
         If[hostConfig =!= Null,
             (* Execute with proxy. *)
             httpClient@executeMethod[hostConfig, method],
         (* else *)
             (* No proxy. *)
             httpClient@executeMethod[method]
         ];
    
      (* Process response *)
      Switch[statusCode, 
        200, 
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
        Message[InstallServiceOperation::wsdl];
        Return[$Failed];
      ];    
      response
    ]
  ]
    

getPorts[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "service"}, 
    {___}, 
    children:{___}]] :=
  Cases[children, XMLElement[{"http://schemas.xmlsoap.org/wsdl/", "port"}, {___, {"", "name"}->name_String, ___}, {___}]]
  
getPorts[___] := {}

getDocumentation[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", _String}, 
    {___}, 
    {___, 
     XMLElement[
       {"http://schemas.xmlsoap.org/wsdl/", "documentation"}, 
       {___}, 
       {doc_String}],
     ___}]] := doc

getDocumentation[___] := Null

mapService[
  service:XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "service"}, 
    attributes:{___, {"", "name"}->name_String,___}, 
    {___}],
  namespaces_List,
  defaultNamespace_String] := 

  Module[{ports, results, serviceNamespaces = mapNamespaces[attributes, namespaces],
          serviceDefaultNamespace = getDefaultNamespace[attributes, defaultNamespace]},
    
    getServiceName[] = name;
    If[getUserContext[] === "",  
      (* Check to make sure it is a legal name.  Drop illegal characters. *)
      getServiceName[] = normalizeSymbolName[name];
                                      
      (* Check to make sure portName does not begin with a number *)
      If[DigitQ[StringTake[name, 1]], 
        Message[InstallService::symbol, name]; 
        Return[{}];
      ];      
      If[$PrintWSDLDebug === True, Print["Service Name: ", getServiceName[]]];
    ];

    getServiceDocumentation[] := getDocumentation[service];

    ports = getPorts[service];
    If[ports === {},
      Message[InstallService::ports, name];
    ];

    results = mapPort[#, serviceNamespaces, serviceDefaultNamespace] & /@ ports;
    
    Unset[getServiceName[]];
    Unset[getServiceDocumentation[]];
    
    results 
  ]

mapService[___] := {}

getBindingName[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "port"}, 
    {___, {"", "binding"}->binding_String, ___}, 
    {___}]] := binding

getBindingName[___] := Null

getEndPoint[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "port"}, 
    {___}, 
    {___, 
     XMLElement[
       {"http://schemas.xmlsoap.org/wsdl/soap/", "address"}, 
       {___, {"", "location"}->location_String, ___}, 
       {___}], 
     ___}]] := location

getEndPoint[___] := Null

getBinding[
  bindingName_String, 
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/","definitions"},
    {___}, 
    children:{___}]] := 
  Module[{results},
    (* Use Cases because we want to match a specific binding, 
       the value is passed in as an argument. *)
    results = 
      Cases[
        children, 
        XMLElement[
          {"http://schemas.xmlsoap.org/wsdl/", "binding"}, 
          {___, {"", "name"} -> bindingName, ___}, 
          {___}]];
    If[Length[results] > 0, First[results], Null]
  ]
        
getBinding[___] := Null

getSOAPBinding[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "binding"}, 
    {___}, 
    {___, 
     soapBinding:XMLElement[
      {"http://schemas.xmlsoap.org/wsdl/soap/", "binding"}, 
      {___}, 
      {___}], 
     ___}]] := soapBinding
  
getSOAPBinding[___] := Null

getBindingTransport[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/soap/", "binding"}, 
    {___, {"", "transport"}->transport_String, ___}, 
    {___}]] := transport

getBindingTransport[___] := ""

getBindingOperationStyle[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/soap/", "binding"}, 
    {___, {"", "style"}->style_String, ___}, 
    {___}]] := style

getBindingOperationStyle[___] := "document"

getBindingOperations[binding:XMLElement[{"http://schemas.xmlsoap.org/wsdl/", "binding"}, 
                       {___}, children:{___}]] := 
  Cases[children, XMLElement[{"http://schemas.xmlsoap.org/wsdl/", "operation"}, {___, {"", "name"}->name_String, ___}, {___}]]
  
getBindingOperations[___] := {}

getPortTypeName[port:XMLElement[{"http://schemas.xmlsoap.org/wsdl/", "binding"}, 
                 {___, {"", "type"}->portType_String, ___}, {___}]] := portType

getPortTypeName[___] := Null

getPortType[portTypeName_String, def:XMLElement[{"http://schemas.xmlsoap.org/wsdl/","definitions"},
            {___},children:{___}]] := 
  Module[{results},
    results = Cases[children, XMLElement[{"http://schemas.xmlsoap.org/wsdl/", "portType"}, 
                     {___, {"", "name"} -> portTypeName, ___}, {___}]];
    If[Length[results] > 0, First[results], Null]
  ]
        
getPortType[___] := Null

mapPort[port:XMLElement[{"http://schemas.xmlsoap.org/wsdl/", "port"}, 
          attributes:{___, {"", "name"}->name_String, ___}, {___}],
        namespaces_List,
        defaultNamespace_String] :=
  Module[{bindingName, binding, soapBinding, portTypeName,
          bindingOperations, qname, portType, results,           
          portNamespaces = mapNamespaces[attributes, namespaces],
          portDefaultNamespace = getDefaultNamespace[attributes, defaultNamespace],
          bindingNamespaces, bindingDefaultNamespace, definitionsNamespaces, 
          definitionsDefaultNamespace, schema, types, typesNamespaces, 
          typesDefaultNamespace, allowShortContext, namespaceContexts, schemaContext}, 

    getPortName[] = name;
    If[getUserContext[] === "",  
      getPortName[] = normalizeSymbolName[name];
                                      
      If[DigitQ[StringTake[getPortName[], 1]], 
        Message[InstallService::symbol, getPortName[]]; 
        Return[];
      ];      
      If[$PrintWSDLDebug === True, Print["Port Name: ", getPortName[]]];    
    ];  

    getSOAPEndPoint[] = getEndPoint[port];
    (* This quietly exits if there is no SOAP binding. *)
    If[getSOAPEndPoint[] === Null, Return[]];
    If[!StringMatchQ[getSOAPEndPoint[],"http://*"] && !StringMatchQ[getSOAPEndPoint[],"https://*"], 
      Message[InstallService::endpoint, getSOAPEndPoint[]];
      Return[];
    ];
    If[$PrintWSDLDebug === True, Print["SOAP End Point: ", getSOAPEndPoint[]]];

    bindingName = getBindingName[port];
    If[bindingName === Null, 
      Message[InstallService::bindingname, name];
      Return[]
    ];
    qname = getQName[bindingName, portNamespaces, portDefaultNamespace];

    binding = getBinding[Last[qname], getWSDLDefinitions[First[qname]]];    
    If[binding === Null, 
      Message[InstallService::binding, bindingName];
      Return[];
    ];
    definitionsNamespaces = mapNamespaces[getWSDLDefinitions[First[qname]][[2]], {}];
    bindingNamespaces = mapNamespaces[binding[[2]], definitionsNamespaces];
    definitionsDefaultNamespace = getDefaultNamespace[getWSDLDefinitions[First[qname]][[2]], ""];
    bindingDefaultNamespace = getDefaultNamespace[binding[[2]], definitionsDefaultNamespace];

    soapBinding = getSOAPBinding[binding];
    If[soapBinding === Null, 
      Message[InstallService::soapbinding, bindingName];
      Return[];
    ];
    
    getTransport[] = getBindingTransport[soapBinding];
    If[!StringMatchQ[getTransport[], "http://schemas.xmlsoap.org/soap/http"], 
      Message[InstallService::transport, getTransport[]];
      Return[];
    ];
    If[$PrintWSDLDebug === True, Print["Transport: ", getTransport[]]];

    getBindingOperationStyle[] = getBindingOperationStyle[soapBinding];    
    If[!StringMatchQ[getBindingOperationStyle[], "rpc"] && 
       !StringMatchQ[getBindingOperationStyle[], "document"], 
      Message[InstallService::style, getBindingOperationStyle[]];
      Return[];
    ];
    If[$PrintWSDLDebug === True, Print["Binding Operation Style: ", getBindingOperationStyle[]]];

    portTypeName = getPortTypeName[binding];
    If[portTypeName === Null, 
      Message[InstallService::porttypename, bindingName];
      Return[]
    ];
    qname = getQName[portTypeName, bindingNamespaces, bindingDefaultNamespace];
    portType = getPortType[Last[qname], getWSDLDefinitions[First[qname]]];
    If[portType === Null, 
      Message[InstallService::porttype, portTypeName];
      Return[];
    ];
    
    bindingOperations =  getBindingOperations[binding];    
    If[bindingOperations === {},
      Message[InstallService::operations, bindingName];
    ];
    
    If[getUserContext[] =!= "", 
      getContext[] = getUserContext[],
      getContext[] = getServiceName[] <> "`" <> getPortName[] <> "`";
    ];
               
    (* Process schema types *) 
    types = getTypes[getDefinition[]];
    If[types =!= Null,
      typesNamespaces = mapNamespaces[getDefinition[][[2]], {}];
      typesNamespaces = mapNamespaces[types[[2]], definitionsNamespaces];
      typesDefaultNamespace = getDefaultNamespace[getDefinition[][[2]], ""];
      typesDefaultNamespace = getDefaultNamespace[types[[2]], definitionsDefaultNamespace];
      
      {allowShortContext, namespaceContexts, schemaContext} = 
        {"SchemaAllowShortContext", "NamespaceContexts", "SchemaContext"} /. 
          getOptions[] /. Options[InstallService];
      If[StringQ[schemaContext], getContext[] = schemaContext];
    
      schema = Catch[LoadSchema[#, getContext[], 
        "AllowShortContext"->allowShortContext, 
        "NamespaceContexts"->namespaceContexts,
        "NamespacePrefixes"->typesNamespaces, 
        "DefaultNamespace"->typesDefaultNamespace] & /@ getTypesSchemas[types]];
      If[MatchQ[schema, _SchemaException], Return[{}]];
    ];

    results = 
      mapOperation[
        #, 
        bindingNamespaces, 
        bindingDefaultNamespace, 
        portType, 
        definitionsNamespaces, 
        definitionsDefaultNamespace
      ] & /@ bindingOperations;
      
    Unset[getContext[]];   
    Unset[getPortName[]];
    Unset[getSOAPEndPoint[]];
    Unset[getTransport[]];
    Unset[getBindingOperationStyle[]];
    
    results
  ]

mapPort[___] :={}

getInputName[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "operation"}, 
    {___, {"", "name"} -> _String, ___}, 
    {___, 
     XMLElement[
       {"http://schemas.xmlsoap.org/wsdl/", "input"}, 
       {___,{"", "name"}->inputName_String,___} , 
       {___}],
     ___}]] := inputName

getInputName[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "operation"}, 
    {___, {"", "name"} -> operationName_String, ___}, 
    {___, 
     XMLElement[
       {"http://schemas.xmlsoap.org/wsdl/", "input"}, 
       {___} , 
       {___}],
     ___}]] := operationName <> "Request"

getInputName[___] := Null
                              
getOutputName[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "operation"}, 
    {___, {"", "name"} -> _String, ___}, 
    {___, 
     XMLElement[
       {"http://schemas.xmlsoap.org/wsdl/", "output"}, 
       {___,{"", "name"}->outputName_String,___} , 
       {___}],
     ___}]] := outputName

getOutputName[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "operation"}, 
    {___, {"", "name"} -> operationName_String, ___}, 
    {___, 
     XMLElement[
       {"http://schemas.xmlsoap.org/wsdl/", "output"}, 
       {___} , 
       {___}],
     ___}]] := operationName <> "Response"

getOutputName[___] := Null

opEqualQ[
  op1:XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "operation"}, 
    {___, {"", "name"} -> opName1_String, ___}, 
    {___}],
  op2:XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "operation"}, 
    {___, {"", "name"} -> opName2_String, ___}, 
    {___}]] := 
  (If[(opName1 === opName2) && 
      ((getOutputName[getOutput[op1]] === getOutputName[getOutput[op2]]) && 
       (getInputName[getInput[op1]] === getInputName[getInput[op2]])), True, False])
  
opEqualQ[___] := False

getPortTypeOperation[
  bindingOp:XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "operation"}, 
    {___}, 
    {___}],
  portType:XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "portType"}, 
    {___}, 
    children:{___}]] := 
  Module[{results},
    results = Select[children, opEqualQ[#, bindingOp] &];
    If[Length[results] > 0, First[results], Null]
  ]
  
getPortTypeOperation[___] := Null

setOption[options_List, option_?OptionQ] := 
  Module[{opts},
    If[(First[option] /. options) === First[option],
      opts = Append[options, option],
      opts = options /. {Rule[First[option],_]->option}
    ]
  ]

mapOperation[bindingOperation:XMLElement[
               {"http://schemas.xmlsoap.org/wsdl/", "operation"}, 
               attributes:{___, {"", "name"}->name_String, ___}, 
               {___}],
             namespaces_List,
             defaultNamespace_String,
             portType:XMLElement[
               {"http://schemas.xmlsoap.org/wsdl/", "portType"}, 
               attributes2:{___}, 
               {___}],
             namespaces2_List, 
             defaultNamespace2_String] :=
  Module[{operationName, portTypeOperation, params, returnType, 
          options, opts, doc = "Documentation was not provided.", 
          bindingOperationNamespaces = mapNamespaces[attributes, namespaces], 
          bindingOperationDefaultNamespace = getDefaultNamespace[attributes, defaultNamespace],
          portTypeNamespaces = mapNamespaces[attributes2, namespaces2],
          portTypeDefaultNamespace = getDefaultNamespace[attributes2, defaultNamespace2]}, 
          
      
    If[$PrintWSDLDebug === True, Print["Operation Name: ", name]];
    
    mapBindingOperation[bindingOperation, bindingOperationNamespaces, bindingOperationDefaultNamespace];
      
    getOperation[] = Null;
    If[StringMatchQ[getOperationStyle[], "rpc"], 
      getOperation[] = {getNamespaceURI[], name};
      If[$PrintWSDLDebug === True, Print["Operation: ", getOperation[]]];
    ];

    operationName = normalizeSymbolName[name];
    If[DigitQ[StringTake[name, 1]], 
      Message[InstallService::symbol, operationName]; 
      Return[];
    ];

    symbol = ToExpression[getContext[] <> operationName];
    If[!MatchQ[symbol, _Symbol], 
      symbol = Unique[getContext[] <> operationName]
    ];

    portTypeOperation = getPortTypeOperation[bindingOperation, portType];
    If[portTypeOperation === Null, 
      Message[InstallService::porttypeoperation, name];
      Return[]
    ];

    params = mapPortTypeOperation[portTypeOperation, portTypeNamespaces, portTypeDefaultNamespace];
    If[params === $Failed, Return[]];
    If[$PrintWSDLDebug === True, Print["Params: ", params]];

    returnType = mapReturnType[portTypeOperation, portTypeNamespaces, portTypeDefaultNamespace];
    If[returnType === $Failed || returnType === Null, returnType = Automatic];
    If[$PrintWSDLDebug === True, Print["Return Type: ", returnType]];
   
    options = getOptions[];    
    opts = {
      OperationName -> getOperation[],
      OperationStyle -> getOperationStyle[],
      SOAPActionURI -> getSOAPActionURI[],
      ReturnType -> returnType,
      TransportStyleURI -> getTransport[],
      EncodingStyleURI -> getEncodingStyleURI[],
      EncodingStyle->getEncodingStyle[]
    };
    
    (opts = setOption[opts, #] ) &  /@ options;
    Unprotect[#]&[symbol];   
    InstallServiceOperation[
      symbol, 
      getSOAPEndPoint[], 
      params, 
      getHeaderParams[], 
      opts];
    Protect[#]&[symbol];

    doc = "Documentation was not provided.";             
    If[getDocumentation[] =!= Null && StringQ[getDocumentation[]], 
      doc = getDocumentation[]
    ];
    If[$PrintWSDLDebug === True, Print["Documentation: ", doc]];
        
    If[StringQ[MessageName[Evaluate[symbol], "usage"]], 
      MessageName[Evaluate[symbol], "usage"] = 
        MessageName[Evaluate[symbol], "usage"] <> "\n\n" <> doc, 
      MessageName[Evaluate[symbol], "usage"] = doc;
    ];
    
    Unset[getSOAPActionURI[]];
    Unset[getEncodingStyleURI[]];
    Unset[getNamespaceURI[]];
    Unset[getEncodingStyle[]];
    Unset[getBindingParts[]];
    Unset[getOperation[]];
    Unset[getOperationStyle[]];
    Unset[getHeaderParams[]];
    
    symbol
  ]

mapOperation[___] := (Message[InstallService::namespaces]; {})

getSOAPActionURI[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "operation"}, {___}, 
    {___, 
     XMLElement[
       {"http://schemas.xmlsoap.org/wsdl/soap/", "operation"}, 
       {___, {"", "soapAction"}->soapAction_String, ___}, 
       {___}], 
     ___}]] := soapAction

getSOAPActionURI[___] := Null

getOperationStyle[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "operation"}, {___}, 
    {___, 
     XMLElement[
       {"http://schemas.xmlsoap.org/wsdl/soap/", "operation"}, 
       {___, {"", "style"}->style_String, ___}, 
       {___}], 
     ___}]] := style
     
getOperationStyle[___] := Null

getInput[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "operation"}, 
    {___}, 
    {___,
     input:XMLElement[
       {"http://schemas.xmlsoap.org/wsdl/", "input"}, 
       {___}, 
       {___}], 
     ___}]] := input
  
getInput[___] := Null

getUse[
  input:XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "input"}, 
    {___}, 
    {___, 
     XMLElement[
       {"http://schemas.xmlsoap.org/wsdl/soap/", "body"}, 
       {___, {"", "use"}->use_String,___}, 
       {___}], 
     ___}]] := use

getUse[___] := ""

getEncodingStyle[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "input"}, 
    {___}, 
    {___, 
     XMLElement[
       {"http://schemas.xmlsoap.org/wsdl/soap/", "body"}, 
       {___, {"", "encodingStyle"}->encodingStyle_String,___}, 
       {___}], 
     ___}]] := encodingStyle

getEncodingStyle[___] := ""

getNamespace[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "input"}, 
    {___}, 
    {___, 
     XMLElement[
       {"http://schemas.xmlsoap.org/wsdl/soap/", "body"}, 
       {___, {"", "namespace"}->namespace_String,___}, 
       {___}], 
     ___}]] := namespace

getNamespace[___] := ""

getBindParts[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "input"}, {___}, 
    {___, 
     XMLElement[
       {"http://schemas.xmlsoap.org/wsdl/soap/", "body"}, 
       {___, {"", "parts"}->parts_String,___}, 
       {___}], 
     ___}]] := parts

getBindParts[___] := ""

getSOAPHeaders[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "input"}, 
    {___}, 
    children:{___}]] := 
  Cases[
    children, 
    XMLElement[
      {"http://schemas.xmlsoap.org/wsdl/soap/", "header"}, 
      {___}, 
      {___}]]

getSOAPHeaders[___] := {}

getMessageName[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/soap/", "header"}, 
    {___, {"", "message"}->messageName_String, ___}, 
    {___}]] := messageName

getPartName[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/soap/", "header"}, 
    {___, {"", "part"}->partName_String, ___}, 
    {___}]] := partName

getPartName[___] := Null

getPart[
  partName_String, 
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/","message"},
    {___},
    children:{___}]] := 
  Module[{results},
    results = 
      Cases[
        children, 
        XMLElement[
          {"http://schemas.xmlsoap.org/wsdl/", "part"}, 
          {___, {"", "name"} -> partName, ___}, 
          {___}]];
    If[Length[results] > 0, First[results], Null]
  ]

getPart[___] := Null

getUse[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/soap/", "header"}, 
    {___, {"", "use"}->use_String, ___}, 
    {___}]] := use

getEncodingStyle[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/soap/", "header"}, 
    {___, {"", "encodingStyle"}->encodingStyle_String, ___}, 
    {___}]] := encodingStyle

getNamespace[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/soap/", "header"}, 
    {___, {"", "namespace"}->namespace_String, ___}, 
    {___}]] := namespace

mapHeader[
  header:XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/soap/", "header"}, 
    attributes:{___}, 
    {___}],
  namespaces_List,
  defaultNamespace_String] :=
  
  Module[{messageName = getMessageName[header], message, 
          partName = getPartName[header], part, 
          headerNamespace = getNamespace[header],
          headerDefaultNamespace = getDefaultNamespace[attributes, defaultNamespace],
          headerNamespaces = mapNamespaces[attributes, namespaces],
          messageNamespaces, messageDefaultNamespace},

    messageName = getQName[messageName, headerNamespaces, headerDefaultNamespace];
    message = getMessage[Last[messageName], getWSDLDefinitions[First[messageName]]];
    If[message === Null, 
      Message[InstallService::message, messageName];
      Return[$Failed];
    ];
    messageNamespaces = mapNamespaces[getWSDLDefinitions[First[messageName]][[2]], {}];
    messageNamespaces = mapNamespaces[message[[2]], messageNamespaces];
    messageDefaultNamespace = getDefaultNamespace[getWSDLDefinitions[First[messageName]][[2]], ""];
    messageDefaultNamespace = getDefaultNamespace[message[[2]], messageDefaultNamespace];
    
    part = getPart[partName, message];
    If[part === Null,
      Message[InstallService::docstyleparts2, messageName];
      Return[$Failed];
    ];
    
    (* TODO - use encoding style and use *)
    headerParam = mapHeaderPart[part, headerNamespace, messageNamespaces, messageDefaultNamespace]
  ]

(* Part defined using a type *)
mapHeaderPart[
  part:XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "part"}, 
    attributes:{___,{"", "type"}->typeName_String,___}, 
    {___}], 
  namespace_String,
  namespaces_List,
  defaultNamespace_String] :=
  Module[{name = getPartName[part], type, elementSymbol, typeSymbol, 
          partNamespaces = mapNamespaces[attributes, namespaces],
          partDefaultNamespace = getDefaultNamespace[attributes, defaultNamespace]},

    name = normalizeSymbolName[name];
    elementSymbol = Symbol[getContext[] <> name <> "$" <> ToString[$ModuleNumber++]];
    Unprotect[Evaluate[elementSymbol]];
    ElementQ[elementSymbol] ^= True;
    ElementNamespace[elementSymbol] ^= namespace;
    ElementLocalName[elementSymbol] ^= name;
    type = getQName[typeName, partNamespaces, partDefaultNamespace];              
    typeSymbol = TypeSymbol@@type;
    If[typeSymbol === Null, 
      Message[InstallService::typeschema, type];
      Return[$Failed]
    ];          
    ElementType[elementSymbol] ^= typeSymbol;
    ElementTypeName[elementSymbol] ^= type;
    Protect[elementSymbol];
    elementSymbol
  ]

(* Part defined using a element *)
mapHeaderPart[
  part:XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "part"}, 
    attributes:{___,{"", "element"}->elementName_String,___}, 
    {___}], 
  namespace_String,
  namespaces_List,
  defaultNamespace_String] :=
  
  Module[{element, elementSymbol, 
          partNamespaces = mapNamespaces[attributes, namespaces],
          partDefaultNamespace = getDefaultNamespace[attributes, defaultNamespace]},
          
    element = getQName[elementName, partNamespaces, partDefaultNamespace];    
    elementSymbol = ElementSymbol@@element;
    If[elementSymbol === Null, 
      Message[InstallService::elementschema, elementName];
      $Failed,
      elementSymbol
    ]
  ]

mapHeaderPart[___] := $Failed

mapBindingOperation[
  bindingOperation:XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "operation"}, 
    attributes:{___, {"", "name"}->name_String, ___}, 
    {___}],
  namespaces_List,
  defaultNamespace_String] :=
  
  Module[{bindingInput, headers, 
          bindingOperationNamespaces = mapNamespaces[attributes, namespaces],
          bindingOperationDefaultNamespace = getDefaultNamespace[attributes, defaultNamespace],
          bindingInputNamespaces, bindingInputDefaultNamespace},
  
    getOperationStyle[] = getOperationStyle[bindingOperation];
    If[getOperationStyle[] === Null, 
       getOperationStyle[] = getBindingOperationStyle[],
       If[!StringMatchQ[getOperationStyle[], "rpc"] && 
          !StringMatchQ[getOperationStyle[], "document"], 
         Message[InstallService::style2, getOperationStyle[]];
         getOperationStyle[] = getBindingOperationStyle[];
       ];
    ];
    If[$PrintWSDLDebug === True, Print["Operation Style: ", getOperationStyle[]]];    

    getSOAPActionURI[] = getSOAPActionURI[bindingOperation];
    If[getSOAPActionURI[] === Null, Message[InstallService::soapaction, name]];
    If[$PrintWSDLDebug === True, Print["SOAP Action URI: ", getSOAPActionURI[]]];    
    
    bindingInput = getInput[bindingOperation];
    If[bindingInput === Null, 
      Message[InstallService::bindinginput, name];
      ,
      bindingInputNamespaces = mapNamespaces[bindingInput[[2]], bindingOperationNamespaces];
      bindingInputDefaultNamespace = getDefaultNamespace[bindingInput[[2]], bindingOperationDefaultNamespace];
    ];

    getHeaderEncodingStyleURI[] = Automatic;
    headers = getSOAPHeaders[bindingInput];
    getHeaderParams[] = (mapHeader[#, bindingInputNamespaces, bindingInputDefaultNamespace] & /@ headers);
    If[$PrintWSDLDebug === True, Print["Header Params: ", getHeaderParams[]]];
      
    (* This is use in WSDL language *)
    getEncodingStyle[] = getUse[bindingInput];
    If[!StringMatchQ[getEncodingStyle[], "literal"] && 
       !StringMatchQ[getEncodingStyle[], "encoded"], 
      Message[InstallService::use, getEncodingStyle[]];
    ];  
    If[$PrintWSDLDebug === True, Print["Encoding Style: ", getEncodingStyle[]]];

    getEncodingStyleURI[] = ImportString[getEncodingStyle[bindingInput], "Words"];
    If[$PrintWSDLDebug === True, Print["Encoding Style URI: ", getEncodingStyleURI[]]];
    
    getNamespaceURI[] = getNamespace[bindingInput];
    If[$PrintWSDLDebug === True, Print["Namespace URI: ", getNamespaceURI[]]];
    
    getBindingParts[] = ImportString[getBindParts[bindingInput], "Words"];
    If[$PrintWSDLDebug === True, Print["Binding Parts: ", getBindingParts[]]];    
  ]

mapBindingOperation[___] := Message[InstallService::namespaces]

getMessageName[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", ("input"|"output")}, 
    {___, {"", "message"}->messageName_String, ___}, 
    {___}]] := messageName
                 
getMessageName[___] := Null

getMessage[
  messageName_String, 
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/","definitions"},
    {___},
    children:{___}]] := 
  Module[{results},
    results = 
      Cases[
        children, 
        XMLElement[
          {"http://schemas.xmlsoap.org/wsdl/", "message"}, 
          {___, {"", "name"} -> messageName, ___}, 
          {___}]];
    If[Length[results] > 0, First[results], Null]
  ]
        
getMessage[___] := Null

getParts[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "message"}, 
    {___}, 
    children:{___}]] :=
  Cases[
    children, 
    XMLElement[
      {"http://schemas.xmlsoap.org/wsdl/", "part"}, 
      {___, {"", "name"} -> _String, ___}, 
      {___}]]
  
getParts[___] := {}
              
samePartQ[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "part"}, 
    {___, {"", "name"} -> partName_String, ___}, 
    {___}], 
  name_String] := 
  MatchQ[partName, name]

samePartQ[___] := False

mapPortTypeOperation[
  portTypeOperation:XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "operation"}, 
    attributes:{___, {"", "name"}->name_String, ___},
    {___}],
  namespaces_List,
  defaultNamespace_String] :=
  
  Module[{doc = Null, portTypeInput, messageName, message, parts, 
          params, qname, messageNamespaces, messageDefaultNamespace,
          portTypeOperationNamespaces = mapNamespaces[attributes, namespaces],
          portTypeOperationDefaultNamespace = getDefaultNamespace[attributes, defaultNamespace],
          portTypeInputNamespaces, portTypeInputDefaultNamespace},

    doc = getDocumentation[portTypeOperation];
    If[doc =!= Null, 
      getDocumentation[] = doc, 
      getDocumentation[] = getServiceDocumentation[]
    ];
    
    portTypeInput = getInput[portTypeOperation];
    (* This cannot be Null because the binding operation must have an input and 
       and therefore in order for the port type operation to be found it must 
       also have an input.
     *) 
    portTypeInputNamespaces = mapNamespaces[portTypeInput[[2]], portTypeOperationNamespaces];
    portTypeInputDefaultNamespace = getDefaultNamespace[portTypeInput[[2]], portTypeOperationDefaultNamespace];
    
    messageName = getMessageName[portTypeInput];
    If[messageName === Null, 
      Message[InstallService::messagename, getInputName[portTypeOperation]];
      Return[$Failed]
    ];
    qname = getQName[messageName, portTypeInputNamespaces, portTypeInputDefaultNamespace];
    
    message = getMessage[Last[qname], getWSDLDefinitions[First[qname]]];    
    If[message === Null, 
      Message[InstallService::message, messageName];
      Return[$Failed];
    ];
    messageNamespaces = mapNamespaces[getWSDLDefinitions[First[qname]][[2]], {}];
    messageNamespaces = mapNamespaces[message[[2]], messageNamespaces];
    messageDefaultNamespace = getDefaultNamespace[getWSDLDefinitions[First[qname]][[2]], ""];
    messageDefaultNamespace = getDefaultNamespace[message[[2]], messageDefaultNamespace];
        
    parts = getParts[message];
    If[getBindingParts[] =!= {}, 
      parts = Intersection[parts, getBindingParts[], SameTest->samePartQ];
    ];
    
    (* TODO use parameterOrder *)
    If[getOperationStyle[] === "document",
      If[Length[parts] > 1, Message[InstallService::docstyleparts, messageName];Return[$Failed]];
      If[Length[parts] === 0, Message[InstallService::docstyleparts2, messageName];Return[$Failed]];
      params = mapPart[First[parts], messageNamespaces, messageDefaultNamespace],
      params = mapPart[#, messageNamespaces, messageDefaultNamespace] & /@ parts;
    ];
    If[Length[Cases[params, $Failed, Infinity]] > 0, Return[$Failed]];
    params

  ]

mapPortTypeOperation[___] := (Message[InstallService::namespaces]; {})

getPartName[
  XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "part"}, 
    {___, {"", "name"}->name_String, ___}, 
    {___}]] := name

getPartName[___] := Null

(* Part defined using a type *)
mapPart[
  part:XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "part"}, 
    attributes:{___,{"", "type"}->typeName_String,___}, 
    {___}], 
  namespaces_List,
  defaultNamespace_String] :=
  Module[{name = getPartName[part], type, elementSymbol, typeSymbol, 
          partNamespaces = mapNamespaces[attributes, namespaces],
          partDefaultNamespace = getDefaultNamespace[attributes, defaultNamespace]},

    type = getQName[typeName, partNamespaces, partDefaultNamespace];
          
    If[getOperationStyle[] === "rpc", 
      name = normalizeSymbolName[name];
      elementSymbol = Symbol[getContext[] <> name <> "$" <> ToString[$ModuleNumber++]];
      Unprotect[Evaluate[elementSymbol]];
      ElementQ[elementSymbol] ^= True;
      ElementNamespace[elementSymbol] ^= "";
      ElementLocalName[elementSymbol] ^= name;
      typeSymbol = TypeSymbol@@type;
      If[typeSymbol === Null, 
        Message[InstallService::typeschema, typeName];
        Return[$Failed]
      ];      
      ElementType[elementSymbol] ^= typeSymbol;
      ElementTypeName[elementSymbol] ^= type;
      Protect[elementSymbol];
      Return[elementSymbol]
    ];
    
    If[getOperationStyle[] === "document",    
         
      (* Using the NamespaceURI but it could be that it should have a blank namespace *)
      If[getOperation[] === Null,
        getOperation[] = {getNamespaceURI[], name};
        If[$PrintWSDLDebug === True, Print["Operation: ", getOperation[]]];
      ];
      
      typeSymbol = TypeSymbol@@type;
      If[typeSymbol === Null, Message[InstallService::typeschema, type];Return[$Failed]];          
      Return[TypeElements[typeSymbol]];
    ];
    $Failed
  ]

(* Part defined using a element *)
mapPart[
  part:XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "part"}, 
    attributes:{___,{"", "element"}->elementName_String,___}, 
    {___}], 
  namespaces_List,
  defaultNamespace_String] :=
  
  Module[{element, 
          elementSymbol, typeSymbol, 
          partNamespaces = mapNamespaces[attributes, namespaces],
          partDefaultNamespace = getDefaultNamespace[attributes, defaultNamespace]},
          
    element = getQName[elementName, partNamespaces, partDefaultNamespace];
    
    elementSymbol = ElementSymbol@@element;
    If[elementSymbol === Null, Message[InstallService::elementschema, elementName];Return[$Failed]];
    
    If[getOperationStyle[] === "rpc", Return[elementSymbol]];
    
    If[getOperationStyle[] === "document", 
      If[getOperation[] === Null, 
        getOperation[] = {ElementNamespace[elementSymbol], ElementLocalName[elementSymbol]};
        If[$PrintWSDLDebug === True, Print["Operation: ", getOperation[]]];
      ];

      typeSymbol = ElementType[elementSymbol];
      If[typeSymbol === Null, Message[InstallService::elementtype, elementName];Return[$Failed]]; 
      Return[TypeElements[typeSymbol]];
    ];
    $Failed
  ]

mapPart[___] := $Failed
    
getOutput[
  operation:XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "operation"}, 
    {___}, 
    {___, 
     input:XMLElement[
       {"http://schemas.xmlsoap.org/wsdl/", "output"}, 
       {___}, 
       {___}], 
     ___}]] := input
  
getOutput[___] := Null

mapReturnType[
  portTypeOperation:XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "operation"}, 
    attributes:{___}, 
    {___}],
  namespaces_List,
  defaultNamespace_String] :=

  Module[{portTypeOutput, messageName, message, parts, qname, 
          operationNamespaces = mapNamespaces[attributes, namespaces], 
          operationDefaultNamespace = getDefaultNamespace[attributes, defaultNamespace], 
          messageNamespaces, messageDefaultNamespace,
          portTypeOutputNamespaces, portTypeOutputDefaultNamespace},
  
    portTypeOutput = getOutput[portTypeOperation];
    (* This cannot be Null because the binding operation must have an output and 
       and therefore in order for the port type operation to be found it must 
       also have an output.
     *) 
    portTypeOutputNamespaces = mapNamespaces[portTypeOutput[[2]], operationNamespaces];
    portTypeOutputDefaultNamespace = getDefaultNamespace[portTypeOutput[[2]], operationDefaultNamespace];
    
    messageName = getMessageName[portTypeOutput];
    If[messageName === Null, 
      Message[InstallService::messagename, getOutputName[portTypeOperation]];
      Return[$Failed]
    ];
    qname = getQName[messageName, portTypeOutputNamespaces, portTypeOutputDefaultNamespace];
    
    message = getMessage[Last[qname], getWSDLDefinitions[First[qname]]];    
    If[message === Null, 
      Message[InstallService::message, messageName];
      Return[$Failed];
    ];
    messageNamespaces = mapNamespaces[getWSDLDefinitions[First[qname]][[2]], {}];
    messageNamespaces = mapNamespaces[message[[2]], messageNamespaces];
    messageDefaultNamespace = getDefaultNamespace[getWSDLDefinitions[First[qname]][[2]], ""];
    messageDefaultNamespace = getDefaultNamespace[message[[2]], messageDefaultNamespace];

    parts = getParts[message];
    If[getOperationStyle[] === "document", 
      Switch[Length[parts],
        0,
          return = Null,
        1,
          return = ElementType[mapReturnPart[First[parts], messageNamespaces, messageDefaultNamespace]],
        _,
          Message[InstallService::returnparts];
      ],
      return = Symbol[getContext[] <> "complexType" <> "$" <> ToString[$ModuleNumber++]];
      TypeQ[return] ^= True;
      TypeNamespace[return] ^= "";
      TypeLocalName[return] ^= getOperation[] <> "Return";
      TypeElements[return] ^= mapReturnPart[#, messageNamespaces, messageDefaultNamespace] & /@ parts;
    ];
    return
  ]

mapReturnType[___] := Null
      
(* Part defined using a type *)
mapReturnPart[
  part:XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "part"}, 
    attributes:{___,{"", "type"}->typeName_String,___}, 
    {___}], 
  namespaces_List,
  defaultNamespace_String] :=
  Module[{name = getPartName[part], type, elementSymbol, typeSymbol, 
          partNamespaces = mapNamespaces[attributes, namespaces],
          partDefaultNamespace = getDefaultNamespace[attributes, defaultNamespace]},
    type = getQName[typeName, partNamespaces, partDefaultNamespace];
           
    name = normalizeSymbolName[name];
    elementSymbol = Symbol[getContext[] <> name <> "$" <> ToString[$ModuleNumber++]];
    Unprotect[Evaluate[elementSymbol]];
    ElementQ[elementSymbol] ^= True;
    ElementNamespace[elementSymbol] ^= "";
    ElementLocalName[elementSymbol] ^= name;
    typeSymbol = TypeSymbol@@type;
    If[typeSymbol === Null, 
      Message[InstallService::typeschema, typeName];
      Return[$Failed]
    ];      
    ElementType[elementSymbol] ^= typeSymbol;
    ElementTypeName[elementSymbol] ^= type;
    Protect[Evaluate[elementSymbol]];
    elementSymbol
  ]

(* Part defined using a element *)
mapReturnPart[
  part:XMLElement[
    {"http://schemas.xmlsoap.org/wsdl/", "part"}, 
    attributes:{___,{"", "element"}->elementName_String,___}, 
    {___}], 
  namespaces_List, 
  defaultNamespace_String] :=
  
  Module[{element, elementSymbol, partNamespaces = mapNamespaces[attributes, namespaces],
          partDefaultNamespace = getDefaultNamespace[attributes, defaultNamespace]},
    
    element = getQName[elementName, partNamespaces, partDefaultNamespace];
    
    elementSymbol = ElementSymbol@@element;
    If[elementSymbol === Null, 
      Message[InstallService::elementschema, elementName];
      $Failed,
      elementSymbol
    ]    
  ]

mapReturnPart[___] := $Failed

(*
 * Function used to clean up Strings used to make Symbols for InstallService and
 * InstallServiceOperation.
 *)     
normalizeSymbolName[name_String] :=
  StringReplace[name, {"."->"", "_"->"", "~"->"", "!"->"", 
                       "@"->"", "#"->"", "$"->"", "%"->"", "^"->"", 
                       "&"->"", "*"->"", "("->"", ")"->"", "-"->"", 
                       "+"->"", "="->"", "{"->"", "["->"", "}"->"", 
                       "]"->"", "|"->"", "\\"->"", ":"->"", ";"->"",
                       "\""->"", "\'"->"", "<"->"", ","->"", ">"->"",
                       "?"->"", "/"->"", " "->""}]
      
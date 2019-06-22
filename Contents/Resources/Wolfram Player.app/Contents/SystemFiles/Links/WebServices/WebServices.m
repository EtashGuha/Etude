(* :Author: Chris Williamson *)

BeginPackage["WebServices`", {"XMLSchema`"}]

WebServices`Information`$Version = "Web Services Version 2.1.0"
WebServices`Information`$VersionNumber = 2.1
WebServices`Information`$ReleaseNumber = 0

WebServices`Information`$CreationID = If[SyntaxQ["53"], ToExpression["53"], 0]
WebServices`Information`$CreationDate = If[SyntaxQ["{2018, 08, 26, 19, 44, 07}"], ToExpression["{2018, 08, 26, 19, 44, 07}"], {0,0,0,0,0,0}]


(*** Functions ***)
InstallService::usage="InstallService[wsdlURL] installs the web service operations specified in the WSDL URL into a default context defined by the service name and port name of the service operation.  InstallService[wsdlURL, myContext] installs the web service operations specified in the WSDL URL into a context specified by the second parameter."
InstallServiceOperation::usage="InstallServiceOperation[symbolName, endPoint, arguments, options] creates a function for the web service operation using the end point, arguments, and options."
ToServiceRequest::usage="ToServiceRequest[parameters, options] builds a request message using input provided in the parameters.  ToServiceRequest[symbol, parameters] builds the request message for a web service operation using input provided in the parameters and information linked with the symbol."
InvokeServiceOperation::usage="InvokeServiceOperation[endPointURL, requestMessage, options]  invokes a web service operation using the request message.  The message is sent to the end point specified in the first argument.  The response message is returned.  InvokeServiceOperation[symbol, parameters, options] builds a request message using the parameters provided and invokes a web service operation using the request message and information linked with the symbol.  The response message is returned.  InvokeServiceOperation[symbol, requestMessage, options] invokes a web service operation using the request message and information linked with the symbol.  The response message is returned."
FromServiceResponse::usage="FromServiceResponse[response] converts a response message into a Mathematica expression."
  
(*** Service Annotations ***)
OperationName::usage="OperationName[symbol] specifies the name of the root element for a web service operation.";
OperationStyle::usage="OperationStyle[symbol] identifies the style of operation.  This can either be 'rpc' or 'document'."
OperationElements::usage="OperationElements[symbol] specifies the elements that make up the arguments of the service operation."
OperationHeaders::usage="OperationHeaders[symbol] specifies the elements that make up the headers of the service operation."
SOAPActionURI::usage="SOAPActionURI[symbol] specifies the SOAP Action URI for a service operation.  This helps identify the intention of the SOAP message."
TransportStyleURI::usage="TransportStyleURI[symbol] specifies the transport that is used for a service."
EncodingStyle::usage="EncodingStyle[symbol] specifies whether a service should be encoded."
EncodingStyleURI::usage="EncodingStyleURI[symbol] specifies the encoding of a service."
ReturnType::usage="ReturnType[symbol] specifies the return type of a service."
  
(*** Global Options ***)
$InstalledServices::usage="$InstalledServices is a list of the web service operations installed."
$PrintServiceRequest::usage="$PrintServiceRequest uses the Mathematica Print function to print the message sent to a web service.  The default is False."
$PrintServiceResponse::usage="$PrintServiceResponse uses the Mathematica Print function to print the message received from a web service before it is deserialized into a Rule syntax expression.  The default is False."  
$PrintShortErrorMessages::usage="$PrintShortErrorMessages specifies whether error messages will be shortened for the user to avoid long intimidating error messages.  The default is True."
$PrintWSDLDebug::usage="$PrintWSDLDebug specifies whether WSDL debugging information will be printed when installing a web service.  The default is False."

Begin["`Private`"]

Unprotect[ InstallService]
Unprotect[ $InstalledServices]
Unprotect[ $PrintServiceRequest]
Unprotect[ $PrintServiceResponse]
Unprotect[ $PrintShortErrorMessages]
Unprotect[ $PrintWSDLDebug]
Unprotect[ FromServiceResponse]
Unprotect[ InstallServiceOperation]
Unprotect[ InvokeServiceOperation]
Unprotect[ ToServiceRequest]

(*** Defaults ***)
$InstalledServices = {}
$PrintServiceRequest = False
$PrintServiceResponse = False
$PrintShortErrorMessages = True
$PrintWSDLDebug = False
$PrintPerformanceNumbers = False
 
(*** Options ***)
Options[ToServiceRequest] =
  {   
    (* SOAP Options *)
    "OperationName"->None,
    "EncodingStyleURI" -> {}
  }

Options[FromServiceResponse] = 
  {
    "ReturnType" -> None
  }

Options[InvokeServiceOperation] = 
  Join[
    Options[ToServiceRequest], 
    Options[FromServiceResponse], 
    {      
      "SOAPActionURI" -> "",
      "TransportStyleURI" -> "http://schemas.xmlsoap.org/soap/http",
      "Username" -> None,
      "Password" -> None,
      "Timeout"->None
    }
  ]

Options[InstallServiceOperation] = 
  Join[
    {
      "AllowShortContext" -> True,
      "OperationStyle"->None,
      "EncodingStyle" -> Automatic
    },
    Options[InvokeServiceOperation]
  ]
  
Options[InstallService] = 
  {
    "AllowShortContext" -> True,
    "SchemaAllowShortContext"->False,
    "NamespaceContexts" -> {},
    "SchemaContext"->Automatic, 
    "Username"->None,
    "Password"->None,
    "Timeout"->None
  }

(*** Messages ***)
InstallService::context = "Illegal Context: `1`, `2`"	
InstallService::symbol = "Illegal Symbol (begins with a digit): `1`"	
InstallService::definition = "The Definition element is not found for this WSDL."
InstallService::endpoint = "Invalid SOAP endpoint: `1`"    
InstallService::services = "There were no valid service elements found in this WSDL."    
InstallService::ports = "There were no valid port elements found the `1` service element in this WSDL."
InstallService::bindingname = "There is not a valid binding name attribute on the `1` port element in this WSDL."
InstallService::binding = "There is not a valid binding named `1` in this WSDL."
InstallService::soapbinding = "There is not a valid SOAP binding element in the `1` binding in this WSDL."
InstallService::transport = "Transport must be http://schemas.xmlsoap.org/soap/http."
InstallService::style = "Style not supported: `1`"
InstallService::porttypename = "There is not a valid port type name attribute on the `1` binding element in this WSDL."
InstallService::porttype = "There is not a valid port type named `1` in this WSDL."
InstallService::operations = "There were no valid operation elements found the `1` binding element in this WSDL."
InstallService::style2 = "Style not supported: `1`.  Using the style specified in the SOAP binding element."
InstallService::soapaction = "There is not a valid SOAP Action attribute on the `1` operation element in this WSDL."
InstallService::bindinginput = "There is not a valid input element in the `1` binding operation."
InstallService::use = "Use not supported: `1`"
InstallService::porttypeoperation = "The port type operation is not found that matches the `1` binding operation."
InstallService::messagename = "There is not a valid message name attribute on the `1` input element in this WSDL."
InstallService::message = "There is not a valid message named `1` in this WSDL."
InstallService::docstyleparts = "There is more than one part element found in the `1` message for a document style operation in this WSDL."    
InstallService::docstyleparts2 = "There is not a valid part found in the `1` message for a document style operation in this WSDL."    
InstallService::elementtype = "The type cannot be found for the element `1`."
InstallService::elementschema = "The element schema cannot be found for `1`."
InstallService::typeschema = "The type schema cannot be found for `1`."
InstallService::returnparts = "Document-style operations must consist of zero or one return parts."
	
InvokeServiceOperation::native = "An error occurred: `1`"
InvokeServiceOperation::env = "Invalid SOAP envelope: `1`"
InvokeServiceOperation::multrtn = "Multiple return values are not supported"
InvokeServiceOperation::prtcl = "Incorrect protocol used in end point URL: `1`.  Should be: `2`"
InvokeServiceOperation::opttype = "`1` is not the correct type for Option `2` "
InvokeServiceOperation::opstyle = "OperationStyle must be set to 'rpc' or 'document'."
InvokeServiceOperation::rspns = "Invoke Failed.  Response is $Failed or Null."
InvokeServiceOperation::rspnsdoc = "Response cannot be parsed:\n `1`"
InvokeServiceOperation::rspnsenv = "SOAP Envelope cannot be parsed from response:\n `1`"
InvokeServiceOperation::rspnsbdy = "SOAP Body cannot be parsed from response:\n `1`"
InvokeServiceOperation::rspnsrtn = "SOAP Return Type cannot be parsed from response:\n `1`"
InvokeServiceOperation::rspnsinv = "SOAP Return element invalid:\n `1`"
InvokeServiceOperation::rspnsflt = "SOAPFault occurred: `1`"
InvokeServiceOperation::mapparam = "Could not map parameters: \n`1`,\n`2`"
InvokeServiceOperation::transport = "Transport must be http://schemas.xmlsoap.org/soap/http."
InvokeServiceOperation::httpcode = "`1` (`2`)"
InvokeServiceOperation::circularredirect = "The request has been redirected to `1` more than once."
InvokeServiceOperation::redirectlocation = "The request to `1` has been redirected but was not given a new location."
	
(*** Helper Functions ***)

(* canonicalOptions takes options and converts them to strings automatically.
 * This is because it allows the packaage to have options defined as strings to 
 * avoid symbol collisions.
 *)
SetAttributes[ canonicalOptions, {Listable}]
canonicalOptions[name_Symbol -> val_] := SymbolName[name] -> val
canonicalOptions[expr___] := expr

(* mapNamespaces parses namespace definitions from a list of rules and adds them to 
 * the current list of namespaces 
 *)
mapNamespaces[attributes:{___Rule}, namespaces_List] :=
  Join[Cases[attributes, ({_?xmlNamespaceQ, namespace_String}->val_String) :> namespace->val], namespaces]
  
mapNamespaces[___] := {}

(* getDefaultNamespaces parses a default namespace from a list of rules *)
getDefaultNamespace[
  attributes:{___Rule, {_?xmlNamespaceQ, "xmlns"}->val_String, ___Rule}, 
  default_String] := val

getDefaultNamespace[attributes:{___Rule}, default_String] := default
  
(* schemaNamespaceQ returns whether a string matches the XMLSchema namespace *)
schemaNamespaceQ[x_String] :=   
  StringMatchQ[x, "http://www.w3.org/1999/XMLSchema"  ] ||
  StringMatchQ[x, "http://www.w3.org/2000/XMLSchema"  ] ||
  StringMatchQ[x, "http://www.w3.org/2001/XMLSchema"  ]
  
schemaNamespaceQ[___] := False

(* xmlNamespaceQ returns whether a string matches the XML namespace *)
xmlNamespaceQ[x_String] := 
         StringMatchQ[x, "http://www.w3.org/1999/xmlns/"] || 
         StringMatchQ[x, "http://www.w3.org/2000/xmlns/"] || 
         StringMatchQ[x, "http://www.w3.org/2001/xmlns/"] || 
         StringMatchQ[x, "xmlns"]

xmlNamespaceQ[___] := False

(* prefixedQNameQ returns whether a string matches the qname pattern *)
prefixedQNameQ[x_String] := StringMatchQ[x, "*:*"]

prefixedQNameQ[___] := False

(* getQName returns a list containing the namespace and localname of a qname *)
getQName[t_String?prefixedQNameQ, namespaces_List, _String] :=
  Module[{position, prefix, namespace, type},
    position = First[First[StringPosition[t, ":"]]];
    prefix = StringTake[t, {1, position - 1}];
    namespace = Cases[namespaces, (prefix->val_) :> val];
    If[namespace === {}, Message[LoadSchema::namespaceprefix, prefix];Return[{"", t}]];
    namespace = First[namespace];
    type = StringTake[t, {position+1, StringLength[t]}];
    {namespace, type}
  ] 

getQName[t_String, namespaces_List, defaultNamespace_String] := {defaultNamespace, t}

(* Set package directory used to find implementation files *)
$webServicesPackageDirectory = DirectoryName[System`Private`FindFile[$Input]]

Get[ToFileName[{$webServicesPackageDirectory, "Kernel"}, "Implementation.m"]]
Get[ToFileName[{$webServicesPackageDirectory, "Kernel"}, "WSDL.m"]]

End[]
EndPackage[]

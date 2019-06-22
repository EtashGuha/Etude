BeginPackage["XMLSchema`", {"JLink`", "Security`"}];

If[$VersionNumber < 6, 
  Needs["XMLSchema`DateString`"];
];

XMLSchema`Information`$Version = "XMLSchema Version 1.0.0";

XMLSchema`Information`$VersionNumber = 1.0;

XMLSchema`Information`$ReleaseNumber = 0;

XMLSchema`Information`$CreationID = If[SyntaxQ["115"], ToExpression["115"], 0]
XMLSchema`Information`$CreationDate = If[SyntaxQ["{2019, 05, 19, 20, 07, 22}"], ToExpression["{2019, 05, 19, 20, 07, 22}"], {0,0,0,0,0,0}]


(* Functions used to work with schema definitions *)
LoadSchema::usage=
  "LoadSchema[schema, namespaces] loads a schema definition and installs the definitions as symbols.";
GenerateSchema::usage=
  "GenerateSchema[context, namespace] generates a schema definition in a given namespace based on the symbols in a given context.";
DefineSchema::usage=
  "DefineSchema[definitions___] defines type meta data used by schema and associates them with the symbols in the definitions.";
  
(* Functions used to work with schema instances *)
SerializeSchemaInstance::usage=
  "SerializeSchemaInstance[item] serializes an item into XML.";
DeserializeSchemaInstance::usage=
  "DeserializeSchemaInstance[item] deserializes an item into an expression.";
ValidateSchemaInstance::usage=
  "ValidateSchemaInstance[item] validates an item against the schema definitions."  
  
(* Utility functions for schema instances *)
NewSchemaInstance::usage="NewSchemaInstance[type, {element->value...}] is used to create a new instance of a type defined using XMLSchema.";
ReplaceSchemaInstanceValue::usage="ReplaceSchemaInstanceValue[instance] set the element value of a schema instance.";
ValidSchemaInstanceQ::usage="ValidSchemaInstanceQ[instance] returns True if the instance is valid and False if it is not."

(* Datatypes *)
SchemaAnyType::usage="SchemaAnyType is used to denote when an element can have any type.";
SchemaBase64Binary::usage="SchemaBase64Binary[{___Integer}] represents data that will be converted to/from base64 binary when used with XMLSchema.";
SchemaDateTime::usage="SchemaDateTime[year_Integer, month_Integer, day_Integer, hour_Integer, minute_Integer, second_Real] represents a dateTime that may be used with XMLSchema.";
SchemaDate::usage="SchemaDate[year_Integer, month_Integer, day_Integer] represents a date that may be used with XMLSchema.";
SchemaTime::usage="SchemaDateTime[hour_Integer, minute_Integer, second_Real] represents a time that may be used with XMLSchema.";
SchemaMathML::usage="SchemaMathML[_XMLElement] represents symbolic MathML that will be converted to/from MathML when used with XMLSchema.";
SchemaExpr::usage="SchemaExpr[_] represents Mathematica expressions that will be converted to/from ExpressionML when used with XMLSchema.";
SchemaNaN::usage="SchemaNaN represents 'not a number' and may be used with Real data types within XMLSchema.";
SchemaException::usage="SchemaException[message_] is thrown when there is an error working with XMLSchema functions.";

(* Meta-data *)
ElementSymbol::usage="ElementSymbol[namespace, name] returns the symbol that is associated with the element.";
TypeSymbol::usage="TypeSymbol[namespace, name] returns the symbol that is associated with the type.";
AttributeSymbol::usage="AttributeSymbol[namespace, name] returns the symbol that is associated with the attribute.";
ElementQ::usage="ElementQ[symbol] specifies whether the symbol is an element.";
TypeQ::usage="TypeQ[symbol] specifies whether the symbol is a type.";
AttributeQ::usage="AttributeQ[symbol] specifies whether the symbol is an attribute.";
ElementGlobalQ::usage="ElementGlobalQ[symbol] specifies whether the symbol is a global element.";
ElementNamespace::usage="ElementNamespace[symbol] returns the namespace associated with the element defined by the symbol.";
ElementLocalName::usage="ElementLocalName[symbol] returns the local name associated with the element defined by the symbol.";
ElementAppInfo::usage="ElementAppInfo[symbol] returns the appinfo associated with the element defined by the symbol.";
ElementDocumentation::usage="ElementDocumentation[symbol] returns the documentation associated with the element defined by the symbol.";
ElementType::usage="ElementType[symbol] returns the type symbol associated with the element symbol.";
ElementTypeName::usage="ElementTypeName[symbol] returns the type name associated with the element symbol.";
ElementMaxOccurs::usage="ElementMaxOccurs[symbol] returns the maximum number of times an element may occur.";
ElementMinOccurs::usage="ElementMinOccurs[symbol] returns the minimum number of times an element may occur.";
ElementDefaultValue::usage="ElementDefaultValue[symbol] returns the default value of an element.";
ElementFixedValue::usage="ElementFixedValue[symbol] returns the fixed value of an element."
ElementReference::usage="ElementReference[symbol] returns the symbol of reference element used to define this element."
TypeGlobalQ::usage="TypeGlobalQ[symbol] specifies whether the symbol is a global type.";
TypeNamespace::usage="TypeNamespace[symbol] returns the namespace associated with the type defined by the symbol.";
TypeLocalName::usage="TypeLocalName[symbol] returns the local name associated with the type defined by the symbol.";
TypeAppInfo::usage="TypeAppInfo[symbol] returns the appinfo associated with the type defined by the symbol.";
TypeDocumentation::usage="TypeDocumentation[symbol] returns the documentation associated with the type defined by the symbol.";
TypeArrayQ::usage="TypeArrayQ[symbol] returns True if the type is an array.";
TypeElements::usage="TypeElements[symbol] returns the elements associated with the type symbol.";
TypeAttributes::usage="TypeAttributes[symbol] returns the attributes associated with the type symbol.";
AttributeGlobalQ::usage="AttributeGlobalQ[symbol] specifies whether the symbol is a global attribute.";
AttributeNamespace::usage="AttributeNamespace[symbol] returns the namespace associated with the attribute defined by the symbol.";
AttributeLocalName::usage="AttributeLocalName[symbol] returns the local name associated with the attribute defined by the symbol.";
AttributeAppInfo::usage="AttributeAppInfo[symbol] returns the appinfo associated with the attribute defined by the symbol.";
AttributeDocumentation::usage="AttributeDocumentation[symbol] returns the documentation associated with the attribute defined by the symbol.";
AttributeType::usage="AttributeType[symbol] returns the type symbol associated with the attribute symbol.";
AttributeTypeName::usage="AttributeTypeName[symbol] returns the type name associated with the attribute symbol.";
AttributeMaxOccurs::usage="AttributeMaxOccurs[symbol] returns the maximum number of times an attribute may occur.";
AttributeMinOccurs::usage="AttributeMinOccurs[symbol] returns the minimum number of times an attribute may occur.";
AttributeDefaultValue::usage="AttributeDefaultValue[symbol] returns the default value of an attribute.";
AttributeFixedValue::usage="AttributeFixedValue[symbol] returns the fixed value of an attribute."
AttributeReference::usage="AttributeReference[symbol] returns the symbol of reference attribute used to define this attribute."
SOAPArrayType::usage="SOAPArrayType[symbol] returns the arrayType associated with a type symbol."

(* SOAP encoding utility functions *)
SOAPReference::usage = "SOAPReference[href] returns the value of a SOAP reference.  These may be set for multi-references in SOAP encoding or for attachments."

$SchemaExprSecurityFunction::usage = "$SchemaExprSecurityFunction is used to for a custom security test when deserializing SchemaExpr expressions."


Begin["`Private`"]

$schemaPackageDirectory = DirectoryName[System`Private`FindFile[$Input]];

(* meta-data defaults *)
ElementSymbol[___] := Null;
TypeSymbol[___] := Null;
AttributeSymbol[___] := Null;
ElementQ[___] := False;
TypeQ[symbol___] := 
  If[MatchQ[symbol, Verbatim[(True | False)] | {String} | {Integer} | {Real} | 
                    {SchemaDateTime} | {SchemaDate} | {SchemaTime} | XMLElement], 
    True, 
    False
  ];
AttributeQ[___] := False;
ElementGlobalQ[___] := False;
ElementNamespace[___] := "";
ElementLocalName[___] := Null;
ElementAppInfo[___] := Null;
ElementDocumentation[___] := Null;
ElementType[___] := SchemaAnyType;
ElementTypeName[___] := Null;
ElementMaxOccurs[___] := 1;
ElementMinOccurs[___] := 1;
ElementDefaultValue[___] := Null;
ElementFixedValue[___] := Null;
ElementReference[___] := Null;
TypeGlobalQ[symbol___] := If[symbol === (True | False), True, False];
TypeNamespace[symbol___] := If[symbol === (True | False), "http://www.w3.org/2001/XMLSchema", ""];
TypeLocalName[symbol___] := If[symbol === (True | False), "boolean", Null];
TypeAppInfo[___] := Null;
TypeDocumentation[___] := Null;
TypeArrayQ[___] := False; 
TypeElements[___] := {}; 
TypeAttributes[___] := {};
AttributeGlobalQ[___] := False;
AttributeNamespace[___] := "";
AttributeLocalName[___] := Null;
AttributeAppInfo[___] := Null;
AttributeDocumentation[___] := Null;
AttributeType[___] := Null;
AttributeTypeName[___] := Null;
AttributeMaxOccurs[___] := 1;
AttributeMinOccurs[___] := 0;
AttributeDefaultValue[___] := Null;
AttributeFixedValue[___] := Null;
AttributeReference[___] := Null;

(* Messages *)
LoadSchema::attributename = "The attribute `1` does not have a name.";
LoadSchema::illegalref = "Global elements and attribute cannot contain ref attributes: `1`.";
LoadSchema::elementname = "The element `1` does not have a name.";
LoadSchema::elementschema = "The element schema cannot be found for {`1`, `2`}.";
LoadSchema::elementschema2 = "The element schema could not be processed correctly: `1`.";
LoadSchema::attributeschema = "The attribute schema cannot be found for {`1`, `2`}.";
LoadSchema::attributechema2 = "The attribute schema could not be processed correctly: `1`.";
LoadSchema::typeschema = "The type schema cannot be found for {`1`, `2`}.";
LoadSchema::typeschema2 = "The type schema could not be processed correctly: `1`.";
LoadSchema::namespaceprefix = "The namespace prefix could not be found: `1`";
LoadSchema::typename = "The global type `1` does not have a name.";
LoadSchema::unsupported = "`1` is not supported in `2`.";
LoadSchema::file = "Error parsing schema file: `1`";
LoadSchema::namespacecontext = "Context not defined for namespace `1`."
LoadSchema::redefinetype = "Type `1` has already been defined.  It may only be redefined in a redefine element."
LoadSchema::redefineelement = "Element `1` has already been defined.  It may only be redefined in a redefine element."
LoadSchema::redefineattribute = "Attribute `1` has already been defined.  It may only be redefined in a redefine element."
LoadSchema::include = "The target namespace `1` for `2` does not match the current namespace `3`."
LoadSchema::import = "The target namespace `1` for `2` does not match the current namespace `3`."
LoadSchema::import2 = "`1` is not natively supported.  The definitions in this namespace should be previously installed or you will be forced to use Symbolic XML and Strings to represent the data."
LoadSchema::elementtype = "Element `1` contains multiple type definitions."
LoadSchema::attributetype = "Attribute `1` contains multiple type definitions."
LoadSchema::illegalattribute = "Element definition `1` has an illegal attribute `2`."
LoadSchema::illegalattribute2 = "Attribute definition `1` has an illegal attribute `2`."
LoadSchema::valueconstraint = "Element `1` defines both default and fixed attributes."
LoadSchema::valueconstraint2 = "Element `1` defines both default and fixed attributes."
LoadSchema::maxoccurs = "Illegal value for maxoccurs `2`.";
LoadSchema::minoccurs = "Illegal value for minoccurs `2`.";
LoadSchema::cardinality = "Element `1` defines minOccurs `2` greater than maxOccurs `3`.";
LoadSchema::restrictiontype = "Restriction defines both a base type attribute and base type content.";
LoadSchema::listitemtype = "List defines both a item type attribute and item type content.";
LoadSchema::listitemtype2 = "`1` is not supported by list as an item type.";
LoadSchema::schemafile = "`1` does not contain a schema file.";
LoadSchema::requiredchild = "Missing required child from `1` in `2`."
LoadSchema::use = "Illegal value for use `2`.";
 
GenerateSchema::namespace = "Invalid namespace definition: `1`";
GenerateSchema::namespace2 = "Namespace for the symbol (`1`) does not match the namespace for the schema being generated (`2`)."
GenerateSchema::name = "`1` does not contain a valid name.";
GenerateSchema::elementtype = "`1` does not contain a valid type.";
GenerateSchema::minoccurs = "`1` does not contain a valid minoccurs (`2`).";
GenerateSchema::minoccurs2 = "`1` contains a minoccurs value greater than maxoccurs.";
GenerateSchema::maxoccurs = "`1` does not contain a valid maxoccurs (`2`).";
GenerateSchema::maxoccurs2 = "`1` does not contain a maxoccurs value less than minoccurs.";
GenerateSchema::default = "`1` does not contain a valid default value (`2`).";
GenerateSchema::fixed = "`1` does not contain a valid fixed value (`2`).";
GenerateSchema::typeelements = "`1` does not contain a valid list of elements (`2`).";
GenerateSchema::unsupported = "Generate schema does not support `1`."

DefineSchema::pattern = "This syntax of `1` is not supported by DefineSchema: `2`";
DefineSchema::occuranceconstraint = "This pattern defines occurance constraints and may only be used when defining an element inside a complex type: `1`";

SerializeSchemaInstance::element = "Illegal element: `1`";
SerializeSchemaInstance::value = "`1` is not a valid value for type `2`.";
SerializeSchemaInstance::name = "Invalid name `1` for element `2`.";
SerializeSchemaInstance::namespace = "Invalid namespace `1` for element `2`.";
SerializeSchemaInstance::compoundtype = "Invalid compound type: `1`.";
SerializeSchemaInstance::typeelements = "Invalid type elements `1`.";
SerializeSchemaInstance::children = "Invalid children: `1`.";
SerializeSchemaInstance::fixedvalue = "Element `1` must be fixed to `2`.  It was set to `3`.";
SerializeSchemaInstance::maxoccurs = "Element `1` has surpassed the maxoccurs of `2`.";
SerializeSchemaInstance::prohibitedelement = "Element `1` is prohibited.";
SerializeSchemaInstance::requiredelement = "Element `1` is required next and is missing or out-of-order.";
SerializeSchemaInstance::type = "Type `1` is missing the a namespace or localname definition."
SerializeSchemaInstance::prohibitedattribute = "Attribute `1` is prohibited.";
SerializeSchemaInstance::requiredattribute = "Attribute `1` is required.";
SerializeSchemaInstance::illegalattributes = "Attributes need to be a list of Rules: `1`";

DeserializeSchemaInstance::element = "Illegal element: `1`";
DeserializeSchemaInstance::type = "Could not find the type for symbol: `1`";
DeserializeSchemaInstance::value= "`1` is not a valid value for type `2`.";
DeserializeSchemaInstance::compoundtype = "Invalid compound type: `1`.";
DeserializeSchemaInstance::typeelements = "Invalid type elements `1`.";
DeserializeSchemaInstance::children = "Invalid children: `1`.";
DeserializeSchemaInstance::attributes = "Invalid attributes: `1`";
DeserializeSchemaInstance::fixedvalue = "Element `1` must be fixed to `2`.  It was set to `3`.";
DeserializeSchemaInstance::maxoccurs = "Element `1` has surpassed the maxoccurs of `2`.";
DeserializeSchemaInstance::prohibitedelement = "Element `1` is prohibited.";
DeserializeSchemaInstance::requiredelement = "Element `1` is required next and is missing or out-of-order.";
DeserializeSchemaInstance::name = "Invalid name `1` for element `2`.";
DeserializeSchemaInstance::namespace = "Invalid namespace `1` for element `2`.";
DeserializeSchemaInstance::insecure = "Expression `1` failed the security test.";
DeserializeSchemaInstance::prohibitedattribute = "Attribute `1` is prohibited.";
DeserializeSchemaInstance::requiredattribute = "Attribute `1` is required.";

ValidateSchemaInstance::element = "Illegal element: `1`";
ValidateSchemaInstance::value = "`1` is not a valid value for type `2`.";
ValidateSchemaInstance::name = "Invalid name `1` for element `2`.";
ValidateSchemaInstance::namespace = "Invalid namespace `1` for element `2`.";
ValidateSchemaInstance::compoundtype = "Invalid compound type: `1`.";
ValidateSchemaInstance::typeelements = "Invalid type elements `1`.";
ValidateSchemaInstance::children = "Invalid children: `1`.";
ValidateSchemaInstance::fixedvalue = "Element `1` must be fixed to `2`.  It was set to `3`.";
ValidateSchemaInstance::maxoccurs = "Element `1` has surpassed the maxoccurs of `2`.";
ValidateSchemaInstance::prohibitedelement = "Element `1` is prohibited.";
ValidateSchemaInstance::requiredelement = "Element `1` is required next and is missing or out-of-order.";
ValidateSchemaInstance::type = "`1` is not a type symbol."
ValidateSchemaInstance::prohibitedattribute = "Attribute `1` is prohibited.";
ValidateSchemaInstance::requiredattribute = "Attribute `1` is required.";
ValidateSchemaInstance::illegalattributes = "Attributes need to be a list of Rules: `1`";

ReplaceSchemaInstanceValue::compoundtype = "Invalid compound type: `1`.";
ReplaceSchemaInstanceValue::notfound = "Element `1` not found in type `2`.";

TypeSymbol["http://www.w3.org/2001/XMLSchema", "anyType"] = SchemaAnyType;
TypeGlobalQ[SchemaAnyType] = True;
TypeNamespace[SchemaAnyType] = "http://www.w3.org/2001/XMLSchema";
TypeLocalName[SchemaAnyType] = "anyType";
TypeQ[SchemaAnyType] = True;

TypeSymbol["http://www.w3.org/2001/XMLSchema", "string"] = String;
TypeGlobalQ[String] = True;
TypeNamespace[String] = "http://www.w3.org/2001/XMLSchema";
TypeLocalName[String] = "string";
TypeQ[String] = True;

TypeSymbol["http://www.w3.org/2001/XMLSchema", "int"] = Integer;
TypeGlobalQ[Integer] = True;
TypeNamespace[Integer] = "http://www.w3.org/2001/XMLSchema";
TypeLocalName[Integer] = "int";
TypeQ[Integer] = True;

TypeSymbol["http://www.w3.org/2001/XMLSchema", "double"] = Real;
TypeGlobalQ[Real] = True;
TypeNamespace[Real] = "http://www.w3.org/2001/XMLSchema";
TypeLocalName[Real] = "double";
TypeQ[Real] = True;

TypeSymbol["http://www.w3.org/2001/XMLSchema", "boolean"] = (True | False);
TypeSymbol["http://www.w3.org/2001/XMLSchema", "normalizedString"] = String;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "token"] = String;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "hexBinary"] = String;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "integer"] = Integer;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "positiveInteger"] = Integer;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "negativeInteger"] = Integer;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "nonNegativeInteger"] = Integer;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "nonPositiveInteger"] = Integer;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "long"] = Integer;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "unsignedLong"] = Integer;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "unsignedInt"] = Integer;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "short"] = Integer;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "unsignedShort"] = Integer;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "byte"] = Integer;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "unsignedByte"] = Integer;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "decimal"] = Real;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "float"] = Real;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "boolean"] = (True | False);
TypeSymbol["http://www.w3.org/2001/XMLSchema", "duration"] = String;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "gYear"] = String;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "gYearMonth"] = String;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "gMonth"] = String;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "gMonthDay"] = String;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "gDay"] = String;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "Name"] = String;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "QName"] = String; (* List *)
TypeSymbol["http://www.w3.org/2001/XMLSchema", "NCName"] = String;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "anyURI"] = String;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "language"] = String;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "ID"] = String;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "IDREF"] = String;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "IDREFS"] = String; (* List *)
TypeSymbol["http://www.w3.org/2001/XMLSchema", "ENTITY"] = String;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "ENTITIES"] = String; (* List *)
TypeSymbol["http://www.w3.org/2001/XMLSchema", "NOTATION"] = String;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "NMTOKEN"] = String;
TypeSymbol["http://www.w3.org/2001/XMLSchema", "NMTOKENS"] = String; (* List *)

TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "base64"] = SchemaBase64Binary;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "array"] = SchemaAnyType;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "string"] = String;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "int"] = Integer;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "double"] = Real;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "boolean"] = (True | False);
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "normalizedString"] = String;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "token"] = String;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "hexBinary"] = String;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "integer"] = Integer;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "positiveInteger"] = Integer;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "negativeInteger"] = Integer;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "nonNegativeInteger"] = Integer;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "nonPositiveInteger"] = Integer;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "long"] = Integer;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "unsignedLong"] = Integer;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "unsignedInt"] = Integer;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "short"] = Integer;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "unsignedShort"] = Integer;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "byte"] = Integer;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "unsignedByte"] = Integer;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "decimal"] = Real;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "float"] = Real;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "boolean"] = (True | False);
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "duration"] = String;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "gYear"] = String;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "gYearMonth"] = String;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "gMonth"] = String;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "gMonthDay"] = String;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "gDay"] = String;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "Name"] = String;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "QName"] = String; (* List *)
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "NCName"] = String;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "anyURI"] = String;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "language"] = String;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "ID"] = String;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "IDREF"] = String;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "IDREFS"] = String; (* List *)
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "ENTITY"] = String;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "ENTITIES"] = String; (* List *)
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "NOTATION"] = String;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "NMTOKEN"] = String;
TypeSymbol["http://schemas.xmlsoap.org/soap/encoding/", "NMTOKENS"] = String; (* List *)

TypeSymbol["http://www.w3.org/2001/XMLSchema", "base64Binary"] = SchemaBase64Binary;
TypeGlobalQ[SchemaBase64Binary] ^= True;
TypeNamespace[SchemaBase64Binary] ^= "http://www.w3.org/2001/XMLSchema";
TypeLocalName[SchemaBase64Binary] ^= "base64Binary";
TypeQ[SchemaBase64Binary] ^= True;
Protect[SchemaBase64Binary];

TypeSymbol["http://www.w3.org/2001/XMLSchema", "dateTime"] = SchemaDateTime;
TypeGlobalQ[SchemaDateTime] ^= True;
TypeNamespace[SchemaDateTime] ^= "http://www.w3.org/2001/XMLSchema";
TypeLocalName[SchemaDateTime] ^= "dateTime";
TypeQ[SchemaDateTime] ^= True;
Protect[SchemaDateTime];

TypeSymbol["http://www.w3.org/2001/XMLSchema", "date"] = SchemaDate;
TypeGlobalQ[SchemaDate] ^= True;
TypeNamespace[SchemaDate] ^= "http://www.w3.org/2001/XMLSchema";
TypeLocalName[SchemaDate] ^= "date";
TypeQ[SchemaDate] ^= True;
Protect[SchemaDate];

TypeSymbol["http://www.w3.org/2001/XMLSchema", "time"] = SchemaTime;
TypeGlobalQ[SchemaTime] ^= True;
TypeNamespace[SchemaTime] ^= "http://www.w3.org/2001/XMLSchema";
TypeLocalName[SchemaTime] ^= "time";
TypeQ[SchemaTime] ^= True;
Protect[SchemaTime];

TypeSymbol["http://www.w3.org/1998/Math/MathML", "math.type"] = SchemaMathML;
TypeGlobalQ[SchemaMathML] ^= True;
TypeNamespace[SchemaMathML] ^= "http://www.w3.org/1998/Math/MathML";
TypeLocalName[SchemaMathML] ^= "math.type";
TypeQ[SchemaMathML] ^= True;
Protect[SchemaMathML];

TypeSymbol["http://www.wolfram.com/XML/", "Expression"] = SchemaExpr;
TypeGlobalQ[SchemaExpr] ^= True;
TypeNamespace[SchemaExpr] ^= "http://www.wolfram.com/XML/";
TypeLocalName[SchemaExpr] ^= "Expression";
TypeQ[SchemaExpr] ^= True;
Protect[SchemaExpr];
(*
  Set the HoldAllComplete as a security measure, designed to prevent 
  this from being a security hole.
*)
SetAttributes[ SchemaExpr, HoldAllComplete]

$baseTypes = {
  {"http://www.w3.org/2001/XMLSchema", "anyType"},
  {"http://www.w3.org/2001/XMLSchema", "string"},
  {"http://www.w3.org/2001/XMLSchema", "int"},
  {"http://www.w3.org/2001/XMLSchema", "double"},
  {"http://www.w3.org/2001/XMLSchema", "boolean"},
  {"http://www.w3.org/2001/XMLSchema", "normalizedString"},
  {"http://www.w3.org/2001/XMLSchema", "token"},
  {"http://www.w3.org/2001/XMLSchema", "hexBinary"}, 
  {"http://www.w3.org/2001/XMLSchema", "integer"},
  {"http://www.w3.org/2001/XMLSchema", "positiveInteger"},
  {"http://www.w3.org/2001/XMLSchema", "negativeInteger"},
  {"http://www.w3.org/2001/XMLSchema", "nonNegativeInteger"},
  {"http://www.w3.org/2001/XMLSchema", "nonPositiveInteger"},
  {"http://www.w3.org/2001/XMLSchema", "long"},
  {"http://www.w3.org/2001/XMLSchema", "unsignedLong"},
  {"http://www.w3.org/2001/XMLSchema", "unsignedInt"},
  {"http://www.w3.org/2001/XMLSchema", "short"},
  {"http://www.w3.org/2001/XMLSchema", "unsignedShort"},
  {"http://www.w3.org/2001/XMLSchema", "byte"},
  {"http://www.w3.org/2001/XMLSchema", "unsignedByte"},
  {"http://www.w3.org/2001/XMLSchema", "decimal"},
  {"http://www.w3.org/2001/XMLSchema", "float"},
  {"http://www.w3.org/2001/XMLSchema", "boolean"},
  {"http://www.w3.org/2001/XMLSchema", "duration"},
  {"http://www.w3.org/2001/XMLSchema", "gYear"},
  {"http://www.w3.org/2001/XMLSchema", "gYearMonth"},
  {"http://www.w3.org/2001/XMLSchema", "gMonth"},
  {"http://www.w3.org/2001/XMLSchema", "gMonthDay"},
  {"http://www.w3.org/2001/XMLSchema", "gDay"},
  {"http://www.w3.org/2001/XMLSchema", "Name"},
  {"http://www.w3.org/2001/XMLSchema", "QName"}, (* List *)
  {"http://www.w3.org/2001/XMLSchema", "NCName"},
  {"http://www.w3.org/2001/XMLSchema", "anyURI"},
  {"http://www.w3.org/2001/XMLSchema", "language"},
  {"http://www.w3.org/2001/XMLSchema", "ID"},
  {"http://www.w3.org/2001/XMLSchema", "IDREF"},
  {"http://www.w3.org/2001/XMLSchema", "IDREFS"}, (* List *)
  {"http://www.w3.org/2001/XMLSchema", "ENTITY"},
  {"http://www.w3.org/2001/XMLSchema", "ENTITIES"}, (* List *)
  {"http://www.w3.org/2001/XMLSchema", "NOTATION"},
  {"http://www.w3.org/2001/XMLSchema", "NMTOKEN"},
  {"http://www.w3.org/2001/XMLSchema", "NMTOKENS"}, (* List *)
  {"http://www.w3.org/2001/XMLSchema", "base64Binary"},
  {"http://www.w3.org/2001/XMLSchema", "dateTime"},
  {"http://www.w3.org/2001/XMLSchema", "date"},
  {"http://www.w3.org/2001/XMLSchema", "time"},
  {"http://www.wolfram.com/XML/", "Expression"},
  {"http://www.w3.org/1998/Math/MathML", "math.type"}
};

$supportedNamespaces = {
  "http://www.w3.org/2001/XMLSchema",
  "http://schemas.xmlsoap.org/soap/encoding/",
  "http://schemas.xmlsoap.org/wsdl/",
  "http://www.wolfram.com/XML/",
  "http://www.w3.org/1998/Math/MathML"
};

GetNamespaceContext[x___] := 
  createSchemaException[LoadSchema::namespacecontext, x];

SOAPReference[___] := {Null, {}, ""};

Options[LoadSchema] = 
  {
    "AllowShortContext"->True,
    "DefaultNamespace"->"",
    "NamespaceContexts" -> {},
    "NamespacePrefixes" -> {}, 
    "Path"->Null 
  };

(*** Process Symbolic XML Schema Data ***)
LoadSchema[
      this:XMLElement[
          {_?schemaNamespaceQ,"schema"},
          attributes:{___Rule},
          children:{___XMLElement}],
      context_String,
      options___?OptionQ] :=
  Module[{allowShortContext, namespaceContexts, namespaces, path, defaultNamespace,
          targetNamespace = getSchemaTargetNamespace[this]},
          
    {allowShortContext, defaultNamespace, namespaceContexts, namespaces, path} = 
      {"AllowShortContext", "DefaultNamespace", "NamespaceContexts", "NamespacePrefixes", "Path"} /. 
        canonicalOptions[Flatten[{options}]] /. 
          Options[LoadSchema];

    (* TODO - validate context *)
    (* TODO - validate defaultNamespace *)
    (* TODO - validate allowShortContext *)
    (* TODO - validate namespaces *)
    
    If[MatchQ[namespaceContexts, {___Rule}],
      (GetNamespaceContext[First[#]] = Last[#];
       If[allowShortContext,
         AppendTo[$ContextPath, Last[#]]
       ]) & /@ namespaceContexts;      
    ];
    
    GetNamespaceContext[targetNamespace] = context;
    If[allowShortContext,
      AppendTo[$ContextPath, context]
    ];

    $includedFiles = {path};
    Block[{$schemaPath = path},
      processSchema[this, targetNamespace, namespaces, defaultNamespace, True]
    ];
    
    (* Clean up *)
    $includedFiles =.;
    If[MatchQ[namespaceContexts, {___Rule}],
      GetNamespaceContext[First[#]] =. & /@ namespaceContexts;      
    ];
    GetNamespaceContext[targetNamespace] =.;
    Clear[ElementSchema];
    Clear[ElementNamespaces];
    Clear[ElementDefaultNamespace];
    Clear[ElementDefaultElementFormQ];
    Clear[ElementDefaultAttributeFormQ];
    Clear[TypeSchema];
    Clear[TypeNamespaces];
    Clear[TypeDefaultNamespace];
    Clear[TypeDefaultElementFormQ];
    Clear[TypeDefaultAttributeFormQ];
    Clear[AttributeSchema];
    Clear[AttributeNamespaces];
    Clear[AttributeDefaultNamespace];
    Clear[AttributeDefaultElementFormQ];
    Clear[AttributeDefaultAttributeFormQ];
  ];

LoadSchema[XMLObject["Document"][{___}, root_XMLElement, {___}], 
    context_String,
    options___?OptionQ] :=
  LoadSchema[root, context, options];
  
LoadSchema[
      file_String, 
      context_String,
      options___?OptionQ] :=
  Module[{xml},
    xml = loadSchemaFile[file];
    LoadSchema[xml, context, options, "Path"->file]
  ];

LoadSchema[
      this:XMLElement[
          ("schema" | {_String , "schema"}),
          {___Rule},
          {___XMLElement}],
      context_String,
      options___?OptionQ] :=
  LoadSchema[XML`ToVerboseXML[this], context, options]
  
hashSchemaDefinitions[x:XMLElement[{_?schemaNamespaceQ, ("simpleType" | "complexType")}, {___, {"", "name"}->name_String, ___}, {___}],
                         targetNamespace_String, 
                         schemaNamespaces_List, 
                         defaultNamespace_String, 
                         defaultElementFormQ:(True | False), 
                         defaultAttributeFormQ:(True | False),
                         redefine:(True | False)] := 
  (
    If[(TypeSchema[targetNamespace, name] === Processed || MatchQ[TypeSchema[targetNamespace, name], _XMLElement]) && !redefine, 
      createSchemaException[LoadSchema::redefinetype, {targetNamespace, name}]
    ];
    TypeSchema[targetNamespace, name] = x;
    TypeNamespaces[targetNamespace, name] = schemaNamespaces;
    TypeDefaultNamespace[targetNamespace, name] = defaultNamespace;
    TypeDefaultElementFormQ[targetNamespace, name] = defaultElementFormQ;
    TypeDefaultAttributeFormQ[targetNamespace, name] = defaultAttributeFormQ;    
  );  
                 
hashSchemaDefinitions[x:XMLElement[{_?schemaNamespaceQ, "element"}, {___, {"", "name"}->name_String, ___}, {___}],
                         targetNamespace_String, 
                         schemaNamespaces_List, 
                         defaultNamespace_String, 
                         defaultElementFormQ:(True | False), 
                         defaultAttributeFormQ:(True | False),
                         redefine:(True | False)] := 
  (
    If[(ElementSchema[targetNamespace, name] === Processed || MatchQ[ElementSchema[targetNamespace, name], _XMLElement]) && !redefine, 
      createSchemaException[LoadSchema::redefineelement, {targetNamespace, name}]
    ];
    ElementSchema[targetNamespace, name] = x;
    ElementNamespaces[targetNamespace, name] = schemaNamespaces;
    ElementDefaultNamespace[targetNamespace, name] = defaultNamespace;
    ElementDefaultElementFormQ[targetNamespace, name] = defaultElementFormQ;
    ElementDefaultAttributeFormQ[targetNamespace, name] = defaultAttributeFormQ;
  );

hashSchemaDefinitions[x:XMLElement[{_?schemaNamespaceQ, "attribute"}, {___, {"", "name"}->name_String, ___}, {___}],
                         targetNamespace_String, 
                         schemaNamespaces_List, 
                         defaultNamespace_String, 
                         defaultElementFormQ:(True | False), 
                         defaultAttributeFormQ:(True | False),
                         redefine:(True | False)] := 
  (
    If[(AttributeSchema[targetNamespace, name] === Processed || MatchQ[ElementSchema[targetNamespace, name], _XMLElement]) && !redefine, 
      createSchemaException[LoadSchema::redefineattribute, {targetNamespace, name}]
    ];
    AttributeSchema[targetNamespace, name] = x;
    AttributeNamespaces[targetNamespace, name] = schemaNamespaces;
    AttributeDefaultNamespace[targetNamespace, name] = defaultNamespace;
    AttributeDefaultElementFormQ[targetNamespace, name] = defaultElementFormQ;
    AttributeDefaultAttributeFormQ[targetNamespace, name] = defaultAttributeFormQ;
  );
  
getSchemaTargetNamespace[
  XMLElement[
    {_?schemaNamespaceQ,"schema"},
    {___, {"","targetNamespace"}->targetNamespace_String,___},
    {___}]
  ] := targetNamespace;
          
getSchemaTargetNamespace[___] := "";

processSchema[
  this_XMLElement,
  namespace_String,
  namespaces_List,
  defaultNamespace_String,
  global:(True | False)] :=
  Module[{attributes = this[[2]], content = this[[3]],
          targetNamespace = "", schemaDefaultNamespace = defaultNamespace,
          defaultElementFormQ = False, defaultAttributeFormQ = False,
          schemaNamespaces = namespaces},

    Switch[First[#], 
      {"", "targetNamespace"}, targetNamespace = Last[#],
      {"", "elementFormDefault"}, defaultElementFormQ = MatchQ[Last[#], "qualified"], 
      {"", "attributeFormDefault"}, defaultAttributeFormQ = MatchQ[Last[#], "qualified"], 
      (* blockDefault *)
      (* finalDefault *)
      (* id *)
      (* version *)
      (* xml:lang *)
      {_?xmlNamespaceQ, "xmlns"}, schemaDefaultNamespace = Last[#],
      {_?xmlNamespaceQ, _String}, schemaNamespaces = Join[{Last[First[#]] -> Last[#]}, schemaNamespaces]
    ] & /@ attributes;

    hashSchemaDefinitions[#, targetNamespace, schemaNamespaces, schemaDefaultNamespace, defaultElementFormQ, defaultAttributeFormQ, False] & /@ content;

    Block[{$defaultElementFormQ = defaultElementFormQ, 
           $defaultAttributeFormQ = defaultAttributeFormQ},
      Switch[getContentName[#],
        "element", 
          processSchemaElement[#, targetNamespace, schemaNamespaces, schemaDefaultNamespace, True],
        ("complexType"|"simpleType"), 
          processSchemaType[#, targetNamespace, schemaNamespaces, schemaDefaultNamespace, True],
        "include", 
          processSchemaInclude[#, targetNamespace, schemaNamespaces, schemaDefaultNamespace, True],
        "import", 
          processSchemaImport[#, targetNamespace, schemaNamespaces, schemaDefaultNamespace, True], 
        "annotation", 
          Null,
        "redefine", 
          Message[LoadSchema::unsupported, "redefine", "schema"];$Failed,       
        "group", 
          Message[LoadSchema::unsupported, "group", "schema"];$Failed,         
        "attributeGroup", 
          Message[LoadSchema::unsupported, "attributeGroup", "schema"];$Failed,         
        "attribute",
          processSchemaAttribute[#, targetNamespace, schemaNamespaces, schemaDefaultNamespace, True],        
        "notation",
          Message[LoadSchema::unsupported, "notation", "schema"];$Failed       
      ] & /@ content
    ]
  ];
  
getContentName[
  XMLElement[
    {_?schemaNamespaceQ, name_String}, 
    {___Rule}, 
    {___XMLElement}]] := name; 

getContentName[___] := "";

(*** Process an Include ***)
processSchemaInclude[
  this_XMLElement,
  namespace_String,
  namespaces_List,
  defaultNamespace_String,
  global:(True | False)] :=

  Module[{attributes = this[[2]], content = this[[3]], 
          location = Null, includeNamespaces = namespaces, 
          schema, targetNamespace, includeDefaultNamespace = defaultNamespace}, 
  
    Switch[First[#], 
      {"","schemaLocation"}, location = Last[#],
      (* id *)
      {_?xmlNamespaceQ, "xmlns"}, includeDefaultNamespace = Last[#],
      {_?xmlNamespaceQ, _String}, includeNamespaces = Join[{Last[First[#]] -> Last[#]}, includeNamespaces]
    ] & /@ attributes;

    (* Get file name *)
    location = getCanonicalPath[$schemaPath, location];

    If[!MemberQ[$includedFiles, location], 
      (* load file *)
      schema = loadSchemaFile[location];
        
      If[MatchQ[schema, XMLElement[{_?schemaNamespaceQ, "schema"}, {___Rule}, {___XMLElement}]], 
        targetNamespace = getTargetNamespace[schema];
        If[targetNamespace =!= namespace, 
          createSchemaException[LoadSchema::include, InputForm[targetNamespace], InputForm[location], InputForm[namespace]];
        ];  
        Block[{$schemaPath = location}, 
          processSchema[schema, namespace, {}, "", True]
        ]; 
        ,
        createSchemaException[LoadSchema::schemafile, location]      
      ];
      AppendTo[$includedFiles, location];
    ];
  ];

processSchemaImport[
  this_XMLElement,
  namespace_String,
  namespaces_List,
  defaultNamespace_String,
  global:(True | False)] :=
  Module[{attributes = this[[2]], content = this[[3]],
          importNamespace = "", location = Null,
          importNamespaces = namespaces, 
          importDefaultNamespace = defaultNamespace,
          schema, targetNamespace}, 
  
    Switch[First[#], 
      {"","namespace"}, importNamespace = Last[#],
      {"","schemaLocation"}, location = Last[#],
      (* id *)
      {_?xmlNamespaceQ, "xmlns"}, importDefaultNamespace = Last[#],
      {_?xmlNamespaceQ, _String}, importNamespaces = Join[{Last[First[#]] -> Last[#]}, importNamespaces]
    ] & /@ attributes;

    If[location =!= Null, 
      (* Get file name *)
      location = getCanonicalPath[$schemaPath, location];

      If[!MemberQ[$includedFiles, location] && 
         !MemberQ[$supportedNamespaces, importNamespace], 
        (* load file *)
        schema = loadSchemaFile[location];   
        If[MatchQ[schema, XMLElement[{_?schemaNamespaceQ, "schema"}, {___Rule}, {___XMLElement}]], 
          targetNamespace = getSchemaTargetNamespace[schema];
          If[targetNamespace =!= importNamespace, 
            createSchemaException[LoadSchema::import, InputForm[targetNamespace], InputForm[loc], InputForm[importNamespace]];
          ];        
          Block[{$schemaPath = location},
            processSchema[schema, importNamespace, {}, "", True]
          ];
          ,
          createSchemaException[LoadSchema::schemafile, location]
        ];
        AppendTo[$includedFiles, location];
      ],
      If[importNamespace =!= "" && !MemberQ[$supportedNamespaces, importNamespace],
        Message[LoadSchema::import2, importNamespace];
      ];
    ];    
  ];

loadSchemaFile[file_String] :=
  Module[{xml},
    xml = XML`Parser`XMLGet[file];
    If[xml === $Failed, 
      createSchemaException[LoadSchema::file, file];
    ];
    xml = XML`ToVerboseXML[xml][[2]]
  ];

getCanonicalPath[parent:(_String | Null), file_String] := 
  Module[{path, loc = file, url},
    (* Get file name *)
    If[parent =!= Null,
      If[!StringMatchQ[parent, "http://*"] && !StringMatchQ[parent, "https://*"], 
        path = "file:" <> parent, 
        path = parent
      ];
      JavaBlock[
        url = JavaNew["java.net.URL", JavaNew["java.net.URL", path], file];
        loc = url@toString[];
      ];
      If[!StringMatchQ[parent, "http://*"] && !StringMatchQ[parent, "https://*"], 
        loc = StringDrop[loc, 5]
      ];
    ];    
    loc 
  ]
  
(*** Process an Element ***)
processSchemaElement[
      this_XMLElement,
      namespace_String, 
      namespaces_List,
      defaultNamespace_String, 
      global:(True | False)] :=

    Module[{attributes = this[[2]], content = this[[3]], name = Null,
            type = Null, minOccurs = 1, maxOccurs = 1, default = Null, 
            fixed = Null, ref = Null, qualifiedFormQ = False,
            elementSymbol = Null, typeSymbol = Null, refSymbol = Null,
            elementNamespaces = namespaces, 
            elementDefaultNamespace = defaultNamespace, doc},

      (* process attributes *)
      Switch[First[#], 
        {"", "name"}, name = Last[#],
        {"", "type"}, type = Last[#],
        {"", "minOccurs"}, minOccurs = 
                             Which[
                               DigitQ[Last[#]] && NonNegative[ToExpression[Last[#]]], ToExpression[Last[#]], 
                               True, createSchemaException[LoadSchema::minoccurs, Last[#]]
                             ], 
        {"", "maxOccurs"}, maxOccurs = 
                             Which[
                               DigitQ[Last[#]] && Positive[ToExpression[Last[#]]], ToExpression[Last[#]], 
                               Last[#] === "unbounded", Infinity,
                               True, createSchemaException[LoadSchema::maxoccurs, Last[#]]
                             ],
        {"", "default"}, default = Last[#], 
        {"", "fixed"}, fixed = Last[#], 
        {"", "ref"}, ref = Last[#], 
        {"", "form"}, qualifiedFormQ = MatchQ[Last[#], "qualified"],
        (* abstract *)
        (* block *)
        (* final *) 
        (* id *)
        (* nillable *)
        (* substitutionGroup *)
        {_?xmlNamespaceQ, "xmlns"}, elementDefaultNamespace = Last[#],
        {_?xmlNamespaceQ, _String}, elementNamespaces = Join[{Last[First[#]] -> Last[#]}, elementNamespaces]
      ] & /@ attributes;
      (* throw an exception if ref and name attribute both present *)
      If[ref =!= Null && name =!= Null, 
        createSchemaException[LoadSchema::illegalattribute, name, "ref"];
      ];

      (* process annotation content *)
      (* TODO - turn this into a function? *)
      If[Length[content] > 0 &&
         MatchQ[First[content], 
           XMLElement[{_?schemaNamespaceQ, ("annotation")}, {___}, {___}]],
        Switch[#, 
          XMLElement[
            {_?schemaNamespaceQ, "documentation"},
            {___}, 
            {_String}], 
            ElementDocumentation[elementSymbol] ^= First[Last[#]],         
          XMLElement[
            {_?schemaNamespaceQ, "appInfo"},
            {___}, 
            {_String}], 
            ElementAppInfo[elementSymbol] ^= First[Last[#]]
        ] & /@ Last[First[content]];
        content = Rest[content]
      ];
                 
      (* call the ref function if ref present *)
      If[ref =!= Null, 
      
        If[global, 
          createSchemaException[LoadSchema::illegalref, this];
        ];
        ref = getQName[ref, elementNamespaces, elementDefaultNamespace];
        refSymbol = getElementSymbol@@ref;
      
        (* Create new symbol and put info into it *)
        elementSymbol = getSymbolName[GetNamespaceContext[ElementNamespace[refSymbol]], ElementLocalName[refSymbol], False];
        ElementQ[elementSymbol] ^= True;
        ElementLocalName[elementSymbol] ^= ElementLocalName[refSymbol];
        ElementNamespace[elementSymbol] ^= ElementNamespace[refSymbol];
        ElementDefaultValue[elementSymbol] ^= ElementDefaultValue[refSymbol];
        ElementFixedValue[elementSymbol] ^= ElementFixedValue[refSymbol];
        ElementType[elementSymbol] ^= ElementType[refSymbol];
        ElementTypeName[elementSymbol] ^= ElementTypeName[refSymbol];
        ElementMinOccurs[elementSymbol] ^= minOccurs;
        ElementMaxOccurs[elementSymbol] ^= maxOccurs;
        ElementReference[elementSymbol] ^= refSymbol;

        If[default =!= Null, createSchemaException[LoadSchema::illegalattribute, "ref", "default"]]; 
        If[fixed =!= Null, createSchemaException[LoadSchema::illegalattribute, "ref", "fixed"]]; 
        If[type =!= Null, createSchemaException[LoadSchema::illegalattribute, "ref", "type"]];  
        ,        
        (* throw an exception if name is not found *)
        If[name === Null, 
          createSchemaException[LoadSchema::elementname, this];
        ];

        (* Lookup elementSymbol *)
        If[global, 
          (* if schema already processed then return the symbol *)
          If[ElementSchema[namespace, name] === Processed, 
            Return[ElementSymbol[namespace, name]];
          ];

          (* mark as processed, so it is not processed multiple times *)
          ElementSchema[namespace, name] = Processed;
  
          (* Lookup symbol name *)
          elementSymbol = ElementSymbol[namespace, name];
          If[!MatchQ[elementSymbol, _Symbol], 
            createSchemaException[LoadSchema::symbol, name];
          ];
          (* Unprotect the symbol so we can update it with schema annotations *)
          If[elementSymbol =!= Null && MemberQ[Attributes[Evaluate[elementSymbol]], Protected], 
            Unprotect[Evaluate[elementSymbol]]
          ]; 
        ];
            
        (* create a new symbol if no symbol exists *)
        If[elementSymbol === Null, 
          (* create a new symbol based on the element name *)
          elementSymbol = getSymbolName[GetNamespaceContext[namespace], name, global];
          (* Unprotect the symbol so we can update it with schema annotations *)
          If[MemberQ[Attributes[Evaluate[elementSymbol]], Protected], 
            Unprotect[Evaluate[elementSymbol]]
          ];   
          ,
          (* clear element annotations if they exist *) 
          clearElement[elementSymbol];
        ];
      
        (* annotate symbol *)
        ElementQ[elementSymbol] ^= True;
        ElementLocalName[elementSymbol] ^= name;        
        If[!global, 
          ElementMinOccurs[elementSymbol] ^= minOccurs;
          ElementMaxOccurs[elementSymbol] ^= maxOccurs;
          If[minOccurs > maxOccurs, 
            createSchemaException[LoadSchema::cardinality, name, minOccurs, maxOccurs];
          ];
          If[qualifiedFormQ || $defaultElementFormQ, 
            ElementNamespace[elementSymbol] ^= namespace,
            ElementNamespace[elementSymbol] ^= "";
          ];
          ,
          ElementSymbol[namespace, name] = elementSymbol;
          ElementGlobalQ[elementSymbol] ^= True;
          ElementNamespace[elementSymbol] ^= namespace;
        ];
        ElementDefaultValue[elementSymbol] ^= default;
        ElementFixedValue[elementSymbol] ^= fixed;
        If[default =!= Null && fixed =!= Null,
          createSchemaException[LoadSchema::valueconstraint, name]  
        ];      
        (* TODO Check to make sure the type is simple or complex with simple or mixed *)
        (* TODO Check to make sure it matches the type *)
            
        (* process type information and get a type symbol *)
        If[type =!= Null, 
          type = getQName[type, elementNamespaces, elementDefaultNamespace];
          typeSymbol = getTypeSymbol@@type
          ,
          If[Length[content] > 0 &&
             MatchQ[First[content], 
               XMLElement[{_?schemaNamespaceQ, ("simpleType" | "complexType")}, {___Rule}, {___XMLElement}]],
            typeSymbol = processSchemaType[First[content], namespace, elementNamespaces, elementDefaultNamespace, False];
            content = Rest[content]        
          ];          
        ];
        If[typeSymbol === Null, 
          typeSymbol = SchemaAnyType;
        ];
        (* Unprotect again after types just in case they match globally *)
        If[MemberQ[Attributes[Evaluate[elementSymbol]], Protected], 
          Unprotect[Evaluate[elementSymbol]]
        ];   
        (* annotate the symbol with type information *)
        ElementType[elementSymbol] ^= typeSymbol;
        If[MatchQ[type, {_String, _String}], 
          ElementTypeName[elementSymbol] ^= type
        ];
      ];
      
      Switch[getContentName[#], 
        "simpleType",
          createSchemaException[LoadSchema::elementtype, name]
        (* unique *)
        (* key *)
        (* keyref *)
      ] & /@ content;
                  
      (* protect so values may not be assigned to this symbol *)
      Protect[Evaluate[elementSymbol]];
      
      elementSymbol
    ];

getElementSymbol[namespace_String, name_String] :=
  Module[{elementSymbol, elementSchema = ElementSchema[namespace, name]},
    Switch[elementSchema, 
      Processed, 
        elementSymbol = ElementSymbol[namespace, name],
      _XMLElement, 
         Block[{$defaultElementFormQ = ElementDefaultElementFormQ[namespace, name], 
                $defaultAttributeFormQ = ElementDefaultAttributeFormQ[namespace, name]},
           elementSymbol = processSchemaElement[elementSchema, namespace, ElementNamespaces[namespace, name], ElementDefaultNamespace[namespace, name], True]
         ],
      _, 
        createSchemaException[LoadSchema::elementschema, namespace, name]
    ];
    If[elementSymbol === Null, 
      createSchemaException[LoadSchema::elementschema2, elementSchema];
    ];    
    elementSymbol
  ]
  
(*** Process a Complex Type ***)
processSchemaType[
  this:XMLElement[{_String, "complexType"}, {___}, {___}],
  namespace_String, 
  namespaces_List,
  defaultNamespace_String,
  global:(True | False)] :=

    Module[{attributes = this[[2]], content = this[[3]],
            name = Null, typeSymbol = Null, 
            typeNamespaces = namespaces, typeDefaultNamespace = defaultNamespace, 
            contentName, elements = {}, attrs = {}},
            
      (* process attributes *)
      Switch[First[#], 
        {"", "name"}, name = Last[#],
        (* abstract *)
        (* block *)
        (* final *)
        (* id *)
        (* mixed *)
        {_?xmlNamespaceQ, "xmlns"}, typeDefaultNamespace = Last[#],
        {_?xmlNamespaceQ, _String}, typeNamespaces = Join[{Last[First[#]] -> Last[#]}, typeNamespaces]
      ] & /@ attributes;

      (* Lookup symbol *)
      If[global, 
        (* throw an exception if a global type does not define a name *)
        If[name === Null, 
          createSchemaException[LoadSchema::typename, this];
        ];
        (* if schema already processed then return the symbol *)
        If[TypeSchema[namespace, name] === Processed, 
          Return[TypeSymbol[namespace, name]];
        ];

        (* mark type as processed so it doesn't get processed again *)
        TypeSchema[namespace, name] = Processed;

        (* Lookup symbol name *)
        typeSymbol = TypeSymbol[namespace, name];
        If[!MatchQ[typeSymbol, _Symbol], 
          createSchemaException[LoadSchema::symbol, name];
        ];
        (* Give the processor a chance to fix unsupported types *)
        If[MatchQ[typeSymbol, XMLElement],
          typeSymbol = Null
        ];        
        (* Unprotect the symbol so we can update it with schema annotations *)
        If[typeSymbol =!= Null && MemberQ[Attributes[#]&[typeSymbol], Protected], 
          Unprotect[#]&[typeSymbol]
        ]; 
      ];

      (* create a new symbol if no symbol exists *)
      If[typeSymbol === Null, 
        If[!global, 
          (* create an anonymous type symbol *)
          typeSymbol = getSymbolName[GetNamespaceContext[namespace], "complexType", global]
          ,
          (* create a new symbol based on the type name *)          
          typeSymbol = getSymbolName[GetNamespaceContext[namespace], name, global];
          (* Unprotect the symbol so we can update it with schema annotations *)
          If[MemberQ[Attributes[#]&[typeSymbol], Protected], 
            Unprotect[#]&[typeSymbol]
          ];
        ],
        (* clear type annotations if they exist *) 
        clearType[typeSymbol];
      ];
      
      (* annotate symbol *)
      TypeQ[typeSymbol] ^= True;
      If[global,
        TypeSymbol[namespace, name] = typeSymbol;            
        TypeNamespace[typeSymbol] ^= namespace;
        TypeLocalName[typeSymbol] ^= name;
        TypeGlobalQ[typeSymbol] ^= True;        
      ];
        
      (* process annotation content *)
      (* TODO - turn this into a function? *)
      If[Length[content] > 0 &&
         MatchQ[First[content], 
           XMLElement[{_?schemaNamespaceQ, ("annotation")}, {___}, {___}]],
        Switch[#, 
          XMLElement[
            {_?schemaNamespaceQ, "documentation"},
            {___}, 
            {_String}], 
            TypeDocumentation[elementSymbol] ^= First[Last[#]],         
          XMLElement[
            {_?schemaNamespaceQ, "appInfo"},
            {___}, 
            {_String}], 
            TypeAppInfo[elementSymbol] ^= First[Last[#]]
        ] & /@ Last[First[content]];
        content = Rest[content]
      ];

      If[Length[content] > 0, 
        contentName = getContentName[First[content]];
        Catch[
          Switch[contentName, 
            "simpleContent",
              (* Simple type with attributes *)
              createUnsupportedException[LoadSchema::unsupported, "simpleContent", "complexType"], 
            "complexContent", 
              processSchemaComplexContent[First[content], namespace, typeNamespaces, typeDefaultNamespace, typeSymbol],
            "sequence" | "all" | "choice" | "group" | "attribute" | "attributeGroup" | "anyAttribute",
              Switch[contentName,
                "sequence",
                  elements = processSchemaSequence[First[content], namespace, typeNamespaces, typeDefaultNamespace, False],
                "all", 
                  elements = processSchemaAll[First[content], namespace, typeNamespaces, typeDefaultNamespace, False],
                "choice",
                  elements = processSchemaChoice[First[content], namespace, typeNamespaces, typeDefaultNamespace, False], 
                "group", 
                  createUnsupportedException[LoadSchema::unsupported, "group", "complexType"] 
              ];
              If[MemberQ[Attributes[#]&[typeSymbol], Protected], 
                Unprotect[#]&[typeSymbol]
              ];
              TypeElements[typeSymbol] ^= elements;            
              If[Length[TypeElements[typeSymbol]] === 1 && ElementMaxOccurs[First[TypeElements[typeSymbol]]] > 1,
                TypeArrayQ[typeSymbol] ^= True;
              ];
              content = Rest[content];
              Switch[getContentName[#], 
                "attribute",
                  AppendTo[attrs, processSchemaAttribute[#, namespace, typeNamespaces, typeDefaultNamespace, False]],
                "attributeGroup",
                  createUnsupportedException[LoadSchema::unsupported, "attributeGroup", "complexType"],
                "anyAttribute",          
                  createUnsupportedException[LoadSchema::unsupported, "anyAttribute", "complexType"]
              ] & /@ content;
              TypeAttributes[typeSymbol] ^= attrs;            
          ],
          "unsupported",
          (typeSymbol = XMLElement) &
        ];
        If[typeSymbol === XMLElement, 
          If[global, TypeSymbol[namespace, name] = XMLElement];
          Return[XMLElement]
        ];
      ];
      setTypeDocumentation[typeSymbol];
      setTypeAccessor[typeSymbol];
      Protect[#]&[typeSymbol];
      typeSymbol      
    ];

processSchemaComplexContent[
  this_XMLElement,
  namespace_String,
  namespaces_List,
  defaultNamespace_String,
  typeSymbol_Symbol] := 
  Module[{attributes = this[[2]], content = this[[3]],
          complexContentNamespaces = namespaces, 
          complexContentDefaultNamespace = defaultNamespace},

    (* process attributes *)
    Switch[First[#], 
      (* id *)
      (* mixed *)
      {_?xmlNamespaceQ, "xmlns"}, complexContentDefaultNamespace = Last[#],
      {_?xmlNamespaceQ, _String}, complexContentNamespaces = Join[{Last[First[#]] -> Last[#]}, complexContentNamespaces]
    ] & /@ attributes;
             
    (* annotation *)
    If[Length[content] > 0 &&
       MatchQ[First[content], 
         XMLElement[{_?schemaNamespaceQ, ("annotation")}, {___}, {___}]],
       content = Rest[content];
    ];
              
    If[Length[content] > 0, 
      Switch[getContentName[First[content]], 
        "restriction",
          processSchemaComplexContentRestriction[First[content], namespace, complexContentNamespaces, complexContentDefaultNamespace, typeSymbol],
        "extension",
          createUnsupportedException[LoadSchema::unsupported, "extension", "complexContent"],
          (* TODO - make this use XMLElement since it is not supported *)
        _, 
          createSchemaException[LoadSchema::requiredchild, "complexContent", this]        
      ], 
      createSchemaException[LoadSchema::requiredchild, "complexContent", this]        
    ];
  ];

processSchemaComplexContentRestriction[
  this_XMLElement,
  namespace_String,
  namespaces_List,
  defaultNamespace_String,
  typeSymbol_Symbol] := 
  Module[{attributes = this[[2]], content = this[[3]],
          base = Null, arrayElementSymbol,
          restrictionTypeSymbol = Null, elements = {},
          arrayTypeNamespace,arrayTypeName,arrayTypeSymbol,
          complexContentRestrictionNamespaces = namespaces,
          complexContentRestrictionDefaultNamespace = defaultNamespace},

    (* process attributes *)
    Switch[First[#], 
      {"", "base"}, base = Last[#], 
      (* id *)
      (* mixed *)
      {_?xmlNamespaceQ, "xmlns"}, complexContentRestrictionDefaultNamespace = Last[#],
      {_?xmlNamespaceQ, _String}, complexContentRestrictionNamespaces = Join[{Last[First[#]] -> Last[#]}, complexContentRestrictionNamespaces]
    ] & /@ attributes;

    (* base *)
    base = getQName[base, complexContentRestrictionNamespaces, complexContentRestrictionDefaultNamespace];
    restrictionTypeSymbol = TypeSymbol @@ base; 
    (* TODO - Throw an exception if restriction base is not found *)
    (* TODO check restriction symbol to make sure it is a complex type *)
    (* TODO check to make sure it is a valid symbol *)
    (* TODO figure out what to do with the base type *)
                     
    (* annotation *)
    If[Length[content] > 0 &&
       MatchQ[First[content], 
         XMLElement[{_?schemaNamespaceQ, ("annotation")}, {___}, {___}]],
       content = Rest[content];
    ];
                
    If[Length[content] > 0, 
      Switch[getContentName[First[content]],
        "sequence",
          elements = processSchemaSequence[First[content], namespace, complexContentRestrictionNamespaces, complexContentRestrictionDefaultNamespace, False],
        "all", 
          elements = processSchemaAll[First[content], namespace, complexContentRestrictionNamespaces, complexContentRestrictionDefaultNamespace, False],
        "choice",
          elements = processSchemaChoice[First[content], namespace, complexContentRestrictionNamespaces, complexContentRestrictionDefaultNamespace, False]
        (* group *)
      ];
      If[MemberQ[Attributes[#]&[typeSymbol], Protected], 
        Unprotect[#]&[typeSymbol]
      ];
      TypeElements[typeSymbol] ^= elements;
      (* TODO Check Restrictions *)
      If[Length[TypeElements[typeSymbol]] === 1 && ElementMaxOccurs[First[TypeElements[typeSymbol]]] > 1, 
        TypeArrayQ[typeSymbol] ^= True;
      ]
    ];
    (* TODO - attributes *)
    (* SOAP Arrays *)                
    If[base === {"http://schemas.xmlsoap.org/soap/encoding/","Array"}, 
      SOAPArrayType[typeSymbol] ^= getQName[getSOAPArrayType[this], complexContentRestrictionNamespaces, complexContentRestrictionDefaultNamespace];
      If[TypeElements[typeSymbol] === {}, 
        arrayTypeNamespace = First[SOAPArrayType[typeSymbol]];
        arrayTypeName = First[StringSplit[Last[SOAPArrayType[typeSymbol]], "["]];     
        arrayTypeSymbol = TypeSymbol[arrayTypeNamespace, arrayTypeName ];
        If[arrayTypeSymbol === Null, 
          arrayTypeSymbol = getTypeSymbol[arrayTypeNamespace, arrayTypeName];
          If[MemberQ[Attributes[#]&[typeSymbol], Protected], 
            Unprotect[#]&[typeSymbol]
          ];                      
        ];
        arrayElementSymbol = getSymbolName[GetNamespaceContext[namespace], "item", False];
        ElementNamespace[arrayElementSymbol] ^= "*";
        ElementLocalName[arrayElementSymbol] ^= "*";
        ElementGlobalQ[arrayElementSymbol] ^= False;
        ElementMinOccurs[arrayElementSymbol] ^= 0;
        ElementMaxOccurs[arrayElementSymbol] ^= Infinity;
        ElementType[arrayElementSymbol] ^= arrayTypeSymbol;
        TypeElements[typeSymbol] ^= {
          arrayElementSymbol
        };                 
      ];
      TypeArrayQ[typeSymbol] ^= True;
    ]
  ];
  
getSOAPArrayType[
  XMLElement[
    {_?schemaNamespaceQ, "restriction"}, 
    {___}, 
    {___, 
     XMLElement[
       {_?schemaNamespaceQ, "attribute"}, 
       {___,{"http://schemas.xmlsoap.org/wsdl/", "arrayType"}->arrayType_String,___}, 
       {___}],
     ___}]] := arrayType;

getSOAPArrayType[___] := Null

(* Process a Sequence *)
processSchemaSequence[
  item_XMLElement,
  namespace_String,
  namespaces_List, 
  defaultNamespace_String,
  global:(True | False)] :=
  Module[{attributes = item[[2]], content = item[[3]], 
          sequenceNamespaces = namespaces, elements = {},
          sequenceDefaultNamespace = defaultNamespace},
    
    (* process attributes *)
    Switch[First[#], 
      (* id *)
      (* minOccurs *)
      (* maxOccurs *)
      {_?xmlNamespaceQ, "xmlns"}, sequenceDefaultNamespace = Last[#],
      {_?xmlNamespaceQ, _String}, sequenceNamespaces = Join[{Last[First[#]] -> Last[#]}, sequenceNamespaces]
    ] & /@ attributes;

    (* annotation *)
    If[Length[content] > 0 &&
       MatchQ[First[content], 
         XMLElement[{_?schemaNamespaceQ, ("annotation")}, {___}, {___}]],
       content = Rest[content];
    ];

    Switch[getContentName[#],
      "sequence",
        AppendTo[elements, processSchemaSequence[#, namespace, sequenceNamespaces, sequenceDefaultNamespace, False]],
      "element", 
        AppendTo[elements, processSchemaElement[#, namespace, sequenceNamespaces, sequenceDefaultNamespace, False]],
      "choice",
        AppendTo[elements, processSchemaChoice[#, namespace, sequenceNamespaces, sequenceDefaultNamespace, False]],
      "any", 
        createUnsupportedException[LoadSchema::unsupported, "any", "sequence"],
      "group", 
        createUnsupportedException[LoadSchema::unsupported, "group", "sequence"]
    ] & /@ content;
    elements 
  ]
    
(* Process an all *)
processSchemaAll[
  this_XMLElement,
  namespace_String,
  namespaces_List, 
  defaultNamespace_String,
  global:(True | False)] :=
  Module[{attributes = this[[2]], content = this[[3]], 
          allNamespaces = namespaces, elements = {},
          allDefaultNamespace = defaultNamespace},

    (* process attributes *)
    Switch[First[#], 
      (* id *)
      (* minOccurs *)
      (* maxOccurs *)
      {_?xmlNamespaceQ, "xmlns"}, allDefaultNamespace = Last[#],
      {_?xmlNamespaceQ, _String}, allNamespaces = Join[{Last[First[#]] -> Last[#]}, allNamespaces]
    ] & /@ attributes;

    (* annotation *)
    If[Length[content] > 0 &&
       MatchQ[First[content], 
         XMLElement[{_?schemaNamespaceQ, ("annotation")}, {___}, {___}]],
       content = Rest[content];
    ];

    Switch[getContentName[#],  
      "element", 
        AppendTo[elements, processSchemaElement[#, namespace, allNamespaces, allDefaultNamespace, False]]
    ] & /@ content;
    If[Length[elements] < 2, 
      elements, 
      And @@ elements
    ]
  ]

(* Process a choice *)
processSchemaChoice[
  this_XMLElement,
  namespace_String,
  namespaces_List, 
  defaultNamespace_String,
  global:(True | False)] :=
  Module[{attributes = this[[2]], content = this[[3]],
          choiceNamespaces = namespaces, elements = {},
          choiceDefaultNamespace = defaultNamespace},

    (* process attributes *)
    Switch[First[#], 
      (* id *)
      (* minOccurs *)
      (* maxOccurs *)
      {_?xmlNamespaceQ, "xmlns"}, choiceDefaultNamespace = Last[#],
      {_?xmlNamespaceQ, _String}, choiceNamespaces = Join[{Last[First[#]] -> Last[#]}, choiceNamespaces]
    ] & /@ attributes;

    (* annotation *)
    If[Length[content] > 0 &&
       MatchQ[First[content], 
         XMLElement[{_?schemaNamespaceQ, ("annotation")}, {___}, {___}]],
       content = Rest[content];
    ];

    Switch[getContentName[#],
      "sequence",
        AppendTo[elements, processSchemaSequence[#, namespace, choiceNamespaces, choiceDefaultNamespace, False]],
      "element", 
        AppendTo[elements, processSchemaElement[#, namespace, choiceNamespaces, choiceDefaultNamespace, False]],
      "choice",
        AppendTo[elements, processSchemaChoice[#, namespace, choiceNamespaces, choiceDefaultNamespace, False]],
      "any", 
        createUnsupportedException[LoadSchema::unsupported, "any", "choice"],
      "group",
        createUnsupportedException[LoadSchema::unsupported, "group", "choice"]
    ] & /@ content;
    If[Length[elements] < 2, 
      elements, 
      Or @@ elements
    ]
  ]

(*** Process a Simple Type ***)
processSchemaType[
  this:XMLElement[{_String,"simpleType"}, {___}, {___}],
  namespace_String, 
  namespaces_List,
  defaultNamespace_String,
  global:(True | False)] :=

  Module[{attributes = this[[2]], content = this[[3]], 
          name = Null, typeSymbol = Null, typeNamespaces = namespaces,
          typeDefaultNamespace = defaultNamespace},
                  
    (* process attributes *)
    Switch[First[#], 
      {"", "name"}, name = Last[#],
      (* final *)
      (* id *)
      {_?xmlNamespaceQ, "xmlns"}, typeDefaultNamespace = Last[#],
      {_?xmlNamespaceQ, _String}, typeNamespaces = Join[{Last[First[#]] -> Last[#]}, typeNamespaces]
    ] & /@ attributes;

    (* annotation *)
    If[Length[content] > 0 &&
       MatchQ[First[content], 
         XMLElement[{_?schemaNamespaceQ, ("annotation")}, {___}, {___}]],
      content = Rest[content];
    ];

    If[Length[content] > 0, 
      Switch[getContentName[First[content]], 
        "restriction",
          typeSymbol = processSchemaSimpleTypeRestriction[First[content], namespace, typeNamespaces, typeDefaultNamespace, False]
          ,
        "list",
          typeSymbol = processSchemaSimpleTypeList[First[content], namespace, typeNamespaces, typeDefaultNamespace, False]
          ,
        "union",
          Message[LoadSchema::unsupported, "union", "simpleType"];
          typeSymbol = XMLElement,
        _, 
          createSchemaException[LoadSchema::requiredchild, "simpleType", this]            
      ]
    ];            
    If[global, 
      TypeSymbol[namespace, type] = typeSymbol;
      TypeSchema[namespace, type] = Processed;
    ];
    typeSymbol      
  ];

processSchemaSimpleTypeRestriction[
  this_XMLElement, 
  namespace_String,
  namespaces_List,
  defaultNamespace_String,
  global:(True | False)] :=
  
  Module[{attributes = this[[2]], content = this[[3]],
          base = Null, typeSymbol = Null,
          simpleTypeRestrictionNamespaces = namespaces,
          simpleTypeRestrictionDefaultNamespace = defaultNamespace},

    (* process attributes *)
    Switch[First[#], 
      {"", "base"}, base = Last[#], 
      (* id *)
      {_?xmlNamespaceQ, "xmlns"}, simpleTypeRestrictionDefaultNamespace = Last[#],
      {_?xmlNamespaceQ, _String}, simpleTypeRestrictionNamespaces = Join[{Last[First[#]] -> Last[#]}, simpleTypeRestrictionNamespaces]
    ] & /@ attributes;
            
    (* annotation *)
    If[Length[content] > 0 &&
       MatchQ[First[content], 
         XMLElement[{_?schemaNamespaceQ, ("annotation")}, {___}, {___}]],
      content = Rest[content];
    ];

    (* process type information and get a type symbol *)
    If[base =!= Null, 
      base = getQName[base, simpleTypeRestrictionNamespaces, simpleTypeRestrictionDefaultNamespace];
      typeSymbol = getTypeSymbol@@base
      ,
      If[Length[content] > 0 &&
         MatchQ[First[content], 
           XMLElement[{_?schemaNamespaceQ, "simpleType"}, {___Rule}, {___XMLElement}]],
        typeSymbol = processSchemaType[First[content], namespace, simpleTypeRestrictionNamespaces, simpleTypeRestrictionDefaultNamespace, False];
        content = Rest[content]        
      ];          
    ];
    If[typeSymbol === Null, 
      typeSymbol = String
    ];
    Switch[getContentName[#], 
      "simpleType",
        createSchemaException[LoadSchema::restrictiontype]
    ] & /@ content;
    typeSymbol
  ];

processSchemaSimpleTypeList[
  this_XMLElement, 
  namespace_String,
  namespaces_List,
  defaultNamespace_String,
  global:(True | False)] :=
  
  Module[{attributes = this[[2]], content = this[[3]], 
          itemType = Null, typeSymbol = Null,
          simpleTypeListNamespaces = namespaces,
          simpleTypeListDefaultNamespace = defaultNamespace},

    (* process attributes *)
    Switch[First[#], 
      {"", "itemType"}, itemType = Last[#], 
      (* id *)
      {_?xmlNamespaceQ, "xmlns"}, simpleTypeListDefaultNamespace = Last[#],
      {_?xmlNamespaceQ, _String}, simpleTypeListNamespaces = Join[{Last[First[#]] -> Last[#]}, simpleTypeListNamespaces]
    ] & /@ attributes;
            
    (* annotation *)
    If[Length[content] > 0 &&
       MatchQ[First[content], 
         XMLElement[{_?schemaNamespaceQ, ("annotation")}, {___}, {___}]],
      content = Rest[content];
    ];
            
    (* process type information and get a type symbol *)
    If[itemType =!= Null, 
      itemType = getQName[itemType, simpleTypeListNamespaces, simpleTypeListDefaultNamespace];
      typeSymbol = getTypeSymbol@@itemType
      ,
      If[Length[content] > 0 &&
         MatchQ[First[content], 
           XMLElement[{_?schemaNamespaceQ, "simpleType"}, {___Rule}, {___XMLElement}]],
        typeSymbol = processSchemaType[First[content], namespace, simpleTypeListNamespaces, simpleTypeListDefaultNamespace, False];
        content = Rest[content]        
      ];          
    ];
    If[typeSymbol === Null, 
      typeSymbol = List
    ];
    Switch[getContentName[#], 
      "simpleType",
        createSchemaException[LoadSchema::listitemtype]
    ] & /@ content;
    (* Check certain types *)
    If[MatchQ[typeSymbol, (String | Integer | Real | SchemaDate | SchemaTime | SchemaDateTime)],             
      typeSymbol = {typeSymbol},
      createSchemaException[LoadSchema::listitemtype2, typeSymbol];
    ]
  ];

getTypeSymbol[namespace_String, name_String] :=
  Module[{typeSymbol},
    (* Get base types *)
    Which[
      TypeSchema[namespace, name] === Processed || MemberQ[$baseTypes, {namespace, name}],
        typeSymbol = TypeSymbol[namespace, name],
      MatchQ[TypeSchema[namespace, name], _XMLElement], 
         Block[{$defaultElementFormQ = TypeDefaultElementFormQ[namespace, name], 
                $defaultAttributeFormQ = TypeDefaultAttributeFormQ[namespace, name]},
           typeSymbol = processSchemaType[TypeSchema[namespace, name], namespace, TypeNamespaces[namespace, name], TypeDefaultNamespace[namespace, name], True]
         ],
      True, 
        createSchemaException[LoadSchema::typeschema, namespace, name]
    ];
    If[typeSymbol === Null, 
      createSchemaException[LoadSchema::typeschema2, TypeSchema[namespace, name]];
    ];    
    typeSymbol
  ]

(*** Process an Attribute ***)
processSchemaAttribute[
      this_XMLElement,
      namespace_String, 
      namespaces_List,
      defaultNamespace_String,
      global:(True | False)] :=

  Module[{attributes = this[[2]], content = this[[3]],
          name= Null, type = Null, use = "optional", default = Null, 
          fixed = Null, ref = Null, qualifiedFormQ = False,
          attributeSymbol = Null, typeSymbol = Null, refSymbol = Null,
          attributeNamespaces = namespaces, 
          attributeDefaultNamespace = defaultNamespace},

    (* process attributes *)
    Switch[First[#], 
      {"", "name"}, name = Last[#],
      {"", "type"}, type = Last[#],
      {"", "use"}, minOccurs = Last[#], 
      {"", "default"}, default = Last[#], 
      {"", "fixed"}, fixed = Last[#], 
      {"", "ref"}, ref = Last[#], 
      {"", "form"}, qualifiedFormQ = MatchQ[Last[#], "qualified"],
      (* id *)
      {_?xmlNamespaceQ, "xmlns"}, attributeDefaultNamespace = Last[#],
      {_?xmlNamespaceQ, _String}, attributeNamespaces = Join[{Last[First[#]] -> Last[#]}, attributeNamespaces]
    ] & /@ attributes;
      
    (* throw an exception if ref and name attribute both present *)
    If[ref =!= Null && name =!= Null, 
      createSchemaException[LoadSchema::illegalattribute2, name, "ref"];
    ];
      
    (* process annotation content *)
    (* TODO - turn this into a function? *)
    If[Length[content] > 0 &&
       MatchQ[First[content], 
         XMLElement[{_?schemaNamespaceQ, ("annotation")}, {___}, {___}]],
      Switch[#, 
        XMLElement[
          {_?schemaNamespaceQ, "documentation"},
          {___}, 
          {_String}], 
          AttributeDocumentation[attributeSymbol] ^= First[Last[#]],         
        XMLElement[
          {_?schemaNamespaceQ, "appInfo"},
          {___}, 
          {_String}], 
          AttributeAppInfo[attributeSymbol] ^= First[Last[#]]
      ] & /@ Last[First[content]];
      content = Rest[content]
    ];
           
    (* call the ref function if ref present *)
    If[ref =!= Null, 
      If[global, 
        createSchemaException[LoadSchema::illegalref, this];
      ];
      ref = getQName[ref, attributeNamespaces, attributeDefaultNamespace];
      refSymbol = getAttributeSymbol@@ref;
      
      (* Create new symbol and put info into it *)
      attributeSymbol = getSymbolName[GetNamespaceContext[AttributeNamespace[refSymbol]], AttributeLocalName[refSymbol], False];
      AttributeQ[attributeSymbol] ^= True;
      AttributeLocalName[attributeSymbol] ^= AttributeLocalName[refSymbol];
      AttributeNamespace[attributeSymbol] ^= AttributeNamespace[refSymbol];
      AttributeDefaultValue[attributeSymbol] ^= AttributeDefaultValue[refSymbol];
      AttributeFixedValue[attributeSymbol] ^= AttributeFixedValue[refSymbol];
      AttributeType[attributeSymbol] ^= AttributeType[refSymbol];
      AttributeTypeName[attributeSymbol] ^= AttributeTypeName[refSymbol];
      Switch[use,
        "optional",
          AttributeMinOccurs[attributeSymbol] ^= 0;
          AttributeMaxOccurs[attributeSymbol] ^= 1,
        "prohibited",
          AttributeMinOccurs[attributeSymbol] ^= 0;
          AttributeMaxOccurs[attributeSymbol] ^= 0,
        "required", 
          AttributeMinOccurs[attributeSymbol] ^= 1;
          AttributeMaxOccurs[attributeSymbol] ^= 1,
        _,
          createSchemaException[LoadSchema::use, use]
      ];
      AttributeReference[attributeSymbol] ^= refSymbol;

      If[default =!= Null, createSchemaException[LoadSchema::illegalattribute2, "ref", "default"]]; 
      If[fixed =!= Null, createSchemaException[LoadSchema::illegalattribute2, "ref", "fixed"]]; 
      If[type =!= Null, createSchemaException[LoadSchema::illegalattribute2, "ref", "type"]];         
      ,
      (* throw an exception if name is not found *)
      If[name === Null, 
        createSchemaException[LoadSchema::attributename, this];
      ];

      (* Lookup attributeSymbol *)
      If[global, 
        (* if schema already processed then return the symbol *)
        If[AttributeSchema[namespace, name] === Processed, 
          Return[AttributeSymbol[namespace, name]];
        ];

        (* mark as processed, so it is not processed multiple times *)
        AttributeSchema[namespace, name] = Processed;
 
        (* Lookup symbol name *)
        attributeSymbol = AttributeSymbol[namespace, name];
        If[!MatchQ[attributeSymbol, _Symbol], 
          createSchemaException[LoadSchema::symbol, name];
        ];
        (* Unprotect the symbol so we can update it with schema annotations *)
        If[attributeSymbol =!= Null && MemberQ[Attributes[Evaluate[attributeSymbol]], Protected], 
          Unprotect[Evaluate[attributeSymbol]]
        ]; 
      ];
            
      (* create a new symbol if no symbol exists *)
      If[attributeSymbol === Null, 
        (* create a new symbol based on the attribute name *)
        attributeSymbol = getSymbolName[GetNamespaceContext[namespace], name, global];
        (* Unprotect the symbol so we can update it with schema annotations *)
        If[MemberQ[Attributes[Evaluate[attributeSymbol]], Protected], 
          Unprotect[Evaluate[attributeSymbol]]
        ];   
        ,
        (* clear attribute annotations if they exist *) 
        clearAttribute[attributeSymbol];
      ];
      
      (* annotate symbol *)
      AttributeQ[attributeSymbol] ^= True;
      AttributeLocalName[attributeSymbol] ^= name;        
      If[!global, 
        Switch[use,
          "optional",
            AttributeMinOccurs[attributeSymbol] ^= 0;
            AttributeMaxOccurs[attributeSymbol] ^= 1,
          "prohibited",
            AttributeMinOccurs[attributeSymbol] ^= 0;
            AttributeMaxOccurs[attributeSymbol] ^= 0,
          "required", 
            AttributeMinOccurs[attributeSymbol] ^= 1;
            AttributeMaxOccurs[attributeSymbol] ^= 1,
          _,
            createSchemaException[LoadSchema::use, use]
        ];
        If[qualifiedFormQ || $defaultAttributeFormQ, 
          AttributeNamespace[attributeSymbol] ^= namespace,
          AttributeNamespace[attributeSymbol] ^= "";
        ];
        ,
        AttributeSymbol[namespace, name] = attributeSymbol;
        AttributeGlobalQ[attributeSymbol] ^= True;
        AttributeNamespace[attributeSymbol] ^= namespace;
      ];
      AttributeDefaultValue[attributeSymbol] ^= default;
      AttributeFixedValue[attributeSymbol] ^= fixed;
      If[default =!= Null && fixed =!= Null,
        createSchemaException[LoadSchema::valueconstraint2, name]  
      ];      
      (* TODO Check to make sure the type is simple or complex with simple or mixed *)
      (* TODO Check to make sure it matches the type *)
           
      (* process type information and get a type symbol *)
      If[type =!= Null, 
        type = getQName[type, attributeNamespaces, attributeDefaultNamespace];
        typeSymbol = getTypeSymbol@@type
        ,
        If[Length[content] > 0 &&
           MatchQ[First[content], 
             XMLElement[{_?schemaNamespaceQ, "simpleType"}, {___Rule}, {___XMLElement}]],
          typeSymbol = processSchemaType[First[content], namespace, attributeNamespaces, attributeDefaultNamespace, False];
          content = Rest[content]        
        ];          
      ];
      If[typeSymbol === Null, 
        typeSymbol = SchemaAnyType
      ];
      (* Unprotect again after types just in case they match globally *)
      If[MemberQ[Attributes[Evaluate[attributeSymbol]], Protected], 
        Unprotect[Evaluate[attributeSymbol]]
      ];   
      (* annotate the symbol with type information *)
      AttributeType[attributeSymbol] ^= typeSymbol;
      If[MatchQ[type, {_String, _String}], 
        AttributeTypeName[attributeSymbol] ^= type
      ];
    ];
      
    Switch[getContentName[#], 
      ("simpleType" | "complexType"),
        createSchemaException[LoadSchema::attributetype, name]
    ] & /@ content;
                 
    (* protect so values may not be assigned to this symbol *)
    Protect[Evaluate[attributeSymbol]];
      
    attributeSymbol
  ];
   
getAttributeSymbol[namespace_String, name_String] :=
  Module[{attributeSymbol},
    Switch[AttributeSchema[namespace, name], 
      Processed, 
        attributeSymbol = AttributeSymbol[namespace, name],
      _XMLElement, 
         Block[{$defaultElementFormQ = AttributeDefaultElementFormQ[namespace, name], 
                $defaultAttributeFormQ = AttributeDefaultAttributeFormQ[namespace, name]},
           attributeSymbol = processSchemaAttribute[AttributeSchema[namespace, name], namespace, AttributeNamespaces[namespace, name], AttributeDefaultNamespace[namespace, name], True]
         ],
      _, 
        createSchemaException[LoadSchema::attributeschema, namespace, name]
    ];
    If[attributeSymbol === Null, 
      createSchemaException[LoadSchema::attributeschema2, AttributeSchema[namespace, name]];
    ];    
    attributeSymbol
  ]

Options[GenerateSchema] = 
  {
    "ElementDefaultForm" -> Automatic,
    "AttributeDefaultForm" -> Automatic,
    "DefaultNamespace"->Automatic,
    "Namespaces"-> {}
  }   

(*** Generate a Schema ***) 
GenerateSchema[context_String, namespace_String, options___?OptionQ] := 
  Module[{symbols},
    symbols = Names[context <> "*"];
    symbols = Symbol /@ symbols;
    symbols = Select[symbols, TypeGlobalQ[#] || ElementGlobalQ[#] &];
    GenerateSchema[symbols, namespace, options]
  ];
  
(*** Generate a Schema using the symbolList as a filter ***) 
GenerateSchema[symbols_List, namespace_String, options___?OptionQ] := 
  Module[{children = {}, attributes = {}, namespaces, xsdNamespace},
      
    {namespaces, $elementDefaultForm } = 
      {"Namespaces", "ElementDefaultForm"} /. 
         canonicalOptions[Flatten[{options}]] /. Options[GenerateSchema];
    
    (* Check namespaces *)
    namespaces = (validateNamespaceDefinition /@ namespaces);
    $namespaceQ = namespaces;
        
    (* Add schema namespace *)
    If[!MemberQ[$namespaceQ, _String->"http://www.w3.org/2001/XMLSchema"], 
      AppendTo[$namespaceQ, 
        "xsd" -> "http://www.w3.org/2001/XMLSchema"];
    ];

    $addMathMLDef = False;
    $addExprDef = False;    
    namespaceID = 1;
    
    children = {
      generateSchemaElement[#, namespace],
      generateSchemaType[#, namespace], 
      generateSchemaAttribute[#, namespace]
    } & /@ symbols;
    children = Select[Flatten[children], # =!= Null&];    

    (* Cleanup for expr and mathml basetypes *)
    If[TrueQ[$addExprDef], 
      PrependTo[children, 
        XMLElement[{getNamespacePrefix["http://www.w3.org/2001/XMLSchema", $namespaceQ], "import"}, 
          {"namespace" -> "http://www.wolfram.com/XML/"}, 
          {}]]
    ];
    If[TrueQ[$addMathMLDef], 
      PrependTo[children, 
        XMLElement[{getNamespacePrefix["http://www.w3.org/2001/XMLSchema", $namespaceQ], "import"}, 
          {"namespace" -> "http://www.w3.org/1998/Math/MathML"}, 
          {}]]
    ];
    
    (* Get the namespace before the namespaceQ is cleared. *)
    xsdNamespace = getNamespacePrefix["http://www.w3.org/2001/XMLSchema", $namespaceQ];
    
    $namespaceQ = Complement[$namespaceQ, namespaces];
    $namespaceQ = $namespaceQ /. {(ns_String -> val_String) -> {"xmlns", ns} -> val};
    attributes = $namespaceQ;
    $namespaceQ = {};
    
    If[$elementDefaultForm =!= Automatic, 
      PrependTo[attributes, {"", "elementDefaultForm"} -> $elementDefaultForm];
    ];
    PrependTo[attributes, {"", "targetNamespace"} -> namespace];
    
    XMLElement[{xsdNamespace, "schema"}, attributes, children]
  ];
  
generateSchemaElement[symbol_?ElementQ, namespace_String] :=
  Module[{attributes = {}, children = {}, type = Null},
  
      (* Add code to check namespace *)
      If[ElementGlobalQ[symbol] && ElementNamespace[symbol] =!= namespace, 
        (* TODO Check to see whether the definition follows $elementDefaultForm 
           Perhaps add an elementForm attribute to the element if it is set and 
           elementDefaultForm is set to unqualified.
         *)
        createSchemaException[GenerateSchema::namespace2, ElementNamespace[symbol], namespace]        
      ];     
       
      (* Name *)
      If[StringQ[ElementLocalName[symbol]], 
        AppendTo[attributes, {"", "name"} -> ElementLocalName[symbol]], 
        createSchemaException[GenerateSchema::namespace2, ElementNamespace[symbol], namespace];
      ];
      
      (* Type *)
      Switch[ElementType[symbol], 
        String | Integer | Real | _Alternatives | SchemaExpr | SchemaMathML | 
        SchemaBase64Binary | SchemaDateTime | SchemaDate | SchemaTime | _?TypeGlobalQ,
          If[ElementType[symbol] === SchemaExpr, 
            If[!MemberQ[$namespaceQ, _String->"http://www.wolfram.com/XML/"], 
              AppendTo[$namespaceQ, 
                "wolfram" -> "http://www.wolfram.com/XML/"];
            ];
            $addExprDef = True
          ];
          If[ElementType[symbol] === SchemaMathML, 
            If[!MemberQ[$namespaceQ, _String->"http://www.w3.org/1998/Math/MathML"], 
              AppendTo[$namespaceQ, 
                "mathml" -> "http://www.w3.org/1998/Math/MathML"];
            ];
            $addMathMLDef = True
          ];
          type = ElementTypeName[symbol];
          If[MatchQ[type, {_String, _String}], 
            AppendTo[attributes, {"", "type"} -> getPrefixedValue@@type],
            type = {TypeNamespace[ElementType[symbol]], TypeLocalName[ElementType[symbol]]};
            If[MatchQ[type, {_String, _String}],            
              AppendTo[attributes, {"", "type"} -> getPrefixedValue@@type], 
              createSchemaException[GenerateSchema::name, ElementType[symbol]]
            ];
          ], 
        Null,    
          createSchemaException[GenerateSchema::elementtype, symbol],
        _,
          AppendTo[children, generateSchemaType[ ElementType[symbol], namespace]]
      ];
      
      (* Min Occurs *)
      If[ElementMinOccurs[symbol] =!= 1, 
        Switch[ElementMinOccurs[symbol],
          _?IntegerQ,
            If[ElementMinOccurs[symbol] > ElementMaxOccurs[symbol], 
              createSchemaException[GenerateSchema::minoccurs2, symbol]
            ];
            AppendTo[attributes, {"", "minOccurs"} -> ToString[ElementMinOccurs[symbol]]], 
          _, 
            createSchemaException[GenerateSchema::minoccurs, symbol, ElementMinOccurs[symbol]]
        ];
      ];
      (* Max Occurs *)
      If[ElementMaxOccurs[symbol] =!= 1, 
        Switch[ElementMaxOccurs[symbol],
          _?IntegerQ, 
            If[ElementMaxOccurs[symbol] < ElementMinOccurs[symbol], 
              createSchemaException[GenerateSchema::maxoccurs2, symbol]
            ];
            AppendTo[attributes, {"", "maxOccurs"} -> ToString[ElementMaxOccurs[symbol]]],
          Infinity,
            AppendTo[attributes, {"", "maxOccurs"} -> "unbounded"],
          _, 
            createSchemaException[GenerateSchema::maxoccurs, symbol, ElementMaxOccurs[symbol]]
        ];
      ];
      
      (* Default Value *)
      If[ElementDefaultValue[symbol] =!= Null,
        Switch[ElementDefaultValue[symbol], 
          _String | _Integer | _Real | True | False | _SchemaExpr | _SchemaMathML | 
          _SchemaBase64Binary | _SchemaDateTime | _SchemaDate | _SchemaTime,         
            AppendTo[attributes, {"", "default"} -> 
              serializeSimpleType[ElementDefaultValue[symbol], ElementType[symbol]]],
          _, 
            createSchemaException[GenerateSchema::default, symbol, ElementDefaultValue[symbol]]
        ];
      ];
      
      (* Fixed Value *)
      If[ElementFixedValue[symbol] =!= Null,
        Switch[ElementFixedValue[symbol], 
          _String | _Integer | _Real | True | False | _SchemaExpr | _SchemaMathML | 
          _SchemaBase64Binary | _SchemaDateTime | _SchemaDate | _SchemaTime,         
            AppendTo[attributes, {"", "fixed"} -> 
              serializeSimpleType[ElementFixedValue[symbol], ElementType[symbol]]],
          _, 
            createSchemaException[GenerateSchema::fixed, symbol, ElementFixedValue[symbol]]
        ];
      ];
      
      XMLElement[{getNamespacePrefix["http://www.w3.org/2001/XMLSchema", $namespaceQ], "element"}, 
                  attributes, 
                  children]
  ];

generateSchemaElement[___] := Null;

generateSchemaType[symbol_?TypeQ, namespace_String] := 
  Module[{attributes = {}, children = {}, contentModel = {}, elements = {}, attrs = {}},

      (* Check namespace *)
      If[TypeGlobalQ[symbol] && TypeNamespace[symbol] =!= namespace, 
        createSchemaException[GenerateSchema::namespace2, TypeNamespace[symbol], namespace]
      ];     

      (* Name *)
      If[TypeGlobalQ[symbol], 
        If[StringQ[TypeLocalName[symbol]], 
          AppendTo[attributes, {"", "name"} -> TypeLocalName[symbol]],
          createSchemaException[GenerateSchema::name, TypeLocalName[symbol]]
        ];
      ];

      (* Elements *)  
      elements = generateSchemaElement[#, namespace] & /@ (List @@ TypeElements[symbol]);
      Switch[Head[TypeElements[symbol]], 
        List, 
          contentModel = {XMLElement[{getNamespacePrefix["http://www.w3.org/2001/XMLSchema", $namespaceQ], "sequence"}, {}, elements]},
        And, 
          contentModel = {XMLElement[{getNamespacePrefix["http://www.w3.org/2001/XMLSchema", $namespaceQ], "all"}, {}, elements]},
        _, 
          createSchemaException[GenerateSchema::typeelements, symbol, TypeElements[symbol]]
      ];
      XMLElement[{getNamespacePrefix["http://www.w3.org/2001/XMLSchema", $namespaceQ], "complexType"}, attributes, contentModel]
  ];  
generateSchemaType[___] := Null;

generateSchemaAttribute[symbol_?AttributeQ, namespace_String] :=
  Module[{attributes = {}, children = {}},
      XMLElement[{getNamespacePrefix["http://www.w3.org/2001/XMLSchema", $namespaceQ], "attribute"}, attributes, children]
  ];
generateSchemaAttribute[___] := Null;

DefineSchema[namespace_String, {definitions___}, context_String:"Global`"] :=
    DefineSchema[namespace, {definitions}, context, True];
    
DefineSchema[namespace_String, {definitions___}, context_String:"Global`", global : (True | False)] := 
  Module[{defs = {definitions}, pattern, default = Null, symbol, args,
          exists, localName, element, arrayType = Null, counter = Null, elements},
    Switch[Head[#],
      (* Elements *)
      Pattern | Optional | Blank | BlankSequence | BlankNullSequence | Symbol |
      Alternatives | Integer | Real | String | List | SchemaDate | SchemaTime | 
      SchemaDateTime,
        (* Optional *)
        If[Head[#] === Optional, 
          If[Length[#] == 2, 
            pattern = First[#];
            default = Last[#]
            ,
            createSchemaException[DefineSchema::pattern, Head[#], #]
          ],
          default = Null;
          pattern = #;
        ];

        (* Pattern *)
        If[Head[pattern] === Pattern, 
          
          If[Length[#] =!= 2,
            createSchemaException[DefineSchema::pattern, Head[#], #]
          ];

          symbol = First[pattern];
          If[!MatchQ[symbol, _Symbol] || MatchQ[symbol, (True | False | Null)],
            createSchemaException[DefineSchema::pattern, Head[pattern], #]
          ];
          localName = SymbolName[symbol];
          If[global,           
            If[MemberQ[Attributes[#]&[symbol], Protected], Unprotect[#]&[symbol]]; 
            clearElement[symbol],
            symbol = Unique[symbol]
          ];
          pattern = Last[pattern],
          (* 'anonymous' element *)
          If[counter === Null, 
            localName = "element";
            counter = 2, 
            localName = "element" <> ToString[counter];
            counter++
          ];
          symbol = getSymbolName[context, localName, False];
          pattern = #;
        ];
        Switch[Head[pattern],
          (* Single element *)
          Blank, 
            Switch[Length[pattern],
              1,
                If[Head[First[pattern]] === Symbol,
                  ElementType[symbol] ^= First[pattern],
                  createSchemaException[DefineSchema::pattern, Head[pattern], #];
                ],
(*              0,
                ElementType[symbol] ^= SchemaExpr,*)
              _, 
                createSchemaException[DefineSchema::pattern, Head[pattern], #];
            ],
          (* 1 or more elements *)
          BlankSequence, 
            If[global, 
              createSchemaException[DefineSchema::occuranceconstraint, #];
            ];
            Switch[Length[pattern],
              1,
                If[Head[First[pattern]] === Symbol,
                  ElementType[symbol] ^= First[pattern],
                  createSchemaException[DefineSchema::pattern, Head[pattern], #];
                ],
(*              0,
                ElementType[symbol] ^= SchemaExpr,*)
              _, 
                createSchemaException[DefineSchema::pattern, Head[pattern], #];
            ];
            ElementMaxOccurs[symbol] ^= Infinity,
          (* 0 or more elements *)
          BlankNullSequence,
            If[global, 
              createSchemaException[DefineSchema::occuranceconstraint, #];
            ];
            Switch[Length[pattern],
              1,
                If[Head[First[pattern]] === Symbol,
                  ElementType[symbol] ^= First[pattern],
                  createSchemaException[DefineSchema::pattern, Head[pattern], #];
                ],
(*              0,
                ElementType[symbol] ^= SchemaExpr,*)
              _, 
                createSchemaException[DefineSchema::pattern, Head[pattern], #];
            ];
            ElementMinOccurs[symbol] ^= 0;
            ElementMaxOccurs[symbol] ^= Infinity,            
          (* Boolean *) 
          Alternatives, 
            If[pattern === (True | False) || pattern === (False | True),
              ElementType[symbol] ^= (True | False), 
              createSchemaException[DefineSchema::pattern, Head[pattern], #];
            ],
          (* Fixed base type values *)
          String | Integer | Real | SchemaDate | SchemaTime | SchemaDateTime, 
            ElementType[symbol] ^= Head[pattern];
            ElementFixedValue[symbol] ^= pattern,
          (* Fixed symbol values *)
          Symbol, 
            If[pattern === True || pattern === False,
              ElementType[symbol] ^= (True | False);
              ElementFixedValue[symbol] ^= pattern,
              createSchemaException[DefineSchema::pattern, pattern, #];
            ], 
          (* Arrays *)  
          List,
            If[Length[pattern] =!= 1, 
              createSchemaException[DefineSchema::pattern, Head[pattern], #];
            ];
            {element} = DefineSchema[namespace, pattern, context, False];
            arrayType = 
              Symbol[context <> SymbolName[ElementType[element]] <> "Array"];
            ElementType[symbol] ^= arrayType;
            If[!TypeGlobalQ[arrayType], 
              If[MemberQ[Attributes[#]&[arrayType], Protected], Unprotect[arrayType]]; 
              TypeGlobalQ[arrayType] ^= True;
              TypeNamespace[arrayType] ^= namespace;
              TypeLocalName[arrayType] ^= SymbolName[arrayType];
              TypeQ[arrayType] ^= True;
              TypeArrayQ[arrayType] ^= True;
              TypeElements[arrayType] ^= List[element];              
            ];
            ,         
          (* Anonymous Type *)
          _, 
            {type} = DefineSchema[namespace, {pattern}, context, False];
            ElementType[symbol] ^= type;
        ];
        If[global, 
          ElementGlobalQ[symbol] ^= True;
          ElementNamespace[symbol] ^= namespace;
          ElementSymbol[namespace, localName] = symbol;
        ];
        ElementLocalName[symbol] ^= localName;
        ElementQ[symbol] ^= True;

        (* set default value - do now because we have the symbol where 
           above we didn't yet *)
        If[default =!= Null, 
          ElementDefaultValue[symbol] ^= default;
        ];        

        setElementDocumentation[symbol];
        Protect[#]&[symbol];
        Protect[#]&[arrayType];
        symbol
        ,
      Repeated | RepeatedNull | PatternTest | Condition | Except | Verbatim | 
      HoldPattern,
        createSchemaException[DefineSchema::pattern, Head[#], #],
      (* Types *)
      _,
        If[Head[List@@#] === List, 
          symbol = Head[#];
          args = List@@#;
          If[args === {None}, 
            args = {};
          ];
          symbol = Head[#];
          If[!MatchQ[symbol, _Symbol] || MatchQ[symbol, (True | False | Null)],
            createSchemaException[DefineSchema::pattern, Head[pattern], #]
          ];
          localName = SymbolName[symbol];
          If[global,           
            If[MemberQ[Attributes[#]&[symbol], Protected], Unprotect[#]&[symbol]]; 
            clearType[symbol],
            symbol = Unique[symbol]
          ];
          elements = DefineSchema[namespace, args, context, False];
          If[global, 
            TypeGlobalQ[symbol] ^= True;
            TypeNamespace[symbol] ^= namespace;
            TypeSymbol[namespace, SymbolName[symbol]] = symbol;
          ];
          TypeQ[symbol] ^= True;
          TypeLocalName[symbol] ^= localName;
          TypeElements[symbol] ^= elements;
          setTypeDocumentation[symbol];
          setTypeAccessor[typeSymbol];
          Protect[#]&[symbol];
          ,
          createSchemaException[DefineSchema::pattern, Head[pattern], #];
        ];
        symbol
    ] & /@ defs
  ]
  
(*** SerializeSchemaInstance an Instance ***)
SerializeSchemaInstance[
  elem_->val_,
  namespaces_List:{}, 
  encoding_List:{}] := 
  Module[{element = elem, value = val, type = Head[val], namespace, xml,  
        localName, name, attributes, children, instanceNamespaces = namespaces, 
        options = Options[val], namespaceReset = False, typeName = Null },

    value = Catch[
      (* Try to convert element into something useful if it is not a symbol *)
      Which[      
        StringQ[element] && NameQ[element],  
          element = Symbol[element],
        MatchQ[element, {_String, _String}],
          If[ElementSymbol@@element =!= Null, 
            element = ElementSymbol@@element;
          ]; 
      ];
    
      (* Process element for useful information *)
      Which[
        ElementQ[element],
          namespace = ElementNamespace[element];
          localName = ElementLocalName[element];
          type = ElementType[element];
          typeName = ElementTypeName[element];
          (* Transform array into standard type definition *)
          If[TypeArrayQ[type], 
            If[MatchQ[TypeElements[type], {_}], 
              If[ListQ[value], 
                value = type[First[TypeElements[type]] -> value]
              ],
              createSchemaException[SerializeSchemaInstance::arraytype, type];
            ];
          ],
        StringQ[element],  
          namespace = "";
          localName = element;
          type = SchemaAnyType,
        MatchQ[element, {_String, _String}],
          namespace = First[element];
          localName = Last[element];
          type = SchemaAnyType,
        Head[element] === Symbol, 
          namespace = "";
          localName = SymbolName[element];
          type = SchemaAnyType,
        True, 
          createSchemaException[SerializeSchemaInstance::element, element];
      ];  

      If[namespaceID === 1, namespaceReset = True];
 
      {attributes} = {"Attributes"} /. options /. {"Attributes"->{}};
      If[!MatchQ[attributes, {___Rule}],
        Message[SerializeSchemaInstance::illegalattributes, attributes];
        attributes = {};
      ];
      instanceNamespaces = (validateNamespaceDefinition /@ instanceNamespaces);
      instanceNamespaces = mapNamespaces[attributes, instanceNamespaces];
      value = DeleteCases[value, "Attributes"->_List];

      (* Name *)
      Which[
        namespace === Null, 
          namespace = "",
        !StringQ[namespace],
          createSchemaException[SerializeSchemaInstance::namespace, namespace, localName];
      ];
      If[!StringQ[localName], 
        createSchemaException[SerializeSchemaInstance::name, localName, element],
        name = {getNamespacePrefix[namespace, instanceNamespaces], localName}
      ];

      (* Attributes *)
      attributes = serializeAttributes[attributes, TypeAttributes[type], instanceNamespaces];
      attributes = Join[attributes, serializeAttributes[$namespaceQ /. {(ns_String->v_String)->({"xmlns", ns}->v)}, {}, {}]];
      instanceNamespaces = Join[instanceNamespaces, $namespaceQ];
      $namespaceQ = {};
 
      If[type === SchemaAnyType, 
        type = Head[value]
      ];
 
      (* Value *)      
      Switch[type,
        String | Integer | Real | Symbol | Verbatim[(True | False)]| 
        SchemaBase64Binary | SchemaDateTime | SchemaDate | SchemaTime |
        SchemaMathML | SchemaExpr,
          If[value === Null, 
            value = {},
            value = {serializeSimpleType[value, type]}
          ],
        {String} | {Integer} | {Real} | {SchemaDateTime} | {SchemaDate} | {SchemaTime}, 
          If[value === Null, 
            value = {},
            value = {ExportString[serializeSimpleType[#, First[type]] & /@ value, "Words"]}
          ],
        XMLElement,
          If[value === Null, 
            value = {},
            value = {value}
          ],
        _,
          If[value === Null,
            (* TODO Add a nil attribute if nillable and serializeComplexTypes*) 
            value = {},
            If[Head[value] =!= type, 
              createSchemaException[SerializeSchemaInstance::value, value, type]
            ]
          ];
          If[!MatchQ[value, _[___Rule]], 
            createSchemaException[SerializeSchemaInstance::compoundtype, value];
          ];        
          {children, value} = serializeComplexTypes[List @@ value, TypeElements[type], instanceNamespaces, encoding];
          If[Length[children] > 0, 
            createSchemaException[SerializeSchemaInstance::children, children];
          ];          
      ];
      xml = XMLElement[name, attributes, value];
      If[MemberQ[encoding, "http://schemas.xmlsoap.org/soap/encoding/"],
        soapEncode[xml, If[typeName === Null, type, typeName], instanceNamespaces], 
        xml
      ]      
    ];
    If[namespaceReset, namespaceID = 1];
    If[MatchQ[value, _SchemaException], Throw[value]];
    value
  ];

soapEncode[XMLElement[name_, attributes:{___Rule}, value_], type_, namespaces_List] :=
  Module[{attrs = attributes, typeNamespace, typeName}, 
    If[MatchQ[type, {_String, _String}],
      {typeNamespace, typeName} = type, 
      If[!StringQ[TypeNamespace[type]] || !StringQ[TypeLocalName[type]] || TypeNamespace[type] === "", 
        createSchemaException[SerializeSchemaInstance::type, type], 
        typeNamespace = TypeNamespace[type];
        typeName = TypeLocalName[type];
      ];
    ];
    AppendTo[attrs, {getNamespacePrefix["http://www.w3.org/2001/XMLSchema-instance", namespaces], "type"}->
      getNamespacePrefix[typeNamespace, namespaces] <> ":" <> typeName];
    If[MatchQ[SOAPArrayType[type], {_String, _String}], 
      AppendTo[attrs, {getNamespacePrefix["http://schemas.xmlsoap.org/soap/encoding/", namespaces], "arrayType"}->
        getNamespacePrefix[First[SOAPArrayType[type]], namespaces] <> ":" <> Last[SOAPArrayType[type]]];
    ];
    XMLElement[name, attrs, value]
  ]

serializeSimpleType[value_, type_] := 
  Module[{format, val = value},
    If[!validateSimpleType[value, type], 
      createSchemaException[SerializeSchemaInstance::value, value, type];
    ];
    Switch[value,
      _String?StringQ, 
        value,
      True, 
        "true",
      False, 
        "false",  
      _Integer?IntegerQ,
        ToString[value],
      _Real?NumberQ,
        ToString[NumberForm[value, 16, NumberFormat -> (If[#3 == "", #1,SequenceForm[#1, "E", #3]] &)]],
      Infinity, 
        "INF",
      -Infinity, 
        "-INF", 
      SchemaNaN, 
        "NaN", 
      SchemaBase64Binary[{___Integer}],
        base64Encode[First[value]],
      SchemaDateTime[_Integer, _Integer, _Integer, _Integer, _Integer, _Real | _Integer],
        StringReplace[DateString[List@@value, {"Year", "-", "Month", "-", "Day", "T", "Hour24", ":", "Minute", ":", "Second", If[FractionalPart[value[[6]]] == 0, "", StringDrop[ToString[FractionalPart[value[[6]]]],1]], "TimeZone"}], "GMT"-> ""],
      SchemaDateTime[{_Integer, _Integer, _Integer, _Integer, _Integer, _Real | _Integer}],
        StringReplace[DateString[First[value], {"Year", "-", "Month", "-", "Day", "T", "Hour24", ":", "Minute", ":", "Second", If[FractionalPart[value[[1,6]]] == 0, "", StringDrop[ToString[FractionalPart[value[[1, 6]]]],1]], "TimeZone"}], "GMT"-> ""],
      SchemaDate[_Integer, _Integer, _Integer, _Integer, _Integer, _Real | _Integer],
        DateString[List@@value, {"Year", "-", "Month", "-", "Day"}],
      SchemaDate[{_Integer, _Integer, _Integer, _Integer, _Integer, _}],
        DateString[First[value], {"Year", "-", "Month", "-", "Day"}],
      SchemaDate[_Integer, _Integer, _Integer],
        DateString[Join[ List@@value, {0,0,0}], {"Year", "-", "Month", "-", "Day"}],
      SchemaDate[{_Integer, _Integer, _Integer}],
        DateString[Join[ First[value], {0,0,0}], {"Year", "-", "Month", "-", "Day"}],
      SchemaTime[_Integer, _Integer, _Integer, _Integer, _Integer, _Real | _Integer],
        StringReplace[DateString[List@@value, {"Hour24", ":", "Minute", ":", "Second", If[FractionalPart[value[[6]]] == 0, "", StringDrop[ToString[FractionalPart[value[[6]]]], 1]], "TimeZone"}], "GMT"-> ""],
      SchemaTime[{_Integer, _Integer, _Integer, _Integer, _Integer, _}],
        StringReplace[DateString[First[value], {"Hour24", ":", "Minute", ":", "Second", If[FractionalPart[value[[1, 6]]] == 0, "", StringDrop[ToString[FractionalPart[value[[1,6]]]], 1]], "TimeZone"}], "GMT"-> ""],
      SchemaTime[_Integer, _Integer, _Real | _Integer],
        StringReplace[DateString[Join[{1900,1,1}, List@@value], {"Hour24", ":", "Minute", ":", "Second", If[FractionalPart[value[[3]]] == 0, "", StringDrop[ToString[FractionalPart[value[[3]]]], 1]], "TimeZone"}], "GMT"-> ""],
      SchemaTime[{_Integer, _Integer, _Real | _Integer}],
        StringReplace[DateString[Join[{1900,1,1}, First[value]], {"Hour24", ":", "Minute", ":", "Second", If[FractionalPart[value[[1, 3]]] == 0, "", StringDrop[ToString[FractionalPart[value[[1, 3]]]], 1]], "TimeZone"}], "GMT"-> ""],
      SchemaMathML[XMLElement[(_String | {_String, _String}), {___}, {_XMLElement}]],
        val = XML`ToVerboseXML[First[value]];
        val = val[[3,1]],
      SchemaExpr[_], 
        val = XML`NotebookML`ExpressionToSymbolicExpressionML[value];
        val = XML`ToVerboseXML[val];
        val = Replace[ val, 
        		XMLObject["Document"][_, 
   					XMLElement[_, _, {XMLElement[_, _, {_, x_}]}], _] -> x];
   		If[ Head[val] === XMLObject["Document"], createSchemaException[SerializeSchemaInstance::value, value, type]];
   		val
        ,       
      _,
        createSchemaException[SerializeSchemaInstance::value, value, type];
    ]
  ]
  
serializeComplexTypes[childElements:{___Rule}, (typeElements_List | typeElements_Or), namespaces_List, encoding_List] := 
  Module[{children = childElements, elements = List @@ typeElements, value = {}, childElement, 
          childValue, element, value2, counter, head = Head[typeElements], length},

    While[Length[elements] > 0, 
    
      element = First[elements];
      If[MatchQ[element, (_List  | _Or)], 
        length = Length[children];      
        {children, value2} = serializeComplexTypes[children, element, namespaces, encoding];
        elements = Rest[elements];
        value = Join[value, value2];
        If[length =!= Length[children] && head === Or, 
          Break[],
          Continue[]
        ]
      ];
      
      If[!MatchQ[element, _Symbol], 
        createSchemaException[SerializeSchemaInstance::element, element];
      ];
      
      If[Length[children] < 1, 
        childElement = Null;
        childValue = Null, 
        childElement = First[First[children]];
        childValue = Last[First[children]];
      ];      
         
      If[!StringQ[ElementNamespace[element]],
        createSchemaException[SerializeSchemaInstance::namespace, ElementNamespace[element], element];
      ];      
      If[!StringQ[ElementLocalName[element]],
        createSchemaException[SerializeSchemaInstance::name, ElementLocalName[element], element];
      ];
            
      (* Check Name *)
      childElement = getElementSymbol[childElement->childValue, element];
      If[childElement === Null, childValue = Null];
      (* Check Occurrance Constraints *)      
      Which[
        childElement === Null, 
          If[ElementMinOccurs[element] > 0 && head === List, 
            createSchemaException[SerializeSchemaInstance::requiredelement, ElementLocalName[element]];
          ],
        ElementMaxOccurs[element] < 1, 
          If[head === List, 
            createSchemaException[SerializeSchemaInstance::prohibitedelement, ElementLocalName[element]]
          ],
        ElementFixedValue[element] =!= Null, 
          Which[
            ElementFixedValue[element] === childValue || childValue === Null, 
              AppendTo[value, SerializeSchemaInstance[element->ElementFixedValue[element], namespaces, encoding]];
              children = Rest[children];
              If[head === Or, 
                Break[]
              ],
            True, 
              createSchemaException[SerializeSchemaInstance::fixedvalue, ElementLocalName[element], ElementFixedValue[element], childValue];
          ],
        ElementMaxOccurs[element] > 1, 
          If[ListQ[childValue], 
            If[ElementMaxOccurs[element] < Length[childValue], 
              createSchemaException[SerializeSchemaInstance::maxoccurs, ElementLocalName[element], ElementMaxOccurs[element]];
            ];
            AppendTo[value, SerializeSchemaInstance[element->Replace[#, Null->ElementDefaultValue[element]], namespaces, encoding]] & /@ childValue;
            children = Rest[children];
            If[head === Or, 
              Break[]
            ]
            ,
            If[childValue === Null, childValue = ElementDefaultValue[element]];
            AppendTo[value, SerializeSchemaInstance[element->childValue, namespaces, encoding]];
            children = Rest[children];
            counter = 1;
            While[Length[children] > 0, 
              childElement = getElementSymbol[First[First[children]]->Last[First[children]], element];
              If[childElement =!= Null,
                If[ElementMaxOccurs[element] === counter, 
                  createSchemaException[SerializeSchemaInstance::maxoccurs, ElementLocalName[element], ElementMaxOccurs[element]];
                  ,
                  If[Last[First[children]] === Null, 
                    childValue = ElementDefaultValue[element],
                    childValue = Last[First[children]]
                  ];
                  AppendTo[value, SerializeSchemaInstance[element->childValue, namespaces, encoding]];
                  children = Rest[children];
                  counter++
                ]; 
                , 
                Break[];
              ];
            ];      
            If[head === Or, 
              Break[]
            ]
          ],
        True,  
          If[childValue === Null, childValue = ElementDefaultValue[element]];
          AppendTo[value, SerializeSchemaInstance[element->childValue, namespaces, encoding]];
          children = Rest[children];
          If[head === Or, 
            Break[]
          ]
      ];
      elements = Rest[elements];
    ];
    {children, value}
  ];
  
serializeComplexTypes[childElements:{___Rule}, typeElements_And, namespaces_List, encoding_List] := 
  Module[{children = childElements, elements = List @@ typeElements, value = {}, childElement, 
          childValue, element},
    While[Length[children] > 0, 
      childElement = First[First[children]];
      childValue = Last[First[children]];

      (* get element and remove it from the list *)
      elementSymbols = {};
      element = Null;
      For[i = 1, i <= Length[elements], i++,  
        If[element === Null, 
          element = getElementSymbol[childElement->childValue, elements[[i]]];
          If[element === Null,         
            AppendTo[elementSymbols, elements[[i]]]
          ], 
          AppendTo[elementSymbols, elements[[i]]]
        ]
      ];
      elements = elementSymbols;
      If[element === Null, 
        createSchemaException[SerializeSchemaInstance::children, childElement->childValue]
      ];
                  
      (* Check Occurrance Constraints *)      
      Which[
        ElementMaxOccurs[element] < 1, 
          createSchemaException[SerializeSchemaInstance::prohibitedelement, ElementLocalName[element]],
        ElementFixedValue[element] =!= Null, 
          Which[
            ElementFixedValue[element] === childValue || childValue === Null, 
              AppendTo[value, SerializeSchemaInstance[element->ElementFixedValue[element], namespaces, encoding]],
            True, 
              createSchemaException[SerializeSchemaInstance::fixedvalue, ElementLocalName[element], ElementFixedValue[element], childValue];
          ],
        True,  
          If[childValue === Null, childValue = ElementDefaultValue[element]];
          AppendTo[value, SerializeSchemaInstance[element->childValue, namespaces, encoding]];
      ];
      children = Rest[children];
    ];
    If[Length[elements] > 0, 
      If[ElementMinOccurs[#] > 0, 
        createSchemaException[SerializeSchemaInstance::requiredelement, ElementLocalName[#]]
      ] & /@ elements
    ];
    {{}, value}
  ];

serializeComplexTypes[childElements:{___Rule}, typeElements_, namespaces_List, encoding_List] := 
  createSchemaException[SerializeSchemaInstance::typeelements, typeElements]
  
serializeAttributes[attributes_List, typeAttributes_List, namespaces_List] := 
  Module[{typeAttrs = typeAttributes, attrs = attributes, value = {}, 
          typeAttr, attr, attributeSymbols, type = SchemaAnyType},
    While[attrs =!= {}, 
    
      attr = First[attrs]; 
                 
      (* get element and remove it from the list *)
      attributeSymbols = {};
      typeAttr = Null;
      For[i = 1, i <= Length[typeAttrs], i++,  
        If[typeAttr === Null, 
          typeAttr = getElementSymbol[attr, typeAttrs[[i]]];
          If[typeAttr === Null,         
            AppendTo[attributeSymbols, typeAttrs[[i]]]
          ], 
          AppendTo[attributeSymbols, typeAttrs[[i]]]
        ]
      ];
      typeAttrs = attributeSymbols;
      
      (* Check Occurrance Constraints *)      
      Which[
        typeAttr === Null, 
          (* TODO allow certain attributes not all attributes *)
          Switch[First[attr],
            _String, 
              attrName = {"", First[attr]},
            {_String, _String}, 
              attrName = {getNamespacePrefix[First[First[attr]], namespaces], Last[First[attr]]},
            _, 
              attrName = {AttributeNamespace[First[attr]], AttributeLocalName[First[attr]]};
              type = AttributeType[First[attr]];
              If[!MatchQ[attrName, {_String, _String}],
                Message[SerializeSchemaInstance::illegalname, attrName];
                attrName = Null,
                attrName = {getNamespacePrefix[First[attrName], namespaces], Last[attrName]}
              ];
          ];
          If[type === SchemaAnyType, 
            Which[
              MatchQ[Last[attr], (True | False)],
                type = (True | False),
              True,
                type = Head[Last[attr]]
            ];
          ];
          If[attrName =!= Null, 
            AppendTo[value, attrName->serializeSimpleType[Last[attr], type]]
          ],
        AttributeMaxOccurs[typeAttr] > 0, (* Optional or Required *)
          type = AttributeType[typeAttr];
          If[type === SchemaAnyType, 
            type = Head[Last[attr]];
          ];
          AppendTo[value, {getNamespacePrefix[AttributeNamespace[typeAttr], namespaces], AttributeLocalName[typeAttr]}->serializeSimpleType[Last[attr], type]];
          typeAttrs = Rest[typeAttrs],
        AttributeMaxOccurs[typeAttr] < 1, (* Prohibited *)
          Message[SerializeSchemaInstance::prohibitedattribute, {AttributeNamespace[typeAttr], AttributeLocalName[typeAttr]}]
      ];
      attrs = Rest[attrs];      
    ];
    If[Length[typeAttrs] > 0, 
      If[AttributeMinOccurs[#] > 0, 
        createSchemaException[SerializeSchemaInstance::requiredattribute, AttributeLocalName[#]]
      ] & /@ typeAttrs
    ];
    (* Go through and add defaults and fixed and messages for required *)
    value
  ];

getElementSymbol[childElement_->childValue_, element_] := 
  If[elementNameMatchQ[childElement->childValue, ElementNamespace[element], ElementLocalName[element]], 
    element,
    Null
  ];

getElementSymbol[value:XMLElement[elementName_String | {elementNamespace_String, elementName_String}, {___},{___}], element_] := 
  If[elementNameMatchQ[value, ElementNamespace[element], ElementLocalName[element]], 
    element,
    Null
  ];

getElementSymbol[___] := Null; 

elementNameMatchQ[XMLElement[elementName_String, {___}, {___}], 
          namespace_String, name_String] :=
  StringMatchQ[elementName, name];

elementNameMatchQ[XMLElement[{elementNamespace_String, elementName_String}, {___}, {___}], 
          namespace_String, name_String] :=
  StringMatchQ[elementNamespace, namespace] && StringMatchQ[elementName, name];
      
elementNameMatchQ[{ns_String, localName_String}->_, namespace_String, name_String] :=
  StringMatchQ[ns, namespace] && StringMatchQ[localName, name];

elementNameMatchQ[localName_String->_, namespace_String, name_String] :=
  StringMatchQ[localName, name];

elementNameMatchQ[symbol_->_, namespace_String, name_String] :=
  (If[!StringQ[ElementNamespace[symbol]] || !StringQ[ElementLocalName[symbol]], Return[False]];
   StringMatchQ[ElementNamespace[symbol], namespace] && StringMatchQ[ElementLocalName[symbol], name]);

elementNameMatchQ[___] := False;

attributeNameMatchQ[XMLElement[{elementNamespace_String, elementName_String}, {___}, {___}], 
          namespace_String, name_String] :=
  If[namespace === elementNamespace && name === elementName, True, False ];
      
attributeNameMatchQ[{ns_String, localName_String}->_, namespace_String, name_String] :=
  If[namespace === ns && name === localName, True, False ];

attributeNameMatchQ[localName_String->_, namespace_String, name_String] :=
  If[namespace === "" && name === localName, True, False ];

attributeNameMatchQ[symbol_->_, namespace_String, name_String] :=
  If[namespace === AttributeNamespace[symbol] && name === AttributeLocalName[symbol], True, False ];

base64Encode[ bytes:{___Integer} ] :=
  Module[{stream, b},
    LoadJavaClass["org.apache.commons.codec.binary.Base64"];
    FromCharacterCode[Base64`encodeBase64[bytes]]
  ];

base64Decode[str_String] :=
  Module[{stream, bytes},    
    LoadJavaClass["org.apache.commons.codec.binary.Base64"];
    stream = StringToStream[str];
    bytes = BinaryReadList[stream, "Integer8"];
    Close[stream];
    bytes = Base64`decodeBase64[bytes];
    If[# < 0, # + 256, #] & /@ bytes
  ];

DeserializeSchemaInstance[
  XMLElement[
    (_String | {_String, _String}), 
    {___, {"", "href"}->id_String, ___}, 
    {___}], 
  symbol:(_Symbol | Verbatim[(True | False)] | {_Symbol}),
  _List:{},
  _String:""] := 
  
  Module[{e, namespaces, defaultNamespace},
    {e, namespaces, defaultNamespace} = SOAPReference[id];
    If[e === Null, 
      createSchemaException[DeserializeSchemaInstance::href],
      DeserializeSchemaInstance[e, symbol, namespaces, defaultNamespace]
    ]
  ]

DeserializeSchemaInstance[
  item:XMLElement[
    name:(_String | {_String, _String}), 
    attributes:{___}, 
    content:{___}], 
  symbol:(_Symbol | Verbatim[(True | False)] | {_Symbol}),
  namespaces_List:{},
  defaultNamespace_String:""] := 
  
  Module[{type, element = Null, value, attrs = {}, extras = {}, children = content, 
          instanceNamespaces = mapNamespaces[attributes, namespaces], 
          instanceDefaultNamespace = getDefaultNamespace[attributes, defaultNamespace]},
          
    Switch[symbol,
      _?TypeQ, 
        type = symbol,
      _?ElementQ, 
        type = ElementType[symbol];
        If[!TypeQ[type], 
          createSchemaException[DeserializeSchemaInstance::type, type];
        ];
        element = symbol,
      _, 
        createSchemaException[DeserializeSchemaInstance::type, symbol];
    ];
    
    (* If type is SchemaAnyType, get the type from the instance and lookup the type *)
    If[type === SchemaAnyType,
      type = Cases[attributes, Rule[{_?schemaInstanceNamespaceQ, "type"}, val_String] :> val];
      If[Length[type] === 1, 
        type = getQName[First[type], instanceNamespaces, instanceDefaultNamespace];
        type = TypeSymbol@@type,
        type === Null
      ];
      If[type === Null, type = XMLElement];
    ];
         
    Switch[type,
      String | Integer | Real | Verbatim[(True | False)] | SchemaBase64Binary | 
      SchemaDateTime | SchemaDate | SchemaTime, 
        Switch[children,
          {_String}, 
            value = deserializeSimpleType[First[children], type],
          {}, 
            value = Null,
          _,
            createSchemaException[DeserializeSchemaInstance::value, children, type];
        ],
      {String} | {Integer} | {Real} | {SchemaDateTime} | {SchemaDate} | {SchemaTime}, 
        Switch[children,
          {_String}, 
            value = ImportString[First[children], "Words"];
            value = (deserializeSimpleType[#, First[type]] & /@ value),
          {}, 
            value = Null,
          _,
            createSchemaException[DeserializeSchemaInstance::value, children, type];
        ],
      SchemaMathML,
        value = SchemaMathML[XMLElement[{"http://www.w3.org/1998/Math/MathML", "math"}, {}, children]],
      SchemaExpr,
        value = deserializeExpr[children],
      XMLElement, 
        value = item,
      _,
        If[!MatchQ[children, {XMLElement[(_String | {_String, _String}), {___Rule}, {(___XMLElement | _String)}]...}], 
          createSchemaException[DeserializeSchemaInstance::children, children];
        ]; 
        (* TODO Check against the element definition to see if it is nillable *) 
        If[MatchQ[attributes, {___, {"http://www.w3.org/2001/XMLSchema-instance", "nil"}->"true", ___}], 
          value = Null,
          {children, value} = deserializeComplexTypes[children, TypeElements[type], instanceNamespaces, instanceDefaultNamespace];
          If[Length[children] > 0, 
            createSchemaException[DeserializeSchemaInstance::children, children];
          ];
        ];   

        If[!MatchQ[attributes, {___Rule}], 
          createSchemaException[DeserializeSchemaInstance::attributes, attributes];
        ]; 
        attrs = deserializeAttributes[attributes, TypeAttributes[type]];
        If[Length[attrs] > 0, AppendTo[extras, "Attributes"->attrs]];

        (* Make arrays look like arrays *)
        If[TypeArrayQ[type], 
          Switch[value, 
            {_Rule}, 
              value = Last[First[value]], 
            {}, 
              value = {}, 
            _,               
              createSchemaException[DeserializeSchemaInstance::array, value]
          ],
          If[value === {}, 
            value = Null, 
            value = type @@ Join[value, extras];
          ];
        ];
    ];
    If[element === Null,
      value,
      If[ElementLocalName[element] === Null, 
        createSchemaException[DeserializeSchemaInstance::name, Null, element],
        ElementLocalName[element]->value
      ]
    ]  
  ];
    
deserializeSimpleType[value_String, type_] := 
  Module[{val},
    Switch[type, 
      String, 
        value,
      Integer,
        val = ToExpression[value, InputForm, HoldComplete];
	    If[MatchQ[val, HoldComplete[_Integer]], 
	      ReleaseHold[val], 
          createSchemaException[DeserializeSchemaInstance::value, value, Integer];
        ],
      Real,
        Switch[value, 
          "INF", 
            Infinity,
          "-INF", 
            -Infinity,
          "NaN",
            SchemaNaN,
          _, 
            val = ToExpression[StringReplace[value, "E" -> "*^", IgnoreCase -> True], InputForm, HoldComplete];
	        If[MatchQ[val, HoldComplete[(_Real | _Integer)]], 
	          N[ReleaseHold[val]],
              createSchemaException[DeserializeSchemaInstance::value, value, Real];
            ]
          ],
      Verbatim[(True | False)], 
        If[StringMatchQ[value, "true", IgnoreCase -> True],
          True,
          False
        ],
      SchemaBase64Binary,
        SchemaBase64Binary[base64Decode[value]],
      SchemaDateTime, 
        val = parseISODateTime[value,  {"Year", "-", "Month", "-", "Day", "T", "Hour24", ":", "Minute", ":", "Second", "TimeZone"}];
        If[val === $Failed, 
          createSchemaException[DeserializeSchemaInstance::value, value, SchemaDateTime],
          SchemaDateTime@@val
        ],
      SchemaDate,
        val = parseISODateTime[value, {"Year", "-", "Month", "-", "Day"}];
        If[val === $Failed, 
          createSchemaException[DeserializeSchemaInstance::value, value, SchemaDate],
          SchemaDate@@Take[val, {1, 3}]
        ],
      SchemaTime,
        val = parseISODateTime[value, {"Hour24",":","Minute",":","Second","TimeZone"}];
        If[val === $Failed, 
          createSchemaException[DeserializeSchemaInstance::value, value, SchemaTime],
          SchemaTime@@Take[val, {4, 6}]
        ],
      _,
        createSchemaException[DeserializeSchemaInstance::value, value, type];
    ]
  ];
  
parseISODateTime[dt_String,pattern_List]:=
  Module[{fractional, datetime}, 
    fractional = StringCases[dt, RegularExpression["(\\.\\d+)"]:>ToExpression["$1"]];
    datetime = StringReplace[dt, {RegularExpression["(\\.\\d+)"]->""}];
    datetime = 
      StringReplace[datetime, {
        "Z"->"GMT+00:00", 
        RegularExpression["([+-]\\d\\d:\\d\\d)"]:>"GMT"<>"$1",                  
        RegularExpression["(\\d\\d:\\d\\d:\\d\\d)$"]->
          StringJoin["$1", "GMT", 
            ToString[PaddedForm[-6 (* Should use TimeZone[] *), 2,NumberPadding->{"0", "0"}, SignPadding->True]],":00"]}];
    If[$VersionNumber < 6, 
      datetime = FromDateString[datetime, pattern],
      datetime = DateList[{datetime, pattern}]      
    ];
    If[datetime =!= $Failed && MatchQ[fractional, {_Real}], 
      ReplacePart[datetime, datetime[[6]] + First[fractional],6],
      datetime
    ]
  ]
    
(* sequence content model *)
deserializeComplexTypes[
  childElements:{___XMLElement}, 
  (typeElements_List | typeElements_Or), 
  namespaces_List, 
  defaultNamespace_String
  ] := 
  Module[{elements = List @@ typeElements, children = childElements, value = {}, element, 
          childElement, child, list, value2, counter, head = Head[typeElements], length},
    While[Length[elements] > 0, 
      element = First[elements];
      
      If[MatchQ[element, (_List | _Or )], 
        length = Length[children];
        {children, value2} = deserializeComplexTypes[children, element, namespaces, defaultNamespace];
        elements = Rest[elements];
        value = Join[value, value2];
        If[length =!= Length[children] && head === Or, 
          Break[],
          Continue[]
        ]
      ];
      
      If[!MatchQ[element, _Symbol], 
        createSchemaException[DeserializeSchemaInstance::element, element];
      ];

      If[Length[children] < 1, 
        child = Null, 
        child = First[children];
      ];      
         
      If[!StringQ[ElementNamespace[element]],
        createSchemaException[DeserializeSchemaInstance::namespace, ElementNamespace[element], element];
      ];      
      If[!StringQ[ElementLocalName[element]],
        createSchemaException[DeserializeSchemaInstance::name, ElementLocalName[element], element];
      ];

      childElement = getElementSymbol[child, element];
      If[childElement === Null, child = Null];
            
      (* Check Occurrance Constraints *)      
      Which[
        child === Null, 
          If[ElementMinOccurs[element] > 0 && head === List, 
            createSchemaException[DeserializeSchemaInstance::requiredelement, ElementLocalName[element]]
          ],
        ElementMaxOccurs[element] < 1, 
          If[head === List, 
            createSchemaException[DeserializeSchemaInstance::prohibitedelement, ElementLocalName[element]]
          ],
        ElementFixedValue[element] =!= Null, 
          child = DeserializeSchemaInstance[child, element, namespaces, defaultNamespace];
          Which[
            ElementFixedValue[element] === Last[child] || Last[child] === Null, 
              AppendTo[value, ElementLocalName[element]->ElementFixedValue[element]];
              children = Rest[children];
              If[head === Or, 
                Break[]
              ],
            True, 
              createSchemaException[DeserializeSchemaInstance::fixedvalue, ElementLocalName[element], ElementFixedValue[element], Last[child]];
          ],
        ElementMaxOccurs[element] > 1,           
          list = {};
          child = DeserializeSchemaInstance[child, ElementType[element], namespaces, defaultNamespace];
          If[child === Null, child = ElementDefaultValue[element]];
          AppendTo[list, child];
          children = Rest[children];
          counter = 1;
          While[Length[children] > 0, 
            childElement = getElementSymbol[First[children], element];
            Which[
              childElement === Null,
                Break[],
              ElementMaxOccurs[element] === counter, 
                createSchemaException[DeserializeSchemaInstance::maxoccurs, ElementLocalName[element], ElementMaxOccurs[element]],
              True,
                child = DeserializeSchemaInstance[First[children], ElementType[element], namespaces, defaultNamespace];
                If[child === Null, child = ElementDefaultValue[element]];
                AppendTo[list, child];
                children = Rest[children];
                counter++
            ];
          ];
          AppendTo[value, ElementLocalName[element]->list];
          If[head === Or, 
            Break[]
          ],
        True,  
          child = DeserializeSchemaInstance[child, element, namespaces, defaultNamespace];
          If[Last[child] === Null, 
            AppendTo[value, ElementLocalName[element]->ElementDefaultValue[element]], 
            AppendTo[value, child];
          ];                    
          children = Rest[children];
          If[head === Or, 
            Break[]
          ]
      ];
      elements = Rest[elements];
    ];
    {children, value}
  ];
  
(* all content model *)
deserializeComplexTypes[
  childElements:{___XMLElement}, 
  typeElements_And,
  namespaces_List,
  defaultNamespace_String] := 
  Module[{elements = List @@ typeElements, children = childElements, 
          value = {}, element, child, elementSymbols},
    While[Length[children] > 0, 
      child = First[children];

      (* get element and remove it from the list *)
      elementSymbols = {};
      element = Null;
      For[i = 1, i <= Length[elements], i++,  
        If[element === Null, 
          element = getElementSymbol[child, elements[[i]]];
          If[element === Null,         
            AppendTo[elementSymbols, elements[[i]]]
          ], 
          AppendTo[elementSymbols, elements[[i]]]
        ]
      ];
      elements = elementSymbols;
      If[element === Null, 
        createSchemaException[DeserializeSchemaInstance::children, child]
      ];
                  
      (* Check Occurrance Constraints *)      
      Which[
        ElementMaxOccurs[element] < 1, 
          createSchemaException[DeserializeSchemaInstance::prohibitedelement, ElementLocalName[element]],
        ElementFixedValue[element] =!= Null, 
          child = DeserializeSchemaInstance[child, element, namespaces, defaultNamespace];
          Which[
            ElementFixedValue[element] === Last[child] || Last[child] === Null, 
              AppendTo[value, ElementLocalName[element]->ElementFixedValue[element]],
            True, 
              createSchemaException[DeserializeSchemaInstance::fixedvalue, ElementLocalName[element], ElementFixedValue[element], Last[child]];
          ],
        True,  
          child = DeserializeSchemaInstance[child, element, namespaces, defaultNamespace];
          If[Last[child] === Null, 
            AppendTo[value, ElementLocalName[element]->ElementDefaultValue[element]], 
            AppendTo[value, child];
          ];                    
      ];
      children = Rest[children];
    ];
    If[Length[elements] > 0, 
      If[ElementMinOccurs[#] > 0, 
        createSchemaException[DeserializeSchemaInstance::requiredelement, ElementLocalName[#]]
      ] & /@ elements
    ];
    {{}, value}
  ];

deserializeAttributes[attributes_List, typeAttributes_List] := 
  Module[{typeAttrs = typeAttributes, attrs = attributes, value = {}, 
          typeAttr, attr, attributeSymbols},
    While[Length[attrs] > 0, 
      
      attr = First[attrs]; 

      If[MatchQ[First[attr], {_?xmlNamespaceQ, _String} | {_?schemaInstanceNamespaceQ, "type"} | {"", "id"}],
        attrs = Rest[attrs];
        Continue[];
      ];
                 
      (* get element and remove it from the list *)
      attributeSymbols = {};
      typeAttr = Null;
      For[i = 1, i <= Length[typeAttrs], i++,  
        If[typeAttr === Null, 
          typeAttr = getElementSymbol[attr, typeAttrs[[i]]];
          If[typeAttr === Null,         
            AppendTo[attributeSymbols, typeAttrs[[i]]]
          ], 
          AppendTo[attributeSymbols, typeAttrs[[i]]]
        ]
      ];
      typeAttrs = attributeSymbols;
            
      (* Check Occurrance Constraints *)      
      Which[
        typeAttr === Null, 
          AppendTo[value, First[attr]->deserializeSimpleType[Last[attr], String]],
        AttributeMaxOccurs[typeAttr] > 0, (* Optional and Required *)
          AppendTo[value, typeAttr->deserializeSimpleType[Last[attr], AttributeType[typeAttr]]],
        AttributeMaxOccurs[typeAttr] < 1, (* Prohibited *)
          Message[DeserializeSchemaInstance::prohibitedattribute, {AttributeNamespace[typeAttr], AttributeLocalName[typeAttr]}]
      ];
      attrs = Rest[attrs];
    ];
    If[Length[typeAttrs] > 0, 
      If[AttributeMinOccurs[#] > 0, 
        createSchemaException[DeserializeSchemaInstance::requiredattribute, AttributeLocalName[#]]
      ] & /@ typeAttrs
    ];
    value
  ];
  
deserializeExpr[value_List]:=
  Module[{e},
    Block[{n, val},
      e = First[value] /. 
        XMLElement[n:(_String | {_String, _String}), _List, val_List] -> XMLElement[n, {}, val];
    ];      
    e = XMLElement[{"http://www.wolfram.com/XML/", "Expression"}, {}, {
          XMLElement[{"http://www.wolfram.com/XML/", "Function"}, {}, 
           {XMLElement[{"http://www.wolfram.com/XML/", "Symbol"}, {}, {"HoldComplete"}], e}]}];
    e = XML`NotebookML`SymbolicExpressionMLToExpression[e];
    If[e === $Failed || e === HoldComplete[Null],
      createSchemaException[DeserializeSchemaInstance::value, value, SchemaExpr];
    ];
    If[securityFails[e] === True,
      createSchemaException[DeserializeSchemaInstance::insecure, e],
      e = Apply[ SchemaExpr, e]
    ]
  ];  

securityFails[ e_] :=
	If[ValueQ[$SchemaExprSecurityFunction],
		$SchemaExprSecurityFunction[InsecureExprQ[e]],
		InsecureExprQ[e]]


ValidateSchemaInstance[element_Symbol -> val_] := ValidateSchemaInstance[val, ElementType[element]]

ValidateSchemaInstance[element_String -> val_] := ValidateSchemaInstance[val, ElementType[Symbol[element]]]

ValidateSchemaInstance[element:{_String, _String} -> val_] := ValidateSchemaInstance[val, ElementType[ElementSymbol@@element]]

ValidateSchemaInstance[val_] := ValidateSchemaInstance[val, SchemaAnyType]

(*** ValidateSchemaInstance ***)
ValidateSchemaInstance[val_, symbol:(_Symbol | Verbatim[(True | False)] | {_Symbol})] := 
  Module[{value = val, type, attributes, options = Options[val]},
    value = Catch[
      Switch[symbol,
        _?TypeQ, 
          type = symbol,
        _?ElementQ, 
          type = ElementType[symbol];
          If[!TypeQ[type], 
            createSchemaException[ValidateSchemaInstance::type, type];
          ],
        _, 
          createSchemaException[ValidateSchemaInstance::type, symbol];
      ];

      (* Transform array into standard type definition *)
      If[TypeArrayQ[type], 
        If[MatchQ[TypeElements[type], {_}], 
          If[ListQ[value], 
            value = type[First[TypeElements[type]] -> value]
          ],
          createSchemaException[ValidateSchemaInstance::arraytype, type];
        ];
      ];

(* TODO attributes
      {attributes} = {"Attributes"} /. options /. {"Attributes"->{}};
      If[!MatchQ[attributes, {___Rule}],
        Message[ValidateSchemaInstance::illegalattributes, attributes];
        attributes = {};
      ];
      If[validateAttributes[attributes, TypeAttributes[type]], Return[False]];
      value = DeleteCases[value, "Attributes"->_List];
*)
      (* If type is SchemaAnyType, get the type from the instance and lookup the type *)
      If[type === SchemaAnyType,
        (* TODO lists/arrays *)
        (* TODO Null *)
        Which[
          MatchQ[value, (True | False)],
            type = (True | False),
          True,
            type = Head[value]
        ];
      ];
         
      Switch[type,
        String | Integer | Real | Verbatim[(True | False)] | SchemaBase64Binary | 
        SchemaDateTime | SchemaDate | SchemaTime | SchemaMathML | SchemaExpr | XMLElement, 
          If[value === Null, 
            value = True, 
            value = validateSimpleType[value, type]
          ],
        {String} | {Integer} | {Real} | {SchemaDateTime} | {SchemaDate} | {SchemaTime}, 
          If[value === Null, 
            value = True,
            value = And @@ (validateSimpleType[#, First[type]] & /@ value)
          ],
        _,
          If[value === Null,
            value = {}
          ];
          If[!MatchQ[value, _[___Rule]], 
              createSchemaException[ValidateSchemaInstance::compoundtype, value]
          ];
          {children, value} = validateComplexTypes[List @@ value, TypeElements[type]];
          If[Length[children] > 0, 
            createSchemaException[ValidateSchemaInstance::children, children];
          ];          
      ];
      value
    ];
    If[MatchQ[value, _SchemaException], 
      False,
      value
    ]
  ];

validateSimpleType[value_, type_] := 
  Which[
    type === String && StringQ[value] || 
    type === (True | False) && value === True ||
    type === (True | False) && value === False ||
    type === Integer && IntegerQ[value] ||
    type === Real && NumberQ[value] || 
    type === Real && value === Infinity || 
    type === Real && value === -Infinity || 
    type === Real && value === SchemaNaN || 
    type === SchemaBase64Binary && MatchQ[value, SchemaBase64Binary[{___Integer}]] ||      
    type === SchemaDateTime && MatchQ[value, SchemaDateTime[_Integer, _Integer, _Integer, _Integer, _Integer, _Real | _Integer]] ||
    type === SchemaDateTime && MatchQ[value, SchemaDateTime[{_Integer, _Integer, _Integer, _Integer, _Integer, _Real | _Integer}]] ||      
    type === SchemaDate && MatchQ[value, SchemaDate[_Integer, _Integer, _Integer, _Integer, _Integer, _Real | _Integer]] ||
    type === SchemaDate && MatchQ[value, SchemaDate[{_Integer, _Integer, _Integer, _Integer, _Integer, _}]] ||
    type === SchemaDate && MatchQ[value, SchemaDate[_Integer, _Integer, _Integer]] ||
    type === SchemaDate && MatchQ[value, SchemaDate[{_Integer, _Integer, _Integer}]] ||
    type === SchemaTime && MatchQ[value, SchemaTime[_Integer, _Integer, _Integer, _Integer, _Integer, _Real | _Integer]] ||
    type === SchemaTime && MatchQ[value, SchemaTime[{_Integer, _Integer, _Integer, _Integer, _Integer, _}]] ||
    type === SchemaTime && MatchQ[value, SchemaTime[_Integer, _Integer, _Real | _Integer]] ||
    type === SchemaTime && MatchQ[value, SchemaTime[{_Integer, _Integer, _Real | _Integer}]] || 
    type === SchemaMathML && MatchQ[value, SchemaMathML[_]] && XML`SymbolicXMLQ[First[value], True] ||
    type === SchemaExpr && MatchQ[value, SchemaExpr[_]] || 
    type === XMLElement && XML`SymbolicXMLQ[value, True],  
      True,
    True,
      createSchemaException[ValidateSchemaInstance::value, value, type];    
  ]
  
validateComplexTypes[childElements:{___Rule}, (typeElements_List | typeElements_Or)] := 
  Module[{children = childElements, elements = List @@ typeElements, value = True, childElement, 
          childValue, element, value2, counter, head = Head[typeElements], length},

    While[Length[elements] > 0, 
    
      element = First[elements];
      If[MatchQ[element, (_List  | _Or)], 
        length = Length[children];      
        {children, value2} = validateComplexTypes[children, element];
        elements = Rest[elements];
        value = And[value, value2];
        If[length =!= Length[children] && head === Or, 
          Break[],
          Continue[]
        ]
      ];
      
      If[!MatchQ[element, _Symbol], 
        createSchemaException[ValidateSchemaInstance::element, element];
      ];
      
      If[Length[children] < 1, 
        childElement = Null;
        childValue = Null, 
        childElement = First[First[children]];
        childValue = Last[First[children]];
      ];      
         
      If[!StringQ[ElementNamespace[element]],
        createSchemaException[ValidateSchemaInstance::namespace, ElementNamespace[element], element];
      ];      
      If[!StringQ[ElementLocalName[element]],
        createSchemaException[ValidateSchemaInstance::name, ElementLocalName[element], element];
      ];
            
      (* Check Name *)
      childElement = getElementSymbol[childElement->childValue, element];
      If[childElement === Null, childValue = Null];
      (* Check Occurrance Constraints *)      
      Which[
        childElement === Null, 
          If[ElementMinOccurs[element] > 0 && head === List, 
            createSchemaException[ValidateSchemaInstance::requiredelement, ElementLocalName[element]];
          ],
        ElementMaxOccurs[element] < 1, 
          If[head === List, 
            createSchemaException[ValidateSchemaInstance::prohibitedelement, ElementLocalName[element]]
          ],
        ElementFixedValue[element] =!= Null, 
          Which[
            ElementFixedValue[element] === childValue || childValue === Null, 
              value = And[value, ValidateSchemaInstance[element->ElementFixedValue[element]]];
              children = Rest[children];
              If[head === Or, 
                Break[]
              ],
            True, 
              createSchemaException[ValidateSchemaInstance::fixedvalue, ElementLocalName[element], ElementFixedValue[element], childValue];
          ],
        ElementMaxOccurs[element] > 1, 
          If[ListQ[childValue], 
            If[ElementMaxOccurs[element] < Length[childValue], 
              createSchemaException[ValidateSchemaInstance::maxoccurs, ElementLocalName[element], ElementMaxOccurs[element]];
            ];
            (value = And[value, ValidateSchemaInstance[element->Replace[#, Null->ElementDefaultValue[element]]]]) & /@ childValue;
            children = Rest[children];
            If[head === Or, 
              Break[]
            ]
            ,
            If[childValue === Null, childValue = ElementDefaultValue[element]];
            value = And[value, ValidateSchemaInstance[element->childValue]];
            children = Rest[children];
            counter = 1;
            While[Length[children] > 0, 
              childElement = getElementSymbol[First[First[children]]->Last[First[children]], element];
              If[childElement =!= Null,
                If[ElementMaxOccurs[element] === counter, 
                  createSchemaException[ValidateSchemaInstance::maxoccurs, ElementLocalName[element], ElementMaxOccurs[element]];
                  ,
                  If[Last[First[children]] === Null, 
                    childValue = ElementDefaultValue[element],
                    childValue = Last[First[children]]
                  ];
                  value = And[value, ValidateSchemaInstance[element->childValue]];
                  children = Rest[children];
                  counter++
                ]; 
                , 
                Break[];
              ];
            ];      
            If[head === Or, 
              Break[]
            ]
          ],
        True,  
          If[childValue === Null, childValue = ElementDefaultValue[element]];
          value = And[value, ValidateSchemaInstance[element->childValue]];
          children = Rest[children];
          If[head === Or, 
            Break[]
          ]
      ];
      elements = Rest[elements];
    ];
    {children, value}
  ];
  
validateComplexTypes[childElements:{___Rule}, typeElements_And] := 
  Module[{children = childElements, elements = List @@ typeElements, value = True, childElement, 
          childValue, element},
    While[Length[children] > 0, 
      childElement = First[First[children]];
      childValue = Last[First[children]];

      (* get element and remove it from the list *)
      elementSymbols = {};
      element = Null;
      For[i = 1, i <= Length[elements], i++,  
        If[element === Null, 
          element = getElementSymbol[childElement->childValue, elements[[i]]];
          If[element === Null,         
            AppendTo[elementSymbols, elements[[i]]]
          ], 
          AppendTo[elementSymbols, elements[[i]]]
        ]
      ];
      elements = elementSymbols;
      If[element === Null, 
        createSchemaException[ValidateSchemaInstance::children, childElement->childValue]
      ];
                  
      (* Check Occurrance Constraints *)      
      Which[
        ElementMaxOccurs[element] < 1, 
          createSchemaException[ValidateSchemaInstance::prohibitedelement, ElementLocalName[element]],
        ElementFixedValue[element] =!= Null, 
          Which[
            ElementFixedValue[element] === childValue || childValue === Null, 
              value = And[value, ValidateSchemaInstance[element->ElementFixedValue[element]]],
            True, 
              createSchemaException[ValidateSchemaInstance::fixedvalue, ElementLocalName[element], ElementFixedValue[element], childValue];
          ],
        True,  
          If[childValue === Null, childValue = ElementDefaultValue[element]];
          value = And[value, ValidateSchemaInstance[element->childValue]];
      ];
      children = Rest[children];
    ];
    If[Length[elements] > 0, 
      If[ElementMinOccurs[#] > 0, 
        createSchemaException[ValidateSchemaInstance::requiredelement, ElementLocalName[#]]
      ] & /@ elements
    ];
    {{}, value}
  ];

validateComplexTypes[childElements:{___Rule}, typeElements_] := 
  createSchemaException[ValidateSchemaInstance::typeelements, typeElements]
  
validateAttributes[attributes_List, typeAttributes_List] := 
  Module[{typeAttrs = typeAttributes, attrs = attributes, value = True, 
          typeAttr, attr, attributeSymbols},
    While[attrs =!= {}, 
    
      attr = First[attrs]; 
                 
      (* get element and remove it from the list *)
      attributeSymbols = {};
      typeAttr = Null;
      For[i = 1, i <= Length[typeAttrs], i++,  
        If[typeAttr === Null, 
          typeAttr = getElementSymbol[attr, typeAttrs[[i]]];
          If[typeAttr === Null,         
            AppendTo[attributeSymbols, typeAttrs[[i]]]
          ], 
          AppendTo[attributeSymbols, typeAttrs[[i]]]
        ]
      ];
      typeAttrs = attributeSymbols;
      
      (* Check Occurrance Constraints *)      
      Which[
        typeAttr === Null, 
          (* TODO allow certain attributes not all attributes *)
          Switch[First[attr],
            _String, 
              attrName = {"", First[attr]},
            {_String, _String}, 
              attrName = {getNamespacePrefix[First[First[attr]], namespaces], Last[First[attr]]},
            _, 
              attrName = {AttributeNamespace[First[attr]], AttributeLocalName[First[attr]]};
              If[!MatchQ[attrName, {_String, _String}],
                Message[ValidateSchemaInstance::illegalname, attrName];
                attrName = Null,
                attrName = {getNamespacePrefix[First[attrName], namespaces], Last[attrName]}
              ];
          ];
          If[attrName =!= Null, 
            value = And[value, validateSimpleType[Last[attr], typeAttr]]
          ],
        AttributeMaxOccurs[typeAttr] > 0, (* Optional or Required *)
          value = And[value, validateSimpleType[Last[attr], typeAttr]];
          typeAttrs = Rest[typeAttrs],
        AttributeMaxOccurs[typeAttr] < 1, (* Prohibited *)
          Message[ValidateSchemaInstance::prohibitedattribute, {AttributeNamespace[typeAttr], AttributeLocalName[typeAttr]}]
      ];
      attrs = Rest[attrs];      
    ];
    If[Length[typeAttrs] > 0, 
      If[AttributeMinOccurs[#] > 0, 
        createSchemaException[ValidateSchemaInstance::requiredattribute, AttributeLocalName[#]]
      ] & /@ typeAttrs
    ];
    (* Go through and add defaults and fixed and messages for required *)
    value
  ];

NewSchemaInstance[element_Symbol?ElementQ] := 
  element->NewSchemaInstance[ElementType[element], {}];
  
NewSchemaInstance[element_Symbol?ElementQ, values:{___Rule}] := 
  element->NewSchemaInstance[ElementType[element], values];

NewSchemaInstance[type_Symbol?TypeQ] := NewSchemaInstance[type, {}];
  
NewSchemaInstance[type_Symbol?TypeQ, values:{___Rule}] := 
  Module[{elements = TypeElements[type], element, val, instance},
    params = (
      (element = #;
       If[ElementFixedValue[element] =!= Null, 
         val = ElementFixedValue[element], 
         val = ElementDefaultValue[element]
       ];
       ElementLocalName[element] -> val) & /@ elements);
    instance = type @@ params;
    ReplaceSchemaInstanceValue[instance, values]
  ]

ReplaceSchemaInstanceValue[instance_, rules:{___Rule}] := 
  Module[{inst = instance}, 
    (inst = ReplaceSchemaInstanceValue[inst, #]) & /@ rules;
    inst
  ]
        
ReplaceSchemaInstanceValue[instance_, element_String -> val_] := 
  Module[{value, type, symbols},
    If[MatchQ[instance, _[___Rule]], 
      (* Value *)
      symbols = Cases[TypeElements[Head[instance]], e_/;StringMatchQ[ElementLocalName[e], element]];
      If[!MatchQ[symbols, {__}], 
        Message[ReplaceSchemaInstanceValue::notfound, element]; 
        Return[instance]
      ];
      value = Replace[instance, {(left_/;(MatchQ[ElementLocalName[left], element] || left === element)->_) -> (left->val)}, 1];
      If[!ValidateSchemaInstance[value], value = instance];
      , 
      Message[ReplaceSchemaInstanceValue::compoundtype, instance];
      value = instance;
    ];
    value
  ] 

setTypeAccessor[typeSymbol_?TypeQ] :=
  With[{typeSymbol = typeSymbol}, 
    typeSymbol[elements___Rule][element_String] :=      
      Module[{cases},
        cases = Cases[{elements}, x_/;(First[x]===element || MatchQ[ElementLocalName[First[x]], element]) :> Last[x]];
        If[Length[cases] > 0, 
          First[cases],
          If[MemberQ[List@@ElementLocalName /@ TypeElements[typeSymbol], element], 
            Null, 
            Message[ReplaceSchemaInstanceValue::notfound, element, typeSymbol];
            $Failed
          ]
        ]
      ]
  ];

(* helper function for creating usage messages *)
setTypeDocumentation[typeSymbol_?TypeQ] :=
  Module[{doc},
    If[TypeDocumentation[typeSymbol] =!= Null, 
      doc = ElementDocumentation[typeSymbol];        
    ];
    doc = 
      "TYPE PROPERTIES\n" <> 
      ToString[TableForm[{
        {"Name", TypeLocalName[typeSymbol]}, 
        {"Namespace", TypeNamespace[typeSymbol]}, 
        {"Global", TypeGlobalQ[typeSymbol]},  
        {"Array", TypeArrayQ[typeSymbol]}
      }, TableSpacing -> {0, 1}]] <> "\n\n" <>
      "TYPE ELEMENTS\n" <> 
      ToString[TableForm[
        Prepend[
          {
            ElementLocalName[#], 
            Which[
              MatchQ[ElementType[#], Verbatim[(True | False)]], "(True | False)", 
              ListQ[ElementType[#]], ToString[ElementType[#]], 
              True, SymbolName[ElementType[#]]], 
            ElementMinOccurs[#], 
            ElementMaxOccurs[#], 
            ElementDefaultValue[#], 
            ElementFixedValue[#],
            SymbolName[#]
          } & /@ List@@TypeElements[typeSymbol],
          {"Name", "Type", "MinOccurs", "MaxOccurs", "Default", "Fixed", "Symbol"}
        ]
      , TableSpacing -> {0, 1}]];
    MessageName[Evaluate[typeSymbol], "usage"] = doc;
  ]  

(* helper functions for clearing symbols *)
clearElement[symbol_] := 
  Module[{tagUnsetMesssageOff = MatchQ[TagUnset::"norep", $Off[_]]},
    Off[TagUnset::"norep"];
    TagUnset[#, ##2[#]] &[symbol, ElementQ];
    TagUnset[#, ##2[#]] &[symbol, ElementGlobalQ];
    TagUnset[#, ##2[#]] &[symbol, ElementNamespace];
    TagUnset[#, ##2[#]] &[symbol, ElementLocalName];
    TagUnset[#, ##2[#]] &[symbol, ElementAppInfo];
    TagUnset[#, ##2[#]] &[symbol, ElementDocumentation];
    If[ElementType[symbol] =!= Null && MatchQ[ElementType[symbol], _Symbol] && !TypeGlobalQ[ElementType[symbol]], 
      If[MemberQ[Attributes[#]&[ElementType[symbol]], Protected], 
        Unprotect[#]&[ElementType[symbol]]
      ]; 
      clearType[ElementType[symbol]];
      Remove[#]&[ElementType[symbol]]
    ];
    TagUnset[#, ##2[#]] &[symbol, ElementType];
    TagUnset[#, ##2[#]] &[symbol, ElementTypeName];
    TagUnset[#, ##2[#]] &[symbol, ElementMaxOccurs];
    TagUnset[#, ##2[#]] &[symbol, ElementMinOccurs];
    TagUnset[#, ##2[#]] &[symbol, ElementDefaultValue];
    TagUnset[#, ##2[#]] &[symbol, ElementFixedValue];
    If[!tagUnsetMessageOff, 
      On[TagUnset::"norep"];
    ]
  ];
  
clearAttribute[symbol_] := 
  Module[{tagUnsetMesssageOff = MatchQ[TagUnset::"norep", $Off[_]]},
    Off[TagUnset::"norep"];
    TagUnset[#, ##2[#]] &[symbol, AttributeQ];
    TagUnset[#, ##2[#]] &[symbol, AttributeGlobalQ];
    TagUnset[#, ##2[#]] &[symbol, AttributeNamespace];
    TagUnset[#, ##2[#]] &[symbol, AttributeLocalName];
    TagUnset[#, ##2[#]] &[symbol, AttributeAppInfo];
    TagUnset[#, ##2[#]] &[symbol, AttributeDocumentation];
    TagUnset[#, ##2[#]] &[symbol, AttributeType];
    TagUnset[#, ##2[#]] &[symbol, AttributeTypeName];
    TagUnset[#, ##2[#]] &[symbol, AttributeMaxOccurs];
    TagUnset[#, ##2[#]] &[symbol, AttributeMinOccurs];
    TagUnset[#, ##2[#]] &[symbol, AttributeDefaultValue];
    TagUnset[#, ##2[#]] &[symbol, AttributeFixedValue];
    If[!tagUnsetMessageOff, 
      On[TagUnset::"norep"];
    ]
  ];

clearType[symbol_] := 
  Module[{tagUnsetMesssageOff = MatchQ[TagUnset::"norep", $Off[_]]},
    Off[TagUnset::"norep"];
    TagUnset[#, ##2[#]] &[symbol, TypeQ];
    TagUnset[#, ##2[#]] &[symbol, TypeGlobalQ];
    TagUnset[#, ##2[#]] &[symbol, TypeNamespace];
    TagUnset[#, ##2[#]] &[symbol, TypeLocalName];
    TagUnset[#, ##2[#]] &[symbol, TypeAppInfo];
    TagUnset[#, ##2[#]] &[symbol, TypeDocumentation];
    If[ListQ[TypeElements[symbol]], 
      removeElement /@ TypeElements[symbol]
    ];
    TagUnset[#, ##2[#]] &[symbol, TypeElements];
    TagUnset[#, ##2[#]] &[symbol, TypeArrayQ];
    TagUnset[#, ##2[#]] &[symbol, SOAPArrayType];
    TagUnset[#, ##2[#]] &[symbol, TypeAttributes];
    If[!tagUnsetMessageOff, 
      On[TagUnset::"norep"];
    ]
  ];

removeElement[symbol_] := 
  (
   If[MemberQ[Attributes[#]&[symbol], Protected], 
     Unprotect[#]&[symbol]
   ]; 
   clearElement[symbol];
   Remove[#]&[symbol]
  );

(* helper function for creating symbols *)
getSymbolName[context_String, name_String, global:(True | False)] :=
  Module[{newName = name},
    
    If[!LetterQ[StringTake[name,1]] || StringMatchQ[name, "xml*", IgnoreCase -> True], 
      newName = "my" <> name;
    ];
        
    newName = StringReplace[newName, {"."->"", "_"->"", "~"->"", "!"->"", " "->"",
                                      "@"->"", "#"->"", "$"->"", "%"->"", "^"->"", 
                                      "&"->"", "*"->"", "("->"", ")"->"", "-"->"", 
                                      "+"->"", "="->"", "{"->"", "["->"", "}"->"", 
                                      "]"->"", "|"->"", "\\"->"", ":"->"", ";"->"",
                                      "\""->"", "\'"->"", "<"->"", ","->"", ">"->"",
                                      "?"->"", "/"->"", " "->""}]  ;
    If[!global, 
      Symbol[context <> newName <> "$" <> ToString[$ModuleNumber++]],	
      Symbol[context <> newName]
    ]
  ];

(* helper function for parsing options *)
SetAttributes[ canonicalOptions, {Listable}];
canonicalOptions[name_Symbol -> val_] := SymbolName[name] -> val;
canonicalOptions[expr___] := expr;

(* mapNamespaces parses namespace definitions from a list of rules and adds them to 
 * the current list of namespaces 
 *)
mapNamespaces[attributes:{___Rule}, namespaces_List] :=
  Join[Cases[attributes, ({_?xmlNamespaceQ, namespace_String}->val_String) :> namespace->val], namespaces]
  
mapNamespaces[___] := {}

(* getDefaultNamespaces parses a default namespace from a list of rules *)
getDefaultNamespace[
  attributes:{___Rule, {_?xmlNamespaceQ, "xmlns"}->val_String, ___Rule}, 
  default_String] := val;

getDefaultNamespace[attributes:{___Rule}, default_String] := default;
  
(* returns whether a string matches the XMLSchema namespace *)
schemaNamespaceQ[x_String] :=   
  StringMatchQ[x, "http://www.w3.org/1999/XMLSchema"  ] ||
  StringMatchQ[x, "http://www.w3.org/2000/XMLSchema"  ] ||
  StringMatchQ[x, "http://www.w3.org/2001/XMLSchema"  ]
  
schemaNamespaceQ[___] := False

(* returns whether a string matches the XMLSchema-instance namespace *)
schemaInstanceNamespaceQ[x_String] :=   
  StringMatchQ[x, "http://www.w3.org/1999/XMLSchema-instance"] ||
  StringMatchQ[x, "http://www.w3.org/2000/XMLSchema-instance"] ||
  StringMatchQ[x, "http://www.w3.org/2001/XMLSchema-instance"]

schemaInstanceNamespaceQ[___] := False

(* returns whether a string matches the XML namespace *)
xmlNamespaceQ[x_String] := 
         StringMatchQ[x, "http://www.w3.org/1999/xmlns/"] || 
         StringMatchQ[x, "http://www.w3.org/2000/xmlns/"] || 
         StringMatchQ[x, "http://www.w3.org/2001/xmlns/"] || 
         StringMatchQ[x, "xmlns"]

xmlNamespaceQ[___] := False

(* returns whether a string matches the qname pattern *)
prefixedQNameQ[x_String] := StringMatchQ[x, "*:*"]

prefixedQNameQ[___] := False

(* returns a list containing the namespace and localname of a qname *)
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

(* helper functions for managing namespace prefixes
 *
 * Note: These would not be needed if symbolic XML managed this for values of attributes.
 *)
 
$namespaceQ = {};
namespaceID = 1;

getNamespacePrefix[ns_String, namespaces_List] := 
  Module[{nsMappings, prefixes, prefix},
    If[ns === "", Return[""]];
    If[xmlNamespaceQ[ns], Return["xmlns"]];
    nsMappings = Union[namespaces, $namespaceQ];
    prefixes = Cases[nsMappings, (val_String->ns) :> val];
    If[Length[prefixes] > 0, 
      prefix = First[prefixes],
      prefix = "ns"<> ToString[namespaceID];
      namespaceID++;
      AppendTo[$namespaceQ, prefix->ns];
    ];
    prefix
  ]
  
getPrefixedValue["", val_String] := val;
getPrefixedValue[namespace_String, val_String] := 
   getNamespacePrefix[namespace, $namespaceQ] <> ":" <> val

(* TODO - do these functions stay around? *)
validateNamespaceDefinition[val_String->ns_String] := val->ns;
validateNamespaceDefinition[x___] := (Message[GenerateSchema::namespace, x];$Failed);

(* Exceptions *)
SetAttributes[createSchemaException, {HoldFirst}];
createSchemaException[message_, params___] := 
  (Message[message, params];
   Throw[SchemaException[ToString[StringForm[message, params]]]]);

SetAttributes[createUnsupportedException, {HoldFirst}];
createUnsupportedException[message_, params___] := 
  (Message[message, params];
   Throw[UnsupportedException[ToString[StringForm[message, params]]], "unsupported"]);

End[];

EndPackage[];
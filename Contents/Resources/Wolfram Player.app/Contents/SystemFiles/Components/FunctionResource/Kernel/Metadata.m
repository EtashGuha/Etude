(* ::Package:: *)

(* Wolfram Language Package *)

BeginPackage["FunctionResource`"]

Begin["`Private`"] (* Begin Private Context *)


(* ::Section:: *)
(*Validation*)


(* ::Subsection:: *)
(*Function name*)


capitalize // ClearAll;
capitalize // Attributes = { Listable };

capitalize[ s_String? StringQ ] :=
  Replace[ Capitalize @ s,
           Except[ _String? StringQ ] :>
             StringReplace[ s,
                            first_ ~~ rest___ :>
                              ToUpperCase @ first <> rest
             ]
  ];

capitalize[ ___ ] := $failed;


toCamelCase // ClearAll;

toCamelCase[ s_String? StringQ ] :=
  With[ { split = StringSplit @ StringTrim @ s },
      Replace[ split,
               {
                   { first_, rest___ } :>
                     StringJoin @ Prepend[ capitalize @ { rest }, first ],
                   ___ :> s
               }
      ]
  ];

toCamelCase[ ___ ] := $failed;


removePublisherFromName[ name_String ] :=
  Last @ StringSplit[ name, ":"|"_" ];


makeShortName // ClearAll;

makeShortName[ name_String? StringQ ] :=
  With[ { camelCase = toCamelCase @ removePublisherFromName @ name },
      camelCase /; validResourceFunctionNameQ @ camelCase
  ];

makeShortName[ ___ ] := $failed;


resourceFunctionName // ClearAll;
resourceFunctionName[ name_String ] := name;
resourceFunctionName[ KeyValuePattern[ "ShortName" -> name_ ] ] := name;
resourceFunctionName[ (ResourceFunction|ResourceObject)[ info_, ___ ] ] := resourceFunctionName @ info;


validResourceFunctionNameQ // ClearAll;
validResourceFunctionNameQ[ args___ ] :=
  With[ { validQ = TrueQ @ iValidResourceFunctionNameQ @ args },
      If[ ! validQ, Message[ ResourceFunction::symname, resourceFunctionName @ args ] ];
      validQ
  ];


iValidResourceFunctionNameQ // ClearAll;

iValidResourceFunctionNameQ[ name_String ] :=
  Internal`SymbolNameQ @ name;

iValidResourceFunctionNameQ[ KeyValuePattern[ "ShortName" -> name_ ] ] :=
  iValidResourceFunctionNameQ @ name;

iValidResourceFunctionNameQ[ (ResourceFunction|ResourceObject)[ info_, ___ ] ] :=
  iValidResourceFunctionNameQ @ info;

iValidResourceFunctionNameQ[ ___ ] := False;


invalidResourceFunctionNameQ // ClearAll;
invalidResourceFunctionNameQ[ args___ ] := ! TrueQ @ validResourceFunctionNameQ @ args;


(* ::Subsection:: *)
(*Submission parameters*)


validateParameter[$FunctionResourceTypes, "ShortName", name_? validResourceFunctionNameQ ] := name;
validateParameter[$FunctionResourceTypes, "Function", expr_] := expr
validateParameter[$FunctionResourceTypes, "FunctionLocation", None] := None;
validateParameter[$FunctionResourceTypes, "FunctionLocation", "Inline" ] := "Inline";
validateParameter[$FunctionResourceTypes, "FunctionLocation", obj_CloudObject] := obj /; FileExistsQ @ obj && FileByteCount @ obj > 0;
validateParameter[$FunctionResourceTypes, "FunctionLocation", loc_PersistenceLocation ] := loc;
validateParameter[$FunctionResourceTypes, "FunctionLocation", str : Except[ _CloudObject ]] := str/;fileExistsQ[str]
validateParameter[$FunctionResourceTypes, "SymbolName", name_String ? NameQ ] := name;
validateParameter[$FunctionResourceTypes, "DefinitionData", bytes_ByteArray ? ByteArrayQ ] := bytes;
validateParameter[$FunctionResourceTypes, "Usage", usage_String? StringQ ] := usage;
validateParameter[$FunctionResourceTypes, "RelatedSymbols", syms: { ___String? StringQ } ] := syms;

(******************************************************************************)

validateParameter[$FunctionResourceTypes,"Documentation",as_Association]:=as

validateParameter[$FunctionResourceTypes,"VerificationTests",tests:HoldComplete[_VerificationTest...]]:=tests

validateParameter[$FunctionResourceTypes,"Categories",l_List]:=l/;Complement[l,
	ResourceSystemClient`Private`resourceSortingProperties["Function"]["Categories"]]==={}

ResourceSystemClient`Private`defaultSortingProperties[$FunctionResourceTypes]:=(
	ResourceSystemClient`Private`defaultSortingProperties["Function"]=Association["Categories" -> {}])


ResourceSystemClient`Private`uselessResourceProperties["Function"]:=
	(ResourceSystemClient`Private`uselessResourceProperties["Function"]=DeleteCases[
		ResourceSystemClient`Private`uselessResourceProperties[Automatic],"DefinitionNotebook"|"DefinitionNotebookObject"])
		
End[] (* End Private Context *)

EndPackage[]

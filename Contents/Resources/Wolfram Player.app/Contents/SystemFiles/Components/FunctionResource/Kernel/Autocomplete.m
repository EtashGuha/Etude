(* Mathematica Package *)
(* Created by Mathematica Plugin for IntelliJ IDEA *)

(* :Title: Autocomplete *)
(* :Context: FunctionResource`Autocomplete` *)
(* :Author: richardh@wolfram.com *)
(* :Date: 2018-12-03 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: *)
(* :Copyright: (c) 2018 Wolfram Research *)
(* :Keywords: *)
(* :Discussion: *)


BeginPackage[ "FunctionResource`Autocomplete`" ];

ClearAll @@ Names[ $Context ~~ ___ ];

(* Exported symbols added here with SymbolName::usage *)

InitializeAutocomplete;
AddAutocompleteNames;
RemoveAutocompleteNames;
$AutocompleteExcludedNames;

GeneralUtilities`SetUsage @ "\
InitializeAutocomplete[] populates the autocomplete menu for ResourceFunction \
with available repository names.";

GeneralUtilities`SetUsage @ "\
AddAutocompleteNames['name$1','name$2',$$] adds all the 'name$i' to the \
autocomplete menu for ResourceFunction.";

GeneralUtilities`SetUsage @ "\
RemoveAutocompleteNames['name$1','name$2',$$] removes all the 'name$i' from \
the autocomplete menu for ResourceFunction.";

GeneralUtilities`SetUsage @ "\
$AutocompleteExcludedNames is a global variable that gives a string pattern to \
exclude names from being added to the ResourceFunction autocomplete list.";

Begin[ "`Private`" ];


(******************************************************************************)
(* ::Section::Closed:: *)
(*InitializeAutocomplete*)


InitializeAutocomplete[ ] :=
  resetAutoCompleteNames[ ];


(******************************************************************************)
(* ::Section::Closed:: *)
(*AddAutocompleteNames*)


AddAutocompleteNames[ names___ ] :=
  addAutoCompleteNames @@ Flatten @ List @ names;


(******************************************************************************)
(* ::Section::Closed:: *)
(*RemoveAutocompleteNames*)


RemoveAutocompleteNames[ names___ ] :=
  removeAutoCompleteNames @@ Flatten @ List @ names;


(******************************************************************************)
(* ::Section::Closed:: *)
(*$ExcludedNames*)


$AutocompleteExcludedNames = "Untitled" ~~ DigitCharacter...;


(******************************************************************************)
(* ::Section::Closed:: *)
(*Properties*)


$ResourceFunctionProperties = {
    "Attributes",
    "AutoUpdate",
    "Categories",
    "ContributorInformation",
    "Definition",
    "DefinitionData",
    "DefinitionList",
    "DefinitionNotebook",
    "DefinitionNotebookObject",
    "Description",
    "Details",
    "DocumentationLink",
    "DocumentationNotebook",
    "DownloadedVersion",
    "ExampleNotebook",
    "ExampleNotebookObject",
    "FullDefinition",
    "Function",
    "FunctionLocation",
    "Keywords",
    "LatestUpdate",
    "Name",
    "Options",
    "Originator",
    "Properties",
    "ReleaseDate",
    "RepositoryLocation",
    "ResourceLocations",
    "ResourceObject",
    "ResourceType",
    "SeeAlso",
    "ShortName",
    "SourceMetadata",
    "Symbol",
    "SymbolName",
    "TestReport",
    "Usage",
    "UUID",
    "VerificationTests",
    "Version",
    "WolframLanguageVersionRequired"
};


(******************************************************************************)
(* ::Section::Closed:: *)
(*Name Persistence*)


$AutocompletePrefix = "ResourceFunctionAutocompleteNames/";


getCachedNames[ HoldPattern[ locations_: $PersistencePath ] ] :=
  Module[ { objects, fullNames, names, unused },
      objects = PersistentObjects[ $AutocompletePrefix <> "*", locations ];
      fullNames = Union @ Cases[ objects, PersistentObject[ name_String, ___ ] :> name ];
      names = fromPersistentName /@ fullNames;
      unused = clearUnusedCachedNames @ names;
      Complement[ names, unused ]
  ];


clearUnusedCachedNames[ HoldPattern[ cachedNames_: getCachedNames[ ] ] ] :=
  Module[ { unused, obj },
      unused = Select[ cachedNames, Not @* FunctionResource`Private`usedNameQ ];
      obj = PersistentObjects /@ toPersistentName @ unused;
      DeleteObject /@ Flatten @ obj;
      unused
  ];


cacheName[ name_, locations_ ] :=
  PersistentValue[ toPersistentName @ name, locations ] = name;


clearCachedName[ name_, HoldPattern[ locations_: $PersistencePath ] ] :=
  DeleteObject /@ PersistentObjects[ toPersistentName @ name, locations ];


(******************************************************************************)
(* ::Section::Closed:: *)
(*ResourceSystemClient Overrides*)


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*registerResourceNameExtensions*)


ResourceSystemClient`Private`registerResourceNameExtensions[ "Function", args___ ] :=
  registerResourceNameExtensions @ args;

registerResourceNameExtensions[ _, KeyValuePattern[ "Name" -> name_ ], loc_, ___ ] := (
    cacheName[ name, loc ];
    addAutoCompleteNames @ name
);


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*unregisterResourceNameExtensions*)


ResourceSystemClient`Private`unregisterResourceNameExtensions[ "Function", args___ ] :=
  unregisterResourceNameExtensions @ args;

unregisterResourceNameExtensions[ _, KeyValuePattern[ "Name" -> name_ ], loc_, ___ ] := (
    clearCachedName[ name, loc ];
    removeAutoCompleteNames @ name
);


(******************************************************************************)
(* ::Section::Closed:: *)
(*Utilities*)


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*Persistent Name Encoding*)


toPersistentName // Attributes = { Listable };

toPersistentName[ name_String ] :=
  $AutocompletePrefix <> URLEncode @ name;


fromPersistentName // Attributes = { Listable };

fromPersistentName[ name_String ] :=
  URLDecode @ StringDelete[ name, StartOfString ~~ $AutocompletePrefix ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*postProcessNameList*)


postProcessNameList[ list_ ] :=
  DeleteDuplicates @
    DeleteCases[ Cases[ list, _String? StringQ ],
                 _? (StringMatchQ @ $AutocompleteExcludedNames)
    ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*Name Cache*)


$publicNames := $publicNames =
  Sort @ postProcessNameList @
    ResourceSystemClient`Private`publicResourceInformation[ "Names" ][ "Function" ];


$localNames := $localNames =
  Sort @ postProcessNameList @ getCachedNames[ ];


$resourceFunctionNames := $resourceFunctionNames =
  Union[ $publicNames, $localNames ];


addResourceFunctionNames[ new___String ] :=
  $resourceFunctionNames =
    postProcessNameList @ Join[ { new }, $resourceFunctionNames ];


removeResourceFunctionNames[ remove___String ] :=
  $resourceFunctionNames =
    postProcessNameList @
      Join[ $publicNames,
            DeleteCases[ $resourceFunctionNames,
                         Alternatives @ remove
            ]
      ];


resetResourceFunctionNames[ ] :=
  $resourceFunctionNames = Union[ $publicNames, $localNames ];


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*Autocomplete Update*)


addAutoCompleteNames[ name___String ] :=
  updateNames @ addResourceFunctionNames @ name;


removeAutoCompleteNames[ name___String ] :=
  updateNames @ removeResourceFunctionNames @ name;


resetAutoCompleteNames[ ] :=
  updateNames @ resetResourceFunctionNames[ ];


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*Update Names*)


updateNames[ { } ] :=
  { };


updateNames[ names_ ] /; $UseAutocomplete :=
  With[ { props = $ResourceFunctionProperties },
      formatOutput @ FE`Evaluate @
        FEPrivate`AddSpecialArgCompletion[
            "ResourceFunction" -> { names, props }
        ]
  ];


updateNames[ _ ] :=
  Missing[ "NoFrontEnd" ];

updateNames[ ___ ] :=
  $Failed;


$UseAutocomplete :=
  TrueQ @ And[
      $EvaluationEnvironment === "Session",
      ! $CloudEvaluation,
      $Notebooks
  ];


formatOutput[ "ResourceFunction" -> { names: { ___String } } ] :=
  names;

formatOutput[ other___ ] :=
  other;


(******************************************************************************)
(* ::Section::Closed:: *)
(*EndPackage*)


End[ ]; (* `Private` *)

EndPackage[ ];

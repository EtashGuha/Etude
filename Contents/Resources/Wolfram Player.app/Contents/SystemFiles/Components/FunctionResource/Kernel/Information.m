(* Mathematica Package *)
(* Created by Mathematica Plugin for IntelliJ IDEA *)

(* :Title: Information *)
(* :Context: FunctionResource` *)
(* :Author: richardh@wolfram.com *)
(* :Date: 2018-10-30 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: *)
(* :Copyright: (c) 2018 Wolfram Research *)
(* :Keywords: *)
(* :Discussion: *)

BeginPackage[ "FunctionResource`" ];
(* Exported symbols added here with SymbolName::usage *)


ResourceFunctionInformationData;


Begin[ "`Private`" ];



(******************************************************************************)
(* ::Section::Closed:: *)
(*ResourceFunctionInformation*)


$allowMessageProps = "TestReport" | "VerificationTests";


ResourceFunctionInformation // ClearAll;


ResourceFunctionInformation[ rf: HoldPattern @ ResourceFunction[ id_ ] ] :=
  failOnMessage @ deleteFailedOrMissingValues @ First @ ResourceObject @ id;


ResourceFunctionInformation[ id_, All ] :=
  failOnMessage @ deleteFailedOrMissingValues @
    Join[ ResourceFunctionInformation @ id,
          resourceObjectProperty[ id, All ]
    ];

ResourceFunctionInformation[ id_, props_List ] :=
  With[ { info = ResourceFunctionInformation[ id, All ] },
      deleteFailedOrMissingValues @
        AssociationMap[ Lookup[ info, #, ResourceFunctionInformation[ id, # ] ] &,
                        props
        ] /; AssociationQ @ info
  ];


ResourceFunctionInformation[ id_, "ObjectType" ] :=
  "ResourceFunction";


ResourceFunctionInformation[ id_, "Usage" ] :=
  failOnMessage @ usageString @ ResourceFunction @ id;


ResourceFunctionInformation[ id_, "Documentation" ] :=
  failOnMessage @ documentationLinks @ ResourceFunction @ id;


ResourceFunctionInformation[ HoldPattern @ ResourceFunction[ rf_ ], property: $allowMessageProps ] :=
  ResourceFunction[ rf, property ];


ResourceFunctionInformation[ HoldPattern @ ResourceFunction[ rf_ ], property_ ] :=
  failOnMessage @ ResourceFunction[ rf, property ];


ResourceFunctionInformation[ id_ ] :=
  failOnMessage @ ResourceFunctionInformation @ ResourceFunction @ id;


ResourceFunctionInformation[ id_, property_ ] :=
  failOnMessage @ ResourceFunction[ id, property ];


ResourceFunctionInformation[ ___ ] := $failed;




(******************************************************************************)
(* ::Section::Closed:: *)
(*ResourceFunctionInformationData*)


ResourceFunctionInformationData[ rf_ResourceFunction ] :=
  Module[ { basic, subset, extra },

      basic = ResourceFunctionInformation @ rf;
      subset = KeyDrop[ basic, { "Documentation", "VerificationTests", "ExampleNotebook" } ];
      extra = Join[ subset,
          <|
              "ObjectType"    -> "ResourceFunction",
              "Usage"         -> ResourceFunctionInformation[ rf, "Usage" ],
              "Documentation" -> ResourceFunctionInformation[ rf, "Documentation" ],
              "Options"       -> ResourceFunctionInformation[ rf, "Options" ],
              "Attributes"    -> ResourceFunctionInformation[ rf, "Attributes" ]
          |>
      ];
      cleanResourceFunctionInfo @ extra
  ];


ResourceFunctionInformationData[ rf_, prop_ ] :=
  ResourceFunctionInformation[ rf, prop ];




(******************************************************************************)
(* ::Section::Closed:: *)
(*Utilities*)


cleanResourceFunctionInfo[ info: KeyValuePattern @ { "Name" -> name_, "ShortName" -> name_ } ] :=
  cleanResourceFunctionInfo @ KeyDrop[ info, "ShortName" ];

cleanResourceFunctionInfo[ info: KeyValuePattern[ "SourceMetadata" -> <| |> ] ] :=
  cleanResourceFunctionInfo @ KeyDrop[ info, "SourceMetadata" ];

cleanResourceFunctionInfo[ info_Association ] :=
  Module[ { dropped, ordered },
      dropped = KeyDrop[ info, "ResourceType" ];
      ordered = Join[ dropped[[ $priorityKeys ]], dropped ];
      deleteFailedOrMissingValues @ ordered
  ];


$priorityKeys = {
    "Name",
    "Description",
    "Usage",
    "Documentation",
    "Attributes",
    "Options"
};


deleteFailedOrMissingValues[ data_ ] :=
  DeleteCases[ data, _? FailureQ | _Missing | { } ];


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*usageString*)


usageString // ClearAll;

usageString[ BoxData[ boxes_ ] ] := usageString[ BoxData[ boxes ] ] =
  ToString[ Style[ RawBoxes @ insertButtonData @ boxes,
                   ShowStringCharacters -> True
            ],
            StandardForm
  ];

usageString[ TextData[ text_List ] ] := usageString[ TextData[ text ] ] =
  StringJoin[ usageString /@ text ];

usageString[ Cell[ BoxData[ b_ ], "InlineFormula" ] ] :=
  ToString[ RawBoxes @ StyleBox[ insertButtonData @ b, ShowStringCharacters -> True ], StandardForm ];

usageString[ cell_Cell ] :=
  ToString[ RawBoxes @ insertButtonData @ cell, StandardForm ];

usageString[ string_String ] :=
  string;

usageString[ a: KeyValuePattern @ { "Usage" -> usage_, "Description" -> description_ } ] /; ! KeyExistsQ[ a, "Name" ] :=
  StringJoin[ usageString @ usage, " ", usageString @ description ];

usageString[ KeyValuePattern[ "Documentation" -> KeyValuePattern[ "Usage" -> usage_ ] ] ] :=
  usageString @ usage;

usageString[ r_ResourceObject ] :=
  With[ { string = Quiet @ usageString @ r[ All ] },
      If[ StringQ @ string,
          string,
          usageString @ First @ r
      ]
  ];

usageString[ HoldPattern @ ResourceFunction[ ro_, ___ ] ] :=
  usageString @ ro;

usageString[ usage: { KeyValuePattern @ { "Usage" -> _, "Description" -> _ } ... } ] :=
  StringRiffle[ usageString /@ usage, "\n" ];

usageString[ info: KeyValuePattern[ "SymbolName" -> name_ ] /; ! KeyExistsQ[ info, "Documentation" ] ] :=
  Module[ { heldSym, usage },
      loadResourceFunction @ info;
      heldSym = ToExpression[ name, InputForm, HoldComplete ];
      usage = Replace[ heldSym,
                       HoldComplete[ s_Symbol? symbolQ ] :> Quiet[ s::usage ]
              ];
      usage /; StringQ @ usage
  ];

usageString[ info: KeyValuePattern[ "Name" -> name_ ] /; ! KeyExistsQ[ info, "Documentation" ] ] :=
  ToString[ Unevaluated @ ResourceFunction @ name, InputForm ];

usageString[ box: (f_Symbol)[ ___ ] ] /; StringEndsQ[ SymbolName @ f, "Box" ] :=
  ToString[ RawBoxes @ box, StandardForm ];

usageString[ ___ ] :=
  "";



insertButtonData // ClearAll;
insertButtonData[ boxes_ ] :=
  boxes /.
    box: ButtonBox[ name_String, a___, BaseStyle -> "Link", b___ ] /; FreeQ[ box, ButtonData -> _, { 1 } ] :>
    ButtonBox[ name, a, BaseStyle -> "Link", b, ButtonData -> "paclet:ref/" <> StringTrim[ name, "\"" ] ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*documentationLinks*)


documentationLinks[ info_ ] :=
  DeleteMissing @ Flatten @ { documentationLocalLink @ info, documentationWebLink @ info };

documentationLinks[ ___ ] :=
  Missing[ ];


documentationWebLink[ (r: ResourceFunction|ResourceObject)[ info_, ___ ] ] :=
  documentationWebLink @ info;

documentationWebLink[ KeyValuePattern[ "DocumentationLink" -> url_ ] ] :=
  documentationWebLink @ url;

documentationWebLink[ URL[ url_String ] ] :=
  documentationWebLink @ url;

documentationWebLink[ url_String ] :=
  With[ { string = URLBuild @ URLParse @ url },
        List @ Hyperlink[ "Web \[RightGuillemet]", string ] /; StringQ @ string
  ];

documentationWebLink[ info: KeyValuePattern[
    "RepositoryLocation" -> URL[ url_ /; MatchQ[ url, $ResourceSystemBase ] ]
] ] :=
  documentationWebLink @ Quiet @ ResourceObject[ info ][ "DocumentationLink" ];

documentationWebLink[ ___ ] :=
  Missing[ ];



documentationLocalLink[ id_ ] := documentationLocalLink[ id ] =
  If[ FunctionResource`DocumentationNotebook`LocalDocumentationAvailableQ @ id,
      Button[ MouseAppearance[ "Local \[RightGuillemet]", "LinkHand" ],
              ResourceFunction;
              FunctionResource`DocumentationNotebook`ViewDocumentationNotebook @ ResourceFunction @ id,
              Appearance -> None,
              BaseStyle -> Dynamic @ If[ CurrentValue[ "MouseOver" ], "HyperlinkActive", "Hyperlink" ],
              Method -> "Queued"
      ],
      Missing[ ]
  ];

documentationLocalLink[ ___ ] :=
  Missing[ ];




(******************************************************************************)
(* ::Subsection::Closed:: *)
(*enableResourceFunctionInformationFormatting*)


enableResourceFunctionInformationFormatting[ ] :=
  Internal`WithLocalSettings[
      Unprotect @ System`InformationData
      ,
      PrependTo[
          FormatValues @ System`InformationData,
          HoldPattern @
            MakeBoxes[
                System`InformationData[ info: KeyValuePattern[ "ObjectType" -> "ResourceFunction" ], ___ ],
                fmt_
            ] :>
              ReplaceAll[
                  ToBoxes @ System`InformationData @ Insert[ info, "ObjectType" -> "Symbol", Key @ "ObjectType" ],
                  {
                      StyleBox[ "\" Symbol\"", "InformationTitleText", a___ ] :>
                        StyleBox[ "\" Resource Function\"", "InformationTitleText", a ],
                      a: KeyValuePattern[ "ObjectType" -> "Symbol" ] :>
                        Insert[ a, "ObjectType" -> "ResourceFunction", Key @ "ObjectType" ]
                  }
              ]
      ];
      ,
      Protect @ System`InformationData
  ];





End[ ]; (* `Private` *)

EndPackage[ ];

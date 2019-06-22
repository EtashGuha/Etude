(* Created with the Wolfram Language : www.wolfram.com *)

BeginPackage["TwilioFunctions`"]

TImport::usage = "";
TGetCloudXML::usage = "";
TFormatRule::usage = "";

Begin["`Private`"]
TFormatRule[rule_Rule] :=
    StringJoin[
      Function[word,
        StringReplacePart[word, ToUpperCase[StringTake[word, 1]], 1]] /@
        StringSplit[
        StringReplace[
         rule[[1]], {
        "img" -> "Image",
        "url" -> "URL",
        "uri" -> "URI",
          "api" -> "API",
        "sms" -> "SMS",
              "mms" -> "MMS"
      }], "_"]] -> rule[[2]];

TImport[raw_] :=
    Module[ {response = ImportString[raw[[1]], "JSON"],status = raw[[2]]},
        If[ response===$Failed,
            Throw[$Failed]
        ];

        If[ !MatchQ[status,200|201],
            (Message[ServiceExecute::apierr,"message"/.response];
             Throw[$Failed]),
            Map[If[ Length[#] > 0 && Head[#[[1]]] === Rule,
                    Association[
                     TFormatRule /@ #],
                    #
                ] &,
            Association[TFormatRule /@ (response /. {
              ("date_updated" -> val_ /; val =!= Null) :> ("date_updated" -> DateObject[First[StringSplit[val, " +"]]]),
              ("date_created" -> val_ /; val =!= Null) :> ("date_created" -> DateObject[First[StringSplit[val, " +"]]]),
              ("date_sent" -> val_ /; val =!= Null) :> ("date_sent" -> DateObject[First[StringSplit[val, " +"]]])
              })],
            -2]
        ]
    ];

TGetCloudXML[url_, body_: ""] :=
    CloudDeploy[
        ExportForm[
            ImportString[StringTemplate["<?xml version=\"1.0\" encoding=\"UTF-8\" ?><Response><Say>``</Say><Play>``</Play></Response>"][body,url],"XML"],"XML"
        ],Permissions->"Public"
    ][[1]];

End[]

EndPackage[]

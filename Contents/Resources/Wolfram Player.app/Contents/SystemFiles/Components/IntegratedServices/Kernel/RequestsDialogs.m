(* Mathematica Package *)
BeginPackage["IntegratedServices`"]

IntegratedServices`CreateTOSDialogs
IntegratedServices`CreatePhoneVerificationDialogs

Begin["`Private`"]

IntegratedServices`CreateTOSDialogs[args__] :=
    Catch[createTOSDialogs[args]]
IntegratedServices`CreatePurchasingDialogs[args__] :=
    Catch[createPurchasingDialogs[args]]
IntegratedServices`CreateQuotaDialogs[args__] :=
    Catch[createQuotaDialogs[args]]

IntegratedServices`CreatePhoneVerificationDialogs[args__] :=
    Catch[createPhoneVerificationDialogs[args]]

integratedServiceCost = Association@{"BingSearch"->1,"GoogleCustomSearch"->3,"GoogleTranslate"->{10,1000},"MicrosoftTranslator"->{5,1000}};
getNumberServiceCredits[serviceName_] :=
    integratedServiceCost[serviceName]

integratedServiceName =
  Association@{"Twilio" -> "SendMessage", "BingSearch" -> "WebSearch",
     "GoogleCustomSearch" -> "WebSearch",
    "GoogleTranslate" -> "TextTranslation",
    "MicrosoftTranslator" -> "TextTranslation",
    "DigitalGlobeGeoServer" -> "Geo imagery",
    "Blockchain" -> "Wolfram Blockchain",
  "Freesound" -> "Freesound"}

getISName[serviceName_] :=
    integratedServiceName[serviceName]

serviceConnectNames =
 Association@{"DigitalGlobeGeoServer" -> "DigitalGlobe",
   "Twilio" -> "Twilio", "Twilio" -> "Twilio",
   "BingSearch" -> "Bing Web Search",
   "GoogleCustomSearch" -> "Google Web Search",
   "GoogleTranslate" -> "Google Translate",
   "MicrosoftTranslator" -> "Microsoft Translator",
   "Blockchain" -> "Wolfram Blockchain",
 "Freesound"-> "Freesound"}

getServiceConnectName[serviceName_] :=
    serviceConnectNames[serviceName]

descriptionTOS = "By continuing, you are agreeing to the `1` ";

descriptionPurchasing[headFunction_] :=
    With[ {head = ToString[headFunction]},
        Row[{
        Style[head, Bold] ,
        " accesses external services that require Wolfram Service Credits.\n\nYou currently have no available Wolfram Service Credits. To use ",
        head,
        " you need to add credits to your account."}
        ]
    ]

descriptionPurchasing[headFunction_] :=
    With[ {head = ToString[headFunction]},
        Row[{
        Style[head, Bold] ,
        " accesses external services that require Wolfram Service Credits.\n\nYou currently have no available Wolfram Service Credits. To use ",
        head,
        " you need to add credits to your account."}
        ]
    ]

descriptionPhoneVerification[headFunction_] :=
    With[ {head = ToString[headFunction]},
        Row[{
        Style[head, Bold] ,
        " needs to know your mobile phone number to send SMS and MMS messages.\n\nRun ",
        Style[head, Bold] ,
        " again after you have added a phone number to your account. "}
        ]
    ]

getTitleTOS[head_,"DigitalGlobeGeoServer"] :=
    getTitleTOS["Geo Image"]
getTitleTOS[head_,sn_] :=
    getTitleTOS[head]
getTitleTOS[head_] :=
    Style[head,Bold]

getHeaderTOS[head_,"DigitalGlobeGeoServer"] :=
    {"High\[Hyphen]resolution geo imagery requires access to ",
      Style["DigitalGlobe", Bold],
      " external\[NonBreakingSpace]service.\n"}

getHeaderTOS[head_,"Blockchain"] :=
    With[ {},
        {Style[head, Bold], " is requesting access to the ",
          Style[getServiceConnectName["Blockchain"], Bold],
        ".\n"}
]



getHeaderTOS[head_,serviceName_] :=
    With[ {},
        {Style[head, Bold], " is requesting access to the ",
          Style[getServiceConnectName[serviceName], Bold],
          " external service.\n"}
    ]

getHeaderPurchasing[logo_] :=
    Grid[{{logo,
    Pane[Grid[{{Item[
    Style["Integrated Services", 12,Bold, Black,
    FontFamily -> "Helvetica", FontWeight -> "Light"],
    Alignment -> {Left, Bottom}, ItemSize -> All]}, {Item[
    Style["Wolfram Service Credits Required ", 14, Black,
    FontFamily -> "Helvetica", FontWeight -> "Bold"],
    Alignment -> {Left, Center}, ItemSize -> All]}}, Alignment -> Left,
    ItemSize -> Full],
    ImageMargins -> {{0, 0}, {0, 2}}]}}, Frame -> None,
    Alignment -> {Left, Center}, Spacings -> {{0, .75, 0}, {0, 0, 0}},
    ItemSize -> Full]


approvetosURL = URLBuild[{$IntegratedServicesAPIBase,"ApproveTerms"}];

getHeaderTOS2[integratedservicename_,logo_] :=
    Grid[
     {{Pane[
        Grid[
         {{Framed[
            Item[Style[integratedservicename, 18, Black, FontFamily -> "Helvetica",
              FontWeight -> "Bold"]],
            (*FrameStyle\[Rule]Directive[1,RGBColor[0,0,0,0]],*)
            (*Background\[Rule]LightBlue,*)
            FrameStyle -> None,
            FrameMargins -> {{0, 0}, {0, 0}},
            RoundingRadius -> 0],
           Framed[Item[logo],
            (*FrameStyle\[Rule]Directive[1,RGBColor[0,0,0,0]],*)
            (*Background\[Rule]LightPurple,*)
            FrameStyle -> None,
            FrameMargins -> {{0, 0}, {2, 0}},
            RoundingRadius -> 0],
           Framed[
            Item[Style["Integrated", 18, Black, FontFamily -> "Helvetica",
               FontWeight -> "Regular"]],
            (*FrameStyle\[Rule]Directive[1,RGBColor[0,0,0,0]],*)
            (*Background\[Rule]LightYellow,*)
            FrameStyle -> None,
            FrameMargins -> {{0, 0}, {0, 0}},
            RoundingRadius -> 0],
           Framed[
            Item[Style["Service", 18, Black, FontFamily -> "Helvetica",
              FontWeight -> "Regular"]],
            (*FrameStyle\[Rule]Directive[1,RGBColor[0,0,0,0]],*)
            (*Background\[Rule]LightYellow,*)
            FrameStyle -> None,
            FrameMargins -> {{0, 0}, {0, 0}},
            RoundingRadius -> 0]}},
         Alignment -> Left,
         ItemSize -> Full,
         Spacings -> {.15, 0}
         ],
        ContentPadding -> False,
        FrameMargins -> {{0, 0}, {0, 0}},
        ImageMargins -> {{0, 0}, {0, 0}}
        ]}},
     Frame -> Directive[1, RGBColor[0, 0, 0, 0]],
     Spacings -> {0, 0},
     ItemSize -> Automatic(*,
     Background\[Rule]LightOrange*)]

getNoCredHeader[logo_] :=
    Grid[{{
     logo,
     Pane[
      Grid[{
        {Grid[
          {{Framed[
             Item[Style["Integrated", 12, Black,
               FontFamily -> "Helvetica", FontWeight -> "Light"]],
             (*FrameStyle\[Rule]Directive[1,RGBColor[0,0,0,0]],*)
             FrameStyle -> None,
             (*Background\[Rule]LightBlue,*)
             FrameMargins -> {{0, 0}, {0, 0}},
             RoundingRadius -> 0],

            Framed[Item[
              Style["Services", 12, Black, FontFamily -> "Helvetica",
               FontWeight -> "Light"]],
             (*FrameStyle\[Rule]Directive[1,RGBColor[0,0,0,0]],*)
             FrameStyle -> None,
             (*Background\[Rule]LightPurple,*)
             FrameMargins -> {{0, 0}, {0, 0}},
             RoundingRadius -> 0]
            }},
          Alignment -> {Left, Center},
          ItemSize -> Full,
          Spacings -> {.1, 0}
          ]},
        {Grid[
          {{Framed[
             Item[Style["Wolfram", 16, Black, FontFamily -> "Helvetica",
                FontWeight -> "Bold"]],
             (*FrameStyle\[Rule]Directive[1,RGBColor[0,0,0,0]],*)
             FrameStyle -> None,
             (*Background\[Rule]LightBlue,*)
             FrameMargins -> {{0, 0}, {0, 0}},
             RoundingRadius -> 0],

            Framed[Item[
              Style["Service", 16, Black, FontFamily -> "Helvetica",
               FontWeight -> "Bold"]],
             (*FrameStyle\[Rule]Directive[1,RGBColor[0,0,0,0]],*)
             FrameStyle -> None,
             (*Background\[Rule]LightPurple,*)
             FrameMargins -> {{0, 0}, {0, 0}},
             RoundingRadius -> 0],

            Framed[Item[
              Style["Credits", 16, Black, FontFamily -> "Helvetica",
               FontWeight -> "Bold"]],
             (*FrameStyle\[Rule]Directive[1,RGBColor[0,0,0,0]],*)
             FrameStyle -> None,
             (*Background\[Rule]LightGreen,*)
             FrameMargins -> {{0, 0}, {0, 0}},
             RoundingRadius -> 0],

            Framed[Item[
              Style["Required", 16, Black, FontFamily -> "Helvetica",
               FontWeight -> "Bold"]],
             (*FrameStyle\[Rule]Directive[1,RGBColor[0,0,0,0]],*)
             FrameStyle -> None,
             (*Background\[Rule]LightYellow,*)
             FrameMargins -> {{0, 0}, {0, 0}},
             RoundingRadius -> 0]}},
          Alignment -> Left,
          ItemSize -> Full,
          Spacings -> {.11, 0}
          ]}},
       Alignment -> {Left, Center},
       ItemSize -> Full,
       (*Background\[Rule]Pink,*)
       Frame -> RGBColor[0, 0, 0, 0],
       Spacings -> {{0, 0, 0}, {.4, -.15, 0}}]]}},
    Frame -> RGBColor[0, 0, 0, 0],
    Alignment -> {Left, Center},
    Spacings -> {{0, .6, 0}, {0, 0, 0}}]

 getHeaderPhoneVerification[logo_] :=
     Grid[{{
        logo,
        Pane[
         Grid[{
           {Grid[
             {{Framed[
                Item[Style["Integrated", 12, Black,
                  FontFamily -> "Helvetica", FontWeight -> "Light"]],
                (*FrameStyle\[Rule]Directive[1,RGBColor[0,0,0,0]],*)
                FrameStyle -> None,
                (*Background\[Rule]LightBlue,*)
                FrameMargins -> {{0, 0}, {0, 0}},
                RoundingRadius -> 0],

               Framed[Item[
                 Style["Services", 12, Black, FontFamily -> "Helvetica",
                  FontWeight -> "Light"]],
                (*FrameStyle\[Rule]Directive[1,RGBColor[0,0,0,0]],*)
                FrameStyle -> None,
                (*Background\[Rule]LightPurple,*)
                FrameMargins -> {{0, 0}, {0, 0}},
                RoundingRadius -> 0]
               }},
             Alignment -> {Left, Center},
             ItemSize -> Full,
             Spacings -> {.1, 0}
             ]},
           {Grid[
             {{Framed[
                Item[Style["Your", 16, Black, FontFamily -> "Helvetica",
                   FontWeight -> "Bold"]],
                (*FrameStyle\[Rule]Directive[1,RGBColor[0,0,0,0]],*)
                FrameStyle -> None,
                (*Background\[Rule]LightBlue,*)
                FrameMargins -> {{0, 0}, {0, 0}},
                RoundingRadius -> 0],

               Framed[Item[
                 Style["Mobile", 16, Black, FontFamily -> "Helvetica",
                  FontWeight -> "Bold"]],
                (*FrameStyle\[Rule]Directive[1,RGBColor[0,0,0,0]],*)
                FrameStyle -> None,
                (*Background\[Rule]LightPurple,*)
                FrameMargins -> {{0, 0}, {0, 0}},
                RoundingRadius -> 0],

               Framed[Item[
                 Style["Phone", 16, Black, FontFamily -> "Helvetica",
                  FontWeight -> "Bold"]],
                (*FrameStyle\[Rule]Directive[1,RGBColor[0,0,0,0]],*)
                FrameStyle -> None,
                (*Background\[Rule]LightGreen,*)
                FrameMargins -> {{0, 0}, {0, 0}},
                RoundingRadius -> 0],

               Framed[Item[
                 Style["is Required", 16, Black, FontFamily -> "Helvetica",
                  FontWeight -> "Bold"]],
                (*FrameStyle\[Rule]Directive[1,RGBColor[0,0,0,0]],*)
                FrameStyle -> None,
                (*Background\[Rule]LightYellow,*)
                FrameMargins -> {{0, 0}, {0, 0}},
                RoundingRadius -> 0]}},
             Alignment -> Left,
             ItemSize -> Full,
             Spacings -> {.11, 0}
             ]}},
          Alignment -> {Left, Center},
          ItemSize -> Full,
          (*Background\[Rule]Pink,*)
          Frame -> RGBColor[0, 0, 0, 0],
          Spacings -> {{0, 0, 0}, {.4, -.15, 0}}]]}},
      Frame -> RGBColor[0, 0, 0, 0],
      Alignment -> {Left, Center},
      Spacings -> {{0, .6, 0}, {0, 0, 0}}]

createTOSDialogs[serviceName_, headFunction_] :=
    Module[ {icon, tosLink = getTOSLink[serviceName/.$ServiceNamesRules],
      serviceName2 = (serviceName/.$ServiceNamesRules),head = ToString[headFunction],header,desc0,desc1,desc2,actionBtnLabel(*, learnLink = getLearnLink[]*)},
        icon = If[ $CloudEvaluateQ,
                   getIconCloud[],
                   getIcon["ISIcon"]
               ];
        header = getHeaderTOS2[headFunction,icon];
        desc0 = getHeaderTOS[head,serviceName2];
        desc1 = StringTemplate[descriptionTOS][getServiceConnectName[serviceName]];
        desc2 = Hyperlink["Terms of Use", tosLink, BaseStyle -> {RGBColor["#1d7bbf"], FontFamily -> "Helvetica",FontSize -> 11}];
        actionBtnLabel = "Continue";
        descriptionText = TextCell[Row[{Sequence@@desc0, desc1, desc2, "."}],ParagraphSpacing -> 0.25];
        CreateIntegratedServicesDialog[header,descriptionText,actionBtnLabel,(updateTOS[serviceName2])&,Null&,"GenericTitleBelow"->$CloudEvaluateQ,"ShowLearnButton" -> False,"GenericTitle" -> Row[{icon," "," Integrated Service"}]]
    ]

updateTOS[serviceName_] :=
    With[ {res = Quiet[URLExecute[CloudObject[approvetosURL], {"ServiceName" -> serviceName}, "RawJSON",VerifySecurityCertificates->$CloudPRDBaseQ]]},
        If[ TrueQ[res["Approved"]],
            DialogReturn[True],
            DialogReturn[False]
        ]
    ]

createPurchasingDialogs[serviceName_,headFunction_, nca_:0] :=
    Module[ {header,icon,descriptionText,actionBtnLabel},
        icon = If[ $CloudEvaluateQ,
                   getIconCloud[],
                   getIcon["ISIcon"]
               ];
        descriptionText = TextCell[descriptionPurchasing[headFunction]];
        actionBtnLabel = "Purchase Service Credits";
        header = getNoCredHeader[icon];
        IntegratedServices`Private`CreateIntegratedServicesDialog[
          header,
          descriptionText,
          actionBtnLabel,
          (SystemOpen[BillingURL];
            DialogReturn[True])&,
          (SystemOpen[ServiceCreditsLearnMoreURL])&,
          "GenericTitleBelow" -> True]
    ]

createQuotaDialogs[serviceName_,headFunction_] :=
    Null;

createPhoneVerificationDialogs[headFunction_] :=
    Module[ {header,icon,descriptionText,actionBtnLabel},
        icon = If[ $CloudEvaluateQ,
                   getIconCloud[],
                   getIcon["ISIcon"]
               ];
        descriptionText = TextCell[descriptionPhoneVerification[headFunction]];
        actionBtnLabel = "Add phone number";
        header = getHeaderPhoneVerification[icon];
        IntegratedServices`Private`CreateIntegratedServicesDialog[
          header,
          descriptionText,
          actionBtnLabel,
          (SystemOpen[PhoneVerificationURL];
           DialogReturn[True])&,
          (SystemOpen[SendMessageDocumentationURL])&,
          "GenericTitleBelow" -> True]
    ]

End[] (* End Private Context *)

EndPackage[]

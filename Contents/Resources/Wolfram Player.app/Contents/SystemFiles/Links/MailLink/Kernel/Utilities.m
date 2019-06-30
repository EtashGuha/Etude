BeginPackage["MailLink`Utilities`"]
icon
sequenceIDQ
encrypt
lookupid
$fetchmailid
overWriteDefaults
getMSettings
incomingMailSettingsQ
Begin["`Private`"] (* Begin Private Context *)
Dataset;
TypeSystem`NestedGrid`PackagePrivate`abox[other_] := 
          If[NumericQ[other],
                TextString @ N@ other,
                StyleBox[MakeBoxes @ other, FontFamily -> "Source Code Pro", FontSize -> 13]
         ];

Needs["MailLink`icons`"]
icon[name_String] := Graphics[Text[
        Style[name,
                  GrayLevel[0.7],
                  Bold,
                  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
        ]],Sequence[Background -> GrayLevel[0.93], Axes -> False, AspectRatio -> 1,
  ImageSize -> {Automatic, Dynamic[3.5*(CurrentValue["FontCapHeight"]/
       AbsoluteCurrentValue[Magnification]), ImageSizeCache -> {27.,
{0., 10.}}]},
  Frame -> True, FrameTicks -> None, FrameStyle ->
Directive[Thickness[Tiny], GrayLevel[0.55]]]
]





System`MailServerConnection /: MakeBoxes[ System`MailServerConnection[data_], fmt_]/;KeyExistsQ[data,"Folders"] :=
BoxForm`ArrangeSummaryBox[
                MailServerConnection,
                System`MailServerConnection[data],
                   None,
                	{
                       
                       BoxForm`SummaryItem[{"URL: ", data["URL"]}],
                       BoxForm`SummaryItem[{"Total Folders: ", Length@data["Folders"]}]
                       
                   },
                   {
                   	BoxForm`SummaryItem[{"Folders: ", InsertLinebreaks[StringJoin@Riffle[data["Folders"],", "],40]}]
                   },
                   fmt
           ]
           


System`MailFolder /: MakeBoxes[ System`MailFolder[data_], fmt_] :=
BoxForm`ArrangeSummaryBox[
                System`MailFolder,
                System`MailFolder[data],
                   MailLink`icons`mailFolder,
                {
                	BoxForm`SummaryItem[{"Folder: ",data["Name"]}],
                    BoxForm`SummaryItem[{"Total Messages: ", data["TotalMessageCount"]}],
					BoxForm`SummaryItem[{"Unread Messages: ", data["UnreadMessageCount"]}]	                       
                    
                       
                   },
                   {
                   	Flatten@Map[
                   		BoxForm`SummaryItem[{#<>": ",data[#]}]&,
                   		Complement[
                   			Keys[data],
                   			{"Name","TotalMessageCount","UnreadMessageCount","InternalID","ObjectType"}
                   			](*Complement*)
                   		](*Map*)
                   },
                   fmt
           ]
fontChange1[x_] := Style[x, FontFamily -> "Roboto"]

titleChange1[x_] := Style[x, FontFamily -> "Roboto", 11]

lineStyle = Directive[1, RGBColor[0, 0, 0, 0]];
makeprettyMessage[istring_String, maxlineLength_Integer] := 
 Module[{string = istring}, 
  string = trimString[string];
  If[StringLength[string] > maxlineLength, 
   StringTake[string, maxlineLength] <> "...", 
   StringPadRight[
    StringJoin[
     Riffle[StringTrim /@ 
       StringSplit@StringReplace[string, ("\n" | "\r") -> " "], " "]],
     maxlineLength]]]
makeprettyMessageBody[istring_String, maxlineLength_] := 
 Module[{string = istring, pos, newLinePos, l1, l2, lines,l1pos,l2pos}, 
  string = trimString[string];
  If[
  	StringLength[string]<maxlineLength,
  	  string
  	  ,
	  pos = StringPosition[string, " "];
	  Do[
	  	Which[
		     pos[[i]][[1]] === maxlineLength,
		     newLinePos=i; 
		     Return[i]
		     ,
		     pos[[i]][[1]] > maxlineLength, 
		     newLinePos=i-1;
		     Return[i - 1]
		     ,
		     pos[[i]][[1]]<maxlineLength,
		     newLinePos = i
	     ], 
	     {i, 1, Length[pos]}
	  ];
	  If[pos === {}, newLinePos = 0];
	  If[newLinePos =!= Null,
	  		If[newLinePos =!= 0,
  				  l1pos = pos[[newLinePos]][[1]] - 1; 
				  l2pos = {pos[[newLinePos]][[1]] + 1, -1}
				  ,
				  l1pos = maxlineLength;
				  l2pos = {maxlineLength + 1, -1}
				  ];
	   		l1 = StringTake[string, l1pos];
	   		l2 = StringTake[string, l2pos];
	   		lines = {l1,If[StringLength[l2] > maxlineLength, StringTake[l2, UpTo[maxlineLength]] <> "...", StringTake[l2, UpTo[maxlineLength]]]}
	   		,
			string
		];
		StringJoin[Riffle[lines, "\n"]]
	]
  ]
trimString=Function[{string},StringJoin[
    Riffle[StringTrim /@ 
      StringSplit@
       StringReplace[
        StringReplace[string, ("\n" | "\r" | "\t") -> " "], 
        Repeated[" "] -> " "], " "]] ]
emailContent[content_String] :=
 Style[
     Item[
      makeprettyMessageBody[content,67]],
     FontFamily -> "Roboto",
     FontSize -> 10,
     LineSpacing -> {1.5, 1.5},
     FontColor -> RGBColor[0.537254, 0.537254, 0.537254],
     LineIndent -> 0]             
emailContent[___]:=""
subjectLine[text_String]:=
 Style[
     makeprettyMessage[text,48],
     FontFamily -> "Roboto",
     FontSize -> 12,
     FontColor -> RGBColor[0.286274, 0.286274, 0.286274]]
subjectLine[___]:=""     
   
fromSender[senderName_String,senderAddress_String] :=
 Style[
     makeprettyMessage[senderName,40],
     FontFamily -> "Roboto",
     FontSize -> 12,
     FontColor -> RGBColor[0.39215, 0.39215, 0.39215]]
fromSender[senderName_Missing,senderAddress_String] :=
 Style[
     makeprettyMessage[senderAddress,40],
     FontFamily -> "Roboto",
     FontSize -> 12,
     FontColor -> RGBColor[0.39215, 0.39215, 0.39215]]
fromSender[___] :=""
makeToString[toName_List,toAddress_List]:=
StringJoin[
	Riffle[
		Table[
			If[toName[[i]] === Missing["NotAvailable"], 
				If[toAddress[[i]]=!=Missing["NotAvailable"],If[StringQ[toAddress[[i]]],toAddress[[i]],""]]
			  , 
			If[toName[[i]]=!=Missing["NotAvailable"],If[StringQ[toName[[i]]],toName[[i]],""]]
				], 
  		{i, 1,Length[toName]}
  		]
  		,", "]
  		]    
toRecipient[toName_List,toAddress_List] :=
 Style[
    makeprettyMessage[ 
    	"To: "<>
    	makeToString[toName,toAddress]
  ,70],
     FontFamily -> "Roboto",
     FontSize -> 10,
     FontColor -> RGBColor[0.537254, 0.537254, 0.537254]]
toRecipient[___]:="" 
carbonCopied[ccName_List,ccAddress_List]:=
 Style[
     "Cc: "<>makeprettyMessage[StringJoin[Riffle[Table[If[ccName[[i]] === Missing["NotAvailable"], 
  ccAddress[[i]], ccName[[i]]], {i, 1, 
  Length[ccName]}],", "]],20],
     FontFamily -> "Roboto",
     FontSize -> 10,
     FontColor -> RGBColor[0.537254, 0.537254, 0.537254]]
carbonCopied[{},{}]:=SpanFromLeft
    
carbonCopied[{}]:=SpanFromLeft   
styledDateString[date_DateObject] :=
 Style[
     DateString[date,"DateShort"]
      ,
     FontFamily -> "Roboto",
     FontSize -> 10,
     FontColor -> RGBColor[0.39215, 0.39215, 0.39215]]
     
styledDateString[___]:=""
getName=Function[assoc,If[assoc["FromName"]===Missing["Name"],assoc["FromAddress"],assoc["Name"]]]      

System`MailItem /: Format[System`MailItem[data_],StandardForm ]:=With[{display = Framed[
   Grid[{
    {Grid[{
       {If[Not[TrueQ@data["Flags"]["Seen"]],MailLink`icons`newMail,MailLink`icons`readMail]},
       {If[Not[TrueQ@data["Flags"]["Answered"]],MailLink`icons`empty,MailLink`icons`replied]},
       {Which[
       	Not[KeyExistsQ[data,"Attachments"]],
       	MailLink`icons`empty
       	,
       	TrueQ[KeyExistsQ[data,"Attachments"]] && Length[data["Attachments"]]>0,
       	MailLink`icons`attachment
       	,
       	True,
       	MailLink`icons`empty
       	
       	]
       	},
       {If[Not[TrueQ@data["Flags"]["Flagged"]],MailLink`icons`empty,MailLink`icons`flag]}
       },
      Alignment -> Center,
      Background -> None,
      Spacings -> {{0, 0}, {1, .5}},
      Frame -> All,
      FrameStyle -> lineStyle],
     
     Framed[
      Grid[{{""}},
       Background -> RGBColor[0.898039, 0.898039, 0.898039],
       Spacings -> {{0, 0}, {15.5}},
       Frame -> All,
       FrameStyle -> Directive[1, White],
       Alignment -> {Left, Center},
       ItemSize -> {.05, Full}],
      FrameMargins -> {{5, 5}, {0, 0}},
      Background -> None,
      FrameStyle -> lineStyle,
      Alignment -> {Center, Center}
      ],
     
     Grid[{
       {fromSender[data["FromName"],data["FromAddress"]], styledDateString[data["OriginatingDate"]]},
       {subjectLine[data["Subject"]], SpanFromLeft},
       {toRecipient[data["ToNameList"]~Join~data["CcNameList"],data["ToAddressList"]~Join~data["CcAddressList"]],SpanFromLeft},
       {emailContent[data["Body"]], SpanFromLeft}
       },
      Alignment ->
       {Center, Automatic,
        {{1, 1} -> {Left},
         {1, 2} -> {Right},
         {2, 1} -> {Left},
         {3, 1} -> {Left},
         
         
         
         {4, 1} -> {Left}}
        },
      Background -> None,
      Spacings -> 0,
      Frame -> All,
      FrameStyle -> lineStyle,
      ItemSize -> {{14,16}}]}
    },
   Alignment -> Top,
   Spacings -> 0,
   Frame -> All,
   FrameStyle -> lineStyle],
  FrameMargins -> {{5, 8}, {0, 0}},
  FrameStyle -> Directive[1, RGBColor[0.898039, 0.898039, 0.898039]],
  RoundingRadius -> 3
  ]},Interpretation[display,MailItem[data]]
 ]
(*System`MailItem /: Format[System`MailItem[data_Association], StandardForm]:= System`MailItem["<" <> ToString@data["SequenceID"] <> ">"]*)

initencrypt[] := Symbol["NumberTheory`AESDump`RijndaelDecryption"][]
incomingMailSettingsQ[assoc_?AssociationQ]:=TrueQ[KeyTake[assoc,{"Username","Server"}]===KeyTake[System`$IncomingMailSettings,{"Username","Server"}]]
encrypt[args___] := 
 With[{ef = (initencrypt[]; 
     Symbol["NumberTheory`AESDump`Private`rijndaelEncryption"])}, 
  ef[args]]

lookupid[args___] := 
 With[{df = (initencrypt[]; 
     Symbol["NumberTheory`AESDump`RijndaelDecryption"])}, 
  Block[{DataPaclets`SocialMediaDataDump`Private`flagQ = True, 
    NumberTheory`AESDump`Private`flagQ = True}, df[args]]]

$credsDir = 
  FileNameJoin[{$UserBaseDirectory, "ApplicationData", "Mail", 
    "Authentication"}];
$fetchmailid := 
 Internal`HouseKeep[$credsDir, {"machine_ID" -> $MachineID, 
   "version" -> $Version, "system_ID" -> $SystemID, 
   "user_name" -> $UserName}]

(* Wolfram Language Package *)

sequenceIDQ = Function[{input},MatchQ[input, _Integer] || StringMatchQ[input, NumberString]]


overWriteDefaults=
Function[{auth,ms},
	Select[Merge[{auth, ms},If[Length[Cases[#, _String]] > 0,Cases[#, _String][[1]], #[[1]]] &], # =!= {} &]
]

getMSettings[assoc_]:=changeKeys[Lookup[assoc,"IncomingMailServer",assoc]]
changeKeys=Function[assoc,AssociationMap[changeKey,assoc]]
changeKey["Host"->h_]:="Server"->h
changeKey[rule_]:=rule
End[] (* End Private Context *)

EndPackage[]

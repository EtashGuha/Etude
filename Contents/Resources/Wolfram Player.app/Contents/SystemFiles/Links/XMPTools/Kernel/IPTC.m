(**************************)
(**************************)
(**************************)
(********VALIDATION********)
(**************************)
(**************************)
(**************************)
                     
ValidateIPTC[res_] := AssociationMap[IPTCObjectValidate, DeleteCases[Association@KeyValueMap[#1 ->  DeleteCases[#2, _?(StringMatchQ[ToString@#,Whitespace ..] &)] &, res], _?(# == <||> &)]]

IPTCObjectValidate[Rule[key_, assoc_Association]] := Rule[key, AssociationMap[IPTCObjectValidate, assoc]]
IPTCObjectValidate[Rule[key_, val_]] := Which[
										    (MemberQ[DateTags, key] && !DateObjectQ[val]), Rule[key, Missing["Disputed"]],
   
   									        (MemberQ[TimeTags, key] && !TimeObjectQ[val]), Rule[key, Missing["Disputed"]],
   
     									    (MemberQ[QuantityTags, key] && QuantityQ@val && (List @@ val // First) < 0), Rule[key, Missing["Disputed"]],
										
	     									SameQ[key, "PreviewFormat"] || SameQ[key, "FileFormat"], If[val < 0 || val > 29, Rule[key, Missing["Disputed"]], Rule[key, val]],
		 								
		    								True, Rule[key, val]]
		    								
(**************************)
(**************************)
(**************************)
(**********IMPORT**********)
(***********IPTC***********)
(**************************)
(**************************)
ValidateIPTCAssociation[iptc_] := 
                            If[StringLength[iptc] > 5 && !SameQ[ToString@iptc, "LibraryFunctionError[LIBRARY_USER_ERROR,-2]"], 
            	                KeyMap[StringTrim, DeleteMissing[ToExpression[Quiet@StringReplace[iptc, WordCharacter .. ~~ " -> ," -> ""]]]]
 	                        ]

ParseIntAndRealTagsIPTC[state_] := Module[{cs = state, app2 = state["Application2"], env = state["Envelope"]},
 								       If[app2 =!= Missing["KeyAbsent", "Application2"], AssociateTo[app2, # -> If[ListQ@cs["Application2"][#], cs["Application2"][#], ToExpression@cs["Application2"][#]] & /@ DeleteDuplicates[Join[Intersection[ExportApplication2Number, Keys[cs["Application2"]]],Intersection[ExportEnvelopeNumber,Keys[cs["Application2"]]]]]]; AssociateTo[cs, "Application2" -> app2]];
 									   If[env =!= Missing["KeyAbsent", "Envelope"], AssociateTo[env, # -> If[ListQ@cs["Envelope"][#], cs["Envelope"][#], ToExpression@cs["Envelope"][#]] & /@ DeleteDuplicates[Join[Intersection[ExportApplication2Number, Keys[cs["Envelope"]]],Intersection[ExportEnvelopeNumber,Keys[cs["Envelope"]]]]]]; AssociateTo[cs, "Envelope" -> env]];
 									   cs
                                   ]

ParseDateTimeTagsIPTC[state_]   := Module[{cs = state, app2 = state["Application2"], env = state["Envelope"]},
 									   (*TimeTags*)
 									   If[app2 =!= Missing["KeyAbsent", "Application2"], AssociateTo[app2, # -> TimeObject[ToExpression@Drop[StringSplit[cs["Application2"][#], ":" | " " | "+"], -2], TimeZone->0] & /@ Intersection[TimeTags, Keys[cs["Application2"]]]]; AssociateTo[cs, "Application2" -> app2]];
 									   If[env =!= Missing["KeyAbsent", "Envelope"], AssociateTo[env, # -> TimeObject[ToExpression@Drop[StringSplit[cs["Envelope"][#], ":" | " " | "+"], -2], TimeZone->0] & /@ Intersection[TimeTags, Keys[cs["Envelope"]]]]; AssociateTo[cs, "Envelope" -> env]];
 									   (*DateTags*)
 									   If[app2 =!= Missing["KeyAbsent", "Application2"], AssociateTo[app2, # -> With[{tstDate = ToExpression@StringSplit[cs["Application2"][#], "-"], date = Take[DateList[{cs["Application2"][#], {"Year", "-", "Month", "-", "Day"}}], 3]},If[!System`ContainsAny[tstDate, {0}], DateObject[date, TimeZone -> $TimeZone], -1]] & /@ Intersection[DateTags, Keys[cs["Application2"]]]]; AssociateTo[cs, "Application2" -> app2]];
 									   If[env =!= Missing["KeyAbsent", "Envelope"], AssociateTo[env, # -> With[{tstDate = ToExpression@StringSplit[cs["Envelope"][#], "-"], date = Take[DateList[{cs["Envelope"][#], {"Year", "-", "Month", "-", "Day"}}], 3]},If[!System`ContainsAny[tstDate, {0}], DateObject[date, TimeZone -> $TimeZone], -1]] & /@ Intersection[DateTags, Keys[cs["Envelope"]]]]; AssociateTo[cs, "Envelope" -> env]];
 									   cs
                                    ]

ParseMultiValueTagsIPTC[state_] := Module[{cs = state, app2 = state["Application2"], env = state["Envelope"]},
 									   If[app2 =!= Missing["KeyAbsent", "Application2"], AssociateTo[app2, # -> If[StringContainsQ[ToString@cs["Application2"][#], "," | " "], ToExpression@StringSplit[ToString@cs["Application2"][#], ","], ToExpression@cs["Application2"][#]] & /@Intersection[MultiValues, Keys[cs["Application2"]]]]; AssociateTo[cs, "Application2" -> app2]];
 									   If[env =!= Missing["KeyAbsent", "Envelope"], AssociateTo[env, # -> If[StringContainsQ[ToString@cs["Envelope"][#], "," | " "], ToExpression@StringSplit[ToString@cs["Envelope"][#], ","], ToExpression@cs["Envelope"][#]] & /@Intersection[MultiValues, Keys[cs["Envelope"]]]]; AssociateTo[cs, "Envelope" -> env]];
 									   cs
 ]

ParseStringTagsIPTC[state_] := Module[{cs = state, app2 = state["Application2"], env = state["Envelope"]},
 									If[app2 =!= Missing["KeyAbsent", "Application2"], AssociateTo[app2, # -> If[StringQ@cs["Application2"][#], StringTrim[cs["Application2"][#]], cs["Application2"][#]] & /@ DeleteDuplicates[Join[Intersection[ExportApplication2Number, Keys[cs["Application2"]]], Intersection[ExportApplication2String, Keys[cs["Application2"]]]]]]; AssociateTo[cs, "Application2" -> app2]];
 									If[env =!= Missing["KeyAbsent", "Envelope"], AssociateTo[env, # -> If[StringQ@cs["Envelope"][#], StringTrim[cs["Envelope"][#]], cs["Envelope"][#]] & /@ DeleteDuplicates[Join[Intersection[ExportEnvelopeString, Keys[cs["Envelope"]]], Intersection[ExportEnvelopeString, Keys[cs["Envelope"]]]]]]; AssociateTo[cs, "Envelope" -> env]];
 									cs
 ]
 
ParseIndividualTagsIPTC[state_] := Module[{cs = state, env = state["Envelope"]},
  									   If[env =!= Missing["KeyAbsent", "Envelope"], AssociateTo[env, "CharacterSet" -> If[StringContainsQ[cs["Envelope"]["CharacterSet"], "%G"], "UTF8", cs["Envelope"]["CharacterSet"]]]; 
                                           AssociateTo[cs, "Envelope" -> env]
                                       ];
                                       cs
                                    ]
  
ParseValuesInGroupsIPTC[valEx_] := 
                             Module[{curState = valEx},
                                 curState = ParseDateTimeTagsIPTC[curState];
                                 curState = ParseMultiValueTagsIPTC[curState];
                                 curState = ParseIntAndRealTagsIPTC[curState];
                                 curState = ParseStringTagsIPTC[curState];
                                 curState = ParseIndividualTagsIPTC[curState];
                                 curState
                             ]  
         
 ParseStringTagsIPTCRaw[state_] := Module[{cs = state, badList = {}},
	                                   cs = AssociateTo[cs, # -> If[StringQ[cs[#]], StringTrim@cs[#], cs[#]] & /@ DeleteDuplicates[Join[Intersection[ExportApplication2Number, Keys[cs]], Intersection[ExportEnvelopeString, Keys[cs]], Intersection[ExportEnvelopeString, Keys[cs]], Intersection[ExportApplication2String, Keys[cs]]]]];
	                                   If[StringTrim[ToString[cs[#]]] == "", badList = Append[badList, #]] & /@ Keys[cs];
  							           cs = KeyDrop[cs, # &/@ badList];
	                                   cs = Append[cs, # -> Missing["NotAvailable"] & /@  DeleteCases[$AllIPTC, Alternatives @@ Sequence @@@ Keys[cs]]];
  								       cs
                                    ]

GetIPTCAll[] := With[{tmp = validatePossibleAssociation[$ReadIPTCAll[]]},
	If[tmp === "<||>",
		<||>,
		ParseValuesInGroupsIPTC[ValidateIPTCAssociation[tmp]]
	]
]

ReadIPTCIndividualTag[tag_]:= Module[{tmp, pth,res = None},
							      tmp = Quiet[validatePossibleString[$ReadIPTCIndividualTag[tag]]];
								  tmp = If[StringContainsQ[ToString[tmp], "LibraryFunctionError"] || tmp === Null || tmp === "", None, tmp];
								  If[MemberQ[IPTCEnvelope, tag], pth =  <|"Envelope"-> <|tag->tmp|>|>, pth = <|"Application2"-> <|tag->tmp|>|>];
								  If[tmp =!= None, res = First@First@ValidateIPTC[ParseValuesInGroupsIPTC[pth]]];
								  res
                              ]

ReadIPTC[tag_, rule_ : False] :=
	Block[{$Context = "XMPTools`TempContext`"},
		Module[{name = tag},
			Switch[name,
				"All"   , With[{iptcAll = GetIPTCAll[]}, If[Quiet[AssociationQ[iptcAll]], iptcAll, <||>]],
				"AllRaw",
				Module[
					{resTmp = validatePossibleAssociation[$ReadIPTCAllRaw[]],
						tmp
					},

					If[resTmp === "<||>",
						<||>,
						tmp = ParseStringTagsIPTCRaw[ParseIntAndRealTagsIPTC[ToExpression[resTmp]]];
						If[Quiet[AssociationQ[tmp]], tmp, <||>]
					]
				],
				_, ReadIPTCIndividualTag[tag]
			]
		]
	]

ReadIPTC[tag_] := ReadIPTC[tag, False]

(**************************)
(**************************)
(**************************)
(**********EXPORT**********)
(***********IPTC***********)
(**************************)
(**************************)
IPTCProcessToRaw[Rule[key_, assoc_Association]] := Rule[key, AssociationMap[IPTCProcessToRaw, assoc]]
IPTCProcessToRaw[Rule[key_, val_]] := Which[
											MemberQ[DateTags, key] && DateObjectQ[val], Rule[key, DateString[val, {"Year", "-", "MonthShort", "-", "DayShort"}]],
  
  											MemberQ[TimeTags, key] && TimeObjectQ[val], Rule[key, DateString[val, {"HourShort", ":", "MinuteShort", ":", "SecondShort", "+00:00"}]],
  																							   
                                            SameQ[key, "CharacterSet"], Rule[key, If[SameQ[val, "UTF8"], "\[RawEscape]%G", val]], 
  
  											True, Rule[key, Normal @@ val]
  									]
  									
PrepareIPTCMetaFromProcess[assc_] := Block[{$Context = "XMPTools`TempContext`"},
                                         AssociationMap[IPTCProcessToRaw, DeleteCases[Association@KeyValueMap[#1 -> DeleteCases[#2, _?(StringMatchQ[ToString@#, Whitespace ..] &)] &, assc], _?(# == <||> &)]]
                                      ]

PrepareIPTCMeta[] :=
	With[{tmp = validatePossibleAssociation[$ReadIPTCAll[]]},

		If[tmp === "<||>",
			<||>,
			Quiet@DeleteCases[ToExpression@StringReplace[validatePossibleAssociation[$ReadIPTCAll[]], WordCharacter .. ~~ " -> ," -> ""], Rule[_, _Missing], Infinity] /.
				(key_ /; (MemberQ[ExportApplication2Number, key] || MemberQ[ExportEnvelopeNumber, key]) -> val_) :> (key -> If[ListQ@val, val, ToExpression@val])
		]
	]

WriteIPTC[tag_, val_] := 
                     Block[{$Context = "XMPTools`TempContext`"}, 
                         Which[
 	                         MemberQ[ExportApplication2Number, tag]  ||
 	                         MemberQ[ExportEnvelopeNumber, tag]      , Quiet@$WriteIPTCInt[tag, val],
 	                         
 	                         MemberQ[ExportApplication2String, tag]  ||
 	                         MemberQ[ExportEnvelopeString, tag]      , If[StringContainsQ[tag, "Date"] && ToString@val === "-1", val =  "0-00-00"];Quiet@$WriteIPTCString[tag, val],
 	                         
 	                         True, _     
                         ]
                     ]


WriteIPTCRule[listOfRules : {__Rule}] := WriteIPTC @@@ listOfRules
WriteIPTCAssociation[list_Association]:= WriteIPTC @@@ Normal[list]
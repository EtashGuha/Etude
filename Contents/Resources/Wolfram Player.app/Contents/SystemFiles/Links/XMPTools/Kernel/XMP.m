(**************************)
(**************************)
(**************************)
(********VALIDATION********)
(**************************)
(**************************)
(**************************)
			                   
ValidateXMP[res_]  := AssociationMap[XMPObjectValidate, DeleteCases[Association@KeyValueMap[#1 ->  DeleteCases[#2, _?(StringMatchQ[ToString@#,Whitespace ..] &)] &, res], _?(# == <||> &)]]

XMPPositiveValuesOnly = {"SaveID", "TrackNumber", "VideoPixelAspectRatio", "FileDataRate", "Tempo", "NumberofBeats", "CropWidth", "CropHeight", "InitialViewHeadingDegrees", "InitialViewPitchDegrees", "InitialViewRollDegrees", "SourcePhotosCount", "CroppedAreaImageWidthPixels", "CroppedAreaImageHeightPixels",
                         "FullPanoWidthPixels", "FullPanoHeightPixels", "CroppedAreaLeftPixels", "CroppedAreaTopPixels", "PoseHeadingDegrees", "PosePitchDegrees", "PoseRollDegrees", "InitialHorizontalFOVDegrees", "InitialCameraDolly"}

XMPObjectValidate[Rule[key_, assoc_Association]] := Rule[key, AssociationMap[XMPObjectValidate, assoc]]
XMPObjectValidate[Rule[key_, val_]] := Module[{miss = Rule[key, Missing["Disputed"]], vRes = Rule[key, val], strToExpr = Quiet[ToExpression[val]]},
									       If[SameQ[ToString@strToExpr,"Disputed"], strToExpr = -10000];
									           Which[
										           MemberQ[DateTags, key] && !DateObjectQ[val],                             If[StringQ[val], 
										           	                                                                            With[{dt = DateObject[StringDrop[StringReplace[val, {"T" -> " "}], -6]]}, 
                                                                                                                                    Rule[key,
                                                                                                                                             If[DateObjectQ[dt], 
                                                                                                                                                 dt,
                                                                                                                                                 miss
                                                                                                                                              ]
                                                                                                                                    ]
                                                                                                                                ]
                                                                                                                            , miss],
										MemberQ[XMPPositiveValuesOnly, key]                                       	        , miss,
									   (MemberQ[QuantityTags, key] && QuantityQ@val && (List @@ val // First) < 0)		    , miss,

										SameQ["Rating", key] && (strToExpr < -1 || strToExpr > 5)                      		, miss,
										SameQ["Urgency", key] && (strToExpr < 0 || strToExpr > 8)                     	    , miss,
										SameQ["BlueHue", key] && (strToExpr < -100 || strToExpr > 100)                 	    , miss,
										SameQ["BlueSaturation", key] && (strToExpr < -100 || strToExpr > 100)      		    , miss, 
										SameQ["Brightness", key] && (strToExpr < 0 || strToExpr > 150)                 		, miss,
										SameQ["ChromaticAberrationBlue", key] && (strToExpr < -100 || strToExpr > 100) 		, miss,
										SameQ["ChromaticAberrationRed", key] && (strToExpr < -100 || strToExpr > 100) 	    , miss,
										SameQ["ColorNoiseReduction", key] && (strToExpr < 0 || strToExpr > 100)    		    , miss,
										SameQ["Contrast", key] && (strToExpr < -50 || strToExpr > 100)             		    , miss, 
										SameQ["GreenHue", key] && (strToExpr < -100 || strToExpr > 100)            		    , miss, 
										SameQ["GreenSaturation", key] && (strToExpr < -100 || strToExpr > 100)      		, miss, 
										SameQ["LuminanceSmoothing", key] && (strToExpr < -100 || strToExpr > 100)    	    , miss, 
										SameQ["RedHue", key] && (strToExpr < -100 || strToExpr > 100)               		, miss, 
										SameQ["RedSaturation", key] && (strToExpr < -100 || strToExpr > 100)         	    , miss, 
										SameQ["Saturation", key] && (strToExpr < -100 || strToExpr > 100)              		, miss, 
										SameQ["Shadows", key] && (strToExpr < 0 || strToExpr > 100)                  	    , miss, 
										SameQ["ShadowTint", key] && (strToExpr < -100 || strToExpr > 100)      		        , miss, 
										SameQ["Sharpness", key] && (strToExpr < 0 || strToExpr > 100)    	                , miss, 
										SameQ["Temperature", key] && ((List @@ val // First) < 2000 || 
										                                         (List @@ val // First) > 50000)  			, miss,            
										SameQ["Tint", key] && (strToExpr < -150 || strToExpr > 150)                    		, miss, 
										SameQ["VignetteAmount", key] && (strToExpr < -100 || strToExpr > 100)     	        , miss, 
										SameQ["VignetteMidpoint", key] && (strToExpr < 0 || strToExpr > 100)		        , miss, 
										SameQ["Exposure", key] && (strToExpr < -4.0 || strToExpr > 4.0)                		, miss, 
										SameQ["CropUnits", key] && (strToExpr =!= 0 || strToExpr =!= 1 || strToExpr =!= 2)  , miss, 
										True                                                                      , vRes]]
										
(**************************)
(**************************)
(**************************)
(**********IMPORT**********)
(***********XMP************)
(**************************)
(**************************)
splitAndGroup[list_List] := Module[{splitedList}, splitedList = MapAt[StringSplit[#, "."] &, list, {All, 1}];
  							    If[Length@splitedList[[1, 1]] == 1, Return@list];
  							    Normal@GroupBy[splitedList, (#[[1, 1]] &), MapAt[Last, {All, 1}]]
  							]  

MakeAsoc[xmp_] := Block[{getUtil = Needs["GeneralUtilities`"]},
	                  Quiet[DeleteCases[ReplaceAll[Replace[(xmp // GeneralUtilities`ToAssociations) /. Association -> foo, Rule[a_, b_] :> Rule[StringReplace[ToString[a], StringTake[ToString@a, 1] :> 
          			  ToUpperCase[StringTake[ToString@a, 1]]], If[StringContainsQ[ToString@b, "del"], StringDelete[ToString@b, "del"], b]], {0, Infinity}] /. foo -> Association, {true -> True, false -> False}], del, Infinity]]]

ValidateXMPAssociation[xmp_] := If[StringLength[xmp] > 5 && ! SameQ[xmp, "LibraryFunctionError[LIBRARY_USER_ERROR,-2]"], Quiet[Module[{tmp = Map[splitAndGroup, ToExpression@StringReplace[StringReplace[Quiet@StringReplace[xmp, WordCharacter .. ~~ " -> ," -> ""],
								 {"/crs:" -> "."}], {"lang=\"x-default\"" | "\"type=\"Struct\"\"" | "\"type=\"Seq\"\"" -> "del"}], {-3}]}, RemoveBlankValues[MakeAsoc[tmp]]]], <||>]


TrimoutBadValuesXMPRaw[xmp_] := Module[{cs = xmp, badList = {}},
	 	                            If[StringTrim@ToString[cs[#]] == "" || StringTrim@ToString[cs[#]] == "del", badList = Append[badList, #]] & /@ Keys[cs];
  							        cs = KeyDrop[cs, # &/@ badList];
  							        cs = Append[cs, # -> Missing["NotAvailable"] & /@  DeleteCases[$AllXMP, Alternatives @@ Sequence @@@ Keys[cs]]];
  							        cs
                                 ]
FinalParseXMPRaw[xmp_]:= Module[{cs = xmp},
                             cs = AssociateTo[cs, # -> If[! MatchQ[cs[#], Missing["NotAvailable"]], If[StringContainsQ[ToString[cs[#]], "," | " "], ToExpression[StringSplit[ToString[cs[#]], " "]], ToExpression[cs[#]]], cs[#]] & /@ Intersection[MultiValues, Keys[cs]]];
                             cs = AssociateTo[cs, # -> If[NumberQ[ToExpression[cs[#]]], ToExpression[cs[#]], cs[#]] & /@ DeleteDuplicates[Join[Intersection[IntegerTags, Keys[cs]], Intersection[RealTags, Keys[cs]]]]];
                             cs
                         ]

ValidateXMPAssociationRaw[xmp_] :=
    If[StringLength[xmp] > 5 && ! SameQ[xmp, "LibraryFunctionError[LIBRARY_USER_ERROR,-2]"],
	    Quiet[
		    Module[
			    {tmp = StringReplace[StringReplace[Quiet@StringReplace[xmp, WordCharacter .. ~~ " -> ," -> ""],
								 {"/crs:" -> "."}], {"lang=\"x-default\"" | "\"type=\"Struct\"\"" | "\"type=\"Seq\"\"" -> "del"}]}, If[StringQ[tmp], tmp = ToExpression[tmp]]; FinalParseXMPRaw[TrimoutBadValuesXMPRaw[tmp]]]], <||>]

RemoveBlankValues[state_] := Module[{cs = state, badList = {}, xmpMM = state["MediaManagementSchema"], dc = state["DublinCoreSchema"], xmp = state["BasicSchema"], 
	                                digiKam = state["PhotoManagementSchema"], crs = state["CameraRawSchema"], MicrosoftPhoto = state["MicrosoftPhotoSchema"], 
   								    photoshop = state["PhotoshopSchema"], xmpRights = state["RightsManagementSchema"], xmpBJ = state["BasicJobTicketSchema"], 
   								    xmpTPg = state["PagedTextSchema"], pdf = state["AdobePDFSchema"], GPano = state["GooglePhotoSphereSchema"], Iptc4xmpCore = state["IPTCCoreSchema"]},

                                If[GPano =!= Missing["KeyAbsent", "GooglePhotoSphereSchema"], 
	                            	If[StringTrim[ToString[GPano[#]]] == "", badList = Append[badList, #]] & /@ Keys[GPano];
	                            	GPano = KeyDrop[GPano, # &/@ badList];
	                                AssociateTo[cs, "GooglePhotoSphereSchema" -> GPano]];
	                            
	                            badList = {};
	                            
	                            If[Iptc4xmpCore =!= Missing["KeyAbsent", "IPTCCoreSchema"], 
	                                If[StringTrim[ToString[Iptc4xmpCore[#]]] == "", badList = Append[badList, #]] & /@ Keys[Iptc4xmpCore];
	                                Iptc4xmpCore = KeyDrop[Iptc4xmpCore, # &/@ badList];
	                                AssociateTo[cs, "IPTCCoreSchema" -> Iptc4xmpCore]];
	                            
	                            badList = {};

	                            If[pdf =!= Missing["KeyAbsent", "AdobePDFSchema"], 
	                            	If[StringTrim[ToString[pdf[#]]] == "", badList = Append[badList, #]] & /@ Keys[pdf];
	                            	pdf = KeyDrop[pdf, # &/@ badList];
	                                AssociateTo[cs, "AdobePDFSchema" -> pdf]];
	                            
	                            badList = {};
	                            
	                            If[xmpTPg =!= Missing["KeyAbsent", "PagedTextSchema"], 
	                            	If[StringTrim[ToString[xmpTPg[#]]] == "", badList = Append[badList, #]] & /@ Keys[xmpTPg];
	                            	xmpTPg = KeyDrop[xmpTPg, # &/@ badList];
	                                AssociateTo[cs, "PagedTextSchema" -> xmpTPg]];
	                            
	                            badList = {};

	                            If[xmpBJ =!= Missing["KeyAbsent", "BasicJobTicketSchema"], 
	                            	If[StringTrim[ToString[xmpBJ[#]]] == "", badList = Append[badList, #]] & /@ Keys[xmpBJ];
	                            	xmpBJ = KeyDrop[xmpBJ, # &/@ badList];
	                                AssociateTo[cs, "BasicJobTicketSchema" -> xmpBJ]];
	                            
	                            badList = {};
	                            
	                            If[xmpRights =!= Missing["KeyAbsent", "RightsManagementSchema"], 
	                            	If[StringTrim[ToString[xmpRights[#]]] == "", badList = Append[badList, #]] & /@ Keys[xmpRights];
	                            	xmpRights = KeyDrop[xmpRights, # &/@ badList];
	                                AssociateTo[cs, "RightsManagementSchema" -> xmpRights]];
	                            
	                            badList = {};
	                            
	                            If[photoshop =!= Missing["KeyAbsent", "PhotoshopSchema"], 
	                            	If[StringTrim[ToString[photoshop[#]]] == "", badList = Append[badList, #]] & /@ Keys[photoshop];
	                            	photoshop = KeyDrop[photoshop, # &/@ badList];
	                                AssociateTo[cs, "PhotoshopSchema" -> photoshop]];
	                            
	                            badList = {};

	                            If[MicrosoftPhoto =!= Missing["KeyAbsent", "MicrosoftPhotoSchema"], 
	                            	If[StringTrim[ToString[MicrosoftPhoto[#]]] == "", badList = Append[badList, #]] & /@ Keys[MicrosoftPhoto];
	                            	MicrosoftPhoto = KeyDrop[MicrosoftPhoto, # &/@ badList];
	                                AssociateTo[cs, "MicrosoftPhotoSchema" -> MicrosoftPhoto]];
	                            
	                            badList = {}; 
	                            
	                            If[crs =!= Missing["KeyAbsent", "CameraRawSchema"], 
	                            	If[StringTrim[ToString[crs[#]]] == "", badList = Append[badList, #]] & /@ Keys[crs];
	                            	crs = KeyDrop[crs, # &/@ badList];
	                                AssociateTo[cs, "CameraRawSchema" -> crs]];
	                            
	                            badList = {}; 

	                            If[digiKam =!= Missing["KeyAbsent", "PhotoManagementSchema"], 
	                            	If[StringTrim[ToString[digiKam[#]]] == "", badList = Append[badList, #]] & /@ Keys[digiKam];
	                            	digiKam = KeyDrop[digiKam, # &/@ badList];
	                                AssociateTo[cs, "PhotoManagementSchema" -> digiKam]];
	                            
	                            badList = {}; 
	          
	                            If[xmpMM =!= Missing["KeyAbsent", "MediaManagementSchema"], 
	                            	If[StringTrim[ToString[xmpMM[#]]] == "", badList = Append[badList, #]] & /@ Keys[xmpMM];
	                            	xmpMM = KeyDrop[xmpMM, # &/@ badList];
	                                AssociateTo[cs, "MediaManagementSchema" -> xmpMM]];
	                            
	                            badList = {};    

	                            If[dc =!= Missing["KeyAbsent", "DublinCoreSchema"], 
	                            	If[StringTrim[ToString[dc[#]]] == "", badList = Append[badList, #]] & /@ Keys[dc];
	                            	dc = KeyDrop[dc, # &/@ badList];
	                                AssociateTo[cs, "DublinCoreSchema" -> dc]];
	                            
	                            badList = {};

	                            If[xmp =!= Missing["KeyAbsent", "BasicSchema"], 
	                            	If[StringTrim[ToString[dc[#]]] == "", badList = Append[badList, #]] & /@ Keys[xmp];
	                            	xmp = KeyDrop[xmp, # &/@ badList];
	                                AssociateTo[cs, "BasicSchema" -> xmp]];

  								 cs
  ] 

ParseDateTimeTagsXMP[state_] :=  Module[{cs = state, xmp = state["BasicSchema"], digiKam = state["PhotoManagementSchema"], photoshop = state["PhotoshopSchema"], GPano = state["GooglePhotoSphereSchema"]},
	
   If[xmp =!= Missing["KeyAbsent", "BasicSchema"], AssociateTo[xmp, # -> With[{dt = cs["BasicSchema"][#]}, If[StringLength@dt <= 10, DateObject[Take[DateList[{dt, {"Year", "-", "Month", "-", "Day"}}], {1, 3}], TimeZone -> $TimeZone],DateObject[DateList[{If[StringLength@dt > 19, StringTake[dt, {1, 19}], dt], {"Year", ":", "Month", ":", "Day", "T", "Hour", ":", "Minute", ":", "Second"}}], TimeZone -> $TimeZone]]] & /@ 
   		DeleteCases[Intersection[DateTags, Keys[cs["BasicSchema"]]], "When"]]; AssociateTo[cs, "BasicSchema" -> xmp]];
   		
   If[GPano =!= Missing["KeyAbsent", "GooglePhotoSphereSchema"], AssociateTo[GPano, # -> With[{dt = cs["GooglePhotoSphereSchema"][#]}, If[StringLength@dt <= 10, DateObject[Take[DateList[{dt, {"Year", "-", "Month", "-", "Day"}}], {1, 3}], TimeZone -> $TimeZone],DateObject[DateList[{If[StringLength@dt > 19, StringTake[dt, {1, 19}], dt], {"Year", ":", "Month", ":", "Day", "T", "Hour", ":", "Minute", ":", "Second"}}], TimeZone -> $TimeZone]]] & /@ 
   		Intersection[DateTags, Keys[cs["GooglePhotoSphereSchema"]]]]; AssociateTo[cs, "GooglePhotoSphereSchema" -> GPano]];
   
   If[digiKam =!= Missing["KeyAbsent", "PhotoManagementSchema"], AssociateTo[digiKam, # -> With[{dt=cs["PhotoManagementSchema"][#]},If[StringLength@dt <= 10, DateObject[Take[DateList[{dt, {"Year", "-", "Month", "-", "Day"}}], {1, 3}], TimeZone -> $TimeZone],DateObject[DateList[{If[StringLength@dt > 19, StringTake[dt, {1, 19}], dt], {"Year", ":", "Month", ":", "Day", "T", "Hour", ":", "Minute", ":", "Second"}}], TimeZone -> $TimeZone]]] & /@ 
   		DeleteCases[Intersection[DateTags, Keys[cs["PhotoManagementSchema"]]], "When"]]; AssociateTo[cs, "PhotoManagementSchema" -> digiKam]];
   
   If[photoshop =!= Missing["KeyAbsent", "PhotoshopSchema"], AssociateTo[photoshop, # -> With[{dt=cs["PhotoshopSchema"][#]},If[StringLength@dt <= 10, DateObject[Take[DateList[{dt, {"Year", "-", "Month", "-", "Day"}}], {1, 3}], TimeZone -> $TimeZone],DateObject[DateList[{If[StringLength@dt > 19, StringTake[dt, {1, 19}], dt], {"Year", ":", "Month", ":", "Day", "T", "Hour", ":", "Minute", ":", "Second"}}], TimeZone -> $TimeZone]]] & /@ 
   		DeleteCases[Intersection[DateTags, Keys[cs["PhotoshopSchema"]]], "When"]]; AssociateTo[cs, "PhotoshopSchema" -> photoshop]];
   
   cs
  ]

ParseMultiValueTagsXMP[state_] := Module[{cs = state, xmpMM = state["MediaManagementSchema"], dc = state["DublinCoreSchema"], xmp = state["BasicSchema"], digiKam = state["PhotoManagementSchema"], crs = state["CameraRawSchema"], MicrosoftPhoto = state["MicrosoftPhotoSchema"], 
   										  photoshop = state["PhotoshopSchema"], xmpRights = state["RightsManagementSchema"], xmpBJ = state["BasicJobTicketSchema"], xmpTPg = state["PagedTextSchema"], pdf = state["AdobePDFSchema"]},
   										  
  If[xmpMM =!= Missing["KeyAbsent", "MediaManagementSchema"], AssociateTo[xmpMM, # -> If[StringContainsQ[ToString@cs["MediaManagementSchema"][#], "," | " "], 
  		ToExpression@StringSplit[ToString@cs["MediaManagementSchema"][#], ","], If[StringQ[cs["MediaManagementSchema"][#]], ToExpression@cs["MediaManagementSchema"][#], cs["MediaManagementSchema"][#]]] & /@ Intersection[MultiValues, Keys[cs["MediaManagementSchema"]]]]; 
  		AssociateTo[cs, "MediaManagementSchema" -> xmpMM]];
  
  If[dc =!= Missing["KeyAbsent", "DublinCoreSchema"], AssociateTo[dc, # -> If[StringContainsQ[ToString@cs["DublinCoreSchema"][#], "," | " "], 
  		ToExpression@StringSplit[ToString@cs["DublinCoreSchema"][#], ","], If[StringQ[cs["DublinCoreSchema"][#]], ToExpression@cs["DublinCoreSchema"][#], cs["DublinCoreSchema"][#]]] & /@ Intersection[MultiValues, Keys[cs["DublinCoreSchema"]]]]; 
  		AssociateTo[cs, "DublinCoreSchema" -> dc]];
  
  If[xmp =!= Missing["KeyAbsent", "BasicSchema"], AssociateTo[xmp, # -> If[StringContainsQ[ToString@cs["BasicSchema"][#], "," | " "], 
  		ToExpression@StringSplit[ToString@cs["BasicSchema"][#], ","], If[StringQ[cs["BasicSchema"][#]], ToExpression@cs["BasicSchema"][#], cs["BasicSchema"][#]]] & /@ Intersection[MultiValues, Keys[cs["BasicSchema"]]]]; 
  		AssociateTo[cs, "BasicSchema" -> xmp]];
  
  If[digiKam =!= Missing["KeyAbsent", "PhotoManagementSchema"], AssociateTo[digiKam, # -> If[StringContainsQ[ToString@cs["PhotoManagementSchema"][#], "," | " "], 
  		ToExpression@StringSplit[ToString@cs["PhotoManagementSchema"][#], ","], If[StringQ[cs["PhotoManagementSchema"][#]], ToExpression@cs["PhotoManagementSchema"][#], cs["PhotoManagementSchema"][#]]] & /@ Intersection[MultiValues, Keys[cs["PhotoManagementSchema"]]]]; 
  		AssociateTo[cs, "PhotoManagementSchema" -> digiKam]];
  
  If[crs =!= Missing["KeyAbsent", "CameraRawSchema"], AssociateTo[crs, # -> If[StringContainsQ[ToString@cs["CameraRawSchema"][#],  "," | " "], 
  	    ToExpression@StringSplit[ToString@cs["CameraRawSchema"][#], ","], If[StringQ[cs["CameraRawSchema"][#]], ToExpression@cs["CameraRawSchema"][#], cs["CameraRawSchema"][#]]] & /@ Intersection[MultiValues, Keys[cs["CameraRawSchema"]]]];
  	    AssociateTo[cs, "CameraRawSchema" -> crs]];
  
  If[MicrosoftPhoto =!= Missing["KeyAbsent", "MicrosoftPhotoSchema"], AssociateTo[MicrosoftPhoto, # -> If[StringContainsQ[ToString@cs["MicrosoftPhotoSchema"][#],  "," | " "], ToExpression@StringSplit[ToString@cs["MicrosoftPhotoSchema"][#], ","], 
        ToExpression@cs["MicrosoftPhotoSchema"][#]] & /@ Intersection[MultiValues, Keys[cs["MicrosoftPhotoSchema"]]]]; AssociateTo[cs, "MicrosoftPhotoSchema" -> MicrosoftPhoto]];
  
  If[photoshop =!= Missing["KeyAbsent", "PhotoshopSchema"], AssociateTo[photoshop, # -> If[StringContainsQ[ToString@cs["PhotoshopSchema"][#], "," | " "], ToExpression@StringSplit[ToString@cs["PhotoshopSchema"][#], ","], 
        ToExpression@cs["PhotoshopSchema"][#]] & /@ Intersection[MultiValues, Keys[cs["PhotoshopSchema"]]]]; AssociateTo[cs, "PhotoshopSchema" -> photoshop]];
  
  If[xmpRights =!= Missing["KeyAbsent", "RightsManagementSchema"], AssociateTo[xmpRights, # ->If[StringContainsQ[ToString@cs["RightsManagementSchema"][#], "," | " "], 
        ToExpression@StringSplit[ToString@cs["RightsManagementSchema"][#], ","], If[StringQ[cs["RightsManagementSchema"][#]], ToExpression@cs["RightsManagementSchema"][#], cs["RightsManagementSchema"][#]]] & /@ Intersection[MultiValues, Keys[cs["RightsManagementSchema"]]]]; 
        AssociateTo[cs, "RightsManagementSchema" -> xmpRights]];
  
  If[xmpBJ =!= Missing["KeyAbsent", "BasicJobTicketSchema"], AssociateTo[xmpBJ, # -> If[StringContainsQ[ToString@cs["BasicJobTicketSchema"][#], "," | " "], 
        ToExpression@StringSplit[ToString@cs["BasicJobTicketSchema"][#], ","],If[StringQ[cs["BasicJobTicketSchema"][#]], ToExpression@cs["BasicJobTicketSchema"][#], cs["BasicJobTicketSchema"][#]]] & /@Intersection[MultiValues, Keys[cs["BasicJobTicketSchema"]]]]; 
   		AssociateTo[cs, "BasicJobTicketSchema" -> xmpBJ]];
  
  If[xmpTPg =!= Missing["KeyAbsent", "PagedTextSchema"], AssociateTo[xmpTPg, # -> If[StringContainsQ[ToString@cs["PagedTextSchema"][#], "," | " "], 
        ToExpression@StringSplit[ToString@cs["PagedTextSchema"][#], ","],If[StringQ[cs["PagedTextSchema"][#]], ToExpression@cs["PagedTextSchema"][#], cs["PagedTextSchema"][#]]] & /@Intersection[MultiValues, Keys[cs["PagedTextSchema"]]]]; 
   		AssociateTo[cs, "PagedTextSchema" -> xmpTPg]];
  
  If[pdf =!= Missing["KeyAbsent", "AdobePDFSchema"], AssociateTo[pdf, # -> If[StringContainsQ[ToString@cs["AdobePDFSchema"][#], "," | " "], 
        ToExpression@StringSplit[ToString@cs["AdobePDFSchema"][#], ","], If[StringQ[cs["AdobePDFSchema"][#]], ToExpression@cs["AdobePDFSchema"][#], cs["AdobePDFSchema"][#]]] & /@ Intersection[MultiValues, Keys[cs["AdobePDFSchema"]]]]; 
   		AssociateTo[cs, "AdobePDFSchema" -> pdf]];
  
  cs
  ]

ParseIndividualTagsXMP[state_] := Module[{cs = state, crs = state["CameraRawSchema"]}, 
	If[crs =!= Missing["KeyAbsent", "CameraRawSchema"] && crs["Temperature"] =!= Missing["KeyAbsent", "Temperature"], 
		AssociateTo[crs, # -> Quantity[ToExpression[crs["Temperature"]], "Kelvins"] & /@ {"Temperature"}]; 
	   	AssociateTo[cs, "CameraRawSchema" -> crs]];
  
    cs
  ]

ParseIntAndRealTagsXMP[state_] := Module[{cs = state, dateTags, GPano = state["GooglePhotoSphereSchema"],  xmpMM = state["MediaManagementSchema"], dc = state["DublinCoreSchema"], xmp = state["BasicSchema"], digiKam = state["PhotoManagementSchema"], crs = state["CameraRawSchema"], MicrosoftPhoto = state["MicrosoftPhotoSchema"],
   										  photoshop = state["PhotoshopSchema"],xmpRights = state["RightsManagementSchema"], xmpBJ = state["BasicJobTicketSchema"], xmpTPg = state["PagedTextSchema"], pdf = state["AdobePDFSchema"]},
	

  dateTags = Append[DeleteCases[DateTags, "when"], "WhiteBalance"];
  
  If[GPano =!= Missing["KeyAbsent", "GooglePhotoSphereSchema"], AssociateTo[GPano, # -> If[DateObjectQ[cs["GooglePhotoSphereSchema"][#]] || ListQ@cs["GooglePhotoSphereSchema"][#], cs["GooglePhotoSphereSchema"][#], 
        ToExpression@cs["GooglePhotoSphereSchema"][#]] & /@ DeleteCases[DeleteDuplicates[Join[Intersection[IntegerTags, Keys[cs["GooglePhotoSphereSchema"]]], Intersection[BooleanTags, Keys[cs["GooglePhotoSphereSchema"]]], Intersection[RealTags, Keys[cs["GooglePhotoSphereSchema"]]]]], dateTags]];
   		AssociateTo[cs, "GooglePhotoSphereSchema" -> GPano]];
  
  If[xmpMM =!= Missing["KeyAbsent", "MediaManagementSchema"], AssociateTo[xmpMM, # -> If[DateObjectQ[cs["MediaManagementSchema"][#]] || ListQ@cs["MediaManagementSchema"][#], cs["MediaManagementSchema"][#], 
        ToExpression@cs["MediaManagementSchema"][#]] & /@ DeleteCases[DeleteDuplicates[Join[Intersection[IntegerTags, Keys[cs["MediaManagementSchema"]]], Intersection[BooleanTags, Keys[cs["MediaManagementSchema"]]], Intersection[RealTags, Keys[cs["MediaManagementSchema"]]]]], dateTags]]; 
   		AssociateTo[cs, "MediaManagementSchema" -> xmpMM]];
  
  If[dc =!= Missing["KeyAbsent", "DublinCoreSchema"], AssociateTo[dc, # -> If[DateObjectQ[cs["DublinCoreSchema"][#]] || ListQ@cs["DublinCoreSchema"][#], cs["DublinCoreSchema"][#], 
  		ToExpression@cs["DublinCoreSchema"][#]] & /@ DeleteCases[DeleteDuplicates[Join[Intersection[IntegerTags, Keys[cs["DublinCoreSchema"]]], Intersection[BooleanTags, Keys[cs["DublinCoreSchema"]]], Intersection[RealTags, Keys[cs["DublinCoreSchema"]]]]], dateTags]]; 
   		AssociateTo[cs, "DublinCoreSchema" -> dc]];
  
  If[xmp =!= Missing["KeyAbsent", "BasicSchema"], AssociateTo[xmp, # -> If[DateObjectQ[cs["BasicSchema"][#]] || ListQ@cs["BasicSchema"][#], cs["BasicSchema"][#], 
  	    ToExpression@cs["BasicSchema"][#]] & /@ DeleteCases[DeleteDuplicates[Join[Intersection[IntegerTags, Keys[cs["BasicSchema"]]], Intersection[BooleanTags, Keys[cs["BasicSchema"]]], Intersection[RealTags, Keys[cs["BasicSchema"]]]]], dateTags]]; 
  		AssociateTo[cs, "BasicSchema" -> xmp]];
  
  If[digiKam =!= Missing["KeyAbsent", "PhotoManagementSchema"], AssociateTo[digiKam, # -> If[DateObjectQ[cs["Application2"][#]] || ListQ@cs["PhotoManagementSchema"][#], cs["PhotoManagementSchema"][#], 
  		ToExpression@cs["PhotoManagementSchema"][#]] & /@ DeleteCases[DeleteDuplicates[Join[Intersection[IntegerTags, Keys[cs["Application2"]]], Intersection[BooleanTags, Keys[cs["PhotoManagementSchema"]]], Intersection[RealTags, Keys[cs["PhotoManagementSchema"]]]]], dateTags]]; 
   		AssociateTo[cs, "PhotoManagementSchema" -> digiKam]];
  
  If[crs =!= Missing["KeyAbsent", "CameraRawSchema"], AssociateTo[crs, # -> If[DateObjectQ[cs["CameraRawSchema"][#]] || ListQ@cs["CameraRawSchema"][#], cs["CameraRawSchema"][#], 
  	    ToExpression@cs["CameraRawSchema"][#]] & /@ DeleteCases[DeleteDuplicates[Join[Intersection[IntegerTags, Keys[cs["CameraRawSchema"]]], Intersection[BooleanTags, Keys[cs["CameraRawSchema"]]], Intersection[RealTags, Keys[cs["CameraRawSchema"]]]]], dateTags]]; 
   		AssociateTo[cs, "CameraRawSchema" -> crs]];
  
  If[MicrosoftPhoto =!= Missing["KeyAbsent", "MicrosoftPhotoSchema"], AssociateTo[MicrosoftPhoto, # -> If[DateObjectQ[cs["MicrosoftPhotoSchema"][#]] || ListQ@cs["MicrosoftPhotoSchema"][#], cs["MicrosoftPhotoSchema"][#],
  		ToExpression@cs["MicrosoftPhotoSchema"][#]] & /@ DeleteCases[DeleteDuplicates[Join[Intersection[IntegerTags, Keys[cs["MicrosoftPhotoSchema"]]], Intersection[BooleanTags, Keys[cs["MicrosoftPhotoSchema"]]], Intersection[RealTags, Keys[cs["MicrosoftPhotoSchema"]]]]], dateTags]]; 
   		AssociateTo[cs, "MicrosoftPhotoSchema" -> MicrosoftPhoto]];
  
  If[photoshop =!= Missing["KeyAbsent", "PhotoshopSchema"], AssociateTo[photoshop, # -> If[DateObjectQ[cs["PhotoshopSchema"][#]] || ListQ@cs["PhotoshopSchema"][#], cs["PhotoshopSchema"][#], 
        ToExpression@cs["PhotoshopSchema"][#]] & /@ DeleteCases[DeleteDuplicates[Join[Intersection[IntegerTags, Keys[cs["PhotoshopSchema"]]], Intersection[BooleanTags, Keys[cs["PhotoshopSchema"]]], Intersection[RealTags, Keys[cs["PhotoshopSchema"]]]]], dateTags]]; 
   		AssociateTo[cs, "PhotoshopSchema" -> photoshop]];
  
  If[xmpRights =!= Missing["KeyAbsent", "RightsManagementSchema"], AssociateTo[xmpRights, # -> If[DateObjectQ[cs["RightsManagementSchema"][#]] || ListQ@cs["RightsManagementSchema"][#], cs["RightsManagementSchema"][#], 
        ToExpression@cs["RightsManagementSchema"][#]] & /@ DeleteCases[DeleteDuplicates[Join[Intersection[IntegerTags, Keys[cs["RightsManagementSchema"]]], Intersection[BooleanTags, Keys[cs["RightsManagementSchema"]]], Intersection[RealTags, Keys[cs["RightsManagementSchema"]]]]], dateTags]]; 
   		AssociateTo[cs, "RightsManagementSchema" -> xmpRights]];
  
  If[xmpBJ =!= Missing["KeyAbsent", "BasicJobTicketSchema"], AssociateTo[xmpBJ, # -> If[DateObjectQ[cs["BasicJobTicketSchema"][#]] || ListQ@cs["BasicJobTicketSchema"][#], cs["BasicJobTicketSchema"][#], 
        ToExpression@cs["BasicJobTicketSchema"][#]] & /@ DeleteCases[DeleteDuplicates[Join[Intersection[IntegerTags, Keys[cs["BasicJobTicketSchema"]]], Intersection[BooleanTags, Keys[cs["BasicJobTicketSchema"]]], Intersection[RealTags, Keys[cs["BasicJobTicketSchema"]]]]], dateTags]]; 
   		AssociateTo[cs, "BasicJobTicketSchema" -> xmpBJ]];
  
  If[xmpTPg =!= Missing["KeyAbsent", "PagedTextSchema"], AssociateTo[xmpTPg, # -> If[DateObjectQ[cs["PagedTextSchema"][#]] || ListQ@cs["PagedTextSchema"][#], cs["PagedTextSchema"][#], 
        ToExpression@cs["PagedTextSchema"][#]] & /@ DeleteCases[DeleteDuplicates[Join[Intersection[IntegerTags, Keys[cs["PagedTextSchema"]]], Intersection[BooleanTags, Keys[cs["PagedTextSchema"]]], Intersection[RealTags, Keys[cs["PagedTextSchema"]]]]], dateTags]]; 
  		AssociateTo[cs, "PagedTextSchema" -> xmpTPg]];
  
  If[pdf =!= Missing["KeyAbsent", "AdobePDFSchema"], AssociateTo[pdf, # -> If[DateObjectQ[cs["AdobePDFSchema"][#]] || ListQ@cs["AdobePDFSchema"][#], cs["AdobePDFSchema"][#], 
        ToExpression@cs["AdobePDFSchema"][#]] & /@ DeleteCases[DeleteDuplicates[Join[Intersection[IntegerTags, Keys[cs["AdobePDFSchema"]]], Intersection[BooleanTags, Keys[cs["AdobePDFSchema"]]], Intersection[RealTags, Keys[cs["AdobePDFSchema"]]]]], dateTags]]; 
   		AssociateTo[cs, "AdobePDFSchema" -> pdf]];
  
  cs
  ]
  
  ParseStringTagsXMP[state_] := Module[{cs = state, Iptc4xmpCore = state["IPTCCoreSchema"], GPano = state["GooglePhotoSphereSchema"], xmpMM = state["MediaManagementSchema"], dc = state["DublinCoreSchema"], xmp = state["BasicSchema"], digiKam = state["PhotoManagementSchema"], crs = state["CameraRawSchema"], MicrosoftPhoto = state["MicrosoftPhotoSchema"], 
   									    photoshop = state["PhotoshopSchema"], xmpRights = state["RightsManagementSchema"], xmpBJ = state["BasicJobTicketSchema"], xmpTPg = state["PagedTextSchema"], pdf = state["AdobePDFSchema"]},
  
  If[Iptc4xmpCore =!= Missing["KeyAbsent", "IPTCCoreSchema"], AssociateTo[Iptc4xmpCore, # ->  If[StringQ@cs["IPTCCoreSchema"][#], StringTrim@cs["IPTCCoreSchema"][#], cs["IPTCCoreSchema"][#]] & /@ Keys[cs["IPTCCoreSchema"]]]; AssociateTo[cs, "IPTCCoreSchema" -> Iptc4xmpCore]];
  
  If[GPano =!= Missing["KeyAbsent", "GooglePhotoSphereSchema"], AssociateTo[GPano, # -> If[StringQ@cs["GooglePhotoSphereSchema"][#], StringTrim@cs["GooglePhotoSphereSchema"][#], cs["GooglePhotoSphereSchema"][#]] & /@ Keys[cs["GooglePhotoSphereSchema"]]]; AssociateTo[cs, "GooglePhotoSphereSchema" -> GPano]];
  
  If[xmpMM =!= Missing["KeyAbsent", "MediaManagementSchema"], AssociateTo[xmpMM, # -> If[StringQ@cs["MediaManagementSchema"][#], StringTrim@cs["MediaManagementSchema"][#], cs["MediaManagementSchema"][#]] & /@ Keys[cs["MediaManagementSchema"]]]; AssociateTo[cs, "MediaManagementSchema" -> xmpMM]];
  
  If[dc =!= Missing["KeyAbsent", "DublinCoreSchema"], AssociateTo[dc, # -> If[StringQ@cs["DublinCoreSchema"][#], StringTrim@cs["DublinCoreSchema"][#], cs["DublinCoreSchema"][#]] & /@ Keys[cs["DublinCoreSchema"]]]; AssociateTo[cs, "DublinCoreSchema" -> dc]]; 
  
  If[xmp =!= Missing["KeyAbsent", "BasicSchema"], AssociateTo[xmp, # ->   If[StringQ@cs["BasicSchema"][#], StringTrim@cs["BasicSchema"][#], cs["BasicSchema"][#]] & /@ Keys[cs["BasicSchema"]]]; AssociateTo[cs, "BasicSchema" -> xmp]]; 
  
  If[digiKam =!= Missing["KeyAbsent", "PhotoManagementSchema"], AssociateTo[digiKam, # -> If[StringQ@cs["PhotoManagementSchema"][#], StringTrim@cs["PhotoManagementSchema"][#], cs["PhotoManagementSchema"][#]] & /@ Keys[cs["PhotoManagementSchema"]]]; AssociateTo[cs, "PhotoManagementSchema" -> digiKam]];
  
  If[crs =!= Missing["KeyAbsent", "CameraRawSchema"], AssociateTo[crs, # -> If[StringQ@cs["CameraRawSchema"][#], StringTrim@cs["CameraRawSchema"][#], cs["CameraRawSchema"][#]] & /@ Keys[cs["CameraRawSchema"]]]; AssociateTo[cs, "CameraRawSchema" -> crs]];
  
  If[MicrosoftPhoto =!= Missing["KeyAbsent", "MicrosoftPhotoSchema"], AssociateTo[MicrosoftPhoto, # -> If[StringQ@cs["MicrosoftPhotoSchema"][#], StringTrim@cs["MicrosoftPhotoSchema"][#], cs["MicrosoftPhotoSchema"][#]] & /@ Keys[cs["MicrosoftPhotoSchema"]]]; AssociateTo[cs, "MicrosoftPhotoSchema" -> MicrosoftPhoto]];
  
  If[photoshop =!= Missing["KeyAbsent", "PhotoshopSchema"], AssociateTo[photoshop, # -> If[StringQ@cs["PhotoshopSchema"][#], StringTrim@cs["PhotoshopSchema"][#], cs["PhotoshopSchema"][#]] & /@ Keys[cs["PhotoshopSchema"]]]; AssociateTo[cs, "PhotoshopSchema" -> photoshop]];
  
  If[xmpRights =!= Missing["KeyAbsent", "RightsManagementSchema"], AssociateTo[xmpRights, # -> If[StringQ@cs["RightsManagementSchema"][#], StringTrim@cs["RightsManagementSchema"][#], cs["RightsManagementSchema"][#]] & /@ Keys[cs["RightsManagementSchema"]]]; AssociateTo[cs, "RightsManagementSchema" -> xmpRights]];
  
  If[xmpBJ =!= Missing["KeyAbsent", "BasicJobTicketSchema"], AssociateTo[xmpBJ, # -> If[StringQ@cs["BasicJobTicketSchema"][#], StringTrim@cs["BasicJobTicketSchema"][#], cs["BasicJobTicketSchema"][#]] & /@ Keys[cs["BasicJobTicketSchema"]]]; AssociateTo[cs, "BasicJobTicketSchema" -> xmpBJ]];
  
  If[xmpTPg =!= Missing["KeyAbsent", "PagedTextSchema"], AssociateTo[xmpTPg, # -> If[StringQ@cs["PagedTextSchema"][#], StringTrim@cs["PagedTextSchema"][#], cs["PagedTextSchema"][#]] & /@ Keys[cs["PagedTextSchema"]]]; AssociateTo[cs, "PagedTextSchema" -> xmpTPg]];
  
  If[pdf =!= Missing["KeyAbsent", "AdobePDFSchema"], AssociateTo[pdf, # -> If[StringQ@cs["AdobePDFSchema"][#], StringTrim@cs["AdobePDFSchema"][#], cs["AdobePDFSchema"][#]] & /@ Keys[cs["AdobePDFSchema"]]]; AssociateTo[cs, "AdobePDFSchema" -> pdf]];
  
  cs
  ]
  
  ParseValuesInGroupsXMP[valEx_] := 
                              Module[{curState = valEx},
                                  curState = ParseDateTimeTagsXMP[curState];
                                  curState = ParseMultiValueTagsXMP[curState];
                                  curState = ParseIntAndRealTagsXMP[curState];
                                  curState = ParseStringTagsXMP[curState];
                                  curState = ParseIndividualTagsXMP[curState];
                                  curState
                              ]

GetXMPAll[] :=
    With[{tmp = validatePossibleAssociation[$ReadXMPAll[]]},
		If[tmp === "<||>",
			<||>,
			ParseValuesInGroupsXMP[ValidateXMPAssociation[tmp]]
		]
	]

 ParseTagsXMPRaw[state_] := Module[{cs = state},
	                             cs = Append[cs, # -> Missing["NotAvailable"] & /@  DeleteCases[$AllXMP, Alternatives @@ Sequence @@@ Keys[cs]]];
  								 cs
  ]

ReadXMP[tag_, rule_ : False] :=
	Block[{$Context = "XMPTools`TempContext`"},
		Module[{name = tag},
			If[SameQ[name, "All"], GetXMPAll[],
				If[SameQ[name, "AllRaw"],
					Module[
						{
							resTmp = validatePossibleAssociation[$ReadXMPAllRaw[]],
							tmp
						},
						If[resTmp === "<||>",
							<||>,
							tmp = ValidateXMPAssociationRaw[resTmp];
							If[Quiet[AssociationQ[tmp]], tmp, <||>]
						]
					]
				]
			]
		]
	]
		
ReadXMP[tag_] := ReadXMP[tag, False]

(**************************)
(**************************)
(**************************)
(**********EXPORT**********)
(************XMP***********)
(**************************)
(**************************)

(* Interface function to parse the values that were processed during import, back to raw version. *)
XMPProcessToRaw[Rule[key_, assoc_Association]] :=
	Rule[
		key,
		AssociationMap[
			XMPProcessToRaw,
			assoc
		]
	];

(* A function to parse a key-value pair, that was processed during import, back to raw version. *)
XMPProcessToRaw[Rule[key_, val_]] :=
	If[
		And[
			MemberQ[DateTags, key],
			DateObjectQ[val]
		]
		,
		Switch[DateValue[val, "Granularity"],
			"Instant",
			Rule[key,
				DateString[val, {"Year", "-", "MonthShort", "-", "DayShort"}]
			]
			,
			"Day",
			Rule[key,
				DateString[val, {"Year", "-", "MonthShort", "-", "DayShort", " ", "HourShort", ":", "MinuteShort", ":", "SecondShort"}]
			]
			,
			_, (* This should technically never happen, since a date object is always supposed to represent either date or date-and-time. *)
			Rule[key, "1970-01-01 00:00:00"] (* This is the UNIX digital time start. *)
		]
		,
	(*else*)
		If[MatchQ[val, Missing[_]], (* This should technically never happen, since all the missing values are removed from processed representation during import. *)
			Rule[key, 0]
			,
		(*else*)
			Rule[key,
				If[ListQ[val], (* If the values is a list (i.e. {1, 2, 3, 4}) we have to convert it to XMP understandable representation (i.e. "1, 2, 3, 4"). *)
					StringTake[ToString[val], {2, -2}]
					,
				(*else*)
					Normal @@ val (* This is an ultimate solution for getting the actual value from Quantity, Integer, Real, String, etc.: all in one case. *)
				]
			]
		]
	];

PrepareXMPMetaFromProcessWithoutDomains[res_] := Block[{$Context = "XMPTools`TempContext`"}, 
	                                                 AssociationMap[XMPProcessToRaw, DeleteCases[Association@KeyValueMap[#1 -> DeleteCases[#2, _?(StringMatchQ[ToString@#, Whitespace ..] &)] &, res], _?(# == <||> &)]]
	                                             ]

constractDomainList[domain_, ass_Association] := Module[{dom = domain},
  													XMPDomainName[Rule[key_, assoc_Association]] := Rule[key, AssociationMap[XMPDomainName, assoc] ];
      												XMPDomainName[Rule[key_, val_] ] := Rule[StringJoin["Xmp.", dom, ".", key], val];
  													ConstructXMPDomain[res_] := AssociationMap[XMPDomainName, DeleteCases[Association@KeyValueMap[#1 -> DeleteCases[#2, _?(StringMatchQ[ToString@#, Whitespace ..] &)] &, res], _?(# == <||> &)]];
  													ConstructXMPDomain[ass]
]

flattenStructuralTag[tag_, code_, ass_Association] := Module[{tagname = tag, tagCode = code},
	 												StructureName[Rule[key_, assoc_Association]] := Rule[key, AssociationMap[StructureName, assoc]];
  													StructureName[Rule[key_, val_]] := Rule[StringJoin[StringJoin[StringSplit[key, "."][[1]], "." , StringSplit[key, "."][[2]], "."], tagname, tagCode, StringSplit[key, "."][[3]]], val];
  													ConstructStructure[res_] := AssociationMap[StructureName, DeleteCases[Association@KeyValueMap[#1 -> DeleteCases[#2, _?(StringMatchQ[ToString@#, Whitespace ..] &)] &, res], _?(# == <||> &)]];
  													
  													ConstructStructure[ass]]

(*
DerivedFrom - ["DerivedFrom", "DerivedFrom"], /stRef:
History - History[1]...[4] /stEvt:
RenditionOf - RenditionOf[1]...[4] /stEvt:
ManagedFrom - ManagedFrom[1]...[4] /stEvt:
JobRef / JobRef[1]...[4] /stJob:
Colorant - /xmpG:
*)

concat[assc_] := Module[{WRI = <||>, Iptc4xmpCore = <||>, GPano = <||>,  xmpMM = <||>, dc = <||>, xmp = <||>, digiKam = <||>, crs = <||>, MicrosoftPhoto = <||>, photoshop = <||>, xmpRights = <||>, xmpBJ = <||>, xmpTPg = <||>, pdf = <||>, 
   																																												final = <||>},
   						  If[assc["IPTCCoreSchema"] =!= Missing["KeyAbsent", "IPTCCoreSchema"], Iptc4xmpCore = assc["IPTCCoreSchema"]];
   						  If[assc["GooglePhotoSphereSchema"] =!= Missing["KeyAbsent", "GooglePhotoSphereSchema"], GPano = assc["GooglePhotoSphereSchema"]];
  					 	  If[assc["MediaManagementSchema"] =!= Missing["KeyAbsent", "MediaManagementSchema"], xmpMM = assc["MediaManagementSchema"]];
  						  If[assc["AdobePDFSchema"] =!= Missing["KeyAbsent", "AdobePDFSchema"], pdf = assc["AdobePDFSchema"]];
						  If[assc["DublinCoreSchema"] =!= Missing["KeyAbsent", "DublinCoreSchema"], dc = assc["DublinCoreSchema"]];
						  If[assc["BasicSchema"] =!= Missing["KeyAbsent", "BasicSchema"], xmp = assc["BasicSchema"]];
						  If[assc["PhotoManagementSchema"] =!= Missing["KeyAbsent", "PhotoManagementSchema"], digiKam = assc["PhotoManagementSchema"]];
						  If[assc["MicrosoftPhotoSchema"] =!= Missing["KeyAbsent", "MicrosoftPhotoSchema"], MicrosoftPhoto = assc["MicrosoftPhotoSchema"]];
						  If[assc["CameraRawSchema"] =!= Missing["KeyAbsent", "CameraRawSchema"],crs = assc["CameraRawSchema"]];
						  If[assc["PhotoshopSchema"] =!= Missing["KeyAbsent", "PhotoshopSchema"], photoshop = assc["PhotoshopSchema"]];
						  If[assc["RightsManagementSchema"] =!= Missing["KeyAbsent", "RightsManagementSchema"], xmpRights = assc["RightsManagementSchema"]];
						  If[assc["BasicJobTicketSchema"] =!= Missing["KeyAbsent", "BasicJobTicketSchema"], xmpBJ = assc["BasicJobTicketSchema"]];
						  If[assc["PagedTextSchema"] =!= Missing["KeyAbsent", "PagedTextSchema"], xmpTPg = assc["PagedTextSchema"]];
						  If[assc["WolframSchema"] =!= Missing["KeyAbsent", "WolframSchema"], WRI = assc["WolframSchema"]];
  
  						  If[Iptc4xmpCore =!= <||>, Iptc4xmpCore = constractDomainList["iptc", Iptc4xmpCore]];
						  If[GPano =!= <||>, GPano = constractDomainList["GPano", GPano]];
						  If[xmpMM =!= <||>, xmpMM = constractDomainList["xmpMM", xmpMM]];
						  If[pdf =!= <||>, pdf = constractDomainList["pdf", pdf]];
						  If[dc =!= <||>, dc = constractDomainList["dc", dc]];
						  If[xmp =!= <||>, xmp = constractDomainList["xmp", xmp]];
						  If[digiKam =!= <||>, digiKam = constractDomainList["digiKam", digiKam]];
						  If[MicrosoftPhoto =!= <||>, MicrosoftPhoto = constractDomainList["MicrosoftPhoto", MicrosoftPhoto]];
						  If[crs =!= <||>, crs = constractDomainList["crs", crs]];
						  If[photoshop =!= <||>, photoshop = constractDomainList["photoshop", photoshop]];
						  If[xmpRights =!= <||>, xmpRights = constractDomainList["xmpRights", xmpRights]];
						  If[xmpBJ =!= <||>, xmpBJ = constractDomainList["xmpBJ", xmpBJ]];
						  If[xmpTPg =!= <||>, xmpTPg = constractDomainList["xmpTPg", xmpTPg]];
						  If[WRI =!= <||>, WRI = constractDomainList["Wolfram", WRI]];
  
						  If[Iptc4xmpCore =!= <||>, final = Join[final, Iptc4xmpCore]];
						  If[GPano =!= <||>, final = Join[final, GPano]];
						  If[xmpMM =!= <||>, final = Join[final, xmpMM]];
						  If[pdf =!= <||>, final = Join[final, pdf]];
						  If[dc =!= <||>, final = Join[final, dc]];
						  If[xmp =!= <||>, final = Join[final, xmp]];
						  If[digiKam =!= <||>, final = Join[final, digiKam]];
						  If[MicrosoftPhoto =!= <||>, final = Join[final, MicrosoftPhoto]];
						  If[crs =!= <||>, final = Join[final, crs]];
						  If[photoshop =!= <||>, final = Join[final, photoshop]];
						  If[xmpRights =!= <||>, final = Join[final, xmpRights]];
						  If[xmpBJ =!= <||>, final = Join[final, xmpBJ]];
						  If[xmpTPg =!= <||>, final = Join[final, xmpTPg]];
						  If[WRI =!= <||>, final = Join[final, WRI]];
  
  						  final
  ]

separateStructureTagsAndConstructBack[asc_]:= Module[{final = <||>, assc = asc, ver = <||>, sgroups=<||>, col = <||>, his = <||>, der = <||>, rend = <||>, manag = <||>, job = <||>, h = <||>, d = <||>, 
  																														   v=<||>, cci = <||>, c=<||>, r = <||>, m = <||>, j = <||>},
 													  final = assc;
 
 													  If[assc["History"] =!= Missing["KeyAbsent", "History"],
														  his = assc["History"];
														  final = KeyDrop[final, "History"];
  
  														  h = Module[{fin = <||>, his1 = <||>, his2 = <||>, his3 = <||>, his4 = <||>},
																	    
																	    If[his["History[1]"] =!= Missing["KeyAbsent", "History[1]"],
																	     his1 = flattenStructuralTag["History[1]", "/stEvt:", his["History[1]"]];
																	     PrependTo[his1, <|"Xmp.xmpMM.History[1]" -> "type= Seq"|>];
																	     ];
																	    
																	    If[his["History[2]"] =!= Missing["KeyAbsent", "History[2]"],
																	     his2 = flattenStructuralTag["History[2]", "/stEvt:", his["History[2]"]];
																	     PrependTo[his2, <|"Xmp.xmpMM.History[2]" -> "type= Seq"|>];
																	     ];
																	    
																	    If[his["History[3]"] =!= Missing["KeyAbsent", "History[3]"],
																	     his3 = flattenStructuralTag["History[3]", "/stEvt:", his["History[3]"]];
																	     PrependTo[his3, <|"Xmp.xmpMM.History[3]" -> "type= Seq"|>];
																	     ];
																	    
																	    If[his["History[4]"] =!= Missing["KeyAbsent", "History[4]"],
																	     his4 = flattenStructuralTag["History[4]", "/stEvt:", his["History[4]"]];
																	     PrependTo[his4, <|"Xmp.xmpMM.History[4]" -> "type= Seq"|>];
																	     ];
    
																	    If[his1 =!= <||>, fin = Join[fin, his1]];
																	    If[his2 =!= <||>, fin = Join[fin, his2]];
																	    If[his3 =!= <||>, fin = Join[fin, his3]];
																	    If[his4 =!= <||>, fin = Join[fin, his4]];
																	    fin
    																];
															  		If[h =!= <||>, PrependTo[h, <|"Xmp.xmpMM.History" -> "type= Seq"|>]];
															  		AppendTo[final, h];
															  ];
															  
													If[assc["CreatorContactInfo"] =!= Missing["KeyAbsent", "CreatorContactInfo"],
														  cci = assc["CreatorContactInfo"];
														  final = KeyDrop[final, "CreatorContactInfo"];
  
  														  h = Module[{fin = <||>, CiAdrCity = <||>, CiAdrCtry = <||>, CiAdrExtadr = <||>, CiAdrPcode = <||>, CiUrlWork = <||>},
																	    
																	    If[cci["Xmp.iptc.CiAdrCity"] =!= Missing["KeyAbsent", "CiAdrCity"],
																	     PrependTo[CiAdrCity, <|"Xmp.iptc.CreatorContactInfo/Iptc4xmpCore:CiAdrCity" -> cci["Xmp.iptc.CiAdrCity"]|>];
																	     ];
																	     
																	    If[cci["Xmp.iptc.CiAdrCtry"] =!= Missing["KeyAbsent", "CiAdrCtry"],
																	     PrependTo[CiAdrCtry, <|"Xmp.iptc.CreatorContactInfo/Iptc4xmpCore:CiAdrCtry" -> cci["Xmp.iptc.CiAdrCtry"]|>];
																	     ];
																	     
																	    If[cci["Xmp.iptc.CiAdrExtadr"] =!= Missing["KeyAbsent", "CiAdrExtadr"],
																	     PrependTo[CiAdrExtadr, <|"Xmp.iptc.CreatorContactInfo/Iptc4xmpCore:CiAdrExtadr" -> cci["Xmp.iptc.CiAdrExtadr"]|>];
																	     ];
																	     
																	    If[cci["Xmp.iptc.CiAdrPcode"] =!= Missing["KeyAbsent", "CiAdrPcode"],
																	     PrependTo[CiAdrPcode, <|"Xmp.iptc.CreatorContactInfo/Iptc4xmpCore:CiAdrPcode" -> cci["Xmp.iptc.CiAdrPcode"]|>];
																	     ];
																	     
																	    If[cci["Xmp.iptc.CiUrlWork"] =!= Missing["KeyAbsent", "CiUrlWork"],
																	     PrependTo[CiUrlWork, <|"Xmp.iptc.CreatorContactInfo/Iptc4xmpCore:CiUrlWork" -> cci["Xmp.iptc.CiUrlWork"]|>];
																	     ];
																	    

    
																	    If[CiAdrCity =!= <||>, fin = Join[fin, CiAdrCity]];
																	    If[CiAdrCtry =!= <||>, fin = Join[fin, CiAdrCtry]];
																	    If[CiAdrExtadr =!= <||>, fin = Join[fin, CiAdrExtadr]];
																	    If[CiAdrPcode =!= <||>, fin = Join[fin, CiAdrPcode]];
																	    If[CiUrlWork =!= <||>, fin = Join[fin, CiUrlWork]];
																	    fin
    																];
															  		If[h =!= <||>, PrependTo[h, <|"Xmp.iptc.CreatorContactInfo" -> "type= Struct"|>]];
															  		AppendTo[final, h];
															  ];
															  		  
													If[assc["Colorants"] =!= Missing["KeyAbsent", "Colorants"],
														  col = assc["Colorants"];
														  final = KeyDrop[final, "Colorants"];
  
  														  c = Module[{fin = <||>, col1 = <||>, col2 = <||>, col3 = <||>, col4 = <||>},
																	    
																	    If[col["Colorants[1]"] =!= Missing["KeyAbsent", "Colorants[1]"],
																	     col1 = flattenStructuralTag["Colorants[1]", "/xmpG:", col["Colorants[1]"]];
																	     PrependTo[col1, <|"Xmp.xmpTPg.Colorants[1]" -> "type= Seq"|>];
																	     ];
																	    
																	    If[col["Colorants[2]"] =!= Missing["KeyAbsent", "Colorants[2]"],
																	     col2 = flattenStructuralTag["Colorants[2]", "/xmpG:", col["Colorants]"]];
																	     PrependTo[col2, <|"Xmp.xmpTPg.Colorants[2]" -> "type= Seq"|>];
																	     ];
																	    
																	    If[col["Colorants[3]"] =!= Missing["KeyAbsent", "Colorants[3]"],
																	     col3 = flattenStructuralTag["Colorants[3]", "/xmpG:", col["Colorants[3]"]];
																	     PrependTo[col3, <|"Xmp.xmpTPg.Colorants[3]" -> "type= Seq"|>];
																	     ];
																	    
																	    If[col["Colorants[4]"] =!= Missing["KeyAbsent", "Colorants[4]"],
																	     col4 = flattenStructuralTag["Colorants[4]", "/xmpG:", col["Colorants[4]"]];
																	     PrependTo[col4, <|"Xmp.xmpTPg.Colorants[4]" -> "type= Seq"|>];
																	     ];
    
																	    If[col1 =!= <||>, fin = Join[fin, col1]];
																	    If[col2 =!= <||>, fin = Join[fin, col2]];
																	    If[col3 =!= <||>, fin = Join[fin, col3]];
																	    If[col4 =!= <||>, fin = Join[fin, col4]];
																	    fin
    																];
															  		If[c =!= <||>, PrependTo[c, <|"Xmp.xmpTPg.Colorants" -> "type= Seq"|>]];
															  		AppendTo[final, c];
															  ];
													(*		 
													If[assc["SwatchGroups"] =!= Missing["KeyAbsent", "SwatchGroups"],
														  sgroups = assc["SwatchGroups"];
														  final = KeyDrop[final, "SwatchGroups"];
  
  														  sg = Module[{fin = <||>, sg1 = <||>, sg2 = <||>, sg3 = <||>, sg4 = <||>},
																	    
																	    If[col["SwatchGroups[1]"] =!= Missing["KeyAbsent", "SwatchGroups[1]"],
																	     sg1 = flattenStructuralTag["SwatchGroups[1]", "/xmpG:", sgroups["SwatchGroups[1]"]];
																	     PrependTo[sg1, <|"Xmp.xmpTPg.SwatchGroups[1]" -> "type= Seq"|>];
																	     ];
																	    
																	    If[col["SwatchGroups[2]"] =!= Missing["KeyAbsent", "SwatchGroups[2]"],
																	     sg2 = flattenStructuralTag["SwatchGroups[2]", "/xmpG:", sgroups["SwatchGroups]"]];
																	     PrependTo[sg2, <|"Xmp.xmpTPg.SwatchGroups[2]" -> "type= Seq"|>];
																	     ];
																	    
																	    If[col["SwatchGroups[3]"] =!= Missing["KeyAbsent", "SwatchGroups[3]"],
																	     sg3 = flattenStructuralTag["SwatchGroups[3]", "/xmpG:", sgroups["SwatchGroups[3]"]];
																	     PrependTo[sg3, <|"Xmp.xmpTPg.SwatchGroups" -> "type= Seq"|>];
																	     ];
																	    
																	    If[col["SwatchGroups[4]"] =!= Missing["KeyAbsent", "SwatchGroups[4]"],
																	     sg4 = flattenStructuralTag["SwatchGroups[4]", "/xmpG:", sgroups["SwatchGroups[4]"]];
																	     PrependTo[sg4, <|"Xmp.xmpTPg.SwatchGroups" -> "type= Seq"|>];
																	     ];
    
																	    If[sg1 =!= <||>, fin = Join[fin, sg1]];
																	    If[sg2 =!= <||>, fin = Join[fin, sg2]];
																	    If[sg3 =!= <||>, fin = Join[fin, sg3]];
																	    If[sg4 =!= <||>, fin = Join[fin, sg4]];
																	    fin
    																];
															  		If[sg =!= <||>, PrependTo[sg, <|"Xmp.xmpTPg.SwatchGroups" -> "type= Seq"|>]];
															  		AppendTo[final, sg];
															  ];		  		  
                                                    *)
 													If[assc["Versions"] =!= Missing["KeyAbsent", "Versions"],
														  ver = assc["Versions"];
														  final = KeyDrop[final, "Versions"];
  
  														  v = Module[{fin = <||>, ver1 = <||>, ver2 = <||>, ver3 = <||>, ver4 = <||>},
																	    
																	    If[ver["Versions[1]"] =!= Missing["KeyAbsent", "Versions[1]"],
																	     ver1 = flattenStructuralTag["Versions[1]", "/stEvt:", ver["Versions[1]"]];
																	     PrependTo[ver1, <|"Xmp.xmpMM.Versions[1]" -> "type= Seq"|>];
																	     ];
																	    
																	    If[ver["Versions[2]"] =!= Missing["KeyAbsent", "Versions[2]"],
																	     ver2 = flattenStructuralTag["Versions[2]", "/stEvt:", ver["Versions[2]"]];
																	     PrependTo[ver2, <|"Xmp.xmpMM.Versions[2]" -> "type= Seq"|>];
																	     ];
																	    
																	    If[ver["Versions[3]"] =!= Missing["KeyAbsent", "Versions[3]"],
																	     ver3 = flattenStructuralTag["Versions[3]", "/stEvt:", ver["Versions[3]"]];
																	     PrependTo[ver3, <|"Xmp.xmpMM.Versions[3]" -> "type= Seq"|>];
																	     ];
																	    
																	    If[ver["Versions[4]"] =!= Missing["KeyAbsent", "Versions[4]"],
																	     ver4 = flattenStructuralTag["Versions[4]", "/stEvt:", ver["Versions[4]"]];
																	     PrependTo[ver1, <|"Xmp.xmpMM.Versions[4]" -> "type= Seq"|>];
																	     ];
    
																	    If[ver1 =!= <||>, fin = Join[fin, ver1]];
																	    If[ver2 =!= <||>, fin = Join[fin, ver2]];
																	    If[ver3 =!= <||>, fin = Join[fin, ver3]];
																	    If[ver4 =!= <||>, fin = Join[fin, ver4]];
																	    fin
    																];
															  		If[v =!= <||>, PrependTo[v, <|"Xmp.xmpMM.Versions" -> "type= Seq"|>]];
															  		AppendTo[final, v];
															  ];
 
 													  If[assc["DerivedFrom"] =!= Missing["KeyAbsent", "DerivedFrom"],
														  der = assc["DerivedFrom"];
														  final = KeyDrop[final, "DerivedFrom"];
														  
  														  d = Module[{fin = <||>, der0 = <||>,der1 = <||>, der2 = <||>, der3 = <||>, der4 = <||>},
																
																If[der["DerivedFrom"] =!= Missing["KeyAbsent", "DerivedFrom"],
															     der0 = flattenStructuralTag["DerivedFrom", "/stRef:", der["DerivedFrom"]];
															     PrependTo[der0, <|"Xmp.xmpMM.DerivedFrom" -> " type= Struct"|>];
															     ];
																	    
															    If[der["DerivedFrom[1]"] =!= Missing["KeyAbsent", "DerivedFrom[1]"],
															     der1 = flattenStructuralTag["DerivedFrom[1]", "/stRef:", der["DerivedFrom[1]"]];
															     PrependTo[der1, <|"Xmp.xmpMM.DerivedFrom[1]" -> " type= Struct"|>];
															     ];
															    
															    If[der["DerivedFrom[2]"] =!= Missing["KeyAbsent", "DerivedFrom[2]"],
															     der2 = flattenStructuralTag["DerivedFrom[2]", "/stRef:", der["DerivedFrom[2]"]];
															     PrependTo[der2, <|"Xmp.xmpMM.DerivedFrom[2]" -> " type= Struct"|>];
															     ];
															    
															    If[der["DerivedFrom[3]"] =!= Missing["KeyAbsent", "DerivedFrom[3]"],
															     der3 = flattenStructuralTag["DerivedFrom[3]", "/stRef:", der["dertory[3]"]];
															     PrependTo[der3, <|"Xmp.xmpMM.DerivedFrom[3]" -> " type= Struct"|>];
															     ];
															    
															    If[der["DerivedFrom[4]"] =!= Missing["KeyAbsent", "DerivedFrom[4]"],
															     der4 = flattenStructuralTag["DerivedFrom[4]", "/stRef:", der["DerivedFrom[4]"]];
															     PrependTo[der1, <|"Xmp.xmpMM.DerivedFrom[4]" -> " type= Struct"|>];
															     ];
																
																If[der0 =!= <||>, fin = Join[fin, der0]];
															    If[der1 =!= <||>, fin = Join[fin, der1]];
															    If[der2 =!= <||>, fin = Join[fin, der2]];
															    If[der3 =!= <||>, fin = Join[fin, der3]];
															    If[der4 =!= <||>, fin = Join[fin, der4]];
															    fin
															];
															
													  	  If[d =!= <||>, PrependTo[d, <|"Xmp.xmpMM.DerivedFrom" -> " type= Struct"|>]];
													  	  AppendTo[final, d];
													  ];
 
 													  If[assc["RenditionOf"] =!= Missing["KeyAbsent", "RenditionOf"],
														  rend = assc["RenditionOf"];
														  final = KeyDrop[final, "RenditionOf"];
  
  														  r = Module[{fin = <||>, rend1 = <||>, rend2 = <||>, rend3 = <||>, rend4 = <||>},
  														  	
															    If[rend["RenditionOf[1]"] =!= Missing["KeyAbsent", "RenditionOf[1]"],
															     rend1 = flattenStructuralTag["RenditionOf[1]", "/stEvt:", rend["RenditionOf[1]"]];
															     PrependTo[rend1, <|"Xmp.xmpMM.RenditionOf[1]" -> " type= Struct"|>];
															     ];
															    
															    If[rend["RenditionOf[2]"] =!= Missing["KeyAbsent", "RenditionOf[2]"],
															     rend2 = flattenStructuralTag["RenditionOf[2]", "/stEvt:", rend["RenditionOf[2]"]];
															     PrependTo[rend2, <|"Xmp.xmpMM.RenditionOf[2]" -> " type= Struct"|>];
															     ];
															    
															    If[rend["RenditionOf[3]"] =!= Missing["KeyAbsent", "RenditionOf[3]"],
															     rend3 = flattenStructuralTag["RenditionOf[3]", "/stEvt:", rend["RenditionOf[3]"]];
															     PrependTo[rend3, <|"Xmp.xmpMM.RenditionOf[3]" -> " type= Struct"|>];
															     ];
															    
															    If[rend["RenditionOf[4]"] =!= Missing["KeyAbsent", "RenditionOf[4]"],
															     rend4 = flattenStructuralTag["RenditionOf[4]", "/stEvt:", rend["RenditionOf[4]"]];
															     PrependTo[rend4, <|"Xmp.xmpMM.RenditionOf[4]" -> " type= Struct"|>];
															     ];
															    
															    If[rend1 =!= <||>, fin = Join[fin, rend1]];
															    If[rend2 =!= <||>, fin = Join[fin, rend2]];
															    If[rend3 =!= <||>, fin = Join[fin, rend3]];
															    If[rend4 =!= <||>, fin = Join[fin, rend4]];
															    fin
															 ];
  															If[r =!= <||>, PrependTo[r, <|"Xmp.xmpMM.RenditionOf" -> " type= Seq"|>]];
  															AppendTo[final, r];
  														];
 
 														If[assc["ManagedFrom"] =!= Missing["KeyAbsent", "ManagedFrom"],
															  manag = assc["ManagedFrom"];
															  final = KeyDrop[final, "ManagedFrom"];
  
 															  m = Module[{fin = <||>, manag1 = <||>, manag2 = <||>, manag3 = <||>, manag4 = <||>},
																	 
																	    If[manag["ManagedFrom[1]"] =!= Missing["KeyAbsent", "ManagedFrom[1]"],
																	       manag1 = flattenStructuralTag["ManagedFrom[1]", "/stEvt:", manag["ManagedFrom[1]"]];
																	       PrependTo[manag1, <|"Xmp.xmpMM.ManagedFrom[1]" -> " type= Struct"|>];
																	     ];
																	    
																	    If[manag["ManagedFrom[2]"] =!= Missing["KeyAbsent", "ManagedFrom[2]"],
																	       manag2 = flattenStructuralTag["ManagedFrom[2]", "/stEvt:", manag["ManagedFrom[2]"]];
																	       PrependTo[manag2, <|"Xmp.xmpMM.ManagedFrom[2]" -> " type= Struct"|>];
																	     ];
																	    
																	    If[manag["ManagedFrom[3]"] =!= Missing["KeyAbsent", "ManagedFrom[3]"],
																	       manag3 = flattenStructuralTag["ManagedFrom[3]", "/stEvt:", manag["ManagedFrom[3]"]];
																	       PrependTo[manag3, <|"Xmp.xmpMM.ManagedFrom[3]" -> " type= Struct"|>];
																	     ];
																	    
																	    If[manag["ManagedFrom[4]"] =!= Missing["KeyAbsent", "ManagedFrom[4]"],
																	       manag4 = flattenStructuralTag["ManagedFrom[4]", "/stEvt:", manag["ManagedFrom[4]"]];
																	       PrependTo[manag4, <|"Xmp.xmpMM.ManagedFrom[4]" -> " type= Struct"|>];
																	     ];
																	    
																	    If[manag1 =!= <||>, fin = Join[fin, manag1]];
																	    If[manag2 =!= <||>, fin = Join[fin, manag2]];
																	    If[manag3 =!= <||>, fin = Join[fin, manag3]];
																	    If[manag4 =!= <||>, fin = Join[fin, manag4]];
																	    fin
																   ];
																  If[m =!= <||>, PrependTo[m, <|"Xmp.xmpMM.ManagedFrom" -> " type= Seq"|>]];
																  AppendTo[final, m];
															 ];
 
 														 If[assc["JobRef"] =!= Missing["KeyAbsent", "JobRef"], 
 														 	 job = assc["JobRef"];
  															 final = KeyDrop[final, "JobRef"];
  
  															 j = Module[{fin = <||>, job1 = <||>, job2 = <||>, job3 = <||>, job4 = <||>},

													                     If[job["JobRef[1]"] =!= Missing["KeyAbsent", "JobRef[1]"], 
													                       job1 = flattenStructuralTag["JobRef[1]", "/stJob:", job["JobRef[1]"]];
													                       PrependTo[job1, <|"Xmp.xmpBJ.JobRef[1]" -> " type= Struct"|>];
													                     ];
													    
													                     If[job["JobRef[2]"] =!= Missing["KeyAbsent", "JobRef[2]"],
													                       job2 = flattenStructuralTag["JobRef[2]", "/stJob:", job["JobRef[2]"]];
													                       PrependTo[job2, <|"Xmp.xmpBJ.JobRef[2]" -> " type= Struct"|>];
													                     ];
													    
													                     If[job["JobRef[3]"] =!= Missing["KeyAbsent", "JobRef[3]"],
													                       job3 = flattenStructuralTag["JobRef[3]", "/stJob:", job["JobRef[3]"]];
													                       PrependTo[job3, <|"Xmp.xmpBJ.JobRef[3]" -> " type= Struct"|>];
													                     ];
													    
													                     If[job["JobRef[4]"] =!= Missing["KeyAbsent", "JobRef[4]"],
													                       job4 = flattenStructuralTag["JobRef[4]", "/stJob:", job["JobRef[4]"]];
													                       PrependTo[job4, <|"Xmp.xmpBJ.JobRef[4]" -> " type= Struct"|>];
													                     ];
													    
													                     If[job1 =!= <||>, fin = Join[fin, job1]];
													                     If[job2 =!= <||>, fin = Join[fin, job2]];
													                     If[job3 =!= <||>, fin = Join[fin, job3]];
													                     If[job4 =!= <||>, fin = Join[fin, job4]];
													                     fin
													               ];
  
  													         If[j =!= <||>, PrependTo[j, <|"Xmp.xmpBJ.JobRef" -> " type= Seq"|>]];
 													         AppendTo[final, j];
 												           ];
 
 					final
 ]

PrepareXMPMetaFromProcess[assoc_]:= Module[{firstStepRes = PrepareXMPMetaFromProcessWithoutDomains[assoc]},
											separateStructureTagsAndConstructBack[concat[firstStepRes]]
			                          ]

WriteXMP[tag_, val_] := Block[{$Context = "XMPTools`TempContext`"}, 
	                        $WriteXMPString[tag, ToString@val]
                        ]


WriteXMPRule[listOfRules : {__Rule}] := WriteXMP @@@ listOfRules

WriteXMPAssociation[in_Association]:=
	Block[{$Context = "XMPTools`TempContext`"},
		Module[
			{
				normalValues,
				structTags
			},

			structTags = {
				"Xmp.xmpTPg.Colorants", "Xmp.xmpTPg.SwatchGroups", "Xmp.xmpMM.DerivedFrom", "Xmp.xmpMM.RenditionOf",
				"Xmp.xmpMM.ManagedFrom", "Xmp.xmpMM.History", "Xmp.iptc.CreatorContactInfo", "Xmp.xmpMM.Versions",
				"Xmp.xmpBJ.JobRef"
			};


			normalValues =
				KeySelect[in,
					!(
						StringQ[#] &&
						StringMatchQ[#, Alternatives @@ Map[Function[x, x <> "*"], structTags]]
					) &
				];

			WriteXMPRule[Normal[normalValues]];

			Map[
				Function[x,
					If[KeyExistsQ[in, x],
						writeComplexStruct[KeySelect[in, StringQ[#] && StringMatchQ[#, x <> "*"] &], x]
					];
				]
				,
				structTags
			];
		];
	];


writeComplexStruct[in_Association, key_String] :=
	Module[
		{
			strK, strV,
			res, keys, vals
		},

		res = processSingleStructTag[in, key, #] & /@ Range[5];
		{keys, vals} = Apply[Join, Transpose[res], {1}];

		{strK, strV} = ToString /@ {keys, vals};

		If[StringLength[strK] > 5 && StringLength[strV] > 5 ,
			keys = StringTake[strK, {2, StringLength[strK] - 1}];
			vals = StringTake[strV, {2, StringLength[strV] - 1}];
			$WriteXMPStructure[keys, vals, key];
		];
	];

processSingleStructTag[in_, key_, ind_] :=
	Module[
		{

			h = KeySelect[in, StringQ[#] && StringMatchQ[#, key <> "[" <> ToString[ind] <> "]*"] &],
			keys = {}, values = {}
		},
		
		If[Positive[Length[h]],
			{keys, values} = StringRiffle[ToString /@ #, ", "] & /@ {Keys[h], Values[h]};
		];

		Return[{keys, values}];
	];
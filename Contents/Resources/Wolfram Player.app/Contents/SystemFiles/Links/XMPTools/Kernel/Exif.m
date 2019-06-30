(**************************)
(**************************)
(**************************)
(********VALIDATION********)
(**************************)
(**************************)
(**************************)
			
ValidateExif[res_] := AssociationMap[ExifObjectValidate, DeleteCases[Association@KeyValueMap[#1 ->  DeleteCases[#2, _?(StringMatchQ[ToString@#,Whitespace ..] &)] &, res], _?(# == <||> &)]]

ExifObjectValidate[Rule[key_, assoc_Association]] := Rule[key, AssociationMap[ExifObjectValidate, assoc]]
 ExifObjectValidate[Rule[key_, val_]] :=  Module[{miss = Rule[key, Missing["Disputed"]], vRes = Rule[key, val]}, 
 								              Which[
 									               (MemberQ[DateTags, key] && !DateObjectQ[val]), Rule[key, Missing["Disputed"]],
   
   									               (MemberQ[TimeTags, key] && !TimeObjectQ[val]), Rule[key, Missing["Disputed"]],
   
                                                   (MemberQ[QuantityTags, key] && !QuantityQ@val) , Rule[key, Missing["Disputed"]],
   
									               (SameQ[key, "FocalLength"] && (N@(List @@ val // First) === 0 || N@(List @@ val // First) === 0.)), Rule[key, Missing["Unknown"]],   
									  
   
                								   SameQ[key, "GPSDifferential"] || SameQ["ColorimetricReference", key], 
   									               If[If[NumberQ@val, val < 0 || val > 1 , !StringMatchQ[ToString@val, "Without correction", 
        							                   IgnoreCase -> True] && !StringMatchQ[ToString@val, "Differential correction applied", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Correction applied", IgnoreCase -> True]], miss, vRes],
   
   									               SameQ[key, "SubjectDistanceRange"], If[If[NumberQ@val, val < 0 || val > 3 , !StringMatchQ[ToString@val, "Unknown", IgnoreCase -> True] && 
   									                   !StringMatchQ[ToString@val, "Macro", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Close view", IgnoreCase -> True] && 
   									                   !StringMatchQ[ToString@val, "Distant view", IgnoreCase -> True]], miss, vRes],
   
   									               SameQ[key, "Sharpness"], If[If[NumberQ@val, val < 0 || val > 2, !StringMatchQ[ToString@val, "Normal", IgnoreCase -> True] && 
   									                   !StringMatchQ[ToString@val, "Soft", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Hard", IgnoreCase -> True] ], miss, vRes],
   
   								                   SameQ[key, "Saturation"], If[If[NumberQ@val, val < 0 || val > 2, !StringMatchQ[ToString@val, "Normal", IgnoreCase -> True] && 
   								                       !StringMatchQ[ToString@val, "Low", IgnoreCase -> True] && !StringMatchQ[ToString@val, "High", IgnoreCase -> True]], miss, vRes],
   								      
   								                   StringMatchQ[key, "SceneType"], If[If[NumberQ@val, val < 0 || val > 2, !StringMatchQ[ToString@val, "Directly photographed", IgnoreCase -> True]], miss, vRes],
   								      
   								                   SameQ[key, "FileSource"], If[If[NumberQ@val, val < 0 || val > 2, !StringMatchQ[ToString@val, "Film scanner", IgnoreCase -> True] && 
   								                       !StringMatchQ[ToString@val, "Reflexion print scanner", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Digital still camera", IgnoreCase -> True]], miss, vRes],
   
   									               SameQ[key, "Contrast"], If[If[NumberQ@val, val < 0 || val > 2, !StringMatchQ[ToString@val, "Normal", IgnoreCase -> True] && 
   									                   !StringMatchQ[ToString@val, "Soft", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Hard", IgnoreCase -> True] ], miss, vRes],
   
  									               SameQ[key, "GainControl"], If[If[NumberQ@val, (val < 0 || val > 4), !StringMatchQ[ToString@val, "None", IgnoreCase -> True] && 
  									                   !StringMatchQ[ToString@val, "Low gain up", IgnoreCase -> True] && !StringMatchQ[ToString@val, "High gain up", IgnoreCase -> True] && 
  									                   !StringMatchQ[ToString@val, "Low gain down", IgnoreCase -> True] && !StringMatchQ[ToString@val, "High gain down", IgnoreCase -> True] ], miss, vRes],
   
   									               SameQ[key, "SceneCaptureType"], If[ If[NumberQ@val, (val < 0 || val > 3), (!StringMatchQ[ToString@val, "Standard", IgnoreCase -> True] && 
   									                   !StringMatchQ[ToString@val, "Landscape", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Portrait", IgnoreCase -> True] && 
   									                   !StringMatchQ[ToString@val, "Night scene", IgnoreCase -> True])], miss, vRes],
      
                                                   SameQ[key, "PlanarConfiguration"], If[val=!=1 && val=!=2, miss, vRes],
   
   									               SameQ[key, "WhiteBalance"], If[If[NumberQ@val, val < 0 && val > 1 ,  !StringMatchQ[ToString@val, "Auto", IgnoreCase -> True] && 
   									                   !StringMatchQ[ToString@val, "Manual", IgnoreCase -> True]], miss, vRes],
   
   									               SameQ[key, "ExposureMode"], If[If[NumberQ@val, val < 0 || val > 2,  !StringMatchQ[ToString@val, "Auto", IgnoreCase -> True] &&  
   									                   !StringMatchQ[ToString@val, "Manual", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Auto bracket", IgnoreCase -> True] ], miss, vRes],
   
   									               SameQ[key, "SensingMethod"], If[If[NumberQ@val, val < 1 || val > 8,  !StringMatchQ[ToString@val, "Not defined", IgnoreCase -> True] && 
   									                   !StringMatchQ[ToString@val, "One-chip color area", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Two-chip color area", 
        							                   IgnoreCase -> True] && !StringMatchQ[ToString@val, "Three-chip color area", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Color sequential area", 
        							                   IgnoreCase -> True] && !StringMatchQ[ToString@val, "Trilinear", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Color sequential linear", 
        							                   IgnoreCase -> True]], miss, vRes],
   
   									               StringMatchQ[key, "FocalPlaneResolutionUnit"], If[If[NumberQ@val, val < 1 || val > 3 , !StringMatchQ[ToString@val, "No absolute unit of measurement", 
        							                   IgnoreCase -> True] && !StringMatchQ[ToString@val, "None", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Inch", IgnoreCase -> True] && !StringMatchQ[ToString@val, "cm", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Centimeter", IgnoreCase -> True]], 
    								                   miss, vRes],
      
   									               SameQ[key, "MeteringMode"], If[If[NumberQ@val, val < 0 || (val > 6 && val =!= 255), !StringMatchQ[ToString@val, "Unknown", IgnoreCase -> True] && 
        							                   !StringMatchQ[ToString@val, "Average", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Center Weighted Average", IgnoreCase -> True] && 
        							                   !StringMatchQ[ToString@val, "Spot", IgnoreCase -> True] && !StringMatchQ[ToString@val, "MultiSpot", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Multi-segment", IgnoreCase -> True] && 
        							                   !StringMatchQ[ToString@val, "Pattern", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Partial", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Segment", IgnoreCase -> True] && 
        							                   !StringMatchQ[ToString@val, "Reserved", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Other", IgnoreCase -> True]], miss, vRes],
   
   									               SameQ[key, "ExposureProgram"], If[If[NumberQ@val, val < 0  ||  val > 9 , !StringMatchQ[ToString@val, "Not defined", IgnoreCase -> True] && 
   									                   !StringMatchQ[ToString@val, "Auto", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Manual", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Normal program", IgnoreCase -> True] && 
   									                   !StringMatchQ[ToString@val, "Aperture priority", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Shutter priority", IgnoreCase -> True] && 
   									                   !StringMatchQ[ToString@val, "Creative program", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Action program", IgnoreCase -> True] && 
   									                   !StringMatchQ[ToString@val, "Portrait mode", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Landscape mode", IgnoreCase -> True] && 
   									                   !StringMatchQ[ToString@val, "Bulb", IgnoreCase -> True]], miss, vRes],
   
   									               SameQ[key, "CalibrationIlluminant2"] || SameQ[key, "CalibrationIlluminant1"], If[val === 0, miss, vRes],
   									  
   									               SameQ[ToString@key, "FlashInfo"] ||  SameQ[ToString@key, "Orientation"], If[val === {}, miss, vRes],
   									  
   									               SameQ[key, "CFALayout"], If[val < 1 || val > 5, miss, vRes], 
   									  
   									               SameQ[key, "LightSource"], If[If[NumberQ@val, (val < 0  || (val > 4  && (9! < val && val ! < 15) && (17! < val! < 24)) && val =!= 255), 
   									                   !StringMatchQ[ToString@val, "Unknown", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Daylight", IgnoreCase -> True] && 
   									                   !StringMatchQ[ToString@val, "Fluorescent", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Tungsten (incandescent light)", IgnoreCase -> True] && 
   									                   !StringMatchQ[ToString@val, "Flash", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Fine weather", IgnoreCase -> True] && 
   									                   !StringMatchQ[ToString@val, "Cloudy weather", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Shade", IgnoreCase -> True] && 
   									                   !StringMatchQ[ToString@val, "Daylight fluorescent (D 5700 - 7100K", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Day white fluorescent (N 4600 - 5400K)", 
        							                   IgnoreCase -> True] && !StringMatchQ[ToString@val, "Cool white fluorescent (W 3900 - 4500K)", IgnoreCase -> True] && !StringMatchQ[ToString@val, 
        							                   "White fluorescent (WW 3200 - 3700K", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Standard light A", IgnoreCase -> True] && 
        							                   !StringMatchQ[ToString@val, "Standard light B", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Standard light C", IgnoreCase -> True] && 
        							                   !StringMatchQ[ToString@val, "D55", IgnoreCase -> True] && !StringMatchQ[ToString@val, "D65", IgnoreCase -> True] && !StringMatchQ[ToString@val, "D75", IgnoreCase -> True] && 
        							                   !StringMatchQ[ToString@val, "D50", IgnoreCase -> True] && !StringMatchQ[ToString@val, "ISO studio tungsten", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Other light source", 
        							                   IgnoreCase -> True] && !StringMatchQ[ToString@ToString@val, "Reserved", IgnoreCase -> True]], miss, vRes],
   
               									  SameQ[key, "YCbCrPositioning"], If[If[NumberQ@val, val < 1 || val > 2, !StringMatchQ[ToString@val, "Centered", IgnoreCase -> True] && 
   									                   !StringMatchQ[ToString@val, "Cosited", IgnoreCase -> True] && 
   									                   !StringMatchQ[ToString@val, "Co-sited", IgnoreCase -> True] ], miss, vRes],
   
   									              SameQ[key, "YCbCrSubSampling"], If[val[[1]] < 0 || val[[2]] < 0, miss, vRes], SameQ[key, "OPIProxy"], If[If[NumberQ@val, val < 0 || val > 1, 
   									                   !StringMatchQ[ToString@val, "A higher-resolution version of this image does not exist", IgnoreCase -> True] && !StringMatchQ[ToString@val, "A higher-resolution version of this image exists, and the
									                   name of that image is found in the ImageID tag", IgnoreCase -> True]], miss, vRes],
   
  		 							              SameQ[key, "Indexed"], If[If[NumberQ@val, val < 0 || val > 1, !StringMatchQ[ToString@val, "Not indexed", IgnoreCase -> True] && 
  		 							                   !StringMatchQ[ToString@val, "Indexed", IgnoreCase -> True] ], miss, vRes],
   
   									              SameQ[key, "TileLength"] || SameQ[key, "TileWidth"] || SameQ[key, "NumberOfInks"] || SameQ[key, "XClipPathUnits"] || 
   	 								              SameQ[key, "YClipPathUnits"] || SameQ[key, "CellWidth"] || SameQ[key, "CellLength"] || SameQ[key, "SamplesPerPixel"], If[val < 0 , miss, vRes],
   
   									              SameQ[key, "InkSet"], If[val < 1 || val > 2, miss, vRes], 
   									  
   									              SameQ[key, "SubfileType"], If[If[NumberQ@val, val < 1 || val > 3, !StringMatchQ[ToString@val, "full-resolution image data", 
        							                   IgnoreCase -> True] && !StringMatchQ[ToString@val, "reduced-resolution image data", IgnoreCase -> True] && !StringMatchQ[ToString@val,  "single page of a multi-page image", IgnoreCase -> True]], 
   									                   miss, vRes],
   
   									              SameQ[key, "Compression"], If[If[NumberQ@val, val < 1 || (val > 10 && val =!= 32773), !StringMatchQ[ToString@val, "No compression", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Uncompressed", IgnoreCase -> True] &&
   									                   !StringMatchQ[ToString@val, "CCITT modified Huffman RLE", IgnoreCase -> True] && !StringMatchQ[ToString@val, "PackBits compression, aka Macintosh RLE", 
        							                   IgnoreCase -> True] && !StringMatchQ[ToString@val, "CCITT modified Huffman RLE", IgnoreCase -> True] && !StringMatchQ[ToString@val, "CCITT Group 3 fax encoding", 
        							                   IgnoreCase -> True] && !StringMatchQ[ToString@val, "CCITT Group 4 fax encoding", IgnoreCase -> True] && !StringMatchQ[ToString@val, "LZW", IgnoreCase -> True] && 
        							                   !StringMatchQ[ToString@val, "JPEG", IgnoreCase -> True]&&!StringMatchQ[ToString@val, "JPEG (new-style)", IgnoreCase -> True] && !StringMatchQ[ToString@val, "JPEG (old-style)", IgnoreCase -> True] && !StringMatchQ[ToString@val, "Deflate", IgnoreCase -> True] && 
        							                   !StringMatchQ[ToString@val, "Defined by TIFF-F and TIFF-FX standard (RFC 2301) as ITU-T Rec. T.82 coding, using ITU-T Rec. T.85", IgnoreCase -> True] && 
        							                   !StringMatchQ[ToString@val, "Defined by TIFF-F and TIFF-FX standard (RFC 2301) as ITU-T Rec. T.82 coding, using ITU-T Rec. T.43", IgnoreCase -> True]], miss, 
    								                   vRes],
   
   		            							  SameQ[key, "BrightnessValue"], If[val < -99.99 || val > 99.99, miss, vRes],
                  
   			            						  SameQ[key, "BlackLevel"], If[val < 0 || val >= 1, miss, vRes], SameQ[key, "BayerGreenSplit"], If[val < 0 || val >= 5000, miss, vRes],
               
   				            				      SameQ[key, "JPEGProc"], If[val =!= 0 && val != 14, miss, vRes],
   
   							            		  SameQ[key, "T6ptions"], If[val =!= 0 && val != 2, miss, vRes],
   
   									              SameQ[key, "T4Options"], If[val < 0 || val > 4, miss, vRes],
                           
   						            			  SameQ[key, "WhitePoint"], If[N@val[[1]] <= 0 || N@val[[1]] > 1 || N@val[[2]] <= 0 || N@val[[2]] > 1, miss, vRes],
   
   									              SameQ[key, "LensSpecification"], If[(val[[1]] < 0 && val[[1]] != Missing["Indeterminate"] ) || (val[[2]] < 0 && val[[2]] != Missing["Indeterminate"] ) || (val[[3]] < 0 && 
       								                  val[[3]] != Missing["Indeterminate"] ) || (val[[4]] < 0 && 
      								                  val[[4]] != Missing["Indeterminate"] ), miss, vRes],
   
  		            							  SameQ[key, "FillOrder"], If[val =!= 1 && val =!= 2, miss, vRes],
   
   					            				  SameQ[key, "WhiteLevel"], If[val =!= 0 && val =!= 1 && val =!= 2, miss, vRes],
   									  
   								            	  SameQ[key, "GPSVersionID"], 
   									                  Module[{tmp},
   									               	      Which[
   									               	          StringLength[StringReplace[val, "." | " " -> ""]] === 4, tmp = N[ToExpression[StringReplace[val, "." -> ""]]/1000, 3],
   									               	          StringLength[StringReplace[val, "." | " " -> ""]]   > 4, tmp = miss,
   									               	          True, tmp = val
   									               	      ];
   									               	      If[Quiet[MatchQ[ToExpression[tmp], _Real]], Rule[key, ToString@tmp], miss] 
   									               	  ],
   
   									       	      (MemberQ[ExifPositiveValuesOnly, key] && !MemberQ[QuantityTags, val] && !SameQ[key, "FocalLength"] && !SameQ[key, "FocalLengthIn35mmFilm"] && (NumberQ[N@val] || Quiet[NumberQ[ToExpression@val]])), If[N@val < 0 && ! ListQ@val, miss, vRes],	
   									     
   									       	      SameQ[key, "CustomRendered"], If[If[NumberQ@val, val < 0 || val > 1,  !StringMatchQ[ToString@val, "Normal process", IgnoreCase -> True] && 
   									        	      !StringMatchQ[ToString@val, "Custom process", IgnoreCase -> True]], miss, vRes],
   									  	
   									              True, If[MatchQ[val, "(-1)"], miss, vRes]]]
   									              
(**************************)
(**************************)
(**************************)
(**********IMPORT**********)
(***********EXIF***********)
(**************************)
(**************************)

ParseFlashInfo[state_] := Module[{cs = state, flash, flashInfo, fu, ffs, fm, ffp, ryc},
	                      flashInfo = cs["FlashInfo"];
	                      If[!IntegerQ[flashInfo], Return[cs]];
	                      Switch[ToExpression@flashInfo,
	                      	0 , fu = False;              ffs = "No strobe return detection function"; fm = Missing["Unknown"];             ffp = True;               ryc = False,
	                      	1 , fu = True;               ffs = "No strobe return detection function"; fm = Missing["Unknown"];             ffp = True;               ryc = False,
	                      	5 , fu = True;               ffs = Missing["Unknown"];                    fm = Missing["Unknown"];             ffp = True;               ryc = False,
	                      	7 , fu = True;               ffs = "Strobe return light detected";        fm = Missing["Unknown"];             ffp = True;               ryc = False,
	                      	8 , fu = False;              ffs = "No strobe return detection function"; fm = "Compulsory flash suppression"; ffp = True;               ryc = False,
	                      	9 , fu = True;               ffs = "No strobe return detection function"; fm = "Compulsory flash suppression"; ffp = True;               ryc = False,
	                      	13, fu = True;               ffs = Missing["Unknown"];                    fm = "Compulsory flash suppression"; ffp = True;               ryc = False,
	                      	15, fu = True;               ffs = "Strobe return light detected";        fm = "Compulsory flash suppression"; ffp = True;               ryc = False,
	                      	16, fu = False;              ffs = "No strobe return detection function"; fm = "Compulsory flash firing";      ffp = True;               ryc = False,
	                      	20, fu = False;              ffs = Missing["Unknown"];                    fm = "Compulsory flash firing";      ffp = True;               ryc = False,
	                      	24, fu = False;              ffs = "No strobe return detection function"; fm = Automatic;                      ffp = True;               ryc = False,
	                      	25, fu = True;               ffs = "No strobe return detection function"; fm = Automatic;                      ffp = True;               ryc = False,
	                      	29, fu = True;               ffs = Missing["Unknown"];                    fm = Automatic;                      ffp = True;               ryc = False,
	                      	31, fu = True;               ffs = "Strobe return light detected";        fm = Automatic;                      ffp = True;               ryc = False,
	                      	32, fu = False;              ffs = "No strobe return detection function"; fm = Missing["Unknown"];             ffp = False;              ryc = False,
	                      	48, fu = False;              ffs = "No strobe return detection function"; fm = "Compulsory flash firing";      ffp = False;              ryc = False,
	                      	65, fu = True;               ffs = "No strobe return detection function"; fm = Missing["Unknown"];             ffp = True;               ryc = True,
	                      	69, fu = True;               ffs = Missing["Unknown"];                    fm = Missing["Unknown"];             ffp = True;               ryc = True,
	                      	71, fu = True;               ffs = "Strobe return light detected";        fm = Missing["Unknown"];             ffp = True;               ryc = True,
	                      	73, fu = True;               ffs = "No strobe return detection function"; fm = "Compulsory flash suppression"; ffp = True;               ryc = True,
	                      	77, fu = True;               ffs = Missing["Unknown"];                    fm = "Compulsory flash suppression"; ffp = True;               ryc = True,
	                      	79, fu = True;               ffs = "Strobe return light detected";        fm = "Compulsory flash suppression"; ffp = True;               ryc = True,
	                      	80, fu = True;               ffs = "No strobe return detection function"; fm = "Compulsory flash firing";      ffp = True;               ryc = True,
	                      	88, fu = True;               ffs = "No strobe return detection function"; fm = Automatic;                      ffp = True;               ryc = True,
	                      	89, fu = True;               ffs = "No strobe return detection function"; fm = Automatic;                      ffp = True;               ryc = True,
	                      	93, fu = True;               ffs = Missing["Unknown"];                    fm = Automatic;                      ffp = True;               ryc = True,
	                      	95, fu = True;               ffs = "Strobe return light detected";        fm = Automatic;                      ffp = True;               ryc = True,
	                      	_ , fu = Missing["Unknown"]; ffs = Missing["Unknown"];                    fm = Missing["Unknown"];             ffp = Missing["Unknown"]; ryc = Missing["Unknown"];
	                      ];
	                      flash = <|"FlashUsed" -> fu, "FlashFiringStatus" -> ffs, "FlashMode" -> fm, "FlashFunctionPresent" -> ffp, "RedEyeCorrection" -> ryc|>;
	                      cs = AssociateTo[cs, "FlashInfo"-> flash];
	                      cs
   ]

ParseOrientation[state_] := Module[{cs = state, orientation, orientationInfo, cto, mir},
						        orientationInfo = cs["Orientation"];
	                            If[!IntegerQ[orientationInfo], Return[cs]];
	                            Switch[ToExpression@orientationInfo,
	                      	        1, cto = Top;                mir = False,
	                      	        2, cto = Top;                mir = True,
	                      	        3, cto = Bottom;             mir = False,
	                      	        4, cto = Bottom;             mir = True,
	                      	        5, cto = Left;               mir = True,
	                      	        6, cto = Right;              mir = False,
	                      	        7, cto = Right;              mir = True,
	                      	        8, cto = Left;               mir = False,
	                      	        _, cto = Missing["Unknown"]; mir = Missing["Unknown"];
	                            ];
	                            orientation = <|"CameraTopOrientation" -> cto, "Mirrored" -> mir|>;
	                            cs = AssociateTo[cs, "Orientation"-> orientation];
	                            cs
                             ]
    
ValidateExifAssociation[exif_] := 
                                 Module[{tmp, badList={}}, 
                                      If[StringLength[exif] > 5 && !SameQ[ToString@exif, "LibraryFunctionError[LIBRARY_USER_ERROR,-2]"], 
 	                                      tmp = DeleteMissing[ToExpression[Quiet@StringReplace[exif, WordCharacter .. ~~ " -> ," -> ""]]]; 
 	                                      badList = DeleteCases[Flatten@(If[StringTrim[ToString@tmp[[#]]] === "xxx", Append[badList, #]] & /@ Keys[tmp]), Null];
 	                                      KeyDropFrom[tmp, # & /@ badList]
 	                                  ]
 	                              ]

ParseTagsExifRaw[state_]:= 
                           Module[{cs = state, badList = {}}, 
						       cs = AssociateTo[cs, # -> If[StringQ@cs[#] && cs[#] =!= Missing["NotAvailable"], StringTrim@cs[#], cs[#]] & /@ Join[Intersection[GPSTags, Keys[cs]], Intersection[StringTags, Keys[cs]], Intersection[QuantityTags, Keys[cs]], Intersection[IntegerTags, Keys[cs]], Intersection[RationalTags, Keys[cs]], Intersection[RealTags, Keys[cs]]]];
 							   cs = AssociateTo[cs, # -> ToString[cs[#]] & /@ DeleteCases[Intersection[StringTags, Keys[cs]], "Orientation"]];
 							   cs = AssociateTo[cs, # -> If[cs[#] =!= Missing["NotAvailable"] && NumberQ[ToExpression[cs[#]]], ToExpression[cs[#]], cs[#]] & /@ DeleteDuplicates[Join[Intersection[IntegerTags, Keys[state]], Intersection[RealTags, Keys[state]]]]];
  							   If[ToString[cs[#]] == "", badList = Append[badList, #]] & /@ Keys[cs];
  							   cs = KeyDrop[cs, # &/@ Join[badList, {"GPSTag", "ExifTag", "XMLPacket", "IPTCNAA", "InterColorProfile"}]];
 							   cs = Append[cs, # -> Missing["NotAvailable"] & /@  DeleteCases[$AllExif, Alternatives @@ Sequence @@@ Keys[cs]]];
 							   cs
                            ]

ParseStringTagsExif[state_]:=
                             Module[{cs = state, badList = {}},
	                             cs = AssociateTo[cs, # -> If[StringQ[cs[#]], StringTrim@cs[#], cs[#]] & /@ Join[Intersection[GPSTags, Keys[cs]], Intersection[StringTags, Keys[cs]], Intersection[QuantityTags, Keys[cs]], Intersection[IntegerTags, Keys[cs]], Intersection[RationalTags, Keys[cs]], Intersection[RealTags, Keys[cs]]]];
  							     cs = AssociateTo[cs, # -> ToString[cs[#]] & /@ DeleteCases[Intersection[StringTags, Keys[cs]], "Orientation"]];
  							     If[ToString[cs[#]] == "", badList = Append[badList, #]] & /@ Keys[cs];
                                 cs = KeyDrop[cs, # &/@ Join[badList, {"GPSTag", "ExifTag", "XMLPacket", "IPTCNAA", "InterColorProfile"}]];
  								 cs
                             ]
 
ParseDateTimeTagsExif[state_]:= 
                             Module[{cs = state, tz},
	                             tz = Quiet[cs["TimeZoneOffset"]];
	                             If[!IntegerQ[tz], tz = None];   
  								 cs = AssociateTo[cs, # -> If[! MatchQ[ cs[#], Missing["NotAvailable"]], TimeObject[If[ListQ[cs[#]], cs[#], IntegerPart[ToExpression[StringSplit[cs[#], ":"]]]], TimeZone -> tz], cs[#]] & /@ Intersection[TimeTags, Keys[cs]]];
 								 cs = AssociateTo[cs, # -> If[! MatchQ[ cs[#], Missing["NotAvailable"]], With[{dt = cs[#]},If[SameQ[#, "GPSDateStamp"], tz = 0]; If[StringLength@dt <=10, DateObject[Take[DateList[{dt,{"Year", ":", "Month", ":", "Day"}}], 3]], DateObject[DateList[{dt,{"Year", ":", "Month", ":", "Day" , " ", "Hour", ":", "Minute", ":", "Second"}}], TimeZone -> tz]]], cs[#]] & /@ Intersection[DateTags, Keys[cs]]];
  								 cs
                             ]

ParseMultiValueTagsExif[state_]:= 
                             Module[{cs = state},      
  								 cs = AssociateTo[cs, # -> If[! MatchQ[ cs[#], Missing["NotAvailable"]], If[StringContainsQ[ToString[cs[#]], "," | " "], ToExpression[StringSplit[ToString[cs[#]], " "]], ToExpression[cs[#]]], cs[#]] & /@ Intersection[MultiValues, Keys[cs]]];
  								 cs
                             ]
  
ParseIntAndRealTagsExif[state_]:=
                             Module[{cs = state},      
  							     cs = AssociateTo[cs, # -> If[NumberQ[ToExpression[cs[#]]], ToExpression[cs[#]], cs[#]] & /@ DeleteDuplicates[Join[Intersection[IntegerTags, Keys[state]], Intersection[RealTags, Keys[state]]]]];
  								 cs
                             ]

ParseIndividualTagsExif[state_]:= 
                             Module[{cs = state, GPSAR = state["GPSAltitudeRef"], GPSS = state["GPSSpeed"], GPSF = state["GPSDOP"], BV = state["BrightnessValue"], SSV = state["ShutterSpeedValue"], CC=state["ComponentsConfiguration"], CS = state["ColorSpace"], SA = state["SubjectArea"], RBW = state["ReferenceBlackWhite"], LS = state["LensSpecification"], AV = state["ApertureValue"], MAV = state["MaxApertureValue"], concatList = {}},            
                                 If[GPSAR =!= Missing["KeyAbsent", "GPSAltitudeRef"]  , concatList = Append[concatList, "GPSAltitudeRef" -> If[SameQ[ToString[GPSAR], "Below sea level"], "BelowSeaLevel", "AboveSeaLevel"]]];
                                 If[CS =!= Missing["KeyAbsent", "ColorSpace"]         , concatList = Append[concatList, "ColorSpace" -> If[SameQ[ToString[CS], "sRGB"], "RGBColor", CS]]];
  						         If[SA =!= Missing["KeyAbsent", "SubjectArea"]        , concatList = Append[concatList, "SubjectArea" -> Switch[Count[SA, _Integer],      
       																														2, Point[SA],                                      
      	 																													3, Circle[{SA[[1]], SA[[2]]}, SA[[3]]],                                                            
       																														4, Rectangle[{SA[[1]] - SA[[3]]/2, SA[[2]] - SA[[4]]/2}, {SA[[1]] + SA[[3]]/2, SA[[2]] + SA[[4]]/2}],
 																															_, SA]]];
  						         If[RBW =!= Missing["KeyAbsent", "ReferenceBlackWhite"] && RBW =!=Null, concatList = Append[concatList, "ReferenceBlackWhite" -> {{RBW[[1]], RBW[[3]], RBW[[5]]}, {RBW[[2]], RBW[[4]], RBW[[6]]}}]];
  						         If[LS =!= Missing["KeyAbsent", "LensSpecification"]   , concatList = Append[concatList, "LensSpecification" -> If[StringContainsQ[ToString[LS], "Indeterminate"], ToExpression@StringReplace[ToString[LS], "Indeterminate" -> "0/0"(*"Missing[\"Indeterminate\"]"*)], LS]]];
  						         If[AV =!= Missing["KeyAbsent", "ApertureValue"]       , concatList = Append[concatList, "ApertureValue" -> N[AV]]];
  					             If[MAV =!= Missing["KeyAbsent", "MaxApertureValue"]   , concatList = Append[concatList, "MaxApertureValue" -> N[MAV]]];
  					             If[SSV =!= Missing["KeyAbsent", "ShutterSpeedValue"]  , concatList = Append[concatList, "ShutterSpeedValue" -> N[SSV]]];
  					             If[BV =!= Missing["KeyAbsent",  "BrightnessValue"]     , concatList = Append[concatList, "BrightnessValue" -> N[BV]]];
  					             If[GPSF =!= Missing["KeyAbsent", "GPSDOP"]            , concatList = Append[concatList, "GPSDOP" -> N[GPSF]]];
  					             If[GPSS =!= Missing["KeyAbsent", "GPSSpeed"]          , concatList = Append[concatList, "GPSSpeed" -> N[GPSS]]];
   						         If[CC =!= Missing["KeyAbsent", "ComponentsConfiguration"]  , concatList = Append[concatList, "ComponentsConfiguration" -> ToString[CC]]];
  						         cs = AssociateTo[cs, concatList];
  						         cs
                             ]

ParseRationalTagsExif[state_]:= 
                           Module[{cs = state},
  						       cs = AssociateTo[cs, # -> If[MemberQ[QuantityTags, #], StringReplace[ToString[cs[#]], d1 : DigitCharacter .. ~~ "/" ~~ d2 : DigitCharacter .. :> "\!\(\*FractionBox[\(" ~~ d1 ~~ "\), \(" ~~ d2 ~~ "\)]\)"], 
        																			      ToExpression@StringReplace[ToString[cs[#]], d1 : DigitCharacter .. ~~ "/" ~~ d2 : DigitCharacter .. :> "\!\(\*FractionBox[\(" ~~ d1 ~~ "\), \(" ~~ d2 ~~ "\)]\)"]
        																				  ] & /@ DeleteCases[DeleteCases[DeleteCases[Intersection[RationalTags,  Keys[state]], "FNumber"], "ExposureBiasValue"], "BaselineExposure"]];
  						       cs
                           ]

ParseQuantityTagsExif[state_]:=  
                    Module[{cs = state, FLF = state["FocalLengthIn35mmFilm"], SST = state["SubSecTime"], SSTO = state["SubSecTimeOriginal"], SSTD = state["SubSecTimeDigitized"], ET = state["ExposureTime"], FL = state["FocalLength"], SD = state["SubjectDistance"], GPSS = state["GPSSpeed"], GPSA = state["GPSAltitude"], 
   						    GPST = state["GPSTrack"], GPSID = state["GPSImgDirection"], GPSBV = state["ExposureBiasValue"], GPSBE = state["BaselineExposure"], GPSLo = state["GPSLongitude"], GPSLa = state["GPSLatitude"], concatList = {}},                                                  	
 						If[FLF =!= Missing["KeyAbsent", "FocalLengthIn35mmFilm"], concatList = Append[concatList, "FocalLengthIn35mmFilm" -> If[Quiet@StringQ[FLF] && Quiet[StringLength[FLF] >= 3], With[{tmp = ToExpression@StringTrim@StringTake[FLF, {1, -3}]}, If[Quiet@Internal`RealValuedNumericQ[tmp] === True && tmp > 0, Quantity[IntegerPart[tmp], "Millimeters"], FLF]],If[Quiet@Internal`NonNegativeIntegerQ[FLF] === True,Quantity[FLF, "Millimeters"], FLF]]]];
 						If[SST =!= Missing["KeyAbsent", "SubSecTime"]          , concatList = Append[concatList, "SubSecTime" -> Quantity[SST, "Milliseconds"]]];
  						If[SSTO =!= Missing["KeyAbsent", "SubSecTimeOriginal"] , concatList = Append[concatList, "SubSecTimeOriginal" -> Quantity[SSTO, "Milliseconds"]]];
  						If[SSTD =!= Missing["KeyAbsent", "SubSecTimeDigitized"], concatList = Append[concatList, "SubSecTimeDigitized" -> Quantity[SSTD, "Milliseconds"]]];
  						If[ET =!= Missing["KeyAbsent", "ExposureTime"]         , concatList = Append[concatList, "ExposureTime" -> Quantity[ET, "Seconds"]]];
  						If[FL =!= Missing["KeyAbsent", "FocalLength"]          , concatList = Append[concatList, "FocalLength" -> Quantity[FL, "Millimeters"]]];
  						If[SD =!= Missing["KeyAbsent", "SubjectDistance"]      , concatList = Append[concatList, "SubjectDistance" -> Quantity[SD, "Meters"]]];
  						If[GPSS =!= Missing["KeyAbsent", "GPSSpeed"]           , concatList = Append[concatList, "GPSSpeed" -> Quantity[GPSS, "Kilometers"/"Hours"]]];
                        If[GPSA =!= Missing["KeyAbsent", "GPSAltitude"]        , concatList = Append[concatList, "GPSAltitude" -> Quantity[If[NumberQ[GPSA], GPSA, N[ToExpression[StringDelete[GPSA, c_ /; !DigitQ[c] && ! StringMatchQ[c, "." | "/"]]]]], "Meters"]]];
                        If[GPST =!= Missing["KeyAbsent", "GPSTrack"]           , concatList = Append[concatList, "GPSTrack" -> Quantity[GPST, "AngularDegrees"]]];
                        If[GPSID =!= Missing["KeyAbsent", "GPSImgDirection"]   , concatList = Append[concatList, "GPSImgDirection" -> Quantity[GPSID, "AngularDegrees"]]]; 
                        If[GPSBV =!= Missing["KeyAbsent", "ExposureBiasValue"] , concatList = Append[concatList, "ExposureBiasValue" -> Quantity[N[GPSBV], IndependentUnit["exposure values"]]]];
                        If[GPSBE =!= Missing["KeyAbsent", "BaselineExposure"]  , concatList = Append[concatList, "BaselineExposure" -> Quantity[N[GPSBE], IndependentUnit["exposure values"]]]];
                        If[GPSLo =!= Missing["KeyAbsent", "GPSLongitude"]      , concatList = Append[concatList, "GPSLongitude" -> 
                        	                                                                          Block[{tmpGPS = GPSLo}, 
                                             									                        If[ListQ[tmpGPS], 
                                             									                          tmpGPS = StringReplace[StringTake[ToString@N[tmpGPS], {2, -2}], "," -> ""]]; 
                                            									                            Module[{tmp = If[StringContainsQ[tmpGPS, "/"], 
                                            									                     	                    Select[N[ToExpression@StringSplit[tmpGPS, " "]], NumberQ[#] &], 
                                                									                                        ToExpression[StringCases[tmpGPS, NumberString]]]
                                                									                               },
                                                									                            
                                             									                                   Switch[Length@tmp, 
                                             									                                   	 1, First@Quantity[N[tmp], "AngularDegrees"], 
                                             									                              	     2, Quantity[N[tmp[[1]] + tmp[[2]]/60], "AngularDegrees"], 
                                             									                              	     3, Quantity[N[tmp[[1]] + tmp[[2]]/60 + tmp[[3]]/3600], "AngularDegrees"]]
                                             									                               ]
                                             									                          ]
                                             									                        ]
                                             									                     ];
  						If[GPSLa =!= Missing["KeyAbsent", "GPSLatitude"]       , concatList = Append[concatList, "GPSLatitude" -> 
                        	                                                                          Block[{tmpGPS = GPSLa}, 
                                             									                        If[ListQ[tmpGPS], 
                                             									                          tmpGPS = StringReplace[StringTake[ToString@N[tmpGPS], {2, -2}], "," -> ""]]; 
                                            									                            Module[{tmp = If[StringContainsQ[tmpGPS, "/"], 
                                            									                     	                    Select[N[ToExpression@StringSplit[tmpGPS, " "]], NumberQ[#] &], 
                                                									                                        ToExpression[StringCases[tmpGPS, NumberString]]]
                                                									                               },
                                                									                            
                                             									                                   Switch[Length@tmp, 
                                             									                                   	 1, First@Quantity[N[tmp], "AngularDegrees"], 
                                             									                              	     2, Quantity[N[tmp[[1]] + tmp[[2]]/60], "AngularDegrees"], 
                                             									                              	     3, Quantity[N[tmp[[1]] + tmp[[2]]/60 + tmp[[3]]/3600], "AngularDegrees"]]
                                             									                               ]
                                             									                          ]
                                             									                        ]
                                             									                     ];
  
 			 			cs = AssociateTo[cs, concatList] ;
  						cs
  ]


ParseValuesInGroupsExif[valEx_] := 
                             Module[{curState = valEx},	
                                 curState = ParseStringTagsExif[curState];
                                 curState = ParseDateTimeTagsExif[curState];
                                 curState = ParseMultiValueTagsExif[curState];
                                 curState = ParseIntAndRealTagsExif[curState];
                                 curState = ParseFlashInfo[curState];
                                 curState = ParseOrientation[curState];
                                 curState = ParseIndividualTagsExif[curState];
                                 curState = ParseQuantityTagsExif[curState];
                                 curState
                             ]
  
ParseValuesInGroupsExifOLD[valEx_] := 
                                Module[{curState = valEx},
                                    curState = ModifyMakerNoteRawExifOLD[curState];
                                    curState = RemoveNotExifTags[curState];
                                    curState = ParseDateTimeTagsExif[curState];
                                    curState = ParseMultiValueTagsExif[curState];
                                    curState = ParseIntAndRealTagsExif[curState];
                                    curState = ParseFlashInfo[curState];
                                    curState = ParseOrientation[curState];
                                    curState = ParseIndividualTagsExif[curState];
                                    curState = ParseQuantityTagsExif[curState];
                                    curState = ParseVersionTagsExifOLD[curState];  
                                    curState
                                 ]

GetExifAll[] :=
	With[{tmp = validatePossibleAssociation[$ReadExifAllRaw[True]]},

		If[tmp === "<||>",
			<||>,
			RemoveNotExifTags[Delete[ParseValuesInGroupsExif[ValidateExifAssociation[tmp]], "ImageResources"]]
		]
	]
RemoveNotExifTags[asc_] := Block[{newAsc=asc}, KeyDropFrom[newAsc,  # & /@ Complement[Keys[newAsc], Join[$AllExif, {"MakerNote"}]]]]

ModifyMakerNoteRawExif[asc_] := Module[{tmp, res = None, mkNote},
	                                mkNote = asc["MakerNote"];
	                                If[mkNote === Missing["KeyAbsent", "MakerNote"], Return[asc]];
	                                tmp = asc;
	                                mkNote = Quiet@ToExpression[StringSplit[mkNote]];
	                                mkNote = If[ListQ@mkNote, ByteArray[mkNote], ByteArray[{}]];
	                                res = Quiet@AssociateTo[tmp, "MakerNote"-> mkNote];
	                                If[AssociationQ[res], Return[res], Return[<||>]];
                                 ]
  
ModifyMakerNoteRawExifOLD[asc_] := Module[{tmp, res = None, mkNote}, 
	                                   mkNote = asc["MakerNote"];
 	                                   If[mkNote === Missing["KeyAbsent", "MakerNote"], Return[asc]];
                                       tmp = asc;
 	                                   mkNote = If[ListQ@mkNote, ByteArray[mkNote], ByteArray[{}]];
 	                                   res = Quiet@AssociateTo[tmp, "MakerNote" -> mkNote];
 	                                   If[AssociationQ[res], Return[res], Return[<||>]];
                                    ]
  
ParseVersionTagsExifOLD[asc_] :=
                          Module[{tmp, res = None, ex, fl, ip, gps, fnum, gpsLat, gpsLon}, 
	                            ex = asc["ExifVersion"];
	                            fl = asc["FlashpixVersion"];
	                            ip = asc["InteroperabilityVersion"];
	                            gps = asc["GPSVersionID"];
	                            gpsLat = asc["GPSLatitude"];
	                            gpsLon = asc["GPSLongitude"];
	                            fnum = asc["FNumber"];
	                            tmp = asc;
 	                            If[ex =!= Missing["KeyAbsent", "ExifVersion"], tmp = AssociateTo[tmp, "ExifVersion"->parstVersions[ex]]];
 	                            If[fl =!= Missing["KeyAbsent", "FlashpixVersion"], tmp = AssociateTo[tmp, "FlashpixVersion"->parstVersions[fl]]];
 	                            If[ip =!= Missing["KeyAbsent", "InteroperabilityVersion"], tmp = AssociateTo[tmp, "InteroperabilityVersion"->parstVersions[ip]]];
 	                            If[gps =!= Missing["KeyAbsent", "GPSVersionID"], tmp = AssociateTo[tmp, "GPSVersionID"->StringReplace[StringTake[ToString[gps], {2, -2}], ","->""]]];
 	                            If[fnum =!= Missing["KeyAbsent", "FNumber"], tmp = AssociateTo[tmp, "FNumber"->StringJoin["f/", ToString@N@fnum]]];
 	                            res = tmp;
 	                            If[AssociationQ[res], Return[res], Return[<||>]];
                           ]

parstVersions[tmp_] := ToString@ToExpression@StringInsert[StringJoin[ToString /@ FromCharacterCode /@ tmp], ".", 3]

ReadExifIndividualTag[tag_]:= 
                         Module[{tmp, res = None},
						     tmp = Quiet[StringTrim[validatePossibleString[$ReadExifIndividualTag[tag]]]];
							 tmp = If[SameQ[tag, "Orientation"] || SameQ[tag, "FlashInfo"], ToExpression@tmp, tmp];
							 tmp = If[StringContainsQ[ToString[tmp], "LibraryFunctionError"] || tmp === Null || tmp === "", None, tmp];
							 If[tmp =!= None, res = First[ValidateExif[ParseValuesInGroupsExif[<|tag->tmp|>]]]];
							 res
                         ]

ReadExif[tag_, rule_ : False] := Block[{$Context = "XMPTools`TempContext`"},
	Switch[tag,
		"AllRaw"              ,
		Module[
			{
				resTmp = validatePossibleAssociation[$ReadExifAllRaw[False]],
				tmp
			},

			If[resTmp === "<||>",
				<||>,
				tmp = ModifyMakerNoteRawExif@ParseTagsExifRaw[Quiet@ParseMultiValueTagsExif[ToExpression[resTmp]]];
				If[AssociationQ[tmp], RemoveNotExifTags[tmp], <||>]
			]
		],
		"All", Module[{tmp = GetExifAll[]}, If[Quiet[AssociationQ[tmp]], tmp, <||>]],
		_    , ReadExifIndividualTag[tag]
	]
]

ReadExif[tag_] := ReadExif[tag, False]


(**************************)
(**************************)
(**************************)
(**********EXPORT**********)
(***********EXIF***********)
(**************************)
(**************************)
GetOrientationNumber[assc_] :=  If[assc =!= Missing["KeyAbsent", "Orientation"],
								  Which[
								   assc["CameraTopOrientation"] === Top                && assc["Mirrored"] === False, 1,
								   assc["CameraTopOrientation"] === Top                && assc["Mirrored"] === True,  2,
								   assc["CameraTopOrientation"] === Bottom             && assc["Mirrored"] === False, 3,
								   assc["CameraTopOrientation"] === Bottom             && assc["Mirrored"] === True,  4,
								   assc["CameraTopOrientation"] === Left               && assc["Mirrored"] === True,  5,
								   assc["CameraTopOrientation"] === Right              && assc["Mirrored"] === False, 6,
								   assc["CameraTopOrientation"] === Right              && assc["Mirrored"] === True,  7,
								   assc["CameraTopOrientation"] === Left               && assc["Mirrored"] === False, 8,
								   True                                                                             , 255
								   ]]

GetFlashNumber[assc_] := If[assc =!= Missing["KeyAbsent", "FlashInfo"],
  							Which[
							   assc["FlashUsed"] === False && assc["FlashFiringStatus"] === "No strobe return detection function"  && assc["FlashMode"] === Missing["Unknown"] && 
		    				   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === False, 0,
							   
							   assc["FlashUsed"] === True  && assc["FlashFiringStatus"] === "No strobe return detection function"  && assc["FlashMode"] === Missing["Unknown"] && 
							   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === False, 1,
							   							   
							   assc["FlashUsed"] === True  && assc["FlashFiringStatus"] === Missing["Unknown"]                     && assc["FlashMode"] === Missing["Unknown"] && 
							   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === False, 5,
							   
							   assc["FlashUsed"] === True  && assc["FlashFiringStatus"] === "Strobe return light detected"         && assc["FlashMode"] === Missing["Unknown"] && 
							   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === False, 7,
							   
							   assc["FlashUsed"] === False && assc["FlashFiringStatus"] === "No strobe return detection function"  && assc["FlashMode"] === "Compulsory flash suppression" && 
							   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === False, 8,
							   
							   assc["FlashUsed"] === True  && assc["FlashFiringStatus"] === "No strobe return detection function"  && assc["FlashMode"] === "Compulsory flash suppression" && 
							   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === False, 9,
							   
							   assc["FlashUsed"] === True  && assc["FlashFiringStatus"] === Missing["Unknown"]                     && assc["FlashMode"] === "Compulsory flash suppression" && 
							   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === False, 13,
							   
							   assc["FlashUsed"] === True && assc["FlashFiringStatus"] === "Strobe return light detected"          && assc["FlashMode"] === "Compulsory flash suppression" && 
							   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === False, 15,
							   
							   assc["FlashUsed"] === False && assc["FlashFiringStatus"] === "No strobe return detection function"  && assc["FlashMode"] === "Compulsory flash firing" && 
							   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === False, 16,
							   
							   assc["FlashUsed"] === False && assc["FlashFiringStatus"] === Missing["Unknown"]                     && assc["FlashMode"] === "Compulsory flash firing" && 
							   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === False, 20,
							   
							   assc["FlashUsed"] === False && assc["FlashFiringStatus"] === "No strobe return detection function"  && assc["FlashMode"] === Automatic && 
							   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === False, 24,
							   
							   assc["FlashUsed"] === True  && assc["FlashFiringStatus"] === "No strobe return detection function"  && assc["FlashMode"] === Automatic && 
							   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === False, 25,
							   
							   assc["FlashUsed"] === True  && assc["FlashFiringStatus"] === Missing["Unknown"]                     && assc["FlashMode"] === Automatic && 
							   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === False, 29,
							   
							   assc["FlashUsed"] === True  && assc["FlashFiringStatus"] === "Strobe return light detected"         && assc["FlashMode"] === Automatic && 
							   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === False, 31,
							   
							   assc["FlashUsed"] === False && assc["FlashFiringStatus"] === "No strobe return detection function"  && assc["FlashMode"] === Missing["Unknown"] && 
							   assc["FlashFunctionPresent"] === False && assc["RedEyeCorrection"] === False, 32,
							   
							   assc["FlashUsed"] === False && assc["FlashFiringStatus"] === "No strobe return detection function"  && assc["FlashMode"] === "Compulsory flash firing" && 
							   assc["FlashFunctionPresent"] === False && assc["RedEyeCorrection"] === False, 48,
							   
							   assc["FlashUsed"] === True  && assc["FlashFiringStatus"] === "No strobe return detection function"  && assc["FlashMode"] === Missing["Unknown"] && 
							   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === True, 65,
							   
							   assc["FlashUsed"] === True  && assc["FlashFiringStatus"] === Missing["Unknown"]                     && assc["FlashMode"] === Missing["Unknown"] && 
							   assc["FlashFunctionPresent"] === True && assc["Strobe return light not detected"] === True, 69,
							   
							   assc["FlashUsed"] === True  && assc["FlashFiringStatus"] === "Strobe return light detected"         && assc["FlashMode"] === Missing["Unknown"] && 
							   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === True, 71,
							   
							   assc["FlashUsed"] === True  && assc["FlashFiringStatus"] === "No strobe return detection function"  && assc["FlashMode"] === "Compulsory flash suppression" && 
							   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === True, 73,
							    					   
							   assc["FlashUsed"] === True  &&  assc["FlashFiringStatus"] === Missing["Unknown"]                    && assc["FlashMode"] === "Compulsory flash suppression" && 
							   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === True, 77,
							   
							   assc["FlashUsed"] === True  &&  assc["FlashFiringStatus"] === "Strobe return light detected"        && assc["FlashMode"] === "Compulsory flash suppression" && 
							   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === True, 79, 
							   
							   assc["FlashUsed"] === True  &&  assc["FlashFiringStatus"] === "No strobe return detection function" && assc["FlashMode"] === "Compulsory flash firing" && 
							   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === True, 80,
							   
							   assc["FlashUsed"] === True  &&  assc["FlashFiringStatus"] === "No strobe return detection function" && assc["FlashMode"] === Automatic && 
							   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === True, 88,
							   
							   assc["FlashUsed"] === True  &&  assc["FlashFiringStatus"] === "No strobe return detection function" && assc["FlashMode"] === Automatic && 
							   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === True, 89,
							   
							   assc["FlashUsed"] === True  &&  assc["FlashFiringStatus"] === Missing["Unknown"]                    && assc["FlashMode"] === Automatic && 
							   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === True, 93,
							   
							   assc["FlashUsed"] === True  &&  assc["FlashFiringStatus"] === "Strobe return light detected"        && assc["FlashMode"] === Automatic && 
							   assc["FlashFunctionPresent"] === True && assc["RedEyeCorrection"] === True, 95,
							   
							   True, 0
							   ]]

ExifProcessToRaw[Rule[key_, assoc_Association]] := Rule[key, AssociationMap[ExifProcessToRaw, assoc]]
ExifProcessToRaw[Rule[key_, val_]] := Which[   
									   SameQ["ReferenceBlackWhite", ToString@key]	, Rule[key, Module[{a = {val[[1, 1]], val[[2, 1]], val[[1, 2]], val[[2, 2]], val[[1, 3]], val[[2, 3]]}}, 
																							 					StringSplit[StringJoin[Riffle[ToString[#, InputForm] & /@ a, " "]], " " ] /. 
																							   				    s_String /; StringMatchQ[s, DigitCharacter ..] :> s <> "/1" // StringRiffle]],	
									   
									   SameQ[ToString[key], "GPSVersionID"]          , Rule[key, Module[{v = StringReplace[val, "." -> ""]},
																													 Switch[StringCount[v, _],
																														  1, v = StringInsert[v, "000", -1],
																														  2, v = StringInsert[v, "00", -1],
																														  3, v = StringInsert[v, "0", -1],
																														  4, v
																													  ];
																													 v = StringSplit[v, ""];
																													 v = StringReplace[StringTake[ToString@v, {2, -2}], "," -> ""];
																													 v
																											 ]
																									 ],
									   
									   StringContainsQ[ToString[key], "Version"]            , Rule[key, Module[{v = StringReplace[val, "." -> ""]},
																													 Switch[StringCount[v, _],
																														  1, v = StringInsert[v, "000", 1],
																														  2, v = StringInsert[v, "00", 1],
																														  3, v = StringInsert[v, "0", 1],
																														  4, v
																													  ];
																													 v = StringSplit[v, ""];
																													 v = First@ToCharacterCode[v[[#]]] & /@ Range[Count[v, _]];
																													 v = StringReplace[StringTake[ToString@v, {2, -2}], "," -> ""];
																													 v
																											 ]
																									 ],
									   
									   SameQ[ToString@key, "GPSSpeedRef"],                Rule[key, Which[StringContainsQ[val, "km"], "K", StringContainsQ[val, "mi"], "M", StringContainsQ[val, "knot"], "N", True, val]],
									  
									   SameQ[ToString@key, "GPSDestBearingRef"],          Rule[key, Which[StringContainsQ[val, "True"], "T", StringContainsQ[val, "Magnetic"], "M", True, val]],
									   
									   StringContainsQ[ToString[key], "GPSSpeed"] ||
									   StringContainsQ[ToString[key], "GPSTrack"] ||
									   StringContainsQ[ToString[key], "GPSImgDirection"] ||
									   StringContainsQ[ToString@key,  "GPS", IgnoreCase->True] && 
									   (StringContainsQ[ToString[key],"Latitude"] ||
									   StringContainsQ[ToString[key], "Longitude"] ||
									   StringContainsQ[ToString[key], "Altitude"]) &&
									   !StringContainsQ[ToString[key],"Ref"],                Rule[key, Normal@@val],
									   val === Missing["Disputed"],                          Rule[key, -1],
									   val === Missing["Unknown"] ,                          Rule[key,  0],
									   SameQ["SubjectArea", ToString@key],            Rule[key, Module[{v1 = Flatten[(List @@ val)]}, 
																										 Which[Count[v1, _] === 2, ToString[Round@N[v1[[1]]]], 
																										  Count[v1, _] === 3, 
																										  StringJoin[ToString[Round@N[v1[[1]]]], " ", 
																										   ToString[Round@N[v1[[2]]]], " ", ToString[Round@N[v1[[3]]]]], 
																										  Count[v1, _] === 4, 
																										  StringJoin[ToString[Round@N[(v1[[1]] + v1[[3]])/2]], " ", 
																										   ToString[Round@N[(v1[[4]] + v1[[2]])/2]], " ", 
																										   ToString[Round@N[(v1[[3]] + v1[[1]])/2]], " ", 
																										   ToString[Round@N[(v1[[4]] + v1[[2]])/2]]]]]],																								   
									  									   
									  StringContainsQ[ToString@key, "subsec", IgnoreCase->True], Rule[key, If[QuantityQ[val], QuantityMagnitude[val], val]],
									  
									  SameQ[ToString@key, "DateTime"] ||
									  SameQ[ToString@key, "DateTimeOriginal"] || 
									  SameQ[ToString@key, "DateTimeDigitized"] ,      	      Rule[key, If[DateObjectQ[val],
																												DateString[val,
																													{"Year", "-", "MonthShort", "-", "DayShort",
																												    " ",
																													"HourShort", ":", "MinuteShort", ":", "SecondShort"}
																													],
																												-1
																											]
																								],
									   
									   MemberQ[DateTags, ToString@key] ||
									   SameQ[ToString@key, "GPSDateStamp"] , 	       Rule[key, If[DateObjectQ[val], DateString[val,	{"Year", "-", "MonthShort", "-", "DayShort"}], -1]],
							     																   
									   SameQ[ToString@key, "GPSTimeStamp"] ,           Rule[key, If[TimeObjectQ[val], DateString[val,	{"HourShort", ":", "MinuteShort", ":", "SecondShort"}], -1]],
									   MemberQ[TimeTags, ToString@key],                Rule[key, If[TimeObjectQ[val], DateString[val,	{"HourShort", ":", "MinuteShort", ":", "SecondShort"}], -1]],
									   
                                       SameQ[ToString@key, "FocalLengthIn35mmFilm"] ||
									   (
									    !SameQ[ToString@key, "FNumber"] &&
									    !SameQ[ToString@key, "BaselineExposure"] &&
									    !SameQ[ToString@key, "ExposureBiasValue"] &&
									    MemberQ[RationalTags, ToString@key]),				 If[!NumberQ[val], Rule[key, If[QuantityQ[val],
 																								 				If[StringContainsQ[ToString[Normal @@ val], "/"|"-"|"--"|"---"], 
 																								 						StringJoin[StringCases[ToString[Normal @@ val], NumberString][[1]],"/", StringCases[ToString[Normal @@ val], NumberString][[2]]]
     																													,
   																														StringCases[ToString[Normal @@ val], NumberString][[1]]
   																												]
 																					 						    ,
  																												If[StringContainsQ[ToString[val], LetterCharacter],
   																														StringJoin[ToString[Round[ToExpression[StringCases[val, NumberString][[1]]]]], "/", 
   																															If[Count[StringCases[val, NumberString], _] === 2, 
 																																ToString[Round[ToExpression[StringCases[val, NumberString][[2]]]]], "1"]], val]
  																																]
 																								], Rule[key, val]],
									   
									   SameQ[key, "LensSpecification"],  			 Rule[key, StringSplit[StringReplace[StringReplace[ToString@InputForm[val], "Missing[Indeterminate]" -> "0/0"], 
									        														   "," | "{" | "}" -> " "]]/. s_String /; StringMatchQ[s, DigitCharacter ..] :> s <> "/1" // StringRiffle],
									   
									   SameQ[key, "FNumber"] || 
									   SameQ[key, "Lens"],    						 Rule[key, StringDelete[val, "f/"]],
									   
									   SameQ[key, "MakerNote"],                      Rule[key, StringReplace[StringTake[ToString@Normal[val], {2, -2}], "," -> ""]],
							   
									   SameQ[ToString@key, "BaselineExposure"] ||
									   SameQ[ToString@key, "ExposureBiasValue"] ,     Rule[key, (List @@ val)[[1]]],
									   True,                                                 Rule[key, Which[ListQ@val, StringTake[ToString[val], {2, StringLength[ToString[val]] - 1}],
									     															         QuantityQ@val, StringJoin[ToString@(List @@ val[[1]]), " ", (ToString@List @@ val[[2]])],
									     																	 True, Normal @@ val]]
						];

convertOldStyleExif[oldData_] :=  Module[{asc = With[{ky = Select[Keys[Association[oldData]], NumberQ[#] &]},KeyDrop[Association[oldData], # & /@ ky]]},
  
  							If[asc["ReferenceBlackWhite"] =!= Missing["KeyAbsent", "ReferenceBlackWhite"], 
   								With[{val = Quiet@Partition[asc["ReferenceBlackWhite"], 3]}, 
   								    AssociateTo[asc, "ReferenceBlackWhite" -> Quiet@Module[{a = {val[[1, 1]], val[[2, 1]], val[[1, 2]], val[[2, 2]], val[[1, 3]], val[[2, 3]]}}, 
       																		  StringSplit[StringJoin[Riffle[ToString[#, InputForm] & /@ a, " "]], " "] /. s_String /; StringMatchQ[s, DigitCharacter ..] :> s <> "/1" // StringRiffle]]]];
  
  							If[asc["BitsPerSample"] =!= Missing["KeyAbsent", "BitsPerSample"], 
   								With[{val = asc["BitsPerSample"]}, 
   									AssociateTo[asc, "BitsPerSample" -> Quiet@StringTake[StringDelete[ToString[val], ","], {2, -2}]]]];
  
  							If[asc["ExifVersion"] =!= Missing["KeyAbsent", "ExifVersion"], 
   								With[{val = asc["ExifVersion"]}, 
    								AssociateTo[asc, "ExifVersion" -> Quiet@StringTake[StringDelete[ToString[val], ","], {2, -2}]]]];
    								
							If[asc["CFAPattern"] =!= Missing["KeyAbsent", "CFAPattern"], 
   								With[{val = asc["CFAPattern"]}, 
    								AssociateTo[asc, "CFAPattern" -> Quiet@StringTake[StringDelete[ToString[val], ","], {2, -2}]]]];
  
  							If[asc["ComponentsConfiguration"] =!= Missing["KeyAbsent", "ComponentsConfiguration"], 
   								With[{val = asc["ComponentsConfiguration"]}, 
    								AssociateTo[asc, "ComponentsConfiguration" -> Quiet@StringTake[StringDelete[ToString[val], ","], {2, -2}]]]];
  
  							If[asc["FlashpixVersion"] =!= Missing["KeyAbsent", "FlashpixVersion"], 
   								With[{val = asc["FlashpixVersion"]}, 
    								AssociateTo[asc, "FlashpixVersion" -> Quiet@StringTake[StringDelete[ToString[val], ","], {2, -2}]]]];
  
  							If[asc["GPSVersionID"] =!= Missing["KeyAbsent", "GPSVersionID"], 
   								With[{val = asc["GPSVersionID"]}, 
    								AssociateTo[asc, "GPSVersionID" -> Quiet@StringTake[StringDelete[ToString[val], ","], {2, -2}]]]];
  
   							If[asc["SceneType"] =!= Missing["KeyAbsent", "SceneType"], 
   								With[{val = asc["SceneType"]}, 
    								AssociateTo[asc, "SceneType" -> Quiet@ToString@val]]];
  
   							If[asc["GPSLongitude"] =!= Missing["KeyAbsent", "GPSLongitude"], 
   								With[{val = ToString@N@asc["GPSLongitude"]}, 
    								AssociateTo[asc, "GPSLongitude" -> Quiet@Module[{tmp = If[StringContainsQ[val, "/"], Select[N[ToExpression@StringSplit[val, " "]], NumberQ[#] &], ToExpression[StringCases[val, NumberString]]]}, 
       																			Switch[Length@tmp,
       																					 1, First@N[tmp],
       																					 2, N[tmp[[1]] + tmp[[1]]/60], 
       																					 3, N[tmp[[1]] + tmp[[1]]/60 + tmp[[1]]/3600]]]]]];
  
  							If[asc["GPSLatitude"] =!= Missing["KeyAbsent", "GPSLatitude"], 
   								With[{val = ToString@N@asc["GPSLatitude"]}, 
    								AssociateTo[asc, "GPSLatitude" -> Quiet@Module[{tmp =  If[StringContainsQ[val, "/"], Select[N[ToExpression@StringSplit[val, " "]], NumberQ[#] &], ToExpression[StringCases[val, NumberString]]]}, 
       																			Switch[Length@tmp, 
   																						 1, First@N[tmp], 
   																						 2, N[tmp[[1]] + tmp[[1]]/60], 
   																						 3, N[tmp[[1]] + tmp[[1]]/60 + tmp[[1]]/3600]]]]]];
  
   							If[asc["MakerNote"] =!= Missing["KeyAbsent", "MakerNote"], 
   								With[{val = asc["MakerNote"]}, 
    								AssociateTo[asc, "MakerNote" -> Quiet@ToString@val]]];
  
   							If[asc["LensSpecification"] =!= Missing["KeyAbsent", "LensSpecification"], 
   								With[{val = asc["LensSpecification"]}, 
    								AssociateTo[asc, "LensSpecification" -> Quiet[(StringSplit[StringReplace[StringReplace[ToString@InputForm[val], "Missing[Indeterminate]" -> "0/0"], "," | "{" | "}" -> " "]] /. 
         																	s_String /; StringMatchQ[s, DigitCharacter ..] :> s <> "/1" // StringRiffle)]]]];
  
  							asc
  ]

PrepareExifMetaFromProcess[assc_] := AssociationMap[ExifProcessToRaw, DeleteCases[Association@KeyValueMap[#1 -> DeleteCases[#2, _?(StringMatchQ[ToString@#, Whitespace ..] &)] &, assc], _?(# == <||> &)]]
PrepareExifMetaForExport[assc_]   := Block[{$Context = "XMPTools`TempContext`"}, Module[{Exif = assc},Module[{or = If[Exif["Orientation"] =!= Missing["KeyAbsent", "Orientation"], 
    										Append[PrepareExifMetaFromProcess[Exif], <|"Orientation" -> GetOrientationNumber[PrepareExifMetaFromProcess[Exif]["Orientation"]]|>], 
    										PrepareExifMetaFromProcess[Exif]]},  If[Exif["FlashInfo"] =!= Missing["KeyAbsent", "FlashInfo"], 
  										    Append[or, "FlashInfo" -> GetFlashNumber[PrepareExifMetaFromProcess[Exif]["FlashInfo"]]], or]]]
 ]
IntToString = 
 <|
  "GPSStatus"                -> <|"Measurement in progress" -> "A", "Measurement Interoperability" -> "V", "Measurement is Interoperability" -> "V", "Measurement is interoperability" -> "V"|>,
  "GPSMeasureMode"           -> <|"Three-dimensional measurement" -> "3", "Two-dimensional measurement" -> "2"|>,
  "GPSLongitudeRef"          -> <|"West" -> "W", "East" -> "E"|>,
  "GPSLatitudeRef"           -> <|"North" -> "N", "South" -> "S"|>,
  "GPSAltitudeRef"           -> <|"AboveSeaLevel" -> 0, "BelowSeaLevel" -> 1|>,
  "GPSTrackRef"              -> <|"Magnetic direction" -> "M", "True direction" -> "T"|>,
  "GPSImgDirectionRef"       -> <|"Magnetic direction" -> "M", "True direction" -> "T"|>,
  "Sharpness"                -> <|"Normal" -> 0, "Soft" -> 1, "Hard" -> 2|>,
  "Saturation"               -> <|"Normal" -> 0, "Low" -> 1, "Hard" -> 2|>,
  "Contrast"                 -> <|"Normal" -> 0, "Soft" -> 1, "Hard" -> 2|>,
  "GainControl"              -> <|"None" -> 0, "Low gain up" -> 1, "High gain up" -> 2, "Low gain down" -> 3, "High gain down" -> 4|>,
  "SceneCaptureType"         -> <|"Standard" -> 0, "Landscape" -> 1, "Portrait" -> 2, "Night" -> 3|>,
  "WhiteBalance"             -> <|"Auto" -> 0, "Manual" -> 1|>,
  "ExposureMode"             -> <|"Auto" -> 0, "Manual" -> 1, "Auto Bracket"-> 2|>,
  "CustomRendered"           -> <|"Normal process" -> 0, "Custom process" -> 1|>,
  "SensingMethod"            -> <|"Not defined" -> 1, "Monochrome" -> 1, "One-chip color area" -> 2, "Two-chip color area" -> 3, "Three-chip color area" -> 4, "Color sequential area" -> 5, "Monochrome linear" -> 6,  "Trilinear" -> 7, "Color sequential linear" -> 8|>,
  "FocalPlaneResolutionUnit" -> <|"None" -> 1, "inch" -> 2, "cm" -> 3, "mm" -> 4, "um" -> 5|>,
  "ColorSpace"               -> <|"RGBColor" -> 1, "Uncalibrated" -> 65535|>,
  "LightSource"              -> <|"Unknown" -> 0, "Daylight Fluorescent" -> 12, "D55" -> 20, "Daylight" -> 1, "Day White Fluorescent" -> 13, "D65" -> 21, "Fluorescent" -> 2 , "Cool White Fluorescent" -> 14 , "D75" -> 22, "Tungsten (Incandescent)" -> 3, 
    							  "White Fluorescent" -> 15, "D50" -> 23, "Flash" -> 4, "Warm White Fluorescent" -> 16, "ISO Studio Tungsten" -> 24, "Fine Weather" -> 9 , "Standard Light A" -> 17,
    				    		  "Cloudy" -> 10, "Standard Light B" -> 18, "Shade" -> 11, "Standard Light C" -> 19|>,
  "MeteringMode"             -> <|"Unknown" -> 0, "Average" -> 1, "Center weighted average" -> 2, "Spot" -> 3, "Multi-spot" -> 4, "Multi-segment" -> 5, "Partial" -> 6|>,
  "PreviewColorSpace"        -> <|"Unknown" -> 0, "Gray Gamma 2.2" -> 1, "sRGB" -> 2, "Adobe RGB" -> 3, "ProPhoto RGB" -> 4|>, 
  "ExposureProgram"          -> <|"Not defined" -> 0, "Auto" -> 2, "Manual" -> 1, "Normal program" -> 2, "Aperture priority" -> 3, "Shutter priority" -> 4, "Creative program" -> 5, "Action program" -> 6, "Portrait mode" -> 7, "Landscape mode" -> 8, "Bulb" -> 9|>,
  "YCbCrPositioning"         -> <|"Centered" -> 1, "Cosited" -> 2, "Co-sited" -> 2|>,
  "OPIProxy"                 -> <|"Higher resolution image does not exist" -> 0, "Higher resolution image exists" -> 1 |>,
  "Indexed"                  -> <|"Not indexed" -> 0, "Indexed" -> 1|>,
  "SampleFormat"             -> <|"Unsigned" -> 1, " Unsigned integer data " -> 1, "Signed" -> 2, " Signed integer data " -> 2, "Float" -> 3, "Undefined" -> 4, "Complex int" -> 5, "Complex float" -> 6|>,
  "ExtraSamples"             -> <|"Unspecified" -> 0, "Associated Alpha" -> 1, "Unassociated Alpha" -> 2|>, "InkSet" -> <|"CMYK" -> 1, "Not CMYK" -> 2|>,
  "ResolutionUnit"           -> <|"None" -> 1, "inch" -> 2, "cm" -> 3|>,
  "GrayResponseUnit"         -> <|"0.1" -> 1, "0.001" -> 2, "0.0001" -> 3, "1e-05" -> 4, "1e-06" -> 5|>,"PlanarConfiguration" -> <|"Chunky" -> 1, "Planar" -> 2|>,
  "FillOrder"                -> <|"Normal" -> 1, "Reserved" -> 2|>,
  "Thresholding"             -> <|"No dithering or halftoning" -> 1, "Ordered dither or halftone" -> 2, "Randomized dither" -> 3|>,
  "PhotometricInterpretation"-> <|"WhiteIsZero" -> 0, "BlackIsZero" -> 1, "RGB" -> 2, "RGB Palette" -> 3, "Transparency Mask" -> 4, "CMYK" -> 5, "YCbCr" -> 6, "CIELab" -> 8 ,
   								  "ICCLab" -> 9, "ITULab" -> 10, "Color Filter Array" -> 32803, "Pixar LogL" -> 32844, "Pixar LogLuv" -> 32845, "Linear Raw" -> 34892|>,
  "Compression"              -> <|"Uncompressed" -> 1, "CCITT modified Huffman RLE" -> 2, "PackBits compression, aka Macintosh RLE" -> 32773, "CCITT Group 3 fax encoding" -> 3, 
  								  "CITT Group 4 fax encoding" -> 4, "LZW" -> 5, "JPEG (new-style)" -> 7, "JPEG (old-style)" -> 6, "Deflate" -> 7, "Defined by TIFF-F and TIFF-FX standard (RFC 2301) as ITU-T Rec. T.82 coding, using ITU-T Rec. T.85" -> 9, 
  								  "Defined by TIFF-F and TIFF-FX standard (RFC 2301) as ITU-T Rec. T.82 coding, using ITU-T Rec. T.43" -> 10|>,
  "SubfileType"              -> <|"Full-resolution image data" -> 0, "Reduced-resolution image" -> 1, "Single page of multi-page image" -> 2|>,
  "SubjectDistanceRange"     -> <|"Unknown" -> 0, "Macro" -> 1, "Close view" -> 2, "Distance view" -> 3|>,
  "GPSDifferential"          -> <|"Without correction" -> 0, "Correction applied"-> 1|>,
  "FileSource"               -> <|"Film scanner" -> 1, "Reflexion print scanner" -> 2, "Digital still camera"-> 3, _ -> "None"|>,
  "SceneType"                -> <|"Directly photographed" -> "1", _ -> "None"|>,
  "NewSubfileType"           -> <|"Primary image" -> 0, "Thumbnail/Preview image" -> 1, _ -> 0|>
  |> 
 
ParseIntToString[tag_, value_] := If[IntToString[tag, value] =!= Missing["KeyAbsent", value], IntToString[tag, value], -1] 

ModifyIntValuesForExport[assc_]:=
                                    Module[{ass =  Quiet[PrepareExifMetaForExport[assc]]},
 										If[ass["ComponentsConfiguration"] =!= Missing["KeyAbsent", "ComponentsConfiguration"],
 										Module[{cc = StringReplace[ass["ComponentsConfiguration"], {"Y" -> "1,", "Cb" -> "2,", "Cr" -> "3,", "R" -> "4,", "G" -> "5,", "B" -> "6,"}]},
  											 Which[
   												 Count[StringSplit[cc, ","], _String] === 3, cc = StringJoin[cc, "0"],
    											 Count[StringSplit[cc, ","], _String] === 2, cc = StringJoin[cc, "0,0"],
    											 Count[StringSplit[cc, ","], _String] === 1, cc = StringJoin[cc, "0,0,0"],
    											 True, cc
    										  ];
   											ass["ComponentsConfiguration"] = StringReplace[cc, "," -> " "];
   										]];
   										AssociateTo[ass, "DateTime"->DateString[Now, {"Year", "-", "MonthShort", "-", "DayShort", " ", "HourShort", ":", "MinuteShort", ":", "SecondShort"}]];
 										Association@(Normal@ass /. (key_ /; (!SameQ[ToString@key, "FocalLengthIn35mmFilm"] && !SameQ[ToString@key, "GPSVersionID"] && (SameQ[ToString@key, "SceneType"] ||MemberQ[ExportExifGPSDualValues, ToString@key] || MemberQ[ExportExifGPSInt, ToString@key] || MemberQ[ExportExifPhotoInt, ToString@key] || MemberQ[ExportExifImageInt, ToString@key] && !MemberQ[MultiValues, ToString@key] && !SameQ[key, "CFAPattern"] || SameQ[ToString@key, "FileSource"])) ->val_) :> (key -> If[StringQ@val, ParseIntToString[key, val], val]))
                                     ]

WriteExif[tag_, val_] := 
                     Block[{$Context = "XMPTools`TempContext`"},
                         Which[
  							 SameQ["ReferenceBlackWhite", ToString@tag]	,Quiet@$WriteExifString[tag, val],										      
 	                         SameQ["BitsPerSample", tag],      Quiet@$WriteExifString[tag, StringDelete[val, ","]],
 	                         SameQ["SubjectArea", tag],        Quiet@$WriteExifString[tag, val],
 	                         SameQ["MakerNote", tag]  ,        Quiet@$WriteExifString[tag, val],
 	                         SameQ["LensSpecification", tag]||
 	                         StringContainsQ[tag, "Version"]        , Quiet@$WriteExifString[tag, val],  
 	                         MemberQ[MultiValues, tag]              , Quiet@$WriteExifString[tag, Module[{res = val}, If[! ListQ@res, StringTrim@StringJoin[" " <> ToString[#, InputForm] & /@ (List@res)], 
  																							StringTrim@StringJoin[" " <> ToString[#, InputForm] & /@ res]]]],
 	                         
 	                         MemberQ[ExportExifGPSString, tag]   ||
 	                         MemberQ[ExportExifPhotoString, tag] ||
 	                         MemberQ[ExportExifImageString, tag] ||
 	                         MemberQ[ExportExifIopString, tag]      , Which[
 	                         	                           SameQ[tag, "CFAPattern"]              , If[StringContainsQ[val, " "] && StringFreeQ[val, LetterCharacter ..], Quiet@$WriteExifString[tag, ToString@val], _ ],
 	                         	                           SameQ[tag, "ComponentsConfiguration"] , If[StringContainsQ[val, " "] && StringFreeQ[val, LetterCharacter ..], $WriteExifString[tag, ToString@val], _ ],
 	                         	                           SameQ[tag, "OECF"]                    , If[StringContainsQ[val, " "] && StringFreeQ[val, LetterCharacter ..], $WriteExifString[tag, ToString@val], _ ],
 	                         	                           SameQ[tag, "SceneType"]               , Quiet@$WriteExifString[tag,  If[SameQ[val, "None"], "0", val]],
 	                         	                           SameQ[tag, "FileSource"]              , If[NumberQ[ToExpression@val], Quiet@$WriteExifString[tag, ToString@val], _ ],
 	                         	                           SameQ[tag, "SpatialFrequencyResponse"], If[NumberQ[ToExpression@val], Quiet@$WriteExifString[tag, ToString@val], _ ],
 	                         	                           SameQ[tag, "DeviceSettingDescription"], If[NumberQ[ToExpression@val], Quiet@$WriteExifString[tag, ToString@val], _ ],
 	                         	                           SameQ[tag, "RelatedSoundFile"]        , If[NumberQ[ToExpression@val], Quiet@$WriteExifString[tag, ToString@val], _ ],
 	                         	                           SameQ[tag, "GPSTimeStamp"]            , Quiet@$WriteExifString[tag, val],
 	                         	                           True, Quiet@$WriteExifString[tag, ToString@val]
 	                         	                          ],
 	                         MemberQ[ExportExifGPSInt, tag]    ||
 	                         MemberQ[ExportExifPhotoInt, tag]  ||
 	                         MemberQ[ExportExifImageInt, tag]  ||
 	                         MemberQ[ExportExifIopNumber, tag]      , Quiet@If[!NumberQ[ToExpression@val], 0 , Quiet@$WriteExifInt[tag, ToExpression@val]],
 	                         
 	                         MemberQ[ExportExifGPSRat, tag]  ||
 	                         MemberQ[ExportExifPhotoRat, tag]  ||
 	                         MemberQ[ExportExifImageReal, tag] ||
 	                         MemberQ[ExportExifImageRat, tag] &&
 	                         !SameQ[tag, "ExposureBiasValue"] &&
 	                         !SameQ[tag, "BaselineExposure"] , Quiet@$WriteExifReal[tag, If[StringQ@val,(ToExpression@val)//N, val//N]],
 	                         
 	                         True, _     
                         ]
                     ]

WriteExifRule[listOfRules : {__Rule}] := WriteExif @@@ listOfRules
WriteExifAssociation[list_Association]:= WriteExif @@@ Normal[list]
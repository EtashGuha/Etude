(* ::Package:: *)

(* ::Subsubtitle:: *)
(*BioSemi Data Format (BDF) Importer *)


(* ::Section::Closed:: *)
(*COPYRIGHT*)


(*************************************************************************

                        Mathematica source file

        Copyright 1986 through 2010 by Wolfram Research Inc.

This material contains trade secrets and may be registered with the
U.S. Copyright Office as an unpublished work, pursuant to Title 17,
U.S. Code, Section 408.  Unauthorized copying, adaptation, distribution
or display is prohibited.

*************************************************************************)


(* ::Section::Closed:: *)
(*FILE LAYOUT*)


(*****************************************
  This code is divided into two sections:
  I:  Functions seen directly by the registration
      
      ImportBDFElements[Stream]
      ImportBDFHeader[Stream]
      ImportBDFLabeledData[Stream]
      ImportBDFSignalData[String][Stream]
      ImportBDFSignalLabeledData[String][Stream]
      
  II: Utility functions.
*******************************************)


(* ::Section::Closed:: *)
(*BEGIN CONVERTER CONTEXT*)


Begin["System`Convert`BDFDump`"]


(* ::Section:: *)
(*IMPORT*)


(* ::Subsection:: *)
(*Importers*)


ImportBDFElements[strm_InputStream, opts___?OptionQ] :=
"Elements" -> Sort[{
  "Data", "ReocrdsData", "LabeledData", "Annotations", "RecordTimes",
  "ChannelCount", "FilterInformation", "Labels", "PatientID",
  "Title", "DataRanges", "DataUnits", "RecordLength", "StartDate", "Device"
}]


ImportBDFHeader[strm_InputStream, opts___]:=
Module[
  {out},
  out = GetBDFCompleteHeader[strm];
  If[ out === $Failed, Message[Import::fmterr, "BDF"]; Return[$Failed] ];
  out = out /. {Rule["RecordLength", x_] :> Rule["RecordLength", Quantity[x,"Seconds"]]};
  Return[Drop[out,-6]]
]


ImportBDFData[test_][strm_InputStream, opts___?OptionQ] := 
Block[{$ByteOrdering = -1},
Module[
  {Header, SigData, AnnoData, RecDuration, SignalPos,  AnnotationPos,
   RecordTimes, nDataRecords, ContQ},
                                        
  If[ !MemberQ[{"Data", "RecordsData", "LabeledData", "Annotations", "RecordTimes", All}, test],
      Message[Import::noelem, test, "BDF"]; Return[$Failed]
  ];

  Header = GetBDFCompleteHeader[strm];
  If[ Header === $Failed, Message[Import::fmterr, "BDF"]; Return[$Failed] ];
  SignalPos     = "SignalPos"/.Header;
  AnnotationPos = "AnnotationPos"/.Header;
  nDataRecords  = "RecordCount"/.Header;
  RecDuration   = "RecordLength"/.Header;

  If[ AnnotationPos =!= {},
      (* There should be only one Annotation block. *)
      AnnoData = ReadData[strm, Header, First@AnnotationPos ];
      RecordTimes = If[Length@AnnoData === 1, {}, First /@ AnnoData];
      AnnoData    = Flatten[Last /@ AnnoData, 1],
      (* If no annotation, simply reutn empty List. *)
      RecordTimes = Table[(i-1)*RecDuration,{i,1,nDataRecords}];
      AnnoData    = {}
  ];
  
  If[ test === "Annotations" || test === "RecordTimes",
      Return[{ "Annotations" -> AnnoData, "RecordTimes" -> If[RecordTimes=!={}, Map[Quantity[#,"Seconds"]&, RecordTimes], {}] }]
  ];
  
  (* Check whether records are continuous *)
  If[ RecordTimes =!= {},
      ContQ = SameQ[ RecordTimes, Table[ RecordTimes[[1]] + (i - 1)*RecDuration, {i, 1, Length@RecordTimes}] ],
      ContQ = True
  ];

  If[ Length@SignalPos =!= 0,
      SigData = First[Last[Reap[Do[ 
                 Sow[ReadData[strm, Header, SignalPos[[j]] ]],
                 {j, 1, Length[SignalPos]}
                ]]]],
      SigData = {}       
  ];
  
  Which[
    test === "RecordsData",
    Return[ "RecordsData" -> SigData ],

    test === "LabeledData",
    Return[ "LabeledData" -> MapThread[Rule,{"Labels"/.Header, SigData} ] ],

    test === "Data",
    Return[ "Data" -> Map[Flatten, SigData] ],

    True,
    Return[$Failed]
  ]
]
]


ImportBDFSignalData[channel_String, test_][strm_InputStream ,opts___?OptionQ] := 
Block[{$ByteOrdering = -1},
Module[
  {Header, SigLabels, SignalPos,
   AnnotationPos, AnnoData, RecordTimes, RecDuration,
   nDataRecords, SigData, ContQ},
  
  Header = GetBDFCompleteHeader[strm];
  If[ Header === $Failed, Message[Import::fmterr, "BDF"]; Return[$Failed] ];
  SignalPos     = "SignalPos"/.Header;
  AnnotationPos = "AnnotationPos"/.Header;
  RecDuration   = "RecordLength"     /.Header;
  nDataRecords  = "RecordCount"/.Header;
  SigLabels     = ("RawSignalHeader"/.Header)[[1]];

  If[ AnnotationPos =!= {},
      (* There should be only one Annotation block. *)
      AnnoData = ReadData[strm, Header, First@AnnotationPos ];
      RecordTimes = If[Length@AnnoData === 1, {}, First /@ AnnoData];
      AnnoData    = Flatten[Last /@ AnnoData, 1],
      
      RecordTimes = Table[(i-1)*RecDuration,{i,1,nDataRecords}];
      AnnoData    = {}
  ];

  (* Check whether records are continuous *)
  If[ RecordTimes =!= {},
      ContQ = SameQ[ RecordTimes, Table[ RecordTimes[[1]] + (i - 1)*RecDuration, {i, 1, Length@RecordTimes}] ],
      ContQ = True
  ];
  SignalPos = Position[ SigLabels, channel];
  If[ SignalPos === {}, Message[Import::dataset, channel];Return[$Failed]];
  SigData = ReadData[strm, Header, First@First@SignalPos];

  Which[
    test === "RecordsData",
    Return[ "RecordsData" -> {channel -> SigData} ],

    test === "LabeledData",
    Return[ "LabeledData" -> {channel -> Rule[channel,SigData]} ],

    test === "Data",
    Return[ "Data" -> {channel -> Flatten[SigData]} ],

    True,
    Return[$Failed]
  ]
]
]


ImportBDFLabeledData[strm_InputStream]:=
"LabeledData" -> MapThread[ Rule, {"DataLabels"/.GetBDFCompleteHeader[strm], "Data"/.ImportBDFData["Data"][strm]}]


(* ::Subsection:: *)
(*Utility Functions*)


deleteWhiteSpaces[str_String] :=
    StringReplace[str, RegularExpression["^\\s+|\\s+$"] -> ""]


(* This outputs a pure function *)
DigitalToPhysical[dmin_, dmax_, pmin_, pmax_]:=
Function[x, mDP[dmin, dmax, pmin, pmax] x + bDP[dmin, dmax, pmin, pmax]]


gain[dmin_, dmax_, pmin_, pmax_]:=  (pmax - pmin)/(dmax - dmin)
offset[dmin_, dmax_, pmin_, pmax_]:= -(dmax*pmin - dmin*pmax )/(dmax - dmin) 


CheckDataRecords[strm_InputStream, nBytesHeader_Integer, nrSamples_List]:=
Module[
  {pos, end},
  pos = StreamPosition[strm];
  end = SetStreamPosition[strm, Infinity];
  SetStreamPosition[strm, pos];
  (end - nBytesHeader) / Sum[nrSamples[[i]], {i, 1, Length@nrSamples}] / 3
]


GetFileHeader[str_InputStream] :=
Block[{$ByteOrdering = -1},
Module[
  {temp, lst},
  (* file size must be at least 256 bytes *) 
  If[ SetStreamPosition[str, Infinity] < 256, Return[$Failed] ];
  
  (* first byte used as identifier for BDF or EDF *)
  SetStreamPosition[str, 0];  
  temp = BinaryRead[str, "Byte"];
  
  lst =  FromCharacterCode[
            Map[ DeleteCases[BinaryReadList[str, "Byte", #],0] &, {7, 80, 80, 8, 8, 8, 44, 8, 8, 4}],
         "ASCII"];      
  
  (* check format identifier *)
  If[ First@lst =!= "BIOSEMI", Return[$Failed]];
  
  lst = lst /.s_String->deleteWhiteSpaces[s];
  lst = lst /.{exp_String /; StringMatchQ[exp, NumberString] :> ToExpression@exp};
  Return[lst];
] 
]


GetSignalHeader[str_InputStream, ns_Integer] :=
Block[{$ByteOrdering = -1},
Module[
  {lst},
  lst =  Map[FromCharacterCode, 
               Partition[
                 Map[ Function[x, DeleteCases[BinaryReadList[str, "Byte", x],0]],
                      Flatten[Transpose[Table[{16, 80, 8, 8, 8, 8, 8, 80, 8, 32}, {ns}]]]
                  ],ns
               ]
         ];
  lst = lst /.s_String->deleteWhiteSpaces[s];
  lst = lst /.{exp_String /; StringMatchQ[exp, NumberString] :> ToExpression@exp};         
  Return[lst]
]
]


ProcessRawAnno[in_List]:=
Module[
  {AnnoList, time},
  AnnoList = Map[Drop[#,-1]&, DeleteCases[Split[in, (#1 =!=0)&],{0}] ];
  (* The first item in each AnnoList is a TimeStamp. *)
  time = ToExpression[FromCharacterCode@First@Split[AnnoList[[1]],#2=!=20&]];
  AnnoList = Drop[AnnoList,1];
  (* Print[AnnoList];*)
  AnnoList = Map[ProcessAnnoList, AnnoList];
  {time, AnnoList}
]
  


ProcessAnnoList[in_List]:=
Module[
  {anno},
  (* To standardize annoation, we input a duration element if absent *)
  If[ !MemberQ[in, 21],
      anno = Flatten[Insert[in, {21,48}, First@First@Position[in, 20]]],
      anno = in
  ];
  anno = Map[FromCharacterCode[Drop[#,-1]]&, DeleteCases[Split[ anno, !(#1 === 20 || #1 ===21 )&], {20}]];
  {ToExpression[ anno[[1]] ],ToExpression[ anno[[2]] ]} -> Drop[anno,2] 
]



(********
  This function returns a list of header elements used by
  other importers.
 *******)
GetBDFCompleteHeader[strm_InputStream, opts___]:=
Module[
  {FileHeader, nDataRecords, nBytesHeader, ns,
   SignalHeader, SigLabels, nrSamples, SignalPos, AnnotationPos},
 
  FileHeader = GetFileHeader[strm];
  If[ FileHeader === $Failed,             Return[$Failed] ];
  nBytesHeader = ToExpression@FileHeader[[6 ]];
  nDataRecords = ToExpression@FileHeader[[8 ]];       
  ns           = ToExpression@FileHeader[[10]];
  
  (* Number of data records can be -1. If so, calculate based on file size. *)
  If[ nDataRecords === -1,
      nDataRecords = CheckDataRecords[strm, nBytesHeader, nrSamples]
  ];
  If[ !IntegerQ[nDataRecords], Return[$Failed] ];  
  
  (* Do the error check again since there might be some corruption in the file *)
  If[ (!MatchQ[ns,_Integer] || !MatchQ[nBytesHeader,_Integer]||!MatchQ[nDataRecords,_Integer]),
      Return[$Failed]
  ];         
  SignalHeader  = GetSignalHeader[strm, ns];       (* Read the signal header *)         
  SigLabels     = SignalHeader[[1]];               (* List of signal labels *)            
  nrSamples     = ToExpression@SignalHeader[[9]];  (* List of samples of all signals *)
  SignalPos     = Map[First@First@Position[SigLabels, #] &, Select[SigLabels, # =!= "BDF Annotations" &]];
  AnnotationPos = Map[First@First@Position[SigLabels, #] &, Select[SigLabels, # === "BDF Annotations" &]];

  {
     (* Items from the FileHeader *)
     "PatientID"     ->  FileHeader[[2]]/.{""->None},
     "Title"         ->  FileHeader[[3]]/.{""->None},
     "StartDate"     ->  Join[Drop[DateList[{FileHeader[[4]],{"Day",".","Month",".","YearShort"}}],-3],Drop[DateList[{FileHeader[[5]],{"Hour24",".","Minute",".","Second"}}],3]],
     "RecordCount"   ->  nDataRecords,
     "RecordLength"  ->  FileHeader[[9]],
     "ChannelCount"  ->  Length@SignalPos,
     (* Items from the SignalHeader *)
     "Labels"      ->  SignalHeader[[1]][[SignalPos]]/.{""->None},
     "Device"      ->  SignalHeader[[2]][[SignalPos]]/.{""->None},
     "DataUnits"   ->  SignalHeader[[3]][[SignalPos]]/.{"mV"->"Millivolts",
                                              "uV"->"Microvolts", "nV"->"Nanovolts", "pV"->"Picovolts", ""-> None,
                                              "%"-> "Percentage"},
     "DataRanges"  ->  MapThread[ {#1 , #2}&, {SignalHeader[[4]][[SignalPos]], SignalHeader[[5]][[SignalPos]]} ],
     "FilterInformation" ->     SignalHeader[[8]][[SignalPos]]/.{""->None},
     (* Items used in other functions *)
     "SignalPos"       ->  SignalPos,
     "AnnotationPos"   ->  AnnotationPos,
     "HeaderBytes"     ->  nBytesHeader,
     "ns"              ->  ns,
     "RawSignalHeader" ->  SignalHeader,
     "DigitalRange"     -> MapThread[ {#1 , #2}&, {SignalHeader[[6]][[SignalPos]], SignalHeader[[7]][[SignalPos]]} ]
  }
]

(**** 
   given the stream and the header information,
   this function reads either a data block or an annotation block,
   depending on the value ToRead
 ****)
ReadData[strm_InputStream, Header_, ToRead_Integer]:=
Module[
  {byteSkip, i, j, nDataRecords, nBytesHeader, RecDuration, SigLabels, nrSamples,
   dRange, pRange, m, b, out},
  nDataRecords = "RecordCount" /.Header;      
  nBytesHeader = "HeaderBytes"  /.Header;
  RecDuration  = "RecordLength" /.Header;

  SigLabels    = ("RawSignalHeader"/.Header)[[1]];             
  nrSamples    = ("RawSignalHeader"/.Header)[[9]];
  byteSkip = Table[ nBytesHeader +  (i-1)*3*Total[nrSamples] + 3*Total[Take[nrSamples, (ToRead-1)]], {i, 1, nDataRecords}];
  
  If[ SigLabels[[ ToRead ]] =!= "BDF Annotations",
      (* Read a typical signal block *)
      dRange   = ("DigitalRange"/.Header)[[ToRead]];
      pRange   = ("DataRanges" /.Header)[[ToRead]];
      m = N[  gain[First@dRange, Last@dRange, First@pRange, Last@pRange]];
      b = N[offset[First@dRange, Last@dRange, First@pRange, Last@pRange]];
      out = First[Last[Reap[Do[
         SetStreamPosition[ strm, byteSkip[[j]] ];
         Sow[ Developer`ToPackedArray[BinaryReadList[strm, "Integer24", nrSamples[[ ToRead ]], ByteOrdering -> -1]]  ],
         {j, 1, nDataRecords}
      ]]]];
      m out + b
      ,
      (* Read the annotation block *)
      First[Last[Reap[Do[
         SetStreamPosition[ strm, byteSkip[[j]] ];
         Sow[ ProcessRawAnno[ Developer`ToPackedArray[BinaryReadList[strm, "Byte", 3*nrSamples[[ ToRead ]], ByteOrdering -> -1]]] ],
         {j, 1, nDataRecords}
      ]]]]
  ]       
]   


(* ::Section:: *)
(*BEGIN CONVERTER CONTEXT*)


End[];

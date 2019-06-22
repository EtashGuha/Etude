BeginPackage["HTTPLink`ImportExport`"]

Begin["`Private`"]

Needs["HTTPLink`"];

(*note the ordering of this association is relevant - it determines the display order in the dataset*)
$WolframWARCHeaderKeyMapping = <|
	"warc-target-uri"->"URL",
	"content"->"Content",
	"content-type"->"ContentType",
	"warc-date"->"AccessDate",
	"warc-type"->"WARCType",
	"warcversion"->"WARCVersion",
	"warc-record-id"->"WARCRecordID"
|>;

wolframizeHeadersAssocs[assocs_]:= (
	(*this keytake will order the association in the appropriate order*)
	KeyTake[Values[$WolframWARCHeaderKeyMapping]]@
		KeyMap[$WolframWARCHeaderKeyMapping]@
			KeyTake[Keys[$WolframWARCHeaderKeyMapping]]@
				(KeyMap[ToLowerCase]@#)
	)& /@ assocs

wolframizeHeadersListRules[rules_]:=(
	(*this keytake will order the association in the appropriate order*)
	KeyTake[Values[$WolframWARCHeaderKeyMapping]]@
		KeyMap[$WolframWARCHeaderKeyMapping]@<|
			FilterRules[
				MapAt[ToLowerCase,#,{All,1}],
				Keys[$WolframWARCHeaderKeyMapping]
			]
	|>)& /@ rules


(*simple wrapper functions around the importers / exporters*)
HTTPLink`ImportExport`Symbols`importHTTPRequestStream[stream_InputStream,opts:OptionsPattern[]] := 
	{"HTTPRequest"->ParseHTTPRequest[BinaryReadList[stream]]}
HTTPLink`ImportExport`Symbols`exportHTTPRequestStream[stream_OutputStream,expr_HTTPRequest,opts:OptionsPattern[]] := 
	(*because the format is binary, to stop any encodings, rather than Write the string out on the stream*)
	(*which may have been opened with some wierd character encoding, we just write out the raw bytes of the *)
	(*http request out using ToCharacterCode with the character encoding ExportHTTPRequest found*)
	With[{res = ExportHTTPRequest[expr]}, If[FailureQ[res],res,BinaryWrite[stream,res]]]


HTTPLink`ImportExport`Symbols`importHTTPResponseStream[stream_InputStream,opts:OptionsPattern[]] := 
	{"HTTPResponse"->ParseHTTPResponse[BinaryReadList[stream]]}

HTTPLink`ImportExport`Symbols`exportHTTPResponseStream[stream_OutputStream,expr_HTTPResponse,opts:OptionsPattern[]] :=
	(*same as for request exporting, we write out the raw bytes on the stream*)
	With[{res = ExportHTTPResponse[expr]}, If[FailureQ[res],res,BinaryWrite[stream,res]]]


(*invalid exporter arguments*)
HTTPLink`ImportExport`Symbols`exportHTTPResponseStream[stream_OutputStream,expr_,opts:OptionsPattern[]] := (
	Message[Export::fmterr,"HTTPResponse"];
	$Failed	
)
HTTPLink`ImportExport`Symbols`exportHTTPRequestStream[stream_OutputStream,expr_,opts:OptionsPattern[]] := (
	Message[Export::fmterr,"HTTPRequest"];
	$Failed	
)


(*helper function for decoding warc types that don't have a defined charset*)
(*we just hope that it's utf-8, falling back to ISO8859-1 if anything fails*)
tryUtf8Decode[bytes_]:=Quiet[
	Check[
		FromCharacterCode[bytes, "UTF-8"],
		FromCharacterCode[bytes, "ISO8859-1"]
	],
	$CharacterEncoding::utf8
]

HTTPLink`ImportExport`Symbols`importWARCStreamRawData[stream_InputStream,opts:OptionsPattern[]] := 
Block[{notDone = True},
	{"RawData"->
		(Join@@Last@Reap[
			While[notDone,
				Block[
					{warcMetaData, httpResponse, warcLines, warcElement, payloadString, warcVersion},
					(
						(*read the warc metadata first*)
						warcMetaData = Read[stream, Record, NullRecords->True, RecordSeparators -> {"\r\n\r\n","\n\n\n\n","\n\n"}];
						If[warcMetaData === EndOfFile,
							notDone = False,
							(*to get the metadata for the warc metadata, we can just split by the CRLF sequence and make every line *)
							(
								(*except the first into a rule*)
								(*the first line of the warcMetaData should always be the version number of WARC*)
								splitData = StringSplit[StringTrim[warcMetaData],"\r\n"|"\n",2];
								If[Length[splitData] === 2,
									{warcVersion, warcLines} = splitData;
									,
									Continue[]
								];

								If[StringMatchQ["WARC/" ~~ ((DigitCharacter ..) ~~ ".") .. ~~ DigitCharacter ..] @ warcVersion,
									(*THEN*)
									(*got the version number, add that to the assoc*)
									(
										warcElement = {
											"WARCVersion" -> warcVersion,
											(*parse the rest of the rules*)
											Sequence @@ (HTTPLink`Private`importHeaderString[warcLines])
										};

										(*now here we want to skip past the record separator that's currently on the stream*)
										(*this is necessary because we don't know how many characters are now at the head of the*)
										(*stream, which we need to skip over, but we do know exactly how many bytes we should be *)
										(*expecting, so we have to drop those delimiters using TokenWords and Skip*)
										Skip[stream, Word, TokenWords->{"\r\n\r\n","\n\n\n\n","\n\n"}, RecordSeparators->{},WordSeparators->{}];
										(*now we can read the length from the stream appropriately*)
										(*we have to use BinaryReadList to get all of the individual characters as bytes*)
										(*this is because in some instances there may be an "invalid" escape*)
										(*sequence inside the string such as the string he\llo which is an invalid *)
										(*mathematica string use of the \ character, but entirely valid in the format*)
										(*we also use this NestWhile to drop all the leading white space characters from*)
										(*the bytes*)
										payloadBytes = NestWhile[
											(*we have to drop all the leading whitespace because the payload shouldn't have this*)
											Drop[#, 1] &,
											BinaryReadList[
												stream,
												"Byte",
												FromDigits[
													Lookup["content-length"]@MapAt[ToLowerCase,warcElement,{All,1}],
													10
												]
											],
											MemberQ[First[#]]@{9, 10, 11, 12, 13, 32} &
										];

										Sow[Append[warcElement,"content"->ByteArray[payloadBytes]]];
									),
									(*ELSE*)
									(*we got an invalid warc stream, return $Failed*)
									(
										(*raise message and Sow failed so that if at least there are other*)
										(*warc elements that succeeded up to this point it still works*)
										Message[Import::warcversion];
										notDone = False;
										Sow[$Failed]
									)
								]
							)
						]
					)
				]
			]
		]
		(*if the only result failed, and is just a {$Failed}, then we make it $Failed so it can easily be compared with FailureQ[] in the other*)
		(*element importers*)
		)/.{$Failed} :> $Failed
	}
]

parsePayload[payload_,type_,contentType_,recordID_] := (
	If[#===$Failed,
		(*THEN*)
		(*if we failed to parse the request or response, or otherwise get $Failed, just return back the payload as we got it*)
		(
			Message[Import::warccontent,recordID];
			payload
		),
		(*ELSE*)
		(*didn't fail to parse, so just pass back*)
		#
	]&@Which[
		(*for warc requests and responses, check to make sure that the content-type is application/http*)
		type == "request" && StringMatchQ[contentType, "application/http"~~___],
		HTTPLink`ParseHTTPRequest[Normal@payload],
		type == "response" && StringMatchQ[contentType, "application/http"~~___],
		HTTPLink`ParseHTTPResponse[Normal@payload],
		(*for all other types, there isn't a "default" character encoding for the body bytes of the *)
		(*warc element, so we just hope that it's UTF-8, falling back to ISO8859-1 if FromCharacterCode complains*)
		type == "metadata",
		(*don't make this into an association because for example with outLink, there could be *)
		(*duplicate rules*)
		HTTPLink`Private`importHeaderString[tryUtf8Decode[Normal@payload]],
		True,
		tryUtf8Decode[Normal@payload]
	]
	
)


(*"RawDataset" is a step above the "RawData" in that we conform some of the keys like payload and WARC-Date to be WL objects*)
(*but still leave all of the headers in the elements*)
HTTPLink`ImportExport`Symbols`importWARCStreamRawDataset[stream_InputStream,opts:OptionsPattern[]] := With[
	{res = Lookup["RawData"]@HTTPLink`ImportExport`Symbols`importWARCStreamRawData[stream,opts]},
	{
		"RawDataset" -> If[FailureQ[res],
			(*THEN*)
			(*pass along the failure*)
			res,
			(*ELSE*)
			(*not a failure, we can proceed with parsing*)
			(*to turn the raw data into the raw dataset, we just fix the Content key into the proper object*)
			Dataset[
				With[{warcElement = MapAt[ToLowerCase,#,{All,1}]},
					Merge[
						Association/@Join[Select[!StringMatchQ[First[#1], "content"|"warc-date"|"warc-type"|"warcversion", IgnoreCase->True] &]@ #,
							{
								"Content"->parsePayload[
									Lookup["content"]@warcElement,
									Lookup["warc-type"]@warcElement,
									Lookup["content-type"]@warcElement,
									Lookup["warc-record-id"]@warcElement
								],
								"WARC-Date"->DateObject[Lookup["warc-date"]@warcElement],
								"WARCType"->Capitalize[Lookup["warc-type"]@warcElement],
								"WARCVersion"->StringDelete[Lookup["warcversion"]@warcElement,StartOfString~~"WARC/"]
							}
						],
						If[Length[#]===1,First[#],Identity[#]]&
					]
				]& /@ Select[!FailureQ[#] &]@res
			]
		]
	}
]

(*finally "Dataset" is the last step above "RawDataset", where we take the headers parsed to WL names and parse out the values*)
(*of the headers and such*)
HTTPLink`ImportExport`Symbols`importWARCStreamDataset[stream_InputStream,opts:OptionsPattern[]] := With[
	{res = Lookup["RawDataset"]@HTTPLink`ImportExport`Symbols`importWARCStreamRawDataset[stream,opts]},
	{
		"Dataset"->If[FailureQ[res],
			res,
			Dataset[wolframizeHeadersAssocs[Select[!FailureQ[#] &]@Normal[res]]]
		]
	}
]

(*"RawStringDataset" is a step above "RawData" insofar as we still keep the raw data for each header, but we rename*)
(*all the headers and we only keep the ones we care about*)
HTTPLink`ImportExport`Symbols`importWARCStreamRawStringDataset[stream_InputStream,opts:OptionsPattern[]] := With[
	{res = Lookup["RawData"]@HTTPLink`ImportExport`Symbols`importWARCStreamRawData[stream,opts]},
	{
		"RawStringDataset"->If[FailureQ[res],
			res,
			Dataset[wolframizeHeadersListRules[Select[!FailureQ[#] &]@res]]
		]
	}
]




End[]

EndPackage[]
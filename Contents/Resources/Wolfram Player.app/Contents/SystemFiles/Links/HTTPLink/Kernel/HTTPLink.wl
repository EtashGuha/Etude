(* Wolfram Language Source File *)

BeginPackage["HTTPLink`"]

ParseHTTPRequest::usage = "parses an http/1.1 or http/1.0 protocol string request"
ParseHTTPResponse::usage = "parse an http protocol string response"
ExportHTTPResponse::usage = "exports a symbolic HTTPResponse object to a string"
ExportHTTPRequest::usage = "exports a symbolic HTTPRequest object to a string"

Begin["`Private`"]

initializeHTTPLinkLibs[]:=If[!ValueQ[$Initalized] || !$Initalized,
	(*THEN*)
	(*need to initialize the library*)
	With[{lib = FindLibrary["libhttplink"]},
		LibraryLoad[lib];
		parseFun = LibraryFunctionLoad[lib, "ParseHTTP", LinkObject, LinkObject];
		$Initalized = True;
	]
	(*ELSE*)
	(*already initialized, nothing to do*)
];

(*initialize the library*)
initializeHTTPLinkLibs[];


fixEndOfHTTPReq[req_?ListQ]/; Length[req]>=4 := If[
	(*last 2 chars are newlines - valid*)
	(req[[-1]] === 10 && req[[-2]] === 10) ||
	(*last 4 chars are \r\n\r\n*)
	(req[[-4]] === 13 && req[[-3]] === 10 && req[[-2]] === 13 && req[[-1]] === 10),
	(*THEN*)
	(*the request is good and we can just return it as is*)
	req,
	(*ELSE*)
	(*missing the end of the request, but since we're being passed the entire thing anyways as a list/string*)
	(*we should just append it to the end so it's valid*)
	(*for example when reading from a stream and specifying 
		Read[strm,Record,RecordSeparators->{"\r\n\r\n","\n\n"}]
	then the string record won't contain the record separator and we still want easy parsing in this case*)
	(*most efficient to just append LFLF as opposed to 2 CRLF sequences*)
	Join[req,{10,10}]
]

(*any other form, just return it as is*)
fixEndOfHTTPReq[req_]:=req

(*we have to specify how many headers to allow, as they're not dynamically allocated in the parse loop*)
$DefaultMaxHeaders = 100

(*this assumes that the request is a Sequence of byte values*)
(*and parses the headers assuming they are ascii, then turns the*)
(*body into an appropriate string using the character encoding determined*)
(*from the headers*)
ParseHTTPRequest[req_?ListQ] := Block[
	{
		res = parseFun[fixEndOfHTTPReq[req],0,$DefaultMaxHeaders],
		lowerCaseHeaders
	},
	If[AssociationQ[res],
		(*THEN*)
		(*succeeded, then we can build up the HTTPRequest*)
		(
			lowerCaseHeaders = KeyMap[ToLowerCase]@res["Headers"];
			(*check required headers*)
			If[!KeyExistsQ[lowerCaseHeaders, "host"],
				(*THEN*)
				(*missing host header, can't continue as per http/1.1*)
				(
					Message[Import::httpreqhost];
					$Failed
				),
				(
					HTTPRequest[
						Join[
							(*to get stuff like the query parameters, etc. we can just build up the entire raw url and let URLParse handle all that parsing*)
							(*note that the headers names are case insensitive, so we have to make them all lowercase before*)
							(*getting the host out*)
							URLParse["http://"<>lowerCaseHeaders["host"]<>res["RawPath"]],
							(*from the result that we get from mathlink, take out the http version, the method, headers, and body as those can be used as normal with the HTTPRequest*)
							KeyTake[res,{"Method","Headers","Body","HTTPVersion"}]
						]
					]
				)
			]
		),
		(*ELSE*)
		(*failed - issue message and exit*)
		(
			(*TODO: issue message*)
			Message[Import::fmterr,"HTTPRequest"];
			$Failed
		)
	]
]



(*this helper function takes a list of bytes and will iterate through them finding where the actual content is*)
(*so that the chunked length encoding can be discarded and the entire content can be returned*)
parseChunkedEncoding[buf_] := Block[{i = 1, lenStart, chunklen},
	(*join all the data together into a single list*)
	Join @@ (
		(*Reap will give us the positions of the buffer we need to extract, so map over the extraction*)
		(*to get all the individual chunks*)
		Take[buf, #] & /@ (
			Join @@ Last@Reap[
				(*iterate while we haven't walked off the buffer*)
				While[i <= Length[buf],
					(
						(*start looking for the end of the length from here*)
						lenStart = i;
						If[i < Length[buf],
							(*start i at where the \n char should be*)
							i++,
							(*invalid spec - never found the length spec in the *)
							Return[$Failed]
						];
						While[! (buf[[i - 1]] == 13 && buf[[i]] == 10),
							If[i < Length[buf],
								(*keep going*)
								i++,
								(*invalid spec - never found the length spec in the *)
								Return[$Failed]
							]
						];
						(*found start end of line for the chunk length*)
						(*compute the length of the chunk using the bytes as ASCII encoded base16*)
						chunklen = FromDigits[StringTrim[FromCharacterCode[buf[[lenStart ;; i - 2]], "ASCII"]],16];
						(*increment i past the separator chars*)
						i++;
						(*now sow this chunk's start and length*)
						Sow[{i, i + chunklen - 1}];
						(*now increment i past this chunk before continuing on looking for the next chunk*)
						i += chunklen + 1
					)
				]
			]
		)
	)
]


parseChunkedEncodingStream[strm_] := Block[{lenChunk},
	(*join all the data together into a single list*)
	(*Reap will give us the individual chunks together here as we read them*)
	Join@@(
		Join @@ Last@Reap[
			(*iterate while we haven't walked off the buffer*)
			lenChunk = Read[strm, Record, RecordSeparators->{"\r\n"}];
			While[lenChunk =!= EndOfFile,
				(
					(*now skip past the record separator*)
					Skip[strm, Word, TokenWords->{"\r\n"}, RecordSeparators->{},WordSeparators->{}];

					(*now decode the len into an actual length to read off of the stream*)
					Sow[
						BinaryReadList[
							strm,
							"Byte",
							FromDigits[StringTrim[lenChunk],16]
						]
					];

					(*read up to the \r\n for the next token length*)
					lenChunk = Read[strm, Record, RecordSeparators->{"\r\n"}];
				)
			]
		]
	)
]

(*same disclaimer about bytes arguments for http response as http request*)
ParseHTTPResponse[resp_?ListQ] := Block[
	{
		res = parseFun[resp,1,$DefaultMaxHeaders]
	},
	If[AssociationQ[res],
		(*THEN*)
		(*succeeded, then we can build up the HTTPResponse*)
		(
			(*note though that while picohttp parser will correctly handle chunked encoding, we need to remove those byte chunks and chunk lengths from the body*)
			(*this means we can check if the Transfer-Encoding is chunked, in which case we have to parse through the chunks*)
			If[(KeyMap[ToLowerCase]@res["Headers"])["transfer-encoding"] === "chunked",
				(*THEN*)
				(*need to go through and remove all the extra chunk length headers*)
				(
					Quiet[
					(*if we run into issues parsing the chunked encoding, it's probably because it was already decoded*)
					(*so just return the body as we found it in the request*)
						Check[
							HTTPResponse[parseChunkedEncoding[res["Body"]], KeyDrop["Body"]@res],
							HTTPResponse[res["Body"], KeyDrop["Body"]@res]
						]
					]
				),
				(*ELSE*)
				(*no special handling it wasn't chunked*)
				HTTPResponse[res["Body"], KeyDrop["Body"]@res]
			]
		),
		(*ELSE*)
		(*failed - issue message and exit*)
		(
			(*TODO: issue message*)
			Message[Import::fmterr,"HTTPResponse"];
			$Failed
		)
	]
]

importHeaderString[headerStr_?StringQ] := (Rule @@@ 
	MapAt[
		StringTrim,
		StringSplit[#, ":", 2] & /@ StringSplit[headerStr, "\r\n"|"\n"],
		{All,2}
	]
)

handleHeaders[headers_]:=(StringJoin[Riffle[#, ": "]] & /@ List @@@ headers)


ExportHTTPResponse[___] := (Message[Export::fmterr,"HTTPResponse"];$Failed)
(*an invalid HTTPResponse will not have any properties*)
ExportHTTPResponse[resp_HTTPResponse] /; resp["Properties"] =!= {} := Block[{bodyArr = resp["BodyByteArray"]},
	Join@@{
		StringToByteArray[
			StringRiffle[
				{
					(*for now, we just always use http version 1.1*)
					"HTTP/1.1 " <> ToString[resp["StatusCode"]] <> " " <> resp["StatusCodeDescription"],
					Sequence @@ handleHeaders[resp["Headers"]]
				},
				{"","\r\n","\r\n\r\n"}
			],
			(*okay to use ASCII here, all HTTP headers are required to be ASCII*)
			"ASCII"
		],
		(*if the transfer-encoding header is chunked, then we need to add the chunked length*)
		If[(!MissingQ[#] && ToLowerCase[StringTrim[#]] === "chunked")& @ Lookup["transfer-encoding"]@MapAt[ToLowerCase,resp["Headers"],{All,1}],
			ByteArray[ToCharacterCode[IntegerString[Length[bodyArr], 16]<>"\r\n"]],
			Nothing
		],
		bodyArr
	}
]

ExportHTTPRequest[___] := (Message[Export::fmterr,"HTTPRequest"];$Failed)
ExportHTTPRequest[req_HTTPRequest] /; req["Properties"] =!= {} := Block[
	{
		urlParseKeys = {"Scheme", "User", "Domain", "Port", "Path", "Query", "Fragment", "PathString"},
		reqAssoc = Quiet[
			AssociationMap[
				req,
				Append[DeleteCases[req["Properties"],"BodyBytes"|"Body"],CharacterEncoding]
			]
		],
		urlParsed,
		bodyArr
	},
	urlParsed = AssociationThread[urlParseKeys, URLParse[reqAssoc["URL"],urlParseKeys]];
	bodyArr = reqAssoc["BodyByteArray"];
	If[!StringQ[urlParsed["Domain"]],
		(*THEN*)
		(*the domain is missing or wrong - issue message and fail*)
		(
			Message[Export::httpreqdom,req];
			$Failed
		),
		(*ELSE*)
		(*the domain is good to go - build up the string*)
		Join[
			StringToByteArray[
				StringRiffle[
					{
						reqAssoc[Method] <> " " <> (urlParsed["PathString"] /. {"" -> "/"}) <> If[reqAssoc["QueryString"] =!= "", "?" <> reqAssoc["QueryString"], ""] <> " HTTP/1.1",
						(*manually handle the Host parameter using the url parsed version we got*)
						"host: " <> urlParsed["Domain"] <> If[IntegerQ[#], ":" <> ToString[#], ""] &@urlParsed["Port"],
						(*check if the method is post, in which case we have to check on the body*)
						If[reqAssoc["BodyByteArray"] =!= {} && reqAssoc[Method] === "POST",
							(*THEN*)
							(*we need to add the Content-Length*)
							"content-length: "<>ToString[Length[bodyArr]],
							(*ELSE*)
							(*dont' need to add the Content-Length cause it's either not a POST*)
							(*or the body is an empty string*)
							Nothing
						],
						(*we need to drop the host header cause we build it ourselves*)
						Sequence @@ handleHeaders[Select[req["Headers"], !StringMatchQ[First[#], "host", IgnoreCase -> True] &]]
					},
					{"","\n","\n\n"}
				],
				(*okay to use ASCII here, all HTTP headers are required to be in ASCII*)
				"ASCII"
			],
			bodyArr
		]
	]
]


End[]

EndPackage[]

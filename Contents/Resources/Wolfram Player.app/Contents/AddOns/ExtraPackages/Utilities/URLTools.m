(* :Copyright: Copyright 2010, Wolfram Research, Inc. *)

(* :Mathematica Version: 8.0 *)

(* :Title: Tools for working with URLs *)

(* :Author: Yifan Hu, Tom Wickham-Jones, Igor C. Antonio *)

(* :Keywords: FetchURL, UncompressGZIPFile *)


(* :Summary:
This package uses J/Link calls to copy files from a URL to a local file. It also
provides a tool for unpacking GZIP files.
*)

BeginPackage["Utilities`URLTools`", {"JLink`"}]

Needs["PacletManager`"]
Needs["CURLLink`"]

FetchURL::usage = "FetchURL[ url, new] copies the file referenced by url to the file new on the local machine. FetchURL[ url ] copies the file creating a new filename. FetchURL returns the name of the temporary file used on the local machine."

FetchURLWithHeaders::usage = "FetchURLWithHeaders[ url, new] copies the file referenced by url to the file new on the local machine.  FetchURLWithHeaders[ url ] copies the file creating a new filename.  FetchURLWithHeaders returns a list of two elements: the name of the temporary file used on the local machine and the HTTP header fields."

URLInformation::usage = URLInformation::obs = "URLInformation has been removed. Use FetchURLWithHeaders to get the response headers associated with a given URL."

UncompressGZIPFile::usage = "UncompressGZIPFile[ old, new] unformats the gzip file old into new."

FileFilters::usage = "FileFilters is an option of FetchURL that sets up pairs {patt, fun}. Each of these apply the function 'fun' to the file if the name matches patt. This can be used for decompressing files."



Begin["`Private`"];

$UserAgent = Automatic;

FetchURL::conopen="The connection to URL `1` cannot be opened. If the URL is correct, you might need to configure your firewall program, or you might need to set a proxy in the Internet connectivity tab of the Preferences dialog (or by calling SetInternetProxy).  For HTTPS connections, you might need to inspect the authenticity of the server's SSL certificate and choose to accept it."

FetchURL::httperr="The request to URL `1` was not successful. The server returned the HTTP status code `2`."

FetchURL::contime =
    "The maximum connection time of `1` seconds has been exceeded.";

FetchURL::nofile = UncompressGZIPFile::nofile = "File `1` cannot be opened.";

FetchURL::nolib = "THe CURLLink library could not be located or initialized."

FetchURL::erropts = "The value of option `1` -> `2` is invalid. Using default setting."

UncompressGZIPFile::outfile = "File name `1` is not a string.";


DEBUG = False; MONITOR = False;


(* exceptions *)
{NOLIB, DISALLOWINTERNET, BADCONNECTION, CANTOPENOUTFILE, BADFILE, CONTIMEOUT, NOTSTRING};


(*
Create a temporary unique file name. If a suffix is given this attached to
the file after three _ characters.
*)

createTempFile[ prefix_String] := createTempFile[ prefix, ""]

createTempFile[ prefix_String, suffix_String] :=
    Close[OpenTemporary[]] <> "_" <> prefix <> If[ StringLength[ suffix] === 0, "", "___" <> suffix]



(*
 Returns the chars after the last '/' character.
*)

getFileName[ str_String] :=
    Module[ {pos},
      (* drop all except word characters (including _), and . *)
      pos = Complement[
          StringPosition[str, RegularExpression["\\W"]],
          StringPosition[str, "."]
      ];
        If[ pos === {},
            "unknown.dat",
            pos = pos[[-1,1]];
            StringDrop[ str, pos]]
    ]

(*
 Process any exceptions that were thrown, issue messages as appropriate.
*)

processException[ _, URLToolsException[ except_, arg_]] :=
    (Switch[ except,
            NOLIB, Message[FetchURL::nolib],
            DISALLOWINTERNET, Message[FetchURL::offline],
            BADCONNECTION, Message[FetchURL::conopen, arg],
            BADSTATUSCODE, Message[FetchURL::httperr, arg[[1]], statusCodeToDescription[arg[[2]]]],
            CONTIMEOUT, Message[FetchURL::contime, CONTIME],
            BADFILE, Message[UncompressGZIPFile::nofile, arg],
            CANTOPENOUTFILE, Message[FetchURL::nofile, arg],
            NOTSTRING, Message[UncompressGZIPFile::outfile, arg],
            _,1]; $Failed)


(* Arbitrary set of predefined messages. Fallthrough will just give the status code and no extra text. *)
statusCodeToDescription[400] = "400 (\"Bad Request\")"
statusCodeToDescription[401] = "401 (\"Unauthorized\")"
statusCodeToDescription[403] = "403 (\"Forbidden\")"
statusCodeToDescription[404] = "404 (\"Not Found\")"
statusCodeToDescription[407] = "407 (\"Proxy Authentication Required\")"
statusCodeToDescription[500] = "500 (\"Internal Server Error\")"
statusCodeToDescription[503] = "503 (\"Service Unavailable\")"
statusCodeToDescription[statusCode_] := ToString[statusCode]

(*
 Implementation of FetchURL.
*)

(* Hate to mix string and non-string options, but then I hate to introduce a symbol like
   Timeout for a pseudo-documented internal utility function. It is already used as a
   string option in WebServices, so I copy that here.
*)
Options[FetchURL] = Options[FetchURLInternal] = Options[FetchURLWithHeaders] = {
    FileFilters -> {{".gz", UncompressGZIPFile}},
    "ServerAuthentication" -> Automatic, (* An old name; synonym for VerifyPeer (and actually, a better name than that one). *)
    "VerifyPeer" -> False,
    "Timeout" -> Automatic, (* No longer used. Leave in for possible future restoration? *)
    "RequestMethod"-> "GET",
    "RequestParameters" -> {},
    "RequestHeaderFields" -> {},
    "Username" -> None,
    "Password" -> None
}


ValidFilter[ {{_String, _}..}] := True

ValidFilter[ ___] := False


(*
 If patt matches the end of name, then apply fun to the file.
 Returning the new file name.
*)

ApplyFilter[ {patt_String, fun_}, file_, nameIn_, autoName_] :=
    Module[ {fileOut = file, name = nameIn},
        If[ StringMatchQ[ file, "*" <> patt],
            name = StringDrop[ name, -StringLength[ patt]];
            fileOut = name;
            If[ autoName,
                fileOut = createTempFile[ "Temp", fileOut]];
            fileOut = fun[ file, fileOut]];
            {fileOut, name}
    ]


(* Useful to have the MIME type for the URL *)
FetchURLWithHeaders[ url_String, outFile_String:"", opts:OptionsPattern[]] :=
    FetchURLInternal[url, outFile, opts]


FetchURL[ url_String, outFile_String:"", opts:OptionsPattern[]] :=
    If[ListQ[#], First[#], #]& @ FetchURLInternal[url, outFile, opts]


FetchURLInternal[ url_String, outFileIn_String, opts:OptionsPattern[]] :=
    Module[ {epil, timeout, file, headers, outFile = outFileIn, name = outFileIn, autoName = False,
              requestparameters, requestheaderfields, requestmethod, serverAuth, verifyPeer, username, password},

        {epil, timeout, requestmethod, requestparameters, requestheaderfields, verifyPeer, serverAuth, username, password} =
            OptionValue[{FileFilters, "Timeout", "RequestMethod", "RequestParameters",
                         "RequestHeaderFields", "VerifyPeer", "ServerAuthentication", "Username", "Password"}];

        (* error checking *)
        If[requestmethod =!= "POST" && requestmethod =!= "GET",
            Message[Import::erropts, requestmethod, "RequestMethod"];
            Return[$Failed]
        ];
        If[Not@MatchQ[requestheaderfields, {((Rule|RuleDelayed)[_String,_String])...}],
            Message[Import::erropts, requestheaderfields, "RequestHeaderFields"];
            Return[$Failed]
        ];
        If[Not@MatchQ[requestparameters, {((Rule|RuleDelayed)[_String,_String])...}],
            Message[Import::erropts, requestparameters, "RequestParameters"];
            Return[$Failed]
        ];

        If[ !StringQ[ outFile] || StringLength[outFile] < 1,
            autoName = True;
            name = getFileName[ url];
            outFile = createTempFile[ "Temp", name]
        ];

        Catch[
            Switch[requestmethod,
                "GET",
                        {file, headers} = iFetchURL["GET", url, outFile, timeout, TrueQ[verifyPeer] || TrueQ[serverAuth],
                                                      requestheaderfields, requestparameters, username, password];
                        If[ ValidFilter[ epil],
                            Map[ ({file, name} = ApplyFilter[ #, file, name, autoName])&, epil]
                        ];
                        {file, headers},

                "POST",
                        {file, headers} = iFetchURL["POST", url, outFile, timeout, TrueQ[verifyPeer] || TrueQ[serverAuth],
                                                      requestheaderfields, requestparameters, username, password];
                        If[ ValidFilter[ epil],
                            Map[ ({file, name} = ApplyFilter[ #, file, name, autoName])&, epil]
                        ];
                        {file, headers},
                _,
                        Throw[Null, URLToolsException[$Failed, Null]]
            ]
            ,
            _URLToolsException,
            processException
        ]

    ]

validateOption["ServerAuthentication" , value : (Automatic | True | False ) ] := True
validateOption[___] := False

(*Returns True if the pathname begins with a relative path metacharacter.
  This is copied from Converteres.m*)
beginsRelativeMetaCharQ[str_String] := StringMatchQ[str, "."] ||
    StringMatchQ[str, ".."] ||
    StringMatchQ[str, ToFileName[{"."}, "*"]] ||
    StringMatchQ[str, ToFileName[{".."},"*"]] ||
    ($SystemID === "Windows" && $Language === "Japanese" && StringTake[str, {2}] != ":") (*hack to make sure full pathname on Japanese Windows*)

beginsRelativeMetaCharQ[___] := False


(* set a file to working directory unless absolute path is specified *)

setToWorkingDirectory[file_] := Module[
  {ffile = file},
  If[beginsRelativeMetaCharQ[ffile] || DirectoryName[ffile] === "",
   ffile = ToFileName[{Directory[]}, ffile]];
  ffile]


(* URLInformation has been eliminated because it was apparently unused, and not convenient to implement. *)

URLInformation[url_String] := Null /; Message[URLInformation::obs]


iFetchURL[method:("GET" | "POST"), url_String, outFile1_String, timeout_, verifyPeer_,
             requestheaderfields_List, requestparameters_List, username:(_String | None), password:(_String | None)] :=
    Module[{outFile = outFile1, requestHeaders, requestParams, fetchResult,
              statusCode, bytes, outstrm, resultHeaders, responseElements},

        If[!PacletManager`$AllowInternet, Throw[Null, URLToolsException[DISALLOWINTERNET, Null]]];

        (* make the file in the Mathematica working directory Directory[],
        unless the user specified another directory*)
        outFile = setToWorkingDirectory[outFile];

        (* CURLLink code takes only Rule, not RuleDelayed, so fix the unlikely case where a user used RuleDelayed. *)
        requestHeaders = requestheaderfields /. RuleDelayed->Rule;
        requestParams = requestparameters /. RuleDelayed->Rule;
        
        (* Headers as a result type are only relevant for HTTP, and CURLLink will give errors in they
           are requested in FTP, so handle those two cases differently.
        *)
        responseElements = If[StringMatchQ[url, "http*", IgnoreCase->True], {"StatusCode", "ContentData", "Headers"}, {"StatusCode", "ContentData"}];

        fetchResult = URLFetch[url, responseElements, 
                                 "Method" -> method, "VerifyPeer" -> TrueQ[verifyPeer],
                                   "Headers" -> requestHeaders, "Parameters" -> requestParams,
                                     "Username" -> If[StringQ[username], username, ""],
                                       "Password" -> If[StringQ[password], password, ""]];
        (* I think it will always be a list of three elements or $Failed. *)
        Which[
            Length[fetchResult] == 3,
                (* http *)
                {statusCode, bytes, resultHeaders} = fetchResult;
                If[!(200 <= statusCode < 300),
                    Throw[Null, URLToolsException[BADSTATUSCODE, {url, statusCode}]]
                ],
            Length[fetchResult] == 2,
                (* ftp *)
                {statusCode, bytes} = fetchResult;
                (* Not meaningful for ftp, but supply a dummy value, which will be ignored. *)
                resultHeaders = "";
                If[!(200 <= statusCode < 300),
                    Throw[Null, URLToolsException[BADSTATUSCODE, {url, statusCode}]]
                ],
            True,
                (* Message has been issued by URLFetch. For example, could not locate server. *)
                Throw[Null, URLToolsException[BADCONNECTION, url]]
        ];

        outstrm = OpenWrite[outFile, BinaryFormat->True];
        BinaryWrite[outstrm, bytes, "Byte"];
        Close[outstrm];

        {outFile, resultHeaders}
    ]
    

(*
 Implementation of UncompressGZIPFile.
*)

UncompressGZIPFile[ file_, outFile_:Null] :=
        Catch[ If[ outFile === Null, iUncompressGZIPFile[ file], iUncompressGZIPFile[ file, outFile]],
            _URLToolsException,
            processException]


iUncompressGZIPFile[ file_] := iUncompressGZIPFile[ file, createTempFile[ "Temp"]]


iUncompressGZIPFile[file2_, outFile_] := JavaBlock[
Internal`DeactivateMessages[
    Module[
          {t, fis, gzis, fos, totalBytes = 0, buf, outFile2 = outFile, file = file2},
        InstallJava[];
        If[!StringQ[ outFile2],
                Throw[Null, URLToolsException[ NOTSTRING, outFile2]]
            ];
        If[!StringQ[file],
                Throw[Null, URLToolsException[ NOTSTRING, file]]
            ];

      (* make the file in the Mathemnatica working directory Directory[],
         unless the user specified another directory*)
      outFile2 = setToWorkingDirectory[outFile2];
      file = setToWorkingDirectory[file];

          t = Timing[
            fis = JavaNew["java.io.BufferedInputStream",
                JavaNew["java.io.FileInputStream", file]];
            If[fis === $Failed,
                Throw[Null, URLToolsException[ BADFILE, file]]
            ];

            gzis = JavaNew["java.util.zip.GZIPInputStream", fis];
            fos = JavaNew["java.io.BufferedOutputStream",
                JavaNew["java.io.FileOutputStream", outFile2]];
            buf = JavaNew["[B", 16*1024];

            While[(numRead = gzis@read[buf]) > 0,
                totalBytes += numRead;
                If[DEBUG, Print[totalBytes, " bytes uncompressed and written"]];
                fos@write[buf, 0, numRead]
            ];
            fis@close[];
              gzis@close[];
              fos@close[];
        ];
        If [MONITOR,
            Print["uncompress time = ", t[[1]], " bytes = ", totalBytes];];
        outFile2
    ]]];



End[]; (* end private *)
EndPackage[];

(* :Title: Unzip.m *)

(* :Author:
        Todd Gayley
        tgayley@wolfram.com
*)

(* :Package Version: 3.0 *)

(* :Mathematica Version: 9.0 *)

(* :Copyright: Mathematica source code (c) 1999-2019, Wolfram Research, Inc. All rights reserved. *)

(* :Discussion: This file is a component of the PacletManager Mathematica source code. *)


(* Zip archive functionality. At the moment, only extraction is supported. This code could be put into
   Packer.m, but because it is of more general use I am splitting it out into a separate file that exposes
   a general API.

   LibraryLink is used for a small DLL that links against zlib and includes some code from the minizip app
   that is bundled with zlib.

   Import can already handle some zipfile-related functionality based on J/Link (and also UnpackPaclet),
   but the implementation here is motivated by a requirement to avoid Java.
*)

Unprotect[ZipExtractArchive, ZipGetFile]


(* This function would perhaps be better named Unzip, but I envision a set of Zip-related functions,
   and in that case it makes sense for them to all begin with "Zip".
*)
ZipExtractArchive::usage = "ZipExtractArchive[\"zipfile\", \"destdir\"]"
ZipGetFile::usage = "ZipGetFile[\"zipfile\", \"requestedFile\"]"


Begin["`Package`"]


End[]  (* `Package` *)



(* Current context will be PacletManager`. *)

Begin["`Zip`Private`"]


Options[ZipExtractArchive] = {"FilenameEncoding" -> Automatic, "ArchiveEncoding" -> Automatic, 
                                  "Overwrite" -> False, Verbose -> False}


ZipExtractArchive::notfound = "Could not find specified archive file `1`."
ZipExtractArchive::destdir = "Destination directory `1` does not exist."
ZipExtractArchive::ziperr = "Error reading zip file `1`."
ZipExtractArchive::llink = "Could not load the Wolfram Library that implements Zip functionality."
ZipExtractArchive::overwrite = "The destination file `1` already exists. Halting the extraction. Use the \"Overwrite\"->True option to allow existing files to be overwritten during Zip file extraction."
ZipExtractArchive::fencoding = "A character in the file name of the Zip archive `1` could not be decoded using the specified character encoding. You can use the \"FilenameEncoding\" option to specify a different encoding."
ZipExtractArchive::encoding = "A character in a file name within the Zip archive `1` could not be decoded using the default character encoding. You can use the \"FilenameEncoding\" option to specify a different encoding. If the archive was created on Windows, try \"IBM-850\"."

(*

   If called without a destDir spec, defaults to the dir in which the archive resides.
   
   The system function ExtractArchive returns the list of files extracted, and that is probably the
   only generally sensible value, but for the needs of the PacletManager, what is most convenient is 
   the path to the top-level dir created during the extraction. In general, such a dir need not exist,
   as for example with a zip archive built from just a handful of separate files. But the PM always
   has a top-level dir containing the unzipped paclet.
*)

(* TODO: Consider unloading DLL after each call to ZipExtractArchive. The unload/reload is very fast. *)

(* Returns the full path to the first entry in the archive, unpacked. Or $Failed if there was a problem. *)

ZipExtractArchive[zipFile_String, opts:OptionsPattern[]] := ZipExtractArchive[zipFile, Automatic, opts]

ZipExtractArchive[zipFile_String, destDir:(_String | Automatic), OptionsPattern[]] :=
    Module[{unzFilePtr, filenameEncoding, archiveEncoding, verbose, overwrite,  wasZipError, wasOverwriteError,
               moreEntries, fileInfo, filePath, isDir, fullDestDir, parentDir, fullNewFilePath, buffer,
                   numBytesRead, strm, filenameBytes, resultPath, isUnixLikeOS},
        If[!loadLibraryFunctions[],
            Message[ZipExtractArchive::llink];
            Return[$Failed]
        ];
        If[!FileExistsQ[zipFile],
            Message[ZipExtractArchive::notfound, zipFile];
            Return[$Failed]
        ];
        Which[
            destDir === Automatic,
                fullDestDir = DirectoryName[ExpandFileName[zipFile]],
            DirectoryQ[destDir],
                fullDestDir = ExpandFileName[destDir],
            True,
                Message[ZipExtractArchive::destdir, destDir];
                Return[$Failed]
        ];

        {filenameEncoding, archiveEncoding, overwrite, verbose} = 
                 OptionValue[{"FilenameEncoding", "ArchiveEncoding", "Overwrite", Verbose}];
        If[!StringQ[filenameEncoding],
            filenameEncoding = If[StringMatchQ[$SystemID, "Windows*"], "Unicode", "UTF8"]
        ];
        (* The Pacletmanager's PackPaclet function, and Mathematica's own CreateArchive function, use Java and
           therefore they get filename entries encoded using UTF8, so this is our default.
        *)
        If[!StringQ[archiveEncoding],
            archiveEncoding = "UTF8"
        ];
        
        isUnixLikeOS = MemberQ[{"Linux", "Linux-x86-64", "MacOSX-x86", "MacOSX-x86-64"}, $SystemID];
        
        If[verbose,
            Print["Extracting archive ", ExpandFileName[zipFile]]
        ];

        (* Note that we don't use the specified filenameEncoding option for the archive filename itself. 
           Thst option is for entries within the archive, which will have differnet encodings depending on what
           zip program created it. The filename encoding at the OS level is something else, and we just hard-code
           it here to be the platform defaults.
        *)
        filenameBytes = ToCharacterCode[ExpandFileName[zipFile], filenameEncoding];
        If[!MatchQ[filenameBytes, {__Integer}],
            (* This should be quite rare, as the use of the FilenameEncoding option is rare. An example of this
               error is if caller specifies "ASCII" and the filename has non-ASCII chars in it, then ToCharacterCode
               can return None for some chars.
            *)
            Message[ZipExtractArchive::fencoding, zipFile];
            Return[$Failed]
        ];
        unzFilePtr = unzOpen[filenameBytes];
        If[unzFilePtr === 0,
            Message[ZipExtractArchive::ziperr, zipFile];
            Return[$Failed]
        ];
        
        try[
            buffer = Table[0, {8192}];  (* Arbitrary buffer size. *)
            wasZipError = False;
            wasOverwriteError = False;
            moreEntries = True;
            resultPath = $Failed;
            While[moreEntries && !wasZipError && !wasOverwriteError,
                fileInfo = unzGetCurrentFileInfo[unzFilePtr];
                (* This is {error code, filename bytes, uncompressed size, dos-style date, external file attrs, extra field data (Null or first 3 bytes)}. *)
                If[MatchQ[fileInfo, {0, _List, _Integer, _Integer, _Integer, _List | Null}],
                    filePath = decodeEntryName[fileInfo, archiveEncoding];
                    If[filePath === $Failed,
                        (* Issue a message and keep going with the next entry. *)
                        Message[ZipExtractArchive::encoding, zipFile],
                    (* else *)
                        If[verbose, Print["   ... " <> filePath]];
                        isDir = StringMatchQ[filePath, __ ~~ "\\"] || StringMatchQ[filePath, __ ~~ "/"];
                        fullNewFilePath = FileNameJoin[{fullDestDir, filePath}];
                        (* Try to capture the path to the top-level dir in the archive, which is used as
                           the return value. As discussed in the comments at the top of this function, this
                           is not a meaningful result for all archives, but it is for ones created by the PM.
                           If you zip up a dir and all its contents, there is no entry found that corresponds
                           to the top directory. Instead, the first entry
                           found is a file or subdir in the top-level dir. But if the archive is a dir, we want that
                           top-level dir to be the result returned from ZipExtractArchive. Thus the
                           FileNameTake[filePath, 1] below is used to get the name of the top-level dir.
                        *)
                        If[!StringQ[resultPath],
                            resultPath = FileNameJoin[{fullDestDir, FileNameTake[filePath, 1]}]
                        ];
                        If[isDir,
                            If[!DirectoryQ[fullNewFilePath],
                                CreateDirectory[fullNewFilePath, CreateIntermediateDirectories -> True]
                            ],
                        (* else *)
                            parentDir = DirectoryName[fullNewFilePath];
                            If[!DirectoryQ[parentDir],
                                CreateDirectory[parentDir, CreateIntermediateDirectories -> True]
                            ];
                            If[unzOpenCurrent[unzFilePtr] === 0,
                                numBytesRead = 0;
                                If[!FileExistsQ[fullNewFilePath] || TrueQ[overwrite],
                                    strm = OpenWrite[fullNewFilePath, BinaryFormat -> True];
                                    While[(numBytesRead = unzReadCurrent[unzFilePtr, buffer]) > 0,
                                        BinaryWrite[strm, Take[buffer, numBytesRead]];
                                    ];
                                    Close[strm];
                                    SetFileDate[fullNewFilePath, fileDateAsMDate[fileInfo]];
                                    (* The Java paclet-packing code inserts this magic string into the extra data field
                                       to indicate that this entry was an executable file. We want to restore that property.
                                    *)
                                    If[isUnixLikeOS && Last[fileInfo] == ToCharacterCode["PMx"],
                                        Quiet[Run["chmod a+x '" <> fullNewFilePath <> "'"]]
                                    ],
                               (* else *)
                                    (* Will not be overwriting any existing files. Bail out immediately. *)
                                    Message[ZipExtractArchive::overwrite, fullNewFilePath];
                                    wasOverwriteError = True
                                ];
                                unzCloseCurrent[unzFilePtr];
                                wasZipError = numBytesRead < 0,
                            (* else *)
                                wasZipError = True
                            ]
                        ]
                    ],
                (* else *)
                    (* unzGetCurrentFileInfo returned an error. *)
                    wasZipError = True
                ];
                If[!wasZipError, moreEntries = unzGoToNext[unzFilePtr] === 0]
            ],
        (* finally *)
            unzClose[unzFilePtr]
        ];
        Which[
            wasZipError,
                Message[ZipExtractArchive::ziperr, zipFile];
                $Failed,
            wasOverwriteError,
                (* Message already issued. *)
                $Failed,
            True,
                resultPath
        ]
    ]


Options[ZipGetFile] = {"FilenameEncoding" -> Automatic, "ArchiveEncoding" -> Automatic}

ZipGetFile::notfound = ZipExtractArchive::notfound
ZipGetFile::noentry = "Requested file `1` was not found in Zip file `2`."
ZipGetFile::ziperr = "Error reading Zip file `1`."
ZipGetFile::llink = "Could not load the Wolfram Library that implements Zip functionality."
ZipGetFile::encoding = ZipExtractArchive::encoding
ZipGetFile::fencoding = ZipExtractArchive::fencoding


(*
    If you sepcify a path in the requested entry, as in "foo/bar.txt", it looks for exactly that entry,
    but if you specify only a filename, as in "bar.txt", it looks for the file anywhere in the archive.
*)

ZipGetFile[zipFile_String, requestedFileName_String, OptionsPattern[]] :=
    Module[{unzFilePtr, wasZipError, foundFile, moreEntries, fileInfo, filenameEncoding, filenameBytes,
               archiveEncoding, filePath, fileNameParts, requestedFileHasPath, uncompressedSize, buf, numBytesRead},
        If[!loadLibraryFunctions[],
            Message[ZipGetFile::llink];
            Return[$Failed]
        ];
        If[!FileExistsQ[zipFile],
            Message[ZipGetFile::notfound, zipFile];
            Return[$Failed]
        ];
        {filenameEncoding, archiveEncoding} = OptionValue[{"FilenameEncoding", "ArchiveEncoding"}];
        If[!StringQ[filenameEncoding],
            filenameEncoding = If[StringMatchQ[$SystemID, "Windows*"], "Unicode", "UTF8"]
        ];
        (* The Pacletmanager's PackPaclet function, and Mathematica's own CreateArchive function, use Java and
           therefore they get filename entries encoded using UTF8, so this is our default.
        *)
        If[!StringQ[archiveEncoding],
            archiveEncoding = "UTF8"
        ];
        (* Don't use FileNameSplit here, as Unix version cannot handle Windows-style separators. *)
        fileNameParts = StringSplit[requestedFileName, {"\\", "/"}];
        requestedFileHasPath = Length[fileNameParts] > 1;

        filenameBytes = ToCharacterCode[ExpandFileName[zipFile], filenameEncoding];
        If[!MatchQ[filenameBytes, {__Integer}],
            (* This should be quite rare, as the use of the FilenameEncoding option is rare. An example of this
               error is if caller specifies "ASCII" and the filename has non-ASCII chars in it, then ToCharacterCode
               can return None for some chars.
            *)
            Message[ZipGetFile::fencoding, zipFile];
            Return[$Failed]
        ];
        unzFilePtr = unzOpen[filenameBytes];
        If[unzFilePtr === 0,
            Message[ZipGetFile::ziperr, zipFile];
            Return[$Failed]
        ];
        try[
            wasZipError = False;
            foundFile = False;
            moreEntries = True;
            While[moreEntries && !foundFile && !wasZipError,
                fileInfo = unzGetCurrentFileInfo[unzFilePtr];
                (* This is {error code, filename bytes, uncompressed size, dos-style date, external file attrs, extra field data (Null or first 3 bytes)}. *)
                If[MatchQ[fileInfo, {0, _List, _Integer, _Integer, _Integer, _List | Null}],
                    filePath = decodeEntryName[fileInfo, archiveEncoding];
                    If[filePath === $Failed,
                        (* Issue a message and keep going with the next entry. *)
                        Message[ZipGetFile::encoding, zipFile],
                    (* else *)
                        (* If requested file was specified with separators, then it must match exactly to the
                           full path within the archive, but if it is just a filename with no path, then look for
                           it anywhere.
                        *)
                        If[fileNameParts == FileNameSplit[filePath] || !requestedFileHasPath && FileNameTake[filePath] == requestedFileName,
                            (* This is the requested file. *)
                            foundFile = True;
                            uncompressedSize = fileInfo[[3]];
                            If[unzOpenCurrent[unzFilePtr] === 0,
                                buf = Table[0, {Max[1000, uncompressedSize]}];
                                numBytesRead = unzReadCurrent[unzFilePtr, buf];
                                unzCloseCurrent[unzFilePtr];
                                wasZipError = numBytesRead =!= uncompressedSize,
                            (* else *)
                                wasZipError = True
                            ]
                        ]
                    ],
                (* else *)
                    (* unzGetCurrentFileInfo returned an error. *)
                    wasZipError = True
                ];
                If[!wasZipError, moreEntries = unzGoToNext[unzFilePtr] === 0]
            ],
        (* finally *)
            unzClose[unzFilePtr]
        ];
        Which[
            wasZipError,
                Message[ZipGetFile::ziperr, zipFile];
                $Failed,
            !foundFile,
                Message[ZipGetFile::noentry, requestedFileName, zipFile];
                $Failed,
            True,
                Take[buf, numBytesRead]
        ]
    ]


(* In addition to decoding the entry names using the given char encoding, this also converts \ separators to /.
   It is convenient to force all separators to Unix-style, as Mathematica's filename operations on Unix cannot handle
   \-style separators, whereas the functions on Windows _can_ handle /-style.
*)
decodeEntryName[fileInfo_List, encoding_String] :=
    Module[{filenameBytes},
        filenameBytes = fileInfo[[2]];
        Quiet[
            Check[
                StringReplace[FromCharacterCode[filenameBytes, encoding], "\\" -> "/"],
                (* On message: *)
                $Failed
            ]
        ]
    ]


(* Took this algorithm from unzip.c in zlib/contrib/minizip. It converts the "DOS-style date" recorded
   in the archive into an M-style date list.
*)
fileDateAsMDate[fileInfo_List] :=
    Module[{dosDate, d, year, month, day, hour, min, sec},
        dosDate = fileInfo[[4]];
        d = BitShiftRight[dosDate, 16];
        year = BitAnd[d, 16^^FE00]/16^^200 + 1980;
        month =  BitAnd[d, 16^^1E0]/16^^20;
        day = BitAnd[d, 16^^1f];
        hour = BitAnd[dosDate, 16^^F800]/16^^800;
        min =  BitAnd[dosDate, 16^^7E0]/16^^20;
        sec =  2 BitAnd[dosDate, 16^^1f];
        {year, month, day, hour, min, sec}
    ]


(**********************************  loadLibraryFunctions  *********************************)

libPath =
    FileNameJoin[{
        $pmDir,
        "LibraryResources",
        $SystemID
    }]

If[FreeQ[$LibraryPath, libPath],
    PrependTo[$LibraryPath, libPath]
]


loadLibraryFunctions[] :=
    Module[{libFile},
        libFile = FindLibrary["WRIunzip"];
        If[StringQ[libFile],
            Quiet[
                If[StringMatchQ[$SystemID, "Windows*"],
                    unzOpen = LibraryFunctionLoad[libFile, "unzipOpenWide", {{Integer, 1, "Constant"}}, Integer],
                (* else *)
                    unzOpen = LibraryFunctionLoad[libFile, "unzipOpen", {{Integer, 1, "Constant"}}, Integer]
                ];
                unzClose = LibraryFunctionLoad[libFile, "unzipClose", {Integer}, Integer];
                unzGoToFirst = LibraryFunctionLoad[libFile, "unzipGoToFirstFile", {Integer}, Integer];
                unzGoToNext = LibraryFunctionLoad[libFile, "unzipGoToNextFile", {Integer}, Integer];
                unzLocate = LibraryFunctionLoad[libFile, "unzipLocateFile", {Integer, {Integer, 1, "Constant"}}, Integer];
                unzGetCurrentFileInfo = LibraryFunctionLoad[libFile, "unzipGetCurrentFileInfo", LinkObject, LinkObject];
                unzOpenCurrent = LibraryFunctionLoad[libFile, "unzipOpenCurrentFile", {Integer}, Integer];
                unzCloseCurrent = LibraryFunctionLoad[libFile, "unzipCloseCurrentFile", {Integer}, Integer];
                unzReadCurrent = LibraryFunctionLoad[libFile, "unzipReadCurrentFile", {Integer, {Integer, 1, "Shared"}}, Integer]
            ];
            !MemberQ[{unzOpen, unzClose, unzGoToFirst, unzGoToNext, unzLocate, unzGetCurrentFileInfo,
                        unzOpenCurrent, unzCloseCurrent, unzReadCurrent}, $Failed],
        (* else *)
            (* Library could not be found; message issued later. *)
            False
        ]
    ]


End[]


SetAttributes[{ZipExtractArchive, ZipGetFile}, {Protected, ReadProtected}]
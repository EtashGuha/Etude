(* :Title: Packer.m *)

(* :Author:
        Todd Gayley
        tgayley@wolfram.com
*)

(* :Package Version: 3.0 *)

(* :Mathematica Version: 6.0 *)

(* :Copyright: Mathematica source code (c) 1999-2019, Wolfram Research, Inc. All rights reserved. *)

(* :Discussion: This file is a component of the PacletManager Mathematica source code. *)


(* Functionality for packing component files and dirs into a .paclet file, and unpacking. *)


PackPaclet::usage = "PackPaclet[\"dir\"] creates a .paclet file from the contents of dir. The directory must contain a PacletInfo.m file."
UnpackPaclet::usage = "UnpackPaclet[\"file\"] unpacks the specified .paclet file into the same directory in which the .paclet file is located. UnpackPaclet[\"file\", \"dir\"] unpacks the file into the specified directory."


Begin["`Package`"]


End[]  (* `Package` *)



(* Current context will be PacletManager`. *)

Begin["`Packer`Private`"]


(* For now, perhaps forever, packing operations require Java. Only unpacking, which needs to be
   done by all clients, is moved into a WolframLibrary.
*)

Options[PackPaclet] = {Verbose -> False}


PackPaclet::notfound = "Could not find specified file or directory `1`."

(*
    PackPaclet creates a single .paclet file out of the component files
    and dirs that make up a paclet. The name of the .paclet file is
    determined automatically from info in the PacletInfo.m file.
    Returns the full path to the created .paclet file.
    
    Note that PackPaclet uses Java. On platforms where Java is not supported,
    this function is not available.

    PackPaclet["dir"]
        dir must contain a PacletInfo.m file. Entire contents of dir
        including subdirs (but not the dir itself) will be packed into
        the paclet file. Paclet file will be placed parallel to dir.

    PackPaclet["dir", "destDir"]
        Same as above, except .paclet file will be placed into destDir.

    PackPaclet[{"fileOrDir1", "fileOrDir2", ...}, "destDir"]
        Paclet will contain the given files and dirs. One of the supplied
        files must be a PacletInfo.m file, or a dir that contains a
        PacletInfo.m file at its top level.

        In the above form, each "fileOrDir1" can be a pair {"root", "child"}.
        See the PacletPacker JavaDocs for more info.
*)

PackPaclet[dir_String] :=
    Module[{fullPath = getFullPath[dir]},
        If[StringQ[fullPath],
            PackPaclet[{dir}, DirectoryName[fullPath]],
        (* else *)
            Message[PackPaclet::notfound, dir];
            $Failed
        ]
    ]

PackPaclet[dir_String, destDir_String] := PackPaclet[{dir}, destDir]

PackPaclet[components:{(_String | {_String, _String})..}, destDir:_String:Directory[]] :=
    (
        Needs["JLink`"];
        JLink`JavaBlock[
            Module[{fullPairs, fullPaths, packer, pacletFile},
                JLink`InstallJava[];
                fullPairs = If[StringQ[#], {#, ""}, #]& /@ components;
                fullPaths = {getFullPath[#1], #2}& @@@ fullPairs;
                If[!MatchQ[fullPaths, {{_String, _String}..}],
                    Message[PackPaclet::notfound, Last[#]]& /@
                        Select[Thread[{First /@ fullPaths, First /@ fullPairs}], First[#] === Null &];
                    Return[$Failed]
                ];
                (* TODO: Verbose otpion. *)
                packer = JLink`JavaNew["com.wolfram.paclet.PacletPacker"];
                packer@addSourceLocation[##]& @@@ fullPaths;
                packer@setDestination[destDir];
                pacletFile = packer@pack[];
                If[JLink`JavaObjectQ[pacletFile],
                    pacletFile@getAbsolutePath[],
                (* else *)
                    (* Rely on error message from pack() method. *)
                    $Failed
                ]
            ]
        ]
    )



(* UnpackPaclet uncompresses the specified .paclet file into the specified directory (if it is Automatic, then the
   dest dir becomes the dir in which the .paclet file resides).

   The return value is the top-level dir of the unpacked paclet.
*)

Options[UnpackPaclet] = {Verbose -> False}

UnpackPaclet::notfound = "Could not find specified paclet file `1`."
UnpackPaclet::destdir = "Destination directory `1` does not exist."


UnpackPaclet[pacletFile_String] := UnpackPaclet[pacletFile, Automatic]

UnpackPaclet[pacletFile_String, destDir:(_String | Automatic)] :=
    Module[{fullDestPath},
        If[!FileExistsQ[pacletFile],
            Message[UnpackPaclet::notfound, pacletFile];
            Return[$Failed]
        ];
        If[destDir =!= Automatic && !DirectoryQ[destDir],
            Message[UnpackPaclet::destdir, destDir];
            Return[$Failed]
        ];
        If[destDir === Automatic,
            fullDestPath = DirectoryName[getFullPath[pacletFile]],
        (* else *)
            fullDestPath = getFullPath[destDir]
        ];
        ZipExtractArchive[pacletFile, fullDestPath, Verbose->OptionValue[Verbose]]
    ]


(* Returns the full path to the file or dir, or Null if it could not be found.
   fileOrDir can be a full or partial path.
*)
getFullPath[fileOrDir_String] :=
    Module[{f},
        Scan[
            Function[dir,
                f = ToFileName[{dir}, fileOrDir];
                If[FileType[f] =!= None, Return[ExpandFileName[f]]]
            ],
            (* "" is first entry, to handle case where fileOrDir is a full path.
               Next check Directory[] before going to $Path. This allows people
               to call SetDirectory[] and be sure that files will be found
               in current dir first (current dir is NOT first on $Path).
            *)
            {"", Directory[]} ~Join~ $Path
        ]
    ]


End[]
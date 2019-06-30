(* ::Package:: *)

BeginPackage["EmbedCode`"]


EmbedCode`Common`iEmbedCode

Begin["`Private`"]

(* The dir that holds modules for each supported embedding environment ("Java", "Python", etc. *)
$environmentsDir = FileNameJoin[{DirectoryName[$InputFileName], "EmbeddingEnvironments"}]


(****************************  Embed Code  ******************************)

(*  TODOs:
        authentication/token system for calls
*)

Unprotect[EmbedCode]

EmbedCode::noenv = "Embedding environment `1` is not supported."
EmbedCode::noembed = "The given object `1` at `2` cannot be embedded in `3`."
EmbedCode::badsig = "Invalid ExternalTypeSignature specification: `1`."
EmbedCode::baddeploy = "The CloudDeploy operation failed, therefore EmbedCode cannot operate on the object."
EmbedCode::arg = "EmbedCode for the environment `1` does not operate on arguments of that type."
EmbedCode::argcount = "The ExternalTypeSignature option does not specify the correct number of required parameters to the APIFunction."
EmbedCode::invsize = "The given ImageSize `1` is not Automatic or a list of two positive integers."
EmbedCode::invsav = "The code can only be saved to an non-existing directory or an archive filename."

Options[EmbedCode] = {ExternalTypeSignature -> Automatic, ExternalFunctionName -> Automatic, ExternalOptions -> <||>,
                       Permissions -> Automatic, ImageSize -> Automatic}
                       
archiveFormats = {"zip", "tar", "tgz", "tb2", "tbz2"};                       
$archiveFormats = Join[archiveFormats, ToUpperCase /@ archiveFormats]                      

EmbedCode[expr_, opts:OptionsPattern[]] := EmbedCode[expr, "HTML", opts]

EmbedCode[expr_?isAutoDeploy, env_String,opts:OptionsPattern[]] :=
	EmbedCode[expr, env, "", opts]

EmbedCode[expr_?isAutoDeploy, env_String, saveFile_String, opts:OptionsPattern[]] :=
    Module[{perms, obj},
    	If[saveFile =!= "" && DirectoryQ[saveFile] && Not[MemberQ[$archiveFormats, FileExtension[saveFile]]],
        	Message[EmbedCode::invsav];
            Return[$Failed]        	
        ];
        perms = OptionValue[Permissions];
        If[perms === Automatic, perms = "Public"];
        obj = CloudDeploy[expr, Permissions -> perms];
        If[Head[obj] === CloudObject,
            EmbedCode[obj, env, saveFile, DeleteCases[Flatten[{opts}], "Permissions" -> _]],
        (* else *)
            Message[EmbedCode::baddeploy];
            $Failed
        ]
    ]

(* Allow URL arg instead of CloudObject *)
EmbedCode[URL[url_], rest___] /; StringMatchQ[url, "http*"] := EmbedCode[CloudObject[url], rest]

EmbedCode[obj:CloudObject[uri_], env_String, opts:OptionsPattern[]] :=
	EmbedCode[obj, env, "", opts]    

EmbedCode[obj:CloudObject[uri_], env_String, saveFile_String, opts:OptionsPattern[]] :=
    Module[{cloudObjectContents, sig, perms, canonicalEnvName, availableEnvironFiles, matchingEnvironFile, commonEnvironFile, assoc},
        If[saveFile =!= "" && DirectoryQ[saveFile] && Not[MemberQ[$archiveFormats, FileExtension[saveFile]]],
        	Message[EmbedCode::invsav];
            Return[$Failed]        	
        ];
        (* For HTML (and other non-APIFunction embeddings in the future), the CloudObject will not be an APIFunction, so we don't want to Get it.
           Instead, we merely want to check that it exists and trigger a message if it does not. That's what the call to CloudObjectInformation is for.
           We assign a dummy value of Null to cloudObjectContents.
        *)
        If[canonicalizeEnvName[env] === "html",
            cloudObjectContents = Null;
            If[CloudObjectInformation[obj] === $Failed,
                Return[$Failed]
            ],
        (* else *)
            cloudObjectContents = Get[obj];
            If[cloudObjectContents === $Failed,
                (* CloudObject::notfound will already have been issued by the Get call above. *)
                Return[$Failed]
            ]
        ];
        {sig, perms} = OptionValue[{ExternalTypeSignature, Permissions}];
        If[perms =!= Automatic,
            SetOptions[obj, Permissions -> perms]
        ];
        (* Sanity check the TypeSignature value. Here are the meanings of various values:
             Automatic      -Detemine args and types from func param spec; return type is specific to language (e.g., byte[] in Java)
             {Automatic, Automatic}    -Same as Automatic
             {Automatic, _String}      -Determine args and types from func param spec; return type is as given
             {{}, Automatic}   -Zero args; return type is specific to language (e.g., byte[] in Java)
             {{(_String | {_String -> _String})...}, ret_String}  -standard spec with exact types and ret type
         *)
        If[!MatchQ[sig, Automatic | {{(_String | {_String -> _String})...} | Automatic, _String | Automatic}],
            Message[EmbedCode::badsig, sig];
            Return[$Failed]
        ];
        (* Lookup and load embedding environment code. Note that we use case-insensitive lookup, so users don't have to get
           the capitalization of their desired language/environment correct.
           We look for two .m files:
           1) one named after the requested environment (e.g., Java.m, Java-Jersey.m, Python.m)
           2) A common one shared by all variants of the same language (e.g., Java-Common.m). This allows developers to
              put code shared by all flavors of a language in a common place.
           Both files are loaded, if found.
        *)
        canonicalEnvName = canonicalizeEnvName[env];
        availableEnvironFiles = FileNames["*.m" | "*.wl", $environmentsDir];
        matchingEnvironFile = SelectFirst[availableEnvironFiles, (ToLowerCase[FileBaseName[FileNameTake[#]]] == canonicalEnvName)&];
        If[StringQ[matchingEnvironFile],
            (* The files are small, so just go ahead and load them every time. *)
            commonEnvironFile = SelectFirst[availableEnvironFiles,
                    (ToLowerCase[FileBaseName[FileNameTake[#]]] == StringJoin[ToLowerCase[First[StringSplit[canonicalEnvName, "-"]]], "-common"])&];
            (* If present, load the XXX-Common.m one first. *)
            If[StringQ[commonEnvironFile], Get[commonEnvironFile]];
            Get[matchingEnvironFile];
            (* Each language implementation file will define a rule for EmbedCode`Common`iEmbedCode["lang", ...]. Implementations can call Throw to
               bail out quickly. Also, some functions defined in this file (e.g., createFullArgumentSpec) can also Throw to here.
            *)
            assoc =
                Catch[
                    EmbedCode`Common`iEmbedCode[canonicalEnvName, cloudObjectContents, uri, Flatten[{opts} ~Append~ ("OriginalEnvironmentName" -> env)]]
                ];
            (* Rely on the language implementation file to have issued a meaningful message if $Failed. *)
			If[Head[assoc] === Association,
				If[saveFile === "",
					EmbeddingObject[assoc ~Append~ ("CloudObject" -> obj)],
					saveCode[assoc, saveFile]
				]
				,
            (* else *)
                $Failed
            ],
        (* else *)
            Message[EmbedCode::noenv, env];
            $Failed
        ]
    ]
    
EmbedCode[obj_Databin, env_String, opts:OptionsPattern[]] :=
	EmbedCode[obj, env, "",  opts]    

EmbedCode[obj_Databin, env_String, saveFile_String, opts:OptionsPattern[]] :=
	Module[{canonicalEnvName, availableEnvironFiles, matchingEnvironFile, commonEnvironFile, assoc},
		If[saveFile =!= "" && DirectoryQ[saveFile] && Not[MemberQ[$archiveFormats, FileExtension[saveFile]]],
        	Message[EmbedCode::invsav];
            Return[$Failed]        	
        ];
		canonicalEnvName = canonicalizeEnvName[env];
        availableEnvironFiles = FileNames["*.m" | "*.wl", $environmentsDir];
        matchingEnvironFile = SelectFirst[availableEnvironFiles, (ToLowerCase[FileBaseName[FileNameTake[#]]] == canonicalEnvName)&];
        If[StringQ[matchingEnvironFile],
            (* The files are small, so just go ahead and load them every time. *)
            commonEnvironFile = SelectFirst[availableEnvironFiles,
                    (ToLowerCase[FileBaseName[FileNameTake[#]]] == StringJoin[ToLowerCase[First[StringSplit[canonicalEnvName, "-"]]], "-common"])&];
            (* If present, load the XXX-Common.m one first. *)
            If[StringQ[commonEnvironFile], Get[commonEnvironFile]];
            Get[matchingEnvironFile];
            (* Each language implementation file will define a rule for EmbedCode`Common`iEmbedCode["lang", ...]. Implementations can call Throw to
               bail out quickly.
            *)
            assoc =
                Catch[
                    EmbedCode`Common`iEmbedCode[canonicalEnvName, obj, Null, Flatten[{opts} ~Append~ ("OriginalEnvironmentName" -> env)]]
                ];
            (* Rely on the language implementation file to have issued a meaningful message if $Failed. *)
            If[Head[assoc] === Association,
                If[saveFile === "",
					EmbeddingObject[assoc ~Append~ ("CloudObject" -> obj)],
					saveCode[assoc, saveFile]
				],
            (* else *)
                $Failed
            ],
        (* else *)
            Message[EmbedCode::noenv, env];
            $Failed
        ]
	];

EmbedCode[_, env_String, OptionsPattern[]] := (Message[EmbedCode::arg, env]; $Failed)

SetAttributes[EmbedCode, {Protected, ReadProtected}]


(* Embedding environment names are already case-insensitive, but here we add alternate forms and spellings of some environments
   as a courtesy to users. Make sure that what is added here is all lowercase. The lhs of each rule is an incorrect-but-supported
   name, and the rhs is the true name. It is not intended that this list will grow large. Users are expected to use the correct name.
*)
$alternateEnvNames = {
    "cppvisualsutdio" -> "c++-visualstudio",
    "cpp-visualsutdio" -> "c++-visualstudio",
    "csharp" -> "c#"
}
canonicalizeEnvName[env_String] := ToLowerCase[env] /. $alternateEnvNames

(* This function determines whether an expression passed to EmbedCode is something that should have CloudDeploy
   called automatically on it. In other words, expressions, that are meaningful only when deployed as CloudObjects.
*)
isAutoDeploy[expr_] := MatchQ[Head[expr], APIFunction | FormFunction]

saveCode[assoc_Association, saveFile_String] := 
	Module[{codeSection, content, fileName, filePath, dirPath},
		codeSection = assoc["CodeSection"];
		content = codeSection["Content"];
		fileName = codeSection["Filename"];
		Which[ 
			MemberQ[$archiveFormats, FileExtension[saveFile]],
			If[FileExistsQ[saveFile], DeleteFile[saveFile]];
			dirPath = CreateDirectory[FileNameJoin[{$TemporaryDirectory, ToString[AbsoluteTime[DateString[]]], "Wolfram"}]];
			saveLibrary[assoc, dirPath];
			filePath = FileNameJoin[{dirPath, fileName}];
			BinaryWrite[filePath, content];
			Close[filePath];
			CreateArchive[dirPath, saveFile]
			,
			!DirectoryQ[saveFile],
			CreateDirectory[saveFile];
			saveLibrary[assoc, saveFile];
			filePath = FileNameJoin[{saveFile, fileName}];
			BinaryWrite[filePath, content];
			Close[filePath];
			saveFile
			,
			True,
			Message[EmbedCode::invsav];
            $Failed
		]
	]

saveLibrary[assoc_Association, saveFile_String] :=
	Module[{data, fileList, url, dest},
		If[hasItem[assoc, "FilesSection"]
			,
			data = assoc["FilesSection"];
			fileList = data["FileList"];
			If[Length[fileList] > 1
				,
				url = data["ZipDownloadURL"];
				dest = URLDownload[url, FileNameJoin[{saveFile, Last[URLParse[url, "Path"]]}]];
				ExtractArchive[dest, saveFile]
				,
				url = data["DownloadURL"];
				URLDownload[url, FileNameJoin[{saveFile, Last[URLParse[url, "Path"]]}]]
			]		
		]
	]
	

(****************  Utility functions for language-specific implementations  *****************)

(* These are functions that will be useful in some or all of the language-specific implementation files.
   They are collected here and defined in the EmbedCode`Common` context. Callers will use the fully-qualified
   names of these symbols to refer to them. The point of defining EmbedCode`Common` is to avoid having these
   symbols be in the current context, which is CloudObject`EmbedCode`Private`. I cannot bear to have symbols
   defined in a manifestly "private" context be used outside of the file that defines them.
*)

Begin["EmbedCode`Common`"]


Options[EmbedCode`Common`iEmbedCode] = Options[EmbedCode] ~Append~ ("OriginalEnvironmentName" -> None)

(* Fall-through definition for iEmbedCode. If a language implementation fails to define a rule for a particular type
   of expr to embed, it will fall through to here.
*)

iEmbedCode[env_, expr_, uri_, OptionsPattern[]] := (Message[EmbedCode::noembed, expr, uri, OptionValue["OriginalEnvironmentName"]]; $Failed)


(*  Gets param info (names, types, defaults) from an APIFunction spec. Docs currently list this as the allowed APIFunction param spec:
       {"name" -> (type  | {type, defaultValue}, ...}
    I will also support here the obvious: "name" (i.e., not a rule, just a name taking any string value).
    The type can be string or symbol or expression like Restricted[...] or Alternatives[...].
    The param spec can also be completely left out.
    The function fixes what is likely to be a common mistake--specifying a symbol like Integer instead of the string "Integer" (this is
    the If[Head[type] === Symbol...] bits below).

    Returns a list of triples:  {{"name", type, default}...}
    If there was no default specified for the paramter, the default value is returned as None.

    This function is called from language implementation files, so don't change it.
*)
getParamInfo[APIFunction[body_]] = {}

(* Make a list out of the arg spec if user did not. *)
getParamInfo[APIFunction[params:Except[_List], rest__]] := getParamInfo[APIFunction[{params}, rest]]

getParamInfo[APIFunction[params_List, __]] :=
    Replace[params,
        {
            name_String :> {name, "String", None},
            HoldPattern[name_String -> type_ -> default_] :> {name, If[Head[type] === Symbol, ToString[type], type], default},
            HoldPattern[name_String -> type_] :> {name, If[Head[type] === Symbol, ToString[type], type], None}
        },
        {1}
    ]


(* This function does the complex task of weaving together the paramInfo from the APIFunction and the argTypes part of the
   ExternalTypeSignature option to create a fully-fleshed out list of all the arguments (name and type of each) that will
   be passed to the cloud function, in the proper order. This job would be simple if not for the fact that we want to allow
   the ExternalTypeSignature option to not require argument names, as in {"int", "int"}, instead of requiring {"x"->"int", "y"->"int"}.
   This means the sig spec can be a mix of named and positional arguments. Consider an APIFunction with the following param spec:

        {"arg1" -> "Integer", "arg2" -> "Integer"}

   The user calls EmbedCode with this value for the ExternalTypeSignature option (this is just the argTypes part of that option value, leaving
   out the returnType part):

        {"arg2" -> "int", "int"}

   This spec indicates that the user wants to call the APIFunction with a signature (in Java, say) like this:

        call(int arg2, int arg1)

   The interpreterTypeToNativeTypeConverterFunc argument is a function that maps Interpreter types ("Integer", "DateObject", etc.)
   to language native types. It is only used to convert Automatic as an argSpec. For example, a caller can specify Automatic as the
   entire argTypes part, or Automatic for an individual argument: {"x"->"int", Automatic, "z"->"double"}.

   Returns a list that has an entry for every argument that will be passed to the cloud function. Each element looks like
   {"name", "native type", ...optional extra values}. The "native type" value and any optional extra values are returned by the
   interpreterTypeToNativeTypeConverterFunc that you pass in. Because there are optional extra values, access the parts by position
   ([[1]] for the name, [[2]] for the native type, etc.) and don't call functions like Last or Reverse on the elements. One use for
   an optional extra value is the for where the APIFunction param was DelimitedSequence[type, separatorSpec], and you want to record
   the separatorSpec for later use. Other uses might come later. The implementation for each embedding language writes its own
   interpreterTypeToNativeTypeConverterFunc, so you only need to make sure that you can handle the sequence of extra values that your
   function produces.
*)

createFullArgumentSpec[paramInfoFromAPIFunction_, argsFromSignature_, interpreterTypeToNativeTypeConverterFunc_] :=
    Module[{fullArgSpec, paramInfoWithSigNamesRemoved, nextPositionalArgPos, nativeType, thisParamInfo, interpreterTypeFromParamInfo},
        (* argsFromSignature could be Automatic, which means to construct types based on
           the ones named in the APIFunction params spec.
           If argTypes is not Automatic, it is a list {("java type" | {"name" -> "java type"})...}.

           We are going to convert argsFromSignature into a fully-fleshed out list, where each arg has an explicit name
           and native type. The names are either (1) given in the sig spec; (2) deduced by positional comparison to the params spec in
           the APIFunction; or (3) constructed with generic names like "argN" if there is no matching param.
        *)
        If[argsFromSignature === Automatic,
            fullArgSpec = {#1, Sequence @@ interpreterTypeToNativeTypeConverterFunc[#2]}& @@@ paramInfoFromAPIFunction,
        (* else *)
            paramInfoWithSigNamesRemoved = DeleteCases[paramInfoFromAPIFunction, {name_, type_} /; MemberQ[First /@ Cases[argsFromSignature, HoldPattern[_String -> _String]], name]];
            nextPositionalArgPos = 1;
            fullArgSpec = Function[{sigSpec},
                If[MatchQ[sigSpec, _Rule],
                    (* sigSpec was of the form "name"->"native type". Turn it into a list instead of a rule, and deal with Automatic. *)
                    interpreterTypeFromParamInfo = FirstCase[paramInfoFromAPIFunction, {sigSpec[[1]], type_, ___} :> type];
                    nativeType = interpreterTypeToNativeTypeConverterFunc[interpreterTypeFromParamInfo];
                    (* If sigSpec[[2]] == Automatic, nativeType now holds the info we need, because interpreterTypeToNativeTypeConverterFunc
                       returns everything. But if the user specified their own Java type, and the type from paramInfo is a DelimitedSequence,
                       then we want to add information about the separator spec.
                    *)
                    If[sigSpec[[2]] =!= Automatic && Length[nativeType] > 1,
                        nativeType = ReplacePart[nativeType, 1 -> sigSpec[[2]]]
                    ];
                    {sigSpec[[1]], Sequence @@ nativeType},
                (* else *)
                    (* sigSpec for this arg was just a native type, with no name specified. *)
                    If[nextPositionalArgPos <= Length[paramInfoWithSigNamesRemoved],
                        thisParamInfo = paramInfoWithSigNamesRemoved[[nextPositionalArgPos++]];
                        nativeType = interpreterTypeToNativeTypeConverterFunc[thisParamInfo[[2]]];
                        (* If sigSpec == Automatic, nativeType now holds the info we need, because interpreterTypeToNativeTypeConverterFunc
                           returns everything. But if the user specified their own Java type, we use a part of the result from
                           interpreterTypeToNativeTypeConverterFunc: the separator spec that appears when the type from paramInfo is a DelimitedSequence.
                        *)
                        If[sigSpec =!= Automatic,
                            If[Length[nativeType] > 1,
                                (* nativeType was a list giving a separator spec. *)
                                nativeType = ReplacePart[nativeType, 1 -> sigSpec],
                            (* else *)
                                nativeType = sigSpec
                            ]
                        ];
                        {thisParamInfo[[1]], Sequence @@ nativeType},
                    (* else *)
                        (* Positional arg that has no match in the listed APIFunction params. This is not necessarily an error, since the
                           APIFunction might take more args than it lists (it might even list no args and take an arbitrary number).
                           Construct a dummy name.
                        *)
                        {"arg" <> ToString[nextPositionalArgPos++], If[sigSpec === Automatic, "String", sigSpec]}
                    ]
                ]
            ] /@ argsFromSignature;
            (* Another sanity check: Make sure that the sig specifies a sufficient number of non-optional arguments.
               paramInfo will be a list of triples: {{"name", type, default (None if no default)}, ...}
            *)
            If[Length[fullArgSpec] < Length[DeleteCases[paramInfoFromAPIFunction, {_, _, Except[None]}]],
                Message[EmbedCode::argcount];
                (* This Throw gets caught all the way up in EmbedCode itself, so language implementations don't have to worry about
                   handling $Failed as a return value from this function.
                *)
                Throw[$Failed]
            ]
        ];
        fullArgSpec
    ]

createFullArgumentSpecOLD[paramInfoFromAPIFunction_, argsFromSignature_, interpreterTypeToNativeTypeConverterFunc_] :=
    Module[{fullArgSpec, paramInfoWithSigNamesRemoved, nextPositionalArgPos},
        (* argsFromSignature could be Automatic, which means to construct types based on
           the ones named in the APIFunction params spec.
           If argTypes is not Automatic, it is a list {("java type" | {"name" -> "java type"})...}.

           We are going to convert argsFromSignature into a fully-fleshed out list, where each arg has an explicit name
           and native type. The names are either (1) given in the sig spec; (2) deduced by positional comparison to the params spec in
           the APIFunction; or (3) constructed with generic names like "argN" if there is no matching param.
        *)
        If[argsFromSignature === Automatic,
            fullArgSpec = {#1, Sequence @@ interpreterTypeToNativeTypeConverterFunc[#2]}& @@@ paramInfoFromAPIFunction,
        (* else *)
            paramInfoWithSigNamesRemoved = DeleteCases[paramInfoFromAPIFunction, {name_, type_} /; MemberQ[First /@ Cases[argsFromSignature, HoldPattern[_String -> _String]], name]];
            nextPositionalArgPos = 1;
            fullArgSpec = Function[{sigSpec},
                If[MatchQ[sigSpec, name_ -> type_],
                    (* sigSpec was of the form "name"->"native type". Turn it into a list instead of a rule. *)
                    {sigSpec[[1]], If[# === Automatic, Sequence @@ interpreterTypeToNativeTypeConverterFunc[#], #]& @ sigSpec[[2]]},
                (* else *)
                    (* sigSpec for this arg was just a native type, with no name specified. *)
                    If[nextPositionalArgPos <= Length[paramInfoWithSigNamesRemoved],
                        {#1, If[sigSpec === Automatic, Sequence @@ interpreterTypeToNativeTypeConverterFunc[sigSpec], sigSpec]}& @@ paramInfoWithSigNamesRemoved[[nextPositionalArgPos++]],
                    (* else *)
                        (* Positional arg that has no match in the listed APIFunction params. This is not necessarily an error, since the
                           APIFunction might take more args than it lists (it might even list no args and tak an arbitrary number).
                           Construct a dummy name.
                        *)
                        {"arg" <> ToString[nextPositionalArgPos++], If[sigSpec === Automatic, Sequence @@ interpreterTypeToNativeTypeConverterFunc[sigSpec], sigSpec]}
                    ]
                ]
            ] /@ argsFromSignature;
            (* Another sanity check: Make sure that the sig specifies a sufficient number of non-optional arguments.
               paramInfo will be a list of triples: {{"name", type, default (None if no default)}, ...}
            *)
            If[Length[fullArgSpec] < Length[DeleteCases[paramInfoFromAPIFunction, {_, _, Except[None]}]],
                Message[EmbedCode::argcount];
                Return[$Failed]
            ]
        ];
        fullArgSpec
    ]


$multipartBoundary = "92afc99d-494c-41aa-8f16-2febb8b310dd"

End[] (* EmbedCode`Common` *)


(***********************  $EmbeddingEnvironments  *************************)

Unprotect[$EmbedCodeEnvironments]

$EmbedCodeEnvironments := First /@ embeddingNamesAndImplementations[]

findFileForEmbedding[env_String] := Last /@ embeddingNamesAndImplementations[]

(* Returns a list of pairs: {{"Java", "/path/to/Java.m"}, {"Python", "/path/to/Python.m"}, ...}.*)
embeddingNamesAndImplementations[] :=
    Module[{langDirEnvironFiles, pacletEnvironFiles, nameAndFilePairs},
        langDirEnvironFiles = FileNames["*.m" | "*.wl", $environmentsDir];
        (* TODO: PacletManager support for finding new embedding defs. *)
        (* pacletEnvironFiles = Paclet... *) pacletEnvironFiles = {};
        nameAndFilePairs = {FileBaseName[FileNameTake[#]], #}& /@ Join[pacletEnvironFiles, langDirEnvironFiles];
        (* Delete duplicates, giving preference to files provided in paclets over the ones in the layout dir
           (this happens just because paclet files are listed first in the set of files).
        *)
        nameAndFilePairs = DeleteDuplicates[nameAndFilePairs, First[#1]==First[#2]&];
        (* Delete XXX-Common, as these are not actually standalone embeddings. *)
        nameAndFilePairs = Select[nameAndFilePairs, !StringMatchQ[First[#], "*-Common"]&];
        SortBy[nameAndFilePairs, First]
    ]

SetAttributes[$EmbedCodeEnvironments, {Protected, ReadProtected}]


(**************************  EmbeddingObject  ****************************)

Unprotect[EmbeddingObject]

Format[EmbeddingObject[data_Association], StandardForm] :=
    Interpretation[
        embedCodePanel[data],
        EmbeddingObject[data]
    ]

(* You can look up fields in the assoc as if the EmbeddingObject itself was an Association. *)
EmbeddingObject[data_Association][key_String] := data[key]

SetAttributes[EmbeddingObject, {Protected, ReadProtected}]


hasItem[assoc_, key_] := !MatchQ[assoc[key], _Missing]


embedCodePanel[data_Association] :=
    Module[{envName, libPane, codePane},
        envName = data["EnvironmentName"];
        libPane = If[hasItem[data, "FilesSection"], makeLibraryPane[data["FilesSection"]], Sequence @@ {}];
        codePane = If[hasItem[data, "CodeSection"], makeCodePane[data["CodeSection"]], Sequence @@ {}];
        Framed[
            Panel[
                Column[
                    {
                        If[!TrueQ[$CloudEvaluation],
                            Style["Embeddable Code", "ControlStyle", FontSize -> Larger, FontWeight -> Bold, FontColor -> GrayLevel[0.3]],
                        (* else *)
                            (* Cloud *)
                            Style["Embeddable Code", FontSize -> 14, FontWeight -> Bold, FontColor -> GrayLevel[0.3]]
                        ],
                        Style[
                            If[hasItem[data, "FilesSection"],
                                "Use the files and code below to call the Wolfram Cloud function from ",
                            (* else *)
                                "Use the code below to call the Wolfram Cloud function from "
                            ] <> envName <> ":",
                            If[!TrueQ[$CloudEvaluation], Sequence @@ {}, FontSize -> 11],
                            FontColor -> RGBColor[.23,.23,.23]
                        ],
                        libPane,
                        codePane
                    },
                    Spacings -> {0, {0.3, 0.4, 0.8, 1.5}}
                ],
                BaseStyle -> {"Deploy"},
                ImageSize -> If[!TrueQ[$CloudEvaluation], 630 (* Just fits in standard Untitled window size *), 750],
                Background ->RGBColor[.87,.87,.87]
            ],
            RoundingRadius -> 6,
            FrameMargins -> {{2,2},{1,1}},
            Background -> RGBColor[.87,.87,.87],
            FrameStyle -> RGBColor[.76,.76,.76]
        ]
    ]

If[!TrueQ[$CloudEvaluation],
    makeLibraryPane[data_Association] :=
        Module[{fileList, title, desc, downloadButton},
            fileList = data["FileList"];
            title = data["Title"];
            If[!StringQ[title], title = "Libraries"];
            desc = data["Description"];
            If[!StringQ[desc], desc = "Add these files to your project:"];
            If[Length[fileList] > 1,
                downloadButton = libraryDownloadControl["ActionMenu", data["ZipDownloadURL"], data["TarDownloadURL"]],
            (* else *)
                downloadButton = libraryDownloadControl["Button", data["DownloadURL"]]
            ];
            Framed[
                Grid[{{#, Item[downloadButton, Alignment -> Right]}},
                    Alignment -> {Left, Top},
                    ItemSize -> Fit
                ]& @
                Column[
                    {
                        Style[title, "ControlStyle", FontWeight -> Bold, FontColor -> RGBColor[.33,.33,.33]],
                        Style[desc, FontColor -> RGBColor[.33,.33,.33]],
                        Sequence @@ (Row[{"\[Bullet] ", Hyperlink[Style[#[[1]], FontColor->RGBColor[.11,.35,.73]], #[[2]]]}]& /@ fileList)
                    },
                    Spacings->{0,{0.5,0.3,0.5,{0.1}}}
                ],
                Background -> RGBColor[.96,.96,.96],
                FrameMargins -> {{10,12},{10,10}},
                FrameStyle -> RGBColor[.69,.69,.69]
            ]
        ];
    libraryDownloadControl["Button", downloadURL_] :=
        Button[
            "Download file",
            doDownload[downloadURL],
            Appearance :> FEPrivate`FrontEndResource["FEExpressions", "GrayButtonNinePatchAppearance"],
            FrameMargins -> {{10,10},{0,0}},
            ImageSize -> Automatic,
            BaseStyle -> "DialogStyle",
            Method -> "Queued"
        ];
    libraryDownloadControl["ActionMenu", zipDownloadURL_, tarDownloadURL_] :=
        ActionMenu[
            Button["Download All Files \[DownPointer]",
                Null,
                Appearance :> FEPrivate`FrontEndResource["FEExpressions", "GrayButtonNinePatchAppearance"],
                FrameMargins -> {{10,10},{0,0}},
                ImageSize -> Automatic,
                BaseStyle -> "DialogStyle"
            ],
            {
                "as .zip" :> doDownload[zipDownloadURL],
                "as .tar" :> doDownload[tarDownloadURL]
            },
            Appearance -> None,
            BaseStyle -> "DialogStyle",
            MenuStyle -> "DialogStyle",
            Method -> "Queued"
        ],
(* else *)
    (* Cloud version *)
    makeLibraryPane[data_Association] :=
        Module[{fileList, title, desc, downloadButton},
            fileList = data["FileList"];
            title = data["Title"];
            If[!StringQ[title], title = "Libraries"];
            desc = data["Description"];
            If[!StringQ[desc], desc = "Add these files to your project:"];
            If[Length[fileList] > 1,
                downloadButton = libraryDownloadControl["ActionMenu", data["ZipDownloadURL"], data["TarDownloadURL"]],
            (* else *)
                downloadButton = libraryDownloadControl["Button", data["DownloadURL"]]
            ];
            Framed[
                Grid[{{#, Item[downloadButton, Alignment -> Right, ItemSize -> Fit]}},
                    Alignment -> {Left, Top}
                ]& @
                Column[
                    {
                        Style[title, "ControlStyle", FontWeight -> Bold, FontColor -> RGBColor[.33,.33,.33]],
                        Style[desc, FontColor -> RGBColor[.33,.33,.33], FontSize -> 11],
                        Sequence @@ (Row[{"\[Bullet] ", Hyperlink[Style[#[[1]], FontColor->RGBColor[.11,.35,.73]], #[[2]]]}]& /@ fileList)
                    },
                    Spacings->{0,{0.5,0.3,0.5,{0.1}}}
                ],
                Background -> RGBColor[.96,.96,.96],
                FrameMargins -> {{10,12},{10,10}},
                FrameStyle -> RGBColor[.69,.69,.69]
            ]
        ];
    libraryDownloadControl["Button", downloadURL_] :=
        Button[
            Hyperlink["Download File", downloadURL],
            Null,
            FrameMargins -> {{10,10},{2,2}},
            ImageSize -> Automatic,
            BaseStyle -> "DialogStyle",
            FontSize->12,
            Method -> "Queued"
        ];
    libraryDownloadControl["ActionMenu", zipDownloadURL_, tarDownloadURL_] :=
        ActionMenu[
            Item[Button[Style["Download All Files \[DownPointer]", FontSize->12],
                Null,
                FrameMargins -> {{10,10},{2,2}},
                ImageSize -> Automatic,
                BaseStyle -> "DialogStyle"
            ], ItemSize->Fit],
            {
                Hyperlink[Style["as .zip", FontSize->12], zipDownloadURL],
                Hyperlink[Style["as .tar", FontSize->12], tarDownloadURL]
            },
            Appearance -> None,
            BaseStyle -> "DialogStyle",
            MenuStyle -> "DialogStyle",
            FontSize->12,
            Method -> "Queued"
        ]
]

If[!TrueQ[$CloudEvaluation],
    makeCodePane[data_Association] :=
    Module[{code, title, desc},
        code = data["Content"];
        title = data["Title"];
        If[!StringQ[title], title = "Code"];
        desc = data["Description"];
        Switch[desc,
            Automatic,
                desc = "Use this code to call your Wolfram Cloud function:",
           _Missing,
                desc = ""
        ];
        Framed[
            Column[{
                    Grid[{{Column[{Style[title, "ControlStyle", FontWeight -> Bold, FontColor -> RGBColor[.33,.33,.33]], Style[desc, FontColor -> RGBColor[.33,.33,.33]]}],
                           With[{code = code},
                               Item[Button["Copy to Clipboard", CopyToClipboard[code], Appearance :> FEPrivate`FrontEndResource["FEExpressions", "GrayButtonNinePatchAppearance"],
                                   FrameMargins -> {{10,10},{0,0}}, ImageSize -> Automatic, BaseStyle -> "DialogStyle", Method -> "Queued"], Alignment->Right]
                           ]
                         }},
                         Alignment -> {Left, Top},
                         ItemSize -> Fit
                    ],
                    Framed[
                        Pane[
                            TextCell[code, Deployed -> False, Editable -> False, StripOnInput -> True],
                            ImageSize -> {Scaled[1], Max[60, Min[Round[30 StringCount[code, "\n"]], 300]]},
                            Scrollbars -> Automatic,
                            AppearanceElements -> {}
                        ],
                        FrameStyle -> RGBColor[.69,.69,.69],
                        Background -> White,
                        BaseStyle -> "StandardForm"
                    ]
                },
                ItemSize -> Fit,
                Spacings->If[desc === "", {0,{0.2,{0.5}}}, {0,{0.7,{0.7}}}]
            ],
            Background -> RGBColor[.96,.96,.96],
            FrameMargins -> {{14,14},{14,10}},
            FrameStyle -> RGBColor[.69,.69,.69]
        ]
    ],
 (* else *)
     (* Cloud *)
    makeCodePane[data_Association] :=
    Module[{code, title, desc, filename, obj},
        code = data["Content"];
        title = data["Title"];
        If[!StringQ[title], title = "Code"];
        desc = data["Description"];
        filename =
            If[StringQ[data["Filename"]],
                data["Filename"],
            (* else *)
                "embedcode.txt"
            ];
        (* To create a unique filename for the downloadable code file, create a UUID, take the first 3 characters, and append that to the file, as in WolframCloudCall-d12c.java" *)
        filename = StringReplace[filename, name___ ~~ "." ~~ suffix:(Except["."]..) :> name <> "-" <> StringTake[CreateUUID[], 3] <> "." <> suffix];
        With[{code = code},
            obj = CloudDeploy[APIFunction[{}, HTTPResponse[code, "ContentType"->"application/octet-stream"]&, "Text"], filename, Permissions->"Public"]
        ];
        Switch[desc,
            Automatic,
                desc = "Use this code to call your Wolfram Cloud function:",
           _Missing,
                desc = ""
        ];
        Framed[
            Column[{
                    Grid[{{Column[{Style[title, "ControlStyle", FontWeight -> Bold, FontColor -> RGBColor[.33,.33,.33]],
                                   Style[desc, FontColor -> RGBColor[.33,.33,.33], FontSize -> 11]}
                           ],
                           (*
                           Item[
                               Framed[
                                   Hyperlink[Style["Download Code", FontColor->RGBColor[.11,.35,.73], FontSize->12], First[obj]]
                               ],
                               Alignment->Right, ItemSize->Fit
                           ]
                           *)
                           Item[Button[Hyperlink[Style["Download Code", FontSize->12, FontColor -> RGBColor[0,0,0]], First[obj]], Null,
                               FrameMargins -> {{10,10},{3,3}}, ImageSize -> Automatic, BaseStyle -> "DialogStyle", Evaluator -> None], Alignment->Right, ItemSize->Fit]
                         }},
                         Alignment -> {Left, Top}
                    ],
                    Framed[
                        Pane[
                            (* At the moment, in cloud notebooks empty lines in these code cells are being stripped for some reason. Therefore we insert a space char
                               onto blank lines with the StringReplace below.
                            *)
                            TextCell[Style[StringReplace[code, "\n\n" -> "\n \n"], FontFamily->"Courier", FontSize->11], Deployed -> False, Editable -> False, StripOnInput -> True],
                            ImageSize -> {Scaled[1], Max[60, Min[Round[30 StringCount[code, "\n"]], 300]]},
                            Scrollbars -> Automatic,
                            AppearanceElements -> {}
                        ],
                        FrameStyle -> RGBColor[.69,.69,.69],
                        Background -> White,
                        BaseStyle -> "StandardForm"
                    ]
                },
                ItemSize -> Fit,
                Spacings->If[desc === "", {0,{0.2,{0.5}}}, {0,{0.7,{0.7}}}]
            ],
            Background -> RGBColor[.96,.96,.96],
            FrameMargins -> {{14,14},{14,10}},
            FrameStyle -> RGBColor[.69,.69,.69]
        ]
    ]
]



doDownload[url_String] :=
    Module[{dir},
        dir = SystemDialogInput["Directory"];
        If[StringQ[dir],
            URLSave[url, FileNameJoin[{dir, Last[URLParse[url, "Path"]]}]]
        ]
    ]

  
End[]

EndPackage[]

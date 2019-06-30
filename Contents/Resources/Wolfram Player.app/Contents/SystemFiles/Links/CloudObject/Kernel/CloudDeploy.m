BeginPackage["CloudObject`"]

System`CloudDeploy;
CloudObject`CloudDeployActiveQ;

Begin["`Private`"]

(* Dependencies *)
System`EmbeddedHTML;
System`GrammarRules;
System`CloudBase;
System`SourceLink;
Hold[System`$SourceLink];
Hold[System`$CloudEvaluation];

SetAttributes[{headDeployFormat, expressionMimeType}, HoldFirst];

headDeployFormat[APIFunction] = "API";
headDeployFormat[Delayed|Dynamic] = "Computation";
headDeployFormat[FormFunction] = "Form";
headDeployFormat[ScheduledTask] = "Task";
headDeployFormat[GrammarRules] = "Grammar";
headDeployFormat[expr_] := SymbolName[expr];

expressionMimeType["CloudCDF"] := "application/vnd.wolfram.notebook";
expressionMimeType["HTMLCloudCDF"] := "application/vnd.wolfram.cloudcdf.html";
expressionMimeType["NBElement"] := "application/vnd.wolfram.notebook.element";
expressionMimeType["Expression"|Expression] := "application/vnd.wolfram.expression";
expressionMimeType["Notebook"|Notebook] := "application/mathematica";
expressionMimeType["ExternalBundle"|ExternalBundle] := "application/vnd.wolfram.bundle";
expressionMimeType["Directory"|Directory] := "inode/directory";
expressionMimeType[expr_String] := "application/vnd.wolfram.expression." <> ToLowerCase[expr];
expressionMimeType[expr_[___]] := expressionMimeType[expr];
expressionMimeType[expr_] := "application/vnd.wolfram.expression." <> ToLowerCase[headDeployFormat[expr]];

CloudDeployActiveQ[HoldPattern[Alternatives[
    _Delayed,
    _FormFunction,
    _System`FormPage, (* System should be removed after FormPage will be in the build *)
    _APIFunction,
    _ScheduledTask,
    _URLDispatcher,
    _GrammarRules
    ]]] := True;
    
CloudDeployActiveQ[_] := False;

nonMetaOption = CloudBase | CloudObjectNameFormat | CloudObjectURLType;

Unprotect[CloudDeploy];

$SourceLink = Automatic;

Options[CloudDeploy] = objectFunctionOptionsJoin[$objectCreationOptions, {CloudBase -> Automatic, IncludeDefinitions -> True, SourceLink -> Automatic, AutoCopy -> False}];

Options[CloudPublish] = Normal[Association[Options[CloudDeploy], Permissions -> {All -> Automatic}, AutoCopy -> True]];

CloudDeploy[bundle:ExternalBundle[bundleElements_List], dest:CloudObject[url_, objopts:OptionsPattern[CloudObject]], opts:OptionsPattern[]] :=
    Module[{optsNew, elementObjects, bundleexpr, destNew},
    	destNew = CloudObject[url, handleOptsForURL[CloudDeploy, {opts, objopts}]];
        optsNew = Sequence @@ FilterRules[{opts}, Except[nonMetaOption]];
        (* Step 1 of 3. Ensure the bundle directory exists *)
        Replace[
            Quiet[createBundle[destNew], CloudDeploy::notparam],
            HTTPError[___] :> Return[$Failed]
        ];

        (* Step 2 of 3. deploy the individual elements *)
        elementObjects = $lastBundleDeployResult = Map[
            deployBundleElement[destNew, #, optsNew]&,
            bundleElements
        ];
        If[Position[elementObjects, $Failed, Infinity, 1] =!= {},
            Return[$Failed]
        ];

        (* Step 3 of 3. deploy the ExternalBundle content *)
        bundleexpr = ExternalBundle[elementObjects];
        CloudPut[bundleexpr, FileNameJoin[{destNew, ".bundle"}], optsNew];

        destNew
    ];

assocToList[assoc_Association] := Map[assoc[#]&, Keys[assoc]] (* workaround for certain Mathematica builds where Normal[_Association] normalizes deeply *)

deployBundleElement[dir_CloudObject, name_String -> elements_List, opts:OptionsPattern[]] := Replace[
    CreateDirectory[FileNameJoin[{dir, name}]],
    {
        subdir_CloudObject :>
        name -> Map[deployBundleElement[subdir, #, opts]&, elements],
        _ :> $Failed
    }
]

deployBundleElement[dir_CloudObject, name_String -> direlements_Association, opts:OptionsPattern[]] :=
    deployBundleElement[dir, name -> assocToList[direlements], opts]

deployBundleElement[dir_CloudObject, name_String -> expr_, opts:OptionsPattern[]] := Replace[
    CloudDeploy[expr, FileNameJoin[{dir, name}], opts],
    {
        obj_CloudObject :> name -> obj,
        _ :> $Failed
    }
]

CloudDeploy[ExternalBundle[elements_Association], dest_CloudObject, opts:OptionsPattern[]] :=
    CloudDeploy[ExternalBundle[assocToList[elements]], dest, Sequence @@ FilterRules[{opts}, Except[CloudBase]]]

CloudDeploy[HoldPattern[grammar_GrammarRules], obj_CloudObject, opts:OptionsPattern[]] :=
With[{newGrammar = Semantic`PLIDump`addDefinitions[grammar]},
    Which[
        MatchQ[newGrammar, Except[_GrammarRules]],
        (* presumably a Message was already issued *)
        $Failed
        ,
        TrueQ[$CloudEvaluation],
        Semantic`PLIDump`iGrammarDeploy[newGrammar, obj, Sequence @@ FilterRules[{opts}, Except[CloudBase]]]
        ,
        True,
        internalCloudEvaluate[CloudDeploy[newGrammar, obj, Sequence @@ FilterRules[{opts}, Except[CloudBase]]]]
    ]
]

CloudDeploy[expr_?CloudDeployActiveQ, obj_CloudObject, opts:OptionsPattern[]] := 
	Catch[
    	iCloudPut[Unevaluated[expr], obj, expressionMimeType[expr], CloudDeploy,
    		(* iCloudPut has IncludeDefinitions->False as default *) 
    		IncludeDefinitions -> handleIncludeDefinition[OptionValue[IncludeDefinitions], CloudDeploy], 
    		Sequence @@ FilterRules[{opts}, Except[CloudBase|IncludeDefinitions]]],
    	BadOption
	]

CloudDeploy[ExportForm[expr_, format_, rest___], obj_CloudObject, opts:OptionsPattern[]] :=
    CloudExport[Unevaluated[expr], format, obj, rest, Sequence @@ FilterRules[{opts}, Except[CloudBase]]]

CloudDeploy[redirect:HTTPRedirect[_String|_URL|_CloudObject, ___], rest___] := 
    CloudDeploy[GenerateHTTPResponse[redirect], rest]

CloudDeploy[res:HTTPResponse[body:$BodyPattern, meta_?AssociationQ, rest___], obj_CloudObject, opts:OptionsPattern[]] := 
    Catch[
        (* Server doesn't support MX for static responses, force text format; see CLOUD-15159 *)
        Block[{CloudObject`Private`$AllowMXCloudPut = False},
            iCloudPut[
                HTTPResponse[body, Join[meta, res["Meta"]], rest], 
                obj, 
                "application/vnd.wolfram.httpresponse",
                CloudDeploy, 
                IncludeDefinitions -> handleIncludeDefinition[OptionValue[IncludeDefinitions], CloudDeploy],
                Sequence @@ FilterRules[{opts}, Except[CloudBase|IncludeDefinitions]]
            ]
        ],
        BadOption
    ]

CloudDeploy[expr_, obj:CloudObject[uri_, objopts:OptionsPattern[CloudObject]], opts:OptionsPattern[]] :=
	Module[{nameFormat, urlType, objOptsNew, optsNew},
		nameFormat = Quiet[OptionValue[CloudDeploy, {opts, objopts}, CloudObjectNameFormat], OptionValue::nodef];
		urlType = Quiet[OptionValue[CloudDeploy, {opts, objopts}, CloudObjectURLType], OptionValue::nodef];
		objOptsNew = FilterRules[{objopts}, Except[CloudObjectNameFormat | CloudObjectURLType]];
		optsNew = DeleteDuplicates[Join[{CloudObjectNameFormat -> nameFormat, CloudObjectURLType -> urlType}, FilterRules[{opts}, Except[CloudBase]]]];
		CloudDeploy[ExportForm[expr], CloudObject[uri, objOptsNew], optsNew]
	]
	
CloudDeploy[expr_, uri_String, opts:OptionsPattern[]] :=
    Module[ {cbase, nameFormat, urlType, obj, optsForWriting},
      	Catch[
        	cbase = handleCBase[OptionValue[CloudBase], CloudDeploy];
        	nameFormat = OptionValue[CloudObjectNameFormat];
        	urlType = OptionValue[CloudObjectURLType];
        	obj = Block[{$CloudBase = cbase}, CloudObject[uri]];
        	optsForWriting = FilterRules[{opts}, Except[nonMetaOption]];
        	If[MatchQ[cloudDeployPreprocess[cbase, Unevaluated[expr], obj, Sequence @@ optsForWriting], _CloudObject],
        	    CloudObject[obj, CloudObjectNameFormat -> nameFormat, CloudObjectURLType -> urlType],
        	    $Failed
        	]
      		,
      		BadOption
      	]
    ]
    
CloudDeploy[expr_, URL[uri_String], opts:OptionsPattern[]] := CloudDeploy[Unevaluated[expr], uri, opts]  
    
CloudDeploy[expr_, opts:OptionsPattern[]] :=
    Module[{cbase, obj, optsForWriting},
      Catch[
    	cbase = handleCBase[OptionValue[CloudBase], CloudDeploy];
    	obj = Block[{$CloudBase = cbase}, CloudObject[]];
    	optsForWriting = FilterRules[{opts}, Except[nonMetaOption]];
        If[MatchQ[cloudDeployPreprocess[cbase, Unevaluated[expr], obj, Sequence @@ optsForWriting], _CloudObject],
        	    CloudObject[obj, CloudObjectNameFormat -> OptionValue[CloudObjectNameFormat], CloudObjectURLType -> OptionValue[CloudObjectURLType]], 
        	    $Failed
        	]  
        ,
        BadOption
      ]
    ]

(* this needs to follow the CloudDeploy[expr_, opts:OptionsPattern[]] definition *)
CloudDeploy[expr_, dest_, opts:OptionsPattern[]]:=
    (Message[CloudDeploy::invcloudobj, dest]; $Failed)
    
CloudDeploy[args___] := (ArgumentCountQ[CloudDeploy,Length[DeleteCases[{args},_Rule,Infinity]],1,2];Null/;False)

CloudDeploy[args_, failureObject_Failure] := failureObject

Options[cloudDeployPreprocess] = FilterRules[Options[CloudDeploy], Except[CloudBase]]

cloudDeployPreprocess[cbase_, expr_, obj_Failure, opts:OptionsPattern[]] := obj

cloudDeployPreprocess[cbase_, expr_, obj_CloudObject, opts:OptionsPattern[]] :=
    Block[{$CloudBase = cbase},
        CloudDeploy[Unevaluated[expr], obj, handleCloudDeployOptions[obj, False, CloudDeploy, opts]]]

cloudDeployPreprocess[cbase_String, expr_, obj_, opts:OptionsPattern[]] := $Failed 

(* It Throws with the BadOption tag if there is invalid option value. *)
handleCloudDeployOptions[obj_CloudObject, useOriginal_?BooleanQ, msghd_, opts___?OptionQ] :=
    Module[ {optsNew, metaNew, includeDefNew},
        optsNew = Association[Options[msghd], opts];
        metaNew = Join[handleMetaInfo[optsNew[MetaInformation], msghd], {
            handleAppearanceRules[optsNew[AppearanceRules], msghd], 
            handleAutoCopy[optsNew[AutoCopy], msghd],
            handleSourceLink[optsNew[SourceLink], useOriginal, obj, msghd]}];
        includeDefNew = handleIncludeDefinition[optsNew[IncludeDefinitions], msghd];
        permissionsFormatValidation[optsNew[Permissions], msghd];
        Join[FilterRules[Normal[optsNew], {CloudObjectNameFormat, IconRules}],
            {IncludeDefinitions -> includeDefNew, MetaInformation -> metaNew, Permissions -> optsNew[Permissions]}]
    ]

handleCBase[Automatic, msghd_:CloudObject] := $CloudBase
handleCBase[cbase_String, msghd_:CloudObject] := Replace[cbase, $cloudBaseAbbreviations]
handleCBase[URL[cbase_String], msghd_:CloudObject] := handleCBase[cbase]
handleCBase[cbase_, msghd_:CloudObject] := (Message[msghd::invbase, cbase]; Throw[$Failed, BadOption])

handleMetaInfo[info_Rule, msghd_] := {info}
handleMetaInfo[info_?AssociationQ, msghd_] := Normal[info]
handleMetaInfo[info:{_Rule ...}, msghd_] := info
handleMetaInfo[other_, msghd_] := (Message[msghd::invmeta, other]; Throw[$Failed, BadOption])

(* For both CloudDeploy and CloudPublish *)
handleSourceLink[Automatic, useOriginal_?BooleanQ, origin_CloudObject, msghd_] :=
	handleSourceLink[handleSLink[If[useOriginal, origin, $SourceLink]], msghd]

handleSourceLink[src_, useOriginal_?BooleanQ, origin_CloudObject, msghd_] := handleSourceLink[src, msghd]	

handleSourceLink[None, msghd_] := Nothing
handleSourceLink[src_CloudObject, msghd_] := "__SourceLink" -> normalizeSourceLink[src]
handleSourceLink[src_, msghd_] := (Message[msghd::invsrc, src]; Throw[$Failed, BadOption])

handleSLink[Automatic] := $EvaluationCloudObject
handleSLink[src_] := src

normalizeSourceLink[CloudObject[url_String]] := url
normalizeSourceLink[None] := None

(* For both CloudDeploy and CloudPublish *)
handleAutoCopy[autocp_?BooleanQ, msghd_] := "__AutoCopy" -> autocp
handleAutoCopy[autocp_, msghd_] := (Message[msghd::invautocp, autocp]; Throw[$Failed, BadOption])

handleIncludeDefinition[includedf_?BooleanQ, msghd_] := includedf
handleIncludeDefinition[includedf_, msghd_] := (Message[msghd::invincludedf, includedf]; Throw[$Failed, BadOption])

handleAppearanceRules[appRules_Rule, msghd_] := handleAppearanceRules[{appRules}, msghd]
handleAppearanceRules[appRules:{Rule["Branding", None | Automatic | True | False]}, msghd_] := "__AppearanceRules" -> appRules
handleAppearanceRules[appRules_, msghd_] := (Message[msghd::invaprl, appRules]; Throw[$Failed, BadOption])

permClasses = All | _SecuredAuthenticationKey | _ApplicationIdentificationKey | _PermissionsGroup | _PermissionsKey | _String | _Association
permClassList = {permClasses..}
   
permissionsFormatValidation[permissions : Automatic| _String | _PermissionsKey | Rule[permClasses | permClassList, _] | {Rule[permClasses | permClassList, _] ..}, msghd_] := permissions
permissionsFormatValidation[permissions_, msghd_] := (Message[msghd::invperm, permissions]; Throw[$Failed, BadOption])

shareeClasses = _PermissionsGroup | _String

handleSharingList[sharees : {}| shareeClasses | {shareeClasses ..}, msghd_] := sharees
handleSharingList[sharees_, msghd_] := (Message[msghd::invsharing, sharees]; Throw[$Failed, BadOption])

handleCloudObjectURLType[Automatic, msghd_]:= handleCloudObjectURLType[$CloudObjectURLType, msghd]
handleCloudObjectURLType["Object", msghd_]:= $objRoot
handleCloudObjectURLType["Environment", msghd_]:= $envRoot
handleCloudObjectURLType[type_, msghd_]:= (Message[msghd::invurltp, type]; Throw[$Failed, BadOption])

handleOptsForURL[symbol_Symbol, opts_List] :=
	Module[{nameFormat, urlType},
		nameFormat = Quiet[OptionValue[CloudDeploy, opts, CloudObjectNameFormat], OptionValue::nodef];
		urlType = Quiet[OptionValue[CloudDeploy, opts, CloudObjectURLType], OptionValue::nodef];
		{CloudObjectNameFormat -> nameFormat, CloudObjectURLType -> urlType}
	]

SetAttributes[CloudDeploy, {ReadProtected}];
Protect[CloudDeploy];

createBundle[dest_CloudObject, mimeTypeExtension_String:""] :=
    responseCheck[execute[dest, Automatic, UseUUID -> False,
        Type -> "application/vnd.wolfram.bundle"<>mimeTypeExtension], CloudDeploy, dest];
        
(*****************************************************************************)
(* CloudPublish *)

CloudPublish[opts:OptionsPattern[]] :=
    If[TrueQ[$CloudEvaluation],
        CloudPublish[$EvaluationCloudObject, opts],
        CloudPublish[EvaluationNotebook[], opts]
    ]

CloudPublish[obj_CloudObject, opts:OptionsPattern[]] :=
	Module[{cbase, dest},
      Catch[
      	cbase = handleCBase[OptionValue[CloudBase], CloudPublish];
    	dest = Block[{$CloudBase = cbase}, CloudObject[]];
        CloudPublish[obj, dest, Sequence @@ FilterRules[{opts}, Except[CloudBase]]]
        ,
        BadOption
      ]
    ]

CloudPublish[obj_CloudObject, dest_String, opts:OptionsPattern[]] :=
	Module[{cbase, destObj},
      Catch[
      	cbase = handleCBase[OptionValue[CloudBase], CloudPublish];
    	destObj = Block[{$CloudBase = cbase}, CloudObject[dest]];
        CloudPublish[obj, destObj, Sequence @@ FilterRules[{opts}, Except[CloudBase]]]
        ,
        BadOption
      ]
    ]   

CloudPublish[obj_CloudObject, dest_CloudObject, opts:OptionsPattern[]] :=
    Module[{res, optsNew},
        Catch [
            res = Quiet[CopyFile[obj, dest]]; (* TODO: do this with one CopyFile call once CLOUD-11500 is fixed *)
            If[MatchQ[res, _CloudObject],
            	(* filter out the options that only make sense for CloudPublish[expr] case *)
            	optsNew = FilterRules[handleCloudDeployOptions[obj, True, CloudPublish, opts],  Except[IncludeDefinitions | CloudObjectNameFormat]];
                SetOptions[res, optsNew];
                res,
                (* Else *)
                Message[CloudPublish::srverr, obj];
                $Failed
            ]
            ,
            BadOption
        ]
    ]

CloudPublish[expr_, opts:OptionsPattern[]] :=
	Module[{cbase, dest},
      Catch[
      	cbase = handleCBase[OptionValue[CloudBase], CloudPublish];
    	dest = Block[{$CloudBase = cbase}, CloudObject[]];
        CloudPublish[expr, dest, Sequence @@ FilterRules[{opts}, Except[CloudBase]]]
        ,
        BadOption
      ]
    ]

CloudPublish[expr_, dest_String, opts:OptionsPattern[]] :=
	Module[{cbase, destObj},
      Catch[
      	cbase = handleCBase[OptionValue[CloudBase], CloudPublish];
    	destObj = Block[{$CloudBase = cbase}, CloudObject[dest]];
        CloudPublish[expr, destObj, Sequence @@ FilterRules[{opts}, Except[CloudBase]]]
        ,
        BadOption
      ]
    ]

CloudPublish[expr_, dest_CloudObject, opts:OptionsPattern[]] :=
    Catch [
        (* Set IconRules->None until the bug is fixed where this closes the corresponding notebook. *)
        CloudDeploy[expr, dest, IconRules -> None, handleCloudDeployOptions[dest, False, CloudPublish, opts]]
        ,
        BadOption
    ]

CloudPublish[failureObj_Failure, opts:OptionsPattern[]] := failureObj

CloudPublish[failureObjSrc_Failure, dest_, opts:OptionsPattern[]] := failureObjSrc

CloudPublish[src_, failureObjDest_Failure, opts:OptionsPattern[]] := failureObjDest

CloudPublish[failureObjSrc_Failure, failureObjDest_Failure, opts:OptionsPattern[]] := failureObjSrc

CloudPublish[args___]:=
    (ArgumentCountQ[CloudPublish, Length[DeleteCases[{args}, _Rule, Infinity]], 0, 2]; Null /; False)                

End[]

EndPackage[]

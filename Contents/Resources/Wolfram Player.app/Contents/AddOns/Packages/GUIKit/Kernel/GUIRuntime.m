
(* GUIRuntime subpackage *)

(* :Context: GUIKit` *)

(* :Copyright: Copyright 2004, Wolfram Research, Inc. *)

(* Each subpackage is called within `Private` *)


(*******************************
   Options
 *******************************)

Options[GUIRun] = {
  Debug -> False,
  Context -> Automatic,
  ReleaseMethod -> Automatic,
  IncludedScriptContexts -> {}
  };

Options[GUIRunModal] = Join[ Options[GUIRun],
  {
    ReturnScript :> None
  }];

Options[GUIResolve] = Join[ Options[GUIRun],
  {
    GUIObject -> Automatic
  }];

Options[GUIResolveObject] = Options[GUIResolve];

Options[GUILoad] = Options[GUIRun];

Options[ReleaseGUIObject] = {
  Debug -> False
  };

Options[CloseGUIObject] = {
  Debug -> False
  };
  
Options[GUIObject] = {
  };
  
Options[GUIInformation] = {
  };

(*******************************
   Messages
 *******************************)

GUIRun::err = "The following GUIKit runtime error occurred :\n`1`."

GUIRun::nffil = "File not found during `1`."
GUIRunModal::nffil = "File not found during `1`."
GUIResolve::nffil = "File not found during `1`."
GUILoad::nffil = "File not found during `1`."

GUIRun::nvalid = "The GUI definition contains invalid content."
GUIRunModal::nvalid = "The GUI definition contains invalid content."
GUIResolve::nvalid = "The GUI definition contains invalid content."
GUILoad::nvalid = "The GUI definition contains invalid content."

GUIObject::run = "GUIObject is already currently running."
GUIObject::navail = "The GUIObject reference is no longer associated with an active user interface."
GUIObject::nvalid = "The GUI definition resolved to an invalid runtime."

WidgetReference::noref = "The widget reference `1` could not be found."
PropertyValue::noref = "The widget reference `1` could not be found."
SetPropertyValue::noref = "The widget reference `1` could not be found."
InvokeMethod::noref = "The widget reference `1` could not be found."
GUIInformation::noref = "The widget reference `1` could not be found."

Widget::deferr = "`1` is not valid GUI definition content."

(*******************************
   Formatting
 *******************************)

Format[po_GUIObject /; guiObjectQ[po], OutputForm] :=
  "-GUIObject-";

Format[po_GUIObject /; guiObjectQ[po], TextForm] :=
  "-GUIObject-";

GUIObject /: MakeBoxes[po_GUIObject /; guiObjectQ[po], fmt_] :=
  InterpretationBox[ 
    RowBox[{"\[SkeletonIndicator]", "GUIObject", "\[SkeletonIndicator]"}], 
    po];


(*******************************
   GUIRun
 *******************************)

GUIRun[file_?guiResourceQ, args_List:{}, opts___?OptionQ] :=
  Module[{driver, ffile = resolveMathematicaFile[file], relMeth, relMethObj},
    If[ ffile === $Failed,
      Message[GUIRun::nffil, HoldForm[GUIRun[file,opts]] ]; 
      Return[$Failed] ];
    driver = createGUIDriver[GUIRun, args, opts];
    relMeth = ReleaseMethod /. Flatten[{opts}] /. Options[GUIRun];
    relMethObj = If[ relMeth === Manual, GUIKitDriver`RELEASEUMANUAL, GUIKitDriver`RELEASEUONCLOSE];
    createGUIObject[driver @ runFile[ffile, relMethObj], driver]
    ]

GUIRun[doc_ /; Head[doc] === XMLObject["Document"], args_List:{}, opts___?OptionQ] :=
  Module[{str, driver = createGUIDriver[GUIRun, args, opts], relMeth, relMethObj},
    str = ExportString[doc, "XML", "ElementFormatting" -> None];
    relMeth = ReleaseMethod /. Flatten[{opts}] /. Options[GUIRun];
    relMethObj = If[ relMeth === Manual, GUIKitDriver`RELEASEUMANUAL, GUIKitDriver`RELEASEUONCLOSE];
    createGUIObject[driver @ runContent[str, relMethObj], driver] 
    ]
    
GUIRun[ expr_?guiExpressionQ, args_List:{}, opts___?OptionQ] :=
  Module[{sXML},
    sXML = WidgetToSymbolicGUIKitXML[expr];
    If[ sXML === $Failed,
      Message[GUIRun::nvalid];
      Return[$Failed]];
    GUIRun[ sXML, args, opts]
    ]


(* This pattern is the form that GUILoad creates for a delayed execution

   Also see if in this mode the driver could be reused instead of forcing a complete driver shutdown??
   Things like Context where created on load and symbol definitions so a reuse of the
   same context would occur by default.. maybe good or bad
*)

GUIRun[po_GUIObject, args_List:{}, opts___?OptionQ] :=
  Module[{dbg, driver = guiDriver[po], sourceObj = guiRootObject[po], 
    isRunning = False, newSourceObj, relMeth, relMethObj},

    If[ !GUIObjectQ[po],
      Message[GUIObject::navail];
      Return[$Failed];
      ];
      
    isRunning = TrueQ[ driver @ getIsRunning[]];
    relMeth = ReleaseMethod /. Flatten[{opts}] /. Options[GUIRun];
    relMethObj = If[ relMeth === Manual, GUIKitDriver`RELEASEUMANUAL, GUIKitDriver`RELEASEUONCLOSE];
    
    {dbg} = {Debug} /. FilterRules[Flatten[{opts}], Options[GUIRun]] /. Options[GUIRun] /.
      {Debug -> False};

    driver @ setDebug[ TrueQ[dbg]];

    If[ !isRunning,
      (* Here we should setup args on driver *)
      setScriptArguments[driver, args];
      ];
      
    newSourceObj = driver @ execute[sourceObj, relMethObj, !isRunning];
    If[ !SameObjectQ[newSourceObj, sourceObj],
      ReleaseJavaObject[sourceObj];
      ];
    createGUIObject[newSourceObj, driver]
    ]


(*******************************
   GUIRunModal
 *******************************)

(* For displayed things to show without context being shown explicitly *)
GUIKitDecontext[x_] := (x /. s_Symbol :> If[StringMatchQ[Context[s], "GUIKit`Private`Script*"], Symbol[SymbolName[s]], s]);

(* A new feature that allows for checking if interrupts
   were attempted on GUIRunModal windows *)
$CheckModalInterrupts = False;

GUIRunModal[file_?guiResourceQ, args_List:{}, opts___?OptionQ] :=
  Module[{driver, ffile = resolveMathematicaFile[file], attemptModal = True, result, relMeth, relMethObj},
    If[ ffile === $Failed,
      Message[GUIRunModal::nffil, HoldForm[GUIRunModal[file,opts]] ]; 
      Return[$Failed] ];
    driver = createGUIDriver[GUIRunModal, args, opts];
    relMeth = ReleaseMethod /. Flatten[{opts}] /. Options[GUIRunModal];
    relMethObj = If[ relMeth === Manual, GUIKitDriver`RELEASEUMANUAL, GUIKitDriver`RELEASEUONCLOSE];
    attemptModal = driver @ runModalFile[ffile, relMethObj, $CheckModalInterrupts];
    result = If[ TrueQ[attemptModal], 
      DoModal[], 
      Message[GUIRunModal::nvalid];
      $Failed];
    If[ relMeth =!= Manual,
      cleanupGUIDriver[driver];
      ];
    GUIKitDecontext[result]
    ]

GUIRunModal[doc_ /; Head[doc] === XMLObject["Document"], args_List:{}, opts___?OptionQ] :=
  Module[{str, driver = createGUIDriver[GUIRunModal, args, opts], 
      attemptModal = True, result, relMeth, relMethObj},
    str = ExportString[doc, "XML", "ElementFormatting" -> None];
    relMeth = ReleaseMethod /. Flatten[{opts}] /. Options[GUIRunModal];
    relMethObj = If[ relMeth === Manual, GUIKitDriver`RELEASEUMANUAL, GUIKitDriver`RELEASEUONCLOSE];
    attemptModal = driver @ runModalContent[str, relMethObj, $CheckModalInterrupts];
    result = If[ TrueQ[attemptModal], 
      DoModal[], 
      Message[GUIRunModal::nvalid];
      $Failed];
    If[ relMeth =!= Manual,
      cleanupGUIDriver[driver];
      ];
    GUIKitDecontext[result]
    ]
    
GUIRunModal[ expr_?guiExpressionQ, args_List:{}, opts___?OptionQ] :=
  Module[{sXML},
    sXML = WidgetToSymbolicGUIKitXML[expr];
    If[ sXML === $Failed,
      Message[GUIRunModal::nvalid];
      Return[$Failed]];
    GUIRunModal[ sXML, args, opts]
    ]
  

(* This pattern is the form that GUILoad creates for a delayed execution *)

GUIRunModal[po_GUIObject, args_List:{}, opts___?OptionQ] :=
  Module[{dbg, driver = guiDriver[po], sourceObj = guiRootObject[po], newSourceObj,
          closeCode, attemptModal = True, result, isRunning = False, relMeth, relMethObj},
        
    If[ !GUIObjectQ[po],
      Message[GUIObject::navail];
      Return[$Failed];
      ];
      
    isRunning = TrueQ[ driver @ getIsRunning[]];
    relMeth = ReleaseMethod /. Flatten[{opts}] /. Options[GUIRunModal];
    relMethObj = If[ relMeth === Manual, GUIKitDriver`RELEASEUMANUAL, GUIKitDriver`RELEASEUONCLOSE]; 
     
    closeCode = Cases[Flatten[{opts}], 
       HoldPattern[RuleDelayed][ ReturnScript, _] | HoldPattern[Rule][ ReturnScript, _], 1, 1];
    
    {dbg} = {Debug} /. FilterRules[Flatten[{opts}], Options[GUIRunModal]] /. 
       Options[GUIRunModal] /. {Debug -> False};

    driver @ setDebug[ TrueQ[dbg]];
     
    If[ !isRunning,
      If[ closeCode =!= {}, 
         returnCode = Function[x, convertToScriptString[x], {HoldAllComplete}] @@ 
            Extract[closeCode, {1, 2}, Hold];
         driver @ setReturnScript[returnCode] 
         ];

      (* Here we should setup args on driver *)
      setScriptArguments[driver, args];
      ];
      
    attemptModal = driver @ executeModal[sourceObj, relMethObj, !isRunning, $CheckModalInterrupts];

    result = If[ TrueQ[attemptModal], 
      DoModal[], 
      Message[GUIRunModal::nvalid];
      $Failed];
    (* Depending upon mode run or existing system we may not call these always *)
    If[ !isRunning, 
      If[ relMeth =!= Manual,
        cleanupGUIDriver[driver];
        cleanupGUIObject[po];
        ,
        newSourceObj = driver @ lookupObject[ GUIKitDriver`IDUROOTOBJECT ];
        If[ !SameObjectQ[newSourceObj, sourceObj],
           ReleaseJavaObject[newSourceObj] ];
        ];
      GUIKitDecontext[result],
      (* Do we issue warning message when modal called on running non-modal, for now
       since the modal version can't run *)
      Message[GUIObject::run];
      po
      ]
    ]


(*******************************
   GUIResolve

   will immediately resolve to the root object and
   cleanup the driver setup without expecting
   ever to be used with GUIRun so a SharedKernel[]
   JLink state is not used

   User would be responsible for calling ReleaseJavaObject[]
 *******************************)

GUIResolve[file_?guiResourceQ, args_List:{}, opts___?OptionQ] :=
  Module[{driver, io, ffile = resolveMathematicaFile[file], result},
    If[ ffile === $Failed,
      Message[GUIResolve::nffil, HoldForm[GUIResolve[file,opts]] ]; 
      Return[$Failed] ];
    io = GUIObject /. Cases[Flatten[{opts}], HoldPattern[GUIObject->_GUIObject]] /. 
      Cases[Options[GUIResolve], HoldPattern[GUIObject ->_GUIObject]];
    If[ Head[io] === GUIObject,
      If[ guiObjectQ[io],
        driver = guiDriver[io];
        ,
        Message[GUIObject::navail];
        Return[$Failed];
        ];
      ,
      If[ JavaObjectQ[Symbol[$Context <> "Private`BSF`driver"]],
         driver = Symbol[$Context <> "Private`BSF`driver"] ];
      ];
    If[ JavaObjectQ[driver],
      result = driver @ resolveFile[ffile, False];
      ,
      driver = createGUIDriver[GUIResolve, args, opts];
      result = driver @ resolveFile[ffile];
      cleanupGUIDriver[driver];
      ];
    result
    ]

GUIResolve[doc_ /; Head[doc] === XMLObject["Document"], args_List:{}, opts___?OptionQ] :=
  Module[{str, driver, io, result},
    io = GUIObject /. Cases[Flatten[{opts}], HoldPattern[GUIObject->_GUIObject]] /. 
      Cases[Options[GUIResolve], HoldPattern[GUIObject ->_GUIObject]];
    str = ExportString[doc, "XML", "ElementFormatting" -> None];
    If[ Head[io] === GUIObject,
      If[ guiObjectQ[io],
        driver = guiDriver[io];
        ,
        Message[GUIObject::navail];
        Return[$Failed];
        ];
      ,
      If[ JavaObjectQ[Symbol[$Context <> "Private`BSF`driver"]],
         driver = Symbol[$Context <> "Private`BSF`driver"] ];
      ];
    If[ JavaObjectQ[driver],
      result = driver @ resolveContent[str, False];
      ,
      driver = createGUIDriver[GUIResolve, args, opts];
      result = driver @ resolveContent[str];
      cleanupGUIDriver[driver];
      ];
    result
    ]

(* This function works to convert context versions of definition symbols that
   would immediately evaluate, back into the inert GUIKit` versions before
   being able to pass this expression to GUIResolve
*)
Attributes[ConvertResolveContent] = {HoldAllComplete};
Attributes[ConvertResolveOne] = {HoldAllComplete};

ConvertResolveContent[h_, {args___}, context_] := 
  h @@ ConvertResolveOne[{args}, context];
  
ConvertResolveOne[s_Script, context_] := s;

ConvertResolveOne[{args___}, context_] :=
  Map[ Function[{x},ConvertResolveOne[x, context], {HoldAllComplete}], Unevaluated[{args}]]
  
ConvertResolveOne[(h_Symbol)[args___], context_] :=
  (ToExpression["GUIKit`" <> SymbolName[Unevaluated[h]]] @@ 
    ConvertResolveOne[{args}, context]) /; 
      (Context[Unevaluated[h]] === context &&
    MemberQ[{"BindEvent", "Widget", "WidgetReference", 
      "PropertyValue", "SetPropertyValue", "InvokeMethod"}, SymbolName[Unevaluated[h]]] );

ConvertResolveOne[h_[args___], context_] :=
  ConvertResolveOne[h, context] @@ ConvertResolveOne[{args}, context];
  
ConvertResolveOne[x_, content_] := x


GUIResolve[expr_?guiExpressionQ, args_List:{}, opts___?OptionQ] :=
  Module[{sXML},
    sXML = WidgetToSymbolicGUIKitXML[expr];
    If[ sXML === $Failed,
      Message[GUIResolve::nvalid];
      Return[$Failed]];
    GUIResolve[ sXML, args, opts]
    ]


(*******************************
   GUIResolveObject

   NOTE: currently this is not used anywhere, but one
   potential use is to have scripts and GUIObjects use
   ResolveObject instead of Resolve in places where objects
    are dynamically created so that GUITypedObject properties
    are preserved.  This however would mean not getting lower level
      JavaObjects in scripts which may normally not be all that
      desirable
      
   will immediately resolve to the root GUITypedObject and
   cleanup the driver setup without expecting
   ever to be used with GUIRun so a SharedKernel[]
   JLink state is not used

   Internal Private` version of GUIResolve that
     returns the full GUITypedObject and not just
     the underlying JavaObject
 *******************************)

GUIResolveObject[file_?guiResourceQ, args_List:{}, opts___?OptionQ] :=
  Module[{driver, io, ffile = resolveMathematicaFile[file], result},
    If[ ffile === $Failed,
      Message[GUIResolve::nffil, HoldForm[GUIResolve[file,opts]] ]; 
      Return[$Failed] ];
    io = GUIObject /. Cases[Flatten[{opts}], HoldPattern[GUIObject->_GUIObject]] /. 
      Cases[Options[GUIResolve], HoldPattern[GUIObject ->_GUIObject]];
    If[ Head[io] === GUIObject,
      If[ guiObjectQ[io],
        driver = guiDriver[io];
        ,
        Message[GUIObject::navail];
        Return[$Failed];
        ];
      ,
      If[ JavaObjectQ[Symbol[$Context <> "Private`BSF`driver"]],
         driver = Symbol[$Context <> "Private`BSF`driver"] ];
      ];
    If[ JavaObjectQ[driver],
      result = driver @ resolveFileObject[ffile, False];
      ,
      driver = createGUIDriver[GUIResolve, args, opts];
      result = driver @ resolveFileObject[ffile];
      cleanupGUIDriver[driver];
      ];
    result
    ]

GUIResolveObject[doc_ /; Head[doc] === XMLObject["Document"], args_List:{}, opts___?OptionQ] :=
  Module[{str, driver, io, result},
    io = GUIObject /. Cases[Flatten[{opts}], HoldPattern[GUIObject->_GUIObject]] /. 
      Cases[Options[GUIResolve], HoldPattern[GUIObject ->_GUIObject]];
    str = ExportString[doc, "XML", "ElementFormatting" -> None];
    If[ Head[io] === GUIObject,
      If[ guiObjectQ[io],
        driver = guiDriver[io];
        ,
        Message[GUIObject::navail];
        Return[$Failed];
        ];
      ,
      If[ JavaObjectQ[Symbol[$Context <> "Private`BSF`driver"]],
         driver = Symbol[$Context <> "Private`BSF`driver"] ];
      ];
    If[ JavaObjectQ[driver],
      result = driver @ resolveContentObject[str, False];
      ,
      driver = createGUIDriver[GUIResolve, args, opts];
      result = driver @ resolveContentObject[str];
      cleanupGUIDriver[driver];
      ];
    result
    ]
    
GUIResolveObject[expr_?guiExpressionQ, args_List:{}, opts___?OptionQ] :=
  Module[{sXML},
    sXML = WidgetToSymbolicGUIKitXML[expr];
    If[ sXML === $Failed,
      Message[GUIResolve::nvalid];
      Return[$Failed]];
    GUIResolveObject[ sXML, args, opts]
    ]



(*******************************
   GUILoad
 *******************************)


GUILoad[file_?guiResourceQ, args_List:{}, opts___?OptionQ] :=
  Module[{driver, ffile = resolveMathematicaFile[file]},
    If[ ffile === $Failed,
      Message[GUILoad::nffil, HoldForm[GUILoad[file,opts]] ]; 
      Return[$Failed] ];
    driver = createGUIDriver[GUILoad, args, opts];
    createGUIObject[driver @ loadFile[ffile], driver]
    ]

GUILoad[doc_ /; Head[doc] === XMLObject["Document"], args_List:{}, opts___?OptionQ] :=
  Module[{str, driver = createGUIDriver[GUILoad, args, opts]},
    str = ExportString[doc, "XML", "ElementFormatting" -> None];
    createGUIObject[driver @ loadContent[str], driver]
    ]
    
GUILoad[expr_?guiExpressionQ, args_List:{}, opts___?OptionQ] :=
  Module[{sXML},
    sXML = WidgetToSymbolicGUIKitXML[expr];
    If[ sXML === $Failed,
      Message[GUILoad::nvalid];
      Return[$Failed]];
    GUILoad[ sXML, args, opts]
    ]
    

(*******************************
   ReleaseGUIObject
 *******************************)

 ReleaseGUIObject[po_GUIObject, opts___?OptionQ] :=
  Module[{dbg, driver = guiDriver[po], rt},

    (* Since we are asking for release we can silently return without a message *)
    If[ !GUIObjectQ[po],
      Return[];
      ];
      
    {dbg} = {Debug} /. FilterRules[Flatten[{opts}], Options[ReleaseGUIObject]] /. Options[ReleaseGUIObject] /.
      {Debug -> False};

    driver @ setDebug[ TrueQ[dbg]];
 
    rt = guiRootObject[po];
    If[ !JavaObjectQ[rt],
      Message[GUIObject::navail];
      Return[$Failed];
      ];
    driver @ requestRelease[ rt];

    cleanupGUIDriver[driver];
    cleanupGUIObject[po];
    ]

(*******************************
   CloseGUIObject
 *******************************)

 CloseGUIObject[po_GUIObject, opts___?OptionQ] :=
  Module[{dbg, driver = guiDriver[po], rt},

    (* Release request should definitely not issue message but it is debateable
       whether Close should also silently return?? *)
    If[ !GUIObjectQ[po],
      Return[];
      ];
      
    {dbg} = {Debug} /. FilterRules[Flatten[{opts}], Options[CloseGUIObject]] /. Options[CloseGUIObject] /.
      {Debug -> False};

    driver @ setDebug[ TrueQ[dbg]];
 
    rt = guiRootObject[po];
		If[ !JavaObjectQ[rt],
		  Message[GUIObject::navail];
		  Return[$Failed];
      ];
    driver @ requestClose[ rt];
    ]
    
 (*******************************
    GUIObject variants of GUIKit definition functions
    
    Use with GUIObject causes 
    evaluation, other forms when used in definitions
    are left unevaluated
    
    TODO think about whether we can make this left associative
    like JavaObject use so parenthesis will not be needed
    when mixing with Java method calls
   
    We may remove most of these and make
    GUIResolve[..., GUIObject -> ref]
    the preferred method though this limits
    use of existing JavaObjects outside of Script[]
 *******************************)

WidgetReference /: (po_GUIObject)[ WidgetReference[args__]] :=
   Block[{result},
  If[ GUIObjectQ[po],
    result = JavaBlock[guiWrapWidgetReference[guiDriver[po], args]],
    Message[GUIObject::navail];
    result = $Failed;
    ];
  result /; Head[result] =!= guiWrapWidgetReference
  ];
  
SetWidgetReference /: (po_GUIObject)[ SetWidgetReference[args__]] :=
  Block[{result},
  If[ GUIObjectQ[po],
    result = JavaBlock[guiWrapSetWidgetReference[guiDriver[po], args]],
    Message[GUIObject::navail];
    result = $Failed;
    ];
   result /; Head[result] =!= guiWrapSetWidgetReference
  ];
  
UnsetWidgetReference /: (po_GUIObject)[ UnsetWidgetReference[args__]] :=
  Block[{result},
  If[ GUIObjectQ[po],
    result = JavaBlock[guiWrapUnsetWidgetReference[guiDriver[po], args]],
    Message[GUIObject::navail];
    result = $Failed;
    ];
   result /; Head[result] =!= guiWrapUnsetWidgetReference
  ];
  
PropertyValue /: (po_GUIObject)[ PropertyValue[args__]] :=
  Block[{result},
  If[ GUIObjectQ[po],
    result = JavaBlock[guiWrapPropertyValue[guiDriver[po], args]],
    Message[GUIObject::navail];
    result = $Failed;
    ];
   result /; Head[result] =!= guiWrapPropertyValue
  ];
  
SetPropertyValue /: (po_GUIObject)[ SetPropertyValue[args__]] :=
  Block[{result},
  If[ GUIObjectQ[po],
    result = JavaBlock[guiWrapSetPropertyValue[guiDriver[po], args]],
    Message[GUIObject::navail];
    result = $Failed;
    ];
   result /; Head[result] =!= guiWrapSetPropertyValue
  ];
  
(* We might want to disable this version shorthand as some users
   found it confusing using this with GUIObject will not work
   in Script blocks.  So we might only allow Rule shorthand in definitions themselves
 *)
GUIObject /: (po_GUIObject)[ (Rule | RuleDelayed)[lhs_, rhs_]] :=
  Block[{result},
  If[ GUIObjectQ[po],
    result = JavaBlock[guiWrapSetPropertyValue[guiDriver[po], lhs, rhs]],
    Message[GUIObject::navail];
    result = $Failed;
    ];
   result /; Head[result] =!= guiWrapSetPropertyValue
  ];
  
InvokeMethod /: (po_GUIObject)[ InvokeMethod[args__]] :=
  Block[{result},
  If[ GUIObjectQ[po],
    result = JavaBlock[guiWrapInvokeMethod[guiDriver[po], args]],
    Message[GUIObject::navail];
    result = $Failed;
    ];
   result /; Head[result] =!= guiWrapInvokeMethod
  ];
  
BindEvent /: (po_GUIObject)[ BindEvent[args__]] :=
  If[ GUIObjectQ[po],
    GUIResolve[ BindEvent[args], GUIObject -> po],
    Message[GUIObject::navail];
    Return[$Failed];
    ];

Script /: (po_GUIObject)[ Script[args__]] :=
  If[ GUIObjectQ[po],
    GUIResolve[ Script[args], GUIObject -> po],
    Message[GUIObject::navail];
    Return[$Failed];
    ];

Widget /: (po_GUIObject)[ Widget[args__]] :=
  If[ GUIObjectQ[po],
    GUIResolve[ Widget[args], GUIObject -> po],
    Message[GUIObject::navail];
    Return[$Failed];
    ];

GUIInformation /: GUIInformation[po_GUIObject, args___] :=
  Block[{result},
  If[ GUIObjectQ[po],
    result = JavaBlock[guiWrapGUIInformation[guiDriver[po], args]],
    Message[GUIObject::navail];
    result = $Failed;
    ];
   result /; Head[result] =!= guiWrapGUIInformation
  ];

GUIInformation /: (po_GUIObject)[ GUIInformation[args__]] :=
  Block[{result},
  If[ GUIObjectQ[po],
    result = JavaBlock[guiWrapGUIInformation[guiDriver[po], args]],
    Message[GUIObject::navail];
    result = $Failed;
    ];
   result /; Head[result] =!= guiWrapGUIInformation
  ];
  
  
(*******************************
   Utility functions
 *******************************)
 
 initGUI[] := (
    InstallJava[];
    LoadJavaClass["com.wolfram.guikit.GUIKitUtils"];
    );
    
 createGUIDriver[func_, args_List:{}, opts___?OptionQ] :=
    Module[{driver, dbg, closeCode, cntxt, addContexts},
     initGUI[];
     
     closeCode = Cases[Flatten[{opts}], 
       HoldPattern[RuleDelayed][ ReturnScript, _] | HoldPattern[Rule][ ReturnScript, _], 1, 1];
     
     {dbg, cntxt, addContexts} = {Debug, Context, IncludedScriptContexts} /.
       FilterRules[Flatten[{opts}], Options[func]] /. Options[func] /.
         {Debug -> False, Context -> Automatic, IncludedScriptContexts -> {}};
 
     driver = JavaNew["com.wolfram.guikit.GUIKitDriver"];
 
     driver @ setDebug[ TrueQ[dbg]];
     If[ cntxt =!= Automatic && StringQ[cntxt], driver @ setContext[cntxt]];
     (* Currently we do not expose the linkcommandline setting from Mathematica *)
     (* If[ lnkCmdLne =!= Automatic && StringQ[lnkCmdLne], driver @ setLinkCommandLine[lnkCmdLne]]; *)
     
     If[ closeCode =!= {}, 
         returnCode = Function[x, convertToScriptString[x], {HoldAllComplete}] @@ 
            Extract[closeCode, {1, 2}, Hold];
         driver @ setReturnScript[returnCode] 
         ];
 
     (* When the driver is used in Mathematica we benefit from having Java windows displayed with JavaShow[] *)
     driver @ setUseJavaShow[True];
 
     addContexts = Flatten[{addContexts}];
     If[ MatchQ[addContexts, {___String}],
       driver @ setAdditionalScriptContexts[addContexts] ];
     
     (* Setup any script args that might exist *)
     setScriptArguments[driver, args];
 
     driver
     ]
    
 cleanupGUIDriver[driver_, opts___?OptionQ] :=
   ReleaseJavaObject[driver];
 
 
 createGUIObject[$Failed, __] := $Failed
 
 createGUIObject[sourceObj_, driver_] :=
  Module[{},
    If[!JavaObjectQ[driver],
      Message[GUIObject::nvalid];
      Return[$Failed]
      ];
    GUIObject[sourceObj, driver]
    ]
   
 createGUIObject[___] := $Failed
 
 (* we do not use guiObjectQ as driver may already have been Removed[] *)
 cleanupGUIObject[ GUIObject[rootObj_?JavaObjectQ, driver_], opts___?OptionQ] :=
   ReleaseJavaObject[ rootObj];

 (* Keep this simple if we can *)
 
 GUIObjectQ[expr___] := guiObjectQ[expr]
 
 guiObjectQ[ GUIObject[ rootObj_, obj_?JavaObjectQ] ] := True
 guiObjectQ[___] := False
 
 guiDriver[ propObjRef_?guiObjectQ] := Part[propObjRef, 2]
 guiDriver[___] := None
 
 guiRootObject[ propObjRef_?guiObjectQ] := Part[propObjRef, 1]
 guiRootObject[___] := None
 
 createJavaObjectArgument[expr_] :=
   Block[{result},
     Off[MakeJavaObject::arg];
     result = toTargetObject[expr];
     On[MakeJavaObject::arg];
     If[ JavaObjectQ[result], result, 
       Block[{$ContextPath = {"Global`", "System`", "GUIKit`", "JLink`"}},
         MakeJavaExpr[expr]
         ]
       ]
   ]
   
 (* This needs to send an array of GUI TypedObjects[] to remove the Vector util method *)
 setScriptArguments[ driver_, args_List] :=
   Module[{l, objs, objsQ},
     If[ Length[args] <= 0, Return[]];
     l = JavaNew["java.util.Vector"];
     objsQ = JavaObjectQ /@ args;
     objs = createJavaObjectArgument /@ args;
     (l @ add[#])& /@ objs;
     driver @ setScriptArguments[l];
     MapThread[ If[!#1, ReleaseJavaObject[#2]]&, {objsQ, objs}];
     ReleaseJavaObject[l];
     ]

 
(* End not called in subpackages *)
(* EndPackage not called in subpackages *)

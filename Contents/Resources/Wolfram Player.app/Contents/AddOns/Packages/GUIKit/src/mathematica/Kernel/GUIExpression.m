
(* GUIExpression subpackage *)

(* :Context: GUIKit` *)

(* :Copyright: Copyright 2004, Wolfram Research, Inc. *)

(* Each subpackage is called within `Private` *)

(*******************************
   Options
 *******************************)

Options[Widget] = {
  InitialArguments -> None,
  Name -> None,
  ExposeWidgetReferences -> Automatic,
  WidgetLayout -> Automatic
  };

Options[WidgetReference] = {
  WidgetLayout -> Automatic
  };
  
Options[WidgetGroup] = {
  Name -> None,
  WidgetLayout -> Automatic
  };
  
(* NOTE: WidgetLayout may need to exist on all other functions 
   that potentially my return an object instance to allow for
   prevention of a widget to automatically get added to a layout
 *)
 
Options[PropertyValue] = {
  Name -> None,
  InvokeThread -> "Current",
  InvokeWait -> Automatic  (* Automatic | True | False *)
  };
   
Options[SetPropertyValue] = {
  Name -> None,
  InvokeThread -> "Current",
  InvokeWait -> Automatic  (* Automatic | True | False *)
  };
  
Attributes[Script] = {HoldAllComplete}

Options[Script] = {
  Language -> Automatic,
  ScriptSource -> None
  };
  
Options[InvokeMethod] = {
  Name -> None,
  InvokeThread -> "Current",
  InvokeWait -> Automatic  (* Automatic | True | False *)
  };
  
Attributes[BindEvent] = {HoldRest};

Options[BindEvent] = {
  Name -> None,
  InvokeThread -> "Current",
  InvokeWait -> Automatic  (* Automatic | True | False *)
  };
  

(*******************************
   guiWrap Functions
   
   Listable on target and name
   wraps calls for Property*Value, SetProperty*Value from 
     Java engine
 *******************************)

$GUIUtilsClassName = "com.wolfram.guikit.GUIKitUtils";
  
(* Adds listable target names *)
guiWrapPropertyValue[ bsfFunc_?JavaObjectQ, {target_List, args__}, others___] :=
  guiWrapPropertyValue[bsfFunc, {#,args}, others]& /@ target;
  
(* Mathematica and XML is 1-based, Java is 0-based *)

guiWrapPropertyValue[ bsfFunc_?JavaObjectQ, {target_, name_, index_:All}, opts___?extendedOptionQ] :=
   Module[{pt = MakeJavaObject[If[ index === All, Null, index -1]], result, id, invThread, invWait},
    checkTarget = Not[JavaObjectQ[target]];
    (* Supports check for Name option and register this object with the id *)
    {id, invThread, invWait} = {"Name", InvokeThread, InvokeWait} /. 
      canonicalOptions[Flatten[{opts}]] /. canonicalOptions[Options[PropertyValue]];
    id = MakeJavaObject[ If[ id =!= None && id =!= Null, ToString[id], Null]];
    result = If[ Head[name] === List,
      GUIKitUtils`resolveAndGetObjectProperty[ bsfFunc, toTargetObject[target], ToString[#], pt, id, 
        checkTarget, ToString[invThread], ToString[invWait] ]& /@ name,
      GUIKitUtils`resolveAndGetObjectProperty[ bsfFunc, toTargetObject[target], ToString[name], pt, id, 
        checkTarget, ToString[invThread], ToString[invWait] ]
      ];
    If[ result === $Failed || MemberQ[result, $Failed], Return[$Failed]];
    result
    ];

 
(* Adds listable target names *)
guiWrapSetPropertyValue[ bsfFunc_?JavaObjectQ, {target_List, args__}, others__] :=
  guiWrapSetPropertyValue[bsfFunc, {#, args}, others]& /@ target;
  
(* Mathematica and XML is 1-based, Java is 0-based *)

guiWrapSetPropertyValue[ bsfFunc_?JavaObjectQ, 
    {target_, name_, index_:All}, val_, opts___?extendedOptionQ] :=
   Module[{pt = MakeJavaObject[If[ index === All, Null, index - 1]], wasValObj = False,
      valObj, id, invThread, invWait, checkTarget = True},
    checkTarget = Not[JavaObjectQ[target]];
    wasValObj = JavaObjectQ[val];
    valObj = createJavaObjectArgument[val];
    (* Supports check for Name option and register this object with the id *)
    {id, invThread, invWait} = {"Name", InvokeThread, InvokeWait} /. 
      canonicalOptions[Flatten[{opts}]] /. canonicalOptions[Options[SetPropertyValue]];
    id = MakeJavaObject[ If[ id =!= None && id =!= Null, ToString[id], Null]];
    If[ Head[name] === List,
      GUIKitUtils`resolveAndSetObjectProperty[ bsfFunc, toTargetObject[target], ToString[#], pt, valObj, id, 
        checkTarget, ToString[invThread], ToString[invWait]]& /@ name,
      GUIKitUtils`resolveAndSetObjectProperty[ bsfFunc, toTargetObject[target], ToString[name], pt, valObj, id,
        checkTarget, ToString[invThread], ToString[invWait]]
      ];
    If[ !wasValObj, ReleaseJavaObject[valObj]];
    ];

(* Adds listable target names *)
guiWrapInvokeMethod[ bsfFunc_?JavaObjectQ, {target_List, args__}, others___] :=
  guiWrapInvokeMethod[bsfFunc, {#,args}, others]& /@ target
  
guiWrapInvokeMethod[ bsfFunc_?JavaObjectQ, {target_, name_}, contents___?(!invokeOrBindOptionQ[#]&), opts___?invokeOrBindOptionQ] :=
   Module[{result, id, aobjs = Null, len = Length[{contents}], checkTarget = True, invThread, invWait,
      contentObjsQ, contentObjs},
    checkTarget = Not[JavaObjectQ[target]];
    
    (* Supports check for Name option and register this object with the id *)
    {id, invThread, invWait} = {"Name", InvokeThread, InvokeWait} /. 
      canonicalOptions[Flatten[{opts}]] /. canonicalOptions[Options[InvokeMethod]];
    (* TODO see if we can speed up InvokeMethod by not creating this Vector instance *)
    If[ len > 0,
      aobjs = JavaNew["java.util.Vector", len];
      contentObjsQ = JavaObjectQ /@ {contents};
      contentObjs = createJavaObjectArgument /@ {contents};
      (aobjs@add[#]&) /@ contentObjs;
      ];
    id = MakeJavaObject[ If[ id =!= None && id =!= Null, ToString[id], Null]];
    result = If[ Head[name] === List,
      GUIKitUtils`resolveAndInvokeObjectMethodName[bsfFunc, ToString[#], toTargetObject[target], aobjs, id,
        checkTarget, ToString[invThread], ToString[invWait]]& /@ name,
      GUIKitUtils`resolveAndInvokeObjectMethodName[bsfFunc, ToString[name], toTargetObject[target], aobjs, id,
        checkTarget, ToString[invThread], ToString[invWait]]
      ];
    If[ len > 0,
      MapThread[ If[!#1, ReleaseJavaObject[#2]]&, {contentObjsQ, contentObjs}];
      ];
    If[ result === $Failed || MemberQ[result, $Failed], Return[$Failed]];
    result
    ];
  
(* GUIInformation needs to support the following arguments
   "WidgetNames", "PropertyNames", "EventNames", "MethodNames", All
*)

(* For now ignores choices for info since only "WidgetNames" exists for GUIObject *)
guiWrapGUIInformation[ bsfFunc_?JavaObjectQ] :=
  Module[{result},
    result = GUIKitUtils`resolveAndGetWidgetNames[bsfFunc];
    If[ result === $Failed, Return[guiWrapGUIInformation[]]];
    "WidgetNames" -> Union[result]
    ];
  
guiWrapGUIInformation[ bsfFunc_?JavaObjectQ, target_GUIObject, others___] :=
  guiWrapGUIInformation[bsfFunc, others];
  
allChoicesList = {"MethodNames", "EventNames", "PropertyNames"};

guiWrapGUIInformation[ bsfFunc_?JavaObjectQ, target_, choices___?(!OptionQ[#]&), opts___?OptionQ] :=
  Module[{checkTarget, useChoices},
    checkTarget = Not[JavaObjectQ[target]];
    If[ TrueQ[checkTarget] && !StringQ[target], Return[guiWrapGUIInformation[]]];
    
    useChoices = Flatten[{choices}];
    useChoices = Union[ Flatten[{useChoices /. 
      All -> allChoicesList}]];
    If[ useChoices === {}, useChoices = allChoicesList];
    (* ToString called to strip linear syntax from possible names *)
    DeleteCases[ Flatten[{handleOneInformationChoice[bsfFunc, If[ StringQ[target], ToString[target], target], checkTarget, #]& /@ useChoices}], $Failed]
    ];
    
    
handleOneInformationChoice[bsfFunc_, target_, checkTarget_, "WidgetNames"] :=
  Module[{result},
    result = GUIKitUtils`resolveAndGetWidgetNames[bsfFunc, True];
    If[ result === $Failed, Return[$Failed]];
    "WidgetNames" -> Union[result]
    ];
    
handleOneInformationChoice[bsfFunc_, target_, checkTarget_, "PropertyNames"] :=
  Module[{result},
    result = GUIKitUtils`resolveAndGetObjectProperties[ bsfFunc, toTargetObject[target], checkTarget];
    If[ result === $Failed, Return[$Failed]];
    "PropertyNames" -> Union[result]
    ];
 
handleOneInformationChoice[bsfFunc_, target_, checkTarget_, "EventNames"] :=
  Module[{result},
    result = GUIKitUtils`resolveAndGetObjectEvents[ bsfFunc, toTargetObject[target], checkTarget];
    If[ result === $Failed, Return[$Failed]];
    "EventNames" -> Union[result]
    ];
    
handleOneInformationChoice[bsfFunc_, target_, checkTarget_, "MethodNames"] :=
  Module[{result},
    result = GUIKitUtils`resolveAndGetObjectMethodNames[ bsfFunc, toTargetObject[target], checkTarget];
    If[ result === $Failed, Return[$Failed]];
    "MethodNames" -> Union[result]
    ];
    
handleOneInformationChoice[bsfFunc_, target_, checkTarget_, "SetPropertyValuePatterns"] :=
  Module[{result},
    result = GUIKitUtils`resolveAndGetObjectSetPropertyValuePatterns[ bsfFunc, toTargetObject[target], 
      checkTarget, False];
    If[ result === $Failed, Return[$Failed]];
    "SetPropertyValuePatterns" -> Union[result]
    ];
    
handleOneInformationChoice[bsfFunc_, target_, checkTarget_, "PropertyValuePatterns"] :=
  Module[{result},
    result = GUIKitUtils`resolveAndGetObjectPropertyValuePatterns[ bsfFunc, toTargetObject[target], 
      checkTarget, False];
    If[ result === $Failed, Return[$Failed]];
    "PropertyValuePatterns" -> Union[result]
    ];
    
handleOneInformationChoice[bsfFunc_, target_, checkTarget_, "InvokeMethodPatterns"] :=
  Module[{result},
    result = GUIKitUtils`resolveAndGetObjectInvokeMethodPatterns[ bsfFunc, toTargetObject[target], 
      checkTarget, False];
    If[ result === $Failed, Return[$Failed]];
    "InvokeMethodPatterns" -> Union[result]
    ];
    
handleOneInformationChoice[bsfFunc_, target_, checkTarget_, "VerboseSetPropertyValuePatterns"] :=
  Module[{result},
    result = GUIKitUtils`resolveAndGetObjectSetPropertyValuePatterns[ bsfFunc, toTargetObject[target], 
      checkTarget, True];
    If[ result === $Failed, Return[$Failed]];
    "VerboseSetPropertyValuePatterns" -> Union[result]
    ];
    
handleOneInformationChoice[bsfFunc_, target_, checkTarget_, "VerbosePropertyValuePatterns"] :=
  Module[{result},
    result = GUIKitUtils`resolveAndGetObjectPropertyValuePatterns[ bsfFunc, toTargetObject[target], 
      checkTarget, True];
    If[ result === $Failed, Return[$Failed]];
    "VerbosePropertyValuePatterns" -> Union[result]
    ];
    
handleOneInformationChoice[bsfFunc_, target_, checkTarget_, "VerboseInvokeMethodPatterns"] :=
  Module[{result},
    result = GUIKitUtils`resolveAndGetObjectInvokeMethodPatterns[ bsfFunc, toTargetObject[target], 
      checkTarget, True];
    If[ result === $Failed, Return[$Failed]];
    "VerboseInvokeMethodPatterns" -> Union[result]
    ];
    
handleOneInformationChoice[___] := {};


(* Adds listable target names *)
guiWrapWidgetReference[ bsfFunc_?JavaObjectQ, {ids__String}, others___] :=
  guiWrapWidgetReference[bsfFunc, #, others]& /@ {ids};
  
guiWrapWidgetReference[ bsfFunc_?JavaObjectQ, id_String, opts___?extendedOptionQ] :=
  GUIKitUtils`resolveAndGetReference[bsfFunc, toTargetObject[id]];
    
guiWrapSetWidgetReference[ bsfFunc_?JavaObjectQ, id_String, val_, opts___?extendedOptionQ] :=
  Module[{valObjQ, valObj},
    valObjQ = JavaObjectQ[val];
    valObj = createJavaObjectArgument[val];
    GUIKitUtils`resolveAndSetReference[bsfFunc, toTargetObject[id], valObj];
    If[ !valObjQ, ReleaseJavaObject[valObj]];
    ];
    
(* Adds listable target names *)
guiWrapUnsetWidgetReference[ bsfFunc_?JavaObjectQ, {ids__String}, others___] :=
  guiWrapUnsetWidgetReference[bsfFunc, #, others]& /@ {ids};
  
guiWrapUnsetWidgetReference[ bsfFunc_?JavaObjectQ, id_String, opts___?extendedOptionQ] :=
  GUIKitUtils`resolveAndUnsetReference[bsfFunc, toTargetObject[id]];
    

guiWrapCloseGUIObject[ bsfFunc_?JavaObjectQ, others___] :=
  GUIKitUtils`resolveAndCloseGUIObject[bsfFunc];
  
guiWrapGUIObject[ bsfFunc_?JavaObjectQ] :=
  GUIObject[bsfFunc @ lookupBean[GUIKitDriver`IDUROOTOBJECT ], bsfFunc];
  
guiWrapGUIObject[ source_, driver_] :=
  GUIObject[source, driver];

(* End not called in subpackages *)
(* EndPackage not called in subpackages *)

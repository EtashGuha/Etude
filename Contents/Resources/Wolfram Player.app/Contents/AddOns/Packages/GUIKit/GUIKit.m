
(* :Title: GUIKit *)

(* :Context: GUIKit` *)

(* :Summary: GUI expression API for Mathematica  *)

(* :Keywords: GUIKit GUI Widget Bean Scripting Framework Java *)

(* :Copyright: Copyright 2004, Wolfram Research, Inc. *)

(*******************************************)
BeginPackage["GUIKit`", {"JLink`"}]
(*******************************************)

(*******  Information Context  *****************)

(* Programmers can use these values (using their full context, as in
   GUIKit`Information`$ReleaseNumber)
   to test version information about a user's GUIKit installation.
*)

`Information`$VersionNumber = 1.1
`Information`$ReleaseNumber = 0
`Information`$Version = "GUIKit Version 1.1.0 (December, 2006)"
`Information`$CreationID;
`Information`$CreationDate;
  
(*******************************
   Usage Messages
 *******************************)

(* Common *)

$GUIPath::usage = "$GUIPath defines a search path for finding any public GUI definitions
 within a distribution including both System and user AddOns."

(* GUIExpression *)

Widget::usage = "Widget[src, {content}, opts] defines a new object instance created at runtime."
WidgetReference::usage = "WidgetReference[\"name\"] looks up a reference to an existing object instance registered with name \"name\"."
WidgetGroup::usage = "WidgetGroup[elems, opts] represents a grouped set of widgets which an optional shared set of layout options as well as option values that control how child widgets are layed out together."

(* TODO may need to force the use of strings here? *)
Tabs::usage = "Tabs is an option value for WidgetLayout."
WidgetLayout::usage = "WidgetLayout is an option of Widget, WidgetReference and WidgetGroup which contains suboptions controlling layout."
WidgetSpace::usage = "WidgetSpace[n] is a spacing placeholder useful for widget layout when spacing is needed between widgets."
WidgetFill::usage = "WidgetFill[] is a placeholder useful for widget layout when adjacent widgets should not dominate the layout space."
WidgetAlign::usage = "WidgetAlign[] is a placeholder useful for widget layout to provide a marker that other WidgetAlign[] markers will align with if present.
 WidgetAlign[{\"ref\", After}, Before] makes a specific request for the next widget to align to another widget by named reference ref, and aligned based on Before or After
  specifications on both widgets."
 
(*
 Temporary changes while the new graph property mechanism is hooked up.
*)
Unprotect[ PropertyValue]
Off[ PropertyValue::argr]
PropertyValue::usage = "PropertyValue[\"name\"] returns the property value of the current contexted widget identified by name. 
 PropertyValue[{target, \"name\"}] returns the name property of the target widget instance."

SetPropertyValue::usage = "SetPropertyValue[\"name\", val] sets the property value of the current contexted widget identified by name to val. 
 SetPropertyValue[{target, \"name\"}, val] sets the name property of the target widget instance."

BindEvent::usage = "BindEvent[\"name\", script] binds the execution of script whenever the name event occurs on the contexted widget.
 BindEvent[{\"target\", \"name\"}, script] binds to the specified widget target."

Script::usage = "Script[exprs] defines a Mathematica script code block which can be used in Widget definitions or used with BindEvent to execute
when certain events occur."

InvokeMethod::usage = "InvokeMethod[\"name\", args] calls the method identified by name with possible arguments on the current contexted object. 
InvokeMethod[{target, \"name\"}, args] makes the method call on the widget instance identified by target."

GUIInformation::usage = "GUIInformation[obj] gives information about the GUI or widget represented by the specified object or name. GUIInformation[obj, name] gives information about a specific named feature of the specified object."

Name::usage = "Name is an option to GUIKit expression functions and identifies a string name to register this widget instance with in the active widget registry. This widget can then be found with WidgetReference[\"name\"]."
InitialArguments::usage = "InitialArguments is an option to Widget which specifies the argument list to use when instantiating this widget definition."
ExposeWidgetReferences::usage = "ExposeWidgetReferences is an option to Widget which specifies a list of normally internal children widget reference names to include as publically available instances."
ScriptSource::usage = "ScriptSource is an option to Script which specifies a location to load the script code from."

InvokeThread::usage = "InvokeThread is an option to GUIKit symbols which specifies what thread the invocation shoould execute on."
InvokeWait::usage = "InvokeWait is an option to GUIKit symbols which determines if a threaded invocation should wait for a result or continue on immediately without a response."

(* GUIRuntime *)

GUIRun::usage = "GUIRun[file] loads, parses and runs a widget definition file.
GUIRun[xml] executes a SymbolicXML reresentation of the widget definition."

GUIRunModal::usage = "GUIRunModal[file] loads, parses and runs a widget definition file in a modal session.
GUIRunModal[xml] executes a SymbolicXML reresentation of the widget definition."

GUILoad::usage = "GUILoad[file] loads and preprocesses the widget definition source and returns
an object that can later be given to GUIRun or
alternatively ReleaseGUIObject to remove any of its resources.
GUILoad allows for delayed execution of the result. This could be useful for
preloading and creating all the resources and classes and only delaying when the
GUI is shown to the user."

ReleaseGUIObject::usage = "ReleaseGUIObject[obj] unloads the resources of obj that was created with an
initial call to GUILoad. ReleaseGUIObject must be used with any objects created by GUILoad that were
never given to GUIRun to execute."

CloseGUIObject::usage = "CloseGUIObject[obj] requests that the user interface represented by the
GUI object close its execution. Unlike ReleaseGUIObject[obj] which forces complete termination and clearing of resources, CloseGUIObject
 only initiates the visual closing of the user interface, and depending upon the independently set ReleaseMethod the user interface
 will either additionally cleanup or stay active for further use."

GUIResolve::usage = "GUIResolve[file] loads the widget definition source and returns the root
JavaObjectReference from processing the widget definition to completion.
It does not execute the result even if it is a graphical user interface widget.
This could be useful for creating Java objects that may not be graphical user interface
widgets or objects that necessarily ever execute as an application."

GUIObject::usage = "GUIObject[rootObject, driver] represents a parsed and hydrated GUIKit
definition that is ready to be executed with either GUIRun or GUIRunModal.  This expression may also be removed with ReleaseGUIObject or asked
to visually end with CloseGUIObject."

GUIObjectQ::usage = "GUIObjectQ[obj] yields True if the expression obj represents a live GUIObject, and yields False otherwise."

ReleaseMethod::usage = "ReleaseMethod is an option to GUIRun and related functions that determines when an GUIObject is released. Automatic will
 setup the user interface to be removed from the system on user interface closing, while Manual will require calling ReleaseGUIObject to remove the definition from the system.
 Manual allows for reuse of the same GUIObject instance and using CloseGUIObject will not clear up any of the user interface definitons but just visually hide the user interface."
 
IncludedScriptContexts::usage = "IncludedScriptContexts is an option to GUIRun and related functions that can extend the active $ContextPath that Script blocks within a definition will use."
 
ReturnScript::usage = "ReturnScript is an option to GUIRunModal which specifies a string version
 of Mathematica code to execute when the window ends its modal session and becomes the return value of the RunModal call."

SetWidgetReference::usage = "SetWidgetReference[\"name\", obj] registers a new widget with user interface runtime using the given name."
UnsetWidgetReference::usage = "UnsetWidgetReference[\"name\"] unregisters a widget registered originally using name."

(* XML conversions *)

WidgetToSymbolicGUIKitXML::usage = "WidgetToSymbolicGUIKitXML[expr, options] converts a widget definition into its equivalent SymbolicXML GUIKitXML definition."
SymbolicGUIKitXMLToWidget::usage = "SymbolicGUIKitXMLToWidget[expr, options] converts a SymbolicXML GUIKitXML definition into its equivalent widget definition."

(* Utilities *)

GUIScreenShot::usage = "GUIScreenShot[obj] creates a Mathematica raster graphic of the current rendering of the user interface obj. This object can be either
 an existing GUIObject instance, the name of a user interface definititon which will load, paint and then close, or a Java user interface object."
 
(*******************************)
Begin["`Private`"]
(*******************************)

GUIKit`Information`$CreationID := JavaBlock[ Module[{id = Null, jf, jar, mn, atts},
	InstallJava[];
	jf = Select[GetClassPath[], (StringMatchQ[#, "*GUIKit.jar"] || StringMatchQ[#, "*GUIKit-debug.jar"]) &, 1];
	If[ Length[jf] < 1, Return[id]];
	jar = JavaNew["java.util.jar.JarInputStream", JavaNew["java.io.FileInputStream", First[jf]]];
	If[ jar === Null, Return[id]];
	mn = jar @ getManifest[];
	If[ mn === Null, Return[id]];
	atts = mn @ getMainAttributes[];
	If[ atts === Null, Return[id]];
	id = atts @ getValue["Signature-CreationID"];
	jar @ close[];
	id
  ]];
  
GUIKit`Information`$CreationDate := Module[{id},
  id = GUIKit`Information`$CreationID;
  If[ id === Null, Return[Null]];
  ToExpression["{" <> StringInsert[id, ",", {5, 7, 9, 11, 13}] <> "}"]
  ];
  
(*******************************
   Options
 *******************************)

(*******************************
   Messages
 *******************************)

$GUIPath::install = "Multiple installations of GUIKit exist at `1`. This may lead to unpredictable results when running GUIKit.";

(*******************************
   $GUIPath
*******************************)

autoGUIPath[] :=
  Module[{dir, appPaths = {}, appDirs, guiDirs(*, addOnBase, guiCopies*)},
    dir = If[ NameQ["$InstallationDirectory"], ToExpression["$InstallationDirectory"], $TopDirectory];
    If[StringQ[dir],
      PrependTo[appPaths, ToFileName[{dir, "AddOns", "ExtraPackages"}]];
      PrependTo[appPaths, ToFileName[{dir, "AddOns", "StandardPackages"}]];
      PrependTo[appPaths, ToFileName[{dir, "AddOns", "Autoload"}]];
      PrependTo[appPaths, ToFileName[{dir, "AddOns", "Applications"}]];
      PrependTo[appPaths, ToFileName[{dir, "SystemFiles", "Links"}]];
      PrependTo[appPaths, ToFileName[{dir, "AddOns", "Packages"}]];
      ];
    If[!Developer`$ProtectedMode,
        dir = If[ NameQ["$BaseDirectory"], ToExpression["$BaseDirectory"], $AddOnsDirectory];
        If[StringQ[dir],
            (* This branch is for 4.2 and later (4.1.5 on Mac OSX). *)
            PrependTo[appPaths, ToFileName[{dir, "Autoload"}]];
            PrependTo[appPaths, ToFileName[{dir, "Applications"}]];
        ];
        dir = If[ NameQ["$UserBaseDirectory"], ToExpression["$UserBaseDirectory"], $UserAddOnsDirectory];
        If[StringQ[dir],
            (* 4.2 and later *)
            PrependTo[appPaths, ToFileName[{dir, "Autoload"}]];
            PrependTo[appPaths, ToFileName[{dir, "Applications"}]];
        ,
            (* else *)
            PrependTo[appPaths, ToFileName[{$PreferencesDirectory, "AddOns", "Autoload"}]];
            PrependTo[appPaths, ToFileName[{$PreferencesDirectory, "AddOns", "Applications"}]];
        ]
    ];
     
    (* FileNames sorts all results so we need to apply this to each appPaths separate to 
       preserve desired path order *)
    appDirs = Select[ Flatten[FileNames["*", #]& /@ appPaths], (FileType[#] === Directory)&];
  
    guiDirs = Select[ToFileName[{#, "GUI"}]& /@ appDirs, (FileType[#] === Directory)&];
    
    (* This code checks if multiple GUIKit AddOns exist on the path and issue
       a warning to the user *)
    (* This is now turned off by default because all platforms should put local copies
       before bundled versions correctly *)
    (*
    addOnBase = ToFileName[{"*", "GUIKit", "GUI"}];
    guiCopies = Select[guiDirs, StringMatchQ[#, addOnBase]&];
    If[ Length[guiCopies] > 1,
       Message[$GUIPath::install, guiCopies];
       ];
    *)
    
    (* Here we support checking for a GUIKit`Utility`$ExtraGUIPaths variable and if it 
       exists as a list prepending it to the guiDirs.
       This may be set, for instance, within the Wolfram Workbench or debugging tool before
       GUIKit is loaded *)
       
    If[ MatchQ[GUIKit`Utility`$ExtraGUIPaths, {__String}],
       guiDirs = Join[GUIKit`Utility`$ExtraGUIPaths, guiDirs];
       ];
       
     (* Here we put the special WRI SystemFiles dir first. The thinking for putting it absolutely first
        in the search path is that it becomes a convenient place to put GUIs and be sure they override any
        provided by applications. It can be used to resolve application conflicts in this way.
     *)
    PrependTo[guiDirs, ToFileName[{$TopDirectory, "SystemFiles", "GUI"}]];
    
    guiDirs
    ];

$GUIPath = autoGUIPath[];

$GUIPackageDirectory = DirectoryName[System`Private`FindFile[$Input]];


(*******************************
   Utility functions
 *******************************)

$GUIKitXMLFormat = "GUIKitXML";

kernelDebugQ[] := 
  If[ $VersionNumber >= 6.0 && NameQ["RuntimeTools`ExecutionState"],
     TrueQ["RuntimeAnalysisTools" /. Symbol["RuntimeTools`ExecutionState"][]],
     False
     ];
     
(*******************************
   canonicalOptions forces the use of Strings
   for options processing that ignores the
   original symbol's context
 *******************************)
 
SetAttributes[ canonicalOptions, {Listable}];

canonicalOptions[ name_Symbol /; SymbolName[Unevaluated[name]] === "Name" -> val_] := "Name" -> val;

canonicalOptions[expr___] := expr;

(*******************************
   simplified custom optionQ
 *******************************)

Attributes[extendedOptionQ]={HoldAll}
extendedOptionQ[x_] :=  MatchQ[Unevaluated @ x, _Rule | _RuleDelayed]

Attributes[invokeOrBindOptionQ]={HoldAll}
invokeOrBindOptionQ[x_] :=  MatchQ[canonicalOptions[Unevaluated[x]], Rule["Name", _] | RuleDelayed["Name", _] |
  Rule[InvokeThread, _] | RuleDelayed[InvokeThread, _] | 
  Rule[InvokeWait, _] | RuleDelayed[InvokeWait, _]]

(* Returns True if the pathname begins with a
   relative path metacharacter *)
(* for MacOS *)
beginsRelativeMetaCharQ[str_String] :=
   (StringMatchQ[str, ":"] || StringMatchQ[str, "::"] ||
    StringMatchQ[str, ToFileName[{":"},"*"]] ||
    StringMatchQ[str, ToFileName[{"::"},"*"]]) /;
      StringMatchQ[$OperatingSystem, "MacOS"]
(* for non-MacOS *)
beginsRelativeMetaCharQ[str_String] :=
   StringMatchQ[str, "."] || StringMatchQ[str, ".."] ||
   StringMatchQ[str, ToFileName[{"."},"*"]] ||
   StringMatchQ[str, ToFileName[{".."},"*"]]
beginsRelativeMetaCharQ[___] := False


(* NOTE: Java code calls this private utility function so if its context
   or signature changes, update the Java call as well
   Consider if this should be a public symbol in GUIKit if useful to developers
   directly, possibly.
 *)

findGUIFile[file_String] :=
  Block[{ffile},
    ffile = System`Private`FindFile[file];
    If[ StringQ[ffile],
      ffile = System`Private`ExpandFileName[ffile];
      If[ beginsRelativeMetaCharQ[ffile],
        ffile = ToFileName[{Directory[]}, ffile] ];
      If[ FileType[ffile] =!= File, ffile = $Failed];
    (* else *),
    ffile = $Failed];
  ffile
  ]
  
resolveMathematicaFile[file_String, otherDirs___] :=
  Block[{$Path = Join[ {otherDirs}, $GUIPath, $Path], 
         ffile = $Failed},
    ffile = findGUIFile[file];
    (* fallback for missing or unresolved files 
       allows for implied .xml and .m extension finds 
     *)
    If[ ffile === $Failed,
      ffile = findGUIFile[file <> ".xml"];
      If[ ffile === $Failed,
        ffile = findGUIFile[file <> ".m"];
        ];
      ];
    
    ffile
  ]

(* Currently the Java code will resolve the URL string 
   Since we also do not check for implied .xml or .m extensions here
   code that actually resolves URL would need to add this check
*)
resolveMathematicaFile[url_System`URL] := First[url]

(*******************************
   guiExpressionQ
 *******************************)

Attributes[guiExpressionQ] = {HoldAllComplete};

guiExpressionQ[x_] := True /; 
  MemberQ[{Integer, Real, String, Symbol,
    Widget, WidgetReference, WidgetGroup, List, WidgetSpace, WidgetFill, WidgetAlign,
    Script, PropertyValue, Rule, RuleDelayed, SetPropertyValue, InvokeMethod, BindEvent}, Head[Unevaluated[x]]];
    
guiExpressionQ[___] := False


(*******************************
   convertToScriptContent converts various
   Mathematica versions of script code
   to a String or hopefully a valid GUIKit expression
 *******************************)

Attributes[convertToScriptString] = {HoldAllComplete};

convertToScriptString[expr_String] := expr

(* Changing the current context is done to make sure explicit contexts are always
   included in String version *)
   
convertToScriptString[(expr_HoldComplete | expr_Hold) /; kernelDebugQ[] ] := 
  Block[{$ContextPath = {$Context, "GUIKit`", "JLink`", "System`"}},
    Module[{ef},
      ef = Symbol["RuntimeTools`AddTags"][ expr];
      ef = Hold @@ ef;
      ef = Map[ Symbol["RuntimeTools`ProcessTags"], ef, {1}];
      ef = ToString[ef, InputForm];
      StringTake[ ef, {6, -2}]
      ]
    ]  
convertToScriptString[expr___ /; kernelDebugQ[]] := 
  Block[{$ContextPath = {$Context, "GUIKit`", "JLink`", "System`"}},
    Module[{ef},
		  ef = Symbol["RuntimeTools`AddTags"][ Hold[expr]];
		  ef = Hold @@ ef;
		  ef = Map[ Symbol["RuntimeTools`ProcessTags"], ef, {1}];
		  ef = ToString[ef, InputForm];
		  StringTake[ ef, {6, -2}]
      ]
    ] 

convertToScriptString[expr_HoldComplete | expr_Hold] := 
  Block[{$ContextPath = {$Context, "GUIKit`", "JLink`", "System`"}},
    (Function[x, ToString[Unevaluated[x], InputForm], {HoldAllComplete}] @@ expr )
    ]  
convertToScriptString[expr___] := 
  Block[{$ContextPath = {$Context, "GUIKit`", "JLink`", "System`"}},
    (Function[x, ToString[Unevaluated[x], InputForm], {HoldAllComplete}] @ expr )
    ] 

(*******************************
   toTargetObject
   
   This prepares either a String or a JavaObject
   for calling into Java as a target name,
   the use of ToString[str] on strings here
   is to strip out possible styleboxes and 
   linear syntax inside the string
 *******************************)
 
 (* Investigate if these created Java objects are being left in the VM somewhere
   and whether we need to track and do a ReleaseJavaObject *)
 toTargetObject[obj_] := MakeJavaObject[If[StringQ[obj], ToString[obj], obj]];
 
(*******************************
   guiResourceQ
 *******************************)

Attributes[guiResourceQ] = {HoldAllComplete};

guiResourceQ[file_String] := True
guiResourceQ[url_System`URL] := True
guiResourceQ[___] := False


(*******************************
   GUIScreenShot 
   
   This is a utility function for generating Mathematica
   raster bitmaps of user interface objects
   
   When given an GUIObject a grab/show method
   is used otherwise a widget is asked to paint into the graphics.
   This is the only reliable choice based on cross-platform 
   performance for now
 *******************************)
 
Options[GUIScreenShot] = {
  };
 
GUIScreenShot[obj_?JavaObjectQ, opts___] :=
  JavaBlock[
    Module[{useObj = obj, w, h, src},
      InstallJava[];
      If[ !InstanceOf[useObj, LoadJavaClass["java.awt.Component"]], 
        Return[$Failed]];
      If[ InstanceOf[useObj, LoadJavaClass["javax.swing.RootPaneContainer"]],
          useObj = useObj @ getContentPane[]];
      w = useObj @ getWidth[]; 
      h = useObj @ getHeight[];
      src = JavaNew["java.awt.image.BufferedImage", w, h, BufferedImage`TYPEUINTURGB];
      useObj @ paint[src @ getGraphics[]];
      produceGraphicsFromImage[src, w, h]
      ]
    ];
 
GUIScreenShot[ref_GUIObject, opts___] :=
  JavaBlock[
    Module[{useObj, w, h, src, data},
      If[ !GUIObjectQ[ref],
        Message[GUIObject::navail];
        Return[$Failed];
        ];
      InstallJava[];
      useObj = First[ref];
      If[ !InstanceOf[useObj, LoadJavaClass["java.awt.Component"]], 
        Return[$Failed]];
      data = useObj @ getBounds[];
      w = data @ width;
      h = data @ height;
      If[ InstanceOf[useObj, LoadJavaClass["java.awt.Window"]],
        JavaShow[useObj];
        ];
      src = JavaNew["java.awt.Robot"] @ createScreenCapture[data];
      produceGraphicsFromImage[src, w, h]
      ]
    ];

(*
 Care to set the PlotRange the same as the coordindates in the Raster
 and the ImageSize to set the actual number of pixels.
*)
produceGraphicsFromImage[src_, w_, h_] :=
  JavaBlock[
    Module[{data},
      data = src@getRaster[]@ getPixels[0, 0, w, h, JavaNew["[I", 3 w h]];
      data = Developer`ToPackedArray[N[data]];
      Graphics[ Raster[ Reverse[Partition[Partition[data/255., 3], w]],
        {{0, 0}, {w, h}}, ColorFunction -> RGBColor, ColorFunctionScaling -> False], 
        ImageSize -> {w, h}, 
        PlotRange -> {{0, w}, {0, h}},
        AspectRatio -> Automatic]
     ]
   ];
   
GUIScreenShot[{{xmin_Integer, xmax_Integer}, {ymin_Integer, ymax_Integer}}] :=
  JavaBlock[
    Module[{src},
      InstallJava[];
      src = JavaNew["java.awt.Robot"] @ createScreenCapture[ 
        JavaNew["java.awt.Rectangle", Min[xmin, xmax], Min[ymin, ymax], Abs[xmax-xmin], Abs[ymax-ymin] ]];
      produceGraphicsFromImage[src, Abs[xmax-xmin], Abs[ymax-ymin]]
      ]
   ];
  
GUIScreenShot[] :=
  JavaBlock[
    Module[{t,s},
      InstallJava[];
      LoadJavaClass["java.awt.Toolkit"];
      t = Toolkit`getDefaultToolkit[];
      s = t @ getScreenSize[];
      GUIScreenShot[{{0, s @ width}, {0, s @ height}}]
      ]
    ];
 
GUIScreenShot[file_String, opts___] := 
  resolveAndScreenShot[file, opts];
    
GUIScreenShot[expr_?guiExpressionQ, opts___] := 
  resolveAndScreenShot[expr, opts];
    
GUIScreenShot[doc_ /; Head[doc] === XMLObject["Document"], opts___] := 
  resolveAndScreenShot[doc, opts];
    
resolveAndScreenShot[int_, opts___] := 
  Module[{ref, result},
    ref = GUIRun[int];
    If[ ref === $Failed, Return[$Failed]];
    result = GUIScreenShot[First[ref], opts];
    ReleaseGUIObject[ref];
    result
    ];
   
(*******************************
   Subpackages

   Now we load separated subpackages inside
   `Private` context

   All public symbols and usage messages should be
   declared in this main file
 *******************************)
(
    Get[ToFileName[#, "GUIExpression.m"]];
    Get[ToFileName[#, "GUIRuntime.m"]];
    Get[ToFileName[#, "GUIToXML.m"]];
    Get[ToFileName[#, "GUIFromXML.m"]];
)& @ ToFileName[$GUIPackageDirectory, "Kernel"]


(*******************************)
End[]   (* end private context *)
(*******************************)

(*******************************)
EndPackage[]
(*******************************)

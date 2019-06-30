
(* :Title: RLink *)

(* :Context: RLink` *)

(* :Author:
        Leonid Shifrin
        leonids@wolfram.com
*)

(* :Package Version: 1.1 *)

(* :Mathematica Version: 10.0 *)

(* :Copyright: RLink source code (c) 2011-2014, Wolfram Research, Inc. All rights reserved.  *)

(* :Discussion:

   RLink is a Mathematica enhancement that integrates R and Mathematica. You can use 
   RLink to call R from Mathematica.   
*)

(* :Keywords: R Interoperability *)



BeginPackage["RLink`"]


(* Inert heads *)

RVector::usage = 
"RVector[type,data,attributes] represents an internal form of R vector in RLink.";

RList::usage =
"RList[{elems}, attributes] represents an internal form of R list in RLink"; 

RNull::usage = 
"RNull[] is RLink's representation of  R NULL object";

RAttributes::usage = 
"RAttributes[attrules] is an RLink  container for  attributes of R objects";

RObject::usage = 
"RObject[data, attributes] represents a general R object, usually having \
non-trivial attributes";

RCode::usage = "RCode[codeParts:{__String}] contains the inert (\"deparsed\") 
representation of R objects which can not be directly imported to Mathematica, 
obtained by calling R \"deparse\" function on those R objects. In general, 
such elements can not be successfully imported back to R, but in some cases
and for some data types (e.g. top-level functions - R closures), they can";

REnvironment::usage = "REnvironment[] represents the current top - level R 
environment. Environments other than top - level are not supported";

RFunction::usage = "RFunction[code] creates a function in R workspace from code \
code. If code represents a valid R function definition, a function reference \
(also with the head RFunction) is returned. Otherwise, $Failed is returned. \

RFunction[code][args] can be used to call an R function defined by code on the \
arguments args. \

RFunction[type, RCode[code], ref_, attr_] represents a reference to a function \
in R workspace. The type can be either \"closure\" or \"builtin\". The code \
is a string representing the code of the function. The ref can be either an \
Integer or Automatic. The former is for functions explicitly returned from R \
by REvaluate. The latter is for functions defined via the short form RFunction[code] \
from Mathematica. ";


(* Type conversion functions *)

ToRForm::usage = 
"ToRForm[expr] returns a  full  form of expr used by RLink internally  \
to communicate with R. For expressions which do not correspond to any \
supported R type and can not be converted, ToRForm returns $Failed.";

FromRForm::usage = 
"FromRForm[expr] converts a full form of an expression expr representing \
some valid R object, to the short form. Returns $Failed on illegal expressions,\
 which do not form a valid RLink representation of an R object.";


(* Main high-level interface *)


RSet::usage = 
"RSet[var, expr] assigns the value of Mathematica expression expr to a \
variable var in R workspace, returning back the value of expr upon success \
and $Failed upon failure.";

REvaluate::usage = 
"REvaluate[code] evaluates the string of R code, and returns the result, as \
a Mathematica expression";

(*

RVariable::usage = "RVariable[lhs] is used to perform an assignment to the string \
argument lhs in R workspace,as RVariable[lhs] = rhs, where rhs can be any Mathematica \
expression convertable to R";

*)

(* Initialization / deinitialization *)


InstallR::usage = "InstallR[opts] installs RLink and launches R runtime.";

UninstallR::usage = "UninstallR[] uninstalls R runtime / RLink";



(* Data type extension system *)


RTypeOfHighLevelExpression::usage = 
"RTypeOfHighLevelExpression[expr] returns a type of the R object which a given \
expression represents. Can be used to test RLink's interpretation of a given \
Mathematica expression";

RTypeOfLowLevelExpression::usage = 
"RTypeOfHighLevelExpression[expr] returns a type of the R object which a given \
(low-level, expressed in heads internal to RLink, such as RLink or RVector ) expression \  
represents. Can be used to test RLink's interpretation of a given Mathematica expression";

RDataTypeRegisteredQ::usage = "RDataTypeRegisteredQ[type_] tests if a given data type \
(represented usually as a string) has been registered with RLink";

RDataTypeRegister::usage = 
"RDataTypeRegister[type,directTypePattern,  forwardConverterRule, inverseTypePattern, \
backwardConverterRule] registers a new Mathematica representation for a data type with \
name type. Expressions matching directTypePattern pattern will be considered by RLink \
to be of type type, and the rule  forwardConverterRule instructs RLink how to convert \
the new Mathematica representation to RObject - based representation. The inverse pattern \
inverseTypePattern identifies types of RObject - based expressions which RLink considers \
representing objects of type type, and backwardConverterRule instructs RLink how to convert \
such RObject - based expressions to a more convenient higher-level Mathematica representation";

RDataTypeUnregister::usage = 
"RDataTypeUnregister[ nameType ] unregisters the type nameType, removing it from a set of \
extended types RLink knows about. If the type has not yet been registered or has been already \
unregistered, RDataTypeUnregister does nothing";

$RDataTypePath::usage = "$RDataTypePath is a global variable which stores a list of locations \
(directories) where RLink looks for definitions of extended data types";

RDataTypeDefinitionsReload::usage = "RDataTypeDefinitionsReload[opts] reloads all extended data \
types, searching a list of locations stored in $RDataTypePath for files with data type \
definitions. It is called internally by InstallR, but can also be called manually. It takes \
the \"AddToRDataTypePath\" option (same as InstallR) which can be used to indicate additional \
places to search for .m files with type definitions";


(* Paclet download mechanism for RLinkRuntime paclet *)

RLinkResourcesInstall::usage = "RLinkResourcesInstall[opts] downloads and installs RLink runtime. \
RLinkResourcesInstall[path, opts] installs RLink runtime from the RLinkRuntime paclet located \
at the specified path";

RLinkResourcesUninstall::usage = "RLinkResourcesUninstall[] uninstalls RLinkRuntime paclet.";



Begin["`Developer`"];

Needs["JLink`"];

$logFile = RLink`Private`$logFile;

getLog[]:=
	Import[$logFile,"String"];
	

(* This must be done before RLinkInit[] is called, to take effect *)
setLogFile[file_String]:=
	RLink`Private`$logFile = file;
	
jstr[s_]:= JavaNew["java.lang.String", s];
jb = JavaBlock;
	
logIt[msg_String,"TRACE"]:= jb@RLinkInit`rlogger@trace[jstr@msg];
logit[msg_String,"DEBUG"]:= jb@RLinkInit`rlogger@debug[jstr@msg];
logit[msg_String,"INFO"]:= jb@RLinkInit`rlogger@info[jstr@msg];
logit[msg_String,"WARN"]:= jb@RLinkInit`rlogger@warn[jstr@msg];
logit[msg_String,"ERROR"]:= jb@RLinkInit`rlogger@error[jstr@msg];
logit[msg_String,"FATAL"]:= jb@RLinkInit`rlogger@fatal[jstr@msg];
logit[msg_String]:= logit[msg,"INFO"];


clearLog[]:= If[FileExistsQ[$logFile],DeleteFile[$logFile]];



ClearAll[debug];
SetAttributes[debug, HoldAll];
debug[code_] := 
	Internal`InheritedBlock[{Message}, 
 		Module[{inMessage}, 
 			Unprotect[Message];
   			Message[args___] /; ! MatchQ[First[Hold[args]], _$Off] := 
    			Block[{inMessage = True}, 
      				Print[
      					{
      						Shallow /@ Replace[#, HoldForm[f_[___]] :> HoldForm[f], 1], 
          					Style[Map[Short, Last[#], {2}], Red]
          				} &@
          					Drop[Drop[Stack[_], -7], 4]
          			];
      				Message[args];
      				Throw[$Failed, Message];
      			] /; ! TrueQ[inMessage];
   			Protect[Message];
   		];
  		Catch[StackComplete[code], Message]
  	];



End[]  (* Developer`*)





Begin["`Private`"]
(* Implementation of the package *)

Needs["JLink`"];
Needs["ResourceLocator`"];
Needs["PacletManager`"];
(* 
 * This is not strictly necessary, as long as the package was loaded 
 *)
Needs["RLink`RCodeHighlighter`"]; 




(******************************************************************************)
(******************************************************************************)
(************ 		RLinkRuntime downloading / installation		***************)
(******************************************************************************)
(******************************************************************************)


$RLinkResourcesPacletName = "RLinkRuntime";


pacletFind[]:=
	PacletFind[$RLinkResourcesPacletName];
	
RLinkRuntimeInstalledQ[]:=
	pacletFind[]=!={};
	

quoteFileName[file_String] :=
	If[needsQuotesQ[file],
		StringJoin["\"", file, "\""],
		file
	]

needsQuotesQ[str_String] := !StringMatchQ[str, "\"" ~~ ___ ~~ "\""]


getAndCheckBooleanOption[head_, name_, msgName_, opts_]:=
	With[{ov = OptionValue[head, opts, name]},
		If[!MatchQ[ov,True|False],
			Message[MessageName[head, msgName], ov];
			Throw[$Failed, error[getAndCheckBooleanOption]],
			(* else *)
			ov	
		]
	];
	
	
locatePacletAndGetName[]:=	
	Module[{paclets = PacletFindRemote[$RLinkResourcesPacletName], paclet},
		If[paclets === {},
			Throw[$Failed, error[locatePacletAndGetName]]
		];
		paclet = First[paclets];
		"QualifiedName" /. PacletInformation[paclet]	
	];
	
	
	
ClearAll[customDialog];

Options[customDialog] = {WindowSize -> {500, FitAll}};

customDialog[text_, title_: None, opts : OptionsPattern[]] :=
 DialogInput[
 	DialogNotebook[{
    	TextCell[text, NotebookDefault, "DialogStyle", "ControlStyle", 
     		TextJustification -> 1, TextAlignment -> Left
     	],
    	ExpressionCell[
     		ChoiceButtons[{DialogReturn[True], DialogReturn[False]}],
     		TextAlignment -> Center
     	]}
    ],
  	Evaluate[
   		Sequence @@ FilterRules[
   			{opts}~Join~Options[customDialog], 
     		Options[DialogInput]]],
  	WindowTitle -> title
 ];
  
  
$RInstallationConfirmationText = 
"You have chosen to install the R Project for Statistical Computing \
(\"R\") as a separate module using the RLink installer. R is \
licensed under Version 2 of the GNU General Public License. By installing R \
using the RLink installer, you understand and agree that R is a \
separate program under a separate license.";  


RInstallationConfirm[] :=
	customDialog[$RInstallationConfirmationText,"R Installation Confirmation"]	
	
SyntaxInformation[RLinkResourcesInstall] = {"ArgumentsPattern" -> {_., OptionsPattern[]}};

RLinkResourcesInstall::dwnld = 
"Internet download or installation of data for RLink runtime failed. \
Use Help > Internet Connectivity... to test or reconfigure internet \
connectivity."

RLinkResourcesInstall::intcnct= 
"Internet connectivity is currently disabled. Use Help > Internet Connectivity... to enable it";


Options[RLinkResourcesInstall] = {Update -> False, "Force" -> True}

RLinkResourcesInstall[opts:OptionsPattern[]] := RLinkResourcesInstall["Web", opts]

(* TODO:  remove code duplication for the web and non-web versions *)	
RLinkResourcesInstall[Automatic | "Web", opts:OptionsPattern[]] :=
	Module[{paclet, updateQ,  check},		
		check[mname_,oname_]:=
			getAndCheckBooleanOption[RLinkResourcesInstall,mname, oname, {opts}];				
		updateQ = check[Update,"update"];				
		If[updateQ,
			RLinkResourcesUninstall[]
		];
		paclet = pacletFind[];			
		If[paclet =!= {},
			Return[paclet]
		];	
		If[!TrueQ[PacletManager`$AllowInternet],
			Message[RLinkResourcesInstall::intcnct];
			Return[$Failed]
		];
		If[!RInstallationConfirm[],
			Return[$Failed];
		];
							
		Quiet@Check[
			PacletTools`PacletGet[$RLinkResourcesPacletName, RLink, Automatic, "Update" -> True],
			(* some error *)
			Message[RLinkResourcesInstall::dwnld];					
			Return[$Failed]
		];
		paclet = pacletFind[];
		If[paclet === {},
			$Failed,
			(* else *)
			paclet
		]		
	]

RLinkResourcesInstall[pth_String?FileExistsQ, opts:OptionsPattern[]] :=
	Module[{updateQ,forceQ, paclet,check},
		If[!RInstallationConfirm[],
			Return[$Failed];
		];
		check[mname_,oname_]:=
			getAndCheckBooleanOption[RLinkResourcesInstall,mname, oname, {opts}];				
		updateQ = check[Update,"update"];				
		forceQ = check["Force","force"];			
		If[updateQ,
			RLinkResourcesUninstall[]
		];			
		paclet = pacletFind[];		
		If[paclet =!= {} && !forceQ,
			Return[paclet]
		];			
		
		PacletManager`PacletInstall[pth];
		paclet = pacletFind[];
		If[paclet === {},
			$Failed,
			(* else *)
			paclet
		]					
	]
	
(* TODO: error message here *)	
RLinkResourcesInstall[_,OptionsPattern[]]:= $Failed;


SyntaxInformation[RLinkResourcesUninstall] = {"ArgumentsPattern" -> {}};

RLinkResourcesUninstall[] :=
	(
		PacletUninstall /@ pacletFind[];
		PacletManager`RebuildPacletData[]
	);



getRLinkRuntimePath[paclets_List]:=
	With[{systemFilePath = PacletResource[First@paclets, "SystemFiles"]},		
		FileNameDrop[
			systemFilePath,
			-1
		]/;StringQ[systemFilePath]&&FileExistsQ[systemFilePath]
	];
	
getRLinkRuntimePath[___]:=Throw[$Failed, error[getRLinkRuntimePath]];
 

(* TODO Not sure if remembering the path is a good idea *)
$RRuntimeLocation := 
	With[{paclets = pacletFind[]},
		If[paclets === {}, 
			Throw[$Failed, error[pacletFind]],
			(* else *)
			$RRuntimeLocation = getRLinkRuntimePath[paclets]
		]		
	]



(******************************************************************************)
(******************************************************************************)
(************ 			General settings and utilities			***************)
(******************************************************************************)
(******************************************************************************)


(*****************************************************************************)
(****************  				 Option configurator  			**************)
(*****************************************************************************)


(* 
**  This has to be defined before we load test configurations 
*)
ClearAll[setOptionConfiguration, getOptionConfiguration, withOptionConfiguration];
SetAttributes[withOptionConfiguration, HoldFirst];
	Module[{optionConfiguration},
  		optionConfiguration[_][_] = {};
  		setOptionConfiguration[f_, tag_, {opts___?OptionQ}] :=
     		optionConfiguration[f][tag] = (* FilterRules[{opts}, Options[f]];*) {opts};
  		getOptionConfiguration[f_, tag_] := optionConfiguration[f][tag];
  		withOptionConfiguration[f_[args___], tag_] :=
  			f[args, Sequence @@ optionConfiguration[f][tag]];
	]; 

makeConfigurator[s_String]:=		  	
	Function[code,withOptionConfiguration[code,s],HoldFirst];
	
 
(* 
**  Test configurations, when they exist
*)
Quiet@Get["RLink`Kernel`TestConfigurations`"];




(*****************************************************************************)
(****************   General error - handling - NEEDS MORE WORK  **************)
(*****************************************************************************)


(* Change this to error messages for specific function calls ?? *)
ClearAll[RLink];
RLink::err = "General RLink error in function `1`";
RLink::noinst = "The R runtime has not been installed. Install it first, by running InstallR";

ClearAll[throwError];
throwError[fun_, args___]:=Throw[$Failed, error[fun, {args}]];

ClearAll[defError];
defError[fun_Symbol, extra___]:=
	fun[args___]:= 
		If[TrueQ[$debug], 
			throwError[fun, "args", extra, {args}],
			(* else *)
			throwError[fun, "args", extra]
		];


ClearAll[makeCrashHandler];
SetAttributes[makeCrashHandler, HoldFirst];
makeCrashHandler[crashFlag_Symbol]:=
	Function[
		code
		,
		Quiet[
			Check[
				crashFlag = False;
				code
				,
				crashFlag = True;
				$Failed
				,
				{LinkObject::linkd,LinkObject::linkn}
			]
			,
			{LinkObject::linkd,LinkObject::linkn}
		]
		,
		HoldAll
	];
	

ClearAll[checkRLinkInstalled];
checkRLinkInstalled[attachMessageTo_Symbol]:=
	Function[
		code
		, 
		If[TrueQ[$RCurrentlyInstalled],
			code, 
			(* else *)
			Message[attachMessageTo::noinst];
   			$Failed
		]
		,
		HoldFirst
	];


(*TODO Change the default to False, and remove explicit GenerateErrorMessage ->False in calling functions *)
ClearAll[handleError];
SetAttributes[handleError, HoldFirst];
Options[handleError] = {
   	GenerateErrorMessage -> True,
   	TestInstall -> True  
};
   
handleError[code_, attachMessageTo_Symbol, opts : OptionsPattern[]]/;
TrueQ[OptionValue[handleError,{opts},TestInstall]] && !TrueQ[$RCurrentlyInstalled]:=
   	(
   		Message[attachMessageTo::noinst];
   		$Failed
   	);
   
handleError[code_, attachMessageTo_Symbol, opts : OptionsPattern[]] := 
  	With[{msgQ = OptionValue[GenerateErrorMessage]},
   		Catch[code, _error,
    		Function[{value, tag},
     			If[TrueQ@msgQ,
      				Message[attachMessageTo::err, Style[First@tag, Red]]
      			];
     			value
     		]
     	]
	];
     
handleError[code_]:=
	Catch[code,_error,
		 Function[{value, tag},
		 	{value, tag}
		 ]
	];
	
	
(*****************************************************************************)
(****************  				Global settings  				**************)
(*****************************************************************************)	


$testMode = ValueQ[$testConfiguration];

$debug = False;

$projectLocation = 
	With[{
		pos = 
			Position[
				FileNameSplit[$InputFileName], 
				"RLink"
			][[If[$testMode, -2,-1],1]]
		},
		FileNameTake[$InputFileName,pos]
	];



$RCurrentlyInstalled = False;

$currentJavaLink = Null;

If[!ValueQ[$RWasInstalledDuringMathematicaSession], 
    $RWasInstalledDuringMathematicaSession = False
];
	
$temporaryDirectory = ApplicationDataUserDirectory["RLink"];

logit = RLink`Developer`logit;
clearLog = RLink`Developer`clearLog;
$logFile = FileNameJoin[{$temporaryDirectory,"RLink.log"}];



(*****************************************************************************)
(****************  		General helper / utility functions		**************)
(*****************************************************************************)

showIt[x__] := (Print[x]; x)
	

SetAttributes[CleanUp, HoldAll];
CleanUp[expr_, cleanup_] :=
 Module[{exprFn, result, abort = False, rethrow = True, seq}, 
 	exprFn[] := expr;
  	result =
   		CheckAbort[
    		Catch[
    			Catch[result = exprFn[]; rethrow = False; result],
    			_, 
    			seq[##] &
    		],
    		abort = True
    	];
  	cleanup;
  	If[abort, Abort[]];
  	If[rethrow, Throw[result /. seq -> Sequence]];
  	result
 ];
  


Clear[getBadArgsAndPositions];
getBadArgsAndPositions[args_List, pred_] :=
	Transpose @ Select[
    	Transpose[{args, Range[Length[args]]}], 
    	! pred[First@#] &
    ];


ClearAll[LetL];
SetAttributes[LetL, HoldAll];
LetL /: Verbatim[SetDelayed][lhs_, rhs : HoldPattern[LetL[{__}, _]]] := 
	Block[{With}, 
		Attributes[With] = {HoldAll};
   		lhs := Evaluate[rhs]
   	];
LetL[{}, expr_] := expr;
LetL[{head_}, expr_] := With[{head}, expr];
LetL[{head_, tail__}, expr_] := 
	Block[{With}, 
		Attributes[With] = {HoldAll};
  		With[{head}, Evaluate[LetL[{tail}, expr]]]
  	];
   

Clear[unflatten];
unflatten[arr_List, dims : {__Integer}] /; Length[arr] == Times @@ dims :=
	First @ Fold[
    	Map[
    		Function[l, Partition[l, Length[l]/First[#2]]], 
    		#, 
    		{Last[#2]}
    	] &, 
    	{arr}, 
    	Transpose[{#, Range[Length[#]]}] &@Most[dims]
    ];


(*****************************************************************************)
(****************   	General Java - related utilities		**************)
(*****************************************************************************)


ClearAll[javaIterate];
SetAttributes[javaIterate, HoldAll];
javaIterate[var_Symbol, code_][iteratorInstance_?JavaObjectQ] :=
	While[iteratorInstance@hasNext[],
		With @@ Hold[{var = iteratorInstance@next[]}, code]
  	];
   
javaIterate[___][___] := Throw[$Failed, error[javaIterate]];


ClearAll[save, collect];
save = Sow;

SetAttributes[collect, HoldAll];
collect[code_] := If[# === {}, #, First@#] &@Reap[code][[2]];


Clear[jstring];
jstring[str_String] := JavaNew["java.lang.String", str];



(******************************************************************************)
(******************************************************************************)
(************ 		RLink internal (lower-level) mechanics		***************)
(******************************************************************************)
(******************************************************************************)



(*****************************************************************************)
(****************   	RExecutor singleton and related			**************)
(*****************************************************************************)


ClearAll[generateRandomVariables];
generateRandomVariables[varNum_Integer]:=
	With[{exec = First[getRExecutor[]]},
		Table[exec@getRandomVariable[], {varNum}]
	];


ClearAll[withRTemporaryVariables];
SetAttributes[withRTemporaryVariables, HoldRest];
withRTemporaryVariables[vars : {__String}, code_] :=
   	CleanUp[
   		code
   		, 
   		(* The test is necessary because the code could 
   		have caused a crash of R / JVM *)
   		If[javaLinkValidQ[],
   			With[{exec = First[getRExecutor[]]},
   				Scan[exec@removeRVariable[#] &, vars]
   			]
   		];
   	];
   	
   
withRTemporaryVariables[varNum_Integer, {vars_Symbol, code_}] :=  
	With @@ Hold[
		{vars = generateRandomVariables[varNum]}
		,
     	withRTemporaryVariables[vars, code]
     ];
     
withRTemporaryVariables[___] := 
	Throw[$Failed, error[withRTemporaryVariables]];
  
  
ClearAll[javaLinkValidQ]
javaLinkValidQ[]:=  MemberQ[Links[], $currentJavaLink];

ClearAll[getRExecutor];
getRExecutor[] := 
	If[javaLinkValidQ[],
		RExecutor[com`wolfram`links`rlink`RLinkInit`getRExecutor[]],
		(* else *)
		Throw[$Failed, error["low_level_crash"]]
	];
	


(*****************************************************************************)
(***********   Dealing with Java classes: In and Out - type instances  *******)
(*****************************************************************************)



ClearAll[getInVectorClassNameForType];
getInVectorClassNameForType[type_String] :=
	type /. {
    	"integer" :> "RIntegerVectorInType",
    	"double" :> "RDoubleVectorInType",
    	"character" :> "RCharacterVectorInType",
    	"complex" :> "RComplexVectorInType",
    	"logical" :> "RLogicalVectorInType",
    	_ :> Throw[$Failed, error[getInVectorClassNameForType]]
    };
    

ClearAll[getOutVectorClassNameForType];
getOutVectorClassNameForType[type_String] :=
	type /. {
    	"integer" :> "RIntegerVectorOutType",
    	"double" :> "RDoubleVectorOutType",
    	"character" :> "RCharacterVectorOutType",
    	"complex" :> "RComplexVectorOutType",
    	"logical" :> "RLogicalVectorOutType",
    	_ :> Throw[$Failed, error[getOutVectorClassNameForType]]
    };


ClearAll[getFullClassName];
Options[getFullClassName] = {
	PackageName -> "com.wolfram.links.rlink"
};
getFullClassName[shortName_String, opts : OptionsPattern[]] :=
  	StringJoin[OptionValue[PackageName], ".", shortName];


Clear[getFullInOutTypeClassName];
Options[getFullInOutTypeClassName] = {
	InPackageName -> "com.wolfram.links.rlink.dataTypes.inTypes",
	OutPackageName -> "com.wolfram.links.rlink.dataTypes.outTypes"
};
getFullInOutTypeClassName[inOrOut : (In | Out), shortName_String, opts : OptionsPattern[]] :=
	getFullClassName[
  		shortName, 
  		opts,
   		PackageName ->  
    		If[inOrOut === In, 
    			OptionValue[InPackageName]
    			, 
     			OptionValue[OutPackageName]
    		]
    ];


ClearAll[newRLinkTypeInstance];
newRLinkTypeInstance[
	className_String, 
	args_List, 
	classNameF_: getFullClassName, 
	opts : OptionsPattern[]
] :=
  JavaNew[classNameF[className, opts], Sequence @@ args];


ClearAll[newRLinkInOutTypeInstance];
newRLinkInOutTypeInstance[inOrOut : (In | Out), className_String, args_List, opts : OptionsPattern[]] :=
	newRLinkTypeInstance[className, args, getFullInOutTypeClassName[inOrOut, ##] &];


ClearAll[getVectorInstance];
getVectorInstance[inOrOut : (In | Out), type_String, args_List] :=
	With[{nameF = inOrOut /. {
       In -> getInVectorClassNameForType,
       Out -> getOutVectorClassNameForType
       }},
       newRLinkInOutTypeInstance[inOrOut, nameF[type], args]
	];


ClearAll[outTypeInstanceQ];
outTypeInstanceQ[o_] :=
	JavaObjectQ[o] && InstanceOf[o, getFullInOutTypeClassName[Out, "IROutType"]]



(*****************************************************************************)
(*********   Conversion between user-side and internal forms of data  ********)
(*****************************************************************************)


(*
**	Type system  and conversions between usual Mathematica and 
** 	R-aware forms of expressions. - NEEDS MORE WORK
*)


SyntaxInformation[RList] = {"ArgumentsPattern" -> {_, _}};

(* TODO Note that typeOf presently mixes the information which can be read \
without referring to Java, and that which requires Java calls. It may be \
cleaner to separate these, and make a separate operator for the calls to Java \
(types of Java instances)*)



ClearAll[typeOf];
typeOf[o_?outTypeInstanceQ] := o@ getType[]@getStringType[];

typeOf[var_String] := typeOf[var, getRExecutor[]];
(* typeOf[var_String?outputSuppressedQ,_RExecutor]:="SuppressedOutput"; *)
typeOf[var_String, RExecutor[exec_]] := 
	With[{type = exec@getRObjectType[var]},
		type/;type=!=Null];
typeOf[var_String,_RExecutor]:= "UnknownRCodeType";
	
typeOf[RVector[type_, data_, att_]] := type;
typeOf[RList[data_, att_]] := "list";
typeOf[RNull[]] := "NULL";
typeOf[_] := Throw[$Failed, error[typeOf]];



Clear[rDataTypeQ];
rDataTypeQ[value_] := 
  MatchQ[value, _RVector | _RList | _RNull |_RCode|_REnvironment| _RFunction];


(* Use that the set of attributes has the same syntax as options *)
SyntaxInformation[RAttributes] = {"ArgumentsPattern" -> {OptionsPattern[]}};


ClearAll[toRAttributes];
toRAttributes[RAttributes[atts___]]:= toRAttributes[atts];
toRAttributes[atts : ((_String :> _) ...)] :=
	RAttributes @@ Replace[
 		{atts},
   		(aname_ :> aval_) :> (aname :> Evaluate[toRDataType@aval]), 
   		{1}
   	];


ClearAll[arrayToRForm];
arrayToRForm[arr_, type_String] :=
	With[{dims = Dimensions@arr},
  		RVector[
  			type, 
  			(* Column - major order in R, need to transpose *)
  			Flatten[Transpose[arr,Reverse@Range@Length@dims]], 
   			toRAttributes["dim" :> dims]
   		]
   	];


ClearAll[vectorElementTypeRules];
vectorElementTypeRules[] :=
	{
		_Integer :> "integer",
		_Real :> "double",
		(_Complex | 0) :> "complex",
		_String :> "character",
		(True | False) :> "logical"
   	};


ClearAll[vectorTypeRules];
vectorTypeRules[] :=
	Append[
   		vectorElementTypeRules[] /. (pt_ :> type_) :> 
   			({(pt | Missing[]) ..} :> type),
   		{__} :> "unknown"
   	];


ClearAll[$machineIntegerLimit];
$machineIntegerLimit = 2^31;


ClearAll[withinMachineLimitsSymmetricQ];
withinMachineLimitsSymmetricQ[arr_, lim_] :=
	With[{fl =  Flatten@{arr}},
		Total[2-UnitStep[lim-fl]  - UnitStep[lim + fl]] == 0
	];
   
   
ClearAll[withinMachineLimitsAsymmetricQ];
withinMachineLimitsAsymmetricQ[arr_, lim_] := 
	With[{fl = Flatten@{arr}},
		Total[UnitStep[fl - lim] + 1 - UnitStep[fl + lim]] == 0
	];


ClearAll[withinMachineIntegerLimitsQ];
withinMachineIntegerLimitsQ[arr_] :=
  withinMachineLimitsAsymmetricQ[arr, $machineIntegerLimit];
  
  
ClearAll[withinMachineRealLimitsQ];
withinMachineRealLimitsQ[arr_] :=
	withinMachineLimitsSymmetricQ[arr, $MaxMachineNumber];


ClearAll[vectorTypeOf];
vectorTypeOf[data_] := data /. vectorTypeRules[];


ClearAll[checkNumericLimits];
checkNumericLimits[data_,type_]:=
	Block[{Missing = Sequence},
		With[{withinLimitsQ = 
				Switch[type,
					"integer",
						withinMachineIntegerLimitsQ[data],
					"double",
						withinMachineRealLimitsQ[data],
					"complex",
						withinMachineRealLimitsQ[Re[data]]&&withinMachineRealLimitsQ[Im[data]],
					_,
						True
				]
			},
			If[!withinLimitsQ,Throw[$Failed, error[checkNumericLimits]]];
		]
	];
					
	
ClearAll[vectorDataQ];
vectorDataQ[type_String] := type =!= "unknown";


ClearAll[toRDataType];
toRDataType[RCode[code_,att_RAttributes]]:=
	RCode[code,toRAttributes@att];
	
toRDataType[REnvironment[]]:= REnvironment[];	

toRDataType[RList[data_List/;MatchQ[data,Except[{__?rDataTypeQ}]],att_RAttributes]]:=
	RList[toRDataType/@data,toRAttributes@att];

toRDataType[value_?rDataTypeQ] := value;

toRDataType[sc : (_String | _Integer | _Real | _Complex | True | False)] := 
	toRDataType[{sc}];
  
toRDataType[data_List] :=
  With[{type  = vectorTypeOf[data]}, 
   		(
   			checkNumericLimits[data,type];
   			RVector[type, data, RAttributes[]]
   		) /; vectorDataQ[type]
  ];
   
toRDataType[data_?ArrayQ] :=
  With[{type  = vectorTypeOf[Flatten@data]},
  		(
  			checkNumericLimits[data,type];
   			arrayToRForm[data, type] 
   		) /; vectorDataQ[type]
  ];
   
toRDataType[RObject[data_, RAttributes[atts___]]] :=
	Replace[
		toRDataType[data],
		RAttributes[datts___] :> toRAttributes[datts, atts],
		{1}
	];
   
toRDataType[Null] := RNull[];

toRDataType[lst_List] :=
  	RList[toRDataType /@ lst, RAttributes[]];
  
toRDataType[arg_] :=
  	Throw[$Failed, error[toRDataType]];


ClearAll[fromRDataType];
fromRDataType[atts_RAttributes] :=
	Replace[
		atts,
		(aname_ :> aval_) :> (aname :> Evaluate[fromRDataType@aval]),
		{1}
	];
    
fromRDataType[RVector[_, data_List, RAttributes[]]] := data;

fromRDataType[RVector[type_, data_List, a : RAttributes[atts__]]] :=
	With[{dims  = "dim" /. {atts}},
		fromRDataType @ RVector[
			type,
      		(* Transforming to row-major order *)
      		Transpose[
      			unflatten[data,#],
      			Reverse @ Range @ Length@#
      		]& @ Reverse @ fromRDataType[dims]
      		, 
      		DeleteCases[a, "dim" :> _]
      	] /; dims =!= "dim"
    ];
      
fromRDataType[RVector[type_, data_List, atts_RAttributes]] :=
  	RObject[data, fromRDataType[atts]];
  
fromRDataType[RNull[]] := Null;

fromRDataType[RList[data_List, RAttributes[]]] :=
  	Map[fromRDataType, data];
  
fromRDataType[RList[data_List, atts_RAttributes]] :=
  	RObject[fromRDataType@RList[data, RAttributes[]], fromRDataType[atts]];
  
fromRDataType[r_RObject]:=r;
	
fromRDataType[RCode[code_,atts_RAttributes]]:=
	RCode[code, fromRDataType[atts]]; 
	
fromRDataType[env_REnvironment]:= env;  

fromRDataType[f_RFunction]:=f;
  
fromRDataType[_]:=Throw[$Failed, error[fromRDataType]];
  
(* fromRDataType[SuppressedOutput[]]:=Null; *)  


ClearAll[toCoreRMathematicaRepresentation, fromCoreRMathematicaRepresentation];
toCoreRMathematicaRepresentation["Core"] = {};
fromCoreRMathematicaRepresentation["Core"] = {};


Clear[ToRForm];
SyntaxInformation[ToRForm] = {"ArgumentsPattern" -> {_}};
ToRForm::invld = "The argument `1` does not have a valid type";

ToRForm[expr_] :=
	With[{
		result =
  			handleError[     
      			toRDataType[
      				expr //. Flatten @ DownValues[
   							toCoreRMathematicaRepresentation][[All, 2]]
       			],
       			ToRForm,
       			GenerateErrorMessage -> False,
       			TestInstall -> False
     		]
    	},
   		result /; result =!= $Failed
   	];
   
ToRForm[arg_] := 
	(
   		Message[ToRForm::invld, Style[Short[arg], Red]];
   		$Failed   		
	);


Clear[FromRForm];
SyntaxInformation[FromRForm] = {"ArgumentsPattern" -> {_}};
FromRForm::err = "General error in function `1`";

FromRForm[expr_] :=
  handleError[   
  	With[{llexpr = fromRDataType[expr]},
  		llexpr//.
  			Flatten @ DownValues[
   				fromCoreRMathematicaRepresentation][[All, 2]]  		
   	],
   	FromRForm,
    GenerateErrorMessage -> False,
    TestInstall -> False
  ];


ClearAll[convertableToRFormQ];
convertableToRFormQ[expr_] :=  
  	Quiet[
   		ToRForm[expr] =!= $Failed,
   		{ToRForm::invld}
   	];



(*****************************************************************************)
(****************   		Information / data transfer 		**************)
(*****************************************************************************)


ClearAll[RVectorQ];
RVectorQ["integer" | "double" | "character" | "logical" | "complex"] := True;
RVectorQ[_] := False;


ClearAll[RNullQ];
RNullQ["NULL"] = True;
RNullQ[_] := False;


ClearAll[RListQ];
RListQ["list"] = True;
RListQ[_] := False;


ClearAll[REnvironmentQ];
REnvironmentQ["environment"] = True;
REnvironmentQ[_]:=False;


ClearAll[RFunctionQ];
RFunctionQ["closure"|"builtin"] = True;
RFunctionQ[_]:=False;


ClearAll[getRAttributes];
getRAttributes[RAttributes[atts : ((_String :> _) ...)]] :=
	With[{map  = JavaNew["java.util.HashMap"]},
		(* ReplaceAll used to induce side effects *)
   		{atts} /. (att_String :> rhs_) :>
     		map @ put[
       			JavaNew["java.lang.String", att],
       			getInTypeInstance@rhs
       		];	
   		newRLinkInOutTypeInstance[In, "RInAttributesImpl", {map}]
   	];


ClearAll[vectorMissingDataReplacement];
vectorMissingDataReplacement[type_String] :=
	type /. {
    	"integer" | "double" | "complex" ->  0,
    	"character" -> "",
    	"logical" -> False
    };


ClearAll[vectorInMissingDataPreprocess];
vectorInMissingDataPreprocess[RVector[type_String, data_, att_RAttributes]] :=  
	With[{flat = Flatten[data]},
   		With[{pos = Position[flat, Missing[]]},
    		{
    			ReplacePart[
    				flat, 
    				pos -> vectorMissingDataReplacement[type]
    			], 
     			Flatten[pos]
     		}
     	]
     ];


ClearAll[functionRefToJavaObject];	
functionRefToJavaObject[ref_Integer]:=
	newRLinkInOutTypeInstance[
			In,
			"RFunctionInType",
			{ref}
	];
	
functionRefToJavaObject[___]:= 
	Throw[$Failed, error[functionRefToJavaObject]];


$evalDeparsedCode = True;


(*
 * Populates the "In" - type Java object from the Mathematica representation of 
 * a given R object, for data sent from Mathematica to R, and returns the resulting 
 * Java object's reference to Mathematica.
 *
 * In - type objects are disposable objects created on the Java side to be used 
 * only once, for a single data transfer from Mathematica to R via Java. They are
 * never reused.
 *
*)
ClearAll[getInTypeInstance];
getInTypeInstance::fail = "Failed to instantiate an appropriate Java class with the data `1`";
  
getInTypeInstance[vec : RVector[type_String, data_, att_RAttributes]] :=
	getVectorInstance[
		In, 
		type,
		{Sequence @@ vectorInMissingDataPreprocess[vec], getRAttributes[att]}
	];
   
getInTypeInstance[RList[data_, att_RAttributes]] :=
	With[{lst  = JavaNew["java.util.ArrayList", Length[data]]},
   		Scan[lst@add[getInTypeInstance@#] &, data];
   		newRLinkInOutTypeInstance[In, "RListInType", {lst, getRAttributes[att]}]
   	];
   
getInTypeInstance[RNull[]] :=
  	newRLinkInOutTypeInstance[In, "RNullInType", {}];
  
(* Note: the evaluation scheme for deparsed code sent back to R is rather ad-hoc *)  
getInTypeInstance[arg:RCode[code_,att_RAttributes]]:=		
	newRLinkInOutTypeInstance[
		In, 
		"RDeparsedCodeInType", 
		{
			MakeJavaObject@code, 
			Block[{$evalDeparsedCode = False},
				getRAttributes[att]
			],
			MakeJavaObject@$evalDeparsedCode
		}		
	];  
	
getInTypeInstance[REnvironment[]]:=
	newRLinkInOutTypeInstance[In, "REnvironmentInType", {}]; 	
			
getInTypeInstance[RFunction["closure",RCode[code_String], Automatic, _RAttributes]]:=
	withFunctionReferenceCheck[
		rFunctionDefinedInMathematicaHash[code],
		ref,		
		functionRefToJavaObject[ref],
		(* else *)
		getInTypeInstance[RFunction[code]] (* re-hash if reference is not valid *)
				
	];
		
getInTypeInstance[RFunction[__, refIndex_Integer, _RAttributes]]:=
	withFunctionReferenceCheck[
		refIndex,
		ref,		
		functionRefToJavaObject[ref],
		(* else *)
		Message[RFunction::invldref];
		Throw[$Failed,error[withFunctionReferenceCheck]]		
	];

getInTypeInstance[args___] :=
	(
   		Message[getInTypeInstance::fail , {args}];
   		Throw[$Failed, error[getInTypeInstance]]
   	);


(*
 * Requests the REngine (Java-side of the connection to R) to populates the 
 * "Out" - type Java object from the value contained in a variable var on 
 * the R side, for the purposes of data transfer from R to Mathematica,and 
 * returns the resulting Java object's reference to Mathematica.
 *
 * Out - type objects are disposable objects created on the Java side to be 
 * used only once, for a single data transfer from R to Mathematica via Java. 
 * They are never reused.
 *
*)
ClearAll[getOutTypeInstance];
getOutTypeInstance[var_String, type_String?RVectorQ] :=
	getVectorInstance[Out, type, {var}];
  
getOutTypeInstance[var_String, type_String?RNullQ] :=
	newRLinkInOutTypeInstance[Out, "RNullOutType", {var}];
  
getOutTypeInstance[var_String, type_String?RListQ] :=
	newRLinkInOutTypeInstance[Out, "RListOutType", {var}]; 
   
getOutTypeInstance[var_String, type_String?REnvironmentQ]:=
	newRLinkInOutTypeInstance[Out, "REnvironmentOutType", {var}];
	
getOutTypeInstance[var_String, type_String?RFunctionQ]:=
	With[{class = 
		If[type === "closure",
			"RClosureOutType",
			(* else *)
			"RBuiltinOutType"
		]
		},
		newRLinkInOutTypeInstance[Out,class, {var}]	
	];	     
   
getOutTypeInstance[var_String, type_String]/; type =!= "UnknownRCodeType":=
	newRLinkInOutTypeInstance[Out, "RDeparsedCodeOutType", {var}];  
   
getOutTypeInstance[___] :=
  	Throw[$Failed, error[getOutTypeInstance]];


ClearAll[numericTypeQ];
numericTypeQ[type_String] := MemberQ[{"integer", "double", "complex"}, type];


(*
 * A helper function prescribing how to build a Mathematica-side representation
 * for a given R object / expression, from the Java Out-type object / instance, 
 * already containing the data taken from the R side. 
*)
ClearAll[buildROutTypeFunction];
buildROutTypeFunction[type_String?RVectorQ, e_RExecutor] :=
	Function[{outInstance, attributes},
   		With[{
   			missingPositions = outInstance@getMissingElementPositions[],
     		infOrNanF = 
      			If[numericTypeQ[type],
       				Function[{methodName}, outInstance@methodName[]],
       				(* else *)
       				Function[{methodName}, {}]
       			]
       		},
    		With[{
      			NaNPositions = infOrNanF[getNaNElementPositions],
      			pInfPositions = infOrNanF[getPositiveInfinityPositions],
      			negInfPositions =  infOrNanF[getNegativeInfinityPositions],
      			complexInfinityPositions =  infOrNanF[getComplexInfinityPositions]
      			},
     			RVector[
      				type, 
      				Fold[
       					ReplacePart,
       					outInstance@getElements[],
       					{
         					missingPositions -> Missing[],
         					NaNPositions -> Indeterminate,
         					pInfPositions -> Infinity,
         					negInfPositions -> -Infinity,
         					complexInfinityPositions -> ComplexInfinity
         				} /. (p_ -> val_) :> (Transpose[{p}] -> val)
       				], 
      				attributes
      			]
      		]
      	]
	];
     
buildROutTypeFunction[type_String?RNullQ, e_RExecutor] :=
	Function[{outInstance, attributes},
   		RNull[]
   	];
      
buildROutTypeFunction[type_String?RListQ, e_RExecutor] :=
	Function[{outInstance, attributes},
   		With[{jlist = outInstance@getList[]},
    		RList[
     			Table[
      				RDataTypeFromOutTypeInstance[typeOf[#], #, e] &[jlist@get[i]],
      				{i, 0, jlist@size[] - 1}
      			],
     			attributes
     		]
     	]
     ];    
         
buildROutTypeFunction[type_String?REnvironmentQ, e_RExecutor] :=
  	Function[{outInstance, attributes},  
  		REnvironment[]
  	]; 

buildROutTypeFunction[type_String?RFunctionQ, e_RExecutor] :=
  	Function[{outInstance, attributes},
   		RFunction[
   			type, 
   			RCode[
   				StringJoin @ Riffle[
   					outInstance@getDeparsedFunctionSource[]@getDeparsedCode[],
   					"\n"
   				]
   			], 
   			outInstance@getRefNumber[],
   			attributes
   		]
  	];

buildROutTypeFunction[type_String, e_RExecutor] :=
  	Function[{outInstance, attributes},  
  		If[outInstance@getType[]@getStringType[] =!= "unknown",
  			$Failed,
  			(* else *)
  			RCode[outInstance@getDeparsedCode[], attributes]  	
  		] 	
  	];   
       
buildROutTypeFunction[_] :=
  	Throw[$Failed, error[buildROutTypeFunction]];



(*****************************************************************************)
(*****************************************************************************)
(****************   IMPLEMENTATION OF TOP - LEVEL FUNCTIONS 	**************)
(*****************************************************************************)
(*****************************************************************************)


ClearAll[numericalPreprocess];
numericalPreprocess[data_]:=
	Chop[data, $MinMachineNumber];
	

ClearAll[assignFunction];
assignFunction[var_String, ref_Integer]:=
	With[{rFunctionRef = RFunctionOutType`getFunctionHashElement[ref]},
		RExecute[
			StringJoin[var, " <- ", rFunctionRef]
		] /; StringQ[rFunctionRef]
	];

assignFunction[___]:= Throw[$Failed, assignFunction];


ClearAll[iRSet];
iRSet[var_String, val:RFunction["closure",RCode[code_String], Automatic, _RAttributes]]:=	
	withFunctionReferenceCheck[
		rFunctionDefinedInMathematicaHash[code],
		ref,		
		assignFunction[var,ref];
		val,
		(* else *)
		iRSet[var,RFunction[code]] (* re-hash if reference is not valid *)				
	];
		
iRSet[var_String, val:RFunction[__,refIndex_Integer,_RAttributes]]:=
	withFunctionReferenceCheck[
		refIndex,
		ref,
		assignFunction[var,ref];
		val,
		(* else *)
		Message[RFunction::invldref];
		Throw[$Failed,error[withFunctionReferenceCheck]]
	];	
	
iRSet[var_String, value_] := 
	iRSet[var, value, getRExecutor[]];
	
iRSet[var_String, value_?rDataTypeQ, RExecutor[exec_]] :=
	JavaBlock[
   		With[{inTypeInstance = getInTypeInstance[value]},
    		If[!TrueQ[inTypeInstance@rPut[var, exec]],
     			Throw[$Failed, error[iRSet]],
     			(* else *)
     			value
     		]
    	]
    ];
    
iRSet[var_String, value_, e_RExecutor] :=
  	With[{converted = numericalPreprocess@ToRForm@value},
   		Module[{},
     		iRSet[var, converted, e];
     		value
     	] /; converted =!= $Failed
   	]; 
   
iRSet[___] := Throw[$Failed, error[iRSet]];


ClearAll[RAttributesFromOutTypeInstance];
RAttributesFromOutTypeInstance[outTypeInstance_?JavaObjectQ, e : RExecutor[exec_]] :=
  	Module[{atts, attIter},
   		atts = outTypeInstance@getAttributes[];
   		If[atts === Null, 
   			Return[RAttributes[]]
   		];
   		attIter = atts@getAllAttributeNames[exec]@iterator[];
   		RAttributes @@ collect[
     		javaIterate[
     			att,
       			LetL[{
         			attInstance = atts@getAttribute[jstring@att, exec],
         			attData = 
          				RDataTypeFromOutTypeInstance[typeOf[attInstance], attInstance, e]
         			},
        			save[att :> attData]
        		]
        	][attIter]
        ]
   	];


ClearAll[RDataTypeFromOutTypeInstance];
RDataTypeFromOutTypeInstance[type_String, outTypeInstance_, e_RExecutor] :=
  	buildROutTypeFunction[type, e][
   		outTypeInstance,
   		RAttributesFromOutTypeInstance[outTypeInstance, e]
   	];
   
RDataTypeFromOutTypeInstance[___] :=
  	Throw[$Failed, error[RDataTypeFromOutTypeInstance]];


ClearAll[rToJavaObject];
rToJavaObject::errget = 
  "
  The resulting object could not be retrieved, likely because it contains elements \
  not currently supported by RLink. Its type was found to be `1`   and its class is `2`";

rToJavaObject[var_, type_String, e : RExecutor[exec_]] :=
	With[{outTypeInstance = getOutTypeInstance[var, type]},
   		If[! TrueQ[outTypeInstance@rGet[exec]],
   			With[{class = exec@evalGetString["class(" <> var <> ")"]},
    			Message[rToJavaObject::errget, Style[type, Blue],Style[class,Blue]];
    			Throw[$Failed, error[rToJavaObject]]
    		]
    	];
   		outTypeInstance
   	];


ClearAll[getRDataFromR];
getRDataFromR::fail = "Failed to retrieve the result from R - unknown result type";

getRDataFromR[var_String] := 
	getRDataFromR[var, getRExecutor[]];
	
getRDataFromR[var_String, e : RExecutor[exec_]] := 
  	getRDataFromR[var, typeOf[var, e], e];
  
(*
getRDataFromR[var_,"SuppressedOutput",_RExecutor]:=
	(RExecute[var];SuppressedOutput[]);
*)

getRDataFromR[var_, "UnknownRCodeType",_RExecutor]:=
	(
		Message[getRDataFromR::fail];
		Throw[$Failed, error[getRDataFromR]]
	);
  
getRDataFromR[var_, type_String, e : RExecutor[exec_]] :=
  	JavaBlock[
   		RDataTypeFromOutTypeInstance[
    		type,  
    		rToJavaObject[var, type, e],
    		e
    	]
   	];


ClearAll[iRExecute];
iRExecute[code_String] := 
	iRExecute[code, getRExecutor[]];

iRExecute[code_String, RExecutor[exec_]] :=
	Module[{crashed = False, crashHandler, result},
		crashHandler = makeCrashHandler[crashed];
		result = 
			crashHandler @ handleError[
				exec@eval[code],
				iRExecute, 
				GenerateErrorMessage -> False
			];
		If[crashed,
			(* TODO: add purely Mathematica-level logger, to log the crash *)
			Throw[$Failed, error["low_level_crash"]]
		];
		Null /; TrueQ[result]
   	];
   
iRExecute[___] := $Failed;


ClearAll[rfcall, rassign, argsToTempVars, newLineWrap,rfcallWrapped];
rfcall[f_, fargs_List] :=
  	StringJoin["(", f, ")", "(", Sequence @@ Riffle[fargs, ","], ")"];
  
rassign[lhs_String, rhs_String] := 
	StringJoin[lhs, " <- ", "(", rhs, ")"];
	
newLineWrap[codeParts__String]:=
	StringJoin[Riffle[{codeParts}, "\n"]];
	
rfcallWrapped[f_,fargs_List]:=
	With[{funVars  = generateRandomVariables[Length[fargs]]},
		newLineWrap[
			"(function(){",
			Sequence @@ Thread[rassign[funVars,fargs]],
			rfcall[f,funVars],
			"})()"
		]
	];

argsToTempVars[vars_List, argums_List] /; Length[vars] == Length[argums] :=
  	Scan[
   		If[RSet @@ # === $Failed, Return[#]] &,
   		Transpose[{vars, argums}]
   	];
   
   
ClearAll[outputSuppressedQ];
outputSuppressedQ[code_String]:=StringMatchQ[code, __~~";"];


ClearAll[iREvaluate];
iREvaluate[code_String?outputSuppressedQ]:=
	RExecute[code];
	
iREvaluate[code_String] :=
	Module[{tempvars,res},
  		Quiet[
  			Check[
  				withRTemporaryVariables[
  					1,
  					{
  						tempvars,
  						(* body *)
  						Quiet[
  							res = 
  								RExecute @ StringJoin[First@tempvars,	"<- (",	code,	")"	]
  							,
  							{RExecute::rerr}
  						];
  						(* Print["in iREvaluate: testing the result"]; *)
    					If[res=!=$Failed,
    						FromRForm@getRDataFromR@First@tempvars,
    						(* else *)
    						res
    					]
  					}
  				], 
    			$Failed, 
    			{Java::excptn}
    		],
   			{Java::excptn}
  		]
  	];


ClearAll[iRApply];
iRApply[fname_String, args_List, errorHandler_] :=
  	Module[{tempVars, putResult, evalResult, tag, catch, msgAndFailIf},
   		catch = Function[code, Catch[code, tag], HoldAll];
   		SetAttributes[msgAndFailIf, HoldAll];
   		msgAndFailIf[cond_, mname_, margs_List] :=
   			If[cond,
   				errorHandler[mname] @@ margs;
   				Throw[$Failed, tag]
   			];
   		(* body *)
   		catch@withRTemporaryVariables[
   			Length[args] + 1
   			,
   			{
   				tempVars
   				,
   				With[{argVars = Most@tempVars, resultVar = Last@tempVars},
   					putResult = argsToTempVars[argVars, args];
   					msgAndFailIf[
   						putResult =!= Null,
   						RFunction::puterr,
   						{putResult}
   					];
   					evalResult =
   						iRExecute[
   							rassign[resultVar, rfcallWrapped[fname, argVars]]
   						];
   					Block[{$rApplyTemporaryVars = argVars},
   						msgAndFailIf[
   							evalResult === $Failed,
   							RFunction::callerr,
   							{fname, args}
   						]
   					];
   					With[{result =
   							handleError[
   								iREvaluate[resultVar],
   								iREvaluate,
   								GenerateErrorMessage -> False
   							]
   						},
   						msgAndFailIf[
   							result === $Failed,
   							RFunction::rtomerr,
   							{}
   						];
   						result
   					]
   				]
   			}
   		]
   	];


ClearAll[getLastRErrorMessage];
getLastRErrorMessage[] :=	
  	Style[
  		StringReplace[
  			StringTrim[First@REvaluate["geterrmessage()"]],
  			RLinkInit`FUNCTIONUHASHUVARUNAME~~"[["~~(DigitCharacter..)~~"]]" :> "<function>"
  		], 
  		Red
  	];


(*
**	Public (high - level) interface
*)

Clear[RSet];

SyntaxInformation[RSet] = {"ArgumentsPattern" -> {_,_}};

RSet::err = "General error in function `1`";
RSet::noinst = "The R runtime has not been installed. Install it first, by running InstallR";
RSet::puterr = "Error putting the expression `1` into a variable `2` in R. The last error message issued by R was `3`";
RSet::badval = "The expression `1` is not convertable to R";
RSet::badvar  = "The assigned variable name must be a String";
RSet::argnum = 
  "The function was called with `1` argument(s). Two arguments were expected";

RSet[var_String, value_?convertableToRFormQ] :=
  	With[{
  		result  = 
     		handleError[iRSet[var, value], RSet, GenerateErrorMessage -> False]
     	},
   		If[result === $Failed && TrueQ[$RCurrentlyInstalled],
    		Message[
    			RSet::puterr,
    			Style[value, Red],
    			Style[var, Red],
    			Style[getLastRErrorMessage[],Red]
    		]
   		];
   		result
   	];
   
RSet[_String, value_] :=
  (Message[RSet::badval, Style[value, Red]]; $Failed );
  
RSet[_, _] := 
  (Message[RSet::badvar ]; $Failed);
  
RSet[args___] :=
  (Message[RSet::argnum , Length[{args}]]; $Failed);


(* TODO: seems to be a failed concept (in terms of usability). Remove? *)
ClearAll[RVariable];
RVariable /: Set[RVariable[lhs_String],rhs_]:=
	RSet[lhs,rhs];


ClearAll[RExecute];
RExecute::rerr = "The following R error encountered: `1`";
RExecute::badarg = "The argument must be a string";
RExecute::argnum = 
  "The function was called with `1` argument(s). One argument was expected";
  
RExecute[code_String] :=
  	With[{result = iRExecute[code]},
   		Null /; result =!= $Failed
   	];
   
RExecute[code_String] :=
  	(
   		Message[RExecute::rerr, getLastRErrorMessage[]];
   		$Failed
   	);
   
RExecute[_] :=
  	(
  		Message[RExecute::badarg];
  		$Failed
  	);
  
RExecute[args___] :=
  	(	
  		Message[RExecute::argnum , Length[{args}]];
  		$Failed
  	);


Clear[REvaluate];

SyntaxInformation[REvaluate] = {"ArgumentsPattern" -> {_}};

(* REvaluate::rerr = "The following R error encountered: `1`"; *)

REvaluate::fnerr = "General error in function `1`";
REvaluate::err = "General error. Error code: `1`";
REvaluate::interr = "General internal error";
REvaluate::noinst = "The R runtime has not been installed. Install it first, by running InstallR";

REvaluate::badargs = "Bad number and / or type of arguments";

REvaluate::rerr = 
  "Failed to retrieve the value for variable or piece of code `1`. The following R error was \
encountered: `2`";

REvaluate::crash = "Crash in low-level RLink component or in R runtime. Please reinstall RLink via InstallR";


REvaluate[var_String] :=
	checkRLinkInstalled[REvaluate] @ Module[{res, errorType},
		res = 
			handleError[
  				With[{result = iREvaluate[var]},
    				If[result === $Failed,
     					Message[
     						REvaluate::rerr,
      						Style[var, Blue],
      						getLastRErrorMessage[]
      					]
     				];
    				result
    			]
  			];
  		If[!FreeQ[res, error],
  			errorType = First@Last@res;
  			(*Print["errorType: ", errorType]; *)
  			Switch[errorType,
  				"low_level_crash",
  					Message[REvaluate::crash];
  					$RCurrentlyInstalled = False;
  					Update[InstallR];
  					Null,
  				_String,
  					Message[REvaluate::err, Style[errorType, Red]],
  				_Symbol,
  					Message[REvaluate::fnerr, Style[errorType, Red]],
  				_,
  					Message[REvaluate::interr]
  			];
  			Return[$Failed]
  		];
  		res
	];

REvaluate[___] :=
 	(
 		Message[REvaluate::badargs]; 
 		$Failed
 	);



(******************************************************************************)
(******************************************************************************)
(****** 	RLINK FUNCTION APPLICATION AND FUNCTION REFERENCES		***********)
(******************************************************************************)
(******************************************************************************)


Clear[RApply,rFunctionMessage];
RApply::badrargs = "Arguments set to a called function must be a list";
RApply::badargs = "Bad number and / or type of arguments";


SetAttributes[rFunctionMessage, HoldAll];

rFunctionMessage[RFunction::rincomp] :=
	Function[
		badArgValuesPositions
		,
   		Message[
   			RFunction::rincomp,
    		Style[First@badArgValuesPositions, Red],
    		Style[Last@badArgValuesPositions, Red]
    	]
    ];

rFunctionMessage[RFunction::rtomerr ] :=
  	Function[Null, Message[RFunction::rtomerr ]];

rFunctionMessage[RFunction::puterr] :=
  	Function[
  		putResult
  		,
   		Message[
   			RFunction::puterr,
    		Style[Last@putResult, Red]
    		(*,Style[First@putResult, Red]*)
    	]
    ];

rFunctionMessage[RFunction::callerr] :=
	Function[{fname, args},
   		Message[
   			RFunction::callerr,
    		(* Style[fname, Red], *)
    		Style[args, Red],
    		getLastRErrorMessage[] /. s_String :> 
    			StringReplace[
    				s,
    				Thread[$rApplyTemporaryVars -> Map[ToString,args]]
    			] 
    	]
    ];

RApply[fname_String, args : {___?convertableToRFormQ}] :=
  	handleError[
  		iRApply[fname, args, rFunctionMessage],
  		RApply
  	];
  
RApply[fname_String, args_List] :=
  	With[{
  		badArgValuesPositions = 
     		getBadArgsAndPositions[args, convertableToRFormQ]
    	},
   		rFunctionMessage[RFunction::rincomp][badArgValuesPositions];
   		$Failed
   	];
      
RApply[fname_String, args_] :=
  	(
  		Message[RApply::badrargs]; 
  		$Failed
  	);   
  
RApply[___] :=
 	(
 		Message[RApply::badargs]; 
 		$Failed
 	);


ClearAll[rFunctionDefinedInMathematicaHash, clearFunctionsHash];
clearFunctionsHash[]:=
	(
		ClearAll[rFunctionDefinedInMathematicaHash];
		rFunctionDefinedInMathematicaHash[_]=Null;
	)

clearFunctionsHash[];


(* A helper macro to test that the RFunction object has a valid reference *)
ClearAll[withFunctionReferenceCheck];
SetAttributes[withFunctionReferenceCheck, HoldRest];
withFunctionReferenceCheck[refIndex_,refSym_Symbol, code_,errCode_]:=
	Block[{refSym = refIndex},
		LoadJavaClass["com.wolfram.links.rlink.dataTypes.outTypes.RFunctionOutType"];
		If[!IntegerQ[refIndex] || !RFunctionOutType`isValidFunctionReference[refIndex,First@getRExecutor[]],
			errCode,
			(* else *)
			code
		]
	];


Clear[RFunction];

SyntaxInformation[RFunction] = {"ArgumentsPattern" -> {_,_.,_.,_.}};

RFunction::invldcode = "The code `1` does not define a valid R function";

RFunction::invldref = "The function reference used in a function call is no longer valid. The most likely reason is that \
R runtime was restarted after this reference has been created / saved in a variable";

RFunction::invld = "Invalid R function call";

RFunction::err = "General error in function `1`";

RFunction::noinst = "The R runtime has not been installed. Install it first, by running InstallR";

RFunction::puterr = 
  "Unable to transfer the expression `1` to R ";
  
RFunction::callerr = 
  "Function call for the function with arguments `1` resulted in the \
following error on R side: `2`";

RFunction::rincomp = 
  "Arguments `1` at positions `2` could not be converted to the R - \
transferrable Mathematica expressions";

RFunction::rtomerr = 
  "Error getting the result of the computation back from R, perhaps due to an \
error during the computation";

RFunction::noinst = "The R runtime has not been installed. Install it first, by running InstallR";
(* 
For functions defined from Mathematica. Such expressions will always be 
a valid reference for a function, unlike references for functions returned 
from R. 
*)

RFunction["closure",RCode[code_String], Automatic, RAttributes[]][args___]:=
	checkRLinkInstalled[RFunction] @ withFunctionReferenceCheck[
		rFunctionDefinedInMathematicaHash[code],
		ref,		
		RApply[RFunctionOutType`getFunctionHashElement[ref],{args}],
		(* else *)
		RFunction[code][args]
	];	
	

RFunction[__,refIndex_Integer,_RAttributes][args___]:=
	checkRLinkInstalled[RFunction] @ withFunctionReferenceCheck[
		refIndex,
		ref,		
		RApply[RFunctionOutType`getFunctionHashElement[ref],{args}],
		(* else *)	
		Message[RFunction::invldref];
		$Failed	
	]
	
	
	
(* If a given code string is in hash,test the reference,and if valid, return it *)	
RFunction[code_String]:= 
	With[{result = 
		checkRLinkInstalled[RFunction] @ withFunctionReferenceCheck[
			rFunctionDefinedInMathematicaHash[code],
			ref,
			RFunction["closure", RCode[code], Automatic, RAttributes[]],
			(* else *)
			Null
		]},
		result /; result=!=Null
	]; 
				
(* The code string is not in hash. Attempt to evaluate the code string and create and
hash a valid new function reference. Return it upon success *)	
RFunction[code_String]:=
	With[{ref = Quiet@REvaluate[code]},
		(
			Replace[ref, 
				RFunction[__,refindex_,_RAttributes] :>	
					(rFunctionDefinedInMathematicaHash[code] =  refindex)
			];
			RFunction["closure", RCode[code], Automatic, RAttributes[]]
		)/;	MatchQ[ref,_RFunction]
	];
	
(* Fallthrough case for invalid code string *)	
RFunction[code_String]:= (
	Message[RFunction::invldcode,Style[code,Red]];
	$Failed
);

(* Fallthrough  for generic erroneous function call, for whatever reason *)
RFunction[___][___]:=	(
	Message[RFunction::invld];
	$Failed
);



(* Formatting rules for RFunction *)


(*  Formatting disabled for the time being


$openingDoubleAngularBracket = FromCharacterCode[171];
$closingDoubleAngularBracket = FromCharacterCode[187];


Format[RFunction[type_, code_RCode, ref_, _RAttributes], OutputForm] := 
  StringJoin["<<RFunction[" , type , ",", "<<code>>", ",", ToString[ref], "]>>"];
  
Format[RFunction[type_, code_RCode, ref_, _RAttributes], TextForm] := 
  StringJoin["<<RFunction[" , type , ",", "<<code>>", ",", ToString[ref], "]>>"];
  

RFunction /: 
  MakeBoxes[rf : RFunction[type_, code_RCode, ref_, _RAttributes], fmt_] := 
  	With[{strRef = ToString[ref],openbr = $openingDoubleAngularBracket, closebr = $closingDoubleAngularBracket},
   		InterpretationBox[
   			RowBox[{
   				openbr, 
   				RowBox[{"RFunction", "[", 
        				RowBox[{type, ",", "<<code>>", ",", strRef}], 
        			"]"
        		}], 
        		closebr
        	}], 
        	rf]
   	];
  
*)



(******************************************************************************)
(******************************************************************************)
(************ 			RLINK TYPE EXTENSION SYSTEM				***************)
(******************************************************************************)
(******************************************************************************)


$defaultDataTypeDirectory = 
	If[$testMode,
		FileNameJoin[{$projectLocation,"RLink","Kernel","DataTypes"}],
		(* else*)
		FileNameJoin[{$projectLocation,"Kernel","DataTypes"}]
	];
	
	
$RDataTypePath = {$defaultDataTypeDirectory};


loadType[fname_String?FileExistsQ]:=
	Get[fname];


ClearAll[RDataTypeDefinitionsReload];
Options[RDataTypeDefinitionsReload] = {
	"AddToRDataTypePath" :> None	
};

SyntaxInformation[RDataTypeDefinitionsReload] = 
	{"ArgumentsPattern" -> {OptionsPattern[]}};

SyntaxInformation[RTypeOfHighLevelExpression] = 
	{"ArgumentsPattern" -> {_}};
	
SyntaxInformation[RTypeOfLowLevelExpression] = 
	{"ArgumentsPattern" -> {_}};
	
SyntaxInformation[RDataTypeRegisteredQ] = 
	{"ArgumentsPattern" -> {_}};

RDataTypeRegisteredQ::badarg = "String or symbol expected at position 1 in `1`";

RDataTypeDefinitionsReload[opts:OptionsPattern[]]:=	
	Module[{path, extra = OptionValue["AddToRDataTypePath"], 
		files,common, helper, defArgx, funs, fullName},		
		funs = {
			RTypeOfHighLevelExpression, 
			RTypeOfLowLevelExpression, 
			RDataTypeRegisteredQ
		};		
		defArgx[fun_]:=
			(fun[args___]/;Length[{args}]!=1:= 
				"never happens"/;Message[General::argx, fun ,Length[{args}]]);
		fullName[name_]:=
			FileNameJoin[{$defaultDataTypeDirectory, name}];
		
		Clear @@ funs;		
		path = If[extra === None, $RDataTypePath, Join[$RDataTypePath,extra]];
		common = fullName["Common.m"];	
		helper = fullName["RDataTypeTools.m"];	
		files = {common} ~ Join ~ FileNames["*.m",path];	
		Block[{$ContextPath},
			Get[helper]
		];	
		Scan[Get,files];
		
		(* Note: it is important that these definitions are given last *)
		RTypeOfHighLevelExpression[expr_?convertableToRFormQ] := 
			RTypeOfLowLevelExpression[ToRForm[expr]];
		RTypeOfHighLevelExpression[_] := None;		
		
		RTypeOfLowLevelExpression[_?rDataTypeQ] := "core";
		RTypeOfLowLevelExpression[_] := None;		
		
		RDataTypeRegisteredQ["core"] = True;
		RDataTypeRegisteredQ[_String|_Symbol] = False;
		call : RDataTypeRegisteredQ[arg_]:=
			"never happens"/;Message[RDataTypeRegisteredQ::badarg, HoldForm[call]];
		
		(* Error handling *)	
		Map[defArgx,funs];					
	];	



Clear[RDataTypeRegister];
RDataTypeRegister::argnum = "RDataTypeRegister called with `1` arguments. Five arguments are expected.";
RDataTypeRegister::badarg = "RDataTypeRegister called with wrong argument types. The arguments were `1`.";


SyntaxInformation[RDataTypeRegister] = {"ArgumentsPattern" -> {_,_,_,_,_}};

RDataTypeRegister::duplreg = 
  "The type with the name `1` has already been registered";
RDataTypeRegister[
	nameType : (_String | _Symbol), 
	directTypePattern_, 
	forwardConverterRule_, 
	inverseTypePattern_, 
	backwardConverterRule_
] :=
  	Module[{},
  		If[RDataTypeRegisteredQ[nameType],
    		Message[RDataTypeRegister::duplreg, nameType];
    		Return[$Failed]
   		];
   		RTypeOfHighLevelExpression[directTypePattern] := nameType;
   		RTypeOfLowLevelExpression[inverseTypePattern] := nameType;
   		toCoreRMathematicaRepresentation[nameType] := forwardConverterRule;
   		fromCoreRMathematicaRepresentation[nameType] := backwardConverterRule;
   		RDataTypeRegisteredQ[nameType] = True;
  	];

RDataTypeRegister[args___]:=
	With[{argn = Length[{args}]},
		(
			Message[RDataTypeRegister::argnum,argn];
			$Failed
		)/;argn != 5
	];

RDataTypeRegister[args___]:=
	(
		Message[RDataTypeRegister::badarg,{args}];
		$Failed
	);



Clear[RDataTypeUnregister];

SyntaxInformation[RDataTypeUnregister] = {"ArgumentsPattern" -> {_}};

RDataTypeUnregister::strsym = "String or symbols expected in position 1 in `1`";

RDataTypeUnregister[
   nameType : (_String | _Symbol) /; RDataTypeRegisteredQ[nameType]] :=  
  	Scan[
   		(DownValues[#] =
      		DeleteCases[DownValues[#], dv_ /; ! FreeQ[dv, nameType]]) &,
   		{
   			RDataTypeRegisteredQ,
    		RTypeOfHighLevelExpression,
    		RTypeOfLowLevelExpression,
    		toCoreRMathematicaRepresentation,
    		fromCoreRMathematicaRepresentation
   	 	}
   	];
  
RDataTypeUnregister[nameType : (_String | _Symbol)] := Null;

call: RDataTypeUnregister[arg_]:= 
	(
		Message[RDataTypeUnregister::strsym,HoldForm[call]];
		$Failed
	);
	
RDataTypeUnregister[args___]:=
	(
		Message[General::argx, RDataTypeUnregister, Length[{args}]];
		$Failed
	);



(******************************************************************************)
(******************************************************************************)
(************ 		RLINK INSTALLATION AND UNINSTALLATION		***************)
(******************************************************************************)
(******************************************************************************)


(******************************************************************************)
(************ 		Some utilities / helpper functions 			***************)
(******************************************************************************)

(* 
**  If code ever calls this function, this means there is an error: 
**  the location of R should have been determined prior to that.
*)
ClearAll[getRHomeLocation];
getRHomeLocation[]:=
	Throw[$Failed, error[getRHomeLocation]];


ClearAll[autoSet];
SetAttributes[autoSet,HoldAll];
autoSet[var_,value_]:=
	If[var === Automatic, var = value];
	
	
ClearAll[initLogger];
initLogger[]:=
	Module[{},	
		LoadJavaClass["com.wolfram.links.rlink.RLinkInit"];	
        LoadJavaClass["java.lang.System"];
        If[TrueQ[$CloudEvaluation], java`lang`System`setOut[Null]];
		RLinkInit`rlogger@removeAllAppenders[];
		(* clearLog[]; *)
		RLinkInit`rlogger@addAppender[
			JavaNew["org.apache.log4j.ConsoleAppender"]
		];
		RLinkInit`rlogger@addAppender[
			JavaNew[
				"org.apache.log4j.FileAppender", 
				JavaNew["org.apache.log4j.SimpleLayout"],			
				$logFile
			]
		];
		logIt["Logger initialized"];
	];	


ClearAll[subDirectoryQ];
subDirectoryQ[subDir_String, dir_String]:=
 MatchQ[FileNameSplit[subDir],Append[FileNameSplit[dir],__]];



(******************************************************************************)
(************ 	JRI library version / location detection 		***************)
(******************************************************************************)


ClearAll[versionQ];
versionQ[{_Integer?NonNegative,_Integer?NonNegative,_Integer?NonNegative}]:=True;
versionQ[_]:=False;


ClearAll[$stringVersionToListRule, $stringVersionPattern];
$stringVersionToListRule = 
	StringExpression[
		main:DigitCharacter,
		".", 
		sub:Repeated[DigitCharacter,{1,2}],
		".",
		subsub:Repeated[DigitCharacter,{1,2}]
	]:> {main, sub, subsub};

$stringVersionPattern = First @ $stringVersionToListRule;


ClearAll[applyToRHS];
applyToRHS[f_, lhs_ :> rhs_]:= lhs :> f[rhs];


ClearAll[detectRVersion];
detectRVersion[v_Integer?Positive]:=detectRVersion[{v}];

detectRVersion[{v_Integer?Positive}]:= detectRVersion[{v,0}];

detectRVersion[{v_Integer?Positive, sub_Integer?NonNegative}]:= 
	detectRVersion[{v, sub, 0}];

detectRVersion[v_?versionQ]:=v;

detectRVersion[v_String]:=
	With[{vv = FirstCase[_]@StringCases[v,"R-"~~x__ :> x]},
		detectRVersion[vv] /; !MatchQ[vv,_Missing]
	];

detectRVersion[v_String /; StringMatchQ[v,DigitCharacter]]:=
	detectRVersion[ToExpression[v]];

detectRVersion[v_String]:=
	Module[{strip},
	(* Working around the bug in StringReplace in V10.0.0 (strip) *)
		strip[StringExpression[arg_]]:=arg;
		strip[arg_]:=arg;
		strip @ StringReplace[
			v,
			{
				applyToRHS[
					Composition[detectRVersion, Map[ToExpression]], 
					$stringVersionToListRule
				]
				,
				_ :> Return[$Failed, Module]
			}
		]
	];

(* TODO: improve error-handling here *)
detectRVersion[___]:=$Failed;


ClearAll[detectRVersionByLocation];
detectRVersionByLocation[location_String?DirectoryQ]:=
	detectRVersion[FileNameTake[location,-1]];

defError[detectRVersionByLocation, DirectoryQ];


(* Higher-level logic of R version detection from the option settings passed by the user *)
ClearAll[detectRVersionComplete];
detectRVersionComplete[Automatic, rHomeLocation_?DirectoryQ, errorParams_Association: Association[{}]]:=
	With[{detectedRVersion = detectRVersionByLocation[rHomeLocation]},
		If[detectedRVersion === $Failed,
			Throw[
				$Failed
				, 
				error[
					"R_version_detection",
					Append[
						errorParams,
						"RPath" -> rHomeLocation
					]
				]
			]
		];
		detectedRVersion
	];
	
detectRVersionComplete[RVersion_, rHomeLocation_?DirectoryQ]:=	
	With[{detectedRVersion = detectRVersion[RVersion]},
		If[detectedRVersion === $Failed,
			(* Explicit user-provided version invalid, attempting to auto-detect from the path *)
			Return[
				detectRVersionComplete[
					Automatic, 
					rHomeLocation, 
					Association["RVersionProvided" -> RVersion ]
				]
			]		
		];
		detectedRVersion		
	];

defError @ detectRVersionComplete;


ClearAll[vless, versionLessThan, pickVersion];
vless[{},_]=vless[_,{}]=False;
vless[{ff_,fr___},{ff_,sr___}]:=vless[{fr},{sr}];
vless[{ff_, fr___},{sf_,sr___}]:=ff<sf;


versionLessThan[fst_?versionQ, sec_?versionQ]:= 
	vless[fst, sec];

defError[detectRVersionComplete, versionQ];


pickVersion[versions:{__?versionQ}, type:(Min|Max)]:=
	With[{fn = If[type === Min, Identity, Not]},
		Fold[
			If[fn @ versionLessThan[#1,#2],#1,#2]&,
			First@versions,
			Rest@versions
		]
	];
	
pickVersion[versions_List, Min|Max]:= 
	Throw[
		$Failed, 
		error[pickVersion, "args", versionQ, If[TrueQ[$debug], versions, Sequence @@ {} ]]
	];
	
defError @ pickVersion;


(* Constructs an Association RVersion -> LibLocation *)
ClearAll[constructVersionDirMap];
constructVersionDirMap[libdir_String?DirectoryQ]:=
	With[{
		specificVersionPattern = $stringVersionPattern,
		toNumericVersion = detectRVersion
		},
		Composition[
			KeyMap[
				If[StringMatchQ[#,specificVersionPattern],
					toNumericVersion[#],
					(* else *)
					#
				]&
			],
			Association,
			Select[
				StringMatchQ[
					First @ #, 
					"AllVersions" | specificVersionPattern
				]&
			],
			Map[FileNameTake[#,-1]-> #&]
		]@ FileNames["*",{libdir}]
	];
	
defError @ constructVersionDirMap;


ClearAll[pickCorrectRVersion];
pickCorrectRVersion[RVersion_?versionQ, availableVersions:{("AllVersions"|_?versionQ)..}]:=
	Module[{specificVersions, result},
			If[MemberQ[availableVersions, RVersion],
				(* Exact match: library available for this specific version *)
				result = RVersion,
				(* else *)
				specificVersions = Select[versionQ] @ availableVersions;
				Which[
					(* No specific version libs are at all available. Look into AllVersions directory *)
					specificVersions === {},
						(* 
						 * Assume here that we always have AllVersions directory 
						 * if no specific version directories exist
						 *)
						Assert[MemberQ[availableVersions,"AllVersions" ]];
						result = "AllVersions"
					,
					(* Version smaller than the smallest for which library is available *)
					versionLessThan[RVersion, pickVersion[specificVersions, Min]],
						result = pickVersion[specificVersions, Min]
					,
					(* There are some versions smaller then this, for which libs are 
					available. Find the maximal version of those, and attempt to use that*)
					True,
						result = 
							Composition[
								pickVersion[#,Max]&,
								Select[versionLessThan[#, RVersion]&]
							] @ specificVersions
							
				] (* Which *)
			];
			result
		];
		
pickCorrectRVersion[_?versionQ, {}]:= 
	Throw[$Failed, error[pickCorrectRVersion, "args", "no_versions_available"]];

defError @ pickCorrectRVersion;


(* Attempts to detect the location of JRI native library, for a ggiven R 
version - needed only for external R *)
ClearAll[getJRILibLocation];
getJRILibLocation[RVersion_?versionQ, libdir_String?DirectoryQ]:=
	With[{dirAssoc = constructVersionDirMap[libdir]},
		dirAssoc[pickCorrectRVersion[RVersion, Keys @ dirAssoc]]
	];

defError @ getJRILibLocation;



(******************************************************************************)
(************ 				Main function to load R 			***************)
(******************************************************************************)

persistentValuesValidQ[val_] := MatchQ[val, <|"RHomeLocation" -> _, "RVersion" -> _|>]

ClearAll[rLinkInit];
Options[rLinkInit] = {
	"JRELocation" :> Automatic, 
	"TargetPlatform" :> Automatic,
	"RCommandLine" :> Automatic,
	"AddToRDataTypePath" :> None,	
	ProjectLocation :> $projectLocation,	
	ProjectJarLocation :> Automatic,	
	"NativeLibLocation" :> Automatic,
	NativeLibLocationInternal :>Automatic,
	"RHomeLocation" :> Automatic,	
	RHomeDefaultLocation :> Automatic,
	"RVersion" -> Automatic		
}

(* TODO: This is very disruptive at the moment - will not coexist with other
programs using JLink, kills the currently running jre. Need to find a better
way (run in a separate sub-kernel?)*)

(* TODO: handle possible errors in ReinstallJava[] (jre not found) *)
rLinkInit[opts:OptionsPattern[]]:=
Module[{javaReinstallQ, 
		prloc = OptionValue[ProjectLocation],
		jreloc = OptionValue["JRELocation"], 
		(* undocumented InstallR option *)
		nlibloc = OptionValue["NativeLibLocation"], 
		prJarloc = OptionValue[ProjectJarLocation],
		platform = OptionValue["TargetPlatform"],
		rHomeLocation = OptionValue["RHomeLocation"],
		rCommandLineArgs = OptionValue["RCommandLine"],
		RNativeLibsLocation	= None,
		RVersion = OptionValue["RVersion"],
        usingPersistentSettings = False, 
        usingBundledR = False						
	},	
	
	
	RDataTypeDefinitionsReload[
		Sequence@@FilterRules[{opts},Options[RDataTypeDefinitionsReload]]
	];
	autoSet[prJarloc, FileNameJoin[{prloc,"Java"}]];
	autoSet[platform, $SystemID];
	
    With[{pval = PersistentValue["RLink.RHomeAndVersion"]},
        If[ rHomeLocation === Automatic && persistentValuesValidQ[pval],
            {rHomeLocation, RVersion} = Lookup[pval, {"RHomeLocation", "RVersion"}];
            usingPersistentSettings = True;
        ]
    ];

    If[rHomeLocation === None, 
        (* 
        **  This is needed if one wants to still use bundled R when some external 
        **  R R_HOME has been cached 
        *)
        rHomeLocation = Automatic
    ];
	
	If[!$testMode && rHomeLocation =!= Automatic, 	(* User's own R distribution *)
		(* This is for JRI library location *)
		
		(* 
		 * Try to detect R version from explicit option setting. Failing that, try to 
		 * extract it from the path to R (R_HOME). All this only happens if nlibloc
		 * has not been set to a non-default value.
		 *)	
		If[!DirectoryQ[rHomeLocation],
			Throw[$Failed, error["bad_R_dir"]]
		];  
		autoSet[
			nlibloc,
			getJRILibLocation[
				detectRVersionComplete[RVersion, rHomeLocation], 
				FileNameJoin[{prloc,"SystemFiles","Libraries",platform}]
			]	
		];		
		(* 
		 * Only set for Windows since only on Windows we can add directories 
		 * to the library load path programmatically 
		 *)
		RNativeLibsLocation = 			
			Switch[platform,
				"Windows",
					FileNameJoin[{rHomeLocation,"bin","i386"}],
				"Windows-x86-64",
					FileNameJoin[{rHomeLocation,"bin","x64"}],
				_,
					None
			];	
		,
		(* else - default R distro coming with RLink *)
		(* If the user does not specify a non-standard JRI library location, fall back to defaults *)
		autoSet[nlibloc, OptionValue[NativeLibLocationInternal]];			
		RNativeLibsLocation = 
			Switch[platform,
				"Windows"|"Windows-x86-64",
					nlibloc,
				_,
					None
			];
		rHomeLocation = OptionValue[RHomeDefaultLocation];
		autoSet[rHomeLocation, getRHomeLocation[]];
        usingBundledR = True;
	];
	
	(* Print["nlibloc: ", nlibloc]; *)
	
	autoSet[rCommandLineArgs,{}];
	
	javaReinstallQ = 
	   $RWasInstalledDuringMathematicaSession 
	   || jreloc =!= Automatic 
	   || JavaLink[] === Null;
	   
	(* Print["Reinstall Java? ",javaReinstallQ]; *)
	
	If[ javaReinstallQ ,		
		$currentJavaLink = ReinstallJava[
			If[jreloc === Automatic,
				Sequence @@ {},
				(* else *)			
				CommandLine -> 	jreloc
			]		
		]
	];		
	clearFunctionsHash[];	
		
	AddToClassPath[prJarloc];
	LoadJavaClass["java.lang.System"];
	LoadJavaClass["com.wolfram.links.rlink.RLinkInit"];
	LoadJavaClass["com.wolfram.links.rlink.Environment"];
	
	If[RLinkInit`isCurrentValidSession[],
		Return[True]
	];
	
	initLogger[];
	logit["Starting parameters:"];
	logit["Platform: "<>platform];
	logit["Project location: "<>prloc];
	logit["JRI Library location: "<>nlibloc];
	logit["R_HOME location: "<>rHomeLocation];	

	logit["Setting the property java.io.dir set to "<> $temporaryDirectory];
	System`setProperty["java.io.tmpdir",$temporaryDirectory];
	logit["The property java.io.dir set to "<> System`getProperty["java.io.tmpdir"]];
				
	logit["Setting R_HOME locally (using Environment class) to "<>rHomeLocation];
	(* For R_HOME variable, Enviroment`lib@setenv works on all platforms *)
	Environment`libc@setenv["R_HOME",rHomeLocation, 1];
		
	If[RNativeLibsLocation =!= None, (* This mechanism only works on Windows *)
		logit["Appending PATH locally (using Environment class) with "<>RNativeLibsLocation];	
		Environment`libc@setenv["PATH", Environment["PATH"] <> ";" <> RNativeLibsLocation, 1]
	];		
	SetComplexClass["com.wolfram.links.rlink.dataTypes.auxiliary.Complex"];
	(* Enable Java to find the JRI native library *)		
	RLinkInit`setNativeLibLocation[nlibloc];
	logit["Starting R with these command-line arguments: "<>ToString[rCommandLineArgs]];
    With[{result = com`wolfram`links`rlink`RLinkInit`installR[rCommandLineArgs]},
        If[TrueQ[usingBundledR],
            Return[result]
        ];
        If[TrueQ[result],
            PersistentValue["RLink.RHomeAndVersion"] = <|
                "RHomeLocation" -> rHomeLocation,
                "RVersion" -> RVersion
            |>,
            (* else *)
            If[TrueQ[usingPersistentSettings],
                Remove[PersistentValue["RLink.RHomeAndVersion"]]
            ]
        ];
        result
    ]   
];


ClearAll[iUninstallR];
iUninstallR[]:=
	Module[{},
		If[Quiet@LoadJavaClass["com.wolfram.links.rlink.RLinkInit"] === $Failed,
			Return[]
		];
		If[!TrueQ[RLinkInit`isCurrentValidSession[]],
			Return[]
		];
		RLinkInit`uninstallR[];	
	];


(******************************************************************************)
(************ 			CONFIGURATIONS FOR PRODUCTION			***************)
(******************************************************************************)

	
setOptionConfiguration[rLinkInit,"Linux32Standalone", {
		ProjectJarLocation :> 		 	
		 	FileNameJoin[{$projectLocation,"Java"}],		 
		NativeLibLocationInternal :> 
			FileNameJoin[{$RRuntimeLocation,"SystemFiles","Linux","R","lib"}],
		RHomeDefaultLocation :>
			FileNameJoin[{$RRuntimeLocation,"SystemFiles","Linux","R"}]
	}];
	
	
setOptionConfiguration[rLinkInit,"Linux64Standalone", {
		ProjectJarLocation :> 		 	
		 	FileNameJoin[{$projectLocation,"Java"}],		 
		NativeLibLocationInternal :> 
			FileNameJoin[{$RRuntimeLocation,"SystemFiles","Linux-x86-64","R","lib"}],
		RHomeDefaultLocation :>
			FileNameJoin[{$RRuntimeLocation,"SystemFiles","Linux-x86-64","R"}]
	}];
	
	
setOptionConfiguration[rLinkInit,"Win32Standalone", {
		ProjectJarLocation :> 		 	
		 	FileNameJoin[{$projectLocation,"Java"}],		 
		NativeLibLocationInternal :> 
			FileNameJoin[{$RRuntimeLocation,"SystemFiles","Windows","R","bin","i386"}],
		RHomeDefaultLocation :> 
			FileNameJoin[{$RRuntimeLocation,"SystemFiles","Windows","R"}]			
	}];	
	

(* We use the same R layout for both 32 and 64 bits on Windows *)		
setOptionConfiguration[rLinkInit,"Win64Standalone", {
		ProjectJarLocation :> 		 	
		 	FileNameJoin[{$projectLocation,"Java"}],		 
		NativeLibLocationInternal :> 
			
			FileNameJoin[{$RRuntimeLocation,"SystemFiles","Windows","R","bin","x64"}],
		RHomeDefaultLocation :> 
			FileNameJoin[{$RRuntimeLocation,"SystemFiles","Windows","R"}]			
	}];		
	
		
setOptionConfiguration[rLinkInit,"Mac64Standalone", {
		ProjectJarLocation :> 		 	
		 	FileNameJoin[{$projectLocation,"Java"}],		 
		NativeLibLocationInternal :> 
			FileNameJoin[{$RRuntimeLocation,"SystemFiles","MacOSX-x86-64","R","lib","x86_64"}],
		RHomeDefaultLocation :> 
			FileNameJoin[{$RRuntimeLocation,"SystemFiles","MacOSX-x86-64","R"}]			
	}];		


$optionConfigurationRules = {	
	"Windows" ->  "Win32Standalone",
	"Linux" -> "Linux32Standalone",
	"Windows-x86-64" -> "Win64Standalone",
	"Linux-x86-64" -> "Linux64Standalone",
	"MacOSX-x86-64" -> "Mac64Standalone", 
	_ :> Throw[$Failed,error["NoOptionConfiguration"]]
};



(******************************************************************************)
(************ 				INSTALL / UNINSTALL					***************)
(******************************************************************************)


ClearAll[bundledRAutoInstall];
bundledRAutoInstall[]:=
	Module[{runtimePacletInstalledOK, noInternetConnection = False},
		runtimePacletInstalledOK = 
			Quiet@ 
				Check[
					Check[
						RLinkResourcesInstall[]
						,
						(* No internet connectivity *)
						noInternetConnection = True;
						$Failed
						,						
						{RLinkResourcesInstall::intcnct}
					],
					(* General error *)
					$Failed					
				] =!= $Failed;
		Association[{
			"installed" -> TrueQ[runtimePacletInstalledOK],
			"no_internet" -> noInternetConnection
		}]
	];


If[$testMode,
	$optionConfigurator := 
		makeConfigurator[$testConfiguration],
	(* else *)
	$optionConfigurator := 
		$SystemID /. $optionConfigurationRules /. s_String:> makeConfigurator[s]	
];	


Clear[InstallR];
InstallR::fail = "Failed to install R. The following error was encountered: `1`";

SyntaxInformation[InstallR] = {"ArgumentsPattern" -> {OptionsPattern[]}};

Options[InstallR] = {
	"JRELocation" :> Automatic, 
	"TargetPlatform" :> Automatic,
	"RCommandLine" :> Automatic,
	"AddToRDataTypePath" :> None,
	"RHomeLocation" :> Automatic,
	"EnableResourcesAutoinstall" -> True,
	"NativeLibLocation" :> Automatic,
	"RVersion" -> Automatic		
};


InstallR::nopaclet = "Could not find RLink runtime installed. Please use RLinkResourcesInstall to install it";

InstallR::badpaclet = "Could not find a path to R runtime. Possibly the installed RLink runtime is corrupt, \
try re-installing it using RLinkResourcesInstall with an option Update->True ";

InstallR::intcnct= 
"RLink runtime paclet needs to be downloaded and installed prior to first use of RLink. InstallR will attempt \
to do that automatically. However, internet connectivity is currently disabled. Use Help > Internet Connectivity... \
to enable it, and then call InstallR again";

InstallR::instfail = "Automatic install of the RLinkRuntime paclet failed. Please use RLinkResourcesInstall to install it manually";

InstallR::invldrhome = "The specified path to R home directory does not point to a valid directory";

InstallR::rundetctd ="Failed to detect the R version from the specified path to R home directory. \
Try using the \"RVersion\" option to specify the R version explicitly";

InstallR::argx = "Incorrect number or type of arguments to InstallR. \
The argument was `1`.";

InstallR::generr = "General error in InstallR. The error code is `1`";

(*
** A temporary workaround to avoid the necessity to manually do
** InstallR[] / UninstallR[] / InstallR[] on the first run
*)
InstallR[opts:OptionsPattern[]] /; !TrueQ[$RWasInstalledDuringMathematicaSession] && !TrueQ[$inInstallR]:=
    Block[{$inInstallR = True}, (* Block used to avoid infinite recursion here *)
      If[InstallR[opts] === $Failed,
        $Failed,
        (* else *)
        UninstallR[];
        InstallR[opts]
      ]
    ]

InstallR[opts:OptionsPattern[]] /; !TrueQ[$RCurrentlyInstalled]:= 
	Module[{result, errorType, handleJVMOrNativeLibCrash, crashed = False, 
		bundledRInstallInfo, needInstallBundledR
		},
		handleJVMOrNativeLibCrash = makeCrashHandler[crashed];
        needInstallBundledR = And[
			Or[
                And[
                    OptionValue["RHomeLocation"] === Automatic,
			        !persistentValuesValidQ[PersistentValue["RLink.RHomeAndVersion"]]
                ],
                OptionValue["RHomeLocation"] === None
            ],
			!RLinkRuntimeInstalledQ[], 
			TrueQ[OptionValue["EnableResourcesAutoinstall"]]
        ];
		(* 
		* Attempting to auto-install RLinkRuntime, if using a 
		* built-in R, which has not yet been installed 
		*)		
		If[needInstallBundledR,
			bundledRInstallInfo = bundledRAutoInstall[];
			If[!bundledRInstallInfo["installed"],
				If[bundledRInstallInfo["no_internet"],
					Message[InstallR::intcnct],
					(* else *)
					Message[InstallR::instfail];
				];				
				Return[$Failed]
			]		
		];
		(* Actual code to start RLink *)
		result = 
			handleJVMOrNativeLibCrash @ handleError[
				$optionConfigurator @ rLinkInit[
					Sequence @@ FilterRules[
						{opts,  Sequence @@ Options[InstallR]},
						Options[rLinkInit]
					]
				]
			];
		(* Error-handling *)
		If[!FreeQ[result, error],
			(* Exception was thrown in the Mathematica part of RLink *)
			errorType = First@Last@result;
			Switch[errorType,
				pacletFind,
					Message[InstallR::nopaclet],
				getRLinkRuntimePath,
					Message[InstallR::badpaclet],
				"bad_R_dir",
					Message[InstallR::invldrhome],
				"R_version_detection",
					Message[InstallR::rundetctd],
				_,
					Message[InstallR::generr, errorType]
			];
			Return[$Failed];
		];		
		If[crashed,
			Message[
				InstallR::fail,
				"crash in low-level RLink component or in R runtime"
			];
			Return[$Failed]
		];
		If[!crashed && !TrueQ[result],
			(* Error happening on the Java side *)
			Message[
				InstallR::fail,
				Style[RLinkInit`lastError,Red]
			];
			Return[$Failed]
		];
		(* Normal execution - final steps *)
		$RCurrentlyInstalled = True;
		$RWasInstalledDuringMathematicaSession = True;
		Update[InstallR];
		Null	
	];
	
InstallR[OptionsPattern[]]:= Null;

InstallR[args___]:= 
(
	Message[InstallR::argx,{args}];
	$Failed
);

	
Clear[UninstallR];

SyntaxInformation[UninstallR] = {"ArgumentsPattern" -> {}};

UninstallR::noinst = "RLink has not been installed or has been already uninstalled";

UninstallR[]:=
	Module[{},	
		If[$RCurrentlyInstalled,
			iUninstallR[];	
			$RCurrentlyInstalled = False;
			,
			(* else *)
			Message[UninstallR::noinst];
		]
	];

UninstallR[args___]:=
	(
		Message[General::argx, UninstallR, Length[{args}]];
		$Failed	
	);



(******************************************************************************)
(************ 	CODE HIGHLIGHTING - experimental				***************)
(******************************************************************************)


(* TODO Remove replR in favor of REValuate, which now acts exactly like this *)

ClearAll[replR];
replR[code_String /; StringTake[code, -1] === ";"] :=
  	RExecute[code];
replR[code_String] := REvaluate[code];




Clear[rcell];
rcell :=
 	Module[{},
 		SelectionMove[EvaluationNotebook[], All, Cell, AutoScroll -> False];
  		SelectionMove[EvaluationNotebook[], Previous, Cell, AutoScroll -> False];
  		NotebookWrite[
   			EvaluationNotebook[],
   			Cell[
   				TextData[{"Enter R code"}], 
   				"Program", 
   				CellEventActions -> eventActions,
    			Evaluatable -> True, 
    			CellEvaluationFunction -> (replR[#1] &), 
    			CellFrameLabels -> {{None, "R code"}, {None, None}}
    		],
   			All
   		];
  		SelectionMove[EvaluationNotebook[], All, CellContents, AutoScroll -> False];
  		SelectionMove[EvaluationNotebook[], Previous, Character];
 	];



End[]

EndPackage[]

RDataTypeDefinitionsReload[];
(* ::Package:: *)

BeginPackage["WolframScript`"]

Unprotect[System`$ScriptInputString];
System`$ScriptInputString = None;
Protect[System`$ScriptInputString];

WolframScriptOption::charinv="Unicode character set is not yet available in WolframScript.";
WolframScriptInitialize::BadArgs="Invalid argument provided to WolframScript initialization"
WolframScriptEvaluate::BadArgs="Invalid argument provided to WolframScript evaluation"
WolframScriptNotebook::nfnd = "Notebook file `1` not found."
WolframScriptExecute::BadArgs="Invalid argument provided to WolframScriptExecute"

Begin["`Private`"]

Output::ExitCode="`1`";

Options[WolframScriptExecute]={TimeConstraint->0, "format"->ScriptForm, "timedout"->$TimedOut,
		"charset"->$CharacterEncoding, "print"->False, "line"->0, "formatoptions"->"{}",
		"cloudevaluation"-> False, "verbose"->False, "source"->"code", "stdoutbuffer"->False,
		"inputfilename"->""};

(* Redirect the old private function to the semi-visible one. *)
Options[WolframScript`Private`WolframScriptExecute]=Options[WolframScriptExecute];
WolframScript`Private`WolframScriptExecute[args___] := WolframScriptExecute[args]

(*Legacy interface for versions of WolframScript released 11.0.1 and earlier.
 This interface may be removed when we drop support for 11.0.1. This is now a
 wrapper for normal execution. It was changed because it used some magic
 numbers and confusing variables, which have now been replaced with options.*)
WolframScriptExecute[code_String, scriptArgs_String, evalType_Integer, setupType_Integer,
		version_String, options:OptionsPattern[]]:=
	WolframScriptExecute[
		code,
		scriptArgs,
		version,
		"cloudevaluation"->(setupType >= 3),
		"verbose"->(setupType == 2 || setupType == 4),
		"source"-> <|1->"code", 2->"file", 3->"function"|>[evalType],
		"print"-><|0->False, 1->True, 2->All|>[OptionValue["print"]],
		options
	]

WolframScriptExecute[code_String, scriptArgs_String, version_String, options:OptionsPattern[]]:=Block[{parentLinkStore=$ParentLink,
	outputstream, $Output=$Output, $Messages=$Messages, outputsave=$Output, messagessave=$Messages},
	(*Message if character set is Unicode*)
	If[OptionValue["charset"] === "Unicode", Message[WolframScriptOption::charinv];Return[""]];

	TimeConstrained[
		Quiet@initialize[scriptArgs, options];
		evaluate[code, options]
	,
		If[OptionValue[TimeConstraint] == 0, Infinity, OptionValue[TimeConstraint]]
	,
		$ParentLink=parentLinkStore;
		OptionValue["timedout"]
	]
];

WolframScriptExecute[___]/;Message[WolframScriptExecute::BadArgs]:=$Failed


characterSetMapping = <|"adobestandard"->"AdobeStandard", "ascii"->"ASCII", "cp936"->"CP936","cp949"->"CP949","eucjp"->"EUC-JP",
    "euc"->"EUC","ibm850"->"IBM-850","iso106461"->"ISO10646-1","iso88591"->"ISO8859-1","iso88592"->"ISO8859-2",
    "iso88593"->"ISO8859-3","iso88594"->"ISO8859-4","iso88595"->"ISO8859-5","iso88596"->"ISO8859-6","iso88597"->"ISO8859-7",
    "iso88598"->"ISO8859-8","iso88599"->"ISO8859-9","iso885910"->"ISO8859-10","iso885911"->"ISO8859-11","iso885913"->"ISO8859-13",
    "iso885914"->"ISO8859-14","iso885915"->"ISO8859-15","iso885916"->"ISO8859-16","isolatin1"->"ISOLatin1","isolatin2"->"ISOLatin2",
    "isolatin3"->"ISOLatin3","isolatin4"->"ISOLatin4","isolatincyrillic"->"ISOLatinCyrillic","koi8r"->"koi8-r","macintosharabic"->"MacintoshArabic",
    "macintoshchinesesimplified"->"MacintoshChineseSimplified","macintoshchinesetraditional"->"MacintoshChineseTraditional",
    "macintoshcroatian"->"MacintoshCroatian","macintoshcyrillic"->"MacintoshCyrillic","macintoshgreek"->"MacintoshGreek",
    "macintoshhebrew"->"MacintoshHebrew","macintoshicelandic"->"MacintoshIcelandic","macintoshkorean"->"MacintoshKorean",
    "euckr"->"MacintoshKorean","macintoshnoncyrillicslavic"->"MacintoshNonCyrillicSlavic","macintoshromanian"->"MacintoshRomanian",
    "macintoshroman"->"MacintoshRoman","macintoshromanpdfexport"->"MacintoshRomanPDFExport","macintoshthai"->"MacintoshThai",
    "macintoshturkish"->"MacintoshTurkish","macintoshuktrainian"->"MacintoshUkrainian","printableascii"->"PrintableASCII",
    "shiftjis"->"ShiftJIS","symbol"->"Symbol","math1"->"Math1","math2"->"Math2","math3"->"Math3","math4"->"Math4",
    "math5"->"Math5","mathematica1"->"Mathematica1","mathematica2"->"Mathematica2","mathematica3"->"Mathematica3",
    "mathematica4"->"Mathematica4","mathematica5"->"Mathematica5","mathematica6"->"Mathematica6","mathematica7"->"Mathematica7",
    "utf8"->"UTF8","windowsansi"->"WindowsANSI","windowbaltic"->"WindowsBaltic","windowscyrillic"->"WindowsCyrillic",
    "windowseasteurope"->"WindowsEastEurope","windowsgreek"->"WindowsGreek","windowsthai"->"WindowsThai",
    "windowsturkish"->"WindowsTurkish", "ansix31101983"->"ASCII", "ansix341968"->"ASCII", "usascii"->"ASCII", "armscii8"->"ISO8859-5", 
    "asmo449"->"ISO8859-6", "cp100007"->"MacintoshCyrillic", "cp1250"->"ISO8859-2", "cp1251"->"WindowsCyrillic",
    "cp1252"->"WindowsANSI", "cp1253"->"WindowsGreek", "cp1254"->"WindowsTurkish", "cp1255"->"ISO8859-8",
    "cp1256"->"ISO8859-6", "cp1257"->"WindowsBaltic", "cp1258"->"WindowsGreek", "gb2312"->"MacintoshChineseSimplified",
    "hz"->"MacintoshChineseSimplified",  "gbk"->"MacintoshChineseSimplified",  "koi8ru"->"koi8-r", "koi8t"->"koi8-r", "koi8u"->"koi8-r",
    "big5"->"MacintoshChineseTraditional", "big5hkscs"->"MacintoshChineseTraditional", "cns"->"MacintoshChineseTraditional",
    "eten"->"MacintoshChineseTraditional","euccn"->"MacintoshChineseSimplified", "936"->"CP936", "949"->"CP949", "20932"->"EUC-JP",
    "51950"->"EUC", "850"->"IBM-850", "1200"->"ISO10646-1", "28591"->"ISO8859-1", "28592"->"ISO8859-2", "28593"->"ISO8859-3",
    "28594"->"ISO8859-4", "28595"->"ISO8859-5", "28596"->"ISO8859-6", "28597"->"ISO8859-7", "28598"->"ISO8859-8", "28599"->"ISO8859-9",
    "28600"->"ISO8859-10","874"->"ISO8859-11","28603"->"ISO8859-13","28604"->"ISO8859-14","28605"->"ISO8859-15", 
    "28606"->"ISO8859-16", "10002"->"MacintoshChineseTraditional", "10004"->"MacintoshArabic", "10008"->"MacintoshChineseSimplified", 
    "10082"->"MacintoshCroatian", "10007"->"MacintoshCyrillic", "10006"->"MacintoshGreek", "10005"->"MacintosHebrew",
    "10079"->"MacintoshIcelandic", "10003"->"MacintoshKorean", "10029"->"MacintoshNonCyrillicSlavic", "10010"->"MacintoshRomanian",
    "10000"->"MacintoshRoman","10021"->"MacintoshThai", "10081"->"MacintoshTurkish", "10017"->"MacintoshUkrainian", "932"->"ShiftJIS",
    "65001"->"UTF8","1252"->"WindowsANSI", "1257"->"WindowsBaltic", "1250"->"WindowsEastEurope", "1253"->"WindowsGreek",
    "874"->"WindowsThai", "1254"->"WindowsTurkish", "he"->"ISO8859-8"|>;


Options[initialize] = Options[WolframScriptExecute];

initialize[scriptArgs_String, p:OptionsPattern[]]:= Module[{format=OptionValue["format"], formatOptions=OptionValue["formatoptions"],
			charset=OptionValue["charset"], cloudEvaluation=TrueQ@OptionValue["cloudevaluation"], stdoutBuffer=OptionValue["stdoutbuffer"],
			inputfilename=OptionValue["inputfilename"]},
	Unprotect[$EvaluationEnvironment];
	$EvaluationEnvironment="Script";
	Protect[$EvaluationEnvironment];

	(*Finding character encoding*)
	charset = characterSetMapping[ToLowerCase[StringReplace[charset,RegularExpression["[^A-Za-z0-9]"]->""]]];
	
	(*Setting character encoding and format*)		
	If[MemberQ[$CharacterEncodings,charset],$CharacterEncoding=charset, $CharacterEncoding="ISO8859-1"];
	If[!MemberQ[$ExportFormats, format] && StringQ[format],
		If[MemberQ[Map[Hold,$PrintForms],
			ToExpression[format,InputForm,Hold]]
		,
			format=ToExpression[format]
		]
	];
	If[format == $Failed, format = ScriptForm];

	(*Redefine Exit, WolframScript wants the exit code from the User's script. This redefinition
	is used to catch the exit code and pass it back to the local program*)
	If[!MemberQ[Attributes[Exit],Locked] && (cloudEvaluation || $VersionNumber < 11.3),
		Unprotect[Exit];
		Exit[]:=Throw[wsExitCode[0], "WolframScriptExitCode"];
		Exit[code_Integer]:=Throw[wsExitCode[code], "WolframScriptExitCode"];
		Protect[Exit];
	];

	If[!MemberQ[Attributes[Quit],Locked] && ($VersionNumber < 11.3),
		Unprotect[Quit];
		Quit[]:=Throw[wsExitCode[0], "WolframScriptExitCode"];
		Quit[code_Integer]:=Throw[wsExitCode[code], "WolframScriptExitCode"];
		Protect[Quit];
	];

	(*Open a new stream to direct output at. This catches print statements*)
	If[StringQ[stdoutBuffer],
		outputstream=OpenWrite[stdoutBuffer, CharacterEncoding->$CharacterEncoding, FormatType->If[!StringQ[format], format, ScriptForm],
			BinaryFormat->If[MemberQ[$ExportFormats, format],True,False], PageWidth->Infinity];
	];
	If[!MatchQ[outputstream, _OutputStream],
		outputstream=OpenWrite[CharacterEncoding->$CharacterEncoding, FormatType->If[!StringQ[format], format, ScriptForm],
			BinaryFormat->If[MemberQ[$ExportFormats, format],True,False], PageWidth->Infinity];
	];
	$Output={outputstream};
	$Messages={outputstream};

	(*Redefine Input[] on the cloud. It is not useful for the script, and this avoids confusion.*)
	If[cloudEvaluation &&!MemberQ[Attributes[Input], Locked]&&!MemberQ[Attributes[InputString], Locked],
		Unprotect[Input, InputString];
		Clear[Input];Input[args___]:=Missing["NotAvailable"];
		Clear[InputString];InputString[args___]:=Missing["NotAvailable"];
		Protect[Input, InputString]
	];

	(*Reload $ScriptInputString if local*)
	If[!cloudEvaluation,
		Unprotect[System`$ScriptInputString];
		Quiet[System`$ScriptInputString=ReadString[System`$ScriptInputString]];
		If[!StringQ[System`$ScriptInputString], System`$ScriptInputString=""];
		Protect[System`$ScriptInputString]
	];

	(*Setup $ScriptCommandLine*)
	If[!MemberQ[Attributes[$ScriptCommandLine],Locked],
		Unprotect[$ScriptCommandLine];
		Set[$ScriptCommandLine,ToExpression[scriptArgs]];
		Protect[$ScriptCommandLine]
	];

	(*Setup $InputFileName*)
	Quiet[System`Private`$InputFileName=OptionValue["inputfilename"]];

	(*Set up formatter which applys ExportString to output*)
	Unprotect[Internal`$PrintFormatter];
	If[MemberQ[$ExportFormats, format],
		If[StringQ[formatOptions], formatOptions = ToExpression[formatOptions]];
		If[!OptionQ[formatOptions], formatOptions = {}];
		Internal`$PrintFormatter = SequenceForm@@Map[ExportString[#, format, formatOptions]&,#]&
		,If[format==="numeric",Internal`$PrintFormatter=SequenceForm@@Map[If[NumericQ[#], #, "NaN"]&,#]&]
	];
	Protect[Internal`$PrintFormatter];
];

initialize[___]/;Message[WolframScriptInitialize::BadArgs]:=$Failed;


Options[evaluate] = Options[WolframScriptExecute];

evaluate[expr_String, p:OptionsPattern[]]:=Block[{cloudEvaluation=TrueQ@OptionValue["cloudevaluation"],
		verbose=TrueQ@OptionValue["verbose"], line=TrueQ@OptionValue["line"], print=OptionValue["print"],
		source=OptionValue["source"], cloudOutput="", pipedList, exitCode=0},
	(*Check if linewise version needs to be run*)
	Block[{$ParentLink=Null, output=""},
		exitCode = Catch[
			If[line,
				pipedList = StringSplit[$ScriptInputString, {"\r\n","\n"}];
				Map[Block[{$ScriptInputString=#},
					If[print == All, ReleaseHold[If[#=!=Null,Print[#]]&/@ToExpression[expr, InputForm, Hold]]];
					If[print =!= All, output=ToExpression[expr]];
					If[print || source =!= "file",Print[output]];
				]&, pipedList]
			,
				If[print == All, ReleaseHold[If[#=!=Null,Print[#]]&/@ToExpression[expr, InputForm, Hold]]];
				If[print =!= All, output=ToExpression[expr]];
				If[print || source =!= "file",Print[output]];
			]
		, "WolframScriptExitCode"]
	];
	
	(*On cloud, read collected print statements. Then clean up and leave*)
	If[cloudEvaluation && MatchQ[outputstream, _OutputStream],
		cloudOutput = ReadString[First@outputstream];
		If[!StringQ[cloudOutput], cloudOutput=""]
	];
	Close[outputstream];
	$Output=outputsave;
	$Messages=messagessave;
	If[Head[exitCode] === wsExitCode, exitCode = exitCode[[1]], exitCode=0];
	If[exitCode =!= 0,
		If[cloudEvaluation, Print["ExitCode=", exitCode], Message[Output::ExitCode, exitCode]]
	];
	If[!verbose && cloudEvaluation, cloudOutput=ExportString[cloudOutput, "Base64"]];
	cloudOutput
];

evaluate[___]/;Message[WolframScriptEvaluate::BadArgs]:=$Failed;

openNotebook[filename_String, opts:OptionsPattern[]] := Block[{filepath,wlsnb},
	filepath = Quiet[AbsoluteFileName[filename]];
	If[FailureQ[filepath], Message[WolframScriptNotebook::nfnd, filename]; Return[""]];
	Developer`InstallFrontEnd["Server"->False];
	UsingFrontEnd[
		wlsnb = NotebookOpen[filepath, opts];
		While[MemberQ[Notebooks[], wlsnb], Pause[.5]];
	];
	""
];

generateNotebook[filename_String, args_Association, opts:OptionsPattern[]] := Block[{filepath,wlsnb,wsresult},
	filepath = Quiet[AbsoluteFileName[filename]];
	If[FailureQ[filepath], Message[WolframScriptNotebook::nfnd, filename]; Return[""]];
	Developer`InstallFrontEnd["Server"->False];
	UsingFrontEnd[
		wlsnb = Quiet[GenerateDocument[filepath, args], DeleteFile::privv];
		While[MemberQ[Notebooks[], wlsnb], Pause[.5]];
	];
	Developer`UninstallFrontEnd[];
	""
];


Attributes[evalNotebook] = {HoldFirst};
evalNotebook[cell_:Cell[expr__], opts:OptionsPattern[]] := Block[{filepath, wsresult},
	Developer`InstallFrontEnd["Server"->True];
	UsingFrontEnd[
		wlsnb = CreateDocument[cell, opts];
		wsresult = NotebookEvaluate[wlsnb];
	];
	Developer`UninstallFrontEnd[];
	wsresult
];
evalNotebook[filename_String, opts:OptionsPattern[]] := Block[{filepath, wsresult},
	filepath = Quiet[AbsoluteFileName[filename]];
	If[FailureQ[filepath], Message[WolframScriptNotebook::nfnd, filename]; Return[""]];
	Developer`InstallFrontEnd["Server"->False];
	UsingFrontEnd[
		wsresult = NotebookEvaluate[filepath];
	];
	Developer`UninstallFrontEnd[];
	wsresult
];

Attributes[buildNotebook] = {HoldFirst};
buildNotebook[expr__, opts:OptionsPattern[]] := Block[{wlsexpr,wlsnb},
	Developer`InstallFrontEnd["Server"->False];
	savedAttr = Attributes[ToString];
	Attributes[ToString] = {HoldAll, Protected};
	wlsexpr = ToString[expr, InputForm];
	Attributes[ToString] = savedAttr;
	UsingFrontEnd[
		wlsnb = CreateDocument[Cell[BoxData[wlsexpr], "Input"], opts];
		While[MemberQ[Notebooks[], wlsnb], Pause[.5]];
	];
	""
]
buildNotebook[cell:Cell[expr__], opts:OptionsPattern[]] := Block[{wlsexpr,wlsnb},
	Developer`InstallFrontEnd["Server"->False];
	UsingFrontEnd[
		wlsnb = CreateDocument[cell, opts];
		While[MemberQ[Notebooks[], wlsnb], Pause[.5]];
	];
	""
]

checkKey[elem_String] := StringStartsQ[elem, "-"]
makeAssoc[] := Block[
	{arglist = $ScriptCommandLine, i, lastWasKey = False, len, key, val, assoc = <||>},
	len = Length[arglist];
	For[i = 1, i <= len, i++,
		Which[
			lastWasKey,
			lastWasKey = False;
			AppendTo[assoc, <|key -> arglist[[i]]|>],

			checkKey[arglist[[i]]],
			key = StringDrop[arglist[[i]], 1];
			lastWasKey = True
		]
	];
	assoc
];

Attributes[makeFunctionUI] = {HoldAll};
makeFunctionUI[APIFunction[form_List, func_], opts:OptionsPattern[]] := makeFunctionUI[form, func, opts];
makeFunctionUI[FormFunction[form_List, func_], opts:OptionsPattern[]] := makeFunctionUI[form, func, opts];
makeFunctionUI[form_List, func_, opts:OptionsPattern[]] := 
	Block[{heldExpr, strExpr, header, dest, simplified, content, 
	variables, boxes, types, wlsnb, args, argsexpr, boxexpr, 
	boxLength, interp, savedAttr, messages, testinterp},
	Developer`InstallFrontEnd["Server" -> False];
	variables = First /@ form;
	types = (#[[2]]) & /@ form;
	savedAttr = Attributes[ToString];
		Attributes[ToString] = {HoldAll, Protected};
	boxes = 
	Map[StringJoin["{\"", #, "\"//OutputForm,InputField[Dynamic[", #, 
	   "var],String]}"] &, variables];
	messages = 
	Map[StringJoin["{Null,Dynamic[If[FailureQ[", #, "],ToString[", #, 
	   "], \"\"]//OutputForm]}"] &, variables];
	boxes = Riffle[boxes, messages];
	interp = 
	MapThread[
	 StringJoin[#, "=Interpreter[", #2, "][", #, 
	   "var];"] &, {variables, types}];
	testinterp = 
	StringRiffle[variables, {"(FailureQ[", "] || FailureQ[" , "])"}]; 
	Attributes[ToString] = savedAttr;
	args = 
	StringRiffle[
	 Map[StringJoin["\"", #, "\"->", #] &, variables], {"<|", ",", 
	  "|>"}];
	boxexpr = ToExpression[boxes, InputForm];
	argsexpr = ToExpression[args, InputForm, Hold];
	boxexpr[[0]] = Sequence;
	With[{argsexpr=argsexpr, interp=interp, testinterp=testinterp},
	UsingFrontEnd[
		wlsnb = CreateDocument[
			DynamicModule[{wsAPIResult = ""//OutputForm},
				Pane[
					Grid[
						{
							boxexpr,
							{
								Button["Evaluate",
									ToExpression[interp];
									If[TrueQ[ToExpression[testinterp]],
										wsAPIResult = $Failed,
										wsAPIResult = APIFunction[func][ReleaseHold[argsexpr]]
									]
									, Appearance -> "DefaultButton", 
									ImageSize -> Dynamic[CurrentValue["DefaultButtonSize"]]
								]
								, SpanFromLeft
							},
							{Null, Null},
							{
								Dynamic[wsAPIResult], SpanFromLeft
							}
						}
						,AutoDelete -> False, Editable -> False
					]
					,FrameMargins -> {{10, 10}, {10, 10}}
				]
			]
			,opts
		];
		While[MemberQ[Notebooks[], wlsnb], Pause[.5]];
	]];
	""
];


End[]

EndPackage[]

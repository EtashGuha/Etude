(* ::Package:: *)

BeginPackage["ProcessLink`"];

System`StartProcess;
System`RunProcess;
System`KillProcess;
System`ProcessInformation;
System`ProcessStatus;
System`ProcessObject;
System`Processes;
System`SystemProcessData;
System`$SystemShell;
System`SystemProcesses;

System`ProcessDirectory;
System`ProcessEnvironment;

(*TODO: move to Streams.m file*)
System`WriteLine;
System`ProcessConnection;
System`EndOfBuffer;
System`ReadString;
System`ReadLine;
(*TODO END*)


Begin["`Private`"];

Unprotect[ProcessObject,ReadString,ReadLine,WriteLine,ProcessConnection,StartProcess,Read,ReadByteArray,SystemProcesses,SystemProcessData];

(*TODO: move to Streams.m file*)
Clear[getCachedStream, ProcessConnection, ProcessObject, cachedReadString, setStreamCache, popStreamCache];

$Signals = Switch[$OperatingSystem,
	"MacOSX",<|
	(*copied from /usr/include/sys/signal.h *)
	"SIGHUP"->1,
	"SIGINT"->2,
	"SIGQUIT"->3,
	"SIGILL"->4,
	"SIGTRAP"->5,
	"SIGABRT"->6,
	"SIGIOT"->6,
	"SIGEMT"->7,
	"SIGFPE"->8,
	"SIGKILL"->9,
	"SIGBUS"->10,
	"SIGSEGV"->11,
	"SIGSYS"->12,
	"SIGPIPE"->13,
	"SIGALARM"->14,
	"SIGTERM"->15,
	"SIGURG"->16,
	"SIGSTOP"->17,
	"SIGTSTP"->18,
	"SIGCONT"->19,
	"SIGCHLD"->20,
	"SIGTTIN"->21,
	"SIGTTOU"->22,
	"SIGIO"->23,
	"SIGXCPU"->24,
	"SIGXFSZ"->25,
	"SIGVTALARM"->26,
	"SIGPROF"->27,
	"SIGWINCH"->28,
	"SIGINFO"->29,
	"SIGUSR1"->30,
	"SIGUSR2"->31|>,
	"Windows",<||>, (*No signals on windows*)
	"Linux",<|
	(*copied from /usr/include/bits/signum-generic.h *)
	"SIGINT"->2,
	"SIGILL"->4,
	"SIGABRT"->6,
	"SIGFPE"->8,
	"SIGSEGV"->11,
	"SIGTERM"->15,
	"SIGHUP"->1,
	"SIGQUIT"->3,
	"SIGTRAP"->5,
	"SIGKILL"->9,
	"SIGBUS"->10,
	"SIGSYS"->12,
	"SIGPIPE"->13,
	"SIGALRM"->14,
	"SIGURG"->16,
	"SIGSTOP"->17,
	"SIGTSTP"->18,
	"SIGCONT"->19,
	"SIGCHLD"->20,
	"SIGTTIN"->21,
	"SIGTTOU"->22,
	"SIGPOLL"->23,
	"SIGXCPU"->24,
	"SIGXFSZ"->25,
	"SIGVTALARM"->26,
	"SIGPROF"->27,
	"SIGWINCH"->28,
	"SIGUSR1"->30,
	"SIGUSR2"->31|>]

$Loaded;

If[!TrueQ[$Loaded],
	(* the cache always contains a string *)
	Clear[$StreamCache];
	$StreamCache[_] := "";

	(* this cache contains the actual streams *)
	Clear[$StreamObjectCache];
	$StreamObjectCache[___] := None;
	
	$Loaded = True;
];

(* Infinity works fine here, but it causes bug https://bugs.wolfram.com/show?number=277599
"ReadLine is incredibly slow on large file (5 megas)" *)
binaryReadBlock = 10000;

patternOfPatterns = (_Blank | _BlankSequence | _BlankNullSequence | _PatternTest | 
	_Pattern | _PatternSequence | _StringExpression | _Alternatives);

terminatorPattern = EndOfBuffer | EndOfFile | _String | patternOfPatterns;

safeStringTake[l_, n_] := If[Abs[n] > StringLength[l], l, StringTake[l, n]]
safeStringDrop[l_, n_] := If[Abs[n] > StringLength[l], "", StringDrop[l, n]]

HoldFirst[safeTimeConstrained];
safeTimeConstrained[expr_, time_, ret_] := If[time <= 0, ret, TimeConstrained[ReleaseHold[expr], time, ret]]

types = <|"StandardOutput"->iGetInputStream,"StandardInput"->iGetOutputStream, "StandardError"->iGetErrorStream|>

getCachedStream[p_, type_] := If[$StreamObjectCache[p, type] =!= None,
	$StreamObjectCache[p, type],
	$StreamObjectCache[p, type] = types[type][p]
]


(*Only detects Unicode LE BOM for now now*)
unicodeStreams = <||>
checkUnicode[p_, stream_] := (
	If[StreamPosition[stream] === 0,
		If[BinaryRead[stream, {"Byte", "Byte"}, AllowIncomplete->True] === {255, 254},
			unicodeStreams[p] = True
			,
			SetStreamPosition[stream,0];
			unicodeStreams[p] = False
		]
		,
		TrueQ[unicodeStreams[p]]
	])

(*This only works for Strings and Unicode LE. It should be replaced
	by using Read with Unicode support when available.*)
convertUnicode[unformatted_String] := ImportString[unformatted, "Text", CharacterEncoding->"Unicode", ByteOrdering->-1]
convertUnicode[unformatted___] := unformatted

iGetOutputStream[p_] := If[NumberQ[p], 
	OpenWrite["in:" <> ToString[p], Method -> "RunPipe", BinaryFormat -> True], 
	Message[StartProcess::stopen, "standard output"]; $Failed]

iGetInputStream[p_] := If[NumberQ[p],
	OpenRead[stdoutFileNames[p], Method->"File", BinaryFormat->True, AppendCheck->True],
	Message[StartProcess::stopen, "standard output"]; $Failed]

iGetErrorStream[p_] := If[NumberQ[p],
	OpenRead[stderrFileNames[p], Method->"File", BinaryFormat->True, AppendCheck->True],
	Message[StartProcess::stopen, "standard error"]; $Failed]

processStreamQ[pr_ProcessObject] := subprocessQ[pr];
processStreamQ[InputStream[s_String, ___]] := StringStartsQ[s, $TempPrefix];
processStreamQ[st_OutputStream] := MemberQ[Method/.Options[st], "RunPipe"];
processStreamQ[___] := False;

popStreamCache[st_InputStream] := With[{c = $StreamCache[st]},
	Quiet[Unset[$StreamCache[st]], Unset::norep];
	c
]
popStreamCache[ProcessObject[<|"ManagedProcess" -> True, "UID" -> p_,___|>]] := popStreamCache[getCachedStream[p, "StandardOutput"]]

popStreamCache[___] := $Failed

setStreamCache[st_InputStream, value_String] := $StreamCache[st] = value;


setStreamCache[ProcessObject[<|"ManagedProcess" -> True, "UID" -> p_,___|>], value_String] := setStreamCache[getCachedStream[p, "StandardOutput"], value]

setStreamCache[___] := $Failed

(* read all from a stream (non-blocking), including its cache, and empty the cache
returns _String | EndOfFile | $Failed *)
cachedReadString[st_InputStream] := With[{l = BinaryReadListEOF[st]},
	If[l === EndOfFile,
		With[{c = popStreamCache[st]}, If[c =!= "", c, EndOfFile]],
		popStreamCache[st] <> l
	]
]
cachedReadString[ProcessObject[<|"ManagedProcess" -> True, "UID" -> p_,___|>]] := cachedReadString[getCachedStream[p, "StandardOutput"]]
cachedReadString[_] := $Failed


(* non-blocking binary read list, but can return EOF (output is _String | EOF)  *)
BinaryReadListEOF[st_] := Block[{l},
	l = BinaryReadList[st, "Character8", binaryReadBlock, "AllowIncomplete" -> True];
	If[l =!= {}, Return[StringJoin@l]];
	If[!processStreamQ[st] || !processRunningQ[findProcessFromStream[st]],
		EndOfFile
		,
		""
	]
]



getMaxStringLength[_] := $Failed
getMaxStringLength[str_String] := StringLength[str]
getMaxStringLength[alt_Alternatives] := With[{lens = getMaxStringLength /@ List@@alt},
	If[Cases[lens, $Failed, {1}, 1] === {},
		Max[lens],
		$Failed
	]
]

ReadString::notfound = "Specified terminator not found.";

ReadString::iterm = "Invalid terminator value `1`.";

(* higher performance implementation of ReadString, only works on terminators that are strings or string alternatives *)
(* GenericGetString reads the stream every time, storing the result in a buffer, and then
uses Position in the full buffer to look for terminator. That is very wasteful, when the buffer
becomes too big, Position gets slower and slower. quickGetString only looks for the terminator in
the last n bytes of the buffer + the most recently read characters, where n is the string length
of the terminator.
 *)
processStreamQuickGetString[ProcessObject[<|"ManagedProcess" -> True, "UID" -> p_,___|>], terminator_, constraintTime_] := processStreamQuickGetString[getCachedStream[p, "StandardOutput"], terminator, constraintTime];
processStreamQuickGetString[st_, terminator_, constraintTime_] := Block[
	{termlen, oldbuf = Internal`Bag[], lastbuf = "", str, pos = {}, first = True, startTime, readTmp, aborted = False, doublecheck = True, po = findProcessFromStream[st]},
	
	startTime = SessionTime[];
	
	(* oldbuf = buffer of all text that was already checked
	lastbuf = some extra remaining chars that can still contain parts of the terminator
	str = most recently read buffer *)
	termlen = getMaxStringLength[terminator];
	While[pos === {} && !aborted,
		If[first,
			str = popStreamCache[st];
			first = False;
			,
			readTmp = safeTimeConstrained[BinaryReadList[st, AllowIncomplete->True], constraintTime - (SessionTime[] - startTime), $TimedOut];
			str = Switch[readTmp,
				$TimedOut, aborted = True; "",
				{}, If[!processRunningQ[po] && Developer`EndOfStreamQ[st], EndOfFile,""],
				_, FromCharacterCode[readTmp]
			];
		];
		If[!StringQ[str],
			Internal`StuffBag[oldbuf, lastbuf];
			lastbuf = StringJoin@@Internal`BagPart[oldbuf, All];
			If[lastbuf =!= "",
				Message[ReadString::notfound]
				,
				If[str == EndOfFile, Return[EndOfFile]]
			];
			Return[lastbuf];
		];
		lastbuf = lastbuf <> str;
		str = "";
		
		pos = StringPosition[lastbuf, terminator];
		If[pos === {},
			Internal`StuffBag[oldbuf, safeStringDrop[lastbuf, -(termlen - 1)]];
			lastbuf = safeStringTake[lastbuf, -(termlen - 1)];
		];
		
		If[constraintTime <= SessionTime[] - startTime, aborted = True];
	];
	
	If[pos =!= {},
		setStreamCache[st, StringDrop[lastbuf, pos[[1, 2]]]];
		Internal`StuffBag[oldbuf, StringTake[lastbuf, pos[[1, 1]] -1]]
		,
		Internal`StuffBag[oldbuf, lastbuf]
	];
	StringJoin@@Internal`BagPart[oldbuf, All]
]

delimitersToStringList[term_]:= Switch[Head[term], 
	String, {term}, 
	Symbol,{},
	_,Complement[Apply[List,term],{EndOfFile}]];
 
(* simple generic version of ReadString, can work for any delimiter, but it is most likely very
inefficient, except for simple terminators like EndOfFile and EndOfBuffer *)
genericGetString[st_, terminator_, constraintTime_] := Block[{str, buff = "", bag = Internal`Bag[], pos = {}, startTime, aborted = False},
	startTime = SessionTime[];
	While[pos === {} && !aborted,
		str = cachedReadString[st];
		If[!StringQ[str],
			buff = StringJoin@@Internal`BagPart[bag, All];
			If[buff === "",
				If[processStreamQ[st], Return[""], Return[str]];
				,
				Message[ReadString::notfound];
				Return[buff]
			];
		];
		Internal`StuffBag[bag, str];
		pos = StringPosition[str, terminator, 1];

		If[constraintTime <= SessionTime[] - startTime, aborted = True];
	];

	buff = StringJoin@@Internal`BagPart[bag, All];

	If[pos =!= {},
		pos = StringPosition[buff, terminator, 1];
		setStreamCache[st, StringDrop[buff, pos[[1, 2]]]];
		StringTake[buff, pos[[1, 1]] - 1]
		,
		buff
	]
]
genericGetString[st_, EndOfBuffer, constraintTime_] := cachedReadString[st]
genericGetString[st_, EndOfFile, constraintTime_] := Read[st, Record, RecordSeparators->{}]

(* reads full contents until terminator is found; by default term is EndOfFile so it reads the stream fully *)
Options[ReadString] = {
	TimeConstraint -> Infinity
};

subprocessQ[process_ProcessObject]:= 
	If[process["ManagedProcess"],True, (
		MemberQ[process["PID"],pidforuid/@$RunningProcesses]
	)]
internalizeProcess[process_ProcessObject]:= If[process["ManagedProcess"]==True,process,ProcessObject[<|"ManagedProcess"->True,"UID"->Cases[ProcessLink`Private`$RunningProcesses, x_Integer/;pidforuid[process[x]]==process["PID"]][[1]]|>]];

ReadString[st:(_ProcessObject | _InputStream), terminator : terminatorPattern : EndOfFile, opts:OptionsPattern[]] := With[
	{termlen = getMaxStringLength[terminator]},
	If[NumberQ[termlen],
		If[MatchQ[st,_ProcessObject],
			If[!subprocessQ[st],Message[ProcessObject::noio,"I/O can only be performed on processes spawned by the Wolfram Engine."];Return[None]];
			processStreamQuickGetString[internalizeProcess[st], terminator, OptionValue[TimeConstraint]],
			Read[st, Record, RecordSeparators->delimitersToStringList[terminator]]
		],
		genericGetString[st, terminator, OptionValue[TimeConstraint]]
	]
]

ReadString[URL[str_String], opts___] := ReadString[str, opts]; 
ReadString[File[str_String], opts___] := ReadString[str, opts]; 

readStringStreams=Association[{}];

getFileStream[st_String]:=Module[{res},res=readStringStreams[st];
    If[Head[res]===Missing,
	    res=OpenRead[st,BinaryFormat->True];
		    If[!MatchQ[res,_InputStream],Return[$Failed],
			    readStringStreams=Append[readStringStreams,st->res]]];
res]

deleteStream[st_String] := readStringStreams=KeyDrop[readStringStreams,st]


ReadString[filename_String, opts___] := Module[{file, result},
	file = getFileStream[filename];
	If[!MatchQ[file, _InputStream], Return[$Failed]];
	result = ReadString[file, opts];
	If[Replace[{opts},{{term:Except[_?OptionQ],___}:>term,_:>EndOfFile}] === EndOfFile, deleteStream[filename];Close[file]];
	result
]

stdoutFileNames = <||>;
stderrFileNames = <||>;

findProcessFromStream[s_InputStream] := Block[{streamName = First@s, id},
	If[StringStartsQ[streamName, StringJoin[$TempPrefix, "out-"]],
		id = PositionIndex[stdoutFileNames][streamName];
		If[ListQ[id], Return[ProcessObject[<|"ManagedProcess"->True,"UID"->First@id|>]]]
	];
	If[StringStartsQ[streamName, StringJoin[$TempPrefix, "err-"]],
		id = PositionIndex[stderrFileNames][streamName];
		If[ListQ[id], Return[ProcessObject[<|"ManagedProcess"->True,"UID"->First@id|>]]]
	];
	ProcessObject[<|"ManagedProcess" -> True, "UID" -> -1|>]
]
findProcessFromStream[process_ProcessObject] := process

ReadString[st:(_ProcessObject | _InputStream), terminator:_] /; (Message[ReadString::iterm, terminator]; False) := None

ReadString[URL[a___], opts___] /; (Message[ReadString::strmi, URL[a]]; False) := None; 
ReadString[File[a___], opts___] /; (Message[ReadString::strmi, File[a]]; False) := None; 

ReadString[args___] /; (If[Length[{args}] < 1 || Length[{args}] > 2, Message[ReadString::argt, ReadString, Length[{args}], 1, 2]]; False) := None

(*  reads line until completion, blocks; can return EndOfFile if no more to read *)
Options[ReadLine] = {
	TimeConstraint -> Infinity
};

ReadLine[URL[str_String], opts___] := ReadLine[str, opts]; 
ReadLine[File[str_String], opts___] := ReadLine[str, opts]; 
ReadLine[st:(_ProcessObject | _InputStream | _String), opts:OptionsPattern[]] :=
	Quiet[ReadString[st, ("\n" | "\r\n" | "\r" ), TimeConstraint -> OptionValue[TimeConstraint]], ReadString::notfound]

ReadLine[URL[a___], opts___] /; (Message[ReadLine::strmi, URL[a]]; False) := None; 
ReadLine[File[a___], opts___] /; (Message[ReadLine::strmi, File[a]]; False) := None; 

ReadLine[args___] /; (If[Length[{args}] =!= 1, Message[ReadLine::argx, ReadLine, Length[{args}]]]; False) := None


getInt32[orig_List] := getInt32[orig, 1];
getInt32[orig_List, start_Integer] := Module[
	{tmp, t},
	If[Length[orig] < start+3, Return[Missing["NotAvailable"]]];
	t = orig[[start;;start+3]];
	tmp =
		BitShiftLeft[t[[4]], 24] + BitShiftLeft[t[[3]], 16] +
		BitShiftLeft[t[[2]], 8] + t[[1]];
	If[tmp == 4294967295,Missing["NotAvailable"],tmp]
];
getInt32[___] := Missing["NotAvailable"];

getInt64[orig_List] := getInt64[orig, 1];
getInt64[orig_List, start_Integer] := Module[
	{tmp, t},
	If[Length[orig] < start+7, Return[Missing["NotAvailable"]]];
	t = orig[[start;;start+7]];
	tmp =
		BitShiftLeft[t[[8]], 56] + BitShiftLeft[t[[7]], 48] +
		BitShiftLeft[t[[6]], 40] + BitShiftLeft[t[[5]], 32] +
		BitShiftLeft[t[[4]], 24] + BitShiftLeft[t[[3]], 16] +
		BitShiftLeft[t[[2]], 8] + t[[1]];
	If[tmp == 18446744073709551615,Missing["NotAvailable"],tmp]
];
getInt64[___] := Missing["NotAvailable"];

getString[t_List, start_Integer, length_Integer] := Module[{},
	If[length==0 || (Length[t] < start+length-1) ,Return[Missing["NotAvailable"]]];
	FromCharacterCode[t[[start ;; start+length-1]]]
]

findByteUnit[bytes_Integer]:=Block[{QuantityUnits`$AutomaticUnitTimes = False,i,k=1000,sizes={"Bytes","Kilobytes","Megabytes","Gigabytes","Terabytes","Petabytes","Exabytes","Zetabytes","Yottabytes"}},
	If[bytes==0,Return[Quantity[bytes, "Bytes"]];];
	If[bytes >= k^9, Return[Quantity[N[bytes/k^8], "Yobibytes"]]];
	i=Floor[Log[k,bytes]];
	Quantity[N[bytes/k^i], sizes[[i+1]]]
]

getMemory[t_List] :=  getMemory[t, 1];
getMemory[t_List, start_Integer] :=
 Module[{amount = getInt64[t, start]},
  If[MissingQ[amount], amount, findByteUnit[amount]]]
getMemory[___] := Missing["NotAvailable"];

getTime[t_List] := getTime[t, 1];
getTime[t_List, start_Integer] := Module[{s,us},
(
	s = getInt64[t, start];
	us = getInt64[t, start+8];
	If[MissingQ[s]||MissingQ[us],Missing["NotAvailable"],N[s+(us/1000000)]]
)]
getTime[___] := Missing["NotAvailable"];

formatTime[t_] := Block[{d, h, m, s, dUnit = "Days", hUnit = "Hours", mUnit = "Minutes", sUnit = "Seconds"},
	If[!MatchQ[t, _?NumberQ], Return[t]];
	s = SetPrecision[Mod[t, 60], 5];
	m = Mod[Floor[t/60], 60];
	h = Mod[Floor[t/3600], 24];
	d = Floor[t/86400];
	If[d <= 0, d = Nothing; dUnit = Nothing];
	If[h <= 0, h = Nothing; hUnit = Nothing];
	If[m <= 0, m = Nothing; mUnit = Nothing];
	Quantity[MixedMagnitude[{d, h, m, s}], MixedUnit[{dUnit, hUnit, mUnit, sUnit}]]
]

readProcess[t_List] := 
 Module[{prog = 1, len = 0,tmp=0, data = <||>},
	data["PID"] = getInt32[t, prog]; prog+=4;
	data["PPID"] = getInt32[t, prog]; prog+=4;
	data["Memory"] = getMemory[t, prog]; prog+=8;
	data["StartTime"] = getInt64[t, prog]; prog+=8;
	If[!MissingQ[data["StartTime"]],
		data["RealTime"] = UnixTime[]-data["StartTime"],
		data["RealTime"] = Missing["NotAvailable"]
	];
	data["SystemTime"] = getTime[t, prog]; prog+=16;
	data["UserTime"] = getTime[t, prog]; prog+=16;
	data["Threads"] = getInt32[t, prog]; prog+=4;
	len = getInt32[t, prog]; prog+=4;
	data["User"] = getString[t, prog, len]; prog+=len;
	len = getInt32[t, prog]; prog+=4;
    data["Program"] = getString[t, prog, len]; prog+=len;
	len = getInt32[t, prog]; prog+=4;
	data["Path"] = getString[t, prog, len]; prog+=len;
	data
]

getProcAssocForPID[pid_Integer]:= Module[{res},
	(
		res = getProcessInfoForPID[pid];
		If[MatchQ[res,_LibraryFunctionError],res,readProcess[Flatten[Normal[res]]]]
	)]
	
getProcAssocForUID[uid_Integer]:= Module[{res},
	(
		res = getProcessInfoForUID[uid];
		If[MatchQ[res,_LibraryFunctionError],res,readProcess[Flatten[Normal[res]]]]
	)]


$StringFields={"Program","Path","User"}

$ComparisonOperators = {EqualTo, GreaterEqualThan, GreaterThan, LessEqualThan, LessThan, UnequalTo}
timeProperties={"UserTime", "SystemTime", "RealTime"}


filter[procs_List,rule_Rule] := Block[{lhs=rule[[1]], rhs=rule[[2]], comparison, checkType, unixTime, dateObject},

	Which[
		MemberQ[$StringFields,lhs] && MatchQ[rhs, (_String | _StringExpression | _RegularExpression)],
			Select[procs,(If[StringQ[#[lhs]], StringContainsQ[#[lhs], rhs], False])&]
		,
		MemberQ[$ComparisonOperators, Head[rhs]],
			checkType = NumberQ;
			comparison = rhs;
			Which[
				lhs === "Memory" && MatchQ[rhs, _[_?NumberQ]],
					comparison = Head[rhs][Quantity[rhs[[1]], "Bytes"]];
					checkType = QuantityQ
				,
				MemberQ[{"UserTime", "SystemTime", "RealTime"}, lhs] && MatchQ[rhs, _[_Quantity]],
					comparison = UnitConvert[rhs[[1]], "Seconds"];
					If[!QuantityQ[comparison], Return[$Failed]];
					comparison = Head[rhs][QuantityMagnitude[comparison]];
				,
				MemberQ[{"StartTime"}, lhs] && MatchQ[rhs, _[_DateObject] | _[_?NumberQ]],
					checkType = (MatchQ[#, _?NumberQ | _DateObject])&;
					If[DateObjectQ[rhs[[1]]],
						unixTime = UnixTime[rhs[[1]]];
						dateObject = rhs[[1]]
						,
						unixTime = rhs[[1]];
						dateObject = FromUnixTime[rhs[[1]]];
					];
					comparison = (If[NumberQ[#], Head[rhs][unixTime][#], Head[rhs][dateObject][#]])&
				,
				MatchQ[rhs, _[_Quantity]],
					checkType = QuantityQ
			];
			Select[procs, (If[checkType[#[lhs]], comparison[#[lhs]]])&]
		,
		True,
			Select[procs,(MatchQ[#[lhs], rhs])&]
	]
]

buildDataset[pid_Integer]:= Module[{assoc},
	(
		assoc = ProcessLink`Private`getProcAssocForPID[pid];
		If[MatchQ[assoc,_LibraryFunctionError],
			Association[#->Missing["NotAvailable"]&/@{"PPID","User","Path","Memory","Threads","StartTime","UserTime","SystemTime","RealTime"}],
			assoc["StartTime"] = 
				If[
					MissingQ[assoc["StartTime"]],
					assoc["StartTime"],
					FromUnixTime[assoc["StartTime"]]
				];
			assoc["UserTime"] = formatTime[assoc["UserTime"]];
			assoc["SystemTime"] = formatTime[assoc["SystemTime"]];
			assoc["RealTime"] = formatTime[assoc["RealTime"]];
			assoc
		]
	)]
renderForDataset[pids_List] := Module[{out}, 
	(
		out = buildDataset/@pids;
		DeleteCases[out,Null]
	)]

If[TrueQ[$CloudEvaluation] && !TrueQ[Lookup[CloudSystem`KernelInitialize`$ConfigurationProperties, "AllowSystemProcessFunctionality"]],
(* Running in cloud environment, define dummy functions that tell the user this functionality is not yet available. *)
	System`SystemProcesses[___] := (Message[General::cloudf, HoldForm@SystemProcesses]; $Failed);
	System`SystemProcessData[___] := (Message[General::cloudf, HoldForm@SystemProcessData]; $Failed);
,
(* Else define as usual *)
	System`SystemProcessData[]:= Dataset[renderForDataset@Sort[ProcessLink`Private`getProcs[True]]];
	System`SystemProcessData[All]:= Dataset[renderForDataset@Sort[ProcessLink`Private`getProcs[False]]];

	System`SystemProcessData[p: (_String | _StringExpression | _RegularExpression)] := Block[{processes, filtered},
		processes = renderForDataset@Sort[ProcessLink`Private`getProcs[False]];
		filtered = Select[processes, If[StringQ[#["Path"]], StringContainsQ[#["Path"], p], False] &];
		Dataset[filtered]
	];

	System`SystemProcessData[p : {Repeated[_ -> _]}]:= Dataset[Fold[filter,renderForDataset@Sort[ProcessLink`Private`getProcs[False]], p]];
	System`SystemProcessData[p_Rule]:= Dataset[filter[renderForDataset@Sort[ProcessLink`Private`getProcs[False]], p]];
	makeExternalProcessObjects[pids_List] := DeleteCases[Module[{data = getProcAssocForPID[#]},	If[MatchQ[data,_LibraryFunctionError],Null,ProcessObject[<|"ManagedProcess" -> False, "PID" -> #, "PPID"->data["PPID"],"Program" -> data["Program"], "Path"->data["Path"],"User"->data["User"],"StartTime"->data["StartTime"]|>]]]&/@pids,Null];
	System`SystemProcesses[] := makeExternalProcessObjects@Sort[ProcessLink`Private`getProcs[True]];
	System`SystemProcesses[All] := makeExternalProcessObjects@Sort[ProcessLink`Private`getProcs[False]];

	System`SystemProcesses[p: (_String | _StringExpression | _RegularExpression)] := Block[{processes},
		processes = makeExternalProcessObjects@Sort[ProcessLink`Private`getProcs[False]];
		Select[processes, If[StringQ[#["Path"]], StringContainsQ[#["Path"], p], False] &]
	];

	System`SystemProcesses[p_Rule]:= filter[makeExternalProcessObjects@Sort[ProcessLink`Private`getProcs[False]], p];
	System`SystemProcesses[p : {Repeated[_ -> _]}]:= Fold[filter,makeExternalProcessObjects@Sort[ProcessLink`Private`getProcs[False]], p];
]; (* End of section marked as disabled on cloud *)

ProcessObject::noio = "I/O can only be performed on processes spawned by the Wolfram Engine.";

ProcessObject::noprop = "`1` is not a valid property for ProcessObject";

ProcessObject::argx = "ProcessObject called with `1` arguments. 1 argument is expected.";

ProcessObject[data_Association]["Dataset"] := 
	Dataset[
		Association[
				(
				#1 -> ProcessObject[data][#1]
				)& /@ DeleteCases[ProcessObject[data]["Properties"], "Dataset"]
			]
		];

ProcessObject[data_Association]["StandardInput"] := Quiet[ProcessConnection[ProcessObject[data], "StandardInput"]];

ProcessObject[data_Association]["StandardOutput"] := Quiet[ProcessConnection[ProcessObject[data], "StandardOutput"]];

ProcessObject[data_Association]["StandardError"] := Quiet[ProcessConnection[ProcessObject[data], "StandardError"]];

getIDKey[procAssoc_, procKey_] := Block[{},
	If[!AssociationQ[procAssoc],
		If[MemberQ[propertiesList, procKey],
			Return[Missing["NotAvailable"]]
			,
			Message[ProcessObject::noprop, procKey];
			Return[]
		]
	];
	If[KeyExistsQ[procAssoc, procKey],
		Return[procAssoc[procKey]];,
		Message[ProcessObject::noprop, procKey];
		Return[];
	];
]

ProcessObject[data_Association][key_] :=
	If[
		KeyExistsQ[data,key],
		data[key],
		If[
			data["ManagedProcess"]==True,
			getIDKey[getProcAssocForUID[data["UID"]], key],
			getIDKey[getProcAssocForPID[data["PID"]], key]
		]
	];

ProcessObject[data_Association][x___] := Message[ProcessObject::argx, Length[{x}]];

propertiesList = {"Program","PID","PPID","User","Path","Memory","Threads", "StartTime","UserTime","SystemTime","RealTime", "Dataset", "StandardInput", "StandardOutput", "StandardError"};
managedPropertiesList = {"Program","UID","PID","PPID","User","Path","Memory","Threads","StartTime","UserTime","SystemTime","RealTime", "Dataset", "StandardInput", "StandardOutput", "StandardError"};

ProcessObject[data : <|"ManagedProcess"->True,___|>]["Properties"] := managedPropertiesList;

ProcessObject[data : <|"ManagedProcess"->False,___|>]["Properties"] := propertiesList;

patchMissingData[cachedData_Association, fetchedData_Association] := 
 Merge[{cachedData, fetchedData}, 
  Function[val, 
   If[Length[val] > 1, 
    If[MatchQ[val[[2]], _Missing], val[[1]], val[[2]]], val[[1]]]]]

ProcessObject/:MakeBoxes[p: ProcessObject[props_Association],StandardForm]/;
	If[props["ManagedProcess"]==True,isValidProcess[p],ProcessLink`Private`hasProcess[props["PID"]]] || Length[Keys[props]] > 2 :=
		(
BoxForm`ArrangeSummaryBox[
		ProcessObject,
		p,		
		None,
		{BoxForm`SummaryItem[{"Program: ",p["Program"]}],
			BoxForm`SummaryItem[{"PID: ",p["PID"]}]},
		Module[{fetch=patchMissingData[props,buildDataset[p["PID"]]]},
			BoxForm`SummaryItem/@{
				{"Parent PID: ",fetch["PPID"]},
				{"User: ",fetch["User"]},
				{"Path: ",fetch["Path"]},
				{"Memory:",fetch["Memory"]},
				{"Threads: ",fetch["Threads"]},
				{"Start Time: ",fetch["StartTime"]},
				{"System Time: ",fetch["SystemTime"]},
				{"User Time: ",fetch["UserTime"]},
				{"Real Time: ",fetch["RealTime"]}}
			]
		,StandardForm]
		)

ProcessObject /: BinaryReadList[pr_ProcessObject, args___, opts:OptionsPattern[]] := Block[{cachedStream},
	If[!subprocessQ[pr],Message[ProcessObject::noio,"I/O can only be performed on processes spawned by the Wolfram Engine."];Abort[]];
	With[{p = internalizeProcess[pr]},
	(
		cachedStream = getCachedStream[p["UID"], "StandardOutput"];
		If[TrueQ@OptionValue["AllowIncomplete"],
			While[Developer`EndOfStreamQ[cachedStream] && processRunningQ[internalizeProcess[p]]];
			,
			While[processRunningQ[internalizeProcess[p]]]
		];
		BinaryReadList[cachedStream, args, opts]
	)]
]

ProcessObject /: BinaryRead[pr_ProcessObject, args___, opts:OptionsPattern[]] := Block[{cachedStream},
	If[!subprocessQ[pr],Message[ProcessObject::noio,"I/O can only be performed on processes spawned by the Wolfram Engine."];Abort[]];
	With[{p = internalizeProcess[pr]},
	(
		cachedStream=getCachedStream[p["UID"], "StandardOutput"];
		While[Developer`EndOfStreamQ[cachedStream] && processRunningQ[p]];
		BinaryRead[cachedStream, args, opts]
	)]
]

ProcessObject /: ReadByteArray[pr_ProcessObject, args___, opts:OptionsPattern[]] := (
	If[!subprocessQ[pr],Message[ProcessObject::noio,"I/O can only be performed on processes spawned by the Wolfram Engine."];Abort[]];
	ReadByteArray[getCachedStream[internalizeProcess[pr]["UID"], "StandardOutput"], args, opts]
)

ReadByteArray[s_InputStream, args___, opts:OptionsPattern[]]/;(processStreamQ[s]&&processStreamRead=!=True) := Block[{p = findProcessFromStream[s], processStreamRead = True, res},
	res = ReadByteArray[s, args, opts];
	If[res === {} && Developer`EndOfStreamQ[s] && !processRunningQ[p],
		EndOfFile,
		res
	]
]

ProcessObject /: BinaryWrite[pr_ProcessObject, args___] := (
	If[!subprocessQ[pr],Message[ProcessObject::noio,"I/O can only be performed on processes spawned by the Wolfram Engine."];Abort[]];
	BinaryWrite[getCachedStream[internalizeProcess[pr]["UID"], "StandardInput"], args]
)

ProcessConnection[pr_ProcessObject, stream : ("StandardOutput" | "StandardInput" | "StandardError")] := (
	If[!subprocessQ[pr],Message[ProcessObject::noio,"I/O can only be performed on processes spawned by the Wolfram Engine."];Abort[]];
	getCachedStream[internalizeProcess[pr]["UID"], stream]
)

ProcessConnection[args___] /; (If[Length[{args}] =!= 2, Message[ProcessConnection::argrx, ProcessConnection, Length[{args}], 2]]; False) := None

ProcessObject /: Import[pr_ProcessObject, args___] := (
	If[!subprocessQ[pr],Message[ProcessObject::noio,"I/O can only be performed on processes spawned by the Wolfram Engine."];Abort[]];
	Import[getCachedStream[internalizeProcess[pr]["UID"], "StandardOutput"], args]
)

ProcessObject /: Read[pr_ProcessObject, args___, opts:OptionsPattern[]] := (
	If[!subprocessQ[pr],Message[ProcessObject::noio,"I/O can only be performed on processes spawned by the Wolfram Engine."];Abort[]];
	Read[getCachedStream[internalizeProcess[pr]["UID"], "StandardOutput"], args, opts]
)

(*To maintain behavior, Process reads must block for new data or until Process has ended. *)
Read[s_InputStream, type : (Record | Word), args___, opts:OptionsPattern[]]/;(processStreamQ[s]&&processStreamRead=!=True) := Block[{p = findProcessFromStream[s], processStreamRead = True},
	While[processRunningQ[p]];
	If[checkUnicode[p, s],
		convertUnicode[Read[s, type, args, opts]]
		,
		Read[s, type, args, opts]
	]
]

Read[s_InputStream, args___, opts:OptionsPattern[]]/;(processStreamQ[s]&&processStreamRead=!=True) := Block[{p = findProcessFromStream[s], processStreamRead = True},
	While[Developer`EndOfStreamQ[s] && processRunningQ[p]];
	If[checkUnicode[p, s],
		convertUnicode[Read[s, args, opts]]
		,
		Read[s, args, opts]
	]
]

ProcessObject /: Write[pr_ProcessObject, args___] := (
	If[!subprocessQ[pr],Message[ProcessObject::noio,"I/O can only be performed on processes spawned by the Wolfram Engine."];Abort[]];
	Write[getCachedStream[internalizeProcess[pr]["UID"], "StandardInput"], args]
 )

ProcessObject /: WriteString[pr_ProcessObject, args___] := 	(
	If[!subprocessQ[pr],Message[ProcessObject::noio,"I/O can only be performed on processes spawned by the Wolfram Engine."];Abort[]];
	BinaryWrite[pr, args]
)

WriteLine[File[st_String], str__] := WriteLine[st, str];
WriteLine[st:(_ProcessObject | _OutputStream | _String), str__] := WriteString[st, StringRiffle[{str}, {"", "\n", "\n"}]];

WriteLine[st_List, str__] := Block[{data = StringRiffle[{str}, {"", "\n", "\n"}]}, WriteString[st, data]/@st;]

WriteLine[File[a___], opts___] /; (Message[WriteLine::strmi, File[a]]; False) := None; 
WriteLine[args___] /; (If[Length[{args}] =!= 2, Message[WriteLine::argrx, WriteLine, Length[{args}], 2]]; False) := None

ReadString::usage = "ReadString[\"file\"] reads the complete contents of a file and returns it as a string.\nReadString[stream] reads everything from a stream and returns it as a string.\nReadString[proc] reads everything generated by an external process and returns it as a string.\nReadString[src,term] reads until the terminator term is encountered.";
ProcessConnection::usage = "ProcessConnection[proc, \"stream\"] returns the stream object for a given stream.";
WriteLine::usage = "WriteLine[stream, \"string\"] writes \"string\", followed by a newline, to the specified output stream.\nWriteLine[proc, \"string\"] writes \"string\" to an external process proc.";
EndOfBuffer::usage = "EndOfBuffer is a symbol that represents the end of currently available data in the buffer for a process or stream.";
ReadLine::usage = "ReadLine[stream] reads a line of text from a stream and returns it as a string.\nReadLine[proc] reads a line of text generated by an external process and returns it as a string.";


SetAttributes[
	{ProcessConnection, ReadLine, ReadString, WriteLine, ReadByteArray},
	{Protected, ReadProtected}];
(*TODO END*)
Unprotect[KillProcess, ProcessInformation, ProcessObject, Processes, ProcessStatus, RunProcess, StartProcess]
Clear[runProcess, KillProcess, RunProcess, startProcess, commandToAbsolutePath];

WindowsQ = !StringFreeQ[$System, "Windows"];

$SystemShell = If[WindowsQ, "cmd", "/bin/sh"];

addRunningProcess[rp_] := ($RunningProcesses = Flatten[{$RunningProcesses, rp["UID"]}];);
removeRunningProcesses[rps: {___ProcessObject}] := ($RunningProcesses = Complement[$RunningProcesses,#["UID"]&/@rps]);  
removeRunningProcesses[rps: {___Integer}] := ($RunningProcesses = Complement[$RunningProcesses, rps]);  

isValidProcess[proc_]:= MemberQ[$RunningProcesses, proc["UID"]];
isNotValidProcess[proc_]:= Not[isValidProcess[proc]];
processRunningQ[p: ProcessObject[<|"ManagedProcess" -> True, "UID" ->id_,___|>]] := If[isValidProcess[p]&&!hasFinishedQ[id, 0], True, False];
processRunningQ[s_InputStream] := processRunningQ[findProcessFromStream[s]];

If[!StringQ[$libraryPath],
	$libraryPath = FindLibrary["libProcessLink"];
	If[$libraryPath === $Failed,
		Message[LibraryFunction::load, "libProcessLink"];
	,
		$RunningProcesses = {};
		If[LibraryLoad[$libraryPath] === $Failed, 
			Message[LibraryFunction::load, $libraryPath]];
	];
];
maxRunArgs = 1000;
runs[numArgs_Integer] := Block[{loaded},
	loaded = Quiet[
		LibraryFunctionLoad[$libraryPath, "run", Join[{Integer}, Table["UTF8String", numArgs+3] ], Integer],
		LibraryFunction::overload
	];
	runs[numArgs] = loaded;
	loaded
]
killProcess = LibraryFunctionLoad[$libraryPath, "killProcess", {Integer, Integer}, "Boolean"];
killExtProcess = LibraryFunctionLoad[$libraryPath, "killExternalProcess", {Integer, Integer}, "Boolean"];
freeProcess = LibraryFunctionLoad[$libraryPath, "freeProcess", {Integer}, "Void"];
hasFinishedQ = LibraryFunctionLoad[$libraryPath, "hasFinishedQ", {Integer, Integer}, "Boolean"];
getExitValue = LibraryFunctionLoad[$libraryPath, "getExitValue", {Integer}, Integer];
hasExitValue = LibraryFunctionLoad[$libraryPath, "hasExitValue", {Integer}, "Boolean"];
waitFor = LibraryFunctionLoad[$libraryPath, "waitFor", {Integer}, "Void"];
getProcs = LibraryFunctionLoad[$libraryPath, "getAllPIDs", {True|False}, {Integer,1}];
getProcessInfoForPID = LibraryFunctionLoad[$libraryPath, "getProcessInformationForPID", {Integer}, {"RawArray"}];
getProcessInfoForUID = LibraryFunctionLoad[$libraryPath, "getProcessInformationForUID", {Integer}, {"RawArray"}];
hasProcess = LibraryFunctionLoad[$libraryPath, "ProcessExists", {Integer}, {"Boolean"}];
pidforuid = LibraryFunctionLoad[$libraryPath, "PIDForUID", {Integer},Integer];
sendsignal = LibraryFunctionLoad[$libraryPath,"SendSignal",{Integer,Integer},Integer];

environmentToList[environment_Association] :=
	Select[StringJoin[ToString[StringTrim[#]] & /@ {#[[1]], "=", #[[2]]}] & /@ Normal[environment],
		StringCount[#, "="] == 1 &]

StartProcess::argtm = "Too many arguments.";

absoluteFileNameQ[file_] := With[{abs = Quiet[ExpandFileName[file]]},
	StringQ[abs] && StringMatchQ[abs, file, IgnoreCase -> True]
];

caseInsensitiveValue[a_Association, key_String] := With[
	{values = Select[Normal[a], StringMatchQ[key, #[[1]], IgnoreCase -> True] &]},
	
	If[MatchQ[values, {__}],
		values[[1, 2]],
		Missing["KeyAbsent", key]
	]	
];

getEnvValue[environment_Association, key_String] := If[WindowsQ,
	caseInsensitiveValue[environment, key],
	environment[key]
];

commandToAbsolutePath[command_String, environment_Association, processDirectory_String] := Module[
	{paths, systemPath, fixedSlash},

	If[absoluteFileNameQ[command] && (FileExistsQ[command] || FileExistsQ[command<>".exe"]
			|| FileExistsQ[command<>".com"] && Not[DirectoryQ[command]]), Return[command]];
	If[WindowsQ,
		fixedSlash = FileNameJoin[command];
		If[absoluteFileNameQ[fixedSlash] && (FileExistsQ[fixedSlash] || FileExistsQ[fixedSlash<>".exe"]
				|| FileExistsQ[fixedSlash<>".com"] && Not[DirectoryQ[fixedSlash]]), Return[fixedSlash]]
	];

	systemPath = getEnvValue[environment, "PATH"];
	systemPath = If[StringQ[systemPath],
		StringSplit[systemPath, If[WindowsQ, ";", ":"]],
		None
	];
	
	paths = Select[Flatten[{processDirectory, Directory[], systemPath}],
		StringQ[#] && absoluteFileNameQ[#] && DirectoryQ[#] &];
	
	paths = FileNameJoin[{#, command}] & /@ paths;
	
	If[WindowsQ && !StringMatchQ[command, ___ ~~ (".exe" | ".com"), IgnoreCase -> True],
		paths = DeleteDuplicates[Flatten[{paths, (# <> ".exe") & /@ paths, (# <> ".com") & /@ paths}]]
	];
	
	paths = Select[paths, FileExistsQ, 1];
	
	If[MatchQ[paths, {__}], paths[[1]], $Failed]
];

commandToAbsolutePath[command_, environment_, processDirectory_] := $Failed;

(*Used with AbsoluteTime to avoid repeats*)
increment = 0;
(*Do not change "wl-stream-std" without also modifying in CSource/win.c*)
$TempPrefix := FileNameJoin[{$UserBaseDirectory, "ApplicationData", "ProcessLink", "Streams", "wl-stream-std"}];
(*Generate temporary file names for stdout and stderr*)
makeFileStreamNames[] := Block[{shortUID, directory, prefix},
	shortUID = IntegerString[Hash@{$SessionID,AbsoluteTime[],increment++}, 36, 13];
	{StringJoin[$TempPrefix, "out-", shortUID],
	StringJoin[$TempPrefix, "err-", shortUID]}
]

removeNonRules[assocList_List] := Select[assocList, (Head[#1] === Rule)&]

internalCMDs = {"ASSOC","BREAK","CALL","CD","CHDIR","CLS","COLOR","COPY","DATE","DEL","DIR","DPATH","ECHO","ENDLOCAL","ERASE","EXIT","FOR","FTYPE","GOTO","IF","KEYS","MD","MKDIR","MKLINK","MOVE","PATH","PAUSE","POPD","PROMPT","PUSHD","REM","REN","RENAME","RD","RMDIR","SET","SETLOCAL","SHIFT","START","TIME","TITLE","TYPE","VER","VERIFY","VOL"};

isInternalCMD[cmd_] := WindowsQ && StringQ[cmd] && MemberQ[internalCMDs, ToUpperCase[cmd]];

startProcess[method_String, {cmd_, args___}, environment : (_List | _Association | Inherited), processDirectory : (_String | Inherited)] :=
Module[{cmd2, commands, runid, fullArgs, retid, currDir, environment2, envList, process, fileOut, fileErr,data},
	environment2 = Switch[
		environment,
		_List,
		Association[removeNonRules[environment]],
		_Association,
		environment,
		Inherited,
		Association[removeNonRules[GetEnvironment[]]]
	];
	envList = environmentToList[environment2];
	currDir = If[processDirectory === Inherited, Directory[], processDirectory];

	cmd2 = commandToAbsolutePath[cmd, environment2, currDir];
	If[cmd2 === $Failed,
		If[isInternalCMD[cmd],
			commands = ToString /@ {"C:\\Windows\\System32\\cmd.exe", "/c", cmd, args};
			,
			If[method == "StartProcess", Message[StartProcess::pnfd, cmd], Message[RunProcess::pnfd, cmd]];
			Return[$Failed];
		];
		,
		commands = ToString/@{cmd2, args};
	];
	commands = ToString/@{cmd2, args};
	(* we remove the quotes on windows, the arguments have to be parsed by the OS, so quotes on commands aren't harmless. To be improved *)
	commands = If[WindowsQ, StringReplace[#, "\"" -> ""]& /@ commands, commands];
	(* we find the function in "runs" which has the right number of arguments, and we call it *)
	runid = Length[Join[commands, envList]];
	{fileOut, fileErr} = makeFileStreamNames[];
	CreateFile[fileOut];
	CreateFile[fileErr];
	If[runid > maxRunArgs + 1,
		Message[StartProcess::argtm];
		Return[$Failed]
	];
	fullArgs = Flatten[{Length[envList], currDir, commands, envList, fileOut, fileErr}];
	retid = runs[runid] @@ fullArgs;
	If[!NumberQ[retid],
		If[method == "StartProcess", Message[StartProcess::pnfd, cmd], Message[RunProcess::pnfd, cmd]];
		Return[$Failed]
	];
	stdoutFileNames[retid] = fileOut;
	stderrFileNames[retid] = fileErr;
	data = getProcAssocForUID[retid];
	process = ProcessObject[<|"ManagedProcess" -> True, "UID" -> retid,"PID"->data["PID"], "PPID"->data["PPID"],"Program" -> data["Program"], "Path"->data["Path"],"User"->data["User"],"StartTime"->data["StartTime"]|>];
	addRunningProcess[process];
	process
]

handleFileWrapper[arg : (_List | _String | _File)] :=
	Module[{commandOrg, command},
		If[
			Head[arg] === List,
			If[
				Length[arg] > 0,
				commandOrg = arg[[1]],
				Return[arg];
			],
			commandOrg = arg;
		];

		If[
			Head[commandOrg] === File,
			If[Length[commandOrg] != 1,
				Message[RunProcess::pnfd, ""];
				Return[$Failed];
			];
			command = commandOrg[[1]];,
			command = commandOrg;
		];
		command = StringTrim[command, "file://", IgnoreCase -> True];

		If[
			Head[arg] === List,
			Return[
				Prepend[
					arg[[2;;]],
					command
				]
			],
			Return[
				command
			];
		];
	];

startProcess[method_, _, environment : (_List | _Association | Inherited), processDirectory : (_String | Inherited)] :=
	Message[StartProcess::nffil, method];

Options[StartProcess] = {
	ProcessEnvironment -> Inherited,
	ProcessDirectory -> Inherited
};

StartProcess[arg : (_List | _String | _File), args___, opts:OptionsPattern[]] :=
	Block[{$inStartProcessF = True, procCommand},
		procCommand = handleFileWrapper[arg];
		If[
			FailureQ[procCommand], 
			Return[$Failed]
		];
		StartProcess[procCommand, args, opts]
	] /; !TrueQ[$inStartProcessF];

StartProcess[command_String, opts:OptionsPattern[]] :=
	startProcess["StartProcess", {command}, OptionValue[ProcessEnvironment], OptionValue[ProcessDirectory]]

StartProcess[{command_String, args: (_ | (_ -> _))...}, opts:OptionsPattern[]] := With[
	{cmdlist = Prepend[Flatten[List @@@ List[args]], command]},
	startProcess["StartProcess", cmdlist, OptionValue[ProcessEnvironment], OptionValue[ProcessDirectory]]
]

StartProcess[args___] /; (If[Length[{args}] < 1 || Length[{args}] > 2, Message[StartProcess::argt, StartProcess, Length[{args}], 1, 2]]; False) := None 

StartProcess::stopen = "Could not open the `1` stream"; 

iGetExitValue[pr : ProcessObject[<|"ManagedProcess" -> True, "UID" -> id_,___|>]] := If[hasExitValue[id], getExitValue[id], None, None]

iWaitFor[pr : ProcessObject[<|"ManagedProcess" -> True, "UID" -> id_,___|>]] := (waitFor[id]; iGetExitValue[pr])

NoInputProvided;(* used as default value for RunProcess' input expr *)

runProcessOutputs = {"ExitCode", "StandardOutput", "StandardError"};

runProcess[commands_List, environment : (_List | _Association | Inherited), processDirectory : (_String | Inherited), inputexpr_, return_: All] := Module[
	{process, out, err, all, done = False},

	process = startProcess["RunProcess", commands, environment, processDirectory];
	If[!MatchQ[process, _ProcessObject], Return[$Failed]];
	(*WithLocalSettings is not documented: args are preprocessing, code, postprocessing, will run both with
	abort and timeconstrained (checkAbort doesn't work with time constrained)*)
	Internal`WithLocalSettings[None,
		If[!MatchQ[process, ProcessObject[___]], Return[$Failed]];
		If[inputexpr =!= NoInputProvided, BinaryWrite[process, ToString[inputexpr]];Close[ProcessConnection[process, "StandardInput"]]];
		{out, err} = If[StringQ[#],#,""]&/@(ReadString[#]&/@{process, ProcessConnection[process, "StandardError"]});
		all = With[{t = Thread[runProcessOutputs -> {iGetExitValue[process], out, err}]}, Association[t] ];
		done = True;
		If[return === All, all, all[return]]
	, (KillProcess[process])]
]

Options[RunProcess] = {
	ProcessEnvironment -> Inherited,
	ProcessDirectory -> Inherited
};

commandExistQ[command_String, envir_, dir_] := Module[{ e1 = envir, d1 = dir},
	e1 = Switch[
		e1,
		_List,
		Association[removeNonRules[e1]],
		_Association,
		e1,
		Inherited,
		Association[removeNonRules[GetEnvironment[]]]
	];
	If[d1 === Inherited, d1 = Directory[]];
	StringQ[commandToAbsolutePath[command, e1, d1]] || isInternalCMD[command]
	];
	
containsSpaceQ[s_String] := StringContainsQ[s, " "];

RunProcess[arg : (_List | _String | _File), args___, opts:OptionsPattern[]] :=
	Block[{$inRunProcessF = True, procCommand},
		procCommand = handleFileWrapper[arg];
		If[
			FailureQ[procCommand], 
			Return[$Failed]
		];
		RunProcess[procCommand, args, opts]
	] /; !TrueQ[$inRunProcessF];

internalCMDs = {"ASSOC","BREAK","CALL","CD","CHDIR","CLS","COLOR","COPY","DATE","DEL","DIR","DPATH","ECHO","ENDLOCAL","ERASE","EXIT","FOR","FTYPE","GOTO","IF","KEYS","MD","MKDIR","MKLINK","MOVE","PATH","PAUSE","POPD","PROMPT","PUSHD","REM","REN","RENAME","RD","RMDIR","SET","SETLOCAL","SHIFT","START","TIME","TITLE","TYPE","VER","VERIFY","VOL"};

RunProcess[command_String, opts:OptionsPattern[]] := 
	If[commandExistQ[command, OptionValue[ProcessEnvironment], OptionValue[ProcessDirectory]],
		RunProcess[{command}, opts],
		If[containsSpaceQ[command], Message[RunProcess::pnfds, command], Message[RunProcess::pnfd, command]];$Failed]
	
RunProcess[commands_List, opts:OptionsPattern[]] := 
	runProcess[commands, OptionValue[ProcessEnvironment], OptionValue[ProcessDirectory], NoInputProvided, All]

RunProcess[command_String, ret: (All | Alternatives @@ runProcessOutputs), inputexpr_ : NoInputProvided, opts:OptionsPattern[]] /; !MatchQ[inputexpr, _Rule | _RuleDelayed] :=
	If[commandExistQ[command, OptionValue[ProcessEnvironment], OptionValue[ProcessDirectory]],
		RunProcess @@ {{command}, ret, inputexpr, opts}, (* using @@ to get rid of a pointless Workbench warning *)
		If[containsSpaceQ[command], Message[RunProcess::pnfds, command], Message[RunProcess::pnfd, command]];$Failed]
		 
RunProcess[commands_List, ret: (All | Alternatives @@ runProcessOutputs), inputexpr_ : NoInputProvided, opts:OptionsPattern[]] /; !MatchQ[inputexpr, _Rule | _RuleDelayed] :=
	runProcess[commands, OptionValue[ProcessEnvironment], OptionValue[ProcessDirectory], inputexpr, ret]

RunProcess[commands : (_List | _String), ret_String, inputexpr_ : NoInputProvided, opts:OptionsPattern[]] /; !MatchQ[inputexpr, _Rule | _RuleDelayed] :=
	With[{values = StringJoin@@Riffle[Append[ToString[#, InputForm] & /@ runProcessOutputs, "and All"], ", "]},
		Message[General::optvp, 2, "RunProcess", values];
	]

RunProcess[args___] /; (If[Length[{args}] < 1 || Length[{args}] > 3, Message[RunProcess::argb, RunProcess, Length[{args}], 1, 3]]; False) :=
	None 

KillProcess[po:ProcessObject[<|"ManagedProcess" -> t_,___|>], signal_ : -1] := (
	(* we quiet the error about streams that have already been closed *)
	If[t == True,
		(
			Quiet[Close[ProcessConnection[po, #]]& /@ {"StandardInput", "StandardOutput", "StandardError"}];
			killProcess[po["UID"], signal];
			FreeProcess[po];
		),
		Module[{pid = po["PID"],match},
		(
			If[MemberQ[pidforuid/@ProcessLink`Private`$RunningProcesses,pid],
			(
				match = Cases[ProcessLink`Private`$RunningProcesses, x_Integer/;pidforuid[x]==pid][[1]];
				killProcess[match, signal];
				FreeProcess[match];
			)];
			killExtProcess[pid,signal];
		)]]
)

KillProcess[args___ /; ArgumentCountQ[KillProcess, Length[{args}], 1, 2]] := $Failed /; False;

PackageScope["FreeProcess"];
FreeProcess[pobj : ProcessObject[<|"ManagedProcess" -> True, "UID" -> p_,___|>]] := (
	removeRunningProcesses[{pobj}];
	freeProcess[p];
)

ProcessInformation[pr : ProcessObject[<|"ManagedProcess" -> True, "UID" -> p_,___|>]] := With[{t = {"ExitCode" -> ProcessInformation[pr, "ExitCode"]}}, Association[t] ]

ProcessInformation[pr : ProcessObject[<|"ManagedProcess" -> True, "UID" -> p_,___|>], "ExitCode"] := iGetExitValue[pr]

ProcessInformation[args___] /; (If[Length[{args}] < 1 || Length[{args}] > 2, Message[ProcessInformation::argt, ProcessInformation, Length[{args}], 1, 2]]; False) := None

ProcessStatus[pr : ProcessObject[<|"ManagedProcess"->True,"UID"->id_, ___|>]] := If[isNotValidProcess[pr]|| hasFinishedQ[id, 100], "Finished", "Running", "Finished"]

ProcessStatus[pr : ProcessObject[<|"ManagedProcess"->False,"PID"->pid_,___|>]] := If[ProcessLink`Private`hasProcess[pid], "Finished", "Running"]

ProcessStatus[pr : ProcessObject[___], r_] := (ProcessStatus[pr] === r)

ProcessStatus[args___] /; (If[Length[{args}] < 1 || Length[{args}] > 2, Message[ProcessStatus::argt, ProcessStatus, Length[{args}], 1, 2]]; False) := None

Processes[] := (
	removeRunningProcesses[Select[$RunningProcesses, hasFinishedQ[#, 100] &]];
	ProcessObject[<|"ManagedProcess"->True,"UID"->#|>]&/@$RunningProcesses
)

Processes[args__] /; (Message[Processes::argrx, Processes, Length[{args}], 0]; False) := None

SignalProcesses::nosig = "Specified signal does not exist on this platform."

SignalProcesses::os = "Signals not supported on this platform."

SignalProcess[___]/;WindowsQ := Message[SignalProcess::os];$Failed

SignalProcess[pr : ProcessObject[<|"ManagedProcess"->True,"UID"->id_, ___|>], signal: (String|Integer)]:= signalProcess[ProcessLink`Private`pidforuid[id],signal]

SignalProcess[pr : ProcessObject[<|"ManagedProcess"->False,"PID"->id_, ___|>], signal: (String|Integer)]:= signalProcess[id,signal]

signalProcess[pid_Integer,signal_String]:= Module[{matches},(
	matches = KeySelect[ProcessLink`Private`$Signals, StringMatchQ[signal | signal, IgnoreCase -> True]];
	If[Length@matches==0,Message[SignalProcesses::nosig,"Specified signal does not exist on this platform."];Return[None],signalProcess[pid,matches[[1]][[2]]]]
)]
signalProcess[pid_Integer,signal_Integer] := (
	If[!MemberQ[ProcessLink`Private`$Signals,signal],Message[SignalProcess::nosig,"Specified signal does not exist on this platform."];Return[None]];
	If[MatchQ[ProcessLink`Private`sendsignal[process,signal],_LibraryFunctionError|-1],Message[SignalProcess::err,"Unable to send specied signal to given process."]];
)

StartProcess::usage = "StartProcess[\"executable\"] executes an external program, yielding a ProcessObject to represent the resulting subprocess.\nStartProcess[{\"executable\", arg 1, arg 2, ...}] executes an external program, passing it the specified arguments arg i.";
RunProcess::usage = "RunProcess[\"command\"] runs the specified external command, returning information on the outcome.\nRunProcess[{\"command\", arg 1, arg 2, \[Ellipsis]}] runs the specified command, with command-line arguments arg i.\nRunProcess[command, \"prop\"] returns only the specified property.\nRunProcess[command, prop, input] feeds the specified initial input to the command."  
KillProcess::usage = "KillProcess[proc] kills the external process represented by the ProcessObject proc.";
KillProcess::perm = "The Wolfram Language was unable to kill the specified process due to insufficient permissions."
ProcessInformation::usage = "ProcessInformation[proc] gives information about an external process proc.\nProcessInformation[proc, \"prop\"] gives information about the property \"prop\".";
ProcessStatus::usage = "ProcessStatus[proc] gives the current status of the external process represented by the ProcessObject proc.\nProcessStatus[proc, \"status\"] returns True if the process has the status given, and returns False otherwise.";
ProcessObject::usage = "ProcessObject[...] is an object that represents a runnable external process.";
Processes::usage = "Processes[] returns a list of currently running external processes, started in this Wolfram Language session.";
$SystemShell::usage = "$SystemShell is a symbol that specifies the system shell for the OS that is currently being used.";
SystemProcesses::usage = "SystemProcesses[] returns a list of ProcessObjects that represent system programs.";
SystemProcessData::usage = "SystemProcessData[] returns a Dataset containing information on the processes running on the host computer.";
Protect[Read];

SetAttributes[
	{KillProcess, ProcessInformation, ProcessObject, Processes, ProcessStatus, RunProcess, StartProcess,SystemProcesses,SystemProcessData},
	{Protected}];


End[];

EndPackage[];

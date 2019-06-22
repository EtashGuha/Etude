(* Mathematica Package *)

(* Created by the Wolfram Workbench Apr 9, 2013 *)

BeginPackage["SerialLink`"]
(* Exported symbols added here with SymbolName::usage *) 
(* Exported symbols added here with SymbolName::usage *) 
SerialPort::usage  = "SerialPort[\"name\", {connectionParameters}] is an object that represents a serial port that has been opened by Mathematica. "
SerialPortOpen::usage = "SerialPortOpen[\"name\"] opens a serial port in Mathematica."
SerialPorts::usage = "SerialPorts[] lists the serial ports that are currently opened in Mathematica."
SerialPortRead::usage = "SerialPortRead[serialPort, format] reads from the serial port in the format specified (\"Byte\" or \"String\")."
SerialPortWrite::usage = "SerialPortWrite[serialPort, data] writes data synchronously to an opened serial port."
SerialPortClose::usage = "SerialPortClose[ serialPort] closes the serial port specified."

CreateSerialPortWriteAsynchronousTask::usage = "CreateSerialPortWriteAsynchronousTask creates an asynchronous task for writing to the serial port."
CreateSerialPortReadAsynchronousTask::usage = "CreateSerialPortReadAsynchronousTask creates an asynchronous task for reading from the serial port."
RunSerialPortWriteAsynchronousTask::usage ="RunSerialPortWriteAsynchronousTask creates and starts an asynchronous task for writing to the serial port."
RunSerialPortReadAsynchronousTask::usage = "RunSerialPortReadAsynchronousTask creates and starts an asynchronous task for reading from the serial port."
CurrentSerialPortData::usage = "CurrentSerialPortData[asyncObj] returns the latest data read asynchronously from the SerialPort associated with the AsynchronousTaskObject specified."
SerialPortWriteAsynchronous::usage = "SerialPortWriteAsynchronous queues data for writing to the SerialPort specified.  A writeID is returned."
SerialPortFlush::usage = "SerialPortFlush flushes data queued for writing and triggers a \"flush\" event."
SerialPortReadyQ::usage = "SerialPortReadyQ[ serialPort] returns True if data is available for reading, and False otherwise."

$SerialPortWriteAsyncTasks={};
Begin["`Private`"]
(* Implementation of the package *)
$PortID = 0;
$PortIDs = {};


defineDevMessages[]:=(
	(*SerialPort::usage  = "SerialPort Port Object";
	SerialPortOpen::usage = "SerialPortOpen  ";
	SerialPorts::usage = "SerialPorts ";
	SerialPortRead::usage = "SerialPort Read ";
	SerialPortWrite::usage = "SerialPort Write ";
	SerialPortClose::usage = "SerialPort Port Close";
	CreateSerialPortWriteAsynchronousTask::usage = "Create an asynchronous task for writing to the serial port.";
	CreateSerialPortReadAsynchronousTask::usage = "Create an asynchronous serial read task but do not start it.";
	RunSerialPortWriteAsynchronousTask::usage ="Create and start an asynchronous task for writing to the serial port.";
	RunSerialPortReadAsynchronousTask::usage = "Create and start an asynchronous serial read task.";
	CurrentSerialPortData::usage = "SerialPort current data.";
	SerialPortWriteAsynchronous::usage = "Queue data for writing to the port as soon as possible.  A writeID iterated integer counter is returned.";
	SerialPortFlush::usage = "Flush data queued for writing.  Triggers a \"flush\" event via the event function.";
	SerialPortReadyQ::usage = "Check to see if data is available on the FIFO byte buffer.  Returns True|False.";*)
	General::baudrate="Value of option BaudRate -> `1` is not 110, 300, 600, 1200, 2400, 4800, 9600, 14400, 19200, 28800, 38400, 56000, 57600, 115200, 128000, 256000.";
	General::parity="Value of option Parity -> `1` is not None, Odd, or Even.";
	General::stopbits="Value of option StopBits -> `1` is not None, 0, 1, 1.5 or 2.";
	General::databits="Value of option DataBits -> `1` is not 5, 6, 7 or 8.";
	General::handshake="Value of option Handshake -> `1` is not None, RTS or XOnXOff.";
	General::buffersize="Value of option BufferSize -> `1` is not positive integer.";
	General::readterminator = "Value of option ReadTerminator ->`1` is not an interger (0-255), or a character with character code between 0 and 255.";
	General::writeterminator = "Value of option WriteTerminator ->`1` is not a valid string or a list of intgers, ranging from 0-255.";
	General::chtype = "First argument `1` is not a valid port.";
	SerialPortOpen::nopen= "Could not open the port `1`.`2`"; 
	(* SerialPortOpen::nopen= "Could not open the port \"`1`\"."; *)
	General::serialport = "SerialPort port object specified is not valid.  It may have been closed."; (* TODO add `1` for port string *)
	(* General::serialport = "SerialPort port object `1` specified is not valid.  It may have been closed."; *)
	General::dataarg = "Data provided is invalid.  A string or a list of integers between 0 and 255 is expected.";
	General::numbytes="Specified number of bytes to read is invalid."; (* TODO add `1` *)
	(*General::numbytes="Specified number of bytes `1` to read is invalid.";*)
	(*General::bytesperbuffer="bad bytes per buffer";*)
	General::formatarg="Specified read format is invalid.  \"String\" or \"Bytes\" is expected.";
	General::repeatspec="Invalid number of repetitions has been specified.";
	General::timeoutarg = "The timeout period specified is invalid."; (* TODO add `1` *)
	(*General::timeoutarg = "The timeout period `1` specified is invalid."; *)
	)
$PortID = 0;
$PortIDs = {};
$PortData={};
$SerialPortWriteAsyncTasks={};

(* Implementation of the package *)


$asyncObj = Null;
$asynchronousWriteIterator = 0;
$adapterInitialized;
$listOfOpenedSerialPorts = {};

$packageFile = $InputFileName
$libName = Switch[$SystemID,
	"Windows"|"Windows-x86-64",
		"SerialLink.dll",
	"MacOSX"|"MacOSX-x86-64",
		"SerialLink.dylib",
	"Linux-x86-64",
		"SerialLink.so",
	"Linux",
		"SerialLink.so",
	"Linux-ARM",
		"SerialLink.so"
]

$adapterLib = FileNameJoin[{FileNameTake[$packageFile, {1,-2}], "LibraryResources", $SystemID, $libName}];
(*$adapterLib ="/Users/fahimc/Documents/workspace/WolframSerialPortIO/WolframSerialPortIO/SystemFiles/MacOSX-x86-64/MSerialAdapter.dylib";*)

loadAdapter[]:=  
(	
	defineDevMessages[];
    If[!ValueQ@$adapterInitialized,
    	iLoadAdapter      = LibraryFunctionLoad[$adapterLib, "LoadAdapter", {}, Integer];
    	iConnectDevice    = LibraryFunctionLoad[$adapterLib, "connectDevice", LinkObject, LinkObject];
    	iDisconnectDevice = LibraryFunctionLoad[$adapterLib, "disconnectDevice", LinkObject, LinkObject];
    	close             = LibraryFunctionLoad[$adapterLib, "serialClose", LinkObject, LinkObject];
    	open              = LibraryFunctionLoad[$adapterLib, "serialOpen", LinkObject, LinkObject];	
    	write             = LibraryFunctionLoad[$adapterLib, "serialWrite", LinkObject, LinkObject];
    	createAsyncWrite  = LibraryFunctionLoad[$adapterLib, "createSerialAsynchronousWrite", LinkObject, LinkObject];
    	asyncWrite        = LibraryFunctionLoad[$adapterLib, "serialWriteAsynchronous", LinkObject, LinkObject];
    	read              = LibraryFunctionLoad[$adapterLib, "serialRead", LinkObject,LinkObject];
    	createAsyncRead   = LibraryFunctionLoad[$adapterLib, "createSerialAsynchronousRead", LinkObject,LinkObject];
    	readFlush         = LibraryFunctionLoad[$adapterLib, "serialReadFlush", LinkObject,LinkObject];
    	writeFlush        = LibraryFunctionLoad[$adapterLib, "serialWriteFlush", LinkObject,LinkObject];
        break             = LibraryFunctionLoad[$adapterLib, "serialBreak",LinkObject,LinkObject];
        errorHandler   	  = LibraryFunctionLoad[$adapterLib, "createErrorHandler",LinkObject,LinkObject];
        readyQ            = LibraryFunctionLoad[$adapterLib, "serialReadyQ",LinkObject,LinkObject];
        iLoadAdapter[];
        $adapterInitialized       = True;
    ]
)

validateBaudRate[baudrate_]:=Module[
	{res},
	defineDevMessages[];
	res = baudrate;
	Return@res;
	]
	
validateStopBit[stopbit_]:=Module[
	{res},
	defineDevMessages[];
	res =stopbit/. {None->0,0->1,1->2,1.5->3,2->4};
	Return@res;
	]

validateParity[parity_]:=Module[
	{res},
	defineDevMessages[];
	If[$OperatingSystem === "Windows",res = parity/.{None->0,"Odd"->1,"Even"->2,"Mark"->3,"Space"->4},res = parity/.{None->0,"Odd"->1,"Even"->2}];
	(*res = parity/.{None->0,"Odd"->1,"Even"->2};*)
	Return@res;
	]

validateWriteTerminator[terminator_]:=Module[
	{res},
	defineDevMessages[];
	res = terminator/.{None->""};
	Return@res;
]

validateHandshake[handshake_]:=Module[
	{res},
	defineDevMessages[];
	res = handshake/.{None->0,"RTS"->1,"XOnXOff"->2};
	Return@res;
	]

validateDataBits[databits_]:=Module[
	{res},
	res = databits;
	Return@res;
	]

validateIgnoreBreak[breakStatus_]:=Module[
	{res},
	res = breakStatus/.{True->1,False->0};
	Return@res;
]

validateBreakDelimiter[breakDelimiter_]:=Module[
	{res,MULTIBYTE  = 21,SINGLEBYTE = 22},
	(*reason behind mapping MULTIBYTE and SINGLE byte values to 21 and 22 is because of thevalues is myenums.h*)
	res = breakDelimiter/.{"MultiByte"->MULTIBYTE,"SingleByte"->SINGLEBYTE};
	Return@res;
]
validateBufferSize[buffersize_]=Module[
	{res},
	res = buffersize;
	Return@res;
	]
validBytesListPattern =  List[Repeated[_Integer?((# >= 0 && # <= 255) &), Infinity]];
validReadTerminator = _Integer?(# >= 0 && # <= 255 &) | _Symbol?(# ===Automatic &) | _String?(MatchQ[ToCharacterCode[#][[1]],validReadTerminator] && StringLength[#] == 1 &)
validWriteTerminator  =_Symbol?(# === None &) | _String?(# === "" &) | _List?(# === {} &) | validBytesListPattern | _String?(MatchQ[ToCharacterCode[#][[1]], validReadTerminator] &)
validNumBytes = _Integer?(# > 0 && # <= 102400 &) | _Symbol?(# === Automatic &)
$ValidBaudRates ={110,300,600,1200,2400,4800,9600,14400,19200,28800,38400,56000,57600,115200,128000,256000};
Options[validateOptions]:=Join[
	Options[SerialPortOpen],
	Options[SerialPortRead],
	Options[CreateSerialPortReadAsynchronousTask],
	Options[CreateSerialPortWriteAsynchronousTask],
	Options[SerialPortWrite]];(*{"IgnoreBreak"->True,"BreakDelimiter"->"MultiByte","WriteTerminator"->None,"Timeout"->0,"BaudRate"->115200,"Parity"->None,"Handshake"->None,"StopBits"->None,"DataBits"->8,
	"ReadBufferSize"->4096,"ErrorHandler"->None,"ReadTerminator"->Automatic,"WriteHandler"->None,"ReadHandler"->None,"ReadCompleteHandler"->None,"BreakHandler"->None,"FlushHandler"->None}*)


validateOptions[f_,opts:OptionsPattern[]]:=Module[
	{validIgnoreBreakQ,validBreakDelimiterQ,validWriteTerminatorQ,validBufferSizeQ,validDataBitsQ,validHandshakeQ,validParityQ,validStopBitQ,validBaudRateQ,validTimeoutQ,validReadTerminatorQ},
	
	defineDevMessages[];
	validDataBitsQ = MemberQ[{5,6,7,8},OptionValue@"DataBits"];
	If[!validDataBitsQ===True,Message[f::databits,OptionValue@"DataBits"];validDataBitsQ=False];
	
	validWriteTerminatorQ = MatchQ[OptionValue@"WriteTerminator",validWriteTerminator];
	If[!validWriteTerminatorQ===True,Message[f::writeterminator,OptionValue@"WriteTerminator"];validWriteTerminatorQ=False];

	
	validHandshakeQ = MemberQ[{None,"RTS","XOnXOff"},OptionValue@"Handshake"];
	If[!validHandshakeQ===True,Message[f::handshake,OptionValue@"Handshake"];validHandshakeQ=False];
	
	validBreakDelimiterQ = MemberQ[{"MultiByte","SingleByte"},OptionValue@"BreakDelimiter"];
	If[!validBreakDelimiterQ ===True,Message[f::breakDelimiter,OptionValue@"BreakDelimiter"];validBreakDelimiterQ=False];
	
	validBufferSizeQ = IntegerQ[OptionValue@"ReadBufferSize"];
	If[!validBufferSizeQ===True,Message[f::buffersize,OptionValue@"ReadBufferSize"];validBufferSizeQ=False;];
	
	validParityQ = ($OperatingSystem==="Windows" && MemberQ[{None,"Odd","Even","Mark","Space"},OptionValue@"Parity"])||(MemberQ[{None,"Odd","Even"},OptionValue@"Parity"]);
	If[!validParityQ===True,Message[f::parity,OptionValue@"Parity"];validParityQ=False;];
	
	validIgnoreBreakQ = MemberQ[{True,False},OptionValue@"IgnoreBreak"];
	If[!validIgnoreBreakQ===True,Message[f::ignoreBreak,OptionValue@"IgnoreBreak"];validIgnoreBreakQ=False;];
	
	validStopBitQ = MemberQ[{None,0,1,1.5,2},OptionValue@"StopBits"];
	If[!validStopBitQ===True,Message[f::stopbits,OptionValue@"StopBits"];validStopBitQ=False;];
	
	validBaudRateQ = MemberQ[$ValidBaudRates,OptionValue@"BaudRate"];
	If[!validBaudRateQ===True,Message[f::baudrate,OptionValue@"BaudRate"];validBaudRateQ=False;];
	
	validReadTerminatorQ = MatchQ[OptionValue@"ReadTerminator",validReadTerminator];
	If[!validReadTerminatorQ===True,Message[f::readterminator,OptionValue@"ReadTerminator"];validReadTerminatorQ=False;];
	
	validTimeoutQ = (NumberQ@OptionValue@"Timeout"&&OptionValue@"Timeout">=0)===True;
	If[!validTimeoutQ===True,Message[f::timeoutarg];validTimeoutQ=False;];
	
	Return@validIgnoreBreakQ&&validBufferSizeQ&&validDataBitsQ&&validHandshakeQ&&validParityQ&&validStopBitQ&&validBaudRateQ&&validTimeoutQ&&validReadTerminatorQ&&validWriteTerminatorQ&&validBreakDelimiterQ;
	]


validateRepeatSpec[repeatSpec_Integer]:=(If[repeatSpec>0,Return@repeatSpec,Return@$Failed])
validateRepeatSpec[repeatSpec_]:=(If[repeatSpec===Infinity,Return@-1,Return@$Failed])

vldArgs[f_,"portname",portName_]:=(defineDevMessages[];If[!StringQ[portName],Message[SerialPortOpen::chtype,portName];False,True])

vldArgs[f_,"object",object_]:=Module[
	{comport,devID},
	defineDevMessages[];

	comport = Quiet[Extract[object,1]];
	devID=comport/.$PortIDs;
	If[!Head[object]===SerialPort,Message[f::chtype,object];Return@False;];
	If[Head[devID]===Integer,Return@True;,Message[f::serialport];Return@False;]
	]
vldArgs[f_,"eventfunction",eventfunc_]:=Module[
	{res},
	defineDevMessages[];
	res = !MemberQ[Names["System`*"],ToString[eventfunc,InputForm]];
	If[res === True,Return@True,Message[f::bdEventFuncName];False]
]
vldArgs[f_,"format",format_]:=(defineDevMessages[];If[format==="Byte"||format==="String",True,Message[f::formatarg];False])

vldArgs[f_,"numBytes",num_]:=(defineDevMessages[];If[(Head[num]===Integer && num >0 && num < 10240)===True || num===Automatic,True,Message[f::numbytes];False])

vldArgs[f_,"repeatSpec",repeatSpec_]:=(defineDevMessages[];If[(repeatSpec >0)===True || repeatSpec===Infinity,True,Message[f::repeatspec];False])

vldArgs[f_,"data",data_List]:=(defineDevMessages[];If[Quiet@MatchQ[data,validBytesListPattern]===True,Return@True,Message[f::dataarg];Return@False])

vldArgs[f_,"data",data_String]:=(defineDevMessages[];If[Quiet@MatchQ[ToCharacterCode[data],validBytesListPattern]===True,Return@True,Message[f::dataarg];Return@False])

vldArgs[f_,"data",Break]:=True;

vldArgs[f_,"data",data_]:=(defineDevMessages[];Message[f::dataarg];Return@False)

vldArgs[f_,"asyncObj",asyncObj_]:=(defineDevMessages[];If[MemberQ[$SerialPortWriteAsyncTasks,asyncObj]===True,True,Message[f::asyncobj];False])

vldArgs[f_SerialPortRead,srlPrtObj_,format_,numBytes_]:=(defineDevMessages[];vldArgs[f,"object",srlPrtObj]&&vldArgs[f,"format",format]&&vldArgs[f,"numBytes",numBytes];Return[vldArgs[f,"object",srlPrtObj]&&vldArgs[f,"format",format]&&vldArgs[f,"numBytes",numBytes]];)
vldArgs[f_CreateSerialPortReadAsynchronousTask,srlPrtObj_,eventfunc_,format_,repeatSpec_, numberBytesPerBuffer_]:=Return[vldArgs[f,"repeatSpec",repeatSpec]&&vldArgs[f,"object",srlPrtObj]&&vldArgs[f,"format",format]&&vldArgs[f,"numBytes",numberBytesPerBuffer]];
validOptionsQ[f_,opts___]:=Module[
	{passedOptionNames,trueOptionNames,invalidOptionNames,invalidOptions,invalidOption,validOptionQ},
	defineDevMessages[];
	passedOptionNames =Cases[{opts},(x_->y_)->x];
	trueOptionNames =Cases[Options@f,(x_->y_)->x];
	invalidOptionNames = Complement[passedOptionNames,trueOptionNames];
	invalidOptions = FilterRules[{opts},invalidOptionNames];
	invalidOption = Quiet[First[invalidOptions]];
	
	validOptionQ=If[invalidOptionNames==={},Return@True,Message[f::optx,invalidOption,f];Return@False];
]

Options[SerialPortOpen]:={"BaudRate"->9600,"Parity"->None,"Handshake"->None,"StopBits"->None,"DataBits"->8,"ReadBufferSize"->4096,"ErrorHandler"->None,"IgnoreBreak"->True,"WriteTerminator"->None,"BreakDelimiter"->"SingleByte"}

SerialPortOpen[portName_/;Quiet@vldArgs[SerialPortOpen,"portname",portName],opts:OptionsPattern[]]/;(If[validOptionsQ[SerialPortOpen,opts]===True,validateOptions[SerialPortOpen,opts],False]):=Module[
	{portNameLocal,breakDelimiter,portOpenQ,baudRate,dataBits,stopBit,parity,handshake,err,bufferSize,serialPortObject,comport,devID,callback2,asyncObj,ignoreBreakStatus}, 
	loadAdapter[];
	defineDevMessages[];
	callback2  			= OptionValue@"ErrorHandler";
	
	baudRate   			= validateBaudRate[OptionValue@"BaudRate"];
	
	dataBits   			= validateDataBits[OptionValue@"DataBits"];
	
	stopBit    			= validateStopBit[OptionValue@"StopBits"];
	
	parity     			= validateParity[OptionValue@"Parity"];
	
	handshake  			= validateHandshake[OptionValue@"Handshake"];
	
	bufferSize 			= validateBufferSize[OptionValue@"ReadBufferSize"];
	
	ignoreBreakStatus 	= validateIgnoreBreak[OptionValue@"IgnoreBreak"];
	
	breakDelimiter 		= validateBreakDelimiter[OptionValue@"BreakDelimiter"];
	portOpenQ = (Length[Cases[$PortIDs,(portName->x_)->portName]]>=1)===True;

	If[!portOpenQ,
		AppendTo[$PortIDs,portName->++$PortID],
		serialPortObject = Cases[$listOfOpenedSerialPorts,SerialPort[__,__], Infinity][[1]];Return@serialPortObject
		];
	
	iConnectDevice[$PortID];
	If[$OperatingSystem==="Windows",portNameLocal=FromCharacterCode[{ 92, 46, 92}]<>portName,portNameLocal=portName];(*fahimc: bug#:278087*)
	err = open[$PortID,portNameLocal, baudRate, dataBits,stopBit,parity,handshake,bufferSize,ignoreBreakStatus,breakDelimiter]; 
	
	If[err===$Failed,$PortIDs=Delete[$PortIDs,-1];iDisconnectDevice[$PortID--];If[$OperatingSystem==="Linux",Message[SerialPortOpen::nopen,portName," Check that the port is available and that you have the necessary access permissions"],Message[SerialPortOpen::nopen,portName,""]];Return@$Failed];(*decrease port id exactly here and not in Close!!!*)(*decrease port id exactly here and not in Close!!!*)
	serialPortObject =  SerialPort[portName,{OptionValue@"BaudRate",OptionValue@"DataBits",OptionValue@"StopBits",OptionValue@"Parity",OptionValue@"Handshake"}];
	
	AppendTo[$listOfOpenedSerialPorts,serialPortObject];
	(********************************************************)
	comport = Extract[serialPortObject,1];
	devID = comport/.$PortIDs;
	asyncObj = Internal`CreateAsynchronousTask[errorHandler,{devID},(errHandlerCallback[serialPortObject, #2, #3];callback2[serialPortObject,#3]) &,"TaskDetail"->{"ErrorHandler",serialPortObject},"Visible"->False];
	
	(********************************************************)
	(*Populate port data-struct*)
	
	MapThread[update,{Table[serialPortObject,{9}],
		{"Name","BaudRate","StopBits","Parity","Handshake","ReadBufferSize","IgnoreBreak","WriteTerminator","ErrorHandler"},Flatten@{portName,Thread[OptionValue[SerialPortOpen,{"BaudRate","StopBits","Parity","Handshake","ReadBufferSize","IgnoreBreak","WriteTerminator"}]],asyncObj}}];
	
	update[serialPortObject,"BaudRate",9600];
	Return@ serialPortObject;
]

(*SerialPortOpen[arg1_String,
	opts__]/;If[Select[{opts},!OptionQ[#]&]==={},False,defineDevMessages[];Message[SerialPortOpen::nonopt,Select[{opts},!OptionQ[#]&],1,SerialPortOpen]]:=Null;
*)
serialPortOpenMessage[x___]:=Module[{m,len,args,MAXSERIALOPENARGS=1},
	m = {1->"portname"};
	args = {x};
	
	len=Length[args];
	If[len===0,Return@Message[SerialPortOpen::argx,SerialPortOpen,0]];
	If[len>MAXSERIALOPENARGS,len=MAXSERIALOPENARGS];

	For[i=1,i<=len,i++,
	
	If[!Quiet[vldArgs[SerialPortOpen,i/.m,args[[i]]]] && !Quiet[OptionQ[args[[i]]]],Return@vldArgs[SerialPortOpen,i/.m,args[[i]]]]
	(*for*)];
	len=Length[args];
	If[len>MAXSERIALOPENARGS,
		
	If[!Quiet[OptionQ[args[[2]]]],Return[
		Message[SerialPortOpen::nonopt,args[[2]],1,HoldForm[SerialPortOpen[x]] (*message*)]
		(*return*)]
		(*if*)];]
	
	];
SerialPortOpen[x___]/;serialPortOpenMessage[x]:=Null	


SerialPorts[]:= Return@$listOfOpenedSerialPorts



(*Synchronous write*)
Options[SerialPortWrite] = {"Timeout"->20}
(*write break*)
SerialPortWrite[srlPrtObj_/;Quiet@vldArgs[SerialPortWrite,"object",srlPrtObj],Break]:=Module[
    {comport,devID},
    defineDevMessages[];
    comport = Extract[srlPrtObj,1];
    devID = comport/.$PortIDs;
    If[IntegerQ@devID,break[devID],Return@$Failed];
    ]
(*write command/data*)
SerialPortWrite[srlPrtObj_/;Quiet@vldArgs[SerialPortWrite,"object",srlPrtObj],
			byteString_/;Quiet@vldArgs[SerialPortWrite,"data",byteString],
			opts:OptionsPattern[]]/;(If[validOptionsQ[SerialPortWrite,opts]===True,validateOptions[SerialPortWrite,opts],False]):=Module[
	{comport,devID,cmd,timeout,result,writeTerminator},
	defineDevMessages[];
	comport =Extract[srlPrtObj,1];
	devID = comport/.$PortIDs;
	timeout = IntegerPart[OptionValue@"Timeout"];
	writeTerminator = validateWriteTerminator["WriteTerminator"/.(srlPrtObj/.$PortData)];
	If[Head[byteString]===List,cmd=FromCharacterCode[byteString];result = write[devID,cmd<>writeTerminator,timeout];Return@result;];
	
	result = write[devID,byteString<>writeTerminator,timeout];
	Return@result;
	]


serialWriteMessage[x___]:=Module[{m,len,args,MAXARGS=2},
	m = {1->"object",2->"data"};
	args = {x};
	
	len=Length[args];
	If[len===0,Return@Message[SerialPortWrite::argrx,SerialPortWrite,0,2]];
	If[len===1,Return@Message[SerialPortWrite::argr,SerialPortWrite,2]];
	If[len>MAXARGS,len=MAXARGS];
	For[i=1,i<=len,i++,
	
	If[!Quiet[vldArgs[SerialPortWrite,i/.m,args[[i]]]] && !Quiet[OptionQ[args[[i]]]],Return@vldArgs[SerialPortWrite,i/.m,args[[i]]]]
	(*for*)];
	len=Length[args];
	If[len>MAXARGS,
		
	If[!Quiet[OptionQ[args[[3]]]],Return[
		Message[SerialPortWrite::nonopt,args[[4]],3,HoldForm[SerialPortWrite[x]] (*message*)]
		(*return*)]
		(*if*)];]
	
	];
SerialPortWrite[x___]/;serialWriteMessage[x]:=Null (*TODO*)


SerialPortReadyQ[srlPrtObj_/;Quiet@vldArgs[SerialPortReadyQ,"object",srlPrtObj]]:=SerialPortReadyQ[srlPrtObj,1]

SerialPortReadyQ[srlPrtObj_/;Quiet@vldArgs[SerialPortReadyQ,"object",srlPrtObj],numBytes_/;Quiet@vldArgs[SerialPortReadyQ,"numBytes",numBytes]]:=Module[
	{comport,devID},
	defineDevMessages[];
	comport = Extract[srlPrtObj,1];
	devID  = comport/.$PortIDs;
	Return@readyQ[devID,numBytes];
]

serialReadyQMessage[x___]:=Module[{m,len,args,MAXARGS=2},
	m = {1->"object",2->"numBytes"};
	args = {x};
	
	len=Length[args];
	If[len===0,Return@Message[SerialPortReadyQ::argx,SerialPortReadyQ,0]];

	If[len>MAXARGS,len=MAXARGS];
	For[i=1,i<=len,i++,
	
	If[!Quiet[vldArgs[SerialPortReadyQ,i/.m,args[[i]]]] && !Quiet[OptionQ[args[[i]]]],Return@vldArgs[SerialPortReadyQ,i/.m,args[[i]]]]
	(*for*)];
	len=Length[args];
	If[len>MAXARGS,Return@Message[SerialPortReadyQ::argt,SerialPortReadyQ,len,1,2];]
	
	];
SerialPortReadyQ[x___]/;serialReadyQMessage[x]:=Null (*TODO*)

Options[CreateSerialPortWriteAsynchronousTask]:={"FlushHandler"->None,"WriteHandler"->None,"UserData"->None}

Options[RunSerialPortWriteAsynchronousTask]:=Options[CreateSerialPortWriteAsynchronousTask]

CreateSerialPortWriteAsynchronousTask[srlPrtObj_/;vldArgs[CreateSerialPortWriteAsynchronousTask,"object",srlPrtObj],
										opts:OptionsPattern[]]/;validateOptions[CreateSerialPortWriteAsynchronousTask,opts]&&Length[Cases[AsynchronousTasks[], 
 AsynchronousTaskObject[{"SerialPortWriteAsynchronousTask", 
   srlPrtObj, ___}, __]]]<1:=Module[
    {writeHandler,flushHandler,taskObj,comport,devID,asyncObj},
    defineDevMessages[];
    comport = Extract[srlPrtObj,1];
    devID = comport/.$PortIDs;
    writeHandler = OptionValue["WriteHandler"];
    flushHandler = OptionValue["FlushHandler"];
    
    taskObj = SerialPort[{"SerialPortIO", portName,"Write"},"DeviceTask"];
    If[!IntegerQ@devID,Return@$Failed];
    asyncObj = Internal`CreateAsynchronousTask[createAsyncWrite,{devID}, 
        (asyncWriteCallback[comport, #1, #2, #3];If[#2==="writedone"&&!writeHandler===None,writeHandler[#1,#3],None];If[#2==="flush"&&!flushHandler===None,flushHandler[#1,#2,#3],None]) &,
        "TaskDetail"->{"SerialPortWriteAsynchronousTask",srlPrtObj},
        "UserData"->OptionValue["UserData"]];
    (*AppendTo[$SerialPortWriteAsyncTasks,asyncObj];*)
    Pause[0.2];

    Return@asyncObj;
    ]
    
RunSerialPortWriteAsynchronousTask[ srlPrtObj_/;vldArgs[RunSerialPortWriteAsynchronousTask,"object",srlPrtObj],
								eventfunc_,opts:OptionsPattern[]]/;validateOptions[RunSerialPortWriteAsynchronousTask,opts]&&Length[Cases[AsynchronousTasks[], 
 AsynchronousTaskObject[{"SerialPortWriteAsynchronousTask", 
   srlPrtObj, ___}, __]]]<1:=Module[
	{asyncObj},
	defineDevMessages[];
	update[srlPrtObj,"SerialPortWriteAsynchronousTaskCreatedQ",False];
	asyncObj = CreateSerialPortWriteAsynchronousTask[srlPrtObj,eventfunc,opts];
	Pause[0.2];
	StartAsynchronousTask[asyncObj];
	Return@asyncObj;
]

(*when writing a list of character codes*)
SerialPortWriteAsynchronous[asyncTaskObj_(*/;vldArgs[SerialPortWriteAsynchronous,"asyncObj",asyncTaskObj]*),cmd_(*/;vldArgs[SerialPortWriteAsynchronous,"data",cmd]*)]:=Module[
	{srlPrtObj,comport,devID,writeTerminator},
	defineDevMessages[];
	srlPrtObj = Extract[asyncTaskObj,{1,2}];
    comport = Extract[srlPrtObj,1];
    devID = comport/.$PortIDs;
    writeTerminator = validateWriteTerminator["WriteTerminator"/.(srlPrtObj/.$PortData)];
    If[Head[cmd]===List,cmd = FromCharacterCode[cmd]];
    asyncWrite[devID,cmd<>writeTerminator,++$asynchronousWriteIterator];
    Return@$asynchronousWriteIterator;
]


(*read*)


Options[RunSerialPortReadAsynchronousTask]:={"ReadTerminator"->Automatic,"UserData"->None,"FlushHandler"->None,"ReadCompleteHandler"->None,"ReadHandler"->None,"BreakHandler"->None,"UserData"->None}

Options[CreateSerialPortReadAsynchronousTask]:=Options@RunSerialPortReadAsynchronousTask


CreateSerialPortReadAsynchronousTask[
	srlPrtObj_/;Quiet@vldArgs[CreateSerialPortReadAsynchronousTask,"object",srlPrtObj],
	opts:OptionsPattern[]]/;(If[validOptionsQ[CreateSerialPortReadAsynchronousTask,opts]===True,validateOptions[CreateSerialPortReadAsynchronousTask,opts],False]):=	
	(Return@CreateSerialPortReadAsynchronousTask[srlPrtObj,"Byte",Infinity,Automatic,opts];)

RunSerialPortReadAsynchronousTask[
	srlPrtObj_/;Quiet@vldArgs[RunSerialPortReadAsynchronousTask,"object",srlPrtObj],
	opts:OptionsPattern[]]/;(If[validOptionsQ[RunSerialPortReadAsynchronousTask,opts]===True,validateOptions[RunSerialPortReadAsynchronousTask,opts],False]):=
	Return@RunSerialPortReadAsynchronousTask[srlPrtObj,"Byte",Infinity,Automatic,opts];

CreateSerialPortReadAsynchronousTask[
	srlPrtObj_/;Quiet@vldArgs[CreateSerialPortReadAsynchronousTask,"object",srlPrtObj],
	format_/;Quiet@vldArgs[CreateSerialPortReadAsynchronousTask,"format",format], 
	opts:OptionsPattern[]]/;(If[validOptionsQ[CreateSerialPortReadAsynchronousTask,opts]===True,validateOptions[CreateSerialPortReadAsynchronousTask,opts],False]):=
	(Return@CreateSerialPortReadAsynchronousTask[srlPrtObj,format,Infinity,Automatic,opts];)

RunSerialPortReadAsynchronousTask[
	srlPrtObj_/;Quiet@vldArgs[RunSerialPortReadAsynchronousTask,"object",srlPrtObj],
	format_/;Quiet@vldArgs[RunSerialPortReadAsynchronousTask,"format",format],
	opts:OptionsPattern[]]/;(If[validOptionsQ[RunSerialPortReadAsynchronousTask,opts]===True,validateOptions[RunSerialPortReadAsynchronousTask,opts],False]):=
	(Return@RunSerialPortReadAsynchronousTask[srlPrtObj,format,Infinity,Automatic,opts];)

CreateSerialPortReadAsynchronousTask[
	srlPrtObj_/;Quiet@vldArgs[CreateSerialPortReadAsynchronousTask,"object",srlPrtObj],
	format_/;Quiet@vldArgs[CreateSerialPortReadAsynchronousTask,"format",format],
	repeatSpec_/;Quiet@vldArgs[CreateSerialPortReadAsynchronousTask,"repeatSpec",repeatSpec],
	opts:OptionsPattern[]]/;(If[validOptionsQ[CreateSerialPortReadAsynchronousTask,opts]===True,validateOptions[CreateSerialPortReadAsynchronousTask,opts],False]):= 
	(Return@CreateSerialPortReadAsynchronousTask[srlPrtObj,format,repeatSpec,Automatic,opts])

RunSerialPortReadAsynchronousTask[
	srlPrtObj_/;Quiet@vldArgs[RunSerialPortReadAsynchronousTask,"object",srlPrtObj],
	format_/;Quiet@vldArgs[RunSerialPortReadAsynchronousTask,"format",format],
	repeatSpec_/;Quiet@vldArgs[RunSerialPortReadAsynchronousTask,"repeatSpec",repeatSpec],
	opts:OptionsPattern[]]/;(If[validOptionsQ[RunSerialPortReadAsynchronousTask,opts]===True,validateOptions[RunSerialPortReadAsynchronousTask,opts],False]):=
	Return@RunSerialPortReadAsynchronousTask[srlPrtObj,format,repeatSpec,Automatic,opts]

CreateSerialPortReadAsynchronousTask[srlPrtObj_/;Quiet@vldArgs[CreateSerialPortReadAsynchronousTask,"object",srlPrtObj],
								 format_/;Quiet@vldArgs[CreateSerialPortReadAsynchronousTask,"format",format],
								 repeatSpec_/;Quiet@vldArgs[CreateSerialPortReadAsynchronousTask,"repeatSpec",repeatSpec],
								 numberBytesPerBuffer_/;Quiet@vldArgs[CreateSerialPortReadAsynchronousTask,"numBytes",numberBytesPerBuffer],
								 opts:OptionsPattern[]]/;(If[validOptionsQ[CreateSerialPortReadAsynchronousTask,opts]===True,validateOptions[CreateSerialPortReadAsynchronousTask,opts],False])&&Length[Cases[AsynchronousTasks[], 
 AsynchronousTaskObject[{"SerialPortReadAsynchronousTask", 
   srlPrtObj, __}, __]]]<1:=
Module[
	{readHandler,readCompleteHandler,flushHandler,breakHandler,readTerminator,asyncObj,comport,devID,repeatspec,BYTES=0,STRING =1,byteCount},
	defineDevMessages[];
	Pause[0.2];
	readTerminator = validateReadTerminator[OptionValue@"ReadTerminator"];
	repeatspec = validateRepeatSpec[repeatSpec];
	byteCount  =  validateNumBytes[numberBytesPerBuffer];
	comport    = Extract[srlPrtObj,1];
    devID = comport/.$PortIDs;

	readHandler = OptionValue["ReadHandler"];
	readCompleteHandler = OptionValue["ReadCompleteHandler"];
	flushHandler = OptionValue["FlushHandler"];
	breakHandler = OptionValue["BreakHandler"];
	If[!IntegerQ@devID,Return@$Failed];

	If[format==="String",asyncObj = Quiet[Internal`CreateAsynchronousTask[createAsyncRead,{devID,byteCount,readTerminator,repeatspec,STRING}, 
		(
		Switch[#2,
			"data",If[!readHandler===None,readHandler[#1,FromCharacterCode[#3[[1]]]]],
			"repeatSpecDone",If[!readCompleteHandler===None,readCompleteHandler[#1,FromCharacterCode[#3[[1]]]]],
			"flush",If[!flushHandler===None,flushHandler[#1,FromCharacterCode[#3[[1]]]]],
			"break",If[!breakHandler===None,breakHandler[#1,FromCharacterCode[#3[[1]]]]]
		];callback[#1, #2, #3,format];) &,
		"TaskDetail"->{"SerialPortReadAsynchronousTask",srlPrtObj,format},
		"UserData"->OptionValue["UserData"]]];];
	
	If[format==="Byte",asyncObj = Quiet[Internal`CreateAsynchronousTask[createAsyncRead,{devID,byteCount,readTerminator,repeatspec,BYTES}, 
		(
		Switch[#2,
			"data",If[!readHandler===None,readHandler[#1,#3[[1]]]],
			"repeatSpecDone",If[!readCompleteHandler===None,readCompleteHandler[#1,#3[[1]]]],
			"flush",If[!flushHandler===None,flushHandler[#1,#3[[1]]]],
			"break",If[!breakHandler===None,breakHandler[#1,#3[[1]]]]
		];callback[#1, #2, #3,format];) &,
		"TaskDetail"->{"SerialPortReadAsynchronousTask",srlPrtObj,format},
		"UserData"->OptionValue["UserData"]]];];
	update[comport,"AsynchronousTask",asyncObj];
	Pause[0.2];
	Return@asyncObj;
	
	]

(*CreateSerialPortReadAsynchronousTask[srlPrtObj_,eventfunc_,format_,repeatSpec_,numberBytesPerBuffer_,opts__]/;If[Select[{opts},!OptionQ[#]&]==={},False,Message[SerialPortRead::nonopt,Select[{opts},!OptionQ[#]&],5,SerialPortRead]]:=Null;*)

RunSerialPortReadAsynchronousTask[srlPrtObj_/;Quiet@vldArgs[RunSerialPortReadAsynchronousTask,"object",srlPrtObj],
							  format_/;Quiet@vldArgs[RunSerialPortReadAsynchronousTask,"format",format],
							  repeatSpec_/;Quiet@vldArgs[RunSerialPortReadAsynchronousTask,"repeatSpec",repeatSpec],
							  numberBytesPerBuffer_/;Quiet@vldArgs[RunSerialPortReadAsynchronousTask,"numBytes",numberBytesPerBuffer],
							  opts:OptionsPattern[]]/;(If[validOptionsQ[RunSerialPortReadAsynchronousTask,opts]===True,validateOptions[RunSerialPortReadAsynchronousTask,opts],False])&&Length[Cases[AsynchronousTasks[], 
 AsynchronousTaskObject[{"SerialPortReadAsynchronousTask", 
   srlPrtObj, __}, __]]]<1(*/;(If[validOptionsQ[RunSerialPortReadAsynchronousTask,opts]===True &&vldArgs[RunSerialPortReadAsynchronousTask,srlPrtObj,format,repeatSpec,numberBytesPerBuffer],validateOptions[RunSerialPortReadAsynchronousTask,opts],False])*):=Module[
	{asyncObj},
	defineDevMessages[];
	
	asyncObj = CreateSerialPortReadAsynchronousTask[srlPrtObj,format,repeatSpec,numberBytesPerBuffer,opts];
	Pause[0.2];
	StartAsynchronousTask[asyncObj];
	Return@asyncObj;
]
asyncReadMessage[f_,x___]:=Module[{m,len,args,MAXARGS=4},
	m = {1->"object",2->"format",3->"repeatspec",4->"numBytes"};
	args = {x};
	
	len=Length[args];
	If[len===0,Return@Message[f::argrx,f,0,2]];
	If[len===1,Return@Message[f::argr,f,2]];
	If[len>MAXARGS,len=MAXARGS];

	For[i=1,i<=len,i++,
	
	If[!Quiet[vldArgs[f,i/.m,args[[i]]]] && !Quiet[OptionQ[args[[i]]]],Return@vldArgs[f,i/.m,args[[i]]]]
	(*for*)];
	len=Length[args];
	If[len>MAXARGS,
		
	If[!Quiet[OptionQ[args[[3]]]],Return[
		Message[f::nonopt,args[[3]],1,HoldForm[f[x]] (*message*)]
		(*return*)]
		(*if*)];]
	
	];
CreateSerialPortReadAsynchronousTask[x___]/;asyncReadMessage[CreateSerialPortReadAsynchronousTask,x]:=Null
RunSerialPortReadAsynchronousTask[x___]/;asyncReadMessage[RunSerialPortReadAsynchronousTask,x]:=Null

SerialPortFlush[srlPrtObj_SerialPort,"Read"]:=
Module[
	{comport,devID},
	defineDevMessages[];
	comport = Extract[srlPrtObj,1];
	devID = comport/.$PortIDs;
	readFlush[devID];
]
SerialPortFlush[srlPrtObj_SerialPort,"Write"]:=
Module[
	{comport,devID},
	defineDevMessages[];
	comport = Extract[srlPrtObj,1];
	devID = comport/.$PortIDs;
	writeFlush[devID];
]

CurrentSerialPortData[asyncReadObj_AsynchronousTaskObject]/;(Extract[asyncReadObj, {1, 1}]==="SerialPortReadAsynchronousTask"):=
Module[
	{a,s,format,returnVal},
	defineDevMessages[];
	format =Extract[asyncReadObj, {1, 3}] ;
	a = Extract[asyncReadObj,{1,2,1}];
	s=ToString[Extract[asyncReadObj,2]];(*StringJoin[StringSplit[a,{"/" -> "", "." -> "", "-" -> ""}]];*)
	Switch[format,
	"Byte",returnVal={},
	"String",returnVal=""];
	If[!(Head[Symbol["$latestData"<>s]]===List||Head[Symbol["$latestData"<>s]]===String),Return@returnVal];

	Return@Symbol["$latestData"<>s]
];

(*synchronous read*)

validateReadTerminator[readTerminator_]:= 
Module[
	{},
	defineDevMessages[];
	Switch[Head[readTerminator],
	String,Return@ToCharacterCode[readTerminator][[1]],
	Integer,Return@readTerminator,
	Symbol,Return@-1;
	];
]

validateNumBytes[numBytes_]:=
Module[
	{},
	defineDevMessages[];
	Switch[Head[numBytes],
		Integer,If[numBytes >0 && numBytes<102400,Return@numBytes,Return@$Failed],
		Symbol,Return@-1
		];
]

validateTimeout[timeout_Integer]:=(defineDevMessages[];If[timeout>=0,Return@timeout,Return@$Failed];)

validateTimeout[timeout_]:= Return@$Failed;

Options[SerialPortRead] := {"ReadTerminator"->Automatic,"Timeout"->10}

SerialPortRead[	srlPrtObj_/;Quiet@vldArgs[SerialPortRead,"object",srlPrtObj],
			opts:OptionsPattern[]]/;(If[validOptionsQ[SerialPortRead,opts]===True,validateOptions[SerialPortRead,opts],False]):=(Return@SerialPortRead[srlPrtObj,"Byte",Automatic,opts];)

SerialPortRead[	srlPrtObj_/;Quiet@vldArgs[SerialPortRead,"object",srlPrtObj],
			format_/;Quiet@vldArgs[SerialPortRead,"format",format],
			opts:OptionsPattern[]]/;(If[validOptionsQ[SerialPortRead,opts]===True,validateOptions[SerialPortRead,opts],False]):=(Return@SerialPortRead[srlPrtObj,format,Automatic,opts];)

SerialPortRead[	srlPrtObj_/;Quiet@vldArgs[SerialPortRead,"object",srlPrtObj],
			format_/;Quiet@vldArgs[SerialPortRead,"format",format],
			numBytes_/;Quiet@vldArgs[SerialPortRead,"numBytes",numBytes],
			opts:OptionsPattern[]]/;(If[validOptionsQ[SerialPortRead,opts]===True,validateOptions[SerialPortRead,opts],False]):=
Module[
	{readterminator,byteCount,timeout,str,comport,devID},
	defineDevMessages[];
	comport = Extract[srlPrtObj,1];
    devID = comport/.$PortIDs;
	byteCount = validateNumBytes[numBytes];
	If[byteCount===$Failed,Return@$Failed];
	
	readterminator = validateReadTerminator[OptionValue@"ReadTerminator"];
	
	timeout = validateTimeout[OptionValue@"Timeout"];
	
	str = read[devID,byteCount,readterminator,timeout];
	If[Head[str] ===List,Print["Hello"];str = str[[2]]];
	If[str===$TimedOut||str===$Failed,Return@str];
	If[str===Null && format ==="Byte",Return@{}];
	If[str===Null && format ==="String",Return@""];
	Switch[
		format,
		"Byte",Return@ToCharacterCode[str],
		"String",Return@str];
]

serialReadMessage[x___]:=Module[{m,len,args,MAXSERIALREADARGS=3},
	m = {1->"object",2->"format",3->"numBytes"};
	args = {x};
	
	len=Length[args];
	If[len===0,Return@Message[SerialPortRead::argx,SerialPortRead,0]];
	
	If[len>MAXSERIALREADARGS,len=MAXSERIALREADARGS];
	For[i=1,i<=len,i++,
	
	If[!Quiet[vldArgs[SerialPortRead,i/.m,args[[i]]]] && !Quiet[OptionQ[args[[i]]]],Return@vldArgs[SerialPortRead,i/.m,args[[i]]]]
	(*for*)];
	len=Length[args];
	If[len>MAXSERIALREADARGS,
		
	If[!Quiet[OptionQ[args[[4]]]],Return[
		Message[SerialPortRead::nonopt,args[[4]],3,HoldForm[SerialPortRead[x]] (*message*)]
		(*return*)]
		(*if*)];]
	
	];
SerialPortRead[x___]/;serialReadMessage[x]:=Null (*TODO*)

(*SerialPortRead[opts___]/;If[Select[{opts},!OptionQ[#]&]==={},False,Message[SerialPortRead::nonopt,Select[{opts},!OptionQ[#]&],3,SerialPortRead]]:=Null;*)
	
SerialPortClose[srlPrtObj_/;vldArgs[SerialPortRead,"object",srlPrtObj]]:=
Module[
	{devID,comport,err,asynReadObj,asyncWriteObj},
	defineDevMessages[];
	comport = Extract[srlPrtObj,1];
    devID = comport/.$PortIDs;
	$asyncObj = Null;

	asynReadObj = Quiet@Cases[AsynchronousTasks[],AsynchronousTaskObject[{"SerialPortReadAsynchronousTask", srlPrtObj, ___}, ___]][[1]];
	If[!asynReadObj==={},Quiet@RemoveAsynchronousTask[asynReadObj]];
	Pause[0.2];
	
	asyncWriteObj = Quiet[Cases[AsynchronousTasks[],AsynchronousTaskObject[{"SerialPortWriteAsynchronousTask", srlPrtObj, ___}, ___]][[1]]];
	If[!asyncWriteObj==={},Quiet@RemoveAsynchronousTask[asyncWriteObj]];
	Pause[0.2];
	
	
	RemoveAsynchronousTask["ErrorHandler"/.(srlPrtObj/.$PortData)];
	
	Pause[0.2];
	err = close[devID];
	iDisconnectDevice[devID];
	$PortIDs = DeleteCases[$PortIDs,(comport->_)];
	delete[srlPrtObj];
	$listOfOpenedSerialPorts = DeleteCases[$listOfOpenedSerialPorts,srlPrtObj];
]

storeCurrentData[asyncObj_, eventType_, eventData_,format_] := (
    mySetStringVariable["$latestData",ToString[Extract[asyncObj,2]], eventData,format])

callback[asyncObj_, eventType_,  eventData_,format_] := Module[{srlPrtObj},
	srlPrtObj = Extract[asyncObj,{1,2}];
	If[eventType==="data",storeCurrentData[asyncObj, eventType, eventData,format];];
	If[eventType=="repeatSpecDone",RemoveAsynchronousTask[asyncObj]];
	If[eventType=="taskCreated",StopAsynchronousTask[asyncObj]];
	If[eventType=="ReadTaskRemove",StopAsynchronousTask[asyncObj]];
	If[eventType=="overflow",Print["overflow"]];
]

asyncWriteCallback[connectionID_, asyncObj_, eventType_,  eventData_]:=(
(*If[eventType=="writedone",Print[eventData];];*)
If[eventType=="flush",Print[eventData];];
If[eventType == "taskCreated",Print["taskcreated"];StopAsynchronousTask[asyncObj];];
)
errHandlerCallback[serialPortObject_,eventType_,eventData_]:=
	(If[eventType==="Device not configured"||eventType==="portnotavailable",Message[SerialPort::ioerr];SerialPortClose[serialPortObject];])
mySetStringVariable[varNameRoot_String, index_String, value_,format_] := 
    (With[{v = ToExpression[varNameRoot <> index, InputForm, Unevaluated]},If[format==="String", v = FromCharacterCode[value[[1]]],v= value,v=value[[1]]]])
    
update[port_, propertyName_, value_] := 
 Module[{p, new},
 	p = Quiet[Position[$PortData,Cases[$PortData, (port -> x_) -> port -> x, Infinity][[1]]]];
  Switch[p, {_}, p = p[[1, 1]];
   
   delete[port, propertyName];
   new = Append[port /. $PortData, propertyName -> value];
   $PortData = Delete[$PortData, p];
   AppendTo[$PortData, port -> new];, 
   	{}, 
   AppendTo[$PortData, port -> {propertyName -> value}]]]

delete[port_, propertyName_] := 
 Module[{p, new, value,a}, 
  a = Cases[$PortData, (port -> x_) -> port -> x, Infinity];
  
  If[!a==={},p = Position[$PortData, a[[1]]]];
  
  Switch[p, {_}, p = p[[1, 1]];
   new = DeleteCases[port /. $PortData, propertyName -> value_];
   $PortData = Delete[$PortData, p];
   AppendTo[$PortData, port -> new];, {}, Null;]
   ]
delete[port_] := ($PortData = 
   Delete[$PortData, Position[$PortData, port][[1, 1]]])
(*done*)
End[]

EndPackage[]




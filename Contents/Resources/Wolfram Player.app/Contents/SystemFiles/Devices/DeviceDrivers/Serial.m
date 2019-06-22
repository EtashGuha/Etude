(* Mathematica Package *)

(* $Id$ *)

(*BeginPackage["SerialPort`", { "SerialLink`"}]*)
BeginPackage["SerialPort`"]

Needs["SerialLink`"]
(* Exported symbols added here with SymbolName::usage *)  

Begin["`Private`"] (* Begin Private Context *) 


(* ::Section:: *) (* API Registration Function *)
DeviceFramework`DeviceClassRegister[ "Serial",
	"OpenFunction" -> iSerialPortOpen,
	"CloseFunction" -> iSerialPortClose,
	"ReadFunction" -> iSerialPortRead,
	"WriteFunction" -> iSerialPortWrite,
	"ReadBufferFunction" -> iSerialPortReadBuffer,
	"WriteBufferFunction" -> iSerialPortWriteBuffer,
	"ExecuteFunction" -> iSerialPortMethods,
	"DeviceIconFunction" -> iSerialIconFunction
]

DeviceRead::serialargs = "No arguments are expected for DeviceRead on a Serial device.";
DeviceWrite::serialargs = "A single byte or a Item[\"Break\"] is expected."
DeviceReadBuffer::serialargs = "The number of bytes to read or a \"ReadTerminator\" is expected.";
DeviceWriteBuffer::serialargs = "A list of bytes or a string is expected.";
DeviceExecute::serialargs = "DeviceExecute was called with an unrecognized method or unexpected arguments.";
DeviceExecute::serialargs = "DeviceExecute of `1` expects `2`.";

(* ::Section:: *) (* Device API Adapter Functions *)

(* ::SubSection:: *) (* Port Management *)

$defaultSerialAddress = Which[
	$OperatingSystem === "Windows", "COM3",
	$SystemID === "LinuxARM","/dev/ttyAMA0",
	$OperatingSystem === "MacOSX" || $OperatingSystem === "Unix", "/dev/ttyS0"
]

(*iSerialPortOpen[ iHandle_]:= Check[  SerialPortOpen[ $defaultSerialAddress], $Failed];*)

iSerialPortOpen[ iHandle_]:= SerialPortOpen["/dev/ttyAMA0"];

iSerialPortOpen[ iHandle_, args___]:= Check[ SerialPortOpen[ args], $Failed]

iSerialPortClose[ {iHandle_, dHandle_}, args___]:= Check[ SerialPortClose[ dHandle], $Failed]

(* ::SubSection:: *) (* Synchronous Read *)

Options[ iSerialPortRead]:= Options[ SerialPortRead] 

(* Sync Read a single byte.  ReadTerminator is ignored *)
iSerialPortRead[{ iHandle_, port_SerialPort}, opts:OptionsPattern[]]:= iSerialPortRead[{ iHandle, port}, "Byte", opts]
iSerialPortRead[{ iHandle_, port_SerialPort}, "Byte", opts:OptionsPattern[]]:= Module[{ res}, res = SerialPortRead[ port, "Byte", 1, opts]; If[ res =!= $TimedOut, Return@First@res;, Return@res;];]
iSerialPortRead[{ iHandle_, port_SerialPort}, "String", opts:OptionsPattern[]]:= SerialPortRead[ port, "String", 1, opts]
iSerialPortRead[{ iHandle_, port_SerialPort}, args___]:= (
	Message[ DeviceRead::serialargs];
	Return[$Failed];
)
(*iSerialPortRead[{ iHandle_, port_SerialPort}, "String", opts:OptionsPattern[]]:= SerialPortRead[ port, "String", 1, opts]*)

(* Sync Read a list of bytes already buffered and available. *)

iSerialPortReadBuffer[{ iHandle_, port_SerialPort}, buffer_]:= SerialPortRead[ port, "Byte"];
iSerialPortReadBuffer[{ iHandle_, port_SerialPort}, n_Integer, buffer_]:= SerialPortRead[ port, "Byte", n]
iSerialPortReadBuffer[{ iHandle_, port_SerialPort}, "ReadTerminator" -> rt_, buffer_]:= SerialPortRead[ port, "Byte", "ReadTerminator"->rt]

iSerialPortReadBuffer[{ iHandle_, port_SerialPort}, args___]:= (
	Message[ DeviceReadBuffer::serialargs];
	Return[$Failed];
)

(* Read a list of bytes and format as a string *)
(*iSerialPortReadBuffer[{ iHandle_, port_SerialPort}, "String", opts:OptionsPattern[]]:= SerialPortRead[ port, "String", opts]*)

(*
(* Sync Read a list of byteCount bytes. *)
iSerialPortReadBuffer[{ iHandle_, port_SerialPort}, "Byte", byteCount_Integer, opts:OptionsPattern[]]:= SerialPortRead[ port, "Byte", byteCount, opts]:=
	SerialPortRead[ port, "Byte", byteCount, opts]

(* Sync Read a list of byteCount bytes and format as a string. *)
iSerialPortReadBuffer[{ iHandle_, port_SerialPort}, "String", byteCount_Integer, opts:OptionsPattern[]]:= SerialPortRead[ port, "String", byteCount, opts]*)

(* ::SubSection:: *) (* Synchronous Write*)

iSerialPortWrite[{ iHandle_, port_SerialPort}, byte_Integer]:= SerialPortWrite[ port, { byte}]

iSerialPortWrite[{ iHandle_, port_SerialPort}, Item["Break"]]:= SerialPortWrite[ port, Break]

iSerialPortWrite[{ iHandle_, port_SerialPort}, bytes__Integer]:= SerialPortWrite[ port, {bytes}]

iSerialPortWrite[{ iHandle_, port_SerialPort}, str_String]:= SerialPortWrite[ port, str]

iSerialPortWrite[{ iHandle_, port_SerialPort}, args___]:= (
	Message[ DeviceWrite::serialargs];
	Return[$Failed];
)

(* ::SubSection:: *) (* Synchronous Write Buffer*)
	
(*Options[ iSerialPortWriteBuffer] = { "Timeout" -> 0}*)

iSerialPortWriteBuffer[{ iHandle_, port_SerialPort}, bytes__Integer]:= SerialPortWrite[ port, {bytes}]

iSerialPortWriteBuffer[{ iHandle_, port_SerialPort}, string_String]:= SerialPortWrite[ port, string]

iSerialPortWriteBuffer[{ iHandle_, port_SerialPort}, args___]:= (
	Message[ DeviceWriteBuffer::serialargs];
	Return[$Failed];
)
	
(* ::SubSection:: *) (* Synchronous Write *)
	
iSerialPortMethods[ { iHandle_, port_SerialPort}, "SerialReadyQ", args___]:= iSerialPortReadyQ[ port, args]
iSerialPortReadyQ[ port_SerialPort]:= SerialPortReadyQ[ port]
iSerialPortReadyQ[ port_SerialPort, numBytes_Integer]:= SerialPortReadyQ[ port, numBytes]
iSerialPortReadyQ[ { iHandle_, port_SerialPort}, args___]:=  (
	Message[ DeviceExecute::serialargs, "SerialReadyQ", "no arguments or the number of bytes available"];
	Return[$Failed];
)


iSerialPortMethods[ { iHandle_, port_SerialPort}, "ReadFlush", args___]:= SerialPortFlush[ port, "Read"]

iSerialPortMethods[ port_SerialPort, args___]:= (
	Message[ DeviceExecute::serialargs];
	Return[$Failed];
)

iSerialIconFunction[{ iHandle_, port_SerialPort}, ___ ]:= Graphics[{Thickness[0.038461538461538464], 
  Style[{FilledCurve[{{{1, 4, 3}, {0, 1, 0}, {1, 3, 3}, {0, 1, 
        0}, {1, 3, 3}, {0, 1, 0}, {1, 3, 3}, 
                  {0, 1, 0}}}, {{{25.5, 2.5}, {25.5, 1.395}, {24.605, 
        0.5}, {23.5, 0.5}, {2.5, 0.5}, {1.395, 0.5}, {0.5, 
        1.395}, {0.5, 2.5}, {0.5, 23.5}, 
                  {0.5, 24.605}, {1.395, 25.5}, {2.5, 25.5}, {23.5, 
        25.5}, {24.605, 25.5}, {25.5, 24.605}, {25.5, 23.5}, {25.5, 
        2.5}}}]}, 
          FaceForm[RGBColor[0.941, 0.961, 0.957, 1.]]], 
        Style[{JoinedCurve[{{{1, 4, 3}, {0, 1, 0}, {1, 3, 3}, {0, 1, 
        0}, {1, 3, 3}, {0, 1, 0}, {1, 3, 3}, {0, 1, 0}}}, 
              {{{25.5, 2.5}, {25.5, 1.395}, {24.605, 0.5}, {23.5, 
        0.5}, {2.5, 0.5}, {1.395, 0.5}, {0.5, 1.395}, {0.5, 
        2.5}, {0.5, 23.5}, {0.5, 24.605}, 
                  {1.395, 25.5}, {2.5, 25.5}, {23.5, 25.5}, {24.605, 
        25.5}, {25.5, 24.605}, {25.5, 23.5}, {25.5, 2.5}}}, 
     CurveClosed -> {1}]}, 
          JoinForm[{"Miter", 10.}], RGBColor[0.7, 0.7, 0.7, 1.]], 
        Style[{FilledCurve[{{{1, 4, 3}, {0, 1, 0}, {1, 3, 3}, {0, 1, 
        0}, {1, 3, 3}, {0, 1, 0}, {1, 3, 3}, {0, 1, 0}}}, 
              {{{11.133, 18.727999999999998}, {11.133, 
        18.451999999999998}, {11.357000000000001, 
        18.227999999999998}, {11.633, 18.227999999999998}, 
                  {14.792, 18.227999999999998}, {15.068, 
        18.227999999999998}, {15.292, 18.451999999999998}, {15.292, 
        18.727999999999998}, 
                  {15.292, 20.639000000000003}, {15.292, 
        20.915}, {15.068, 21.139000000000003}, {14.792, 
        21.139000000000003}, {11.633, 21.139000000000003}, 
                  {11.357000000000001, 21.139000000000003}, {11.133, 
        20.915}, {11.133, 20.639000000000003}, {11.133, 
        18.727999999999998}}}]}, 
          FaceForm[RGBColor[0.5423, 0.63104, 0.63597, 1.]]], 
  Style[{FilledCurve[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}}, 
              {{{12.357000000000001, 1.}, {14.113000000000001, 
        1.}, {14.113000000000001, 9.554}, {12.357000000000001, 
        9.554}}}]}, 
          FaceForm[RGBColor[0.5, 0.5, 0.5, 1.]]], 
  Style[{FilledCurve[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 
        0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, 
                  {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 
        0}}}, {{{15.876000000000001, 19.799}, {8.122, 
        19.799}, {8.122, 11.516}, {10.573, 11.516}, 
                  {10.573, 11.493}, {11.982000000000001, 
        9.171}, {14.539, 9.171}, {15.876000000000001, 
        11.493}, {15.876000000000001, 11.516}, 
                  {18.326, 11.516}, {18.326, 
        19.799}, {15.876000000000001, 19.799}}}], 
    FilledCurve[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}}, 
              {{{8.806000000000001, 7.993}, {9.995000000000001, 
        7.993}, {9.995000000000001, 11.008}, {8.806000000000001, 
        11.008}}}], 
            
    FilledCurve[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}}, {{{16.5, 
        7.993}, {17.689, 7.993}, {17.689, 11.008}, {16.5, 11.008}}}]}, 
          FaceForm[RGBColor[0.624375, 0.695304, 0.691308, 1.]]]}, 
 ImageSize -> {26., 26.}, PlotRange -> {{0., 26.}, {0., 26.}}, 
      AspectRatio -> Automatic]

End[] (* End Private Context *)

EndPackage[]
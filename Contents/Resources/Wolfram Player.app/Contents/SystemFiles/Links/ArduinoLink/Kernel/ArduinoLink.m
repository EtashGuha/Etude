(* ::Package:: *)

(*==========================================================================================================
			
					ARDUINO LINK
			
Author: Ian Johnson
			
Copyright (c) 2015 Wolfram Research. All rights reserved.			


ArduinoLink is a package with DeviceFramework functionality setup to interface the Wolfram Language with 
an Arduino. 

CURRENT SUPPORTED BOARDS:
~Arduino Uno

USER ACCESSIBLE FUNCTIONS:

==========================================================================================================*)


BeginPackage["ArduinoLink`"]
(* Exported symbols added here with SymbolName::usage *) 

(*no public functions, as all functionality is through the device driver framework*)

Begin["`Private`"]
(* Implementation of the package *)

(*the arduino driver is an extension of the firmata driver*)
(*arduino upload is used for DeviceConfigure*)


$thisFileDir = DirectoryName@$InputFileName;

Needs["ArduinoUpload`"];
Needs["Firmata`"];

(*SymbolicC is for if any functions specified in DeviceConfigure are SymbolicC functions*)
Needs["SymbolicC`"];


(*paclet manager is for managing the paclet directory and such*)
Needs["PacletManager`"];

(*this is the only part where any of the packages needs to know where it is located*)
(*all other locations are passed as arguments from this package*)


$arduinoInternalInstallDirectory= FileNameJoin[{$UserBaseDirectory,"ApplicationData","Arduino"}]




(*BOOK KEEPING VARIABLES*)

(*the front end does weird things like try and call the configure function repeatedly for unknown reasons, *)
(*and so there is a check to make sure that DeviceConfigure doesn't get errantly called too many times*)
(*this fixes some random issues where when initially opening up a new kernel and evaluating DeviceOpen wouldn't actually upload because the evaluation was within*)
(*a second of the kernel being started and the initial value being set too close to the first use*)
(*so rather than initialize this to the current time, we'll just initialize it to 0, so that the first time it gets evaluated it doesn't get blocked by being too close to this value *)
(*being initialized*)
$lastConfigCall=0;
$pinConfigs = <||>;
$previousFunctions=<||>;
$functionCalls=<||>;

$SupportedBoardModels={"Uno","Yun"};

ports={{0,1,2,3,4,5,6,7},{8,9,10,11,12,13},{"A0","A1","A2","A3","A4","A5"}};

$DeviceStates=<||>;


(*for checking the pins, these are all the possible valid pins that can be used for an arduino uno*)
arduinoUnoPWMPins={3,5,6,9,10,11};
arduinoUnoPins={2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,"A0","a0","A1","a1","A2","a2","A3","a3","A4","a4","A5","a5"};
arduinoUnoAnalogPins={14,15,16,17,18,19,"A0","a0","A1","a1","A2","a2","A3","a3","A4","a4","A5","a5"};
arduinoPinToKey=<|
	14->"A0",15->"A1",16->"A2",17->"A3",18->"A4",19->"A5",
	"A0"->"A0","A1"->"A1","A2"->"A2","A3"->"A3","A4"->"A4","A5"->"A5",
	"a0"->"A0","a1"->"A1","a2"->"A2","a3"->"A3","a4"->"A4","a5"->"A5",
	0->"D0",1->"D1",2->"D2",3->"D3",4->"D4",5->"D5",6->"D6",7->"D7",8->"D8",9->"D9",10->"D10",11->"D11",12->"D12",13->"D13",
	(*this is added for additional security, just in case $pinConfiguration is ever accessed with "D8" or something*)
	"D0"->"D0","D1"->"D1","D2"->"D2","D3"->"D3","D4"->"D4","D5"->"D5","D6"->"D6","D7"->"D7","D8"->"D8","D9"->"D9","D10"->"D10","D11"->"D11","D12"->"D12","D13"->"D13"
|>;
pinKeyToDataDropName=<|Join[Table["A"<>ToString[i]->"AnalogPin"<>ToString[i],{i,0,5}],Table["D"<>ToString[i]->"DigitalPin"<>ToString[i],{i,0,5}]]|>;
pinToPort=<|14->2,"A0"->2,15->2,"A1"->2,16->2,"A2"->2,17->2,"A3"->2,18->2,"A4"->2,19->2,"A5"->2,0->0,1->0,2->0,3->0,4->0,5->0,6->0,7->0,8->1,9->1,10->1,11->1,12->1,13->1|>;

analogNumericPin=<|"A0"->0,"a0"->0,"A1"->1,"a1"->1,"A2"->2,"a2"->2,"A3"->3,"a3"->3,"A4"->4,"a4"->4,"A5"->5,"a5"->5,14->0,15->1,16->2,17->3,18->4,19->5|>;

analogStringPin=
<|
	"A0"->"A0", "a0"->"A0",14->"A0",
	"A1"->"A1","a1"->"A1",15->"A1",
	"A2"->"A2","a2"->"A2",16->"A2",
	"A3"->"A3","a3"->"A3",17->"A3",
	"A4"->"A4","a4"->"A4",18->"A4",
	"A5"->"A5","a5"->"A5",19->"A5"
|>

$compilationProgressIndicator = 0;

$arduinoSoftwarePresent=False;
$arduinoInstallLocation=None;
$avrgccInstallLocation=None;
$avrdudeInstallLocation=None;
$avrdudeConfigFileLocation=None;


(*MESSAGES*)
DeviceWrite::serialPin="Pins 0 and 1 are required for Serial communication";
DeviceWrite::notPWMPin="Pin `1` is not a PWM pin";
DeviceWrite::invalidPin="The pin `1` is not a valid Arduino Uno pin";
DeviceWrite::config="Pin `1` configured as an input, cannot write to it";
DeviceWrite::pwmresolution="The value `1` is not within the resolution supported, a value of `2` was used instead";
DeviceWrite::nonBooleanWrite="The value `1` is not a boolean (0 or 1), using `2` instead";
DeviceWrite::numericValue="The value `1` is not numeric";
DeviceWrite::bootFunction="A setup function was uploaded to the device, the function may still be running and the current operation may timeout."

DeviceRead::invalidPin="The pin `1` is not a valid Arduino Uno pin";
DeviceRead::config="Pin `1` configured as an output, cannot read from it";
DeviceRead::invalidArgs="DeviceRead takes a list of valid Arduino pins, `1` is not a valid Arduino pin or list of valid Arduino pins"
DeviceRead::bootFunction=DeviceWrite::bootFunction;

DeviceExecute::funcName="Function name `1` not found";
DeviceExecute::past="Cannot execute the function `1` seconds in the past";
DeviceExecute::invalidTiming="The time specification `1` is invalid";
DeviceExecute::noFunc="Function name `1` not found";
DeviceExecute::invalidFunc="Not a valid form of a function name";
DeviceExecute::invalidArgs="The arguments `1` are invalid for this function";
DeviceExecute::taskRunning="There is already a task running on the arduino, either wait for it to finish or delete the task";
DeviceExecute::needsArgs="The function `1` needs arguments";
DeviceExecute::bootFunction=DeviceWrite::bootFunction;
DeviceExecute::invalidBin="The Databin `1` is invalid and cannot be used";
DeviceExecute::invalidChannel="The Channel `1` is invalid and cannot be used";
DeviceExecute::noDatabin="A default Databin was not specified, one must be provided with the \"Databin\" option";
DeviceExecute::noChannel="A default Channel was not specified, one must be provided with the \"Channel\" option";
DeviceExecute::databinPin="A pin must be specified to upload to Data Drop";

DeviceExecute::invalidPin="The pin `1` is not a valid pin";
DeviceExecute::invalidKey="The key `1` is not a valid key for uploading to `2`";
DeviceExecute::invalidPinMode="The pinmode `1` is invalid and cannot be used";
DeviceExecute::onlyYun="`1` is only available on the Yun board.";
DeviceExecute::noDatabinAddFuncReg="A DatabinAdd function hasn't been uploaded, upload one with DeviceConfigure";
DeviceExecute::invalidAnalogPin="The pin `1` is not a valid analog input pin";
DeviceConfigure::invalidPin=DeviceExecute::invalidPin;
DeviceConfigure::channelPin="A pin must be specified to send on a channel";
DeviceConfigure::invalidAnalogPin=DeviceExecute::invalidAnalogPin;
DeviceConfigure::invalidMode="The mode `1` is not a valid pin configuration mode";
DeviceConfigure::invalidArgs="The argument `1` does not match the required type";
DeviceConfigure::noArduinoInstall="The Arduino software is not installed";
DeviceConfigure::invalidOptions="The option `1` is not a valid configuration option";
DeviceConfigure::invalidSetupFunc="The setup function option is invalid";
DeviceConfigure::bootFunction=DeviceWrite::bootFunction;
DeviceConfigure::databinID="A Databin ID must be specified to use the DatabinAdd BootFunction";
DeviceConfigure::channelID="A Channel must be specified to use the ChannelSend BootFunction";
DeviceConfigure::invalidDatabin="The Databin ID `1` is invalid and cannot be used"
DeviceConfigure::invalidChannel="The Channel `1` is invalid and cannot be used"
DeviceConfigure::databinPin="A pin must be specifed for DatabinAdd";
DeviceConfigure::invalidPin=DeviceRead::invalidPin;
DeviceConfigure::invalidKey="The specified key `1` is not a string and cannot be used";
DeviceConfigure::invalidPinMode=DeviceExecute::invalidPinMode;
DeviceConfigure::invalidAnalogPin="The pin `1` is not a valid analog pin";
DeviceConfigure::datadropDisabled="DataDrop support has explicitly been disabled with the option \"DataDrop\", set \"DataDrop\" to True or Automatic to use DataDrop functionality ";
DeviceConfigure::channelDisabled="ChannelFramework support has been explictly disabled with the option \"ChannelFramework\"; set \"ChannelFramework\" to True or Automatic to use ChannelFramework functionality"
DeviceConfigure::onlyYunSetupFunc="The DatabinAdd BootFunction is only available on the Arduino Yun";
DeviceConfigure::onlyYunDatadrop="DataDrop support is only available on the Arduino Yun";
DeviceConfigure::onlyYunChannelFramework="ChannelFramework support is only available on the Arduino Yun";
DeviceConfigure::noFunctionCode="The option \"Code\" must be specified when uploading a function in \"Functions\"";
DeviceConfigure::invalidFunction="The function `1` is invalid and cannot be uploaded";
DeviceConfigure::pinReadmodeShape="The number of pins specified and the number of Readmodes specified must be the same length";
DeviceConfigure::invalidCustomKey="The ID `1` is invalid, only ASCII Strings are supported";

DeviceSetProperty::invalidInstallLocation="The location `1` is not a valid Arduino installation";
DeviceSetProperty::noWrite="The property `1` isn't writable";

DeviceOpen::port="The name of the Serial port the Arduino is attached to must be specified";
DeviceOpen::serialDrivers="Serial drivers for the Arduino were not installed, install and reopen ";
DeviceOpen::raspiAutoInstall="The Arduino software was not found on this machine. On the Raspberry Pi, the Arduino software cannot be automatically installed, please install it with \"sudo apt-get update && sudo apt-get install -y arduino-core\"";
DeviceOpen::invalidInput="Please enter either True or False";
DeviceOpen::unsupportedBoard="The board model `1` is not supported.";
DeviceOpen::extractFail="Failed extracting Arduino software from `1`"
(*FUNCTIONS*)

driversInstalled[]:=Module[{},
	(
		TrueQ[Quiet[FileExistsQ[First[FileNames["arduino*",FileNameJoin[{Environment["windir"],"System32","DriverStore","FileRepository"}]]]]]]
	)	
];


(*manager handle driver is called in the device open sequence of events, and returns a unique uuid for each device that is used as the ihandle*)
ManagerHandleDriver[args___]:=CreateUUID[];



(*downloads and installs the arduino software into $UserBaseDirectory/arduino*)
(*returns the location of the base arduino directory*)
installArduino[]:=Module[
	{
		windowsURL = "http://arduino.cc/download.php?f=/arduino-1.6.7-windows.zip",
		linux32URL = "http://arduino.cc/download.php?f=/arduino-1.6.7-linux32.tar.xz",
		linux64URL = "http://arduino.cc/download.php?f=/arduino-1.6.7-linux64.tar.xz",
		macURL = "http://arduino.cc/download.php?f=/arduino-1.6.7-macosx.zip",
		progress=0.0,
		tempLocation,
		tempPrintCell,
		asynchObject,
		progressBarCell,
		driverOpen,
		extractRes
	},
	(
		(*progFunction is the function that URLSaveAsynchronous calls for progress*)
		(*the addition of a very small number stops it from being indeterminate*)
		progFunction[_, "progress", {dlnow_, dltotal_, _, _}] := Quiet[progress = dlnow/(dltotal+0.0000000000000001)];
		Switch[$OperatingSystem,
			"Windows",
			(
				tempLocation=FileNameJoin[{$TemporaryDirectory,Last@FileNameSplit@windowsURL}];
				(*print message for current task - first is downloading*)
				tempPrintCell = PrintTemporary["Downloading Arduino IDE from "<>windowsURL];
				(*only print off the progress bar if we have a front end*)
				If[$FrontEnd=!=Null,progressBarCell = PrintTemporary[Dynamic[ProgressIndicator[progress]]]];
				(*we use URLSave asynchronous to display a progress bar for the user*)
				asynchObject=URLSaveAsynchronous[windowsURL,tempLocation,progFunction,"ConnectTimeout"->5,"Progress"->True];
				(*now wait until the download is done*)
				WaitAsynchronousTask[asynchObject];
				(*delete the first message and the progress bar*)
				NotebookDelete[tempPrintCell];
				(*if we printed off the progress bar, delete it now*)
				If[$FrontEnd=!=Null,NotebookDelete[progressBarCell]];
				(*now while extracting it print a new message*)
				tempPrintCell = PrintTemporary["Extracting archive into "<>$arduinoInternalInstallDirectory];
				(*extract the downloaded archive file and make the arduino ide folder inside $UserBaseDirectory if it doesn't already exist*)
				ExtractArchive[tempLocation,$arduinoInternalInstallDirectory,CreateIntermediateDirectories->True];
				(*finally delete the print message again*)
				NotebookDelete[tempPrintCell];
				(*delete the temp file as well*)
				DeleteFile[tempLocation];
				
				(*check to ensure that the folder extracted is named appropriately*)
				With[{possibleFolderNames = FileNames["arduino*",$arduinoInternalInstallDirectory]},
					If[Length[possibleFolderNames] == 1,
						(*THEN*)
						(*then there's only one possible folder name, so it succeeded extracting*)
						(
							If[FileNameTake[First[possibleFolderNames]]=!="arduino",
								(*THEN*)
								(*the directory isn't named arduino, so rename it*)
								RenameDirectory[First[possibleFolderNames],FileNameJoin[{FileNameDrop[First[possibleFolderNames]],"arduino"}]]
							];
						),
						(*ELSE*)
						(*failed to extract*)
						(
							Message[DeviceOpen::extractFail,windowsURL];
							Return[$Failed];
						)
					]
				];
				
				
				(*before we're done with installing the arduino software, we need to check the driver situation*)
				(*if the drivers are installed, then we don't need to do anything and can just return*)
				(*if the drivers are not installed, opening the serial port will always fail. However, we don't want to be responsible for installing drivers,*)
				(*so we instead open up the path to the driver install utility in the driver, and make the user do it*)
				If[driversInstalled[],
					(*THEN*)
					(*drivers are installed, just return the directory we installed the software to back*)
					(
						Return[FileNameJoin[{$arduinoInternalInstallDirectory,"arduino"}]]
					),
					(*ELSE*)
					(*drivers are not installed, so we need to open up the directory for the user to download the drivers*)
					(
						driverOpen=ChoiceDialog[
							Column[
								{
									TextCell[
										"The Arduino software has been installed but the Arduino USB driver " <>
										"is still required in order to use ArduinoLink. You can install the " <> "driver by running " <> 
										Switch[$SystemID, "Windows-x86-64", "dpinst-amd64.exe", "Windows", "dpinst-x86.exe"]<>".\n\n"
									],
									Style["Note: Administrative privileges will be required to install the driver.", Bold], 
									TextCell[
										"\nPress \"Take me there\" to open the directory containing this driver or \"Cancel\" "<>
										"to return to the Wolfram System."
									]
								}
							],
							{"Take me there" -> True,"Cancel" -> False}, "WindowSize" -> {500, 250}
						];
						If[TrueQ[driverOpen],
							SystemOpen[FileNameJoin[{$arduinoInternalInstallDirectory, "arduino", "drivers"}]]
						];
						(*now return the directory that was just installed*)
						(*raise a message about serial drivers and return $Failed, because the drivers weren't installed*)
						Message[DeviceOpen::serialDrivers];
						Return[$Failed];
					)
				];
			),
			"MacOSX",
			(
				tempLocation=FileNameJoin[{$TemporaryDirectory,Last@FileNameSplit@macURL}];
				(*print message for current task - first is downloading*)
				tempPrintCell = PrintTemporary["Downloading Arduino IDE from "<>macURL];
				(*only print off the progress bar if we have a front end*)
				If[$FrontEnd=!=Null,progressBarCell = PrintTemporary[Dynamic[ProgressIndicator[progress]]]];
				(*we use URLSave asynchronous to display a progress bar for the user*)
				asynchObject=URLSaveAsynchronous[macURL,tempLocation,progFunction,"ConnectTimeout"->5,"Progress"->True];
				(*now wait until the download is done*)
				WaitAsynchronousTask[asynchObject];
				(*delete the first message and the progress bar*)
				NotebookDelete[tempPrintCell];
				(*if we printed off the progress bar, delete it now*)
				If[$FrontEnd=!=Null,NotebookDelete[progressBarCell]];
				(*now while extracting it print a new message*)
				tempPrintCell = PrintTemporary["Extracting archive into "<>$arduinoInternalInstallDirectory];
				(*extract the downloaded archive file and make the arduino ide folder inside $UserBaseDirectory if it doesn't already exist*)
				extractRes = ExtractArchive[tempLocation,$arduinoInternalInstallDirectory,CreateIntermediateDirectories->True];
				If[extractRes === $Failed,
					(*THEN*)
					(*failed to extract the file, raise message and fail*)
					(
						Message[DeviceOpen::extractFail,macURL];
						Return[$Failed];
					)
				];
				(*finally delete the print message again*)
				NotebookDelete[tempPrintCell];
				(*delete the temp file as well*)
				DeleteFile[tempLocation];
				Return[FileNameJoin[{$arduinoInternalInstallDirectory,"Arduino.app"}]];
			),
			"Unix",
			(
				(*switch on type of operating sysyem, there are two different archives*)
				Switch[$ProcessorType,
					"ARM",
					(
						(*note we don't handle linux arm as we can't install the arduino software on raspi*)
						Message[DeviceOpen::raspiAutoInstall];
						Return[$Failed];
					),
					"x86",
					(
						tempPrintCell = PrintTemporary["Downloading Arduino IDE from "<>linux32URL];
						tempLocation = FileNameJoin[{$TemporaryDirectory,Last@FileNameSplit@linux32URL}];
						(*only print off the progress bar if we have a front end*)
						If[$FrontEnd=!=Null,progressBarCell = PrintTemporary[Dynamic[ProgressIndicator[progress]]]];
						(*we use URLSave asynchronous to display a progress bar for the user*)
						asynchObject = URLSaveAsynchronous[linux32URL,tempLocation,progFunction,"ConnectTimeout"->5,"Progress"->True];
					),
					"x86-64",
					(
						tempPrintCell = PrintTemporary["Downloading Arduino IDE from "<>linux64URL];
						tempLocation = FileNameJoin[{$TemporaryDirectory,Last@FileNameSplit@linux64URL}];
						(*only print off the progress bar if we have a front end*)
						If[$FrontEnd=!=Null,progressBarCell = PrintTemporary[Dynamic[ProgressIndicator[progress]]]];
						(*we use URLSave asynchronous to display a progress bar for the user*)
						asynchObject = URLSaveAsynchronous[linux64URL,tempLocation,progFunction,"ConnectTimeout"->5,"Progress"->True];
					)
				];
				(*now wait until the download is done*)
				WaitAsynchronousTask[asynchObject];
				(*delete the first message and the progress bar*)
				NotebookDelete[tempPrintCell];
				(*if we printed off the progress bar, delete it now*)
				If[$FrontEnd=!=Null,NotebookDelete[progressBarCell]];
				(*now while extracting it print a new message*)
				tempPrintCell = PrintTemporary["Extracting archive into "<>$arduinoInternalInstallDirectory];
				(*because we won't use ExtractArchive, and are using the terminal tar command instead, we should make sure the directory doesn't exist first, then we can just rename the extracted archive*)
				If[Not[DirectoryQ[$arduinoInternalInstallDirectory]],
					(*THEN*)
					(*the directory doesn't exist, so we need to create it*)
					(
						CreateDirectory[$arduinoInternalInstallDirectory];
					),
					(*ELSE*)
					(*the directory already exists, so we should delete it first and then make it fresh*)
					(
						DeleteDirectory[$arduinoInternalInstallDirectory,DeleteContents->True];
						CreateDirectory[$arduinoInternalInstallDirectory];
					)
				];
				tempPrintCell = PrintTemporary["Extracting archive into "<>$arduinoInternalInstallDirectory];
				(*ExtractArchive doesn't support xz file format, so use the terminal and do it manually instead*)
				extractRes = RunProcess[{"tar","xf",tempLocation,"-C",$arduinoInternalInstallDirectory}];
				If[extractRes["ExitCode"] =!= 0,
					(*THEN*)
					(*failed to extract the file, raise message and fail*)
					(
						Message[DeviceOpen::extractFail,Switch[$ProcessorType,"x86",linux32URL,"x86-64",linux64URL]];
						Return[$Failed];
					)
				];
				(*now rename the Directory*)
				RenameDirectory[First[FileNames["*",$arduinoInternalInstallDirectory]],FileNameJoin[{$arduinoInternalInstallDirectory,"arduino"}]];
				NotebookDelete[tempPrintCell];
				DeleteFile[tempLocation];
				Return[FileNameJoin[{$arduinoInternalInstallDirectory,"arduino"}]];
			),
			_,
			(
				(*any other kind of $OperatingSystem, don't do anything*)
				Return[$Failed];
			)
		]
	)
];

findavrGCC[location_String]:=Module[{avrGCC},
	(
		Switch[$OperatingSystem,
			"Windows",
			(
				(*on windows, only valid location is at hardware/tools/avr/bin*)
				avrGCC=FileNameJoin[{location,"hardware","tools","avr","bin","avr-gcc.exe"}];
				If[FileExistsQ[avrGCC],
					(*THEN*)
					(*it exists, return it*)
					Return[FileNameDrop@avrGCC],
					(*ELSE*)
					(*somehow doesn't exist, return $Failed*)
					Return[$Failed]
				];
			),
			"MacOSX",
			(
				(*on mac, could be one of :
					Contents/Resources/Java/hardware/tools/avr/bin/avr-gcc
					Contents/Java/hardware/tools/avr/bin/avr-gcc
					hardware/tools/avr/bin/avr-gcc
				*)
				Which[
					FileExistsQ[avrGCC=FileNameJoin[{location,"Contents","Resources","Java","hardware","tools","avr","bin","avr-gcc"}]],
					(
						Return[FileNameDrop@avrGCC];
					),
					FileExistsQ[avrGCC=FileNameJoin[{location,"Contents","Java","hardware","tools","avr","bin","avr-gcc"}]],
					(
						Return[FileNameDrop@avrGCC];
					),
					FileExistsQ[avrGCC=FileNameJoin[{location,"hardware","tools","avr","bin","avr-gcc"}]],(*legacy 00XX versions*)
					(
						Return[FileNameDrop@avrGCC];
					),
					True,
					(
						(*can't find it anywhere, return $Failed*)
						Return[$Failed];
					)
				];
			),
			"Unix",
			(
				(*on linux, just have to check:
					$PATH
					/hardware/tools/avr/bin/avr-gcc
					/hardware/tools/avr/bin.gcc/avr-gcc
				*)
				Which[
					FileExistsQ[avrGCC=FileNameJoin[{location,"hardware","tools","avr","bin","avr-gcc"}]],
					(
						Return[FileNameDrop@avrGCC];
					),
					FileExistsQ[avrGCC=FileNameJoin[{location,"hardware","tools","avr","bin.gcc","avr-gcc"}]],
					(
						Return[FileNameDrop@avrGCC];
					),
					True,
					(
						(*check the path*)
						avrGCC=Select[FileExistsQ]@(FileNameJoin[{#,"avr-gcc"}]&/@StringSplit[Environment["PATH"],":"]);
						If[avrGCC==={},
							(*THEN*)
							(*didn't find avr-gcc anywhere*)
							Return[$Failed],
							(*ELSE*)
							(*found at least one, so return the first one*)
							Return[FileNameDrop@First[avrGCC]];
						];
					)
				];
			),
			_,
			(
				(*if somehow other operating system, it's unsupported*)
				Return[$Failed];
			)
		]
	)
];

findavrdude[location_String]:=Module[{avrdude},
	(
		Switch[$OperatingSystem,
			"Windows",
			(
				(*on windows, only valid location is at hardware/tools/avr/bin*)
				avrdude=FileNameJoin[{location,"hardware","tools","avr","bin","avrdude.exe"}];
				If[FileExistsQ[avrdude],
					(*THEN*)
					(*it exists, return it*)
					Return[FileNameDrop@avrdude],
					(*ELSE*)
					(*somehow doesn't exist, return $Failed*)
					Return[$Failed]
				];
			),
			"MacOSX",
			(
				(*on mac, could be one of :
					Contents/Resources/Java/hardware/tools/avr/bin/avrdude
					Contents/Java/hardware/tools/avr/bin/avrdude
					hardware/tools/avr/bin/avrdude
				*)
				Which[
					FileExistsQ[avrdude=FileNameJoin[{location,"Contents","Resources","Java","hardware","tools","avr","bin","avrdude"}]],
					(
						Return[FileNameDrop@avrdude];
					),
					FileExistsQ[avrdude=FileNameJoin[{location,"Contents","Java","hardware","tools","avr","bin","avrdude"}]],
					(
						Return[FileNameDrop@avrdude];
					),
					FileExistsQ[avrdude=FileNameJoin[{location,"hardware","tools","avr","bin","avrdude"}]],(*legacy 00XX versions*)
					(
						Return[FileNameDrop@avrdude];
					),
					True,
					(
						(*can't find it anywhere, return False*)
						Return[$Failed];
					)
				];
			),
			"Unix",
			(
				(*on linux, just have to check:
					$PATH
					/hardware/tools/avrdude
					/hardware/tools/avr/bin/avrdude
				*)
				(*check avrdude first, trying to find the right directory, or using the path if necessary*)
				Which[
					FileExistsQ[avrdude=FileNameJoin[{location,"hardware","tools","avr","bin","avrdude"}]],
					(
						Return[FileNameDrop@avrdude];
					),
					FileExistsQ[avrdude=FileNameJoin[{location,"hardware","tools","avrdude"}]],
					(
						Return[FileNameDrop@avrdude];
					),
					True,
					(
						(*check the path*)
						avrdude=Select[FileExistsQ]@(FileNameJoin[{#,"avrdude"}]&/@StringSplit[Environment["PATH"],":"]);
						If[avrdude==={},
							(*THEN*)
							(*didn't find avr-gcc anywhere*)
							Return[$Failed],
							(*ELSE*)
							(*found at least one, so return the first one*)
							Return[FileNameDrop@First[avrdude]];
						];
					)
				];
			),
			_,
			(
				(*if somehow other operating system, it's unsupported*)
				Return[$Failed];
			)
		]
	)
];


(*avrdude needs a configure file, so we have to find that in the arduino install too*)
findavrdudeConf[location_String]:=Module[{avrdudeConf},
	(
		Switch[$OperatingSystem,
			"Windows",
			(
				(*on windows, only valid location is at hardware/tools/avr/etc*)
				avrdudeConf=FileNameJoin[{location,"hardware","tools","avr","etc","avrdude.conf"}];
				If[FileExistsQ[avrdudeConf],
					(*THEN*)
					(*it exists, return it*)
					Return[avrdudeConf],
					(*ELSE*)
					(*somehow doesn't exist, return $Failed*)
					Return[$Failed]
				];
			),
			"MacOSX",
			(
				(*on mac, could be one of :
					Contents/Resources/Java/hardware/tools/avr/etc/avrdude.conf
					Contents/Java/hardware/tools/avr/etc/avrdude.conf
					hardware/tools/avr/etc/avrdude.conf
				*)
				Which[
					FileExistsQ[avrdudeConf=FileNameJoin[{location,"Contents","Resources","Java","hardware","tools","avr","etc","avrdude.conf"}]],
					(
						Return[avrdudeConf];
					),
					FileExistsQ[avrdudeConf=FileNameJoin[{location,"Contents","Java","hardware","tools","avr","etc","avrdude.conf"}]],
					(
						Return[avrdudeConf];
					),
					FileExistsQ[avrdudeConf=FileNameJoin[{location,"hardware","tools","avr","etc","avrdude.conf"}]],(*legacy 00XX versions*)
					(
						Return[avrdudeConf];
					),
					True,
					(
						(*can't find it anywhere, return False*)
						Return[$Failed];
					)
				];
			),
			"Unix",
			(
				(*on linux, just have to check:
					/etc/avrdude.conf
					/hardware/tools/avrdude.conf
					/hardware/tools/avr/etc/avrdude.conf
				*)
				(*check avrdude first, trying to find the right directory, or using the path if necessary*)
				Which[
					FileExistsQ[avrdudeConf=FileNameJoin[{location,"hardware","tools","avr","etc","avrdude.conf"}]],
					(
						Return[avrdudeConf];
					),
					FileExistsQ[avrdudeConf=FileNameJoin[{location,"hardware","tools","avrdude.conf"}]],
					(
						Return[avrdudeConf];
					),
					True,
					(
						(*check the installed path*)
						If[FileExistsQ[avrdudeConf=FileNameJoin[{$RootDirectory,"etc","avrdude.conf"}]],
							(*THEN*)
							(*found it, return that*)
							(
								Return[avrdudeConf];
							),
							(*ELSE*)
							(*don't try searching anywhere else, no other general location to try and look*)
							(
								Return[$Failed];
							)
						];
					)
				];
			),
			_,
			(
				(*if somehow other operating system, it's unsupported*)
				Return[$Failed];
			)
		]
	)
];


(*validArduinoInstallLocation will verify that the given location is a usable installation of the arduino software*)
(*specifically, just makes sure we can find avr-gcc, avrdude, and avrdude's config file*)
validArduinoInstallLocation[location_String]:=
(
	(*just make sure we can find avrdude and avrgcc*)
	Return[findavrdude[location]=!=$Failed && findavrGCC[location]=!=$Failed && findavrdudeConf[location]=!=$Failed]
)


(*arduinoSoftwarePresent checks a few different locations for the arduino IDE software*)
arduinoSoftwarePresent[]:=Module[
	{
		(*locations to check*)
		macLocation=FileNameJoin[{$RootDirectory,"Applications","Arduino.app"}],
		programFilesLocation = FileNameJoin[
			{
				(*this is the current drive*)
				First[FileNameSplit[$InstallationDirectory]],
				"Program Files (x86)",
				"Arduino"
			}],
		linuxLocation=FileNameJoin[{$RootDirectory,"usr","share","arduino"}],
		(*for mac, can't just check the base directory, need to check the app folder inside that*)
		internalLocation=FileNameJoin[{$arduinoInternalInstallDirectory,"arduino"}],
		macInternalLocation = FileNameJoin[{$arduinoInternalInstallDirectory,"Arduino.app"}]
	},
	(
		(*this directory is different on Mac, on mac it is inside the .app Contents folder, then inside the Java folder*)
		If[$OperatingSystem==="MacOSX",
			(*THEN*)
			(*this is a mac, so check the mac locations*)
			If[FileExistsQ[macInternalLocation]&&validArduinoInstallLocation[macInternalLocation],
				Return[True];
			],
			(*ELSE*)
			(*check the normal locations*)
			If[FileExistsQ[internalLocation]&&validArduinoInstallLocation[internalLocation],
				Return[True];
			]
		];
		Switch[$OperatingSystem,
			"MacOSX",
			(
				(*on mac check the Arduino.app folder*)
				Return[FileExistsQ[macLocation]&&validArduinoInstallLocation[macLocation]]
			),
			"Windows",
			(
				(*on windows check the Arduino program files directory folder*)
				Return[FileExistsQ[programFilesLocation]&&validArduinoInstallLocation[programFilesLocation]]
			),
			"Unix",
			(
				(*this is the confirmed location from apt-get install on debian and ubuntu*)
				(*on linux, check the Arduino usr/share/arduino directory folder*)
				(*note that on linux, avr-gcc could be on the path, via apt or yum or something, and validArduinoInstallLocation checks this location last, so if we don't also check*)
				(*to make sure that the linux location exists, we can still find it, as validArduinoInstallLocation will work properly*)
				Return[validArduinoInstallLocation[linuxLocation]]
			),
			_,
			(*not sure what other $OperatingSystem this would be, so just return False*)
			(
				Return[False];
			)
		]
	)
];


(*getArduinoSoftwareLocation does the exact same thing as arduinoSoftwarePresent, except it returns the location*) 
(*instead of a boolean*)
getArduinoSoftwareLocation[]:=Module[
	{
		macLocation=FileNameJoin[{$RootDirectory,"Applications","Arduino.app"}],
		windowsLocation = FileNameJoin[
			{
				(*this is the current drive*)
				First[FileNameSplit[$InstallationDirectory]],
				"Program Files (x86)",
				"Arduino"
			}],
		linuxLocation=FileNameJoin[{$RootDirectory,"usr","share","arduino"}],
		internalLocation=FileNameJoin[{$arduinoInternalInstallDirectory,"arduino"}],
		macInternalLocation = FileNameJoin[{$arduinoInternalInstallDirectory,"Arduino.app"}]
	},
	(
		If[$OperatingSystem==="MacOSX",
			(*THEN*)
			(*check the mac internal location inside the Arduino.app first before checking anything else*)
			If[FileExistsQ[macInternalLocation]&&validArduinoInstallLocation[macInternalLocation],
				(*THEN*)
				(*the internal applications data folder exists on mac and it is valid, so use that one*)
				Return[macInternalLocation];
				(*ELSE*)
				(*go on to check the other locations*)
			],
			(*ELSE*)
			(*check the normal location first*)
			(
				If[FileExistsQ[internalLocation]&&validArduinoInstallLocation[internalLocation],
					(*THEN*)
					(*it does exist in the Mac internal directory, so return that*)
					(
						Return[internalLocation];
					)
					(*ELSE*)
					(*go on to check the other locations*)
				]
			)
		];
		Switch[$OperatingSystem,
			"MacOSX",
			(
				(*on mac check the Arduino.app folder*)
				If[FileExistsQ[macLocation],
					(*THEN*)
					(*it exists, so return wether or not it is a valid location*)
					(
						If[validArduinoInstallLocation[macLocation],
							(*THEN*)
							(*it is valid, return that directory*)
							(
								Return[macLocation];
							),
							(*ELSE*)
							(*it doesn't exist, return None*)
							(
								Return[None];
							)
						]
					),
					(*ELSE*)
					(*it doesn't exist, so return False*)
					(
						Return[None];
					)
				]
			),
			"Windows",
			(
				(*on windows check the Arduino program files directory folder*)
				If[FileExistsQ[windowsLocation],
					(*THEN*)
					(*it exists, so return wether or not it is a valid location*)
					(
						If[validArduinoInstallLocation[windowsLocation],
							(*THEN*)
							(*it is valid, return that directory*)
							(
								Return[windowsLocation];
							),
							(*ELSE*)
							(*it isn't valid, return None*)
							(
								Return[None];
							)
						];
					),
					(*ELSE*)
					(*it doesn't exist, so return False*)
					(
						Return[None];
					)
				]
			),
			"Unix",
			(
				(*this is the confirmed location from apt-get install on debian and ubuntu*)
				(*on linux, check the Arduino usr/share/arduino directory folder*)
				(*as before, if we don't check to make sure that linuxLocation actually exists, we can silently find the versions of avr-gcc and avrdude that are on the path in some*)
				(*installs*)
				If[validArduinoInstallLocation[linuxLocation],
					(*THEN*)
					(*the location is valid, return that directory*)
					(
						Return[linuxLocation];
					),
					(*ELSE*)
					(*it isn't valid, return None*)
					(
						Return[None];
					)
				]
			),
			_,
			(*not sure what this case is, but return None for any other kind of $OperatingSystem*)
			(
				Return[None];
			)
		]
	)
];


ArduinoOpenDriver[ihandle_,___,OptionsPattern[]]:=Module[{},
	(
		Message[DeviceOpen::port];
		Return[$Failed];
	)
];


(*ArduinoOpenDriver will make sure that the arduino software is installed, and if it isn't,
it will prompt the user to install it.*)
Options[ArduinoOpenDriver]=
	{
		"InitialUpload"->True,
		"BoardType"->Default
	};
ArduinoOpenDriver[ihandle_,serialPort_String,OptionsPattern[]]:=Module[
	{
		defaultDeviceState=<|
			"D0"-><|"Direction"->"Reserved","LastWriteValue"->"Reserved","PWM"->"Reserved","ADC"->"Reserved","LastReadValue"->"Reserved"|>,
			"D1"-><|"Direction"->"Reserved","LastWriteValue"->"Reserved","PWM"->"Reserved","ADC"->"Reserved","LastReadValue"->"Reserved"|>,
			"D2"-><|"Direction"->Default,"LastWriteValue"->None,"PWM"->False,"ADC"->False,"LastReadValue"->None|>,
			"D3"-><|"Direction"->Default,"LastWriteValue"->None,"PWM"->True,"ADC"->False,"LastReadValue"->None|>,
			"D4"-><|"Direction"->Default,"LastWriteValue"->None,"PWM"->False,"ADC"->False,"LastReadValue"->None|>,
			"D5"-><|"Direction"->Default,"LastWriteValue"->None,"PWM"->True,"ADC"->False,"LastReadValue"->None|>,
			"D6"-><|"Direction"->Default,"LastWriteValue"->None,"PWM"->True,"ADC"->False,"LastReadValue"->None|>,
			"D7"-><|"Direction"->Default,"LastWriteValue"->None,"PWM"->False,"ADC"->False,"LastReadValue"->None|>,
			"D8"-><|"Direction"->Default,"LastWriteValue"->None,"PWM"->False,"ADC"->False,"LastReadValue"->None|>,
			"D9"-><|"Direction"->Default,"LastWriteValue"->None,"PWM"->True,"ADC"->False,"LastReadValue"->None|>,
			"D10"-><|"Direction"->Default,"LastWriteValue"->None,"PWM"->True,"ADC"->False,"LastReadValue"->None|>,
			"D11"-><|"Direction"->Default,"LastWriteValue"->None,"PWM"->True,"ADC"->False,"LastReadValue"->None|>,
			"D12"-><|"Direction"->Default,"LastWriteValue"->None,"PWM"->False,"ADC"->False,"LastReadValue"->None|>,
			"D13"-><|"Direction"->Default,"LastWriteValue"->None,"PWM"->False,"ADC"->False,"LastReadValue"->None|>,
			"A0"-><|"Direction"->Default,"LastWriteValue"->None,"PWM"->False,"ADC"->True,"LastReadValue"->None|>,
			"A1"-><|"Direction"->Default,"LastWriteValue"->None,"PWM"->False,"ADC"->True,"LastReadValue"->None|>,
			"A2"-><|"Direction"->Default,"LastWriteValue"->None,"PWM"->False,"ADC"->True,"LastReadValue"->None|>,
			"A3"-><|"Direction"->Default,"LastWriteValue"->None,"PWM"->False,"ADC"->True,"LastReadValue"->None|>,
			"A4"-><|"Direction"->Default,"LastWriteValue"->None,"PWM"->False,"ADC"->True,"LastReadValue"->None|>,
			"A5"-><|"Direction"->Default,"LastWriteValue"->None,"PWM"->False,"ADC"->True,"LastReadValue"->None|>
		|>,
		installLocation,
		install,
		input
	},
	(
		If[arduinoSoftwarePresent[],
			(*THEN*)
			(*software is present, so save the location of the main software, as well as the toolchain we need*)
			(
				$arduinoInstallLocation = getArduinoSoftwareLocation[];
				$avrgccInstallLocation=findavrGCC[$arduinoInstallLocation];
				$avrdudeInstallLocation=findavrdude[$arduinoInstallLocation];
				$avrdudeConfigFileLocation=findavrdudeConf[$arduinoInstallLocation];
			),
			(*ELSE*)
			(*it's not present, so prompt the user to install it*)
			(
				install = If[$MachineID === "4801-62204-12672",
					(*THEN*)
					(*we're on a raspberry pi, so we can't automatically install the arduino software*)
					(
						Message[DeviceOpen::raspiAutoInstall];
						False
					),
					(*ELSE*)
					(*we're not on a raspberry pi, so check normally*)
					(
						If[$FrontEnd =!= Null,
						(*THEN*)
						(*there is a front end, so display the message normally as a popup window*)
						ChoiceDialog[
							"The Arduino software package was not found on your "<>
							"computer.\n\nArduinoLink requires the Arduino software package, "<>
							"portions of which may be under separate license.\n\nBy proceeding, "<>
							"you understand and agree that Arduino is a separate software package "<>
							"with separate licensing. \n\nNote:  You may manually install the "<>
							"Arduino software and provide the install directory with the Device "<>
							"property \"ArduinoInstallLocation\".", 
							{"Install" -> True,"Do Not Install" -> False}, 
							WindowTitle -> "Arduino Software Install", WindowFloating -> True, 
							WindowSize -> {700, 225}, WindowFrame -> "ModalDialog"
						],
						(*ELSE*)
						(*there isn't a front end, so we have to print off the message, then take input as to whether or not the user accepts*)
						(
							Print["The Arduino software package was not found on your computer."<>
								"ArduinoLink requires the Arduino software package, portions of which may be under separate license."<>
								"By proceeding, you understand and agree that Arduino is a separate software package with separate licensing."<>
								" Note:  You may manually install the Arduino software and provide the install directory with the Device property"<>
								" \"ArduinoInstallLocation\".\nWould you like to install the Arduino software now?\nPlease enter True or False\n\n"];
							While[True,
								(
									(*now take the user's input as a string with InputString*)
									input = InputString[];
									If[MemberQ[{"true","yes","false","no"},ToLowerCase[input]],
										(*THEN*)
										(*input is good, we can exit*)
										Break[],
										(*ELSE*)
										(*input is invalid, continue again, but issue a message first*)
										Message[DeviceOpen::invalidInput]
									]
								)
							];
							MemberQ[{"true","yes"},ToLowerCase[input]]
						)
					]
					)
				];
				If[TrueQ[install],
					(*THEN*)
					(*user wants the software installed, so run the install subroutine*)
					(
						installLocation=installArduino[];
						If[installLocation===$Failed,
							(*THEN*)
							(*it failed to install, so return $Failed*)
							Return[$Failed],
							(*ELSE*)
							(*it didn't fail, so set the locations*)
							(
								$arduinoInstallLocation=installLocation;
								$avrgccInstallLocation=findavrGCC[installLocation];
								$avrdudeInstallLocation=findavrdude[installLocation];
								$avrdudeConfigFileLocation=findavrdudeConf[installLocation];
							)
						]
					),
					(*ELSE*)
					(*user doesn't want the software installed, so set it to none*)
					(
						$arduinoInstallLocation=None;
						$avrgccInstallLocation=None;
						$avrdudeInstallLocation=None;
						$avrdudeConfigFileLocation=None;
					)
				]
			)
		];
		$DeviceStates[ihandle]=<||>;
		$DeviceStates[ihandle,"PinConfigurations"]=defaultDeviceState;
		$DeviceStates[ihandle,"SerialPort"]=serialPort;
		$DeviceStates[ihandle,"UploadOnOpen"]=OptionValue["InitialUpload"];
		(*now set the board model*)
		If[OptionValue["BoardType"]===Default,
			(*THEN*)
			(*board is an arduino uno, set that in $DeviceStates*)
			(
				$DeviceStates[ihandle,"BoardType"]="Uno"
			),
			(*ELSE*)
			(*something else was specified, so check it*)
			(
				If[MemberQ[$SupportedBoardModels,OptionValue["BoardType"]],
					(*THEN*)
					(*the board is supported, use that one*)
					(
						$DeviceStates[ihandle,"BoardType"]=OptionValue["BoardType"];
					),
					(*ELSE*)
					(*board model is ivalid, return $Failed*)
					(
						Message[DeviceOpen::unsupportedBoard,OptionValue["BoardType"]];
						Return[$Failed];
					)
				]
			)
		];
		(*finally, before actually opening the port, issue a reset command to the device*)
		(*note that doing this now before the DeviceFramework opens up the device can save almost 5 seconds on calls to DeviceOpen on windows platforms*)
		(*if we did this in the pre configure stage, then we would have to close the serial port through the device framework*)
		(*which can take up to 5 seconds on windows*)
		(*also if we are uploading to the board in the pre-configure resetting here is pointless, so only do it if we aren't uploading*)
		If[$avrdudeInstallLocation=!=None && $arduinoInstallLocation=!=None && $avrdudeConfigFileLocation=!=None && Not[TrueQ[$DeviceStates[ihandle,"UploadOnOpen"]]],
			(*only issue a reset if we have a valid location for arduinoInstall and avrdude install*)
			Switch[$DeviceStates[ihandle,"BoardType"],
				"Uno",
				(
					arduinoReset[
						serialPort,
						$arduinoInstallLocation,
						"AVRDUDELocation"->$avrdudeInstallLocation,
						"AVRDUDEConfLocation"->$avrdudeConfigFileLocation,
						"Programmer"->Default,
						"ChipPartNumber"->Default,
						"BaudRate"->Default
					]
				),
				"Yun",
				(
					arduinoReset[
						serialPort,
						$arduinoInstallLocation,
						"AVRDUDELocation"->$avrdudeInstallLocation,
						"AVRDUDEConfLocation"->$avrdudeConfigFileLocation,
						"Programmer"->"avr109",
						"ChipPartNumber"->"atmega32u4",
						"BaudRate"->57600
					]
				)
			]
		];
		portObject=DeviceFramework`DeviceDriverOption["Firmata","OpenFunction"][Null,serialPort,"BaudRate"->115200];
		portObject
	)
];


(*TODO: re-add the pre configure function to upload the sketch if requested on initially opening the device*)
ArduinoPreConfigureDriver[dev_]:=Module[
	{
		ihandle = DeviceFramework`DeviceManagerHandle[dev],
		dhandle = DeviceFramework`DeviceHandle[dev]
	},
	(
		If[TrueQ[$DeviceStates[ihandle]["UploadOnOpen"]],
			(*THEN*)
			(*we need to upload the sketch to the device before returning to the user*)
			(
				DeviceConfigure[dev,"Upload"];
			)
			(*ELSE*)
			(*we don't need to do anything*)
		];
		All
	)
];


ArduinoCommunityLogo[{ihandle_,dhandle_},___,OptionsPattern[]]:=Module[{},
	(
		Import[PacletResource["DeviceDriver_Arduino","Logo"]]
	)
];

Options[ArduinoWriteDriver]={};
ArduinoWriteDriver[{ihandle_,dhandle_},pin_->value_,OptionsPattern[]]:=Module[
	{
		writeValue=value
	},
	(
		If[MemberQ[arduinoUnoPins,pin],
			(*THEN*)
			(*check if the pin can be written to*)
			If[okayToWrite[$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[pin]]["Direction"]],
				(*THEN*)
				(*it can be written to, check if the pin is PWM*)
				(
					If[MemberQ[arduinoUnoPWMPins,pin],
						(*THEN*)
						(*the pin is a pwm pin, so do check if the value is a compatible unit first*)
						(
							If[CompatibleUnitQ[value,"Volts"],
								(*THEN*)
								(*the value is an actual unit that can be converted to volts, so convert it and check the magnitude*)
								(
									writeValue = UnitConvert[value,"Volts"];
									If[QuantityMagnitude[writeValue]>=0 && QuantityMagnitude[writeValue] <=5,
										(*THEN*)
										(*the value is within the range, so just convert it to the bits*)
										(
											arduinoAnalogWrite[{ihandle,dhandle},pin,Floor[QuantityMagnitude[writeValue]/5*255]];
										),
										(*ELSE*)
										(*the value is not within the range, so raise a message and normalize it*)
										(
											Message[DeviceWrite::voltMagnitude,value,
												writeValue=Quantity[5*booleanize[QuantityMagnitude[writeValue]/5],"Volts"]];
											arduinoDigitalWrite[{ihandle,dhandle},pin,Floor[QuantityMagnitude[writeValue]/5]];
										)
									]
								),
								(*ELSE*)
								(*the value is not a unit, so check if it is numeric*)
								(
									If[NumericQ[value],
										(*THEN*)
										(*the value can be normalized, so check if it is between 0 and 1*)
										(
											If[value >0 && value <1,
												(*THEN*)
												(*the value is already within the correct range, so convert it, then write it*)
												(
													arduinoAnalogWrite[{ihandle,dhandle},pin,Floor[value*255]];
												),
												(*ELSE*)
												(*check if the value is equal to 1 or 0*)
												If[value ==0 || value ==1,
													(*THEN*)
													(*the value is boolean, so use arduinoDigitalWrite*)
													(
														arduinoDigitalWrite[{ihandle,dhandle},pin,Round[value]];
													),
													(*ELSE*)
													(*the value is not within the correct range, so raise a message and normalize it*)
													(
														Message[DeviceWrite::pwmresolution,value,writeValue=pwmize[255*writeValue]/255];
														arduinoDigitalWrite[{ihandle,dhandle},pin,writeValue];
													)
												]
											]
										),
										(*ELSE*)
										(*the value cannot be normalized, raise a message and return $Failed*)
										(
											Message[DeviceWrite::numericValue,value];
											Return[$Failed];
										)
									]
								)
							];
						),
						(*ELSE*)
						(*the pin is a normal digital out pin, so check if the value is 1 or 0*)
						(
							If[value===1||value===0,
								(*THEN*)
								(*the pin is good to write to*)
								(
									arduinoDigitalWrite[{ihandle,dhandle},pin,value];
								),
								(*ELSE*)
								(*the pin is not good to write to, so see if the value can be normalized*)
								If[NumericQ[value],
									(*THEN*)
									(*the value is numeric, and it can be normalized*)
									(
										Message[DeviceWrite::nonBooleanWrite,value,booleanize[value]];
										arduinoDigitalWrite[{ihandle,dhandle},numericalPin[pin],writeValue=booleanize[value]];
									),
									(*ELSE*)
									(*the value is not numeric and can't be normalized, so raise message and return $Failed*)
									(
										Message[DeviceWrite::numericValue,value];
										Return[$Failed];
									)
								]
							]
						)
					];
					(*finally update the configuration association with the write value and the direction*)
					$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[pin]]["LastWriteValue"]=(DateObject[]->writeValue);
					If[$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[pin]]["Direction"]===Default||$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[pin]]["Direction"]==="SoftInput",
						(*THEN*)
						(*the pin hasn't been configured or written to yet, or it was a soft input previously, and we just wrote to it,*)
						(*so change it to "SoftOutput"*)
						$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[pin]]["Direction"]="SoftOutput";
						(*ELSE*)
						(*to have written to it and have it not be default or "SoftInput", it must have been "HardOutput", so don't change it*)
					]
				),
				(*ELSE*)
				(*the pin has already been configured as a HardInput, so we can't write to it*)
				(
					Message[DeviceWrite::config,pin];
					Return[$Failed];
				)
			],
			(*ELSE*)
			(*the pin is not a valid arduino uno pin, check if it is a serial communication pin and raise a seperate message for that*)
			(
				If[pin===0||pin===1,
					(*THEN*)
					(*the pins are reserved for serial, raise a special message for that*)
					(
						Message[DeviceWrite::serialPin];
						Return[$Failed];
					),
					(*ELSE*)
					(*the pin just isn't a pin on the arduino uno at all*)
					(
						Message[DeviceWrite::invalidPin,pin];
						Return[$Failed];
					)
				]
				
			)
		]
	)
]


(*the following are for association or lists of rules*)
ArduinoWriteDriver[{ihandle_,dhandle_},pins_List,OptionsPattern[]]:=Module[{},
	(
		ArduinoWriteDriver[{ihandle,dhandle},#]&/@pins
	)
];

ArduinoWriteDriver[{ihandle_,dhandle_},pins_Association,OptionsPattern[]]:=Module[{},
	(
		ArduinoWriteDriver[{ihandle,dhandle},#]&/@Normal[pins]
	)
];


(*note this function does not do any checking of the pin or value, so that must be done previously*)
arduinoAnalogWrite[{ihandle_,dhandle_},pin_,value_]:=Module[{},
	(
		checkBootFunctionMessage[ihandle];
		DeviceFramework`DeviceDriverOption["Firmata","WriteFunction"][{ihandle,dhandle},{pin,value},"WriteMode"->"Analog"]
	)
];


arduinoDigitalWrite[{ihandle_,dhandle_},pin_,value_]:=Module[
	{
		valueBitMask=Table[0,{8}],
		port=pinToPort[pin]
	},
	(
		(*any pin can be a digital output, so we don't have to do any pin checking*)
		Switch[port,
			0, (*PORT 0 - digital pins 0-7*)
				(
					valueBitMask=(
						If[okayToWrite[$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[#]]["Direction"]],
							(*THEN*)
							(*the pin is okay to write to*)
							If[$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[#]]["LastWriteValue"]===None,
								(*THEN*)
								(*the pin has either never been written to before, so use None as the value*)
								None,
								(*ELSE*)
								(*it has been written to before, write whatever value was last written there as the value*)
								$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[#]]["LastWriteValue"][[2]]
							],
							(*ELSE*)
							(*the pin isn't okay to write to, so use a zero*)
							0
						]
					)&/@{2,3,4,5,6,7};
					valueBitMask=Flatten[Prepend[valueBitMask,{0,0}]];
				),
			1,(*PORT 1 - digital pins 8-13*)
				(
					valueBitMask=(
						If[okayToWrite[$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[#]]["Direction"]],
							(*THEN*)
							(*the pin is okay to write to*)
							If[$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[#]]["LastWriteValue"]===None,
								(*THEN*)
								(*the pin has either never been written to before, so use None as the value*)
								None,
								(*ELSE*)
								(*it has been written to before, write whatever value was last written there as the value*)
								$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[#]]["LastWriteValue"][[2]]
							],
							(*ELSE*)
							(*the pin isn't okay to write to, so use a zero*)
							0
						]
					)&/@{8,9,10,11,12,13};
					valueBitMask=Flatten[Append[valueBitMask,{0,0}]];
				),
			2, (*PORT 2 - analog pins*)
				(
					valueBitMask=(
						If[okayToWrite[$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[#]]["Direction"]],
							(*THEN*)
							(*the pin is okay to write to*)
							If[$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[#]]["LastWriteValue"]===None,
								(*THEN*)
								(*the pin has either never been written to before, so use None as the value*)
								None,
								(*ELSE*)
								(*it has been written to before, write whatever value was last written there as the value*)
								$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[#]]["LastWriteValue"][[2]]
							],
							(*ELSE*)
							(*the pin isn't okay to write to, so use a zero*)
							0
						]
					)&/@{14,15,16,17,18,19};
					valueBitMask=Flatten[Append[valueBitMask,{0,0}]];
				),
			_,(*all other ports, return $Failed*)
			Return[$Failed]
		];
		(*None is for pins that have never been written to, so put those as 0*)
		valueBitMask = valueBitMask/.None->0;
		(*now set the pin we want to configure to have the value the user requested*)
		If[Head[pin]=!=String&&Not[MemberQ[Range[14,19],pin]],
			(*THEN*)
			(*the pin number is a normal integer, use that*)
			valueBitMask=ReplacePart[valueBitMask,(Mod[pin,8]+1)->value],
			(*ELSE*)
			(*the pin number is a string, so convert it and use that instead*)
			valueBitMask=ReplacePart[valueBitMask,Mod[analogNumericPin[pin]+1,8]->value]
		];
		(*if any of the pins previously had values that weren't 1 or 0, make those zero*)
		valueBitMask = If[#===0||#===1,#,0]&/@valueBitMask;
		(*reverse the bit mask and make it into a binary number*)
		finalPortValue = FromDigits[Reverse[valueBitMask],2];
		checkBootFunctionMessage[ihandle];
		DeviceFramework`DeviceDriverOption["Firmata","WriteFunction"][{ihandle,dhandle},{port,finalPortValue},
			"PinAddressing"->"Port",
			"WriteMode"->"Digital",
			(*if the pin we are writing to is a pwm pin, then we are digitally writing to a pwm pin,*)
			(* so we need to enforce disabling of the pwm timer on that pin with a hidden bit in the firmata packet*)
			If[MemberQ[arduinoUnoPWMPins,pin],
				(*THEN*)
				(*then it's a member, which one*)
				Which[
					(*pack the first hidden bit if it is pin 3 or pin 9*)
					pin === 3 || pin === 9,"HiddenBits"->FromDigits["1",2],
					(*pack the second hidden bit if it is pin 5 or pin 10*)
					pin === 5 || pin === 10,"HiddenBits"->FromDigits["10",2],
					(*pack the third hidden bit if it is pin 6 or pin 11*)
					pin === 6 || pin === 11, "HiddenBits"->FromDigits["100",2],
					(*default case of none*)
					True, "HiddenBits"->None
				],
				(*ELSE*)
				(*it's not a member so set this option to None*)
				"HiddenBits"->None
			]
		];
	)
];


Options[ArduinoReadDriver]=
{
	"ReadMode"->Automatic,
	"ReturnFunction"->Automatic
};
(*this is for debugging, it will read the entire buffer for the serial port*)
ArduinoReadDriver[{ihandle_,dhandle_},"Raw",OptionsPattern[]]:=Module[{},
	(
		DeviceFramework`DeviceDriverOption["Firmata","ReadFunction"][{ihandle,dhandle},"raw"]
	)
];


(*with no arguments, all pins are read*)
ArduinoReadDriver[{ihandle_,dhandle_},OptionsPattern[]]:=Module[{},
	(
		If[OptionValue["ReadMode"]==="Analog",
			(*THEN*)
			(*the user specified an analog read of Arduino, so just return the analog pins*)
			(
				Return[
					ArduinoReadDriver[
						{ihandle,dhandle},
						(*this gets a list of all valid arduino pins (but not duplicates)*)
						{"A0","A1","A2","A3","A4","A5"},
						Sequence[#->OptionValue[#]&/@Options[ArduinoReadDriver][[All,1]]]
					]
				]
			),
			(*ELSE*)
			(*didn't specify analog read mode, so we can safely read from all pins*)
			(
				Return[
					ArduinoReadDriver[
						{ihandle,dhandle},
						(*this gets a list of all valid arduino pins (but not duplicates)*)
						Join[Complement[arduinoUnoPins,arduinoUnoAnalogPins],{"A0","A1","A2","A3","A4","A5"}],
						Sequence[#->OptionValue[#]&/@Options[ArduinoReadDriver][[All,1]]]
					]
				]
			)
		]
		
	)
]

(*normal case that handles : DeviceRead["Arduino",3]*)
ArduinoReadDriver[{ihandle_,dhandle_},pin_,OptionsPattern[]]:=Module[{},
	(
		(*first check if the pin exists at all*)
		If[MemberQ[arduinoUnoPins,pin],
			(*THEN*)
			(*it exists, now check if we can read from it*)
			If[okayToRead[$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[pin]]["Direction"]],
				(*THEN*)
				(*we are good to read*)
				(
					checkBootFunctionMessage[ihandle];
					value=DeviceFramework`DeviceDriverOption["Firmata","ReadFunction"][{ihandle,dhandle},pin,
						"ReadMode"->OptionValue["ReadMode"],
						(*this check is if we are reading from a pwm pin, then we need to specify whether to the *)
						(*arduino to turn off PWM on that pin, else the PWM timer will override the read and will just return 0*)
						If[MemberQ[arduinoUnoPWMPins,pin],
							(*THEN*)
							(*then it's a member, which one*)
							Which[
								(*pack the first hidden bit if it is pin 3 or pin 9*)
								pin === 3 || pin === 9,"HiddenBits"->FromDigits["1",2],
								(*pack the second hidden bit if it is pin 5 or pin 10*)
								pin === 5 || pin === 10,"HiddenBits"->FromDigits["10",2],
								(*pack the third hidden bit if it is pin 6 or pin 11*)
								pin === 6 || pin === 11, "HiddenBits"->FromDigits["100",2],
								(*default case of none*)
								True, "HiddenBits"->None
							],
							(*ELSE*)
							(*it's not a member so set this option to None*)
							"HiddenBits"->None
						]
						];
					(*if value is $Failed, don't do anything else and just return $Failed*)
					If[value===$Failed,
						Return[$Failed]
					]
				),
				(*ELSE*)
				(*user hard configured this pin to be an output, so we can't read from it*)
				(
					Message[DeviceRead::config,pin];
					Return[$Failed];
				)
			];
			Switch[OptionValue["ReturnFunction"],
				Automatic,
				(
					(*for automatic return function, this will depend on the type of pin*)
					Switch[OptionValue["ReadMode"],
						Automatic,
						(
							(*the behavior for automatic read mode is that if the pin is an analog pin, an analog read was performed, if it is digital,
							we can just return the value*)
							If[MemberQ[arduinoUnoAnalogPins,pin],
								(*then the pin is an analog pin, and we should convert it to a voltage*)
								(
									$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[pin]]["LastReadValue"]=DateObject[]->Quantity[N[5*value/1023,6],"Volts"];
									Return[Quantity[N[5*value/1023,6],"Volts"]]
								),
								(*ELSE*)
								(*the pin isn't an analog pin, so just return the pin*)
								(
									$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[pin]]["LastReadValue"]=DateObject[]->value;
									Return[value];
								)
							]
						),
						"Analog",
						(
							(*the pin must have been an analog pin, or else we are going to convert $Failed*)
							$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[pin]]["LastReadValue"]=DateObject[]->Quantity[N[5*value/1023,6],"Volts"];
							Return[Quantity[N[5*value/1023,6],"Volts"]];
						),
						"Digital",
						(
							(*just return normal value*)
							$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[pin]]["LastReadValue"]=DateObject[]->value;
							Return[value];
						),
						_,
						(
							(*anything else, just return the value normally*)
							$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[pin]]["LastReadValue"]=DateObject[]->value;
							Return[value];
						)
					]
				),
				_,
				(
					(*the user specified the function, so try applying their function to it*)
					$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[pin]]["LastReadValue"]=DateObject[]->OptionValue["ReturnFunction"][value];
					Return[OptionValue["ReturnFunction"][value]]
				)
			],
			(*ELSE*)
			(*the pin doesn't exist, issue a message and return $Failed*)
			(
				Message[DeviceRead::invalidPin,pin];
				Return[$Failed];
			)
		]
	)
];

(*version that handles the case of DeviceRead["Arduino",{{2,3,4}}]*)
ArduinoReadDriver[{ihandle_,dhandle_},pins_List,OptionsPattern[]]:=Module[{},
	(
		Association[
			Function[pin,
				(pin->ArduinoReadDriver[{ihandle,dhandle},pin,"ReadMode"->OptionValue["ReadMode"],"ReturnFunction"->OptionValue["ReturnFunction"]])]/@pins]
	)
];


(*version that handles the case of DeviceRead["Arduino",{2,3,4}]*)
ArduinoReadDriver[{ihandle_,dhandle_},pins__?(MemberQ[arduinoUnoPins,#]&)..,OptionsPattern[]]:=Module[{},
	(
		Association[
			Function[pin,
				(pin->ArduinoReadDriver[{ihandle,dhandle},pin,"ReadMode"->OptionValue["ReadMode"],"ReturnFunction"->OptionValue["ReturnFunction"]])]/@{pins}]
	)
];


(*catch all case for anything else not matched to return $Failed and issue message*)
ArduinoReadDriver[{ihandle_,dhandle},args___]:=(
	Message[DeviceRead::invalidArgs,{args}];
	$Failed
)



(*this is to check whether or not there are any scheduled tasks running on the arduino at any given time*)
$scheduledTaskRunning=<|"Running"->False,"startTime"->AbsoluteTime[],"endTime"->AbsoluteTime[]|>;


(*the execute driver will basically just use the internal association for functions currently uploaded to 
the arduino to build the packet to send to Firmata, noting to Firmata if there are return packets expected,
if there are those will be interpreted and sent back to the user*)
Options[ArduinoExecute]={};
ArduinoExecute[{ihandle_,dhandle_},"DeleteTask",functionName_,OptionsPattern[]]:=Module[{},
	(
		(*first confirm that the function requested actually exists*)
		If[MemberQ[Keys[$functionCalls],functionName],
			(*THEN*)
			(*the function exists, so send the delete request*)
			(
				(*TODO: in the sketch, change the delete task specified from 1 back to 6*)
				packet={FromDigits["F0",16],FromDigits["01",16],$functionCalls[functionName][[1,3]],FromDigits["f7",16]};
				checkBootFunctionMessage[ihandle];
				DeviceFramework`DeviceDriverOption["Firmata","ExecuteFunction"][{ihandle,dhandle},packet];
				(*lastly, update the scheduledTask association to not be running*)
				$scheduledTaskRunning["Running"]=False;
				$scheduledTaskRunning["startTime"]=AbsoluteTime[];
				$scheduledTaskRunning["endTime"]=AbsoluteTime[];
			),
			(*ELSE*)
			(*the function doesn't exist, so issue a message and return $Failed*)
			(
				Message[DeviceExecute::funcName,functionName];
				Return[$Failed];
			)
		]
	)
];


(*Software reset uses AVRDUDE to issue a null command to the device, effectively resetting the device*)
ArduinoExecute[{ihandle_,dhandle_},"SoftwareReset",OptionsPattern[]]:=Module[{},
	(*reset the arduino before returning it to the user*)
	(*before we reset the device, we have to close the serial port so it can be accessed by AVRDUDE*)
	DeviceFramework`DeviceDriverOption["Firmata","CloseFunction"][{ihandle,dhandle}];
	Switch[$DeviceStates[ihandle,"BoardType"],
		"Uno",
		(
			arduinoReset[
				$DeviceStates[ihandle,"SerialPort"],
				$arduinoInstallLocation,
				"AVRDUDELocation"->$avrdudeInstallLocation,
				"AVRDUDEConfLocation"->$avrdudeConfigFileLocation,
				"Programmer"->Default,
				"ChipPartNumber"->Default,
				"BaudRate"->Default
			]
		),
		"Yun",
		(
			arduinoReset[
				$DeviceStates[ihandle,"SerialPort"],
				$arduinoInstallLocation,
				"AVRDUDELocation"->$avrdudeInstallLocation,
				"AVRDUDEConfLocation"->$avrdudeConfigFileLocation,
				"Programmer"->"avr109",
				"ChipPartNumber"->"atmega32u4",
				"BaudRate"->57600
			]
		)
	];
	(*now we re-open the serial port*)
	DeviceFramework`DeviceDriverOption["Firmata","OpenFunction"][{ihandle,dhandle},$DeviceStates[ihandle]["SerialPort"],"BaudRate"->115200];
	
]


ArduinoScheduleExecute[{ihandle_,dhandle_},functionTask_,OptionsPattern[]]:=Module[
	{
		functionName=functionTask[[1]],
		args=functionTask[[2]],
		timespec=functionTask[[3]],
		funcID=Quiet[$functionCalls[functionTask[[1]]][[1,3]]],
		function=Quiet[$functionCalls[functionTask[[1]]][[2]]]
	},
	(
		(*first confirm that the function requested actually exists*)
		If[MemberQ[Keys[$functionCalls],functionName],
			(*THEN*)
			(*the function does exist*)
			(*TODO: implement the grabber functionality with a ScheduledTask on the Mathematica side for the return value of arduino scheduled tasks*)
			(*TODO: expand timespec to all cases the arduino supports*)
			(*right now it just ignores it and doesn't send them if it doesn't have to*)
			(*make sure that a scheduled task isn't already running*)
			If[TrueQ[$scheduledTaskRunning["Running"]],
				(*THEN*)
				(*one was running, see if it expired*)
				(
					If[AbsoluteTime[] >= $scheduledTaskRunning["endTime"],
						(*THEN*)
						(*it expired, so we are good to go*)
						(
							$scheduledTaskRunning["Running"]=False;
							$scheduledTaskRunning["endTime"]=AbsoluteTime[];
							$scheduledTaskRunning["startTime"]=AbsoluteTime[];
						),
						(*ELSE*)
						(*it hasn't expired so return $Failed and raise a message*)
						(
							Message[DeviceExecute::taskRunning];
							Return[$Failed]
						)
					]
				)
				(*ELSE*)
				(*no else case, just allow to continue normally*)
			];
			Switch[timespec,
				_Integer|_Real,
				(
					(*this is for the case of ScheduledTask[expr, syncTime], where the task is just
					run every syncTime seconds infinitely*)
					(*a run time length and iteration count of 0 represents an infinite task to the arduino*)
					(*also, the time specified by the user is in seconds, but the arduino needs in milliseconds, so multiply the seconds by 1000*)
					functionPacket=functionCallPacketSend[funcID,
						"SyncTime"->Floor[timespec*1000],"RunTimeLength"->0,"IterationCount"->0,
						"LongArgumentNumber"->longArgsNum[function],
						"FloatArgumentNumber"->floatArgsNum[function],
						"StringArgumentNumber"->stringArgsNum[function],
						"LongArrayArgumentNumber"->longArrayArgsNum[function],
						"FloatArrayArgumentNumber"->floatArrayArgsNum[function]];
					(*check if there are any arguments to send*)
					If[hasArgs[function],
						(*THEN*)
						(*the function has args, so send those too*)
						(
							(*now check to make sure that the arguments sent are good to go*)
							argPackets = sendArgs[$functionCalls[functionName][[2]],args];
							If[ argPackets =!= $Failed,
								(*THEN*)
								(*the arguments are valid, we can schedule the function now*)
								(
									checkBootFunctionMessage[ihandle];
									(*always default to not waiting for a return value*)
									DeviceFramework`DeviceDriverOption["Firmata","ExecuteFunction"][{ihandle,dhandle},
										Flatten[Join[functionPacket,argPackets]],
										"ReturnValue"->False];
									(*a function is now running on the arduino, so don't let the user run anymore until this one is done running or deleted*)
									$scheduledTaskRunning["Running"]=True;
									$scheduledTaskRunning["startTime"]=AbsoluteTime[];
									(*this is an infinite task*)
									$scheduledTaskRunning["endTime"]=Infinity
								),
								(*ELSE*)
								(*the arguments are invalid, so raise message and return $Failed*)
								(
									Message[DeviceExecute::invalidArgs,args];
									Return[$Failed];
								)
							]
						),
						(*ELSE*)
						(*the function doesn't have args, so don't send those, but still check if the user specified arguments*)
						(
							If[args=!={},
								(*THEN*)
								(*the user specified arguments, when this function doesn't take arguments*)
								(
									Message[DeviceExecute::invalidArgs,args];
									Return[$Failed];
								),
								(*ELSE*)
								(*the user didn't specify arguments, so run normally*)
								(
									checkBootFunctionMessage[ihandle];
									(*always default to not waiting for a return value*)
									DeviceFramework`DeviceDriverOption["Firmata","ExecuteFunction"][{ihandle,dhandle},functionPacket,"ReturnValue"->False];
									(*a function is now running on the arduino, so don't let the user run anymore until this one is done running or deleted*)
									$scheduledTaskRunning["Running"]=True;
									$scheduledTaskRunning["startTime"]=AbsoluteTime[];
									(*this is an infinite task*)
									$scheduledTaskRunning["endTime"]=Infinity
								)
							];
							
						)
					];
					
				),
				{_Integer|_Real,_Integer},
				(
					(*this is for the case of ScheduledTask[expr,{syncTime, count}], where the task is
					run every syncTime seconds for a total number of count times*)
					functionPacket=functionCallPacketSend[funcID,
						"SyncTime"->Floor[timespec[[1]]*1000],"RunTimeLength"->0,"IterationCount"->timespec[[2]],
						"LongArgumentNumber"->longArgsNum[function],
						"FloatArgumentNumber"->floatArgsNum[function],
						"StringArgumentNumber"->stringArgsNum[function],
						"LongArrayArgumentNumber"->longArrayArgsNum[function],
						"FloatArrayArgumentNumber"->floatArrayArgsNum[function]];
					(*now send the packets to the arduino*)
					(*first check if there are any arguments to send*)
					If[hasArgs[function],
						(*THEN*)
						(*the function has args, so send those too*)
						(
							(*now let's check the arguments the user passed and see if those are good to go*)
							argPackets = sendArgs[$functionCalls[functionName][[2]],args];
							(*always default to not waiting for a return value*)
							If[argPackets =!= $Failed,
								(*THEN*)
								(*the arguments are valid, so send the task along*)
								(
									checkBootFunctionMessage[ihandle];
									DeviceFramework`DeviceDriverOption["Firmata","ExecuteFunction"][{ihandle,dhandle},
										Flatten[Join[functionPacket,argPackets]],
										"ReturnValue"->False];
									(*a function is now running on the arduino, so don't let the user run anymore until this one is done running or deleted*)
									$scheduledTaskRunning["Running"]=True;
									$scheduledTaskRunning["startTime"]=AbsoluteTime[];
									(*this isn't an infinite task, so set the end time to be count * waitTime + 1*)
									$scheduledTaskRunning["endTime"]=AbsoluteTime[]+1+timespec[[1]]*timespec[[2]];
								),
								(*ELSE*)
								(*the arguments are invalid, so raise message and return $Failed*)
								(
									Message[DeviceExecute::invalidArgs,args];
									Return[$Failed];
								)
							]
						),
						(*ELSE*)
						(*the function doesn't have arguments, so don't send those*)
						(
							If[args=!={},
								(*THEN*)
								(*the user specified arguments, when this function doesn't take arguments*)
								(
									Message[DeviceExecute::invalidArgs,args];
									Return[$Failed];
								),
								(*ELSE*)
								(*the user didn't specify arguments, so run normally*)
								(
									checkBootFunctionMessage[ihandle];
									DeviceFramework`DeviceDriverOption["Firmata","ExecuteFunction"][{ihandle,dhandle},functionPacket,"ReturnValue"->False];
									(*a function is now running on the arduino, so don't lwt the user run anymore until this one is done running or deleted*)
									$scheduledTaskRunning["Running"]=True;
									$scheduledTaskRunning["startTime"]=AbsoluteTime[];
									(*this isn't an infinite task, so set the end time to be count * waitTime + 1*)
									$scheduledTaskRunning["endTime"]=AbsoluteTime[]+1+timespec[[1]]*timespec[[2]];
								)
							]
						)
					];
				),
				{_Integer|_Real},
				(
					(*this is for the case of ScheduledTask[expr, {delayTime}], where the task is to be 
					run in delayTime seconds*)
					functionPacket=functionCallPacketSend[funcID,"InitialDelayTime"->Floor[First[timespec]*1000],
						"LongArgumentNumber"->longArgsNum[function],
						"FloatArgumentNumber"->floatArgsNum[function],
						"StringArgumentNumber"->stringArgsNum[function],
						"LongArrayArgumentNumber"->longArrayArgsNum[function],
						"FloatArrayArgumentNumber"->floatArrayArgsNum[function]];
					(*now send the packets to the arduino*)
					(*first check if there are args to send as well*)
					If[hasArgs[function],
						(*THEN*)
						(*the function has args, so send those too*)
						(
							(*now validate the arguments before sending them*)
							argPackets = sendArgs[$functionCalls[functionName][[2]],args];
							If[argPackets =!= $Failed,
								(*THEN*)
								(*the arguments are valid, so schedule the task*)
								(
									checkBootFunctionMessage[ihandle];
									(*always default to not waiting for a return value*)
									DeviceFramework`DeviceDriverOption["Firmata","ExecuteFunction"][{ihandle,dhandle},
										Flatten[Join[functionPacket,argPackets]],
										"ReturnValue"->False];
									(*a function is now running on the arduino, so update the association for that*)
									$scheduledTaskRunning["Running"]=True;
									$scheduledTaskRunning["startTime"]=AbsoluteTime[];
									(*this task will be run in timespec seconds, so set the end time to be that*)
									$scheduledTaskRunning["endTime"]=First[AbsoluteTime[]+timespec+1];
								),
								(*ELSE*)
								(*the arguments are invalid, so raise a message and return $Failed*)
								(
									Message[DeviceExecute::invalidArgs,args];
									Return[$Failed];
								)
							]
						),
						(*ELSE*)
						(*the function doesn't have arguments, so don't send those*)
						(
							If[args=!={},
								(*THEN*)
								(*the user specified arguments, when this function doesn't take arguments*)
								(
									Message[DeviceExecute::invalidArgs,args];
									Return[$Failed];
								),
								(*ELSE*)
								(*the user didn't specify arguments, so run normally*)
								(
									checkBootFunctionMessage[ihandle];
									DeviceFramework`DeviceDriverOption["Firmata","ExecuteFunction"][{ihandle,dhandle},functionPacket,"ReturnValue"->False];
									(*a function is now running on the arduino, so update the association for that*)
									$scheduledTaskRunning["Running"]=True;
									$scheduledTaskRunning["startTime"]=AbsoluteTime[];
									(*this task will be run in timespec seconds, so set the end time to be that*)
									$scheduledTaskRunning["endTime"]=First[AbsoluteTime[]+timespec+1];
								)
							]
						)
					];
				),(*
				_String,
				(
					(*TODO: implement this case for Hourly, Monthly, etc.*)
					(*TODO: implement this case for cron tab spec*)
					Null
				),*)
				_DateObject,
				(
					(*this case is for running once at the time specified by the DateObject*)
					(*for this case, we just calculate the amount of seconds between now and when the date object is for*)
					(*if it is positive (or less than a second ago), throw it in, if it is not, then issue a message*)
					If[(waitTime=AbsoluteTime[timespec]-AbsoluteTime[Now])>-1,
						(*THEN*)
						(*it is in fact in the future, so run it normally*)
						(
							functionPacket=functionCallPacketSend[funcID,"InitialDelayTime"->waitTime,
								"LongArgumentNumber"->longArgsNum[function],
								"FloatArgumentNumber"->floatArgsNum[function],
								"StringArgumentNumber"->stringArgsNum[function],
								"LongArrayArgumentNumber"->longArrayArgsNum[function],
								"FloatArrayArgumentNumber"->floatArrayArgsNum[function]];
							(*Print["sending ",functionPacket];*)
							(*now send the packets to the arduino*)
							(*first check if there are arguments to send*)
							If[hasArgs[function],
								(*THEN*)
								(*the function has args, so send those too*)
								(
									(*then check the arguments the user passed to see if they are good*)
									argPackets= sendArgs[$functionCalls[functionName][[2]],args];
									If[ argPacket =!= $Failed,
										(*THEN*)
										(*the arguments are good, so send it along*)
										(
											checkBootFunctionMessage[ihandle];
											(*always default to not waiting for a return value*)
											DeviceFramework`DeviceDriverOption["Firmata","ExecuteFunction"][{ihandle,dhandle},
												Flatten[Join[functionPacket,argPackets]],
												"ReturnValue"->False];
												(*update the scheduledTask association because a function is now running on the arduino*)
												$scheduledTaskRunning["Running"]=True;
												$scheduledTaskRunning["startTime"]=AbsoluteTime[];
												$scheduledTaskRunning["endTime"]=AbsoluteTime[]+waitTime+1;
										),
										(*ELSE*)
										(*arguments are invalid, so raise a message and return $Failed*)
										(
											Message[DeviceExecute::invalidArgs,args];
											Return[$Failed];
										)
									];
								),
								(*ELSE*)
								(*the function doesn't have arguments, so don't send those*)
								(
									If[args=!={},
										(*THEN*)
										(*the user specified arguments, when this function doesn't take arguments*)
										(
											Message[DeviceExecute::invalidArgs,args];
											Return[$Failed];
										),
										(*ELSE*)
										(*the user didn't specify arguments, so run normally*)
										(
											checkBootFunctionMessage[ihandle];
											DeviceFramework`DeviceDriverOption["Firmata","ExecuteFunction"][{ihandle,dhandle},functionPacket,"ReturnValue"->False];
											(*update the scheduledTask association because a function is now running on the arduino*)
											$scheduledTaskRunning["Running"]=True;
											$scheduledTaskRunning["startTime"]=AbsoluteTime[];
											$scheduledTaskRunning["endTime"]=AbsoluteTime[]+waitTime+1;
										)
									]
								)
							];
						),
						(*ELSE*)
						(*not in the future, so not much we can do here*)
						(
							Message[DeviceExecute::past,waitTime];
							Return[$Failed];
						)
					]
				),
				_,
				(
					(*any other kind of timespec should be invalid, so issue message and return $Failed*)
					Message[DeviceExecute::invalidTiming,timespec];
					Return[$Failed];
				)
			],
			(*ELSE*)
			(*the function doesn't exist, issue message and reutrn $Failed*)
			(
				Message[DeviceExecute::noFunc,functionName];
				Return[$Failed];
			)
		]
	)
];


ArduinoExecute[{ihandle_,dhandle_},functionName_,args__,OptionsPattern[]]:=Module[{},
	(
		(*before execution, make sure the requested function exists*)
		If[MemberQ[Keys[$functionCalls],functionName],
			(*THEN*)
			(*function exists, so make sure a function isn't already running on the arduino*)
			(
				(*make sure that a scheduled task isn't already running*)
				If[TrueQ[$scheduledTaskRunning["Running"]],
					(*THEN*)
					(*one was running, see if it expired*)
					(
						If[AbsoluteTime[] >= $scheduledTaskRunning["endTime"],
							(*THEN*)
							(*it expired, so we are good to go*)
							(
								$scheduledTaskRunning["Running"]=False;
								$scheduledTaskRunning["endTime"]=AbsoluteTime[];
								$scheduledTaskRunning["startTime"]=AbsoluteTime[];
							),
							(*ELSE*)
							(*it hasn't expired so return $Failed and raise a message*)
							(
								Message[DeviceExecute::taskRunning];
								Return[$Failed]
							)
						]
					)
					(*ELSE*)
					(*no else case, just allow to continue normally*)
				];
				(*get the arg packets, and check to make sure that didn't fail*)
				argPackets = sendArgs[$functionCalls[functionName][[2]],args];
				If[argPackets =!= $Failed,
					(*THEN*)
					(*it worked, argPackets didn't fail*)
					(
						checkBootFunctionMessage[ihandle];
						Return[DeviceFramework`DeviceDriverOption["Firmata","ExecuteFunction"]
							[{ihandle,dhandle},
								Flatten[Join[$functionCalls[functionName][[1]],argPackets]],
								"ReturnValue"->ReturnQ[$functionCalls[functionName][[2]]]]]
					),
					(*ELSE*)
					(*it failed, so raise a message about the arguments*)
					(
						Message[DeviceExecute::invalidArgs,args];
						Return[$Failed];
					)
				]
			),
			(*ELSE*)
			(*the function doesn't exist, issue message and return $Failed*)
			(
				Message[DeviceExecute::noFunc,functionName];
				Return[$Failed];
			)
		]
	)
];





ArduinoExecute[{ihandle_,dhandle_},functionName_,OptionsPattern[]]:=Module[{},
	(
		(*before execution, make sure that the requested function exists*)
		If[MemberQ[Keys[$functionCalls],functionName],
			(*THEN*)
			(*the function exists, so make sure that one isn't already running before running a new one*)
			(
				If[TrueQ[$scheduledTaskRunning["Running"]],
					(*THEN*)
					(*one was running, see if it expired*)
					(
						If[AbsoluteTime[] >= $scheduledTaskRunning["endTime"],
							(*THEN*)
							(*it expired, so we are good to go, but need to reset the $scheduledTaskRunning*)
							(
								$scheduledTaskRunning["Running"]=False;
								$scheduledTaskRunning["endTime"]=AbsoluteTime[];
								$scheduledTaskRunning["startTime"]=AbsoluteTime[];
							),
							(*ELSE*)
							(*it hasn't expired so return $Failed and raise a message*)
							(
								Message[DeviceExecute::taskRunning];
								Return[$Failed]
							)
						]
					)
					(*ELSE*)
					(*no else case, just allow to continue normally*)
				];
				(*function exists, so make sure it doesn't need args, then call it*)
				(*TracePrint[hasArgs[$functionCalls[functionName][[2]]]];*)
				If[hasArgs[$functionCalls[functionName][[2]]],
					(*THEN*)
					(*it needs arguments, and we didn't get any, so raise message and return $Failed*)
					(
						Message[DeviceExecute::needsArgs,functionName];
						Return[$Failed];
					),
					(*ELSE*)
					(*it doesn't need arguments, and we didn't get any, so don't send any*)
					(
						checkBootFunctionMessage[ihandle];
						Return[DeviceFramework`DeviceDriverOption["Firmata","ExecuteFunction"]
							[{ihandle,dhandle},$functionCalls[functionName][[1]],
								"ReturnValue"->ReturnQ[$functionCalls[functionName][[2]]]]]
					)
				]
			),
			(*ELSE*)
			(*the function doesn't exist, issue message and return $Failed*)
			(
				Message[DeviceExecute::noFunc,functionName];
				Return[$Failed];
			)
		]
	)
];


(*this is basically a wrapper function that converts the user's raw input from DeviceExecute into the individual arguments necessary for it*)
(*possible forms of DeviceExecute supported:*)
(*DeviceExecute["Arduino","DeleteTask","func"]*)
(*DeviceExecute["Arduino","func"]*)
(*DeviceExecute["Arduino","func",args]*)
(*DeviceExecute["Arduino","func",{args,"Scheduling"->timespec}]*)
(*DeviceExecute["Arduino","func","Scheduling"->timespec]*)
(*DeviceExecute["Arduino",DatabinAdd,"A0"]*)
(*DeviceExecute["Arduino",DatabinAdd,{"A0","Key"->"MyKey","Databin"->binID}]*)
(*DeviceExecute["Arduino",ChannelSend,"A0"]*)
(*DeviceExecute["Arduino",ChannelSend,{"A0","ID"->"myCustomKeyValue","Channel"->channelSpec}]*)
(*DeviceExecute["Arduino",ChannelSend,{"A0","A5",4}]*)
Options[ArduinoExecuteDriver]=
	{
		"Scheduling"->None,
		(*key specifies the name of the key that is uploaded for a single databin add*)
		"Key"->Default,
		(*readmode specifies the mode to read the value of the pin from when doing either a ChannelSend or a DatabinAdd*)
		"ReadMode"->Default,
		(*id specifies the custom key to send with the cURL request for publishing on a channel*)
		"ID"->Default,
		(*databin specifies which databin to upload to*)
		"Databin"->Default,
		(*channel specifies which channel to publish on*)
		"Channel"->Default
	};
ArduinoExecuteDriver[{ihandle_,dhandle_},args__,OptionsPattern[]]:=Module[
	{
		allArgs={args},
		cloudOpts=<|#->OptionValue[#]&/@{"Key","ReadMode","ID","Databin","Channel"}|>,
		binID,
		keyName,
		readMode,
		pinSpec
	},
	(	
		Switch[First[allArgs],
			"SoftwareReset",
			(
				Return[ArduinoExecute[{ihandle,dhandle},"SoftwareReset"]];
			),
			"DeleteTask",
			(
				Return[ArduinoExecute[{ihandle,dhandle},"DeleteTask",allArgs[[2]]]];
			),
			"DatabinAdd"|"ChannelSend"|DatabinAdd|ChannelSend,
			(
				(*before anything else, make sure we're on a Yun*)
				If[$DeviceStates[ihandle,"BoardType"]=!="Yun",
					(*THEN*)
					(*issue message and fail, can't do any data drop or channel framework stuff on anything other than a Yun*)
					(
						Message[DeviceExecute::onlyYun,First[allArgs]];
						Return[$Failed];
					)
				];
				
				(*next make sure that we have at least two arguments to allArgs*)
				If[Length[allArgs]<2,
					Message[DeviceExecute::needsArgs,First[allArgs]];
					Return[$Failed];
				];
				
				(*now we want to check the form of the pins*)
				(*there are a few different ways a user could specify the pins*)
				(*single pin => uses default channel / bin and default key naming convention*)
				(*list/sequence of pins => uses default channel / bin and default key naming convention*)
				pinSpec = allArgs[[2;;]];

				(*now check the pins*)
				If[Not[AllTrue[pinSpec,MemberQ[arduinoUnoPins,#]&]],
					(*THEN*)
					(*at least one of the pins is invalid, issue message and return $Failed*)
					(
						Message[DeviceExecute::invalidPin,#]&/@Select[Not[MemberQ[arduinoUnoPins,#]]&]@pinSpec;
						Return[$Failed];
					)
					(*ELSE*)
					(*all the pins are good*)
				];
				
				(*finally, now that we have checked the pin and readMode actually perform the relevant execute for the type we are doing*)
				If[MemberQ[{DatabinAdd,"DatabinAdd"},First[allArgs]],
					(*THEN*)
					(*user is adding to a databin, so call the databin add execute function*)
					Return[
						ArduinoExecuteDatabin[
							{ihandle,dhandle},
							pinSpec,
							cloudOpts["Key"],
							cloudOpts["ReadMode"],
							cloudOpts["Databin"],
							OptionValue["Scheduling"]
						]
					],
					(*ELSE*)
					(*user is doing a ChannelSend, do that execute*)
					(
						Return[
							ArduinoExecuteChannel[
								{ihandle,dhandle},
								pinSpec,
								cloudOpts["ID"],
								cloudOpts["ReadMode"],
								cloudOpts["Channel"],
								OptionValue["Scheduling"]
							]
						];
					)
				];
			),
			_, (*all other cases*)
			Switch[Length[allArgs],
				1,
				(*this case is where a function does not have any arguments, so args is just the function's name*)
				(
					(*check if this execution is scheduled*)
					If[OptionValue["Scheduling"]===None,
						(*THEN*)
						(*no scheduling necessary, so run normally*)
						(
							Return[ArduinoExecute[{ihandle,dhandle},First@allArgs]];
						),
						(*ELSE*)
						(*scheduling isn't none, so use ArduinoScheduleExecute with a list of {func name, args, scheduling}*)
						(
							Return[ArduinoScheduleExecute[{ihandle,dhandle},{First@allArgs,{},OptionValue["Scheduling"]}]];
						)
					]
				),
				_Integer,
				(*for case of more than one argument, include the args as a list as the second argument*)
				(
					(*check if this execution is scheduled*)
					If[OptionValue["Scheduling"]===None,
						(*THEN*)
						(*no scheduling necessary, so run normally*)
						(
							Return[ArduinoExecute[{ihandle,dhandle},First[allArgs],allArgs[[2;;]]]];
						),
						(*ELSE*)
						(*scheduling isn't none, so use ArduinoScheduleExecute with a list of {func name, args, scheduling}*)
						(
							Return[ArduinoScheduleExecute[{ihandle,dhandle},{First@allArgs,allArgs[[2;;]],OptionValue["Scheduling"]}]];
						)
					]
				)
			]
		]
	)
];


ArduinoExecuteDatabin[{ihandle_,dhandle_},pinSpec_,key_,readModeSpec_,databinSpec_,scheduling_]:=Module[
	{
		binID,
		(*we will always call this with exactly 1 pin*)
		pin = First[pinSpec],
		keyName,
		readMode
	},
	(
		(*now determine the databin info from the options association we got*)
		Which[
			databinSpec===Default, (*default value, check the device properties for a bin ID*)
			(
				If[KeyExistsQ[$DeviceStates[ihandle],"DefaultDatabin"],
					(*THEN*)
					(*we have a default to use*)
					(
						binID = $DeviceStates[ihandle,"DefaultDatabin"];
					),
					(*ELSE*)
					(*no default to use, so fail*)
					(
						Message[DeviceExecute::noDatabin];
						Return[$Failed];
					)
				];
			),
			StringQ[databinSpec],
			(*THEN*)
			(*the value is at least a string, so validate the id*)
			(
				If[StringQ[Databin[databinSpec]["ShortID"]],
					(*THEN*)
					(*worked, use that as the binID*)
					(
						binID = Databin[databinSpec]["ShortID"];
					),
					(*ELSE*)
					(*failed for some reason*)
					(
						Message[DeviceExecute::invalidBin,databinSpec];
						Return[$Failed];
					)
				]
			),
			MatchQ[optsAssoc["Databin"],_Databin], (*it's a databin object, get the ID*)
			(
				If[StringQ[databinSpec["ShortID"]],
					(*THEN*)
					(*worked, use that as the binID*)
					(
						binID = databinSpec["ShortID"];
					),
					(*ELSE*)
					(*failed for some reason*)
					(
						Message[DeviceExecute::invalidBin,databinSpec];
						Return[$Failed];
					)
				];
			),
			True, (*any other case*)
			(
				(*invalid, need to get a databin*)
				Message[DeviceExecute::invalidBin,databinSpec];
				Return[$Failed];
			)
		];
		
		(*next check the key*)
		Which[(key===Default||key===Automatic)&&KeyExistsQ[arduinoPinToKey,pin],(*use automatic pin name*)
			(
				(*rename the pin to match the pattern DigitalPinX or AnalogPinX*)
				keyName = arduinoPinToKey[pin];
			),
			StringQ[key], (*user specified key*)
			(
				keyName = key;
			),
			Not[KeyExistsQ[arduinoPinToKey,pin]], (*pin specified to read from doesn't exist*)
			(
				Message[DeviceExecute::invalidPin,pin];
				Return[$Failed];
			),
			True, (*any other case*)
			(
				Message[DeviceExecute::invalidKey,key,"DataDrop"];
				Return[$Failed];
			)
		];
		
		(*validate the ReadMode option*)
		readMode = validateReadModes[{readModeSpec},pinSpec];
		If[readMode === $Failed,
			(*THEN*)
			(*it failed, don't issue messages, just return $Failed, as validateReadModes will have generated the messages for us*)
			Return[$Failed],
			(*ELSE*)
			(*the readMode is valid, get the first option*)
			readMode = First[readMode]
		];
		
		(*now that we have the binID, all that's left is to call the function*)
		If[scheduling===None,
			(*THEN*)
			(*no scheduling necessary, so run normally*)
			(
				(*get the name of the function from $DeviceStates*)
				Return[ArduinoExecute[{ihandle,dhandle},$DeviceStates[ihandle,"DatabinAddFunctionName"],{binID,pin,readMode,keyName}]];
			),
			(*ELSE*)
			(*scheduling isn't none, so use ArduinoScheduleExecute with a list of {func name, args, scheduling}*)
			(
				Return[ArduinoScheduleExecute[{ihandle,dhandle},{$DeviceStates[ihandle,"DatabinAddFunctionName"],{binID,pin,readMode,keyName},scheduling}]];
			)
		]
	)
];


ArduinoExecuteChannel[{ihandle_,dhandle_},pinSpec_,customKey_,readMode_,channelSpec_,scheduling_]:=Module[
	{
		channelID,
		customID,
		readModes,
		fail = False
	},
	(
		(*now determine the channel info from the options association we got*)
		Which[
			channelSpec===Default, (*default value, check the device properties for a bin ID*)
			(
				If[KeyExistsQ[$DeviceStates[ihandle],"DefaultChannel"],
					(*THEN*)
					(*we have a default to use*)
					(
						binID = $DeviceStates[ihandle,"DefaultChannel"];
					),
					(*ELSE*)
					(*no default to use, so fail*)
					(
						Message[DeviceExecute::noChannel];
						Return[$Failed];
					)
				];
			),
			StringQ[channelSpec]||MatchQ[channelSpec,_ChannelObject], (*either a string spec of the channel or the channel object itself*)
			(
				(*try parsing out the url*)
				With[{urlPath = getChannelURLPath[If[StringQ[channelSpec],ChannelObject[#],#]&@channelSpec]},
					(*check the parsing to see if it worked*)
					If[StringQ[urlPath],
						(*THEN*)
						(*worked, use that as the binID*)
						(
							channelID = urlPath;
						),
						(*ELSE*)
						(*failed for some reason*)
						(
							Message[DeviceExecute::invalidChannel,channelSpec];
							Return[$Failed];
						)
					]
				]
			),
			True, (*any other case*)
			(
				(*invalid, need to get a databin*)
				Message[DeviceExecute::invalidChannel,channelSpec];
				Return[$Failed];
			)
		];

		(*now check the ID for this message*)
		Which[(customKey===Default||customKey===Automatic),(*use automatic custom key spec*)
			(
				(*make the key a random 4 digit id of letters and numbers*)
				customID = uniqueRandomAlphanumericString[6];
			),
			StringQ[customKey], (*user specified key*)
			(
				customID = customKey;
			),
			True, (*any other specification is invalid*)
			(
				Message[DeviceExecute::invalidKey,customKey,"ChannelFramework"];
				Return[$Failed];
			)
		];
		
		(*validate the read modes and return $Failed if it doesn't validate*)
		(*validateReadModes will generate the message for us*)
		readModes = validateReadModes[readMode,pinSpec];
		If[readModes === $Failed,
			Return[$Failed]
		];
		
		(*now that we have the binID, all that's left is to call the function*)
		(*check if we have to schedule it or not*)
		If[scheduling === None,
			(*THEN*)
			(*no scheduling necessary, so run normally*)
			(
				(*get the name of the function from $DeviceStates*)
				ArduinoExecute[
					{ihandle,dhandle},
					$DeviceStates[ihandle,"ChannelSendFunctionName"],
					{channelID,pinSpec,readModes,Length[pinSpec],customID}
				]
			),
			(*ELSE*)
			(*scheduling isn't none, so use ArduinoScheduleExecute with a list of {func-name, args, scheduling}*)
			(
				ArduinoScheduleExecute[
					{ihandle,dhandle},
					{
						$DeviceStates[ihandle,"ChannelSendFunctionName"],
						{channelID,pinSpec,readModes,Length[pinSpec],customID},
						scheduling
					}
				]
			)
		]
	)
];


ArduinoConfigureDriverWrapper[{ihandle_,dhandle_},args__]:=Module[{},
	(
		Switch[Head[args],
			Association,(*this is for Pin Configure*)ArduinoConfigureDriver[{ihandle,dhandle},args],
			Rule,(*need to switch on the first one*)
			(
				Switch[args[[1]],
					"Upload",ArduinoConfigureDriver[{ihandle,dhandle},"Upload",args[[2]]],
					"PinConfigure"|"PinConfigurations",ArduinoConfigureDriver[{ihandle,dhandle},"PinConfigure"->args[[2]]]
				]
			),
			String,(*this might be just upload with no options, but check*)
			(
				If[args==="Upload",
					(*THEN*)
					(*perform a default upload*)
					ArduinoConfigureDriver[{ihandle,dhandle},"Upload"],
					(*ELSE*)
					(*the string is something, else raise a message*)
					(
						Message[DeviceConfigure::invalidOptions,args];
					)
				]
			),
			_,(*this is for anything else*)
			(
				Message[DeviceConfigure::invalidOptions,args];
			)
		]
	)
];


(*the configure driver basically will just pass the options for the configuration to the arduinoUpload 
function if the user wants to add a new function, or if the user just wants to configure pin mode,
then we just send those packets and update the internal association *)
Options[ArduinoConfigureDriver]=
{
		"Debug"->False,
		"Libraries" -> {},
		"CleanIntermediate"->True,
		"FlashVerify"->True,
		Initialization->"",
		"Functions"-><||>,
		"BootFunction"->None,
		"DataDrop"->Automatic,
		"ChannelFramework"->Automatic,
		"Databin"->None,
		"Channel"->None
};
ArduinoConfigureDriver[{ihandle_,dhandle_},"Upload",OptionsPattern[]]:=Module[
	{
		compilerLocation=$arduinoInstallLocation,
		debug = TrueQ[OptionValue["Debug"]],
		customFunctions = OptionValue["Functions"],
		setupFunc = OptionValue["BootFunction"],
		channelVal = OptionValue["Channel"],
		databinVal = OptionValue["Databin"],
		setupFuncName,
		functionIndex,
		externFunc,
		arduinoVersion,
		setupFuncOpts = None,
		databinFuncOptions,
		channelFuncOptions,
		binID,
		channelID,
		pin,
		pins,
		readModes,
		keyName,
		channelFrameworkFuncName,
		databinAddFuncName,
		dataDropEnabled = OptionValue["DataDrop"],
		(*the c code that gets uploaded to the arduino to execute a DatabinAdd*)
		databinFunc="
void databinSendExecute(char * binID, int pin, int readMode, char * key)
{
    int value = -1;
    float voltage = -1.0;

    //first get the value
    if(readMode == DIGITAL_READ)
    {
        value = digitalRead(pin);
        //now upload the value to the specified databin
        DatabinIntegerAdd(binID,key,value);
    }
    else if (readMode == ANALOG_READ)
    {
        voltage = analogRead(pin)*ADC_TO_VOLTAGE_CONV;
        //now upload the value to the specified databin
        DatabinRealAdd(binID,key,voltage);
    }
}",
		(*the c code that gets uploaded to the arduino to execute a ChannelSend*)
		channelFunc="
void channelComboSend(char * channelPath, long * pins, long * readModes, long len, char * key)
{
    //make arrays to store the values in
    //for the keyname char strings, the extra space is for the last string
    char ** keyNames = (char **)calloc(len + 1,sizeof(char *));
    String * stringVals = (String *) calloc(len+1,sizeof(String));

    //now read in the values, saving them as strings
    int index;
    for(index = 0; index < len; index++)
    {
        //store the number of this pin in the string as the second (and possibly third) character
        if(pins[index] > 9)
        {
            //because the number has two digits, we need 3 characters for the key (and the nul character)
            keyNames[index] = (char *) calloc(4,sizeof(char));

            //store both digits
            keyNames[index][1] = '0' + (pins[index] / 10) % 10;
            keyNames[index][2] = '0' + (pins[index]) % 10;
        }
        else
        {
            //only need to allocate 3 characters for a pin with 1 digit in the number
            keyNames[index] = (char *) calloc(3,sizeof(char));

            //just store the one digit
            keyNames[index][1] = '0' + (pins[index]) % 10;
        }

        //now actually perform the read and set the first character depending on what type of read it is
        if(readModes[index] == ANALOG_READ)
        {
            //then do an analog read, multipying the result by the conversion factor to get the voltage
            stringVals[index] = String(analogRead(pins[index]) * ADC_TO_VOLTAGE_CONV);

            //store the type as 'A' for analog
            keyNames[index][0] = 'A';
        }
        else if (readModes[index] == DIGITAL_READ)
        {
            //then do an digital read
            stringVals[index] = String(digitalRead(pins[index]));

            //store the type as 'D' for digital
            keyNames[index][0] = 'D';
        }
    }

    //before calling the final function, set the custom key as the last element
    keyNames[len] = (char *) calloc(3,sizeof(char));

    //the key of the custom key is 'ID', the value is whatever is specified from the argument to this function
    keyNames[len][0] = 'I';
    keyNames[len][0] = 'D';
    stringVals[len] = String(key);

    //finally, perform the upload
    cloudSend(channelPath,keyNames,stringVals,len,channelFrameworkURL);

    //free up the memory now
    free(stringVals);

    //less than or equal to because the length of the array is really len + 1
    for(index = 0; index <= len; index++)
    {
        free(keyNames[index]);
    }

    //free the array of points as well
    free(keyNames);
}",
		(*the templated C code for adding to a databin from a BootFunction*)
		voidDatabinFunc="
void databinSend()
{
    /*GENERATED DATABINADD FUNCTION FOLLOWS*/
    `valType` value = `readFunc`(`pin`)`multiplier`;
    `databinAddFunc`(`binID`,`keyName`,value);
    /*END GENERATED DATABINADD FUNCTION*/
}",
		(*the templated C code for sending to a channel from a BootFunction*)
		voidChannelFunc="
void databinBootSend()
{
    /*GENERATED CHANNELSEND FUNCTION FOLLOWS*/

    char * channelPath = \"`channelID`\";

    byte pins[] = {
        `pins`
    };

    byte readModes[] = {
        `readMode`
    };

    int len = `len`;

    char * key = \"`key`\";

    //make arrays to store the values in
    //for the keyname char strings, the extra space is for the last string
    char ** keyNames = (char **)calloc(len + 1,sizeof(char *));
    String * stringVals = (String *) calloc(len+1,sizeof(String));

    //now read in the values, saving them as strings
    int index;
    for(index = 0; index < len; index++)
    {
        //store the number of this pin in the string as the second (and possibly third) character
        if(pins[index] > 9 && readModes[index] != ANALOG_READ)
        {
            //because the number has two digits, we need 3 characters for the key (and the nul character)
            keyNames[index] = (char *) calloc(4,sizeof(char));

            //store both digits
            keyNames[index][1] = '0' + (pins[index] / 10) % 10;
            keyNames[index][2] = '0' + (pins[index]) % 10;
        }
        else if(pins[index] > 9 && readModes[index] == ANALOG_READ)
        {
            //only need to allocate 3 characters for a pin with 1 digit in the number
            keyNames[index] = (char *) calloc(3,sizeof(char));

            //just store the one digit - but note we need to subtract 18 because the variable A0 is defined to be 18, A1 is defined to be 19, etc.
            keyNames[index][1] = '0' + (pins[index] - 18) % 10;
        }
        else
        {
            //only need to allocate 3 characters for a pin with 1 digit in the number
            keyNames[index] = (char *) calloc(3,sizeof(char));

            //just store the one digit
            keyNames[index][1] = '0' + (pins[index]) % 10;
        }

        //now actually perform the read and set the first character depending on what type of read it is
        if(readModes[index] == ANALOG_READ)
        {
            //then do an analog read, multipying the result by the conversion factor to get the voltage
            stringVals[index] = String(analogRead(pins[index]) * ADC_TO_VOLTAGE_CONV);

            //store the type as 'A' for analog
            keyNames[index][0] = 'A';
        }
        else if (readModes[index] == DIGITAL_READ)
        {
            //then do an digital read
            stringVals[index] = String(digitalRead(pins[index]));

            //store the type as 'D' for digital
            keyNames[index][0] = 'D';
        }
    }

    //before calling the final function, set the custom key as the last element
    keyNames[len] = (char *) calloc(3,sizeof(char));

    //the key of the custom key is 'ID', the value is whatever is specified from the argument to this function
    keyNames[len][0] = 'I';
    keyNames[len][0] = 'D';
    stringVals[len] = String(key);

    //finally, perform the upload
    cloudSend(channelPath,keyNames,stringVals,len,channelFrameworkURL);

    //free up the memory now
    free(stringVals);

    //less than or equal to because the length of the array is really len + 1
    for(index = 0; index <= len; index++)
    {
        free(keyNames[index]);
    }

    //free the array of points as well
    free(keyNames);
	/*END GENERATED CHANNELSEND FUNCTION*/
}"
	},
	(
		(*first though, check if an upload has been previously done in the past 1 seconds*)
		(*this check prevents the Front End from calling this function repeatedly sometimes causing strange behavior*)
		If[AbsoluteTime[]-$lastConfigCall>1,
			(*THEN*)
			(*we are okay to call the function, enough time has passed since last call*)
			(
				(*first check to make sure we have the arduino software installed*)
				If[$avrdudeInstallLocation == None || $avrdudeConfigFileLocation == None || $avrgccInstallLocation == None,
					(*THEN*)
					(*the compiler location isn't configured, so we can't do anything else*)
					(
						Message[DeviceConfigure::noArduinoInstall];
						Return[$Failed];
					)
					(*ELSE*)
					(*it's not none, so assume that it is alright to use*)
				];
				
				(*TODO: implement remembering*)
				(*this is a workaround to prevent the "remembering" feature from being utilized, will be enabled in a future release*)
				$previousFunctions=<||>;
				$functionCalls=<||>;
				(*now append the setup function if there is one*)
				If[setupFunc=!=None,
					(*THEN*)
					(*it has a non default value, validate it, it should be of the form ArduinoCode[<|"Code"->String,opts|>] or <|"Code"->String,opts|>*)
					(
						If[Length[setupFunc]>=1&&Head[First[setupFunc]]===Association, 
							(*THEN*)
							(*is an ArduinoCode specified BootFunction, so get the normal association out of from it*)
							(
								setupFunc = First[setupFunc]
							)
						];
						If[KeyExistsQ["Code"][setupFunc]&&validSchedulingOption[setupFunc["Scheduling"]],
							(*THEN*)
							(*it's valid append it to the customFunctions*)
							(
								setupFuncName = CreateUUID[];
								(*check if this is a Databin function*)
								Which[
									MemberQ[{DatabinAdd,"DatabinAdd"},setupFunc["Code"]],
									(
										(*make sure that DataDrop support is enabled*)
										Which[
											(TrueQ[OptionValue["DataDrop"]]||OptionValue["DataDrop"]===Automatic)&&$DeviceStates[ihandle,"BoardType"]==="Yun",
											(*it's enabled, proceed with generating the code for the setup function*)
											(
												(*then it's a databin function, so we need to do some work in generating the function text before adding it normally as a setup function*)
												databinFuncOptions=<|#->setupFunc[#]&/@{"Key","Databin","ReadMode","Pin"}|>;
												(*firstly check to make sure we have a pin option*)
												If[MissingQ[databinFuncOptions["Pin"]],
													(*THEN*)
													(*didn't provide which pin, and we can't infer this, so issue message and fail*)
													(
														Message[DeviceConfigure::databinPin];
														Return[$Failed];
													),
													(*ELSE*)
													(*we have a pin specified*)
													(
														(*we will validate this later*)
														pin = databinFuncOptions["Pin"];
													)
												];
												(*now check on the databin id*)
												If[MissingQ[databinFuncOptions["Databin"]],
													(*THEN*)
													(*wasn't provided a databin id, check if there's a default one set*)
													(
														If[KeyExistsQ[$DeviceStates[ihandle],"DefaultDatabin"],
															(*THEN*)
															(*it's there, and we can use that*)
															(
																binID = $DeviceStates[ihandle,"DefaultDatabin"];
															),
															(*ELSE*)
															(*we don't have any databin to use then*)
															(
																Message[DeviceConfigure::databinID];
																Return[$Failed];
															)
														];
													),
													(*ELSE*)
													(*we have a databin value*)
													(
														(*quietly attempt to get the bin id, we will validate it in a moment*)
														binID = Databin[databinFuncOptions["Databin"]];
													)
												];
												(*now check the readmode option*)
												If[MissingQ[databinFuncOptions["ReadMode"]],
													(*THEN*)
													(*use the pin name to determine the type*)
													(
														(*because the canonacial name from pinKeyToDataDropName returns something like Analog0, Digital4, etc. we can use string cases*)
														(*to get which type it is*)
														readModes = Quiet[First[StringCases[pinKeyToDataDropName[arduinoPinToKey[pin]], {"Analog", "Digital"}]]];
													),
													(*ELSE*)
													(*user specified one, so use that*)
													(
														readModes = databinFuncOptions["ReadMode"];
													)
												];
												If[MissingQ[databinFuncOptions["Key"]],
													(*THEN*)
													(*user didn't specify a key name, so use the pin*)
													(
														keyName = Quiet[pinKeyToDataDropName[arduinoPinToKey[pin]]];
													),
													(*ELSE*)
													(*use the user specified value*)
													(
														keyName = databinFuncOptions["Key"];
													)
												];
												(*if we get here it means we have the necessary options, so validate them*)
												If[Not[StringQ[Databin[binID]["ShortID"]]],
													(*THEN*)
													(*the bin ID is invalid, return $Failed*)
													(
														Message[DeviceConfigure::invalidDatabin,binID];
														Return[$Failed];
													)
												];
												(*next do the pin*)
												If[MissingQ[arduinoPinToKey[pin]/.{"D0"->Missing,"D1"->Missing}],
													(*THEN*)
													(*the pin isn't a valid pin*)
													(
														Message[DeviceConfigure::invalidPin,pin];
														Return[$Failed];
													)
												];
												(*now check the pinMode*)
												Which[readMode=!="Digital"&&readMode=!="Analog",
													(*THEN*)
													(*the read mode is invalid*)
													(
														Message[DeviceConfigure::invalidPinMode,readModes];
														Return[$Failed];
													),
													readMode==="Analog",
													(
														If[Not[MemberQ[arduinoUnoAnalogPins,pin]],
															(*THEN*)
															(*not a valid analog pin*)
															(
																Message[DeviceConfigure::invalidAnalogPin,pin]
															)
														];
													)
												];
												(*finally check the key name*)
												If[Not[StringQ[keyName]],
													(*THEN*)
													(*the read mode is invalid*)
													(
														Message[DeviceConfigure::invalidKey,keyName,"DataDrop"];
														Return[$Failed];
													)
												];
												(*now generate the function text*)
												
												PrependTo[customFunctions,
													setupFuncName-><|"Code"->
														StringTemplate[voidDatabinFunc][<|
															"valType"->Switch[readModes,"Digital","int","Analog","float"],
															"readFunc"->Switch[readModes,"Digital","digitalRead","Analog","analogRead"],
															(*the multiplier is to convert the digital value of the analog ports to a raw voltage value*)
															"multiplier"->Switch[readModes,"Digital","","Analog","*ADC_TO_VOLTAGE_CONV"],
															"pin"->Last@Characters@arduinoPinToKey@pin,
															"databinAddFunc"->Switch[readModes,"Digital","DatabinIntegerAdd","Analog","DatabinRealAdd"],
															"binID"->StringJoin[{"\"",Databin[binID]["ShortID"],"\""}],
															"keyName"->StringJoin[{"\"",keyName,"\""}]
															|>
														]
													|>
												];
												setupFuncOpts = setupFunc;
												$DeviceStates[ihandle,"BootFunction"]=True;
											),
											$DeviceStates[ihandle,"BoardType"]=!="Yun"&&TrueQ[OptionValue["DataDrop"]],
											(*we're not on a yun, but the user tried to specify a setup function of DatabinAdd*)
											(
												Message[DeviceConfigure::onlyYunSetupFunc];
											),
											$DeviceStates[ihandle,"BoardType"]==="Yun"&&OptionValue["DataDrop"]===False,
											(*datadrop support has been explicitly disabled, nothing we can do*)
											(
												Message[DeviceConfigure::datadropDisabled];
											)
										];
									),
									MemberQ[{"ChannelSend",ChannelSend},setupFunc["Code"]],
									(
										(*make sure that channelframework is enabled*)
										Which[
											(TrueQ[OptionValue["ChannelFramework"]]||OptionValue["ChannelFramework"]===Automatic)&&$DeviceStates[ihandle,"BoardType"]==="Yun",
											(*it's enabled, proceed with generating the code for the setup function*)
											(
												(*then it's a databin function, so we need to do some work in generating the function text before adding it normally as a setup function*)
												channelFuncOptions=<|#->setupFunc[#]&/@{"ID","Channel","ReadMode","Pins"}|>;
												(*firstly check to make sure we have a pin option*)
												If[MissingQ[channelFuncOptions["Pins"]],
													(*THEN*)
													(*didn't provide which pin, and we can't infer this, so issue message and fail*)
													(
														Message[DeviceConfigure::channelPin];
														Return[$Failed];
													),
													(*ELSE*)
													(*we have a pin specified*)
													(
														(*we will validate this later*)
														pin = channelFuncOptions["Pins"];
													)
												];
												(*now check on the channel id*)
												If[MissingQ[channelFuncOptions["Channel"]],
													(*THEN*)
													(*wasn't provided a channel id, check if there's a default one set*)
													(
														If[KeyExistsQ[$DeviceStates[ihandle],"DefaultChannel"],
															(*THEN*)
															(*it's there, and we can use that*)
															(
																channelID = $DeviceStates[ihandle,"DefaultChannel"];
															),
															(*ELSE*)
															(*we don't have any databin to use then*)
															(
																Message[DeviceConfigure::channelID];
																Return[$Failed];
															)
														];
													),
													(*ELSE*)
													(*a channel specified, confirm it*)
													(
														If[StringQ[channelFuncOptions["Channel"]]||MatchQ[channelFuncOptions["Channel"],_ChannelObject],
															(*THEN*)
															(*a valid channel was specified, so use it*)
															With[
																{
																	channelURLPath = getChannelURLPath[
																		If[StringQ[#],ChannelObject[#],#]&@channelFuncOptions["Channel"]
																	]
																},
																If[StringQ[channelURLPath],
																	(*THEN*)
																	(*it's a valid channel*)
																	channelID = channelURLPath,
																	(*ELSE*)
																	(*invalid channel specified*)
																	(
																		Message[DeviceConfigure::invalidChannel,channelFuncOptions["Channel"]];
																		Return[$Failed];
																	)
																]
															],
															(*ELSE*)
															(*the channel option isn't the correct type*)
															(
																Message[DeviceConfigure::invalidChannel,channelFuncOptions["Channel"]];
																Return[$Failed];
															)
														]
													)
												];
												
												(*check the pins next*)
												If[Not[AllTrue[channelFuncOptions["Pins"],MemberQ[arduinoUnoPins,#]&]],
													(*THEN*)
													(*at least one of the pins is invalid, issue message and return $Failed*)
													(
														Message[DeviceConfigure::invalidPin,#]&/@Select[Not[MemberQ[arduinoUnoPins,#]]&]@channelFuncOptions["Pins"];
														Return[$Failed];
													),
													(*ELSE*)
													(*all the pins are good*)
													pins = channelFuncOptions["Pins"];
												];
												
												(*now check the readmode option*)
												readModes = validateReadModes[channelFuncOptions["ReadMode"],pins];
												If[readModes === $Failed,
													Return[$Failed]
												];
												
												(*now confirm that the custom key / ID is okay*)
												Which[
													MissingQ[channelFuncOptions["ID"]],
													(*user didn't specify a key name, so use a randomly generated one*)
													(
														(*length of 8 gives us 2,176,782,336 possibilities, more than enough to avoid collisions*)
														keyName = uniqueRandomAlphanumericString[6];
													),
													StringQ[channelFuncOptions["ID"]]&&PrintableASCIIQ[channelFuncOptions["ID"]],
													(*use the user specified value*)
													(
														keyName = channelFuncOptions["ID"];
													),
													True,(*invalid id, must be a string and an ASCII string*)
													(
														Message[DeviceConfigure::invalidCustomKey,channelFuncOptions["ID"]];
														Return[$Failed];
													)
												];
												
												(*now generate the function text*)
												PrependTo[customFunctions,
													setupFuncName-><|"Code"->
														StringTemplate[voidChannelFunc][
															<|
																(*for the pins because we are verbatim pasting it in, the more portable way to specify*)
																(*analog pins is to use the string form so the preprocessor defines the pin appropriately*)
																"pins" -> StringJoin[Riffle[ToString /@ ReplaceAll[pins,analogStringPin], ",\n"]],
																"key" -> keyName,
																"len" -> Length[pins],
																"channelID" -> channelID, 
																"readMode" -> StringJoin[Riffle[ToString /@ readModes, ",\n"]]
															|>
														]
													|>
												];
												setupFuncOpts = setupFunc;
												$DeviceStates[ihandle,"BootFunction"] = True;
												
												(*finally we need to disable the DataDrop functionality if it's set to Automatic, else we wont' have space*)
												(*on the device*)
												If[dataDropEnabled === Automatic,
													(*THEN*)
													(*disable it for further processing*)
													dataDropEnabled = False
													(*ELSE*)
													(*the user specifically enabled it or disabled it, don't modify it*)
												];
											),
											$DeviceStates[ihandle,"BoardType"]=!="Yun"&&TrueQ[OptionValue["ChannelFramework"]],
											(*we're not on a yun, but the user tried to specify a setup function of DatabinAdd*)
											(
												Message[DeviceConfigure::onlyYunSetupFunc];
											),
											$DeviceStates[ihandle,"BoardType"]==="Yun"&&OptionValue["ChannelFramework"]===False,
											(*datadrop support has been explicitly disabled, nothing we can do*)
											(
												Message[DeviceConfigure::channelDisabled];
											),
											True,
											(
												Message[Error];
												Return[$Failed];
											)
										]
									),
									StringQ[setupFunc["Code"]], (*it's just a normal setup function, handle normally*)
									(
										(*only pass in the Code option*)
										PrependTo[customFunctions,setupFuncName->KeyTake[setupFunc,"Code"]];
										(*the full object with any of the other options is passed in as setupFuncOpts*)
										setupFuncOpts = setupFunc;
										$DeviceStates[ihandle]["BootFunction"] = True;
									),
									True,
									(
										Message[DeviceConfigure::invalidSetupFunc];
									)
								];
								
							),
							(*ELSE*)
							(*it's invalid*)
							(
								Message[DeviceConfigure::invalidSetupFunc];
								Return[$Failed];
							)
						];
					),
					(*ELSE*)
					(*it doesn't exist, we should reset the boot function message tracking variable*)
					(
						$DeviceStates[ihandle,"BootFunction"]=False;
					)
				];
				(*lastly, before looking at all the user custom functions, append the DatabinAdd function if the option DataDrop is True and we are on an Arduino Yun*)
				(*also do the same check for the corresponding ChannelFramework function*)
				Which[
					((TrueQ[dataDropEnabled]||dataDropEnabled===Automatic)&&$DeviceStates[ihandle,"BoardType"]==="Yun") &&
					((TrueQ[OptionValue["ChannelFramework"]]||OptionValue["ChannelFramework"]===Automatic)),
					(*both the DataDrop and ChannelFramework functionality is enabled*)
					(
						databinAddFuncName = CreateUUID[];
						AppendTo[customFunctions,databinAddFuncName-><|"Code"->databinFunc,"ArgumentTypes"->{String,Integer,Integer,String}|>];
						$DeviceStates[ihandle,"DatabinAddFunctionName"] = databinAddFuncName;
						(*also check if the databin passed in exists and if it does if it is a valid bin*)
						If[databinVal=!=None,
							(*THEN*)
							(*the option was specified, verify it and set it if it is valid*)
							(
								If[StringQ[Databin[databinVal]["ShortID"]],
									(*THEN*)
									(*valid id, use that one*)
									(
										$DeviceStates[ihandle,"DefaultDatabin"]=Databin[databinVal]["ShortID"];
									),
									(*ELSE*)
									(*invalid, issue message and continue*)
									(
										Message[DeviceConfigure::invalidDatabin,databinVal];
									)
								];
							)
							(*ELSE*)
							(*none specified, move along*)
						];
						
						(*now do the channel framework equivalent*)
						channelFrameworkFuncName = CreateUUID[];
						AppendTo[customFunctions,channelFrameworkFuncName-><|"Code"->channelFunc,"ArgumentTypes"->{String,{Integer},{Integer},Integer,String}|>];
						$DeviceStates[ihandle,"ChannelSendFunctionName"] = channelFrameworkFuncName;
						(*check which kind of channel was specified*)
						Which[
							MatchQ[channelVal,_ChannelObject]||StringQ[channelVal],
							(*channel was specified as a string channel spec or a ChannelObject*)
							(
								With[{parsedURLPath = getChannelURLPath[If[StringQ[channelVal],ChannelObject@#,#]&@channelVal]},
									If[StringQ[parsedURLPath],
										(*THEN*)
										(*valid id, use that one*)
										(
											$DeviceStates[ihandle,"DefaultChannel"] = parsedURLPath;
										),
										(*ELSE*)
										(*invalid, issue message and continue*)
										(
											Message[DeviceConfigure::invalidChannel,channelVal];
										)
									]
								];
							),
							channelVal=!=None, (*invalid specification for the channel*)
							(
								Message[DeviceConfigure::invalidChannel,channelVal];
							)
						];
					),
					(TrueQ[OptionValue["ChannelFramework"]]||OptionValue["ChannelFramework"]===Automatic)&&$DeviceStates[ihandle,"BoardType"]==="Yun",
					(*only ChannelFramework is enabled, we need to add the custom function specifically for DataDrop*)
					(
						(*now do the channel framework equivalent*)
						channelFrameworkFuncName = CreateUUID[];
						AppendTo[customFunctions,channelFrameworkFuncName-><|"Code"->channelFunc,"ArgumentTypes"->{String,{Integer},{Integer},Integer,String}|>];
						$DeviceStates[ihandle,"ChannelSendFunctionName"] = channelFrameworkFuncName;
						(*check which kind of channel was specified*)
						Which[channelVal=!=None&&MatchQ[channelVal,_ChannelObject],
							(*channel was specified as a ChannelObject*)
							(
								With[{parsedURLPath = URLBuild[URLParse[First[channelVal]["URL"]]["Path"]]},
									If[StringQ[parsedURLPath],
										(*THEN*)
										(*valid id, use that one*)
										(
											$DeviceStates[ihandle,"DefaultChannel"] = parsedURLPath;
										),
										(*ELSE*)
										(*invalid, issue message and continue*)
										(
											Message[DeviceConfigure::invalidChannel,channelVal];
										)
									]
								];
							),
							channelVal=!=None&&StringQ[channelVal],
							(*channel was specified as a string, make it into a ChannelObject to extract the URL path properly*)
							(
								With[{parsedURLPath = URLBuild[URLParse[First[ChannelObject[channelVal]]["URL"]]["Path"]]},
									If[StringQ[parsedURLPath],
										(*THEN*)
										(*valid id, use that one*)
										(
											$DeviceStates[ihandle,"DefaultChannel"] = parsedURLPath;
										),
										(*ELSE*)
										(*invalid, issue message and continue*)
										(
											Message[DeviceConfigure::invalidChannel,channelVal];
										)
									]
								]
								
							)
							(*ELSE*)
							(*none specified, move along*)
						];
					),
					(TrueQ[dataDropEnabled]||dataDropEnabled===Automatic)&&$DeviceStates[ihandle,"BoardType"]==="Yun",
					(*only DataDrop is enabled, we need to add the custom function specifically for DataDrop*)
					(
						databinAddFuncName = CreateUUID[];
						AppendTo[customFunctions,databinAddFuncName-><|"Code"->databinFunc,"ArgumentTypes"->{String,Integer,Integer,String}|>];
						$DeviceStates[ihandle,"DatabinAddFunctionName"]=databinAddFuncName;
						(*also check if the databin passed in exists and if it does if it is a valid bin*)
						If[databinVal=!=None,
							(*THEN*)
							(*the option was specified, verify it and set it if it is valid*)
							(
								If[StringQ[Databin[databinVal]["ShortID"]],
									(*THEN*)
									(*valid id, use that one*)
									(
										$DeviceStates[ihandle,"DefaultDatabin"]=Databin[databinVal]["ShortID"];
									),
									(*ELSE*)
									(*invalid, issue message and continue*)
									(
										Message[DeviceConfigure::invalidDatabin,databinVal];
									)
								];
							)
							(*ELSE*)
							(*none specified, move along*)
						];
					),
					$DeviceStates[ihandle,"BoardType"]=!="Yun"&&TrueQ[OptionValue["DataDrop"]],
					(*not on a yun, but still tried to set DataDrop to True*)
					(
						Message[DeviceConfigure::onlyYunDatadrop];
					),
					$DeviceStates[ihandle,"BoardType"]=!="Yun"&&TrueQ[OptionValue["ChannelFramework"]],
					(*not on a yun, but still tried to set ChannelFramework to True*)
					(
						Message[DeviceConfigure::onlyYunChannelFramework];
					),
					$DeviceStates[ihandle,"BoardType"]==="Yun"&&
						Not[TrueQ[OptionValue["ChannelFramework"]]||OptionValue["ChannelFramework"]===Automatic]&&
						OptionValue["Channel"]=!=None,
					(*user tried to specify ChannelFramework->False but Channel->non-default*)
					(
						Message[DeviceConfigure::channelDisabled];
					),
					$DeviceStates[ihandle,"BoardType"]==="Yun"&&
						Not[TrueQ[OptionValue["DataDrop"]]||OptionValue["DataDrop"]===Automatic]&&
						OptionValue["Channel"]=!=None,
					(*user tried to specify ChannelFramework->False but Channel->non-default*)
					(
						Message[DeviceConfigure::datadropDisabled];
					)
				];
				(*we need to convert symbolic c to the ExternalFunction type of specification before sending it to arduinoUpload*)
				newCustomFunctions=Table[0,{Length[customFunctions]}];
				For[functionIndex=1,functionIndex<=Length[customFunctions],functionIndex++,
					(
						(*for each function in the list, check if it is a SymbolicC function*)
						If[Head[Normal[customFunctions][[functionIndex,2]]]===SymbolicC`CFunction,
							(*THEN*)
							(*this function is a SymbolicC function, so we have to compile it down to a c code string*)
							(
								newCustomFunctions[[functionIndex]]=
									Keys[customFunctions][[functionIndex]]->symbolicCFunctionFull[Values[customFunctions][[functionIndex]]]
							),
							(*ELSE*)
							(*the function is just a normally specified function, so wrap it into the old ExternalFunction*)
							(
								externFunc = externalFunctionConvert[Values[customFunctions][[functionIndex]]];
								If[externFunc === $Failed,
									(*THEN*)
									(*the function given was invalid, so issue message and return $Failed*)
									(
										Message[DeviceConfigure::invalidFunction,customFunctions[[functionIndex]]];
										Return[$Failed];
									)
								];
								newCustomFunctions[[functionIndex]]=Keys[customFunctions][[functionIndex]]->externFunc;
							)
						]
					)
				];
				customFunctions=Association[newCustomFunctions];
				(*TODO: make a function which finds the arduino version properly*)
				arduinoVersion="16700";
				Switch[$DeviceStates[ihandle,"BoardType"],
					Default|"Uno", (*default case*)
					(
						arduinoPinVersion="standard";
						arduinoBoardDefine="ARDUINO_AVR_UNO";
						arduinoChipPartNumber="atmega328p";
						arduinoExtraCompileDefines={};
						arduinoProgrammer="arduino";
						arduinoProgrammerBaudRate=115200;
					),
					"Yun",
					(
						arduinoPinVersion="yun";
						arduinoBoardDefine="ARDUINO_AVR_YUN";
						arduinoChipPartNumber="atmega32u4";
						(*the list here is so that Nothing disappears as it should if specified*)
						arduinoExtraCompileDefines=
						<|{
							"USB_VID"->"0x2341",
							"USB_PID"->"0x8041",
							"USB_MANUFACTURER"->"\"Unknown\"",
							"USB_PRODUCT"->"\"\\\"Arduino Yun\\\"\"",
							(*if datadrop is enabled, define it when compiling*)
							If[TrueQ[OptionValue["DataDrop"]]||OptionValue["DataDrop"]===Automatic,"YUN_DATADROP_FUNCTIONS"->"1",Nothing],
							(*if channels are enabled, define it when compiling*)
							If[TrueQ[OptionValue["ChannelFramework"]]||OptionValue["ChannelFramework"]===Automatic,"YUN_CHANNELFRAMEWORK_FUNCTIONS"->"1",Nothing]
						}
						|>;
						arduinoProgrammer="avr109";
						arduinoProgrammerBaudRate=57600;
					)
				];
				
				(*reset the progress bar indicator*)
				$compilationProgressIndicator = 0;
				(*also print off the progress bar for the upload process*)
				(*only print it off if we have a front end*)
				If[$FrontEnd =!= Null, PrintTemporary[ProgressIndicator[Dynamic[$compilationProgressIndicator],{0,30}]]];
				If[debug,Print["Debugging turned on"]];
				$serialPort = $DeviceStates[ihandle]["SerialPort"];
				If[debug,Print["Closing the serial connection"]];
				$timeToCloseSerialPort=AbsoluteTime[];
				(*close the connection to the device before uploading*)
				DeviceFramework`DeviceDriverOption["Firmata","CloseFunction"][{ihandle,dhandle}];
				If[debug,Print["Took "<>ToString[AbsoluteTime[]-$timeToCloseSerialPort]<>" seconds to close the serial port"]];
				
				(*finally actually perform the upload*)
				uploadResult=arduinoUpload
				[
					$serialPort,
					compilerLocation,
					PacletResource["DeviceDriver_Arduino","Sketch"],
					"Debug"->debug,
					"FlashVerify"->OptionValue["FlashVerify"],
					"Libraries"->OptionValue["Libraries"],
					Initialization->OptionValue[Initialization],
					"CleanIntermediate"->OptionValue["CleanIntermediate"],
					"Functions"->Join[$previousFunctions,customFunctions],
					"AVRDUDELocation"->$avrdudeInstallLocation,
					"AVRDUDEConfLocation"->$avrdudeConfigFileLocation,
					"AVRGCCLocation"->$avrgccInstallLocation,
					"ArduinoVersion"->arduinoVersion,
					"PinsVersion"->arduinoPinVersion,
					"BootFunction"->setupFuncOpts,
					"ChipPartNumber"->arduinoChipPartNumber,
					"ArduinoBoardDefine"->arduinoBoardDefine,
					"ExtraDefines"->arduinoExtraCompileDefines,
					"Programmer"->arduinoProgrammer,
					"BaudRate"->arduinoProgrammerBaudRate
				];
				If[debug,Print["Reopening the serial connection"]];
				(*now reopen the connection to the device*)
				DeviceFramework`DeviceDriverOption["Firmata","OpenFunction"][{ihandle,dhandle},$serialPort,"BaudRate"->115200];
				(*finally, if a new function was uploaded, add it's information to the internal association storing information on calling the functions*)
				If[uploadResult===$Failed,
					(*THEN*)
					(*the upload failed, so print off message if debug is on, and return $Failed*)
					(
						If[debug,
							Print["Upload failed"];
						];
						Return[$Failed];
					)
					(*ELSE*)
					(*the upload succeeded*)
				];

				(*now update the list of function calls with the other custom functions (if there was any)*)
				If[customFunctions===<||>,
					(*THEN*)
					(*there wasn't a function uploaded, so we don't need to add it*)
					Null,
					(*ELSE*)
					(*there was a function uploaded, we need to add it*)
					(
						$previousFunctions = Join[$previousFunctions,customFunctions];
						(*now need to update the function call options*)
						$functionCalls = addNewFunctionCalls[$functionCalls,customFunctions,"BootFunction"->setupFuncName];
					)
				];
				(*finally, because we re-uploaded data to it, there will no longer be a scheduled function on it running, so set the association for that*)
				$scheduledTaskRunning["Running"]=False;
				$scheduledTaskRunning["endTime"]=AbsoluteTime[];
				$scheduledTaskRunning["startTime"]=AbsoluteTime[];
			),
			(*ELSE*)
			(*don't call the function, this is the front end being weird and calling the configure function automatically*)
			(
				(*also this is a specific return so that we don't set the last time called, as that should only be updated when the upload is actually run*)
				Return[];
			)
		];
		(*we set the time that the function was last called as a workaround to prevent the front end from trying to call the function automatically*)
		$lastConfigCall=AbsoluteTime[];
	)
];


ArduinoConfigureDriver[{ihandle_,dhandle_},"PinConfigure"->config_,OptionsPattern[]] := setPinConfigurations[{ihandle,dhandle},config];

ArduinoConfigureDriver[{ihandle_,dhandle_},pinConfigs_Association,OptionsPattern[]] := setPinConfigurations[{ihandle,dhandle},pinConfigs];

ArduinoConfigureDriver[{ihandle_,dhandle_},pin_->mode_,OptionsPattern[]] := setPinConfigurations[{ihandle,dhandle},{pin->mode}];


setPinConfigurations[{ihandle_,dhandle_},pin_->mode_] := setPinConfigurations[{ihandle,dhandle},{pin->mode}];

setPinConfigurations[{ihandle_,dhandle_},config_Association] := setPinConfigurations[{ihandle,dhandle},Normal[config]];

setPinConfigurations[{ihandle_,dhandle_},config_List]:=Module[{pin,configIndex},
	(
		For[configIndex=1,configIndex<=Length[config],configIndex++,
			(*for each config passed in, first verify the configuration and then actually send the configuration to the arduino first*)
			pin=config[[configIndex,1]];
			mode=config[[configIndex,2]];
			Switch[mode,
				"Input"|"DigitalInput",
					(*any pin can be an input or digital input, so just make sure the pin is a valid pin*)
					(
						If[MemberQ[arduinoUnoPins,pin],
							(*THEN*)
							(*the pin is a valid analog pin, so update the internal configuration and configure it on the hardware*)
							(
								$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[pin]]["Direction"]="HardInput";
								DeviceFramework`DeviceDriverOption["Firmata","ConfigureFunction"][{ihandle,dhandle},config[[configIndex]]];
							),
							(*ELSE*)
							(*the pin isn't an analog pin, so issue a message and return $Failed*)
							(
								Message[DeviceConfigure::invalidPin,pin];
								Return[$Failed];
							)
						]
					),
				"AnalogInput",
					(*only analog pins can be analog input, but functionally analog input is the same configuration as any other kind of input*)
					(
						If[MemberQ[arduinoUnoAnalogPins,pin],
							(*THEB*)
							(*the pin is a valid analog pin, so update the internal configuration and configure it on the hardware*)
							(
								$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[pin]]["Direction"]="HardInput";
								DeviceFramework`DeviceDriverOption["Firmata","ConfigureFunction"][{ihandle,dhandle},config[[configIndex]]];
							),
							(*ELSE*)
							(*the pin isn't an analog pin, so issue a message and return $Failed*)
							(
								Message[DeviceConfigure::invalidAnalogPin,pin];
								Return[$Failed];
							)
						]
					),
				"Output"|"DigitalOutput",
					(*any pin can be an output or digital output, so just make sure that the pin is a valid pin*)
					(
						If[MemberQ[arduinoUnoPins,pin],
							(*THEN*)
							(*the pin is valid, so configure it and set the $DeviceStates association*)
							(
								$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[pin]]["Direction"]="HardOutput";
								DeviceFramework`DeviceDriverOption["Firmata","ConfigureFunction"][{ihandle,dhandle},config[[configIndex]]];
							),
							(*ELSE*)
							(*the pin isn't a valid pin to begin with*)
							(
								Message[DeviceConfigure::invalidPin,pin];
								Return[$Failed];
							)
						]
					),
				"PWMOutput"|"AnalogOutput",
					(*only pwm pins can be PWMOutput or analog output, but functionally, analog output is the same configuration as any other kind of output*)
					(
						If[MemberQ[arduinoUnoPWMPins,pin],
							(*THEN*)
							(*the pin is valid, so configure it and set the $DeviceStates association*)
							(
								$DeviceStates[ihandle]["PinConfigurations"][arduinoPinToKey[pin]]["Direction"]="HardOutput";
								DeviceFramework`DeviceDriverOption["Firmata","ConfigureFunction"][{ihandle,dhandle},config[[configIndex]]];
							),
							(*ELSE*)
							(*the pin isn't a valid pin to begin with*)
							(
								Message[DeviceConfigure::notPWMPin,pin];
								Return[$Failed];
							)
						]
					),
				_,
					(
						Message[DeviceConfigure::invalidMode,mode];
						Return[$Failed];
					)
			];
		]
	)
];



ArduinoPropertyGetDriver[dev_,property_]:=Module[{handle},
	(
		If[property === "ArduinoInstallLocation", Return[$arduinoInstallLocation]];
		handle = Quiet[DeviceFramework`DeviceManagerHandle[dev]];
		If[!StringQ[handle], Return[Missing["NotAvailable"]]];
		Switch[property,
			"PinConfigurations",
				Return[Dataset[$DeviceStates[handle,"PinConfigurations"]]]
			,
			"SerialPort",
				Return[$DeviceStates[handle,"SerialPort"]]
			,
			"BoardType",
				Return[$DeviceStates[handle, "BoardType"]]
		]
	)
];


(*only property that can be set is ArduinoInstallLocation*)
ArduinoPropertySetDriver[dev_,property_,value_]:=Module[{},
	(
		Switch[property,
			"ArduinoInstallLocation",
			(
				(*need to check the location first*)
				If[validArduinoInstallLocation[ExpandFileName[value]],
					(*THEN*)
					(*it is valid, set the internal variable*)
					(
						$arduinoInstallLocation=ExpandFileName[value]
					),
					(*ELSE*)
					(*it's not valid, raise a message and don't change it, and lastly return $Failed*)
					(
						Message[DeviceSetProperty::invalidInstallLocation,value];
						$Failed
					)
				];
			),
			True,
			(
				Message[DeviceSetProperty::noWrite,property];
				$Failed
			)
		]
	)
];


(*check setup function message will check if the setup function message needs to be printed off, and if so prints it off and sets the property to false for further calls*)
checkBootFunctionMessage[ihandle_]:=If[TrueQ[$DeviceStates[ihandle]["BootFunction"]],
	Message[DeviceWrite::bootFunction];
	$DeviceStates[ihandle]["BootFunction"]=False;
];


validSchedulingOption[schedOpt_]:=Module[{},
	(
		Switch[schedOpt,
        	_Missing,(*no scheduling option specified, so it's fine by default*)
        	(
        		True
        	),
        	_Integer|_Real, (*run infinitely every x seconds*)
        	(
        		(*ensure that the amount of time to wait before each iteration is within valid unsigned long limits*)
        		(*have to multiply by 1000 because the time is passed in milliseconds*)
        		(schedOpt*1000)>0&&(schedOpt*1000)<2^32
        	),
        	{_Integer|_Real,_Integer}, (*run every x seconds for a maximum of y times*)
        	(
        		(*ensure that the amount of time to wait before each iteration is within valid unsigned long limits*)
        		(*have to multiply by 1000 because the time is passed in milliseconds*)
        		(*same for both of the properties *)
        		And@@((#*1000)>0&&(#*1000)<2^32&/@schedOpt)
        	),
        	{_Integer|_Real}, (*run once in x seconds*)
        	(
        		(*ensure that the amount of time is within a valid unsigned long limits*)
        		(First[schedOpt]*1000)>0&&(First[schedOpt]*1000)<2^32
        	),
        	_,(*any other values are false*)
        	(
        		False
        	)
        ]
	)
];
	

(*addNewFunctionCalls will create the firmata packets necessary for all the functions inside the association passed to it*)
Options[addNewFunctionCalls]={"BootFunction"->None};
addNewFunctionCalls[allFunctions_,functionInfo_Association,OptionsPattern[]]:=Module[{},
	(
		(*if the BootFunction has a name, then increment it to start at 1, rather than 0*)
		functionIDStart=Length[allFunctions]+If[StringQ[OptionValue["BootFunction"]],1,0];
		(*finally if there was a value for BootFunction, we need to drop that from the allFunctions to prevent users from calling the setup function*)
		Return[Join[allFunctions,Association[individualFunctionPacket[functionIDStart++,#]&/@Normal[KeyDrop[functionInfo,OptionValue["BootFunction"]]]]]];
	)
];

(*individualFunctionPacket will make a packet from the function information*)
individualFunctionPacket[functionNumber_,functionName_->function_]:=Module[{},
	(
		(*need to call functionCallPacketSend with the new function numer (aka it's ID), as well as information about the arguments*)
		functionName->
			{
				functionCallPacketSend[functionNumber,
					"LongArgumentNumber"->longArgsNum[function],
					"FloatArgumentNumber"->floatArgsNum[function],
					"StringArgumentNumber"->stringArgsNum[function],
					"LongArrayArgumentNumber"->longArrayArgsNum[function],
					"FloatArrayArgumentNumber"->floatArrayArgsNum[function]
				],
				function
			}
	)
];


(*gets the number of *)
longArgsNum[function_]:=Module[{},
	(
		If[Head[function[[1]]]===Rule,
			(*THEN*)
			(*function has a return type, ignore it*)
			Return[Association[Rule@@@Tally[function[[1,1]]]][Integer]/.Missing[___]->0],
			(*ELSE*)
			(*function doesn't have a return type*)
			Return[Association[Rule@@@Tally[function[[1]]]][Integer]/.Missing[___]->0]
		]
	)	
];

floatArgsNum[function_]:=Module[{},
	(
		If[Head[function[[1]]]===Rule,
			(*THEN*)
			(*function has a return type, ignore it*)
			Return[Association[Rule@@@Tally[function[[1,1]]]][Real]/.Missing[___]->0],
			(*ELSE*)
			(*function doesn't have a return type*)
			Return[Association[Rule@@@Tally[function[[1]]]][Real]/.Missing[___]->0]
		]
	)	
];

stringArgsNum[function_]:=Module[{},
	(
		If[Head[function[[1]]]===Rule,
			(*THEN*)
			(*function has a return type, ignore it*)
			Return[Association[Rule@@@Tally[function[[1,1]]]][String]/.Missing[___]->0],
			(*ELSE*)
			(*function doesn't have a return type*)
			Return[Association[Rule@@@Tally[function[[1]]]][String]/.Missing[___]->0]
		]
	)	
];

longArrayArgsNum[function_]:=Module[{},
	(
		If[Head[function[[1]]]===Rule,
			(*THEN*)
			(*function has a return type, ignore it*)
			Return[Association[Rule@@@Tally[function[[1,1]]]][{Integer}]/.Missing[___]->0],
			(*ELSE*)
			(*function doesn't have a return type*)
			Return[Association[Rule@@@Tally[function[[1]]]][{Integer}]/.Missing[___]->0]
		]
	)	
];

floatArrayArgsNum[function_]:=Module[{},
	(
		If[Head[function[[1]]]===Rule,
			(*THEN*)
			(*function has a return type, ignore it*)
			Return[Association[Rule@@@Tally[function[[1,1]]]][{Real}]/.Missing[___]->0],
			(*ELSE*)
			(*function doesn't have a return type*)
			Return[Association[Rule@@@Tally[function[[1]]]][{Real}]/.Missing[___]->0]
		]
	)
];


(*sendArgs will confirm the arguments are correctly typed and then package them up in a packet*)
sendArgs[function_,arg_]:=Module[{},
	(
		If[ReturnQ[function],
			(*THEN*)
			(*has a return value*)
			If[function[[1,1]]==={},
				(*THEN*)
				(*the function doesn't have any arguments, and thus nothing to send*)
				Return[$Failed],
				(*ELSE*)
				(*check to make sure that the single argument is of the right type by comparing it to the original type signature from the ExternalFunction object*)
				If[function[[1,1,1]]===Head[arg],
					(*THEN*)
					(*it is valid, so return the correct packet*)
					Return[
						(*switch on which type*)
						Switch[Head[arg],
							Real,sendFloatPacket[arg],
							Integer,sendLongPacket[arg],
							String,sendStringPacket[arg],
							(*list means array, so have to check which kind of array*)
							List,Switch[Head[arg[[1]]],
								Real,sendFloatArrayPacket[arg],
								Integer,sendLongArrayPacket[arg]
							],
							(*default case for any other kind of type, return $Failed*)
							_,
							(
								Return[$Failed];
							)
						]
					],
					(*ELSE*)
					(*invalid type for this argument*)
					(
						Return[$Failed]
					)
				]
			],
			(*ELSE*)
			(*doesn't have a return value*)
			If[function[[1]]==={},
				(*THEN*)
				(*the function doesn't have any arguments and thus nothing to send*)
				Return[$Failed],
				(*ELSE*)
				(*the function has arguments, so check tomake sure that the user's argument matches the expected arg type*)
				If[function[[1,1]]===Head[arg],
					(*THEN*)
					(*it does, so return the correct packet*)
					Return[
						Switch[Head[arg],
							Real,sendFloatPacket[arg],
							Integer,sendLongPacket[arg],
							String,sendStringPacket[arg],
							(*list means array, so we need to check which kind of array*)
							List,Switch[Head[arg[[1]]],
								Real,sendFloatArrayPacket[arg],
								Integer,sendLongArrayPacket[arg]
							],
							(*default case of any other head, return $Failed*)
							_,
							(
								Return[$Failed]
							)
						]
					],
					(*ELSE*)
					(*the arg doesn't have the right type, so return $Failed*)
					(
						Return[$Failed]
					)
				]
			]
		]
	)
];


(*sendArgs will confirm the arguments are correctly typed and then package them up in a packet*)
sendArgs[function_,args_List]:=Module[{},
	(
		packet={};
		signature=function[[1]];
		If[ReturnQ[function],
			(*THEN*)
			(*there is a return from the function*)
			If[signature[[1]]==={},
				(*THEN*)
				(*there are no arguments passed to send, so return $Failed*)
				Return[$Failed],
				(*ELSE*)
				(*there are arguments, check to see if the number of arguments is correct first*)
				If[Length[signature[[1]]]===Length[args],
					(*THEN*)
					(*the right number of arguments was passed*)
					(
						(*there are arguments, so for each argument, check it against the expected type defined in DeviceConfigure*)
						For[argNum=1,argNum<=Length[args],argNum++,
							If[Head[args[[argNum]]] === (signature[[1,argNum]]/.{{Real}->List,{Integer}->List}),
								(*THEN*)
								(*the arg is good, we can append it to the list*)
								Switch[Head[args[[argNum]]],
									Real,AppendTo[packet,sendFloatPacket[args[[argNum]]]],
									Integer,AppendTo[packet,sendLongPacket[args[[argNum]]]],
									String,AppendTo[packet,sendStringPacket[args[[argNum]]]],
									List,Switch[Head[args[[argNum,1]]],
										Real,AppendTo[packet,sendFloatArrayPacket[args[[argNum]]]],
										Integer,AppendTo[packet,sendLongArrayPacket[args[[argNum]]]]
									]
								],
								(*ELSE*)
								(*return failed, the user tried to pass invalid args*)
								(
									Return[$Failed];
								)
							]
						]
					),
					(*ELSE*)
					(*wrong number of packets*)
					(
						Return[$Failed];
					)
				]
			],
			(*ELSE*)
			(*the function doesn't have a return type*)
			If[signature==={},
				(*THEN*)
				(*there are no arguments to send for this function, so return $Failed, this function shouldn't have been called in the first place*)
				Return[$Failed],
				(*ELSE*)
				(*there are arguments, check to see if the number of arguments is correct first*)
				If[Length[signature]===Length[args],
					(*THEN*)
					(*the right number of arguments was passed*)
					(
						(*there are arguments, so for each argument, check it against the expected type defined in DeviceConfigure*)
						For[argNum=1,argNum<=Length[args],argNum++,
							If[Head[args[[argNum]]] === (signature[[argNum]]/.{{Real}->List,{Integer}->List}),
								(*THEN*)
								(*the arg is good, we can append it*)
								Switch[Head[args[[argNum]]],
									Real,AppendTo[packet,sendFloatPacket[args[[argNum]]]],
									Integer,AppendTo[packet,sendLongPacket[args[[argNum]]]],
									String,AppendTo[packet,sendStringPacket[args[[argNum]]]],
									List,Switch[Head[args[[argNum,1]]],
										Real,AppendTo[packet,sendFloatArrayPacket[args[[argNum]]]],
										Integer,AppendTo[packet,sendLongArrayPacket[args[[argNum]]]]
									]
								],
								(*ELSE*)
								(*return failed, the user tried to pass invalid args*)
								(
									Return[$Failed];
								)
							]
						]
					),
					(*ELSE*)
					(*wrong number of packets, return $Failed*)
					(
						Return[$Failed];
					)
				]
			]
		];
		(*finally return the packet*)
		Return[packet];
	)
];



(*boolean of whether or not the ExternalFunction has a return type or not*)
ReturnQ[function_]:=Module[{},
	(
		Return[Head[function[[1]]]===Rule]
	)
];

(*takes an ExternalFunction object and returns whether or not that object has any arguments*)
hasArgs[function_]:=Module[{},
	(
		(*return whether the args of the object is equal to {} or not*)
		If[ReturnQ[function],
			(*THEN*)
			(*the function has a return type*)
			(
				Return[Flatten[function[[1,1]]]=!={}];
			),
			(*ELSE*)
			(*the function doesn't have a return type*)
			(
				Return[Flatten[function[[1]]]=!={}];
			)
		]
	)
];


(*FIRMATA PACKET FUNCTIONS*)
(*this will build the packet for a function call, given the function id and the associated paramaters*)
Options[functionCallPacketSend]=
{
	"LongArgumentNumber"->0,
	"FloatArgumentNumber"->0,
	"StringArgumentNumber"->0,
	"LongArrayArgumentNumber"->0,
	"FloatArrayArgumentNumber"->0,
	"SyncTime"->0,
	"IterationCount"->1,
	"RunTimeLength"->0,
	"InitialDelayTime"->0
};
functionCallPacketSend[funcID_,OptionsPattern[]]:=Module[
	{
		start=FromDigits["f0",16],
		functionCall=FromDigits["02",16],
		end=FromDigits["f7",16]
	},
	Flatten[
		{
			(*first byte is sysex start*)
			start,
			(*next byte is the function call add*)
			functionCall,
			(*then the function id*)
			funcID,
			(*then the number of long and float arguments*)
			BitOr[BitShiftLeft[BitAnd[OptionValue["LongArgumentNumber"],15],4],BitAnd[OptionValue["FloatArgumentNumber"],15]],
			(*then the number of string and long array, as well as float array arguments*)
			BitOr[BitShiftLeft[BitAnd[OptionValue["StringArgumentNumber"],15],4],
				BitAnd[BitOr[BitShiftLeft[BitAnd[OptionValue["LongArrayArgumentNumber"],3],2],BitAnd[OptionValue["FloatArrayArgumentNumber"],3]],15]],
			(*next is the timing byte, more info in each individual bit*)
			timingByte= FromDigits[
				{
					(*the top 4 bits of this byte are currently unused for anything*)
					0,
					0,
					0,
					0,
					(*for the low nibble, each bit will tell the arduino to expect another packet with the value of each of the timing information bits*)
					(*then the next bit represents whether or not the iteration count is 0 or not*)
					If[OptionValue["IterationCount"]===1,0,1],
					(*the next bit is whether or not the run time length is 0 or not*)
					If[OptionValue["RunTimeLength"]===0,0,1],
					(*then the time to wait in between calls*)
					If[OptionValue["SyncTime"]===0,0,1],
					(*finally how long to wait after the task is recieved*)
					If[OptionValue["InitialDelayTime"]===0,0,1]
				},2],
			(*finally end the function call packet with a sysex end byte*)
			end,
			(*these last values are only populated if the corresponding bit is high*)
			(*if it is then a long number packet of the number that is to be sent is sent immeadiately after the function call packet*)
			If[OptionValue["InitialDelayTime"]!=0,sendLongPacket[OptionValue["InitialDelayTime"]],{}],
			If[OptionValue["SyncTime"]!=0,sendLongPacket[OptionValue["SyncTime"]],{}],
			If[OptionValue["RunTimeLength"]!=0,sendLongPacket[OptionValue["RunTimeLength"]],{}],
			If[OptionValue["IterationCount"]===1,{},sendLongPacket[OptionValue["IterationCount"]]
			]
	}
	]
];


(*builds a firmata formatted long array packet*)
sendLongArrayPacket[nums_List]:=Module[{},
	Flatten[{
		(*first is the sysex start byte*)
		FromDigits["f0",16],
		(*then the long array identifier byte*)
		FromDigits["07",16],
		(*then the length of the array*)
		Length[nums],
		(*then the actual data bytes for all the numbers in order*)
		ToCharacterCode[ExportString[#,"Integer32",ByteOrdering->1]]&/@nums,
		(*finally the sysex end byte*)
		FromDigits["f7",16]
		}
	]
];

(*builds a firmata formatted float array packet*)
sendFloatArrayPacket[nums_List]:=Module[{},
	Flatten[{
		(*first is the sysex start byte*)
		FromDigits["f0",16],
		(*the next byte is the float array identifier*)
		FromDigits["06",16],
		(*then the length of the array*)
		Length[nums],
		(*then the actual data bytes, for all the numbers in order*)
		ToCharacterCode[ExportString[#,"Real32",ByteOrdering->1]]&/@nums,
		(*then the sysex end byte*)
		FromDigits["f7",16]
		}
	]
];

(*builds a firmata formatted long packet*)
sendLongPacket[num_Integer]:=Module[{},
	Flatten[{
		(*first byte is the sysex start*)
		FromDigits["f0",16],
		(*then the long number byte identifier*)
		FromDigits["05",16],
		(*the next four bytes are actually the data bytes for the number, the arduino is big endian, so we use ByteOrdering->1*)
		ToCharacterCode@ExportString[num,"Integer32",ByteOrdering->1],
		(*finally the sysex end byte*)
		FromDigits["f7",16]
		}
	]
	(*only run the function if the number is within the limits of the arduino*)
]/;Abs[num]<=2^31-1;


(*builds a firmata formatted float packet*)
sendFloatPacket[num_Real]:=Module[{},
	Flatten[{
		(*first byte is the sysex start byte*)
		FromDigits["f0",16],
		(*next is the float number byte identifier*)
		FromDigits["04",16],
		(*then the actual data bytes, with big endian ordering for the arduino*)
		ToCharacterCode@ExportString[num,"Real32",ByteOrdering->1],
		(*finally the sysex end byte*)
		FromDigits["f7",16]
		}
	]
];

(*TODO: check the string for non-ascii values before it gets this far*)
(*builds a firmata formatted string packet*)
sendStringPacket[string_String] := Module[{},
	Flatten[{
		(*first the sysex start byte*)
		FromDigits["f0", 16],
		(*then a string identifier byte*)
		FromDigits["71", 16],
		(*then send the length of this string*)
		Length[Characters[string]],
		(*then send each character in the string as an ascii byte*)
		(*note this does mean that the user should only send strings with ascii bytes, but this is not enforced anywhere*)
		ToCharacterCode[string, "ASCII"],
		(*then finally the sysex end*)
		FromDigits["f7", 16]
		}
	]
];


(*this just checks to make sure that the direction is not "HardInput"*)
okayToWrite[dirValue_]:=Module[{},
	(
		dirValue==="SoftOutput"||dirValue===Default||dirValue==="HardOutput"||dirValue==="SoftInput"
	)
];

(*this just checks to make sure that the direction is not "HardOutput"*)
okayToRead[dirValue_]:=Module[{},
	(
		dirValue==="SoftInput"||dirValue===Default||dirValue==="HardInput"||dirValue==="SoftOutput"
	)
];



pwmize[val_]:=Module[{},
	If[TrueQ[val>255],
		(*THEN*)
		(*upper limit is 255*)
		255,
		(*ELSE*)
		(*check the lower limit of 0*)
		If[TrueQ[val<0],
			(*THEN*)
			(*the value is negative, so just make it 0*)
			0,
			(*the value is within the correct range, so just make it an integer*)
			Floor[val]
		]
	]
];


booleanize[val_]:=Module[{},
	If[val<=0,
		0,
		If[val>=1,
			1,
			(*ELSE*)
			(*it is within 0 to 1, so round it*)
			Round[val]
		]
	]
];


(*this converts the string representation for a analog pin into the numberical version*)
(*TODO: deprecate this, it shouldn't be necessary*)
numericalPin[pin_]:=Module[{},pin/.{"A0"->14,"a0"->14,"a1"->15,"A1"->15,"a2"->16,"A2"->16,"a3"->17,"A3"->17,"a4"->18,"A4"->18,"a5"->19,"A5"->19}];



(*SYMBOLIC C FUNCTIONS*)

(*gets the name of the function*)
symcolicCFuncName[function_CFunction]:=function[[2]];

(*gets the return type of the function, in an ArduinoLink friendly format*)
symbolicCFuncReturnType[function_CFunction]:=function[[1]]/.{
		"long"->Integer,
		"int"->Integer,
		"short"->Integer,
		"byte"->Integer,
		"char"->Integer,
		"float"->Real,
		"double"->Real,
		(*number arrays are not implemented as return types*)
		(*CPointerType["double"]->{Real},
		CPointerType["float"]->{Real},*)
		CPointerType["char"]->String
		(*CPointerType["long"]->{Integer},
		CPointerType["int"]->{Integer},
		CPointerType["short"]->{Integer},
		CPointerType["byte"]->{Integer}*)
		};

(*gets the list of argument types for the function, in an ArduinoLink friendly format*)
symbolicCFuncArgumentTypes[function_CFunction]:=
	function[[3,All,1]]/.{
		"long"->Integer,
		"int"->Integer,
		"short"->Integer,
		"byte"->Integer,
		"char"->Integer,
		"float"->Real,
		"double"->Real,
		CPointerType["double"]->{Real},
		CPointerType["float"]->{Real},
		CPointerType["char"]->String,
		CPointerType["long"]->{Integer},
		CPointerType["int"]->{Integer},
		CPointerType["short"]->{Integer},
		CPointerType["byte"]->{Integer}
		};
		
(*basically just a rule of the argument types to the return type*)
symbolicCFunctionArgType[function_CFunction]:=Module[{},
	(
		symbolicCFuncArgumentTypes[function]->symbolicCFuncReturnType[function]
	)
];

(*wraps all the function information inside the ExternalFunction wrapper that is necessary for ArduinoLink*)
symbolicCFunctionFull[function_CFunction]:=Module[{},
	(
		ExternalFunction[symbolicCFunctionArgType[function],ToCCodeString[function]]
	)
];


(*this function is a wrapper to convert Association style (or ArduinoCode style) specification to ExternalFunction way of converting that the rest of the package depends on*)
(*input will be of the type:<|"Code"->codeString,"ArgumentTypes"->{Integer,{Integer},...},"ReturnType"->Integer}|>*)
	(*(or ArduinoCode[<|"ArgumentTypes"->{Integer,{Integer},...},"ReturnType"->Integer,"Code"->codeString|>])*)
(*output will be of the type: ExternalFunction[{Integer,{Integer}}->Integer,codeString]*)
	(*or of the type ExternalFunction[{Integer,{Integer}},codeString]*)
externalFunctionConvert[arduinoFunc_]:=Module[
	{
		function
	},
	(
		If[Head[arduinoFunc]===Association,
			(*THEN*)
			(*just use what was passed in, it's an association*)
			(
				function = arduinoFunc
			),
			(*ELSE*)
			(*check if the first argument is an association*)
			(
				If[Length[arduinoFunc]===1&&Head[First[arduinoFunc]]===Association,
					(*THEN*)
					(*this is the ArduinoCode case, so use the first argument to that*)
					(
						function = First[arduinoFunc];
					),
					(*ELSE*)
					(*what was passed in wasn't a correct function specficiation*)
					(
						Return[$Failed];
					)
				];
			)
		];
		(*before returning, check that the "Code" option is present*)
		If[Not[KeyExistsQ["Code"][function]],
			(*THEN*)
			(*issue a message about not having any code to use and return $Failed*)
			(
				Message[DeviceConfigure::noFunctionCode];
				Return[$Failed];
			)
		];
		If[KeyExistsQ["ReturnType"][function]&&function["ReturnType"]=!={},
			(*THEN*)
			(*it exists, and it's not an empty list, so include a return type*)
			(
				Return[ExternalFunction[Flatten[{function["ArgumentTypes"]},1]->function["ReturnType"],function["Code"]]/.{_Missing->{}}]
			),
			(*ELSE*)
			(*it doesn't exist, or it is an empty list, so don't include a return type*)
			(
				Return[ExternalFunction[Flatten[{function["ArgumentTypes"]},1],function["Code"]]/.{_Missing->{}}]
			)
		]
	)
];


(*given a channel url such as : "https://channelbroker.wolframcloud.com/users/ijohnson@wolfram.com/yun"*)
(*this will parse out as /users/ijohnson@wolfram.com/yun*)
getChannelURLPath[channel_ChannelObject]:=URLBuild[URLParse[First[channel]["URL"]]["Path"]];


(*this will validate the readmode modes specified for the pins specified*)
(*note that the pins specified here must have already been verified*)
validateReadModes[readModeOption_,pinSpec_]:=Block[
	{readModes,fail,readModeSpec},
	(*for channel send, the readmode option should be of one of these 4 forms:*)
	(*
		"ReadMode" -> {"Analog","Digital",...} => same length as the pinSpec
		"ReadMode" -> "Analog" => all pins are read as analog (note this needs all pins specified need to be analog)
		"ReadMode" -> "Digital" => all pins are digital (this always works for validly specified pins)
		"ReadMode" -> Automatic => all analog pins are read in analog mode, all digital pins are read in digital mode
		"ReadMode" -> Default => same as Automatic
	*)
	(*if the readModeSpec is missing, make it Default*)
	If[MissingQ[readModeOption],
		readModeSpec = Default,
		readModeSpec = readModeOption
	];
	Which[readModeSpec===Default||readModeSpec===Automatic,
		(
			If[MemberQ[arduinoUnoAnalogPins,#],
				(*THEN*)
				(*analog pin, use that mode*)
				(
					(*analog read task on Arduino side is 3*)
					3
				),
				(*ELSE*)
				(*digital pin, use that mode*)
				(
					(*digital read task on Arduino side is 1*)
					1
				)
			]&/@pinSpec
		),
		readModeSpec==="Digital",
		(
			(*digital read task on Arduino side is 1 for all of the pins*)
			Table[1,{Length[pinSpec]}]
		),
		readModeSpec==="Analog",
		(
			(*make sure that the pins specified are actually an analog pin*)
			readModes = If[MemberQ[arduinoUnoAnalogPins,#],
				(*THEN*)
				(*okay to use analog mode*)
				(
					(*analog read task on Arduino side is 3*)
					3
				),
				(*ELSE*)
				(*invalid pin mode, this isn't an analog pin*)
				(
					Message[DeviceExecute::invalidAnalogPin,#];
					(*can't return directly out of a function from inside a function, so set a flag*)
					fail = True;
				)
			]&/@pinSpec;
			(*check the flag*)
			If[TrueQ[fail],
				$Failed,
				readModes
			];
		),
		SubsetQ[{"Analog", "Digital", Automatic, Default}, Union[readModeSpec]],
		(
			
			If[Length[readModeSpec]===Length[pinSpec],
				(*THEN*)
				(*they are the same length, generate them*)
				(
					(*map over the pinSpec ensuring that all of the specs are valid*)
					readModes = Map[
						Switch[#2,
							"Analog",(*ensure that the pin is a valid pinSpec*)
							If[MemberQ[arduinoUnoAnalogPins,#1],
								(*THEN*)
								(*okay to use analog mode*)
								(
									(*analog read task on Arduino side is 3*)
									3
								),
								(*ELSE*)
								(*invalid pin mode, this isn't an analog pin*)
								(
									Message[DeviceExecute::invalidAnalogPin,#1];
									fail = True;
								)
							],
							"Digital", (*any valid pin is a digital pin*)
							1,
							Automatic|Default, (*okay to use analog mode - analog read task on Arduino side is 3, digital read task is 1*)
							MemberQ[arduinoUnoAnalogPins,#1]/.{True->3,False->1}
						]&@@#&,
						Transpose[{pinSpec,readModeSpec}]
					];
					(*if we failed on one of the pins, return $Failed, else return all the readModes*)
					If[TrueQ[fail],
						$Failed,
						readModes
					]
				),
				(*ELSE*)
				(*they aren't the same size, we can't map properly*)
				(
					Message[DeviceConfigure::pinReadmodeShape];
					$Failed
				)
			]
		),
		True,(*invalid read mode*)
		(
			Message[DeviceExecute::invalidPinMode,readModeSpec];
			$Failed
		)
	]
];

(*for the yun custom id generation*)
$usedIDs={};
uniqueRandomAlphanumericString[len_Integer/;len>0]:=Block[
	{id = StringJoin[RandomChoice[Join[Alphabet[], ToString /@ Range[0, 9]], 4]]},
	While[MemberQ[$usedIDs,id],
		id = StringJoin[RandomChoice[Join[Alphabet[], ToString /@ Range[0, 9]], 4]];
	];
	AppendTo[$usedIDs,id];
	id
];

(*check to see if an Arduino driver is already registered before registering the class*)
If[Not[Devices`DeviceAPI`DeviceDump`knownClassQ["Arduino"]],
	(*THEN*)
	(*there aren't any arduino drivers registered, so we should register it one*)
	DeviceFramework`DeviceClassRegister["Arduino",
		"Firmata",
		"ReadFunction"->ArduinoLink`Private`ArduinoReadDriver,
		"WriteFunction"->ArduinoLink`Private`ArduinoWriteDriver,
		"ConfigureFunction"->ArduinoLink`Private`ArduinoConfigureDriverWrapper,
		"PreconfigureFunction"->ArduinoLink`Private`ArduinoPreConfigureDriver,
		"DeregisterOnClose"->False,
		"ExecuteFunction"->ArduinoLink`Private`ArduinoExecuteDriver,
		"Properties"->{
			"PinConfigurations"->ArduinoLink`Private`$DeviceStates,
			"ArduinoInstallLocation"->None,
			"SerialPort"->"",
			"BoardType"->Default
			(*TODO: Implement the scheduled task handler property*)
			(*"ScheduledTaskValues"->{}*)
		},
		"GetPropertyFunction"->ArduinoLink`Private`ArduinoPropertyGetDriver,
		"SetPropertyFunction"->ArduinoLink`Private`ArduinoPropertySetDriver,
		"OpenFunction"->ArduinoLink`Private`ArduinoOpenDriver,
		"DeviceIconFunction"->ArduinoLink`Private`ArduinoCommunityLogo,
		"MakeManagerHandleFunction"->ArduinoLink`Private`ManagerHandleDriver,
		"DriverVersion"->1.1,
		"ReadTimeSeriesFunction"->DeviceRead
	]
	(*ELSE*)
	(*there already is a driver registered, so don't register anything*)
]




End[];

EndPackage[]

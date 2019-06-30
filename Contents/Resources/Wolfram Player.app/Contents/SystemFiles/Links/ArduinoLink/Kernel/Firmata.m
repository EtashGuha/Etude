(* Wolfram Language Package *)

(*==========================================================================================================
			
					FIRMATA DRIVER
			
Author: Ian Johnson
			
Copyright (c) 2015 Wolfram Research. All rights reserved.			


Firmata is a open source standard for Serial communication over a wire. This is a small subset of the standard
implemented for the ArduinoLink package.
CURRENT SUPPORTED BOARDS:
~Arduino Uno

USER ACCESSIBLE FUNCTIONS:
None

==========================================================================================================*)


BeginPackage["Firmata`"]
(* Exported symbols added here with SymbolName::usage *)  

Begin["`Private`"] (* Begin Private Context *) 


(*for other future board models*)
pinToPort[pin_,boardModel_]:=Null;

pinToPort[pin_,"ArduinoUno"]:=Floor[pin/8]/;(pin>=0&&pin<=19);

pinToPort[pin_,"ArduinoUno"]:=None/;pin<0||pin>19;

ArduinoAnalogPins={"A0","A1","A2","A3","A4","A5","a0","a1","a2","a3","a4","a5",14,15,16,17,18,19};

analogPinToNumberPin["ArduinoUno"]=<|"A0"->0,"A1"->1,"A2"->2,"A3"->3,"A4"->4,"A5"->5,
									"a0"->0,"a1"->1,"a2"->2,"a3"->3,"a4"->4,"a5"->5,
									14->0,15->1,16->2,17->3,18->4,19->5|>
arduinoPinToFirmataPin=<|"A0"->14,"A1"->15,"A2"->16,"A3"->17,"A4"->18,"A5"->19,
									"a0"->14,"a1"->15,"a2"->16,"a3"->17,"a4"->18,"a5"->19,
									14->14,15->15,16->16,17->17,18->18,19->19,
									2->2,3->3,4->4,5->5,6->6,7->7,8->8,9->9,10->10,11->11,12->12,13->13|>


DeviceRead::invalidAnalogPin="The pin `1` is not a valid analog input pin"
DeviceRead::readTimeout="Timed out waiting for response from arduino."
DeviceExecute::readTimeout=DeviceRead::readTimeout


Options[FirmataReadDriver]=
{
	"ReadMode"->Automatic,
	"PinAddressing"->"Pin",
	"HiddenBits"->None
};
FirmataReadDriver[{ihandle_,dhandle_},args__,OptionsPattern[]]:=Module[{},
	(
		(*then we will need to validate the arguments before we actually perform the read*)
		If[TrueQ@ValidateReadArgs[args,"PinAddressing"->OptionValue["PinAddressing"]],
			(*THEN*)
			(*the args are valid, we can perform the read*)
			If[args==="raw",
				DeviceFramework`DeviceDriverOption["Serial","ReadBufferFunction"][{ihandle,dhandle},Automatic],
				FirmataRead[{ihandle,dhandle},args,"PinAddressing"->OptionValue["PinAddressing"],"ReadMode"->OptionValue["ReadMode"],"HiddenBits"->OptionValue["HiddenBits"]]
			],
			(*ELSE*)
			(*the arguments are invalid, so return $Failed*)
			Return[$Failed];
		]
	)
]



Options[FirmataRead]=
{
	"PinAddressing"->"Pin",
	"ReadMode"->Automatic,
	"HiddenBits"->None
};
FirmataRead[{ihandle_,dhandle_},pinOrPort_,OptionsPattern[]]:=Module[{},
	(
		Switch[OptionValue["PinAddressing"],
			"Pin",
			(
				(*for pins, if the readmode is automatic, we check if a pin is analog, if it is, then we use that as the readmode*)
				Switch[OptionValue["ReadMode"],
					Automatic,
					(
						(*for pins, if the readmode is automatic, we check if a pin is analog, if it is, then we use that as the readmode*)
						If[MemberQ[ArduinoAnalogPins,pinOrPort],
							(*THEN*)
							(*the pin is an analog pin, so use analog read*)
							Return[FirmataAnalogRead[{ihandle,dhandle},pinOrPort]],
							(*ELSE*)
							(*the pin is a digital pin, so use digital read*)
							Return[FirmataDigitalRead[{ihandle,dhandle},pinOrPort,"PinAddressing"->"Pin","HiddenBits"->OptionValue["HiddenBits"]]];
						]
					),
					"Digital",
					(
						Return[FirmataDigitalRead[{ihandle,dhandle},pinOrPort,"PinAddressing"->"Pin","HiddenBits"->OptionValue["HiddenBits"]]];
					),
					"Analog",
					(
						(*because not all pins can be analog pins, check that here*)
						If[MemberQ[ArduinoAnalogPins,pinOrPort],
							(*THEN*)
							(*it's a valid analog pin, so proceed with the read*)
							(
								Return[FirmataAnalogRead[{ihandle,dhandle},pinOrPort,"PinAddressing"->"Pin"]]
							),
							(*ELSE*)
							(*it's an invalid analog pin, so issue a message and exit*)
							(
								Message[DeviceRead::invalidAnalogPin,pinOrPort];
								Return[$Failed]
							)
						]
					),
					_,
					(
						Message[DeviceRead::pinMode];
						Return[$Failed];
					)
				];
			),
			"Port",
			(
				(*can't address ports in an analog fashion, so raise an error if read mode is analog*)
				If[OptionValue["ReadMode"]===Analog,
					(*THEN*)
					(*this is an error, you cannot address ports in an analog fashion*)
					Message[DeviceRead::readModePort,OptionValue["ReadMode"]],
					(*ELSE*)
					(*the read mode is anything else, just do a normal digital read*)
					Return[FirmataDigitalRead[{ihandle,dhandle},pinOrPort,"HiddenBits"->OptionValue["HiddenBits"]]]
				]
			),
			_,
			(
				Message[DeviceRead::pinMode];
				Return[$Failed];
			)
		]
	)
]



	
	
	

Options[FirmataDigitalRead]=
{
	"PinAddressing"->"Port",
	"HiddenBits"->None
};
FirmataDigitalRead[{ihandle_,dhandle_},port_,OptionsPattern[]]:=Module[
	{
		packet,
		hiddenBits=If[OptionValue["HiddenBits"]===None,0,BitShiftLeft[OptionValue["HiddenBits"],1]]
	},
	(
		(*to perform a digital read, we basically just turn port reporting on, so the format of the packet is:*)
		(*0xDX --- 0xD0 is the identifier for digital port reporting, while the 0x0X is the port to report*)
		(*0x01 or 0x00 --- 1 refers to turning on reporting, while 0 refers to turning off reporting*)
		(*the most recent version of the Firmata protocl specifies that upon enabling reporting, the current value*)
		(*should be sent back, so here we turn on reporting, then wait for the response and return the response*)
		Switch[OptionValue["PinAddressing"],
			"Port",
			(
				packet={BitOr[FromDigits["D0",16],arduinoPinToFirmataPin[port]],BitOr[1,hiddenBits]};
			),
			"Pin",
			(
				(*user requested a single pin, so change the port (which is actually a pin in this circumstance) to a port number*)
				(*we need to see if we are reading an analog pin, as the analog pins are all on their own port*)
				If[MemberQ[Range[14,19],arduinoPinToFirmataPin[port]],
					(*THEN*)
					(*the pin requested is in fact an analog pin, we need to apply different logic to it*)
					(
						packet={BitOr[FromDigits["D0",16],2],BitOr[1,hiddenBits]};
					),
					(*ELSE*)
					(*the pin is a normal digital pin, use normal bitshifting*)
					(
						packet={BitOr[FromDigits["D0",16],pinToPort[arduinoPinToFirmataPin[port],"ArduinoUno"]],BitOr[1,hiddenBits]};
					)
				]
				
			),
			_ ,
			(
				(*default to assuming the value passed was a port*)
				packet={BitOr[FromDigits["D0",16],port],BitOr[1,hiddenBits]};
			)
		];
		(* Print["packet is ",packet]; *)
		(*next, write the packet over the serial connection, after emptying the read buffer*)
		DeviceFramework`DeviceDriverOption["Serial","ReadBufferFunction"][{ihandle,dhandle},Automatic];
		(DeviceFramework`DeviceDriverOption["Serial","WriteFunction"][{ihandle,dhandle},#])&/@packet;
		(*now we have to wait for the response, so first wait for the first byte*)
		$startWaitTime=AbsoluteTime[];
		While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
			(
				(*timeout of 5 seconds*)
				If[AbsoluteTime[]-$startWaitTime>5,
					(
						Message[DeviceRead::readTimeout];
						Return[$Failed]
					)
				];
			)
		];
		(*now wait for the second byte*)
		firstByte = DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}];
		$startWaitTime=AbsoluteTime[];
		While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
			(
				(*timeout of 5 seconds*)
				If[AbsoluteTime[]-$startWaitTime>5,
					(
						Message[DeviceRead::readTimeout];
						Return[$Failed]
					)
				];
			)
		];
		secondByte = DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}];
		(*wait for the last byte*)
		$startWaitTime=AbsoluteTime[];
		While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
			(
				(*timeout of 5 seconds*)
				If[AbsoluteTime[]-$startWaitTime>5,
					(
						Message[DeviceRead::readTimeout];
						Return[$Failed]
					)
				];
			)
		];
		thirdByte = DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}];
		(*Print["recieved ",{firstByte,secondByte,thirdByte}];*)
		(*verify that the packet we got is a digital message packet, it will be if it is 0x9X, where X is the port number*)
		If[BitAnd[firstByte,FromDigits["F0",16]]==FromDigits["90",16],
			(*THEN*)
			(*the packet is good, we have a digital message*)
			(*switch on the original addressing mode we were in, if the user wants the full byte of just the value of one pin*)
			Switch[OptionValue["PinAddressing"],
				"Port",
				(
					(*put all the values in a list*)
					Return[Reverse[IntegerDigits[BitOr[BitShiftLeft[thirdByte,7],BitAnd[secondByte,FromDigits["01111111",2]]],2,8]]];
				),
				"Pin",
				(
					(*user requested a single pin, so only return that bit*)
					(*we need to check if the pin requested was an analog pin, as those pins don't follow the normal logic*)
					If[MemberQ[Range[14,19],arduinoPinToFirmataPin[port]],
						(*THEN*)
						(*the pin is an analog pin, only grab the *)
						(
							(*we subtract 14 from the arduinoPinToFirmataPin to get the bit position for that analog pin, then we add 1 as Mathematica is one indexed*)
							Return[Reverse[IntegerDigits[BitOr[BitShiftLeft[thirdByte,7],BitAnd[secondByte,FromDigits["01111111",2]]],2,8]][[arduinoPinToFirmataPin[port]-13]]];
						),
						(*ELSE*)
						(*the pin is a normal digital pin, so just take the mod of the pin to get the bit position and add 1 to it because 1 indexed*)
						(
							Return[Reverse[IntegerDigits[BitOr[BitShiftLeft[thirdByte,7],BitAnd[secondByte,FromDigits["01111111",2]]],2,8]][[Mod[arduinoPinToFirmataPin[port],8]+1]]];
						)
					]
				),
				_ ,
				(
					(*default to assuming the value passed was a port*)
					Return[Reverse[IntegerDigits[BitOr[BitShiftLeft[thirdByte,7],BitAnd[secondByte,FromDigits["01111111",2]]],2,8]]];
				)
			],
			(*ELSE*)
			(*the packet is corrupted or not what we were looking for, so return $Failed*)
			(
(*				Print["packet is ",{firstByte,secondByte,thirdByte}];
				Print["rest of buffer is ",DeviceFramework`DeviceDriverOption["Serial","ReadBufferFunction"][{ihandle,dhandle},Automatic]];*)
				Return[$Failed];
			)
		]
	)
];


Options[FirmataAnalogRead]=
{
	"PinAddressing"->"Pin"
};
FirmataAnalogRead[{ihandle_,dhandle_},pin_,OptionsPattern[]]:=Module[{packet},
	(
		(*to perform an analog read, we basically just turn pin reporting on, so the format of the packet is:*)
		(*0xCX --- 0xC0 is the identifier for analog pin reporting, while the 0x0X is the port to report*)
		(*0x01 or 0x00 --- 1 refers to turning on reporting, while 0 refers to turning off reporting*)
		(*the most recent version of the Firmata protocl specifies that upon enabling reporting, the current value*)
		(*should be sent back, so here we turn on reporting, then wait for the response and return the response*)
		If[OptionValue["PinAddressing"]=="Pin",
			(*THEN*)
			(*valid addressing mode, continue normally*)
			(
				packet={BitOr[FromDigits["C0",16],analogPinToNumberPin["ArduinoUno"][pin]],1};
				(*Print["sending ",packet];*)
				(*send the packet over the serial connection after clearing the read buffer*)
				DeviceFramework`DeviceDriverOption["Serial","ReadBufferFunction"][{ihandle,dhandle},Automatic];
				(DeviceFramework`DeviceDriverOption["Serial","WriteFunction"][{ihandle,dhandle},#])&/@packet;
				(*now we have to wait for the response, so first wait for the first byte*)
				$startWaitTime=AbsoluteTime[];
				While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
					(
						(*timeout of 5 seconds*)
						If[AbsoluteTime[]-$startWaitTime>5,
							(
								Message[DeviceRead::readTimeout];
								Return[$Failed]
							)
						];
					)
				];
				(*now wait for the second byte*)
				firstByte = DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}];
				$startWaitTime=AbsoluteTime[];
				While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
					(
						(*timeout of 5 seconds*)
						If[AbsoluteTime[]-$startWaitTime>5,
							(
								Message[DeviceRead::readTimeout];
								Return[$Failed]
							)
						];
					)
				];
				secondByte = DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}];
				(*wait for the last byte*)
				$startWaitTime=AbsoluteTime[];
				While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
					(
						(*timeout of 5 seconds*)
						If[AbsoluteTime[]-$startWaitTime>5,
							(
								Message[DeviceRead::readTimeout];
								Return[$Failed]
							)
						];
					)
				];
				thirdByte = DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}];
				(*now, verify that the packet we recieved is an analog message packet, if it is the first byte will be 0xE0*)
				(*Print["recieved ",{firstByte,secondByte,thirdByte}];*)
				If[BitAnd[firstByte,FromDigits["F0",16]]==FromDigits["E0",16],
					(*THEN*)
					(*the packet is good, we have an analog message*)
					(*switch on the original addressing mode we were in, if the user wants the full byte of just the value of one pin*)
					Return[BitOr[BitShiftLeft[thirdByte,7],BitAnd[secondByte,FromDigits["01111111",2]]]],
					(*ELSE*)
					(*the packet is corrupted or not what we were looking for, so return $Failed*)
					Return[$Failed];
				]
			),
			(*ELSE*)
			(*invalid addressing mode, return $Failed*)
			Return[$Failed]
		]
	)
];



Options[FirmataWriteDriver]=
{
	"WriteMode"->Automatic,
	"PinAddressing"->"Pin",
	"HiddenBits"->None
};
FirmataWriteDriver[{ihandle_,dhandle_},args__,OptionsPattern[]]:=Module[{},
	(
		If[TrueQ@ValidateWriteArgs[args,"PinAddressing"->OptionValue["PinAddressing"]],
			(*THEN*)
			(*the args are valid, we can perform the read*)
			FirmataWrite[{ihandle,dhandle},args[[1]],args[[2]],
				"PinAddressing"->OptionValue["PinAddressing"],
				"WriteMode"->OptionValue["WriteMode"]
				,"HiddenBits"->OptionValue["HiddenBits"]],
			(*ELSE*)
			(*the arguments are invalid, so return $Failed*)
			Return[$Failed];
		];
	)
]



Options[FirmataWrite]=
{
	"PinAddressing"->"Pin",
	"WriteMode"->"Digital",
	"HiddenBits"->None
};
FirmataWrite[{ihandle_,dhandle_},pinOrPort_,value_,OptionsPattern[]]:=Module[{},
	(
		Switch[OptionValue["PinAddressing"],
			"Pin",
			(
				(*for pins, if the readmode is automatic, we check if a pin is analog, if it is, then we use that as the readmode*)
				Switch[OptionValue["WriteMode"],
					Automatic,
					(
						(*for pins, if the readmode is automatic, we check if a pin is analog, if it is, then we use that as the readmode*)
						If[MemberQ[ArduinoAnalogPins,pinOrPort],
							(*THEN*)
							(*the pin is an analog pin, so use analog write*)
							(
								FirmataAnalogWrite[{ihandle,dhandle},pinOrPort,value]
							),
							(*ELSE*)
							(*the pin is a digital pin, so use digitial read*)
							FirmataDigitalWrite[{ihandle,dhandle},pinOrPort,value,"PinAddressing"->"Pin","HiddenBits"->OptionValue["HiddenBits"]];
						];
					),
					"Digital",
					(
						FirmataDigitalWrite[{ihandle,dhandle},pinOrPort,value,"PinAddressing"->"Pin","HiddenBits"->OptionValue["HiddenBits"]];
					),
					"Analog",
					(
						FirmataAnalogWrite[{ihandle,dhandle},pinOrPort,value,"PinAddressing"->"Pin"];
					)
				];
			),
			"Port",
			(
				(*can't address ports in an analog fashion, so raise an error if read mode is analog*)
				If[OptionValue["WriteMode"]===Analog,
					(*THEN*)
					(*this is an error, you cannot address ports in an analog fashion*)
					Message[DeviceRead::writeModePort,OptionValue["WriteMode"]],
					(*ELSE*)
					(*the read mode is anything else, just do a normal digital write*)
					FirmataDigitalWrite[{ihandle,dhandle},pinOrPort,value,"HiddenBits"->OptionValue["HiddenBits"]]
				]
			)
		]
	)
]



Options[FirmataAnalogWrite]=
{
	"PinAddressing"->"Pin"
};
FirmataAnalogWrite[{ihandle_,dhandle_},pin_,value_,OptionsPattern[]]:=Module[{},
	(
		(*to perform an analog write, we just send the analog message packet, which is formatted as such:*)
		(*0xEX --- 0xE0 is the identifier for analog message, while the low nibble is the port that is being written to*)
		(*the next two bytes are value bytes, and each byte is limited to 7 bits, so the maximum resolution is 14 bits *)
		If[OptionValue["PinAddressing"]=="Pin",
			(*THEN*)
			(*valid addressing mode, continue normally*)
			(
				(*first check the value to see if it is over 255, which is the max value the arduino supports on its PWM pins*)
				If[value>255,
					Message[DeviceWrite::maxValue,value]
				];
				(*we don't need to quit, the arduino will just max out at 5v with a PWM value of 255*)
				(*to get the two value bytes, take the least significant 7 bits, and the next least significant 7 bits*)
				packet={BitOr[FromDigits["E0",16],pin],BitAnd[FromDigits["1111111",2],value],BitShiftRight[BitAnd[FromDigits["11111110000000",2],value],7]};
				(*Print["analog packet is ",packet];*)
				(*send the packet over the serial connection, after clearing the read buffer*)
				DeviceFramework`DeviceDriverOption["Serial","ReadBufferFunction"][{ihandle,dhandle},Automatic];
				(DeviceFramework`DeviceDriverOption["Serial","WriteFunction"][{ihandle,dhandle},#])&/@packet;
				(*we don't have to wait for any acknowlegement packets from the arduino, so we can just exit*)
			),
			(*ELSE*)
			(*invalid addressing mode, return $Failed*)
			(*TODO: add message here*)
			Return[$Failed];
		]
	)
]


Options[FirmataDigitalWrite]=
{
	"PinAddressing"->"Port",
	"HiddenBits"->None
};
FirmataDigitalWrite[{ihandle_,dhandle_},pinOrPort_,value_,OptionsPattern[]]:=Module[{},
	(
		(*to perform a digital write, we basically just send a digital message packet, so the format of the packet is:*)
		(*0x9X --- 0x90 is the identifier for digital messages, while the 0x0X is the port to write to*)
		(*the next two bytes are value bytes, and each byte is limited to 7 bits, so the last pin in the port is on the last byte *)
		Switch[OptionValue["PinAddressing"],
			"Port",
			(
				packet=
					{
						BitOr[FromDigits["90",16],pinOrPort],
						BitAnd[FromDigits["1111111",2],value],
						BitShiftRight[BitAnd[FromDigits["10000000",2],value],7]
					};
				If[!(OptionValue["HiddenBits"]===None),
					(*THEN*)
					(*there are some hidden bits to pack*)
					packet=ReplacePart[packet,3->BitOr[packet[[3]],BitAnd[BitShiftLeft[OptionValue["HiddenBits"],1],FromDigits["11111110",2]]]];
					(*ELSE*)
					(*there aren't any hidden bits to pack, so don't change the packet*)
				]
			),
			"Pin",
			(
				(*user requested a single pin, so change the port (which is actually a pin in this circumstance) to a port number*)
				(*also need to bit shift up the value to the correct position in the port*)
				bitMask = BitShiftLeft[value,Mod[pinOrPort,8]];
				packet=
					{
						BitOr[FromDigits["90",16],pinToPort[pinOrPort,"ArduinoUno"]],
						BitAnd[FromDigits["1111111",2],bitMask],
						BitAnd[FromDigits["10000000",2],bitMask]
					};
				If[!(OptionValue["HiddenBits"]===None),
					(*THEN*)
					(*there are some hidden bits to pack*)
					packet=ReplacePart[packet,3->BitOr[packet[[3]],BitAnd[BitShiftLeft[OptionValue["HiddenBits"],1],FromDigits["11111110",2]]]];
					(*ELSE*)
					(*there aren't any hidden bits to pack, so don't change the packet*)
				]
			),
			_ ,
			(
				(*default to assuming the value passed was a port*)
				packet=
					{
						BitOr[FromDigits["90",16],pinOrPort],
						BitAnd[FromDigits["1111111",2],value],
						BitAnd[FromDigits["10000000",2],value]
					};
				If[!(OptionValue["HiddenBits"]===None),
					(*THEN*)
					(*there are some hidden bits to pack*)
					packet=ReplacePart[packet,3->BitOr[packet[[3]],BitAnd[BitShiftLeft[OptionValue["HiddenBits"],1],FromDigits["11111110",2]]]];
					(*ELSE*)
					(*there aren't any hidden bits to pack, so don't change the packet*)
				]
			)
		];
		(*Print["digital packet is ",packet];*)
		(*next, write the packet over the serial connection, after clearing out the read buffer*)
		DeviceFramework`DeviceDriverOption["Serial","ReadBufferFunction"][{ihandle,dhandle},Automatic];
		(DeviceFramework`DeviceDriverOption["Serial","WriteFunction"][{ihandle,dhandle},#])&/@packet
	)
]


(*TODO: make sysex packets detection automatic, so it will detect whether or not there are sysex start and sysex end packets in the packet to send*)
Options[FirmataExecuteDriver]=
{
	"SysexPackets"->False,
	"ReturnValue"->False
}
FirmataExecuteDriver[{ihandle_,dhandle_},executePacket_,OptionsPattern[]]:=Module[
	{
		packet ={}
	},
	(
		(*Print["sending ", executePacket];*)
		If[OptionValue["SysexPackets"]===True,
			(*THEN*)
			(*the user doesn't have sysex packets in their message, so add them in*)
			(
				DeviceFramework`DeviceDriverOption["Serial","ReadBufferFunction"][{ihandle,dhandle},Automatic];
				DeviceFramework`DeviceDriverOption["Serial","WriteFunction"][{ihandle,dhandle},#]&/@
					Append[Prepend[executePacket,FromDigits["F0",16]],FromDigits["F7",16]]
			),
			(*ELSE*)
			(*the user does have their own sysex packets in the message*)
			(
				DeviceFramework`DeviceDriverOption["Serial","ReadBufferFunction"][{ihandle,dhandle},Automatic];
				DeviceFramework`DeviceDriverOption["Serial","WriteFunction"][{ihandle,dhandle},#]&/@executePacket;
			)
		];
		If[OptionValue["ReturnValue"]===True,
			(*THEN*)
			(*we are expecting return packets, so we have to wait for those*)
			(
				(*first read off the first packet to determine what kind of packet we are dealing with*)
				$startWaitTime=AbsoluteTime[];
				While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
					(
						(*timeout of 5 seconds*)
						If[AbsoluteTime[]-$startWaitTime>5,
							(
								Message[DeviceExecute::readTimeout];
								Return[$Failed]
							)
						];
					)
				];
				(*now append for the byte to the packet*)
				AppendTo[packet,DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}]];
				(*now check the byte to make sure it is a sysex packet like we were expecting*)
				If[Last[packet]===FromDigits["F0",16],
					(*THEN*)
					(*it is, so we are good to continue reading*)
					(
						(*next we need to read off the next byte to see what kind of sysex packet this is*)
						$startWaitTime=AbsoluteTime[];
						While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
							(
								(*timeout of 5 seconds*)
								If[AbsoluteTime[]-$startWaitTime>5,
									(
										Message[DeviceExecute::readTimeout];
										Return[$Failed]
									)
								];
							)
						];
						(*now append for the byte*)
						AppendTo[packet,DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}]];
						(*now switch on what kind of sysex message this is*)
						Switch[Last[packet],
							4|5,(*this is a standard number packet, which has a total of 5 more bytes to wait for*)
							(
								For[byteIndex = 1, byteIndex <= 5, byteIndex ++,
									(
										$startWaitTime=AbsoluteTime[];
										While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
											(
												(*timeout of 5 seconds*)
												If[AbsoluteTime[]-$startWaitTime>5,
													(
														Message[DeviceExecute::readTimeout];
														Return[$Failed]
													)
												];
											)
										];
										(*now append for the byte*)
										AppendTo[packet,DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}]];
									)
								];
								(*now interpret the packet and send it back to the user*)
								(*Print["packet is ",packet];*)
								Return[readFirmataNumberPacket[packet]];
							),
							FromDigits["71",16],(*this is a string packet*)
							(
								(*a string packet can be arbitrarily long, but the next two bytes tell us how long it is*)
								$startWaitTime=AbsoluteTime[];
								While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
									(
										(*timeout of 5 seconds*)
										If[AbsoluteTime[]-$startWaitTime>5,
											(
												Message[DeviceExecute::readTimeout];
												Return[$Failed]
											)
										];
									)
								];
								(*now append for the byte*)
								AppendTo[packet,DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}]];
								(*now wait for the next one*)
								$startWaitTime=AbsoluteTime[];
								While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
									(
										(*timeout of 5 seconds*)
										If[AbsoluteTime[]-$startWaitTime>5,
											(
												Message[DeviceExecute::readTimeout];
												Return[$Failed]
											)
										];
									)
								];
								(*now append for the byte*)
								AppendTo[packet,DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}]];
								(*now combine the two bytes into a number*)
								stringLength = BitOr[BitAnd[packet[[-2]],127],BitShiftLeft[packet[[-1]],7]];
								(*now we know how many more bytes to expect, so read those off in a for loop*)
								(*note the end sysex message is still on the buffer, so we have to read that off as well, hence the stringLength + 1*)
								For[byteIndex = 1, byteIndex <= stringLength+1, byteIndex ++,
									(
										$startWaitTime=AbsoluteTime[];
										While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
											(
												(*timeout of 5 seconds*)
												If[AbsoluteTime[]-$startWaitTime>5,
													(
														Message[DeviceExecute::readTimeout];
														Return[$Failed]
													)
												];
											)
										];
										(*now append for the byte*)
										AppendTo[packet,DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}]];
									)
								];
								(*now we have the full string, so we can just take the relevant part and return that as a string*)
								(*Print["packet is ",packet];*)
								Return[FromCharacterCode[packet[[5;;-2]]]];
							)
						]
					),
					(*ELSE*)
					(*it is not, so we got junk data, flush the read buffer and return $Failed*)
					(
						(*Print["junk data"];*)
						DeviceFramework`DeviceDriverOption["Serial","ReadBufferFunction"][{ihandle,dhandle},Automatic];
						Return[$Failed];
					)
				];
			),
			(*ELSE*)
			(*there is no return packet expected, so we can just return to the user*)
			Return[];
		]
	)
];


FirmataConfigureDriver[{ihandle_,dhandle_},pin_->mode_,OptionsPattern[]]:=Module[{},
	(
		(*Print[pin->mode];*)
		(*first confirm that the given pin supports the given mode*)
		Switch[mode,
			"Input"|"AnalogInput"|"DigitalInput",
			(
				(*any pin except the serial pins can be input*)
				(*the packet for a config task contains the first byte 0xF4, and the next byte is the pin number, the final byte is 0 for input*)
				packet={FromDigits["F4",16],numericalPin[pin],0};
			),
			"Output"|"AnalogOutput"|"PWMOutput"|"DigitalOutput",
			(
				(*any pin except the serial pins can be output*)
				(*the packet for a config task contains the first byte 0xF4, and the next byte as the pin number, with the final byte is 1 for output*)
				packet={FromDigits["F4",16],numericalPin[pin],1};
			),
			_,
			(
				Message[DeviceConfigure::invalidMode,mode];
				Return[$Failed];
			)
		];
		(*Print["packet is ",packet];*)
		(*clear the read buffer before sending the packet*)
		DeviceFramework`DeviceDriverOption["Serial","ReadBufferFunction"][{ihandle,dhandle},Automatic];
		DeviceFramework`DeviceDriverOption["Serial","WriteFunction"][{ihandle,dhandle},#]&/@packet;
	)
]



(*TODO: implement these two functions to make sure that the arguments passed to the read and write functions are valid, so that 
unknown bytes aren't sent over the serial port, currently that checking is in the Arduino driver*)
(*this makes sure basically that only analog pins are attempted to read from in an analog manner*)
Options[ValidateReadArgs]=
{
	"PinAddressing"->"Pin"
};
ValidateReadArgs[args_,OptionsPattern[]]:=Module[{},
	(
		True
	)
];

(*this makes sure that analog writes are only allowed to PWM pins*)
Options[ValidateWriteArgs]=
{
	"PinAddressing"->"Pin"
};
ValidateWriteArgs[args_,OptionsPattern[]]:=Module[{},
	(
		True
	)
];




DeviceFramework`DeviceClassRegister[
	"Firmata",
	(*inherits from the Serial driver super class*)
	"Serial",
	"ReadFunction"->FirmataReadDriver,
	"WriteFunction"->FirmataWriteDriver,
	"ExecuteFunction"->FirmataExecuteDriver,
	"ConfigureFunction"->FirmataConfigureDriver
];

echo=(Print@#;#)&;

(*readBytes is currently not utilized, for a future release when we care about a scheduled tasks's return value*)

(*********************************UNIMPLEMENTED*******************************)

(*=========================================================================================================
============================ READBYTES ====================================================================
===========================================================================================================

readBytes will read bytes off of the serial buffer, interpreting groups of them as packets. It will only
read as many bytes/packets as it needs to find the requested type of response on the Serial Buffer. This 
typically would not be more than one extra packet before the correct one is found, but it is general enough
to grab an arbitrary amount of other packets until it finds the one it is looking for. If it ever times out
waiting for another byte, it will still return a normal association, with whatever value it was currently
expecting or trying to finish building as $Failed, so other data that was valid that was found previously 
is still returned to the user. It defaults to looking for analog or digital read packets, but can also be
configured to look for data value packets (wrapped inside sysex packets). Any "junk" data it finds, i.e. 
sysex packets that do not have a recognized header, random bytes that do not correspond to anything, and 
other such anomalies are appended into the "UnknownPacket" key of the returned association

===========================================================================================================
=====================ALGORITHM=============================================================================
===========================================================================================================

(*TODO: finish typing up the alogrithmic logic for this function*)
The general algorithm for this function is as follows:
===========================================================================================================
Step 1.		First step
===========================================================================================================
Step 2. 	Second step
===========================================================================================================
Step 3. 	Third step
===========================================================================================================


===========================================================================================================
=============================PARAMETERS====================================================================
===========================================================================================================

	expectedData - expectedData is the type of data that is expected on the serial port. Possible values
					are "ReadValue", and "DataValue". ReadValue is the type of packet that is sent back
					when an analog or digital read is performed. DataValue will expect a sysex number 
					packet. This argument can be omitted, in which case expectedData defaults to ReadValue

===========================================================================================================


===========================================================================================================
================================RETURN=====================================================================
===========================================================================================================
	
	Returns an association of the data elements it found. If it times out trying to find any data it will
	return an association with all keys going to $Failed. If it finds any data it can recognize, it will 
	timestamp the data, as a DateObject -> (interpreted data), and append it to a list. So the form of the 
	association is as follows:
	<|
		"ReadValue"->{time1->num1,time2->num2,...},
		"DataValue"->{time1->num1,time2->num2,...},
		"UnknownPacket"->{time1->packet1,time2->packet2,...}
	|>

===========================================================================================================


===========================================================================================================
==================================OPTIONS==================================================================
===========================================================================================================

	N/A

===========================================================================================================


=================================FUNCTION CODE FOLLOWS=====================================================
===========================================================================================================
=========================================================================================================*)

(*the default is for this to be expecting a read value packet, the only other valid possibility is "DataValue" *)
(*readBytes[{ihandle_,dhandle_},expectedData_:"ReadValue"]:=Module[
	{
		(*notComplete is for reading an unknown number of individual bytes*)
		notComplete=True,
		(*notFound is for reading an unknown number of individual packets (each composed of individual bytes)*)
		notFound=True,
		sysexPacket={},
		returnAssociation=
		<|
			"UnknownPacket"->{},
			"ReadValue"->{},
			"DataValue"->{}
		|>
	},
	(
		(*this function will read bytes from the serial port, handling the case where there are more bytes than expected*)
		(*first, wait until there are bytes in the buffer, returning $Failed if it times out*)
		$startWaitTime=AbsoluteTime[];
		While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
			(
				(*timeout of 5 seconds - set all fields to failed then return the returnAssociation*)
				If[AbsoluteTime[]-$startWaitTime>5,
					(*THEN*)
					(*we didn't find any data bytes on the serial line, so set all fields to $Failed and return that*)
					(
						Return[
							<|
								"UnknownPacket"->{DateObject[]->$Failed},
								"ReadValue"->{DateObject[]->$Failed},
								"DataValue"->{DateObject[]->$Failed}
							|>
						];
					)
				];
			)
		];
		(*now there's data available, so read off the first byte and switch on the high nibble of that byte*)
		firstDataByte=DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}];
		Switch[BitAnd[firstDataByte,FromDigits["F0",16]],
			FromDigits["F0",16],
			(*sysex message packet - there can be an arbitrary number of more bytes, we will know once we find an end sysex byte that the packet is complete*)
			(
				(*first add the original first data byte to the sysex packet*)
				AppendTo[sysexPacket,firstDataByte];
				(*now go into the loop, grabbing all of the bytes until a sysex end is found*)
				While[notComplete,
					$startWaitTime=AbsoluteTime[];
					While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
						(
							(*timeout of 5 seconds - put the data we have so far into the Unknown packet key, and then return that*)
							If[AbsoluteTime[]-$startWaitTime>5,
								(*THEN*)
								(*we didn't find any more data on the serial line, but we also don't have enough data for a full packet of any kind, so just put it in Unknown packet key*)
								(
									returnAssociation=Append[returnAssociation["UnkownPacket"],DateObject[]->sysexPacket];
									Return[returnAssociation];
								)
							];
						)
					];
					(*now read off the byte into the sysex packet*)
					AppendTo[sysexPacket,DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}]];
					If[Last[sysexPacket]===FromDigits["F7",16],
						(*THEN*)
						(*we are at the end of the byte, so set notComplete to false to exit the loop*)
						notComplete=False;
					];
				];
				(*now we have a full sysex packet, so make sure it is a data number packet*)
				If[sysexPacket[[2]]===FromDigits["04",16]||sysexPacket[[2]]===FromDigits["05",16],
					(*THEN*)
					(*the packet is a valid data packet, so interpret it and add it to the return association*)
					(
						returnAssociation["DataValue"]=Append[returnAssociation["DataValue"],DateObject[]->readFirmataNumberPacket[sysexPacket]];
					),
					(*ELSE*)
					(*the sysex packet is something else entirely, so just throw it in the unknown part of the association*)
					(
						returnAssociation["UnknownPacket"]=Append[returnAssociation["UnknownPacket"],DateObject[]->sysexPacket];
					)
				];
				(*now that we have read off a sysex packet, we have to check if we are expecting a different kind of packet, if we are we have to wait for that*)
				If[expectedData==="ReadValue",
					(*THEN*)
					(*we didn't find the packet we were looking for, so enter a while loop that continually loops until we get the packet we were looking for*)
					(
						While[notFound,
							(*first wait for the next byte to arrive to determine what kind of packet this is*)
							(
								$startWaitTime=AbsoluteTime[];
								While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
									(
										(*timeout of 5 seconds - note that we don't just return $Failed, because we didn't entirely fail,*)
										(* we just got an unexpected packet and no sign of the packet we were looking for, so append $Failed to the read value and then return that*)
										If[AbsoluteTime[]-$startWaitTime>5,
											(*THEN*)
											(*append $Failed to the association and return the association*)
											(
												returnAssociation["ReadValue"]=Append[returnAssociation["ReadValue"],DateObject[]->$Failed];
												Return[returnAssociation];
											)
										];
									)
								];
								(*now we can read off the first byte*)
								firstDataByte=DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}];
								(*make sure that the data byte is in fact a read packet*)
								Switch[BitAnd[firstDataByte,FromDigits["F0",16]],
									FromDigits["F0",16],
									(*it's a sysex packet, so we have to go through the arbitrary length wait for the rest of the packet*)
									(
										(*first add the original first data byte to the sysex packet*)
										AppendTo[sysexPacket,firstDataByte];
										(*now go into the loop, grabbing all of the bytes until a sysex end is found*)
										While[notComplete,
											$startWaitTime=AbsoluteTime[];
											While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
												(
													(*timeout of 5 seconds - but note we don't just return $Failed, we need to add what data we have to the unknown packet key, and add $Failed to the read value*)
													If[AbsoluteTime[]-$startWaitTime>5,
														(*THEN*)
														(*timeout, so add what data we have to the Unknown packet key, then add $Failed to the read value key, then return the returnAssociation*)
														(
															returnAssociation["UnknownPacket"]=Append[returnAssociation["UnknownPacket"],DateObject[]->sysexPacket];
															returnAssociation["ReadValue"]=Append[returnAssociation["ReadValue"],DateObject[]->$Failed];
															Return[returnAssociation];
														)
													];
												)
											];
											(*the byte is there, now read off the byte into the sysex packet*)
											AppendTo[sysexPacket,DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}]];
											If[Last[sysexPacket]===FromDigits["F7",16],
												(*THEN*)
												(*we are at the end of the packet, so set notComplete to false to exit the byte gathering loop, but not the packet gathering loop*)
												notComplete=False;
											];
										];
										(*now we have read off an entire sysex packet, so check to see if this sysex packet is a data number packet*)
										If[sysexPacket[[2]]===FromDigits["04",16]||sysexPacket[[2]]===FromDigits["05",16],
											(*THEN*)
											(*the packet is a valid data packet, so interpret it and add it to the return association*)
											(
												returnAssociation["DataValue"]=Append[returnAssociation["DataValue"],DateObject[]->readFirmataNumberPacket[sysexPacket]];
											),
											(*ELSE*)
											(*the sysex packet is something else entirely, so just throw it in the unknown part of the association*)
											(
												returnAssociation["UnknownPacket"]=Append[returnAssociation["UnknownPacket"],DateObject[]->sysexPacket];
											)
										];
										(*we just finished reading off another packet sysex packet, so we know we didn't find what we were looking for and we can just continue to the next packet*)
										Continue[];
									),
									FromDigits["E0",16],
									(*analog read packet, there will be a total of 3 bytes, the first one has already been read off*)
									(
										(*first wait to read off the second byte until it is actually there*)
										$startWaitTime=AbsoluteTime[];
										While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
											(
												(*timeout of 5 seconds - we started to find an analog read packet, but it failed, so appent $Failed to the ReadValue and return that*)
												If[AbsoluteTime[]-$startWaitTime>5,
													(*THEN*)
													(*append $Failed to ReadValue key and return that*)
													returnAssociation["ReadValue"]=Append[returnAssociation["ReadValue"],DateObject[]->$Failed];
													Return[returnAssociation];
												];
											)
										];
										(*we can now read off the byte*)
										secondDataByte=DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}];
										(*wait for the last byte*)
										$startWaitTime=AbsoluteTime[];
										While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
											(
												(*timeout of 5 seconds - we started to find an analog read packet, but it failed, so appent $Failed to the ReadValue and return that*)
												If[AbsoluteTime[]-$startWaitTime>5,
													(*THEN*)
													(*append $Failed to ReadValue key and return that*)
													returnAssociation["ReadValue"]=Append[returnAssociation["ReadValue"],DateObject[]->$Failed];
													Return[returnAssociation];
												];
											)
										];
										thirdDataByte=DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}];
										(*now that we have all the data bytes, we have to convert it to an actual number*)
										readValue=analogReadPacket[{firstDataByte,secondDataByte,thirdDataByte}];
										(*append the value to the association*)
										returnAssociation["ReadValue"]=Append[returnAssociation["ReadValue"],DateObject[]->readValue];
										(*we were looking for a read packet and we found it, so we are good to return now*)
										Return[returnAssociation];
									),
									FromDigits["90",16],
									(*digital read packet, there will be a total of 3 bytes, and the first one has already been read off*)
									(
										(*first wait to read off the second byte until it is actually there*)
										$startWaitTime=AbsoluteTime[];
										While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
											(
												(*timeout of 5 seconds - we started to find an analog read packet, but it failed, so appent $Failed to the ReadValue and return that*)
												If[AbsoluteTime[]-$startWaitTime>5,
													(*THEN*)
													(*append $Failed to ReadValue key and return that*)
													returnAssociation["ReadValue"]=Append[returnAssociation["ReadValue"],DateObject[]->$Failed];
													Return[returnAssociation];
												];
											)
										];
										(*we can now read off the byte*)
										secondDataByte=DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}];
										(*wait for the last byte*)
										$startWaitTime=AbsoluteTime[];
										While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
											(
												(*timeout of 5 seconds - we started to find an analog read packet, but it failed, so appent $Failed to the ReadValue and return that*)
												If[AbsoluteTime[]-$startWaitTime>5,
													(*THEN*)
													(*append $Failed to ReadValue key and return that*)
													returnAssociation["ReadValue"]=Append[returnAssociation["ReadValue"],DateObject[]->$Failed];
													Return[returnAssociation];
												];
											)
										];
										thirdDataByte=DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}];
										(*now that we have all the data bytes, we have to convert it to an actual number*)
										readValue=digitalReadPacket[{firstDataByte,secondDataByte,thirdDataByte}];
										(*append the value to the association*)
										returnAssociation["ReadValue"]=Append[returnAssociation["ReadValue"],DateObject[]->readValue];
										(*we were looking for a read packet and we found it, so we are good to return now*)
										Return[returnAssociation];
									),
									_,
									(*this is for all other cases of the high nibble of the first byte*)
									(*note it is likely that there is more data on the buffer, just we don't know what it is, so just write this individual byte into the Unknown packet key*)
									(
										returnAssociation["UnknownPacket"]=Append[returnAssociation["UnknownPacket"],DateObject[]->firstDataByte];
										Continue[];
									)
								]
							)
						]
					),
					(*ELSE*)
					(*just confirm we are looking for a data value packet, if not, well still return as more handling of that is not currently handled*)
					(
						If[expectedData==="DataValue",
							(*THEN*)
							(*we found it, so we can just return*)
							Return[returnAssociation],
							(*ELSE*)
							(*we didn't find it, but it also wasn't a data value we were looking for, so just return anyways*)
							(*TODO: implement more cases of expectedData*)
							Return[returnAssociation]
						]
					)
				]
			),
			FromDigits["E0",16],
			(*analog read packet, there will be a total of 3 bytes, and the first one has already been read off*)
			(
				$startWaitTime=AbsoluteTime[];
				While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
					(
						(*timeout of 5 seconds*)
						If[AbsoluteTime[]-$startWaitTime>5,Return[$Failed]];
					)
				];
				(*we can now read off the byte*)
				secondDataByte=DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}];
				(*wait for the last byte*)
				$startWaitTime=AbsoluteTime[];
				While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
					(
						(*timeout of 5 seconds*)
						If[AbsoluteTime[]-$startWaitTime>5,Return[$Failed]];
					)
				];
				thirdDataByte=DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}];
				(*now that we have all the data bytes, we have to convert it to an actual number*)
				readValue=analogReadPacket[{firstDataByte,secondDataByte,thirdDataByte}];
				(*append the value to the association*)
				returnAssociation["ReadValue"]=Append[returnAssociation["ReadValue"],DateObject[]->readValue];
				(*if that was what we were looking for, then we can return*)
				If[expectedData==="ReadValue",
					(*THEN*)
					(*we were indeed looking for a read value, and found it so just return the returnAssociation*)
					(
						Return[returnAssociation];
					),
					(*ELSE*)
					(*we didn't find what we were looking for, so continue searching*)
					(
						While[notFound,
							(*first wait for the next byte to arrive to determine what kind of packet this is*)
							(
								$startWaitTime=AbsoluteTime[];
								While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
									(
										(*timeout of 5 seconds - note that we don't just return $Failed, because we didn't entirely fail,*)
										(* we just got an unexpected packet and no sign of the packet we were looking for, so append $Failed to the data value and then return that*)
										If[AbsoluteTime[]-$startWaitTime>5,
											(*THEN*)
											(*append $Failed to the association and return the association*)
											(
												returnAssociation["DataValue"]=Append[returnAssociation["DataValue"],DateObject[]->$Failed];
												Return[returnAssociation];
											)
										];
									)
								];
								(*now we can read off the first byte*)
								firstDataByte=DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}];
								(*make sure that the data byte is in fact a read packet*)
								Switch[BitAnd[firstDataByte,FromDigits["F0",16]],
									FromDigits["F0",16],
									(*it's a sysex packet, so we have to go through the arbitrary length wait for the rest of the packet*)
									(
										(*first add the original first data byte to the sysex packet*)
										AppendTo[sysexPacket,firstDataByte];
										(*now go into the loop, grabbing all of the bytes until a sysex end is found*)
										While[notComplete,
											$startWaitTime=AbsoluteTime[];
											While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
												(
													(*timeout of 5 seconds - but note we don't just return $Failed, we need to add what data we have to the unknown packet key, and add $Failed to the read value*)
													If[AbsoluteTime[]-$startWaitTime>5,
														(*THEN*)
														(*timeout, so add what data we have to the Unknown packet key, then add $Failed to the data value key, then return the returnAssociation*)
														(
															returnAssociation["UnknownPacket"]=Append[returnAssociation["UnknownPacket"],DateObject[]->sysexPacket];
															returnAssociation["DataValue"]=Append[returnAssociation["DataValue"],DateObject[]->$Failed];
															Return[returnAssociation];
														)
													];
												)
											];
											(*the byte is there, now read off the byte into the sysex packet*)
											AppendTo[sysexPacket,DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}]];
											If[Last[sysexPacket]===FromDigits["F7",16],
												(*THEN*)
												(*we are at the end of the packet, so set notComplete to false to exit the byte gathering loop, but not the packet gathering loop*)
												notComplete=False;
											];
										];
										(*now we have read off an entire sysex packet, so check to see if this sysex packet is a data number packet*)
										If[sysexPacket[[2]]===FromDigits["04",16]||sysexPacket[[2]]===FromDigits["05",16],
											(*THEN*)
											(*the packet is a valid data packet, so interpret it and add it to the return association, then return the association*)
											(
												returnAssociation["DataValue"]=Append[returnAssociation["DataValue"],DateObject[]->readFirmataNumberPacket[sysexPacket]];
												Return[returnAssociation];
											),
											(*ELSE*)
											(*the sysex packet is something else entirely, so just throw it in the unknown part of the association*)
											(
												returnAssociation["UnknownPacket"]=Append[returnAssociation["UnknownPacket"],DateObject[]->sysexPacket];
											)
										];
										(*we either just read off a valid data value packet and returned, or we read off junk data and put it in UnknownPacket*)
										(*to get to this point, we have to continue, as we wouldn't have found it yet*)
										Continue[];
									),
									FromDigits["E0",16],
									(*analog read packet, there will be a total of 3 bytes, the first one has already been read off*)
									(
										(*first wait to read off the second byte until it is actually there*)
										$startWaitTime=AbsoluteTime[];
										While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
											(
												(*timeout of 5 seconds - we started to find an analog read packet, but it failed, 
												so append $Failed to the ReadValue and return that, and add $Failed to the data value packet*)
												If[AbsoluteTime[]-$startWaitTime>5,
													(*THEN*)
													(*append $Failed to ReadValue key and return that*)
													returnAssociation["ReadValue"]=Append[returnAssociation["ReadValue"],DateObject[]->$Failed];
													returnAssociation["DataValue"]=Append[returnAssociation["DataValue"],DateObject[]->$Failed];
													Return[returnAssociation];
												];
											)
										];
										(*we can now read off the byte*)
										secondDataByte=DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}];
										(*wait for the last byte*)
										$startWaitTime=AbsoluteTime[];
										While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
											(
												(*timeout of 5 seconds - we started to find an analog read packet, but it failed,*) 
												(*so append $Failed to the ReadValue and return that, and add $Failed to the data value packet*)
												If[AbsoluteTime[]-$startWaitTime>5,
													(*THEN*)
													(*append $Failed to ReadValue key and return that*)
													returnAssociation["ReadValue"]=Append[returnAssociation["ReadValue"],DateObject[]->$Failed];
													returnAssociation["DataValue"]=Append[returnAssociation["DataValue"],DateObject[]->$Failed];
													Return[returnAssociation];
												];
											)
										];
										thirdDataByte=DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}];
										(*now that we have all the data bytes, we have to convert it to an actual number*)
										readValue=analogReadPacket[{firstDataByte,secondDataByte,thirdDataByte}];
										(*append the value to the association*)
										returnAssociation["ReadValue"]=Append[returnAssociation["ReadValue"],DateObject[]->readValue];
										(*we still haven't found our data value packet, so continue to the next packet*)
										Continue[];
									),
									FromDigits["90",16],
									(*digital read packet, there will be a total of 3 bytes, and the first one has already been read off*)
									(
										(*first wait to read off the second byte until it is actually there*)
										$startWaitTime=AbsoluteTime[];
										While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
											(
												(*timeout of 5 seconds - we started to find an analog read packet, but it failed, 
												so append $Failed to the ReadValue and return that, and add $Failed to the data value packet*)
												If[AbsoluteTime[]-$startWaitTime>5,
													(*THEN*)
													(*append $Failed to ReadValue key and return that*)
													returnAssociation["ReadValue"]=Append[returnAssociation["ReadValue"],DateObject[]->$Failed];
													returnAssociation["DataValue"]=Append[returnAssociation["DataValue"],DateObject[]->$Failed];
													Return[returnAssociation];
												];
											)
										];
										(*we can now read off the byte*)
										secondDataByte=DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}];
										(*wait for the last byte*)
										$startWaitTime=AbsoluteTime[];
										While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
											(
												(*timeout of 5 seconds - we started to find an analog read packet, but it failed, 
												so append $Failed to the ReadValue and return that, and add $Failed to the data value packet*)
												If[AbsoluteTime[]-$startWaitTime>5,
													(*THEN*)
													(*append $Failed to ReadValue key and return that*)
													returnAssociation["ReadValue"]=Append[returnAssociation["ReadValue"],DateObject[]->$Failed];
													returnAssociation["DataValue"]=Append[returnAssociation["DataValue"],DateObject[]->$Failed];
													Return[returnAssociation];
												];
											)
										];
										thirdDataByte=DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}];
										(*now that we have all the data bytes, we have to convert it to an actual number*)
										readValue=digitalReadPacket[{firstDataByte,secondDataByte,thirdDataByte}];
										(*append the value to the association*)
										returnAssociation["ReadValue"]=Append[returnAssociation["ReadValue"],DateObject[]->readValue];
										(*we still haven't found our data value packet, so continue to the next packet*)
										Continue[];
									),
									_,
									(*this is for all other cases of the high nibble of the first byte*)
									(*note it is likely that there is more data on the buffer, just we don't know what it is, so just write this individual byte into the Unknown packet key*)
									(
										returnAssociation["UnknownPacket"]=Append[returnAssociation["UnknownPacket"],DateObject[]->firstDataByte];
										Continue[];
									)
								]
							)
						]
					)
				];
			),
			FromDigits["90",16],
			(*digital read packet, there will be a total of 3 bytes, and the first one has already been read off*)
			(
				$startWaitTime=AbsoluteTime[];
				While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
					(
						(*timeout of 5 seconds*)
						If[AbsoluteTime[]-$startWaitTime>5,Return[$Failed]];
					)
				];
				(*we can now read off the byte*)
				secondDataByte=DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}];
				(*wait for the last byte*)
				$startWaitTime=AbsoluteTime[];
				While[Not[DeviceFramework`DeviceDriverOption["Serial","ExecuteFunction"][{ihandle,dhandle},"SerialReadyQ"]],
					(
						(*timeout of 5 seconds*)
						If[AbsoluteTime[]-$startWaitTime>5,Return[$Failed]];
					)
				];
				thirdDataByte=DeviceFramework`DeviceDriverOption["Serial","ReadFunction"][{ihandle,dhandle}];
				(*now that we have all the data bytes, we have to convert it to an actual number*)
				readValue=digitalReadPacket[{firstDataByte,secondDataByte,thirdDataByte}];
				(*append tthe value to the association*)
				returnAssociation["ReadValue"]=Append[returnAssociation["ReadValue"],DateObject[]->readValue];
				(*if that was what we were looking for, then we can return*)
			),
			_,
			(*for any other mode, append it to the unknown packet key, and keep looking*)
			(
				returnAssociation["UnknownPacket"]=Append[returnAssociation["UnknownPacket"],DateObject[]->firstDataByte];
				(*wait for another byte on the serial buffer*)
			)
		]
	)
]*)



(*FIRMATA PACKET UTILTIES*)

(*TODO: implement this function*)
(*for interpreting digital read packets*)
(*digitalReadPacket[packet_List]:=Module[{},
	(
		Null
	)
];
*)

(*TODO: implement the bitshifting and such for this function*)
(*for interpreting analog read packets*)
(*analogReadPacket[packet_List]:=Module[{},
	(
		If[BitAnd[First[packet],FromDigits["F0",16]]==FromDigits["E0",16],
			(*THEN*)
			(*it is a valid analog read packet, so read in the value from the last two bytes*)
			(
				Return[];
			),
			(*ELSE*)
			(*it is invalid, return $Failed*)
			(
				Return[$Failed];
			)
		]
	)
]/;Length[packet]===3 (*only run if it seems like a legit packet*)
*)

(*for sending integer packets*)
sendLongPacket[num_Integer] := Module[{},
	Flatten[{
		FromDigits["f0", 16], (*sysex start*)
		FromDigits["05", 16], (*long number identifier*)
		ToCharacterCode@ExportString[num, "Integer32", ByteOrdering -> 1], (*actual data bytes*)
		FromDigits["f7", 16]} (*end sysex*)
	]
]/;Abs[num] <= 2^31 - 1 (*only run it if it is within the correct bounds*);


(*for sending floating point number packets with firmata protocol*)
sendFloatPacket[num_Real] := Module[{},
	Flatten[
		{
			FromDigits["F0",16],(*sysex start*)
			FromDigits["04",16],(*float number identifier*)
			ToCharacterCode@ExportString[num,"Real32",ByteOrdering -> 1], (*actual data bytes*)
			FromDigits["F7", 16] (*end sysex*)
		}
	]
];


readFirmataNumberPacket[packet_List] := Module[{},
	Switch[packet[[2]],
		4,First@ImportString[FromCharacterCode[packet[[3;;6]]],"Real32",ByteOrdering -> 1],
		5,First@ImportString[FromCharacterCode[packet[[3;;6]]],"Integer32",ByteOrdering -> 1],
		FromDigits["71",16],(*FromCharacterCode[packet[[4;;-1]]]*)packet
     ]
] 


numericalPin[pin_]:=Module[{},pin/.{"A0"->14,"a0"->14,"a1"->15,"A1"->15,"a2"->16,"A2"->16,"a3"->17,"A3"->17,"a4"->18,"A4"->18,"a5"->19,"A5"->19}]


End[] (* End Private Context *)

EndPackage[]
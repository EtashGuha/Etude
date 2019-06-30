(* ::Package:: *)

(* $Id$ *)

BeginPackage["TinkerForgeWeatherStation`", {"TinkerForgeWeatherStationTools`"}]

Begin["`Private`"];

$scaleFactor1 = 0.1;
$scaleFactor2 = 0.001;
$scaleFactor3 = 0.01;
$hostName = "localhost";
$portNumber = 4223;
$lightSensorID = "";(*"ibF"*);
$humSensorID = "";(*"hUi"*);
$pressTempSensorID = "";(*"g2h"*) ;
$lcdID = "";(*"gLp";*)
$strList = {"Illuminance", "Humidity", "Pressure", "Temperature"};

(*DeviceRead::devconfig = "DeviceRead was called prior to assigning one or more of the bricklet UIDs through DeviceConfigure.";*)
(*DeviceWrite::devconfig = "DeviceWrite was called prior to assigning the LCD bricklet UID through DeviceConfigure.";*)
(*DeviceExecute::devconfig = "DeviceExecute was called prior to assigning the LCD bricklet UID through DeviceConfigure.";*)
(*DeviceWrite::nuid = "A device communication was attempted prior to assigning one or more of the bricklet UIDs through DeviceConfigure."*)


DeviceFramework`Devices`TinkerForgeWeatherStation::nuid = "Unable to communicate with device. Please assign proper bricklet UIDs 
using DeviceConfigure."

(*TinkerForge::ndev = "The Head of argument `1` must be DeviceObject."
TinkerForge::badarg = "Argument `1` is not one of "<>ToString[$strList]<>" or any combination thereof."*)

(*prepFunction[]:= (PrependTo[$Path, NotebookDirectory[]];
If[$OperatingSystem === "Windows",
Install[ "tinkerforge-mathlink-proxy.exe" ],
Install["tinkerforge-mathlink-proxy"]
]
)*)

prepFunction[]:= TinkerForgeWeatherStationTools`InstallMathLinkEXE[];

makeHandleFunction[]:= Tinkerforge`IPConnectionCreate[]

configFunction[{ihandle_,dhandle_},arg__]:= (With[{rest = {arg}},( 
 $lightSensorID = extractID[FilterRules[rest,"AmbientLightBricklet"]];
 $humSensorID = extractID[FilterRules[rest,"HumidityBricklet"]];
 $pressTempSensorID = extractID[FilterRules[rest,"BarometerBricklet"]];
 $lcdID = extractID[FilterRules[rest,"LCDBricklet"]]; 
)
 ]
 )
openFunction[ipconn_]:= Tinkerforge`IPConnectionConnect[ipconn, $hostName, $portNumber];
	 
readFunction[args___]:= readFunction[{args}]

readFunction[{{ipconn_,rest__}}]:= readFun[{ipconn,rest},"fromRF"]

readFunction[{{ipconn_,rest__},"Temperature"}]:= (If[$pressTempSensorID === {} || $pressTempSensorID === "",
(Message[DeviceFramework`Devices`TinkerForgeWeatherStation::nuid];Return[$Failed];),
(With[{baro = Tinkerforge`BrickletBarometerCreate[$pressTempSensorID, ipconn]},
Quantity[Tinkerforge`BrickletBarometerGetChipTemperature[baro]*$scaleFactor3,"Celsius"]
])
]
)
readFunction[{{ipconn_,rest__},"Humidity"}]:= If[ $humSensorID === {} || $humSensorID === "",
(Message[DeviceFramework`Devices`TinkerForgeWeatherStation::nuid];Return[$Failed];),
(With[{hum = Tinkerforge`BrickletHumidityCreate[$humSensorID, ipconn]},
 Quantity[Tinkerforge`BrickletHumidityGetHumidity[hum]*$scaleFactor1,"Percent"]
])
]
readFunction[{{ipconn_,rest__},"Pressure"}]:= If[$pressTempSensorID === {} || $pressTempSensorID === "",
(Message[DeviceFramework`Devices`TinkerForgeWeatherStation::nuid];Return[$Failed];),
(With[{baro = Tinkerforge`BrickletBarometerCreate[$pressTempSensorID, ipconn]},
Quantity[ Tinkerforge`BrickletBarometerGetAirPressure[baro]*$scaleFactor2,"Millibar"]
])
]
readFunction[{{ipconn_,rest__},"Illuminance"}]:= If[$lightSensorID === {} || $lightSensorID === "",
(Message[DeviceFramework`Devices`TinkerForgeWeatherStation::nuid];Return[$Failed];),
(With[ {ill = Tinkerforge`BrickletAmbientLightCreate[$lightSensorID, ipconn]},
Quantity[Tinkerforge`BrickletAmbientLightGetIlluminance[ill]*$scaleFactor1,"Lux"]
])	 
]
readFun[{ipconn_,__},str_] := If[MemberQ[{$lightSensorID,$humSensorID,$pressTempSensorID},""|{}],
(Message[DeviceFramework`Devices`TinkerForgeWeatherStation::nuid];Return[$Failed];),
   ( With[ {baro = Tinkerforge`BrickletBarometerCreate[$pressTempSensorID, ipconn],
    hum = Tinkerforge`BrickletHumidityCreate[$humSensorID, ipconn],
    ill = Tinkerforge`BrickletAmbientLightCreate[$lightSensorID, ipconn]
    },
        If[ str === "fromRF",
            Return[ {
            "Temperature"-> Quantity[Tinkerforge`BrickletBarometerGetChipTemperature[baro]*$scaleFactor3,"Celsius"],
            "Humidity"-> Quantity[Tinkerforge`BrickletHumidityGetHumidity[hum]*$scaleFactor1,"Percent"],
            "Pressure"-> Quantity[Tinkerforge`BrickletBarometerGetAirPressure[baro]*$scaleFactor2,"Millibar"],
            "Illuminance"-> Quantity[Tinkerforge`BrickletAmbientLightGetIlluminance[ill]*$scaleFactor1,"Lux"]
            }],
            Return[ {
            Tinkerforge`BrickletAmbientLightGetIlluminance[ill]*$scaleFactor1,
            Tinkerforge`BrickletHumidityGetHumidity[hum]*$scaleFactor1,
            Tinkerforge`BrickletBarometerGetAirPressure[baro]*$scaleFactor2,
            Tinkerforge`BrickletBarometerGetChipTemperature[baro]*$scaleFactor3
            }]
        ]
    ])
]
closeFunction[{ipconn_,args__}] := Tinkerforge`IPConnectionDisconnect[ipconn];


releaseFunction[link___]:=((*RemoveScheduledTask[ScheduledTasks[]];*)LinkClose[link])


writeFunction[{ipconn_,handles__}, rules_]/;VectorQ[rules,ruleQ] :=
    Module[ {ill, hum, press, temp},
        {ill, hum, press, temp} = Lookup[rules,{"Illuminance","Humidity", "Pressure", "Temperature"}];
        If[ MemberQ[{$lightSensorID,$humSensorID,$pressTempSensorID},""|{}],
            (Message[DeviceFramework`Devices`TinkerForgeWeatherStation::nuid];
             Return[$Failed])
        ];
        lcd =  Tinkerforge`BrickletLCD20x4Create[$lcdID, ipconn];
        Tinkerforge`BrickletLCD20x4BacklightOn[lcd];
        If[ QuantityQ[temp],
            (temp = UnitConvert[temp, "Celsius"];
             Tinkerforge`BrickletLCD20x4WriteLine[lcd, 3, 0, Tinkerforge`StringToKS0066U["Temperature" <> ToString[PaddedForm[N[temp], {4, 2}]] <>  " \[Degree]C"]];
            )
        ];
        If[ QuantityQ[hum],
            (
            Tinkerforge`BrickletLCD20x4WriteLine[lcd, 1, 0, "Humidity   " <> ToString[PaddedForm[N[hum], {4, 2}]] <> " % "];
   )
        ];
        If[ QuantityQ[press],
            (
            press = UnitConvert[press, "Millibar"];
            Tinkerforge`BrickletLCD20x4WriteLine[lcd, 2, 0, "Air Press" <> ToString[PaddedForm[N[press], {6, 2}]] <> " mb"];    
            )
        ];
        If[ QuantityQ[ill],
            (ill = UnitConvert[ill, "Lux"];
             Tinkerforge`BrickletLCD20x4WriteLine[lcd, 0, 0, "Illuminanc" <> ToString[PaddedForm[N[ill], {5, 2}]] <> " lx"];
             )
        ];
    ]


writeFunction[{ipconn_,handles__}, All] :=
    Module[ {lcd,ill, hum, press, temp},
	    If[MemberQ[{$lightSensorID,$humSensorID,$pressTempSensorID},""|{}],(Message[DeviceFramework`Devices`TinkerForgeWeatherStation::nuid];Return[$Failed])];
        lcd =  Tinkerforge`BrickletLCD20x4Create[$lcdID, ipconn];
		{ill,hum, press, temp} = readFun[{ipconn,handles},"fromWF"];
        Tinkerforge`BrickletLCD20x4BacklightOn[lcd];
        Tinkerforge`BrickletLCD20x4WriteLine[lcd, 3, 0, Tinkerforge`StringToKS0066U["Temperature" <> ToString[PaddedForm[N[temp], {4, 2}]] <> 
           " \[Degree]C"]];
        Tinkerforge`BrickletLCD20x4WriteLine[lcd, 1, 0, "Humidity   " <> ToString[PaddedForm[N[hum], {4, 2}]] <> " % "];
        Tinkerforge`BrickletLCD20x4WriteLine[lcd, 2, 0, "Air Press" <> ToString[PaddedForm[N[press], {6, 2}]] <> " mb"];
        Tinkerforge`BrickletLCD20x4WriteLine[lcd, 0, 0, "Illuminanc" <> ToString[PaddedForm[N[ill], {5, 2}]] <> " lx"];
    ]

writeFunction[{ipconn_,handles__},None] :=
    With[ {lcd =  Tinkerforge`BrickletLCD20x4Create[$lcdID, ipconn]},
        Tinkerforge`BrickletLCD20x4ClearDisplay[lcd]
    ]

writeFunction[{ipconn_,handles__},str_String]:= writeFunction[{ipconn,handles},0,0,str]

writeFunction[{ipconn_,handles_},arg__] :=
    Module[ {n,p,str},
        {n, p,str} = {arg};
        writeFunctionFormatted[{ipconn,handles},{n,p,str}]
    ]


writeFunctionFormatted[{ipconn_,handles_},{nn_Integer,pp_Integer,string_String}] :=
    Module[ {lcd,n,p,
    str, rem,quot,lin = 20,str1,str2,retStr,writeStr1,writeStr2,writeStr3,writeStr4},
        n = nn+ 1;
        p = pp(*+ 1*);
        lcd =  Tinkerforge`BrickletLCD20x4Create[$lcdID, ipconn];
        Tinkerforge`BrickletLCD20x4BacklightOn[lcd];
        str = getWriteString[string,n,p]; (*this is the formatted string*)
        {str1,str2} = {StringTake[str,lin-p+1],StringDrop[str,lin-p+1]};
        {quot,rem} = QuotientRemainder[StringLength[str2],20];
        Which[
        quot===3,
        (retStr = StringTake[str2,{20,{21,40},{41,60}}];
         {writeStr1,writeStr2,writeStr3,writeStr4} = Insert[retStr[[1;;3]],str1,1];
         Tinkerforge`BrickletLCD20x4WriteLine[lcd,0,p,writeStr1];
         Tinkerforge`BrickletLCD20x4WriteLine[lcd,1,0,writeStr2];
         Tinkerforge`BrickletLCD20x4WriteLine[lcd,2,0,writeStr3];
         Tinkerforge`BrickletLCD20x4WriteLine[lcd,3,0,writeStr4];
        ),
        quot===2,
        (retStr = StringTake[str2,{20,{21,40}}];
         {writeStr1,writeStr2,writeStr3} = Insert[retStr[[1;;2]],str1,1];
         Tinkerforge`BrickletLCD20x4WriteLine[lcd,1,p,writeStr1];
         Tinkerforge`BrickletLCD20x4WriteLine[lcd,2,0,writeStr2];
         Tinkerforge`BrickletLCD20x4WriteLine[lcd,3,0,writeStr3];
        ),
        quot==1,
        (retStr = StringTake[str2,20];
         {writeStr1,writeStr2} = {str1,retStr};(*Insert[retStr[[1]],str1,1];*)
         Tinkerforge`BrickletLCD20x4WriteLine[lcd,2,p,writeStr1];
         Tinkerforge`BrickletLCD20x4WriteLine[lcd,3,0,writeStr2];
        ),
        True,
        (*retStr=str2*)
        (writeStr1 = str1;
         Tinkerforge`BrickletLCD20x4WriteLine[lcd,3,p,writeStr1];
        )
        ];
    ]
    
(*utility function: to be moved to TinkerForgeWeatherStationTools`*)
format[n_,p_] :=
    Module[ {lin = 4,P = 20,elems},
        elems = (P-p+1)(lin-n+1)+(lin-n)(p-1);
        elems
    ]

getWriteString[str_,n_,p_] :=
    Module[ {charLength = StringLength[str],formLength = format[n,p]},
        If[ charLength===formLength,
            str,
            If[ charLength>formLength,
                StringTake[str,formLength],
                StringJoin@PadRight[Characters@str,formLength," "]
            ]
        ]
    ]
	
extractID[{arg_Rule}]:= arg[[2]];
extractID[___]:= "";

Attributes[ruleQ] = {HoldAll};
ruleQ[x_] := MatchQ[Unevaluated@x, _Rule | _RuleDelayed]  

executeFunction[{ipconn_,handles_}, "ClearLCDDisplay" ,args___] := If[$lcdID === {} || $lcdID === "",
(Message[DeviceFramework`Devices`TinkerForgeWeatherStation::nuid];Return[$Failed];),
    With[ {lcd =  Tinkerforge`BrickletLCD20x4Create[$lcdID, ipconn]},
        Tinkerforge`BrickletLCD20x4ClearDisplay[lcd]
    ]
]
DeviceFramework`DeviceClassRegister["TinkerForgeWeatherStation",
    "OpenManagerFunction" :> prepFunction,
    "MakeManagerHandleFunction" -> (makeHandleFunction[]&),
	"ConfigureFunction" -> (configFunction),
    "CloseFunction" :> closeFunction,
    "ReleaseFunction" :> releaseFunction(*(RemoveScheduledTask[ScheduledTasks[]];LinkClose)*),
    "OpenFunction" :> openFunction,
    "ReadFunction" -> readFunction(*(readFunction[#1,"fromRF"]&)*),
    "WriteFunction" -> (writeFunction),
    "ExecuteFunction" -> (executeFunction)
]

End[];

EndPackage[]

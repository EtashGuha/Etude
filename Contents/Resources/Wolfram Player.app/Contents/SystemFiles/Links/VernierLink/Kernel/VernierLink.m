(*Begin Package*)

BeginPackage["VernierLink`"]
findDevice::usage = "";
deviceOpen::usage = "";
deviceProperties::usage = "";
deviceRead::usage = "";
deviceReadBuffer::usage = "";
deviceClose::usage = "";
deviceConfigure::usage = "";

Begin["`Private`"]

Needs["PacletManager`"];


(*Library Loading*)

(*The following allows us to have a more generalized way of loading the library, still not generalized for multiple OS's though*)

loadLibrary[]:=With[
	{
		libgoIO = FindLibrary[
			(*unfortunately the name of the go io dynamic library is different on the various platforms...*)
			Switch[$OperatingSystem,
				"Windows", "GoIO_DLL",
				"MacOSX", "libGoIOUniversal",
				"Unix", "libGoIO"
			]
		],
		(*for some reason linux when compiled doesn't have upper cases and FindLibrary can't find it that way*)
		libvernierLink = FindLibrary[If[$OperatingSystem =!= "Unix",
			"libVernierLink",
			"libvernierlink"
		]]
	},
	(
		(*load the go io base library - only on non-linux platforms because on linux, the rpath loader handles all that for us*)
		If[$OperatingSystem =!= "Unix",
			LibraryLoad[libgoIO]
		];
		
		iFindDevice = LibraryFunctionLoad[libvernierLink, "FindDevice", {"UTF8String"}, {Integer, 1}];
		iDeviceProperties = LibraryFunctionLoad[libvernierLink, "DeviceProperties", {"UTF8String", "UTF8String"}, {Real, 1}];
		iDeviceOpen = LibraryFunctionLoad[libvernierLink, "DeviceOpen", {"UTF8String"}, {Integer, 1}];
		iDeviceConfigure = LibraryFunctionLoad[libvernierLink, "DeviceConfigure", {"UTF8String", Real}, "Void"];
		iDeviceRead = LibraryFunctionLoad[libvernierLink, "DeviceRead", {"UTF8String"}, Real];
		iDeviceReadBuffer = LibraryFunctionLoad[libvernierLink, "DeviceReadBuffer", {"UTF8String", "UTF8String"}, {Real, 1}];
		iDeviceClose = LibraryFunctionLoad[libvernierLink, "DeviceClose", {"UTF8String"}, "Void"];
		iReinit = LibraryFunctionLoad[libvernierLink, "GoIO_Reinit", {}, "Void"];
	)
];

loadLibrary[]

$DeviceCalibrationModels=<||>;

(*this variable trackes the device names that are passed to DeviceOpen and their mappings to the actual argument needed*)
$DeviceNameToUSBhandleMappings = <||>;

$OpenDevices=<||>;

(*WL Wrappers*)

(*Find Device*)

findDevice[arg_?(MemberQ[{"productID", "vendorID"}, #] &)] := iFindDevice[arg];

findDevice["deviceName"] := iFindDevice["deviceName"];
deviceListSplit[list_] := Module[{codes, local = list}, 
	{codes, local} = TakeDrop[Rest[list], First[list]];
   	Sow[codes];
  	local
]

findDevice[] := Module[{deviceList, errorCode, ids}, 
    deviceList = iFindDevice["deviceName"];
    {errorCode, deviceList} = TakeDrop[deviceList, 1];
    ids = If[deviceList =!= {},
    	FromCharacterCode /@ (Reap[NestWhile[deviceListSplit, deviceList, Length[#] > 0 &]][[2, 1]]),
		{}
	];
	(*delete the open devices from the ids we got back, as iFindDevice will report those as well*)
	ids=Complement[ids,Values@$OpenDevices];
	(*reset all the device name mappings each time we run findDevice so that we don't have zombie devices showing up*)
	$DeviceNameToUSBhandleMappings=<||>;
	(*now we need to open up all the attached devices to get their device name mapping to the id's we got above*)
	(*this is because the DeviceFramework labels the devices with the argument to open them, which in this case the argument needed to see them is a *)
	(*ugly looking USB ID, so instead we provide the device name as the argument, and internally look up which device that is at DeviceOpen to get the right identifier*)
	($DeviceNameToUSBhandleMappings[#1]=#2)&@@@(Block[
		{
			dhandle=FromCharacterCode[Rest[iDeviceOpen[#]]],
			name
		},
		(*get the name from our dataset in this file*)
		name = $SensorsInformation[[Key[deviceProperties[dhandle,"Integer", "charID"]],"Simple Name"]]/.{_?MissingQ:>"Unknown Vernier Sensor"};
		(*close the device as we don't actually want it to be opened by the user, just discovered*)
		iDeviceClose[dhandle];
		{name,#}
	]&/@ids);
	(*all the proper device open arguments are the keys of the device name mappings*)
	Keys[$DeviceNameToUSBhandleMappings]
];
    
findDevice[___] := Message[findDevice::badarg];


(*Device Open*)

DeviceOpen::invalidDev="The device `1` isn't an available Vernier device. Use FindDevices[\"Vernier\"] to discover available Vernier devices";

DeviceOpen::noDev="There are no available Vernier devices to open"

deviceOpen[ihandle_, deviceName_] := Block[
	{
		dhandle,
		usbHandle = $DeviceNameToUSBhandleMappings[deviceName]
	},
	If[MemberQ[findDevice[],deviceName],
		(*THEN*)
		(*the device is good to open, it is a valid identifier - note we look up the actual arg from the association tracking the actual usb location ID*)
		(
			dhandle = FromCharacterCode[Rest[iDeviceOpen[usbHandle]]];
			(*save this device as open*)
			$OpenDevices[dhandle]=$DeviceNameToUSBhandleMappings[deviceName];
			dhandle
		),
		(*ELSE*)
		(*not a valid identifier - issue a message and return $Failed*)
		(
			Message[DeviceOpen::invalidDev,deviceName];
			$Failed
		)
	]
];

deviceOpen[ihandle_] := With[{devs = findDevice[]},
	If[devs =!= {},
		(*THEN*)
		(*use the first one we found*)
		deviceOpen[ihandle,First@devs],
		(*ELSE*)
		(*none found, so just issue message and fail*)
		(
			Message[DeviceOpen::noDev];
			$Failed
		)
	]
];

(*Device Properties*)

deviceProperties[dhandle_, "Integer", arg_String] := Last[Round[iDeviceProperties[dhandle, arg]]];

(*only select printable characters from the string we get back - also don't use the first character which is the status code*)
deviceProperties[dhandle_, "String", arg_String] := With[{res=Round/@iDeviceProperties[dhandle, arg]},
	If[Length[res]>0 && First[res]===0,
		(*THEN*)
		(*no error - just take the PrintableASCII characters*)
		StringJoin@Select[FromCharacterCode/@Rest[res],PrintableASCIIQ],
		(*ELSE*)
		(*error - return $Failed*)
		$Failed
	]
];

deviceProperties[dhandle_, "Float", arg_String] := Last[iDeviceProperties[dhandle, arg]];

probeAssoc = AssociationThread[
	Join[Range[0, 7], Range[10, 13]], 
	{
		"No Probe",
		"Time", 
		"Analog 5V", 
		"Analog 10V", 
		"Heat Pulser", 
		"Analog Output", 
		"MD", 
		"Photo Gate", 
		"Digital Count", 
		"Rotary", 
		"Digital Output", 
		"Labquest Audio"
	}
];
deviceProperties[dev_, "ProbeType"] := Lookup[probeAssoc,deviceProperties[DeviceFramework`DeviceHandle[dev], "Integer", "probeType"],Missing["UnknownProbeType"]];

deviceProperties[dev_, "SensorID"] := deviceProperties[DeviceFramework`DeviceHandle[dev], "Integer", "charID"];

deviceProperties[dev_, "SensorDescription"] := deviceProperties[DeviceFramework`DeviceHandle[dev], "String", "deviceDesc"];

deviceProperties[dev_, "SensorName"] := deviceProperties[DeviceFramework`DeviceHandle[dev], "String", "longName"];

(*the buffer length is an internal feature of the library and not configurable at runtime*)
deviceProperties[dev_, "BufferLength"] := 1200;

deviceProperties[dev_, "MinMeasurementInterval"] := Quantity[deviceProperties[DeviceFramework`DeviceHandle[dev], "Float", "minimumMeasurementPeriod"],"Seconds"];

deviceProperties[dev_, "MeasurementInterval"] := Quantity[deviceProperties[DeviceFramework`DeviceHandle[dev], "Float", "measurementPeriod"],"Seconds"];

deviceProperties[dev_, "MaxMeasurementInterval"] := Quantity[deviceProperties[DeviceFramework`DeviceHandle[dev], "Float", "maximumMeasurementPeriod"],"Seconds"];

deviceProperties[dev_, "EquationType"] := deviceProperties[DeviceFramework`DeviceHandle[dev], "Integer", "calibrationEquation"];

deviceProperties[arg_?(MemberQ[devicePropertiesArgs, #] &)] := Last[iDeviceProperties[First[findDevice[]], arg]];

deviceProperties[any___] := Message[deviceProperties::badarg];


(*Device Read*)

deviceRead[{ihandle_, dhandle_}] := (
	(*lookup the units from the $SensorsInformation dataset that is defined below with all the known sensors*)
	(*if the unit isn't found, then just use DimensionlessUnit*) 	
	Quantity[
		(*if we have a model stored for this device, apply it to the return result, else just use whatever we got*)
		If[KeyExistsQ[dhandle]@$DeviceCalibrationModels,$DeviceCalibrationModels[dhandle],Identity]@
			iDeviceRead[dhandle],
		getUnit[dhandle]
	]
)


deviceRead[deviceName_, "units"] := FromCharacterCode[Round[DeleteCases[iDeviceRead[deviceName][[3;;]], 40. | 41.]]]

deviceRead["units"] := With[{devices = findDevice[]}, AssociationThread[devices -> (deviceRead[#, "units"]& /@ devices)]];

deviceRead[] := With[{devices = findDevice[]}, AssociationThread[devices -> (deviceRead /@ devices)]];

deviceRead[___] := Message[deviceRead::badarg];


(*Device Read Buffer*)

deviceReadBuffer[{ihandle_, dhandle_}, "units"] := deviceRead[dhandle, "units"]

deviceReadBuffer["units"] := deviceRead["units"];

deviceReadBuffer[dhandle_, arg_?(MemberQ[{"units", "values", "volts", "rawMeasurements"}, #] &), Automatic] := iDeviceReadBuffer[dhandle, arg];

deviceReadBuffer[{ihandle_, dhandle_}, Automatic] := (
	(*lookup the units from the $SensorsInformation dataset that is defined below with all the known sensors*)
	(*if the unit isn't found, then just use DimensionlessUnit*) 
	QuantityArray[
		(*if there's a calibration model for this function use it, else just take whatever we got*)
		If[KeyExistsQ[dhandle]@$DeviceCalibrationModels,$DeviceCalibrationModels[dhandle]/@#&,Identity]@
			iDeviceReadBuffer[dhandle,"values"],
		getUnit[dhandle]
	]
)

deviceReadBuffer[any___] := Message[DeviceReadBuffer::badarg];


(*Device Close*)

(*we just have to delete this device from the list of open devices and then release the handle in the C code*)
deviceClose[{ihandle_, dhandle_},___] := (KeyDropFrom[$OpenDevices,dhandle]; iDeviceClose[dhandle]);

deviceClose[any___] := Message[DeviceClose::badarg];


(*Device Configure*)
DeviceConfigure::badMeasurementIntervalUnit = "The unit `1` is an invalid Quantity; only units of time are acceptable";

DeviceConfigure::badCalibrationUnit = "The unit `1` is not compaible with the units of this sensor (`2`)";

DeviceConfigure::invalidModel = "The model `1` is invalid and cannot be used";

DeviceConfigure::sizeMeasurementInterval = "The unit `1` is `2`, defaulting to `3` of `4`";

deviceConfigure[{ihandle_, dhandle_}, "MeasurementInterval" -> msPeriod_Quantity] := 
	If[CompatibleUnitQ[msPeriod,"Seconds"],
		(*THEN*)
		(*valid time unit - convert it to milliseconds and try to use it*)
		(
			With[
				{
					mag = UnitConvert[msPeriod,"Seconds"],
					min = Quantity[deviceProperties[dhandle, "Float", "minimumMeasurementPeriod"],"Seconds"],
					max = Quantity[deviceProperties[dhandle, "Float", "maximumMeasurementPeriod"],"Seconds"]
				},
				Which[
					mag >= min && mag <= max, (*it's within the range and fine to use as is*)
					(
						iDeviceConfigure[dhandle,QuantityMagnitude@mag]
					),
					(*ELSE - error conditions*)
					(*too large or small, issue message and use largest/smallest allowed*)
					mag < min, (*too small*)
					(
						Message[DeviceConfigure::sizeMeasurementInterval,msPeriod,"too small","min",min];
						iDeviceConfigure[dhandle,QuantityMagnitude@min]
					),
					mag > max, (*too large*)
					(
						Message[DeviceConfigure::sizeMeasurementInterval,msPeriod,"too large","max",max];
						iDeviceConfigure[dhandle,QuantityMagnitude@max]
					),
					True,
					(
						(*something is wrong..., but should be impossible, no way to have a number that's both not in the range and not not in the range*)
						$Failed
					)
				]
			]
		),
		(*not a valid unit*)
		(
			Message[DeviceConfigure::badMeasurementIntervalUnit,msPeriod];
			$Failed
		)
	];



deviceConfigure[{ihandle_, dhandle_}, "MeasurementInterval" -> msPeriod_Real] := deviceConfigure[{ihandle, dhandle},"MeasurementInverval" -> Quantity[msPeriod,"Seconds"]];

deviceConfigure[{ihandle_, dhandle_}, "MeasurementInterval" -> msPeriod_Integer] := deviceConfigure[{ihandle, dhandle},"MeasurementInverval" -> Quantity[msPeriod,"Seconds"]];

deviceConfigure[{ihandle_, dhandle_}, "MeasurementInterval" -> Automatic] := iDeviceConfigure[dhandle, First[deviceProperties[dhandle, "Float", "minimumMeasurementPeriod"]]];

deviceConfigure[{ihandle_, dhandle_}, "MeasurementInterval" -> Default] := deviceConfigure[{ihandle, dhandle}, "MeasurementInterval" -> Automatic];


(*these forms are for calibrating the sensor and ensuring that future values are scaled appropriately*)
deviceConfigure[{ihandle_, dhandle_}, "CalibrationModel" -> val_Quantity] := Block[
	{
		sensorUnit = getUnit[dhandle],
		func
	},
	If[CompatibleUnitQ[val,sensorUnit],
		(*THEN*)
		(*the unit is compatible and we can use the current value as this*)
		(
			(*the function to apply to all future values is *)
			(*new = old raw val - (val when configured - val to be configured as)*)
			func = With[{scaling = QuantityMagnitude[deviceRead[{ihandle,dhandle}] - UnitConvert[val,sensorUnit]]},Function[{data},data-scaling]];
			$DeviceCalibrationModels[dhandle] = func;
		),
		(*ELSE*)
		(*not compatible, raise message*)
		(
			Message[DeviceConfigure::badCalibrationUnit,val,sensorUnit];
			$Failed
		)
	]
];

deviceConfigure[{ihandle_, dhandle_}, "CalibrationModel" -> 0] := deviceConfigure[{ihandle, dhandle}, "CalibrationModel" -> Quantity[0,getUnit[dhandle]]]

(*for fitted model objects, check that the model returns a real number and then save it*)
deviceConfigure[{ihandle_, dhandle_}, "CalibrationModel" -> model_FittedModel] := 
	If[TrueQ@Element[model[0],Reals],
		(*THEN*)
		(*then the model is valid*)
		$DeviceCalibrationModels[dhandle] = model,
		(*ELSE*)
		(*model didn't return a number, so it's invalid, issue message and fail*)
		(
			Message[DeviceConfigure::invalidModel,model];
			$Failed
		)
	];

deviceConfigure[any___] := Message[DeviceConfigure::badarg];


preConfigure[dev_]:= ( 
	(*set the status labels to use the Simple Name from the association Dataset below*)
	DeviceFramework`DeviceStatusLabels[dev] = With[
		{
			(*default to using "Unknown Vernier Sensor as the name of the device if we can't identify it"*)
			name = $SensorsInformation[[Key[dev["SensorID"]],"Simple Name"]]/.{_?MissingQ:>"Unknown Vernier Sensor"}
		},
		(*first name is the Connected label, second one is the Disconnected label*)
		{"Connected ("<>name<>")","Not Connected ("<>name<>")"}
	];
	All 
)

$image = Rasterize[ImageResize[Import[PacletResource["DeviceDriver_Vernier", "Logo"]],70]]

iconf[___]:=$image

(*Device Framework*)

If[Not[Devices`DeviceAPI`DeviceDump`knownClassQ["Vernier"]],
	(*THEN*)
	(*there aren't any vernier drivers registered, so we should register it one*)
	DeviceFramework`DeviceClassRegister["Vernier",
		"FindFunction" -> (List /@ VernierLink`findDevice[] &),
		"OpenFunction" -> VernierLink`deviceOpen,
		"CloseFunction" -> VernierLink`deviceClose,
		"ReadFunction" -> VernierLink`deviceRead,
		"ReadBufferFunction" -> VernierLink`deviceReadBuffer,
		"Properties" -> {
			"SensorID" -> Null,
			"SensorName" -> Null,
			"BufferLength"->1200, (*a constant - never changes*)
			"SensorDescription" -> Null,
			"MeasurementInterval" -> Null,
			"MinMeasurementInterval" -> Null,
			"MaxMeasurementInterval" -> Null,
			"ProbeType" -> Null,
			"EquationType" -> Null
		},
		"GetPropertyFunction" -> VernierLink`deviceProperties,
		"ConfigureFunction" -> VernierLink`deviceConfigure,
		"PreconfigureFunction"-> VernierLink`preConfigure,
		"DeviceIconFunction" -> VernierLink`iconf,
		"DeregisterOnClose" -> True
	]
	(*ELSE*)
	(*there already is a driver registered, so don't need to register anything*)
]


(*Error Handling*)

LibraryFunction::devicenotfound = "Unable to find Go device";
LibraryFunction::badargument = "Bad argument passed";
LibraryFunction::msperiodlow = "Set measurement period too low";

getUnit[dhandle_]:= $SensorsInformation[[Key[deviceProperties[dhandle,"Integer","charID"]],"S.I. Units (page 0)"]]/. {_Missing :> "DimensionlessUnit"}


$SensorsInformation = <|
	0 -> <|"Sensor ID" -> 0, "Simple Name" -> "Unknown Sensor", "Simple Units (page 0)" -> ""|>,
	8 -> <|"Sensor ID" -> 8, "Simple Name" -> "Differential Voltage Sensor", "Simple Units (page 0)" -> "", "S.I. Units (page 0)" -> "Volts"|>,
	9 -> <|"Sensor ID" -> 9, "Simple Name" -> "Current Sensor", "Simple Units (page 0)" -> "", "S.I. Units (page 0)" -> "Amperes"|>,
	15 -> <|"Sensor ID" -> 15, "Simple Name" -> "EKG Sensor", "Simple Units (page 0)" -> "", "S.I. Units (page 0)" -> "Millivolts"|>,
	20 -> <|"Sensor ID" -> 20, "Simple Name" -> "Ph Sensor", "Simple Units (page 0)" -> "", "S.I. Units (page 0)" -> IndependentUnit["pH"]|>,
	21 -> <|"Sensor ID" -> 21, "Simple Name" -> "Conduct 200 Sensor", "Simple Units (page 0)" -> "(MICS)", "S.I. Units (page 0)" -> "Microsiemens"/"Centimeters"|>,
	22 -> <|"Sensor ID" -> 22, "Simple Name" -> "Conduct 2000 Sensor", "Simple Units (page 0)" -> "(MICS)", "S.I. Units (page 0)" -> "Microsiemens"/"Centimeters"|>,
	23 -> <|"Sensor ID" -> 23, "Simple Name" -> "Conduct 20000 Sensor", "Simple Units (page 0)" -> "(MICS)", "S.I. Units (page 0)" -> "Microsiemens"/"Centimeters"|>,
	24 -> <|"Sensor ID" -> 24, "Simple Name" -> "Gas Pressure Sensor", "Simple Units (page 0)" -> "(KPA)", "S.I. Units (page 0)" -> "Kilopascals"|>,
	25 -> <|"Sensor ID" -> 25, "Simple Name" -> "Dual R Force 10 Sensor", "Simple Units (page 0)" -> "(N)", "S.I. Units (page 0)" -> "Newtons"|>,
	26 -> <|"Sensor ID" -> 26, "Simple Name" -> "Dual R Force 50 Sensor", "Simple Units (page 0)" -> "(N)", "S.I. Units (page 0)" -> "Newtons"|>,
	27 -> <|"Sensor ID" -> 27, "Simple Name" -> "25g Accel Sensor", "Simple Units (page 0)" -> "(m/s^2)", "S.I. Units (page 0)" -> "Meters"/"Seconds"^2|>,
	28 -> <|"Sensor ID" -> 28, "Simple Name" -> "Low G Accel Sensor", "Simple Units (page 0)" -> "(m/s^2)", "S.I. Units (page 0)" -> "Meters"/"Seconds"^2|>,
	29 -> <|"Sensor ID" -> 29, "Simple Name" -> "X Axis Accel Sensor", "Simple Units (page 0)" -> "(m/s^2)", "S.I. Units (page 0)" -> "Meters"/"Seconds"^2|>,
	30 -> <|"Sensor ID" -> 30, "Simple Name" -> "Y Axis Accel Sensor", "Simple Units (page 0)" -> "(m/s^2)", "S.I. Units (page 0)" -> "Meters"/"Seconds"^2|>,
	31 -> <|"Sensor ID" -> 31, "Simple Name" -> "Z Axis Accel Sensor", "Simple Units (page 0)" -> "(m/s^2)", "S.I. Units (page 0)" -> "Meters"/"Seconds"^2|>,
	(*for the units of microphone see https://www.vernier.com/til/656/*)
	33 -> <|"Sensor ID" -> 33, "Simple Name" -> "Microphone Sensor", "Simple Units (page 0)" -> "", "S.I. Units (page 0)" -> "Volts"|>,
	34 -> <|"Sensor ID" -> 34, "Simple Name" -> "Light 600 Sensor", "Simple Units (page 0)" -> "(LX)", "S.I. Units (page 0)" -> "Lux"|>,
	35 -> <|"Sensor ID" -> 35, "Simple Name" -> "Light 6000 Sensor", "Simple Units (page 0)" -> "(LX)", "S.I. Units (page 0)" -> "Lux"|>,
	36 -> <|"Sensor ID" -> 36, "Simple Name" -> "Light 150000 Sensor", "Simple Units (page 0)" -> "(LX)", "S.I. Units (page 0)" -> "Lux"|>,
	37 -> <|"Sensor ID" -> 37, "Simple Name" -> "D. Oxygen Sensor", "Simple Units (page 0)" -> "(MG/L)", "S.I. Units (page 0)" -> "Milligrams"/"Liters"|>,
	38 -> <|"Sensor ID" -> 38, "Simple Name" -> "Ca Ise Sensor", "Simple Units (page 0)" -> "(MG/L)", "S.I. Units (page 0)" -> "Milligrams"/"Liters"|>,
	39 -> <|"Sensor ID" -> 39, "Simple Name" -> "Nh4 Ise Sensor", "Simple Units (page 0)" -> "(MG/L)", "S.I. Units (page 0)" -> "Milligrams"/"Liters"|>,
	40 -> <|"Sensor ID" -> 40, "Simple Name" -> "No3 Ise Sensor", "Simple Units (page 0)" -> "(MG/L)", "S.I. Units (page 0)" -> "Milligrams"/"Liters"|>,
	41 -> <|"Sensor ID" -> 41, "Simple Name" -> "Cl Ise Sensor", "Simple Units (page 0)" -> "(MG/L)", "S.I. Units (page 0)" -> "Milligrams"/"Liters"|>,
	42 -> <|"Sensor ID" -> 42, "Simple Name" -> "Flow Rate Sensor", "Simple Units (page 0)" -> "(M/S)", "S.I. Units (page 0)" -> "Meters"/"Seconds"|>,
	43 -> <|"Sensor ID" -> 43, "Simple Name" -> "Turbidity Sensor", "Simple Units (page 0)" -> "(NTU)", "S.I. Units (page 0)" -> IndependentUnit["numbers of transfer units"]|>,
	44 -> <|"Sensor ID" -> 44, "Simple Name" -> "Hi Magnet Fld Sensor", "Simple Units (page 0)" -> "(MT)", "S.I. Units (page 0)" -> "Milliteslas"|>,
	45 -> <|"Sensor ID" -> 45, "Simple Name" -> "Lo Magnet Fld Sensor", "Simple Units (page 0)" -> "(MT)", "S.I. Units (page 0)" -> "Milliteslas"|>,
	46 -> <|"Sensor ID" -> 46, "Simple Name" -> "Barometer Sensor", "Simple Units (page 0)" -> "(KPA)", "S.I. Units (page 0)" -> "Kilopascals"|>,
	47 -> <|"Sensor ID" -> 47, "Simple Name" -> "Rel Humidity Sensor", "Simple Units (page 0)" -> "(PCT)", "S.I. Units (page 0)" -> "Percent"|>,
	48 -> <|"Sensor ID" -> 48, "Simple Name" -> "Custom/generic 5v Sensor", "Simple Units (page 0)" -> "(V)", "S.I. Units (page 0)" -> "Volts"|>,
	49 -> <|"Sensor ID" -> 49, "Simple Name" -> "Custom/generic 10v Sensor", "Simple Units (page 0)" -> "(V)", "S.I. Units (page 0)" -> "Volts"|>,
	50 -> <|"Sensor ID" -> 50, "Simple Name" -> "Force Plate 850 Sensor", "Simple Units (page 0)" -> "(N)", "S.I. Units (page 0)" -> "Newtons"|>,
	51 -> <|"Sensor ID" -> 51, "Simple Name" -> "Force Plate 3500 Sensor", "Simple Units (page 0)" -> "(N)", "S.I. Units (page 0)" -> "Newtons"|>,
	52 -> <|"Sensor ID" -> 52, "Simple Name" -> "Uva Sensor", "Simple Units (page 0)" -> "mw/m^2", "S.I. Units (page 0)" -> "Milliwatts"/"Meters"^2|>,
	53 -> <|"Sensor ID" -> 53, "Simple Name" -> "Uvb Sensor", "Simple Units (page 0)" -> "mw/m^2", "S.I. Units (page 0)" -> "Milliwatts"/"Meters"^2|>,
	54 -> <|"Sensor ID" -> 54, "Simple Name" -> "Colorimeter Sensor", "Simple Units (page 0)" -> "", "S.I. Units (page 0)" -> "AbsorbanceUnits"|>,
	58 -> <|"Sensor ID" -> 58, "Simple Name" -> "Electrode Amp Sensor", "Simple Units (page 0)" -> "(mV)", "S.I. Units (page 0)" -> "Millivolts"|>,
	59 -> <|"Sensor ID" -> 59, "Simple Name" -> "Thermocouple Sensor", "Simple Units (page 0)" -> "(C)", "S.I. Units (page 0)" -> "DegreesCelsiusDifference"|>,
	61 -> <|"Sensor ID" -> 61, "Simple Name" -> "Salinity Sensor", "Simple Units (page 0)" -> "(ppt)", "S.I. Units (page 0)" -> "PartsPerThousand"|>,
	62 -> <|"Sensor ID" -> 62, "Simple Name" -> "Pressure Sensor", "Simple Units (page 0)" -> "(kPa)", "S.I. Units (page 0)" -> "Kilopascals"|>,
	63 -> <|"Sensor ID" -> 63, "Simple Name" -> "Charge 5 Sensor", "Simple Units (page 0)" -> "(nC)", "S.I. Units (page 0)" -> "Nanocoulombs"|>,
	64 -> <|"Sensor ID" -> 64, "Simple Name" -> "Charge 20 Sensor", "Simple Units (page 0)" -> "(nC)", "S.I. Units (page 0)" -> "Nanocoulombs"|>,
	65 -> <|"Sensor ID" -> 65, "Simple Name" -> "Charge 100 Sensor", "Simple Units (page 0)" -> "(nC)", "S.I. Units (page 0)" -> "Nanocoulombs"|>,
	66 -> <|"Sensor ID" -> 66, "Simple Name" -> "Blood Pressure Sensor", "Simple Units (page 0)" -> "(mm Hg)", "S.I. Units (page 0)" -> "MillimetersOfMercury"|>,
	67 -> <|"Sensor ID" -> 67, "Simple Name" -> "Force Sensor", "Simple Units (page 0)" -> "(N)", "S.I. Units (page 0)" -> "Newtons"|>,
	68 -> <|"Sensor ID" -> 68, "Simple Name" -> "Flow Rate Sensor", "Simple Units (page 0)" -> "(L/s)", "S.I. Units (page 0)" -> "Liters"/"Seconds"|>,
	70 -> <|"Sensor ID" -> 70, "Simple Name" -> "Soil Moisture Sensor", "Simple Units (page 0)" -> "(%)", "S.I. Units (page 0)" -> "Percent"|>,
	71 -> <|"Sensor ID" -> 71, "Simple Name" -> "D. Co2 Sensor", "Simple Units (page 0)" -> "(mg/L)", "S.I. Units (page 0)" -> "Milligrams"/"Liters"|>,
	72 -> <|"Sensor ID" -> 72, "Simple Name" -> "Current Sensor", "Simple Units (page 0)" -> "(A)", "S.I. Units (page 0)" -> "Amperes"|>,
	73 -> <|"Sensor ID" -> 73, "Simple Name" -> "Temperature Sensor", "Simple Units (page 0)" -> "(C)", "S.I. Units (page 0)" -> "DegreesCelsiusDifference"|>,
	74 -> <|"Sensor ID" -> 74, "Simple Name" -> "Sound Level Sensor", "Simple Units (page 0)" -> "(dB)", "S.I. Units (page 0)" -> IndependentUnit["decibels"]|>,
	75 -> <|"Sensor ID" -> 75, "Simple Name" -> "Co2 Low Sensor", "Simple Units (page 0)" -> "(ppm)", "S.I. Units (page 0)" -> "PartsPerMillion"|>,
	76 -> <|"Sensor ID" -> 76, "Simple Name" -> "Co2 High Sensor", "Simple Units (page 0)" -> "(ppm)", "S.I. Units (page 0)" -> "PartsPerMillion"|>,
	77 -> <|"Sensor ID" -> 77, "Simple Name" -> "Oxygen Gas Sensor", "Simple Units (page 0)" -> "(PCT)", "S.I. Units (page 0)" -> "Percent"|>,
	78 -> <|"Sensor ID" -> 78, "Simple Name" -> "Temperature Sensor", "Simple Units (page 0)" -> "(C)", "S.I. Units (page 0)" -> "DegreesCelsiusDifference"|>,
	79 -> <|"Sensor ID" -> 79, "Simple Name" -> "Current Sensor", "Simple Units (page 0)" -> "(A)", "S.I. Units (page 0)" -> "Amperes"|>,
	81 -> <|"Sensor ID" -> 81, "Simple Name" -> "Potential Sensor", "Simple Units (page 0)" -> "(mV)", "S.I. Units (page 0)" -> "Millivolts"|>,
	82 -> <|"Sensor ID" -> 82, "Simple Name" -> "Potential Sensor", "Simple Units (page 0)" -> "(mV)", "S.I. Units (page 0)" -> "Millivolts"|>,
	83 -> <|"Sensor ID" -> 83, "Simple Name" -> "Potential Sensor", "Simple Units (page 0)" -> "(mV)", "S.I. Units (page 0)" -> "Millivolts"|>,
	84 -> <|"Sensor ID" -> 84, "Simple Name" -> "Potential Sensor", "Simple Units (page 0)" -> "(mV)", "S.I. Units (page 0)" -> "Millivolts"|>,
	85 -> <|"Sensor ID" -> 85, "Simple Name" -> "Potential Sensor", "Simple Units (page 0)" -> "(mV)", "S.I. Units (page 0)" -> "Millivolts"|>,
	86 -> <|"Sensor ID" -> 86, "Simple Name" -> "Potential Sensor", "Simple Units (page 0)" -> "(mV)", "S.I. Units (page 0)" -> "Millivolts"|>,
	87 -> <|"Sensor ID" -> 87, "Simple Name" -> "Intensity Sensor", "Simple Units (page 0)" -> "(%)", "S.I. Units (page 0)" -> "Percent"|>,
	88 -> <|"Sensor ID" -> 88, "Simple Name" -> "Intensity Sensor", "Simple Units (page 0)" -> "(%)", "S.I. Units (page 0)" -> "Percent"|>,
	89 -> <|"Sensor ID" -> 89, "Simple Name" -> "Intensity Sensor", "Simple Units (page 0)" -> "(%)", "S.I. Units (page 0)" -> "Percent"|>,
	90 -> <|"Sensor ID" -> 90, "Simple Name" -> "Current Sensor", "Simple Units (page 0)" -> "(A)", "S.I. Units (page 0)" -> "Amperes"|>,
	91 -> <|"Sensor ID" -> 91, "Simple Name" -> "Speed Sensor", "Simple Units (page 0)" -> "(m/s)", "S.I. Units (page 0)" -> "Meters"/"Seconds"|>,
	92 -> <|"Sensor ID" -> 92, "Simple Name" -> "Temperature Sensor", "Simple Units (page 0)" -> "(C)", "S.I. Units (page 0)" -> "DegreesCelsiusDifference"|>,
	93 -> <|"Sensor ID" -> 93, "Simple Name" -> "Ethanol Sensor", "Simple Units (page 0)" -> "(PCT)", "S.I. Units (page 0)" -> "Percent"|>,
	94 -> <|"Sensor ID" -> 94, "Simple Name" -> "Current Sensor Sensor", "Simple Units (page 0)" -> "(A)", "S.I. Units (page 0)" -> "Amperes"|>,
	95 -> <|"Sensor ID" -> 95, "Simple Name" -> "Illumination Sensor", "Simple Units (page 0)" -> "(rel)", "S.I. Units (page 0)" -> IndependentUnit["Relative Light"]|>,
	96 -> <|"Sensor ID" -> 96, "Simple Name" -> "Irradiance Sensor", "Simple Units (page 0)" -> "(W/m^2)", "S.I. Units (page 0)" -> "Watts"/"Meters"^2|>,
	97 -> <|"Sensor ID" -> 97, "Simple Name" -> "Ethanol Sensor", "Simple Units (page 0)" -> "(PCT)", "S.I. Units (page 0)" -> "Percent"|>,
	98 -> <|"Sensor ID" -> 98, "Simple Name" -> "D. Oxygen Sensor", "Simple Units (page 0)" -> "(mg/L)", "S.I. Units (page 0)" -> "Milligrams"/"Liters"|>,
	99 -> <|"Sensor ID" -> 99, "Simple Name" -> "D. Oxygen Sensor", "Simple Units (page 0)" -> "(PCT)", "S.I. Units (page 0)" -> "Percent"|>,
	112 -> <|"Sensor ID" -> 112, "Simple Name" -> "Angle Sensor", "Simple Units (page 0)" -> "(DEG)", "S.I. Units (page 0)" -> "AngularDegrees"|>,
	113 -> <|"Sensor ID" -> 113, "Simple Name" -> "K Ise Sensor", "Simple Units (page 0)" -> "(MG/L)", "S.I. Units (page 0)" -> "Milligrams"/"Liters"|>,
	114 -> <|"Sensor ID" -> 114, "Simple Name" -> "Par Sensor", "Simple Units (page 0)" -> "(PPFD)", "S.I. Units (page 0)" -> "Micromoles"/("Meters"^2*"Seconds")|>,
	115 -> <|"Sensor ID" -> 115, "Simple Name" -> "Potential Sensor", "Simple Units (page 0)" -> "(V)", "S.I. Units (page 0)" -> "Volts"|>,
	116 -> <|"Sensor ID" -> 116, "Simple Name" -> "Current Sensor", "Simple Units (page 0)" -> "(mA)", "S.I. Units (page 0)" -> "Milliamperes"|>,
	117 -> <|"Sensor ID" -> 117, "Simple Name" -> "Force Sensor", "Simple Units (page 0)" -> "(N)", "S.I. Units (page 0)" -> "Newtons"|>,
	118 -> <|"Sensor ID" -> 118, "Simple Name" -> "Sound Level Sensor", "Simple Units (page 0)" -> "(dB)", "S.I. Units (page 0)" -> IndependentUnit["decibels"]|>,
	119 -> <|"Sensor ID" -> 119, "Simple Name" -> "1 N Force Sensor", "Simple Units (page 0)" -> "(N)", "S.I. Units (page 0)" -> "Newtons"|>,
	120 -> <|"Sensor ID" -> 120, "Simple Name" -> "5 N Force Sensor", "Simple Units (page 0)" -> "(N)", "S.I. Units (page 0)" -> "Newtons"|>,
	121 -> <|"Sensor ID" -> 121, "Simple Name" -> "Pressure Sensor", "Simple Units (page 0)" -> "(KPA)", "S.I. Units (page 0)" -> "Kilopascals"|>,
	122 -> <|"Sensor ID" -> 122, "Simple Name" -> "Conductivity Sensor", "Simple Units (page 0)" -> "(MICS)", "S.I. Units (page 0)" -> "Microsiemens"/"Centimeters"|>,
	123 -> <|"Sensor ID" -> 123, "Simple Name" -> "Thermocouple Sensor", "Simple Units (page 0)" -> "(C)", "S.I. Units (page 0)" -> "DegreesCelsiusDifference"|>,
	124 -> <|"Sensor ID" -> 124, "Simple Name" -> "Extra Long Temp Sensor", "Simple Units (page 0)" -> "(C)", "S.I. Units (page 0)" -> "DegreesCelsiusDifference"|>
|>





(*End Package*)

End[]
EndPackage[]
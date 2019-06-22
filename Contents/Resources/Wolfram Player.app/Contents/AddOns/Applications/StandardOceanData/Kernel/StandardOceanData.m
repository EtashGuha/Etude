Begin["Tools`StandardOceanData`Private`"]

$ProtectedSymbols = {
	System`StandardOceanData
};
	
Unprotect@@$ProtectedSymbols;
Clear@@$ProtectedSymbols;

loadLibraryFunctions[] :=
    Block[{path},

        path = FindLibrary["libsod"];
        If[path === $Failed,
            path = FileNameJoin[{DirectoryName[DirectoryName[$InputFileName]],
                "LibraryResources", $SystemID, "libsod"}]
        ];
        (
            TEOS10["AbsolutePressure"] = LibraryFunctionLoad[path, "teos10_AbsPressurefromp", {Real}, Real];
			TEOS10["AbsoluteSalinity"] = LibraryFunctionLoad[path, "teos10_safromsp", {Real, Real, Real, Real}, Real];
			TEOS10["AbsoluteSalinityAnomaly"] = LibraryFunctionLoad[path, "teos10_deltasafromsp", {Real, Real, Real, Real}, Real];
			TEOS10["AdiabaticLapseRate"] = LibraryFunctionLoad[path, "teos10_adiabaticlapseratefromct", {Real, Real, Real}, Real];
			TEOS10["Argon"] = LibraryFunctionLoad[path, "teos10_Arsol", {Real, Real, Real, Real, Real}, Real];
			TEOS10["ArgonConcentrationStandard"] = LibraryFunctionLoad[path, "teos10_ArsolSPpt", {Real, Real, Real}, Real];
			TEOS10["CabbelingCoefficient"] = LibraryFunctionLoad[path, "teos10_cabbeling", {Real, Real, Real}, Real];
			TEOS10["ChemicalPotentialSeaWater"] = LibraryFunctionLoad[path, "teos10_chempotentialrelativetexact", {Real, Real, Real}, Real];
			TEOS10["ChemicalPotentialSalt"] = LibraryFunctionLoad[path, "teos10_chempotentialsalttexact", {Real, Real, Real}, Real];
			TEOS10["ChemicalPotentialWater"] = LibraryFunctionLoad[path, "teos10_chempotentialwatertexact", {Real, Real, Real}, Real];
			TEOS10["Conductivity"] = LibraryFunctionLoad[path, "teos10_cfromsp", {Real, Real, Real}, Real];
			TEOS10["ConservativeTemperature"] = LibraryFunctionLoad[path, "teos10_ctfromt", {Real, Real, Real}, Real];
			TEOS10["Density"] = LibraryFunctionLoad[path, "teos10_rhotexact", {Real, Real, Real}, Real];
			TEOS10["Depth"] = LibraryFunctionLoad[path, "teos10_zfromp", {Real, Real}, Real];
			TEOS10["DynamicEnthalpy"] = LibraryFunctionLoad[path, "teos10_dynamicenthalpy", {Real, Real, Real}, Real];
			TEOS10["VaporizationHeat"] = LibraryFunctionLoad[path, "teos10_latentheatevapt", {Real, Real}, Real];
			TEOS10["FusionHeat"] = LibraryFunctionLoad[path, "teos10_latentheatmelting", {Real, Real}, Real];
			TEOS10["ConservativeMeltingPoint"] = LibraryFunctionLoad[path, "teos10_ctfreezing", {Real, Real, Real}, Real];
			TEOS10["MeltingPoint"] = LibraryFunctionLoad[path, "teos10_tfreezing", {Real, Real, Real}, Real];
			TEOS10["GravityAcceleration"] = LibraryFunctionLoad[path, "teos10_grav", {Real, Real}, Real];
			TEOS10["HelmholtzEnergy"] = LibraryFunctionLoad[path, "teos10_Helmholtzenergytexact", {Real, Real, Real}, Real];
			TEOS10["Helium"] = LibraryFunctionLoad[path, "teos10_Hesol", {Real, Real, Real, Real, Real}, Real];
			TEOS10["HeliumConcentrationStandard"] = LibraryFunctionLoad[path, "teos10_HesolSPpt", {Real, Real, Real}, Real];
			TEOS10["InternalEnergy"] = LibraryFunctionLoad[path, "teos10_internalenergy", {Real, Real, Real}, Real];
			TEOS10["IonicStrength"] = LibraryFunctionLoad[path, "teos10_ionicstrengthfromSA", {Real}, Real];
			TEOS10["IsentropicCompressibility"] = LibraryFunctionLoad[path, "teos10_kappatexact", {Real, Real, Real}, Real];
			TEOS10["IsobaricHeatCapacity"] = LibraryFunctionLoad[path, "teos10_cptexact", {Real, Real, Real}, Real];
			TEOS10["IsochoricHeatCapacity"] = LibraryFunctionLoad[path, "teos10_isochoricheatcaptexact", {Real, Real, Real}, Real];
			TEOS10["Krypton"] = LibraryFunctionLoad[path, "teos10_Krsol", {Real, Real, Real, Real, Real}, Real];
			TEOS10["KryptonConcentrationStandard"] = LibraryFunctionLoad[path, "teos10_KrsolSPpt", {Real, Real, Real}, Real];
			TEOS10["Molality"] = LibraryFunctionLoad[path, "teos10_molalityfromSA", {Real}, Real];
			TEOS10["Neon"] = LibraryFunctionLoad[path, "teos10_Nesol", {Real, Real, Real, Real, Real}, Real];
			TEOS10["NeonConcentrationStandard"] = LibraryFunctionLoad[path, "teos10_NesolSPpt", {Real, Real, Real}, Real];
			TEOS10["Nitrogen"] = LibraryFunctionLoad[path, "teos10_N2sol", {Real, Real, Real, Real, Real}, Real];
			TEOS10["NitrogenConcentrationStandard"] = LibraryFunctionLoad[path, "teos10_N2solSPpt", {Real, Real, Real}, Real];
			TEOS10["NitrousOxide"] = LibraryFunctionLoad[path, "teos10_N2Osol", {Real, Real, Real, Real, Real}, Real];
			TEOS10["NitrousOxideConcentrationStandard"] = LibraryFunctionLoad[path, "teos10_N2OsolSPpt", {Real, Real, Real}, Real];
			TEOS10["Oxygen"] = LibraryFunctionLoad[path, "teos10_O2sol", {Real, Real, Real, Real, Real}, Real];
			TEOS10["OxygenConcentrationStandard"] = LibraryFunctionLoad[path, "teos10_O2solSPpt", {Real, Real, Real}, Real];
			TEOS10["OsmoticCoefficient"] = LibraryFunctionLoad[path, "teos10_osmoticcoefficienttexact", {Real, Real, Real}, Real];
			TEOS10["OsmoticPressure"] = LibraryFunctionLoad[path, "teos10_osmoticpressuretexact", {Real, Real, Real}, Real];
			TEOS10["PotentialDensity"] = LibraryFunctionLoad[path, "teos10_potrhotexact", {Real, Real, Real, Real}, Real];
			TEOS10["PotentialDensityAnomaly"] = LibraryFunctionLoad[path, "teos10_sigma0", {Real, Real}, Real];
			TEOS10["PotentialTemperature"] = LibraryFunctionLoad[path, "teos10_pt0fromt", {Real, Real, Real}, Real];
			TEOS10["PracticalSalinity"] = LibraryFunctionLoad[path, "teos10_spfromsa", {Real, Real, Real, Real}, Real];
			TEOS10["PreformedSalinity"] = LibraryFunctionLoad[path, "teos10_sstarfromsa", {Real, Real, Real, Real}, Real];
			TEOS10["ReferenceSalinity"] = LibraryFunctionLoad[path, "teos10_srfromsp", {Real}, Real];
			TEOS10["SalineContractionCoefficient"] = LibraryFunctionLoad[path, "teos10_betaconsttexact", {Real, Real, Real}, Real];
			TEOS10["SoundSpeed"] = LibraryFunctionLoad[path, "teos10_soundspeedtexact", {Real, Real, Real}, Real];
			TEOS10["Enthalpy"] = LibraryFunctionLoad[path, "teos10_enthalpytexact", {Real, Real, Real}, Real];
			TEOS10["Entropy"] = LibraryFunctionLoad[path, "teos10_entropyfromt", {Real, Real, Real}, Real];
			TEOS10["GibbsEnergy"] = LibraryFunctionLoad[path, "teos10_gibbs", {Integer,Integer,Integer,Real, Real, Real}, Real];
			TEOS10["SpecificVolume"] = LibraryFunctionLoad[path, "teos10_specvoltexact", {Real, Real, Real}, Real];
			TEOS10["SpecificVolumeAnomaly"] = LibraryFunctionLoad[path, "teos10_specvolanom", {Real, Real, Real}, Real];
			TEOS10["Temperature"] = LibraryFunctionLoad[path, "teos10_tfromct", {Real, Real, Real}, Real];
			TEOS10["ThermalExpansionCoefficient"] = LibraryFunctionLoad[path, "teos10_alphawrttexact", {Real, Real, Real}, Real];
			TEOS10["ThermobaricCoefficient"] = LibraryFunctionLoad[path, "teos10_thermobaric", {Real, Real, Real}, Real];
            True/;SameQ[LibraryFunction,Sequence@@(Head[TEOS10[#]] &/@
            	{"AbsolutePressure","AbsoluteSalinity","AbsoluteSalinityAnomaly","AdiabaticLapseRate","Argon",
        "ArgonConcentrationStandard", "CabbelingCoefficient", "ChemicalPotentialSeaWater", "ChemicalPotentialSalt",
        "ChemicalPotentialWater","Conductivity","ConservativeTemperature","Density","Depth","DynamicEnthalpy",
        "VaporizationHeat","FusionHeat","ConservativeMeltingPoint","MeltingPoint",
        "GravityAcceleration","HelmholtzEnergy","Helium","HeliumConcentrationStandard",
        "InternalEnergy","IonicStrength","IsentropicCompressibility","IsobaricHeatCapacity","IsochoricHeatCapacity",
        "Krypton","KryptonConcentrationStandard","Molality","Neon",
        "NeonConcentrationStandard","Nitrogen","NitrogenConcentrationStandard",
        "NitrousOxide","NitrousOxideConcentrationStandard","Oxygen",
        "OxygenConcentrationStandard","OsmoticCoefficient","OsmoticPressure","PotentialDensity",
        "PotentialDensityAnomaly","PotentialTemperature","PracticalSalinity","PreformedSalinity","ReferenceSalinity",
        "SalineContractionCoefficient","SoundSpeed","Enthalpy","Entropy","GibbsEnergy",
        "SpecificVolume","SpecificVolumeAnomaly","Temperature","ThermalExpansionCoefficient","ThermobaricCoefficient"})]
        ) /; (path =!= $Failed)
    ]
librariesLoaded = loadLibraryFunctions[];

$properties=Sort@{"AbsoluteSalinity", "AbsoluteSalinityAnomaly", "AdiabaticLapseRate", "CabbelingCoefficient", "Conductivity", 
	"ConservativeMeltingPoint", "ConservativeTemperature", "Depth", "DynamicEnthalpy", "VaporizationHeat", "FusionHeat",
	"GravityAcceleration", "Density", "MeltingPoint", (*"Temperature",*) "InternalEnergy", 
	"IsentropicCompressibility", "IsobaricHeatCapacity", "PotentialDensity", "PotentialDensityAnomaly", 
	"PotentialTemperature", "PracticalSalinity", "PreformedSalinity", "ReferenceSalinity", "SalineContractionCoefficient", "SoundSpeed", 
	"Enthalpy", "Entropy", "GibbsEnergy", "SpecificVolumeAnomaly", "SpecificVolume", "ThermalExpansionCoefficient", 
	"ThermobaricCoefficient",
	
	"AbsolutePressure","Concentration","ChemicalPotentialSeaWater","ChemicalPotentialSalt","ChemicalPotentialWater",
	"HelmholtzEnergy","IonicStrength","IsochoricHeatCapacity","Molality","OsmoticCoefficient","OsmoticPressure"};

qvnorm[key_]:=key/.{QuantityVariable[_,x_String]:>x,QuantityVariable[x_String]:>x}

$gasrules={Entity[_,x_]:>x,"MolecularOxygen"->"Oxygen","MolecularNitrogen"->"Nitrogen"};

formatInputProperty[quantity_Quantity,"AbsoluteSalinity"]:=Module[{res},If[CompatibleUnitQ[quantity,Quantity["Grams"/"Kilograms"]],
	res=QuantityMagnitude[quantity,"Grams"/"Kilograms"];
	If[TrueQ[Element[res,Reals]],res,Message[System`StandardOceanData::pqr,quantity,"AbsoluteSalinity"];$Failed],
	Message[System`StandardOceanData::pq,quantity,"AbsoluteSalinity"];$Failed]
];
formatInputProperty[input_,"AbsoluteSalinity"]:=(Message[System`StandardOceanData::npq,"AbsoluteSalinity"];$Failed);
formatInputProperty[value_?NumericQ,"PracticalSalinity"]:=value;
formatInputProperty[input_,"PracticalSalinity"]:=(Message[System`StandardOceanData::inv,"PracticalSalinity"];$Failed);
formatInputProperty[quantity_Quantity,p:"Pressure"|"ReferencePressure"]:=Module[{res},If[
	CompatibleUnitQ[quantity,Quantity["Decibars"]],
	res=QuantityMagnitude[quantity,"Decibars"];
	If[TrueQ[Element[res,Reals]],res,Message[System`StandardOceanData::pqr,quantity,p];10.1325],
	Message[System`StandardOceanData::pq,quantity,p];10.1325]
];
formatInputProperty[input_,p:"Pressure"|"ReferencePressure"]:=(Message[System`StandardOceanData::npq,p];10.1325);
formatInputProperty[quantity_Quantity,"Temperature"]:=Module[{res},If[CompatibleUnitQ[quantity,Quantity["DegreesCelsius"]],
	res=QuantityMagnitude[quantity,"DegreesCelsius"];
	If[TrueQ[Element[res,Reals]],res,Message[System`StandardOceanData::pqr,quantity,"Temperature"];20],
	Message[System`StandardOceanData::pq,quantity,"Temperature"];20]
];
formatInputProperty[input_,"Temperature"]:=(Message[System`StandardOceanData::npq,"Temperature"];20);
formatInputProperty[position_GeoPosition,"Position"]:=QuantityMagnitude[position[[1,;;2]]];
formatInputProperty[{lat_?(QuantityQ[#]||NumericQ[#]&),long_?(QuantityQ[#]||NumericQ[#]&)},"Position"]:=QuantityMagnitude[{lat,long}];
formatInputProperty[position_Entity,"Position"]:=With[{coords=EntityValue[position, "Position"]},
	If[MatchQ[coords,_GeoPosition],
		QuantityMagnitude[coords[[1,;;2]]],
		(Message[System`StandardOceanData::geo,position];{0,180})
	]
];
formatInputProperty[input_,"Position"]:=(Message[System`StandardOceanData::geo,input];{0,180});
formatInputProperty[quantity_Quantity,"SaturationFraction"]:=Module[{res},If[CompatibleUnitQ[quantity,Quantity["Percent"]],
	res=QuantityMagnitude[quantity,"Percent"]/100;
	If[TrueQ[Element[res,Reals]],res,Message[System`StandardOceanData::pqr,quantity,"SaturationFraction"];1],
	Message[System`StandardOceanData::pq,quantity,"SaturationFraction"];1]
];
formatInputProperty[input_?NumericQ,"SaturationFraction"]:=input;
formatInputProperty[input_,"SaturationFraction"]:=(Message[System`StandardOceanData::inv,"SaturationFraction"];1);

propertyUnit[value_?(#>10^12&),prop_,True]:=(Message[System`StandardOceanData::range];
	Missing["NotAvailable"])
propertyUnit[value_,"AbsolutePressure",_]:=Quantity[value,"Pascals"];
propertyUnit[value_,"AbsoluteSalinity"|"AbsoluteSalinityAnomaly"|"PreformedSalinity"|"ReferenceSalinity",_]:=Quantity[value,"Grams"/"Kilograms"];
propertyUnit[value_,"AdiabaticLapseRate",_]:=Quantity[value,"Kelvins"/"Pascals"];
propertyUnit[value_,"Argon"|"Helium"|"Krypton"|"Neon"|"Nitrogen"|
			"NitrousOxide"|"Oxygen",_]:=Quantity[value,"Micromoles"/"Kilograms"];
propertyUnit[value_,"CabbelingCoefficient",_]:=Quantity[value,1/"Kelvins"^2];
propertyUnit[value_,"ChemicalPotentialSeaWater"|"ChemicalPotentialSalt"|"ChemicalPotentialWater",_]:=Quantity[value,"Joules"/"Grams"];
propertyUnit[value_,"Concentration",_]:=value
propertyUnit[value_,"Conductivity",_]:=Quantity[value,"Millisiemens"/"Centimeters"];
propertyUnit[value_,"ConservativeTemperature"|"ConservativeMeltingPoint"|"Temperature"|"MeltingPoint"|
	"PotentialTemperature",_]:=Quantity[value,"DegreesCelsius"];
propertyUnit[value_,"Depth",_]:=Quantity[-value,"Meters"];
propertyUnit[value_,"DynamicEnthalpy"|"FusionHeat"|"VaporizationHeat"|"InternalEnergy"|"Enthalpy",_]:=Quantity[value,"Joules"/"Kilograms"];
propertyUnit[value_,"Density"|"PotentialDensity"|"PotentialDensityAnomaly",_]:=Quantity[value,"Kilograms"/"Meters"^3];
propertyUnit[value_,"GravityAcceleration",_]:=Quantity[value,"Meters"/"Seconds"^2];
propertyUnit[value_,"HelmholtzEnergy",_]:=Quantity[value,"Joules"/"Kilograms"];
propertyUnit[value_,"IsochoricHeatCapacity"|"IsobaricHeatCapacity"|"Entropy",_]:=Quantity[value,"Joules"/"Kilograms"/"Kelvins"];
propertyUnit[value_,"IonicStrength"|"Molality",_]:=Quantity[value,"Moles"/"Kilograms"];
propertyUnit[value_,"IsentropicCompressibility",_]:=Quantity[value,1/"Pascals"];
propertyUnit[value_,"KnudsenSalinity",_]:=Quantity[value,"PartsPerThousand"];
propertyUnit[value_,"OsmoticCoefficient"|"PracticalSalinity",_]:=value;
propertyUnit[value_,"OsmoticPressure",_]:=UnitConvert[Quantity[value,"Decibars"],"Pascals"];
propertyUnit[value_,"SalineContractionCoefficient",_]:=Quantity[value,"Kilograms"/"Grams"];
propertyUnit[value_,"SoundSpeed",_]:=Quantity[value,"Meters"/"Seconds"];
propertyUnit[value_,"GibbsEnergy",_]:=Quantity[value,"Joules"/"Kilograms"];
propertyUnit[value_,"SpecificVolume"|"SpecificVolumeAnomaly",_]:=Quantity[value,"Meters"^3/"Kilograms"];
propertyUnit[value_,"ThermalExpansionCoefficient",_]:=Quantity[value,1/"Kelvins"];
propertyUnit[value_,"ThermobaricCoefficient",_]:=Quantity[value,1/("Kelvins""Pascals")];

iStandardOceanData[assoc_,property_]:=Module[{temperature,pressure,salinity,lat,lon,sp,satfrac,pref},
	If[Not[librariesLoaded],Return[$Failed]];
	If[Not[MemberQ[Append[$properties,All],property]],
		Message[System`StandardOceanData::notprop, property, System`StandardOceanData];Return[$Failed]];
	(*extract input information*)
	temperature=If[KeyExistsQ[assoc,"Temperature"],formatInputProperty[assoc["Temperature"],"Temperature"],20];
	(*TODO: adjust average ocean temperature, determine practical limits*)
	pressure=If[KeyExistsQ[assoc,"Pressure"],formatInputProperty[assoc["Pressure"],"Pressure"],10.1325];
	pref=If[KeyExistsQ[assoc,"ReferencePressure"],
		formatInputProperty[assoc["ReferencePressure"],"ReferencePressure"],100];
	(*TODO: using 10 meter depth as reference, include in Notes, determine practical limits*)
	salinity=If[KeyExistsQ[assoc,"AbsoluteSalinity"],formatInputProperty[assoc["AbsoluteSalinity"],"AbsoluteSalinity"],$Failed];
	sp=If[KeyExistsQ[assoc,"PracticalSalinity"],formatInputProperty[assoc["PracticalSalinity"],"PracticalSalinity"],$Failed];
	{lat,lon}=If[KeyExistsQ[assoc,"Position"],formatInputProperty[assoc["Position"],"Position"],{0,180}];
	If[TrueQ[lon>180||lon<-180],Message[System`StandardOceanData::bound,"Longitude",-180,180,180];lon=180];
	Which[TrueQ[lat>90],Message[System`StandardOceanData::bound,"Latitude",-86,90,90];lat=90,
		TrueQ[lat<-86],Message[System`StandardOceanData::bound,"Latitude",-86,90,-86];lat=-86];
	Which[salinity===$Failed&&sp===$Failed,
		salinity=35.16504; (*Standard Ocean Reference Salinity*)
		sp=TEOS10["PracticalSalinity"][salinity,pressure,lon,lat],
		salinity===$Failed,
		If[TrueQ[sp>42],Message[System`StandardOceanData::bound,"PracticalSalinity",0,42,42];sp=42];
		If[TrueQ[sp<0],Message[System`StandardOceanData::bound,"PracticalSalinity",0,42,0];sp=0];
		salinity=TEOS10["AbsoluteSalinity"][sp,pressure,lon,lat],
		sp===$Failed,
		If[TrueQ[salinity>42.19833516846399`],Message[System`StandardOceanData::bound,"AbsoluteSalinity",0,42,42];salinity=42];
		If[TrueQ[salinity<0],Message[System`StandardOceanData::bound,"AbsoluteSalinity",0,42,0];salinity=0];
		sp=TEOS10["PracticalSalinity"][salinity,pressure,lon,lat],
		True,
		If[TrueQ[sp>42],Message[System`StandardOceanData::bound,"PracticalSalinity",0,42,42];sp=42];
		If[TrueQ[sp<0],Message[System`StandardOceanData::bound,"PracticalSalinity",0,42,0];sp=0];
		salinity=TEOS10["AbsoluteSalinity"][sp,pressure,lon,lat]
	];
	satfrac=If[KeyExistsQ[assoc,"SaturationFraction"],formatInputProperty[assoc["SaturationFraction"],"SaturationFraction"],1];
	If[satfrac>1,Message[System`StandardOceanData::bound,"SaturationFraction",0,1,1];satfrac=1];
	If[satfrac<0,Message[System`StandardOceanData::bound,"SaturationFraction",0,1,0];satfrac=0];
	iStandardOceanCompute[{salinity,sp,temperature,pressure,lat,lon,satfrac,pref},property]
]
	
iStandardOceanCompute[{salinity_,sp_,temperature_,pressure_,lat_,lon_,satfrac_,pref_},All]:=Module[
	{ct=TEOS10["ConservativeTemperature"][salinity,temperature,pressure],result,temp},
	result={"AbsoluteSalinity"->propertyUnit[salinity,"AbsoluteSalinity",Null],
		"PracticalSalinity"->propertyUnit[sp,"PracticalSalinity",Null],"ConservativeTemperature"->propertyUnit[ct,"ConservativeTemperature",Null]};
	temp=#->propertyUnit[TEOS10[#][salinity,temperature,pressure],#,Null]&/@{"ChemicalPotentialSeaWater","ChemicalPotentialSalt",
		"ChemicalPotentialWater","Density","HelmholtzEnergy","IsentropicCompressibility","IsochoricHeatCapacity","IsobaricHeatCapacity",
		"OsmoticCoefficient","OsmoticPressure","PotentialTemperature", "SalineContractionCoefficient","SoundSpeed","Enthalpy",
		"Entropy","SpecificVolume","ThermalExpansionCoefficient"};
	result=Join[result,temp];
	result=Join[result,#->propertyUnit[TEOS10[#][salinity],#,Null]&/@{"IonicStrength","Molality"}];
	temp=#->propertyUnit[TEOS10[#][pressure,lat],#,Null]&/@{"Depth","GravityAcceleration"};
	result=Join[result,temp];
	temp=#->propertyUnit[TEOS10[#][salinity,pressure,satfrac],#,Null]&/@{"ConservativeMeltingPoint","MeltingPoint"};
	result=Join[result,temp];
	temp=#->propertyUnit[TEOS10[#][salinity,ct,pressure],#,Null]&/@{"AdiabaticLapseRate","CabbelingCoefficient", "DynamicEnthalpy", 
			"InternalEnergy", "SpecificVolumeAnomaly", "ThermobaricCoefficient"};
	result=Join[result,temp];
	temp="Concentration"->Association[#->propertyUnit[TEOS10[
			#//.$gasrules]
		[salinity,ct,pressure,lon,lat],#//.$gasrules,Null]&/@
		{Entity["Element", "Argon"], Entity["Element", "Helium"], Entity["Element", "Krypton"], 
			Entity["Chemical", "MolecularNitrogen"], Entity["Chemical", "MolecularOxygen"], Entity["Element", "Neon"], 
 			Entity["Chemical", "NitrousOxide"]}];
	result=Append[result,temp];
	result=Join[result,{"AbsolutePressure"->propertyUnit[TEOS10["AbsolutePressure"][pressure],"AbsolutePressure",Null],
		"PotentialDensity"->propertyUnit[TEOS10["PotentialDensity"][salinity,temperature,pressure,pref],"PotentialDensity",Null],
		"GibbsEnergy"->propertyUnit[TEOS10["GibbsEnergy"][0,0,0,salinity,temperature,pressure],"GibbsEnergy",Null],
		"VaporizationHeat"->propertyUnit[TEOS10["VaporizationHeat"][salinity,temperature],"VaporizationHeat",Null],
		"FusionHeat"->propertyUnit[TEOS10["FusionHeat"][salinity,pressure],"FusionHeat",Null],
		"PreformedSalinity"->propertyUnit[TEOS10["PreformedSalinity"][salinity,pressure,lon,lat],"PreformedSalinity",Null],
		"AbsoluteSalinityAnomaly"->propertyUnit[TEOS10["AbsoluteSalinityAnomaly"][sp,pressure,lon,lat],"AbsoluteSalinityAnomaly",Null],
		"Conductivity"->propertyUnit[TEOS10["Conductivity"][sp,temperature,pressure],"Conductivity",Null],
		"ReferenceSalinity"->propertyUnit[TEOS10["ReferenceSalinity"][sp],"ReferenceSalinity",Null],
		"PotentialDensityAnomaly"->propertyUnit[TEOS10["PotentialDensityAnomaly"][salinity,ct],"PotentialDensityAnomaly",Null]
		}];
	If[FreeQ[result,LibraryFunction | $Failed],Association[Sort[result]],$Failed]
];
iStandardOceanCompute[{salinity_,sp_,temperature_,pressure_,lat_,lon_,satfrac_,pref_},property_]:=Module[{ct,result},
	result=Which[
		MatchQ[property,"AbsoluteSalinity"],
			salinity,
		MatchQ[property,"PracticalSalinity"],
			sp,
		MatchQ[property,"AbsolutePressure"],
			TEOS10[property][pressure],
		MemberQ[{"IonicStrength","Molality"},property],
			TEOS10[property][salinity],
		MemberQ[{"ChemicalPotentialSeaWater","ChemicalPotentialSalt","ChemicalPotentialWater","ConservativeTemperature","Density",
			"HelmholtzEnergy","IsentropicCompressibility","IsochoricHeatCapacity","IsobaricHeatCapacity",
			"OsmoticCoefficient","OsmoticPressure","PotentialTemperature", "SalineContractionCoefficient","SoundSpeed","Enthalpy",
			"Entropy","SpecificVolume","ThermalExpansionCoefficient"},property],
			TEOS10[property][salinity,temperature,pressure],
		MatchQ[property,"PotentialDensity"],
			TEOS10[property][salinity,temperature,pressure,pref],
		MatchQ[property,"GibbsEnergy"],
			TEOS10[property][0,0,0,salinity,temperature,pressure],
		MatchQ[property,"VaporizationHeat"],
			TEOS10[property][salinity,temperature],
		MatchQ[property,"FusionHeat"],
			TEOS10[property][salinity,pressure],
		MemberQ[{"PreformedSalinity", "PracticalSalinity"},property],
			TEOS10[property][salinity,pressure,lon,lat],
		MemberQ[{"Depth","GravityAcceleration"},property],
			TEOS10[property][pressure,lat],
		MemberQ[{"ConservativeMeltingPoint","MeltingPoint"},property],
			TEOS10[property][salinity,pressure,satfrac],
		MatchQ["AbsoluteSalinityAnomaly",property],
			TEOS10[property][sp,pressure,lon,lat],
		MatchQ["Conductivity",property],
			TEOS10[property][sp,temperature,pressure],
		MatchQ["ReferenceSalinity",property],
			TEOS10[property][sp],
		ct=TEOS10["ConservativeTemperature"][salinity,temperature,pressure];
		MemberQ[{"AdiabaticLapseRate","CabbelingCoefficient", "DynamicEnthalpy", (*"Temperature",*) "InternalEnergy",
			"SpecificVolumeAnomaly", "ThermobaricCoefficient"},property],
			TEOS10[property][salinity,ct,pressure],
		MatchQ["Concentration",property],
			Association[#->propertyUnit[TEOS10[#//.$gasrules][salinity,ct,pressure,lon,lat],#//.$gasrules,Null]&/@
		{Entity["Element", "Argon"], Entity["Element", "Helium"], Entity["Element", "Krypton"], 
			Entity["Chemical", "MolecularNitrogen"], Entity["Chemical", "MolecularOxygen"], Entity["Element", "Neon"], 
 			Entity["Chemical", "NitrousOxide"]}],
		MemberQ[{"Argon","Helium","Krypton","Neon","Nitrogen","NitrousOxide","Oxygen"},property//.$gasrules],
			TEOS10[property/.Entity[_,x_]:>x][salinity,ct,pressure,lon,lat],
		MatchQ["PotentialDensityAnomaly",property],
			TEOS10[property][salinity,ct]
	];
	If[FreeQ[result,LibraryFunction | $Failed],propertyUnit[result,property,True],$Failed]
]

System`StandardOceanData["Properties"]:=$properties
System`StandardOceanData[assoc_Association,"Concentration",element_]:=With[{res=iStandardOceanData[KeyMap[qvnorm,assoc],element]},
	res/;res=!=$Failed]
System`StandardOceanData[assoc_Association,property_]:=With[{res=iStandardOceanData[KeyMap[qvnorm,assoc],property]},
	res/;res=!=$Failed]
System`StandardOceanData[assoc_Association]:=With[{res=iStandardOceanData[KeyMap[qvnorm,assoc],All]},
	res/;res=!=$Failed]
System`StandardOceanData[assoc_Rule,property__]:=System`StandardOceanData[Association[assoc],property]
System`StandardOceanData[assoc_Rule]:=System`StandardOceanData[Association[assoc]]
System`StandardOceanData[input_]:=With[{res=(Message[System`StandardOceanData::notent,input,System`StandardOceanData];$Failed)},
	res/;res=!=$Failed]
System`StandardOceanData[args___]:=With[{res = If[SameQ[args, {}], $Failed, 
	(System`Private`Arguments[System`StandardOceanData[args], {1, 3}]; $Failed)]},
    res /; res =!= $Failed
];

With[{symbols = $ProtectedSymbols},(*SetAttributes is HoldFirst*)
	SetAttributes[symbols, {ReadProtected}]
];

Protect@@$ProtectedSymbols;

End[];
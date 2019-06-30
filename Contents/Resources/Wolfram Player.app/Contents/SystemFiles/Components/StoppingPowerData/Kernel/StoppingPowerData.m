(* Mathematica package *)
Begin["Tools`StoppingPowerData`Private`"]

$ProtectedSymbols = {
	System`StoppingPowerData
};
	
Unprotect@@$ProtectedSymbols;
Clear@@$ProtectedSymbols;
$tag = "StoppingPowerDataCatchThrowTag";

AbortProtect[stoppingpowerdata = {#[[1]], #[[2]]} & /@ If[
	FileExistsQ[#],
	Import[#],
	Message[StoppingPowerData::init];Throw[$Failed,"MissingResource"]
]&[FileNameJoin[{DirectoryName[$InputFileName],"stoppingpowerpaclet.m"}]]];

System`StoppingPowerData[]:=With[{res=iStoppingPowerData[]},res/;res=!=$Failed]
System`StoppingPowerData[All|"Elements"]:=With[{res=iStoppingPowerData[]},res/;res=!=$Failed]
System`StoppingPowerData["Properties"]:=With[{res=iStoppingPowerData["Properties"]},res/;res=!=$Failed]
System`StoppingPowerData[ent_,prop:(_String|_EntityProperty)]:=With[{res=iStoppingPowerData[ent,prop]},res/;res=!=$Failed]
System`StoppingPowerData[ent:(_String|_Entity),pq_Association,prop:(_String|_EntityProperty)]:=With[
	{res=iStoppingPowerData[ent,prop,pq]},
	res/;res=!=$Failed
]
System`StoppingPowerData[ent:(_String|_Entity),pq_List?(VectorQ[#,Function[{x},Head[x]===Rule]]&),prop:(_String|_EntityProperty)]:=With[
	{res=iStoppingPowerData[ent,prop,pq]},
	res/;res=!=$Failed
]
System`StoppingPowerData[arg:Except[_String|_Entity],___]:=(Message[System`StoppingPowerData::notent, arg, System`StoppingPowerData];Null/;False)
System`StoppingPowerData[_,_,_,args__]:=(Message[System`StoppingPowerData::argrx,System`StoppingPowerData,Length[{args}]+3,3];Null/;False)

AbortProtect[$prules={QuantityVariable[_,x_String]:>x,QuantityVariable[x_String]:>x,Entity[_,x_String]:>x};
$translations = {"Aluminium"->"Aluminum"};
$LMrules={"BetheFormulaLinearStoppingPower"->"BetheFormulaStoppingPower", "BetheFormulaMassStoppingPower"->"BetheFormulaStoppingPower", 
	"CollisionLinearStoppingPower"->"CollisionStoppingPower","CollisionMassStoppingPower"->"CollisionStoppingPower", 
	"CSDARangeLinearStoppingPower"->"CSDARange", "CSDARangeMassStoppingPower"->"CSDARange", 
	"ElectronicLinearStoppingPower"->"ElectronicStoppingPower", "ElectronicMassStoppingPower"->"ElectronicStoppingPower", 
	"NuclearLinearStoppingPower"->"NuclearStoppingPower", "NuclearMassStoppingPower"->"NuclearStoppingPower", 
	"RadiativeLinearStoppingPower"->"RadiativeStoppingPower", "RadiativeMassStoppingPower"->"RadiativeStoppingPower", 
	"LinearStoppingPower"->"StoppingPower", "MassStoppingPower"->"StoppingPower"};
$Photonrules={"AbsorptanceFraction"->"MassAttenuationCoefficient", "HalfValueLayer"->"MassAttenuationCoefficient", "ShieldingThickness"->"CSDARange", 
	"TenthValueLayer"->"MassAttenuationCoefficient", "TransmittanceFraction"->"MassAttenuationCoefficient"}];

invalidArgumentsQ[ent_String]:=If[MemberQ[System`StoppingPowerData[],ent/.$translations],False,
		Message[System`StoppingPowerData::notent,ent,System`StoppingPowerData];True]
invalidArgumentsQ[Entity[type_,ent_]]:=If[MemberQ[System`StoppingPowerData[],ent/.$translations],False,
		Message[System`StoppingPowerData::notent,Entity[type,ent], System`StoppingPowerData];True]
invalidArgumentsQ[ent:(_String|_Entity),prop:(_String|_EntityProperty)]:=Module[{ep=True,property=prop/.EntityProperty[_,x_String,___]:>x},
	ep=invalidArgumentsQ[ent];
	If[And[StringQ[property],Not[MemberQ[System`StoppingPowerData["Properties"],property]]],
		Message[System`StoppingPowerData::notprop,prop,System`StoppingPowerData];ep=True];
	ep
]
invalidArgumentsQ[ent:(_String|_Entity),prop:(_String|_EntityProperty),pq_List]:=Module[
		{part="Particle"/.(pq/.$prules),e="Energy"/.(pq/.$prules),t="Temperature"/.(pq/.$prules),p="Pressure"/.(pq/.$prules),ep=True,
		speed="Speed"/.(pq/.$prules)},
	ep=invalidArgumentsQ[ent,prop];
	e=quantcheck[e,"Joules","Energy"];
	t=quantcheck[t,"Kelvins","Temperature"];
	p=quantcheck[p,"Pascals","Pressure"];
	speed=quantcheck[speed,"Meters"/"Seconds","Speed"];
	part=If[MemberQ[{"Electron","Photon","Proton","AlphaParticle","Particle"},part],
		False,
		Message[System`StoppingPowerData::particle,part];True
	];
	part||e||ep
]
invalidArgumentsQ[ent:(_String|_Entity),prop:(_String|_EntityProperty),pq_Rule]:=invalidArgumentsQ[ent,prop,{pq}]
invalidArgumentsQ[___]:=True

quantcheck[e_,unit_,pq_]:=Which[MatchQ[e,_Quantity],
		Which[
			Not[CompatibleUnitQ[Quantity[300,unit],QuantityUnit[e]]],
			Message[System`StoppingPowerData::unit,pq];True,
			Not[TrueQ[Element[QuantityMagnitude[e],Reals]]],
			Message[System`StoppingPowerData::quant,QuantityMagnitude[e]];True,
			True,
			False],
		ArrayQ[e,_,Function[{x},Head[x]===Quantity]],
		Which[
			Not[And@@(CompatibleUnitQ[Quantity[300,unit],#]&/@Flatten@QuantityUnit[e])],
			Message[System`StoppingPowerData::unit,pq];True,
			Not[And@@Flatten@Map[NumericQ,QuantityMagnitude[e],{-1}]],
			Message[System`StoppingPowerData::quants,QuantityMagnitude[e]];True,
			True,
			False],
		True,
		If[(e=!=pq),
			Message[System`StoppingPowerData::npq,pq];True,
			False
		]]
		
calculateRKE[speed_,p_]:=Module[{gamma=1/Sqrt[1-QuantityMagnitude[UnitConvert[speed,"SpeedOfLight"]]^2],r},
	r=UnitConvert[(gamma-1)*Quantity["SpeedOfLight"]^2*Switch[p,"AlphaParticle",Quantity["AlphaParticleMass"],
		"Proton",Quantity["ProtonMass"],"Electron",Quantity["ElectronMass"]],"Megaelectronvolts"];
	If[QuantityQ[r],r,Automatic]
]

AbortProtect[properties = {"BetheFormulaStoppingPower", "CollisionStoppingPower", "CSDARange", "DensityEffectParameter", "ElectronicStoppingPower", 
   "StoppingPower", "MassAttenuationCoefficient", "MassEnergyAbsorptionCoefficient", "NuclearStoppingPower", "RadiationYield", 
   "RadiativeStoppingPower"};
particles = {"AlphaParticle", "Electron", "Photon", "Proton"};

$singleprops={"Density","Name"};
$PPcombinations = Join[$singleprops, 
   	Flatten[Table[{properties[[i]], particles[[j]]}, {i, Length[properties]}, {j, Length[particles]}], 1]]];
singleprops[ent_,prop_]:=Module[{entity=(ent/.$prules)/.$translations,entdata,pos},
	entdata=myFirst@Cases[stoppingpowerdata,{entity,_}];
	pos=myFirst@Position[$singleprops,prop];
	If[entdata==={}||prop==={},
		Missing["NotAvailable"],
		stoppingpowerunit[myFirst@entdata[[2,pos]],prop,1]
	]
]

iStoppingPowerData[]:=Sort@Select[stoppingpowerdata[[All,1]],StringQ]
iStoppingPowerData["Properties"]:=Sort@{"AbsorptanceFraction", "BetheFormulaLinearStoppingPower", "Name",
	"BetheFormulaMassStoppingPower", "CollisionLinearStoppingPower", "CollisionMassStoppingPower", "CSDARangeLinearStoppingPower", 
	"CSDARangeMassStoppingPower", "Density", "DensityEffectParameter", "ElectronicLinearStoppingPower", "ElectronicMassStoppingPower", 
	"HalfValueLayer", "MassAttenuationCoefficient", "MassEnergyAbsorptionCoefficient", "NuclearLinearStoppingPower", 
	"NuclearMassStoppingPower", "RadiationYield", "RadiativeLinearStoppingPower", "RadiativeMassStoppingPower", 
	"ShieldingThickness", "LinearStoppingPower", "MassStoppingPower", "TenthValueLayer", "TransmittanceFraction"};
iStoppingPowerData[_]:=$Failed;
iStoppingPowerData[ent:(_String|_Entity),prop_]:=If[invalidArgumentsQ[ent,prop],
	$Failed,
	If[prop=!="Name",$Failed,
		Module[{entity=(ent/.$prules)/.$translations,entdata},
			entdata=myFirst@Cases[stoppingpowerdata,{entity,_}];
			entdata[[2,2]]
		]
	]
];
iStoppingPowerData[ent_,prop_,pq_Association]:=Module[{particle,energy,speed,pqassoc=KeyMap[(#/.$prules)/.$translations&,pq]},
	particle=If[KeyExistsQ[pqassoc,"Particle"],pqassoc["Particle"],Automatic];
	energy=If[KeyExistsQ[pqassoc,"Energy"],pqassoc["Energy"],Automatic];
	speed=If[KeyExistsQ[pqassoc,"Speed"],pqassoc["Speed"],Automatic];
	If[speed===Automatic,
		iStoppingPowerData[ent,prop,{"Particle"->particle,"Energy"->energy}],
		iStoppingPowerData[ent,prop,{"Particle"->particle,"Speed"->speed}]
	]
]
iStoppingPowerData[ent:(_String|_Entity),prop_?(MemberQ[$singleprops,#]&),pq_List]:=If[invalidArgumentsQ[ent,prop],
	$Failed,
	singleprops[ent,prop]
]
iStoppingPowerData[ent:(_String|_Entity),prop:(_String|_EntityProperty),pq_List]:=If[
	invalidArgumentsQ[ent,prop,pq],
	$Failed,
	Module[
	{particle=("Particle"/.(pq/.$prules))/."Particle"->Automatic,energy=("Energy"/.(pq/.$prules))/."Energy"->Automatic,
		speed=("Speed"/.(pq/.$prules)),entity=(ent/.$prules)/.$translations, property=prop/.EntityProperty[_,x_String,___]:>x,position,
		entdata,data,density,s,e,coeff},
	If[particle===Automatic,particle=defaultparticle[(property)/.$LMrules/.$Photonrules]];
	If[energy===Automatic&&Not[StringQ[speed]]&&MemberQ[{"AlphaParticle","Proton","Electron"},particle],energy=calculateRKE[speed,particle]];
	If[particle===$Failed,Return[$Failed]];
	entdata=myFirst@Cases[stoppingpowerdata,{entity,_}];
	position=First@Flatten[Position[$PPcombinations,{((property)/.$LMrules)/.$Photonrules,particle}]];
	If[Not[NumericQ[position]],Return[$Failed]];
	data=entdata[[2,position]];
	If[StringMatchQ[property, ___~~"Linear"~~__]||MemberQ[$Photonrules[[All,1]],property],
		density=entdata[[2,1]],
		density=1
	];
	If[Not[NumericQ[density]],Return[Missing["NotAvailable"]]];
	If[QuantityQ[energy],
		If[Length[data]>2,
			{s,e}=Quantity[10^6*#,"Electronvolts"]&/@data[[{1, -1}, 1]];
			If[Not[LessEqual[s,energy,e]],Message[System`StoppingPowerData::orng,s,e];Return[$Failed]],
			Return[Missing["NotAvailable"]]
		];
		coeff=DeleteDuplicates[Log[datadisjoinShift[data]],First[#1] === First[#2] &];
		coeff=Exp[Interpolation[coeff,InterpolationOrder -> 1][Log[QuantityMagnitude[UnitConvert[energy,"Megaelectronvolts"]]]]],
		Return[Missing["NotAvailable"]]
	];
	
	Which[
		MemberQ[{"TenthValueLayer", "HalfValueLayer"},property],
		If[MatchQ[data,_Missing]||MatchQ[data,$Failed],
			data,
			-Log[E, If["TenthValueLayer"===property,.1,0.5]]/Quantity[coeff, "Centimeters"^2/"Grams"]/Quantity[density, "Kilograms"/"Meters"^3]
		],
		property==="ShieldingThickness",
		Replace[stoppingpowerunit[coeff,"CSDARangeLinearStoppingPower",density], x_ /; !TrueQ[QuantityMagnitude[x] > 0] :> Missing["NotAvailable"]],
		MemberQ[{"AbsorptanceFraction", "TransmittanceFraction"},property],
		If[MatchQ[data,_Missing]||MatchQ[data,$Failed],
			data,
			Replace[QuantityMagnitude@UnitConvert[stoppingpowerunit[coeff,"MassAttenuationCoefficient",density]*Quantity[density, "Kilograms"/"Meters"^3],1/"Meters"], 
				{x_ /; TrueQ[x > 0] :> (If["AbsorptanceFraction"===property,1-Exp[-x*0.001],Exp[-x*0.001]]&[0.001]), _ -> Missing["NotAvailable"]}]
		],
		True,
		If[NumericQ[coeff],
			stoppingpowerunit[coeff,property,density],
			Missing["NotAvailable"]
		]
	]
]]
iStoppingPowerData[___]:=$Failed

datadisjoinShift[data_List] := Module[{part = Partition[data, 2, 1]},
  If[First[#1] === First[#2], #1 + {-$MachineEpsilon, 0}, #1] & @@@ 
   part]

myFirst[x_?(Length[#]>0&)]:=First[x]
myFirst[x_?(Length[#]===0&)]:=x

AbortProtect[defaultparticle["AbsorptanceFraction"|"HalfValueLayer"|"MassAttenuationCoefficient"|"MassEnergyAbsorptionCoefficient"|"ShieldingThickness"|
	"TenthValueLayer"|"TransmittanceFraction"]="Photon";
defaultparticle["CollisionStoppingPower"|"CSDARange"|"DensityEffectParameter"|"RadiationYield"|"RadiativeStoppingPower"|
	"StoppingPower"]="Electron";
defaultparticle["BetheFormulaStoppingPower"|"ElectronicStoppingPower"|"NuclearStoppingPower"]="Proton";
defaultparticle[p_String]:="Electron"]
defaultparticle[___]:=$Failed

stoppingpowerunit[v_,p:"AbsorptanceFraction"|"DensityEffectParameter"|"RadiationYield"|"TransmittanceFraction"|"Name",d_]:=v
stoppingpowerunit[v_,p:"BetheFormulaMassStoppingPower"|"CollisionMassStoppingPower"|"ElectronicMassStoppingPower"|"NuclearMassStoppingPower"|"RadiativeMassStoppingPower"|
	"MassStoppingPower",d_]:=Quantity[v,"Megaelectronvolts"/("Grams"/"Centimeters"^2)]
stoppingpowerunit[v_,p:"BetheFormulaLinearStoppingPower"|"CollisionLinearStoppingPower"|"ElectronicLinearStoppingPower"|"NuclearLinearStoppingPower"|"RadiativeLinearStoppingPower"|
	"LinearStoppingPower",d_]:=Quantity[d*v/1000,"Megaelectronvolts"/"Centimeters"]
stoppingpowerunit[v_,p:"CSDARangeMassStoppingPower",d_]:=Quantity[v,"Grams"/"Centimeters"^2]
stoppingpowerunit[v_,p:"CSDARangeLinearStoppingPower",d_]:=Quantity[10*v/d,"Meters"]
stoppingpowerunit[v_,p:"MassAttenuationCoefficient"|"MassEnergyAbsorptionCoefficient",d_]:=Quantity[v,"Centimeters"^2/"Grams"]
stoppingpowerunit[v_,p:"HalfValueLayer"|"ShieldingThickness"|"TenthValueLayer",d_]:=Quantity[v,"Meters"]
stoppingpowerunit[v_,p:"Density",d_]:=Quantity[v,"Kilograms"/"Meters"^3]


With[{symbols = $ProtectedSymbols},(*SetAttributes is HoldFirst*)
	SetAttributes[symbols, {ReadProtected}]
];

Protect@@$ProtectedSymbols;

End[];
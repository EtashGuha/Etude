(* ::Package:: *)

(* :Title: PhysicalConstants *)

(* :Author: Stephen Wolfram *)

(* :Summary:
This package provides the values of various commonly used physical constants.
*)

(* :Package Version: 1.6 *)

(* :Copyright: Copyright 1988-2008, Wolfram Research, Inc. *)

(* :Context: PhysicalConstants` *)

(* :History: 
	Version 1.1 by Stephen Wolfram (Wolfram Research), 1988. 
	Revised by ECM (Wolfram Research), 1990, 1996, 1997.
	Added support for CosmicBackgroundTemperature, SolarSchwarzschildRadius,
	  GalacticUnit, and SolarLuminosity;  made minor changes in
	  values for ClassicalElectronRadius, EarthMass, EarthRadius,
	  ElectronComptonWavelength, GravitationalConstant, MolarGasConstant,
	  ThomsonCrossSection;  improved some usage messages. 
	  Barbara Ercolano (Wolfram Research), 1997.
    Version 1.4, Adjusted values for CODATA 1998, John M. Novak, April 2000. 
    Updated for new paclet scheme for Mathematica 6, Brian Van Vertloo, 
      December 2006.
    Version 1.5, Adjusted values for CODATA 2006, John M. Novak, July 2007.
    Version 1.6, adjust some exact values to be represented as such, May 2008.
    (Mark as Obsolete, August 2012 - functionality can be found in
     Quantity and data paclets such as IsotopeData, AstronomicalData, and
     the like.)
*)

(* :Keywords: *)

(* :Source:
    P.J. Mohr, B.N. Taylor, and D.B. Newell (2007), "The 2006 CODATA
        Recommended Values of the Fundamental Physical Constants"
        (Web Version 5.0). This database was developed by J. Baker,
        M. Douma, and S. Kotochigova.
	Available: http://physics.nist.gov/constants [as of 2007, July 22].
	National Institute of Standards and Technology, Gaithersburg, MD 20899.
       (Cited in code below as CODATA 2006.)
    CRC Handbook of Chemistry and Physics, 80th Edition, (David R. Lide,
      editor-in-chief) 1999-2000. (Cited in code below as HCAP 80.)
 *)

(* :Warning: None. *)

(* :Mathematica Version: 4.0-6.0 *)

(* :Limitation: None. *)

(* :Discussion:
Note that all values are expressed in SI units, which has been integrated 
into the Units package.

As of CODATA 1998, some conventions used for the electron and muon
g-factors, and for the electron, muon and neutron magnetic moments,
are different than before; they are all expressed as a negative
number in CODATA 1998, and a factor of two that was previously divided
out of the electron g-factor is present.

For the QuantizedHallConductance, HCAP 80 gives a value for e^2/h,
while CODATA 1998 gives a value for 2*e^2/h. I took the CODATA value
and divided out the factor of 2, to match the HCAP and the previous
use in this package.

These notes apply to the CODATA 2006 update, as well.
*)

Message[General::obspkg, "PhysicalConstants`"]

BeginPackage["PhysicalConstants`", "Units`"]

If[FileType[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"]]===File,
Select[FindList[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"],"PhysicalConstants`"],
StringMatchQ[#,StartOfString~~"PhysicalConstants`*"]&]//ToExpression;
];

(* ======================== GENERAL CONSTANTS ======================== *)

(* Universal Constants *)

If[Not@ValueQ[SpeedOfLight::usage],SpeedOfLight::usage =
"SpeedOfLight is the speed of light in a vacuum, a universal constant."];
If[Not@ValueQ[VacuumPermeability::usage],VacuumPermeability::usage =
"VacuumPermeability is the permeability of vacuum, a universal constant."];
If[Not@ValueQ[VacuumPermittivity::usage],VacuumPermittivity::usage =
"VacuumPermittivity is the permittivity of vacuum, a universal constant."];
If[Not@ValueQ[GravitationalConstant::usage],GravitationalConstant::usage =
"GravitationalConstant is the coefficient of proportionality in Newton's law of \
gravitation."];
If[Not@ValueQ[AccelerationDueToGravity::usage],AccelerationDueToGravity::usage =
"AccelerationDueToGravity is the acceleration of a body freely falling in a \
vacuum on Earth at sea level."];
If[Not@ValueQ[PlanckConstant::usage],PlanckConstant::usage =
"PlanckConstant is a universal constant of nature which relates the energy \
of a quantum of radiation to the frequency of the oscillator which emitted it."];
If[Not@ValueQ[PlanckConstantReduced::usage],PlanckConstantReduced::usage =
"PlanckConstantReduced is PlanckConstant/(2 Pi), a universal constant."];
If[Not@ValueQ[PlanckMass::usage],PlanckMass::usage = "PlanckMass is a universal constant."];

(* Electromagnetic Constants *)

If[Not@ValueQ[ElectronCharge::usage],ElectronCharge::usage =
"ElectronCharge is elementary charge, an electromagnetic constant."];
If[Not@ValueQ[MagneticFluxQuantum::usage],MagneticFluxQuantum::usage =
"MagneticFluxQuantum is magnetic flux quantum, an electromagnetic constant."];
If[Not@ValueQ[QuantizedHallConductance::usage],QuantizedHallConductance::usage =
"QuantizedHallConductance is quantized Hall conductance, an \
electromagnetic constant."];
(* BohrMagneton is Bohr magnetron, an electromagnetic constant.  But it
is also a unit of magnetic moment, so it is introduced only in Units.m, to
avoid shadowing. *)


(* =================== ATOMIC AND NUCLEAR CONSTANTS ==================== *)

If[Not@ValueQ[FineStructureConstant::usage],FineStructureConstant::usage =
"FineStructureConstant is the fine structure constant, an atomic constant."];
If[Not@ValueQ[RydbergConstant::usage],RydbergConstant::usage =
"RydbergConstant is an atomic constant appearing in the Rydberg formula \
expressing the wave-numbers of the lines in a spectral series."];
If[Not@ValueQ[BohrRadius::usage],BohrRadius::usage = "BohrRadius is the Bohr radius, an atomic constant."];

(* Electron *)

If[Not@ValueQ[ElectronMass::usage],ElectronMass::usage = "ElectronMass is the mass of an electron."];
If[Not@ValueQ[ElectronComptonWavelength::usage],ElectronComptonWavelength::usage =
"ElectronComptonWavelength is the electron Compton wavelength, given by \
PlanckConstant/(ElectronMass SpeedOfLight)."];
If[Not@ValueQ[ClassicalElectronRadius::usage],ClassicalElectronRadius::usage =
"ClassicalElectronRadius is the classical electron radius, an atomic constant."];
If[Not@ValueQ[ThomsonCrossSection::usage],ThomsonCrossSection::usage =
"ThomsonCrossSection is the Thomson cross section, an atomic constant."];
If[Not@ValueQ[ElectronMagneticMoment::usage],ElectronMagneticMoment::usage =
"ElectronMagneticMoment is the electron magnetic moment."];
If[Not@ValueQ[ElectronGFactor::usage],ElectronGFactor::usage = "ElectronGFactor is the electron g-factor."];

(* Muon *)

If[Not@ValueQ[MuonMass::usage],MuonMass::usage = "MuonMass is the mass of a muon."];
If[Not@ValueQ[MuonMagneticMoment::usage],MuonMagneticMoment::usage = "MuonMagneticMoment is the muon magnetic moment."];
If[Not@ValueQ[MuonGFactor::usage],MuonGFactor::usage = "MuonGFactor is the muon g-factor."];

(* Proton *)

If[Not@ValueQ[ProtonComptonWavelength::usage],ProtonComptonWavelength::usage =
"ProtonComptonWavelength the proton Compton wavelength, given by \
PlanckConstant/(ProtonMass SpeedOfLight)."];
If[Not@ValueQ[ProtonMagneticMoment::usage],ProtonMagneticMoment::usage =
"ProtonMagneticMoment is the proton magnetic moment."]; (* scalar magnitude *)
If[Not@ValueQ[ProtonMass::usage],ProtonMass::usage = "ProtonMass is the mass of a proton."];

(* Neutron *)

If[Not@ValueQ[NeutronComptonWavelength::usage],NeutronComptonWavelength::usage =
"NeutronComptonWavelength the neutron Compton wavelength, given by \
PlanckConstant/(NeutronMass SpeedOfLight)."];
If[Not@ValueQ[NeutronMagneticMoment::usage],NeutronMagneticMoment::usage =
"NeutronMagneticMoment is the neutron magnetic moment."]; (* scalar magnitude *)
If[Not@ValueQ[NeutronMass::usage],NeutronMass::usage = "NeutronMass is the mass of a neutron."];

(* Deuteron *)

If[Not@ValueQ[DeuteronMass::usage],DeuteronMass::usage = "DeuteronMass is the mass of a neutron."];
If[Not@ValueQ[DeuteronMagneticMoment::usage],DeuteronMagneticMoment::usage =
"DeuteronMagneticMoment is the deuteron magnetic moment."];

(* Electroweak *)
If[Not@ValueQ[WeakMixingAngle::usage],WeakMixingAngle::usage = "WeakMixingAngle is a physical constant."];

(* ==================== PHYSICO-CHEMICAL CONSTANTS ==================== *)

If[Not@ValueQ[AvogadroConstant::usage],AvogadroConstant::usage =
"AvogadroConstant is the number of molecules in one mole or gram molecular \
weight of a substance."];
If[Not@ValueQ[FaradayConstant::usage],FaradayConstant::usage =
"FaradayConstant is the product of the Avogadro constant (AvogadroConstant) \
and the elementary charge (ElectronCharge)."];
If[Not@ValueQ[MolarGasConstant::usage],MolarGasConstant::usage =
"MolarGasConstant is a physico-chemical constant."];
If[Not@ValueQ[BoltzmannConstant::usage],BoltzmannConstant::usage =
"BoltzmannConstant is the ratio of the universal gas constant \
(MolarGasConstant) to Avogadro's number (AvogadroConstant)."];
If[Not@ValueQ[MolarVolume::usage],MolarVolume::usage =
"MolarVolume is the volume occupied by a mole or a gram molecular weight of any \
gas measured at standard conditions."];
If[Not@ValueQ[SackurTetrodeConstant::usage],SackurTetrodeConstant::usage =
"SackurTetrodeConstant (absolute entropy constant), is a physico-chemical \
constant."];
If[Not@ValueQ[StefanConstant::usage],StefanConstant::usage =
"StefanConstant is the Stefan-Boltzmann constant, a universal constant of \
proportionality between the radiant emittance of a black body and the \
fourth power of the body's absolute temperature."];

(* ======================== ASTRONOMICAL CONSTANTS ===================== *)

If[Not@ValueQ[AgeOfUniverse::usage],AgeOfUniverse::usage = "AgeOfUniverse is the age of the Universe, a physical \
constant."];
If[Not@ValueQ[CosmicBackgroundTemperature::usage],CosmicBackgroundTemperature::usage=
"CosmicBackgroundTemperature is the temperature of the cosmic background \
radiation."];
If[Not@ValueQ[EarthMass::usage],EarthMass::usage = "EarthMass is the mass of the Earth, a physical constant."];
If[Not@ValueQ[EarthRadius::usage],EarthRadius::usage = "EarthRadius is the radius of the Earth, a physical \
constant."];
If[Not@ValueQ[HubbleConstant::usage],HubbleConstant::usage = "HubbleConstant is a measure of the rate at which the \
expansion of the universe varies with distance."];
If[Not@ValueQ[SolarRadius::usage],SolarRadius::usage = "SolarRadius is a physical constant."];
If[Not@ValueQ[SolarSchwarzschildRadius::usage],SolarSchwarzschildRadius::usage =
"SolarSchwarzschildRadius is a physical constant."];
If[Not@ValueQ[SolarConstant::usage],SolarConstant::usage =
"SolarConstant is the rate at which solar radiation is received outside the \
earth's atmosphere on a surface normal to the incident radiation and at the \
earth's mean distance from the sun, integrated across all wavelengths. Also \
known as total solar irradiance."];
If[Not@ValueQ[GalacticUnit::usage],GalacticUnit::usage =
"GalacticUnit is the approximate distance of the Sun from the center of the Milky Way \
Galaxy."];
If[Not@ValueQ[SolarLuminosity::usage],SolarLuminosity::usage = "SolarLuminosity is a physical constant."];



(* ========================== OTHER CONSTANTS ========================== *)

If[Not@ValueQ[SpeedOfSound::usage],SpeedOfSound::usage =
"SpeedOfSound is the speed of sound at sea level in the standard atmosphere."];
If[Not@ValueQ[IcePoint::usage],IcePoint::usage =
"IcePoint is the temperature at which a mixture of air-saturated pure \
water and pure ice may exist in equilibrium at a pressure of one \
standard atmosphere."];


Begin["`Private`"]

AccelerationDueToGravity = (980665/10^5) Meter/Second^2 (* exact: HCAP 80, p. 1-6 *)

AgeOfUniverse = 4.7*^17 Second

AvogadroConstant = 6.02214179*^23 Mole^-1  (* CODATA 2006 *)

BohrRadius = 0.52917720859*^-10 Meter  (* infinite mass nucleus : CODATA 2006 *)

(* BohrMagnetron is introduced in Units.m...
BohrMagneton = 9.2740154*^-24 Ampere Meter^2
*)

BoltzmannConstant = 1.3806504*^-23 Joule/Kelvin  (* CODATA 2006 *)

CosmicBackgroundTemperature = 2.726 Kelvin

ClassicalElectronRadius = 2.8179402894*^-15 Meter  (* CODATA 2006 *)

DeuteronMagneticMoment = 0.433073465*^-26 Joule/Tesla  (* CODATA 2006 *)

DeuteronMass = 3.34358320*^-27 Kilogram  (* CODATA 2006 *)

EarthMass = 5.9742*^24 Kilogram  (* HCAP 80, p. 14-3 *)

EarthRadius = 6378140 Meter   (* equatorial radius: HCAP 80, p. 14-1.
                                 The IUGG value for this is 6378136 m. *)

ElectronCharge = 1.602176487*^-19 Coulomb  (* CODATA 2006 *)

ElectronComptonWavelength = 2.4263102175*^-12 Meter (* CODATA 2006 *)

ElectronGFactor = -2.0023193043622	(* -2(1+Subscript[\[Alpha], e]) : CODATA 2006 *)

ElectronMagneticMoment = -928.476377*^-26 Joule/Tesla  (* CODATA 2006 *)

ElectronMass = 9.10938215*^-31 Kilogram  (* CODATA 2006 *)

FaradayConstant = 96485.3399 Coulomb/Mole  (* CODATA 2006 *)

FineStructureConstant = 7.2973525376*^-3  (* CODATA 2006 *)

GalacticUnit = 2.6*^20 Meter (* approximate value, 8.5 kPsc, derived from
                                 various atronomy texts; actual distance
                                 believed to vary from 8.4 to 9.7 kPsc *)

GravitationalConstant = 6.67428*^-11 Newton Meter^2 Kilogram^-2  (* CODATA 2006 *)

HubbleConstant = 3.2*^-18 Second^-1

IcePoint = 273.15 Kelvin (* F-88 CRC Hdbk Chem & Phys, 68th Ed. *)

MagneticFluxQuantum = 2.067833667*^-15 Weber  (* h/(2 e) *) (* CODATA 2006 *)

MolarGasConstant = 8.314472 Joule Kelvin^-1 Mole^-1  (* CODATA 2006 *)

MolarVolume = 22.413996*^-3 Meter^3/Mole 
    (* ideal gas, T = 273.15 K, P = 101.325 kPa : CODATA 2006 *)

MuonGFactor = -2.0023318414 (* CODATA 2006 *)

MuonMagneticMoment = -4.49044786*^-26 Joule/Tesla (* CODATA 2006 *)

MuonMass = 1.88353130*^-28 Kilogram  (* CODATA 2006 *)

NeutronComptonWavelength = 1.3195908951*^-15 Meter (* CODATA 2006 *)

NeutronMagneticMoment = -0.96623641*^-26 Joule/Tesla (* CODATA 2006 *)

NeutronMass = 1.674927211*^-27 Kilogram  (* CODATA 2006 *)

PlanckConstant = 6.62606896*^-34 Joule Second  (* CODATA 2006 *)

PlanckConstantReduced = PlanckConstant / (2 Pi)  (* definition *)

PlanckMass = 2.17644*^-8 Kilogram  (* CODATA 2006 *)

ProtonComptonWavelength = 1.3214098446*^-15 Meter (* CODATA 2006 *)

ProtonMagneticMoment = 1.410606662*^-26 Joule/Tesla (* CODATA 2006 *)

ProtonMass = 1.672621637*^-27 Kilogram  (* CODATA 2006 *)

QuantizedHallConductance = 3.8740458502*^-5 Ampere/Volt (* e^2/h *)
   (* computed from CODATA 2006, which gives a value for 2*e^2/h *)

RydbergConstant = 10973731.568527 Meter^-1  (* CODATA 2006 *)

SackurTetrodeConstant = -1.1517047 (* 100 kPa : CODATA 2006 *)

SpeedOfLight = 299792458 Meter/Second  (* by definition: verified CODATA 2006 *)

SpeedOfSound = 340.29205 Meter/Second  (* standard atmosphere *)

SolarConstant = 1.3661*^3 Watt/Meter^2 
    (* used in draft ISO standard DIS 21348, see "Status of ISO
       Draft International Standard for Determiing Solar Irradiances
       (DIS 21348)", Tobiska, W. Kent; Nusinov, Anatoliy A.,
       J. Adv. Space Research, in press. Note that this is not in
       fact a constant, but variabe over time, with a cycle imposed
       by the solar cycle. *)

SolarLuminosity = 3.84*^26 Watt  (* computed by definition from
      the SolarConstant, verified by literature citations. Definition
      is 4 Pi (1 AU)^2 * SolarConstant, given 1 AU in meters. *)

SolarRadius = 6.9599*^8 Meter  (* HCAP 80, p. 14-2 *)

SolarSchwarzschildRadius = 2.95325008*^3 Meter

StefanConstant = 5.670400*^-8 Watt Meter^-2 Kelvin^-4  (* CODATA 2006 *)

ThomsonCrossSection = 0.6652458558*^-28 Meter^2  (* CODATA 2006 *)

VacuumPermeability = 4 Pi * 10^-7 Volt Second/Meter/Ampere  (* definition *)

VacuumPermittivity = 1/(VacuumPermeability SpeedOfLight^2) (* exact, definition *)

WeakMixingAngle = 0.22255 (* Sin[ThetaW]^2 : CODATA 2006 *)

End[]

EndPackage[]

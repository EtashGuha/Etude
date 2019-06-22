(* ::Package:: *)

Begin["System`Convert`SP3Dump`"]


ImportExport`RegisterImport[
  "SP3",
  {
	"Elements" :> getSP3Elements,
	ImportSP3
  },
  "FunctionChannels" -> {"Streams"},
  "AvailableElements" -> {"Agency", "ClockCorrectionChangeRateErrors", "ClockCorrectionChangeRates", "ClockCorrectionErrors",
			"ClockCorrections", "Comments", "CoordinateSystem", "DataDescriptorCode", "GPSDate", "ModifiedJulianDays",
			"OrbitAccuracy", "OrbitalDataType", "PositionCorrelations", "PositionErrors", "Positions", "SatelliteID", "SatelliteSystems",
			"TimeInterval", "Times", "TimeSystem", "Velocities", "VelocityCorrelations", "VelocityErrors", "Version"},
  "DefaultElement" -> "Positions"
]


End[]

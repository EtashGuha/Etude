(* ::Package:: *)

Begin["System`Convert`AffymetrixDump`"]


ImportExport`RegisterImport[
  "Affymetrix",
  {
    "Elements"            :> ImportAffymetrixElement,
    "Header"              :> ImportAffymetrixHeader,
    "RawHeader"           :> ImportAffymetrixRawHeader,
    {"Data", s_String}    :> (ImportAffymetrixProbe[s][##]&),
    ImportAffymetrixData
  },
  "Sources" -> ImportExport`DefaultSources[{
                  "Affymetrix",
                  "AffymetrixCEL",
                  "AffymetrixCHP",
                  "AffymetrixGIN",
                  "AffymetrixPSI",
                  "AffymetrixCDF"
               }],
  "FunctionChannels" -> {"Streams"},
  "DefaultElement" -> "Data",
  "AvailableElements" -> {
            "Alleles", "ConfidenceValues", "Contrasts", "Data", "DataErrors",
			"DetectionSignificances", "DetectionStates", "ForcedCalls", "Header",
			"Masks", "Outliers", "PixelRanges", "ProbePairs", "ProbePairsUsed",
			"ProbeSetNames", "QCData", "Quantifications", "RawHeader", "SignalA",
			"SignalB", "Strengths", "Subgrids"},
  "BinaryFormat"-> True
]


End[]

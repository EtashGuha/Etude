Begin["System`Convert`AgilentMicroarrayDump`"]

ImportExport`RegisterImport[
 "AgilentMicroarray",
 {
     "Header":>getAgilentMetadata,
     "Statistics":>getAgilentStatistics,
     "LabeledFeatureTable":>(getAgilentFeatureTable[True][##]&),
     "FeatureTable":>(getAgilentFeatureTable[False][##]&),
    AgilentFallthroughDummy (*See note in converter*)
  },
  "AvailableElements" -> {"FeatureTable", "Header", "LabeledFeatureTable", "Statistics"},
  "DefaultElement"-> "FeatureTable"
]

End[]
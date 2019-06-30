(* ::Package:: *)

Begin["System`Convert`HDF5Dump`"]


ImportExport`RegisterImport[
 "HDF5",
 {
 	"DataEncoding"																	:> ImportDatasetMetadata["DataEncoding", None],
 	"Dimensions"																	:> ImportDatasetMetadata["Dimensions", None],
 	"DataFormat"			 														:> ImportDatasetMetadata["DataFormat", None],
 	{elem:("DataEncoding"|"Dimensions"|"DataFormat"), All|"All"}					:> ImportDatasetMetadata[elem, All],
 	{elem:("DataEncoding"|"Dimensions"|"DataFormat"), d:(_String|_Integer|_List)}	:> ImportDatasetMetadata[elem, d],
 	elem_String /; StringMatchQ[elem, "Annotations"]								:> ImportAnnotations,
 	"Attributes"																	:> ImportAttributes[None],
 	{"Attributes", All|"All"} 														:> ImportAttributes[All],
 	{"Attributes", obj:(_String|_Integer|_List)} 									:> ImportAttributes[obj],
 	"Data" 																			:> ReadAllDatasets[None],
 	{"Data", All|"All"} 															:> ReadAllDatasets[All], 
 	{"Data", obj:(_String|_Integer)} 												:> ReadDataset[obj, "Data"],
 	{"Data", l_List} 																:> ReadDatasets[l],
 	{"Data", obj_String, lvl_Integer} 												:> ReadDatasetsFlat[obj, lvl],
	"RawData" 																		:> ReadAllDatasetsRaw[None],
	{"RawData", All|"All"} 															:> ReadAllDatasetsRaw[All],
	{"RawData", obj:(_String|_Integer)} 											:> ReadDatasetRaw[obj],
	{"RawData", l_List} 															:> ReadDatasetsRaw[l],
	{"RawData", obj_String, lvl_Integer} 											:> ReadDatasetsFlatRaw[obj, lvl],
 	"Datasets" 																		:> ImportDatasetNames,
 	{"Datasets", obj:(_String|_Integer)} 											:> ReadDataset[obj, "Datasets"],
 	elem_String /; StringMatchQ[elem, "GroupData"]									:> ReadGroupByName[None],
 	{"GroupData", All|"All"}		 												:> ReadGroupByName[All],
 	{"GroupData", gr:(_String|_List)} 												:> ReadGroupByName[gr],
 	{"GroupData", gr:(_String|_List), lvl_Integer} 									:> ReadGroupByName[gr, lvl],
 	elem_String /; StringMatchQ[elem, "GroupInformation"]							:> ImportGroupInfo[None],
 	{"GroupInformation", All|"All"}		 											:> ImportGroupInfo[All],
 	{"GroupInformation", gr:(_String|_List)} 										:> ImportGroupInfo[gr],
 	{"GroupInformation", gr:(_String|_List), lvl_Integer} 							:> ImportGroupInfo[gr, lvl],
 	"Groups" 																		:> ImportGroupNames,
 	"StructureGraph" 																:> ImportFileStructure[All],
  	{"StructureGraph", obj_String}													:> ImportFileStructure[obj],
 	"StructureGraphLegend"															:> ImportGraphLegend,
 	"Summary"																		:> ImportSummary,
 	"Elements"																		:> GetListOfElements,
 	dsetName_String																	:> ReadDataset[dsetName],
  	dsetNumber_Integer																:> ReadDataset[dsetNumber],	
    ImportDatasetNames
 },
 "Sources" -> ImportExport`DefaultSources[{"HDF5", "DataCommon"}],
 "AvailableElements" -> {"Attributes", "Data", "DataEncoding", "DataFormat", "Datasets", "Dimensions", "Elements", "GroupData", "GroupInformation", 
 	"Groups", "RawData", "StructureGraph", "StructureGraphLegend", "Summary", _String},
 "DefaultElement" -> "Datasets",
 "BinaryFormat" -> True
]


End[]

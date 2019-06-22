

Begin["System`Convert`MESHDump`"]

ImportExport`RegisterImport[
	"MESH",
	{
		"Elements"		:> GetElements,
		"MeshRegion"	:> CreateMeshRegion,
		"ElementMesh"	:> CreateElementMesh,
		CreateMeshRegion
	}
	, "AvailableElements" -> {"Elements", "MeshRegion", "ElementMesh"}
	, "BinaryFormat" -> False
	, "DefaultElement" -> "MeshRegion"
	, "FunctionChannels" -> {"FileNames"}
	, "Options" -> {}
	, "Sources" -> {"Convert`Common3D`", "Convert`MESH`"}
]


End[]

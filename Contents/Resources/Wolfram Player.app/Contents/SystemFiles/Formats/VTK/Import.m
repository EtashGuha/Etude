(* ::Package:: *)

Begin["System`Convert`VTKDump`"]


ImportExport`RegisterImport[
	"VTK",
	System`Convert`VTKDump`ImportVTK,
	{
		"CuboidObjects"        :> (System`Convert`VTKDump`ImportVTKObjects[(Cuboid@@#&), "CuboidData", ##]&),
		"Graphics3D" 		:> System`Convert`VTKDump`ImportVTKGraphics,
		"GraphicsComplex" 	:> System`Convert`VTKDump`ImportVTKGraphicsComplex,
		"PolygonObjects" 	:> (System`Convert`VTKDump`ImportVTKObjects[Polygon, "PolygonData", ##]&),
		"LineObjects"        :> (System`Convert`VTKDump`ImportVTKObjects[Line, "LineData", ##]&),
		"PointObjects"        :> (System`Convert`VTKDump`ImportVTKObjects[Point, "PointData", ##]&)
	},
	"Sources" 			-> {"Convert`Common3D`", "Convert`VTK`"},
	"FunctionChannels" 	-> {"Streams"},
	"AvailableElements" -> {"BinaryFormat", "CuboidData", "CuboidObjects", "Graphics3D", "GraphicsComplex", "InvertNormals", "LineData", "LineObjects",
                            "PolygonData", "PolygonObjects", "PointObjects", "PointData", "VertexData", "VertexNormals", "VerticalAxis"},
	"DefaultElement"	-> "Graphics3D",
	"Options" 			-> {"BinaryFormat", "VerticalAxis"},
	"BinaryFormat"        -> True
]


End[]


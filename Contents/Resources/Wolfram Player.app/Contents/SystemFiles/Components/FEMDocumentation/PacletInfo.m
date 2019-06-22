(* Paclet Info File *)

(* created 2013/03/06*)

Paclet[
	Name -> "FEMDocumentation",
	Version -> "10.0.0",
	MathematicaVersion -> "10+",
	Extensions -> {
			{"Application", Context -> "NDSolve`FEM`"},

			(**)
			(* Package Documentation *)
			(**)
			(* Note: to trigger notebooks to be visible in the documentation
				tab in the paclet brwoser it may be nessesaty to create a test
				Symbol; then the path should be detected , Root -> "FEMDocumentation/Documentation"
				but should not be commited as then the in product docu does not work. *)
			{"Documentation", Language -> All, MainPage -> "Guides/FEMDocumentationOverview",
			Resources -> {

				{"ReferencePages/Messages/ToBoundaryMesh/fembdct",				"ReferencePages/Messages/ElementMesh/fembdct.nb"},
				{"ReferencePages/Messages/ToElementMesh/fembdct",				"ReferencePages/Messages/ElementMesh/fembdct.nb"},

				{"ReferencePages/Messages/ToBoundaryMesh/fembdel",				"ReferencePages/Messages/ElementMesh/fembdel.nb"},
				{"ReferencePages/Messages/ToElementMesh/fembdel",				"ReferencePages/Messages/ElementMesh/fembdel.nb"},

				{"ReferencePages/Messages/ToBoundaryMesh/femimq",				"ReferencePages/Messages/ElementMesh/femimq.nb"},
				{"ReferencePages/Messages/ToElementMesh/femimq",				"ReferencePages/Messages/ElementMesh/femimq.nb"},
				{"ReferencePages/Messages/ElementMeshInterpolation/femimq",		"ReferencePages/Messages/ElementMesh/femimq.nb"},

				{"ReferencePages/Messages/ToElementMesh/femtbmscf",				"ReferencePages/Messages/ToBoundaryMesh/femtbmscf.nb"},
				
				{"ReferencePages/Messages/ToBoundaryMesh/femtemnbb",			"ReferencePages/Messages/ToElementMesh/femtemnbb.nb"}

				}
			},

			(**)
			(* System extensions *)
			(**)
			{"Documentation", LinkBase->"WolframMathematica",
			Resources -> {

				{"ReferencePages/Messages/NDSolve/fembdct",						"ReferencePages/Messages/ElementMesh/fembdct.nb"},
				{"ReferencePages/Messages/NDSolveValue/fembdct",				"ReferencePages/Messages/ElementMesh/fembdct.nb"},
				{"ReferencePages/Messages/NDEigensystem/fembdct",				"ReferencePages/Messages/ElementMesh/fembdct.nb"},
				{"ReferencePages/Messages/NDEigenvalues/fembdct",				"ReferencePages/Messages/ElementMesh/fembdct.nb"},
				{"ReferencePages/Messages/ParametricNDSolve/fembdct",			"ReferencePages/Messages/ElementMesh/fembdct.nb"},
				{"ReferencePages/Messages/ParametricNDSolveValue/fembdct",		"ReferencePages/Messages/ElementMesh/fembdct.nb"},

				{"ReferencePages/Messages/NDSolve/fembdel",						"ReferencePages/Messages/ElementMesh/fembdel.nb"},
				{"ReferencePages/Messages/NDSolveValue/fembdel",				"ReferencePages/Messages/ElementMesh/fembdel.nb"},
				{"ReferencePages/Messages/NDEigensystem/fembdel",				"ReferencePages/Messages/ElementMesh/fembdel.nb"},
				{"ReferencePages/Messages/NDEigenvalues/fembdel",				"ReferencePages/Messages/ElementMesh/fembdel.nb"},
				{"ReferencePages/Messages/ParametricNDSolve/fembdel",			"ReferencePages/Messages/ElementMesh/fembdel.nb"},
				{"ReferencePages/Messages/ParametricNDSolveValue/fembdel",		"ReferencePages/Messages/ElementMesh/fembdel.nb"},

				{"ReferencePages/Messages/NDSolve/fembdnl",						"ReferencePages/Messages/InitializeBoundaryConditions/fembdnl.nb"},
				{"ReferencePages/Messages/NDSolveValue/fembdnl",				"ReferencePages/Messages/InitializeBoundaryConditions/fembdnl.nb"},
				{"ReferencePages/Messages/NDEigensystem/fembdnl",				"ReferencePages/Messages/InitializeBoundaryConditions/fembdnl.nb"},
				{"ReferencePages/Messages/NDEigenvalues/fembdnl",				"ReferencePages/Messages/InitializeBoundaryConditions/fembdnl.nb"},
				{"ReferencePages/Messages/ParametricNDSolve/fembdnl",			"ReferencePages/Messages/InitializeBoundaryConditions/fembdnl.nb"},
				{"ReferencePages/Messages/ParametricNDSolveValue/fembdnl",		"ReferencePages/Messages/InitializeBoundaryConditions/fembdnl.nb"},

				{"ReferencePages/Messages/NDSolve/femcmsd",						"ReferencePages/Messages/InitializePDECoefficients/femcmsd.nb"},
				{"ReferencePages/Messages/NDSolveValue/femcmsd",				"ReferencePages/Messages/InitializePDECoefficients/femcmsd.nb"},
				{"ReferencePages/Messages/NDEigensystem/femcmsd",				"ReferencePages/Messages/InitializePDECoefficients/femcmsd.nb"},
				{"ReferencePages/Messages/NDEigenvalues/femcmsd",				"ReferencePages/Messages/InitializePDECoefficients/femcmsd.nb"},
				{"ReferencePages/Messages/ParametricNDSolve/femcmsd",			"ReferencePages/Messages/InitializePDECoefficients/femcmsd.nb"},
				{"ReferencePages/Messages/ParametricNDSolveValue/femcmsd",		"ReferencePages/Messages/InitializePDECoefficients/femcmsd.nb"},

				{"ReferencePages/Messages/NDSolve/femcscd",						"ReferencePages/Messages/InitializePDECoefficients/femcscd.nb"},
				{"ReferencePages/Messages/NDSolveValue/femcscd",				"ReferencePages/Messages/InitializePDECoefficients/femcscd.nb"},
				{"ReferencePages/Messages/NDEigensystem/femcscd",				"ReferencePages/Messages/InitializePDECoefficients/femcscd.nb"},
				{"ReferencePages/Messages/NDEigenvalues/femcscd",				"ReferencePages/Messages/InitializePDECoefficients/femcscd.nb"},
				{"ReferencePages/Messages/ParametricNDSolve/femcscd",			"ReferencePages/Messages/InitializePDECoefficients/femcscd.nb"},
				{"ReferencePages/Messages/ParametricNDSolveValue/femcscd",		"ReferencePages/Messages/InitializePDECoefficients/femcscd.nb"},

				{"ReferencePages/Messages/NDSolve/femcsp",						"ReferencePages/Messages/InitializePDECoefficients/femcsp.nb"},
				{"ReferencePages/Messages/NDSolveValue/femcsp",					"ReferencePages/Messages/InitializePDECoefficients/femcsp.nb"},
				{"ReferencePages/Messages/NDEigensystem/femcsp",				"ReferencePages/Messages/InitializePDECoefficients/femcsp.nb"},
				{"ReferencePages/Messages/NDEigenvalues/femcsp",				"ReferencePages/Messages/InitializePDECoefficients/femcsp.nb"},
				{"ReferencePages/Messages/ParametricNDSolve/femcsp",			"ReferencePages/Messages/InitializePDECoefficients/femcsp.nb"},
				{"ReferencePages/Messages/ParametricNDSolveValue/femcsp",		"ReferencePages/Messages/InitializePDECoefficients/femcsp.nb"},

				{"ReferencePages/Messages/NDSolve/femimq",						"ReferencePages/Messages/ElementMesh/femimq.nb"},
				{"ReferencePages/Messages/NDSolveValue/femimq",					"ReferencePages/Messages/ElementMesh/femimq.nb"},
				{"ReferencePages/Messages/NDEigensystem/femimq",				"ReferencePages/Messages/ElementMesh/femimq.nb"},
				{"ReferencePages/Messages/NDEigenvalues/femimq",				"ReferencePages/Messages/ElementMesh/femimq.nb"},
				{"ReferencePages/Messages/ParametricNDSolve/femimq",			"ReferencePages/Messages/ElementMesh/femimq.nb"},
				{"ReferencePages/Messages/ParametricNDSolveValue/femimq",		"ReferencePages/Messages/ElementMesh/femimq.nb"},
				{"ReferencePages/Messages/Interpolation/femimq",				"ReferencePages/Messages/ElementMesh/femimq.nb"},

				{"ReferencePages/Messages/NDSolve/femnlmdor",					"ReferencePages/Messages/InitializePDECoefficients/femnlmdor.nb"},
				{"ReferencePages/Messages/NDSolveValue/femnlmdor",				"ReferencePages/Messages/InitializePDECoefficients/femnlmdor.nb"},
				{"ReferencePages/Messages/ParametricNDSolve/femnlmdor",			"ReferencePages/Messages/InitializePDECoefficients/femnlmdor.nb"},
				{"ReferencePages/Messages/ParametricNDSolveValue/femnlmdor",	"ReferencePages/Messages/InitializePDECoefficients/femnlmdor.nb"},

				{"ReferencePages/Messages/NDSolve/femtbmscf",					"ReferencePages/Messages/ToBoundaryMesh/femtbmscf.nb"},
				{"ReferencePages/Messages/NDSolveValue/femtbmscf",				"ReferencePages/Messages/ToBoundaryMesh/femtbmscf.nb"},
				{"ReferencePages/Messages/NDEigensystem/femtbmscf",				"ReferencePages/Messages/ToBoundaryMesh/femtbmscf.nb"},
				{"ReferencePages/Messages/NDEigenvalues/femtbmscf",				"ReferencePages/Messages/ToBoundaryMesh/femtbmscf.nb"},
				{"ReferencePages/Messages/ParametricNDSolve/femtbmscf",			"ReferencePages/Messages/ToBoundaryMesh/femtbmscf.nb"},
				{"ReferencePages/Messages/ParametricNDSolveValue/femtbmscf",	"ReferencePages/Messages/ToBoundaryMesh/femtbmscf.nb"},

				{"ReferencePages/Messages/NDSolve/femtemnbb",					"ReferencePages/Messages/ToElementMesh/femtemnbb.nb"},
				{"ReferencePages/Messages/NDSolveValue/femtemnbb",				"ReferencePages/Messages/ToElementMesh/femtemnbb.nb"},
				{"ReferencePages/Messages/NDEigensystem/femtemnbb",				"ReferencePages/Messages/ToElementMesh/femtemnbb.nb"},
				{"ReferencePages/Messages/NDEigenvalues/femtemnbb",				"ReferencePages/Messages/ToElementMesh/femtemnbb.nb"},
				{"ReferencePages/Messages/NIntegrate/femtemnbb",				"ReferencePages/Messages/ToElementMesh/femtemnbb.nb"},
				{"ReferencePages/Messages/ParametricNDSolve/femtemnbb",			"ReferencePages/Messages/ToElementMesh/femtemnbb.nb"},
				{"ReferencePages/Messages/ParametricNDSolveValue/femtemnbb",	"ReferencePages/Messages/ToElementMesh/femtemnbb.nb"}


				}
			}
		}
]




(* Provide NDSolve`FEM` on context path *)
BeginPackage["NDSolve`FEM`"];

FEMPackageLoaded = True;

(* Function to construct the argument completion *)
mkArgCompletion = With[{cr = Rule[#1, #2]},                                                       FE`Evaluate[FEPrivate`AddSpecialArgCompletion[cr]]]&; 


(* ToElementMesh *)
EMOpts = ToString /@ Options[ToElementMesh][[All, 1]];
ToEMArgs = {"ToElementMesh", {
			{"Coordinates"},
			Join[{"MeshElements"}, EMOpts], 
			Join[{"BoundaryElements"}, EMOpts],
			Join[{"PointElements"}, EMOpts],
			EMOpts
		}};                                     
 
mkArgCompletion @@ ToEMArgs;


(* ToBoundaryMesh *)
BMOpts = ToString /@ Options[ToBoundaryMesh][[All, 1]];
ToBMArgs = {"ToBoundaryMesh", {
			{"Coordinates"},
			Join[{"BoundaryElements"}, BMOpts],
			Join[{"PointElements"}, BMOpts],
			BMOpts
		}};                                     
                                                                               
mkArgCompletion @@ ToBMArgs;


(* InitializePDECoefficients *)
InitPDEOpts = ToString /@ Options[InitializePDECoefficients][[All, 1]];
InitPDEArgs = {"InitializePDECoefficients", {
			{NDSolve`VariableData},
			{NDSolve`SolutionData},
			InitPDEOpts
		}};                                     
                                                                               
mkArgCompletion @@ InitPDEArgs;



EndPackage[];

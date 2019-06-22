Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]	


(******************************************************************************)

DeclarePostloadCode @ DefineCustomBoxes[ArrayOptimizer,
	sym_ArrayOptimizer :> ArrayOptimizerBoxes[sym],
	"UseUpValues" -> False
];

ArrayOptimizerBoxes[opt:ArrayOptimizer[assoc_Association]] := Scope[
	arrayInfo = Map[
		makeArrayItems[assoc[#], #]&, 
		{"Arrays", "Gradients"}
	];
	updates = Extract[assoc, "Updates", Dynamic];
	UnpackAssociation[assoc, method, arrays, gradients, operator, stateArrays];
	numStates = Length[stateArrays] / Length[arrays];
	info = {
		makeExecItem["Method", method],
		makeArrayItems[arrays, "Arrays"]
	};
	extraInfo = {
		makeExecItem["Operator", operator],
		makeExecItem["Updates", updates],
		makeExecItem["States/array", numStates]
	};
	BoxForm`ArrangeSummaryBox[
		ArrayOptimizer, opt, None,
		info, extraInfo,
		StandardForm
	]
];

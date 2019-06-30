BeginPackage["Compile`Core`Optimization`CollectBasicBlocks`"]

CollectBasicBlocksPass;

Begin["`Private`"] 

Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]



markChildren[data_, bb_] :=
	Module[ {},
		If[ data["keyExistsQ", bb["id"]],
			Return[]];
		data["associateTo", bb["id"] -> bb];
		Scan[markChildren[data,#]&, bb["getChildren"]];
	]


run[fm_, opts_] :=
	Module[{bbsFound = CreateReference[<||>], start, bbList},
		start = fm["firstBasicBlock"];
		markChildren[bbsFound, start];
		bbList = fm["basicBlockMap"]["values"];
		Scan[
			If[ !bbsFound["keyExistsQ", #["id"]],
				#["remove"]]&
			,bbList];
	]



RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"CollectBasicBlocks",
	"Removes unreachable BasicBlocks.",
	"This is necessary after dead branch elmination since entire graphs of  " <>
	"BasicBlocks may not be reachable. The technique is to start with the first BasicBlock " <>
	"add its ID to a list and then follow the children.  Any BasicBlock not found in this way "  <>
	"is unreachable and should be removed."
];

CollectBasicBlocksPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
	},
	"postPasses" -> {
	}
|>];

RegisterPass[CollectBasicBlocksPass]
]]


End[] 

EndPackage[]

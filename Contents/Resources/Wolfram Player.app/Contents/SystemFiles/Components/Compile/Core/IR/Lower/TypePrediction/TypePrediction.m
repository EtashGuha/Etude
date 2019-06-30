BeginPackage["Compile`Core`IR`Lower`TypePrediction`TypePrediction`"]


Begin["`Private`"]

Needs["Compile`Utilities`"]
Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`IR`Lower`Utilities`LoweringTools`"]
Needs["Compile`TypeSystem`FunctionData`KernelFunction`"]


$predictionData = <||>

expandFileName[name_] :=
	FileNameJoin[{Compile`Utilities`$CompileRootDirectory, "CompileResources", "TypePrediction",  name}]

loadFile[ fullName_] :=
	Module[ {contents},
		If[ !FileExistsQ[fullName],
			<|"error" -> True|>
			,
			contents = Quiet[ Get[ fullName]];
			<|"error" -> False, "contents" -> contents|>]
	]

setup[ st_] :=
	Module[ {indexResult, indexData},
		$predictionData = <||>;
		indexResult = loadFile[expandFileName["TypePrediction_Index.wl"]];
		indexData = Lookup[ indexResult, "contents", {}];
		If[ MatchQ[indexData, { (_Symbol) ...}],
			Map[ setupFunction[st, #]&, indexData]]
	]
	
setupFunction[st_, sym_] :=
	Module[ {file, data},
		file = expandFileName[ SymbolName[sym] <> ".wl"];
		If[!FileExistsQ[file],
			ThrowException[{"Cannot find TypePrediction resource file for", sym}]];
		If[ KeyExistsQ[$predictionData, sym],
			ThrowException[{"Multiple TypePrediction definitions for", sym}]];
		data = CreateReference[<|"name" -> sym, "file" -> file|>];
		AssociateTo[$predictionData, sym -> data];
		RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[sym], lower];
	]


lower[state_, mexpr_, opts_] :=
	Module[ {h = mexpr["head"], sym, data, fileData, contents, env},
		If[ !h["symbolQ"],
			ThrowException[{"Head expected to be a symbol", mexpr}]];
		sym = h["symbol"];
		data = Lookup[ $predictionData, sym, Null];
		If[ data === Null,
			ThrowException[{"Cannot find TypePrediction definition", sym}]];
		If[ !data["lookup", "isInitialized", False],
			fileData = loadFile[data["lookup", "file", ""]];
			contents = Lookup[fileData, "contents", {}];
			If[!MatchQ[ contents, { (_List -> _)...}],
				ThrowException[{"TypePrediction data not found for", sym}]];
			env = state["typeEnvironment"];
			env["reopen"];
			Scan[
				AddKernelFunction[ env, sym, #]&, contents];
			env["finalize"];
			data["associateTo", "isInitialized" -> True]];
		(*
		  Now just delegate to LowerGeneral to insert a CallInstruction
		*)
		LowerGeneral[state, mexpr, opts]
	]

RegisterCallback["RegisterPrimitive", setup]



End[]

EndPackage[]



BeginPackage["Compile`API`ExportLibrary`"]


Begin["`Private`"]

Needs["Compile`"]
Needs["Compile`CompileToLibrary`"]
Needs["CompiledLibrary`"]
Needs[ "Compile`API`Utilities`"]


prots = Unprotect[FunctionCompileExportLibrary]

Clear[FunctionCompileExportLibrary]

FunctionCompileExportLibrary::path = "Path `1` is not a string."
FunctionCompileExportLibrary::pathw = "Cannot open `1`."

Options[FunctionCompileExportLibrary] = 
	{
	CompilerOptions -> Automatic
	}


FunctionCompileExportLibrary[ path_, func_, opts:OptionsPattern[]] :=
	Module[ {res = Catch[
					iFunctionCompileExportLibrary[path, func, opts],
					_compileExportException,
					Part[#2,1]&]
	},
		res /; res =!= Null
	]

FunctionCompileExportLibrary[args___ /; (compileArgumentError[{args}, FunctionCompileExportLibrary, 2]; False)] :=
	Null


iFunctionCompileExportLibrary[pathIn_, funcIn_, opts:OptionsPattern[]] :=
	Module[ {func, lib, libName, path, libOpts = getCompilerOptions[FunctionCompileExportLibrary, compileExportException, {opts}]},
		path = checkFilePath[ testData, pathIn];
		func = checkFunctionForExport[testData, funcIn];	
		libName = FileNameTake[path];
		lib = CompileToLibrary[func, libOpts, "IncludeInput" -> True, "LibraryName" -> libName];
		If[!MatchQ[lib, CompiledLibrary[_String]],
			Throw[Null, compileExportException[$Failed]]];
		Quiet[If[FileExistsQ[path], DeleteFile[path]]];
		If[CopyFile[First[lib], path, OverwriteTarget->True] === $Failed,
			Message[FunctionCompileExportLibrary::pathw, path];
			Throw[Null, compileExportException[$Failed]]];
		path
	]



testData = <| 
	"head" -> FunctionCompileExportLibrary
|>



End[]

EndPackage[]

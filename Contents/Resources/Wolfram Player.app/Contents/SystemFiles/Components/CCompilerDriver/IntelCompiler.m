BeginPackage["CCompilerDriver`IntelCompiler`", {"CCompilerDriver`"}];

IntelCompiler::usage = "IntelCompiler can be passed to the \"Compiler\" option of CreateLibrary, CreateExecutable, or CreateObjectFile to compile with the Intel C++ compiler.";

IntelCompiler::nodir = "`1` was not found or is not a directory.";

Switch[$OperatingSystem,
	"Windows", Get["CCompilerDriver`IntelCompilerWindows`"],
	"Unix", Get["CCompilerDriver`IntelCompilerLinux`"],
	"MacOSX", Get["CCompilerDriver`IntelCompilerOSX`"]
]

EndPackage[];

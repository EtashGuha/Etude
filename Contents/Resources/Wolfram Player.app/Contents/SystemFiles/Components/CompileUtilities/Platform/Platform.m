
BeginPackage["CompileUtilities`Platform`Platform`"]

MachineIntegerSizeFromSystemID

Begin["`Private`"]


$IDSizeMap =
	<|
	"Windows" -> 32,
	"Windows-x86-64" -> 64,
	"Linux" -> 32,
	"Linux-x86-64" -> 64,
	"MacOSX" -> 32,
	"MacOSX-x86-64" -> 64,
	"Linux-ARM" -> 32
	|>

MachineIntegerSizeFromSystemID[systemID_String] :=
	Lookup[$IDSizeMap, systemID, 64]


End[]

EndPackage[]

Begin["NeuralNetworks`IOStub`Private`"]

$mxnetelems = {
	"Net", "UninitializedNet", "LayerAssociation", 
	"ArrayList", "ArrayAssociation", "RawArrayList", "RawArrayAssociation",
	"InputNames", "ArrayNames",
	"NodeGraph", "NodeGraphPlot", "NodeDataset"
};

getMXNetElements[___] := "Elements" -> ("ImportElements" /. System`ConvertersDump`FileFormatDataFull["MXNet"]);
importMXNet[prop_][file_, opts___] := prop -> NeuralNetworks`MXNetImport[file, prop, opts];

ImportExport`RegisterImport["MXNet",
	{
		prop:Apply[Alternatives, $mxnetelems] :> importMXNet[prop],
		"Elements" :> getMXNetElements
	}
	,
	"BinaryFormat" -> True,
	"DefaultElement" -> "Net",
	"AvailableElements" -> $mxnetelems,
	"Options" -> {"ArrayPath"}
]

End[]
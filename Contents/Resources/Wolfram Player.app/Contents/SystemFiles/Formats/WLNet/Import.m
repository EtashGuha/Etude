Begin["NeuralNetworks`IOStub`Private`"]

$wlnetelems = {
	"Net", "UninitializedNet", 
	"WLVersion", 
	"ArrayList", "ArrayAssociation", "RawArrayList", "RawArrayAssociation"
};

getWLNetElements[___] := "Elements" -> ("ImportElements" /. System`ConvertersDump`FileFormatDataFull["WLNet"]);
importWLNet[prop_][file_, opts___] := prop -> NeuralNetworks`WLNetImport[file, prop, opts];

ImportExport`RegisterImport["WLNet",
	{
		prop:Apply[Alternatives, $wlnetelems] :> importWLNet[prop],
		"Elements" :> getWLNetElements
	}
	,
	"BinaryFormat" -> True,
	"DefaultElement" -> "Net",
	"Extensions" -> {"*.wlnet"},
	"AvailableElements" -> $wlnetelems
]

End[]
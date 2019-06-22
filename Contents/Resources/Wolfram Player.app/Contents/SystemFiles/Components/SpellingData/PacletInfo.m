(* ::Package:: *)

Paclet[
	Name -> "SpellingData",
	Version -> "1.0.3",
	MathematicaVersion -> "11.1+",
	Description -> "SpellingData contains the spelling dictionaries",
	Extensions -> {
		{"Kernel", Root->"Kernel", Context->{"SpellingData`", "SpellingDataLoader`"}},
		{"SpellingDictionary", Root->"SpellingDictionaries"}
	}
]

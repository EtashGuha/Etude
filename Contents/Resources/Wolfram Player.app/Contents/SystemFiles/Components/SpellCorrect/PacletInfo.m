(* Paclet Info File *)

(* created 2009/12/15*)

Paclet[
  Name -> "SpellCorrect",
  Version -> "1.0.0",
  MathematicaVersion -> "10.4+",
  Loading -> Automatic,
  Extensions -> {
  	{"Kernel",
			Root->"Kernel",
			Context->{
                "SpellCorrectLoader`",
                "SpellCorrect`"
            },
			Symbols-> {
				"System`DictionaryWordQ",
				"System`SpellingCorrectionList"
			}
  	},
    {"LibraryLink", Root -> "LibraryResources"}
}]

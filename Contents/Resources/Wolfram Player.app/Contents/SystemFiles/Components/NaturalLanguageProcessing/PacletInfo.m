(* Paclet Info File *)

(* created 2016/02/18*)

Paclet[
	Name -> "NaturalLanguageProcessing",
	Version -> "12.8",
	MathematicaVersion -> "11+",
	Description -> "Natural Language Processing Utilities",
	Creator -> "Gopal Sarma <gopals@wolfram.com>, Jerome Louradour <jeromel@wolfram.com>",
	Loading -> Automatic,
	Extensions -> 
		{
			{"Resource", Root -> "Resources", Resources ->
				{
					"WLTagger",
					"stanfordnlp/target/classes/lib"
				}
			},
			{"Kernel", Symbols -> 
				{
					"System`WordStem",
					"System`LanguageIdentify",
					"System`TextPosition",
					"System`TextCases",
					"System`TextContents",
					"System`TextElement",
					"System`TextStructure",
					"System`FindTextualAnswer",
					"System`Containing",
					"System`VerifyInterpretation"
				}
			, Context -> 
				{"NaturalLanguageProcessingLoader`", "NaturalLanguageProcessing`"}
			}
		}
]



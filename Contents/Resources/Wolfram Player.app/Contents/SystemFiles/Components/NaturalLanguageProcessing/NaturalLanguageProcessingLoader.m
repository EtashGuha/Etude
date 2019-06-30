(* All loading of the paclet's Wolfram Language code should go through this file. *)

(* Developer maintains this list of symbols.
   autoloadSymbols must agree with the symbols listed in the Kernel extension in the PacletInfo.m file.
*)
NaturalLanguageProcessing`Private`autoloadSymbols = {
	"System`WordStem",
	"System`LanguageIdentify",
		
	"System`TextPosition",
	"System`TextCases",
	"System`TextContents",
			
	(*"System`Containing",*)
	(*"System`VerifyInterpretation",*)
			
	"System`TextElement",
	"System`TextStructure",
				
	"System`FindTextualAnswer"
};

NaturalLanguageProcessing`Private`symsToProtect = {};


PacletManager`Package`loadWolframLanguageCode["NaturalLanguageProcessing", "NaturalLanguageProcessing`", DirectoryName[$InputFileName], "Extern.m",
	"AutoUpdate" -> True,
	"ForceMX" -> False, 
	"Lock" -> False,
	"AutoloadSymbols" -> NaturalLanguageProcessing`Private`autoloadSymbols,
	"HiddenImports" -> {"PacletManager`", "Developer`", "GeneralUtilities`", "Macros`", "NeuralNetworks`", "JLink`"},
	"SymbolsToProtect" -> NaturalLanguageProcessing`Private`symsToProtect
]

(***** General messages *****)

General::nlpstrfile = "String, File, ContentObject or non-empty list of these textual objects expected for the text at position ``.";
General::nlpstrse = "String or non-empty list of strings expected at position ``.";
General::nlpstr = "String or non-empty list of strings expected at position ``.";
General::nlpstrlist = "Non-empty list of strings expected at position ``.";

General::nlpnear3 = "Third argument should be a non-negative integer.";

General::nlpperfgoal = "Value of option PerformanceGoal -> `1` should be Automatic, \"Speed\" or \"Quality\".";

General::nlpverifint = "Boolean expected for option VerifyInterpretation";
General::nlpaccthr = "Real between 0 and 1 expected for option AcceptanceThreshold";

General::nlpint3 = "Positive integer or Infinity expected at position 3.";

General::nlpnumalt = "Invalid number of alternatives at position 3. The number of alternatives must be a positive integer.";
General::nlpoverlap = "Value of type `` is not a valid for option Overlaps. Possible values are True, All or False.";
General::nlpmergeprob = "`1` is not an available option for MergeProbabilities. Possible options are `2` or a function that applies to a list of numerical values (Total, Identity, \[Ellipsis]).";
General::nlptfidf = "TF-IDF based text filtering is disabled. Results might differ.";

General::nlpjava = "Java is not enabled. Results might differ."; (* The Stanford tools cannot be used because ... *)
General::nlpjavafailure = "Java is not enabled. `1` cannot be evaluated.";
General::nlpinterpreter = "Results may be wrong because Interpreter is not available. ``";


(* cf in NeuralFunctions : General::nnlderr = "An internal error occurred while loading a neural net. Please try again."; *)
General::nlpnetmodel = "An internal error occurred while loading a neural net needed to evaluate the function. Please try again.";

General::nlpnocf = "No data available for `1`.";
General::nlpdlfail = "Error downloading or installing data `1`.";
General::nlpallowinternet = "Data `1` could not be downloaded. You will need to allow Internet use via the Help/Internet Connectivity dialog box, or by setting $AllowInternet = True.";

(* Semantic Distance *)
General::embrank = "The sentence embedding has unexpected rank `1`.";
General::embarray = "The sentence embedding is not a regular array.";
General::nlpnear3 = "Third argument should be a non-negative integer.";

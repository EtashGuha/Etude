(* All loading of the paclet's Wolfram Language code should go through this file. *)

(* Developer maintains this list of symbols.
   autoloadSymbols must agree with the symbols listed in the Kernel extension in the PacletInfo.m file.
*)
Iconize`Private`autoloadSymbols = {          
				"Iconize`IconizedImage"
         }
    
Iconize`Private`symsToProtect = {};

PacletManager`Package`loadWolframLanguageCode["Iconize", "Iconize`", DirectoryName[$InputFileName], "NotebookRasterize.m",
         "AutoUpdate" -> True,
         "ForceMX" -> TrueQ[Iconize`$ForceMX], 
         "Lock" -> False,
         "AutoloadSymbols" -> Iconize`Private`autoloadSymbols,
         "HiddenImports" -> {"PacletManager`", "Developer`", "GeneralUtilities`", "Macros`"},
         "SymbolsToProtect" -> Iconize`Private`symsToProtect
]
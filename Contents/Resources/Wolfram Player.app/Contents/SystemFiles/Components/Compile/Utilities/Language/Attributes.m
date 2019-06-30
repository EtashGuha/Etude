
(*
Generated using

	allSyms = WolframLanguageData[All, "Name"];
	allSymsInfo = WolframLanguageData[All, "Attributes"];

	$SystemAttributes =
		Map[Map[FromEntity]] @*
		Select[ListQ] @
 		AssociationThread[allSyms -> allSymsInfo]
*)

BeginPackage["Compile`Utilities`Language`Attributes`"]

$SystemAttributes

Begin["`Private`"]

(*

The big long association of symbols->attrs that used to be here has been removed in the interest of
faster loading.

If this is ever added back, consider making it an MX file or using some lazier way.

*)

$SystemAttributes = <||>

End[]

EndPackage[]
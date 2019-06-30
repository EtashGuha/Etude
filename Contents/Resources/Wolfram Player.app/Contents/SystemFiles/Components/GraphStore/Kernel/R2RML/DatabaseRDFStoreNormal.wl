(* R2RML: RDB to RDF Mapping Language *)
(* https://www.w3.org/TR/r2rml/ *)

BeginPackage["GraphStore`R2RML`DatabaseRDFStoreNormal`", {"GraphStore`", "GraphStore`R2RML`"}];
Begin["`Private`"];

DatabaseRDFStore /: Normal[store_DatabaseRDFStore, Repeated[DatabaseRDFStore, {0, 1}]] := With[{res = Catch[iDatabaseRDFStoreNormal[store], $failTag]}, res /; res =!= $failTag];


fail[___, f_Failure, ___] := Throw[f, $failTag];
fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


clear[iDatabaseRDFStoreNormal];
iDatabaseRDFStoreNormal[store_DatabaseRDFStore] := With[
	{data = store //
	Replace[f_?FailureQ :> fail[f]] //
	SPARQLSelect[Alternatives[
		RDFTriple[SPARQLVariable["x"], SPARQLVariable["y"], SPARQLVariable["z"]],
		SPARQLGraph[SPARQLVariable["g"], RDFTriple[SPARQLVariable["x"], SPARQLVariable["y"], SPARQLVariable["z"]]]
	]] //
	Replace[{f_?FailureQ :> fail[f], Except[_List] :> fail[]}] //
	GroupBy[Key["g"] -> Function[RDFTriple[#x, #y, #z]]]},
	With[
		{key = Missing["KeyAbsent", "g"]},
		RDFStore[
			Lookup[data, key, {}],
			KeyDrop[data, key]
		]
	]
];


End[];
EndPackage[];

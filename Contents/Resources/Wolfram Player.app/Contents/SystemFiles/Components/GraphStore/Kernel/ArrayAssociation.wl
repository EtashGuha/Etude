BeginPackage["GraphStore`ArrayAssociation`"];

ArrayAssociation;

Begin["`Private`"];

ArrayAssociation /: ArrayDepth[ArrayAssociation[_, structure_List]] := Max[LengthWhile[structure, MatchQ[None]], LengthWhile[structure, MatchQ[_List]]];
ArrayAssociation /: CountDistinct[ArrayAssociation[data_, _]] := CountDistinct[data];
ArrayAssociation /: DeleteDuplicates[a_ArrayAssociation, args___] := With[{res = Catch[iDeleteDuplicates[a, args], $failTag]}, res /; res =!= $failTag];
ArrayAssociation /: DeleteDuplicatesBy[a_ArrayAssociation, args___] := With[{res = Catch[iDeleteDuplicatesBy[a, args], $failTag]}, res /; res =!= $failTag];
ArrayAssociation /: Dimensions[a : ArrayAssociation[data_, _]] := Dimensions[data, ArrayDepth[a]];
ArrayAssociation /: Drop[a_ArrayAssociation, args___] := With[{res = Catch[iDrop[a, args], $failTag]}, res /; res =!= $failTag];
ArrayAssociation /: Join[x___, a_ArrayAssociation, y___] := With[{res = Catch[iJoin[x, a, y], $failTag]}, res /; res =!= $failTag];
ArrayAssociation /: JoinAcross[x___, a_ArrayAssociation, y__] := With[{res = Catch[iJoinAcross[x, a, y], $failTag]}, res /; res =!= $failTag];
ArrayAssociation /: KeyDrop[a_ArrayAssociation, keys_] := With[{res = Catch[iKeyDrop[a, keys], $failTag]}, res /; res =!= $failTag];
ArrayAssociation /: KeyTake[a_ArrayAssociation, keys_] := With[{res = Catch[iKeyTake[a, keys], $failTag]}, res /; res =!= $failTag];
ArrayAssociation /: Length[ArrayAssociation[data_, _List]] := Length[data];
Unprotect[Lookup]; Lookup[a_ArrayAssociation, args__] := With[{res = Catch[iLookup[a, args], $failTag]}, res /; res =!= $failTag]; Protect[Lookup];
ArrayAssociation /: Normal[a_ArrayAssociation, Repeated[ArrayAssociation, {0, 1}]] := With[{res = Catch[iNormal[a], $failTag]}, res /; res =!= $failTag];
ArrayAssociation /: Part[a_ArrayAssociation, args___] := With[{res = Catch[iPart[a, args], $failTag]}, res /; res =!= $failTag];
ArrayAssociation /: Pick[a_ArrayAssociation, args__] := With[{res = Catch[iPick[a, args], $failTag]}, res /; res =!= $failTag];
ArrayAssociation /: Sort[a_ArrayAssociation, args___] := With[{res = Catch[iSort[a, args], $failTag]}, res /; res =!= $failTag];
ArrayAssociation /: Take[a_ArrayAssociation, args___] := With[{res = Catch[iTake[a, args], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


(* delete duplicates *)
clear[iDeleteDuplicates];
iDeleteDuplicates[ArrayAssociation[data_, structure : {None, ___}], test : Repeated[_, {0, 1}]] := ArrayAssociation[DeleteDuplicates[data, test], structure];
iDeleteDuplicates[ArrayAssociation[data_List, {keys_List, srest___}], test_ : SameQ] := With[
	{d = DeleteDuplicates[Thread[{data, keys}], test[First[#1], First[#2]] &]},
	ArrayAssociation[d[[All, 1]], {d[[All, 2]], srest}]
];


(* delete duplicates by *)
clear[iDeleteDuplicatesBy];
iDeleteDuplicatesBy[ArrayAssociation[data_, {None, srest___}], f_] := DeleteDuplicatesBy[Normal[ArrayAssociation[#, {srest}]] & /@ data, f];
iDeleteDuplicatesBy[ArrayAssociation[data_List, {keys_List, srest___}], f_] := With[
	{d = DeleteDuplicatesBy[Thread[{Normal[ArrayAssociation[#, {srest}]] & /@ data, keys}], First /* f]},
	ArrayAssociation[d[[All, 1]], {d[[All, 2]]}]
];


(* drop *)
clear[iDrop];
iDrop[ArrayAssociation[data_, structure_List], seqs___] /; Length[structure] >= Length[{seqs}] := ArrayAssociation[
	Drop[data, seqs],
	Join[
		MapThread[
			Function[{s, seq},
				If[s === None, None, Drop[s, seq]]
			],
			{Take[structure, UpTo[Length[{seqs}]]], {seqs}}
		],
		Drop[structure, UpTo[Length[{seqs}]]]
	]
];


(* join *)
clear[iJoin];
iJoin[ArrayAssociation[data1_List, {None, rest___}], ArrayAssociation[data2_List, {None, rest___}]] := ArrayAssociation[Join[data1, data2], {None, rest}];
iJoin[a1 : ArrayAssociation[data1_List, {keys1_List, rest___}], a2 : ArrayAssociation[data2_List, {keys2_List, rest___}]] := With[
	{a1c = a1[[Complement[keys1, keys2]]]},
	ArrayAssociation[Join[Hold[a1c][[1, 1]], data2], {Join[Hold[a1c][[1, 2, 1]], keys2], rest}]
];
iJoin[l1_, l2_] := Join[Normal[l1, ArrayAssociation], Normal[l2, ArrayAssociation]];


(* join across *)
clear[iJoinAcross];
iJoinAcross[a1 : ArrayAssociation[_, {None, _List, ___}] | _List, a2 : ArrayAssociation[_, {None, _List, ___}] | _List, keys_, rest___] := JoinAcross[
	Normal[a1, ArrayAssociation],
	Normal[a2, ArrayAssociation],
	keys,
	rest
];


(* key drop *)
clear[iKeyDrop];
iKeyDrop[a_, {}] := a;
iKeyDrop[a : ArrayAssociation[data_List, {l : Repeated[None, {0, 1}], keys_List, srest___}], dropkeys_] := With[
	{droppos = DeleteMissing[keyPosition[keys, #] & /@ Flatten[{dropkeys}]]},
	If[droppos === {},
		a,
		a[[##]] & @@ {
			If[{l} === {}, Nothing, All],
			Complement[Range[Length[keys]], droppos]
		}
	]
];


(* key take *)
clear[iKeyTake];
iKeyTake[ArrayAssociation[_, {_List, rest___}], {}] := ArrayAssociation[{}, {{}, rest}];
iKeyTake[ArrayAssociation[data_List, {None, _List, rest___}], {}] := ArrayAssociation[ConstantArray[{}, Length[data]], {None, {}, rest}];
iKeyTake[a : ArrayAssociation[data_List, {l : Repeated[None, {0, 1}], keys_List, srest___}], takekeys_] := With[
	{takepos = DeleteMissing[keyPosition[keys, #] & /@ Flatten[{takekeys}]]},
	a[[##]] & @@ {
		If[{l} === {}, Nothing, All],
		takepos
	}
];


(* lookup *)
clear[iLookup];
SetAttributes[iLookup, HoldAllComplete];
iLookup[a : ArrayAssociation[_List, {_List}], lookupkeys_, default : Repeated[_, {0, 1}]] := Lookup[iNormal[a], lookupkeys, default];
iLookup[ArrayAssociation[{}, {None, _List}], _, Repeated[_, {0, 1}]] := {};
iLookup[ArrayAssociation[data_List, {None, keys_List}], Except[_List, lookupkey_], default : Repeated[_, {0, 1}]] := keyPosition[keys, lookupkey] // Replace[{
	pos_Integer :> data[[All, pos]],
	m_Missing :> ConstantArray[First[{default}, m], Length[data]]
}];
iLookup[ArrayAssociation[data_List, {None, _List}], {}, Repeated[_, {0, 1}]] := ConstantArray[{}, Length[data]];
iLookup[a : ArrayAssociation[data_List, {None, keys_List}], lookupkeys_List, default : Repeated[_, {0, 1}]] := If[ContainsAll[keys, Replace[lookupkeys, Key[k_] :> k, {1}]],
	With[{x = a[[All, Replace[lookupkeys, Except[_Key, k_] :> Key[k], {1}]]]},
		Hold[x][[1, 1]]
	],
	Transpose[
		With[
			{assoc = AssociationThread[keys, Transpose[data]]},
			If[Hold[default] === Hold[],
				Replace[Lookup[assoc, lookupkeys], m_Missing :> ConstantArray[m, Length[data]], {1}],
				Lookup[assoc, lookupkeys, ConstantArray[default, Length[data]]]
			]
		]
	]
];


(* normal *)
clear[iNormal];
iNormal[ArrayAssociation[data_, {None ...}]] := data;
iNormal[ArrayAssociation[data_List, {None, keys_List}] /; MatchQ[Dimensions[data, 2], {_, Length[keys]}]] := Module[
	{arr = ConstantArray[<||>, Length[data]]},
	arr[[All, Key /@ keys]] = data;
	arr
];
iNormal[ArrayAssociation[data_List, {keys_List}] /; Length[data] === Length[keys]] := AssociationThread[keys, data];
iNormal[ArrayAssociation[data_List, {k1_, krest__}]] := iNormal[ArrayAssociation[iNormal[ArrayAssociation[#, {krest}]] & /@ data, {k1}]];


(* part *)
clear[iPart];
iPart[a : ArrayAssociation[_, structure_List], all : All ...] /; Length[structure] >= Length[{all}] := a;
iPart[
	ArrayAssociation[data_List, structure_List],
	pspec : (All | _Integer | _String | _Key | {(_Integer | _String | _Key) ...}) ..
] /; Length[structure] >= Length[{pspec}] := With[
	{sp = {Take[structure, Length[{pspec}]], {pspec}}},
	ArrayAssociation[
		Part[
			data,
			Sequence @@ MapThread[transformPartSpec, sp]
		],
		Join[
			MapThread[transformStructure, sp],
			Drop[structure, Length[{pspec}]]
		]
	]
];


(* pick *)
clear[iPick];
iPick[ArrayAssociation[data_List, {keys : None | _List, srest___}], sel_List, patt : Repeated[_, {0, 1}]] := ArrayAssociation[
	Pick[data, sel, patt],
	{
		If[keys === None,
			None,
			Pick[keys, sel, patt]
		],
		srest
	}
];


(* sort *)
clear[iSort];
iSort[a : ArrayAssociation[data_List, structure : {keys : None | _List, srest___}], p_ : Order] := If[{srest} === {} || First[{srest}] === None || p === Order,
	If[keys === None,
		ArrayAssociation[Sort[data, p], structure],
		With[
			{d = Sort[Thread[{data, keys}], p[First[#1], First[#2]] &]},
			ArrayAssociation[d[[All, 1]], {d[[All, 2]], srest}]
		]
	],
	Sort[Normal[a], p]
];


(* take *)
clear[iTake];
iTake[ArrayAssociation[data_, structure_List], seqs___] /; Length[structure] >= Length[{seqs}] := ArrayAssociation[
	Take[data, seqs],
	Join[
		MapThread[
			Function[{s, seq},
				If[s === None, None, Take[s, seq]]
			],
			{Take[structure, UpTo[Length[{seqs}]]], {seqs}}
		],
		Drop[structure, UpTo[Length[{seqs}]]]
	]
];


clear[transformPartSpec];
transformPartSpec[_, p : All | _Integer | {___Integer}] := p;
transformPartSpec[s_List, key_String | Key[key_]] := First[FirstPosition[s, key, fail[], {1}, Heads -> False]];
transformPartSpec[s_, p_List] := transformPartSpec[s, #] & /@ p;

clear[transformStructure];
transformStructure[_, _Integer | _String | _Key] := Nothing;
transformStructure[None, _] := None;
transformStructure[s_List, All] := s;
transformStructure[s_List, p_List] := s[[transformPartSpec[s, p]]];

clear[keyPosition];
keyPosition[s_List, Key[key_] | key_] := First[FirstPosition[s, key, {Missing["KeyAbsent", key]}, {1}, Heads -> False]];


End[];
EndPackage[];

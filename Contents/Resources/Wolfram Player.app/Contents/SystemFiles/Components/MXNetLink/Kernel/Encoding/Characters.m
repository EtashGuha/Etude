Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["CharacterEncodingData"]

DeclarePostloadCode @ DefineCustomBoxes[CharacterEncodingData,
	CharacterEncodingData[1, _, _, ncodes_, spec_] :> ToBoxes[Short[angle @ (Developer`ToList @ DeleteCases[spec, IgnoreCase -> _])]],
	"UseUpValues" -> False
];

angle[{e_}] := e;
angle[e_] := AngleBracket @@ e;

CharacterEncodingData /: Normal[CharacterEncodingData[1, _, _, _, spec_]] := spec;

(******************************************************************************)

PackageExport["CharacterEncodingDataSize"]

CharacterEncodingDataSize[CharacterEncodingData[1, _, _, ncodes_, spec_]] := ncodes;

(******************************************************************************)

PackageExport["CharacterEncodingEOSCode"]

CharacterEncodingEOSCode[CharacterEncodingData[1, forward_, _, _, _]] := Replace[
	Switch[
		NumericArrayType[forward],
			(* Dense encoding *)
			"UnsignedInteger8", Normal[forward][[3]],
			(* Sparse encoding *)
			_, Cases[Normal[forward], {0, code_} :> code - 1]
	],
	0 -> None
]

(******************************************************************************)

PackageScope["$NamedEncodingSpecs"]

parseSimple[s_String] := ToCharacterCode[s];
parseSimple[list_List] := Flatten @ Map[parseSimple, list];
parseSimple[Span[a_, b_]] := Range[toInt@a, toInt@b];

toInt[s_String] := First @ ToCharacterCode[s];
toInt[i_Integer] := i;

$CharacterGroups = parseSimple /@ Association[
	Automatic :> $DefaultCharacterEncodingDataSpec,
	AutomaticLowercase :> $DefaultCharacterEncodingDataSpecLowercase,
	LetterCharacter -> {"a";;"z", "A";;"Z"},
	LowercaseLetterCharacter -> {"a";;"z"},
	WordCharacter -> {"a";;"z", "A";;"Z", "0";;"9"},
	LowercaseWordCharacter -> {"a";;"z", "0";;"9"},
	DigitCharacter -> {"0";;"9"},
	PunctuationCharacter -> {33;;64, 91;;96, 123;;126},
	Whitespace -> " \t\n",
	WhitespaceCharacter -> " \t\n"
];

$icRules = {
	LetterCharacter -> LowercaseLetterCharacter,
	WordCharacter -> LowercaseWordCharacter,
	Automatic -> AutomaticLowercase
};

(******************************************************************************)

PackageScope["$DefaultCharacterEncodingDataSpec"]

$DefaultCharacterEncodingDataSpec = Join[{"\t", "\n"}, CharacterRange[32, 126]];
$DefaultCharacterEncodingDataSpecLowercase = ToLowerCase[$DefaultCharacterEncodingDataSpec];

(******************************************************************************)

PackageExport["ToCharacterEncodingData"]

ToCharacterEncodingData[Automatic, name_, spec_] := 
	ToCharacterEncodingData[$DefaultCharacterEncodingDataSpec, name, spec];

ToCharacterEncodingData[spec_] := ToCharacterEncodingData[spec, None, False];

ToCharacterEncodingData[c_CharacterEncodingData, _, ic_] := c;

ToCharacterEncodingData[ispec_, name_, ic_] := Scope[
	name = If[name =!= None, name, ispec];
	$table = toCharacterEncodingTable[ispec, ic];
	isDense = Max[Keys[$table], Values[$table]] < 128;
	maxCode = Max[$table];
	If[isDense,
		fill = Lookup[$table, -2, 128];
		data = Lookup[$table, Range[-2, 127], fill];
		If[$ic, data[[68;;93]] = data[[100;;125]]];
		forward = RawArray["Byte", data];
		index = Last /@ PositionIndex[data]-3; index[fill] = 0;
		reverse = RawArray["Byte", Ramp @ Lookup[index, Range[maxCode], 0]];
	,
		fill = $table[-2]; If[MissingQ[fill], fill = 1, fill++];
		If[$ic, 
			icrules = DeleteMissing @ Thread[Range[65, 90] -> Lookup[$table, Range[97, 122]]];
			AssociateTo[$table, icrules];
		];
		{begin, end} = Lookup[$table, {-1, 0}];
		KeyDropFrom[$table, {-2, -1, 0}];
		index = Reverse[Normal[$table], 2];
		reverse = RawArray["UnsignedInteger32", Ramp @ Lookup[index, Range[maxCode], 0]];
		data = Join[{{fill, begin}, {0, end}}, AssociationPairs[$table]];
		forward = RawArray["UnsignedInteger32", MapAt[incPositive, data, {All, 2}]];
	];
	System`Private`ConstructNoEntry[CharacterEncodingData, 1, forward, reverse, maxCode, name]
];

toCharacterEncodingTable[ispec_, ic_] := Module[{spec, futureCodes},
	$ic = TrueQ[ic]; $n = 1;
	spec = parseSpec[ispec];
	$table = Association[-1 -> 0, 0 -> 0];
	futureCodes = Reverse @ FoldList[Union, {}, Reverse @ Rest @ Keys[spec]];
	ScanThread[applySpec, {spec, futureCodes}];
	(* We return $table just for clarity here, because applySpec actually 
	   changes $table directly, so there's no real need to return it 
	   (and to assign it in ToCharacterEncodingData). *)
	$table
];

incPositive[z_] := If[Positive[z], z+1, z];

applySpec[codes_ -> rhs_, future_] := 
	applySpec[DeleteCases[codes, Alternatives @@ future] -> rhs];

applySpec[codes_ -> All] := Do[
	$table[code] = $n++,
	{code, codes}
];

applySpec[codes_ -> Automatic] := applySpec[codes -> $n++];

applySpec[codes_ -> None] := applySpec[codes -> 0];

applySpec[codes_ -> n_Integer] := (
	Do[
		$table[code] = n,
		{code, codes}
	];
	If[n + 1 > $n, $n = n + 1]
)

Clear[parseGroup, parseClause, parseSpec];

(* specs are either a single group or a list of clauses *)

parseSpec[list_List] := 
	If[FreeQ[list, _Rule],
		List[parseGroup[list] -> All],
		Flatten @ Map[parseClause, list]
	];

parseSpec[e_] := Flatten @ List @ parseClause[e];

(* clauses are rules *)

DeclarePostloadCode[
General::invstrenccode = "Explicit code value `` should be positive."
]

parseClause[group_ -> n_Integer] := 
	If[Positive[n], parseGroup[group] -> n,
		ThrowFailure["invstrenccode", n]];

parseClause[group_ -> rhs:(Automatic | None | All)] :=
	parseGroup[group] -> rhs;

parseClause[other_] :=
	parseGroup[other] -> All;

(* groups are the LHS of clauses *)

oneLetterQ[s_] := StringLength[s] === 1;
parseGroup[Span[a_String ? oneLetterQ, b_]] := parseGroup[Span[ToCharacterCode[a], b]];
parseGroup[Span[a_, b_String ? oneLetterQ]] := parseGroup[Span[a, ToCharacterCode[b]]];
parseGroup[Span[a_Integer, b_Integer]] /; a > 1 && b > 1 && a <= b && (b-a) < 65535 := Range[a, b];

parseGroup[i:_Integer | {__Integer}] /; Min[i] > 0 := parseGroup @ ToCharacterCode[i];

parseGroup[alts_Alternatives] := parseGroup[List @@ alts];

parseGroup[list_List] := DeleteDuplicates @ Flatten @ Map[parseGroup, list];

parseGroup[Verbatim[_]] := {-2};

parseGroup[StartOfString] := {-1};

parseGroup[EndOfString] := {0};

parseGroup[spec_String] := 
	ToCharacterCode[If[$ic, ToLowerCase[spec], spec]];

parseGroup[symbol_Symbol] := 
	Lookup[
		$CharacterGroups, 
		Replace[symbol, If[$ic, $icRules, {}]],
		ThrowFailure["invstrencgrpname", symbol, Select[Keys[$CharacterGroups], Context[#] === "System`"&]]
	];

DeclarePostloadCode[
General::invstrencgrpname = "Character group specification `` should be one of ``.";
General::invstrencspec = "Character specification `` isn't a symbol, string, integer, or list of these.";
]

parseGroup[spec_] := ThrowFailure["invstrencspec", spec];

(******************************************************************************)

PackageExport["CharacterEncodingAlphabet"]

CharacterEncodingAlphabet[CharacterEncodingData[1, forward_, reverse_, _, _]] := Scope[
	FromCharacterCode[Normal[reverse]]
];

CharacterEncodingAlphabet[_] := versionpanic[];

(******************************************************************************)

CharacterEncodingData[1, forward_, _, _, _][input_String] := 
	Internal`UnsafeQuietCheck @ mxlStringCharacterEncode[input, forward];

_CharacterEncodingData[_] := versionpanic[];

(******************************************************************************)

PackageExport["ToCharacterEncodingFunction"]

mxlDeclare[mxlStringCharacterEncode, {"String", "NumericArray"}, "IntegerVector"]

ToCharacterEncodingFunction[CharacterEncodingData[1, forward_, _, _, _], failf_:Panic] := Function[
	input,
	Replace[
		Internal`UnsafeQuietCheck @ mxlStringCharacterEncode[input, forward],
		Except[_List] :> failf[]
	],
	{Listable}
];

ToCharacterEncodingFunction[___] := versionpanic[];

(******************************************************************************)

PackageScope["StringCharacterDecodeSequence"]

(* This is a duplicate of neural networks implementation CodersUtil.m *)
ListMaxIndex = Compile[{{values, _Real, 2}},
	Map[If[Length[#] === 0, 0, First @ Ordering[#, -1]]&, values]
];

StringCharacterDecodeSequence[probs_, codes_] := Scope[
	c = codes[[ListMaxIndex[probs]]];
	If[AllTrue[c, IntegerQ], FromCharacterCode[c], "\t"]
]

StringCharacterDecodeSequence[probs_, codes_NumericArray] := StringCharacterDecodeSequence[probs, Normal[codes]]

(******************************************************************************)

PackageExport["ToCharacterDecodingFunction"]

ToCharacterDecodingFunction[CharacterEncodingData[1, _, backward_, _, _]] := 
	Function[input, StringCharacterDecodeSequence[input, backward]];

ToCharacterDecodingFunction[_] := versionpanic[];

DeclarePostloadCode[
General::futcencd = "Character encoding is from a future version of the Wolfram Language and is not compatible with this version."
]

versionpanic[] := ThrowFailure["futcencd"];
Package["NeuralNetworks`"]


PackageScope["$TrainableIconBoxes"]
PackageScope["$UntrainableIconBoxes"]

$TrainableIconBoxes := $TrainableIconBoxes = Uncompress @ "
1:eJxTTMoPSmNiYGAo5gYS7kWJBRmZycVO+RUQQSEkQef83IKc1AqQHCtIjgVI+GQWl0BUwniZQJoBTK
CKFzGAwYP9mYw45exxy/3AJgcWwmIXGg+kqpgdSPgXJCZnllQWGYPBZXuEjE9mXirIY2x4PAa2jAmHI5
hxiLNgirPiMIcVhzmscHOYMb0lCCQck4rzc0pLUgPyM/NKgjOrUsFaIKo5gESQu5Nzfk5+UZF86+vAHX
JX7TEZEMNAisGGYMQxxDNwl4Odiea3oNKc1GJOIMMzNzE9FewMVSAPAL2Gdfk=";

$UntrainableIconBoxes := $UntrainableIconBoxes = Uncompress @ "
1:eJzlmD9ME3EUx0sBUYwyiIkaw5+Ef0YEFNHEBFtLbC3UEMCRpZSrnrRcvbZGcGFhcYAY46AxLsTBhc
UYjZGEYAyLDhoNmhiTTg444ICDLv5+73f34vV371pbhRBvaH93n7t77/fe997v3dWPaINRt8vlSu5kPw
E9nLikRpI+7Vp0Bz9Yxn5CajIVLeF7/NAFxsfGlWRSr4FtxhPdZjmzlO9tZz+DAV+PFtN0Pf7l58Nvbz
MeeWBzcu306sDTmvceeWDnw+2xqqmF5RmDVbCf/kQ4oqYmdBdsawbhJno1ddyv6XExXdPfoXI2OK+mFN
24pMErvNptXKKM9qT1qwoPSYkcEuu0zT2V30jldmCUdaqb3jN8OOkVg6ljXlveascH0zEFsgj+9sS0pD
JqtSycKS88Xf9NBry9OTLgLyYDVvMYHn84osjhsclQJ2yrHnngkAW3GVK/GouRIS2l97JDagNL8oNWk2
UOybhePR/InDrhnIxFr1eHGvGyO5u/WG8Znh2xTZb1+q/s8unVg96icyNuxB4aaVBAbmyM/evcbIgR67
ycBIAJqlyZS3T3ns5O4HeW36XHXbkTzC+vXDmQzUE3D34XSF7ewMAryQ0O1+UhN5H5WntOyr0AOU5dfp
66te+TRx4UXSrKZNOYb0g131NLZTGUmMeLhVZ/djkk7lxmsnp+mRX116/41pYd+CtMG3NrQZJDHfExDo
OlDpLDjRJS4pGLZuY4aR8K3mRXNkf/CQ63vd9HcphWKETaR074j5yYP9on4pcj/pstbKd6tBnCtumo8y
vCf82DP4DgEFXb83ooQbPvJNHyZXPgQ5DkUPv3Mj67xFTZL4kaOSFq5IPc0k1J1Gj/7h227adFTXBYcp
70kRwOT4RI+8gJ/5ET80f7RPzk+FcVkS3k/D9QL3EoPNN9JIdntC5ktmKdJCeigRzSui5FE+3DiVPtpP
8EhwakkuaQBC1I2kdO+I+cmD/aJ+KXK/458rfZJXjr9hZiST1sNnWSMCAfmWaSgxy6G40y8EYSDnJCOM
hBQVVSGUD7MAjb9w7cf4IPw33bSC6ehw7SPnKnB5dzYv5on4hfjvhvtrC3Um+xMR4UDvPvLVAU0BA/k0
Qr+uRmksMzdaPBaHh9kqiRE6JGLjpjSdRoX3TWUm+A/hNc9OFtJIfe4GMHaR+5U2/BOTF/tE/ET47/nn
yyBYtUBVFiWkku3iqOmK8nkrdi7WszPhOdJWbTbtx/WVp7o3xr6jDXVpoT0UQOhtxHbdXwg/kPn/Ee0S
WO4GB14RDJhVst5uufpAZoWdJMjVBr7xFqbTI7Wcl/8fraaJTgRZoT8UEO/n2W4itE2ELm30E/8E2Wf7
IOxsMXlSF1UrFeq7bwPX7CmXRKi4dTauQXM1RAYg==";


PackageScope["SummaryForm"]

DeclareMethod[SummaryForm, SimpleSummaryForm, ContainerSummaryForm, OperatorSummaryForm];

SimpleSummaryForm[assoc_] := NSymbol[assoc];

ContainerSummaryForm[assoc_] := 
	Row[{NSymbol[assoc], " ", Row[{"(", Length @ assoc["Nodes"], " nodes)"}, BaseStyle -> Gray]}];

OperatorSummaryForm[assoc_] := Scope[
	subnets = GetSubNets[assoc];
	inner = If[Length[subnets] === 1, 
		SummaryForm[First[subnets]],
		Skeleton[Length[subnets]]
	];
	Compose[HoldForm, SimpleSummaryForm[assoc], inner]
]

PackageScope["decapitalizeStrings"]

$ulRE = RegularExpression["\"[A-Z][a-z]"];
decapitalizeStrings[boxes_] := ReplaceAll[boxes, {
	TemplateBox[tb_, rest___] :> RuleCondition @ TemplateBox[decapitalizeStrings[tb], rest],
	s_String /; StringStartsQ[s, $ulRE] :> RuleCondition @ If[s === "\"ReLU\"", s, decapitalize[s]]
}];

decapitalize[str_String] := ToLowerCase[StringTake[str, 2]] <> StringDrop[str, 2];


PackageScope["$DummyVar"]

$DummyVar = Style["x", Italic];


PackageScope["deCamelCase"]

$camelRE = RegularExpression["[a-z][A-Z]"];
deCamelCase[str_] := 
	StringReplace[str, cam:$camelRE  :> StringInsert[
		If[$LowercaseTraditionalForm, ToLowerCase[cam], cam], 
		" ", 2]
	];

PackageScope["SpecializedSummary"]

(* currently used by ThreadingLayer and ElementwiseLayer to return a 
summary that should be used in a NetGraph legend if it is the only
layer of the given type. If it isn't, the second arg is used, and
the first arg is put as label above the vertex instead *)


PackageScope["$uninitializedColor"]

$uninitializedColor = RGBColor[0.66, 0, 0];
$uninitializedColor2 = RGBColor[0.58, 0.25, 0.25];


PackageScope["$HideArraysInSummaryBox"]
PackageScope["$ShowAllInSummaryBox"]

If[!ValueQ[$ShowAllInSummaryBox], $ShowAllInSummaryBox := False];

PackageScope["MakeLayerBoxes"]

SetHoldFirst[MakeLayerBoxes];

MakeLayerBoxes[layer:head_Symbol[assoc_Association, meta_]] := Scope @ WithAmbientSharedArrays[assoc,
	icon = Which[
		!NetHasArraysQ[assoc],
			$UntrainableIconBoxes,
		InitializedNetQ[layer], 
			$TrainableIconBoxes, 
		True,
			fmtUninitializedIcon @ $TrainableIconBoxes
	];
	{short, long} = LengthVarScope[assoc, infoItems[assoc]];
	If[TrueQ[$ShowAllInSummaryBox], short = long];
	OptimizedArrangeSummaryBox[head, icon, short, long, True]
];

MakeLayerBoxes[_] := Fail;


PackageScope["fmtUninitializedIcon"]
PackageScope["toGrayIcon"]

toGrayIcon[e_, l1_:0.6, l2_:0.9] := 
	Append[
		ReplaceAll[e, RGBColor[r_,g_,b_]:> RuleCondition @ GrayLevel[
			If[Max[{r,g,b}]>0.8,l2,l1]]
		],
		BaseStyle -> GrayLevel[l1]
	];

fmtUninitializedIcon[boxes_] := OverlayBox[{
	toGrayIcon @ boxes, 
	StyleBox["uninitialized", $uninitializedColor, FontSize -> 8, FontFamily -> "Roboto", Background -> GrayLevel[1,0.65]]},
	Alignment->{Center,Scaled[0.0001]}
]


PackageScope["infoItems"]

$ioOnlyLayers = "Linear" | "Reshape";

infoItems[assoc_] := Scope[
	type = assoc["Type"];
	$addDivider = False;
	params = fmtSection[assoc["Parameters"], "Parameters", False];
	arrays = If[TrueQ[$HideArraysInSummaryBox], {}, fmtSection[assoc["Arrays"], "Arrays", True]];
	ports = fmtSection[preserveKeys @ Join[$in /@ assoc["Inputs"], $out /@ assoc["Outputs"]], "Ports", False];
	states = fmtSection[preserveKeys @ assoc["States"], "States", False];
	long = ToList[params, arrays, ports, states];
	If[DigitStringKeysQ @ assoc["Inputs"] && Length[params] == 0,
		short = Take[ports, {2,-2}],
		short = If[Length[params] == 0 || MatchQ[type, $ioOnlyLayers], take2 @ ports, take2 @ params];
	];
	If[Length[short] === 1, 
		If[$LayerData[type, "IsLoss"], 
			short = Join[short, Take[ports, {2, 1 + Length[assoc["Inputs"]]}]],
			(* ^ loss ports will always show both inputs *)
			AppendTo[short, Last[ports]]
			(* ^ if only one param, put output port in short as well *)
		];
	];
	{short, long}
];

take2[e_] := Take[e, {2, UpTo[3]}];

preserveKeys[a_Associaion] := KeyMap[AvoidDecamel, e];
preserveKeys[e_] := e;

PackageScope["fmtSection"]

Clear[fmtSection];
$addDivider = True;

fmtSection[_Missing | <||>, _, _] := {};

fmtSection[assoc_, title_, arrayQ_] := Scope[
	$colorUninit = arrayQ;
	list = fmtEntries @ assoc;
	If[list === {}, Return[{}]];
	If[$addDivider, 
		frameStyle = Sequence[Frame -> {{False, False}, {False, True}}, FrameStyle -> LightGray],
		frameStyle = Sequence[];
	];
	PrependTo[list, {
		ItemBox[StyleBox[title, Bold], frameStyle],
		ItemBox[If[$addDivider, spacerBoxes[{1,11}], ""], frameStyle]
	}];
	$addDivider ^= True;
	list
];

spacerBoxes[sz_] := StyleBox[GraphicsBox[{}, ImageSize -> sz, BaselinePosition -> Scaled[0.] -> Baseline], "CacheGraphics" -> False];

PackageScope["fmtEntries"]
PackageScope["fmtEntry"]
PackageScope["$divideAbove"]

fmtEntries[assoc_] := KeyValueMap[fmtEntry, assoc];

Clear[fmtEntry];

fmtEntry[$Multiport, v_] := fmtEntry["1,2,..,n", v];

fmtEntry[k_String, v_] /; StringStartsQ[k, "$"] := Nothing;

PackageScope["$InEntryFormatting"]

$InEntryFormatting = False;

fmtEntry[k_, NetSharedArray[name_]] :=
	fmtEntry[fmtSharedAs[Decamel[k], name], $AmbientSharedArrays[name]];

fmtEntry[k_, $divideAbove[v_]] :=
	Map[ItemBox[#, Frame -> {{False, False}, {False, LightGray}}]&, fmtEntry[k, v]];

fmtEntry[k_, v_] := Scope[
	v2 = If[StringQ[k] && !StringFreeQ[k, "Dimensions"] && VectorQ[v, IntegerQ], fmtDimsList, fmtItem][v];
	k = Decamel[k];
	k = ToBoxes[k];
	If[$colorUninit && UninitializedArrayQ[v], k = StyleBox[k, $uninitializedColor2]];
	$InEntryFormatting = True;
	List[
		StyleBox[TemplateBox[{k, "\":\""}, "RowDefault"], "SummaryItemAnnotation"], 
		StyleBox[ToBoxes @ v2, "SummaryItem"]
	]
];

fmtSharedAs[k_, name_] := Row[{k, " (shared as ", name, ")"}];

PackageScope["tensorName"]

tensorName[n_, True] := tensorName[n] <> "s";
tensorName[n_, False] := n;

tensorName[SizeT|NaturalT] := "array";
tensorName[0] = "scalar";
tensorName[1] = "vector";
tensorName[2] = "matrix";
tensorName[2, True] := "matrices";
tensorName[n_Integer] := "rank-" <> IntegerString[n] <> " array";

genTensorName[n_Integer /; n > 2] := 
	Row[{"array", " ", fmtNote["rank", n]}];

genTensorName[n_] := tensorName[n];


PackageScope["fmtItem"]

courierBold[e_] := Style[e, FontFamily -> "Courier", FontWeight -> Bold, FontSize -> Larger];

Clear[fmtItem];
fmtItem[<|"Type" -> t_String, ___|>] := With[{sym = $TypeToSymbol[t]}, HoldForm[sym["\[Ellipsis]"]]];

fmtPHList[n_, t_] := Tooltip[
	If[TrueQ[1 <= n <= 3], Table[$PlaceholderIcon, n], "{\[Ellipsis]}"], 
	IndefiniteTypeForm[ListT[n, t]]
];

$plural = False;
fmtPluralItem[e_] := Block[{$plural = True}, fmtItem[e]];

toPlural[s_] := If[$plural, pluralStr[s], s];

fmtItem[SizeListT[n_]] := fmtPHList[n, SizeT];
fmtItem[t:{__SizeT}] := fmtPHList[Length[t], SizeT];
fmtItem[$in[e_]] := fmtItem[e];
fmtItem[$out[e_]] := fmtItem[e]; (* couldn't find a nice icon *)
fmtItem[SequenceT[n_LengthVar, t_]] := Row[{"vector of ", Style[FormatLengthVar[n], Italic], " ", fmtPluralItem[t]}];
fmtItem[SequenceT[2, t_NetEncoder]] := Row[{"pair of ", fmtPluralItem[t]}];
fmtItem[SequenceT[n_Integer, p:CoderP]] := Row[{"vector of ", n, " ", fmtPluralItem[p]}];
fmtItem[e_List] := Which[
	StringVectorQ[e] || Length[e] > 4, Short[e],
	True, fmtItem /@ e
];
(*
fmtItem[enc_NetEncoder] := Row[{toPlural[CoderKind[enc]], " encoded as ", fmtItem[CoderType[enc]]}];
fmtItem[dec_NetDecoder] := Row[{fmtItem[CoderType[dec]], " decoded as ", CoderKind[dec]}];
*)
fmtItem[enc_NetEncoder] := toPlural[CoderKind[enc]];
fmtItem[dec_NetDecoder] := toPlural[CoderKind[dec]];

fmtItem[RepeatedInteger[n_]] := Row[{n, ".."}];
fmtItem[_LengthVar] := Style["variable", Gray];
fmtItem[RealT] := toPlural @ "real";
fmtItem[TensorT[ListT[n_, _]]] := toPlural @ genTensorName[n];
fmtItem[e_EnumT] := Tooltip[$PlaceholderIcon, Alternatives @@ First[e]];
fmtItem[EitherT[e_]] := Alternatives @@ Map[fmtItem, e];
fmtItem[TensorT[{}, RealT]] := toPlural @ "real";
fmtItem[TensorT[{}, AtomT]] := toPlural @ "scalar";
fmtItem[TensorT[{}, i_IndexIntegerT]] := fmtItem @ i;
fmtItem[TensorT[list_List]] := fmtTensorDims[list];
fmtItem[TensorT[dims_List, i_IndexIntegerT]] := Row[{fmtTensorDims[dims], " of ", fmtPluralItem[i]}];
fmtItem[t_TensorT] := IndefiniteTypeForm[t, $plural];
fmtItem[r_NumericArray] := fmtTensorDims[Dimensions[r]];
fmtItem[NetSharedArray[a_]] := Row[{Framed[a, BaseStyle -> Gray, FrameMargins -> 2, ContentPadding -> False, FrameStyle -> GrayLevel[0.2]], " ", fmtNote["shared array"]}];
fmtItem[r_DummyArray] := fmtTensorDims[First[r]];
fmtItem[NaturalT] := Tooltip[$PlaceholderIcon, "non-negative integer"];
fmtItem[SizeT | PosIntegerT] := Tooltip[$PlaceholderIcon, "positive integer"];
fmtItem[IndexIntegerT[max_Integer]] := Row[{If[$plural, "indices", "index"], " ", fmtNote["range", Row[{1, "..", max}]]}];
fmtItem[IndexIntegerT[Infinity]] := (*Row[{*) If[$plural, "indices", "index"] (*, " ", fmtNote["range", Row[{1, "..", Infinity}]]}]*);
fmtItem[IndexIntegerT[All]] := If[$plural, "integers", "integer"];
fmtItem[IndexIntegerT[_]] := If[$plural, "indices", "index"];
fmtItem[SymbolicRandomArray[NNConstantDist[val_], dims_]] := Row[{fmtTensorDims[dims], " = ", val}];
fmtItem[SymbolicRandomArray[dist_, dims_]] := Row[{fmtTensorDims[dims], " \[Distributed] ", fmtDist @ dist}];
fmtItem[t:True|False] := t;
fmtItem[IntegerT] := If[$plural, "integers", "integer"];
fmtItem[Nullable[t_]] := Row[{"optional", " ", fmtItem[t]}];
fmtItem[HoldPattern @ a_Audio] := "Audio[<" <> ToString[SetPrecision[Audio`Utilities`Duration[a], 3]] <> "s>]"
fmtItem[File[f_String]] := "File[\"\[Ellipsis]"<>StringTake[f,-15] <> "\"]"

fmtItem[f_Function] := fmtFunction[f]

fmtItem[ValidatedParameter[e_]] := fmtValidatedParameter[e];

fmtDist[HoldPattern @ NormalDistribution[m_, sd_]] := TraditionalForm["\[ScriptCapitalN]"[m, sd]];
fmtDist[HoldPattern @ UniformDistribution[{min_, max_}]] := TraditionalForm["\[ScriptCapitalU]"[min, max]];
fmtDist[e_] := e;

fmtValidatedParameter[assoc_Association] := Grid[
	List[Row[RawBoxes /@ #, "  "]]& /@ KeyValueMap[fmtEntry, assoc],
	Alignment -> {Left, Automatic},
	BaselinePosition -> 1
];

fmtValidatedParameter[f_Function] := fmtFunction[f];

fmtFunction[f_] := Style[
	If[ByteCount[f] > 1000, HoldForm[Function]["\[Ellipsis]"], PrettyForm[f]], 
	FontFamily -> "Source Code Pro"
];

fmtValidatedParameter[sf_ScalarFunctionObject] := TraditionalForm[
	Apply[
		ScalarFunctionToPureFunction[sf],
		Take[{"x","y","z","u","v","w","q","a","b","c","d","e"}, Length[First[sf]]]
	] /. r_Real /; IntegerQ[Rationalize[r]] :> RuleCondition @ Round[r] 
];

fmtValidatedParameter[s_Symbol] := SymbolName[s];

fmtValidatedParameter[e_] := Pane[
	RawBoxes @ ToBoxes @ Short[e, 3],
	{250}, BaseStyle -> {IndentMaxFraction -> 0.1, ShowStringCharacters -> False}, ContentPadding -> False
];

fmtItem[ListT[n_Integer, t_]] /; n < 4 := CTable[fmtItem[t], n];

fmtItem[l:ListT[_, t_]] := Scope[
	inner = fmtItem[t];
	If[Head[inner] =!= Tooltip, (* <- avoid saying list of <placeholder>s *)
		Row[{"list of ", inner, "s"}], (* TODO: do this properly *)
		fmtPHList @@ l
	]
];

fmtItem[e_ /; !ValidTypeQ[e]] := e;

fmtItem[HoldPattern @ img_Image] := Thumbnail[img, If[$VersionNumber >= 11, UpTo[32], 32], Padding -> None];


PackageScope["$PlaceholderIcon"]

$PlaceholderIcon = "\[DottedSquare]";
fmtItem[e_] := If[ConcreteParameterQ[e], e, $PlaceholderIcon];



fmtTensorDims[e_List] := Row[{
	toPlural @ If[Length[e] > 2, "array", tensorName[Length[e]]], " ",
	fmtNote["size", Row[fmtDim /@ e, "\[Times]"]]
}];
fmtTensorDims[___] := "array";

fmtNote[prop_, value_] := 
	Style[Row[{"(", "\[VeryThinSpace]", prop, ":", " ", value,  "\[VeryThinSpace]", ")"}], Gray];

fmtNote[text_] := Style[Row[{"(", "\[VeryThinSpace]", text,  "\[VeryThinSpace]", ")"}], Gray];


PackageScope["fmtDim"]

fmtDim[n_Integer] := IntegerString[n];
fmtDim[lv_LengthVar] := Style[FormatLengthVar[lv], Italic];
fmtDim[_] := $PlaceholderIcon;


PackageScope["fmtDims"]

Clear[fmtDims];
fmtDims[_] := "";
fmtDims[TensorT[_ListT]] := "";(*tensorName[n];*)
fmtDims[r_NumericArray] := fmtDims[Dimensions[r]];
fmtDims[cod:CoderP] := fmtDims[CoderType[cod]];
fmtDims[TensorT[{}, t_]] := fmtDims[t];
fmtDims[RealT] := "\[DoubleStruckCapitalR]";
fmtDims[IndexIntegerT[n_Integer]] := Subscript["\[DoubleStruckCapitalN]", n];
fmtDims[TensorT[list_List, _]] := fmtDimsList[list];

PackageScope["fmtDimsList"]

fmtDimsList[{}] := Row[{"{}", Style["  (scalar)", Gray]}];
fmtDimsList[list_] := Row[fmtDim /@ list, "\[Cross]"];


PackageScope["typeInfo"]

typeInfo[key_ -> type_] := 
	typeInfo[key -> <|"Form" -> type|>];

typeInfo[key_ -> assoc_Association] := 
	infoGrid["Port", "Port", fmtKey @ key, fmtEntries @ assoc];

typeInfo[key_ -> cod:CoderP] := 
	infoGrid["Port", "Port", fmtKey @ key,
		fmtEntries @ Prepend[
			CoderData[cod],
			{"Form" -> CoderType[cod], "Type" -> CoderName[cod]}
		]
	];

PackageScope["itemInfo"]

itemInfo[key_ -> x_] := x;
itemInfo[key_ -> assoc_Association] := 
	infoGrid[
		NSymbol[assoc], 
		"Layer", fmtKey[key],
		showInfo[assoc]
	];

showInfo[assoc_Association] := Switch[
	assoc["Type"], 
	"Graph", List @ {netGraphPlot[assoc], "\[SpanFromLeft]"},
	"Chain", List @ {ToBoxes @ netChainGrid[assoc], "\[SpanFromLeft]"},
	_, Last @ infoItems[assoc]
];

infoGrid[header_, type_, key_, grid_] := 
	TagBox[
		GridBox[
			Prepend[{PaneBox[StyleBox[RowBox[{StyleBox[key, Gray], header}], Bold, 12], ImageSize -> {All, 13}], "\[SpanFromLeft]"}] @ grid,
			GridBoxAlignment -> {"Columns" -> {{Left}}, "Rows" -> {{Left}}},
			RowSpacings -> {2, 1},
			GridFrameMargins -> {{0, 0}, {5, 5}},
			GridBoxBackground -> {"Rows" -> {GrayLevel[0.92], {None}}}
		], 
		Deploy, DefaultBaseStyle -> "Deploy"
	];

fmtKey[key_] := If[IntStringQ[key], ToBoxes[key <> ": "], StyleBox[ToBoxes[key <> ": "], Gray]];


PackageScope["LengthVarScope"]

$nvar := $nvar = NameToLengthVar["n"];

SetHoldRest[LengthVarScope];
LengthVarScope[vars_, expr_] := Block[
	{$lengthVarNames = <||>, $inLVScope, uvs},
	(* only use subscript form if there is more than one legnthvar *)
	uvs = UniqueLengthVars[vars];
	$inLVScope = Length[Discard[uvs, NamedLengthVarQ]] > 1 || !FreeQ[uvs, $nvar];
	primeLengthVarNames[vars["Inputs"]];
	(* most of the time this is called on layer assocs, we want the input anonymous
	dims to get n1, n2, etc *)
	expr
];

primeLengthVarNames[_Missing] := Null;
primeLengthVarNames[expr_] := FormatLengthVar /@ UniqueLengthVars[expr];


PackageScope["FormatLengthVar"]

FormatLengthVar[lv_LengthVar ? NamedLengthVarQ] := LengthVarToName[lv];

FormatLengthVar[lv_LengthVar] := Which[
	!TrueQ[$inLVScope], 
		"n",
	$Notebooks && !$NNTestingMode,
		Subscript["n", Style[getLVName[lv], 7]],
	True,
		"n" <> getLVName[lv]
];

getLVName[lv_] := IntegerString @ CacheTo[$lengthVarNames, lv, Length[$lengthVarNames]+1];



PackageScope["OptimizedArrangeSummaryBox"]
PackageScope["$ExpandNetSummaryBoxByDefault"]

If[!FastValueQ[$ExpandNetSummaryBoxByDefault],
	$ExpandNetSummaryBoxByDefault = False;
];

(* cut straight to DynamicModuleBox becuase we're generating boxes directly for efficiency reasons but 
ArrangeSummaryBoxes ironically can't take boxes, it insists on boxifying the grid it is provided. *)

OptimizedArrangeSummaryBox[head_, icon_, closedGrid_List, openGrid_List, disjointGrids_:False] := Module[
	{boxes, interpretable, typedHead, leftBracket, rightBracket, closedButton, openButton},
	closedButton = DynamicBox[FEPrivate`FrontEndResource["FEBitmaps","SquarePlusIconMedium"]];
	openButton = DynamicBox[FEPrivate`FrontEndResource["FEBitmaps","SquareMinusIconMedium"]];
	typedHead = StyleBox[TagBox[SymbolName[head],"SummaryHead"], "NonInterpretableSummary"];
	leftBracket = StyleBox["[", "NonInterpretableSummary"];
	rightBracket = StyleBox["]", "NonInterpretableSummary"];
	innerBox = If[openGrid === {}, 
		makePanel @ makeGrid[icon, closedGrid, Nothing],
		Block[{UseTextFormattingQ = False}, With[
			{icon2 = If[icon === None, Nothing, icon], initExpand = $ExpandNetSummaryBoxByDefault}, 
			{grid1 = makeGrid[icon2, closedGrid, closedButton :> Set[Typeset`open, True]],
			 grid2 = makeGrid[icon2, If[disjointGrids, openGrid, Join[closedGrid, openGrid]], openButton :> Set[Typeset`open, False]]},
			{panel = makePanel[PaneSelectorBox[{False -> grid1, True -> grid2}, Dynamic[Typeset`open], ImageSize -> Automatic]]},
			DynamicModuleBox[{Typeset`open = initExpand}, panel]
		]
	]];
	boxes = RowBox[{typedHead, leftBracket, innerBox, rightBracket}];
	With[{copyOut = SymbolName[head] <> "[<>]"}, 
		TagBox[
			TemplateBox[{boxes}, "CopyTag", DisplayFunction->(#1&), InterpretationFunction -> (copyOut&) ], 
			False, Selectable -> False, Editable -> False, SelectWithContents->True
		]
	]
]

(* $iconSize = {Automatic, 3.2 * CurrentValue["FontCapHeight"] / AbsoluteCurrentValue[Magnification]} *)

makePanel[contents_] := PanelBox[
	contents,
	BaselinePosition -> Baseline,
	BaseStyle -> {ShowStringCharacters -> False, NumberMarks -> False, PrintPrecision -> 3, ShowSyntaxStyles -> False}
]

ClearAll[makeGrid];

makeGrid[icon_, grid_, button_ :> expr_] :=
	makeGrid[
		icon,
		grid,
		PaneBox[
			ButtonBox[button, ButtonFunction :> expr, Appearance -> None, Evaluator -> Automatic, Method -> "Preemptive"], 
			Alignment -> {Center,Center}, 
			ImageSize -> {Automatic, 24}
		]
	];

makeGrid[icon_, grid_, button_] := 
	GridBox[
		List @ List[
			button,
			icon,
			alignedGridBox[grid, {Automatic}, If[Length[grid] > 2, {2, 1}, Automatic]]
		],
		GridBoxAlignment -> {"Rows" -> {{Top}}},
		GridBoxItemSize -> {"Columns" -> {{Automatic}}, "Rows" -> {{Automatic}}},
		GridFrameMargins -> {{0, 0}, {0, 0}},
		BaselinePosition -> If[Length[grid] < 2, {1,1}, {1, 3}]
	];

PackageScope["alignedGridBox"]

alignedGridBox[grid_, hspacings_:{Automatic}, baselinePos_:Automatic, cspace_:2] := 
	GridBox[
		grid,
		BaselinePosition -> baselinePos,
		GridBoxAlignment -> {"Columns" -> {{Left}}, "Rows" -> {{Automatic}}},
		GridBoxItemSize -> {"Columns" -> {{Automatic}}, "Rows" -> {{Automatic}}},
		GridBoxSpacings -> {"Columns" -> {{cspace}}, "Rows" -> {hspacings}}
	];

PackageScope["NodeForm"]

NodeForm[name_] := MsgForm[NetPath["Nodes", name]];


PackageScope["MsgForm"]

MsgForm[f_Failure] := TextString[f];
MsgForm[path_NetPath] := NetPathString[path];
MsgForm[net_ ? ValidNetQ] := MsgForm[NData[net]];
MsgForm[assoc:<|"Type" -> _, ___|>] := NetShallowString[assoc];
MsgForm[type_ ? ValidTypeQ] := TypeString[type];
MsgForm[s_String] := QuotedString[s];
MsgForm[{}] := "{}";
MsgForm[s_List ? StringVectorQ] /; Length[s] < 16 := QuotedStringRow[s, " and "];
MsgForm[ra_NumericArray] := StringForm["array (``)", DimsString[Dimensions[ra]]];
MsgForm[s_StringForm] := s;
MsgForm[e_] := If[Depth[e] > 5 || ByteCount[e] > 1000, Shallow[e], e] /. ra_NumericArray :> RuleCondition @ MsgForm[ra];


PackageScope["CoderForm"]

CoderForm[expr_] := ReplaceAll[
	expr,
	(c_NetEncoder|c_NetDecoder) ? System`Private`HoldNoEntryQ :> RuleCondition @ CoderFormString[c]
];


PackageScope["CoderFormString"]

SetHoldFirst[CoderFormString];

CoderFormString[(head:NetEncoder|NetDecoder)[kind_, data_, type_]] := Scope[
	posArgCount = If[head === NetEncoder, $EncoderData, $DecoderData][kind, "PosArgCount"];
	head = SymbolName[head]; 
	kind = QuotedString[kind];
	StringJoin @ If[posArgCount === 0, 
		{head, "[", kind, "]"},
		{head, "[{", kind, ", \[Ellipsis]}]"}
	]
];

(* This is more complex than we really need, but don't want to throw it away *)
(*
toCoderString[(head:NetEncoder|NetDecoder)[kind_, data_, type_]] := Scope[
	cdata = If[head === NetEncoder, $EncoderData, $DecoderData][kind];
	UnpackAssociation[cdata, posArgCount, parameterDefaults, parameterCoercions];
	head = SymbolName[head]; 
	args = Take[data, posArgCount];
	If[kind === "Image", AssociateTo[parameterDefaults, {"ColorSpace" -> "RGB", "ColorChannels" -> 3}]];
	PeekOption = data; 
	(* ^ needed to make IgnoreCase peeking work in case of Token coders *)
	defaultLen = LengthWhile[
		Reverse @ Normal @ args, 
		Apply[Lookup[parameterCoercions, #1, Identity][Lookup[parameterDefaults, #1]] === #2&]
	];
	argStrings = Map[argString, Values @ Drop[args, -defaultLen]];
	argStrings = Replace[argStrings, {Shortest[a___], $Failed, ___} :> {a}];
	kind = QuotedString[kind];
	StringJoin @ If[Length[argStrings] === 0, 
		{head, "[", kind, "]"},
		{head, "[{", StringRiffle[Prepend[kind] @ argStrings, ", "], "}]"}
	]
];*)


PackageScope["NetShallowString"]

NetShallowString[net_NetP] := NShallowString[net];

DeclareMethod[NShallowString, LayerShallowString, ContainerShallowString];

ContainerShallowString[net_] := 
	StringJoin[
		SymbolName @ NSymbol[net], "[", 
			SkeletonString @ Length @ net["Nodes"], 
			If[net["Type"] === "Graph", {",", SkeletonString @ Length @ net["Edges"]}, {}],
		"]"
	]

LayerShallowString[net_] := Scope[
	{min, max} = NProperty[net, {"MinArgCount", "MaxArgCount"}];
	If[min === 0 && max > 0, min = 1]; (* don't be stingy, e.g. CELoss *)
	args = Values @ Take[net["Parameters"], min];
	argStrings = Map[argString, args];
	argStrings = Replace[argStrings, {Shortest[a___], $Failed, ___} :> {a}];
	AppendTo[argStrings, "\[Ellipsis]"];
	SymbolName[NSymbol[net]] <> "[" <> Riffle[argStrings, ","] <> "]"
];

argString[ValidatedParameter[sym_Symbol]] := SymbolName[sym];
argString[_ValidatedParameter] := "\[Ellipsis]";
argString[net:<|"Type" -> _, ___|>] := NShallowString[net];
argString[e_String] := "\"" <> e <> "\"";
argString[e_Integer | e_Real] := TextString[e];
$simpleAtom = _Integer | _String | _Symbol;
argString[e:{RepeatedNull[$simpleAtom, 3]}] := StringJoin["{", Riffle[Map[argString, e], ","], "}"];
argString[e_List] := "{\[Ellipsis]}";
argString[e_] := $Failed;


PackageScope["SkeletonString"]

SkeletonString[n_] := StringJoin["\[LeftGuillemet]", IntegerString[n], "\[RightGuillemet]"];


PackageScope["StringFormToString"]

StringFormToString[s_StringForm] := TextString @ ReplaceRepeated[s, {
	Style[a_, ___] :> a,
	Row[a_List, ","] :> Row[a, ", "],
	Subscript[a_, b_] :> Row[{a, b}],
	lv_LengthVar :> FormatLengthVar[lv],
	TwoWayRule[a_, b_] -> Row[{a, " <-> ", b}]
}];
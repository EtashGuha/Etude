(* Generated using

allSyms = WolframLanguageData[All];
allSymsShort = WolframLanguageData[All, "ShortNotations"];

shortMap =
	Composition[
		KeyMap[FromEntity],
		Map[First],
		Select[!MissingQ[#]&]
	][
		AssociationThread[allSyms -> allSymsShort]
	]
AssociateTo[shortMap, Times -> "*"];
AssociateTo[shortMap, Rule -> "\[Rule]"];
KeyDropFrom[shortMap, Function];
KeyDropFrom[shortMap, Association];
KeyDropFrom[shortMap, Part];
shortMap

*)


BeginPackage["CompileAST`Language`ShortNames`"]

$ShortNames
$SystemShortNames

Begin["`Private`"]

symName[x_] := Quiet[With[{sym = SymbolName[x]}, If[AtomQ[sym], sym, Nothing]]]
$ShortNames := $ShortNames =
	KeyMap[symName, $SystemShortNames]
$SystemShortNames := $SystemShortNames =
<|Union -> "\[Union]",  
 Star -> "\[Star]", Square -> "\[Square]", Cross -> "\[Cross]", 
 Power -> "^", Element -> "\[Element]", I -> "\[ImaginaryI]", 
 D -> "\[PartialD]_\[Placeholder]\[SelectionPlaceholder]", 
 E -> "\[ExponentialE]", Nor -> "\[Nor]", Diamond -> "\[Diamond]", 
 Intersection -> "\[Intersection]", And -> "&&", 
 Blank -> "_", Cap -> "\[Cap]", Colon -> "\[Colon]", 
 Curl -> "\[Del]_\\[SelecttionPlaceholder]\[Cross]\[Placeholder]", \
\[Degree] -> "\[Degree]", Del -> "\[Del]", 
 Floor -> "\[LeftFloor]\[Ellipsis]\[RightFloor]", 
 Grad -> "\[Del]_\[SelectionPlaceholder]\[Placeholder]", Less -> "<", 
 List -> "{\[Ellipsis]}", Nand -> "\[Nand]", Not -> "!", 
 Or -> "||", \[Pi] -> "\[Pi]", Rule -> "\[Rule]", Set -> "=", Slot -> "#", 
 Span -> ";;", 
 Sum -> "\[Sum]_\[Placeholder]=\[Placeholder]^\[Placeholder]\
\\[SelctionPlaceholder]", Times -> "*", Vee -> "\[Vee]", 
 Wedge -> "\[Wedge]", Dot -> ".", Greater -> ">", 
 Laplacian -> "\[Del]^2_\[SelectionPlaceholder]\[Placeholder]", 
 Map -> "/@", Backslash -> "\[Backslash]", 
 Coproduct -> "\[Coproduct]", Out -> "%", Piecewise -> "\[Piecewise]",
  Plus -> "+", 
 ConjugateTranspose -> "\[SelectionPlaceholder]\[ConjugateTranspose]",
  PlusMinus -> "\[PlusMinus]", 
 AngleBracket -> 
  "\[LeftAngleBracket]\[SelectionPlaceholder]\[RightAngleBracket]", 
 Because -> "\[Because]", Condition -> "/;", 
 Congruent -> "\[Congruent]", Cup -> "\[Cup]", 
 DiscreteRatio -> "\[DiscreteRatio]", 
 DiscreteShift -> "\[DiscreteShift]", Divide -> "/", Equal -> "==", 
 Equivalent -> "\[Equivalent]", Exists -> "\[Exists]", 
 ForAll -> "\[ForAll]", 
 Implies -> "\[Implies]", \[Infinity] -> "\[Infinity]", 
 Precedes -> "\[Precedes]", Prefix -> "@", 
 Product -> 
  "\[Product]_\[Placeholder]=\[Placeholder]^\[Placeholder]\
\\[SelctionPlaceholder]", Proportional -> "\[Proportional]", 
 RightTriangle -> "\[RightTriangle]", SmallCircle -> "\[SmallCircle]",
  Subset -> "\[Subset]", Succeeds -> "\[Succeeds]", 
 SuchThat -> "\[SuchThat]", Superset -> "\[Superset]", 
 Therefore -> "\[Therefore]", Tilde -> "\[Tilde]", 
 Transpose -> "\[Transpose]", Unequal -> "!=", AddTo -> "+=", 
 Alternatives -> "|", Apply -> "@@", BlankNullSequence -> "___", 
 BlankSequence -> "__", 
 BracketingBar -> 
  "\[LeftBracketingBar]\[SelectionPlaceholder]\[RightBracketingBar]", 
 CapitalDifferentialD -> 
  "\[CapitalDifferentialD]\[SelectionPlaceholder]", 
 Ceiling -> "\[LeftCeiling]\[SelectionPlaceholder]\[RightCeiling]", 
 CenterDot -> "\[CenterDot]", CircleDot -> "\[CircleDot]", 
 CircleMinus -> "\[CircleMinus]", CirclePlus -> "\[CirclePlus]", 
 CircleTimes -> "\[CircleTimes]", CompoundExpression -> ";", 
 Conditioned -> "\[Conditioned]", Conjugate -> "\[Conjugate]", 
 ContinuedFractionK -> "\[ContinuedFractionK]", CupCap -> "\[CupCap]",
  Decrement -> "--", DifferenceDelta -> "\[DifferenceDelta]", 
 DirectedEdge -> "\[DirectedEdge]", Distributed -> "\[Distributed]", 
 DivideBy -> "/=", DotEqual -> "\[DotEqual]", 
 DoubleBracketingBar -> 
  "\[LeftDoubleBracketingBar]\[SelectionPlaceholder]\
\[RightDoubleBracketingBar]", DoubleDownArrow -> "\[DoubleDownArrow]",
  DoubleUpArrow -> "\[DoubleUpArrow]", 
 DoubleVerticalBar -> "\[DoubleVerticalBar]", 
 DownArrow -> "\[DownArrow]", DownArrowBar -> "\[DownArrowBar]", 
 DownLeftRightVector -> "\[DownLeftRightVector]", 
 DownLeftTeeVector -> "\[DownLeftTeeVector]", 
 DownLeftVector -> "\[DownLeftVector]", 
 DownLeftVectorBar -> "\[DownLeftVectorBar]", 
 DownRightTeeVector -> "\[DownRightTeeVector]", 
 DownRightVector -> "\[DownRightVector]", 
 DownRightVectorBar -> "\[DownRightVectorBar]", 
 DownTeeArrow -> "\[DownTeeArrow]", EqualTilde -> "\[EqualTilde]", 
 Equilibrium -> "\[Equilibrium]", Get -> "<<", GreaterEqual -> ">=", 
 GreaterEqualLess -> "\[GreaterEqualLess]", 
 GreaterFullEqual -> "\[GreaterFullEqual]", 
 GreaterGreater -> "\[GreaterGreater]", 
 GreaterLess -> "\[GreaterLess]", 
 GreaterSlantEqual -> "\[GreaterSlantEqual]", 
 GreaterTilde -> "\[GreaterTilde]", HumpDownHump -> "\[HumpDownHump]",
  HumpEqual -> "\[HumpEqual]", Increment -> "++", 
 Infix -> "\[Placeholder]~\[Placeholder]", 
 Integrate -> 
  "\[Integral]\[SelectionPlaceholder] \[DifferentialD]\[Placeholder]",
  LeftDownTeeVector -> "\[LeftDownTeeVector]", 
 LeftDownVector -> "\[LeftDownVector]", 
 LeftDownVectorBar -> "\[LeftDownVectorBar]", 
 LeftRightVector -> "\[LeftRightVector]", 
 LeftTeeVector -> "\[LeftTeeVector]", 
 LeftTriangle -> "\[LeftTriangle]", 
 LeftTriangleBar -> "\[LeftTriangleBar]", 
 LeftTriangleEqual -> "\[LeftTriangleEqual]", 
 LeftUpDownVector -> "\[LeftUpDownVector]", 
 LeftUpTeeVector -> "\[LeftUpTeeVector]", 
 LeftUpVector -> "\[LeftUpVector]", 
 LeftUpVectorBar -> "\[LeftUpVectorBar]", 
 LeftVector -> "\[LeftVector]", LeftVectorBar -> "\[LeftVectorBar]", 
 LessEqual -> "<=", LessEqualGreater -> "\[LessEqualGreater]", 
 LessFullEqual -> "\[LessFullEqual]", LessGreater -> "\[LessGreater]",
  LessLess -> "\[LessLess]", LessSlantEqual -> "\[LessSlantEqual]", 
 LessTilde -> "\[LessTilde]", MapAll -> "//@", MessageName -> "::", 
 MinusPlus -> "\[MinusPlus]", 
 NestedGreaterGreater -> "\[NestedGreaterGreater]", 
 NestedLessLess -> "\[NestedLessLess]", 
 NonCommutativeMultiply -> "**", NotCongruent -> "\[NotCongruent]", 
 NotCupCap -> "\[NotCupCap]", 
 NotDoubleVerticalBar -> "\[NotDoubleVerticalBar]", 
 NotElement -> "\[NotElement]", NotExists -> "\[NotExists]", 
 NotGreater -> "\[NotGreater]", 
 NotGreaterEqual -> "\[NotGreaterEqual]", 
 NotGreaterFullEqual -> "\[NotGreaterFullEqual]", 
 NotGreaterGreater -> "\[NotGreaterGreater]", 
 NotGreaterLess -> "\[NotGreaterLess]", 
 NotGreaterSlantEqual -> "\[NotGreaterSlantEqual]", 
 NotGreaterTilde -> "\[NotGreaterTilde]", 
 NotHumpDownHump -> "\[NotHumpDownHump]", 
 NotHumpEqual -> "\[NotHumpEqual]", 
 NotLeftTriangle -> "\[NotLeftTriangle]", 
 NotLeftTriangleBar -> "\[NotLeftTriangleBar]", 
 NotLeftTriangleEqual -> "\[NotLeftTriangleEqual]", 
 NotLess -> "\[NotLess]", NotLessEqual -> "\[NotLessEqual]", 
 NotLessFullEqual -> "\[NotLessFullEqual]", 
 NotLessGreater -> "\[NotLessGreater]", 
 NotLessLess -> "\[NotLessLess]", 
 NotLessSlantEqual -> "\[NotLessSlantEqual]", 
 NotLessTilde -> "\[NotLessTilde]", 
 NotNestedGreaterGreater -> "\[NotNestedGreaterGreater]", 
 NotNestedLessLess -> "\[NotNestedLessLess]", 
 NotPrecedes -> "\[NotPrecedes]", 
 NotPrecedesEqual -> "\[NotPrecedesEqual]", 
 NotPrecedesSlantEqual -> "\[NotPrecedesSlantEqual]", 
 NotPrecedesTilde -> "\[NotPrecedesTilde]", 
 NotReverseElement -> "\[NotReverseElement]", 
 NotRightTriangle -> "\[NotRightTriangle]", 
 NotRightTriangleBar -> "\[NotRightTriangleBar]", 
 NotRightTriangleEqual -> "\[NotRightTriangleEqual]", 
 NotSquareSubset -> "\[NotSquareSubset]", 
 NotSquareSubsetEqual -> "\[NotSquareSubsetEqual]", 
 NotSquareSuperset -> "\[NotSquareSuperset]", 
 NotSquareSupersetEqual -> "\[NotSquareSupersetEqual]", 
 NotSubset -> "\[NotSubset]", NotSubsetEqual -> "\[NotSubsetEqual]", 
 NotSucceeds -> "\[NotSucceeds]", 
 NotSucceedsEqual -> "\[NotSucceedsEqual]", 
 NotSucceedsSlantEqual -> "\[NotSucceedsSlantEqual]", 
 NotSucceedsTilde -> "\[NotSucceedsTilde]", 
 NotSuperset -> "\[NotSuperset]", 
 NotSupersetEqual -> "\[NotSupersetEqual]", NotTilde -> "\[NotTilde]",
  NotTildeEqual -> "\[NotTildeEqual]", 
 NotTildeFullEqual -> "\[NotTildeFullEqual]", 
 NotTildeTilde -> "\[NotTildeTilde]", 
 NotVerticalBar -> "\[NotVerticalBar]", Optional -> ".", 
 Pattern -> ":", PatternTest -> "?", 
 Postfix -> "\[SelectionPlaceholder]//", 
 PrecedesEqual -> "\[PrecedesEqual]", 
 PrecedesSlantEqual -> "\[PrecedesSlantEqual]", 
 PrecedesTilde -> "\[PrecedesTilde]", PreDecrement -> "--", 
 PreIncrement -> "++", Proportion -> "\[Proportion]", 
 Repeated -> "..", RepeatedNull -> "...", ReplaceAll -> "/.", 
 ReplaceRepeated -> "//.", ReverseElement -> "\[ReverseElement]", 
 ReverseEquilibrium -> "\[ReverseEquilibrium]", 
 ReverseUpEquilibrium -> "\[ReverseUpEquilibrium]", 
 RightDownTeeVector -> "\[RightDownTeeVector]", 
 RightDownVector -> "\[RightDownVector]", 
 RightDownVectorBar -> "\[RightDownVectorBar]", 
 RightTeeVector -> "\[RightTeeVector]", 
 RightTriangleBar -> "\[RightTriangleBar]", 
 RightTriangleEqual -> "\[RightTriangleEqual]", 
 RightUpDownVector -> "\[RightUpDownVector]", 
 RightUpTeeVector -> "\[RightUpTeeVector]", 
 RightUpVector -> "\[RightUpVector]", 
 RightUpVectorBar -> "\[RightUpVectorBar]", 
 RightVector -> "\[RightVector]", 
 RightVectorBar -> "\[RightVectorBar]", RuleDelayed -> ":>", 
 SameQ -> "===", SetDelayed -> ":=", SlotSequence -> "##", 
 Sqrt -> "\[Sqrt]", SquareIntersection -> "\[SquareIntersection]", 
 SquareSubset -> "\[SquareSubset]", 
 SquareSubsetEqual -> "\[SquareSubsetEqual]", 
 SquareSuperset -> "\[SquareSuperset]", 
 SquareSupersetEqual -> "\[SquareSupersetEqual]", 
 SquareUnion -> "\[SquareUnion]", StringExpression -> "~~", 
 StringJoin -> "<>", SubsetEqual -> "\[SubsetEqual]", Subtract -> "-",
  SubtractFrom -> "-=", SucceedsEqual -> "\[SucceedsEqual]", 
 SucceedsSlantEqual -> "\[SucceedsSlantEqual]", 
 SucceedsTilde -> "\[SucceedsTilde]", 
 SuperDagger -> "\[SelectionPlaceholder]^\[Dagger]", 
 SupersetEqual -> "\[SupersetEqual]", TagSet -> "/:\[Ellipsis]=", 
 TagSetDelayed -> "/:\[Ellipsis]:=", TagUnset -> "/:\[Ellipsis]=.", 
 TildeEqual -> "\[TildeEqual]", TildeFullEqual -> "\[TildeFullEqual]",
  TildeTilde -> "\[TildeTilde]", TimesBy -> "*=", 
 UndirectedEdge -> "\[UndirectedEdge]", UnionPlus -> "\[UnionPlus]", 
 UnsameQ -> "=!=", Unset -> "=.", UpArrow -> "\[UpArrow]", 
 UpArrowBar -> "\[UpArrowBar]", UpEquilibrium -> "\[UpEquilibrium]", 
 VerticalBar -> "\[VerticalBar]", 
 VerticalSeparator -> "\[VerticalSeparator]", 
 VerticalTilde -> "\[VerticalTilde]", Xnor -> "\[Xnor]", 
 Xor -> "\[Xor]", DoubleLeftTee -> "\[DoubleLeftTee]", 
 DoubleRightTee -> "\[DoubleRightTee]", DownTee -> "\[DownTee]", 
 LeftTee -> "\[LeftTee]", RightTee -> "\[RightTee]", 
 UpTee -> "\[UpTee]", Div -> "\[Del]\[Ellipsis].\[Ellipsis]"|>
End[]

EndPackage[]

BeginPackage["CompileAST`Class`Operators`"]

$ExprOperators;
$ExprSymbols;

Begin["`Private`"]

operators := operators =
	Internal`Bag[]
symbols := symbols =
	Internal`Bag[]

declareOperator[op_] :=
	Internal`StuffBag[operators, op]
declareSymbol[sym_] :=
	Internal`StuffBag[symbols, sym]

(* From /Java/Libraries/MExpr/com/wolfram/mexpr/MExpr.java *)

declareOperator["Plus"];
declareOperator["RightComposition"];
declareOperator["Composition"];
declareOperator["PreIncrement"];
declareOperator["PreDecrement"];
declareOperator["Increment"];
declareOperator["Decrement"];

declareOperator["Times"];
declareOperator["Rule"];
declareOperator["UndirectedEdge"];
declareOperator["RuleDelayed"];
declareOperator["Power"];
declareOperator["List"];
declareOperator["Association"];

declareOperator["Set"];
declareOperator["SetDelayed"];
declareOperator["UpSet"];
declareOperator["UpSetDelayed"];
declareOperator["TagSet"];
declareOperator["TagSetDelayed"];

declareOperator["Blank"];
declareOperator["BlankSequence"];
declareOperator["BlankNullSequence"];
 
declareOperator["Pattern"];
declareOperator["Optional"];
declareOperator["Condition"];
declareOperator["Alternatives"];
   
declareOperator["Dot"];
declareOperator["Function"];
declareOperator["Part"];
declareOperator["CompoundExpression"];

declareOperator["Not"];

declareOperator["Factorial"];
declareOperator["Factorial2"];
declareOperator["Derivative1"];
  
declareOperator["PatternTest"];

declareOperator["Null"];

declareOperator["SameQ"];
declareOperator["UnsameQ"];

declareOperator["Equal"];
declareOperator["Unequal"];
declareOperator["Greater"];
declareOperator["Less"];
declareOperator["GreaterEqual"];
declareOperator["LessEqual"];
declareOperator["Inequality"];

declareOperator["Get"];
declareOperator["Put"];
declareOperator["PutAppend"];

declareOperator["Package"];
declareOperator["JoinPackage"];

declareOperator["MessageName"];
declareOperator["Unset"];
declareOperator["TagUnset"];

declareOperator["Repeated"];
declareOperator["RepeatedNull"];

declareOperator["Slot"];
declareOperator["SlotSequence"];

declareOperator["Map"];
declareOperator["MapAll"];
declareOperator["Apply"];
declareOperator["ApplyOne"];

declareOperator["And"];
declareOperator["Or"];

declareOperator["StringJoin"];
declareOperator["StringExpression"];
  
declareOperator["ReplaceAll"];
declareOperator["ReplaceRepeated"];

declareOperator["Parenthesis"];

declareOperator["Minus"];
declareOperator["Subtract"];
declareOperator["Divide"];

declareOperator["Reciprocal"];

declareOperator["AddTo"];
declareOperator["SubtractFrom"];
declareOperator["TimesBy"];
declareOperator["DivideBy"];
declareOperator["NonCommutativeMultiply"];
declareOperator["Out"];


declareOperator["Information"];
declareOperator["InformationLong"];

declareOperator["Span"];

declareSymbol["<ENULL>"];

declareSymbol["Symbol"];
declareSymbol["String"];
declareSymbol["Integer"];
declareSymbol["Real"];
declareSymbol["Typeset"];

declareSymbol["ERROR_NODE"];

declareOperator["TypesetParen"];
declareOperator["TypesetBracket"];
declareOperator["TypesetSuperscript"];
declareOperator["TypesetSubscript"];
declareOperator["TypesetDivide"];
declareOperator["TypesetSqrt"];
declareOperator["TypesetFullForm"];

declareSymbol["Sequence"];


$ExprOperators := $ExprOperators = 
	Internal`BagPart[operators, All];
$ExprSymbols := $ExprSymbols = 
	Internal`BagPart[symbols, All];

End[]

EndPackage[]
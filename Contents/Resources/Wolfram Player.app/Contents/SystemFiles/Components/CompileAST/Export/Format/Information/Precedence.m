

BeginPackage["CompileAST`Export`Format`Information`Precedence`"]

$OperatorPrecedence;

formattingNeedsParenQ;

Begin["`Private`"] 


Needs["CompileAST`Class`Symbol`"]


(* TODO: Should MAXVALUE be used somewhere? Currently it isn't. *)
MAXVALUE = -1;
NOVALUE = -2;
NOPAREN = -3;

formattingNeedsParenQ[innerHead_?MExprSymbolQ, outerHead_?MExprSymbolQ] :=
	If[!innerHead["isOperator"] || !outerHead["isOperator"],
		False,
		(* Else *)
		With[{
			precInner = Lookup[$OperatorPrecedence, innerHead["name"], NOVALUE],
			precOuter = Lookup[$OperatorPrecedence, outerHead["name"], NOVALUE]
		},
			Which[
				precInner === MAXVALUE,
					False,
				precOuter === MAXVALUE,
					True,
				(* This case only would only ever execute because of an implementation error. If
				   `innerHead` and `outerHead` are operators (as tested above), they should have an
				   entry in $OperatorPrecedence. This isn't quite spectacular enough to error though
				   so we'll do the conservative thing signal it needs parentheses *)
				precInner === NOVALUE || precOuter === NOVALUE,
					True,
				precInner === NOPAREN || precOuter === NOPAREN,
					False,
				True,
					precInner <= precOuter
			]
		]
	]


$OperatorPrecedence := $OperatorPrecedence = <|
	"Parenthesis" -> NOPAREN,
	"List" -> NOPAREN,
	
	"Blank" -> 760,
	"BlankSequence" -> 760,
	"BlankNullSequence" -> 760,
	"MessageName" -> 750,
	"Slot" -> 740,
	"Get" -> 720,
	"PutAppend" -> 720,
	"PatternTest" -> 680,
	"Information" -> 680,
	"InformationLong" -> 680,
	"Part" -> 670,
	"Increment" -> 660,
	"PreIncrement" -> 660,
	"PreDecrement" -> 660,
	"Decrement" -> 660,
	"Map" -> 620,
	"MapAll" -> 620,
	"Apply" -> 620,
	"ApplyOne" -> 620,
	
	"Factorial" -> 610,
	"Factorial2" -> 610,
	"Derivative" -> 605,
	"Derivative1" -> 605,
	"Join" -> 600,
	"Power" -> 590,
	"Dot" -> 490,
	"Minus" -> 480,
	"Divide" -> 470,
	"Reciprocal" -> 470,
	"Times" -> 400,
	"Plus" -> 310,
	"Subtract" -> 310,
	"Span" -> 305,
	"Equal" -> 290,
	"Unequal" -> 290,
	"Greater" -> 290,
	"Less" -> 290,
	"GreaterEqual" -> 290,
	"LessEqual" -> 290,
	"SameQ" -> 260,
	"UnsameQ" -> 260,
	"Not" -> 230,
	"And" -> 215,
	"Or" -> 215,
	"Repeated" -> 170,
	"RepeatedNull" -> 170,
	"Alternatives" -> 160,
	"Pattern" -> 150,
	"Optional" -> 140,
	"Condition" -> 130,
	"Rule" -> 120,
	"RuleDelayed" -> 120,
	"ReplaceAll" -> 110,
	"AddTo" -> 100,
	"SubtractFrom" -> 100,
	"TimesBy" -> 100,
	"DivideBy" -> 100,
	"Function" -> 90,
	"Set" -> 40,
	"SetDelayed" -> 40,
	"UpSet" -> 40,
	"UpSetDelayed" -> 40,
	"TagSet" -> 40,
	"TagSetDelayed" -> 40,
	"Put" -> 30,
	"CompoundExpression" -> 10,
	"StringJoin" -> NOVALUE,
	"StringExpression" -> NOVALUE
|>;

End[]

EndPackage[]

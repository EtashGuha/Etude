

BeginPackage["CompileAST`Utilities`Node`"]

$ASTNodes
ASTNodeType

Begin["`Private`"] 


ASTNodeType = <||>

ASTNodeType[Function] = "Function";
ASTNodeType[Compile] = "Compile";

ASTNodeType[Native`DeclareVariable] = "DeclareVariable";

ScopingNodes = {
	Module,
	With
};

Do[
	ASTNodeType[node] = "Scope",
	{node, ScopingNodes}
]

DynamicScopingNodes = {
	Block
};

Do[
	ASTNodeType[node] = "DynamicScope",
	{node, DynamicScopingNodes}
]

StatementNodes = {
};

Do[
	ASTNodeType[node] = "Statement",
	{node, StatementNodes}
]

ExprNodes = {
	List
};

Do[
	ASTNodeType[node] = "Expr",
	{node, ExprNodes}
]

AtomNodes = {
	Integer,
	Real,
	Complex,
	Rational,
	String,
	Symbol
};

Do[
	ASTNodeType[node] = "Atom",
	{node, AtomNodes}
]

BinaryOpNodes = {
	Plus,
	Subtract,
	Divide,
	Times,
	Equal,
	SameQ,
	Less,
	Minus,
	AddTo,
	SubtractFrom,
	TimesBy,
	DivideBy
};

Do[
	ASTNodeType[node] = "BinaryOp",
	{node, BinaryOpNodes}
]

$ASTNodes = Join[
	ExprNodes,
	AtomNodes,
	ScopingNodes,
	StatementNodes,
	BinaryOpNodes
];


End[]

EndPackage[]

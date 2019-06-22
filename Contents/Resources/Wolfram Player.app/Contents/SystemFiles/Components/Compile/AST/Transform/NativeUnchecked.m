
BeginPackage["Compile`AST`Transform`NativeUnchecked`"]

NativeUncheckedPass



Begin["`Private`"] 

Needs["CompileAST`Utilities`MExprVisitor`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`PassManager`MExprPass`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileAST`Class`Base`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["CompileAST`Create`Construct`"]


visitNormal[self_, mexpr_] := Module[{checkedSet = False},
	
	Which[
		mexpr["hasHead", Native`CheckedBlockRestore],
            self["setPeekChecked"];
            checkedSet = True;
            mexpr["setHead", CreateMExprSymbol[Module]];
            mexpr["setArguments", Prepend[mexpr["arguments"], CreateMExpr[{}]]]
            ,
		mexpr["hasHead", Native`CheckedBlock],
            self["pushChecked", True];
            checkedSet = True;
            mexpr["setHead", CreateMExprSymbol[Module]];
            mexpr["setArguments", Prepend[mexpr["arguments"], CreateMExpr[{}]]]
            ,
		mexpr["hasHead", Native`UncheckedBlock] || mexpr["hasHead", Native`UnCheckedBlock],
			self["pushChecked", False];
			checkedSet = True;
            mexpr["setHead", CreateMExprSymbol[Module]];
            mexpr["setArguments", Prepend[mexpr["arguments"], CreateMExpr[{}]]];
            ,
		self["isChecked"] === False && mexpr["head"]["symbolQ"],
			Switch[mexpr["head"]["symbol"],
				Plus,
					mexpr["setHead", Native`Unchecked[Plus]],
				Subtract,
					mexpr["setHead", Native`Unchecked[Subtract]],
				Times,
					mexpr["setHead", Native`Unchecked[Times]],
				Divide,
					mexpr["setHead", Native`Unchecked[Divide]],
				Minus,
					mexpr["setHead", Native`Unchecked[Minus]],
				Cos,
					mexpr["setHead", Native`Unchecked[Cos]],
				Sin,
					mexpr["setHead", Native`Unchecked[Sin]],
				Tan,
					mexpr["setHead", Native`Unchecked[Tan]],
				BitAnd,
					mexpr["setHead", Native`Unchecked[BitAnd]],
				BitOr,
					mexpr["setHead", Native`Unchecked[BitOr]],
				BitShiftLeft,
					mexpr["setHead", Native`Unchecked[BitShiftLeft]],
				BitShiftRight,
					mexpr["setHead", Native`Unchecked[BitShiftRight]],
				BitXor,
					mexpr["setHead", Native`Unchecked[BitXor]],
				Sqrt,
					mexpr["setHead", Native`Unchecked[Sqrt]],
				Power,
					mexpr["setHead", Native`Unchecked[Power]],
				Exp,
					mexpr["setHead", Native`Unchecked[Exp]],
				Log,
					mexpr["setHead", Native`Unchecked[Log]],
				Abs,
					mexpr["setHead", Native`Unchecked[Abs]],
				Min,
					mexpr["setHead", Native`Unchecked[Min]],
				Max,
					mexpr["setHead", Native`Unchecked[Max]],
				Floor,
					mexpr["setHead", Native`Unchecked[Floor]],
				Ceiling,
					mexpr["setHead", Native`Unchecked[Ceiling]],
				BitLength,
					mexpr["setHead", Native`Unchecked[BitLength]],
				Native`GetPartUnary,
					mexpr["setHead", Native`Unchecked[Native`GetPartUnary]],
				Native`SetPartUnary,
					mexpr["setHead", Native`Unchecked[Native`SetPartUnary]],
				_,
					Null
			],
		True,
			Null
	];
	mexpr["head"]["accept", self];
	Scan[ #["accept",self]&, mexpr["arguments"]];
	If[checkedSet,
		self["popChecked"]];
	False
]

setPeekChecked[self_] :=
	Module[{val},
		val = self["checkedStack"]["last"];
		self["pushChecked", val];
	]

pushChecked[self_,val_] :=
	Module[{},
		self["checkedStack"]["pushBack", self["isChecked"]];
		self["setIsChecked", val];
	]

popChecked[self_] :=
	self["setIsChecked",self["checkedStack"]["popBack"]];

RegisterCallback["DeclareCompileClass", Function[{st},
DeclareClass[NativeUncheckedPassVisitor,
    <| 
    	"pushChecked" -> (pushChecked[Self,#]&),
    	"popChecked" -> (popChecked[Self]&),
    	"setPeekChecked" -> (setPeekChecked[Self]&),
        "visitNormal"   -> (visitNormal[Self, ##]&)
    |>,
    {"isChecked", "checkedStack"},
    Extends -> {MExprVisitorClass}
];
]]


run[mexpr_?MExprQ, opts_:<||>] :=
    With[{
       visitor = CreateObject[NativeUncheckedPassVisitor, 
			<| 	"isChecked" -> True, (* Default checked is true *)
				"checkedStack" -> CreateReference[{True}] (* So Native`CheckedBlockRestore can be called first *)
			|>
		]
    },
       mexpr["accept", visitor];
       mexpr
    ]
    

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
    "NativeUnchecked",
    "Converts the operations checked operations within the program into their unchecked counterpart. " <>
    "The code does not modify nodes within blocks that are marked Native`CheckedBlock."
];


NativeUncheckedPass = CreateMExprPass[<|
    "information" -> info,
    "runPass" -> run
|>];
RegisterPass[NativeUncheckedPass]
]]


End[]

EndPackage[]

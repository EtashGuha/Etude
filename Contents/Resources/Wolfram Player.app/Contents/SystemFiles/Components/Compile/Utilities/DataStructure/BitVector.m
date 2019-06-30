
BeginPackage["Compile`Utilities`DataStructure`BitVector`"]

BitVector;
BitVectorQ;
BitVectorClass;
CreateBitVector;

Begin["`Private`"] 

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Asserter`Assert`"]
Needs["CompileUtilities`Callback`"]




complement[self_] := (
	self["set_data", self["_data"] /. {0 -> 1, 1 -> 0}];
	self
)
complement[self_, other_] :=
	Module[{otherClone},
		AssertThat["Complement only accepts a bitvector", other]["isA", BitVector];
		otherClone = other["clone"];
		otherClone["complement"];
		self["intersection", otherClone];
		self
	]
clone[self_, ___] :=
	CreateObject[
		BitVector,
		<|
			"_elements" -> self["_elements"],
			"_data" -> self["_data"]
		|>
	]


RegisterCallback["DeclareCompileClass", Function[{st},
BitVectorClass = DeclareClass[
	BitVector,
	<|
		"clone" -> (clone[Self]&),
		"elements" -> Function[{},
			Pick[
				Keys[Self["_elements"]],
				Self["_data"],
				1
			]
		], 
		"toList" -> Function[{}, Self["elements"]],
		"intersection" -> Function[{other},
			Module[{keys, data},
				AssertThat["Intersection only accepts a bitvector", other]["isA", BitVector];
				keys = Union[Keys[Self["_elements"]], Keys[other["_elements"]]];
				data = Sign[Map[Self["getElement", #] * other["getElement", #]&, keys]];
				Self["set_elements", AssociationThread[keys -> Range[Length[keys]]]];
				Self["set_data", data];
				Self
			]
		],
		"union" -> Function[{other},
			Module[{keys, data},
				AssertThat["Union only accepts a bitvector", other]["isA", BitVector];
				keys = Union[Keys[Self["_elements"]], Keys[other["_elements"]]];
				data = Sign[Map[Self["getElement", #] + other["getElement", #]&, keys]];
				Self["set_elements", AssociationThread[keys -> Range[Length[keys]]]];
				Self["set_data", data];
				Self
			]
		],
		"complement" -> (complement[Self, ##]&),
		"count" -> Function[{}, Count[Self["_data"], 1]],
		"any" -> Function[{}, Count[Self["_data"], 1] > 0],
		"all" -> Function[{}, Count[Self["_data"], 1] === Length[Self["_data"]]],
		"none" -> Function[{}, !Self["any"]],
		"first" -> Function[{}, SelectFirst[Self["_data"], # === 1&, None]],
		"clear" -> Function[{}, Self["set_data", ConstantArray[0, Length[Self["_data"]]]]; Self],
		"addElement" -> Function[{elem},
			If[KeyExistsQ[Self["_elements"], elem],
				Self["set_data", ReplacePart[Self["_data"], Self["_elements"][elem] -> 1]]
				,
				Self["set_data", Append[Self["_data"], 1]];
				Self["_elements", elem -> Length[Self["_data"]]]
			];
			Self
		],
		"removeElement" -> Function[{elem},
			If[KeyExistsQ[Self["_elements"], elem],
				Self["set_data", ReplacePart[Self["_data"], Self["_elements"][elem] -> 0]]
			];
			Self
		],
		"getElement" -> Function[{elem}, 
			If[KeyExistsQ[Self["_elements"], elem],
				Part[Self["_data"], Self["_elements"][elem]],
				0
			]
		],
		"flipElement" -> Function[{elem},
			Self["addElement",
				If[Self["getElement", elem] === 1, 0, 1]
			];
			Self
		],
		"equals" -> Function[{other},
			Self["size"] === other["size"] &&
			Self["_data"] === other["_data"] &&
			Self["_elements"] === other["_elements"]
		],
		"size" -> Function[{}, Length[Self["_data"]]],
		"toString" -> Function[{}, ToString[Self["toList"]]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
		"_elements",
		"_data"
	},
	Predicate -> BitVectorQ
]
]]


CreateBitVector[elements_?ListQ] :=
	CreateObject[
		BitVector,
		<|
			"_elements" -> AssociationThread[elements -> Range[Length[elements]]],
			"_data" -> ConstantArray[0, Length[elements]]
		|>
	]
CreateBitVector[bv_?BitVectorQ] :=
	bv["clone"]

(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)

icon := Graphics[Text[
  Style["BVec", GrayLevel[0.7], Bold, 
   CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];  
     
toBoxes[var_?BitVectorQ, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		BitVector,
		var,
  		icon,
  		{
  		    BoxForm`SummaryItem[{"elements: ", var["toList"]}]
  		},
  		{}, 
  		fmt
  	]
End[]

EndPackage[]

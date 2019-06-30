
BeginPackage["TypeFramework`Inference`ConstraintData`"]

ConstraintDataQ
CreateConstraintData


Begin["`Private`"]


Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["TypeFramework`ConstraintObjects`EqualConstraint`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]


(*
  The constraint data holds constraints to be solved,  it applies substitutions
  to constraints,  selects constraints to be solved.
    
  It keeps a map called the variable Map which goes from variables to constraints that 
  use that variable.  This has the form varId -> <|consId -> cons, ...|>
  
  When solving we get a solution for a constraint,  we then use the free variables from
  that constraint to find constraints that use those variables.  
  For this we use the variableMap.
  
  If a constraint is removed from the map we call deleteLink.
  
  If a constraint is added to the map we call addLink.
  
  The system also keeps lists to be solved.  The function nextToSolve returns the 
  most optimal constraint to be solved removing it from the lists.  Since the constraints
  in the lists might have been changed by substitution we keep a map of redirects.
  
  If we replace conOld with conNew we use the redirectMap and reverseRedirectMap.
  
  1) lookup in reverseRedirectMap for conOld with a default of conOld
     
     this returns conOld if it has not already been substituted for
     
  2) we then add an entry to redirectMap from the result of 1) to conNew
  
  3) we then add an entry to reverseRedirectMap from conNew to the result of 1
   
  when we take a constraint from the constraint list we then look it up in 
  the redirectMap to see if it has been modified, we return the modified version 
  or the constraint if it is not there.
  
  Initially equal constraints are put into the equalConstraint list and the others into
  otherConstraints.  We take from the equalConstraint list first.  If there is an error 
  solving a constraint it gets put to the back of the respective queue, except that an 
  equalConstraint is put to the back of the  otherConstraints list the second time it 
  errors.  We try to solve more than once because it might be that solving other constraints
  helps a constraint to be solved.
*)

RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
ConstraintDataClass = DeclareClass[
	ConstraintData,
	<|
		"applySubstitution" -> Function[{cons, subs}, applySubstitution[Self, cons, subs]],
		"insert" -> Function[{newCons}, insert[Self, newCons]],
		"insertBack" -> Function[{cons}, insertBack[Self, cons]],
		"getOriginal" -> Function[{c}, getOriginal[Self,c]],
		"getUnsolved" -> Function[{}, getUnsolved[Self]],
		"deleteLink" -> Function[{cons}, deleteLink[Self, cons]],
		"createRules" -> Function[{}, createRules[Self]],
		"createGraph" -> Function[{}, createGraph[Self]],
		"emptyQ" -> Function[{}, emptyQ[Self]],
		"nextToSolve" -> Function[{}, nextToSolve[Self]],
		"format" -> Function[{}, format[Self]],
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
		"variableMap",
		"equalConstraints",
		"otherConstraints",
		"redirectMap",
		"reverseRedirectMap"
	},
	Predicate -> ConstraintDataQ
]
]]

CreateConstraintData[consList_] :=
	Module[ {obj},
		obj = CreateObject[ConstraintData, <|
			"variableMap" -> CreateReference[<||>],
			"equalConstraints" -> CreateReference[{}],
			"otherConstraints" -> CreateReference[{}],
			"redirectMap" -> CreateReference[<||>],
			"reverseRedirectMap" -> CreateReference[<||>]
		|>];
		setupMaps[obj, consList];		
		Scan[
			insertConstraint[obj, #]&, consList];
		verify[obj];
		obj
	]






(*
   apply the substitution to the constraints reachable from cons.
   This might result in constraints getting changed in which case 
   we need to update the redirect,  removing the old constraint and
   add the new one.
*)
applySubstitution[self_, cons_, subs_] :=
	Module[ {varEnt, conNew, consList, consSet = CreateReference[<||>], freeVars = Values[cons["free"]]},
		
		Scan[
			Function[var,
				varEnt = self["variableMap"]["lookup", var["id"], Null];
				If[ varEnt === Null,
					ThrowException[{"Cannot find variable entry in variable map", var}]
				];
				Scan[consSet["associateTo", #["id"] -> #]&, varEnt["values"]]
			],
			freeVars];
		consList = consSet["values"];
		(*
		 The consList will contain cons,  so we don't want to subsitute into this
		 since it is going away. 
		*)
		Scan[
			If[ #["id"] =!= cons["id"],
				conNew = subs["apply", #];
				If[#["id"] =!= conNew["id"],
					updateRedirect[self, #, conNew];
					deleteLink[self, #];
					addLink[self, conNew]];
			]&,
			consList];
	]





(*
  Remove cons from the variableMap, as though it was never there.
  Visit all constraints that are linked from this constraint and remove
  their links to this constraint.  
  Remove this constraint from the variableMap by visiting its free variables.
*)
deleteLink[self_, cons_] :=
	Module[{consId = cons["id"], freeVarIds},
		freeVarIds = Keys[cons["free"]];
		Scan[ dropConstraintFromVariableMap[self, #, consId]&, freeVarIds];
		deleteRedirect[self, cons];
	]



(*
   add cons to the to the variableMap.
*)
addLink[ self_, cons_] :=
	Module[ {},
		updateVariableMap[self, cons];
	]

getOriginal[self_, cons_] :=
	self["reverseRedirectMap"]["lookup", cons["id"], cons]


setupMaps[ self_, consList_] :=
	Module[ {},
		Scan[updateVariableMap[self, #]&, consList];
	]


updateVariableMap[self_, cons_] :=
	Module[ {varMap = self["variableMap"], freeVarIds = Keys[cons["free"]], varData},
		Scan[ 
			(
			varData = varMap["lookup", #, Null];
			If[varData === Null,
				varData = CreateReference[<||>];
				varMap["associateTo", # -> varData]];
			varData["associateTo", cons["id"] -> cons];
			)&, freeVarIds];
	]

dropConstraintFromVariableMap[self_, varId_, consId_] :=
	Module[ {varData},
		varData = self["variableMap"]["lookup", varId, Null];
		If[varData === Null,
			ThrowException[{"Cannot find variable entry"}]
		];
		If[ varData["keyExistsQ", consId],
			varData["keyDropFrom", consId],
			ThrowException[{"Cannot find constraint entry"}]
		];
	]



(*
  code for working with lists of constraints to be solved.
*)

emptyQ[self_] :=
	self["equalConstraints"]["isEmpty"] && self["otherConstraints"]["isEmpty"]

insert[self_, cons_List] :=
	 Map[ insert[self, #]&, cons]

insert[self_, cons_] :=
	(
	insertConstraint[self, cons];
	addLink[self, cons];
	)

insertBack[self_, cons_List] :=
	 Map[ insertBack[self, #]&, cons]

insertBack[self_, cons_] :=
	(
	insertConstraintBack[self, cons];
	addLink[self, cons];
	)

insertConstraint[ self_, cons_] :=
	Module[{},
		If[ 
			EqualConstraintQ[cons],
				If[Length[cons["free"]] > 1,
					self["equalConstraints"]["pushBack", cons]
					,
					self["equalConstraints"]["pushFront", cons]],
				self["otherConstraints"]["pushFront", cons]]
	]


(*
  Insert the equal constraint to the back.  If "fixedError" is 1 this means it 
  has been looked at once and goes at the back of the equalConstraints.  If it is
  2 then this is the second time and it goes to the back of the other constraints.
*)
insertConstraintBack[ self_, cons_] :=
	Module[{},
		If[ 
			EqualConstraintQ[cons] && cons["getProperty", "fixedError", 0] < 2,
				self["equalConstraints"]["pushBack", cons],
				self["otherConstraints"]["pushBack", cons]]
	]

getFirst[ref_] :=
	Module[ {},
		ref["popFront"]
	]

nextToSolve[ self_] :=
	Module[{cons},
		cons = 
			Which[
				self["equalConstraints"]["length"] > 0,
					getFirst[self["equalConstraints"]],
				self["otherConstraints"]["length"] > 0,
					getFirst[self["otherConstraints"]],
				True,
					Null];
	If[ cons =!= Null,
		self["redirectMap"]["lookup", cons["id"], cons],
		Null]
	]



(*
  The redirect map exists so that we can modify the original constraints
  stored in the graph without having to update the constraint lists.
  See comments at top.
*)
updateRedirect[ self_, conOld_, conNew_] :=
	Module[ {orig},
		orig = self["reverseRedirectMap"]["lookup", conOld["id"], conOld];
		self["redirectMap"]["associateTo", orig["id"] -> conNew];
		self["reverseRedirectMap"]["associateTo", conNew["id"] -> orig];
	]

(*
  con is being removed so remove it from the redirectMap.
  This is done when deleteLink is called.  It's important that 
  when a con is removed we don't try to link to it with updateRedirect.
*)
deleteRedirect[self_, con_] :=
	Module[ {orig},
		orig = self["reverseRedirectMap"]["lookup", con["id"], Null];
		If[ orig =!= Null,
			self["reverseRedirectMap"]["keyDropFrom", con["id"]];
			self["reverseRedirectMap"]["keyDropFrom", orig["id"]]];	
	]

(*
  Return the unsolved constraints,  this is really for logging purposes.
*)
getUnsolved[self_] :=
	Module[{cons},
		cons = Join[self["equalConstraints"]["get"], self["otherConstraints"]["get"]];
		Map[ self["redirectMap"]["lookup", #["id"], #]&, cons]
	]


(*
  verify goes through the data in self and verifies that it is consistent.
*)

checkEntry[ cons_, varId_] :=
	Module[ {freeIds = Map[#["id"]&, cons["free"]]},
		MemberQ[freeIds, varId]
	]

verifyVariableEntry[self_, varId_] :=
	Module[ {varMap, varEnt},
		varMap = self["variableMap"];
		varEnt = varMap["lookup", varId, Null];
		If[ varEnt === Null,
			ThrowException[{"Cannot find variable entry"}]
		];
		Scan[
			(
			If[ !checkEntry[#, varId],
				ThrowException[{"Cannot find variable in constraint"}]
			];
			)&, varEnt["values"]];
	]

verify[ self_] :=
	Module[ {varMap},
		varMap = self["variableMap"];
		Scan[ verifyVariableEntry[self, #]&, varMap["keys"]];
	]




(*
  Various functions for visualizing the constraint graph.
*)

createBase[self_] :=
    Module[ {
    	conRefs = self["variableMap"]["values"], rules,
    	labels = CompileUtilities`Reference`CreateReference[<||>],
    	edges = CompileUtilities`Reference`CreateReference[<||>]},
        rules = 
        	Map[
          		Function[conRef,
          			Module[{cons = conRef["values"]},
           			Map[
           				Function[con,
             				labels["associateTo", con["id"] -> con["format"]];
             				Map[
             					Module[{id1 = con["id"], id2 = #["id"]},
             						If[ ! edges["lookup", {id2, id1}, False] && id1 =!= id2,
                     					edges["associateTo", {id1, id2} -> True];
                     					UndirectedEdge[id1, id2],
                     					{}
                 					]	
             					]&
             					, cons
             				]	
             			], 
             			cons]
          			]
             	], 
             	conRefs];
    	{Flatten[rules], VertexLabels -> Normal[labels["get"]]}
    ]

createGraph[self_] :=
    Module[ { base = createBase[self]},
    	Graph @@ base
    ]

createRules[self_] :=
    Module[ { base = createBase[self]},
    	First[base]
    ]


(**************************************************)

icon := Graphics[Text[
	Style["CON\nD",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[obj_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"ConstraintData",
		obj,
  		icon,
		{
			BoxForm`SummaryItem[{Pane["length: ", {90, Automatic}], obj["equalConstraints"]["length"]+obj["otherConstraints"]["length"]}]
  		},
  		{
   		},
  		fmt
  	]


toString[env_] := "ConstraintData[<>]"


format[ self_] :=
	Module[ {equals, others},
		equals = ToString[Map[ #["format"] &, self["equalConstraints"]["get"]]];
		others = ToString[Map[ #["format"] &, self["otherConstraints"]["get"]]];
 		StringJoin[
 			"ConstraintData[\n",
 			equals, "\n", others,
 			"\n]"
 		]
	]



End[]

EndPackage[]


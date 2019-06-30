
ClearAll[$SubjectName]
ClearAll[$Propositions]


named[tester_, actualValue_, name_] :=
	(
		Block[{$SubjectName = name},
 			tester[actualValue][##]
		]
	)&

fails[tester_, actualValue_] :=
	emitMessage[
		tester,
		"Failure",
		"`propositionName` was `verb` to always fail, but the result was `actualValue`.",
		<|
			"actualValue" -> actualValue,
			"expectedValue" -> False
		|>
	]
	
isTrue[tester_, actualValue_] :=
	If[TrueQ[actualValue],
		True,
		emitMessage[
			tester,
			"isTrue",
			"`propositionName` was `verb` to be `expectedValue`, but the result was `actualValue`.",
			<|
				"actualValue" -> actualValue,
				"expectedValue" -> True
			|>
		]
	]

isFalse[tester_, actualValue_] :=
	If[TrueQ[actualValue],
		emitMessage[
			tester,
			"isFalse",
			"`propositionName` was `verb` to be `expectedValue`, but the result was `actualValue`.",
			<|
				"actualValue" -> actualValue,
				"expectedValue" -> False
			|>
		],
		True
	]
	

isEqualTo[tester_, actualValue_, expectedValue_] :=
	If[actualValue =!= expectedValue,
		emitMessage[
			tester,
			"isEqualTo",
			"`propositionName` was `verb` to equal `expectedValue`, but the result was `actualValue`.",
			<|
				"actualValue" -> actualValue,
				"expectedValue" -> expectedValue
			|>
		],
		True
	]

isNotEqualTo[tester_, actualValue_, expectedValue_] :=
	If[actualValue === expectedValue,
		emitMessage[
			tester,
			"isNotEqualTo",
			"`propositionName` was `verb` to not equal `expectedValue`, but the result was `actualValue`.",
			<|
				"actualValue" -> actualValue,
				"expectedValue" -> expectedValue
			|>
		],
		True
	]

isA[tester_, actualValue_, expectedValue_] :=
	If[Head[actualValue] === expectedValue || TrueQ[ObjectInstanceQ[actualValue] && actualValue["isA", expectedValue]],
		True,
		emitMessage[
			tester,
			"isA",
			"`propositionName` was `verb` to be an instance of `expectedValue`, but the result was `actualValue`.",
			<|
				"actualValue" -> actualValue,
				"expectedValue" -> expectedValue
			|>
		]
	]

isNotA[tester_, actualValue_, expectedValue_] :=
	If[Head[actualValue] === expectedValue || TrueQ[ObjectInstanceQ[actualValue] && actualValue["isA", expectedValue]],
		emitMessage[
			tester,
			"isNotA",
			"`propositionName` was `verb` to not be an instance of `expectedValue`, but the result was `actualValue`.",
			<|
				"actualValue" -> actualValue,
				"expectedValue" -> expectedValue
			|>
		],
		True
	]


isNull[tester_, actualValue_] :=
	If[actualValue =!= Null,
		emitMessage[
			tester,
			"isNull",
			"`propositionName` was `verb` to be `expectedValue`, but the result was `actualValue`.",
			<|
				"actualValue" -> actualValue,
				"expectedValue" -> Null
			|>
		],
		True
	]
	
isNotNull[tester_, actualValue_] :=
	If[actualValue === Null,
		emitMessage[
			tester,
			"isNotNull",
			"`propositionName` was `verb` to not be `expectedValue`, but the result was `actualValue`.",
			<|
				"actualValue" -> actualValue,
				"expectedValue" -> Null
			|>
		],
		True
	]
	
hasLengthOf[tester_, actualValue_, expectedValue_] :=
	If[TrueQ[Length[actualValue] === expectedValue],
		True,
		emitMessage[
			tester,
			"hasLengthOf",
			"`propositionName` was `verb` have a length of `expectedValue`, but the length of `internalActualValue` was `actualValue`.",
			<|
				"internalActualValue" -> actualValue,
				"actualValue" -> Length[actualValue],
				"expectedValue" -> expectedValue
			|>
		]
	]
	
isGreaterThan[tester_, actualValue_, expectedValue_] :=
	If[TrueQ[actualValue > expectedValue],
		True,
		emitMessage[
			tester,
			"isGreaterThan",
			"`propositionName` was `verb` to be greater than `expectedValue`, but the result was `actualValue`.",
			<|
				"actualValue" -> actualValue,
				"expectedValue" -> expectedValue
			|>
		]
	]
	
isGreaterThanEqual[tester_, actualValue_, expectedValue_] :=
	If[TrueQ[actualValue >= expectedValue],
		True,
		emitMessage[
			tester,
			"isGreaterThanEqual",
			"`propositionName` was `verb` to be greater than or equal to `expectedValue`, but the result was `actualValue`.",
			<|
				"actualValue" -> actualValue,
				"expectedValue" -> expectedValue
			|>
		]
	]


isLessThan[tester_, actualValue_, expectedValue_] :=
	If[TrueQ[actualValue < expectedValue],
		True,
		emitMessage[
			tester,
			"isLessThan",
			"`propositionName` was `verb` to be less than `expectedValue`, but the result was `actualValue`.",
			<|
				"actualValue" -> actualValue,
				"expectedValue" -> expectedValue
			|>
		]
	]
	
isLessThanEqual[tester_, actualValue_, expectedValue_] :=
	If[TrueQ[actualValue <= expectedValue],
		True,
		emitMessage[
			tester,
			"isLessThanEqual",
			"`propositionName` was `verb` to be less than or equal to `expectedValue`, but the result was `actualValue`.",
			<|
				"actualValue" -> actualValue,
				"expectedValue" -> expectedValue
			|>
		]
	]
	
satisfies[tester_, actualValue_, check_] :=
	If[TrueQ[check[actualValue]],
		True,
		emitMessage[
			tester,
			"satisfies",
			"`propositionName` was `verb` to satisfy the `expectedValue` function, but the result was `actualValue` for `internalActualValue`.",
			<|
				"actualValue" -> check[actualValue],
				"internalActualValue" -> actualValue,
				"expectedValue" -> check
			|>
		]
	]
	
doesNotSatisfy[tester_, actualValue_, check_] :=
	If[TrueQ[check[actualValue]] === False,
		True,
		emitMessage[
			tester,
			"doesNotSatisfy",
			"`propositionName` was `verb` to not satisfy the `expectedValue` function, but the result was `actualValue` for `internalActualValue`.",
			<|
				"actualValue" -> check[actualValue],
				"internalActualValue" -> actualValue,
				"expectedValue" -> check
			|>
		]
	]
	
	
elementsSatisfy[tester_, actualValue_, check_] :=
	If[AllTrue[actualValue, check],
		True,
		emitMessage[
			tester,
			"elementsSatisfy",
			"`propositionName` elements were `verb` to satisfy the `expectedValue` function, but the result was `actualValue` for `internalActualValue`.",
			<|
				"actualValue" -> AllTrue[actualValue, check],
				"internalActualValue" -> actualValue,
				"expectedValue" -> check
			|>
		]
	]
	
elementsDoNotSatisfy[tester_, actualValue_, check_] :=
	If[AllTrue[actualValue, check],
		True,
		emitMessage[
			tester,
			"elementsDoNotSatisfy",
			"`propositionName` elements were `verb` to not satisfy the `expectedValue` function, but the result was `actualValue` for `internalActualValue`.",
			<|
				"actualValue" -> AllTrue[actualValue, check],
				"internalActualValue" -> actualValue,
				"expectedValue" -> check
			|>
		]
	]
	
satisfiesAllOf[tester_, actualValue_, checks_] :=
	If[ListQ[checks] && AllTrue[Through[checks[actualValue]], TrueQ],
		True,
		emitMessage[
			tester,
			"satisfiesAllOf",
			"`propositionName` was `verb` to satisfy all of the `expectedValue` functions, but the result was `actualValue` for `internalActualValue`.",
			<|
				"actualValue" -> ((# -> #[actualValue])& /@ checks),
				"internalActualValue" -> actualValue,
				"expectedValue" -> checks
			|>
		]
	]
	
satisfiesAnyOf[tester_, actualValue_, checks_] :=
	If[ListQ[checks] && AnyTrue[Through[checks[actualValue]], TrueQ],
		True,
		emitMessage[
			tester,
			"satisfiesAnyOf",
			"`propositionName` was `verb` to satisfy any of the `expectedValue` functions, but the result was `actualValue` for `internalActualValue`.",
			<|
				"actualValue" -> ((# -> #[actualValue])& /@ checks),
				"internalActualValue" -> actualValue,
				"expectedValue" -> checks
			|>
		]
	]
	
satisfiesNoneOf[tester_, actualValue_, checks_] :=
	If[ListQ[checks] && NoneTrue[Through[checks[actualValue]], TrueQ],
		True,
		emitMessage[
			tester,
			"satisfiesNoneOf",
			"`propositionName` was `verb` to satisfy none of the `expectedValue` functions, but the result was `actualValue` for `internalActualValue`.",
			<|
				"actualValue" -> ((# -> #[actualValue])& /@ checks),
				"internalActualValue" -> actualValue,
				"expectedValue" -> checks
			|>
		]
	]
	
	
elementsSatisfyAllOf[tester_, actualValue_, checks_] :=
	If[ListQ[actualValue] && ListQ[checks] && AllTrue[Flatten[Through[checks[#]]& /@ actualValue], TrueQ],
		True,
		emitMessage[
			tester,
			"elementsSatisfyAllOf",
			"`propositionName` was `verb` to satisfy all of the `expectedValue` functions, but the result was `actualValue` for `internalActualValue`.",
			<|
				"actualValue" -> ((# -> Through[checks[#]])& /@ actualValue),
				"internalActualValue" -> actualValue,
				"expectedValue" -> checks
			|>
		]
	]
	
elementsSatisfyAnyOf[tester_, actualValue_, checks_] :=
	If[ListQ[actualValue] && ListQ[checks] && AnyTrue[Flatten[Through[checks[#]]& /@ actualValue], TrueQ],
		True,
		emitMessage[
			tester,
			"elementsSatisfyAnyOf",
			"`propositionName` was `verb` to satisfy any of the `expectedValue` functions, but the result was `actualValue` for `internalActualValue`.",
			<|
				"actualValue" -> ((# -> Through[checks[#]])& /@ actualValue),
				"internalActualValue" -> actualValue,
				"expectedValue" -> checks
			|>
		]
	]
	
elementsSatisfyNoneOf[tester_, actualValue_, checks_] :=
	If[ListQ[actualValue] && ListQ[checks] && NoneTrue[Flatten[Through[checks[#]]& /@ actualValue], TrueQ],
		True,
		emitMessage[
			tester,
			"elementsSatisfyNoneOf",
			"`propositionName` was `verb` to satisfy none of the `expectedValue` functions, but the result was `actualValue` for `internalActualValue`.",
			<|
				"actualValue" -> ((# -> Through[checks[#]])& /@ actualValue),
				"internalActualValue" -> actualValue,
				"expectedValue" -> checks
			|>
		]
	]
	
isMemberOf[tester_, key_, assoc_?AssociationQ] :=
	If[KeyExistsQ[assoc, key],
		True,
		emitMessage[
			tester,
			"isMemberOf",
			"`propositionName` was `verb` to be a key in `expectedValue`, but the was `actualValue` was not a key in the association.",
			<|
				"actualValue" -> key,
				"expectedValue" -> Keys[assoc]
			|>
		]
	]
isMemberOf[tester_, elem_, lst_?ListQ] :=
	If[MemberQ[lst, elem],
		True,
		emitMessage[
			tester,
			"isMemberOf",
			"`propositionName` was `verb` to be an element in `expectedValue`, but the was `actualValue` was not an element of the list.",
			<|
				"actualValue" -> elem,
				"expectedValue" -> lst
			|>
		]
	]
	
isNotAMemberOf[tester_, key_, assoc_?AssociationQ] :=
	If[!KeyExistsQ[assoc, key],
		True,
		emitMessage[
			tester,
			"isNotAMemberOf",
			"`propositionName` was `verb` to not be a key in `expectedValue`, but the was `actualValue` was found as a key in the association.",
			<|
				"actualValue" -> key,
				"expectedValue" -> Keys[assoc]
			|>
		]
	]
isNotAMemberOf[tester_, elem_, lst_?ListQ] :=
	If[FreeQ[lst, elem],
		True,
		emitMessage[
			tester,
			"isNotAMemberOf",
			"`propositionName` was `verb` to not be an element in `expectedValue`, but the was `actualValue` was an element of the list.",
			<|
				"actualValue" -> elem,
				"expectedValue" -> lst
			|>
		]
	]

$Propositions = Map[curriedForm,
	<|
		"named" -> Function[{tester, actualValue, name}, named[tester, actualValue, name]],
		"fails" -> Function[{tester, actualValue, $noarg}, fails[tester, actualValue]],
		"isTrue" -> Function[{tester, actualValue, $noarg}, isTrue[tester, actualValue]],
		"isFalse" -> Function[{tester, actualValue, $noarg}, isFalse[tester, actualValue]],
		"isEqualTo" -> Function[{tester, actualValue, expectedValue}, isEqualTo[tester, actualValue, expectedValue]],
		"isNotEqualTo" -> Function[{tester, actualValue, expectedValue}, isNotEqualTo[tester, actualValue, expectedValue]],
		"isAList" -> Function[{tester, actualValue, $noarg}, isA[tester, actualValue, List]],
		"isAFunction" -> Function[{tester, actualValue, $noarg}, isA[tester, actualValue, Function]],
		"isAString" -> Function[{tester, actualValue, $noarg}, isA[tester, actualValue, String]],
		"isAnInteger" -> Function[{tester, actualValue, $noarg}, isA[tester, actualValue, Integer]],
		"isAnAssociation" -> Function[{tester, actualValue, $noarg}, isA[tester, actualValue, Association]],
		"isAReal" -> Function[{tester, actualValue, $noarg}, isA[tester, actualValue, Real]],
		"isA" -> Function[{tester, actualValue, expectedValue}, isA[tester, actualValue, expectedValue]],
		"isNotA" -> Function[{tester, actualValue, expectedValue}, isNotA[tester, actualValue, expectedValue]],
		"isAn" -> Function[{tester, actualValue, expectedValue}, isA[tester, actualValue, expectedValue]],
		"isNotAn" -> Function[{tester, actualValue, expectedValue}, isNotA[tester, actualValue, expectedValue]],
		"isAnInstanceOf" -> Function[{tester, actualValue, expectedValue}, isA[tester, actualValue, expectedValue]],
		"isNotAnInstanceOf" -> Function[{tester, actualValue, expectedValue}, isNotA[tester, actualValue, expectedValue]],
		"isNull" -> Function[{tester, actualValue, $noarg}, isNull[tester, actualValue]],
		"isNotNull" -> Function[{tester, actualValue, $noarg}, isNotNull[tester, actualValue]],
		"hasLengthOf" -> Function[{tester, actualValue, expectedValue}, hasLengthOf[tester, actualValue, expectedValue]],
		"isGreaterThan" -> Function[{tester, actualValue, expectedValue}, isGreaterThan[tester, actualValue, expectedValue]],
		"isGreaterThanEqual" -> Function[{tester, actualValue, expectedValue}, isGreaterThanEqual[tester, actualValue, expectedValue]],
		"isLessThan" -> Function[{tester, actualValue, expectedValue}, isLessThan[tester, actualValue, expectedValue]],
		"isLessThanEqual" -> Function[{tester, actualValue, expectedValue}, isLessThanEqual[tester, actualValue, expectedValue]],
		"satisfies" -> Function[{tester, actualValue, expectedValue}, satisfies[tester, actualValue, expectedValue]],
		"doesNotSatisfy" -> Function[{tester, actualValue, expectedValue}, doesNotSatisfy[tester, actualValue, expectedValue]],
		"elementsSatisfy" -> Function[{tester, actualValue, expectedValue}, elementsSatisfy[tester, actualValue, expectedValue]],
		"elementsDoNotSatisfy" -> Function[{tester, actualValue, expectedValue}, elementsDoNotSatisfy[tester, actualValue, expectedValue]],
		"satisfiesAllOf" -> Function[{tester, actualValue, expectedValue}, satisfiesAllOf[tester, actualValue, expectedValue]],
		"satisfiesAnyOf" -> Function[{tester, actualValue, expectedValue}, satisfiesAnyOf[tester, actualValue, expectedValue]],
		"satisfiesNoneOf" -> Function[{tester, actualValue, expectedValue}, satisfiesNoneOf[tester, actualValue, expectedValue]],
		"elementsSatisfyAllOf" -> Function[{tester, actualValue, expectedValue}, elementsSatisfyAllOf[tester, actualValue, expectedValue]],
		"elementsSatisfyAnyOf" -> Function[{tester, actualValue, expectedValue}, elementsSatisfyAnyOf[tester, actualValue, expectedValue]],
		"elementsSatisfyNoneOf" -> Function[{tester, actualValue, expectedValue}, elementsSatisfyNoneOf[tester, actualValue, expectedValue]],
		"isMemberOf" -> Function[{tester, actualValue, expectedValue}, isMemberOf[tester, actualValue, expectedValue]],
		"isNotAMemberOf" -> Function[{tester, actualValue, expectedValue}, isNotAMemberOf[tester, actualValue, expectedValue]]
		(*
		"contains" -> Function[{tester, actualValue, expectedValue}, contains[tester, actualValue, expectedValue]],
		"startsWith" -> Function[{tester, actualValue, expectedValue}, startsWith[tester, actualValue, expectedValue]],
		"endsWith" -> Function[{tester, actualValue, expectedValue}, endsWith[tester, actualValue, expectedValue]],
		"hasLength" -> Function[{tester, actualValue, expectedValue}, hasLength[tester, actualValue, expectedValue]],
		"matches" -> Function[{tester, actualValue, expectedValue}, matches[tester, actualValue, expectedValue]] *)
	|>
]

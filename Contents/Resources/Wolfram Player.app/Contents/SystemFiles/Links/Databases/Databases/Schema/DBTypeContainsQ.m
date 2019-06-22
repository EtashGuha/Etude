Package["Databases`Schema`"]


PackageImport["Databases`"]
PackageImport["Databases`Common`"]

PackageExport["DBTypeContainsQ"]

DBTypeContainsQ[x: Except[_DBType?DBTypeQ], y_] := With[
	{t = DBType[x]},
	DBTypeQ[t] && DBTypeContainsQ[t, y]
]

DBTypeContainsQ[x_, y: Except[_DBType?DBTypeQ]] := With[
	{t = DBType[y]},
	DBTypeQ[t] && DBTypeContainsQ[x, t]
]

DBTypeContainsQ[
	DBType[t: "Query" | "DatabaseModelInstance"],
	DBType[t: "Query" | "DatabaseModelInstance", a_?AssociationQ]
] := True

DBTypeContainsQ[
	DBType[t: "Query" | "DatabaseModelInstance", a_?AssociationQ],
	DBType[t: "Query" | "DatabaseModelInstance", b_?AssociationQ]
] := a === b


DBTypeContainsQ[
	DBType[t: "Time" | "DateTime", a: _?AssociationQ: <||>],
	DBType[t: "Time" | "DateTime", b: _?AssociationQ: <||>]
] := MatchQ[
	Lookup[b, "TimeZone"],
	Lookup[a, "TimeZone", True | False | _Missing]
]

(*TODO this below is too simplistic*)
DBTypeContainsQ[DBType[x_?StringQ, ___], DBType[y_?StringQ, ___]] := x === y

DBTypeContainsQ[x_DBType, y: DBType[_DBTypeUnion, ___]] := AllTrue[
	y["Constituents"],
	DBTypeContainsQ[x, #]&
]

DBTypeContainsQ[
	x: DBType[_DBTypeUnion, ___],
	y_DBType
] := AnyTrue[
	x["Constituents"],
	DBTypeContainsQ[#, y]&
]

DBTypeContainsQ[
	x: DBType[_DBTypeIntersection, ___],
	y_DBType
] := AllTrue[
	x["Constituents"],
	DBTypeContainsQ[#, y]&
]

DBTypeContainsQ[x_DBType, y: DBType[_DBTypeIntersection, ___]] := AnyTrue[
	y["Constituents"],
	DBTypeContainsQ[x, #]&
]

DBTypeContainsQ[
	dbt1: DBType[RepeatingElement[t1_DBType?DBTypeQ, ___], ___],
	dbt2: DBType[RepeatingElement[t2_DBType?DBTypeQ, ___], ___]
] := And[
	DBTypeContainsQ[t1, t2],
	checkDimensions[dbt1, dbt2]
]

DBTypeContainsQ[
	dbt1: DBType[CompoundElement[l1: _List | _Association?AssociationQ], ___],
	dbt2: DBType[CompoundElement[l2: _List | _Association?AssociationQ], ___]
] := And[
	Or[
		ListQ[l1] && ListQ[l2] && Length[l1] === Length[l2],
		AssociationQ[l1] && AssociationQ[l2] && Keys[l1] === Keys[l2]
	],
	AllTrue[
		Transpose[{l1, l2}, AllowedHeads -> All],
		Apply[DBTypeContainsQ]
	]
]

DBTypeContainsQ[
	dbt1: DBType[RepeatingElement[t1_DBType?DBTypeQ, ___], ___],
	dbt2: DBType[CompoundElement[l_List], ___]
] := And[
	AllTrue[l, DBTypeContainsQ[t1, #]&],
	checkDimensions[dbt1, dbt2]
]

DBTypeContainsQ[
	dbt1: DBType[CompoundElement[l_List], ___],
	dbt2: DBType[RepeatingElement[t2_DBType?DBTypeQ, ___], ___]
] := And[
	AllTrue[l, DBTypeContainsQ[#, t2]&],
	checkDimensions[dbt1, dbt2]
]

(*machinery for dealing with matrix types*)

checkDimensions[outer_, inner_] := With[
	{outerdims = outer["Dimensions"], innerdims = inner["Dimensions"]},
	And[
		Length[outerdims] <= Length[innerdims],
		AllTrue[
			Transpose[{Take[outerdims, Length[innerdims]], innerdims}],
			Apply[IntervalMemberQ]
		]
	]
]

isSquare[dbt: DBType[_SquareRepeatingElement, ___], ___] := True
isSquare[dbt_DBType?DBTypeQ] := MatchQ[
	dbt["Dimensions"],
	Alternatives[
		{Interval[{x_?IntegerQ, x_}], Interval[{x_, x_}], ___},
		{x_?IntegerQ, x_, ___}
	]
]

getType[t_DBType?DBTypeQ] := iGetType[t, 0]

iGetType[
	DBType[(SquareRepeatingElement | RectangularRepeatingElement)[t1_DBType?DBTypeQ, ___], ___],
	0
] := t1
iGetType[
	DBType[(SquareRepeatingElement | RectangularRepeatingElement)[t1_DBType?DBTypeQ, ___], ___],
	1
] := DBType[RepeatingElement[t1]]
iGetType[dbt1: DBType[CompoundElement[_List], ___], depth_] /;
	depth < 2 := DBType[
	Apply[
		DBTypeUnion,
		iGetType[#, depth + 1]& /@ dbt1["Constituents"]
	]
]
iGetType[DBType[RepeatingElement[t1_DBType?DBTypeQ, ___], ___], depth_] /;
	depth < 2 := iGetType[t1, depth + 1]
iGetType[t_DBType?DBTypeQ, depth_] /; depth === 2 := t

DBTypeContainsQ[
	dbt1 : DBType[(SquareRepeatingElement | RectangularRepeatingElement)[
		t1_DBType?DBTypeQ,
		___
	], ___],
	dbt2_DBType?DBTypeQ
] := And[
	DBTypeContainsQ[getType[dbt1], getType[dbt2]],
	Head[First[dbt1]] =!= SquareRepeatingElement || isSquare[dbt2],
	checkDimensions[dbt1, dbt2]
]

DBTypeContainsQ[
	dbt1_DBType?DBTypeQ,
	dbt2: DBType[(SquareRepeatingElement | RectangularRepeatingElement)[t1_DBType?DBTypeQ, ___], ___]
] := And[
	DBTypeContainsQ[getType[dbt1], getType[dbt2]],
	checkDimensions[dbt1, dbt2]
]



DBTypeContainsQ[
	_,
	_
] := False

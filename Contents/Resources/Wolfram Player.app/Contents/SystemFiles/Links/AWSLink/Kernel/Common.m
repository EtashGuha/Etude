(*******************************************************************************

Common: global variables plus common utility functions

*******************************************************************************)

Package["AWSLink`"]

PackageImport["JLink`"]
PackageImport["GeneralUtilities`"]
(****** Global Variables ******)


(*----------------------------------------------------------------------------*)
PackageExport["AWSCollectionRead"]
AWSCollectionRead[head_[e_]] := Replace[Read[e], IteratorExhausted -> Null];

PackageExport["AWSCollectionReadList"]
AWSCollectionReadList[head_[e_]] := Replace[ReadList[e], IteratorExhausted -> Null];
AWSCollectionReadList[head_[e_], num_] := Replace[ReadList[e, num], IteratorExhausted -> Null];

PackageScope["classNameFromInstance"]
classNameFromInstance[x_] := First[JLink`Package`classFromInstance[x], $Failed];

PackageScope["toJavaDate"]
toJavaDate[date_DateObject] := JavaNew["java.util.Date", (UnixTime[date]*1000)];

PackageScope["fromJavaDate"]
fromJavaDate[date_/;classNameFromInstance[date]==="java.util.Date"] := FromUnixTime[date@getTime[]/1000];

(*we will us stream to extract values and avoid large java calls*)
PackageScope["JavaPrimitiveOf"]
JavaPrimitiveOf[primitiveClassName_] :=
	JLinkClassLoader`classFromName[primitiveClassName]@getField["TYPE"]@get[Null];

PackageScope["MapJavaMethod"]
Options[MapJavaMethod] = {"ToArray" -> True};
MapJavaMethod[objList_, objFullClass_, methName_, methodOutFullClass_, OptionsPattern[]]:=
	JavaBlock@Module[
		{
			lookup, methodType, methodHandle,
			methodProxy, objArray, objStrm, resultsList
		},
		LoadJavaClass["java.util.stream.Stream"];
		LoadJavaClass["java.util.stream.Collectors"];

		LoadJavaClass["com.wolfram.jlink.JLinkClassLoader"];
		LoadJavaClass["java.lang.invoke.MethodHandles"];
		LoadJavaClass["java.lang.invoke.MethodType"];
		LoadJavaClass["java.lang.invoke.MethodHandleProxies"];
		
		lookup = MethodHandles`lookup[];
		methodType = 
			Switch[methodOutFullClass,
				_String,
					MethodType`methodType[JLinkClassLoader`classFromName[methodOutFullClass]],
				_ (*assume java primitive case: int, long, ect...*),
					MethodType`methodType[methodOutFullClass]
			];
		methodHandle = safeLibraryInvoke[lookup@findVirtual[JLinkClassLoader`classFromName[objFullClass], methName, methodType]];
		methodProxy = safeLibraryInvoke[MethodHandleProxies`asInterfaceInstance[JLinkClassLoader`classFromName["java.util.function.Function"], methodHandle]];
		
		objArray = safeLibraryInvoke[ReturnAsJavaObject[objList@toArray[]]];
		objStrm = safeLibraryInvoke[Stream`of[objArray]];
		resultsList = safeLibraryInvoke[objStrm@map[methodProxy]@collect[Collectors`toList[]]];
		If[ OptionValue["ToArray"],
			resultsList@toArray[],
			resultsList
		]
	];
PackageScope["MapJavaMethods"]
MapJavaMethods[objList_, FullClassSequence_, methNameSequence_] :=
	Module[
		{},
		If[ Length@methNameSequence > 1,
			MapJavaMethods[MapJavaMethod[objList, FullClassSequence[[1]], methNameSequence[[1]], FullClassSequence[[2]], "ToArray" -> False], Rest@FullClassSequence, Rest@methNameSequence],
			MapJavaMethod[objList, FullClassSequence[[1]], methNameSequence[[1]], FullClassSequence[[2]]]
		]
	];

PackageScope["AWSDryRun"]
AWSDryRun[jclient_, req_] :=
	JavaBlock @ Module[
		{res},
		res = safeLibraryInvoke @ jclient@dryRun[req];
		Failure[
			If[res@isSuccessful[],"DryRunOperation", "UnauthorizedOperation"],
			<|"Message" -> res@getMessage[]|>
		]
	];

PackageScope["getTemplateKeys"]
getTemplateKeys[Template_] :=
	Keys @ Template;

PackageScope["getPropertiesFromTemplate"]
getPropertiesFromTemplate[Template_, javaobj_] := Module[
	{},
	Switch[Template,
		_Association,
			AssociationThread[
				Keys@Template,
				Map[igetProperties[#, javaobj] &, Values@Template]
			],	
		_,
			ThrowFailure[General::"awswrgtemplate", classNameFromInstance[javaobj], If[NameQ[ToString[Template]], SymbolName[Template], Template]]
	]
];

igetProperties[value_, javaobj_] := JavaBlock[
	Module[
		{internalJavaobj, internalValue},
		
		Switch[
			value,
			(* something more to be extracted *)
			_Missing,
				internalJavaobj = ReleaseHold[Hold[javaobj][ToExpression[First@value]]];
				internalValue = Last@value;
				
				(* in case the propertie was not set *)
				If[ internalJavaobj === Null,
					Return[postProcess[Null]]
				];
				
				Switch[
					internalValue,
					_Association,
						(* one of our templates *)
						getPropertiesFromTemplate[internalValue, internalJavaobj],
					_List,
						(* used for java Collections constructs *)
						Map[
							Switch[ First@internalValue,
								_Association,
									getPropertiesFromTemplate[First@internalValue, #] &,
								_,
									igetProperties[First@internalValue, #] &
							]
							,
							Table[internalJavaobj@get[i], {i, 0, internalJavaobj@size[]-1}]
						],
					_Rule,
						(*used for java Maps constructs *)
						AssociationThread[igetProperties[First@internalValue, internalJavaobj] , igetProperties[Last@internalValue, internalJavaobj]],
					_,
						(* oupsy ! should not have happened *)
						Echo[value];
						Echo[internalValue];
						Null
				],
			(*post-processing of Mathematica Object*)
			_Rule,
				(Last@value) @
					Switch[ First@value,
					_Association,
						getPropertiesFromTemplate[First@value, #],
					_,
						igetProperties[First@value, #]
					]& @ javaobj,
			(* grab the property *)
			_String,
				postProcess@ReleaseHold[Hold[javaobj][ToExpression[value]]],
			_,
				(* oupsy ! should not have happened *)
				Echo[value];
				Null
	   ]
	]
];

postProcess[x_String] := x;
postProcess[Null] := Missing["Null"];
postProcess[x_/;classNameFromInstance[x]==="java.util.Date"] := Module[{date}, date = fromJavaDate[x]; ReleaseJavaObject[x]; date];
postProcess[x_/;classNameFromInstance[x]==="com.amazonaws.internal.SdkInternalList"] := x@toArray[]; (*Block[{i},Table[x@get[i],{i,0,x@size[]-1}]];*)
postProcess[x_] := x;

PackageScope["getCurrentProperties"]
getCurrentProperties[service_String, object_, DescribeFun_, availProps:{__String}, "Properties"] :=
	availProps;
getCurrentProperties[service_String, object_, DescribeFun_, availProps:{__String}, prop_String] :=
	First@getCurrentProperties[service, object, (True)&, DescribeFun, availProps, {prop}]
getCurrentProperties[service_String, object_, DescribeFun_, availProps:{__String}, props:{__String}] :=
	getCurrentProperties[service, object, (True)&, DescribeFun, availProps, props];
getCurrentProperties[service_String, object_, validTest_, DescribeFun_, availProps:{__String}, "Properties"] :=
	availProps;
getCurrentProperties[service_String, object_, validTest_, DescribeFun_, availProps:{__String}, prop:_String] :=
	First@getCurrentProperties[service, object, validTest, DescribeFun, availProps, {prop}]
getCurrentProperties[service_String, object_, validTest_, DescribeFun_, availProps:{__String}, props:{__String}] :=
	Module[
		{queryProps, failedProps, valueRules, failureRules},
		If[validTest[object],
			queryProps = Intersection[props, availProps];
			failedProps = Complement[props, availProps];
			valueRules = 
				If[queryProps =!= {},
					Thread[availProps -> Lookup[DescribeFun[object], availProps]],
					{}
				];
			failureRules =
				Thread[
					failedProps ->
						(Failure["Missing",
							<|
							"MessageTemplate" :> General::"awsnoprop",
							"MessageParameters" -> {#, Head@object, service}
							|>
						]& /@ failedProps)
				];
				
			Replace[Replace[props, valueRules, {1}], failureRules, {1}]
			,
			ThrowFailure["awsunvalid", Head@object, service]
		]
	];


PackageScope["CheckInput"]
CheckInput::"ill" = "Ill formated input check - throwing!";
CheckInput[testsActions:{___Rule}, input_, NoneOfTheAbove_Hold] := Module[
	{},
	If[testsActions === {},
		ReleaseHold[NoneOfTheAbove],
		Switch[ testsActions[[1, 1]],
			_Function,
				If[ testsActions[[1, 1]][input],
					Switch[ testsActions[[1, 2]],
						_Function,
							testsActions[[1, 2]][input],
						_,
							testsActions[[1, 2]]
					],
					CheckInput[Rest[testsActions], input, NoneOfTheAbove]
				],
			_,
				If[ MatchQ[input, testsActions[[1, 1]]],
					Switch[ testsActions[[1, 2]],
						_Function,
							testsActions[[1, 2]][input],
						Hold[_],
							ReleaseHold[testsActions[[1, 2]]],
						_,
							testsActions[[1, 2]]
					],
					CheckInput[Rest[testsActions], input, NoneOfTheAbove]
				]
		]
	]
];
CheckInput[testsActions:{___Rule}, input_, name_String] :=
	CheckInput[
		testsActions,
		input,
		Hold[ThrowFailure["awswrongopt", name, input]]
	];
CheckInput[wrg__, NoneOfTheAbove_Hold] := Module[{}, Message[CheckInput::"ill"]; ReleaseHold[NoneOfTheAbove]];
CheckInput[___] := Throw[Failure["wrong input check", <|"Message"->"Unexpected input - test failed"|>]];
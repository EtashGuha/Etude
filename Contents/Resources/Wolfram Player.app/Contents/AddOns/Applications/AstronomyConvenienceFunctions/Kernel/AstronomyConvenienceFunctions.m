(* ::Package:: *)

System`Private`NewContextPath[{"System`","ExternalService`"}];

(Unprotect[#];Clear[#];)&/@{
	System`Sunset,
	System`Sunrise,
	System`SunPosition,
	System`MoonPosition,
	System`MoonPhase,
	System`SolarEclipse,
	System`LunarEclipse,
	System`SiderealTime,
	System`CelestialSystem,
	System`TimeDirection,
	System`EclipseType,
	System`DaylightQ,
	System`AltitudeMethod
};

System`ResamplingMethod;

Unprotect["System`AstronomyConvenienceFunctions`Private`*"]
ClearAll["System`AstronomyConvenienceFunctions`Private`*"]

Begin["AstronomyConvenienceFunctions`Private`"]

Attributes[APICompute] = {HoldAll};
APICompute[function_,type_, args_] := Internal`MWACompute[type,args,"ContextPath" -> {"Internal`MWASymbols`", "System`", 
  "AstronomyConvenienceFunctions`Private`"},"MessageHead"->function]


(* begin domain specific functionality *)

angleUnitQ[q_?QuantityQ] := SameQ[UnitDimensions[q], {{"AngleUnit", 1}}]
angleUnitQ[___] := False 

radiansToDegrees[r_] := r/Degree
radiansToHours[r_] := r/Degree/15
degreesToRadians[d_] := d*Degree

Clear[validLocationQ];

validLocationQ[loc:HoldPattern[_System`Entity], callingfunc_] := Module[{temp},
	temp=Quiet[!FreeQ[EntityValue[loc, {"Polygon", "Position"}], _GeoPosition | {_GeoPosition ..} | _Polygon | {_Polygon ..}], {Entity::etype, EntityProperty::qname, EntityProperty::pname}];
   If[temp, True, False]];
validLocationQ[loc:HoldPattern[_System`EntityClass], callingfunc_] := Module[{temp},
	temp=Quiet[!FreeQ[EntityValue[loc, {"Polygon", "Position"}], _GeoPosition | {_GeoPosition ..} | _Polygon | {_Polygon ..}], {Entity::etype, EntityProperty::qname, EntityProperty::pname}];
   If[temp, True, False]];
validLocationQ[HoldPattern[_System`GeoVariant], callingfunc_] := True;
(*validLocationQ[HoldPattern[_System`GeoGroup], callingfunc_] := True;*)
validLocationQ[HoldPattern[_System`Polygon], callingfunc_] := True;
validLocationQ[HoldPattern[_System`Line], callingfunc_] := True;
validLocationQ[HoldPattern[_System`Point], callingfunc_] := True;
validLocationQ[HoldPattern[_System`Rectangle], callingfunc_] := True;
validLocationQ[HoldPattern[_System`GeoDisk], callingfunc_] := True;
validLocationQ[HoldPattern[_System`GeoCircle], callingfunc_] := True;
validLocationQ[loc:HoldPattern[_System`GeoPosition], callingfunc_] := With[{loclen=Length[List@@loc]},
	If[loclen==1, True, If[(loclen>1)&&(!MatchQ[Last[List@@loc], System`Entity["Planet", "Earth"]]), False, True]]];
validLocationQ[HoldPattern[_System`GeoPath], callingfunc_] := True;
validLocationQ[HoldPattern[_System`GeoMarker], callingfunc_] := True;
validLocationQ[HoldPattern[_System`BezierCurve], callingfunc_] := True;
validLocationQ[HoldPattern[_System`BSplineCurve], callingfunc_] := True;
validLocationQ[HoldPattern[_System`FilledCurve], callingfunc_] := True;
validLocationQ[HoldPattern[_System`GraphicsComplex], callingfunc_] := True;
(*
validLocationQ[HoldPattern[_System`Arrow], callingfunc_] := True;
validLocationQ[HoldPattern[_System`Circle], callingfunc_] := True;
validLocationQ[HoldPattern[_System`Disk], callingfunc_] := True;
validLocationQ[HoldPattern[_System`JoinedCurve], callingfunc_] := True;
*)
(* validation of numbers is tricky since sometimes the lat/long are reversed and this is valid so *)
(* we only check that the numbers are between -360 and 360 *)
validLocationQ[{lat_?NumericQ, long_?NumericQ}, callingfunc_]/;((-360<=lat<=360)&&(-360<=long<=360)) := True;
validLocationQ[{lat_?NumericQ, long_?NumericQ, elm_?NumericQ}, callingfunc_]/;((-360<=lat<=360)&&(-360<=long<=360)) := True;
validLocationQ[{lat_?angleUnitQ, long_?angleUnitQ}, callingfunc_]/;((-90<=QuantityMagnitude[UnitConvert[lat, "AngularDegrees"]]<=90)&&(-360<=QuantityMagnitude[UnitConvert[long, "AngularDegrees"]]<=360)) := True;
validLocationQ[loc_, callingfunc_] := False;

validDateQ[{_?NumericQ..}, callingfunc_]:=True;
validDateQ[_DateObject, callingfunc_]:=True;
validDateQ[{_DateObject, _DateObject, _Quantity}, callingfunc_]:=True;
validDateQ[{_DateObject, _DateObject, {_NumericQ, _String}}, callingfunc_]:=True;
validDateQ[date_String, callingfunc_]:=MatchQ[Quiet[Check[DateList[date],False],DateList::str],{_,_,_,_,_,_}]
validDateStringQ[date_String, callingfunc_]:=MatchQ[Quiet[Check[DateList[date],False],DateList::str],{_,_,_,_,_,_}];
validDateQ[_, callingfunc_]:=False;

validTimeZoneQ[_?Internal`RealValuedNumericQ] := True;
validTimeZoneQ[s_String?System`Utilities`TimeZoneStringQ] := True;
(*validTimeZoneQ[HoldPattern[Entity["TimeZone", s_String?System`Utilities`TimeZoneStringQ]]] := True;*)
(*validTImeZoneQ[HoldPattern[e_Entity]] := checkIfEntityHasTimeZone[e];*)
validTimeZoneQ[_] := False;

validIntervalValue[Automatic, callingfunc_]:=True;
validIntervalValue[Interval, callingfunc_]:=True;
validIntervalValue[Mean, callingfunc_]:=True;
validIntervalValue[Min, callingfunc_]:=True;
validIntervalValue[Max, callingfunc_]:=True;
validIntervalValue[StandardDeviation, callingfunc_]:=True;
validIntervalValue[_, callingfunc_]:=False;

validTimeDateFlag[Automatic, callingfunc_]:=True
validTimeDateFlag[System`TimeObject, callingfunc_]:=True
validTimeDateFlag[System`DateObject, callingfunc_]:=True
validTimeDateFlag[_, callingfunc_]:=False

validTimeDirection[Automatic, callingfunc_]:=True
validTimeDirection[1, callingfunc_]:=True
validTimeDirection[-1, callingfunc_]:=True
validTimeDirection[_, callingfunc_]:=False

validLocationQualifier["Everywhere", callingfunc_]:=True
validLocationQualifier["Somewhere", callingfunc_]:=True
validLocationQualifier["Nowhere", callingfunc_]:=True
validLocationQualifier[_, callingfunc_]:=False

validAssociationQ[assoc_Association, callingfunc_] /; AssociationQ[assoc] := With[{tmp=!FreeQ[Keys[assoc], "Location"|"Date"]},
	If[tmp, 
		If[TrueQ[validLocationQ[assoc["Location"], callingfunc]||MatchQ[assoc["Location"], Missing["KeyAbsent", ___]]]||TrueQ[validDateQ[assoc["Date"], callingfunc]||MatchQ[assoc["Date"], Missing["KeyAbsent", ___]]], True, False], False]
]
validAssociationQ[_, callingfunc_] := False;

validIncrementQ[_?NumericQ] := True
validIncrementQ[q_?QuantityQ] := 
 And[QuantityUnits`Private`NumericQuantityQ[q], 
  UnitDimensions[q] === {{"TimeUnit", 1}}]
validIncrementQ[{q_?NumericQ, u_String}] := 
 And[QuantityUnits`Private`NumericQuantityQ[Quantity[q, u]], 
  UnitDimensions[Quantity[q, u]] === {{"TimeUnit", 1}}]
validIncrementQ[__] := False

(*
validDateQ[date_List]:=MatchQ[Quiet[Check[DateList[date],False],DateList::arg],{_,_,_,_,_,_}]
validDateQ[date_?NumericQ]:=MatchQ[Quiet[Check[DateList[date],False],DateList::arg],{_,_,_,_,_,_}]
validDateQ[_]=False;*)

resolveGeoPosition[x_GeoPosition]:= GIS`GeoReverse[x]

preProcessCoordinates[{lata_?angleUnitQ, longa_?angleUnitQ}]:={longa, lata}
preProcessCoordinates[loc_System`Polygon]:=Module[{poly},
	poly = loc /. x_GeoPosition :> resolveGeoPosition[x];(*System`GeoGraphicsDump`iGeoEvaluation[loc];*)
	If[MatchQ[poly, _System`Polygon], 
		System`Polygon @@ (Graphics`Mesh`SimplifyPolyLine[poly, VerticesGoal -> 20]),
		{}]
	]
preProcessCoordinates[loc_System`GeoVariant]:=Module[{poly},
	poly = EntityValue[loc, "Polygon"];(*System`GeoGraphicsDump`iGeoEvaluation[loc];*)
	Which[
		MatchQ[poly, _System`Polygon | {_System`Polygon..}], 
		System`Polygon @@ (Graphics`Mesh`SimplifyPolyLine[poly/. x_GeoPosition :> resolveGeoPosition[x], VerticesGoal -> 20]),
		MatchQ[poly, _System`GeoPosition], poly/. x_GeoPosition :> resolveGeoPosition[x],
		True, {}]
	]
preProcessCoordinates[loc_System`Rectangle]:=Module[{poly},
	poly = (loc/. x_GeoPosition :> resolveGeoPosition[x])/.Rectangle[{xmin_, ymin_}, {xmax_, ymax_}] :> Polygon[{{xmin, ymin}, {xmax, ymin}, {xmax, ymax}, {xmin, ymax}}];(*System`GeoGraphicsDump`iGeoEvaluation[loc];*)
	If[MatchQ[poly, _System`Polygon], 
		System`Polygon @@ (Graphics`Mesh`SimplifyPolyLine[poly, VerticesGoal -> 20]),
		{}]
	]
preProcessCoordinates[loc_System`FilledCurve]:=Module[{poly},
	poly = With[{tmp=Cases[loc, _Line|_BezierCurve|_BSplineCurve, Infinity]}, If[Length[tmp]>0, tmp[[1]]/.x_GeoPosition :> resolveGeoPosition[x], $Failed]];
	Switch[poly, 
		_System`Line, System`Polygon @@ (Graphics`Mesh`SimplifyPolyLine[Polygon@@poly, VerticesGoal -> 20]),
		_System`BezierCurve, preProcessCoordinates[poly],
		_System`BSplineCurve, preProcessCoordinates[poly],
		_, {}]
	]
preProcessCoordinates[loc_System`BezierCurve]:=Module[{line},
	line = With[{points = 36}, 
		Table[
			Evaluate[(BezierFunction @@ (loc/.x_GeoPosition :> resolveGeoPosition[x]))[s]], 
				{s, 0, 1, 1/points}]];
	Line[line]
	]
preProcessCoordinates[loc_System`BSplineCurve]:=Module[{line},
	line = With[{points = 36}, 
		Table[Evaluate[(BSplineFunction @@ (loc/.x_GeoPosition :> resolveGeoPosition[x]))[s]], 
			{s, 0, 1, 1/points}]];
	Line[line]
	]
preProcessCoordinates[loc_System`Line]:=Module[{line},
	line = loc/. x_GeoPosition :> resolveGeoPosition[x];(*System`GeoGraphicsDump`iGeoEvaluation[loc];*)
	line
	]
preProcessCoordinates[loc_System`GeoDisk]:=Module[{poly},
	poly = GeoGraphics`GeoEvaluate[loc]/.{System`GeoGraphicsDump`reversedGeoPosition[x_]:>x};
	If[MatchQ[poly, _System`Polygon], 
		System`Polygon @@ (Graphics`Mesh`SimplifyPolyLine[poly, VerticesGoal -> 20]),
		{}]
	]
preProcessCoordinates[loc_System`GeoCircle]:=Module[{poly},
	poly = GeoGraphics`GeoEvaluate[loc]/.{System`GeoGraphicsDump`reversedGeoPosition[x_]:>x};
	If[MatchQ[poly, _System`Line], 
		System`Line @@ (Graphics`Mesh`SimplifyPolyLine[Polygon@@poly, VerticesGoal -> 20]),
		{}]
	]
preProcessCoordinates[loc_System`Arrow]:=Module[{line},
	line = loc/. x_GeoPosition :> resolveGeoPosition[x];(*System`GeoGraphicsDump`iGeoEvaluation[loc];*)
	System`Line @@ line
	]
preProcessCoordinates[loc_System`GeoPath]:=Module[{line},
	line = GeoGraphics`GeoEvaluate[loc]/.{System`GeoGraphicsDump`reversedGeoPosition[x_]:>x};
	System`Line @@ line
	]
preProcessCoordinates[loc_System`GeoMarker]:=loc/. x_GeoPosition :> resolveGeoPosition[x]
preProcessCoordinates[loc_System`GeoPosition]:=resolveGeoPosition[loc]
preProcessCoordinates[loc_System`GraphicsComplex]:=Module[{poly},
	poly = Normal[loc]/. x_GeoPosition :> resolveGeoPosition[x];(*System`GeoGraphicsDump`iGeoEvaluation[loc];*)
	Flatten[preProcessCoordinates/@poly]/.{x_}:>x
	]
preProcessCoordinates[{lat_?NumericQ, long_?NumericQ}]:={long, lat}
preProcessCoordinates[{lat_?NumericQ, long_?NumericQ, elm_?NumericQ}]:={long, lat, elm}
preProcessCoordinates[loc_]:=loc/. {x_GeoPosition :> resolveGeoPosition[x]}

postProcessDateResults[apires_, tz_, iv_, td_] := Which[
	Head[apires] === System`Quantity, apires,
	apires === {}, $Failed,
	td===DateObject,
	Which[
		Head[apires] === Interval, 
		Interval[{System`DateObject[apires[[1, 1]], TimeZone -> tz], System`DateObject[apires[[1, 2]], TimeZone -> tz]}],
		((Head[apires] === List)&&(Length[Dimensions[apires]]===1)), System`DateObject[apires, TimeZone -> tz],
		((Head[apires] === List)&&(Length[Dimensions[apires]]>1)), System`DateObject[#, TimeZone -> tz]&/@apires,
		True, $Failed],
	td===TimeObject,
	Which[
		Head[apires] === Interval, 
		Interval[{System`TimeObject[apires[[1, 1]][[4;;]], TimeZone -> tz], System`TimeObject[apires[[1, 2]][[4;;]], TimeZone -> tz]}],
		((Head[apires] === List)&&(Length[Dimensions[apires]]===1)), System`TimeObject[apires[[4;;]], TimeZone -> tz],
		((Head[apires] === List)&&(Length[Dimensions[apires]]>1)), System`TimeObject[#[[4;;]], TimeZone -> tz]&/@apires,
		True, $Failed],
	td===Automatic,
	Which[
		iv===Automatic,
		Which[
			Head[apires] === Interval, 
			Interval[{System`DateObject[apires[[1, 1]], TimeZone -> tz], System`DateObject[apires[[1, 2]], TimeZone -> tz]}],
			((Head[apires] === List)&&(Length[Dimensions[apires]]===1)), System`DateObject[apires, TimeZone -> tz],
			((Head[apires] === List)&&(Length[Dimensions[apires]]>1)), System`DateObject[#, TimeZone -> tz]&/@apires,
			True, $Failed],
		MatchQ[iv, Min|Max|Mean|Interval],
		Which[
			Head[apires] === Interval, 
			Interval[{System`TimeObject[apires[[1, 1]][[4;;]], TimeZone -> tz], System`TimeObject[apires[[1, 2]][[4;;]], TimeZone -> tz]}],
			((Head[apires] === List)&&(Length[Dimensions[apires]]===1)), System`TimeObject[apires[[4;;]], TimeZone -> tz],
			((Head[apires] === List)&&(Length[Dimensions[apires]]>1)), System`TimeObject[#[[4;;]], TimeZone -> tz]&/@apires,
			True, $Failed],
		True, $Failed],
	True, $Failed
  ]

getArgumentType[x_, callingfunc_] := 
Which[
	validLocationQ[x, callingfunc]||MatchQ[x, {_?(validLocationQ[#, callingfunc]&)..}], "location",
	validDateQ[x, callingfunc]||MatchQ[x, {_?(validDateQ[#, callingfunc]&)..}]||MatchQ[x, {_?(validDateQ[#, callingfunc]&), _?(validDateQ[#, callingfunc]&), _?NumericQ}]||MatchQ[x, {_?(validDateQ[#, callingfunc]&), _?(validDateQ[#, callingfunc]&), {_?NumericQ, _String}}], "date",
	validIntervalValue[x, callingfunc], "applicationFunction",
	validLocationQualifier[x, callingfunc], "locationQualifier",
    validTimeDateFlag[x, callingfunc], "dateTime",
    validAssociationQ[x, callingfunc], "association",
    MatchQ[x, {{_?(validLocationQ[#, callingfunc]&), _?(validDateQ[#, callingfunc]&)}..}], "locationDatePairs",
    True, None]

makeSunriseArgumentsAndMessages[argList_List]  := 
Module[{argTypeList,expandedArguments,messages},
argTypeList = getArgumentType[#, Sunrise]& /@ argList;

{expandedArguments, messages} = 
Which[(* Sunrise[] *)
      Length[argList]  === 0,
      {{$Location, $Date, $ApplicationFunction, $DateTime}, {}},
 
      (* Sunrise[_] *)
      Length[argList] === 1,
      Which[(* good argument patterns *)
      	    MatchQ[argTypeList, {"locationDatePairs"}],
      	    {{argList[[1]], $ApplicationFunction, $DateTime}, {}},
            MatchQ[argTypeList, {"location"}],
            {{argList[[1]], $Date, $ApplicationFunction, $DateTime}, {}},
            MatchQ[argTypeList, {"date"}],
            {{$Location, argList[[1]], $ApplicationFunction, $DateTime}, {}},
            MatchQ[argTypeList, {"applicationFunction"}],
            {{$Location, $Date, argList[[1]], $DateTime}, {}},
            MatchQ[argTypeList, {"dateTime"}],
            {{$Location, $Date, $ApplicationFunction, argList[[1]]}, {}},
            MatchQ[argTypeList, {"association"}],
            Module[{tmploc, tmpdt, vlq, vdtq},
            	tmploc = argList[[1]]["Location"];
            	tmpdt = argList[[1]]["Date"];
            	vlq = validLocationQ[tmploc, Sunrise]||MatchQ[tmploc, Missing["KeyAbsent", ___]];
            	vdtq = validDateQ[tmpdt, Sunrise]||MatchQ[tmpdt, Missing["KeyAbsent", ___]];
            	If[!vlq||!vdtq, {{},{If[!vlq, tmploc -> {"loc"}, Sequence@@{}], If[!vdtq, tmpdt -> {"dtspec"}, Sequence@@{}]}}, {{tmploc/._Missing:>$Location, tmpdt/._Missing:>$Date, $ApplicationFunction, $DateTime}, {}}]],
            (* bad argument patterns *)
            True,
            {{}, {argList[[1]] -> {"locdate"}}}
           ],

      (* Sunrise[_, _] *)
      Length[argList] === 2,
       Which[(* good argument patterns *)
       	    MatchQ[argTypeList, {"locationDatePairs", "dateTime"}],
      	    {{argList[[1]], $ApplicationFunction, argList[[2]]}, {}},
            MatchQ[argTypeList, {"location", "date"}],
            {{argList[[1]], argList[[2]], $ApplicationFunction, $DateTime}, {}},
            MatchQ[argTypeList, {"location", "applicationFunction"}],
            {{argList[[1]], $Date, argList[[2]], $DateTime}, {}},
            MatchQ[argTypeList, {"location", "dateTime"}],
            {{argList[[1]], $Date, $ApplicationFunction, argList[[2]]}, {}},
           MatchQ[argTypeList, {"date", "applicationFunction"}],
            {{$Location, argList[[1]], argList[[2]], $DateTime}, {}},
           MatchQ[argTypeList, {"date", "dateTime"}],
            {{$Location, argList[[1]], $ApplicationFunction, argList[[2]]}, {}},
           MatchQ[argTypeList, {"applicationFunction", "dateTime"}],
            {{$Location, $Date, argList[[1]], argList[[2]]}, {}},
           MatchQ[argTypeList, {"locationDatePairs", "applicationFunction"}],
      	    {{argList[[1]], argList[[2]], $DateTime}, {}},
      	   MatchQ[argTypeList, {"association", "applicationFunction"}],
            Module[{tmploc, tmpdt, vlq, vdtq},
            	tmploc = argList[[1]]["Location"];
            	tmpdt = argList[[1]]["Date"];
            	vlq = validLocationQ[tmploc, Sunrise]||MatchQ[tmploc, Missing["KeyAbsent", ___]];
            	vdtq = validDateQ[tmpdt, Sunrise]||MatchQ[tmpdt, Missing["KeyAbsent", ___]];
            	If[!vlq||!vdtq, {{},{If[!vlq, tmploc -> {"loc"}, Sequence@@{}], If[!vdtq, tmpdt -> {"dtspec"}, Sequence@@{}]}}, {{tmploc/._Missing:>$Location, tmpdt/._Missing:>$Date, argList[[2]], $DateTime}, {}}]],
      	   MatchQ[argTypeList, {"association", "dateTime"}],
            Module[{tmploc, tmpdt, vlq, vdtq},
            	tmploc = argList[[1]]["Location"];
            	tmpdt = argList[[1]]["Date"];
            	vlq = validLocationQ[tmploc, Sunrise]||MatchQ[tmploc, Missing["KeyAbsent", ___]];
            	vdtq = validDateQ[tmpdt, Sunrise]||MatchQ[tmpdt, Missing["KeyAbsent", ___]];
            	If[!vlq||!vdtq, {{},{If[!vlq, tmploc -> {"loc"}, Sequence@@{}], If[!vdtq, tmpdt -> {"dtspec"}, Sequence@@{}]}}, {{tmploc/._Missing:>$Location, tmpdt/._Missing:>$Date, $ApplicationFunction, argList[[2]]}, {}}]],
           (* bad argument patterns *)
           MatchQ[argTypeList, {"location", _}],
           {{}, {argList[[2]] -> {"dtspec"}}},
           MatchQ[argTypeList, {"date", _}],
           {{}, {argList[[2]] -> {"locfun", "dtoba"}}},
      	   MatchQ[argTypeList, {"locationDatePairs", "location"}],
      	    {{}, {argList[[2]] -> {"dtoba"}}},
      	   MatchQ[argTypeList, {"locationDatePairs", "date"}],
      	    {{}, {argList[[2]] -> {"dtoba"}}},
      	   MatchQ[argTypeList, {"association", "location"}],
      	    {{}, {argList[[2]] -> {"locfun"}}},
      	   MatchQ[argTypeList, {"association", "date"}],
      	    {{}, {argList[[2]] -> {"locfun"}}},
           (* identical types *)
           MatchQ[argTypeList, {"location", "location"}],
            {{}, {argList[[2]] -> {"dtspec"}}},
           MatchQ[argTypeList, {"date", "date"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "applicationFunction"}],
            {{}, {argList[[1]] -> {"locdate"}}},
           MatchQ[argTypeList, {"dateTime", "dateTime"}],
            {{}, {argList[[1]] -> {"locdate"}}},
           MatchQ[argTypeList, {"association", "association"}],
            {{}, {argList[[1]] -> {"locdate"}}},
           (* incorrect order *)
           MatchQ[argTypeList, {"date", "location"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "location"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "date"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"dateTime", "date"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"dateTime", "location"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"dateTime", "applicationFunction"}],
            {{}, {argList[[1]] -> {"locdate"}}},
           MatchQ[argTypeList, {"applicationFunction", "association"}],
            {{}, {argList[[1]] -> {"locdate"}}},
           MatchQ[argTypeList, {"dateTime", "association"}],
            {{}, {argList[[1]] -> {"locdate"}}},

           (* unknown argument types *)
           MatchQ[argTypeList, {None, "location" | "date" | "association" | "applicationFunction" | "dateTime"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"location" | "date" | "association" | "applicationFunction" | "dateTime", None}],
            {{}, {argList[[2]] -> {"arg"}}},
           True,
            {{},{argList[[1]] -> {"locdate"}}}
           ],

      (* Sunrise[_, _, _] *)
      Length[argList] === 3,
      Which[(* good argument patterns *)
            MatchQ[argTypeList, {"location", "date", "applicationFunction"}],
            {{argList[[1]], argList[[2]], argList[[3]], $DateTime}, {}},
            MatchQ[argTypeList, {"location", "date", "dateTime"}],
            {{argList[[1]], argList[[2]], $ApplicationFunction, argList[[3]]}, {}},
            MatchQ[argTypeList, {"location", "applicationFunction", "dateTime"}],
            {{argList[[1]], $Date, argList[[2]], argList[[3]]}, {}},
            MatchQ[argTypeList, {"date", "applicationFunction", "dateTime"}],
            {{$Location, argList[[1]], argList[[2]], argList[[3]]}, {}},
            MatchQ[argTypeList, {"locationDatePairs", "applicationFunction", "dateTime"}],
            {{argList[[1]], argList[[2]], argList[[3]]}, {}},
            MatchQ[argTypeList, {"association", "applicationFunction", "dateTime"}],
            Module[{tmploc, tmpdt, vlq, vdtq},
            	tmploc = argList[[1]]["Location"];
            	tmpdt = argList[[1]]["Date"];
            	vlq = validLocationQ[tmploc, Sunrise]||MatchQ[tmploc, Missing["KeyAbsent", ___]];
            	vdtq = validDateQ[tmpdt, Sunrise]||MatchQ[tmpdt, Missing["KeyAbsent", ___]];
            	If[!vlq||!vdtq, {{},{If[!vlq, tmploc -> {"loc"}, Sequence@@{}], If[!vdtq, tmpdt -> {"dtspec"}, Sequence@@{}]}}, {{tmploc/._Missing:>$Location, tmpdt/._Missing:>$Date, argList[[2]], argList[[3]]}, {}}]],
           (* bad argument patterns *)
            MatchQ[argTypeList, {"location", "date", _}],
            {{}, {argList[[3]] -> {"dtoba"}}},
             MatchQ[argTypeList, {"location", _, "applicationFunction"}],
            {{}, {argList[[2]] -> {"dtspec"}}},
            MatchQ[argTypeList, {"location", "applicationFunction", _}],
            {{}, {argList[[3]] -> {"dtoba"}}},
             MatchQ[argTypeList, {_, "location", "date"}],
            {{}, {argList[[1]] -> {"loc"}}},
            MatchQ[argTypeList, {"location", _, _}],
            {{}, {argList[[2]] -> {"dtspec"}}},
            MatchQ[argTypeList, {"date", "applicationFunction", _}],
            {{},{argList[[3]] -> {"dtoba"}}},
            MatchQ[argTypeList, {"date", _, _}],
            {{},{argList[[2]] -> {"locfun"}}},
            MatchQ[argTypeList, {"locationDatePairs", "dateTime", _}],
            {{}, {argList[[3]] -> {"arg"}}},
            MatchQ[argTypeList, {"locationDatePairs", _, _}],
            {{}, {argList[[2]] -> {"locfun"}}},
           MatchQ[argTypeList, {"association", "location", _}],
            {{}, {argList[[2]] -> {"locfun"}}},
           MatchQ[argTypeList, {"association", "date", _}],
            {{}, {argList[[2]] -> {"locfun"}}},
           (* identical types *)
           MatchQ[argTypeList, {"location", "location", _}],
            {{}, {argList[[2]] -> {"dtspec"}}},
           MatchQ[argTypeList, {"date", "date", _}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "applicationFunction",_}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"dateTime", "dateTime",_}],
            {{}, {argList[[1]] -> {"locdate"}}},
           MatchQ[argTypeList, {"association", "association",_}],
            {{}, {argList[[2]] -> {"locfun"}}},
           True,
            {{},{argList[[1]] -> {"locdate"}}}
            ],
      (* Sunrise[_, _, _, _] *)
      Length[argList] === 4,
      Which[(* good argument patterns *)
            MatchQ[argTypeList, {"location", "date", "applicationFunction", "dateTime"}],
            {{argList[[1]], argList[[2]], argList[[3]], argList[[4]]}, {}},
           (* bad argument patterns *)
           MatchQ[argTypeList, {"locationDatePairs", "dateTime", _, _}],
            {{}, {argList[[3]] -> {"arg"}}},
           MatchQ[argTypeList, {"locationDatePairs", _, _, _}],
            {{}, {argList[[2]] -> {"dtoba"}}},
           MatchQ[argTypeList, {"location", "date", "applicationFunction", _}],
            {{}, {argList[[4]] -> {"dtoba"}}},
            MatchQ[argTypeList, {"location", "date", _, _}],
            {{}, {argList[[3]] -> {"locfun"}}},
           MatchQ[argTypeList, {"location", _, _, _}],
            {{}, {argList[[2]] -> {"dtspec"}}},
            MatchQ[argTypeList, {"association",_, _, _}],
            {{}, {argList[[2]] -> {"locfun"}}},
           (* identical types *)
           MatchQ[argTypeList, {"location", "location", _, _}],
            {{}, {argList[[2]] -> {"dtspec"}}},
           MatchQ[argTypeList, {"date", "date", _, _}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "applicationFunction", _, _}],
            {{}, {argList[[1]] -> {"locdate"}}},
           MatchQ[argTypeList, {"dateTime", "dateTime", _, _}],
            {{}, {argList[[1]] -> {"locdate"}}},
           MatchQ[argTypeList, {"association", "association", _, _}],
            {{}, {argList[[2]] -> {"locfun"}}},
           True,
            {{},{argList[[1]] -> {"locdate"}}} 
            ]
    ];

(* now resolve the meaning of $ApplicationFunction depending on the other arguments *)
If[!MatchQ[$GeoLocation, {_?NumericQ..}|_GeoPosition]&&!FreeQ[expandedArguments, $Location], Union[AppendTo[messages, $GeoLocation->{"geoloc"}]]];
{expandedArguments, messages}/.{$Location :> (With[{tmp=$GeoLocation}, If[!MatchQ[tmp, {_?NumericQ..}|_GeoPosition], tmp, If[!MatchQ[tmp, {_?NumericQ..}|_GeoPosition], {40.1,-88.2}, tmp]]]/.GeoPosition[x_]:>x[[1;;2]]), $Date :> DateList[], $ApplicationFunction -> Automatic, $DateTime -> Automatic}

]

makeSunsetArgumentsAndMessages[argList_List]  := 
Module[{argTypeList,expandedArguments,messages},
argTypeList = getArgumentType[#, Sunset]& /@ argList;

{expandedArguments, messages} = 
Which[(* Sunset[] *)
      Length[argList]  === 0,
      {{$Location, $Date, $ApplicationFunction, $DateTime}, {}},
 
      (* Sunset[_] *)
      Length[argList] === 1,
      Which[(* good argument patterns *)
      	    MatchQ[argTypeList, {"locationDatePairs"}],
      	    {{argList[[1]], $ApplicationFunction, $DateTime}, {}},
            MatchQ[argTypeList, {"location"}],
            {{argList[[1]], $Date, $ApplicationFunction, $DateTime}, {}},
            MatchQ[argTypeList, {"date"}],
            {{$Location, argList[[1]], $ApplicationFunction, $DateTime}, {}},
            MatchQ[argTypeList, {"applicationFunction"}],
            {{$Location, $Date, argList[[1]], $DateTime}, {}},
            MatchQ[argTypeList, {"dateTime"}],
            {{$Location, $Date, $ApplicationFunction, argList[[1]]}, {}},
            MatchQ[argTypeList, {"association"}],
            Module[{tmploc, tmpdt, vlq, vdtq},
            	tmploc = argList[[1]]["Location"];
            	tmpdt = argList[[1]]["Date"];
            	vlq = validLocationQ[tmploc, Sunset]||MatchQ[tmploc, Missing["KeyAbsent", ___]];
            	vdtq = validDateQ[tmpdt, Sunset]||MatchQ[tmpdt, Missing["KeyAbsent", ___]];
            	If[!vlq||!vdtq, {{},{If[!vlq, tmploc -> {"loc"}, Sequence@@{}], If[!vdtq, tmpdt -> {"dtspec"}, Sequence@@{}]}}, {{tmploc/._Missing:>$Location, tmpdt/._Missing:>$Date, $ApplicationFunction, $DateTime}, {}}]],
            (* bad argument patterns *)
            True,
            {{}, {argList[[1]] -> {"locdate"}}}
           ],

      (* Sunset[_, _] *)
      Length[argList] === 2,
       Which[(* good argument patterns *)
       	    MatchQ[argTypeList, {"locationDatePairs", "dateTime"}],
      	    {{argList[[1]], $ApplicationFunction, argList[[2]]}, {}},
            MatchQ[argTypeList, {"location", "date"}],
            {{argList[[1]], argList[[2]], $ApplicationFunction, $DateTime}, {}},
            MatchQ[argTypeList, {"location", "applicationFunction"}],
            {{argList[[1]], $Date, argList[[2]], $DateTime}, {}},
            MatchQ[argTypeList, {"location", "dateTime"}],
            {{argList[[1]], $Date, $ApplicationFunction, argList[[2]]}, {}},
           MatchQ[argTypeList, {"date", "applicationFunction"}],
            {{$Location, argList[[1]], argList[[2]], $DateTime}, {}},
           MatchQ[argTypeList, {"date", "dateTime"}],
            {{$Location, argList[[1]], $ApplicationFunction, argList[[2]]}, {}},
           MatchQ[argTypeList, {"applicationFunction", "dateTime"}],
            {{$Location, $Date, argList[[1]], argList[[2]]}, {}},
            MatchQ[argTypeList, {"locationDatePairs", "applicationFunction"}],
      	    {{argList[[1]], argList[[2]], $DateTime}, {}},
      	   MatchQ[argTypeList, {"association", "applicationFunction"}],
            Module[{tmploc, tmpdt, vlq, vdtq},
            	tmploc = argList[[1]]["Location"];
            	tmpdt = argList[[1]]["Date"];
            	vlq = validLocationQ[tmploc, Sunset]||MatchQ[tmploc, Missing["KeyAbsent", ___]];
            	vdtq = validDateQ[tmpdt, Sunset]||MatchQ[tmpdt, Missing["KeyAbsent", ___]];
            	If[!vlq||!vdtq, {{},{If[!vlq, tmploc -> {"loc"}, Sequence@@{}], If[!vdtq, tmpdt -> {"dtspec"}, Sequence@@{}]}}, {{tmploc/._Missing:>$Location, tmpdt/._Missing:>$Date, argList[[2]], $DateTime}, {}}]],
      	   MatchQ[argTypeList, {"association", "dateTime"}],
            Module[{tmploc, tmpdt, vlq, vdtq},
            	tmploc = argList[[1]]["Location"];
            	tmpdt = argList[[1]]["Date"];
            	vlq = validLocationQ[tmploc, Sunset]||MatchQ[tmploc, Missing["KeyAbsent", ___]];
            	vdtq = validDateQ[tmpdt, Sunset]||MatchQ[tmpdt, Missing["KeyAbsent", ___]];
            	If[!vlq||!vdtq, {{},{If[!vlq, tmploc -> {"loc"}, Sequence@@{}], If[!vdtq, tmpdt -> {"dtspec"}, Sequence@@{}]}}, {{tmploc/._Missing:>$Location, tmpdt/._Missing:>$Date, $ApplicationFunction, argList[[2]]}, {}}]],
           (* bad argument patterns *)
           MatchQ[argTypeList, {"location", _}],
           {{}, {argList[[2]] -> {"dtspec"}}},
            MatchQ[argTypeList, {"date", _}],
           {{}, {argList[[2]] -> {"locfun", "dtoba"}}},
      	   MatchQ[argTypeList, {"locationDatePairs", "location"}],
      	    {{}, {argList[[2]] -> {"dtoba"}}},
      	    MatchQ[argTypeList, {"locationDatePairs", "date"}],
      	    {{}, {argList[[2]] -> {"dtoba"}}},
      	   MatchQ[argTypeList, {"association", "location"}],
      	    {{}, {argList[[2]] -> {"locfun"}}},
      	   MatchQ[argTypeList, {"association", "date"}],
      	    {{}, {argList[[2]] -> {"locfun"}}},
           (* identical types *)
           MatchQ[argTypeList, {"location", "location"}],
            {{}, {argList[[2]] -> {"dtspec"}}},
           MatchQ[argTypeList, {"date", "date"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "applicationFunction"}],
            {{}, {argList[[1]] -> {"locdate"}}},
           MatchQ[argTypeList, {"dateTime", "dateTime"}],
            {{}, {argList[[1]] -> {"locdate"}}},
           MatchQ[argTypeList, {"association", "association"}],
            {{}, {argList[[1]] -> {"locdate"}}},
           (* incorrect order *)
           MatchQ[argTypeList, {"date", "location"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "location"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "date"}],
            {{}, {argList[[1]] -> {"loc"}}},
            MatchQ[argTypeList, {"dateTime", "date"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"dateTime", "location"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"dateTime", "applicationFunction"}],
            {{}, {argList[[1]] -> {"locdate"}}},
           MatchQ[argTypeList, {"applicationFunction", "association"}],
            {{}, {argList[[1]] -> {"locdate"}}},
           MatchQ[argTypeList, {"dateTime", "association"}],
            {{}, {argList[[1]] -> {"locdate"}}},

           (* unknown argument types *)
           MatchQ[argTypeList, {None, "location" | "date" | "association" | "applicationFunction" | "dateTime"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"location" | "date" | "association" | "applicationFunction" | "dateTime", None}],
            {{}, {argList[[2]] -> {"arg"}}},
           True,
            {{},{argList[[1]] -> {"locdate"}}}
           ],

      (* Sunset[_, _, _] *)
      Length[argList] === 3,
      Which[(* good argument patterns *)
            MatchQ[argTypeList, {"location", "date", "applicationFunction"}],
            {{argList[[1]], argList[[2]], argList[[3]], $DateTime}, {}},
            MatchQ[argTypeList, {"location", "date", "dateTime"}],
            {{argList[[1]], argList[[2]], $ApplicationFunction, argList[[3]]}, {}},
            MatchQ[argTypeList, {"location", "applicationFunction", "dateTime"}],
            {{argList[[1]], $Date, argList[[2]], argList[[3]]}, {}},
            MatchQ[argTypeList, {"date", "applicationFunction", "dateTime"}],
            {{$Location, argList[[1]], argList[[2]], argList[[3]]}, {}},
            MatchQ[argTypeList, {"locationDatePairs", "applicationFunction", "dateTime"}],
            {{argList[[1]], argList[[2]], argList[[3]]}, {}},
            MatchQ[argTypeList, {"association", "applicationFunction", "dateTime"}],
            Module[{tmploc, tmpdt, vlq, vdtq},
            	tmploc = argList[[1]]["Location"];
            	tmpdt = argList[[1]]["Date"];
            	vlq = validLocationQ[tmploc, Sunset]||MatchQ[tmploc, Missing["KeyAbsent", ___]];
            	vdtq = validDateQ[tmpdt, Sunset]||MatchQ[tmpdt, Missing["KeyAbsent", ___]];
            	If[!vlq||!vdtq, {{},{If[!vlq, tmploc -> {"loc"}, Sequence@@{}], If[!vdtq, tmpdt -> {"dtspec"}, Sequence@@{}]}}, {{tmploc/._Missing:>$Location, tmpdt/._Missing:>$Date, argList[[2]], argList[[3]]}, {}}]],
           (* bad argument patterns *)
            MatchQ[argTypeList, {"location", "date", _}],
            {{}, {argList[[3]] -> {"dtoba"}}},
             MatchQ[argTypeList, {"location", _, "applicationFunction"}],
            {{}, {argList[[2]] -> {"dtspec"}}},
            MatchQ[argTypeList, {"location", "applicationFunction", _}],
            {{}, {argList[[3]] -> {"dtoba"}}},
             MatchQ[argTypeList, {_, "location", "date"}],
            {{}, {argList[[1]] -> {"loc"}}},
            MatchQ[argTypeList, {"location", _, _}],
            {{}, {argList[[2]] -> {"dtspec"}}},
            MatchQ[argTypeList, {"date", "applicationFunction", _}],
            {{},{argList[[3]] -> {"dtoba"}}},
            MatchQ[argTypeList, {"date", _, _}],
            {{},{argList[[2]] -> {"locfun"}}},
            MatchQ[argTypeList, {"locationDatePairs", "dateTime", _}],
            {{}, {argList[[3]] -> {"arg"}}},
            MatchQ[argTypeList, {"locationDatePairs", _, _}],
            {{}, {argList[[2]] -> {"locfun"}}},
           MatchQ[argTypeList, {"association", "location", _}],
            {{}, {argList[[2]] -> {"locfun"}}},
           MatchQ[argTypeList, {"association", "date", _}],
            {{}, {argList[[2]] -> {"locfun"}}},
           (* identical types *)
           MatchQ[argTypeList, {"location", "location", _}],
            {{}, {argList[[2]] -> {"dtspec"}}},
           MatchQ[argTypeList, {"date", "date", _}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "applicationFunction",_}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"dateTime", "dateTime",_}],
            {{}, {argList[[1]] -> {"locdate"}}},
           MatchQ[argTypeList, {"association", "association",_}],
            {{}, {argList[[2]] -> {"locfun"}}},
           True,
            {{},{argList[[1]] -> {"locdate"}}}
            ],
      (* Sunset[_, _, _, _] *)
      Length[argList] === 4,
      Which[(* good argument patterns *)
           MatchQ[argTypeList, {"location", "date", "applicationFunction", "dateTime"}],
            {{argList[[1]], argList[[2]], argList[[3]], argList[[4]]}, {}},
           (* bad argument patterns *)
           MatchQ[argTypeList, {"locationDatePairs", "dateTime", _, _}],
            {{}, {argList[[3]] -> {"arg"}}},
           MatchQ[argTypeList, {"locationDatePairs", _, _, _}],
            {{}, {argList[[2]] -> {"dtoba"}}},
           MatchQ[argTypeList, {"location", "date", "applicationFunction", _}],
            {{}, {argList[[4]] -> {"dtoba"}}},
           MatchQ[argTypeList, {"location", "date", _, _}],
            {{}, {argList[[3]] -> {"locfun"}}},
           MatchQ[argTypeList, {"location", _, _, _}],
            {{}, {argList[[2]] -> {"dtspec"}}},
           MatchQ[argTypeList, {"association",_, _, _}],
            {{}, {argList[[2]] -> {"locfun"}}},
           (* identical types *)
           MatchQ[argTypeList, {"location", "location", _, _}],
            {{}, {argList[[2]] -> {"dtspec"}}},
           MatchQ[argTypeList, {"date", "date", _, _}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "applicationFunction", _, _}],
            {{}, {argList[[1]] -> {"locdate"}}},
           MatchQ[argTypeList, {"dateTime", "dateTime", _, _}],
            {{}, {argList[[1]] -> {"locdate"}}},
           MatchQ[argTypeList, {"association", "association", _, _}],
            {{}, {argList[[2]] -> {"locfun"}}},
           True,
            {{},{argList[[1]] -> {"locdate"}}} 
            ]
    ];

(* now resolve the meaning of $ApplicationFunction depending on the other arguments *)
If[!MatchQ[$GeoLocation, {_?NumericQ..}|_GeoPosition]&&!FreeQ[expandedArguments, $Location], Union[AppendTo[messages, $GeoLocation->{"geoloc"}]]];
{expandedArguments, messages}/.{$Location :> (With[{tmp=$GeoLocation}, If[!MatchQ[tmp, {_?NumericQ..}|_GeoPosition], tmp, If[!MatchQ[tmp, {_?NumericQ..}|_GeoPosition], {40.1,-88.2}, tmp]]]/.GeoPosition[x_]:>x[[1;;2]]), $Date :> DateList[], $ApplicationFunction -> Automatic, $DateTime -> Automatic}

]

Options[makeSunPositionArgumentsAndMessages] = {CelestialSystem -> "Horizon"} 

makeSunPositionArgumentsAndMessages[argList_List, OptionsPattern[makeSunPositionArgumentsAndMessages]]  := 
Module[{argTypeList,expandedArguments,messages, cs},
argTypeList = getArgumentType[#, SunPosition]& /@ argList;
cs = OptionValue[CelestialSystem];
{expandedArguments, messages} = 
Which[(* SunPosition[] *)
      Length[argList]  === 0,
      {{$Location, $Date, $ApplicationFunction}, {}},
 
      (* SunPosition[_] *)
      Length[argList] === 1,
      Which[(* good argument patterns *)
      	    MatchQ[argTypeList, {"locationDatePairs"}],
      	    {{argList[[1]], $ApplicationFunction}, {}},
            MatchQ[argTypeList, {"location"}],
            {{argList[[1]], $Date, $ApplicationFunction}, {}},
            MatchQ[argTypeList, {"date"}],
            {{$Location, argList[[1]], $ApplicationFunction}, {}},
            MatchQ[argTypeList, {"applicationFunction"}],
            {{$Location, $Date, argList[[1]]}, {}},
            MatchQ[argTypeList, {"association"}],
            Module[{tmploc, tmpdt, vlq, vdtq},
            	tmploc = argList[[1]]["Location"];
            	tmpdt = argList[[1]]["Date"];
            	vlq = validLocationQ[tmploc, SunPosition]||MatchQ[tmploc, Missing["KeyAbsent", ___]];
            	vdtq = validDateQ[tmpdt, SunPosition]||MatchQ[tmpdt, Missing["KeyAbsent", ___]];
            	If[!vlq||!vdtq, {{},{If[!vlq, tmploc -> {"loc"}, Sequence@@{}], If[!vdtq, tmpdt -> {"dtspec"}, Sequence@@{}]}}, {{tmploc/._Missing:>$Location, tmpdt/._Missing:>$Date, $ApplicationFunction, $DateTime}, {}}]],
            (* bad argument patterns *)
            True,
            {{}, {argList[[1]] -> {"locdate"}}}
           ],

      (* SunPosition[_, _] *)
      Length[argList] === 2,
       Which[(* good argument patterns *)
            MatchQ[argTypeList, {"location", "date"}],
            {{argList[[1]], argList[[2]], $ApplicationFunction}, {}},
            MatchQ[argTypeList, {"location", "applicationFunction"}],
            {{argList[[1]], $Date, argList[[2]]}, {}},
           MatchQ[argTypeList, {"date", "applicationFunction"}],
            {{$Location, argList[[1]], argList[[2]]}, {}},
           MatchQ[argTypeList, {"locationDatePairs", "applicationFunction"}],
      	    {{argList[[1]], argList[[2]]}, {}},
      	   MatchQ[argTypeList, {"association", "applicationFunction"}],
            Module[{tmploc, tmpdt, vlq, vdtq},
            	tmploc = argList[[1]]["Location"];
            	tmpdt = argList[[1]]["Date"];
            	vlq = validLocationQ[tmploc, SunPosition]||MatchQ[tmploc, Missing["KeyAbsent", ___]];
            	vdtq = validDateQ[tmpdt, SunPosition]||MatchQ[tmpdt, Missing["KeyAbsent", ___]];
            	If[!vlq||!vdtq, {{},{If[!vlq, tmploc -> {"loc"}, Sequence@@{}], If[!vdtq, tmpdt -> {"dtspec"}, Sequence@@{}]}}, {{tmploc/._Missing:>$Location, tmpdt/._Missing:>$Date, argList[[2]]}, {}}]],
           (* bad argument patterns *)
           MatchQ[argTypeList, {"location", _}],
           {{}, {argList[[2]] -> {"dtspec"}}},
           MatchQ[argTypeList, {"date", _}],
           {{}, {argList[[2]] -> {"locfun"}}},
      	   MatchQ[argTypeList, {"locationDatePairs", "location"}],
      	    {{}, {argList[[2]] -> {"locfun"}}},
      	    MatchQ[argTypeList, {"locationDatePairs", "date"}],
      	    {{}, {argList[[2]] -> {"locfun"}}},
      	   MatchQ[argTypeList, {"association", "location"}],
      	    {{}, {argList[[2]] -> {"locfun"}}},
      	   MatchQ[argTypeList, {"association", "date"}],
      	    {{}, {argList[[2]] -> {"locfun"}}},
           (* identical types *)
           MatchQ[argTypeList, {"location", "location"}],
            {{}, {argList[[2]] -> {"dtspec"}}},
           MatchQ[argTypeList, {"date", "date"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "applicationFunction"}],
            {{}, {argList[[1]] -> {"locdate"}}},
           MatchQ[argTypeList, {"association", "association"}],
            {{}, {argList[[2]] -> {"locfun"}}},
           (* incorrect order *)
           MatchQ[argTypeList, {"date", "location"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "location"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "date"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "association"}],
            {{}, {argList[[1]] -> {"locdate"}}},

           (* unknown argument types *)
           MatchQ[argTypeList, {None, "location" | "date" | "association" | "applicationFunction"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"location" | "date" | "association" | "applicationFunction", None}],
            {{}, {argList[[2]] -> {"arg"}}},
           True,
            {{},{argList[[1]] -> {"locdate"}}}
           ],

      (* SunPosition[_, _, _] *)
      Length[argList] === 3,
      Which[(* good argument patterns *)
            MatchQ[argTypeList, {"location", "date", "applicationFunction"}],
            {{argList[[1]], argList[[2]], argList[[3]]}, {}},
           (* bad argument patterns *)
            MatchQ[argTypeList, {"location", "date", _}],
            {{}, {argList[[3]] -> {"locfun"}}},
             MatchQ[argTypeList, {"location", _, "applicationFunction"}],
            {{}, {argList[[2]] -> {"dtspec"}}},
            MatchQ[argTypeList, {"location", "applicationFunction", _}],
            {{}, {argList[[3]] -> {"arg"}}},
             MatchQ[argTypeList, {_, "location", "date"}],
            {{}, {argList[[1]] -> {"loc"}}},
            MatchQ[argTypeList, {"location", _, _}],
            {{}, {argList[[2]] -> {"dtspec"}}},
            MatchQ[argTypeList, {"date", "applicationFunction", _}],
            {{},{argList[[3]] -> {"arg"}}},
            MatchQ[argTypeList, {"date", _, _}],
            {{},{argList[[2]] -> {"locfun"}}},
            MatchQ[argTypeList, {"locationDatePairs", _, _}],
            {{}, {argList[[2]] -> {"locfun"}}},
           MatchQ[argTypeList, {"association", _, _}],
            {{}, {argList[[3]] -> {"arg"}}},
           (* identical types *)
           MatchQ[argTypeList, {"location", "location", _}],
            {{}, {argList[[2]] -> {"dtspec"}}},
           MatchQ[argTypeList, {"date", "date", _}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "applicationFunction",_}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"association", "association", _}],
            {{}, {argList[[2]] -> {"dtspec"}}},
           True,
            {{},{argList[[1]] -> {"locdate"}}}
            ]
    ];

(* now resolve the meaning of $ApplicationFunction depending on the other arguments *)
If[!MatchQ[$GeoLocation, {_?NumericQ..}|_GeoPosition]&&!MatchQ[cs, "Equatorial"]&&!FreeQ[expandedArguments, $Location], Union[AppendTo[messages, $GeoLocation->{"geoloc"}]]];
{expandedArguments, messages}/.{$Location :> (With[{tmp=$GeoLocation}, If[!MatchQ[tmp, {_?NumericQ..}|_GeoPosition]&&!MatchQ[cs, "Equatorial"], tmp, If[!MatchQ[tmp, {_?NumericQ..}|_GeoPosition]&&MatchQ[cs, "Equatorial"], {40.1,-88.2}, tmp]]]/.GeoPosition[x_]:>x[[1;;2]]), $Date :> DateList[], $ApplicationFunction -> Automatic}
]

Options[makeMoonPositionArgumentsAndMessages] = {CelestialSystem -> "Horizon"}

makeMoonPositionArgumentsAndMessages[argList_List, OptionsPattern[makeMoonPositionArgumentsAndMessages]]  := 
Module[{argTypeList,expandedArguments,messages, cs},
argTypeList = getArgumentType[#, MoonPosition]& /@ argList;

cs = OptionValue[CelestialSystem];
{expandedArguments, messages} = 
Which[(* MoonPosition[] *)
      Length[argList]  === 0,
      {{$Location, $Date, $ApplicationFunction}, {}},
 
      (* MoonPosition[_] *)
      Length[argList] === 1,
      Which[(* good argument patterns *)
      	    MatchQ[argTypeList, {"locationDatePairs"}],
      	    {{argList[[1]], $ApplicationFunction}, {}},
            MatchQ[argTypeList, {"location"}],
            {{argList[[1]], $Date, $ApplicationFunction}, {}},
            MatchQ[argTypeList, {"date"}],
            {{$Location, argList[[1]], $ApplicationFunction}, {}},
            MatchQ[argTypeList, {"applicationFunction"}],
            {{$Location, $Date, argList[[1]]}, {}},
            MatchQ[argTypeList, {"association"}],
            Module[{tmploc, tmpdt, vlq, vdtq},
            	tmploc = argList[[1]]["Location"];
            	tmpdt = argList[[1]]["Date"];
            	vlq = validLocationQ[tmploc, MoonPosition]||MatchQ[tmploc, Missing["KeyAbsent", ___]];
            	vdtq = validDateQ[tmpdt, MoonPosition]||MatchQ[tmpdt, Missing["KeyAbsent", ___]];
            	If[!vlq||!vdtq, {{},{If[!vlq, tmploc -> {"loc"}, Sequence@@{}], If[!vdtq, tmpdt -> {"dtspec"}, Sequence@@{}]}}, {{tmploc/._Missing:>$Location, tmpdt/._Missing:>$Date, $ApplicationFunction}, {}}]],
            (* bad argument patterns *)
            True,
            {{}, {argList[[1]] -> {"locdate"}}}
           ],

      (* MoonPosition[_, _] *)
      Length[argList] === 2,
       Which[(* good argument patterns *)
            MatchQ[argTypeList, {"location", "date"}],
            {{argList[[1]], argList[[2]], $ApplicationFunction}, {}},
            MatchQ[argTypeList, {"location", "applicationFunction"}],
            {{argList[[1]], $Date, argList[[2]]}, {}},
           MatchQ[argTypeList, {"date", "applicationFunction"}],
            {{$Location, argList[[1]], argList[[2]]}, {}},
           MatchQ[argTypeList, {"locationDatePairs", "applicationFunction"}],
      	    {{argList[[1]], argList[[2]]}, {}},
      	   MatchQ[argTypeList, {"association", "applicationFunction"}],
            Module[{tmploc, tmpdt, vlq, vdtq},
            	tmploc = argList[[1]]["Location"];
            	tmpdt = argList[[1]]["Date"];
            	vlq = validLocationQ[tmploc, MoonPosition]||MatchQ[tmploc, Missing["KeyAbsent", ___]];
            	vdtq = validDateQ[tmpdt, MoonPosition]||MatchQ[tmpdt, Missing["KeyAbsent", ___]];
            	If[!vlq||!vdtq, {{},{If[!vlq, tmploc -> {"loc"}, Sequence@@{}], If[!vdtq, tmpdt -> {"dtspec"}, Sequence@@{}]}}, {{tmploc/._Missing:>$Location, tmpdt/._Missing:>$Date, argList[[2]]}, {}}]],
           (* bad argument patterns *)
           MatchQ[argTypeList, {"location", _}],
           {{}, {argList[[2]] -> {"dtspec"}}},
           MatchQ[argTypeList, {"date", _}],
           {{}, {argList[[2]] -> {"locfun"}}},
      	   MatchQ[argTypeList, {"locationDatePairs", "location"}],
      	    {{}, {argList[[2]] -> {"locfun"}}},
      	    MatchQ[argTypeList, {"locationDatePairs", "date"}],
      	    {{}, {argList[[2]] -> {"locfun"}}},
      	   MatchQ[argTypeList, {"association", "location"}],
      	    {{}, {argList[[2]] -> {"locfun"}}},
      	   MatchQ[argTypeList, {"association", "date"}],
      	    {{}, {argList[[2]] -> {"locfun"}}},
           (* identical types *)
           MatchQ[argTypeList, {"location", "location"}],
            {{}, {argList[[2]] -> {"dtspec"}}},
           MatchQ[argTypeList, {"date", "date"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "applicationFunction"}],
            {{}, {argList[[1]] -> {"locdate"}}},
           MatchQ[argTypeList, {"association", "association"}],
            {{}, {argList[[2]] -> {"locfun"}}},
           (* incorrect order *)
           MatchQ[argTypeList, {"date", "location"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "location"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "date"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "association"}],
            {{}, {argList[[1]] -> {"locdate"}}},

           (* unknown argument types *)
           MatchQ[argTypeList, {None, "location" | "date" | "association" | "applicationFunction"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"location" | "date" | "association" | "applicationFunction", None}],
            {{}, {argList[[2]] -> {"arg"}}},
           True,
            {{},{argList[[1]] -> {"locdate"}}}
           ],

      (* MoonPosition[_, _, _] *)
      Length[argList] === 3,
      Which[(* good argument patterns *)
            MatchQ[argTypeList, {"location", "date", "applicationFunction"}],
            {{argList[[1]], argList[[2]], argList[[3]]}, {}},
           (* bad argument patterns *)
            MatchQ[argTypeList, {"location", "date", _}],
            {{}, {argList[[3]] -> {"locfun"}}},
             MatchQ[argTypeList, {"location", _, "applicationFunction"}],
            {{}, {argList[[2]] -> {"dtspec"}}},
            MatchQ[argTypeList, {"location", "applicationFunction", _}],
            {{}, {argList[[3]] -> {"arg"}}},
             MatchQ[argTypeList, {_, "location", "date"}],
            {{}, {argList[[1]] -> {"loc"}}},
            MatchQ[argTypeList, {"location", _, _}],
            {{}, {argList[[2]] -> {"dtspec"}}},
            MatchQ[argTypeList, {"date", "applicationFunction", _}],
            {{},{argList[[3]] -> {"arg"}}},
            MatchQ[argTypeList, {"date", _, _}],
            {{},{argList[[2]] -> {"locfun"}}},
            MatchQ[argTypeList, {"locationDatePairs", _, _}],
            {{}, {argList[[2]] -> {"locfun"}}},
           MatchQ[argTypeList, {"association", _, _}],
            {{}, {argList[[3]] -> {"arg"}}},
           (* identical types *)
           MatchQ[argTypeList, {"location", "location", _}],
            {{}, {argList[[2]] -> {"dtspec"}}},
           MatchQ[argTypeList, {"date", "date", _}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "applicationFunction",_}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"association", "association", _}],
            {{}, {argList[[2]] -> {"dtspec"}}},
           True,
            {{},{argList[[1]] -> {"locdate"}}}
            ]
    ];

(* now resolve the meaning of $ApplicationFunction depending on the other arguments *)
If[!MatchQ[$GeoLocation, {_?NumericQ..}|_GeoPosition]&&!MatchQ[cs, "Equatorial"]&&!FreeQ[expandedArguments, $Location], Union[AppendTo[messages, $GeoLocation->{"geoloc"}]]];
{expandedArguments, messages}/.{$Location :> (With[{tmp=$GeoLocation}, If[!MatchQ[tmp, {_?NumericQ..}|_GeoPosition]&&!MatchQ[cs, "Equatorial"], tmp, If[!MatchQ[tmp, {_?NumericQ..}|_GeoPosition]&&MatchQ[cs, "Equatorial"], {40.1,-88.2}, tmp]]]/.GeoPosition[x_]:>x[[1;;2]]), $Date :> DateList[], $ApplicationFunction -> Automatic}

]

makeSiderealTimeArgumentsAndMessages[argList_List]  := 
Module[{argTypeList,expandedArguments,messages},
argTypeList = getArgumentType[#, SiderealTime]& /@ argList;

{expandedArguments, messages} = 
Which[(* SiderealTime[] *)
      Length[argList]  === 0,
      {{$Location, $Date, $ApplicationFunction}, {}},
 
      (* SiderealTime[_] *)
      Length[argList] === 1,
      Which[(* good argument patterns *)
      	    MatchQ[argTypeList, {"locationDatePairs"}],
      	    {{argList[[1]], $ApplicationFunction}, {}},
            MatchQ[argTypeList, {"location"}],
            {{argList[[1]], $Date, $ApplicationFunction}, {}},
            MatchQ[argTypeList, {"date"}],
            {{$Location, argList[[1]], $ApplicationFunction}, {}},
            MatchQ[argTypeList, {"applicationFunction"}],
            {{$Location, $Date, argList[[1]]}, {}},
            (* bad argument patterns *)
            True,
            {{}, {argList[[1]] -> {"locdate"}}}
           ],

      (* SiderealTime[_, _] *)
      Length[argList] === 2,
       Which[(* good argument patterns *)
            MatchQ[argTypeList, {"location", "date"}],
            {{argList[[1]], argList[[2]], $ApplicationFunction}, {}},
            MatchQ[argTypeList, {"location", "applicationFunction"}],
            {{argList[[1]], $Date, argList[[2]]}, {}},
           MatchQ[argTypeList, {"date", "applicationFunction"}],
            {{$Location, argList[[1]], argList[[2]]}, {}},
           MatchQ[argTypeList, {"locationDatePairs", "applicationFunction"}],
      	    {{argList[[1]], argList[[2]]}, {}},
           (* bad argument patterns *)
           MatchQ[argTypeList, {"location", _}],
           {{}, {argList[[2]] -> {"dtspec"}}},
           MatchQ[argTypeList, {"date", _}],
           {{}, {argList[[2]] -> {"locfun"}}},
      	   MatchQ[argTypeList, {"locationDatePairs", "location"}],
      	    {{}, {argList[[2]] -> {"locfun"}}},
      	    MatchQ[argTypeList, {"locationDatePairs", "date"}],
      	    {{}, {argList[[2]] -> {"locfun"}}},
           (* identical types *)
           MatchQ[argTypeList, {"location", "location"}],
            {{}, {argList[[2]] -> {"dtspec"}}},
           MatchQ[argTypeList, {"date", "date"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "applicationFunction"}],
            {{}, {argList[[1]] -> {"locdate"}}},
           (* incorrect order *)
           MatchQ[argTypeList, {"date", "location"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "location"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "date"}],
            {{}, {argList[[1]] -> {"loc"}}},

           (* unknown argument types *)
           MatchQ[argTypeList, {None, "location" | "date" | "applicationFunction"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"location" | "date" | "applicationFunction", None}],
            {{}, {argList[[2]] -> {"arg"}}},
           True,
            {{},{argList[[1]] -> {"locdate"}}}
           ],

      (* SiderealTime[_, _, _] *)
      Length[argList] === 3,
      Which[(* good argument patterns *)
            MatchQ[argTypeList, {"location", "date", "applicationFunction"}],
            {{argList[[1]], argList[[2]], argList[[3]]}, {}},
           (* bad argument patterns *)
            MatchQ[argTypeList, {"location", "date", _}],
            {{}, {argList[[3]] -> {"locfun"}}},
             MatchQ[argTypeList, {"location", _, "applicationFunction"}],
            {{}, {argList[[2]] -> {"dtspec"}}},
            MatchQ[argTypeList, {"location", "applicationFunction", _}],
            {{}, {argList[[3]] -> {"arg"}}},
             MatchQ[argTypeList, {_, "location", "date"}],
            {{}, {argList[[1]] -> {"loc"}}},
            MatchQ[argTypeList, {"location", _, _}],
            {{}, {argList[[2]] -> {"dtspec"}}},
            MatchQ[argTypeList, {"date", "applicationFunction", _}],
            {{},{argList[[3]] -> {"arg"}}},
            MatchQ[argTypeList, {"date", _, _}],
            {{},{argList[[2]] -> {"locfun"}}},
            MatchQ[argTypeList, {"locationDatePairs", _, _}],
            {{}, {argList[[2]] -> {"locfun"}}},
           (* identical types *)
           MatchQ[argTypeList, {"location", "location", _}],
            {{}, {argList[[2]] -> {"dtspec"}}},
           MatchQ[argTypeList, {"date", "date", _}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"applicationFunction", "applicationFunction",_}],
            {{}, {argList[[1]] -> {"loc"}}},
           True,
            {{},{argList[[1]] -> {"locdate"}}}
            ]
    ];

(* now resolve the meaning of $ApplicationFunction depending on the other arguments *)
If[!MatchQ[$GeoLocation, {_?NumericQ..}|_GeoPosition]&&!FreeQ[expandedArguments, $Location], Union[AppendTo[messages, $GeoLocation->{"geoloc"}]]];
{expandedArguments, messages}/.{$Location :> (With[{tmp=$GeoLocation}, If[!MatchQ[tmp, {_?NumericQ..}|_GeoPosition], tmp, If[!MatchQ[tmp, {_?NumericQ..}|_GeoPosition], {40.1,-88.2}, tmp]]]/.GeoPosition[x_]:>x[[1;;2]]), $Date :> DateList[], $ApplicationFunction -> Automatic}

]

makeDaylightQArgumentsAndMessages[argList_List]  := 
Module[{argTypeList,expandedArguments,messages},
argTypeList = getArgumentType[#, DaylightQ]& /@ argList;

{expandedArguments, messages} = 
Which[(* DaylightQ[] *)
      Length[argList]  === 0,
      {{$Location, $Date, $LocationQualifier}, {}},
 
      (* DaylightQ[_] *)
      Length[argList] === 1,
      Which[(* good argument patterns *)
      	    MatchQ[argTypeList, {"locationDatePairs"}],
      	    {{argList[[1]], $LocationQualifier}, {}},
            MatchQ[argTypeList, {"location"}],
            {{argList[[1]], $Date, $LocationQualifier}, {}},
            MatchQ[argTypeList, {"date"}],
            {{$Location, argList[[1]], $LocationQualifier}, {}},
             MatchQ[argTypeList, {"locationQualifier"}],
            {{$Location, $Date, argList[[1]]}, {}},
            (* bad argument patterns *)
            True,
            {{}, {argList[[1]] -> {"locdate"}}}
           ],

      (* DaylightQ[_, _] *)
      Length[argList] === 2,
       Which[(* good argument patterns *)
           MatchQ[argTypeList, {"location", "date"}],
            {{argList[[1]], argList[[2]], $LocationQualifier}, {}},
           MatchQ[argTypeList, {"location", "locationQualifier"}],
            {{argList[[1]], $Date, argList[[2]]}, {}},
           MatchQ[argTypeList, {"date", "locationQualifier"}],
            {{$Location, argList[[1]], argList[[2]]}, {}},
      	   MatchQ[argTypeList, {"locationDatePairs", "locationQualifier"}],
      	    {{argList[[1]], argList[[2]]}, {}},
           (* bad argument patterns *)
           MatchQ[argTypeList, {"location", _}],
           {{}, {argList[[2]] -> {"dtspec"}}},
           MatchQ[argTypeList, {"date", _}],
           {{}, {argList[[2]] -> {"incspec"}}},
      	   MatchQ[argTypeList, {"locationDatePairs", "location"}],
      	    {{}, {argList[[2]] -> {"incspec"}}},
      	    MatchQ[argTypeList, {"locationDatePairs", "date"}],
      	    {{}, {argList[[2]] -> {"incspec"}}},
           (* identical types *)
           MatchQ[argTypeList, {"location", "location"}],
            {{}, {argList[[2]] -> {"dtspec"}}},
           MatchQ[argTypeList, {"date", "date"}],
            {{}, {argList[[1]] -> {"loc"}}},
            MatchQ[argTypeList, {"locationQualifier", "locationQualifier"}],
            {{}, {argList[[1]] -> {"locdate"}}},
           (* incorrect order *)
           MatchQ[argTypeList, {"date", "location"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"locationQualifier", "location"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"locationQualifier", "date"}],
            {{}, {argList[[1]] -> {"loc"}}},

           (* unknown argument types *)
           MatchQ[argTypeList, {None, "location" | "date" | "locationQualifier"}],
            {{}, {argList[[1]] -> {"loc"}}},
           MatchQ[argTypeList, {"location" | "date" | "locationQualifier", None}],
            {{}, {argList[[2]] -> {"arg"}}},
           True,
            {{},{argList[[1]] -> {"locdate"}}}
           ],

      (* DaylightQ[_, _, _] *)
      Length[argList] === 3,
      Which[(* good argument patterns *)
            MatchQ[argTypeList, {"location", "date", "locationQualifier"}],
            {{argList[[1]], argList[[2]], argList[[3]]}, {}},
           (* bad argument patterns *)
            MatchQ[argTypeList, {"location", "date", _}],
            {{}, {argList[[3]] -> {"incspec"}}},
            MatchQ[argTypeList, {"location", _, _}],
            {{}, {argList[[2]] -> {"dtspec"}}},
            MatchQ[argTypeList, {"date", _, _}],
            {{},{argList[[2]] -> {"incspec"}}},
            MatchQ[argTypeList, {"locationDatePairs", _, _}],
            {{}, {argList[[2]] -> {"incspec"}}},
           (* identical types *)
           MatchQ[argTypeList, {"location", "location", _}],
            {{}, {argList[[2]] -> {"dtspec"}}},
           MatchQ[argTypeList, {"date", "date", _}],
            {{}, {argList[[1]] -> {"loc"}}},
           True,
            {{},{argList[[1]] -> {"locdate"}}}
            ]
    ];

(* now resolve the meaning of $ApplicationFunction depending on the other arguments *)
If[!MatchQ[$GeoLocation, {_?NumericQ..}|_GeoPosition]&&!FreeQ[expandedArguments, $Location], Union[AppendTo[messages, $GeoLocation->{"geoloc"}]]];
{expandedArguments, messages}/.{$Location :> (With[{tmp=$GeoLocation}, If[!MatchQ[tmp, {_?NumericQ..}|_GeoPosition], tmp, If[!MatchQ[tmp, {_?NumericQ..}|_GeoPosition], {40.1,-88.2}, tmp]]]/.GeoPosition[x_]:>x[[1;;2]]), $Date :> DateList[], $LocationQualifier -> "Everywhere"}

]

issueParserMessages[callingfunc_, metadata_] := Module[{},
  If[! MatchQ[metadata, {}],
   Function[{rule},
      With[{tmp = rule[[1]]}, 
       Function[{tag},
         Message[MessageName[callingfunc, tag], HoldForm[tmp]]] /@ 
        rule[[2]]]] /@ metadata;
   ]
  ]

CleanQ[expr_, other___] :=
	FreeQ[expr, Missing|$Failed|$CalculateFailed|other];

CorruptQ[expr_, other___] :=
	!CleanQ[expr, other]

astroToGMT[date_, intz_] := DateList[date, TimeZone -> $TimeZone - intz]

angleUnitQ[q_Quantity] := CompatibleUnitQ[q, Quantity[1, "AngularDegrees"]]
angleUnitQ[___] := False 

Clear[getCoordinates];

getCoordinates[loc_, n_Integer] := With[{coords = getCoordinates[loc]},
        If[VectorQ[coords, NumericQ] && Length[coords] >= n,
                Take[coords, n],
                coords
        ]    
];

getCoordinates[loc_Entity]:= Module[{polys, redpolys,geocode=EntityValue[loc, "Position"]/.gp_GeoPosition:>QuantityMagnitude[{Latitude[gp], Longitude[gp]}]},
	polys=Switch[loc[[1]], 
			"Country"|"CountryClass", CountryData[loc, "Polygon"],
			(*"River"|"RiverClass", RiverData[loc, "Shape"],*)
			"AdministrativeDivision"|"AdministrativeDivisionClass", AdministrativeDivisionData[loc, "Polygon"],
			"USState"|"USStateClass", AdministrativeDivisionData[loc, "Polygon"]/.{x_}:>x,
			"Ocean"|"OceanClass", OceanData[loc, "Polygon"],
			"City"|"CityClass", EntityValue[loc, "Position"]/.gp_GeoPosition:>QuantityMagnitude[{Latitude[gp], Longitude[gp]}],
			_, {}
		]/._Missing:>{};
	If[CleanQ[geocode]||!MatchQ[polys,{}],
	redpolys=If[!MatchQ[polys, {}], Cases[
		Graphics`Mesh`SimplifyPolyLine[#, VerticesGoal -> 10]&/@Cases[{polys}, _Polygon, Infinity], 
			{lat_?NumericQ, long_?NumericQ}:>{long, lat}, Infinity], polys];
	If[!MatchQ[redpolys, {}], 
		redpolys, 
		(Reverse@geocode)], 
	$Failed]]

getCoordinates[loc_EntityClass]:= Module[{polys, redpolys, geocode=EntityValue[loc, "Position"]/.gp_GeoPosition:>QuantityMagnitude[{Latitude[gp], Longitude[gp]}]},
	polys=Switch[loc[[1]], 
			"Country"|"CountryClass", CountryData[loc, "Polygon"],
			(*"River"|"RiverClass",  RiverData[loc, "Shape"],*)
			"AdministrativeDivision"|"AdministrativeDivisionClass", AdministrativeDivisionData[loc, "Polygon"],
			"USState"|"USStateClass", AdministrativeDivisionData[loc, "Polygon"]/.{x_}:>x,
			"Ocean"|"OceanClass", OceanData[loc, "Polygon"],
			"City"|"CityClass", (EntityValue[#, "Position"]/.gp_GeoPosition:>QuantityMagnitude[{Latitude[gp], Longitude[gp]}])&/@CityData[loc],
			_, {}
		]/._Missing:>{};
	If[CleanQ[geocode]||!MatchQ[polys,{}],
	redpolys=If[!MatchQ[polys, {}], Cases[
		Graphics`Mesh`SimplifyPolyLine[#, VerticesGoal -> 10]&/@Cases[{polys}, _Polygon, Infinity], 
			{lat_?NumericQ, long_?NumericQ}:>{long, lat}, Infinity], polys];
	If[!MatchQ[redpolys, {}], redpolys, Reverse@geocode], 
	$Failed]]

getCoordinates[loc_Polygon]:=Replace[Map[If[MatchQ[#, {_?angleUnitQ, _?angleUnitQ}], getCoordinates[#], #]&, Map[If[MatchQ[#, _Entity], Sequence@@getCoordinates[#], #] &, Cases[loc, {_?NumericQ, _?NumericQ} | {_?angleUnitQ, _?angleUnitQ}| _Entity, Infinity]]], {{long_?NumericQ, lat_?NumericQ}} :> {Mod[long, 360, -180], Mod[lat, -90, 90]}]
getCoordinates[loc_Line]:=Replace[Map[If[MatchQ[#, {_?angleUnitQ, _?angleUnitQ}], getCoordinates[#], #]&, Map[If[MatchQ[#, _Entity], Sequence@@getCoordinates[#], #] &, Cases[loc, {_?NumericQ, _?NumericQ} | {_?angleUnitQ, _?angleUnitQ}| _Entity, Infinity]]], {{long_?NumericQ, lat_?NumericQ}} :> {Mod[long, 360, -180], Mod[lat, -90, 90]}]
getCoordinates[loc_Point]:=Replace[Map[If[MatchQ[#, {_?angleUnitQ, _?angleUnitQ}], getCoordinates[#], #]&, Map[If[MatchQ[#, _Entity], Sequence@@getCoordinates[#], #] &, Cases[loc, {_?NumericQ, _?NumericQ} | {_?angleUnitQ, _?angleUnitQ}| _Entity, Infinity]]], {{long_?NumericQ, lat_?NumericQ}} :> {Mod[long, 360, -180], Mod[lat, -90, 90]}]
getCoordinates[loc_GeoPosition]:=With[{tmp=GeoPosition[loc[[1,1;;2]]]},
	Replace[
		Map[If[MatchQ[#, {_?angleUnitQ, _?angleUnitQ}], 
			getCoordinates[#], 
			#]&, 
			Map[
				If[MatchQ[#, _Entity], 
					Sequence@@getCoordinates[#], 
					#] &, 
				Cases[tmp, {_?NumericQ, _?NumericQ, ___?NumericQ} | {_?angleUnitQ, _?angleUnitQ}| _Entity, Infinity]
			]
		], {{lat_?NumericQ, long_?NumericQ, elm___?NumericQ}} :> {Mod[long, 360, -180], Mod[lat, -90, 90], elm}]]
(*getCoordinates[loc_GeoDisk]:=Replace[Map[If[MatchQ[#, {_?angleUnitQ, _?angleUnitQ}], getCoordinates[#], #]&, Map[If[MatchQ[#, _MWAEntity], Sequence@@getCoordinates[#], #] &, Cases[loc, {_?NumericQ, _?NumericQ} | {_?angleUnitQ, _?angleUnitQ}| _MWAEntity, Infinity]]], {{x_?NumericQ, y_?NumericQ}} :> {x, y}]
getCoordinates[loc_GeoArrow]:=Replace[Map[If[MatchQ[#, {_?angleUnitQ, _?angleUnitQ}], getCoordinates[#], #]&, Map[If[MatchQ[#, _MWAEntity], Sequence@@getCoordinates[#], #] &, Cases[loc, {_?NumericQ, _?NumericQ} | {_?angleUnitQ, _?angleUnitQ}| _MWAEntity, Infinity]]], {{x_?NumericQ, y_?NumericQ}} :> {x, y}]
getCoordinates[loc_GeodesicArrow]:=Replace[Map[If[MatchQ[#, {_?angleUnitQ, _?angleUnitQ}], getCoordinates[#], #]&, Map[If[MatchQ[#, _MWAEntity], Sequence@@getCoordinates[#], #] &, Cases[loc, {_?NumericQ, _?NumericQ} | {_?angleUnitQ, _?angleUnitQ}| _MWAEntity, Infinity]]], {{x_?NumericQ, y_?NumericQ}} :> {x, y}]
getCoordinates[loc_GeodesicLine]:=Replace[Map[If[MatchQ[#, {_?angleUnitQ, _?angleUnitQ}], getCoordinates[#], #]&, Map[If[MatchQ[#, _MWAEntity], Sequence@@getCoordinates[#], #] &, Cases[loc, {_?NumericQ, _?NumericQ} | {_?angleUnitQ, _?angleUnitQ}| _MWAEntity, Infinity]]], {{x_?NumericQ, y_?NumericQ}} :> {x, y}]
getCoordinates[loc_GeoMarker]:=Replace[Map[If[MatchQ[#, {_?angleUnitQ, _?angleUnitQ}], getCoordinates[#], #]&, Map[If[MatchQ[#, _MWAEntity], Sequence@@getCoordinates[#], #] &, Cases[loc, {_?NumericQ, _?NumericQ} | {_?angleUnitQ, _?angleUnitQ}| _MWAEntity, Infinity]]], {{x_?NumericQ, y_?NumericQ}} :> {x, y}]*)
getCoordinates[{a1_?angleUnitQ, a2_?angleUnitQ}]:={Mod[QuantityMagnitude[a1, "AngularDegrees"], 360, -180], Mod[QuantityMagnitude[a2, "AngularDegrees"], -90, 90]}
getCoordinates[{long_?NumericQ, lat_?NumericQ, elm___?NumericQ}]:={long, lat, elm}
getCoordinates[loc_]:=Reverse@(EntityValue[loc, "Position"]/.gp_GeoPosition:>{Latitude[gp], Longitude[gp]})

Options[dateObjectToDateList] = {TimeZone :> $TimeZone}

dateObjectToDateList[d_DateObject, OptionsPattern[]]:=Module[{tz},
	tz=OptionValue[TimeZone];
	DateList[d, TimeZone -> tz]
	];

dateObjectToDateList[d_?CorruptQ, OptionsPattern[]]:= Missing["NotAvailable"];
	
dateObjectToDateList[d_,OptionsPattern[]]:=d;

Clear[getTimeZone];
getTimeZone[{n1_?NumericQ, n2_?NumericQ}]:= If[MatchQ[$GeoLocation, GeoPosition[{n1, n2}]], $TimeZone, LocalTimeZone[GeoPosition[{n1, n2}]]]
getTimeZone[loc_]:= LocalTimeZone[loc]

solarRiseTimeAndCheckForPolarM10[coords_, gmttime_, eventincr_, tz_] := Module[{nextrise, msun, pnight},
  If[CleanQ[coords],
   nextrise = Replace[PlanetaryAstronomy`Private`RiseSetsX["Sun", gmttime, coords, -1, 1/1440,"Number"->1], lst_List:>(lst+{0,0,0,tz,0,0})[[;;5]]];
   If[CleanQ[nextrise],
    nextrise, 
    msun = PlanetaryAstronomy`Private`MidnightSun3[gmttime, coords, -1, 1/1440, "Number" -> 0][[1, All]];
    pnight = PlanetaryAstronomy`Private`MidnightSun3[gmttime, coords, 1, 1/1440, "Number" -> 0][[1, All]];
    Which[
    	IntervalMemberQ[Interval[AbsoluteTime/@msun], AbsoluteTime@gmttime], 
    	If[eventincr===1, 
    		Replace[PlanetaryAstronomy`Private`RiseSetsX["Sun", msun[[2]], coords, -1, 1/1440,"Number"->1], lst_List:>(lst+{0,0,0,tz,0,0})[[;;5]]], 
    		Replace[PlanetaryAstronomy`Private`RiseSetsX["Sun", msun[[1]], coords, -1, 1/1440,"Number"->1], lst_List:>(lst+{0,0,0,tz,0,0})[[;;5]]]
    		],
    	IntervalMemberQ[Interval[AbsoluteTime/@pnight], AbsoluteTime@gmttime], 
    	If[eventincr===1, 
    		Replace[PlanetaryAstronomy`Private`RiseSetsX["Sun", pnight[[2]], coords, -1, 1/1440,"Number"->1], lst_List:>(lst+{0,0,0,tz,0,0})[[;;5]]], 
    		Replace[PlanetaryAstronomy`Private`RiseSetsX["Sun", pnight[[1]]-{0,0,1,0,0,0}, coords, -1, 1/1440,"Number"->1], lst_List:>(lst+{0,0,0,tz,0,0})[[;;5]]]
    		],
    	True, Missing["NotAvailable"]
    ]], Missing["NotAvailable"]]]

solarSetTimeAndCheckForPolarM10[coords_, gmttime_, eventincr_, tz_] := Module[{nextset, msun, pnight},
  If[CleanQ[longlat],
   nextset = Replace[PlanetaryAstronomy`Private`RiseSetsX["Sun", gmttime, coords, 1, 1/1440,"Number"->1], lst_List:>(lst+{0,0,0,tz,0,0})[[;;5]]];
   If[CleanQ[nextset],
    nextset, 
    msun = PlanetaryAstronomy`Private`MidnightSun3[gmttime, coords, -1, 1/1440, "Number" -> 0][[1, All]];
    pnight = PlanetaryAstronomy`Private`MidnightSun3[gmttime, coords, 1, 1/1440, "Number" -> 0][[1, All]];
    Which[
    	IntervalMemberQ[Interval[AbsoluteTime/@msun], AbsoluteTime@gmttime], 
    	If[eventincr===1, 
    		Replace[PlanetaryAstronomy`Private`RiseSetsX["Sun", msun[[2]], coords, 1, 1/1440,"Number"->1], lst_List:>(lst+{0,0,0,tz,0,0})[[;;5]]], 
    		Replace[PlanetaryAstronomy`Private`RiseSetsX["Sun", msun[[1]]-{0,0,1,0,0,0}, coords, 1, 1/1440,"Number"->1], lst_List:>(lst+{0,0,0,tz,0,0})[[;;5]]]
    		],
    	IntervalMemberQ[Interval[AbsoluteTime/@pnight], AbsoluteTime@gmttime], 
    	If[eventincr===1, 
    		Replace[PlanetaryAstronomy`Private`RiseSetsX["Sun", pnight[[2]], coords, 1, 1/1440,"Number"->1], lst_List:>(lst+{0,0,0,tz,0,0})[[;;5]]], 
    		Replace[PlanetaryAstronomy`Private`RiseSetsX["Sun", pnight[[1]], coords, 1, 1/1440,"Number"->1], lst_List:>(lst+{0,0,0,tz,0,0})[[;;5]]]
    		],
    	True, Missing["NotAvailable"]
    ]], Missing["NotAvailable"]]]

getIntervalResult[vals_, iv_, loc_]:=Switch[iv,
	        Automatic, If[Length[Dimensions[vals]]>1, 
	        	Interval[{vals[[1]], vals[[-1]]}], 
	        	vals],
			Interval, If[Length[Dimensions[vals]]>1,
				Interval[{vals[[1]], vals[[-1]]}],
				Interval[{vals, vals}]],
			Mean, If[Length[Dimensions[vals]]>1,
				DateList[Mean[AbsoluteTime/@vals]],
				vals],
			Min, If[Length[Dimensions[vals]]>1,
				vals[[1]],
				vals],
			Max, If[Length[Dimensions[vals]]>1,
				vals[[-1]],
				vals],
			StandardDeviation, If[Length[Dimensions[vals]]>1,
				Quantity[StandardDeviation[AbsoluteTime/@vals], "Seconds"],
				Quantity[0, "Seconds"]],
			_, $Failed
		]

myQuantityStandardDeviation[list_List] := 
 Sqrt[Total[(list - Mean[list])^2/(Length[list] - 1)]]

getIntervalResultForQuantityPairs[{vals1_, vals2_}, iv_, loc_]:=Switch[iv,
	        Automatic, If[ListQ[vals1], 
				With[{qpvals1={QuantityMagnitude[vals1[[1]]], QuantityMagnitude[vals1[[-1]]]}, qpvals2={QuantityMagnitude[vals2[[1]]], QuantityMagnitude[vals2[[-1]]]}, upvals1=QuantityUnit[vals1[[1]]], upvals2=QuantityUnit[vals2[[1]]]},
				{Quantity[Interval@qpvals1, upvals1], Quantity[Interval@qpvals2, upvals2]}],
				{vals1, vals2}
				],
			Interval, If[ListQ[vals1], 
				With[{qpvals1={QuantityMagnitude[vals1[[1]]], QuantityMagnitude[vals1[[-1]]]}, qpvals2={QuantityMagnitude[vals2[[1]]], QuantityMagnitude[vals2[[-1]]]}, upvals1=QuantityUnit[vals1[[1]]], upvals2=QuantityUnit[vals2[[1]]]},
				{Quantity[Interval@qpvals1, upvals1], Quantity[Interval@qpvals2, upvals2]}],
				With[{qpvals1={QuantityMagnitude[vals1], QuantityMagnitude[vals1]}, qpvals2={QuantityMagnitude[vals2], QuantityMagnitude[vals2]}, upvals1=QuantityUnit[vals1],upvals2=QuantityUnit[vals2]},
				{Quantity[Interval@qpvals1, upvals1], Quantity[Interval@qpvals2, upvals2]}]
				],
			StandardDeviation, If[ListQ[vals1],
				{myQuantityStandardDeviation[vals1], myQuantityStandardDeviation[vals2]},
				{Quantity[0, "Seconds"], Quantity[0, "Seconds"]}],
			Mean, If[ListQ[vals1],
				{Mean[vals1], Mean[vals2]},
				{vals1, vals2}],
			Min, If[ListQ[vals1],
				{vals1[[1]], vals2[[1]]},
				{vals1, vals2}],
			Max, If[ListQ[vals1],
				{vals1[[-1]], vals2[[-1]]},
				{vals1, vals2}],
			_, $Failed
		]

getIntervalResultForQuantity[vals_, iv_, loc_]:=Switch[iv,
	        Automatic, If[ListQ[vals], 
				With[{qpvals={QuantityMagnitude[vals[[1]]], QuantityMagnitude[vals[[-1]]]}, upvals=QuantityUnit[vals[[1]]]},
				Quantity[Interval@qpvals, upvals]],
				vals
				],
			Interval, If[ListQ[vals], 
				With[{qpvals={QuantityMagnitude[vals[[1]]], QuantityMagnitude[vals[[-1]]]}, upvals=QuantityUnit[vals[[1]]]},
				Quantity[Interval@qpvals, upvals]],
				With[{qpvals={QuantityMagnitude[vals], QuantityMagnitude[vals]}, upvals=QuantityUnit[vals]},
				Quantity[Interval@qpvals, upvals]]
				],
			StandardDeviation, If[ListQ[vals],
				myQuantityStandardDeviation[vals],
				Quantity[0, "SecondsOfRightAscension"]],
			Mean, If[ListQ[vals],
				Mean[vals],
				vals],
			Min, If[ListQ[vals],
				vals[[1]],
				vals],
			Max, If[ListQ[vals],
				vals[[-1]],
				vals],
			_, $Failed
		]

(* astronomy convenience functions *)
Clear[AstronomyConvenienceFunction];
(* Sunrise *)
AstronomyConvenienceFunction["Sunrise", loc_?(validLocationQ[#, Sunrise]&), date:{_?NumericQ..}, utz_, tz_, ugeoloc_, eventincr_, intervalvalue_]:= Module[{idate, coords, coorddata, rangeres,res},
	idate = If[eventincr===1, astroToGMT[date, utz], DateList[astroToGMT[date, utz]-{0,0,1,0,0,0}]];
	coords = getCoordinates[loc];
	If[CleanQ[coords]&&Length[Dimensions[coords]]>1, 
		rangeres=Sort[Select[solarRiseTimeAndCheckForPolarM10[#, idate, eventincr, tz]&/@(coords),CleanQ[#]&]];
		getIntervalResult[rangeres, intervalvalue, loc],
		coorddata = If[MatchQ[coords, {_?NumericQ, _?NumericQ, ___?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
		If[!MatchQ[coorddata, {_?NumericQ, _?NumericQ, ___?NumericQ}], Return[Missing["InvalidCoordinates"]]];
		res=solarRiseTimeAndCheckForPolarM10[coorddata, idate, eventincr, tz];
		getIntervalResult[res, intervalvalue, loc]
	]
 ]

AstronomyConvenienceFunction["Sunrise", loc_?(validLocationQ[#, Sunrise]&), date:{{_?NumericQ..}..}, utz_, tz_, ugeoloc_, eventincr_, intervalvalue_]:= Module[{res, idates, coords, coorddata, rangeres},
	idates = If[eventincr===1, astroToGMT[#, utz], DateList[astroToGMT[#, utz]-{0,0,1,0,0,0}]]&/@date;
	coords = getCoordinates[loc];
	If[CleanQ[coords]&&Length[Dimensions[coords]]>1, 
		rangeres=Function[{idate}, Sort[Select[solarRiseTimeAndCheckForPolarM10[#, idate, eventincr, tz]&/@(coords),CleanQ[#]&]]]/@idates;
		DeleteDuplicates[Transpose[{date,getIntervalResult[#, intervalvalue, loc]&/@rangeres}]],
		coorddata = If[MatchQ[coords, {_?NumericQ, _?NumericQ, ___?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
		If[!MatchQ[coorddata, {_?NumericQ, _?NumericQ, ___?NumericQ}], Return[Missing["InvalidCoordinates"]]];
		res=solarRiseTimeAndCheckForPolarM10[coorddata, #, eventincr, tz]&/@idates;
		DeleteDuplicates[Transpose[{date,getIntervalResult[#, intervalvalue, loc]&/@res}]]]
 ];

AstronomyConvenienceFunction["Sunrise", loc:{(_?(validLocationQ[#, Sunrise]&))..}, date:{_?NumericQ..}, utz_, tz_, ugeoloc_, eventincr_, intervalvalue_]:= Module[{idate, coords, coorddata, rangeres, res},
	idate = If[eventincr===1, astroToGMT[date, utz], DateList[astroToGMT[date, utz]-{0,0,1,0,0,0}]];
	Function[{location},
		coords = getCoordinates[location];
		If[CleanQ[coords]&&Length[Dimensions[coords]]>1, 
		rangeres=Sort[Select[solarRiseTimeAndCheckForPolarM10[#, idate, eventincr, tz]&/@(coords),CleanQ[#]&]];
		getIntervalResult[rangeres, intervalvalue, location],
		coorddata = If[MatchQ[coords, {_?NumericQ, _?NumericQ, ___?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
		If[!MatchQ[coorddata, {_?NumericQ, _?NumericQ, ___?NumericQ}], Return[Missing["InvalidCoordinates"]]];
		res=solarRiseTimeAndCheckForPolarM10[coorddata, idate, eventincr, tz];
		getIntervalResult[res, intervalvalue, location]
	]]/@loc
 ];

AstronomyConvenienceFunction["Sunrise", loc:{(_?(validLocationQ[#, Sunrise]&))..}, date:{{_?NumericQ..}..}, utz_, tz_, ugeoloc_, eventincr_, intervalvalue_]:= Module[{res, idates, coords, coorddata, rangeres},
	idates = If[eventincr===1, astroToGMT[#, utz], DateList[astroToGMT[#, utz]-{0,0,1,0,0,0}]]&/@date;
	Function[{location},
		coords = getCoordinates[location];
		If[CleanQ[coords]&&Length[Dimensions[coords]]>1, 
		rangeres=Function[{idate}, Sort[Select[solarRiseTimeAndCheckForPolarM10[#, idate, eventincr, tz]&/@(coords),CleanQ[#]&]]]/@idates;
		DeleteDuplicates[Transpose[{date,getIntervalResult[#, intervalvalue, location]&/@rangeres}]],
		coorddata = If[MatchQ[coords, {_?NumericQ, _?NumericQ, ___?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
		If[!MatchQ[coorddata, {_?NumericQ, _?NumericQ, ___?NumericQ}], Return[Missing["InvalidCoordinates"]]];
		res=solarRiseTimeAndCheckForPolarM10[coorddata, #, eventincr, tz]&/@idates;
		DeleteDuplicates[Transpose[{date,getIntervalResult[#, intervalvalue, location]&/@res}]]]]/@loc
 ]

AstronomyConvenienceFunction["Sunrise", locdatepairs:{{(_?(validLocationQ[#, Sunrise]&)),{_?NumericQ..}}..}, utz_, tz_, ugeoloc_, eventincr_, intervalvalue_]:= Module[{idate, coords, coorddata, rangeres, res},
	Function[{pair},
		idate = If[eventincr===1, astroToGMT[pair[[2]], utz], DateList[astroToGMT[pair[[2]], utz]-{0,0,1,0,0,0}]];
		coords = getCoordinates[pair[[1]]];
		If[CleanQ[coords]&&Length[Dimensions[coords]]>1, 
			rangeres=Sort[Select[solarRiseTimeAndCheckForPolarM10[#, idate, eventincr, tz]&/@(coords),CleanQ[#]&]];
			getIntervalResult[rangeres, intervalvalue, pair[[1]]],
			coorddata = If[MatchQ[coords, {_?NumericQ, _?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
			If[!MatchQ[coorddata, {_?NumericQ, _?NumericQ, ___?NumericQ}], Return[Missing["InvalidCoordinates"]]];
			res=solarRiseTimeAndCheckForPolarM10[coorddata, idate, eventincr, tz];
			getIntervalResult[res, intervalvalue, pair[[1]]]
	]]/@locdatepairs
 ]

(* Sunset *)
AstronomyConvenienceFunction["Sunset", loc_?(validLocationQ[#, Sunset]&), date:{_?NumericQ..}, utz_, tz_, ugeoloc_, eventincr_, intervalvalue_]:= Module[{idate, coords, coorddata, rangeres,res},
	idate = If[eventincr===1, astroToGMT[date, utz], DateList[astroToGMT[date, utz]-{0,0,1,0,0,0}]];
	coords = getCoordinates[loc];
	If[CleanQ[coords]&&Length[Dimensions[coords]]>1, 
		rangeres=Sort[Select[solarSetTimeAndCheckForPolarM10[#, idate, eventincr, tz]&/@(coords),CleanQ[#]&]];
		getIntervalResult[rangeres, intervalvalue, loc],
		coorddata = If[MatchQ[coords, {_?NumericQ, _?NumericQ, ___?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
		If[!MatchQ[coorddata, {_?NumericQ, _?NumericQ, ___?NumericQ}], Return[Missing["InvalidCoordinates"]]];
		res=solarSetTimeAndCheckForPolarM10[coorddata, idate, eventincr, tz];
		getIntervalResult[res, intervalvalue, loc]
	]
 ];

AstronomyConvenienceFunction["Sunset", loc_?(validLocationQ[#, Sunset]&), date:{{_?NumericQ..}..}, utz_, tz_, ugeoloc_, eventincr_, intervalvalue_]:= Module[{res, idates, coords, coorddata, rangeres},
	idates = If[eventincr===1, astroToGMT[#, utz], DateList[astroToGMT[#, utz]-{0,0,1,0,0,0}]]&/@date;
	coords = getCoordinates[loc];
	If[CleanQ[coords]&&Length[Dimensions[coords]]>1, 
		rangeres=Function[{idate}, Sort[Select[solarSetTimeAndCheckForPolarM10[#, idate, eventincr, tz]&/@(Reverse/@coords),CleanQ[#]&]]]/@idates;
		DeleteDuplicates[Transpose[{date,getIntervalResult[#, intervalvalue, loc]&/@rangeres}]],
		coorddata = If[MatchQ[coords, {_?NumericQ, _?NumericQ, ___?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
		If[!MatchQ[coorddata, {_?NumericQ, _?NumericQ, ___?NumericQ}], Return[Missing["InvalidCoordinates"]]];
		res=solarSetTimeAndCheckForPolarM10[coorddata, #, eventincr, tz]&/@idates;
		DeleteDuplicates[Transpose[{date,getIntervalResult[#, intervalvalue, loc]&/@res}]]]
 ];

AstronomyConvenienceFunction["Sunset", loc:{(_?(validLocationQ[#, Sunset]&))..}, date:{_?NumericQ..}, utz_, tz_, ugeoloc_, eventincr_, intervalvalue_]:= Module[{idate, coords, coorddata, rangeres, res},
	idate = If[eventincr===1, astroToGMT[date, utz], DateList[astroToGMT[date, utz]-{0,0,1,0,0,0}]];
	Function[{location},
		coords = getCoordinates[location];
		If[CleanQ[coords]&&Length[Dimensions[coords]]>1, 
		rangeres=Sort[Select[solarSetTimeAndCheckForPolarM10[#, idate, eventincr, tz]&/@(coords),CleanQ[#]&]];
		getIntervalResult[rangeres, intervalvalue, location],
		coorddata = If[MatchQ[coords, {_?NumericQ, _?NumericQ, ___?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
		If[!MatchQ[coorddata, {_?NumericQ, _?NumericQ, ___?NumericQ}], Return[Missing["InvalidCoordinates"]]];
		res=solarSetTimeAndCheckForPolarM10[coorddata, idate, eventincr, tz];
		getIntervalResult[res, intervalvalue, location]
	]]/@loc
 ];

AstronomyConvenienceFunction["Sunset", loc:{(_?(validLocationQ[#, Sunset]&))..}, date:{{_?NumericQ..}..}, utz_, tz_, ugeoloc_, eventincr_, intervalvalue_]:= Module[{res, idates, coords, coorddata, rangeres},
	idates = If[eventincr===1, astroToGMT[#, utz], DateList[astroToGMT[#, utz]-{0,0,1,0,0,0}]]&/@date;
	Function[{location},
		coords = getCoordinates[location];
		If[CleanQ[coords]&&Length[Dimensions[coords]]>1, 
		rangeres=Function[{idate}, Sort[Select[solarSetTimeAndCheckForPolarM10[#, idate, eventincr, tz]&/@(coords),CleanQ[#]&]]]/@idates;
		DeleteDuplicates[Transpose[{date,getIntervalResult[#, intervalvalue, location]&/@rangeres}]],
		coorddata = If[MatchQ[coords, {_?NumericQ, _?NumericQ, ___?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
		If[!MatchQ[coorddata, {_?NumericQ, _?NumericQ, ___?NumericQ}], Return[Missing["InvalidCoordinates"]]];
		res=solarSetTimeAndCheckForPolarM10[coorddata, #, eventincr, tz]&/@idates;
		DeleteDuplicates[Transpose[{date,getIntervalResult[#, intervalvalue, location]&/@res}]]]]/@loc
 ]

AstronomyConvenienceFunction["Sunset", locdatepairs:{{(_?(validLocationQ[#, Sunset]&)),{_?NumericQ..}}..}, utz_, tz_, ugeoloc_, eventincr_, intervalvalue_]:= Module[{idate, coords, coorddata, rangeres, res},
	Function[{pair},
		idate = If[eventincr===1, astroToGMT[pair[[2]], utz], DateList[astroToGMT[pair[[2]], utz]-{0,0,1,0,0,0}]];
		coords = getCoordinates[pair[[1]]];
		If[CleanQ[coords]&&Length[Dimensions[coords]]>1, 
			rangeres=Sort[Select[solarSetTimeAndCheckForPolarM10[#, idate, eventincr, tz]&/@(coords),CleanQ[#]&]];
			getIntervalResult[rangeres, intervalvalue, pair[[1]]],
			coorddata = If[MatchQ[coords, {_?NumericQ, _?NumericQ, ___?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
			If[!MatchQ[coorddata, {_?NumericQ, _?NumericQ, ___?NumericQ}], Return[Missing["InvalidCoordinates"]]];
			res=solarSetTimeAndCheckForPolarM10[coorddata, idate, eventincr, tz];
			getIntervalResult[res, intervalvalue, pair[[1]]]
	]]/@locdatepairs
 ];

(* SunPosition *)
AstronomyConvenienceFunction["SunPosition", loc_?(validLocationQ[#, SunPosition]&), date:{_?NumericQ..}, coordsys_, utz_, tz_, ugeoloc_, intervalvalue_, altm_]:= Module[{jd, az, alt, ra, dec, long, lat, coords, rarange, decrange, altrange, azrange, dims},
	jd = PlanetaryAstronomy`Private`JDU[astroToGMT[date, utz]];
	(* Since getCoordinates may contain elevation data, strip it out for now until PACoordinatesX, etc. *)
	(* can be made to support more than just latitude and longitude arguments *)
	coords = getCoordinates[loc, 2];
	dims = Dimensions[coords];
	If[Length[dims]>1,
		If[coordsys === "Equatorial",
			{rarange, decrange} = Function[lst, Sort[Select[lst, CleanQ[#]&]]]/@Transpose[(
				Function[pacx, {
					Replace[radiansToHours[pacx[[3, 1]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "HoursOfRightAscension"]], 
					Replace[radiansToDegrees[pacx[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "AngularDegrees"]]
				}]@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Sun", {"EarthEquatorialApparentTopo", {degreesToRadians[#[[1]]], degreesToRadians[#[[2]]], 0}}])&/@coords];
			getIntervalResultForQuantityPairs[{rarange, decrange}, intervalvalue, loc]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]},
			{azrange, altrange} = Function[lst, Sort[Select[lst, CleanQ[#]&]]]/@Transpose[(
				Function[pacx, {
					Replace[Mod[180 + radiansToDegrees[pacx[[3, 1]]], 360], n_?NumericQ :> Quantity[SetAccuracy[n, 3], "AngularDegrees"]],
					Replace[radiansToDegrees[pacx[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[If[MatchQ[altm, "ApparentAltitude"], n + (1.02 Cot[(n + 10.3/(n + 5.11)) Degree])/60., n], 3], "AngularDegrees"]]
				}]@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Sun", {"EarthApparent", {degreesToRadians[#[[1]]], degreesToRadians[#[[2]]], 0}}])&/@coords];
			getIntervalResultForQuantityPairs[{azrange, altrange}, intervalvalue, loc]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}
		],
		{long, lat} = If[MatchQ[coords, {_?NumericQ, _?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
		If[!MatchQ[{long, lat}, {_?NumericQ, _?NumericQ}]&&!MatchQ[coordsys, "Equatorial"], Return[Missing["InvalidCoordinates"]]];
		If[coordsys === "Equatorial", 
			{ra, dec} = {
					Replace[radiansToHours[#[[3, 1]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "HoursOfRightAscension"]], 
					Replace[radiansToDegrees[#[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "AngularDegrees"]]
					} &@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Sun", {"EarthEquatorialApparentTopo", {long*Degree, lat*Degree, 0}}];
			getIntervalResultForQuantityPairs[{ra, dec}, intervalvalue, loc]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]},
			{az, alt} = {
					Replace[Mod[180 + radiansToDegrees[#[[3, 1]]], 360], n_?NumericQ :> Quantity[SetAccuracy[n, 3], "AngularDegrees"]], 
					Replace[radiansToDegrees[#[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[If[MatchQ[altm, "ApparentAltitude"], n + (1.02 Cot[(n + 10.3/(n + 5.11)) Degree])/60., n], 3], "AngularDegrees"]]
				} &@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Sun", {"EarthApparent", {degreesToRadians[long], degreesToRadians[lat], 0}}];
			getIntervalResultForQuantityPairs[{az, alt}, intervalvalue, loc]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}
		]
	]
	];
	
AstronomyConvenienceFunction["SunPosition", loc_?(validLocationQ[#, SunPosition]&), date:{{_?NumericQ..}..}, coordsys_, utz_, tz_, ugeoloc_, intervalvalue_, altm_]:= Module[{az, alt, ra, dec, long, lat, coords, rarange, decrange, azrange, altrange, dims},
	(* Since getCoordinates may contain elevation data, strip it out for now until PACoordinatesX, etc. *)
	(* can be made to support more than just latitude and longitude arguments *)
	coords = getCoordinates[loc, 2];
	dims = Dimensions[coords];
	If[Length[dims]>1, 
		If[coordsys === "Equatorial", 
			{rarange, decrange} = Transpose[Function[{idate},
				Function[lst, Sort[Select[lst, CleanQ[#]&]]]/@Transpose[(
				Function[pacx, {
					Replace[radiansToHours[pacx[[3, 1]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "HoursOfRightAscension"]], 
					Replace[radiansToDegrees[pacx[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "AngularDegrees"]]
				}]@ PlanetaryAstronomy`Private`PACoordinatesX[PlanetaryAstronomy`Private`JDU[astroToGMT[idate, utz]], "Sun", {"EarthEquatorialApparentTopo", {#[[1]]*Degree, #[[2]]*Degree, 0}}])&/@coords]]/@date];
			DeleteDuplicates[Transpose[{date, (getIntervalResultForQuantityPairs[#, intervalvalue, loc]&/@Transpose[{rarange, decrange}])/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}}]],
			{azrange, altrange} = Transpose[Function[{idate},
				Function[lst, Sort[Select[lst, CleanQ[#]&]]]/@Transpose[(
				Function[pacx, {
					Replace[Mod[180 + radiansToDegrees[pacx[[3, 1]]], 360], n_?NumericQ :> Quantity[SetAccuracy[n, 3], "AngularDegrees"]],
					Replace[radiansToDegrees[pacx[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[If[MatchQ[altm, "ApparentAltitude"], n + (1.02 Cot[(n + 10.3/(n + 5.11)) Degree])/60., n], 3], "AngularDegrees"]]
				}]@ PlanetaryAstronomy`Private`PACoordinatesX[PlanetaryAstronomy`Private`JDU[astroToGMT[idate, utz]], "Sun", {"EarthApparent", {#[[1]]*Degree, #[[2]]*Degree, 0}}])&/@coords]]/@date];
			DeleteDuplicates[Transpose[{date, ((getIntervalResultForQuantityPairs[#, intervalvalue, loc])&/@Transpose[{azrange,altrange}])}]]],
		{long, lat} = If[MatchQ[coords, {_?NumericQ, _?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
		If[!MatchQ[{long, lat}, {_?NumericQ, _?NumericQ}]&&!MatchQ[coordsys, "Equatorial"], Return[Missing["InvalidCoordinates"]]];
		If[coordsys === "Equatorial",
			{ra, dec} = Transpose[({
					Replace[radiansToHours[#[[3, 1]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "HoursOfRightAscension"]], 
					Replace[radiansToDegrees[#[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "AngularDegrees"]]
					} &@ PlanetaryAstronomy`Private`PACoordinatesX[PlanetaryAstronomy`Private`JDU[astroToGMT[#, utz]], "Sun", {"EarthEquatorialApparentTopo", {long*Degree, lat*Degree, 0}}])&/@date]; 
			DeleteDuplicates[Transpose[{date, (getIntervalResultForQuantityPairs[#, intervalvalue, loc]&/@Transpose[{ra, dec}])/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}}]],
			{az, alt} = Transpose[({
					Replace[Mod[180 + radiansToDegrees[#[[3, 1]]], 360], n_?NumericQ :> Quantity[SetAccuracy[n, 3], "AngularDegrees"]], 
					Replace[radiansToDegrees[#[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[If[MatchQ[altm, "ApparentAltitude"], n + (1.02 Cot[(n + 10.3/(n + 5.11)) Degree])/60., n], 3], "AngularDegrees"]]
				} &@ PlanetaryAstronomy`Private`PACoordinatesX[PlanetaryAstronomy`Private`JDU[astroToGMT[#, utz]], "Sun", {"EarthApparent", {long*Degree, lat*Degree, 0}}])&/@date]; 
			DeleteDuplicates[Transpose[{date, (getIntervalResultForQuantityPairs[#, intervalvalue, loc]&/@Transpose[{az, alt}])/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}}]]
			]
		]
	
	];
	
AstronomyConvenienceFunction["SunPosition", locs:{(_?(validLocationQ[#, SunPosition]&))..}, date:{_?NumericQ..}, coordsys_, utz_, tz_, ugeoloc_, intervalvalue_, altm_]:= Module[{jd, az, alt, ra, dec, long, lat, coords, rarange, decrange, altrange, azrange, dims},
	jd = PlanetaryAstronomy`Private`JDU[astroToGMT[date, utz]];
	Function[{location},
	(* Since getCoordinates may contain elevation data, strip it out for now until PACoordinatesX, etc. *)
	(* can be made to support more than just latitude and longitude arguments *)
	coords = getCoordinates[location, 2];
		dims = Dimensions[coords];
		If[Length[dims]>1,
			If[coordsys === "Equatorial",
				{rarange, decrange} = Function[lst, Sort[Select[lst, CleanQ[#]&]]]/@Transpose[(
					Function[pacx, {
						Replace[radiansToHours[pacx[[3, 1]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "HoursOfRightAscension"]], 
						Replace[radiansToDegrees[pacx[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "AngularDegrees"]]
						}]@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Sun", {"EarthEquatorialApparentTopo", {#[[1]]*Degree, #[[2]]*Degree, 0}}])&/@coords];
				getIntervalResultForQuantityPairs[{rarange, decrange}, intervalvalue, location]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]},
				{azrange, altrange} = Function[lst, Sort[Select[lst, CleanQ[#]&]]]/@Transpose[(
					Function[pacx, {
						Replace[Mod[180 + radiansToDegrees[pacx[[3, 1]]], 360], n_?NumericQ :> Quantity[SetAccuracy[n, 3], "AngularDegrees"]],
						Replace[radiansToDegrees[pacx[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[If[MatchQ[altm, "ApparentAltitude"], n + (1.02 Cot[(n + 10.3/(n + 5.11)) Degree])/60., n], 3], "AngularDegrees"]]
					}]@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Sun", {"EarthApparent", {#[[1]]*Degree, #[[2]]*Degree, 0}}])&/@coords];
				getIntervalResultForQuantityPairs[{azrange, altrange}, intervalvalue, location]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}
			],
			{long, lat} = If[MatchQ[coords, {_?NumericQ, _?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
			If[!MatchQ[{long, lat}, {_?NumericQ, _?NumericQ}]&&!MatchQ[coordsys, "Equatorial"], Return[Missing["InvalidCoordinates"]]];
			If[coordsys === "Equatorial", 
				{ra, dec} = {
					Replace[radiansToHours[#[[3, 1]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "HoursOfRightAscension"]], 
					Replace[radiansToDegrees[#[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "AngularDegrees"]]
					} &@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Sun", {"EarthEquatorialApparentTopo", {long*Degree, lat*Degree, 0}}];
				getIntervalResultForQuantityPairs[{ra, dec}, intervalvalue, location]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]},
				{az, alt} = {
					Replace[Mod[180 + radiansToDegrees[#[[3, 1]]], 360], n_?NumericQ :> Quantity[SetAccuracy[n, 3], "AngularDegrees"]], 
					Replace[radiansToDegrees[#[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[If[MatchQ[altm, "ApparentAltitude"], n + (1.02 Cot[(n + 10.3/(n + 5.11)) Degree])/60., n], 3], "AngularDegrees"]]
				} &@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Sun", {"EarthApparent", {long*Degree, lat*Degree, 0}}];
				getIntervalResultForQuantityPairs[{az, alt}, intervalvalue, location]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}
			]
		]
	]/@locs
	];

AstronomyConvenienceFunction["SunPosition", loc:{(_?(validLocationQ[#, SunPosition]&))..}, date:{{_?NumericQ..}..}, coordsys_, utz_, tz_, ugeoloc_, intervalvalue_, altm_]:= Module[{az, alt, ra, dec, long, lat, coords, rarange, decrange, azrange, altrange, dims},
	Function[{location},
		(* Since getCoordinates may contain elevation data, strip it out for now until PACoordinatesX, etc. *)
		(* can be made to support more than just latitude and longitude arguments *)
		coords = getCoordinates[location, 2];
		dims = Dimensions[coords];
		If[Length[dims]>1, 
			If[coordsys === "Equatorial", 
				{rarange, decrange} = Transpose[Function[{idate},
					Function[lst, Sort[Select[lst, CleanQ[#]&]]]/@Transpose[(
						Function[pacx, {
							Replace[radiansToHours[pacx[[3, 1]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "HoursOfRightAscension"]], 
							Replace[radiansToDegrees[pacx[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "AngularDegrees"]]
						}]@ PlanetaryAstronomy`Private`PACoordinatesX[PlanetaryAstronomy`Private`JDU[astroToGMT[idate, utz]], "Sun", {"EarthEquatorialApparentTopo", {#[[1]]*Degree, #[[2]]*Degree, 0}}])&/@coords]]/@date];
				DeleteDuplicates[Transpose[{date, (getIntervalResultForQuantityPairs[#, intervalvalue, location]&/@Transpose[{rarange, decrange}])/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}}]],
				{azrange, altrange} = Transpose[Function[{idate},
					Function[lst, Sort[Select[lst, CleanQ[#]&]]]/@Transpose[(
						Function[pacx, {
							Replace[Mod[180 + radiansToDegrees[pacx[[3, 1]]], 360], n_?NumericQ :> Quantity[SetAccuracy[n, 3], "AngularDegrees"]],
							Replace[radiansToDegrees[pacx[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[If[MatchQ[altm, "ApparentAltitude"], n + (1.02 Cot[(n + 10.3/(n + 5.11)) Degree])/60., n], 3], "AngularDegrees"]]
						}]@ PlanetaryAstronomy`Private`PACoordinatesX[PlanetaryAstronomy`Private`JDU[astroToGMT[idate, utz]], "Sun", {"EarthApparent", {#[[1]]*Degree, #[[2]]*Degree, 0}}])&/@coords]]/@date];
				DeleteDuplicates[Transpose[{date, ((getIntervalResultForQuantityPairs[#, intervalvalue, location])&/@Transpose[{azrange,altrange}])}]]],
				{long, lat} = If[MatchQ[coords, {_?NumericQ, _?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
				If[!MatchQ[{long, lat}, {_?NumericQ, _?NumericQ}]&&!MatchQ[coordsys, "Equatorial"], Return[Missing["InvalidCoordinates"]]];
				If[coordsys === "Equatorial",
					{ra, dec} = Transpose[({
						Replace[radiansToHours[#[[3, 1]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "HoursOfRightAscension"]], 
						Replace[radiansToDegrees[#[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "AngularDegrees"]]
					} &@ PlanetaryAstronomy`Private`PACoordinatesX[PlanetaryAstronomy`Private`JDU[astroToGMT[#, utz]], "Sun", {"EarthEquatorialApparentTopo", {long*Degree, lat*Degree, 0}}])&/@date]; 
					DeleteDuplicates[Transpose[{date, (getIntervalResultForQuantityPairs[#, intervalvalue, location]&/@Transpose[{ra, dec}])/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}}]],
					{az, alt} = Transpose[({
						Replace[Mod[180 + radiansToDegrees[#[[3, 1]]], 360], n_?NumericQ :> Quantity[SetAccuracy[n, 3], "AngularDegrees"]], 
						Replace[radiansToDegrees[#[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[If[MatchQ[altm, "ApparentAltitude"], n + (1.02 Cot[(n + 10.3/(n + 5.11)) Degree])/60., n], 3], "AngularDegrees"]]
						} &@ PlanetaryAstronomy`Private`PACoordinatesX[PlanetaryAstronomy`Private`JDU[astroToGMT[#, utz]], "Sun", {"EarthApparent", {long*Degree, lat*Degree, 0}}])&/@date]; 
					DeleteDuplicates[Transpose[{date, (getIntervalResultForQuantityPairs[#, intervalvalue, location]&/@Transpose[{az, alt}])/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}}]]
				]
		]
	]/@loc
	
]

AstronomyConvenienceFunction["SunPosition", locdatepairs:{{(_?(validLocationQ[#, SunPosition]&)), {_?NumericQ..}}..}, coordsys_, utz_, tz_, ugeoloc_, intervalvalue_, altm_]:= Module[{jd, az, alt, ra, dec, long, lat, coords, rarange, decrange, altrange, azrange, dims},
	Function[{pair},
		jd = PlanetaryAstronomy`Private`JDU[astroToGMT[pair[[2]], utz]];
		(* Since getCoordinates may contain elevation data, strip it out for now until PACoordinatesX, etc. *)
		(* can be made to support more than just latitude and longitude arguments *)
		coords = getCoordinates[pair[[1]], 2];
		dims = Dimensions[coords];
		If[Length[dims]>1,
			If[coordsys === "Equatorial",
				{rarange, decrange} = Function[lst, Sort[Select[lst, CleanQ[#]&]]]/@Transpose[(
					Function[pacx, {
						Replace[radiansToHours[pacx[[3, 1]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "HoursOfRightAscension"]], 
						Replace[radiansToDegrees[pacx[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "AngularDegrees"]]
						}]@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Sun", {"EarthEquatorialApparentTopo", {#[[1]]*Degree, #[[2]]*Degree, 0}}])&/@coords];
				getIntervalResultForQuantityPairs[{rarange, decrange}, intervalvalue, pair[[1]]]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]},
				{azrange, altrange} = Function[lst, Sort[Select[lst, CleanQ[#]&]]]/@Transpose[(
					Function[pacx, {
						Replace[Mod[180 + radiansToDegrees[pacx[[3, 1]]], 360], n_?NumericQ :> Quantity[SetAccuracy[n, 3], "AngularDegrees"]],
						Replace[radiansToDegrees[pacx[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[If[MatchQ[altm, "ApparentAltitude"], n + (1.02 Cot[(n + 10.3/(n + 5.11)) Degree])/60., n], 3], "AngularDegrees"]]
					}]@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Sun", {"EarthApparent", {#[[1]]*Degree, #[[2]]*Degree, 0}}])&/@coords];
				getIntervalResultForQuantityPairs[{azrange, altrange}, intervalvalue, pair[[1]]]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}
			],
			{long, lat} = If[MatchQ[coords, {_?NumericQ, _?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
			If[!MatchQ[{long, lat}, {_?NumericQ, _?NumericQ}]&&!MatchQ[coordsys, "Equatorial"], Return[Missing["InvalidCoordinates"]]];
			If[coordsys === "Equatorial", 
				{ra, dec} = {
					Replace[radiansToHours[#[[3, 1]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "HoursOfRightAscension"]], 
					Replace[radiansToDegrees[#[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "AngularDegrees"]]
					} &@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Sun", {"EarthEquatorialApparentTopo", {long*Degree, lat*Degree, 0}}];
				getIntervalResultForQuantityPairs[{ra, dec}, intervalvalue, pair[[1]]]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]},
				{az, alt} = {
					Replace[Mod[180 + radiansToDegrees[#[[3, 1]]], 360], n_?NumericQ :> Quantity[SetAccuracy[n, 3], "AngularDegrees"]], 
					Replace[radiansToDegrees[#[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[If[MatchQ[altm, "ApparentAltitude"], n + (1.02 Cot[(n + 10.3/(n + 5.11)) Degree])/60., n], 3], "AngularDegrees"]]
				} &@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Sun", {"EarthApparent", {long*Degree, lat*Degree, 0}}];
				getIntervalResultForQuantityPairs[{az, alt}, intervalvalue, pair[[1]]]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}
			]
		]
	]/@locdatepairs
];

(* MoonPosition *)

AstronomyConvenienceFunction["MoonPosition", loc_?(validLocationQ[#, MoonPosition]&), date:{_?NumericQ..}, coordsys_, utz_, tz_, ugeoloc_, intervalvalue_, altm_]:= Module[{jd, az, alt, ra, dec, long, lat, coords, rarange, decrange, altrange, azrange, dims},
	jd = PlanetaryAstronomy`Private`JDU[astroToGMT[date, utz]];
	(* Since getCoordinates may contain elevation data, strip it out for now until PACoordinatesX, etc. *)
	(* can be made to support more than just latitude and longitude arguments *)
	coords = getCoordinates[loc, 2];
	dims = Dimensions[coords];
	If[Length[dims]>1,
		If[coordsys === "Equatorial",
			{rarange, decrange} = Function[lst, Sort[Select[lst, CleanQ[#]&]]]/@Transpose[(
				Function[pacx, {
					Replace[radiansToHours[pacx[[3, 1]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "HoursOfRightAscension"]], 
					Replace[radiansToDegrees[pacx[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "AngularDegrees"]]
				}]@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Moon", {"EarthEquatorialApparentTopo", {#[[1]]*Degree, #[[2]]*Degree, 0}}])&/@coords];
			getIntervalResultForQuantityPairs[{rarange, decrange}, intervalvalue, loc]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]},
			{azrange, altrange} = Function[lst, Sort[Select[lst, CleanQ[#]&]]]/@Transpose[(
				Function[pacx, {
					Replace[Mod[180 + radiansToDegrees[pacx[[3, 1]]], 360], n_?NumericQ :> Quantity[SetAccuracy[n, 3], "AngularDegrees"]],
					Replace[radiansToDegrees[pacx[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[If[MatchQ[altm, "ApparentAltitude"], n + (1.02 Cot[(n + 10.3/(n + 5.11)) Degree])/60., n], 3], "AngularDegrees"]]
				}]@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Moon", {"EarthApparent", {#[[1]]*Degree, #[[2]]*Degree, 0}}])&/@coords];
			getIntervalResultForQuantityPairs[{azrange, altrange}, intervalvalue, loc]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}
		],
		{long, lat} = If[MatchQ[coords, {_?NumericQ, _?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
		If[!MatchQ[{long, lat}, {_?NumericQ, _?NumericQ}]&&!MatchQ[coordsys, "Equatorial"], Return[Missing["InvalidCoordinates"]]];
		If[coordsys === "Equatorial", 
			{ra, dec} = {
					Replace[radiansToHours[#[[3, 1]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "HoursOfRightAscension"]], 
					Replace[radiansToDegrees[#[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "AngularDegrees"]]
				} &@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Moon", {"EarthEquatorialApparentTopo", {long*Degree, lat*Degree, 0}}];
			getIntervalResultForQuantityPairs[{ra, dec}, intervalvalue, loc]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]},
			{az, alt} = {
					Replace[Mod[180 + radiansToDegrees[#[[3, 1]]], 360], n_?NumericQ :> Quantity[SetAccuracy[n, 3], "AngularDegrees"]], 
					Replace[radiansToDegrees[#[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[If[MatchQ[altm, "ApparentAltitude"], n + (1.02 Cot[(n + 10.3/(n + 5.11)) Degree])/60., n], 3], "AngularDegrees"]]
				} &@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Moon", {"EarthApparent", {long*Degree, lat*Degree, 0}}];
			getIntervalResultForQuantityPairs[{az, alt}, intervalvalue, loc]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}
		]
	]
	];
	
AstronomyConvenienceFunction["MoonPosition", loc_?(validLocationQ[#, MoonPosition]&), date:{{_?NumericQ..}..}, coordsys_, utz_, tz_, ugeoloc_, intervalvalue_, altm_]:= Module[{az, alt, ra, dec, long, lat, coords, rarange, decrange, azrange, altrange, dims},
	(* Since getCoordinates may contain elevation data, strip it out for now until PACoordinatesX, etc. *)
	(* can be made to support more than just latitude and longitude arguments *)
	coords = getCoordinates[loc, 2];
	dims = Dimensions[coords];
	If[Length[dims]>1, 
		If[coordsys === "Equatorial", 
			{rarange, decrange} = Transpose[Function[{idate},
				Function[lst, Sort[Select[lst, CleanQ[#]&]]]/@Transpose[(
				Function[pacx, {
					Replace[radiansToHours[pacx[[3, 1]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "HoursOfRightAscension"]], 
					Replace[radiansToDegrees[pacx[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "AngularDegrees"]]
				}]@ PlanetaryAstronomy`Private`PACoordinatesX[PlanetaryAstronomy`Private`JDU[astroToGMT[idate, utz]], "Moon", {"EarthEquatorialApparentTopo", {#[[1]]*Degree, #[[2]]*Degree, 0}}])&/@coords]]/@date];
			DeleteDuplicates[Transpose[{date, (getIntervalResultForQuantityPairs[#, intervalvalue, loc]&/@Transpose[{rarange, decrange}])/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}}]],
			{azrange, altrange} = Transpose[Function[{idate},
				Function[lst, Sort[Select[lst, CleanQ[#]&]]]/@Transpose[(
				Function[pacx, {
					Replace[Mod[180 + radiansToDegrees[pacx[[3, 1]]], 360], n_?NumericQ :> Quantity[SetAccuracy[n, 3], "AngularDegrees"]],
					Replace[radiansToDegrees[pacx[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[If[MatchQ[altm, "ApparentAltitude"], n + (1.02 Cot[(n + 10.3/(n + 5.11)) Degree])/60., n], 3], "AngularDegrees"]]
				}]@ PlanetaryAstronomy`Private`PACoordinatesX[PlanetaryAstronomy`Private`JDU[astroToGMT[idate, utz]], "Moon", {"EarthApparent", {#[[1]]*Degree, #[[2]]*Degree, 0}}])&/@coords]]/@date];
			DeleteDuplicates[Transpose[{date, ((getIntervalResultForQuantityPairs[#, intervalvalue, loc])&/@Transpose[{azrange,altrange}])}]]],
		{long, lat} = If[MatchQ[coords, {_?NumericQ, _?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
		If[!MatchQ[{long, lat}, {_?NumericQ, _?NumericQ}]&&!MatchQ[coordsys, "Equatorial"], Return[Missing["InvalidCoordinates"]]];
		If[coordsys === "Equatorial",
			{ra, dec} = Transpose[({
					Replace[radiansToHours[#[[3, 1]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "HoursOfRightAscension"]], 
					Replace[radiansToDegrees[#[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "AngularDegrees"]]
					} &@ PlanetaryAstronomy`Private`PACoordinatesX[PlanetaryAstronomy`Private`JDU[astroToGMT[#, utz]], "Moon", {"EarthEquatorialApparentTopo", {long*Degree, lat*Degree, 0}}])&/@date]; 
			DeleteDuplicates[Transpose[{date, (getIntervalResultForQuantityPairs[#, intervalvalue, loc]&/@Transpose[{ra, dec}])/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}}]],
			{az, alt} = Transpose[({
					Replace[Mod[180 + radiansToDegrees[#[[3, 1]]], 360], n_?NumericQ :> Quantity[SetAccuracy[n, 3], "AngularDegrees"]], 
					Replace[radiansToDegrees[#[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[If[MatchQ[altm, "ApparentAltitude"], n + (1.02 Cot[(n + 10.3/(n + 5.11)) Degree])/60., n], 3], "AngularDegrees"]]
				} &@ PlanetaryAstronomy`Private`PACoordinatesX[PlanetaryAstronomy`Private`JDU[astroToGMT[#, utz]], "Moon", {"EarthApparent", {long*Degree, lat*Degree, 0}}])&/@date]; 
			DeleteDuplicates[Transpose[{date, (getIntervalResultForQuantityPairs[#, intervalvalue, loc]&/@Transpose[{az, alt}])/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}}]]
			]
		]
	
	];
	
AstronomyConvenienceFunction["MoonPosition", locs:{(_?(validLocationQ[#, MoonPosition]&))..}, date:{_?NumericQ..}, coordsys_, utz_, tz_, ugeoloc_, intervalvalue_, altm_]:= Module[{jd, az, alt, ra, dec, long, lat, coords, rarange, decrange, altrange, azrange, dims},
	jd = PlanetaryAstronomy`Private`JDU[astroToGMT[date, utz]];
	Function[{location},
		(* Since getCoordinates may contain elevation data, strip it out for now until PACoordinatesX, etc. *)
		(* can be made to support more than just latitude and longitude arguments *)
		coords = getCoordinates[location, 2];
		dims = Dimensions[coords];
		If[Length[dims]>1,
			If[coordsys === "Equatorial",
				{rarange, decrange} = Function[lst, Sort[Select[lst, CleanQ[#]&]]]/@Transpose[(
					Function[pacx, {
						Replace[radiansToHours[pacx[[3, 1]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "HoursOfRightAscension"]], 
						Replace[radiansToDegrees[pacx[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "AngularDegrees"]]
						}]@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Moon", {"EarthEquatorialApparentTopo", {#[[1]]*Degree, #[[2]]*Degree, 0}}])&/@coords];
				getIntervalResultForQuantityPairs[{rarange, decrange}, intervalvalue, location]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]},
				{azrange, altrange} = Function[lst, Sort[Select[lst, CleanQ[#]&]]]/@Transpose[(
					Function[pacx, {
						Replace[Mod[180 + radiansToDegrees[pacx[[3, 1]]], 360], n_?NumericQ :> Quantity[SetAccuracy[n, 3], "AngularDegrees"]],
						Replace[radiansToDegrees[pacx[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[If[MatchQ[altm, "ApparentAltitude"], n + (1.02 Cot[(n + 10.3/(n + 5.11)) Degree])/60., n], 3], "AngularDegrees"]]
					}]@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Moon", {"EarthApparent", {#[[1]]*Degree, #[[2]]*Degree, 0}}])&/@coords];
				getIntervalResultForQuantityPairs[{azrange, altrange}, intervalvalue, location]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}
			],
			{long, lat} = If[MatchQ[coords, {_?NumericQ, _?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
			If[!MatchQ[{long, lat}, {_?NumericQ, _?NumericQ}]&&!MatchQ[coordsys, "Equatorial"], Return[Missing["InvalidCoordinates"]]];
			If[coordsys === "Equatorial", 
				{ra, dec} = {
					Replace[radiansToHours[#[[3, 1]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "HoursOfRightAscension"]], 
					Replace[radiansToDegrees[#[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "AngularDegrees"]]
					} &@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Moon", {"EarthEquatorialApparentTopo", {long*Degree, lat*Degree, 0}}];
				getIntervalResultForQuantityPairs[{ra, dec}, intervalvalue, location]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]},
				{az, alt} = {
					Replace[Mod[180 + radiansToDegrees[#[[3, 1]]], 360], n_?NumericQ :> Quantity[SetAccuracy[n, 3], "AngularDegrees"]], 
					Replace[radiansToDegrees[#[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[If[MatchQ[altm, "ApparentAltitude"], n + (1.02 Cot[(n + 10.3/(n + 5.11)) Degree])/60., n], 3], "AngularDegrees"]]
				} &@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Moon", {"EarthApparent", {long*Degree, lat*Degree, 0}}];
				getIntervalResultForQuantityPairs[{az, alt}, intervalvalue, location]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}
			]
		]
	]/@locs
	];

AstronomyConvenienceFunction["MoonPosition", loc:{(_?(validLocationQ[#, MoonPosition]&))..}, date:{{_?NumericQ..}..}, coordsys_, utz_, tz_, ugeoloc_, intervalvalue_, altm_]:= Module[{az, alt, ra, dec, long, lat, coords, rarange, decrange, azrange, altrange, dims},
	Function[{location},
		(* Since getCoordinates may contain elevation data, strip it out for now until PACoordinatesX, etc. *)
		(* can be made to support more than just latitude and longitude arguments *)
		coords = getCoordinates[location, 2];
		dims = Dimensions[coords];
		If[Length[dims]>1, 
			If[coordsys === "Equatorial", 
				{rarange, decrange} = Transpose[Function[{idate},
					Function[lst, Sort[Select[lst, CleanQ[#]&]]]/@Transpose[(
						Function[pacx, {
							Replace[radiansToHours[pacx[[3, 1]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "HoursOfRightAscension"]], 
							Replace[radiansToDegrees[pacx[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "AngularDegrees"]]
						}]@ PlanetaryAstronomy`Private`PACoordinatesX[PlanetaryAstronomy`Private`JDU[astroToGMT[idate, utz]], "Moon", {"EarthEquatorialApparentTopo", {#[[1]]*Degree, #[[2]]*Degree, 0}}])&/@coords]]/@date];
				DeleteDuplicates[Transpose[{date, (getIntervalResultForQuantityPairs[#, intervalvalue, location]&/@Transpose[{rarange, decrange}])/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}}]],
				{azrange, altrange} = Transpose[Function[{idate},
					Function[lst, Sort[Select[lst, CleanQ[#]&]]]/@Transpose[(
						Function[pacx, {
							Replace[Mod[180 + radiansToDegrees[pacx[[3, 1]]], 360], n_?NumericQ :> Quantity[SetAccuracy[n, 3], "AngularDegrees"]],
							Replace[radiansToDegrees[pacx[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[If[MatchQ[altm, "ApparentAltitude"], n + (1.02 Cot[(n + 10.3/(n + 5.11)) Degree])/60., n], 3], "AngularDegrees"]]
						}]@ PlanetaryAstronomy`Private`PACoordinatesX[PlanetaryAstronomy`Private`JDU[astroToGMT[idate, utz]], "Moon", {"EarthApparent", {#[[1]]*Degree, #[[2]]*Degree, 0}}])&/@coords]]/@date];
				DeleteDuplicates[Transpose[{date, ((getIntervalResultForQuantityPairs[#, intervalvalue, location])&/@Transpose[{azrange,altrange}])}]]],
				{long, lat} = If[MatchQ[coords, {_?NumericQ, _?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
				If[!MatchQ[{long, lat}, {_?NumericQ, _?NumericQ}]&&!MatchQ[coordsys, "Equatorial"], Return[Missing["InvalidCoordinates"]]];
				If[coordsys === "Equatorial",
					{ra, dec} = Transpose[({
						Replace[radiansToHours[#[[3, 1]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "HoursOfRightAscension"]], 
						Replace[radiansToDegrees[#[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "AngularDegrees"]]
					} &@ PlanetaryAstronomy`Private`PACoordinatesX[PlanetaryAstronomy`Private`JDU[astroToGMT[#, utz]], "Moon", {"EarthEquatorialApparentTopo", {long*Degree, lat*Degree, 0}}])&/@date]; 
					DeleteDuplicates[Transpose[{date, (getIntervalResultForQuantityPairs[#, intervalvalue, location]&/@Transpose[{ra, dec}])/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}}]],
					{az, alt} = Transpose[({
						Replace[Mod[180 + radiansToDegrees[#[[3, 1]]], 360], n_?NumericQ :> Quantity[SetAccuracy[n, 3], "AngularDegrees"]], 
						Replace[radiansToDegrees[#[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[If[MatchQ[altm, "ApparentAltitude"], n + (1.02 Cot[(n + 10.3/(n + 5.11)) Degree])/60., n], 3], "AngularDegrees"]]
						} &@ PlanetaryAstronomy`Private`PACoordinatesX[PlanetaryAstronomy`Private`JDU[astroToGMT[#, utz]], "Moon", {"EarthApparent", {long*Degree, lat*Degree, 0}}])&/@date]; 
					DeleteDuplicates[Transpose[{date, (getIntervalResultForQuantityPairs[#, intervalvalue, location]&/@Transpose[{az, alt}])/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}}]]
				]
		]
	]/@loc
	
]

AstronomyConvenienceFunction["MoonPosition", locdatepairs:{{(_?(validLocationQ[#, MoonPosition]&)), {_?NumericQ..}}..}, coordsys_, utz_, tz_, ugeoloc_, intervalvalue_, altm_]:= Module[{jd, az, alt, ra, dec, long, lat, coords, rarange, decrange, altrange, azrange, dims},
	Function[{pair},
		jd = PlanetaryAstronomy`Private`JDU[astroToGMT[pair[[2]], utz]];
		(* Since getCoordinates may contain elevation data, strip it out for now until PACoordinatesX, etc. *)
		(* can be made to support more than just latitude and longitude arguments *)
		coords = getCoordinates[pair[[1]], 2];
		dims = Dimensions[coords];
		If[Length[dims]>1,
			If[coordsys === "Equatorial",
				{rarange, decrange} = Function[lst, Sort[Select[lst, CleanQ[#]&]]]/@Transpose[(
					Function[pacx, {
						Replace[radiansToHours[pacx[[3, 1]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "HoursOfRightAscension"]], 
						Replace[radiansToDegrees[pacx[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "AngularDegrees"]]
						}]@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Moon", {"EarthEquatorialApparentTopo", {#[[1]]*Degree, #[[2]]*Degree, 0}}])&/@coords];
				getIntervalResultForQuantityPairs[{rarange, decrange}, intervalvalue, pair[[1]]]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]},
				{azrange, altrange} = Function[lst, Sort[Select[lst, CleanQ[#]&]]]/@Transpose[(
					Function[pacx, {
						Replace[Mod[180 + radiansToDegrees[pacx[[3, 1]]], 360], n_?NumericQ :> Quantity[SetAccuracy[n, 3], "AngularDegrees"]],
						Replace[radiansToDegrees[pacx[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[If[MatchQ[altm, "ApparentAltitude"], n + (1.02 Cot[(n + 10.3/(n + 5.11)) Degree])/60., n], 3], "AngularDegrees"]]
					}]@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Moon", {"EarthApparent", {#[[1]]*Degree, #[[2]]*Degree, 0}}])&/@coords];
				getIntervalResultForQuantityPairs[{azrange, altrange}, intervalvalue, pair[[1]]]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}
			],
			{long, lat} = If[MatchQ[coords, {_?NumericQ, _?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
			If[!MatchQ[{long, lat}, {_?NumericQ, _?NumericQ}]&&!MatchQ[coordsys, "Equatorial"], Return[Missing["InvalidCoordinates"]]];
			If[coordsys === "Equatorial", 
				{ra, dec} = {
					Replace[radiansToHours[#[[3, 1]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "HoursOfRightAscension"]], 
					Replace[radiansToDegrees[#[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[n, 4], "AngularDegrees"]]
					} &@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Moon", {"EarthEquatorialApparentTopo", {long*Degree, lat*Degree, 0}}];
				getIntervalResultForQuantityPairs[{ra, dec}, intervalvalue, pair[[1]]]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]},
				{az, alt} = {
					Replace[Mod[180 + radiansToDegrees[#[[3, 1]]], 360], n_?NumericQ :> Quantity[SetAccuracy[n, 3], "AngularDegrees"]], 
					Replace[radiansToDegrees[#[[3, 2]]], n_?NumericQ :> Quantity[SetAccuracy[If[MatchQ[altm, "ApparentAltitude"], n + (1.02 Cot[(n + 10.3/(n + 5.11)) Degree])/60., n], 3], "AngularDegrees"]]
				} &@ PlanetaryAstronomy`Private`PACoordinatesX[jd, "Moon", {"EarthApparent", {long*Degree, lat*Degree, 0}}];
				getIntervalResultForQuantityPairs[{az, alt}, intervalvalue, pair[[1]]]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}
			]
		]
	]/@locdatepairs
];

(* MoonPhase *)
getSignedFraction[frac_, name_]:= Module[{phasename},
	phasename = If[MatchQ[name, _Entity], name[[2]], "bad"];
	Switch[phasename,
		"Full"|"New"|"FirstQuarter"|"WaxingCrescent"|"WaxingGibbous", frac,
		"LastQuarter"|"WaningCrescent"|"WaningGibbous", -frac,
		_, frac
		]
]

AstronomyConvenienceFunction["MoonPhase", prop_, date:{_?NumericQ..}, utz_, tz_, ugeoloc_]:= Module[{},
 Switch[prop,
 	"Fraction", PlanetaryAstronomy`Private`IlluminationFraction["Moon", astroToGMT[date, utz]],
 	"SignedFraction", getSignedFraction[PlanetaryAstronomy`Private`IlluminationFraction["Moon", astroToGMT[date, utz]], Entity["MoonPhase", PlanetaryAstronomy`Private`AlphaMoonPhase[astroToGMT[date, utz]]]],
 	"Name", Entity["MoonPhase", PlanetaryAstronomy`Private`AlphaMoonPhase[astroToGMT[date, utz]]],
 	"Icon", PlanetaryAstronomy`Private`MoonIcon[astroToGMT[date, utz]],
 	_, Missing["NotAvailable"]
 ]
 
 ];

AstronomyConvenienceFunction["MoonPhase", prop_, date:{{_?NumericQ..}..}, utz_, tz_, ugeoloc_]:= Module[{},
 Switch[prop,
 	"Fraction", DeleteDuplicates[Transpose[{date, PlanetaryAstronomy`Private`IlluminationFraction["Moon", astroToGMT[#, utz]]&/@date}]],
 	"SignedFraction", DeleteDuplicates[Transpose[{date, getSignedFraction[PlanetaryAstronomy`Private`IlluminationFraction["Moon", astroToGMT[#, utz]], Entity["MoonPhase", PlanetaryAstronomy`Private`AlphaMoonPhase[astroToGMT[#, utz]]]]&/@date}]],
 	"Name", DeleteDuplicates[Transpose[{date, Entity["MoonPhase", PlanetaryAstronomy`Private`AlphaMoonPhase[astroToGMT[#, utz]]]&/@date}]],
 	"Icon", DeleteDuplicates[Transpose[{date, PlanetaryAstronomy`Private`MoonIcon[astroToGMT[#, utz]]&/@date}]],
 	_, Missing["NotAvailable"]
 ]
 
 ];

(* Sidereal time *)

ApparentLST[longlat_, gmttime_]:=If[MatchQ[gmttime, _Missing]||CorruptQ[longlat], 
	Missing["NotAvailable"], 
	Mod[
		radiansToHours[PlanetaryAstronomy`Private`PATransformationX[PlanetaryAstronomy`Private`PASystem[{"Earth", longlat Degree}, PlanetaryAstronomy`Private`JDU[gmttime], {0, 1.57079, 1}], "EarthEquatorial"][[3, 1]]], 
		24]/.n_?NumericQ:>Quantity[n, "HoursOfRightAscension"]
]

AstronomyConvenienceFunction["SiderealTime", loc_?(validLocationQ[#, SiderealTime]&), date:{_?NumericQ..}, utz_, tz_, ugeoloc_, intervalvalue_]:= Module[{idate, lat, long, coords, rangeres,res},
	idate = astroToGMT[date, utz];
	(* Since getCoordinates may contain elevation data, strip it out for now until PACoordinatesX, etc. *)
	(* can be made to support more than just latitude and longitude arguments *)
	coords = getCoordinates[loc, 2];
	If[CleanQ[coords]&&Length[Dimensions[coords]]>1, 
		rangeres=Sort[Select[ApparentLST[#, idate]&/@(coords),CleanQ[#]&]];
		getIntervalResultForQuantity[rangeres, intervalvalue, loc],
		{long, lat} = If[MatchQ[coords, {_?NumericQ, _?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
		If[!MatchQ[{long, lat}, {_?NumericQ, _?NumericQ}], Return[Missing["InvalidCoordinates"]]];
		res=ApparentLST[{long, lat}, idate];
		getIntervalResultForQuantity[res, intervalvalue, loc]
	]
 ]

AstronomyConvenienceFunction["SiderealTime", loc_?(validLocationQ[#, SiderealTime]&), date:{{_?NumericQ..}..}, utz_, tz_, ugeoloc_, intervalvalue_]:= Module[{st, strange, long, lat, coords, dims},
	(* Since getCoordinates may contain elevation data, strip it out for now until PACoordinatesX, etc. *)
	(* can be made to support more than just latitude and longitude arguments *)
	coords = getCoordinates[loc, 2];
	dims = Dimensions[coords];
	If[Length[dims]>1, 
		strange = Function[{idate}, Sort[Select[ApparentLST[#, astroToGMT[idate, utz]]&/@(coords), CleanQ[#]&]]]/@date;
		DeleteDuplicates[Transpose[{date, ((getIntervalResultForQuantity[#, intervalvalue, loc])&/@strange)}]],
		{long, lat} = If[MatchQ[coords, {_?NumericQ, _?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
		If[!MatchQ[{long, lat}, {_?NumericQ, _?NumericQ}], Return[Missing["InvalidCoordinates"]]];
		st = (ApparentLST[{long, lat}, astroToGMT[#, utz]]&/@date);
		DeleteDuplicates[Transpose[{date, (getIntervalResultForQuantity[#, intervalvalue, loc]&/@st)/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}}]]
		]
	
	];
	
AstronomyConvenienceFunction["SiderealTime", locs:{_?(validLocationQ[#, SiderealTime]&)..}, date:{_?NumericQ..}, utz_, tz_, ugeoloc_, intervalvalue_]:= Module[{st, strange, lat, long, coords, dims},
	Function[{location},
	(* Since getCoordinates may contain elevation data, strip it out for now until PACoordinatesX, etc. *)
	(* can be made to support more than just latitude and longitude arguments *)
	coords = getCoordinates[location, 2];
	dims = Dimensions[coords];
	If[Length[dims]>1,
		strange=Sort[Select[ApparentLST[{#[[1]], #[[2]]}, astroToGMT[date, utz]]&/@(coords),CleanQ[#]&]];
		getIntervalResultForQuantity[strange, intervalvalue, location]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]},
		{long, lat} = If[MatchQ[coords, {_?NumericQ, _?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
		If[!MatchQ[{long, lat}, {_?NumericQ, _?NumericQ}], Return[Missing["InvalidCoordinates"]]];
		st = ApparentLST[{long, lat}, astroToGMT[date, utz]];
		getIntervalResultForQuantity[st, intervalvalue, location]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}
	]]/@locs
	];

AstronomyConvenienceFunction["SiderealTime", loc:{_?(validLocationQ[#, SiderealTime]&)..}, date:{{_?NumericQ..}..}, utz_, tz_, ugeoloc_, intervalvalue_]:= Module[{st, strange, long, lat, coords, dims},
	Function[{location},
	(* Since getCoordinates may contain elevation data, strip it out for now until PACoordinatesX, etc. *)
	(* can be made to support more than just latitude and longitude arguments *)
	coords = getCoordinates[location, 2];
	dims = Dimensions[coords];
	If[Length[dims]>1, 
		strange = Function[{idate}, Sort[Select[ApparentLST[#, astroToGMT[idate, utz]]&/@(coords), CleanQ[#]&]]]/@date;
		DeleteDuplicates[Transpose[{date, ((getIntervalResultForQuantity[#, intervalvalue, location])&/@strange)}]],
		{long, lat} = If[MatchQ[coords, {_?NumericQ, _?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
		If[!MatchQ[{long, lat}, {_?NumericQ, _?NumericQ}], Return[Missing["InvalidCoordinates"]]];
		st = (ApparentLST[{long ,lat}, astroToGMT[#, utz]]&/@date); 
		DeleteDuplicates[Transpose[{date, (getIntervalResultForQuantity[#, intervalvalue, location]&/@st)/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}}]]
		]]/@loc
	
	]

AstronomyConvenienceFunction["SiderealTime", locdatepairs:{{_?(validLocationQ[#, SiderealTime]&), {_?NumericQ..}}..}, utz_, tz_, ugeoloc_, intervalvalue_]:= Module[{st, strange, long, lat, coords, dims},
	Function[{pair},
	(* Since getCoordinates may contain elevation data, strip it out for now until PACoordinatesX, etc. *)
	(* can be made to support more than just latitude and longitude arguments *)
	coords = getCoordinates[pair[[1]], 2];
	dims = Dimensions[coords];
	If[Length[dims]>1,
		strange=Sort[Select[ApparentLST[{#[[1]], #[[2]]}, astroToGMT[pair[[2]], utz]]&/@(coords),CleanQ[#]&]];
		getIntervalResultForQuantity[strange, intervalvalue, pair[[1]]]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]},
		{long, lat} = If[MatchQ[coords, {_?NumericQ, _?NumericQ}], coords, {Missing["NotAvailable"], Missing["NotAvailable"]}];
		If[!MatchQ[{long, lat}, {_?NumericQ, _?NumericQ}], Return[Missing["InvalidCoordinates"]]];
		st = ApparentLST[{long, lat}, astroToGMT[pair[[2]], utz]];
		getIntervalResultForQuantity[st, intervalvalue, pair[[1]]]/.{$Failed, $Failed}->{Missing["NotAvailable"], Missing["NotAvailable"]}
	]]/@locdatepairs
	];

(* fallthrough case which should hopefully never be reached *)

AstronomyConvenienceFunction[___]:= $Failed

(* options *)
Options[Sunrise] = {TimeDirection :> 1, TimeZone :> $TimeZone};

Options[Sunset] = {TimeDirection :> 1, TimeZone :> $TimeZone};

Options[SunPosition] = {AltitudeMethod -> "TrueAltitude", CelestialSystem -> "Horizon"};

Options[MoonPosition] = {AltitudeMethod -> "TrueAltitude", CelestialSystem -> "Horizon"};

Options[SolarEclipse] = {EclipseType :> Automatic, TimeDirection :> 1, TimeZone :> $TimeZone};

Options[LunarEclipse] = {EclipseType :> Automatic, TimeDirection :> 1, TimeZone :> $TimeZone}; 

Options[DaylightQ] = {"SunAngleTest" -> (#>0&)};

Clear[iSunrise];
iSunrise[args___, OptionsPattern[Sunrise]] := Module[{msam, dt, tz, igeolocation, ei, api, ipairs},
	tz = TimeZoneOffset[OptionValue[TimeZone]];
	ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
	If[!validTimeZoneQ[tz], Message[Sunrise::zone, tz];tz = $SystemTimeZone];
	If[!validTimeDirection[ei, Sunrise], Message[Sunrise::evincr, ei];ei=Automatic];
	msam = makeSunriseArgumentsAndMessages[{args}];
	issueParserMessages[Sunrise, msam[[2]]];
	If[msam[[1]]==={}, Return[$Failed]];
	If[MatchQ[msam[[1]], {{{_?(validLocationQ[#, Sunrise]&), _?(validDateQ[#, Sunrise]&)}..}, _, _}],
		ipairs = {preProcessCoordinates[#[[1]]], (#[[2]]/.d_DateObject :> DateList[d])/.x_String/;MatchQ[Quiet[DateList[x]], _List]:>DateList[x]}&/@(msam[[1, 1]]/.GeoPosition[x_]:>x);
		api = AstronomyConvenienceFunction["Sunrise", ipairs, $TimeZone, tz, $GeoLocation/.GeoPosition[x_]:>x[[1;;2]], ei, Automatic];
		postProcessDateResults[api, tz, msam[[1, 2]], msam[[1, 3]]],
	igeolocation = preProcessCoordinates[msam[[1,1]]]/.GeoPosition[x_]:>x;
	If[MatchQ[igeolocation, {}], Message[Sunrise::loc, msam[[1,1]]]];
	dt=((msam[[1,2]]/.{start_?(validDateQ[#, Sunrise]&), end_?(validDateQ[#, Sunrise]&), incr_?(validIncrementQ[#]&)}:>DateRange[start, end, incr])/.d_DateObject :> DateList[d])/.x_String/;MatchQ[Quiet[DateList[x]], _List]:>DateList[x];
	If[!FreeQ[dt, $Failed|_$Failed|_DateRange], Return[$Failed]];
	api = AstronomyConvenienceFunction["Sunrise", igeolocation, dt, $TimeZone, tz, $GeoLocation/.GeoPosition[x_]:>x[[1;;2]], ei, msam[[1,3]]];
	If[!FreeQ[api, $Failed["ComputationTimeout"]],
		Message[Sunrise::time] 
		];
	Which[
		validDateQ[dt, Sunrise]&&validLocationQ[msam[[1,1]], Sunrise], If[!MatchQ[api, $Failed|_$Failed], 
			postProcessDateResults[api, tz, msam[[1,3]], msam[[1,4]]], 
			$Failed],
		MatchQ[dt, {_?(validDateQ[#, Sunrise]&)..}]&&validLocationQ[msam[[1,1]], Sunrise], If[!MatchQ[api, $Failed|_$Failed],
			EventSeries[postProcessDateResults[#, tz, msam[[1,3]], msam[[1,4]]]&/@((api)[[All, 2]]), {(api)[[All, 1]]}], 
			$Failed],
		validDateQ[dt, Sunrise]&&MatchQ[msam[[1,1]], {_?(validLocationQ[#, Sunrise]&)..}], If[!MatchQ[api, $Failed|_$Failed],
			postProcessDateResults[#, tz, msam[[1,3]], msam[[1,4]]]&/@(api), 
			$Failed],
		MatchQ[dt, {_?(validDateQ[#, Sunrise]&)..}]&&MatchQ[msam[[1,1]], {_?(validLocationQ[#, Sunrise]&)..}], If[!MatchQ[api, $Failed|_$Failed],
			Function[{loc}, EventSeries[postProcessDateResults[#, tz, msam[[1,3]], msam[[1,4]]]&/@(loc[[All, 2]]), {loc[[All, 1]]}]]/@(api), 
			$Failed],
		True, $Failed
		]
	]
]
 
Sunrise[args___, opts:OptionsPattern[]] /; (ArgumentCountQ[Sunrise,Length[DeleteCases[{args}, _Rule, Infinity]],0,4]) := Block[{res},
  res = iSunrise[args, opts];
  res /; !MatchQ[res, _Missing|$Failed]
  ]/;FreeQ[{args},_Rule,{1}]

Clear[iSunset];
iSunset[args___, OptionsPattern[Sunset]] := Module[{msam, dt, tz, igeolocation, ei, api, ipairs},
	tz = TimeZoneOffset[OptionValue[TimeZone]];
	ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
	If[!validTimeZoneQ[tz], Message[Sunset::zone, tz];tz = $SystemTimeZone];
	If[!validTimeDirection[ei, Sunset], Message[Sunset::evincr, ei];ei=Automatic];
	msam = makeSunsetArgumentsAndMessages[{args}];
	issueParserMessages[Sunset, msam[[2]]];
	If[msam[[1]]==={}, Return[$Failed]];
	If[MatchQ[msam[[1]], {{{_?(validLocationQ[#, Sunset]&), _?(validDateQ[#, Sunset]&)}..}, _, _}],
		ipairs = {preProcessCoordinates[#[[1]]], (#[[2]]/.d_DateObject :> DateList[d])/.x_String/;MatchQ[Quiet[DateList[x]], _List]:>DateList[x]}&/@(msam[[1, 1]]/.GeoPosition[x_]:>x);
		api = AstronomyConvenienceFunction["Sunset", ipairs, $TimeZone, tz, $GeoLocation/.GeoPosition[x_]:>x[[1;;2]], ei, Automatic];
		postProcessDateResults[api, tz, msam[[1, 2]], msam[[1, 3]]],
	igeolocation = preProcessCoordinates[msam[[1,1]]]/.GeoPosition[x_]:>x;
	If[MatchQ[igeolocation, {}], Message[Sunset::loc, msam[[1,1]]]];
	dt=((msam[[1,2]]/.{start_?(validDateQ[#, Sunset]&), end_?(validDateQ[#, Sunset]&), incr_?(validIncrementQ[#]&)}:>DateRange[start, end, incr])/.d_DateObject :> DateList[d])/.x_String/;MatchQ[Quiet[DateList[x]], _List]:>DateList[x];
	If[!FreeQ[dt, $Failed|_$Failed|_DateRange], Return[$Failed]];
	api = AstronomyConvenienceFunction["Sunset", igeolocation, dt, $TimeZone, tz, $GeoLocation/.GeoPosition[x_]:>x[[1;;2]], ei, msam[[1,3]]];
	If[!FreeQ[api, $Failed["ComputationTimeout"]],
		Message[Sunset::time] 
		];
	Which[
		validDateQ[dt, Sunset]&&validLocationQ[msam[[1,1]], Sunset], If[!MatchQ[api, $Failed|_$Failed], 
			postProcessDateResults[api, tz, msam[[1,3]], msam[[1,4]]], 
			$Failed],
		MatchQ[dt, {_?(validDateQ[#, Sunset]&)..}]&&validLocationQ[msam[[1,1]], Sunset], If[!MatchQ[api, $Failed|_$Failed],
			EventSeries[postProcessDateResults[#, tz, msam[[1,3]], msam[[1,4]]]&/@((api)[[All, 2]]), {(api)[[All, 1]]}], 
			$Failed],
		validDateQ[dt, Sunset]&&MatchQ[msam[[1,1]], {_?(validLocationQ[#, Sunset]&)..}], If[!MatchQ[api, $Failed|_$Failed],
			postProcessDateResults[#, tz, msam[[1,3]], msam[[1,4]]]&/@(api), 
			$Failed],
		MatchQ[dt, {_?(validDateQ[#, Sunset]&)..}]&&MatchQ[msam[[1,1]], {_?(validLocationQ[#, Sunset]&)..}], If[!MatchQ[api, $Failed|_$Failed],
			Function[{loc}, EventSeries[postProcessDateResults[#, tz, msam[[1,3]], msam[[1,4]]]&/@(loc[[All, 2]]), {loc[[All, 1]]}]]/@(api), 
			$Failed],
		True, $Failed
		]
	]
]
 
Sunset[args___, opts:OptionsPattern[]] /; (ArgumentCountQ[Sunset,Length[{args}],0,4]) := Block[{res},
  res = iSunset[args, opts];
  res /; !MatchQ[res, _Missing|$Failed]
  ]/;FreeQ[{args},_Rule,{1}]
  
Clear[iSunPosition];
Clear[iMoonPosition];
iSunPosition[args___, OptionsPattern[SunPosition]] := Module[{utz, tz, coord, igeolocation, msam, dt, api, ipairs, altm},
  tz = $TimeZone;
  coord = OptionValue[CelestialSystem];
  altm = OptionValue[AltitudeMethod];
  If[!validTimeZoneQ[tz], Message[SunPosition::zone, tz];tz = $SystemTimeZone];
  If[!MatchQ[coord, "Horizon"|"Equatorial"], Message[SunPosition::coord, coord];coord="Horizon"];
  If[!MatchQ[altm, "TrueAltitude"|"ApparentAltitude"], Message[SunPosition::altm, altm];altm="TrueAltitude"];
  msam = makeSunPositionArgumentsAndMessages[{args}, CelestialSystem -> coord];
  issueParserMessages[SunPosition, msam[[2]]];
 (* utz = If[MatchQ[msam[[1, 2]], _DateObject], DateValue[msam[[1,2]], "TimeZone"], $TimeZone];*)
  utz = $TimeZone;
  If[msam[[1]]==={}, Return[$Failed]];
    If[MatchQ[msam[[1]], {{{_?(validLocationQ[#, SunPosition]&), _?(validDateQ[#, SunPosition]&)}..}, _}],
		ipairs = {preProcessCoordinates[#[[1]]], (#[[2]]/.d_DateObject :> DateList[d])/.x_String/;MatchQ[Quiet[DateList[x]], _List]:>DateList[x]}&/@(msam[[1, 1]]/.GeoPosition[x_]:>x);
		api = AstronomyConvenienceFunction["SunPosition", ipairs, coord, utz, tz, $GeoLocation/.GeoPosition[x_]:>x[[1;;2]], msam[[1,2]], altm];
		api,
  igeolocation = preProcessCoordinates[msam[[1,1]]]/.GeoPosition[x_]:>x;
  If[coord==="Equatorial"&&Head[igeolocation]===Missing, igeolocation = {40.1, -88.2}];
  If[MatchQ[igeolocation, {}], Message[SunPosition::loc, msam[[1,1]]]];
  dt=((msam[[1,2]]/.{start_?(validDateQ[#, SunPosition]&), end_?(validDateQ[#, SunPosition]&), incr_?(validIncrementQ[#]&)}:>DateRange[start, end, incr])/. d_DateObject :> DateList[d])/.x_String/;MatchQ[Quiet[DateList[x]], _List]:>DateList[x];
  If[!FreeQ[dt, $Failed|_$Failed|_DateRange], Return[$Failed]];
  api = AstronomyConvenienceFunction["SunPosition", igeolocation, dt, coord, utz, tz, $GeoLocation/.GeoPosition[x_]:>x[[1;;2]], msam[[1,3]], altm];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[SunPosition::time] 
  	];
	Which[
		validDateQ[dt, SunPosition]&&validLocationQ[msam[[1,1]], SunPosition], If[!MatchQ[api, $Failed|_$Failed], 
			api, 
			$Failed],
		MatchQ[dt, {_?(validDateQ[#, SunPosition]&)..}]&&validLocationQ[msam[[1,1]], SunPosition], If[!MatchQ[api, $Failed|_$Failed],
			TimeSeries[(api)[[All, 2]], {(api)[[All, 1]]}, ResamplingMethod -> {"Interpolation", InterpolationOrder->1}], 
			$Failed],
		validDateQ[dt, SunPosition]&&MatchQ[msam[[1,1]], {_?(validLocationQ[#, SunPosition]&)..}], If[!MatchQ[api, $Failed|_$Failed],
			api, 
			$Failed],
		MatchQ[dt, {_?(validDateQ[#, SunPosition]&)..}]&&MatchQ[msam[[1,1]], {_?(validLocationQ[#, SunPosition]&)..}], If[!MatchQ[api, $Failed|_$Failed],
			Function[{loc}, TimeSeries[loc[[All, 2]], {loc[[All, 1]]}, ResamplingMethod -> {"Interpolation", InterpolationOrder->1}]]/@(api), 
			$Failed],
		True, $Failed
	]
	]
   ]
  
SunPosition[args___, opts:OptionsPattern[]] /; (ArgumentCountQ[SunPosition,Length[{args}],0,3]) := Block[{res},
  res = iSunPosition[args, opts];
  res /; !MatchQ[res, _Missing|$Failed]
  ]/;FreeQ[{args},_Rule,{1}]

iMoonPosition[args___, OptionsPattern[MoonPosition]] := Module[{tz, coord, igeolocation, msam, dt, api, ipairs, altm},
  tz = $TimeZone;
  coord = OptionValue[CelestialSystem];
  altm = OptionValue[AltitudeMethod];
  If[!validTimeZoneQ[tz], Message[MoonPosition::zone, tz];tz = $SystemTimeZone];
  If[!MatchQ[coord, "Horizon"|"Equatorial"], Message[MoonPosition::coord, coord];coord="Horizon"];
  If[!MatchQ[altm, "TrueAltitude"|"ApparentAltitude"], Message[MoonPosition::altm, altm];altm="TrueAltitude"];
  msam = makeMoonPositionArgumentsAndMessages[{args}, CelestialSystem -> coord];
  issueParserMessages[MoonPosition, msam[[2]]];
  If[msam[[1]]==={}, Return[$Failed]];
    If[MatchQ[msam[[1]], {{{_?(validLocationQ[#, MoonPosition]&), _?(validDateQ[#, MoonPosition]&)}..}, _}],
		ipairs = {preProcessCoordinates[#[[1]]], (#[[2]]/.d_DateObject :> DateList[d])/.x_String/;MatchQ[Quiet[DateList[x]], _List]:>DateList[x]}&/@(msam[[1, 1]]/.GeoPosition[x_]:>x);
		api = AstronomyConvenienceFunction["MoonPosition", ipairs, coord, $TimeZone, tz, $GeoLocation/.GeoPosition[x_]:>x[[1;;2]], msam[[1,2]], altm];
		api,
  igeolocation = preProcessCoordinates[msam[[1,1]]]/.GeoPosition[x_]:>x;
  If[coord==="Equatorial"&&Head[igeolocation]===Missing, igeolocation = {40.1, -88.2}];
  If[MatchQ[igeolocation, {}], Message[MoonPosition::loc, msam[[1,1]]]];
  dt=((msam[[1,2]]/.{start_?(validDateQ[#, MoonPosition]&), end_?(validDateQ[#, MoonPosition]&), incr_?(validIncrementQ[#]&)}:>DateRange[start, end, incr])/.d_DateObject :> DateList[d])/.x_String/;MatchQ[Quiet[DateList[x]], _List]:>DateList[x];
  If[!FreeQ[dt, $Failed|_$Failed|_DateRange], Return[$Failed]];
  api = AstronomyConvenienceFunction["MoonPosition", igeolocation, dt, coord, $TimeZone, tz, $GeoLocation/.GeoPosition[x_]:>x[[1;;2]], msam[[1,3]], altm];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[MoonPosition::time] 
  	];
	Which[
		validDateQ[dt, MoonPosition]&&validLocationQ[msam[[1,1]], MoonPosition], If[!MatchQ[api, $Failed|_$Failed], 
			api, 
			$Failed],
		MatchQ[dt, {_?(validDateQ[#, MoonPosition]&)..}]&&validLocationQ[msam[[1,1]], MoonPosition], If[!MatchQ[api, $Failed|_$Failed],
			TimeSeries[(api)[[All, 2]], {(api)[[All, 1]]}, ResamplingMethod -> {"Interpolation", InterpolationOrder->1}], 
			$Failed],
		validDateQ[dt, MoonPosition]&&MatchQ[msam[[1,1]], {_?(validLocationQ[#, MoonPosition]&)..}], If[!MatchQ[api, $Failed|_$Failed],
			api, 
			$Failed],
		MatchQ[dt, {_?(validDateQ[#, MoonPosition]&)..}]&&MatchQ[msam[[1,1]], {_?(validLocationQ[#, MoonPosition]&)..}], If[!MatchQ[api, $Failed|_$Failed],
			Function[{loc}, TimeSeries[loc[[All, 2]], {loc[[All, 1]]}, ResamplingMethod -> {"Interpolation", InterpolationOrder->1}]]/@(api), 
			$Failed],
		True, $Failed
	]
	]
   ]
   
MoonPosition[args___, opts:OptionsPattern[]] /; (ArgumentCountQ[MoonPosition,Length[{args}],0,3]) := Block[{res},
  res = iMoonPosition[args, opts];
  res /; !MatchQ[res, _Missing|$Failed]
  ]/;FreeQ[{args},_Rule,{1}]

(* MoonPhase functionality *)
Clear[iMoonPhase];
iMoonPhase["Properties"] = {"Fraction", "SignedFraction", "Icon", "Name"}

iMoonPhase[] := Module[{tz, igeolocation, apiresult, api},
  tz = $TimeZone;
  If[!validTimeZoneQ[tz], Message[MoonPhase::zone, tz];tz = $SystemTimeZone];
  igeolocation = preProcessCoordinates[$GeoLocation]/.GeoPosition[x_]:>x;
  api = AstronomyConvenienceFunction["MoonPhase", "Fraction", DateList[], $TimeZone, tz, igeolocation];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[MoonPhase::time] 
  ];
   apiresult = If[!MatchQ[api, $Failed|_$Failed], api, $Failed]
  ]

iMoonPhase[date_?(validDateQ[#, MoonPhase]&)] := Module[{tz, igeolocation, apiresult, api},
  tz = $TimeZone;
  If[!validTimeZoneQ[tz], Message[MoonPhase::zone, tz];tz = $SystemTimeZone];
  igeolocation = preProcessCoordinates[$GeoLocation]/.GeoPosition[x_]:>x;
  api = AstronomyConvenienceFunction["MoonPhase", "Fraction", DateList[date], $TimeZone, tz, igeolocation];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[MoonPhase::time] 
  ];
   apiresult = If[!MatchQ[api, $Failed|_$Failed], api, $Failed]
  ]

iMoonPhase[date:{_?(validDateQ[#, MoonPhase]&)..}] := Module[{tz, igeolocation, apiresult, api},
  tz = $TimeZone;
  If[!validTimeZoneQ[tz], Message[MoonPhase::zone, tz];tz = $SystemTimeZone];
  igeolocation = preProcessCoordinates[$GeoLocation]/.GeoPosition[x_]:>x;
  api = AstronomyConvenienceFunction["MoonPhase", "Fraction", DateList/@date, $TimeZone, tz, igeolocation];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[MoonPhase::time] 
  ];
   apiresult = If[!MatchQ[api, _Missing|$Failed|_$Failed], TimeSeries[api, ResamplingMethod -> {"Interpolation", InterpolationOrder -> 1}], $Failed]
  ]

iMoonPhase[{start_?(validDateQ[#, MoonPhase]||validDateStringQ[#, MoonPhase]&), end_?(validDateQ[#, MoonPhase]||validDateStringQ[#, MoonPhase]&), incr_}] := 
 iMoonPhase[DateRange[start, end, incr]]

iMoonPhase[prop:"Fraction"|"SignedFraction"|"Name"|"Icon"] := Module[{tz, igeolocation, apiresult, api},
  tz = $TimeZone;
  If[!validTimeZoneQ[tz], Message[MoonPhase::zone, tz];tz = $SystemTimeZone];
  igeolocation = preProcessCoordinates[$GeoLocation]/.GeoPosition[x_]:>x;
  api = AstronomyConvenienceFunction["MoonPhase", prop, DateList[], $TimeZone, tz, igeolocation];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[MoonPhase::time] 
  ];
   apiresult = If[!MatchQ[api, $Failed|_$Failed], api, $Failed]
  ]

iMoonPhase[date_?(validDateQ[#, MoonPhase]&), prop:"Fraction"|"SignedFraction"|"Name"|"Icon"] := Module[{tz, igeolocation, apiresult, api},
  tz = $TimeZone;
  If[!validTimeZoneQ[tz], Message[MoonPhase::zone, tz];tz = $SystemTimeZone];
  igeolocation = preProcessCoordinates[$GeoLocation]/.GeoPosition[x_]:>x;
  api = AstronomyConvenienceFunction["MoonPhase", prop, DateList[date], $TimeZone, tz, igeolocation];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[MoonPhase::time] 
  ];
   apiresult = If[!MatchQ[api, $Failed|_$Failed], api, $Failed]
  ]

iMoonPhase[date:{_?(validDateQ[#, MoonPhase]&)..}, prop:"Fraction"|"SignedFraction"|"Name"|"Icon"] := Module[{tz, igeolocation, apiresult, api},
  tz = $TimeZone;
  If[!validTimeZoneQ[tz], Message[MoonPhase::zone, tz];tz = $SystemTimeZone];
  igeolocation = preProcessCoordinates[$GeoLocation]/.GeoPosition[x_]:>x;
  api = AstronomyConvenienceFunction["MoonPhase", prop, DateList/@date, $TimeZone, tz, igeolocation];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[MoonPhase::time] 
  ];
  apiresult =  If[!MatchQ[api, _Missing|$Failed|_$Failed], 
   	Switch[prop,
   		"Fraction", TimeSeries[api, ResamplingMethod -> {"Interpolation", InterpolationOrder -> 1}],
   		"SignedFraction", TimeSeries[api, ResamplingMethod -> {"Interpolation", InterpolationOrder -> 1}],
   		"Name", EventSeries[(api)[[All, 2]], {(api)[[All, 1]]}],
   		"Icon", EventSeries[(api)[[All, 2]], {(api)[[All, 1]]}],
   		_, Missing["NotAvailable"]
   	], $Failed]
  ]

iMoonPhase[date_?(validDateQ[#, MoonPhase]&), prop_] := (Message[MoonPhase::noprop, HoldForm[prop]];$Failed)

iMoonPhase[date:{_?(validDateQ[#, MoonPhase]&)..}, prop_] := (Message[MoonPhase::noprop, HoldForm[prop]];$Failed)

iMoonPhase[{start_?(validDateQ[#, MoonPhase]||validDateStringQ[#, MoonPhase]&), end_?(validDateQ[#, MoonPhase]||validDateStringQ[#, MoonPhase]&), incr_}, prop_] := 
 iMoonPhase[DateRange[start, end, incr], prop]

iMoonPhase[args___]:=(Message[MoonPhase::dtspec, HoldForm[args]];$Failed)
  
MoonPhase[args___] /; (ArgumentCountQ[MoonPhase,Length[{args}],0,2]) := Block[{res},
  res = iMoonPhase[args];
  res /; !MatchQ[res, _Missing|$Failed]
  ]

(* Solar Eclipse functionality *)

fixGraphicsForGeoPosition[arg_]:=arg/.{
	Polygon[x:{{_?NumericQ,_?NumericQ}..}]:>Polygon[GeoPosition[Reverse/@x]],
	Polygon[x:{{{_?NumericQ,_?NumericQ}..}}]:>Polygon[GeoPosition[Reverse/@x[[1]]]],
	Line[x:{{_?NumericQ,_?NumericQ}..}]:>Line[GeoPosition[Reverse/@x]],
	Line[x:{{{_?NumericQ,_?NumericQ}..}}]:>Line[GeoPosition[Reverse/@x][[1]]]}

Clear[iSolarEclipse];
iSolarEclipse["Properties"] = {"GraphicsData", "MaximumEclipseDate", "PartialPhasePolygon", 
	"TotalPhaseEndDate", "TotalPhaseStartDate", "TotalPhaseCenterLine", "TotalPhasePolygon", "Type"}

iSolarEclipse[OptionsPattern[SolarEclipse]] := Module[{tz, igeolocation, apiresult, ei, et, api, placeholderloc},
  tz = TimeZoneOffset[OptionValue[TimeZone]];
  ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
  et = OptionValue[EclipseType];
  If[!validTimeZoneQ[tz], Message[SolarEclipse::zone, tz];tz = $SystemTimeZone];
  If[!validTimeDirection[ei, SolarEclipse], Message[SolarEclipse::evincr, ei];ei=Automatic];
  If[!MatchQ[et, Automatic|"Annular"|"Hybrid"|"Partial"|"Total"], Message[SolarEclipse::ectype, et];et=Automatic];
  placeholderloc = GeoPosition[{40.1, -88.2}];
  igeolocation = preProcessCoordinates[placeholderloc]/.GeoPosition[x_]:>x;
  api = ReleaseHold[APICompute[SolarEclipse,"AstronomyConvenienceFunction", {"SolarEclipse", "MaximumEclipseDate", igeolocation, DateList[], ei, et, $TimeZone, tz, igeolocation}]];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[SolarEclipse::time] 
  ];
   apiresult = If[!MatchQ[api, $Failed|_$Failed], postProcessDateResults["Result" /. api, tz, Automatic, Automatic], $Failed]
  ]

iSolarEclipse[date_?(validDateQ[#, SolarEclipse]&), OptionsPattern[SolarEclipse]] := Module[{tz, igeolocation, apiresult, ei, et, api, placeholderloc},
  tz = TimeZoneOffset[OptionValue[TimeZone]];
  ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
  et = OptionValue[EclipseType];
  If[!validTimeZoneQ[tz], Message[SolarEclipse::zone, tz];tz = $SystemTimeZone];
  If[!validTimeDirection[ei, SolarEclipse], Message[SolarEclipse::evincr, ei];ei=Automatic];
  If[!MatchQ[et, Automatic|"Annular"|"Hybrid"|"Partial"|"Total"], Message[SolarEclipse::ectype, et];et=Automatic];
  placeholderloc = GeoPosition[{40.1, -88.2}];
  igeolocation = preProcessCoordinates[placeholderloc]/.GeoPosition[x_]:>x;
  api = ReleaseHold[APICompute[SolarEclipse,"AstronomyConvenienceFunction", {"SolarEclipse", "MaximumEclipseDate", igeolocation, DateList[date], ei, et, $TimeZone, tz, igeolocation}]];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[SolarEclipse::time] 
  ];
   apiresult = If[!MatchQ[api, $Failed|_$Failed], postProcessDateResults["Result" /. api, tz, Automatic, Automatic], $Failed]
  ]

iSolarEclipse[date:{_?(validDateQ[#, SolarEclipse]&)..}, OptionsPattern[SolarEclipse]] := Module[{tz, igeolocation, apiresult, ei, et, api, placeholderloc},
  tz = TimeZoneOffset[OptionValue[TimeZone]];
  ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
  et = OptionValue[EclipseType];
  If[!validTimeZoneQ[tz], Message[SolarEclipse::zone, tz];tz = $SystemTimeZone];
  If[!validTimeDirection[ei, SolarEclipse], Message[SolarEclipse::evincr, ei];ei=Automatic];
  If[!MatchQ[et, Automatic|"Annular"|"Hybrid"|"Partial"|"Total"], Message[SolarEclipse::ectype, et];et=Automatic];
  placeholderloc = GeoPosition[{40.1, -88.2}];
  igeolocation = preProcessCoordinates[placeholderloc]/.GeoPosition[x_]:>x;
  api = ReleaseHold[APICompute[SolarEclipse,"AstronomyConvenienceFunction", {"SolarEclipse", "MaximumEclipseDate", igeolocation, DateList/@date, ei, et, $TimeZone, tz, igeolocation}]];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[SolarEclipse::time] 
  ];
   apiresult = If[!MatchQ[api, _Missing|$Failed|_$Failed], EventSeries[postProcessDateResults[#, tz, Automatic, Automatic]&/@(("Result" /. api)[[All, 2]]), {("Result" /. api)[[All, 1]]}], $Failed]
  ]

iSolarEclipse[{start_?(validDateQ[#, SolarEclipse]||validDateStringQ[#, SolarEclipse]&), end_?(validDateQ[#, SolarEclipse]||validDateStringQ[#, SolarEclipse]&), All}, opts:OptionsPattern[SolarEclipse]] := 
 Module[{tz, igeolocation, apiresult, ei, et, api, placeholderloc},
  tz = TimeZoneOffset[OptionValue[TimeZone]];
  ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
  et = OptionValue[EclipseType];
  If[!validTimeZoneQ[tz], Message[SolarEclipse::zone, tz];tz = $SystemTimeZone];
  If[!validTimeDirection[ei, SolarEclipse], Message[SolarEclipse::evincr, ei];ei=Automatic];
  If[!MatchQ[et, Automatic|"Annular"|"Hybrid"|"Partial"|"Total"], Message[SolarEclipse::ectype, et];et=Automatic];
  placeholderloc = GeoPosition[{40.1, -88.2}];
  igeolocation = preProcessCoordinates[placeholderloc]/.GeoPosition[x_]:>x;
  api = ReleaseHold[APICompute[SolarEclipse,"AstronomyConvenienceFunction", {"SolarEclipse", "MaximumEclipseDate", igeolocation, DateList/@{start, end}, ei, et, $TimeZone, tz, igeolocation, {All, "v2"}}]];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[SolarEclipse::time] 
  ];
   apiresult = If[!MatchQ[api, _Missing|$Failed|_$Failed], 
   postProcessDateResults[#, tz, Automatic, Automatic]&/@("Result" /. api), $Failed]/.$Failed -> Missing["NotAvailable"]
  ]

iSolarEclipse[{start_?(validDateQ[#, SolarEclipse]||validDateStringQ[#, SolarEclipse]&), end_?(validDateQ[#, SolarEclipse]||validDateStringQ[#, SolarEclipse]&), incr_}, opts:OptionsPattern[SolarEclipse]] := 
 iSolarEclipse[DateRange[start, end, incr], opts]

iSolarEclipse[prop:"MaximumEclipseDate"|"TotalPhaseStartDate"|"TotalPhaseEndDate", OptionsPattern[SolarEclipse]] := Module[{tz, igeolocation, apiresult, ei, et, api, placeholderloc},
  tz = TimeZoneOffset[OptionValue[TimeZone]];
  ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
  et = OptionValue[EclipseType];
  If[!validTimeZoneQ[tz], Message[SolarEclipse::zone, tz];tz = $SystemTimeZone];
  If[!validTimeDirection[ei, SolarEclipse], Message[SolarEclipse::evincr, ei];ei=Automatic];
  If[!MatchQ[et, Automatic|"Annular"|"Hybrid"|"Partial"|"Total"], Message[SolarEclipse::ectype, et];et=Automatic];
  placeholderloc = GeoPosition[{40.1, -88.2}];
  igeolocation = preProcessCoordinates[placeholderloc]/.GeoPosition[x_]:>x;
  api = ReleaseHold[APICompute[SolarEclipse,"AstronomyConvenienceFunction", {"SolarEclipse", prop, igeolocation, DateList[], ei, et, $TimeZone, tz, igeolocation}]];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[SolarEclipse::time] 
  ];
   apiresult = If[!MatchQ[api, $Failed|_$Failed], postProcessDateResults["Result" /. api, tz, Automatic, Automatic], $Failed]
  ]

iSolarEclipse[prop:"GraphicsData"|"PartialPhasePolygon"|"TotalPhasePolygon"|"TotalPhaseCenterLine"|"Type", OptionsPattern[SolarEclipse]] := Module[{tz, igeolocation, apiresult, ei, et, api, placeholderloc},
  tz = TimeZoneOffset[OptionValue[TimeZone]];
  ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
  et = OptionValue[EclipseType];
  If[!validTimeZoneQ[tz], Message[SolarEclipse::zone, tz];tz = $SystemTimeZone];
  If[!validTimeDirection[ei, SolarEclipse], Message[SolarEclipse::evincr, ei];ei=Automatic];
  If[!MatchQ[et, Automatic|"Annular"|"Hybrid"|"Partial"|"Total"], Message[SolarEclipse::ectype, et];et=Automatic];
  placeholderloc = GeoPosition[{40.1, -88.2}];
  igeolocation = preProcessCoordinates[placeholderloc]/.GeoPosition[x_]:>x;
  api = ReleaseHold[APICompute[SolarEclipse,"AstronomyConvenienceFunction", {"SolarEclipse", prop, igeolocation, DateList[], ei, et, $TimeZone, tz, igeolocation}]];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[SolarEclipse::time] 
  ];
   apiresult = If[!MatchQ[api, $Failed|_$Failed], 
   	fixGraphicsForGeoPosition[("Result" /. api)], 
   	$Failed]
  ]

iSolarEclipse[date_?(validDateQ[#, SolarEclipse]&), prop:"MaximumEclipseDate"|"TotalPhaseStartDate"|"TotalPhaseEndDate", OptionsPattern[SolarEclipse]] := Module[{tz, igeolocation, apiresult, ei, et, api, placeholderloc},
  tz = TimeZoneOffset[OptionValue[TimeZone]];
  ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
  et = OptionValue[EclipseType];
  If[!validTimeZoneQ[tz], Message[SolarEclipse::zone, tz];tz = $SystemTimeZone];
  If[!validTimeDirection[ei, SolarEclipse], Message[SolarEclipse::evincr, ei];ei=Automatic];
  If[!MatchQ[et, Automatic|"Annular"|"Hybrid"|"Partial"|"Total"], Message[SolarEclipse::ectype, et];et=Automatic];
  placeholderloc = GeoPosition[{40.1, -88.2}];
  igeolocation = preProcessCoordinates[placeholderloc]/.GeoPosition[x_]:>x;
  api = ReleaseHold[APICompute[SolarEclipse,"AstronomyConvenienceFunction", {"SolarEclipse", prop, igeolocation, DateList[date], ei, et, $TimeZone, tz, igeolocation}]];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[SolarEclipse::time] 
  ];
   apiresult = If[!MatchQ[api, $Failed|_$Failed], postProcessDateResults["Result" /. api, tz, Automatic, Automatic], $Failed]
  ]

iSolarEclipse[date_?(validDateQ[#, SolarEclipse]&), prop:"GraphicsData"|"PartialPhasePolygon"|"TotalPhasePolygon"|"TotalPhaseCenterLine"|"Type", OptionsPattern[SolarEclipse]] := Module[{tz, igeolocation, apiresult, ei, et, api, placeholderloc},
  tz = TimeZoneOffset[OptionValue[TimeZone]];
  ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
  et = OptionValue[EclipseType];
  If[!validTimeZoneQ[tz], Message[SolarEclipse::zone, tz];tz = $SystemTimeZone];
  If[!validTimeDirection[ei, SolarEclipse], Message[SolarEclipse::evincr, ei];ei=Automatic];
  If[!MatchQ[et, Automatic|"Annular"|"Hybrid"|"Partial"|"Total"], Message[SolarEclipse::ectype, et];et=Automatic];
  placeholderloc = GeoPosition[{40.1, -88.2}];
  igeolocation = preProcessCoordinates[placeholderloc]/.GeoPosition[x_]:>x;
  api = ReleaseHold[APICompute[SolarEclipse,"AstronomyConvenienceFunction", {"SolarEclipse", prop, igeolocation, DateList[date], ei, et, $TimeZone, tz, igeolocation}]];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[SolarEclipse::time] 
  ];
   apiresult = If[!MatchQ[api, $Failed|_$Failed], 
   	fixGraphicsForGeoPosition[("Result" /. api)], 
   	$Failed]
  ]

iSolarEclipse[date_?(validDateQ[#, SolarEclipse]&), prop_, OptionsPattern[SolarEclipse]] := (Message[SolarEclipse::noprop, HoldForm[prop]];$Failed)

iSolarEclipse[date:{_?(validDateQ[#, SolarEclipse]&)..}, prop:"GraphicsData"|"PartialPhasePolygon"|"TotalPhasePolygon"|"TotalPhaseCenterLine"|"Type"|"MaximumEclipseDate"|"TotalPhaseStartDate"|"TotalPhaseEndDate", OptionsPattern[SolarEclipse]] := Module[{tz, igeolocation, apiresult, ei, et, api, placeholderloc},
  tz = TimeZoneOffset[OptionValue[TimeZone]];
  ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
  et = OptionValue[EclipseType];
  If[!validTimeZoneQ[tz], Message[SolarEclipse::zone, tz];tz = $SystemTimeZone];
  If[!validTimeDirection[ei, SolarEclipse], Message[SolarEclipse::evincr, ei];ei=Automatic];
  If[!MatchQ[et, Automatic|"Annular"|"Hybrid"|"Partial"|"Total"], Message[SolarEclipse::ectype, et];et=Automatic];
  placeholderloc = GeoPosition[{40.1, -88.2}];
  igeolocation = preProcessCoordinates[placeholderloc]/.GeoPosition[x_]:>x;
  api = ReleaseHold[APICompute[SolarEclipse,"AstronomyConvenienceFunction", {"SolarEclipse", prop, igeolocation, DateList/@date, ei, et, $TimeZone, tz, igeolocation}]];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[SolarEclipse::time] 
  ];
   apiresult = If[!MatchQ[api, _Missing|$Failed|_$Failed], 
   	Switch[prop,
   		"MaximumEclipseDate"|"TotalPhaseStartDate"|"TotalPhaseEndDate", EventSeries[postProcessDateResults[#, tz, Automatic, Automatic]&/@(("Result" /. api)[[All, 2]]), {("Result" /. api)[[All, 1]]}],
   		"GraphicsData"|"TotalPhasePolygon"|"PartialPhasePolygon"|"TotalPhaseCenterLine", EventSeries[fixGraphicsForGeoPosition[(("Result" /. api))[[All, 2]]], {(("Result" /. api))[[All, 1]]}],
   		"Type", EventSeries[("Result" /. api)[[All, 2]], {("Result" /. api)[[All, 1]]}],
   		_, Missing["NotAvailable"]
   	], $Failed]
  ]

iSolarEclipse[date:{_?(validDateQ[#, SolarEclipse]&)..}, prop_, OptionsPattern[SolarEclipse]] := (Message[SolarEclipse::noprop, HoldForm[prop]];$Failed)

iSolarEclipse[{start_?(validDateQ[#, SolarEclipse]||validDateStringQ[#, SolarEclipse]&), end_?(validDateQ[#, SolarEclipse]||validDateStringQ[#, SolarEclipse]&), All}, prop:"GraphicsData"|"PartialPhasePolygon"|"TotalPhasePolygon"|"TotalPhaseCenterLine"|"Type", opts:OptionsPattern[SolarEclipse]] := Module[{tz, igeolocation, apiresult, ei, et, api, placeholderloc},
  tz = TimeZoneOffset[OptionValue[TimeZone]];
  ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
  et = OptionValue[EclipseType];
  If[!validTimeZoneQ[tz], Message[SolarEclipse::zone, tz];tz = $SystemTimeZone];
  If[!validTimeDirection[ei, SolarEclipse], Message[SolarEclipse::evincr, ei];ei=Automatic];
  If[!MatchQ[et, Automatic|"Annular"|"Hybrid"|"Partial"|"Total"], Message[SolarEclipse::ectype, et];et=Automatic];
  placeholderloc = GeoPosition[{40.1, -88.2}];
  igeolocation = preProcessCoordinates[placeholderloc]/.GeoPosition[x_]:>x;
  api = ReleaseHold[APICompute[SolarEclipse,"AstronomyConvenienceFunction", {"SolarEclipse", prop, igeolocation, DateList/@{start, end}, ei, et, $TimeZone, tz, igeolocation, {All, "v2"}}]];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[SolarEclipse::time] 
  ];
   apiresult = If[!MatchQ[api, _Missing|$Failed|_$Failed], 
   fixGraphicsForGeoPosition[("Result" /. api)], 
   	$Failed]/.$Failed -> Missing["NotAvailable"]
  ]

iSolarEclipse[{start_?(validDateQ[#, SolarEclipse]||validDateStringQ[#, SolarEclipse]&), end_?(validDateQ[#, SolarEclipse]||validDateStringQ[#, SolarEclipse]&), All}, prop:"MaximumEclipseDate"|"TotalPhaseStartDate"|"TotalPhaseEndDate", opts:OptionsPattern[SolarEclipse]] := Module[{tz, igeolocation, apiresult, ei, et, api, placeholderloc},
  tz = TimeZoneOffset[OptionValue[TimeZone]];
  ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
  et = OptionValue[EclipseType];
  If[!validTimeZoneQ[tz], Message[SolarEclipse::zone, tz];tz = $SystemTimeZone];
  If[!validTimeDirection[ei, SolarEclipse], Message[SolarEclipse::evincr, ei];ei=Automatic];
  If[!MatchQ[et, Automatic|"Annular"|"Hybrid"|"Partial"|"Total"], Message[SolarEclipse::ectype, et];et=Automatic];
  placeholderloc = GeoPosition[{40.1, -88.2}];
  igeolocation = preProcessCoordinates[placeholderloc]/.GeoPosition[x_]:>x;
  api = ReleaseHold[APICompute[SolarEclipse,"AstronomyConvenienceFunction", {"SolarEclipse", prop, igeolocation, DateList/@{start, end}, ei, et, $TimeZone, tz, igeolocation, {All, "v2"}}]];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[SolarEclipse::time] 
  ];
   apiresult = If[!MatchQ[api, _Missing|$Failed|_$Failed], 
   	postProcessDateResults[#, tz, Automatic, Automatic]&/@("Result" /. api), 
   	$Failed]/.$Failed -> Missing["NotAvailable"]
  ] 

iSolarEclipse[{start_?(validDateQ[#, SolarEclipse]||validDateStringQ[#, SolarEclipse]&), end_?(validDateQ[#, SolarEclipse]||validDateStringQ[#, SolarEclipse]&), incr_}, prop:"GraphicsData"|"PartialPhasePolygon"|"TotalPhasePolygon"|"TotalPhaseCenterLine"|"Type"|"MaximumEclipseDate"|"TotalPhaseStartDate"|"TotalPhaseEndDate", opts:OptionsPattern[SolarEclipse]] := 
 iSolarEclipse[DateRange[start, end, incr], prop, opts]

iSolarEclipse[{start_?(validDateQ[#, SolarEclipse]||validDateStringQ[#, SolarEclipse]&), end_?(validDateQ[#, SolarEclipse]||validDateStringQ[#, SolarEclipse]&), _}, prop_, opts:OptionsPattern[SolarEclipse]] := (Message[SolarEclipse::noprop, HoldForm[prop]];$Failed)

iSolarEclipse[args___, opts:OptionsPattern[SolarEclipse]]:=(Message[SolarEclipse::dtspec, HoldForm[args]];$Failed)
  
SolarEclipse[args___, opts:OptionsPattern[SolarEclipse]] /; (ArgumentCountQ[SolarEclipse,Length[DeleteCases[{args}, _Rule]],0,2]) := Block[{res},
  res = iSolarEclipse[args, opts];
  res /; !MatchQ[res, _Missing|$Failed]
  ]/;FreeQ[{args},_Rule,{1}]

(* Lunar Eclipse functionality *)
Clear[iLunarEclipse];
iLunarEclipse["Properties"] = {"GraphicsData", "MaximumEclipseDate", "PartialPhaseEndDate", "PartialPhaseStartDate", 
	"PartialPhasePolygon", "TotalPhaseEndDate", "TotalPhaseStartDate", "TotalPhasePolygon", "Type"}

iLunarEclipse[OptionsPattern[LunarEclipse]] := Module[{tz, igeolocation, apiresult, ei, et, api, placeholderloc},
  tz = TimeZoneOffset[OptionValue[TimeZone]];
  ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
  et = OptionValue[EclipseType];
  If[!validTimeZoneQ[tz], Message[LunarEclipse::zone, tz];tz = $SystemTimeZone];
  If[!validTimeDirection[ei, LunarEclipse], Message[LunarEclipse::evincr, ei];ei=Automatic];
  If[!MatchQ[et, Automatic|"Partial"|"Total"], Message[LunarEclipse::ectype, et];et=Automatic];
  placeholderloc = GeoPosition[{40.1, -88.2}];
  igeolocation = preProcessCoordinates[placeholderloc]/.GeoPosition[x_]:>x;
  api = ReleaseHold[APICompute[LunarEclipse,"AstronomyConvenienceFunction", {"LunarEclipse", "MaximumEclipseDate", igeolocation, DateList[], ei, et, $TimeZone, tz, igeolocation}]];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[LunarEclipse::time] 
  ];
   apiresult = If[!MatchQ[api, $Failed|_$Failed], postProcessDateResults["Result" /. api, tz, Automatic, Automatic], $Failed]
  ]

iLunarEclipse[date_?(validDateQ[#, LunarEclipse]&), OptionsPattern[LunarEclipse]] := Module[{tz, igeolocation, apiresult, ei, et, api, placeholderloc},
  tz = TimeZoneOffset[OptionValue[TimeZone]];
  ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
  et = OptionValue[EclipseType];
  If[!validTimeZoneQ[tz], Message[LunarEclipse::zone, tz];tz = $SystemTimeZone];
  If[!validTimeDirection[ei, LunarEclipse], Message[LunarEclipse::evincr, ei];ei=Automatic];
  If[!MatchQ[et, Automatic|"Partial"|"Total"], Message[LunarEclipse::ectype, et];et=Automatic];
  placeholderloc = GeoPosition[{40.1, -88.2}];
  igeolocation = preProcessCoordinates[placeholderloc]/.GeoPosition[x_]:>x;
  api = ReleaseHold[APICompute[LunarEclipse,"AstronomyConvenienceFunction", {"LunarEclipse", "MaximumEclipseDate", igeolocation, DateList[date], ei, et, $TimeZone, tz, igeolocation}]];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[LunarEclipse::time] 
  ];
   apiresult = If[!MatchQ[api, $Failed|_$Failed], postProcessDateResults["Result" /. api, tz, Automatic, Automatic], $Failed]
  ]

iLunarEclipse[date:{_?(validDateQ[#, LunarEclipse]&)..}, OptionsPattern[LunarEclipse]] := Module[{tz, igeolocation, apiresult, ei, et, api, placeholderloc},
  tz = TimeZoneOffset[OptionValue[TimeZone]];
  ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
  et = OptionValue[EclipseType];
  If[!validTimeZoneQ[tz], Message[LunarEclipse::zone, tz];tz = $SystemTimeZone];
  If[!validTimeDirection[ei, LunarEclipse], Message[LunarEclipse::evincr, ei];ei=Automatic];
  If[!MatchQ[et, Automatic|"Partial"|"Total"], Message[LunarEclipse::ectype, et];et=Automatic];
  placeholderloc = GeoPosition[{40.1, -88.2}];
  igeolocation = preProcessCoordinates[placeholderloc]/.GeoPosition[x_]:>x;
  api = ReleaseHold[APICompute[LunarEclipse,"AstronomyConvenienceFunction", {"LunarEclipse", "MaximumEclipseDate", igeolocation, DateList/@date, ei, et, $TimeZone, tz, igeolocation}]];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[LunarEclipse::time] 
  ];
   apiresult = If[!MatchQ[api, _Missing|$Failed|_$Failed], EventSeries[postProcessDateResults[#, tz, Automatic, Automatic]&/@(("Result" /. api)[[All, 2]]), {("Result" /. api)[[All, 1]]}], $Failed]
  ]

iLunarEclipse[{start_?(validDateQ[#, LunarEclipse]||validDateStringQ[#, LunarEclipse]&), end_?(validDateQ[#, LunarEclipse]||validDateStringQ[#, LunarEclipse]&), All}, opts:OptionsPattern[LunarEclipse]] := Module[{tz, igeolocation, apiresult, ei, et, api, placeholderloc},
  tz = TimeZoneOffset[OptionValue[TimeZone]];
  ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
  et = OptionValue[EclipseType];
  If[!validTimeZoneQ[tz], Message[LunarEclipse::zone, tz];tz = $SystemTimeZone];
  If[!validTimeDirection[ei, LunarEclipse], Message[LunarEclipse::evincr, ei];ei=Automatic];
  If[!MatchQ[et, Automatic|"Partial"|"Total"], Message[LunarEclipse::ectype, et];et=Automatic];
  placeholderloc = GeoPosition[{40.1, -88.2}];
  igeolocation = preProcessCoordinates[placeholderloc]/.GeoPosition[x_]:>x;
  api = ReleaseHold[APICompute[LunarEclipse,"AstronomyConvenienceFunction", {"LunarEclipse", "MaximumEclipseDate", igeolocation, DateList/@{start, end}, ei, et, $TimeZone, tz, igeolocation, {All, "v2"}}]];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[LunarEclipse::time] 
  ];
   apiresult = If[!MatchQ[api, _Missing|$Failed|_$Failed], 
   postProcessDateResults[#, tz, Automatic, Automatic]&/@("Result" /. api), $Failed]/.$Failed -> Missing["NotAvailable"]
  ]

iLunarEclipse[{start_?(validDateQ[#, LunarEclipse]||validDateStringQ[#, LunarEclipse]&), end_?(validDateQ[#, LunarEclipse]||validDateStringQ[#, LunarEclipse]&), incr_}, opts:OptionsPattern[LunarEclipse]] := 
 iLunarEclipse[DateRange[start, end, incr], opts]

iLunarEclipse[prop:"MaximumEclipseDate"|"TotalPhaseStartDate"|"TotalPhaseEndDate"|"TotalPhaseStartDate"|"PartialPhaseStartDate"|"PartialPhaseEndDate", OptionsPattern[LunarEclipse]] := Module[{tz, igeolocation, apiresult, ei, et, api, placeholderloc},
  tz = TimeZoneOffset[OptionValue[TimeZone]];
  ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
  et = OptionValue[EclipseType];
  If[!validTimeZoneQ[tz], Message[LunarEclipse::zone, tz];tz = $SystemTimeZone];
  If[!validTimeDirection[ei, LunarEclipse], Message[LunarEclipse::evincr, ei];ei=Automatic];
  If[!MatchQ[et, Automatic|"Partial"|"Total"], Message[LunarEclipse::ectype, et];et=Automatic];
  placeholderloc = GeoPosition[{40.1, -88.2}];
  igeolocation = preProcessCoordinates[placeholderloc]/.GeoPosition[x_]:>x;
  api = ReleaseHold[APICompute[LunarEclipse,"AstronomyConvenienceFunction", {"LunarEclipse", prop, igeolocation, DateList[], ei, et, $TimeZone, tz, igeolocation}]];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[LunarEclipse::time] 
  ];
   apiresult = If[!MatchQ[api, $Failed|_$Failed], postProcessDateResults["Result" /. api, tz, Automatic, Automatic], $Failed]
  ]

iLunarEclipse[prop:"GraphicsData"|"PartialPhasePolygon"|"TotalPhasePolygon"|"Type", OptionsPattern[LunarEclipse]] := Module[{tz, igeolocation, apiresult, ei, et, api, placeholderloc},
  tz = TimeZoneOffset[OptionValue[TimeZone]];
  ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
  et = OptionValue[EclipseType];
  If[!validTimeZoneQ[tz], Message[LunarEclipse::zone, tz];tz = $SystemTimeZone];
  If[!validTimeDirection[ei, LunarEclipse], Message[LunarEclipse::evincr, ei];ei=Automatic];
  If[!MatchQ[et, Automatic|"Partial"|"Total"], Message[LunarEclipse::ectype, et];et=Automatic];
  placeholderloc = GeoPosition[{40.1, -88.2}];
  igeolocation = preProcessCoordinates[placeholderloc]/.GeoPosition[x_]:>x;
  api = ReleaseHold[APICompute[LunarEclipse,"AstronomyConvenienceFunction", {"LunarEclipse", prop, igeolocation, DateList[], ei, et, $TimeZone, tz, igeolocation}]];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[LunarEclipse::time] 
  ];
   apiresult = If[!MatchQ[api, $Failed|_$Failed], 
   	fixGraphicsForGeoPosition[("Result" /. api)], 
   	$Failed]
  ]

iLunarEclipse[date_?(validDateQ[#, LunarEclipse]&), prop:"MaximumEclipseDate"|"TotalPhaseStartDate"|"TotalPhaseEndDate"|"TotalPhaseStartDate"|"PartialPhaseStartDate"|"PartialPhaseEndDate", OptionsPattern[LunarEclipse]] := Module[{tz, igeolocation, apiresult, ei, et, api, placeholderloc},
  tz = TimeZoneOffset[OptionValue[TimeZone]];
  ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
  et = OptionValue[EclipseType];
  If[!validTimeZoneQ[tz], Message[LunarEclipse::zone, tz];tz = $SystemTimeZone];
  If[!validTimeDirection[ei, LunarEclipse], Message[LunarEclipse::evincr, ei];ei=Automatic];
  If[!MatchQ[et, Automatic|"Partial"|"Total"], Message[LunarEclipse::ectype, et];et=Automatic];
  placeholderloc = GeoPosition[{40.1, -88.2}];
  igeolocation = preProcessCoordinates[placeholderloc]/.GeoPosition[x_]:>x;
  api = ReleaseHold[APICompute[LunarEclipse,"AstronomyConvenienceFunction", {"LunarEclipse", prop, igeolocation, DateList[date], ei, et, $TimeZone, tz, igeolocation}]];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[LunarEclipse::time] 
  ];
   apiresult = If[!MatchQ[api, $Failed|_$Failed], postProcessDateResults["Result" /. api, tz, Automatic, Automatic], $Failed]
  ]

iLunarEclipse[date_?(validDateQ[#, LunarEclipse]&), prop:"GraphicsData"|"PartialPhasePolygon"|"TotalPhasePolygon"|"Type", OptionsPattern[LunarEclipse]] := Module[{tz, igeolocation, apiresult, ei, et, api, placeholderloc},
  tz = TimeZoneOffset[OptionValue[TimeZone]];
  ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
  et = OptionValue[EclipseType];
  If[!validTimeZoneQ[tz], Message[LunarEclipse::zone, tz];tz = $SystemTimeZone];
  If[!validTimeDirection[ei, LunarEclipse], Message[LunarEclipse::evincr, ei];ei=Automatic];
  If[!MatchQ[et, Automatic|"Partial"|"Total"], Message[LunarEclipse::ectype, et];et=Automatic];
  placeholderloc = GeoPosition[{40.1, -88.2}];
  igeolocation = preProcessCoordinates[placeholderloc]/.GeoPosition[x_]:>x;
  api = ReleaseHold[APICompute[LunarEclipse,"AstronomyConvenienceFunction", {"LunarEclipse", prop, igeolocation, DateList[date], ei, et, $TimeZone, tz, igeolocation}]];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[LunarEclipse::time] 
  ];
   apiresult = If[!MatchQ[api, $Failed|_$Failed], 
   	fixGraphicsForGeoPosition[("Result" /. api)], 
   	$Failed]
  ]

iLunarEclipse[date_?(validDateQ[#, LunarEclipse]&), prop_, OptionsPattern[LunarEclipse]]:= (Message[LunarEclipse::noprop, HoldForm[prop]];$Failed)

iLunarEclipse[date:{_?(validDateQ[#, LunarEclipse]&)..}, prop:"GraphicsData"|"PartialPhasePolygon"|"TotalPhasePolygon"|"Type"|"MaximumEclipseDate"|"TotalPhaseStartDate"|"TotalPhaseEndDate"|"TotalPhaseStartDate"|"PartialPhaseStartDate"|"PartialPhaseEndDate", OptionsPattern[LunarEclipse]] := Module[{tz, igeolocation, apiresult, ei, et, api, placeholderloc},
  tz = TimeZoneOffset[OptionValue[TimeZone]];
  ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
  et = OptionValue[EclipseType];
  If[!validTimeZoneQ[tz], Message[LunarEclipse::zone, tz];tz = $SystemTimeZone];
  If[!validTimeDirection[ei, LunarEclipse], Message[LunarEclipse::evincr, ei];ei=Automatic];
  If[!MatchQ[et, Automatic|"Partial"|"Total"], Message[LunarEclipse::ectype, et];et=Automatic];
  placeholderloc = GeoPosition[{40.1, -88.2}];
  igeolocation = preProcessCoordinates[placeholderloc]/.GeoPosition[x_]:>x;
  api = ReleaseHold[APICompute[LunarEclipse,"AstronomyConvenienceFunction", {"LunarEclipse", prop, igeolocation, DateList/@date, ei, et, $TimeZone, tz, igeolocation}]];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[LunarEclipse::time] 
  ];
  apiresult = If[!MatchQ[api, _Missing|$Failed|_$Failed], 
   	Switch[prop,
   		"MaximumEclipseDate"|"TotalPhaseStartDate"|"TotalPhaseEndDate"|"TotalPhaseStartDate"|"PartialPhaseStartDate", EventSeries[postProcessDateResults[#, tz, Automatic, Automatic]&/@(("Result" /. api)[[All, 2]]), {("Result" /. api)[[All, 1]]}],
   		"GraphicsData"|"PartialPhasePolygon"|"TotalPhasePolygon", EventSeries[fixGraphicsForGeoPosition[(("Result" /. api))[[All, 2]]], {(("Result" /. api))[[All, 1]]}],
   		"Type", EventSeries[("Result" /. api)[[All, 2]], {("Result" /. api)[[All, 1]]}],
   		_, Missing["NotAvailable"]
   	], $Failed]
  ]

iLunarEclipse[date:{_?(validDateQ[#, LunarEclipse]&)..}, prop_, OptionsPattern[LunarEclipse]] := (Message[LunarEclipse::noprop, HoldForm[prop]];$Failed)

iLunarEclipse[{start_?(validDateQ[#, LunarEclipse]||validDateStringQ[#, LunarEclipse]&), end_?(validDateQ[#, LunarEclipse]||validDateStringQ[#, LunarEclipse]&), All}, prop:"MaximumEclipseDate"|"TotalPhaseStartDate"|"TotalPhaseEndDate"|"TotalPhaseStartDate"|"PartialPhaseStartDate"|"PartialPhaseEndDate", opts:OptionsPattern[LunarEclipse]] := Module[{tz, igeolocation, apiresult, ei, et, api, placeholderloc},
  tz = TimeZoneOffset[OptionValue[TimeZone]];
  ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
  et = OptionValue[EclipseType];
  If[!validTimeZoneQ[tz], Message[LunarEclipse::zone, tz];tz = $SystemTimeZone];
  If[!validTimeDirection[ei, LunarEclipse], Message[LunarEclipse::evincr, ei];ei=Automatic];
  If[!MatchQ[et, Automatic|"Partial"|"Total"], Message[LunarEclipse::ectype, et];et=Automatic];
  placeholderloc = GeoPosition[{40.1, -88.2}];
  igeolocation = preProcessCoordinates[placeholderloc]/.GeoPosition[x_]:>x;
  api = ReleaseHold[APICompute[LunarEclipse,"AstronomyConvenienceFunction", {"LunarEclipse", prop, igeolocation, DateList/@{start, end}, ei, et, $TimeZone, tz, igeolocation, {All, "v2"}}]];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[LunarEclipse::time] 
  ];
  apiresult = If[!MatchQ[api, _Missing|$Failed|_$Failed], 
   	postProcessDateResults[#, tz, Automatic, Automatic]&/@("Result" /. api), $Failed]/.$Failed -> Missing["NotAvailable"]
  ]

iLunarEclipse[{start_?(validDateQ[#, LunarEclipse]||validDateStringQ[#, LunarEclipse]&), end_?(validDateQ[#, LunarEclipse]||validDateStringQ[#, LunarEclipse]&), All}, prop:"GraphicsData"|"PartialPhasePolygon"|"TotalPhasePolygon"|"Type", opts:OptionsPattern[LunarEclipse]] := Module[{tz, igeolocation, apiresult, ei, et, api, placeholderloc},
  tz = TimeZoneOffset[OptionValue[TimeZone]];
  ei = OptionValue[TimeDirection]/.{"Next"|"After"|Next|After->1, "Last"|"Previous"|"Before"|Previous|Last|Before->-1};
  et = OptionValue[EclipseType];
  If[!validTimeZoneQ[tz], Message[LunarEclipse::zone, tz];tz = $SystemTimeZone];
  If[!validTimeDirection[ei, LunarEclipse], Message[LunarEclipse::evincr, ei];ei=Automatic];
  If[!MatchQ[et, Automatic|"Partial"|"Total"], Message[LunarEclipse::ectype, et];et=Automatic];
  placeholderloc = GeoPosition[{40.1, -88.2}];
  igeolocation = preProcessCoordinates[placeholderloc]/.GeoPosition[x_]:>x;
  api = ReleaseHold[APICompute[LunarEclipse,"AstronomyConvenienceFunction", {"LunarEclipse", prop, igeolocation, DateList/@{start, end}, ei, et, $TimeZone, tz, igeolocation, {All, "v2"}}]];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[LunarEclipse::time] 
  ];
  apiresult = If[!MatchQ[api, _Missing|$Failed|_$Failed], 
   	fixGraphicsForGeoPosition[("Result" /. api)], 
   	$Failed]/.$Failed -> Missing["NotAvailable"]
  ]

iLunarEclipse[{start_?(validDateQ[#, LunarEclipse]||validDateStringQ[#, LunarEclipse]&), end_?(validDateQ[#, LunarEclipse]||validDateStringQ[#, LunarEclipse]&), incr_}, prop:"GraphicsData"|"MaximumEclipseDate"|"PartialPhaseEndDate"|"PartialPhaseStartDate"|"PartialPhasePolygon"|"TotalPhaseEndDate"|"TotalPhaseStartDate"|"TotalPhasePolygon"|"Type", opts:OptionsPattern[LunarEclipse]] := 
 iLunarEclipse[DateRange[start, end, incr], prop, opts]

iLunarEclipse[{start_?(validDateQ[#, LunarEclipse]||validDateStringQ[#, LunarEclipse]&), end_?(validDateQ[#, LunarEclipse]||validDateStringQ[#, LunarEclipse]&), _}, prop_, opts:OptionsPattern[LunarEclipse]] := (Message[LunarEclipse::noprop, HoldForm[prop]];$Failed)

iLunarEclipse[args___, opts:OptionsPattern[LunarEclipse]]:=(Message[LunarEclipse::dtspec, HoldForm[args]];$Failed)
  
LunarEclipse[args___, opts:OptionsPattern[LunarEclipse]] /; (ArgumentCountQ[LunarEclipse,Length[DeleteCases[{args}, _Rule]],0,2]) := Block[{res},
  res = iLunarEclipse[args, opts];
  res /; !MatchQ[res, _Missing|$Failed]
  ]/;FreeQ[{args},_Rule,{1}]

(* SiderealTime *)

iSiderealTime[args___] := Module[{tz, igeolocation, msam, dt, api, ipairs},
  tz = $TimeZone;
  If[!validTimeZoneQ[tz], Message[SiderealTime::zone, tz];tz = $SystemTimeZone];
  msam = makeSiderealTimeArgumentsAndMessages[{args}];
  issueParserMessages[SiderealTime, msam[[2]]];
  If[msam[[1]]==={}, Return[$Failed]];
    If[MatchQ[msam[[1]], {{{_?(validLocationQ[#, SiderealTime]&), _?(validDateQ[#, SiderealTime]&)}..}, _}],
		ipairs = {preProcessCoordinates[#[[1]]], (#[[2]]/.d_DateObject :> DateList[d])/.x_String/;MatchQ[Quiet[DateList[x]], _List]:>DateList[x]}&/@(msam[[1, 1]]/.GeoPosition[x_]:>x);
		api = AstronomyConvenienceFunction["SiderealTime", ipairs, $TimeZone, tz, $GeoLocation/.GeoPosition[x_]:>x[[1;;2]], msam[[1,2]]];
		api,
  igeolocation = preProcessCoordinates[msam[[1,1]]]/.GeoPosition[x_]:>x;
  If[MatchQ[igeolocation, {}], Message[SiderealTime::loc, msam[[1,1]]]];
  dt=((msam[[1,2]]/.{start_?(validDateQ[#, SiderealTime]&), end_?(validDateQ[#, SiderealTime]&), incr_?(validIncrementQ[#]&)}:>DateRange[start, end, incr])/.d_DateObject :> DateList[d])/.x_String/;MatchQ[Quiet[DateList[x]], _List]:>DateList[x];
  If[!FreeQ[dt, $Failed|_$Failed|_DateRange], Return[$Failed]];
  api = AstronomyConvenienceFunction["SiderealTime", igeolocation, dt, $TimeZone, tz, $GeoLocation/.GeoPosition[x_]:>x[[1;;2]], msam[[1,3]]];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[SiderealTime::time] 
  	];
	Which[
		validDateQ[dt, SiderealTime]&&validLocationQ[msam[[1,1]], SiderealTime], If[!MatchQ[api, $Failed|_$Failed], 
			(api)/.q_Quantity:>UnitConvert[q, MixedRadix["HoursOfRightAscension", "MinutesOfRightAscension", "SecondsOfRightAscension"]], 
			$Failed],
		MatchQ[dt, {_?(validDateQ[#, SiderealTime]&)..}]&&validLocationQ[msam[[1,1]], SiderealTime], If[!MatchQ[api, $Failed|_$Failed],
			TimeSeries[((api)[[All, 2]])/.q_Quantity:>UnitConvert[q, MixedRadix["HoursOfRightAscension", "MinutesOfRightAscension", "SecondsOfRightAscension"]], {(api)[[All, 1]]}, ResamplingMethod -> {"Interpolation", InterpolationOrder->1}], 
			$Failed],
		validDateQ[dt, SiderealTime]&&MatchQ[msam[[1,1]], {_?(validLocationQ[#, SiderealTime]&)..}], If[!MatchQ[api, $Failed|_$Failed],
			(api)/.q_Quantity:>UnitConvert[q, MixedRadix["HoursOfRightAscension", "MinutesOfRightAscension", "SecondsOfRightAscension"]], 
			$Failed],
		MatchQ[dt, {_?(validDateQ[#, SiderealTime]&)..}]&&MatchQ[msam[[1,1]], {_?(validLocationQ[#, SiderealTime]&)..}], If[!MatchQ[api, $Failed|_$Failed],
			Function[{loc}, TimeSeries[(loc[[All, 2]])/.q_Quantity:>UnitConvert[q, MixedRadix["HoursOfRightAscension", "MinutesOfRightAscension", "SecondsOfRightAscension"]], {loc[[All, 1]]}, ResamplingMethod -> {"Interpolation", InterpolationOrder->1}]]/@(api), 
			$Failed],
		True, $Failed
	]
	]
   ]
   
SiderealTime[args___, opts:OptionsPattern[]] /; (ArgumentCountQ[SiderealTime,Length[{args}],0,3]) := Block[{res},
  res = iSiderealTime[args, opts];
  res /; !MatchQ[res, _Missing|$Failed]
  ]/;FreeQ[{args},_Rule,{1}]

Clear[iDaylightQ];
iDaylightQ[args___, OptionsPattern[DaylightQ]] := Module[{tz, igeolocation, msam, dt, api, ipairs, sunangletest},
  tz = $TimeZone;
  sunangletest = OptionValue["SunAngleTest"];
  If[!validTimeZoneQ[tz], Message[DaylightQ::zone, tz];tz = $SystemTimeZone];
  msam = makeDaylightQArgumentsAndMessages[{args}];
  issueParserMessages[DaylightQ, msam[[2]]];
  If[msam[[1]]==={}, Return[$Failed]];
    If[MatchQ[msam[[1]], {{{_?(validLocationQ[#, DaylightQ]&), _?(validDateQ[#, DaylightQ]&)}..}, _}],
    	If[!MatchQ[msam[[1,2]], "Everywhere"|"Somewhere"|"Nowhere"], Message[DaylightQ::arg, msam[[1,2]]];msam[[1,2]]="Everywhere"];
		ipairs = {preProcessCoordinates[#[[1]]], (#[[2]]/.d_DateObject :> DateList[d])/.x_String/;MatchQ[Quiet[DateList[x]], _List]:>DateList[x]}&/@(msam[[1,1]]/.GeoPosition[x_]:>x);
		api = AstronomyConvenienceFunction["SunPosition", ipairs, "Horizon", $TimeZone, tz, $GeoLocation/.GeoPosition[x_]:>x[[1;;2]], Automatic, "TrueAltitude"];
		If[MatchQ[#, {_Quantity, _Quantity}],
				Which[
				msam[[1,2]]==="Everywhere",
				If[(TrueQ[Or[Evaluate@sunangletest[QuantityMagnitude@Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}]],Evaluate@sunangletest[QuantityMagnitude[Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]])&&(TrueQ[Or[Evaluate@sunangletest[Min[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}]],Evaluate@sunangletest[QuantityMagnitude[Min[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]]), True, False],
				msam[[1,2]]==="Somewhere",
				If[TrueQ[Or[Evaluate@sunangletest[QuantityMagnitude@Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}]],Evaluate@sunangletest[QuantityMagnitude[Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]], True, False],
				msam[[1,2]]==="Nowhere",
				If[(TrueQ[Or[Evaluate@sunangletest[QuantityMagnitude@Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}]],Evaluate@sunangletest[QuantityMagnitude[Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]])&&(TrueQ[Or[Evaluate@sunangletest[Min[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}]],Evaluate@sunangletest[QuantityMagnitude[Min[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]]), False, True]
				],
				Missing["NotAvailable"]
				]&/@(api),
  If[!MatchQ[msam[[1,3]], "Everywhere"|"Somewhere"|"Nowhere"], Message[DaylightQ::arg, msam[[1,3]]];msam[[1,3]]="Everywhere"];
  igeolocation = preProcessCoordinates[msam[[1,1]]]/.GeoPosition[x_]:>x;
  If[MatchQ[igeolocation, {}], Message[DaylightQ::loc, msam[[1,1]]]];
  dt=((msam[[1,2]]/.{start_?(validDateQ[#, DaylightQ]&), end_?(validDateQ[#, DaylightQ]&), incr_?(validIncrementQ[#]&)}:>DateRange[start, end, incr])/.d_DateObject :> DateList[d])/.x_String/;MatchQ[Quiet[DateList[x]], _List]:>DateList[x];
  If[!FreeQ[dt, $Failed|_$Failed|_DateRange], Return[$Failed]];
  api = AstronomyConvenienceFunction["SunPosition", igeolocation, dt, "Horizon", $TimeZone, tz, $GeoLocation/.GeoPosition[x_]:>x[[1;;2]], Automatic, "TrueAltitude"];
  If[!FreeQ[api, $Failed["ComputationTimeout"]],
  	Message[DaylightQ::time] 
  	];
	Which[
		validDateQ[dt, DaylightQ]&&validLocationQ[msam[[1,1]], DaylightQ], If[!MatchQ[api, $Failed|_$Failed], 
			If[MatchQ[api, {_Quantity, _Quantity}],
				Which[
				msam[[1,3]]==="Everywhere",
				If[(TrueQ[Or[Evaluate@sunangletest[QuantityMagnitude[Max[(api)[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]],Evaluate@sunangletest[QuantityMagnitude[Max[(api)[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]])&&(TrueQ[Or[Evaluate@sunangletest[QuantityMagnitude[Min[(api)[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]],Evaluate@sunangletest[QuantityMagnitude[Min[(api)[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]]), True, False],
				msam[[1,3]]==="Somewhere",
				If[TrueQ[Or[Evaluate@sunangletest[QuantityMagnitude[Max[(api)[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]],Evaluate@sunangletest[QuantityMagnitude[Max[(api)[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]], True, False],
				msam[[1,3]]==="Nowhere",
				If[(TrueQ[Or[Evaluate@sunangletest[QuantityMagnitude[Max[(api)[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]],Evaluate@sunangletest[QuantityMagnitude[Max[(api)[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]])&&(TrueQ[Or[Evaluate@sunangletest[QuantityMagnitude[Min[(api)[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]],Evaluate@sunangletest[QuantityMagnitude[Min[(api)[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]]), False, True]
				],
				Missing["NotAvailable"]
				], 
			$Failed],
		MatchQ[dt, {_?(validDateQ[#, DaylightQ]&)..}]&&validLocationQ[msam[[1,1]], DaylightQ], If[!MatchQ[api, $Failed|_$Failed],
			EventSeries[If[MatchQ[#, {_Quantity, _Quantity}],
				Which[
				msam[[1,3]]==="Everywhere",
				If[(TrueQ[Or[Evaluate@sunangletest[QuantityMagnitude[Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]],Evaluate@sunangletest[QuantityMagnitude[Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]])&&(TrueQ[Or[Evaluate@sunangletest[QuantityMagnitude[Min[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]],Evaluate@sunangletest[QuantityMagnitude[Min[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]]), True, False],
				msam[[1,3]]==="Somewhere",
				If[TrueQ[Or[Evaluate@sunangletest[QuantityMagnitude[Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]],Evaluate@sunangletest[QuantityMagnitude[Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]], True, False],
				msam[[1,3]]==="Nowhere",
				If[(TrueQ[Or[Evaluate@sunangletest[QuantityMagnitude[Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]],Evaluate@sunangletest[QuantityMagnitude[Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]])&&(TrueQ[Or[Evaluate@sunangletest[QuantityMagnitude[Min[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]],Evaluate@sunangletest[QuantityMagnitude[Min[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]]), False, True]
				],
				Missing["NotAvailable"]
				]&/@((api)[[All, 2]]), {(api)[[All, 1]]}], 
			$Failed],
		validDateQ[dt, DaylightQ]&&MatchQ[msam[[1,1]], {_?(validLocationQ[#, DaylightQ]&)..}], If[!MatchQ[api, $Failed|_$Failed],
			If[MatchQ[#, {_Quantity, _Quantity}],
				Which[
				msam[[1,3]]==="Everywhere",
				If[(TrueQ[Or[Evaluate@sunangletest[QuantityMagnitude[Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]],Evaluate@sunangletest[QuantityMagnitude[Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]])&&(TrueQ[Or[Evaluate@sunangletest[QuantityMagnitude[Min[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]],Evaluate@sunangletest[QuantityMagnitude[Min[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]]), True, False],
				msam[[1,3]]==="Somewhere",
				If[TrueQ[Or[Evaluate@sunangletest[QuantityMagnitude[Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]],Evaluate@sunangletest[QuantityMagnitude[Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]], True, False],
				msam[[1,3]]==="Nowhere",
				If[(TrueQ[Or[Evaluate@sunangletest[QuantityMagnitude[Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]],Evaluate@sunangletest[QuantityMagnitude[Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]])&&(TrueQ[Or[Evaluate@sunangletest[QuantityMagnitude[Min[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]],Evaluate@sunangletest[QuantityMagnitude[Min[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]]), False, True]
				],
				Missing["NotAvailable"]
				]&/@(api), 
			$Failed],
		MatchQ[dt, {_?(validDateQ[#, DaylightQ]&)..}]&&MatchQ[msam[[1,1]], {_?(validLocationQ[#, DaylightQ]&)..}], If[!MatchQ[api, $Failed|_$Failed],
			Function[{loc}, EventSeries[If[MatchQ[#, {_Quantity, _Quantity}],
				Which[
				msam[[1,3]]==="Everywhere",
				If[(TrueQ[Or[Evaluate@sunangletest[QuantityMagnitude[Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]],Evaluate@sunangletest[QuantityMagnitude[Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]])&&(TrueQ[Or[Evaluate@sunangletest[QuantityMagnitude[Min[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]],Evaluate@sunangletest[QuantityMagnitude[Min[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]]), True, False],
				msam[[1,3]]==="Somewhere",
				If[TrueQ[Or[Evaluate@sunangletest[QuantityMagnitude[Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]],Evaluate@sunangletest[QuantityMagnitude[Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]], True, False],
				msam[[1,3]]==="Nowhere",
				If[(TrueQ[Or[Evaluate@sunangletest[QuantityMagnitude[Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]],Evaluate@sunangletest[QuantityMagnitude[Max[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]])&&(TrueQ[Or[Evaluate@sunangletest[QuantityMagnitude[Min[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]],Evaluate@sunangletest[QuantityMagnitude[Min[#[[2]]/.Quantity[Interval[{a_, b_}], unit_] :> {Quantity[a, unit], Quantity[b, unit]}], "AngularDegrees"]]]]), False, True]
				],
				Missing["NotAvailable"]
				]&/@(loc[[All, 2]]), {loc[[All, 1]]}]]/@(api), 
			$Failed],
		True, $Failed
	]
	]
   ]
  
DaylightQ[args___, opts:OptionsPattern[]] /; (ArgumentCountQ[DaylightQ,Length[{args}],0,3]) := Block[{res},
  res = iDaylightQ[args, opts];
  res /; !MatchQ[res, _Missing|$Failed]
  ]/;FreeQ[{args},_Rule,{1}]

End[]

SetAttributes[{System`Sunrise,System`Sunset,System`Sunset,System`SunPosition,System`MoonPosition,
	System`MoonPhase,System`SolarEclipse,System`LunarEclipse, System`SiderealTime,System`CelestialSystem, System`TimeDirection,System`EclipseType,System`DaylightQ,System`AltitudeMethod},
        {ReadProtected, Protected}
];

System`Private`RestoreContextPath[];

{
	System`Sunrise,
	System`Sunset,
	System`Sunset,
	System`SunPosition,
	System`MoonPosition,
	System`MoonPhase,
	System`SolarEclipse,
	System`LunarEclipse,
	System`SiderealTime,
	System`CelestialSystem,
	System`TimeDirection,
	System`EclipseType,
	System`DaylightQ,
	System`AltitudeMethod
	}
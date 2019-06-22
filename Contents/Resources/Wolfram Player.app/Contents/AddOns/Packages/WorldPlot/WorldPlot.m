(* ::Package:: *)

(* :Title: WorldPlot *)

(* :Author: John M. Novak *)

(* :Summary:
This package allows one to plot graphic objects, where
positions are expressed in terms of latitute and longitude
(i.e., maps).  A number of standard map projections are
supported.  Loading the package will also load data 
describing the names and outlines of countries of the world.
*)

(* :Context: WorldPlot` *)

(* :Package Version: 2.0.2 *)

(* :Copyright: Copyright 1990-2007, Wolfram Research, Inc. *)

(* :History: 
	Version 1.0 by John M. Novak (Wolfram Research), November 1990.
	Version 2.0 by John M. Novak; Conversion of 1.0 to
			Mathematica V. 2.0. January 1991.
    Version 2.0.2 by John M. Novak, May 1999. Fix some problems with
            awkward handling of bad results from the projection and
            with broken code in the polyclipint routine.
    Updated Sept. 2006 to fit with new paclet structure for M-- 6.0.
*)

(* :Keywords: cartography, geography, graphics *)

(* :Sources:
	Bradley, A. Day, Mathematics of Map Projections and
		Navigation, Hunter College of the City of New York, 1938.
	Esselte Map Service AB (Sweden), The Concise EARTHBOOK World
		Atlas, Earthbooks Incorporated, 1990.
	Pearson, Frederick II, Map Projections: Theory and
		Applications, CRC Press, 1990.
	Snyder, John P., Map Projections--A Working Manual,
		U.S. Geological Survey Professional Paper 1395,
		United States Government Printing Office, Washington, D.C., 1987.
*)

(* :Warning: adds definitions to the functions Show and Graphics. *)

(* :Mathematica Version: 2.0 *)

(* :Limitation: performs clipping before application of projection,
	so if projection would cause clipping, or has abnormal
	singularities, (e.g., interrupted projection),
	problems may arise in output. Also applies
	if projection joins a section of a map previously unjoined. *)

(* :Limitation: only works with certain primitives: Line, Polygon,
	Point, and Text. Note that the text is not curved to match
	the projection. *)

(* :Limitation: may generate irregularities on boundaries of
	rotated coordinate systems (i.e., when using the WorldRotation
	option.) *)

(* :Limitation: WorldGraphics objects embedded in complex
	graphics (i.e., in GraphicsArray, or Rectangle) do not
	automatically convert to graphics; one must explicitly wrap
	Graphics around them.  *)

(* :Limitation: if the graphic primitive contains a combination of
	scaled and non-scaled coordinates, the non-scaled coordinates
	will not be transformed. *)

(* :Limitation: there is no error checking for unknown options;
	unknown options are simply ignored. *)

(* :Limitation: polygons should not be allowed to cross back on
	themselves; this may cause problems on boundaries. *)

BeginPackage["WorldPlot`"]

If[FileType[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"]]===File,
Select[FindList[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"],"WorldPlot`"],
StringMatchQ[#,StartOfString~~"WorldPlot`*"]&]//ToExpression;
];

Get[ToFileName["WorldPlot","WorldNames.m"]];
Get[ToFileName["WorldPlot","WorldData.m"]];

(* Usage Messages *)

If[Not@ValueQ[WorldPlot::usage],WorldPlot::usage = 
"WorldPlot[list] is a function to generate maps of the world that \
draw data from a specified database, where list is a list of countries \
from the database. Country names are usually specified as strings. \
WorldPlot[{list,shading}] produces a map where shading is either a \
function applied to each name in list, producing a color or grayscale, \
or is a list of shades, one per name in list. A Graphics object or a \
WorldGraphics object can be produced. It accepts WorldGraphics and \
Graphics options."];

If[Not@ValueQ[WorldGraphics::usage],WorldGraphics::usage =
"WorldGraphics[primitives, options] represents a planetary map. It applies \
to standard graphics primitives. Unscaled Lines, Polygons, Points, and \
Text locations are given in minutes of latitude and longitude, and \
transformed to x and y coordinates by the specified projection."];

If[Not@ValueQ[WorldProjection::usage],WorldProjection::usage =
"WorldProjection is an option for WorldGraphics specified as a pure \
function that takes two arguments and returns a list, where the arguments \
are the latitude and longitude of a point, and the list is the x and y \
coordinates of the projection. The default is the Equirectangular \
projection. Note that latitude and longitude are specified in minutes of \
arc."];

If[Not@ValueQ[WorldGrid::usage],WorldGrid::usage =
"WorldGrid is an option for WorldGraphics that places lines of latitude \
and longitude at specified locations. Arguments can be a number \
(which specifies spacing for both latitude and longitude from 0), a pair \
of numbers (which specifies for latitude and longitude, respectively), or \
a pair of lists (which specifies particular locations). All values are in \
degrees. The default is 30."];

If[Not@ValueQ[WorldGridBehind::usage],WorldGridBehind::usage =
"WorldGridBehind is an option for WorldGraphics; it specifies whether the \
lines of longitude and latitude should be rendered behind or in \
front of the graphic; the default is True (behind)."];

If[Not@ValueQ[WorldGridStyle::usage],WorldGridStyle::usage =
"WorldGridStyle is an option for WorldGraphics that specifies the style \
of lines of latitude and longitude."];

If[Not@ValueQ[WorldPoints::usage],WorldPoints::usage =
"WorldPoints is an option for WorldGraphics that specifies the number \
of divisions between the minimum and maximum longitudes of the range, \
with the latitude scaled accordingly. This is the spacing used in the \
grid lines or along the edges of clipped polygons. The default is 100."];

If[Not@ValueQ[WorldBorders::usage],WorldBorders::usage =
"WorldBorders is an option for WorldPlot that can specify a style for \
the borders, or either of the following: None means do not put borders \
around polygons, while Automatic means not to put borders if a shading \
function or explicit shadings are present."];

If[Not@ValueQ[WorldRange::usage],WorldRange::usage =
"WorldRange is an option for WorldGraphics that specifies the range of \
latitude and longitude that will be plotted. It is expressed as \
{{minlat,maxlat},{minlong,maxlong}}. If Automatic, the range is chosen to \
the nearest 10 degrees, encompassing all points to be plotted."];

If[Not@ValueQ[WorldClipping::usage],WorldClipping::usage =
"WorldClipping is an option for WorldGraphics that expresses the type \
of clipping to be done on polygons (in the interest of computational \
efficiency). None will allow any polygon or line that is partially inside \
the WorldRange to remain, Simple will remove them, and Full (the default) \
will properly clip them."];

If[Not@ValueQ[WorldRotation::usage],WorldRotation::usage =
"WorldRotation is an option for WorldGraphics that turns any given \
projection into a Transverse or Oblique projection. It takes a list of \
three parameters, each an angle in degrees; the first expresses a rotation \
about the y axis (assuming the planetary axis starts aligned with the \
z axis, with the x axis through the Prime Meridian), the second rotates \
about the z axis, and the third rotates about the rotated planetary axis. \
The default is {0,0,0}."];

If[Not@ValueQ[WorldBackground::usage],WorldBackground::usage =
"WorldBackground is an option for WorldGraphics. WorldBackground->None \
specifies no background. WorldBackground->style specifies the style \
of polygon representing background (the ocean, for instance)."];

If[Not@ValueQ[WorldFrame::usage],WorldFrame::usage =
"WorldFrame is an option for WorldGraphics. WorldFrame->None specifies \
no frame. WorldFrame->style specifies the style of line around the edge of \
the map."];

If[Not@ValueQ[WorldRotatedRange::usage],WorldRotatedRange::usage =
"WorldRotatedRange is an option for WorldGraphics. If True, the \
WorldRange is applied to the coordinate system formed from the WorldRotation."];

If[Not@ValueQ[WorldFrameParts::usage],WorldFrameParts::usage =
"WorldFrameParts is an option for WorldGraphics. When a WorldFrame is \
rendered, WorldFrameParts allows only parts of it to be shown (particularly \
useful for certain azimuthal projections). It is of the form of \
a list of four zeros and ones; a one indicates that a particular side is \
to be displayed.  The sides are ordered clockwise from the eastern edge \
of the range."];

If[Not@ValueQ[WorldDatabase::usage],WorldDatabase::usage =
"WorldDatabase is an option for WorldPlot that specifies the symbol in \
which polygon data are stored. The default is WorldData."];

If[Not@ValueQ[WorldCountries::usage],WorldCountries::usage =
"WorldCountries is an option for WorldPlot that specifies the list of \
names of polygon sets (countries) in the database, where the names are usually \
specified as strings."];

If[Not@ValueQ[WorldToGraphics::usage],WorldToGraphics::usage =
"WorldToGraphics is an option for WorldPlot. Specifying True or False \
tells whether to produce a Graphics or WorldGraphics object. The default is \
False."];

If[Not@ValueQ[ShowTooltips::usage],ShowTooltips::usage =
"ShowTooltips is an option for WorldPlot that indicates whether or \
not to display Tooltip labels for each country. The default is True."];

If[Not@ValueQ[RandomColors::usage],RandomColors::usage =
"RandomColors is a function to produce random colors."];

If[Not@ValueQ[RandomGrays::usage],RandomGrays::usage =
"RandomGrays is a function to produce random grayscales."];

If[Not@ValueQ[ToMinutes::usage],ToMinutes::usage =
"ToMinutes is a function to convert to minutes; it accepts degrees, \
or degree/minute/second forms. ToMinutes[degs] converts degrees; \
ToMinutes[{degs,mins,secs}] converts from DMS form.  Also, a list of \
coordinates can be converted, \
i.e., ToMinutes[{{{d,m,s},{d,m,s}}, {{d,m,s},{d,m,s}}...}] will \
be converted to {{lat,long},{lat,long}...} with the coordinates in minutes."];

(*
Full::usage =
"Full is an argument for the option WorldClipping.";
*)

If[Not@ValueQ[Simple::usage],Simple::usage =
"Simple is an argument for the option WorldClipping."];

If[Not@ValueQ[Equirectangular::usage],Equirectangular::usage =
"Equirectangular is a map projection for use with WorldGraphics. \
Directly maps longitude to x, latitude to y. It is the simplest \
projection, mathematically."];

If[Not@ValueQ[LambertCylindrical::usage],LambertCylindrical::usage =
"LambertCylindrical is a map projection for use with WorldGraphics."];

If[Not@ValueQ[LambertAzimuthal::usage],LambertAzimuthal::usage =
"LambertAzimuthal is a map projection for use with WorldGraphics. \
Warning: with this projection, you cannot represent the point opposite \
your viewpoint."];

If[Not@ValueQ[Sinusoidal::usage],Sinusoidal::usage =
"Sinusoidal is a map projection for use with WorldGraphics."];

If[Not@ValueQ[Mercator::usage],Mercator::usage =
"Mercator is a map projection for use with WorldGraphics. Warning: \
Mercator goes to Infinity at the poles."];

If[Not@ValueQ[Albers::usage],Albers::usage =
"Albers[p1, p2] is a map projection for use with WorldGraphics, where \
p1, p2 specify the primary latitudes to be used for the projection."];

If[Not@ValueQ[Mollweide::usage],Mollweide::usage =
"Mollweide is a map projection for use with WorldGraphics."];

If[Not@ValueQ[Orthographic::usage],Orthographic::usage =
"Orthographic is a map projection for use with WorldGraphics. Warning: \
the projection is only good for a single hemisphere; ranges must be \
carefully chosen."];

(* Start of Private Section. *)

Begin["`Private`"]

(* Projections *)

Equirectangular = N[{#2,#1} &];

LambertCylindrical = N[{#2,5400 Sin[Degree/60 #1]} &];

LambertAzimuthal = N[Module[{kp},
		kp = Sqrt[2/(1+Cos[Degree/60 #1]Cos[Degree/60 #2])];
		Evaluate[{kp Cos[Degree/60 #1] Sin[Degree/60 #2],
			kp Sin[Degree/60 #1]}]&]];

Sinusoidal = N[{#2 Cos[Degree/60 #1],#1} &];

Mercator = N[{#2,
	60/Degree Log[Tan[(Degree/60 #1)/2 + Pi/4]]} &];

Albers[p1_:0,p2_:0] =
	N[Module[{n, c, r0, th, rho},
		n = (Sin[p1 Degree] + Sin[p2 Degree])/2;
		c = Cos[p1 Degree]^2 + 2 n Sin[p1 Degree];
		r0 = c^(1/2)/n;
		th = n (#2 Degree/60);
		rho = (c - 2n Sin[#1 Degree/60])^(1/2)/n;
		Evaluate[{rho Sin[th],r0 - rho Cos[th]}] &]];

Mollweide = 
	Module[{th},
	(N[{Sqrt[8]/Pi #2 Degree/60 Cos[th],
		Sqrt[2] Sin[th]}/.FindRoot[2 th + Sin[2 th] ==
			Pi Sin[#1/60 Degree],{th,0}]] )&];

Orthographic = N[{Cos[#1 Degree/60] Sin[#2 Degree/60],
		Sin[#1 Degree/60]}&];

(* Declarations of Error Messages. *)

WorldPlot::badshades =
"WorldPlot shading `1` is not valid. Shading should consist of a \
list of Mathematica color primitives such as RGBColor or Hue, or of \
a function that generates such colors when applied to country names.";

WorldPlot::badinput = 
"There is no data for country(ies) `1`.";

WorldGraphics::badrng =
"WorldGraphics range `1` is invalid."

WorldGraphics::badrot =
"Rotation values `1` are invalid."

WorldGraphics::badposn =
"Warning: internal function posnapply given bad arguments `1`; \
substituting {}."

WorldGraphics::badgrid =
"Requested grid `1` is not in a valid form."

WorldGraphics::badint =
"Internal function clipint found an invalid intersection in `1`."

WorldGraphics::badproj =
"The projection function `2` did not return a valid coordinate, returning \
instead the value `1`."

WorldGraphics::clipfail =
"Failure in clipping. Aborting."

WorldGraphics::gcbfail =
"Failure in generating clip borders with arguments `1` and flag `2`. \
Aborting."

WorldGraphics::cbqfail =
"Failure in checking corner in border with arguments `1` and `2`. \
Aborting."

WorldGraphics::genpolys =
"Failure in generating polygons with arguments `1`, `2`, and `3`. \
Aborting."

WorldGraphics::cgenpolys =
"Error generated in internal function genpolys from clipping; \
arguments are `1` and `2`. Aborting."

WorldGraphics::sgenpolys =
"Error generated in internal function genpolys from internal \
function singularitycut; argument is `1`. Aborting."

(* WorldPlot Defns. *)

Clear[WorldPlot]

Options[WorldPlot] = {WorldDatabase->WorldData,
	WorldCountries->World,WorldRange->Automatic,
	WorldBorders->Automatic,WorldToGraphics->False,
	WorldProjection -> Equirectangular,
	WorldGrid->Automatic,WorldGridBehind->True,
	WorldGridStyle->Thickness[.001], WorldPoints->100,
	WorldClipping->Full,WorldRotation -> {0,0,0},
	WorldBackground->None,WorldFrame->Automatic,
	WorldRotatedRange->False,WorldFrameParts->{1,1,1,1},
    ShowTooltips -> True};

WorldPlot[name_String,opts___] :=
	WorldPlot[{{name},Null},opts]

WorldPlot[{names__String},opts___] :=
	WorldPlot[{{names},Null},opts]

WorldPlot[{names:{__String},shadefunc_},opts___] :=
	Module[{shades,s},
		shades = Map[shadefunc,names];
		If[(s = Select[shades,
				!MemberQ[{CMYKColor,RGBColor,Hue,GrayLevel},
					Head[#]]&]) === {},
			WorldPlot[{names,shades},opts],
			Message[WorldPlot::badshades,s];
				Return[$Failed]]]/;
		(shadefunc =!= Null) && (Head[shadefunc] =!= List)

WorldPlot[{names:{__String},shades_},opts___] :=
	Module[{s,database,allcountries,wr,map,bord,tograph,
				dat,wg, tooltipsq},
		(* get options. *)
		{database,allcountries,wr,bord,tograph, tooltipsq} =
			{WorldDatabase,WorldCountries,WorldRange,
				WorldBorders,WorldToGraphics, ShowTooltips}/.
			{opts}/.Options[WorldPlot];
        tooltipsq = TrueQ[tooltipsq];
		(* check shades; if shades exist, pull the ones that
			are color primitives from the list; the resulting
			list should be empty. If not, generate error. *)
		If[shades =!= Null,
			If[(s = DeleteCases[shades,CMYKColor[___] |
					Hue[___] | RGBColor[___] |
					GrayLevel[___]])=!={},
			Message[WorldPlot::badshades,s];
				Return[$Failed]]];

		(* get data; If no data exists for a country, then
			generate an error. *)
		If[(s = Select[dat = Map[database,names],
				Head[#] =!= List &]) =!= {},
			Message[WorldPlot::badinput,s];
				Return[$Failed]];

		(* check if all countries possible are being requested.
			If so, then range is the entire world. *)
		If[And[wr === Automatic,names === allcountries],
			wr = {{-90,90},{-180,180}}];

		(* make polygons; if shades have been generated, put
			the correct shade with the correct set of polygons. 
            If we have tooltip capability, add tooltips. *)
        map = Map[Polygon, dat, {2}];
        If[Context[Tooltip] === "System`" && tooltipsq, (* V6 *)
            If[shades === Null,
                map = {GrayLevel[1], MapThread[Tooltip, {map, names}]},
                map = MapThread[Tooltip[{#1, #2}, #3]&, {shades, map, names}]
            ],
		    If[shades === Null,
			    map = {GrayLevel[1], map},
			    map = Transpose[{shades, map}]
            ]
        ];

		(* if borders are desired, generate borders, append
			to map. *)
		If[bord =!= None,
			If[bord === Automatic,
				If[shades === Null,
					map = {map,Flatten[{{Thickness[.001],
						GrayLevel[0]},Map[Line,dat,{2}]}]}],
				map = {map,Flatten[{bord,Map[Line,dat,{2}]}]}]];

		(* Turn map into WorldGraphics object, then render in
			desired fashion. *)
		wg = WorldGraphics[map,WorldRange->wr,opts];
		If[TrueQ[tograph],
			Show[Graphics[wg]],
			Show[wg]]
		]/;Or[shades===Null, Head[shades]===List]

(* WorldGraphics defns. *)

Clear[WorldGraphics]

Options[WorldGraphics] =  {WorldProjection -> Equirectangular,
	WorldGrid->Automatic,WorldGridBehind->True,
	WorldGridStyle->Thickness[.001], WorldPoints->100,
	WorldRange->Automatic,
	WorldClipping->Full,WorldRotation -> {0,0,0},
	WorldBackground->None,WorldFrame->Automatic,
	WorldRotatedRange->False,WorldFrameParts->{1,1,1,1}};

If[$VersionNumber < 6.0,
    Format[WorldGraphics[___]] := "-WorldGraphics-",
        WorldGraphics /: MakeBoxes[w_WorldGraphics, fmt_] :=
            Typeset`MakeBoxes[#, fmt, Graphics]& @@ {Graphics[w]}
]

WorldGraphics/: Graphics[WorldGraphics[graph_, opts___],
	gopts___] := Module[
	{proj, grid, gb, gstyle, gp, wr, cliptype, tilt, tmp, i, pos,
		dat = If[ListQ[graph], graph, {graph}], ppos, wback, gr, inc, edge,
        frame, rr, fp,
		world = {{-90,90},{-180,180}}, tiltfunc, clipfunc,
		pedge, func, cedge,
		pat = (Polygon[{___List}] | Line[{___List}] |
			Point[_List] | Text[_,_List,___])},
	
	(* get options. *)
	{proj,grid,gb,gstyle,gp,wr,cliptype,tilt,wback,frame,
		rr,fp} = 
		{WorldProjection,WorldGrid,WorldGridBehind,
		WorldGridStyle,WorldPoints,WorldRange,
		WorldClipping,WorldRotation,WorldBackground,
		WorldFrame,WorldRotatedRange,WorldFrameParts}/.
		{opts}/.Options[WorldGraphics];

	(* check range; if Auto, then generate and set. *)
	If[Head[wr] === List,
		wr = Table[Map[If[Abs[#]>90 i,Sign[#] 90 i,#]&,
				wr[[i]]],{i,1,2}],
		tmp = {};
		mapposns[posnapply[AppendTo[tmp,{##}]&,#]&,dat];
		If[tmp =!= {},
            tmp = Transpose[tmp];
            wr = {{Floor[Min[First[tmp]]/600] 10,
                    Ceiling[Max[First[tmp]]/600] 10},
                {Floor[Min[Last[tmp]]/600] 10,
                    Ceiling[Max[Last[tmp]]/600] 10}},
            wr = {{-90, 90}, {-180, 180}}
        ]
	];

	(* check options *)
	If[GreaterEqual @@ wr[[1]],
		Message[WorldGraphics::badrng,wr];
		Return[Graphics[{}]]];
	If[!(VectorQ[tilt] && (Length[tilt] == 3)),
		Message[WorldGraphics::badrot,tilt];
		Return[Graphics[{}]]];

	(* generate useful values and structures *)
	inc = Abs[(Subtract @@ (60 Last[wr]))/gp];
	edge = generateedge[inc,wr];

	(* generate grid; note that if the reverse order
		clipping/rotation is done, the range needs to
		be adjusted... *)
	If[grid =!= None,
		If[TrueQ[rr],
			gr = Flatten[{gstyle,generategrid[grid,inc,world]}],
			gr = Flatten[{gstyle,generategrid[grid,inc,wr]}]];
		If[TrueQ[gb],dat = {gr,dat},dat = {dat,gr}]];

	(* set up for clipping; clipping is not done if entire
		world is displayed; why bother? *)
	If[wr != world,
		If[GreaterEqual @@ wr[[2]],
			cedge = Map[tiltworld[90,0,wr[[2,1]] + 180,
							Sequence @@ #]&,
						Flatten[edge,1]],
			cedge = Flatten[edge,1]];
		clipfunc = clip[#,wr,cliptype,cedge]&,
		clipfunc = Identity[#]&];

	(* set up for rotation. *)
	If[tilt =!= {0,0,0},
		tiltfunc = posnapply[tiltworld[Sequence @@
				({90,0,0} + tilt {-1,1,-1}),
				#1,#2]&,#]&,
		tiltfunc = Identity[#]&];

	(* set up transform function;*)
	If[TrueQ[rr],
		func = Composition[posnapply[proj,#,True]&,
			singularitycut[#,inc]&,clipfunc,tiltfunc],
		func = Composition[posnapply[proj,#,True]&,
			singularitycut[#,inc]&,tiltfunc,clipfunc]];

	(* do transform; note that p is local to the pattern, and
		does not need to be declared in the Module.  *)
	dat = Catch[dat/.p:pat :> func[p], "ProjectionException"];
    If[dat === $Failed, Return[Graphics[{}]]];

	(* generate frame and background *)
	If[frame =!= None,
		If[frame === Automatic, frame = gstyle];
		(* remember that we need to account for partial frames. *)
		pedge = Map[If[#[[1]] == 1,
			Line[#[[2]]],
			{}]&,	Transpose[{fp,edge}]];
		frame = Flatten[{frame,
			If[wr != world && !TrueQ[rr],
				pedge/.p:pat :> func[p],
				pedge/.p:pat :> Composition[posnapply[proj,#]&,
			singularitycut[#,inc]&][p]]}],
		frame = {}];
	
	If[wback =!= None,
		wback = Flatten[{wback,
			If[wr != world && !TrueQ[rr],
				func[Polygon[Flatten[edge,1]]],
				Composition[posnapply[proj,#]&,
			singularitycut[#,inc]&][Polygon[Flatten[edge,1]]]]}],
		wback = {}];

	dat = {wback,dat,frame};

	(* display graphic *)	
	Graphics[dat,
                Sequence@@FilterRules[Flatten[{gopts}],Options[Graphics]],
				Sequence@@FilterRules[Flatten[{opts}],Options[Graphics]],
		AspectRatio->Automatic]
]

mapposns[func_,graph_List] :=
	Module[{pos},
		pos = Position[graph,Polygon[{___List}] | 
			Line[{___List}] | Point[_List] | 
			Text[_,_List,___]];
		MapAt[func,graph,pos]]

Attributes[posnapply] = {Listable};

posnapply[func_, {}, _] := {}

posnapply[func_, shape_Polygon, flag_:False] :=
	Polygon[If[flag, checkposncoords[#, func], #]&[Apply[func,shape[[1]],{1}]]]

posnapply[func_, shape_Line, flag_:False] :=
	Line[If[flag, checkposncoords[#, func], #]&[Apply[func,shape[[1]],{1}]]]

posnapply[func_, shape_Point, flag_:False] :=
	Point[If[flag, checkposncoords[#, func], #]&[func @@ shape[[1]]]]

posnapply[func_, Text[first_, coord_, opt___], flag_:False] :=
	Text[first,If[flag, checkposncoords[#, func], #]&[func @@ coord],opt]

posnapply[func_, Tooltip[arg_, rest___], flag_] :=
    Tooltip[posnapply[func, arg, flag], rest]

posnapply[x___] :=
	(Message[WorldGraphics::badposn,{x}];
	{})

(* routine to verify that projection created a valid coordinate or
   list of coordinates *)
(* case 1 : on off chance that something makes it through the transform
   as a packed array...*)
checkposncoords[l_?Developer`PackedArrayQ, _] := l

(* indeterminate and infinities are allowed, as there may be poles
   or branch cuts in the projection that cause problems in parts of
   the projection without breaking most points; anything else is
   assumed to break the entire projection. *)
isanumberpattern =
    _Real | _Integer | _Rational | Infinity | -Infinity | Indeterminate

(* following construction allows for deeply nested lists *)
checkposncoords[l:{___List}, f_] := (Scan[checkposncoords[#,f]&, l];l)
checkposncoords[l_?VectorQ, f_] := (checkonecoord[l, f];l)

(* anything else is a failure case -- throw an exception *)
checkonecoord[{isanumberpattern, isanumberpattern}, _] := True
checkonecoord[any_, func_] :=
    (Message[WorldGraphics::badproj, any, func];
     Throw[$Failed, "ProjectionException"])

(* Grid generation; also, edge 
	(for frame, background, and clipping) *)

Clear[generategrid]

generategrid[Automatic,i_,r_] :=
	generategrid[{30,30},i,r]

generategrid[x_?NumberQ,i_,r_] :=
	generategrid[{x,x},i,r]

generategrid[{x_?NumberQ,y_?NumberQ},i_,wr_] :=
	Module[{lats = Range[0,90,x],longs = Range[0,180,y]},
	generategrid[{Flatten[{lats,-lats}],
		Flatten[{longs,-longs}]},i,wr]]

generategrid[{x_List,y_List},inc_,wr_] :=
	Module[{r = wr,yp},
		r[[2,2]] = r[[2,2]] + 360;
		yp = Map[If[# < r[[2,1]],# + 360,#]&,y];
		Map[Line[adjust[#[[1]]]]&,generategrid[{x,yp},inc,r]]]/;
			First[wr[[2]]] >= Last[wr[[2]]]

generategrid[{x_List,y_List},inc_,wr_] := 
	Module[{lat,long,glat,glong,latr,longr},
		lat = 60 Union[Select[x,And[#>=wr[[1,1]],
			#<=wr[[1,2]]]&]];
		long = 60 Union[Select[y,And[#>=wr[[2,1]],
			#<=wr[[2,2]]]&]];
		latr = Append[Range[Sequence @@ (60 wr[[2]]),inc],
			60 wr[[2,2]]];
		longr = Append[Range[Sequence @@ (60 wr[[1]]),inc],
			60 wr[[1,2]]];
		glat = Outer[List,lat,latr];
		glong = Transpose[Outer[List,longr,long]];
		Map[Line,Join[glong,glat]]]

generategrid[x_,_,_] := (
	Message[WorldGraphics::badgrid,x];
	{})

Clear[generateedge]

generateedge[inc_,wr_] :=
	Module[{r = wr},
		r[[2,2]] = r[[2,2]] + 360;
		Map[adjust,generateedge[inc,r]]]/;
			First[wr[[2]]] >= Last[wr[[2]]]

generateedge[inc_,wr_] :=
	Module[{latr,longr,glat,glong},
	latr = Append[Range[Sequence @@ (60 wr[[2]]),inc],
			60 wr[[2,2]]];
	longr = Append[Range[Sequence @@ (60 wr[[1]]),inc],
			60 wr[[1,2]]];
	glat = Outer[List,60 wr[[1]],latr];
	glong = Transpose[Outer[List,longr,60 wr[[2]] ]];
	RotateRight[
		MapAt[Reverse,Flatten[Transpose[{glat,glong}],1]
			,{{1},{4}}]]]


Clear[adjust]

adjust[coords_] :=
	Map[If[#[[2]] > 180 60, # - {0,360 60},#]&,coords]

(* Tiltworld (to change central point on map) *)

Clear[tiltworld]

tiltworld[90,0,lamnought_,lat_,long_] :=
	Module[{ln = lamnought 60,longp},
		If[Abs[longp = long - ln] > 180 60,
			longp = longp - Sign[longp] 360 60];
		N[{lat,longp}]]

tiltworld[alpha_,beta_,lamnought_,lat_,long_] :=
	Module[{al = N[alpha Degree],bt = N[beta Degree],
		phi = N[lat Degree/60],lam = N[long Degree/60],latp,
		longp,a,b,ln = N[lamnought Degree]},
	latp = ArcSin[Sin[al] Sin[phi] -
		Cos[al] Cos[phi] Cos[lam - ln]];
	a = Cos[phi] Sin[lam - ln];
	b = Sin[al] Cos[phi] Cos[lam - ln] + Cos[al] Sin[phi];
	longp = Which[b==0. && a==0.,bt,
		b==0. && a!=0.,Sign[a] Pi/2 + bt,
		b<0. && a!=0.,ArcTan[a/b] + bt + Sign[a] Pi,
		b<0. && a==0.,ArcTan[a/b] + bt + Pi,
		True,ArcTan[a/b] + bt];
	If[N[Abs[longp]] > N[Pi],longp = longp - Sign[longp] 2 Pi];
	N[60/Degree {latp,longp}]]

inrangeQ[{latitude_,longitude_},{{latmin_,latmax_},
	{longmin_,longmax_}}] :=
	Module[{lm,lat = Chop[latitude],
				long = Chop[longitude]},
		If[longmax <= longmin,lm = longmax + 360,
			lm = longmax];
		(lat >= 60 latmin && lat <= 60 latmax &&
			If[long < 60 longmin,
				long + 60 360 <= 60 lm,
				long <= 60 lm])
	]

roundtheworldQ[{{_,long1_},{_,long2_}}] :=
	And[Sign[long1] =!= Sign[long2],
		Abs[long2 - long1] > 180 60 + 1]

clip[pnt_Point,wr_List,type_Symbol,_] :=
	If[inrangeQ[pnt[[1]],wr],pnt,{}]

clip[txt_Text,wr_List,type_Symbol,_] :=
	If[inrangeQ[txt[[2]],wr],txt,{}]

clip[shape_,wr_List,type_Symbol,edge_] :=
	Module[{list = shape[[1]],inrng,fpos,atwpos = {}},
		inrng = Map[inrangeQ[#,wr]&,list];
		fpos = Position[inrng,False];
		(* a particularly ugly hack to deal with a special
			case.  Warning: there are similar special cases
			that are not dealt with here... (I need a better
			routine... ) *)
		If[(Head[shape] == Polygon) &&
				(wr[[2]] == {-180,180}) &&
				(Or @@ (atwpos = Map[roundtheworldQ,
					Partition[Append[shape[[1]],
						First[shape[[1]]]],2,1]])),
			atwpos = Position[atwpos,True],
			atwpos = {}];
		If[fpos === {} && atwpos === {},Return[shape]];
		If[Length[fpos] === Length[list],Return[{}]];
		Switch[type,
			None,Return[shape],
			Simple,Return[{}],
			Full,Return[clipped[shape,wr,inrng,edge,atwpos]],
			_,Return[shape]  (* error condition *)
		]
	]/;Or[Head[shape] === Line,Head[shape] === Polygon]

Clear[clipint]

clipint[seg1_,seg2_] := Module[{int},
	If[(int = intersection[Map[If[Negative[#[[2]]],
							# + {0,360 60 + 1},#]&,seg1],
							seg2]) === Null,
		int = intersection[Map[If[Positive[#[[2]]],
							# - {0,360 60 + 1}, #]&,seg1],
							seg2]];
	If[int =!= Null && Abs[int[[2]]]>N[180 60],
		int[[2]] = int[[2]] - Sign[int[[2]]] 180 60];
	int]/;roundtheworldQ[seg1]

clipint[seg1_,seg2_] := Module[{int},
	int = intersection[seg1,seg2];
	If[int =!= Null && Abs[int[[2]]] > N[180 60],
		int[[2]] = int[[2]] - Sign[int[[2]]] 180 60];
	int]

clipped[shape_Line,range_,inrange_List,__] :=
	Module[{list = shape[[1]],pos,segs,lines,bords,wr = range,
				backflag = False,n,bsegs,prod,isegs},
		If[GreaterEqual @@ Last[wr],
				backflag = True;
				list = Map[tiltworld[90,0,wr[[2,1]]+180,Sequence @@ #]&,
							list];
				wr = {wr[[1]],{-180,180 - wr[[2,1]] + wr[[2,2]]}}];
		segs = Partition[list,2,1];
		pos = Position[Apply[Not[SameQ[##]]&,
			Partition[inrange,2,1],{1}],True];
		isegs = Chop[Map[segs[[Sequence @@ #]]&,pos]];
		wr = 60 wr; (* Degrees to Minutes *)
		prod = Outer[List,Sequence @@ wr];
		bsegs = Flatten[{prod,Transpose[prod]},1];
		bords = Table[clipint[isegs[[i]],bsegs[[j]]],
				{i,Length[isegs]},{j,Length[bsegs]}];
		If[Or @@ Map[#=={Null,Null,Null,Null}&,bords],
			Message[WorldGraphics::badint,bords];Abort[]];
		bords = Map[First[Select[#,# =!= Null &]]&,bords];
		If[TrueQ[First[inrange]],
			PrependTo[pos,{0}];
			PrependTo[bords,First[list]]];
		If[TrueQ[Last[inrange]],
			AppendTo[pos,{Length[inrange]}];
			AppendTo[bords,Last[list]]];
		If[OddQ[Length[pos]],
			Message[WorldGraphics::clipfail];Abort[]];
		lines = Map[Take[list,#]&,
				Transpose[Transpose[
					Partition[Flatten[pos],2]] + {1,0}]];
		Do[lines[[n]] = Join[{bords[[2 n - 1]]},lines[[n]],
			{bords[[2 n]]}],{n,Length[lines]}];
		If[backflag,
			lines = Map[tiltworld[90,0,-range[[2,1]]-180,Sequence @@ #]&,
						lines,{2}]];
		Map[Line,lines]]

clipped[shape_Polygon,range_,ir_List,edge_,atwpos_] :=
	Module[{list = Append[shape[[1]],First[shape[[1]]]],
			inrange = Append[ir,First[ir]],
			pos,segs,lines,bords,wr = range,cornerflag,n,
			backflag = False,isegs,prod,bsegs,atwbords,tbords,
			bordsegs,polys,irsegs,ratwpos = atwpos},
		If[GreaterEqual @@ Last[wr],
			backflag = True;
			list = Map[tiltworld[90,0,wr[[2,1]]+180,Sequence @@ #]&,
							list];
			wr = {wr[[1]],{-180,180 - wr[[2,1]] + wr[[2,2]]}}];
		segs = Partition[list,2,1];
		pos = Position[Apply[Not[SameQ[##]]&,
			irsegs = Partition[inrange,2,1],{1}],True];
		isegs = Map[segs[[Sequence @@ #]]&,pos];
		wr = 60 wr; (* Degrees to Minutes *)
		prod = Outer[List,Sequence @@ wr];
		bsegs = Flatten[{prod,Transpose[prod]},1];
		bords = Table[clipint[isegs[[i]],bsegs[[j]]],
				{i,Length[isegs]},{j,Length[bsegs]}];
		If[Or @@ Map[#=={Null,Null,Null,Null}&,bords],
			Message[WorldGraphics::badint,bords];Abort[]];
		bords = Map[First[Select[#,# =!= Null &]]&,bords];
		(* continuation of special case hack *)
		If[atwpos != {},
			atwbords = Apply[{{##},
				intersection[Map[If[Negative[#[[2]]],
						# + {0,360 60 + 1},#]&,segs[[##]]],
						{{90,180},{-90,180}} 60]}&,
				atwpos,{1}];
			atwbords = Select[atwbords,
				(inrangeQ[#[[2]],range] == True) &];
			ratwpos = Map[First, atwbords];
			atwbords = Map[If[Sign[list[[Sequence @@ #[[1]],2 ]] ] == 1,
					{#[[1]],{#[[2]],#[[2]] {1,-1}}},
					{#[[1]],{#[[2]] {1,-1},#[[2]]}}]&,atwbords];
			tbords = Transpose[{pos,bords}];
			tbords = Sort[Join[tbords,atwbords],
					If[First[#1] === First[#2] && 
							irsegs[[ First[First[#1]] ]] === {True, False},
						!OrderedQ[{#1,#2}],
						OrderedQ[{#1,#2}]
					]&
				];
			bords = Map[If[Head[#[[2,1]] ] =!= List,
				#[[2]], Sequence @@ #[[2]] ]&,tbords]];
		cornerflag = cornerinborderQ[segs,wr];
		bordsegs = genclipbords[wr,bords,cornerflag,edge];
		If[atwbords =!= {},
			pos = Sort[Join[pos,ratwpos,ratwpos]]];
		If[TrueQ[First[inrange]],
			pos = Join[{0},pos,{Length[inrange]}]];
		If[OddQ[Length[pos]],
			Message[WorldGraphics::clipfail];Abort[]];
		lines = Map[Take[list,#]&,
				Transpose[Transpose[
					Partition[Flatten[pos],2]] + {1,0}]];
		If[TrueQ[First[inrange]],
			Do[AppendTo[lines[[n]],bords[[2 n - 1]]];
				PrependTo[lines[[n + 1]],bords[[2 n]]],
				{n,Length[lines] - 1}],
			Do[lines[[n]] = Join[{bords[[2 n - 1]]},lines[[n]],
				{bords[[2 n]]}],{n,Length[lines]}]];
		polys = Check[genpolys[lines,bordsegs],
					Message[WorldGraphics::cgenpolys,lines,bordsegs];
					Abort[]];
		If[backflag,
			polys = Map[tiltworld[90,0,-range[[2,1]]-180,
							Sequence @@ #]&,
					polys,{2}]];
		Map[Polygon,polys]]

genclipbords[rng_,bords_,flag_,cedge_] :=
	Module[{sbords,segs,first,second,fseg = {},n},
		If[OddQ[Length[bords]],
			Message[WorldGraphics::gcbfail,bords,flag];
			Abort[]];
		sbords = Sort[bords,bordsort[#1,#2,rng]&];
		If[flag,
			first = First[sbords];second = Last[sbords];
			sbords = Take[sbords,{2,Length[sbords]-1}];
			fseg = Join[{second},
				Sort[Select[cedge,
					bordsort[second,#,rng] === True &],
						bordsort[#1,#2,wr]&],
				Select[cedge,
					bordsort[#,first,rng] === True &],
					{first}]];
		sbords = Partition[sbords,2];
		segs = Map[gcbselect[#,cedge,rng]&,sbords];
		segs = Table[Join[{sbords[[n,1]]},segs[[n]],
			{sbords[[n,2]]}],{n,Length[sbords]}];
		If[fseg =!= {},
			Join[segs,{fseg}],
			segs]
	]

gcbselect[{first_,second_},border_,range_] :=
	Select[border,bordsort[#,first,range] === False &&
		bordsort[#,second,range] === True &]

bordsort[{x1_,y1_},{x2_,y2_},{{mit_,mat_},{mig_,mag_}}] :=
	Which[y1 == mig && y2 == mig, OrderedQ[{x1,x2}],
		y1 == mig || y2 == mig, TrueQ[y1 == mig],
		x1 == mat && x2 == mat, OrderedQ[{y1,y2}],
		x1 == mat || x2 == mat, TrueQ[x1 == mat],
		y1 == mag && y2 == mag, Not[OrderedQ[{x1,x2}]],
		y1 == mag || y2 == mag, TrueQ[y1 == mag],
		x1 == mit && x2 == mit, Not[OrderedQ[{y1,y2}]],
		x1 == mit || x2 == mit, TrueQ[x1 == mit],
		True,False]

cornerinborderQ[segpoly_,range_] := Module[{flag = False,
		tstseg = {{range[[1,1]],range[[2,1]]},
			{-90 60,range[[2,1]]}},
		tmp,list,trues,indet,rest,even},
	If[OddQ[Length[Position[	(* determine if S. Polar *)
			Map[roundtheworldQ,segpoly],True]]],
		tmp = First[Transpose[Flatten[segpoly,1]]];
		If[Abs[-90 60 - Min[tmp]] <=
				Abs[90 60 - Max[tmp]],
			flag = True]];
	If[Equal @@ tstseg, Return[flag]];
	list = Map[polyclipint[#,tstseg]&,segpoly];
	(* check how many intersections of polygon segments with
		test segment. trues are full intersections,
		indet are when one end of a segment intersects;
		the Select[] in rest pulls the points. *)
	trues = Length[Select[list,TrueQ]];
	indet = Select[list,Head[#] === List &];
	rest = Length[Select[indet,!(# == {0,0})&]];
	If[OddQ[rest],
		Message[WorldGraphics::cbqfail,segpoly,indet];
		Abort[]];
	even = EvenQ[trues + rest/2];
	Not[Xor[even,flag]]]

Clear[polyclipint]

polyclipint[seg1_,seg2_] := Module[{tmp1 = seg1,tmp2 = seg2},
	If[tmp2[[1,2]] < 0,
		If[tmp1[[1,2]] > 0,
			tmp1[[1,2]] = tmp1[[1,2]] - 360 60,
			tmp1[[2,2]] = tmp1[[2,2]] - 360 60],
		If[tmp1[[1,2]] < 0,
			tmp1[[1,2]] = tmp1[[1,2]] + 360 60,
			tmp1[[2,2]] = tmp1[[2,2]] + 360 60]];
	polyclipint[tmp1,tmp2]]/;roundtheworldQ[seg1]

polyclipint[seg1_,seg2_] :=
	Module[{vec1 = vectoreqn[seg1],const1 = Det[seg1],ans1,
				vec2,const2,ans2,tmp},
		ans1 = Sign[Chop[seg2 . vec1 - const1,10^-5]];
		If[Equal @@ ans1,
			Return[]];
		vec2 = vectoreqn[seg2];const2 = Det[seg2];
		ans2 = Sign[Chop[seg1 . vec2 - const2,10^-5]];
		If[Equal @@ ans2,
			Return[]];
		If[First[ans1] == 0 || Last[ans1] == 0,
			Return[ans1]];
		Return[True]
	]

vectoreqn[{{x1_,y1_},{x2_,y2_}}] :=
	{y2 - y1,x1 - x2}

Clear[intersection]

intersection[seg1_,seg2_] :=
	Module[{vec1 = vectoreqn[seg1],const1 = Det[seg1],ans1,
				vec2,const2,ans2},
		ans1 = Sign[Chop[seg2 . vec1 - const1,10^-5]];
		If[Equal @@ ans1 (* || TrueQ[ans1[[2]] == 0] *),
			Return[]];
		vec2 = vectoreqn[seg2];const2 = Det[seg2];
		ans2 = Sign[Chop[seg1 . vec2 - const2,10^-5]];
		If[Equal @@ ans2 (* || TrueQ[ans2[[2]] == 0] *),
			Return[]];
		Reverse[Inner[Times,{const2,const1},{vec1,vec2},
			Subtract] {1,-1}]/Det[{vec1,vec2}]
	]

Clear[singularitycut]

Attributes[singularitycut] = {Listable}

singularitycut[{},_] := {}

singularitycut[shape_Point,_] := shape

singularitycut[shape_Text,_] := shape

singularitycut[shape_Line,_] :=
	Module[{list = shape[[1]],pos,segs,lines,bords,n},
		pos = Position[
				Map[roundtheworldQ[#]&,
					segs = Partition[list,2,1]],
				True];
		If[pos === {}, Return[shape]];
		bords = Apply[
					intersection[Map[If[Negative[#[[2]]],
							# + {0,360 60 + 1},#]&,segs[[##]]],
							{{90,180},{-90,180}} 60]&,
					pos,{1}];
		PrependTo[pos,{0}];AppendTo[pos,{Length[list]}];
		lines = Map[Take[list,#]&,
				Transpose[Transpose[
					Partition[Flatten[pos],2,1]] + {1,0}]];
		Do[AppendTo[lines[[n]],
			bords[[n]] {1,Sign[Last[Last[lines[[n]] ]]]}];
			PrependTo[lines[[n+1]],
			bords[[n]] {1,Sign[Last[First[lines[[n+1]] ]]]}],
				{n,Length[lines] - 1}
		];
		Map[Line,lines]]

singularitycut[shape_Polygon,inc_] :=
	Module[{list = Append[shape[[1]],First[shape[[1]]]],
	pos,segs,bords,bordpt = Null,n},
		pos = Position[Map[roundtheworldQ,
			segs = Partition[list,2,1]],True];
		If[pos === {},Return[shape]];
		bords = Sort[Apply[
					{intersection[Map[If[Negative[#[[2]]],
							# + {0,360 60 + 1},#]&,segs[[##]]],
							{{90,180},{-90,180}} 60],{#}}&,
					pos,{1}]];
		If[OddQ[Length[pos]],
			If[(Abs[-90 - First[bords][[1,1]]] <=
					Abs[90 - Last[bords][[1,1]]]),
				bordpt = First[bords];bords = Drop[bords,1],
				bordpt = Last[bords];bords = Drop[bords,-1]
			]];
		If[bordpt =!= Null,
			Module[{lats,longs,side,edge,border},
			 lats = Sign[bordpt[[1,1]]] Append[
			 	Range[Abs[bordpt[[1,1]]],90 60,inc],90 60];
			 longs = Append[Range[-180 60,180 60,inc],
			 	180 60];
			 side = Map[{#,180 60}&,lats];
			 edge = Map[{Sign[bordpt[[1,1]]] 90 60,#} &,
			 		longs];
			 border = Join[Map[{1,-1} # &,side],edge,
			 	Reverse[side]];
			 If[Sign[Last[First[
			 		segs[[bordpt[[2,1]] ]] ]] ] === -1,
			 	list = Insert[list,
			  		Hold[Sequence @@ border],
			 		bordpt[[2,1]] + 1],
			 	list = Insert[list,
			 		Hold[Sequence @@ Reverse[border]],
			 		bordpt[[2,1]] + 1]];
			 list = ReleaseHold[list];
			 bords = Map[If[#[[2,1]] > bordpt[[2,1]],
			 		{#[[1]],#[[2]] + Length[border]},
			 		#]&,bords]
			 ]];
		If[bords === {},
			Polygon[list],
			Module[{bordsegs,linesegs,polys},
			 bordsegs = generatebords[bords,inc];
			 bords = Sort[Transpose[Reverse[Transpose[bords]]]];
			 linesegs = Map[Take[list,#]&,Map[# + {1,0}&,
			 				Partition[Flatten[
								Join[{0},Transpose[bords][[1]],
							{Length[list]}]],2,1]]
					];
			 Do[AppendTo[linesegs[[n]],
				bords[[n,2]] {1,Sign[Last[Last[linesegs[[n]] ]]]}];
				PrependTo[linesegs[[n+1]],
				bords[[n,2]] {1,Sign[Last[First[linesegs[[n+1]] ]]]}],
					{n,Length[linesegs] - 1}];
			 If[Last[Last[linesegs]] == First[First[linesegs]],
				linesegs[[1]] = Join[Last[linesegs], First[linesegs]];
				linesegs = Drop[linesegs,-1]
			];
			 polys = Check[genpolys[linesegs,bordsegs],
			 			Message[WorldGraphics::sgenpolys,list];
			 			Abort[]];
			 Map[Polygon,polys]]
			]
		]


genpolys[lines_,edges_] := Module[{segs = lines,
	bords = edges,polys = {},poly,loc,part},
		While[segs =!= {},
			poly = segs[[1]];segs = Drop[segs,1];
			While[First[poly] =!= Last[poly],
				 part = Check[Last[Select[bords,
						(Last[#] == Last[poly] ||
						First[#] == Last[poly]) &]],
							Message[WorldGraphics::genpolys,
								lines,edges,poly];
							Abort[]];
				 loc = Position[bords,part];
				 bords = Drop[bords,Flatten[{loc,loc}]];
				 If[Last[part] == Last[poly],
					part = Reverse[part]];
				 poly = Join[poly,part];
				If[Last[poly] == First[poly], Break[]];
				 part = Check[Last[Select[segs,
						(First[#] == Last[poly] ||
                         Last[#] == Last[poly])&]],
							Message[WorldGraphics::genpolys,
								lines,edges,poly];
							Abort[]];
				 loc = Position[segs,part];
				 segs = Drop[segs,Flatten[{loc,loc}]];
				 If[Last[part] == Last[poly],
					part = Reverse[part]];
				 poly = Join[poly,part]];
			AppendTo[polys,poly]
		];
		polys]
				 

generatebords[bords_List,inc_] := Module[{segs,lats,lines,n},
	segs = Partition[Transpose[bords][[1]],2];
	lats = Map[Range[#[[1,1]],#[[2,1]],inc]&,segs];
	lats = Map[{#,180 60}&,lats,{2}];
	lines = Table[{segs[[n,1]],Sequence @@ lats[[n]],segs[[n,2]]},
		{n,Length[segs]}]; 
	Join[lines,Map[{1,-1} # &,lines,{2}]]]

Unprotect[Show];

Clear[Show]

(* In V6.0, graphics display is handled by the formatting function
   (MakeBoxes), so Show should only be used for option combination
   (else the conversion to graphics will occur twice...) *)

Show[WorldGraphics[graph_,wopts___],opts___] :=
	WorldGraphics[graph,
		Sequence@@FilterRules[Flatten[{opts}],Options[WorldGraphics]],
		Sequence@@FilterRules[Flatten[{opts}],Options[Graphics]],
		Sequence@@FilterRules[Flatten[{wopts}],Options[WorldGraphics]],
		Sequence@@FilterRules[Flatten[{wopts}],Options[Graphics]]]

Show[graph:{__WorldGraphics},opts___] := Module[{g,o},
	g = Map[First,graph];
	o = Flatten[{opts,
			Reverse[Map[(List @@ Drop[#,1])&,graph]]}];
	Show[WorldGraphics[g,Sequence @@ o]]]

Protect[Show];

RandomColors = RGBColor[Random[],Random[],Random[]]&;

RandomGrays = GrayLevel[Random[]]&;

ToMinutes[coords:{{{__},{__}}..}] := Map[ToMinutes,coords,{2}]

ToMinutes[coords:{{__},{__}}] := Map[ToMinutes,coords]

ToMinutes[deg_?NumberQ] := 60 deg

ToMinutes[{_?(#==0&), min_:0, sec_:0}] := min +
	If[min == 0, sec/60, (Sign[min] Abs[sec])/60]

ToMinutes[{deg_?NumberQ,min_:0,sec_:0}] := 60 deg + (Sign[deg] Abs[min]) +
	(Sign[deg] Abs[sec])/60

End[]

EndPackage[]

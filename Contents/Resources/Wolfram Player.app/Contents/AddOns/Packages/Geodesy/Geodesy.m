(* ::Package:: *)

(* :Title: Geodesy *)

(* :Author: John M. Novak *)

(* :Summary:
This package contains functions useful for or derived from geodesy, 
the science of measuring and mapping the surface of the Earth.
For instance, the package includes functions for finding
the distance between two points on the surface of the planet,
using different models.
*)

(* :Context: Geodesy` *)

(* :Package Version: 1.0 *)

(* :History:
	V.1.0, April 1991, by John M. Novak.
    V.1.1, February 1995, by John M. Novak -- fixes major bugs in
       SpheroidalDistance by replacing model with an approximation formula
       that works (but is less accurate than ideal case).
    September 2006, by Brian Van Vertloo -- changed contexts to fit new
       paclet structure for Mathematica 6.0.
*)

(* :Copyright: Copyright 1991-2007, Wolfram Research, Inc. *)

(* :Keywords:
	geodesy, geography, distance, Earth
*)

(* :Sources:
	Griffin, Frank, An Introduction to Spherical Trigonometry,
		(Houghton Mifflin Co., 1943).
	Pearson, Frederick II, Map Projections: Theory and
		Applications, (CRC Press, 1990).
*)

(* :Limitations:
The formula for spheroidal distance is an approximation formula;
as such, I have limited all computation to machine precision.
The formula is unfortunately unstable near poles and a variety of
other points; I have placed an approximation for most of these, but
some will still cause errors.
*)

(* :Mathematica Version: 2.0-6.0 *)

Message[General::obspkg, "Geodesy`"]

BeginPackage["Geodesy`"]

If[Not@ValueQ[SphericalDistance::usage],SphericalDistance::usage =
"SphericalDistance[pt1,pt2] gives the distance between two \
points on the earth using the spherical model of the planet. \
The points are expressed as {lat,long}, where lat, long can \
be in degrees, or {d,m} or {d,m,s} form."];

If[Not@ValueQ[SpheroidalDistance::usage],SpheroidalDistance::usage =
"SpheroidalDistance[pt1,pt2] gives the distance between two \
points on the Earth in km, using the spheroidal model of \
the planet. Coordinates are expressed as in SphericalDistance. \
Note that the model is an approximation formula that only employs \
machine precision computation. It is fairly accurate to distances \
of up to 10000 kilometers on the standard model of the Earth."];

If[Not@ValueQ[ToAuthalicRadius::usage],ToAuthalicRadius::usage =
"ToAuthalicRadius[a,e] gives the radius of the authalic \
sphere based on the spheroid with semimajor axis a and \
eccentricity e."];

If[Not@ValueQ[GeodeticToAuthalic::usage],GeodeticToAuthalic::usage =
"GeodeticToAuthalic[{lat,long},e] returns the latitude and \
longitude of a point on an authalic sphere based on a \
spheroid of eccentricity e, where lat and long are the \
geodetic coordinates of that point on the spheroid."];

If[Not@ValueQ[SemimajorAxis::usage],SemimajorAxis::usage =
"SemimajorAxis is an option for SpheroidalDistance, that \
sets the semimajor axis for the spheroid defining the \
planet, in km. Default is for the Earth, from the WGS-84 \
standard."];

If[Not@ValueQ[Eccentricity::usage],Eccentricity::usage =
"Eccentricity is an option for SpheroidalDistance, which \
sets the eccentricity of the ellipsoid defining the planet. \
The default is for the Earth, from the WGS-84 standard."];

If[Not@ValueQ[Radius::usage],Radius::usage =
"Radius is an option for SphericalDistance, which sets the \
radius of the sphere defining the planet, in km. The default \
is for the Earth, for the authalic sphere based on the \
WGS-84 standard spheroid."];

If[Not@ValueQ[ToDegrees::usage],ToDegrees::usage =
"ToDegrees[{d,m}] or ToDegrees[{d,m,s}] converts \
degree-minute-second forms of coordinates to degrees. \
The coordinates are \
adjusted to stay within [-180,180] degrees. Note that the \
sign of d is enforced on m and s; so {-34,3,2} means 34 deg, \
3 min, and 2 sec W (or S), as does {-34,-3,-2}."];

If[Not@ValueQ[ToDMS::usage],ToDMS::usage =
"ToDMS[deg] takes a coordinate in degrees and converts it \
to a degree-minute-second format, to the nearest second. \
The coordinates are within [-180,180] degrees."];

Begin["`Private`"]

(* Ground distance between points on the surface of the planet.  *)

Options[SpheroidalDistance] =
	{SemimajorAxis->6378.137,
	Eccentricity->.081819}

SpheroidalDistance::nonnum =
"SpheroidalDistance is an approximation formula that requires numeric \
values for the coordinates, the semimajor axis, and the eccentricity.";

(* SpheroidalDistance conversion from dms form *)
SpheroidalDistance[{p1_, l1_}, {p2_, l2_}, opts___] :=
   SpheroidalDistance[
       {ToDegrees[p1], ToDegrees[l1]},
       {ToDegrees[p2], ToDegrees[l2]},
   opts]/;Head[p1] === List || Head[p2] === List || Head[l1] ===List ||
           Head[l2] === List

(* the spheroidal distance formula is numerically unstable for small
 distances, and for a several other points. Use of high precision
 computation would help, but there wouldn't be much point due to the
 nature of the approximation formula. The following rules provide
 (worse than average) approximations for the trouble spots. *)
(* both points on one pole or the other, or both points close *)
SpheroidalDistance[{p1_, l1_}, {p2_, l2_}, opts___?OptionQ] :=
    0.0/;(Chop[N[90 - Abs[p1]], 10^-5] === 0 &&
          Chop[N[90 - Abs[p2]], 10^-5] === 0 &&
          Sign[p1] === Sign[p2]) ||
         (Chop[N[p1 - p2], 10^-6] === 0 &&
          Chop[N[(l1 - l2)], 10^-6] === 0) ||
         (Chop[N[180 - Abs[l1]], 10^-5] === 0 &&
          Chop[N[180 - Abs[l2]], 10^-5] === 0 &&
          Chop[N[p1 - p2], 10^-6] === 0)

(* both points on opposite poles *)
SpheroidalDistance[{p1_, l1_}, {p2_, l2_}, opts___?OptionQ] :=
   ((N[2 #1 EllipticE[#2^2]]) & @@
       ({SemimajorAxis, Eccentricity}/.
            Flatten[{opts}]/.Options[SpheroidalDistance]))/;
     (Chop[N[90 - Abs[p1]], 10^-5] === 0 &&
      Chop[N[90 - Abs[p2]], 10^-5] === 0 &&
      Sign[p1] === - Sign[p2])

(* both points on equator and 180 degrees apart *)
SpheroidalDistance[{p1_, l1_}, {p2_, l2_}, opts___?OptionQ] :=
    N[(SemimajorAxis/.Flatten[{opts}]/.Options[SpheroidalDistance]) Pi]/;
        (Chop[N[180 - Abs[l1 - l2]], 10^-4] === 0 &&
         Chop[N[p1],10^-3] === 0 && Chop[N[p2], 10^-3] === 0)

(* intermittent failures for points 180 degrees apart in longitude,
   with latitude of one == - latitude of other is also unstable; but
   I don't have a formula for it. It will be ignored at this time. *)

(* rest of points *)
SpheroidalDistance[{p1_,l1_},{p2_,l2_},opts___?OptionQ] :=
	Module[{a,e,phi,phi1,phi2,lam1,lam2},
		{a,e} = N[{SemimajorAxis,Eccentricity}/.Flatten[{opts}]/.
				Options[SpheroidalDistance]];
		{phi1,phi2,lam1,lam2} = N[Map[ToDegrees,{p1,p2,l1,l2}] Degree];
		spheroidaldistance[phi1, phi2, lam1, lam2, a, e]/;
		    And @@ Map[NumberQ, {phi1, phi2, lam1, lam2, a, e}]
	]

SpheroidalDistance[{_,_}, {_,_},___?OptionQ] :=
   $Dummy/;(Message[SpheroidalDistance::nonnum]; False)

spheroidaldistance[phi1_, phi2_, lam1_, lam2_, a_, e_] :=
    Module[{tf = Sqrt[1 - e^2], dlamm, thm, dthm},
		dlamm = adjustlongitude[lam2 - lam1]/2;
        {thm, dthm} = {(#1 + #2)/2, (#2 - #1)/2}& @@
                 {ArcTan[tf Tan[phi1]], ArcTan[tf Tan[phi2]]};
        spheroidaldistancefun[dlamm, thm, dthm, a, 1 - tf]
	]

adjustlongitude[lon_] :=
    If[ N[Abs[lon] > Pi],
        If[N[lon < 0],
           N[lon + 2 Pi],
           N[lon - 2 Pi]
        ],
        lon
   ]

(* this approximation formula is constructed and compiled at load-time *)
spheroidaldistancefun =
Module[{dlamm, thm, dthm, a, f, f4, f64,
           sindlamm, costhm, sinthm, cosdthm, sindthm,
           cL, cosd, d, cE, sind, cY, cT, cX, cD, cA, cB},
   Compile[{dlamm, thm, dthm, a, f},
       Evaluate[
          f4 = f/4; f64 = f^2/64;
          sindlamm = Sin[dlamm];
          costhm = Cos[thm]; sinthm = Sin[thm];
          cosdthm = Cos[dthm]; sindthm = Sin[dthm];
          cL = sindthm^2 + (cosdthm^2 - sinthm^2) sindlamm^2;
          d = ArcCos[cosd = 1 - cL - cL];
          cE = 2 cosd;
          sind = Sin[d];
          cY = sinthm cosdthm;
          cY = cY (2 cY)/(1 - cL);
          cT = sindthm costhm;
          cT = cT (2 cT)/cL;
          cX = cY + cT;
          cY = cY - cT;
          cT = d/sind;
          cD = 4 cT^2;
          cA = cD cE;
          cB = 2 cD;
          a sind (cT - f4 (cT cX - cY) +
            f64 (cX (cA + (cT - (cA - cE)/2) cX) -
            cY (cB + cE cY) + cD cX cY))
       ]
   ]
];

Options[SphericalDistance] =
	{Radius->6371007/1000}

SphericalDistance[{p1_,l1_},{p2_,l2_},opts___] :=
	Module[{r,phi1,phi2,lam1,lam2},
		{r} = {Radius}/.{opts}/.Options[SphericalDistance];
		{phi1,phi2,lam1,lam2} = Map[ToDegrees,{p1,p2,l1,l2}] Degree;
		r ArcCos[Sin[phi1] Sin[phi2] +
			Cos[phi1] Cos[phi2] Cos[Abs[lam1 - lam2]]]
	]

(* Conversions between models. *)

ToAuthalicRadius[a_?NumberQ,0] := a

ToAuthalicRadius[a_?NumberQ,1] := a Sqrt[1/2]

ToAuthalicRadius[a_?NumberQ,e_?NumberQ] :=
	a Sqrt[(2 e + (1 - e^2) (Log[1 + e] - Log[1 - e]))/(4e)]

GeodeticToAuthalic[{lt_,ln_},0] := {ToDegrees[lt], ToDegrees[ln]}

GeodeticToAuthalic[{lt_,ln_},1] := {0,ToDegrees[ln]}

GeodeticToAuthalic[{lt_,ln_},e_?NumberQ] :=
	Module[{lat = ToDegrees[lt] Degree,long = ToDegrees[ln] Degree//N,
			arg},
		arg = ((1 - e^2) (2 e Sin[lat] -
				(1 - e^2 Sin[lat]^2) (Log[1 - e Sin[lat]] -
				Log[1 + e Sin[lat]])))/
			(Sin[lat] (1 - e^2 Sin[lat]^2) (2 e -
				(1 - e^2) (Log[1 - e] - Log[1 + e])));
		{Re[ArcSin[Sin[lat] arg]],long}/Degree
	]

(* Auxilliary Functions *)

ToDegrees[deg_?NumberQ] :=
	If[Abs[deg] > 180,
		deg - 360 Ceiling[Quotient[deg,180]/2],
		deg]


(* correctly handle sign of angle *)
ToDegrees[l:{_?(#==0&), _?(#==0&), s_}] := todegrees[l,Sign[s]]

ToDegrees[l:{_?(#==0&), m_, s_:0}] := todegrees[l,Sign[m]]

ToDegrees[l:{d_, m_:0, s_:0}] := todegrees[l,Sign[d]]

todegrees[{d_, m_:0, s_:0}, sign_] :=
    ToDegrees[d + (sign Abs[m])/60 + (sign Abs[s])/3600]

ToDMS[deg_?NumberQ] :=
	Module[{tmp,d,m,s},
		tmp = ToDegrees[deg]; (* Make sure deg is in valid range *)
		d = Floor[Abs[tmp]];
		m = Floor[(Abs[tmp] - d) 60];
		s = Round[(Abs[tmp] - d - m/60) 3600];
		Sign[tmp]{d,m,s}]

End[]

EndPackage[]

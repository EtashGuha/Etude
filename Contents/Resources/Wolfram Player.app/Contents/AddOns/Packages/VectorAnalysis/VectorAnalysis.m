(* ::Package:: *)

(* :Name: Calculus`VectorAnalysis` *)

(* :Title: Three-Dimensional Vector Analysis *)

(* :Author: Stephen Wolfram, Bruce K. Sawhill, Jerry B. Keiper *) 

(* :Summary:
This package performs standard vector differential operations in 
orthogonal coordinate systems.  Coordinate transformations are 
also provided.
*)  

(* :Context: Calculus`VectorAnalysis` *)

(* :Package Version: 2.1 *)

(* :Copyright: Copyright 1990-2007,  Wolfram Research, Inc.
*)

(* :History:
	Version 1.0 by Stephen Wolfram, 1988.
	Revised by Bruce K. Sawhill, August 1990.
	Extensively revised by Jerry B. Keiper, November 1990.
	Version 2.1 by M. Trott and J. Keiper, October 1994:
		revised default coordinate symbols.
*)

(* :Keywords:
	coordinates, orthogonal coordinates, curvilinear coordinates,
	div, curl, grad, Laplacian, biharmonic, Jacobian, coordinate
	systems, coordinate transformations, Cartesian, cylindrical,
	spherical, parabolic cylindrical, paraboloidal, elliptic
	cylindrical, prolate spheroidal, oblate spheroidal, bipolar,
	bispherical, toroidal, conical, confocal ellipsoidal, confocal
	paraboloidal.
*)
	
(* :Source:
	Murray R. Spiegel, Schaum's Mathematical Handbook of 
	Formulas and Tables, McGraw-Hill, New York, 1968, pp. 116-130.

	Philip M. Morse and Herman Feshbach, Methods of Theoretical
	Physics, McGraw-Hill, New York, 1953.

	George B. Arfken, Mathematical Methods for Physicists, 3rd ed.,
	Academic Press, New York, 1985.
*)

(* :Mathematica Version: 2.0 *)

(* :Limitation:
	The branches associated with the signs of the square roots in the
	coordinate transformations between Cartesian coordinates and
	confocal ellipsoidal or confocal paraboloidal cannot be dealt
	with in the manner used in this package.  In particular the
	transformation from the confocal ellipsoidal system always gives
	a point in the octant with Xx, Yy, Zz > 0.  The transformation from
	the confocal paraboloidal coordinate system always gives a point
	in the quadrant with Xx, Yy > 0.
	
	Only right-handed coordinate systems are implemented.

	Parameters and coordinates cannot take on complex values in some
	of the coordinate systems.

	Only three-dimensional coordinate systems are implemented.
*)

(* Make sure system overrides have been loaded *)
{System`Div, System`Curl, System`Grad, System`JacobianMatrix, System`Laplacian};

Message[General::obspkg, "VectorAnalysis`"]

BeginPackage["VectorAnalysis`"]

(* get formatted VectorAnalysis messages, except for special cases *)
If[FileType[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"]]===File,
Select[FindList[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"],"VectorAnalysis`"],
StringMatchQ[#,StartOfString~~"VectorAnalysis`*"]&&
!StringMatchQ[#,"VectorAnalysis`"~~("Div"|"Curl"|"Grad"|"JacobianMatrix"|"Laplacian")~~__]&]//ToExpression;
]

Unprotect[Cartesian, Cylindrical, Spherical, ParabolicCylindrical,
    Paraboloidal, EllipticCylindrical, ProlateSpheroidal, OblateSpheroidal,
    Bipolar, Bispherical, Toroidal, Conical, ConfocalEllipsoidal,
    ConfocalParaboloidal];

Unprotect[DotProduct, CrossProduct, ScalarTripleProduct, SetCoordinates,
    Coordinates, Parameters, CoordinateRanges, CoordinatesToCartesian,
    CoordinatesFromCartesian, ScaleFactors, ArcLengthFactor,
    JacobianDeterminant, System`JacobianMatrix, System`Grad, System`Div, System`Curl, System`Laplacian,
    Biharmonic, CoordinateSystem];

Unprotect[Eeta, Llambda, Mmu, Nnu, Pphi, Rr, Ttheta, Uu, Vv, Xx, Xxi, Yy, Zz]

Block[{$NewMessage},
If[Not@ValueQ[DotProduct::usage],DotProduct::usage = "DotProduct[v1, v2] gives the dot product (sometimes \
called inner product or scalar product) of the two vectors v1, v2 in three \
space in the default coordinate system. DotProduct[v1, v2, coordsys] gives \
the dot product of v1 and v2 in the coordinate system coordsys."];

If[Not@ValueQ[CrossProduct::usage],CrossProduct::usage = "CrossProduct[v1, v2] gives the cross product \
(sometimes called vector product) of the two vectors v1, v2 in three space \
in the default coordinate system. CrossProduct[v1, v2, coordsys] gives \
the cross product of v1 and v2 in the coordinate system coordsys."];

If[Not@ValueQ[ScalarTripleProduct::usage],ScalarTripleProduct::usage = "ScalarTripleProduct[v1, v2, v3] gives the scalar \
triple product of the three vectors v1, v2, and v3 in three space in the \
default coordinate system. ScalarTripleProduct[v1, v2, v3, coordsys] gives \
the scalar triple product of v1, v2, and v3 in the coordinate system coordsys."];

If[Not@ValueQ[SetCoordinates::usage],SetCoordinates::usage = "SetCoordinates[coordsys] sets the default \
coordinate system to be coordsys with default variables. \
SetCoordinates[coordsys[c1, c2, c3]] sets the default coordinate system to be \
coordsys with variables c1, c2, and c3. Certain coordinate systems \
have parameters associated with them, and these parameters can be set by \
specifying the full description of the coordinate system. For example, \
SetCoordinates[Conical[lambda, mu, nu]] sets only the variables of the \
conical coordinate system, but SetCoordinates[Conical[lambda, mu, nu, a, b]] \
sets both the variables and the parameters of the conical coordinate \
system."];

If[Not@ValueQ[CoordinateSystem::usage],CoordinateSystem::usage = "CoordinateSystem gives the name of the default \
coordinate system."];

If[Not@ValueQ[Coordinates::usage],Coordinates::usage = "Coordinates[ ] gives a list of the default \
coordinate variables in the default coordinate system. \
Coordinates[coordsys] gives a list of the default coordinate variables \
in the coordinate system coordsys."];

If[Not@ValueQ[Parameters::usage],Parameters::usage = "Parameters[ ] gives a list of the default parameters \
of the default coordinate system. Parameters[coordsys] gives a list of the \
default parameters in the coordinate system coordsys. (Many of the \
coordinate systems do not have parameters.)"];

If[Not@ValueQ[CoordinateRanges::usage],CoordinateRanges::usage = "CoordinateRanges[ ] gives the intervals over which \
each of the coordinates in the default coordinate system may range. \
CoordinateRanges[coordsys] gives the ranges for each of the coordinates \
in the coordinate system coordsys."];

If[Not@ValueQ[ParameterRanges::usage],ParameterRanges::usage = "ParameterRanges[ ] gives the intervals over which \
each the of the parameters (if any) in the default coordinate system may range. \
ParameterRanges[coordsys] gives the ranges for each of the parameters in \
the coordinate system coordsys."];

If[Not@ValueQ[CoordinatesToCartesian::usage],CoordinatesToCartesian::usage =
"CoordinatesToCartesian[pt] gives the Cartesian coordinates \
of the point pt given in the default coordinate system. \
CoordinatesToCartesian[pt, coordsys] gives the Cartesian \
coordinates of the point given in the coordinate system coordsys."];

If[Not@ValueQ[CoordinatesFromCartesian::usage],CoordinatesFromCartesian::usage =
"CoordinatesFromCartesian[pt] gives the coordinates in the \
default coordinate system of the point pt given in Cartesian coordinates. \
CoordinatesFromCartesian[pt, coordsys] gives the coordinates \
in the coordinate system coordsys of the point given in Cartesian coordinates."];

If[Not@ValueQ[ScaleFactors::usage],ScaleFactors::usage = "ScaleFactors[pt] gives a list of the scale factors \
at the point pt in the default coordinate system. ScaleFactors[pt, coordsys] \
gives a list of the scale factors at the point pt in the coordinate system \
coordsys. If pt is not given, the default coordinate variables are \
used."];

If[Not@ValueQ[ArcLengthFactor::usage],ArcLengthFactor::usage = "ArcLengthFactor[pt, t] gives the derivative of \
the arc length of a curve given by pt with respect to the parameter t in \
the default coordinate system. ArcLengthFactor[pt, t, coordsys] gives the \
derivative of the arc length of a curve given by pt with respect to the \
parameter t in the coordinate system coordsys. If pt is not given, the \
default coordinate variables are used."];

If[Not@ValueQ[JacobianDeterminant::usage],JacobianDeterminant::usage = "JacobianDeterminant[pt] gives the determinate \
of the Jacobian matrix of the transformation from the default coordinate \
system to the Cartesian coordinate system at the point pt. \
JacobianDeterminant[pt, coordsys] gives the determinate of the Jacobian \
matrix of the transformation from the coordinate system coordsys to the \
Cartesian coordinate system at the point pt. If pt is not given, the \
default coordinate variables are used."];

If[Not@ValueQ[System`JacobianMatrix::usage],System`JacobianMatrix::usage = "JacobianMatrix[pt] gives the Jacobian matrix of \
the transformation from the default coordinate system to the Cartesian \
coordinate system at the point pt. JacobianMatrix[pt, coordsys] gives the \
Jacobian matrix of the transformation from the coordinate system coordsys \
to the Cartesian coordinate system at the point pt. If pt is not given, \
the default coordinate variables are used."];

If[Not@ValueQ[System`Grad::usage],System`Grad::usage = "Grad[f] gives the gradient of the scalar function f in the \
default coordinate system. Grad[f, coordsys] gives the gradient of f in the \
coordinate system coordsys."];

If[Not@ValueQ[System`Div::usage],System`Div::usage = "Div[f] gives the divergence of the vector-valued function f \
in the default coordinate system. Div[f, coordsys] gives the divergence of f \
in the coordinate system coordsys."];

If[Not@ValueQ[System`Curl::usage],System`Curl::usage = "Curl[f] gives the curl of the vector-valued function f in the \
default coordinate system. Curl[f, coordsys] gives the curl of f in the \
coordinate system coordsys."];

If[Not@ValueQ[System`Laplacian::usage],System`Laplacian::usage = "Laplacian[f] gives the Laplacian of the scalar- or \
vector-valued function f in the default coordinate system. \
Laplacian[f, coordsys] gives the Laplacian of f in the coordinate \
system coordsys."];

If[Not@ValueQ[Biharmonic::usage],Biharmonic::usage = "Biharmonic[f] gives the Laplacian of the Laplacian \
of the scalar function f in the default coordinate system. \
Biharmonic[f, coordsys] gives the biharmonic of f in the coordinate system \
coordsys."];

If[Not@ValueQ[Cartesian::usage],Cartesian::usage = "Cartesian represents the Cartesian coordinate system \
with default variables Xx, Yy, and Zz. Cartesian[x, y, z] represents the \
Cartesian coordinate system with variables x, y, and z."];

If[Not@ValueQ[Cylindrical::usage],Cylindrical::usage = "Cylindrical represents the cylindrical coordinate system \
with default variables Rr, Ttheta, and Zz. Cylindrical[r, theta, z] \
represents the cylindrical coordinate system with variables r, theta, and \
z."];

If[Not@ValueQ[Spherical::usage],Spherical::usage = "Spherical represents the spherical coordinate system \
with default variables Rr, Ttheta, and Pphi. Spherical[r, theta, phi] \
represents the spherical coordinate system with variables r, theta, and \
phi. The coordinate system is defined using the convention from \
physics, where r is the radius (distance from the origin), theta \
is the angle with respect to the vertical axis, and phi is the azimuth \
in the horizontal plane."];

If[Not@ValueQ[ParabolicCylindrical::usage],ParabolicCylindrical::usage = "ParabolicCylindrical represents the parabolic \
cylindrical coordinate system with default variables Uu, Vv, and Zz. \
ParabolicCylindrical[u, v, z] represents the parabolic cylindrical coordinate \
system with variables u, v, and z."];
	(* Parabolic cylindrical coordinates: Holding either of the first
	two variables constant while the third variable is held constant
	produces parabolas facing in opposite directions; the third
	variable specifies distance along the axis of common focus. *)
	
If[Not@ValueQ[Paraboloidal::usage],Paraboloidal::usage = "Paraboloidal represents the paraboloidal coordinate \
system with default variables Uu, Vv, and Pphi. Paraboloidal[u, v, phi] \
represents the paraboloidal coordinate system with variables u, v, and \
phi."];
	(* Paraboloidal coordinates: Holding either of the first two
	variables constant while the third variable is held constant
	produces opposite facing parabolas; the third parameter
	describes rotations about their common bisectors. *)

If[Not@ValueQ[EllipticCylindrical::usage],EllipticCylindrical::usage = "EllipticCylindrical represents the elliptic \
cylindrical coordinate system with default variables Uu, Vv, and Zz and \
default parameter value 1. EllipticCylindrical[u, v, z] represents the \
elliptic cylindrical coordinate system with variables u, v, and z and \
default parameter value 1. EllipticCylindrical[u, v, z, a] represents the \
elliptic cylindrical coordinate system with variables u, v, and z and \
parameter a."];
	(* Elliptic cylindrical coordinates: The parameter a is one-half
	the distance between the two foci.  Holding the first variable
	constant produces confocal ellipses, holding the second variable
	constant produces confocal hyperbolas, and the third variable
	specifies distance along the axis of common focus. *)
	
If[Not@ValueQ[ProlateSpheroidal::usage],ProlateSpheroidal::usage = "ProlateSpheroidal represents the prolate spheroidal \
coordinate system with default variables Xxi, Eeta, and Pphi and \
default parameter value 1. ProlateSpheroidal[xi, eta, phi] represents the \
prolate spheroidal coordinate system with variables xi, eta, and phi and \
default parameter value 1. ProlateSpheroidal[xi, eta, phi, a] represents the \
prolate spheroidal coordinate system with variables xi, eta, and phi and \
parameter a."];
	(* Prolate spheroidal coordinates: These coordinates
	are obtained by rotating elliptic cylindrical coordinates
	about the axis joining the two foci, which are separated by
	a distance 2*a. The third variable parameterizes the rotation. *)

If[Not@ValueQ[OblateSpheroidal::usage],OblateSpheroidal::usage = "OblateSpheroidal represents the oblate spheroidal \
coordinate system with default variables Xxi, Eeta, and Pphi and \
default parameter value 1. OblateSpheroidal[xi, eta, phi] represents the \
oblate spheroidal coordinate system with variables xi, eta, and phi and \
default parameter value 1. OblateSpheroidal[xi, eta, phi, a] represents the \
oblate spheroidal coordinate system with variables xi, eta, and phi and \
parameter a."];
	(* Oblate spheroidal coordinates: These coordinates are
    	obtained by rotating elliptic cylindrical coordinates about an
    	axis perpendicular to the axis joining the two foci, which are
    	separated by a distance 2*a.  The third variable parameterizes the
    	rotation. *)

If[Not@ValueQ[Bipolar::usage],Bipolar::usage = "Bipolar represents the bipolar coordinate system with \
default variables Uu, Vv, and Zz and default parameter value 1. \
Bipolar[u, v, z] represents the bipolar coordinate system with variables \
u, v, and z and default parameter value 1. Bipolar[u, v, z, a] represents the \
bipolar coordinate system with variables u, v, and z, and parameter a."];
	(* Bipolar coordinates: These coordinates are built around two
	foci, separated by a distance 2*a. Holding the first variable
	fixed produces circles that pass through both foci. Holding
	the second variable fixed produces degenerate ellipses about one
	of the foci. The third variable parameterizes distance along the
	axis of common foci. *)

If[Not@ValueQ[Bispherical::usage],Bispherical::usage = "Bispherical represents the bispherical coordinate system \
with default variables Uu, Vv, and Pphi and default parameter value 1. \
Bispherical[u, v, phi] represents the bispherical coordinate system with \
variables u, v, and phi and default parameter value 1. \
Bispherical[u, v, phi, a] represents the bispherical coordinate system with \
variables u, v, and phi and parameter a."];
	(* Bispherical coordinates: These coordinates are related
	to bipolar coordinates except that the third coordinate
	measures azimuthal angle rather than Zz. *)

If[Not@ValueQ[Toroidal::usage],Toroidal::usage = "Toroidal represents the toroidal coordinate system with \
default variables Uu, Vv, and Pphi and default parameter value 1. \
Toroidal[u, v, phi] represents the toroidal coordinate system with variables \
u, v, and phi and default parameter value 1. Toroidal[u, v, phi, a] \
represents the toroidal coordinate system with variables u, v, phi and \
parameter a."];
	(* Toroidal coordinates: These are obtained by rotating bipolar
	coordinates about an axis perpendicular to the line joining the
	two foci which are separated by a distance 2*a. The third variable
	parameterizes this rotation. *)
	
If[Not@ValueQ[Conical::usage],Conical::usage = "Conical represents the conical coordinate system with default \
variables Llambda, Mmu, and Nnu and default parameter values 1 and 2. \
Conical[lambda, mu, nu] represents the conical coordinate system with variables \
lambda, mu, and nu and default parameter values 1 and 2. \
Conical[lambda, mu, nu, a, b] represents the conical coordinate system with \
variables lambda, mu, and nu and parameters a and b."];
	(* Conical coordinates: The coordinate surfaces are spheres
	with centers at the origin and radius Llambda (Llambda constant),
	cones with apexes at the origin and axes along the z-axis
	(Mmu constant), and cones with apexes at the origin and axes
	along the y-axis (Nnu constant). *)
	
If[Not@ValueQ[ConfocalEllipsoidal::usage],ConfocalEllipsoidal::usage = "ConfocalEllipsoidal represents the confocal \
ellipsoidal coordinate system with default variables Llambda, Mmu, and Nnu \
and default parameter values 3, 2, and 1. ConfocalEllipsoidal[lambda, mu, nu] \
represents the confocal ellipsoidal coordinate system with variables \
lambda, mu, and nu and default parameter values 3, 2, and 1. \
ConfocalEllipsoidal[lambda, mu, nu, a, b, c] represents the confocal \
ellipsoidal coordinate system with variables lambda, mu, and nu and \
parameters a, b, and c."];
	(* Confocal ellipsoidal coordinates: The coordinate surfaces are
	ellipsoids (Llambda constant), hyperboloids of one sheet (Mmu
	constant), and hyperboloids of two sheets (Nnu constant). *)

If[Not@ValueQ[ConfocalParaboloidal::usage],ConfocalParaboloidal::usage = "ConfocalParaboloidal represents the confocal \
paraboloidal coordinate system with default variables Llambda, Mmu, and \
Nnu and default parameter values 2 and 1. ConfocalParaboloidal[lambda, mu, nu] \
represents the confocal paraboloidal coordinate system with variables \
lambda, mu, and nu and default parameter values 2 and 1. \
ConfocalParaboloidal[lambda, mu, nu, a, b] represents the confocal \
paraboloidal coordinate system with variables lambda, mu, and nu \
and parameters a and b."];
	(* Confocal paraboloidal coordinates: The coordinate surfaces are
	confocal families of elliptic paraboloids extending in the
	direction of the negative z-axis (Llambda constant), hyperbolic
	paraboloids (Mmu constant), and elliptic paraboloids extending in
	the direction of the positive z-axis (Nnu constant). *)


(* default coordinates *)

If[Not@ValueQ[Xx::usage],Xx::usage = Yy::usage =
"{Xx,Yy,Zz} are the default Cartesian coordinates."];

If[Not@ValueQ[Zz::usage],Zz::usage =
"{Xx,Yy,Zz} are the default Cartesian coordinates; {Rr,Ttheta,Zz} are the \
default Cylindrical coordinates; {Uu,Vv,Zz} are the default coordinates for the \
ParabolicCylindrical, EllipticCylindrical, and Bipolar coordinate systems."];

If[Not@ValueQ[Rr::usage],Rr::usage = Ttheta::usage =
"{Rr,Ttheta,Zz} are the default Cylindrical coordinates; {Rr,Ttheta,Pphi} are the \
default Spherical coordinates."];

If[Not@ValueQ[Pphi::usage],Pphi::usage =
"{Rr,Ttheta,Pphi} are the default Spherical coordinates; {Uu,Vv,Pphi} are the \
default coordinates for the Paraboloidal, Bispherical, and Toroidal coordinate \
systems; {Xxi,Eeta,Pphi} are the default coordinates for the Prolate Spheroidal \
and Oblate Spheroidal coordinate systems."];

If[Not@ValueQ[Uu::usage],Uu::usage = Vv::usage =
"{Uu,Vv,Zz} are the default coordinates for the ParabolicCylindrical, \
EllipticCylindrical, and Bipolar coordinate systems; {Uu,Vv,Pphi} are the \
default coordinates for the Paraboloidal, Bispherical, and Toroidal coordinate \
systems."];

If[Not@ValueQ[Xxi::usage],Xxi::usage = Eeta::usage =
"{Xxi,Eeta,Pphi} are the default coordinates for the ProlateSpheroidal and \
OblateSpheroidal coordinate systems."];

If[Not@ValueQ[Llambda::usage],Llambda::usage = Mmu::usage = Nnu::usage =
"{Llambda,Mmu,Nnu} are the default coordinates for the Conical, \
ConfocalEllipsoidal, and ConfocalParaboloidal coordinate systems."];
]

Begin["VectorAnalysis`Private`"]

Clear[$CoordinateSystem];  (* a person might read in this package twice! *)

$VecQ[v_] := VectorQ[v] && Length[v] == 3;

    (* these are the coordinate systems that are supported *)
$CoordSysList = {Cartesian, Cylindrical, Spherical, ParabolicCylindrical,
    Paraboloidal, EllipticCylindrical, ProlateSpheroidal, OblateSpheroidal,
    Bipolar, Bispherical, Toroidal, Conical, ConfocalEllipsoidal,
    ConfocalParaboloidal}

Coordinates::invalid = "`1` is not a valid coordinate system specification."

$BadCoordSys[coordsys_] := (Message[Coordinates::invalid, coordsys]; $Failed)

$ExpandCoordSys[coordsys_Symbol] :=
	If[MemberQ[$CoordSysList, coordsys],
		Apply[coordsys, Flatten[{Coordinates[coordsys],
				Parameters[coordsys]}]],
	    (* else *)
		$BadCoordSys[coordsys]];

$ExpandCoordSys[coordsys_[vp___]] :=
    Module[{vpars = {vp}, np, len},
	If[!MemberQ[$CoordSysList, coordsys],
		Return[$BadCoordSys[coordsys[vp]]]];
	len = Length[vpars];
	np = Length[Parameters[coordsys]];
	Which[
	    len == 0,
		Apply[coordsys,
		    Flatten[{Coordinates[coordsys], Parameters[coordsys]}]],
	    len == 3,
		If[Head /@ vpars =!= {Symbol, Symbol, Symbol},
		    $BadCoordSys[coordsys[vp]],
		    Apply[coordsys, Flatten[{vpars, Parameters[coordsys]}]]],
	    len == 3 + np,
		If[(Head /@ Take[vpars, 3]) =!= {Symbol, Symbol, Symbol},
		    $BadCoordSys[coordsys[vp]], Apply[coordsys, vpars]],
	    True,
		$BadCoordSys[coordsys[vp]]
	    ]
	]

$PointRule[cs_, pt_] :=
	{cs[[1]] -> pt[[1]], cs[[2]] -> pt[[2]], cs[[3]] -> pt[[3]]}

$DAbsSign = {Sign'[_] -> 0, Abs'[x_] -> Sign[x]};

  (* ===================== Dot Product ====================== *)

DotProduct[v1_?$VecQ, v2_?$VecQ, coordsys_:$CoordinateSystem] :=
    Module[{cs = $ExpandCoordSys[coordsys], cv1, cv2},
	cv1 . cv2 /; (cs =!= $Failed &&
			(cv1 = $CTToCart[v1, cs]) =!= $Failed &&
			(cv2 = $CTToCart[v2, cs]) =!= $Failed)];

  (* ===================== Cross Product ====================== *)

CrossProduct[v1_?$VecQ, v2_?$VecQ, coordsys_:$CoordinateSystem] :=
    Module[{cs = $ExpandCoordSys[coordsys], cv1, cv2, m},
	(m = Minors[{cv1, cv2}, 2][[1]];
	$CTFromCart[{m[[3]], -m[[2]], m[[1]]}, cs]) /; (cs =!= $Failed &&
				(cv1 = $CTToCart[v1, cs]) =!= $Failed &&
				(cv2 = $CTToCart[v2, cs]) =!= $Failed)];

  (* ================== Scalar Triple Product ====================== *)

ScalarTripleProduct[v1_?$VecQ, v2_?$VecQ, v3_?$VecQ,
					coordsys_:$CoordinateSystem] :=
    Module[{cs = $ExpandCoordSys[coordsys], cv1, cv2, cv3},
	Det[{cv1, cv2, cv3}] /; (cs =!= $Failed &&
				(cv1 = $CTToCart[v1, cs]) =!= $Failed &&
				(cv2 = $CTToCart[v2, cs]) =!= $Failed &&
				(cv3 = $CTToCart[v3, cs]) =!= $Failed)];

  (* ==================== Coordinates ========================== *)

Coordinates[coordsys_:$CoordinateSystem] :=
    Module[{cs = $ExpandCoordSys[coordsys]},
	(List @@ Take[cs, 3]) /; cs =!= $Failed]

  (* ==================== Parameters ========================== *)

Parameters[coordsys_:$CoordinateSystem] :=
    Module[{cs = $ExpandCoordSys[coordsys]},
	(List @@ Drop[cs, 3]) /; cs =!= $Failed]

  (* ==================== SetCoordinates ========================== *)

	(* ----------------- defaults -------------------- *)

Coordinates[Cartesian] ^= {Xx, Yy, Zz}
Coordinates[Cylindrical] ^= {Rr, Ttheta, Zz}
Coordinates[Spherical] ^= {Rr, Ttheta, Pphi} 
Coordinates[ParabolicCylindrical] ^= {Uu, Vv, Zz}
Coordinates[Paraboloidal] ^= {Uu, Vv, Pphi} 
Coordinates[EllipticCylindrical] ^= {Uu, Vv, Zz}
Coordinates[ProlateSpheroidal] ^= {Xxi, Eeta, Pphi} 
Coordinates[OblateSpheroidal] ^= {Xxi, Eeta, Pphi} 
Coordinates[Bipolar] ^= {Uu, Vv, Zz}
Coordinates[Bispherical] ^= {Uu, Vv, Pphi}
Coordinates[Toroidal] ^= {Uu, Vv, Pphi} 
Coordinates[Conical] ^= {Llambda, Mmu, Nnu} 
Coordinates[ConfocalEllipsoidal] ^= {Llambda, Mmu, Nnu} 
Coordinates[ConfocalParaboloidal] ^= {Llambda, Mmu, Nnu}

Parameters[Cartesian] ^= {}
Parameters[Cylindrical] ^= {}
Parameters[Spherical] ^= {} 
Parameters[ParabolicCylindrical] ^= {}
Parameters[Paraboloidal] ^= {} 
Parameters[EllipticCylindrical] ^= {1}
Parameters[ProlateSpheroidal] ^= {1}
Parameters[OblateSpheroidal] ^= {1} 
Parameters[Bipolar] ^= {1}
Parameters[Bispherical] ^= {1}
Parameters[Toroidal] ^= {1} 
Parameters[Conical] ^= {1, 2} 
Parameters[ConfocalEllipsoidal] ^= {3, 2, 1} 
Parameters[ConfocalParaboloidal] ^= {2, 1}

CoordinateSystem := $CoordinateSystem

SetCoordinates::range = "The parameter `1` is not positive."

SetCoordinates::ranges = "The parameters `1` do not satisfy the range \
requirements `2` for `3`."

SetCoordinates[coordsys_:$CoordinateSystem] :=
    Module[{cs = $ExpandCoordSys[coordsys], hcs, rr, par},
	cs /; ((cs =!= $Failed) &&
		(rr = ParameterRanges[cs];
		hcs = Head[cs];
		par = Apply[List, Drop[cs, 3]];
		If[rr =!= Null && (Evaluate[rr]& @@ par === False),
		    If[Length[par] == 1,
			Message[SetCoordinates::range, par[[1]]],
			Message[SetCoordinates::ranges, par, rr, hcs]
			];
		    False,
		  (* else *)
		    Unprotect[Evaluate[hcs]];
		    Coordinates[hcs] ^= Apply[List, Take[cs, 3]];
		    Parameters[hcs] ^= par;
		    Protect[Evaluate[hcs]];
		    $CoordinateSystem = hcs;
		    True
		    ]))
	];

  (* ================== Coordinate Ranges ====================== *)

CoordinateRanges[coordsys_:$CoordinateSystem] :=
    Module[{cs = $ExpandCoordSys[coordsys]},
	Switch[Head[cs],
	    Cartesian,
		{ -Infinity < cs[[1]] < Infinity,
		  -Infinity < cs[[2]] < Infinity,
		  -Infinity < cs[[3]] < Infinity },
	    Cylindrical,
		{ 0 <= cs[[1]] < Infinity,
		  -Pi < cs[[2]] <= Pi,
		  -Infinity < cs[[3]] < Infinity },
	    Spherical,
		{ 0 <= cs[[1]] < Infinity,
		  0 <= cs[[2]] <= Pi,
		  -Pi < cs[[3]] <= Pi },
	    ParabolicCylindrical,
		{ -Infinity < cs[[1]] < Infinity,
		  0 <= cs[[2]] < Infinity,
		  -Infinity < cs[[3]] < Infinity },
	    Paraboloidal,
		{ 0 <= cs[[1]] < Infinity,
		  0 <= cs[[2]] < Infinity,
		  -Pi <= cs[[3]] <= Pi },
	    EllipticCylindrical,
		{ 0 <= cs[[1]] < Infinity,
		  -Pi < cs[[2]] <= Pi,
		  -Infinity < cs[[3]] < Infinity },
	    ProlateSpheroidal,
		{ 0 <= cs[[1]] < Infinity,
		  0 <= cs[[2]] <= Pi,
		  -Pi < cs[[3]] <= Pi },
	    OblateSpheroidal,
		{ 0 <= cs[[1]] < Infinity,
 		  -Pi/2 <= cs[[2]] <= Pi/2,
		  -Pi < cs[[3]] <= Pi },
	    Bipolar,
		{ -Pi < cs[[1]] <= Pi,
		  -Infinity < cs[[2]] < Infinity,
		  -Infinity < cs[[3]] < Infinity },
	    Bispherical,
		{ 0 <= cs[[1]] <= Pi,
		  -Infinity < cs[[2]] < Infinity,
		  -Pi < cs[[3]] <= Pi },
	    Toroidal,
		{ -Pi < cs[[1]] <= Pi,
 		  0 <= cs[[2]] < Infinity,
		  -Pi < cs[[3]] <= Pi },
	    Conical,
		{ -Infinity < cs[[1]] < Infinity,
		  cs[[4]]^2 < cs[[2]]^2 < cs[[5]]^2,
		  cs[[3]]^2 < cs[[4]]^2 },
	    ConfocalEllipsoidal,
		{ -Infinity < cs[[1]] < cs[[6]]^2,
		  cs[[6]]^2 < cs[[2]] < cs[[5]]^2,
		  cs[[5]]^2 < cs[[3]] < cs[[4]]^2 },
	    ConfocalParaboloidal,
		{ -Infinity < cs[[1]] < cs[[5]]^2,
		  cs[[5]]^2 < cs[[2]] < cs[[4]]^2,
		  cs[[4]]^2 < cs[[3]] < Infinity }
	] /; (cs =!= $Failed)
    ]

  (* ================== Parameter Ranges ====================== *)

ParameterRanges[coordsys_:$CoordinateSystem] :=
    Module[{cs = $ExpandCoordSys[coordsys]},
	Switch[Head[cs],
	    Conical,
		0 < #1 < #2 < Infinity,
	    ConfocalEllipsoidal,
		0 < #3 < #2 < #1 < Infinity,
	    ConfocalParaboloidal,
		0 < #2 < #1 < Infinity,
	    _ (* all others *),
		If[Length[cs] == 4, 0 < #1 < Infinity, Null]
	] /; (cs =!= $Failed)
    ]

    (* ==================== ScaleFactors ===========================*)

ScaleFactors[arg_:$CoordinateSystem] :=
    Module[{pt, cs},
	ScaleFactors[pt, cs] /; (If[$VecQ[arg],
					cs = $CoordinateSystem;
					pt = arg,
					cs = $ExpandCoordSys[arg];
					If[cs =!= $Failed,
					    pt = List @@ Take[cs, 3]]];
				cs =!= $Failed)];

ScaleFactors[pt_?$VecQ, coordsys_:$CoordinateSystem] :=
    Module[{cs = $ExpandCoordSys[coordsys], tmp},
	(tmp = Switch[Head[cs],
	    Cartesian, {1, 1, 1},
	    Cylindrical, {1, cs[[1]], 1},
	    Spherical, {1, cs[[1]], cs[[1]] Sin[cs[[2]]]},
	    ParabolicCylindrical, 
		tmp = Sqrt[cs[[1]]^2 + cs[[2]]^2];
		{tmp, tmp, 1},
	    Paraboloidal,
		tmp = Sqrt[cs[[1]]^2 + cs[[2]]^2];
		{tmp, tmp, cs[[1]] cs[[2]]},
	    EllipticCylindrical,
		tmp = cs[[4]] Sqrt[Sinh[cs[[1]]]^2 + Sin[cs[[2]]]^2];
		{tmp, tmp, 1},
	    ProlateSpheroidal,
		tmp = cs[[4]] Sqrt[Sinh[cs[[1]]]^2+Sin[cs[[2]]]^2];
		{tmp, tmp, cs[[4]] Sinh[cs[[1]]] Sin[cs[[2]]]},
	    OblateSpheroidal,
		tmp = cs[[4]] Sqrt[Sinh[cs[[1]]]^2+Sin[cs[[2]]]^2];
		{tmp, tmp, cs[[4]] Cosh[cs[[1]]] Cos[cs[[2]]]},
	    Bipolar,
		tmp = cs[[4]]/(Cosh[cs[[2]]] - Cos[cs[[1]]]);
		{tmp, tmp, 1},
	    Bispherical,
		tmp = cs[[4]]/(Cosh[cs[[2]]] - Cos[cs[[1]]]);
		{tmp, tmp, Sin[cs[[1]]] tmp},
	    Toroidal,
		tmp = cs[[4]]/(Cosh[cs[[2]]] - Cos[cs[[1]]]);
		{tmp, tmp, tmp Sinh[cs[[2]]]},
	    Conical,
		tmp = Abs[cs[[1]]] Sqrt[cs[[2]]^2-cs[[3]]^2];
		{1, tmp/(Sqrt[cs[[2]]^2-cs[[4]]^2] Sqrt[cs[[5]]^2-cs[[2]]^2]),
		(* terms rearranged to coerce real-value out for valid coords, c.f. 63307 *)
		   -tmp/(Sqrt[(cs[[3]]^2-cs[[4]]^2)*(cs[[3]]^2-cs[[5]]^2)])},
	    ConfocalEllipsoidal,
		tmp = 2 Sqrt[(cs[[4]]^2 - #)(cs[[5]]^2 - #)(cs[[6]]^2 - #)]&;
		{Sqrt[(cs[[2]]-cs[[1]])(cs[[3]]-cs[[1]])]/tmp[cs[[1]]],
		 (* terms rearranged to coerce real-value out for valid coords, c.f. 63307 *)
		 Sqrt[(cs[[1]]-cs[[2]])/(cs[[6]]^2-cs[[2]])]*
		  Sqrt[(cs[[3]]-cs[[2]])/((cs[[4]]^2-cs[[2]])*(cs[[5]]^2-cs[[2]]))]/2,
		 Sqrt[(cs[[1]]-cs[[3]])(cs[[2]]-cs[[3]])]/tmp[cs[[3]]]},
	    ConfocalParaboloidal,
		tmp = 2 Sqrt[(cs[[4]]^2 - #)(cs[[5]]^2 - #)]&;
		{Sqrt[(cs[[2]]-cs[[1]])(cs[[3]]-cs[[1]])]/tmp[cs[[1]]],
		(* terms rearranged to coerce real-value out for valid coords, c.f. 63307 *)
		 Sqrt[(cs[[3]]-cs[[2]])/(cs[[4]]^2-cs[[2]])]*
		 Sqrt[(cs[[1]]-cs[[2]])/(cs[[5]]^2-cs[[2]])]/2,
		 Sqrt[(cs[[1]]-cs[[3]])(cs[[2]]-cs[[3]])]/tmp[cs[[3]]]}
	    ];
	tmp /. $PointRule[cs, pt]) /; (cs =!= $Failed)
    ]

  (* ================= CoordinatesToCartesian ===================*)

CoordinatesToCartesian[v_?$VecQ, coordsys_:$CoordinateSystem] :=
    Module[{cs = $ExpandCoordSys[coordsys], cv},
	cv /; (cs =!= $Failed && (cv = $CTToCart[v, cs]) =!= $Failed)
    ]

CoordinatesToCartesian::range = "The point `1` does not satisfy the \
range requirements `2` for `3`."

$CTToCart[{a_, b_, c_}, cs_] :=
    Module[{tmp, d},
	If[MemberQ[{Conical,ConfocalEllipsoidal,ConfocalParaboloidal},Head[cs]],
	    tmp = CoordinateRanges[cs];
	    tmp = And @@ (tmp /. {cs[[1]] -> a, cs[[2]] -> b, cs[[3]] -> c});
	    If[tmp === False,
		Message[CoordinatesToCartesian::range,
			{a, b, c}, CoordinateRanges[cs], cs];
		Return[$Failed]
		]
	    ];
	Switch[Head[cs],
	    Cartesian, {a, b, c},
	    Cylindrical, {a Cos[b], a Sin[b], c},
	    Spherical, {a Sin[b] Cos[c], a Sin[b] Sin[c], a Cos[b]},
	    ParabolicCylindrical, {(a^2 - b^2)/2, a b, c},
	    Paraboloidal, {a b Cos[c], a b Sin[c], (a^2 - b^2)/2},
	    EllipticCylindrical,
		{cs[[4]] Cosh[a] Cos[b], cs[[4]] Sinh[a] Sin[b], c},
	    ProlateSpheroidal,
		tmp = cs[[4]] Sinh[a] Sin[b];
		{tmp Cos[c], tmp Sin[c], cs[[4]] Cosh[a] Cos[b]},
	    OblateSpheroidal,
		tmp = cs[[4]] Cosh[a] Cos[b];
		{tmp Cos[c], tmp Sin[c], cs[[4]] Sinh[a] Sin[b]},
	    Bipolar,
		tmp = cs[[4]]/(Cosh[b]-Cos[a]);
		{tmp Sinh[b], tmp Sin[a], c},
	    Bispherical,
		tmp = cs[[4]]/(Cosh[b]-Cos[a]);
		d = Sin[a];
		{tmp d Cos[c], tmp d Sin[c], tmp Sinh[b]},
	    Toroidal,
		tmp = cs[[4]]/(Cosh[b]-Cos[a]);
		d = Sinh[b];
		{tmp d Cos[c], tmp d Sin[c], tmp Sin[a]},
	    Conical,
		tmp = 1/(cs[[4]]^2 - cs[[5]]^2);
		{ a Abs[b c] / (cs[[4]] cs[[5]]),
		  Abs[a] Sign[b] Sqrt[(b^2 - cs[[4]]^2)(c^2 - cs[[4]]^2) tmp]/
			cs[[4]],
		  Abs[a] Sign[c] Sqrt[(b^2 - cs[[5]]^2)(c^2 - cs[[5]]^2)(-tmp)]/
			cs[[5]] },
	    ConfocalEllipsoidal,
		tmp = (#^2 - a)(#^2 - b)(#^2 - c)&;
		d = 1/((cs[[#1]]^2 - cs[[#2]]^2)(cs[[#1]]^2 - cs[[#3]]^2))&;
		d = { Sqrt[tmp[cs[[4]]] d[4, 5, 6]],
		      Sqrt[tmp[cs[[5]]] d[5, 4, 6]],
		      Sqrt[tmp[cs[[6]]] d[6, 4, 5]] },
	    ConfocalParaboloidal,
		tmp = (#^2 - a)(#^2 - b)(#^2 - c)&;
		{ Sqrt[tmp[cs[[4]]]/(cs[[5]]^2 - cs[[4]]^2)],
		  Sqrt[tmp[cs[[5]]]/(cs[[4]]^2 - cs[[5]]^2)],
		  (cs[[4]]^2 + cs[[5]]^2 - a - b - c)/2}
	]
    ]

  (* ================= CoordinatesFromCartesian ===================*)

$ArcTan[x_, y_] := If[TrueQ[x == y == 0], 0, ArcTan[x, y]];

CoordinatesFromCartesian[v_?$VecQ, coordsys_:$CoordinateSystem] :=
    Module[{cs = $ExpandCoordSys[coordsys]},
	$CTFromCart[v, cs] /; (cs =!= $Failed)
    ]

$CTFromCart[{x_, y_, z_}, cs_] :=
    Module[{tmp, a, b, c, d, e},
	Switch[Head[cs],
	    Cartesian, {x, y, z},
	    Cylindrical, {Sqrt[x^2 + y^2], $ArcTan[x, y], z},
	    Spherical,
		a = Sqrt[x^2 + y^2 + z^2];
		{a, If[TrueQ[a == 0], 0, ArcCos[z/a]], $ArcTan[x, y]},
	    ParabolicCylindrical,
		d = Sqrt[x^2 + y^2];
		If[TrueQ[d == 0],
		    a = b = d,
		    If[TrueQ[x < 0],
			a = Sqrt[d + x] Sign[y]; b = y/a,
			b = Sqrt[d - x]; a = y/b]];
		{a, b, z},
	    Paraboloidal,
		d = Sqrt[x^2 + y^2 + z^2];
		If[TrueQ[d == 0],
		    {d, d, d},
		    e = Sqrt[x^2 + y^2];
		    If[TrueQ[z >= 0],
			a = Sqrt[d + z]; b = e/a,
			b = Sqrt[d - z]; a = e/b];
		    {a, b, $ArcTan[x, y]}],
	    EllipticCylindrical,
		e = ArcCosh[(x + I y)/cs[[4]]];
		{Re[e], Im[e], z},
	    ProlateSpheroidal,
		e = ArcCosh[(z + I Sqrt[x^2 + y^2])/cs[[4]]];
		{Re[e], Im[e], $ArcTan[x, y]},
	    OblateSpheroidal,
		e = ArcCosh[(Sqrt[x^2 + y^2] + I z)/cs[[4]]];
		{Re[e], Im[e], $ArcTan[x, y]},
	    Bipolar,
		e = 2 ArcCoth[(x + I y)/cs[[4]]];
		{-Im[e], Re[e], z},
	    Bispherical,
		e = 2 ArcCoth[(z + I Sqrt[x^2 + y^2])/cs[[4]]];
		{-Im[e], Re[e], $ArcTan[x, y]},
	    Toroidal,
		e = 2 ArcCoth[(Sqrt[x^2 + y^2] + I z)/cs[[4]]];
		{-Im[e], Re[e], $ArcTan[x, y]},
	    Conical,
		a = Sqrt[x^2 + y^2 + z^2];
		If[TrueQ[a == 0],
		    {a, 0, 0},
		  (* else *)
		    c = b - cs[[4]]^2;
		    d = b - cs[[5]]^2;
		    e = b /. Solve[x^2 c d + y^2 b d + z^2 b c == 0, b];
		    Prepend[Sqrt[Reverse[Sort[e]]],a] {Sign[x],Sign[y],Sign[z]}
		],
	    ConfocalEllipsoidal,
		a = d - cs[[4]]^2;
		b = d - cs[[5]]^2;
		c = d - cs[[6]]^2;
		Sort[d /. Solve[x^2 b c + y^2 a c + z^2 a b + a b c == 0, d]],
	    ConfocalParaboloidal,
		a = d - cs[[4]]^2;
		b = d - cs[[5]]^2;
		Sort[d /. Solve[x^2 b + y^2 a - 2 z a b - d a b == 0, d]]
	]
    ]

(* === Here are all of the rules for differential operations on vectors === *)

ArcLengthFactor[t_Symbol, coordsys_:$CoordinateSystem] :=
    Module[{pt, cs = $ExpandCoordSys[coordsys]},
	ArcLengthFactor[pt, t, cs] /;
	    (If[cs =!= $Failed, pt = List @@ Take[cs, 3]]; cs =!= $Failed)]

ArcLengthFactor[pt_?$VecQ, t_Symbol, coordsys_:$CoordinateSystem] :=
    Module[{cs = $ExpandCoordSys[coordsys]},
	Sqrt[(ScaleFactors[pt, cs]^2) . (Dt[pt, t]^2)] /; (cs =!= $Failed)]

JacobianDeterminant[arg_:$CoordinateSystem] :=
    Module[{pt, cs},
	JacobianDeterminant[pt, cs] /; (If[$VecQ[arg],
					cs = $CoordinateSystem;
					pt = arg,
					cs = $ExpandCoordSys[arg];
					If[cs =!= $Failed,
					    pt = List @@ Take[cs, 3]]];
					cs =!= $Failed)];

JacobianDeterminant[pt_?$VecQ, coordsys_:$CoordinateSystem] :=
    Module[{cs = $ExpandCoordSys[coordsys]},
	(Times @@ ScaleFactors[pt, cs]) /; (cs =!= $Failed)]

System`JacobianMatrix[arg_:$CoordinateSystem] :=
    Module[{pt, cs},
	System`JacobianMatrix[pt, cs] /; (If[$VecQ[arg],
					cs = $CoordinateSystem;
					pt = arg,
					cs = $ExpandCoordSys[arg];
					If[cs =!= $Failed,
					    pt = List @@ Take[cs, 3]]];
					cs =!= $Failed)];

System`JacobianMatrix[pt_?$VecQ, coordsys_:$CoordinateSystem] :=
    Module[{cs = $ExpandCoordSys[coordsys], uvw},
	(uvw = List @@ Take[cs, 3];
	Outer[D, $CTToCart[uvw, cs], uvw] /. $DAbsSign /.
		$PointRule[cs, pt]) /; (cs =!= $Failed)]/;
   MemberQ[$CoordSysList, coordsys] || MemberQ[$CoordSysList, Head[coordsys]]

(* Due to rule-ordering issues with the Tensor system, we must explicitly
   force rules for the following functions to be placed in front of the
   downvalue list; this is an ugly hack, but the alernative seems to be
   to shadow the System` functionality. (At present, JacobianMatrix
   seems to just be a FindRoot option, so extreme measures aren't
   needed for that above.) The new test on coordsys is an abbreviated
   form of the test used in ExpandCoordSys, and is thus redundant, except
   that it allows the tensor syntax to pass through to their controlling
   rules without erroring in the VectorAnalysis code. --JMN 26.08.12 *)

PrependTo[DownValues[System`Grad],
    HoldPattern[System`Grad[f:Except[_List], coordsys_:$CoordinateSystem]] :> 
  Module[{cs = $ExpandCoordSys[coordsys]}, 
    Outer[D, {f}, List @@ Take[cs, 3]][[1]]/ScaleFactors[cs] /; 
     cs =!= $Failed] /; 
   MemberQ[$CoordSysList, coordsys] || MemberQ[$CoordSysList, Head[coordsys]]
]

PrependTo[DownValues[System`Div],
    HoldPattern[System`Div[f_?$VecQ, coordsys_:$CoordinateSystem]] :>
    Module[{cs = $ExpandCoordSys[coordsys], sf, psf},
	(sf = ScaleFactors[cs];
	psf = Times @@ sf;
	(Inner[D, f psf/sf, List @@ Take[cs, 3]] /. $DAbsSign)/psf) /;
		(cs =!= $Failed)]/;
   MemberQ[$CoordSysList, coordsys] || MemberQ[$CoordSysList, Head[coordsys]]
]

PrependTo[DownValues[System`Curl],
  HoldPattern[System`Curl[f_?$VecQ, coordsys_:$CoordinateSystem]] :>
    Module[{cs = $ExpandCoordSys[coordsys], sf},
	(sf = ScaleFactors[cs];
	{(D[sf[[3]] f[[3]], cs[[2]]] - D[sf[[2]] f[[2]], cs[[3]]])/
		(sf[[2]] sf[[3]]),
	 (D[sf[[1]] f[[1]], cs[[3]]] - D[sf[[3]] f[[3]], cs[[1]]])/
		(sf[[1]] sf[[3]]),
	 (D[sf[[2]] f[[2]], cs[[1]]] - D[sf[[1]] f[[1]], cs[[2]]])/
		(sf[[1]] sf[[2]])} /. $DAbsSign) /; (cs =!= $Failed)]/;
   MemberQ[$CoordSysList, coordsys] || MemberQ[$CoordSysList, Head[coordsys]]
]

DownValues[System`Laplacian] =
{HoldPattern[System`Laplacian[f_?$VecQ, coordsys_:$CoordinateSystem]] :>
    Module[{cs = $ExpandCoordSys[coordsys]},
	System`Grad[System`Div[f, cs], cs] - System`Curl[System`Curl[f, cs], cs] /; (cs =!= $Failed)]/;
   MemberQ[$CoordSysList, coordsys] || MemberQ[$CoordSysList, Head[coordsys]],

 HoldPattern[System`Laplacian[f:Except[_List], coordsys_:$CoordinateSystem]] :>
    Module[{cs = $ExpandCoordSys[coordsys]},
	System`Div[System`Grad[f, cs], cs] /; (cs =!= $Failed)]/;
   MemberQ[$CoordSysList, coordsys] || MemberQ[$CoordSysList, Head[coordsys]]
} ~Join~ DownValues[System`Laplacian];

Biharmonic[f:Except[_List], coordsys_:$CoordinateSystem] :=
    Module[{cs = $ExpandCoordSys[coordsys]},
	System`Laplacian[System`Laplacian[f, cs], cs] /; (cs =!= $Failed)]

SetCoordinates[Cartesian[ ]];  (* default coordinate system *)

End[ ];		(* "Calculus`VectorAnalysis`Private`" *)

Attributes[DotProduct] = {ReadProtected};
Attributes[CrossProduct] = {ReadProtected};
Attributes[ScalarTripleProduct] = {ReadProtected};
Attributes[SetCoordinates] = {ReadProtected};
Attributes[CoordinateSystem] = {ReadProtected};
Attributes[Coordinates] = {ReadProtected};
Attributes[Parameters] = {ReadProtected};
Attributes[CoordinateRanges] = {ReadProtected};
Attributes[CoordinatesToCartesian] = {ReadProtected};
Attributes[CoordinatesFromCartesian] = {ReadProtected};
Attributes[ScaleFactors] = {ReadProtected};
Attributes[ArcLengthFactor] = {ReadProtected};
Attributes[JacobianDeterminant] = {ReadProtected};
Attributes[System`JacobianMatrix] = {ReadProtected};
Attributes[System`Grad] = {ReadProtected};
Attributes[System`Div] = {ReadProtected};
Attributes[System`Curl] = {ReadProtected};
Attributes[System`Laplacian] = {ReadProtected};
Attributes[Biharmonic] = {ReadProtected};

Protect[Cartesian, Cylindrical, Spherical, ParabolicCylindrical,
    Paraboloidal, EllipticCylindrical, ProlateSpheroidal, OblateSpheroidal,
    Bipolar, Bispherical, Toroidal, Conical, ConfocalEllipsoidal,
    ConfocalParaboloidal];

Protect[DotProduct, CrossProduct, ScalarTripleProduct, SetCoordinates,
    Coordinates, Parameters, CoordinateRanges, CoordinatesToCartesian,
    CoordinatesFromCartesian, ScaleFactors, ArcLengthFactor,
    JacobianDeterminant, System`JacobianMatrix, System`Grad, System`Div, System`Curl, System`Laplacian,
    Biharmonic, CoordinateSystem];

Protect[Eeta, Llambda, Mmu, Nnu, Pphi, Rr, Ttheta, Uu, Vv, Xx, Xxi, Yy, Zz]

EndPackage[ ]

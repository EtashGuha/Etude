(*

   IF YOU WANT TO CUSTOMIZE YOUR SECURITY SYSTEM DO *NOT* MODIFY 
   THIS FILE. THE WAY TO CUSTOMIZE SECURITY IS DESCRIBED IN THE 
   SECURITY SECTION OF THE USER GUIDE.

*)


(* :Name: Security` *)

(* :Title: Mathematica Server Pages Security *)

(* :Author: Tom Wickham-Jones *)

(* :Copyright: 
       webMathematica source code (c) 1999-2003,
       Wolfram Research, Inc. All rights reserved.
*)

(* :Mathematica Version: 4.2 *)

(* :Package Version: 2.0 *)

(* :History:
   Original Version by Tom Wickham-Jones as part of MSP tools.
   Developed January - August 2000.
*)

(*:Summary:
   This package provides security features to be used by 
   Mathematica Web Tools.
   
   
   InsecureExprQ[ expr] returns True if expr contains 'insecure symbols' 
   and False otherwise.
   
   Symbols are tested to see if they are allowed or not. The 
   following tests are performed that look at the names and contexts 
   of the symbols.  They try to collect any symbols that are disallowed.
   
   
   1) If AllowedContexts is set to a list of contexts then all symbols 
      with contexts on this list are allowed, the remainder are added to 
      the disallowed list.
      
   2) If AllowedContexts is not set to a list of contexts then all symbols with 
      contexts in DisallowedContexts are added to the disallowed list.
   
   The remaining disallowed symbols are then modified as follows:
   
   3) If AllowedSymbols is set to a list of symbols then all symbols 
      that appear in the list are removed from the disallowed list.
      
   4) If DisallowedSymbols is set to a list of symbols then all symbols 
      that appeared in the original expression are added to the disallowed 
      list.
      
   
   If any symbols are found in the disallowed list then the expr is deemed to 
   be insecure.
      
   
   Using DisallowedSymbols and DisallowedContexts gives most flexibility 
   but with a higher risk.
   
   Using AllowedSymbols and AllowedContexts gives less flexibity but with 
   lower risk.
   
   See examples at bottom.
   
*)

(* :Context: Security` *)



BeginPackage[ "Security`"]

(* :Exports: *)

InsecureExprQ

SetSecurity

$AllowedContexts;
$DisallowedContexts;
$AllowedSymbols;
$DisallowedSymbols;

ToExpressionSecure

MakeSecurityFunction

FindInsecureSymbols

SecurityData

LoadSecurityData::usage = "LoadSecurityData[ dir,file] returns the security data for the given file. It returns $Failed on any error."

`Information`$Version = "Security Version 1.0.0";
`Information`$VersionNumber = 1.0;
`Information`$ReleaseNumber = 0;
`Information`$CreationID = If[SyntaxQ["114"], ToExpression["114"], 0]
`Information`$CreationDate = If[SyntaxQ["{2019, 05, 19, 20, 52, 53}"], ToExpression["{2019, 05, 19, 20, 52, 53}"], {0,0,0,0,0,0}]

Begin[ "`Private`"]

ToExpressionSecure::security = "Input expression `1` is not secure."


ToExpressionSecure[ expr_, fmt_:InputForm, head_:Null] :=
	ToExpressionSecure[ expr, {$AllowedContexts, $AllowedSymbols, $DisallowedContexts, $DisallowedSymbols}, fmt, head]


ToExpressionSecure[ expr_, data_, fmt_:InputForm, head_:Null] :=
	Module[ {ef},
		ef = ToExpression[ expr, fmt, HoldComplete];
		If[ InsecureExprQ[ ef, data],  
			Message[ ToExpressionSecure::security, expr];$Failed,
			If[ head === Null, ReleaseHold[ ef], head @@ ef]
			]
	]



SetAttributes[ SymbolInsecureByContextQ, HoldFirst]

(*
    If AllowedContexts is a list return True if context of symbol is 
    not in list.   Else return True if context of symbol is in 
    DisallowedContexts.
*)

SymbolInsecureByContextQ[ x_Symbol, allowedContexts_, disallowedContexts_] := 
    If[ ListQ[ allowedContexts],
            !MemberQ[ allowedContexts, Context[x]],
            MemberQ[ disallowedContexts, Context[ x]]]

SymbolInsecureByContextQ[ ___] := False

FindInsecureSymbols[e_] :=
	FindInsecureSymbols[ e, {$AllowedContexts, $AllowedSymbols, $DisallowedContexts, $DisallowedSymbols}]

FindInsecureSymbols[ e_HoldComplete, 
	{allowedContexts_, allowedSymbols_, disallowedContexts_, disallowedSymbols_}] :=
    Module[ {atoms, disallowed},
        atoms = Level[ e, {-1}, HoldComplete, Heads->True] ;
       	disallowed = Select[ atoms, Function[Null, SymbolInsecureByContextQ[#, allowedContexts, disallowedContexts], {HoldAllComplete}]];
        If[ 
        	Head[ allowedSymbols] === HoldComplete,
        		disallowed = Complement[ disallowed, allowedSymbols]];
        If[
        	Head[ disallowedSymbols] === HoldComplete,
        		disallowed = Union[ disallowed, Intersection[ atoms, disallowedSymbols]]];
		disallowed
    ]

InsecureExprQ[ e_HoldComplete, data_] :=
    Module[ {work = FindInsecureSymbols[e, data]},
        work =!= HoldComplete[] && work =!= HoldComplete[ HoldComplete]
    ]

InsecureExprQ[ a_, data_] := True

InsecureExprQ[ a_] :=
	InsecureExprQ[ a, {$AllowedContexts, $AllowedSymbols, $DisallowedContexts, $DisallowedSymbols}]


LoadSecurityConfiguration[ file_] :=
    Module[ {},
        If[ FileType[ file] === File,
            Get[ file]; 
            True,
            False]
        ]
        
LoadSecurityConfiguration[ dir_, file_] :=
	LoadSecurityConfiguration[ ToFileName[ dir, file]]

SecurityOpenQ = True;

SetSecurity[ ] :=
    SetSecurity[ None]

SetSecurity[ dir_, file_] :=
	SetSecurity[ ToFileName[ dir, file]];
	
SetSecurity[ file_] :=
    Module[ {found = True},
        If[ SecurityOpenQ,
            If[ StringQ[ file],
                found = LoadSecurityConfiguration[ file]] ;
            Which[ 
                MatchQ[ $AllowedContexts, {___String}],
                    1,
                MatchQ[ $DisallowedContexts, {___String}],
                    Clear[ $AllowedContexts],
                True,
                    $AllowedContexts = defaultAllowedContexts];

            Which[ 
                MatchQ[ $AllowedSymbols, HoldComplete[___Symbol]],
                    1,
                MatchQ[ $DisallowedSymbols, HoldComplete[___Symbol]],
                    Clear[ $AllowedSymbols],
                True,
                    $AllowedSymbols = defaultAllowedSymbols];

            LockProtectSymbol[ $AllowedContexts];
            LockProtectSymbol[ $DisallowedContexts];
            LockProtectSymbol[ $AllowedSymbols];
            LockProtectSymbol[ $DisallowedSymbols];
            LockProtectSymbol[ InsecureExprQ];
            SecurityOpenQ = False;
            ];
        found
    ]


SetAttributes[ LockProtectSymbol, HoldAllComplete]

LockProtectSymbol[sym_] :=
    (
    Protect[ sym] ;
    SetAttributes[ sym, Locked];
    )


    
MakeSecurityFunction[ {allowedContexts_, allowedSymbols_, disallowedContexts_, disallowedSymbols_}] :=
    Module[ {newFun},
        SetAttributes[ newFun, HoldAllComplete];
        newFun[SecurityData] := 
        	{allowedContexts, allowedSymbols, disallowedContexts, disallowedSymbols};
        newFun[ InsecureExprQ[ e_]] := 
        	InsecureExprQ[ e, {allowedContexts, allowedSymbols, disallowedContexts, disallowedSymbols}];
        newFun[ ToExpressionSecure[e_, fmt_:InputForm, head_:Null]] := 
        	ToExpressionSecure[e, {allowedContexts, allowedSymbols, disallowedContexts, disallowedSymbols}, fmt, head];
        newFun[ FindInsecureSymbols[ e_]] :=
        	FindInsecureSymbols[e, {allowedContexts, allowedSymbols, disallowedContexts, disallowedSymbols}];
        LockProtectSymbol[ newFun];
        newFun
    ]



(*
 Load the SecurityData from dir/file, if a problem return $Failed, 
 otherwise return { allowedContexts, allowedSymbols, disallowedContexts, disallowedSymbols}
*)
LoadSecurityData[ file_] :=
	Module[{obj, allCont, allSym, disCont, disSym},
        If[ FileType[ file] =!= File, Return[ $Failed]];
        obj = Get[ file];
        If[ !MatchQ[ obj, { (_String -> _) ..}], 
        						Return[ $Failed]];
        allCont = getValue[ "AllowedContexts", obj];
        allSym = getValue[ "AllowedSymbols", obj];
       	disCont = getValue[ "DisallowedContexts", obj];
       	disSym = getValue[ "DisallowedSymbols", obj];
		If[ !checkContext[allCont], Return[$Failed]];
		If[ !checkContext[disCont], Return[$Failed]];
		If[ !checkSymbols[allSym], Return[$Failed]];
		If[ !checkSymbols[disSym], Return[$Failed]];
		If[ allCont === Null && disCont === Null, Return[$Failed]];
		{allCont, allSym, disCont, disSym}
	]

LoadSecurityData[ dir_, file_] :=
	LoadSecurityData[ ToFileName[ dir, file]]


checkContext[ Null] := True

checkContext[ c_] :=
	MatchQ[ c, {_String ..}]

checkSymbols[ Null] := True

checkSymbols[ s_] :=
	MatchQ[ s, HoldComplete[_Symbol ...]]


getValue[ name_, obj_] :=
	Module[ {t},
		t = name /. obj;
		If[ t === name, Null, t]	
	]

(*
  Basic Data
*)

defaultAllowedContexts = 
    {"Global`"}

defaultAllowedSymbols = 
    HoldComplete[ 
    Plus, Minus, Times, Power, Sqrt, Log, Log2, Log10, Exp,
    HoldComplete,
    Infinity,  Pi, E, Degree, GoldenRatio, Catalan, EulerGamma,
    OutputForm, StandardForm, List, 
    Sin, Cos, Tan, Sec, Csc, Cot,
    Sinc, 
    Sinh, Cosh, Tanh, Sech, Csch, Coth,
    ArcSin, ArcCos, ArcTan, ArcSec, ArcCsc, ArcCot,
    ArcSinh, ArcCosh, ArcTanh, ArcSech, ArcCsch, ArcCoth,
    True, False, Derivative, D, Dt, I,
    Greater, Less, GreaterEqual, LessEqual, Inequality, Equal,
    Re, Im, Abs, Sign, Conjugate, Arg, 
    Round, Floor, Ceiling, Max, Min, 
    Mod, Quotient,
    Not, And, Or, Xor,Union, Intersection, Complement,
    BitNot, 
    EvenQ, OddQ,
	FractionalPart, IntegerPart, Unitize,
    AiryAi, AiryAiPrime, AiryBi, AiryBiPrime,
    BesselJ, BesselK, BesselI, BesselY,
    Factorial, Binomial, Multinomial, 
    Gamma, Beta, LogGamma, PolyGamma,
    LegendreP, SphericalHarmonicY,
    HermiteH, LaguerreL,
    Erf,  Erfc,  Erfi, InverseErf, InverseErfc,
    ClebschGordan, ThreeJSymbol, SixJSymbol,
    Zeta, FresnelS,
    FresnelC, CosIntegral, SinIntegral, ExpIntegralE, 
    ExpIntegralEi, SinhIntegral, CoshIntegral,
    HypergeometricPFQ, Hypergeometric0F1, Hypergeometric1F1,
    Hypergeometric2F1, HypergeometricPFQRegularized,
    MeijerG,AppellF1,
    EllipticK, EllipticF, EllipticE, EllipticPi,
    JacobiZeta, EllipticNomeQ, EllipticLog,
    InverseEllipticNomeQ, JacobiAmplitude, 
    EllipticExp,
    DiracDelta, UnitStep, DiscreteDelta, KroneckerDelta,
    Identity, Function, Slot,
    GrayLevel, Hue, RGBColor, CMYKColor,
    Automatic, None, All, Null, O, C]



End[]

EndPackage[]

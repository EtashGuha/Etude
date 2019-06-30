
BeginPackage["TypeFramework`Utilities`TypeToPattern`"]

TypeToPattern;

Begin["`Private`"] 

Needs["TypeFramework`"]
Needs["TypeFramework`TypeObjects`TypeArrow`"]



TypeToPattern[ty0_] :=
    With[{
        ty = ty0["canonicalize"]["unresolve"]
    },
        toPattern[<||>, ty, <||>]
    ];

makePatternVariable[t0_?StringQ] :=
    With[{t = Symbol[t0]}, Pattern[t, Blank[]]];

toPattern[st_, Type[t_], opts_?AssociationQ] :=
    TypeSpecifier[toPattern[st, t, opts]];

toPattern[st_, TypeSpecifier[t_], opts_?AssociationQ] :=
    TypeSpecifier[toPattern[st, t, opts]];

toPattern[st_, TypeForAll[vars_, qual_], opts_?AssociationQ] :=
    With[{
        pvars = makePatternVariable /@ vars
    },
    With[{
        newSt = Join[st, AssociationThread[vars -> pvars]]
    },
        toPattern[newSt, qual, opts]
    ]];
    
toPattern[st_, ts_?ListQ, opts_?AssociationQ] :=
    toPattern[st, #, opts]& /@ ts;

toPattern[st_, TypeVariable[name_], opts_?AssociationQ] :=
    TypeVariable[toPattern[st, name, opts]];
    
toPattern[st_, TypeConstructor[name_], opts_?AssociationQ] :=
    TypeConstructor[name];
    
toPattern[st_, TypeApplication[type_, args_], opts_?AssociationQ] :=
    TypeApplication[toPattern[st, type, opts], toPattern[st, args, opts]];
    
toPattern[st_, TypeArrow[args_, res_], opts_?AssociationQ] :=
    TypeArrow[toPattern[st, args, opts], toPattern[st, res, opts]];
    
toPattern[st_, TypeSequence[type_, {min_, max_}], opts_?AssociationQ] := (
    Print["this needs to desugar to a blank null or blank ... "];
    TypeSequence[toPattern[st, type, opts], {min, max}];
);
    
toPattern[st_, TypeQualified[preds_, type_], opts_?AssociationQ] :=
    With[{
        ppreds = toPattern[st, preds, opts],
        ptype = toPattern[st, type, opts]
    },
        If[ppreds === {},
            ptype,
            TypeQualified[ppreds, ptype]
        ]
    ];
    
toPattern[st_, TypePredicate[types_, test_], opts_?AssociationQ] :=
    TypePredicate[toPattern[st, types, opts], test];
    
toPattern[st_, args_?ListQ -> res_, opts_?AssociationQ] :=
    toPattern[st, args, opts] -> toPattern[st, res, opts];
    
toPattern[st_, name_?StringQ, opts_?AssociationQ] :=
    Lookup[st, name, name];
    
End[];

EndPackage[];

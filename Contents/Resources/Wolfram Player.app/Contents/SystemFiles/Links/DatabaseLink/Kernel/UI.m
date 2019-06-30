(* Author:          Dillon Tracy *)
(* Copyright:       Copyright 2013, Wolfram Research, Inc. *)

Begin["`UI`Private`"]

(* Window that opens when the user must supply a password to access a database. *)

Options[PasswordDialog] = Join[
    {
        "Label" -> "Enter username and password"
    },
	Options[Grid],
	Options[DialogInput]
];
    
SetOptions[PasswordDialog,
	Alignment -> {{Right, Left}, Automatic, {{1, 1} -> Center, {4, 1} -> Center}}
    , Spacings -> {Automatic, 0.75}
    , WindowTitle -> "Enter credentials"
    , WindowFloating -> True
];
  
$defaultPasswordDialogGridOptions = FilterRules[Options[PasswordDialog], Options[Grid]];
$defaultPasswordDialogDialogInputOptions = FilterRules[Options[PasswordDialog], 
   FilterRules[Options[DialogInput], Except@Options[Grid]]];

PasswordDialog[o:OptionsPattern[]] := PasswordDialog[{Null, Null}, o];
PasswordDialog[{u_, p_}, o:OptionsPattern[]] := Module[
    {res, gridOpts, dialogOpts, lab = OptionValue["Label"], diOpts},
  
    gridOpts = FilterRules[Flatten[{o}], Options[Grid]];
    dialogOpts = FilterRules[Flatten[{o}], FilterRules[Options[DialogInput], Except@Options[Grid]]];
  
    diOpts = Join[
        {
            WindowTitle -> (WindowTitle /. dialogOpts /. Options[PasswordDialog]),
            WindowFloating -> (WindowFloating /. dialogOpts /. Options[PasswordDialog])
        },
        dialogOpts
    ];
    (* InputField[] will null non-string arguments for username, password *)
    With[ {opts = diOpts },
    res = DialogInput[
    	{myUser = u, myPass = p},
        Grid[{
                {lab, SpanFromLeft},
	            {"Username:", InputField[Dynamic[myUser], String, FieldHint -> "username"]},
	            {"Password:", InputField[Dynamic[myPass], String, FieldMasked -> True, FieldHint -> "password"]},
	            {ChoiceButtons[{DialogReturn[{myUser, myPass}], DialogReturn[$Canceled]}], SpanFromLeft}
            }
            , Sequence @@ gridOpts
            , Sequence @@ $defaultPasswordDialogGridOptions
        ],
       opts
        (*,
             Bug? DialogInput fails with its own options. 
        $defaultPasswordDialogDialogInputOptions*)
        ]
	];
	res
]


(* 
 * This version can be called from within a DialogInput, provided
 * the status flag is Dynamically monitored.
 *)

Options[NestablePasswordDialog] = Join[
    {
        "Label" -> "Enter username and password"
    },
    Options[Grid],
    Options[CreateDialog]
];

SetOptions[NestablePasswordDialog,
    Alignment -> {{Right, Left}, Automatic, {{1, 1} -> Center, {4, 1} -> Center}}
    , Spacings -> {Automatic, 0.75}
    , WindowTitle -> "Enter credentials"
    , NotebookEventActions -> {}
    , Modal -> True
    , WindowFloating -> True
];

$defaultNestablePasswordDialogGridOptions = FilterRules[Options[NestablePasswordDialog], Options[Grid]];
$defaultNestablePasswordDialogCreateDialogOptions = FilterRules[Options[NestablePasswordDialog], 
   FilterRules[Options[CreateDialog], Except@Options[Grid]]];

SetAttributes[NestablePasswordDialog, HoldAll]; 
NestablePasswordDialog[{u_, p_}, status_, o:OptionsPattern[]] := Module[
    {gridOpts, dialogOpts},

    status = Null;

    gridOpts = FilterRules[Flatten[{o}], Options[Grid]];
    dialogOpts = FilterRules[Flatten[{o}], FilterRules[Options[CreateDialog], Except@Options[Grid]]];

    CreateDialog[
    	Grid[{
    	       	{"Enter username and password", SpanFromLeft},
    		    {"Username:", InputField[Dynamic[u], String, FieldHint -> "username"]}, 
    		    {"Password:", InputField[Dynamic[p], String, FieldMasked -> True, FieldHint -> "password"]},
    		    {ChoiceButtons[{NotebookClose[]; status = True, NotebookClose[];}], SpanFromLeft}
    	   }
    	   , Sequence @@ gridOpts
    	   , Sequence @@ $defaultNestablePasswordDialogGridOptions
        ],
        Sequence @@ Join[
        	{
                WindowTitle -> (WindowTitle /. dialogOpts /. Options[NestablePasswordDialog]),
                NotebookEventActions -> (NotebookEventActions /. dialogOpts /. Options[NestablePasswordDialog]),
                Modal -> (Modal /. dialogOpts /. Options[NestablePasswordDialog]),
                WindowFloating -> (WindowFloating /. dialogOpts /. Options[NestablePasswordDialog])
        	},
            dialogOpts
        ]
    ]
];
  
End[] (* "`UI`Private`" *)
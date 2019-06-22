Package["Databases`Common`"]

PackageImport["Databases`"]



PackageExport["DBResource"]
PackageExport["DBResourceQ"]
PackageExport["DBResourceCheckInstall"]


pacletFind[name_String] := PacletManager`PacletFind[name]

pacletUninstall[name_String] := (
		PacletManager`PacletUninstall /@ pacletFind[name];
		PacletManager`RebuildPacletData[]
	)

Options[pacletInstall] = {
    "Update" -> False
}

pacletInstall[name_String, opts:OptionsPattern[]] :=
	Module[{paclet, updateQ,  check},						
		If[TrueQ[OptionValue["Update"]],
			pacletUninstall[name]
		];
		paclet = pacletFind[name];			
		If[paclet =!= {},
			Return[paclet]
		];	
		If[!TrueQ[PacletManager`$AllowInternet],
            DBRaise[pacletInstall, "internet_disallowed", {name}];
		];					
		Quiet @ Check[
			PacletTools`PacletGet[name, name, Automatic, "Update" -> True],
			(* some error *)
			DBRaise[pacletInstall, "paclet_download_error", {name}]
		];
		paclet = pacletFind[name];
		If[paclet === {},
			DBRaise[pacletInstall, "paclet_install_failure", {name}],
			(* else *)
			paclet
		]		
	]

SetAttributes[DBResource, HoldFirst]

Options[DBResource] = {
  "CheckInstallFunction" ->  Automatic,
  "Installer" -> Function[{name, o}, pacletInstall[name]],
  "Initializer" -> None,
  "InstallListeners" -> <||>
}

DBResource[
  pacletName:_?StringQ,
  OptionsPattern[]
] :=
    Module[{data},
        With[{
            checkInstalled = Replace[
                OptionValue["CheckInstallFunction"], {
                    Automatic -> Function[pacletInfo, pacletInfo =!= {}],
                    check_ :> Function[pacletInfo, pacletInfo =!= {} && check[pacletInfo]]
                }
            ],
            installer = OptionValue["Installer"],
            o =  DBResource[data],
            init = If[# === None, Function[Null], #] & [OptionValue["Initializer"]],
            listeners = OptionValue["InstallListeners"]
            },
            data = <|
                "Installed" :>  TrueQ[checkInstalled[data["PacletInfo"]]],
                "Install"   :>
                    If[!data["Installed"],
                        With[ {installed = installer[pacletName, o]},
                            If[FailureQ[installed] || !data["Installed"],
                                DBRaise[DBResource, "resource_failed_to_install", {o}]
                            ];
                            Scan[#[o]&, data["InstallListeners"]];
                            installed
                        ]
                    ],
                "Uninstall" :>  pacletUninstall[pacletName],   
                "PacletInfo" :> PacletManager`PacletInformation[pacletName],
                "PacletResource" -> Function[name, PacletManager`PacletResource[pacletName, name]],
                "Paths"     ->  <||>,
                "InstallListeners" -> listeners,
                "Meta"  -> <|
                    "RequiresInstall" -> {"Paths"}
                |>
            |>;
            init[o];
            o
        ]
    ]

DBResourceQ[DBResource[var_Symbol]] := MatchQ[var, _Association?AssociationQ]

DBResource[var_]["Set"][path__, value_] :=
    var[[path]] = value

DBResource[var_]["SetDelayed"][path___, key_ :> val_] :=
    (AppendTo[var[[path]], key :> val];)

(r: DBResource[data_])["Get"][fst_, rest___] /; And[
    MemberQ[data["Meta",  "RequiresInstall"], fst],
    ! r["Installed"]
] :=
    Module[{},
        r["Install"];
        If[!TrueQ[r["Installed"]],
            DBRaise[
                DBResource, 
                "resource_install_failed_on_property_or_method_call", 
                {r, {fst, rest}}
            ]
        ];
        r["Get"][fst, rest]
    ]

(r: DBResource[data_])["Get"][path__] := With[{result = data[[path]]},
    If[MissingQ[result],
        DBRaise[DBResource, "invalid_property_path", {path}],
        (* else *)
        result
    ]
]

(r: DBResource[data_])["PropertyExists"][path__] :=
    !MissingQ[data[[path]]]

(r:_DBResource ? DBResourceQ)["Append"][path___, val_] :=
    r ["Set"][path, Append[r["Get"][path], val]]


(r:_DBResource ? DBResourceQ)["AddInstallListener"][tag_ -> f_] :=
    r["Append"]["InstallListeners", tag -> f]

(r:_DBResource ? DBResourceQ)[path___, prop_String] := r["Get"][path, prop] /; !MatchQ[
    {path, prop},
    {"Get" | "Set" | "SetDelayed" | "PropertyExists" | "Append" | "AddInstallListener" }
]


DBResourceCheckInstall[obj_DBResource ? DBResourceQ, cond_] :=
    Function[
        code
        ,
        If[TrueQ[cond], obj["Install"]];
        code
        ,
        HoldFirst
    ]
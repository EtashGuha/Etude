Package["Databases`"]


(* NOTE!! We need here Needs rather than PackageImport, since PackageImport loads
** dependencies before resolving symbols in the current package - which we can't
** afford here, to avoid circular dependencies, given that we are importing Databases`
** into modules. OTOH, Needs imports at the time package code actually runs, and
** is fine for us.
**
** So, the sequence of actions is this: first all symbols get resolved in Databases`
** (this package, which is the main one), then the modules get loaded with Needs, when
** this code below actually runs. Because by that time, all symbols from Databases`
** have been resolved, we can import Databases` into modules and be sure that these
** pubic symbols from Databases` will be available to the code in modules.
**
** HOWEVER, the side effect of this is that if we want to use any symbols exported
** by the modules, here, we need to use fully - qualified names. This is because the
** symbol resolver of new-style packages is static (based on static pass / analysis),
** and so, as noted already, symbol resolution here (in Databases`) is done *before*
** the modules are loaded / added to the $ContextPath, so using short names for
** module-exported symbols would prompt the symbol resolver to create them in the
** current Databases`Interface`PackagePrivate` context and use those ones, which is
** not what we want - thus, fully-qualified names.
**
** OTOH, fully-qualified names of module-exported symbols  actually are advantageous
** here, because they tell us clearly, where (in which modules) symbols currently live.
*)

Block[{$ContextPath},
    (* NOTE: Loading order DOES MATTER. With an exception of Databases` module, which
    ** gets imported into every other module as described above, with the loading
    ** order as below, each module only depends on / imports some of the modules
    ** which have been loaded before it, but none of those that come after.
    **
    ** Note also that this is not a hard requirement, but rather our preferred way.
    ** Since all modules are separately and independently available on $Path during
    ** this loading process, any loading order would technically work, since they
    ** would lazily load the ones they depend on.
    *)
    Needs["Databases`Common`"];
    Needs["Databases`Python`"];
    Needs["Databases`Schema`"];
    Needs["Databases`SQL`"];
    Needs["Databases`Database`"];
    Needs["Databases`Entity`"];
]

If[
    ! ValueQ[$Databases],
    $Databases = Missing["DatabaseAbsent"]
]


If[
    ! ValueQ[$DatabaseAuthentications],
    $DatabaseAuthentications = Missing["AuthenticationAbsent"]
]


Databases`Common`DBSetDefaultStackTraceFormatting[
    p: Alternatives[
        _RelationalDatabase,
        _Databases`SQL`DBQueryObject,
        _Databases`SQL`DBQueryBuilderObject,
        _Databases`DatabaseStore,
        _Databases`Database`DBResultSet,
        _System`EntityStore
    ] :> Databases`Common`DBStackFrameMakeBoxes[p]
]

Databases`Common`DBSetDefaultStackTraceFormatting[
    expr: _Databases`SQL`DBRawFieldName | _Blank :> MakeBoxes[expr]
]

Databases`Common`DBSetDefaultStackTraceFormatting[
    pythonFailure: Failure[
        "DatabaseFailure", assoc_/; KeyExistsQ[ assoc, "Traceback"]
    ] :> MakeBoxes[pythonFailure]
]

Internal`DeleteCache[i$DBResourceObject]

$DBResourceObject := Databases`Common`DBCheckAndSetCache[
    i$DBResourceObject,
    Databases`Common`DBResource[
        "DatabasesResources",
        "CheckInstallFunction" -> Function[pacletInfo,
            (* To exclude dev. version of the paclet, matters only for development *)
            (* TODO: remove this, once we split the resources paclet into a separate repo *)
            !OrderedQ[{Lookup[pacletInfo,  "Version"], "0.0.1"}]
        ],
        "Initializer" -> Function[obj,
            obj["SetDelayed"][
                "Paths",
                "Oracle" :> FileNameJoin[
                    {obj["PacletResource"]["SystemFiles"], "Oracle"}
                ]
            ]
        ],
        "InstallListeners" -> <|
            "Python" -> Function[obj, Databases`Python`DBClearPythonProcessCache[]]
        |>
    ]
]
Package["Databases`"]


SetAttributes[DatabaseFunction, HoldAll]

SyntaxInformation[DatabaseFunction] = {"LocalVariables" -> {"Solve", {0, Infinity}}}

Databases`Common`DBCreateProxyDefinitions[
    DatabaseReferences,
    HoldPattern[DatabaseReferences[]] :> Databases`Schema`DBReferences[]
]

Databases`Common`DBCreateProxyDefinitions[
    DatabaseQueryToSymbolicSQL,
    HoldPattern[DatabaseQueryToSymbolicSQL[q_?Databases`Database`DBQueryQ]] :>
        Databases`Database`DBQueryToSymbolicSQL[q]
]

With[{schemaPattern = Databases`Schema`$DBSchemaPattern},
    Databases`Common`DBCreateProxyDefinitions[
        DatabaseRunQuery,
        HoldPattern[DatabaseRunQuery[
            query:_?Databases`Database`DBQueryQ|_String, 
            schema:schemaPattern:Inherited, 
            method:_String|{__String}|All:"Preview", 
            keys:Except[_?OptionQ]:Automatic, 
            opts:OptionsPattern[]
        ]] :>
            Databases`Database`DBRunQuery[query, schema, method, keys, opts]
    ]
]


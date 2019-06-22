Paclet[
    Name -> "Databases",
    Version -> "0.3",
    MathematicaVersion -> "12+",
    Loading -> Automatic,
    Extensions -> {
        {
            "Kernel",
            Context -> {"DatabasesLoader`", {"Databases`", "DatabasesLoader.m"}},
            Symbols -> {
                "System`DatabaseReference",
                "System`RelationalDatabase",
                "System`IncludeRelatedTables",
                "System`DatabaseConnect",
                "System`DatabaseDisconnect"
			}
        },
        {"JLink"},
        {"Documentation", Language -> "English"},
        {"Resource", Root -> ".", Resources -> {
            "Python",
            "Dependencies",
            "Data"
        }}
    }
]

Paclet[
    Name -> "Templating",
    Version -> "1.40",
    MathematicaVersion -> "10.1+",
    Description -> "Templating",
    Creator -> "Riccardo Di Virgilio <riccardod@wolfram.com>, Carlo Barbieri <carlob@wolfram.com>, Taliesin Beynon <taliesinb@wolfram.com>",
    Loading -> Automatic,
    Extensions -> {
        {"Resource", 
            Root -> "Resources", 
            Resources -> {"StaticLoader", "TemplateLoader"}
        },
        {"Kernel", 
            HiddenImport -> True,
            Context -> {"Templating`"}, 
            Symbols -> {
                (* Templating symbols *)
                "System`$TemplatePath",
                "System`CombinerFunction",
                "System`DefaultValue",
                "System`FileTemplate",
                "System`FileTemplateApply",
                "System`InsertionFunction",
                "System`StringTemplate",
                "System`TemplateApply",
                "System`TemplateEvaluate",
                "System`TemplateExpression",
                "System`TemplateIf",
                "System`TemplateObject",
                "System`TemplateSequence",
                "System`TemplateSlot",
                "System`TemplateUnevaluated",
                "System`TemplateVerbatim",
                "System`TemplateWith",
                "System`XMLTemplate",

                (* HTML Utilities *)
                "Templating`ExportHTML",
                "System`$HTMLExportRules",

                (* Symbolic pages *)
                "System`GalleryView",
                "System`Pagination",
                "Templating`Webpage",
                "Templating`HTMLTemplate",

                (* Panel language *)
                "Templating`WebHorizontalLayout",
                "Templating`InterfaceSwitched",
                "Templating`WebItem",
                "Templating`WebVerticalLayout",
                "Templating`DynamicLayout"
            }
        }
    }
]
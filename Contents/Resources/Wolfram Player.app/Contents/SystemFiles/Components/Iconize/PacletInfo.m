Paclet[
	Name -> "Iconize",
	Version -> "2.0",
	MathematicaVersion -> "10+",
    Description -> "Generate an icon for an arbitrary WL expression",
	Creator -> "Gopal Sarma <gopals@wolfram.com>",
	Loading -> Automatic,
	Extensions -> {
		{"Resource", Root -> "Resources", Resources -> {"CategoryIcons", "DeploymentIcons", "defaultSpikey.png", "formatFormats.m", "notebook_template.png"}}, 
		{
			"Kernel", 
			HiddenImport -> True,
			Context -> {"IconizeLoader`", "Iconize`"},
			Symbols -> {
				"Iconize`IconizedImage"
			}
		}
	}
]
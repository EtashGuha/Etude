(* Paclet Info File *)

(* created 2015/04/16*)

Paclet[
    Name -> "SMTPLink",
    (* For paclet builds, the Version field is ignored and Jenkins constructs its own version number at build time.
     * See the 'Paclet.SMTPLink.prebuild' target in re_build_SMTPLink.xml.
     * For the convenience of developers using local branches, set the Version higher 
     * than the paclet build numbers: *)
    Version -> "11.3.0",
    MathematicaVersion -> "11.3+",
    Loading -> Automatic,
	Extensions -> {
		{"Kernel",
		MathematicaVersion->"11.3",
			Root->"Kernel",
			Context->{
				"SMTPLinkLoader`",
				"SMTPLink`"
		},
			Symbols-> {}
        },
        {"Kernel",
            MathematicaVersion->"11.3+",
            Root->"Kernel",
            Context->{
                "SMTPLinkLoader`",
                "SMTPLink`"
            },
            Symbols-> {
            }
        },
		{"FrontEnd", Prepend -> True},
		{"Documentation", Language -> "English"}
	}
]


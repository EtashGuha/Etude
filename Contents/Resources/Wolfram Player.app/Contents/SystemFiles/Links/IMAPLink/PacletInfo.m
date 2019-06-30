(* Paclet Info File *)

(* created 2015/04/16*)

Paclet[
    Name -> "IMAPLink",
    (* For paclet builds, the Version field is ignored and Jenkins constructs its own version number at build time.
     * See the 'Paclet.IMAP.prebuild' target in re_build_IMAP.xml.
     * For the convenience of developers using local branches, set the Version higher 
     * than the paclet build numbers: *)
    Version -> "11.3.8",
    MathematicaVersion -> "11.3+",
    Loading -> Automatic,
	Extensions -> {
		{"Kernel",
		HiddenImport -> True,	
			Root->"Kernel",
			Context->{
				"IMAPLinkLoader`",
				"IMAPLink`"
		},
			Symbols-> {"IMAPLink`IMAPConnect", "IMAPLink`IMAPExecute"}
        },
		{"FrontEnd", Prepend -> True},
		{"Documentation", Language -> "English"}
	}
]


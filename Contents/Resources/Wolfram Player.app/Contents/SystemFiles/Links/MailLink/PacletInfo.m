(* Paclet Info File *)

(* created 2015/04/16*)

Paclet[
    Name -> "MailLink",
    (* For paclet builds, the Version field is ignored and Jenkins constructs its own version number at build time.
     * See the 'Paclet.Mail.prebuild' target in re_build_Mail.xml.
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
				"MailLinkLoader`",
				"MailLink`"
		},
			Symbols-> {
				"System`MailServerConnect",
				"System`MailServerConnection",
				"System`MailFolder",
				"System`MailSearch",
				"System`MailItem",
				"System`MailExecute",
				"System`$DefaultMailbox",
				"System`$IncomingMailSettings"
				}
        },

		{"FrontEnd", Prepend -> True},
		{"Documentation", Language -> "English"}
	}
]


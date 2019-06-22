(* Paclet Info File *)

(* created 2017/10/26*)

Paclet[
    Name -> "ServiceConnection_Twilio",
    Version -> "11.3.14",
    MathematicaVersion -> "10.2+",
    Loading -> Automatic,
    Extensions ->
        {
            {"Kernel", Root -> "Kernel", Context ->
                {"Twilio`", "TwilioLoad`", "TwilioSendMessage`", "TwilioFunctions`"}
            },
            {"FrontEnd"},
            {"Documentation", MainPage -> "ReferencePages/Symbols/Twilio", Language -> All}
        }
]

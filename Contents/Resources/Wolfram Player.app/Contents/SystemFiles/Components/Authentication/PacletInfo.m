(* Paclet Info File *)

(* created 2018/10/09*)

Paclet[
    Name -> "Authentication",
    Version -> "12.0.2",
    MathematicaVersion -> "11.2+",
    Description -> "Authentication Framework",
    Creator -> "Juan Leiva <juanl@wolfram.com>",
    Loading -> Automatic,
    Extensions -> 
        {
            {"Kernel", Symbols -> 
                {
                 "Authentication`AuthenticateHTTPRequest",
                 "System`SecuredAuthenticationKey",
                 "System`SecuredAuthenticationKeys",
                 "System`GenerateSecuredAuthenticationKey",
                 "System`SetSecuredAuthenticationKey",
                 "System`$SecuredAuthenticationKeyTokens",
                 "System`Authenticate"
                },
                HiddenImport -> True,
                Context -> {"Authentication`"}
            }
        }
]



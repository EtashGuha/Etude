Paclet[
    Name -> "Blockchain",
    Version -> "12.0.103",
    MathematicaVersion -> "12+",
    Description -> "Blockchain Library",
    Creator -> "Daniel Suarez, Piero Sanchez, Akira Toma",
    Loading -> Automatic,
    Extensions -> {
        {
            "Resource", 
            Root -> "../Resources", 
            Resources -> {"Solidity_Templates", "Icons"}
        },
        {"Kernel", 
            Symbols -> {
                
                "System`BlockchainData",
                "System`BlockchainBlockData",
                "System`BlockchainTransactionData",
                "System`BlockchainTransaction",
                "System`BlockchainTransactionSign",
                "System`BlockchainTransactionSubmit",
                "System`BlockchainAddressData",
                "System`BlockchainTokenData",
                "System`BlockchainGet",
                "System`BlockchainPut",                
                "System`BlockchainKeyEncode",
                "System`BlockchainContractValue",
                "System`BlockchainBase",
                "Blockchain`ContractEncode",
                "Blockchain`ContractDecode",
                "Blockchain`BlockchainContract",
                "Blockchain`ValidBlockchainBaseQ",
                "Blockchain`$TemplateBase"
            },
            HiddenImport -> True,
            Context -> {"Blockchain`"}
        }
    }
]

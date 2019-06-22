(* ::Package:: *)

Package["Cryptography`"]


PackageExport["EncryptedObject"]

(*EncryptedObject::usage = "Represents a piece of encrypted information."*)

EncryptedObject[data_Association][key_] := data[key]
EncryptedObject /: Normal[enc_EncryptedObject] := enc["Data"]
EncryptedObject[data_Association]["Properties"] := Keys[data];

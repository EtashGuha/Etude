(* ::Package:: *)

Package["Cryptography`"]


getEncryptionTarget[source_, target_] := target
getEncryptionTarget[source_, Automatic] := CloudObject[]
getEncryptionTarget[source_String, Automatic] := source<>".mx"
getEncryptionTarget[source_CloudObject, Automatic] := CloudObject[]


getDecryptionTarget[source_, target_] := target
getDecryptionTarget[source_, Automatic] := CloudObject[]
getDecryptionTarget[source_String, Automatic] := If[StringMatchQ[source, __~~".mx"], StringDrop[source,-3], FileNameJoin[{DirectoryName[source],FileBaseName[source]<>"_decrypted"}]]
getDecryptionTarget[source_CloudObject, Automatic] := CloudObject[]


getPlaintext[source_] := $Failed
getPlaintext[source_String] := If[FileExistsQ[source], Quiet@Check[ByteArray@Import[source, "Binary"], $Failed], $Failed]
getPlaintext[source_CloudObject] := If[FileExistsQ[source], Quiet@Check[ByteArray@CloudImport[source,"Binary"], $Failed], $Failed]


getCiphertext[source_] := $Failed
getCiphertext[source_String] := If[FileExistsQ[source], Quiet@Check[Import[source, "MX"], $Failed], $Failed]
getCiphertext[source_CloudObject] := If[FileExistsQ[source], Quiet@Check[CloudImport[source, "MX"], $Failed], $Failed]


putPlaintext[plaintext_, target_] := $Failed
putPlaintext[plaintext_, target_String] := Quiet@Check[Export[target, plaintext, "Binary"], $Failed]
putPlaintext[plaintext_, target_CloudObject] := Quiet@Check[CloudExport[plaintext, "Binary", target], $Failed]


putCiphertext[ciphertext_, target_] := $Failed
putCiphertext[ciphertext_, target_String] := Quiet@Check[Export[target, ciphertext, "MX"], $Failed]
putCiphertext[ciphertext_, target_CloudObject] := Quiet@Check[CloudExport[ciphertext, "MX", target], $Failed]


overwriteQ[target_] := False
overwriteQ[target_String] := FileExistsQ[target]


PackageExport["EncryptFile"]


Clear[EncryptFile];


Options[EncryptFile] = {OverwriteTarget -> False, Permissions -> Automatic};

(*
EncryptFile::overwrite = DecryptFile::overwrite = "Set OverwriteTarget -> True to allow overwriting.";
EncryptFile::source = "Could not read plaintext from source `1`.";
EncryptFile::target = "Could not write ciphertext to target `1`.";
DecryptFile::source = "Could not read ciphertext from source `1`.";
DecryptFile::target = "Could not write plaintext to target `1`.";
DecryptFile::perm = "Default permissions not set to \"Private\".  Decrypted plaintext may be publicly available on the cloud.  Specify permissions with the Permissions option to stop this message.";
*)

EncryptFile[key_, source_, Optional[targetI:Except[_Rule], Automatic], OptionsPattern[]] :=
	Catch @ Module[{plaintext, ciphertext, target, result},
		target = getEncryptionTarget[source, targetI];
		If[(!OptionValue[OverwriteTarget]) && targetI === Automatic && overwriteQ[target], Message[EncryptFile::overwrite]; Throw[$Failed]];
		plaintext = getPlaintext[source];
		If[plaintext === $Failed, Message[EncryptFile::source, source]; Throw[$Failed]];
		ciphertext = Check[Encrypt[key, plaintext], Throw[$Failed]];
		result = putCiphertext[ciphertext, target];
		If[result === $Failed, Message[EncryptFile::target, target]; Throw[$Failed]];
		If[targetI === Automatic && MatchQ[target, _CloudObject], SetPermissions[target, Replace[OptionValue[Permissions], Automatic :> $Permissions]]];
		result
	]


PackageExport["DecryptFile"]


Clear[DecryptFile]


Options[DecryptFile] = {OverwriteTarget -> False, Permissions -> Automatic};


DecryptFile[key_, source_, Optional[targetI:Except[_Rule], Automatic], OptionsPattern[]] :=
	Catch @ Module[{ciphertext, plaintext, target, result},
		target = getDecryptionTarget[source, targetI];
		If[(!OptionValue[OverwriteTarget]) && targetI === Automatic && overwriteQ[target], Message[DecryptFile::overwrite]; Throw[$Failed]];
		ciphertext = getCiphertext[source];
		If[ciphertext === $Failed, Message[DecryptFile::source, source]; Throw[$Failed]];
		plaintext = Check[Decrypt[key, ciphertext], Throw[$Failed]];
		result = putPlaintext[plaintext, target];
		If[result === $Failed, Message[DecryptFile::target, target]; Throw[$Failed]];
		If[
			targetI === Automatic && MatchQ[target, _CloudObject],
			SetPermissions[target,
				Replace[OptionValue[Permissions],
					Automatic :> (If[$Permissions =!= "Private", Message[DecryptFile::perm]]; $Permissions)]
		]];
		result
	]

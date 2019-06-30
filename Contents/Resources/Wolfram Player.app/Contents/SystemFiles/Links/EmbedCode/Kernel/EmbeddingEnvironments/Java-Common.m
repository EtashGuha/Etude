(* This "Common" file is intended to be shared by all Java language implementations (e.g., "Java", "Java-Jersey", Java-HttpClient", etc.)
   
   Pick a unique context. Other files that call functions defined here will use their fully-qualified context names.
*)

Begin["EmbedCode`JavaCommon`"]

(* The default name of the generated method that makes the APIFunction call. Overriden by the ExternalFunctionName option to EmbedCode. *)
$defaultFunctionName = "call"


(* This is the default delimiter that will be used for sending DelimitedSequence arguments. *)
$defaultDelimiter = ";"

(* Mappings from types recognized by the Interpreter[] function (these are the native WL-side types for APIFunction calls)
  to Java native types. This function should return either a string representing a native type name, or a list {"native type", extra values..}.
  In the case of returning a list, its elements will be Sequence`d in.
*)
interpreterTypeToJavaType["Integer" | "Integer8" | "Integer16" | "Integer32" | Integer] = "int"
interpreterTypeToJavaType["Integer64"] = "long"
interpreterTypeToJavaType["Number"] = "double"
interpreterTypeToJavaType["Boolean"] = "boolean"
interpreterTypeToJavaType["Date" | "DateTime"] = "java.util.Date"
interpreterTypeToJavaType["Image"] = "java.awt.image.BufferedImage"

interpreterTypeToJavaType[Restricted[type_, __]] := interpreterTypeToJavaType[type]

interpreterTypeToJavaType[DelimitedSequence[type_]] := interpreterTypeToJavaType[DelimitedSequence[type, $defaultDelimiter]]
interpreterTypeToJavaType[DelimitedSequence[type_, sep_String]] := {interpreterTypeToJavaType[type] <> "[]", sep}
interpreterTypeToJavaType[DelimitedSequence[type_, {first_, sep_, last_}]] :=  {interpreterTypeToJavaType[type] <> "[]", {first, sep, last}}

interpreterTypeToJavaType[_] = "String"


(* isBinaryType is applied to the result of interpreterTypeToJavaType. At the moment it is used to determine whether
   a request will need form-multipart treatment.
*)
isBinaryType[type_String] := isImage[type] || type == "byte[]"

isImage[type_String] := MatchQ[type, "Image" | "java.awt.Image" | "BufferedImage" | "java.awt.image.BufferedImage"]

is1DArrayType[type_String] := StringMatchQ[type, ___ ~~ WordCharacter ~~ "[]"]

End[]
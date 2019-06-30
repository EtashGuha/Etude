(* Wolfram Language package *)
Package["Databases`Python`"]

PackageScope["pythonSerialize"]
PackageScope["PyFunctionCall"]
PackageScope["PyMethodCall"]

SetAttributes[{serialize, serializeHead}, HoldAllComplete];
SetAttributes[{pythonSerialize}, HoldFirst];

serializeHead[head_Symbol] := {"_sym('", SymbolName[Unevaluated @ head], "')"};
serializeHead[expr_String] := {"_sym('", expr, "')"};
serializeHead[expr_] := serialize[expr]

serialize[key_, value_] := {serialize[key], ": ", serialize[value]}
serialize[PyMethodCall[method_String, expr_, args___]] := 
    {"(", serialize[expr], ").", serialize @ PyFunctionCall[method, args]}
serialize[PyFunctionCall[name_String, args___]] := 
    {name, "(", Riffle[List @@ Map[serialize, Hold[args]], ", "], ")"}
serialize[True] := "True"
serialize[False] := "False"
serialize[None | Null | _Missing] := "None"
serialize[b_ByteArray] := ByteArrayToString[b] (* this needs to be fixed, we need a way to transfer bytes to python *)
serialize[expr_Symbol] := serializeHead[expr];
serialize[Association[args___]] := {
    "{", 
    Riffle[
        List @@ serialize @@@ Hold[args], 
        ", "
    ], 
    "}"
}
serialize[List[args___]] := 
    {"[",  Riffle[List @@ Map[serialize, Hold[args]], ", "], "]"}
serialize[head_[args___]] :=
    {serializeHead[head], "(", Riffle[List @@ Map[serialize, Hold[args]], ", "],")"}
serialize[expr_] := ToString[expr, CForm]

pythonSerialize[expr_] := StringJoin[serialize[expr]]
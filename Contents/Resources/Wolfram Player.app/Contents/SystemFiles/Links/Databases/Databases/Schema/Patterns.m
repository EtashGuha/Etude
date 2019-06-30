(* Wolfram Language package *)

Package["Databases`Schema`"]

PackageExport["$DBSchemaPattern"]
PackageExport["$DBTablePattern"]
PackageExport["$DBReferencePattern"]
PackageExport["$DBStrictSchemaPattern"]


$DBSchemaPattern = Alternatives[
    Rule[_String, _],
    RuleDelayed[_String, _],
    _Association?AssociationQ,
    {RepeatedNull[_Rule | _RuleDelayed]},
    _File,
    None,
    Automatic,
    Inherited
]

$DBSchemaPattern = Append[
    $DBSchemaPattern, 
    Verbatim[RelationalDatabase][$DBSchemaPattern, ___]
]

$DBTablePattern = Alternatives[ _String | All, {RepeatedNull[_String]}]

$DBReferencePattern = Alternatives[
    _String | _URL | _File | _DatabaseReference | Automatic | None | Inherited
]

$DBStrictSchemaPattern = RelationalDatabase[_Association?AssociationQ, ___]
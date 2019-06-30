Begin["System`Convert`BSONDump`"]

Needs["MongoLink`"]

importBSON[path_String, ___] := importBSON[path];

importBSON[path_String] := Block[
    {bytes = Quiet @ BinaryReadList[path]},
	If[FailureQ[bytes], 
		Message[Import::format, "BSON"]; 
		$Failed,
		"Data" -> Replace[
            Quiet @ ToBSON[ByteArray[bytes]],
            {
                bo_BSONObject :> Normal[bo],
                a_ :> (Message[Import::format, "BSON"];$Failed)
            }
        ]
	]
];

exportBSON[path_String, any_, ___] := (Message[Export::invbson, any]; $Failed)

exportBSON[path_String, expr_Association ? AssociationQ] :=
    Replace[
        Quiet @ ToBSON[expr],
        {
            bo_BSONObject :> BinaryWrite[path, ByteArray[bo]],
            a_ :> (Message[Export::invbson, expr]; $Failed)
        }
    ];

End[]
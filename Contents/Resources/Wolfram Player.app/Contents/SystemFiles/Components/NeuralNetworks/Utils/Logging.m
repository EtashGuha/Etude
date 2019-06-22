Package["NeuralNetworks`"]


PackageScope["ToLogger"]

ToLogger[File[file_String]] /; StringEndsQ[file, ".wl"] := (
	If[FileExistsQ[file], DeleteFile[file]];
	myPutAppend[wrapWithTime[#], file]&
)

myPutAppend[expr_, file_] := Scope[
	stream = OpenAppend[file, PageWidth -> Infinity];
	If[FailureQ[stream], Return[$Failed]];
	Write[stream, expr];
	Close[stream];
];

General::badnetlogft = "Logging file `` is of an unsupported type. Supported types include '.wl' files."
ToLogger[File[file_String]] := 
	ThrowFailure["badnetlogft", file]

ToLogger[b_Bag] := 
	BagPush[b, wrapWithTime[#]]&;

ToLogger[True] :=
	Print;

ToLogger["StandardError"] := 
	ToLogger @ If[$Notebooks, First[$Output], OutputStream["stderr", 2]];

ToLogger[stream_OutputStream] := (
	SetOptions[stream, PageWidth -> Infinity];
	PutAppend[wrapWithTime[#], stream]&
);

ToLogger[False|None] := Hold;

General::badnetlogspec = "`` is not a supported spec for a logger, which should be one of True, False, File[...], OutputStream[...], or Internal`.`Bag[...].";
ToLogger[e_] := 
	ThrowFailure["badnetlogspec", e];

wrapWithTime[e_] := {SessionTime[], If[IntegerQ[$absoluteBatch], $absoluteBatch, 0]} -> e;
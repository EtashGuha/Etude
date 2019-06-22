Package["NeuralNetworks`"]


PackageExport["LengthVar"]

SetUsage @ "
LengthVar[id$] represents a unique variable representing an unknown sequence."


PackageScope["NewLengthVar"]

SetUsage @ "
NewLengthVar[] makes a LengthVar[$$] containing a random id."

NewLengthVar[] := LengthVar[RandomInteger[2^31]]


PackageScope["NameToLengthVar"]
PackageScope["LengthVarToName"]
PackageScope["NamedLengthVarQ"]

NameToLengthVar[s_] := LengthVar[2^32 + LetterNumber[s]];
LengthVarToName[LengthVar[n_] | n_Integer] := FromLetterNumber[n - 2^32];
NamedLengthVarQ[LengthVar[n_] | n_Integer] := n >= 2^32;


PackageScope["GetLengthVarID"]

GetLengthVarID[VarSequenceP[id_]] := id;
GetLengthVarID[coder:CoderP] := GetLengthVarID[CoderType[coder]];
GetLengthVarID[_] := $Failed;


PackageScope["MakeVarSeq"]

MakeVarSeq[t_] := SequenceT[NewLengthVar[], t];
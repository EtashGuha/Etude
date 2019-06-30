Package["ReinforcementLearning`"]

PackageImport["GeneralUtilities`"]

(*----------------------------------------------------------------------------*)
PackageScope["withRandomSeeding"]

SetAttributes[withRandomSeeding, HoldRest];

withRandomSeeding[None, body_] := body;
withRandomSeeding[other_, body_] := BlockRandom[body, RandomSeeding -> other];

(*----------------------------------------------------------------------------*)
PackageScope["environmentCreateHandle"]
environmentCreateHandle[args___] := CreateUUID[];
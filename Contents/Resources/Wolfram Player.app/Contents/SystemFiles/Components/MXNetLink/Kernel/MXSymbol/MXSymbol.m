Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["MXSymbol"]

SetUsage @ "
MXSymbol[id$] represents an abstract computation DAG managed by MXNet."

(******************************************************************************)

PackageExport["MXSymbolQ"]

MXSymbolQ[MXSymbol[_Integer]?System`Private`NoEntryQ] := True
MXSymbolQ[_] = False

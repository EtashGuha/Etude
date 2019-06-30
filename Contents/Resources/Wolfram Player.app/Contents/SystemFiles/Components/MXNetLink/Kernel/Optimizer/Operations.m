Package["MXNetLink`"]


PackageExport["ArrayOptimizer"]


(******************************************************************************)

ArrayOptimizer[assoc_Association][] := assoc["UpdateFunction"][];

(******************************************************************************)

ArrayOptimizer /: Part[ArrayOptimizer[assoc_Association], key_] := Part[assoc, key];

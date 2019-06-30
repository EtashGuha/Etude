Package["ExternalStorage`"]

(* This is a workaround for https://bugs.wolfram.com/show?number=361528 *)
Needs["CURLLink`"]
CURLLink`CURLInitialize[];
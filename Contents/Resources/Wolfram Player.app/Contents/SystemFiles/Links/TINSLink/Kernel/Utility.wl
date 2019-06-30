
(* Wolfram Language Package *)

BeginPackage["TINSLink`Utility`"]

DatasetRow::usage = "Formats a list of packet information into a dataset row.";
DatasetFrom::usage = "Applies DatasetRow for each packet in a given list.";
DatasetFileFrom::usage = "";
DatasetFileRow::usage = "";

Begin["Private`"]

DatasetRow[n_Integer,t_, t2_, info_] := Association["PacketNumber" -> n, "RelativeTime" -> UnitConvert[t,"Milliseconds"], "AbsoluteTime"-> t2, "Info" -> info];
DatasetRow[n_Integer,t_, t2_, iface_, info_] := Association["PacketNumber" -> n, "RelativeTime" -> UnitConvert[t,"Milliseconds"], "AbsoluteTime"-> t2, "Interface" -> iface, "Info" -> info];

DatasetFrom[l_List] := Dataset[
	MapIndexed[
		DatasetRow[First[#2],Sequence@@#1]&,
		SortBy[First]@l
	]
];

DatasetFrom[{{}}] := Dataset[{}];
DatasetFrom[] := Dataset[{}];


DatasetFileRow[n_Integer,t_, info_] := Association["PacketNumber" -> n, "AbsoluteTime"-> t, "Info" -> info];
DatasetFileFrom[l_List] := Dataset[
	Function[{all}, 
 Join @@@ Transpose[{KeyTake["PacketNumber"] /@ all, 
    Function[{allt}, <|"RelativeTime" -> # - Min[allt]|> & /@ 
       allt]@(all[[All, "AbsoluteTime"]]), 
    KeyDrop["PacketNumber"] /@ all}]]@MapIndexed[
		DatasetFileRow[First[#2],Sequence@@#1]&,
		SortBy[First]@l
	]
];
DatasetFileFrom[{{}}] := Dataset[{}];
DatasetFileFrom[] := Dataset[{}];

(* In case something goes wrong, pass the error through *)
DatasetFrom[e_LibraryFunctionError] := e;
DatasetFileFrom[e_LibraryFunctionError] := e;


End[]

EndPackage[]

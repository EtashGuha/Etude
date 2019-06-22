Package["Databases`Database`"]

PackageImport["Databases`SQL`"]

PackageScope["serializeData"]

serializeData[data: <|(_String?StringQ -> {<|(_String?StringQ -> _)...|>...})...|>, backend_] := 
	Map[
		serializeData[#, backend] &,
		data,
		{3}
	]

serializeData[_Missing?MissingQ, ___] := None

serializeData[d: True|False, backend_] := 
	serializeData[d, backend, "Boolean"]

serializeData[d_TimeObject, backend_] := 
	serializeData[d, backend, "Time"]

serializeData[d_DateObject, backend_] := 
	Replace[
		d["Granularity"], {
			"Day" :> serializeData[d, backend, "Date"],
			"Instant"|"Second" :> serializeData[d, backend, "DateTime"],
			_ :> DBRaise[serializeData, "unsupported_date", {d}]
		}
	]

serializeData[d_, _] := d

(* We are using part extraction because DateList is converting the values to $TimeZone *)
serializeData[d_, _, "Time"] :=
    "SATime"[DateValue[d, "TimeZone"], Sequence @@ d[[1, 1;;3]]]

serializeData[d_, _, "Date"] :=
    "SADate"[DateValue[d, "TimeZone"], Sequence @@ d[[1, 1;;3]]]

serializeData[d_, _, "DateTime"] :=
    "SADateTime"[DateValue[d, "TimeZone"], Sequence @@ d[[1, 1;;6]]]

serializeData[d_, __] := d

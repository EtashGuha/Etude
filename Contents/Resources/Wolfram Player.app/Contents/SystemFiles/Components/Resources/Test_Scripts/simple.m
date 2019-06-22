
temperature = WeatherData[GeoPosition[Entity["City", {"Lima", "Lima", "Peru"}]], "Temperature"];

result = Developer`WriteRawJSONString[<|"Description" -> "Temperature",
  "Location" -> "Lima, Peru",
  "Magnitude" -> QuantityMagnitude[temperature],
  "Unit" -> QuantityUnit[temperature],
  "DateTime" -> DateString["ISODateTime"]|>, "Compact" -> True]

Begin["EmbedCode`PHPCommon`"]

interpreterTypeToJavaType["Integer" | "Integer8" | "Integer16" | "Integer32" | Integer] = "Integer"
interpreterTypeToJavaType["Number"] = "Float"
interpreterTypeToJavaType["Date" | "DateTime"] = "date"
interpreterTypeToJavaType[_] = "String"

End[]
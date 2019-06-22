
BeginPackage["Compile`TypeSystem`Mangle`Mangle`"]

MangleFunction

Begin["`Private`"]



Needs["Compile`Core`IR`FunctionModule`"]
Needs["TypeFramework`"]
Needs["TypeFramework`TypeObjects`TypeConstructor`"]
Needs["TypeFramework`TypeObjects`TypeApplication`"]
Needs["TypeFramework`TypeObjects`TypeArrow`"]
Needs["CompileUtilities`Asserter`Assert`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)


(*
- https://github.com/numba/numba/blob/master/numba/itanium_mangler.py
- https://github.com/gchatelet/gcc_cpp_mangling_documentation/blob/master/README.md
- https://github.com/adobe/avmplus/blob/master/halfmoon/templates/mangle.py
- http://mearie.org/documents/mscmangle/
- https://en.wikiversity.org/wiki/Visual_C%2B%2B_name_mangling
- https://www.agner.org/optimize/calling_conventions.pdf
*)

 (* Itanium ABI spec says:
     * v	void
     * w	wchar_t
     * b	bool
     * c	char
     * a	signed char
     * h	unsigned char
     * s	short
     * t	unsigned short
     * i	int
     * j	unsigned int
     * l	long
     * m	unsigned long
     * x	long long, __int64
     * y	unsigned long long, __int64
     * n	__int128
     * o	unsigned __int128
     * f	float
     * d	double
     * e	long double, __float80
     * g	__float128
     * z	ellipsis
     * u <source-name>	# vendor extended type
 *)

itaniumConstructorMangleMap := itaniumConstructorMangleMap = <|
	"Void" -> "v",
	"VoidHandle" -> pointerOf["Itanium", "v"],
	"WideChar" -> "w",
	"Boolean" -> "b",
	"Integer8" -> "a",
	"UnsignedInteger8" -> "h",
	"Integer16" -> "s",
	"UnsignedInteger16" -> "t",
	"Integer32" -> "i",
	"UnsignedInteger32" -> "j",
	(*"Integer64" -> "l",
	"UnsignedInteger64" -> "m",*)
	"Integer64" -> "x",
	"UnsignedInteger64" -> "y",
	"Integer128" -> "n",
	"UnsignedInteger128" -> "o",
	"Real32" -> "f",
	"Real64" -> "d",
	"Real80" -> "e",
	"Real128" -> "g"
|>; 

(* MSVC ABI spec says:
	?	Type modifier, Template parameter
	$	Type modifier, Template parameter4	__w64 (prefix)
	0-9	Back reference
	A	Type modifier (reference)
	B	Type modifier (volatile reference)
	C	signed char
	D	char	__int8
	E	unsigned char	unsigned __int8
	F	short	__int16
	G	unsigned short	unsigned __int16
	H	int	__int32
	I	unsigned int	unsigned __int32
	J	long	__int64
	K	unsigned long	unsigned __int64
	L		__int128
	M	float	unsigned __int128
	N	double	bool
	O	long double	Array
	P	Type modifier (pointer)
	Q	Type modifier (const pointer)
	R	Type modifier (volatile pointer)
	S	Type modifier (const volatile pointer)
	T	Complex Type (union)
	U	Complex Type (struct)
	V	Complex Type (class)
	W	Enumerate Type (enum)	wchar_t
	X	void, Complex Type (coclass)	Complex Type (coclass)
	Y	Complex Type (cointerface)	Complex Type (cointerface)
	Z	... (elipsis)
*)

msvcConstructorMangleMap := msvcConstructorMangleMap = <|
	"Void" -> "X",
	"VoidHandle" -> pointerOf["MSVC", "X"],
	"WideChar" -> "_W",
	"Boolean" -> "_N",
	"Integer8" -> "_D",
	"UnsignedInteger8" -> "_E",
	"Integer16" -> "_F",
	"UnsignedInteger16" -> "_G",
	"Integer32" -> "_H",
	"UnsignedInteger32" -> "_I",
	"Integer64" -> "_J",
	"UnsignedInteger64" -> "_K",
	"Integer128" -> "_L",
	"UnsignedInteger128" -> "_M",
	"Real32" -> "M",
	"Real64" -> "N",
	"Real128" -> "O"
|>; 


mangle[m_, ty_?TypeConstructorQ] :=
	With[{
		name = ty["typename"],
		map = If[m === "Itanium", itaniumConstructorMangleMap, msvcConstructorMangleMap]
	},
		If[KeyExistsQ[map, name],
			Lookup[map, name],
			ThrowException[ {"The type specification for MangleFunction is not a C++ pod type.", m, ty}]
		]
	]


pointerOf["Itanium", s_] := "P" <> s
pointerOf["MSVC", s_] := "PEA" <> s

mangle[m_, ty_?TypeApplicationQ] :=
	Which[
		ty["type"]["sameQ", TypeSpecifier["CArray"]] || 
		ty["type"]["sameQ", TypeSpecifier["CUDABaseArray"]] || 
		ty["type"]["sameQ", Type["CArray"]] || 
		ty["type"]["sameQ", Type["CUDABaseArray"]] || 
		ty["isNamedApplication", "MIterator"] ||
		ty["type"]["isConstructor", "CArray"] || 
		ty["type"]["isConstructor", "Handle"],
			AssertThat[Length[ty["arguments"]] == 1];
			pointerOf[m, mangle[m, First[ty["arguments"]]]]
		,
		(*
		  Should look at the argument first
		*)
		ty["type"]["isConstructor", "Complex"],
			"St7complexI" <> mangle[m, First[ty["arguments"]]] <> "E"
		,
		True,
			ThrowException[{"Cannot mangle type ", m, ty}]
	]
	
mangle[m_, ty_?TypeArrowQ] := 
	StringJoin[Flatten[{
		If[m === "MSVC",
			mangle[m, ty["result"]],
			""
		],
		mangle[m, #]& /@ ty["arguments"]
	}]]

encodeName["Itanium", name_] := StringJoin[ToString[StringLength[name]], name]
encodeName["MSVC", name_] := StringJoin[name, "@@YA"]

prefix["Itanium"] = "_Z"
prefix["MSVC"] = "?"

suffix["Itanium"] = ""
suffix["MSVC"] = "@Z"

mangle[m_, name_, ty_] := 
	StringJoin[
		prefix[m],
		encodeName[m, name],
		mangle[m, ty],
		suffix[m]
	]
	
getMethod[opts_] :=
	Switch[Lookup[opts, Method, Automatic],
		Automatic,
			If[$OperatingSystem === "Windows",
				"MSVC",
				"Itanium"
			],
		"Itanium",
			"Itanium",
		"MSVC",
			"MSVC",
		_,
			ThrowException[{"Invalid mangling method ", opts}]
	]
	
Options[MangleFunction] = {
	Method -> Automatic
}
MangleFunction[name_, ty_, opts:OptionsPattern[]] :=
	MangleFunction[name, ty, Association[opts]]	

MangleFunction[name_, ty_, opts_?AssociationQ] :=
	mangle[getMethod[opts], name, ty]
	
End[]

EndPackage[]



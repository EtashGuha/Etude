(* ::Package:: *)

(**********************************************************************
BinarySearch Functions
**********************************************************************)

Package["NumericArrayUtilities`"]
PackageImport["GeneralUtilities`"]

(*********************************************************************)
PackageExport["SetMantissaPrecision"]

SetUsage[SetMantissaPrecision, "
SetMantissaPrecision[x$, p$, b$] \
reduces mantissa of any element of the array x$ to p$ base b$ digits. \
It is used to make the array x$ more compressible. \
Possible values of p$: 53 (double precision), 24 (single precision), \
11 (half precision). The default value for the base b$ is 2.

Options:
- Parallelization: If True, allow parallelization
"]

DeclareLibraryFunction[mTensorMantissaPrecision,
	"mantissa_precision_MTensor",
	{
		{Real, _, "Constant"},
                Integer,
                Integer,
                "Boolean"
	},
	{Real, _}
]

DeclareLibraryFunction[mNumericArrayMantissaPrecision,
	"mantissa_precision_MNumericArray",
	{
		{"NumericArray", "Constant"}, 
                Integer,
                Integer,
                "Boolean"
	},
	{"NumericArray"}
]

Options[SetMantissaPrecision] = {
  Parallelization -> False
}

SetMantissaPrecision[list_, prec_Integer, base_Integer:2,
                     opts:OptionsPattern[]] :=
Scope[
        runOMP = OptionValue[Parallelization];
        If [Length[list] == 0,
            {}
          , (* else *)
	    If [Head[list] === NumericArray,
		mNumericArrayMantissaPrecision[list, prec, base, runOMP]
	      ,
		mTensorMantissaPrecision[list, prec, base, runOMP]
	    ]
        ]
]

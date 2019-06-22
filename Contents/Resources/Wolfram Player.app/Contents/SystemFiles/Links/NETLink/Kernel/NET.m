(* :Title: NET.m *)

(* :Author:
        Todd Gayley
        tgayley@wolfram.com
*)

(* :Package Version: 1.7 *)

(* :Mathematica Version: 5.0 *)
		     
(* :Copyright: .NET/Link source code (c) 2003-2019, Wolfram Research, Inc. All rights reserved.

   Use is governed by the terms of the .NET/Link license agreement, which can be found at
   www.wolfram.com/solutions/mathlink/netlink.
*)

(* :Discussion:
	
   This file is a component of the .NET/Link Mathematica source code.
   It is not a public API, and should never be loaded directly by users or programmers.

   .NET/Link uses a special system wherein one package context (NETLink`) has its implementation
   split among a number of .m files. Each component file has its own private context, and also
   potentially introduces public symbols (in the NETLink` context) and so-called "package" symbols,
   where the term "package" comes from Java terminology, referring to symbols that are visible
   everywhere within the implementation of .NET/Link, but not to clients. The NETLink.m file itself
   is produced by an automated tool from the component files and contains only declarations.
   
   Do not modify the special comment markers that delimit Public- and Package-level exports.
*)


(*<!--Public From NET.m


-->*)


(*<!--Package From NET.m

// These n* functions are the ones whose definitions are created by Install.cs.
// They implement the special set of CallPackets used to call into .NET.
nCall
nLoadType1
nLoadType2
nLoadExistingType
nLoadAssembly
nLoadAssemblyFromDir
nGetAssemblyObject
nGetTypeObject
nReleaseObject
nMakeObject
nCreateDelegate
nVal
nReflectType
nReflectAsm
nSetComplex
nInstanceOf
nCast
nSameQ
nPeekTypes
nPeekObjects
nPeekAssemblies
nCreateDLL1
nCreateDLL2
nDefineDelegate
nDlgTypeName
nAddHandler
nRemoveHandler
nModal
nShow
nShareKernel
nAllowUIComputations
nIsCOMProp
nCreateCOM
nGetActiveCOM
nReleaseCOM
nLoadTypeLibrary

nGetException

nConnectToFEServer
nDisconnectToFEServer

nUILink

noop
noop2

-->*)


(* Current context will be NETLink`. *)

Begin["`NET`Private`"]

(* Because these symbols are Cleared and re-defined in UninstallNET/ReinstallNET, if we leave tracking
   on for them, then every call to UninstallNET or ReinstallNET will trigger a re-fire of all Dynamics
   that have a call to .NET in them.
*)
Internal`SetValueNoTrack[#, True]& /@ {
	nCall,
	nLoadType1,
	nLoadType2,
	nLoadExistingType,
	nLoadAssembly,
	nLoadAssemblyFromDir,
	nGetAssemblyObject,
	nGetTypeObject,
	nReleaseObject,
	nMakeObject,
	nCreateDelegate,
	nVal,
	nReflectType,
	nReflectAsm,
	nSetComplex,
	nInstanceOf,
	nCast,
	nSameQ,
	nPeekTypes,
	nPeekObjects,
	nPeekAssemblies,
	nCreateDLL1,
	nCreateDLL2,
	nDefineDelegate,
	nDlgTypeName,
	nAddHandler,
	nRemoveHandler,
	nModal,
	nShow,
	nShareKernel,
	nAllowUIComputations,
	nIsCOMProp,
	nCreateCOM,
	nGetActiveCOM,
	nReleaseCOM,
	nLoadTypeLibrary,
	nGetException,
	nConnectToFEServer,
	nDisconnectToFEServer,
	nUILink,
	noop,
	noop2
	}

End[]

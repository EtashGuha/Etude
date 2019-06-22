(* :Title: CallNET.m *)

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


(*<!--Public From CallNET.m

LoadNETAssembly::usage =
"LoadNETAssembly[assemblySpec] loads the specified assembly into the .NET runtime and returns a NETAssembly \
expression that can be used to identify the assembly. You can call LoadNETAssembly more than once on the same \
assembly--if it has already been loaded then LoadNETAssembly will return quickly. The assemblySpec argument can \
be a simple name like \"System.Web\", a full name like \
\"System.Web, Version=1.0.5000.0, Culture=neutral, PublicKeyToken=b03f5f7f11d50a3a\", or a path or URL \
to the assembly file itself. LoadNETAssembly[\"directory\"] loads all the assemblies in the given directory and returns \
a list of NETAssembly expressions. LoadNETAssembly[\"ApplicationContext`\"] loads all the assemblies in the \"assembly\" \
subdirectory of the main application directory corresponding to the given context. \
LoadNETAssembly[\"assemblyName\", \"directory\"] loads the named assembly from the given directory, if possible. \
LoadNETAssembly[\"assemblyName\", \"ApplicationContext`\"] loads the named assembly from the \"assembly\" \
subdirectory of the main application directory corresponding to the given context, if possible."

LoadNETType::usage =
"LoadNETType[\"typeName\"] loads the specified type into the .NET runtime and returns a NETType expression that can be \
used to identify the type. You can load any of the types defined in .NET: classes, interfaces, structs (value types), \
enumerations, and delegates. The assembly in which the type is defined must have previously been loaded using LoadNETAssembly. \
LoadNETType[\"typeName\", assemblySpec] loads the type from the given assembly. The assemblySpec argument can be an \
assembly name, a NETAssembly expression, a .NET Assembly object, or a path or URL to an assembly file. If it is an assembly \
name, the assembly must already have been loaded."

NETNew::usage =
"NETNew[nettype, args] constructs a new object of the specified .NET type, passing the supplied argument sequence to the constructor. \
The nettype argument can be either a NETType expression that was returned from LoadNETType, or a string giving the type's name. \
The assembly in which the type resides must have been loaded with LoadNETAssembly. NETNew[{\"typeName\", assemblySpec}, args] constructs \
the object from the named type in the specified assembly. The assemblySpec argument can be an assembly name, a NETAssembly \
expression returned from LoadNETAssembly, or a path or URL to an assembly file. The assembly will be loaded if necessary. \
NETNew[{\"typeName\", \"assemblyName\", \"dir\"}, args] uses the named assembly from the specified directory, if possible. \
NETNew[{\"typeName\", \"assemblyName\", \"AppContext`\"}, args] uses the named assembly from the \"assembly\" subdirectory \
of the main application directory corresponding to the given context, if possible."

NETType::usage =
"NETType[\"typeName\", n] represents a .NET type with the specified name. The second argument is an integer index that is not \
relevant to users. NETType expressions cannot be typed in by the user; they are returned by LoadNETType."

NETAssembly::usage =
"NETAssembly[\"asmName\", n] represents a .NET assembly with the specified name. The second argument is an integer index that is not \
relevant to users. NETAssembly expressions can be used in LoadNETType to specify the assembly from which you want to load the type. \
NETAssembly expressions cannot be typed in by the user; they are returned by LoadNETAssembly."

GetAssemblyObject::usage =
"GetAssemblyObject[asm_NETAssembly] returns the .NET Assembly object corresponding to the specified NETAssembly expression. \
This is a rarely-used method provided for programmers who have a need to obtain an actual .NET Assembly object corresponding to a \
loaded assembly."

GetTypeObject::usage =
"GetTypeObject[type_NETType] returns the .NET Type object corresponding to the specified NETType expression. \
This is a rarely-used method provided for programmers who have a need to obtain an actual .NET Type object corresponding to a \
loaded type."

NETObject::usage =
"NETObject is used to denote an expression that refers to an object residing in the .NET runtime."

NETObjectQ::usage =
"NETObjectQ[expr] gives True if expr is a valid reference to a .NET object, and it gives False otherwise."

CastNETObject::usage = 
"CastNETObject[obj, type] casts the specified object to a different type. The cast must be valid, meaning that \
the object must be an instance of the given class or interface type. The type can be specified by a fully-qualified \
type name or as a NETType expression. CastNETObject is rarely needed. There are two main situations where it is used. \
The first case is where you need to \"upcast\" an object to call an inherited version of a method that is hidden \
by a version of the same method declared as \"new\" in a class lower in the inheritance hierarchy. This rare \
situation is discussed in the .NET/Link User Guide. The second case is where you have a \
\"raw\" COM object (these appear as <<NETObject[System.__ComObject]>> or <<NETObject[COMInterface[...]]>>) but \
you know that it can be successfully cast to a certain managed type. It is generally more convenient to work with \
managed types than raw COM objects."

ReturnAsNETObject::usage =
"ReturnAsNETObject[expr] causes a .NET call during the evaluation of expr to return its result as \
an object reference (i.e., a NETObject expression), not a value. Most .NET objects are returned as references, \
but those that have a meaningful Wolfram Language representation are returned \"by value\". Such objects include strings, arrays, \
and so-called \"boxed\" values like System.Int32. ReturnAsNETObject overrides the normal behavior and forces any object returned \
to the Wolfram Language to be sent only as a reference. ReturnAsNETObject is typically used to avoid needlessly sending large arrays of numbers back \
and forth between .NET and the Wolfram Language. You can use ReturnAsNETObject to cause only a reference to be sent; then you can use the \
NETObjectToExpression function at the end if the final value is needed."

NETObjectToExpression::usage =
"NETObjectToExpression[netObject] converts the specified .NET object reference into its value as a \"native\" Wolfram Language \
expression. Most .NET objects that have a meaningful \"by value\" representation in the Wolfram Language are returned by \
value to the Wolfram Language automatically. Such objects include strings, arrays (which become lists), and so-called \"boxed\" values \
like System.Int32. However, you can get a reference form of one of these types if you explicitly call NETNew or use the \
ReturnAsNETObject function. In such cases, you can use NETObjectToExpression to retrieve the value. NETObjectToExpression \
also converts into values some types that are normally sent by reference. This includes converting enum objects to their \
integer values and collections into lists. NETObjectToExpression has no effect on object references that have no meaningful \
\"by value\" representation in the Wolfram Language."

AddNETObjectListener::usage =
"AddNETObjectListener[func] adds the given function to the set of active listeners. Your listener function will be called every \
time a .NET object is created and released in the Wolfram Language. The function will be called as follows: listener[object, typeName, action], \
where 'action' is either \"Create\" or \"Release\"."

RemoveNETObjectListener::usage =
"RemoveNETObjectListener[func] removes the given function from the set of active listeners."

NETObjectListeners::usage = 
"NETObjectListeners[] returns the currently active list of listeners managed by the AddNETObjectListener and RemoveNETObjectListener functions."

NETObjectSummaryBox::usage =
"NETObjectSummaryBox[obj, typeName, isCOMObject] is called to produce a Graphics object representing a summary box for a .NET object. \
You can define your own rules for this function to produce custmom summary boxes for .NET objects of specified types. If isCOMObject \
is true, the typeName argument will be the COM interface name, if available."

-->*)


(*<!--Package From CallNET.m

// Used in netlinkExternalCall to direct output to the appropriate link (NETLink[] or NETLink[]).
getActiveNETLink
$inExternalCall

clearNETDefs
callAllUnloadTypeMethods

// These are called directly from .NET.
netlinkDefineExternal
loadTypeFromNET
createInstanceDefs
hasAliasedVersions

outParam

argTypeToInteger

getAQTypeName
aqTypeNameFromStaticSymbol
getFullAsmName

toLegalName

runListeners

-->*)


(* Current context will be NETLink`. *)

Begin["`CallNET`Private`"]


(***************************************  General Messages  *******************************************)

NET::obj = "Attempting to use invalid .NET object: `1`."
NET::staticfield = "Attempting to use invalid .NET static field: `1`."
NET::staticprop = "Attempting to use invalid .NET static property: `1`."
NET::lval = "Invalid attempt to assign a value to the result of a .NET method call. If you are trying to assign to a property or field, the correct syntax is 'obj@FieldOrPropertyName = val' or 'obj@ParameterizedPropertyName[index] = val'."
NET::outparam = "Method `1` takes a \"by-reference\" parameter at position `2` (\"out\" or \"ref\" in C# terminology, \"ByRef\" in Visual Basic). You must pass a symbol as an argument at that position if you want an assignment to be made to the byref parameter. The return value of this method is unaffected by this warning."


(*****************************************  LoadNETAssembly  ******************************************)

LoadNETAssembly::noload = "The assembly `1` either was not found or could not be loaded."
LoadNETAssembly::noload2 = "The assembly `1` either was not found in the directory `2`, or it could not be loaded."
LoadNETAssembly::ctxt = "The application directory corresponding to the context `1` could not be found. No assemblies were loaded."
LoadNETAssembly::asmdir = "The application directory corresponding to the context `1` does not have the required subdirectory named \"assembly\". No assemblies can be loaded from this context."
LoadNETAssembly::arg2 = "The second argument must be an application context or valid directory in which to search for the assembly."

(**
    LoadNETAssembly loads the specified assembly into the .NET runtime and returns a NETAssembly expression
    that can be used throughout .NET/Link to identify the assembly. The assembly may already have been loaded--it
    is perfectly safe to call LoadNETAssembly many times for the same assembly. In fact, LoadNETAssembly is not
    so much a way to load an assembly, but rather a way to convert a way of describing an assembly (like a name or path)
    into a NETAssembly expression.
    
    LoadNETAssembly can be called with many different types of arguments. Here is the complete set:
    
        ** = assembly must be already loaded, or in GAC

        ** LoadNETAssembly["Assembly.Name"]
        ** LoadNETAssembly["Assembly.Name, Version=.. etc"]
           LoadNETAssembly["path/to/assemblyfile"]
           LoadNETAssembly["http://url/to/assemblyfile"]
           LoadNETAssembly[assemblyObject]   (* Not all that useful, as assembly is already loaded. *)
           LoadNETAssembly["Assembly.Name", "path/to/dir"]
           LoadNETAssembly["Assembly.Name", "AppContext`"]
           LoadNETAssembly["AssemblyName.dll", "path/to/dir"]
           LoadNETAssembly["AssemblyName.dll", "AppContext`"]

          (* Load a batch of assemblies: *)
          LoadNETAssembly["path/to/dir"]
          LoadNETAssembly["AppContext`"]
    
   Returns a NETAssembly expression, or a list of these for the batch versions.
**)

(* Block is used for speed only in these definitions. *)

LoadNETAssembly[pathOrContextOrAssemblyName_String] := 
    loadNETAssembly[pathOrContextOrAssemblyName, False]

(* This form isn't very useful, as the assembly is already loaded into .NET. But you might just want to get
   a NETAssembly expression out of it. Plus, it's useful for completeness. Users might expect it to work.
*)
LoadNETAssembly[assemblyObj_?NETObjectQ] :=
    If[InstanceOf[assemblyObj, "System.Reflection.Assembly"],
        LoadNETAssembly[assemblyObj@FullName],
    (* else *)
        Message[LoadNETAssembly::asmobj, assemblyObj];
        $Failed
    ]

LoadNETAssembly[assemblyName_String, dirOrContext_String] := 
    Block[{lookedUp, isContext, isDir, asmDir, result},
        (* See if the assembly has already been loaded by this exact name and dirOrContext. *)
        lookedUp = lookupAssembly[assemblyName, dirOrContext];
        If[Head[lookedUp] === NETAssembly,
            Return[lookedUp]
        ];
        isContext = StringMatchQ[dirOrContext, "*`"];
        isDir = !isContext && FileType[dirOrContext] === Directory;
        If[!isContext && !isDir,
            Message[LoadNETAssembly::arg2];
            Return[$Failed]
        ];
        If[isContext,
            (* User can specify a context (e.g., MyApp`) as the argument. We search for the assembly
                in that app's assembly directory.
            *)
            asmDir = findAppDir[dirOrContext];
            If[asmDir === $Failed,
                Message[LoadNETAssembly::ctxt, dirOrContext];
                Return[$Failed]
            ];
            asmDir = ToFileName[asmDir, "assembly"];
            If[FileType[asmDir] =!= Directory,
                Message[LoadNETAssembly::asmdir, dirOrContext];
                Return[$Failed]
            ],
        (* else *)
            (* Is directory. *)
            asmDir = dirOrContext
        ];
        (* At this point, we know we have a legit directory. *)
        If[StringMatchQ[assemblyName, "*.dll", IgnoreCase->True] || StringMatchQ[assemblyName, "*.exe", IgnoreCase->True],
            (* If asm was specified as a DLL name like Assembly.Name.dll, just build a full filename and load it that way. *)
            LoadNETAssembly[ToFileName[asmDir, assemblyName]],
        (* else *)
            (* Assembly was specified as a short or long name. We first load all the assemblies in the directory. *)
            LoadNETAssembly[asmDir];
            result = nLoadAssemblyFromDir[assemblyName, asmDir];
            If[ListQ[result],
                lookupAssembly[assemblyName, dirOrContext] = makeNETAssembly @@ result,
            (* else *)
                Message[LoadNETAssembly::noload2, assemblyName, asmDir];
                $Failed
            ]
        ]
    ]
    

(* Private worker function. The suppressErrors argument only affects the BadImageFormatException you get when
   a non-managed DLL or EXE file is loaded.
*)
loadNETAssembly[pathOrContextOrAssemblyName_String, suppressErrors:(True | False)] := 
    Block[{isContext, isDLLName, isDir, appDir, appAsms, dirAsms, lookedUp, result, wasOn},
        (* First see if the assembly has already been loaded by this exact name. This will only succeed
           if pathOrContextOrAssemblyName is a reference to a single assembly, not a context or dir.
        *)
        lookedUp = lookupAssembly[pathOrContextOrAssemblyName];
        If[Head[lookedUp] === NETAssembly,
            Return[lookedUp]
        ];
        isContext = StringMatchQ[pathOrContextOrAssemblyName, "*`"];
        isDLLName = (StringMatchQ[pathOrContextOrAssemblyName, "*.dll"] || StringMatchQ[pathOrContextOrAssemblyName, "*.exe"]) &&
                        !StringMatchQ[pathOrContextOrAssemblyName, "*\\*"] && !StringMatchQ[pathOrContextOrAssemblyName, "*/*"];
        isDir = !isContext && !isDLLName && FileType[pathOrContextOrAssemblyName] === Directory;
        Which[
            isContext,
                (* User can specify a context (e.g., MyApp`) as the argument. We search for and load all assemblies
                   in that app's assembly directory.
                *)
                appDir = findAppDir[pathOrContextOrAssemblyName];
                If[appDir === $Failed,
                    Message[LoadNETAssembly::ctxt, pathOrContextOrAssemblyName];
                    Return[{}],
                (* else *)
                    Return[DeleteCases[LoadNETAssembly /@ FileNames[{"*.dll", "*.exe"}, ToFileName[appDir, "assembly"]], $Failed]]
                ],
            isDir,
                dirAsms = FileNames[{"*.dll", "*.exe"}, pathOrContextOrAssemblyName];
                (* For the case of loading all assemblies from a dir, turn off the failure message
                   if an assembly fails to load. This prevents seeing errors from non-.NET .dll and .exe
                   files in the dir. This is the one case where the presence on non-assembly files in the
                   set of files to load is probably not a user error. The True 2nd arg to nLoadAssembly
                   prevents a .NET exception and error message from appearing.
                *)
                wasOn = Head[LoadNETAssembly::noload] =!= $Off;
                Off[LoadNETAssembly::noload];
                result = DeleteCases[loadNETAssembly[#, True]& /@ dirAsms, $Failed];
                If[wasOn, On[LoadNETAssembly::noload]];
                Return[result]
        ];
        (* We get here if pathOrContextOrAssemblyName was NOT a context or directory AND it has not been loaded
           by this exact name before. It could be either:
           - a simple or long asm name: Assembly.Name   or   Assembly.Name,Version=...
           - a URL
           - a DLL or EXE name without path info: Assembly.Name.dll (we'll search among loaded asms for that filename and then in GAC)
        *)
        InstallNET[];
        result = nLoadAssembly[pathOrContextOrAssemblyName, suppressErrors];
        If[ListQ[result],
            lookupAssembly[pathOrContextOrAssemblyName] = makeNETAssembly @@ result,
        (* else *)
            Message[LoadNETAssembly::noload, pathOrContextOrAssemblyName];
            $Failed
        ]
    ]


(* Returns the top application directory corresponding to the given context, or $Failed if it cannot be found. 
   Because it relies on the System`Private`FindFile function, it ues the standard search algorithm for app dirs,
   in the correct order of priority.
*)
findAppDir[context_String] :=
    Module[{appFile, appDir, appPaths},
        appFile = System`Private`FindFile[context];
        If[StringQ[appFile],
            appDir = DirectoryName[appFile];
            (* If FindFile found a MyApp/Kernel/init.m file, we need to go up one more level to the true app dir. *)
            If[StringMatchQ[appDir, __ ~~ $PathnameSeparator ~~ "Kernel" ~~ $PathnameSeparator],
                appDir = DirectoryName[appDir]
            ];
            appDir,
        (* else *)
            (* FindFile failed. FindFile looks for a .m file, so it will fail if the user has created
               a dummy app dir just to hold assemblies. If it fails we use a manual search.
            *)
            appPaths = {ToFileName[{$TopDirectory, "AddOns", "Applications"}], ToFileName[{$TopDirectory, "AddOns", "ExtraPackages"}]};
            If[StringQ[$AddOnsDirectory],
                (* This branch is for 4.2 and later. *)
                PrependTo[appPaths, ToFileName[{$AddOnsDirectory, "Applications"}]]
            ];
            If[StringQ[$UserAddOnsDirectory],
                (* 4.2 and later *)
                PrependTo[appPaths, ToFileName[{$UserAddOnsDirectory, "Applications"}]],
            (* else *)
                PrependTo[appPaths, ToFileName[{$PreferencesDirectory, "AddOns", "Applications"}]]
            ];
            appDir = Select[FileNames["*", appPaths], (FileType[#] === Directory && StringMatchQ[#, "*" <> StringDrop[context, -1]])&];
            If[Length[appDir] > 0,
                First[appDir],
            (* else *)
                $Failed
            ]
        ]
    ]

    
(***************************************  GetAssemblyObject  ******************************************)

(* GetAssemblyObject does exactly one thing--it takes a NETAssembly expression and returns an Assembly object for that type.
   Because .NET/Link uses a moniker (NETAssembly) instead of actual Assembly objects, it is important to have a method
   to get the actual corresponding Assembly object from it.
*)

GetAssemblyObject::arg = "The argument must be a NETAssembly expression returned from LoadNETAssembly."
GetAssemblyObject::asm = "The argument is not a valid NETAssembly expression. Perhaps the .NET runtime has been restarted since LoadNETAssembly was called."

GetAssemblyObject[asm_NETAssembly] :=
    Module[{asmName, obj},
        asmName = getFullAsmName[asm];
        If[!StringQ[asmName],
            Message[GetAssemblyObject::asm];
            Return[$Failed]
        ];
        (* If getFullAsmName returns a string, it will always be correct and valid for the current .NET session. *)
        nGetAssemblyObject[asmName]
    ]

GetAssemblyObject[x_] :=
    (
        Message[GetAssemblyObject::arg];
        $Failed
    )
    
    
(*********************************************  LoadNETType  **********************************************)

(*  LoadNETType loads the specified type into the .NET runtime and returns a NETType expression
    that can be used throughout .NET/Link to identify the type. The type may already have been loaded--it
    is perfectly safe to call LoadNETType many times for the same type.
    
    You can load any .NET type: classes, interfaces, structs, enums, delegates.
        
    LoadNETType accepts the following argument sequences:

        *  = assembly must be already loaded
        ** = assembly must be already loaded, or in GAC

        *  LoadNETType["Type.Name"]
        ** LoadNETType["Type.Name,AssemblyName"]
        ** LoadNETType["Type.Name", "Assembly.Name"]
        *  LoadNETType["Type.Name", "AssemblyName.dll"]
        *  LoadNETType["Type.Name", _NETAssembly]
           LoadNETType["Type.Name", assemblyObj]
           LoadNETType["Type.Name", "http://assemblyURL"]
           LoadNETType["Type.Name", "path/to/assemblyFile"]
           LoadNETType["Type.Name", "Assembly.Name", "path/to/assemblyDir"]
           LoadNETType["Type.Name", "Assembly.Name", "AppContext`"]
           LoadNETType["Type.Name", "AssemblyName.dll", "path/to/assemblyDir"]
           LoadNETType["Type.Name", "AssemblyName.dll", "AppContext`"]
        
           LoadNETType[typeObj]    -- For when you have created a type dynamically and want to load it into M.
        
*)

LoadNETType::args = "Improper count or type of arguments."
LoadNETType::typeobj = "Object `1` is not an instance of the Type class."
LoadNETType::asmobj = "Object `1` is not an instance of the Assembly class."
LoadNETType::asm = "Type `1` could not be loaded from the specified assembly because that assembly could not be found."
LoadNETType::fail = ".NET failed to load type `1`."

(* The symbols for these two options are defined in JLink`, but we use context-independent option processing. *)
Options[LoadNETType] = {StaticsVisible->False, AllowShortContext->True}


LoadNETType[name_String, opts___?OptionQ] := loadNETType[{name, Null}, {"", "", Null, Null}, opts]
LoadNETType[name_String, assemblyName_String, opts___?OptionQ] := loadNETType[{name, Null}, {assemblyName, "", Null, Null}, opts]
LoadNETType[name_String, netAssembly_NETAssembly, opts___?OptionQ] := loadNETType[{name, Null}, {"", "", netAssembly, Null}, opts]
LoadNETType[name_String, assemblyObj_?NETObjectQ, opts___?OptionQ] := loadNETType[{name, Null}, {"", "", Null, assemblyObj}, opts]

LoadNETType[name_String, assemblyName_String, dirOrContext_String, opts___?OptionQ] :=
    loadNETType[{name, Null}, {assemblyName, dirOrContext, Null, Null}, opts]

(* This form is for when you have obtained a Type object through some means (e.g. a dynamic assembly) and you want to
   load it into .NET/Link. The type already exists in the .NET runtime, but you want to make it available via .NET/Link.
*)
LoadNETType[typeObject_?NETObjectQ, opts___?OptionQ] :=
    If[InstanceOf[typeObject, "System.Type"],
        loadNETType[{"", typeObject}, {"", "", Null, Null}, opts],
    (* else *)
        Message[LoadNETType::typeobj, typeObject];
        $Failed
    ]

LoadNETType[___] := (Message[LoadNETType::args]; $Failed)


(* The worker function called by LoadNETType. The type will be sepcified by either its name or by a Type object.
   The assembly to load from will be specified by one of assemblyName, netAssembly, or assemblyObj (or none of these
   if no assembly information is explicitly supplied). The assembly spec is ignored if we are loading via a Type object.
   
   Block is used here for speed reasons only. 
*)
loadNETType[{name_String, typeObj_}, {assemblyName_String, dirOrContext_String, netAssembly_, assemblyObj_}, opts___?OptionQ] :=
    Block[{fixedName, lookedUp, netAsm, asmName, systemAsmName, assemblyCommaPos, nRes, typeName, aqTypeName, namespace,
           staticFields, staticProps, staticMethods, staticEvents, nonPrimitiveFlds, hasIndxr,
           wasOn1, wasOn2, wasOn3, allowShortCtxt, staticsVisible, ctxt, shortCtxt, result},
                
        InstallNET[];
        
        fixedName = fixGenericTypeName[name];
        
        Which[
            typeObj =!= Null,
                (* Load a preexisting Type, such as one created in a dynamic assembly. *)
                If[!InstanceOf[typeObj, "System.Type"],
                    Message[LoadNETType::typeobj, typeObj];
                    Return[$Failed]
                ];
                (* See if the type has already been loaded by this Type obj. *)
                lookedUp = lookupType[typeObj@AssemblyQualifiedName];
                If[Head[lookedUp] === NETType,
                    Return[lookedUp]
                ];
                nRes = nLoadExistingType[typeObj],
            assemblyObj =!= Null,
                (* Assembly to load from passed in as an object. *)
                If[!InstanceOf[assemblyObj, "System.Reflection.Assembly"],
                    Message[LoadNETType::asmobj, assemblyObj];
                    Return[$Failed]
                ];
                (* See if the type has already been loaded by this exact name/assembly. *)
                lookedUp = lookupType[fixedName, assemblyObj@FullName];
                If[Head[lookedUp] === NETType,
                    Return[lookedUp]
                ];
                nRes = nLoadType2[fixedName, assemblyObj],
            True,
                (* Assembly name not supplied separately, or supplied as a string or as a NETAssembly or as
                   a path or URL to the file. Use LoadNETAssembly to get a NETAssembly from whatever the user
                   supplied as the assembly specification. If there was no assembly spec at all (assemblyName == ""),
                   we cannot use LoadNETAssembly. That's OK, because the type name must then be assembly-qualified.
                *)
                netAsm = netAssembly;
                Which[
					(* It is an AQ type name if it has a comma snd it doesn't end in a ]. *)
                    StringMatchQ[fixedName, "*,*"] && !StringMatchQ[fixedName, "*]"],
                        (* If the type name is assembly-qualified, call LoadNETAssembly on the assembly
                           name in case it hasn't already been loaded. Array type names also have commas,
                           so the test above ensures we don't enter this branch on an array type name foo[,]
                           or generic type name Foo`1[System.String,System.String] that has no assembly spec,
                           and the test below ensures we are looking at the comma between the type name and assembly name.
                        *)
                        assemblyCommaPos = 
							If[StringMatchQ[fixedName, "*],*"],
								Last[Flatten[StringPosition[fixedName, "],"]]],
							(* else *)
								First[Flatten[StringPosition[fixedName, ","]]]
							];
                        (* Need to drop the spaces, if present, after the comma before the assembly name begins. *)
                        While[StringTake[fixedName, {assemblyCommaPos + 1}] == " ", assemblyCommaPos++];
                        (* Note that we don't assign to netAsm, just load the assembly. Because the type name is
                           assembly-qualified, we leave the assembly name passed into .NET as just "".
                        *)
                        LoadNETAssembly[StringDrop[fixedName, assemblyCommaPos]],
                    Head[netAsm] =!= NETAssembly,
                        (* Assembly was not specified by a NETAssembly. Try to obtain one and assign it to netAsm. *)
                        If[assemblyName != "",
                            (* Assembly name was supplied as a separate string. *)
                            netAsm =
                                If[dirOrContext === "",
                                    LoadNETAssembly[assemblyName],
                                (* else *)
                                    LoadNETAssembly[assemblyName, dirOrContext]
                                ];
                            (* If we cannot satisfy the request for that specific assembly, fail right away. *)
                            If[Head[netAsm] =!= NETAssembly,
                                Message[LoadNETType::asm, name];
                                Return[$Failed]
                            ],
                        (* else *)
                            (* No assembly spec was provided, so see if the type is a System.* type and if we
                               can auto-load its assembly.
                            *)
                            If[StringMatchQ[fixedName, "System.*"],
                                systemAsmName = typeNameToSystemAssemblyName[fixedName];
                                If[StringQ[systemAsmName],
                                    netAsm = LoadNETAssembly[systemAsmName]
                                ]
                            ]
                        ]
                ];
                asmName = If[Head[netAsm] === NETAssembly, getFullAsmName[netAsm], ""];
                (* See if the type has already been loaded by this exact name/assembly. *)
                lookedUp = lookupType[fixedName, asmName];
                If[Head[lookedUp] === NETType,
                    Return[lookedUp]
                ];
                
                nRes = nLoadType1[fixedName, asmName]
        ];
                
        If[nRes === $Failed,
			Message[LoadNETType::fail, name];
            Return[$Failed]
        ];
        
        {allowShortCtxt, staticsVisible} =
                contextIndependentOptions[{AllowShortContext, StaticsVisible}, Flatten[{opts}], Options[LoadNETType]];

        AbortProtect[
            If[Length[nRes] == 2,
                (* If type has already been loaded (i.e. with a slightly different partial assembly name),
                   nLoadType will return just the {typeName, aqTypeName} of the type.
                *)
                {typeName, aqTypeName} = nRes,
            (* else *)
                (* nRes looks like:
                
                      {typeName, aqTypeName, staticFields, staticProps, staticMethods, staticEvents, nonPrimitiveFlds, hasIndxr}
                    
                   where:
                   
                      typeName_String            is the short readable type name: "System.Windows.Forms.Button"
                      aqTypeName_String          is the full assembly-qualified type name (a complete description of the type)
                      namespace_String           
                      staticFields:{___String}
                      staticProps:{{name_String, isParameterized:(True | False)}...}
                      staticMethods:{___String}
                      staticEvents:{___String}
                      nonPrimitiveFlds:{___String} list of all fields/props that are reference types (things that would come
                                                   into M as NETObjects).
                      hasIndxr:(True | False)    tells whether the class has an indexer in the C# sense
                                                 (a default paramaterized prop in the VB sense)
                *)
                {typeName, aqTypeName, namespace, staticFields, staticProps, staticMethods, staticEvents, nonPrimitiveFlds, hasIndxr} = nRes;

                hasIndexer[aqTypeName] = hasIndxr;
                (isNonPrimitiveFieldOrSimpleProp[aqTypeName, #] = True)& /@ nonPrimitiveFlds;
                
                (* Arrays are treated specially in toContextName (as are generics). We can't make a legal context name
                   out of an array type name like foo[], so we just put all statics for arrays into System`Array`,
                   which is their parent class. We also want to avoid doing this more than once (no real problem in
                   doing that, it just wastes time) so we check if it has been done already.
                *)
                ctxt = toLegalName[toContextName[typeName]];
                If[ctxt == "System`Array`", namespace = "System"];                
                If[ctxt != "System`Array`" || !MemberQ[$netContexts, "System`Array`"],
                    AppendTo[$netContexts, ctxt];
                    If[TrueQ[allowShortCtxt] && StringLength[namespace] > 0,
                        shortCtxt = shortClassContextFromClassContext[ctxt, namespace];
                        AppendTo[$netContexts, shortCtxt],
                    (* else *)
                        shortCtxt = ctxt
                    ];
                    
                    {wasOn1, wasOn2, wasOn3} = (Head[#] =!= $Off &) /@ {General::shdw, General::spell, General::spell1};
                    Off[General::shdw];
                    Off[General::spell];
                    Off[General::spell1];
            
                    (* Lahey Fortran creates public methods (but not meant to be called by users) that start
                       with a period. Trying to create defs for these methods would result in many
                       error messages. We could exand the following test to reject other non-legal members.
                    *)
                    staticMethods = Select[staticMethods, !StringMatchQ[#, ".*"]&];

                    createStaticFieldOrPropertyDef[aqTypeName, ctxt, shortCtxt, #, False]& /@ staticFields;
                    createStaticFieldOrPropertyDef[aqTypeName, ctxt, shortCtxt, #[[1]], #[[2]]]& /@ Union[staticProps];
                    createStaticMethodDef[aqTypeName, ctxt, shortCtxt, #]& /@ Union[staticMethods];
                    createStaticEventDef[aqTypeName, ctxt, shortCtxt, #]& /@ Union[staticEvents];
                    
                    If[staticsVisible,
                        BeginPackage[ctxt];
                        EndPackage[]
                    ];

                    If[wasOn1, On[General::shdw]];
                    If[wasOn2, On[General::spell]];
                    If[wasOn3, On[General::spell1]]
                ]
            ];
            
            result = makeNETType[typeName, aqTypeName];

            (* Cache the NETType expression so later calls to LoadNETType with the same args are fast.
               Note that when an object argument was supplied, we cache based on an identifying string
               obtained from that object, not the object itself. That allows the cache info to stay
               correct even if the type or assembly objects are released. It also prevents lookupType
               from getting cluttered with rules for Removed objects.
            *)
            Which[
                typeObj =!= Null,
                    lookupType[typeObj@AssemblyQualifiedName] = result,
                assemblyObj =!= Null,
                    lookupType[fixedName, assemblyObj@FullName] = result,
                True,
                    lookupType[fixedName, asmName] = result
            ]
        ];
        result
    ]
    

(* Called from .NET whenever classes need to be loaded by .NET code. This is currently in two circumstances: loading
   parent classes of a class the user has manually loaded using LoadNETType or NETNew; or classes loaded because an object
   of their type is being returned from .NET.
   Note that if you want to have a class loaded with your own settings for the options of LoadNETType, then you had better
   load it yourself, before it is autoloaded for you.
*)
loadTypeFromNET[assemblyQualifiedTypeName_String] := LoadNETType[assemblyQualifiedTypeName, "", StaticsVisible->False]


makeNETType[typeName_String, aqTypeName_String] := NETType[typeName, aqTypeToID[aqTypeName]]
makeNETAssembly[simpleName_String, displayName_String] := NETAssembly[simpleName, asmNameToID[displayName]]


(* These have just the following definitions. Defs are never added (instead, upvalues are placed on the NETObjectN symbols). *)
NETObjectQ[_] = False
NETObjectQ[Null] = True
Internal`SetValueNoTrack[NETObjectQ, True]

isCOMObject[_] = False
hasAliasedVersions[_] = False

aqTypeNameFromInstance[_] = $Failed


(* Maps type names in the System hierarchy to their assemblies. This allows us to avoid forcing users to manually load
   .NET Framework assemblies. Returns Null if no assembly information is available.
*)
typeNameToSystemAssemblyName[typeName_String] :=
    Which[
        (* No need for System.Windows.Forms, as it is loaded at startup.
           Some of these other assemblies may be loaded at startup also, but it is not clear
           if this behavior is identical in all versions of the .NET Framework.
        *)
        !StringMatchQ[typeName, "System.*"],
            Null,
        StringMatchQ[typeName, "System.Drawing.*"],
            "System.Drawing",
        StringMatchQ[typeName, "System.Data.*"],
            "System.Data",
        StringMatchQ[typeName, "System.Xml.*"],
            "System.Xml",
        StringMatchQ[typeName, "System.Web.Services.*"],
            "System.Web.Services",
        StringMatchQ[typeName, "System.Web.*"],
            "System.Web",
        StringMatchQ[typeName, "System.Messaging.*"],
            "System.Messaging",
        StringMatchQ[typeName, "System.Management.*"],
            "System.Management",
        StringMatchQ[typeName, "System.DirectoryServices.*"],
            "System.Directoryservices",
        StringMatchQ[typeName, "System.ServiceProcess.*"],
            "System.Serviceprocess",
        StringMatchQ[typeName, "System.Windows.Forms.Design.*"],
            "System.Design",
        StringMatchQ[typeName, "System.Linq.*"],
            "System.Core",
        True,
            Null
    ]
    

(* Fixes up generic type names to allow users to specify convenient but incorrect names for generic types.
   This function is used to massage the user-entered type name into its correct form.
   For example, the correct name is:
       System.Collections.Generic.Dictionary`2[System.String, System.String]
       System.Collections.Generic.Dictionary`2[[System.String, mscorlib, ...aq stuff], [System.String, mscorlib, ...aq stuff]]
   but we want to allow users to be able to enter:
       System.Collections.Generic.Dictionary`2[System.String, System.String]
       System.Collections.Generic.Dictionary`2[String, String]
       System.Collections.Generic.Dictionary<String, String>
       System.Collections.Generic.Dictionary<String, int>
   Cannot handle nested generics unless you use the full proper syntax.
   This method returns the name unchanged if it is not a generic type.
*)
fixGenericTypeName[name_String] :=
	Which[
		StringMatchQ[name, "*`*`*"] || StringMatchQ[name, "*<*<*"],
			(* Nested generic, just bail out and hope user got the name exactly right. *)
			name,
		StringMatchQ[name, __ ~~ "`" ~~ DigitCharacter ~~ "[" ~~ __],
			StringReplace[name, first:(__ ~~ "`" ~~ DigitCharacter ~~ "[") ~~ types__ ~~ "]" :> 
							StringJoin[first, Riffle[fixGenericTypeName /@ trim /@ StringSplit[types, ","], ","], "]"]],
		StringMatchQ[name, __ ~~ "<" ~~ __ ~~ ">"],
			StringReplace[name, first__ ~~ "<" ~~ types__ ~~ ">" :> 
							StringJoin[first, "`", ToString[Length[StringSplit[types, ","]]],
								 "[", Riffle[fixGenericTypeName /@ trim /@ StringSplit[types, ","], ","], "]"]],
		StringMatchQ[name, "[*]"],
			(* Type args in a generic type can be wrapped in []: List`1[[System.Int32]]. Generally the extra brackets
			   are only used if the type name is assembly-qualified (needed because that has commas in it).
			*)
			StringReplace[name, StartOfString ~~ "[" ~~ type__ ~~ "]" ~~ EndOfString :> "[" <> fixGenericTypeName[type] <> "]"],
		True,
			name /. $csharpToNETTypeRules /. $vbToNETTypeRules
	]

trim[s_String] := StringReplace[s, {StartOfString ~~ Whitespace -> "", Whitespace ~~ EndOfString -> ""}]
   

createInstanceDefs[typeName_String, aqTypeName_String, obj_Symbol, loadType_, isAnAlias_, isCOMObj_:False, comInterfaceName_String:""] :=
    (
        If[loadType && Head[loadTypeFromNET[aqTypeName]] =!= NETType,
            Return[$Failed]
        ];
        
        SetAttributes[obj, {HoldAllComplete}];
        Internal`SetValueNoTrack[obj, True];
        NETObjectQ[obj] ^= True;
        If[isAnAlias,
			(* The isAnAlias parameter means that this object has been seen before in another guise. Typically,
			   it will have been seen first in an uncasted form and now it is coming in casted, but this order
			   could be reversed. An object is added to a NETBlock only the first time it is seen. We need
			   to marak any pre-existing alias as having the property that other aliases exist for it.
			*)
			(hasAliasedVersions[#] = True)& /@ findAliases[obj],
		(* else *)
			addToNETBlock[obj]
		];
                
        (* This defeats normal precedence for @ operator. Needed for chaining: obj@meth1[]@meth2[]. This def
           must be made before the one below it.
        *)
        obj[(meth:_[___])[args___]] := obj[meth][args];
        (* This lets props/fields chain: obj@prop1@prop2@field.
           The isCOMNonPrimitiveFieldOrSimpleProp test always gives false for raw RCW COM objects for which type
           info is not available. That means you cannot chain field/property calls for such objects.
        *)
        obj[fieldOrProp_Symbol[arg__]] := obj[fieldOrProp][arg] /;
                (isNonPrimitiveFieldOrSimpleProp[aqTypeName, SymbolName[Unevaluated[fieldOrProp]]] ||
                (isCOMObject[obj] && isCOMNonPrimitiveFieldOrSimpleProp[obj, SymbolName[Unevaluated[fieldOrProp]]]));
        (* This next pattern also works for parameterized property gets: obj@Item[0]. *)
        obj[meth_[args___]] := netInstanceMethod[obj, meth, args];
        obj[fieldOrProp_Symbol] := netFieldOrPropertyGet[obj, fieldOrProp];
        If[hasIndexer[aqTypeName] || isCOMObj,
            (* C# indexer syntax: obj[1], obj[2.0], obj[True], obj["string"]. *)
            obj[True] := netIndexerGet[obj, True];
            obj[False] := netIndexerGet[obj, False];
            obj[s__String] := netIndexerGet[obj, s];
            obj[indices__Integer] := netIndexerGet[obj, indices];
            obj[indices__Real] := netIndexerGet[obj, indices];
        ];
        (* Field, property, and indexer _sets_ are handling via Set hack. *)
        (**** TODO: ??? Do I add a fallthru rule for obj[___] that gives a "no valid indexer" message or just
              ignore this?
        *)
        
        aqTypeNameFromInstance[obj] ^= aqTypeName;
        isCOMObject[obj] ^= isCOMObj;
        
        If[comInterfaceName == "",
            (* Normal object, COM object typed as a .NET interface, or COM object without any type info. *)
            Format[obj, OutputForm] = Format[obj, TextForm] = "<<NETObject[" <> typeName <> "]>>";
            With[{tn = typeName, summaryBoxData = NETObjectSummaryBox[obj, typeName, False]},
                If[Head[summaryBoxData ] === Association,
                    obj /: MakeBoxes[obj, fmt_] := BoxForm`ArrangeSummaryBox[NETObject, obj, summaryBoxData["Icon"], summaryBoxData["Items"], summaryBoxData["ExpandedItems"], StandardForm],
                (* else *)
                    obj /: MakeBoxes[obj, fmt_] = InterpretationBox[RowBox[{"\[LeftGuillemet]", RowBox[{"NETObject", "[", tn, "]"}], "\[RightGuillemet]"}], obj]
                ]
            ],
        (* else *)
            (* COM object has type info in the form of the name of a COM interface. Give the object a special
               appearance that indicates its "type".
            *)
            Format[obj, OutputForm] = Format[obj, TextForm] = "<<NETObject[COMInterface[" <> comInterfaceName <> "]]>>";
            With[{tn = comInterfaceName, summaryBoxData = NETObjectSummaryBox[obj, comInterfaceName, True]},
                If[Head[summaryBoxData ] === Association,
                    obj /: MakeBoxes[obj, fmt_] := BoxForm`ArrangeSummaryBox[NETObject, obj, summaryBoxData["Icon"], summaryBoxData["Items"], summaryBoxData["ExpandedItems"], StandardForm],
                (* else *)
                    obj /: MakeBoxes[obj, fmt_] = InterpretationBox[RowBox[{"\[LeftGuillemet]", RowBox[{"NETObject", "[", "COMInterface", "[", tn, "]", "]"}], "\[RightGuillemet]"}], obj]
                ]
            ]
        ];
        
        runListeners[obj, "Create"];
        
        (* Because we can handle chained fields/props, including sets (obj@propA@propB = 42), it is possible that
           we have made a nasty error easier. If you do this: (obj@propA)@propB = 42, then the NETObjectXXXX symbol returned
           from obj@propA will have an assignment made to it: NETObjectXXXX[propB] = 42. In other words, this call does not
           go into .NET like it should. This is very hard to figure out, since from then on every time you eval obj@propA@propB
           you get 42, no matter how many times you reassign to it, even if you do the reassignment correctly. You also
           get this error if you write %@Prop = 42. In fact, you get it whenever you have an LHS of the Set that doesn't
           begin with a symbol that evaluates to a NETObject expression.
           By protecting the obj symbols, we get a rather cryptic Set::write "protected" error message on the object
           if things go wrong.
        *)
        Protect[obj];
        obj
    )


createStaticMethodDef[aqTypeName_String, ctxt_String, shortCtxt_String, methName_String, isOperator_:False] :=
    Module[{legalName = toLegalName[methName]},
        With[{sym = Symbol[ctxt <> legalName]},
            (* For operator overloads, create an additional method opXXX for each op_XXX. This is for convenience of programmers. *)
            If[StringMatchQ[methName, "op_*"],
                createStaticMethodDef[aqTypeName, ctxt, shortCtxt, StringReplace[methName, "op_" -> "op"], True]
            ];
            (* HoldAll needed to implement out/ref params. *)
            Attributes[sym] = {HoldAll};
            If[!isOperator,
                sym[args___] := netStaticMethod[aqTypeName, methName, args],
            (* else *)
                (* If we are in this function with isOperator == True, we are making defs for a method opXXX that is an alias
                   for op_XXX. Make sure that the meth name we pass to .NET is the original one with the underscore.
                *)
                With[{actualMethName = StringReplace[methName, "op" -> "op_"]},
                    sym[args___] := netStaticMethod[aqTypeName, actualMethName, args]
                ]
            ];
            (* Downvalues of isNETStaticSymbol are used to record which symbols in a context hav been given defs.
               This is used only in clearOutClassContext, to avoid clearing non-.NET symbols in case the same
               context name is being used by a Mathematica package. No need to do this for the shortCtxt symbols,
               as they do not need to be cleared when the class is unloaded. They just point to their deep-context
               counterparts, which will get cleared.
            *)
            isNETStaticSymbol[ctxt <> legalName] = True;
            isNETStaticSymbol[shortCtxt <> legalName] = True;
            (* Now make def also available in "short" class context. Evaluate is necessary here but not
               for DownValues above due to strangeness of Set partial lhs eval. If user has specified to not allow
               short contexts, the short context will be the same as the long one, hence the test.
            *)
            If[shortCtxt != ctxt, Evaluate[ToExpression[shortCtxt <> legalName]] = sym]
        ];
    ]

createStaticFieldOrPropertyDef[aqTypeName_String, ctxt_String, shortCtxt_String, fldName_String, isParameterized_] :=
    Module[{legalName = toLegalName[fldName]},
        createStaticFieldOrPropertyDef0[ToHeldExpression[ctxt <> legalName], aqTypeName, ctxt, fldName, legalName, isParameterized];
        If[shortCtxt != ctxt,
             (* Because we set UpValues for Set calls, it is not enough to just define the shortCtxt field symbols to be the
                deep context symbols, as is done with methods. Instead, we must explicitly make definitions for the shortCtxt ones.
             *)
             createStaticFieldOrPropertyDef0[ToHeldExpression[shortCtxt <> legalName], aqTypeName, shortCtxt, fldName, legalName, isParameterized]
        ]
    ]
    
createStaticFieldOrPropertyDef0[Hold[sym_], aqTypeName_String, ctxt_String, fldName_String, legalFldName_String, isParameterized_] :=
    (* The !ValueQ test prevents this from being called twice on a symbol. This will happen if you load two classes
       with the same short context. Calling it twice can cause all sorts of bad behavior.
    *)
    If[!ValueQ[sym],
        (* If we wanted to completely remove all reliance on Set for statics we could just remove the
           following two lines. There is no reason to want to do that, though.
        *)
        (* This for field and simple property sets. *)
        sym /: Set[sym, val_] :=
                    netStaticFieldOrPropertySet[aqTypeNameFromStaticSymbol[sym], fieldNameFromStaticSymbol[sym], val];
        If[isParameterized,
            (* For static parameterized properties: *)
            sym /: Set[x:sym[params__], val_] :=
                        netStaticFieldOrPropertySet[aqTypeNameFromStaticSymbol[sym], fieldNameFromStaticSymbol[sym], params, val]
        ];
        (* Must make this def last. *)
        If[isParameterized,
            sym[params__] := netStaticFieldOrPropertyGet[aqTypeName, fldName, params],
        (* else *)
            sym := netStaticFieldOrPropertyGet[aqTypeName, fldName]
        ];
        aqTypeNameFromStaticSymbol[sym] = aqTypeName;
        fieldNameFromStaticSymbol[sym] = fldName;
        isNETStaticSymbol[ctxt <> legalFldName] = True
    ]

createStaticEventDef[aqTypeName_String, ctxt_String, shortCtxt_String, evtName_String] :=
    Module[{legalName = toLegalName[evtName]},
        createStaticEventDef0[ToHeldExpression[ctxt <> legalName], aqTypeName, ctxt, evtName, legalName];
        If[shortCtxt != ctxt,
            createStaticEventDef0[ToHeldExpression[shortCtxt <> legalName], aqTypeName, shortCtxt, evtName, legalName]
        ]
    ]

createStaticEventDef0[Hold[sym_], aqTypeName_String, ctxt_String, evtName_String, legalEvtName_String] :=
    If[!ValueQ[sym],
        (* For events, we want an error message issued every time the user tries to refer to one
           outside the context of Add/RemoveEventHandler. These are the only defs needed.
        *)
        sym /: Set[sym, val_] := (Message[NET::event, legalEvtName]; $Failed);
        sym := (Message[NET::event, legalEvtName]; $Failed);
        aqTypeNameFromStaticSymbol[sym] = aqTypeName;
        isNETStaticSymbol[ctxt <> legalEvtName] = True
    ]


(****************************************   NETObjectListener  *****************************************)

If[!ListQ[$listeners], $listeners = {}]

AddNETObjectListener[listenerFunc_] :=
    If[!MemberQ[$listeners, listenerFunc],
        AppendTo[$listeners, listenerFunc]
    ]        

RemoveNETObjectListener[listenerFunc_] :=
    $listeners = DeleteCases[$listeners, listenerFunc]       

NETObjectListeners[] := $listeners

(* Methods are "Create" and "Release". *)
SetAttributes[runListeners, Listable]
runListeners[obj_, method_String] /; Length[$listeners] > 0 :=
    With[{typeName = First[StringSplit[aqTypeNameFromInstance[obj], ","]]}, Scan[#[obj, typeName, method]&, $listeners]]


(*******************************************  GetTypeObject  *******************************************)

(* GetTypeObject does exactly one thing--it takes a NETType expression and returns a Type object for that type.
   Because .NET/Link uses a moniker (NETType) instead of actual Type objects, it is important to have a method
   to get the actual corresponding Type object from it.
*)

GetTypeObject::arg = "The argument must be a NETType expression returned from LoadNETType."
GetTypeObject::type = "The argument is not a valid NETType expression. Perhaps the .NET runtime has been restarted since LoadNETType was called."

GetTypeObject[type_NETType] :=
    Module[{aqType, obj},
        aqType = getAQTypeName[type];
        If[!StringQ[aqType],
            Message[GetTypeObject::type];
            Return[$Failed]
        ];
        (* If getAQTypeName returns a string, it will always be correct and valid for the current .NET session. *)
        nGetTypeObject[aqType]
    ]

GetTypeObject[x_] :=
    (
        Message[GetTypeObject::arg];
        $Failed
    )
    
    
(********************************************  netXXX Methods  **********************************************)

(* All calls into .NET for member accesses (i.e., ctors, methods, fields, props, etc., static or otherwise) go through
   functions named netXXX. All the code for these functions is in this section. Another way of saying this is that
   all calls to nCall[] happen here. This is the layer at which to intercept or wrap all such calls, if that is
   desired in the future.
   
   Block used instead of Module in these functions for speed only, except for $outParamHolder, $wereOutParams.
*)

(* 
   Format for nCall:
   
      nCall[typeName_String, obj_Symbol, callType_Integer, isByRef:(True | False), memberName_String, argCount_Integer, types___, args___]
      
      The callTypes are:
        0  constructor
        1  field or simple prop get
        2  field or simple prop set
        3  parameterized prop get (statics only--cannot distinguish instance param prop gets from method calls)
        4  parameterized prop set
        5  method calls and non-static param prop gets.
*)


(* No attributes necessary for netConstructor. *)

netConstructor[aqTypeName_String, args___] :=
    nCall[aqTypeName, Null, 0, True, "", Length[{args}], argTypes[args], args]


Attributes[netInstanceMethod] = {HoldRest}

netInstanceMethod[obj_, meth_, args___] :=
    netMethod[aqTypeNameFromInstance[obj], obj, TrueQ[$byRef], ToString[Unevaluated[meth]], Hold[args]];


Attributes[netFieldOrPropertyGet] = {HoldRest}
Attributes[netFieldOrPropertySet] = {HoldRest}
(* No attributes necessary for netIndexerGet. *)

(* Typical instance field or prop get: obj@prop. There is no signature for this that takes args for a parameterized prop, because
   they look just like method calls and thus go through netInstanceMethod[].
*)
netFieldOrPropertyGet[obj_, fieldOrProp_] :=
    nCall[aqTypeNameFromInstance[obj], obj, 1, TrueQ[$byRef], ToString[Unevaluated[fieldOrProp]], 0]

(* Typical instance field or property set: obj@prop = val. *)
netFieldOrPropertySet[obj_, fieldOrProp_Symbol, val_] :=
    If[# === $Failed, $Failed, val]& @ 
        nCall[aqTypeNameFromInstance[obj], obj, 2, False, ToString[Unevaluated[fieldOrProp]], 1, argTypes[val], val]

(* Chained instance simple property/field set: obj@prop@something = val. *)
netFieldOrPropertySet[obj_, fieldOrProp_Symbol[arg_], val_] /;
            (isNonPrimitiveFieldOrSimpleProp[aqTypeNameFromInstance[obj], SymbolName[Unevaluated[fieldOrProp]]] ||
             (isCOMObject[obj] && isCOMNonPrimitiveFieldOrSimpleProp[obj, SymbolName[Unevaluated[fieldOrProp]]])) :=
    netFieldOrPropertySet[obj[fieldOrProp], arg, val]

(* Chained instance simple property/field set: obj@meth[]@prop = val. *)
netFieldOrPropertySet[obj_, (meth:_[___])[arg_], val_] :=
    netFieldOrPropertySet[obj[meth], arg, val]

(* Parameterized instance property set: obj@prop[index] = val. *)
netFieldOrPropertySet[obj_, prop_Symbol[params__], val_] :=
    If[# === $Failed, $Failed, val]& @ 
        nCall[aqTypeNameFromInstance[obj], obj, 4, False, ToString[Unevaluated[prop]], Length[{params}] + 1, argTypes[params], argTypes[val], params, val]

(* C# indexer set: obj[index] = val. *)
netFieldOrPropertySet[obj_, indices__, val_] /; hasIndexer[aqTypeNameFromInstance[obj]] :=
    If[# === $Failed, $Failed, val]& @ 
        nCall[aqTypeNameFromInstance[obj], obj, 4, False, "", Length[{indices}] + 1, argTypes[indices], argTypes[val], indices, val]

netFieldOrPropertySet[obj_, __, val_] := (Message[NET::lval]; $Failed)
  
netIndexerGet[obj_, indices__] :=
    nCall[aqTypeNameFromInstance[obj], obj, 3, TrueQ[$byRef], "", Length[{indices}], argTypes[indices], indices]


(* Helper function shared by both netInstanceMethod and netStaticMethod. No attributes necessary. *)
netMethod[aqTypeName_, obj_, isByRef_, methName_, args_Hold] :=
    Block[{result, lval, $outParamHolder, $wereOutParams},
        result = nCall[aqTypeName, obj, 5, isByRef, methName, Length[args], argTypes @@ args, ReleaseHold[args]];
        If[$wereOutParams,
            Do[
                If[With[ {i = i}, ValueQ[$outParamHolder[i]] ],
                    lval = args[[{i}]];
                    If[MatchQ[lval, Hold[_Symbol]],
                        Function[sym, sym = $outParamHolder[i], {HoldFirst}] @@ lval
                    (**********  Do nothing if arg was not a symbol--just quietly fail to make an assignment.
                                 This behavior is like VB, which allows you to pass literals for ByRef params.
                    (* else *)
                        Message[NET::outparam, ToString[Unevaluated[methName]], i]
                    ***********)
                    ]
                ],
                {i, Length[args]}
            ]
        ];
        result
    ]

(*******  Statics  ********)

Attributes[netStaticMethod] = {HoldRest}

netStaticMethod[aqTypeName_String, methName_String, args___] :=
    netMethod[aqTypeName, Null, TrueQ[$byRef], methName, Hold[args]]


netStaticFieldOrPropertyGet[aqTypeName_String, fieldOrPropName_String, propertyParams___] :=
    Block[{callType = If[Length[{propertyParams}] == 0, 1, 3]},
        nCall[aqTypeName, Null, callType, TrueQ[$byRef], fieldOrPropName, Length[{propertyParams}], argTypes[propertyParams], propertyParams]
    ]

netStaticFieldOrPropertySet[aqTypeName_String, fieldOrPropName_String, propertyParams___, val_] :=
    Block[{callType = If[Length[{propertyParams}] == 0, 2, 4]},
        If[# === $Failed, $Failed, val]& @ 
            nCall[aqTypeName, Null, callType, False, fieldOrPropName, Length[{propertyParams}] + 1, argTypes[propertyParams], argTypes[val], propertyParams, val]
    ]
    
netStaticFieldOrPropertySet[aqTypeNameFromStaticSymbol[fieldOrPropName_Symbol], _] :=
    (Message[NET::staticfield, HoldForm[fieldOrPropName]]; $Failed)
    
netStaticFieldOrPropertySet[aqTypeNameFromStaticSymbol[fieldOrPropName_Symbol], params__, _] :=
    (Message[NET::staticprop, HoldForm[fieldOrPropName]]; $Failed)


(*********************************************  NETNew  **********************************************)

(**
    NETNew accepts the folowing types of arguments:
    
        NETNew[_NETType, args]
        NETNew["Type.Name", args]
        NETNew["Type.Name, AssemblyName", args]
        NETNew[typeObject, args]
        
        NETNew[{"Type.Name", "Assembly.Name"}, args]
        NETNew[{"Type.Name", "AssemblyName.dll"}, args]
        NETNew[{"Type.Name", _NETAssembly}, args]
        NETNew[{"Type.Name", assemblyObj}, args]
        NETNew[{"Type.Name", "path/to/assemblyFile"}, args]
        NETNew[{"Type.Name", "http://url/to/assemblyFile"}, args]
        
        NETNew[{"Type.Name", "assemblyName", "path/to/assemblyDir"}, args]
        NETNew[{"Type.Name", "assemblyName", "AppContext`"}, args]

   The assembly must already have been loaded, or you must provide enough information in the arguments to
   allow the assembly to be loaded.
**)

NETNew::args = "Improper count or type of arguments."
NETNew::typeobj = "Object `1` is not an instance of the Type class."
NETNew::asmobj = "Object `1` is not an instance of the Assembly class."


NETNew[typeName_String, args___] := NETNew[{typeName, ""}, args]

NETNew[type_NETType, args___] := netConstructor[getAQTypeName[type], args]

(* Create an object from a Type object, such as one obtained by reflection. *)
NETNew[typeObj_?NETObjectQ, args___] :=
    If[InstanceOf[typeObj, "System.Type"],
        NETNew[LoadNETType[typeObj], args],
    (* else *)
        Message[NETNew::typeobj, typeObj];
        $Failed
    ]
    
NETNew[{typeName_String, assembly:(_String | _NETAssembly)}, args___] :=
    Module[{type},
        type = LoadNETType[typeName, assembly];
        If[Head[type] === NETType,
            netConstructor[getAQTypeName[type], args],
        (* else *)
            (* Message will have already been issued by LoadNETType. *)
            $Failed
        ]
    ]
    
NETNew[{typeName_String, assembly_?NETObjectQ}, args___] :=
    If[InstanceOf[assembly, "System.Reflection.Assembly"],
        NETNew[{typeName, LoadNETAssembly[assembly]}],
    (* else *)
        Message[NETNew::asmobj, assembly];
        $Failed
    ]

NETNew[{typeName_String, assembly_String, dirOrContext_String}, args___] :=
    Module[{type},
        type = LoadNETType[typeName, assembly, dirOrContext];
        If[Head[type] === NETType,
            netConstructor[getAQTypeName[type], args],
        (* else *)
            (* Message will have already been issued by LoadNETType. *)
            $Failed
        ]
    ]
    
    
NETNew[___] := (Message[NETNew::args]; $Failed)


(*****************************************  CastNETObject  *****************************************)

(* "Casts" the specified object to the given type. Works for either .NET objects or COM objects.

   For .NET objects, this is generally an upcast so that you can call a version of a method that is
   hidden-by-signature by another version in a lower class. Casting an object does not create a new
   reference on the .NET side, nor is a new object entered into the current NETBlock. Casted versions
   of an object are "aliases" of a single object that appears only once in the .NET/Link reference-
   management system.

   If you have a COM object that is not typed as a managed type (it is a raw __ComObject with no type info,
   or it has type info and thus appears as <<NETObject[COMInterface["..."]]>>)
   you can use this function to create a wrapper object of the desired managed type. The managed type can be a
   class or interface. There will be an error if the object cannot be cast to the specified type.
   
   An example of this is if you call a method in an interop assembly that is typed to return plain object (this is not
   at all uncommon). In such cases you get a raw __ComObject. But you might know that this object supports a certain
   managed interface, for example if it was obtained from the Excel Worksheets collection you know it is a Worksheet.
   You can use CastCOMObject to cast it to the desired managed type. Basically, this is the Mathematica equivalent
   of what goes on under the hood in a C# program when you cast like this (this is pseducode, might not be exactly right:)
   
        Worksheet sheet = (Worksheet) excel.Worksheets(2);
        
        // Even better to cast to the class type (Worksheet is an interface):
        WorksheetClass sheet = (WorksheetClass) excel.Worksheets(2);
*)

CastNETObject::unktype = "The specified type could not be loaded by .NET/Link."

CastNETObject[obj_?NETObjectQ, type_NETType] :=
    Module[{aqTypeName},
        aqTypeName = getAQTypeName[type];
        Block[{$internalNETExceptionHandler = associateMessageWithSymbol[CastNETObject]},
            nCast[obj, aqTypeName]
        ]
    ]

CastNETObject[obj_?NETObjectQ, typeName_String] :=
    Module[{type},
        type = LoadNETType[typeName];
        If[Head[type] === NETType,
            Block[{$internalNETExceptionHandler = associateMessageWithSymbol[CastNETObject]},
                CastNETObject[obj, type]
            ],
        (* else *)
            Message[CastNETObject::unktype, typeName];
            $Failed
        ]
    ]

(* Message here is defined for General. *)
CastNETObject[obj_, _] := (Message[CastNETObject::netobj1, obj]; $Failed)


(****************************************  Set Modification  *********************************************)

(* Cost of this Set hack is that it makes Set run 2 times slower for args that match the pattern,
   and f in f[symbol] = val gets evaluated twice.
*)
prot = Unprotect[Set]
Set[sym_Symbol[args__], val_] /; NETObjectQ[sym] := netFieldOrPropertySet[sym, args, val]
Protect[Evaluate[prot]]


(******************************************  ExternalCall fix  *********************************************)

(* Calls to external functions via the standard Install mechanism are accomplished by the function ExternalCall.
   Unfortunately, ExternalCall is deficient in its handling of aborts. Thus, I need my own version, netlinkExternalCall.
   I also need my own version of DefineExternal to create definitions that call netlinkExternalCall instead of
   ExternalCall. 
   
   These functions would have to be revised if anything about the internals of Mathematica's Install/Uninstall
   mechanism changed.
*)

(* This is identical to DefineExternal except that it creates defs that call netlinkExternalCall instead of ExternalCall.
   This function is called directly from .NET.
*)
netlinkDefineExternal[p_String, a_, n_] := 
	Module[{e, pat = ToHeldExpression[p], args = ToHeldExpression[a]}, 
		If[pat === $Failed || args === $Failed, 
			Message[DefineExternal::des, n, InputForm[$CurrentLink]],
		(* else *)
			e = Hold[_ := netlinkExternalCall[_, CallPacket[_, _]]];
			e = ReplaceHeldPart[e, pat, {1, 1}];
			e = ReplaceHeldPart[e, Hold[getActiveNETLink[]], {1, 2, 1}];
			e = ReplacePart[e, n, {1, 2, 2, 1}];
			e = ReplaceHeldPart[e, args, {1, 2, 2, 2}];
			ReleaseHold[e];
			System`Dump`defined = Append[System`Dump`defined, HoldForm @@ pat /.
				{Literal[ThisLink] -> $CurrentLink, Literal[$CurrentLink] -> $CurrentLink}]
		];
	]



Internal`SetValueNoTrack[$inExternalCall, True]
Internal`SetValueNoTrack[$inPreemptiveCallToNET, True]


(* This function differs functionally from ExternalCall in that it wraps the write-read pair in AbortProtect, so that you
   cannot leave the link in an "off-by-one" state by aborting between the write and read. This AbortProtect does not
   prevent the necessary behavior that user aborts fired while the kernel is blocking in LinkRead are sent to .NET
   as MLAbortMessages. In other words, the ability to abort .NET computations is not affected. The AbortProtect does
   prevent the behavior of being able to do a "hard" abort via the two-step combination of
   "Interrupt Evaluation/Abort Command Being Evaluated". This procedure causes Mathematica to treat the
   abort like any other abort and ignore that it is in LinkRead. This is not very useful, though, since the link will
   probably be out of sync because the result is not read. The correct way to handle this is to select "Kill linked program"
   in the Interrupt dialog box, not "Abort Command Being Evaluated". This causes .NET to quit.
   
   The code itself is quite different from ExternalCall. Gone is the need for ExternalAnswer and the recursive way
   in which that was implemented.
   
   The link that will be passed in here is the one given by getActiveNETLink[].
   
   This function also rejects preemptive calls into .NET when they are unsafe. Note that the criterion for rejection is
   slightly broader than with J/Link. This is because there is only one thread in .NET/Link, and among other things this
   means that making a call to .NET on NETLink[] is not allowed if a call on the NETUILink[] is in progress. A corollary
   is that is is never safe to allow a preemptive call to .NET when NETUILink[] is $ParentLink, because a transaction
   on the UI link might be in progress and therefore the NETLink[] is not available to use. Unfortunately, this bars
   preemptive calls to .NET for a wider range of time than when they are truly unsafe, because a non-preemptive call
   from .NET (e.g., a modeless dialog click) will leave $ParentLink set to the UI link until input arrives from another
   link. During this period, periodicals that try to call .NET will be rejected. Note that this is only a problem
   for periodicals.
   
   Note that there is still an unsafety in this that cannot be avoided. Say an event is triggered in
   .NET that causes a non-preemptive (i.e., main loop) call to M. At the same moment, a call to .NET occurs from
   a periodical. This call is not rejected by netlinkExternalCall because the call from .NET into M has not
   begun yet. But it cannot proceed because the nonpreemptive call from .NET is occurring from within the yield
   function on NETLink[]. Thus, deadlock.
*)

netlinkExternalCall[link_LinkObject, packet_CallPacket] :=
	Block[{ThisLink = link, $CurrentLink = link, pkt = packet, res, isPreemptive = TrueQ[MathLink`IsPreemptive[]],
	         $inExternalCall = $inExternalCall, $inPreemptiveCallToNET = $inPreemptiveCallToNET},
		AbortProtect[
			(* Here we reject calls that are unsafe. *)
			If[isPreemptive && link === NETLink[] && ($ParentLink === NETUILink[] || (TrueQ[$inExternalCall] && !TrueQ[$inPreemptiveCallToNET])),
				Message[NET::preemptive];
				$Failed,
			(* else *)
				$inExternalCall = True;
				If[isPreemptive, $inPreemptiveCallToNET = True];
			    While[True,
				    If[LinkWrite[link, pkt] === $Failed, Return[$Failed]];
				    res = LinkReadHeld[link];
				    Switch[res,
					    Hold[EvaluatePacket[_]],
						    (* Re-enable aborts during the computation in Mathematica of EvaluatePacket contents, but have
						    them just cause $Aborted to be returned to .NET, not call Abort[].
						    *)
						    pkt = ReturnPacket[CheckAbort[res[[1,1]], $Aborted]],
					    Hold[ReturnPacket[_]],
						    Return[res[[1,1]]],
					    Hold[_],
						    Return[res[[1]]],
					    _,
						    Return[res]
				    ]
			    ]
			]		
		]
	]


(* This gives the link that will be used for any given call to .NET. Note that preemptive calls
   to .NET will never use the UI link unless they are just callbacks during a preemptive comp
   that began in .NET. Note also that NETUILink[] will never be returned unless it is safe to use it.
*)
getActiveNETLink[] :=
	Block[{isPreemptive = TrueQ[MathLink`IsPreemptive[]]},
		If[!isPreemptive && $ParentLink === NETUILink[] && $ParentLink =!= Null ||
				isPreemptive && $inPreemptiveCallFromNET,
			NETUILink[],
		(* else *)
			NETLink[]
		]
	]
	
	
NET::preemptive = "Calls into .NET cannot be made from a preemptive computation while another call into .NET is in progress."


(******************************   ReturnAsNETObject/NETObjectToExpression   *********************************)

(* Note that ReturnAsNETObject sets up an "environment" where all calls return by ref. This means that
   ReturnAsNETObject[foo[obj@method[]]] will work, but be careful if there are more NET calls embedded in the
   expression, as in ReturnAsNETObject[obj@method[SomeClass`FOO]], as these deeper calls will also return by ref.
*)

SetAttributes[ReturnAsNETObject, HoldAll]

ReturnAsNETObject[x_] := Block[{$byRef = True}, x]


NETObjectToExpression[x_?NETObjectQ] := nVal[x]

NETObjectToExpression[x_] := x   (* Perhaps this should issue a message? *)


(*****************************************   Loaded Types   **********************************************)

LoadedNETTypes[] := (InstallNET[]; makeNETType @@@ nPeekTypes[])

(* Union here because the same Assembly object can show up twice. I think this happens when the assembly
   is loaded once via Load and once via LoadFrom.
*)
LoadedNETAssemblies[] := (InstallNET[]; makeNETAssembly @@@ unsortedUnion[nPeekAssemblies[]])


unsortedUnion[x_] := Module[{result, f}, f[y_] := (f[y] = Sequence[]; y); result = f /@ x; Clear[f]; result]


(*******************************  Various Package-Level and Private Funcs  **********************************)

getAQTypeName[type_NETType] := idToAQType[type[[2]]]
getAQTypeName[Null] = ""
getAQTypeName[obj_?NETObjectQ] := aqTypeNameFromInstance[obj]

getFullAsmName[asm_NETAssembly] := idToAsmName[asm[[2]]]


(* outParam is returned from .NET to wrap out param info. $outParamHolder is localized by a Block during each method call. *)
outParam[argPosition_Integer, value_] := ($wereOutParams = True; $outParamHolder[argPosition] = value;)


(* argTypeToInteger is defined by .NET code during InstallNET. *)
argTypes[] := Sequence[]
argTypes[arg_] := argTypeToInteger[arg]
argTypes[args__] := Sequence @@ (argTypeToInteger /@ {args})


(* Called during Un/InstallNET to wipe out defs created as classes are loaded and objects created. *)
clearNETDefs[] :=
    (
        clearAllNETContexts[];
        clearPersistentData[];
        Unprotect["NETLink`Objects`*"];
        ClearAll["NETLink`Objects`*"];
    )


clearAllNETContexts[] := Scan[clearOutNETContext, $netContexts];

(* This one is used only when .NET is being started/stopped. *)
clearOutNETContext[ctxt_String] := 
    Module[{nms, netNames},
        (* Must get rid of $ContextPath or symbols in visible contexts will have their names returned without
           the context prefix.
        *)
        nms = Block[{$ContextPath}, Names[ctxt <> "*"]];
        (* Downvalues of isNETStaticSymbol are used to record which symbols in a context have been given defs.
           We use this to avoid clearing non-NET symbols in case the same context name is being used by a
           Mathematica package.
        *)
        netNames = Select[nms, isNETStaticSymbol];
        (* ClearAll["sym1", "sym2", ...] is vastly more expensive than ClearAll[sym1, sym2, ...] or ClearAll["ctxt`*"],
           so avoid the first method at all costs. We only do selective clearing when it is necessary (this will likely
           be only in cases where a read-in context has the same name as a NET-created one), and when we do, do it
           for symbol names rather than strings.
        *)
        If[netNames =!= {},
            If[Length[nms] == Length[netNames],
                ClearAll @@ {ctxt <> "*"},
            (* else *)
                (* The Unevaluated wrapped around each symbol does not interfere with ClearAll. *)
                ClearAll @@ (ToExpression[#, InputForm, Unevaluated]& /@ netNames)
            ]
        ];
    ]


(* This form only called if there is a namespace (StringLength[namespace] > 0). *)
shortClassContextFromClassContext[ctxt_String, namespace_String] := StringDrop[ctxt, StringLength[namespace] + 1]
(* This form is called for generic types, where we don't have a namespace, so we just drop everything but
   the last part of the context (but not the Gn part), e.g., A`B`C`G2` --> C`G2`.
*)
shortClassContextFromClassContext[ctxt_String] :=
	StringReplace[ctxt,
		{ShortestMatch[__] ~~ last:(Except["`"].. ~~ "`G" ~~ DigitCharacter ~~ "`" ~~ __) ~~ EndOfString :> last,
		 ShortestMatch[___] ~~ last:(Except["`"].. ~~ "`") ~~ EndOfString :> last
		}
	]

(* Converts a type name into a context. Works for generic types also. These type names come from .NET, not the user,
   and it is assumed that the full assembly-qualified info for generic params has been stripped on the C# side already.
   That is, types look like A`1[System.Int32] and never A`1[[System.Int32,mscorlib,....]].
   Generic types are a concatenation of the parent and the generic args (themselves in short form), with a
   G prepended to the arg count number.
   For example,
         A.B`2[System.Int32,C.D`1[System.String]]
      becomes
         A`B`G2`Int32`C`D`G1`String`
   Note that we replace "+" (the separator for a nested class) with ` in the context. This is what happens for
   a nested type within a generic type:
         System.Collections.Generic.List`1+Enumerator[System.String]
      becomes
         System`Collections`Generic`List`G1`Enumerator`String`
   Note how the nestedEnumerator type follows the G1 part, which is weird but that's what type names look like.
*)
toContextName[typeName_String] :=
	Module[{genericType, level = 0, betweenBrackets, nestedCommasReplaced, typesList},
		Which[
			StringMatchQ[typeName, "*[]"]  || StringMatchQ[typeName, "*,]"],
				(* Array type *)
				"System`Array`",
			StringMatchQ[typeName, "*]"],
				(* Generic type. Could be nested generic:
				     A.B`2[System.Int32,C.D`1[System.String]]
				*)
				genericType = toContextName[StringReplace[typeName, a:Except["`"].. ~~ "`" ~~ dig:DigitCharacter ~~ nestedType:Except["["]... ~~ "[" ~~ __ :> 
												a <> "`G" <> dig <> nestedType]];
				(* Parse the generic args. Look for bracket depth changes because could be nested generics. *)
				betweenBrackets = StringTake[typeName, {First@Flatten@StringPosition[typeName, "["] + 1, -2}];
				nestedCommasReplaced = Switch[#, "[", ++level;#, "]", --level;#, ",", If[level === 0, #, "\000"], _, #] & /@ Characters[betweenBrackets];
				typesList = trim /@ (StringReplace[#, "\000" -> ","]&) /@ StringSplit[StringJoin[nestedCommasReplaced], ","];
				(* Build the full type, which would be A`B`G2`Int32`C`D`G1`String` in the above example. *)
				genericType <> (shortClassContextFromClassContext /@ toContextName /@ typesList),
			StringMatchQ[typeName, "*,*"],
				StringReplace[typeName, type:Except[","].. ~~ "," ~~ ___ :> type] <> "`",
			True,
				typeName <> "`"
		] // StringReplace[#, {"." -> "`", "+" -> "`", "<" -> "", ">" -> ""}]&
		(* The < and > chars can appear in bizarre names from classes generated to handle 'yield return'. Just get rid of them. *)
	]
     
toLegalName[s_String] := StringReplace[s, "_" -> "U"]


(**********************************  Persistent Data Structures  ***************************************)

(*
    $netContexts
    
            List of all contexts corresponding to loaded .NET types (short and long contexts).
    
    isNETStaticSymbol["sym"]
    
    isNonPrimitiveFieldOrSimpleProp[aqTypeName, "prop"]
    
            Tells whether a given field or simple prop is of a reference type (something that would come
            into M as an object). This is used to allow chaining of field/prop accesses: obj@prop@prop@fld.
    
    lookupType[typeName, assemblyName]
    
            Returns a NETType for types that have been loaded. It maps user-supplied type,
            assembly names to types returned by LoadNETType. A simple optimization, nothing more.
            
    idToAQType[id_Integer]
    $idToAQTypeRules
    $typeIndex
    
            Caches the assembly-qualified type name associated with the id. The id is simply a way
            to avoid putting the whole AQ name in the NETType expression.
            
    aqTypeToID[aqTypeName]
    
            Reverse direction of idToAQTypeName.
    
    lookupAssembly[assemblyNameOrPath]
    
            Returns a NETAssembly for assemblies that have been loaded. A simple optimization, nothing more.
            
    idToAsmName[id_Integer]
    $idToAsmNameRules
    $asmIndex
    
            Caches the full name of the assembly associated with the id. The id is simply a way
            to avoid putting the whole asm name in the NETAssembly expression.
            
    asmNameToID[asmName]
    
            Reverse direction of idToAsmName.
    
    hasIndexer[aqTypeName]
    
            Tells whether this type has a C#-style indexer.

    aqTypeNameFromInstance[obj]
    
            Caches the full assembly-qualified type name for a given instance.
            
    aqTypeNameFromStaticSymbol[sym]
    
            Associates a full assembly-qualified type name with a static field/prop symbol (e.g. MyClass`MyProp)
            Used for sets: MyClass`MyProp = 42.
    
    fieldNameFromStaticSymbol[sym]

            Associates a field/prop name with a static field/prop symbol (e.g. MyClass`MyProp)
            would be mapped to "MyProp". This can be done programmatically, but it is expensive.
            Easier to just cache it. Like aqTypeNameFromStaticSymbol, this is only used for
            sets: MyClass`MyProp = 42.    
*)

(* Ensure that modifications to any of these globals do not trigger Dynamic updating mechanism. *)
Internal`SetValueNoTrack[#, True]& /@
	{$netContexts, isNETStaticSymbol, isNonPrimitiveFieldOrSimpleProp, lookupType, idToAQType,
	 $idToAQTypeRules, $typeIndex, aqTypeToID, lookupAssembly, idToAsmName, $idToAsmNameRules,
	 $asmIndex, asmNameToID, hasIndexer, aqTypeNameFromInstance, aqTypeNameFromStaticSymbol, fieldNameFromStaticSymbol}

If[!ValueQ[$netContexts],
    $netContexts = {}
]

If[!ValueQ[$idToAQTypeRules],
    $idToAQTypeRules = {};
    $typeIndex = 1;
]

(* Change in behavior of Dispatch from M9 to M10. This is the function that turns a Dispatch[...] expr into a list of rules. *)
rulesFromDispatchFunc = If[$VersionNumber >= 10, Normal, First]

(* This direction doesn't need to be fast. *)
aqTypeToID[aqName_String] :=
    Module[{rules, existingID, newID},
        (* Once the list of rules grows long enough, Dispatch starts creating an expression
            with head Dispatch, where the list of rules is the first arg.
        *)
        rules = If[Head[$idToAQTypeRules] === Dispatch, rulesFromDispatchFunc[$idToAQTypeRules], $idToAQTypeRules];
        existingID = Cases[rules, HoldPattern[id_ -> aqName] -> id];
        If[existingID === {},
            (* Not added yet. *)
            newID = $typeIndex++;
            $idToAQTypeRules = Dispatch[Append[rules, newID -> aqName]];
            newID,
        (* else *)
            First[existingID]
        ]
    ] 
idToAQType[id_Integer] := id /. $idToAQTypeRules

If[!ValueQ[$idToAsmNameRules],
    $idToAsmNameRules = {};
    $asmIndex = 1;
]

(* This direction doesn't need to be fast. *)
asmNameToID[asmName_String] :=
    Module[{rules, existingID, newID},
        (* Once the list of rules grows long enough, Dispatch starts creating an expression
            with head Dispatch, where the list of rules is the first arg.
        *)
        rules = If[Head[$idToAsmNameRules] === Dispatch, rulesFromDispatchFunc[$idToAsmNameRules], $idToAsmNameRules];
        existingID = Cases[rules, HoldPattern[id_ -> asmName] -> id];
        If[existingID === {},
            (* Not added yet. *)
            newID = $asmIndex++;
            $idToAsmNameRules = Dispatch[Append[rules, newID -> asmName]];
            newID,
        (* else *)
            First[existingID]
        ]
    ] 
idToAsmName[id_Integer] := id /. $idToAsmNameRules

Attributes[aqTypeNameFromStaticSymbol] = {HoldFirst}
Attributes[fieldNameFromStaticSymbol] = {HoldFirst}

clearPersistentData[] :=
    AbortProtect[
        Clear[lookupType];
        Clear[lookupAssembly];
        
        (* Note that we do not clear $asmIndex and $typeIndex. We want those to be unique
           throughout a kernel session, not unique throughout a .NET runtime life. In this way,
           NETAssembly and NETType expressions left over from a previous .NET session will
           be easily detectable as bogus (since we never reuse ids, the old ids will not
           have entries in the $idToAsmNameRules and $idToAQTypeRules tables).
        *)
        $idToAQTypeRules = {};
        $idToAsmNameRules = {};
        $netContexts = {};

        Clear[aqTypeNameFromInstance];
        
        Clear[hasIndexer];
        Clear[isNETStaticSymbol];
        Clear[isNonPrimitiveFieldOrSimpleProp];
        Clear[aqTypeNameFromStaticSymbol];
        Clear[fieldNameFromStaticSymbol];
    ]
    
            
End[]

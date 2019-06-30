
(* :Title: NETLink *)

(* :Context: NETLink` *)

(* :Author:
        Todd Gayley
        tgayley@wolfram.com
*)

(* :Package Version: 1.7 *)

(* :Mathematica Version: 5.0 *)
		     
(* :Copyright: NET/Link source code (c) 2003-2019, Wolfram Research, Inc. All rights reserved.

   Use is governed by the terms of the NET/Link license agreement, which can be found at
   www.wolfram.com/solutions/mathlink/netlink.
*)

(* :Discussion:
   NET/Link is a Mathematica enhancement that integrates Microsoft's .NET platform and Mathematica.
   You can use NET/Link to call .NET (specifically, the CLR) from Mathematica or call Mathematica from .NET.
   Find out more at www.wolfram.com/solutions/mathlink/netlink.

   NET/Link uses a special system wherein one package context (NETLink`) has its implementation
   split among a number of .m files. Each component file has its own private context, and also
   potentially introduces public symbols (in the NETLink` context) and so-called "package" symbols,
   where the term "package" comes from Java terminology, referring to symbols that are visible
   everywhere within the implementation of NET/Link, but not to clients.
   
   This file is generated automatically from a tool that processes the component .m files in the NETLink/Kernel
   directory. It contains only declarations, not implementation. Users who want to examine implementation
   details in the code can consult the appropriate .m files.
*)

(* :Keywords: .NET MathLink COM C# DLL VisualBasic VB *)



BeginPackage["NETLink`", {"JLink`"}]


(***************************************  Information Context  ******************************************)

(* Programmers can use these values (using their full context, as in NETLink`Information`$ReleaseNumber)
   to test version information about a user's .NET/Link installation.
*)

`Information`$VersionNumber = 1.7
`Information`$ReleaseNumber = 0
`Information`$Version = "NET/Link Version 1.7.0"
`Information`$CreationDate = {2019, 5, 19, 21, 7, 41.9161565}
`Information`$CreationID = 2019051901


(********************************  Usage Messages (Public NET/Link API)  **********************************)

(******** From InstallNET.m ********)
InstallNET::usage =
"InstallNET[] launches the .NET runtime and prepares it to be used from the Wolfram Language. Only one .NET runtime is ever launched; \
subsequent calls to InstallNET after the first have no effect."
UninstallNET::usage =
"UninstallNET[] shuts down the .NET runtime that was started by InstallNET. It is provided mainly for developers who are \
actively recompiling .NET types for use in the Wolfram Language and therefore need to shut down and restart the .NET runtime to reload \
the modified types. Users generally have no reason to call UninstallNET. The .NET runtime is a shared resource used by \
potentially many Wolfram Language programs. You should leave it running unless you are absolutely sure you need to shut it down."
ReinstallNET::usage =
"ReinstallNET[] is a convenience function that calls UninstallNET[] followed by InstallNET[]. See the usage messages for \
InstallNET and UninstallNET for more information."
NETLink::usage =
"NETLink[] returns the MathLink LinkObject that is used to communicate with the .NET/Link .NET runtime. It will return \
Null if .NET is not running."
NETUILink::usage =
"NETUILink[] returns the MathLink LinkObject used by calls to the Wolfram Language that originate from .NET user-interface actions, or Null if no such link is present."
(******** From CallNET.m ********)
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
(******** From NETBlock.m ********)
NETBlock::usage =
"NETBlock[expr] causes all new .NET objects returned to the Wolfram Language during the evaluation of expr to be released when expr \
finishes. It is an error to refer to such an object after NETBlock ends. See the usage message for ReleaseNETObject for more \
information. NETBlock only affects new objects, not additional references to ones that have previously been seen. NETBlock is \
a way to mark a set of objects as temporary so they can be automatically cleaned up on both the Wolfram Language and .NET sides."
BeginNETBlock::usage =
"BeginNETBlock[] and EndNETBlock[] are equivalent to the NETBlock function, except that they work across a larger span than \
the evaluation of a single expr. Every BeginNETBlock[] must have a paired EndNETBlock[]."
EndNETBlock::usage =
"BeginNETBlock[] and EndNETBlock[] are equivalent to the NETBlock function, except that they work across a larger span than \
the evaluation of a single expr. Every BeginNETBlock[] must have a paired EndNETBlock[]."
ReleaseNETObject::usage =
"ReleaseNETObject[netobject] tells the .NET memory-management system to forget any references to the specified NETObject \
that are being maintained solely for the sake of the Wolfram Language. The NETObject in the Wolfram Language is no longer valid after the call. \
You call ReleaseNETObject when you are completely finished with an object in the Wolfram Language, and you want to allow it to be \
garbage-collected in .NET."
KeepNETObject::usage =
"KeepNETObject[object] causes the specified object(s) not to be released when the current NETBlock ends. \
KeepNETObject allows an object to \"escape\" from the current NETBlock. It only has an effect if the object was in fact \
slated to be released by that block. The object is promoted to the \"release\" list of the next-enclosing NETBlock, \
if there is one. The object will be released when that block ends (unless you call KeepNETObject again in the outer block). \
KeepNETObject[object, Manual] causes the specified object to escape from all enclosing NETBlocks, meaning that the object \
will only be released if you manually call ReleaseNETObject."
LoadedNETObjects::usage =
"LoadedNETObjects[] returns a list of all the .NET objects that have been loaded into the current session."
LoadedNETTypes::usage =
"LoadedNETTypes[] returns a list of all the .NET types that have been loaded into the current session."
LoadedNETAssemblies::usage =
"LoadedNETAssemblies[] returns a list of all the .NET assemblies that have been loaded into the current session."
(******** From MakeNETObject.m ********)
MakeNETObject::usage =
"MakeNETObject[expr] constructs a .NET object that represents the given Wolfram Language value. It operates on numbers, \
True/False, strings, and lists of these, up to 3-deep. It is a shortcut to calling NETNew, and it is especially useful \
for arrays, as the array object can be created and initialized with values in a single call. MakeNETObject[expr, type] \
creates an object of the specified type from expr. Use the type argument to force a non-standard type. For example, \
MakeNETObject[{1,2,3}] will create an array of int (type System.Int32[]). If you want an array of Int16, you would use \
MakeNETObject[{1,2,3}, \"System.Int16[]\"]."
(******** From Reflection.m ********)
NETTypeInfo::usage =
"NETTypeInfo[type] prints information about the specified type, including its inheritance hierarchy, assembly name, \
and its public members (constructors, methods, properties, and so on.) The type argument can be a fully-qualified type name \
given as a string, or a NETType expression. NETTypeInfo[obj] prints information about the object's type. NETTypeInfo[assembly] \
prints information about the types in the assembly specified by the given NETAssembly expression. NETTypeInfo[type, members] \
prints information about only the specified members, which can be any of the following strings (or a list of them): \
\"Type\", \"Constructors\", \"Methods\", \"Fields\", \"Properties\", or \"Events\". When calling NETTypeInfo on a NETAssembly \
expression, the members argument must be any of the following strings (or a list of them): \"Classes\", \"Interfaces\", \
\"Structures\", \"Delegates\", or \"Enums\". NETTypeInfo[type, members, pattern] prints only the members whose names \
match the given string pattern. For example, \"Set*\" shows all members with names that start with Set."
LanguageSyntax::usage =
"LanguageSyntax is an option to NETTypeInfo that specifies which language syntax will be used to display the type information. \
The possible values are the strings \"CSharp\" (or just \"C#\") and \"VisualBasic\" (or just \"VB\"). The default is C#."
(******** From Delegates.m ********)
NETNewDelegate::usage =
"NETNewDelegate[type, func] creates a new instance of the specified .NET delegate type whose action is to call the named \
Wolfram Language function when triggered. The type argument can be a string name, a NETType expression, or a Type object. The \
func argument can be the name of a function as a symbol or string, or a pure function. The function you supply will be \
called with whatever arguments the delegate type takes. NETNewDelegate is a low-level function that is not often used; \
see AddEventHandler if you want to create a Wolfram Language callback triggered by some user interface action."
SendDelegateArguments::usage =
"SendDelegateArguments is an option to AddEventHandler and NETNewDelegate that specifies which of the delegate arguments you want \
to be passed to your Wolfram Language callback function. By default, all the arguments in the delegate's signature are sent to the \
Wolfram Language function assigned to the delegate. If you are not interested in some or all of the arguments, you can make the \
callback more efficient by using SendDelegateArguments to eliminate some of the arguments. Efficiency is generally only a concern \
for arguments that are objects, not primitive types like integers or strings. The default value is All, but you can use a list of \
numbers that represent argument indices to specify which arguments to send. For example, SendDelegateArguments -> {1,3} means to \
only send the first and third arguments. Use None or {} to specify that no arguments should be sent."
CallsUnshare::usage =
"CallsUnshare is an option to AddEventHandler and NETNewDelegate that specifies whether or not the Wolfram Language callback function \
assigned to the delegate calls UnshareKernel or UnshareFrontEnd. UnshareKernel and UnshareFrontEnd are advanced functions \
that most programmers will not call directly, preferring to use DoNETModeless instead, which encapsulates the use of these \
functions. However, if you are calling an Unshare function directly in a Wolfram Language callback from a delegate, you must use the \
CallsUnshare -> True option. The default is False."
WrapInNETBlock::usage =
"WrapInNETBlock is an option to AddEventHandler and NETNewDelegate that specifies whether or not the Wolfram Language callback function \
assigned to the delegate should be implicitly wrapped in NETBlock. The default is True, so that objects sent to callback \
functions or created within them are treated as temporary and released when the callback completes. If you need an object created \
in a callback function to persist in the Wolfram Language after the callback completes, use WrapInNETBlock -> False."
DefineNETDelegate::usage =
"DefineNETDelegate[name, returnType, parameterTypes] creates a new .NET delegate type with the given name, return type, \
and parameter types. This is a rarely-used function whose main use is for creating delegates for DLL function pointers. \
In such cases there is probably not an existing .NET delegate type that is suitable, so you need to create one. DefineNETDelegate \
simply lets you do this entirely in Wolfram Language code, without resorting to writing in C# or Visual Basic. You typically go \
on to call NETNewDelegate to create a new instance of the new delegate type."
AddEventHandler::usage =
"AddEventHandler[obj@eventName, func] assigns the specified Wolfram Language function to be called when the given event fires. \
You use AddEventHandler to wire up Wolfram Language callbacks for events in .NET user interfaces, like a button click. \
The func argument can be the name of a Wolfram Language function as a string or symbol, or a pure function. The function will be \
called with whatever arguments the event sends. You can also manually create a delegate using NETNewDelegate and pass that \
instead of a function for the second argument. AddEventHandler returns a delegate object. You can pass this delegate object \
to RemoveEventHandler to remove the callback function."
RemoveEventHandler::usage =
"RemoveEventHandler[obj@eventName, delegate] removes the specified delegate from the named event. The delegate object you pass \
must have been returned from a call to AddEventHandler for that same event."
(******** From DLL.m ********)
DefineDLLFunction::usage =
"DefineDLLFunction[\"funcName\", \"dllName\", returnType, argTypes] returns a Wolfram Language function that calls the \
specified function in the specified unmanaged DLL. The argsTypes argument is a list of type specifications for the arguments, and \
can be omitted if the function takes no arguments. The type specifications for argTypes and returnType are strings or, \
less commonly, NETType expressions. Strings can be given in C-style syntax (such as \"char*\"), C# syntax \
(\"string\"), Visual Basic .NET syntax (\"ByVal As String\") or by using many Windows API types (such as \"HWND\", \"DWORD\", \
\"BOOL\", and so on.) Priority is given to the C interpretation of type names, so char and long have their meanings in C \
(1 and 4 bytes, respectively), not C#. You need to give the full pathname to the DLL if it is not located in a standard \
location (standard locations are a directory on your system PATH or a DLL subdirectory in a Wolfram System application directory, \
such as $InstallationDirectory\\AddOns\\Applications\\SomeApp\\DLL). DefineDLLFunction[\"declaration\"] lets you write a full \
C#-syntax 'extern' function declaration. Use this form when you need to write a complex function declaration that requires \
features not available using options to DefineDLLFunction, such as specific \"MarshalAs\" atributes on each of the parameters."
ReferencedAssemblies::usage =
"ReferencedAssemblies is an option to DefineDLLFunction that specifies assemblies needed in your function declaration. \
For example, if your DLL function involves a type from another assembly, such as System.Drawing.Rectangle from the \
System.Drawing assembly, you would specify ReferencedAssemblies->{\"System.Drawing.dll\"}. Note that you should use the \
actual filename of the assembly, not its display name (which would be just \"System.Drawing\" in this example)."
MarshalStringsAs::usage =
"MarshalStringsAs is an option to DefineDLLFunction that specifies how string arguments should be marshaled into the DLL function. \
This applies to any arguments that are mapped to the System.String class, which includes types specified in your declaration \
as \"char*\", \"string\", or \"ByVal As String\". The possible values are \"ANSI\", \"Unicode\", and Automatic. The default is \
\"ANSI\", meaning that strings will be sent as single-byte C-style strings. This is appropriate for most DLL functions, which \
generally expect C-style strings. \"Unicode\" means to send strings as 2-byte Unicode strings. Use this if you know the function \
expects 2-byte strings (e.g., if the type name in the C prototype is wchar_t* ). The Automatic setting picks the platform \
default (\"Unicode\" on Windows NT/2000/XP, \"ANSI\" on 98/ME). Automatic should rarely be used, as it is intended mainly for \
certain Windows API functions that automatically switch behaviors on different versions of Windows."
CallingConvention::usage =
"CallingConvention is an option to DefineDLLFunction that specifies what calling convention the DLL function uses. The possible \
values are \"StdCall\", \"CDecl\", \"ThisCall\", \"WinApi\", and Automatic. The string values for this option are not case \
sensitive. The default is Automatic, which means use the platform default (\"StdCall\" on all platforms except Windows CE, \
which is not supported by .NET/Link). \
Most DLL funtions use the \"StdCall\" convention. For more information on these values, see the .NET Framework documentation \
for the System.Runtime.InteropServices.CallingConvention enumeration."
(******** From UI.m ********)
DoNETModal::usage =
"DoNETModal[form] displays the specified .NET form in the foreground and does not return until the form window is closed. \
DoNETModal[form, expr] evaluates expr just before the form is closed and returns the result. \
Typically, DoNETModal is used to implement a modal dialog box that needs to interact with the Wolfram Language while the dialog box \
is active, or one that returns a result to the Wolfram Language when it is dismissed."
DoNETModeless::usage =
"DoNETModeless[form] displays the specified .NET form in the foreground and then returns. The form can interact with the Wolfram Language \
while it is active, but it will not interfere with normal use of the Wolfram Language via the notebook front end. That is what is meant \
by the \"modeless\" state--the form does not monopolize the Wolfram Language kernel while it is active."
EndNETModal::usage =
"EndNETModal[] causes DoNETModal to return. It is rarely called directly by programmers. When a form is activated with DoNETModal, \
.NET/Link arranges for EndNETModal[] to be called automatically when the form is closed. In advanced scenarios, programmers might \
want to call EndNETModal directly."
ShowNETWindow::usage =
"ShowNETWindow[form] displays the specified .NET form in the foreground. It is used internally by DoNETModal and DoNETModeless, \
so programmers using either of those functions will not need to call it. You can call ShowNETWindow to activate a form that does \
not need to interact with the kernel (and therefore does not need DoNETModal or DoNETModeless), or to ensure that a form that has \
previously been displayed is brought in front of any notebook windows and un-minimized if necessary."
FormStartPosition::usage =
"FormStartPosition is an option to DoNETModal, DoNETModeless, ShowNETWindow, and ShowNETConsole that controls the location on \
screen of the form when it first appears. The possible values are Center (the form will be centered on the screen), Automatic \
(the form will have the Windows default location), and Manual (the form will appear at a location specified elsewhere, for example, \
by setting the form's Location property). The default value is Center. This option only controls the location of the form \
when it is first made visible."
ActivateWindow::usage =
"ActivateWindow is an option to DoNETModeless that specifies whether to make the window visible. The default is True. Set it to \
False if you want to enter the modeless state but not display the window until a later time."
ShowNETConsole::usage =
"ShowNETConsole[] displays the .NET console window and begins capturing output sent to the Console.Out and Console.Error \
streams. Anything written to these streams before ShowNETConsole is first called will not appear, and closing the console window \
will stop capturing the streams (until ShowNETConsole is called again). ShowNETConsole[\"stdout\"] captures only Console.Out, \
and ShowNETConsole[\"stderr\"] captures only Console.Error."
(******** From ComplexType.m ********)
GetComplexType::usage =
"GetComplexType[] returns the .NET type that is currently mapped to Wolfram Language Complex numbers. This is the \
type that will be used when Complex numbers are sent to .NET, and objects of this type will be converted to Complex \
when sent to the Wolfram Language. It returns Null when no type has yet been designated via SetComplexType."
SetComplexType::usage =
"SetComplexType[type] tells .NET/Link to map the specified type to Wolfram Language Complex numbers. This is the \
type that will be used when Complex numbers are sent to .NET, and objects of this type will be converted to Complex \
when sent to the Wolfram Language. The type argument can be specified as a string or as a NETType expression obtained \
from LoadNETType."
(******** From Exceptions.m ********)
GetNETException::usage =
"GetNETException[] returns the .NET exception object that was thrown in the most recent call from the Wolfram Language to .NET. \
It returns Null if no exception was thrown in the most recent call. You can use GetNETException in conjunction with \
$NETExceptionHandler to implement a custom exception-handling scheme in the Wolfram Language."
$NETExceptionHandler::usage =
"$NETExceptionHandler allows you to control how exceptions thrown in .NET are handled in the Wolfram Language. The default behavior \
is for exceptions to appear as messages in the Wolfram Language. If you want to override this behavior (e.g., to temporarily \
silence messages from exceptions), assign a value to $NETExceptionHandler. The value of $NETExceptionHandler is treated as \
a function that will be passed 3 arguments: the symbol associated with the message (usually the symbol NET), \
the message tag (the string \"netexcptn\" for a typical exception or \"netpexcptn\" for an exception generated \
by a \"manual return\" method where the exception occurs after the method has manually sent its result back to the Wolfram Language), \
and the descriptive string of text associated with the message. You will typically set $NETExceptionHandler within a Block \
so that its effect will be limited to a precisely defined segment of code, as in the following example that silences messages: \
Block[{$NETExceptionHandler = Null&}, obj@Method[]]. You can use GetNETException[] within your handler function to obtain \
the actual .NET exception object that was thrown."
NET::usage =
"NET is only used as a generic symbol for some messages."
(******** From Utils.m ********)
FixCRLF::usage =
"FixCRLF[\"str\"] changes the linefeeds in the given string to the CR/LF Windows convention. Use this function on strings that \
are generated in the Wolfram Language and need to be placed into text boxes or other .NET GUI elements. Wolfram Language strings use just the \\n \
character (ASCII 10) for newlines, and these characters generally show up as rectangles in Windows text-based controls."
(******** From JLinkCommon.m ********)
Off[General::shdw]
NETLink`InstanceOf = JLink`InstanceOf
JLink`InstanceOf::usage = JLink`InstanceOf::usage <>
"\n\nInstanceOf[netobject, nettype] gives True if netobject is an instance of the type nettype, or a subtype. \
Otherwise, it returns False. It mimics the behavior of the C# language's 'is' operator. The nettype argument can \
be either the fully-qualified class or interface name as a string, or a NETType expression."
NETLink`InstanceOf::usage = JLink`InstanceOf::usage
NETLink`SameObjectQ = JLink`SameObjectQ
JLink`SameObjectQ::usage = JLink`SameObjectQ::usage <>
"\n\nSameObjectQ[netobject1, netobject1] returns True if and only if the NETObject expressions netobject1 and netobject2 \
refer to the same .NET object. It is a shortcut to calling Object`ReferenceEquals[netobject1, netobject2]."
NETLink`SameObjectQ::usage = JLink`SameObjectQ::usage
On[General::shdw]
(******** From COM.m ********)
CreateCOMObject::usage =
"CreateCOMObject[str] creates a COM object specified by the string str, which can be either a ProgID (such as \"Excel.Application\") \
or a CLSID (such as \"{8E27C92B-1264-101C-8A2F-040224009C02}\"). CreateCOMObject is analogous to the COM API function CoCreateInstance."
GetActiveCOMObject::usage =
"GetActiveCOMObject[str] acquires an already-running COM object specified by the string str, which can be either a ProgID \
(such as \"Excel.Application\") or a CLSID (such as \"{8E27C92B-1264-101C-8A2F-040224009C02}\"). GetActiveCOMObject is analogous \
to the COM API function GetActiveObject."
ReleaseCOMObject::usage =
"ReleaseCOMObject[obj] releases COM resources held by the specified .NET object. Although any COM resources will be \
released when the .NET object is garbage-collected, it is often desirable to force their release explicitly. Each call to \
ReleaseCOMObject decrements the reference count on the COM resources held by the object. The resources will be freed when \
the reference count goes to 0 (or the .NET object is garbage-collected). ReleaseCOMObject returns the new reference count on the \
COM resources, or a list of these counts if it was passed a list of objects. ReleaseCOMObject should not be \
confused with ReleaseNETObject. ReleaseNETObject allows the .NET object to be garbage-collected, but does not force this to \
happen in a timely manner. ReleaseCOMObject can be used to force the immediate release of the COM resources held by the object."
CastCOMObject::usage = 
"CastCOMObject is deprecated. Use the more general CastNETObject instead."
LoadCOMTypeLibrary::usage =
"LoadCOMTypeLibrary[typeLibPath] creates a so-called \"interop\" assembly from the named type library and loads that assembly. \
Once a type library has been loaded in this way, all its types will have managed equivalents created for them, so you can program \
with these types as if they were native .NET types. LoadCOMTypeLibrary is the programmatic equivalent of running the tlbimp.exe \
tool that is part of the .NET Framework SDK. The assembly can optionally be saved to disk (using the SaveAssemblyAs option) so that \
you do not have to call LoadCOMTypeLibrary in the future. If you plan to do serious work with COM objects described in a given type \
library, it is recommended that you use LoadCOMTypeLibrary or the tlbimp.exe tool to create an interop assembly and then use that \
assembly."
SafeArrayAsArray::usage =
"SafeArrayAsArray is an option to LoadCOMTypeLibrary that specifies whether to import all SAFEARRAY's as System.Array rather than \
a typed, single dimensional managed array. The default is False. See the .NET Framework documentation for the \
System.Runtime.InteropServices.TypeLibImporterFlags enumeration for more details on this advanced option."
SaveAssemblyAs::usage =
"SaveAssemblyAs is an option to LoadCOMTypeLibrary that allows you to specify a file name into which to write the interop assembly \
that gets generated. LoadCOMTypeLibrary can be time-consuming for large type libraries, so it is useful to save the generated \
assembly in a file. It can then be loaded directly, bypassing future calls to LoadCOMTypeLibrary. You can specify a directory name only \
and get a default name for the assembly."


(***********************************  End of Public NET/Link API  *************************************)


(********************************  Package-Visiblity Declarations  **********************************)

(* These functions are not public, and not intended for users to call. *)

Begin["`Package`"]

(******** From InstallNET.m ********)
$inPreemptiveCallFromNET
(******** From CallNET.m ********)
(* Used in netlinkExternalCall to direct output to the appropriate link (NETLink[] or NETLink[]). *)
getActiveNETLink
$inExternalCall
clearNETDefs
callAllUnloadTypeMethods
(* These are called directly from .NET. *)
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
(******** From NET.m ********)
(* These n* functions are the ones whose definitions are created by Install.cs. *)
(* They implement the special set of CallPackets used to call into .NET. *)
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
(******** From NETBlock.m ********)
addToNETBlock
resetNETBlock
findAliases
(******** From Delegates.m ********)
(* Called directly from .NET. *)
delegateCallbackWrapper
(******** From DLL.m ********)
fixType
$csharpToNETTypeRules
$vbToNETTypeRules
$netTypeToCSharpRules
$netTypeToVBRules
(******** From Exceptions.m ********)
$internalNETExceptionHandler
(* Used for changing the generic message NET::netexcptn to SomeSymbol::netexcptn. Use this when a function in .NET/Link
   uses the CallPacket mechanism but is not a user-level ctor/method call. For example, CreateCOMObject is implemented with
   its own CallPacket and can generate exceptions with messages that come from inside COM. It's convenient to let these
   exceptions percolate up to top level as NET::netexcptn, and then associate them with CreateCOMObject as CreateCOMObject::netexcptn.
*)
associateMessageWithSymbol
(* All these are called only from .NET *)
prepareForManualReturn
handleException
manualException
specialException
(******** From Utils.m ********)
osIsWindows
osIsMacOSX
isPreemptiveKernel
isServiceFrontEnd
contextIndependentOptions
filterOptions
preemptProtect
(******** From MathKernel.m ********)
computeWrapper
(******** From COM.m ********)
isCOMNonPrimitiveFieldOrSimpleProp


$netlinkDir = DirectoryName[$InputFileName]


End[]   (* Ends `Package` context *)

(* Make the Package` symbols visible to all implementation files as they are read in. *)
AppendTo[$ContextPath, "NETLink`Package`"]


(********************************  Read in the Implementation Files  **********************************)

(
    Get[ToFileName[#, "InstallNET.m"]];
    Get[ToFileName[#, "CallNET.m"]];
    Get[ToFileName[#, "NET.m"]];
    Get[ToFileName[#, "NETBlock.m"]];
    Get[ToFileName[#, "MakeNETObject.m"]];
    Get[ToFileName[#, "Reflection.m"]];
    Get[ToFileName[#, "Delegates.m"]];
    Get[ToFileName[#, "DLL.m"]];
    Get[ToFileName[#, "UI.m"]];
    Get[ToFileName[#, "ComplexType.m"]];
    Get[ToFileName[#, "Exceptions.m"]];
    Get[ToFileName[#, "Utils.m"]];
    Get[ToFileName[#, "JLinkCommon.m"]];
    Get[ToFileName[#, "MathKernel.m"]];
    Get[ToFileName[#, "COM.m"]];
)& @ ToFileName[$netlinkDir, "Kernel"]


EndPackage[]

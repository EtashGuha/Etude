(* Wolfram Language Package *)

BeginPackage["TINSLink`Libraries`"]

initializeLibrariesTINSLink::usage="loads the libraries for tinslink";

(* Functions exported by libtinslink *)

iStartPacketCapture;
iStopPacketCapture;
iStopAllPacketCaptures;
iGetPacketSpeed;

iGetActivePacketCaptures;

iImportPacketCapture;
iTestInterface;

iGetDefaultInterface;
iGetAllInterfaces;
iGetAllInterfacesCryptic;

Begin["Private`"]

initializeLibrariesTINSLink[] := Module[
    {
        tinsLib = FindLibrary["libtins"],
        tinsLinkLib = FindLibrary["libtinslink"],
        pcapLib
    },
    (*on unix we have to load libpcap and libtins before loading anything else*)
    If[$OperatingSystem === "Unix",
        pcapLib = FindLibrary["libpcap"];
        LibraryLoad[pcapLib];
        LibraryLoad[tinsLib]
    ];

    iGetPacketSpeed = LibraryFunctionLoad[tinsLinkLib, "GetPacketSpeed", {"UTF8String"}, {Integer, 1}];
    iStartPacketCapture = LibraryFunctionLoad[tinsLinkLib, "StartPacketCapture", {"UTF8String", "UTF8String"}, "Void"];
    iStopPacketCapture = LibraryFunctionLoad[tinsLinkLib, "StopPacketCapture", LinkObject, LinkObject];
    iStopAllPacketCaptures = LibraryFunctionLoad[tinsLinkLib, "StopAllPacketCaptures", LinkObject, LinkObject];

    iGetActivePacketCaptures = LibraryFunctionLoad[tinsLinkLib, "GetActivePacketCaptures", LinkObject, LinkObject];

    iImportPacketCapture = LibraryFunctionLoad[tinsLinkLib, "ImportPacketCapture", LinkObject, LinkObject];

    iGetDefaultInterface = LibraryFunctionLoad[tinsLinkLib, "GetDefaultInterface", LinkObject, LinkObject];
    iGetAllInterfaces = LibraryFunctionLoad[tinsLinkLib, "GetAllInterfaces", LinkObject, LinkObject];

    iGetAllInterfacesCryptic = LibraryFunctionLoad[tinsLinkLib, "GetAllInterfacesCryptic", LinkObject, LinkObject];

    iTestInterface = LibraryFunctionLoad[tinsLinkLib, "TestInterface", {}, Integer];

]

End[]

EndPackage[]

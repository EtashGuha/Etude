(* ::Package:: *)

System`SystemTest

TestReport

BeginPackage["SystemTest`"]

Begin["`Private`"]

networkTests := Join[{
		Hold@VerificationTest[$NetworkConnected,True,TestID->"Network connection"],
		Hold@VerificationTest[PacletManager`$AllowInternet,True,TestID->"Internet access allowed"]
	}, 
	resourceSystemTests,
	dataDropTests,
	pacletTests,
	cloudSystemTests,
	wolframAlphaTests,
	geoTests
];

fileSystemTests := {
	(* file location tests *)
	Hold@VerificationTest[$DefaultLocalBase,_String,SameTest->MatchQ,TestID->"$DefaultLocalBase"],
	Hold@VerificationTest[$InstallationDirectory,_String,SameTest->MatchQ,TestID->"$InstallationDirectory"],
	Hold@VerificationTest[$UserBaseDirectory,_String,SameTest->MatchQ,TestID->"$UserBaseDirectory"],
	Hold@VerificationTest[$BaseDirectory,_String,SameTest->MatchQ,TestID->"$BaseDirectory"]
}

resourceSystemTests := {
	Hold@VerificationTest[$ResourceSystemBase,"https://www.wolframcloud.com/objects/resourcesystem/api/1.0",TestID->"$ResourceSystemBase"],
	Hold@VerificationTest[ResourceUpdate[ResourceObject["Fireballs and Bolides"]],_ResourceObject,{_},SameTest->MatchQ,TestID->"ResourceObject retrieval"]
}

dataDropTests := {
	Hold@VerificationTest[Databins[],_List,SameTest->MatchQ,TestID->"DataDrop framework connection"]
}

pacletTests := {
	Hold@VerificationTest[PacletManager`$PacletSite,"http://pacletserver.wolfram.com",SameTest->MatchQ,TestID->"$PacletSite"],
	Hold@VerificationTest[PacletManager`PacletSiteUpdate[PacletManager`$PacletSite],_PacletManager`PacletSite,SameTest->MatchQ,TestID->"Paclet Manager connection"]
}

cloudSystemTests := {
	Hold@VerificationTest[$CloudConnected,True,TestID->"Wolfram Cloud connection"],
	Hold@VerificationTest[$CloudBase,"https://www.wolframcloud.com/",TestID->"Default $CloudBase"],
	Hold@VerificationTest[$CloudCreditsAvailable>0,True,TestID->"Available cloud credits"]
}

wolframAlphaTests = {
	Hold@VerificationTest[WolframAlpha["1","Result"],1,TestID->"Wolfram|Alpha connection"]
}

geoTests := {
	Hold@VerificationTest[$GeoLocation,_GeoPosition,SameTest->MatchQ,TestID->"$GeoLocation"],
	Hold@VerificationTest[GeoImage[Here],_Image,SameTest->MatchQ,TestID->"GeoImage"]
}

deviceTests := {
	Hold@VerificationTest[$DefaultAudioInputDevice,Except[None],SameTest->MatchQ,TestID->"Audio input device"],
	Hold@VerificationTest[$DefaultAudioOutputDevice,Except[None],SameTest->MatchQ,TestID->"Audio output device"],
	Hold@VerificationTest[$DefaultImagingDevice,Except[None],SameTest->MatchQ,TestID->"Image capture device"]
}

linkTests := {
	Hold@VerificationTest[$Linked,True,TestID->"Kernel-Frontend connection"]
}

serviceTests := {
	Hold@VerificationTest[$Services,Except[{}],SameTest->MatchQ,TestID->"$Services"],
	Hold@VerificationTest[$ServiceCreditsAvailable>0,True,TestID->"Available service credits"]
}

importExportTests := {
	Hold@VerificationTest[$ImportFormats,Except[{}],SameTest->MatchQ,TestID->"$ImportFormats"],
	Hold@VerificationTest[$ExportFormats,Except[{}],SameTest->MatchQ,TestID->"$ExportFormats"]
}

formatDataset[tr_TestReportObject] := (
	Dataset[(AssociationThread[{"TestID", "Outcome", "Input", "ActualOutput", "AbsoluteTimeUsed"}, (#/@ {"TestID", "Outcome", "Input", "ActualOutput", "AbsoluteTimeUsed"})]) & /@ tr["TestResults"]]
)

systemTest["Network"] := formatDataset[TestReport[ReleaseHold@networkTests]]
systemTest["FileSystem"] := formatDataset[TestReport[ReleaseHold@fileSystemTests]]
systemTest["ResourceSystem"] := formatDataset[TestReport[ReleaseHold@resourceSystemTests]]
systemTest["DataDrop"] := formatDataset[TestReport[ReleaseHold@dataDropTests]]
systemTest["Paclets"] := formatDataset[TestReport[ReleaseHold@pacletTests]]
systemTest["Devices"] := formatDataset[TestReport[ReleaseHold@deviceTests]]
systemTest["CloudSystem"] := formatDataset[TestReport[ReleaseHold@cloudSystemTests]]
systemTest["WolframAlpha"] := formatDataset[TestReport[ReleaseHold@wolframAlphaTests]]
systemTest["Links"] := formatDataset[TestReport[ReleaseHold@linkTests]]
systemTest["Services"] := formatDataset[TestReport[ReleaseHold@serviceTests]]
systemTest["GeoGraphics"] := formatDataset[TestReport[ReleaseHold@geoTests]]
systemTest["ImportExport"] := formatDataset[TestReport[ReleaseHold@importExportTests]]

SystemTest::nocomp = "`1` is not a component recognized by SystemTest"
systemTest[missing_] := Message[SystemTest::nocomp, missing]

systemTest["TestReport"] := 
TestReport[ReleaseHold[Join[
	networkTests,
	fileSystemTests,
	deviceTests,
	linkTests,
	serviceTests,
	importExportTests
]]]

systemTest[] := TabView[
	AssociationThread[{"Network", "FileSystem", "Devices", "Links", "Services", "ImportExport"}, 
		SystemTest /@ {"Network", "FileSystem", "Devices", "Links", "Services", "ImportExport"}
	]
]

System`SystemTest[] := SystemTest`Private`systemTest[];
System`SystemTest[keys_] := SystemTest`Private`systemTest[keys];
System`SystemTest["Properties"] := {"Network", "FileSystem", "Devices", "Links", "Services", "ImportExport"};

End[]
EndPackage[]

(* ::Package:: *)

(* Mathematica Package *)

(* Created by the Wolfram Workbench Jan 30, 2012 *)

Language`SetMutationHandler[AudioFileStreamTools`MetaTags, AudioFileStreamTools`Private`MetaTagsHandler];

BeginPackage["AudioFileStreamTools`"];
(* Exported symbols added here with SymbolName::usage *)

AudioFileStreamTools`FileStreamOpenRead::usage = "AudioFileStreamTools`FileStreamOpenRead[ fileName] opens a read stream to the file specified. An AudioFileStreamTools`FileSteamObject is returned.";
AudioFileStreamTools`FileStreamReadN::usage = "AudioFileStreamTools`FileStreamReadN[ obj, n] reads n frames from the current position of obj and moves the read pointer accordingly.";
AudioFileStreamTools`FileStreamClose::usage = "AudioFileStreamTools`FileStreamClose[ obj] closes the specified AudioFileStreamTools`FileSteamObject.";
AudioFileStreamTools`FileStreamObject::usage = "AudioFileStreamTools`FileStreamObject represents an audio file stream.";
AudioFileStreamTools`FileStreamOpenWrite::usage = "AudioFileStreamTools`FileStreamOpenWrite[ fileName] opens a write stream to the file specified. An AudioFileStreamTools`FileSteamObject is returned.";
AudioFileStreamTools`FileStreamWrite::usage = "AudioFileStreamTools`FileStreamWrite[ obj, audioData] writes the information in the audioData to the specified file stream object.";
AudioFileStreamTools`FileStreamSetReadPosition::usage = "AudioFileStreamTools`FileStreamSetReadPosition[ obj, pos] sets the read position of obj. ";
AudioFileStreamTools`FileStreamGetReadPosition::usage = "AudioFileStreamTools`FileStreamGetReadPosition[ obj] returns the read position of obj.";
AudioFileStreamTools`FileStreamGetMetaInformation::usage = "AudioFileStreamTools`FileStreamGetMetaInformation[ obj] returns the meta information for a stream.";
AudioFileStreamTools`FileStreamGetStreamInformation::usage = "AudioFileStreamTools`FileStreamGetStreamInformation[ obj] returns the stream information for a stream.";

AudioFileStreamTools`InternetStreamReadN::usage = "AudioFileStreamTools`InternetStreamReadN[ obj, n] reads n frames from the current position of obj and moves the read pointer accordingly.";
AudioFileStreamTools`InternetStreamClose::usage = "AudioFileStreamTools`InternetStreamClose[ obj] closes the specified AudioFileStreamTools`FileSteamObject.";
AudioFileStreamTools`InternetStreamSetReadPosition::usage = "AudioFileStreamTools`InternetStreamSetReadPosition[ obj, pos] sets the read position of obj. ";
AudioFileStreamTools`InternetStreamGetReadPosition::usage = "AudioFileStreamTools`InternetStreamGetReadPosition[ obj] returns the read position of obj.";
AudioFileStreamTools`InternetStreamGetMetaInformation::usage = "AudioFileStreamTools`InternetStreamGetMetaInformation[ obj] returns the meta information for a stream.";
AudioFileStreamTools`InternetStreamGetStreamInformation::usage = "AudioFileStreamTools`InternetStreamGetStreamInformation[ obj] returns the stream information for a stream.";

AudioFileStreamTools`InternetStreamOpenRead::usage = "AudioFileStreamTools`InternetStreamOpenRead[ url] or AudioFileStreamTools`FileStreamOpenRead[ url, func] opens a read stream to the url specified. An AudioFileStreamTools`FileSteamObject is returned.";
AudioFileStreamTools`InternetStreamGetBufferedRange::usage = "AudioFileStreamTools`InternetStreamGetBufferedRange[ obj] returns the range of buffered audio frames as {startPosition, endPosition}";
AudioFileStreamTools`InternetStreamDownloadStatus::usage = "AudioFileStreamTools`InternetStreamDownloadStatus[ obj] returns the current status of the download, either \"InProgress\", \"Complete\", or \"Aborted\"";
AudioFileStreamTools`InternetStreamDownloadPercent::usage = "AudioFileStreamTools`InternetStreamDownloadPercent[ obj] returns the current percentage of the download as a Real from 0 to 1.";

AudioFileStreamTools`MetaTags::usage = "AudioFileStreamTools`MetaTags[obj] accesses the MetaData associated with an open FileStreamObject.";

(* Messages *)

AudioFileStreamTools`FileStreamOpenRead::openreadfail =
	AudioFileStreamTools`InternetStreamOpenRead::openreadfail = "The format of file \"`1`\" is not supported, or the file is empty.";

AudioFileStreamTools`FileStreamOpenRead::invalidcontainer =
	AudioFileStreamTools`InternetStreamOpenRead::invcontainer = "The \"ContainerType\" \"`1`\" is invalid.";

AudioFileStreamTools`FileStreamOpenRead::setdelopts =
	AudioFileStreamTools`FileStreamOpenWrite::setdelopts =
	AudioFileStreamTools`InternetStreamOpenRead::setdelopts = "The file deletion options (\"DeleteFileOnClose\" and \"DeleteFileOnExit\") have already been set for this file.";

AudioFileStreamTools`FileStreamOpenRead::streamtypeconflict =
	AudioFileStreamTools`InternetStreamOpenRead::streamtype = "Cannot open \"`3`\" as a \"`1`\" stream. It is currently open as a \"`2`\" stream.";

AudioFileStreamTools`FileStreamOpenWrite::nowriteperm =
	AudioFileStreamTools`InternetStreamOpenRead::nowriteperm = "Unable to open the file \"`1`\" for writing. Please check your access permissions for the file or directory.";

AudioFileStreamTools`FileStreamReadN::invalidargs =
	AudioFileStreamTools`FileStreamClose::invalidargs =
	AudioFileStreamTools`FileStreamGetReadPosition::invalidargs =
	AudioFileStreamTools`FileStreamSetReadPosition::invalidargs =
	AudioFileStreamTools`FileStreamGetMetaInformation::invalidargs =
	AudioFileStreamTools`FileStreamGetStreamInformation::invalidargs =
	AudioFileStreamTools`FileStreamOpenWrite::invalidargs =
	AudioFileStreamTools`FileStreamWrite::invalidargs =
	AudioFileStreamTools`InternetStreamOpenRead::invalidargs =
	AudioFileStreamTools`InternetStreamGetBufferedRange::invalidargs =
	AudioFileStreamTools`InternetStreamOpenRead::invalidargs =
	AudioFileStreamTools`InternetStreamOpenRead::invalidargs = "The arguments entered for this function are invalid.";

AudioFileStreamTools`FileStreamOpenRead::nofile = "The file \"`1`\" does not exist.";
AudioFileStreamTools`FileStreamOpenWrite::openwritefail = "Failed to open file for writing due to invalid arguments for the specified file format."; (* TODO: Need different error types returned from C *)
AudioFileStreamTools`FileStreamOpenWrite::currentlyopen = "Failed to open \"`1`\" for writing. The file is currently open with another stream.";
AudioFileStreamTools`FileStreamGetMetaInformation::noinfo = "\"`1`\" metainformation does not exist for this stream.";
AudioFileStreamTools`FileStreamGetStreamInformation::noinfo = "\"`1`\" stream information does not exist for this stream.";
AudioFileStreamTools`FileStreamWrite::invaliddimensions = "The number of dimensions of the audio data is invalid.";
AudioFileStreamTools`FileStreamWrite::dimensionmismatch = "The number of dimensions of the audio data does not match the number of channels for stream `1`.";
AudioFileStreamTools`FileStreamWrite::containermismatch = "Audio data container type does not match the stream's data container type, `1`.";
AudioFileStreamTools`FileStreamWrite::invalidtype = "The audio data type is not valid.";
AudioFileStreamTools`FileStreamSetReadPosition::invalidposition = "The position `1` is invalid. It must be between 1 and the EndOfFile inclusive, where the EndOfFile is the \"FrameCount\"+1.";
AudioFileStreamTools`FileStreamSetReadPosition::stmrng = "Cannot set the current point in stream `1` to position `2`. The requested position exceeds the current number of frames in the stream.";
AudioFileStreamTools`FileStreamReadN::reachedendoffile = "Attempted to read past the current EndOfFile. The data output has been truncated.";
AudioFileStreamTools`FileStreamReadN::positionpastendoffile = "The current read position is past EndOfFile. No data could be read.";
AudioFileStreamTools`FileStreamReadN::numframesoutofbounds = "Attempted to read an invalid number of frames. The number of frames must be between 1 and 2147483647 inclusive.";

AudioFileStreamTools`InternetStreamOpenRead::filenotfound = "The file located at \"`1`\" was not found. Check that the URL is valid and that you have permission to access it.";
AudioFileStreamTools`InternetStreamOpenRead::forbidden = "The server returned a \"403 Forbidden\" response to the request for \"`1`\". Check that the URL is valid and that you have permission to access it.";
AudioFileStreamTools`InternetStreamOpenRead::authenticationrequired = "The server returned a \"401 Unauthorized \" response to the request for \"`1`\". Check that you have supplied the necessary credentials, that the URL is valid, and that you have permission to access it.";
AudioFileStreamTools`InternetStreamOpenRead::servernotfound = "Failed to resolve the server for the request of \"`1`\". Check that the URL is valid.";
AudioFileStreamTools`InternetStreamOpenRead::invalidsslcertificate = "The SSL certificate for the server hosting \"`1`\" is invalid. To attempt to connect using the invalid certificate, set the following option: \"VerifySSLCertificate\" -> False";
AudioFileStreamTools`InternetStreamOpenRead::unknownerror = "An unknown error occurred when attempting to access \"`1`\" and the connection attempt was aborted. Check that the URL is valid and that you have permission to access it.";
AudioFileStreamTools`InternetStreamOpenRead::unsupportedprotocol = "The protocol specified by \"`1`\" is not supported. Supported procols are HTTP and HTTPS.";
AudioFileStreamTools`InternetStreamOpenRead::timedout = "The connection attempt timed out when attempting to access \"`1`\". Check that the URL is valid and that you have permission to access it.";
AudioFileStreamTools`InternetStreamOpenRead::couldnotconnect = "Could not connect to the server for the request \"`1`\". Check that the URL is valid and that your Internet connection is functioning correctly.";
AudioFileStreamTools`InternetStreamOpenRead::readerror = "An error occured when reading data from \"`1`\".";
AudioFileStreamTools`InternetStreamOpenRead::filepathignored = "The \"FilePath\" option, \"`1`\", when opening the URL, \"`2`\", was ignored since this URL is already opened with another stream. This stream is a reference to the original stream.";
AudioFileStreamTools`InternetStreamOpenRead::openreadfailfilepath = "Failed to open the stream to the URL, \"`2`\". The \"FilePath\", \"`1`\", is in use by another URL.";
AudioFileStreamTools`InternetStreamOpenRead::authfailed = "Authentication attempts failed.";
AudioFileStreamTools`InternetStreamOpenRead::filepathnonedeprecated = "DEPRECATED: Specifying no \"FilePath\" option, or \"FilePath\" -> None is deprecated and no longer supported. The stream may not function as expected.";

AudioFileStreamTools`InternetStreamOpenRead::proxyauthenticationrequired = "The proxy server, \"`1`\", returned a \"407 Proxy authentication required \" response. Check that you have supplied the necessary credentials, that the proxy server is valid, and that you have permission to access it.";
AudioFileStreamTools`InternetStreamOpenRead::proxyservernotfound = "Failed to resolve the proxy server \"`1`\". Check that the URL is valid.";
AudioFileStreamTools`InternetStreamOpenRead::proxyunknownerror = "An unknown error occurred when attempting to access the proxy server \"`1`\" and the connection attempt was aborted. Check that the URL is valid and that you have permission to access it.";
AudioFileStreamTools`InternetStreamOpenRead::proxyinvalidsslcertificate = "The SSL certificate for the proxy server \"`1`\" is invalid. To attempt to connect using the invalid certificate, set the following option: \"VerifySSLCertificate\" -> False";
AudioFileStreamTools`InternetStreamOpenRead::proxyunsupportedprotocol = "The connection attempt to the proxy server, \"`1`\", failed. The specified protocol in the request URL is not supported. Supported procols are HTTP and HTTPS.";
AudioFileStreamTools`InternetStreamOpenRead::proxytimedout = "The connection attempt timed out when attempting to access the proxy server, \"`1`\". Check that the URL is valid and that you have permission to access it.";
AudioFileStreamTools`InternetStreamOpenRead::proxycouldnotconnect = "Could not connect to the proxy server, \"`1`\". Check that the URL is valid and that your Internet connection is functioning correctly.";
AudioFileStreamTools`InternetStreamOpenRead::proxyreaderror = "An error occured when reading data from the proxy server, \"`1`\".";

AudioFileStreamTools`InternetStreamOpenRead::invalidsoundcloudurl = "The SoundCloud URL \"`1`\" is not valid and could not be parsed.";
AudioFileStreamTools`InternetStreamOpenRead::soundcloudauthfailed = "Failed to authenticate with SoundCloud.";
AudioFileStreamTools`InternetStreamOpenRead::usernamepasswordformat = "Both \"UserName\" and \"Password\" must be specified together, or not at all.";
AudioFileStreamTools`InternetStreamOpenRead::proxyusernamepasswordformat = "Both \"ProxyUserName\" and \"ProxyPassword\" must be specified together, or not at all.";
AudioFileStreamTools`InternetStreamOpenRead::usernamepasswordimmutable = "The \"UserName\" and \"Password\" options have already been set for this URL.";
AudioFileStreamTools`InternetStreamOpenRead::proxyusernamepasswordimmutable = "The \"ProxyUserName\" and \"ProxyPassword\" options have already been set for this URL.";
AudioFileStreamTools`InternetStreamOpenRead::nodeletionoptions = "Deletion options (\"DeleteFileOnClose\" and \"DeleteFileOnExit\") should not be specified if \"FilePath\" is not specified.";
AudioFileStreamTools`InternetStreamOpenRead::flacnotsupported = "The FLAC format is not currently supported for decoding as an InternetStream. If a \"FilePath\" was specified, the file will be downloaded and saved. It is not possible to decode without closing after the download completes and reopening with FileStreamOpenRead[].";

AudioFileStreamTools`InternetStreamGetBufferedRange::invalidtype = "The index type `1` is not valid, must be either \"Frames\" or \"Bytes\"";

AudioFileStreamTools::missingdependency = "The dependency \"`1`\" was not found. AudioFileStreamTools may not function properly.";

AudioFileStreamTools`MetaTags::noset = "`1`";
AudioFileStreamTools`MetaTags::invalid = "`1`";
AudioFileStreamTools`MetaTags::unsupported = "`1`";

Begin["`Private`"]
(* Implementation of the package *)

$packageFile = $InputFileName;
$libraryExtension = Switch[$OperatingSystem, "Windows", ".dll", "MacOSX", ".dylib", "Unix", ".so"];
$CURLLinkDir = FileNameJoin[{$InstallationDirectory, "SystemFiles", "Links", "CURLLink"}];
$CURLLinkLibDir = FileNameJoin[{$CURLLinkDir, "LibraryResources", $SystemID}];
$MP3ToolsDir = FileNameJoin[{$InstallationDirectory, "SystemFiles", "Links", "MP3Tools"}];
$MP3ToolsLibDir = FileNameJoin[{$MP3ToolsDir, "LibraryResources", $SystemID}];
$curlLibFileNames = Switch[$OperatingSystem, "Windows", {"libcurl.dll"}, "MacOSX", {"libcurl.dylib"}, "Unix", {"libcurl.so"}];
(* The libraries must be loaded in this order: Windows: "libeay32.dll", "ssleay32.dll", "libssh2.dll", "libcurl.dll" OSX: "libcrypto.dylib", "libssl.dylib", "libssh2.dylib", "libcurl.dylib"*)
(* CURLLink`CURLInitialize[]; should now take care of loading the libcurl dependencies. *)
$curlLibs = FileNameJoin[{$CURLLinkLibDir, #}]& /@ $curlLibFileNames;
$lameLibFileNames = Switch[$OperatingSystem, "Windows", {"libmp3lame.dll"}, "MacOSX", {"libmp3lame.dylib"}, "Unix", {"libmp3lame.so"}]; (* msvcr100.dll needs to be loaded before libmp3lame.dll *)
$lameLibs = FileNameJoin[{$MP3ToolsLibDir, #}]& /@ $lameLibFileNames;
$adapterLibFileName = Switch[$OperatingSystem, "Windows", "AudioFileStreamTools.dll", "MacOSX", "AudioFileStreamTools.dylib", "Unix", "AudioFileStreamTools.so"];
$otherLibFileNames = Switch[$OperatingSystem, "Windows", {}, "MacOSX", {}, "Unix", {}];
$adapterLibDir = FileNameJoin[{FileNameTake[$packageFile, {1,-2}], "LibraryResources", $SystemID}];
$adapterLib = FileNameJoin[{$adapterLibDir, $adapterLibFileName}];
$otherLibs = (FileNameJoin[{$adapterLibDir, #}])& /@ $otherLibFileNames;
$openStreams = Association[];
$$adapterInitialized = False;
$signedInt32BitMax = 2147483647;
$metaInformationFields = <|"BitDepth" -> 1, "ChannelCount" -> 2, "FrameCount" -> 3, "SampleRate" -> 4, "TotalFrameCount" -> 5, "Encoding" -> 6|>;
$streamInformationFields = {"FilePath", "DataContainer", "InternetStream", "InternetStreamType", "URL"};

Get[FileNameJoin[{FileNameTake[$packageFile, {1,-2}], "InternetStreamUtilities.m"}]];
LoadInternetStreamResources[$CURLLinkDir];
Get[FileNameJoin[{FileNameTake[$packageFile, {1,-2}], "MetaTags", "MetaTagsUtilities.m"}]];
LoadMetaTagsResources[FileNameJoin[{FileNameTake[$packageFile, {1,-2}], "MetaTags"}]];

assertDependency[depencencyPath_] := If[!FileExistsQ[depencencyPath], Message[AudioFileStreamTools::missingdependency, depencencyPath];];

loadAdapter[]:= If[ !$$adapterInitialized,
    Scan[(assertDependency[#])&, $otherLibs];
    (* $SystemID guard here and in LibraryLoad[...] below added to handle RPi platform's use of System library dependencies *)
    If[$SystemID =!= "Linux-ARM", Scan[(assertDependency[#])&, $lameLibs];];
    If[$SystemID =!= "Linux-ARM", Scan[(assertDependency[#])&, $curlLibs];];
    assertDependency[$CACERT];

    Scan[(LibraryLoad[#])&, $otherLibs];
    If[$SystemID =!= "Linux-ARM", Scan[(LibraryLoad[#])&, $lameLibs];];
    Needs["CURLLink`"];
    CURLLink`CURLInitialize[];

    lfFileStreamOpenRead = LibraryFunctionLoad[ $adapterLib, "FileStreamOpenRead", LinkObject, LinkObject];
    lfFileStreamOpenReadExtension = LibraryFunctionLoad[ $adapterLib, "FileStreamOpenReadExtension", LinkObject, LinkObject];
    lfFileStreamReadN = LibraryFunctionLoad[ $adapterLib, "FileStreamReadN", {Integer, Integer}, {Real, _}];
    lfFileStreamReadNRawArray = LibraryFunctionLoad[ $adapterLib, "FileStreamReadNRA", {Integer, Integer}, {"RawArray"}];
    lfFileStreamOpenWrite = LibraryFunctionLoad[ $adapterLib, "FileStreamOpenWrite", LinkObject, LinkObject];
    lfFileStreamWrite = LibraryFunctionLoad[$adapterLib,"FileStreamWrite",{Integer,{_Real,_,"Constant"}},{Integer}];
    lfFileStreamWriteRawArray = LibraryFunctionLoad[$adapterLib,"FileStreamWriteRA",{Integer,{"RawArray","Constant"}},{Integer}];
    lfFileStreamSetReadPosition = LibraryFunctionLoad[ $adapterLib, "FileStreamSetReadPosition", {Integer, Integer}, {Integer}];
    lfFileStreamGetReadPosition = LibraryFunctionLoad[ $adapterLib, "FileStreamGetReadPosition", {Integer}, {Integer}];
    lfFileStreamClose=LibraryFunctionLoad[$adapterLib,"FileStreamClose",{Integer},{Integer}];
    lfInternetStreamOpenReadMemory = LibraryFunctionLoad[ $adapterLib, "InternetStreamOpenReadMemory", LinkObject, LinkObject];
    lfInternetStreamOpenReadFile = LibraryFunctionLoad[ $adapterLib, "InternetStreamOpenReadFile", LinkObject, LinkObject];
    lfInternetStreamStartDownload=LibraryFunctionLoad[$adapterLib, "InternetStreamStartDownload", {Integer, Integer, Integer}, Integer];
    lfInternetStreamDownloadStatus=LibraryFunctionLoad[$adapterLib, "InternetStreamDownloadStatus", {Integer}, Integer];
    lfInternetStreamCurrentDownloadSize=LibraryFunctionLoad[$adapterLib, "InternetStreamCurrentDownloadSize", {Integer}, Real];
    lfInternetStreamFinalDownloadSize=LibraryFunctionLoad[$adapterLib, "InternetStreamFinalDownloadSize", {Integer}, Real];
    lfInternetStreamWaitForTransferInitialization=LibraryFunctionLoad[$adapterLib, "InternetStreamWaitForTransferInitialization", {Integer}, Integer];
    (*
	lfFileStreamGetWritePosition
	lfFileStreamSetWritePosition
	*)
    lfFileStreamGetMetaInformation = LibraryFunctionLoad[ $adapterLib, "FileStreamGetMetaInformation", LinkObject, LinkObject];

    lfGetEnvironmentProxySettings = LibraryFunctionLoad[ $adapterLib, "GetEnvironmentProxySettings", LinkObject, LinkObject];
    lfURLIsOpenedAsInternetStream = LibraryFunctionLoad[ $adapterLib, "URLIsOpenedAsInternetStream", LinkObject, LinkObject];
    lfFileIsOpenedAsInternetStream = LibraryFunctionLoad[ $adapterLib, "FileIsOpenedAsInternetStream", LinkObject, LinkObject];
    lfURLOpenedAsInternetStreamType = LibraryFunctionLoad[ $adapterLib, "URLOpenedAsInternetStreamType", LinkObject, LinkObject];
    lfFileOpenedAsInternetStreamType = LibraryFunctionLoad[ $adapterLib, "FileOpenedAsInternetStreamType", LinkObject, LinkObject];
    lfInternetStreamFilePathGetURL = LibraryFunctionLoad[ $adapterLib, "InternetStreamFilePathGetURL", LinkObject, LinkObject];
    lfInternetStreamURLGetFilePath = LibraryFunctionLoad[ $adapterLib, "InternetStreamURLGetFilePath", LinkObject, LinkObject];
    lfStreamHasOperatingSystemWritePermissions = LibraryFunctionLoad[ $adapterLib, "StreamHasOperatingSystemWritePermissions", {Integer}, Integer];

    lfFileStreamOpenTags = LibraryFunctionLoad[ $adapterLib, "FileStreamOpenTags", {Integer}, "Void"];
    lfFileStreamCloseTags = LibraryFunctionLoad[ $adapterLib, "FileStreamCloseTags", {Integer, Integer}, "Void"];
    lfFileStreamRemoveTag = LibraryFunctionLoad[$adapterLib, "FileStreamRemoveTag", {Integer, Integer}, "Void"];

    lfFileStreamRemoveID3v2Frame = LibraryFunctionLoad[$adapterLib, "FileStreamRemoveID3v2Frame", {Integer, {_Integer,_,"Constant"}}, "Void"];
    lfFileStreamAddID3v2Frame = LibraryFunctionLoad[$adapterLib, "FileStreamAddID3v2Frame", {Integer, {_Integer,_,"Constant"}, "RawArray"}, "Void"];

    lfFileStreamClearID3v2TableOFContentsFrameChildElements = LibraryFunctionLoad[$adapterLib, "FileStreamClearID3v2TableOFContentsFrameChildElements",{Integer, {_Integer,_,"Constant"}, Integer}, "Void"];
    lfFileStreamClearID3v2FrameSynchedText = LibraryFunctionLoad[$adapterLib, "FileStreamClearID3v2FrameSynchedText",{Integer, {_Integer,_,"Constant"}, Integer}, "Void"];
    lfFileStreamClearID3v2FrameChannels = LibraryFunctionLoad[$adapterLib, "FileStreamClearID3v2FrameChannels",{Integer, {_Integer,_,"Constant"}, Integer}, "Void"];

    lfFileStreamGetID3v2TableOFContentsFrameChildElementCount = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2TableOFContentsFrameChildElementCount", {Integer, {_Integer,_,"Constant"}, Integer}, Integer];
    lfFileStreamGetID3v2TableOFContentsFrameChildElementIdentifier = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2TableOFContentsFrameChildElementIdentifier", {Integer, {_Integer,_,"Constant"}, Integer, Integer}, "RawArray"];

    lfFileStreamGetID3v2FramesList = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FramesList", {Integer}, "RawArray"];
    lfFileStreamGetID3v2FrameID = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameID", {Integer, {_Integer,_,"Constant"}, Integer}, "RawArray"];
    lfFileStreamGetID3v2FrameValues = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameValues", {Integer, {_Integer,_,"Constant"}, Integer}, "RawArray"];
    lfFileStreamGetID3v2FrameDescription = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameDescription", {Integer, {_Integer,_,"Constant"}, Integer}, "RawArray"];

    lfFileStreamGetID3v2ChapterFrameValues = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2ChapterFrameValues", {Integer, {_Integer,_,"Constant"}, Integer}, "RawArray"];
    lfFileStreamGetID3v2ChapterFrameEmbeddedFramesList = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2ChapterFrameEmbeddedFramesList", {Integer, {_Integer,_,"Constant"}, Integer}, "RawArray"];
    lfFileStreamGetID3v2FrameLanguage = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameLanguage", {Integer, {_Integer,_,"Constant"}, Integer}, "RawArray"];
    lfFileStreamGetID3v2FrameText = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameText", {Integer, {_Integer,_,"Constant"}, Integer}, "RawArray"];

    lfFileStreamGetID3v2FrameTimeStampFormat = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameTimeStampFormat", {Integer, {_Integer,_,"Constant"}, Integer}, Integer];
    lfFileStreamGetID3v2FrameSynchedEvents = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameSynchedEvents", {Integer, {_Integer,_,"Constant"}, Integer}, "RawArray"];

    lfFileStreamGetID3v2FrameMimeType = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameMimeType", {Integer, {_Integer,_,"Constant"}, Integer}, "RawArray"];
    lfFileStreamGetID3v2FrameFileName = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameFileName", {Integer, {_Integer,_,"Constant"}, Integer}, "RawArray"];
    lfFileStreamGetID3v2FrameObject = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameObject", {Integer, {_Integer,_,"Constant"}, Integer}, "RawArray"];
    lfFileStreamGetID3v2FramePicture = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FramePicture", {Integer, {_Integer,_,"Constant"}, Integer}, "RawArray"];
    lfFileStreamGetID3v2FramePictureType = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FramePictureType", {Integer, {_Integer,_,"Constant"}, Integer}, Integer];

    lfFileStreamGetID3v2FramePricePaid = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FramePricePaid", {Integer, {_Integer,_,"Constant"}, Integer}, "RawArray"];
    lfFileStreamGetID3v2FramePurchaseDate = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FramePurchaseDate", {Integer, {_Integer,_,"Constant"}, Integer}, "RawArray"];
    lfFileStreamGetID3v2FrameSeller = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameSeller", {Integer, {_Integer,_,"Constant"}, Integer}, "RawArray"];
    lfFileStreamGetID3v2FrameEmail = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameEmail", {Integer, {_Integer,_,"Constant"}, Integer}, "RawArray"];
    lfFileStreamGetID3v2FrameCounter = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameCounter", {Integer, {_Integer,_,"Constant"}, Integer}, Integer];
    lfFileStreamGetID3v2FrameRating = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameRating", {Integer, {_Integer,_,"Constant"}, Integer}, Integer];

    lfFileStreamGetID3v2FrameOwner = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameOwner", {Integer, {_Integer,_,"Constant"}, Integer}, "RawArray"];
    lfFileStreamGetID3v2FrameChannels = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameChannels", {Integer, {_Integer,_,"Constant"}, Integer}, "RawArray"];
    lfFileStreamGetID3v2FrameData = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameData", {Integer, {_Integer,_,"Constant"}, Integer}, "RawArray"];

    lfFileStreamGetID3v2FramePeakVolume = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FramePeakVolume", {Integer, {_Integer,_,"Constant"}, Integer, Integer}, "RawArray"];
    lfFileStreamGetID3v2FramePeakBits = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FramePeakBits", {Integer, {_Integer,_,"Constant"}, Integer, Integer}, Integer];
    (*lfFileStreamGetID3v2FrameVolumeAdjustment = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameVolumeAdjustment", {Integer, {_Integer,_,"Constant"}, Integer, Integer}, Real];*)
    lfFileStreamGetID3v2FrameVolumeAdjustmentIndex = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameVolumeAdjustmentIndex", {Integer, {_Integer,_,"Constant"}, Integer, Integer}, Integer];

    lfFileStreamGetID3v2FrameLyricsType = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameLyricsType", {Integer, {_Integer,_,"Constant"}, Integer}, Integer];
    lfFileStreamGetID3v2FrameTopLevel = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameTopLevel", {Integer, {_Integer,_,"Constant"}, Integer}, Integer];
    lfFileStreamGetID3v2FrameOrdered = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameOrdered", {Integer, {_Integer,_,"Constant"}, Integer}, Integer];

    lfFileStreamGetID3v2FrameSynchedTextTimes = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameSynchedTextTimes", {Integer, {_Integer,_,"Constant"}, Integer}, "RawArray"];
    lfFileStreamGetID3v2FrameSynchedTextList = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameSynchedTextList", {Integer, {_Integer,_,"Constant"}, Integer}, "RawArray"];
    lfFileStreamGetID3v2FrameIdentifier = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameIdentifier", {Integer, {_Integer,_,"Constant"}, Integer}, "RawArray"];
    lfFileStreamGetID3v2FrameURL = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v2FrameURL", {Integer, {_Integer,_,"Constant"}, Integer}, "RawArray"];

    (* ID3v2 Frame Setters *)

    lfFileStreamSetID3v2TableOFContentsFrameChildElements = LibraryFunctionLoad[$adapterLib,"FileStreamSetID3v2TableOFContentsFrameChildElements",{Integer, {_Integer,_,"Constant"}, Integer, "RawArray"},"Void"];
    lfFileStreamSetID3v2FrameSynchedText = LibraryFunctionLoad[$adapterLib,"FileStreamSetID3v2FrameSynchedText",{Integer, {_Integer,_,"Constant"}, Integer, "RawArray", "RawArray"},"Void"];
    lfFileStreamSetID3v2FrameSynchedEvents = LibraryFunctionLoad[$adapterLib,"FileStreamSetID3v2FrameSynchedEvents",{Integer, {_Integer,_,"Constant"}, Integer, "RawArray"},"Void"];

    lfFileStreamSetID3v2FrameChannel = LibraryFunctionLoad[$adapterLib,"FileStreamSetID3v2FrameChannel",{Integer, {_Integer,_,"Constant"}, Integer, Integer, Integer, Integer, "RawArray"},"Void"];

    lfFileStreamSetID3v2FramePictureType = LibraryFunctionLoad[ $adapterLib, "FileStreamSetID3v2FramePictureType", {Integer, {_Integer,_,"Constant"}, Integer, Integer}, "Void"];
    lfFileStreamSetID3v2FrameLyricsType = LibraryFunctionLoad[ $adapterLib, "FileStreamSetID3v2FrameLyricsType", {Integer, {_Integer,_,"Constant"}, Integer, Integer}, "Void"];
    lfFileStreamSetID3v2FrameTimeStampFormat = LibraryFunctionLoad[ $adapterLib, "FileStreamSetID3v2FrameTimeStampFormat", {Integer, {_Integer,_,"Constant"}, Integer, Integer}, "Void"];

    lfFileStreamSetID3v2FrameDescription = LibraryFunctionLoad[$adapterLib,"FileStreamSetID3v2FrameDescription",{Integer, {_Integer,_,"Constant"}, Integer, "RawArray"},"Void"];
    lfFileStreamSetID3v2FrameValues = LibraryFunctionLoad[$adapterLib,"FileStreamSetID3v2FrameValues",{Integer, {_Integer,_,"Constant"}, Integer, "RawArray"},"Void"];
    lfFileStreamSetID3v2FrameLanguage = LibraryFunctionLoad[$adapterLib,"FileStreamSetID3v2FrameLanguage",{Integer, {_Integer,_,"Constant"}, Integer, "RawArray"},"Void"];
    lfFileStreamSetID3v2FrameFileName = LibraryFunctionLoad[$adapterLib,"FileStreamSetID3v2FrameFileName",{Integer, {_Integer,_,"Constant"}, Integer, "RawArray"},"Void"];

    lfFileStreamSetID3v2FrameMimeType = LibraryFunctionLoad[$adapterLib,"FileStreamSetID3v2FrameMimeType",{Integer, {_Integer,_,"Constant"}, Integer, "RawArray"},"Void"];
    lfFileStreamSetID3v2FramePicture = LibraryFunctionLoad[$adapterLib,"FileStreamSetID3v2FramePicture",{Integer, {_Integer,_,"Constant"}, Integer, "RawArray"},"Void"];
    lfFileStreamSetID3v2FrameSeller = LibraryFunctionLoad[$adapterLib,"FileStreamSetID3v2FrameSeller",{Integer, {_Integer,_,"Constant"}, Integer, "RawArray"},"Void"];
    lfFileStreamSetID3v2FramePurchaseDate = LibraryFunctionLoad[$adapterLib,"FileStreamSetID3v2FramePurchaseDate",{Integer, {_Integer,_,"Constant"}, Integer, "RawArray"},"Void"];
    lfFileStreamSetID3v2FramePricePaid = LibraryFunctionLoad[$adapterLib,"FileStreamSetID3v2FramePricePaid",{Integer, {_Integer,_,"Constant"}, Integer, "RawArray"},"Void"];
    lfFileStreamSetID3v2FrameEmail = LibraryFunctionLoad[$adapterLib,"FileStreamSetID3v2FrameEmail",{Integer, {_Integer,_,"Constant"}, Integer, "RawArray"},"Void"];
    lfFileStreamSetID3v2FrameObject = LibraryFunctionLoad[$adapterLib,"FileStreamSetID3v2FrameObject",{Integer, {_Integer,_,"Constant"}, Integer, "RawArray"},"Void"];

    lfFileStreamSetID3v2FrameOwner = LibraryFunctionLoad[$adapterLib,"FileStreamSetID3v2FrameOwner",{Integer, {_Integer,_,"Constant"}, Integer, "RawArray"},"Void"];
    lfFileStreamSetID3v2FrameData = LibraryFunctionLoad[$adapterLib,"FileStreamSetID3v2FrameData",{Integer, {_Integer,_,"Constant"}, Integer, "RawArray"},"Void"];
    lfFileStreamSetID3v2FrameIdentifier = LibraryFunctionLoad[$adapterLib,"FileStreamSetID3v2FrameIdentifier",{Integer, {_Integer,_,"Constant"}, Integer, "RawArray"},"Void"];
    lfFileStreamSetID3v2FrameURL = LibraryFunctionLoad[$adapterLib,"FileStreamSetID3v2FrameURL",{Integer, {_Integer,_,"Constant"}, Integer, "RawArray"},"Void"];
    lfFileStreamSetID3v2FrameText = LibraryFunctionLoad[$adapterLib,"FileStreamSetID3v2FrameText",{Integer, {_Integer,_,"Constant"}, Integer, "RawArray"},"Void"];

    lfFileStreamSetID3v2FrameEndOffset = LibraryFunctionLoad[ $adapterLib, "FileStreamSetID3v2FrameEndOffset", {Integer, {_Integer,_,"Constant"}, Integer, Integer}, "Void"];
    lfFileStreamSetID3v2FrameStartOffset = LibraryFunctionLoad[ $adapterLib, "FileStreamSetID3v2FrameStartOffset", {Integer, {_Integer,_,"Constant"}, Integer, Integer}, "Void"];
    lfFileStreamSetID3v2FrameStartTime = LibraryFunctionLoad[ $adapterLib, "FileStreamSetID3v2FrameStartTime", {Integer, {_Integer,_,"Constant"}, Integer, Integer}, "Void"];
    lfFileStreamSetID3v2FrameEndTime = LibraryFunctionLoad[ $adapterLib, "FileStreamSetID3v2FrameEndTime", {Integer, {_Integer,_,"Constant"}, Integer, Integer}, "Void"];
    lfFileStreamSetID3v2FrameOrdered = LibraryFunctionLoad[ $adapterLib, "FileStreamSetID3v2FrameOrdered", {Integer, {_Integer,_,"Constant"}, Integer, Integer}, "Void"];
    lfFileStreamSetID3v2FrameTopLevel = LibraryFunctionLoad[ $adapterLib, "FileStreamSetID3v2FrameTopLevel", {Integer, {_Integer,_,"Constant"}, Integer, Integer}, "Void"];

    (* ID3v1 *)

    lfFileStreamGetID3v1Element = LibraryFunctionLoad[ $adapterLib, "FileStreamGetID3v1Element", {Integer, Integer}, "RawArray"];
    lfFileStreamSetID3v1Element = LibraryFunctionLoad[ $adapterLib, "FileStreamSetID3v1Element", {Integer, Integer, "RawArray"}, "Void"];

    (* APE *)

    lfFileStreamGetAPEItemKeys = LibraryFunctionLoad[ $adapterLib, "FileStreamGetAPEItemKeys", {Integer}, "RawArray"];
    lfFileStreamGetAPEItemTypes = LibraryFunctionLoad[ $adapterLib, "FileStreamGetAPEItemTypes", {Integer}, "RawArray"];
    lfFileStreamAddAPEItem = LibraryFunctionLoad[ $adapterLib, "FileStreamAddAPEItem", {Integer, "RawArray", Integer}, "Void"];
    lfFileStreamRemoveAPEItem = LibraryFunctionLoad[ $adapterLib, "FileStreamRemoveAPEItem", {Integer, "RawArray"}, "Void"];
    lfFileStreamGetAPEItemValues = LibraryFunctionLoad[ $adapterLib, "FileStreamGetAPEItemValues", {Integer, "RawArray"}, "RawArray"];
    lfFileStreamGetAPEItemData = LibraryFunctionLoad[ $adapterLib, "FileStreamGetAPEItemData", {Integer, "RawArray"}, "RawArray"];
    lfFileStreamSetAPEItemValues = LibraryFunctionLoad[ $adapterLib, "FileStreamSetAPEItemValues", {Integer, "RawArray", "RawArray"}, "Void"];
    lfFileStreamSetAPEItemData = LibraryFunctionLoad[ $adapterLib, "FileStreamSetAPEItemData", {Integer, "RawArray", "RawArray"}, "Void"];

    (* Xiph *)

    lfFileStreamGetXiphKeys = LibraryFunctionLoad[ $adapterLib, "FileStreamGetXiphKeys", {Integer}, "RawArray"];
    lfFileStreamGetXiphValues = LibraryFunctionLoad[ $adapterLib, "FileStreamGetXiphValues", {Integer, "RawArray"}, "RawArray"];
    lfFileStreamSetXiphValues = LibraryFunctionLoad[ $adapterLib, "FileStreamSetXiphValues", {Integer, "RawArray", "RawArray"}, "Void"];
    lfFileStreamAddXiphKey = LibraryFunctionLoad[ $adapterLib, "FileStreamAddXiphKey", {Integer, "RawArray"}, "Void"];
    lfFileStreamRemoveXiphKey = LibraryFunctionLoad[ $adapterLib, "FileStreamRemoveXiphKey", {Integer, "RawArray"}, "Void"];
    lfFileStreamRemoveXiphKeyWithValue = LibraryFunctionLoad[ $adapterLib, "FileStreamRemoveXiphKeyWithValue", {Integer, "RawArray", "RawArray"}, "Void"];

    (* M4A *)

    lfFileStreamGetM4AItemBytes = LibraryFunctionLoad[ $adapterLib, "FileStreamGetM4AItemBytes", {Integer, "RawArray"}, "RawArray"];
    lfFileStreamGetM4AItemStrings = LibraryFunctionLoad[ $adapterLib, "FileStreamGetM4AItemStrings", {Integer, "RawArray"}, "RawArray"];
    lfFileStreamGetM4AItemBoolean = LibraryFunctionLoad[ $adapterLib, "FileStreamGetM4AItemBoolean", {Integer, "RawArray"}, "Boolean"];
    lfFileStreamGetM4AItemInt = LibraryFunctionLoad[ $adapterLib, "FileStreamGetM4AItemInt", {Integer, "RawArray", Integer}, Integer];
    lfFileStreamGetM4AItemIntPair = LibraryFunctionLoad[ $adapterLib, "FileStreamGetM4AItemIntPair", {Integer, "RawArray"}, "RawArray"];
    lfFileStreamGetM4AItemCoverArtN = LibraryFunctionLoad[ $adapterLib, "FileStreamGetM4AItemCoverArtN", {Integer, "RawArray"}, Integer];
    lfFileStreamGetM4AItemCoverArt = LibraryFunctionLoad[ $adapterLib, "FileStreamGetM4AItemCoverArt", {Integer, "RawArray", Integer}, "RawArray"];
    lfFileStreamGetM4AItemCoverArtFormat = LibraryFunctionLoad[ $adapterLib, "FileStreamGetM4AItemCoverArtFormat", {Integer, "RawArray", Integer}, "RawArray"];
    lfFileStreamGetM4AItemType = LibraryFunctionLoad[ $adapterLib, "FileStreamGetM4AItemType", {Integer, "RawArray"}, Integer];
    lfFileStreamSetM4AItemType = LibraryFunctionLoad[ $adapterLib, "FileStreamSetM4AItemType", {Integer, "RawArray", Integer}, "Void"];
    lfFileStreamGetM4AItemKeys = LibraryFunctionLoad[ $adapterLib, "FileStreamGetM4AItemKeys", {Integer}, "RawArray"];
    lfFileStreamSetM4AItem = LibraryFunctionLoad[ $adapterLib, "FileStreamSetM4AItem", {Integer, "RawArray", "RawArray", Integer, Integer}, "Void"];
    lfFileStreamSetM4AItemInList = LibraryFunctionLoad[ $adapterLib, "FileStreamSetM4AItemInList", {Integer, "RawArray", "RawArray", Integer, Integer, Integer}, "Void"];
    lfFileStreamRemoveM4AItemKey = LibraryFunctionLoad[ $adapterLib, "FileStreamRemoveM4AItemKey", {Integer, "RawArray"}, "Void"];

    $$adapterInitialized = True;
]

loadAdapter[];

$trueOpenedFileStreamsRefCount = <||>;
$deleteFileOnCloseAssoc = <||>;
$streamTypeAssoc = <||>;
$originalStreamTypes = <|1 -> "File", 2 -> "Memory"|>;
Internal`SetValueNoTrack[$trueOpenedFileStreamsRefCount, True];
Internal`SetValueNoTrack[$deleteFileOnCloseAssoc, True];
Internal`SetValueNoTrack[$streamTypeAssoc, True];

Options[AudioFileStreamTools`FileStreamOpenRead] = {"ContainerType" -> "RawArray", "DeleteFileOnClose" -> Automatic, "DeleteFileOnExit" -> Automatic};

AudioFileStreamTools`FileStreamOpenRead[fileName_String, OptionsPattern[]]:=
    Module[{streamID, containerType, filePath, format = "", deleteOnClose = 0, deleteOnExit = 0, originalStreamType = 0},
        containerType = OptionValue["ContainerType"];
        If[!(containerType === "RawArray") && !(containerType === "MTensor"), Message[FileStreamOpenRead::invalidcontainer, containerType];Return[$Failed];];
        loadAdapter[];
        filePath = ImportExport`FileUtilities`GetFilePath[fileName];
        If[filePath == $Failed, Message[FileStreamOpenRead::nofile, fileName]; Return[$Failed];];
        If[!FileExistsQ[filePath], Message[FileStreamOpenRead::nofile, filePath]; Return[$Failed];];
        If[$streamTypeAssoc[filePath] === "Write", Message[FileStreamOpenRead::streamtypeconflict, "Read", $streamTypeAssoc[filePath], filePath]; Return[$Failed];];

        If[!MissingQ[$trueOpenedFileStreamsRefCount[filePath]], deleteOnExit = -1];

        If[OptionValue["DeleteFileOnClose"] =!= Automatic || OptionValue["DeleteFileOnExit"] =!= Automatic,
            If[lfFileIsOpenedAsInternetStream[filePath] === 1 || !MissingQ[$trueOpenedFileStreamsRefCount[filePath]],
                Message[FileStreamOpenRead::setdelopts];
                ,
                If[(OptionValue["DeleteFileOnClose"] === False || OptionValue["DeleteFileOnClose"] === Automatic) && OptionValue["DeleteFileOnExit"] === True, deleteOnExit = 1;];
                If[OptionValue["DeleteFileOnClose"] == True, deleteOnClose = 1;];
            ];
        ];

        (* Using FileFormat here to assert that we have read permissions for the file, JUCE will hang stuck in a loop if we do not. *)
        If[Quiet[FileFormat[filePath]] == $Failed, Message[FileStreamOpenRead::noreadperm, filePath]; Return[$Failed];];
        originalStreamType = lfFileOpenedAsInternetStreamType[filePath];

	(* handle case where file has no extension, since AFST relies on extension to determine audio format *)
	(*If[Length[StringPosition[filePath, "."]] > 0,
		Print["file had extension: ", StringDrop[filePath,First[Last[StringPosition[filePath,"."]]]-1]];
		streamID = lfFileStreamOpenRead[filePath, deleteOnClose, deleteOnExit]; (* will this handle .WAV extension properly? *)
		,
		Print["file had NO extension; format is: ", "."<>ToLowerCase[FileFormat[filePath]]];
		format = "."<>ToLowerCase[FileFormat[filePath]];
		streamID = lfFileStreamOpenReadExtension[filePath, format, deleteOnClose, deleteOnExit];
	];*)
		(* use FileFormat[] for all files, to handle .tmp files from URLSave *)
		format = "."<>ToLowerCase[FileFormat[filePath]];
		streamID = lfFileStreamOpenReadExtension[filePath, format, deleteOnClose, deleteOnExit];
        If[Head[streamID] === LibraryFunctionError, Message[FileStreamOpenRead::openreadfail, filePath]; Return[$Failed]];
        setField[streamID, "FilePath", filePath];
        setField[streamID, "DataContainer", containerType];

        If[lfFileIsOpenedAsInternetStream[filePath] === 0,
            If[MissingQ[$trueOpenedFileStreamsRefCount[filePath]],
                $trueOpenedFileStreamsRefCount[filePath] = 1;
                $streamTypeAssoc[filePath] = "Read";
                If[OptionValue["DeleteFileOnClose"] === Automatic,
                    $deleteFileOnCloseAssoc[filePath] = False;
                    ,
                    $deleteFileOnCloseAssoc[filePath] = OptionValue["DeleteFileOnClose"];
                ];
                ,
                $trueOpenedFileStreamsRefCount[filePath] = $trueOpenedFileStreamsRefCount[filePath] + 1;
            ];
        ];

        If[!IntegerQ[originalStreamType] || originalStreamType === 0,
            setField[streamID, "InternetStream", False];
            setField[streamID, "InternetStreamType", None];
            setField[streamID, "URL", None];
            ,
            setField[streamID, "InternetStream", True];
            setField[streamID, "InternetStreamType", $originalStreamTypes[originalStreamType]];
            setField[streamID, "URL", lfInternetStreamFilePathGetURL[filePath]];
        ];

        Return[ AudioFileStreamTools`FileStreamObject[ streamID]];
    ]

AudioFileStreamTools`FileStreamReadN[ obj_AudioFileStreamTools`FileStreamObject, numFrames_Integer]:=
    Module[{ streamID, res, readPos, eof},
        loadAdapter[];
        streamID = getStreamID@obj;
        If[!KeyExistsQ[$openStreams, streamID], Message[FileStreamReadN::afstnostream, obj]; Return[$Failed]];
        If[numFrames <= 0, Message[FileStreamReadN::numframesoutofbounds]; Return[$Failed];];
        If[numFrames > $signedInt32BitMax, Message[FileStreamReadN::numframesoutofbounds]; Return[$Failed];];
        If[getField[streamID, "DataContainer"] == "RawArray", res = lfFileStreamReadNRawArray[streamID, numFrames];, res = lfFileStreamReadN[streamID, numFrames];];
        If[Head[res] === LibraryFunctionError, Message[FileStreamReadN::positionpastendoffile]; Return[EndOfFile]];
        readPos = FileStreamGetReadPosition[obj]; (* M-indexed *)
        (* from FileStreamSetReadPosition documentation, eof is defined as "FrameCount" + 1 *)
        eof = FileStreamGetMetaInformation[obj]["FrameCount"] + 1;
        If[IntegerQ[eof] && IntegerQ[readPos],
            If[readPos > eof, Message[FileStreamReadN::reachedendoffile]; FileStreamSetReadPosition[obj, eof];];
        ];
        Return[res];
    ]

AudioFileStreamTools`FileStreamReadN[___]:= (Message[FileStreamReadN::invalidargs]; Return[$Failed]);

AudioFileStreamTools`FileStreamClose[ obj_AudioFileStreamTools`FileStreamObject]:=
    Module[{ streamID, filePath, refCount},
        loadAdapter[];
        streamID = getStreamID@obj;
        If[!KeyExistsQ[$openStreams, streamID], Message[FileStreamClose::afstnostream, obj]; Return[$Failed]];
        lfFileStreamClose[streamID]; (* TODO ERR *)
        filePath = getField[streamID, "FilePath"];
        If[!StringQ[filePath], Return[];];

        refCount = $trueOpenedFileStreamsRefCount[filePath];
        If[!MissingQ[refCount],
            refCount--;
            $trueOpenedFileStreamsRefCount[filePath] = refCount;
            If[refCount === 0,
                If[$deleteFileOnCloseAssoc[filePath] === True,
                    DeleteFile[filePath];
                ];
                KeyDropFrom[$trueOpenedFileStreamsRefCount, filePath];
                KeyDropFrom[$deleteFileOnCloseAssoc, filePath];
                KeyDropFrom[$streamTypeAssoc, filePath];
            ];
            ,
            If[lfFileIsOpenedAsInternetStream[filePath] === 0,
                KeyDropFrom[$streamTypeAssoc, filePath]; (* Remove on last virtual stream closed. *)
            ]
        ];

        removeTagReferences[streamID];
        KeyDropFrom[$openStreams, streamID];
    ]

AudioFileStreamTools`FileStreamClose[___]:= (Message[FileStreamClose::invalidargs]; Return[$Failed]);

AudioFileStreamTools`FileStreamGetReadPosition[ obj_AudioFileStreamTools`FileStreamObject]:=
    Module[{ streamID, position},
        loadAdapter[];
        streamID = getStreamID@obj;
        If[!KeyExistsQ[$openStreams, streamID], Message[FileStreamGetReadPosition::afstnostream, obj]; Return[$Failed]];

        position = lfFileStreamGetReadPosition[streamID];
        If[Head[position] == LibraryFunctionError, Message[FileStreamGetReadPosition::afstnostream, obj]; Return[$Failed]];
        Return[position];
    ]

AudioFileStreamTools`FileStreamGetReadPosition[___]:= (Message[FileStreamGetReadPosition::invalidargs]; Return[$Failed]);

AudioFileStreamTools`FileStreamSetReadPosition[ obj_AudioFileStreamTools`FileStreamObject, pos_Integer]:=
    Module[{ streamID, position},
        loadAdapter[];
        streamID = getStreamID@obj;
        If[!KeyExistsQ[$openStreams, streamID], Message[FileStreamSetReadPosition::afstnostream, obj]; Return[$Failed]];
        If[pos <= 0,
            Message[FileStreamSetReadPosition::invalidposition, pos]; Return[$Failed];
        ];
        If[pos > lfFileStreamGetMetaInformation[streamID, $metaInformationFields["FrameCount"]]+1,
            Message[FileStreamSetReadPosition::stmrng, obj, pos];
        ];
        position = lfFileStreamSetReadPosition[streamID, pos];

        If[position == LibraryFunctionError["LIBRARY_DIMENSION_ERROR", 3], Message[FileStreamSetReadPosition::invalidposition, pos]; Return[$Failed]];
		If[Head[position] == LibraryFunctionError, Message[FileStreamSetReadPosition::afstnostream, obj]; Return[$Failed]];
		Return[position];
    ]

AudioFileStreamTools`FileStreamSetReadPosition[___]:= (Message[FileStreamSetReadPosition::invalidargs]; Return[$Failed]);

AudioFileStreamTools`FileStreamGetMetaInformation[ obj_AudioFileStreamTools`FileStreamObject]:=
    Module[{ streamID},
        loadAdapter[];
        streamID = getStreamID@obj;
        If[!KeyExistsQ[$openStreams, streamID], Message[FileStreamGetMetaInformation::afstnostream, obj]; Return[$Failed]];
        Return[ AssociationMap[(lfFileStreamGetMetaInformation[ streamID, $metaInformationFields[#]])&, Keys[$metaInformationFields]]];
    ]

AudioFileStreamTools`FileStreamGetMetaInformation[ obj_AudioFileStreamTools`FileStreamObject, field_String]:=
    Module[{ streamID},
        loadAdapter[];
        streamID = getStreamID@obj;
        If[!KeyExistsQ[$openStreams, streamID], Message[FileStreamGetMetaInformation::afstnostream, obj]; Return[$Failed]];
        If[!KeyExistsQ[$metaInformationFields, field], Message[FileStreamGetMetaInformation::noinfo, field]; Return[$Failed]];
        Return[ lfFileStreamGetMetaInformation[ streamID, $metaInformationFields[field]]];
    ]

AudioFileStreamTools`FileStreamGetMetaInformation[ obj_AudioFileStreamTools`FileStreamObject, fields_List]:=
    Module[{streamID, results},
        loadAdapter[];
        streamID = getStreamID@obj;
        If[!KeyExistsQ[$openStreams, streamID], Message[FileStreamGetMetaInformation::afstnostream, obj]; Return[$Failed]];
        results = Map[(If[!KeyExistsQ[$metaInformationFields, #], Message[FileStreamGetMetaInformation::noinfo, #]; 1, 0])&, fields];
        If[Total[results] =!= 0, Return[$Failed]];
        Return[ AssociationMap[(lfFileStreamGetMetaInformation[ streamID, $metaInformationFields[#]])&, fields]];
    ]

AudioFileStreamTools`FileStreamGetMetaInformation[___]:= (Message[FileStreamGetMetaInformation::invalidargs]; Return[$Failed]);

AudioFileStreamTools`FileStreamGetStreamInformation[ obj_AudioFileStreamTools`FileStreamObject]:=
    Module[{streamID},
        loadAdapter[];
        streamID = getStreamID@obj;
        If[!KeyExistsQ[$openStreams, streamID], Message[FileStreamGetStreamInformation::afstnostream, obj]; Return[$Failed]];
        Return[ AssociationMap[(getField[streamID, #])&, $streamInformationFields]];
    ]

AudioFileStreamTools`FileStreamGetStreamInformation[ obj_AudioFileStreamTools`FileStreamObject, field_String]:=
    Module[{streamID},
        loadAdapter[];
        streamID = getStreamID@obj;
        If[!KeyExistsQ[$openStreams, streamID], Message[FileStreamGetStreamInformation::afstnostream, obj]; Return[$Failed]];
        If[!MemberQ[$streamInformationFields, field], Message[FileStreamGetStreamInformation::noinfo, field]; Return[$Failed]];
        Return[getField[streamID, field]];
	]

AudioFileStreamTools`FileStreamGetStreamInformation[ obj_AudioFileStreamTools`FileStreamObject, fields_List]:=
    Module[{streamID, results},
        loadAdapter[];
        streamID = getStreamID@obj;
        If[!KeyExistsQ[$openStreams, streamID], Message[FileStreamGetStreamInformation::afstnostream, obj]; Return[$Failed]];
        results = Map[(If[!MemberQ[$streamInformationFields, #], Message[FileStreamGetStreamInformation::noinfo, #]; 1, 0])&, fields];
        If[Total[results] =!= 0, Return[$Failed]];
        Return[ AssociationMap[(getField[streamID, #])&, fields]];
    ]

AudioFileStreamTools`FileStreamGetStreamInformation[___]:= (Message[FileStreamGetStreamInformation::invalidargs]; Return[$Failed]);

Options[AudioFileStreamTools`FileStreamOpenWrite] = {"ContainerType" -> "RawArray", "DeleteFileOnClose" -> Automatic, "DeleteFileOnExit" -> Automatic};

AudioFileStreamTools`FileStreamOpenWrite[fileName_String, sampleFreq_, dim_, bitdepth_, OptionsPattern[]]:=
    Module[{ streamID, filePath, directory, deleteOnClose = 0, deleteOnExit = 0, hasWritePermissions},
        loadAdapter[];

        directory = FileNameTake[fileName, {1, -2}];
        If[directory === "", directory = Directory[];];
        filePath = ImportExport`FileUtilities`GetFilePath[directory];
        If[!DirectoryQ[directory] || filePath === $Failed,
            Message[FileStreamOpenWrite::dirnex, directory];
            Return[$Failed];
        ];
        filePath = filePath <> FileNameTake[fileName];

        If[!MissingQ[$streamTypeAssoc[filePath]], Message[FileStreamOpenWrite::currentlyopen, filePath]; Return[$Failed];];

        If[!MissingQ[$trueOpenedFileStreamsRefCount[filePath]], deleteOnExit = -1];

        If[OptionValue["DeleteFileOnClose"] =!= Automatic || OptionValue["DeleteFileOnExit"] =!= Automatic,
            If[lfFileIsOpenedAsInternetStream[filePath] === 1 || !MissingQ[$trueOpenedFileStreamsRefCount[filePath]],
                Message[FileStreamOpenWrite::setdelopts];
                ,
                If[(OptionValue["DeleteFileOnClose"] === False || OptionValue["DeleteFileOnClose"] === Automatic) && OptionValue["DeleteFileOnExit"] === True, deleteOnExit = 1;];
                If[OptionValue["DeleteFileOnClose"] == True, deleteOnClose = 1;];
            ];
        ];

        streamID = lfFileStreamOpenWrite[filePath, sampleFreq, dim, bitdepth, deleteOnClose, deleteOnExit]; (* TODO ERR *)
        If[Head[streamID] === LibraryFunctionError, Message[FileStreamOpenWrite::openwritefail]; Return[$Failed];];
        setField[streamID, "FilePath", filePath];
        hasWritePermissions = lfStreamHasOperatingSystemWritePermissions[streamID];
        If[Head[hasWritePermissions] === LibraryFunctionError, Message[FileStreamOpenWrite::openwritefail]; lfFileStreamClose[streamID]; Return[$Failed]];
        If[hasWritePermissions =!= 1, Message[FileStreamOpenWrite::nowriteperm, filePath, directory]; lfFileStreamClose[streamID]; Return[$Failed]];

        setField[streamID, "DataContainer", OptionValue["ContainerType"]];
        setField[streamID, "Channels", dim];

        If[lfFileIsOpenedAsInternetStream[filePath] === 0,
            If[MissingQ[$trueOpenedFileStreamsRefCount[filePath]],
                $trueOpenedFileStreamsRefCount[filePath] = 1;
                $streamTypeAssoc[filePath] = "Write";
                If[OptionValue["DeleteFileOnClose"] === Automatic,
                    $deleteFileOnCloseAssoc[filePath] = False;
                    ,
                    $deleteFileOnCloseAssoc[filePath] = OptionValue["DeleteFileOnClose"];
                ];
                ,
                $trueOpenedFileStreamsRefCount[filePath] = $trueOpenedFileStreamsRefCount[filePath] + 1;
            ];
        ];

        Return[AudioFileStreamTools`FileStreamObject[streamID]];
    ]

AudioFileStreamTools`FileStreamOpenWrite[___]:= (Message[FileStreamOpenWrite::invalidargs]; Return[$Failed]);

AudioFileStreamTools`FileStreamWrite[obj_AudioFileStreamTools`FileStreamObject, audioDataMatrix_]:=
    Module[{streamID, res, dataIsRawArray, streamDataContainer, streamIsRawArray, dataIsMTensor, streamIsMTensor, numChannels, dimensions, rank, typeMismatch},
        loadAdapter[];
        streamID = getStreamID@obj;
        If[!KeyExistsQ[$openStreams, streamID], Message[FileStreamWrite::afstnostream, obj]; Return[$Failed]];
        dataIsRawArray = Developer`RawArrayQ[audioDataMatrix];
        dataIsMTensor = Head[audioDataMatrix] == List;
        streamDataContainer = getField[streamID, "DataContainer"];
        streamIsRawArray = streamDataContainer == "RawArray";
        streamIsMTensor = streamDataContainer == "MTensor";
        If[dataIsRawArray != streamIsRawArray, Message[FileStreamWrite::containermismatch, streamDataContainer]; Return[$Failed]];
        If[dataIsMTensor != streamIsMTensor, Message[FileStreamWrite::containermismatch, streamDataContainer]; Return[$Failed]];
        numChannels = getField[streamID, "Channels"];
        dimensions = Dimensions[audioDataMatrix];
        rank = Length[dimensions];
        If[(rank != 2), Message[AudioFileStreamTools`FileStreamWrite::invaliddimensions]; Return[$Failed]];
        If[(numChannels != dimensions[[1]]), Message[AudioFileStreamTools`FileStreamWrite::dimensionmismatch, obj]; Return[$Failed]];

        typeMismatch = If[dataIsRawArray, Developer`RawArrayType[audioDataMatrix] != "Real32",
            If[rank == 1,
                Scan[If[!(Developer`RealQ[#] || IntegerQ[#]), Return[True]]&]
                ,
                !MatrixQ[audioDataMatrix,(Developer`RealQ@#||IntegerQ@#)&]
            ]
        ];
        If[typeMismatch, Message[AudioFileStreamTools`FileStreamWrite::invalidtype]; Return[$Failed]];
        If[dataIsRawArray, res = lfFileStreamWriteRawArray[streamID, audioDataMatrix];, res = Check[lfFileStreamWrite[streamID,audioDataMatrix], Message[AudioFileStreamTools`FileStreamWrite::dimensionmismatch, obj]; $Failed];];
        If[res === 0, Return[Null], Return[$Failed]];
    ]

AudioFileStreamTools`FileStreamWrite[___]:= (Message[FileStreamWrite::invalidargs]; Return[$Failed]);

(* AudioFileStreamTools`GetWritePosition[]:= foo
 AudioFileStreamTools`SetWritePosition[]:= foo *)

(* Stream Handle Functions *)

check1DListType[element_] := If[!(Developer`RealQ[#] || IntegerQ[#]), Return[True]]&

getStreamID[ obj_AudioFileStreamTools`FileStreamObject]:= Return[ obj[[1]]]

setField[key_,field_,value_]:=Module[{innerAssoc},
    innerAssoc=Lookup[$openStreams,key];
    If[Head@innerAssoc === Missing, innerAssoc=Association[];];
    AssociateTo[innerAssoc, field->value];
    AssociateTo[$openStreams,key->innerAssoc]
]

getField[key_,field_]:=Module[{innerAssoc},
    innerAssoc=Lookup[$openStreams,key];
    If[Head@innerAssoc===Missing,Return[Missing["NotAvailable"]]];
    Lookup[innerAssoc,field]
]

getField[key_]:=Module[{innerAssoc},
    innerAssoc=Lookup[$openStreams,key];
    If[Head@innerAssoc===Missing,Return[Missing["NotAvailable"]]];
    Return[innerAssoc]
]

End[]

EndPackage[]

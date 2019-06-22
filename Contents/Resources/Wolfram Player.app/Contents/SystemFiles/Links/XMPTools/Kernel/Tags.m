(**************************)
(**************************)
(**************************)
(*********ALL TAGS*********)
(**************************)
(**************************)
(**************************)

MNStringTags = {"ImageType", "FirmwareVersion", "OwnerName", "LensModel", "InternalSerialNumber", "DustRemovalData", "SerialNumber", "Quality", "FileSource", "CameraSettingsZ1", 
	"PrintIM", "CameraSettings5D", "WBInfoA100", "ImageStabilizationData", "CameraSettings7D", "CameraSettingsStdNew", "CameraSettingsStdOld", "ColorMode", "Sharpening",
	"WhiteBalance", "Focus", "FlashSetting", "ISOSelection", "ImageAdjustment", "AuxiliaryLens", "BodyFirmwareVersion", "CameraType", "PictureInfo", "CameraID", "Software", 
	"Firmware", "SerialNumber2", "BabyAge1", "LensType", "LensSerialNumber", "AccessoryType", "AccessorySerialNumber", "BabyAge2", "LocalLocationName", "LocationName", "FirmwareName", 
    "LensFirmware", "DriveMode", "ResolutionMode", "AutofocusMode", "FocusSetting","ExposureMode", "MeteringMode", "LensRange", "ColorSpace", "Exposure", "Contrast",
    "Shadow", "Highlight", "Saturation", "Sharpness", "FillLight", "ColorAdjustment", "AdjustmentMode", "AutoBracket", "0x2003", "ColorReproduction"}

IntegerTags = {(*Exif*) "SubjectArea", "NewSubfileType", "SubfileType", "ImageWidth", "ImageLength", "BitsPerSample", "PhotometricInterpretation",
"Threshholding", "CellWidth", "CellLength", "FillOrder", "StripOffsets", "Orientation", "SamplesPerPixel", "RowsPerStrip", "StripByteCounts", "PlanarConfiguration",
"GrayResponseUnit", "GrayResponseCurve", "T4Options", "T6Options", "TransferFunction", "Predictor", "ColorMap", "HalftoneHints", "TileWidth",
"TileLength", "TileOffsets", "TileByteCounts", "SubIFDs", "InkSet", "NumberOfInks", "DotRange", "ExtraSamples", "SampleFormat", "SMinSampleValue", "SMaxSampleValue",
"TransferRange", "ClipPath", "XClipPathUnits", "YClipPathUnits", "Indexed", "OPIProxy", "JPEGProc", "JPEGRestartInterval",
"JPEGLosslessPredictors", "JPEGPointTransforms", "JPEGQTables", "JPEGDCTables", "JPEGACTables", "YCbCrSubSampling", "YCbCrPositioning", "XMLPacket", "Rating", "RatingPercent",
"CFARepeatPatternDim", "IPTCNAA", "ImageResources", "ExifTag", "SpectralSensitivity", "GPSTag", "Interlace", "LocalizedCameraModel", "CFAPlaneColor", "CFALayout", "LinearizationTable",
"TimeZoneOffset", "SelfTimerMode", "ImageNumber", "TIFFEPStandardID", "XPTitle", "XPComment", "XPAuthor", "XPKeywords", "XPSubject", "DNGVersion", "DNGBackwardVersion",
"BlackLevelRepeatDim", "WhiteLevel", "DefaultCropOrigin", "DefaultCropSize", "AsShotNeutral", "BayerGreenSplit", "DNGPrivateData", "MakerNoteSafety", "CalibrationIlluminant1",
"CalibrationIlluminant2", "RawDataUniqueID", "OriginalRawFileName", "ActiveArea", "MaskedAreas", "ColorimetricReference", "CameraCalibrationSignature", "ProfileCalibrationSignature",
"AsShotProfileName", "ProfileName", "ProfileHueSatMapDims", "ProfileEmbedPolicy", "ProfileCopyright", "PreviewApplicationName", "PreviewApplicationVersion", "PreviewSettingsName",
"PreviewSettingsDigest", "PreviewColorSpace", "SubTileBlockSize", "RowInterleaveFactor", "ProfileLookTableDims", "NewSubfileType", "SubfileType", "ImageWidth", "ImageLength",
"Compression", "ResolutionUnit", "JPEGInterchangeFormat", "JPEGInterchangeFormatLength", "ExposureProgram", "ISOSpeedRatings", "SensitivityType", "StandardOutputSensitivity", "RecommendedExposureIndex", "ISOSpeed",
"ISOSpeedLatitudeyyy", "ISOSpeedLatitudezzz", "MeteringMode", "LightSource", "FlashInfo", "ColorSpace", "PixelXDimension", "PixelYDimension", "InteroperabilityTag", "FocalPlaneResolutionUnit",
"SubjectLocation", "SensingMethod", "CustomRendered", "ExposureMode", "FocalLengthIn35mmFilm", "SceneCaptureType", "GainControl", "Contrast", "Saturation", "Sharpness",
"SubjectDistanceRange",  "RelatedImageWidth", "RelatedImageLength",

(*IPTC*)"ARMVersion", "ARMId", "FileVersion", "FileFormat", "ModelVersion", "PreviewVersion", "PreviewFormat", "RecordVersion",

(*XMP*)
"BlueHue", "BlueSaturation", "Brightness", "ChromaticAberrationB", "ChromaticAberrationR", "ColorNoiseReduction", "Contrast", "CropUnits", "GreenHue", "GreenSaturation",
"LuminanceSmoothing", "RedHue", "RedSaturation", "Saturation", "Shadows", "ShadowTint", "Temperature", "Tint", "VignetteAmount", "VignetteMidpoint",
"AutoLateralCA", "AutoWhiteVersion", "Blacks2012", "BlueHue", "BlueSaturation", "Brightness", "ChromaticAberrationB",
"ChromaticAberrationR", "Clarity", "Clarity2012", "ColorNoiseReduction", "ColorNoiseReductionDetail", "ColorNoiseReductionSmoothness",
"Contrast", "Contrast2012", "CropConstrainToWarp", "Defringe", "DefringeGreenAmount", "DefringeGreenHueHi", "DefringeGreenHueLo",
"DefringePurpleAmount", "DefringePurpleHueHi", "DefringePurpleHueLo", "FillLight", "GrainAmount", "GrainFrequency", "GrainSize",
"GrayMixerAqua", "GrayMixerBlue", "GrayMixerGreen", "GrayMixerMagenta", "GrayMixerOrange", "GrayMixerPurple", "GrayMixerRed", 
"GrayMixerYellow", "GreenHue", "GreenSaturation", "HighlightRecovery", "Highlights2012", "HueAdjustmentAqua", "HueAdjustmentBlue",
"HueAdjustmentGreen", "HueAdjustmentMagenta", "HueAdjustmentOrange", "HueAdjustmentPurple", "HueAdjustmentRed", "HueAdjustmentYellow",
"IncrementalTemperature", "IncrementalTint", "LensManualDistortionAmount", "LensProfileChromaticAberrationScale", "LensProfileDistortionScale",
"LensProfileEnable", "LensProfileVignettingScale", "LuminanceAdjustmentAqua", "LuminanceAdjustmentBlue", "LuminanceAdjustmentGreen", 
"LuminanceAdjustmentMagenta", "LuminanceAdjustmentOrange", "LuminanceAdjustmentPurple", "LuminanceAdjustmentRed", "LuminanceAdjustmentYellow",
"LuminanceNoiseReductionContrast", "LuminanceNoiseReductionDetail", "LuminanceSmoothing", "ParametricDarks", "ParametricHighlights", 
"ParametricHighlightSplit", "ParametricLights", "ParametricMidtoneSplit", "ParametricShadows", "ParametricShadowSplit", "PerspectiveAspect",
"PerspectiveHorizontal", "PerspectiveScale", "PerspectiveUpright", "PerspectiveVertical", "PostCropVignetteAmount", "PostCropVignetteFeather",
"PostCropVignetteHighlightContrast", "PostCropVignetteMidpoint", "PostCropVignetteRoundness", "PostCropVignetteStyle", "RedHue", "RedSaturation",
"Saturation", "SaturationAdjustmentAqua", "SaturationAdjustmentBlue", "SaturationAdjustmentGreen", "SaturationAdjustmentMagenta", "SaturationAdjustmentOrange",
"SaturationAdjustmentPurple", "SaturationAdjustmentRed", "SaturationAdjustmentYellow", "Shadows", "Shadows2012", "ShadowTint", "SharpenDetail", 
"SharpenEdgeMasking", "Sharpness", "Smoothness", "SplitToningBalance", "SplitToningHighlightHue", "SplitToningHighlightSaturation", 
"SplitToningShadowHue", "SplitToningShadowSaturation", "ColorTemperature", "Tint", "UprightCenterMode", "UprightFocalMode",
"UprightTransformCount", "UprightVersion", "Vibrance", "VignetteAmount", "VignetteMidpoint", "Exposure2012", "SharpenRadius", "ColorMode", "id",
"InitialViewHeadingDegrees", "InitialViewPitchDegrees", "InitialViewRollDegrees", "SourcePhotosCount", "CroppedAreaImageWidthPixels", "CroppedAreaImageHeightPixels",
"FullPanoWidthPixels", "FullPanoHeightPixels", "CroppedAreaLeftPixels", "CroppedAreaTopPixels"
};

RealTags = { "WhitePoint", "PrimaryChromaticities", "YCbCrCoefficients", "ReferenceBlackWhite", "BatteryLevel", "ProfileLookTableData",
"BlackLevel", "BlackLevelDeltaH", "BlackLevelDeltaV", "DefaultScale", "ColorMatrix1", "ColorMatrix2", "AsShotWhiteXY", "NoiseReductionApplied", "ForwardMatrix1", "ForwardMatrix2", "NoiseProfile",
"CameraCalibration1", "CameraCalibration2", "ReductionMatrix1", "ReductionMatrix2", "AnalogBalance", "BaselineExposure", "BaselineNoise", "BaselineSharpness", "LinearResponseLimit", "LensInfo",
"ChromaBlurRadius", "AntiAliasStrength", "ShadowScale", "BestQualityScale", "AsShotPreProfileMatrix", "CurrentPreProfileMatrix", "ProfileHueSatMapData1", "ProfileHueSatMapData2", "ProfileToneCurve",
"ExposureTime", "FNumber", "ShutterSpeedValue", "CompressedBitsPerPixel", "ApertureValue", "FocalLengthIn35mmFilm", "DigitalZoomRatio", "LensSpecification",
"BrightnessValue", "ExposureBiasValue", "MaxApertureValue", "SubjectDistance", "FocalLength", "FlashEnergy", "FocalPlaneXResolution", "FocalPlaneYResolution", "ExposureIndex",
"XResolution", "YResolution", "GPSDOP","GPSSpeed", "GPSTrack", "GPSImgDirection",

"CropTop", "CropLeft", "CropBottom", "CropRight", "CropAngle", "CropWidth", "CropHeight", "Exposure", "CorrectionAmount", "LocalBrightness", "LocalClarity",
"LocalClarity2012", "LocalContrast", "LocalContrast2012", "LocalDefringe", "LocalExposure", "LocalExposure2012", "LocalHighlights2012", "LocalLuminanceNoise", 
"LocalMoire", "LocalSaturation", "LocalShadows2012", "LocalSharpness", "LocalTemperature", "LocalTint", "LocalToningHue", "LocalToningSaturation",
"MaskValue", "Radius", "Flow", "CenterWeight", "PoseHeadingDegrees", "PosePitchDegrees", "PoseRollDegrees", "InitialHorizontalFOVDegrees", "InitialCameraDolly"
};

RationalTags = {"XResolution", "YResolution", "WhitePoint", "PrimaryChromaticities", "YCbCrCoefficients", "ReferenceBlackWhite", "BatteryLevel", 
"ExposureTime", "FNumber", "CompressedBitsPerPixel", "ApertureValue", "MaxApertureValue", "FocalLength", "FlashEnergy", "FocalPlaneXResolution", "FocalPlaneYResolution", 
"ExposureIndex", "BlackLevel", "DefaultScale", "AnalogBalance", "AsShotWhiteXY", "BaselineNoise", "BaselineSharpness", "LinearResponseLimit", "LensInfo", "ChromaBlurRadius",
"AntiAliasStrength", "BestQualityScale", "NoiseReductionApplied", "SubjectDistance", "DigitalZoomRatio", 
"ShutterSpeedValue", "BrightnessValue", "ExposureBiasValue", "BlackLevelDeltaH", "BlackLevelDeltaV", "ColorMatrix1", "ColorMatrix2", "CameraCalibration1",
"CameraCalibration2", "ReductionMatrix1", "ReductionMatrix2", "BaselineExposure", "ShadowScale", "AsShotPreProfileMatrix", "CurrentPreProfileMatrix", "ForwardMatrix1", 
"ForwardMatrix2"
};

StringTags = {(*Exif*) "ProcessingSoftware", "ImageDescription", "Make", "Model", "Software", "Artist", "HostComputer",
"InkNames", "TargetPrinter", "ImageID", "Copyright", "SpectralSensitivity", "SecurityClassification", "ImageHistory", "UniqueCameraModel",
"CameraSerialNumber", "DateTime", "PreviewDateTime", "SpectralSensitivity", "SubSecTime", "SubSecTimeOriginal", "SubSecTimeDigitized", "DateTimeDigitized", "GPSDateStamp",
"RelatedSoundFile", "ImageUniqueID", "CameraOwnerName", "BodySerialNumber", "LensMake", "LensModel", "LensSerialNumber", "DateTimeOriginal", "InteroperabilityIndex", "RelatedImageFileFormat", "GPSLatitudeRef", "GPSLongitudeRef", "GPSSatellites", "GPSStatus", "GPSMeasureMode", "GPSSpeedRef", "GPSTrackRef",
"GPSImgDirectionRef", "GPSMapDatum", "GPSDestLatitudeRef", "GPSDestLongitudeRef", "GPSDestBearingRef", "GPSDestDistanceRef", "JPEGTables", "InterColorProfile", "OECF", "SpatialFrequencyResponse", "Noise", "PrintImageMatching", "OriginalRawFileData",
"AsShotICCProfile", "CurrentICCProfile", "RawImageDigest", "OriginalRawFileDigest", "Opcodelist1", "Opcodelist2", "Opcodelist3", "OECF", "ExifVersion", "ComponentsConfiguration", "MakerNote", "UserComment", "FlashpixVersion", "SpatialFrequencyResponse",
"FileSource", "SceneType", "CFAPattern", "DeviceSettingDescription", "InteroperabilityVersion", "Opcodelist2", "Opcodelist3", "ColorSpace", "FocalPlaneResolutionUnit", "GPSDifferential",
"SecurityClassification", "ExposureProgram", "SensitivityType", "Predictor", "ExtraSamples", "InkSet", "Orientation", "ResolutionUnit", "FillOrder", "OldSubfileType", "Thresholding",
"SubfileType", "Compression", "PhotometricInterpretation", "PlanarConfiguration", "YCbCrPositioning", "Contrast", "CustomRendered", "ExposureMode", "GainControl", "LightSource",
"FlashInfo", "MeteringMode", "WhiteBalance", "SubjectDistanceRange", "Saturation", "SceneCaptureType", "SensingMethod", "FileSource", "Sharpness", 

(*XMP*) "contributor", "coverage", "creator", "date", "description", "format", "identifier",
"language", "publisher", "rights", "source", "subject", "title", "type", "Thumbnails", "Rating", "Nickname", "ModifyDate", "MetadataDate", "Label", "CreatorTool",
"CreateDate", "BaseURL", "Identifier", "Advisory", "Certificate", "Marked", "Owner", "UsageTerms", "WebStatement", "DerivedFrom", "DocumentID", "InstanceID", "ManagedFrom", "Manager", "ManageTo", "ManageUI",
"ManagerVariant", "RenditionClass", "RenditionParams", "VersionID", "Versions", "LastURL", "RenditionOf", "SaveID", "JobRef", 
"MaxPageSize", "NPages", "Fonts", "Colorants", "PlateNames", "AutoBrightness", "AutoContrast", "AutoExposure", "AutoShadows", "BlueHue", "BlueSaturation", "Brightness",
"CameraProfile", "ChromaticAberrationB", "ChromaticAberrationR", "ColorNoiseReduction", "Contrast", "CropTop", "CropLeft", "CropBottom", "CropRight",
"CropAngle", "CropWidth", "CropHeight", "CropUnits", "Exposure", "GreenHue", "GreenSaturation", "HasCrop", "HasSettings", "LuminanceSmoothing",
"RawFileName", "RedHue", "RedSaturation", "Saturation", "Shadows", "ShadowTint", "Sharpness", "Temperature", "Tint", "ToneCurve", "ToneCurveName",
"Version", "VignetteAmount", "VignetteMidpoint", "WhiteBalance", "Tagslist", "CaptionsAuthorNames", "CaptionsDateTimeStamps", "ImageHistory", "LensCorrectionSettings",
"Tagslist", "CaptionsAuthorNames", "CaptionsDateTimeStamps", "ImageHistory", "LensCorrectionSettings", "CameraSerialNumber", "DateAcquired", "FlashManufacturer", "FlashModel", "LastKeywordIPTC",
"LastKeywordXMP", "LensManufacturer", "Rating", "Keywords", "PDFVersion", "Producer", "AuthorsPosition", "CaptionWriter", "Category", "City", "Country", "Credit", "DateCreated", "Headline", "Instructions",
"Source", "State", "SupplementalCategories", "TransmissionReference", "Urgency", "ICCProfile", "AutoRotated", "Software",
"CaptureSoftware", "StitchingSoftware", "ProjectionType", "FirstPhotoDate", "LastPhotoDate","CreatorContactInfo", "CiAdrExtadr", "CiAdrCity", "CiAdrRegion", "CiAdrPcode", "CiAdrCtry", "CiEmailWork", "CiTelWork", "CiUrlWork", 
"IntellectualGenre", "Scene", "SubjectCode", "Location", "CountryCode",

(*IPTC*) "ServiceId", "EnvelopeNumber", "ProductId", "EnvelopePriority", "CharacterSet",
"UNO", "TimeSent", "DateSent", "TimeSent", "ObjectType", "ObjectAttribute", "ObjectName", "EditStatus", "EditorialUpdate", "Urgency", "Subject", "Category",
"FixtureId", "Keywords", "LocationCode", "LocationName", "SpecialInstructions", "ActionAdvised", "ReferenceService", "ReferenceNumber", "Program", "ProgramVersion",
"ObjectCycle", "Byline", "BylineTitle", "City", "SubLocation", "ProvinceState", "CountryCode", "CountryName", "TransmissionReference", "Headline", "Credit", "Source",
"Copyright", "Contact", "Caption", "Writer", "ImageType", "ImageOrientation", "Language", "AudioType", "AudioRate", "AudioResolution", "AudioDuration", "AudioOutcue",
"DigitizationDate", "DateCreated", "ReferenceDate", "ExpirationDate", "ReleaseDate", "DigitizationTime", "TimeCreated", "ExpirationTime", "ReleaseTime",
"TimeSent", "ReleaseTime", "DigitizationTime", "RasterizedCaption", "Preview", "SuppCategory"};

QuantityTags = {"SubSecTime", 
   "SubSecTimeOriginal", "SubSecTimeDigitized", "ExposureTime", 
   "FocalLength", "Lens", "FocalLengthIn35mmFilm",
   "GPSAltitude", "TargetShutterSpeed", "GPSLatitude", "GPSLongitude", "Temperature", "SubjectDistance", "GPSImgDirection", "GPSTrack", "GPSSpeed", 
   "BaselineExposure", "ExposureBiasValue"};
   
DateTags = {"DateTime", "DateTimeOriginal", "DateTimeDigitized", "PreviewDateTime", "GPSDateStamp",
   "DateSent", "DigitizationDate", "DateCreated", "ReferenceDate", "ExpirationDate", "ReleaseDate",
 
   "ModifyDate", "MetadataDate", "CreateDate", "CaptionsDateTimeStamps", "DateAcquired", "DateCreated", "DeprecatedOn",
   
   "When", "DateSent", "SentDate", "FirstPhotoDate", "LastPhotoDate"
   };
   
TimeTags = { "TimeSent", "DigitizationTime", "TimeCreated", "ReleaseTime", "GPSTimeStamp"}

MultiValues = { "BitsPerSample", "HalftoneHints", "YCbCrSubSampling", 
	"CFARepeatPatternDim", "TimeZoneOffset", "WhitePoint", "PrimaryChromaticities", 
	"YCbCrCoefficients", "ReferenceBlackWhite", "LensInfo", "CameraInfo",
	"Version", "LensSpecification", "SubjectArea" , "StripByteCounts", "StripOffsets", "RowsPerStrip", "CFAPattern", "PrintImageMatching",
	
	"ToneCurvePV2012", "ToneCurvePV2012Blue", "ToneCurvePV2012Green", "ToneCurvePV2012Red", "ToneCurve", "ToneCurveBlue", "ToneCurveGreen", "ToneCurveRed"
};

BooleanTags = {
	"FlashUsed",
	
	"AlreadyApplied", "AutoBrightness", "AutoContrast", "AutoExposure", "AutoShadows", "CircGradBasedCorrActive", "ConvertToGrayscale", 
	"HasCrop", "HasSettings", "CorrectionActive",
	
	"Marked", "AutoRotated", "ExposureLockUsed", "UsePanoramaViewer"
};

GPSTags = {
	"GPSVersionID", "GPSLatitudeRef", "GPSLatitude", "GPSLongitudeRef", "GPSLongitude", "GPSAltitudeRef", "GPSAltitude",
"GPSTimeStamp", "GPSSatellites", "GPSStatus", "GPSMeasureMode", "GPSDOP", "GPSSpeedRef", "GPSSpeed", "GPSTrackRef", "GPSTrack", "GPSImgDirectionRef",
"GPSImgDirection", "GPSMapDatum", "GPSDestLatitudeRef", "GPSDestLatitude", "GPSDestLongitudeRef", "GPSDestLongitude", "GPSDestBearingRef", "GPSDestBearing",
"GPSDestDistanceRef", "GPSDestDistance", "GPSProcessingMethod", "GPSAreaInformation", "GPSDateStamp", "GPSDifferential"
};

IPTCEnvelope = { "ModelVersion", "Destination", "FileFormat", "FileVersion", "ServiceId", "EnvelopeNumber", "ProductId", "EnvelopePriority", "DateSent", "TimeSent", "CharacterSet",
  "UNO", "ARMId", "ARMVersion"
};

$AllExif = { "BitsPerSample", "Compression", "PhotometricInterpretation", "Threshholding", "CellWidth", "CellLength", "FillOrder", "StripOffsets", "Orientation", "SamplesPerPixel", "RowsPerStrip", "StripByteCounts", "PlanarConfiguration",
		"GrayResponseUnit", "GrayResponseCurve", "T4Options", "T6Options", "ResolutionUnit", "TransferFunction", "Predictor", "ColorMap", "HalftoneHints", "TileWidth", "TileLength", "TileOffsets", "TileByteCounts", "SubIFDs",
		"InkSet", "NumberOfInks", "DotRange", "ExtraSamples", "SampleFormat", "SMinSampleValue", "SMaxSampleValue", "TransferRange", "ClipPath", "XClipPathUnits", "YClipPathUnits", "Indexed", "OPIProxy", "JPEGProc", "JPEGInterchangeFormat",
		"JPEGInterchangeFormatLength", "JPEGRestartInterval", "JPEGLosslessPredictors", "JPEGPointTransforms", "JPEGQTables", "JPEGDCTables", "JPEGACTables", "YCbCrSubSampling", "YCbCrPositioning", "XMLPacket", "Rating", "RatingPercent",
		"CFARepeatPatternDim", "IPTCNAA", "SpectralSensitivity","Interlace", "LocalizedCameraModel", "CFAPlaneColor", "CFALayout", "LinearizationTable", "TimeZoneOffset", "SelfTimerMode",
		"ImageNumber", "TIFFEPStandardID", "XPTitle", "XPComment", "XPAuthor", "XPKeywords", "XPSubject", "DNGVersion", "DNGBackwardVersion", "BlackLevelRepeatDim", "WhiteLevel", "DefaultCropOrigin", "DefaultCropSize", "AsShotNeutral",
		"BayerGreenSplit", "DNGPrivateData", "MakerNoteSafety", "CalibrationIlluminant1", "CalibrationIlluminant2", "RawDataUniqueID", "OriginalRawFileName", "ActiveArea", "MaskedAreas", "ColorimetricReference", "CameraCalibrationSignature",
		"ProfileCalibrationSignature", "AsShotProfileName", "ProfileName", "ProfileHueSatMapDims", "ProfileEmbedPolicy", "ProfileCopyright", "PreviewApplicationName", "PreviewApplicationVersion", "PreviewSettingsName", "PreviewSettingsDigest",
		"PreviewColorSpace", "SubTileBlockSize", "RowInterleaveFactor", "ProfileLookTableDims", "NewSubfileType", "SubfileType", "ImageWidth", "ImageLength", "ExposureProgram", "ISOSpeedRatings", "SensitivityType", "StandardOutputSensitivity",
		"RecommendedExposureIndex", "ISOSpeed", "ISOSpeedLatitudeyyy", "ISOSpeedLatitudezzz", "MeteringMode", "LightSource", "FlashInfo", "ColorSpace", "PixelXDimension", "PixelYDimension", "InteroperabilityTag", "FocalPlaneResolutionUnit",
		"SubjectLocation", "SensingMethod", "CustomRendered", "ExposureMode", "WhiteBalance", "SceneCaptureType", "GainControl", "Contrast", "Saturation", "Sharpness", "SubjectDistanceRange", "RelatedImageWidth", "RelatedImageLength",
		"ProcessingSoftware", "ImageDescription", "Make", "Model", "Software", "DateTime", "Artist", "HostComputer", "InkNames", "TargetPrinter", "ImageID", "Copyright", "SecurityClassification", "ImageHistory", "UniqueCameraModel",
		"CameraSerialNumber", "PreviewDateTime", "DateTimeOriginal", "DateTimeDigitized", "SubSecTime", "SubSecTimeOriginal", "SubSecTimeDigitized", "RelatedSoundFile", "ImageUniqueID", "CameraOwnerName", "BodySerialNumber", "LensMake",
		"LensModel", "LensSerialNumber", "InteroperabilityIndex", "RelatedImageFileFormat", "JPEGTables", "InterColorProfile", "SpatialFrequencyResponse", "Noise", "PrintImageMatching", "OriginalRawFileData", "AsShotICCProfile",
		"CurrentICCProfile", "RawImageDigest", "OriginalRawFileDigest", "OpcodeList1", "OpcodeList2", "OpcodeList3", "OECF", "ExifVersion", "ComponentsConfiguration",(* "UserComment",*) "FlashpixVersion", "FileSource", "SceneType",
		"CFAPattern", "DeviceSettingDescription", "InteroperabilityVersion", "XResolution", "YResolution", "WhitePoint", "PrimaryChromaticities", "YCbCrCoefficients", "ReferenceBlackWhite", "BatteryLevel", "ProfileLookTableData",
		"BlackLevel", "BlackLevelDeltaH", "BlackLevelDeltaV", "DefaultScale", "ColorMatrix1", "ColorMatrix2", "AsShotWhiteXY", "NoiseReductionApplied", "ForwardMatrix1", "ForwardMatrix2", "NoiseProfile", "CameraCalibration1",
		"CameraCalibration2", "ReductionMatrix1", "ReductionMatrix2", "AnalogBalance", "BaselineExposure", "BaselineNoise", "BaselineSharpness", "LinearResponseLimit", "ChromaBlurRadius", "AntiAliasStrength",
		"ShadowScale", "BestQualityScale", "AsShotPreProfileMatrix", "CurrentPreProfileMatrix", "ProfileHueSatMapData1", "ProfileHueSatMapData2", "ProfileToneCurve", "ExposureTime", "FNumber", "ShutterSpeedValue", "CompressedBitsPerPixel",
		"ApertureValue", "FocalLengthIn35mmFilm", "DigitalZoomRatio", "LensSpecification", "BrightnessValue", "ExposureBiasValue", "MaxApertureValue", "SubjectDistance", "FocalLength", "FlashEnergy", "FocalPlaneXResolution",
		"FocalPlaneYResolution", "ExposureIndex", "GPSVersionID", "GPSLatitudeRef", "GPSLatitude", "GPSLongitudeRef", "GPSLongitude", "GPSAltitudeRef", "GPSAltitude", "GPSTimeStamp", "GPSSatellites", "GPSStatus", "GPSMeasureMode",
		"GPSDOP", "GPSSpeedRef", "GPSSpeed", "GPSTrackRef", "GPSTrack", "GPSImgDirectionRef", "GPSImgDirection", "GPSMapDatum", "GPSDestLatitudeRef", "GPSDestLatitude", "GPSDestLongitudeRef", "GPSDestLongitude", "GPSDestBearingRef",
		"GPSDestBearing", "GPSDestDistanceRef", "GPSDestDistance", "GPSProcessingMethod", "GPSAreaInformation", "GPSDateStamp", "GPSDifferential", "Gamma", "PhotographicSensitivity"
	};
	
$AllIPTC = {
		"DateSent", "ReleaseDate", "ExpirationDate", "ReferenceDate", "DateCreated", "DigitizationDate", "ModelVersion", "FileFormat", "FileVersion", "ARMId", "ARMVersion", "RecordVersion",
		"PreviewFormat", "PreviewVersion", "Destination", "ServiceId", "EnvelopeNumber", "ProductId", "EnvelopePriority", "CharacterSet", "Writer", "UNO", "ObjectType", "ObjectAttribute",
		"ObjectName", "EditStatus", "EditorialUpdate", "Urgency", "Subject", "Category", "SuppCategory", "FixtureId", "Keywords", "LocationCode", "LocationName", "SpecialInstructions",
		"ActionAdvised", "ReferenceService", "ReferenceNumber", "Program", "ProgramVersion", "ObjectCycle", "Byline", "BylineTitle", "City", "SubLocation", "ProvinceState", "CountryCode",
		"CountryName", "TransmissionReference", "Headline", "Credit", "Source", "Copyright", "Contact", "Caption", "Writer", "ImageType", "ImageOrientation", "Language", "AudioType",
		"AudioRate", "AudioResolution", "AudioDuration", "AudioOutcue", "TimeSent", "ReleaseTime", "ExpirationTime", "TimeCreated", "DigitizationTime", "RasterizedCaption", "Preview"
	};

$AllXMP = { "Advisory", "AuthorsPosition", "AutoBrightness", "AutoContrast", "AutoExposure", "AutoLateralCA", "AutoShadows", "AutoWhiteVersion", "BaseURL",
	"Blacks2012", "BlueHue", "BlueSaturation", "Brightness", "CameraProfile", "CameraSerialNumber", "CaptionsAuthorNames", "CaptionsDateTimeStamps", "CaptionWriter", "Category", "CenterWeight",
	"Certificate", "ChromaticAberrationB", "ChromaticAberrationR", "City", "Clarity", "Clarity2012", "Colorants", "ColorNoiseReduction", "ColorNoiseReductionDetail", "ColorNoiseReductionSmoothness",
	"ColorTemperature", "Contrast", "Contrast2012", "contributor", "CorrectionAmount", "Country", "coverage", "CreateDate", "creator", "CreatorTool", "Credit", "CropAngle", "CropBottom",
	"CropConstrainToWarp", "CropHeight", "CropLeft", "CropRight", "CropTop", "CropUnits", "CropWidth", "date", "DateAcquired", "DateCreated", "Defringe", "DefringeGreenAmount", "DefringeGreenHueHi",
	"DefringeGreenHueLo", "DefringePurpleAmount", "DefringePurpleHueHi", "DefringePurpleHueLo", "DerivedFrom", "description", "DocumentID", "Exposure", "FillLight", "FlashManufacturer", "FlashModel",
	"Flow", "Fonts", "format", "GrainAmount", "GrainFrequency", "GrainSize", "GrayMixerAqua", "GrayMixerBlue", "GrayMixerGreen", "GrayMixerMagenta", "GrayMixerOrange", "GrayMixerPurple", "GrayMixerRed",
	"GrayMixerYellow", "GreenHue", "GreenSaturation", "HasCrop", "HasSettings", "Headline", "HighlightRecovery", "Highlights2012", "History", "HueAdjustmentAqua", "HueAdjustmentBlue", "HueAdjustmentGreen",
	"HueAdjustmentMagenta", "HueAdjustmentOrange", "HueAdjustmentPurple", "HueAdjustmentRed", "HueAdjustmentYellow", "identifier", "Identifier", "ImageHistory", "IncrementalTemperature", "IncrementalTint",
	"InstanceID", "Instructions", "JobRef", "Keywords", "Label", "language", "LastKeywordIPTC", "LastKeywordXMP", "LastURL", "LensCorrectionSettings", "LensManualDistortionAmount", "LensManufacturer",
	"LensModel", "LensProfileChromaticAberrationScale", "LensProfileDistortionScale", "LensProfileEnable", "LensProfileVignettingScale", "LocalBrightness", "LocalClarity", "LocalClarity2012", "LocalContrast",
	"LocalContrast2012", "LocalDefringe", "LocalExposure", "LocalExposure2012", "LocalHighlights2012", "LocalLuminanceNoise", "LocalMoire", "LocalSaturation", "LocalShadows2012", "LocalSharpness",
	"LocalTemperature", "LocalTint", "LocalToningHue", "LocalToningSaturation", "LuminanceAdjustmentAqua", "LuminanceAdjustmentBlue", "LuminanceAdjustmentGreen", "LuminanceAdjustmentMagenta", "LuminanceAdjustmentOrange",
	"LuminanceAdjustmentPurple", "LuminanceAdjustmentRed", "LuminanceAdjustmentYellow", "LuminanceNoiseReductionContrast", "LuminanceNoiseReductionDetail", "LuminanceSmoothing", "ManagedFrom", "Manager",
	"ManagerVariant", "ManageTo", "ManageUI", "Marked", "MaskValue", "MaxPageSize", "MetadataDate", "ModifyDate", "Nickname", "NPages", "Owner", "ParametricDarks", "ParametricHighlights",
	"ParametricHighlightSplit", "ParametricLights", "ParametricMidtoneSplit", "ParametricShadows", "ParametricShadowSplit", "PDFVersion", "PerspectiveAspect", "PerspectiveHorizontal", "PerspectiveScale",
	"PerspectiveUpright", "PerspectiveVertical", "PlateNames", "PostCropVignetteAmount", "PostCropVignetteFeather", "PostCropVignetteHighlightContrast", "PostCropVignetteMidpoint", "PostCropVignetteRoundness",
	"PostCropVignetteStyle", "Producer", "publisher", "Radius", "Rating", "RawFileName", "RedHue", "RedSaturation", "RenditionClass", "RenditionOf", "RenditionParams", "rights", "Saturation", "SaturationAdjustmentAqua",
	"SaturationAdjustmentBlue", "SaturationAdjustmentGreen", "SaturationAdjustmentMagenta", "SaturationAdjustmentOrange", "SaturationAdjustmentPurple", "SaturationAdjustmentRed", "SaturationAdjustmentYellow",
	"SaveID", "Shadows", "Shadows2012", "ShadowTint", "SharpenDetail", "SharpenEdgeMasking", "Sharpness", "Smoothness", "source", "Source", "SplitToningBalance", "SplitToningHighlightHue", "SplitToningHighlightSaturation",
	"SplitToningShadowHue", "SplitToningShadowSaturation", "State", "subject", "SupplementalCategories", "Tagslist", "TagsList", "Temperature", "Thumbnails", "Tint", "title", "ToneCurve", "ToneCurveBlue",
	"ToneCurveGreen", "ToneCurveName", "ToneCurvePV2012", "ToneCurvePV2012Blue", "ToneCurvePV2012Green", "ToneCurvePV2012Red", "ToneCurveRed", "TransmissionReference", "type", "UprightCenterMode",
	"UprightFocalMode", "UprightTransformCount", "UprightVersion", "Urgency", "UsageTerms", "Version", "VersionID", "Versions", "Vibrance", "VignetteAmount", "VignetteMidpoint", "WebStatement", "WhiteBalance", "Flash", "ColorMode", "ICCProfile",
	"UsePanoramaViewer", "CaptureSoftware", "StitchingSoftware", "ProjectionType", "PoseHeadingDegrees", "PosePitchDegrees", "PoseRollDegrees", "InitialViewHeadingDegrees", "InitialViewPitchDegrees", "InitialViewRollDegrees",
	"InitialHorizontalFOVDegrees", "FirstPhotoDate", "LastPhotoDate", "SourcePhotosCount", "ExposureLockUsed", "CroppedAreaImageWidthPixels", "CroppedAreaImageHeightPixels", "FullPanoWidthPixels", "FullPanoHeightPixels",
	"CroppedAreaLeftPixels", "CroppedAreaTopPixels", "InitialCameraDolly", "CreatorContactInfo", "CiAdrExtadr", "CiAdrCity", "CiAdrRegion", "CiAdrPcode", "CiAdrCtry", "CiEmailWork", "CiTelWork", "CiUrlWork", 
	"IntellectualGenre", "Scene", "SubjectCode", "Location", "CountryCode"
	};
	

ExifPositiveValuesOnly = {"XResolution", "YResolution", "CompressedBitsPerPixel",  "SubjectDistance", "FocalPlaneXResolution", 
   "FocalPlaneYResolution", "ExposureIndex", "BlackLevelDeltaH", "BlackLevelDeltaV", "LinearResponseLimit", "ChromaBlurRadius"."AntiAliasStrength",
   "BestQualityScale", "DigitalZoomRatio", "GPSLongitude", "NewSubfileType", "ImageWidth", "ImageLength", "StripOffsets", "RowsPerStrip", "StripByteCounts", 
   "SubIFDs", "JPEGInterchangeFormat", "JPEGInterchangeFormatLength", "IPTCNAA", "ExifTag", "GPSTag", "ProfileHueSatMapDims", "RowInterleaveFactor", 
   "ProfileLookTableDims", "PixelXDimension", "PixelYDimension", "InteroperabilityTag", "RelatedImageWidth", "RelatedImageHeight", "ISOSpeed", "BitsPerSample", 
   "SensitivityType", "GrayResponseCurve", "SubTileBlockSize", "StandardOutputSensitivity", "RecommendedExposureIndex", "ISOSpeedLatitudeyyy", "ISOSpeedLatitudezzz", 
   "FocalLengthIn35mmFilm", "BaselineExposure", "FlashEnergy"};
	
(**********)
(**PHOTO***)
(**********)
ExportExifPhotoInt = { "ExposureProgram", "SubjectDistanceRange", "Sharpness", "Saturation", "Contrast", "GainControl", "SceneCaptureType", "FocalLengthIn35mmFilm",
	"WhiteBalance", "ExposureMode", "CustomRendered", "SensingMethod", "SubjectLocation", "FocalPlaneResolutionUnit", "InteroperabilityTag", "PixelYDimension", 
	"PixelXDimension", "ColorSpace", "FlashInfo", "LightSource", "MeteringMode", "ISOSpeedRatings"
};

ExportExifPhotoRat = { "DigitalZoomRatio", "ExposureIndex", "FocalPlaneYResolution", "FocalPlaneXResolution", "FlashEnergy", "FocalLength", 
	"SubjectDistance", "MaxApertureValue", "ExposureBiasValue", "BrightnessValue", "ApertureValue", "ShutterSpeedValue", "CompressedBitsPerPixel",
	"ISOSpeedLatitudezzz", "ISOSpeedLatitudeyyy", "ISOSpeed", "RecommendedExposureIndex", "StandardOutputSensitivity", "SensitivityType", 
	"FNumber", "ExposureTime", 	"FNumber", "ExposureTime"
};

ExportExifPhotoString = { "LensSerialNumber", "LensModel", "LensMake", "BodySerialNumber", "CameraOwnerName", "ImageUniqueID", "DeviceSettingDescription", "CFAPattern",
	"SceneType", "FileSource", "SpatialFrequencyResponse", "RelatedSoundFile", "FlashpixVersion", "SubSecTimeDigitized", "SubSecTimeOriginal", "SubSecTime", "UserComment",
	"ComponentsConfiguration", "DateTimeDigitized", "DateTimeOriginal", "ExifVersion", "OECF", "SpectralSensitivity",
	
	"SubjectArea", "LensSpecification"
};

(**********)
(****IOP***)
(**********)
ExportExifIopNumber = { "RelatedImageLength", "RelatedImageWidth"
};

ExportExifIopString = { "RelatedImageFileFormat", "InteroperabilityVersion", "InteroperabilityIndex"
};

(**********)
(**Image***)
(**********)
ExportExifImageInt = { "ProfileLookTableDims", "RowInterleaveFactor", "SubTileBlockSize", "PreviewColorSpace", "PreviewSettingsDigest", "PreviewSettingsName", "PreviewApplicationVersion",
	"PreviewApplicationName", "ProfileCopyright", "ProfileEmbedPolicy", "ProfileToneCurve", "ProfileHueSatMapDims", "ProfileName", "AsShotProfileName", "ProfileCalibrationSignature",
	"CameraCalibrationSignature", "ColorimetricReference", "MaskedAreas", "ActiveArea", "OriginalRawFileName", "RawDataUniqueID", "CalibrationIlluminant2", "CalibrationIlluminant1",
	"MakerNoteSafety", "DNGPrivateData", "BayerGreenSplit", "AsShotNeutral", "DefaultCropSize", "DefaultCropOrigin", "WhiteLevel", "BlackLevelRepeatDim", "LinearizationTable",
	"CFALayout", "CFAPlaneColor", "LocalizedCameraModel", "DNGBackwardVersion", "DNGVersion", "XPSubject", "XPKeywords", "XPAuthor", "XPComment", "XPTitle", 
	"TIFFEPStandardID", "ImageNumber", "SelfTimerMode", "TimeZoneOffset", "Interlace",
	"GPSTag", "ExifTag", "ImageResources", "IPTCNAA", "CFAPattern", "CFARepeatPatternDim", "RatingPercent", "Rating", "XMLPacket", "YCbCrPositioning",
	"YCbCrCoefficients", "JPEGACTables", "JPEGDCTables", "JPEGQTables", "JPEGPointTransforms", "JPEGLosslessPredictors", "JPEGRestartInterval", "JPEGInterchangeFormatLength",
	"JPEGInterchangeFormat", "JPEGProc", "OPIProxy", "Indexed", "YClipPathUnits", "XClipPathUnits", "ClipPath", "TransferRange", "SMaxSampleValue", "SMinSampleValue", "SampleFormat",
	"ExtraSamples", "DotRange", "NumberOfInks", "InkSet", "SubIFDs", "TileByteCounts", "TileOffsets", "TileLength", "TileWidth", "HalftoneHints", "ColorMap", "Predictor",
	"TransferFunction", "PageNumber", "ResolutionUnit", "T6Options", "T4Options", "GrayResponseCurve", "GrayResponseUnit", "PlanarConfiguration", "StripByteCounts", "RowsPerStrip",
	"SamplesPerPixel", "Orientation", "StripOffsets", "FillOrder", "CellLength", "CellWidth", "Thresholding", "PhotometricInterpretation", "Compression", "BitsPerSample",
	"ImageLength", "ImageWidth", "SubfileType", "NewSubfileType"
};

ExportExifImageReal = { "NoiseProfile", "ProfileLookTableData", "ProfileHueSatMapData2", "ProfileHueSatMapData1"
};

ExportExifImageRat = { "ForwardMatrix2", "ForwardMatrix1", "NoiseReductionApplied", "CurrentPreProfileMatrix", "AsShotPreProfileMatrix", "BestQualityScale", "ShadowScale",
	"AntiAliasStrength", "ChromaBlurRadius", "LinearResponseLimit", "BaselineSharpness", "BaselineNoise", "BaselineExposure", "AsShotWhiteXY", "AnalogBalance", "ReductionMatrix2",
	"ReductionMatrix1", "CameraCalibration2", "CameraCalibration1", "ColorMatrix2", "ColorMatrix1", "DefaultScale", "BlackLevelDeltaV", "BlackLevelDeltaH", "BlackLevel",
	"FocalLength", "BatteryLevel", "ReferenceBlackWhite", "PrimaryChromaticities", "WhitePoint", "YResolution", "XResolution"
};

ExportExifImageString = { "OpcodeList3", "OpcodeList2", "OpcodeList1", "OriginalRawFileDigest", "RawImageDigest", "PreviewDateTime", "CurrentICCProfile", "AsShotICCProfile",
	"OriginalRawFileData", "CameraSerialNumber", "UniqueCameraModel", "PrintImageMatching", "ImageHistory", "SecurityClassification", "Noise",
    "InterColorProfile", "Copyright", "ImageID", "JPEGTables", "TargetPrinter", "InkNames", "HostComputer", "Artist",
	"DateTime", "Software", "Model", "Make", "ImageDescription", "DocumentName"
};

(**********)
(****GPS***)
(**********)
ExportExifGPSInt = {"GPSVersionID", "GPSAltitudeRef", "GPSDifferential"
};

ExportExifGPSRat = { "GPSDOP", "GPSSpeed", "GPSTrack", "GPSImgDirection", "GPSDestLatitude"	, "GPSLatitude", "GPSLongitude", "GPSAltitude"
};

ExportExifGPSString = {"GPSDestLongitude", "GPSDestBearing", "GPSDestDistance", "GPSTimeStamp", "GPSLatitudeRef", "GPSLongitudeRef", "GPSSatellites", "GPSStatus", "GPSMeasureMode", "GPSSpeedRef", "GPSTrackRef", "GPSImgDirectionRef",
	"GPSMapDatum", "GPSDestLatitudeRef", "GPSDestLongitudeRef", "GPSDestBearingRef", "GPSDestDistanceRef", "GPSProcessingMethod", "GPSAreaInformation", "GPSDateStamp"
};

ExportExifGPSDualValues = {"GPSStatus", "GPSMeasureMode", "GPSLongitudeRef", "GPSLatitudeRef", "GPSAltitudeRef", "GPSTrackRef", "GPSImgDirectionRef"
};

(**********)
(*Envelope*)
(**********)
ExportEnvelopeNumber = {"ModelVersion", "FileVersion", "FileFormat", "ARMVersion", "ARMId"
};

ExportEnvelopeString = { "UNO", "TimeSent", "ServiceId", "ProductId", "EnvelopePriority", "EnvelopeNumber", "Destination", "DateSent", "CharacterSet"
};

(**************)
(*Application2*)
(**************)
ExportApplication2Number = { "PreviewFormat", "PreviewVersion", "RecordVersion"
};

ExportApplication2String = { "ActionAdvised", "AudioDuration", "AudioOutcue", "AudioRate", "AudioResolution", "AudioType", "Byline", "BylineTitle", "Caption",
	"Category", "City", "Contact", "Copyright", "CountryCode", "CountryName", "Credit", "DateCreated", "DigitizationDate", "DigitizationTime", "EditStatus",
	"EditorialUpdate", "ExpirationDate", "ExpirationTime", "FixtureId", "Headline", "ImageOrientation", "ImageType", "Keywords", "Language", "LocationCode",
	"LocationName", "ObjectAttribute", "ObjectCycle", "ObjectName", "ObjectType", "Preview", "Program", "ProgramVersion", "ProvinceState", "RasterizedCaption",
	"ReferenceDate", "ReferenceNumber", "ReferenceService", "ReleaseDate", "ReleaseTime", "Source", "SpecialInstructions", "SubLocation", "Subject", "SuppCategory",
	"TimeCreated", "TransmissionReference", "Urgency", "Writer"
};
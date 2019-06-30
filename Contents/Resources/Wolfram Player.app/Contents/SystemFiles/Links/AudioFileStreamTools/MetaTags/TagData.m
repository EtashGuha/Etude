(* APE *)

i$apeElementsAssociation = <|
	"Title" -> <| "Description" -> "", "Alias" -> {"Title"}, "InterpretationFunction" -> Identity |>,
	"Subtitle" -> <| "Description" -> "", "Alias" -> {"Subtitle"}, "InterpretationFunction" -> Identity |>,
	"Artist" -> <| "Description" -> "", "Alias" -> {"Artist"}, "InterpretationFunction" -> Identity |>,
	"Album" -> <| "Description" -> "", "Alias" -> {"Album"}, "InterpretationFunction" -> Identity |>,
	"Debut album" -> <| "Description" -> "", "Alias" -> {"DebutAlbum"}, "InterpretationFunction" -> Identity |>,
	"Publisher" -> <| "Description" -> "", "Alias" -> {"Publisher"}, "InterpretationFunction" -> Identity |>,
	"Conductor" -> <| "Description" -> "", "Alias" -> {"Conductor"}, "InterpretationFunction" -> Identity |>,
	"Track" -> <| "Description" -> "", "Alias" -> {"Track"}, 
		"InterpretationFunction" -> (interpretTrackNumber[#1]&), "InterpretationInverse" -> (uninterpretTrackNumber[#1]&), "ValidationFunction" -> (stringOrTrackNumberQ[#1]&) |>,
	"Composer" -> <| "Description" -> "", "Alias" -> {"Composer"}, "InterpretationFunction" -> Identity |>,
	"Comment" -> <| "Description" -> "", "Alias" -> {"Comment"}, "InterpretationFunction" -> Identity |>,
	"Copyright" -> <| "Description" -> "", "Alias" -> {"Copyright"}, "InterpretationFunction" -> Identity |>,
	"Publicationright" -> <| "Description" -> "", "Alias" -> {"PublicationRight"}, "InterpretationFunction" -> Identity |>,
	"File" -> <| "Description" -> "", "Alias" -> {"FileLinks"}, 
		"InterpretationFunction" -> (stringToLink[#1]&), "InterpretationInverse" -> (interpretedValueToString[#1, "Prefix":>If[MatchQ[#1,_File], "file:", None]]&), "ValidationFunction" -> (stringOrLinkQ[#1]&) |>,
	"EAN/UPC" -> <| "Description" -> "", "Alias" -> {"EAN/UPC"}, 
		"InterpretationFunction" -> (stringToNumeric[#1]&), "InterpretationInverse" -> (interpretedValueToString[#1]&) |>,
	"ISBN" -> <| "Description" -> "", "Alias" -> {"ISBN"}, "InterpretationFunction" -> Identity |>,
	"Catalog" -> <| "Description" -> "", "Alias" -> {"Catalog"}, "InterpretationFunction" -> Identity |>,
	"LC" -> <| "Description" -> "", "Alias" -> {"LabelCode"}, "InterpretationFunction" -> Identity |>,
	"Year" -> <| "Description" -> "", "Alias" -> {"RecordingYear"}, 
		"InterpretationFunction" -> (numberOrStringToDateObject[#1, "DateGranularity" -> "Year"]&), "InterpretationInverse" -> (interpretedValueToString[#1, "DateStringElements" -> "Year", "StringValidationFunction" -> (StringMatchQ[#1,DatePattern["Year"]]&)]&), "ValidationFunction" -> (dateSpecQ[#1]&) |>,
	"Record Date" -> <| "Description" -> "", "Alias" -> {"RecordingDate"}, 
		"InterpretationFunction" -> (numberOrStringToDateObject[#1]&), "InterpretationInverse" -> (interpretedValueToString[#1]&), "ValidationFunction" -> (dateSpecQ[#1]&) |>,
	"Record Location" -> <| "Description" -> "", "Alias" -> {"RecordingLocation"}, "InterpretationFunction" -> Identity |>,
	"Genre" -> <| "Description" -> "", "Alias" -> {"Genre"}, "InterpretationFunction" -> Identity |>,
	"Media" -> <| "Description" -> "", "Alias" -> {"MediaTime"}, 
		"InterpretationFunction" -> (Replace[stringToTimeStamp[#1], Except[_?QuantityQ] :> interpretTrackNumber[#1]]&), "InterpretationInverse" -> (If[QuantityQ[#1], timeStampToString[#1], Replace[uninterpretTrackNumber[#1], $Failed :> Replace[#1, Except[_?StringQ] -> $Failed]]]&), "ValidationFunction" -> ((stringOrTrackNumberQ[#1] || stringOrTimeStampQ[#1])&) |>,
	"Index" -> <| "Description" -> "", "Alias" -> {"IndexTime"}, 
		"InterpretationFunction" -> (stringToTimeStamp[#1]&), "InterpretationInverse" -> (timeStampToString[#1]&), "ValidationFunction" -> (stringOrTimeStampQ[#1]&) |>,
	"Related" -> <| "Description" -> "", "Alias" -> {"RelatedLinks"}, 
		"InterpretationFunction" -> (stringToLink[#1]&), "InterpretationInverse" -> (interpretedValueToString[#1, "Prefix":>If[MatchQ[#1,_File], "file:", None]]&), "ValidationFunction" -> (stringOrLinkQ[#1]&) |>,
	"ISRC" -> <| "Description" -> "", "Alias" -> {"ISRC"}, "InterpretationFunction" -> Identity |>,
	"Abstract" -> <| "Description" -> "", "Alias" -> {"Abstract"}, 
		"InterpretationFunction" -> (stringToLink[#1]&), "InterpretationInverse" -> (interpretedValueToString[#1, "Prefix":>If[MatchQ[#1,_File], "file:", None]]&), "ValidationFunction" -> (stringOrLinkQ[#1]&) |>,
	"Language" -> <| "Description" -> "", "Alias" -> {"Language"}, "InterpretationFunction" -> Identity |>,
	"Bibliography" -> <| "Description" -> "", "Alias" -> {"Bibliography"}, 
		"InterpretationFunction" -> (stringToLink[#1]&), "InterpretationInverse" -> (interpretedValueToString[#1, "Prefix":>If[MatchQ[#1,_File], "file:", None]]&), "ValidationFunction" -> (stringOrLinkQ[#1]&) |>,
	"Introplay" -> <| "Description" -> "", "Alias" -> {"Introplay"}, "InterpretationFunction" -> Identity |>,
	"Dummy" -> <| "Description" -> "", "Alias" -> {"Dummy"}, "InterpretationFunction" -> Identity |>
|>;
$apeElementsAssociation = AssociationThread[ToUpperCase /@ Keys[i$apeElementsAssociation], Values[i$apeElementsAssociation]];

$APETypesAssociation = <|"Text" -> 0, "Binary" -> 1, 0 -> "Text", 1 -> "Binary", 2 -> "Binary"|>;

(* Xiph *)

$xiphElementsAssociation = <|
	"TITLE" -> <| "Description" -> "Track/Work name", "Alias" -> {"Title"}, "InterpretationFunction" -> Identity |>,
	"VERSION" -> <| "Description" -> "", "Alias" -> {"Version"}, "InterpretationFunction" -> Identity |>,
	"ALBUM" -> <| "Description" -> "", "Alias" -> {"Album"}, "InterpretationFunction" -> Identity |>,
	"TRACKNUMBER" -> <| "Description" -> "", "Alias" -> {"TrackNumber"}, "InterpretationFunction" -> (interpretTrackNumber[#1]&), "InterpretationInverse" -> (uninterpretTrackNumber[#1]&), "ValidationFunction" -> (stringOrTrackNumberQ[#1]&) |>,
	"ARTIST" -> <| "Description" -> "", "Alias" -> {"Artist"}, "InterpretationFunction" -> Identity |>,
	"PERFORMER" -> <| "Description" -> "", "Alias" -> {"Performers"}, "InterpretationFunction" -> Identity |>,
	"COPYRIGHT" -> <| "Description" -> "", "Alias" -> {"Copyright"}, "InterpretationFunction" -> Identity |>,
	"LICENSE" -> <| "Description" -> "", "Alias" -> {"License"}, "InterpretationFunction" -> Identity |>,
	"ORGANIZATION" -> <| "Description" -> "", "Alias" -> {"Organization"}, "InterpretationFunction" -> Identity |>,
	"DESCRIPTION" -> <| "Description" -> "", "Alias" -> {"Description"}, "InterpretationFunction" -> Identity |>,
	"GENRE" -> <| "Description" -> "", "Alias" -> {"Genre"}, "InterpretationFunction" -> Identity |>,
	"DATE" -> <| "Description" -> "Date the track was recorded", "Alias" -> {"RecordingDate"}, "InterpretationFunction" -> (numberOrStringToDateObject[#1]&), "InterpretationInverse" -> (interpretedValueToString[#1, "DateStringElements" -> "ISODateTime"]&), "ValidationFunction" -> (dateSpecQ[#1]&) |>,
	"LOCATION" -> <| "Description" -> "Location where the track was recorded", "Alias" -> {"RecordingLocation"}, "InterpretationFunction" -> Identity |>,
	"CONTACT" -> <| "Description" -> "Creator or distributor contact information", "Alias" -> {"Contact"}, "InterpretationFunction" -> Identity |>,
	"ISRC" -> <| "Description" -> "", "Alias" -> {"ISRC"}, "InterpretationFunction" -> Identity |>
|>;

(* ID3v1 *)

$genreTypes = <| 
	"Blues" -> 0, "Classic Rock" -> 1, "Country" -> 2, "Dance" -> 3, "Disco" -> 4, "Funk" -> 5, "Grunge" -> 6, "Hip-Hop" -> 7, "Jazz" -> 8, "Metal" -> 9, 
	"New Age" -> 10, "Oldies" -> 11, "Other" -> 12, "Pop" -> 13, "Rhythm and Blues" -> 14, "Rap" -> 15, "Reggae" -> 16, "Rock" -> 17, "Techno" -> 18, "Industrial" -> 19, 
	"Alternative" -> 20, "Ska" -> 21, "Death Metal" -> 22, "Pranks" -> 23, "Soundtrack" -> 24, "Euro-Techno" -> 25, "Ambient" -> 26, "Trip-Hop" -> 27, "Vocal" -> 28, "Jazz & Funk" -> 29, 
	"Fusion" -> 30, "Trance" -> 31, "Classical" -> 32, "Instrumental" -> 33, "Acid" -> 34, "House" -> 35, "Game" -> 36, "Sound Clip" -> 37, "Gospel" -> 38, "Noise" -> 39, 
	"Alternative Rock" -> 40, "Bass" -> 41, "Soul" -> 42, "Punk" -> 43, "Space" -> 44, "Meditative" -> 45, "Instrumental Pop" -> 46, "Instrumental Rock" -> 47, "Ethnic" -> 48, "Gothic" -> 49, 
	"Darkwave" -> 50, "Techno-Industrial" -> 51, "Electronic" -> 52, "Pop-Folk" -> 53, "Eurodance" -> 54, "Dream" -> 55, "Southern Rock" -> 56, "Comedy" -> 57, "Cult" -> 58, "Gangsta" -> 59, 
	"Top 40" -> 60, "Christian Rap" -> 61, "Pop/Funk" -> 62, "Jungle" -> 63, "Native US" -> 64, "Cabaret" -> 65, "New Wave" -> 66, "Psychedelic" -> 67, "Rave" -> 68, "Showtunes" -> 69, 
	"Trailer" -> 70, "Lo-Fi "-> 71, "Tribal" -> 72, "Acid Punk" -> 73, "Acid Jazz" -> 74, "Polka" -> 75, "Retro" -> 76, "Musical" -> 77, "Rock 'n' Roll" -> 78, "Hard Rock" -> 79, 
	"Folk" -> 80, "Folk-Rock" -> 81, "National Folk" -> 82, "Swing" -> 83, "Fast Fusion" -> 84, "Bebop" -> 85, "Latin" -> 86, "Revival" -> 87, "Celtic" -> 88, "Bluegrass" -> 89, 
	"Avantgarde" -> 90, "Gothic Rock" -> 91, "Progressive Rock" -> 92, "Psychedelic Rock" -> 93, "Symphonic Rock" -> 94, "Slow Rock" -> 95, "Big Band" -> 96, "Chorus" -> 97, "Easy Listening" -> 98, "Acoustic" -> 99, 
	"Humour" -> 100, "Speech" -> 101, "Chanson" -> 102, "Opera" -> 103, "Chamber Music" -> 104, "Sonata" -> 105, "Symphony" -> 106, "Booty Bass" -> 107, "Primus" -> 108, "Porn Groove" -> 109, 
	"Satire" -> 110, "Slow Jam" -> 111, "Club" -> 112, "Tango" -> 113, "Samba" -> 114, "Folklore" -> 115, "Ballad" -> 116, "Power Ballad" -> 117, "Rhythmic Soul" -> 118, "Freestyle" -> 119, 
	"Duet" -> 120, "Punk Rock" -> 121, "Drum Solo" -> 122, "A cappella" -> 123, "Euro-House" -> 124, "Dance Hall" -> 125, "Goa" -> 126, "Drum & Bass" -> 127, "Club-House" -> 128, "Hardcore Techno" -> 129, 
	"Terror" -> 130, "Indie" -> 131, "BritPop" -> 132, "Negerpunk" -> 133, "Polsk Punk" -> 134, "Beat" -> 135, "Christian Gangsta Rap" -> 136, "Heavy Metal" -> 137, "Black Metal" -> 138, "Crossover" -> 139, 
	"Contemporary Christian" -> 140, "Christian Rock" -> 141, "Merengue" -> 142, "Salsa" -> 143, "Thrash Metal" -> 144, "Anime" -> 145, "Jpop" -> 146, "Synthpop" -> 147, "Abstract" -> 148, "Art Rock" -> 149, 
	"Baroque" -> 150, "Bhangra" -> 151, "Big Beat" -> 152, "Breakbeat" -> 153, "Chillout" -> 154, "Downtempo" -> 155, "Dub" -> 156, "EBM" -> 157, "Eclectic" -> 158, "Electro" -> 159, 
	"Electroclash" -> 160, "Emo" -> 161, "Experimental" -> 162, "Garage" -> 163, "Global" -> 164, "IDM" -> 165, "Illbient" -> 166, "Industro-Goth" -> 167, "Jam Band" -> 168, "Krautrock" -> 169, 
	"Leftfield" -> 170, "Lounge" -> 171, "Math Rock" -> 172, "New Romantic" -> 173, "Nu-Breakz" -> 174, "Post-Punk" -> 175, "Post-Rock" -> 176, "Psytrance" -> 177, "Shoegaze" -> 178, "Space Rock" -> 179, 
	"Trop Rock" -> 180, "World Music" -> 181, "Neoclassical" -> 182, "Audiobook" -> 183, "Audio Theatre" -> 184, "Neue Deutsche Welle" -> 185, "Podcast" -> 186, "Indie Rock" -> 187, "G-Funk" -> 188, "Dubstep" -> 189, 
	"Garage Rock" -> 190, "Psybient" -> 191, "Unknown" -> 255
|>;
$genreTypes = Join[$genreTypes, Association[($genreTypes[#] -> #)& /@ Keys[$genreTypes]]];

$id3v1ElementsAssociation = <|
	"TITLE" -> <| "Description" -> "", "Alias" -> {"SongTitle"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	"ARTIST" -> <| "Description" -> "", "Alias" -> {"Artist"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	"ALBUM" -> <| "Description" -> "", "Alias" -> {"Album"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	"YEAR" -> <| "Description" -> "", "Alias" -> {"Year"}, "InterpretationFunction" -> (numberOrStringToDateObject[#1, "DateGranularity" -> "Year"]&), "InterpretationInverse" -> (dateToIntegerYear[#1]&), "ValidationFunction" -> (dateSpecQ[#1]&) |>,
	"COMMENT" -> <| "Description" -> "", "Alias" -> {"Comment"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	"GENRE" -> <| "Description" -> "", "Alias" -> {"Genre"}, "InterpretationFunction" -> (Replace[#1, _?IntegerQ -> transformGenre[#1]]&), "InterpretationInverse" -> (Replace[#1, _?StringQ -> transformGenre[#1]]&), "ValidationFunction" -> ((StringQ[#1] || Internal`NonNegativeIntegerQ[#1])&) |>,
	"TRACK" -> <| "Description" -> "", "Alias" -> {"Track"}, "InterpretationFunction" -> (interpretTrackNumber[#1]&), "InterpretationInverse" -> (uninterpretTrackNumber[#1, "ToString" -> False]&), "ValidationFunction" -> (stringOrTrackNumberQ[#1]&) |>
|>;

$ID3v1Keys = <|"TITLE" -> 0, "ARTIST" -> 1, "ALBUM" -> 2, "YEAR" -> 3, "COMMENT" -> 4, "GENRE" -> 5, "TRACK" -> 6|>;

$ID3v1Types = <|"TITLE" -> "String", "ARTIST" -> "String", "ALBUM" -> "String", "YEAR" -> "Number", "COMMENT" -> "String", "GENRE" -> "Number", "TRACK" -> "Number"|>;

(* ID3v2 *)

$eventTimingCodes = <|
	0 -> "Padding", 1 -> "EndOfInitialSilence", 2 -> "IntroStart",
	3 -> "MainPartStart", 4 -> "OutroStart", 5 -> "OutroEnd",
	6 -> "VerseStart", 7 -> "RefrainStart", 8 -> "InterludeStart",
	9 -> "ThemeStart", 10 -> "VariationStart", 11 -> "KeyChange",
	12 -> "TimeChange", 13 -> "MomentaryUnwantedNoise",
	14 -> "SustainedNoise", 15 -> "SustainedNoiseEnd", 16 -> "IntroEnd",
	17 -> "MainPartEnd", 18 -> "VerseEnd", 19 -> "RefrainEnd",
	20 -> "ThemeEnd", 21 -> "Profanity", 22 -> "ProfanityEnd",
	224 -> "NotPredefinedSynch0", 225 -> "NotPredefinedSynch1",
	226 -> "NotPredefinedSynch2", 227 -> "NotPredefinedSynch3",
	228 -> "NotPredefinedSynch4", 229 -> "NotPredefinedSynch5",
	230 -> "NotPredefinedSynch6", 231 -> "NotPredefinedSynch7",
	232 -> "NotPredefinedSynch8", 233 -> "NotPredefinedSynch9",
	234 -> "NotPredefinedSynchA", 235 -> "NotPredefinedSynchB",
	236 -> "NotPredefinedSynchC", 237 -> "NotPredefinedSynchD",
	238 -> "NotPredefinedSynchE", 239 -> "NotPredefinedSynchF",
	253 -> "AudioEnd", 254 -> "AudioFileEnds",
	"Padding" -> 0, "EndOfInitialSilence" -> 1, "IntroStart" -> 2,
	"MainPartStart" -> 3, "OutroStart" -> 4, "OutroEnd" -> 5,
	"VerseStart" -> 6, "RefrainStart" -> 7, "InterludeStart" -> 8,
	"ThemeStart" -> 9, "VariationStart" -> 10, "KeyChange" -> 11,
	"TimeChange" -> 12, "MomentaryUnwantedNoise" -> 13, "SustainedNoise"
	-> 14, "SustainedNoiseEnd" -> 15, "IntroEnd" -> 16, "MainPartEnd" ->
	17, "VerseEnd" -> 18, "RefrainEnd" -> 19, "ThemeEnd" -> 20,
	"Profanity" -> 21, "ProfanityEnd" -> 22, "NotPredefinedSynch0" ->
	224, "NotPredefinedSynch1" -> 225, "NotPredefinedSynch2" -> 226,
	"NotPredefinedSynch3" -> 227, "NotPredefinedSynch4" -> 228,
	"NotPredefinedSynch5" -> 229, "NotPredefinedSynch6" -> 230,
	"NotPredefinedSynch7" -> 231, "NotPredefinedSynch8" -> 232,
	"NotPredefinedSynch9" -> 233, "NotPredefinedSynchA" -> 234,
	"NotPredefinedSynchB" -> 235, "NotPredefinedSynchC" -> 236,
	"NotPredefinedSynchD" -> 237, "NotPredefinedSynchE" -> 238,
	"NotPredefinedSynchF" -> 239, "AudioEnd" -> 253, "AudioFileEnds" -> 254
|>;

$pictureTypes = <|
	0 -> "Other", "Other" -> 0, 1 -> "FileIcon", "FileIcon" -> 1,
	2 -> "OtherFileIcon", "OtherFileIcon" -> 2, 3 -> "FrontCover",
	"FrontCover" -> 3, 4 -> "BackCover", "BackCover" -> 4,
	5 -> "LeafletPage", "LeafletPage" -> 5, 6 -> "Media", "Media" -> 6,
	7 -> "LeadArtist", "LeadArtist" -> 7, 8 -> "Artist", "Artist" -> 8,
	9 -> "Conductor", "Conductor" -> 9, 10 -> "Band", "Band" -> 10,
	11 -> "Composer", "Composer" -> 11, 12 -> "Lyricist",
	"Lyricist" -> 12, 13 -> "RecordingLocation",
	"RecordingLocation" -> 13, 14 -> "DuringRecording",
	"DuringRecording" -> 14, 15 -> "DuringPerformance",
	"DuringPerformance" -> 15, 16 -> "MovieScreenCapture",
	"MovieScreenCapture" -> 16, 17 -> "ColouredFish",
	"ColouredFish" -> 17, 18 -> "Illustration", "Illustration" -> 18,
	19 -> "BandLogo", "BandLogo" -> 19, 20 -> "PublisherLogo",
	"PublisherLogo" -> 20
|>;

$lyricsTypes = <|
	0 -> "Other", "Other" -> 0, 1 -> "Lyrics", "Lyrics" -> 1,
	2 -> "TextTranscription", "TextTranscription" -> 2, 3 -> "Movement",
	"Movement" -> 3, 4 -> "Events", "Events" -> 4, 5 -> "Chord",
	"Chord" -> 5, 6 -> "Trivia", "Trivia" -> 6, 7 -> "WebpageUrls",
	"WebpageUrls" -> 7, 8 -> "ImageUrls", "ImageUrls" -> 8
|>;

$eventTimestampFormats = <|
	0 -> "Unknown", 1 -> "AbsoluteMpegFrames", 2 -> "AbsoluteMilliseconds",
	"Unknown" -> 0, "AbsoluteMpegFrames" -> 1, "AbsoluteMilliseconds" -> 2
|>;

$channelTypes = <|
	0 -> "Other", "Other" -> 0, 1 -> "MasterVolume",
	"MasterVolume" -> 1, 2 -> "FrontRight", "FrontRight" -> 2,
	3 -> "FrontLeft", "FrontLeft" -> 3, 4 -> "BackRight",
	"BackRight" -> 4, 5 -> "BackLeft", "BackLeft" -> 5,
	6 -> "FrontCentre", "FrontCentre" -> 6, 7 -> "BackCentre",
	"BackCentre" -> 7, 8 -> "Subwoofer", "Subwoofer" -> 8
|>;

$id3v2FramesAssociation = <|
	"T***" -> <|"Elements" -> {"Values"}, "Alias" -> {None}, "InterpretationFunction" -> (transformID3v2Elements[#1,#2]&), "InterpretationInverse" -> (transformID3v2Elements[#1,#2, "ToRawForm" -> True]&),
		"Description" -> "Text Information", "Singleton" -> True|>,

	"TXXX" -> <|"Elements" -> {"Values", "Description"}, "Alias" -> {"UserText"}, "InterpretationFunction" -> (transformID3v2Elements[#1,#2]&), "InterpretationInverse" -> (transformID3v2Elements[#1,#2, "ToRawForm" -> True]&),
		"Description" -> "User Defined Text Information", "Singleton" -> False|>,

	"APIC" -> <|
		"Elements" -> {"Picture", "MimeType", "PictureType","Description"}, 
		"Alias" -> {"AttachedPicture"}, "InterpretationFunction" -> (transformID3v2Elements[#1,#2]&), "InterpretationInverse" -> (transformID3v2Elements[#1,#2, "ToRawForm" -> True]&),
		"Description" -> "Attached Picture", "Singleton" -> False|>,

	"UFID" -> <|"Elements" -> {"Owner", "Identifier"}, "Alias" -> {"FileID"}, "InterpretationFunction" -> (transformID3v2Elements[#1,#2]&), "InterpretationInverse" -> (transformID3v2Elements[#1,#2, "ToRawForm" -> True]&),
		"Description" -> "Unique File Identifier", "Singleton" -> False|>,

	"OWNE" -> <|"Elements" -> {"PricePaid", "PurchaseDate", "Seller"},
		"Alias" -> {"Ownership"},  "InterpretationFunction" -> (transformID3v2Elements[#1,#2]&), "InterpretationInverse" -> (transformID3v2Elements[#1,#2, "ToRawForm" -> True]&), "Description" -> "Ownership Information", "Singleton" -> True|>,

	"PRIV" -> <|"Elements" -> {"Owner", "Data"}, "Alias" -> {"PrivateInformation"}, "InterpretationFunction" -> (transformID3v2Elements[#1,#2]&), "InterpretationInverse" -> (transformID3v2Elements[#1,#2, "ToRawForm" -> True]&),
		"Description" -> "Private Information", "Singleton" -> False|>,

	"POPM" -> <|"Elements" -> {"Email", "Rating", "Counter"},
		"Alias" -> {"PopularityMeter"}, "InterpretationFunction" -> (transformID3v2Elements[#1,#2]&), "InterpretationInverse" -> (transformID3v2Elements[#1,#2, "ToRawForm" -> True]&), "Description" -> "Popularimeter", "Singleton" -> False|>,

	"GEOB" -> <|
		"Elements" -> {"Object", "MimeType", "FileName", "Description"},
		"Alias" -> {"GeneralEncapsulatedObject"}, "InterpretationFunction" -> (transformID3v2Elements[#1,#2]&), "InterpretationInverse" -> (transformID3v2Elements[#1,#2, "ToRawForm" -> True]&),
		"Description" -> "General Encapsulated Object", "Singleton" -> False|>,

	"COMM" -> <|"Elements" -> {"Language", "Description", "Text"},
		"Alias" -> {"Comments"}, "InterpretationFunction" -> (transformID3v2Elements[#1,#2]&), "InterpretationInverse" -> (transformID3v2Elements[#1,#2, "ToRawForm" -> True]&), "Description" -> "Comments", "Singleton" -> False|>,

	"CHAP" -> <|
		"Elements" -> {"StartTime", "EndTime", "StartOffset", "EndOffset", "EmbeddedFrames", "Identifier"}, 
		"Alias" -> {"Chapter"}, "InterpretationFunction" -> (transformID3v2Elements[#1,#2]&), "InterpretationInverse" -> (transformID3v2Elements[#1,#2, "ToRawForm" -> True]&),
		"Description" -> "Chapter", "Singleton" -> False|>,

	"CTOC" -> <|
		"Elements" -> {"Ordered", "TopLevel", "EmbeddedFrames", "Identifier", "ChildElements"}, 
		"Alias" -> {"TableOfContents"}, "InterpretationFunction" -> (transformID3v2Elements[#1,#2]&), "InterpretationInverse" -> (transformID3v2Elements[#1,#2, "ToRawForm" -> True]&),
		"Description" -> "Table of Contents", "Singleton" -> False|>,

	"ETCO" -> <|"Elements" -> {"TimestampFormat", "SynchedEvents"}, "Alias" -> {"EventTimingCodes"}, "InterpretationFunction" -> (transformID3v2Elements[#1,#2]&), "InterpretationInverse" -> (transformID3v2Elements[#1,#2, "ToRawForm" -> True]&),
		"Description" -> "Event Timing Codes", "Singleton" -> True|>,

	"RVA2" -> <|"Elements" -> {"Description", "Channels"}, "Alias" -> {"RelativeVolumeAdjustment"}, "InterpretationFunction" -> (transformID3v2Elements[#1,#2]&), "InterpretationInverse" -> (transformID3v2Elements[#1,#2, "ToRawForm" -> True]&),
		"Description" -> "Relative Volume Adjustment", "Singleton" -> False|>,

	"SYLT" -> <|
		"Elements" -> {"Language", "Description", "TimestampFormat", "LyricsType", "SynchedText"}, 
		"Alias" -> {"SynchronisedLyrics"}, "InterpretationFunction" -> (transformID3v2Elements[#1,#2]&), "InterpretationInverse" -> (transformID3v2Elements[#1,#2, "ToRawForm" -> True]&),
		"Description" -> "Synchronised Lyrics or Text", "Singleton" -> False|>,

	"USLT" -> <|"Elements" -> {"Language", "Description", "Text"},
		"Alias" -> {"UnsynchronisedLyrics"}, "InterpretationFunction" -> (transformID3v2Elements[#1,#2]&), "InterpretationInverse" -> (transformID3v2Elements[#1,#2, "ToRawForm" -> True]&),
		"Description" -> "Unsynchronised Lyrics or Text", "Singleton" -> False|>,

	"W***" -> <|"Elements" -> {"URL"}, "Alias" -> {None}, "InterpretationFunction" -> (transformID3v2Elements[#1,#2]&), "InterpretationInverse" -> (transformID3v2Elements[#1,#2, "ToRawForm" -> True]&),
		"Description" -> "URL Link", "Singleton" -> True|>,

	"WXXX" -> <|"Elements" -> {"URL", "Description"}, "Alias" -> {"UserURL"}, "InterpretationFunction" -> (transformID3v2Elements[#1,#2]&), "InterpretationInverse" -> (transformID3v2Elements[#1,#2, "ToRawForm" -> True]&),
		"Description" -> "User Defined URL Link", "Singleton" -> False|>
|>;

$id3v2ExtendedFramesAssociation = <|
	"TALB" -> <| "Description"-> "Album/Movie/Show title", "Alias" -> {"Album"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TBPM" -> <| "Description"-> "BPM (beats per minute)", "Alias" -> {"BeatsPerMinute"}, 
		"InterpretationFunction" -> (numberOrStringToQuantity[#1, IndependentUnit["beats"]/"Minutes"]&), "InterpretationInverse" -> (interpretedValueToString[#1]&), "ValidationFunction" -> (stringOrQuantityQ[#1, "UnitStringPattern" -> $unitBeatsPerMinutePatt, "UnitPattern" -> (IndependentUnit["beats"]/"Minutes"), "AllowNumericValuesTest" -> (Internal`NonNegativeIntegerQ[#]&)]&) |>,
	
	"TCOM" -> <| "Description"-> "Composer", "Alias" -> {"Composer"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TCON" -> <| "Description"-> "Content type", "Alias" -> {"ContentType"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TCOP" -> <| "Description"-> "Copyright message", "Alias" -> {"Copyright"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TDAT" -> <| "Description"-> "Date", "Alias" -> {"RecordingDate"}, (* DDMM *)
		"InterpretationFunction" -> (numberOrStringToDateObject[#1, "DatePattern" -> DatePattern[{"Day","Month"},""], "DateElements" -> {"Day","","Month"}]&), "InterpretationInverse" -> (interpretedValueToString[#1, "DateStringElements" -> {"Day", "Month"}, "StringValidationFunction" -> (StringMatchQ[#1,DatePattern[{"Day","Month"},""]]&)]&), "ValidationFunction" -> (dateSpecQ[#1, "NumberTest" -> (MatchQ[#1,_Real]&)]&) |>,
	
	"TDEN" -> <| "Description"-> "Encoding time", "Alias" -> {"EncodingTime"},
		"InterpretationFunction" -> (numberOrStringToDateObject[#1]&), "InterpretationInverse" -> (interpretedValueToString[#1, "DateStringElements" -> "ISODateTime"]&), "ValidationFunction" -> (dateSpecQ[#1]&) |>,
	
	"TDLY" -> <| "Description"-> "Playlist delay", "Alias" -> {"PlaylistDelay"}, 
		"InterpretationFunction" -> (numberOrStringToQuantity[#1, "Milliseconds", "ConversionUnits" -> "AutomaticDuration"]&), "InterpretationInverse" -> (interpretedValueToString[#1, "ConversionUnits" -> "Milliseconds"]&), "ValidationFunction" -> (stringOrQuantityQ[#1, "UnitStringPattern" -> $unitDurationPatt, "AllowNumericValuesTest" -> (Positive[#1]&)]&) |>,
	
	"TDOR" -> <| "Description"-> "Original release time", "Alias" -> {"OriginalReleaseTime"}, 
		"InterpretationFunction" -> (numberOrStringToDateObject[#1]&), "InterpretationInverse" -> (interpretedValueToString[#1, "DateStringElements" -> "ISODateTime"]&), "ValidationFunction" -> (dateSpecQ[#1]&) |>,
	
	"TDRC" -> <| "Description"-> "Recording time", "Alias" -> {"RecordingDateTime"}, 
		"InterpretationFunction" -> (numberOrStringToDateObject[#1]&), "InterpretationInverse" -> (interpretedValueToString[#1, "DateStringElements" -> "ISODateTime"]&), "ValidationFunction" -> (dateSpecQ[#1]&) |>,
	
	"TDRL" -> <| "Description"-> "Release time", "Alias" -> {"ReleaseTime"}, 
		"InterpretationFunction" -> (numberOrStringToDateObject[#1]&), "InterpretationInverse" -> (interpretedValueToString[#1, "DateStringElements" -> "ISODateTime"]&), "ValidationFunction" -> (dateSpecQ[#1]&) |>,
	
	"TDTG" -> <| "Description"-> "Tagging time", "Alias" -> {"TaggingTime"}, 
		"InterpretationFunction" -> (numberOrStringToDateObject[#1]&), "InterpretationInverse" -> (interpretedValueToString[#1, "DateStringElements" -> "ISODateTime"]&), "ValidationFunction" -> (dateSpecQ[#1]&) |>,
	
	"TENC" -> <| "Description"-> "Encoded by", "Alias" -> {"EncodedBy"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TEXT" -> <| "Description"-> "Lyricist/Text writer", "Alias" -> {"Lyricist"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TFLT" -> <| "Description"-> "File type", "Alias" -> {"FileType"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TIME" -> <| "Description"-> "Recording Time", "Alias" -> {"RecordingTime"}, (* HHMM *)
		"InterpretationFunction" -> (stringToTimeObject[#1, "DatePattern" -> DatePattern[{"Hour","Minute"},""], "DateElements" -> {"Hour","","Minute"}]&), "InterpretationInverse" -> (interpretedValueToString[#1, "DateStringElements" -> {"Hour", "Minute"}, "StringValidationFunction" -> (StringMatchQ[#1,DatePattern[{"Hour","Minute"},""]]&)]&), "ValidationFunction" -> (dateSpecQ[#1, "AllowTimeObject" -> True]&) |>,
	
	"TIT1" -> <| "Description"-> "Content group description", "Alias" -> {"ContentGroup"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TIT2" -> <| "Description"-> "Title/songname/content description", "Alias" -> {"Title"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TIT3" -> <| "Description"-> "Subtitle/Description refinement", "Alias" -> {"Subtitle"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TKEY" -> <| "Description"-> "Initial key", "Alias" -> {"InitialKey"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TLAN" -> <| "Description"-> "Language(s)", "Alias" -> {"Language"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TLEN" -> <| "Description"-> "Length", "Alias" -> {"Duration"}, 
		"InterpretationFunction" -> (numberOrStringToQuantity[#1, "Milliseconds", "ConversionUnits" -> "AutomaticDuration"]&), "InterpretationInverse" -> (interpretedValueToString[#1, "ConversionUnits" -> "Milliseconds"]&), "ValidationFunction" -> (stringOrQuantityQ[#1, "UnitStringPattern" -> $unitDurationPatt, "AllowNumericValuesTest" -> (Positive[#1]&)]&) |>,
	
	"TMCL" -> <| "Description"-> "Musician credits list", "Alias" -> {"MusicianCredits"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TMED" -> <| "Description"-> "Media type", "Alias" -> {"MediaType"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TMOO" -> <| "Description"-> "Mood", "Alias" -> {"Mood"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TOAL" -> <| "Description"-> "Original album/movie/show title", "Alias" -> {"OriginalAlbum"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TOFN" -> <| "Description"-> "Original filename", "Alias" -> {"OriginalFileName"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TOLY" -> <| "Description"-> "Original lyricist(s)/text writer(s)", "Alias" -> {"OriginalLyricist"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TOPE" -> <| "Description"-> "Original artist(s)/performer(s)", "Alias" -> {"OriginalPerformer"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TORY" -> <| "Description"-> "Original release year", "Alias" -> {"OriginalReleaseYear"}, 
		"InterpretationFunction" -> (numberOrStringToDateObject[#1, "DateGranularity" -> "Year"]&), "InterpretationInverse" -> (interpretedValueToString[#1, "DateStringElements" -> "Year", "StringValidationFunction" -> (StringMatchQ[#1,DatePattern["Year"]]&)]&), "ValidationFunction" -> (dateSpecQ[#1]&) |>,
	
	"TOWN" -> <| "Description"-> "File owner/licensee", "Alias" -> {"FileOwner"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TPE1" -> <| "Description"-> "Lead performer(s)/Soloist(s)", "Alias" -> {"LeadPerformer"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TPE2" -> <| "Description"-> "Band/orchestra/accompaniment", "Alias" -> {"AdditionalPerformers"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TPE3" -> <| "Description"-> "Conductor/performer refinement", "Alias" -> {"Conductor"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TPE4" -> <| "Description"-> "Interpreted, remixed, or otherwise modified by", "Alias" -> {"ModifiedBy"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TPOS" -> <| "Description"-> "Part of a set", "Alias" -> {"PartOfSet"}, 
		"InterpretationFunction" -> (interpretTrackNumber[#1]&), "InterpretationInverse" -> (uninterpretTrackNumber[#1]&), "ValidationFunction" -> (stringOrTrackNumberQ[#1]&) |>,
	
	"TPRO" -> <| "Description"-> "Produced notice", "Alias" -> {"ProductionRight"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TPUB" -> <| "Description"-> "Publisher", "Alias" -> {"Publisher"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TRCK" -> <| "Description"-> "Track number/Position in set", "Alias" -> {"TrackNumber"}, 
		"InterpretationFunction" -> (interpretTrackNumber[#1]&), "InterpretationInverse" -> (uninterpretTrackNumber[#1]&), "ValidationFunction" -> (stringOrTrackNumberQ[#1]&) |>,
	
	"TRDA" -> <| "Description"-> "Recording dates", "Alias" -> {"RecordingDates"},
		"InterpretationFunction" -> ((numberOrStringToDateObject /@ If[StringQ[#1], StringSplit[#1, ","], #1])&), "InterpretationInverse" -> (interpretedValueToString[#1, "Separator" -> ", "(*, "DateStringElements" -> {"Day", " ", "MonthName"}*)]&), "ValidationFunction" -> ((StringQ[#1] || VectorQ[#1, dateSpecQ])&) |>,
	
	"TRSN" -> <| "Description"-> "Internet radio station name", "Alias" -> {"RadioStationName"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TRSO" -> <| "Description"-> "Internet radio station owner", "Alias" -> {"RadioStationOwner"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TSIZ" -> <| "Description"-> "Size", "Alias" -> {"FileSize"}, 
		"InterpretationFunction" -> (numberOrStringToQuantity[#1, "Bytes", "ConversionUnits" -> "AutomaticFileSize"]&), "InterpretationInverse" -> (interpretedValueToString[#1, "ConversionUnits" -> "Bytes"]&), "ValidationFunction" -> (stringOrQuantityQ[#1, "UnitStringPattern" -> ("*bytes"|"b"|RegularExpression["(k|m|g)(b|ib)"]), "AllowNumericValuesTest" -> (Internal`NonNegativeIntegerQ[#1]&)]&) |>,
	
	"TSOA" -> <| "Description"-> "Album sort order", "Alias" -> {"AlbumSortOrder"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TSOP" -> <| "Description"-> "Performer sort order", "Alias" -> {"PerformerSortOrder"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TSOT" -> <| "Description"-> "Title sort order", "Alias" -> {"TitleSortOrder"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TSRC" -> <| "Description"-> "ISRC (international standard recording code)", "Alias" -> {"ISRC"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TSSE" -> <| "Description"-> "Software/Hardware and settings used for encoding", "Alias" -> {"EncodingSettings"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TSST" -> <| "Description"-> "Set subtitle", "Alias" -> {"SetSubtitle"}, 
		"InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&) |>,
	
	"TYER" -> <| "Description"-> "Year", "Alias" -> {"RecordingYear"},
		"InterpretationFunction" -> (numberOrStringToDateObject[#1, "DateGranularity" -> "Year"]&), "InterpretationInverse" -> (interpretedValueToString[#1, "DateStringElements" -> "Year", "StringValidationFunction" -> (StringMatchQ[#, DatePattern["Year"]]&)]&), "ValidationFunction" -> (dateSpecQ[#1]&) |>,
	
	"WCOM" -> <| "Description"-> "Commercial information", "Alias" -> {"CommercialURL"}, 
		"InterpretationFunction" -> (stringToLink[#1, "Wrapper"->URL]&), "InterpretationInverse" -> (interpretedValueToString[#1]&), "ValidationFunction" -> ((stringOrLinkQ[#1, "Wrapper"->URL])&) |>,
	
	"WCOP" -> <| "Description"-> "Copyright/Legal information", "Alias" -> {"CopyrightURL"}, 
		"InterpretationFunction" -> (stringToLink[#1, "Wrapper"->URL]&), "InterpretationInverse" -> (interpretedValueToString[#1]&), "ValidationFunction" -> ((stringOrLinkQ[#1, "Wrapper"->URL])&) |>,
	
	"WOAF" -> <| "Description"-> "Official audio file webpage", "Alias" -> {"AudioFileURL"}, 
		"InterpretationFunction" -> (stringToLink[#1, "Wrapper"->URL]&), "InterpretationInverse" -> (interpretedValueToString[#1]&), "ValidationFunction" -> ((stringOrLinkQ[#1, "Wrapper"->URL])&) |>,
	
	"WOAR" -> <| "Description"-> "Official artist/performer webpage", "Alias" -> {"ArtistURL"}, 
		"InterpretationFunction" -> (stringToLink[#1, "Wrapper"->URL]&), "InterpretationInverse" -> (interpretedValueToString[#1]&), "ValidationFunction" -> ((stringOrLinkQ[#1, "Wrapper"->URL])&) |>,
	
	"WOAS" -> <| "Description"-> "Official audio source webpage", "Alias" -> {"AudioSourceURL"}, 
		"InterpretationFunction" -> (stringToLink[#1, "Wrapper"->URL]&), "InterpretationInverse" -> (interpretedValueToString[#1]&), "ValidationFunction" -> ((stringOrLinkQ[#1, "Wrapper"->URL])&) |>,
	
	"WORS" -> <| "Description"-> "Official Internet radio station homepage", "Alias" -> {"RadioStationURL"}, 
		"InterpretationFunction" -> (stringToLink[#1, "Wrapper"->URL]&), "InterpretationInverse" -> (interpretedValueToString[#1]&), "ValidationFunction" -> ((stringOrLinkQ[#1, "Wrapper"->URL])&) |>,
	
	"WPAY" -> <| "Description"-> "Payment", "Alias" -> {"PaymentURL"}, 
		"InterpretationFunction" -> (stringToLink[#1, "Wrapper"->URL]&), "InterpretationInverse" -> (interpretedValueToString[#1]&), "ValidationFunction" -> ((stringOrLinkQ[#1, "Wrapper"->URL])&) |>,
	
	"WPUB" -> <| "Description"-> "Publishers official webpage", "Alias" -> {"PublisherURL"}, 
		"InterpretationFunction" -> (stringToLink[#1, "Wrapper"->URL]&), "InterpretationInverse" -> (interpretedValueToString[#1]&), "ValidationFunction" -> ((stringOrLinkQ[#1, "Wrapper"->URL])&) |>
|>;

$id3v2ElementsAssociation = <|
	"Channels" -> <| "ValidationFunction" -> ((AssociationQ[#] 
						&& SubsetQ[Keys[$channelTypes], Keys[#]] 
						&& VectorQ[Values[#], (AssociationQ[#] && Length[#] > 0
									&& SubsetQ[{"PeakVolume", "BitsRepresentingPeak", "VolumeAdjustment"}, Keys[#]] 
									&& (MatchQ[#["PeakVolume"], _?MissingQ|_?ByteArrayQ])
									&& (MatchQ[#["BitsRepresentingPeak"], _?MissingQ|_?Internal`NonNegativeIntegerQ])
									&& (MatchQ[#["VolumeAdjustment"], _?MissingQ|_?IntegerQ|Quantity[_, "Decibels"]])
								)&])&), 
						"InterpretationFunction" -> ((Association[Normal[#] /. {Rule["VolumeAdjustment", n_?IntegerQ] :> Rule["VolumeAdjustment", Quantity[n, "Decibels"]]}]& /@ #1)&),
						"InterpretationInverse" -> ((Association[Normal[#] /. {Rule["VolumeAdjustment", n_?QuantityQ] :> Rule["VolumeAdjustment", IntegerPart[QuantityMagnitude[n]]]}]& /@ #1)&) 
						|>,
	"SynchedEvents" -> <| "ValidationFunction" -> ((AssociationQ[#]
							&& VectorQ[Keys[#], numberOrQuantityQ[#, "UnitStringPattern"->$unitDurationPatt, "NumberTest"->(Internal`NonNegativeIntegerQ[#]&)]&]
							&& VectorQ[Values[#], MemberQ[Keys[$eventTimingCodes], #]&])&), 
						"InterpretationFunction" -> (AssociationThread[numberListToTimestampFormatQuantity[Keys[#1], #2], If[IntegerQ[#],$eventTimingCodes[#],#]& /@ Values[#1]]&), 
						"InterpretationInverse" -> (AssociationThread[quantityToNumeric[#, "ConversionUnits"->"Milliseconds", "ToInteger"->True]& /@ Keys[#1], If[IntegerQ[#],#,$eventTimingCodes[#]]& /@ Values[#1]]&) 
						|>,
	"SynchedText" -> <| "ValidationFunction" -> ((AssociationQ[#]
							&& VectorQ[Keys[#], numberOrQuantityQ[#, "UnitStringPattern"->$unitDurationPatt, "NumberTest"->(Internal`NonNegativeIntegerQ[#]&)]&]
							&& VectorQ[Values[#], StringQ])&), 
						"InterpretationFunction" -> (AssociationThread[numberListToTimestampFormatQuantity[Keys[#1], #2], Values[#1]]&), 
						"InterpretationInverse" -> (AssociationThread[quantityToNumeric[#, "ConversionUnits"->"Milliseconds", "ToInteger"->True]& /@ Keys[#1], Values[#1]]&) 
						|>,
	"ChildElements" -> <| "ValidationFunction" -> ((VectorQ[#, (StringQ[#] || ByteArrayQ[#])&])&), "InterpretationFunction" -> Identity |>,
	"PictureType" -> <| "ValidationFunction" -> (MemberQ[Keys[$pictureTypes], #]&), "InterpretationFunction" -> (If[IntegerQ[#1],$pictureTypes[#1],#1]&), "InterpretationInverse" -> (If[IntegerQ[#1],#1,$pictureTypes[#1]]&) |>,
	"LyricsType" -> <| "ValidationFunction" -> (MemberQ[Keys[$lyricsTypes], #]&), "InterpretationFunction" -> (If[IntegerQ[#1],$lyricsTypes[#1],#1]&), "InterpretationInverse" -> (If[IntegerQ[#1],#1,$lyricsTypes[#1]]&) |>,
	"TimestampFormat" -> <| "ValidationFunction" -> (MemberQ[Keys[$eventTimestampFormats], #]&), "InterpretationFunction" -> (If[IntegerQ[#1],$eventTimestampFormats[#1],#1]&), "InterpretationInverse" -> (If[IntegerQ[#1],#1,$eventTimestampFormats[#1]]&) |>,
	"Description" -> <| "ValidationFunction" -> ((StringQ[#])&), "InterpretationFunction" -> Identity |>,
	"Values" -> <| "ValidationFunction" -> ((VectorQ[#, StringQ])&), "InterpretationFunction" -> Identity |>,
	"Language" -> <| "ValidationFunction" -> ((StringQ[#] (* && StringLength[#] == 3 *))&), "InterpretationFunction" -> Identity |>,
	"FileName" -> <| "ValidationFunction" -> ((StringQ[#] || MatchQ[#, _File])&), "InterpretationFunction" -> (If[StringQ[#1], File[#1], #1]&), "InterpretationInverse" -> (interpretedValueToString[#1]&) |>,
	"MimeType" -> <| "ValidationFunction" -> ((StringQ[#])&), "InterpretationFunction" -> Identity |>,
	"Picture" -> <| "ValidationFunction" -> ((ByteArrayQ[#] || ImageQ[#])&), "InterpretationFunction" -> (byteArrayToImage[#1, #2]&), "InterpretationInverse" -> (imageToByteArray[#1, #2]&) |>,
	"Seller" -> <| "ValidationFunction" -> ((StringQ[#])&), "InterpretationFunction" -> Identity |>,
	"PurchaseDate" -> <| "ValidationFunction" -> ((dateSpecQ[#])&), "InterpretationFunction" -> (numberOrStringToDateObject[#1]&), "InterpretationInverse" -> (interpretedValueToString[#1, "DateStringElements"->{"Year", "Month", "Day"}]&) |>,
	"PricePaid" -> <| "ValidationFunction" -> ((QuantityQ[#] || (StringQ[#] && StringMatchQ[#, RegularExpression["^([A-Z]{3,3}|[0-9]{3,3})[0-9]*([.][0-9]*){0,1}$"]]))&), "InterpretationFunction" -> (stringToPricePaidQuantity[#1, #2]&), "InterpretationInverse" -> (pricePaidQuantityToString[#1]&) |>,
	"Email" -> <| "ValidationFunction" -> ((StringQ[#])&), "InterpretationFunction" -> Identity |>,
	"Counter" -> <| "ValidationFunction" -> ((IntegerQ[#] || ByteArrayQ[#])&), "InterpretationFunction" -> Identity |>,
	"Rating" -> <| "ValidationFunction" -> ((Internal`NonNegativeIntegerQ[#] && (# <= 255))&), "InterpretationFunction" -> Identity |>,
	"Object" -> <| "ValidationFunction" -> ((ByteArrayQ[#])&), "InterpretationFunction" -> Identity |>,
	"Owner" -> <| "ValidationFunction" -> ((stringOrLinkQ[#])&), "InterpretationFunction" -> (stringToLink[#1]&), "InterpretationInverse" -> (interpretedValueToString[#1]&) |>,
	"Data" -> <| "ValidationFunction" -> ((ByteArrayQ[#])&), "InterpretationFunction" -> Identity |>,
	"Identifier" -> <| "ValidationFunction" -> ((ByteArrayQ[#] || StringQ[#])&), "InterpretationFunction" -> Identity |>,
	"URL" -> <| "ValidationFunction" -> ((StringQ[#] || MatchQ[#, _URL])&), "InterpretationFunction" -> (If[StringQ[#1], URL[#1], #1]&), "InterpretationInverse" -> (interpretedValueToString[#1]&) |>,
	"Text" -> <| "ValidationFunction" -> ((StringQ[#])&), "InterpretationFunction" -> Identity |>,
	"EndOffset" -> <| "ValidationFunction" -> (numberOrQuantityQ[#, "UnitStringPattern"->"*seconds", "NumberTest"->(Internal`NonNegativeIntegerQ[#]&)]&), "InterpretationFunction" -> (numberOrStringToQuantity[#1, "Milliseconds", "ConversionUnits" -> "AutomaticDuration"]&), "InterpretationInverse" -> (quantityToNumeric[#1, "ConversionUnits" -> "Milliseconds", "ToInteger" -> True]&) |>,
	"StartOffset" -> <| "ValidationFunction" -> (numberOrQuantityQ[#, "UnitStringPattern"->"*seconds", "NumberTest"->(Internal`NonNegativeIntegerQ[#]&)]&), "InterpretationFunction" -> (numberOrStringToQuantity[#1, "Milliseconds", "ConversionUnits" -> "AutomaticDuration"]&), "InterpretationInverse" -> (quantityToNumeric[#1, "ConversionUnits" -> "Milliseconds", "ToInteger" -> True]&) |>,
	"EndTime" -> <| "ValidationFunction" -> (numberOrQuantityQ[#, "UnitStringPattern"->"*seconds", "NumberTest"->(Internal`NonNegativeIntegerQ[#]&)]&), "InterpretationFunction" -> (numberOrStringToQuantity[#1, "Milliseconds", "ConversionUnits" -> "AutomaticDuration"]&), "InterpretationInverse" -> (quantityToNumeric[#1, "ConversionUnits" -> "Milliseconds", "ToInteger" -> True]&) |>,
	"StartTime" -> <| "ValidationFunction" -> (numberOrQuantityQ[#, "UnitStringPattern"->"*seconds", "NumberTest"->(Internal`NonNegativeIntegerQ[#]&)]&), "InterpretationFunction" -> (numberOrStringToQuantity[#1, "Milliseconds", "ConversionUnits" -> "AutomaticDuration"]&), "InterpretationInverse" -> (quantityToNumeric[#1, "ConversionUnits" -> "Milliseconds", "ToInteger" -> True]&) |>,
	"Ordered" -> <| "ValidationFunction" -> ((BooleanQ[#])&), "InterpretationFunction" -> (If[IntegerQ[#1], (#1 == 1), #1]&), "InterpretationInverse" -> (If[TrueQ[#1], 1, 0]&) |>,
	"TopLevel" -> <| "ValidationFunction" -> ((BooleanQ[#])&), "InterpretationFunction" -> (If[IntegerQ[#1], (#1 == 1), #1]&), "InterpretationInverse" -> (If[TrueQ[#1], 1, 0]&) |>,
	"EmbeddedFrames" -> <| "ValidationFunction" -> ((validateTag["ID3v2", #])&), "InterpretationFunction" -> Identity (*(transformID3v2Frames[#]&)*) |>
|>;

(* M4A *)

$m4aElementsAssociation = <|
	"\[Copyright]nam" -> <| "Description" -> "title", "Alias" -> {"Title"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]day" -> <| "Description" -> "creation date", "Alias" -> {"CreationDate"}, "InterpretationFunction" -> (numberOrStringToDateObject[#1, "DateGranularity" -> "Year"]&), "InterpretationInverse" -> (interpretedValueToString[#1, "DateStringElements"->"Year", "StringValidationFunction"->(StringMatchQ[#,DatePattern["Year"]]&)]&), "ValidationFunction" -> (dateSpecQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]wrt" -> <| "Description" -> "composer/writer", "Alias" -> {"Composer"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,

	"\[Copyright]alb" -> <| "Description" -> "album", "Alias" -> {"Album"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]ART" -> <| "Description" -> "artist", "Alias" -> {"Artist"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"aART" -> <| "Description" -> "album artist", "Alias" -> {"AlbumArtist"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]cmt" -> <| "Description" -> "comment", "Alias" -> {"Comment"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"desc" -> <| "Description" -> "description (usually used in podcasts)", "Alias" -> {"Description"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"purd" -> <| "Description" -> "purchase date", "Alias" -> {"PurchaseDate"}, "InterpretationFunction" -> (numberOrStringToDateObject[#1]&), "InterpretationInverse" -> (interpretedValueToString[#1, "DateStringElements"->"ISODateTime"]&), "ValidationFunction" -> (dateSpecQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]grp" -> <| "Description" -> "grouping", "Alias" -> {"Grouping"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]gen" -> <| "Description" -> "genre", "Alias" -> {"Genre"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]lyr" -> <| "Description" -> "lyrics", "Alias" -> {"Lyrics"}, "InterpretationFunction" -> (interpretLyrics[#1]&), "InterpretationInverse" -> (uninterpretLyrics[#1]&), "ValidationFunction" -> ((StringQ[#1]||AssociationQ[#1])&), "ItemType" -> "StringList" |>,
	"purl" -> <| "Description" -> "podcast URL", "Alias" -> {"PodcastURL"}, "InterpretationFunction" -> (stringToLink[#1, "Wrapper"->URL]&), "InterpretationInverse" -> (interpretedValueToString[#1]&), "ValidationFunction" -> (stringOrLinkQ[#1, "Wrapper"->URL]&), "ItemType" -> "StringList" |>,
	"egid" -> <| "Description" -> "podcast episode GUID", "Alias" -> {"PodcastEpisodeGUID"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"catg" -> <| "Description" -> "podcast category", "Alias" -> {"PodcastCategory"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"keyw" -> <| "Description" -> "podcast keywords", "Alias" -> {"PodcastKeywords"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]too" -> <| "Description" -> "encoded by", "Alias" -> {"EncodedBy"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"cprt" -> <| "Description" -> "copyright", "Alias" -> {"Copyright"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"soal" -> <| "Description" -> "album sort order", "Alias" -> {"AlbumSortOrder"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"soaa" -> <| "Description" -> "album artist sort order", "Alias" -> {"AlbumArtistSortOrder"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"soar" -> <| "Description" -> "artist sort order", "Alias" -> {"ArtistSortOrder"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"sonm" -> <| "Description" -> "title sort order", "Alias" -> {"TitleSortOrder"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"soco" -> <| "Description" -> "composer sort order", "Alias" -> {"ComposerSortOrder"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"sosn" -> <| "Description" -> "show sort order", "Alias" -> {"ShowSortOrder"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"tvsh" -> <| "Description" -> "show name", "Alias" -> {"ShowTitle"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]wrk" -> <| "Description" -> "work", "Alias" -> {"Work"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]mvn" -> <| "Description" -> "movement", "Alias" -> {"Movement"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"cpil" -> <| "Description" -> "part of a compilation", "Alias" -> {"PartOfCompilation"}, "InterpretationFunction" -> (numberToBoolean[#1]&), "InterpretationInverse" -> (numberToBoolean[#1]&), "ValidationFunction" -> (numberOrBooleanQ[#1]&), "ItemType" -> "Boolean" |>,
	"pgap" -> <| "Description" -> "part of a gapless album", "Alias" -> {"PartOfGaplessAlbum"}, "InterpretationFunction" -> (numberToBoolean[#1]&), "InterpretationInverse" -> (numberToBoolean[#1]&), "ValidationFunction" -> (numberOrBooleanQ[#1]&), "ItemType" -> "Boolean" |>,
	"pcst" -> <| "Description" -> "podcast (iTunes reads this only on import)", "Alias" -> {"Podcast"}, "InterpretationFunction" -> (numberToBoolean[#1]&), "InterpretationInverse" -> (numberToBoolean[#1]&), "ValidationFunction" -> (numberOrBooleanQ[#1]&), "ItemType" -> "Boolean" |>,
	"tmpo" -> <| "Description" -> "tempo/BPM", "Alias" -> {"BeatsPerMinute"}, "InterpretationFunction" -> (numberOrStringToQuantity[#1, IndependentUnit["beats"]/"Minutes"]&), "InterpretationInverse" -> (quantityToNumeric[#1,"ToInteger"->True]&), "ValidationFunction" -> (numberOrQuantityQ[#1, "UnitStringPattern" -> $unitBeatsPerMinutePatt, "UnitPattern" -> (IndependentUnit["beats"]/"Minutes"), "NumberTest" -> (Internal`NonNegativeIntegerQ[#]&)]&), "ItemType" -> "UnsignedInteger" |>,
	"\[Copyright]mvc" -> <| "Description" -> "Movement Count", "Alias" -> {"MovementCount"}, "InterpretationFunction" -> (stringToNumeric[#1]&), "InterpretationInverse" -> (interpretedValueToString[#1(* , "KeepFractionalForm" -> True *)]&), "ValidationFunction" -> (stringOrNumberQ[#1, "NumberTest"->(Internal`NonNegativeIntegerQ[#]&)(* , "Fractional" -> True *)]&), "ItemType" -> "StringList" |>,
	"\[Copyright]mvi" -> <| "Description" -> "Movement Index", "Alias" -> {"MovementIndex"}, "InterpretationFunction" -> (stringToNumeric[#1]&), "InterpretationInverse" -> (interpretedValueToString[#1(* , "KeepFractionalForm" -> True *)]&), "ValidationFunction" -> (stringOrNumberQ[#1, "NumberTest"->(Internal`NonNegativeIntegerQ[#]&)(* , "Fractional" -> True *)]&), "ItemType" -> "StringList" |>,
	"shwm" -> <| "Description" -> "work/movement", "Alias" -> {"ShowWorkAndMovement"}, "InterpretationFunction" -> (stringToNumeric[#1, "Boolean" -> True]&), "InterpretationInverse" -> (interpretedValueToString[#1, "Boolean" -> {"1","0"}]&), "ValidationFunction" -> (stringOrBooleanQ[#1, "AllowNumericValues"->True]&), "ItemType" -> "StringList" |>,
	"stik" -> <| "Description" -> "Media Kind", "Alias" -> {"MediaKind"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (IntegerQ[#1]&), "ItemType" -> "SignedInteger" |>,
	"rtng" -> <| "Description" -> "Content Rating", "Alias" -> {"ContentRating"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (IntegerQ[#1]&), "ItemType" -> "SignedInteger" |>,
	"trkn" -> <| "Description" -> "Track Number", "Alias" -> {"TrackNumber"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (Internal`NonNegativeIntegerQ[#1]&), "ItemType" -> "UnsignedInteger" |>,
	"disk" -> <| "Description" -> "Disc Number", "Alias" -> {"DiscNumber"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (Internal`NonNegativeIntegerQ[#1]&), "ItemType" -> "UnsignedInteger" |>,
	"tves" -> <| "Description" -> "TV Episode", "Alias" -> {"TVEpisode"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (Internal`NonNegativeIntegerQ[#1]&), "ItemType" -> "UnsignedInteger" |>,
	"tvsn" -> <| "Description" -> "TV Season", "Alias" -> {"TVSeason"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (Internal`NonNegativeIntegerQ[#1]&), "ItemType" -> "UnsignedInteger" |>,
	"plID" -> <| "Description" -> "iTunes Internal ID", "Alias" -> {None}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (IntegerQ[#1]&), "ItemType" -> "SignedInteger" |>,
	"cnID" -> <| "Description" -> "iTunes Internal ID", "Alias" -> {None}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (IntegerQ[#1]&), "ItemType" -> "SignedInteger" |>,
	"geID" -> <| "Description" -> "iTunes Internal ID", "Alias" -> {None}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (IntegerQ[#1]&), "ItemType" -> "SignedInteger" |>,
	"atID" -> <| "Description" -> "iTunes Internal ID", "Alias" -> {None}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (IntegerQ[#1]&), "ItemType" -> "SignedInteger" |>,
	"sfID" -> <| "Description" -> "iTunes Internal ID", "Alias" -> {None}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (IntegerQ[#1]&), "ItemType" -> "SignedInteger" |>,
	"akID" -> <| "Description" -> "iTunes Internal ID", "Alias" -> {None}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (IntegerQ[#1]&), "ItemType" -> "SignedInteger" |>,
	"cmID" -> <| "Description" -> "iTunes Internal ID", "Alias" -> {None}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"covr" -> <| "Description" -> "cover artwork, list of MP4Cover objects (tagged strings)", "Alias" -> {"CoverImages"}, "InterpretationFunction" -> (byteArrayToImage[#1, #2]&), "InterpretationInverse" -> (imageToByteArray[#1, #2]&), "ValidationFunction" -> ((ByteArrayQ[#1] || ImageQ[#1] || (AssociationQ[#1] && ({} === Complement[Keys[#1],{"Picture","MimeType"}]) && ((MissingQ[#] || ByteArrayQ[#] || ImageQ[#])& @ #1["Picture"]) && ((MissingQ[#] || MemberQ[{"JPEG","GIF","PNG","BMP"}, #])& @ #1["MimeType"])))&), "ItemType" -> "CoverArtList" |>,

	"\[Copyright]arg" -> <| "Description" -> "Name of arranger", "Alias" -> {"Arranger"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]ark" -> <| "Description" -> "Keywords for arranger", "Alias" -> {"ArrangerKeywords"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]cok" -> <| "Description" -> "Keywords for composer", "Alias" -> {"ComposerKeyworks"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]com" -> <| "Description" -> "Name of composer", "Alias" -> {"Composer"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]cpy" -> <| "Description" -> "Copyright statement", "Alias" -> {"Copyright"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]dir" -> <| "Description" -> "Name of movie's director", "Alias" -> {"Director"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]ed1" -> <| "Description" -> "Edit dates and descriptions 1", "Alias" -> {"EditDate1"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]ed2" -> <| "Description" -> "Edit dates and descriptions 2", "Alias" -> {"EditDate2"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]ed3" -> <| "Description" -> "Edit dates and descriptions 3", "Alias" -> {"EditDate3"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]ed4" -> <| "Description" -> "Edit dates and descriptions 4", "Alias" -> {"EditDate4"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]ed5" -> <| "Description" -> "Edit dates and descriptions 5", "Alias" -> {"EditDate5"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]ed6" -> <| "Description" -> "Edit dates and descriptions 6", "Alias" -> {"EditDate6"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]ed7" -> <| "Description" -> "Edit dates and descriptions 7", "Alias" -> {"EditDate7"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]ed8" -> <| "Description" -> "Edit dates and descriptions 8", "Alias" -> {"EditDate8"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]ed9" -> <| "Description" -> "Edit dates and descriptions 9", "Alias" -> {"EditDate9"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]fmt" -> <| "Description" -> "Indication of movie format (computer-generated, digitized, and so on)", "Alias" -> {"Format"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]inf" -> <| "Description" -> "Information about the movie", "Alias" -> {"Information"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]isr" -> <| "Description" -> "ISRC code", "Alias" -> {"ISRC"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]lab" -> <| "Description" -> "Name of record label", "Alias" -> {"RecordLabel"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]lal" -> <| "Description" -> "URL of record label", "Alias" -> {"RecordLabelURL"}, "InterpretationFunction" -> (stringToLink[#1, "Wrapper"->URL]&), "InterpretationInverse" -> (interpretedValueToString[#1]&), "ValidationFunction" -> (stringOrLinkQ[#1, "Wrapper"->URL]&), "ItemType" -> "StringList" |>,
	"\[Copyright]mak" -> <| "Description" -> "Name of file creator or maker", "Alias" -> {"FileCreator"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]mal" -> <| "Description" -> "URL of file creator or maker", "Alias" -> {"FileCreatorURL"}, "InterpretationFunction" -> (stringToLink[#1, "Wrapper"->URL]&), "InterpretationInverse" -> (interpretedValueToString[#1]&), "ValidationFunction" -> (stringOrLinkQ[#1, "Wrapper"->URL]&), "ItemType" -> "StringList" |>,
	"\[Copyright]nak" -> <| "Description" -> "Title keywords of the content", "Alias" -> {"TitleKeywords"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]pdk" -> <| "Description" -> "Keywords for producer", "Alias" -> {"ProducerKeywords"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]phg" -> <| "Description" -> "Recording copyright statement, normally preceded by the symbol ../art/phono_symbol.gif", "Alias" -> {"SoundRecordingCopyright"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]prd" -> <| "Description" -> "Name of producer", "Alias" -> {"Producer"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]prf" -> <| "Description" -> "Names of performers", "Alias" -> {"Performers"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]prk" -> <| "Description" -> "Keywords of main artist and performer", "Alias" -> {"PerformerKeywords"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]prl" -> <| "Description" -> "URL of main artist and performer", "Alias" -> {"PerformerURL"}, "InterpretationFunction" -> (stringToLink[#1, "Wrapper"->URL]&), "InterpretationInverse" -> (interpretedValueToString[#1]&), "ValidationFunction" -> (stringOrLinkQ[#1, "Wrapper"->URL]&), "ItemType" -> "StringList" |>,
	"\[Copyright]req" -> <| "Description" -> "Special hardware and software requirements", "Alias" -> {"Software/HardwareRequirements"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]snk" -> <| "Description" -> "Subtitle keywords of the content", "Alias" -> {"SubtitleKeywords"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]snm" -> <| "Description" -> "Subtitle of content", "Alias" -> {"Subtitle"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]src" -> <| "Description" -> "Credits for those who provided movie source content", "Alias" -> {"SourceCredits"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]swf" -> <| "Description" -> "Name of songwriter", "Alias" -> {"SongWriter"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]swk" -> <| "Description" -> "Keywords for songwriter", "Alias" -> {"SongWriterKeywords"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>,
	"\[Copyright]swr" -> <| "Description" -> "Name and version number of the software (or hardware) that generated this movie", "Alias" -> {"Software/HardwareVersion"}, "InterpretationFunction" -> Identity, "ValidationFunction" -> (StringQ[#1]&), "ItemType" -> "StringList" |>
|>;

$m4aAtomTypesAssoc = <|
	"Implicit" -> 0,
	"UTF8String" -> 1,
	"UTF16String" -> 2,
	"SJIS" -> 3,
	"HTML" -> 6,
	"XML" -> 7,
	"UUID" -> 8,
	"ISRC" -> 9,
	"MI3P" -> 10,
	"GIF" -> 12,
	"JPEG" -> 13,
	"PNG" -> 14,
	"URL" -> 15,
	"MillisecondDuration" -> 16,
	"DateTime" -> 17,
	"Genres" -> 18,
	"Integer" -> 21,
	"ParentalAdvisory" -> 24,
	"UniversalProductCode" -> 25,
	"BMP" -> 27,
	"QTSignedInteger64" -> 74,
	"QTUnsignedInteger32" -> 77,
	"Undefined" -> 255
|>;
$m4aAtomTypesAssoc = Join[$m4aAtomTypesAssoc, Association[($m4aAtomTypesAssoc[#] -> #)& /@ Keys[$m4aAtomTypesAssoc]]];

$m4aItemTypesAssoc = AssociationThread[#, Range[Length[#]]-1]& @ {
    "ByteVectorList",
    "StringList",
    "CoverArtList",
    "Byte",
    "Boolean",
    "SignedInteger",
    "UnsignedInteger",
    "LongInteger",
    "IntegerPair"
};

(* Frame Name Translation *)

mapValuesToKey[key_,values_] := (If[# === None, Nothing, # -> key]& /@ values)

$apeTranslationAssociation = KeyMap[ToUpperCase, Association[Flatten@MapThread[mapValuesToKey[#1, #2]&, 
	{Keys[$apeElementsAssociation], $apeElementsAssociation[#, "Alias"]& /@ Keys[$apeElementsAssociation]}]]];

$xiphTranslationAssociation = KeyMap[ToUpperCase, Association[Flatten@MapThread[mapValuesToKey[#1, #2]&, 
	{Keys[$xiphElementsAssociation], $xiphElementsAssociation[#, "Alias"]& /@ Keys[$xiphElementsAssociation]}]]];

$id3v1TranslationAssociation = KeyMap[ToUpperCase, Association[Flatten@MapThread[mapValuesToKey[#1, #2]&, 
	{Keys[$id3v1ElementsAssociation], $id3v1ElementsAssociation[#, "Alias"]& /@ Keys[$id3v1ElementsAssociation]}]]];

$id3v2TranslationAssociation = KeyMap[ToUpperCase, Association[Flatten@MapThread[mapValuesToKey[#1, #2]&, 
	{Join[Keys[$id3v2FramesAssociation], Keys[$id3v2ExtendedFramesAssociation]], 
	 Join[$id3v2FramesAssociation[#, "Alias"]& /@ Keys[$id3v2FramesAssociation], $id3v2ExtendedFramesAssociation[#, "Alias"]& /@ Keys[$id3v2ExtendedFramesAssociation]]}]]];

$m4aTranslationAssociation = Association[Flatten@MapThread[mapValuesToKey[#1, #2]&, 
	{Keys[$m4aElementsAssociation], $m4aElementsAssociation[#, "Alias"]& /@ Keys[$m4aElementsAssociation]}]];

translateTagKey[tagType_, tagID_] := Switch[tagType,
	"ID3v2", $id3v2TranslationAssociation[ToUpperCase@tagID] /. _Missing -> tagID,
	"ID3v1", $id3v1TranslationAssociation[ToUpperCase@tagID] /. _Missing -> tagID,
	"APE", $apeTranslationAssociation[ToUpperCase@tagID] /. _Missing -> tagID,
	"M4A", $m4aTranslationAssociation[tagID] /. _Missing -> tagID,
	"Xiph", $xiphTranslationAssociation[ToUpperCase@tagID] /. _Missing -> tagID
]

generalizeTagKey[tagID_] := StringReplace[ToUpperCase@translateTagKey["ID3v2",tagID], {RegularExpression["T(?!XXX)..."] -> "T***", RegularExpression["W(?!XXX)..."] -> "W***"}]


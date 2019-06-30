(* Common *)
General::resyserr = "A Resource System error occurred. Please contact Wolfram Research.";
General::nneverr = "An internal error occurred during a neural net evaluation. Please try again.";
General::nnlderr = "An internal error occurred while loading a neural net. Please try again.";

(* Image *)

ImageIdentify::imginv = "Expecting an image or a list of images instead of `1`.";
ImageIdentify::numinv =
AudioIdentify::numinv = "`1` is not a positive integer or All.";
ImageIdentify::propinv =
AudioIdentify::propinv = "`1` is not a valid property.";
ImageIdentify::nores = "ImageIdentify cannot identify the image with confidence higher than the specified threshold.";

ImageInstanceQ::objinv =
ImageContainsQ::objinv = "`1` is not a known object or an Alternative between known objects.";

ImageIdentify::catinv = 
ImageInstanceQ::catinv = 
ImagePosition::catinv = 
ImageCases::catinv = 
ImageBoundingBoxes::catinv =  
ImageContents::catinv = "`1` is not a valid object category.";
AudioIdentify::catinv = "`1` is not a valid sound category.";

ImageIdentify::bdtrg = 
ImageInstanceQ::bdtrg =
AudioIdentify::bdtrg = "Value of option SpecificityGoal -> `1` is not Automatic, \"High\", \"Low\", or a real number between 0 and 1.";

ImageIdentify::thrs = 
ImageInstanceQ::thrs = 
ImageContainsQ::thrs = 
ImagePosition::thrs = 
ImageCases::thrs = 
ImageBoundingBoxes::thrs =  
ImageContents::thrs =
AudioIdentify::thrs = "Value of option AcceptanceThreshold -> `1` must be a number between 0 and 1.";

ImageIdentify::thrs2 = 
ImageInstanceQ::thrs2 = "Value of option RecognitionThreshold -> `1` must be a number between 0 and 1.";


ImageIdentify::rcat = 
ImageInstanceQ::rcat = "The specified RecognitionCategory `1` could not be interpreted.";

ImageIdentify::mlbdpg = 
ImageInstanceQ::mlbdpg =
ImageContainsQ::mlbdpg = 
ImagePosition::mlbdpg = 
ImageCases::mlbdpg = 
ImageBoundingBoxes::mlbdpg =
FindText::mlbdpg =
ImageContents::mlbdpg = "Value of option PerformanceGoal -> `1` is not Automatic, \"Speed\", or \"Quality\".";

AudioIdentify::msk = SpeechRecognize::msk = "`1` is not a valid Masking specification. It should be a real number, \
a Quantity object representing time, a Quantity object representing samples, a pair of such values \
or a list of pairs of such values";

ImageIdentify::allow = 
ImageInstanceQ::allow = 
ImageContainsQ::allow = 
ImagePosition::allow = 
ImageCases::allow = 
ImageBoundingBoxes::allow =  
ImageContents::allow = "The required data could not be downloaded. You will need to allow Internet use via the Help/Internet Connectivity dialog box, or by setting $AllowInternet = True.";
ImageIdentify::nocf = ImageIdentify::dlfail = 
ImageInstanceQ::nocf = ImageInstanceQ::dlfail = 
ImageContainsQ::nocf = ImageContainsQ::dlfail = 
ImagePosition::nocf = ImagePosition::dlfail = 
ImageCases::nocf = ImageCases::dlfail = 
ImageBoundingBoxes::nocf =  ImageBoundingBoxes::dlfail =  
ImageContents::nocf = ImageContents::dlfail = "Error downloading or installing the required data.";

ImageRestyle::styleinv = "invalid style specification."
ImageRestyle::piuns = "ImageRestyle is not supported on Raspberry Pi."

FacialFeatures::featinv = "`1` is not a supported facial feature or a list of features."
FacialFeatures::prepinv = "The function `1` does not generate valid face bounding boxes."

(* Messages for Filter faces *)
General::slotinv = "Named slots `1` in `2` do not correspond to any computable feature.";
General::testinv = "Test function `1` did not evaluate to True or False.";

FindText::inptdmns = "Value of option `1` is not Automatic, a positive integer, or a pair of equal positive integers.";
FindText::binthrs = "Value of option `1` is not Automatic, or a positive real number in the range between 0 and 1";

SpeechRecognize::lvl = "`1` is not a valid level specification. Possible values are \"Word\", \"Sentence\".";
SpeechRecognize::prop = "`1` is not a valid property specification. Possible values are \"Text\", \"Interval\" or \"Audio\".";
SpeechRecognize::beam = "Expecting a positive integer instead of `1`."


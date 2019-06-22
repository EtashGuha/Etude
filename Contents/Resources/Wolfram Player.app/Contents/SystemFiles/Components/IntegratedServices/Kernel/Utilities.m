(* Mathematica Package *)
BeginPackage["IntegratedServices`"]

IntegratedServices::errmsg = "The service has returned the following error(s): `1`";
IntegratedServices::invali = "Invalid input.";
IntegratedServices::invaln = "Invalid service name.";
IntegratedServices::noconn = "Unable to connect to service.";
IntegratedServices::offline = "The Wolfram Language is currently configured not to use the Internet. To allow Internet use, check the \"Allow the Wolfram Language to use the Internet\" box in the Help \[FilledRightTriangle] Internet Connectivity dialog.";
IntegratedServices::unreach = "Unable to reach the service server. Please check your internet connection.";
IntegratedServices::unexp = "This service is currently not available. Please try again in a few minutes.";

BillingURL = "https://billing.wolfram.com/api/addtocart.html?pid=servicecredits";
ServiceCreditsLearnMoreURL = "https://www.wolfram.com/service-credits"

PhoneVerificationURL = "https://account.wolfram.com/wolframid/add-mobile-phone";
SendMessageDocumentationURL = "http://reference.wolfram.com/language/ref/SendMessage.html";

Begin["`Private`"]

ISReturnMessage[tag_, errorcode_, params___] := With[
    {msg = MessageName[Evaluate@Symbol["IntegratedServices"], errorcode]},
    If[ MatchQ[tag, Symbol["IntegratedServices"]],
        Message[MessageName[IntegratedServices`IntegratedServices, errorcode], params];,
        MessageName[tag, errorcode] = msg;
        Message[MessageName[tag, errorcode], params];
    ]
]

iconNames = <|"BingSearch" -> "microsoft-bing.png",
			"GoogleCustomSearch" -> "google.png",
			"BingWebImageSearch" -> "microsoft-bing.png",
			"GoogleWebImageSearch" -> "google.png",
			"GoogleTranslate" -> "google-translate.png",
			"MicrosoftTranslator" -> "microsoft-translator.png",
			"Twilio" -> "twilio.png",
			"IntegratedServices" -> "wolframIntservice.png",
      "ISIcon" -> "IntserviceIcon.png"|>;

serviceTOS = <| "BingSearch"->"https://azure.microsoft.com/en-us/support/legal/",
				"GoogleCustomSearch"->"https://developers.google.com/custom-search/terms",
				"BingWebImageSearch"->"https://azure.microsoft.com/en-us/support/legal/",
				"GoogleWebImageSearch"->"https://developers.google.com/custom-search/terms",
				"GoogleTranslate"->"https://cloud.google.com/terms/",
				"MicrosoftTranslator"->"https://azure.microsoft.com/en-us/support/legal/",
				"Twilio"->"https://www.twilio.com/legal/tos",
        "DigitalGlobeGeoServer"->"https://mapsapidocs.digitalglobe.com/docs/end-user-derive-license",
        "Blockchain"->"http://www.wolfram.com/service-credits",
        "Freesound"->"https://freesound.org/docs/api/terms_of_use.html"|>;

$ServiceNamesRules = {"BingWebImageSearch"->"BingSearch","GoogleWebImageSearch"->"GoogleCustomSearch"};

getImage[imageName_]:= Dynamic[RawBoxes@FEPrivate`ImportImage[FrontEnd`ToFileName[{}, imageName]]]

getIcon[serviceName_:"IntegratedServices"]:= getImage[iconNames[serviceName]]

getIconCloud[]:= Uncompress["1:eJzVV72O00AQNgdIgK4IBYgCoT0JKGhwBVTI3BXouiiG3j7YiyKFZOXkQEflBzjJDdTuUiHS0vkBKCwhcRIF8iPkEcwOmokmi72x7weBpUl217vzzXwzO+vd2hv39i87jjOBn903YV/ub0D3iv7phe+eRVF46Hd05+VoMuiP5Ovd0VT2ZfRk76IevINyyfl/n9ls1tFSaun+BSxXS6wlwL7CMYXj4pxwC/SxRDyhZcHGoO2eAy7HSLXkrL8cP0M8iGWCkgGnOD6vwM1OgROgPxS/tIo/nGfiqhNixqb94C++65r5a/gM9iUNMIA/j/TiWFbHG+j9nBzlP687go8x3z20zco3y4scOU2MnAWZcxzdhv6csL5MxwHzIcG4wLrCgsv1u4xnyluwIdU4GYrSUqKo7/dvFTBe4btr28uGvwI5prWCdHx7uLVgeKZw3KRt/WD8xIxTD33MbLgwR9vWZdzVclyHzftaX27BW5Hjezd5TSkb4qVVtUbr6zTEhjmUVxCrwND7x9nB+OV7EGIUf/r4Iflxe7Noiluhm/Z2UOPvoioubXiGuTW6a+sI5vKyflA9aZpXx3dvxHofZ2Yusz2ieG2y2EH8KPTbimvgxBhnNVs9v9aeF7hmmQ+gO3/0oAS/eN2geoJrXJ4/LKfa5nhM3HzdeexSnHidxL5gHHHfVVNc4gbbHtmPvAnC4fWarYX5HrYF2sG/C2z55VJM0YYC+UrZHF6XCId/F9DZTTnqmfXIgm+ewQv0QRg+xDifx3Lt+WvBNc9gOv+VETeKw9y05YS4Vd9rMXu/cs5hP0P8tfvUgmvyXKKf4jR6W2BnKF3Gb6sz7gzsEIhbWYf/9ceHS+H24RQvlNDrHQzl5Jpu7IyH48hX4SvpwzWy93zbmHTVwduoP3gv5QXd21gqfurRv7FoExbp62k0lOHbwaj/+82L6ED+Aon74NY="];

getTOSLink[serviceName_]:= (getTOSLink[serviceName] = serviceTOS[serviceName])

getLearnLink[] = "http://www.wolfram.com/service-credits"

verifyTOS[type_]:= Quiet[URLExecute[CloudObject[URLBuild[{$IntegratedServicesAPIBase, "TermsStatus"}]], {"ServiceName" -> type}, "RawJSON", VerifySecurityCertificates->$CloudPRDBaseQ]]


tosApproved[type_,user_]:= With[ {resp = verifyTOS[type/.$ServiceNamesRules]},
	If[TrueQ[resp["Approved"]],
		tosApproved[type, user] = True,
		False
	]
]

End[] (* End Private Context *)

EndPackage[]

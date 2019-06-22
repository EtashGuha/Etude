(*Web-page related commands*)
OpenWebPage[x___] := seturl[x];
RefreshWebPage[x___] := refresh[x];
PageBack[x___] := back[x];
PageForward[x___] := forward[x];


formatURL[CloudObject[str_, ___]] := formatURL[str];
formatURL[URL[str_, ___]] := formatURL[str];
formatURL[str_String] := Block[{url = str},
	If[! StringContainsQ[url, "://"],
		url  = "https://" <> url;
	];
	url
]
formatURL[___]:=""

CaptureWebPage[x___] := screenshot[x];
WebPageTitle[x___] := title[x];


PageLinks[] := PageLinks[$CurrentWebSession];
PageLinks[sessionInfo_] := JavascriptExecute[sessionInfo, "
	var result = [];
	for( i=0; i<document.links.length; i++ ) {
	 result[i] = document.links[i].href;
	};
	return result;
"];

GetPageHtml[] := GetPageHtml[$CurrentWebSession];
GetPageHtml[sessionInfo_] := JavascriptExecute[sessionInfo, "return document.getElementsByTagName('html')[0].innerHTML;"];
(* Wolfram Language package *)

BeginPackage["MixpanelFunctions`"];

$mprsrvnamelist::usage = "";
$mpunitlimit::usage = "";
$mpfunnelcamel::usage = "";
camelcase::usage = "";

Begin["`Private`"];

$mprsrvnamelist = {"CampaignID" -> "mp_campaign_id", "CampaignDelivery"->"$campaign_delivery","CountryCode" -> "mp_country_code", 
	"DistinctID" -> "mp_distinct_id", "Ip" -> "mp_ip", "Length" -> "mp_length", "Lib" -> "mp_lib", 
	"NameTag" -> "mp_name_tag", "Note" -> "mp_note", "Time" -> "mp_time", "Token" -> "mp_token",
	"AndroidDevices" -> "$android_devices",	"AppRelease" -> "$app_release", "AppVersion" -> "$app_version",
	"BluetoothEnabled" -> "$bluetooth_enabled", "BluetoothVersion" -> "$bluetooth_version",
	"Brand" -> "$brand", "Browser" -> "$browser", "Carrier" -> "$carrier", "City" -> "$city",
	"Device" -> "$device", "Email" -> "$email", "FirstName" -> "$first_name", "GooglePlayServices" -> "$google_play_services", 
	"HasNfc" -> "$has_nfc", "HasTelephone" -> "$has_telephone", "InitialReferrer" -> "$initial_referrer",
	"InitialReferringDomain" -> "$initial_referring_domain", "IosDevices" -> "$ios_devices", "LastName" -> "$last_name",
	"LibVersion" -> "$lib_version", "Manufacturer" -> "$manufacturer", "Model" -> "$model", "Name" -> "$name", "Os" -> "$os", 
	"OsVersion" -> "$os_version", "Phone" -> "$phone", "Referrer" -> "$referrer", "ReferringDomain" -> "$referring_domain",
	"Region" -> "$region", "ScreenDpi" -> "$screen_dpi", "ScreenHeight" -> "$screen_height", "ScreenWidth" -> "$screen_width",
	"SearchEngine" -> "$search_engine", "Signup" -> "$signup", "Timezone" -> "$timezone", "Wifi" -> "$wifi"}

$mpunitlimit = {"minute"-> 1440, "hour" -> 8640, "day" -> 3650, "week" -> 520, "month" -> 120}

$mpfunnelcamel = {"count" -> "Count", "step_conv_ratio" -> "StepConversionRatio", "goal" -> "Goal", "overall_conv_ratio" -> "OverallConversionRatio",
	"avg_time" -> "AverageTime", "event" -> "Event", "completion" -> "Completion", "starting_amount" -> "StartingAmount", "steps" -> "Steps", "worst" -> "Worst"}

camelcase[l_List, rest___]:=camelcase[#,rest]&/@l
camelcase[str_String, separators_:{"_"}]:=StringReplace[
 StringReplace[
  StringReplace[str, 
   Thread[separators -> " "]], {WordBoundary ~~ word_ :> 
    ToUpperCase[word]}], {"Id"~~WordBoundary->"ID",WhitespaceCharacter -> "","Url"~~WordBoundary->"URL","Urls"~~WordBoundary->"URLs"}]

End[];
 
EndPackage[];

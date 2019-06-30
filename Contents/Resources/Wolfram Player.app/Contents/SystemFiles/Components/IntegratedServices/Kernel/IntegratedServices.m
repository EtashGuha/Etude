IntegratedServices`Private`$exported = {
"System`$ServiceCreditsAvailable",
"IntegratedServices`$IntegratedServicesBase",
"IntegratedServices`BillingURL",
"IntegratedServices`CreatePurchasingDialogs",
"IntegratedServices`CreateQuotaDialogs",
"IntegratedServices`CreateTOSDialogs",
"IntegratedServices`IntegratedServices",
"IntegratedServices`RemoteServiceExecute",
"IntegratedServices`ServiceCreditsAvailable",
"IntegratedServices`ServiceCreditsLearnMoreURL",
"IntegratedServices`CreatePhoneVerificationDialogs"
}

Unprotect/@IntegratedServices`Private`$exported;
ClearAll/@IntegratedServices`Private`$exported;

Get["IntegratedServices`IntegratedServicesManagment`"]
Get["IntegratedServices`Utilities`"]
Get["IntegratedServices`Dialogs`"]
Get["IntegratedServices`RequestsDialogs`"]
Get["IntegratedServices`Requests`"]
Get["IntegratedServices`Account`"]
Get["IntegratedServices`RemoteServiceExecute`"]

Protect/@IntegratedServices`Private`$exported;

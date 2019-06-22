(* Wolfram Language Package *)

BeginPackage["ResourceSystemClient`"]
(* Exported symbols added here with SymbolName::usage *)  

Begin["`Private`"] (* Begin Private Context *) 

ResourceObject::accfun="`1` is not supported by the specified ResourceObject."
ResourceObject::noas="The argument `1` should be the name or id of an existing resource or an Association defining a new resource."
ResourceObject::crname="The ResourceObject information must include a Name."
ResourceObject::unkbase="The resource system location for this $CloudBase is unknown. Please set $ResourceSystemBase."

ResourceObject::invro="The ResourceObject should contain an Association."
ResourceObject::invronb="A resource could not be created from the values provided."
ResourceObject::invcont="\"Content\" and `1` can not both be used to specify the content of a location."
ResourceObject::twocont="The default \"Content\" element is defined twice."
ResourceObject::twodef="Only one of \"Content\", \"ContentLocation\", and \"DefaultContentElement\" can be used."
ResourceObject::invas="The value of `1` should be an Association."
ResourceObject::elemcon="The element(s) `1` have multiple definitions."
ResourceObject::nocont="The ResourceObject does not contain any content."
ResourceObject::invdefa="The default element `1` is not one of the elements provided."
ResourceObject::invloc="ContentLocation should be a LocalObject or CloudObject."
ResourceObject::notf="The specified ResourceObject could not be found."
ResourceObject::cloudc="You must connect to the Wolfram cloud to access the resource."
ResourceObject::newest="The most recent version of resource `1` is already downloaded."
ResourceObject::requpdate="There is a required update for resource `1` available, the current setting prevented it from being automatically installed."
ResourceObject::updav="There is an update available for resource \"`1`\". Use ResourceUpdate to get the update."
ResourceObject::updavb="There is an update available for \"`1`\". `2`."
ResourceObject::optunav="No version of the resource `1` is available for the specified options `2`."
ResourceObject::depnbcl="The example notebook for this resource is no longer available and will not be included in the deployment. Recreate the resource to include the examples."

ResourceObject::nocofile="The content of the resource could not be retrieved. Try using ResourceUpdate to ensure the content location is up-to-date."
ResourceData::baddl="The downloaded content does not match the copy in the repository. You can use DeleteObject on the ResourceObject to force a new download."

ResourceSystemClient`ResourceDownload::exists="The resource `1` is already downloaded."
ResourceSystemClient`ResourceDownload::cloudc="Connect to the Wolfram cloud using CloudConnect to retrieve the resource content.";
ResourceObject::exists="There is already a stored version of the resource."
ResourceSubmit::exists="That resource already exists in the Wolfram Resource System."
ResourceSubmit::cloudc="You must connect to the Wolfram cloud to submit the resource."
ResourceSearch::cloudc="You must connect to the Wolfram cloud to search for resources."
ResourceAcquire::cloudc="You must connect to the Wolfram cloud to acquire resources."
ResourceObject::cloudcd="You must connect to the Wolfram cloud to deploy this resource."
ResourceObject::nocdep="The resource contains local content that may not be available in the deployment."
ResourceObject::nocdepe="The resource element `1` contains local content that may not be available in the deployment."
ResourceObject::defkey="The resource must have a value for `1`."
ResourceObject::unacq="The content of `1` can not be used until the resource has been acquired. Use ResourceAcquire[`1`]."
ResourceData::dllock="The content is currently downloading in another process, try again when it is finished. If there is not a download in progress use DeleteObject on the ResourceObject before trying again."
ResourceFunction::unacq="The resource function `1` can not be used until the resource has been acquired. Use ResourceAcquire[`1`]."


General::rscloudc="You must connect to the Wolfram cloud to use the resource."

ResourceData::invelem="`1` is not an element of the resource."
ResourceData::invelem1="The element `1` is not available."

ResourceObject::unkpar="The argument `1` is not a known property."
ResourceObject::unkrt="`1` is not a supported ResourceType."
ResourceObject::ronb1="The provided notebook is not a resource object notebook."
ResourceObject::ronb2="The resource type of the provided notebook is not a known."
ResourceObject::version="This resource is intended for version `2` and above; you are using version `1`. "

ResourceSubmit::noro="ResourceSubmit takes a ResourceObject."
ResourceSubmit::invparam="The value given for `1` is invalid."
ResourceSubmit::invparam2="Some of the specified options are not valid."
ResourceSubmit::invinfo="The information value `1` should be a short string."
ResourceSubmit::invcon="The provided content could not be used."
ResourceSubmit::noncont="The resource must include content to be submitted."
ResourceSubmit::invrt="The resource type `1` is not valid. Try DataResource."
ResourceSubmit::appperms="The permissions of `1` must allow the marketplace reviewer to read the contents."
ResourceSubmit::invprop="The submission includes invalid properties."
ResourceSubmit::enbdf="The example notebook could not be used."
ResourceSubmit::nopubid="Resource submissions must include a publisher ID. Contact Wolfram Research if you are interested in creating a publisher account."
ResourceSubmit::nopubidl="Resource submissions must include a publisher ID. If you are interested in creating a publisher account, complete the form here: `1`"
ResourceSubmit::iopts="The options in ResourceSubmit are not valid."
ResourceSubmit::invupd="The given value `1` should be a resource object from the public repository."
ResourceSubmit::invupdbase="To update the specified resource, you must submit your resource to the same resource system."
ResourceSubmit::subsuc="Your resource has been submitted for review. Your submission id is `SubmissionID`."


ResourceRegister::badpl="The location `1` can not be used. Use values supported by PersistenceLocation."
ResourceRegister::noro="A resource object was expected instead of `1`."

ResourceObject::noexamp="There are no examples available for this resource."
ResourceSearch::invcount="MaxItems `1` should be an integer greater than 0."
ResourceSearch::invquery="The query `1` should match the forms in TextSearch."
ResourceSearch::resformat="The given result format `1` is unknown, try Automatic, \"Dataset\" or \"Associations\"."
ResourceSearch::invloc="The specified search locations are invalid give a list containing \"Local\", \"ResourceSystemBase\" and \"Cloud\"."

ResourceObject::apierr="`1`"
ResourceSystemClient`ResourceDownload::apierr="`1`"
ResourceAcquire::apierr="`1`"
ResourceSearch::apierr="`1`"
ResourceSubmissionObject::apierr="`1`"
ResourceSubmit::apierr="`1`"

ResourceObject::apiwarn="`1`"
ResourceSystemClient`ResourceDownload::apiwarn="`1`"
ResourceSearch::apiwarn="`1`"

ResourceSearch::rsunavail=ResourceObject::rsunavail=ResourceData::rsunavail=ResourceAcquire::rsunavail=ResourceSubmit::rsunavail="Failed to receive data from the resource system."
ResourceUpdate::notressys="The specified resource is not from a known repository and can not be updated."
ResourceRemove::unkro="The specified resource `1` is not cached locally."

ResourceFunction::invtype="The resource type `1` is not supported by ResourceFunction."
ResourceSystemBase::unkbase="Could not communicate with the specified ResourceSystemBase `1`."

General::respaclet = "The paclet for resource type \"`1`\" is not available.";
ResourceFunction::frpaclet = "ResourceFunction is not yet available; check back soon"
General::ronotaq="This resource object must be acquired before it can be used. Use ResourceAcquire to activate it."
End[] (* End Private Context *)

EndPackage[]

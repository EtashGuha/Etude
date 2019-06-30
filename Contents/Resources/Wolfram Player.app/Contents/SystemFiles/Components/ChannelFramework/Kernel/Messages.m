(* To be moved to errmsg.m, eventually.  *)

$ChannelFrameworkVersion::usage = "$ChannelFrameworkVersion is a string that gives the version of the Channel Framework you are running.";
$ChannelFrameworkBuildNumber::usage = "$ChannelFrameworkBuildNumber is an integer that gives the current Channel Framework build number and that increases in successive versions.";

ChannelBrokerConnect::uerr = ChannelObject::uerr = RemoveChannelListener::uerr = RemoveChannelSubscribers::uerr = ChannelBrokerSockets::uerr = "Unable to perform operation because of an error: `3` (status code `2`).";
ChannelListen::access = "Access denied to `1`.";
(* same as SendMail::authfail *)
ChannelListen::authfail = "Login credentials were denied by the server \"`1`\".";
ChannelListen::cmsg = "Message `1` appears to be corrupted, and may not be processed normally."
ChannelListen::dbin = "A value of the option ChannelDatabin -> `1` must be one of None, Automatic, a Databin object, or a string ID of a databin."
ChannelListen::err = "The listener `1` aborted because of an error: `2`.";
ChannelListen::rerr = "The listener `1` relaunched because of an error: `2`.";
ChannelListen::trmt = "The listener `1` terminated by server `2`.";
ChannelListen::fbin = "Failed to store message in the databin `1`."
ChannelListen::lcon = "Host `1` did not accept the \"listen\" connection.";
ChannelListen::nxst = "No handler function is defined for `1`. Arriving messages will be recorded, but no further action will be taken.";
ChannelListen::safe = ChannelSend::safe = "Communication on external channels, such as `1`, can create a significant security risk. You must explicitly authorize communication by choosing an appropriate setting of $AllowExternalChannelFunctions.";
ChannelListen::stempu = "The listener `1` at `2` aborted because the server is currently unavailable. Try restarting the listener in a few moments.";
ChannelListen::uerr = "Unable to listen on channel `1` because of an error: `3` (status code `2`).";
ChannelListen::ltdcont = "One or more messages may not have been delivered on channel `1`.";
ChannelListener::pspec = "Part specification `1` is not a machine-sized integer, All, or Automatic.";
ChannelListener::sprop = "The message count `1` will be ignored since \"`2`\" is a property of the listener, which is the same for all messages.";
ChannelListenerStatus::obj = "Argument `1` does not represent a valid `2` object.";
ChannelObject::auth = CreateChannel::auth = "You must connect to the Wolfram Cloud to work with channel \"`1`\" in your home directory.";
ChannelObject::chspec = "`1` does not represent a valid channel name or a fully qualified URL.";
ChannelObject::copts = "Option(s) `1` of `2` appear(s) to be corrupted. The channel cannot be used."
ChannelObject::farg = CreateChannel::farg = DeleteChannel::farg = ChannelListen::farg = "`1` is not a string or valid channel object.";
ChannelObject::hfun = "Value of option HandlerFunctions -> `1` is not an association or list of rules in the form \"event\" :> f.";
ChannelObject::port = "Unable to connect using `1`. Please specify port.";
ChannelObject::qdom = "Invalid channel base `1`; a fully qualified domain expected.";
ChannelObject::ronly = "Option `1` of `2` is read only.";
ChannelObject::scheme = "Unable to connect on port `1`. Please specify scheme.";
ChannelObject::invchar = "Invalid character in `1`.";
ChannelObject::invaliduri = "The URI `1` is not valid.";
ChannelSend::enc = "Unable to convert the outgoing data `1` to a suitable format.";
ChannelSend::farg = "The first argument `1` must be a string, a valid channel object, or a valid ChannelListener object.";
ChannelSend::rcvr = "Nobody is listening on `1`.";
ChannelSend::uerr = "Unable to send to channel `1` because of an error: `3` (status code `2`).";
ChannelSubscribers::auth = "You must connect to the Wolfram Cloud to access your subscriber list.";
ChannelSubscribers::farg = RemoveChannelSubscribers::farg = "The first argument `1` must be a string, a valid channel object, or a list containing strings or channel objects.";
ChannelSubscribers::fmterr = "The server `1` did not return a valid response.";
ChannelSubscribers::uauthl = "Access to the following channel(s) denied for `1`: `2` (status code 403). The channel(s) may not exist.";
RemoveChannelSubscribers::alstr = "`1` is not a string, list of strings, or All.";
RemoveChannelSubscribers::nusrs = "No users are currently subscribed to the channel `1`.";
RemoveChannelSubscribers::usrs = "None of the users in `1` are currently subscribed to the channel `2`.";
CreateChannel::anon = DeleteChannel::anon = ChannelObject::anon = ChannelSend::anon = ChannelListen::anon = RemoveChannelListener::anon = ChannelBrokerConnect::anon = ChannelBrokerSessions::anon = RemoveChannelSubscribers::anon = "Access to `1` denied for anonymous user (status code 403).";
CreateChannel::auth = DeleteChannel::auth = ChannelObject::auth = ChannelSend::auth = ChannelListen::auth = RemoveChannelListener::auth = ChannelBrokerConnect::auth = "Unable to perform operation because of an error: 401 Unauthorized.";
CreateChannel::cbcon = DeleteChannel::cbcon = ChannelObject::cbcon = FindChannels::cbcon = ChannelSend::cbcon = ChannelListen::cbcon = RemoveChannelListener::cbcon = ChannelBrokerConnect::cbcon = ChannelBrokerDisconnect::cbcon = ChannelSubscribers::cbcon = "Unable to connect to `1`.";
CreateChannel::exst = "Channel `1` already exists. If you want to change its options, use SetOptions.";
CreateChannel::nxst = DeleteChannel::nxst = ChannelObject::nxst = ChannelSend::nxst = RemoveChannelListener::nxst = ChannelBrokerConnect::nxst = "Channel `1` does not exist (status code 404).";
CreateChannel::owner = "You cannot change permissions of the channel at `1` because you are not the owner of that channel. Use Permissions -> Automatic.";
CreateChannel::uauth = ChannelBrokerSessions::uauth = ChannelBrokerSockets::uauth = "Access to `1` denied for `2` (status code 403).";
DeleteChannel::uauth = ChannelObject::uauth = ChannelSend::uauth = ChannelListen::uauth = ChannelSubscribers::uauth = ChannelBrokerConnect::uauth = RemoveChannelSubscribers::uauth = "Access to `1` denied for `2` (status code 403). The channel may not exist.";
CreateChannel::uerr = "Unable to create channel `1` because of an error: `3` (status code `2`).";
DeleteChannel::uerr = "Unable to delete channel `1` because of an error: `3` (status code `2`).";
FindChannels::auth = "You must connect to the Wolfram Cloud to access channels in your home directory.";
FindChannels::farg = "The optional first argument `1` must be a string pattern, All, None, or Anonymous.";
FindChannels::str = "`1` is not a valid string pattern for a channel.";
FindChannels::url = "`1` is not a valid pattern for a channel URL.";
ChannelReceiverFunction::invastr = ParseChannelMessageBody::invastr = "The argument `1` is not a valid Association or a string.";

General::cbnauthx =  "The server could not verify your cloud credentials. You may need to re-connect (status code `1` for `2`).";
General::cbrateltd = "Rate limit exceeded. Connection refused  at `1` (status code `2`).";

(* in errmsg.m for V11; patch legacy versions *)
If[ $VersionNumber < 11,

	General::chobj = "Argument `1` does not represent a valid channel object.";
	General::cbrefused = "Connection at `1` refused because too many concurrent sessions are open. Please disconnect from the server in some of your sessions.";
	General::cbnauth =  "Not authorized (status code `1`) for `2`.";
	General::nauth =  "Not authorized (status code `1`).";

];

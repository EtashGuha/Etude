Package["Iconize`"]

PackageExport["IconizedImage"]

IconizedImage::deployment = "Deployment type `1` is not one of \"Default\", \"API\", \"WebForm\", \"EmbedCode\", \"Report\", \"Mobile\", \"Publish\", \"EditPage\", \"EmbedContent\", \"WebComputation\", \"ExportDoc\", or \"ScheduledProg\".";
IconizedImage::usage = "IconizedImage generates thumbnail icons of WL expressions";

SetAttributes[IconizedImage, HoldFirst];

Options[IconizedImage] = {ImageSize->Small, Platform->"FileBrowser"};
IconizedImage[expr_, dpl_: "Default", OptionsPattern[]] := Module[
	{$iconType, itheicon, paneColor, imageSize, deploy, sz, width, height, dIcon, dColor, theicon, iconpositions, temp},
	(*$iconType will be used to determine whether or not to fade the bottom of the image*)

	imageSize = OptionValue[ImageSize];
	sz = Switch[imageSize,
		Small, {85,85},
		Medium, {128, 128},
		Large, {256, 256},
		_, imageSize
	];
	
	width = sz[[1]];
	height = sz[[2]];

	(*Alert the user if an invalid deployment is specified*)
	deploy = If[MemberQ[$deployments, dpl], dpl, Message[IconizedImage::deployment,dpl];"Default"];

	(*Deployment icon and color*)
	dIcon = If[deploy == "API" && sz == {85,85}, $smallAPI, ImageResize[deployInfo[deploy][[1]], $iconFraction*sz[[2]]]];
	(*dColor = deployInfo[deploy][[2]];*)
	dColor = "#F7F7F7";

	(*Make default icon*)
	$default := With[{icon = toicon[Evaluate@$defaultSpikey, {width, $paneFraction*height}, dColor]},
			ImagePad[icon[[2]],{{0,0},{0,1}}, Padding->hexToRGB["#b2b2b2"]]
		];

	(*Primary icon for expression*)
	(*The image pad is for the thin border separating the upper and lower panes*)
	itheicon = Quiet @ With[{icon = toicon[Unevaluated[expr], {width, $paneFraction*height}, dColor]},
			$iconType = icon[[1]];
			ImagePad[icon[[2]],{{0,0},{0,1}}, Padding->hexToRGB["#b2b2b2"]]
		];

	(*If for whatever reason, the iconization process fails, return the default icon*)
	theicon = If[UnsameQ[Head[itheicon], Image], $default, itheicon];	

	(*This list formerly held the category icons, which might return in the future.*)
	iconpositions = {};
	
	(*If the deployment is not default, then place the deployment icon in the upper RHS*)
	If[dpl != "Default",
	AppendTo[iconpositions, Inactive[Sequence][dIcon,{$heightFraction*width, $paneFraction*height + .5*(1-$paneFraction)*height}]]];
	
	(*The background color for the upper pane can be different for different platforms.
	For the mobile browser, we want the pane to match the background of the surrounding box, 
	so that it looks transparent.*)
	paneColor = Switch[OptionValue[Platform],
		"mobile", $mobilePaneColor,
		"Android", $mobilePaneColor,
		"IOS", $mobilePaneColor,
		"FileBrowser", hexToRGB["#fafafa"],
		"Experimental", hexToRGB[deployInfo[deploy][[2]]],
		_, $mobilePaneColor
	];
	
	temp = Fold[Inactive[ImageCompose], ImagePad[theicon,{{0,0},{0, $diffFraction*height}}, paneColor], iconpositions];

	Switch[$iconType, 
		"Expression", fadeArray[Activate@temp, hexToRGB[dColor], .3], 
		_, Activate@temp
		]
]

Package["Iconize`"]

(*******************************Global Variables*********************************************)
PackageScope["$deployments"]
$deployments = {"Default", "API", "WebForm", "EmbedCode", 
	"Report", "Mobile", "Publish", "EditPage", "EmbedContent", 
	"WebComputation", "ExportDoc", "ScheduledProg"};

PackageScope["$graphicsHeads"]	
$graphicsHeads = {Graphics, Graphics3D, GeoGraphics, Image, Legended, Graph};

(*
	The HoldPattern here is to prevent autoloading of the
	corresponding system paclets
*)

PackageScope["$dynamicObjects"]
$dynamicObjects = HoldPattern /@ {Manipulate, MenuView, Slider, Button, 
   ChoiceButtons, PopupMenu, ActionMenu, TabView, FlipView, 
   SlideView, OpenerView, Opener, Checkbox, Toggler, Setter, Panel, 
   Slider2D, VerticalSlider, ProgressIndicator, Animator, 
   Manipulator, Control};

PackageScope["$formatFormats"]
$formatFormats := $formatFormats = Get[PacletManager`PacletResource["Iconize", "formatFormats.m"]];
 
(*TODO: get rid of this*) 
PackageScope["$deployColors"]   
$deployColors = {"#e8e8e8", "#ffdd9b", "#ffe4c9", "#dfb5ff", "#e0f1fc", 
"#bdc8f9", "#eef9bd", "#ffd5b3", "#dbfff3", "#d2e7ff", "#dfffca", "#fbe2ff"}; 

(*TODO get rid of this*)
PackageScope["$deployColorInfo"]    
$deployColorInfo = Association[Rule@@@Transpose[{$deployments,$deployColors}]];

PackageScope["$smallAPI"]
PackageScope["$defaultSpikey"]
PackageScope["$notebookTemplate"]
$smallAPI := $smallAPI = Import[FileNameJoin[{PacletManager`PacletResource["Iconize","DeploymentIcons"], "API_small.png"}]];
$defaultSpikey := $defaultSpikey = Import[PacletManager`PacletResource["Iconize", "defaultSpikey.png"]];
$notebookTemplate := $notebookTemplate = Import[PacletManager`PacletResource["Iconize", "notebook_template.png"]];

PackageScope["$ImageResolution"]
$ImageResolution = 72;  

(*********************Image constants*************************)
(*The fraction of the icon devoted to the expression (i.e. the remainder without the pane)*)
PackageScope["$paneFraction"]
$paneFraction = N[68.4/85];

(*The x position to place the first icon, i.e. the deployment icon*)
PackageScope["$startFraction"]
$startFraction = N[59.76/85];

(*The vertical position of the category icons*)
(*This is also the horizontal position in the case of square icons*)
PackageScope["$heightFraction"]
$heightFraction = N[76.4/85];

(*The spacing between category icons*)
(*Also the size of the upper icon pane*)
PackageScope["$diffFraction"]
$diffFraction = N[16.6/85];

(*The relative width of the category icons compared to the whole icon*)
PackageScope["$catIconFraction"]
$catIconFraction = N[20/128];

(*LHS padding*)
PackageScope["$textSpacing"]
$textSpacing = .12;

(*Upper padding*)
PackageScope["$upperTextSpacing"]
$upperTextSpacing = .12;

(*Width of category icons*)
PackageScope["$iconFraction"]
$iconFraction = N[12/85];

(*Aspect ratio of Helvetica*)
PackageScope["$aspectRatio"]
$aspectRatio = .52;

(*Height of document icons*)
PackageScope["$documentHeight"]
$documentHeight = N[50/85];

(*Background color of upper pane for mobile icons*)
PackageScope["$mobilePaneColor"]
$mobilePaneColor = RGBColor[247,247,247];

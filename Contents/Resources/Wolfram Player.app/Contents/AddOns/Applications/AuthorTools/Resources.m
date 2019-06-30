(* English Resources *)

(* See AuthorTools`Common`$Resource for details *)


BeginPackage["AuthorTools`Resources`"]

Begin["`Private`"]


(* convenience for recursive calls: *)
$Resource = AuthorTools`Common`$Resource;


(* Styles *)

Resource["Button1Background"] = RGBColor[0, 0.32549, 0.537255];
Resource["Button2Background"] = RGBColor[0.329412, 0.584314, 0.694118];
Resource["Button3Background"] = RGBColor[0.537255, 0.72549, 0.843137];

Resource["Button1Text"] = GrayLevel[1];
Resource["Button2Text"] = GrayLevel[1];
Resource["Button3Text"] = GrayLevel[1];
Resource["ErrorText"] = RGBColor[1,0,0];

Resource["Font"] = "Helvetica";

Resource["Restore", "Typeset color"] = RGBColor[0,1,0];
Resource["Restore", "Graphic color"] = RGBColor[0,0,1];
Resource["Restore", "Corrupt color"] = GrayLevel[1];
Resource["Restore", "Corrupt background"] = RGBColor[1,0,0];
Resource["Restore", "Corrupt font"] = "Courier";


Resource["Italic"] = "Italic";

Resource["WideMargin"] = 6;
Resource["SlimMargin"] = 3;

Resource["GrayBackground"] = GrayLevel[0.85098];
Resource["WhiteBackground"] = GrayLevel[1];


(* English Strings *)

Resource["Close"] = "Close";
Resource["Cancel"] = "Cancel";
Resource["OK"] = "OK";
Resource["WideOK"] = "       OK       ";
Resource["Apply"] = "Apply";
Resource["Yes"] = "Yes";
Resource["No"] = "No";
Resource["Browse..."] = "Browse\[Ellipsis]";
Resource["Show"] = "Show";

Resource["Set Option", "Title"] = "Set Option";

Resource["Project", "Bad Directory"] = "Select a valid directory first.";
Resource["Project", "no files"] = "--None--";

Resource["Index", "Title"] = "Edit Notebook Index";
Resource["Index", "Add Tags"] = "Add Indexing Tags to the Current Cell";
Resource["Index", "Editing"] = "Editing:  ";
Resource["Index", "Cancel edit"] = "Cancel Edit";
Resource["Index", "Error"] = "Error: make sure you have the correct cell selected.";
Resource["Index", "Tag button"] = "Tag Current Cell";
Resource["Index", "Update button"] = "Refresh";
Resource["Index", "Edit"] = "Edit";
Resource["Index", "Copy"] = "Copy";
Resource["Index", "Remove"] = "Remove";
Resource["Index", "Main Entry"] = "Index Main Entry:";
Resource["Index", "Sub Entry"] = "Index Sub-Entry (Optional):";
Resource["Index", "Short Form"] = "Short Form (Optional\[LongDash]For master index only):";
Resource["Index", "List of entries"] = "Index Entries for the Current Cell:";
Resource["Index", "Caption"] = "Reading index information\[Ellipsis]";
Resource["Index", "No index"] = "There were no index entries in the specified notebook(s).";
Resource["Index", "Dash"] = "\[Dash]";
Resource["Index", "List"] = ": List of IndexData expressions ";
Resource["Index", "Index"] = " Index";
Resource["Index", "not found"] = "match not found";


Resource["Headers", "Title"] = "Headers and Footers";
Resource["Headers", "page1"] = "1";
Resource["Headers", "page2"] = "2";
Resource["Headers", "for"] = "for notebook: ";
Resource["Headers", "active"] = "\[FilledCircle]";
Resource["Headers", "inactive"] = "\[EmptyCircle]";
Resource["Headers", "Set Facing"] = "Set Facing Pages:";
Resource["Headers", "Left aligned"] = "Left-Aligned";
Resource["Headers", "Centered"] = "Centered";
Resource["Headers", "Right aligned"] = "Right-Aligned";
Resource["Headers", "Insert Value"] = "Insert Value\[Ellipsis]";
Resource["Headers", "Running Head"] = "Insert Running Head\[Ellipsis]";

Resource["Header", "on first page"] = "Header on First Page?";
Resource["Footer", "on first page"] = "Footer on First Page?";
Resource["HeaderLine", "Left"] = "Left Page Header Lines?";
Resource["HeaderLine", "Right"] = "Right Page Header Lines?";
Resource["HeaderLine", "None"] = "Header Lines?";
Resource["FooterLine", "Left"] = "Left Page Footer Lines?";
Resource["FooterLine", "Right"] = "Right Page Footer Lines?";
Resource["FooterLine", "None"] = "Footer Lines?";

Resource["Header", "Left"] = "Left Page Headers";
Resource["Header", "Right"] = "Right Page Headers";
Resource["Header", "None"] = "Headers";
Resource["Footer", "Left"] = "Left Page Footers";
Resource["Footer", "Right"] = "Right Page Footers";
Resource["Footer", "None"] = "Footers";

Resource["Printing", "no nb"] = "Open a notebook to use this button.";
Resource["Printing", "StartingPageNumber"] = "Specify the page number for the first page of the notebook.";
Resource["Printing", "PrintingMarings"] = "Specify the {{left, right}, {bottom, top}} margins for the printed notebook.";
Resource["Printing", "CellBrackets"] = "Specify whether cell brackets should appear on the printout.";
Resource["Printing", "Highlighting"] = "Specify whether highlighted expressions should appear highlighted on the printout.";
Resource["Printing", "Crop"] = "Specify whether registration (crop) marks should appear on the printout.";
Resource["Printing", "MultipleHorizontal"] = "Specify whether cells wider than one page should continue printing on another horizontal page.";
Resource["Printing", "Pick a style"] = "Enter the desired style name:";
Resource["Printing", "Pick a tag"] = "Enter the desired cell tag:";

Resource["Export", "Pick a style"] = "Enter the desired style name:";
Resource["Export", "Pick a tag"] = "Enter the desired cell tag:";
Resource["Export", "Extracting..."] = "Extracting\[Ellipsis]";

Resource["Bilateral", "SampleText"] = "This is the sum of two numbers.";
Resource["Bilateral", "SampleInput"] = "2.3 + 5.63";
Resource["Bilateral", "SampleOutput"] = "7.93`";

Resource["Categories", "Reading..."] = "Reading category information\[Ellipsis]";
Resource["Categories", "Tagging..."] = "Adding CellTags used by the BrowserCategories...";

Resource["Contents", "Reading..."] = "Reading contents information\[Ellipsis]";
Resource["Contents", "Tagging..."] = "Adding CellTags used by the Table of Contents\[Ellipsis]"
Resource["Contents", "List"] = ": List of ContentsData expressions";
Resource["Contents", "Contents"] = " Contents";
Resource["Contents", "Categories"] = " Categories";
Resource["Contents", "Open"] = "Open ";

Resource["Diff", "Starting"] = "Starting DiffReport...";
Resource["Diff", "Starting nb"] = "Starting NotebookDiff...";
Resource["Diff", "Processing"] = "Processing...";
Resource["Diff", "Finished"] = "Finished.";

Resource["Diff", "Style title"] = "StyleSheet Differences";
Resource["Diff", "No name"] = "Notebook[\[Ellipsis]]";
Resource["Diff", "Sheet 1"] = "Style sheet 1";
Resource["Diff", "Sheet 2"] = "Style sheet 2";
Resource["Diff", "1"] = "1";
Resource["Diff", "2"] = "2";
Resource["Diff", "Environments"] = "Environments";
Resource["Diff", "Styles"] = "Styles";
Resource["Diff", "Notebook Options"] = "Notebook Options";

Resource["Diff", "Kernel prompt"] = "Enter a filename:";
Resource["Diff", "Notebook title"] = "Notebook Differences";
Resource["Diff", "Other title"] = "Project or Directory Differences";
Resource["Diff", "Selector title", "Notebook"] = "Select Notebook";
Resource["Diff", "Selector title", "Notebooks"] = "Select Notebooks";
Resource["Diff", "Selector title", "Files"] = "Select Files";
Resource["Diff", "Selector No Notebooks"] = "No notebooks";
Resource["Diff", "Running"] = "Running...";
Resource["Diff", "Change View"] = "Change View";
Resource["Diff", "Update"] = "Update";
Resource["Diff", "Close"] = "Save All and Close";
Resource["Diff", "Help"] = "Help";
Resource["Diff", "Computing"] = "Computing differences\[Ellipsis]";
Resource["Diff", "Apply left"] = "\[LeftArrow]Apply to 1st";
Resource["Diff", "Apply right"] = "Apply to 2nd\[RightArrow]";
Resource["Diff", "Extra in new"] = "Extra cells in first.";
Resource["Diff", "Extra in old"] = "Extra cells in second.";
Resource["Diff", "Different cells"] = "Different cells."
Resource["Diff", "New"] = "First";
Resource["Diff", "Old"] = "Second";
Resource["Diff", "In new"] = "In First:";
Resource["Diff", "In old"] = "In Second:";
Resource["Diff", "Different opts"] = "Different options."
Resource["Diff", "Moved"] = "Moved cells.";
Resource["Diff", "Same cells"] = "No differences in cells.";
Resource["Diff", "Same opts"] = "No difference in options.";
Resource["Diff", "Identical cells"] = "These cells are identical.";
Resource["Diff", "Can't compare"] = "Cannot compare the content of these cells."
Resource["Diff", "CellDiffs"] = "Cell Differences";
Resource["Diff", "Select cells"] = "Select Cells to Compare";
Resource["Diff", "Select one"] = "Select one cell from each notebook.";
Resource["Diff", "View diffs"] = "View Differences";
Resource["Diff", "New", "Notebook"] = "First Notebook:";
Resource["Diff", "Old", "Notebook"] = "Second Notebook:";
Resource["Diff", "New", "Files"] = "First set of Files:";
Resource["Diff", "Old", "Files"] = "Second set of Files:";
Resource["Diff", "Compare"] = "Compare";
Resource["Diff", "Compare note", "notebooks"] = "Compare selected notebooks";
Resource["Diff", "Compare note", "files"] = "Compare selected files";
Resource["Diff", "Changed notebooks"] = "Notebooks that Differ";
Resource["Diff", "Identical notebooks"] = "Notebooks that are Identical";
Resource["Diff", "", "Differences"] = "Differences";
Resource["Diff", "Notebook List ", "Differences"] = "Notebook List Differences";
Resource["Diff", "Project ", "Differences"] = "Project Differences";
Resource["Diff", "Directory ", "Differences"] = "Directory Differences";
Resource["Diff", "Only first list"] = "Notebooks that are only in first list.";
Resource["Diff", "Only second list"] = "Notebooks that are only in second list.";
Resource["Diff", "Only prefix", file_] = {"Notebooks that are only in ", file, "."};
Resource["Diff", "cells"] = "cells";


Resource["Diff", "Select button text"] = "Select";
Resource["Diff", "Choose notebook"] = "Choose a notebook already open in Mathematica.";
Resource["Diff", "Select text"] = " an open notebook";
Resource["Diff", "or"] = ", or ";
Resource["Diff", "Browse button text"] = "Browse";
Resource["Diff", "Browse text"] = " for a notebook file.";
Resource["Diff", "Choose file"] = "Choose the filename of a notebook.";
Resource["Diff", "Choose dir"] = "Choose a file from a directory.";
Resource["Diff", "Choose proj"] = "Choose the filename of a project.";
Resource["Diff", "Select dir"] = " for a file from a directory";
Resource["Diff", "Select proj"] = " for a project file.";

Resource["Diff", "Select notebook", nb_] :=
{
  $Resource["Diff", "Select button", nb],
  $Resource["Diff", "Select text"],
  $Resource["Diff", "or"],
  $Resource["Diff", "Browse button1", nb],
  $Resource["Diff", "Browse text"]
};

Resource["Diff", "Select files", nb_] := 
{
  $Resource["Diff", "Browse button2", nb],
  $Resource["Diff", "Select dir"],
  $Resource["Diff", "or"],
  $Resource["Diff", "Browse button3", nb],
  $Resource["Diff", "Select proj"]
};

Resource["Diff", "Select button", nb_] := 
StyleBox[
  ButtonBox[$Resource["Diff", "Select button text"],
    ButtonFunction:>(Needs[ "AuthorTools`NotebookDiff`"]; AuthorTools`NotebookDiff`Private`openNBFunction[nb, ButtonNotebook[ ]]),
    ButtonNote->$Resource["Diff", "Choose notebook"]
  ],
  FontColor->RGBColor[0.100008, 0.4, 0.6],
  FontVariations->{"Underline"->True}
];

Resource["Diff", "Browse button1", nb_] := 
StyleBox[
  ButtonBox[$Resource["Diff", "Browse button text"],
     ButtonFunction:>(Needs[ "AuthorTools`NotebookDiff`"]; AuthorTools`NotebookDiff`Private`browseButtonFunction[nb, ButtonNotebook[ ], AuthorTools`NotebookDiff`Private`myFileBrowse[ False]]),
     ButtonNote->$Resource["Diff", "Choose file"]
  ],
  FontColor->RGBColor[0.100008, 0.4, 0.6],
  FontVariations->{"Underline"->True}
];

Resource["Diff", "Browse button2", nb_] :=
StyleBox[
  ButtonBox[$Resource["Diff", "Browse button text"],
    ButtonFunction:>CompoundExpression[Needs[ "AuthorTools`NotebookDiff`"], AuthorTools`NotebookDiff`Private`openDirectoryFunction[nb, ButtonNotebook[ ]]],
    ButtonNote->$Resource["Diff", "Choose dir"]
  ],
  FontColor->RGBColor[0.100008, 0.4, 0.6],
  FontVariations->{"Underline"->True}
];

Resource["Diff", "Browse button3", nb_] :=
StyleBox[
  ButtonBox[$Resource["Diff", "Browse button text"],
    ButtonFunction:>CompoundExpression[Needs[ "AuthorTools`NotebookDiff`"], AuthorTools`NotebookDiff`Private`openProjectFunction[nb, ButtonNotebook[ ]]],
    ButtonNote->$Resource["Diff", "Choose proj"]
  ],
  FontColor->RGBColor[0.100008, 0.4, 0.6],
  FontVariations->{"Underline"->True}
];





(* Logos *)

Resource["Logo"] =
Cell[GraphicsData["Bitmap", "\<\
CF5dJ6E]HGAYHf4PAg9QL6QYHg<PAVmbKF5d0`40001Y00004b000`400?l00000o`00003o00<00000
00P820000000I000000100<30`40000000000`00000020P80000001T000000400`<30@0000000003
0000000820P0000006@000000@030`<10000000000<0000000P8200000006`0000000`0R8R80A4A4
014A4@0?00000080<c<c3`00000203<c<`D000000P0c<c<7000000808R8R5000000100<30`400000
00000`00000020P80000000@0000000309VIV@3oool0MgMg00@000000P3oool00`0A4A40MgMg0?oo
o`020?ooo`0808R8R00000004A4A0?ooo`2k^k/000000;^k^`3oool2000000050=gMg@3oool00000
014A4@3<c<`00P3oool01P3MgMd0EEEE0000001gMgL0oooo04A4A0<0000000H04A4A0?ooo`0c<c<0
0000014A4@3<c<`20?ooo`050=gMg@0c<c<00000014A4@3<c<`00P3oool02P3MgMd0<c<c0000001E
EED0gMgM0000001gMgL0oooo0=gMg@2IVITC000000400`<30@00000000030000000820P000000100
000000@08R8R0>k^kP3^k^h04A4A0P000000100c<c<0oooo0=gMg@0000020?ooo`0I06IVIP2ZZZX0
oooo06IVIP000000gMgM0>k^kP000000VIVI0?ooo`14A4@000000;^k^`3oool08R8R09VIV@3oool0
ZZZZ07MgM`3^k^h0oooo04A4A01VIVH0oooo07MgM`040000000I0>k^kP1EEED0000009VIV@3<c<`0
4A4A0000001gMgL0k^k^028R8P2IVIT0c<c<014A4@000000MgMg0>k^kP0R8R80<c<c0?ooo`000000
A4A403<c<`000000c<c<06IVIP0B000000400`<30@00000000030000000820P000000140000000<0
MgMg0?ooo`3oool0103oool07@2IVIT000000?ooo`3<c<`00000028R8P3oool0^k^k0000002k^k/0
oooo014A4@1gMgL0oooo06IVIP000000MgMg0?ooo`14A4@0^k^k0?ooo`0R8R80000003<c<`3oool0
ZZZZ04A4A03oool0VIVI00@0000000D0^k^k07MgM`000000^k^k07MgM`0300000080VIVI00<0^k^k
07MgM`0000000P00000209VIV@030000003oool0<c<c0080000000<0IVIV0>k^kP1VIVH04P000001
00<30`40000000000`00000020P80000000A0000000;014A4@3^k^h0oooo0<c<c02k^k/0k^k^0?oo
o`1gMgL000000<c<c03oool00P0000020>k^kP0300000028R8P0oooo0080A4A400T0oooo07MgM`00
0000IVIV0?ooo`1gMgL0ZZZZ0?ooo`0R8R800P0000020?ooo`030000003oool0c<c<00@0000000D0
R8R809VIV@000000VIVI08R8R0030000000407MgM`2k^k/0VIVI08R8R0<0000000T0MgMg0;^k^`00
0000c<c<05EEE@000000^k^k0>k^kP1gMgL04`00000100<30`40000000000`00000020P80000000B
0000000o07MgM`3oool0MgMg0000002k^k/0oooo04A4A0000000ZZZZ0?ooo`0c<c<000000;^k^`3o
ool0000009VIV@3oool0R8R8028R8P3oool0gMgM014A4@1EEED0oooo07MgM`1EEED0oooo0<c<c00A
4A40A4A40?ooo`3^k^h000000=gMg@3oool0IVIV0000000A4A40000007MgM`2k^k/0000003<c<`3o
ool0IVIV0000000A4A40c<c<09VIV@0c<c<0oooo06IVIP0000004A4A0<c<c02IVIT000000:ZZZP1g
MgL000000?ooo`0R8R804A4A01<000000@030`<10000000000<0000000P8200000004`0000001@3M
gMd0k^k^0000003^k^h0oooo0080000000P0MgMg0?ooo`14A4@0000009VIV@3oool0A4A40=gMg@80
oooo00@0^k^k0?ooo`3oool0k^k^0P3oool00`1gMgL0000009VIV@040?ooo`0307MgM`000000^k^k
00<0oooo00@0IVIV00000014A4@0oooo0P000000201EEED0gMgM0=gMg@3oool0c<c<028R8P000000
EEEE0P3MgMd02`3oool0c<c<028R8P000000MgMg0:ZZZP000000R8R80=gMg@3^k^h0A4A401800000
0@030`<10000000000<0000000P8200000004`0000001@1VIVH0oooo07MgM`3oool0c<c<00800000
00h0A4A407MgM`14A4@0000004A4A01gMgL08R8R06IVIP3oool0gMgM06IVIP3<c<`0oooo04A4A080
MgMg0`0000001014A4@0MgMg07MgM`14A4@20000000905EEE@1gMgL0A4A407MgM`14A4@00000028R
8P3oool08R8R00<000000P14A4@500000080A4A40`000000101VIVH0^k^k00000000000203<c<a<0
00000@030`<10000000000<0000000P82000000050000000103MgMd0oooo0?ooo`28R8P:00000006
0?ooo`3MgMd000000:ZZZP3oool0<c<c4P0000000`3oool0A4A40000000>0000000304A4A03oool0
000001H000000@030`<10000000000<0000000P820000000500000001014A4@0oooo0?ooo`1VIVH:
0000000606IVIP1gMgL0000007MgM`3oool0EEEE4000000204A4A0040<c<c028R8P0A4A403<c<`h0
000000<0oooo028R8P0000005@00000200@4100000<0000000P8200000005@0000000`2k^k/0oooo
03<c<`0=0000000305EEE@3oool0MgMg010000001@2k^k/00`2ZZZX000000000000<000000030=gM
g@14A4@0000001D000000@030`<101`L700000<06ATI00<30`0000009@0000000`14A4@0oooo09VI
V@0T000000030;^k^`1gMgL0000001D000000@030`<101TI6@0000<06a/K00<30`000000C0000000
0`0c<c<08R8R0000000E000000400`<30@0K6a/0000305YJFP071`L0000006@000000@071`L105YJ
FP0000<0Z:RX04a<C00410@0H`00000100@41040C4a<0@2XZ:P000060>3Ph02XZ:P0FUYJ02DU9@0F
5QH04Q8B300000000`0B4Q800P8200820P0500820QL04Q8B00<051@D0000000000007@000000100;
2`/04Q8B018B4P0B4Q8<000000804Q8B00D02`/;01HF5P0U9BD0FUYJ0:RXZ0010>3Ph000\
\>"], "Graphics",
  CellFrame->{{0, 0}, {2, 0}},
  ShowCellBracket->False,
  CellMargins->{{4, 0}, {0, 8}},
  Active -> False,
  Editable -> False,
  Selectable -> False,
  Evaluatable->False,
  CellFrameMargins->False,
  ImageSize->{105, 19},
  ImageMargins->{{0, 0}, {0, 0}},
  ImageRegion->{{0, 1}, {0, 1}},
  CellTags->"LongTab"]


Resource["ShortLogo"] =
Cell[GraphicsData["Bitmap", "\<\
CF5dJ6E]HGAYHf4PAg9QL6QYHg<PAVmbKF5d0`40001A00004b000`400?l00000o`00003o00<00000
00P820000000C000000100<30`40000000000`00000020P80000001<000000400`<30@0000000003
0000000820P0000004`000000@030`<10000000000<0000000P8200000003`0000000`0R8R80A4A4
014A4@0?00000080<c<c3`00000203<c<`D000000P0c<c<7000000808R8R2000000100<30`400000
00000`00000020P8000000040000000309VIV@3oool0MgMg00@000000P3oool00`0A4A40MgMg0?oo
o`020?ooo`0808R8R00000004A4A0?ooo`2k^k/000000;^k^`3oool2000000050=gMg@3oool00000
014A4@3<c<`00P3oool01P3MgMd0EEEE0000001gMgL0oooo04A4A0<0000000H04A4A0?ooo`0c<c<0
0000014A4@3<c<`20?ooo`050=gMg@0c<c<00000014A4@3<c<`00P3oool02P3MgMd0<c<c0000001E
EED0gMgM0000001gMgL0oooo0=gMg@2IVIT7000000400`<30@00000000030000000820P0000000@0
000000@08R8R0>k^kP3^k^h04A4A0P000000100c<c<0oooo0=gMg@0000020?ooo`0I06IVIP2ZZZX0
oooo06IVIP000000gMgM0>k^kP000000VIVI0?ooo`14A4@000000;^k^`3oool08R8R09VIV@3oool0
ZZZZ07MgM`3^k^h0oooo04A4A01VIVH0oooo07MgM`040000000I0>k^kP1EEED0000009VIV@3<c<`0
4A4A0000001gMgL0k^k^028R8P2IVIT0c<c<014A4@000000MgMg0>k^kP0R8R80<c<c0?ooo`000000
A4A403<c<`000000c<c<06IVIP06000000400`<30@00000000030000000820P0000000D0000000<0
MgMg0?ooo`3oool0103oool07@2IVIT000000?ooo`3<c<`00000028R8P3oool0^k^k0000002k^k/0
oooo014A4@1gMgL0oooo06IVIP000000MgMg0?ooo`14A4@0^k^k0?ooo`0R8R80000003<c<`3oool0
ZZZZ04A4A03oool0VIVI00@0000000D0^k^k07MgM`000000^k^k07MgM`0300000080VIVI00<0^k^k
07MgM`0000000P00000209VIV@030000003oool0<c<c0080000000<0IVIV0>k^kP1VIVH01P000001
00<30`40000000000`00000020P8000000050000000;014A4@3^k^h0oooo0<c<c02k^k/0k^k^0?oo
o`1gMgL000000<c<c03oool00P0000020>k^kP0300000028R8P0oooo0080A4A400T0oooo07MgM`00
0000IVIV0?ooo`1gMgL0ZZZZ0?ooo`0R8R800P0000020?ooo`030000003oool0c<c<00@0000000D0
R8R809VIV@000000VIVI08R8R0030000000407MgM`2k^k/0VIVI08R8R0<0000000T0MgMg0;^k^`00
0000c<c<05EEE@000000^k^k0>k^kP1gMgL01`00000100<30`40000000000`00000020P800000006
0000000o07MgM`3oool0MgMg0000002k^k/0oooo04A4A0000000ZZZZ0?ooo`0c<c<000000;^k^`3o
ool0000009VIV@3oool0R8R8028R8P3oool0gMgM014A4@1EEED0oooo07MgM`1EEED0oooo0<c<c00A
4A40A4A40?ooo`3^k^h000000=gMg@3oool0IVIV0000000A4A40000007MgM`2k^k/0000003<c<`3o
ool0IVIV0000000A4A40c<c<09VIV@0c<c<0oooo06IVIP0000004A4A0<c<c02IVIT000000:ZZZP1g
MgL000000?ooo`0R8R804A4A00L000000@030`<10000000000<0000000P8200000001`0000001@3M
gMd0k^k^0000003^k^h0oooo0080000000P0MgMg0?ooo`14A4@0000009VIV@3oool0A4A40=gMg@80
oooo00@0^k^k0?ooo`3oool0k^k^0P3oool00`1gMgL0000009VIV@040?ooo`0307MgM`000000^k^k
00<0oooo00@0IVIV00000014A4@0oooo0P000000201EEED0gMgM0=gMg@3oool0c<c<028R8P000000
EEEE0P3MgMd02`3oool0c<c<028R8P000000MgMg0:ZZZP000000R8R80=gMg@3^k^h0A4A400H00000
0@030`<10000000000<0000000P8200000001`0000001@1VIVH0oooo07MgM`3oool0c<c<00800000
00h0A4A407MgM`14A4@0000004A4A01gMgL08R8R06IVIP3oool0gMgM06IVIP3<c<`0oooo04A4A080
MgMg0`0000001014A4@0MgMg07MgM`14A4@20000000905EEE@1gMgL0A4A407MgM`14A4@00000028R
8P3oool08R8R00<000000P14A4@500000080A4A40`000000101VIVH0^k^k00000000000203<c<`L0
00000@030`<10000000000<0000000P82000000020000000103MgMd0oooo0?ooo`28R8P:00000006
0?ooo`3MgMd000000:ZZZP3oool0<c<c4P0000000`3oool0A4A40000000>0000000304A4A03oool0
000000X000000@030`<10000000000<0000000P820000000200000001014A4@0oooo0?ooo`1VIVH:
0000000606IVIP1gMgL0000007MgM`3oool0EEEE4000000204A4A0040<c<c028R8P0A4A403<c<`h0
000000<0oooo028R8P0000002@00000200@4100000<0000000P8200000002@0000000`2k^k/0oooo
03<c<`0=0000000305EEE@3oool0MgMg010000001@2k^k/00`2ZZZX000000000000<000000030=gM
g@14A4@0000000T000000@030`<101`L700000<06ATI00<30`0000006@0000000`14A4@0oooo09VI
V@0T000000030;^k^`1gMgL0000000T000000@030`<101TI6@0000<06a/K00<30`000000@0000000
0`0c<c<08R8R00000009000000400`<30@0K6a/0000305YJFP071`L0000004`000000@071`L105YJ
FP0000<0Z:RX04a<C00410@0B`00000100@41040C4a<0@2XZ:P000050>3Ph02XZ:P0FUYJ02DU9@0F
5QH00P0B4Q8700820QL04Q8B00<051@D0000000000007@0000000`0;2`/04Q8B018B4P03018B4P05
00/;2`0F5QH09BDU05YJFP2XZ:P00@3Ph>000001\
\>"], "Graphics",
  CellFrame->{{0, 0}, {2, 0}},
  ShowCellBracket->False,
  CellMargins->{{4, 0}, {0, 6}},
  Active -> False,
  Editable -> False,
  Selectable -> False,
  Evaluatable->False,
  CellFrameMargins->False,
  ImageSize->{81, 19},
  ImageMargins->{{0, 0}, {0, 0}},
  ImageRegion->{{0, 1}, {0, 1}},
  CellTags->"ShortTab"]



Resource["ClockIcon"] =
Cell[GraphicsData["Bitmap", "\<\
CF5dJ6E]HGAYHf4PAg9QL6QYHg<PAVmbKF5d0`40000c0000<b000`400?l00000o`00003o4`3IfMT3
0=SHf0L0emOG0`3Hf=PC0=WIf@003`3IfMT20=SHf080emOG0P3Fe]H00`3EeMD0e=CD0=?Cd`040=?C
d`040=CDe03EeMD0e]KF0=KFeP80emOG0P3Hf=P?0=WIf@003@3IfMT20=SHf0060=OGe`3Fe]H0eMGE
0=?Cd`3AdM40d=3@103>c/h00`3<c<`0c/k>0<k>cP020<k>cP060=3@d03AdM40dm?C0=GEe@3Fe]H0
emOG0P3Hf=P=0=WIf@002`3IfMT20=SHf0060=OGe`3Fe]H0e=CD0=7Ad@3@d=00c/k>0P3:b/X00`37
alL0a<C40<C4a0050<C4a0090<O7a`3:b/X0b/[:0<k>cP3@d=00dM7A0=CDe03Fe]H0emOG0080f=SH
2`3IfMT000X0fMWI00X0f=SH0=OGe`3Fe]H0e=CD0=7Ad@3>c/h0b/[:0<S8b034a<@0`/;20P2n_[h7
0;Zj^P80_[jn00X0`/;20<C4a038b<P0b/[:0<k>cP3AdM40e=CD0=KFeP3GemL0f=SH2P3IfMT000T0
fMWI00d0f=SH0=KFeP3EeMD0dm?C0<k>cP3:b/X0alO70<C4a02n_[h0^[Zj0;Rh^02e]KD0/k>c00<0
/;2`0140[Zj^0;2`/02`/;00/;2`0;>c/`2e]KD0^;Rh0;Zj^P2n_[h0a<C40<O7a`3:b/X0c/k>0=?C
d`3EeMD0e]KF0=SHf0090=WIf@001`3IfMT20=SHf00:0=KFeP3De=@0d=3@0<k>cP38b<P0a<C40;jn
_P2j^[X0]KFe0;>c/`80[Zj^00@0R8b?06I_M01:Eel0>DYE0P0G=4P03`0;;D<05cA802A5G00aEg40
F6M`06f4U02CUiX0^[Zj0;jn_P32`/80b<S80<k>cP3@d=00e=CD0=KFeP020=SHf0L0fMWI00060=WI
f@0C0=SHf03GemL0e]KF0=CDe03@d=00b/[:0<C4a02n_[h0^[Zj0;Fe]@2`/;00[Zj^08jAU01HIg00
<T9=00hQ;P0993D02be300`bB`0200hjE`804D=T0P0AAFP30153I00;03]RN`1]Q9@0WZJ[0;jn_P34
a<@0b/[:0=3@d03De=@0e]KF0=OGe`3Hf=P01P3IfMT000H0fMWI0180f=SH0=KFeP3Cdm<0c/k>0<[:
bP34a<@0_[jn0;Rh^02c/k<0[Zj^08jAU01AGF@07Bhj00TT=@0;;D<03SYG0153I00BBFh201ABN`03
09Fba@0FFHH05UV600805UV60P0EEX4301ABN`09035GL@1]Q9@0XZfe0<C4a03:b/X0c/k>0=?Cd`3F
e]H0f=SH00H0fMWI00050=WIf@0F0=SHf03Fe]H0e=CD0<k>cP3:b/X0a<C40;bl_02e]KD0[Zj^0:^[
Z`1VKg@0:SU300HK:00;;D<03SYG0199KP0DDW/05UV601MNSP0HHI4066>D09Nid0<06FJH01006FBG
01QSU00HHi@0666A01MNSP0GGHX05UV601EFP@0fJhd0M9Ra0<;2`P3:b/X0d=3@0=CDe03Fe]H0f=SH
1@3IfMT000@0fMWI01<0f=SH0=KFeP3De=@0d=3@0<[:bP34a<@0^[Zj0;Fe]@2^[Zh0TiNJ04YGG`06
6bP02B@e00hjE`0BBFh05EJ101MNSP0HHi@06FFH00/06FJI00@06FJH01UTU`0HHi@05f2?0P0GGXh0
1`1OR:<0ZKFl0<[:bP3@d=00e=CD0=KFeP3Hf=P0103IfMT000<0fMWI0180f=SH0=OGe`3EeMD0d=3@
0<[:bP34a<@0_;bl0;Fe]@2^[Zh0R8b?03U:E@066bP02be30153I00EEX405ej>01QSU00IIYP@01UV
V@0;01UVV00IIIP06FBG01UUV01AQ:H0ZKFl0<[:bP3@d=00eMGE0=OGe`3Hf=P00`3IfMT000<0fMWI
0100f=SH0=KFeP3Cdm<0c/k>0<C4a02n_[h0]KFe0:j^[P28S8l0<T9=00HK:00<<T/04TU^01IIQP0H
Hi@06FFH4`0IIYT02P0JIiT06fNJ01eYV`0MJI/0DHBV0:Ve_03>c/h0dm?C0=KFeP3Hf=P30=WIf@00
0P3IfMT0403Hf=P0emOG0=CDe03>c/h0b<S80;jn_P2h^;P0[Zj^09>GVP0iBUD01Q/X00`bB`0CCG<0
5ef:01QSU00IIYPE01UVV@0:01YWV@0MJI/08fbM02=/W@1IRjd0/kg30<k>cP3De=@0emOG0=SHf080
fMWI00020=WIf@0?0=SHf03Fe]H0dM7A0<[:bP34a<@0^[Zj0;>c/`2[Zj/0BUMO00HK:00<<T/04dec
01MMRP0II9L06FJH01L06FJI00T06fNJ01eYV`0WKil0;7>Q07BH/@3:b/X0dM7A0=KFeP3Hf=P00P3I
fMT0000?0=WIf@3Hf=P0emOG0=CDe03>c/h0alO70;jn_P2e]KD0[Zj^06I_M0066bP02be30199KP0G
GHX06FBG01X06FJI00T06fNJ02=/W@0/Lj40=GVU092[_@3>c/h0e=CD0=OGe`3Hf=P00@3IfMT0000?
0=WIf@3Hf=P0e]KF0=7Ad@3:b/X0a<C40;Zj^P2`/;00SY6D02Xi@`0993D04D=T01IIQP0HHi@06FJH
01X06FJI00T06VNI01eYV`0WKil0=GVU05V;[@2c_L<0dM7A0=KFeP3Hf=P00@3IfMT0000>0=WIf@3G
emL0eMGE0=3@d038b<P0_[jn0;Fe]@2^[Zh0DEeT00HK:00<<T/0559k01QQT@0IIYPL01UVV@0801YW
V@0SK9d0<7JS03amZ@22XkT0d=3@0=GEe@3GemL10=WIf@0000d0fMWI0=OGe`3Cdm<0c/k>0<C4a02j
^[X0/k>c08jAU00M;SX02B@e0199KP0GGXh06FFH01h06FJI00L07FVK02UaX00lOJT0FHnc0;C2b`3C
dm<0emOG0040fMWI0000303Hf=P0e]KF0=7Ad@3:b/X0`/;20;Rh^02^[Zh0F6M`00HK:00>>UL05EJ1
01QSU1l06FJI00L06VNI02=/W@0eNJD0BHJ^092[_@3AdM40e]KF0040f=SH0000303Hf=P0e]KF0=3@
d03:b/X0_[jn0;Fe]@2^[Zh0<T9=00TT=@0AAFP05ej>01UVV2006FJI00H07FVK02idXP13PZ`0Jibm
0=3@d03Fe]H10=SHf00000/0f=SH0=GEe@3>c/h0alO70;jn_P2c/k<0R8b?00HK:00;;D<0559k01QS
U00Q01UVV@0601YWV@0YLJ00@h:/05>=/`2d`//0eMGE0@3Hf=P0000;0=OGe`3De=@0c/k>0<C4a02j
^[X0/;2`06I_M0066bP03SYG01IIQP0IIIP08P0IIYT01@0SK9d0?7fY05>=/`2O]L@0e=CD0040emOG
00002P3GemL0dm?C0<k>cP34a<@0^[Zj0;2`/01:Eel02B@e0153I00GGXhS01UVV@0502=/W@0jNZD0
Dhfc08Z]a03Cdm<00@3GemL0000:0=OGe`3Cdm<0c/k>0<C4a02j^[X0/;2`0392C@0993D04TU^01QQ
TB<06FJI00D07FVK03EiY@1CSK<0OJK20=?Cd`010=OGe`0000X0emOG0=?Cd`3<c<`0a<C40;Zj^P2^
[Zh07Bhj00/]@`0CCG<066>D400IIYT00`2bc=d0g^Wa0=7Pj`0?0=7Pj`0601eYV`2G^M00Xl;F05>=
/`1bXL00dm?C0@3GemL0000:0=OGe`3Cdm<0c/k>0<C4a02j^[X0/;2`00hQ;P0<<T/0UK;509Nid0l0
6FJI00@08fbM0?ooo`3oool0`=GS3`2S`]H01P0WKil06VNI031fX`1CSK<0Jibm0=?Cd`40emOG0000
2P3GemL0dm?C0<k>cP34a<@0^[Zj0;2`/0066bP0339;01ABN`0IIIP?01UVV@0402idXP3oool0oooo
0<3Eha006FJI00D06VNI031fX`1CSK<0Jibm0=?Cd`010=OGe`0000X0emOG0=CDe03>c/h0a<C40;Zj
^P2`/;003R4^00`bB`0EEX406FFH3`0IIYT0100^M:80oooo0?ooo`30eN<@01UVV@0501]WVP0`MZ<0
FHnc07RR_`3De=@00@3GemL0000:0=SHf03EeMD0c/k>0<O7a`2n_[h0/k>c01d^>P0<<T/0559k01UT
U`l06FJI00@0;WBR0?ooo`3oool0`=GS400IIYT01@0MJI/0=GVU05^B]P1mY/80eMGE0040f=SH0000
2P3Hf=P0e]KF0=3@d03:b/X0_[jn0;Fe]@0b@Td0339;01ABN`0HHi@?01UVV@0402idXP3oool0oooo
0<3Eha006FJI00D07FVK03amZ@1QU[T0RZg40=KFeP010=SHf00000X0f=SH0=KFeP3AdM40b/[:0<;2
`P2h^;P0BUMO00`bB`0CCG<0666A3`0IIYT0100^M:80oooo0?ooo`2[amX@01UVV@0502M_W`13PZ`0
Jibm09VfbP3Fe]H00@3Hf=P0000;0=WIf@3GemL0dm?C0<k>cP34a<@0_;bl06=gP`0;;D<04DEX01MN
SP0IIYP03P0IIYT0100^M:80oooo0?ooo`2[amX?01UVV@0601YWV@0YLJ00CXVa07:Q`02^`Ld0emOG
0@3IfMT0000;0=WIf@3GemL0eMGE0=3@d038b<P0_[jn09>GVP0;;D<04D=T01IIQP0IIIP03P0IIYT0
100^M:80oooo0?ooo`2[amX?01UVV@0601eYV`0eNJD0Fi:f07RR_`34c=80emOG0@3IfMT0000;0=WI
f@3Hf=P0e]KF0=7Ad@3:b/X0a<C40;Zj^P0iBUD03SYG01ABN`0HHi@03P0IIYT0100^M:80oooo0?oo
o`2[amX>01UVV@0701YWV@0SK9d0@h:/06^L_@2A//P0e]KF0=SHf0010=WIf@0000`0fMWI0=SHf03G
emL0e=CD0<k>cP37alL0_[jn06=gP`0>>UL04dec01MNSP0IIIP=01UVV@0402idXP3oool0oooo0:_7
fPh06FJI00L07FVK031fX`1CSK<0LZ700:k1c@3GemL0f=SH0040fMWI00020=WIf@0:0=SHf03Fe]H0
dM7A0<[:bP34a<@0WZJ[02A5G00AAFP05UV601QSU0d06FJI00@0;WBR0?ooo`3oool0ZlOJ3@0IIYT0
1`0JIiT08fbM04>2[01VV;T0RZg40<C<dP3Hf=P00P3IfMT00080fMWI00/0f=SH0=OGe`3De=@0c/k>
0<S8b02n_[h0HgN300hjE`0DDW/05ej>01UUV00<01UVV@0402idXP3oool0oooo0:_7fPd06FJI00L0
7FVK031fX`1CSK<0LZ700:Vnc@3GemL0f=SH0080fMWI00030=WIf@0;0=SHf03Fe]H0dm?C0<k>cP34
a<@0WZJ[035GL@0BBFh05UV601QSU00IIYP02`0IIYT0100^M:80oooo0?ooo`2[amX<01UVV@0701eY
V`0YLJ00BHJ^06^L_@2A//P0a<cB0=SHf0030=WIf@000`3IfMT0303Hf=P0emOG0=GEe@3@d=00b/[:
0<C4a01]Q9@04DEX01ABN`0GGXh06FBG01UVV0X06FJI00@0;WBR0?ooo`3oool0ZlOJ2`0IIYT0200K
IiX09fnO03amZ@1QU[T0OJK20;C2b`3GemL0f=SH0`3IfMT000@0fMWI00`0f=SH0=KFeP3De=@0d=3@
0<[:bP32`/80HgN301==L`0EEX40666A01UTU`0IIYP901UVV@0402idXP3oool0oooo0:_7fPX06FJI
00P06fNJ02=/W@0lOJT0Fi:f07RR_`2S^l/0e]KF0=SHf0@0fMWI00050=WIf@0<0=SHf03Fe]H0e=CD
0<k>cP3:b/X0ZKFl04]cSP0DDW/05UV601QQT@0II9L06FJH200IIYT0100^M:80oooo0?ooo`2[amX9
01UVV@0801eYV`0WKil0?7fY05V?/`1bXL00VkS;0<_@e03Hf=P50=WIf@001P3IfMT02`3Hf=P0e]KF
0=?Cd`3>c/h0b/[:0::]]@1;Lhh05EJ101MNSP0HHi@06FFH00T06FJI00<0CXVa0:_7fP1>RK401`0I
IYT02@0JIiT07FVK02UaX00lOJT0FHnc07:Q`02I]/X0a<cB0=SHf0060=WIf@001P3IfMT02`3Hf=P0
emOG0=KFeP3De=@0d=3@0<[:bP2R[KD0Bg>>01MMRP0HHI406FFH01406FJI00/06VNI01eYV`0SK9d0
<7JS04>2[01KT[H0LZ7009VfbP34c=80emOG0=SHf0060=WIf@001`3IfMT20=SHf0080=KFeP3De=@0
d=3@0<k>cP2c_L<0HHRR01MNSP0II9L201]WVP806VNI2`0IIYT02P0JIiT07FVK02=/W@0/Lj40?7fY
04j9/@1QU[T0LZ700:>kb`3;d=@20=SHf0L0fMWI00090=WIf@090=SHf03Fe]H0eMGE0=?Cd`3>c/h0
b/[:08:S^@0jNZD07FVK00<08fbM0`0MJI/01@0KIiX06VNI01YWV@0JIiT06fNJ00807FVK00/08fbM
02M_W`0`MZ<0?7fY04V6[P1KT[H0Jibm08Z]a02^`Ld0e]KF0=SHf0090=WIf@002P3IfMT02P3Hf=P0
emOG0=KFeP3De=@0dM7A0<k>cP2^_<D0M9Ra03YjY@0WKil202UaX00C02acX@0YLJ00:G6P02M_W`2S
`]H0:G6P02acX@0`MZ<0=GVU03amZ@13PZ`0CXVa05^B]P1VV;T0OJK20:>kb`34c=80emOG0=SHf00:
0=WIf@002`3IfMT20=SHf00:0=OGe`3Fe]H0e=CD0=7Ad@3@d=00[[c508:S^@1IRjd0<7JS03EiY@80
>WZU00d0?7fY0:_7fP13PZ`0@h:/04V6[P1>RK40Dhfc05^B]P1VV;T0RZg40:>kb`34c=80emOG0080
f=SH2`3IfMT000d0fMWI0P3Hf=P05@3GemL0e]KF0=GEe@3Cdm<0dM7A0=3@d02^_<D0T:^m07RR_`1V
V;T0FHnc05>=/`1>RK40Fi:f06JH^@1hX[l0RZg40:>kb`34c=80e]KF0=OGe`020=SHf0d0fMWI000?
0=WIf@80f=SH0P3GemL20=KFeP030=GEe@3De=@0dm?C00@0dm?C00@0e=CD0=GEe@3Fe]H0e]KF0P3G
emL20=SHf0l0fMWI000C0=WIf@<0f=SH1`3GemL30=SHf1<0fMWI0000\
\>"], "Graphics",
  Active -> False,
  Editable -> False,
  Selectable -> False,
  Evaluatable->False,
  ImageSize->{51, 51},
  ImageMargins->{{0, 0}, {0, 0}},
  ImageRegion->{{0, 1}, {0, 1}},
  CellTags -> "ClockIcon"]


Resource["WarningIcon"] =
Cell[GraphicsData["Bitmap", "\<\
CF5dJ6E]HGAYHf4PAg9QL6QYHg<PAVmbKF5d0`40000c0000<b000`400?l00000o`00003o4`3IfMT3
0=SHf0L0emOG0`3Hf=PC0=WIf@003`3IfMT20=SHf080emOG0P3Fe]H00`3EeMD0e=CD0=?Cd`040=?C
d`040=CDe03EeMD0e]KF0=KFeP80emOG0P3Hf=P?0=WIf@003@3IfMT20=SHf0060=OGe`3Fe]H0eMGE
0=?Cd`3AdM40d=3@103>c/h00`3<c<`0c/k>0<k>cP020<k>cP060=3@d03AdM40dm?C0=GEe@3Fe]H0
emOG0P3Hf=P=0=WIf@002`3IfMT20=SHf0060=OGe`3Fe]H0e=CD0=7Ad@3@d=00c/k>0P3:b/X00`37
alL0a<C40<C4a0050<C4a0090<O7a`3:b/X0b/[:0<k>cP3@d=00dM7A0=CDe03Fe]H0emOG0080f=SH
2`3IfMT000X0fMWI00X0f=SH0=OGe`3Fe]H0e=CD0=7Ad@3>c/h0b/[:0<S8b034a<@0`/;20P2n_[h7
0;Zj^P80_[jn00X0`/;20<C4a038b<P0b/[:0<k>cP3AdM40e=CD0=KFeP3GemL0f=SH2P3IfMT000T0
fMWI00d0f=SH0=KFeP3EeMD0dm?C0<k>cP3:b/X0alO70<C4a02n_[h0^[Zj0;Rh^02e]KD0/k>c00<0
/;2`0140[Zj^0;2`/02`/;00/;2`0;>c/`2e]KD0^;Rh0;Zj^P2n_[h0a<C40<O7a`3:b/X0c/k>0=?C
d`3EeMD0e]KF0=SHf0090=WIf@001`3IfMT20=SHf00:0=KFeP3De=@0d=3@0<k>cP38b<P0a<C40;jn
_P2j^[X0]KFe0;>c/`80[Zj^00@0R8b?06I_M01:Eel0>DYE0P0G=4P03`0;;D<05cA802A5G00aEg40
F6M`06f4U02CUiX0^;Rh0;jn_P32`/80b<S80<k>cP3@d=00e=CD0=KFeP020=SHf0L0fMWI00060=WI
f@0C0=SHf03GemL0e]KF0=CDe03@d=00b/[:0<C4a02n_[h0^[Zj0;Fe]@2`/;00[Zj^08jAU01HIg00
<T9=00hQ;P0993D02be300`bB`0200hjE`804D=T0P0AAFP30153I00;03UWQ01]Q9@0WjR^0;jn_P34
a<@0b/[:0=3@d03De=@0e]KF0=OGe`3Hf=P01P3IfMT000H0fMWI0180f=SH0=KFeP3Cdm<0c/k>0<[:
bP34a<@0_[jn0;Rh^02c/k<0[Zj^08jAU01AGF@07Bhj00TT=@0;;D<03SYG0153I00BBFh201ABN`03
01EFPP0FFHH05UV600805UV60P0EEX8301ABN`09035GL@1]Q9@0XZfe0<C4a03:b/X0c/k>0=?Cd`3F
e]H0f=SH00H0fMWI00050=WIf@0F0=SHf03Fe]H0e=CD0<k>cP3:b/X0a<C40;bl_02e]KD0[Zj^0:^[
Z`1VKg@0:SU300HK:00;;D<03SYG0199KP0DDW/05UV601MOSP0HHI4066>D01UTU`<06FJH01006FBG
01QSU00HHi@0666A01MOSP0GGH/05UV601EFPP0iIh@0M9Ra0<;2`P3:b/X0c/k>0=CDe03Fe]H0f=SH
1@3IfMT000@0fMWI01<0f=SH0=KFeP3De=@0d=3@0<[:bP34a<@0^[Zj0;Fe]@2^[Zh0TiNJ04YGG`06
6bP02B@e00hjE`0BBFh05EJ201MOSP0HHi@06FFH00/06FJI00<06FJH01UTU`0HHi@00`0GGhh01`1P
R:<0ZKFl0<[:bP3@d=00e=CD0=KFeP3Hf=P0103IfMT000<0fMWI0180f=SH0=OGe`3EeMD0d=3@0<[:
bP34a<@0_;bl0;Fe]@2^[Zh0R8b?03U:E@066bP02be30153I00DDW/05en>01QSU00IIYP@01UVV@0;
01UVV00IIIP06FBG01UUV01CQZP0ZKFl0<[:bP3@d=00eMGE0=OGe`3Hf=P00`3IfMT000<0fMWI0100
f=SH0=KFeP3Cdm<0c/k>0<C4a02n_[h0]KFe0:j^[P28S8l0<T9=00HK:00<<T/04TU^01IIQP0HHi@0
6FFH4`0IIYT02P0JIiT06fNJ01eYV`0MJI/0DhJX0:Ve_03>c/h0dm?C0=KFeP3Hf=P30=WIf@000P3I
fMT0403Hf=P0emOG0=CDe03>c/h0b<S80;jn_P2h^;P0[Zj^09>GVP0iBUD01Q/X00`bB`0CCG<05ef;
01QSU00IIYPE01UVV@0:01YWV@0MJI/08fbM02=/W@1JSJl0/kg30<k>cP3De=@0emOG0=SHf080fMWI
00020=WIf@0?0=SHf03Fe]H0dM7A0<[:bP34a<@0^[Zj0;>c/`2[Zj/0BUMO00HK:00<<T/04dec01MM
R`0II9L06FJH01L06FJI00T06fNJ01eYV`0SK9d0:W:Q07BH/@3:b/X0dM7A0=KFeP3Hf=P00P3IfMT0
000?0=WIf@3Hf=P0emOG0=CDe03>c/h0alO70;jn_P2e]KD0[Zj^06I_M0066bP02be30199KP0GGH/0
6FBG01X06FJI00T06fNJ02=/W@0ZLZ40=GVU092[_@3>c/h0e=CD0=OGe`3Hf=P00@3IfMT0000?0=WI
f@3Hf=P0e]KF0=7Ad@3:b/X0a<C40;Zj^P2`/;00SY6D02Xi@`0993D04D=T01IIQP0HHi@06FJH00X0
6FJI00<0Q:g90?ooo`3oool00P3oool;01UVV@0901YWV@0MJI/09g2O03EiY@1JSJl0/kg30=7Ad@3F
e]H0f=SH0040fMWI00003P3IfMT0emOG0=GEe@3@d=00b<S80;jn_P2e]KD0[Zj^055MI0066bP0339;
01ABN`0HHI406FJH2`0IIYT00`24[LT0oooo0?ooo`020?ooo``06FJI00P06VNI02=/W@0`MJ<0?7fY
08BU^P3@d=00eMGE0=OGe`40fMWI00003@3IfMT0emOG0=?Cd`3>c/h0a<C40;Zj^P2c/k<0SY6D01d^
>P0993D04TU^01MMR`0IIIP0300IIYT00`24[LT0oooo0?ooo`020?ooo`d06FJI00L07FVK02YbX@0l
OJT0FXf_0;C2b`3Cdm<0emOG0040fMWI0000303Hf=P0e]KF0=7Ad@3:b/X0`/;20;Rh^02^[Zh0F6M`
00HK:00>>UL05EJ201QSU0d06FJI00<0Q:g90?ooo`3oool00P3oool=01UVV@0701YWV@0SK9d0=GVU
04V6[P2@Zkd0dM7A0=KFeP010=SHf00000`0f=SH0=KFeP3@d=00b/[:0;jn_P2e]KD0[Zj^0392C@09
93D04DEX01MMR`0IIYP=01UVV@0305J?]024[LT0Q:g90080Q:g93P0IIYT01P0MJI/0<7FS04>2[01a
X;l0d=3@0=KFeP40f=SH00002`3Hf=P0eMGE0<k>cP37alL0_[jn0;>c/`28S8l01Q/X00/]@`0DDW/0
66>D02406FJI00H06VNI02M`W`13PZ`0E8fc0;C2b`3EeMD10=SHf00000/0emOG0=CDe03>c/h0a<C4
0;Zj^P2`/;00IVmd00HK:00>>UL05UV601UUV00?01UVV@0307FS`P32e^@0U;S?01006FJI00D08fbM
03amZ@1DSK<0WkG40=CDe0010=OGe`0000X0emOG0=?Cd`3>c/h0a<C40;Zj^P2`/;00BUMO00TT=@0A
@f@05ef;400IIYT0102bc=d0oooo0>S`m@0KIiX?01UVV@0502=/W@0iNZH0E8fc08^]a03Cdm<00@3G
emL0000:0=OGe`3Cdm<0c/k>0<C4a02j^[X0/;2`0392C@0993D04TU^01QQTA006FJI00<0`]KT0?oo
o`3oool0400IIYT01@0MJI/0=GVU05></P1oY[l0dm?C0040emOG00002P3GemL0dm?C0<c<c034a<@0
^[Zj0:j^[P0M;SX02be301==L`0HHi@@01UVV@040>S`m@3oool0oooo03UjYPl06FJI00D06fNJ031e
X`1@R[40LJ2o0=?Cd`010=OGe`0000X0emOG0=?Cd`3>c/h0a<C40;Zj^P2`/;003R4^00`bB`0DDW/0
6FBG400IIYT30?ooo`0305J?]00IIYT06FJI00d06FJI00D06VNI031eX`1CS;80LJ2o0=?Cd`010=OG
e`0000X0emOG0=?Cd`3>c/h0a<C40;Zj^P2`/;001Q/X00`bB`0EEX806FFH3`0IIYT01@0iNZH0oooo
0?ooo`3oool0MJ?200l06FJI00D06VNI031eX`1DSK<0JYbm0=?Cd`010=OGe`0000X0emOG0=CDe03>
c/h0a<C40;Zj^P2`/;003R4^00`bB`0DDW/06FFH3`0IIYT01@1FSk@0oooo0?ooo`3oool0Q:g900l0
6FJI00D06fNJ031eX`1FSk@0MJ?20=CDe0010=OGe`0000X0f=SH0=GEe@3>c/h0alO70;jn_P2c/k<0
7Bhj00`bB`0DDW/06FBG3`0IIYT01@1UV;X0oooo0?ooo`3oool0U;S?00l06FJI00D07FVK03UjYP1K
T[H0O:K30=GEe@010=SHf00000X0f=SH0=KFeP3@d=00b/[:0;jn_P2e]KD0<T9=00`bB`0DDW/066>D
3`0IIYT01@24[LT0oooo0?ooo`3oool0`]KT00l06FJI00D07FVK03amZ@1QU[T0Rjg40=KFeP010=SH
f00000X0f=SH0=KFeP3AdM40b/[:0<;2`P2h^;P0BUMO00`bB`0CCG<0666A3`0IIYT01@2D^<l0oooo
0?ooo`3oool0gMgM00l06FJI00D08fbM04>2[01ZW;d0VKK:0=KFeP010=SHf00000/0fMWI0=OGe`3C
dm<0c/k>0<C4a02l_;`0HgN300/]@`0AAFP05ef;01UVV00>01UVV@030<;Fi03oool0oooo0080oooo
3P0IIYT01P0JIiT0:W:Q04f9/01aX;l0[L3=0=OGe`40fMWI00002`3IfMT0emOG0=GEe@3@d=00b<S8
0;jn_P2CUiX02be30153I00FFHH06FFH00h06FJI00<0`]KT0?ooo`3oool00P3oool>01UVV@0601eY
V`0eNJD0Fi:f07FS`P34c=80emOG0@3IfMT0000;0=WIf@3Hf=P0e]KF0=7Ad@3:b/X0a<C40;Zj^P0i
BUD03SYG01ABN`0HHi@03P0IIYT00`32e^@0oooo0?ooo`020?ooo`d06FJI00L06VNI02=/W@13PZ`0
JYbm09:bb03Fe]H0f=SH0040fMWI0000303IfMT0f=SH0=OGe`3De=@0c/k>0<O7a`2n_[h0HgN300hj
E`0CCG<05en>01UUV0d06FJI00<0`]KT0?ooo`3oool00P3oool=01UVV@0701eYV`0`MJ<0Dhbb07FS
`P2]`<d0emOG0=SHf0010=WIf@000P3IfMT02P3Hf=P0e]KF0=7Ad@3:b/X0a<C409bSY`0TAE`04DEX
01IIQP0HHi@=01UVV@030<;Fi03oool0oooo0080oooo300IIYT01`0JIiT08fbM04>2[01UV;X0Rjg4
0<C<dP3Hf=P00P3IfMT00080fMWI00/0f=SH0=OGe`3De=@0c/k>0<S8b02n_[h0HgN300hjE`0DDW/0
5en>01UUV00<01UVV@030<;Fi03oool0oooo0080oooo300IIYT01`0MJI/0<7FS05B=/`1eXl80[L3=
0=OGe`3Hf=P00P3IfMT000<0fMWI00/0f=SH0=KFeP3Cdm<0c/k>0<C4a02OZ:h0<EMa0199KP0FFHH0
66>D01UVV00;01UVV@0307FS`P24[LT0Q:g90080Q:g92`0IIYT01`0MJI/0:W:Q04V6[P1ZW;d0Sk;8
0<C<dP3Hf=P00`3IfMT000<0fMWI00`0f=SH0=OGe`3EeMD0d=3@0<[:bP34a<@0KHBD0155J00DDW/0
5ef;01UTU`0IIYPI01UVV@0801]WVP0WL9l0?7fY066F^@1lY/<0]<;;0=OGe`3Hf=P30=WIf@00103I
fMT0303Hf=P0e]KF0=CDe03@d=00b/[:0<;2`P1SMh<04TU^01EFPP0HHI406FBG01UVV1L06FJI00P0
6fNJ02=/W@0lOJT0Fi:f07FS`P2S^l/0e]KF0=SHf0@0fMWI00050=WIf@0<0=SHf03Fe]H0e=CD0<k>
cP3:b/X0ZKFl04]cSP0DDW/05UV601QQT@0II9L06FJH5@0IIYT0200MJI/09g2O03amZ@1FSk@0LJ2o
09^hb`3;d=@0f=SH1@3IfMT000H0fMWI00/0f=SH0=KFeP3Cdm<0c/k>0<[:bP2R[KD0Bg>>01EFPP0G
GH/066>D01UUV00C01UVV@0901YWV@0MJI/0:W:Q03amZ@1FSk@0LJ2o09VfbP34c=80f=SH00H0fMWI
00060=WIf@0;0=SHf03GemL0e]KF0=CDe03@d=00b/[:0::]]@1;Lhh05ef;01QQT@0IIIP04@0IIYT0
2`0JIiT07FVK02=/W@0`MJ<0@h:/05^B]P1aX;l0VKK:0<C<dP3GemL0f=SH00H0fMWI00070=WIf@80
f=SH00P0e]KF0=CDe03@d=00c/k>0;>m``1PR:<05en>01UUV0806fNJ0P0JIiT;01UVV@0:01YWV@0M
JI/08fbM02YbX@0lOJT0D8Za066F^@1eXl80Xk_;0<_@e080f=SH1`3IfMT000T0fMWI00T0f=SH0=KF
eP3EeMD0dm?C0<k>cP3:b/X0RIf[03UjYP0MJI/00`0SK9d301eYV`0501]WVP0JIiT06VNI01YWV@0K
IiX00P0MJI/02`0SK9d09g2O031eX`0lOJT0BHJ^05^B]P1ZW;d0Rjg40:g0c@3Fe]H0f=SH00T0fMWI
000:0=WIf@0:0=SHf03GemL0e]KF0=CDe03AdM40c/k>0:jla@1dV;40>GZV02M`W`D0:W:Q0P0WL9l2
02YbX@0<031eX`0eNJD0?7fY04>2[01@R[40Fi:f06FH^P1lY/<0Xk_;0<C<dP3GemL0f=SH2P3IfMT0
00/0fMWI0P3Hf=P02@3GemL0e]KF0=CDe03AdM40d=3@0:jla@1nY;d0FXf_03EiY@0203UjYP<0?7fY
0P13PZ`02@1=RK00D8Za05J?]01KT[H0IIRj08FZ`P2S^l/0a<cB0=OGe`020=SHf0/0fMWI000=0=WI
f@80f=SH01D0emOG0=KFeP3EeMD0dm?C0=7Ad@3@d=00[[c5092[_@1jXK/0IiJd05Z=[`1CS;80D8Za
05^B]P1UV;X0NZ6k08^]a02S^l/0a<cB0=KFeP3GemL00P3Hf=P=0=WIf@003`3IfMT20=SHf080emOG
0P3Fe]H00`3EeMD0e=CD0=?Cd`040=?Cd`040=CDe03EeMD0e]KF0=KFeP80emOG0P3Hf=P?0=WIf@00
4`3IfMT30=SHf0L0emOG0`3Hf=PC0=WIf@00\
\>"], "Graphics",
  Active -> False,
  Editable -> False,
  Selectable -> False,
  Evaluatable->False,
  ImageSize->{51, 51},
  ImageMargins->{{0, 0}, {0, 0}},
  ImageRegion->{{0, 1}, {0, 1}},
  CellTags -> "WarningIcon"]






End[]

EndPackage[]

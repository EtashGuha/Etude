(* :Context: AuthorTools`MakeProject` *)

(* :Author: Louis J. D'Andria *)

(* :Summary:
    This package allows the user to keep track of projects
    consisting of multiple notebook files in a easy to manage,
    centralized way.
*)

(* :Copyright: *)

(* :Package Version: $Revision: 1.25 $, $Date: 2015/11/24 19:54:52 $ *)

(* :Mathematica Version: 4.2 *)

(* :History:

*)

(* :Keywords:
    document, notebook, formatting 
*)

(* :Discussion:
    The result of using the project viewer dialog is to save a 
    file containing project data. This file is used as a source
    of information the next time the user wants to do something
    to the project, such as add or reorder notebooks.
*)

(* :Warning:
    
*)



BeginPackage["AuthorTools`MakeProject`", "AuthorTools`Common`"];


ProjectDataQ::usage = "ProjectDataQ[file] returns True if file exists and contains valid information about a project, and returns False otherwise.";

ProjectInformation::usage = "ProjectInformation[file] returns a list of information about the project which was in the given project file.";

WriteProjectData::usage = "WriteProjectData[file, data] rewrites file as a project data file containing the given data.";

ProjectName::usage = "ProjectName[file] returns the name of the project as stored in the project file.";

ProjectDirectory::usage = "ProjectDirectory[file] returns the location of the project files as specified by the project file.";

ProjectFiles::usage = "ProjectFiles[file] returns the list of notebooks that make up the project, as specified by the project file.";

ProjectFileLocation::usage = "ProjectFileLocation[nb] reads the location of the current project file as stored in the TaggingRules of nb.";

ProjectDialogQ::usage = "ProjectDialogQ[nb] returns True if indeed the specified notebook is the project dialog.";

ReadProjectDialog::usage = "ReadProjectDialog[nb] reads the data currently in the project dialog nb and returns a list of rules, just as would be stored in a project file.";


Begin["`Private`"];


(*
   Functions used within the MakeProject palette are of the form
   ProjectDialogFunction[token, args], where token specifies the
   behavior. Here's the list of valid tokens.

   ProjectDialogFunction["Load", nb, file] displays the content
   of the specified project file in the project dialog nb.
   ProjectDialogFunction["Load", nb] allows the user to choose
   the file with a standard file open dialog.

   ProjectDialogFunction["Save", nb] saves the project
   information currently in the project dialog nb to the value of
   ProjectFileLocation in the notebook's TaggingRules.

   ProjectDialogFunction["SaveAs", nb] saves the project
   information currently in the project dialog nb to a file of
   the users choosing.

   ProjectDialogFunction["Clear", nb] restores the content of the
   project dialog nb to their defaults.

   ProjectDialogFunction["ListFiles", nb] lists all notebooks
   files from the project directory in the project dialog nb.

   ProjectDialogFunction["ChooseDirectory", nb] lets the user
   choose the project directory using a standard file open
   dialog.
*)



(* Using the project data file *)


ProjectDataQ[projFile_String] := 
  FileType[projFile] === File && 
  StringMatchQ[projFile, "*.m"] &&
  {"Directory", "Files", "Name"} === 
      Union @ Map[First, Get[projFile]]

ProjectDataQ[___] := False

ProjectInformation[projFile_String?ProjectDataQ] := Get[projFile]

(*
   If ProjectInformation is passed a notebook which is not a
   project data file, then return the values returned by
   ProjectName, ProjectDirectory, and ProjectFiles
*)

ProjectInformation[nbFile_String] :=
{
  "Name"      -> ProjectName[nbFile],
  "Directory" -> ProjectDirectory[nbFile],
  "Files"     -> ProjectFiles[nbFile]
} /; FileType[nbFile] === File && StringMatchQ[nbFile, "*.nb"]


WriteProjectData::name = "The \"Name\" setting must be a string.";
WriteProjectData::dir = "The \"Directory\" setting must specify an existing directory.";
WriteProjectData::files = "The \"Files\" setting must specify a set of existing files in the indicated directory.";


validateProjectData[data_]:=
Block[{pn, pd, pf},
  {pn, pd, pf} = {"Name", "Directory", "Files"} /. data;
  
  If[!MatchQ[pn, _String],
    MessageDisplay[WriteProjectData::name];
    Abort[]
  ];  
  If[!MatchQ[pd, _String] || FileType[pd] =!= Directory,
    MessageDisplay[WriteProjectData::dir];
    Abort[]
  ];
  If[!MatchQ[pf, {__String}] || Union[
        FileType[ToFileName[{pd},#]]& /@ pf] =!= {File},
    MessageDisplay[WriteProjectData::files];
    Abort[]
  ];
]


WriteProjectData[projFile_, data_] :=
Block[{pn, pd, pf},
  validateProjectData[data];
  {pn, pd, pf} = {"Name", "Directory", "Files"} /. data;
  Put[{"Name" -> pn, "Directory" -> pd, "Files" -> pf}, projFile]
]

(*
WriteProjectData[projFile_, data_] :=
Block[{st, tmp},
  st = OpenWrite[projFile, PageWidth -> Infinity];
  WriteString[st,
    "{\n\"Name\" -> ",
    ToString["Name" /. data, InputForm],
    ",\n\"Directory\" -> ",
    ToString["Directory" /. data, InputForm],
    ",\n\"Files\" -> {\n",
    Sequence @@ 
      BoxForm`Intercalate[
        Map["  " <> ToString[#, InputForm] &, "Files" /. data], ",\n"],
    "\n}}"];
  Close[st]
]
*)


(*
  To make other coding easier, ProjectName, ProjectDirectory, and
  ProjectFiles should return sensible output for any file name,
  whether it is a project file or a notebook file. In the case of a
  notebook file, the name is should be the name of the notebook
  without the ".nb", the directory should be the notebook's parent
  folder, and the files should be the path to the notebook as a
  singleton list.
*)

ProjectName[projFile_] :=
If[ProjectDataQ[projFile],
  "Name" /. ProjectInformation[projFile],
  StringReplace[projFile,
    {DirectoryName[projFile]->"", $PathnameSeparator->"", ".nb"->""}]
] /; FileType[projFile] === File

ProjectDirectory[projFile_] :=
If[ProjectDataQ[projFile],
  "Directory" /. ProjectInformation[projFile],
  DirectoryName[projFile]
] /; FileType[projFile] === File

ProjectFiles[projFile_] :=
If[ProjectDataQ[projFile],
  "Files" /. ProjectInformation[projFile],
  {StringReplace[projFile, DirectoryName[projFile] -> ""]}
] /; FileType[projFile] === File



(* interface for creating the project data file *)


ProjectDataQ::noproj = "The file you selected is not a valid project file.";

ProjectDialogFunction["Load", nb_] :=
  ProjectDialogFunction["Load", nb, SystemDialogInput["FileOpen"]];


ProjectDialogFunction["Load", nb_, projFile_String] :=
Block[{pn, pd, pf, data},
  If[!ProjectDataQ[projFile],
    MessageDisplay[ProjectDataQ::noproj];
    Abort[]
  ];
  data = ProjectInformation[projFile];
  {pn, pd, pf} = {"Name", "Directory", "Files"} /. data;
  SetOptions[nb, TaggingRules -> {"ProjectFileLocation" -> projFile}];
  updateProjectDialog[nb, {pn, pd, pf}];
]


ProjectDialogFunction["ChooseDirectory", nb_] :=
Block[{file},
  file = SystemDialogInput["FileOpen"];
  If[Head[file] === String && FileType[DirectoryName[file]] === Directory,
    updateProjectDialog[nb, {Automatic, DirectoryName @ file, Automatic}];
  ]
]



ProjectDialogFunction["ListFiles", nb_] :=
Block[{dir},
  dir = "Directory" /. ReadProjectDialog[nb];
  If[FileType[dir] =!= Directory,
    updateProjectDialog[nb, 
      {Automatic, Automatic, {AuthorTools`Common`$Resource["Project", "Bad Directory"]}}]
    ,
    files = FileNames["*.nb", dir];
    (* Some drive formats store resource forks in files starting with ._ *)
    files = Select[files, !StringMatchQ[#, "*" <> $PathnameSeparator <> "._" <> "*"]&];
    If[files =!= {},
      files = StringReplace[files, {dir->"",$PathnameSeparator->""}]];
    updateProjectDialog[nb, {Automatic, Automatic, files}]
  ]
]




stringToList[str_] := 
StringJoin /@ DeleteCases[
  Split[Characters[str], #2 =!= "\n" && #2 =!= "\r" &], "\n" | "\r", Infinity]



listToString[{}] := $Resource["Project", "no files"];

listToString[{str__}] := 
  Flatten[{"\n", #}& /@ {str}]//Rest//StringJoin



ProjectDialogQ[nb_NotebookObject] :=
  !FreeQ[Options[nb, TaggingRules], "ProjectFileLocation"]

ProjectDialogQ[___] := False


ProjectFileLocation::save = "Please save your project data before performing a project action.";


ProjectDialogFunction["SaveWarning", nb_] :=
Block[{data1, data2},
  data1 = ReadProjectDialog @ nb;
  data2 = ProjectInformation @ ProjectFileLocation @ nb;
  If[data1 =!= data2,
    messageDialog[ProjectFileLocation::save];
    Abort[]
  ]
]


ProjectFileLocation[nb_?ProjectDialogQ] := 
 "ProjectFileLocation" /. (TaggingRules /. Options[nb, TaggingRules])
 

ReadProjectDialog[nb_] :=
With[{nbg = NotebookGet[nb]},
  {"Name" -> Cases[nbg,
      Cell[x_, ___, CellTags -> "ProjectName", ___] :> x, Infinity][[1]], 
   "Directory" -> Cases[nbg, 
      Cell[x_, ___, CellTags -> "ProjectDirectory", ___] :> x, Infinity][[1]], 
   "Files" -> stringToList[Cases[nbg, 
      Cell[x_, ___, CellTags -> "ProjectFiles", ___] :> x, Infinity][[1]]]
  }
]


ProjectDialogFunction["Save", nb_] :=
Block[{data, projFile},
  projFile = ProjectFileLocation[nb];
  If[projFile === "" || FileType[projFile] =!= File,
    ProjectDialogFunction["SaveAs", nb]
    ,
    data = ReadProjectDialog[nb];
    WriteProjectData[projFile, data]
  ]
]
  

ProjectDialogFunction["SaveAs", nb_] :=
Block[{data, projFile},
  data = ReadProjectDialog[nb];  
  validateProjectData[data];
  projFile = SystemDialogInput["FileSave"];
    
  If[Head[projFile]=!=String, Abort[]];
  If[StringTake[projFile,-2] =!=".m", projFile = projFile <> ".m"];
  
  SetOptions[nb, TaggingRules -> {"ProjectFileLocation" -> projFile}];
  WriteProjectData[projFile, data]
]


ProjectDialogFunction["Clear", nb_] :=
Block[{tlis},
  tlis = TaggingRules /. Options[nb, TaggingRules];
  SetOptions[nb, TaggingRules -> Join[
    {"ProjectFileLocation" -> ""},
    DeleteCases[tlis, _["ProjectFileLocation",_]]]
  ];
  updateProjectDialog[nb, {"", "", {""}}]
]


updateProjectDialog[nb_, {pn_, pd_, pf_}] :=
Block[{},
  If[pf =!= Automatic,
    NotebookFind[nb, "ProjectFiles", All, CellTags];
    SelectionMove[nb, All, CellContents];
    NotebookWrite[nb, listToString @ pf]
  ];
  If[pd =!= Automatic,
    NotebookFind[nb, "ProjectDirectory", All, CellTags];
    SelectionMove[nb, All, CellContents];
    NotebookWrite[nb, pd]
  ];
  If[pn =!= Automatic,
    NotebookFind[nb, "ProjectName", All, CellTags];
    SelectionMove[nb, All, CellContents];
    NotebookWrite[nb, pn]
  ];
  SelectionMove[nb, Before, Notebook];
]



End[];

EndPackage[];


Widget["WizardFrame", {

  "wizard" -> Widget["Wizard", {
    "title" -> "Text Import Wizard",
    "sideBarTitle" -> "Steps",
    
    "pages" -> {
  
    Widget["WizardPage", {
      "title" -> "Overview",
      "navigationNames" -> {"Back", "Next", "Cancel"},
      "content" -> Widget["WizardHTMLPanel", {
        "text" -> "This wizard will help you import data into <i>Mathematica</i> from delimited text files.<p>
          You can choose to provide source data for the import as either:<ol><li>External file<li>Pasted text into the wizard</ol><p>
          You will then be able to optionally modify conversion options used during the import."
          }],
      "sideBarContent" -> Widget["WizardHTMLPanel", {
          "text" -> "<b>1. Overview</b><p>2. Choose Source Type<p>3. Select Source Data<p>4. Import Options"
          }]
      }, Name -> "overviewPage"],
      
    Widget["WizardPage", {
      "title" -> "Source Data",
      "navigationNames" -> {"Back", "Next", "Cancel"},
      "content" -> Widget["Panel", {
        Widget["WizardHTMLPanel", {
          "text" -> "Choose how you would like to provide data for the import:"}],
        Widget["RadioButton", {"text" -> "External file", "selected" -> True,
          BindEvent["action", 
             SetPropertyValue[{"sourceChoicePage", "nextPage"}, WidgetReference["fileSourcePage"]] ]
           }, Name -> "fileSourceButton"],
        Widget["RadioButton", {"text" -> "Pasted text",
          BindEvent["action", 
             SetPropertyValue[{"sourceChoicePage", "nextPage"}, WidgetReference["textSourcePage"]] ]
           }, Name -> "textSourceButton"],
        Widget["ButtonGroup", {WidgetReference["fileSourceButton"], WidgetReference["textSourceButton"]}],
        WidgetFill[],
        Widget["WizardHTMLPanel", {
          "text" -> "After choosing your source type, choose 'Next' to select the data."}]
        }],
      "sideBarContent" -> Widget["WizardHTMLPanel", {
          "text" -> "1. Overview<p><b>2. Choose Source Type</b><p>3. Select Source Data<p>4. Import Options"
          }]
      }, Name -> "sourceChoicePage"],
      
    Widget["WizardPage", {
      "title" -> "File Source Data",
      "navigationNames" -> {"Back", "Next", "Cancel"},
      "previousPage" -> WidgetReference["sourceChoicePage"],
      BindEvent["pageWillActivate", 
        Script[
          ValidateFileSourcePage[];
          SetPropertyValue[{"fileSourcePage", "nextPage"}, WidgetReference["importOptionsPage"]];
          SetPropertyValue[{"importOptionsPage", "previousPage"}, WidgetReference["fileSourcePage"]];
          ]],
      "content" -> Widget["Panel", {
         Widget["WizardHTMLPanel", {
          "text" -> "Enter the path to the file of your source data or choose the 'Browse' button to find the file."}],
         WidgetSpace[10],
         Widget["TextField", {"text" -> "",
           PropertyValue[{"filenameTextField", "document"}, Name -> "filenameTextFieldDocument"],
           BindEvent[{"filenameTextFieldDocument", "document"},
              Script[ ValidateFileSourcePage[];]],
           BindEvent["action", 
             Script[ ValidateFileSourcePage[];]]
           }, Name -> "filenameTextField"],
         {Widget["Button", {"text" -> "Browse...",
             BindEvent["action", 
               Script[ BrowseFilename[]; ValidateFileSourcePage[]; ]]
             }], 
           WidgetFill[]},
         WidgetFill[],
         Widget["WizardHTMLPanel", {
          "text" -> "Once you have selected the file to use, choose 'Next' to continue the import."}]
          }],
      "sideBarContent" -> Widget["WizardHTMLPanel", {
          "text" -> "1. Overview<p>2. Choose Source Type<p><b>3. Select Source Data</b><p>4. Import Options"
          }]
      }, Name -> "fileSourcePage"],
      
    Widget["WizardPage", {
      "title" -> "Pasted Text Source Data",
      "navigationNames" -> {"Back", "Next", "Cancel"},
      "previousPage" -> WidgetReference["sourceChoicePage"],
      "nextPage" -> WidgetReference["importOptionsPage"],
      BindEvent["pageWillActivate", 
        Script[ 
          ValidateTextSourcePage[];
          SetPropertyValue[{"textSourcePage", "nextPage"}, WidgetReference["importOptionsPage"]];
          SetPropertyValue[{"importOptionsPage", "previousPage"}, WidgetReference["textSourcePage"]];
          ]],
      "content" -> Widget["Panel", {
         Widget["WizardHTMLPanel", {
          "text" -> "Paste the text of what you would like to import into the text area below."}],
         Widget["ScrollPane", {"viewportView" -> 
          Widget["TextArea", {
            "columns" -> 30, "rows" -> 12,
            PropertyValue[{"textSourceArea", "document"}, Name -> "textDocument"],
            BindEvent[{"textDocument", "document"},
              Script[ ValidateTextSourcePage[];]]
            }, Name -> "textSourceArea"]}],
         WidgetFill[],
         Widget["WizardHTMLPanel", {
          "text" -> "Once you are satisfied with the text above, choose 'Next' to continue the import."}]
          }],
      "sideBarContent" -> Widget["WizardHTMLPanel", {
          "text" -> "1. Overview<p>2. Choose Source Type<p><b>3. Select Source Data</b><p>4. Import Options"
          }]
      }, Name -> "textSourcePage"],
      
    Widget["WizardPage", {
      "title" -> "Import Options and Preview",
      "navigationNames" -> {"Back", "Finish", "Cancel"},
      BindEvent["pageWillActivate",
        Script[ UpdatePreviewTable[]; ]],
      "content" -> Widget["Panel", {
         Widget["WizardHTMLPanel", {"text" -> "Select from the following Import conversion options and preview or edit the import
           within the table below."}],
         WidgetSpace[5],
         WidgetGroup[{
          Widget["CheckBox", {"text" -> "Tab", "selected" -> True,
            BindEvent["action", Script[UpdatePreviewTable[]]]}, Name -> "tabCheckBox"], WidgetFill[],
          Widget["CheckBox", {"text" -> "Comma", "selected" -> True,
            BindEvent["action", Script[UpdatePreviewTable[]]]}, Name -> "commaCheckBox"], WidgetFill[],
          Widget["CheckBox", {"text" -> "Space", "selected" -> False,
            BindEvent["action", Script[UpdatePreviewTable[]]]}, Name -> "spaceCheckBox"], WidgetFill[],
          Widget["CheckBox", {"text" -> "Other", "selected" -> False,
            BindEvent["action", Script[UpdatePreviewTable[]]]}, Name -> "otherCheckBox"],
          Widget["TextField", {"text" -> "", "columns" -> 3,
            BindEvent["action", 
             Script[
                SetPropertyValue[{"otherCheckBox","selected"}, True];
                UpdatePreviewTable[];
               ] ]}, 
              Name -> "otherTextField", WidgetLayout -> {"Stretching" -> {None, None}}]
           }, WidgetLayout -> {"Border" -> {"Delimiters:",  {{5, 5}, {0, 2}}} }],
         WidgetSpace[10],
		 Widget["CheckBox", {"text" -> "Ignore empty lines", "selected" -> True,
            BindEvent["action", Script[UpdatePreviewTable[]]]}, Name -> "emptylineCheckBox"], WidgetFill[],
         Widget["Label", {"text" -> "Import Preview:"}],
         Widget["ScrollPane", {
           "viewportView" -> 
             Widget["Table", {
               "tableHeader" -> Null,
               PropertyValue[{"tablePreview", "model"}, Name -> "tablePreviewModel"]
               }, Name -> "tablePreview"]
            }, WidgetLayout -> {"Stretching" -> {Maximize, Maximize}}]
         }],
      "sideBarContent" -> Widget["WizardHTMLPanel", {
          "text" -> "1. Overview<p>2. Choose Source Type<p>3. Select Source Data<p><b>4. Import Options</b>"
          }]
      }, Name -> "importOptionsPage"]
      
    },
    
  Script[{}, ScriptSource -> "TextImportWizardScriptCode.m"]

  }, Name -> "wizard"]
  
}, Name -> "wizardFrame"]
  

Widget["WizardFrame", {
  "wizard" -> Widget["Wizard", {
    "title" -> "Showcase Wizard",
    "pages" -> {
  
    Widget["WizardPage", {
      "title" -> "Overview",
      "navigationNames" -> {"Back", "Next", "Cancel"},
      "content" -> Widget["WizardHTMLPanel", {
        "text" -> "This wizard showcases the available features of the <i>GUIKit`</i> Wizard widgets.
          <p>This page itself is an example of an Overview page choosing common navigation buttons for this
          type of wizard page."
          }]
      }],
      
    Widget["WizardPage", {
      "title" -> "System Requirements",
      "navigationNames" -> {"Back", "Next", "Cancel"},
      BindEvent["pageWillActivate", Script[ Print["Requirements Page will activate"]]],
      BindEvent["pageDidActivate", Script[ Print["Requirements Page did activate"]]],
      BindEvent["pageWillDeactivate", Script[ Print["Requirements Page will deactivate"]]],
      BindEvent["pageDidDeactivate", Script[ Print["Requirements Page did deactivate"]]],
      "content" -> Widget["WizardHTMLPanel", {
        "text" -> "Sometimes it is useful to begin a wizard with a requirements page if certain resources
          may need to exist for the user to effectively use the wizard and they can choose now to 
          exit and cancel the wizard."
          }]
      }],
      
    Widget["WizardPage", {
      "title" -> "Basic Page",
      "allowLast" -> False,
      BindEvent["pageWillActivate", Script[ Print["Basic Page will activate"]]],
      BindEvent["pageDidActivate", Script[ Print["Basic Page did activate"]]],
      BindEvent["pageWillDeactivate", Script[ Print["Basic Page will deactivate"]]],
      BindEvent["pageDidDeactivate", Script[ Print["Basic Page did deactivate"]]],
      "content" -> Widget["WizardHTMLPanel", {
        "text" -> "This is a basic user page showing the default navigation buttons if
           you do not supply a set that should be present for a page's actions."
           }]
      }],
      
    Widget["WizardPage", {
      "allowNext" -> False,
      "navigationNames" -> {"Back", "Next", "Cancel"},
      "title" -> "Controlling Navigation",
      "content" -> Widget["Panel", {
        Widget["WizardHTMLPanel", {
        "text" -> "This page demonstrates a simple control that determines when the Next Button is enabled allowing
          the user to proceed. Typically this would be set by some logic internal to the page's script code."
           }],
        Widget["CheckBox", {"text" -> "Allow Next",
          BindEvent["action",
            SetPropertyValue[{"controllingPage", "allowNext"},  
              PropertyValue[{"nextCheckBox", "selected"}] ]
            ]
          }, Name -> "nextCheckBox"],
        WidgetFill[]
        }]
      }, Name -> "controllingPage"],
      
    Widget["WizardPage", {
      "title" -> "Progress Page",
      "navigationNames" -> {"Back", "Next", "Cancel"},
      "content" -> Widget["Panel", {
        Widget["WizardHTMLPanel", {
        "text" -> "This is an example of a progress page that provides feedback from a potentiallly
          longer wizard action and allows the user to cancel or interrupt this longer action.<br><br>
          TODO: Need to finish this page showing a progress meter<br> interacting with the buttons"
          }]  
         }]
      }],
      
    Widget["WizardPage", {
      "title" -> "Custom SideBar Page",
      "navigationNames" -> {"Back", "Next", "Cancel"},
      
      "sideBarPaint" -> Widget["Color", InitialArguments -> {128, 128, 192}],
      "sideBarImage" -> Widget["ImageLabel", {
          "data" -> Script[ ExportString[ 
              Plot3D[ Sin[x y], {x, 0, Pi}, {y, 0, Pi}, 
                DisplayFunction -> Identity, Boxed -> False, Axes -> False, ImageSize -> {200, 200}], 
                 "GIF", "TransparentColor" -> GrayLevel[1] ]] }],
      
      "content" -> Widget["Panel", {
        Widget["WizardHTMLPanel", {
        "text" -> "This is an example of a page that uses its own sideBar color and image. 
         Individual pages can override the sideBar image and color or these properties can be set 
         on the wizard itself to be shared by all pages. You can also set sideBarTitle and sideBarContent properties on
         individual pages or the wizard to customize the look of the side bar even further."}]
         }]
      }],
      
    Widget["WizardPage", {
      "title" -> "Confirmation",
      "navigationNames" -> {"Back", "Next", "Finish", "Cancel"},
      "content" -> Widget["WizardHTMLPanel", {
        "text" -> "Sometimes it is useful to provide a Confirmation page summarizing the settings of the wizard
          properties and give the user one final view of the actions that will occur when the wizard finishes."
          }]
      }],
      
    Widget["WizardPage", {
      "title" -> "Summary",
      "navigationNames" -> {"Close"},
      "content" -> Widget["WizardHTMLPanel", {
        "text" -> "After a wizard has completed its actions a summary page could also be
          the last page of a wizard providing information of all that has occurred and links to further
          help the user can access."
          }]
      }, Name -> "summaryPage"]
  
    },
    
   Script[ 
     $wizardResult = Null;

     wizardFinished[] := (
       Print["Wizard finished"];
       $wizardResult = "Success";
       );

     wizardCanceled[] := (
       Print["Wizard was canceled"];
       $wizardResult = "Canceled";
       );

     wizardClosed[] := (
       Print["Wizard was closed"];
       );
     ],

   (* Setup a succcessful return result *)
   BindEvent["wizardFinished", Script[wizardFinished[]] ],

   (* Setup a canceled return result *)
   BindEvent["wizardCanceled", Script[wizardCanceled[]] ],

   (* You may not ever need to event off the close since the success/failure is
      determined by finish or cancel being called *)
   BindEvent["wizardClosed", Script[wizardClosed[]] ],

   (* When this wizard frame happens to be run with GUIRunModal we want to return
      a succesful wizard result here. If non-modal use of a wizard is useful,
      one would have some other technique to output the result data in the finish event. *)
   BindEvent["endModal",
     Script[ 
       (* In certain user cases a close could happen without a 
          cancel button being pressed in modal use, so assume cancel and not finished
          if a result state hasn't been setup already
        *)
       If[ $wizardResult === Null, wizardCanceled[]; ];
       $wizardResult
       ]
    ]
      
  }, Name -> "myWizard"]
  
}, Name -> "myWizardFrame"]
  
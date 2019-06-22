(* ::Package:: *)

Paclet[
  Name -> "WebUnit",
  Description -> "WebUnit is a package which automates interaction to a web browser",
  Creator -> "Arnoud Buzing",
  Publisher -> "Wolfram Research",
  Version -> "1.1.4",
  MathematicaVersion -> "11.3+",
  Loading->Automatic,
  Extensions -> { 
    {
      "Kernel",
      Root -> "Kernel",
      Context -> {"WebUnit`"},
      Symbols -> {
        "System`StartWebSession",
        "System`WebExecute",
        "System`WebSessionObject",
        "System`$CurrentWebSession",
        "System`WebSessions",
        "System`WebImage",
        "System`WebWindowObject",
        "System`WebElementObject"
      }
    },
   {
      "Resource",
      Root -> "Resources",
      Resources -> {
          {"Drivers","DriverBinaries"}
        }
    }
  }
]

Paclet[
  Name -> "MUnit",
  Version -> "12.0.0",
  MathematicaVersion -> "12+",
  Loading -> Automatic,
  Extensions -> {
    {
      "Kernel",
      HiddenImport -> False, (* https://jira.wolfram.com/jira/browse/PACMAN-5 *)
      Context -> {"MUnit`", "MUnitLoader`"},
      Symbols -> {
        "System`VerificationTest",
        "System`TestReport",
        "System`TestResultObject",
        "System`TestReportObject",
        "System`MemoryConstraint",
        "System`TestID",
        "System`$TestFileName"
      }
    }
  }
]

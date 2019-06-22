Widget["class:com.wolfram.guikit.graph.ExprAccessibleJGraph", {
  "preferredSize" -> Widget["Dimension", {"width" -> 300, "height" -> 300}]
  },
  InitialArguments -> {
    Widget["class:com.wolfram.guikit.graph.ExprAccessibleGraphPane", {
      "engine" -> PropertyValue[{"ScriptEvaluator", "engine"}]
      },
      InitialArguments -> {
        Widget["class:com.wolfram.guikit.graph.DefaultGraphController", {},
          Name -> "graphController"],
        Widget["class:com.wolfram.guikit.graph.DefaultGraphModel", {},
          Name -> "graphModel"]}]
    },
  Name -> "graphPanel"
 ]
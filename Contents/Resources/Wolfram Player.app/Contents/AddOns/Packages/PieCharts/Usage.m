(* Forward declarations to prevent shadowing messages *)
{System`PieChart};
{PieCharts`PieChart};

BeginPackage["PieCharts`"]

If[!ValueQ[PieChart::usage], PieChart::usage = "\!\(\*RowBox[{\"PieChart\", \"[\", RowBox[{\"{\", RowBox[{SubscriptBox[StyleBox[\"y\", \"TI\"], StyleBox[\"1\", \"TR\"]], \",\", SubscriptBox[StyleBox[\"y\", \"TI\"], StyleBox[\"2\", \"TR\"]], \",\", StyleBox[\"\[Ellipsis]\", \"TI\"]}], \"}\"}], \"]\"}]\) generates a pie chart of the positive values \!\(\*SubscriptBox[StyleBox[\"y\", \"TI\"], StyleBox[\"i\", \"TI\"]]\)."];
If[!ValueQ[PieEdgeStyle::usage], PieEdgeStyle::usage = "PieEdgeStyle is an option for PieChart that specifies the styles for lines in the pie chart."];
If[!ValueQ[PieExploded::usage], PieExploded::usage = "PieExploded is an option for PieChart that specifies the distance to move wedges outwards radially in the pie chart."];
If[!ValueQ[PieLabels::usage], PieLabels::usage = "PieLabels is an option for PieChart that specifies the labels on the pie wedges."];
If[!ValueQ[PieOrientation::usage], PieOrientation::usage = "PieOrientation is an option for PieChart that determines the placement of the first piece of data."];
If[!ValueQ[PieStyle::usage], PieStyle::usage = "PieStyle is an option for PieChart that specifies a style for each wedge of the pie chart."];

EndPackage[]

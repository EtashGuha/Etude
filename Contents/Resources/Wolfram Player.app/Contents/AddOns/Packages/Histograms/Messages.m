BeginPackage["Histograms`"]

Histogram::ticks =
"`` is not a valid tick specification.  Taking Ticks->Automatic.";

Histogram::hcat =
"`` is not a valid histogram categories specification. Taking \
HistogramCategories->Automatic.";

Histogram::rcount = "Frequency count of data in categories failed.";

Histogram::realvec="The first argument to Histogram is expected to be a vector of real values.";

Histogram::noapprox =
"ApproximateIntervals -> `` is a not a valid setting when \
HistogramCategories->{c1, c2, ..., cm}. Taking ApproximateIntervals -> False.";

Histogram::ltail1 =
"Warning: One point from the left tail of the data, strictly less than `1`, \
is not included in histogram.";

Histogram::ltail =
"Warning: `1` points from the left tail of the data, strictly less than `2`, \
are not included in histogram.";

Histogram::rtail1 =
"Warning: One point from the right tail of the data, greater than or equal \
to `1`, is not included in histogram.";

Histogram::rtail =
"Warning: `1` points from the right tail of the data, greater than or equal \
to `2`, are not included in histogram.";

Histogram::range =
"Warning: `` is not a valid setting for HistogramRange. \
Taking HistogramRange -> Automatic.";

Histogram::fdhc =
"Warning: `` is not a valid setting for HistogramCategories when \
FrequencyData -> True.  When the data represents frequencies, \
HistogramCategories should specify Automatic or a list of cutoffs. \
Taking HistogramCategories -> Automatic.";

Histogram::fdfail =
"When FrequencyData -> True and HistogramCategories -> cutoffs, the \
length of the cutoffs vector should be exactly one more than the length \
of the frequency data.";


Histogram3D::ticks =
"`` is not a valid tick specification.  Taking Ticks->Automatic."

Histogram3D::rcount = "Frequency count of data in categories failed."

Histogram3D::rdhc =
"Warning: `` is not a valid setting for HistogramCategories when FrequencyData -> False.  When the data is raw (not frequencies), HistogramCategories should specify (i) Automatic, (ii) a positive integer denoting the total number of bivariate categories that the data is to be divided into, or (iii) a vector {xhc, yhc}.  Either xhc or yhc may be (i) Automatic, (ii) a positive integer denoting the number of categories that the corresponding component of the data is to be divided into, or (iii) a vector of monotonically increasing numbers denoting cutoffs. Taking HistogramCategories -> Automatic."

Histogram3D::fdhc =
"Warning: `` is not a valid setting for HistogramCategories when FrequencyData -> True.  When the data represents frequencies, HistogramCategories should specify Automatic (implying {Automatic, Automatic}), or {xcutoffs, Automatic}, or {Automatic, ycutoffs}, or {xcutoffs, ycutoffs}, where xcutoffs and ycutoffs are vectors of monotonically increasing numbers. Taking HistogramCategories -> Automatic."

Histogram3D::badrg =
"Warning: `` is not a valid value for a component of the setting for the HistogramRange option.  Taking the component to be Automatic."

Histogram3D::rd2d =
"When FrequencyData -> False, the data must be in the form of two-dimensional data {{x1, y1}, {x2, y2}, ...}."

Histogram3D::fdfail =
"When FrequencyData -> True and HistogramCategories -> {xcutoffs, ycutoffs},  the length of the xcutoffs vector should be exactly one more than the number of rows in the frequency data matrix, and the length of the ycutoffs vector should be exactly one more than the number of columns in the frequency data matrix."

Histogram3D::lt1 =
"Warning: `1` point with `2` component strictly less than `3` is not included in histogram."

Histogram3D::lt =
"Warning: `1` points with `2` components strictly less than `3` are not included in histogram."

Histogram3D::gtet1 =
"Warning: `1` point with `2` component greater than or equal to `3` is not included in histogram."

Histogram3D::gtet =
"Warning: `1` points with `2` components greater than or equal to `3` are not included in histogram."

Histogram3D::range =
"Warning: `` is not a valid setting for HistogramRange. Taking HistogramRange -> Automatic."

Histogram3D::noapprox =
"ApproximateIntervals -> `` is a not a valid setting when  HistogramCategories->{c1, c2, ..., cm}.  Taking ApproximateIntervals -> False.";


EndPackage[]

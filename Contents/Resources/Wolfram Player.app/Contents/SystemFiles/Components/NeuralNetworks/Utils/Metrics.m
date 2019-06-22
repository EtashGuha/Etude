Package["NeuralNetworks`"]


(*

NDSummary[min1_, max1_, mean1_] + NDSummary[min2_, max2_, mean2_] ^:= 
	 NDSummary[Min[min1, min2], Max[max1, max2], (mean1 + mean2) / 2];

NDArrayGet[NDMetricArray[nd_, postprocess_]] := 
	postprocess @ NDArrayGet[nd];

 (* TODO: in future it will be more efficient to do the summarization (min, max etc) in MXNetLink *)


PackageScope["
*)
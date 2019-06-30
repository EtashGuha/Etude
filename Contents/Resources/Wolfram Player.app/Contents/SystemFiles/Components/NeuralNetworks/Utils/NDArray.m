Package["NeuralNetworks`"]


PackageScope["NDNoTotalArray"]

NDNoTotalArray /: NDArrayGetPartialTotalNormal[NDNoTotalArray[nd_], excess_, level_] := 
	NDArrayGetNormal[nd];

NDNoTotalArray /: NDArrayGetTotalNormal[NDNoTotalArray[nd_], 1] := 
	NDArrayGetNormal[nd];

NDNoTotalArray /: NDArraySetConstant[NDNoTotalArray[nd_NDArray], constant_] :=
	NDArraySetConstant[nd, constant];

NDNoTotalArray /: NDArrayGetNormal[NDNoTotalArray[nd_NDArray]] := 
	NDArrayGetNormal @ nd;

NDNoTotalArray /: NDArrayGet[NDNoTotalArray[nd_NDArray]] := 
	NDArrayGet @ nd;


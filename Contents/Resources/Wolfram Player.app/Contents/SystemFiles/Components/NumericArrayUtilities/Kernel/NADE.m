Package["NumericArrayUtilities`"]

(******************************************************************************)
PackageExport["GradientNADE"]
PackageExport["LogProbabilityNADE"]

SetUsage[GradientNADE, "
Computes the gradient of the NADE cost function as defined in \
http://homepages.inf.ed.ac.uk/imurray2/pub/11nade/nade.pdf
GradientNADE[theta_0, data, rate] updates theta_0 using \
theta = theta_0-rate*gradient_{data, theta_0}
"
]

SetUsage[LogProbabilityNADE, "
Computes the log-probability of examples using the nade algorithm. \
Parallel implementation for many examples.
"
]


DeclareLibraryFunction[gradientNADE, "gradient_NADE", 
	{
		{Real,_,"Shared"},
		{Real,_,"Shared"}, 
		{Real,1,"Shared"},
		{Real,1,"Shared"}, 
		{Real,2,"Constant"}, 
		Real
	}, 
	"Void"						
	]
	
DeclareLibraryFunction[logprobabilityNADE, "logprobability_NADE_parallel", 
	{
		{Real,2,"Constant"},
		{Real,2,"Constant"}, 
		{Real,1,"Constant"},
		{Real,1,"Constant"}, 
		{Real, 2, "Constant"}
	}, 
	{Real, 1}						
	]	
	
GradientNADE[model_, vector_, lambda_] := Module[
	{b, c, V, W},
	{b, c, V, W} = model;
	Quiet@gradientNADE[V, W, b, c, vector, lambda];
	{b, c, V, W}
]	


LogProbabilityNADE[vector_?(ArrayQ[#, 2]&), {b_List, c_List, matrixV_, matrix_}] := 
	logprobabilityNADE[matrixV, matrix, b, c, vector]
	
	


(* 
	This dummy layer stands for a generic placeholder to whatever mxnet node we can't conveniently import
	with the importer. Notably, that includes nodes whose parsing rules involve checking properties of 
	the surrounding nodes in the mxnet graph, or also groups of nodes that need be converted to a single 
	WL node. 
	The way to use this is to specify the desired nodes in the "PlaceholderNodes" option of MXNetImport, 
	and all specified nodes will be parsed as this placeholder. Once the net is imported, replace all 
	placeholders using NetReplace[net, MXPlaceholderLayer[...] -> ...]. The parameters of this layer carry 
	the original specification of the corresponding mxnet node, and should provide enough information to 
	perform the correct replacement.
 *)
Inputs: 
	$Multiport: RealTensorT

Parameters:
	$MXName: StringT
	$MXType: StringT
	$MXParams: AssocT[StringT, StringT]

Output: RealTensorT
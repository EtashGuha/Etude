//simply try to resolve the packages we need, printing true if we don't throw an exception
try
{
	//the two packages we need are zeromq and vm
	require.resolve('zeromq');
	require.resolve('vm');
	//worked
	console.log('TRUE');
}
catch( e )
{
	//an exception was thrown, so print false
	console.log('FALSE');
}

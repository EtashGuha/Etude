//simple evaluator

//load zmq and create a reply socket
var zmq = require('zeromq');
//var introspect = require('introspect');
var sock = zmq.socket('pair');

var wl = require('WLExpr');

//load the vm module for evaluating code
const vm = require('vm');

//bind to port 3000 on localhost
sock.bindSync('tcp://127.0.0.1:*');
console.log(sock.getsockopt(zmq.ZMQ_LAST_ENDPOINT));

//setup a context to run the code inside
var ctx = vm.createContext();

//store all the global variables into the context so they are accessible as 
//expected
ctx.require = require;
ctx.global = global;
ctx.process = process;
ctx.Buffer = Buffer;
ctx.console = console;

//this one is for deserializing data in WL to python

//we capture stdout with this function, which just sends back the output on the socket
//as it's received
ctx.process.stdout.write = function(string, encoding, fd) {
    //send the string over the socket back to be Print'd off
    sock.send(JSON.stringify({'stdout':string}));
};

function isFunction(functionToCheck) {
	var getType = {};
	return functionToCheck && getType.toString.call(functionToCheck) === '[object Function]';
}

//the replacer function used to produce ExpressionJSON instead of just normal JSON
//this lets us return generic expr's to WL without having to do complicated stuff outside of the 
//Import call
//this is a closure around the sessionUUID which we need to be able to pass to serializer functions
function exprJSONReplacer(sessionUUID) {
	function inner(name,val){

		if (val === Infinity || val === -Infinity) {
			//Infinity in JS is actualy a DirectedInfinity in WL
			return (new wl.Expr('DirectedInfinity',val === Infinity ? 1 : -1)).toExprJSONObj();
		} else if(val && Array.isArray(val)) { 
			if(val.length > 0 && val[0] instanceof wl.HeadString) {
				//already has a head, so just return the array as is
				return val;
			} else {
				//prepend the head List to the array
				return (new wl.Expr('List',val)).toExprJSONObj();
			}
		} else if (val instanceof wl.Expr) {
			//turn this expr object into a ExpressionJSON object list
			//than itself is then transformed again into actual json objects that are primitives and serializable
			return val.toExprJSONObj();
		} else if (val instanceof wl.HeadString) {
			//then this object is a head and we don't want to put quotes around it,
			//so just return the head property, which won't have the json replacer called on it again
			return val.head;
		} else if (typeof val === 'string' || val instanceof String){ 
			//add single quotes around it because it's a string so we have to add quotes around it
			return "'"+val+"'"; 
		} else if (isFunction(val)){
			return (new wl.Expr(
				'ExternalFunction',
				[
					{
						'Name':val.name,
						'System':"NodeJS",
						"Type":"NodeJSFunction",
						'NumberArguments':val.length,
						"Source":val.toString()
					}
				]
			)).toExprJSONObj();
		} else if (val === Object(val)){
			//then this is a "object" in javascript, which means we should attempt to convert it
			//to an association
			//so we first make up the expr with head "Association"
			var args = [];
			//now iterate over all the key value pairs in the object, putting them into the association as rules
			for(var key in val) {
				args.push(new wl.Expr('Rule',[key,val[key]]));
			}
			return (new wl.Expr('Association',args)).toExprJSONObj();
		} else {
			return val;
		}
	}

	return inner;
}

//when messages are received, try and eval the string, sending back the result
sock.on('message',
	function(sockmsg){
		var msgStringJSON = sockmsg.toString();
		var returnString;
		var returnJSON = {};
		try{
			//try to turn the message into a json object
			request = JSON.parse(msgStringJSON)
			if('input' in request) {

				//reset the messages
				global.logstdout = [];
				//then eval the input using the vm in the specifically setup ctx
				returnJSON['output'] = vm.runInContext(request['input'],ctx);
			} else if('function' in request && 'args' in request) {
				//then we have to lookup the function and evaluate it
				var func = vm.runInContext(request['function'],ctx);

				returnJSON['func'] = func;
				//now actually call the function destructuring the arguments
				returnJSON['output'] = func(...request['args']);
			} else {
				//no input key, throw an exception
				throw new Error('Missing input key in request');
			}

			
			//now handle the serialization type, i.e. return_type
			if (request['return_type'] == 'string') {
				//then we need to make the object a string
				returnJSON['output'] = JSON.stringify(returnJSON['output']);
			} else if (request['return_type'] == 'expr') {
				//then export the object into a string using ExpressionJSON
				//the ExternalEvaluate framework will import the outer layer as JSON, then the output 
				//key will get deserialized into expr's with the ExpressionJSON importer
				returnJSON['output'] = JSON.stringify(returnJSON['output'],exprJSONReplacer(request['session_uuid']));
				returnJSON['is_expr'] = true
			} else {
				throw new Error('Invalid return_type key :'+request['return_type']);
			}

			//check if the output is undefined, in which case make it null
			if (('output' in returnJSON && returnJSON['output'] == undefined)) {
				returnJSON['output'] = null;
			}

		} catch(err){
			//reset returnJSON to just be the exception details, as encoding it could be invalid, which is what caused us to get thrown here
			returnJSON = {'error' : {
				'FailureCode': Object.prototype.toString.call(err).slice(8, -1),
				'MessageTemplate': err.toString(),
				'Traceback': err.stack,
			}};
		}
		
		//finally send back the result
		sock.send(JSON.stringify(returnJSON));
	}
);


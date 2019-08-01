
const sudo = require('sudo-prompt');
const fs = require('fs')
var options = {
  name: 'Etude'
};

var hasJDK = false;
require('find-java-home')(function(err, home){
  if(err)return console.log(err);
  hasJDK = true;
});
const etudeFilepath = __dirname.replace("/public/js","").replace("\\public\\js","")
module.exports = {
	macSetup: function() {
		console.log("Hello")
	}
}

function callCommand(executionstring){
	var exec = require('child_process').execSync, child;

	child = exec(executionstring, {
		cwd: etudeFilepath
	},
	  function (error, stdout, stderr) {
		  if (error !== null) {
			   console.log('exec error: ' + error);
		  }
		});
}
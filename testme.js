var shell = require('shelljs');
const fs = require('fs')
var hasJDK = false
fs.readdir("/Library/Java/JavaVirtualMachines", (err, files) => {
  files.forEach(file => {
    if(file.substring(0,3) == 'jdk'){
    	hasJDK = true;
    }
  });
});
var jre = require('node-jre');

var output = jre.spawnSync(  // call synchronously
    ['java'],                // add the relative directory 'java' to the class-path
    'Hello',                 // call main routine in class 'Hello'
    ['World'],               // pass 'World' as only parameter
    { encoding: 'utf8' }     // encode output as string
  ).stdout.trim();           // take output from stdout as trimmed String

console.log(output === 'Hello, World!')

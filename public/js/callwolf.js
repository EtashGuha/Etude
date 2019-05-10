console.log("killa");
var java = require('java');

java.classpath.push("Kernel.jar");
java.classpath.push("/Applications/Wolfram\ Desktop.app/Contents/SystemFiles/Links/JLink/JLink.jar")

var kernel = java.newInstanceSync('p1.Kernel');

console.log(kernel.findTextAnswerSync("foo", "bar"));
console.log(kernel.findTextAnswerSync("choo", "asdf"));
kernel.killKernelSync();


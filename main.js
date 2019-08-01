// Modules to control application life and create native browser window
// const {app, BrowserWindow} = require('electron')
const electron = require('electron')
const app = electron.app
const BrowserWindow = electron.BrowserWindow
const { ipcMain } = require('electron')
const rebuild = require('electron-rebuild');
let childProcess = require('child_process');
const typeOf = require('typeof')
var sudo = require('sudo-prompt');
const os = require('os')
const firstRun = require('electron-first-run');
const etudeFilepath = __dirname.replace("/public/js","").replace("\\public\\js","")
var osvers = os.platform()
var fs = require('fs');
var isFirstRun = firstRun()
const locateJavaHome = require('locate-java-home'); 
var npm = require('npm');
// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow
console.log(Object.values(locateJavaHome))
console.log(locateJavaHome.default)
console.log(typeOf(locateJavaHome))
global.sharedObject = {
  someProperty: ''
}
var hasJdk = false;

ipcMain.on('show_pdf_message', (event, arg) => {
  sharedObject.someProperty = arg
})

function createWindow () {
  // Create the browser window.
const {width, height} = electron.screen.getPrimaryDisplay().workAreaSize
mainWindow = new BrowserWindow({
	height: height,
	width: width,
	minWidth: 600,
	minHeight: 200,
	frame: false,
	backgroundColor: '#ffffff',
	webPreferences: {
	  nodeIntegration: true
	},
	icon: 'assets/images/logo.jpg',})

var options = {
  	name: 'Etude'
};
mainWindow.loadFile('splash.html')
isFirstRun = true;
if(isFirstRun){
	locateJavaHome.default({
	    // Uses semver :) Note that Java 6 = Java 1.6, Java 8 = Java 1.8, etc.
	    version: ">=1.6",
	    mustBeJDK: true
	}, function(error, javaHomes) {
		console.log(javaHomes)
		console.log(javaHomes.length)
	    if(javaHomes.length > 0){
	    	console.log("hasJdk")
	    	hasJdk = true;
	    	console.log(hasJdk)
	    }
	    console.log(hasJdk);
	    moveJava()
	});
}
  // and load the index.html of the app.
  ///////////////////////////////////////mainWindow.setMenu(null)

  // Open the DevTools.
   // mainWindow.webContents.openDevTools()

  // Emitted when the window is closed.
  mainWindow.on('closed', function () {
	// Dereference the window object, usually you would store windows
	// in an array if your app supports multi windows, this is the time
	// when you should delete the corresponding element.
	mainWindow = null
  })
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on('ready', function(){
  createWindow()
})

// Quit when all windows are closed.
app.on('window-all-closed', function () {
  // On macOS it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
	app.quit()
})

app.on('activate', function () {
  // On macOS it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (mainWindow === null) {
	createWindow()
  }
})

function runScript(scriptPath, callback) {

	// keep track of whether callback has been invoked to prevent multiple invocations
	var invoked = false;

	var process = childProcess.fork(scriptPath);

	// listen for errors as they may prevent the exit event from firing
	process.on('error', function (err) {
		if (invoked) return;
		invoked = true;
		callback(err);
	});

	// execute the callback once the process has finished running
	process.on('exit', function (code) {
		if (invoked) return;
		invoked = true;
		var err = code === 0 ? null : new Error('exit code ' + code);
		callback(err);
	});

}

function checkFlag(){
	try {
		var testFolder = etudeFilepath;
		fs.readdir(testFolder, (err, files) => {
			console.log(err)
			files.forEach(file => {
				console.log(file);
			});
		});
		runScript(etudeFilepath + '/node_modules/java/postInstall.js', function (err) {
		 			console.log("running script")
			  		if (err) {
			  			setTimeout(() => checkFlag(), 100);
			  		} else {
			  			setTimeout(() => {mainWindow.loadFile('library.html')}, 1000);
			  		}
			  		console.log('finished running some-script.js');
		  		});
	} catch {
		console.log("checking")
		setTimeout(() => checkFlag(), 100)
	}
}
function moveJava(){
	console.log(hasJdk)
	if(osvers == 'darwin'){
		if(!hasJdk){
			sudo.exec('cp ' + etudeFilepath + '/jdk-11.0.2.jdk /Library/Java/JavaVirtualMachines', options,
		  		function(error, stdout, stderr) {
		    		if (error) throw error;
		    		console.log('stdout: ' + stdout);
		  		}
			);
		} else {
			reinstallNodeJava()
		}

	} else {
		console.log("TRYING ON WINDOWS")
		sudo.exec('set JAVA_HOME=' + etudeFilepath + '/jdk-11.0.1', options,
	  		function(error, stdout, stderr) {
	    		if (error) throw error;
	    		console.log('stdout: ' + stdout);
	  		}
		);
		console.log('setx JAVA_HOME \"' + etudeFilepath + '/jdk-11.0.1\"')
		sudo.exec('setx JAVA_HOME \"' + etudeFilepath + '/jdk-11.0.1\"', options,
	  		function(error, stdout, stderr) {
	    		if (error) throw error;
	    		console.log('stdout: ' + stdout);
	    		reinstallNodeJava()
	  		}
		);
	}
}
function reinstallNodeJava(){
	npm.load(function(err) {

		// install module ffi
		npm.commands.install(['java'], function(er, data) {
			console.log("DONE")
			rebuild.rebuild({
				buildPath: __dirname,
				electronVersion: '4.0.5'
		  	}).then(() => {
		  		checkFlag()
			}).catch((e) => {
		  		console.error("Building modules didn't work!");
		  		console.error(e);
			});
		});

		npm.on('log', function(message) {
			// log installation progress
			console.log(message);
		});
  	});
}
// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.

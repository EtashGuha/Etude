// Modules to control application life and create native browser window
// const {app, BrowserWindow} = require('electron')
const electron = require('electron')
var stripe = require('stripe')('rk_live_pVDuyAoclBtFPPIWIZ8rHCl200kbPvuYWk');
const app = electron.app
const BrowserWindow = electron.BrowserWindow
// Google analytics
const { trackEvent } = require('./analytics');
global.trackEvent = trackEvent;
const {
    ipcMain
} = require('electron')
const {
    autoUpdater
} = require("electron-updater");
var currpathtofile = null;
var fileOpen = false;
const date = require('date-and-time');
const now = new Date();
if (process.platform == 'win32' && process.argv.length >= 2 && process.argv[1] !== ".") {
    currpathtofile = process.argv[1]
    fileOpen = true;

}
const etudeFilepath = __dirname.replace("/public/js", "").replace("\\public\\js", "")
var fs = require('fs');
var options = {
    name: 'Étude'
};
var unpackedDirectory = etudeFilepath.replace("app.asar", "app.asar.unpacked")
const analytics = require('electron-google-analytics');
const analyti = new analytics.default('UA-145681611-1')
const userDataPath = (electron.app || electron.remote.app).getPath('userData');
var ready = false

const Store = require('electron-store');
var store = new Store();

//store.clear();
// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow
global.sharedObject = {
    someProperty: '',
    newWindow: false
}

const freeTrialLength = 14
var isLicensed = store.has("stripeID");

ipcMain.on('show_pdf_message', (event, arg) => {
    console.log("OPENNING A PDF")

    sharedObject.someProperty = arg
    fileOpen = true;
})
var hasFirstTime = store.has("isFirstTime");
var needsToLicencse = false
var isFreeTrial = false
if(!hasFirstTime){
    store.set("isFirstTime", true)
    store.set("startDate", now)
}

isFreeTrial = (date.subtract(now, new Date(store.get("startDate"))).toDays() < freeTrialLength);
needsToLicense = (!isFreeTrial && !isLicensed)


app.on('ready', function() {
    let framebool = true;
    if (process.platform == 'win32') {
        framebool = false;
    }
    const {
        width,
        height
    } = electron.screen.getPrimaryDisplay().workAreaSize
    mainWindow = new BrowserWindow({
        height: height,
        width: width,
        minWidth: 600,
        minHeight: 200,
        frame: framebool,
        backgroundColor: '#ffffff',
        webPreferences: {
            nodeIntegration: true
        },
        icon: 'assets/images/logo.jpg',
    })
    createWindow();
});



app.on('will-finish-launching', function() {
    app.on('open-file', function(ev, path) {
        fileOpen = true;
        ev.preventDefault();

        currpathtofile = path;
        sharedObject.someProperty = path;

        checkForUpdates();
        let framebool = true;
    if (process.platform == 'win32') {
        framebool = false;
    }
        const {
        width,
        height
    } = electron.screen.getPrimaryDisplay().workAreaSize
    mainWindow = new BrowserWindow({
        height: height,
        width: width,
        minWidth: 600,
        minHeight: 200,
        frame: framebool,
        backgroundColor: '#ffffff',
        webPreferences: {
            nodeIntegration: true
        },
        icon: 'assets/images/logo.jpg',
    })
        //This is google analytics stuff
        // analyti.pageview('http://etudereader.com', '/home', 'Example').then((response) => {
        //     return response;
        // });
        trackEvent("Open", "HomePage");
        // mainWindow.webContents.openDevTools()
        if (needsToLicense) {
            mainWindow.loadFile('verify.html')
        } else if(!isFreeTrial && (!store.has("LastCheckDate") || date.subtract(now, new Date(store.get("LastCheckDate"))).toDays() > freeTrialLength)) {
            store.set("LastCheckDate", now)
            console.log(store.get("LastCheckDate"))
            stripe.subscriptions.retrieve(
                store.get("stripeID"),
                function(err, subscription) {
                    if(subscription.status === "trialing" || subscription.status === "active"){
                        sharedObject.newWindow = true
                        mainWindow.loadFile('summarizing.html')
                    } else {
                        mainWindow.loadFile('verify.html')
                    }
                }
            );    
        } else {
            sharedObject.newWindow = true
            mainWindow.loadFile('summarizing.html')
        }
    });
});
ipcMain.on('get-file-data', function(event) {
    fileOpen = true;
    var data = null
    event.returnValue = currpathtofile
    currpathtofile = null
})

function createWindow() {
    checkForUpdates();

    // analyti.pageview('http://etudereader.com', '/home', 'Example').then((response) => {
    //     return response;
    // });
    trackEvent("Open", "HomePage");

    if (needsToLicense) {
        mainWindow.loadFile('verify.html')
    } else if(!isFreeTrial && (!store.has("LastCheckDate") || date.subtract(now, new Date(store.get("LastCheckDate"))).toDays() > freeTrialLength)){
        store.set("LastCheckDate", now)
        console.log(store.get("LastCheckDate"))
        stripe.subscriptions.retrieve(
            store.get("stripeID"),
            function(err, subscription) {
                if(subscription.status === "trialing" || subscription.status === "active"){
                   finishCreatingWindow()
                } else {
                    mainWindow.loadFile('verify.html')
                }
            }
        );
    } else {
        finishCreatingWindow()
    }

    // Emitted when the window is closed.           
    mainWindow.on('closed', function() {
        // Dereference the window object, usually you would store windows
        // in an array if your app supports multi windows, this is the time
        // when you should delete the corresponding element.
        mainWindow = null
    })
}

function finishCreatingWindow(){
    if (fileOpen) {
        sharedObject.someProperty = currpathtofile;
        sharedObject.newWindow = true
        mainWindow.loadFile('summarizing.html')
    } else {
        mainWindow.loadFile('splash.html')
        setTimeout(() => {
            console.log(currpathtofile)
            mainWindow.loadFile('library.html')
        }, 1000);
    }
}
function checkForUpdates() {
    autoUpdater.checkForUpdatesAndNotify();

    function sendStatusToWindow(text) {
        mainWindow.webContents.send('message', text);
    }
    autoUpdater.on('checking-for-update', () => {
        sendStatusToWindow('Checking for update...');
    })
    autoUpdater.on('update-available', (info) => {
        sendStatusToWindow('Update available.');
    })
    autoUpdater.on('update-not-available', (info) => {
        sendStatusToWindow('Update not available.');
    })
    autoUpdater.on('error', (err) => {
        sendStatusToWindow('Error in auto-updater. ' + err);
    })
    autoUpdater.on('download-progress', (progressObj) => {
        let log_message = "Download speed: " + progressObj.bytesPerSecond;
        log_message = log_message + ' - Downloaded ' + progressObj.percent + '%';
        log_message = log_message + ' (' + progressObj.transferred + "/" + progressObj.total + ')';
        sendStatusToWindow(log_message);
    })
    autoUpdater.on('update-downloaded', (info) => {
        sendStatusToWindow('Update downloaded');
        autoUpdater.quitAndInstall();
    });
}
// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.



// Quit when all windows are closed.
app.on('window-all-closed', function() {
    // On macOS it is common for applications and their menu bar
    // to stay active until the user quits explicitly with Cmd + Q
    app.quit()
})

app.on('activate', function() {
    // On macOS it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open.
    if (mainWindow === null) {
        createWindow()
    }
})

module.exports = userDataPath
// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.
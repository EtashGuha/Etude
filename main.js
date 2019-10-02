// Modules to control application life and create native browser window
// const {app, BrowserWindow} = require('electron')
const electron = require('electron')
const app = electron.app
const BrowserWindow = electron.BrowserWindow
const {
    ipcMain
} = require('electron')
const {
    autoUpdater
} = require("electron-updater");

const etudeFilepath = __dirname.replace("/public/js", "").replace("\\public\\js", "")
var fs = require('fs');
var options = {
    name: 'Etude'
};
var unpackedDirectory = etudeFilepath.replace("app.asar", "app.asar.unpacked")
const analytics = require('electron-google-analytics');
const analyti = new analytics.default('UA-145681611-1')
const userDataPath = (electron.app || electron.remote.app).getPath('userData');


// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow
global.sharedObject = {
    someProperty: ''
}

ipcMain.on('show_pdf_message', (event, arg) => {
    sharedObject.someProperty = arg
})

function createWindow() {
    const {
        width,
        height
    } = electron.screen.getPrimaryDisplay().workAreaSize
    mainWindow = new BrowserWindow({
        height: height,
        width: width,
        minWidth: 600,
        minHeight: 200,
        frame: true,
        backgroundColor: '#ffffff',
        webPreferences: {
            nodeIntegration: true
        },
        icon: 'assets/images/logo.jpg',
    })

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
        log_message = log_message + ' (' + pogressObj.transferred + "/" + progressObj.total + ')';
        sendStatusToWindow(log_message);
    })
    autoUpdater.on('update-downloaded', (info) => {
        sendStatusToWindow('Update downloaded');
        autoUpdater.quitAndInstall();
    });
    mainWindow.loadFile('splash.html')
    if (!fs.existsSync(userDataPath + "/tmp")) {
        fs.mkdirSync(userDataPath + "/tmp")
    }
    if (!fs.existsSync(userDataPath + "/folderForHighlightedPDF")) {
        fs.mkdirSync(userDataPath + "/folderForHighlightedPDF")
    }
    //This is google analytics stuff
    analyti.pageview('http://etudereader.com', '/home', 'Example').then((response) => {
        return response;
    });
    mainWindow.webContents.openDevTools()
    setTimeout(() => {
        mainWindow.loadFile('library.html')
    }, 1000);

    mainWindow.webContents.openDevTools()

    // Emitted when the window is closed.
    mainWindow.on('closed', function() {
        // Dereference the window object, usually you would store windows
        // in an array if your app supports multi windows, this is the time
        // when you should delete the corresponding element.
        mainWindow = null
    })
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on('ready', function() {
    createWindow()
})

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
// Modules to control application life and create native browser window
// const {app, BrowserWindow} = require('electron')
const electron = require('electron')
const app = electron.app
const BrowserWindow = electron.BrowserWindow
const { ipcMain } = require('electron')
const {autoUpdater} = require("electron-updater");

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow

global.sharedObject = {
  someProperty: ''
}

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

  mainWindow.loadFile('splash.html')

  setTimeout(() => {mainWindow.loadFile('library.html')}, 1000);

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
	autoUpdater.checkForUpdatesAndNotify();
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

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.

const electronVersion = require('electron-version')
electronVersion(function (err, v) {
  console.log(err, v) // null 'v0.33.4'
})

const ua = require('universal-analytics');
const uuid = require('uuid/v4');
const Store = require('electron-store');
const store = new Store();

var userID;
// Retrieve the userid value, and if it's not there, assign it a new uuid.
if (store.has("userID")) {
    userID = store.get("userID");
} else {
    userID = uuid();
    store.set("userID", userID);
}
console.log(userID)
const usr = ua('UA-145681611-1', userID);

function trackEvent(category, action, label, value) {
    usr
      .event({
        ec: category,
        ea: action,
        el: label,
        ev: value,
      })
      .send();
}

module.exports = { trackEvent };
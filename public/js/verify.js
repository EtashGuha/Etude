const MongoClient = require('mongodb').MongoClient;
const uri = "mongodb+srv://normalUser:etudereader@etude-zno3q.mongodb.net/test?retryWrites=true&w=majority";
const client = new MongoClient(uri, {
  useNewUrlParser: true
});
const Store = require('electron-store');
const windowFrame = require('electron-titlebar')
var store = new Store();


var searchbox = document.getElementById("licenseSubmit");
// Execute a function when the user releases a key on the keyboard
var licenseArr = [];
client.connect(err => {
  const collection = client.db("UserData").collection("Licenses");
  collection.find({}, {
    projection: {
      _id: 0,
      stripeID: 1,
      used: 1
    }
  }).toArray(function(err, result) {
    licenseArr = result;

  });
});



searchbox.addEventListener("click", function(event) {
  var found = false;
  licenseArr.forEach((element) => {
    console.log(element.stripeID === document.getElementById("licenseID").value);
    if (element.stripeID === document.getElementById("licenseID").value) {
      if(element.used === false) {
            element.used = true;
            var myquery = { stripeID: element.stripeID};
            client.db("UserData").collection("Licenses").updateOne(myquery, { $set: element }, function(err, res) {
              if (err) throw err;
              console.log("1 document updated");
              client.close();
            });

      store.set("stripeID", document.getElementById("licenseID").value)
      found = true;
      window.location.href = 'library.html';
    } else {
      found = true;
      document.getElementById("headingTitle").innerHTML = "Lincense Already in Use";
    }
  }});
  if (!found) {
    document.getElementById("headingTitle").innerHTML = "Incorrect License. Please Try Again";
  }


});
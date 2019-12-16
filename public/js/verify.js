const MongoClient = require('mongodb').MongoClient;
const uri = "mongodb+srv://normalUser:etudereader@etude-zno3q.mongodb.net/test?retryWrites=true&w=majority";
const client = new MongoClient(uri, { useNewUrlParser: true });
const Store = require('electron-store');
const windowFrame = require('electron-titlebar')
var store = new Store();


var searchbox = document.getElementById("licenseSubmit");
// Execute a function when the user releases a key on the keyboard
var licenseArr = [];
client.connect(err => {
	  const collection = client.db("UserData").collection("Licenses");
	  collection.find({}, { projection: { _id: 0, stripeID: 1 } }).toArray(function(err, result) {
	  	licenseArr = result;

	  });
	  client.close();
});



searchbox.addEventListener("click", function(event) {
	var found = false;
	licenseArr.forEach((element) => {
  		if(element.stripeID === document.getElementById("licenseID").value && !element.used) {
  			store.set("stripeID", document.getElementById("licenseID").value)
  			found = true;
  			window.location.href = 'library.html';
  		} else if(element.stripeID === document.getElementById("licenseID").value && element.used) {
  			document.getElementById("headingTitle").innerHTML = "Lincense Already in Use";
  		}
  	});
  	if(!found) {
  		document.getElementById("headingTitle").innerHTML = "Incorrect License. Please Try Again";
  	}

	
});
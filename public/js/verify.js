const MongoClient = require('mongodb').MongoClient;
const uri = "mongodb+srv://normalUser:etudereader@etude-zno3q.mongodb.net/test?retryWrites=true&w=majority";
const client = new MongoClient(uri, {
	useNewUrlParser: true
});
const stripe = require('stripe')("sk_live_AM3jYMiKut3IvS8FAzsE7G7q00nfl0xbAv")
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
		if (element.stripeID === document.getElementById("licenseID").value) {
			console.log(element)
			console.log(element.used)
			if (element.used === false) {
				console.log("NOT USED")
				found = true;
				stripe.subscriptions.retrieve(
					element.stripeID,
					function(err, subscription) {
						if (!err && (subscription.status === "trialing" || subscription.status === "active")) {
							element.used = true;
							var myquery = {
								stripeID: element.stripeID
							};
							client.db("UserData").collection("Licenses").updateOne(myquery, {
								$set: element
							}, function(err, res) {
								if (err) throw err;
								console.log("1 document updated");
							});
							store.set("stripeID", document.getElementById("licenseID").value)

							window.location.href = 'library.html';
						} else {
							console.log(err)
							console.log(subscription)
							console.log("NOT A PROPER STATUS")
							document.getElementById("headingTitle").innerHTML = "Incorrect License. Please Try Again.";
						}
					}
				);
			} else {
				found = true;
				document.getElementById("headingTitle").innerHTML = "License Already in Use";
			}
		}
	});
	if (!found) {
		document.getElementById("headingTitle").innerHTML = "Incorrect License. Please Try Again";
	}


});
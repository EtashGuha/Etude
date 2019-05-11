var wolfram = require('wolfram').createClient("GWKU76-Y3HLQH95A7")
 
wolfram.query("integrate 2x", function(err, result) {
  console.log("Result: %j", result)
})
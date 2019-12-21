const fetch = require("node-fetch");
// async function bnana() {

// // Get our user's information from the form fields
// const firstName = "hello"
// const lastName = "anymail"
// const email = "etasaaah@gmail.com"
// const password = "asdfasdfasdf"

// // Create the user
// const response = await fetch("https://api.keygen.sh/v1/accounts/etude/users", {
//   method: "POST",
//   headers: {
//     "Content-Type": "application/vnd.api+json",
//     "Accept": "application/vnd.api+json"
//   },
//   body: JSON.stringify({
//     "data": {
//       "type": "users",
//       "attributes": {
//         firstName,
//         lastName,
//         email,
//         password
//       }
//     }
//   })
// })

// var { data: user, errors } = await response.json()
// if (errors) {
//   console.log(errors)
// }

// console.log(`Our user's name is: ${user.attributes.fullName}`)

// // … handle successful form submission
// }
async function chicken(token) {
	console.log(token)
	const response = await fetch("https://api.keygen.sh/v1/accounts/etude/licenses", {
		method: "POST",
		headers: {
			"Content-Type": "application/vnd.api+json",
			"Accept": "application/vnd.api+json",
			"Authorization": "Bearer " + token
		},
		body: JSON.stringify({
			"data": {
				"type": "licenses",
				"relationships": {
					"policy": {
						"data": {
							"type": "policies",
							"id": "1357f374-d1a0-4abf-b733-bbd67cea6b46"
						}
					},
					"user": {
						"data": {
							"type": "users",
							"id": "efdb1b05-1523-4c87-9a6b-74485fa4d245"
						}
					}
				}
			}
		})
	})

	const {
		data: license,
		errors
	} = await response.json()
	if (errors) {
		console.log(errors)
	}

	console.log(`Our license's key is: ${license.attributes.key}`)
}


const {
	machineId
} = require('node-machine-id')
const crypto = require('crypto')

const getFingerprint = async () => {
	const id = await machineId({
		original: true
	})

	return crypto.createHash('sha512')
		.update(id)
		.digest('hex')
}



async function checkMachine(token) {
	console.log(token)
	const response = await fetch("https://api.keygen.sh/v1/accounts/etude/machines", {
	  method: "POST",
	  headers: {
	    "Content-Type": "application/vnd.api+json",
	    "Accept": "application/vnd.api+json",
	    "Authorization": "Bearer " + token
	  },
	  body: JSON.stringify({
	    "data": {
	      "type": "machines",
	      "attributes": {
	        "fingerprint": await getFingerprint()
	      },
	      "relationships": {
	        "license": {
	          "data": { "type": "licenses", "id": "99a74193-da28-4e2a-bd43-0492ff54884c" }
	        }
	      }
	    }
	  })
	})


	const { data: machine, errors } = await response.json()
	if (errors) {
		console.log(errors)
	}

	console.log(`Machine activated: ${machine.attributes.fingerprint}`)
}

async function purple(token) {
	const validation = await fetch(`https://api.keygen.sh/v1/accounts/etude/licenses/99a74193-da28-4e2a-bd43-0492ff54884c/actions/validate`, {
		method: "POST",
		headers: {
			"Content-Type": "application/vnd.api+json",
			"Accept": "application/vnd.api+json",
			"Authorization": "Bearer " + token
		}
	})

	console.log(await validation.json())
	const {
		meta
	} = await validation.json()
	if (meta.valid) {
		console.log("VALID")
	} else {
		console.log("NOT VALID")
	}
}
async function apple() {
	const email = "etasaaah@gmail.com"
	const password = "asdfasdfasdf"
	const credentials = Buffer.from(`${email}:${password}`).toString('base64')
	const response = await fetch("https://api.keygen.sh/v1/accounts/etude/tokens", {
		method: "POST",
		headers: {
			"Content-Type": "application/vnd.api+json",
			"Accept": "application/vnd.api+json",
			"Authorization": `Basic ${credentials}`
		}
	})

	var {
		data: token,
		errors
	} = await response.json()
	if (errors) {
		console.log(errors)
	}
	checkMachine(token.attributes.token)
}


apple()
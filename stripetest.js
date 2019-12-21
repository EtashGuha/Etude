var stripe = require('stripe')('rk_live_pVDuyAoclBtFPPIWIZ8rHCl200kbPvuYWk');

stripe.subscriptions.retrieve(
  'sub_GOI4CLHCuuLxTj',
  function(err, subscription) {
    console.log(err)
    console.log(subscription)
  }
);
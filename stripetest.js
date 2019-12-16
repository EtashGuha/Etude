var stripe = require('stripe')('rk_live_pVDuyAoclBtFPPIWIZ8rHCl200kbPvuYWk');

stripe.subscriptions.retrieve(
  'pm_1FqAKELij5QWz8qqOE3lckDN',
  function(err, subscription) {
    console.log(err)
    console.log(subscription)
  }
);
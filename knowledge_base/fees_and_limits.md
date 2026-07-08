# Fees and Account Limits

## Transaction fees
Standard domestic payments incur a small percentage + fixed fee, deducted
automatically from the received amount. Fees are reported separately via the
reports tools (`get_fee_summary`) rather than netted silently out of
transaction amounts, so sellers can reconcile gross vs. net.

## Currency support
The account holds balances in multiple currencies (e.g. USD, EUR)
independently. Sending a payment in a currency you don't hold a balance in
will fail with a validation error rather than auto-converting.

## Payout batches
Payout batches let you send the same amount to many recipients in a single
call. Batches are processed asynchronously; check status with
`get_payout_batch_status` rather than assuming immediate completion.

## Withdrawal and funding limits
Adding or withdrawing funds from a linked bank account are both treated as
sensitive actions requiring explicit confirmation, since they move money
in or out of the PayPal balance itself (as opposed to between PayPal users).

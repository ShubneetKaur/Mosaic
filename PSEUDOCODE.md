# Prototype Pseudocode

```text
FOR each federated round:
    FOR each participating traffic location:
        collect local traffic features
        train local prediction model
        create model update
        send model update to central server

    central server aggregates updates using FedAvg
    evaluate global model performance
    record communication cost
    send global model back to locations

REPEAT for the configured number of rounds

use final global model for traffic-flow prediction
use predictions as input to emergency-route decision support
```

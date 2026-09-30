# Federated Traffic-Flow Prediction with Communication-Cost Evaluation

Prototype for PS-003. This demo simulates multiple traffic locations that train local models, send model updates to a central server, use Federated Averaging (FedAvg), measure communication cost, and generate traffic predictions.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Workflow

Traffic Data Collection -> Local Training -> Model Update -> Local/Edge Server -> Central Server -> Global Model -> Local Training -> Repeat -> Traffic Prediction -> Emergency Route Support -> Communication-Cost Evaluation

The prototype uses synthetic traffic data so it can run without private camera/GPS/mobile datasets or API keys. Real permitted datasets can later replace the data generator using the same feature structure.

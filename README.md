# ph-suggestion-engine
🚀 FastAPI Suggestion Engine for Portabel Health (PH)

This repository contains the FastAPI-based suggestion engine for the Portabel Health (PH) platform.
It powers the machine-learning–driven shift recommendations shown to nurses inside the PH App.

The service receives a nurse’s profile and a list of available shifts, scores them using an ML model, and returns a ranked list of “Suggested Shifts” to the main Spring Boot backend.

🧠 Why does this repo exist?

We created this separate FastAPI service to support the PH matching engine for several key reasons:

1. 🔍 Isolate all matching & ML logic

All recommendation algorithms, feature engineering, and ML models live here.

Keeps the Spring Boot backend clean and focused on core business workflows.

Allows the matching logic to evolve independently (e.g., new models, new ranking rules).

2. ⚡ Faster ML experimentation

Python + FastAPI is ideal for machine learning:

scikit-learn

XGBoost / LightGBM

Pandas / NumPy

We can iterate on models without touching or redeploying the Java backend.

3. 🔗 Clear API contract for recommendations

Spring Boot requests recommendations via endpoints like:

POST /score-shifts


The request includes:

Nurse profile & preferences

List of candidate shifts

The response returns:

List of shifts with ML-generated relevance scores

Sorted in descending “match” order

This clean contract isolates intelligence from business logic.

4. 📈 Independent scaling & deployment

Shift scoring can be CPU-intensive, especially once we evaluate:

Many nurses

Large sets of shifts

Heavy models

Running this as its own service allows:

Independent scaling

Blue/green ML model deployments

Lower risk to the main PH API

5. 🧩 Clear boundary between product & intelligence

PH App (Angular + Spring Boot) handles:

UX

Authentication

Shift management

Compliance

Payments / invoicing

This FastAPI service is the “brain” of the system that learns from:

Past applications

Completed assignments

Nurse preferences

Performance history

Its purpose is to improve:

Faster shift fill times

Higher match quality

Better retention

Higher satisfaction for nurses and hospitals

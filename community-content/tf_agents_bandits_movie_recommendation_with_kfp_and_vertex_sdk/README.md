# Building TF-Agents Bandits Based Movie Recommendation Systems using KFP and Vertex AI SDK

This directory demos the use of TF-Agents, Kubeflow Pipelines (KFP) and Vertex
AI in building reinforcement learning applications. The RL applications we build
are movie recommendation systems based on the MovieLens 100K dataset. We adopt
the contextual bandits formulation.

There are 2 demos:

1.  Step-by-step demo: showcase how to use custom training, custom
    hyperparameter tuning, custom prediction and endpoint deployment of Vertex
    AI to build a RL movie recommendation system

2.  End-to-end pipeline demo: showcase how to build a RL-specific MLOps pipeline
    using KFP and Vertex Pipelines, as well as additional Vertex AI and GCP
    services such as BigQuery, Cloud Functions, Cloud Scheduler, Pub/Sub.
    Highlights of this end-to-end pipeline are:

    -   RL-specific implementation for training and prediction
    -   Simulation for initial training data, prediction requests and
        re-training
    -   Closing of the feedback loop from prediction results back to training
    -   Customizable and reproducible KFP components

Each demo contains a notebook that carries out the full workflow and user
instructions, and a `src/` directory for Python modules and unit tests.

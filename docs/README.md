# Agent Platform Training Clusters Blog Series 

This directory contains deep-dive documentation, extended guides, and architectural references for Google Cloud Agent Platform Training Clusters.

## Contents
- **`vertex-training-cluster/`**: Documentation and setup guides for configuring and managing Agent Platform Training Clusters.

## Blog Posts
- [Model Distillation Best Practices](https://googlecloudplatform.github.io/vertex-ai-samples/vertex-training-cluster/model_distillation_best_practices): Explores off-policy model distillation, dataset curation, and hyperparameter scaling laws for training student models on Vertex AI.
- [Forgetting Mitigation via Data Mixing](https://googlecloudplatform.github.io/vertex-ai-samples/vertex-training-cluster/forgetting_mitigation_data_mixing): Discusses catastrophic forgetting in model fine-tuning and how to mitigate it using multi-domain data mixing on Vertex AI.
- [Multi-Turn Reinforcement Learning for τ²-bench](https://googlecloudplatform.github.io/vertex-ai-samples/vertex-training-cluster/multi_turn_reinforcement_learning_for_tau2_bench): Explores multi-turn RL training for tool-calling agents using GRPO on the τ²-bench customer service benchmark with NeMo RL.
- [Post-Training Conversational Agents via Self-Evolving Data Synthesis](https://googlecloudplatform.github.io/vertex-ai-samples/vertex-training-cluster/self_evolving_data_synthesis_for_conversational_agents): Introduces a self-evolving data synthesis pipeline and a post-training recipe that post-train Nemotron 3.5 models into near-SOTA conversational agents on the τ³-Banking benchmark.
- [An Open Recipe for Post-Training Frontier-Level Search Agents](https://googlecloudplatform.github.io/vertex-ai-samples/vertex-training-cluster/post_training_search_agents): Shows how to post-train Nemotron 3.5 Lightning into a near-frontier web search agent using synthetic data and rollouts collected entirely in a high-fidelity local environment.
- [Mixed RL vs. Multi-Teacher On-Policy Distillation for Multi-Task Post-Training](https://googlecloudplatform.github.io/vertex-ai-samples/vertex-training-cluster/mixed_rl_vs_mopd): Compares mixed reinforcement learning with multi-teacher on-policy distillation (MOPD) for post-training one Nemotron 3 Nano 4B model on math, code and instruction following, covering compute, cost and data-mixture findings.

# Kubeflow pipeline on Vertex

## Overview
Theses examples show the codes of KFP v2 with aiplatform SDK
* create endpoint
* train model
* deploy model

## Requirements
* Python 3.8
* Vertex on Google Cloud Platform

## Quickstart
* Compiler
  * use python to compiler example.py to example.json
* Run
  * Open Vertex on GCP
  * Click 'Pipelines' and create run
  * Click 'Upload file' to upload example.json
  * Select bucket then submit

## Cleanup
Following are optional to cleanup
* endpoint
  * Click 'Online prediction'
  * Click 'Actions' to delete endpoint
* model
  * Click 'Model Registry'
  * Click 'More' to delete model

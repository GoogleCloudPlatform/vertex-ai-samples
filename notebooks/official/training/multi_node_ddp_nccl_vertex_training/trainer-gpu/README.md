This `trainer-gpu` directory consists of the trainer code and files for creating a container image for the distributed PyTorch training job. 

This directory contains:

- `__init__.py`: To treat the directory as a package for dockerizing the application. 
- `Dockerfile`: Contains steps to create a container that runs the distributed training job.
- `main.py`: Contains the code for training job.
- `requirements.text`: Contains the package dependencies for training job.

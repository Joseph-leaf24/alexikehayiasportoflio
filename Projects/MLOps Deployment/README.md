<div align="center">
 <img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=0fcbab94-8fbe-4a38-93e8-c2348450a42e" />
  <h1 align="center">Root Analysis Toolkit (ROALT) 🌱</h1>
  <h3 align="center">Deep-learning toolkit bringing innovation to the field of plant phenotyping and segmentation.</h3>

![ROALT Cover](docs/_static/assets/cover.jpg)

[![Coverage](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-cv-1/blob/gh-pages/_static/coverage-badge.svg)](../../actions/workflows/full_ci_pipeline.yml)
[![PEP8](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-cv-1/blob/gh-pages/_static/pep8-badge.svg)](../../actions/workflows/full_ci_pipeline.yml)
[![Roalt API Image Build](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-cv-1/actions/workflows/docker-publish.yml/badge.svg?branch=test_docker_deployment_workflow)](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-cv-1/actions/workflows/docker-publish.yml)

</div>

## Table of Contents
- [Background](#background)
- [Features](#features)
- [Project Structure 📁](#project-structure-📁)
- [Installation 📦](#installation-📦)
  - [Requirements](#requirements)
  - [Quickstart (Docker)](#quickstart-docker)
  - [Poetry Environment](#poetry-environment)
- [Usage](#usage)
  - [Start the UI](#start-the-ui)
  - [Dockerized API Backend 🚀](#dockerized-api-backend-🚀)
  - [Gradio Web UI 📊](#gradio-web-ui-📊)
  - [UI Inference Instruction](#ui-inference-instruction)
- [Other 📚](#other-📚)
  - [Airflow Integration](#airflow-integration)
  - [Notebooks](#notebooks)
- [Contributing 🤝](#contributing-🤝)
- [Acknowledgments 🏆](#acknowledgments-🏆)
- [License](#license)

## Background

Root Analysis Toolkit (ROALT) leverages state-of-the-art deep learning models to automate root segmentation and analysis from plant images. It is designed for researchers and practitioners in plant phenotyping, providing both a user-friendly web interface and a robust API. 💻


## Features

- Deep learning-based root segmentation (ROSE) 🌊
- Gradio-powered web interface for easy interaction 📊
- FastAPI backend for programmatic access 🚀
- Airflow DAGs for scalable pipeline automation 💪
- Dockerized for easy deployment 🐳
- Example Jupyter notebooks 📚
- Supports batch and single-image processing 📈


## Project Structure 📁

```
Root-Analysis-Toolkit/
├── src/
│   ├── app/frontend/app.py      # Gradio web app
│   ├── roalt_api/app/main.py    # FastAPI backend
│   ├── main.py                  # Rich CLI tool with the
│   │                              whole pipeline functionality
│   └── ...
├── notebooks/                  # Example and utility notebooks
├── dev/models/weights/         # (Expected) model weights
├── docs/                       # Sphinx documentation
├── tests/                      # Test suite
└── ...
airflow/
├── dags/                       # Airflow DAGs, requirements, Docker
└── ...
```


## Installation 📦

### Requirements
- Python 3.12+
- [Poetry](https://python-poetry.org/) (recommended)
- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) (for containerized setup)

### Quickstart (Docker)

```sh
# Clone the repository
git clone https://github.com/your-org/2024-25d-fai2-adsai-group-cv-1.git
```

### Poetry Environment

```sh
# Install dependencies Base
poetry install

# OR Intall dependecies with API
poetry install --with api

# OR Intall dependecies with Frontend
poetry install --with frontend

# OR Intall dependecies with GPU
poetry install --with gpu

# OR Intall dependecies with API, Frontend and GPU
poetry install --with api,frontend,gpu

```

**CUDA and PyTorch**
To use the GPU for deep learning model inference, ensure that you have a compatible NVIDIA GPU and the CUDA Toolkit installed. Package uses PyTorch for model inference, so make sure you have the correct version of PyTorch installed that matches your CUDA version.

## Usage

### Start the UI
```sh
# Select the src folder
cd Root-Analysis-Toolkit/src

# Start the UI
poetry python -m app.frontend.app
```

### Dockerized API Backend 🚀

To run the containerized FastAPI backend (for use with the Gradio frontend or other clients), use:

```sh
docker compose -f ./Root-Analysis-Toolkit/src/roalt_api/compose.yaml up -d
```

This will start the API at [http://localhost:8080](http://localhost:8080/docs).

### OR Pull the image from the Docker Hub

Using this [link](https://hub.docker.com/r/denisbezpa/roalt_api)

### Gradio Web UI 📊

1. Open your browser at the provided local URL to interact with the web UI.

2. **To find the API URL:**
   - Go to the **Settings** menu in the Gradio interface.
   - Find the **API** field.
   - Write down the API URL displayed there.

![Settings](docs/_static/assets/ue/frontend-settings.png)

3. Come back to the inference menu and feel free to explore the options

## UI Inference Instruction

> Pre-requisite (API is running and URL is accessible)

Follow the guidelines on the screenshots

![Step 1](docs/_static/assets/ue/instruction_step-1.png)

After everything selected move on to inference

![Step 2](docs/_static/assets/ue/instruction_step-2.png)

## Other 📚

### Airflow Integration
- Airflow DAGs and requirements are in `airflow/dags/`
- Install Airflow and dependencies:
  ```sh
  pip install -r airflow/dags/requirements.txt
  ```
- Configure and run your Airflow instance as needed.

### Notebooks
- Example notebooks are available in `Root-Analysis-Toolkit/notebooks/` for interactive demos and testing.

## Contributing 🤝

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

Feel free to dive in! [Open an issue](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-cv-1/issues/new) or submit PRs.

## Acknowledgments 🏆

We would like to extend our sincere gratitude to NPEC for their invaluable contribution to this project. Their provision of essential data was instrumental in the development of ROALT, and this project would not have been possible without their support.

<p align="center">
  <img src="https://www.wur.nl/upload/024ad272-4061-4937-b069-84941529b82e_NPEC%20-%20Still%205.jpg" alt="NPEC Preview">
</p>

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

*This README follows the [standard-readme specification](https://github.com/RichardLitt/standard-readme).*

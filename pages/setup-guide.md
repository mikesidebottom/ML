---
layout: default
title: Setup Guide
permalink: /pages/setup-guide/
---

# Setup Guide 🛠️

## Recommended Platform: Google Colab

We recommend using [Google Colab](https://colab.research.google.com/) for this workshop as it provides a free, GPU-enabled environment.

### What You Need

* A Google account
* Reliable internet connection

### Running the Notebooks

1. Navigate to the [Workshop Sessions](/pages/workshop-sessions) page
2. Click the "Open in Colab" button for the notebook you want to run

### Enable GPU in Colab

For best performance, enable GPU acceleration:

1. **Runtime > Change runtime type**
2. Set **Hardware Accelerator** to `GPU`
3. Click **Save**

### Install Dependencies

Each notebook starts with a setup cell. Run it first to install all required libraries.

## Alternative: Local Setup

If you prefer running the notebooks locally:

1. Clone the repository:
```bash
git clone https://github.com/CLDiego/uom_fse_dl_workshop.git
cd UoM_fse_dl_workshop
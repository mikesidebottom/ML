---
layout: default
title: Setup Guide
permalink: /pages/setup-guide/
---

# SETUP GUIDE 🛠️

<div class="info-box">
  This guide walks you through setting up your environment for the deep learning workshop. For the best experience, we recommend using Google Colab as it provides free GPU acceleration, which is essential for running neural network models efficiently.
</div>

<div class="setup-card">
  <div class="setup-header">
    <h3>🚀 GETTING STARTED WITH GOOGLE COLAB</h3>
  </div>
  <div class="setup-content">
    <p>Google Colab provides a free, GPU-enabled Jupyter notebook environment that requires no setup.</p>
    
    <h4>WHAT YOU NEED</h4>
    <ul>
      <li>A Google account <i class="fab fa-google"></i></li>
        <li>Reliable internet connection</li>
        <li>Web browser (Chrome recommended)</li>
    </ul>
    
    <div class="progress-container">
      <div class="progress-bar" style="width: 100%"></div>
    </div>
  </div>
</div>

<div class="notebook-card">
  <div class="notebook-header">
    <h3>📓 RUNNING THE NOTEBOOKS</h3>
  </div>
  <div class="notebook-content">
    <ol>
      <li>Navigate to the <a href="{{ site.baseurl }}/pages/workshop-sessions">Workshop Sessions</a> page</li>
      <li>For each session, you'll find two Colab options:
        <ul>
          <li><strong>Code Along</strong> - Start with the exercise notebook</li>
          <li><strong>Solution</strong> - View the completed notebook with solutions</li>
        </ul>
      </li>
      <li>Click the <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="vertical-align: middle;"> button for your chosen notebook</li>
    </ol>
  </div>
</div>

<div class="warning-box">
  <strong>⚠️ IMPORTANT:</strong> Enable GPU acceleration for best performance! Neural network training will be significantly slower without it.
</div>

## ENABLING GPU ACCELERATION

<div class="card">
  <h4>📊 GPU SETUP STEPS</h4>
  <ol>
    <li>In your Colab notebook, select <strong>Runtime > Change runtime type</strong></li>
    <li>Set <strong>Hardware Accelerator</strong> to <code>GPU</code></li>
    <li>Click <strong>Save</strong></li>
  </ol>
  <img src="{{ site.baseurl }}/assets/images/colab-gpu-setup.png" alt="Colab GPU Setup" class="setup-image">
</div>

## INSTALLING DEPENDENCIES

<div class="card">
  <h4>📦 PACKAGE INSTALLATION</h4>
  <p>Each notebook starts with a setup cell that installs all required libraries:</p>
  
  <pre><code class="language-python">!wget -q --show-progress https://raw.githubusercontent.com/CLDiego/uom_fse_dl_workshop/main/colab_utils.txt -O colab_utils.txt
!wget -q --show-progress -x -nH --cut-dirs=3 -i colab_utils.txt</code></pre>
  
  <p><strong>Always run this cell first!</strong> The setup may take a minute to complete.</p>
</div>

<div class="info-box">
  <strong>📘 ADDITIONAL RESOURCES:</strong><br>
  <a href="https://colab.research.google.com/notebooks/basic_features_overview.ipynb" target="_blank">Colab Tips & Features</a> | 
  <a href="https://research.google.com/colaboratory/faq.html" target="_blank">Colab FAQ</a> |
  <a href="https://pytorch.org/tutorials/" target="_blank">PyTorch Tutorials</a>
</div>

<div class="card">
  <h3>NEED HELP? 🆘</h3>
  <p>If you encounter any setup issues:</p>
  <ul>
    <li>Ask a workshop instructor during the session</li>
  </ul>
</div>
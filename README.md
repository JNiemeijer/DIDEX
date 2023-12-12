![Fancy logo](assets/didex-light.png#gh-dark-mode-only)
![Fancy logo](assets/didex-dark.png#gh-light-mode-only)  

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/generalization-by-adaptation-diffusion-based/domain-generalization-on-gta-to-avg)](https://paperswithcode.com/sota/domain-generalization-on-gta-to-avg?p=generalization-by-adaptation-diffusion-based)  
### [Paper](https://arxiv.org/abs/2312.01850) 


 **Generalization by Adaptation: Diffusion-Based Domain Extension for Domain-Generalized Semantic Segmentation**<br>
[Joshua Niemeijer*](https://scholar.google.com/citations?user=SK0mAJ0AAAAJ&hl), [Manuel Schwonberg*](https://scholar.google.com/citations?user=eqsXwGIAAAAJ&hl), [Jan-Aike Termöhlen*](https://scholar.google.com/citations?user=LkhzlxIAAAAJ&hl), [Nico M. Schmidt](https://scholar.google.com/citations?user=Kaei5zsAAAAJ&hl), and [Tim Fingscheidt](https://scholar.google.com/citations?user=KDgUWRMAAAAJ&hl)<br>
Winter Conference on Applications of Computer Vision (WACV) 2024<br>
(* indicates equal contribution)

The full code will be published soon. 

## Installation
To utilize DIDEX please follow the following steps:

For the creation of the pseudo target domain we build on the following repos:
1. https://github.com/Stability-AI/stablediffusion.git
2. https://github.com/lllyasviel/ControlNet.git

For the adaptation to the pseudo target domain we utilize the following repo:
1. https://github.com/lhoyer/MIC.git

To utilize our code please set up the repos following the descriptions they provide.

## Diffusion-Based Domain Extension (Pseudo-Target Domain Generation)
To create the Pseudo target domains please utilize the scripts in the folder dataset_creation.

## Adaptation To Pseudo-Target Domain
To train the model for domain generalization please utilize the scripts in generalization_experiments

## Datasets
We used the dataset structure ...

## Evaluation

## BibTeX
```
@article{Niemeijer2023DIDEX,,
  author        = {Niemeijer, Joshua and Schwonberg, Manuel and Termöhlen, Jan-Aike and Schmidt, Nico M. and Fingscheidt, Tim},
  title         = {{Generalization by Adaptation: Diffusion-Based Domain Extension for Domain-Generalized Semantic Segmentation}},
  year          = {2023},
  month         = dec,
  pages         = {1--16},
  eprint        = {2312.01850},
  archivePrefix = {arXiv}
}
```




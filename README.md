# Clincally Labeled Contrastive Learning for OCT Biomarker Classification

***

This work was done in the [Omni Lab for Intelligent Visual Engineering and Science (OLIVES) @ Georgia Tech](https://ghassanalregib.info/). 
It has recently been accepted for publication in the IEEE Journal for Biomedical and Health Informatics!!
Feel free to check our lab's [Website](https://ghassanalregib.info/publications) 
and [GitHub](https://github.com/olivesgatech) for other interesting work!!!

***

K. Kokilepersaud, S. Trejo Corona, M. Prabhushankar, G. AlRegib, C. Wykoff, 
"Clinically Labeled Contrastive Learning for OCT Biomarker Classification," 
in IEEE Journal of Biomedical and Health Informatics, 2023, May. 15 2023.

## Abstract
This paper presents a novel positive and negative set selection
strategy for contrastive learning of medical images based on labels
that can be extracted from clinical data. In the medical field, there
exists a variety of labels for data that serve different purposes
at different stages of a diagnostic and treatment process. Clinical
labels and biomarker labels are two examples. In general, clinical
labels are easier to obtain in larger quantities because they are
regularly collected during routine clinical care, while biomarker
labels require expert analysis and interpretation to obtain. Within
the field of ophthalmology, previous work has shown that clinical
values exhibit correlations with biomarker structures that manifest
within optical coherence tomography (OCT) scans. We exploit this
relationship by using the clinical data as pseudo-labels for our
data without biomarker labels in order to choose positive and
negative instances for training a backbone network with a supervised contrastive loss. 
In this way, a backbone network learns a
representation space that aligns with the clinical data distribution
available. Afterward, we fine-tune the network trained in this
manner with the smaller amount of biomarker labeled data with
a cross-entropy loss in order to classify these key indicators of
disease directly from OCT scans. We also expand on this concept
by proposing a method that uses a linear combination of clinical
contrastive losses. We benchmark our methods against state of
the art self-supervised methods in a novel setting with biomarkers
of varying granularity. We show performance improvements by as
much as 5% in total biomarker detection AUROC.

## Visual Abstract

## Data

## Code Usage

Supervised Contrastive Learning Experiments on OCT Data

1. Go to starting directory and type:

export PYTHONPATH=$PYTHONPATH:$PWD

before doing any experiments.

2. Set training setup with run_script
3. Type bash run_script to run experiments
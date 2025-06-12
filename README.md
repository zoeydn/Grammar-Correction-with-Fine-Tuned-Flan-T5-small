## Dataset

The project uses the **Write & Improve + LOCNESS v2.1** dataset from the [BEA-2019 Shared Task](https://www.cl.cam.ac.uk/research/nl/bea2019st/#data). This dataset consists of learner-written English annotated with grammatical corrections, making it ideal for GEC tasks.

- Download: [BEA 2019 Shared Task Data](https://www.cl.cam.ac.uk/research/nl/bea2019st/#data)

## Cleaned Dataset Used

https://drive.google.com/file/d/1OmKUufOSmJNzq9Ki93jiMiUJpJ5P15Ur/view?usp=sharing//
Test set is split from the dev set in the original dataset (wi+locness/m2/ABCN.dev.gold.bea19.m2) using split_dev.py in the [cleaned dataset file](https://drive.google.com/file/d/1OmKUufOSmJNzq9Ki93jiMiUJpJ5P15Ur/view?usp=sharing)


## Trained Models

https://drive.google.com/file/d/1IPWX7amaxUb_jbu9gssZCY2J3SIFbyZa/view?usp=sharing

##  Evaluation

Evaluate model performance using:

- **F0.5 score**, which weights precision twice as much as recall.
- **GLEU score**, a GEC-specific metric adapted from BLEU for better alignment with human judgments.

📍 GLEU implementation adapted from:  
https://github.com/cnap/grammatical-error-correction

## Citation

@inproceedings{bryant-etal-2019-bea,
  title = "The {BEA}-2019 Shared Task on Grammatical Error Correction",
  author = "Bryant, Christopher and Felice, Mariano and Andersen, {\O}istein E. and Briscoe, Ted",
  booktitle = "Proceedings of the Fourteenth Workshop on Innovative Use of NLP for Building Educational Applications",
  month = jun,
  year = "2019",
  address = "Florence, Italy",
  publisher = "Association for Computational Linguistics",
  pages = "52--75",
  url = "https://aclanthology.org/W19-4406"
}

@inproceedings{napoles-etal-2015-ground,
  title = "Ground Truth for Grammatical Error Correction Metrics",
  author = "Napoles, Courtney and Sakaguchi, Keisuke and Post, Matt and Tetreault, Joel",
  booktitle = "Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
  month = jul,
  year = "2015",
  address = "Beijing, China",
  publisher = "Association for Computational Linguistics",
  pages = "588--593",
  url = "https://aclanthology.org/P15-2097"
}

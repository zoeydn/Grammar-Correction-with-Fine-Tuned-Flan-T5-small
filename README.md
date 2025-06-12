## Overview

This project improves English grammar correction using parameter-efficient fine-tuning (LoRA) on top of Flan-T5-small. The models are trained with Write & Improve + LOCNESS v2.1 dataset and evaluated using f0.5 score and gleu score.

## Dataset

The project uses the **Write & Improve + LOCNESS v2.1** dataset from the [BEA-2019 Shared Task](https://www.cl.cam.ac.uk/research/nl/bea2019st/#data). This dataset consists of learner-written English annotated with grammatical corrections.

- Download: [BEA 2019 Shared Task Data](https://www.cl.cam.ac.uk/research/nl/bea2019st/#data)

## Cleaned Dataset Used

https://drive.google.com/file/d/1OmKUufOSmJNzq9Ki93jiMiUJpJ5P15Ur/view?usp=sharing
- /train_dev_test/to_instruction.py was used to format the m2 file to a jsonl file with instruction, input and out
- Test set is split from the dev set in the original dataset (wi+locness/m2/ABCN.dev.gold.bea19.m2) using split_dev.py in the [cleaned dataset file](https://drive.google.com/file/d/1OmKUufOSmJNzq9Ki93jiMiUJpJ5P15Ur/view?usp=sharing)


## Models
e.g. fine-tune with lora:<br>
```bash
                  python3 scripts/fine_tune_lora.py \
                          --train_path /wi+locness/m2/ABC.train.gold.bea19.jsonl \
                          --dev_path /train_dev_test/new_dev.jsonl \
                          --output_dir /flan_t5_grammar_lora
```
[Trained Modesl](https://drive.google.com/file/d/1XK22QTXzbBQHBHKUK6hGLeCgO1nGmL3p/view?usp=sharing)

## Predictions
e.g. checkpoint-11000:<br>
```bash
                python3 predictions/fine_tuned_lora/lora_predictions.py \
                        --input_file /train_dev_test/new_test.jsonl \
                        --output_file predictions/fine_tuned_lora/tuned_predictions_11000.jsonl \
                        --checkpoint /flan_t5_grammar_lora/checkpoint-11000
```
## Evaluation

Evaluate model performance using:<br>
https://github.com/chrisjbryant/errant needs to be installed for F0.5 score

- **F0.5 score**, which weights precision twice as much as recall.
- **GLEU score**, a GEC-specific metric adapted from BLEU for better alignment with human judgments.

GLEU implementation adapted from: https://github.com/cnap/grammatical-error-correction

## Demo

Grammar correction model: https://70827dfebb7677c828.gradio.live
- This demo uses a Flan-T5-small model fine-tuned with LoRA for grammar correction on English sentences, loaded from the final checkpoint at step 11000.

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

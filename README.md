# PubBCRPredictor

## License

Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) (see License file).
This software is not to be used for commerical purposes.

Commercial users/for profit organisations can obtain a license from Cambridge Enterprise.

## Overview
The public antibody prediction module leverages the pre-trained **BCR-V-BERT** model to classi-fy heavy chain antibodies (binary classification) and predict light chain antibodies (regression). This feature allows for the identification of public antibodies using advanced transformer-based sequence analysis.

## Setup

### Python Dependency
The public antibody prediction module depends on the **BCR-V-BERT** Python package. Before running predictions, you need to install the required Python environment and dependencies as described in the [BCR-V-BERT README](https://github.com/ZhangLabTJU/BCR-V-BERT).

### Installation
 
To download our code, we recommend creating a clean conda environment with python v3.9, and you will also need to install PyTorch v2.0.0.

To use PubBCRPredictor, install via pip:
```bash
python setup.py install
```

### Input Requirements
The public antibody prediction models require specific input formats depending on the model. Below is a summary of input requirements for each model:

| Model       | Input Requirements |
| ----------- | ----------- | 
| Heavy chain classification (all CDRs)| CDR1-3 amino acid sequences and V gene of the heavy chain antibody |
| Heavy chain classification (CDR3 only)| CDR3 amino acid sequence and V gene of the heavy chain antibody  |
| Light chain regression (all CDRs)| CDR1-3 amino acid sequences and V gene of the light chain antibody|
| Light chain regression (CDR3 only)| CDR3 amino acid sequence and V gene of the light chain antibody  |

***Example Input Format***

| Column Name | Description |
| ----------- | ----------- |
| cdr1        | CDR1 amino acid sequence       |
| cdr2        | CDR1 amino acid sequence       |
| cdr3        | CDR1 amino acid sequence       |
| v_gene      | V gene identifier (e.g., “IGHV1-69”)       |

***Functions and Model Mapping***

fastBCR-p provides a unified function interface for all four models. The specific model to use depends on the model argument passed to the function:
```r
# Predict function interface
predict_public_antibody <- function(data, model = "cdrh", python_env = "r-py-env") {
  # data: Data frame with required input columns
  # model: Model type ("cdrh", "cdrh3", "cdrl", "cdrl3")
  # python_env: Python environment configured with BCR-V-BERT
  ...
}
```

| Model | model Argument | Prediction Type |
| ----------- | ----------- | ----------- |
| Heavy chain classification (all CDRs) | "cdrh"  | Binary classification |
| Heavy chain classification (CDR3 only)| "cdrh3" | Binary classification |
| Light chain regression (all CDRs)     | "cdrl"  | Regression |
| Light chain regression (CDR3 only)    | "cdrl3" | Regression |

### Usage

```python

from PubBCRPredictor import PubBCRPredictor_Runner, MLP
from BCR_V_BERT import BCR_V_BERT_Runner

BCR_V_BERT = BCR_V_BERT_Runner(model='cdrh')
public_runner = PubBCRPredictor_Runner(model='cdrh')
prob = public_runner.predict(feature)
public_runner.plot_metric(data['label'].values,prob.numpy())

cdrh1_seq = 'GYTFTGYW'
cdrh2_seq = 'ILPGSGST'
cdrh3_seq = 'ARDDYDGAWFAY'
vgene = 'IGHV1-9'

feature = BCR_V_BERT.embed(([cdrh1_seq+'|'+cdrh2_seq+'|'+cdrh3_seq],[vgene]))
prob = public_runner.predict(feature)
```

***Sample Data***
Sample datasets for testing the public antibody prediction module are available in the example_paired folder. These datasets include required columns such as cdr1, cdr2, cdr3, and v_gene for both heavy and light chain antibodies.

For detailed information about the BCR-V-BERT Python package and its capabilities, please refer to the [BCR-V-BERT README](https://github.com/ZhangLabTJU/BCR-V-BERT).

## Contact

Please contact jian_zhang@tju.edu.cn to report issues of for any questions.

## Acknowledgements

The sample data is downloaded from Observed Antibody Space:
> Olsen, T.H., Boyles, F., and Deane, C.M. (2022). Observed Antibody Space: A diverse database of cleaned, annotated, and translated unpaired and paired antibody sequences. Protein Sci 31, 141–146. https://doi.org/10.1002/pro.4205.


# ALC

Reference code for [ACL22](https://www.2022.aclweb.org/) paper -  [Answer-level Calibration for Free-form Multiple Choice Question
Answering](https://aclanthology.org/2022.acl-long.49/).

## Dependencies
The code was written with, or depends on:
* Python 3.8
* Pytorch 1.9.1
* Transformers 4.11.3

## Running the code
1. Create a virtualenv and install dependecies
      ```bash
      virtualenv -p python3.8 venv
      source env/bin/activate
      pip3 install -r requirements.txt
      ``` 
1. Set up the environment
      ```bash
      bash setup_env.sh
      ```
1. Run zero-shot experiment using
      ```bash
      bash run_zs.sh ${gpudev} ${dataname} ${split}
      ```
   Run k-shot experiment using
      ```bash
      bash run_fs.sh ${gpudev} ${dataname} ${split} {k}
      ```
   Valid datanames are COPA, commonsenseqa, mctaco, piqa, socialiqa, winogrande, arc_easy, arc_challenge, dream, swag and hendrycks_test.
   For hendrycks_test, additionally pass the category:
      ```bash
      bash run_zs.sh ${gpudev} ${dataname} ${split} "-data_config ${category}"
      ```   
      where the category is in humanities, social_sciences, STEM, other

## Citation
If you use this code, please consider citing:

[1] Sawan Kumar. 2022. Answer-level Calibration for Free-form Multiple Choice Question Answering. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 665–679, Dublin, Ireland. Association for Computational Linguistics.
[[bibtex](https://aclanthology.org/2022.acl-long.49.bib)]
 
## Contact
For any clarification, comments, or suggestions please create an issue or contact sawankumar@iisc.ac.in

# Ferret/Screenspot experiment log

## 07/26/24
When following the instructions to set up the ferret model on local machine (Mac M1), there is a 
bug
```{bash}
RuntimeError: Failed to import transformers.models.llama.modeling_llama because of the following error (look up to see its traceback):
cannot import name 'BUFSIZE' from 'numpy' (/Users/cheng/miniconda3/envs/ferret/lib/python3.10/site-packages/numpy/__init__.py)
```
This error is due to a [recent update of numpy to 2.0.0](https://github.com/microsoft/DeepSpeed/pull/5680). `pip install "numpy<2.0.0"` fixed the issue.

# Citation-based Innovation Motif Graph Structure Learning for Scholar Profiling
##
## Installation

Before running the code, you need to install the required dependencies. You can do this by running the following command:

```bash
pip install -r requirements.txt
```
## How to Run
```python
python main.py \
  --dataset_name NLP \
  --conv_name GIN \
  --PATH data_extend \
  --readout mean \
  --lr 0.001 \
  --tau 0.2 \
  --hidden_dim 8 \
  --set_epsilon 0.2 \
  --model_name GSL_motif \
  --epochs 200 \
  --weight_decay 0.00005 \
  --folds 10 \
  --beta 0.0001 \
  --lamb 0.0001
```



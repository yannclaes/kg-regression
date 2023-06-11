# Knowledge-Guided Additive Modeling for Supervised Regression
## Usage
Firstly, install all required dependencies
```
conda env create -f environment.yml  
```
### Sequential methods

```
python3 baselines.py [--model_name] [--tree_model] [--problem] [--filter_fa] [--extrapolation] [--dataset_id]
```
with:
* ```model_name```: ```fp_with_constant_then_trees``` or ```fp_with_constant_then_mlp```
* ```tree_model```: ```boosting``` or ```rf``` (only relevant for tree methods)
* ```problem```: ```friedman1```, ```corr_friedman1```, ```linear_data```, ```overlap_data```, ```power_plant```, ```concrete```
* ```filter_fa```: ```yes``` or ```no``` (h_a input filtering)
* ```extrapolation```: ```yes``` or ```no``` (only valid for ```power_plant``` and ```concrete```)
* ```dataset_id```: desired dataset (0->9)

### Alternate methods
For MLP, please run:
```
python3 aphynity.py [--problem] [--filter_fa] [--extrapolation] [--dataset_id]
```
with:
* ```problem```: ```friedman1```, ```corr_friedman1```, ```linear_data```, ```overlap_data```, ```power_plant```, ```concrete```
* ```filter_fa```: ```yes``` or ```no``` (h_a input filtering)
* ```extrapolation```: ```yes``` or ```no``` (only valid for ```power_plant``` and ```concrete```)
* ```dataset_id```: desired dataset (0->9)

For GB and RF, please run:
```
python3 aphynitrees.py [--tree_model] [--problem] [--filter_fa] [--extrapolation] [--dataset_id]
```
with:
* ```tree_model```: ```boosting``` or ```rf```
* ```problem```: ```friedman1```, ```corr_friedman1```, ```linear_data```, ```overlap_data```, ```power_plant```, ```concrete```
* ```filter_fa```: ```yes``` or ```no``` (h_a input filtering)
* ```extrapolation```: ```yes``` or ```no``` (only valid for ```power_plant``` and ```concrete```)
* ```dataset_id```: desired dataset (0->9)

### PD-based methods
To train h_k, please run:
```
python3 pdp_train_fp.py [--pdp_method] [--problem] [--fp_term] [--extrapolation] [--dataset_id] [--n_repeats]
```
with:
* ```pdp_method```: ```mlp```, ```boosting``` or ```rf```
* ```problem```: ```friedman1```, ```corr_friedman1```, ```linear_data```, ```overlap_data```, ```power_plant```, ```concrete```
* ```fp_term```: [```first```, ```second```, ```third```] for ```friedman1```, ```corr_friedman1``` otherwise ```first```
* ```extrapolation```: ```yes``` or ```no``` (only valid for ```power_plant``` and ```concrete```)
* ```dataset_id```: desired dataset (0->9)
* ```n_repeats```: number of h_k-h_a fittings

To train h_a, please run:
```
python3 pdp_train_fa.py [--pdp_method] [--fa_model] [--tree_model] [--problem] [--fp_term] [--filter_fa] [--extrapolation] [--dataset_id]
```
with:
* ```pdp_method```: ```mlp```, ```boosting``` or ```rf```
* ```fa_model```: ```trees``` or ```mlp```
* ```tree_model```: ```boosting``` or ```rf``` (only relevant for tree methods)
* ```problem```: ```friedman1```, ```corr_friedman1```, ```linear_data```, ```overlap_data```, ```power_plant```, ```concrete```
* ```fp_term```: [```first```, ```second```, ```third```] for ```friedman1```, ```corr_friedman1``` otherwise ```first```
* ```filter_fa```: ```yes``` or ```no``` (h_a input filtering)
* ```extrapolation```: ```yes``` or ```no``` (only valid for ```power_plant``` and ```concrete```)
* ```dataset_id```: desired dataset (0->9)

### Data-driven methods
```
python3 baselines.py [--model_name] [--tree_model] [--problem] [--filter_fa] [--extrapolation] [--dataset_id]
```
with:
* ```model_name```: ```mlp_only``` or ```trees_only```
* ```tree_model```: ```boosting``` or ```rf``` (only relevant for tree methods)
* ```problem```: ```friedman1```, ```corr_friedman1```, ```linear_data```, ```overlap_data```, ```power_plant```, ```concrete```
* ```filter_fa```: ```yes``` or ```no``` (h_a input filtering)
* ```extrapolation```: ```yes``` or ```no``` (only valid for ```power_plant``` and ```concrete```)
* ```dataset_id```: desired dataset (0->9)

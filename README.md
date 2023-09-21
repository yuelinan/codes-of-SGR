Official implementation "Learning from Shortcut: A Shortcut-guided Approach for Graph Rationalization".

## Data download
- Spurious-Motif: this dataset can be generated via `spmotif_gen/spmotif.ipynb` in [DIR](https://github.com/Wuyxin/DIR-GNN/tree/main). 
- [Graph-SST2](https://github.com/divelab/DIG/tree/main/dig/xgraph/datasets): this dataset can be downloaded [here](https://mailustceducn-my.sharepoint.com/personal/yhy12138_mail_ustc_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyhy12138%5Fmail%5Fustc%5Fedu%5Fcn%2FDocuments%2Fpaper%5Fwork%2FGNN%20Explainability%20Survey%2FSurvey%5FText2graph%2FGraph%2DSST2%2Ezip&parent=%2Fpersonal%2Fyhy12138%5Fmail%5Fustc%5Fedu%5Fcn%2FDocuments%2Fpaper%5Fwork%2FGNN%20Explainability%20Survey%2FSurvey%5FText2graph).
- Open Graph Benchmark (OGBG): this dataset can be downloaded when running student.sh.


## How to run SGR?

To train $SGR$ on OGBG dataset:

```python
sh student.sh
```

To train $SGR$ on Spurious-Motif dataset:

```python
!cd spmotif_codes
sh student.sh
```

To train $SGR$ on Graph-SST2 dataset:

```python
!cd sst_codes
sh student.sh
```



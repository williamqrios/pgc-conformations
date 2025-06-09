# pgc-conformations
AlphaFold2-generated conformations of PGC-1α N-terminal fragment (2-220)

Structures generated with AlphaFold2 running colabfold v1.5.5 locally ([localcolabfold](https://github.com/YoshitakaMo/localcolabfold)). 

### Commands 

**Deep MSA + Recycle** 
```bash 
colabfold_batch --model-type alphafold2_ptm --num-seeds 500 --amber --num-relax 500 --num-models 1 --templates --use-gpu-relax --num-recycle 20 --recycle-early-stop-tolerance 0.5 ./inputs/PGC1alpha_fragment.fasta ./outputs
```

**Deep MSA + Dropout** 
```bash
colabfold_batch --model-type alphafold2_ptm --num-seeds 500 --amber --num-relax 500 --num-models 1 --templates --use-gpu-relax --num-recycle 3 --use-dropout --recycle-early-stop-tolerance 0.5 ./inputs/PGC1alpha_fragment.fasta ./outputs
```

**Shallow MSA + Recycle** 
```bash 
colabfold_batch --model-type alphafold2_ptm --num-seeds 500 --amber --num-relax 500 --num-models 1 --templates --use-gpu-relax --num-recycle 20 --max-seq 16 --max-extra-seq 32 --recycle-early-stop-tolerance 0.5 ./inputs/PGC1alpha_fragment.fasta ./outputs
```

**Shallow MSA + Dropout** 
```bash 
colabfold_batch --model-type alphafold2_ptm --num-seeds 500 --amber --num-relax 500 --num-models 1 --templates --use-gpu-relax --num-recycle 3 --use-dropout --recycle-early-stop-tolerance 0.5 --max-seq 16 --max-extra-seq 32 ./inputs/PGC1alpha_fragment.fasta ./outputs
```
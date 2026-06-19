# Running on Kaggle (recommended over Colab free)

## Why Kaggle over Colab free
- 30 GPU hours/week — free, no credit card
- Sessions run up to 12 hours without disconnecting
- T4 (16GB) handles opus-mt-tc-big-en-fr at batch_size=32 fine
- Easy dataset upload for your TMX file

## Setup steps

### 1. Upload your TMX as a Kaggle dataset
1. kaggle.com → Datasets → New Dataset
2. Upload en_fr.tmx
3. Note the dataset path: `/kaggle/input/<your-dataset-name>/en_fr.tmx`
4. Update `TMX_PATH` in the script

### 2. Create a notebook
1. kaggle.com → Code → New Notebook
2. Settings → Accelerator → GPU T4 x2 (or single T4)
3. Settings → Internet → On (needed to download HuggingFace model)
4. Add your TMX dataset to the notebook

### 3. Run in the notebook
```python
# Cell 1: Install deps
!pip install -q transformers datasets sacrebleu sentencepiece ctranslate2 accelerate

# Cell 2: Copy and run the training script
# Paste finetune_en_fr.py content here, or:
!wget https://raw.githubusercontent.com/.../finetune_en_fr.py
exec(open("finetune_en_fr.py").read())
main()
```

### 4. Save outputs
After training, the notebook output directory persists:
```python
# Check outputs
import os
print(os.listdir("./ct2_model"))
# Should show: config.json, model.bin, shared_vocabulary.json, source.spm, target.spm
```

Save the notebook output as a new Kaggle dataset to download later.

## Expected training time on T4
| Dataset size | Time   |
|-------------|--------|
| 50k pairs   | ~2-3h  |
| 200k pairs  | ~6-8h  |
| 500k pairs  | ~12h   |

## After training
1. Download `ct2_model/` from Kaggle outputs
2. Place at e.g. `/models/en-fr/ct2_model/`
3. In mercury-ml `.env`: `MODEL_PATH=/models/en-fr/ct2_model`
4. Uncomment in requirements.txt:
   ```
   ctranslate2>=4.4.0
   sentencepiece>=0.2.0
   ```
5. Implement `_load()` and `translate()` in `app/model.py` (see comments there)

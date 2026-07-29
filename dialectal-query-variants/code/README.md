
## Data Processing and Indexing Pipeline

### Step 1: Extract Data from HTML Pages

Use the following command to convert HTML files into JSON shards:

```bash
python html_to_json_shards.py \
  --input_dir ArTest_judged_docs/ArTest_judged_docs \
  --out_dir /dialectal_query_variants \
  --docs_per_shard 2000
```

```bash
python html_to_json_shards_for_Dense.py \
  --input_dir /ArTest_judged_docs \
  --out_dir /dialectal_query_variants_for_dense \
  --docs_per_shard 2000
```
### Step 2: Index Documents Using Pyserini

```bash
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /dialectal_query_variants \
  --index index_dialectal_query_variants \
  --generator DefaultLuceneDocumentGenerator \
  --language ar \
  --threads 16 \
  --storePositions \
  --storeDocvectors \
  --storeRaw 
```

### Step 3: Create docs embeddings
```bash
python -m pyserini.encode \
  input --corpus /dialectal_query_variants_for_dense  \
  --fields text \
  output --embeddings embeddings/ar-e5-large/corpus \
  encoder --encoder intfloat/multilingual-e5-large \
  --fields text \
  --batch 128 \
  --fp16
```

### Step 4: #build faiss index
```bash
python -m pyserini.index.faiss \
  --input embeddings/ar-e5-large/corpus \
  --output faiss/ar-e5-large \
  --dim 1024 \
  --threads 16
```

## Retrieval Systems 

python lexical_retrieval.py

python dense_retrieval.py

python reranking.py

## Evaluation

python effectiveness_evaluation.py

python RBO_evaluation.py

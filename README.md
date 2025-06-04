#Important

Install using pip

```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn api-gd:app --port 8000
```

#Performance Using Grounding Dino

On CPU - around 21-30seconds/image

On GPU - around 2-3seconds/image
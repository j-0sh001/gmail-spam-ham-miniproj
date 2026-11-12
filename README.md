# Gmail Spam/Ham Detector (TF-IDF + Multinomial Naive Bayes)

Chrome extension (Manifest V3) that labels Gmail inbox rows as SPAM or HAM using a local Python model.

```
Gmail list view ──content.js──> background.js ──HTTP──> Flask /predict_batch
                                                         TF-IDF -> MultinomialNB
       red/green badge  <────────────── JSON {label, probabilities}
```

## Layout
```
train_model.py     train + evaluate + save model/ artifacts
app.py             Flask API (127.0.0.1:5000)
requirements.txt
extension/         manifest.json, background.js, content.js/.css, sidebar.html/.css/.js
```

## Setup

1. **Install dependencies** (Python 3.10+)
   ```
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Train the model** (downloads the SMS dataset, prints accuracy/precision/recall/F1/confusion matrix)
   ```
   python train_model.py
   ```
   Offline? Download `sms.tsv` from the URL in `train_model.py` and run `python train_model.py --data sms.tsv`.
3. **Start the API**
   ```
   python app.py
   ```
   Check: `curl http://127.0.0.1:5000/health`
4. **Load the extension**
   - Open `chrome://extensions`, enable **Developer mode**
   - **Load unpacked** -> select the `extension/` folder
   - Open `https://mail.google.com` (reload if already open). Badges appear before each subject.
   - Click the extension icon to open the side panel for manual testing.

## Notes
- Spam threshold defaults to 0.5. Change with `SPAM_THRESHOLD=0.7 python app.py` to trade recall for precision.
- The model is trained on SMS data, so it will be weaker on long emails. Retrain on an email corpus (e.g. Enron-Spam, SpamAssassin) for better results; `train_model.py --data` takes any `label<TAB>message` TSV.
- Only the subject and snippet are sent, only to `127.0.0.1`.
- Gmail's DOM class names are obfuscated. If badges vanish after a Gmail update, update the `SEL` selectors at the top of `content.js`.

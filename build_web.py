"""Keep the original ML artifacts private; publish only browser assets."""

from pathlib import Path
import hashlib
from shutil import copytree
from urllib.request import urlopen

root = Path(__file__).resolve().parent
model = root / "enhanced_sentiment_analysis_model.h5"
expected = "dc474c132178c83fc38c85cd643ffe8f08e0a56a45dbf5e80d12eb5606176c17"
if not model.exists() or model.stat().st_size < 1024:
    # Fetch the exact public Git LFS artifact at build time, never at inference time.
    url = "https://media.githubusercontent.com/media/Ennsss/Sentiment_Analysis-customer-feedback/eeb57a70cd3e8dc4d3723bb098c2262534a64394/enhanced_sentiment_analysis_model.h5"
    with urlopen(url, timeout=180) as source, model.open("wb") as target:
        total = 0
        while chunk := source.read(1024 * 1024):
            total += len(chunk)
            if total > 150 * 1024 * 1024:
                raise RuntimeError("Unexpected model size.")
            target.write(chunk)
if hashlib.sha256(model.read_bytes()).hexdigest() != expected:
    raise RuntimeError("Original model checksum mismatch; refusing deployment.")
copytree(root / "static", root / "public/static", dirs_exist_ok=True)

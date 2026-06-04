import re

def is_bad_chunk(text: str) -> bool:
  t = text.lower().strip()

  # ONLY hard-remove full reference sections
  if t.startswith("references") or t.startswith("bibliography"):
      return True

  # OCR / garbage detection
  if len(t) < 60:
      return True

  # extreme symbol-heavy chunks
  alpha_ratio = sum(c.isalpha() for c in t) / max(len(t), 1)
  if alpha_ratio < 0.4:
      return True

  # optional: only kill obvious citation blocks
  if t.count("doi:") > 0 and len(t) < 200:
      return True

  return False


def filter_nodes(nodes):
    return [n for n in nodes if not is_bad_chunk(n.text)]
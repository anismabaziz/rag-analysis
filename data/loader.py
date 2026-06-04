from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element
from typing import List, Dict, Set
from collections import Counter
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFLoader:
  """
  Robust PDF loader for research papers - parsing only, no chunking.
  Returns clean, structured elements ready for semantic chunking downstream.
  """
  
  def __init__(
      self, 
      infer_table_structure: bool = True,
      extract_images_in_pdf: bool = False,
      fallback_strategy: str = "auto"
      ):
    """
    Args:
        infer_table_structure: Extract HTML/structured tables
        extract_images_in_pdf: Extract figure captions (not images themselves)
        fallback_strategy: "auto", "fast", or "ocr_only"
    """
    self.infer_table_structure = infer_table_structure
    self.extract_images_in_pdf = extract_images_in_pdf
    self.fallback_strategy = fallback_strategy
    
    # Element types to keep
    self.KEEP_TYPES = {
        "Title",
        "Header",
        "Section-header", 
        "Subsection-header",
        "NarrativeText",
        "ListItem",
        "Table",
        "FigureCaption",
        "Formula",
        "Footnote",
        "PageBreak",  # Keep for boundary detection
    }
      
  def load(self, path: str) -> List[Dict]:
    """
    Load and parse PDF, return clean elements
    
    Args:
        path: Path to PDF file
        
    Returns:
        List of element dicts with text, type, and metadata
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"PDF not found: {path}")
        
    elements = self._partition_with_fallback(path)
    
    # Clean and filter
    cleaned = self._clean_elements(elements)
    
    # Add structural metadata
    cleaned = self._enrich_metadata(cleaned)
    
    return cleaned
  
  def _partition_with_fallback(self, path: str) -> List[Element]:
    """Try multiple strategies with fallback"""
    strategies = []
    
    if self.fallback_strategy == "auto":
        strategies = ["hi_res", "fast", "ocr_only"]
    elif self.fallback_strategy == "fast":
        strategies = ["fast", "hi_res"]
    else:
        strategies = [self.fallback_strategy]
        
    for strategy in strategies:
        try:
            logger.info(f"Trying strategy: {strategy}")
            elements = partition_pdf(
                filename=path,
                strategy=strategy,
                infer_table_structure=self.infer_table_structure,
                extract_images_in_pdf=self.extract_images_in_pdf,
            )
            if elements:
                logger.info(f"Success with {strategy}: {len(elements)} elements")
                return elements
        except Exception as e:
            logger.warning(f"Strategy {strategy} failed: {e}")
            continue
            
    raise RuntimeError(f"All partition strategies failed for {path}")
  
  def _clean_elements(self, elements: List[Element]) -> List[Dict]:
    """Filter, clean, and normalize elements"""
    # First pass: detect noise patterns across all elements
    repeated = self._detect_repeated_lines(elements)
    page_noise = self._detect_page_boundary_noise(elements)
    
    cleaned = []
    
    for element in elements:
        # Get text and type
        text = self._clean_text(self._get_element_text(element))
        element_type = self._get_element_type(element)
        
        # Filter by type
        if element_type not in self.KEEP_TYPES:
            continue
            
        # Filter empty or too short
        if not text or len(text.strip()) < 3:
            continue
            
        # Filter detected noise
        if text in repeated:
            continue
            
        if self._is_noise_text(text):
            continue
            
        if self._is_boundary_noise(text, page_noise):
            continue
        
        # Special handling: preserve table structure
        if element_type == "Table" and hasattr(element, 'metadata'):
            if hasattr(element.metadata, 'text_as_html') and element.metadata.text_as_html:
                text = element.metadata.text_as_html
        
        cleaned.append({
            "text": text,
            "type": element_type,
            "metadata": self._extract_metadata(element),
        })
        
    return cleaned
  
  def _enrich_metadata(self, elements: List[Dict]) -> List[Dict]:
    """Add useful metadata for downstream chunking"""
    # Track page changes
    current_page = None
    section_stack = []
    
    for elem in elements:
        page = elem.get('metadata', {}).get('page')
        
        # Add sequential index
        elem['index'] = elements.index(elem)
        
        # Add page change flag
        if page and page != current_page:
            elem['page_changed'] = True
            current_page = page
        else:
            elem['page_changed'] = False
        
        # Simple section tracking (can be enhanced)
        if elem['type'] in ['Title', 'Header', 'Section-header']:
            section_stack.append(elem['text'][:100])
            elem['is_section_boundary'] = True
        else:
            elem['is_section_boundary'] = False
            
        elem['current_section'] = section_stack[-1] if section_stack else None
        
    return elements
  
  def _get_element_text(self, element: Element) -> str:
    """Extract text from different element types"""
    if hasattr(element, 'text'):
        return element.text
    elif hasattr(element, '__str__'):
        return str(element)
    return ""
  
  def _get_element_type(self, element: Element) -> str:
    """Normalize element type names"""
    type_map = {
        "UncategorizedText": "NarrativeText",
        "Caption": "FigureCaption",
        "Table": "Table",
        "Formula": "Formula",
        "Footer": "Footnote",
        "Header": "Header",
    }
    
    raw_type = getattr(element, "category", "Unknown")
    return type_map.get(raw_type, raw_type)
  
  def _extract_metadata(self, element: Element) -> Dict:
    """Extract useful metadata for downstream processing"""
    meta = {
        'source': 'pdf',
    }
    
    if hasattr(element, 'metadata'):
        m = element.metadata
        
        # Page number - critical for RAG citation
        if hasattr(m, 'page_number'):
            meta['page'] = m.page_number
        elif hasattr(m, 'page_num'):
            meta['page'] = m.page_num
            
        # Coordinates (useful for layout understanding)
        if hasattr(m, 'coordinates'):
            meta['coordinates'] = m.coordinates
            
        # Filename
        if hasattr(m, 'filename'):
            meta['filename'] = m.filename
            
        # Element ID for potential linking
        if hasattr(m, 'element_id'):
            meta['element_id'] = m.element_id
            
    return meta
  
  def _clean_text(self, text: str) -> str:
    """Clean text while preserving meaning"""
    if not text:
        return ""
        
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text.strip())
    
    # Fix common PDF hyphenation artifacts
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"(\w)\s+-\s+(\w)", r"\1-\2", text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    return text
  
  def _is_noise_text(self, text: str) -> bool:
    """Filter noise content that shouldn't be retrieved"""
    if not text:
        return True
        
    text = text.strip()
    
    # Too short
    if len(text) < 3:
        return True
        
    # Too many symbols (not enough letters/spaces)
    alphanumeric = sum(c.isalnum() or c.isspace() for c in text)
    if alphanumeric / max(len(text), 1) < 0.3:
        return True
        
    # Noise patterns
    NOISE_PATTERNS = [
        r"^\d+$",  # page numbers alone
        r"^[ivxIVX]+\s*$",  # roman numerals
        r"^[\d\W]+$",  # mostly numbers and symbols
        r"^\s*$",  # empty
        r"^(www\.|http|doi:|arXiv:)",  # URLs/DOIs alone
        r"^©|®|™",  # copyright symbols alone
    ]
    
    for pattern in NOISE_PATTERNS:
        if re.match(pattern, text):
            return True
            
    # Check for over-trimmed text (starts/ends mid-sentence)
    if len(text) > 20 and not text[0].isupper() and text[0].isalpha():
        # Might be truncated, but keep it - downstream can handle
        pass
        
    return False
  
  def _detect_repeated_lines(self, elements: List[Element]) -> Set[str]:
    """Detect boilerplate text that appears too frequently"""
    counter = Counter()
    
    for element in elements:
        text = self._clean_text(self._get_element_text(element))
        # Only track medium-length text (likely boilerplate)
        if 10 < len(text) < 300:
            counter[text] += 1
            
    # Text that appears > 5 times OR > 8% of total elements
    threshold = max(5, int(len(elements) * 0.08))
    
    return {text for text, freq in counter.items() if freq >= threshold}
  
  def _detect_page_boundary_noise(self, elements: List[Element]) -> Dict[str, Set[str]]:
    """Detect headers/footers that repeat across pages"""
    page_content = {}
    
    # Group by page
    for element in elements:
        if hasattr(element, 'metadata'):
            page = getattr(element.metadata, 'page_number', None)
            if page:
                if page not in page_content:
                    page_content[page] = []
                text = self._clean_text(self._get_element_text(element))
                if 20 < len(text) < 200:  # Header/footer length range
                    page_content[page].append(text[:100])  # First 100 chars as signature
    
    # Find text that appears on most pages
    all_signatures = []
    for page, texts in page_content.items():
        all_signatures.extend([(text, page) for text in texts])
    
    signature_counter = Counter([sig for sig, _ in all_signatures])
    
    # If appears on > 70% of pages, it's likely a header/footer
    num_pages = len(page_content)
    threshold = max(3, int(num_pages * 0.7))
    
    return {
        'headers_footers': {sig for sig, freq in signature_counter.items() if freq >= threshold}
    }
  
  def _is_boundary_noise(self, text: str, page_noise: Dict) -> bool:
    """Check if text matches detected header/footer patterns"""
    if not page_noise:
        return False
        
    headers_footers = page_noise.get('headers_footers', set())
    text_sig = text[:100]
    
    return text_sig in headers_footers
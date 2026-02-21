# src/utils/data_loader.py
"""Dataset loader for multimodal jailbreak optimization"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class QueryDataset:
    """
    Dataset loader for initial queries
    
    Loads from JSONL file and provides image paths
    """
    
    def __init__(
        self,
        jsonl_path: Optional[str] = None,
        images_dir: Optional[str] = None,
        config = None
    ):
        """
        Initialize dataset
        
        Args:
            jsonl_path: Path to init_query.jsonl
            images_dir: Path to images directory
            config: Config object (if None, will use get_config())
        """
        if config is None:
            from src.utils.config_loader import get_config
            config = get_config()
        
        # Get paths from config or arguments
        self.jsonl_path = Path(jsonl_path or config.get('dataset.init_query_path'))
        self.images_dir = Path(images_dir or config.get('dataset.images_dir'))
        
        # Load data
        self.data = self._load_jsonl()
        
        logger.info(f"Loaded {len(self.data)} queries from {self.jsonl_path}")
    
    def _load_jsonl(self) -> List[Dict]:
        """Load JSONL file"""
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.jsonl_path}")
        
        data = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    
                    # Validate required fields
                    if 'image_id' not in item or 'query' not in item:
                        logger.warning(f"Line {line_num}: missing required fields, skipping")
                        continue
                    
                    data.append(item)
                
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: invalid JSON, skipping. Error: {e}")
        
        return data
    
    def __len__(self) -> int:
        """Number of queries in dataset"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get item by index
        
        Returns:
            {
                'image_id': str,
                'image_path': str (full path),
                'caption': str,
                'query': str
            }
        """
        item = self.data[idx].copy()
        
        # Add full image path
        item['image_path'] = str(self.images_dir / item['image_id'])
        
        return item
    
    def get_by_image_id(self, image_id: str) -> Optional[Dict]:
        """Get item by image_id"""
        for item in self.data:
            if item['image_id'] == image_id:
                result = item.copy()
                result['image_path'] = str(self.images_dir / image_id)
                return result
        return None
    
    def iter_items(self):
        """Iterate over all items"""
        for idx in range(len(self)):
            yield self[idx]
    
    def get_image_path(self, image_id: str) -> str:
        """Get full path for an image"""
        return str(self.images_dir / image_id)


# ========== Convenience Functions ==========

def load_dataset(config=None) -> QueryDataset:
    """
    Load dataset from config
    
    Usage:
        dataset = load_dataset()
        print(f"Loaded {len(dataset)} queries")
        
        item = dataset[0]
        print(item['image_path'])
        print(item['query'])
    """
    return QueryDataset(config=config)


def get_random_query(dataset: Optional[QueryDataset] = None, seed: int = 42):
    """
    Get a random query for testing
    
    Usage:
        item = get_random_query()
        print(item['query'])
    """
    import random
    
    if dataset is None:
        dataset = load_dataset()
    
    random.seed(seed)
    idx = random.randint(0, len(dataset) - 1)
    
    return dataset[idx]

def create_batch_iterator(dataset: Optional[QueryDataset] = None, batch_size: int = 10):
    """
    Create batch iterator for parallel processing
    
    Usage:
        dataset = load_dataset()
        for batch in create_batch_iterator(dataset, batch_size=5):
            for item in batch:
                # Process item
                print(item['query'])
    """
    if dataset is None:
        dataset = load_dataset()
    
    for i in range(0, len(dataset), batch_size):
        batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        yield batch
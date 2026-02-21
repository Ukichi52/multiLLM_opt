# src/storage/strategy_pool.py
import yaml
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class StrategyPool:
    """
    Strategy Pool: Manages all Master Policies and Sub-policies
    
    Responsibilities:
    1. Load YAML configuration files
    2. Provide quick query interfaces
    3. Maintain Master-Sub mapping relationships
    4. Validate configuration integrity
    
    Design Pattern: Repository Pattern
    - Isolate data access logic
    - Provide a unified query interface
    """
    
    def __init__(self, config_path: str = "config/strategies.yaml"):
        """
        Initialize the strategy pool
        
        Args:
            config_path: Path to the YAML configuration file
        
        Design Decisions:
        - Complete all initialization in the constructor (Fail-fast principle)
        - If there are issues with the config file, raise an error immediately instead of waiting until it's used
        """
        self.config_path = Path(config_path)
        
        # Load raw data
        self.raw_config = self._load_yaml()
        
        # Extract components
        self.master_policies = self.raw_config.get('master_policies', [])
        self.sub_policies = self.raw_config.get('sub_policies', [])
        self.mappings = self.raw_config.get('mappings', {})
        
        # Validate configuration
        self._validate_config()
        
        # Build indices (key to performance optimization)
        self._build_indices()
        
        logger.info(f"StrategyPool initialized: {len(self.master_policies)} masters, "
                   f"{len(self.sub_policies)} subs")
    
    def _load_yaml(self) -> Dict:
        """
        Load the YAML file
        
        Why extract this into a separate method?
        1. Facilitates unit testing (this method can be mocked)
        2. Facilitates extension (could load from a database in the future)
        3. Centralizes exception handling
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML format: {e}")
            raise ValueError(f"Failed to parse YAML: {e}")
    
    def _validate_config(self):
        """
        Validate the completeness and correctness of the configuration file
        
        Validation Rules:
        1. Must have three top-level fields: master_policies, sub_policies, mappings
        2. Each Master must have an id and name
        3. Each Sub must have an id, master_id, and prompt_template
        4. Sub IDs in mappings must exist in sub_policies
        5. The master_id of a Sub must exist in master_policies
        
        Why be so strict?
        - Discovering config errors at runtime is too costly (might have been running for hours)
        - Fail-fast: Discover issues at startup
        """
        # Validate top-level fields
        required_fields = ['master_policies', 'sub_policies', 'mappings']
        for field in required_fields:
            if field not in self.raw_config:
                raise ValueError(f"Missing required field in YAML: {field}")
        
        # Validate Master Policies
        master_ids = set()
        for master in self.master_policies:
            if 'id' not in master or 'name' not in master:
                raise ValueError(f"Master policy missing 'id' or 'name': {master}")
            master_ids.add(master['id'])
        
        # Validate Sub-policies
        sub_ids = set()
        for sub in self.sub_policies:
            required_sub_fields = ['id', 'master_id', 'prompt_template']
            for field in required_sub_fields:
                if field not in sub:
                    raise ValueError(f"Sub-policy missing '{field}': {sub.get('id', 'UNKNOWN')}")
            
            # Check if master_id exists
            if sub['master_id'] not in master_ids:
                raise ValueError(f"Sub-policy '{sub['id']}' references unknown master '{sub['master_id']}'")
            
            sub_ids.add(sub['id'])
        
        # Validate Mappings
        for master_id, sub_list in self.mappings.items():
            if master_id not in master_ids:
                raise ValueError(f"Mapping references unknown master: {master_id}")
            for sub_id in sub_list:
                if sub_id not in sub_ids:
                    raise ValueError(f"Mapping references unknown sub-policy: {sub_id}")
        
        logger.info("Configuration validation passed")
    
    def _build_indices(self):
        """
        Build indices to accelerate queries
        
        Index Types:
        1. master_by_id: {master_id -> master_dict}
        2. sub_by_id: {sub_id -> sub_dict}
        3. subs_by_master: {master_id -> [sub_dict, ...]}
        
        Time Complexity Analysis:
        - Building indices: O(n) where n is the total number of policies
        - Querying: O(1)
        - Memory overhead: Extra O(n) space (worth it for the speed tradeoff)
        
        Why not use a database?
        - Small number of policies (< 100)
        - No need for concurrent writes
        - In-memory indices are fast enough
        """
        # Index 1: Quick lookup of Master by ID
        self.master_by_id = {
            m['id']: m for m in self.master_policies
        }
        
        # Index 2: Quick lookup of Sub by ID
        self.sub_by_id = {
            s['id']: s for s in self.sub_policies
        }
        
        # Index 3: Quick lookup of all Subs by Master ID
        # This index is used for "give me all available Subs under this Master"
        self.subs_by_master = {}
        for master_id, sub_ids in self.mappings.items():
            self.subs_by_master[master_id] = [
                self.sub_by_id[sid] for sid in sub_ids 
                if sid in self.sub_by_id  # Defensive programming
            ]
    
    # ========== Public Query Interfaces ==========
    
    def get_master_policy(self, master_id: str) -> Optional[Dict]:
        """
        Get detailed information of a Master Policy
        
        Returns:
            Master dictionary, returns None if it does not exist
        
        Use Cases:
        - Display strategy information to the user
        - Logging
        """
        return self.master_by_id.get(master_id)
    
    def get_sub_policy(self, sub_id: str) -> Optional[Dict]:
        """
        Get detailed information of a Sub-policy
        
        This is the most frequently used interface!
        - Called by StrategicChooser to get the prompt after selecting a strategy
        - Mutator needs prompt_template to rewrite the query
        
        Returns:
            Sub dictionary, including prompt_template and other fields
        """
        return self.sub_by_id.get(sub_id)
    
    def get_sub_policies_for_master(self, master_id: str) -> List[Dict]:
        """
        Get all Sub-policies under a specific Master
        
        Use Cases:
        - StrategicChooser needs to select one from a Master's Subs
        - Analysis module tracks the usage of each Master
        
        Returns:
            List of Sub dictionaries, returns an empty list if Master does not exist
        """
        return self.subs_by_master.get(master_id, [])
    
    def get_all_master_ids(self) -> List[str]:
        """Get the IDs of all Master Policies"""
        return list(self.master_by_id.keys())
    
    def get_all_sub_ids(self) -> List[str]:
        """Get the IDs of all Sub-policies"""
        return list(self.sub_by_id.keys())
    
    def get_prompt_template(self, sub_id: str) -> str:
        """
        Get the Prompt template of a Sub-policy
        
        This is a convenience method
        Essentially shorthand for get_sub_policy(sub_id)['prompt_template']
        
        Why provide this method?
        - Caller only cares about the prompt, doesn't need the entire dictionary
        - Defensive programming: returns an empty string instead of raising an error if sub_id doesn't exist or lacks prompt_template
        """
        sub = self.get_sub_policy(sub_id)
        return sub.get('prompt_template', '') if sub else ''
    
    # ========== Future Extension Interfaces (Reserved) ==========
    
    def add_sub_policy(self, sub_dict: Dict) -> bool:
        """
        Dynamically add a Sub-policy (reserved interface)
        
        Current version: Not implemented (reading from YAML is sufficient)
        Future version: Implement this method if strategies need to be added dynamically at runtime
        
        Returning False indicates "feature not implemented", letting the caller know this is a placeholder
        """
        logger.warning("add_sub_policy not implemented yet")
        return False
    
    def remove_sub_policy(self, sub_id: str) -> bool:
        """Dynamically remove a Sub-policy (reserved interface)"""
        logger.warning("remove_sub_policy not implemented yet")
        return False


# ========== Helper Functions (Optional) ==========

def create_default_strategy_pool() -> StrategyPool:
    """
    Factory function: Creates a StrategyPool with default configuration
    
    Why is a factory function needed?
    - Simplifies caller's code (no need to specify the path)
    - Facilitates testing (this function can be mocked to return a test Pool)
    
    Usage example:
        pool = create_default_strategy_pool()
    """
    return StrategyPool("config/strategies.yaml")

import json
from transformers import AutoTokenizer
from config import TOKENIZER_NAME, MAX_CONTEXT_TOKENS

class ContextManager:
    def __init__(self):
        self.tokenizer = None
        
    def load_tokenizer(self):
        if not self.tokenizer:
            # We use use_fast=True if available, but gemma tokenizer might need standard instantiation
            self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
            
    def calculate_baseline_tokens(self, system_prompt: str, query: str, tool_schemas: list) -> int:
        self.load_tokenizer()
        
        # Combine base content to estimate baseline
        content_to_tokenize = system_prompt + "\n" + query + "\n"
        
        # Add tool schemas to the token string representation
        if tool_schemas:
            content_to_tokenize += json.dumps(tool_schemas)
            
        tokens = self.tokenizer.encode(content_to_tokenize)
        baseline = len(tokens)
        
        return baseline
        
    def get_dynamic_budget(self, baseline_tokens: int) -> int:
        return max(0, MAX_CONTEXT_TOKENS - baseline_tokens)
        
    def count_tokens(self, text: str) -> int:
        self.load_tokenizer()
        return len(self.tokenizer.encode(text))
        
    def truncate_to_budget(self, text: str, budget: int) -> str:
        self.load_tokenizer()
        if budget <= 0:
            return ""

        tokens = self.tokenizer.encode(text)
        if len(tokens) > budget:
            marker = "\n[TRUNCATED BY ORCHESTRATOR]"
            marker_tokens = self.tokenizer.encode(marker)

            # If the marker cannot fit, return the largest raw slice that fits.
            if len(marker_tokens) >= budget:
                return self.tokenizer.decode(tokens[:budget])

            content_budget = budget - len(marker_tokens)
            truncated_tokens = tokens[:content_budget]
            truncated_text = self.tokenizer.decode(truncated_tokens)
            candidate = truncated_text + marker

            # Decode/encode round-trips can vary slightly, so enforce hard ceiling.
            while truncated_tokens and len(self.tokenizer.encode(candidate)) > budget:
                truncated_tokens = truncated_tokens[:-1]
                truncated_text = self.tokenizer.decode(truncated_tokens)
                candidate = truncated_text + marker

            return candidate
        return text

# Global instance
context_manager = ContextManager()

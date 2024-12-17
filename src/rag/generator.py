"""Generator component for the RAG system."""

from typing import List, Dict
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList
)

class FinancialContextProcessor(LogitsProcessor):
    """Custom logits processor for financial context."""
    def __init__(self, financial_constraints: Dict):
        self.constraints = financial_constraints
        
    def __call__(self, input_ids: torch.LongTensor, 
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        # Apply financial domain constraints
        # This is a placeholder for actual constraints
        return scores

class RAGGenerator:
    def __init__(self, config: Dict):
        """Initialize the generator."""
        self.model_name = "gpt2"  # Can be configured based on needs
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.max_length = 512
        
    def prepare_context(self, retrieved_docs: List[Dict]) -> str:
        """Prepare context from retrieved documents."""
        context = ""
        for doc in retrieved_docs:
            context += f"{doc['document']['text']}\n"
        return context.strip()
        
    def generate(self, query: str, retrieved_docs: List[Dict], 
                financial_constraints: Dict = None) -> str:
        """Generate text based on query and retrieved documents."""
        context = self.prepare_context(retrieved_docs)
        prompt = f"Context: {context}\nQuery: {query}\nResponse:"
        
        # Prepare logits processors
        processors = LogitsProcessorList()
        if financial_constraints:
            processors.append(FinancialContextProcessor(financial_constraints))
            
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=self.max_length,
            num_return_sequences=1,
            logits_processor=processors,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


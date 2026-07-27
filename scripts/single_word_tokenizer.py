import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re


class SingleWordTokenizer:
    """
    Tokenizer for processing open response questions Q18.1 and Q19.
    Filters for single-word responses and generates standard embeddings.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize the tokenizer with a CSV file path.
        
        Args:
            csv_path: Path to the Qualtrics export CSV file
        """
        self.csv_path = csv_path
        self.df = None
        self.tokenized_responses = {
            'Q18.1': [],  # List of lists, each containing words for one participant
            'Q19': []     # List of lists, each containing words for one participant
        }
        
    def load_data(self) -> None:
        """Load the CSV data into a pandas DataFrame."""
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded data with shape: {self.df.shape}")
        
    def tokenize_response(self, text: str) -> List[str]:
        """
        Tokenize a text response into individual words.
        
        Args:
            text: Input text string
            
        Returns:
            List of individual words
        """
        if pd.isna(text) or not isinstance(text, str):
            return []
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Skip question text or metadata
        if 'Describe your ideal way' in text or 'If a friend told you' in text:
            return []
        if 'ImportId' in text or text == '':
            return []
            
        # Split by whitespace and return words
        words = text.split()
        return words
        
    def extract_and_tokenize_responses(self) -> Dict[str, List[List[str]]]:
        """
        Extract and tokenize responses from Q18.1 and Q19 columns.
        Each participant's response is tokenized into individual words.
        
        Returns:
            Dictionary with column names as keys and lists of tokenized responses (each response is a list of words)
        """
        if self.df is None:
            self.load_data()
            
        # Process Q18.1 (column AK)
        if 'Q18.1' in self.df.columns:
            q18_responses = self.df['Q18.1'].dropna()
            self.tokenized_responses['Q18.1'] = [
                self.tokenize_response(resp) for resp in q18_responses
            ]
            # Remove empty tokenizations
            self.tokenized_responses['Q18.1'] = [
                tokens for tokens in self.tokenized_responses['Q18.1'] if tokens
            ]
            total_words = sum(len(tokens) for tokens in self.tokenized_responses['Q18.1'])
            print(f"Tokenized {len(self.tokenized_responses['Q18.1'])} responses in Q18.1 ({total_words} total words)")
        else:
            print("Warning: Q18.1 column not found in CSV")
            
        # Process Q19 (column AL)
        if 'Q19' in self.df.columns:
            q19_responses = self.df['Q19'].dropna()
            self.tokenized_responses['Q19'] = [
                self.tokenize_response(resp) for resp in q19_responses
            ]
            # Remove empty tokenizations
            self.tokenized_responses['Q19'] = [
                tokens for tokens in self.tokenized_responses['Q19'] if tokens
            ]
            total_words = sum(len(tokens) for tokens in self.tokenized_responses['Q19'])
            print(f"Tokenized {len(self.tokenized_responses['Q19'])} responses in Q19 ({total_words} total words)")
        else:
            print("Warning: Q19 column not found in CSV")
            
        return self.tokenized_responses
        
    def create_vocabulary(self) -> Dict[str, int]:
        """
        Create a vocabulary from all tokenized responses.
        
        Returns:
            Dictionary mapping words to unique indices
        """
        all_words = []
        for column, tokenized_responses in self.tokenized_responses.items():
            for response_tokens in tokenized_responses:
                all_words.extend([word.strip().lower() for word in response_tokens])
            
        # Create unique vocabulary
        unique_words = sorted(set(all_words))
        vocabulary = {word: idx for idx, word in enumerate(unique_words)}
        
        print(f"Created vocabulary with {len(vocabulary)} unique words")
        return vocabulary
        
    def create_one_hot_embeddings(self, vocabulary: Dict[str, int]) -> Dict[str, np.ndarray]:
        """
        Create one-hot embeddings for the vocabulary.
        
        Args:
            vocabulary: Dictionary mapping words to indices
            
        Returns:
            Dictionary mapping words to one-hot vectors
        """
        vocab_size = len(vocabulary)
        embeddings = {}
        
        for word, idx in vocabulary.items():
            one_hot = np.zeros(vocab_size)
            one_hot[idx] = 1
            embeddings[word] = one_hot
            
        print(f"Created one-hot embeddings with dimension: {vocab_size}")
        return embeddings
        
    def create_frequency_embeddings(self, vocabulary: Dict[str, int]) -> Dict[str, float]:
        """
        Create frequency-based embeddings (TF-like scores).
        
        Args:
            vocabulary: Dictionary mapping words to indices
            
        Returns:
            Dictionary mapping words to frequency scores
        """
        # Count word frequencies
        word_counts = {}
        for column, tokenized_responses in self.tokenized_responses.items():
            for response_tokens in tokenized_responses:
                for word in response_tokens:
                    word_lower = word.strip().lower()
                    word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
        
        # Normalize by total count
        total_count = sum(word_counts.values())
        frequency_embeddings = {
            word: count / total_count 
            for word, count in word_counts.items()
        }
        
        print(f"Created frequency embeddings for {len(frequency_embeddings)} words")
        return frequency_embeddings
        
    def process(self, embedding_type: str = 'one_hot') -> Tuple[Dict[str, List[List[str]]], Dict, Dict]:
        """
        Complete processing pipeline: load data, tokenize responses, 
        and create embeddings.
        
        Args:
            embedding_type: Type of embeddings ('one_hot' or 'frequency')
            
        Returns:
            Tuple of (tokenized_responses, vocabulary, embeddings)
        """
        print("Starting tokenizer processing...")
        
        # Load and tokenize
        self.extract_and_tokenize_responses()
        
        # Create vocabulary
        vocabulary = self.create_vocabulary()
        
        # Create embeddings based on type
        if embedding_type == 'one_hot':
            embeddings = self.create_one_hot_embeddings(vocabulary)
        elif embedding_type == 'frequency':
            embeddings = self.create_frequency_embeddings(vocabulary)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
            
        print("Processing complete!")
        return self.tokenized_responses, vocabulary, embeddings


def main():
    """Example usage of the SingleWordTokenizer."""
    
    # Path to the CSV file (absolute path)
    csv_path = 'c:/Users/grant/Projects_PSYCH_755/final_project/DAGE/data_org/Excerpt For Testing Qualtrics Export.csv'
    
    # Initialize tokenizer
    tokenizer = SingleWordTokenizer(csv_path)
    
    # Process with one-hot embeddings
    responses, vocabulary, embeddings = tokenizer.process(embedding_type='one_hot')
    
    # Print results
    print("\n" + "="*50)
    print("TOKENIZED RESPONSES")
    print("="*50)
    for column, tokenized_list in responses.items():
        print(f"\n{column}:")
        print(f"  Number of responses: {len(tokenized_list)}")
        print(f"  Sample tokenized responses (first 3):")
        for i, tokens in enumerate(tokenized_list[:3]):
            print(f"    Response {i+1}: {tokens}")
    
    print("\n" + "="*50)
    print("VOCABULARY")
    print("="*50)
    print(vocabulary)
    
    print("\n" + "="*50)
    print("EMBEDDINGS (sample)")
    print("="*50)
    for word, vector in list(embeddings.items())[:5]:
        print(f"{word}: {vector}")


if __name__ == '__main__':
    main()

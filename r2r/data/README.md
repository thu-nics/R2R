## Objectives

1. Data Processing & Filtering:
   - Load the CSV data
   - Filter cases where 1.5B (small) predictions != 32B (reference) predictions
   - Save mismatches to a new CSV

2. Context Extraction for Mismatches:
   - For each mismatch:
     - Get sentence_id to identify the data item/paragraph
     - Extract context: real tokens from the same sentence, from last mismatch to current mismatch
     - Record both predictions (small and reference) at the mismatch point

3. Generation Framework:
   - Let both models (small and reference) continue generating from their respective trajectories
   - Generate until the end of a sentence
   - Need to handle KV-cache for efficient generation
   - Custom EOS tokens including "." for sentence endings

4. Verifing System:
   - Use a  LLM to compare the two generated continuations
   - Determine if they convey the same meaning
   - Label and record the results

## Data Structure

The input CSV contains the following columns:
- `row_id`: Unique identifier for each row in the dataset
- `*_token_id`: Unique identifier for each token in the sentence
- `*_sentence_id`: Unique sentence identifier
- `*_real_token`: The actual token id at this position
- `*_predictions`: Model predictions for different model sizes (1.5B, 7.0B, 14.0B, 32.0B)
- `*_entropy`: Entropy values for each model's predictions
- `*_AU`, `*_EU`: Additional metrics for AU (Area Under) and EU (Expected Utility)

where * represents different model sizes (1.5, 7.0, 14.0, 32.0)

## Code structure

### 1. Data Processing Module
```python
class DataProcessor:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        
    def find_mismatches(self):
        # Compare small vs reference predictions
        # Structure: {
        #   'sentence_id': xxx,
        #   'position': xxx,
        #   'token_id': xxx,
        #   'real_token': xxx,
        #   'pred_small': xxx,
        #   'pred_reference': xxx,
        #   'pred_small_token_id': xxx,
        #   'pred_reference_token_id': xxx
        # }
```

### 2. Mismatch Context Extractor
```python
class MismatchContext:
    def __init__(self, sentence_id, position):
        self.sentence_id = sentence_id
        self.current_position = position
        self.context_tokens = []      # real tokens from the same sentence (from last mismatch to current)
        self.context_token_ids = []   # corresponding token IDs of real tokens
        self.target_token = None      # current real token at mismatch point
        self.target_token_id = None   # current token ID at mismatch point
        self.pred_small = None
        self.pred_small_token_id = None
        self.pred_reference = None
        self.pred_reference_token_id = None
```

### 3. Generation Controller
```python
class ModelController:
    def __init__(self):
        self.small_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        self.reference_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        
        # Custom EOS tokens
        self.eos_tokens = {'.', '!', '?', '\n'}
        
    def generate_continuation(self, context, model_type, past_key_values=None):
        # Generate with KV cache
        model = self.small_model if model_type == 'small' else self.reference_model
        outputs = model.generate(
            input_ids=context,
            past_key_values=past_key_values,
            generation_config=gen_config,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True
        )
        
        return {
            'generated_tokens': outputs.sequences,
            'generated_token_ids': outputs.scores,
            'kv_cache': outputs.past_key_values
        }
```

### 4. Generation State Tracker
```python
class GenerationState:
    def __init__(self):
        self.current_kv_cache_small = None
        self.current_kv_cache_reference = None
        self.generated_tokens_small = []
        self.generated_token_ids_small = []
        self.generated_tokens_reference = []
        self.generated_token_ids_reference = []
        
    def update_state(self, model_type, generation_output):
        # Update KV cache and generated sequences for respective model
        if model_type == 'small':
            self.current_kv_cache_small = generation_output['kv_cache']
            self.generated_tokens_small.extend(generation_output['generated_tokens'])
            self.generated_token_ids_small.extend(generation_output['generated_token_ids'])
        else:
            self.current_kv_cache_reference = generation_output['kv_cache']
            self.generated_tokens_reference.extend(generation_output['generated_tokens'])
            self.generated_token_ids_reference.extend(generation_output['generated_token_ids'])
```

### 5. Main Workflow
```python
# Initialize components
processor = DataProcessor('data_analysis_AU_EU_top10.csv')
gen_controller = ModelController()
state_tracker = GenerationState()

# Process mismatches
for sentence_group in processor.group_by_sentence():
    # Reset state for new sentence
    state_tracker = GenerationState()
    
    for mismatch in sentence_group:
        # Extract context with token IDs
        context = MismatchContext(mismatch.sentence_id, mismatch.position)
        
        # Generate with KV cache
        cont_small = gen_controller.generate_continuation(
            context.context_token_ids, 
            'small',
            past_key_values=state_tracker.current_kv_cache_small
        )
        cont_reference = gen_controller.generate_continuation(
            context.context_token_ids,
            'reference',
            past_key_values=state_tracker.current_kv_cache_reference
        )
        
        # Update generation state
        state_tracker.update_state('small', cont_small)
        state_tracker.update_state('reference', cont_reference)
        
        # Compare and store results with token IDs
        store_results(context, cont_small, cont_reference)
```

### 6. Enhanced Output Format
```python
class VerifyModel:
    def __init__(self):
        # Initialize the  model (e.g., GPT-4 or other LLM)
        self.model = None
        self.prompt_template = """
        Compare these two sentence continuations and determine if they convey the same meaning:
        Context: {context}
        Small model continuation: {small_cont}
        Reference model continuation: {ref_cont}
        
        Analyze:
        1. Core meaning comparison
        2. Key differences (if any)
        3. Semantic equivalence
        
        Verdict (True/False):
        Explanation:
        """
    
    def (self, context, small_cont, ref_cont):
        prompt = self.prompt_template.format(
            context=context,
            small_cont=small_cont,
            ref_cont=ref_cont
        )
        response = self.get_judgment(prompt)
        return response.verdict, response.explanation

class ComparisonResult:
    def __init__(self):
        # Input context
        self.sentence_id = None
        self.original_context = None
        self.original_context_ids = None
        self.mismatch_position = None
        
        # Model predictions at mismatch point
        self.pred_small = None
        self.pred_small_id = None
        self.pred_reference = None
        self.pred_reference_id = None
        
        # Generated continuations
        self.continuation_small = None
        self.continuation_small_ids = None
        self.continuation_reference = None
        self.continuation_reference_ids = None
        
        # Judgment results
        self.meaning_match = None
        self.verify_explanation = None
        self.verify_confidence = None
        self.key_differences = None
        
    def to_dict(self):
        # Convert to dictionary for easy saving to JSON/CSV
        return {
            'sentence_id': self.sentence_id,
            'context': self.original_context,
            'context_ids': self.original_context_ids,
            'mismatch_position': self.mismatch_position,
            'pred_small': self.pred_small,
            'pred_small_id': self.pred_small_id,
            'pred_reference': self.pred_reference,
            'pred_reference_id': self.pred_reference_id,
            'continuation_small': self.continuation_small,
            'continuation_small_ids': self.continuation_small_ids,
            'continuation_reference': self.continuation_reference,
            'continuation_reference_ids': self.continuation_reference_ids,
            'meaning_match': self.meaning_match,
            'verify_explanation': self.verify_explanation,
            'verify_confidence': self.verify_confidence,
            'key_differences': self.key_differences
        }
```

class BeSpokeStratosDataset:
    def code_filter(self, example):
        question = next((item['value'] for item in example['conversations'] if item['from'] == 'user'), None)
        prefix = question[:10] if question else ''
        return prefix == 'Return you'
    
    def math_filter(self, example):
        question = next((item['value'] for item in example['conversations'] if item['from'] == 'user'), None)
        prefix = question[:10] if question else ''
        return prefix == 'Generate a'
    
    def qa_filter(self, example):
        question = next((item['value'] for item in example['conversations'] if item['from'] == 'user'), None)
        prefix = question[:10] if question else ''
        return prefix not in ['Return you', 'Generate a']    
    
    def filter_dataset(self, dataset, filter_name):
        if filter_name == 'math_filter':
            return dataset.filter(self.math_filter)
        elif filter_name == 'code_filter':
            return dataset.filter(self.code_filter)
        elif filter_name == 'qa_filter':
            return dataset.filter(self.qa_filter)
        else:
            raise ValueError(f"Filter name {filter_name} not found")

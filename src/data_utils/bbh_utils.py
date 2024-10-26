import json


def load_bbh_config():

    subsets = {
        'boolean_expressions': r'(True|False)',
        'causal_judgement': r'(Yes|No)',
        'date_understanding': r'[A-F]',
        'disambiguation_qa': r'[A-C]',
        'formal_fallacies': r'(invalid|valid)',
        'geometric_shapes': r'[A-K]',
        'hyperbaton': r'[A-B]',
        'logical_deduction_five_objects': r'[A-E]',
        'logical_deduction_seven_objects': r'[A-G]',
        'logical_deduction_three_objects': r'[A-C]',
        'movie_recommendation': r'[A-E]',
        'navigate': r'(Yes|No)',
        'penguins_in_a_table': r'[A-E]',
        'reasoning_about_colored_objects': r'[A-R]',
        'ruin_names': r'[A-D]',
        'salient_translation_error_detection': r'[A-F]',
        'snarks': r'[A-B]',
        'sports_understanding': r'(yes|no)',
        'temporal_sequences': r'[A-D]',
        'tracking_shuffled_objects_five_objects': r'[A-E]',
        'tracking_shuffled_objects_seven_objects': r'[A-G]',
        'tracking_shuffled_objects_three_objects': r'[A-C]',
        'web_of_lies': r'(Yes|No)'
    }

    label_space_map = {
        'boolean_expressions': ['True', 'False'],
        'causal_judgement': ['Yes', 'No'],
        'date_understanding': ['A', 'B', 'C', 'D', 'E', 'F'],
        'disambiguation_qa': ['A', 'B', 'C'],
        'formal_fallacies': ['invalid', 'valid'],
        'geometric_shapes': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'],
        'hyperbaton': ['A', 'B'],
        'logical_deduction_five_objects': ['A', 'B', 'C', 'D', 'E'],
        'logical_deduction_seven_objects': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'logical_deduction_three_objects': ['A', 'B', 'C'],
        'movie_recommendation': ['A', 'B', 'C', 'D', 'E'],
        'navigate': ['Yes', 'No'],
        'penguins_in_a_table': ['A', 'B', 'C', 'D', 'E'],
        'reasoning_about_colored_objects': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                                            'P', 'Q', 'R'],
        'ruin_names': ['A', 'B', 'C', 'D'],
        'salient_translation_error_detection': ['A', 'B', 'C', 'D', 'E', 'F'],
        'snarks': ['A', 'B'],
        'sports_understanding': ['yes', 'no'],
        'temporal_sequences': ['A', 'B', 'C', 'D'],
        'tracking_shuffled_objects_five_objects': ['A', 'B', 'C', 'D', 'E'],
        'tracking_shuffled_objects_seven_objects': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'tracking_shuffled_objects_three_objects': ['A', 'B', 'C'],
        'web_of_lies': ['Yes', 'No']
    }
    is_choices = {
        'date_understanding': ['A', 'B', 'C', 'D', 'E', 'F'],
        'disambiguation_qa': ['A', 'B', 'C'],
        'geometric_shapes': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'],
        'hyperbaton': ['A', 'B'],
        'logical_deduction_five_objects': ['A', 'B', 'C', 'D', 'E'],
        'logical_deduction_seven_objects': ['A', 'B', 'C', 'D', 'E'],
        'logical_deduction_three_objects': ['A', 'B', 'C', 'D', 'E'],
        'movie_recommendation': ['A', 'B', 'C', 'D', 'E'],
        'penguins_in_a_table': ['A', 'B', 'C', 'D', 'E'],
        'reasoning_about_colored_objects': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                                            'P', 'Q', 'R'],
        'ruin_names': ['A', 'B', 'C', 'D'],
        'salient_translation_error_detection': ['A', 'B', 'C', 'D', 'E', 'F'],
        'snarks': ['A', 'B'],
        'temporal_sequences': ['A', 'B', 'C', 'D'],
        'tracking_shuffled_objects_five_objects': ['A', 'B', 'C', 'D', 'E'],
        'tracking_shuffled_objects_seven_objects': ['A', 'B', 'C', 'D', 'E'],
        'tracking_shuffled_objects_three_objects': ['A', 'B', 'C', 'D', 'E']
    }
    
    with open("data/bbh/bbh_task_description.json", "r") as f:
        task_description = json.load(f)
        
    task_type = {
        "bool": ["boolean_expressions", "formal_fallacies", "causal_judgement", "navigate", "sports_understanding", "web_of_lies"],
        "multiple_choice": ["date_understanding", "disambiguation_qa", "geometric_shapes", "hyperbaton", "logical_deduction_five_objects", "logical_deduction_seven_objects", "logical_deduction_three_objects", "movie_recommendation", "penguins_in_a_table", "reasoning_about_colored_objects", "ruin_names", "salient_translation_error_detection", "snarks", "temporal_sequences", "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects"],
    }
        
    return subsets, label_space_map, is_choices, task_description, task_type
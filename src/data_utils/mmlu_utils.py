def load_mmlu_config():

    subsets = ["abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
                "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
                "college_medicine", "college_physics", "computer_security", "conceptual_physics",
                "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
                "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
                "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
                "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
                "high_school_physics", "high_school_psychology", "high_school_statistics",
                "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
                "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management",
                "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
                "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law",
                "professional_medicine", "professional_psychology", "public_relations", "security_studies",
                "sociology", "us_foreign_policy", "virology", "world_religions"]
    
    # STEM
    stem_subjects = [
        "abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry", "college_computer_science", "college_mathematics", "college_physics",
        "computer_security", "conceptual_physics", "electrical_engineering", "elementary_mathematics", "high_school_biology", "high_school_chemistry",  "high_school_computer_science", "high_school_mathematics", "high_school_physics", "high_school_statistics", "machine_learning",
    ]
    
    # Social Sciences
    social_sciences_subjects = [
        "econometrics", "high_school_geography","high_school_government_and_politics", "high_school_macroeconomics", "high_school_microeconomics", "human_sexuality",  "high_school_psychology", "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy",
    ]

    # Humanities
    humanities_subjects = [
        "formal_logic", "high_school_european_history", "high_school_us_history", "high_school_world_history", "international_law", "jurisprudence", "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy", "prehistory", "professional_law", "world_religions"
    ]

    # Other
    other_subjects = [
        "business_ethics",  "clinical_knowledge", "college_medicine", "global_facts",  "human_aging", "management", "marketing", "medical_genetics", "miscellaneous", "nutrition", "professional_accounting", "professional_medicine", "virology"
    ]
    
    return subsets, stem_subjects, social_sciences_subjects, humanities_subjects, other_subjects
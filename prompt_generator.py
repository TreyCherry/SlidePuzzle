import os
import random

class PromptGenerator:
    def __init__(self, themes_dir):
        self.themes_dir = themes_dir
        self.themes = {}
        self.load_themes()

    def load_themes(self):
        for filename in os.listdir(self.themes_dir):
            if filename.endswith('.txt'):
                theme_name = filename[:-4]
                with open(os.path.join(self.themes_dir, filename), 'r') as file:
                    words = [line.strip() for line in file if line.strip()]
                    self.themes[theme_name] = {
                        'adjectives': [w for w in words if w.startswith('ADJ:')],
                        'nouns': [w for w in words if w.startswith('NOUN:')],
                        'verbs': [w for w in words if w.startswith('VERB:')]
                    }

    def generate_prompt(self, active_themes):
        all_adjectives = []
        all_nouns = []
        all_verbs = []

        for theme in active_themes:
            if theme in self.themes:
                all_adjectives.extend(self.themes[theme]['adjectives'])
                all_nouns.extend(self.themes[theme]['nouns'])
                all_verbs.extend(self.themes[theme]['verbs'])

        if not all_adjectives or not all_nouns or not all_verbs:
            return "A simple Loading Icon"

        adjective = random.choice(all_adjectives).replace('ADJ:', '').strip()
        noun = random.choice(all_nouns).replace('NOUN:', '').strip()
        verb = random.choice(all_verbs).replace('VERB:', '').strip()

        return f"{adjective} {noun} {verb}"

    def get_available_themes(self):
        return list(self.themes.keys())

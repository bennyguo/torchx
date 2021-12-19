from addict import Dict
import yaml


def prompt(question):
    inp = input(f"{question} (y/n)").lower().strip()
    if inp and inp == 'y':
        return True
    if inp and inp == 'n':
        return False
    return prompt(question)


def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return Config(config)


class Config(Dict):
    def set(self, key, value):
        keys = key.split('.')
        d = self
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value
    
    def get(self, key):
        keys = key.split('.')
        d = self
        try:
            for k in keys:
                d = d[k]
        except:
            return None
        return d
    
    def export(self, export_path):
        with open(export_path, 'w') as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)

    def __missing__(self, key):
        raise KeyError(key)



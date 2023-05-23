import json
from loguru import logger
import os

class BaseClass:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__dict__.update(kwargs)

    def generate_list_settings(self, list_):
        normal_list = []
        for item in list_:
            if isinstance(item, BaseClass):
                normal_list.append(item.generate_settings())
            elif isinstance(item, dict):
                normal_list.append(self.generate_kwarg_setting(item))
            elif isinstance(item, (tuple, list)):
                normal_list.append(self.generate_list_settings(item))
            else:
                normal_list.append(item)
        return normal_list

    def generate_kwarg_setting(self, kwargs):
        normal_kwargs = dict()
        for kwarg in kwargs:
            if isinstance(kwargs[kwarg], BaseClass):
                normal_kwargs[kwarg] = kwargs[kwarg].generate_settings()
            elif isinstance(kwargs[kwarg], (list, tuple)):
                normal_kwargs[kwarg] = self.generate_list_settings(kwargs[kwarg])
            elif isinstance(kwargs[kwarg], dict):
                normal_kwargs[kwarg] = self.generate_kwarg_setting(kwargs[kwarg])
            else:
                normal_kwargs[kwarg] = kwargs[kwarg]
        
        return normal_kwargs


    def generate_settings(self):
        settings = {
            "class": self.__class__.__name__, 
            **self.generate_kwarg_setting(self.kwargs), 
        }
        
        return settings
    
    def save(self, path):
        settings = self.generate_settings()

        if os.path.dirname(path) != "":
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(settings, f, indent=2)

    @classmethod
    def get_all_subclasses(cls):
        all_subclasses = []

        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(subclass.get_all_subclasses())

        return all_subclasses

    @staticmethod
    def find_class(cls_name):
        for possible_cls in BaseClass.get_all_subclasses():
            if possible_cls.__name__ == cls_name:
                return possible_cls
        return None

    @staticmethod
    def load_from_list_settings(list_):
        output_list = []
        for item in list_:
            if isinstance(item, dict):
                output_list.append(BaseClass.load_from_dict(item))
            elif isinstance(item, (tuple, list)):
                output_list.append(BaseClass.load_from_list_settings(item))
            else:
                output_list.append(item)

        return output_list
    
    @staticmethod
    def load_from_dict(dict_):
        other_class = BaseClass.find_class(dict_.get("class", None))
        if other_class is not None:
            return other_class.load_from_settings(dict_)
        
        output_dict = dict()
        for key in dict_:
            if isinstance(dict_[key], dict):
                output_dict[key] = BaseClass.load_from_dict(dict_[key])
            elif isinstance(dict_[key], (tuple, list)):
                output_dict[key] = BaseClass.load_from_list_settings(dict_[key])
            else:
                output_dict[key] = dict_[key]

        return output_dict

    @staticmethod
    def load_from_settings(settings):
        cls = BaseClass.find_class(settings["class"])

        if cls is None:
            logger.error(f"Could not find class {settings['class']} when loading class.")
            return None

        kwargs = dict()
        for kwarg in settings:
            if kwarg == "class":
                continue
            if isinstance(settings[kwarg], dict):
                kwargs[kwarg] = BaseClass.load_from_dict(settings[kwarg])
            elif isinstance(settings[kwarg], (tuple, list)):
                kwargs[kwarg] = BaseClass.load_from_list_settings(settings[kwarg])
            else:
                kwargs[kwarg] = settings[kwarg]

        return cls(**kwargs)

    @staticmethod
    def load(path):
        with open(path, "r") as f:
            settings = json.load(f)
        cls = BaseClass.find_class(settings["class"])
        return cls.load_from_settings(settings)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.kwargs})"
    
    def __eq__(self, o: object) -> bool:
        if not isinstance(o, BaseClass):
            return False
        
        other_settings = o.generate_settings()
        settings = self.generate_settings()

        return other_settings == settings
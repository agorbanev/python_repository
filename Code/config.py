"""The shared configuration file.
"""

def addId(cls):

    class AddId(cls):

        def __init__(self, id, *args, **kargs):
            super(AddId, self).__init__(*args, **kargs)
            self.__id = id

        def getId(self):
            return self.__id

    return AddId


def decor(cls):
    return cls

class ConfigurationBase:

    def __init_subclass__(cls, swallow, **kwargs):
        cls.swallow = swallow

        super().__init_subclass__(**kwargs)

@decor
class Configuration(ConfigurationBase, swallow='amerika'):

    s = 1

    def __init__(self, p=2, q='d'):

        self.t = 0

        print('test')

class Cats:
    class _cat_iter:
        def __init__(self, cats):
            self.cats = cats
            self.cur = 0

        def myfunc(self):
            x = 3

            class MyClass(object):
                y = x

            return MyClass

        def next(self):
            i = self.cur
            if i >= len(self.cats):
                raise StopIteration
            self.cur += 1
            return self.cats[i]

    class Inner:
        """First Inner Class"""

        class _Inner:
            """Second Inner Class"""

            def inner_display(self, msg):
                print("This is _Inner class")
                print(msg)

        def inner_display(self, msg):
            print("This is Inner class")
            print(msg)

    def __init__(self):
        self.cats = []

    def add(self, name):
        self.cats.append(name)
        return self

    def __iter__(self):
        return Cats._cat_iter(self.cats)

class Meta():

    def __init__(self):
	
	    print('test')
		
class Settings:

    def __init__(self):
	
	    print('test')

CONFIG = dict(

    default_save_path = 'e://jsnt//',
    max_read_memory_limit_MB = 8

)

a = Configuration()
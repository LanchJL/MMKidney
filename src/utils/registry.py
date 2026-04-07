class Registry(dict):
    def register(self, name):
        def deco(obj):
            self[name] = obj
            return obj

        return deco

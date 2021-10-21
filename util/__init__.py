import os
files = [f for f in os.listdir(os.currdir) 
         if os.path.isfile(f) and f.endswith(*.py) and not f == '__init__.py']
__all__ = files

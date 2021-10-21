import os
p = os.path.dirname((os.path.abspath(__file__)))
files = [f for f in os.listdir(p) 
         if os.path.isfile(f) and f.endswith('.py') and not f == '__init__.py']
__all__ = files

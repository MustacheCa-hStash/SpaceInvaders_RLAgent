from ale_py import ALEInterface

ale = ALEInterface()

for name in dir(ale):
    if not name.startswith("_"):
        print(name)
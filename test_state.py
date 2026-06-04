from  state_manager import *

sm = StateManager()

print(
    sm.update_state(
        "Ganesh",
        True
    )
)

print(
    sm.update_state(
        "Ganesh",
        True
    )
)

print(
    sm.update_state(
        "Ganesh",
        False
    )
)
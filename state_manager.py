class StateManager:

    def __init__(self):
        self.person_states = {}

    def update_state(self, person_name, inside_zone):

        previous = self.person_states.get(
            person_name,
            "OUTSIDE"
        )

        event = None

        if previous == "OUTSIDE" and inside_zone:

            current = "INSIDE"
            event = "ENTERED"

        elif previous == "INSIDE" and not inside_zone:

            current = "OUTSIDE"
            event = "EXITED"

        else:

            current = previous

        self.person_states[person_name] = current

        return current, event
import random
import pickle

def _finalizer_sort(e):
    return e[1]

class Chain:
    def __init__(self, weight_adjustment):
        self._finalized_weights: dict[str:list[tuple[str:int]]] = {}
        self._weights: dict[str:{str:int}] = {}
        self.finalized = False
        self._weight_adjustment = weight_adjustment

    def set_weight_adjustment(self, weight_adjustment):
        self._weight_adjustment = weight_adjustment

    def train(self, origin: str, target: str) -> None:
        origin_weights = self._weights.get(origin)
        if not origin_weights:
            origin_weights = {}
            self._weights.update({
                origin: origin_weights
            })
        target_weight = origin_weights.get(target)
        if not target_weight:
            target_weight = 0
        origin_weights.update({
            target: (target_weight + self._weight_adjustment)
        })

    def initialize_for_use(self) -> None:
        for origin, weights in self._weights.items():
            weights_list = []
            for value, weight in weights.items():
                weights_list.append((value, weight))
            weights_list.sort(key=_finalizer_sort, reverse=True)
            self._finalized_weights.update({
                origin:weights_list
            })
        self.finalized = True

    def _steps(self, origin: str, depth: int=10) -> tuple[list[tuple[str, int]], int]:
        weights_list = self._finalized_weights.get(origin)
        if weights_list:
            options = weights_list[-depth:]
            limit = sum(n for _, n in options)
            return options, limit
        return [], 0


    def step(self, origin: str, depth: int=10) -> str:
        options, limit = self._steps(origin, depth)
        current_limit = 0
        seed = random.randint(0, limit-1)
        for value, value_limit in options:
            current_limit += value_limit
            if seed < current_limit:
                return value
        raise Exception("Unknown error occured")

    def save(self, path):
        if not self.finalized:
            raise Exception("Finalize the Chain first!")
        with open(path, "wb+") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

class MultiChain:
    def __init__(self, depth: int, depth_weight_adjustment: float=2.35):
        if type(depth) != int or depth < 0:
            raise ValueError("Argument \"depth\" must be a positive integer.")
        self._depth = depth
        self.finalized = False
        self._chains: list[Chain] = [Chain(pow(depth_weight_adjustment, index + 1)) for index in range(0, depth)]


    def set_weight_adjustment(self, weight_adjustment):
        for index, chain in enumerate(self._chains):
            chain.set_weight_adjustment(pow(weight_adjustment, index + 1))

    def train(self, origin: str, target: str):
        if type(origin) == str:
            parts = origin.split(" ")
        else:
            parts = origin
        for index in range(0, self._depth):
            if len(parts) > index:
                self._chains[index].train(
                    " ".join(parts[-index-1:]),
                    target
                )

    def initialize_for_use(self):
        for chain in self._chains:
            chain.initialize_for_use()
        self.finalized = True

    def _steps(self, origin: str, depth: int=10):
        if type(origin) == str:
            parts = origin.split(" ")
        else:
            parts = origin
        options = []
        limit = 0
        for index in range(0, self._depth):
            if len(parts) > index:
                if type(origin) == str:
                    new_options, new_limit = self._chains[index]._steps(" ".join(parts[-index-1:]), depth=depth)
                else:
                    new_options, new_limit = self._chains[index]._steps(parts[-index-1:], depth=depth)
                options += new_options
                limit += new_limit
        if limit == 0:
            raise ValueError(f"Value \"{origin}\" did not have any weights in the chains.")
        return options, limit

    def step(self, origin: str, depth: int=10):
        options, limit = self._steps(origin, depth)
        current_limit = 0
        seed = random.randint(0, limit-1)
        for value, value_limit in options:
            current_limit += value_limit
            if seed < current_limit:
                return value
        raise Exception("Unknown error occured")

    def save(self, path):
        if not self.finalized:
            raise Exception("Finalize the Chain first!")
        with open(path, "wb+") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_chain(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)

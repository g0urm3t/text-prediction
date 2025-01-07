import random
import pickle

def _finalizer_sort(e):
    return e[1]

class Chain:
    def __init__(self, weights: str|dict[str:list[tuple[str:int]]] = None):
        if isinstance(weights, str):
            self._finalized_weights: dict[str:list[tuple[str:int]]] = self._load(weights)
        elif isinstance(weights, str):
            self._finalized_weights: dict[str:list[tuple[str:int]]] = weights
        else:
            self._finalized_weights: dict[str:list[tuple[str:int]]] = {}
        self._weights: dict[str:{str:int}] = {}
        self.finalized = False

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
            target: (target_weight + 1)
        })

    def initialize_for_use(self, weight_adjustment: float=1.0) -> None:
        for origin, weights in self._weights.items():
            weights_list = []
            for value, weight in weights.items():
                weights_list.append((value, int(weight * weight_adjustment)))
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
    def __init__(self, depth: int, depth_weight_adjustment: float=3.15, load_path=None):
        if type(depth) != int or depth < 0:
            raise ValueError("Argument \"depth\" must be a positive integer.")
        self._depth = depth
        self._depth_weight_adjustment=depth_weight_adjustment
        self.finalized = False
        if load_path:
            self._load(load_path)
        else:
            self._chains: list[Chain] = [Chain() for _ in range(0, depth)]

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
        for index, chain in enumerate(self._chains):
            chain.initialize_for_use(weight_adjustment=(pow(index + 1, self._depth_weight_adjustment)))
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

def train_multi_chain_from_file(path, depth):
    chain = MultiChain(depth=depth)
    with open(path, "r", encoding="utf8") as source_file:
        data = source_file.read().lower().split(" ")
        for index in range(depth, len(data)):
            chain.train(
                data[max(0, index - depth):index],
                data[index]
            )
    chain.initialize_for_use()
    return chain

def load_chain(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)

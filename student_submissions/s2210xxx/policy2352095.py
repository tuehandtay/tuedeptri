from policy import Policy

class Policy2352095(Policy):
    def __init__(self):
        pass
    def get_action(self, observation, info):
        list_prods = sorted(observation["products"], key=lambda prod: prod["size"][0] * prod["size"][1], reverse=True)
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                best_stock_idx = -1
                best_position = None
                min_wasted_space = float("inf")
                stocks_with_sizes = [(i, stock, *self._get_stock_size_(stock)) for i, stock in enumerate(observation["stocks"])]
                stocks_with_sizes = sorted(stocks_with_sizes, key=lambda s: s[2] * s[3], reverse=True)
                for stock_idx, stock, stock_w, stock_h in stocks_with_sizes:
                    prod_w, prod_h = prod_size
                    if stock_w < prod_w or stock_h < prod_h:
                        continue
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                wasted_space = (stock_w - (x + prod_w)) * stock_h + stock_h - (y + prod_h)
                                if wasted_space < min_wasted_space:
                                    min_wasted_space = wasted_space
                                    best_position = (x, y)
                                    best_stock_idx = stock_idx
                if best_position is not None:
                    return {
                        "stock_idx": best_stock_idx,
                        "size": prod_size,
                        "position": best_position,
                    }
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

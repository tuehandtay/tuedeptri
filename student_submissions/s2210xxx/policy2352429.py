from policy import Policy
import numpy as np


class Policy2352429(Policy):    
    def __init__(self):
        """
        Khởi tạo chính sách với các tham số cần thiết.
        """
        super().__init__()

    def get_action(self, observation, info):
        """
        Chọn hành động dựa trên trạng thái quan sát và thông tin bổ sung.

        Args:
            observation (dict): Trạng thái quan sát hiện tại (stocks, products).
            info (dict): Thông tin bổ sung từ môi trường.

        Returns:
            dict | None: Hành động hợp lệ hoặc None nếu không tìm thấy hành động phù hợp.
        """
        stocks = observation.get("stocks", [])
        products = observation.get("products", [])

        if not stocks or not products:
            print("Dữ liệu stocks hoặc products trống!")
            return None

        sorted_products = sorted(products, key=lambda p: p["size"][0] * p["size"][1], reverse=True)

        for product in sorted_products:
            size = product["size"]
            quantity = product["quantity"]

            if quantity > 0:
                for stock_idx, stock in enumerate(stocks):
                    stock_array = np.array(stock)

                    best_fit = None
                    for x in range(stock_array.shape[0] - size[0] + 1):
                        for y in range(stock_array.shape[1] - size[1] + 1):
                            sub_stock = stock_array[x:x + size[0], y:y + size[1]]

                            if np.all(sub_stock == -1):
                                waste = (
                                    stock_array.shape[0] * stock_array.shape[1] -
                                    size[0] * size[1]
                                )
                                if best_fit is None or waste < best_fit["waste"]:
                                    best_fit = {"x": x, "y": y, "waste": waste}

                    if best_fit:
                        return {
                            "stock_idx": stock_idx,
                            "size": size,
                            "position": (best_fit["x"], best_fit["y"]),
                        }

        return None

    def BinPacking2DExample(self):
        """
        Ví dụ minh họa bài toán cắt 2D.
        
        Returns:
            list: Danh sách nhu cầu (mỗi nhu cầu là [dài, rộng]).
            list: Danh sách kho hàng (mỗi kho là [dài, rộng]).
        """
        demands = [[4, 2], [3, 5], [2, 2], [6, 3], [1, 1]]  
        stocks = [[10, 10], [5, 8], [7, 6]]  
        return demands, stocks
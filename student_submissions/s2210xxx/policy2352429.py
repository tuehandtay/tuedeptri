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
        Lựa chọn hành động dựa trên trạng thái quan sát (observation) và thông tin bổ sung (info).

        Args:
            observation (dict): Trạng thái quan sát hiện tại (stocks, products).
            info (dict): Thông tin bổ sung từ môi trường.

        Returns:
            dict | None: Hành động hợp lệ hoặc None nếu không tìm thấy hành động phù hợp.
        """
        stocks = observation.get("stocks", [])
        products = observation.get("products", [])

        # Kiểm tra xem stocks và products có dữ liệu không
        if not stocks or not products:
            print("Dữ liệu stocks hoặc products trống!")
            return None

        # Duyệt qua từng sản phẩm cần cắt
        for product in products:
            size = product["size"]  # Kích thước sản phẩm (dài, rộng)
            quantity = product["quantity"]

            # Kiểm tra nếu vẫn còn số lượng cần cắt
            if quantity > 0:
                # Duyệt qua từng kho để tìm kho phù hợp
                for stock_idx, stock in enumerate(stocks):
                    stock_array = np.array(stock)

                    # Tìm vị trí cắt hợp lệ trong kho (chỉ số đầu tiên thỏa mãn)
                    for x in range(stock_array.shape[0] - size[0] + 1):
                        for y in range(stock_array.shape[1] - size[1] + 1):
                            sub_stock = stock_array[x:x + size[0], y:y + size[1]]

                            # Kiểm tra vùng này có thể cắt được không (tất cả giá trị là -1)
                            if np.all(sub_stock == -1):
                                return {
                                    "stock_idx": stock_idx,
                                    "size": size,
                                    "position": (x, y),
                                }

        # Không tìm thấy hành động hợp lệ
        return None

    def BinPacking2DExample(self):
        """
        Ví dụ minh họa bài toán cắt 2D.
        
        Returns:
            list: Danh sách nhu cầu (mỗi nhu cầu là [dài, rộng]).
            list: Danh sách kho hàng (mỗi kho là [dài, rộng]).
        """
        demands = [[4, 2], [3, 5], [2, 2], [6, 3], [1, 1]]  # Nhu cầu (dài, rộng)
        stocks = [[10, 10], [5, 8], [7, 6]]  # Kho hàng (dài, rộng)
        return demands, stocks
    
    
    
    # area=0
    # demand=[]
    # def __init__(self):
    #     # Student code here
    #     stock=self._get_stock_size_();
    #     width=stock[0:1];
    #     length=stock[1:2];
    #     self.area=width*length
        

    # def get_action(self, observation, info):
    #     # Student code here
    #     pass

    # # Student code here
    # # You can add more functions if needed
    # def BinPackingExample():
    #     B = 9
    #     w = [2,3,4,5,6,7,8]
    #     q = [4,2,6,6,2,2,2]
    #     s=[]
    #     for j in range(len(w)):
    #         for i in range(q[j]):
    #             s.append(w[j])
    #     return s,B

    # def FFD(s, B):
    #     remain = [B]
    #     sol = [[]]
    #     for item in sorted(s, reverse=True):
    #         for j,free in enumerate(remain):
    #             if free >= item:
    #                 remain[j] -= item
    #                 sol[j].append(item)
    #                 break
    #         else:
    #             sol.append([item])
    #             remain.append(B-item)
    #     return sol
    
    # def bpp(s,B):
    #     n = len(s)
    #     U = len(FFD(s,B))
    #     model = Model("bpp")
    #     x,y = {},{}
    #     for i in range(n):
    #         for j in range(U):
    #             x[i,j] = model.addVar(vtype="B", name="x(%s,%s)"%(i,j))
    #     for j in range(U):
    #         y[j] = model.addVar(vtype="B", name="y(%s)"%j)
    #     for i in range(n):
    #         model.addCons(quicksum(x[i,j] for j in range(U)) == 1, "Assign(%s)"%i)
    #     for j in range(U):
    #         model.addCons(quicksum(s[i]*x[i,j] for i in range(n)) <= B*y[j], "Capac(%s)"%j)
    #     for j in range(U):
    #         for i in range(n):
    #             model.addCons(x[i,j] <= y[j], "Strong(%s,%s)"%(i,j))
    #     model.setObjective(quicksum(y[j] for j in range(U)), "minimize")
    #     model.data = x,y
    #     return model
    
    # def solveBinPacking(s,B):
    #     n = len(s)
    #     U = len(FFD(s,B))
    #     model = bpp(s,B)
    #     x,y = model.data
    #     model.optimize()
    #     bins = [[] for i in range(U)]
    #     for (i,j) in x:
    #         if model.getVal(x[i,j]) > .5:
    #             bins[j].append(s[i])
    #     for i in range(bins.count([])):
    #         bins.remove([])
    #     for b in bins:
    #         b.sort()
    #     bins.sort()
    #     return bins

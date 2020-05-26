import torch
import torch.nn.functional as F


# 事後分布(ロジスティクス分布)
class logistic_distribution(torch.distributions.Distribution):
    def __init__(self):
        super(logistic_distribution, self).__init__()

    def log_prob(self, x):
        """
        ロジスティック分布の対数尤度を計算する

        x : 入力テンソル

        出力 : 対数尤度
        """
        return -(F.softplus(x) + F.softplus(-x))
    
    def sample(self, size, device):
        """
        一様分布から、ロジスティクス分布に従って乱数を得る。

        size : 必要なテンソルのサイズ
        device : (str)cpuかgpuか

        出力 : 得られた乱数で埋まった、テンソル
        """
        if device == "cpu":
            z = torch.distributions.Uniform(0., 1.).sample(size)
            return torch.log(z) - torch.log(1. - z)
        else:
            z = torch.distributions.Uniform(0., 1.).sample(size).cuda()
            return torch.log(z) - torch.log(1. - z)

"""
prepare_data

やること(学習)
- ３次元のテンソルを１次元に変換
- ノイズを加える
- 標準化する

やること(生成)
- 1次元のテンソルを3次元に変換
"""
def prepare_data(x, reverse=False):
    """
    x : ミニバッチ
    reverse : 生成か学習か
    """
    if not reverse:# 学習
        # 一次元化
        [B, C, H, W] = list(x.size())
        x = x.reshape((B, C*H*W))

        # 一様のノイズを加える(全ての要素に同じ値を足すのか、要素ごとに違う値を足すのか)
        # 著者のコードを読むと、多分違う値
        # ノイズを加えたのち、256で割る　--> 0~1のデータに圧縮される
        noise = torch.distributions.Uniform(0., 1.).sample(x.size())
        x = (x*255 + noise)/256
        # return x
    else:
        max = torch.max(x, dim=1)
        min = torch.min(x, dim=1)
        # print(max[0])
        # print(min[0])
        [B, W] = list(x.size())

        print(x.size())
        print(min[0].size())

        min_t = min[0].reshape([B,1])
        max_t = max[0].reshape([B,1])

        x = (x-min_t)/(max_t-min_t)
        x = x.reshape((B,1,28,28))
    return x

if __name__=="__main__":
    a = torch.rand((3,3,3,3))*255
    print(a)
    c = prepare_data(a)
    print(c)
    b = prepare_data(c, reverse=True)
    print(b)
    
    # x = (1,100)
    # 何かしら正しい分布になっているか確認する手段を考える
    # sample_class = logistic_distribution()
    # print(sample_class.sample((5,5),"cpu"))

